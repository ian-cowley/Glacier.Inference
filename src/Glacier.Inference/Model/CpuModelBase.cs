namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Abstract base class for CPU SIMD-accelerated model execution runtimes.
/// Manages preallocated unmanaged scratch buffers, embedding lookup, SwiGLU / MoE FFN dispatch,
/// LM head output projection, and stage coordination with zero heap allocations.
/// </summary>
public abstract unsafe class CpuModelBase : ICpuModel
{
    protected readonly ModelWeights _weights;
    protected readonly int _dim;
    protected readonly int _ffnDim;
    protected readonly int _headDim;
    protected readonly int _vHeadDim;
    protected readonly int _nHeads;
    protected readonly int _nHeadsKv;
    protected readonly int _maxSeqLen;

    public const int MaxBatchSize = 64;

    // Preallocated unmanaged scratch buffers (single token)
    protected float* _x;
    protected float* _normX;
    protected float* _normXSums;
    protected float* _gate;
    protected float* _up;
    protected float* _ffnAct;
    protected float* _ffnActSums;
    protected float* _ffnOut;
    protected float* _expertDownOut;
    protected float* _gateUpFused;

    // Preallocated unmanaged scratch buffers (batched chunk prefill <= MaxBatchSize)
    protected float* _xBatch;
    protected float* _normXBatch;
    protected float* _normXSumBatch;
    protected float* _gateBatch;
    protected float* _upBatch;
    protected float* _ffnActBatch;
    protected float* _ffnActSumBatch;
    protected float* _ffnOutBatch;
    protected float* _gateUpBatch;

    protected bool _disposed;

    public ModelWeights Weights => _weights;
    public int MaxSeqLen => _maxSeqLen;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }
    public LoraAdapterWeights? LoraWeights { get; set; }

    protected CpuModelBase(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
    {
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _vHeadDim = weights.ValueDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _maxSeqLen = maxSeqLen;

        StartLayer = startLayer;
        LayerCount = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        IsLastStage = isLastStage;

        int maxFfn = _ffnDim;
        if (_weights.ExpertFeedForwardLength > maxFfn) maxFfn = _weights.ExpertFeedForwardLength;

        // Base unmanaged scratch buffers for single token
        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _gate = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _up = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnAct = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnActSums = (float*)NativeMemory.AllocZeroed((nuint)(((maxFfn + 31) / 32) * sizeof(float)));
        _ffnOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _expertDownOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        int gateUpDim = 2 * maxFfn;
        _gateUpFused = (float*)NativeMemory.AllocZeroed((nuint)(gateUpDim * sizeof(float)));

        // Base unmanaged scratch buffers for batched chunks
        _xBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * (_dim / 32) * sizeof(float)));
        _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _upBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * ((maxFfn + 31) / 32) * sizeof(float)));
        _ffnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _gateUpBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * gateUpDim * sizeof(float)));
    }

    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        ForwardStage(token, pos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null)
    {
        // 1. Embedding lookup or intermediate activation copy
        if (StartLayer == 0)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _x, _dim);
        }
        else
        {
            fixed (float* pInput = inputX)
            {
                Buffer.MemoryCopy(pInput, _x, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 2. Transformer / SSM layers
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var layer = _weights.Layers[modelLayer];

            // Attention / SSM pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // Subclass attention / SSM dispatch
            ForwardAttention(l, modelLayer, pos, _x, kvCache);

            // FFN pre-norm
            QuantKernels.RMSNorm(_x, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // FFN forward pass
            ForwardFfn(l, modelLayer, selectedIndices, selectedWeights);
        }

        // 3. Final normalization and output logits
        if (IsLastStage)
        {
            if (computeLogits && logits.Length > 0)
            {
                QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                fixed (float* pLogits = logits)
                {
                    QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, pLogits, _dim, _weights.VocabSize, _normXSums);
                }
            }
            else if (outputX.Length >= _dim)
            {
                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
            }
        }
        else if (outputX.Length >= _dim)
        {
            fixed (float* pOut = outputX)
            {
                Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected void ForwardFfn(int l, int modelLayer, int* selectedIndices, float* selectedWeights)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFeedForwardLength;
            int numExperts = _weights.ExpertCount;
            int topK = _weights.ExpertUsedCount;

            QuantKernels.RouterTopK(_normX, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);

            new Span<float>(_ffnOut, _dim).Clear();

            long gateSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
            long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
            long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

            for (int k = 0; k < topK; k++)
            {
                int expertIdx = selectedIndices[k];
                float weight = selectedWeights[k];

                byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
                byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;
                byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;

                QuantKernels.MatVecMul(layer.FfnGateExpsType, expGateWeight, _normX, _gate, _dim, expertFfnDim, _normXSums);
                if (layer.FfnGateExpsBias != null) AddVector(_gate, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                QuantKernels.MatVecMul(layer.FfnUpExpsType, expUpWeight, _normX, _up, _dim, expertFfnDim, _normXSums);
                if (layer.FfnUpExpsBias != null) AddVector(_up, layer.FfnUpExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
                QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);

                QuantKernels.MatVecMul(layer.FfnDownExpsType, expDownWeight, _ffnAct, _expertDownOut, expertFfnDim, _dim, _ffnActSums);
                if (layer.FfnDownExpsBias != null) AddVector(_expertDownOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);

                for (int d = 0; d < _dim; d++)
                {
                    _ffnOut[d] += weight * _expertDownOut[d];
                }
            }

            if (layer.FfnGateShexpWeight != null)
            {
                int shexpFfnDim = expertFfnDim * 2;
                if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                {
                    shexpFfnDim = (int)tShexp.Dimensions[1];
                }

                QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, _normX, _gate, _dim, shexpFfnDim, _normXSums);
                QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, _normX, _up, _dim, shexpFfnDim, _normXSums);
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                // Modulate by shared expert gate if present (Qwen 3.5 MoE)
                if (layer.FfnGateInpShexpWeight != null)
                {
                    float gateVal = QuantKernels.VecDotF32(_normX, layer.FfnGateInpShexpWeight, _dim);
                    float shexpGate = 1.0f / (1.0f + MathF.Exp(-gateVal));
                    for (int d = 0; d < _dim; d++)
                    {
                        _expertDownOut[d] *= shexpGate;
                    }
                }

                AddVector(_ffnOut, _expertDownOut, _dim);
            }

            AddVector(_x, _ffnOut, _dim);
        }
        else
        {
            if (layer.HasFusedGateUp)
            {
                int gateUpDim = 2 * _ffnDim;
                QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _gateUpFused, _dim, gateUpDim, _normXSums);
                if (layer.FfnUpBias != null) AddVector(_gateUpFused, layer.FfnUpBias, gateUpDim);

                QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + _ffnDim, _ffnAct, _ffnDim);
            }
            else
            {
                QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim, _normXSums);
                QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim, _normXSums);
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Gate, _normX, _gate, _dim, _ffnDim);
                    LoraWeights.Apply(modelLayer, LoraProjection.Up, _normX, _up, _dim, _ffnDim);
                }
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
            }
            QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, _ffnDim);
            QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, _ffnOut, _ffnDim, _dim, _ffnActSums);
            if (LoraWeights != null)
            {
                LoraWeights.Apply(modelLayer, LoraProjection.Down, _ffnAct, _ffnOut, _ffnDim, _dim);
            }

            AddVector(_x, _ffnOut, _dim);
        }
    }

    /// <summary>
    /// Evaluates a sequence of prompt tokens in batched chunks.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes batched prefill for the layers assigned to this stage.
    /// </summary>
    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null)
    {
        int totalTokens = !tokens.IsEmpty ? tokens.Length : (inputXBatch.Length / _dim);
        int offset = 0;
        while (offset < totalTokens)
        {
            int batchSize = Math.Min(totalTokens - offset, MaxBatchSize);
            bool isLastChunk = (offset + batchSize == totalTokens);
            var inSlice = !inputXBatch.IsEmpty ? inputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var outSlice = !outputXBatch.IsEmpty ? outputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var tokSlice = !tokens.IsEmpty ? tokens.Slice(offset, batchSize) : default;

            ForwardBatchChunk(
                tokSlice,
                startPos + offset,
                inSlice,
                outSlice,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                kvCache,
                isLastChunk && computeLogits);
            offset += batchSize;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardBatchChunk(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        KVCache? kvCache,
        bool computeLogits)
    {
        int batchSize = !chunkTokens.IsEmpty ? chunkTokens.Length : (inputXBatch.Length / _dim);
        int normXChunks = _dim / 32;

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 1. Extract embeddings or copy intermediate input activations
        if (StartLayer == 0)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], _xBatch + t * _dim, _dim);
            }
        }
        else
        {
            fixed (float* pIn = inputXBatch)
            {
                Buffer.MemoryCopy(pIn, _xBatch, (ulong)(batchSize * _dim * sizeof(float)), (ulong)(batchSize * _dim * sizeof(float)));
            }
        }

        // 2. Transformer / SSM layers
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var layer = _weights.Layers[modelLayer];

            // Attention pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.AttnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

            // Subclass batch attention / SSM dispatch
            ForwardBatchChunkAttention(l, modelLayer, chunkStartPos, batchSize, kvCache);

            // FFN pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.FfnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

            // Subclass / Common batch FFN
            ForwardBatchChunkFfn(l, modelLayer, batchSize, normXChunks, selectedIndices, selectedWeights);
        }

        // 3. Final normalization and output logits for the last token in chunk
        if (IsLastStage)
        {
            if (computeLogits && logits.Length > 0)
            {
                int lastIdx = batchSize - 1;
                float* xLast = _xBatch + lastIdx * _dim;
                QuantKernels.RMSNorm(xLast, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                fixed (float* pLogits = logits)
                {
                    QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, pLogits, _dim, _weights.VocabSize, _normXSums);
                }
            }
            else if (outputXBatch.Length >= batchSize * _dim)
            {
                fixed (float* pOut = outputXBatch)
                {
                    Buffer.MemoryCopy(_xBatch, pOut, (ulong)(batchSize * _dim * sizeof(float)), (ulong)(batchSize * _dim * sizeof(float)));
                }
            }
        }
        else if (outputXBatch.Length >= batchSize * _dim)
        {
            fixed (float* pOut = outputXBatch)
            {
                Buffer.MemoryCopy(_xBatch, pOut, (ulong)(batchSize * _dim * sizeof(float)), (ulong)(batchSize * _dim * sizeof(float)));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected void ForwardBatchChunkFfn(int l, int modelLayer, int batchSize, int normXChunks, int* selectedIndices, float* selectedWeights)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFeedForwardLength;
            int numExperts = _weights.ExpertCount;
            int topK = _weights.ExpertUsedCount;

            long gateSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
            long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
            long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

            for (int t = 0; t < batchSize; t++)
            {
                float* normXt = _normXBatch + t * _dim;
                float* normXSumst = _normXSumBatch + t * normXChunks;
                float* ffnOutt = _ffnOutBatch + t * _dim;

                QuantKernels.RouterTopK(normXt, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);

                for (int d = 0; d < _dim; d++) ffnOutt[d] = 0f;

                for (int k = 0; k < topK; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
                    byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;
                    byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;

                    QuantKernels.MatVecMul(layer.FfnGateExpsType, expGateWeight, normXt, _gate, _dim, expertFfnDim, normXSumst);
                    if (layer.FfnGateExpsBias != null) AddVector(_gate, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                    QuantKernels.MatVecMul(layer.FfnUpExpsType, expUpWeight, normXt, _up, _dim, expertFfnDim, normXSumst);
                    if (layer.FfnUpExpsBias != null) AddVector(_up, layer.FfnUpExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);

                    QuantKernels.MatVecMul(layer.FfnDownExpsType, expDownWeight, _ffnAct, _expertDownOut, expertFfnDim, _dim, _ffnActSums);
                    if (layer.FfnDownExpsBias != null) AddVector(_expertDownOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);

                    for (int d = 0; d < _dim; d++)
                    {
                        ffnOutt[d] += weight * _expertDownOut[d];
                    }
                }

                if (layer.FfnGateShexpWeight != null)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                    {
                        shexpFfnDim = (int)tShexp.Dimensions[1];
                    }

                    QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, normXt, _gate, _dim, shexpFfnDim, normXSumst);
                    QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, normXt, _up, _dim, shexpFfnDim, normXSumst);
                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                    QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                    // Modulate by shared expert gate if present (Qwen 3.5 MoE)
                    if (layer.FfnGateInpShexpWeight != null)
                    {
                        float gateVal = QuantKernels.VecDotF32(normXt, layer.FfnGateInpShexpWeight, _dim);
                        float shexpGate = 1.0f / (1.0f + MathF.Exp(-gateVal));
                        for (int d = 0; d < _dim; d++)
                        {
                            _expertDownOut[d] *= shexpGate;
                        }
                    }

                    AddVector(ffnOutt, _expertDownOut, _dim);
                }

                AddVector(_xBatch + t * _dim, ffnOutt, _dim);
            }
        }
        else
        {
            int ffnChunks = _ffnDim / 32;
            if (layer.HasFusedGateUp)
            {
                int gateUpDim = 2 * _ffnDim;
                QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _gateUpBatch, _dim, gateUpDim, batchSize, _normXSumBatch);
                for (int t = 0; t < batchSize; t++)
                {
                    float* src = _gateUpBatch + t * gateUpDim;
                    if (layer.FfnUpBias != null) AddVector(src, layer.FfnUpBias, gateUpDim);
                    QuantKernels.SwiGLU(src, src + _ffnDim, _ffnActBatch + t * _ffnDim, _ffnDim);
                }
            }
            else
            {
                QuantKernels.MatMulBatch(layer.FfnGateType, layer.FfnGateWeight, _normXBatch, _gateBatch, _dim, _ffnDim, batchSize, _normXSumBatch);
                QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _upBatch, _dim, _ffnDim, batchSize, _normXSumBatch);

                for (int t = 0; t < batchSize; t++)
                {
                    float* normXt = _normXBatch + t * _dim;
                    float* g = _gateBatch + t * _ffnDim;
                    float* u = _upBatch + t * _ffnDim;
                    if (LoraWeights != null)
                    {
                        LoraWeights.Apply(modelLayer, LoraProjection.Gate, normXt, g, _dim, _ffnDim);
                        LoraWeights.Apply(modelLayer, LoraProjection.Up, normXt, u, _dim, _ffnDim);
                    }
                    QuantKernels.SwiGLU(g, u, _ffnActBatch + t * _ffnDim, _ffnDim);
                }
            }

            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ComputeBlockSums32(_ffnActBatch + t * _ffnDim, _ffnActSumBatch + t * ffnChunks, _ffnDim);
            }

            QuantKernels.MatMulBatch(layer.FfnDownType, layer.FfnDownWeight, _ffnActBatch, _ffnOutBatch, _ffnDim, _dim, batchSize, _ffnActSumBatch);

            for (int t = 0; t < batchSize; t++)
            {
                float* ffnOutT = _ffnOutBatch + t * _dim;
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Down, _ffnActBatch + t * _ffnDim, ffnOutT, _ffnDim, _dim);
                }
                AddVector(_xBatch + t * _dim, ffnOutT, _dim);
            }
        }
    }

    /// <summary>
    /// Prefills the KV cache with prompt tokens without computing final output logits.
    /// </summary>
    public void PrefillPrompt(ReadOnlySpan<int> tokens, KVCache kvCache)
    {
        ForwardBatch(tokens, 0, Span<float>.Empty, kvCache, computeLogits: false);
    }

    /// <summary>
    /// Subclasses implement their specific layer attention / SSM logic (MHA, GQA, MLA, or GDN).
    /// </summary>
    protected abstract void ForwardAttention(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache);

    /// <summary>
    /// Subclasses implement their specific batched layer attention / SSM logic.
    /// </summary>
    protected abstract void ForwardBatchChunkAttention(int stageLayer, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache);

    /// <summary>
    /// Extracts a normalized embedding vector for the given token sequence using the specified pooling strategy.
    /// Bypasses the LM Head projection for maximum throughput.
    /// </summary>
    public void ExtractEmbedding(
        ReadOnlySpan<int> tokens,
        Span<float> destination,
        KVCache kvCache,
        PoolingStrategy strategy = PoolingStrategy.LastToken)
    {
        if (destination.Length < _dim)
            throw new ArgumentException($"Destination span too small. Expected {_dim}, got {destination.Length}", nameof(destination));

        destination.Slice(0, _dim).Clear();
        if (tokens.IsEmpty) return;

        // Forward tokens through layers
        ForwardBatch(tokens, 0, Span<float>.Empty, kvCache, computeLogits: false);

        int totalTokens = tokens.Length;
        int lastBatchSize = totalTokens % MaxBatchSize;
        if (lastBatchSize == 0 && totalTokens > 0) lastBatchSize = MaxBatchSize;

        if (strategy == PoolingStrategy.LastToken)
        {
            float* xLast = _xBatch + (lastBatchSize - 1) * _dim;
            fixed (float* pDst = destination)
            {
                QuantKernels.RMSNorm(xLast, _weights.OutNormWeight, pDst, _dim, _weights.RmsNormEps);
            }
        }
        else // MeanPooling
        {
            int count = Math.Min(totalTokens, MaxBatchSize);
            Span<float> tempNorm = stackalloc float[Math.Min(_dim, 4096)];
            bool useHeap = _dim > 4096;
            float[]? heapArr = useHeap ? new float[_dim] : null;
            Span<float> normBuffer = useHeap ? heapArr.AsSpan() : tempNorm;

            fixed (float* pNorm = normBuffer)
            {
                for (int t = 0; t < count; t++)
                {
                    float* xt = _xBatch + t * _dim;
                    QuantKernels.RMSNorm(xt, _weights.OutNormWeight, pNorm, _dim, _weights.RmsNormEps);
                    for (int d = 0; d < _dim; d++)
                    {
                        destination[d] += normBuffer[d];
                    }
                }
            }

            float invCount = 1.0f / count;
            for (int d = 0; d < _dim; d++)
            {
                destination[d] *= invCount;
            }
        }

        // L2 Normalize to unit vector
        NormalizeL2Vector(destination.Slice(0, _dim));
    }

    private static void NormalizeL2Vector(Span<float> vec)
    {
        float sumSq = 0f;
        for (int i = 0; i < vec.Length; i++)
        {
            sumSq += vec[i] * vec[i];
        }

        if (sumSq > 0f)
        {
            float invNorm = 1.0f / MathF.Sqrt(sumSq);
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] *= invNorm;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    protected static void AddVector(float* a, float* b, int count)
    {
        int i = 0;
        if (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated)
        {
            int limit = count - 8;
            for (; i <= limit; i += 8)
            {
                var va = System.Runtime.Intrinsics.Vector256.Load(a + i);
                var vb = System.Runtime.Intrinsics.Vector256.Load(b + i);
                (va + vb).Store(a + i);
            }
        }
        for (; i < count; i++)
        {
            a[i] += b[i];
        }
    }

    protected static void FreeIfAllocated(ref float* ptr)
    {
        if (ptr != null)
        {
            NativeMemory.Free(ptr);
            ptr = null;
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            FreeIfAllocated(ref _x);
            FreeIfAllocated(ref _normX);
            FreeIfAllocated(ref _normXSums);
            FreeIfAllocated(ref _gate);
            FreeIfAllocated(ref _up);
            FreeIfAllocated(ref _ffnAct);
            FreeIfAllocated(ref _ffnActSums);
            FreeIfAllocated(ref _ffnOut);
            FreeIfAllocated(ref _expertDownOut);
            FreeIfAllocated(ref _gateUpFused);

            FreeIfAllocated(ref _xBatch);
            FreeIfAllocated(ref _normXBatch);
            FreeIfAllocated(ref _normXSumBatch);
            FreeIfAllocated(ref _gateBatch);
            FreeIfAllocated(ref _upBatch);
            FreeIfAllocated(ref _ffnActBatch);
            FreeIfAllocated(ref _ffnActSumBatch);
            FreeIfAllocated(ref _ffnOutBatch);
            FreeIfAllocated(ref _gateUpBatch);

            _disposed = true;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    ~CpuModelBase()
    {
        Dispose(false);
    }
}
