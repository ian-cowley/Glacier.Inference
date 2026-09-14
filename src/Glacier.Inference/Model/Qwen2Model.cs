namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// High-performance Qwen2 / Qwen2.5 Transformer inference runtime.
/// Executes hardware SIMD quantized GEMV decoding with zero heap allocations.
/// </summary>
public sealed unsafe class Qwen2Model : IDisposable
{
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _headDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _groupSize;
    private readonly float _attnScale;
    private readonly int _maxSeqLen;

    public const int MaxBatchSize = 64;

    // Preallocated unmanaged scratch buffers (single token)
    private float* _x;
    private float* _normX;
    private float* _normXSums;
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProj;
    private float* _gate;
    private float* _up;
    private float* _ffnAct;
    private float* _ffnActSums;
    private float* _ffnOut;
    private float* _expertDownOut;
    private float* _headScores;

    // Preallocated unmanaged scratch buffers (batched chunk prefill <= MaxBatchSize)
    private float* _xBatch;
    private float* _normXBatch;
    private float* _normXSumBatch;
    private float* _qBatch;
    private float* _kBatch;
    private float* _vBatch;
    private float* _attnOutBatch;
    private float* _attnOutSumBatch;
    private float* _attnProjBatch;
    private float* _gateBatch;
    private float* _upBatch;
    private float* _ffnActBatch;
    private float* _ffnActSumBatch;
    private float* _ffnOutBatch;

    private bool _disposed;

    public ModelWeights Weights => _weights;
    public int MaxSeqLen => _maxSeqLen;

    public Qwen2Model(ModelWeights weights, int maxSeqLen = 4096)
    {
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _maxSeqLen = maxSeqLen;

        int expFfn = weights.ExpertFeedForwardLength;
        int maxFfn = Math.Max(_ffnDim, Math.Max(expFfn, expFfn * 2));
        if (maxFfn == 0) maxFfn = _ffnDim;

        int qDim = _nHeads * _headDim;
        int maxAttnOut = Math.Max(_dim, qDim);
        int attnOutChunks = (maxAttnOut + 31) / 32;

        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _q = (float*)NativeMemory.AllocZeroed((nuint)(_nHeads * _headDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(maxAttnOut * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(attnOutChunks * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _gate = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _up = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnAct = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnActSums = (float*)NativeMemory.AllocZeroed((nuint)(((maxFfn + 31) / 32) * sizeof(float)));
        _ffnOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _expertDownOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        long scoreBufferSize = (long)_nHeads * _maxSeqLen * sizeof(float);
        _headScores = (float*)NativeMemory.AllocZeroed((nuint)scoreBufferSize);

        // Batch scratch buffers
        _xBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * (_dim / 32) * sizeof(float)));
        _qBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeads * _headDim * sizeof(float)));
        _kBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
        _vBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
        _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
        _attnOutSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * attnOutChunks * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _upBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * ((maxFfn + 31) / 32) * sizeof(float)));
        _ffnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
    }

    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// Writes logits of size VocabSize into destination buffer if computeLogits is true.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        // 1. Embedding lookup
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _x, _dim);

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 2. Transformer layers
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var layer = _weights.Layers[l];

            // Attention pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // Q, K, V projections (reusing _normXSums across all 3)
            int qDim = _nHeads * _headDim;
            int kvDim = _nHeadsKv * _headDim;
            QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
            QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

            // Add Q, K, V biases if present
            if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
            if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
            if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);

            // Optional QK-Norm (e.g. Qwen3)
            if (layer.AttnQNormWeight != null)
            {
                for (int h = 0; h < _nHeads; h++)
                {
                    float* qHead = _q + h * _headDim;
                    QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _weights.RmsNormEps);
                }
            }
            if (layer.AttnKNormWeight != null)
            {
                for (int h = 0; h < _nHeadsKv; h++)
                {
                    float* kHead = _k + h * _headDim;
                    QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _weights.RmsNormEps);
                }
            }

            // Rotary Position Embedding (RoPE)
            QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase);

            // Store in KV cache
            kvCache.Store(l, pos, _k, _v);

            // Multi-Head / Grouped Query Attention (GQA)
            ComputeAttention(l, pos, kvCache);

            // Attention output projection
            QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
            QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
            if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

            // Residual connection: x = x + attnProj
            AddVector(_x, _attnProj, _dim);

            // FFN pre-norm
            QuantKernels.RMSNorm(_x, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            if (layer.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;

                QuantKernels.RouterTopK(_normX, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights);

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

                // Shared Expert if present
                if (layer.FfnGateShexpWeight != null)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                    {
                        shexpFfnDim = (int)tShexp.Dimensions[1];
                    }

                    QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, _normX, _gate, _dim, shexpFfnDim, _normXSums);
                    QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, _normX, _up, _dim, shexpFfnDim, _normXSums);
                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                    QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                    AddVector(_ffnOut, _expertDownOut, _dim);
                }

                // Residual connection: x = x + ffnOut
                AddVector(_x, _ffnOut, _dim);
            }
            else
            {
                // SwiGLU FFN projections (reusing _normXSums for Gate and Up)
                QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim, _normXSums);
                QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim, _normXSums);
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
                QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, _ffnDim);
                QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, _ffnOut, _ffnDim, _dim, _ffnActSums);

                // Residual connection: x = x + ffnOut
                AddVector(_x, _ffnOut, _dim);
            }
        }

        if (computeLogits)
        {
            // Final RMSNorm
            QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            // LM Head output projection
            fixed (float* logitsPtr = logits)
            {
                QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, logitsPtr, _dim, _weights.VocabSize, _normXSums);
            }
        }
    }

    /// <summary>
    /// Evaluates a sequence of prompt tokens in batched chunks. Weights are streamed once per chunk
    /// across all tokens, reducing memory bus traffic by up to MaxBatchSize-fold.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        int offset = 0;
        while (offset < tokens.Length)
        {
            int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
            bool isLastChunk = (offset + batchSize == tokens.Length);
            ForwardBatchChunk(
                tokens.Slice(offset, batchSize),
                startPos + offset,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                kvCache,
                isLastChunk && computeLogits);
            offset += batchSize;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardBatchChunk(ReadOnlySpan<int> chunkTokens, int chunkStartPos, Span<float> logits, KVCache kvCache, bool computeLogits)
    {
        int batchSize = chunkTokens.Length;
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int normXChunks = _dim / 32;
        int ffnChunks = _ffnDim / 32;

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 1. Extract embeddings into batch buffer
        for (int t = 0; t < batchSize; t++)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], _xBatch + t * _dim, _dim);
        }

        // 2. Transformer layers
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var layer = _weights.Layers[l];

            // Attention pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.AttnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

            // Batched Q, K, V projections (weights streamed once!)
            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
            QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumBatch);
            QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumBatch);

            // Per-token bias, RoPE, KV Cache, and Attention
            for (int t = 0; t < batchSize; t++)
            {
                int pos = chunkStartPos + t;
                float* q = _qBatch + t * qDim;
                float* k = _kBatch + t * kvDim;
                float* v = _vBatch + t * kvDim;

                if (layer.QBias != null) AddVector(q, layer.QBias, qDim);
                if (layer.KBias != null) AddVector(k, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(v, layer.VBias, kvDim);

                // Optional QK-Norm (e.g. Qwen3)
                if (layer.AttnQNormWeight != null)
                {
                    for (int h = 0; h < _nHeads; h++)
                    {
                        float* qHead = q + h * _headDim;
                        QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _weights.RmsNormEps);
                    }
                }
                if (layer.AttnKNormWeight != null)
                {
                    for (int h = 0; h < _nHeadsKv; h++)
                    {
                        float* kHead = k + h * _headDim;
                        QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _weights.RmsNormEps);
                    }
                }

                QuantKernels.RoPE(q, k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase);
                kvCache.Store(l, pos, k, v);

                ComputeAttentionToken(l, pos, q, _attnOutBatch + t * qDim, kvCache);
            }

            // Attention output projection (weights streamed once!)
            int attnOutChunks = (qDim + 31) / 32;
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ComputeBlockSums32(_attnOutBatch + t * qDim, _attnOutSumBatch + t * attnOutChunks, qDim);
            }
            QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, qDim, _dim, batchSize, _attnOutSumBatch);
            if (layer.AttnOutBias != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    AddVector(_attnProjBatch + t * _dim, layer.AttnOutBias, _dim);
                }
            }

            // Residual connection
            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
            }

            // FFN pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.FfnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

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

                    QuantKernels.RouterTopK(normXt, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights);

                    new Span<float>(ffnOutt, _dim).Clear();

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

                    // Shared Expert if present
                    if (layer.FfnGateShexpWeight != null)
                    {
                        int shexpFfnDim = expertFfnDim * 2;
                        if (_weights.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        {
                            shexpFfnDim = (int)tShexp.Dimensions[1];
                        }

                        QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, normXt, _gate, _dim, shexpFfnDim, normXSumst);
                        QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, normXt, _up, _dim, shexpFfnDim, normXSumst);
                        QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                        QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                        QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                        AddVector(ffnOutt, _expertDownOut, _dim);
                    }

                    AddVector(_xBatch + t * _dim, ffnOutt, _dim);
                }
            }
            else
            {
                // Batched Gate and Up projections (weights streamed once!)
                QuantKernels.MatMulBatch(layer.FfnGateType, layer.FfnGateWeight, _normXBatch, _gateBatch, _dim, _ffnDim, batchSize, _normXSumBatch);
                QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _upBatch, _dim, _ffnDim, batchSize, _normXSumBatch);

                // SwiGLU activation
                for (int t = 0; t < batchSize; t++)
                {
                    float* act = _ffnActBatch + t * _ffnDim;
                    QuantKernels.SwiGLU(_gateBatch + t * _ffnDim, _upBatch + t * _ffnDim, act, _ffnDim);
                    QuantKernels.ComputeBlockSums32(act, _ffnActSumBatch + t * ffnChunks, _ffnDim);
                }

                // Batched FFN Down projection (weights streamed once!)
                QuantKernels.MatMulBatch(layer.FfnDownType, layer.FfnDownWeight, _ffnActBatch, _ffnOutBatch, _ffnDim, _dim, batchSize, _ffnActSumBatch);

                // Residual connection
                for (int t = 0; t < batchSize; t++)
                {
                    AddVector(_xBatch + t * _dim, _ffnOutBatch + t * _dim, _dim);
                }
            }
        }

        if (computeLogits)
        {
            // Compute logits only for the last token in the batch
            int lastTokenIdx = batchSize - 1;
            float* xLast = _xBatch + lastTokenIdx * _dim;
            QuantKernels.RMSNorm(xLast, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            fixed (float* logitsPtr = logits)
            {
                QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, logitsPtr, _dim, _weights.VocabSize, _normXSums);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttentionToken(int layer, int pos, float* qHeadBase, float* outHeadBase, KVCache kvCache)
    {
        Parallel.For(0, _nHeads, h =>
        {
            int hKv = h / _groupSize;
            float* qHead = qHeadBase + h * _headDim;
            float* scores = _headScores + (long)h * _maxSeqLen;

            // Score calculation Q * K^T
            for (int t = 0; t <= pos; t++)
            {
                float* kPast = kvCache.GetKeyPtr(layer, hKv, t);
                float dot = QuantKernels.VecDotF32(qHead, kPast, _headDim);
                scores[t] = dot * _attnScale;
            }

            // Softmax over 0..pos with optional attention sink logit
            var layerWeights = _weights.Layers[layer];
            float? sinkLogit = layerWeights.AttnSinksWeight != null ? (float?)layerWeights.AttnSinksWeight[h] : null;
            QuantKernels.Softmax(scores, pos + 1, sinkLogit);

            // Value aggregation
            float* outHead = outHeadBase + h * _headDim;
            for (int d = 0; d < _headDim; d++) outHead[d] = 0f;

            for (int t = 0; t <= pos; t++)
            {
                float* vPast = kvCache.GetValuePtr(layer, hKv, t);
                float w = scores[t];

                if (Vector256.IsHardwareAccelerated)
                {
                    var vw = Vector256.Create(w);
                    for (int d = 0; d < _headDim; d += 8)
                    {
                        var vo = Vector256.Load(outHead + d);
                        var vv = Vector256.Load(vPast + d);
                        vo += vw * vv;
                        vo.Store(outHead + d);
                    }
                }
                else
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        outHead[d] += w * vPast[d];
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttention(int layer, int pos, KVCache kvCache)
    {
        ComputeAttentionToken(layer, pos, _q, _attnOut, kvCache);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddVector(float* a, float* b, int count)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int limit = count - 8;
            for (; i <= limit; i += 8)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                (va + vb).Store(a + i);
            }
        }
        for (; i < count; i++)
        {
            a[i] += b[i];
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            FreeIfAllocated(ref _x);
            FreeIfAllocated(ref _normX);
            FreeIfAllocated(ref _normXSums);
            FreeIfAllocated(ref _q);
            FreeIfAllocated(ref _k);
            FreeIfAllocated(ref _v);
            FreeIfAllocated(ref _attnOut);
            FreeIfAllocated(ref _attnOutSums);
            FreeIfAllocated(ref _attnProj);
            FreeIfAllocated(ref _gate);
            FreeIfAllocated(ref _up);
            FreeIfAllocated(ref _ffnAct);
            FreeIfAllocated(ref _ffnActSums);
            FreeIfAllocated(ref _ffnOut);
            FreeIfAllocated(ref _expertDownOut);
            FreeIfAllocated(ref _headScores);

            FreeIfAllocated(ref _xBatch);
            FreeIfAllocated(ref _normXBatch);
            FreeIfAllocated(ref _normXSumBatch);
            FreeIfAllocated(ref _qBatch);
            FreeIfAllocated(ref _kBatch);
            FreeIfAllocated(ref _vBatch);
            FreeIfAllocated(ref _attnOutBatch);
            FreeIfAllocated(ref _attnOutSumBatch);
            FreeIfAllocated(ref _attnProjBatch);
            FreeIfAllocated(ref _gateBatch);
            FreeIfAllocated(ref _upBatch);
            FreeIfAllocated(ref _ffnActBatch);
            FreeIfAllocated(ref _ffnActSumBatch);
            FreeIfAllocated(ref _ffnOutBatch);

            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    private static void FreeIfAllocated(ref float* ptr)
    {
        if (ptr != null)
        {
            NativeMemory.Free(ptr);
            ptr = null;
        }
    }

    ~Qwen2Model()
    {
        Dispose();
    }
}

