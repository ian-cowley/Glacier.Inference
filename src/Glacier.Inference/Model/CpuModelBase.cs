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
/// Provides access to the trunk model's pre-LM-head hidden state and token embeddings for native MTP drafting.
/// </summary>
public interface IHiddenStateSource
{
    int HiddenDim { get; }
    int VocabSize { get; }
    int MultiStreamHiddenDim => HiddenDim;
    ReadOnlySpan<float> GetLastHiddenState();
    ReadOnlySpan<float> GetLastMultiStreamHiddenState() => GetLastHiddenState();
    void CopyTokenEmbedding(int tokenId, Span<float> destination);
    void EnablePositionHistory();
    bool TryGetHiddenStateAt(int position, Span<float> destination);
    bool TryGetMultiStreamHiddenStateAt(int position, Span<float> destination) => TryGetHiddenStateAt(position, destination);
}

/// <summary>
/// Abstract base class for CPU SIMD-accelerated model execution runtimes.
/// Manages preallocated unmanaged scratch buffers, embedding lookup, SwiGLU / MoE FFN dispatch,
/// LM head output projection, and stage coordination with zero heap allocations.
/// </summary>
public abstract unsafe class CpuModelBase : ICpuModel, IHiddenStateSource
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
    protected float* _hiddenHistory;

    // Coalesced MoE batch scratch & verify logits buffer (preallocated/lazy unmanaged memory)
    protected int* _moeBatchIndices;
    protected float* _moeBatchWeights;
    protected int* _moeSortedSlots;
    protected float* _moeSlotInputs;
    protected float* _moeSlotInputSums;
    protected float* _moeSlotOutputs;
    private float* _verifyLogitsBatch;
    private int _verifyLogitsBatchCapacity;
    private int _moeTailSkipMinRank;

    protected bool _disposed;

    public ModelWeights Weights => _weights;
    public int MaxSeqLen => _maxSeqLen;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }
    public LoraAdapterWeights? LoraWeights { get; set; }

    public int MoeTailSkipMinRank
    {
        get => _moeTailSkipMinRank;
        set => _moeTailSkipMinRank = Math.Max(0, value);
    }

    internal delegate void TieredCpuMoeLayerBatchHandler(
        int modelLayer,
        int batchSize,
        int topK,
        bool allowTailSkip,
        int* batchTopKIndices,
        float* batchTopKWeights,
        float* normXBatch,
        float* normXSumBatch,
        float* ffnOutBatch);

    internal TieredCpuMoeLayerBatchHandler? TieredMoeBatchHandler { get; set; }

    public virtual bool SupportsBatchVerification => true;
    public virtual bool SupportsPerStepVerificationSnapshots => false;

    public int HiddenDim => _dim;
    public int VocabSize => _weights.VocabSize;

    public virtual ReadOnlySpan<float> GetLastHiddenState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<float>(_x, _dim);
    }

    public virtual void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < _dim)
            throw new ArgumentException($"Destination span too small ({destination.Length} < {_dim}).", nameof(destination));
        if (_weights.EmbdWeight == null)
            throw new InvalidOperationException("Model has no token_embd.weight pointer.");
        fixed (float* pDst = destination)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, tokenId, pDst, _dim);
        }
    }

    public virtual int MultiStreamHiddenDim => _dim;

    public virtual ReadOnlySpan<float> GetLastMultiStreamHiddenState() => GetLastHiddenState();

    public virtual void EnablePositionHistory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hiddenHistory != null) return;
        long totalFloats = Math.Max(1L, (long)_maxSeqLen * _dim);
        _hiddenHistory = (float*)NativeMemory.AllocZeroed((nuint)(totalFloats * sizeof(float)));
    }

    public virtual bool TryGetHiddenStateAt(int position, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hiddenHistory == null || (uint)position >= (uint)_maxSeqLen || destination.Length < _dim)
            return false;

        new ReadOnlySpan<float>(_hiddenHistory + (long)position * _dim, _dim).CopyTo(destination);
        return true;
    }

    public virtual bool TryGetMultiStreamHiddenStateAt(int position, Span<float> destination) =>
        TryGetHiddenStateAt(position, destination);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void RecordHiddenStateForPosition(int pos, float* src)
    {
        if (_hiddenHistory != null && IsLastStage && (uint)pos < (uint)_maxSeqLen)
        {
            ulong bytes = (ulong)_dim * sizeof(float);
            Buffer.MemoryCopy(src, _hiddenHistory + (long)pos * _dim, bytes, bytes);
        }
    }

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
        if (_weights.SharedExpertFeedForwardLength > maxFfn) maxFfn = _weights.SharedExpertFeedForwardLength;
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            if ((uint)modelLayer < (uint)_weights.Layers.Length && _weights.Layers[modelLayer].FfnGateShexpWeight != null)
            {
                int shexpFfn = GetSharedExpertFfnDim(modelLayer);
                if (shexpFfn > maxFfn) maxFfn = shexpFfn;
            }
        }

        // Base unmanaged scratch buffers for single token
        int normChunks = Math.Max(1, (_dim + 31) / 32);
        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)(normChunks * sizeof(float)));
        _gate = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _up = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnAct = (float*)NativeMemory.AllocZeroed((nuint)(maxFfn * sizeof(float)));
        _ffnActSums = (float*)NativeMemory.AllocZeroed((nuint)(Math.Max(1, (maxFfn + 31) / 32) * sizeof(float)));
        _ffnOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _expertDownOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        int gateUpDim = 2 * maxFfn;
        _gateUpFused = (float*)NativeMemory.AllocZeroed((nuint)(gateUpDim * sizeof(float)));

        // Base unmanaged scratch buffers for batched chunks
        _xBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * normChunks * sizeof(float)));
        _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _upBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * Math.Max(1, (maxFfn + 31) / 32) * sizeof(float)));
        _ffnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _gateUpBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * gateUpDim * sizeof(float)));

        if (_weights.ExpertCount > 0)
        {
            int moeSlots = MaxBatchSize * Math.Max(1, _weights.ExpertUsedCount);
            _moeBatchIndices = (int*)NativeMemory.AllocZeroed((nuint)(moeSlots * sizeof(int)));
            _moeBatchWeights = (float*)NativeMemory.AllocZeroed((nuint)(moeSlots * sizeof(float)));
            _moeSortedSlots = (int*)NativeMemory.AllocZeroed((nuint)(moeSlots * sizeof(int)));
            _moeSlotInputs = (float*)NativeMemory.AllocZeroed((nuint)(moeSlots * _dim * sizeof(float)));
            _moeSlotInputSums = (float*)NativeMemory.AllocZeroed((nuint)(moeSlots * normChunks * sizeof(float)));
            _moeSlotOutputs = (float*)NativeMemory.AllocZeroed((nuint)(moeSlots * _dim * sizeof(float)));
        }
    }

    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public virtual void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        ForwardStage(token, pos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public virtual void ForwardStage(
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
            RecordHiddenStateForPosition(pos, _x);
            if (computeLogits && logits.Length > 0)
            {
                if (_weights.OutNormWeight != null)
                {
                    QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                }
                else
                {
                    Buffer.MemoryCopy(_x, _normX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
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
        ExecuteSingleTokenFfn(modelLayer, _normX, _normXSums, _ffnOut, selectedIndices, selectedWeights);
        AddVector(_x, _ffnOut, _dim);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected void ExecuteSingleTokenFfn(
        int modelLayer,
        float* normX,
        float* normXSums,
        float* ffnOut,
        int* selectedIndices,
        float* selectedWeights)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFeedForwardLength;
            int numExperts = _weights.ExpertCount;
            int topK = _weights.ExpertUsedCount;

            QuantKernels.RouterTopK(normX, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);

            if (TieredMoeBatchHandler != null)
            {
                TieredMoeBatchHandler(
                    modelLayer,
                    1,
                    topK,
                    false,
                    selectedIndices,
                    selectedWeights,
                    normX,
                    normXSums,
                    ffnOut);
            }
            else
            {
                new Span<float>(ffnOut, _dim).Clear();

                bool fusedExps = layer.FfnGateUpExpsWeight != null && layer.FfnGateExpsWeight == null;
                long fusedSliceBytes = fusedExps ? (long)(2 * expertFfnDim) * GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, _dim) : 0;
                long gateSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
                long upSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
                long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

                for (int k = 0; k < topK; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    ExecuteSingleExpertInto(
                        layer,
                        expertIdx,
                        expertFfnDim,
                        fusedExps,
                        fusedSliceBytes,
                        gateSliceBytes,
                        upSliceBytes,
                        downSliceBytes,
                        normX,
                        normXSums,
                        _expertDownOut);

                    for (int d = 0; d < _dim; d++)
                    {
                        ffnOut[d] += weight * _expertDownOut[d];
                    }
                }
            }

            if (layer.FfnGateShexpWeight != null)
            {
                int shexpFfnDim = GetSharedExpertFfnDim(modelLayer);

                QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, normX, _gate, _dim, shexpFfnDim, normXSums);
                QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, normX, _up, _dim, shexpFfnDim, normXSums);
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                // Modulate by shared expert gate if present (Qwen 3.5 MoE)
                if (layer.FfnGateInpShexpWeight != null)
                {
                    float gateVal = QuantKernels.VecDotF32(normX, layer.FfnGateInpShexpWeight, _dim);
                    float shexpGate = 1.0f / (1.0f + MathF.Exp(-gateVal));
                    for (int d = 0; d < _dim; d++)
                    {
                        _expertDownOut[d] *= shexpGate;
                    }
                }

                AddVector(ffnOut, _expertDownOut, _dim);
            }
        }
        else
        {
            if (layer.HasFusedGateUp)
            {
                int gateUpDim = 2 * _ffnDim;
                QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, normX, _gateUpFused, _dim, gateUpDim, normXSums);
                if (layer.FfnUpBias != null) AddVector(_gateUpFused, layer.FfnUpBias, gateUpDim);
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Gate, normX, _gateUpFused, _dim, _ffnDim);
                    LoraWeights.Apply(modelLayer, LoraProjection.Up, normX, _gateUpFused + _ffnDim, _dim, _ffnDim);
                }

                QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + _ffnDim, _ffnAct, _ffnDim);
            }
            else
            {
                QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, normX, _gate, _dim, _ffnDim, normXSums);
                QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, normX, _up, _dim, _ffnDim, normXSums);
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Gate, normX, _gate, _dim, _ffnDim);
                    LoraWeights.Apply(modelLayer, LoraProjection.Up, normX, _up, _dim, _ffnDim);
                }
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
            }
            QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, _ffnDim);
            QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, ffnOut, _ffnDim, _dim, _ffnActSums);
            if (LoraWeights != null)
            {
                LoraWeights.Apply(modelLayer, LoraProjection.Down, _ffnAct, ffnOut, _ffnDim, _dim);
            }
        }
    }

    protected int GetSharedExpertFfnDim(int modelLayer)
    {
        if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
        {
            return (int)tShexp.Dimensions[1];
        }
        return _weights.SharedExpertFeedForwardLength > 0
            ? _weights.SharedExpertFeedForwardLength
            : _weights.ExpertFeedForwardLength * 2;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteSingleExpertInto(
        LayerWeights layer,
        int expertIdx,
        int expertFfnDim,
        bool fusedExps,
        long fusedSliceBytes,
        long gateSliceBytes,
        long upSliceBytes,
        long downSliceBytes,
        float* normX,
        float* normXSums,
        float* dstOut)
    {
        if (fusedExps)
        {
            byte* expGateUpWeight = layer.FfnGateUpExpsWeight + expertIdx * fusedSliceBytes;
            QuantKernels.MatVecMul(layer.FfnGateUpExpsType, expGateUpWeight, normX, _gateUpFused, _dim, 2 * expertFfnDim, normXSums);
            QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + expertFfnDim, _ffnAct, expertFfnDim);
        }
        else
        {
            byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
            byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;

            QuantKernels.MatVecMul(layer.FfnGateExpsType, expGateWeight, normX, _gate, _dim, expertFfnDim, normXSums);
            if (layer.FfnGateExpsBias != null) AddVector(_gate, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

            QuantKernels.MatVecMul(layer.FfnUpExpsType, expUpWeight, normX, _up, _dim, expertFfnDim, normXSums);
            if (layer.FfnUpExpsBias != null) AddVector(_up, layer.FfnUpExpsBias + (long)expertFfnDim * expertIdx, expertFfnDim);

            QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
        }

        byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;
        QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);
        QuantKernels.MatVecMul(layer.FfnDownExpsType, expDownWeight, _ffnAct, dstOut, expertFfnDim, _dim, _ffnActSums);
        if (layer.FfnDownExpsBias != null) AddVector(dstOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);
    }

    /// <summary>
    /// Executes coalesced batched MoE FFN across all tokens in a batch, grouping repeated experts
    /// into single MatMulBatch passes and batching shared expert weights across the full batch.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected void ExecuteBatchedMoeFfn(
        int modelLayer,
        float* normXBatch,
        float* normXSumBatch,
        float* ffnOutBatch,
        int batchSize,
        bool allowTailSkip)
    {
        if (batchSize <= 0) return;

        if (batchSize == 1 && (!allowTailSkip || (_moeTailSkipMinRank <= 0 && TieredMoeBatchHandler == null)))
        {
            ExecuteSingleTokenFfn(modelLayer, normXBatch, normXSumBatch, ffnOutBatch, _moeBatchIndices, _moeBatchWeights);
            return;
        }

        var layer = _weights.Layers[modelLayer];
        int expertFfnDim = _weights.ExpertFeedForwardLength;
        int numExperts = _weights.ExpertCount;
        int topK = _weights.ExpertUsedCount;
        int normXChunks = Math.Max(1, (_dim + 31) / 32);

        new Span<float>(ffnOutBatch, batchSize * _dim).Clear();

        if (topK > 0 && numExperts > 0)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.RouterTopK(
                    normXBatch + t * _dim,
                    layer.FfnGateInpWeight,
                    layer.FfnGateInpBias,
                    _dim,
                    numExperts,
                    topK,
                    _moeBatchIndices + t * topK,
                    _moeBatchWeights + t * topK,
                    _weights.NormTopK);
            }

            if (TieredMoeBatchHandler != null)
            {
                TieredMoeBatchHandler(
                    modelLayer,
                    batchSize,
                    topK,
                    allowTailSkip,
                    _moeBatchIndices,
                    _moeBatchWeights,
                    normXBatch,
                    normXSumBatch,
                    ffnOutBatch);
            }
            else
            {
                if (allowTailSkip && _moeTailSkipMinRank > 0 && _moeTailSkipMinRank < topK)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    int baseSlot = t * topK;
                    float cutoff = _moeBatchWeights[baseSlot] * 0.25f;
                    float keptSum = 0f;
                    bool skippedAny = false;
                    for (int k = 0; k < topK; k++)
                    {
                        if (k >= _moeTailSkipMinRank && _moeBatchWeights[baseSlot + k] < cutoff)
                        {
                            _moeBatchIndices[baseSlot + k] = -1;
                            _moeBatchWeights[baseSlot + k] = 0f;
                            skippedAny = true;
                        }
                        else
                        {
                            keptSum += _moeBatchWeights[baseSlot + k];
                        }
                    }

                    if (skippedAny && _weights.NormTopK && keptSum > 0f)
                    {
                        float inv = 1.0f / keptSum;
                        for (int k = 0; k < topK; k++)
                        {
                            if (_moeBatchIndices[baseSlot + k] >= 0)
                            {
                                _moeBatchWeights[baseSlot + k] *= inv;
                            }
                        }
                    }
                }
            }

            int slotCount = 0;
            int totalSlots = batchSize * topK;
            for (int slot = 0; slot < totalSlots; slot++)
            {
                if (_moeBatchIndices[slot] >= 0)
                {
                    _moeSortedSlots[slotCount++] = slot;
                }
            }

            for (int i = 1; i < slotCount; i++)
            {
                int keySlot = _moeSortedSlots[i];
                int keyExp = _moeBatchIndices[keySlot];
                int j = i - 1;
                while (j >= 0)
                {
                    int prevSlot = _moeSortedSlots[j];
                    int prevExp = _moeBatchIndices[prevSlot];
                    if (prevExp < keyExp || (prevExp == keyExp && prevSlot <= keySlot))
                        break;
                    _moeSortedSlots[j + 1] = prevSlot;
                    j--;
                }
                _moeSortedSlots[j + 1] = keySlot;
            }

            bool fusedExps = layer.FfnGateUpExpsWeight != null && layer.FfnGateExpsWeight == null;
            long fusedSliceBytes = fusedExps ? (long)(2 * expertFfnDim) * GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, _dim) : 0;
            long gateSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
            long upSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
            long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);
            int expFfnChunks = Math.Max(1, (expertFfnDim + 31) / 32);

            int gStart = 0;
            while (gStart < slotCount)
            {
                int expertIdx = _moeBatchIndices[_moeSortedSlots[gStart]];
                int gEnd = gStart + 1;
                while (gEnd < slotCount && _moeBatchIndices[_moeSortedSlots[gEnd]] == expertIdx)
                {
                    gEnd++;
                }

                int groupCount = gEnd - gStart;
                if (groupCount == 1)
                {
                    int slot = _moeSortedSlots[gStart];
                    int t = slot / topK;
                    ExecuteSingleExpertInto(
                        layer,
                        expertIdx,
                        expertFfnDim,
                        fusedExps,
                        fusedSliceBytes,
                        gateSliceBytes,
                        upSliceBytes,
                        downSliceBytes,
                        normXBatch + t * _dim,
                        normXSumBatch + t * normXChunks,
                        _moeSlotOutputs + slot * _dim);
                }
                else
                {
                    int gPos = gStart;
                    while (gPos < gEnd)
                    {
                        int chunkM = Math.Min(gEnd - gPos, MaxBatchSize);
                        for (int j = 0; j < chunkM; j++)
                        {
                            int t = _moeSortedSlots[gPos + j] / topK;
                            Buffer.MemoryCopy(normXBatch + t * _dim, _moeSlotInputs + j * _dim, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                            Buffer.MemoryCopy(normXSumBatch + t * normXChunks, _moeSlotInputSums + j * normXChunks, (ulong)(normXChunks * sizeof(float)), (ulong)(normXChunks * sizeof(float)));
                        }

                        if (fusedExps)
                        {
                            int gateUpDim = 2 * expertFfnDim;
                            byte* expGateUpWeight = layer.FfnGateUpExpsWeight + expertIdx * fusedSliceBytes;
                            QuantKernels.MatMulBatch(layer.FfnGateUpExpsType, expGateUpWeight, _moeSlotInputs, _gateUpBatch, _dim, gateUpDim, chunkM, _moeSlotInputSums);
                            for (int j = 0; j < chunkM; j++)
                            {
                                float* src = _gateUpBatch + j * gateUpDim;
                                QuantKernels.SwiGLU(src, src + expertFfnDim, _ffnActBatch + j * expertFfnDim, expertFfnDim);
                                QuantKernels.ComputeBlockSums32(_ffnActBatch + j * expertFfnDim, _ffnActSumBatch + j * expFfnChunks, expertFfnDim);
                            }
                        }
                        else
                        {
                            byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
                            byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;
                            QuantKernels.MatMulBatch(layer.FfnGateExpsType, expGateWeight, _moeSlotInputs, _gateBatch, _dim, expertFfnDim, chunkM, _moeSlotInputSums);
                            QuantKernels.MatMulBatch(layer.FfnUpExpsType, expUpWeight, _moeSlotInputs, _upBatch, _dim, expertFfnDim, chunkM, _moeSlotInputSums);
                            for (int j = 0; j < chunkM; j++)
                            {
                                float* g = _gateBatch + j * expertFfnDim;
                                float* u = _upBatch + j * expertFfnDim;
                                if (layer.FfnGateExpsBias != null) AddVector(g, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);
                                if (layer.FfnUpExpsBias != null) AddVector(u, layer.FfnUpExpsBias + (long)expertFfnDim * expertIdx, expertFfnDim);
                                QuantKernels.SwiGLU(g, u, _ffnActBatch + j * expertFfnDim, expertFfnDim);
                                QuantKernels.ComputeBlockSums32(_ffnActBatch + j * expertFfnDim, _ffnActSumBatch + j * expFfnChunks, expertFfnDim);
                            }
                        }

                        byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;
                        QuantKernels.MatMulBatch(layer.FfnDownExpsType, expDownWeight, _ffnActBatch, _moeSlotInputs, expertFfnDim, _dim, chunkM, _ffnActSumBatch);
                        for (int j = 0; j < chunkM; j++)
                        {
                            float* srcOut = _moeSlotInputs + j * _dim;
                            if (layer.FfnDownExpsBias != null) AddVector(srcOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);
                            int slot = _moeSortedSlots[gPos + j];
                            Buffer.MemoryCopy(srcOut, _moeSlotOutputs + slot * _dim, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                        }

                        gPos += chunkM;
                    }
                }

                gStart = gEnd;
            }

            for (int t = 0; t < batchSize; t++)
            {
                float* outT = ffnOutBatch + t * _dim;
                int baseSlot = t * topK;
                for (int k = 0; k < topK; k++)
                {
                    int slot = baseSlot + k;
                    if (_moeBatchIndices[slot] < 0) continue;
                    float weight = _moeBatchWeights[slot];
                    float* expertOut = _moeSlotOutputs + slot * _dim;
                    for (int d = 0; d < _dim; d++)
                    {
                        outT[d] += weight * expertOut[d];
                    }
                }
            }
            }
        }

        if (layer.FfnGateShexpWeight != null)
        {
            int shexpFfnDim = GetSharedExpertFfnDim(modelLayer);
            int shexpChunks = Math.Max(1, (shexpFfnDim + 31) / 32);
            QuantKernels.MatMulBatch(layer.FfnGateShexpType, layer.FfnGateShexpWeight, normXBatch, _gateBatch, _dim, shexpFfnDim, batchSize, normXSumBatch);
            QuantKernels.MatMulBatch(layer.FfnUpShexpType, layer.FfnUpShexpWeight, normXBatch, _upBatch, _dim, shexpFfnDim, batchSize, normXSumBatch);
            for (int t = 0; t < batchSize; t++)
            {
                float* act = _ffnActBatch + t * shexpFfnDim;
                QuantKernels.SwiGLU(_gateBatch + t * shexpFfnDim, _upBatch + t * shexpFfnDim, act, shexpFfnDim);
                QuantKernels.ComputeBlockSums32(act, _ffnActSumBatch + t * shexpChunks, shexpFfnDim);
            }

            QuantKernels.MatMulBatch(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnActBatch, _moeSlotInputs, shexpFfnDim, _dim, batchSize, _ffnActSumBatch);
            for (int t = 0; t < batchSize; t++)
            {
                float* shexpOut = _moeSlotInputs + t * _dim;
                if (layer.FfnGateInpShexpWeight != null)
                {
                    float gateVal = QuantKernels.VecDotF32(normXBatch + t * _dim, layer.FfnGateInpShexpWeight, _dim);
                    float shexpGate = 1.0f / (1.0f + MathF.Exp(-gateVal));
                    for (int d = 0; d < _dim; d++)
                    {
                        shexpOut[d] *= shexpGate;
                    }
                }

                AddVector(ffnOutBatch + t * _dim, shexpOut, _dim);
            }
        }
    }

    protected float* EnsureVerifyLogitsBuffer(int batchSize)
    {
        int needed = Math.Max(batchSize, 16) * _weights.VocabSize;
        if (_verifyLogitsBatch == null || _verifyLogitsBatchCapacity < needed)
        {
            if (_verifyLogitsBatch != null)
            {
                NativeMemory.Free(_verifyLogitsBatch);
            }
            _verifyLogitsBatch = (float*)NativeMemory.Alloc((nuint)((long)needed * sizeof(float)));
            _verifyLogitsBatchCapacity = needed;
        }
        return _verifyLogitsBatch;
    }

    protected static int ComputeGreedyArgmax(float* logits, int vocabSize)
    {
        if (vocabSize <= 0) return 0;
        int bestIdx = 0;
        float bestVal = logits[0];
        for (int i = 1; i < vocabSize; i++)
        {
            if (logits[i] > bestVal)
            {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /// <summary>
    /// Evaluates a sequence of prompt tokens in batched chunks.
    /// </summary>
    public virtual void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes batched multi-token speculative verification and returns greedy argmax token IDs.
    /// </summary>
    public virtual void VerifyBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<int> greedyTokens,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;
        int offset = 0;
        int vocabSize = _weights.VocabSize;
        while (offset < tokens.Length)
        {
            int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
            float* logitsBuf = EnsureVerifyLogitsBuffer(batchSize);
            var logitsSpan = new Span<float>(logitsBuf, batchSize * vocabSize);
            ForwardBatchChunkCore(
                tokens.Slice(offset, batchSize),
                startPos + offset,
                default,
                default,
                Span<float>.Empty,
                logitsSpan,
                vocabSize,
                kvCache,
                computeLastLogits: false,
                computeAllLogits: true,
                allowTailSkip: true);
            for (int t = 0; t < batchSize; t++)
            {
                greedyTokens[offset + t] = ComputeGreedyArgmax(logitsBuf + t * vocabSize, vocabSize);
            }
            offset += batchSize;
        }
    }

    /// <summary>
    /// Executes batched multi-token speculative verification and writes full per-token logits.
    /// </summary>
    public virtual void VerifyBatchLogits(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> logitsOut,
        int vocabStride,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;
        int offset = 0;
        while (offset < tokens.Length)
        {
            int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
            ForwardBatchChunkCore(
                tokens.Slice(offset, batchSize),
                startPos + offset,
                default,
                default,
                Span<float>.Empty,
                logitsOut.Slice(offset * vocabStride),
                vocabStride,
                kvCache,
                computeLastLogits: false,
                computeAllLogits: true,
                allowTailSkip: true);
            offset += batchSize;
        }
    }

    /// <summary>
    /// Executes batched prefill for the layers assigned to this stage.
    /// </summary>
    public virtual void ForwardBatchStage(
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

            ForwardBatchChunkCore(
                tokSlice,
                startPos + offset,
                inSlice,
                outSlice,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                Span<float>.Empty,
                0,
                kvCache,
                computeLastLogits: isLastChunk && computeLogits,
                computeAllLogits: false,
                allowTailSkip: false);
            offset += batchSize;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardBatchChunkCore(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> lastTokenLogits,
        Span<float> allTokensLogits,
        int allTokensLogitsStride,
        KVCache? kvCache,
        bool computeLastLogits,
        bool computeAllLogits,
        bool allowTailSkip)
    {
        int batchSize = !chunkTokens.IsEmpty ? chunkTokens.Length : (inputXBatch.Length / _dim);
        if (batchSize <= 0) return;
        int normXChunks = Math.Max(1, (_dim + 31) / 32);

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
            ForwardBatchChunkFfn(l, modelLayer, batchSize, normXChunks, selectedIndices, selectedWeights, allowTailSkip);
        }

        // 3. Final normalization and output logits
        FinalizeBatchChunkOutputs(
            chunkStartPos,
            batchSize,
            outputXBatch,
            lastTokenLogits,
            allTokensLogits,
            allTokensLogitsStride,
            computeLastLogits,
            computeAllLogits);
    }

    /// <summary>
    /// Records chunk hidden states into <c>_x</c>/<c>_hiddenHistory</c> and computes either all-token
    /// verification logits, single last-token logits, or copies stage outputs.
    /// </summary>
    protected void FinalizeBatchChunkOutputs(
        int chunkStartPos,
        int batchSize,
        Span<float> outputXBatch,
        Span<float> lastTokenLogits,
        Span<float> allTokensLogits,
        int allTokensLogitsStride,
        bool computeLastLogits,
        bool computeAllLogits)
    {
        if (batchSize <= 0) return;
        int normXChunks = Math.Max(1, (_dim + 31) / 32);

        if (IsLastStage)
        {
            for (int t = 0; t < batchSize; t++)
            {
                RecordHiddenStateForPosition(chunkStartPos + t, _xBatch + t * _dim);
            }

            int lastIdx = batchSize - 1;
            Buffer.MemoryCopy(_xBatch + lastIdx * _dim, _x, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));

            if (computeAllLogits && allTokensLogits.Length > 0)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    float* xt = _xBatch + t * _dim;
                    float* normXt = _normXBatch + t * _dim;
                    if (_weights.OutNormWeight != null)
                    {
                        QuantKernels.RMSNorm(xt, _weights.OutNormWeight, normXt, _dim, _weights.RmsNormEps);
                    }
                    else
                    {
                        Buffer.MemoryCopy(xt, normXt, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                    }
                    QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
                }

                Buffer.MemoryCopy(_normXBatch + lastIdx * _dim, _normX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                Buffer.MemoryCopy(_normXSumBatch + lastIdx * normXChunks, _normXSums, (ulong)(normXChunks * sizeof(float)), (ulong)(normXChunks * sizeof(float)));

                int vocabSize = _weights.VocabSize;
                fixed (float* pAll = allTokensLogits)
                {
                    if (allTokensLogitsStride == vocabSize)
                    {
                        QuantKernels.MatMulBatch(_weights.OutType, _weights.OutWeight, _normXBatch, pAll, _dim, vocabSize, batchSize, _normXSumBatch);
                    }
                    else
                    {
                        float* tmpLogits = EnsureVerifyLogitsBuffer(batchSize);
                        QuantKernels.MatMulBatch(_weights.OutType, _weights.OutWeight, _normXBatch, tmpLogits, _dim, vocabSize, batchSize, _normXSumBatch);
                        for (int t = 0; t < batchSize; t++)
                        {
                            Buffer.MemoryCopy(tmpLogits + t * vocabSize, pAll + t * allTokensLogitsStride, (ulong)(vocabSize * sizeof(float)), (ulong)(vocabSize * sizeof(float)));
                        }
                    }
                }
            }
            else if (computeLastLogits && lastTokenLogits.Length > 0)
            {
                if (_weights.OutNormWeight != null)
                {
                    QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                }
                else
                {
                    Buffer.MemoryCopy(_x, _normX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                fixed (float* pLogits = lastTokenLogits)
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
    protected void ComputeBatchChunkFfnOut(int modelLayer, int batchSize, bool allowTailSkip = false)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsMoe)
        {
            ExecuteBatchedMoeFfn(modelLayer, _normXBatch, _normXSumBatch, _ffnOutBatch, batchSize, allowTailSkip);
        }
        else
        {
            int ffnChunks = Math.Max(1, (_ffnDim + 31) / 32);
            if (layer.HasFusedGateUp)
            {
                int gateUpDim = 2 * _ffnDim;
                QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _gateUpBatch, _dim, gateUpDim, batchSize, _normXSumBatch);
                for (int t = 0; t < batchSize; t++)
                {
                    float* normXt = _normXBatch + t * _dim;
                    float* src = _gateUpBatch + t * gateUpDim;
                    if (layer.FfnUpBias != null) AddVector(src, layer.FfnUpBias, gateUpDim);
                    if (LoraWeights != null)
                    {
                        LoraWeights.Apply(modelLayer, LoraProjection.Gate, normXt, src, _dim, _ffnDim);
                        LoraWeights.Apply(modelLayer, LoraProjection.Up, normXt, src + _ffnDim, _dim, _ffnDim);
                    }
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

            if (LoraWeights != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Down, _ffnActBatch + t * _ffnDim, _ffnOutBatch + t * _dim, _ffnDim, _dim);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected void ForwardBatchChunkFfn(int l, int modelLayer, int batchSize, int normXChunks, int* selectedIndices, float* selectedWeights, bool allowTailSkip = false)
    {
        ComputeBatchChunkFfnOut(modelLayer, batchSize, allowTailSkip);
        for (int t = 0; t < batchSize; t++)
        {
            AddVector(_xBatch + t * _dim, _ffnOutBatch + t * _dim, _dim);
        }
    }

    /// <summary>
    /// Prefills the KV cache with prompt tokens without computing final output logits.
    /// </summary>
    public void PrefillPrompt(ReadOnlySpan<int> tokens, KVCache kvCache)
    {
        ForwardBatch(tokens, 0, Span<float>.Empty, kvCache, computeLogits: false);
    }

    /// <inheritdoc />
    public virtual void ResetState()
    {
    }

    /// <inheritdoc />
    public virtual long StateSnapshotBytes => 0;

    /// <inheritdoc />
    public virtual void SaveRecurrentState(Span<byte> destination)
    {
    }

    /// <inheritdoc />
    public virtual void RestoreRecurrentState(ReadOnlySpan<byte> source)
    {
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

    protected static void FreeIfAllocated(ref int* ptr)
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
            FreeIfAllocated(ref _hiddenHistory);

            FreeIfAllocated(ref _moeBatchIndices);
            FreeIfAllocated(ref _moeBatchWeights);
            FreeIfAllocated(ref _moeSortedSlots);
            FreeIfAllocated(ref _moeSlotInputs);
            FreeIfAllocated(ref _moeSlotInputSums);
            FreeIfAllocated(ref _moeSlotOutputs);
            FreeIfAllocated(ref _verifyLogitsBatch);
            _verifyLogitsBatchCapacity = 0;

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
