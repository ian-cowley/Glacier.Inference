namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Weight view passed into <see cref="PleLayerProcessor.ExecuteToken"/> for a PLE layer.
/// </summary>
public unsafe readonly struct PleLayerWeights
{
    public GgufType PleKeyType { get; init; }
    public byte* PleKeyWeight { get; init; }
    public GgufType PleValueType { get; init; }
    public byte* PleValueWeight { get; init; }
    public float* PleNormKey { get; init; }
    public float* PleNormQuery { get; init; }
    public float* PleNormConv { get; init; }
    public float* PleConv1dWeight { get; init; }
}

/// <summary>
/// Weight view passed into <see cref="QsaSparseAttentionIndexer.RecordKeyAndSelect"/>.
/// </summary>
public unsafe readonly struct QsaIndexerWeights
{
    public GgufType QProjType { get; init; }
    public byte* QProjWeight { get; init; }
    public GgufType KProjType { get; init; }
    public byte* KProjWeight { get; init; }
    public float* QNormWeight { get; init; }
    public float* KNormWeight { get; init; }
}

/// <summary>
/// Implements Gated Multi-Stream Hyper-Connections (<c>gr_read</c> / <c>gr_write</c> / final mixer)
/// used in Qwen3.8-Flash-Next (<c>qwen4exp</c>).
/// </summary>
public sealed unsafe class HyperConnectionStack : IDisposable
{
    private readonly int _hcCount;
    private readonly int _dim;
    private readonly int _hcLowRank;
    private readonly int _hcDim;
    private readonly float _rmsEps;
    private readonly int _maxBatchSize;

    private float* _r;
    private float* _xn;
    private float* _lo;
    private float* _up;
    private float* _inject;
    private float* _subOut;
    private bool _disposed;

    public int HcCount => _hcCount;
    public int Dim => _dim;
    public int HcLowRank => _hcLowRank;
    public int HcDim => _hcDim;
    public int MaxBatchSize => _maxBatchSize;

    public float* SubBlockOutputBuffer => _subOut;

    public HyperConnectionStack(
        int hcCount,
        int dim,
        int hcLowRank,
        float rmsEps = 1e-6f,
        int maxBatchSize = 64)
    {
        if (hcCount <= 0) throw new ArgumentOutOfRangeException(nameof(hcCount));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (hcLowRank <= 0) throw new ArgumentOutOfRangeException(nameof(hcLowRank));
        if (maxBatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(maxBatchSize));

        _hcCount = hcCount;
        _dim = dim;
        _hcLowRank = hcLowRank;
        _hcDim = hcCount * dim;
        _rmsEps = rmsEps;
        _maxBatchSize = maxBatchSize;

        _r = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _hcDim * sizeof(float)), 64);
        _xn = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _hcDim * sizeof(float)), 64);
        _lo = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _hcLowRank * sizeof(float)), 64);
        _up = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _hcDim * sizeof(float)), 64);
        _inject = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _hcCount * sizeof(float)), 64);
        _subOut = (float*)NativeMemory.AlignedAlloc((nuint)(maxBatchSize * _dim * sizeof(float)), 64);

        NativeMemory.Clear(_r, (nuint)(maxBatchSize * _hcDim * sizeof(float)));
        NativeMemory.Clear(_inject, (nuint)(maxBatchSize * _hcCount * sizeof(float)));
        NativeMemory.Clear(_subOut, (nuint)(maxBatchSize * _dim * sizeof(float)));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetResidualTokenPtr(int batchIdx = 0) => _r + (long)batchIdx * _hcDim;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetInjectTokenPtr(int batchIdx = 0) => _inject + (long)batchIdx * _hcCount;

    /// <summary>
    /// Broadcasts a token embedding vector <c>[dim]</c> across all <c>hcCount</c> streams.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void InitTokenFromEmbedding(float* embedSrc, int batchIdx = 0)
    {
        float* rDst = GetResidualTokenPtr(batchIdx);
        ulong copyBytes = (ulong)_dim * sizeof(float);
        for (int c = 0; c < _hcCount; c++)
        {
            Buffer.MemoryCopy(embedSrc, rDst + (long)c * _dim, copyBytes, copyBytes);
        }
    }

    /// <summary>
    /// Executes <c>gr_read</c>: per-stream RMSNorm, low-rank bottleneck SiLU((xn @ wDown) / hc),
    /// gated stream average into <paramref name="mixedOut"/>, and optional write-back injection logits.
    /// When <paramref name="wInject"/> is null, runs as the final pre-LM-head stream mixer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Read(
        float* rState,
        float* wNorm,
        GgufType downType,
        byte* wDown,
        GgufType upType,
        byte* wUp,
        GgufType injectType,
        byte* wInject,
        float* mixedOut,
        float* injectOut)
    {
        float invHc = 1.0f / _hcCount;

        // 1. Per-stream RMSNorm
        for (int c = 0; c < _hcCount; c++)
        {
            float* srcC = rState + (long)c * _dim;
            float* dstC = _xn + (long)c * _dim;
            float* wC = wNorm != null ? wNorm + (long)c * _dim : null;
            QuantKernels.RMSNorm(srcC, wC, dstC, _dim, _rmsEps);
        }

        // 2. Low-rank bottleneck: lo = silu((xn @ wDown) / hc)
        QuantKernels.MatVecMul(downType, wDown, _xn, _lo, _hcDim, _hcLowRank);
        for (int k = 0; k < _hcLowRank; k++)
        {
            float z = _lo[k] * invHc;
            _lo[k] = z / (1.0f + MathF.Exp(-z));
        }

        // 3. Up-gate and stream mean: mixed = mean_c(xn[c] * sigmoid(lo @ wUp[c]))
        QuantKernels.MatVecMul(upType, wUp, _lo, _up, _hcLowRank, _hcDim);
        for (int d = 0; d < _dim; d++)
        {
            mixedOut[d] = 0f;
        }

        for (int c = 0; c < _hcCount; c++)
        {
            float* xnc = _xn + (long)c * _dim;
            float* upc = _up + (long)c * _dim;
            for (int d = 0; d < _dim; d++)
            {
                float gate = 1.0f / (1.0f + MathF.Exp(-upc[d]));
                mixedOut[d] += xnc[d] * gate;
            }
        }

        for (int d = 0; d < _dim; d++)
        {
            mixedOut[d] *= invHc;
        }

        // 4. Optional injection logits: inject[c] = dot(xn, wInject[c])
        if (wInject != null && injectOut != null)
        {
            QuantKernels.MatVecMul(injectType, wInject, _xn, injectOut, _hcDim, _hcCount);
        }
    }

    /// <summary>
    /// Batched <c>gr_read</c> over <paramref name="batchSize"/> consecutive slots starting at <c>GetResidualTokenPtr(0)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadBatch(
        int batchSize,
        float* wNorm,
        GgufType downType,
        byte* wDown,
        GgufType upType,
        byte* wUp,
        GgufType injectType,
        byte* wInject,
        float* mixedOutBatch)
    {
        if (batchSize <= 1)
        {
            Read(GetResidualTokenPtr(0), wNorm, downType, wDown, upType, wUp, injectType, wInject, mixedOutBatch, wInject != null ? GetInjectTokenPtr(0) : null);
            return;
        }

        float invHc = 1.0f / _hcCount;
        for (int t = 0; t < batchSize; t++)
        {
            float* rT = _r + (long)t * _hcDim;
            float* xnT = _xn + (long)t * _hcDim;
            for (int c = 0; c < _hcCount; c++)
            {
                float* srcC = rT + (long)c * _dim;
                float* dstC = xnT + (long)c * _dim;
                float* wC = wNorm != null ? wNorm + (long)c * _dim : null;
                QuantKernels.RMSNorm(srcC, wC, dstC, _dim, _rmsEps);
            }
        }

        QuantKernels.MatMulBatch(downType, wDown, _xn, _lo, _hcDim, _hcLowRank, batchSize);
        int totalLo = batchSize * _hcLowRank;
        for (int k = 0; k < totalLo; k++)
        {
            float z = _lo[k] * invHc;
            _lo[k] = z / (1.0f + MathF.Exp(-z));
        }

        QuantKernels.MatMulBatch(upType, wUp, _lo, _up, _hcLowRank, _hcDim, batchSize);

        for (int t = 0; t < batchSize; t++)
        {
            float* xnT = _xn + (long)t * _hcDim;
            float* upT = _up + (long)t * _hcDim;
            float* mixedT = mixedOutBatch + (long)t * _dim;
            for (int d = 0; d < _dim; d++) mixedT[d] = 0f;

            for (int c = 0; c < _hcCount; c++)
            {
                float* xnc = xnT + (long)c * _dim;
                float* upc = upT + (long)c * _dim;
                for (int d = 0; d < _dim; d++)
                {
                    float gate = 1.0f / (1.0f + MathF.Exp(-upc[d]));
                    mixedT[d] += xnc[d] * gate;
                }
            }

            for (int d = 0; d < _dim; d++)
            {
                mixedT[d] *= invHc;
            }
        }

        if (wInject != null)
        {
            QuantKernels.MatMulBatch(injectType, wInject, _xn, _inject, _hcDim, _hcCount, batchSize);
        }
    }

    /// <summary>
    /// Executes <c>gr_write</c> across <paramref name="batchSize"/> consecutive slots.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void WriteBatch(int batchSize, float* blockOutBatch)
    {
        for (int t = 0; t < batchSize; t++)
        {
            Write(GetResidualTokenPtr(t), blockOutBatch + (long)t * _dim, GetInjectTokenPtr(t));
        }
    }

    /// <summary>
    /// Executes <c>gr_write</c>: folds <paramref name="blockOut"/> back into each stream with weight
    /// <c>2 * sigmoid(inject[c] / hc)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Write(float* rState, float* blockOut, float* inject)
    {
        float invHc = 1.0f / _hcCount;
        for (int c = 0; c < _hcCount; c++)
        {
            float weight = 2.0f / (1.0f + MathF.Exp(-(inject[c] * invHc)));
            QuantKernels.VecAddWeighted(blockOut, rState + (long)c * _dim, weight, _dim);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_r != null) { NativeMemory.AlignedFree(_r); _r = null; }
        if (_xn != null) { NativeMemory.AlignedFree(_xn); _xn = null; }
        if (_lo != null) { NativeMemory.AlignedFree(_lo); _lo = null; }
        if (_up != null) { NativeMemory.AlignedFree(_up); _up = null; }
        if (_inject != null) { NativeMemory.AlignedFree(_inject); _inject = null; }
        if (_subOut != null) { NativeMemory.AlignedFree(_subOut); _subOut = null; }
    }
}

/// <summary>
/// Implements Per-Layer N-gram Embedding (PLE) residual folding and causal dilated depthwise convolution
/// over rolling normalized history, including deterministic snapshot/restore for speculative decoding.
/// </summary>
public sealed unsafe class PleLayerProcessor : IDisposable
{
    private readonly int _pleLayerCount;
    private readonly int _hcCount;
    private readonly int _dim;
    private readonly int _hcDim;
    private readonly int _pleDim;
    private readonly PleHashConfig _hashConfig;
    private readonly float _rmsEps;
    private readonly int _convKernel;
    private readonly int _ngramSize;
    private readonly int _historySlots;

    private readonly int[] _prevTokens;
    private float* _convHistory;
    private float* _key;
    private float* _query;
    private float* _value;
    private float* _gated;
    private float* _normalized;
    private float* _convOut;
    private bool _disposed;

    public int PleLayerCount => _pleLayerCount;
    public int HistorySlots => _historySlots;
    public int NgramPrevCount => _prevTokens.Length;
    public long LayerHistoryBytes => (long)_historySlots * _hcDim * sizeof(float);
    public ReadOnlySpan<int> PreviousTokens => _prevTokens;

    public long StateSnapshotBytes =>
        ((long)_prevTokens.Length * sizeof(int)) +
        ((long)_pleLayerCount * _historySlots * _hcDim * sizeof(float));

    /// <summary>
    /// Populates the oldest-first n-gram predecessor window after pushing <paramref name="tokensConsumed"/>
    /// leading tokens from <paramref name="batchTokens"/> onto <paramref name="initialPrev"/>.
    /// </summary>
    public static void ComputePrevWindowAfterSteps(
        ReadOnlySpan<int> initialPrev,
        ReadOnlySpan<int> batchTokens,
        int tokensConsumed,
        Span<int> destination)
    {
        int pLen = initialPrev.Length;
        for (int i = 0; i < pLen; i++)
        {
            int streamIdx = tokensConsumed - pLen + i;
            destination[i] = streamIdx >= 0
                ? batchTokens[streamIdx]
                : initialPrev[pLen + streamIdx];
        }
    }

    public void SaveLayerHistory(int pleLayerSlot, Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)pleLayerSlot >= (uint)_pleLayerCount)
            throw new ArgumentOutOfRangeException(nameof(pleLayerSlot));

        long layerBytes = LayerHistoryBytes;
        if (destination.Length < layerBytes)
            throw new ArgumentException($"Destination too small ({destination.Length} < {layerBytes}).", nameof(destination));

        if (layerBytes > 0)
        {
            float* src = _convHistory + (long)pleLayerSlot * _historySlots * _hcDim;
            fixed (byte* pDst = destination)
            {
                Buffer.MemoryCopy(src, pDst, (ulong)layerBytes, (ulong)layerBytes);
            }
        }
    }

    public PleLayerProcessor(
        int pleLayerCount,
        int hcCount,
        int dim,
        PleHashConfig hashConfig,
        float rmsEps = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(hashConfig);
        if (pleLayerCount <= 0) throw new ArgumentOutOfRangeException(nameof(pleLayerCount));
        if (hcCount <= 0) throw new ArgumentOutOfRangeException(nameof(hcCount));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));

        _pleLayerCount = pleLayerCount;
        _hcCount = hcCount;
        _dim = dim;
        _hcDim = hcCount * dim;
        _hashConfig = hashConfig;
        _pleDim = hashConfig.EmbeddingDim;
        _rmsEps = rmsEps;
        _convKernel = Math.Max(1, hashConfig.ConvKernel);
        _ngramSize = Math.Max(1, hashConfig.NgramSize);
        _historySlots = (_convKernel - 1) * _ngramSize;

        _prevTokens = new int[Math.Max(0, _ngramSize - 1)];
        Array.Fill(_prevTokens, -1);

        long histFloats = Math.Max(1L, (long)_pleLayerCount * _historySlots * _hcDim);
        _convHistory = (float*)NativeMemory.AlignedAlloc((nuint)(histFloats * sizeof(float)), 64);
        NativeMemory.Clear(_convHistory, (nuint)(histFloats * sizeof(float)));

        _key = (float*)NativeMemory.AlignedAlloc((nuint)(_hcDim * sizeof(float)), 64);
        _query = (float*)NativeMemory.AlignedAlloc((nuint)(_hcDim * sizeof(float)), 64);
        _value = (float*)NativeMemory.AlignedAlloc((nuint)(_dim * sizeof(float)), 64);
        _gated = (float*)NativeMemory.AlignedAlloc((nuint)(_hcDim * sizeof(float)), 64);
        _normalized = (float*)NativeMemory.AlignedAlloc((nuint)(_hcDim * sizeof(float)), 64);
        _convOut = (float*)NativeMemory.AlignedAlloc((nuint)(_hcDim * sizeof(float)), 64);
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Array.Fill(_prevTokens, -1);
        long histBytes = (long)_pleLayerCount * _historySlots * _hcDim * sizeof(float);
        if (histBytes > 0)
        {
            NativeMemory.Clear(_convHistory, (nuint)histBytes);
        }
    }

    /// <summary>
    /// Appends an accepted/processed token ID to the oldest-first rolling n-gram predecessor window.
    /// </summary>
    public void PushToken(int tokenId)
    {
        if (_prevTokens.Length == 0) return;
        for (int i = 0; i < _prevTokens.Length - 1; i++)
        {
            _prevTokens[i] = _prevTokens[i + 1];
        }
        _prevTokens[_prevTokens.Length - 1] = tokenId;
    }

    public void SaveState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long needed = StateSnapshotBytes;
        if (destination.Length < needed)
            throw new ArgumentException($"Destination too small ({destination.Length} < {needed}).", nameof(destination));

        int prevBytes = _prevTokens.Length * sizeof(int);
        MemoryMarshal.AsBytes(_prevTokens.AsSpan()).CopyTo(destination);

        long histBytes = (long)_pleLayerCount * _historySlots * _hcDim * sizeof(float);
        if (histBytes > 0)
        {
            fixed (byte* pDst = destination.Slice(prevBytes))
            {
                Buffer.MemoryCopy(_convHistory, pDst, (ulong)histBytes, (ulong)histBytes);
            }
        }
    }

    public void RestoreState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long needed = StateSnapshotBytes;
        if (source.Length < needed)
            throw new ArgumentException($"Source too small ({source.Length} < {needed}).", nameof(source));

        int prevBytes = _prevTokens.Length * sizeof(int);
        source.Slice(0, prevBytes).CopyTo(MemoryMarshal.AsBytes(_prevTokens.AsSpan()));

        long histBytes = (long)_pleLayerCount * _historySlots * _hcDim * sizeof(float);
        if (histBytes > 0)
        {
            fixed (byte* pSrc = source.Slice(prevBytes))
            {
                Buffer.MemoryCopy(pSrc, _convHistory, (ulong)histBytes, (ulong)histBytes);
            }
        }
    }

    /// <summary>
    /// Folds the gathered PLE embedding into the multi-stream residual state <paramref name="rStateOrX"/>.
    /// Updates the causal normalized history buffer in-place for this PLE layer slot.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ExecuteToken(
        int pleLayerSlot,
        float* rStateOrX,
        float* pleEmb,
        in PleLayerWeights w)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)pleLayerSlot >= (uint)_pleLayerCount)
            throw new ArgumentOutOfRangeException(nameof(pleLayerSlot));

        // 1. Key = grouped_rms_norm(ple_key @ emb, ple_norm_key)
        QuantKernels.MatVecMul(w.PleKeyType, w.PleKeyWeight, pleEmb, _key, _pleDim, _hcDim);
        for (int c = 0; c < _hcCount; c++)
        {
            float* kc = _key + (long)c * _dim;
            float* wNorm = w.PleNormKey != null ? w.PleNormKey + (long)c * _dim : null;
            QuantKernels.RMSNorm(kc, wNorm, kc, _dim, _rmsEps);
        }

        // 2. Query = grouped_rms_norm(hidden, ple_norm_query)
        for (int c = 0; c < _hcCount; c++)
        {
            float* rc = rStateOrX + (long)c * _dim;
            float* qc = _query + (long)c * _dim;
            float* wNorm = w.PleNormQuery != null ? w.PleNormQuery + (long)c * _dim : null;
            QuantKernels.RMSNorm(rc, wNorm, qc, _dim, _rmsEps);
        }

        // 3. Value = ple_value @ emb
        QuantKernels.MatVecMul(w.PleValueType, w.PleValueWeight, pleEmb, _value, _pleDim, _dim);

        // 4. Signed-sqrt sigmoid gate per stream
        float invSqrtDim = 1.0f / MathF.Sqrt(_dim);
        for (int c = 0; c < _hcCount; c++)
        {
            float s = QuantKernels.VecDotF32(_key + (long)c * _dim, _query + (long)c * _dim, _dim) * invSqrtDim;
            float mag = MathF.Sqrt(MathF.Max(MathF.Abs(s), 1e-6f));
            float signedSqrt = MathF.CopySign(mag, s);
            float gate = 1.0f / (1.0f + MathF.Exp(-signedSqrt));

            float* gatedC = _gated + (long)c * _dim;
            for (int d = 0; d < _dim; d++)
            {
                gatedC[d] = _value[d] * gate;
            }
        }

        // 5. Normalize gated BEFORE dilated depthwise conv and history insertion
        for (int c = 0; c < _hcCount; c++)
        {
            float* gatedC = _gated + (long)c * _dim;
            float* normC = _normalized + (long)c * _dim;
            float* wNorm = w.PleNormConv != null ? w.PleNormConv + (long)c * _dim : null;
            QuantKernels.RMSNorm(gatedC, wNorm, normC, _dim, _rmsEps);
        }

        // 6. Causal dilated depthwise conv over normalized history (kernel=4, dilation=ngram_size)
        float* histBase = _convHistory + (long)pleLayerSlot * _historySlots * _hcDim;
        float* convW = w.PleConv1dWeight;

        if (convW != null)
        {
            for (int ch = 0; ch < _hcDim; ch++)
            {
                float acc = 0f;
                float* wc = convW + (long)ch * _convKernel;
                for (int k = 0; k < _convKernel; k++)
                {
                    int lag = (_convKernel - 1 - k) * _ngramSize;
                    float sample = lag == 0
                        ? _normalized[ch]
                        : (lag <= _historySlots ? histBase[(long)(_historySlots - lag) * _hcDim + ch] : 0f);
                    acc += wc[k] * sample;
                }
                _convOut[ch] = acc;
            }
        }
        else
        {
            for (int ch = 0; ch < _hcDim; ch++)
            {
                _convOut[ch] = 0f;
            }
        }

        // 7. Roll history left by 1 and push normalized current step
        if (_historySlots > 0)
        {
            if (_historySlots > 1)
            {
                ulong shiftBytes = (ulong)(_historySlots - 1) * (ulong)_hcDim * sizeof(float);
                Buffer.MemoryCopy(histBase + _hcDim, histBase, shiftBytes, shiftBytes);
            }
            ulong rowBytes = (ulong)_hcDim * sizeof(float);
            Buffer.MemoryCopy(_normalized, histBase + (long)(_historySlots - 1) * _hcDim, rowBytes, rowBytes);
        }

        // 8. Residual update: hidden += gated + silu(conv)
        for (int ch = 0; ch < _hcDim; ch++)
        {
            float cv = _convOut[ch];
            float silu = cv / (1.0f + MathF.Exp(-cv));
            rStateOrX[ch] += _gated[ch] + silu;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_convHistory != null) { NativeMemory.AlignedFree(_convHistory); _convHistory = null; }
        if (_key != null) { NativeMemory.AlignedFree(_key); _key = null; }
        if (_query != null) { NativeMemory.AlignedFree(_query); _query = null; }
        if (_value != null) { NativeMemory.AlignedFree(_value); _value = null; }
        if (_gated != null) { NativeMemory.AlignedFree(_gated); _gated = null; }
        if (_normalized != null) { NativeMemory.AlignedFree(_normalized); _normalized = null; }
        if (_convOut != null) { NativeMemory.AlignedFree(_convOut); _convOut = null; }
    }
}

/// <summary>
/// Implements the Qwen Sparse Attention (QSA) Indexer: pooled key block compression,
/// per-head ReLU dot scoring, incomplete-tail protection (<c>+1e9f</c>), and exact top-K cell selection.
/// </summary>
public sealed unsafe class QsaSparseAttentionIndexer : IDisposable
{
    private readonly int _dim;
    private readonly int _idxHeads;
    private readonly int _idxKeyDim;
    private readonly int _blockSize;
    private readonly int _topK;
    private readonly int _rotaryDim;
    private readonly float _ropeFreqBase;
    private readonly float _rmsEps;
    private readonly int _maxSeqLen;
    private readonly int _maxBlocks;

    private float* _tail;
    private float* _dead;
    private float* _pooled;
    private float* _qIdx;
    private float* _blockScores;
    private float* _scores;
    private float* _heapScores;
    private int* _heapIndices;
    private int* _selectedIds;
    private int _lastNkv;
    private bool _disposed;

    public int BlockSize => _blockSize;
    public int TopK => _topK;
    public int MaxSelectionWidth => _topK + _blockSize - 1;

    public long StateSnapshotBytes => (long)(_blockSize + 1) * _idxKeyDim * sizeof(float);

    public ReadOnlySpan<float> LastScores => _lastNkv > 0 ? new ReadOnlySpan<float>(_scores, _lastNkv) : ReadOnlySpan<float>.Empty;

    public QsaSparseAttentionIndexer(
        int dim,
        int idxHeads = 4,
        int idxKeyDim = 128,
        int blockSize = 4,
        int topK = 2048,
        int rotaryDim = 64,
        float ropeFreqBase = 10_000_000.0f,
        float rmsEps = 1e-6f,
        int maxSeqLen = 4096)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (idxHeads <= 0) throw new ArgumentOutOfRangeException(nameof(idxHeads));
        if (idxKeyDim <= 0) throw new ArgumentOutOfRangeException(nameof(idxKeyDim));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _dim = dim;
        _idxHeads = idxHeads;
        _idxKeyDim = idxKeyDim;
        _blockSize = blockSize;
        _topK = topK;
        _rotaryDim = rotaryDim > 0 && rotaryDim <= idxKeyDim ? rotaryDim : idxKeyDim;
        _ropeFreqBase = ropeFreqBase;
        _rmsEps = rmsEps;
        _maxSeqLen = maxSeqLen;
        _maxBlocks = (maxSeqLen + blockSize - 1) / blockSize + 2;

        _tail = (float*)NativeMemory.AlignedAlloc((nuint)(_blockSize * _idxKeyDim * sizeof(float)), 64);
        _dead = (float*)NativeMemory.AlignedAlloc((nuint)(_idxKeyDim * sizeof(float)), 64);
        _pooled = (float*)NativeMemory.AlignedAlloc((nuint)(_maxBlocks * _idxKeyDim * sizeof(float)), 64);
        _qIdx = (float*)NativeMemory.AlignedAlloc((nuint)(_idxHeads * _idxKeyDim * sizeof(float)), 64);
        _blockScores = (float*)NativeMemory.AlignedAlloc((nuint)(_maxBlocks * sizeof(float)), 64);
        _scores = (float*)NativeMemory.AlignedAlloc((nuint)(_maxSeqLen * sizeof(float)), 64);

        int heapCap = Math.Min(_maxSeqLen, MaxSelectionWidth);
        _heapScores = (float*)NativeMemory.AlignedAlloc((nuint)(heapCap * sizeof(float)), 64);
        _heapIndices = (int*)NativeMemory.AlignedAlloc((nuint)(heapCap * sizeof(int)), 64);
        _selectedIds = (int*)NativeMemory.AlignedAlloc((nuint)(_maxSeqLen * sizeof(int)), 64);

        Reset();
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMemory.Clear(_tail, (nuint)(_blockSize * _idxKeyDim * sizeof(float)));
        NativeMemory.Clear(_dead, (nuint)(_idxKeyDim * sizeof(float)));
        NativeMemory.Clear(_pooled, (nuint)(_maxBlocks * _idxKeyDim * sizeof(float)));
        _lastNkv = 0;
    }

    public int TailFloatCount => _blockSize * _idxKeyDim;

    public void SaveTail(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int needed = _blockSize * _idxKeyDim;
        if (destination.Length < needed)
            throw new ArgumentException($"Destination too small ({destination.Length} < {needed}).", nameof(destination));
        new ReadOnlySpan<float>(_tail, needed).CopyTo(destination);
    }

    public void RestoreTail(ReadOnlySpan<float> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int needed = _blockSize * _idxKeyDim;
        if (source.Length < needed)
            throw new ArgumentException($"Source too small ({source.Length} < {needed}).", nameof(source));
        source.Slice(0, needed).CopyTo(new Span<float>(_tail, needed));
    }

    public void SaveState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < StateSnapshotBytes)
            throw new ArgumentException($"Destination too small ({destination.Length} < {StateSnapshotBytes}).", nameof(destination));

        ulong tailBytes = (ulong)_blockSize * (ulong)_idxKeyDim * sizeof(float);
        ulong deadBytes = (ulong)_idxKeyDim * sizeof(float);
        fixed (byte* pDst = destination)
        {
            Buffer.MemoryCopy(_tail, pDst, tailBytes, tailBytes);
            Buffer.MemoryCopy(_dead, pDst + tailBytes, deadBytes, deadBytes);
        }
    }

    public void RestoreState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length < StateSnapshotBytes)
            throw new ArgumentException($"Source too small ({source.Length} < {StateSnapshotBytes}).", nameof(source));

        ulong tailBytes = (ulong)_blockSize * (ulong)_idxKeyDim * sizeof(float);
        ulong deadBytes = (ulong)_idxKeyDim * sizeof(float);
        fixed (byte* pSrc = source)
        {
            Buffer.MemoryCopy(pSrc, _tail, tailBytes, tailBytes);
            Buffer.MemoryCopy(pSrc + tailBytes, _dead, deadBytes, deadBytes);
        }
    }

    /// <summary>
    /// Records the indexer key at position <paramref name="pos"/>, updates the pooled block key when a
    /// block of size <see cref="BlockSize"/> completes, and selects the top-K causal KV cell indices sorted ascending.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void RecordKeyAndSelect(
        int pos,
        float* mixedIn,
        in QsaIndexerWeights w,
        out int* selectedIds,
        out int selectedCount,
        float* blockBias = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)pos >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(pos), $"Position {pos} exceeds maxSeqLen {_maxSeqLen}.");

        int slotInBlock = pos % _blockSize;
        float* kRaw = _tail + (long)slotInBlock * _idxKeyDim;
        QuantKernels.MatVecMul(w.KProjType, w.KProjWeight, mixedIn, kRaw, _dim, _idxKeyDim);

        if (pos == 0)
        {
            QuantKernels.RMSNorm(kRaw, w.KNormWeight, _dead, _idxKeyDim, _rmsEps);
        }

        int nKv = pos + 1;
        int nBid = nKv / _blockSize;

        if (slotInBlock == _blockSize - 1)
        {
            int blockIdx = pos / _blockSize;
            float* pooledBlock = _pooled + (long)blockIdx * _idxKeyDim;
            float invBlock = 1.0f / _blockSize;

            for (int d = 0; d < _idxKeyDim; d++)
            {
                float sum = 0f;
                for (int r = 0; r < _blockSize; r++)
                {
                    sum += _tail[(long)r * _idxKeyDim + d];
                }
                pooledBlock[d] = sum * invBlock;
            }

            QuantKernels.RMSNorm(pooledBlock, w.KNormWeight, pooledBlock, _idxKeyDim, _rmsEps);
            QuantKernels.RoPE(pooledBlock, null, 1, 0, _idxKeyDim, blockIdx * _blockSize, _ropeFreqBase, ropeDim: _rotaryDim);
        }

        ulong keyVecBytes = (ulong)_idxKeyDim * sizeof(float);
        Buffer.MemoryCopy(_dead, _pooled + (long)nBid * _idxKeyDim, keyVecBytes, keyVecBytes);

        int width = Math.Min(nKv, MaxSelectionWidth);
        if (nKv <= width && blockBias == null)
        {
            for (int j = 0; j < nKv; j++)
            {
                _selectedIds[j] = j;
            }
            _lastNkv = 0;
            selectedIds = _selectedIds;
            selectedCount = nKv;
            return;
        }

        // Compute indexer query projections + QNorm + NeOX RoPE at pos
        QuantKernels.MatVecMul(w.QProjType, w.QProjWeight, mixedIn, _qIdx, _dim, _idxHeads * _idxKeyDim);
        for (int h = 0; h < _idxHeads; h++)
        {
            float* qh = _qIdx + (long)h * _idxKeyDim;
            QuantKernels.RMSNorm(qh, w.QNormWeight, qh, _idxKeyDim, _rmsEps);
        }
        QuantKernels.RoPE(_qIdx, null, _idxHeads, 0, _idxKeyDim, pos, _ropeFreqBase, ropeDim: _rotaryDim);

        // Score every block 0..nBid: bias[cb] + sum_h ReLU(dot(pooled[cb], q_idx[h]))
        for (int cb = 0; cb <= nBid; cb++)
        {
            float score = blockBias != null ? blockBias[cb] : 0f;
            float* pKey = _pooled + (long)cb * _idxKeyDim;
            for (int h = 0; h < _idxHeads; h++)
            {
                float dot = QuantKernels.VecDotF32(pKey, _qIdx + (long)h * _idxKeyDim, _idxKeyDim);
                if (dot > 0f)
                {
                    score += dot;
                }
            }
            _blockScores[cb] = score;
        }

        int tailStart = nBid * _blockSize;
        for (int j = 0; j < nKv; j++)
        {
            int cb = j / _blockSize;
            if (cb > nBid) cb = nBid;
            _scores[j] = _blockScores[cb] + (j >= tailStart ? 1e9f : 0f);
        }
        _lastNkv = nKv;

        if (nKv <= width)
        {
            for (int j = 0; j < nKv; j++)
            {
                _selectedIds[j] = j;
            }
            selectedIds = _selectedIds;
            selectedCount = nKv;
            return;
        }

        // Maintain min-heap of size width where equal scores keep the smaller cell index.
        for (int j = 0; j < width; j++)
        {
            _heapScores[j] = _scores[j];
            _heapIndices[j] = j;
        }
        for (int i = (width / 2) - 1; i >= 0; i--)
        {
            SiftDown(i, width);
        }

        for (int j = width; j < nKv; j++)
        {
            float sc = _scores[j];
            if (sc > _heapScores[0])
            {
                _heapScores[0] = sc;
                _heapIndices[0] = j;
                SiftDown(0, width);
            }
        }

        for (int i = 0; i < width; i++)
        {
            _selectedIds[i] = _heapIndices[i];
        }
        new Span<int>(_selectedIds, width).Sort();

        selectedIds = _selectedIds;
        selectedCount = width;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void SiftDown(int idx, int size)
    {
        float curScore = _heapScores[idx];
        int curIndex = _heapIndices[idx];

        while (true)
        {
            int left = (idx * 2) + 1;
            if (left >= size) break;

            int smallest = left;
            int right = left + 1;
            if (right < size && IsHeapWorse(_heapScores[right], _heapIndices[right], _heapScores[left], _heapIndices[left]))
            {
                smallest = right;
            }

            if (!IsHeapWorse(_heapScores[smallest], _heapIndices[smallest], curScore, curIndex))
            {
                break;
            }

            _heapScores[idx] = _heapScores[smallest];
            _heapIndices[idx] = _heapIndices[smallest];
            idx = smallest;
        }

        _heapScores[idx] = curScore;
        _heapIndices[idx] = curIndex;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsHeapWorse(float aScore, int aIdx, float bScore, int bIdx)
    {
        return aScore < bScore || (aScore == bScore && aIdx > bIdx);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_tail != null) { NativeMemory.AlignedFree(_tail); _tail = null; }
        if (_dead != null) { NativeMemory.AlignedFree(_dead); _dead = null; }
        if (_pooled != null) { NativeMemory.AlignedFree(_pooled); _pooled = null; }
        if (_qIdx != null) { NativeMemory.AlignedFree(_qIdx); _qIdx = null; }
        if (_blockScores != null) { NativeMemory.AlignedFree(_blockScores); _blockScores = null; }
        if (_scores != null) { NativeMemory.AlignedFree(_scores); _scores = null; }
        if (_heapScores != null) { NativeMemory.AlignedFree(_heapScores); _heapScores = null; }
        if (_heapIndices != null) { NativeMemory.AlignedFree(_heapIndices); _heapIndices = null; }
        if (_selectedIds != null) { NativeMemory.AlignedFree(_selectedIds); _selectedIds = null; }
    }
}
