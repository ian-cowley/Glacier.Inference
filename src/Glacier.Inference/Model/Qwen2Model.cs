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
public sealed unsafe partial class Qwen2Model : IDisposable
{
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _headDim;
    private readonly int _vHeadDim;
    private readonly int _qkNopeDim;
    private readonly int _qkRopeDim;
    private readonly int _kvLoraRank;
    private float* _yarnInvFreq;
    private readonly float _yarnMscale;
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

    // Preallocated fused QKV & Gate/Up scratch buffers (single token)
    private float* _qkvFused;
    private float* _gateUpFused;

    // Preallocated MLA scratch buffers (single token)
    private float* _compressedKv;
    private float* _cKvNorm;
    private float* _cKvSums;
    private float* _decompressedKv;

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

    // Preallocated fused QKV & Gate/Up scratch buffers (batched)
    private float* _qkvBatch;
    private float* _gateUpBatch;

    // Preallocated MLA scratch buffers (batched)
    private float* _compressedKvBatch;
    private float* _decompressedKvBatch;

    private bool _disposed;

    public ModelWeights Weights => _weights;
    public int MaxSeqLen => _maxSeqLen;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }
    public LoraAdapterWeights? LoraWeights { get; set; }

    public Qwen2Model(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
    {
        _weights = weights;
        StartLayer = startLayer;
        LayerCount = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        IsLastStage = isLastStage;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _vHeadDim = weights.ValueDim;
        _qkNopeDim = weights.QkNopeHeadDim;
        _qkRopeDim = weights.RopeDimensionCount;
        _kvLoraRank = weights.KvLoraRank;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _maxSeqLen = maxSeqLen;

        int expFfn = weights.ExpertFeedForwardLength;
        int maxFfn = Math.Max(_ffnDim, Math.Max(expFfn, expFfn * 2));
        if (maxFfn == 0) maxFfn = _ffnDim;

        int qDim = _nHeads * _headDim;
        int maxAttnOut = Math.Max(_dim, Math.Max(qDim, _nHeads * _vHeadDim));
        int attnOutChunks = (maxAttnOut + 31) / 32;

        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normXSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _q = (float*)NativeMemory.AllocZeroed((nuint)(_nHeads * _headDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _vHeadDim * sizeof(float)));
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

        // MLA buffers
        int kvMqaDim = _kvLoraRank + _qkRopeDim;
        int decompKvDim = _nHeads * (_qkNopeDim + _vHeadDim);
        int halfRope = _qkRopeDim / 2;
        if (_weights.IsMla)
        {
            _compressedKv = (float*)NativeMemory.AllocZeroed((nuint)(kvMqaDim * sizeof(float)));
            _cKvNorm = (float*)NativeMemory.AllocZeroed((nuint)(_kvLoraRank * sizeof(float)));
            _cKvSums = (float*)NativeMemory.AllocZeroed((nuint)(((_kvLoraRank + 31) / 32) * sizeof(float)));
            _decompressedKv = (float*)NativeMemory.AllocZeroed((nuint)(decompKvDim * sizeof(float)));

            _compressedKvBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * kvMqaDim * sizeof(float)));
            _decompressedKvBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * decompKvDim * sizeof(float)));

            _yarnInvFreq = (float*)NativeMemory.AllocZeroed((nuint)(halfRope * sizeof(float)));
            if (weights.Gguf.RopeScalingType.Equals("yarn", StringComparison.OrdinalIgnoreCase))
            {
                QuantKernels.PrecomputeYarnFrequencies(
                    new Span<float>(_yarnInvFreq, halfRope),
                    _qkRopeDim,
                    weights.RopeFreqBase,
                    weights.Gguf.RopeScalingFactor > 0 ? weights.Gguf.RopeScalingFactor : 1.0f,
                    weights.Gguf.RopeScalingOriginalContextLength > 0 ? weights.Gguf.RopeScalingOriginalContextLength : 4096);

                float logMul = weights.Gguf.RopeScalingYarnLogMultiplier > 0 ? weights.Gguf.RopeScalingYarnLogMultiplier : 0.0707f;
                float factor = weights.Gguf.RopeScalingFactor > 0 ? weights.Gguf.RopeScalingFactor : 1.0f;
                float mscaleAllDim = factor > 1.0f ? (logMul * MathF.Log(factor) + 1.0f) : 1.0f;
                float mscaleBase = factor > 1.0f ? (0.1f * MathF.Log(factor) + 1.0f) : 1.0f;
                _yarnMscale = mscaleBase / mscaleAllDim;
                _attnScale = (1.0f / MathF.Sqrt(_headDim)) * mscaleAllDim * mscaleAllDim;
            }
            else
            {
                _yarnMscale = 1.0f;
                _attnScale = 1.0f / MathF.Sqrt(_headDim);
            }
        }
        else
        {
            _yarnMscale = 1.0f;
            _attnScale = 1.0f / MathF.Sqrt(_headDim);
        }

        // Batch scratch buffers
        _xBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _normXSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * (_dim / 32) * sizeof(float)));
        _qBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeads * _headDim * sizeof(float)));
        _kBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
        _vBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
        _attnOutSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * attnOutChunks * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _gateBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _upBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxFfn * sizeof(float)));
        _ffnActSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * ((maxFfn + 31) / 32) * sizeof(float)));
        _ffnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));

        int qkvDim = (_nHeads * _headDim) + (_nHeadsKv * _headDim) + (_nHeadsKv * _vHeadDim);
        int gateUpDim = 2 * maxFfn;
        _qkvFused = (float*)NativeMemory.AllocZeroed((nuint)(qkvDim * sizeof(float)));
        _gateUpFused = (float*)NativeMemory.AllocZeroed((nuint)(gateUpDim * sizeof(float)));
        _qkvBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qkvDim * sizeof(float)));
        _gateUpBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * gateUpDim * sizeof(float)));
    }

    private const int ParallelAttentionSeqThreshold = 256;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttentionToken(int stageLayer, int modelLayer, int pos, float* qHeadBase, float* outHeadBase, KVCache kvCache)
    {
        if (pos < ParallelAttentionSeqThreshold)
        {
            for (int h = 0; h < _nHeads; h++)
            {
                EvaluateSingleHeadAttention(stageLayer, modelLayer, pos, qHeadBase, outHeadBase, kvCache, h);
            }
            return;
        }

        int numChunks = Math.Min(Environment.ProcessorCount, 4);
        int chunkSize = (_nHeads + numChunks - 1) / numChunks;

        Parallel.For(0, numChunks, chunkIdx =>
        {
            int startHead = chunkIdx * chunkSize;
            int endHead = Math.Min(startHead + chunkSize, _nHeads);

            for (int h = startHead; h < endHead; h++)
            {
                EvaluateSingleHeadAttention(stageLayer, modelLayer, pos, qHeadBase, outHeadBase, kvCache, h);
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private void EvaluateSingleHeadAttention(
        int stageLayer,
        int modelLayer,
        int pos,
        float* qHeadBase,
        float* outHeadBase,
        KVCache kvCache,
        int h)
    {
        int hKv = h / _groupSize;
        float* qHead = qHeadBase + h * _headDim;
        float* scores = _headScores + (long)h * _maxSeqLen;

        // Score calculation Q * K^T
        for (int t = 0; t <= pos; t++)
        {
            float* kPast = kvCache.GetKeyPtr(stageLayer, hKv, t);
            float dot = QuantKernels.VecDotF32(qHead, kPast, _headDim);
            scores[t] = dot * _attnScale;
        }

        // Softmax over 0..pos with optional attention sink logit
        var layerWeights = _weights.Layers[modelLayer];
        float? sinkLogit = layerWeights.AttnSinksWeight != null ? (float?)layerWeights.AttnSinksWeight[h] : null;
        QuantKernels.Softmax(scores, pos + 1, sinkLogit);

        // Value aggregation
        float* outHead = outHeadBase + h * _vHeadDim;
        for (int d = 0; d < _vHeadDim; d++) outHead[d] = 0f;

        for (int t = 0; t <= pos; t++)
        {
            float* vPast = kvCache.GetValuePtr(stageLayer, hKv, t);
            float w = scores[t];

            if (Vector512.IsHardwareAccelerated && _vHeadDim >= 16)
            {
                var vw = Vector512.Create(w);
                int d = 0;
                for (; d <= _vHeadDim - 16; d += 16)
                {
                    var vo = Vector512.Load(outHead + d);
                    var vv = Vector512.Load(vPast + d);
                    vo = Vector512.FusedMultiplyAdd(vw, vv, vo);
                    vo.Store(outHead + d);
                }
                for (; d < _vHeadDim; d++) outHead[d] += w * vPast[d];
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                var vw = Vector256.Create(w);
                int d = 0;
                for (; d <= _vHeadDim - 8; d += 8)
                {
                    var vo = Vector256.Load(outHead + d);
                    var vv = Vector256.Load(vPast + d);
                    vo += vw * vv;
                    vo.Store(outHead + d);
                }
                for (; d < _vHeadDim; d++) outHead[d] += w * vPast[d];
            }
            else
            {
                for (int d = 0; d < _vHeadDim; d++)
                {
                    outHead[d] += w * vPast[d];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttention(int stageLayer, int modelLayer, int pos, KVCache kvCache)
    {
        ComputeAttentionToken(stageLayer, modelLayer, pos, _q, _attnOut, kvCache);
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

            FreeIfAllocated(ref _qkvFused);
            FreeIfAllocated(ref _gateUpFused);
            FreeIfAllocated(ref _qkvBatch);
            FreeIfAllocated(ref _gateUpBatch);

            FreeIfAllocated(ref _compressedKv);
            FreeIfAllocated(ref _cKvNorm);
            FreeIfAllocated(ref _cKvSums);
            FreeIfAllocated(ref _decompressedKv);
            FreeIfAllocated(ref _yarnInvFreq);
            FreeIfAllocated(ref _compressedKvBatch);
            FreeIfAllocated(ref _decompressedKvBatch);

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

