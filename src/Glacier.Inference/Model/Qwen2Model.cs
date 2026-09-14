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
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }

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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttentionToken(int stageLayer, int modelLayer, int pos, float* qHeadBase, float* outHeadBase, KVCache kvCache)
    {
        Parallel.For(0, _nHeads, h =>
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
            float* outHead = outHeadBase + h * _headDim;
            for (int d = 0; d < _headDim; d++) outHead[d] = 0f;

            for (int t = 0; t <= pos; t++)
            {
                float* vPast = kvCache.GetValuePtr(stageLayer, hKv, t);
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

