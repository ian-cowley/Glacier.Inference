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
/// Also supports Llama 3, Mistral, Phi-4 (fused QKV), and DeepSeek (MLA) architectures.
/// </summary>
public sealed unsafe partial class Qwen2Model : CpuModelBase
{
    private readonly int _qkNopeDim;
    private readonly int _qkRopeDim;
    private readonly int _kvLoraRank;
    private float* _yarnInvFreq;
    private readonly float _yarnMscale;
    private readonly int _groupSize;
    private readonly float _attnScale;

    // Preallocated unmanaged attention scratch buffers (single token)
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProj;
    private float* _headScores;
    private float* _qkvFused;

    // Preallocated MLA scratch buffers (single token)
    private float* _compressedKv;
    private float* _cKvNorm;
    private float* _cKvSums;
    private float* _decompressedKv;

    // Preallocated unmanaged attention scratch buffers (batched chunk prefill <= MaxBatchSize)
    private float* _qBatch;
    private float* _kBatch;
    private float* _vBatch;
    private float* _attnOutBatch;
    private float* _attnOutSumBatch;
    private float* _attnProjBatch;
    private float* _qkvBatch;

    // Preallocated MLA scratch buffers (batched)
    private float* _compressedKvBatch;
    private float* _decompressedKvBatch;

    public Qwen2Model(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
        : base(weights, maxSeqLen, startLayer, layerCount, isLastStage)
    {
        _qkNopeDim = weights.QkNopeHeadDim;
        _qkRopeDim = weights.RopeDimensionCount;
        _kvLoraRank = weights.KvLoraRank;
        _groupSize = _nHeadsKv > 0 ? _nHeads / _nHeadsKv : 1;

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int maxAttnOut = Math.Max(qDim, _nHeads * _vHeadDim);
        int attnOutChunks = (Math.Max(maxAttnOut, _dim) + 31) / 32;

        _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(maxAttnOut * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(attnOutChunks * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

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

        // Batch attention scratch buffers
        _qBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qDim * sizeof(float)));
        _kBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * kvDim * sizeof(float)));
        _vBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
        _attnOutSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * attnOutChunks * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));

        int qkvDim = qDim + 2 * kvDim;
        _qkvFused = (float*)NativeMemory.AllocZeroed((nuint)(qkvDim * sizeof(float)));
        _qkvBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qkvDim * sizeof(float)));

        _chunkAction = ExecuteAttentionChunk;
    }

    private struct AttentionContext
    {
        public int StageLayer;
        public int ModelLayer;
        public int Pos;
        public float* QHeadBase;
        public float* OutHeadBase;
        public KVCache KvCache;
        public int ChunkSize;
    }

    private AttentionContext _attContext;
    private readonly Action<int> _chunkAction;

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
        if (numChunks <= 1)
        {
            for (int h = 0; h < _nHeads; h++)
            {
                EvaluateSingleHeadAttention(stageLayer, modelLayer, pos, qHeadBase, outHeadBase, kvCache, h);
            }
            return;
        }

        int chunkSize = (_nHeads + numChunks - 1) / numChunks;

        _attContext.StageLayer = stageLayer;
        _attContext.ModelLayer = modelLayer;
        _attContext.Pos = pos;
        _attContext.QHeadBase = qHeadBase;
        _attContext.OutHeadBase = outHeadBase;
        _attContext.KvCache = kvCache;
        _attContext.ChunkSize = chunkSize;

        Parallel.For(0, numChunks, _chunkAction);
    }

    private void ExecuteAttentionChunk(int chunkIdx)
    {
        int startHead = chunkIdx * _attContext.ChunkSize;
        int endHead = Math.Min(startHead + _attContext.ChunkSize, _nHeads);

        for (int h = startHead; h < endHead; h++)
        {
            EvaluateSingleHeadAttention(
                _attContext.StageLayer,
                _attContext.ModelLayer,
                _attContext.Pos,
                _attContext.QHeadBase,
                _attContext.OutHeadBase,
                _attContext.KvCache,
                h);
        }
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

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            FreeIfAllocated(ref _q);
            FreeIfAllocated(ref _k);
            FreeIfAllocated(ref _v);
            FreeIfAllocated(ref _attnOut);
            FreeIfAllocated(ref _attnOutSums);
            FreeIfAllocated(ref _attnProj);
            FreeIfAllocated(ref _headScores);
            FreeIfAllocated(ref _qkvFused);

            FreeIfAllocated(ref _compressedKv);
            FreeIfAllocated(ref _cKvNorm);
            FreeIfAllocated(ref _cKvSums);
            FreeIfAllocated(ref _decompressedKv);
            FreeIfAllocated(ref _yarnInvFreq);

            FreeIfAllocated(ref _qBatch);
            FreeIfAllocated(ref _kBatch);
            FreeIfAllocated(ref _vBatch);
            FreeIfAllocated(ref _attnOutBatch);
            FreeIfAllocated(ref _attnOutSumBatch);
            FreeIfAllocated(ref _attnProjBatch);
            FreeIfAllocated(ref _qkvBatch);

            FreeIfAllocated(ref _compressedKvBatch);
            FreeIfAllocated(ref _decompressedKvBatch);

            base.Dispose(disposing);
        }
    }
}
