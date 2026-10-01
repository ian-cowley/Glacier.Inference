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
/// Dedicated high-performance CPU inference runtime for Qwen 3.5 / 3.6 Hybrid Models.
/// Implements Gated DeltaNet (linear attention SSM) with recurrent associative memory
/// interleaved with full quadratic multi-head attention (e.g. 3:1 ratio via full_attention_interval).
/// </summary>
public sealed unsafe class Qwen3HybridModel : CpuModelBase
{
    private readonly SsmStateCache _ssmCache;
    private readonly int _ssmConvKernel;
    private readonly int _ssmStateDim;
    private readonly int _ssmGroupCount;
    private readonly int _ssmHeads;
    private readonly int _ssmInnerSize;
    private readonly int _gdnConvChannels;
    private readonly int _groupSize;
    private readonly float _attnScale;

    // Preallocated GDN scratch buffers (single token)
    private float* _gdnQkv;
    private float* _gdnConvOut;
    private float* _gdnZ;
    private float* _gdnA;
    private float* _gdnB;
    private float* _gdnY;
    private float* _gdnYSums;

    // Preallocated standard attention scratch buffers for interleaved full-attention layers
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProj;
    private float* _headScores;

    // Batched standard attention scratch buffers
    private float* _qBatch;
    private float* _kBatch;
    private float* _vBatch;
    private float* _attnOutBatch;
    private float* _attnOutSumBatch;
    private float* _attnProjBatch;

    public SsmStateCache SsmCache => _ssmCache;

    public Qwen3HybridModel(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
        : base(weights, maxSeqLen, startLayer, layerCount, isLastStage)
    {
        _ssmConvKernel = weights.Gguf.SsmConvKernel;
        _ssmStateDim = weights.Gguf.SsmStateSize;
        _ssmGroupCount = weights.Gguf.SsmGroupCount;
        _ssmHeads = weights.Gguf.SsmTimeStepRank;
        _ssmInnerSize = weights.Gguf.SsmInnerSize;
        _gdnConvChannels = (_ssmGroupCount * 2 + _ssmHeads) * _ssmStateDim;

        _groupSize = _nHeadsKv > 0 ? _nHeads / _nHeadsKv : 1;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);

        // 1. Allocate unmanaged recurrent state cache for Gated DeltaNet
        _ssmCache = new SsmStateCache(
            weights.BlockCount,
            _ssmConvKernel,
            _gdnConvChannels,
            _ssmHeads,
            _ssmStateDim);

        // 2. Allocate GDN scratch buffers
        _gdnQkv = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnConvOut = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnZ = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        _gdnA = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnB = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnY = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        int yChunks = (_ssmInnerSize + 31) / 32;
        _gdnYSums = (float*)NativeMemory.AllocZeroed((nuint)(yChunks * sizeof(float)));

        // 3. Allocate standard attention buffers for interleaved full-attention layers
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int maxAttnOut = Math.Max(qDim, _nHeads * _vHeadDim);
        int attnOutChunks = (_dim + 31) / 32;

        _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(maxAttnOut * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(attnOutChunks * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        long scoreBufferSize = (long)_nHeads * _maxSeqLen * sizeof(float);
        _headScores = (float*)NativeMemory.AllocZeroed((nuint)scoreBufferSize);

        _qBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qDim * sizeof(float)));
        _kBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * kvDim * sizeof(float)));
        _vBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
        _attnOutSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * attnOutChunks * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardAttention(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsGdn)
        {
            ForwardGdnLayer(stageLayer, modelLayer, x);
        }
        else
        {
            ForwardStandardAttentionLayer(stageLayer, modelLayer, pos, x, kvCache);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayer(int stageLayer, int modelLayer, float* x)
    {
        ForwardGdnLayerWithNorm(stageLayer, modelLayer, x, _normX, _normXSums);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayerWithNorm(int stageLayer, int modelLayer, float* x, float* normX, float* normXSums)
    {
        var layer = _weights.Layers[modelLayer];

        // 1. Compute linear projections from normalized x
        QuantKernels.MatVecMul(layer.QkvType, layer.QkvWeight, normX, _gdnQkv, _dim, _gdnConvChannels, normXSums);
        QuantKernels.MatVecMul(layer.AttnGateType, layer.AttnGateWeight, normX, _gdnZ, _dim, _ssmInnerSize, normXSums);
        QuantKernels.MatVecMul(GgufType.F32, (byte*)layer.SsmAlphaWeight, normX, _gdnA, _dim, _ssmHeads, normXSums);
        QuantKernels.MatVecMul(GgufType.F32, (byte*)layer.SsmBetaWeight, normX, _gdnB, _dim, _ssmHeads, normXSums);

        // 2. Depthwise 1D Causal Convolution with SiLU activation
        float* convState = _ssmCache.GetConvState(modelLayer);
        int convHistLen = _ssmConvKernel - 1; // 3
        for (int c = 0; c < _gdnConvChannels; c++)
        {
            float* w = layer.SsmConv1dWeight + c * _ssmConvKernel;
            float* h = convState + c * convHistLen;

            float acc = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + _gdnQkv[c] * w[3];

            h[0] = h[1];
            h[1] = h[2];
            h[2] = _gdnQkv[c];

            float s = 1.0f / (1.0f + MathF.Exp(-acc));
            _gdnConvOut[c] = acc * s;
        }

        // 3. Partition channels into Q, K, V
        int qLen = _ssmGroupCount * _ssmStateDim;
        int kLen = _ssmGroupCount * _ssmStateDim;
        int vLen = _ssmHeads * _ssmStateDim;

        float* qPtr = _gdnConvOut;
        float* kPtr = _gdnConvOut + qLen;
        float* vPtr = _gdnConvOut + qLen + kLen;

        // 4. L2 Normalization of Q and K per group
        float qkScale = 1.0f / MathF.Sqrt(_ssmStateDim);
        for (int g = 0; g < _ssmGroupCount; g++)
        {
            float* qg = qPtr + g * _ssmStateDim;
            float* kg = kPtr + g * _ssmStateDim;

            float sumSqQ = 0f;
            float sumSqK = 0f;
            for (int d = 0; d < _ssmStateDim; d++)
            {
                sumSqQ += qg[d] * qg[d];
                sumSqK += kg[d] * kg[d];
            }

            float invNormQ = sumSqQ > 1e-12f ? (1.0f / MathF.Sqrt(sumSqQ)) * qkScale : 0f;
            float invNormK = sumSqK > 1e-12f ? (1.0f / MathF.Sqrt(sumSqK)) : 0f;

            for (int d = 0; d < _ssmStateDim; d++)
            {
                qg[d] *= invNormQ;
                kg[d] *= invNormK;
            }
        }

        // 5. Recurrent DeltaNet associative state update across all heads
        int headsPerGroup = _ssmHeads / _ssmGroupCount; // 3
        int dState = _ssmStateDim; // 128

        Parallel.For(0, _ssmHeads, h =>
        {
            int g = h / headsPerGroup;
            float* qHead = qPtr + g * dState;
            float* kHead = kPtr + g * dState;
            float* vHead = vPtr + h * dState;
            float* sMat = _ssmCache.GetRecurrentState(modelLayer, h);

            float aVal = _gdnA[h] + (layer.SsmDtBias != null ? layer.SsmDtBias[h] : 0f);
            float dt = MathF.Log(1.0f + MathF.Exp(aVal));
            float aLog = layer.SsmAWeight != null ? layer.SsmAWeight[h] : 0f;
            float gVal = -MathF.Exp(aLog) * dt;
            float decay = MathF.Exp(gVal);

            float bVal = _gdnB[h];
            float beta = 1.0f / (1.0f + MathF.Exp(-bVal));

            // State update: delta = (v - S_{t-1} k) * beta
            float* delta = stackalloc float[dState];
            for (int i = 0; i < dState; i++)
            {
                float kvMem = 0f;
                float* sRow = sMat + i * dState;
                for (int j = 0; j < dState; j++)
                {
                    kvMem += sRow[j] * kHead[j];
                }
                delta[i] = (vHead[i] - kvMem) * beta;
            }

            // S_t = decay * S_{t-1} + delta * k^T
            for (int i = 0; i < dState; i++)
            {
                float* sRow = sMat + i * dState;
                float dVal = delta[i];
                for (int j = 0; j < dState; j++)
                {
                    sRow[j] = decay * sRow[j] + dVal * kHead[j];
                }
            }

            // y_h = S_t * q
            float* yHead = _gdnY + h * dState;
            for (int i = 0; i < dState; i++)
            {
                float* sRow = sMat + i * dState;
                float dot = 0f;
                for (int j = 0; j < dState; j++)
                {
                    dot += sRow[j] * qHead[j];
                }
                yHead[i] = dot;
            }

            // Head RMSNorm
            if (layer.SsmNormWeight != null)
            {
                QuantKernels.RMSNorm(yHead, layer.SsmNormWeight, yHead, dState, _weights.RmsNormEps);
            }
        });

        // 6. Gating with SiLU on z: y = y * silu(z)
        for (int i = 0; i < _ssmInnerSize; i++)
        {
            float zVal = _gdnZ[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-zVal));
            _gdnY[i] *= (zVal * sig);
        }

        // 7. Output projection back to residual stream
        int yChunks = (_ssmInnerSize + 31) / 32;
        QuantKernels.ComputeBlockSums32(_gdnY, _gdnYSums, _ssmInnerSize);
        QuantKernels.MatVecMul(layer.SsmOutType, layer.SsmOutWeight, _gdnY, _normX, _ssmInnerSize, _dim, _gdnYSums);

        // Residual connection: x = x + normX (projected out)
        AddVector(x, _normX, _dim);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardStandardAttentionLayer(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
        QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
        QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

        if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
        if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
        if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);

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

        QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight);
        kvCache?.Store(stageLayer, pos, _k, _v);

        if (kvCache != null)
        {
            ComputeStandardAttentionToken(stageLayer, modelLayer, pos, _q, _attnOut, kvCache);
        }

        QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
        if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

        AddVector(x, _attnProj, _dim);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeStandardAttentionToken(int stageLayer, int modelLayer, int pos, float* qHeadBase, float* outHeadBase, KVCache kvCache)
    {
        for (int h = 0; h < _nHeads; h++)
        {
            int hKv = h / _groupSize;
            float* qHead = qHeadBase + h * _headDim;
            float* scores = _headScores + (long)h * _maxSeqLen;

            for (int t = 0; t <= pos; t++)
            {
                float* kPast = kvCache.GetKeyPtr(stageLayer, hKv, t);
                float dot = QuantKernels.VecDotF32(qHead, kPast, _headDim);
                scores[t] = dot * _attnScale;
            }

            var layerWeights = _weights.Layers[modelLayer];
            float? sinkLogit = layerWeights.AttnSinksWeight != null ? (float?)layerWeights.AttnSinksWeight[h] : null;
            QuantKernels.Softmax(scores, pos + 1, sinkLogit);

            float* outHead = outHeadBase + h * _vHeadDim;
            for (int d = 0; d < _vHeadDim; d++) outHead[d] = 0f;

            for (int t = 0; t <= pos; t++)
            {
                float* vPast = kvCache.GetValuePtr(stageLayer, hKv, t);
                float w = scores[t];
                for (int d = 0; d < _vHeadDim; d++)
                {
                    outHead[d] += w * vPast[d];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardBatchChunkAttention(int stageLayer, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        int normXChunks = _dim / 32;

        if (layer.IsGdn)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                float* normXSumT = _normXSumBatch + t * normXChunks;
                ForwardGdnLayerWithNorm(stageLayer, modelLayer, xt, normXt, normXSumT);
            }
        }
        else
        {
            int qDim = _nHeads * _headDim;
            int kvDim = _nHeadsKv * _headDim;

            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
            QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumBatch);
            QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumBatch);

            if (layer.QBias != null || layer.KBias != null || layer.VBias != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    if (layer.QBias != null) AddVector(_qBatch + t * qDim, layer.QBias, qDim);
                    if (layer.KBias != null) AddVector(_kBatch + t * kvDim, layer.KBias, kvDim);
                    if (layer.VBias != null) AddVector(_vBatch + t * kvDim, layer.VBias, kvDim);
                }
            }

            for (int t = 0; t < batchSize; t++)
            {
                int pos = chunkStartPos + t;
                float* q = _qBatch + t * qDim;
                float* k = _kBatch + t * kvDim;
                float* v = _vBatch + t * kvDim;

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

                QuantKernels.RoPE(q, k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight);
                kvCache?.Store(stageLayer, pos, k, v);

                if (kvCache != null)
                {
                    ComputeStandardAttentionToken(stageLayer, modelLayer, pos, q, _attnOutBatch + t * qDim, kvCache);
                }
            }

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

            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
            }
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            _ssmCache?.Dispose();

            FreeIfAllocated(ref _gdnQkv);
            FreeIfAllocated(ref _gdnConvOut);
            FreeIfAllocated(ref _gdnZ);
            FreeIfAllocated(ref _gdnA);
            FreeIfAllocated(ref _gdnB);
            FreeIfAllocated(ref _gdnY);
            FreeIfAllocated(ref _gdnYSums);

            FreeIfAllocated(ref _q);
            FreeIfAllocated(ref _k);
            FreeIfAllocated(ref _v);
            FreeIfAllocated(ref _attnOut);
            FreeIfAllocated(ref _attnOutSums);
            FreeIfAllocated(ref _attnProj);
            FreeIfAllocated(ref _headScores);

            FreeIfAllocated(ref _qBatch);
            FreeIfAllocated(ref _kBatch);
            FreeIfAllocated(ref _vBatch);
            FreeIfAllocated(ref _attnOutBatch);
            FreeIfAllocated(ref _attnOutSumBatch);
            FreeIfAllocated(ref _attnProjBatch);

            base.Dispose(disposing);
        }
    }
}
