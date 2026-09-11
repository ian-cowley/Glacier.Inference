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

    // Preallocated unmanaged scratch buffers
    private float* _x;
    private float* _normX;
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnProj;
    private float* _gate;
    private float* _up;
    private float* _ffnAct;
    private float* _ffnOut;
    private float* _headScores;

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

        _x = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _normX = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _q = (float*)NativeMemory.AllocZeroed((nuint)(_nHeads * _headDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _gate = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
        _up = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
        _ffnAct = (float*)NativeMemory.AllocZeroed((nuint)(_ffnDim * sizeof(float)));
        _ffnOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        long scoreBufferSize = (long)_nHeads * _maxSeqLen * sizeof(float);
        _headScores = (float*)NativeMemory.AllocZeroed((nuint)scoreBufferSize);
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

        // 2. Transformer layers
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var layer = _weights.Layers[l];

            // Attention pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);

            // Q, K, V projections
            int qDim = _nHeads * _headDim;
            int kvDim = _nHeadsKv * _headDim;
            QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim);
            QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim);
            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim);

            // Add Q, K, V biases
            AddVector(_q, layer.QBias, qDim);
            AddVector(_k, layer.KBias, kvDim);
            AddVector(_v, layer.VBias, kvDim);

            // Rotary Position Embedding (RoPE)
            QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase);

            // Store in KV cache
            kvCache.Store(l, pos, _k, _v);

            // Multi-Head / Grouped Query Attention (GQA)
            ComputeAttention(l, pos, kvCache);

            // Attention output projection
            QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, _dim, _dim);

            // Residual connection: x = x + attnProj
            AddVector(_x, _attnProj, _dim);

            // FFN pre-norm
            QuantKernels.RMSNorm(_x, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);

            // SwiGLU FFN projections
            QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim);
            QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim);
            QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
            QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, _ffnOut, _ffnDim, _dim);

            // Residual connection: x = x + ffnOut
            AddVector(_x, _ffnOut, _dim);
        }

        if (computeLogits)
        {
            // Final RMSNorm
            QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);

            // LM Head output projection
            fixed (float* logitsPtr = logits)
            {
                QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, logitsPtr, _dim, _weights.VocabSize);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeAttention(int layer, int pos, KVCache kvCache)
    {
        Parallel.For(0, _nHeads, h =>
        {
            int hKv = h / _groupSize;
            float* qHead = _q + h * _headDim;
            float* scores = _headScores + (long)h * _maxSeqLen;

            // Score calculation Q * K^T
            for (int t = 0; t <= pos; t++)
            {
                float* kPast = kvCache.GetKeyPtr(layer, hKv, t);
                float dot = QuantKernels.VecDotF32(qHead, kPast, _headDim);
                scores[t] = dot * _attnScale;
            }

            // Softmax over 0..pos
            QuantKernels.Softmax(scores, pos + 1);

            // Value aggregation
            float* outHead = _attnOut + h * _headDim;
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
            FreeIfAllocated(ref _q);
            FreeIfAllocated(ref _k);
            FreeIfAllocated(ref _v);
            FreeIfAllocated(ref _attnOut);
            FreeIfAllocated(ref _attnProj);
            FreeIfAllocated(ref _gate);
            FreeIfAllocated(ref _up);
            FreeIfAllocated(ref _ffnAct);
            FreeIfAllocated(ref _ffnOut);
            FreeIfAllocated(ref _headScores);
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
