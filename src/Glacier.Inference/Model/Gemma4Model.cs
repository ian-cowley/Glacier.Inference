namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Contiguous unmanaged Key-Value Cache tailored specifically for Gemma 4's
/// alternating Sliding Window Attention (SWA) and full-attention dense layers.
/// SWA layers: 8 KV heads x 256 head dimension = 2048 floats per token
/// Dense layers: 2 KV heads x 512 head dimension = 1024 floats per token
/// </summary>
public sealed unsafe class Gemma4KvCache : IDisposable
{
    private readonly int _layers;
    private readonly int _maxSeqLen;
    private readonly int[] _layerHeadsKv;
    private readonly int[] _layerHeadDims;
    private readonly long[] _layerOffsetsK;
    private readonly long[] _layerOffsetsV;

    private float* _kBuffer;
    private float* _vBuffer;
    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int Layers => _layers;

    public Gemma4KvCache(int layers, int maxSeqLen, bool[]? swaPattern, int[]? headsKvPattern, int defaultHeadDim)
    {
        _layers = layers;
        _maxSeqLen = maxSeqLen;

        _layerHeadsKv = new int[layers];
        _layerHeadDims = new int[layers];
        _layerOffsetsK = new long[layers];
        _layerOffsetsV = new long[layers];

        long runningOffset = 0;
        for (int l = 0; l < layers; l++)
        {
            bool isSwa = swaPattern != null && l < swaPattern.Length ? swaPattern[l] : ((l + 1) % 6 != 0);
            int headsKv = headsKvPattern != null && l < headsKvPattern.Length ? headsKvPattern[l] : (isSwa ? 8 : 2);
            int headDim = isSwa ? 256 : (defaultHeadDim > 0 ? defaultHeadDim : 512);

            _layerHeadsKv[l] = headsKv;
            _layerHeadDims[l] = headDim;
            _layerOffsetsK[l] = runningOffset;
            _layerOffsetsV[l] = runningOffset;

            long layerFloats = (long)headsKv * maxSeqLen * headDim;
            runningOffset += layerFloats;
        }

        long totalBytes = runningOffset * sizeof(float);
        _kBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytes);
        _vBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int layer, int headKv, int pos)
    {
        int headDim = _layerHeadDims[layer];
        long offset = _layerOffsetsK[layer] + ((long)headKv * _maxSeqLen + pos) * headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValPtr(int layer, int headKv, int pos)
    {
        int headDim = _layerHeadDims[layer];
        long offset = _layerOffsetsV[layer] + ((long)headKv * _maxSeqLen + pos) * headDim;
        return _vBuffer + offset;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_kBuffer != null)
            {
                NativeMemory.Free(_kBuffer);
                _kBuffer = null;
            }
            if (_vBuffer != null)
            {
                NativeMemory.Free(_vBuffer);
                _vBuffer = null;
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// Dedicated high-performance CPU inference runtime for Google Gemma 4 models.
/// Implements interleaved Sliding Window Attention (ISWA), dual shared+MoE FFN,
/// GeLU-GLU activations, and output logit softcapping with zero heap allocations on hot paths.
/// </summary>
public sealed unsafe class Gemma4Model : CpuModelBase
{
    private readonly Gemma4KvCache _gemmaKvCache;
    private readonly int _sharedFfnDim;
    private readonly int _expertFfnDim;
    private readonly int _expertCount;
    private readonly int _expertUsedCount;
    private readonly int _slidingWindow;
    private readonly float _finalLogitSoftcapping;
    private readonly float _embeddingScale;

    // Attention scratch buffers
    private float* _attnIn;
    private float* _attnNormSums;
    private float* _q;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProjOut;
    private float* _attnOutResidual;
    private float* _headScores;

    // Shared MLP scratch buffers
    private float* _mlpNorm;
    private float* _mlpNormSums;
    private float* _sharedGate;
    private float* _sharedUp;
    private float* _mlpAct;
    private float* _mlpActSums;
    private float* _mlpOut;

    // MoE scratch buffers
    private readonly int _maxTopK;
    private float* _moeNorm;
    private float* _moeNormSums;
    private float* _routerIn;
    private float* _routerLogits;
    private float* _expertScratchFused;
    private float* _expertScratchAct;
    private float* _expertScratchActSums;
    private float* _expertOutputs;
    private float* _moeOut;

    // Combined FFN scratch buffers
    private float* _combinedFfn;

    // Output scratch buffers
    private float* _outNorm;
    private float* _outNormSums;

    public Gemma4KvCache GemmaCache => _gemmaKvCache;

    public Gemma4Model(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
        : base(weights, maxSeqLen, startLayer, layerCount, isLastStage)
    {
        _sharedFfnDim = weights.FeedForwardLength; // 2112
        _expertFfnDim = weights.ExpertFeedForwardLength > 0 ? weights.ExpertFeedForwardLength : 704;
        _expertCount = weights.ExpertCount > 0 ? weights.ExpertCount : 128;
        _expertUsedCount = weights.ExpertUsedCount > 0 ? weights.ExpertUsedCount : 8;
        _maxTopK = Math.Max(16, _expertUsedCount);
        _slidingWindow = weights.SlidingWindow > 0 ? weights.SlidingWindow : 1024;
        _finalLogitSoftcapping = weights.FinalLogitSoftcapping;
        _embeddingScale = MathF.Sqrt(_dim);

        // Preallocate unmanaged attention buffers
        int maxQDim = 16 * 512; // 8192
        int maxKvDim = 8 * 256; // 2048

        _attnIn = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _attnNormSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _q = (float*)NativeMemory.AllocZeroed((nuint)(maxQDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(maxKvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(maxKvDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(maxQDim * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)((maxQDim / 32) * sizeof(float)));
        _attnProjOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _attnOutResidual = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _headScores = (float*)NativeMemory.AllocZeroed((nuint)(maxSeqLen * sizeof(float)));

        // Preallocate unmanaged shared MLP buffers
        _mlpNorm = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _mlpNormSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _sharedGate = (float*)NativeMemory.AllocZeroed((nuint)(_sharedFfnDim * sizeof(float)));
        _sharedUp = (float*)NativeMemory.AllocZeroed((nuint)(_sharedFfnDim * sizeof(float)));
        _mlpAct = (float*)NativeMemory.AllocZeroed((nuint)(_sharedFfnDim * sizeof(float)));
        _mlpActSums = (float*)NativeMemory.AllocZeroed((nuint)(((_sharedFfnDim + 31) / 32) * sizeof(float)));
        _mlpOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        // Preallocate unmanaged MoE buffers (sized for multithreaded concurrent expert execution)
        _moeNorm = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _moeNormSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));
        _routerIn = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _routerLogits = (float*)NativeMemory.AllocZeroed((nuint)(_expertCount * sizeof(float)));
        _expertScratchFused = (float*)NativeMemory.AllocZeroed((nuint)(_maxTopK * 2 * _expertFfnDim * sizeof(float)));
        _expertScratchAct = (float*)NativeMemory.AllocZeroed((nuint)(_maxTopK * _expertFfnDim * sizeof(float)));
        _expertScratchActSums = (float*)NativeMemory.AllocZeroed((nuint)(_maxTopK * (((_expertFfnDim + 31) / 32) * sizeof(float))));
        _expertOutputs = (float*)NativeMemory.AllocZeroed((nuint)(_maxTopK * _dim * sizeof(float)));
        _moeOut = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        // Combined FFN buffer
        _combinedFfn = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        // Output buffers
        _outNorm = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _outNormSums = (float*)NativeMemory.AllocZeroed((nuint)((_dim / 32) * sizeof(float)));

        // Dedicated Gemma4 KV cache
        _gemmaKvCache = new Gemma4KvCache(
            weights.BlockCount,
            maxSeqLen,
            weights.SlidingWindowPattern,
            weights.HeadCountKvPattern,
            weights.HeadDim);
    }

    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// </summary>
    public override void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        ForwardStage(token, pos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    public override void ForwardStage(
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

            // Gemma embeds are scaled by sqrt(dim)
            int d = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var vScale = Vector256.Create(_embeddingScale);
                int vecLimit = _dim - 8;
                for (; d <= vecLimit; d += 8)
                {
                    var v = Vector256.Load(_x + d);
                    (v * vScale).Store(_x + d);
                }
            }
            for (; d < _dim; d++)
            {
                _x[d] *= _embeddingScale;
            }
        }
        else
        {
            fixed (float* pInput = inputX)
            {
                Buffer.MemoryCopy(pInput, _x, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }

        int maxTopK = _expertUsedCount > 0 ? _expertUsedCount : 8;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 2. Transformer layers execution
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var layer = _weights.Layers[modelLayer];

            // Attention pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _attnIn, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_attnIn, _attnNormSums, _dim);

            // Forward Attention
            ForwardGemma4Attention(l, modelLayer, pos);

            // Post-attention norm and residual connection
            QuantKernels.RMSNorm(_attnProjOut, layer.AttnPostNormWeight, _attnProjOut, _dim, _weights.RmsNormEps);
            for (int i = 0; i < _dim; i++)
            {
                _attnOutResidual[i] = _x[i] + _attnProjOut[i];
            }

            // Forward Dual FFN (Shared MLP + Sparse MoE)
            ForwardGemma4DualFfn(modelLayer, selectedIndices, selectedWeights);

            // Post-FFN norm and residual connection
            QuantKernels.RMSNorm(_combinedFfn, layer.FfnPostNormWeight, _combinedFfn, _dim, _weights.RmsNormEps);
            for (int i = 0; i < _dim; i++)
            {
                _x[i] = _attnOutResidual[i] + _combinedFfn[i];
            }

            // Layer output scalar
            if (layer.LayerOutputScaleWeight != null)
            {
                float outScale = layer.LayerOutputScaleWeight[0];
                int i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vScale = Vector256.Create(outScale);
                    int vecLimit = _dim - 8;
                    for (; i <= vecLimit; i += 8)
                    {
                        var vx = Vector256.Load(_x + i);
                        (vx * vScale).Store(_x + i);
                    }
                }
                for (; i < _dim; i++)
                {
                    _x[i] *= outScale;
                }
            }
        }

        // 3. Final normalization and output logits
        if (IsLastStage)
        {
            if (computeLogits && logits.Length > 0)
            {
                QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _outNorm, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_outNorm, _outNormSums, _dim);

                fixed (float* pLogits = logits)
                {
                    QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _outNorm, pLogits, _dim, _weights.VocabSize, _outNormSums);

                    // Final logit softcapping (30.0 * tanh(logits / 30.0))
                    if (_finalLogitSoftcapping > 0f)
                    {
                        QuantKernels.SoftcapLogits(pLogits, _weights.VocabSize, _finalLogitSoftcapping);
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
        else if (outputX.Length >= _dim)
        {
            fixed (float* pOut = outputX)
            {
                Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGemma4Attention(int stageLayer, int modelLayer, int pos)
    {
        var layer = _weights.Layers[modelLayer];
        bool isSwa = layer.IsSwa;
        int headDim = layer.HeadDim;
        int headsKv = layer.HeadsKv;
        int nHeads = _nHeads;

        int totalQDim = nHeads * headDim;
        int totalKvDim = headsKv * headDim;

        // 1. Q projection
        QuantKernels.MatVecMul(layer.QType, layer.QWeight, _attnIn, _q, _dim, totalQDim, _attnNormSums);

        // Q-norm per head
        for (int h = 0; h < nHeads; h++)
        {
            QuantKernels.RMSNorm(_q + h * headDim, layer.AttnQNormWeight, _q + h * headDim, headDim, _weights.RmsNormEps);
        }

        // Q RoPE
        float freqBase = isSwa ? _weights.RopeFreqBaseSwa : _weights.RopeFreqBase;
        float* ropeFreqs = !isSwa ? _weights.RopeFreqsWeight : null;
        QuantKernels.RoPE(_q, null, nHeads, 0, headDim, pos, freqBase, ropeFreqs: ropeFreqs);

        // 2. K & V projections
        QuantKernels.MatVecMul(layer.KType, layer.KWeight, _attnIn, _k, _dim, totalKvDim, _attnNormSums);

        if (layer.VWeight != null)
        {
            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _attnIn, _v, _dim, totalKvDim, _attnNormSums);
        }
        else
        {
            // Dense layers reuse un-normalized K projection as V
            Buffer.MemoryCopy(_k, _v, (ulong)(totalKvDim * sizeof(float)), (ulong)(totalKvDim * sizeof(float)));
        }

        // K-norm per KV head
        for (int h = 0; h < headsKv; h++)
        {
            QuantKernels.RMSNorm(_k + h * headDim, layer.AttnKNormWeight, _k + h * headDim, headDim, _weights.RmsNormEps);
        }

        // V-norm per KV head (unweighted RMSNorm)
        for (int h = 0; h < headsKv; h++)
        {
            QuantKernels.RMSNorm(_v + h * headDim, null, _v + h * headDim, headDim, _weights.RmsNormEps);
        }

        // K RoPE
        QuantKernels.RoPE(null, _k, 0, headsKv, headDim, pos, freqBase, ropeFreqs: ropeFreqs);

        // 3. Write K and V to Gemma4 KV Cache
        for (int hKv = 0; hKv < headsKv; hKv++)
        {
            float* kDst = _gemmaKvCache.GetKeyPtr(modelLayer, hKv, pos);
            float* vDst = _gemmaKvCache.GetValPtr(modelLayer, hKv, pos);
            Buffer.MemoryCopy(_k + hKv * headDim, kDst, (ulong)(headDim * sizeof(float)), (ulong)(headDim * sizeof(float)));
            Buffer.MemoryCopy(_v + hKv * headDim, vDst, (ulong)(headDim * sizeof(float)), (ulong)(headDim * sizeof(float)));
        }

        // 4. Attention Context Calculation
        int windowStart = isSwa ? Math.Max(0, pos - _slidingWindow + 1) : 0;
        int seqLen = pos - windowStart + 1;
        int groupRatio = nHeads / headsKv;

        new Span<float>(_attnOut, totalQDim).Clear();

        for (int h = 0; h < nHeads; h++)
        {
            int kvHead = h / groupRatio;
            float* qHead = _q + h * headDim;
            float* attnOutHead = _attnOut + h * headDim;

            // Attention scores: dot(Q, K) with attention_scale = 1.0 (no sqrt(headDim) division)
            for (int p = windowStart; p <= pos; p++)
            {
                float* kPtr = _gemmaKvCache.GetKeyPtr(modelLayer, kvHead, p);
                _headScores[p - windowStart] = QuantKernels.VecDotF32(qHead, kPtr, headDim);
            }

            // Softmax over valid window positions
            QuantKernels.Softmax(_headScores, seqLen);

            // Accumulate V vectors weighted by attention probabilities
            for (int p = windowStart; p <= pos; p++)
            {
                float prob = _headScores[p - windowStart];
                if (prob == 0.0f) continue;
                float* vPtr = _gemmaKvCache.GetValPtr(modelLayer, kvHead, p);

                int d = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vProb = Vector256.Create(prob);
                    int vecLimit = headDim - 8;
                    for (; d <= vecLimit; d += 8)
                    {
                        var vAttn = Vector256.Load(attnOutHead + d);
                        var vVal = Vector256.Load(vPtr + d);
                        (vAttn + vVal * vProb).Store(attnOutHead + d);
                    }
                }
                for (; d < headDim; d++)
                {
                    attnOutHead[d] += vPtr[d] * prob;
                }
            }
        }

        // 5. Output projection
        QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, totalQDim);
        QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProjOut, totalQDim, _dim, _attnOutSums);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGemma4DualFfn(int modelLayer, int* selectedIndices, float* selectedWeights)
    {
        var layer = _weights.Layers[modelLayer];

        // ---------------------------------------------------------------------
        // Part 1: Shared MLP (Dense GeLU SwiGLU)
        // ---------------------------------------------------------------------
        QuantKernels.RMSNorm(_attnOutResidual, layer.FfnNormWeight, _mlpNorm, _dim, _weights.RmsNormEps);
        QuantKernels.ComputeBlockSums32(_mlpNorm, _mlpNormSums, _dim);

        QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _mlpNorm, _sharedGate, _dim, _sharedFfnDim, _mlpNormSums);
        QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _mlpNorm, _sharedUp, _dim, _sharedFfnDim, _mlpNormSums);

        QuantKernels.GeluGLU(_sharedGate, _sharedUp, _mlpAct, _sharedFfnDim);
        QuantKernels.ComputeBlockSums32(_mlpAct, _mlpActSums, _sharedFfnDim);

        QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _mlpAct, _mlpOut, _sharedFfnDim, _dim, _mlpActSums);
        QuantKernels.RMSNorm(_mlpOut, layer.FfnPostNorm1Weight, _mlpOut, _dim, _weights.RmsNormEps);

        // ---------------------------------------------------------------------
        // Part 2: Sparse MoE (128 Experts, Top-8 routing)
        // ---------------------------------------------------------------------
        QuantKernels.RMSNorm(_attnOutResidual, layer.FfnPreNorm2Weight, _moeNorm, _dim, _weights.RmsNormEps);
        QuantKernels.ComputeBlockSums32(_moeNorm, _moeNormSums, _dim);

        // Custom router input: RMSNorm(attn_out) / sqrt(dim) * ffn_gate_inp.scale
        QuantKernels.RMSNorm(_attnOutResidual, null, _routerIn, _dim, _weights.RmsNormEps);
        QuantKernels.ScaleAndMul(_routerIn, layer.FfnGateInpScaleWeight, 1.0f / _embeddingScale, _routerIn, _dim);

        // Router logits
        for (int e = 0; e < _expertCount; e++)
        {
            float* row = layer.FfnGateInpWeight + (long)e * _dim;
            _routerLogits[e] = QuantKernels.VecDotF32(row, _routerIn, _dim);
        }

        // Softmax & Top-8 selection with renormalization
        QuantKernels.SoftmaxTopK(_routerLogits, _expertCount, _expertUsedCount, selectedIndices, selectedWeights, normTopK: true);

        // Execute active experts concurrently across CPU cores
        Parallel.For(0, _expertUsedCount, k =>
        {
            int expertIdx = selectedIndices[k];
            float weight = selectedWeights[k];

            float* scratchFused = _expertScratchFused + (long)k * (2 * _expertFfnDim);
            float* scratchAct = _expertScratchAct + (long)k * _expertFfnDim;
            float* scratchActSums = _expertScratchActSums + (long)k * (((_expertFfnDim + 31) / 32));
            float* expertOut = _expertOutputs + (long)k * _dim;

            QuantKernels.ExecuteGemma4Expert(
                expertIdx,
                weight,
                _moeNorm,
                _moeNormSums,
                layer.FfnGateUpExpsWeight,
                layer.FfnGateUpExpsType,
                layer.FfnDownExpsWeight,
                layer.FfnDownExpsType,
                layer.FfnDownExpsScaleWeight,
                _dim,
                _expertFfnDim,
                scratchFused,
                scratchAct,
                scratchActSums,
                expertOut);
        });

        // Reduce expert outputs into _moeOut
        new Span<float>(_moeOut, _dim).Clear();
        for (int k = 0; k < _expertUsedCount; k++)
        {
            float* expertOut = _expertOutputs + (long)k * _dim;
            int i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                int vecLimit = _dim - 8;
                for (; i <= vecLimit; i += 8)
                {
                    var vm = Vector256.Load(_moeOut + i);
                    var ve = Vector256.Load(expertOut + i);
                    (vm + ve).Store(_moeOut + i);
                }
            }
            for (; i < _dim; i++)
            {
                _moeOut[i] += expertOut[i];
            }
        }

        QuantKernels.RMSNorm(_moeOut, layer.FfnPostNorm2Weight, _moeOut, _dim, _weights.RmsNormEps);

        // ---------------------------------------------------------------------
        // Part 3: Combine Shared MLP + Sparse MoE
        // ---------------------------------------------------------------------
        for (int i = 0; i < _dim; i++)
        {
            _combinedFfn[i] = _mlpOut[i] + _moeOut[i];
        }
    }

    /// <summary>
    /// Evaluates prompt tokens sequentially to populate Gemma4 KV Cache.
    /// </summary>
    public override void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool isLast = (i == tokens.Length - 1);
            ForwardStage(tokens[i], startPos + i, default, default, isLast && computeLogits ? logits : Span<float>.Empty, isLast && computeLogits, kvCache);
        }
    }

    protected override void ForwardAttention(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache)
    {
        // Unused directly as ForwardStage implements complete Gemma4 layer graph
    }

    protected override void ForwardBatchChunkAttention(int stageLayer, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache)
    {
        // Unused directly as ForwardBatch routes through ForwardStage
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _gemmaKvCache.Dispose();
            }

            if (_attnIn != null) { NativeMemory.Free(_attnIn); _attnIn = null; }
            if (_attnNormSums != null) { NativeMemory.Free(_attnNormSums); _attnNormSums = null; }
            if (_q != null) { NativeMemory.Free(_q); _q = null; }
            if (_k != null) { NativeMemory.Free(_k); _k = null; }
            if (_v != null) { NativeMemory.Free(_v); _v = null; }
            if (_attnOut != null) { NativeMemory.Free(_attnOut); _attnOut = null; }
            if (_attnOutSums != null) { NativeMemory.Free(_attnOutSums); _attnOutSums = null; }
            if (_attnProjOut != null) { NativeMemory.Free(_attnProjOut); _attnProjOut = null; }
            if (_attnOutResidual != null) { NativeMemory.Free(_attnOutResidual); _attnOutResidual = null; }
            if (_headScores != null) { NativeMemory.Free(_headScores); _headScores = null; }

            if (_mlpNorm != null) { NativeMemory.Free(_mlpNorm); _mlpNorm = null; }
            if (_mlpNormSums != null) { NativeMemory.Free(_mlpNormSums); _mlpNormSums = null; }
            if (_sharedGate != null) { NativeMemory.Free(_sharedGate); _sharedGate = null; }
            if (_sharedUp != null) { NativeMemory.Free(_sharedUp); _sharedUp = null; }
            if (_mlpAct != null) { NativeMemory.Free(_mlpAct); _mlpAct = null; }
            if (_mlpActSums != null) { NativeMemory.Free(_mlpActSums); _mlpActSums = null; }
            if (_mlpOut != null) { NativeMemory.Free(_mlpOut); _mlpOut = null; }

            if (_moeNorm != null) { NativeMemory.Free(_moeNorm); _moeNorm = null; }
            if (_moeNormSums != null) { NativeMemory.Free(_moeNormSums); _moeNormSums = null; }
            if (_routerIn != null) { NativeMemory.Free(_routerIn); _routerIn = null; }
            if (_routerLogits != null) { NativeMemory.Free(_routerLogits); _routerLogits = null; }
            if (_expertScratchFused != null) { NativeMemory.Free(_expertScratchFused); _expertScratchFused = null; }
            if (_expertScratchAct != null) { NativeMemory.Free(_expertScratchAct); _expertScratchAct = null; }
            if (_expertScratchActSums != null) { NativeMemory.Free(_expertScratchActSums); _expertScratchActSums = null; }
            if (_expertOutputs != null) { NativeMemory.Free(_expertOutputs); _expertOutputs = null; }
            if (_moeOut != null) { NativeMemory.Free(_moeOut); _moeOut = null; }

            if (_combinedFfn != null) { NativeMemory.Free(_combinedFfn); _combinedFfn = null; }
            if (_outNorm != null) { NativeMemory.Free(_outNorm); _outNorm = null; }
            if (_outNormSums != null) { NativeMemory.Free(_outNormSums); _outNormSums = null; }

            base.Dispose(disposing);
            _disposed = true;
        }
    }
}
