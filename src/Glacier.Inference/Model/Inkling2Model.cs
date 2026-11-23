namespace Glacier.Inference.Model;

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
#if !ANDROID
using Glacier.Inference.Gpu.D3D12;
#endif
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# .NET 10 SIMD CPU runtime for Thinking Machines Lab's Inkling 2 Mini
/// (3-Pass Looped MoE, Block-Attention Residuals, Relative Positional Attention,
/// Causal Short-Convolutions, SWA-1024 + 8 Global KV caches, SiTU, and NVFP4 experts).
/// </summary>
public sealed unsafe partial class Inkling2Model : ICpuModel
{
    private readonly Inkling2Weights _w;
    private readonly int _maxSeqLen;
    private readonly int _embd;
    private readonly int _nLogical;
    private readonly int _nConvHistory; // l_cache - 1 = 3

    // Per-logical-layer KV cache (SWA layers capped at SlidingWindow = 1024, Global layers at MaxSeqLen)
    private readonly float[][] _kCache;
    private readonly float[][] _vCache;
    private readonly int[] _kvCap;

    // Per-logical-layer short-conv ring buffers: [nConvHistory * channels]
    private readonly float[][] _kConvHist;
    private readonly float[][] _vConvHist;
    private readonly float[][] _attnPostConvHist;
    private readonly float[][] _ffnPostConvHist;

    // Reusable scratch buffers (zero heap allocations on hot token decode)
    private readonly float[] _cur;
    private readonly float[] _prelude;
    private readonly float[] _inpSa;
    private readonly float[] _ffnInp;
    private readonly float[] _in0;
    private readonly float[] _in1;
    private readonly float[] _in2;
    private readonly float[] _q;
    private readonly float[] _k;
    private readonly float[] _v;
    private readonly float[] _r;
    private readonly float[] _sinks;
    private readonly float[] _attnOut;
    private readonly float[] _attnGate;
    private readonly float[] _attnProj;
    private readonly float[] _kDyn;
    private readonly float[] _vDyn;
    private readonly float[] _ctrlCat;
    private readonly float[] _ctrlGate;
    private readonly float[] _ctrlGain;
    private readonly float[] _ctrlForget;
    private readonly float[] _msgBuf;      // 4 * embd
    private readonly float[] _routerLogits;
    private readonly float[] _moeOut;
    private readonly float[][] _expertGateUp; // [ExpertUsedCount][2 * ExpertFfnDim]
    private readonly float[][] _expertHidden; // [ExpertUsedCount][ExpertFfnDim]
    private readonly float[][] _expertDown;   // [ExpertUsedCount][embd]
    private readonly float[] _mixNormBuf;     // max 32 states * embd
    private readonly float[] _mixOutBuf;      // 4 * embd
    private readonly float[] _xSums;          // Q4_K block sums scratch
    private readonly float[][] _headScoreBufs; // [maxHeads][maxSeqLen] zero-alloc attention scores
#if !ANDROID
    private Inkling2D3D12Offload? _d3d12Offload;
    public void AttachD3D12Offload(Inkling2D3D12Offload offload) => _d3d12Offload = offload;
#endif

    public ModelWeights Weights { get; }
    public int MaxSeqLen => _maxSeqLen;
    public int StartLayer => 0;
    public int LayerCount => _w.LogicalLayerCount;
    public bool IsLastStage => true;
    public LoraAdapterWeights? LoraWeights { get; set; }

    public Inkling2Model(ModelWeights weights, int maxSeqLen = 8192)
    {
        Weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _w = weights.Inkling2 ?? throw new InvalidOperationException("ModelWeights.Inkling2 is null.");
        _maxSeqLen = Math.Max(64, maxSeqLen);
        _embd = _w.EmbeddingDim;
        _nLogical = _w.LogicalLayerCount;
        _nConvHistory = _w.ShortConvLCache - 1;

        _kCache = new float[_nLogical][];
        _vCache = new float[_nLogical][];
        _kvCap = new int[_nLogical];
        _kConvHist = new float[_nLogical][];
        _vConvHist = new float[_nLogical][];
        _attnPostConvHist = new float[_nLogical][];
        _ffnPostConvHist = new float[_nLogical][];

        int maxHeads = 0, maxKvDim = 0, maxRelHeads = 0, maxConvProj = 0;
        for (int il = 0; il < _nLogical; il++)
        {
            var layer = _w.PhysicalLayers[PhysicalLayerOf(il)];
            int kvDim = layer.HeadCountKv * layer.HeadDim;
            int cap = layer.IsSwa ? Math.Min(_w.SlidingWindow, _maxSeqLen) : _maxSeqLen;
            _kvCap[il] = cap;
            _kCache[il] = new float[cap * kvDim];
            _vCache[il] = new float[cap * kvDim];

            _kConvHist[il] = new float[_nConvHistory * kvDim];
            _vConvHist[il] = new float[_nConvHistory * kvDim];
            _attnPostConvHist[il] = new float[_nConvHistory * _embd];
            _ffnPostConvHist[il] = new float[_nConvHistory * _embd];

            maxHeads = Math.Max(maxHeads, layer.HeadCount);
            maxKvDim = Math.Max(maxKvDim, kvDim);
            maxRelHeads = Math.Max(maxRelHeads, _w.RelativeLength * layer.HeadCount);
            maxConvProj = Math.Max(maxConvProj, _w.ShortConvLCache * layer.HeadCountKv);
        }

        int qDim = maxHeads * _w.PhysicalLayers[0].HeadDim;
        _cur = new float[_embd];
        _prelude = new float[_embd];
        _inpSa = new float[_embd];
        _ffnInp = new float[_embd];
        _in0 = new float[_embd];
        _in1 = new float[_embd];
        _in2 = new float[_embd];
        _q = new float[qDim];
        _k = new float[maxKvDim];
        _v = new float[maxKvDim];
        _r = new float[maxRelHeads];
        _sinks = new float[maxHeads];
        _attnOut = new float[qDim];
        _attnGate = new float[qDim];
        _attnProj = new float[_embd];
        _kDyn = new float[maxConvProj];
        _vDyn = new float[maxConvProj];
        _ctrlCat = new float[2 * _embd];
        _ctrlGate = new float[_w.GateRank];
        _ctrlGain = new float[_embd];
        _ctrlForget = new float[_embd];
        _msgBuf = new float[4 * _embd];
        _routerLogits = new float[_w.ExpertCount];
        _moeOut = new float[_embd];
        _expertGateUp = new float[_w.ExpertUsedCount][];
        _expertHidden = new float[_w.ExpertUsedCount][];
        _expertDown = new float[_w.ExpertUsedCount][];
        for (int e = 0; e < _w.ExpertUsedCount; e++)
        {
            _expertGateUp[e] = new float[2 * _w.ExpertFfnDim];
            _expertHidden[e] = new float[_w.ExpertFfnDim];
            _expertDown[e] = new float[_embd];
        }
        _mixNormBuf = new float[32 * _embd];
        _mixOutBuf = new float[4 * _embd];
        _xSums = new float[Math.Max(_embd, Math.Max(qDim, _w.ExpertFfnDim)) / 32 + 16];
        _headScoreBufs = new float[maxHeads][];
        for (int h = 0; h < maxHeads; h++) _headScoreBufs[h] = new float[_maxSeqLen];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int PhysicalLayerOf(int il)
        => il < _w.PhysicalLayerCount
            ? il
            : _w.CoreStart + (il - _w.PhysicalLayerCount) % (_w.CoreEnd - _w.CoreStart);

    public void ResetState()
    {
        for (int il = 0; il < _nLogical; il++)
        {
            Array.Clear(_kCache[il]);
            Array.Clear(_vCache[il]);
            Array.Clear(_kConvHist[il]);
            Array.Clear(_vConvHist[il]);
            Array.Clear(_attnPostConvHist[il]);
            Array.Clear(_ffnPostConvHist[il]);
        }
    }

    public void PrefillPrompt(ReadOnlySpan<int> tokens, KVCache kvCache)
    {
        if (tokens.IsEmpty) return;
        if (tokens.Length == 1)
        {
            Forward(tokens[0], 0, kvCache, Span<float>.Empty, computeLogits: false);
            return;
        }
        ExecuteBatchPrefill(tokens, 0, Span<float>.Empty, computeLogits: false);
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        if (tokens.IsEmpty) return;
        if (tokens.Length == 1)
        {
            Forward(tokens[0], startPos, kvCache, logits, computeLogits);
            return;
        }
        ExecuteBatchPrefill(tokens, startPos, logits, computeLogits);
    }

    public void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits, KVCache? kvCache = null)
    {
        Forward(token, pos, kvCache!, logits, computeLogits);
        if (!outputX.IsEmpty) _cur.AsSpan(0, Math.Min(_embd, outputX.Length)).CopyTo(outputX);
    }

    public void ForwardBatchStage(ReadOnlySpan<int> tokens, int startPos, ReadOnlySpan<float> inputXBatch, Span<float> outputXBatch, Span<float> logits, bool computeLogits, KVCache? kvCache = null)
    {
        ForwardBatch(tokens, startPos, logits, kvCache!, computeLogits);
    }

    public void ExtractEmbedding(ReadOnlySpan<int> tokens, Span<float> destination, KVCache kvCache, PoolingStrategy strategy = PoolingStrategy.LastToken)
    {
        ResetState();
        int dim = Math.Min(_embd, destination.Length);
        destination.Clear();
        if (tokens.IsEmpty) return;

        for (int i = 0; i < tokens.Length; i++)
        {
            Forward(tokens[i], i, kvCache, Span<float>.Empty, computeLogits: false);
            if (strategy != PoolingStrategy.LastToken)
            {
                for (int d = 0; d < dim; d++) destination[d] += _cur[d];
            }
            else if (i == tokens.Length - 1)
            {
                _cur.AsSpan(0, dim).CopyTo(destination);
            }
        }
        if (strategy != PoolingStrategy.LastToken)
        {
            float inv = 1.0f / tokens.Length;
            for (int d = 0; d < dim; d++) destination[d] *= inv;
        }
    }

    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        if ((uint)token >= (uint)_w.VocabSizePadded)
            token = 0;

        // 1. Token embedding + RMSNorm
        fixed (float* curPtr = _cur)
        {
            QuantKernels.ExtractEmbedding(_w.TokenEmbdType, _w.TokenEmbd, token, curPtr, _embd);
            RmsNorm(curPtr, _w.TokenEmbdNorm, curPtr, _embd, _w.RmsNormEps);
        }

        float tau = 1.0f + _w.LogScaleAlpha * MathF.Log(MathF.Max(1.0f, (pos + 1.0f) / _w.LogScaleFloor));

        // Block-attention residual history
        var blocks = new List<float[]>(16);

        // Prelude: physical layers 0 .. CoreStart-1
        for (int il = 0; il < _w.CoreStart; il++)
        {
            ExecuteLayer(il, il, pos, tau, blocks, hasMessages: false);
        }

        // Save prelude output & blocks for 3-pass looped core
        Array.Copy(_cur, _prelude, _embd);
        var preludeBlocks = new List<float[]>(blocks.Count);
        foreach (var b in blocks) preludeBlocks.Add((float[])b.Clone());

        // 3-Pass Looped Core: physical layers CoreStart .. CoreEnd-1
        int coreLen = _w.CoreEnd - _w.CoreStart;
        var bank = new List<float[]>(8);
        float[] bankOut = new float[_embd];

        for (int pass = 0; pass < _w.NumLoops; pass++)
        {
            if (pass > 0)
            {
                bank.Clear();
                for (int bi = preludeBlocks.Count + 1; bi < blocks.Count; bi++)
                    bank.Add(blocks[bi]);
                Array.Copy(_cur, bankOut, _embd);

                Array.Copy(_prelude, _cur, _embd);
                blocks.Clear();
                foreach (var b in preludeBlocks) blocks.Add((float[])b.Clone());
            }

            for (int ip = _w.CoreStart; ip < _w.CoreEnd; ip++)
            {
                int il = pass == 0 ? ip : _w.PhysicalLayerCount + (pass - 1) * coreLen + (ip - _w.CoreStart);
                if (pass == 0)
                {
                    ExecuteLayer(il, ip, pos, tau, blocks, hasMessages: false);
                }
                else
                {
                    ComputeCrossPassMessages(ip, bank, bankOut);
                    ExecuteLayer(il, ip, pos, tau, blocks, hasMessages: true);
                }
            }
        }

        // Epilogue: physical layers CoreEnd .. PhysicalLayerCount-1
        for (int il = _w.CoreEnd; il < _w.PhysicalLayerCount; il++)
        {
            ExecuteLayer(il, il, pos, tau, blocks, hasMessages: false);
        }

        if (!computeLogits || logits.IsEmpty)
            return;

        // Output head: Mix(blocks, cur, OutputResQuery) -> RMSNorm -> OutputWeight -> Softcap -> Unpadded vocab
        Mix(blocks, _cur, _w.OutputResQuery, 1, _mixOutBuf);
        fixed (float* normOut = _cur)
        fixed (float* mixPtr = _mixOutBuf)
        {
            RmsNorm(mixPtr, _w.OutputNorm, normOut, _embd, _w.RmsNormEps);
        }

        int vocabOut = Math.Min(_w.VocabSizeUnpadded, logits.Length);
#if !ANDROID
        if (_d3d12Offload != null && _d3d12Offload.OutputHeadPinned)
        {
            _d3d12Offload.ExecuteOutputHead(_cur, logits, vocabOut);
        }
        else
#endif
        {
            MatVecParallel(_w.OutputWeight, _w.OutputWeightType, _cur, logits, vocabOut, _embd);
        }

        if (_w.FinalLogitSoftcapping > 0.0f)
        {
            float cap = _w.FinalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (int v = 0; v < vocabOut; v++)
            {
                logits[v] = cap * MathF.Tanh(logits[v] * invCap);
            }
        }
    }

    private void ComputeCrossPassMessages(int ip, List<float[]> bank, float[] bankOut)
    {
        var layer = _w.PhysicalLayers[ip];
        if (layer.LoopMsgQuery != null)
        {
            Mix(bank, bankOut, layer.LoopMsgQuery, 4, _msgBuf);
            return;
        }

        // Unsloth GGUF layout: loop_res_query [n_embd, 2*coreLen] (attn, mlp) + loop_kv_res_query [n_embd, 2*coreLen] (k, v)
        int cIdx = ip - _w.CoreStart;
        if (_w.LoopResQuery != null && _w.LoopKvResQuery != null)
        {
            Mix(bank, bankOut, _w.LoopResQuery + (2 * cIdx) * _embd, 1, _mixOutBuf);
            Array.Copy(_mixOutBuf, 0, _msgBuf, 0 * _embd, _embd); // Q/Attn

            Mix(bank, bankOut, _w.LoopKvResQuery + (2 * cIdx) * _embd, 1, _mixOutBuf);
            Array.Copy(_mixOutBuf, 0, _msgBuf, 1 * _embd, _embd); // K

            Mix(bank, bankOut, _w.LoopKvResQuery + (2 * cIdx + 1) * _embd, 1, _mixOutBuf);
            Array.Copy(_mixOutBuf, 0, _msgBuf, 2 * _embd, _embd); // V

            Mix(bank, bankOut, _w.LoopResQuery + (2 * cIdx + 1) * _embd, 1, _mixOutBuf);
            Array.Copy(_mixOutBuf, 0, _msgBuf, 3 * _embd, _embd); // MLP
        }
    }

    private void ExecuteLayer(int il, int ip, int pos, float tau, List<float[]> blocks, bool hasMessages)
    {
        var layer = _w.PhysicalLayers[ip];
        bool isSwa = layer.IsSwa;
        int nHead = layer.HeadCount;
        int nHeadKv = layer.HeadCountKv;
        int headDim = layer.HeadDim;
        int qDim = nHead * headDim;
        int kvDim = nHeadKv * headDim;
        int nRel = _w.RelativeLength;

        Array.Copy(_cur, _inpSa, _embd);

        // 1. Attention inputs via block-attention residual mix
        if (layer.AttnResQuery != null)
        {
            int nQ = isSwa ? 1 : 3;
            Mix(blocks, _cur, layer.AttnResQuery, nQ, _mixOutBuf);
            Array.Copy(_mixOutBuf, 0, _in0, 0, _embd);
            Array.Copy(_mixOutBuf, isSwa ? 0 : _embd, _in1, 0, _embd);
            Array.Copy(_mixOutBuf, isSwa ? 0 : 2 * _embd, _in2, 0, _embd);
        }
        else
        {
            Array.Copy(_cur, _in0, _embd);
            Array.Copy(_cur, _in1, _embd);
            Array.Copy(_cur, _in2, _embd);
        }

        // Cross-pass control gates on Q, K, V inputs
        if (hasMessages)
        {
            ApplyLoopControl(_in0, 0 * _embd, layer, 0);
            ApplyLoopControl(_in1, 1 * _embd, layer, 1);
            ApplyLoopControl(_in2, 2 * _embd, layer, 2);
        }

        // Input RMSNorms
        fixed (float* p0 = _in0, p1 = _in1, p2 = _in2)
        {
            if (isSwa && hasMessages)
            {
                RmsNorm(p0, layer.AttnNorm, p0, _embd, _w.RmsNormEps);
                RmsNorm(p1, layer.AttnNorm, p1, _embd, _w.RmsNormEps);
                RmsNorm(p2, layer.AttnNorm, p2, _embd, _w.RmsNormEps);
            }
            else if (isSwa)
            {
                RmsNorm(p0, layer.AttnNorm, p0, _embd, _w.RmsNormEps);
                Array.Copy(_in0, _in1, _embd);
                Array.Copy(_in0, _in2, _embd);
            }
            else
            {
                RmsNorm(p0, layer.AttnQInpNorm, p0, _embd, _w.RmsNormEps);
                RmsNorm(p1, layer.AttnKInpNorm, p1, _embd, _w.RmsNormEps);
                RmsNorm(p2, layer.AttnVInpNorm, p2, _embd, _w.RmsNormEps);
            }
        }

        // 2. Q, K, V, R, Gate projections + Short-Conv on K & V
        bool attnOnGpu = false;
#if !ANDROID
        if (_d3d12Offload != null && _d3d12Offload.IsAttnPinned(ip))
        {
            _d3d12Offload.ExecuteAttnQkvrg(ip, _in0, _in1, _in2, _q, _k, _v, _r, _attnGate);
            attnOnGpu = true;
        }
        else
#endif
        {
            MatVecParallel(layer.Wq, layer.WqType, _in0, _q.AsSpan(0, qDim), qDim, _embd);
            MatVecParallel(layer.Wk, layer.WkType, _in1, _k.AsSpan(0, kvDim), kvDim, _embd);
            MatVecParallel(layer.Wv, layer.WvType, _in2, _v.AsSpan(0, kvDim), kvDim, _embd);
            MatVecParallel(layer.AttnRel, layer.AttnRelType, _in0, _r.AsSpan(0, nRel * nHead), nRel * nHead, _embd);
        }

        float[]? kDyn = null;
        float[]? vDyn = null;
        if (!isSwa && layer.AttnKConvProj != null && layer.AttnVConvProj != null)
        {
            int dynDim = _w.ShortConvLCache * nHeadKv;
            MatVec(layer.AttnKConvProj, layer.AttnKConvProjType, _in1, _kDyn.AsSpan(0, dynDim), dynDim, _embd);
            MatVec(layer.AttnVConvProj, layer.AttnVConvProjType, _in2, _vDyn.AsSpan(0, dynDim), dynDim, _embd);
            kDyn = _kDyn;
            vDyn = _vDyn;
        }

        ApplyShortConv(_k, kvDim, layer.AttnKConv, kDyn, nHeadKv, _kConvHist[il]);
        ApplyShortConv(_v, kvDim, layer.AttnVConv, vDyn, nHeadKv, _vConvHist[il]);

        // L2 Clipping on Q, K, R and RMSNorm on V
        ClipPerHead(_q, headDim, nHead, layer.AttnQClip, _w.QkClip);
        ClipPerHead(_k, headDim, nHeadKv, null, _w.QkClip);
        RmsNormPerHead(_v, headDim, nHeadKv, _w.RmsNormEps);
        ClipPerHead(_r, nRel, nHead, layer.AttnRelClip, _w.RelativeClip);

        if (!isSwa)
        {
            ScaleSpan(_q.AsSpan(0, qDim), tau);
            ScaleSpan(_r.AsSpan(0, nRel * nHead), tau);
        }

        // Store K & V into logical layer KV cache
        int cap = _kvCap[il];
        int slot = pos % cap;
        Array.Copy(_k, 0, _kCache[il], slot * kvDim, kvDim);
        Array.Copy(_v, 0, _vCache[il], slot * kvDim, kvDim);

        // Attention sinks
        MatVec(layer.AttnSinkProj, layer.AttnSinkProjType, _in0, _sinks.AsSpan(0, nHead), nHead, _embd);

        // 3. Relative Positional Attention with Attention Sinks (parallel across heads)
        int startT = isSwa ? Math.Max(0, pos - _w.SlidingWindow + 1) : 0;
        int kvLen = pos - startT + 1;
        int groupSize = nHead / nHeadKv;
        float invSqrtD = 1.0f / MathF.Sqrt(headDim);
        int relExtent = layer.RelExtent;

        Parallel.For(0, nHead, h =>
        {
            int kvHead = h / groupSize;
            int qOff = h * headDim;
            int rOff = h * nRel;
            float sinkVal = _sinks[h];
            float[] kCacheL = _kCache[il];
            float[] vCacheL = _vCache[il];

            float[] scores = _headScoreBufs[h];
            float maxLogit = 0.0f; // virtual zero sink logit

            for (int ti = 0; ti < kvLen; ti++)
            {
                int t = startT + ti;
                int dist = pos - t;
                int kOff = (t % cap) * kvDim + kvHead * headDim;

                float dot = DotProduct(_q, qOff, kCacheL, kOff, headDim) * invSqrtD;
                float relBias = 0.0f;
                if (dist < relExtent)
                {
                    float* biasRow = layer.AttnRelBias + dist * nRel;
                    for (int ri = 0; ri < nRel; ri++)
                        relBias += _r[rOff + ri] * biasRow[ri];
                }

                float s = dot + relBias - sinkVal;
                scores[ti] = s;
                if (s > maxLogit) maxLogit = s;
            }

            float sumExp = MathF.Exp(-maxLogit); // virtual zero sink contribution
            for (int ti = 0; ti < kvLen; ti++)
            {
                float e = MathF.Exp(scores[ti] - maxLogit);
                scores[ti] = e;
                sumExp += e;
            }

            float invSum = 1.0f / sumExp;
            int outOff = h * headDim;
            Array.Clear(_attnOut, outOff, headDim);
            for (int ti = 0; ti < kvLen; ti++)
            {
                float w = scores[ti] * invSum;
                int vOff = ((startT + ti) % cap) * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                    _attnOut[outOff + d] += w * vCacheL[vOff + d];
            }
        });

        // Attention output gate (sigmoid) + Wo + post-conv
        if (!attnOnGpu)
        {
            MatVecParallel(layer.AttnGate, layer.AttnGateType, _in0, _attnGate.AsSpan(0, qDim), qDim, _embd);
        }
        for (int i = 0; i < qDim; i++)
        {
            float sig = 1.0f / (1.0f + MathF.Exp(-_attnGate[i]));
            _attnOut[i] *= sig;
        }

#if !ANDROID
        if (attnOnGpu && _d3d12Offload != null)
        {
            _d3d12Offload.ExecuteAttnWo(ip, _attnOut, _attnProj);
        }
        else
#endif
        {
            MatVecParallel(layer.Wo, layer.WoType, _attnOut, _attnProj.AsSpan(0, _embd), _embd, qDim);
        }
        ApplyShortConv(_attnProj, _embd, layer.AttnPostConv, null, 0, _attnPostConvHist[il]);

        // Block residual stream boundary
        if (ip % _w.AttnResBlockSize == 0)
        {
            blocks.Add((float[])_inpSa.Clone());
            Array.Copy(_attnProj, _cur, _embd);
        }
        else
        {
            for (int i = 0; i < _embd; i++)
                _cur[i] = _inpSa[i] + _attnProj[i];
        }

        Array.Copy(_cur, _ffnInp, _embd);

        // 4. MoE FFN with SiTU + Post-Conv
        Mix(blocks, _cur, layer.FfnResQuery, 1, _mixOutBuf);
        Array.Copy(_mixOutBuf, 0, _cur, 0, _embd);

        if (hasMessages)
        {
            ApplyLoopControl(_cur, 3 * _embd, layer, 3);
        }

        fixed (float* curPtr = _cur)
        {
            RmsNorm(curPtr, layer.FfnNorm, curPtr, _embd, _w.RmsNormEps);
        }

        // Router (sigmoid + bias -> top-K)
        int nExp = _w.ExpertCount;
        int topK = _w.ExpertUsedCount;
        MatVec(layer.FfnGateInp, layer.FfnGateInpType, _cur, _routerLogits.AsSpan(0, nExp), nExp, _embd);

        Span<int> topIds = stackalloc int[topK];
        Span<float> topProbs = stackalloc float[topK];
        Span<float> topScores = stackalloc float[topK];
        for (int k = 0; k < topK; k++) { topIds[k] = -1; topScores[k] = float.NegativeInfinity; }

        for (int e = 0; e < nExp; e++)
        {
            float prob = 1.0f / (1.0f + MathF.Exp(-_routerLogits[e]));
            float score = prob + (layer.ExpProbsBias != null ? layer.ExpProbsBias[e] : 0.0f);
            if (score > topScores[topK - 1])
            {
                int posIns = topK - 1;
                while (posIns > 0 && score > topScores[posIns - 1])
                {
                    topScores[posIns] = topScores[posIns - 1];
                    topProbs[posIns] = topProbs[posIns - 1];
                    topIds[posIns] = topIds[posIns - 1];
                    posIns--;
                }
                topScores[posIns] = score;
                topProbs[posIns] = prob;
                topIds[posIns] = e;
            }
        }

        float probSum = 1e-20f;
        for (int k = 0; k < topK; k++) probSum += topProbs[k];
        float wMul = _w.ExpertWeightsScale / probSum;
        for (int k = 0; k < topK; k++) topProbs[k] *= wMul;

        // Compute top-K experts across ALL CPU cores (24+ threads) with fused SiTU & direct _moeOut accumulation
        int ffExp = _w.ExpertFfnDim;
        long guRowBytes = GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, _embd);
        long dnRowBytes = GgufTypes.GetRowBytes(layer.FfnDownExpsType, ffExp);
        long gateUpExpBytes = (long)(2 * ffExp) * guRowBytes;
        long downExpBytes = (long)_embd * dnRowBytes;
        float beta = _w.SituBeta;
        float invBeta = 1.0f / beta;
        float lb = _w.SituLinearBeta;
        float invLb = lb > 0.0f ? 1.0f / lb : 0.0f;

        int[] activeIds = topIds.ToArray();
        float[] activeWeights = topProbs.ToArray();

#if !ANDROID
        if (_d3d12Offload != null && _d3d12Offload.IsLayerPinned(ip))
        {
            _d3d12Offload.ExecuteMoeLayer(ip, _cur, activeIds, activeWeights, _moeOut);
        }
        else
#endif
        {
            // Pass A: Gate+Up + SiTU sharded across (expert k, 4 row chunks) = 32 parallel tasks
            const int guChunksPerExpert = 4;
            int guChunkSize = (ffExp + guChunksPerExpert - 1) / guChunksPerExpert;
            nint guBaseAddr = (nint)layer.FfnGateUpExps;
            nint guScaleAddr = (nint)layer.FfnGateUpExpsScale;
            var guType = layer.FfnGateUpExpsType;

            fixed (float* curX = _cur)
            fixed (float* xsPtr = _xSums)
            {
                nint xAddr = (nint)curX;
                nint xsAddr = (nint)xsPtr;

                Parallel.For(0, topK * guChunksPerExpert, taskIdx =>
                {
                    int k = taskIdx / guChunksPerExpert;
                    int c = taskIdx % guChunksPerExpert;
                    int jStart = c * guChunkSize;
                    int jEnd = Math.Min(ffExp, jStart + guChunkSize);
                    if (jStart >= jEnd) return;

                    int expId = activeIds[k];
                    byte* expPtr = (byte*)guBaseAddr + expId * gateUpExpBytes;
                    float guScale = guScaleAddr != 0 ? ((float*)guScaleAddr)[expId] : 1.0f;
                    float[] hid = _expertHidden[k];

                    byte* gRow = expPtr + jStart * guRowBytes;
                    byte* uRow = expPtr + (ffExp + jStart) * guRowBytes;
                    for (int j = jStart; j < jEnd; j++)
                    {
                        float g = QuantKernels.ComputeDot(guType, gRow, (float*)xAddr, (float*)xsAddr, _embd) * guScale;
                        float u = QuantKernels.ComputeDot(guType, uRow, (float*)xAddr, (float*)xsAddr, _embd) * guScale;
                        float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                        if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                        hid[j] = act * u;
                        gRow += guRowBytes;
                        uRow += guRowBytes;
                    }
                });
            }

            // Pass B: Down projection + weighted sum sharded across output dimension tiles (all CPU cores, zero reduction)
            int nOutTiles = Math.Max(8, Environment.ProcessorCount);
            int outTileSize = (_embd + nOutTiles - 1) / nOutTiles;
            nint dnBaseAddr = (nint)layer.FfnDownExps;
            nint dnScaleAddr = (nint)layer.FfnDownExpsScale;
            var dnType = layer.FfnDownExpsType;

            fixed (float* moePtr = _moeOut)
            {
                nint moeAddr = (nint)moePtr;
                Parallel.For(0, nOutTiles, tile =>
                {
                    int dStart = tile * outTileSize;
                    int dEnd = Math.Min(_embd, dStart + outTileSize);
                    if (dStart >= dEnd) return;

                    float* outPtr = (float*)moeAddr;
                    for (int d = dStart; d < dEnd; d++) outPtr[d] = 0.0f;

                    for (int k = 0; k < topK; k++)
                    {
                        int expId = activeIds[k];
                        float dnScale = dnScaleAddr != 0 ? ((float*)dnScaleAddr)[expId] : 1.0f;
                        float w = activeWeights[k] * dnScale;
                        byte* dnRow = (byte*)dnBaseAddr + expId * downExpBytes + dStart * dnRowBytes;

                        fixed (float* hidPtr = _expertHidden[k])
                        {
                            for (int d = dStart; d < dEnd; d++)
                            {
                                outPtr[d] += w * QuantKernels.ComputeDot(dnType, dnRow, hidPtr, null, ffExp);
                                dnRow += dnRowBytes;
                            }
                        }
                    }
                });
            }
        }

        ApplyShortConv(_moeOut, _embd, layer.FfnPostConv, null, 0, _ffnPostConvHist[il]);
        for (int d = 0; d < _embd; d++)
            _cur[d] = _ffnInp[d] + _moeOut[d];
    }

    private void Mix(List<float[]> blocks, float[] cur, float* queryPtr, int nQueries, float[] outBuf)
    {
        int nStates = blocks.Count + 1;
        // RMSNorm every state vector with eps = 1e-6f
        fixed (float* normAll = _mixNormBuf)
        {
            for (int s = 0; s < blocks.Count; s++)
            {
                fixed (float* bPtr = blocks[s])
                    RmsNormNoWeight(bPtr, normAll + s * _embd, _embd, 1e-6f);
            }
            fixed (float* cPtr = cur)
                RmsNormNoWeight(cPtr, normAll + blocks.Count * _embd, _embd, 1e-6f);

            float invSqrtD = 1.0f / MathF.Sqrt(_embd);
            Span<float> scores = stackalloc float[nStates];

            for (int q = 0; q < nQueries; q++)
            {
                float* qRow = queryPtr + q * _embd;
                float maxS = float.NegativeInfinity;
                for (int s = 0; s < nStates; s++)
                {
                    float dot = DotPtr(normAll + s * _embd, qRow, _embd) * invSqrtD;
                    scores[s] = dot;
                    if (dot > maxS) maxS = dot;
                }

                float sumExp = 0.0f;
                for (int s = 0; s < nStates; s++)
                {
                    float e = MathF.Exp(scores[s] - maxS);
                    scores[s] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;

                int dstOff = q * _embd;
                Array.Clear(outBuf, dstOff, _embd);
                for (int s = 0; s < nStates; s++)
                {
                    float w = scores[s] * invSum;
                    float* st = normAll + s * _embd;
                    for (int d = 0; d < _embd; d++)
                        outBuf[dstOff + d] += w * st[d];
                }
            }
        }
    }

    private void ApplyLoopControl(float[] cur, int msgOffset, Inkling2LayerWeights layer, int ctrl)
    {
        fixed (float* curPtr = cur)
        fixed (float* msgPtr = _msgBuf)
        fixed (float* catPtr = _ctrlCat)
        {
            RmsNormNoWeight(curPtr, catPtr, _embd, 1e-6f);
            RmsNormNoWeight(msgPtr + msgOffset, catPtr + _embd, _embd, 1e-6f);
        }

        int rank = _w.GateRank;
        MatVec(layer.LoopCtrlDown[ctrl], layer.LoopCtrlDownType[ctrl], _ctrlCat, _ctrlGate.AsSpan(0, rank), rank, 2 * _embd);
        MatVec(layer.LoopCtrlGain[ctrl], layer.LoopCtrlGainType[ctrl], _ctrlGate, _ctrlGain.AsSpan(0, _embd), _embd, rank);
        MatVec(layer.LoopCtrlForget[ctrl], layer.LoopCtrlForgetType[ctrl], _ctrlGate, _ctrlForget.AsSpan(0, _embd), _embd, rank);

        for (int d = 0; d < _embd; d++)
        {
            float g = MathF.Tanh(_ctrlGain[d]);
            float f = MathF.Tanh(_ctrlForget[d]);
            cur[d] = (1.0f - f) * cur[d] + g * _msgBuf[msgOffset + d];
        }
    }

    private void ApplyShortConv(float[] cur, int channels, float* weight, float[]? dynamicTaps, int nHeads, float[] hist)
    {
        int nConv = _w.ShortConvLCache; // 4
        int nHist = _nConvHistory;      // 3
        float scale = 1.0f / MathF.Sqrt((dynamicTaps != null ? 2 : 1) * nConv + 1);
        int headDim = nHeads > 0 ? channels / nHeads : 0;

        for (int c = 0; c < channels; c++)
        {
            int wOff = c * nConv;
            int hOff = c * nHist;
            float x0 = hist[hOff + 0];
            float x1 = hist[hOff + 1];
            float x2 = hist[hOff + 2];
            float x3 = cur[c];

            float conv = weight[wOff + 0] * x0
                       + weight[wOff + 1] * x1
                       + weight[wOff + 2] * x2
                       + weight[wOff + 3] * x3;

            if (dynamicTaps != null)
            {
                int h = c / headDim;
                int dOff = h * nConv;
                // lag=0 -> x3 (cur), lag=1 -> x2, lag=2 -> x1, lag=3 -> x0
                conv += dynamicTaps[dOff + 0] * x3
                      + dynamicTaps[dOff + 1] * x2
                      + dynamicTaps[dOff + 2] * x1
                      + dynamicTaps[dOff + 3] * x0;
            }

            hist[hOff + 0] = x1;
            hist[hOff + 1] = x2;
            hist[hOff + 2] = x3;
            cur[c] = (x3 + conv) * scale;
        }
    }

    private static void ClipPerHead(float[] buf, int dim, int nHeads, float* scalePtr, float threshold)
    {
        for (int h = 0; h < nHeads; h++)
        {
            int off = h * dim;
            float sumSq = 0.0f;
            if (scalePtr != null)
            {
                float* sc = scalePtr + off;
                for (int i = 0; i < dim; i++)
                {
                    float v = buf[off + i] * sc[i];
                    buf[off + i] = v;
                    sumSq += v * v;
                }
            }
            else
            {
                for (int i = 0; i < dim; i++)
                {
                    float v = buf[off + i];
                    sumSq += v * v;
                }
            }

            float norm = MathF.Sqrt(sumSq) + 1e-6f;
            if (norm > threshold)
            {
                float f = threshold / norm;
                for (int i = 0; i < dim; i++) buf[off + i] *= f;
            }
        }
    }

    private static void RmsNormPerHead(float[] buf, int dim, int nHeads, float eps)
    {
        for (int h = 0; h < nHeads; h++)
        {
            int off = h * dim;
            float sumSq = 0.0f;
            for (int i = 0; i < dim; i++) sumSq += buf[off + i] * buf[off + i];
            float inv = 1.0f / MathF.Sqrt(sumSq / dim + eps);
            for (int i = 0; i < dim; i++) buf[off + i] *= inv;
        }
    }

    private static void RmsNorm(float* x, float* w, float* dst, int n, float eps)
    {
        float sumSq = 0.0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float inv = 1.0f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) dst[i] = x[i] * inv * w[i];
    }

    private static void RmsNormNoWeight(float* x, float* dst, int n, float eps)
    {
        float sumSq = 0.0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float inv = 1.0f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) dst[i] = x[i] * inv;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleSpan(Span<float> span, float s)
    {
        for (int i = 0; i < span.Length; i++) span[i] *= s;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotPtr(float* a, float* b, int n)
    {
        int i = 0;
        float acc = 0.0f;
        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
        {
            var vSum = Vector<float>.Zero;
            int w = Vector<float>.Count;
            for (; i + w <= n; i += w)
                vSum += Unsafe.ReadUnaligned<Vector<float>>(a + i) * Unsafe.ReadUnaligned<Vector<float>>(b + i);
            acc = Vector.Sum(vSum);
        }
        for (; i < n; i++) acc += a[i] * b[i];
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotProduct(float[] a, int aOff, float[] b, int bOff, int n)
    {
        fixed (float* pa = &a[aOff])
        fixed (float* pb = &b[bOff])
            return DotPtr(pa, pb, n);
    }

    private void MatVec(byte* weight, GgufType type, float[] x, Span<float> y, int rows, int cols)
    {
        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        fixed (float* xPtr = x)
        fixed (float* xSumsPtr = _xSums)
        {
            if (type == GgufType.Q4_K)
            {
                int nb = cols / 32;
                for (int b = 0; b < nb; b++)
                {
                    float s = 0f;
                    for (int j = 0; j < 32; j++) s += xPtr[b * 32 + j];
                    xSumsPtr[b] = s;
                }
            }
            for (int r = 0; r < rows; r++)
            {
                y[r] = QuantKernels.ComputeDot(type, weight + r * rowBytes, xPtr, xSumsPtr, cols);
            }
        }
    }

    private void MatVecParallel(byte* weight, GgufType type, float[] x, Span<float> y, int rows, int cols)
    {
        if (rows < 256)
        {
            MatVec(weight, type, x, y, rows, cols);
            return;
        }
        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        fixed (float* xPtr = x)
        fixed (float* xSumsPtr = _xSums)
        fixed (float* yPtr = y)
        {
            if (type == GgufType.Q4_K)
            {
                int nb = cols / 32;
                for (int b = 0; b < nb; b++)
                {
                    float s = 0f;
                    for (int j = 0; j < 32; j++) s += xPtr[b * 32 + j];
                    xSumsPtr[b] = s;
                }
            }
            nint wAddr = (nint)weight;
            nint xAddr = (nint)xPtr;
            nint xsAddr = (nint)xSumsPtr;
            nint yAddr = (nint)yPtr;

            int chunk = rows >= 8192 ? 256 : 32;
            int nChunks = (rows + chunk - 1) / chunk;
            Parallel.For(0, nChunks, ci =>
            {
                int rStart = ci * chunk;
                int rEnd = Math.Min(rows, rStart + chunk);
                byte* wBase = (byte*)wAddr + rStart * rowBytes;
                float* yOut = (float*)yAddr;
                for (int r = rStart; r < rEnd; r++)
                {
                    yOut[r] = QuantKernels.ComputeDot(type, wBase, (float*)xAddr, (float*)xsAddr, cols);
                    wBase += rowBytes;
                }
            });
        }
    }

    public void Dispose()
    {
#if !ANDROID
        _d3d12Offload?.Dispose();
        _d3d12Offload = null;
#endif
    }
}
