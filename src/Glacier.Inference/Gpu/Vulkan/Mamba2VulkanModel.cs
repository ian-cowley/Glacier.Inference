using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;

namespace Glacier.Inference.Gpu;

/// <summary>
/// Vulkan 1.1 compute runner for Mamba-2 / Hybrid Mamba-2 + Transformer + MoE models
/// (e.g. NVIDIA Nemotron 3 Nano 30B-A3B, IBM Granite 4.0/4.1 Hybrid) on Linux/non-Windows GPUs.
/// </summary>
public sealed unsafe class Mamba2VulkanModel : ID3D12Model
{
    private enum Patch { None, Pos, KvLen, Token }

    private sealed class Op
    {
        public VulkanCompute.Kernel K = null!;
        public IntPtr Set;
        public uint[] Push = Array.Empty<uint>();
        public uint Gx = 1, Gy = 1;
        public Patch Patch;
    }

    private readonly ModelWeights _w;
    private readonly VulkanContext _ctx;
    private readonly VulkanCompute _vk;
    private readonly int _maxSeqLen, _vocab;

    private readonly VulkanCompute.GBuf _dX, _dNorm, _dQ, _dK, _dV, _dQOut, _dAttnOut, _dProj;
    private readonly VulkanCompute.GBuf _dGate, _dUp, _dLogits, _dDummyNorm;
    private readonly VulkanCompute.GBuf _dEmbd, _dOutNorm, _dOutput;
    private readonly List<Op> _embedOps = new();
    private readonly List<Op> _layerOps = new();
    private readonly List<Op> _finalOps = new();

    private readonly VulkanCompute.Kernel _pRmsNorm, _pAdd, _pSwiGlu, _pGqaPrep, _pGqaDecode, _pM2Conv, _pM2Rec;
    private readonly VulkanCompute.Kernel? _pF32, _pTopk, _pComb;
    private readonly VulkanCompute.Kernel[] _pEmbed = new VulkanCompute.Kernel[4];
    private readonly VulkanCompute.Kernel[] _pQmv = new VulkanCompute.Kernel[6];
    private readonly VulkanCompute.Kernel[] _pMoe = new VulkanCompute.Kernel[6];

    public string DeviceName => _ctx.DeviceName;
    public int MaxSeqLen => _maxSeqLen;

    public static bool IsSupported(ModelWeights w, out string reason)
    {
        reason = "";
        if (w.ArchitectureFamily != UniversalArchitecture.Mamba2) { reason = "not a Mamba-2 model"; return false; }
        if (w.HeadDim > 256 || w.ValueDim != w.HeadDim) { reason = $"unsupported head dim {w.HeadDim}/{w.ValueDim}"; return false; }
        if (w.HasRopeFreqs) { reason = "custom RoPE frequency table"; return false; }

        foreach (var l in w.Layers)
        {
            (string Name, bool Present, GgufType T)[] mats =
            {
                ("attn_q", l.QWeight != null, l.QType), ("attn_k", l.KWeight != null, l.KType),
                ("attn_v", l.VWeight != null, l.VType), ("attn_output", l.AttnOutWeight != null, l.AttnOutType),
                ("ssm_in", l.QkvWeight != null, l.QkvType), ("ssm_out", l.SsmOutWeight != null, l.SsmOutType),
                ("ffn_gate", l.FfnGateWeight != null, l.FfnGateType), ("ffn_up", l.FfnUpWeight != null, l.FfnUpType), ("ffn_down", l.FfnDownWeight != null, l.FfnDownType),
                ("ffn_gate_exps", l.FfnGateExpsWeight != null, l.FfnGateExpsType), ("ffn_up_exps", l.FfnUpExpsWeight != null, l.FfnUpExpsType), ("ffn_down_exps", l.FfnDownExpsWeight != null, l.FfnDownExpsType),
            };
            foreach (var (name, present, t) in mats)
                if (present && !IsKernelType(t)) { reason = $"tensor '{name}' uses unsupported weight type {t}"; return false; }
        }
        if (!IsKernelType(w.EmbdType)) { reason = $"token embedding uses unsupported weight type {w.EmbdType}"; return false; }
        if (!IsKernelType(w.OutType)) { reason = $"output head uses unsupported weight type {w.OutType}"; return false; }
        return true;
    }

    private static bool IsKernelType(GgufType t) => t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.IQ4_XS or GgufType.MXFP4;

    private static int QIdx(GgufType t) => t switch
    {
        GgufType.Q4_K => 0, GgufType.Q5_K => 1, GgufType.Q6_K => 2, GgufType.Q8_0 => 3, GgufType.IQ4_XS => 4, GgufType.MXFP4 => 5,
        _ => throw new NotSupportedException($"Unsupported Vulkan quant type {t}"),
    };

    private static uint FBits(float f) => BitConverter.SingleToUInt32Bits(f);

    public Mamba2VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Mamba-2 Vulkan runner cannot execute this model: {why}.");

        _w = weights;
        _maxSeqLen = maxSeqLen;
        _vocab = weights.VocabSize;
        _ctx = new VulkanContext(deviceOrdinal);
        _vk = new VulkanCompute(_ctx);

        try
        {
            _pRmsNorm = _vk.LoadEmbedded("rmsnorm", 3, 12);
            _pAdd = _vk.LoadEmbedded("elementwise_add", 3, 4);
            _pSwiGlu = _vk.LoadEmbedded("elementwise_swiglu", 3, 4);
            _pGqaPrep = _vk.LoadEmbedded("gqa_prep", 8, 40);
            _pGqaDecode = _vk.LoadEmbedded("gqa_decode", 4, 24);
            _pM2Conv = _vk.LoadEmbedded("mamba2_conv", 5, 8);
            _pM2Rec = _vk.LoadEmbedded("mamba2_rec", 8, 20);
            for (int q = 0; q < 6; q++)
            {
                if (q < 4) _pEmbed[q] = _vk.LoadEmbedded($"embed_q{q}", 2, 8);
                _pQmv[q] = _vk.LoadEmbedded($"qmatvec_q{q}", 3, 8);
                if (_w.IsMoe) _pMoe[q] = _vk.LoadEmbedded($"qmatvec_moe_q{q}", 4, 16);
            }
            if (_w.IsMoe)
            {
                _pF32 = _vk.LoadEmbedded("f32_matvec", 3, 8);
                _pTopk = _vk.LoadEmbedded("moe_topk", 3, 12);
                _pComb = _vk.LoadEmbedded("moe_combine", 6, 12);
            }

            int E = _w.EmbeddingLength;
            int nQ = _w.HeadCount;
            int nKv = _w.HeadCountKv;
            int hd = _w.HeadDim;
            int qDim = nQ * hd;
            int kvDim = nKv * hd;
            int ffnDim = _w.FeedForwardLength;
            float eps = _w.RmsNormEps;

            _dX = _vk.Alloc((long)E * 4);
            _dNorm = _vk.Alloc((long)E * 4);
            _dQ = _vk.Alloc((long)qDim * 4);
            _dK = _vk.Alloc((long)kvDim * 4);
            _dV = _vk.Alloc((long)kvDim * 4);
            _dQOut = _vk.Alloc((long)qDim * 4);
            _dAttnOut = _vk.Alloc((long)qDim * 4);
            _dProj = _vk.Alloc((long)E * 4);
            _dGate = _vk.Alloc((long)ffnDim * 4);
            _dUp = _vk.Alloc((long)ffnDim * 4);
            _dLogits = _vk.Alloc((long)_vocab * 4);
            _dDummyNorm = _vk.Alloc(4);

            int nExp = _w.ExpertCount;
            int topK = _w.ExpertUsedCount;
            int moeF = _w.ExpertFeedForwardLength;
            VulkanCompute.GBuf? dRouterLogits = null, dIdx = null, dTw = null;
            VulkanCompute.GBuf? dMoeGate = null, dMoeUp = null, dMoeAct = null, dMoeDown = null, dZeroE = null;
            if (_w.IsMoe)
            {
                dRouterLogits = _vk.Alloc((long)nExp * 4);
                dIdx = _vk.Alloc((long)topK * 4);
                dTw = _vk.Alloc((long)topK * 4);
                dMoeGate = _vk.Alloc((long)topK * moeF * 4);
                dMoeUp = _vk.Alloc((long)topK * moeF * 4);
                dMoeAct = _vk.Alloc((long)topK * moeF * 4);
                dMoeDown = _vk.Alloc((long)topK * E * 4);
                dZeroE = _vk.Alloc((long)E * 4);
            }

            _dEmbd = UploadBytes(_w.EmbdWeight, GgufTypeSize(_w.EmbdType, (long)_vocab * E));
            _dOutNorm = UploadFloats(_w.OutNormWeight, E);
            _dOutput = UploadBytes(_w.OutWeight, GgufTypeSize(_w.OutType, (long)_vocab * E));

            _embedOps.Add(new Op
            {
                K = _pEmbed[QIdx(_w.EmbdType)],
                Set = _vk.MakeSet(_pEmbed[QIdx(_w.EmbdType)], _dEmbd, _dX),
                Push = [0u, (uint)E],
                Gx = (uint)((E + 255) / 256),
                Patch = Patch.Token,
            });

            int ssmHeads = _w.SsmTimeStepRank;
            int ssmD = _w.SsmInnerSize;
            int ssmHd = ssmHeads > 0 ? ssmD / ssmHeads : 0;
            int ssmSd = _w.SsmStateSize;
            int ssmG = _w.SsmGroupCount;
            int ssmConvDim = ssmD + 2 * ssmG * ssmSd;
            int ssmInDim = ssmD + ssmConvDim + ssmHeads; // [z | xBC | dt]
            VulkanCompute.GBuf? dSsmIn = null, dSsmXbcOut = null, dSsmY = null, dSsmZeroB = null, dSsmZeroD = null;
            if (ssmD > 0)
            {
                dSsmIn = _vk.Alloc((long)ssmInDim * 4);
                dSsmXbcOut = _vk.Alloc((long)ssmConvDim * 4);
                dSsmY = _vk.Alloc((long)ssmD * 4);
                dSsmZeroB = _vk.Alloc((long)ssmConvDim * 4);
                dSsmZeroD = _vk.Alloc((long)ssmHeads * 4);
            }

            long kvBytes = (long)nKv * _maxSeqLen * hd * 4;
            for (int i = 0; i < _w.BlockCount; i++)
            {
                var l = _w.Layers[i];
                var dAttnNorm = UploadFloats(l.AttnNormWeight, E);
                if (l.SsmOutWeight != null)
                {
                    var dInW = UploadBytes(l.QkvWeight, GgufTypeSize(l.QkvType, (long)ssmInDim * E));
                    var dOutW = UploadBytes(l.SsmOutWeight, GgufTypeSize(l.SsmOutType, (long)E * ssmD));
                    var dConvW = UploadFloats(l.SsmConv1dWeight, ssmConvDim * _w.SsmConvKernel);
                    var dConvB = l.SsmConv1dBias != null ? UploadFloats(l.SsmConv1dBias, ssmConvDim) : dSsmZeroB!;
                    var dConvSt = _vk.Alloc((long)ssmConvDim * Math.Max(1, _w.SsmConvKernel - 1) * 4);
                    var dA = UploadFloats(l.SsmAWeight, ssmHeads);
                    var dD = l.SsmDWeight != null ? UploadFloats(l.SsmDWeight, ssmHeads) : dSsmZeroD!;
                    var dDtB = l.SsmDtBias != null ? UploadFloats(l.SsmDtBias, ssmHeads) : dSsmZeroD!;
                    var dSt = _vk.Alloc((long)ssmHeads * ssmHd * ssmSd * 4);

                    AddRmsNorm(_layerOps, _dX, dAttnNorm, _dNorm, E, eps);
                    AddQmv(_layerOps, l.QkvType, dInW, _dNorm, dSsmIn!, E, ssmInDim);
                    // Note: Mamba-2 ssm_in packs [z | xBC | dt]; mamba2_conv & mamba2_rec read from dSsmIn
                    _layerOps.Add(new Op { K = _pM2Conv, Set = _vk.MakeSet(_pM2Conv, dSsmIn!, dConvW, dConvB, dConvSt, dSsmXbcOut!), Push = [(uint)ssmConvDim, (uint)_w.SsmConvKernel], Gx = (uint)((ssmConvDim + 255) / 256) });
                    _layerOps.Add(new Op { K = _pM2Rec, Set = _vk.MakeSet(_pM2Rec, dSsmXbcOut!, dSsmIn!, dSsmIn!, dA, dD, dDtB, dSt, dSsmY!), Push = [(uint)ssmHeads, (uint)ssmHd, (uint)ssmSd, (uint)ssmG, (uint)ssmD], Gx = (uint)ssmHeads });
                    if (l.SsmNormWeight != null)
                    {
                        var dSsmNorm = UploadFloats(l.SsmNormWeight, ssmD);
                        AddRmsNorm(_layerOps, dSsmY!, dSsmNorm, dSsmY!, ssmD, eps);
                    }
                    AddQmv(_layerOps, l.SsmOutType, dOutW, dSsmY!, _dProj, ssmD, E);
                    AddVec(_layerOps, _pAdd, _dX, _dProj, _dX, E);
                }
                else if (l.QWeight != null)
                {
                    var dQNorm = l.AttnQNormWeight != null ? UploadFloats(l.AttnQNormWeight, hd) : _dDummyNorm;
                    var dKNorm = l.AttnKNormWeight != null ? UploadFloats(l.AttnKNormWeight, hd) : _dDummyNorm;
                    var dQ = UploadBytes(l.QWeight, GgufTypeSize(l.QType, (long)qDim * E));
                    var dK = UploadBytes(l.KWeight, GgufTypeSize(l.KType, (long)kvDim * E));
                    var dV = UploadBytes(l.VWeight, GgufTypeSize(l.VType, (long)kvDim * E));
                    var dAttnOut = UploadBytes(l.AttnOutWeight, GgufTypeSize(l.AttnOutType, (long)E * qDim));
                    var kCache = _vk.Alloc(kvBytes);
                    var vCache = _vk.Alloc(kvBytes);

                    AddRmsNorm(_layerOps, _dX, dAttnNorm, _dNorm, E, eps);
                    AddQmv(_layerOps, l.QType, dQ, _dNorm, _dQ, E, qDim);
                    AddQmv(_layerOps, l.KType, dK, _dNorm, _dK, E, kvDim);
                    AddQmv(_layerOps, l.VType, dV, _dNorm, _dV, E, kvDim);

                    uint hasQkNorm = (l.AttnQNormWeight != null && l.AttnKNormWeight != null) ? 1u : 0u;
                    _layerOps.Add(new Op
                    {
                        K = _pGqaPrep,
                        Set = _vk.MakeSet(_pGqaPrep, _dQ, _dK, _dV, dQNorm, dKNorm, _dQOut, kCache, vCache),
                        Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, (uint)_w.RopeDimensionCount, hasQkNorm, 1u, FBits(eps), FBits(_w.RopeFreqBase)],
                        Gx = (uint)(nQ + 2 * nKv),
                        Patch = Patch.Pos,
                    });
                    _layerOps.Add(new Op
                    {
                        K = _pGqaDecode,
                        Set = _vk.MakeSet(_pGqaDecode, _dQOut, kCache, vCache, _dAttnOut),
                        Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, FBits(1.0f / MathF.Sqrt(hd))],
                        Gx = (uint)nQ,
                        Patch = Patch.KvLen,
                    });
                    AddQmv(_layerOps, l.AttnOutType, dAttnOut, _dAttnOut, _dProj, qDim, E);
                    AddVec(_layerOps, _pAdd, _dX, _dProj, _dX, E);
                }

                if (l.FfnNormWeight == null) continue;
                var dFfnNorm = UploadFloats(l.FfnNormWeight, E);

                AddRmsNorm(_layerOps, _dX, dFfnNorm, _dNorm, E, eps);
                if (l.IsMoe)
                {
                    var dRouter = UploadFloats(l.FfnGateInpWeight, nExp * E);
                    long gateSlice = GgufTypeSize(l.FfnGateExpsType, (long)moeF * E);
                    long upSlice = GgufTypeSize(l.FfnUpExpsType, (long)moeF * E);
                    long downSlice = GgufTypeSize(l.FfnDownExpsType, (long)E * moeF);
                    var dGateE = UploadBytes(l.FfnGateExpsWeight, gateSlice * nExp);
                    var dUpE = UploadBytes(l.FfnUpExpsWeight, upSlice * nExp);
                    var dDownE = UploadBytes(l.FfnDownExpsWeight, downSlice * nExp);

                    _layerOps.Add(new Op { K = _pF32!, Set = _vk.MakeSet(_pF32!, dRouter, _dNorm, dRouterLogits!), Push = [(uint)E, (uint)nExp], Gx = (uint)((nExp + 3) / 4) });
                    _layerOps.Add(new Op { K = _pTopk!, Set = _vk.MakeSet(_pTopk!, dRouterLogits!, dIdx!, dTw!), Push = [(uint)nExp, (uint)topK, _w.NormTopK ? 1u : 0u], Gx = 1 });
                    AddMoeMv(_layerOps, l.FfnGateExpsType, dGateE, dIdx!, _dNorm, dMoeGate!, E, moeF, gateSlice, 0, topK);
                    AddMoeMv(_layerOps, l.FfnUpExpsType, dUpE, dIdx!, _dNorm, dMoeUp!, E, moeF, upSlice, 0, topK);
                    AddVec(_layerOps, _pSwiGlu, dMoeGate!, dMoeUp!, dMoeAct!, topK * moeF);
                    AddMoeMv(_layerOps, l.FfnDownExpsType, dDownE, dIdx!, dMoeAct!, dMoeDown!, moeF, E, downSlice, moeF, topK);
                    _layerOps.Add(new Op
                    {
                        K = _pComb!,
                        Set = _vk.MakeSet(_pComb!, _dX, _dNorm, _dDummyNorm, dMoeDown!, dTw!, dZeroE!),
                        Push = [(uint)E, (uint)topK, 0u],
                        Gx = 1,
                    });
                }
                else
                {
                    var dGate = UploadBytes(l.FfnGateWeight, GgufTypeSize(l.FfnGateType, (long)ffnDim * E));
                    var dUp = UploadBytes(l.FfnUpWeight, GgufTypeSize(l.FfnUpType, (long)ffnDim * E));
                    var dDown = UploadBytes(l.FfnDownWeight, GgufTypeSize(l.FfnDownType, (long)E * ffnDim));
                    AddQmv(_layerOps, l.FfnGateType, dGate, _dNorm, _dGate, E, ffnDim);
                    AddQmv(_layerOps, l.FfnUpType, dUp, _dNorm, _dUp, E, ffnDim);
                    AddVec(_layerOps, _pSwiGlu, _dGate, _dUp, _dGate, ffnDim);
                    AddQmv(_layerOps, l.FfnDownType, dDown, _dGate, _dProj, ffnDim, E);
                    AddVec(_layerOps, _pAdd, _dX, _dProj, _dX, E);
                }
            }

            AddRmsNorm(_finalOps, _dX, _dOutNorm, _dNorm, E, eps);
            AddQmv(_finalOps, _w.OutType, _dOutput, _dNorm, _dLogits, E, _vocab);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static long GgufTypeSize(GgufType t, long elements) => t switch
    {
        GgufType.Q4_K => (elements / 256) * 144,
        GgufType.Q5_K => (elements / 256) * 176,
        GgufType.Q6_K => (elements / 256) * 210,
        GgufType.Q8_0 => (elements / 32) * 34,
        GgufType.IQ4_XS => (elements / 256) * 136,
        GgufType.MXFP4 => (elements / 32) * 17,
        _ => throw new NotSupportedException($"Unsupported Vulkan quant type {t}"),
    };

    private VulkanCompute.GBuf UploadBytes(byte* ptr, long bytes) { var b = _vk.Alloc(bytes); b.UploadFrom(ptr, bytes); return b; }
    private VulkanCompute.GBuf UploadFloats(float* ptr, int count) { long bytes = (long)count * 4; var b = _vk.Alloc(bytes); b.UploadFrom(ptr, bytes); return b; }

    private void AddRmsNorm(List<Op> list, VulkanCompute.GBuf x, VulkanCompute.GBuf w, VulkanCompute.GBuf o, int n, float eps) =>
        list.Add(new Op { K = _pRmsNorm, Set = _vk.MakeSet(_pRmsNorm, x, w, o), Push = [(uint)n, 1u, FBits(eps)], Gx = 1 });

    private void AddQmv(List<Op> list, GgufType t, VulkanCompute.GBuf w, VulkanCompute.GBuf x, VulkanCompute.GBuf y, int k, int m)
    {
        var kern = _pQmv[QIdx(t)];
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, w, x, y), Push = [(uint)k, (uint)m], Gx = (uint)((m + 3) / 4) });
    }

    private void AddVec(List<Op> list, VulkanCompute.Kernel kern, VulkanCompute.GBuf a, VulkanCompute.GBuf b, VulkanCompute.GBuf o, int n) =>
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, a, b, o), Push = [(uint)n], Gx = (uint)((n + 255) / 256) });

    private void AddMoeMv(List<Op> list, GgufType t, VulkanCompute.GBuf w, VulkanCompute.GBuf idx, VulkanCompute.GBuf x, VulkanCompute.GBuf y, int k, int m, long sliceBytes, int xStride, int topK)
    {
        var kern = _pMoe[QIdx(t)];
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, w, idx, x, y), Push = [(uint)k, (uint)m, (uint)(sliceBytes / 4), (uint)xStride], Gx = (uint)((m + 3) / 4), Gy = (uint)topK });
    }

    public void ResetKvCache() { }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool last = i == tokens.Length - 1;
            Forward(tokens[i], startPos + i, last && computeLogits ? logits : Span<float>.Empty, last && computeLogits);
        }
    }

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        if ((uint)pos >= (uint)_maxSeqLen) throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position exceeds MaxSeqLen ({_maxSeqLen}).");
        if ((uint)token >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(token), token, "Token id outside vocabulary.");
        bool wantLogits = computeLogits && logits.Length >= _vocab;

        _vk.Begin();
        Run(_embedOps, token, pos);
        Run(_layerOps, token, pos);
        if (wantLogits) Run(_finalOps, token, pos);
        _vk.SubmitAndWait();

        if (wantLogits)
        {
            fixed (float* p = logits) _dLogits.DownloadTo(p, (long)_vocab * 4);
        }
    }

    private void Run(List<Op> ops, int token, int pos)
    {
        foreach (var op in ops)
        {
            switch (op.Patch)
            {
                case Patch.Pos: op.Push[0] = (uint)pos; break;
                case Patch.KvLen: op.Push[0] = (uint)(pos + 1); break;
                case Patch.Token: op.Push[0] = (uint)token; break;
            }
            _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
        }
    }

    public void Dispose() { _vk.Dispose(); _ctx.Dispose(); }
}
