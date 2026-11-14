using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;

namespace Glacier.Inference.Gpu;

/// <summary>
/// Vulkan 1.1 compute runner for standard Transformer models (Dense GQA; MoE &amp; MLA to follow)
/// on Linux/non-Windows GPUs (AMD RADV, Intel ANV, NVIDIA).
/// </summary>
public sealed unsafe class Qwen2VulkanModel : ID3D12Model
{
    private enum Patch { None, Pos, KvLen, Token, DsaSel }

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

    private readonly VulkanCompute.Kernel _pRmsNorm, _pAdd, _pSwiGlu, _pGqaPrep, _pGqaDecode;
    private readonly VulkanCompute.Kernel? _pF32, _pTopk, _pComb, _pMlaPrep, _pMlaDecode, _pDsaTopk, _pMlaSparseDecode, _pShortConv;
    private readonly VulkanCompute.Kernel[] _pEmbed = new VulkanCompute.Kernel[4];
    private readonly VulkanCompute.Kernel[] _pQmv = new VulkanCompute.Kernel[6];
    private readonly VulkanCompute.Kernel[] _pMoe = new VulkanCompute.Kernel[6];

    public string DeviceName => _ctx.DeviceName;
    public int MaxSeqLen => _maxSeqLen;

    public static bool IsSupported(ModelWeights w, out string reason)
    {
        reason = "";
        if (w.IsHybridSsm) { reason = "hybrid Gated DeltaNet models use Qwen35VulkanModel"; return false; }
        if (w.ArchitectureFamily == UniversalArchitecture.Gemma4) { reason = "Gemma 4 is not yet supported on Vulkan"; return false; }
        if (w.IsMla && (w.QkNopeHeadDim + w.RopeDimensionCount > 256 || w.ValueDim > 256)) { reason = "MLA head dim > 256"; return false; }
        if (!w.IsMla && (w.HeadDim > 256 || w.ValueDim != w.HeadDim)) { reason = $"unsupported head dim {w.HeadDim}/{w.ValueDim}"; return false; }
        if (w.HasRopeFreqs) { reason = "custom RoPE frequency table"; return false; }
        if (w.HeadCount % w.HeadCountKv != 0) { reason = "head count not divisible by KV head count"; return false; }

        foreach (var l in w.Layers)
        {
            if (l.HasFusedQkv) { reason = "fused QKV not yet supported on Vulkan"; return false; }
            if (l.HasFusedGateUp) { reason = "fused gate/up FFN not yet supported on Vulkan"; return false; }
            if (l.QBias != null || l.KBias != null || l.VBias != null || l.AttnOutBias != null) { reason = "attention biases not yet supported on Vulkan"; return false; }

            (string Name, bool Present, GgufType T)[] mats =
            {
                ("attn_q", l.QWeight != null, l.QType), ("attn_k", l.KWeight != null, l.KType),
                ("attn_v", l.VWeight != null, l.VType), ("attn_output", l.AttnOutWeight != null, l.AttnOutType),
                ("attn_kv_a_mqa", l.AttnKvAMqaWeight != null, l.AttnKvAMqaType), ("attn_kv_b", l.AttnKvBWeight != null, l.AttnKvBType),
                ("shortconv_in", l.QkvWeight != null, l.QkvType), ("shortconv_out", l.SsmOutWeight != null, l.SsmOutType),
                ("ffn_gate", l.FfnGateWeight != null, l.FfnGateType), ("ffn_up", l.FfnUpWeight != null, l.FfnUpType), ("ffn_down", l.FfnDownWeight != null, l.FfnDownType),
                ("ffn_gate_exps", l.FfnGateExpsWeight != null, l.FfnGateExpsType), ("ffn_up_exps", l.FfnUpExpsWeight != null, l.FfnUpExpsType), ("ffn_down_exps", l.FfnDownExpsWeight != null, l.FfnDownExpsType),
                ("ffn_gate_shexp", l.FfnGateShexpWeight != null, l.FfnGateShexpType), ("ffn_up_shexp", l.FfnUpShexpWeight != null, l.FfnUpShexpType), ("ffn_down_shexp", l.FfnDownShexpWeight != null, l.FfnDownShexpType),
            };
            foreach (var (name, present, t) in mats)
                if (present && !IsKernelType(t)) { reason = $"tensor '{name}' uses unsupported weight type {t} (supported: Q4_K, Q5_K, Q6_K, Q8_0)"; return false; }
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

    public Qwen2VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Vulkan runner cannot execute this model: {why}.");

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
            if (_w.ArchitectureFamily == UniversalArchitecture.Lfm2)
            {
                _pShortConv = _vk.LoadEmbedded("shortconv", 4, 8);
            }
            if (_w.IsMla)
            {
                _pMlaPrep = _vk.LoadEmbedded("mla_prep", 6, 36);
                _pMlaDecode = _vk.LoadEmbedded("mla_decode", 4, 24);
                if (_w.IsDsa)
                {
                    _pDsaTopk = _vk.LoadEmbedded("dsa_topk", 4, 16);
                    _pMlaSparseDecode = _vk.LoadEmbedded("mla_sparse_decode", 5, 24);
                }
            }

            int E = _w.EmbeddingLength;
            int nQ = _w.HeadCount;
            int nKv = _w.HeadCountKv;
            int hd = _w.HeadDim;
            int mlaKDim = _w.QkNopeHeadDim + _w.RopeDimensionCount;
            int mlaVDim = _w.ValueDim;
            int qDim = _w.IsMla ? nQ * mlaKDim : nQ * hd;
            int kvDim = nKv * hd;
            int attnOutDim = _w.IsMla ? nQ * mlaVDim : qDim;
            int ffnDim = _w.FeedForwardLength;
            float eps = _w.RmsNormEps;
            float attnScale = 1.0f / MathF.Sqrt(hd);
            uint isNeox = (_w.ArchitectureFamily is UniversalArchitecture.Qwen or UniversalArchitecture.HybridSsm) ? 1u : 0u;

            _dX = _vk.Alloc((long)E * 4);
            _dNorm = _vk.Alloc((long)E * 4);
            _dQ = _vk.Alloc((long)qDim * 4);
            _dK = _vk.Alloc((long)kvDim * 4);
            _dV = _vk.Alloc((long)kvDim * 4);
            _dQOut = _vk.Alloc((long)qDim * 4);
            _dAttnOut = _vk.Alloc((long)attnOutDim * 4);
            _dProj = _vk.Alloc((long)E * 4);
            _dGate = _vk.Alloc((long)ffnDim * 4);
            _dUp = _vk.Alloc((long)ffnDim * 4);
            _dLogits = _vk.Alloc((long)_vocab * 4);
            _dDummyNorm = _vk.Alloc(4);

            int nExp = _w.ExpertCount;
            int topK = _w.ExpertUsedCount;
            int moeF = _w.ExpertFeedForwardLength;
            int shF = _w.ExpertSharedCount * moeF;
            VulkanCompute.GBuf? dRouterLogits = null, dIdx = null, dTw = null;
            VulkanCompute.GBuf? dMoeGate = null, dMoeUp = null, dMoeAct = null, dMoeDown = null;
            VulkanCompute.GBuf? dShGate = null, dShUp = null, dShAct = null, dShDown = null, dZeroE = null;
            VulkanCompute.GBuf? dKvA = null, dKvNormBuf = null, dKvB = null, dIdxQ = null, dIdxW = null, dTopIdx = null;
            VulkanCompute.GBuf? dScIn = null, dScOut = null;
            if (_w.ArchitectureFamily == UniversalArchitecture.Lfm2)
            {
                dScIn = _vk.Alloc((long)E * 3 * 4);
                dScOut = _vk.Alloc((long)E * 4);
            }
            if (_w.IsMla)
            {
                dKvA = _vk.Alloc((long)(_w.KvLoraRank + _w.RopeDimensionCount) * 4);
                dKvNormBuf = _vk.Alloc((long)_w.KvLoraRank * 4);
                dKvB = _vk.Alloc((long)nQ * (_w.QkNopeHeadDim + mlaVDim) * 4);
                if (_w.IsDsa)
                {
                    dIdxQ = _vk.Alloc((long)_w.DsaIndexHeadCount * _w.DsaIndexHeadDim * 4);
                    dIdxW = _vk.Alloc((long)_w.DsaIndexHeadCount * 4);
                    dTopIdx = _vk.Alloc((long)_w.DsaTopK * 4);
                }
            }
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
                if (shF > 0)
                {
                    dShGate = _vk.Alloc((long)shF * 4);
                    dShUp = _vk.Alloc((long)shF * 4);
                    dShAct = _vk.Alloc((long)shF * 4);
                }
                dShDown = _vk.Alloc((long)E * 4);
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

            long kvBytes = (long)nKv * _maxSeqLen * hd * 4;
            for (int i = 0; i < _w.BlockCount; i++)
            {
                var l = _w.Layers[i];
                var dAttnNorm = UploadFloats(l.AttnNormWeight, E);
                var dFfnNorm = UploadFloats(l.FfnNormWeight, E);
                var dAttnOut = l.AttnOutWeight != null ? UploadBytes(l.AttnOutWeight, GgufTypeSize(l.AttnOutType, (long)E * attnOutDim)) : null!;

                AddRmsNorm(_layerOps, _dX, dAttnNorm, _dNorm, E, eps);
                if (l.SsmConv1dWeight != null && l.SsmOutWeight != null)
                {
                    int scK = Math.Max(1, _w.SsmConvKernel > 0 ? _w.SsmConvKernel : 3);
                    var dInW = UploadBytes(l.QkvWeight, GgufTypeSize(l.QkvType, (long)E * 3 * E));
                    var dConvW = UploadFloats(l.SsmConv1dWeight, E * scK);
                    var dConvSt = _vk.Alloc((long)E * Math.Max(1, scK - 1) * 4);
                    var dOutW = UploadBytes(l.SsmOutWeight, GgufTypeSize(l.SsmOutType, (long)E * E));
                    AddQmv(_layerOps, l.QkvType, dInW, _dNorm, dScIn!, E, E * 3);
                    _layerOps.Add(new Op { K = _pShortConv!, Set = _vk.MakeSet(_pShortConv!, dScIn!, dConvW, dConvSt, dScOut!), Push = [(uint)E, (uint)scK], Gx = (uint)((E + 255) / 256) });
                    AddQmv(_layerOps, l.SsmOutType, dOutW, dScOut!, _dProj, E, E);
                }
                else if (_w.IsMla)
                {
                    int kvMqaDim = _w.KvLoraRank + _w.RopeDimensionCount;
                    int decompKvDim = nQ * (_w.QkNopeHeadDim + mlaVDim);
                    var dQ = UploadBytes(l.QWeight, GgufTypeSize(l.QType, (long)qDim * E));
                    var dKvAMqa = UploadBytes(l.AttnKvAMqaWeight, GgufTypeSize(l.AttnKvAMqaType, (long)kvMqaDim * E));
                    var dKvANorm = UploadFloats(l.AttnKvANormWeight, _w.KvLoraRank);
                    var dKvBW = UploadBytes(l.AttnKvBWeight, GgufTypeSize(l.AttnKvBType, (long)decompKvDim * _w.KvLoraRank));
                    var mlaKCache = _vk.Alloc((long)nQ * _maxSeqLen * mlaKDim * 4);
                    var mlaVCache = _vk.Alloc((long)nQ * _maxSeqLen * mlaVDim * 4);

                    AddQmv(_layerOps, l.QType, dQ, _dNorm, _dQ, E, qDim);
                    AddQmv(_layerOps, l.AttnKvAMqaType, dKvAMqa, _dNorm, dKvA!, E, kvMqaDim);
                    AddRmsNorm(_layerOps, dKvA!, dKvANorm, dKvNormBuf!, _w.KvLoraRank, eps);
                    AddQmv(_layerOps, l.AttnKvBType, dKvBW, dKvNormBuf!, dKvB!, _w.KvLoraRank, decompKvDim);
                    _layerOps.Add(new Op
                    {
                        K = _pMlaPrep!,
                        Set = _vk.MakeSet(_pMlaPrep!, _dQ, dKvB!, dKvA!, _dQOut, mlaKCache, mlaVCache),
                        Push = [0u, (uint)nQ, (uint)_maxSeqLen, (uint)_w.QkNopeHeadDim, (uint)_w.RopeDimensionCount, (uint)mlaVDim, (uint)_w.KvLoraRank, isNeox, FBits(_w.RopeFreqBase)],
                        Gx = (uint)nQ,
                        Patch = Patch.Pos,
                    });
                    float mlaScale = 1.0f / MathF.Sqrt(mlaKDim);
                    if (l.IsDsa)
                    {
                        int nIdxH = _w.DsaIndexHeadCount;
                        int idxD = _w.DsaIndexHeadDim;
                        int dsaK = _w.DsaTopK;
                        var dIdxQW = UploadBytes(l.AttnIdxQWeight, GgufTypeSize(l.AttnIdxQType, (long)nIdxH * idxD * E));
                        var dIdxKW = UploadBytes(l.AttnIdxKWeight, GgufTypeSize(l.AttnIdxKType, (long)idxD * E));
                        var dIdxWW = UploadBytes(l.AttnIdxWWeight, GgufTypeSize(l.AttnIdxWType, (long)nIdxH * E));
                        var dIdxKCache = _vk.Alloc((long)_maxSeqLen * idxD * 4);
                        // Compute indexer Q, K (for current pos at offset 0..idxD-1? Note: for full seq, K is cached), and head weights
                        AddQmv(_layerOps, l.AttnIdxQType, dIdxQW, _dNorm, dIdxQ!, E, nIdxH * idxD);
                        AddQmv(_layerOps, l.AttnIdxKType, dIdxKW, _dNorm, dIdxKCache, E, idxD);
                        AddQmv(_layerOps, l.AttnIdxWType, dIdxWW, _dNorm, dIdxW!, E, nIdxH);
                        _layerOps.Add(new Op
                        {
                            K = _pDsaTopk!,
                            Set = _vk.MakeSet(_pDsaTopk!, dIdxQ!, dIdxKCache, dIdxW!, dTopIdx!),
                            Push = [0u, (uint)nIdxH, (uint)idxD, (uint)dsaK],
                            Gx = 1,
                            Patch = Patch.KvLen,
                        });
                        _layerOps.Add(new Op
                        {
                            K = _pMlaSparseDecode!,
                            Set = _vk.MakeSet(_pMlaSparseDecode!, _dQOut, mlaKCache, mlaVCache, dTopIdx!, _dAttnOut),
                            Push = [0u, (uint)nQ, (uint)_maxSeqLen, (uint)mlaKDim, (uint)mlaVDim, FBits(mlaScale)],
                            Gx = (uint)nQ,
                            Patch = Patch.DsaSel,
                        });
                    }
                    else
                    {
                        _layerOps.Add(new Op
                        {
                            K = _pMlaDecode!,
                            Set = _vk.MakeSet(_pMlaDecode!, _dQOut, mlaKCache, mlaVCache, _dAttnOut),
                            Push = [0u, (uint)nQ, (uint)_maxSeqLen, (uint)mlaKDim, (uint)mlaVDim, FBits(mlaScale)],
                            Gx = (uint)nQ,
                            Patch = Patch.DsaSel,
                        });
                    }
                    AddQmv(_layerOps, l.AttnOutType, dAttnOut, _dAttnOut, _dProj, attnOutDim, E);
                }
                else
                {
                    var dQNorm = l.AttnQNormWeight != null ? UploadFloats(l.AttnQNormWeight, hd) : _dDummyNorm;
                    var dKNorm = l.AttnKNormWeight != null ? UploadFloats(l.AttnKNormWeight, hd) : _dDummyNorm;
                    var dQ = UploadBytes(l.QWeight, GgufTypeSize(l.QType, (long)qDim * E));
                    var dK = UploadBytes(l.KWeight, GgufTypeSize(l.KType, (long)kvDim * E));
                    var dV = UploadBytes(l.VWeight, GgufTypeSize(l.VType, (long)kvDim * E));
                    var kCache = _vk.Alloc(kvBytes);
                    var vCache = _vk.Alloc(kvBytes);

                    AddQmv(_layerOps, l.QType, dQ, _dNorm, _dQ, E, qDim);
                    AddQmv(_layerOps, l.KType, dK, _dNorm, _dK, E, kvDim);
                    AddQmv(_layerOps, l.VType, dV, _dNorm, _dV, E, kvDim);

                    uint hasQkNorm = (l.AttnQNormWeight != null && l.AttnKNormWeight != null) ? 1u : 0u;
                    _layerOps.Add(new Op
                    {
                        K = _pGqaPrep,
                        Set = _vk.MakeSet(_pGqaPrep, _dQ, _dK, _dV, dQNorm, dKNorm, _dQOut, kCache, vCache),
                        Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, (uint)_w.RopeDimensionCount, hasQkNorm, isNeox, FBits(eps), FBits(_w.RopeFreqBase)],
                        Gx = (uint)(nQ + 2 * nKv),
                        Patch = Patch.Pos,
                    });
                    _layerOps.Add(new Op
                    {
                        K = _pGqaDecode,
                        Set = _vk.MakeSet(_pGqaDecode, _dQOut, kCache, vCache, _dAttnOut),
                        Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, FBits(attnScale)],
                        Gx = (uint)nQ,
                        Patch = Patch.KvLen,
                    });
                    AddQmv(_layerOps, l.AttnOutType, dAttnOut, _dAttnOut, _dProj, qDim, E);
                }
                AddVec(_layerOps, _pAdd, _dX, _dProj, _dX, E);

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

                    bool hasShExp = l.FfnGateShexpWeight != null && shF > 0;
                    if (hasShExp)
                    {
                        var dSgW = UploadBytes(l.FfnGateShexpWeight, GgufTypeSize(l.FfnGateShexpType, (long)shF * E));
                        var dSuW = UploadBytes(l.FfnUpShexpWeight, GgufTypeSize(l.FfnUpShexpType, (long)shF * E));
                        var dSdW = UploadBytes(l.FfnDownShexpWeight, GgufTypeSize(l.FfnDownShexpType, (long)E * shF));
                        AddQmv(_layerOps, l.FfnGateShexpType, dSgW, _dNorm, dShGate!, E, shF);
                        AddQmv(_layerOps, l.FfnUpShexpType, dSuW, _dNorm, dShUp!, E, shF);
                        AddVec(_layerOps, _pSwiGlu, dShGate!, dShUp!, dShAct!, shF);
                        AddQmv(_layerOps, l.FfnDownShexpType, dSdW, dShAct!, dShDown!, shF, E);
                    }
                    bool hasShGate = l.FfnGateInpShexpWeight != null;
                    var dShGateInp = hasShGate ? UploadFloats(l.FfnGateInpShexpWeight, E) : _dDummyNorm;
                    _layerOps.Add(new Op
                    {
                        K = _pComb!,
                        Set = _vk.MakeSet(_pComb!, _dX, _dNorm, dShGateInp, dMoeDown!, dTw!, hasShExp ? dShDown! : dZeroE!),
                        Push = [(uint)E, (uint)topK, hasShGate ? 1u : 0u],
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
                case Patch.DsaSel: op.Push[0] = (uint)Math.Min(pos + 1, _w.DsaTopK); break;
            }
            _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
        }
    }

    public void Dispose() { _vk.Dispose(); _ctx.Dispose(); }
}
