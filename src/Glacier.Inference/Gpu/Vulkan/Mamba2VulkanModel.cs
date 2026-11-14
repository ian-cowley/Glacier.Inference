using Glacier.Inference.Gguf;
using Glacier.Inference.Model;

namespace Glacier.Inference.Gpu;

/// <summary>
/// Vulkan 1.1 compute runner for Mamba-2 / Hybrid Mamba-2 + Transformer + MoE models
/// (e.g. NVIDIA Nemotron 3 Nano 30B-A3B, IBM Granite 4.0/4.1 Hybrid) on Linux/non-Windows GPUs.
/// </summary>
public sealed unsafe class Mamba2VulkanModel : VulkanRunnerBase
{
    public static bool IsSupported(ModelWeights w, out string reason)
    {
        reason = "";
        if (w.ArchitectureFamily != UniversalArchitecture.Mamba2) { reason = "not a Mamba-2 model"; return false; }
        if (w.HeadDim > 256 || w.ValueDim != w.HeadDim) { reason = $"unsupported head dim {w.HeadDim}/{w.ValueDim}"; return false; }

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
        if (!IsEmbedKernelType(w.EmbdType)) { reason = $"token embedding uses unsupported weight type {w.EmbdType}"; return false; }
        if (!IsKernelType(w.OutType)) { reason = $"output head uses unsupported weight type {w.OutType}"; return false; }
        return true;
    }

    public Mamba2VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
        : base(weights, maxSeqLen, deviceOrdinal)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Mamba-2 Vulkan runner cannot execute this model: {why}.");

        try
        {
            LoadBaseKernels(_w.IsMoe);
            var pSwiGlu = _vk.LoadEmbedded("elementwise_swiglu", 3, 4);
            var pGqaPrep = _vk.LoadEmbedded("gqa_prep", 8, 40);
            var pGqaDecode = _vk.LoadEmbedded("gqa_decode", 4, 24);
            var pM2Conv = _vk.LoadEmbedded("mamba2_conv", 5, 12);
            var pM2Rec = _vk.LoadEmbedded("mamba2_rec", 8, 28);
            VulkanCompute.Kernel? pF32 = null, pTopk = null, pComb = null;
            if (_w.IsMoe)
            {
                pF32 = _vk.LoadEmbedded("f32_matvec", 3, 8);
                pTopk = _vk.LoadEmbedded("moe_topk", 3, 12);
                pComb = _vk.LoadEmbedded("moe_combine", 6, 12);
            }

            int E = _w.EmbeddingLength, nQ = _w.HeadCount, nKv = _w.HeadCountKv, hd = _w.HeadDim;
            int qDim = nQ * hd, kvDim = nKv * hd, ffnDim = _w.FeedForwardLength;
            float eps = _w.RmsNormEps;

            var dX = _vk.Alloc((long)E * 4);
            var dNorm = _vk.Alloc((long)E * 4);
            var dQ = _vk.Alloc((long)qDim * 4);
            var dK = _vk.Alloc((long)kvDim * 4);
            var dV = _vk.Alloc((long)kvDim * 4);
            var dQOut = _vk.Alloc((long)qDim * 4);
            var dAttnOut = _vk.Alloc((long)qDim * 4);
            var dProj = _vk.Alloc((long)E * 4);
            var dGate = _vk.Alloc((long)ffnDim * 4);
            var dUp = _vk.Alloc((long)ffnDim * 4);
            _dLogits = _vk.Alloc((long)_vocab * 4);
            var dDummyNorm = _vk.Alloc(4);

            int nExp = _w.ExpertCount, topK = _w.ExpertUsedCount, moeF = _w.ExpertFeedForwardLength;
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

            var dEmbd = UploadBytes(_w.EmbdWeight, GgufTypeSize(_w.EmbdType, (long)_vocab * E));
            var dOutNorm = UploadFloats(_w.OutNormWeight, E);
            var dOutput = UploadBytes(_w.OutWeight, GgufTypeSize(_w.OutType, (long)_vocab * E));

            _embedOps.Add(new Op { K = _pEmbed[QIdx(_w.EmbdType)], Set = _vk.MakeSet(_pEmbed[QIdx(_w.EmbdType)], dEmbd, dX), Push = [(uint)E, 0u], Gx = (uint)((E + 255) / 256), Patch = Patch.Token });

            int ssmHeads = _w.SsmTimeStepRank, ssmD = _w.SsmInnerSize, ssmHd = ssmHeads > 0 ? ssmD / ssmHeads : 0;
            int ssmSd = _w.SsmStateSize, ssmG = _w.SsmGroupCount;
            int ssmConvDim = ssmD + 2 * ssmG * ssmSd;
            int ssmInDim = ssmD + ssmConvDim + ssmHeads; // [z (0..ssmD) | xBC (ssmD..+ssmConvDim) | dt (+ssmHeads)]
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
                    var dConvSt = AllocStateZero((long)ssmConvDim * Math.Max(1, _w.SsmConvKernel - 1) * 4);
                    var dA = UploadFloats(l.SsmAWeight, ssmHeads);
                    var dD = l.SsmDWeight != null ? UploadFloats(l.SsmDWeight, ssmHeads) : dSsmZeroD!;
                    var dDtB = l.SsmDtBias != null ? UploadFloats(l.SsmDtBias, ssmHeads) : dSsmZeroD!;
                    var dSt = AllocStateZero((long)ssmHeads * ssmHd * ssmSd * 4);

                    AddRmsNorm(_layerOps, dX, dAttnNorm, dNorm, E, eps);
                    AddQmv(_layerOps, l.QkvType, dInW, dNorm, dSsmIn!, E, ssmInDim);
                    _layerOps.Add(new Op { K = pM2Conv, Set = _vk.MakeSet(pM2Conv, dSsmIn!, dConvW, dConvB, dConvSt, dSsmXbcOut!), Push = [(uint)ssmConvDim, (uint)_w.SsmConvKernel, (uint)ssmD], Gx = (uint)((ssmConvDim + 255) / 256) });
                    _layerOps.Add(new Op { K = pM2Rec, Set = _vk.MakeSet(pM2Rec, dSsmXbcOut!, dSsmIn!, dSsmIn!, dA, dD, dDtB, dSt, dSsmY!), Push = [(uint)ssmHeads, (uint)ssmHd, (uint)ssmSd, (uint)ssmG, (uint)ssmD, 0u, (uint)(ssmD + ssmConvDim)], Gx = (uint)ssmHeads });
                    if (l.SsmNormWeight != null) AddRmsNorm(_layerOps, dSsmY!, UploadFloats(l.SsmNormWeight, ssmD), dSsmY!, ssmD, eps);
                    AddQmv(_layerOps, l.SsmOutType, dOutW, dSsmY!, dProj, ssmD, E);
                    AddVec(_layerOps, _pAdd, dX, dProj, dX, E);
                }
                else if (l.QWeight != null)
                {
                    var dQNorm = l.AttnQNormWeight != null ? UploadFloats(l.AttnQNormWeight, hd) : dDummyNorm;
                    var dKNorm = l.AttnKNormWeight != null ? UploadFloats(l.AttnKNormWeight, hd) : dDummyNorm;
                    var dQW = UploadBytes(l.QWeight, GgufTypeSize(l.QType, (long)qDim * E));
                    var dKW = UploadBytes(l.KWeight, GgufTypeSize(l.KType, (long)kvDim * E));
                    var dVW = UploadBytes(l.VWeight, GgufTypeSize(l.VType, (long)kvDim * E));
                    var dAttnOutW = UploadBytes(l.AttnOutWeight, GgufTypeSize(l.AttnOutType, (long)E * qDim));
                    var kCache = _vk.Alloc(kvBytes);
                    var vCache = _vk.Alloc(kvBytes);

                    AddRmsNorm(_layerOps, dX, dAttnNorm, dNorm, E, eps);
                    AddQmv(_layerOps, l.QType, dQW, dNorm, dQ, E, qDim);
                    AddQmv(_layerOps, l.KType, dKW, dNorm, dK, E, kvDim);
                    AddQmv(_layerOps, l.VType, dVW, dNorm, dV, E, kvDim);

                    uint hasQkNorm = (l.AttnQNormWeight != null && l.AttnKNormWeight != null) ? 1u : 0u;
                    _layerOps.Add(new Op { K = pGqaPrep, Set = _vk.MakeSet(pGqaPrep, dQ, dK, dV, dQNorm, dKNorm, dQOut, kCache, vCache), Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, (uint)_w.RopeDimensionCount, hasQkNorm, 1u, FBits(eps), FBits(_w.RopeFreqBase)], Gx = (uint)(nQ + 2 * nKv), Patch = Patch.Pos });
                    _layerOps.Add(new Op { K = pGqaDecode, Set = _vk.MakeSet(pGqaDecode, dQOut, kCache, vCache, dAttnOut), Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, FBits(1.0f / MathF.Sqrt(hd))], Gx = (uint)nQ, Patch = Patch.KvLen });
                    AddQmv(_layerOps, l.AttnOutType, dAttnOutW, dAttnOut, dProj, qDim, E);
                    AddVec(_layerOps, _pAdd, dX, dProj, dX, E);
                }

                if (l.FfnNormWeight == null) continue;
                var dFfnNorm = UploadFloats(l.FfnNormWeight, E);
                AddRmsNorm(_layerOps, dX, dFfnNorm, dNorm, E, eps);
                if (l.IsMoe)
                {
                    var dRouter = UploadFloats(l.FfnGateInpWeight, nExp * E);
                    long gateSlice = GgufTypeSize(l.FfnGateExpsType, (long)moeF * E);
                    long upSlice = GgufTypeSize(l.FfnUpExpsType, (long)moeF * E);
                    long downSlice = GgufTypeSize(l.FfnDownExpsType, (long)E * moeF);
                    var dGateE = UploadBytes(l.FfnGateExpsWeight, gateSlice * nExp);
                    var dUpE = UploadBytes(l.FfnUpExpsWeight, upSlice * nExp);
                    var dDownE = UploadBytes(l.FfnDownExpsWeight, downSlice * nExp);

                    _layerOps.Add(new Op { K = pF32!, Set = _vk.MakeSet(pF32!, dRouter, dNorm, dRouterLogits!), Push = [(uint)E, (uint)nExp], Gx = (uint)((nExp + 3) / 4) });
                    _layerOps.Add(new Op { K = pTopk!, Set = _vk.MakeSet(pTopk!, dRouterLogits!, dIdx!, dTw!), Push = [(uint)nExp, (uint)topK, _w.NormTopK ? 1u : 0u], Gx = 1 });
                    AddMoeMv(_layerOps, l.FfnGateExpsType, dGateE, dIdx!, dNorm, dMoeGate!, E, moeF, gateSlice, 0, topK);
                    AddMoeMv(_layerOps, l.FfnUpExpsType, dUpE, dIdx!, dNorm, dMoeUp!, E, moeF, upSlice, 0, topK);
                    AddVec(_layerOps, pSwiGlu, dMoeGate!, dMoeUp!, dMoeAct!, topK * moeF);
                    AddMoeMv(_layerOps, l.FfnDownExpsType, dDownE, dIdx!, dMoeAct!, dMoeDown!, moeF, E, downSlice, moeF, topK);
                    _layerOps.Add(new Op { K = pComb!, Set = _vk.MakeSet(pComb!, dX, dNorm, dDummyNorm, dMoeDown!, dTw!, dZeroE!), Push = [(uint)E, (uint)topK, 0u], Gx = 1 });
                }
                else
                {
                    var dGateW = UploadBytes(l.FfnGateWeight, GgufTypeSize(l.FfnGateType, (long)ffnDim * E));
                    var dUpW = UploadBytes(l.FfnUpWeight, GgufTypeSize(l.FfnUpType, (long)ffnDim * E));
                    var dDownW = UploadBytes(l.FfnDownWeight, GgufTypeSize(l.FfnDownType, (long)E * ffnDim));
                    AddQmv(_layerOps, l.FfnGateType, dGateW, dNorm, dGate, E, ffnDim);
                    AddQmv(_layerOps, l.FfnUpType, dUpW, dNorm, dUp, E, ffnDim);
                    AddVec(_layerOps, pSwiGlu, dGate, dUp, dGate, ffnDim);
                    AddQmv(_layerOps, l.FfnDownType, dDownW, dGate, dProj, ffnDim, E);
                    AddVec(_layerOps, _pAdd, dX, dProj, dX, E);
                }
            }

            AddRmsNorm(_finalOps, dX, dOutNorm, dNorm, E, eps);
            AddQmv(_finalOps, _w.OutType, dOutput, dNorm, _dLogits, E, _vocab);
        }
        catch
        {
            Dispose();
            throw;
        }
    }
}
