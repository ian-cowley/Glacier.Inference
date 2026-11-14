using Glacier.Inference.Gguf;
using Glacier.Inference.Model;

namespace Glacier.Inference.Gpu;

/// <summary>
/// Vulkan 1.1 compute runner for Gemma 4 (Dense ISWA &amp; Dual MoE) on Linux/non-Windows GPUs.
/// </summary>
public sealed unsafe class Gemma4VulkanModel : VulkanRunnerBase
{
    public static bool IsSupported(ModelWeights w, out string reason)
    {
        reason = "";
        if (w.ArchitectureFamily != UniversalArchitecture.Gemma4) { reason = "not a Gemma 4 model"; return false; }
        if (w.HeadDim > 256 || w.ValueDim != w.HeadDim) { reason = $"unsupported head dim {w.HeadDim}/{w.ValueDim}"; return false; }
        if (w.HeadCount % w.HeadCountKv != 0) { reason = "head count not divisible by KV head count"; return false; }

        foreach (var l in w.Layers)
        {
            (string Name, bool Present, GgufType T)[] mats =
            {
                ("attn_q", l.QWeight != null, l.QType), ("attn_k", l.KWeight != null, l.KType),
                ("attn_v", l.VWeight != null, l.VType), ("attn_output", l.AttnOutWeight != null, l.AttnOutType),
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

    public Gemma4VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
        : base(weights, maxSeqLen, deviceOrdinal)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Gemma 4 Vulkan runner cannot execute this model: {why}.");

        try
        {
            LoadBaseKernels(_w.IsMoe);
            var pGeGlu = _vk.LoadEmbedded("elementwise_geglu", 3, 4);
            var pScale = _vk.LoadEmbedded("gemma4_scale", 2, 12);
            var pGqaPrep = _vk.LoadEmbedded("gqa_prep", 8, 40);
            var pGqaDecode = _vk.LoadEmbedded("gqa_decode", 4, 24);
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
            _embedOps.Add(new Op { K = pScale, Set = _vk.MakeSet(pScale, dX, dX), Push = [(uint)E, 0u, FBits(MathF.Sqrt(E))], Gx = (uint)((E + 255) / 256) });

            long kvBytes = (long)nKv * _maxSeqLen * hd * 4;
            for (int i = 0; i < _w.BlockCount; i++)
            {
                var l = _w.Layers[i];
                var dAttnNorm = UploadFloats(l.AttnNormWeight, E);
                var dFfnNorm = UploadFloats(l.FfnNormWeight, E);
                var dQNorm = l.AttnQNormWeight != null ? UploadFloats(l.AttnQNormWeight, hd) : dDummyNorm;
                var dKNorm = l.AttnKNormWeight != null ? UploadFloats(l.AttnKNormWeight, hd) : dDummyNorm;
                var dQW = UploadBytes(l.QWeight, GgufTypeSize(l.QType, (long)qDim * E));
                var dKW = UploadBytes(l.KWeight, GgufTypeSize(l.KType, (long)kvDim * E));
                var dVW = l.VWeight != null ? UploadBytes(l.VWeight, GgufTypeSize(l.VType, (long)kvDim * E)) : dKW;
                var vType = l.VWeight != null ? l.VType : l.KType;
                var dAttnOutW = UploadBytes(l.AttnOutWeight, GgufTypeSize(l.AttnOutType, (long)E * qDim));
                var dGateW = UploadBytes(l.FfnGateWeight, GgufTypeSize(l.FfnGateType, (long)ffnDim * E));
                var dUpW = UploadBytes(l.FfnUpWeight, GgufTypeSize(l.FfnUpType, (long)ffnDim * E));
                var dDownW = UploadBytes(l.FfnDownWeight, GgufTypeSize(l.FfnDownType, (long)E * ffnDim));
                var kCache = _vk.Alloc(kvBytes);
                var vCache = _vk.Alloc(kvBytes);

                AddRmsNorm(_layerOps, dX, dAttnNorm, dNorm, E, eps);
                AddQmv(_layerOps, l.QType, dQW, dNorm, dQ, E, qDim);
                AddQmv(_layerOps, l.KType, dKW, dNorm, dK, E, kvDim);
                AddQmv(_layerOps, vType, dVW, dNorm, dV, E, kvDim);

                uint hasQkNorm = (l.AttnQNormWeight != null && l.AttnKNormWeight != null) ? 1u : 0u;
                float freqBase = l.IsSwa && _w.RopeFreqBaseSwa > 0 ? _w.RopeFreqBaseSwa : _w.RopeFreqBase;
                _layerOps.Add(new Op
                {
                    K = pGqaPrep,
                    Set = _vk.MakeSet(pGqaPrep, dQ, dK, dV, dQNorm, dKNorm, dQOut, kCache, vCache),
                    Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, (uint)_w.RopeDimensionCount, hasQkNorm, 1u, FBits(eps), FBits(freqBase)],
                    Gx = (uint)(nQ + 2 * nKv),
                    Patch = Patch.Pos,
                });
                _layerOps.Add(new Op
                {
                    K = pGqaDecode,
                    Set = _vk.MakeSet(pGqaDecode, dQOut, kCache, vCache, dAttnOut),
                    Push = [0u, (uint)nQ, (uint)nKv, (uint)_maxSeqLen, (uint)hd, FBits(1.0f / MathF.Sqrt(hd))],
                    Gx = (uint)nQ,
                    Patch = Patch.KvLen,
                });

                AddQmv(_layerOps, l.AttnOutType, dAttnOutW, dAttnOut, dProj, qDim, E);
                if (l.AttnPostNormWeight != null) AddRmsNorm(_layerOps, dProj, UploadFloats(l.AttnPostNormWeight, E), dProj, E, eps);
                AddVec(_layerOps, _pAdd, dX, dProj, dX, E);

                AddRmsNorm(_layerOps, dX, dFfnNorm, dNorm, E, eps);
                AddQmv(_layerOps, l.FfnGateType, dGateW, dNorm, dGate, E, ffnDim);
                AddQmv(_layerOps, l.FfnUpType, dUpW, dNorm, dUp, E, ffnDim);
                AddVec(_layerOps, pGeGlu, dGate, dUp, dGate, ffnDim);
                AddQmv(_layerOps, l.FfnDownType, dDownW, dGate, dProj, ffnDim, E);
                if (l.FfnPostNormWeight != null) AddRmsNorm(_layerOps, dProj, UploadFloats(l.FfnPostNormWeight, E), dProj, E, eps);
                else if (l.FfnPostNorm1Weight != null) AddRmsNorm(_layerOps, dProj, UploadFloats(l.FfnPostNorm1Weight, E), dProj, E, eps);
                AddVec(_layerOps, _pAdd, dX, dProj, dX, E);

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
                    AddVec(_layerOps, pGeGlu, dMoeGate!, dMoeUp!, dMoeAct!, topK * moeF);
                    AddMoeMv(_layerOps, l.FfnDownExpsType, dDownE, dIdx!, dMoeAct!, dMoeDown!, moeF, E, downSlice, moeF, topK);
                    _layerOps.Add(new Op { K = pComb!, Set = _vk.MakeSet(pComb!, dX, dNorm, dDummyNorm, dMoeDown!, dTw!, dZeroE!), Push = [(uint)E, (uint)topK, 0u], Gx = 1 });
                }
            }

            AddRmsNorm(_finalOps, dX, dOutNorm, dNorm, E, eps);
            AddQmv(_finalOps, _w.OutType, dOutput, dNorm, _dLogits, E, _vocab);
            if (_w.FinalLogitSoftcapping > 0)
                _finalOps.Add(new Op { K = pScale, Set = _vk.MakeSet(pScale, _dLogits, _dLogits), Push = [(uint)_vocab, 1u, FBits(_w.FinalLogitSoftcapping)], Gx = (uint)((_vocab + 255) / 256) });
        }
        catch
        {
            Dispose();
            throw;
        }
    }
}
