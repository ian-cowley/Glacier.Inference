namespace Glacier.Inference.Model;

using System;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;

/// <summary>
/// Zero-copy memory-mapped tensor pointers for a single physical Inkling 2 layer (0..23).
/// </summary>
public sealed unsafe class Inkling2LayerWeights
{
    public bool IsSwa { get; init; }
    public int HeadCount { get; init; }
    public int HeadCountKv { get; init; }
    public int HeadDim { get; init; }
    public int RelExtent { get; init; }

    // Norms
    public float* AttnNorm { get; init; }
    public float* AttnQInpNorm { get; init; }
    public float* AttnKInpNorm { get; init; }
    public float* AttnVInpNorm { get; init; }
    public float* FfnNorm { get; init; }

    // Block-Attention Residual Queries
    public float* AttnResQuery { get; init; } // [n_embd] (SWA) or [n_embd, 3] (Global); null on layer 0
    public float* FfnResQuery { get; init; }  // [n_embd]

    // Attention projections
    public byte* Wq { get; init; }
    public GgufType WqType { get; init; }
    public byte* Wk { get; init; }
    public GgufType WkType { get; init; }
    public byte* Wv { get; init; }
    public GgufType WvType { get; init; }
    public byte* Wo { get; init; }
    public GgufType WoType { get; init; }
    public byte* AttnGate { get; init; }
    public GgufType AttnGateType { get; init; }

    public float* AttnQClip { get; init; }      // [headDim, nHead]
    public byte* AttnSinkProj { get; init; }    // [n_embd, nHead]
    public GgufType AttnSinkProjType { get; init; }

    // Relative-position bias
    public byte* AttnRel { get; init; }         // [n_embd, nRel * nHead]
    public GgufType AttnRelType { get; init; }
    public float* AttnRelBias { get; init; }    // [nRel, nExtent]
    public float* AttnRelClip { get; init; }    // [nRel, nHead]

    // Short convolutions (l_cache = 4)
    public float* AttnKConv { get; init; }      // [nConv, kvDim]
    public float* AttnVConv { get; init; }      // [nConv, kvDim]
    public float* AttnPostConv { get; init; }   // [nConv, n_embd]
    public float* FfnPostConv { get; init; }    // [nConv, n_embd]
    public byte* AttnKConvProj { get; init; }   // Global only: [n_embd, nConv * nHeadKv]
    public GgufType AttnKConvProjType { get; init; }
    public byte* AttnVConvProj { get; init; }   // Global only: [n_embd, nConv * nHeadKv]
    public GgufType AttnVConvProjType { get; init; }

    // MoE Router & 128 Experts
    public byte* FfnGateInp { get; init; }      // [n_embd, nExpert]
    public GgufType FfnGateInpType { get; init; }
    public float* ExpProbsBias { get; init; }   // [nExpert]
    public byte* FfnGateUpExps { get; init; }   // [n_embd, 2*nFfExp, nExpert]
    public GgufType FfnGateUpExpsType { get; init; }
    public float* FfnGateUpExpsScale { get; init; } // Optional [nExpert]
    public byte* FfnDownExps { get; init; }     // [nFfExp, n_embd, nExpert]
    public GgufType FfnDownExpsType { get; init; }
    public float* FfnDownExpsScale { get; init; }   // Optional [nExpert]

    // Cross-pass Loop Control Gates (core layers 6..17 only: 0=Q/Attn, 1=K, 2=V, 3=FFN/MLP)
    public bool HasLoopGates { get; init; }
    public float* LoopMsgQuery { get; init; }   // Optional per-layer [n_embd, 4]
    public byte*[] LoopCtrlDown { get; init; } = new byte*[4];
    public GgufType[] LoopCtrlDownType { get; init; } = new GgufType[4];
    public byte*[] LoopCtrlGain { get; init; } = new byte*[4];
    public GgufType[] LoopCtrlGainType { get; init; } = new GgufType[4];
    public byte*[] LoopCtrlForget { get; init; } = new byte*[4];
    public GgufType[] LoopCtrlForgetType { get; init; } = new GgufType[4];
}

/// <summary>
/// Full hyperparameter and tensor weight table for Thinking Machines Lab's Inkling 2 Mini (3-Pass Looped MoE).
/// </summary>
public sealed unsafe class Inkling2Weights : IDisposable
{
    private readonly List<nint> _allocatedBuffers = new();

    public int PhysicalLayerCount { get; }
    public int LogicalLayerCount { get; }
    public int CoreStart { get; }
    public int CoreEnd { get; }
    public int NumLoops { get; }
    public int GateRank { get; }

    public int EmbeddingDim { get; }
    public int VocabSizePadded { get; }
    public int VocabSizeUnpadded { get; }
    public int ExpertCount { get; }
    public int ExpertUsedCount { get; }
    public int ExpertFfnDim { get; }
    public float ExpertWeightsScale { get; }

    public float RmsNormEps { get; }
    public float QkClip { get; }
    public int AttnResBlockSize { get; }
    public int ShortConvLCache { get; }
    public int RelativeLength { get; }
    public int RelativeExtent { get; }
    public float RelativeClip { get; }
    public int SlidingWindow { get; }
    public int LogScaleFloor { get; }
    public float LogScaleAlpha { get; }
    public float SituBeta { get; }
    public float SituLinearBeta { get; }
    public float FinalLogitSoftcapping { get; }

    // Global tensors
    public byte* TokenEmbd { get; }
    public GgufType TokenEmbdType { get; }
    public float* TokenEmbdNorm { get; }
    public byte* OutputWeight { get; }
    public GgufType OutputWeightType { get; }
    public float* OutputNorm { get; }
    public float* OutputResQuery { get; }
    public float* LoopResQuery { get; }     // Optional global [n_embd, 2 * coreLen] (attn, mlp)
    public float* LoopKvResQuery { get; }   // Optional global [n_embd, 2 * coreLen] (k, v)

    public Inkling2LayerWeights[] PhysicalLayers { get; }

    public Inkling2Weights(GgufFile gguf)
    {
        string arch = gguf.Architecture;
        EmbeddingDim = gguf.EmbeddingLength;
        VocabSizePadded = (int)gguf.GetMetadataUInt32($"{arch}.vocab_size", 201024);
        VocabSizeUnpadded = (int)gguf.GetMetadataUInt32($"{arch}.vocab_size_unpadded", (uint)VocabSizePadded);
        ExpertCount = gguf.ExpertCount > 0 ? gguf.ExpertCount : 128;
        ExpertUsedCount = gguf.ExpertUsedCount > 0 ? gguf.ExpertUsedCount : 8;
        ExpertFfnDim = gguf.ExpertFeedForwardLength > 0 ? gguf.ExpertFeedForwardLength : gguf.FeedForwardLength;
        ExpertWeightsScale = gguf.GetMetadataSingle($"{arch}.expert_weights_scale", 8.0f);

        RmsNormEps = gguf.RmsNormEps;
        QkClip = gguf.GetMetadataSingle($"{arch}.attention.qk_clip", 22.627417f);
        AttnResBlockSize = (int)gguf.GetMetadataUInt32($"{arch}.attn_res.block_size", 6);
        ShortConvLCache = (int)gguf.GetMetadataUInt32($"{arch}.shortconv.l_cache", 4);
        RelativeLength = (int)gguf.GetMetadataUInt32($"{arch}.attention.relative_length", 16);
        RelativeExtent = (int)gguf.GetMetadataUInt32($"{arch}.attention.relative_extent", 1024);
        RelativeClip = gguf.GetMetadataSingle($"{arch}.attention.relative_clip", 8.0f);
        SlidingWindow = (int)gguf.GetMetadataUInt32($"{arch}.attention.sliding_window", 1024);
        LogScaleFloor = (int)gguf.GetMetadataUInt32($"{arch}.attention.log_scale_floor", 8192);
        LogScaleAlpha = gguf.GetMetadataSingle($"{arch}.attention.log_scale_alpha", 0.1f);
        SituBeta = gguf.GetMetadataSingle($"{arch}.activation.situ_beta", 4.0f);
        SituLinearBeta = gguf.GetMetadataSingle($"{arch}.activation.situ_linear_beta", 25.0f);
        FinalLogitSoftcapping = gguf.GetMetadataSingle($"{arch}.final_logit_softcapping", 30.0f);

        CoreStart = (int)gguf.GetMetadataUInt32($"{arch}.loop.core_start", 6);
        CoreEnd = (int)gguf.GetMetadataUInt32($"{arch}.loop.core_end", 18);
        NumLoops = (int)gguf.GetMetadataUInt32($"{arch}.num_loops", 3);
        GateRank = (int)gguf.GetMetadataUInt32($"{arch}.loop.gate_rank", 64);

        // Count physical blocks present in GGUF (e.g. 24 physical -> 48 logical)
        int maxBlk = -1;
        foreach (string key in gguf.Tensors.Keys)
        {
            if (key.StartsWith("blk.", StringComparison.Ordinal))
            {
                int dot = key.IndexOf('.', 4);
                if (dot > 4 && int.TryParse(key.AsSpan(4, dot - 4), out int idx) && idx > maxBlk)
                    maxBlk = idx;
            }
        }
        PhysicalLayerCount = maxBlk + 1;
        if (PhysicalLayerCount <= 0)
            throw new InvalidDataException("Inkling2 GGUF contains no physical blk.* tensors.");

        LogicalLayerCount = PhysicalLayerCount + (NumLoops - 1) * (CoreEnd - CoreStart);

        bool[]? swaPattern = TryReadBoolArray(gguf, $"{arch}.attention.sliding_window_pattern");
        int[]? headArr = gguf.GetMetadataInt32Array($"{arch}.attention.head_count");
        int[]? headKvArr = gguf.GetMetadataInt32Array($"{arch}.attention.head_count_kv");

        // Global tensors
        var tokT = RequireTensor(gguf, "token_embd.weight");
        TokenEmbd = gguf.GetTensorPointer(tokT);
        TokenEmbdType = tokT.Type;
        TokenEmbdNorm = LoadF32(gguf, "token_embd_norm.weight");

        var outT = RequireTensor(gguf, "output.weight");
        OutputWeight = gguf.GetTensorPointer(outT);
        OutputWeightType = outT.Type;
        OutputNorm = LoadF32(gguf, "output_norm.weight");
        OutputResQuery = LoadF32(gguf, "output_res_query.weight");
        LoopResQuery = TryLoadF32(gguf, "loop_res_query.weight");
        LoopKvResQuery = TryLoadF32(gguf, "loop_kv_res_query.weight");

        PhysicalLayers = new Inkling2LayerWeights[PhysicalLayerCount];
        for (int i = 0; i < PhysicalLayerCount; i++)
        {
            bool isSwa = swaPattern != null && i < swaPattern.Length
                ? swaPattern[i]
                : ((i + 1) % AttnResBlockSize != 0);
            int nHead = headArr != null && i < headArr.Length ? headArr[i] : gguf.HeadCount;
            int nHeadKv = headKvArr != null && i < headKvArr.Length ? headKvArr[i] : gguf.HeadCountKv;
            int headDim = gguf.HeadDim;

            var wq = RequireTensor(gguf, $"blk.{i}.attn_q.weight");
            var wk = RequireTensor(gguf, $"blk.{i}.attn_k.weight");
            var wv = RequireTensor(gguf, $"blk.{i}.attn_v.weight");
            var wo = RequireTensor(gguf, $"blk.{i}.attn_output.weight");
            var wgate = RequireTensor(gguf, $"blk.{i}.attn_gate.weight");
            var wsink = RequireTensor(gguf, $"blk.{i}.attn_sink_proj.weight");
            var wrel = RequireTensor(gguf, $"blk.{i}.attn_rel.weight");

            var ffnGateInp = RequireTensor(gguf, $"blk.{i}.ffn_gate_inp.weight");
            var ffnGateUp = RequireTensor(gguf, $"blk.{i}.ffn_gate_up_exps.weight");
            var ffnDown = RequireTensor(gguf, $"blk.{i}.ffn_down_exps.weight");

            gguf.TryGetTensor($"blk.{i}.attn_k_conv_proj.weight", out var kConvProj);
            gguf.TryGetTensor($"blk.{i}.attn_v_conv_proj.weight", out var vConvProj);

            bool hasLoop = i >= CoreStart && i < CoreEnd;
            var ctrlDown = new byte*[4];
            var ctrlDownT = new GgufType[4];
            var ctrlGain = new byte*[4];
            var ctrlGainT = new GgufType[4];
            var ctrlForget = new byte*[4];
            var ctrlForgetT = new GgufType[4];

            if (hasLoop)
            {
                string[][] names =
                [
                    [$"blk.{i}.loop_attn", $"blk.{i}.loop_q_ctrl"],
                    [$"blk.{i}.loop_k", $"blk.{i}.loop_k_ctrl"],
                    [$"blk.{i}.loop_v", $"blk.{i}.loop_v_ctrl"],
                    [$"blk.{i}.loop_mlp", $"blk.{i}.loop_ffn_ctrl"],
                ];
                for (int c = 0; c < 4; c++)
                {
                    var td = RequireAnyTensor(gguf, $"{names[c][0]}_down.weight", $"{names[c][1]}_down.weight");
                    var tg = RequireAnyTensor(gguf, $"{names[c][0]}_gain.weight", $"{names[c][1]}_gain.weight");
                    var tf = RequireAnyTensor(gguf, $"{names[c][0]}_forget.weight", $"{names[c][1]}_forget.weight");
                    ctrlDown[c] = gguf.GetTensorPointer(td); ctrlDownT[c] = td.Type;
                    ctrlGain[c] = gguf.GetTensorPointer(tg); ctrlGainT[c] = tg.Type;
                    ctrlForget[c] = gguf.GetTensorPointer(tf); ctrlForgetT[c] = tf.Type;
                }
            }

            PhysicalLayers[i] = new Inkling2LayerWeights
            {
                IsSwa = isSwa,
                HeadCount = nHead,
                HeadCountKv = nHeadKv,
                HeadDim = headDim,
                RelExtent = isSwa ? SlidingWindow : RelativeExtent,

                AttnNorm = TryLoadF32(gguf, $"blk.{i}.attn_norm.weight"),
                AttnQInpNorm = TryLoadF32(gguf, $"blk.{i}.attn_q_inp_norm.weight"),
                AttnKInpNorm = TryLoadF32(gguf, $"blk.{i}.attn_k_inp_norm.weight"),
                AttnVInpNorm = TryLoadF32(gguf, $"blk.{i}.attn_v_inp_norm.weight"),
                FfnNorm = LoadF32(gguf, $"blk.{i}.ffn_norm.weight"),

                AttnResQuery = TryLoadF32(gguf, $"blk.{i}.attn_res_query.weight"),
                FfnResQuery = LoadF32(gguf, $"blk.{i}.ffn_res_query.weight"),

                Wq = gguf.GetTensorPointer(wq), WqType = wq.Type,
                Wk = gguf.GetTensorPointer(wk), WkType = wk.Type,
                Wv = gguf.GetTensorPointer(wv), WvType = wv.Type,
                Wo = gguf.GetTensorPointer(wo), WoType = wo.Type,
                AttnGate = gguf.GetTensorPointer(wgate), AttnGateType = wgate.Type,

                AttnQClip = LoadF32(gguf, $"blk.{i}.attn_q_clip.weight"),
                AttnSinkProj = gguf.GetTensorPointer(wsink), AttnSinkProjType = wsink.Type,
                AttnRel = gguf.GetTensorPointer(wrel), AttnRelType = wrel.Type,
                AttnRelBias = LoadF32(gguf, $"blk.{i}.attn_rel_bias.weight"),
                AttnRelClip = LoadF32(gguf, $"blk.{i}.attn_rel_clip.weight"),

                AttnKConv = LoadF32(gguf, $"blk.{i}.attn_k_conv.weight"),
                AttnVConv = LoadF32(gguf, $"blk.{i}.attn_v_conv.weight"),
                AttnPostConv = LoadF32(gguf, $"blk.{i}.attn_post_conv.weight"),
                FfnPostConv = LoadF32(gguf, $"blk.{i}.ffn_post_conv.weight"),
                AttnKConvProj = kConvProj != null ? gguf.GetTensorPointer(kConvProj) : null,
                AttnKConvProjType = kConvProj?.Type ?? GgufType.F32,
                AttnVConvProj = vConvProj != null ? gguf.GetTensorPointer(vConvProj) : null,
                AttnVConvProjType = vConvProj?.Type ?? GgufType.F32,

                FfnGateInp = gguf.GetTensorPointer(ffnGateInp), FfnGateInpType = ffnGateInp.Type,
                ExpProbsBias = TryLoadF32(gguf, $"blk.{i}.exp_probs_b.bias"),
                FfnGateUpExps = gguf.GetTensorPointer(ffnGateUp), FfnGateUpExpsType = ffnGateUp.Type,
                FfnGateUpExpsScale = TryLoadF32(gguf, $"blk.{i}.ffn_gate_up_exps.scale"),
                FfnDownExps = gguf.GetTensorPointer(ffnDown), FfnDownExpsType = ffnDown.Type,
                FfnDownExpsScale = TryLoadF32(gguf, $"blk.{i}.ffn_down_exps.scale"),

                HasLoopGates = hasLoop,
                LoopMsgQuery = hasLoop ? TryLoadF32(gguf, $"blk.{i}.loop_msg_query.weight") : null,
                LoopCtrlDown = ctrlDown,
                LoopCtrlDownType = ctrlDownT,
                LoopCtrlGain = ctrlGain,
                LoopCtrlGainType = ctrlGainT,
                LoopCtrlForget = ctrlForget,
                LoopCtrlForgetType = ctrlForgetT,
            };
        }
    }

    private static GgufTensorInfo RequireTensor(GgufFile gguf, string name)
    {
        if (!gguf.TryGetTensor(name, out var t) || t == null)
            throw new InvalidDataException($"Inkling2 GGUF missing required tensor '{name}'.");
        return t;
    }

    private static GgufTensorInfo RequireAnyTensor(GgufFile gguf, string nameA, string nameB)
    {
        if (gguf.TryGetTensor(nameA, out var ta) && ta != null) return ta;
        if (gguf.TryGetTensor(nameB, out var tb) && tb != null) return tb;
        throw new InvalidDataException($"Inkling2 GGUF missing required tensor '{nameA}' (or '{nameB}').");
    }

    private float* LoadF32(GgufFile gguf, string name)
    {
        var t = RequireTensor(gguf, name);
        return ToF32Pointer(gguf, t);
    }

    private float* TryLoadF32(GgufFile gguf, string name)
    {
        if (!gguf.TryGetTensor(name, out var t) || t == null) return null;
        return ToF32Pointer(gguf, t);
    }

    private float* ToF32Pointer(GgufFile gguf, GgufTensorInfo t)
    {
        byte* raw = gguf.GetTensorPointer(t);
        if (t.Type == GgufType.F32)
            return (float*)raw;

        int count = checked((int)t.ElementCount);
        float* buf = (float*)NativeMemory.Alloc((nuint)count * sizeof(float));
        _allocatedBuffers.Add((nint)buf);
        if (t.Type == GgufType.F16)
        {
            Half* h = (Half*)raw;
            for (int i = 0; i < count; i++) buf[i] = (float)h[i];
        }
        else if (t.Type == GgufType.BF16)
        {
            Quant.QuantKernels.DequantizeBF16((ushort*)raw, buf, count);
        }
        else
        {
            throw new NotSupportedException($"Inkling2 F32 conversion not supported for {t.Type} on '{t.Name}'.");
        }
        return buf;
    }

    private static bool[]? TryReadBoolArray(GgufFile gguf, string key)
    {
        if (!gguf.Metadata.TryGetValue(key, out var val) || val == null) return null;
        if (val is bool[] ba) return ba;
        if (val is object[] oa)
        {
            var res = new bool[oa.Length];
            for (int i = 0; i < oa.Length; i++) res[i] = Convert.ToBoolean(oa[i]);
            return res;
        }
        return null;
    }

    public void Dispose()
    {
        foreach (nint p in _allocatedBuffers)
            NativeMemory.Free((void*)p);
        _allocatedBuffers.Clear();
    }
}
