namespace Glacier.Inference.Model;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Pointers to quantized and float tensors for a single transformer layer.
/// Directly maps into memory-mapped file space with zero memory copying.
/// </summary>
public sealed unsafe class LayerWeights
{
    public float* AttnNormWeight { get; init; }
    public GgufType AttnNormType { get; init; }

    public byte* QWeight { get; init; }
    public GgufType QType { get; init; }
    public float* QBias { get; init; }

    public byte* KWeight { get; init; }
    public GgufType KType { get; init; }
    public float* KBias { get; init; }

    public byte* VWeight { get; init; }
    public GgufType VType { get; init; }
    public float* VBias { get; init; }

    public required byte* AttnOutWeight { get; init; }
    public required GgufType AttnOutType { get; init; }
    public float* AttnOutBias { get; init; }

    // Fused QKV (Phi-3, Phi-4, Qwen3.5/3.6 GDN)
    public bool HasFusedQkv => QkvWeight != null;
    public byte* QkvWeight { get; init; }
    public GgufType QkvType { get; init; }
    public float* QkvBias { get; init; }

    // Gated DeltaNet (GDN) / State Space Model (SSM)
    public bool IsGdn { get; init; }
    public byte* AttnGateWeight { get; init; }
    public GgufType AttnGateType { get; init; }
    public float* SsmConv1dWeight { get; init; }
    public ulong[]? SsmConv1dDimensions { get; init; }
    public float* SsmAWeight { get; init; }
    public byte* SsmAlphaWeight { get; init; }
    public GgufType SsmAlphaType { get; init; }
    public byte* SsmBetaWeight { get; init; }
    public GgufType SsmBetaType { get; init; }
    public float* SsmDtBias { get; init; }
    public float* SsmNormWeight { get; init; }
    public float* SsmDWeight { get; init; }
    public float* SsmConv1dBias { get; init; }
    public byte* SsmOutWeight { get; init; }
    public GgufType SsmOutType { get; init; }

    // Multi-Head Latent Attention (MLA)
    public bool IsMla { get; init; }
    public byte* AttnKvAMqaWeight { get; init; }
    public GgufType AttnKvAMqaType { get; init; }
    public float* AttnKvANormWeight { get; init; }
    public byte* AttnKvBWeight { get; init; }
    public GgufType AttnKvBType { get; init; }

    // Optional Q-LoRA
    public byte* AttnQAWeight { get; init; }
    public GgufType AttnQAType { get; init; }
    public float* AttnQANormWeight { get; init; }
    public byte* AttnQBWeight { get; init; }
    public GgufType AttnQBType { get; init; }

    // DeepSeek Sparse Attention (DSA) Lightning Indexer
    public bool IsDsa { get; init; }
    public byte* AttnIdxQWeight { get; init; }
    public GgufType AttnIdxQType { get; init; }
    public byte* AttnIdxKWeight { get; init; }
    public GgufType AttnIdxKType { get; init; }
    public byte* AttnIdxWWeight { get; init; }
    public GgufType AttnIdxWType { get; init; }

    // Optional QK-norm
    public float* AttnQNormWeight { get; init; }
    public float* AttnKNormWeight { get; init; }

    // Optional Attention Sinks (OpenAI gpt-oss, StreamingLLM)
    public float* AttnSinksWeight { get; init; }

    public float* FfnNormWeight { get; init; }
    public GgufType FfnNormType { get; init; }

    // Dense FFN
    public byte* FfnGateWeight { get; init; }
    public GgufType FfnGateType { get; init; }

    public byte* FfnUpWeight { get; init; }
    public GgufType FfnUpType { get; init; }
    public float* FfnUpBias { get; init; }

    public byte* FfnDownWeight { get; init; }
    public GgufType FfnDownType { get; init; }

    // Fused Gate/Up SwiGLU (Phi-3, Phi-4: ffn_up combines gate and up projection)
    public bool HasFusedGateUp => FfnGateWeight == null && FfnUpWeight != null;

    // MoE Router & Experts
    public bool IsMoe { get; init; }
    public float* FfnGateInpWeight { get; init; }
    public float* FfnGateInpBias { get; init; }

    public byte* FfnGateExpsWeight { get; set; }
    public GgufType FfnGateExpsType { get; init; }
    public float* FfnGateExpsBias { get; init; }

    public byte* FfnUpExpsWeight { get; set; }
    public GgufType FfnUpExpsType { get; init; }
    public float* FfnUpExpsBias { get; init; }

    public byte* FfnDownExpsWeight { get; set; }
    public GgufType FfnDownExpsType { get; init; }
    public float* FfnDownExpsBias { get; init; }

    // Shared Experts (DeepSeek, ERNIE)
    public byte* FfnGateShexpWeight { get; init; }
    public GgufType FfnGateShexpType { get; init; }

    public byte* FfnUpShexpWeight { get; init; }
    public GgufType FfnUpShexpType { get; init; }

    public byte* FfnDownShexpWeight { get; init; }
    public GgufType FfnDownShexpType { get; init; }

    // Qwen 3.5 / 3.6 Q-Gate (attn_q outputs query + gate)
    public bool HasQGate { get; init; }

    // Shared Expert Gate (Qwen 3.5 MoE: ffn_gate_inp_shexp)
    public float* FfnGateInpShexpWeight { get; init; }

    // Gemma 4 specific fields
    public bool IsSwa { get; init; }
    public int HeadDim { get; init; }
    public int HeadsKv { get; init; }
    public float* AttnPostNormWeight { get; init; }
    public float* FfnPostNormWeight { get; init; }
    public float* FfnPostNorm1Weight { get; init; }
    public float* FfnPreNorm2Weight { get; init; }
    public float* FfnPostNorm2Weight { get; init; }
    public float* FfnGateInpScaleWeight { get; init; }
    public byte* FfnGateUpExpsWeight { get; set; }
    public GgufType FfnGateUpExpsType { get; init; }
    public float* FfnDownExpsScaleWeight { get; init; }
    public float* LayerOutputScaleWeight { get; init; }

    // Qwen3.8-Flash-Next (qwen4exp): Gated Multi-Stream Hyper-Connections
    public bool HasHyperConnections { get; init; }
    public float* HcAttnNormWeight { get; init; }
    public byte* HcAttnDownWeight { get; init; }
    public GgufType HcAttnDownType { get; init; }
    public byte* HcAttnUpWeight { get; init; }
    public GgufType HcAttnUpType { get; init; }
    public byte* HcAttnInjectWeight { get; init; }
    public GgufType HcAttnInjectType { get; init; }
    public float* HcFfnNormWeight { get; init; }
    public byte* HcFfnDownWeight { get; init; }
    public GgufType HcFfnDownType { get; init; }
    public byte* HcFfnUpWeight { get; init; }
    public GgufType HcFfnUpType { get; init; }
    public byte* HcFfnInjectWeight { get; init; }
    public GgufType HcFfnInjectType { get; init; }

    // Qwen3.8-Flash-Next (qwen4exp): QSA Sparse Attention Indexer
    public bool IsQsa { get; init; }
    public byte* IndexerQProjWeight { get; init; }
    public GgufType IndexerQProjType { get; init; }
    public byte* IndexerKProjWeight { get; init; }
    public GgufType IndexerKProjType { get; init; }
    public float* IndexerQNormWeight { get; init; }
    public float* IndexerKNormWeight { get; init; }
    public int IndexerCompressRatio { get; init; }

    // Qwen3.8-Flash-Next (qwen4exp): Per-Layer N-gram Embedding (PLE)
    public bool HasPle { get; init; }
    public byte* PleKeyWeight { get; init; }
    public GgufType PleKeyType { get; init; }
    public byte* PleValueWeight { get; init; }
    public GgufType PleValueType { get; init; }
    public float* PleNormKeyWeight { get; init; }
    public float* PleNormQueryWeight { get; init; }
    public float* PleNormConvWeight { get; init; }
    public float* PleConv1dWeight { get; init; }
}

/// <summary>
/// Memory-mapped weight pointers and architecture metadata for Transformer and MoE models.
/// </summary>
public sealed unsafe class ModelWeights : IDisposable
{
    public GgufFile Gguf { get; }
    public UniversalArchitecture ArchitectureFamily => ModelArchitectureDetector.Detect(Gguf.Architecture, Gguf);
    public long FileSizeBytes => new FileInfo(Gguf.FilePath).Length;

    public int VocabSize { get; }
    public int BlockCount { get; }
    public int ContextLength { get; }
    public int EmbeddingLength { get; }
    public int FeedForwardLength { get; }
    public int HeadCount { get; }
    public int HeadCountKv { get; }
    public int HeadDim => Gguf.HeadDim;
    public int ValueDim => Gguf.ValueDim;
    public int KvLoraRank => Gguf.KvLoraRank;
    public int RopeDimensionCount => Gguf.RopeDimensionCount;
    public int QkNopeHeadDim => Gguf.QkNopeHeadDim;
    public bool IsMla => Gguf.IsMla;
    public bool IsDsa => Layers.Length > 0 && Layers[0].IsDsa;
    public int DsaTopK => Gguf.Metadata.TryGetValue("deepseek2.indexer.top_k", out var tk) ? Convert.ToInt32(tk)
        : Gguf.Metadata.TryGetValue("glm4.indexer.top_k", out var gk) ? Convert.ToInt32(gk) : 2048;
    public int DsaIndexHeadCount => Gguf.Metadata.TryGetValue("deepseek2.indexer.head_count", out var hc) ? Convert.ToInt32(hc)
        : Gguf.Metadata.TryGetValue("glm4.indexer.head_count", out var ghc) ? Convert.ToInt32(ghc) : 32;
    public int DsaIndexHeadDim => Gguf.Metadata.TryGetValue("deepseek2.indexer.head_dim", out var hd) ? Convert.ToInt32(hd)
        : Gguf.Metadata.TryGetValue("glm4.indexer.head_dim", out var ghd) ? Convert.ToInt32(ghd) : 128;
    public int LeadingDenseBlockCount => Gguf.LeadingDenseBlockCount;
    public float RopeFreqBase { get; }
    public float RmsNormEps { get; }
    public float* RopeFreqsWeight { get; }
    public bool HasRopeFreqs => RopeFreqsWeight != null;

    public bool IsHybridSsm => Gguf.IsHybridSsm;
    public int FullAttentionInterval => Gguf.FullAttentionInterval;
    public int SsmConvKernel => Gguf.SsmConvKernel;
    public int SsmStateSize => Gguf.SsmStateSize;
    public int SsmGroupCount => Gguf.SsmGroupCount;
    public int SsmTimeStepRank => Gguf.SsmTimeStepRank;
    public int SsmInnerSize => Gguf.SsmInnerSize;

    public bool IsMoe => Gguf.IsMoe;
    public bool NormTopK => Gguf.NormTopK;
    public int ExpertCount => Gguf.ExpertCount;
    public int ExpertUsedCount => Gguf.ExpertUsedCount;
    public int ExpertFeedForwardLength => Gguf.ExpertFeedForwardLength > 0 ? Gguf.ExpertFeedForwardLength : FeedForwardLength;
    public int SharedExpertFeedForwardLength { get; }
    public int ExpertSharedCount => Gguf.ExpertSharedCount;

    // Gemma metadata
    public float FinalLogitSoftcapping => Gguf.FinalLogitSoftcapping;
    public float AttnLogitSoftcapping => Gguf.AttnLogitSoftcapping;
    public int SlidingWindow => Gguf.SlidingWindow;
    public float RopeFreqBaseSwa => Gguf.RopeFreqBaseSwa;
    public int KeyLengthSwa => Gguf.KeyLengthSwa;
    public int ValueLengthSwa => Gguf.ValueLengthSwa;
    public int RopeDimensionCountSwa => Gguf.RopeDimensionCountSwa;
    public bool[]? SlidingWindowPattern => Gguf.SlidingWindowPattern;
    public int[]? HeadCountKvPattern => Gguf.HeadCountKvPattern;

    public byte* EmbdWeight { get; }
    public GgufType EmbdType { get; }

    public float* OutNormWeight { get; }
    public GgufType OutNormType { get; }

    public byte* OutWeight { get; }
    public GgufType OutType { get; }

    // Qwen3.8-Flash-Next (qwen4exp): Hyper-Connections, QSA Indexer, PLE, and embedded MTP block index
    public int HyperConnectionCount { get; }
    public int HyperConnectionLowRank { get; }
    public float* OutputHcNormWeight { get; }
    public byte* OutputHcDownWeight { get; }
    public GgufType OutputHcDownType { get; }
    public byte* OutputHcUpWeight { get; }
    public GgufType OutputHcUpType { get; }

    public int IndexerHeadCount => Gguf.IndexerHeadCount;
    public int IndexerKeyLength => Gguf.IndexerKeyLength;
    public int IndexerTopK => Gguf.IndexerTopK;
    public PleHashConfig? PleHashConfig { get; }
    public int EmbeddedMtpBlockIndex { get; } = -1;

    private readonly List<IntPtr> _ownedFloatBuffers = new();
    private readonly List<IntPtr> _registeredTrellisPtrs = new();
    private HostRamExpertArena? _expertArena;
    private bool _disposed;

    public bool HasTrellisWeights => _registeredTrellisPtrs.Count > 0;
    public LayerWeights[] Layers { get; }
    public Inkling2Weights? Inkling2 { get; }
    public HostRamExpertArena? ExpertArena => _expertArena;

    public ModelWeights(GgufFile gguf)
    {
        Gguf = gguf;
        RegisterTrellisWeights(gguf);

        int embeddedMtpBlockIndex = -1;
        int blockCount = gguf.BlockCount;
        while (blockCount > 0 && gguf.TryGetTensor($"blk.{blockCount - 1}.nextn.eh_proj.weight", out var nextnProj) && nextnProj != null)
        {
            embeddedMtpBlockIndex = blockCount - 1;
            blockCount--;
        }
        if (embeddedMtpBlockIndex < 0 && gguf.TryGetTensor($"blk.{blockCount}.nextn.eh_proj.weight", out var trailingMtp) && trailingMtp != null)
        {
            embeddedMtpBlockIndex = blockCount;
        }
        EmbeddedMtpBlockIndex = embeddedMtpBlockIndex;
        BlockCount = blockCount;
        ContextLength = gguf.ContextLength;
        EmbeddingLength = gguf.EmbeddingLength;
        FeedForwardLength = gguf.FeedForwardLength;
        int sharedFfn = (int)gguf.GetMetadataUInt32($"{gguf.Architecture}.expert_shared_feed_forward_length",
            gguf.GetMetadataUInt32("expert_shared_feed_forward_length", 0));
        if (sharedFfn <= 0)
        {
            for (int i = 0; i < blockCount; i++)
            {
                if (gguf.TryGetTensor($"blk.{i}.ffn_gate_shexp.weight", out var shGate) && shGate != null && shGate.Dimensions.Length >= 2)
                {
                    sharedFfn = (int)shGate.Dimensions[1];
                    break;
                }
                if (gguf.TryGetTensor($"blk.{i}.ffn_up_shexp.weight", out var shUp) && shUp != null && shUp.Dimensions.Length >= 2)
                {
                    sharedFfn = (int)shUp.Dimensions[1];
                    break;
                }
            }
        }
        SharedExpertFeedForwardLength = sharedFfn;
        HeadCount = gguf.HeadCount;
        HeadCountKv = gguf.HeadCountKv;
        RopeFreqBase = gguf.RopeFreqBase;
        RmsNormEps = gguf.RmsNormEps;

        HyperConnectionCount = gguf.HyperConnectionCount;
        if (HyperConnectionCount == 0 && gguf.TryGetTensor("blk.0.hc_attn_inject.weight", out var hcInjProbe) && hcInjProbe != null && hcInjProbe.Dimensions.Length >= 2)
        {
            HyperConnectionCount = (int)hcInjProbe.Dimensions[1];
        }
        if (HyperConnectionCount == 0 && gguf.TryGetTensor("blk.0.hc_attn_norm.weight", out var hcNormProbe) && hcNormProbe != null)
        {
            if (hcNormProbe.Dimensions.Length >= 2)
            {
                HyperConnectionCount = (int)hcNormProbe.Dimensions[1];
            }
            else if (EmbeddingLength > 0 && hcNormProbe.ElementCount > (ulong)EmbeddingLength && hcNormProbe.ElementCount % (ulong)EmbeddingLength == 0)
            {
                HyperConnectionCount = (int)(hcNormProbe.ElementCount / (ulong)EmbeddingLength);
            }
        }
        HyperConnectionLowRank = gguf.HyperConnectionLowRank;
        if (HyperConnectionLowRank == 0 && gguf.TryGetTensor("blk.0.hc_attn_down.weight", out var hcDownProbe) && hcDownProbe != null && hcDownProbe.Dimensions.Length >= 2)
        {
            HyperConnectionLowRank = (int)hcDownProbe.Dimensions[1];
        }
        if (HyperConnectionLowRank == 0 && HyperConnectionCount > 0 && gguf.TryGetTensor("blk.0.hc_attn_down_inject.weight", out var hcDownInjProbe) && hcDownInjProbe != null && hcDownInjProbe.Dimensions.Length >= 2)
        {
            int nLo = (int)hcDownInjProbe.Dimensions[1] - HyperConnectionCount;
            if (nLo > 0) HyperConnectionLowRank = nLo;
        }

        if (gguf.PleLayers is { Length: > 0 } || gguf.Tensors.ContainsKey("per_layer_token_embd.weight"))
        {
            PleHashConfig = new PleHashConfig
            {
                NgramSize = gguf.PleNgramSize,
                HeadsPerNgram = gguf.PleHeadsPerNgram,
                HeadDim = gguf.PleRowDim,
                ConvKernel = gguf.PleConvKernel,
                EosTokenId = gguf.PleEosTokenId,
                ImageTokenId = gguf.PleImageTokenId,
                Multipliers = gguf.PleLayerMultipliers ?? PleHashConfig.DefaultMultipliers,
                HeadOffsets = gguf.PleHeadOffsets ?? PleHashConfig.DefaultHeadOffsets,
                HeadVocabSizes = gguf.PleHeadVocabSizes ?? PleHashConfig.DefaultHeadVocabSizes
            };
        }

        // Precomputed RoPE frequencies (Meta LLaMA 3.1)
        if (gguf.TryGetTensor("rope_freqs.weight", out var ropeFreqs) && ropeFreqs != null)
        {
            RopeFreqsWeight = (float*)gguf.GetTensorPointer(ropeFreqs);
        }


        // Embedding
        if (gguf.TryGetTensor("token_embd.weight", out var embdInfo) && embdInfo != null)
        {
            EmbdWeight = gguf.GetTensorPointer(embdInfo);
            EmbdType = embdInfo.Type;
            VocabSize = (int)embdInfo.Dimensions[1];
        }
        else
        {
            EmbdWeight = null;
            EmbdType = GgufType.F32;
            VocabSize = (int)gguf.GetMetadataUInt32("tokenizer.ggml.tokens.length", 151936);
            if (VocabSize <= 0) VocabSize = 151936;
        }

        // Final output norm
        if (gguf.TryGetTensor("output_norm.weight", out var outNormInfo) && outNormInfo != null)
        {
            OutNormWeight = LoadFloat32Pointer(outNormInfo);
            OutNormType = outNormInfo.Type;
        }
        else if (gguf.TryGetTensor("norm.weight", out outNormInfo) && outNormInfo != null)
        {
            OutNormWeight = LoadFloat32Pointer(outNormInfo);
            OutNormType = outNormInfo.Type;
        }
        else
        {
            OutNormWeight = null;
            OutNormType = GgufType.F32;
        }

        // Output head & optional Hyper-Connection output mixer
        if ((gguf.TryGetTensor("output_hc_norm.weight", out var outHcNorm) && outHcNorm != null) ||
            (gguf.TryGetTensor("hc_head_norm.weight", out outHcNorm) && outHcNorm != null))
        {
            OutputHcNormWeight = LoadFloat32Pointer(outHcNorm);
        }
        if ((gguf.TryGetTensor("output_hc_down.weight", out var outHcDown) && outHcDown != null) ||
            (gguf.TryGetTensor("hc_head_down.weight", out outHcDown) && outHcDown != null))
        {
            OutputHcDownWeight = gguf.GetTensorPointer(outHcDown);
            OutputHcDownType = outHcDown.Type;
        }
        if ((gguf.TryGetTensor("output_hc_up.weight", out var outHcUp) && outHcUp != null) ||
            (gguf.TryGetTensor("hc_head_up.weight", out outHcUp) && outHcUp != null))
        {
            OutputHcUpWeight = gguf.GetTensorPointer(outHcUp);
            OutputHcUpType = outHcUp.Type;
        }

        if (gguf.TryGetTensor("output.weight", out var outInfo) && outInfo != null)
        {
            OutWeight = gguf.GetTensorPointer(outInfo);
            OutType = outInfo.Type;
            if (embdInfo == null && outInfo.Dimensions.Length >= 2)
            {
                VocabSize = (int)outInfo.Dimensions[1];
            }
        }
        else if (gguf.TrellisLmHeadBands > 0 && TryRegisterBandedTrellis(gguf, "output", gguf.TrellisLmHeadBands, out var bandedOutPtr, out var bandedOutType, out var bandedVocab))
        {
            OutWeight = bandedOutPtr;
            OutType = bandedOutType;
            if (embdInfo == null && bandedVocab > 0)
            {
                VocabSize = bandedVocab;
            }
        }
        else
        {
            // Tied embeddings fallback
            OutWeight = EmbdWeight;
            OutType = EmbdType;
        }

        // Layers
        if (ArchitectureFamily == UniversalArchitecture.Inkling2)
        {
            Inkling2 = new Inkling2Weights(gguf);
            VocabSize = Inkling2.VocabSizeUnpadded;
            Layers = [];
            return;
        }

        Layers = new LayerWeights[BlockCount];
        for (int l = 0; l < BlockCount; l++)
        {
            if (!gguf.TryGetTensor($"blk.{l}.attn_norm.weight", out var attnNorm) || attnNorm == null)
            {
                gguf.TryGetTensor($"blk.{l}.input_layernorm.weight", out attnNorm);
            }

            gguf.TryGetTensor($"blk.{l}.hc_attn_norm.weight", out var hcAttnNorm);
            if (attnNorm == null && hcAttnNorm == null)
            {
                // Layer not present in this stage slice
                continue;
            }

            GgufTensorInfo? ssmOut = null;
            bool isGdn = gguf.TryGetTensor($"blk.{l}.ssm_out.weight", out ssmOut) && ssmOut != null;

            GgufTensorInfo? kvAMqa = null;
            bool isMla = !isGdn && gguf.TryGetTensor($"blk.{l}.attn_kv_a_mqa.weight", out kvAMqa) && kvAMqa != null;

            byte* qWeight = null;
            GgufType qType = GgufType.F32;
            float* qb = null;
            byte* qAWeight = null;
            GgufType qAType = GgufType.F32;
            float* qANormWeight = null;
            byte* qBWeight = null;
            GgufType qBType = GgufType.F32;

            byte* qkvWeight = null;
            GgufType qkvType = GgufType.F32;
            float* qkvBias = null;

            byte* attnGateWeight = null;
            GgufType attnGateType = GgufType.F32;
            float* ssmConv1dWeight = null;
            ulong[]? ssmConv1dDims = null;
            float* ssmAWeight = null;
            byte* ssmAlphaWeight = null;
            GgufType ssmAlphaType = GgufType.F32;
            byte* ssmBetaWeight = null;
            GgufType ssmBetaType = GgufType.F32;
            float* ssmDtBias = null;
            float* ssmNormWeight = null;
            byte* ssmOutWeight = null;
            GgufType ssmOutType = GgufType.F32;

            byte* attnOutWeight = null;
            GgufType attnOutType = GgufType.F32;
            float* attnOutB = null;

            GgufTensorInfo? qNorm = null;
            GgufTensorInfo? kNorm = null;
            GgufTensorInfo? attnSinks = null;

            byte* kWeight = null;
            GgufType kType = GgufType.F32;
            float* kb = null;
            byte* vWeight = null;
            GgufType vType = GgufType.F32;
            float* vb = null;

            byte* kvAMqaWeight = null;
            GgufType kvAMqaType = GgufType.F32;
            float* kvANormWeight = null;
            byte* kvBWeight = null;
            GgufType kvBType = GgufType.F32;
            bool hasQGate = false;

            if (isGdn)
            {
                // Parse Gated DeltaNet (SSM) layer
                if (gguf.TryGetTensor($"blk.{l}.attn_qkv.weight", out var gdnQkv) && gdnQkv != null)
                {
                    qkvWeight = gguf.GetTensorPointer(gdnQkv);
                    qkvType = gdnQkv.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.attn_gate.weight", out var gdnGate) && gdnGate != null)
                {
                    attnGateWeight = gguf.GetTensorPointer(gdnGate);
                    attnGateType = gdnGate.Type;
                }
                if (qkvWeight == null && attnGateWeight == null &&
                    gguf.TryGetTensor($"blk.{l}.attn_qkv_gate.weight", out var gdnQkvGate) &&
                    gdnQkvGate != null && gdnQkvGate.Dimensions.Length >= 2)
                {
                    int inDim = (int)gdnQkvGate.Dimensions[0];
                    int totalRows = (int)gdnQkvGate.Dimensions[1];
                    int nZ = gguf.SsmStateSize * gguf.SsmTimeStepRank;
                    int nQkv = totalRows - nZ;
                    if (nQkv > 0 && nZ > 0)
                    {
                        byte* basePtr = gguf.GetTensorPointer(gdnQkvGate);
                        long rowBytes = GgufTypes.GetRowBytes(gdnQkvGate.Type, inDim);
                        qkvWeight = basePtr;
                        qkvType = gdnQkvGate.Type;
                        attnGateWeight = basePtr + (long)nQkv * rowBytes;
                        attnGateType = gdnQkvGate.Type;
                    }
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_conv1d.weight", out var gdnConv) && gdnConv != null)
                {
                    int convKernel = gguf.SsmConvKernel;
                    long expectedConvChannels = ArchitectureFamily == UniversalArchitecture.Mamba2
                        ? (long)gguf.SsmInnerSize + 2L * gguf.SsmGroupCount * gguf.SsmStateSize
                        : ((long)gguf.SsmGroupCount * 2L + gguf.SsmTimeStepRank) * gguf.SsmStateSize;
                    long expectedConvElements = (long)convKernel * expectedConvChannels;

                    bool validConvShape = convKernel > 0 && expectedConvChannels > 0 && (
                        (gdnConv.Dimensions.Length == 2 &&
                         gdnConv.Dimensions[0] == (ulong)convKernel &&
                         gdnConv.Dimensions[1] == (ulong)expectedConvChannels) ||
                        (gdnConv.Dimensions.Length == 1 &&
                         gdnConv.Dimensions[0] == (ulong)expectedConvElements));

                    if (!validConvShape)
                    {
                        throw new InvalidDataException(
                            $"Invalid shape [{string.Join(", ", gdnConv.Dimensions)}] for 'blk.{l}.ssm_conv1d.weight' on layer {l}; " +
                            $"expected [{convKernel}, {expectedConvChannels}] (or 1D [{expectedConvElements}]).");
                    }

                    ssmConv1dDims = gdnConv.Dimensions;
                    ssmConv1dWeight = LoadFloat32Pointer(gdnConv);
                }
                else if (attnGateWeight != null)
                {
                    throw new InvalidDataException(
                        $"Model '{Path.GetFileName(gguf.FilePath)}' is missing required GDN tensor 'blk.{l}.ssm_conv1d.weight' on layer {l}.");
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_a", out var gdnA) && gdnA != null)
                {
                    ssmAWeight = LoadFloat32Pointer(gdnA);
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_alpha.weight", out var gdnAlpha) && gdnAlpha != null)
                {
                    ssmAlphaWeight = (byte*)gguf.GetTensorPointer(gdnAlpha);
                    ssmAlphaType = gdnAlpha.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_beta.weight", out var gdnBeta) && gdnBeta != null)
                {
                    ssmBetaWeight = (byte*)gguf.GetTensorPointer(gdnBeta);
                    ssmBetaType = gdnBeta.Type;
                }
                if (ssmAlphaWeight == null && ssmBetaWeight == null &&
                    ((gguf.TryGetTensor($"blk.{l}.ssm_beta_alpha.weight", out var gdnBa) && gdnBa != null) ||
                     (gguf.TryGetTensor($"blk.{l}.ssm_ba.weight", out gdnBa) && gdnBa != null)) &&
                    gdnBa.Dimensions.Length >= 2)
                {
                    int inDim = (int)gdnBa.Dimensions[0];
                    int totalRows = (int)gdnBa.Dimensions[1];
                    int nBeta = totalRows / 2;
                    if (nBeta > 0)
                    {
                        byte* basePtr = gguf.GetTensorPointer(gdnBa);
                        long rowBytes = GgufTypes.GetRowBytes(gdnBa.Type, inDim);
                        ssmBetaWeight = basePtr;
                        ssmBetaType = gdnBa.Type;
                        ssmAlphaWeight = basePtr + (long)nBeta * rowBytes;
                        ssmAlphaType = gdnBa.Type;
                    }
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_dt.bias", out var gdnDt) && gdnDt != null)
                {
                    ssmDtBias = LoadFloat32Pointer(gdnDt);
                }
                if (gguf.TryGetTensor($"blk.{l}.ssm_norm.weight", out var gdnNorm) && gdnNorm != null)
                {
                    ssmNormWeight = LoadFloat32Pointer(gdnNorm);
                }
                ssmOutWeight = gguf.GetTensorPointer(ssmOut!);
                ssmOutType = ssmOut!.Type;
                attnOutWeight = ssmOutWeight;
                attnOutType = ssmOutType;
            }
            else
            {
                // Parse Standard Full-Attention layer
                if (gguf.TryGetTensor($"blk.{l}.attn_q.weight", out var q) && q != null)
                {
                    qWeight = gguf.GetTensorPointer(q);
                    qType = q.Type;
                    if (q.Dimensions.Length > 1 && q.Dimensions[1] == 2 * (ulong)(HeadCount * HeadDim))
                    {
                        hasQGate = true;
                    }
                    if (gguf.TryGetTensor($"blk.{l}.attn_q.bias", out var qBias) && qBias != null)
                    {
                        qb = (float*)gguf.GetTensorPointer(qBias);
                    }
                }
                else if (gguf.TryGetTensor($"blk.{l}.attn_q_a.weight", out var qA) && qA != null)
                {
                    qAWeight = gguf.GetTensorPointer(qA);
                    qAType = qA.Type;
                    if (gguf.TryGetTensor($"blk.{l}.attn_q_a_norm.weight", out var qANorm) && qANorm != null)
                    {
                        qANormWeight = (float*)gguf.GetTensorPointer(qANorm);
                    }
                    if (gguf.TryGetTensor($"blk.{l}.attn_q_b.weight", out var qB) && qB != null)
                    {
                        qBWeight = gguf.GetTensorPointer(qB);
                        qBType = qB.Type;
                    }
                }
                else if (gguf.TryGetTensor($"blk.{l}.attn_qkv.weight", out var qkv) && qkv != null)
                {
                    int layerHeadsKv = (gguf.HeadCountKvPattern != null && l < gguf.HeadCountKvPattern.Length)
                        ? gguf.HeadCountKvPattern[l]
                        : HeadCountKv;
                    int qkvHeadDim = (gguf.TryGetTensor($"blk.{l}.attn_q_norm.weight", out var qNormProbe) && qNormProbe != null && qNormProbe.Dimensions.Length > 0)
                        ? (int)qNormProbe.Dimensions[0]
                        : HeadDim;
                    int qDim = HeadCount * qkvHeadDim;
                    int kvDim = layerHeadsKv * qkvHeadDim;

                    bool isQGateFused = qkv.Dimensions.Length >= 2 && qkv.Dimensions[1] == (ulong)(2 * qDim + 2 * kvDim);
                    bool isHybridFused = gguf.IsHybridSsm && qkv.Dimensions.Length >= 2 && qkv.Dimensions[1] == (ulong)(qDim + 2 * kvDim);
                    if (isQGateFused || isHybridFused)
                    {
                        int inDim = (int)qkv.Dimensions[0];
                        int qRows = isQGateFused ? 2 * qDim : qDim;
                        byte* basePtr = gguf.GetTensorPointer(qkv);
                        long rowBytes = GgufTypes.GetRowBytes(qkv.Type, inDim);
                        qWeight = basePtr;
                        qType = qkv.Type;
                        hasQGate = isQGateFused;
                        kWeight = basePtr + (long)qRows * rowBytes;
                        kType = qkv.Type;
                        vWeight = basePtr + (long)(qRows + kvDim) * rowBytes;
                        vType = qkv.Type;

                        if (gguf.TryGetTensor($"blk.{l}.attn_qkv.bias", out var qkvB) && qkvB != null)
                        {
                            float* bPtr = LoadFloat32Pointer(qkvB);
                            qb = bPtr;
                            kb = bPtr + qRows;
                            vb = bPtr + qRows + kvDim;
                        }
                    }
                    else
                    {
                        qkvWeight = gguf.GetTensorPointer(qkv);
                        qkvType = qkv.Type;
                        if (gguf.TryGetTensor($"blk.{l}.attn_qkv.bias", out var qkvB) && qkvB != null)
                        {
                            qkvBias = (float*)gguf.GetTensorPointer(qkvB);
                        }
                    }
                }

                if (isMla)
                {
                    kvAMqaWeight = gguf.GetTensorPointer(kvAMqa!);
                    kvAMqaType = kvAMqa!.Type;

                    if (gguf.TryGetTensor($"blk.{l}.attn_kv_a_norm.weight", out var kvANorm) && kvANorm != null)
                    {
                        kvANormWeight = (float*)gguf.GetTensorPointer(kvANorm);
                    }

                    if (gguf.TryGetTensor($"blk.{l}.attn_kv_b.weight", out var kvB) && kvB != null)
                    {
                        kvBWeight = gguf.GetTensorPointer(kvB);
                        kvBType = kvB.Type;
                    }
                }
                else
                {
                    if (gguf.TryGetTensor($"blk.{l}.attn_k.weight", out var k) && k != null)
                    {
                        kWeight = gguf.GetTensorPointer(k);
                        kType = k.Type;
                    }
                    if (gguf.TryGetTensor($"blk.{l}.attn_k.bias", out var kBias) && kBias != null)
                    {
                        kb = (float*)gguf.GetTensorPointer(kBias);
                    }
                    if (gguf.TryGetTensor($"blk.{l}.attn_v.weight", out var v) && v != null)
                    {
                        vWeight = gguf.GetTensorPointer(v);
                        vType = v.Type;
                    }
                    if (gguf.TryGetTensor($"blk.{l}.attn_v.bias", out var vBias) && vBias != null)
                    {
                        vb = (float*)gguf.GetTensorPointer(vBias);
                    }
                }

                if (!gguf.TryGetTensor($"blk.{l}.attn_output.weight", out var attnOut) || attnOut == null)
                {
                    gguf.TryGetTensor($"blk.{l}.attn_wo.weight", out attnOut);
                }

                if (attnOut == null)
                {
                    string fileName = Path.GetFileName(gguf.FilePath);
                    throw new InvalidDataException(
                        $"Model '{fileName}' is missing attention output projection tensor for layer {l} ('blk.{l}.attn_output.weight' / 'blk.{l}.attn_wo.weight').");
                }

                attnOutWeight = gguf.GetTensorPointer(attnOut);
                attnOutType = attnOut.Type;

                if (gguf.TryGetTensor($"blk.{l}.attn_output.bias", out var aob) && aob != null)
                {
                    attnOutB = (float*)gguf.GetTensorPointer(aob);
                }

                gguf.TryGetTensor($"blk.{l}.attn_q_norm.weight", out qNorm);
                gguf.TryGetTensor($"blk.{l}.attn_k_norm.weight", out kNorm);
                gguf.TryGetTensor($"blk.{l}.attn_sinks.weight", out attnSinks);
            }

            if (!gguf.TryGetTensor($"blk.{l}.ffn_norm.weight", out var ffnNorm) || ffnNorm == null)
            {
                if (!gguf.TryGetTensor($"blk.{l}.post_attention_norm.weight", out ffnNorm) || ffnNorm == null)
                {
                    gguf.TryGetTensor($"blk.{l}.post_attention_layernorm.weight", out ffnNorm);
                }
            }

            gguf.TryGetTensor($"blk.{l}.hc_ffn_norm.weight", out var hcFfnNorm);
            if (ffnNorm == null && hcFfnNorm == null)
            {
                string fileName = Path.GetFileName(gguf.FilePath);
                throw new InvalidDataException(
                    $"Model '{fileName}' is missing FFN normalization tensor for layer {l} ('blk.{l}.ffn_norm.weight' / 'blk.{l}.hc_ffn_norm.weight').");
            }

            // Check if this layer has MoE experts
            GgufTensorInfo? ffnGateUpExps = null;
            bool hasMoE = (gguf.TryGetTensor($"blk.{l}.ffn_gate_exps.weight", out var ffnGateExps) && ffnGateExps != null) ||
                          (gguf.TryGetTensor($"blk.{l}.ffn_gate_up_exps.weight", out ffnGateUpExps) && ffnGateUpExps != null);

            byte* ffnGateWeight = null;
            GgufType ffnGateType = GgufType.F32;
            byte* ffnUpWeight = null;
            GgufType ffnUpType = GgufType.F32;
            float* ffnUpBias = null;
            byte* ffnDownWeight = null;
            GgufType ffnDownType = GgufType.F32;

            float* gateInpWeight = null;
            float* gateInpBias = null;
            byte* gateExpsWeight = null;
            GgufType gateExpsType = GgufType.F32;
            float* gateExpsBias = null;
            byte* upExpsWeight = null;
            GgufType upExpsType = GgufType.F32;
            float* upExpsBias = null;
            byte* downExpsWeight = null;
            GgufType downExpsType = GgufType.F32;
            float* downExpsBias = null;

            byte* gateUpExpsWeight = null;
            GgufType gateUpExpsType = GgufType.F32;
            float* gateInpScale = null;
            float* downExpsScale = null;
            float* layerOutScale = null;
            float* attnPostNormWeight = null;
            float* ffnPostNormWeight = null;
            float* ffnPostNorm1Weight = null;
            float* ffnPreNorm2Weight = null;
            float* ffnPostNorm2Weight = null;

            byte* shexpGateWeight = null;
            GgufType shexpGateType = GgufType.F32;
            byte* shexpUpWeight = null;
            GgufType shexpUpType = GgufType.F32;
            byte* shexpDownWeight = null;
            GgufType shexpDownType = GgufType.F32;
            float* shexpGateInpWeight = null;

            if (hasMoE)
            {
                if (ffnGateExps != null)
                {
                    gateExpsWeight = gguf.GetTensorPointer(ffnGateExps);
                    gateExpsType = ffnGateExps.Type;
                }
                if (ffnGateUpExps != null)
                {
                    gateUpExpsWeight = gguf.GetTensorPointer(ffnGateUpExps);
                    gateUpExpsType = ffnGateUpExps.Type;
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_exps.bias", out var ffnGateExpsB) && ffnGateExpsB != null)
                {
                    gateExpsBias = (float*)gguf.GetTensorPointer(ffnGateExpsB);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_up_exps.weight", out var ffnUpExps) && ffnUpExps != null)
                {
                    upExpsWeight = gguf.GetTensorPointer(ffnUpExps);
                    upExpsType = ffnUpExps.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.ffn_up_exps.bias", out var ffnUpExpsB) && ffnUpExpsB != null)
                {
                    upExpsBias = (float*)gguf.GetTensorPointer(ffnUpExpsB);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_down_exps.weight", out var ffnDownExps) && ffnDownExps != null)
                {
                    downExpsWeight = gguf.GetTensorPointer(ffnDownExps);
                    downExpsType = ffnDownExps.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.ffn_down_exps.bias", out var ffnDownExpsB) && ffnDownExpsB != null)
                {
                    downExpsBias = (float*)gguf.GetTensorPointer(ffnDownExpsB);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_inp.weight", out var ffnGateInp) && ffnGateInp != null)
                {
                    gateInpWeight = LoadFloat32Pointer(ffnGateInp);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_inp.bias", out var ffnGateInpB) && ffnGateInpB != null)
                {
                    gateInpBias = LoadFloat32Pointer(ffnGateInpB);
                }
                else if (gguf.TryGetTensor($"blk.{l}.exp_probs_b.bias", out var expProbsB) && expProbsB != null)
                {
                    gateInpBias = LoadFloat32Pointer(expProbsB);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_inp_shexp.weight", out var ffnGateInpShexp) && ffnGateInpShexp != null)
                {
                    shexpGateInpWeight = LoadFloat32Pointer(ffnGateInpShexp);
                }
                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var ffnGateShexp) && ffnGateShexp != null)
                {
                    shexpGateWeight = gguf.GetTensorPointer(ffnGateShexp);
                    shexpGateType = ffnGateShexp.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.ffn_up_shexp.weight", out var ffnUpShexp) && ffnUpShexp != null)
                {
                    shexpUpWeight = gguf.GetTensorPointer(ffnUpShexp);
                    shexpUpType = ffnUpShexp.Type;
                }
                if (gguf.TryGetTensor($"blk.{l}.ffn_down_shexp.weight", out var ffnDownShexp) && ffnDownShexp != null)
                {
                    shexpDownWeight = gguf.GetTensorPointer(ffnDownShexp);
                    shexpDownType = ffnDownShexp.Type;
                }
            }

            // Always check for dense FFN projections (e.g. Gemma 4 shared MLP alongside MoE)
            if (gguf.TryGetTensor($"blk.{l}.ffn_gate.weight", out var ffnGate) && ffnGate != null)
            {
                ffnGateWeight = gguf.GetTensorPointer(ffnGate);
                ffnGateType = ffnGate.Type;
            }
            if (gguf.TryGetTensor($"blk.{l}.ffn_up.weight", out var ffnUp) && ffnUp != null)
            {
                ffnUpWeight = gguf.GetTensorPointer(ffnUp);
                ffnUpType = ffnUp.Type;
                if (gguf.TryGetTensor($"blk.{l}.ffn_up.bias", out var ffnUpB) && ffnUpB != null)
                {
                    ffnUpBias = (float*)gguf.GetTensorPointer(ffnUpB);
                }
            }
            if (gguf.TryGetTensor($"blk.{l}.ffn_down.weight", out var ffnDown) && ffnDown != null)
            {
                ffnDownWeight = gguf.GetTensorPointer(ffnDown);
                ffnDownType = ffnDown.Type;
            }

            // Gemma 4 specific norms and scales
            if (gguf.TryGetTensor($"blk.{l}.ffn_gate_inp.scale", out var ffnGateInpScale) && ffnGateInpScale != null)
            {
                gateInpScale = (float*)gguf.GetTensorPointer(ffnGateInpScale);
            }
            if (gguf.TryGetTensor($"blk.{l}.ffn_down_exps.scale", out var ffnDownExpsScale) && ffnDownExpsScale != null)
            {
                downExpsScale = (float*)gguf.GetTensorPointer(ffnDownExpsScale);
            }
            if (gguf.TryGetTensor($"blk.{l}.layer_output_scale.weight", out var lOutScale) && lOutScale != null)
            {
                layerOutScale = (float*)gguf.GetTensorPointer(lOutScale);
            }
            if (gguf.TryGetTensor($"blk.{l}.post_attention_norm.weight", out var apn) && apn != null)
            {
                attnPostNormWeight = (float*)gguf.GetTensorPointer(apn);
            }
            else if (gguf.TryGetTensor($"blk.{l}.attn_post_norm.weight", out apn) && apn != null)
            {
                attnPostNormWeight = (float*)gguf.GetTensorPointer(apn);
            }
            if (gguf.TryGetTensor($"blk.{l}.post_ffw_norm.weight", out var fpn) && fpn != null)
            {
                ffnPostNormWeight = (float*)gguf.GetTensorPointer(fpn);
            }
            else if (gguf.TryGetTensor($"blk.{l}.ffn_post_norm.weight", out fpn) && fpn != null)
            {
                ffnPostNormWeight = (float*)gguf.GetTensorPointer(fpn);
            }
            if (gguf.TryGetTensor($"blk.{l}.post_ffw_norm_1.weight", out var fpn1) && fpn1 != null)
            {
                ffnPostNorm1Weight = (float*)gguf.GetTensorPointer(fpn1);
            }
            else if (gguf.TryGetTensor($"blk.{l}.ffn_post_norm_1.weight", out fpn1) && fpn1 != null)
            {
                ffnPostNorm1Weight = (float*)gguf.GetTensorPointer(fpn1);
            }
            if (gguf.TryGetTensor($"blk.{l}.pre_ffw_norm_2.weight", out var fpn2pre) && fpn2pre != null)
            {
                ffnPreNorm2Weight = (float*)gguf.GetTensorPointer(fpn2pre);
            }
            else if (gguf.TryGetTensor($"blk.{l}.ffn_pre_norm_2.weight", out fpn2pre) && fpn2pre != null)
            {
                ffnPreNorm2Weight = (float*)gguf.GetTensorPointer(fpn2pre);
            }
            if (gguf.TryGetTensor($"blk.{l}.post_ffw_norm_2.weight", out var fpn2post) && fpn2post != null)
            {
                ffnPostNorm2Weight = (float*)gguf.GetTensorPointer(fpn2post);
            }
            else if (gguf.TryGetTensor($"blk.{l}.ffn_post_norm_2.weight", out fpn2post) && fpn2post != null)
            {
                ffnPostNorm2Weight = (float*)gguf.GetTensorPointer(fpn2post);
            }

            bool isSwa = true;
            if (gguf.SlidingWindowPattern != null && l < gguf.SlidingWindowPattern.Length)
            {
                isSwa = gguf.SlidingWindowPattern[l];
            }
            else if (gguf.Architecture.Equals("gemma2", StringComparison.OrdinalIgnoreCase) ||
                     gguf.Architecture.Equals("gemma", StringComparison.OrdinalIgnoreCase))
            {
                // Gemma 2 alternates sliding window attention on even layers (0, 2, 4...)
                isSwa = (l % 2 == 0);
            }
            else if (gguf.Architecture.StartsWith("gemma", StringComparison.OrdinalIgnoreCase))
            {
                isSwa = (l + 1) % 6 != 0;
            }

            float* ssmConv1dBias = null, ssmDWeight = null;
            if (gguf.TryGetTensor($"blk.{l}.ssm_conv1d.bias", out var scbT) && scbT != null) ssmConv1dBias = (float*)gguf.GetTensorPointer(scbT);
            if (gguf.TryGetTensor($"blk.{l}.ssm_d", out var sdT) && sdT != null) ssmDWeight = (float*)gguf.GetTensorPointer(sdT);
            if (qkvWeight == null && gguf.TryGetTensor($"blk.{l}.ssm_in.weight", out var ssmInT) && ssmInT != null)
            { qkvWeight = gguf.GetTensorPointer(ssmInT); qkvType = ssmInT.Type; }
            if (qkvWeight == null && gguf.TryGetTensor($"blk.{l}.shortconv.in_proj.weight", out var scInT) && scInT != null)
            { qkvWeight = gguf.GetTensorPointer(scInT); qkvType = scInT.Type; }
            if (ssmConv1dWeight == null && gguf.TryGetTensor($"blk.{l}.shortconv.conv.weight", out var scConvT) && scConvT != null)
            { ssmConv1dWeight = (float*)gguf.GetTensorPointer(scConvT); }
            if (ssmOutWeight == null && gguf.TryGetTensor($"blk.{l}.shortconv.out_proj.weight", out var scOutT) && scOutT != null)
            { ssmOutWeight = gguf.GetTensorPointer(scOutT); ssmOutType = scOutT.Type; }

            byte* idxQWeight = null, idxKWeight = null, idxWWeight = null;
            GgufType idxQType = GgufType.F32, idxKType = GgufType.F32, idxWType = GgufType.F32;
            if ((gguf.TryGetTensor($"blk.{l}.indexer_q.weight", out var iqT) || gguf.TryGetTensor($"blk.{l}.attn_idx_q.weight", out iqT)) && iqT != null)
            { idxQWeight = gguf.GetTensorPointer(iqT); idxQType = iqT.Type; }
            if ((gguf.TryGetTensor($"blk.{l}.indexer_k.weight", out var ikT) || gguf.TryGetTensor($"blk.{l}.attn_idx_k.weight", out ikT)) && ikT != null)
            { idxKWeight = gguf.GetTensorPointer(ikT); idxKType = ikT.Type; }
            if ((gguf.TryGetTensor($"blk.{l}.indexer_w.weight", out var iwT) || gguf.TryGetTensor($"blk.{l}.attn_idx_w.weight", out iwT)) && iwT != null)
            { idxWWeight = gguf.GetTensorPointer(iwT); idxWType = iwT.Type; }

            int headsKv = HeadCountKv;
            if (gguf.HeadCountKvPattern != null && l < gguf.HeadCountKvPattern.Length)
            {
                headsKv = gguf.HeadCountKvPattern[l];
            }
            else if (gguf.Architecture.StartsWith("gemma", StringComparison.OrdinalIgnoreCase) && HeadCountKv == 0)
            {
                headsKv = isSwa ? 8 : 2;
            }

            int layerHeadDim = HeadDim;
            if (qNorm != null && qNorm.Dimensions.Length > 0)
            {
                layerHeadDim = (int)qNorm.Dimensions[0];
            }
            else if (gguf.Architecture.StartsWith("gemma", StringComparison.OrdinalIgnoreCase) && HeadDim == 0)
            {
                layerHeadDim = isSwa ? 256 : 512;
            }

            // Qwen3.8-Flash-Next (qwen4exp): Hyper-Connections, QSA Indexer, and PLE
            float* hcAttnNormWeight = LoadFloat32Pointer(hcAttnNorm);
            byte* hcAttnDownWeight = null;
            GgufType hcAttnDownType = GgufType.F32;
            byte* hcAttnUpWeight = null;
            GgufType hcAttnUpType = GgufType.F32;
            byte* hcAttnInjectWeight = null;
            GgufType hcAttnInjectType = GgufType.F32;

            if (gguf.TryGetTensor($"blk.{l}.hc_attn_down.weight", out var hcAd) && hcAd != null) { hcAttnDownWeight = gguf.GetTensorPointer(hcAd); hcAttnDownType = hcAd.Type; }
            if (gguf.TryGetTensor($"blk.{l}.hc_attn_up.weight", out var hcAu) && hcAu != null) { hcAttnUpWeight = gguf.GetTensorPointer(hcAu); hcAttnUpType = hcAu.Type; }
            if (gguf.TryGetTensor($"blk.{l}.hc_attn_inject.weight", out var hcAi) && hcAi != null) { hcAttnInjectWeight = gguf.GetTensorPointer(hcAi); hcAttnInjectType = hcAi.Type; }
            if (hcAttnDownWeight == null && hcAttnInjectWeight == null &&
                gguf.TryGetTensor($"blk.{l}.hc_attn_down_inject.weight", out var hcAdi) && hcAdi != null &&
                hcAdi.Dimensions.Length >= 2 && HyperConnectionCount > 0)
            {
                int hcDim = (int)hcAdi.Dimensions[0];
                int nLo = (int)hcAdi.Dimensions[1] - HyperConnectionCount;
                if (nLo > 0)
                {
                    byte* basePtr = gguf.GetTensorPointer(hcAdi);
                    long rowBytes = GgufTypes.GetRowBytes(hcAdi.Type, hcDim);
                    hcAttnDownWeight = basePtr;
                    hcAttnDownType = hcAdi.Type;
                    hcAttnInjectWeight = basePtr + (long)nLo * rowBytes;
                    hcAttnInjectType = hcAdi.Type;
                }
            }

            float* hcFfnNormWeight = LoadFloat32Pointer(hcFfnNorm);
            byte* hcFfnDownWeight = null;
            GgufType hcFfnDownType = GgufType.F32;
            byte* hcFfnUpWeight = null;
            GgufType hcFfnUpType = GgufType.F32;
            byte* hcFfnInjectWeight = null;
            GgufType hcFfnInjectType = GgufType.F32;

            if (gguf.TryGetTensor($"blk.{l}.hc_ffn_down.weight", out var hcFd) && hcFd != null) { hcFfnDownWeight = gguf.GetTensorPointer(hcFd); hcFfnDownType = hcFd.Type; }
            if (gguf.TryGetTensor($"blk.{l}.hc_ffn_up.weight", out var hcFu) && hcFu != null) { hcFfnUpWeight = gguf.GetTensorPointer(hcFu); hcFfnUpType = hcFu.Type; }
            if (gguf.TryGetTensor($"blk.{l}.hc_ffn_inject.weight", out var hcFi) && hcFi != null) { hcFfnInjectWeight = gguf.GetTensorPointer(hcFi); hcFfnInjectType = hcFi.Type; }
            if (hcFfnDownWeight == null && hcFfnInjectWeight == null &&
                gguf.TryGetTensor($"blk.{l}.hc_ffn_down_inject.weight", out var hcFdi) && hcFdi != null &&
                hcFdi.Dimensions.Length >= 2 && HyperConnectionCount > 0)
            {
                int hcDim = (int)hcFdi.Dimensions[0];
                int nLo = (int)hcFdi.Dimensions[1] - HyperConnectionCount;
                if (nLo > 0)
                {
                    byte* basePtr = gguf.GetTensorPointer(hcFdi);
                    long rowBytes = GgufTypes.GetRowBytes(hcFdi.Type, hcDim);
                    hcFfnDownWeight = basePtr;
                    hcFfnDownType = hcFdi.Type;
                    hcFfnInjectWeight = basePtr + (long)nLo * rowBytes;
                    hcFfnInjectType = hcFdi.Type;
                }
            }

            byte* qsaQProjWeight = null, qsaKProjWeight = null;
            GgufType qsaQProjType = GgufType.F32, qsaKProjType = GgufType.F32;
            float* qsaQNormWeight = null, qsaKNormWeight = null;
            if (gguf.TryGetTensor($"blk.{l}.indexer.q_proj.weight", out var qsaQp) && qsaQp != null) { qsaQProjWeight = gguf.GetTensorPointer(qsaQp); qsaQProjType = qsaQp.Type; }
            if (gguf.TryGetTensor($"blk.{l}.indexer.k_proj.weight", out var qsaKp) && qsaKp != null) { qsaKProjWeight = gguf.GetTensorPointer(qsaKp); qsaKProjType = qsaKp.Type; }
            if (gguf.TryGetTensor($"blk.{l}.indexer.q_norm.weight", out var qsaQn) && qsaQn != null) { qsaQNormWeight = LoadFloat32Pointer(qsaQn); }
            if (gguf.TryGetTensor($"blk.{l}.indexer.k_norm.weight", out var qsaKn) && qsaKn != null) { qsaKNormWeight = LoadFloat32Pointer(qsaKn); }

            int[]? compressRatios = gguf.AttentionCompressRatios;
            int idxCompressRatio = compressRatios != null && l < compressRatios.Length && compressRatios[l] > 0
                ? compressRatios[l]
                : (qsaQProjWeight != null && qsaKProjWeight != null ? 4 : 0);

            byte* pleKeyWeight = null, pleValueWeight = null;
            GgufType pleKeyType = GgufType.F32, pleValueType = GgufType.F32;
            float* pleNormKeyWeight = null, pleNormQueryWeight = null, pleNormConvWeight = null, pleConv1dWeight = null;
            if (gguf.TryGetTensor($"blk.{l}.ple_key.weight", out var pkT) && pkT != null) { pleKeyWeight = gguf.GetTensorPointer(pkT); pleKeyType = pkT.Type; }
            if (gguf.TryGetTensor($"blk.{l}.ple_value.weight", out var pvT) && pvT != null) { pleValueWeight = gguf.GetTensorPointer(pvT); pleValueType = pvT.Type; }
            if (gguf.TryGetTensor($"blk.{l}.ple_norm_key.weight", out var pnkT) && pnkT != null) { pleNormKeyWeight = LoadFloat32Pointer(pnkT); }
            if (gguf.TryGetTensor($"blk.{l}.ple_norm_query.weight", out var pnqT) && pnqT != null) { pleNormQueryWeight = LoadFloat32Pointer(pnqT); }
            if (gguf.TryGetTensor($"blk.{l}.ple_norm_conv.weight", out var pncT) && pncT != null) { pleNormConvWeight = LoadFloat32Pointer(pncT); }
            if (gguf.TryGetTensor($"blk.{l}.ple_conv1d.weight", out var pcT) && pcT != null) { pleConv1dWeight = LoadFloat32Pointer(pcT); }

            Layers[l] = new LayerWeights
            {
                AttnNormWeight = LoadFloat32Pointer(attnNorm),
                AttnNormType = attnNorm?.Type ?? GgufType.F32,
                QkvWeight = qkvWeight,
                QkvType = qkvType,
                QkvBias = qkvBias,
                QWeight = qWeight,
                QType = qType,
                QBias = qb,
                KWeight = kWeight,
                KType = kType,
                KBias = kb,
                VWeight = vWeight,
                VType = vType,
                VBias = vb,
                IsMla = isMla,
                AttnKvAMqaWeight = kvAMqaWeight,
                AttnKvAMqaType = kvAMqaType,
                AttnKvANormWeight = kvANormWeight,
                AttnKvBWeight = kvBWeight,
                AttnKvBType = kvBType,
                AttnQAWeight = qAWeight,
                AttnQAType = qAType,
                AttnQANormWeight = qANormWeight,
                AttnQBWeight = qBWeight,
                AttnQBType = qBType,
                IsDsa = idxQWeight != null && idxKWeight != null && idxWWeight != null,
                AttnIdxQWeight = idxQWeight,
                AttnIdxQType = idxQType,
                AttnIdxKWeight = idxKWeight,
                AttnIdxKType = idxKType,
                AttnIdxWWeight = idxWWeight,
                AttnIdxWType = idxWType,
                AttnOutWeight = attnOutWeight!,
                AttnOutType = attnOutType,
                AttnOutBias = attnOutB,
                AttnQNormWeight = LoadFloat32Pointer(qNorm),
                AttnKNormWeight = LoadFloat32Pointer(kNorm),
                AttnSinksWeight = LoadFloat32Pointer(attnSinks),
                IsGdn = isGdn,
                HasQGate = hasQGate,
                AttnGateWeight = attnGateWeight,
                AttnGateType = attnGateType,
                SsmConv1dWeight = ssmConv1dWeight,
                SsmConv1dDimensions = ssmConv1dDims,
                SsmAWeight = ssmAWeight,
                SsmAlphaWeight = ssmAlphaWeight,
                SsmAlphaType = ssmAlphaType,
                SsmBetaWeight = ssmBetaWeight,
                SsmBetaType = ssmBetaType,
                SsmDtBias = ssmDtBias,
                SsmNormWeight = ssmNormWeight,
                SsmDWeight = ssmDWeight,
                SsmConv1dBias = ssmConv1dBias,
                SsmOutWeight = ssmOutWeight,
                SsmOutType = ssmOutType,
                FfnNormWeight = LoadFloat32Pointer(ffnNorm),
                FfnNormType = ffnNorm?.Type ?? GgufType.F32,
                IsMoe = hasMoE,
                FfnGateWeight = ffnGateWeight,
                FfnGateType = ffnGateType,
                FfnUpWeight = ffnUpWeight,
                FfnUpType = ffnUpType,
                FfnUpBias = ffnUpBias,
                FfnDownWeight = ffnDownWeight,
                FfnDownType = ffnDownType,
                FfnGateInpWeight = gateInpWeight,
                FfnGateInpBias = gateInpBias,
                FfnGateExpsWeight = gateExpsWeight,
                FfnGateExpsType = gateExpsType,
                FfnGateExpsBias = gateExpsBias,
                FfnUpExpsWeight = upExpsWeight,
                FfnUpExpsType = upExpsType,
                FfnUpExpsBias = upExpsBias,
                FfnDownExpsWeight = downExpsWeight,
                FfnDownExpsType = downExpsType,
                FfnDownExpsBias = downExpsBias,
                FfnGateInpShexpWeight = shexpGateInpWeight,
                FfnGateShexpWeight = shexpGateWeight,
                FfnGateShexpType = shexpGateType,
                FfnUpShexpWeight = shexpUpWeight,
                FfnUpShexpType = shexpUpType,
                FfnDownShexpWeight = shexpDownWeight,
                FfnDownShexpType = shexpDownType,
                IsSwa = isSwa,
                HeadDim = layerHeadDim,
                HeadsKv = headsKv,
                AttnPostNormWeight = attnPostNormWeight,
                FfnPostNormWeight = ffnPostNormWeight,
                FfnPostNorm1Weight = ffnPostNorm1Weight,
                FfnPreNorm2Weight = ffnPreNorm2Weight,
                FfnPostNorm2Weight = ffnPostNorm2Weight,
                FfnGateInpScaleWeight = gateInpScale,
                FfnGateUpExpsWeight = gateUpExpsWeight,
                FfnGateUpExpsType = gateUpExpsType,
                FfnDownExpsScaleWeight = downExpsScale,
                LayerOutputScaleWeight = layerOutScale,
                HasHyperConnections = hcAttnDownWeight != null && hcFfnDownWeight != null,
                HcAttnNormWeight = hcAttnNormWeight,
                HcAttnDownWeight = hcAttnDownWeight,
                HcAttnDownType = hcAttnDownType,
                HcAttnUpWeight = hcAttnUpWeight,
                HcAttnUpType = hcAttnUpType,
                HcAttnInjectWeight = hcAttnInjectWeight,
                HcAttnInjectType = hcAttnInjectType,
                HcFfnNormWeight = hcFfnNormWeight,
                HcFfnDownWeight = hcFfnDownWeight,
                HcFfnDownType = hcFfnDownType,
                HcFfnUpWeight = hcFfnUpWeight,
                HcFfnUpType = hcFfnUpType,
                HcFfnInjectWeight = hcFfnInjectWeight,
                HcFfnInjectType = hcFfnInjectType,
                IsQsa = qsaQProjWeight != null && qsaKProjWeight != null,
                IndexerQProjWeight = qsaQProjWeight,
                IndexerQProjType = qsaQProjType,
                IndexerKProjWeight = qsaKProjWeight,
                IndexerKProjType = qsaKProjType,
                IndexerQNormWeight = qsaQNormWeight,
                IndexerKNormWeight = qsaKNormWeight,
                IndexerCompressRatio = idxCompressRatio,
                HasPle = pleKeyWeight != null && pleValueWeight != null,
                PleKeyWeight = pleKeyWeight,
                PleKeyType = pleKeyType,
                PleValueWeight = pleValueWeight,
                PleValueType = pleValueType,
                PleNormKeyWeight = pleNormKeyWeight,
                PleNormQueryWeight = pleNormQueryWeight,
                PleNormConvWeight = pleNormConvWeight,
                PleConv1dWeight = pleConv1dWeight
            };
        }
    }

    internal float* LoadFloat32Pointer(GgufTensorInfo? tensor) =>
        DequantizeTensorToFloat32(Gguf, tensor, _ownedFloatBuffers);

    internal static float* DequantizeTensorToFloat32(GgufFile gguf, GgufTensorInfo? tensor, List<IntPtr> ownedBuffers)
    {
        if (tensor == null) return null;
        byte* raw = gguf.GetTensorPointer(tensor);
        if (tensor.Type == GgufType.F32)
            return (float*)raw;

        ulong count = tensor.ElementCount;
        float* buf = (float*)NativeMemory.Alloc((nuint)Math.Max(1UL, count), (nuint)sizeof(float));
        if (tensor.Type == GgufType.BF16)
        {
            QuantKernels.DequantizeBF16((ushort*)raw, buf, (int)count);
        }
        else if (tensor.Type == GgufType.F16)
        {
            Half* h = (Half*)raw;
            for (ulong i = 0; i < count; i++) buf[i] = (float)h[i];
        }
        else
        {
            try
            {
                int rowWidth = tensor.Dimensions.Length >= 1 ? (int)tensor.Dimensions[0] : (int)count;
                if (rowWidth <= 0 || count % (ulong)rowWidth != 0)
                {
                    rowWidth = (int)count;
                }
                int nRows = rowWidth > 0 ? (int)(count / (ulong)rowWidth) : 1;
                for (int r = 0; r < nRows; r++)
                {
                    QuantKernels.ExtractEmbedding(tensor.Type, raw, r, buf + (long)r * rowWidth, rowWidth);
                }
            }
            catch
            {
                NativeMemory.Free(buf);
                throw;
            }
        }

        ownedBuffers.Add((IntPtr)buf);
        return buf;
    }

    public HostRamExpertArena? AttachExpertArena(ulong budgetBytes, bool preferUnbufferedIo = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _expertArena?.Dispose();
        _expertArena = HostRamExpertArena.TryAttach(this, budgetBytes, preferUnbufferedIo);
        return _expertArena;
    }

    private void RegisterTrellisWeights(GgufFile gguf)
    {
        bool anyTrellis = false;
        for (int i = 0; i < gguf.TensorList.Count; i++)
        {
            if (GgufTypes.IsTrellis(gguf.TensorList[i].Type))
            {
                anyTrellis = true;
                break;
            }
        }
        if (!anyTrellis) return;

        if (gguf.TrellisVersion != 1)
            throw new InvalidDataException($"Unsupported NariQ trellis.version={gguf.TrellisVersion}; expected 1.");

        for (int i = 0; i < gguf.TensorList.Count; i++)
        {
            var t = gguf.TensorList[i];
            if (!GgufTypes.IsTrellis(t.Type)) continue;
            if (t.Dimensions.Length != 2 || !t.Name.EndsWith(".weight", StringComparison.Ordinal))
                continue;

            int k = (int)t.Dimensions[0];
            int n = (int)t.Dimensions[1];
            string baseName = t.Name[..^7];
            int[]? desc = gguf.GetTrellisDescriptor(baseName);
            if (desc == null || !TrellisKernels.IsDescriptorSupported(k, n, desc))
                throw new InvalidDataException($"Unsupported or missing trellis descriptor for '{t.Name}'.");

            if (!gguf.TryGetTensor($"{baseName}.trellis_s16", out var s16Info) || s16Info == null ||
                !gguf.TryGetTensor($"{baseName}.trellis_pre", out var preInfo) || preInfo == null ||
                !gguf.TryGetTensor($"{baseName}.trellis_post", out var postInfo) || postInfo == null)
            {
                throw new InvalidDataException($"Missing required NariQ side tensors (s16/pre/post) for '{t.Name}'.");
            }

            gguf.TryGetTensor($"{baseName}.trellis_pin", out var pinInfo);
            gguf.TryGetTensor($"{baseName}.trellis_pout", out var poutInfo);

            byte* codePtr = gguf.GetTensorPointer(t);
            var binding = new TrellisWeightBinding
            {
                Type = t.Type,
                CodeData = codePtr,
                K = k,
                N = n,
                S16 = (ushort*)gguf.GetTensorPointer(s16Info),
                Pre = (float*)gguf.GetTensorPointer(preInfo),
                Post = (float*)gguf.GetTensorPointer(postInfo),
                Pin = pinInfo != null ? (int*)gguf.GetTensorPointer(pinInfo) : null,
                Pout = poutInfo != null ? (int*)gguf.GetTensorPointer(poutInfo) : null,
                Descriptor = desc,
            };
            TrellisWeightRegistry.Register(binding);
            _registeredTrellisPtrs.Add((IntPtr)codePtr);
        }
    }

    private bool TryRegisterBandedTrellis(
        GgufFile gguf, string headName, int bandCount,
        out byte* keyPtr, out GgufType keyType, out int totalN)
    {
        keyPtr = null;
        keyType = GgufType.F32;
        totalN = 0;

        if (!gguf.TryGetTensor($"{headName}.vocab_gather", out var gatherInfo) || gatherInfo == null)
            return false;

        var bands = new TrellisWeightBinding[bandCount];
        int k = 0;
        for (int b = 0; b < bandCount; b++)
        {
            string tensorName = $"{headName}.band{b:D2}.weight";
            if (!gguf.TryGetTensor(tensorName, out var info) || info == null)
                return false;
            byte* ptr = gguf.GetTensorPointer(info);
            if (!TrellisWeightRegistry.TryGet(ptr, out var binding) || binding == null)
                return false;
            bands[b] = binding;
            if (b == 0)
            {
                keyPtr = ptr;
                keyType = info.Type;
                k = binding.K;
            }
            totalN += binding.N;
        }

        var banded = new TrellisBandedWeight
        {
            Bands = bands,
            VocabGather = (int*)gguf.GetTensorPointer(gatherInfo),
            K = k,
            TotalN = totalN,
        };
        TrellisWeightRegistry.RegisterBanded(keyPtr, banded);
        return true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _registeredTrellisPtrs.Count; i++)
        {
            TrellisWeightRegistry.Unregister((byte*)_registeredTrellisPtrs[i]);
        }
        _registeredTrellisPtrs.Clear();

        _expertArena?.Dispose();
        _expertArena = null;

        Inkling2?.Dispose();

        for (int i = 0; i < _ownedFloatBuffers.Count; i++)
        {
            IntPtr ptr = _ownedFloatBuffers[i];
            if (ptr != IntPtr.Zero) NativeMemory.Free((void*)ptr);
        }
        _ownedFloatBuffers.Clear();
        GC.SuppressFinalize(this);
    }

    ~ModelWeights()
    {
        Dispose();
    }
}
