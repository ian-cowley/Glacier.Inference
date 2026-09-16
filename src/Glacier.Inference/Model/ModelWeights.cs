namespace Glacier.Inference.Model;

using System;
using Glacier.Inference.Gguf;

/// <summary>
/// Pointers to quantized and float tensors for a single transformer layer.
/// Directly maps into memory-mapped file space with zero memory copying.
/// </summary>
public sealed unsafe class LayerWeights
{
    public required float* AttnNormWeight { get; init; }
    public required GgufType AttnNormType { get; init; }

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

    // Fused QKV (Phi-3, Phi-4)
    public bool HasFusedQkv => QkvWeight != null;
    public byte* QkvWeight { get; init; }
    public GgufType QkvType { get; init; }
    public float* QkvBias { get; init; }

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

    // Optional QK-norm
    public float* AttnQNormWeight { get; init; }
    public float* AttnKNormWeight { get; init; }

    // Optional Attention Sinks (OpenAI gpt-oss, StreamingLLM)
    public float* AttnSinksWeight { get; init; }

    public required float* FfnNormWeight { get; init; }
    public required GgufType FfnNormType { get; init; }

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

    public byte* FfnGateExpsWeight { get; init; }
    public GgufType FfnGateExpsType { get; init; }
    public float* FfnGateExpsBias { get; init; }

    public byte* FfnUpExpsWeight { get; init; }
    public GgufType FfnUpExpsType { get; init; }
    public float* FfnUpExpsBias { get; init; }

    public byte* FfnDownExpsWeight { get; init; }
    public GgufType FfnDownExpsType { get; init; }
    public float* FfnDownExpsBias { get; init; }

    // Shared Experts (DeepSeek, ERNIE)
    public byte* FfnGateShexpWeight { get; init; }
    public GgufType FfnGateShexpType { get; init; }

    public byte* FfnUpShexpWeight { get; init; }
    public GgufType FfnUpShexpType { get; init; }

    public byte* FfnDownShexpWeight { get; init; }
    public GgufType FfnDownShexpType { get; init; }
}

/// <summary>
/// Memory-mapped weight pointers and architecture metadata for Transformer and MoE models.
/// </summary>
public sealed unsafe class ModelWeights
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
    public int LeadingDenseBlockCount => Gguf.LeadingDenseBlockCount;
    public float RopeFreqBase { get; }
    public float RmsNormEps { get; }
    public float* RopeFreqsWeight { get; }
    public bool HasRopeFreqs => RopeFreqsWeight != null;

    public bool IsMoe => Gguf.IsMoe;
    public bool NormTopK => Gguf.NormTopK;
    public int ExpertCount => Gguf.ExpertCount;
    public int ExpertUsedCount => Gguf.ExpertUsedCount;
    public int ExpertFeedForwardLength => Gguf.ExpertFeedForwardLength > 0 ? Gguf.ExpertFeedForwardLength : FeedForwardLength;
    public int ExpertSharedCount => Gguf.ExpertSharedCount;

    public byte* EmbdWeight { get; }
    public GgufType EmbdType { get; }

    public float* OutNormWeight { get; }
    public GgufType OutNormType { get; }

    public byte* OutWeight { get; }
    public GgufType OutType { get; }

    public LayerWeights[] Layers { get; }

    public ModelWeights(GgufFile gguf)
    {
        Gguf = gguf;

        BlockCount = gguf.BlockCount;
        ContextLength = gguf.ContextLength;
        EmbeddingLength = gguf.EmbeddingLength;
        FeedForwardLength = gguf.FeedForwardLength;
        HeadCount = gguf.HeadCount;
        HeadCountKv = gguf.HeadCountKv;
        RopeFreqBase = gguf.RopeFreqBase;
        RmsNormEps = gguf.RmsNormEps;

        // Precomputed RoPE frequencies (Meta LLaMA 3.1)
        if (gguf.TryGetTensor("rope_freqs.weight", out var ropeFreqs) && ropeFreqs != null)
        {
            RopeFreqsWeight = (float*)gguf.GetTensorPointer(ropeFreqs);
        }

        // Embedding
        var embdInfo = gguf.Tensors["token_embd.weight"];
        EmbdWeight = gguf.GetTensorPointer(embdInfo);
        EmbdType = embdInfo.Type;
        VocabSize = (int)embdInfo.Dimensions[1];

        // Final output norm
        var outNormInfo = gguf.Tensors["output_norm.weight"];
        OutNormWeight = (float*)gguf.GetTensorPointer(outNormInfo);
        OutNormType = outNormInfo.Type;

        // Output head
        if (gguf.TryGetTensor("output.weight", out var outInfo) && outInfo != null)
        {
            OutWeight = gguf.GetTensorPointer(outInfo);
            OutType = outInfo.Type;
        }
        else
        {
            // Tied embeddings fallback
            OutWeight = EmbdWeight;
            OutType = EmbdType;
        }

        // Layers
        Layers = new LayerWeights[BlockCount];
        for (int l = 0; l < BlockCount; l++)
        {
            if (!gguf.TryGetTensor($"blk.{l}.attn_norm.weight", out var attnNorm) || attnNorm == null)
            {
                gguf.TryGetTensor($"blk.{l}.input_layernorm.weight", out attnNorm);
            }

            bool isMla = gguf.TryGetTensor($"blk.{l}.attn_kv_a_mqa.weight", out var kvAMqa) && kvAMqa != null;

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

            if (gguf.TryGetTensor($"blk.{l}.attn_q.weight", out var q) && q != null)
            {
                qWeight = gguf.GetTensorPointer(q);
                qType = q.Type;
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
                qkvWeight = gguf.GetTensorPointer(qkv);
                qkvType = qkv.Type;
                if (gguf.TryGetTensor($"blk.{l}.attn_qkv.bias", out var qkvB) && qkvB != null)
                {
                    qkvBias = (float*)gguf.GetTensorPointer(qkvB);
                }
            }

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
            gguf.TryGetTensor($"blk.{l}.attn_output.bias", out var attnOutB);

            gguf.TryGetTensor($"blk.{l}.attn_q_norm.weight", out var qNorm);
            gguf.TryGetTensor($"blk.{l}.attn_k_norm.weight", out var kNorm);
            gguf.TryGetTensor($"blk.{l}.attn_sinks.weight", out var attnSinks);

            if (!gguf.TryGetTensor($"blk.{l}.ffn_norm.weight", out var ffnNorm) || ffnNorm == null)
            {
                if (!gguf.TryGetTensor($"blk.{l}.post_attention_norm.weight", out ffnNorm) || ffnNorm == null)
                {
                    gguf.TryGetTensor($"blk.{l}.post_attention_layernorm.weight", out ffnNorm);
                }
            }

            // Check if this layer has MoE experts
            bool hasMoE = gguf.TryGetTensor($"blk.{l}.ffn_gate_exps.weight", out var ffnGateExps) && ffnGateExps != null;

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

            byte* shexpGateWeight = null;
            GgufType shexpGateType = GgufType.F32;
            byte* shexpUpWeight = null;
            GgufType shexpUpType = GgufType.F32;
            byte* shexpDownWeight = null;
            GgufType shexpDownType = GgufType.F32;

            if (hasMoE)
            {
                gateExpsWeight = gguf.GetTensorPointer(ffnGateExps!);
                gateExpsType = ffnGateExps!.Type;

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
                    gateInpWeight = (float*)gguf.GetTensorPointer(ffnGateInp);
                }

                if (gguf.TryGetTensor($"blk.{l}.ffn_gate_inp.bias", out var ffnGateInpB) && ffnGateInpB != null)
                {
                    gateInpBias = (float*)gguf.GetTensorPointer(ffnGateInpB);
                }
                else if (gguf.TryGetTensor($"blk.{l}.exp_probs_b.bias", out var expProbsB) && expProbsB != null)
                {
                    gateInpBias = (float*)gguf.GetTensorPointer(expProbsB);
                }

                // Check for shared expert
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
            else
            {
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
            }

            Layers[l] = new LayerWeights
            {
                AttnNormWeight = (float*)gguf.GetTensorPointer(attnNorm!),
                AttnNormType = attnNorm!.Type,
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
                AttnOutWeight = gguf.GetTensorPointer(attnOut!),
                AttnOutType = attnOut!.Type,
                AttnOutBias = attnOutB != null ? (float*)gguf.GetTensorPointer(attnOutB) : null,
                AttnQNormWeight = qNorm != null ? (float*)gguf.GetTensorPointer(qNorm) : null,
                AttnKNormWeight = kNorm != null ? (float*)gguf.GetTensorPointer(kNorm) : null,
                AttnSinksWeight = attnSinks != null ? (float*)gguf.GetTensorPointer(attnSinks) : null,
                FfnNormWeight = (float*)gguf.GetTensorPointer(ffnNorm!),
                FfnNormType = ffnNorm!.Type,
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
                FfnGateShexpWeight = shexpGateWeight,
                FfnGateShexpType = shexpGateType,
                FfnUpShexpWeight = shexpUpWeight,
                FfnUpShexpType = shexpUpType,
                FfnDownShexpWeight = shexpDownWeight,
                FfnDownShexpType = shexpDownType
            };
        }
    }
}
