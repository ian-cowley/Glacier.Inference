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

    public required byte* QWeight { get; init; }
    public required GgufType QType { get; init; }
    public required float* QBias { get; init; }

    public required byte* KWeight { get; init; }
    public required GgufType KType { get; init; }
    public required float* KBias { get; init; }

    public required byte* VWeight { get; init; }
    public required GgufType VType { get; init; }
    public required float* VBias { get; init; }

    public required byte* AttnOutWeight { get; init; }
    public required GgufType AttnOutType { get; init; }

    public required float* FfnNormWeight { get; init; }
    public required GgufType FfnNormType { get; init; }

    public required byte* FfnGateWeight { get; init; }
    public required GgufType FfnGateType { get; init; }

    public required byte* FfnUpWeight { get; init; }
    public required GgufType FfnUpType { get; init; }

    public required byte* FfnDownWeight { get; init; }
    public required GgufType FfnDownType { get; init; }
}

/// <summary>
/// Memory-mapped weight pointers and architecture metadata for Qwen2 / Qwen2.5 models.
/// </summary>
public sealed unsafe class ModelWeights
{
    public GgufFile Gguf { get; }

    public int VocabSize { get; }
    public int BlockCount { get; }
    public int ContextLength { get; }
    public int EmbeddingLength { get; }
    public int FeedForwardLength { get; }
    public int HeadCount { get; }
    public int HeadCountKv { get; }
    public int HeadDim => EmbeddingLength / HeadCount;
    public float RopeFreqBase { get; }
    public float RmsNormEps { get; }

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
            var attnNorm = gguf.Tensors[$"blk.{l}.attn_norm.weight"];
            var q = gguf.Tensors[$"blk.{l}.attn_q.weight"];
            var qb = gguf.Tensors[$"blk.{l}.attn_q.bias"];
            var k = gguf.Tensors[$"blk.{l}.attn_k.weight"];
            var kb = gguf.Tensors[$"blk.{l}.attn_k.bias"];
            var v = gguf.Tensors[$"blk.{l}.attn_v.weight"];
            var vb = gguf.Tensors[$"blk.{l}.attn_v.bias"];
            var attnOut = gguf.Tensors[$"blk.{l}.attn_output.weight"];

            var ffnNorm = gguf.Tensors[$"blk.{l}.ffn_norm.weight"];
            var ffnGate = gguf.Tensors[$"blk.{l}.ffn_gate.weight"];
            var ffnUp = gguf.Tensors[$"blk.{l}.ffn_up.weight"];
            var ffnDown = gguf.Tensors[$"blk.{l}.ffn_down.weight"];

            Layers[l] = new LayerWeights
            {
                AttnNormWeight = (float*)gguf.GetTensorPointer(attnNorm),
                AttnNormType = attnNorm.Type,
                QWeight = gguf.GetTensorPointer(q),
                QType = q.Type,
                QBias = (float*)gguf.GetTensorPointer(qb),
                KWeight = gguf.GetTensorPointer(k),
                KType = k.Type,
                KBias = (float*)gguf.GetTensorPointer(kb),
                VWeight = gguf.GetTensorPointer(v),
                VType = v.Type,
                VBias = (float*)gguf.GetTensorPointer(vb),
                AttnOutWeight = gguf.GetTensorPointer(attnOut),
                AttnOutType = attnOut.Type,
                FfnNormWeight = (float*)gguf.GetTensorPointer(ffnNorm),
                FfnNormType = ffnNorm.Type,
                FfnGateWeight = gguf.GetTensorPointer(ffnGate),
                FfnGateType = ffnGate.Type,
                FfnUpWeight = gguf.GetTensorPointer(ffnUp),
                FfnUpType = ffnUp.Type,
                FfnDownWeight = gguf.GetTensorPointer(ffnDown),
                FfnDownType = ffnDown.Type
            };
        }
    }
}
