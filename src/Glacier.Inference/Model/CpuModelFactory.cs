namespace Glacier.Inference.Model;

using Glacier.Inference.Memory;

/// <summary>
/// Factory that inspects model weights topology and instantiates the optimal
/// CPU SIMD model runtime engine (e.g. Qwen3HybridModel for Gated DeltaNet / SSM,
/// or Qwen2Model for standard Transformer attention, RoPE, MLA, and MoE).
/// </summary>
public static class CpuModelFactory
{
    public static ICpuModel Create(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true,
        PleIoMode pleIoMode = PleIoMode.DirectIo,
        KvCachePrecision kvPrecision = KvCachePrecision.Fp32)
    {
        if (weights.ArchitectureFamily == UniversalArchitecture.Inkling2)
        {
            return new Inkling2Model(weights, maxSeqLen);
        }

        if (weights.ArchitectureFamily == UniversalArchitecture.Gemma4)
        {
            return new Gemma4Model(weights, maxSeqLen, startLayer, layerCount, isLastStage, kvPrecision);
        }

        if (weights.IsHybridSsm || weights.ArchitectureFamily == UniversalArchitecture.HybridSsm)
        {
            return new Qwen3HybridModel(weights, maxSeqLen, startLayer, layerCount, isLastStage, pleTable: null, pleIoMode: pleIoMode);
        }

        return new Qwen2Model(weights, maxSeqLen, startLayer, layerCount, isLastStage);
    }
}
