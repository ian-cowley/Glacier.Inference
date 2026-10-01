namespace Glacier.Inference.Model;

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
        bool isLastStage = true)
    {
        if (weights.IsHybridSsm || weights.ArchitectureFamily == UniversalArchitecture.HybridSsm)
        {
            return new Qwen3HybridModel(weights, maxSeqLen, startLayer, layerCount, isLastStage);
        }

        return new Qwen2Model(weights, maxSeqLen, startLayer, layerCount, isLastStage);
    }
}
