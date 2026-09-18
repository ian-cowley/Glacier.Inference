namespace Glacier.Inference.Engine;

/// <summary>
/// Strategy for pooling sequence token representations into a single normalized vector embedding.
/// </summary>
public enum PoolingStrategy
{
    /// <summary>
    /// Uses the normalized hidden state of the last token in the sequence.
    /// Recommended for causal autoregressive decoder models (e.g. Nomic, Qwen2, Llama).
    /// </summary>
    LastToken,

    /// <summary>
    /// Computes the mean average of all normalized token hidden states across the sequence.
    /// </summary>
    MeanPooling
}
