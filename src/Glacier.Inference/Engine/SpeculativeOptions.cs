namespace Glacier.Inference.Engine;

using Glacier.Inference.Sampling;

/// <summary>
/// Configuration options for speculative decoding execution.
/// </summary>
public sealed record SpeculativeOptions
{
    /// <summary>
    /// Maximum number of speculative candidate tokens to draft per verification step (default: 4).
    /// </summary>
    public int MaxDraftTokens { get; init; } = 4;

    /// <summary>
    /// Minimum n-gram match length for prompt lookup draft provider (default: 2).
    /// </summary>
    public int MinNgramMatch { get; init; } = 2;

    /// <summary>
    /// Maximum n-gram match length for prompt lookup draft provider (default: 4).
    /// </summary>
    public int MaxNgramMatch { get; init; } = 4;

    /// <summary>
    /// Maximum total tokens to generate.
    /// </summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>
    /// Sampling parameters applied to model verification (default: Greedy).
    /// </summary>
    public SamplingOptions Sampling { get; init; } = SamplingOptions.Greedy;
}
