namespace Glacier.Inference.Engine;

using System;

/// <summary>
/// Telemetry metrics for a speculative decoding session.
/// </summary>
public sealed record SpeculativeMetrics
{
    public int TotalTokensGenerated { get; init; }
    public int DraftTokensProposed { get; init; }
    public int DraftTokensAccepted { get; init; }
    public int SpeculativeSteps { get; init; }
    public int SerialFallbackSteps { get; init; }

    /// <summary>
    /// Acceptance rate of speculative candidate tokens: [0.0, 1.0].
    /// </summary>
    public double AcceptanceRate =>
        DraftTokensProposed > 0 ? (double)DraftTokensAccepted / DraftTokensProposed : 0.0;

    /// <summary>
    /// Average tokens emitted per speculative verification round.
    /// </summary>
    public double AverageTokensPerStep =>
        SpeculativeSteps > 0 ? (double)TotalTokensGenerated / SpeculativeSteps : 0.0;

    /// <summary>
    /// Theoretical acceleration factor achieved over serial decoding.
    /// </summary>
    public double EstimatedSpeedup =>
        (SpeculativeSteps + SerialFallbackSteps) > 0
            ? (double)TotalTokensGenerated / (SpeculativeSteps + SerialFallbackSteps)
            : 1.0;
}

/// <summary>
/// Result of a speculative decoding generation run.
/// </summary>
public sealed record SpeculativeGenerationResult
{
    public required string Text { get; init; }
    public required string FinishReason { get; init; }
    public required GenerationMetrics Metrics { get; init; }
    public required SpeculativeMetrics SpeculativeMetrics { get; init; }
}
