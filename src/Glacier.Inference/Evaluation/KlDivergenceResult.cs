namespace Glacier.Inference.Evaluation;

using System;

/// <summary>
/// Configuration thresholds, numerical floor parameters, and execution options for KL-divergence verification.
/// </summary>
public sealed record KlDivergenceOptions
{
    /// <summary>Maximum allowed mean KL divergence (in nats) across evaluated tokens (default: 0.05).</summary>
    public double Threshold { get; init; } = 0.05;

    /// <summary>Optional maximum allowed worst-case per-token KL divergence (in nats).</summary>
    public double? MaxKlThreshold { get; init; }

    /// <summary>Minimum required top-1 argmax agreement rate in [0.0, 1.0].</summary>
    public double MinTop1Agreement { get; init; }

    /// <summary>Minimum probability floor applied to candidate probabilities to avoid log(0) (default: 1e-12).</summary>
    public double Epsilon { get; init; } = 1e-12;

    /// <summary>
    /// When true (default), collects sequence logits via <c>VerifyBatchLogits</c>.
    /// When false, evaluates step-by-step via <c>ForwardToken</c> + <c>CopyCurrentLogits</c>.
    /// </summary>
    public bool UseBatchLogitVerification { get; init; } = true;

    internal void Validate()
    {
        if (double.IsNaN(Threshold) || Threshold < 0.0)
            throw new ArgumentOutOfRangeException(nameof(Threshold), Threshold, "Threshold must be non-negative and not NaN.");

        if (MaxKlThreshold.HasValue && (double.IsNaN(MaxKlThreshold.Value) || MaxKlThreshold.Value < 0.0))
            throw new ArgumentOutOfRangeException(nameof(MaxKlThreshold), MaxKlThreshold.Value, "MaxKlThreshold must be non-negative and not NaN.");

        if (double.IsNaN(MinTop1Agreement) || MinTop1Agreement < 0.0 || MinTop1Agreement > 1.0)
            throw new ArgumentOutOfRangeException(nameof(MinTop1Agreement), MinTop1Agreement, "MinTop1Agreement must be in [0.0, 1.0].");

        if (!double.IsFinite(Epsilon) || Epsilon <= 0.0 || Epsilon >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(Epsilon), Epsilon, "Epsilon must be finite and in (0.0, 1.0).");
    }
}

/// <summary>
/// Summary report from comparing reference and candidate/quantized distributions via <c>D_KL(P_ref || Q_quant)</c>.
/// </summary>
public sealed record KlDivergenceResult
{
    public double MeanKlDivergence { get; init; }
    public double MaxKlDivergence { get; init; }
    public double MinKlDivergence { get; init; }
    public double Top1AgreementRate { get; init; }
    public int Top1Matches { get; init; }
    public int TokenCount { get; init; }
    public int VocabSize { get; init; }
    public double Threshold { get; init; }
    public double? MaxKlThreshold { get; init; }
    public double MinTop1Agreement { get; init; }
    public bool HasNonFiniteQuantizedValues { get; init; }
    public bool Passed { get; init; }
    public string? FailureReason { get; init; }
    public string? ReferenceDevice { get; init; }
    public string? CandidateDevice { get; init; }
    public TimeSpan Duration { get; init; }
}
