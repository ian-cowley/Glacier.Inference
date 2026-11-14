namespace Glacier.Inference.Model;

using System;

/// <summary>
/// Runtime hook for low-rank Sleep-consolidated MoE expert banks.
/// Allows live inference loops to query active sleep-trained adapters with zero managed allocations.
/// </summary>
public interface ISleepExpansionRouter
{
    int Dimension { get; }
    int Rank { get; }
    int ActiveExpertCount { get; }

    /// <summary>
    /// Computes the low-rank MoE delta contribution into <paramref name="deltaOutput"/>
    /// for the given <paramref name="input"/> activation vector.
    /// </summary>
    void ForwardDelta(ReadOnlySpan<float> input, Span<float> deltaOutput, Span<float> rankScratch);
}
