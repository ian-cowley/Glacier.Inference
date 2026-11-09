namespace Glacier.Inference.Continuum;

using System;

/// <summary>
/// Core contract for a HOPE / Titans continuum associative memory module operating at a specified clock frequency.
/// </summary>
public interface IContinuumMemoryModule : IDisposable
{
    /// <summary>
    /// Gets the clock frequency tier of this memory module.
    /// </summary>
    ContinuumFrequency Frequency { get; }

    /// <summary>
    /// Gets the vector dimension <c>d</c> of the <c>[d x d]</c> associative state matrices.
    /// </summary>
    int Dimension { get; }

    /// <summary>
    /// Gets the token stride at which this module performs scheduled updates.
    /// </summary>
    int ClockStride { get; }

    /// <summary>
    /// Gets the number of associative state updates executed by this module.
    /// </summary>
    long SequenceTokenClock { get; }

    /// <summary>
    /// Gets the most recent surprise metric <c>||M_{t-1} k_t - v_t||_2</c> observed during state update.
    /// </summary>
    float LastSurpriseMetric { get; }

    /// <summary>
    /// Gets the total unmanaged bytes allocated by this module.
    /// </summary>
    long TotalUnmanagedBytes { get; }

    /// <summary>
    /// Evaluates the associative memory forward pass <c>output = M_t * query</c> without mutating state.
    /// </summary>
    void Forward(ReadOnlySpan<float> query, Span<float> output);

    /// <summary>
    /// Updates the coupled momentum <c>S_t</c> and associative memory <c>M_t</c> using the Titans recurrence:
    /// <c>r_t = M_{t-1} k_t - v_t</c>, <c>S_t = theta * S_{t-1} - eta * r_t k_t^T</c>, <c>M_t = lambda * M_{t-1} + S_t</c>.
    /// Returns the L2 surprise metric <c>||r_t||_2</c>.
    /// </summary>
    float UpdateAssociativeState(
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        float? lambdaOverride = null,
        float? thetaOverride = null,
        float? etaOverride = null);

    /// <summary>
    /// Resets volatile associative memory <c>M_t</c>, momentum <c>S_t</c>, and adjoint states to zero (Synaptic Pruning).
    /// </summary>
    void ResetVolatileState();

    /// <summary>
    /// Saves the complete unmanaged state (header + <c>M_t</c> + <c>S_t</c>) into the destination buffer.
    /// </summary>
    void SaveState(Span<byte> destination);

    /// <summary>
    /// Restores the complete unmanaged state (header + <c>M_t</c> + <c>S_t</c>) from the source buffer.
    /// </summary>
    void RestoreState(ReadOnlySpan<byte> source);
}
