namespace Glacier.Inference.Engine;

using System;

/// <summary>
/// Provider interface for drafting speculative candidate continuation tokens.
/// </summary>
public interface IDraftProvider : IDisposable
{
    /// <summary>
    /// Drafts candidate continuation tokens given the past sequence of tokens.
    /// </summary>
    /// <param name="tokens">Full sequence of prompt and generated tokens so far.</param>
    /// <param name="maxDraftTokens">Maximum number of candidate tokens to draft.</param>
    /// <param name="draftTokens">Destination span for drafted candidate tokens.</param>
    /// <returns>Number of candidate tokens drafted (0 if no speculation candidate).</returns>
    int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens);
}
