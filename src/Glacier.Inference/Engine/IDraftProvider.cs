namespace Glacier.Inference.Engine;

using System;
using Glacier.Inference.Sampling;

/// <summary>
/// Provider interface for drafting speculative candidate continuation tokens.
/// </summary>
public interface IDraftProvider : IDisposable
{
    /// <summary>
    /// Resets any cached KV or recurrent state held by the draft provider across independent generation requests.
    /// </summary>
    void Reset() { }

    /// <summary>
    /// Drafts candidate continuation tokens given the past sequence of tokens.
    /// </summary>
    /// <param name="tokens">Full sequence of prompt and generated tokens so far.</param>
    /// <param name="maxDraftTokens">Maximum number of candidate tokens to draft.</param>
    /// <param name="draftTokens">Destination span for drafted candidate tokens.</param>
    /// <returns>Number of candidate tokens drafted (0 if no speculation candidate).</returns>
    int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens);

    /// <summary>
    /// Drafts candidate continuation tokens along with their exact draft probability distributions <c>q_i</c>
    /// for Leviathan et al. non-greedy speculative rejection sampling.
    /// Deterministic draft providers (such as N-gram prompt lookup) default to one-hot distributions <c>q_i[draftTokens[i]] = 1.0f</c>.
    /// </summary>
    int DraftWithProbabilities(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        SamplingOptions sampling,
        Sampler sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        int drafted = Draft(tokens, maxDraftTokens, draftTokens);
        if (drafted <= 0) return 0;

        draftProbs.Slice(0, drafted * targetVocabSize).Clear();
        for (int i = 0; i < drafted; i++)
        {
            int tok = draftTokens[i];
            if ((uint)tok < (uint)targetVocabSize)
            {
                draftProbs[i * targetVocabSize + tok] = 1.0f;
            }
        }
        return drafted;
    }
}
