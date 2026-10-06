namespace Glacier.Inference.Engine;

using System;
using Glacier.Inference.Sampling;

/// <summary>
/// Speculative draft provider that uses a secondary smaller model (or CPU draft model)
/// to autoregressively propose candidate tokens.
/// </summary>
public sealed class ModelDraftProvider : IDraftProvider
{
    private readonly ISpeculativeTarget _session;
    private readonly bool _ownsSession;
    private int _lastContextLen;
    private int _lastContextToken = -1;

    public ISpeculativeTarget Session => _session;

    public ModelDraftProvider(ISpeculativeTarget session, bool ownsSession = false)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _ownsSession = ownsSession;
    }

    /// <inheritdoc />
    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
    {
        if (tokens.IsEmpty || maxDraftTokens <= 0 || draftTokens.IsEmpty)
        {
            return 0;
        }

        int count = Math.Min(maxDraftTokens, draftTokens.Length);

        // If context changed, length diverged, or last token differs, resync draft session via Prefill
        if (_lastContextLen != tokens.Length || _lastContextToken != tokens[^1])
        {
            _session.Prefill(tokens);
            _lastContextLen = tokens.Length;
            _lastContextToken = tokens[^1];
        }

        int pos = tokens.Length;

        // The first draft token is sampled from the logits already computed for tokens[^1] during Prefill/Forward
        int nextToken = _session.SampleNextToken(SamplingOptions.Greedy);
        draftTokens[0] = nextToken;

        for (int i = 1; i < count; i++)
        {
            _session.ForwardToken(nextToken, pos, computeLogits: true);
            pos++;
            nextToken = _session.SampleNextToken(SamplingOptions.Greedy);
            draftTokens[i] = nextToken;
        }

        // Forward the final drafted token so the draft model's KV cache and logits are ready
        _session.ForwardToken(nextToken, pos, computeLogits: true);
        _lastContextLen = pos + 1;
        _lastContextToken = nextToken;
        return count;
    }

    public void Dispose()
    {
        if (_ownsSession)
        {
            _session.Dispose();
        }
    }
}
