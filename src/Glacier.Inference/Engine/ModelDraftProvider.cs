namespace Glacier.Inference.Engine;

using System;
using Glacier.Inference.Sampling;

/// <summary>
/// Speculative draft provider that uses a secondary smaller model (or CPU draft model)
/// to autoregressively propose candidate tokens.
/// </summary>
public sealed class ModelDraftProvider : IDraftProvider
{
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private int _lastContextLen;

    public InferenceSession Session => _session;

    public ModelDraftProvider(InferenceSession session, bool ownsSession = false)
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

        // If context changed or rewound, sync draft session
        if (_lastContextLen != tokens.Length - 1)
        {
            _session.Prefill(tokens);
            _lastContextLen = tokens.Length;
        }

        int currentToken = tokens[^1];
        int pos = tokens.Length;

        for (int i = 0; i < count; i++)
        {
            _session.ForwardToken(currentToken, pos, computeLogits: true);
            int nextToken = _session.SampleNextToken(SamplingOptions.Greedy);
            draftTokens[i] = nextToken;
            currentToken = nextToken;
            pos++;
        }

        _lastContextLen = pos;
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
