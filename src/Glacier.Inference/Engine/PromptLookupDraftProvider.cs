namespace Glacier.Inference.Engine;

using System;

/// <summary>
/// Speculative draft provider based on Prompt Lookup / N-gram matching (Assisted Generation).
/// Scans previous prompt and generation history for matching token suffixes and proposes
/// subsequent tokens as speculative candidates in sub-microsecond time with zero extra VRAM.
/// </summary>
public sealed class PromptLookupDraftProvider : IDraftProvider
{
    private readonly int _minMatchLength;
    private readonly int _maxMatchLength;

    public PromptLookupDraftProvider(int minMatchLength = 2, int maxMatchLength = 4)
    {
        if (minMatchLength < 1) throw new ArgumentOutOfRangeException(nameof(minMatchLength), "Min match length must be >= 1");
        if (maxMatchLength < minMatchLength) throw new ArgumentOutOfRangeException(nameof(maxMatchLength), "Max match length must be >= min match length");

        _minMatchLength = minMatchLength;
        _maxMatchLength = maxMatchLength;
    }

    /// <inheritdoc />
    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
    {
        if (tokens.Length <= _minMatchLength || maxDraftTokens <= 0 || draftTokens.IsEmpty)
        {
            return 0;
        }

        int maxL = Math.Min(_maxMatchLength, tokens.Length / 2);
        for (int l = maxL; l >= _minMatchLength; l--)
        {
            ReadOnlySpan<int> targetNgram = tokens.Slice(tokens.Length - l, l);
            int searchLimit = tokens.Length - l;

            // Search backward from most recent occurrences to preserve local phrase locality
            for (int i = searchLimit - l; i >= 0; i--)
            {
                if (tokens.Slice(i, l).SequenceEqual(targetNgram))
                {
                    int followStart = i + l;
                    int available = searchLimit - followStart;
                    if (available > 0)
                    {
                        int count = Math.Min(maxDraftTokens, Math.Min(available, draftTokens.Length));
                        tokens.Slice(followStart, count).CopyTo(draftTokens);
                        return count;
                    }
                }
            }
        }

        return 0;
    }

    public void Dispose()
    {
        // No unmanaged resources
    }
}
