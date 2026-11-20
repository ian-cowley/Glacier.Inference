namespace Glacier.Inference.Engine;

using System;
using System.Collections.Generic;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Controls automatic and explicit reasoning (&lt;think&gt;) token budgets so models
/// cannot exhaust MaxTokens inside thinking tags before emitting a final answer.
/// </summary>
public static class ReasoningBudgetPolicy
{
    /// <summary>
    /// Sentinel value in <see cref="Glacier.Inference.Sampling.SamplingOptions.MaxReasoningTokens"/>
    /// indicating automatic scaling from <c>MaxTokens</c>.
    /// </summary>
    public const int AutoBudgetSentinel = -1;

    /// <summary>
    /// Computes the automatic reasoning token budget for a given generation <paramref name="maxTokens"/> limit.
    /// Reserves closing-tag and final-answer headroom.
    /// </summary>
    public static int ComputeAutoMaxReasoningTokens(int maxTokens)
    {
        if (maxTokens <= 1)
        {
            return 0;
        }

        int minAnswerReserve = maxTokens > 256
            ? Math.Min(256, maxTokens / 4)
            : Math.Max(1, maxTokens / 3);
        int targetShare = Math.Max(1, (int)(maxTokens * 0.65));
        return Math.Max(1, Math.Min(targetShare, maxTokens - minAnswerReserve));
    }

    /// <summary>
    /// Resolves a configured reasoning budget value against <paramref name="maxTokens"/>.
    /// Negative values activate automatic sizing; zero disables capping; positive values cap directly.
    /// </summary>
    public static int ResolveBudget(int configuredBudget, int maxTokens)
    {
        if (configuredBudget < 0)
        {
            return ComputeAutoMaxReasoningTokens(maxTokens);
        }

        return configuredBudget;
    }

    /// <summary>
    /// Parses a CLI or API reasoning budget string (<c>auto</c>, <c>off</c>, <c>0</c>, or positive integer).
    /// Returns <see cref="AutoBudgetSentinel"/> for <c>auto</c>, <c>0</c> for <c>off</c>/<c>0</c>, or the integer cap.
    /// </summary>
    public static bool TryParseSetting(string? raw, out int setting)
    {
        if (string.IsNullOrWhiteSpace(raw) || raw.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            setting = AutoBudgetSentinel;
            return true;
        }

        if (raw.Equals("off", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("none", StringComparison.OrdinalIgnoreCase) ||
            raw.Equals("unlimited", StringComparison.OrdinalIgnoreCase))
        {
            setting = 0;
            return true;
        }

        if (int.TryParse(raw.Trim(), out int parsed) && parsed >= 0)
        {
            setting = parsed;
            return true;
        }

        setting = 0;
        return false;
    }

    /// <summary>
    /// Builds the token sequence for <c>\n&lt;/think&gt;\n\n</c>, guaranteeing the closing special token
    /// ID is included verbatim when present in the model vocabulary.
    /// </summary>
    public static int[] BuildEndThinkTokens(BpeTokenizer tokenizer, int endThinkId)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);

        if (endThinkId >= 0)
        {
            int[] prefix = tokenizer.Encode("\n");
            int[] suffix = tokenizer.Encode("\n\n");
            int[] tokens = new int[prefix.Length + 1 + suffix.Length];
            prefix.CopyTo(tokens, 0);
            tokens[prefix.Length] = endThinkId;
            suffix.CopyTo(tokens, prefix.Length + 1);
            return tokens;
        }

        return tokenizer.Encode("\n</think>\n\n");
    }
}

/// <summary>
/// Tracks per-generation <c>&lt;think&gt;</c> state transitions and detects when the reasoning budget is reached.
/// Supports both single special-token IDs (e.g., Qwen3) and multi-token BPE tags (e.g., Qwen2.5).
/// </summary>
public struct ReasoningBudgetTracker
{
    private const string OpenTag = "<think>";
    private const string CloseTag = "</think>";

    private int _scannedTextLen;
    private int _minTagSearchPos;

    public int ThinkId { get; }
    public int EndThinkId { get; }
    public int MaxReasoningTokens { get; }
    public bool InThink { get; private set; }
    public int ReasoningTokens { get; private set; }
    public bool ForcedThinkClose { get; private set; }

    public ReasoningBudgetTracker(
        int thinkId,
        int endThinkId,
        int maxReasoningTokens,
        ReadOnlySpan<int> promptTokens,
        string? promptText = null)
    {
        ThinkId = thinkId;
        EndThinkId = endThinkId;
        MaxReasoningTokens = maxReasoningTokens;
        InThink = false;
        ReasoningTokens = 0;
        ForcedThinkClose = false;
        _scannedTextLen = 0;
        _minTagSearchPos = 0;

        if (thinkId >= 0 && endThinkId >= 0)
        {
            for (int i = 0; i < promptTokens.Length; i++)
            {
                int pt = promptTokens[i];
                if (pt == thinkId)
                {
                    InThink = true;
                }
                else if (pt == endThinkId)
                {
                    InThink = false;
                }
            }
        }
        else if (!string.IsNullOrEmpty(promptText))
        {
            int lastOpen = promptText.LastIndexOf(OpenTag, StringComparison.Ordinal);
            int lastClose = promptText.LastIndexOf(CloseTag, StringComparison.Ordinal);
            if (lastOpen >= 0 && lastOpen > lastClose)
            {
                InThink = true;
            }
        }
    }

    public void MarkForcedClose()
    {
        ForcedThinkClose = true;
        InThink = false;
    }

    /// <summary>
    /// Observes a sampled token ID (for models with dedicated single-token think delimiters)
    /// and returns <c>true</c> if the caller should now force-inject <c>\n&lt;/think&gt;\n\n</c>.
    /// </summary>
    public bool ObserveTokenAndShouldForceClose(int tokenId)
    {
        if (MaxReasoningTokens <= 0 || ForcedThinkClose)
        {
            return false;
        }

        if (ThinkId >= 0 && tokenId == ThinkId)
        {
            InThink = true;
            ReasoningTokens = 0;
            return false;
        }

        if (EndThinkId >= 0 && tokenId == EndThinkId)
        {
            InThink = false;
            return false;
        }

        return IncrementReasoningAndCheckLimit();
    }

    /// <summary>
    /// Observes the decoded generation buffer for models where <c>&lt;think&gt;</c> or <c>&lt;/think&gt;</c>
    /// splits into multiple BPE tokens, returning <c>true</c> if the caller should force-inject <c>\n&lt;/think&gt;\n\n</c>.
    /// </summary>
    public bool ObserveDecodedTextAndShouldForceClose(System.Text.StringBuilder responseSb)
    {
        ArgumentNullException.ThrowIfNull(responseSb);

        int curLen = responseSb.Length;
        if (MaxReasoningTokens <= 0 || ForcedThinkClose)
        {
            _scannedTextLen = curLen;
            return false;
        }

        int scanStart = Math.Max(_minTagSearchPos, _scannedTextLen - (CloseTag.Length - 1));
        int windowLen = curLen - scanStart;
        bool sawTag = false;

        if (windowLen >= OpenTag.Length)
        {
            Span<char> stackBuf = windowLen <= 256 ? stackalloc char[windowLen] : default;
            char[]? rented = windowLen > 256 ? new char[windowLen] : null;
            Span<char> window = rented is null ? stackBuf : rented.AsSpan();
            responseSb.CopyTo(scanStart, window, windowLen);

            int cursor = 0;
            while (cursor <= window.Length - OpenTag.Length)
            {
                ReadOnlySpan<char> rem = window.Slice(cursor);
                int openRel = rem.IndexOf(OpenTag.AsSpan(), StringComparison.Ordinal);
                int closeRel = rem.IndexOf(CloseTag.AsSpan(), StringComparison.Ordinal);

                if (openRel < 0 && closeRel < 0)
                {
                    break;
                }

                sawTag = true;
                if (openRel >= 0 && (closeRel < 0 || openRel < closeRel))
                {
                    InThink = true;
                    ReasoningTokens = 0;
                    cursor += openRel + OpenTag.Length;
                }
                else
                {
                    InThink = false;
                    cursor += closeRel + CloseTag.Length;
                }

                _minTagSearchPos = scanStart + cursor;
            }
        }

        _scannedTextLen = curLen;
        if (sawTag)
        {
            return false;
        }

        return IncrementReasoningAndCheckLimit();
    }

    private bool IncrementReasoningAndCheckLimit()
    {
        if (!InThink)
        {
            return false;
        }

        ReasoningTokens++;
        if (ReasoningTokens >= MaxReasoningTokens)
        {
            ForcedThinkClose = true;
            InThink = false;
            return true;
        }

        return false;
    }
}
