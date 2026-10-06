namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using Glacier.Inference.Engine;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

public class SpeculativeDecodingTests
{
    [Fact]
    public void PromptLookupDraftProvider_FindsMatchingNgramAndExtractsContinuation()
    {
        using var provider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        // Simulated token sequence:
        // "def fib(n): return fib(n" -> tokens: [10, 20, 30, 40, 50, 60, 10, 20]
        // Suffix [10, 20] occurred earlier at index 0..1, followed by [30, 40, 50, 60]
        int[] context = [10, 20, 30, 40, 50, 60, 10, 20];
        int[] draftBuf = new int[4];

        int drafted = provider.Draft(context.AsSpan(), maxDraftTokens: 4, draftBuf.AsSpan());

        Assert.Equal(4, drafted);
        Assert.Equal([30, 40, 50, 60], draftBuf);
    }

    [Fact]
    public void PromptLookupDraftProvider_ReturnsZeroWhenNoMatchFound()
    {
        using var provider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        int[] context = [1, 2, 3, 4, 5, 6, 7, 8];
        int[] draftBuf = new int[4];

        int drafted = provider.Draft(context.AsSpan(), maxDraftTokens: 4, draftBuf.AsSpan());

        Assert.Equal(0, drafted);
    }

    [Fact]
    public void PromptLookupDraftProvider_RespectsMaxDraftTokensLimit()
    {
        using var provider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        int[] context = [100, 200, 1, 2, 3, 4, 5, 6, 7, 100, 200];
        int[] draftBuf = new int[2]; // only room for 2

        int drafted = provider.Draft(context.AsSpan(), maxDraftTokens: 2, draftBuf.AsSpan());

        Assert.Equal(2, drafted);
        Assert.Equal(1, draftBuf[0]);
        Assert.Equal(2, draftBuf[1]);
    }

    [Fact]
    public void SpeculativeMetrics_ComputesAccurateRatios()
    {
        var metrics = new SpeculativeMetrics
        {
            TotalTokensGenerated = 20,
            DraftTokensProposed = 16,
            DraftTokensAccepted = 12,
            SpeculativeSteps = 5,
            SerialFallbackSteps = 1
        };

        Assert.Equal(0.75, metrics.AcceptanceRate, precision: 4);
        Assert.Equal(4.0, metrics.AverageTokensPerStep, precision: 4);
        Assert.Equal(20.0 / 6.0, metrics.EstimatedSpeedup, precision: 4);
    }

    [Fact]
    public void SpeculativeOptions_DefaultParametersAreValid()
    {
        var options = new SpeculativeOptions();

        Assert.Equal(4, options.MaxDraftTokens);
        Assert.Equal(2, options.MinNgramMatch);
        Assert.Equal(4, options.MaxNgramMatch);
        Assert.Equal(256, options.MaxTokens);
        Assert.NotNull(options.Sampling);
    }

    [Fact]
    public void PromptLookup_MultiNgramFallback_MatchesShorterLength()
    {
        using var provider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        // Sequence where 4-gram and 3-gram don't match, but 2-gram does:
        // Index: 0, 1, 2, 3,  4,  5, 6, 7
        // Tokens: 9, 8, 7, 6, 99, 88, 7, 6
        // Suffix [7, 6] (len 2) matches index 2..3, followed by [99, 88]
        int[] context = [9, 8, 7, 6, 99, 88, 7, 6];
        int[] draftBuf = new int[4];

        int drafted = provider.Draft(context.AsSpan(), maxDraftTokens: 4, draftBuf.AsSpan());

        Assert.True(drafted >= 2);
        Assert.Equal(99, draftBuf[0]);
        Assert.Equal(88, draftBuf[1]);
    }

    [Fact]
    public async System.Threading.Tasks.Task SpeculativeEngine_FullDraftAcceptance_EmitsAllCandidatesPlusBonus()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var target = new MockSpeculativeTarget(tokenizer);

        // Pre-program target:
        // Prefill emits token 4 ("A")
        target.EnqueueSample(4);

        // Program verification:
        // verifyBatch is [4, 5, 6] (where 5="B", 6="C" are drafted)
        // predictions are [5, 6, 2] (matches 5, matches 6, bonus is EOS 2)
        target.SetVerifyHandler((tokens, pos, preds) =>
        {
            preds[0] = 5; // matches draft[0]
            preds[1] = 6; // matches draft[1]
            preds[2] = 2; // bonus token is EOS
        });

        using var fixedDraft = new FixedDraftProvider([5, 6]);
        using var engine = new SpeculativeEngine(target, fixedDraft);

        var options = new SpeculativeOptions { MaxDraftTokens = 2, MaxTokens = 10 };
        var result = await engine.GenerateAsync("Prompt", options, formatChat: false);

        Assert.Equal("stop", result.FinishReason);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensAccepted);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensProposed);
        Assert.Equal(1.0, result.SpeculativeMetrics.AcceptanceRate);
        Assert.Equal(1, result.SpeculativeMetrics.SpeculativeSteps);
    }

    [Fact]
    public async System.Threading.Tasks.Task SpeculativeEngine_PartialDraftRejection_RollsBackAndEmitsCorrection()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var target = new MockSpeculativeTarget(tokenizer);

        target.EnqueueSample(4); // First token: "A"

        // Draft proposes [5, 6] ("B", "C")
        // Target verifies:
        // preds[0] = 5 (matches candidate 5)
        // preds[1] = 2 (EOS correction! Mismatches candidate 6)
        target.SetVerifyHandler((tokens, pos, preds) =>
        {
            preds[0] = 5; // match
            preds[1] = 2; // correction is EOS
        });

        using var fixedDraft = new FixedDraftProvider([5, 6]);
        using var engine = new SpeculativeEngine(target, fixedDraft);

        var options = new SpeculativeOptions { MaxDraftTokens = 2, MaxTokens = 10 };
        var result = await engine.GenerateAsync("Prompt", options, formatChat: false);

        Assert.Equal("stop", result.FinishReason);
        Assert.Equal(1, result.SpeculativeMetrics.DraftTokensAccepted);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensProposed);
        Assert.Equal(0.5, result.SpeculativeMetrics.AcceptanceRate);
    }

    [Fact]
    public async System.Threading.Tasks.Task SpeculativeEngine_ZeroDraftRejection_EmitsImmediateCorrection()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var target = new MockSpeculativeTarget(tokenizer);

        target.EnqueueSample(4); // First token: "A"

        // Draft proposes [5, 6]
        // Target rejects first candidate immediately with EOS 2
        target.SetVerifyHandler((tokens, pos, preds) =>
        {
            preds[0] = 2; // immediate correction is EOS
            preds[1] = 7;
        });

        using var fixedDraft = new FixedDraftProvider([5, 6]);
        using var engine = new SpeculativeEngine(target, fixedDraft);

        var options = new SpeculativeOptions { MaxDraftTokens = 2, MaxTokens = 10 };
        var result = await engine.GenerateAsync("Prompt", options, formatChat: false);

        Assert.Equal("stop", result.FinishReason);
        Assert.Equal(0, result.SpeculativeMetrics.DraftTokensAccepted);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensProposed);
        Assert.Equal(0.0, result.SpeculativeMetrics.AcceptanceRate);
    }

    [Fact]
    public void ModelDraftProvider_DraftsConsecutiveTokens_WithoutDuplicatingLastToken()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var draftTarget = new MockSpeculativeTarget(tokenizer);

        // Queue sampled draft tokens: 5 ("B"), 6 ("C"), 7 ("D")
        draftTarget.EnqueueSample(5);
        draftTarget.EnqueueSample(6);
        draftTarget.EnqueueSample(7);

        using var provider = new ModelDraftProvider(draftTarget);

        int[] context = [3, 4]; // "Prompt", "A" (Length = 2, last token is 4 at pos 1)
        int[] draftBuf = new int[3];

        int drafted = provider.Draft(context.AsSpan(), maxDraftTokens: 3, draftBuf.AsSpan());

        Assert.Equal(3, drafted);
        Assert.Equal(5, draftBuf[0]);
        Assert.Equal(6, draftBuf[1]);
        Assert.Equal(7, draftBuf[2]);

        // Prefill was called with [3, 4]
        Assert.Equal([3, 4], draftTarget.PrefilledTokens);

        // Verify that token 4 was NOT forwarded again at pos 2!
        // The first ForwardToken should be token 5 at pos 2, and token 6 at pos 3, and token 7 at pos 4!
        Assert.Equal(3, draftTarget.ForwardedTokens.Count);
        Assert.Equal((5, 2), draftTarget.ForwardedTokens[0]);
        Assert.Equal((6, 3), draftTarget.ForwardedTokens[1]);
        Assert.Equal((7, 4), draftTarget.ForwardedTokens[2]);
    }

    [Fact]
    public void ModelDraftProvider_ResyncsViaPrefill_WhenContextDivergesOrRewinds()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var draftTarget = new MockSpeculativeTarget(tokenizer);

        draftTarget.EnqueueSample(5);
        draftTarget.EnqueueSample(6);

        using var provider = new ModelDraftProvider(draftTarget);

        int[] context1 = [3, 4];
        int[] draftBuf = new int[2];
        provider.Draft(context1.AsSpan(), 2, draftBuf.AsSpan());
        Assert.Equal([3, 4], draftTarget.PrefilledTokens);

        // Now simulate target model rejecting token 5 and correcting it to token 8 ("E")
        // New context: [3, 4, 8]
        draftTarget.EnqueueSample(5);
        int[] context2 = [3, 4, 8];
        int[] draftBuf2 = new int[1];
        provider.Draft(context2.AsSpan(), 1, draftBuf2.AsSpan());

        // Must have re-prefilled with context2!
        Assert.Equal([3, 4, 8], draftTarget.PrefilledTokens);
    }

    [Fact]
    public async System.Threading.Tasks.Task SpeculativeEngine_WithModelDraftProvider_ExecutesEndToEnd()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "Prompt", "A", "B", "C", "D", "E" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 2);
        using var target = new MockSpeculativeTarget(tokenizer);
        using var draftTarget = new MockSpeculativeTarget(tokenizer);

        // Target emits token 4 ("A") from prompt prefill
        target.EnqueueSample(4);

        // Draft target emits [5, 6] ("B", "C")
        draftTarget.EnqueueSample(5);
        draftTarget.EnqueueSample(6);

        // Target verification matches 5, matches 6, and bonus is EOS 2
        target.SetVerifyHandler((tokens, pos, preds) =>
        {
            preds[0] = 5;
            preds[1] = 6;
            preds[2] = 2; // EOS bonus token
        });

        using var draftProvider = new ModelDraftProvider(draftTarget);
        using var engine = new SpeculativeEngine(target, draftProvider);

        var options = new SpeculativeOptions { MaxDraftTokens = 2, MaxTokens = 10 };
        var result = await engine.GenerateAsync("Prompt", options, formatChat: false);

        Assert.Equal("stop", result.FinishReason);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensAccepted);
        Assert.Equal(2, result.SpeculativeMetrics.DraftTokensProposed);
        Assert.Equal(1.0, result.SpeculativeMetrics.AcceptanceRate);
    }

    private sealed class FixedDraftProvider : IDraftProvider
    {
        private readonly int[] _tokens;
        private int _calls;

        public FixedDraftProvider(int[] tokens)
        {
            _tokens = tokens;
        }

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            if (_calls++ > 0) return 0; // only draft once
            int count = Math.Min(_tokens.Length, Math.Min(maxDraftTokens, draftTokens.Length));
            _tokens.AsSpan(0, count).CopyTo(draftTokens);
            return count;
        }

        public void Dispose() { }
    }

    private sealed class MockSpeculativeTarget : ISpeculativeTarget
    {
        public int MaxSeqLen => 1024;
        public BpeTokenizer Tokenizer { get; }
        public List<int> PrefilledTokens { get; } = new();
        public List<(int Token, int Pos)> ForwardedTokens { get; } = new();

        private readonly Queue<int> _sampleQueue = new();
        private Action<ReadOnlySpan<int>, int, Span<int>>? _verifyHandler;

        public MockSpeculativeTarget(BpeTokenizer tokenizer)
        {
            Tokenizer = tokenizer;
        }

        public void SetVerifyHandler(Action<ReadOnlySpan<int>, int, Span<int>> handler)
        {
            _verifyHandler = handler;
        }

        public void EnqueueSample(int token) => _sampleQueue.Enqueue(token);

        public void ResetKvCache() { }

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            PrefilledTokens.Clear();
            for (int i = 0; i < promptTokens.Length; i++)
            {
                PrefilledTokens.Add(promptTokens[i]);
            }
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            return _sampleQueue.Count > 0 ? _sampleQueue.Dequeue() : 2;
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            if (_verifyHandler != null)
            {
                _verifyHandler(tokens, startPos, predictedTokens);
            }
            else
            {
                for (int i = 0; i < predictedTokens.Length; i++)
                {
                    predictedTokens[i] = tokens[i];
                }
            }
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            ForwardedTokens.Add((token, pos));
        }

        public void Dispose() { }
    }
}
