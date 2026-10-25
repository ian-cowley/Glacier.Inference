namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Sampling;
using Xunit;

public class SamplingTests
{
    [Fact]
    public void GreedySampler_ReturnsArgMaxIndex()
    {
        var sampler = new Sampler();
        float[] logits = [0.1f, -2.5f, 10.4f, 3.2f, 9.8f];

        int sampled = sampler.Sample(logits.AsSpan(), SamplingOptions.Greedy);
        Assert.Equal(2, sampled);
    }

    [Fact]
    public void RepetitionPenalty_PenalizesPastTokens()
    {
        var sampler = new Sampler();
        float[] logits = [5.0f, 5.0f, 5.0f];
        int[] recent = [1]; // token 1 was recently emitted

        var options = new SamplingOptions
        {
            Temperature = 0.0f, // greedy
            RepetitionPenalty = 2.0f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options, recent);
        // Token 1 logit was penalized from 5.0 to 2.5, so token 0 or 2 will be chosen instead
        Assert.NotEqual(1, sampled);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void TopKDisabled_DoesNotThrow_AndReturnsValidToken(int topK)
    {
        var sampler = new Sampler(seed: 1);
        float[] logits = [0.1f, 2.0f, 1.0f, -3.0f];
        var options = new SamplingOptions { Temperature = 1.0f, TopK = topK, TopP = 1.0f, RepetitionPenalty = 1.0f };

        int sampled = sampler.Sample(logits, options);
        Assert.InRange(sampled, 0, logits.Length - 1);
    }

    [Fact]
    public void NaNLogits_AreNeverChosenWhenGreedy()
    {
        var sampler = new Sampler();
        float[] logits = [float.NaN, 1f, 3f, float.NaN];
        Assert.Equal(2, sampler.Sample(logits, SamplingOptions.Greedy));
    }

    [Fact]
    public void TopPZero_KeepsBestToken()
    {
        var sampler = new Sampler(seed: 7);
        float[] logits = [0f, 5f, 1f];
        var options = new SamplingOptions { Temperature = 1.0f, TopK = 3, TopP = 0f, RepetitionPenalty = 1.0f };
        for (int i = 0; i < 20; i++)
        {
            float[] copy = (float[])logits.Clone();
            Assert.Equal(1, sampler.Sample(copy, options));
        }
    }

    [Fact]
    public void SameSeed_IsDeterministic()
    {
        float[] logits = [1f, 1.1f, 0.9f, 1.05f, 0.95f];
        var options = new SamplingOptions { Temperature = 1.0f, TopK = 5, TopP = 1.0f, RepetitionPenalty = 1.0f };
        var a = new Sampler(seed: 42);
        var b = new Sampler(seed: 42);
        for (int i = 0; i < 50; i++)
            Assert.Equal(a.Sample((float[])logits.Clone(), options), b.Sample((float[])logits.Clone(), options));
    }

    [Fact]
    public void TopK_SelectsWithinTopKCandidates()
    {
        var sampler = new Sampler(seed: 123);
        // Candidate 0: 100, Candidate 1: 90, others much lower
        float[] logits = [100f, 90f, 0f, -10f, -50f];
        var options = new SamplingOptions { Temperature = 0.5f, TopK = 2, TopP = 1.0f };

        for (int i = 0; i < 50; i++)
        {
            float[] copy = (float[])logits.Clone();
            int selected = sampler.Sample(copy, options);
            Assert.True(selected == 0 || selected == 1, $"Expected token 0 or 1, but got {selected}");
        }
    }

    [Fact]
    public void EmptyLogits_Throws()
    {
        var sampler = new Sampler();
        Assert.Throws<ArgumentException>(() => sampler.Sample(Span<float>.Empty, SamplingOptions.Greedy));
    }

    [Fact]
    public void RepetitionPenalty_DuplicateTokens_AppliesPenaltyOnlyOnce()
    {
        var sampler = new Sampler();
        float[] logits = [4.0f, 10.0f, 4.0f];
        // Token 1 appears multiple times in recent history
        int[] recentWithDuplicates = [1, 1, 1, 1];

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            RepetitionPenalty = 2.0f
        };

        sampler.Sample(logits.AsSpan(), options, recentWithDuplicates);
        // If penalized once: 10.0 / 2.0 = 5.0. If penalized multiple times: 10 / 2^4 = 0.625
        Assert.Equal(5.0f, logits[1]);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(float.NaN)]
    public void InvalidTemperature_Throws(float temp)
    {
        var sampler = new Sampler();
        float[] logits = [1.0f, 2.0f];
        var opt = new SamplingOptions { Temperature = temp };
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(logits, opt));
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    [InlineData(float.NaN)]
    public void InvalidTopP_Throws(float topP)
    {
        var sampler = new Sampler();
        float[] logits = [1.0f, 2.0f];
        var opt = new SamplingOptions { TopP = topP };
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(logits, opt));
    }

    [Theory]
    [InlineData(0.0f)]
    [InlineData(-1.5f)]
    [InlineData(float.NaN)]
    public void InvalidRepetitionPenalty_Throws(float rep)
    {
        var sampler = new Sampler();
        float[] logits = [1.0f, 2.0f];
        var opt = new SamplingOptions { RepetitionPenalty = rep };
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.Sample(logits, opt));
    }

    [Fact]
    public void LargeK_UsesArrayPool_AndSamplesSuccessfully()
    {
        var sampler = new Sampler(seed: 99);
        // Create 300 candidates (k > 128)
        float[] logits = new float[300];
        for (int i = 0; i < logits.Length; i++)
            logits[i] = (float)i;

        var opt = new SamplingOptions { Temperature = 1.0f, TopK = 200, TopP = 0.95f };
        int token = sampler.Sample(logits, opt);
        Assert.InRange(token, 0, 299);
    }

    [Theory]
    [InlineData(new int[] { 42, 42, 42 }, 1)]                         // Period 1 (repeating token)
    [InlineData(new int[] { 10, 20, 10, 20, 10, 20 }, 2)]             // Period 2 (alternating pair)
    [InlineData(new int[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 }, 3)]          // Period 3
    [InlineData(new int[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 }, 4)] // Period 4
    [InlineData(new int[] { 99, 88, 77, 10, 20, 10, 20, 10, 20 }, 2)] // Loop at tail after prefix
    public void DetectPeriodicReasoningLoop_DetectedCorrectPeriod(int[] tokens, int expectedPeriod)
    {
        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
        Assert.Equal(expectedPeriod, detected);
    }

    [Theory]
    [InlineData(new int[] { 1, 2, 3, 4, 5, 6, 7, 8 })]                // Distinct non-repeating sequence
    [InlineData(new int[] { 10, 20, 10, 20 })]                         // Only 2 repeats (minRepeats = 3)
    [InlineData(new int[] { 1, 2 })]                                   // Shorter than minRepeats
    [InlineData(new int[] { })]                                       // Empty history
    public void DetectPeriodicReasoningLoop_NonRepeating_ReturnsZero(int[] tokens)
    {
        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
        Assert.Equal(0, detected);
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_OptInMinPeriodTwo_IgnoresPeriodOnePunctuation()
    {
        // 3 consecutive periods / dots (period 1)
        int[] ellipsisTokens = [13, 13, 13];
        // By default (minPeriod = 1), period 1 is detected
        int detectedDefault = Sampler.DetectPeriodicReasoningLoop(ellipsisTokens.AsSpan());
        Assert.Equal(1, detectedDefault);

        // When opt-in minPeriod: 2 is explicitly passed, period 1 is ignored
        int detectedOptIn = Sampler.DetectPeriodicReasoningLoop(ellipsisTokens.AsSpan(), minPeriod: 2);
        Assert.Equal(0, detectedOptIn);

        // Alternating pair (period 2) is detected with opt-in minPeriod: 2
        int[] alternatingTokens = [10, 20, 10, 20, 10, 20];
        int detectedPair = Sampler.DetectPeriodicReasoningLoop(alternatingTokens.AsSpan(), minPeriod: 2);
        Assert.Equal(2, detectedPair);
    }

    [Fact]
    public void Sample_WithOptInMinPeriodTwo_IgnoresPeriodOneLoop()
    {
        var sampler = new Sampler();
        float[] logits = [55.0f, 100.0f, 90.0f, 10.0f, 10.0f];
        int[] recentLoop = [1, 1, 1]; // Period 1

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 2.0f,
            ReasoningLoopMinPeriod = 2 // Explicit opt-in: ignore period 1
        };

        // When minPeriod is 2, period-1 loop is ignored, so token 1 retains highest logit
        int sampled = sampler.Sample(logits.AsSpan(), options, recentLoop.AsSpan());
        Assert.Equal(1, sampled);
    }

    [Fact]
    public void Sample_WithPeriodicLoopInRecentTokens_AppliesLoopRecoveryPenalty_BreaksLoop()
    {
        var sampler = new Sampler();
        // Vocab of 5 tokens: [0, 1, 2, 3, 4]
        // Token 1 has highest unpenalized logit (100.0), Token 2 has 90.0, Token 0 has 55.0
        float[] logits = [55.0f, 100.0f, 90.0f, 10.0f, 10.0f];

        // Recent history contains a repeating cycle of [1, 2, 1, 2, 1, 2]
        int[] recentLoop = [1, 2, 1, 2, 1, 2];

        var options = new SamplingOptions
        {
            Temperature = 0.0f, // Greedy decoding
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 2.0f // Penalizes 100.0 -> 50.0, 90.0 -> 45.0
        };

        int sampled = sampler.Sample(logits.AsSpan(), options, recentLoop);

        // Loop recovery penalty was applied to tokens 1 and 2:
        // logits[1] = 50.0, logits[2] = 45.0, logits[0] = 55.0.
        // Token 0 must be selected, breaking the loop!
        Assert.Equal(0, sampled);
    }

    [Fact]
    public void Sample_WithNonLoopingRecentTokens_DoesNotApplyLoopRecoveryPenalty_SelectsTopCandidate()
    {
        var sampler = new Sampler();
        float[] logits = [55.0f, 100.0f, 90.0f, 10.0f, 10.0f];
        // Non-periodic, healthy sequence
        int[] normalTokens = [3, 4, 0, 4, 3, 0];

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 2.0f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options, normalTokens);

        // No loop detected; Token 1 remains top candidate (100.0)
        Assert.Equal(1, sampled);
    }

    [Fact]
    public void Sample_WithLoopGuardDisabled_PermitsLoopContinuation()
    {
        var sampler = new Sampler();
        float[] logits = [55.0f, 100.0f, 90.0f, 10.0f, 10.0f];
        int[] recentLoop = [1, 2, 1, 2, 1, 2];

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = false, // Guard explicitly disabled
            LoopRecoveryRepetitionPenalty = 2.0f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options, recentLoop);

        // Guard disabled: Token 1 sampled despite loop
        Assert.Equal(1, sampled);
    }

    [Fact]
    public void ClampReasoningLogits_PenalizesNonEndThinkTokens_AndBiasesEndThinkToken()
    {
        float[] logits = [10.0f, 20.0f, 30.0f, 5.0f];
        int endThinkTokenId = 1;

        Sampler.ClampReasoningLogits(logits.AsSpan(), endThinkTokenId, penalty: 50.0f, bias: 20.0f);

        // Token 1 was biased: 20.0 + 20.0 = 40.0
        Assert.Equal(40.0f, logits[1]);
        // All other tokens were penalized: 10 - 50 = -40, 30 - 50 = -20, 5 - 50 = -45
        Assert.Equal(-40.0f, logits[0]);
        Assert.Equal(-20.0f, logits[2]);
        Assert.Equal(-45.0f, logits[3]);
    }
}
