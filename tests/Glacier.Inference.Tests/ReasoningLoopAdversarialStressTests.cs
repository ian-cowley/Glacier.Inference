namespace Glacier.Inference.Tests;

using System;
using System.Buffers;
using Glacier.Inference.Sampling;
using Xunit;

/// <summary>
/// Adversarial stress tests and boundary condition validation for REC-1:
/// Reasoning-loop detection, logit clamping, and sampler robustness under extreme conditions.
/// </summary>
public sealed class ReasoningLoopAdversarialStressTests
{
    // =========================================================================
    // 1. Periods 1 to 64: Comprehensive repeating cycle verification
    // =========================================================================

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(16)]
    [InlineData(64)]
    public void DetectPeriodicReasoningLoop_MandatedPeriods_DetectedAccurately(int period)
    {
        // Construct repeating cycle with minimum 3 repeats
        int minRepeats = 3;
        int totalLen = period * minRepeats;
        int[] tokens = new int[totalLen];

        for (int r = 0; r < minRepeats; r++)
        {
            for (int k = 0; k < period; k++)
            {
                tokens[r * period + k] = 1000 + k;
            }
        }

        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
        Assert.Equal(period, detected);
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_AllPeriodsFrom1To64_DetectedAccurately()
    {
        for (int p = 1; p <= 64; p++)
        {
            for (int repeats = 3; repeats <= 5; repeats++)
            {
                int totalLen = p * repeats;
                int[] tokens = new int[totalLen];

                for (int r = 0; r < repeats; r++)
                {
                    for (int k = 0; k < p; k++)
                    {
                        tokens[r * p + k] = 5000 + k;
                    }
                }

                int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
                Assert.Equal(p, detected);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(16)]
    [InlineData(64)]
    public void DetectPeriodicReasoningLoop_WithLeadingNoisePrefix_DetectsTailLoop(int period)
    {
        var rng = new Random(42);
        int prefixLen = 150;
        int minRepeats = 3;
        int totalLen = prefixLen + period * minRepeats;
        int[] tokens = new int[totalLen];

        // Fill prefix with random noise distinct from period range
        for (int i = 0; i < prefixLen; i++)
        {
            tokens[i] = rng.Next(100, 500);
        }

        // Fill tail with repeating cycle
        for (int r = 0; r < minRepeats; r++)
        {
            for (int k = 0; k < period; k++)
            {
                tokens[prefixLen + r * period + k] = 5000 + k;
            }
        }

        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
        Assert.Equal(period, detected);
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_Period65_ReturnsZeroWhenMaxPeriodIs64()
    {
        int p = 65;
        int repeats = 3;
        int[] tokens = new int[p * repeats];
        for (int r = 0; r < repeats; r++)
        {
            for (int k = 0; k < p; k++)
            {
                tokens[r * p + k] = 100 + k;
            }
        }

        // When maxPeriod is default 64, period 65 exceeds maxPeriod and must return 0
        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 64, minRepeats: 3);
        Assert.Equal(0, detected);

        // When maxPeriod is expanded to 65, it detects 65
        int detectedCustom = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan(), minPeriod: 1, maxPeriod: 65, minRepeats: 3);
        Assert.Equal(65, detectedCustom);
    }

    // =========================================================================
    // 2. Degenerate token streams: empty, short, 10k tokens, broken cycles
    // =========================================================================

    [Fact]
    public void DetectPeriodicReasoningLoop_DegenerateShortStreams_ReturnZeroSafely()
    {
        // Empty span
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(ReadOnlySpan<int>.Empty));

        // 1 token
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(new int[] { 42 }));

        // 2 tokens (< minRepeats=3)
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(new int[] { 42, 42 }));

        // 2 distinct tokens
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(new int[] { 10, 20 }));
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_InvalidArguments_ReturnZeroWithoutException()
    {
        int[] sample = [1, 1, 1, 1];
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(sample, minPeriod: 0));
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(sample, minPeriod: -1));
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(sample, minPeriod: 10, maxPeriod: 5)); // maxPeriod < minPeriod
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(sample, minRepeats: 1)); // minRepeats < 2
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(sample, minRepeats: 0));
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_10000RandomTokens_ReturnsZeroHighPerformance()
    {
        var rng = new Random(12345);
        int[] tokens = new int[10_000];
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] = rng.Next(1, 100_000);
        }

        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan());
        Assert.Equal(0, detected);
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_10000Tokens_WithPeriod16AtTail_Detects16()
    {
        var rng = new Random(54321);
        int[] tokens = new int[10_000];
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] = rng.Next(1, 100_000);
        }

        // Overwrite tail with 16-token loop repeated 4 times (64 tokens)
        int p = 16;
        int repeats = 4;
        int loopLen = p * repeats;
        int start = tokens.Length - loopLen;

        for (int r = 0; r < repeats; r++)
        {
            for (int k = 0; k < p; k++)
            {
                tokens[start + r * p + k] = 77000 + k;
            }
        }

        int detected = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan());
        Assert.Equal(16, detected);
    }

    [Fact]
    public void DetectPeriodicReasoningLoop_BrokenCyclesWithNoise_ReturnsZero()
    {
        // 3-token cycle interrupted on final token: [1, 2, 3, 1, 2, 3, 1, 2, 999]
        int[] interruptedTail = [1, 2, 3, 1, 2, 3, 1, 2, 999];
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(interruptedTail.AsSpan()));

        // Cycle interrupted in middle repeat: [1, 2, 3, 1, 999, 3, 1, 2, 3]
        int[] interruptedMid = [1, 2, 3, 1, 999, 3, 1, 2, 3];
        Assert.Equal(0, Sampler.DetectPeriodicReasoningLoop(interruptedMid.AsSpan()));
    }

    // =========================================================================
    // 3. Out-of-bounds endThinkTokenId in ClampReasoningLogits
    // =========================================================================

    [Theory]
    [InlineData(-1)]
    [InlineData(-999)]
    [InlineData(int.MinValue)]
    [InlineData(50)]       // Exact length
    [InlineData(51)]       // Length + 1
    [InlineData(100_000)]  // Huge positive
    [InlineData(int.MaxValue)]
    public void ClampReasoningLogits_OutOfBoundsEndThinkTokenId_NoExceptionAndLogitsUnchanged(int oobTokenId)
    {
        float[] original = new float[50];
        for (int i = 0; i < original.Length; i++) original[i] = 1.5f * i;
        float[] copy = (float[])original.Clone();

        // Must not throw, and must return without modifying logits
        Sampler.ClampReasoningLogits(copy.AsSpan(), oobTokenId, penalty: 50.0f, bias: 20.0f);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], copy[i]);
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(10)]
    public void ClampReasoningLogits_EmptySpan_NoException(int tokenId)
    {
        // Empty span must not throw for any token ID
        Sampler.ClampReasoningLogits(Span<float>.Empty, tokenId);
    }

    [Fact]
    public void ClampReasoningLogits_ValidTokenId_AppliesMathematicalTransformsExactly()
    {
        float[] logits = [10.0f, 20.0f, 30.0f, 40.0f, 50.0f];
        int endThinkTokenId = 2; // initial logit 30.0f
        float penalty = 50.0f;
        float bias = 20.0f;

        Sampler.ClampReasoningLogits(logits.AsSpan(), endThinkTokenId, penalty, bias);

        // Target index 2 receives +20.0f: 30.0 + 20.0 = 50.0f
        Assert.Equal(50.0f, logits[2]);

        // All other indices receive -50.0f:
        Assert.Equal(10.0f - 50.0f, logits[0]);
        Assert.Equal(20.0f - 50.0f, logits[1]);
        Assert.Equal(40.0f - 50.0f, logits[3]);
        Assert.Equal(50.0f - 50.0f, logits[4]);
    }

    // =========================================================================
    // 4. Extreme Logits: All -inf, all 0, all NaN, subnormal, mixed
    // =========================================================================

    [Fact]
    public void SanitizeLogits_AllNegativeInfinity_ConvertsAllToMinus1e9()
    {
        float[] logits = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        int repaired = Sampler.SanitizeLogits(logits.AsSpan());

        Assert.Equal(3, repaired);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(-1e9f, logits[i]);
        }
    }

    [Fact]
    public void SanitizeLogits_AllNaN_ConvertsAllToMinus1e9()
    {
        float[] logits = [float.NaN, float.NaN, float.NaN, float.NaN];
        int repaired = Sampler.SanitizeLogits(logits.AsSpan());

        Assert.Equal(4, repaired);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(-1e9f, logits[i]);
            Assert.False(float.IsNaN(logits[i]));
        }
    }

    [Fact]
    public void SanitizeLogits_AllZero_LeavesUntouched()
    {
        float[] logits = [0f, 0f, 0f];
        int repaired = Sampler.SanitizeLogits(logits.AsSpan());

        Assert.Equal(0, repaired);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(0f, logits[i]);
        }
    }

    [Theory]
    [InlineData(0.0f, 1)]      // Greedy
    [InlineData(0.7f, 40)]     // Stochastic top-K
    [InlineData(1.0f, 0)]      // Stochastic full vocab
    public void Sample_AllNegativeInfinity_DoesNotCrashAndReturnsValidToken(float temperature, int topK)
    {
        var sampler = new Sampler(seed: 42);
        float[] logits = new float[100];
        Array.Fill(logits, float.NegativeInfinity);

        var options = new SamplingOptions
        {
            Temperature = temperature,
            TopK = topK,
            TopP = 0.9f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options);
        Assert.InRange(sampled, 0, logits.Length - 1);
    }

    [Theory]
    [InlineData(0.0f, 1)]      // Greedy
    [InlineData(0.7f, 40)]     // Stochastic top-K
    [InlineData(1.0f, 0)]      // Stochastic full vocab
    public void Sample_AllNaN_DoesNotCrashAndReturnsValidToken(float temperature, int topK)
    {
        var sampler = new Sampler(seed: 42);
        float[] logits = new float[100];
        Array.Fill(logits, float.NaN);

        var options = new SamplingOptions
        {
            Temperature = temperature,
            TopK = topK,
            TopP = 0.9f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options);
        Assert.InRange(sampled, 0, logits.Length - 1);
    }

    [Theory]
    [InlineData(0.0f, 1)]      // Greedy
    [InlineData(0.7f, 40)]     // Stochastic top-K
    [InlineData(1.0f, 0)]      // Stochastic full vocab
    public void Sample_AllZeros_DoesNotCrashAndReturnsValidToken(float temperature, int topK)
    {
        var sampler = new Sampler(seed: 42);
        float[] logits = new float[100];
        Array.Fill(logits, 0.0f);

        var options = new SamplingOptions
        {
            Temperature = temperature,
            TopK = topK,
            TopP = 0.9f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options);
        Assert.InRange(sampled, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_MixedExtremeLogits_SanitizesAndSamplesSafely()
    {
        var sampler = new Sampler(seed: 99);
        float[] logits =
        [
            float.NaN,
            float.NegativeInfinity,
            float.PositiveInfinity, // Sanitized to 1e4f -> will be strongly favored
            0.0f,
            -1e30f,
            1e-38f // subnormal float
        ];

        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 5,
            TopP = 0.95f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options);
        // Token 2 was +Infinity (sanitized to 1e4f, dominant logit)
        Assert.Equal(2, sampled);
    }

    [Fact]
    public void ClampReasoningLogits_OnExtremeLogits_FollowedBySample_SelectsEndThinkToken()
    {
        var sampler = new Sampler();
        float[] logits = new float[100];
        Array.Fill(logits, 0.0f);
        int endThinkTokenId = 42;

        // Clamp logits
        Sampler.ClampReasoningLogits(logits.AsSpan(), endThinkTokenId, penalty: 50.0f, bias: 20.0f);

        // Sample greedy
        int sampled = sampler.Sample(logits.AsSpan(), SamplingOptions.Greedy);
        Assert.Equal(endThinkTokenId, sampled);
    }

    // =========================================================================
    // 5. Zero-Allocation Hot Path Verification
    // =========================================================================

    [Fact]
    public void DetectPeriodicReasoningLoop_ZeroAllocationsOnHotPath()
    {
        int[] tokens = new int[10_000];
        Array.Fill(tokens, 42);

        // Warm up JIT
        _ = Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan());

        int sum = 0;
        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();
        for (int iter = 0; iter < 100; iter++)
        {
            sum += Sampler.DetectPeriodicReasoningLoop(tokens.AsSpan());
        }
        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(100, sum);
        long diff = afterAlloc - beforeAlloc;
        Assert.Equal(0, diff);
    }

    [Fact]
    public void ClampReasoningLogits_ZeroAllocationsOnHotPath()
    {
        float[] logits = new float[152_064]; // Typical Qwen2.5/DeepSeek vocab size

        // Warm up JIT and tiering
        for (int w = 0; w < 10; w++)
        {
            Sampler.ClampReasoningLogits(logits.AsSpan(), 151649);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Warm up on current thread after GC
        for (int w = 0; w < 5; w++)
        {
            Sampler.ClampReasoningLogits(logits.AsSpan(), 151649);
        }

        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();
        for (int iter = 0; iter < 100; iter++)
        {
            Sampler.ClampReasoningLogits(logits.AsSpan(), 151649);
        }
        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();

        long diff = afterAlloc - beforeAlloc;
        Assert.Equal(0, diff);
    }

    [Fact]
    public void Sample_GreedyWithLoopGuard_ZeroAllocationsOnHotPath()
    {
        var sampler = new Sampler(seed: 1);
        float[] logits = new float[1000];
        int[] recentTokens = [10, 20, 10, 20, 10, 20];

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 2.0f
        };

        // Warm up JIT
        _ = sampler.Sample(logits.AsSpan(), options, recentTokens.AsSpan());

        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();
        for (int iter = 0; iter < 50; iter++)
        {
            _ = sampler.Sample(logits.AsSpan(), options, recentTokens.AsSpan());
        }
        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();

        long diff = afterAlloc - beforeAlloc;
        Assert.Equal(0, diff);
    }

    [Fact]
    public void ApplyRepetitionPenalty_LargeHistory_ArrayPoolBitSet_ZeroAllocations()
    {
        float[] logits = new float[152_064];
        // 256 recent tokens (> 128 threshold)
        int[] recentTokens = new int[256];
        for (int i = 0; i < recentTokens.Length; i++) recentTokens[i] = i * 10;

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Warm up on current thread after GC (primes ArrayPool rent/return bucket)
        for (int w = 0; w < 10; w++)
        {
            Sampler.ApplyRepetitionPenalty(logits.AsSpan(), recentTokens.AsSpan(), 1.35f);
        }

        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();
        for (int iter = 0; iter < 100; iter++)
        {
            Sampler.ApplyRepetitionPenalty(logits.AsSpan(), recentTokens.AsSpan(), 1.35f);
        }
        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();

        long diff = afterAlloc - beforeAlloc;
        Assert.Equal(0, diff);
    }

    // =========================================================================
    // 6. Multi-Step Loop Recovery & Fuzzing
    // =========================================================================

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(16)]
    [InlineData(64)]
    public void LoopRecovery_SimulatedGreedyDecoding_EscapesLoopCleanly(int period)
    {
        var sampler = new Sampler();
        int vocabSize = 1000;
        float[] logits = new float[vocabSize];

        // Tokens in loop have high logits: [100.0, 99.0, ...]
        for (int k = 0; k < period; k++)
        {
            logits[k] = 100.0f - (k * 0.1f);
        }

        // An outside token has logit 75.0f
        int escapeToken = 500;
        logits[escapeToken] = 75.0f;

        // Construct history with 3 repeats of the period
        int[] history = new int[period * 3];
        for (int r = 0; r < 3; r++)
        {
            for (int k = 0; k < period; k++)
            {
                history[r * period + k] = k;
            }
        }

        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 2.0f // 100.0 / 2.0 = 50.0f < 75.0f
        };

        int sampled = sampler.Sample(logits.AsSpan(), options, history.AsSpan());

        // Must escape to outside token 500 because all loop tokens are penalized to <= 50.0f
        Assert.Equal(escapeToken, sampled);
    }

    [Fact]
    public void Fuzz_RandomizedLogitsAndTokenStreams_NeverThrowsOrCorruptsMemory()
    {
        var rng = new Random(98765);
        var sampler = new Sampler(seed: 42);

        for (int iteration = 0; iteration < 1000; iteration++)
        {
            int vocabSize = rng.Next(1, 200);
            float[] logits = new float[vocabSize];

            // Fill with mixture of random, NaN, Inf, extreme floats
            for (int i = 0; i < vocabSize; i++)
            {
                int kind = rng.Next(6);
                logits[i] = kind switch
                {
                    0 => float.NaN,
                    1 => float.NegativeInfinity,
                    2 => float.PositiveInfinity,
                    3 => (float)(rng.NextDouble() * 1000.0 - 500.0),
                    4 => 0.0f,
                    _ => (float)(rng.NextDouble() * 1e-30)
                };
            }

            int historyLen = rng.Next(0, 300);
            int[] history = new int[historyLen];
            for (int i = 0; i < historyLen; i++)
            {
                // Can be valid token ID or out-of-bounds token ID
                history[i] = rng.Next(-5, vocabSize + 10);
            }

            var options = new SamplingOptions
            {
                Temperature = (float)(rng.NextDouble() * 2.0),
                TopP = Math.Clamp((float)rng.NextDouble(), 0.01f, 1.0f),
                TopK = rng.Next(0, 100),
                RepetitionPenalty = Math.Clamp((float)(rng.NextDouble() * 3.0), 0.1f, 3.0f),
                EnableReasoningLoopGuard = rng.Next(2) == 1,
                LoopRecoveryRepetitionPenalty = (float)(rng.NextDouble() * 2.0 + 1.0)
            };

            // Random clamping test
            if (rng.Next(2) == 1)
            {
                int endThinkTokenId = rng.Next(-10, vocabSize + 10);
                Sampler.ClampReasoningLogits(logits.AsSpan(), endThinkTokenId);
            }

            // Must execute without exception
            int token = sampler.Sample(logits.AsSpan(), options, history.AsSpan());
            Assert.InRange(token, 0, vocabSize - 1);
        }
    }
}
