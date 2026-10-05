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
}
