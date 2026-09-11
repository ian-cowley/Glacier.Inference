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
}
