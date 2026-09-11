namespace Glacier.Inference.Sampling;

using System;
using System.Collections.Generic;

/// <summary>
/// Configurable sampling parameters for autoregressive generation.
/// </summary>
public sealed class SamplingOptions
{
    public float Temperature { get; set; } = 0.7f;
    public float TopP { get; set; } = 0.9f;
    public int TopK { get; set; } = 40;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int MaxTokens { get; set; } = 512;

    public static SamplingOptions Greedy => new()
    {
        Temperature = 0.0f,
        TopP = 1.0f,
        TopK = 1,
        RepetitionPenalty = 1.0f
    };
}

/// <summary>
/// High-performance token sampler supporting Greedy, Temperature, Top-K, Top-P, and Repetition Penalty.
/// </summary>
public sealed class Sampler
{
    private readonly Random _random;

    public Sampler(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Samples a next token from raw logits.
    /// </summary>
    public int Sample(Span<float> logits, SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        int vocabSize = logits.Length;

        // Apply repetition penalty
        if (options.RepetitionPenalty != 1.0f && !recentTokens.IsEmpty)
        {
            float penalty = options.RepetitionPenalty;
            for (int i = 0; i < recentTokens.Length; i++)
            {
                int tid = recentTokens[i];
                if (tid >= 0 && tid < vocabSize)
                {
                    if (logits[tid] > 0)
                        logits[tid] /= penalty;
                    else
                        logits[tid] *= penalty;
                }
            }
        }

        // Greedy sampling if temperature <= 0 or TopK == 1
        if (options.Temperature <= 0.001f || options.TopK == 1)
        {
            int bestId = 0;
            float bestLogit = logits[0];
            for (int i = 1; i < vocabSize; i++)
            {
                if (logits[i] > bestLogit)
                {
                    bestLogit = logits[i];
                    bestId = i;
                }
            }
            return bestId;
        }

        // Apply temperature
        float invTemp = 1.0f / options.Temperature;
        for (int i = 0; i < vocabSize; i++)
        {
            logits[i] *= invTemp;
        }

        // Find top-K candidates
        int k = Math.Min(options.TopK, vocabSize);
        var candidates = new (int Id, float Logit)[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            candidates[i] = (i, logits[i]);
        }

        // Partial sort top-k
        Array.Sort(candidates, (a, b) => b.Logit.CompareTo(a.Logit));

        // Softmax over top-K candidates
        float maxLogit = candidates[0].Logit;
        float sumExp = 0f;
        for (int i = 0; i < k; i++)
        {
            float exp = MathF.Exp(candidates[i].Logit - maxLogit);
            candidates[i].Logit = exp;
            sumExp += exp;
        }

        // Top-P filtering
        float invSum = 1.0f / sumExp;
        float cumProb = 0f;
        int activeCount = 0;

        for (int i = 0; i < k; i++)
        {
            float prob = candidates[i].Logit * invSum;
            candidates[i].Logit = prob;
            cumProb += prob;
            activeCount++;
            if (cumProb >= options.TopP) break;
        }

        // Random categorical sampling
        float r = (float)_random.NextDouble() * cumProb;
        float acc = 0f;
        for (int i = 0; i < activeCount; i++)
        {
            acc += candidates[i].Logit;
            if (r <= acc)
            {
                return candidates[i].Id;
            }
        }

        return candidates[0].Id;
    }
}
