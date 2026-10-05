namespace Glacier.Inference.Sampling;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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
/// Instances are NOT thread-safe (they own a <see cref="Random"/>); use one sampler per generation stream.
/// </summary>
public sealed class Sampler
{
    /// <summary>Temperatures at or below this value are treated as greedy decoding.</summary>
    public const float GreedyTemperatureThreshold = 0.001f;

    private readonly Random _random;

    public Sampler(int? seed = null)
    {
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Samples a next token from raw logits.
    /// NOTE: when a repetition penalty applies, <paramref name="logits"/> is modified in place.
    /// A <c>TopK</c> of 0 or less disables top-K filtering.
    /// </summary>
    public int Sample(Span<float> logits, SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (logits.IsEmpty) throw new ArgumentException("Logits must not be empty.", nameof(logits));
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

        // Greedy sampling if temperature is (near) zero or TopK == 1
        if (options.Temperature <= GreedyTemperatureThreshold || options.TopK == 1)
        {
            int bestId = 0;
            float bestLogit = float.NegativeInfinity;
            for (int i = 0; i < vocabSize; i++)
            {
                // NaN compares false, so NaN logits are never selected.
                if (logits[i] > bestLogit)
                {
                    bestLogit = logits[i];
                    bestId = i;
                }
            }
            return bestId;
        }

        // TopK <= 0 means "disabled": consider the full vocabulary.
        int k = options.TopK <= 0 ? vocabSize : Math.Min(options.TopK, vocabSize);
        Span<(int Id, float Logit)> candidates = k <= 128
            ? stackalloc (int, float)[k]
            : new (int, float)[k];

        SelectTopK(logits, k, candidates);

        // Apply temperature only to top-K candidates
        float invTemp = 1.0f / options.Temperature;
        for (int i = 0; i < k; i++)
        {
            candidates[i].Logit *= invTemp;
        }

        // Softmax over top-K candidates (candidates[0] is maxLogit)
        float maxLogit = candidates[0].Logit;
        float sumExp = 0f;
        for (int i = 0; i < k; i++)
        {
            float exp = MathF.Exp(candidates[i].Logit - maxLogit);
            candidates[i].Logit = exp;
            sumExp += exp;
        }

        // Degenerate distribution (NaN/Inf logits): fall back to the best candidate.
        if (!(sumExp > 0f) || !float.IsFinite(sumExp))
            return candidates[0].Id;

        // Top-P filtering (always keeps at least one token)
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

    private static void SelectTopK(ReadOnlySpan<float> logits, int k, Span<(int Id, float Logit)> heap)
    {
        for (int i = 0; i < k; i++)
        {
            heap[i] = (i, logits[i]);
        }

        // Build min-heap in place: heap[0] has lowest logit
        for (int i = (k / 2) - 1; i >= 0; i--)
        {
            SiftDown(heap, i, k);
        }

        float minLogit = heap[0].Logit;
        int vocabSize = logits.Length;

        for (int i = k; i < vocabSize; i++)
        {
            float val = logits[i];
            if (val > minLogit)
            {
                heap[0] = (i, val);
                SiftDown(heap, 0, k);
                minLogit = heap[0].Logit;
            }
        }

        // Sort descending: highest logit first
        heap.Sort(default(LogitDescendingComparer));
    }

    private readonly struct LogitDescendingComparer : IComparer<(int Id, float Logit)>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare((int Id, float Logit) x, (int Id, float Logit) y) => y.Logit.CompareTo(x.Logit);
    }

    private static void SiftDown(Span<(int Id, float Logit)> heap, int i, int n)
    {
        while (true)
        {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int smallest = i;

            if (left < n && heap[left].Logit < heap[smallest].Logit)
                smallest = left;
            if (right < n && heap[right].Logit < heap[smallest].Logit)
                smallest = right;

            if (smallest != i)
            {
                var temp = heap[i];
                heap[i] = heap[smallest];
                heap[smallest] = temp;
                i = smallest;
            }
            else
            {
                break;
            }
        }
    }
}
