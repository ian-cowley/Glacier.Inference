namespace Glacier.Inference.Sampling;

using System;
using System.Buffers;
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
    public bool EnableReasoningLoopGuard { get; set; } = true;
    public float LoopRecoveryRepetitionPenalty { get; set; } = 1.35f;
    public int MaxReasoningTokens { get; set; } = 0;
    public int ThinkTokenId { get; set; } = -1;
    public int EndThinkTokenId { get; set; } = -1;

    public static SamplingOptions Greedy => new()
    {
        Temperature = 0.0f,
        TopP = 1.0f,
        TopK = 1,
        RepetitionPenalty = 1.0f
    };
}

/// <summary>
/// High-performance token sampler supporting Greedy, Temperature, Top-K, Top-P, Repetition Penalty,
/// NaN/Inf Logit Sanitization, and Periodic Reasoning Loop Detection &amp; Recovery.
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
    /// Sanitizes raw logits in-place against NaN and +/-Infinity (which can arise in extreme sub-3-bit
    /// MoE quantizations or uninitialized residual lanes). Returns the number of non-finite entries repaired.
    /// </summary>
    public static int SanitizeLogits(Span<float> logits)
    {
        int repaired = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (!float.IsFinite(v))
            {
                logits[i] = float.IsPositiveInfinity(v) ? 1e4f : -1e9f;
                repaired++;
            }
        }
        return repaired;
    }

    /// <summary>
    /// Clamps logits to enforce a reasoning token budget: penalizes continuation tokens
    /// by 50.0 and biases the closing thinking tag by +20.0 to force transition to final answer.
    /// </summary>
    public static void ClampReasoningLogits(Span<float> logits, int endThinkTokenId, float penalty = 50.0f, float bias = 20.0f)
    {
        if ((uint)endThinkTokenId >= (uint)logits.Length) return;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i != endThinkTokenId)
                logits[i] -= penalty;
        }
        logits[endThinkTokenId] += bias;
    }

    /// <summary>
    /// Detects periodic reasoning / code-generation loops in the tail of <paramref name="recentTokens"/>
    /// (e.g., a sequence of period <c>P in [minPeriod, maxPeriod]</c> repeating at least <paramref name="minRepeats"/> times).
    /// Returns the detected period length, or <c>0</c> if no periodic loop is present.
    /// </summary>
    public static int DetectPeriodicReasoningLoop(
        ReadOnlySpan<int> recentTokens,
        int minPeriod = 1,
        int maxPeriod = 64,
        int minRepeats = 3)
    {
        if (minPeriod < 1 || maxPeriod < minPeriod || minRepeats < 2)
            return 0;

        int n = recentTokens.Length;
        int maxCheckPeriod = Math.Min(maxPeriod, n / minRepeats);
        for (int p = minPeriod; p <= maxCheckPeriod; p++)
        {
            int requiredLen = p * minRepeats;
            int start = n - requiredLen;
            bool matches = true;
            for (int r = 1; r < minRepeats && matches; r++)
            {
                int baseOffset = n - p;
                int repOffset = start + (r - 1) * p;
                for (int k = 0; k < p; k++)
                {
                    if (recentTokens[repOffset + k] != recentTokens[baseOffset + k])
                    {
                        matches = false;
                        break;
                    }
                }
            }

            if (matches)
                return p;
        }

        return 0;
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
        if (options.Temperature < 0f || float.IsNaN(options.Temperature))
            throw new ArgumentOutOfRangeException(nameof(options), options.Temperature, "Temperature must be non-negative.");
        if (options.TopP < 0f || options.TopP > 1f || float.IsNaN(options.TopP))
            throw new ArgumentOutOfRangeException(nameof(options), options.TopP, "TopP must be between 0.0 and 1.0.");
        if (options.RepetitionPenalty <= 0f || float.IsNaN(options.RepetitionPenalty))
            throw new ArgumentOutOfRangeException(nameof(options), options.RepetitionPenalty, "RepetitionPenalty must be greater than zero.");

        int vocabSize = logits.Length;

        // Guard against NaN/Inf logits before top-K / greedy selection
        SanitizeLogits(logits);

        float effectivePenalty = options.RepetitionPenalty;
        ReadOnlySpan<int> penaltyWindow = recentTokens;

        if (options.EnableReasoningLoopGuard && !recentTokens.IsEmpty)
        {
            int loopPeriod = DetectPeriodicReasoningLoop(recentTokens);
            if (loopPeriod > 0)
            {
                effectivePenalty = Math.Max(effectivePenalty, options.LoopRecoveryRepetitionPenalty);
                int windowLen = Math.Min(recentTokens.Length, loopPeriod * 3);
                penaltyWindow = recentTokens.Slice(recentTokens.Length - windowLen, windowLen);
            }
        }

        // Apply repetition penalty (each token in penaltyWindow is penalized at most once)
        if (effectivePenalty != 1.0f && !penaltyWindow.IsEmpty)
        {
            ApplyRepetitionPenalty(logits, penaltyWindow, effectivePenalty);
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
        (int Id, float Logit)[]? rented = null;
        Span<(int Id, float Logit)> candidates = k <= 128
            ? stackalloc (int, float)[k]
            : (rented = ArrayPool<(int Id, float Logit)>.Shared.Rent(k)).AsSpan(0, k);

        try
        {
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
        finally
        {
            if (rented != null)
            {
                ArrayPool<(int Id, float Logit)>.Shared.Return(rented);
            }
        }
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

    /// <summary>
    /// Applies repetition penalty to logits in place with zero heap allocations.
    /// Each unique token ID in <paramref name="recentTokens"/> is penalized at most once.
    /// Uses stackalloc or pooled bitsets to eliminate dynamic HashSet allocations on hot decoding loops.
    /// </summary>
    public static void ApplyRepetitionPenalty(Span<float> logits, ReadOnlySpan<int> recentTokens, float penalty)
    {
        if (penalty == 1.0f || recentTokens.IsEmpty || logits.IsEmpty)
            return;

        int vocabSize = logits.Length;
        if (recentTokens.Length <= 128)
        {
            for (int i = 0; i < recentTokens.Length; i++)
            {
                int tid = recentTokens[i];
                if ((uint)tid >= (uint)vocabSize) continue;

                bool seen = false;
                for (int j = 0; j < i; j++)
                {
                    if (recentTokens[j] == tid)
                    {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;

                if (logits[tid] > 0)
                    logits[tid] /= penalty;
                else
                    logits[tid] *= penalty;
            }
        }
        else
        {
            int bitSetWords = (vocabSize + 63) / 64;
            ulong[]? rentedBitSet = null;
            Span<ulong> bitSet = bitSetWords <= 512
                ? stackalloc ulong[bitSetWords]
                : (rentedBitSet = ArrayPool<ulong>.Shared.Rent(bitSetWords)).AsSpan(0, bitSetWords);

            bitSet.Clear();

            try
            {
                for (int i = 0; i < recentTokens.Length; i++)
                {
                    int tid = recentTokens[i];
                    if ((uint)tid >= (uint)vocabSize) continue;

                    int wordIdx = tid >> 6;
                    ulong bitMask = 1UL << (tid & 63);

                    if ((bitSet[wordIdx] & bitMask) == 0)
                    {
                        bitSet[wordIdx] |= bitMask;

                        if (logits[tid] > 0)
                            logits[tid] /= penalty;
                        else
                            logits[tid] *= penalty;
                    }
                }
            }
            finally
            {
                if (rentedBitSet != null)
                {
                    ArrayPool<ulong>.Shared.Return(rentedBitSet);
                }
            }
        }
    }
}
