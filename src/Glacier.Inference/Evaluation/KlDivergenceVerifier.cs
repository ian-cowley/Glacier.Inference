namespace Glacier.Inference.Evaluation;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;

/// <summary>
/// Numerically stable, 3-pass double-precision KL-divergence verifier <c>D_KL(P_ref || Q_quant)</c>
/// for comparing raw logit spans, <see cref="ISpeculativeTarget"/> instances, <see cref="InferenceSession"/>
/// models, and <see cref="TieredMoERunner"/> configurations against reference (<c>Q8_0</c> / <c>FP32</c>) outputs.
/// </summary>
public static class KlDivergenceVerifier
{
    public const double DefaultThreshold = 0.05;
    public const double DefaultEpsilon = 1e-12;

    /// <summary>
    /// Computes log-softmax(logits) in double precision using a 3-pass max-shifted log-sum-exp formulation.
    /// </summary>
    public static void ComputeLogSoftmax(ReadOnlySpan<float> logits, Span<double> logProbs)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits span cannot be empty.", nameof(logits));
        if (logProbs.Length < logits.Length)
            throw new ArgumentException($"Destination span length ({logProbs.Length}) is smaller than logits length ({logits.Length}).", nameof(logProbs));

        double maxLogit = FindMaxAndArgMaxLogits(logits, nameof(logits), out int argMax);
        double sumOther = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i == argMax) continue;
            float v = logits[i];
            if (!float.IsNegativeInfinity(v))
                sumOther += Math.Exp((double)v - maxLogit);
        }

        double logSumExpShift = Log1p(sumOther);
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (float.IsNegativeInfinity(v))
            {
                logProbs[i] = double.NegativeInfinity;
            }
            else if (i == argMax)
            {
                logProbs[i] = -logSumExpShift;
            }
            else
            {
                double lp = ((double)v - maxLogit) - logSumExpShift;
                logProbs[i] = lp > 0.0 ? 0.0 : lp;
            }
        }
    }

    /// <summary>
    /// Computes softmax(logits) in double precision and writes float probabilities to <paramref name="probs"/>.
    /// </summary>
    public static void ComputeSoftmax(ReadOnlySpan<float> logits, Span<float> probs)
    {
        if (logits.IsEmpty)
            throw new ArgumentException("Logits span cannot be empty.", nameof(logits));
        if (probs.Length < logits.Length)
            throw new ArgumentException($"Destination span length ({probs.Length}) is smaller than logits length ({logits.Length}).", nameof(probs));

        double maxLogit = FindMaxAndValidateLogits(logits, nameof(logits));
        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (!float.IsNegativeInfinity(v))
                sumExp += Math.Exp((double)v - maxLogit);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            probs[i] = float.IsNegativeInfinity(v)
                ? 0.0f
                : (float)(Math.Exp((double)v - maxLogit) / sumExp);
        }
    }

    /// <summary>
    /// Returns the index of the maximum element using lowest-index tie-breaking.
    /// </summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.IsEmpty)
            throw new ArgumentException("Values span cannot be empty.", nameof(values));

        float first = values[0];
        if (float.IsNaN(first))
            throw new ArgumentException("Input values cannot contain NaN.", nameof(values));

        int bestIdx = 0;
        float bestVal = first;
        for (int i = 1; i < values.Length; i++)
        {
            float v = values[i];
            if (float.IsNaN(v))
                throw new ArgumentException("Input values cannot contain NaN.", nameof(values));
            if (v > bestVal)
            {
                bestVal = v;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Computes <c>D_KL(P_ref || Q_quant)</c> in nats for a single pair of logit vectors without heap allocations.
    /// </summary>
    public static double ComputeTokenKlDivergence(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double epsilon = DefaultEpsilon)
    {
        if (referenceLogits.IsEmpty)
            throw new ArgumentException("Reference logits span cannot be empty.", nameof(referenceLogits));
        if (referenceLogits.Length != quantizedLogits.Length)
            throw new ArgumentException($"Logit span lengths must match (reference: {referenceLogits.Length}, quantized: {quantizedLogits.Length}).", nameof(quantizedLogits));
        if (!double.IsFinite(epsilon) || epsilon <= 0.0 || epsilon >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon, "Epsilon must be finite and in (0.0, 1.0).");

        return ComputeTokenKlCore(referenceLogits, quantizedLogits, epsilon, out _);
    }

    public static KlDivergenceResult EvaluateSingleTokenLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        if (referenceLogits.IsEmpty)
            throw new ArgumentException("Reference logits span cannot be empty.", nameof(referenceLogits));

        return EvaluateLogits(
            referenceLogits,
            quantizedLogits,
            referenceLogits.Length,
            threshold,
            maxKlThreshold,
            minTop1Agreement,
            epsilon);
    }

    public static KlDivergenceResult EvaluateSingleTokenLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        return EvaluateSingleTokenLogits(
            referenceLogits,
            quantizedLogits,
            opts.Threshold,
            opts.MaxKlThreshold,
            opts.MinTop1Agreement,
            opts.Epsilon);
    }

    public static KlDivergenceResult EvaluateLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        int vocabSize,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement,
            Epsilon = epsilon
        };
        return EvaluateLogits(referenceLogits, quantizedLogits, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateLogits(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        int vocabSize,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateBatchShapes(referenceLogits.Length, quantizedLogits.Length, vocabSize);
        int tokenCount = referenceLogits.Length / vocabSize;

        double sumKl = 0.0;
        double maxKl = 0.0;
        double minKl = double.PositiveInfinity;
        int top1Matches = 0;

        for (int t = 0; t < tokenCount; t++)
        {
            int offset = t * vocabSize;
            var refRow = referenceLogits.Slice(offset, vocabSize);
            var quantRow = quantizedLogits.Slice(offset, vocabSize);

            _ = FindMaxAndValidateLogits(refRow, nameof(referenceLogits));
            if (!TryValidateCandidateLogits(quantRow, t, out string? nonFiniteReason))
            {
                return CreateNonFiniteResult(tokenCount, vocabSize, opts, nonFiniteReason!, null, null, default);
            }

            double kl = ComputeTokenKlCore(refRow, quantRow, opts.Epsilon, out bool top1Match);
            sumKl += kl;
            if (kl > maxKl) maxKl = kl;
            if (kl < minKl) minKl = kl;
            if (top1Match) top1Matches++;
        }

        return BuildResult(sumKl, maxKl, minKl, top1Matches, tokenCount, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateProbabilities(
        ReadOnlySpan<float> referenceProbs,
        ReadOnlySpan<float> quantizedProbs,
        int vocabSize,
        double threshold = DefaultThreshold,
        double? maxKlThreshold = null,
        double minTop1Agreement = 0.0,
        double epsilon = DefaultEpsilon)
    {
        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1Agreement,
            Epsilon = epsilon
        };
        return EvaluateProbabilities(referenceProbs, quantizedProbs, vocabSize, opts);
    }

    public static KlDivergenceResult EvaluateProbabilities(
        ReadOnlySpan<float> referenceProbs,
        ReadOnlySpan<float> quantizedProbs,
        int vocabSize,
        KlDivergenceOptions? options)
    {
        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        ValidateBatchShapes(referenceProbs.Length, quantizedProbs.Length, vocabSize);
        int tokenCount = referenceProbs.Length / vocabSize;

        double sumKl = 0.0;
        double maxKl = 0.0;
        double minKl = double.PositiveInfinity;
        int top1Matches = 0;

        for (int t = 0; t < tokenCount; t++)
        {
            int offset = t * vocabSize;
            var refRow = referenceProbs.Slice(offset, vocabSize);
            var quantRow = quantizedProbs.Slice(offset, vocabSize);

            double sumP = 0.0;
            double sumQ = 0.0;
            int argMaxP = 0;
            int argMaxQ = 0;
            float maxP = float.NegativeInfinity;
            float maxQ = float.NegativeInfinity;

            for (int i = 0; i < vocabSize; i++)
            {
                float rp = refRow[i];
                float qp = quantRow[i];

                if (!float.IsFinite(rp) || rp < 0.0f)
                    throw new ArgumentException($"Reference probability at token {t}, index {i} must be finite and non-negative.", nameof(referenceProbs));
                if (!float.IsFinite(qp))
                    return CreateNonFiniteResult(tokenCount, vocabSize, opts, $"Quantized probability at token {t}, index {i} is non-finite ({qp}).", null, null, default);
                if (qp < 0.0f)
                    throw new ArgumentException($"Quantized probability at token {t}, index {i} must be finite and non-negative.", nameof(quantizedProbs));

                sumP += rp;
                sumQ += qp;

                if (i == 0 || rp > maxP)
                {
                    maxP = rp;
                    argMaxP = i;
                }
                if (i == 0 || qp > maxQ)
                {
                    maxQ = qp;
                    argMaxQ = i;
                }
            }

            if (sumP <= 0.0)
                throw new ArgumentException($"Reference probability distribution at token {t} sums to zero.", nameof(referenceProbs));
            if (sumQ <= 0.0)
                throw new ArgumentException($"Quantized probability distribution at token {t} sums to zero.", nameof(quantizedProbs));

            if (argMaxP == argMaxQ)
                top1Matches++;

            double kl = 0.0;
            for (int i = 0; i < vocabSize; i++)
            {
                double p = (double)refRow[i] / sumP;
                if (p <= 0.0)
                    continue;

                double q = Math.Max((double)quantRow[i] / sumQ, opts.Epsilon);
                kl += p * (Math.Log(p) - Math.Log(q));
            }

            if (kl < 0.0)
                kl = 0.0;

            sumKl += kl;
            if (kl > maxKl) maxKl = kl;
            if (kl < minKl) minKl = kl;
        }

        return BuildResult(sumKl, maxKl, minKl, top1Matches, tokenCount, vocabSize, opts);
    }

    /// <summary>
    /// Evaluates <c>D_KL(P_ref || Q_cand)</c> across every token position of <paramref name="tokens"/>
    /// by resetting KV/recurrent state on both targets and comparing their predicted next-token logit distributions.
    /// </summary>
    public static KlDivergenceResult VerifyTargets(
        ISpeculativeTarget referenceTarget,
        ISpeculativeTarget candidateTarget,
        ReadOnlySpan<int> tokens,
        KlDivergenceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(referenceTarget);
        ArgumentNullException.ThrowIfNull(candidateTarget);
        if (tokens.IsEmpty)
            throw new ArgumentException("Token sequence cannot be empty.", nameof(tokens));

        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        int vocabSize = ValidateTargetCompatibility(referenceTarget, candidateTarget, tokens.Length);
        string? refDevice = GetTargetDeviceDescription(referenceTarget);
        string? candDevice = GetTargetDeviceDescription(candidateTarget);

        long startTs = Stopwatch.GetTimestamp();
        AccumulatedKlStats stats = default;
        stats.MinKl = double.PositiveInfinity;

        bool ok = AccumulateSequenceLogits(
            referenceTarget,
            candidateTarget,
            tokens,
            opts,
            vocabSize,
            ref stats,
            out string? nonFiniteReason);

        TimeSpan duration = Stopwatch.GetElapsedTime(startTs);
        if (!ok)
        {
            return CreateNonFiniteResult(tokens.Length, vocabSize, opts, nonFiniteReason!, refDevice, candDevice, duration);
        }

        return BuildResult(
            stats.SumKl,
            stats.MaxKl,
            stats.MinKl,
            stats.Top1Matches,
            stats.TokenCount,
            vocabSize,
            opts,
            refDevice,
            candDevice,
            duration);
    }

    /// <summary>
    /// Evaluates <c>D_KL(P_ref || Q_cand)</c> at the prompt's next-token prediction and across any
    /// <paramref name="continuationTokens"/> positions.
    /// </summary>
    public static KlDivergenceResult VerifyTargets(
        ISpeculativeTarget referenceTarget,
        ISpeculativeTarget candidateTarget,
        ReadOnlySpan<int> promptTokens,
        ReadOnlySpan<int> continuationTokens,
        KlDivergenceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(referenceTarget);
        ArgumentNullException.ThrowIfNull(candidateTarget);
        if (promptTokens.IsEmpty)
            throw new ArgumentException("Prompt token sequence cannot be empty.", nameof(promptTokens));

        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        int totalPositions = promptTokens.Length + continuationTokens.Length;
        int vocabSize = ValidateTargetCompatibility(referenceTarget, candidateTarget, totalPositions);
        string? refDevice = GetTargetDeviceDescription(referenceTarget);
        string? candDevice = GetTargetDeviceDescription(candidateTarget);

        long startTs = Stopwatch.GetTimestamp();
        AccumulatedKlStats stats = default;
        stats.MinKl = double.PositiveInfinity;

        int evalTokens = 1 + continuationTokens.Length;
        int totalFloats = evalTokens * vocabSize;
        float[] refBuf = ArrayPool<float>.Shared.Rent(totalFloats);
        float[] candBuf = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            referenceTarget.ResetKvCache();
            candidateTarget.ResetKvCache();

            referenceTarget.Prefill(promptTokens);
            candidateTarget.Prefill(promptTokens);

            referenceTarget.CopyCurrentLogits(refBuf.AsSpan(0, vocabSize));
            candidateTarget.CopyCurrentLogits(candBuf.AsSpan(0, vocabSize));

            if (!continuationTokens.IsEmpty)
            {
                Span<float> refCont = refBuf.AsSpan(vocabSize, continuationTokens.Length * vocabSize);
                Span<float> candCont = candBuf.AsSpan(vocabSize, continuationTokens.Length * vocabSize);

                if (opts.UseBatchLogitVerification)
                {
                    referenceTarget.VerifyBatchLogits(continuationTokens, promptTokens.Length, refCont);
                    candidateTarget.VerifyBatchLogits(continuationTokens, promptTokens.Length, candCont);
                }
                else
                {
                    for (int i = 0; i < continuationTokens.Length; i++)
                    {
                        int pos = promptTokens.Length + i;
                        referenceTarget.ForwardToken(continuationTokens[i], pos, computeLogits: true);
                        referenceTarget.CopyCurrentLogits(refCont.Slice(i * vocabSize, vocabSize));

                        candidateTarget.ForwardToken(continuationTokens[i], pos, computeLogits: true);
                        candidateTarget.CopyCurrentLogits(candCont.Slice(i * vocabSize, vocabSize));
                    }
                }
            }

            if (!AccumulateBufferRows(
                    refBuf.AsSpan(0, totalFloats),
                    candBuf.AsSpan(0, totalFloats),
                    evalTokens,
                    vocabSize,
                    opts.Epsilon,
                    ref stats,
                    out string? nonFiniteReason))
            {
                return CreateNonFiniteResult(evalTokens, vocabSize, opts, nonFiniteReason!, refDevice, candDevice, Stopwatch.GetElapsedTime(startTs));
            }

            return BuildResult(
                stats.SumKl,
                stats.MaxKl,
                stats.MinKl,
                stats.Top1Matches,
                stats.TokenCount,
                vocabSize,
                opts,
                refDevice,
                candDevice,
                Stopwatch.GetElapsedTime(startTs));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refBuf);
            ArrayPool<float>.Shared.Return(candBuf);
        }
    }

    /// <summary>
    /// Tokenizes <paramref name="prompts"/> via <paramref name="referenceTarget"/>'s tokenizer and evaluates
    /// <c>D_KL(P_ref || Q_cand)</c> across all prompts into an aggregated <see cref="KlDivergenceResult"/>.
    /// </summary>
    public static KlDivergenceResult VerifyPrompts(
        ISpeculativeTarget referenceTarget,
        ISpeculativeTarget candidateTarget,
        IReadOnlyList<string> prompts,
        bool formatChat = true,
        bool evaluateAllPromptTokens = true,
        KlDivergenceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(referenceTarget);
        ArgumentNullException.ThrowIfNull(candidateTarget);
        ArgumentNullException.ThrowIfNull(prompts);
        if (prompts.Count == 0)
            throw new ArgumentException("At least one prompt is required.", nameof(prompts));

        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        if (referenceTarget.VocabSize != candidateTarget.VocabSize)
            throw new ArgumentException($"Target vocabulary sizes must match (reference: {referenceTarget.VocabSize}, candidate: {candidateTarget.VocabSize}).");

        int vocabSize = referenceTarget.VocabSize;
        string? refDevice = GetTargetDeviceDescription(referenceTarget);
        string? candDevice = GetTargetDeviceDescription(candidateTarget);

        long startTs = Stopwatch.GetTimestamp();
        AccumulatedKlStats stats = default;
        stats.MinKl = double.PositiveInfinity;

        for (int p = 0; p < prompts.Count; p++)
        {
            string promptText = prompts[p] ?? throw new ArgumentException($"Prompt at index {p} cannot be null.", nameof(prompts));
            string formatted = formatChat ? referenceTarget.Tokenizer.FormatChatML(promptText) : promptText;
            int[] tokens = referenceTarget.Tokenizer.Encode(formatted);
            if (tokens.Length == 0)
                throw new ArgumentException($"Prompt at index {p} tokenized to zero tokens.", nameof(prompts));

            ValidateTargetCompatibility(referenceTarget, candidateTarget, tokens.Length);

            if (evaluateAllPromptTokens)
            {
                if (!AccumulateSequenceLogits(
                        referenceTarget,
                        candidateTarget,
                        tokens,
                        opts,
                        vocabSize,
                        ref stats,
                        out string? nonFiniteReason))
                {
                    return CreateNonFiniteResult(
                        stats.TokenCount + tokens.Length,
                        vocabSize,
                        opts,
                        nonFiniteReason!,
                        refDevice,
                        candDevice,
                        Stopwatch.GetElapsedTime(startTs));
                }
            }
            else
            {
                float[] refBuf = ArrayPool<float>.Shared.Rent(vocabSize);
                float[] candBuf = ArrayPool<float>.Shared.Rent(vocabSize);
                try
                {
                    referenceTarget.ResetKvCache();
                    candidateTarget.ResetKvCache();

                    referenceTarget.Prefill(tokens);
                    candidateTarget.Prefill(tokens);

                    referenceTarget.CopyCurrentLogits(refBuf.AsSpan(0, vocabSize));
                    candidateTarget.CopyCurrentLogits(candBuf.AsSpan(0, vocabSize));

                    if (!AccumulateBufferRows(
                            refBuf.AsSpan(0, vocabSize),
                            candBuf.AsSpan(0, vocabSize),
                            1,
                            vocabSize,
                            opts.Epsilon,
                            ref stats,
                            out string? nonFiniteReason))
                    {
                        return CreateNonFiniteResult(
                            stats.TokenCount + 1,
                            vocabSize,
                            opts,
                            nonFiniteReason!,
                            refDevice,
                            candDevice,
                            Stopwatch.GetElapsedTime(startTs));
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(refBuf);
                    ArrayPool<float>.Shared.Return(candBuf);
                }
            }
        }

        return BuildResult(
            stats.SumKl,
            stats.MaxKl,
            stats.MinKl,
            stats.Top1Matches,
            stats.TokenCount,
            vocabSize,
            opts,
            refDevice,
            candDevice,
            Stopwatch.GetElapsedTime(startTs));
    }

    /// <summary>
    /// Opens two GGUF models (<paramref name="referenceModelPath"/>, typically <c>Q8_0</c> or <c>F32</c>/<c>BF16</c>,
    /// and <paramref name="candidateModelPath"/>) and verifies KL divergence across <paramref name="prompts"/>.
    /// </summary>
    public static KlDivergenceResult VerifyModels(
        string referenceModelPath,
        string candidateModelPath,
        IReadOnlyList<string> prompts,
        int maxSeqLen = 2048,
        string? device = null,
        InferenceEngineType engine = InferenceEngineType.Auto,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto,
        bool formatChat = true,
        bool evaluateAllPromptTokens = true,
        KlDivergenceOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(referenceModelPath))
            throw new ArgumentException("Reference model path cannot be empty.", nameof(referenceModelPath));
        if (string.IsNullOrWhiteSpace(candidateModelPath))
            throw new ArgumentException("Candidate model path cannot be empty.", nameof(candidateModelPath));

        using var refSession = new InferenceSession(referenceModelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision);
        using var candSession = new InferenceSession(candidateModelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision);
        return VerifyPrompts(refSession, candSession, prompts, formatChat, evaluateAllPromptTokens, options);
    }

    internal static KlDivergenceResult VerifySyntheticTieredMoE(
        TieredMoERunner referenceRunner,
        TieredMoERunner candidateRunner,
        ReadOnlySpan<float> inputHiddenBatch,
        int batchSize,
        bool isSpeculativeDraftBatch = false,
        KlDivergenceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(referenceRunner);
        ArgumentNullException.ThrowIfNull(candidateRunner);
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");

        var opts = options ?? new KlDivergenceOptions();
        opts.Validate();

        if (referenceRunner.VocabSize != candidateRunner.VocabSize)
            throw new ArgumentException($"Runner vocabulary sizes must match (reference: {referenceRunner.VocabSize}, candidate: {candidateRunner.VocabSize}).");

        int vocabSize = referenceRunner.VocabSize;
        int totalFloats = batchSize * vocabSize;
        float[] refLogits = ArrayPool<float>.Shared.Rent(totalFloats);
        float[] candLogits = ArrayPool<float>.Shared.Rent(totalFloats);
        long startTs = Stopwatch.GetTimestamp();

        try
        {
            referenceRunner.EvaluateSyntheticBatch(inputHiddenBatch, batchSize, isSpeculativeDraftBatch, refLogits.AsSpan(0, totalFloats));
            candidateRunner.EvaluateSyntheticBatch(inputHiddenBatch, batchSize, isSpeculativeDraftBatch, candLogits.AsSpan(0, totalFloats));

            AccumulatedKlStats stats = default;
            stats.MinKl = double.PositiveInfinity;
            if (!AccumulateBufferRows(
                    refLogits.AsSpan(0, totalFloats),
                    candLogits.AsSpan(0, totalFloats),
                    batchSize,
                    vocabSize,
                    opts.Epsilon,
                    ref stats,
                    out string? nonFiniteReason))
            {
                return CreateNonFiniteResult(
                    batchSize,
                    vocabSize,
                    opts,
                    nonFiniteReason!,
                    referenceRunner.DeviceDescription,
                    candidateRunner.DeviceDescription,
                    Stopwatch.GetElapsedTime(startTs));
            }

            return BuildResult(
                stats.SumKl,
                stats.MaxKl,
                stats.MinKl,
                stats.Top1Matches,
                stats.TokenCount,
                vocabSize,
                opts,
                referenceRunner.DeviceDescription,
                candidateRunner.DeviceDescription,
                Stopwatch.GetElapsedTime(startTs));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refLogits);
            ArrayPool<float>.Shared.Return(candLogits);
        }
    }

    private struct AccumulatedKlStats
    {
        public double SumKl;
        public double MaxKl;
        public double MinKl;
        public int Top1Matches;
        public int TokenCount;
    }

    private static int ValidateTargetCompatibility(ISpeculativeTarget referenceTarget, ISpeculativeTarget candidateTarget, int sequenceLength)
    {
        if (referenceTarget.VocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(referenceTarget), "Reference VocabSize must be positive.");
        if (referenceTarget.VocabSize != candidateTarget.VocabSize)
            throw new ArgumentException($"Target vocabulary sizes must match (reference: {referenceTarget.VocabSize}, candidate: {candidateTarget.VocabSize}).");

        if (!referenceTarget.IsRingBufferEnabled && sequenceLength > referenceTarget.MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), $"Sequence length {sequenceLength} exceeds reference MaxSeqLen ({referenceTarget.MaxSeqLen}).");
        if (!candidateTarget.IsRingBufferEnabled && sequenceLength > candidateTarget.MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), $"Sequence length {sequenceLength} exceeds candidate MaxSeqLen ({candidateTarget.MaxSeqLen}).");

        return referenceTarget.VocabSize;
    }

    private static string? GetTargetDeviceDescription(ISpeculativeTarget target) => target switch
    {
        InferenceSession s => s.ActiveDevice,
        TieredMoERunner t => t.DeviceDescription,
        _ => null
    };

    private static bool AccumulateSequenceLogits(
        ISpeculativeTarget referenceTarget,
        ISpeculativeTarget candidateTarget,
        ReadOnlySpan<int> tokens,
        KlDivergenceOptions opts,
        int vocabSize,
        ref AccumulatedKlStats stats,
        out string? nonFiniteReason)
    {
        referenceTarget.ResetKvCache();
        candidateTarget.ResetKvCache();

        if (!opts.UseBatchLogitVerification)
        {
            float[] refRow = ArrayPool<float>.Shared.Rent(vocabSize);
            float[] candRow = ArrayPool<float>.Shared.Rent(vocabSize);
            try
            {
                for (int t = 0; t < tokens.Length; t++)
                {
                    referenceTarget.ForwardToken(tokens[t], t, computeLogits: true);
                    referenceTarget.CopyCurrentLogits(refRow.AsSpan(0, vocabSize));

                    candidateTarget.ForwardToken(tokens[t], t, computeLogits: true);
                    candidateTarget.CopyCurrentLogits(candRow.AsSpan(0, vocabSize));

                    if (!AccumulateBufferRows(
                            refRow.AsSpan(0, vocabSize),
                            candRow.AsSpan(0, vocabSize),
                            1,
                            vocabSize,
                            opts.Epsilon,
                            ref stats,
                            out nonFiniteReason))
                    {
                        return false;
                    }
                }
                nonFiniteReason = null;
                return true;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(refRow);
                ArrayPool<float>.Shared.Return(candRow);
            }
        }

        int totalFloats = tokens.Length * vocabSize;
        float[] refBuf = ArrayPool<float>.Shared.Rent(totalFloats);
        float[] candBuf = ArrayPool<float>.Shared.Rent(totalFloats);
        try
        {
            referenceTarget.VerifyBatchLogits(tokens, 0, refBuf.AsSpan(0, totalFloats));
            candidateTarget.VerifyBatchLogits(tokens, 0, candBuf.AsSpan(0, totalFloats));

            return AccumulateBufferRows(
                refBuf.AsSpan(0, totalFloats),
                candBuf.AsSpan(0, totalFloats),
                tokens.Length,
                vocabSize,
                opts.Epsilon,
                ref stats,
                out nonFiniteReason);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(refBuf);
            ArrayPool<float>.Shared.Return(candBuf);
        }
    }

    private static bool AccumulateBufferRows(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> candidateLogits,
        int rowCount,
        int vocabSize,
        double epsilon,
        ref AccumulatedKlStats stats,
        out string? nonFiniteReason)
    {
        for (int t = 0; t < rowCount; t++)
        {
            int offset = t * vocabSize;
            var refRow = referenceLogits.Slice(offset, vocabSize);
            var candRow = candidateLogits.Slice(offset, vocabSize);

            // Reference logits must always be valid finite distributions (at least one finite entry).
            _ = FindMaxAndValidateLogits(refRow, nameof(referenceLogits));

            if (!TryValidateCandidateLogits(candRow, t, out nonFiniteReason))
                return false;

            double kl = ComputeTokenKlCore(refRow, candRow, epsilon, out bool top1Match);
            stats.SumKl += kl;
            if (kl > stats.MaxKl) stats.MaxKl = kl;
            if (kl < stats.MinKl) stats.MinKl = kl;
            if (top1Match) stats.Top1Matches++;
            stats.TokenCount++;
        }

        nonFiniteReason = null;
        return true;
    }

    private static bool TryValidateCandidateLogits(ReadOnlySpan<float> candRow, int tokenIndex, out string? reason)
    {
        bool hasFinite = false;
        for (int i = 0; i < candRow.Length; i++)
        {
            float q = candRow[i];
            if (float.IsNaN(q) || float.IsPositiveInfinity(q))
            {
                reason = $"Quantized candidate produced non-finite logits ({q}) at token {tokenIndex}, index {i}.";
                return false;
            }
            if (!float.IsNegativeInfinity(q))
            {
                hasFinite = true;
            }
        }

        if (!hasFinite)
        {
            reason = $"Quantized candidate produced exclusively -Infinity logits at token {tokenIndex}.";
            return false;
        }

        reason = null;
        return true;
    }

    private static KlDivergenceResult CreateNonFiniteResult(
        int tokenCount,
        int vocabSize,
        KlDivergenceOptions opts,
        string reason,
        string? refDevice,
        string? candDevice,
        TimeSpan duration)
    {
        return new KlDivergenceResult
        {
            MeanKlDivergence = double.PositiveInfinity,
            MaxKlDivergence = double.PositiveInfinity,
            MinKlDivergence = double.PositiveInfinity,
            Top1AgreementRate = 0.0,
            Top1Matches = 0,
            TokenCount = tokenCount,
            VocabSize = vocabSize,
            Threshold = opts.Threshold,
            MaxKlThreshold = opts.MaxKlThreshold,
            MinTop1Agreement = opts.MinTop1Agreement,
            HasNonFiniteQuantizedValues = true,
            Passed = false,
            FailureReason = reason,
            ReferenceDevice = refDevice,
            CandidateDevice = candDevice,
            Duration = duration
        };
    }

    private static double ComputeTokenKlCore(
        ReadOnlySpan<float> referenceLogits,
        ReadOnlySpan<float> quantizedLogits,
        double epsilon,
        out bool top1Match)
    {
        int n = referenceLogits.Length;

        double maxRef = double.NegativeInfinity;
        double maxQuant = double.NegativeInfinity;
        int argMaxRef = 0;
        int argMaxQuant = 0;
        bool hasFiniteRef = false;
        bool hasFiniteQuant = false;

        for (int i = 0; i < n; i++)
        {
            float r = referenceLogits[i];
            float q = quantizedLogits[i];

            if (float.IsNaN(r) || float.IsPositiveInfinity(r))
                throw new ArgumentException($"Reference logits contain invalid value ({r}) at index {i}.", nameof(referenceLogits));
            if (float.IsNaN(q) || float.IsPositiveInfinity(q))
                throw new ArgumentException($"Quantized logits contain invalid value ({q}) at index {i}.", nameof(quantizedLogits));

            if (!float.IsNegativeInfinity(r))
                hasFiniteRef = true;
            if (!float.IsNegativeInfinity(q))
                hasFiniteQuant = true;

            if (i == 0 || r > maxRef)
            {
                maxRef = r;
                argMaxRef = i;
            }
            if (i == 0 || q > maxQuant)
            {
                maxQuant = q;
                argMaxQuant = i;
            }
        }

        if (!hasFiniteRef)
            throw new ArgumentException("Reference logits cannot consist exclusively of -Infinity.", nameof(referenceLogits));
        if (!hasFiniteQuant)
            throw new ArgumentException("Quantized logits cannot consist exclusively of -Infinity.", nameof(quantizedLogits));

        top1Match = argMaxRef == argMaxQuant;

        double sumOtherRef = 0.0;
        double sumOtherQuant = 0.0;
        for (int i = 0; i < n; i++)
        {
            if (i != argMaxRef)
            {
                float r = referenceLogits[i];
                if (!float.IsNegativeInfinity(r))
                    sumOtherRef += Math.Exp((double)r - maxRef);
            }

            if (i != argMaxQuant)
            {
                float q = quantizedLogits[i];
                if (!float.IsNegativeInfinity(q))
                    sumOtherQuant += Math.Exp((double)q - maxQuant);
            }
        }

        double logSumRef = Log1p(sumOtherRef);
        double logSumQuant = Log1p(sumOtherQuant);
        double logEpsilon = Math.Log(epsilon);

        double kl = 0.0;
        for (int i = 0; i < n; i++)
        {
            float r = referenceLogits[i];
            if (float.IsNegativeInfinity(r))
                continue;

            double logP = (i == argMaxRef) ? -logSumRef : (((double)r - maxRef) - logSumRef);
            if (logP > 0.0) logP = 0.0;
            double p = Math.Exp(logP);
            if (p <= 0.0)
                continue;

            float q = quantizedLogits[i];
            double logQ;
            if (float.IsNegativeInfinity(q))
            {
                logQ = logEpsilon;
            }
            else
            {
                logQ = (i == argMaxQuant) ? -logSumQuant : (((double)q - maxQuant) - logSumQuant);
                if (logQ > 0.0) logQ = 0.0;
                if (logQ < logEpsilon) logQ = logEpsilon;
            }

            kl += p * (logP - logQ);
        }

        return kl < 0.0 ? 0.0 : kl;
    }

    private static double Log1p(double x)
    {
        if (x == 0.0) return 0.0;
        double u = 1.0 + x;
        if (u == 1.0) return x;
        return Math.Log(u) * (x / (u - 1.0));
    }

    private static double FindMaxAndArgMaxLogits(ReadOnlySpan<float> logits, string paramName, out int argMax)
    {
        double maxLogit = double.NegativeInfinity;
        argMax = -1;
        bool hasFinite = false;

        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (float.IsNaN(v) || float.IsPositiveInfinity(v))
                throw new ArgumentException($"Logits contain invalid value ({v}) at index {i}.", paramName);

            if (!float.IsNegativeInfinity(v) && (!hasFinite || v > maxLogit))
            {
                hasFinite = true;
                maxLogit = v;
                argMax = i;
            }
        }

        if (!hasFinite)
            throw new ArgumentException("Logits cannot consist exclusively of -Infinity.", paramName);

        return maxLogit;
    }

    private static double FindMaxAndValidateLogits(ReadOnlySpan<float> logits, string paramName)
        => FindMaxAndArgMaxLogits(logits, paramName, out _);

    private static void ValidateBatchShapes(int refLength, int quantLength, int vocabSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        if (refLength == 0)
            throw new ArgumentException("Reference distribution span cannot be empty.");
        if (refLength != quantLength)
            throw new ArgumentException($"Reference length ({refLength}) must equal quantized length ({quantLength}).");
        if (refLength % vocabSize != 0)
            throw new ArgumentException($"Total element count ({refLength}) must be a multiple of vocabSize ({vocabSize}).");
    }

    private static KlDivergenceResult BuildResult(
        double sumKl,
        double maxKl,
        double minKl,
        int top1Matches,
        int tokenCount,
        int vocabSize,
        KlDivergenceOptions opts,
        string? refDevice = null,
        string? candDevice = null,
        TimeSpan duration = default)
    {
        double meanKl = sumKl / tokenCount;
        double top1Rate = (double)top1Matches / tokenCount;

        List<string>? failures = null;
        if (meanKl > opts.Threshold)
        {
            failures ??= new List<string>();
            failures.Add($"Mean KL divergence {meanKl:G6} exceeded threshold {opts.Threshold:G6}.");
        }
        if (opts.MaxKlThreshold.HasValue && maxKl > opts.MaxKlThreshold.Value)
        {
            failures ??= new List<string>();
            failures.Add($"Max KL divergence {maxKl:G6} exceeded MaxKlThreshold {opts.MaxKlThreshold.Value:G6}.");
        }
        if (top1Rate < opts.MinTop1Agreement)
        {
            failures ??= new List<string>();
            failures.Add($"Top-1 agreement rate {top1Rate:P2} was below MinTop1Agreement {opts.MinTop1Agreement:P2}.");
        }

        bool passed = failures is null;
        return new KlDivergenceResult
        {
            MeanKlDivergence = meanKl,
            MaxKlDivergence = maxKl,
            MinKlDivergence = minKl,
            Top1AgreementRate = top1Rate,
            Top1Matches = top1Matches,
            TokenCount = tokenCount,
            VocabSize = vocabSize,
            Threshold = opts.Threshold,
            MaxKlThreshold = opts.MaxKlThreshold,
            MinTop1Agreement = opts.MinTop1Agreement,
            HasNonFiniteQuantizedValues = false,
            Passed = passed,
            FailureReason = passed ? null : string.Join(" ", failures!),
            ReferenceDevice = refDevice,
            CandidateDevice = candDevice,
            Duration = duration
        };
    }
}
