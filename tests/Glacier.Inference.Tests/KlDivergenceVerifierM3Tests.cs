using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Scheduling;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests;

[Collection("SequentialGpu")]
public class KlDivergenceVerifierM3Tests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
            }
            catch
            {
                // Best effort cleanup
            }
        }
    }

    private string CreateTempPath(string name)
    {
        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"glacier_m3_kl_{name}_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void ComputeLogSoftmax_And_ArgMax_HandleShiftMaskingAndTieBreaks()
    {
        float[] logits = { 1.0f, float.NegativeInfinity, 3.5f, 3.5f, -2.0f };
        float[] shifted = { 1001.0f, float.NegativeInfinity, 1003.5f, 1003.5f, 998.0f };

        double[] logP1 = new double[logits.Length];
        double[] logP2 = new double[logits.Length];
        KlDivergenceVerifier.ComputeLogSoftmax(logits, logP1);
        KlDivergenceVerifier.ComputeLogSoftmax(shifted, logP2);

        Assert.True(double.IsNegativeInfinity(logP1[1]));
        double probSum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i == 1)
            {
                Assert.True(double.IsNegativeInfinity(logP2[i]));
            }
            else
            {
                Assert.InRange(Math.Abs(logP1[i] - logP2[i]), 0.0, 1e-12);
                probSum += Math.Exp(logP1[i]);
            }
        }
        Assert.InRange(Math.Abs(probSum - 1.0), 0.0, 1e-12);

        // Lowest-index tie-breaker
        Assert.Equal(2, KlDivergenceVerifier.ArgMax(logits));

        Assert.Throws<ArgumentException>(() =>
        {
            Span<double> tmp = stackalloc double[2];
            KlDivergenceVerifier.ComputeLogSoftmax(new[] { float.NegativeInfinity, float.NegativeInfinity }, tmp);
        });
        Assert.Throws<ArgumentException>(() =>
        {
            Span<double> tmp = stackalloc double[2];
            KlDivergenceVerifier.ComputeLogSoftmax(new[] { float.NaN, 1.0f }, tmp);
        });
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.ArgMax(new[] { 1.0f, float.NaN }));
    }

    [Fact]
    public void EvaluateLogits_And_EvaluateProbabilities_EnforceAllThresholdsAndNonFiniteGuards()
    {
        float[] refLogits =
        {
            2.0f, 0.5f, -1.0f, 0.0f,
            1.0f, 3.0f, 0.0f, -0.5f
        };
        float[] closeQuantLogits =
        {
            2.02f, 0.49f, -0.98f, 0.01f,
            1.01f, 2.98f, 0.02f, -0.51f
        };

        var exact = KlDivergenceVerifier.EvaluateLogits(refLogits, refLogits, vocabSize: 4);
        Assert.True(exact.Passed);
        Assert.InRange(exact.MeanKlDivergence, 0.0, 1e-12);
        Assert.Equal(1.0, exact.Top1AgreementRate);
        Assert.Equal(2, exact.Top1Matches);
        Assert.Equal(2, exact.TokenCount);
        Assert.False(exact.HasNonFiniteQuantizedValues);

        var close = KlDivergenceVerifier.EvaluateLogits(refLogits, closeQuantLogits, vocabSize: 4, new KlDivergenceOptions
        {
            Threshold = 0.05,
            MaxKlThreshold = 0.05,
            MinTop1Agreement = 1.0
        });
        Assert.True(close.Passed);
        Assert.InRange(close.MeanKlDivergence, 0.0, 0.005);
        Assert.Equal(1.0, close.Top1AgreementRate);

        // Token 0 identical, Token 1 flipped strongly -> triggers MaxKlThreshold & MinTop1Agreement failure
        float[] spikedQuant =
        {
            2.0f, 0.5f, -1.0f, 0.0f,
            4.5f, -1.0f, 0.0f, -0.5f
        };
        var failMaxAndTop1 = KlDivergenceVerifier.EvaluateLogits(refLogits, spikedQuant, vocabSize: 4, new KlDivergenceOptions
        {
            Threshold = 5.0,
            MaxKlThreshold = 0.2,
            MinTop1Agreement = 1.0
        });
        Assert.False(failMaxAndTop1.Passed);
        Assert.Equal(0.5, failMaxAndTop1.Top1AgreementRate);
        Assert.NotNull(failMaxAndTop1.FailureReason);

        // Non-finite quantized value -> clean failure result without throwing
        float[] nanQuant = { 2.0f, float.NaN, -1.0f, 0.0f };
        var nanResult = KlDivergenceVerifier.EvaluateSingleTokenLogits(refLogits.AsSpan(0, 4), nanQuant);
        Assert.False(nanResult.Passed);
        Assert.True(nanResult.HasNonFiniteQuantizedValues);
        Assert.NotNull(nanResult.FailureReason);

        // Non-finite reference value -> ArgumentException
        float[] nanRef = { float.PositiveInfinity, 0.5f, -1.0f, 0.0f };
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceVerifier.EvaluateSingleTokenLogits(nanRef, closeQuantLogits.AsSpan(0, 4)));

        // EvaluateProbabilities matches analytical D_KL([0.8, 0.2] || [0.5, 0.5]) = 0.8*ln(1.6) + 0.2*ln(0.4)
        double expectedKl = 0.8 * Math.Log(0.8 / 0.5) + 0.2 * Math.Log(0.2 / 0.5);
        var probRes = KlDivergenceVerifier.EvaluateProbabilities(
            new[] { 0.8f, 0.2f },
            new[] { 0.5f, 0.5f },
            vocabSize: 2,
            threshold: 0.5);
        Assert.True(probRes.Passed);
        Assert.InRange(Math.Abs(probRes.MeanKlDivergence - expectedKl), 0.0, 1e-6);
    }

    [Fact]
    public void VerifyTargets_CallsResetKvCache_SupportsBatchAndSequential_AndCatchesNonFiniteCandidate()
    {
        var tokenizer = TestTokenizerFactory.CreateSmallTokenizer(vocabSize: 8);
        using var refTarget = new DeterministicLogitTarget(tokenizer, shiftBias: 0.0f, injectNaN: false);
        using var candGood = new DeterministicLogitTarget(tokenizer, shiftBias: 0.02f, injectNaN: false);
        using var candBad = new DeterministicLogitTarget(tokenizer, shiftBias: 0.0f, injectNaN: true);

        int[] tokens = { 1, 2, 3, 4 };
        var batchRes = KlDivergenceVerifier.VerifyTargets(
            refTarget,
            candGood,
            tokens,
            new KlDivergenceOptions { UseBatchLogitVerification = true, Threshold = 0.05 });

        Assert.Equal(1, refTarget.ResetKvCacheCount);
        Assert.Equal(1, candGood.ResetKvCacheCount);
        Assert.True(batchRes.Passed, batchRes.FailureReason);
        Assert.Equal(tokens.Length, batchRes.TokenCount);
        Assert.Equal(1.0, batchRes.Top1AgreementRate);

        var seqRes = KlDivergenceVerifier.VerifyTargets(
            refTarget,
            candGood,
            tokens,
            new KlDivergenceOptions { UseBatchLogitVerification = false, Threshold = 0.05 });

        Assert.Equal(2, refTarget.ResetKvCacheCount);
        Assert.Equal(2, candGood.ResetKvCacheCount);
        Assert.InRange(Math.Abs(batchRes.MeanKlDivergence - seqRes.MeanKlDivergence), 0.0, 1e-12);

        // Prompt + continuation overload
        var contRes = KlDivergenceVerifier.VerifyTargets(
            refTarget,
            candGood,
            promptTokens: new[] { 1, 2 },
            continuationTokens: new[] { 3, 4 },
            new KlDivergenceOptions { UseBatchLogitVerification = true, Threshold = 0.05 });
        Assert.True(contRes.Passed);
        Assert.Equal(3, contRes.TokenCount); // prompt final token + 2 continuations

        // Non-finite candidate returns HasNonFiniteQuantizedValues = true
        var nanRes = KlDivergenceVerifier.VerifyTargets(refTarget, candBad, tokens);
        Assert.False(nanRes.Passed);
        Assert.True(nanRes.HasNonFiniteQuantizedValues);

        // Mismatched vocabularies throw ArgumentException
        var smallTok = TestTokenizerFactory.CreateSmallTokenizer(vocabSize: 6);
        using var wrongVocabTarget = new DeterministicLogitTarget(smallTok, shiftBias: 0.0f, injectNaN: false);
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.VerifyTargets(refTarget, wrongVocabTarget, tokens));
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void VerifyModels_And_VerifyPrompts_RealInferenceSessions_PassForQ8_0_And_FailForCorrupted()
    {
        string refF32Path = CreateTempPath("ref_f32");
        string candQ8Path = CreateTempPath("cand_q8");
        string badIq2Path = CreateTempPath("bad_divergent");

        var f32Layer = new UnslothDynamic31M3Tests.LayerTensorTypes(
            GgufType.F32, GgufType.F32, GgufType.F32, GgufType.F32,
            GgufType.F32, GgufType.F32, GgufType.F32);
        var q8Layer = new UnslothDynamic31M3Tests.LayerTensorTypes(
            GgufType.Q8_0, GgufType.Q8_0, GgufType.Q8_0, GgufType.Q8_0,
            GgufType.Q8_0, GgufType.Q8_0, GgufType.Q8_0);
        var iq2Layer = new UnslothDynamic31M3Tests.LayerTensorTypes(
            GgufType.IQ2_XXS, GgufType.IQ2_XXS, GgufType.IQ2_XXS, GgufType.IQ2_XXS,
            GgufType.IQ2_XXS, GgufType.IQ2_XXS, GgufType.IQ2_XXS);

        UnslothDynamic31M3Tests.WriteMixedQwen2Gguf(refF32Path, GgufType.F32, GgufType.F32, new[] { f32Layer }, seed: 777);
        UnslothDynamic31M3Tests.WriteMixedQwen2Gguf(candQ8Path, GgufType.Q8_0, GgufType.Q8_0, new[] { q8Layer }, seed: 777);
        UnslothDynamic31M3Tests.WriteMixedQwen2Gguf(badIq2Path, GgufType.IQ4_NL, GgufType.Q4_0, new[] { iq2Layer }, seed: 999);

        string[] prompts = { "abcdef", "ghijkl" };

        var cpuPass = KlDivergenceVerifier.VerifyModels(
            refF32Path,
            candQ8Path,
            prompts,
            maxSeqLen: 64,
            device: "cpu",
            engine: InferenceEngineType.Cpu,
            formatChat: false,
            evaluateAllPromptTokens: true,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.99 });
        Assert.True(cpuPass.Passed, cpuPass.FailureReason);
        Assert.InRange(cpuPass.MeanKlDivergence, 0.0, 0.01);
        Assert.Equal(1.0, cpuPass.Top1AgreementRate);

        var badFail = KlDivergenceVerifier.VerifyModels(
            refF32Path,
            badIq2Path,
            prompts,
            maxSeqLen: 64,
            device: "cpu",
            engine: InferenceEngineType.Cpu,
            formatChat: false,
            evaluateAllPromptTokens: true,
            options: new KlDivergenceOptions { Threshold = 0.001, MinTop1Agreement = 0.99 });
        Assert.False(badFail.Passed);
        Assert.NotNull(badFail.FailureReason);

        if (D3D12Context.IsSupported)
        {
            var d3dPass = KlDivergenceVerifier.VerifyModels(
                refF32Path,
                candQ8Path,
                prompts,
                maxSeqLen: 64,
                device: null,
                engine: InferenceEngineType.DirectML,
                formatChat: false,
                evaluateAllPromptTokens: true,
                options: new KlDivergenceOptions { Threshold = 0.05 });
            Assert.True(d3dPass.Passed, d3dPass.FailureReason);
        }
    }

    [Fact]
    public void VerifySyntheticTieredMoE_PassesIdenticalAndFlagsPerturbedLmHead()
    {
        const int dim = 32;
        const int ffnDim = 32;
        const int vocab = 16;

        float[] lmHeadRef = new float[vocab * dim];
        float[] lmHeadClose = new float[vocab * dim];
        float[] lmHeadBad = new float[vocab * dim];
        var rng = new Random(1234);
        for (int i = 0; i < lmHeadRef.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 2.0 - 1.0);
            lmHeadRef[i] = v;
            lmHeadClose[i] = v + 0.002f * MathF.Sin(i);
            lmHeadBad[i] = -v * 3.0f;
        }

        using var refRunner = CreateSyntheticRunner(dim, ffnDim, vocab, lmHeadRef);
        using var closeRunner = CreateSyntheticRunner(dim, ffnDim, vocab, lmHeadClose);
        using var badRunner = CreateSyntheticRunner(dim, ffnDim, vocab, lmHeadBad);

        float[] hiddenBatch = new float[3 * dim];
        for (int i = 0; i < hiddenBatch.Length; i++)
            hiddenBatch[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var pass = KlDivergenceVerifier.VerifySyntheticTieredMoE(
            refRunner,
            closeRunner,
            hiddenBatch,
            batchSize: 3,
            isSpeculativeDraftBatch: false,
            new KlDivergenceOptions { Threshold = 0.05 });
        Assert.True(pass.Passed, pass.FailureReason);
        Assert.Equal(3, pass.TokenCount);
        Assert.Equal(vocab, pass.VocabSize);

        var fail = KlDivergenceVerifier.VerifySyntheticTieredMoE(
            refRunner,
            badRunner,
            hiddenBatch,
            batchSize: 3,
            isSpeculativeDraftBatch: true,
            new KlDivergenceOptions { Threshold = 0.05 });
        Assert.False(fail.Passed);
        Assert.True(fail.MeanKlDivergence > 0.05);
    }

    [Fact]
    public void ComputeLogSoftmax_And_TokenKl_LargeMargin_AvoidsCollapse()
    {
        float[] refLogits = [42f, 0f, 0f];
        float[] quantLogits = [40f, 0f, 0f];
        double[] logProbs = new double[3];

        KlDivergenceVerifier.ComputeLogSoftmax(quantLogits, logProbs);
        double expectedArgMaxLogProb = -2.0 * Math.Exp(-40.0);
        Assert.True(logProbs[0] < 0.0 && double.IsFinite(logProbs[0]), $"Expected negative logProbs[0], got {logProbs[0]}");
        Assert.True(Math.Abs(logProbs[0] - expectedArgMaxLogProb) / Math.Abs(expectedArgMaxLogProb) < 1e-6,
            $"LogSoftmax[0] {logProbs[0]} diverged from {expectedArgMaxLogProb}");

        double expectedKl = 2.0 * Math.Exp(-40.0) - 6.0 * Math.Exp(-42.0);
        double kl = KlDivergenceVerifier.ComputeTokenKlDivergence(refLogits, quantLogits, epsilon: 1e-30);
        Assert.True(kl > 0.0 && double.IsFinite(kl), $"Expected positive KL at large margin, got {kl}");
        Assert.True(Math.Abs(kl - expectedKl) / expectedKl < 1e-4, $"KL {kl} diverged from {expectedKl}");
    }

    private static unsafe TieredMoERunner CreateSyntheticRunner(int dim, int ffnDim, int vocab, float[] lmHeadWeights)
    {
        var layer = new LayerWeights
        {
            AttnOutWeight = null,
            AttnOutType = GgufType.F32
        };
        var routing = new ExpertRoutingProfile(numLayers: 1, numExperts: 2);
        var coScheduler = new BandwidthCoScheduler();

        return new TieredMoERunner(
            new[] { layer },
            dim,
            ffnDim,
            numExperts: 2,
            topK: 1,
            vocab,
            normTopK: false,
            routing,
            coScheduler,
            slotManager: null,
            hostArena: null,
            new TieredRunnerOptions { Policy = TieringExecutionPolicy.HybridGpuHostRam },
            lmHeadWeights);
    }

    private sealed class DeterministicLogitTarget : ISpeculativeTarget
    {
        private readonly float _shiftBias;
        private readonly bool _injectNaN;
        private readonly float[] _currentLogits;
        private int _lastToken;

        public DeterministicLogitTarget(BpeTokenizer tokenizer, float shiftBias, bool injectNaN)
        {
            Tokenizer = tokenizer;
            _shiftBias = shiftBias;
            _injectNaN = injectNaN;
            _currentLogits = new float[tokenizer.VocabSize];
        }

        public int ResetKvCacheCount { get; private set; }
        public int MaxSeqLen => 128;
        public BpeTokenizer Tokenizer { get; }
        public bool SupportsLogitVerification => true;

        public void ResetKvCache() => ResetKvCacheCount++;

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            for (int i = 0; i < promptTokens.Length; i++)
                _lastToken = promptTokens[i];
            PopulateLogits(_lastToken, _currentLogits);
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            _lastToken = token;
            if (computeLogits)
                PopulateLogits(token, _currentLogits);
        }

        public void CopyCurrentLogits(Span<float> destination)
        {
            _currentLogits.CopyTo(destination);
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            for (int i = 0; i < tokens.Length; i++)
                predictedTokens[i] = (tokens[i] + 1) % Tokenizer.VocabSize;
        }

        public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
        {
            int vocab = Tokenizer.VocabSize;
            for (int i = 0; i < tokens.Length; i++)
                PopulateLogits(tokens[i], batchLogitsOut.Slice(i * vocab, vocab));
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
            => (_lastToken + 1) % Tokenizer.VocabSize;

        private void PopulateLogits(int token, Span<float> dst)
        {
            int vocab = dst.Length;
            int preferred = (token + 1) % vocab;
            for (int v = 0; v < vocab; v++)
            {
                dst[v] = v == preferred
                    ? 3.0f - _shiftBias
                    : -0.5f + 0.05f * ((v + token) % 5) + (_shiftBias * (v == 0 ? 1f : -0.2f));
            }
            if (_injectNaN)
                dst[0] = float.NaN;
        }

        public void Dispose() { }
    }

    private static class TestTokenizerFactory
    {
        public static BpeTokenizer CreateSmallTokenizer(int vocabSize)
        {
            var tokens = new string[vocabSize];
            tokens[0] = "<unk>";
            tokens[1] = "<s>";
            tokens[2] = "</s>";
            for (int i = 3; i < vocabSize; i++)
                tokens[i] = $"t{i}";
            return new BpeTokenizer(tokens, eosTokenId: 2, bosTokenId: 1);
        }
    }
}
