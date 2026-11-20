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
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

public sealed unsafe class Tier1QuantAndKlE2ETests
{
    // =========================================================================
    // F8: Unsloth Dynamic 3.1 Quantization Support (T1-F8-01 .. T1-F8-04)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F8_01_GgufTypes_UnslothDynamic31_BlockGeometryIsExact()
    {
        var expected = new (GgufType Type, int BlockElems, int BlockBytes)[]
        {
            (GgufType.IQ2_XXS, 256, 66),
            (GgufType.IQ2_XS,  256, 74),
            (GgufType.IQ2_S,   256, 82),
            (GgufType.IQ3_XXS, 256, 98),
            (GgufType.IQ3_S,   256, 110),
            (GgufType.IQ4_NL,  32,  18),
            (GgufType.IQ4_XS,  256, 136),
        };

        foreach (var (type, elems, bytes) in expected)
        {
            var info = new GgufTensorInfo
            {
                Name = $"test_{type}",
                DimensionsCount = 1,
                Dimensions = new ulong[] { (ulong)(elems * 2) },
                Type = type,
                Offset = 0
            };
            Assert.Equal((long)bytes, GgufTypes.GetRowBytes(type, elems));
            Assert.Equal((long)bytes * 2, GgufTypes.GetRowBytes(type, elems * 2));
            Assert.Equal((ulong)bytes * 2UL, info.GetByteSize());
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F8_02_QuantKernels_UnslothDynamic31_QuantizeAndDequantize_RoundTripValid()
    {
        const int k = 256;
        float[] src = new float[k];
        for (int i = 0; i < k; i++)
        {
            src[i] = MathF.Sin(i * 0.17f) * 0.8f + MathF.Cos(i * 0.05f) * 0.3f;
        }
        float[] dequant = new float[k];

        fixed (float* pSrc = src, pDq = dequant)
        {
            // IQ2_XS
            byte[] iq2xs = new byte[74];
            fixed (byte* pB = iq2xs)
            {
                QuantKernels.QuantizeIQ2_XS(pSrc, (BlockIQ2_XS*)pB, k);
                QuantKernels.DequantizeIQ2_XS((BlockIQ2_XS*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.70f);

            // IQ2_S
            byte[] iq2s = new byte[82];
            fixed (byte* pB = iq2s)
            {
                QuantKernels.QuantizeIQ2_S(pSrc, (BlockIQ2_S*)pB, k);
                QuantKernels.DequantizeIQ2_S((BlockIQ2_S*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.70f);

            // IQ3_XXS
            byte[] iq3xxs = new byte[98];
            fixed (byte* pB = iq3xxs)
            {
                QuantKernels.QuantizeIQ3_XXS(pSrc, (BlockIQ3_XXS*)pB, k);
                QuantKernels.DequantizeIQ3_XXS((BlockIQ3_XXS*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.85f);

            // IQ3_S
            byte[] iq3s = new byte[110];
            fixed (byte* pB = iq3s)
            {
                QuantKernels.QuantizeIQ3_S(pSrc, (BlockIQ3_S*)pB, k);
                QuantKernels.DequantizeIQ3_S((BlockIQ3_S*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.85f);

            // IQ4_NL (8 blocks of 32)
            byte[] iq4nl = new byte[8 * 18];
            fixed (byte* pB = iq4nl)
            {
                QuantKernels.QuantizeIQ4_NL(pSrc, (BlockIQ4_NL*)pB, k);
                QuantKernels.DequantizeIQ4_NL((BlockIQ4_NL*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.95f);

            // IQ2_XXS (1 super-block of 256)
            byte[] iq2xxs = UpgradeScenarioFixtures.QuantizeF32ToIq(src, rows: 1, cols: k, GgufType.IQ2_XXS);
            fixed (byte* pB = iq2xxs)
            {
                QuantKernels.DequantizeIQ2_XXS((BlockIQ2_XXS*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.65f);

            // IQ4_XS (1 super-block of 256)
            byte[] iq4xs = UpgradeScenarioFixtures.QuantizeF32ToIq(src, rows: 1, cols: k, GgufType.IQ4_XS);
            fixed (byte* pB = iq4xs)
            {
                QuantKernels.DequantizeIQ4_XS((BlockIQ4_XS*)pB, pDq, k);
            }
            Assert.All(dequant, v => Assert.True(float.IsFinite(v)));
            Assert.True(Fx1Vectors.CosineSimilarity(src, dequant) > 0.95f);
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F8_03_QuantKernels_UnslothDynamic31_VecDotMatchesDequantDotAcrossAll7Types()
    {
        const int cols = 256;
        float[] x = Fx1Vectors.GenerateRampVector(cols, -0.5f, 0.004f);
        float[] dq = new float[cols];

        var iqTypes = new[]
        {
            GgufType.IQ2_XXS, GgufType.IQ2_XS, GgufType.IQ2_S,
            GgufType.IQ3_XXS, GgufType.IQ3_S, GgufType.IQ4_NL, GgufType.IQ4_XS
        };

        foreach (var type in iqTypes)
        {
            byte[] payload = FxQuantData.CreateValidIqPayload(type, rows: 1, cols: cols, seed: (int)type + 400);
            fixed (byte* pW = payload)
            fixed (float* pX = x, pDq = dq)
            {
                switch (type)
                {
                    case GgufType.IQ2_XXS: QuantKernels.DequantizeIQ2_XXS((BlockIQ2_XXS*)pW, pDq, cols); break;
                    case GgufType.IQ2_XS:  QuantKernels.DequantizeIQ2_XS((BlockIQ2_XS*)pW, pDq, cols);   break;
                    case GgufType.IQ2_S:   QuantKernels.DequantizeIQ2_S((BlockIQ2_S*)pW, pDq, cols);     break;
                    case GgufType.IQ3_XXS: QuantKernels.DequantizeIQ3_XXS((BlockIQ3_XXS*)pW, pDq, cols); break;
                    case GgufType.IQ3_S:   QuantKernels.DequantizeIQ3_S((BlockIQ3_S*)pW, pDq, cols);     break;
                    case GgufType.IQ4_NL:  QuantKernels.DequantizeIQ4_NL((BlockIQ4_NL*)pW, pDq, cols);   break;
                    case GgufType.IQ4_XS:  QuantKernels.DequantizeIQ4_XS((BlockIQ4_XS*)pW, pDq, cols);   break;
                }

                float expectedDot = 0f;
                for (int i = 0; i < cols; i++) expectedDot += dq[i] * x[i];

                float vecDot = type switch
                {
                    GgufType.IQ2_XXS => QuantKernels.VecDotIQ2_XXS((BlockIQ2_XXS*)pW, pX, cols),
                    GgufType.IQ2_XS  => QuantKernels.VecDotIQ2_XS((BlockIQ2_XS*)pW, pX, cols),
                    GgufType.IQ2_S   => QuantKernels.VecDotIQ2_S((BlockIQ2_S*)pW, pX, cols),
                    GgufType.IQ3_XXS => QuantKernels.VecDotIQ3_XXS((BlockIQ3_XXS*)pW, pX, cols),
                    GgufType.IQ3_S   => QuantKernels.VecDotIQ3_S((BlockIQ3_S*)pW, pX, cols),
                    GgufType.IQ4_NL  => QuantKernels.VecDotIQ4_NL((BlockIQ4_NL*)pW, pX, cols),
                    GgufType.IQ4_XS  => QuantKernels.VecDotIQ4_XS((BlockIQ4_XS*)pW, pX, cols),
                    _ => throw new NotSupportedException()
                };

                Assert.True(float.IsFinite(vecDot));
                Assert.True(MathF.Abs(expectedDot - vecDot) < 1e-2f,
                    $"Type {type}: dequant dot {expectedDot:F5} != VecDot {vecDot:F5}");
            }
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F8_04_CpuInferenceSession_LoadsAndRunsSyntheticIqQuantizedGgufModels()
    {
        using var tmp = new TempDirScope();
        var iqTypes = new[]
        {
            GgufType.IQ2_XXS, GgufType.IQ2_XS, GgufType.IQ2_S,
            GgufType.IQ3_XXS, GgufType.IQ3_S, GgufType.IQ4_NL, GgufType.IQ4_XS
        };

        foreach (var iqType in iqTypes)
        {
            string path = SyntheticGgufBuilder.BuildTinyQwen2Model(
                tmp.GetFilePath($"tiny_{iqType}.gguf"),
                dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 16,
                linearType: iqType, seed: 800 + (int)iqType);

            using var session = new InferenceSession(path, maxSeqLen: 16, device: "cpu", engine: InferenceEngineType.Cpu);
            Assert.Equal("Cpu", session.ActiveBackend);

            session.Prefill(new[] { 1, 5 });
            session.ForwardToken(9, 2, computeLogits: true);

            float[] logits = new float[session.VocabSize];
            ((ISpeculativeTarget)session).CopyCurrentLogits(logits);
            Assert.All(logits, x => Assert.True(float.IsFinite(x)));
        }

        // Verify mixed-tensor UD profiles actually emit IQ2_XXS / IQ4_XS tensors
        string ud2Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(tmp.GetFilePath("mixed_ud_iq2xs.gguf"), "UD-IQ2_XS");
        string ud4Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(tmp.GetFilePath("mixed_ud_iq4xs.gguf"), "UD-IQ4_XS");
        using (var g2 = GgufFile.Open(ud2Path))
        {
            Assert.Contains(g2.Tensors.Values, t => t.Type == GgufType.IQ2_XXS);
            Assert.Contains(g2.Tensors.Values, t => t.Type == GgufType.IQ2_XS);
        }
        using (var g4 = GgufFile.Open(ud4Path))
        {
            Assert.Contains(g4.Tensors.Values, t => t.Type == GgufType.IQ4_XS);
            Assert.Contains(g4.Tensors.Values, t => t.Type == GgufType.IQ4_NL);
        }
    }

    // =========================================================================
    // F9: KL Divergence Verification Suite (T1-F9-01 .. T1-F9-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F9_01_KlDivergenceVerifier_IdenticalAndPerturbedLogits_MatchAnalyticalOracle()
    {
        float[] p = Fx4Kl.GenerateDeterministicLogits(vocabSize: 32, seed: 901);
        float[] qSame = (float[])p.Clone();

        double klSame = KlDivergenceVerifier.ComputeTokenKlDivergence(p, qSame);
        Assert.InRange(klSame, 0.0, 1e-9);

        float[] qPerturbed = (float[])p.Clone();
        for (int i = 0; i < qPerturbed.Length; i++) qPerturbed[i] += 0.05f * ((i % 3) - 1);

        double klMeasured = KlDivergenceVerifier.ComputeTokenKlDivergence(p, qPerturbed);
        double klOracle = Fx4Kl.ComputeAnalyticalKl(p, qPerturbed);
        Assert.True(klMeasured > 0.0);
        Assert.True(Math.Abs(klMeasured - klOracle) < 1e-6, $"KL mismatch: measured={klMeasured}, oracle={klOracle}");
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F9_02_KlDivergenceVerifier_EvaluateLogitsSequence_ComputesExactSummaryMetrics()
    {
        const int steps = 4;
        const int vocab = 24;
        float[] pAll = new float[steps * vocab];
        float[] qAll = new float[steps * vocab];

        for (int s = 0; s < steps; s++)
        {
            var row = Fx4Kl.GenerateDeterministicLogits(vocab, seed: 920 + s);
            row.CopyTo(pAll, s * vocab);
            row.CopyTo(qAll, s * vocab);
            qAll[s * vocab] += 0.1f; // Small deterministic perturbation
        }

        var res = KlDivergenceVerifier.EvaluateLogits(pAll, qAll, vocab, options: new KlDivergenceOptions
        {
            Threshold = 0.05,
            MaxKlThreshold = 0.25
        });

        Assert.Equal(steps, res.TokenCount);
        Assert.True(res.MeanKlDivergence > 0.0);
        Assert.True(res.MaxKlDivergence >= res.MeanKlDivergence);
        Assert.True(res.Passed);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F9_03_KlDivergenceVerifier_VerifyTargets_PassesIdenticalAndFailsDivergentTarget()
    {
        using var refTarget = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, perturbation: 0.0f);
        using var identicalTarget = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, perturbation: 0.0f);
        using var divergentTarget = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, perturbation: 2.5f, invertLogits: true);

        int[] prompt = { 2, 4, 7, 10 };

        var passResult = KlDivergenceVerifier.VerifyTargets(
            refTarget, identicalTarget, prompt,
            new KlDivergenceOptions { Threshold = 1e-4, MaxKlThreshold = 1e-3, MinTop1Agreement = 1.0 });
        Assert.True(passResult.Passed);
        Assert.InRange(passResult.MeanKlDivergence, 0.0, 1e-7);
        Assert.Equal(1.0, passResult.Top1AgreementRate);

        var failResult = KlDivergenceVerifier.VerifyTargets(
            refTarget, divergentTarget, prompt,
            new KlDivergenceOptions { Threshold = 0.01, MaxKlThreshold = 0.05, MinTop1Agreement = 0.8 });
        Assert.False(failResult.Passed);
        Assert.True(failResult.MeanKlDivergence > 0.5, $"Expected MeanKlDivergence > 0.5, got {failResult.MeanKlDivergence:F4}");
        Assert.True(failResult.Top1AgreementRate < 0.5, $"Expected Top1AgreementRate < 0.5, got {failResult.Top1AgreementRate:F4}");
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F9_04_KlDivergenceVerifier_VerifySyntheticTieredMoE_ComparesDenseVsTailSkipped()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16);
        using var baseline = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: new TieredRunnerOptions { RouteTailSkipMinRank = 0 }, lmHeadWeights: moe.LmHead);

        using var candidate = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: new TieredRunnerOptions { RouteTailSkipMinRank = 2 }, lmHeadWeights: moe.LmHead);

        float[] inputs = Fx1Vectors.GenerateRampVector(3 * 32, -0.2f, 0.01f);
        var res = KlDivergenceVerifier.VerifySyntheticTieredMoE(
            baseline, candidate, inputs, batchSize: 3, isSpeculativeDraftBatch: false,
            new KlDivergenceOptions { Threshold = 1.0, MaxKlThreshold = 2.0 });

        Assert.Equal(3, res.TokenCount);
        Assert.True(double.IsFinite(res.MeanKlDivergence));
        Assert.True(res.MeanKlDivergence >= 0.0);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F9_05_KlDivergenceVerifier_PostVerifySessionStateIsolation_MatchesFreshSession()
    {
        using var tmp = new TempDirScope();
        string modelPath = UpgradeScenarioFixtures.BuildTinyDenseWithClearMargins(
            tmp.GetFilePath("kl_state_isolation.gguf"),
            dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            seed: 955);

        using var sRef = new InferenceSession(modelPath, maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Fp32);
        using var sCand = new InferenceSession(modelPath, maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Fp32);
        using var sFresh = new InferenceSession(modelPath, maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Fp32);

        // Dirty sessions before verification to prove VerifyTargets resets/isolates prior state
        sRef.Prefill(new[] { 10, 11, 12 });
        sCand.Prefill(new[] { 13, 14, 15 });

        var res = KlDivergenceVerifier.VerifyTargets(
            sRef, sCand, new[] { 1, 3, 5, 7 },
            new KlDivergenceOptions { Threshold = 1e-4, MaxKlThreshold = 1e-3, MinTop1Agreement = 1.0 });
        Assert.True(res.Passed);
        Assert.Equal(4, res.TokenCount);
        Assert.InRange(res.MeanKlDivergence, 0.0, 1e-7);

        // Run subsequent inference on verified session and compare logits against fresh untouched session
        int[] followUp = { 2, 4, 6 };
        sCand.Prefill(followUp);
        sCand.ForwardToken(8, pos: 3, computeLogits: true);
        sFresh.Prefill(followUp);
        sFresh.ForwardToken(8, pos: 3, computeLogits: true);

        float[] candLogits = new float[sCand.VocabSize];
        float[] freshLogits = new float[sFresh.VocabSize];
        ((ISpeculativeTarget)sCand).CopyCurrentLogits(candLogits);
        ((ISpeculativeTarget)sFresh).CopyCurrentLogits(freshLogits);

        float maxDiff = 0f;
        for (int i = 0; i < candLogits.Length; i++)
        {
            float d = MathF.Abs(candLogits[i] - freshLogits[i]);
            if (d > maxDiff) maxDiff = d;
        }

        Assert.True(maxDiff < 1e-5f, $"Post-verify session logits diverged from fresh session: maxDiff={maxDiff:E3}");
        Assert.Equal(KlDivergenceVerifier.ArgMax(freshLogits), KlDivergenceVerifier.ArgMax(candLogits));
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier1")]
[Trait("Hardware", "Gpu")]
public sealed class Tier1QuantGpuE2ETests
{
    private static readonly GgufType[] All7Ud31IqTypes =
    {
        GgufType.IQ2_XXS,
        GgufType.IQ2_XS,
        GgufType.IQ2_S,
        GgufType.IQ3_XXS,
        GgufType.IQ3_S,
        GgufType.IQ4_NL,
        GgufType.IQ4_XS
    };

    [UpgradeD3D12Fact]
    public void T1_F8_05_D3D12_All7Ud31IqTypes_ForwardAndBatch_MatchCpuModel()
    {
        using var tmp = new TempDirScope();
        int[] batchTokens = { 2, 5, 9, 13 };

        foreach (var iqType in All7Ud31IqTypes)
        {
            string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
                tmp.GetFilePath($"d3d12_{iqType}.gguf"),
                dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 16,
                linearType: iqType, seed: 850 + (int)iqType);

            using var gguf = GgufFile.Open(ggufPath);
            var weights = new ModelWeights(gguf);

            using var cpuKv = new KVCache(1, 2, 64, 16, precision: KvCachePrecision.Fp32);
            using var cpuModel = new Qwen2Model(weights);
            float[] cpuSingle = new float[weights.VocabSize];
            cpuModel.Forward(token: 3, pos: 0, cpuKv, cpuSingle, computeLogits: true);

            cpuKv.Reset();
            float[] cpuBatch = new float[weights.VocabSize];
            cpuModel.ForwardBatch(batchTokens, startPos: 0, cpuBatch, cpuKv, computeLogits: true);

            using var d3dCtx = new D3D12Context();
            using var d3dModel = new Qwen2D3D12Model(d3dCtx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp32);
            float[] d3dSingle = new float[weights.VocabSize];
            d3dModel.Forward(token: 3, pos: 0, d3dSingle, computeLogits: true);
            Assert.All(d3dSingle, x => Assert.True(float.IsFinite(x)));
            float cosSingle = Fx1Vectors.CosineSimilarity(cpuSingle, d3dSingle);
            Assert.True(cosSingle > 0.99f, $"D3D12 {iqType} Forward vs CPU cosine {cosSingle:F4} <= 0.99");

            d3dModel.ResetKvCache();
            float[] d3dBatch = new float[weights.VocabSize];
            d3dModel.ForwardBatch(batchTokens, startPos: 0, d3dBatch, computeLogits: true);
            Assert.All(d3dBatch, x => Assert.True(float.IsFinite(x)));
            float cosBatch = Fx1Vectors.CosineSimilarity(cpuBatch, d3dBatch);
            Assert.True(cosBatch > 0.99f, $"D3D12 {iqType} ForwardBatch vs CPU cosine {cosBatch:F4} <= 0.99");
        }
    }

    [UpgradeCudaFact]
    public void T1_F8_05_Cuda_All7Ud31IqTypes_ForwardAndBatch_MatchCpuModel()
    {
        using var tmp = new TempDirScope();
        using var gpu = new GpuContext(0);
        int[] batchTokens = { 2, 5, 9, 13 };

        foreach (var iqType in All7Ud31IqTypes)
        {
            string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
                tmp.GetFilePath($"cuda_{iqType}.gguf"),
                dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 16,
                linearType: iqType, seed: 870 + (int)iqType);

            using var gguf = GgufFile.Open(ggufPath);
            var weights = new ModelWeights(gguf);

            using var cpuKv = new KVCache(1, 2, 64, 16, precision: KvCachePrecision.Fp32);
            using var cpuModel = new Qwen2Model(weights);
            float[] cpuSingle = new float[weights.VocabSize];
            cpuModel.Forward(token: 3, pos: 0, cpuKv, cpuSingle, computeLogits: true);

            cpuKv.Reset();
            float[] cpuBatch = new float[weights.VocabSize];
            cpuModel.ForwardBatch(batchTokens, startPos: 0, cpuBatch, cpuKv, computeLogits: true);

            using var cudaModel = new Qwen2GpuModel(gpu, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp32);
            float[] cudaSingle = new float[weights.VocabSize];
            cudaModel.Forward(token: 3, pos: 0, null, cudaSingle, computeLogits: true);
            Assert.All(cudaSingle, x => Assert.True(float.IsFinite(x)));
            float cosSingle = Fx1Vectors.CosineSimilarity(cpuSingle, cudaSingle);
            Assert.True(cosSingle > 0.99f, $"CUDA {iqType} Forward vs CPU cosine {cosSingle:F4} <= 0.99");

            cudaModel.ResetKvCache();
            float[] cudaBatch = new float[weights.VocabSize];
            cudaModel.ForwardBatch(batchTokens, startPos: 0, cudaBatch, computeLogits: true);
            Assert.All(cudaBatch, x => Assert.True(float.IsFinite(x)));
            float cosBatch = Fx1Vectors.CosineSimilarity(cpuBatch, cudaBatch);
            Assert.True(cosBatch > 0.99f, $"CUDA {iqType} ForwardBatch vs CPU cosine {cosBatch:F4} <= 0.99");
        }
    }
}
