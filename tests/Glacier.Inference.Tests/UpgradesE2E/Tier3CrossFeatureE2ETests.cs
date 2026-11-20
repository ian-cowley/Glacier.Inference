using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Scheduling;
using Vortice.Direct3D12;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier3")]
public sealed unsafe class Tier3CrossFeatureE2ETests
{
    [Fact]
    public void T3_01_F1_F4_CpuQ4KvCache_WithVramReserveZero_ExpandsTokenCapacity()
    {
        const ulong dedicated = 4UL * 1024 * 1024 * 1024;
        const ulong osBudget = 3800UL * 1024 * 1024;
        const ulong modelFootprint = 3650UL * 1024 * 1024;

        var headroomDefault = WddmVramGuard.EvaluateHeadroom(
            dedicated, osBudget, modelFootprint, isWindowsWddm: true, explicitReserveBytes: null);
        var headroomZero = WddmVramGuard.EvaluateHeadroom(
            dedicated, osBudget, modelFootprint, isWindowsWddm: true, explicitReserveBytes: 0UL);

        Assert.Equal(0UL, headroomDefault.SafeAvailableBytes); // Default ~350MB reserve leaves 0 headroom
        Assert.True(headroomZero.SafeAvailableBytes >= 140UL * 1024 * 1024);

        const int layers = 2;
        const int headsKv = 4;
        const int headDim = 64;
        ulong perTokenQ4 = (ulong)(layers * 2 * headsKv * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Q4_0));
        ulong perTokenFp16 = (ulong)(layers * 2 * headsKv * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Fp16));

        // Compare how many tokens fit inside a fixed 512 KiB budget (scaled budget)
        const ulong fixedBudgetBytes = 512UL * 1024;
        int tokensQ4 = (int)(fixedBudgetBytes / perTokenQ4);
        int tokensFp16 = (int)(fixedBudgetBytes / perTokenFp16);

        Assert.True(tokensQ4 >= (int)(tokensFp16 * 3.2), $"Expected Q4 tokens ({tokensQ4}) >= 3.2x Fp16 ({tokensFp16})");

        using var expandedCache = new KVCache(layers, headsKv, headDim, maxSeqLen: tokensQ4, precision: KvCachePrecision.Q4_0);
        Assert.True((ulong)expandedCache.TotalBytes <= fixedBudgetBytes);

        var (k, v, q) = Fx1Vectors.GenerateOrthogonalVectors(headsKv, headsKv, headDim, seed: 3001);
        int lastPos = tokensQ4 - 1;
        expandedCache.Store(1, lastPos, k, v);

        float[] kRead = new float[headDim];
        expandedCache.ReadKey(1, 0, lastPos, kRead);
        Assert.True(Fx1Vectors.CosineSimilarity(k.AsSpan(0, headDim), kRead) > 0.96f);

        fixed (float* pQ = q)
        {
            Assert.True(float.IsFinite(expandedCache.DotKey(1, 0, lastPos, pQ)));
        }
    }

    [Fact]
    public async Task T3_04_F5_F6_F7_TieredMoERunner_MissSplit_TailSkip6_WithDFlashBatch()
    {
        using var tmp = new TempDirScope();
        string moePath = UpgradeScenarioFixtures.BuildTinyQwen2MoeModel(
            tmp.GetFilePath("t3_04_moe.gguf"),
            dim: 64, ffnDim: 64, expertFfnDim: 64,
            nLayers: 2, nHeads: 4, nHeadsKv: 2,
            numExperts: 8, topK: 8, vocabSize: 32, maxSeqLen: 64,
            seed: 3040);

        var seenTelemetry = new List<TieredStepTelemetry>();
        var options = new TieredRunnerOptions
        {
            GpuAdapterIndex = -2, // Deterministic CPU tiered MoE execution
            Policy = TieringExecutionPolicy.Auto,
            RouteTailSkipMinRank = 6,
            VramExpertSlotCount = 2,
            MaxSequenceLength = 64,
            OnStepTelemetry = snap => seenTelemetry.Add(snap)
        };

        using var runner = new TieredMoERunner(moePath, options);
        using var dflash = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(runner, dflash, seed: 304);

        const string prompt = "abcdabcdabcd";
        var specResult = await engine.GenerateAsync(
            prompt,
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, specResult.Metrics.GeneratedTokens);
        Assert.Contains(seenTelemetry, t => t.TotalRoutedExperts > 0);
        Assert.Contains(seenTelemetry, t => t.ExpertsSkippedPercent > 0.0);

        // Verify greedy verification batch on TieredMoEMissSplitter directly preserves non-skipped hits/CPU split
        Assert.Contains(runner.TelemetryHistory, t => t.SkippedRoutedExperts > 0 && t.CpuExecutedExperts > 0);
    }

    [Fact]
    public void T3_05_F5_F4_TieredMoERunner_VramReserveZero_ExpandsCacheSlotsAndReducesMisses()
    {
        // Tight VRAM headroom where default ~350MB reserve reduces expert cache slots vs VramReserveMb = 0
        const ulong dedicated = 4UL * 1024 * 1024 * 1024;
        const ulong osBudget = 3800UL * 1024 * 1024;
        const ulong baseUsage = 3200UL * 1024 * 1024;
        const ulong slotBytes = 64UL * 1024 * 1024;

        var snapDefault = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, baseUsage, isWindowsWddm: true, explicitReserveBytes: null);
        var snapZero = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, baseUsage, isWindowsWddm: true, explicitReserveBytes: 0UL);

        int defaultSlots = WddmVramGuard.ComputeRecommendedSlotCount(snapDefault, slotBytes, 0, maxSlotCapacity: 16, minSlotFloor: 0);
        int zeroSlots = WddmVramGuard.ComputeRecommendedSlotCount(snapZero, slotBytes, 0, maxSlotCapacity: 16, minSlotFloor: 0);
        Assert.True(zeroSlots > defaultSlots, $"Expected zeroSlots ({zeroSlots}) > defaultSlots ({defaultSlots}).");

        using var slotsDefault = new GpuResidentSlotManager(Math.Max(1, defaultSlots), 4096, allocateVram: false);
        using var slotsZero = new GpuResidentSlotManager(Math.Max(1, zeroSlots), 4096, allocateVram: false);

        // Pre-populate as many experts as fit in each slot budget
        for (int e = 0; e < defaultSlots; e++) slotsDefault.AllocateOrEvictSlot(0, e, out _, out _, out _, out _);
        for (int e = 0; e < zeroSlots; e++) slotsZero.AllocateOrEvictSlot(0, e, out _, out _, out _, out _);

        int[] batchIndices = Enumerable.Range(0, zeroSlots).ToArray();
        float[] weights = Enumerable.Repeat(1.0f / zeroSlots, zeroSlots).ToArray();

        var splitterDefault = new TieredMoEMissSplitter();
        var resDefault = splitterDefault.PlanLayerBatch(
            0, batchSize: 1, topK: zeroSlots, numExperts: 16, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.Auto, batchIndices, weights,
            new ExpertRoutingProfile(1, 16), slotsDefault, new BandwidthCoScheduler(), null);

        var splitterZero = new TieredMoEMissSplitter();
        var resZero = splitterZero.PlanLayerBatch(
            0, batchSize: 1, topK: zeroSlots, numExperts: 16, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.Auto, batchIndices, weights,
            new ExpertRoutingProfile(1, 16), slotsZero, new BandwidthCoScheduler(), null);

        Assert.True(resZero.VramHitUniqueCount > resDefault.VramHitUniqueCount);
        Assert.Equal(0, resZero.CpuExecuteUniqueCount + resZero.PcieTransferUniqueCount);

        // Also confirm TieredMoERunner wires VramReserveMb = 0 vs null cleanly
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 8, topK: 2, dim: 32, expertFfnDim: 32);
        using var rZero = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(), null, null,
            new TieredRunnerOptions { VramReserveMb = 0 }, moe.LmHead);
        Assert.Equal(0UL, rZero.EffectiveVramReserveBytes);
    }

    [Fact]
    public async Task T3_07_F7_F8_DFlashDraftProvider_WithSpeculativeEngine_OnUdIq2XsAndUdIq3Models()
    {
        using var tmp = new TempDirScope();
        foreach (string profile in new[] { "UD-IQ2_XS", "UD-IQ3" })
        {
            string modelPath = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
                tmp.GetFilePath($"t3_07_{profile}.gguf"),
                profile,
                dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64,
                sharedWeightsSeed: 3700 + profile.Length);

            using var session = new InferenceSession(modelPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu);
            using var dflash = new DFlashDraftProvider(session.HiddenStateSource!, weights: null, new DFlashOptions { BlockSize = 4 });
            using var engine = new SpeculativeEngine(session, dflash, seed: 307);

            const string prompt = "abcdabcdabcd";
            var specRes = await engine.GenerateAsync(
                prompt,
                new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
                formatChat: false);

            Assert.Equal(6, specRes.Metrics.GeneratedTokens);

            // Pure autoregressive greedy baseline on same InferenceSession
            int[] promptTokens = session.Tokenizer.Encode(prompt);
            session.ResetKvCache();
            session.Prefill(promptTokens);
            int tok = ((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy);
            var expected = new List<int> { tok };
            int pos = promptTokens.Length;
            for (int s = 0; s < 5; s++)
            {
                session.ForwardToken(expected[^1], pos++, computeLogits: true);
                expected.Add(((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy));
            }

            string expectedText = string.Concat(expected.Select(t => session.Tokenizer.DecodeToken(t)));
            Assert.Equal(expectedText, specRes.Text);
        }
    }

    [Fact]
    public void T3_08_F8_F9_UdQuantizedModels_VsQ8_0Reference_ViaKlDivergenceVerifier()
    {
        using var tmp = new TempDirScope();
        const int sharedSeed = 3808;

        string refQ8Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("ref_q8.gguf"), "Q8_0",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed, useQ8ReferenceOnly: true);

        string faithfulUdPath = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("faithful_ud_iq4xs.gguf"), "UD-IQ4_XS",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed);

        string degradedUdPath = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("degraded_ud_iq2xs.gguf"), "UD-IQ2_XS",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed, corruptWeights: true);

        var prompts = new[] { "abcdef", "ghijkl" };
        var passRes = KlDivergenceVerifier.VerifyModels(
            refQ8Path, faithfulUdPath, prompts,
            maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu,
            formatChat: false,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 1.0 });

        Assert.True(passRes.Passed, $"Faithful UD model failed KL gate: {passRes.FailureReason}");
        Assert.True(passRes.MeanKlDivergence < 0.05);
        Assert.Equal(1.0, passRes.Top1AgreementRate);

        var failRes = KlDivergenceVerifier.VerifyModels(
            refQ8Path, degradedUdPath, prompts,
            maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu,
            formatChat: false,
            options: new KlDivergenceOptions { Threshold = 0.05, MinTop1Agreement = 0.8 });

        Assert.False(failRes.Passed);
        Assert.True(failRes.MeanKlDivergence > 0.05);
    }

    [Fact]
    public void T3_09_F9_F10_KlDivergenceParity_AnalyticalVectorsMatchQuantSpec()
    {
        // FX-4.1a: Identical logits -> KL = 0.0, Top1Agreement = 1.0
        float[] id = { 2.0f, 1.0f, 0.0f, -1.0f };
        var rIdent = KlDivergenceVerifier.EvaluateSingleTokenLogits(id, id, threshold: 1e-6);
        Assert.True(rIdent.Passed);
        Assert.InRange(rIdent.MeanKlDivergence, 0.0, 1e-9);
        Assert.Equal(1.0, rIdent.Top1AgreementRate);

        // FX-4.1b: Translation invariance (+1000 shift) -> KL = 0.0
        float[] shifted = { 1002.0f, 1001.0f, 1000.0f, 999.0f };
        var rShift = KlDivergenceVerifier.EvaluateSingleTokenLogits(id, shifted, threshold: 1e-6);
        Assert.True(rShift.Passed);
        Assert.InRange(rShift.MeanKlDivergence, 0.0, 1e-6);
        Assert.Equal(1.0, rShift.Top1AgreementRate);

        // FX-4.1c: P=[0.5, 0.25, 0.125, 0.125], Q=[0.25, 0.25, 0.25, 0.25] => 0.25 * ln(2) = 0.173286795
        float[] pLogits = { MathF.Log(4f), MathF.Log(2f), 0f, 0f };
        float[] qLogits = { 0f, 0f, 0f, 0f };
        var rExactLogits = KlDivergenceVerifier.EvaluateSingleTokenLogits(pLogits, qLogits, threshold: 0.20);
        Assert.True(rExactLogits.Passed);
        Assert.True(Math.Abs(rExactLogits.MeanKlDivergence - 0.173286795) < 1e-5,
            $"Expected 0.173286795, got {rExactLogits.MeanKlDivergence}");

        float[] pProbs = { 0.5f, 0.25f, 0.125f, 0.125f };
        float[] qProbs = { 0.25f, 0.25f, 0.25f, 0.25f };
        var rExactProbs = KlDivergenceVerifier.EvaluateProbabilities(pProbs, qProbs, vocabSize: 4, threshold: 0.20);
        Assert.True(Math.Abs(rExactProbs.MeanKlDivergence - rExactLogits.MeanKlDivergence) < 1e-6);

        // FX-4.1d: Top-1 flip -> KL ~ 4.89, Top1Agreement = 0.0, Passed = false
        float[] pFlip = { 5.0f, 0.0f, 0.0f, 0.0f };
        float[] qFlip = { 0.0f, 5.0f, 0.0f, 0.0f };
        var rFlip = KlDivergenceVerifier.EvaluateSingleTokenLogits(pFlip, qFlip, threshold: 0.5);
        Assert.False(rFlip.Passed);
        Assert.Equal(0.0, rFlip.Top1AgreementRate);
        Assert.InRange(rFlip.MeanKlDivergence, 4.5, 5.2);

        // Asymmetric KL guard against reversed D_KL(Q || P):
        // P = [0.80, 0.15, 0.04, 0.01], Q = [0.25, 0.25, 0.25, 0.25] => KL(P||Q) = 0.74840479 != KL(Q||P) = 1.09978303
        float[] asymP = { 0.80f, 0.15f, 0.04f, 0.01f };
        float[] asymQ = { 0.25f, 0.25f, 0.25f, 0.25f };
        double expectedAsymKl =
            0.80 * Math.Log(0.80 / 0.25) +
            0.15 * Math.Log(0.15 / 0.25) +
            0.04 * Math.Log(0.04 / 0.25) +
            0.01 * Math.Log(0.01 / 0.25);

        var rAsymProbs = KlDivergenceVerifier.EvaluateProbabilities(asymP, asymQ, vocabSize: 4, threshold: 0.80);
        Assert.True(rAsymProbs.Passed);
        Assert.True(Math.Abs(rAsymProbs.MeanKlDivergence - expectedAsymKl) < 1e-4,
            $"Asymmetric prob KL(P||Q) mismatch: expected={expectedAsymKl:F6}, actual={rAsymProbs.MeanKlDivergence:F6}");

        float[] asymPLogits = { MathF.Log(0.80f), MathF.Log(0.15f), MathF.Log(0.04f), MathF.Log(0.01f) };
        float[] asymQLogits = { 0f, 0f, 0f, 0f };
        var rAsymLogits = KlDivergenceVerifier.EvaluateSingleTokenLogits(asymPLogits, asymQLogits, threshold: 0.80);
        Assert.True(Math.Abs(rAsymLogits.MeanKlDivergence - expectedAsymKl) < 1e-4,
            $"Asymmetric logit KL(P||Q) mismatch: expected={expectedAsymKl:F6}, actual={rAsymLogits.MeanKlDivergence:F6}");
    }

    [Fact]
    public void T3_10_F1_F9_Q4KvSession_VsFp16KvSession_ViaKlDivergenceVerifier()
    {
        using var tmp = new TempDirScope();
        string modelPath = UpgradeScenarioFixtures.BuildTinyDenseWithClearMargins(
            tmp.GetFilePath("t3_10_kv_kl.gguf"),
            dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            seed: 3100);

        using var fp16Session = new InferenceSession(modelPath, maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Fp16);
        using var q4Session = new InferenceSession(modelPath, maxSeqLen: 32, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Q4_0);

        int[] tokens = { 2, 4, 7, 10, 15 };
        var batchRes = KlDivergenceVerifier.VerifyTargets(
            fp16Session, q4Session, tokens,
            new KlDivergenceOptions { Threshold = 0.25, MinTop1Agreement = 1.0, UseBatchLogitVerification = true });

        Assert.True(batchRes.Passed, $"Batch KL check failed: {batchRes.FailureReason}");
        Assert.Equal(1.0, batchRes.Top1AgreementRate);
        Assert.True(batchRes.MeanKlDivergence < 0.25);

        var seqRes = KlDivergenceVerifier.VerifyTargets(
            fp16Session, q4Session, tokens,
            new KlDivergenceOptions { Threshold = 0.25, MinTop1Agreement = 1.0, UseBatchLogitVerification = false });

        Assert.True(seqRes.Passed);
        Assert.True(Math.Abs(batchRes.MeanKlDivergence - seqRes.MeanKlDivergence) < 1e-4);
    }

    [Fact]
    public void T3_15_F5_F6_F9_TieredMoERunner_TailSkipVsFullReference_ViaKlDivergenceVerifier()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 2, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16, seed: 3150);

        // Zero out router bias / skew or keep small expert weights -> rank-3 tail expert has small weight, preserving top-1
        using var baseline = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(2, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null,
            options: new TieredRunnerOptions { RouteTailSkipMinRank = 0 }, lmHeadWeights: moe.LmHead);

        using var tailSkipped = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(2, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null,
            options: new TieredRunnerOptions { RouteTailSkipMinRank = 3 }, lmHeadWeights: moe.LmHead);

        // Identical token vectors across batch guarantee unanimous tail rank >= 3
        float[] single = Fx1Vectors.GenerateRampVector(32, -0.15f, 0.01f);
        float[] inputs = new float[4 * 32];
        for (int t = 0; t < 4; t++) single.CopyTo(inputs, t * 32);

        var klRes = KlDivergenceVerifier.VerifySyntheticTieredMoE(
            baseline, tailSkipped, inputs, batchSize: 4, isSpeculativeDraftBatch: true,
            new KlDivergenceOptions { Threshold = 0.50, MinTop1Agreement = 1.0 });

        Assert.True(klRes.Passed, $"Tail-skip MoE KL gate failed: {klRes.FailureReason}");
        Assert.Equal(1.0, klRes.Top1AgreementRate);
        Assert.NotNull(tailSkipped.LastStepTelemetry);
        Assert.True(tailSkipped.LastStepTelemetry!.ExpertsSkippedPercent > 0.0);
    }

    [Fact]
    public async Task T3_18_F7_F6_F1_FullInferenceStack_Q4Kv_TailSkip_DFlash()
    {
        using var tmp = new TempDirScope();
        string moePath = UpgradeScenarioFixtures.BuildTinyQwen2MoeModel(
            tmp.GetFilePath("t3_18_fullstack.gguf"),
            dim: 64, ffnDim: 64, expertFfnDim: 64,
            nLayers: 2, nHeads: 4, nHeadsKv: 2,
            numExperts: 8, topK: 8, vocabSize: 32, maxSeqLen: 64,
            seed: 3180);

        var opts = new TieredRunnerOptions
        {
            GpuAdapterIndex = -2, // CPU path exercises KVCache Q4_0 + TieredMoE + RouteTailSkipMinRank
            KvPrecision = KvCachePrecision.Q4_0,
            VramReserveMb = 0,
            RouteTailSkipMinRank = 6,
            MaxSequenceLength = 64
        };

        using var runner = new TieredMoERunner(moePath, opts);
        Assert.Equal(KvCachePrecision.Q4_0, runner.KvPrecision);
        Assert.Equal(0UL, runner.EffectiveVramReserveBytes);

        using var dflash = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(runner, dflash, seed: 318);

        var res = await engine.GenerateAsync(
            "abcdabcdabcd",
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, res.Metrics.GeneratedTokens);
        Assert.Contains(runner.TelemetryHistory, t => t.ExpertsSkippedPercent > 0.0);
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier3")]
[Trait("Hardware", "Gpu")]
public sealed unsafe class Tier3CrossFeatureGpuE2ETests
{
    [UpgradeCudaFact]
    public async Task T3_02_F2_F4_F7_CudaQ4Kv_VramReserveZero_WithDFlashSpeculativeDecoding()
    {
        var nv = DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor == GpuVendor.Nvidia && d.SupportedEngines.Contains(InferenceEngineType.BareMetal));
        Assert.NotNull(nv);

        using var tmp = new TempDirScope();
        string modelPath = UpgradeScenarioFixtures.BuildTinyDenseWithClearMargins(
            tmp.GetFilePath("t3_02_cuda_dflash.gguf"),
            dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64,
            seed: 3020);

        using var session = new InferenceSession(
            modelPath, maxSeqLen: 64, device: nv!.Id, engine: InferenceEngineType.BareMetal,
            kvPrecision: KvCachePrecision.Q4_0, vramReserveMb: 0);

        Assert.Equal("Cuda", session.ActiveBackend);
        Assert.Equal(KvCachePrecision.Q4_0, session.KvPrecision);
        Assert.Equal(0UL, WddmVramGuard.ResolveExplicitReserveBytes(0));

        using var dflash = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(session, dflash, seed: 302);

        const string prompt = "abcdabcdabcd";
        var specRes = await engine.GenerateAsync(
            prompt,
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, specRes.Metrics.GeneratedTokens);

        // Greedy baseline on same CUDA session
        int[] promptTokens = session.Tokenizer.Encode(prompt);
        session.ResetKvCache();
        session.Prefill(promptTokens);
        int tok = ((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy);
        var expected = new List<int> { tok };
        int pos = promptTokens.Length;
        for (int s = 0; s < 5; s++)
        {
            session.ForwardToken(expected[^1], pos++, computeLogits: true);
            expected.Add(((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy));
        }

        string expectedText = string.Concat(expected.Select(t => session.Tokenizer.DecodeToken(t)));
        Assert.Equal(expectedText, specRes.Text);
    }

    [UpgradeD3D12Fact]
    public void T3_03_F3_F4_F8_D3D12Q4Kv_VramReserveZero_WithMixedTensorUdIq4XsGguf()
    {
        using var tmp = new TempDirScope();
        string modelPath = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("t3_03_d3d12_ud_iq4xs.gguf"),
            "UD-IQ4_XS",
            dim: 256, ffnDim: 256, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: 3303);

        using var gguf = GgufFile.Open(modelPath);
        using var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var d3dModel = new Qwen2D3D12Model(
            ctx, weights, maxSeqLen: 32,
            kvPrecision: KvCachePrecision.Q4_0,
            explicitVramReserveBytes: 0UL);

        Assert.Equal(KvCachePrecision.Q4_0, d3dModel.KvPrecision);
        Assert.Equal(0UL, d3dModel.EffectiveVramReserveBytes);

        using var cpuKv = new KVCache(2, 2, 64, maxSeqLen: 32, precision: KvCachePrecision.Q4_0);
        using var cpuModel = new Qwen2Model(weights, maxSeqLen: 32);

        float[] d3dLogits = new float[weights.VocabSize];
        float[] cpuLogits = new float[weights.VocabSize];

        int[] seq = { 2, 5, 9 };
        for (int pos = 0; pos < seq.Length; pos++)
        {
            bool compute = pos == seq.Length - 1;
            d3dModel.Forward(seq[pos], pos, d3dLogits, computeLogits: compute);
            cpuModel.Forward(seq[pos], pos, cpuKv, cpuLogits, computeLogits: compute);
        }

        Assert.All(d3dLogits, x => Assert.True(float.IsFinite(x)));
        float cos = Fx1Vectors.CosineSimilarity(cpuLogits, d3dLogits);
        Assert.True(cos > 0.96f, $"D3D12 UD-IQ4_XS Q4_0 vs CPU cosine {cos:F4} <= 0.96");
    }

    [UpgradeCudaFact]
    public void T3_06_F1_F2_F3_CrossBackend4BitKvParity_CpuVsCudaVsD3D12()
    {
        const int nHeadsQ = 2;
        const int nHeadsKv = 2;
        const int headDim = 64;
        const int maxSeqLen = 16;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        // FX-1.2a exact orthogonal basis: t=0 has K=[+1,0..], V=[+2,+2..]; t=1 has K=[0,+1..], V=[-1,-1..]; Q=[+10,0..]
        float[] k0 = new float[nHeadsKv * headDim];
        float[] v0 = new float[nHeadsKv * headDim];
        float[] k1 = new float[nHeadsKv * headDim];
        float[] v1 = new float[nHeadsKv * headDim];
        float[] q = new float[nHeadsQ * headDim];
        for (int h = 0; h < nHeadsKv; h++)
        {
            k0[h * headDim + 0] = 1.0f;
            k1[h * headDim + 1] = 1.0f;
            for (int d = 0; d < headDim; d++)
            {
                v0[h * headDim + d] = 2.0f;
                v1[h * headDim + d] = -1.0f;
            }
        }
        for (int h = 0; h < nHeadsQ; h++) q[h * headDim + 0] = 100.0f;

        // 1. CPU KVCache Q4_0
        using var cpuCache = new KVCache(1, nHeadsKv, headDim, maxSeqLen, precision: KvCachePrecision.Q4_0);
        cpuCache.Store(0, 0, k0, v0);
        cpuCache.Store(0, 1, k1, v1);

        float[] cpuOut = new float[nHeadsQ * headDim];
        fixed (float* pQ = q, pCpuOut = cpuOut)
        {
            for (int h = 0; h < nHeadsQ; h++)
            {
                float s0 = cpuCache.DotKey(0, h, 0, pQ + h * headDim) * attnScale;
                float s1 = cpuCache.DotKey(0, h, 1, pQ + h * headDim) * attnScale;
                float m = MathF.Max(s0, s1);
                float e0 = MathF.Exp(s0 - m);
                float e1 = MathF.Exp(s1 - m);
                float inv = 1.0f / (e0 + e1);
                cpuCache.AccumulateValue(0, h, 0, e0 * inv, pCpuOut + h * headDim);
                cpuCache.AccumulateValue(0, h, 1, e1 * inv, pCpuOut + h * headDim);
            }
        }

        // 2. CUDA Q4_0
        float[] cudaOut = new float[nHeadsQ * headDim];
        using (var gpu = new GpuContext(0))
        {
            gpu.MakeCurrent();
            byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
            CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
            try
            {
                CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStore, module, "kv_cache_store_q4_0"), "GetStore");
                CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttn, module, "attention_gqa_q4_0"), "GetAttn");

                int blocksPerHead = (headDim + 31) / 32;
                nuint kvBytes = (nuint)((long)nHeadsKv * maxSeqLen * blocksPerHead * sizeof(BlockQ4_0));
                nuint kvVecBytes = (nuint)(nHeadsKv * headDim * sizeof(float));
                nuint qOutBytes = (nuint)(nHeadsQ * headDim * sizeof(float));

                IntPtr dKC = gpu.AllocateDevice(kvBytes);
                IntPtr dVC = gpu.AllocateDevice(kvBytes);
                IntPtr dK = gpu.AllocateDevice(kvVecBytes);
                IntPtr dV = gpu.AllocateDevice(kvVecBytes);
                IntPtr dQ = gpu.AllocateDevice(qOutBytes);
                IntPtr dOut = gpu.AllocateDevice(qOutBytes);
                IntPtr dScores = gpu.AllocateDevice((nuint)(nHeadsQ * maxSeqLen * sizeof(float)));

                try
                {
                    CuDriver.Check(CuDriver.MemsetD8(dKC, 0, kvBytes), "ZeroK");
                    CuDriver.Check(CuDriver.MemsetD8(dVC, 0, kvBytes), "ZeroV");

                    void** pStore = stackalloc void*[8];
                    for (int pos = 0; pos < 2; pos++)
                    {
                        float[] kVec = pos == 0 ? k0 : k1;
                        float[] vVec = pos == 0 ? v0 : v1;
                        fixed (float* pK = kVec, pV = vVec)
                        {
                            gpu.CopyToDevice(dK, (IntPtr)pK, kvVecBytes);
                            gpu.CopyToDevice(dV, (IntPtr)pV, kvVecBytes);
                        }
                        int nhKv = nHeadsKv, hd = headDim, ms = maxSeqLen, p = pos;
                        pStore[0] = &dKC; pStore[1] = &dVC; pStore[2] = &dK; pStore[3] = &dV;
                        pStore[4] = &nhKv; pStore[5] = &hd; pStore[6] = &ms; pStore[7] = &p;
                        CuDriver.Check(CuDriver.LaunchKernel(fnStore, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pStore, IntPtr.Zero), "Store");
                    }

                    fixed (float* pQ = q) gpu.CopyToDevice(dQ, (IntPtr)pQ, qOutBytes);
                    int nhQ = nHeadsQ, nhKv2 = nHeadsKv, hd2 = headDim, ms2 = maxSeqLen, lastPos = 1;
                    float sc = attnScale;
                    void** pAttn = stackalloc void*[11];
                    pAttn[0] = &dQ; pAttn[1] = &dKC; pAttn[2] = &dVC; pAttn[3] = &dOut; pAttn[4] = &dScores;
                    pAttn[5] = &nhQ; pAttn[6] = &nhKv2; pAttn[7] = &hd2; pAttn[8] = &ms2; pAttn[9] = &lastPos; pAttn[10] = &sc;
                    CuDriver.Check(CuDriver.LaunchKernel(fnAttn, (uint)nHeadsQ, 1, 1, 64, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttn, IntPtr.Zero), "Attn");
                    gpu.Synchronize();

                    fixed (float* pOut = cudaOut) gpu.CopyToHost((IntPtr)pOut, dOut, qOutBytes);
                }
                finally
                {
                    gpu.FreeDevice(dKC); gpu.FreeDevice(dVC); gpu.FreeDevice(dK); gpu.FreeDevice(dV);
                    gpu.FreeDevice(dQ); gpu.FreeDevice(dOut); gpu.FreeDevice(dScores);
                }
            }
            finally
            {
                CuDriver.ModuleUnload(module);
            }
        }

        // 3. D3D12 Q4_0
        float[] d3dOut = new float[nHeadsQ * headDim];
        if (OperatingSystem.IsWindows() && D3D12Context.IsSupported)
        {
            using var ctx = new D3D12Context();
            ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeadsKv, headDim, maxSeqLen, KvCachePrecision.Q4_0);
            ulong kvVecBytes = (ulong)(nHeadsKv * headDim * sizeof(float));
            ulong qOutBytes = (ulong)(nHeadsQ * headDim * sizeof(float));

            var storeParams = new RootParameter[]
            {
                new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigStore = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeParams));
            using var psoStore = ctx.CreatePipelineState(sigStore, ctx.CompileShader(D3D12Shaders.KvCacheStoreQ4_0));

            var attnParams = new RootParameter[]
            {
                new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigAttn = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
            using var psoAttn = ctx.CreatePipelineState(sigAttn, ctx.CompileShader(D3D12Shaders.AttentionGqaQ4_0));

            using var dKC = ctx.CreateDeviceBuffer(kvBytes);
            using var dVC = ctx.CreateDeviceBuffer(kvBytes);
            using var dK = ctx.CreateDeviceBuffer(kvVecBytes);
            using var dV = ctx.CreateDeviceBuffer(kvVecBytes);
            using var dQ = ctx.CreateDeviceBuffer(qOutBytes);
            using var dOut = ctx.CreateDeviceBuffer(qOutBytes);

            for (int pos = 0; pos < 2; pos++)
            {
                float[] kVec = pos == 0 ? k0 : k1;
                float[] vVec = pos == 0 ? v0 : v1;
                fixed (float* pK = kVec, pV = vVec)
                {
                    ctx.CopyToDevice(dK, (IntPtr)pK, kvVecBytes);
                    ctx.CopyToDevice(dV, (IntPtr)pV, kvVecBytes);
                }

                ctx.BeginCommands();
                var cmd = ctx.CommandList;
                cmd.ResourceBarrierTransition(dK, ResourceStates.Common, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(dV, ResourceStates.Common, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(dKC, ResourceStates.Common, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(dVC, ResourceStates.Common, ResourceStates.UnorderedAccess);

                cmd.SetComputeRootSignature(sigStore);
                cmd.SetPipelineState(psoStore);
                cmd.SetComputeRoot32BitConstant(0, (uint)nHeadsKv, 0);
                cmd.SetComputeRoot32BitConstant(0, (uint)headDim, 1);
                cmd.SetComputeRoot32BitConstant(0, (uint)maxSeqLen, 2);
                cmd.SetComputeRoot32BitConstant(0, (uint)pos, 3);
                cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(3, dKC.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(4, dVC.GPUVirtualAddress);
                cmd.Dispatch((uint)nHeadsKv, 1, 1);

                cmd.ResourceBarrierTransition(dK, ResourceStates.UnorderedAccess, ResourceStates.Common);
                cmd.ResourceBarrierTransition(dV, ResourceStates.UnorderedAccess, ResourceStates.Common);
                cmd.ResourceBarrierTransition(dKC, ResourceStates.UnorderedAccess, ResourceStates.Common);
                cmd.ResourceBarrierTransition(dVC, ResourceStates.UnorderedAccess, ResourceStates.Common);
                ctx.EndCommandsAndExecute();
                ctx.Synchronize();
            }

            fixed (float* pQ = q) ctx.CopyToDevice(dQ, (IntPtr)pQ, qOutBytes);
            ctx.BeginCommands();
            var cmdA = ctx.CommandList;
            cmdA.ResourceBarrierTransition(dQ, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmdA.ResourceBarrierTransition(dKC, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmdA.ResourceBarrierTransition(dVC, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmdA.ResourceBarrierTransition(dOut, ResourceStates.Common, ResourceStates.UnorderedAccess);

            cmdA.SetComputeRootSignature(sigAttn);
            cmdA.SetPipelineState(psoAttn);
            cmdA.SetComputeRoot32BitConstant(0, (uint)nHeadsQ, 0);
            cmdA.SetComputeRoot32BitConstant(0, (uint)nHeadsKv, 1);
            cmdA.SetComputeRoot32BitConstant(0, (uint)headDim, 2);
            cmdA.SetComputeRoot32BitConstant(0, (uint)maxSeqLen, 3);
            cmdA.SetComputeRoot32BitConstant(0, 1u, 4);
            cmdA.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(attnScale), 5);
            cmdA.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
            cmdA.SetComputeRootUnorderedAccessView(2, dKC.GPUVirtualAddress);
            cmdA.SetComputeRootUnorderedAccessView(3, dVC.GPUVirtualAddress);
            cmdA.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
            cmdA.Dispatch((uint)nHeadsQ, 1, 1);

            cmdA.ResourceBarrierTransition(dQ, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmdA.ResourceBarrierTransition(dKC, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmdA.ResourceBarrierTransition(dVC, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmdA.ResourceBarrierTransition(dOut, ResourceStates.UnorderedAccess, ResourceStates.Common);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            fixed (float* pOut = d3dOut) ctx.CopyToHost((IntPtr)pOut, dOut, qOutBytes);
        }

        // Verify all 3 match V[0] = +2.0 and match each other within < 1e-2 L2
        for (int i = 0; i < cpuOut.Length; i++)
        {
            Assert.InRange(cpuOut[i], 1.75f, 2.05f);
            Assert.True(MathF.Abs(cpuOut[i] - cudaOut[i]) < 1e-2f, $"CPU vs CUDA diff at {i}: {cpuOut[i]} vs {cudaOut[i]}");
            if (OperatingSystem.IsWindows() && D3D12Context.IsSupported)
            {
                Assert.True(MathF.Abs(cpuOut[i] - d3dOut[i]) < 1e-2f, $"CPU vs D3D12 diff at {i}: {cpuOut[i]} vs {d3dOut[i]}");
            }
        }
    }
}
