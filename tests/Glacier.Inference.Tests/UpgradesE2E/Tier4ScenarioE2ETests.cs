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
using Glacier.Inference.Sampling;
using Glacier.Inference.Scheduling;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier4")]
public sealed class Tier4ScenarioE2ETests
{
    [Fact]
    public void T4_02_UnslothDynamic31_QualificationPipeline_Iq4XsPasses_DegradedIq2XsFails()
    {
        using var tmp = new TempDirScope();
        const int sharedSeed = 4020;

        string refQ8Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("t4_02_ref_q8.gguf"), "Q8_0",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed, useQ8ReferenceOnly: true);

        string udIq4Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("t4_02_ud_iq4xs.gguf"), "UD-IQ4_XS",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed);

        string udIq3Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("t4_02_ud_iq3.gguf"), "UD-IQ3",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed);

        string degradedIq2Path = UpgradeScenarioFixtures.BuildMixedUdQwen2Model(
            tmp.GetFilePath("t4_02_bad_iq2xs.gguf"), "UD-IQ2_XS",
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32,
            sharedWeightsSeed: sharedSeed, corruptWeights: true);

        var prompts = new[] { "abcdef", "ghijkl", "mnopqr" };
        var gateOpts = new KlDivergenceOptions { Threshold = 0.08, MinTop1Agreement = 1.0 };

        var iq4Gate = KlDivergenceVerifier.VerifyModels(
            refQ8Path, udIq4Path, prompts, maxSeqLen: 32, device: "cpu",
            engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Q4_0,
            formatChat: false, options: gateOpts);
        Assert.True(iq4Gate.Passed, $"UD-IQ4_XS failed qualification gate: {iq4Gate.FailureReason}");
        Assert.Equal(1.0, iq4Gate.Top1AgreementRate);

        var iq3Gate = KlDivergenceVerifier.VerifyModels(
            refQ8Path, udIq3Path, prompts, maxSeqLen: 32, device: "cpu",
            engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Q4_0,
            formatChat: false, options: gateOpts);
        Assert.True(iq3Gate.Passed, $"UD-IQ3 failed qualification gate: {iq3Gate.FailureReason}");

        var degradedGate = KlDivergenceVerifier.VerifyModels(
            refQ8Path, degradedIq2Path, prompts, maxSeqLen: 32, device: "cpu",
            engine: InferenceEngineType.Cpu, kvPrecision: KvCachePrecision.Q4_0,
            formatChat: false, options: gateOpts);
        Assert.False(degradedGate.Passed);
        Assert.NotNull(degradedGate.FailureReason);
    }

    [Fact]
    public void T4_04_PreReleaseQuantizationQualityGate_InferenceArm_CandidatePassAndFail()
    {
        const int vocab = 4;

        // Reference Q8_0 / FP32 baseline distributions (FX-4.1)
        float[] refBatch =
        {
            2.0f, 1.0f, 0.0f, -1.0f,
            MathF.Log(4f), MathF.Log(2f), 0f, 0f,
            5.0f, 0.0f, 0.0f, 0.0f
        };

        // Faithful candidate: slight quantization perturbation, preserves top-1 everywhere
        float[] faithfulCand =
        {
            2.02f, 0.98f, 0.01f, -0.99f,
            MathF.Log(4f) - 0.02f, MathF.Log(2f) + 0.01f, 0.0f, 0.0f,
            4.95f, 0.05f, 0.0f, 0.0f
        };

        // Regressed candidate: flips top-1 on token 2 (FX-4.1d)
        float[] regressedCand =
        {
            2.0f, 1.0f, 0.0f, -1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 5.0f, 0.0f, 0.0f
        };

        var gateOpts = new KlDivergenceOptions { Threshold = 0.05, MaxKlThreshold = 0.15, MinTop1Agreement = 1.0 };

        var faithfulReport = KlDivergenceVerifier.EvaluateLogits(refBatch, faithfulCand, vocab, gateOpts);
        Assert.True(faithfulReport.Passed, $"Faithful candidate failed: {faithfulReport.FailureReason}");
        Assert.Equal(1.0, faithfulReport.Top1AgreementRate);
        Assert.True(faithfulReport.MeanKlDivergence < 0.01);
        Assert.True(faithfulReport.MaxKlDivergence >= faithfulReport.MeanKlDivergence);

        var regressedReport = KlDivergenceVerifier.EvaluateLogits(refBatch, regressedCand, vocab, gateOpts);
        Assert.False(regressedReport.Passed);
        Assert.True(regressedReport.Top1AgreementRate < 1.0);
        Assert.True(regressedReport.MaxKlDivergence > 4.5);
        Assert.Contains("KL", regressedReport.FailureReason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void T4_07_DynamicBandwidthShiftUnderLoad_10MoEBatches_RebalancesCpuPcieWithTailSkip()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 2, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16, seed: 4070);
        using var slots = new GpuResidentSlotManager(slotCount: 3, slotByteSize: 4096, allocateVram: false);
        slots.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);

        var profile = new ExpertRoutingProfile(2, 8);
        var coScheduler = new BandwidthCoScheduler(initialPcieGBps: 8.0, initialCpuGBps: 65.0, vramGBps: 272.0, ewmaAlpha: 0.65);

        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, profile, coScheduler, slots, hostArena: null,
            options: new TieredRunnerOptions { Policy = TieringExecutionPolicy.Auto, RouteTailSkipMinRank = 2 },
            lmHeadWeights: moe.LmHead);

        float[] single = Fx1Vectors.GenerateRampVector(32, -0.1f, 0.01f);
        float[] batchInput = new float[4 * 32];
        for (int t = 0; t < 4; t++) single.CopyTo(batchInput, t * 32);
        float[] logits = new float[4 * 16];

        // Phase 1 (batches 0..4): CPU-favored (fast CPU, congested PCIe)
        for (int b = 0; b < 5; b++)
        {
            coScheduler.RecordCpuExecution(512L * 1024 * 1024, elapsedMilliseconds: 2.0);
            coScheduler.RecordPcieTransfer(8L * 1024 * 1024, elapsedMilliseconds: 10.0);
            runner.EvaluateSyntheticBatch(batchInput, batchSize: 4, isSpeculativeDraftBatch: true, logits, primaryTokenId: b);
            Assert.All(logits, x => Assert.True(float.IsFinite(x)));
        }

        int phase1QStar = coScheduler.CalculateOptimalGpuExpertCount(missingCount: 6, residentHitCount: 1);

        // Phase 2 (batches 5..9): PCIe-favored (fast PCIe, throttled CPU)
        for (int b = 5; b < 10; b++)
        {
            coScheduler.RecordPcieTransfer(512L * 1024 * 1024, elapsedMilliseconds: 2.0);
            coScheduler.RecordCpuExecution(8L * 1024 * 1024, elapsedMilliseconds: 15.0);
            runner.EvaluateSyntheticBatch(batchInput, batchSize: 4, isSpeculativeDraftBatch: true, logits, primaryTokenId: b);
            Assert.All(logits, x => Assert.True(float.IsFinite(x)));
        }

        int phase2QStar = coScheduler.CalculateOptimalGpuExpertCount(missingCount: 6, residentHitCount: 1);

        Assert.Equal(10, runner.TelemetryHistory.Count);
        Assert.True(phase2QStar > phase1QStar, $"Expected PCIe q* to grow after load shift: phase1={phase1QStar}, phase2={phase2QStar}");
        Assert.All(runner.TelemetryHistory, t =>
        {
            Assert.True(double.IsFinite(t.ExpertsSkippedPercent));
            Assert.True(t.ExpertsSkippedPercent > 0.0);
        });
    }

    [Fact]
    public async Task T4_08_HighThroughputDFlash_With4BitKvContextExtension_AndVramReserveZero()
    {
        const int layers = 2;
        const int headsKv = 2;
        const int headDim = 32;
        const int targetSeqLen = 128;

        long fp16KvBytes = layers * 2L * headsKv * targetSeqLen * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Fp16);
        long q4KvBytes = layers * 2L * headsKv * targetSeqLen * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Q4_0);
        long tightBudget = fp16KvBytes / 2; // Insufficient for Fp16, plenty for Q4_0
        Assert.True(q4KvBytes < tightBudget && fp16KvBytes > tightBudget);

        // Verify WddmVramGuard reclaims full headroom with VramReserveMb = 0
        var zeroHeadroom = WddmVramGuard.EvaluateHeadroom(
            dedicatedVramBytes: 2UL * 1024 * 1024 * 1024,
            osBudgetBytes: 1800UL * 1024 * 1024,
            currentUsageBytes: 1500UL * 1024 * 1024,
            isWindowsWddm: true,
            explicitReserveBytes: 0UL);
        Assert.Equal(0UL, zeroHeadroom.SafetyReserveBytes);

        using var tmp = new TempDirScope();
        string modelPath = UpgradeScenarioFixtures.BuildTinyDenseWithClearMargins(
            tmp.GetFilePath("t4_08_long_dflash.gguf"),
            dim: 64, ffnDim: 128, nLayers: layers, nHeads: 2, nHeadsKv: headsKv, vocabSize: 32, maxSeqLen: targetSeqLen,
            seed: 4080);

        using var session = new InferenceSession(
            modelPath, maxSeqLen: targetSeqLen, device: "cpu", engine: InferenceEngineType.Cpu,
            kvPrecision: KvCachePrecision.Q4_0, vramReserveMb: 0);

        Assert.Equal(KvCachePrecision.Q4_0, session.KvPrecision);
        Assert.True(session.KVCache!.TotalBytes < tightBudget);

        using var dflash = new DFlashDraftProvider(session.HiddenStateSource!, weights: null, new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(session, dflash, seed: 408);

        string longPrompt = string.Concat(Enumerable.Repeat("abcd", 16)); // 64-token prompt + 12 generated tokens
        var specRes = await engine.GenerateAsync(
            longPrompt,
            new SpeculativeOptions { MaxTokens = 12, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(12, specRes.Metrics.GeneratedTokens);

        int[] promptTokens = session.Tokenizer.Encode(longPrompt);
        session.ResetKvCache();
        session.Prefill(promptTokens);
        int tok = ((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy);
        var expected = new List<int> { tok };
        int pos = promptTokens.Length;
        for (int s = 0; s < 11; s++)
        {
            session.ForwardToken(expected[^1], pos++, computeLogits: true);
            expected.Add(((ISpeculativeTarget)session).SampleNextToken(SamplingOptions.Greedy));
        }

        string expectedText = string.Concat(expected.Select(t => session.Tokenizer.DecodeToken(t)));
        Assert.Equal(expectedText, specRes.Text);
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier4")]
[Trait("Hardware", "Gpu")]
public sealed class Tier4ScenarioGpuE2ETests
{
    [UpgradeD3D12Fact]
    public async Task T4_01_ConstrainedVram_SpeculativeMoE_Q4Kv_ZeroReserve_Split_TailSkip_DFlash()
    {
        using var tmp = new TempDirScope();
        string moePath = UpgradeScenarioFixtures.BuildTinyQwen2MoeModel(
            tmp.GetFilePath("t4_01_d3d12_moe.gguf"),
            dim: 64, ffnDim: 64, expertFfnDim: 64,
            nLayers: 2, nHeads: 4, nHeadsKv: 2,
            numExperts: 8, topK: 8, vocabSize: 32, maxSeqLen: 64,
            seed: 4010);

        var opts = new TieredRunnerOptions
        {
            Policy = TieringExecutionPolicy.Auto,
            KvPrecision = KvCachePrecision.Q4_0,
            VramReserveMb = 0,
            VramExpertSlotCount = 2,
            RouteTailSkipMinRank = 6,
            MaxSequenceLength = 64
        };

        using var runner = new TieredMoERunner(moePath, opts);
        Assert.Equal(0UL, runner.EffectiveVramReserveBytes);
        Assert.Equal(KvCachePrecision.Q4_0, runner.KvPrecision);

        using var dflash = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(runner, dflash, seed: 401);

        const string prompt = "abcdabcdabcd";
        var res = await engine.GenerateAsync(
            prompt,
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, res.Metrics.GeneratedTokens);
        Assert.Contains(runner.TelemetryHistory, t => t.ExpertsSkippedPercent > 0.0);
        Assert.Contains(runner.TelemetryHistory, t => (t.VramHitExperts + t.PcieTransferredExperts + t.CpuExecutedExperts) > 0);

        // Compare greedy output against deterministic autoregressive reference decode on the same runner
        int[] promptTokens = runner.Tokenizer.Encode(prompt);
        runner.ResetKvCache();
        runner.Prefill(promptTokens);
        int firstTok = runner.SampleNextToken(SamplingOptions.Greedy);
        var refTokens = new List<int> { firstTok };
        int pos = promptTokens.Length;
        for (int s = 0; s < 5; s++)
        {
            runner.ForwardToken(refTokens[^1], pos++, computeLogits: true);
            refTokens.Add(runner.SampleNextToken(SamplingOptions.Greedy));
        }

        string refText = string.Concat(refTokens.Select(t => runner.Tokenizer.DecodeToken(t)));
        Assert.Equal(refText, res.Text);
    }

    [UpgradeCudaFact]
    public void T4_06_MultiBackend4BitKvPortability_CpuCudaD3D12_CapacityAndAccuracyParity()
    {
        using var tmp = new TempDirScope();
        string modelPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("t4_06_portability.gguf"),
            dim: 128, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64,
            seed: 4060);

        using var gguf = GgufFile.Open(modelPath);
        using var weights = new ModelWeights(gguf);

        int[] prompt = Enumerable.Range(0, 20).Select(i => 2 + (i % 10)).ToArray();

        // 1. CPU Q4_0 vs Fp16 capacity & accuracy (headDim = 32 -> 18 B vs 64 B = 0.28125x)
        using var cpuQ4Cache = new KVCache(2, 2, weights.HeadDim, 64, precision: KvCachePrecision.Q4_0);
        using var cpuF16Cache = new KVCache(2, 2, weights.HeadDim, 64, precision: KvCachePrecision.Fp16);
        Assert.True((double)cpuQ4Cache.TotalBytes / cpuF16Cache.TotalBytes < 0.35);

        using var cpuModel = new Qwen2Model(weights, maxSeqLen: 64);
        float[] cpuLogits = new float[weights.VocabSize];
        for (int p = 0; p < prompt.Length; p++)
        {
            cpuModel.Forward(prompt[p], p, cpuQ4Cache, cpuLogits, computeLogits: p == prompt.Length - 1);
        }
        int cpuTop1 = KlDivergenceVerifier.ArgMax(cpuLogits);

        // 2. CUDA Q4_0 vs Fp16 capacity & accuracy
        using var gpu = new GpuContext(0);
        using var cudaQ4 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Q4_0);
        using var cudaF16 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16);
        Assert.True((double)cudaQ4.TotalKvCacheBytes / cudaF16.TotalKvCacheBytes < 0.35);

        float[] cudaLogits = new float[weights.VocabSize];
        for (int p = 0; p < prompt.Length; p++)
        {
            cudaQ4.Forward(prompt[p], p, null, cudaLogits, computeLogits: p == prompt.Length - 1);
        }
        int cudaTop1 = KlDivergenceVerifier.ArgMax(cudaLogits);

        Assert.All(cudaLogits, x => Assert.True(float.IsFinite(x)));
        Assert.Equal(cpuTop1, cudaTop1);
        Assert.True(Fx1Vectors.CosineSimilarity(cpuLogits, cudaLogits) > 0.96f);

        // 3. D3D12 Q4_0 vs Fp16 capacity & accuracy
        if (OperatingSystem.IsWindows() && D3D12Context.IsSupported)
        {
            using var ctx = new D3D12Context();
            using var d3dQ4 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Q4_0);
            using var d3dF16 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16);
            Assert.True((double)d3dQ4.TotalKvCacheBytes / d3dF16.TotalKvCacheBytes < 0.35);

            float[] d3dLogits = new float[weights.VocabSize];
            for (int p = 0; p < prompt.Length; p++)
            {
                d3dQ4.Forward(prompt[p], p, d3dLogits, computeLogits: p == prompt.Length - 1);
            }
            int d3dTop1 = KlDivergenceVerifier.ArgMax(d3dLogits);

            Assert.All(d3dLogits, x => Assert.True(float.IsFinite(x)));
            Assert.Equal(cpuTop1, d3dTop1);
            Assert.True(Fx1Vectors.CosineSimilarity(cpuLogits, d3dLogits) > 0.96f);
        }
    }
}
