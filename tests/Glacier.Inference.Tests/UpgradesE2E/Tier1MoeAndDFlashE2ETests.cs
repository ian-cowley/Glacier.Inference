using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Scheduling;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

public sealed unsafe class Tier1MoeAndDFlashE2ETests
{
    // =========================================================================
    // F5: TieredMoERunner Hybrid CPU/GPU Batch Splitting (T1-F5-01 .. T1-F5-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F5_01_TieredMoEMissSplitter_PartitionsDisjointlyIntoVramPcieAndCpu()
    {
        const int layerIdx = 0;
        const int numExperts = 8;
        const int topK = 2;
        const int batchSize = 4;

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(
            totalMoeLayers: 2,
            residentLayerMask: new[] { true, false });
        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        var profile = new ExpertRoutingProfile(2, numExperts);
        var coScheduler = new BandwidthCoScheduler();

        // Pre-populate slots 0..1 with experts (0,0) and (0,1)
        slots.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
        slots.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);

        // Batch touches experts {0, 1, 2, 3, 4, 5}
        int[] indices = { 0, 1, 2, 3, 4, 5, 0, 2 };
        float[] weights = Enumerable.Repeat(0.5f, indices.Length).ToArray();

        var splitter = new TieredMoEMissSplitter();
        var split = splitter.PlanLayerBatch(
            layerIdx,
            batchSize,
            topK,
            numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: false,
            routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.GpuSharedMemory,
            batchExpertIndices: indices,
            batchExpertWeights: weights,
            routingProfile: profile,
            slotManager: slots,
            coScheduler: coScheduler,
            hostArena: arena,
            maxPcieTransfers: 2);

        Assert.Equal(2, split.VramHitUniqueCount);
        Assert.Equal(2, split.PcieTransferUniqueCount);
        Assert.Equal(2, split.CpuExecuteUniqueCount);
        Assert.Equal(6, split.VramHitUniqueCount + split.PcieTransferUniqueCount + split.CpuExecuteUniqueCount);
        Assert.Equal(0, split.SkippedRoutedSlots);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F5_02_TieredMoEMissSplitter_NonResidentArenaLayer_ShiftsMissesFromCpuToPcie()
    {
        const int numExperts = 8;
        const int topK = 2;
        const int batchSize = 3;

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(
            totalMoeLayers: 2,
            residentLayerMask: new[] { true, false });
        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);

        Assert.True(arena.IsLayerResident(0));
        Assert.False(arena.IsLayerResident(1));

        int[] indices = { 2, 3, 4, 5, 6, 7 };
        float[] weights = Enumerable.Repeat(0.5f, indices.Length).ToArray();

        var coScheduler = new BandwidthCoScheduler();
        coScheduler.RecordCpuExecution(256 * 1024 * 1024, 2.0); // Calibrate fast DDR5 host RAM bandwidth
        coScheduler.RecordPcieTransfer(64 * 1024 * 1024, 8.0);

        var splitter = new TieredMoEMissSplitter();
        var splitLayer0 = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize,
            topK,
            numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: false,
            routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices,
            batchExpertWeights: weights,
            routingProfile: new ExpertRoutingProfile(2, numExperts),
            slotManager: slots,
            coScheduler: coScheduler,
            hostArena: arena,
            maxPcieTransfers: 4);

        var splitLayer1 = splitter.PlanLayerBatch(
            layerIdx: 1,
            batchSize,
            topK,
            numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: false,
            routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices,
            batchExpertWeights: weights,
            routingProfile: new ExpertRoutingProfile(2, numExperts),
            slotManager: slots,
            coScheduler: coScheduler,
            hostArena: arena,
            maxPcieTransfers: 4);

        Assert.True(splitLayer0.IsHostArenaLayerResident);
        Assert.False(splitLayer1.IsHostArenaLayerResident);
        Assert.True(splitLayer1.PcieTransferUniqueCount >= splitLayer0.PcieTransferUniqueCount);
        Assert.Equal(6, splitLayer1.PcieTransferUniqueCount + splitLayer1.CpuExecuteUniqueCount);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F5_03_TieredMoERunner_GpuSharedMemory_MatchesAnalyticalReferenceAndSupportsF32Q8()
    {
        // 1. GpuSharedMemory PCIe streaming mode vs analytical (e + 1) * 0.125f reference (< 1e-4)
        using var moeAnalytical = new SyntheticMoeFixture(
            numLayers: 1, numExperts: 8, topK: 2, dim: 32, expertFfnDim: 32, vocabSize: 16,
            expertType: GgufType.F32, seed: 503, analyticalDiagonal: true);
        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        slots.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
        slots.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);

        var pcieOpts = new TieredRunnerOptions { Policy = TieringExecutionPolicy.GpuSharedMemory };
        using var pcieRunner = new TieredMoERunner(
            moeAnalytical.Layers, moeAnalytical.Dim, moeAnalytical.ExpertFfnDim, moeAnalytical.NumExperts,
            moeAnalytical.TopK, moeAnalytical.VocabSize, normTopK: true,
            new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: slots, hostArena: null, options: pcieOpts, lmHeadWeights: moeAnalytical.LmHead);

        const int batchSize = 3;
        float[] input = Fx1Vectors.GenerateRampVector(batchSize * 32, 0.25f, 0.02f);
        float[] pcieOut = new float[batchSize * 32];
        var split = pcieRunner.ExecuteSyntheticMoELayerBatch(0, input, pcieOut, batchSize, isSpeculativeDraftBatch: false);

        Assert.Equal(TieringExecutionPolicy.GpuSharedMemory, split.ResolvedPolicy);
        Assert.True(split.PcieTransferUniqueCount > 0, "Expected GpuSharedMemory to stream missed experts via PCIe.");

        float[] expected = SyntheticMoeFixture.ComputeAnalyticalDiagonalLayerOutput(
            moeAnalytical.Layers[0], input, batchSize, 32, 8, 2, normTopK: true);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - pcieOut[i]) < 1e-4f,
                $"GpuSharedMemory analytical mismatch at {i}: expected={expected[i]:F6}, actual={pcieOut[i]:F6}");
        }

        // 2. F32 and Q8_0 synthetic MoE batch execution coherence
        using var moeF32 = new SyntheticMoeFixture(numLayers: 1, numExperts: 4, topK: 2, dim: 32, expertFfnDim: 32, expertType: GgufType.F32);
        using var moeQ8 = new SyntheticMoeFixture(numLayers: 1, numExperts: 4, topK: 2, dim: 32, expertFfnDim: 32, expertType: GgufType.Q8_0);

        using var runnerF32 = new TieredMoERunner(
            moeF32.Layers, moeF32.Dim, moeF32.ExpertFfnDim, moeF32.NumExperts, moeF32.TopK, moeF32.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 4), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, lmHeadWeights: moeF32.LmHead);

        using var runnerQ8 = new TieredMoERunner(
            moeQ8.Layers, moeQ8.Dim, moeQ8.ExpertFfnDim, moeQ8.NumExperts, moeQ8.TopK, moeQ8.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 4), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, lmHeadWeights: moeQ8.LmHead);

        float[] outF32 = new float[batchSize * 32];
        float[] outQ8 = new float[batchSize * 32];

        runnerF32.ExecuteSyntheticMoELayerBatch(0, input, outF32, batchSize, isSpeculativeDraftBatch: false);
        runnerQ8.ExecuteSyntheticMoELayerBatch(0, input, outQ8, batchSize, isSpeculativeDraftBatch: false);

        Assert.All(outF32, x => Assert.True(float.IsFinite(x)));
        Assert.All(outQ8, x => Assert.True(float.IsFinite(x)));
        Assert.Contains(outF32, x => MathF.Abs(x) > 1e-6f);
        Assert.Contains(outQ8, x => MathF.Abs(x) > 1e-6f);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F5_04_TieredMoERunner_DynamicSplitAndBatch_MatchesAnalyticalReferenceAndEmitsTelemetry()
    {
        // 1. Dynamic mixed CPU + PCIe miss split against analytical (e + 1) * 0.125f reference (< 1e-4)
        using var moeAnalytical = new SyntheticMoeFixture(
            numLayers: 1, numExperts: 8, topK: 6, dim: 32, expertFfnDim: 32, vocabSize: 16,
            expertType: GgufType.F32, seed: 504, analyticalDiagonal: true);
        using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true });
        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        var balancedScheduler = new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0);

        using var splitRunner = new TieredMoERunner(
            moeAnalytical.Layers, moeAnalytical.Dim, moeAnalytical.ExpertFfnDim, moeAnalytical.NumExperts,
            moeAnalytical.TopK, moeAnalytical.VocabSize, normTopK: true,
            new ExpertRoutingProfile(1, 8), balancedScheduler,
            slotManager: slots, hostArena: arena,
            options: new TieredRunnerOptions { Policy = TieringExecutionPolicy.Auto },
            lmHeadWeights: moeAnalytical.LmHead);

        float[] singleInput = Fx1Vectors.GenerateRampVector(32, 0.2f, 0.01f);
        float[] splitOut = new float[32];
        var split = splitRunner.ExecuteSyntheticMoELayerBatch(0, singleInput, splitOut, batchSize: 1, isSpeculativeDraftBatch: false);

        int totalMissed = split.PcieTransferUniqueCount + split.CpuExecuteUniqueCount;
        Assert.Equal(6, totalMissed);
        Assert.True(split.PcieTransferUniqueCount > 0, "Expected balanced Auto split to assign >0 misses to PCIe.");
        Assert.True(split.CpuExecuteUniqueCount > 0, "Expected balanced Auto split to assign >0 misses to CPU.");

        float[] expected = SyntheticMoeFixture.ComputeAnalyticalDiagonalLayerOutput(
            moeAnalytical.Layers[0], singleInput, batchSize: 1, 32, 8, 6, normTopK: true);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - splitOut[i]) < 1e-4f,
                $"CPU+PCIe combined split mismatch at {i}: expected={expected[i]:F6}, actual={splitOut[i]:F6}");
        }

        // 2. Multi-layer EvaluateSyntheticBatch emits telemetry & finite logits
        using var moe = new SyntheticMoeFixture(numLayers: 2, numExperts: 6, topK: 2, dim: 32, expertFfnDim: 32, vocabSize: 24);
        int telemetryCalls = 0;
        var options = new TieredRunnerOptions
        {
            VramReserveMb = 0,
            OnStepTelemetry = _ => telemetryCalls++
        };

        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(2, 6), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: options, lmHeadWeights: moe.LmHead);

        float[] inputs = Fx1Vectors.GenerateRampVector(4 * 32, -0.3f, 0.005f);
        float[] logits = new float[4 * 24];
        runner.EvaluateSyntheticBatch(inputs, batchSize: 4, isSpeculativeDraftBatch: false, logits);

        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
        Assert.True(telemetryCalls >= 1, $"Expected telemetry callback invocation, got {telemetryCalls}");
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F5_05_TieredMoERunner_SpeculativeTarget_VerifyBatchLogits_MatchesSequentialForward()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 2, numExperts: 4, topK: 2, dim: 32, expertFfnDim: 32, vocabSize: 16);
        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(2, 4), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, lmHeadWeights: moe.LmHead);

        int[] tokens = { 1, 3, 7, 11 };
        float[] batchLogits = new float[tokens.Length * moe.VocabSize];
        ((ISpeculativeTarget)runner).VerifyBatchLogits(tokens, startPos: 0, batchLogits);

        float[] seqLast = new float[moe.VocabSize];
        runner.ResetKvCache();
        for (int i = 0; i < tokens.Length; i++)
        {
            runner.ForwardToken(tokens[i], i, computeLogits: i == tokens.Length - 1);
        }
        runner.CopyCurrentLogits(seqLast);

        float[] batchLast = batchLogits.AsSpan((tokens.Length - 1) * moe.VocabSize, moe.VocabSize).ToArray();
        Assert.All(batchLogits, x => Assert.True(float.IsFinite(x)));
        float cos = Fx1Vectors.CosineSimilarity(seqLast, batchLast);
        Assert.True(cos > 0.999f, $"VerifyBatchLogits vs ForwardToken mismatch: cosine={cos:F6}");
    }

    // =========================================================================
    // F6: MoE Rank-Sorted Tail-Skipping (T1-F6-01 .. T1-F6-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F6_01_TieredRunnerOptions_RouteTailSkipMinRank_ValidationAndClamping()
    {
        var opts = new TieredRunnerOptions();
        Assert.Equal(0, opts.RouteTailSkipMinRank);

        opts.RouteTailSkipMinRank = -5;
        Assert.Equal(0, opts.RouteTailSkipMinRank);

        opts.RouteTailSkipMinRank = 3;
        Assert.Equal(3, opts.RouteTailSkipMinRank);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F6_02_PlanLayerBatch_RankNormalizedTailSkip_SkipsExact50PercentForTopK4Rank2()
    {
        const int batchSize = 4;
        const int topK = 4;
        const int numExperts = 8;
        // Tier-separated batch: ranks 0..1 use experts {0,1,2,3}, ranks 2..3 use experts {4,5,6,7}
        int[] indices =
        {
            0, 1, 4, 5,
            2, 3, 6, 7,
            0, 2, 4, 6,
            1, 3, 5, 7
        };
        float[] weights = new float[batchSize * topK];
        for (int b = 0; b < batchSize; b++)
        {
            weights[b * topK + 0] = 0.50f;
            weights[b * topK + 1] = 0.25f;
            weights[b * topK + 2] = 0.15f;
            weights[b * topK + 3] = 0.10f;
        }

        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        var splitter = new TieredMoEMissSplitter();
        var result = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize,
            topK,
            numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 2,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices,
            batchExpertWeights: weights,
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: slots,
            coScheduler: new BandwidthCoScheduler(),
            hostArena: null);

        Assert.Equal(16, result.TotalRoutedSlots);
        Assert.Equal(8, result.SkippedRoutedSlots);
        Assert.InRange(result.ExpertsSkippedPercent, 49.9, 50.1);
    }

    private sealed class CapturingGlacierLogger : IGlacierLogger
    {
        public List<string> Messages { get; } = new();
        public bool IsEnabled(LogLevel level) => true;
        public void Log(LogLevel level, string message, Exception? exception = null)
        {
            lock (Messages)
            {
                Messages.Add(message);
            }
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F6_03_Telemetry_LogsExactExpertsSkippedPercent_And_NonDraftBatchNeverTailSkips()
    {
        const int batchSize = 4;
        const int topK = 4;
        const int numExperts = 8;
        int[] indices =
        {
            0, 1, 4, 5,
            2, 3, 6, 7,
            0, 2, 4, 6,
            1, 3, 5, 7
        };
        float[] weights = Enumerable.Repeat(0.25f, batchSize * topK).ToArray();

        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        var splitter = new TieredMoEMissSplitter();
        var nonDraftResult = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize,
            topK,
            numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: false,
            routeTailSkipMinRank: 2,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices,
            batchExpertWeights: weights,
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: slots,
            coScheduler: new BandwidthCoScheduler(),
            hostArena: null);

        Assert.Equal(0, nonDraftResult.SkippedRoutedSlots);
        Assert.Equal(0.0, nonDraftResult.ExpertsSkippedPercent);

        // Verify TieredStepTelemetry.ExpertsSkippedPercent exact ratio and "% experts skipped" log output
        var logger = new CapturingGlacierLogger();
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16);
        TieredStepTelemetry? capturedTelemetry = null;
        var opts = new TieredRunnerOptions
        {
            RouteTailSkipMinRank = 2,
            OnStepTelemetry = t => capturedTelemetry = t
        };

        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: opts, lmHeadWeights: moe.LmHead);

        float[] single = Fx1Vectors.GenerateRampVector(32, 0.05f, 0.01f);
        float[] batchInput = new float[4 * 32];
        for (int t = 0; t < 4; t++) single.CopyTo(batchInput, t * 32);
        float[] batchOut = new float[4 * 32];

        try
        {
            GlacierDiagnostics.Logger = logger;
            runner.ExecuteSyntheticMoELayerBatch(0, batchInput, batchOut, batchSize: 4, isSpeculativeDraftBatch: true);
        }
        finally
        {
            GlacierDiagnostics.Reset();
        }

        Assert.NotNull(capturedTelemetry);
        Assert.True(capturedTelemetry!.TotalRoutedExperts > 0);
        Assert.True(capturedTelemetry.SkippedRoutedExperts > 0);
        double expectedPct = capturedTelemetry.SkippedRoutedExperts * 100.0 / capturedTelemetry.TotalRoutedExperts;
        Assert.True(Math.Abs(capturedTelemetry.ExpertsSkippedPercent - expectedPct) < 1e-6,
            $"Expected ExpertsSkippedPercent={expectedPct:F6}, got {capturedTelemetry.ExpertsSkippedPercent:F6}");

        string expectedSubstr = $"{capturedTelemetry.ExpertsSkippedPercent:F1}% experts skipped";
        Assert.Contains(logger.Messages, m =>
            m.Contains("[TieredMoERunner]", StringComparison.Ordinal) &&
            m.Contains("% experts skipped", StringComparison.Ordinal) &&
            m.Contains(expectedSubstr, StringComparison.Ordinal));
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F6_04_NonUnanimousTailExpertAcrossDraftBatch_IsNotSkipped()
    {
        // FX-2.3 M_NonUnanimousTail (topK=8, routeTailSkipMinRank=6):
        // Expert 6 has ranks [6, 7, 1, 7] across 4 draft tokens -> min rank = 1 < 6 -> MUST NOT be skipped!
        // Expert 7 has ranks [7, 6, 7, 6] across 4 draft tokens -> min rank = 6 >= 6 -> MUST be skipped!
        const int batchSize = 4;
        const int topK8 = 8;
        const int numExperts = 8;
        int[] indices8 =
        {
            0, 1, 2, 3, 4, 5, 6, 7, // tok 0: exp 6 @ rank 6, exp 7 @ rank 7
            0, 1, 2, 3, 4, 5, 7, 6, // tok 1: exp 7 @ rank 6, exp 6 @ rank 7
            0, 6, 1, 2, 3, 4, 5, 7, // tok 2: exp 6 @ rank 1 (< 6!), exp 7 @ rank 7
            0, 1, 2, 3, 4, 5, 7, 6  // tok 3: exp 7 @ rank 6, exp 6 @ rank 7
        };
        float[] weights8 = Enumerable.Repeat(1.0f / topK8, batchSize * topK8).ToArray();

        using var slots = new GpuResidentSlotManager(slotCount: 6, slotByteSize: 4096, allocateVram: false);
        for (int e = 0; e <= 5; e++) slots.AllocateOrEvictSlot(0, e, out _, out _, out _, out _);

        var splitter = new TieredMoEMissSplitter();
        var res8 = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize: batchSize,
            topK: topK8,
            numExperts: numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 6,
            policy: TieringExecutionPolicy.HybridGpuHostRam,
            batchExpertIndices: indices8,
            batchExpertWeights: weights8,
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: slots,
            coScheduler: new BandwidthCoScheduler(),
            hostArena: null);

        Assert.True(splitter.TryGetPlannedExpert(6, out var exp6));
        Assert.Equal(1, exp6.MinRankInBatch);
        Assert.NotEqual(ExpertExecutionDisposition.Skipped, exp6.Disposition);
        Assert.True(exp6.Disposition == ExpertExecutionDisposition.CpuExecute || exp6.Disposition == ExpertExecutionDisposition.PcieTransfer);

        Assert.True(splitter.TryGetPlannedExpert(7, out var exp7));
        Assert.Equal(6, exp7.MinRankInBatch);
        Assert.Equal(ExpertExecutionDisposition.Skipped, exp7.Disposition);

        // Only expert 7 (4 slots out of 32) is skipped; expert 6 is excluded from skipped count
        Assert.Equal(32, res8.TotalRoutedSlots);
        Assert.Equal(4, res8.SkippedRoutedSlots);
        Assert.True(Math.Abs(res8.ExpertsSkippedPercent - (4.0 * 100.0 / 32.0)) < 1e-6);

        // Also verify topK=4, routeTailSkipMinRank=2 (expert 6 @ [2,3,0,3], expert 7 @ [3,2,3,2])
        const int topK4 = 4;
        int[] indices4 =
        {
            0, 1, 6, 7, // tok 0: exp 6 @ rank 2, exp 7 @ rank 3
            0, 1, 7, 6, // tok 1: exp 7 @ rank 2, exp 6 @ rank 3
            6, 0, 1, 7, // tok 2: exp 6 @ rank 0 (< 2!), exp 7 @ rank 3
            0, 1, 7, 6  // tok 3: exp 7 @ rank 2, exp 6 @ rank 3
        };
        float[] weights4 = Enumerable.Repeat(0.25f, batchSize * topK4).ToArray();
        var res4 = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize: batchSize,
            topK: topK4,
            numExperts: numExperts,
            normTopK: true,
            isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 2,
            policy: TieringExecutionPolicy.HybridGpuHostRam,
            batchExpertIndices: indices4,
            batchExpertWeights: weights4,
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: slots,
            coScheduler: new BandwidthCoScheduler(),
            hostArena: null);

        Assert.True(splitter.TryGetPlannedExpert(6, out var exp6Top4));
        Assert.Equal(0, exp6Top4.MinRankInBatch);
        Assert.NotEqual(ExpertExecutionDisposition.Skipped, exp6Top4.Disposition);

        Assert.True(splitter.TryGetPlannedExpert(7, out var exp7Top4));
        Assert.Equal(2, exp7Top4.MinRankInBatch);
        Assert.Equal(ExpertExecutionDisposition.Skipped, exp7Top4.Disposition);

        Assert.Equal(16, res4.TotalRoutedSlots);
        Assert.Equal(4, res4.SkippedRoutedSlots);
        Assert.True(Math.Abs(res4.ExpertsSkippedPercent - 25.0) < 1e-6);

        // End-to-end TieredMoERunner batch execution with isSpeculativeDraftBatch: true
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16);
        using var runnerSkip = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: new TieredRunnerOptions { RouteTailSkipMinRank = 2 }, lmHeadWeights: moe.LmHead);

        float[] input = Fx1Vectors.GenerateRampVector(3 * 32, -0.1f, 0.01f);
        float[] outSkip = new float[3 * 32];
        var runnerSplit = runnerSkip.ExecuteSyntheticMoELayerBatch(0, input, outSkip, 3, isSpeculativeDraftBatch: true);
        Assert.All(outSkip, x => Assert.True(float.IsFinite(x)));
        Assert.True(runnerSplit.TotalRoutedSlots == 12);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F6_05_TieredMoERunner_Telemetry_Reports75PercentSkippedForTopK4MinRank1()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16);
        double lastSkippedPct = -1.0;
        var opts = new TieredRunnerOptions
        {
            RouteTailSkipMinRank = 1,
            OnStepTelemetry = snap => lastSkippedPct = snap.ExpertsSkippedPercent
        };

        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: opts, lmHeadWeights: moe.LmHead);

        float[] inputs = new float[4 * 32];
        float[] single = Fx1Vectors.GenerateRampVector(32, 0.05f, 0.01f);
        for (int t = 0; t < 4; t++) single.CopyTo(inputs, t * 32);

        float[] logits = new float[4 * 16];
        runner.EvaluateSyntheticBatch(inputs, batchSize: 4, isSpeculativeDraftBatch: true, logits);

        Assert.InRange(lastSkippedPct, 70.0, 80.0);
    }

    // =========================================================================
    // F7: DFlashDraftProvider Block Diffusion Speculative Drafting (T1-F7-01 .. T1-F7-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F7_01_DFlashDraftProvider_OptionsAndMaxBlockSize_AreValid()
    {
        Assert.Equal(1024, DFlashDraftProvider.MaxDFlashBlockSize);

        var opts = new DFlashOptions { BlockSize = 8, DiffusionSteps = 3 };
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        using var provider = new DFlashDraftProvider(trunk, weights: null, options: opts);

        Assert.Equal(8, provider.BlockSize);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F7_02_DFlashDraftProvider_ContextConditioned_PredictsRepeatingPattern()
    {
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        using var provider = new DFlashDraftProvider(trunk, weights: null, new DFlashOptions { BlockSize = 4 });

        int[] history = new int[32];
        for (int i = 0; i < history.Length; i++) history[i] = 10 + (i % 4); // [10,11,12,13, 10,11,12,13,...]

        int[] drafts = new int[4];
        int proposed = provider.Draft(history, maxDraftTokens: 4, drafts);

        Assert.Equal(4, proposed);
        Assert.Equal(new[] { 10, 11, 12, 13 }, drafts);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F7_03_CostGatedDraftPolicy_SelectRoute_PrefersDFlashWhenAvailable()
    {
        var policy = new CostGatedDraftPolicy();
        var routeWithDFlash = policy.SelectRoute(
            maxDraftTokens: 4,
            hasMtp: false,
            lookupCandidateCount: 0,
            matchedNgramLength: 0,
            hasDFlash: true,
            dflashCandidateCount: 4);

        Assert.Equal(DraftRoute.DFlash, routeWithDFlash.Route);

        var routeNoDFlash = policy.SelectRoute(
            maxDraftTokens: 4,
            hasMtp: false,
            lookupCandidateCount: 2,
            matchedNgramLength: 3,
            hasDFlash: false,
            dflashCandidateCount: 0);

        Assert.NotEqual(DraftRoute.DFlash, routeNoDFlash.Route);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F7_04_AdaptiveCompositeDraftProvider_WithDFlash_DraftsAndResetsCleanly()
    {
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        var dflash = new DFlashDraftProvider(trunk, weights: null, new DFlashOptions { BlockSize = 4 });
        var lookup = new PromptLookupDraftProvider();
        using var composite = new AdaptiveCompositeDraftProvider(lookup, mtpProvider: null, new CostGatedDraftPolicy(), ownsProviders: true, dflashProvider: dflash);

        int[] context = { 4, 5, 6, 4, 5, 6, 4, 5, 6 };
        int[] drafts = new int[4];
        int count = composite.Draft(context, maxDraftTokens: 3, drafts);

        Assert.Equal(3, count);
        Assert.Equal(new[] { 4, 5, 6 }, drafts.AsSpan(0, 3).ToArray());

        composite.Reset();
        int[] afterReset = new int[4];
        int countAfter = composite.Draft(new[] { 1 }, maxDraftTokens: 3, afterReset);
        Assert.InRange(countAfter, 0, 3);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public async System.Threading.Tasks.Task T1_F7_05_SpeculativeEngine_WithDFlashDraftProvider_MatchesGreedyTargetTrajectory()
    {
        using var target = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, maxSeqLen: 128);
        using var dflash = new DFlashDraftProvider(target, weights: null, new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(target, dflash, seed: 42);

        // Prompt matches SyntheticSpeculativeTarget's transition cycle ('a'->'d'->'g'->'j'->'m'->'p'->'s'->'v'->'y')
        const string promptText = "adgjmpsvyadgjmpsvya";
        var specResult = await engine.GenerateAsync(
            promptText,
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = Glacier.Inference.Sampling.SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, specResult.Metrics.GeneratedTokens);
        Assert.True(specResult.SpeculativeMetrics.DraftTokensProposed > 0, "Expected DFlashDraftProvider to propose draft tokens (no silent serial fallback).");
        Assert.True(specResult.SpeculativeMetrics.DraftTokensAccepted > 0, "Expected SpeculativeEngine to accept DFlash draft tokens.");

        // Compare against pure target greedy baseline
        int[] promptTokens = target.Tokenizer.Encode(promptText);
        target.ResetKvCache();
        target.Prefill(promptTokens);
        var baselineChars = new System.Text.StringBuilder();
        var decoder = System.Text.Encoding.UTF8.GetDecoder();
        char[] cbuf = new char[16];
        int first = target.SampleNextToken(Glacier.Inference.Sampling.SamplingOptions.Greedy);
        var gen = new List<int> { first };
        int pos = promptTokens.Length;
        for (int step = 0; step < 5; step++)
        {
            target.ForwardToken(gen[^1], pos++, computeLogits: true);
            gen.Add(target.SampleNextToken(Glacier.Inference.Sampling.SamplingOptions.Greedy));
        }
        foreach (int tok in gen)
        {
            byte[] tb = target.Tokenizer.GetTokenBytes(tok);
            int cc = decoder.GetChars(tb, 0, tb.Length, cbuf, 0, flush: false);
            baselineChars.Append(cbuf, 0, cc);
        }

        Assert.Equal(baselineChars.ToString(), specResult.Text);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public async System.Threading.Tasks.Task T1_F7_06_DFlashDraftProvider_LoadFromSidecar_NeuralBlockDiffusion_ExecutesInSpeculativeEngine()
    {
        using var tmp = new TempDirScope();
        string baseModelPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("dflash_base_64.gguf"),
            dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64);
        int[] slotTargets = { 5, 6, 7, 8 }; // 'd', 'e', 'f', 'g'
        string sidecarPath = UpgradeScenarioFixtures.BuildPositiveDFlashSidecarGguf(
            tmp.GetFilePath("dflash_neural_sidecar.gguf"),
            dim: 64, ffnDim: 64, nHeads: 4, nHeadsKv: 2, vocabSize: 32, blockSize: 4,
            slotTargetTokens: slotTargets);

        using var gguf = GgufFile.Open(baseModelPath);
        using var weights = new ModelWeights(gguf);
        // Scripted target predicts 4 ('c'), then [5, 6, 7, 8, 9] matching neural sidecar block [5, 6, 7, 8]
        using var target = new ScriptedEosSpeculativeTarget(new[] { 4, 5, 6, 7, 8, 9 }, vocabSize: 32, dim: 64);

        using var neuralDFlash = DFlashDraftProvider.LoadFromSidecar(
            target, weights, sidecarPath, new DFlashOptions { BlockSize = 4, DiffusionSteps = 2 });

        Assert.True(neuralDFlash.HasNeuralWeights);
        Assert.Equal(4, neuralDFlash.BlockSize);

        // Verify direct multi-token neural block generation
        int[] drafts = new int[4];
        int proposed = neuralDFlash.Draft(new[] { 2, 3, 4 }, maxDraftTokens: 4, drafts);
        Assert.Equal(4, proposed);
        Assert.Equal(slotTargets, drafts);

        // Verify end-to-end SpeculativeEngine multi-token block acceptance with neural sidecar
        using var engine = new SpeculativeEngine(target, neuralDFlash, seed: 76);
        var res = await engine.GenerateAsync(
            "abc",
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = Glacier.Inference.Sampling.SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, res.Metrics.GeneratedTokens);
        Assert.True(res.SpeculativeMetrics.DraftTokensProposed >= 4, $"Expected >= 4 proposed tokens, got {res.SpeculativeMetrics.DraftTokensProposed}");
        Assert.True(res.SpeculativeMetrics.DraftTokensAccepted >= 4, $"Expected >= 4 accepted tokens, got {res.SpeculativeMetrics.DraftTokensAccepted}");
        Assert.Equal("cdefgh", res.Text);
    }
}
