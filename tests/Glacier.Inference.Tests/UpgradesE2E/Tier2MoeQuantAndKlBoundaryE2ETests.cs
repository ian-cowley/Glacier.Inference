using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Scheduling;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier2")]
public sealed unsafe class Tier2MoeQuantAndKlBoundaryE2ETests
{
    // =========================================================================
    // F5 Boundaries: Dynamic CPU/PCIe Expert-Miss Splitting (T2-F5-01 .. T2-F5-05)
    // =========================================================================

    [Fact]
    public void T2_F5_01_PlanLayerBatch_ZeroMisses_AllVramHits_AssignsZeroCpuAndZeroPcie()
    {
        const int layerIdx = 0;
        const int numExperts = 8;
        const int topK = 2;
        const int batchSize = 4;

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(2, new[] { true, true });
        using var slots = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 4096, allocateVram: false);
        var profile = new ExpertRoutingProfile(2, numExperts);
        var coScheduler = new BandwidthCoScheduler();

        slots.AllocateOrEvictSlot(layerIdx, 0, out _, out _, out _, out _);
        slots.AllocateOrEvictSlot(layerIdx, 1, out _, out _, out _, out _);

        int[] indices = { 0, 1, 1, 0, 0, 1, 1, 0 };
        float[] weights = Enumerable.Repeat(0.5f, indices.Length).ToArray();

        var splitter = new TieredMoEMissSplitter();
        var split = splitter.PlanLayerBatch(
            layerIdx, batchSize, topK, numExperts, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices, batchExpertWeights: weights,
            routingProfile: profile, slotManager: slots, coScheduler: coScheduler, hostArena: arena);

        Assert.Equal(2, split.VramHitUniqueCount);
        Assert.Equal(0, split.PcieTransferUniqueCount);
        Assert.Equal(0, split.CpuExecuteUniqueCount);
        Assert.Equal(8, split.VramHitRoutedSlots);
        Assert.Equal(0, split.SkippedRoutedSlots);
        Assert.Equal(0L, slots.EvictionCount);

        Assert.True(splitter.TryGetPlannedExpert(0, out var p0));
        Assert.Equal(ExpertExecutionDisposition.VramHit, p0.Disposition);
        Assert.True(splitter.TryGetPlannedExpert(1, out var p1));
        Assert.Equal(ExpertExecutionDisposition.VramHit, p1.Disposition);
    }

    [Fact]
    public void T2_F5_02_PlanLayerBatch_100PercentMisses_WithZeroVramSlots_SplitsAllMissedAndMatchesCpuOracle()
    {
        const int numExperts = 8;
        const int topK = 2;
        const int batchSize = 3;

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true });
        // Zero active VRAM slot capacity: 100% of routed experts must miss VRAM
        using var zeroCapSlots = new GpuResidentSlotManager(
            slotCount: 4, slotByteSize: 4096, allocateVram: false,
            SlotEvictionPolicy.Lru, routingProfile: null, initialActiveSlots: 0);
        Assert.Equal(0, zeroCapSlots.ActiveSlotCount);

        int[] indices = { 0, 1, 2, 3, 4, 5 };
        float[] weights = Enumerable.Repeat(0.5f, indices.Length).ToArray();

        var splitter = new TieredMoEMissSplitter();
        var zeroCapSplit = splitter.PlanLayerBatch(
            layerIdx: 0, batchSize, topK, numExperts, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: indices, batchExpertWeights: weights,
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: zeroCapSlots, coScheduler: new BandwidthCoScheduler(), hostArena: arena,
            maxPcieTransfers: 6);

        Assert.Equal(0, zeroCapSplit.VramHitRoutedSlots);
        Assert.Equal(0, zeroCapSplit.VramHitUniqueCount);
        Assert.Equal(6, zeroCapSplit.PcieTransferUniqueCount + zeroCapSplit.CpuExecuteUniqueCount);

        // With slotManager: null (VramExpertSlotCount = 0) and balanced bandwidth, 100% misses split across both CPU and PCIe
        var balancedScheduler = new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0);
        var nullSlotSplit = splitter.PlanLayerBatch(
            layerIdx: 0, batchSize, topK, numExperts, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            policy: TieringExecutionPolicy.Auto,
            batchExpertIndices: (int[])indices.Clone(), batchExpertWeights: (float[])weights.Clone(),
            routingProfile: new ExpertRoutingProfile(1, numExperts),
            slotManager: null, coScheduler: balancedScheduler, hostArena: arena,
            maxPcieTransfers: 6);

        Assert.Equal(0, nullSlotSplit.VramHitRoutedSlots);
        Assert.Equal(0, nullSlotSplit.VramHitUniqueCount);
        Assert.Equal(6, nullSlotSplit.PcieTransferUniqueCount + nullSlotSplit.CpuExecuteUniqueCount);
        Assert.True(nullSlotSplit.PcieTransferUniqueCount > 0 && nullSlotSplit.CpuExecuteUniqueCount > 0);

        // Numerical equivalence between 0-VRAM-slot split runner, full-CPU oracle, and analytical (e+1)*0.125f reference (< 1e-4)
        using var moe = new SyntheticMoeFixture(
            numLayers: 1, numExperts: 8, topK: 4, dim: 32, expertFfnDim: 32, vocabSize: 16,
            seed: 2502, analyticalDiagonal: true);
        using var cpuRunner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null,
            options: new TieredRunnerOptions { VramExpertSlotCount = 0 }, lmHeadWeights: moe.LmHead);
        using var splitRunner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 8),
            new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0),
            slotManager: null, hostArena: arena,
            options: new TieredRunnerOptions { VramExpertSlotCount = 0, Policy = TieringExecutionPolicy.Auto },
            lmHeadWeights: moe.LmHead);

        float[] input = Fx1Vectors.GenerateRampVector(batchSize * 32, 0.15f, 0.01f);
        float[] cpuOut = new float[batchSize * 32];
        float[] splitOut = new float[batchSize * 32];

        cpuRunner.ExecuteSyntheticMoELayerBatch(0, input, cpuOut, batchSize, isSpeculativeDraftBatch: false);
        var execSplit = splitRunner.ExecuteSyntheticMoELayerBatch(0, input, splitOut, batchSize, isSpeculativeDraftBatch: false);

        Assert.Equal(0, execSplit.VramHitRoutedSlots);
        Assert.Equal(0, splitRunner.LastStepTelemetry!.VramHitExperts);
        Assert.True(execSplit.CpuExecuteUniqueCount + execSplit.PcieTransferUniqueCount > 0);

        float[] analytical = SyntheticMoeFixture.ComputeAnalyticalDiagonalLayerOutput(
            moe.Layers[0], input, batchSize, 32, 8, 4, normTopK: true);

        for (int i = 0; i < cpuOut.Length; i++)
        {
            Assert.True(MathF.Abs(cpuOut[i] - splitOut[i]) < 1e-4f, $"Index {i}: CPU={cpuOut[i]} vs Split={splitOut[i]}");
            Assert.True(MathF.Abs(analytical[i] - splitOut[i]) < 1e-4f, $"Index {i}: Analytical={analytical[i]} vs Split={splitOut[i]}");
        }
    }

    [Fact]
    public void T2_F5_03_PlanLayerBatch_SingleTokenOneMiss_And_ZeroBatch_NoDivisionByZero()
    {
        using var slots = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 4096, allocateVram: false);
        var splitter = new TieredMoEMissSplitter();

        // Single token, 1 miss under HybridGpuHostRam -> 1 CPU, 0 PCIe
        int[] idx1 = { 3 };
        float[] w1 = { 1.0f };
        var cpuSplit = splitter.PlanLayerBatch(
            0, batchSize: 1, topK: 1, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.HybridGpuHostRam, idx1, w1,
            new ExpertRoutingProfile(1, 8), slots, new BandwidthCoScheduler(), hostArena: null);

        Assert.Equal(1, cpuSplit.CpuExecuteUniqueCount);
        Assert.Equal(0, cpuSplit.PcieTransferUniqueCount);
        Assert.Equal(1, cpuSplit.CpuExecuteUniqueCount + cpuSplit.PcieTransferUniqueCount);
        Assert.Equal(0.0, cpuSplit.ExpertsSkippedPercent);

        // Single token, 1 miss under GpuSharedMemory -> 1 PCIe, 0 CPU
        int[] idx2 = { 4 };
        float[] w2 = { 1.0f };
        var pcieSplit = splitter.PlanLayerBatch(
            0, batchSize: 1, topK: 1, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.GpuSharedMemory, idx2, w2,
            new ExpertRoutingProfile(1, 8), slots, new BandwidthCoScheduler(), hostArena: null);

        Assert.Equal(1, pcieSplit.PcieTransferUniqueCount);
        Assert.Equal(0, pcieSplit.CpuExecuteUniqueCount);
        Assert.Equal(1, pcieSplit.CpuExecuteUniqueCount + pcieSplit.PcieTransferUniqueCount);

        // Zero batch size -> clean 0 counts, finite 0.0% (not NaN)
        var emptySplit = splitter.PlanLayerBatch(
            0, batchSize: 0, topK: 2, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: true, routeTailSkipMinRank: 1,
            TieringExecutionPolicy.Auto, Span<int>.Empty, Span<float>.Empty,
            null, slots, null, null);

        Assert.Equal(0, emptySplit.TotalRoutedSlots);
        Assert.Equal(0, emptySplit.SkippedRoutedSlots);
        Assert.True(double.IsFinite(emptySplit.ExpertsSkippedPercent));
        Assert.Equal(0.0, emptySplit.ExpertsSkippedPercent);
    }

    [Fact]
    public void T2_F5_04_BandwidthCoScheduler_And_MissSplitter_ExtremeThroughputSkew_ClampsWithoutNaN()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BandwidthCoScheduler(initialPcieGBps: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BandwidthCoScheduler(initialCpuGBps: -1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BandwidthCoScheduler(vramGBps: double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BandwidthCoScheduler(ewmaAlpha: double.PositiveInfinity));

        using var slots = new GpuResidentSlotManager(slotCount: 8, slotByteSize: 4096, allocateVram: false);
        var splitter = new TieredMoEMissSplitter();
        int[] indices = { 0, 1, 2, 3, 4, 5 };
        float[] weights = Enumerable.Repeat(0.5f, 6).ToArray();

        // Skew 1: PCIe ultra-slow (0.0001 GB/s) vs CPU ultra-fast (500 GB/s) -> 100% CPU
        var slowPcie = new BandwidthCoScheduler(initialPcieGBps: 0.0001, initialCpuGBps: 500.0, vramGBps: 500.0);
        Assert.Equal(0, slowPcie.CalculateOptimalGpuExpertCount(6, 0));
        slowPcie.RecordPcieTransfer(1024, 0.0);
        slowPcie.RecordPcieTransfer(1024, double.NaN);
        slowPcie.RecordCpuExecution(0, 1.0);
        Assert.True(double.IsFinite(slowPcie.PcieBandwidthGBps));
        Assert.True(double.IsFinite(slowPcie.CpuBandwidthGBps));

        var splitCpuOnly = splitter.PlanLayerBatch(
            0, batchSize: 3, topK: 2, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.Auto, indices, weights,
            new ExpertRoutingProfile(1, 8), slots, slowPcie, hostArena: null);

        Assert.Equal(0, splitCpuOnly.PcieTransferUniqueCount);
        Assert.Equal(6, splitCpuOnly.CpuExecuteUniqueCount);

        // Skew 2: PCIe ultra-fast (500 GB/s) vs CPU ultra-slow (0.0001 GB/s) -> 100% PCIe
        var slowCpu = new BandwidthCoScheduler(initialPcieGBps: 500.0, initialCpuGBps: 0.0001, vramGBps: 1000.0);
        Assert.Equal(6, slowCpu.CalculateOptimalGpuExpertCount(6, 0));

        var splitPcieOnly = splitter.PlanLayerBatch(
            0, batchSize: 3, topK: 2, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.Auto, indices, weights,
            new ExpertRoutingProfile(1, 8), slots, slowCpu, hostArena: null);

        Assert.Equal(6, splitPcieOnly.PcieTransferUniqueCount);
        Assert.Equal(0, splitPcieOnly.CpuExecuteUniqueCount);
    }

    [Fact]
    public void T2_F5_05_MissSplitter_And_HostRamArena_InvalidIndicesAndZeroBudget_HandledSafely()
    {
        var splitter = new TieredMoEMissSplitter();
        Assert.Throws<ArgumentException>(() =>
            splitter.PlanLayerBatch(0, 2, 2, 8, true, false, 0, TieringExecutionPolicy.Auto,
                new int[2], new float[4], null, null, null, null));
        Assert.Throws<ArgumentException>(() =>
            splitter.PlanLayerBatch(0, 2, 2, 8, true, false, 0, TieringExecutionPolicy.Auto,
                new int[4], new float[2], null, null, null, null));

        // Out-of-range expert indices (-1, 999) are safely ignored without OOB access
        int[] dirtyIndices = { -1, 2, 999, 3 };
        float[] weights = { 0.5f, 0.5f, 0.5f, 0.5f };
        var split = splitter.PlanLayerBatch(
            0, batchSize: 2, topK: 2, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: false, routeTailSkipMinRank: 0,
            TieringExecutionPolicy.HybridGpuHostRam, dirtyIndices, weights,
            null, null, null, null);

        Assert.Equal(2, split.TotalRoutedSlots);
        Assert.Equal(2, split.CpuExecuteUniqueCount);
        Assert.False(splitter.TryGetPlannedExpert(-1, out _));
        Assert.False(splitter.TryGetPlannedExpert(999, out _));

        // HostRamExpertArena bounds: budget = 0 / non-MoE returns null, IsLayerResident out-of-range returns false
        using var tmp = new TempDirScope();
        string densePath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("dense_no_arena.gguf"), dim: 64, ffnDim: 64, nLayers: 1);
        using var gguf = GgufFile.Open(densePath);
        using var weightsDense = new ModelWeights(gguf);
        Assert.Null(HostRamExpertArena.TryAttach(weightsDense, budgetBytes: 1024 * 1024));

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(2, new[] { true, false });
        Assert.False(arena.IsLayerResident(-1));
        Assert.False(arena.IsLayerResident(99));
    }

    // =========================================================================
    // F6 Boundaries: Opt-In RouteTailSkipMinRank & Telemetry (T2-F6-01 .. T2-F6-05)
    // =========================================================================

    [Fact]
    public void T2_F6_01_PlanLayerBatch_RouteTailSkipMinRank0Vs1_BoundaryBehavior()
    {
        const int batchSize = 4;
        const int topK = 4;
        const int numExperts = 8;

        // Rank 0 experts: {0, 1}; Rank 1..3 experts: {2, 3, 4, 5, 6, 7}
        int[] indices =
        {
            0, 2, 4, 6,
            1, 3, 5, 7,
            0, 3, 4, 7,
            1, 2, 5, 6
        };
        float[] weights = Enumerable.Repeat(0.25f, batchSize * topK).ToArray();
        var splitter = new TieredMoEMissSplitter();

        var rank0Result = splitter.PlanLayerBatch(
            0, batchSize, topK, numExperts, true, isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 0, TieringExecutionPolicy.Auto,
            (int[])indices.Clone(), (float[])weights.Clone(),
            new ExpertRoutingProfile(1, numExperts), null, new BandwidthCoScheduler(), null);

        Assert.Equal(0, rank0Result.SkippedRoutedSlots);
        Assert.Equal(0.0, rank0Result.ExpertsSkippedPercent);

        var rank1Result = splitter.PlanLayerBatch(
            0, batchSize, topK, numExperts, true, isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 1, TieringExecutionPolicy.Auto,
            (int[])indices.Clone(), (float[])weights.Clone(),
            new ExpertRoutingProfile(1, numExperts), null, new BandwidthCoScheduler(), null);

        Assert.Equal(16, rank1Result.TotalRoutedSlots);
        Assert.Equal(12, rank1Result.SkippedRoutedSlots);
        Assert.InRange(rank1Result.ExpertsSkippedPercent, 74.999, 75.001);
    }

    [Fact]
    public void T2_F6_02_PlanLayerBatch_RouteTailSkipMinRankEqualOrExceedingTopK_SkipsZero()
    {
        const int batchSize = 4;
        const int topK = 4;
        const int numExperts = 8;

        int[] indices =
        {
            0, 1, 6, 7,
            2, 3, 6, 7,
            0, 2, 6, 7,
            1, 3, 6, 7
        };
        float[] weights = Enumerable.Repeat(0.25f, batchSize * topK).ToArray();
        var splitter = new TieredMoEMissSplitter();

        foreach (int rank in new[] { topK, numExperts, 100 })
        {
            var res = splitter.PlanLayerBatch(
                0, batchSize, topK, numExperts, true, isSpeculativeDraftBatch: true,
                routeTailSkipMinRank: rank, TieringExecutionPolicy.Auto,
                (int[])indices.Clone(), (float[])weights.Clone(),
                new ExpertRoutingProfile(1, numExperts), null, new BandwidthCoScheduler(), null);

            Assert.Equal(0, res.SkippedRoutedSlots);
            Assert.Equal(0.0, res.ExpertsSkippedPercent);
        }
    }

    [Fact]
    public void T2_F6_03_RouteTailSkipMinRank_NegativeValue_ClampsAndDisablesCleanly()
    {
        var opts = new TieredRunnerOptions { RouteTailSkipMinRank = -1 };
        Assert.Equal(0, opts.RouteTailSkipMinRank);

        opts.RouteTailSkipMinRank = -100;
        Assert.Equal(0, opts.RouteTailSkipMinRank);

        int[] indices = { 0, 1, 6, 7, 0, 1, 6, 7 };
        float[] weights = Enumerable.Repeat(0.25f, 8).ToArray();
        var splitter = new TieredMoEMissSplitter();

        var res = splitter.PlanLayerBatch(
            0, batchSize: 2, topK: 4, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: true, routeTailSkipMinRank: -1,
            TieringExecutionPolicy.Auto, indices, weights,
            new ExpertRoutingProfile(1, 8), null, new BandwidthCoScheduler(), null);

        Assert.Equal(0, res.SkippedRoutedSlots);
        Assert.Equal(0.0, res.ExpertsSkippedPercent);
    }

    [Fact]
    public void T2_F6_04_PlanLayerBatch_VramResidentHitInBottomTail_IsNeverSkipped()
    {
        const int batchSize = 4;
        const int topK = 4;
        const int numExperts = 8;

        using var slots = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 4096, allocateVram: false);
        // Expert 0 is VRAM-resident hit, even though it appears exclusively at rank 3 across the batch
        slots.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);

        int[] indices =
        {
            1, 2, 7, 0,
            3, 4, 7, 0,
            1, 3, 7, 0,
            2, 4, 7, 0
        };
        float[] weights = Enumerable.Repeat(0.25f, batchSize * topK).ToArray();

        var splitter = new TieredMoEMissSplitter();
        var res = splitter.PlanLayerBatch(
            0, batchSize, topK, numExperts, normTopK: true,
            isSpeculativeDraftBatch: true, routeTailSkipMinRank: 2,
            TieringExecutionPolicy.Auto, indices, weights,
            new ExpertRoutingProfile(1, numExperts), slots, new BandwidthCoScheduler(), null);

        Assert.True(splitter.TryGetPlannedExpert(0, out var hitInTail));
        Assert.Equal(ExpertExecutionDisposition.VramHit, hitInTail.Disposition);

        Assert.True(splitter.TryGetPlannedExpert(7, out var missInTail));
        Assert.Equal(ExpertExecutionDisposition.Skipped, missInTail.Disposition);

        // Only expert 7 (4 slots) is skipped; expert 0 (4 slots) executes as VRAM hit
        Assert.Equal(4, res.VramHitRoutedSlots);
        Assert.Equal(4, res.SkippedRoutedSlots);
        Assert.InRange(res.ExpertsSkippedPercent, 24.999, 25.001);
    }

    [Fact]
    public void T2_F6_05_PlanLayerBatch_SingleTokenVsEmptyBatch_ExactPercentWithoutNaN()
    {
        var splitter = new TieredMoEMissSplitter();

        int[] singleIdx = { 0, 1, 6, 7 };
        float[] singleW = { 0.4f, 0.3f, 0.2f, 0.1f };
        var single = splitter.PlanLayerBatch(
            0, batchSize: 1, topK: 4, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: true, routeTailSkipMinRank: 2,
            TieringExecutionPolicy.Auto, singleIdx, singleW,
            new ExpertRoutingProfile(1, 8), null, new BandwidthCoScheduler(), null);

        Assert.Equal(4, single.TotalRoutedSlots);
        Assert.Equal(2, single.SkippedRoutedSlots);
        Assert.InRange(single.ExpertsSkippedPercent, 49.999, 50.001);
        // Kept weights {0, 1} renormalized: 0.4/(0.7)=0.5714, 0.3/(0.7)=0.4286
        Assert.InRange(singleW[0] + singleW[1], 0.9999f, 1.0001f);
        Assert.Equal(0.0f, singleW[2]);
        Assert.Equal(0.0f, singleW[3]);

        // Empty batch (0 dispatches) yields 0.0% (no 0/0 NaN)
        var zero = splitter.PlanLayerBatch(
            0, batchSize: 0, topK: 4, numExperts: 8, normTopK: true,
            isSpeculativeDraftBatch: true, routeTailSkipMinRank: 2,
            TieringExecutionPolicy.Auto, Span<int>.Empty, Span<float>.Empty,
            null, null, null, null);

        Assert.Equal(0, zero.TotalRoutedSlots);
        Assert.Equal(0, zero.SkippedRoutedSlots);
        Assert.False(double.IsNaN(zero.ExpertsSkippedPercent));
        Assert.Equal(0.0, zero.ExpertsSkippedPercent);
    }

    // =========================================================================
    // F7 Boundaries: DFlashDraftProvider & SpeculativeEngine (T2-F7-01 .. T2-F7-05)
    // =========================================================================

    [Fact]
    public void T2_F7_01_DFlashDraftProvider_MaxDraftTokensZeroAndOne_And_InvalidOptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashOptions { BlockSize = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashOptions { BlockSize = DFlashDraftProvider.MaxDFlashBlockSize + 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashOptions { DiffusionSteps = 0 });

        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        using var provider = new DFlashDraftProvider(trunk, weights: null, new DFlashOptions { BlockSize = 4 });

        int[] history = { 10, 11, 12, 13, 10, 11, 12, 13 };
        int[] buf = new int[4];

        Assert.Equal(0, provider.Draft(history, maxDraftTokens: 0, buf));

        int one = provider.Draft(history, maxDraftTokens: 1, buf);
        Assert.Equal(1, one);
        Assert.Equal(10, buf[0]);
    }

    [Fact]
    public async Task T2_F7_02_SpeculativeEngine_DFlashTotalRejection_MatchesTargetWithoutStall()
    {
        using var target = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, maxSeqLen: 128);
        using var dflash = new DFlashDraftProvider(target, weights: null, new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(target, dflash, seed: 7);

        // Prompt of repeated 'a' (token 2): DFlash predicts [2, 2, 2, 2], while target favors 2->5->8->11->...
        // causing 100% rejection at slot 0 of every DFlash block without stalling.
        const string promptText = "aaaaaaaaaaaa";
        var specRes = await engine.GenerateAsync(
            promptText,
            new SpeculativeOptions { MaxTokens = 6, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(6, specRes.Metrics.GeneratedTokens);

        int[] promptTokens = target.Tokenizer.Encode(promptText);
        target.ResetKvCache();
        target.Prefill(promptTokens);
        int first = target.SampleNextToken(SamplingOptions.Greedy);
        var expected = new List<int> { first };
        int pos = promptTokens.Length;
        for (int s = 0; s < 5; s++)
        {
            target.ForwardToken(expected[^1], pos++, computeLogits: true);
            expected.Add(target.SampleNextToken(SamplingOptions.Greedy));
        }

        string expectedText = string.Concat(expected.Select(t => target.Tokenizer.DecodeToken(t)));
        Assert.Equal(expectedText, specRes.Text);
    }

    [Fact]
    public async Task T2_F7_03_SpeculativeEngine_DFlashTotalAndPartialAcceptance_AdvancesAccurately()
    {
        using var target = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, maxSeqLen: 128);
        using var dflash = new DFlashDraftProvider(target, weights: null, new DFlashOptions { BlockSize = 4, EnableAdaptiveConfidence = true });

        // Verify adaptive block sizing feedback via OnVerificationResult
        double initialRate = dflash.SmoothedAcceptanceRate;
        dflash.OnVerificationResult(draftedCount: 4, acceptedCount: 4, roundElapsedMs: 1.0);
        Assert.True(dflash.SmoothedAcceptanceRate >= initialRate);
        for (int i = 0; i < 15; i++) dflash.OnVerificationResult(draftedCount: 4, acceptedCount: 0, roundElapsedMs: 1.0);
        Assert.True(dflash.EffectiveBlockSize < 4);

        // Reset and run full acceptance trajectory:
        // In SyntheticSpeculativeTarget: token 2 ('a') -> 5 ('d') -> 8 ('g') -> 11 ('j') -> 14 ('m') -> 17 ('p') -> 20 ('s') -> 23 ('v') -> 26 ('y')
        dflash.Reset();
        using var engine = new SpeculativeEngine(target, dflash, seed: 11);
        const string promptText = "adgjmpsvyadgjmpsvya";
        var res = await engine.GenerateAsync(
            promptText,
            new SpeculativeOptions { MaxTokens = 7, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal(7, res.Metrics.GeneratedTokens);
        Assert.True(res.SpeculativeMetrics.DraftTokensAccepted > 0, "Expected DFlash draft tokens to be accepted on matching cycle.");

        int[] promptTokens = target.Tokenizer.Encode(promptText);
        target.ResetKvCache();
        target.Prefill(promptTokens);
        int tok = target.SampleNextToken(SamplingOptions.Greedy);
        var expected = new List<int> { tok };
        int pos = promptTokens.Length;
        for (int s = 0; s < 6; s++)
        {
            target.ForwardToken(expected[^1], pos++, computeLogits: true);
            expected.Add(target.SampleNextToken(SamplingOptions.Greedy));
        }

        string expectedText = string.Concat(expected.Select(t => target.Tokenizer.DecodeToken(t)));
        Assert.Equal(expectedText, res.Text);
    }

    [Fact]
    public async Task T2_F7_04_DFlashDraftProvider_EmptyAndSingleTokenPrompt_HandledGracefully()
    {
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        using var provider = new DFlashDraftProvider(trunk, weights: null, new DFlashOptions { BlockSize = 4 });
        int[] buf = new int[4];

        Assert.Equal(0, provider.Draft(ReadOnlySpan<int>.Empty, maxDraftTokens: 4, buf));
        Assert.Equal(0, provider.Draft(new[] { 5 }, maxDraftTokens: 4, buf));

        using var engine = new SpeculativeEngine(trunk, provider, seed: 3);
        var emptyRes = await engine.GenerateAsync("", new SpeculativeOptions { MaxTokens = 8, MaxDraftTokens = 4 }, formatChat: false);
        Assert.Equal("empty_prompt", emptyRes.FinishReason);
        Assert.Equal(0, emptyRes.Metrics.GeneratedTokens);
    }

    [Fact]
    public async Task T2_F7_05_SpeculativeEngine_DFlashEosMidBlock_StopsCleanlyAtEos()
    {
        // Scripted predictions: prefill->5 ('d'), step1->8 ('g'), step2->1 (<eos>), step3->11 ('j')
        // Prompt repeats tokens [5, 8, 1, 11] so DFlash proposes [8, 1, 11] right after first token 5!
        using var target = new ScriptedEosSpeculativeTarget(new[] { 5, 8, 1, 11, 14 }, vocabSize: 32, dim: 32);
        using var dflash = new DFlashDraftProvider(target, weights: null, new DFlashOptions { BlockSize = 4 });
        using var engine = new SpeculativeEngine(target, dflash, seed: 19);

        // Encode characters mapping to token IDs: 'd'=5, 'g'=8, 'j'=11, 'm'=14; use "dgmdgmd" -> prompt ends with 'd'(5)
        // Wait: to seed transition 8 -> 1 (<eos>) -> 11, pass tokens via Draft directly first? Note Engine calls Reset()!
        // Instead, include a known sequence or test EOS at first-token & verify EOS mid-block via scripted tokens:
        // Wait: BpeTokenizer maps "<eos>" to token 1! Let's encode "d g <eos> j d g <eos> j d" or simply check <eos> in prompt:
        int[] checkToks = target.Tokenizer.Encode("<eos>");
        Assert.Contains(1, checkToks);

        const string prompt = "dg<eos>jdg<eos>jd";
        var res = await engine.GenerateAsync(
            prompt,
            new SpeculativeOptions { MaxTokens = 10, MaxDraftTokens = 4, Sampling = SamplingOptions.Greedy },
            formatChat: false);

        Assert.Equal("stop", res.FinishReason);
        Assert.Equal("dg", res.Text);
        // SpeculativeEngine includes the terminal EOS token in GeneratedTokens ([5, 8, 1]), while stopping before token 11
        Assert.Equal(3, res.Metrics.GeneratedTokens);
    }

    // =========================================================================
    // F8 Boundaries: Unsloth Dynamic 3.1 & GGUF Negative Cases (T2-F8-01 .. T2-F8-05)
    // =========================================================================

    [Fact]
    public void T2_F8_01_GgufFile_CorruptMagicAndUnsupportedVersion_ThrowsCleanly()
    {
        using var tmp = new TempDirScope();

        string badMagic = tmp.GetFilePath("bad_magic.gguf");
        SyntheticGgufBuilder.WriteInvalidMagicGguf(badMagic);
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(badMagic));

        string badVersion = tmp.GetFilePath("bad_ver.gguf");
        UpgradeScenarioFixtures.WriteVersionedEmptyGguf(badVersion, version: 999);
        Assert.Throws<NotSupportedException>(() => GgufFile.Open(badVersion));
    }

    [Fact]
    public void T2_F8_02_GgufFile_TruncatedHeaderAndTensorPayload_ThrowsInvalidDataException()
    {
        using var tmp = new TempDirScope();

        string truncHeader = tmp.GetFilePath("trunc_hdr.gguf");
        SyntheticGgufBuilder.WriteTruncatedHeaderGguf(truncHeader);
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(truncHeader));

        string truncPayload = tmp.GetFilePath("trunc_payload.gguf");
        SyntheticGgufBuilder.WriteTruncatedTensorPayloadGguf(truncPayload);
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(truncPayload));
    }

    [Fact]
    public void T2_F8_03_DFlashSidecar_UnalignedIq2XsDimension_RejectedCleanlyWithoutOob()
    {
        using var tmp = new TempDirScope();
        string baseModel = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("base_64.gguf"), dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32);
        string badSidecar = UpgradeScenarioFixtures.WriteUnalignedOrUnknownDFlashSidecarGguf(
            tmp.GetFilePath("unaligned_iq2xs.gguf"), cols: 100, rows: 64, tensorType: GgufType.IQ2_XS, dflashBlockSize: 4);

        using var gguf = GgufFile.Open(baseModel);
        using var weights = new ModelWeights(gguf);
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 64);

        Assert.Throws<InvalidDataException>(() => DFlashDraftProvider.LoadFromSidecar(trunk, weights, badSidecar));
    }

    [Fact]
    public void T2_F8_04_QuantKernels_AllZeroIqBlocks_ReturnExactZeroWithoutNaN()
    {
        const int cols = 256;
        float[] ones = Fx1Vectors.CreateOnes(cols, 1.0f);
        float[] dq = new float[cols];

        var iqTypes = new[]
        {
            GgufType.IQ2_XXS, GgufType.IQ2_XS, GgufType.IQ2_S,
            GgufType.IQ3_XXS, GgufType.IQ3_S, GgufType.IQ4_NL, GgufType.IQ4_XS
        };

        foreach (var type in iqTypes)
        {
            int rowBytes = (int)GgufTypes.GetRowBytes(type, cols);
            byte[] zeros = new byte[rowBytes]; // All zeros => Delta == (Half)0.0

            fixed (byte* pW = zeros)
            fixed (float* pX = ones, pDq = dq)
            {
                Array.Fill(dq, float.NaN);
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

                Assert.All(dq, v => Assert.Equal(0.0f, v));

                float dot = type switch
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

                Assert.False(float.IsNaN(dot));
                Assert.Equal(0.0f, dot);
            }
        }
    }

    [Fact]
    public void T2_F8_05_UnknownTensorTypeId_And_InvalidDFlashBlockSize_RejectedCleanly()
    {
        using var tmp = new TempDirScope();
        string baseModel = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("base_unk.gguf"), dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32);
        using var gguf = GgufFile.Open(baseModel);
        using var weights = new ModelWeights(gguf);
        using var trunk = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 64);

        string unknownTypeSidecar = UpgradeScenarioFixtures.WriteUnalignedOrUnknownDFlashSidecarGguf(
            tmp.GetFilePath("unknown_type.gguf"), cols: 64, rows: 64, tensorType: (GgufType)9999, dflashBlockSize: 4);
        Assert.Throws<InvalidDataException>(() => DFlashDraftProvider.LoadFromSidecar(trunk, weights, unknownTypeSidecar));

        string hugeBlockSidecar = UpgradeScenarioFixtures.WriteUnalignedOrUnknownDFlashSidecarGguf(
            tmp.GetFilePath("huge_block.gguf"), cols: 64, rows: 64, tensorType: GgufType.F32, dflashBlockSize: 99999);
        Assert.Throws<InvalidDataException>(() => DFlashDraftProvider.LoadFromSidecar(trunk, weights, hugeBlockSidecar));
    }

    // =========================================================================
    // F9 Boundaries: KL Divergence Verifier Numerical & Edge Cases (T2-F9-01 .. T2-F9-05)
    // =========================================================================

    [Fact]
    public void T2_F9_01_KlDivergenceVerifier_TranslationInvariancePlus1000_YieldsZeroKlWithoutOverflow()
    {
        float[] refLogits = { 2.0f, 1.0f, 0.0f, -1.0f };
        float[] shifted10 = { 12.0f, 11.0f, 10.0f, 9.0f };
        float[] shifted1000 = { 1002.0f, 1001.0f, 1000.0f, 999.0f };

        var res10 = KlDivergenceVerifier.EvaluateSingleTokenLogits(refLogits, shifted10, threshold: 1e-5);
        Assert.True(res10.Passed);
        Assert.InRange(res10.MeanKlDivergence, 0.0, 1e-6);
        Assert.Equal(1.0, res10.Top1AgreementRate);

        var res1000 = KlDivergenceVerifier.EvaluateSingleTokenLogits(refLogits, shifted1000, threshold: 1e-5);
        Assert.True(res1000.Passed);
        Assert.True(double.IsFinite(res1000.MeanKlDivergence));
        Assert.InRange(res1000.MeanKlDivergence, 0.0, 1e-6);
        Assert.Equal(1.0, res1000.Top1AgreementRate);
    }

    [Fact]
    public void T2_F9_02_KlDivergenceVerifier_SingleTokenPromptVsEmptySequence_BoundaryBehavior()
    {
        using var refTarget = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, perturbation: 0.0f);
        using var candTarget = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32, perturbation: 0.0f);

        var singleRes = KlDivergenceVerifier.VerifyTargets(refTarget, candTarget, new[] { 3 });
        Assert.Equal(1, singleRes.TokenCount);
        Assert.InRange(singleRes.MeanKlDivergence, 0.0, 1e-7);
        Assert.True(singleRes.Passed);

        Assert.Throws<ArgumentException>(() =>
            KlDivergenceVerifier.VerifyTargets(refTarget, candTarget, ReadOnlySpan<int>.Empty));
        Assert.Throws<ArgumentException>(() =>
            KlDivergenceVerifier.EvaluateLogits(ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, vocabSize: 4));
    }

    [Fact]
    public void T2_F9_03_KlDivergenceVerifier_ThresholdBoundary_ExactMatchPassesAndJustBelowFails()
    {
        // FX-4.1c: P = [0.5, 0.25, 0.125, 0.125], Q = [0.25, 0.25, 0.25, 0.25] => exact KL = 0.25 * ln(2)
        float[] refLogits = { MathF.Log(4f), MathF.Log(2f), 0f, 0f };
        float[] quantLogits = { 0f, 0f, 0f, 0f };

        double expectedKl = 0.25 * Math.Log(2.0);
        double kl = KlDivergenceVerifier.ComputeTokenKlDivergence(refLogits, quantLogits);
        Assert.True(Math.Abs(kl - expectedKl) < 1e-5, $"Expected {expectedKl}, got {kl}");

        var passAtExact = KlDivergenceVerifier.EvaluateSingleTokenLogits(
            refLogits, quantLogits, threshold: kl, maxKlThreshold: kl);
        Assert.True(passAtExact.Passed);

        var failBelowMean = KlDivergenceVerifier.EvaluateSingleTokenLogits(
            refLogits, quantLogits, threshold: kl - 1e-6);
        Assert.False(failBelowMean.Passed);

        var failBelowMax = KlDivergenceVerifier.EvaluateSingleTokenLogits(
            refLogits, quantLogits, threshold: 1.0, maxKlThreshold: kl - 1e-6);
        Assert.False(failBelowMax.Passed);

        // Asymmetric distribution to guard against reversed D_KL(Q || P)
        // P = [0.80, 0.15, 0.04, 0.01], Q = [0.25, 0.25, 0.25, 0.25] => KL(P||Q) = 0.74840479 (vs KL(Q||P) = 1.09978303)
        float[] asymRefLogits = { MathF.Log(0.80f), MathF.Log(0.15f), MathF.Log(0.04f), MathF.Log(0.01f) };
        float[] asymQuantLogits = { 0f, 0f, 0f, 0f };
        double expectedAsymKl =
            0.80 * Math.Log(0.80 / 0.25) +
            0.15 * Math.Log(0.15 / 0.25) +
            0.04 * Math.Log(0.04 / 0.25) +
            0.01 * Math.Log(0.01 / 0.25); // 0.74840478748
        double asymKl = KlDivergenceVerifier.ComputeTokenKlDivergence(asymRefLogits, asymQuantLogits);
        Assert.True(Math.Abs(asymKl - expectedAsymKl) < 1e-4,
            $"Asymmetric KL(P||Q) mismatch (reversed KL regression?): expected={expectedAsymKl:F6}, actual={asymKl:F6}");

        Assert.True(KlDivergenceVerifier.EvaluateSingleTokenLogits(asymRefLogits, asymQuantLogits, threshold: expectedAsymKl + 1e-5).Passed);
        Assert.False(KlDivergenceVerifier.EvaluateSingleTokenLogits(asymRefLogits, asymQuantLogits, threshold: expectedAsymKl - 1e-5).Passed);
    }

    [Fact]
    public void T2_F9_04_KlDivergenceVerifier_VocabSizeAndLengthMismatch_ThrowsArgumentException()
    {
        using var target32 = new SyntheticSpeculativeTarget(vocabSize: 32, dim: 32);
        using var target48 = new SyntheticSpeculativeTarget(vocabSize: 48, dim: 32);

        Assert.Throws<ArgumentException>(() =>
            KlDivergenceVerifier.VerifyTargets(target32, target48, new[] { 2, 4 }));

        float[] a = { 1f, 2f, 3f, 4f };
        float[] b = { 1f, 2f, 3f };
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.ComputeTokenKlDivergence(a, b));
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.EvaluateLogits(a, b, vocabSize: 4));
    }

    [Fact]
    public void T2_F9_05_KlDivergenceVerifier_ExtremeNegativeLogitsAndNonFiniteInputs_HandledSafely()
    {
        // FX-4.1e: Near-zero probabilities clamp cleanly without NaN / +Inf
        float[] refExtreme = { 100.0f, -100.0f, -100.0f, -100.0f };
        float[] quantExtreme = { 100.0f, -1000.0f, -1000.0f, -1000.0f };
        var resExtreme = KlDivergenceVerifier.EvaluateSingleTokenLogits(refExtreme, quantExtreme, threshold: 0.05);
        Assert.True(double.IsFinite(resExtreme.MeanKlDivergence));
        Assert.False(resExtreme.HasNonFiniteQuantizedValues);
        Assert.True(resExtreme.Passed);

        // NaN / +Inf in reference logits throws ArgumentException
        float[] refNan = { float.NaN, 0f, 0f, 0f };
        float[] refInf = { float.PositiveInfinity, 0f, 0f, 0f };
        float[] valid = { 1f, 0f, 0f, 0f };
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.EvaluateSingleTokenLogits(refNan, valid));
        Assert.Throws<ArgumentException>(() => KlDivergenceVerifier.EvaluateSingleTokenLogits(refInf, valid));

        // NaN / +Inf in quantized candidate fails gate explicitly with HasNonFiniteQuantizedValues == true
        var nanCandRes = KlDivergenceVerifier.EvaluateSingleTokenLogits(valid, refNan);
        Assert.False(nanCandRes.Passed);
        Assert.True(nanCandRes.HasNonFiniteQuantizedValues);

        var infCandRes = KlDivergenceVerifier.EvaluateSingleTokenLogits(valid, refInf);
        Assert.False(infCandRes.Passed);
        Assert.True(infCandRes.HasNonFiniteQuantizedValues);
    }
}
