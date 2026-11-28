namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Xunit;

[Collection("SequentialGpu")]
public sealed unsafe class TieredMoERunnerM2Tests
{
    private sealed class CapturingLogger : IGlacierLogger
    {
        public readonly List<string> Messages = new();
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
    public void HostRamExpertArena_IsLayerResident_TracksPerLayerBudget()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_arena_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 3,
                dim: 32,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: 16,
                expertCount: 4,
                expertUsedCount: 2,
                seed: 42);

            using var gguf = GgufFile.Open(tempPath);
            using var weights = new ModelWeights(gguf);

            // Each layer has 3 expert tensors (gate, up, down) of 4 * 32 * 32 * 4 = 16,384 bytes.
            // HostRamExpertArena aligns each tensor (+4096 padding) up to 4096-byte pages -> 20,480 bytes * 3 = 61,440 bytes per layer.
            // Budget of 130,000 bytes fits 2 layers cleanly (122,880 <= 130,000 < 184,320).
            byte* origL0 = weights.Layers[0].FfnGateExpsWeight;
            byte* origL1 = weights.Layers[1].FfnGateExpsWeight;
            byte* origL2 = weights.Layers[2].FfnGateExpsWeight;

            HostRamExpertArena? arena = weights.AttachExpertArena(130_000UL, preferUnbufferedIo: false);
            Assert.NotNull(arena);
            Assert.Equal(3, arena!.TotalMoeLayers);
            Assert.Equal(2, arena.ResidentMoeLayers);
            Assert.True(arena.IsLayerResident(0));
            Assert.True(arena.IsLayerResident(1));
            Assert.False(arena.IsLayerResident(2));
            Assert.False(arena.IsLayerResident(-1));
            Assert.False(arena.IsLayerResident(99));

            Assert.NotEqual((nint)origL0, (nint)weights.Layers[0].FfnGateExpsWeight);
            Assert.NotEqual((nint)origL1, (nint)weights.Layers[1].FfnGateExpsWeight);
            Assert.Equal((nint)origL2, (nint)weights.Layers[2].FfnGateExpsWeight);
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_EndToEndSyntheticGguf_MatchesCpuReferenceAndPreservesM1ReserveZero()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_e2e_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 6,
                expertUsedCount: 2,
                seed: 101);

            int[] prompt = [1, 2, 3];
            int[] verifyTokens = [4, 5];
            float[] refPrefillLogits = new float[vocabSize];
            float[] refVerifyLogits = new float[verifyTokens.Length * vocabSize];

            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch(prompt, 0, refPrefillLogits, refKv, computeLogits: true);
                refModel.VerifyBatchLogits(verifyTokens, prompt.Length, refVerifyLogits, vocabSize, refKv);
            }

            var options = new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.Auto,
                MaxSequenceLength = seqLen,
                HostArenaBudgetBytes = 2UL * 1024 * 1024,
                VramExpertSlotCount = 4,
                VramReserveMb = 0,
                KvPrecision = KvCachePrecision.Fp32,
                RouteTailSkipMinRank = 0
            };

            using var runner = new TieredMoERunner(tempPath, options);
            Assert.True(runner.IsTieredActive);
            Assert.Equal(0UL, runner.EffectiveVramReserveBytes);
            Assert.Equal(0UL, runner.QueryVramHeadroom().SafetyReserveBytes);
            Assert.NotNull(runner.HostArena);
            Assert.True(runner.HostArena!.IsLayerResident(0));
            Assert.True(runner.HostArena.IsLayerResident(1));

            runner.Prefill(prompt);
            float[] prefillLogits = new float[vocabSize];
            runner.CopyCurrentLogits(prefillLogits);
            for (int v = 0; v < vocabSize; v++)
            {
                Assert.True(MathF.Abs(refPrefillLogits[v] - prefillLogits[v]) < 1e-2f,
                    $"Prefill logit mismatch at {v}: ref={refPrefillLogits[v]} actual={prefillLogits[v]}");
            }

            var prefillTelemetry = runner.LastStepTelemetry;
            Assert.NotNull(prefillTelemetry);
            Assert.Equal(0.0, prefillTelemetry!.ExpertsSkippedPercent);
            Assert.Equal(0, prefillTelemetry.SkippedRoutedExperts);
            Assert.True(prefillTelemetry.TotalRoutedExperts > 0);

            float[] verifyLogits = new float[verifyTokens.Length * vocabSize];
            runner.VerifyBatchLogits(verifyTokens, prompt.Length, verifyLogits);
            for (int i = 0; i < verifyLogits.Length; i++)
            {
                Assert.True(MathF.Abs(refVerifyLogits[i] - verifyLogits[i]) < 1e-2f,
                    $"Verify logit mismatch at {i}: ref={refVerifyLogits[i]} actual={verifyLogits[i]}");
            }

            var verifyTelemetry = runner.LastStepTelemetry;
            Assert.NotNull(verifyTelemetry);
            Assert.Equal(0.0, verifyTelemetry!.ExpertsSkippedPercent);
            Assert.Equal(2, runner.TelemetryHistory.Count);
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_EndToEndSyntheticGguf_PoliciesAndSpeculativeTailSkip()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_pol_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 8,
                expertUsedCount: 4,
                seed: 202);

            int[] prompt = [1, 2];
            int[] draft = [3, 4, 5];
            float[] refVerifyLogits = new float[draft.Length * vocabSize];

            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch(prompt, 0, new float[vocabSize], refKv, computeLogits: true);
                refModel.VerifyBatchLogits(draft, prompt.Length, refVerifyLogits, vocabSize, refKv);
            }

            // Test HybridGpuHostRam + GpuSharedMemory (including VramExpertSlotCount=0 PCIe Upload-heap path)
            foreach (var (policy, slotCount) in new[]
            {
                (TieringExecutionPolicy.HybridGpuHostRam, 2),
                (TieringExecutionPolicy.GpuSharedMemory, 0),
                (TieringExecutionPolicy.GpuSharedMemory, 4)
            })
            {
                using var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
                {
                    Policy = policy,
                    MaxSequenceLength = seqLen,
                    HostArenaBudgetBytes = 2UL * 1024 * 1024,
                    VramExpertSlotCount = slotCount,
                    KvPrecision = KvCachePrecision.Fp32,
                    RouteTailSkipMinRank = 0
                });

                runner.Prefill(prompt);
                float[] verifyLogits = new float[draft.Length * vocabSize];
                runner.VerifyBatchLogits(draft, prompt.Length, verifyLogits);
                AssertArraysClose(refVerifyLogits, verifyLogits, 1e-2f);

                var t = runner.LastStepTelemetry!;
                Assert.Equal(policy, t.ResolvedPolicy);
                Assert.Equal(0.0, t.ExpertsSkippedPercent);
                if (policy == TieringExecutionPolicy.HybridGpuHostRam)
                {
                    Assert.Equal(0, t.PcieTransferredExperts);
                    Assert.True(t.CpuExecutedExperts > 0);
                }
                else
                {
                    Assert.True(t.PcieTransferredExperts > 0 || t.CpuExecutedExperts > 0);
                }
            }

            // Test RouteTailSkipMinRank > 0 on end-to-end GGUF VerifyBatch vs Prefill with deterministic routing
            string skipPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_skip_{Guid.NewGuid():N}.gguf");
            try
            {
                WriteSyntheticMoeGguf(
                    skipPath,
                    blockCount: 2,
                    dim: dim,
                    ffnDim: 32,
                    expertFfnDim: 32,
                    nHeads: 2,
                    nHeadsKv: 1,
                    vocabSize: vocabSize,
                    expertCount: 8,
                    expertUsedCount: 4,
                    seed: 203,
                    deterministicRouter: true);

                using var skipRunner = new TieredMoERunner(skipPath, new TieredRunnerOptions
                {
                    Policy = TieringExecutionPolicy.HybridGpuHostRam,
                    MaxSequenceLength = seqLen,
                    HostArenaBudgetBytes = 2UL * 1024 * 1024,
                    VramExpertSlotCount = 0,
                    KvPrecision = KvCachePrecision.Fp32,
                    RouteTailSkipMinRank = 2
                });

                skipRunner.Prefill(prompt);
                Assert.Equal(0, skipRunner.LastStepTelemetry!.SkippedRoutedExperts);
                Assert.Equal(0.0, skipRunner.LastStepTelemetry!.ExpertsSkippedPercent);

                int[] predicted = new int[draft.Length];
                skipRunner.VerifyBatch(draft, prompt.Length, predicted);
                var verifyTelemetry = skipRunner.LastStepTelemetry!;
                Assert.True(verifyTelemetry.TotalRoutedExperts > 0);
                Assert.True(verifyTelemetry.SkippedRoutedExperts > 0, $"Expected SkippedRoutedExperts > 0, got {verifyTelemetry.SkippedRoutedExperts}");
                Assert.True(verifyTelemetry.ExpertsSkippedPercent > 0.0, $"Expected ExpertsSkippedPercent > 0, got {verifyTelemetry.ExpertsSkippedPercent}");
            }
            finally
            {
                TryDelete(skipPath);
            }
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    public void TieredMoERunner_DynamicMissSplitter_SplitsAcrossHybridGpuHostRam_GpuSharedMemory_And_Auto()
    {
        const int dim = 32;
        const int expertFfnDim = 32;
        const int numExperts = 8;
        const int topK = 4;

        using var fixture = SyntheticMoeLayerFixture.Create(layerCount: 1, dim, expertFfnDim, numExperts, topK, seed: 7);

        // Configure router weights so token 0 selects {0,1,2,3} and token 1 selects {2,3,4,5}
        // -> 6 unique experts: {0,1,2,3,4,5}
        fixture.SetRouterPreferredExperts(0, new[]
        {
            (0, new[] { 0, 1, 2, 3 }, new[] { 10f, 8f, 6f, 4f }),
            (1, new[] { 2, 3, 4, 5 }, new[] { 10f, 8f, 6f, 4f })
        });

        float[] batchX = fixture.BuildBasisPairInput(dim);
        float[] refOut = new float[2 * dim];
        fixture.ComputeExactCpuReference(0, batchX, 2, topK, normTopK: true, refOut);

        // 1. HybridGpuHostRam: VRAM hits {0, 1} stay on GPU, all 4 misses {2,3,4,5} run on CPU HostRamExpertArena
        {
            var profile = new ExpertRoutingProfile(1, numExperts);
            using var slotMgr = new GpuResidentSlotManager(6, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
            slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
            slotMgr.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
            using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, allocatedBytes: 65536);
            var coSched = new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0);

            using var runner = new TieredMoERunner(
                fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                profile, coSched, slotMgr, arena,
                new TieredRunnerOptions { Policy = TieringExecutionPolicy.HybridGpuHostRam });

            float[] actualOut = new float[2 * dim];
            var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, batchSize: 2, isSpeculativeDraftBatch: false);

            Assert.Equal(2, split.VramHitUniqueCount);
            Assert.Equal(0, split.PcieTransferUniqueCount);
            Assert.Equal(4, split.CpuExecuteUniqueCount);
            Assert.Equal(TieringExecutionPolicy.HybridGpuHostRam, split.ResolvedPolicy);
            AssertArraysClose(refOut, actualOut, 1e-4f);
        }

        // 2. GpuSharedMemory with 3 free slots (total 5 slots, 2 hits protected): 3 misses go PCIe, 1 overflow spills to CPU
        {
            var profile = new ExpertRoutingProfile(1, numExperts);
            // Pre-warm expert 4 so it ranks ahead of expert 5 when picking the 3 PCIe slots
            profile.RecordActivations(0, new[] { 4 }, new[] { 5.0f });
            using var slotMgr = new GpuResidentSlotManager(5, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
            slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
            slotMgr.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
            using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, allocatedBytes: 65536);
            var coSched = new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0);

            using var runner = new TieredMoERunner(
                fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                profile, coSched, slotMgr, arena,
                new TieredRunnerOptions { Policy = TieringExecutionPolicy.GpuSharedMemory });

            float[] actualOut = new float[2 * dim];
            var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, batchSize: 2, isSpeculativeDraftBatch: false);

            Assert.Equal(2, split.VramHitUniqueCount);
            Assert.Equal(3, split.PcieTransferUniqueCount);
            Assert.Equal(1, split.CpuExecuteUniqueCount);
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(5, out var exp5Plan));
            Assert.Equal(ExpertExecutionDisposition.CpuExecute, exp5Plan.Disposition);
            Assert.True(slotMgr.ContainsSlot(0, 0));
            Assert.True(slotMgr.ContainsSlot(0, 1));
            AssertArraysClose(refOut, actualOut, 1e-4f);
        }

        // 3. Auto: BandwidthCoScheduler dynamically splits misses (q* hottest to PCIe, m - q* coldest to CPU)
        {
            var profile = new ExpertRoutingProfile(1, numExperts);
            // Pre-warm expert 4 as hottest miss, expert 5 as cold miss
            profile.RecordActivations(0, new[] { 4 }, new[] { 20.0f });
            profile.RecordActivations(0, new[] { 2 }, new[] { 10.0f });
            using var slotMgr = new GpuResidentSlotManager(8, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
            slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
            slotMgr.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
            using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, allocatedBytes: 65536);
            var coSched = new BandwidthCoScheduler(initialPcieGBps: 25.0, initialCpuGBps: 25.0, vramGBps: 272.0);
            int expectedQ = coSched.CalculateOptimalGpuExpertCount(missingCount: 4, residentHitCount: 2);
            Assert.InRange(expectedQ, 1, 3);

            using var runner = new TieredMoERunner(
                fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                profile, coSched, slotMgr, arena,
                new TieredRunnerOptions { Policy = TieringExecutionPolicy.Auto });

            float[] actualOut = new float[2 * dim];
            var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, batchSize: 2, isSpeculativeDraftBatch: false);

            Assert.Equal(2, split.VramHitUniqueCount);
            Assert.Equal(expectedQ, split.PcieTransferUniqueCount);
            Assert.Equal(4 - expectedQ, split.CpuExecuteUniqueCount);
            Assert.True(split.IsHostArenaLayerResident);

            // Hottest miss (expert 4) must be promoted via PCIe; coldest miss (expert 5) must run on CPU
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(4, out var exp4Plan));
            Assert.Equal(ExpertExecutionDisposition.PcieTransfer, exp4Plan.Disposition);
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(5, out var exp5Plan));
            Assert.Equal(ExpertExecutionDisposition.CpuExecute, exp5Plan.Disposition);

            var bwTelemetry = coSched.GetTelemetry();
            Assert.True(bwTelemetry.PcieSamplesRecorded > 0);
            Assert.True(bwTelemetry.CpuSamplesRecorded > 0);
            AssertArraysClose(refOut, actualOut, 1e-4f);
        }

        // 4. Auto with non-resident HostRamExpertArena layer shifts more misses to PCIe than resident layer
        {
            var profile = new ExpertRoutingProfile(1, numExperts);
            using var slotMgr = new GpuResidentSlotManager(8, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
            slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
            slotMgr.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
            using var nonResidentArena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { false }, allocatedBytes: 0);
            var coSched = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);
            Assert.Equal(0, coSched.CalculateOptimalGpuExpertCount(4, 2)); // Resident DDR5 would pick 0 PCIe

            using var runner = new TieredMoERunner(
                fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                profile, coSched, slotMgr, nonResidentArena,
                new TieredRunnerOptions { Policy = TieringExecutionPolicy.Auto });

            float[] actualOut = new float[2 * dim];
            var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, batchSize: 2, isSpeculativeDraftBatch: false);

            Assert.False(split.IsHostArenaLayerResident);
            Assert.True(split.PcieTransferUniqueCount > 0, "Cold non-resident mmap layer should shift misses toward PCIe.");
        }
    }

    [Fact]
    public void TieredMoERunner_IntraBatchProtection_NeverEvictsSameBatchVramHit()
    {
        const int dim = 32;
        const int expertFfnDim = 32;
        const int numExperts = 8;
        const int topK = 2;

        using var fixture = SyntheticMoeLayerFixture.Create(1, dim, expertFfnDim, numExperts, topK, seed: 13);
        fixture.SetRouterPreferredExperts(0, new[]
        {
            (0, new[] { 0, 1 }, new[] { 10f, 7f }),
            (1, new[] { 2, 3 }, new[] { 10f, 7f })
        });

        var profile = new ExpertRoutingProfile(1, numExperts);
        using var slotMgr = new GpuResidentSlotManager(2, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
        slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
        slotMgr.AllocateOrEvictSlot(0, 2, out _, out _, out _, out _);

        using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, 65536);
        var coSched = new BandwidthCoScheduler(initialPcieGBps: 50.0, initialCpuGBps: 10.0, vramGBps: 272.0);

        using var runner = new TieredMoERunner(
            fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
            profile, coSched, slotMgr, arena,
            new TieredRunnerOptions { Policy = TieringExecutionPolicy.GpuSharedMemory });

        float[] batchX = fixture.BuildBasisPairInput(dim);
        float[] actualOut = new float[2 * dim];
        var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, batchSize: 2, isSpeculativeDraftBatch: false);

        Assert.Equal(2, split.VramHitUniqueCount);
        Assert.Equal(0, split.PcieTransferUniqueCount);
        Assert.Equal(2, split.CpuExecuteUniqueCount);
        Assert.Equal(0, slotMgr.EvictionCount);
        Assert.True(slotMgr.ContainsSlot(0, 0));
        Assert.True(slotMgr.ContainsSlot(0, 2));
    }

    [Fact]
    public void TieredMoERunner_RouteTailSkipMinRank_DefaultsOff_AndSkipsUniversalTailVramMissesOnDraftBatch()
    {
        var defaultOpts = new TieredRunnerOptions();
        Assert.Equal(0, defaultOpts.RouteTailSkipMinRank);

        defaultOpts.RouteTailSkipMinRank = -5;
        Assert.Equal(0, defaultOpts.RouteTailSkipMinRank);

        const int dim = 32;
        const int expertFfnDim = 32;
        const int numExperts = 8;
        const int topK = 4;

        using var fixture = SyntheticMoeLayerFixture.Create(1, dim, expertFfnDim, numExperts, topK, seed: 29);

        // 3-token batch:
        // Token 0 -> [E0(r0), E1(r1), E2(r2), E3(r3)]
        // Token 1 -> [E4(r0), E2(r1), E5(r2), E6(r3)]
        // Token 2 -> [E0(r0), E7(r1), E5(r2), E3(r3)]
        // With E3 resident in VRAM and RouteTailSkipMinRank = 2:
        // - E3 is a VRAM hit -> NEVER skipped despite minRank = 3
        // - E2 appears at r2 on tok0 AND r1 < 2 on tok1 -> NEVER skipped
        // - E5 appears only at r2 >= 2 (tok1, tok2) and is a VRAM miss -> SKIPPED (2 slots)
        // - E6 appears only at r3 >= 2 (tok1) and is a VRAM miss -> SKIPPED (1 slot)
        // Total routed slots = 12, skipped = 3 -> ExpertsSkippedPercent = 25.0%
        fixture.SetRouterPreferredExperts(0, new[]
        {
            (0, new[] { 0, 1, 2, 3 }, new[] { 10f, 8f, 6f, 4f }),
            (1, new[] { 4, 2, 5, 6 }, new[] { 10f, 8f, 6f, 4f }),
            (2, new[] { 0, 7, 5, 3 }, new[] { 10f, 8f, 6f, 4f })
        });

        float[] batchX = fixture.BuildBasisTripleInput(dim);
        var logger = new CapturingLogger();
        var prevLogger = GlacierDiagnostics.Logger;
        GlacierDiagnostics.Logger = logger;

        try
        {
            var profile = new ExpertRoutingProfile(1, numExperts);
            using var slotMgr = new GpuResidentSlotManager(4, 1024, allocateVram: false, SlotEvictionPolicy.DecayedFrequency, profile);
            slotMgr.AllocateOrEvictSlot(0, 3, out _, out _, out _, out _);

            using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, 65536);
            var coSched = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);

            var observedTelemetry = new List<TieredStepTelemetry>();
            using var runner = new TieredMoERunner(
                fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                profile, coSched, slotMgr, arena,
                new TieredRunnerOptions
                {
                    Policy = TieringExecutionPolicy.HybridGpuHostRam,
                    RouteTailSkipMinRank = 2,
                    OnStepTelemetry = t => observedTelemetry.Add(t)
                });

            // 1. Non-speculative prefill batch must NEVER tail-skip even when RouteTailSkipMinRank > 0
            float[] prefillOut = new float[3 * dim];
            var prefillSplit = runner.ExecuteSyntheticMoELayerBatch(0, batchX, prefillOut, batchSize: 3, isSpeculativeDraftBatch: false);
            Assert.Equal(0, prefillSplit.SkippedRoutedSlots);
            Assert.Equal(0.0, prefillSplit.ExpertsSkippedPercent);
            Assert.Equal(0.0, runner.LastStepTelemetry!.ExpertsSkippedPercent);

            // 2. Speculative draft batch skips E5 (2 occurrences) + E6 (1 occurrence) = 3/12 = 25.0%
            float[] draftOut = new float[3 * dim];
            var draftSplit = runner.ExecuteSyntheticMoELayerBatch(0, batchX, draftOut, batchSize: 3, isSpeculativeDraftBatch: true);

            Assert.Equal(12, draftSplit.TotalRoutedSlots);
            Assert.Equal(3, draftSplit.SkippedRoutedSlots);
            Assert.Equal(25.0, draftSplit.ExpertsSkippedPercent, precision: 6);

            Assert.True(runner.MissSplitter.TryGetPlannedExpert(3, out var e3));
            Assert.Equal(ExpertExecutionDisposition.VramHit, e3.Disposition);
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(2, out var e2));
            Assert.NotEqual(ExpertExecutionDisposition.Skipped, e2.Disposition);
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(5, out var e5));
            Assert.Equal(ExpertExecutionDisposition.Skipped, e5.Disposition);
            Assert.True(runner.MissSplitter.TryGetPlannedExpert(6, out var e6));
            Assert.Equal(ExpertExecutionDisposition.Skipped, e6.Disposition);

            var telemetry = runner.LastStepTelemetry!;
            Assert.Equal(12, telemetry.TotalRoutedExperts);
            Assert.Equal(3, telemetry.SkippedRoutedExperts);
            Assert.Equal(25.0, telemetry.ExpertsSkippedPercent, precision: 6);
            Assert.Equal(2, observedTelemetry.Count);
            Assert.Contains(logger.Messages, m => m.Contains("25.0% experts skipped", StringComparison.Ordinal));

            // Verify mathematical equivalence to reference with {5, 6} masked and kept weights renormalized
            float[] expectedDraftOut = new float[3 * dim];
            fixture.ComputeExactCpuReference(0, batchX, 3, topK, normTopK: true, expectedDraftOut, skippedExperts: new HashSet<int> { 5, 6 });
            AssertArraysClose(expectedDraftOut, draftOut, 1e-4f);
        }
        finally
        {
            GlacierDiagnostics.Logger = prevLogger;
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_D3D12_SharedExpertGate_PrefillForwardAndVerifyBatchLogits_NoEFail()
    {
        if (!D3D12Context.IsSupported)
            return;

        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_shexpgate_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 6,
                expertUsedCount: 2,
                seed: 303,
                includeSharedExpert: true,
                includeShexpGate: true);

            int[] prompt = [1, 2, 3];
            int[] draft = [4, 5];
            float[] refPrefill = new float[vocabSize];
            float[] refVerify = new float[draft.Length * vocabSize];
            float[] refToken = new float[vocabSize];

            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch(prompt, 0, refPrefill, refKv, computeLogits: true);
                refModel.VerifyBatchLogits(draft, prompt.Length, refVerify, vocabSize, refKv);
                refModel.Forward(6, prompt.Length + draft.Length, refKv, refToken, computeLogits: true);
            }

            // Tiered MoE path (both Prefill/VerifyBatchLogits and single-token ForwardToken)
            using (var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.Auto,
                MaxSequenceLength = seqLen,
                HostArenaBudgetBytes = 2UL * 1024 * 1024,
                VramExpertSlotCount = 2,
                KvPrecision = KvCachePrecision.Fp32
            }))
            {
                runner.Prefill(prompt);
                float[] actualPrefill = new float[vocabSize];
                runner.CopyCurrentLogits(actualPrefill);
                AssertArraysClose(refPrefill, actualPrefill, 1e-2f);

                float[] actualVerify = new float[draft.Length * vocabSize];
                runner.VerifyBatchLogits(draft, prompt.Length, actualVerify);
                AssertArraysClose(refVerify, actualVerify, 1e-2f);

                runner.ForwardToken(6, prompt.Length + draft.Length, computeLogits: true);
                float[] actualToken = new float[vocabSize];
                runner.CopyCurrentLogits(actualToken);
                AssertArraysClose(refToken, actualToken, 1e-2f);
            }

            // Non-tiered Qwen2D3D12Model path with ffn_gate_inp_shexp.weight (lines 318 & 982)
            using (var gguf = GgufFile.Open(tempPath))
            using (var weights = new ModelWeights(gguf))
            using (var ctx = new D3D12Context(0))
            using (var model = new Qwen2D3D12Model(ctx, weights, seqLen, KvCachePrecision.Fp32, tieredMoe: false))
            {
                float[] actualPrefill = new float[vocabSize];
                model.ForwardBatch(prompt, 0, actualPrefill, computeLogits: true);
                AssertArraysClose(refPrefill, actualPrefill, 1e-2f);

                float[] actualVerify = new float[draft.Length * vocabSize];
                model.VerifyBatchLogits(draft, prompt.Length, actualVerify, vocabSize);
                AssertArraysClose(refVerify, actualVerify, 1e-2f);

                float[] actualToken = new float[vocabSize];
                model.Forward(6, prompt.Length + draft.Length, actualToken, computeLogits: true);
                AssertArraysClose(refToken, actualToken, 1e-2f);
            }
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_FusedGateUp_DefaultAutoOptions_AndHybridDenseFirstLayer_MatchesCpuReference()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_fused_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            // Layer 0 = dense, Layer 1 = fused GateUp MoE -> tests both fused GateUp under default Auto AND MoE layer scanning
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 6,
                expertUsedCount: 2,
                seed: 404,
                fusedGateUp: true,
                denseFirstLayer: true);

            int[] prompt = [1, 2, 3];
            int[] draft = [4, 5];
            float[] refPrefill = new float[vocabSize];
            float[] refVerify = new float[draft.Length * vocabSize];
            float[] refToken = new float[vocabSize];

            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch(prompt, 0, refPrefill, refKv, computeLogits: true);
                refModel.VerifyBatchLogits(draft, prompt.Length, refVerify, vocabSize, refKv);
                refModel.Forward(6, prompt.Length + draft.Length, refKv, refToken, computeLogits: true);
            }

            // Default TieredRunnerOptions (Policy = Auto, VramExpertSlotCount = null, HostArenaBudgetBytes = null)
            using var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                MaxSequenceLength = seqLen
            });
            Assert.True(runner.IsTieredActive);

            runner.Prefill(prompt);
            float[] actualPrefill = new float[vocabSize];
            runner.CopyCurrentLogits(actualPrefill);
            AssertArraysClose(refPrefill, actualPrefill, 1e-2f);

            float[] actualVerify = new float[draft.Length * vocabSize];
            runner.VerifyBatchLogits(draft, prompt.Length, actualVerify);
            AssertArraysClose(refVerify, actualVerify, 1e-2f);

            runner.ForwardToken(6, prompt.Length + draft.Length, computeLogits: true);
            float[] actualToken = new float[vocabSize];
            runner.CopyCurrentLogits(actualToken);
            AssertArraysClose(refToken, actualToken, 1e-2f);
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_BF16Experts_D3D12AutoAndGpuSharedMemory_MatchesCpuReference()
    {
        if (!D3D12Context.IsSupported)
            return;

        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_bf16_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 6,
                expertUsedCount: 2,
                seed: 505,
                useBf16Experts: true);

            int[] prompt = [1, 2, 3];
            int[] draft = [4, 5];
            float[] refPrefill = new float[vocabSize];
            float[] refVerify = new float[draft.Length * vocabSize];

            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch(prompt, 0, refPrefill, refKv, computeLogits: true);
                refModel.VerifyBatchLogits(draft, prompt.Length, refVerify, vocabSize, refKv);
            }

            foreach (var (policy, slots) in new[]
            {
                (TieringExecutionPolicy.GpuSharedMemory, 0),
                (TieringExecutionPolicy.GpuSharedMemory, 2),
                (TieringExecutionPolicy.Auto, 2)
            })
            {
                using var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
                {
                    Policy = policy,
                    MaxSequenceLength = seqLen,
                    HostArenaBudgetBytes = 2UL * 1024 * 1024,
                    VramExpertSlotCount = slots,
                    KvPrecision = KvCachePrecision.Fp32
                });

                runner.Prefill(prompt);
                float[] actualPrefill = new float[vocabSize];
                runner.CopyCurrentLogits(actualPrefill);
                AssertArraysClose(refPrefill, actualPrefill, 2e-2f);

                float[] actualVerify = new float[draft.Length * vocabSize];
                runner.VerifyBatchLogits(draft, prompt.Length, actualVerify);
                AssertArraysClose(refVerify, actualVerify, 2e-2f);

                var bf16Tel = runner.LastStepTelemetry!;
                Assert.True(bf16Tel.TotalRoutedExperts > 0);
                if (policy == TieringExecutionPolicy.GpuSharedMemory)
                {
                    Assert.True(bf16Tel.PcieTransferredExperts + bf16Tel.VramHitExperts > 0);
                }
                else
                {
                    Assert.True(bf16Tel.PcieTransferredExperts + bf16Tel.VramHitExperts + bf16Tel.CpuExecutedExperts > 0);
                }
            }
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void TieredMoERunner_VerifyBatchLogits_OneTokenAndNineTokens_HonorsRouteTailSkipMinRank()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_1and9_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 8,
                expertUsedCount: 4,
                seed: 606,
                deterministicRouter: true);

            // 1. 1-token VerifyBatchLogits must honor RouteTailSkipMinRank
            using (var runner1 = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.HybridGpuHostRam,
                MaxSequenceLength = seqLen,
                VramExpertSlotCount = 0,
                KvPrecision = KvCachePrecision.Fp32,
                RouteTailSkipMinRank = 2
            }))
            {
                runner1.Prefill([1, 2]);
                Assert.Equal(0, runner1.LastStepTelemetry!.SkippedRoutedExperts);

                float[] logits1 = new float[vocabSize];
                runner1.VerifyBatchLogits([3], 2, logits1);
                var t1 = runner1.LastStepTelemetry!;
                Assert.Equal(8, t1.TotalRoutedExperts);   // 1 tok * 2 layers * topK=4
                Assert.Equal(4, t1.SkippedRoutedExperts); // 1 tok * 2 layers * 2 tail experts
                Assert.Equal(50.0, t1.ExpertsSkippedPercent, precision: 6);
            }

            // 2. 9-token (8k + 1) VerifyBatchLogits must run all 9 tokens with tail skipping (36/72 = 50%)
            int[] batch9 = [3, 4, 5, 6, 7, 8, 9, 10, 11];
            float[] refLogits9 = new float[batch9.Length * vocabSize];
            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch([1, 2], 0, new float[vocabSize], refKv, computeLogits: true);
                refModel.VerifyBatchLogits(batch9, 2, refLogits9, vocabSize, refKv);
            }

            using (var runner0 = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.Auto,
                MaxSequenceLength = seqLen,
                VramExpertSlotCount = 2,
                KvPrecision = KvCachePrecision.Fp32,
                RouteTailSkipMinRank = 0
            }))
            {
                runner0.Prefill([1, 2]);
                float[] actualLogits9 = new float[batch9.Length * vocabSize];
                runner0.VerifyBatchLogits(batch9, 2, actualLogits9);
                AssertArraysClose(refLogits9, actualLogits9, 1e-2f);
                Assert.Equal(72, runner0.LastStepTelemetry!.TotalRoutedExperts);
                Assert.Equal(0, runner0.LastStepTelemetry!.SkippedRoutedExperts);
            }

            using (var runner9 = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.HybridGpuHostRam,
                MaxSequenceLength = seqLen,
                VramExpertSlotCount = 0,
                KvPrecision = KvCachePrecision.Fp32,
                RouteTailSkipMinRank = 2
            }))
            {
                runner9.Prefill([1, 2]);
                float[] logits9 = new float[batch9.Length * vocabSize];
                runner9.VerifyBatchLogits(batch9, 2, logits9);

                var t9 = runner9.LastStepTelemetry!;
                Assert.Equal(72, t9.TotalRoutedExperts);   // 9 tok * 2 layers * topK=4
                Assert.Equal(36, t9.SkippedRoutedExperts); // All 9 tokens skipped 2 tail experts per layer
                Assert.Equal(50.0, t9.ExpertsSkippedPercent, precision: 6);
            }
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    public void TieredMoERunner_CpuFallback_WiresMissSplitterTailSkipAndTelemetry()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_cpufallback_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int dim = 32;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 2,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: 32,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 8,
                expertUsedCount: 4,
                seed: 707,
                denseFirstLayer: true,
                deterministicRouter: true);

            using var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                GpuAdapterIndex = -2,
                Policy = TieringExecutionPolicy.HybridGpuHostRam,
                MaxSequenceLength = seqLen,
                HostArenaBudgetBytes = 2UL * 1024 * 1024,
                VramExpertSlotCount = 2,
                RouteTailSkipMinRank = 2
            });

            Assert.Equal("CPU-TieredMoE", runner.DeviceDescription);

            runner.Prefill([1, 2]);
            var prefillT = runner.LastStepTelemetry!;
            Assert.Equal(8, prefillT.TotalRoutedExperts); // 2 tok * 1 MoE layer * topK=4
            Assert.Equal(0, prefillT.SkippedRoutedExperts);
            Assert.Equal(0.0, prefillT.ExpertsSkippedPercent);
            Assert.True(prefillT.CpuExecutedExperts > 0);

            // Both 1-token and multi-token VerifyBatchLogits honor RouteTailSkipMinRank on CPU fallback
            float[] verify1 = new float[vocabSize];
            runner.VerifyBatchLogits([3], 2, verify1);
            var verify1T = runner.LastStepTelemetry!;
            Assert.Equal(4, verify1T.TotalRoutedExperts);
            Assert.Equal(2, verify1T.SkippedRoutedExperts);
            Assert.Equal(50.0, verify1T.ExpertsSkippedPercent, precision: 6);
            Assert.True(verify1T.CpuExecutedExperts > 0);

            float[] verify3 = new float[3 * vocabSize];
            runner.VerifyBatchLogits([3, 4, 5], 3, verify3);
            var verify3T = runner.LastStepTelemetry!;
            Assert.Equal(12, verify3T.TotalRoutedExperts);
            Assert.Equal(6, verify3T.SkippedRoutedExperts);
            Assert.Equal(50.0, verify3T.ExpertsSkippedPercent, precision: 6);
            Assert.True(runner.CoScheduler.GetTelemetry().CpuSamplesRecorded > 0);

            runner.ForwardToken(6, 6, computeLogits: true);
            var tokT = runner.LastStepTelemetry!;
            Assert.Equal(4, tokT.TotalRoutedExperts);
            Assert.Equal(0, tokT.SkippedRoutedExperts);
            Assert.True(tokT.CpuExecutedExperts > 0);
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public async Task TieredMoERunner_EdgeCases_TailSkipGeTopK_NormTopKFalse_BiasedLayer_And_DisposeIdempotency()
    {
        // 1. RouteTailSkipMinRank >= topK disables tail skipping (0% skipped) & NormTopK = false preserves unnormalized weights
        const int dim = 32;
        const int expertFfnDim = 32;
        const int numExperts = 8;
        const int topK = 4;

        using (var fixture = SyntheticMoeLayerFixture.Create(1, dim, expertFfnDim, numExperts, topK, seed: 808))
        {
            fixture.SetRouterPreferredExperts(0, new[]
            {
                (0, new[] { 0, 1, 2, 3 }, new[] { 10f, 8f, 6f, 4f }),
                (1, new[] { 0, 1, 4, 5 }, new[] { 10f, 8f, 6f, 4f })
            });
            float[] batchX = fixture.BuildBasisPairInput(dim);

            // RouteTailSkipMinRank == topK (4) -> 0 skipped
            {
                var profile = new ExpertRoutingProfile(1, numExperts);
                using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, 65536);
                var coSched = new BandwidthCoScheduler(12.5, 65.0, 272.0);
                using var runner = new TieredMoERunner(
                    fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: true,
                    profile, coSched, null, arena,
                    new TieredRunnerOptions { Policy = TieringExecutionPolicy.HybridGpuHostRam, RouteTailSkipMinRank = topK });

                float[] outBuf = new float[2 * dim];
                var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, outBuf, 2, isSpeculativeDraftBatch: true);
                Assert.Equal(0, split.SkippedRoutedSlots);
                Assert.Equal(0.0, split.ExpertsSkippedPercent);
            }

            // NormTopK = false with RouteTailSkipMinRank = 2 -> kept weights must NOT be renormalized
            {
                var profile = new ExpertRoutingProfile(1, numExperts);
                using var arena = HostRamExpertArena.CreateSyntheticForTesting(1, new[] { true }, 65536);
                var coSched = new BandwidthCoScheduler(12.5, 65.0, 272.0);
                using var runner = new TieredMoERunner(
                    fixture.Layers, dim, expertFfnDim, numExperts, topK, vocabSize: 16, normTopK: false,
                    profile, coSched, null, arena,
                    new TieredRunnerOptions { Policy = TieringExecutionPolicy.HybridGpuHostRam, RouteTailSkipMinRank = 2 });

                float[] actualOut = new float[2 * dim];
                var split = runner.ExecuteSyntheticMoELayerBatch(0, batchX, actualOut, 2, isSpeculativeDraftBatch: true);
                Assert.Equal(4, split.SkippedRoutedSlots); // {2, 3, 4, 5} skipped -> 4/8 = 50%

                float[] expectedUnnorm = new float[2 * dim];
                fixture.ComputeExactCpuReference(0, batchX, 2, topK, normTopK: false, expectedUnnorm, skippedExperts: new HashSet<int> { 2, 3, 4, 5 });
                AssertArraysClose(expectedUnnorm, actualOut, 1e-4f);
            }
        }

        // 2. Biased MoE layer on D3D12 must not mutate _slotManager or report bogus GPU/PCIe hits, AND Dispose idempotency + ObjectDisposedException
        string tempPath = Path.Combine(Path.GetTempPath(), $"glacier_m2_bias_disp_{Guid.NewGuid():N}.gguf");
        try
        {
            const int vocabSize = 16;
            const int seqLen = 64;
            WriteSyntheticMoeGguf(
                tempPath,
                blockCount: 1,
                dim: dim,
                ffnDim: 32,
                expertFfnDim: expertFfnDim,
                nHeads: 2,
                nHeadsKv: 1,
                vocabSize: vocabSize,
                expertCount: 6,
                expertUsedCount: 2,
                seed: 909,
                includeExpertBiases: true);

            float[] refLogits = new float[vocabSize];
            using (var refGguf = GgufFile.Open(tempPath))
            using (var refWeights = new ModelWeights(refGguf))
            using (var refKv = new KVCache(refWeights.BlockCount, refWeights.HeadCountKv, refWeights.HeadDim, seqLen))
            using (var refModel = new Qwen2Model(refWeights, seqLen))
            {
                refModel.ForwardBatch([1, 2], 0, refLogits, refKv, computeLogits: true);
            }

            var runner = new TieredMoERunner(tempPath, new TieredRunnerOptions
            {
                Policy = TieringExecutionPolicy.GpuSharedMemory,
                MaxSequenceLength = seqLen,
                VramExpertSlotCount = 2
            });

            runner.Prefill([1, 2]);
            float[] actualLogits = new float[vocabSize];
            runner.CopyCurrentLogits(actualLogits);
            AssertArraysClose(refLogits, actualLogits, 1e-2f);

            var t = runner.LastStepTelemetry!;
            Assert.Equal(0, t.VramHitExperts);
            Assert.Equal(0, t.PcieTransferredExperts);
            Assert.True(t.CpuExecutedExperts > 0);
            if (runner.SlotManager != null)
            {
                Assert.Equal(0, runner.SlotManager.OccupiedSlotCount);
                Assert.Equal(0, runner.SlotManager.HitCount);
                Assert.True(runner.SlotManager.MissCount >= 0);
                Assert.Equal(0, runner.SlotManager.EvictionCount);
            }

            // TelemetryHistory must be read-only (cannot mutate by casting to List<TieredStepTelemetry>)
            Assert.False(runner.TelemetryHistory is List<TieredStepTelemetry>);

            // Double Dispose idempotency + ObjectDisposedException on all public execution methods
            runner.Dispose();
            runner.Dispose();

            Assert.Throws<ObjectDisposedException>(() => runner.Prefill([1]));
            Assert.Throws<ObjectDisposedException>(() => runner.ForwardToken(1, 0, true));
            Assert.Throws<ObjectDisposedException>(() => runner.VerifyBatch([1], 0, new int[1]));
            Assert.Throws<ObjectDisposedException>(() => runner.VerifyBatchLogits([1], 0, new float[vocabSize]));
            Assert.Throws<ObjectDisposedException>(() => runner.GenerateStreamAsync("tok1", 1));
            await Assert.ThrowsAsync<ObjectDisposedException>(() => runner.GenerateAsync("tok1", 1));
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    private static void AssertArraysClose(float[] expected, float[] actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"Mismatch at index {i}: expected={expected[i]}, actual={actual[i]}");
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch
        {
            // Ignore transient test cleanup lock
        }
    }

    private sealed class SyntheticMoeLayerFixture : IDisposable
    {
        private readonly List<IntPtr> _allocs = new();
        private readonly int _dim;
        private readonly int _expertFfnDim;
        private readonly int _numExperts;
        private readonly int _topK;

        public LayerWeights[] Layers { get; }

        private SyntheticMoeLayerFixture(LayerWeights[] layers, int dim, int expertFfnDim, int numExperts, int topK)
        {
            Layers = layers;
            _dim = dim;
            _expertFfnDim = expertFfnDim;
            _numExperts = numExperts;
            _topK = topK;
        }

        public static SyntheticMoeLayerFixture Create(int layerCount, int dim, int expertFfnDim, int numExperts, int topK, int seed)
        {
            var rng = new Random(seed);
            var fixture = new SyntheticMoeLayerFixture(new LayerWeights[layerCount], dim, expertFfnDim, numExperts, topK);

            for (int l = 0; l < layerCount; l++)
            {
                float* routerW = fixture.AllocFloats(numExperts * dim);
                float* gateW = fixture.AllocFloats(numExperts * expertFfnDim * dim);
                float* upW = fixture.AllocFloats(numExperts * expertFfnDim * dim);
                float* downW = fixture.AllocFloats(numExperts * dim * expertFfnDim);
                float* dummyAttnOut = fixture.AllocFloats(dim * dim);

                FillRandom(rng, gateW, numExperts * expertFfnDim * dim, 0.15f);
                FillRandom(rng, upW, numExperts * expertFfnDim * dim, 0.15f);
                FillRandom(rng, downW, numExperts * dim * expertFfnDim, 0.15f);

                fixture.Layers[l] = new LayerWeights
                {
                    AttnOutWeight = (byte*)dummyAttnOut,
                    AttnOutType = GgufType.F32,
                    IsMoe = true,
                    FfnGateInpWeight = routerW,
                    FfnGateExpsWeight = (byte*)gateW,
                    FfnGateExpsType = GgufType.F32,
                    FfnUpExpsWeight = (byte*)upW,
                    FfnUpExpsType = GgufType.F32,
                    FfnDownExpsWeight = (byte*)downW,
                    FfnDownExpsType = GgufType.F32
                };
            }

            return fixture;
        }

        public void SetRouterPreferredExperts(int layerIdx, (int FeatureCol, int[] Experts, float[] Logits)[] specs)
        {
            float* routerW = Layers[layerIdx].FfnGateInpWeight;
            new Span<float>(routerW, _numExperts * _dim).Fill(-20f);
            foreach (var (col, experts, logits) in specs)
            {
                for (int i = 0; i < experts.Length; i++)
                {
                    routerW[experts[i] * _dim + col] = logits[i];
                }
            }
        }

        public float[] BuildBasisPairInput(int dim)
        {
            float[] x = new float[2 * dim];
            x[0] = 1.0f;
            x[1] = 0.25f;
            x[dim + 1] = 1.0f;
            x[dim + 2] = 0.25f;
            for (int d = 3; d < dim; d++)
            {
                x[d] = 0.05f * ((d % 5) - 2);
                x[dim + d] = -0.05f * ((d % 5) - 2);
            }
            return x;
        }

        public float[] BuildBasisTripleInput(int dim)
        {
            float[] x = new float[3 * dim];
            x[0] = 1.0f;
            x[dim + 1] = 1.0f;
            x[2 * dim + 2] = 1.0f;
            for (int d = 3; d < dim; d++)
            {
                x[d] = 0.04f * (d % 4);
                x[dim + d] = -0.04f * (d % 4);
                x[2 * dim + d] = 0.03f * ((d + 1) % 4);
            }
            return x;
        }

        public void ComputeExactCpuReference(
            int layerIdx,
            float[] batchX,
            int batchSize,
            int topK,
            bool normTopK,
            float[] outBatch,
            HashSet<int>? skippedExperts = null)
        {
            var lw = Layers[layerIdx];
            float[] gate = new float[_expertFfnDim];
            float[] up = new float[_expertFfnDim];
            float[] act = new float[_expertFfnDim];
            float[] actSums = new float[(_expertFfnDim + 31) / 32];
            float[] downOut = new float[_dim];
            float[] normSums = new float[(_dim + 31) / 32];
            int[] idx = new int[topK];
            float[] wt = new float[topK];

            fixed (float* pX = batchX, pOut = outBatch, pGate = gate, pUp = up, pAct = act, pActSums = actSums, pDown = downOut, pNormSums = normSums, pWt = wt)
            fixed (int* pIdx = idx)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    float* normX = pX + t * _dim;
                    float* dst = pOut + t * _dim;
                    new Span<float>(dst, _dim).Clear();
                    QuantKernels.ComputeBlockSums32(normX, pNormSums, _dim);
                    QuantKernels.RouterTopK(normX, lw.FfnGateInpWeight, null, _dim, _numExperts, topK, pIdx, pWt, normTopK);

                    if (skippedExperts != null && skippedExperts.Count > 0)
                    {
                        float keptSum = 0f;
                        bool anySkipped = false;
                        for (int k = 0; k < topK; k++)
                        {
                            if (skippedExperts.Contains(idx[k]))
                            {
                                idx[k] = -1;
                                wt[k] = 0f;
                                anySkipped = true;
                            }
                            else
                            {
                                keptSum += wt[k];
                            }
                        }
                        if (anySkipped && normTopK && keptSum > 0f)
                        {
                            for (int k = 0; k < topK; k++)
                            {
                                if (idx[k] >= 0) wt[k] /= keptSum;
                            }
                        }
                    }

                    for (int k = 0; k < topK; k++)
                    {
                        TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                            lw, idx[k], wt[k], normX, pNormSums, dst, _dim, _expertFfnDim, pGate, pUp, pAct, pActSums, pDown);
                    }
                }
            }
        }

        private float* AllocFloats(int count)
        {
            float* ptr = (float*)NativeMemory.AllocZeroed((nuint)count * sizeof(float));
            _allocs.Add((IntPtr)ptr);
            return ptr;
        }

        private static void FillRandom(Random rng, float* ptr, int count, float scale)
        {
            for (int i = 0; i < count; i++)
            {
                ptr[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
            }
        }

        public void Dispose()
        {
            foreach (IntPtr ptr in _allocs)
            {
                if (ptr != IntPtr.Zero) NativeMemory.Free((void*)ptr);
            }
            _allocs.Clear();
        }
    }

    private static void WriteSyntheticMoeGguf(
        string path,
        int blockCount,
        int dim,
        int ffnDim,
        int expertFfnDim,
        int nHeads,
        int nHeadsKv,
        int vocabSize,
        int expertCount,
        int expertUsedCount,
        int seed,
        bool fusedGateUp = false,
        bool useBf16Experts = false,
        bool includeSharedExpert = false,
        bool includeShexpGate = false,
        bool includeExpertBiases = false,
        bool denseFirstLayer = false,
        bool deterministicRouter = false)
    {
        var rng = new Random(seed);
        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        var kvEntries = new List<(string Key, Action<BinaryWriter> WriteVal)>
        {
            ("general.architecture", w => WriteGgufStringVal(w, "qwen2moe")),
            ("general.alignment", w => WriteGgufUInt32Val(w, 32)),
            ("qwen2moe.block_count", w => WriteGgufUInt32Val(w, (uint)blockCount)),
            ("qwen2moe.context_length", w => WriteGgufUInt32Val(w, 256)),
            ("qwen2moe.embedding_length", w => WriteGgufUInt32Val(w, (uint)dim)),
            ("qwen2moe.feed_forward_length", w => WriteGgufUInt32Val(w, (uint)ffnDim)),
            ("qwen2moe.attention.head_count", w => WriteGgufUInt32Val(w, (uint)nHeads)),
            ("qwen2moe.attention.head_count_kv", w => WriteGgufUInt32Val(w, (uint)nHeadsKv)),
            ("qwen2moe.attention.key_length", w => WriteGgufUInt32Val(w, (uint)headDim)),
            ("qwen2moe.attention.value_length", w => WriteGgufUInt32Val(w, (uint)headDim)),
            ("qwen2moe.attention.layer_norm_rms_epsilon", w => WriteGgufFloat32Val(w, 1e-5f)),
            ("qwen2moe.rope.freq_base", w => WriteGgufFloat32Val(w, 10000.0f)),
            ("qwen2moe.expert_count", w => WriteGgufUInt32Val(w, (uint)expertCount)),
            ("qwen2moe.expert_used_count", w => WriteGgufUInt32Val(w, (uint)expertUsedCount)),
            ("qwen2moe.expert_feed_forward_length", w => WriteGgufUInt32Val(w, (uint)expertFfnDim)),
            ("qwen2moe.expert_weights_norm", w => WriteGgufBoolVal(w, true)),
            ("tokenizer.ggml.tokens", w =>
            {
                w.Write(9u); // GGUF_TYPE_ARRAY
                w.Write(8u); // element type STRING
                w.Write((ulong)vocabSize);
                for (int i = 0; i < vocabSize; i++)
                {
                    WriteGgufString(w, $"tok{i}");
                }
            }),
            ("tokenizer.ggml.eos_token_id", w => WriteGgufUInt32Val(w, (uint)(vocabSize - 1))),
            ("tokenizer.ggml.bos_token_id", w => WriteGgufUInt32Val(w, 0u))
        };

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, float[] Data)>();

        float[] MakeNorm(int len)
        {
            var a = new float[len];
            Array.Fill(a, 1.0f);
            return a;
        }

        float[] MakeRand(int len, float scale = 0.1f)
        {
            var a = new float[len];
            for (int i = 0; i < len; i++)
            {
                a[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
            }
            return a;
        }

        float[] MakeDeterministicRouter(int expCount, int d)
        {
            // Keep weights at 0 and drive deterministic descending expert rankings purely via router bias
            // so arbitrary residual states across layers/positions always rank [0, 1, 2, 3, ...].
            return new float[expCount * d];
        }

        float[] embdData = MakeRand(vocabSize * dim, 0.25f);
        if (deterministicRouter)
        {
            // Ensure feature 0 of every token embedding is strongly positive after RMSNorm
            for (int v = 0; v < vocabSize; v++)
            {
                embdData[v * dim] = 2.0f;
            }
        }

        tensors.Add(("token_embd.weight", new ulong[] { (ulong)dim, (ulong)vocabSize }, GgufType.F32, embdData));
        tensors.Add(("output_norm.weight", new ulong[] { (ulong)dim }, GgufType.F32, MakeNorm(dim)));
        tensors.Add(("output.weight", new ulong[] { (ulong)dim, (ulong)vocabSize }, GgufType.F32, MakeRand(vocabSize * dim, 0.2f)));

        GgufType expType = useBf16Experts ? GgufType.BF16 : GgufType.F32;

        for (int l = 0; l < blockCount; l++)
        {
            tensors.Add(($"blk.{l}.attn_norm.weight", new ulong[] { (ulong)dim }, GgufType.F32, MakeNorm(dim)));
            tensors.Add(($"blk.{l}.attn_q.weight", new ulong[] { (ulong)dim, (ulong)qDim }, GgufType.F32, MakeRand(qDim * dim)));
            tensors.Add(($"blk.{l}.attn_k.weight", new ulong[] { (ulong)dim, (ulong)kvDim }, GgufType.F32, MakeRand(kvDim * dim)));
            tensors.Add(($"blk.{l}.attn_v.weight", new ulong[] { (ulong)dim, (ulong)kvDim }, GgufType.F32, MakeRand(kvDim * dim)));
            tensors.Add(($"blk.{l}.attn_output.weight", new ulong[] { (ulong)qDim, (ulong)dim }, GgufType.F32, MakeRand(dim * qDim)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", new ulong[] { (ulong)dim }, GgufType.F32, MakeNorm(dim)));

            if (denseFirstLayer && l == 0)
            {
                tensors.Add(($"blk.{l}.ffn_gate.weight", new ulong[] { (ulong)dim, (ulong)ffnDim }, GgufType.F32, MakeRand(ffnDim * dim)));
                tensors.Add(($"blk.{l}.ffn_up.weight", new ulong[] { (ulong)dim, (ulong)ffnDim }, GgufType.F32, MakeRand(ffnDim * dim)));
                tensors.Add(($"blk.{l}.ffn_down.weight", new ulong[] { (ulong)ffnDim, (ulong)dim }, GgufType.F32, MakeRand(dim * ffnDim)));
                continue;
            }

            float[] routerWeights = deterministicRouter
                ? MakeDeterministicRouter(expertCount, dim)
                : MakeRand(expertCount * dim, 0.4f);
            tensors.Add(($"blk.{l}.ffn_gate_inp.weight", new ulong[] { (ulong)dim, (ulong)expertCount }, GgufType.F32, routerWeights));
            if (deterministicRouter)
            {
                float[] routerBias = new float[expertCount];
                for (int e = 0; e < expertCount; e++) routerBias[e] = 20.0f - e * 4.0f;
                tensors.Add(($"blk.{l}.ffn_gate_inp.bias", new ulong[] { (ulong)expertCount }, GgufType.F32, routerBias));
            }

            if (fusedGateUp)
            {
                tensors.Add(($"blk.{l}.ffn_gate_up_exps.weight", new ulong[] { (ulong)dim, (ulong)(2 * expertFfnDim), (ulong)expertCount }, expType, MakeRand(expertCount * 2 * expertFfnDim * dim)));
            }
            else
            {
                tensors.Add(($"blk.{l}.ffn_gate_exps.weight", new ulong[] { (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount }, expType, MakeRand(expertCount * expertFfnDim * dim)));
                tensors.Add(($"blk.{l}.ffn_up_exps.weight", new ulong[] { (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount }, expType, MakeRand(expertCount * expertFfnDim * dim)));
            }
            tensors.Add(($"blk.{l}.ffn_down_exps.weight", new ulong[] { (ulong)expertFfnDim, (ulong)dim, (ulong)expertCount }, expType, MakeRand(expertCount * dim * expertFfnDim)));

            if (includeExpertBiases)
            {
                tensors.Add(($"blk.{l}.ffn_gate_exps.bias", new ulong[] { (ulong)expertFfnDim, (ulong)expertCount }, GgufType.F32, MakeRand(expertCount * expertFfnDim, 0.05f)));
                tensors.Add(($"blk.{l}.ffn_up_exps.bias", new ulong[] { (ulong)expertFfnDim, (ulong)expertCount }, GgufType.F32, MakeRand(expertCount * expertFfnDim, 0.05f)));
                tensors.Add(($"blk.{l}.ffn_down_exps.bias", new ulong[] { (ulong)dim, (ulong)expertCount }, GgufType.F32, MakeRand(expertCount * dim, 0.05f)));
            }

            if (includeSharedExpert)
            {
                int shexpFfnDim = expertFfnDim;
                tensors.Add(($"blk.{l}.ffn_gate_shexp.weight", new ulong[] { (ulong)dim, (ulong)shexpFfnDim }, GgufType.F32, MakeRand(shexpFfnDim * dim)));
                tensors.Add(($"blk.{l}.ffn_up_shexp.weight", new ulong[] { (ulong)dim, (ulong)shexpFfnDim }, GgufType.F32, MakeRand(shexpFfnDim * dim)));
                tensors.Add(($"blk.{l}.ffn_down_shexp.weight", new ulong[] { (ulong)shexpFfnDim, (ulong)dim }, GgufType.F32, MakeRand(dim * shexpFfnDim)));
                if (includeShexpGate)
                {
                    tensors.Add(($"blk.{l}.ffn_gate_inp_shexp.weight", new ulong[] { (ulong)dim }, GgufType.F32, MakeRand(dim, 0.25f)));
                }
            }
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // Version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvEntries.Count);

        foreach (var (key, writeVal) in kvEntries)
        {
            WriteGgufString(bw, key);
            writeVal(bw);
        }

        ulong currentOffset = 0;
        foreach (var (name, dims, type, data) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++)
                bw.Write(dims[d]);
            bw.Write((uint)type);
            bw.Write(currentOffset);

            ulong elemSize = type == GgufType.BF16 ? (ulong)sizeof(ushort) : sizeof(float);
            ulong byteLen = (ulong)data.Length * elemSize;
            currentOffset = AlignUp(currentOffset + byteLen, 32UL);
        }

        long pos = fs.Position;
        long alignedDataStart = (long)AlignUp((ulong)pos, 32UL);
        while (fs.Position < alignedDataStart)
        {
            bw.Write((byte)0);
        }

        foreach (var (_, _, type, data) in tensors)
        {
            ulong byteLen;
            if (type == GgufType.BF16)
            {
                ushort[] bf16 = new ushort[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    bf16[i] = (ushort)(BitConverter.SingleToUInt32Bits(data[i]) >> 16);
                }
                ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(bf16.AsSpan());
                bw.Write(bytes);
                byteLen = (ulong)bytes.Length;
            }
            else
            {
                ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(data.AsSpan());
                bw.Write(bytes);
                byteLen = (ulong)bytes.Length;
            }

            long rem = (long)(AlignUp(byteLen, 32UL) - byteLen);
            for (long p = 0; p < rem; p++)
            {
                bw.Write((byte)0);
            }
        }
    }

    private static ulong AlignUp(ulong val, ulong alignment) => (val + alignment - 1UL) & ~(alignment - 1UL);

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static void WriteGgufStringVal(BinaryWriter bw, string s)
    {
        bw.Write(8u); // STRING
        WriteGgufString(bw, s);
    }

    private static void WriteGgufUInt32Val(BinaryWriter bw, uint val)
    {
        bw.Write(4u); // UINT32
        bw.Write(val);
    }

    private static void WriteGgufFloat32Val(BinaryWriter bw, float val)
    {
        bw.Write(6u); // FLOAT32
        bw.Write(val);
    }

    private static void WriteGgufBoolVal(BinaryWriter bw, bool val)
    {
        bw.Write(7u); // BOOL
        bw.Write(val ? (byte)1 : (byte)0);
    }
}
