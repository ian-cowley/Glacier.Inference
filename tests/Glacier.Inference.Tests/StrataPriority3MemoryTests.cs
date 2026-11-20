namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Memory;
using Xunit;

public sealed class StrataPriority3MemoryTests : IDisposable
{
    private readonly string _tempDir;

    public StrataPriority3MemoryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierStrataP3_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, true);
        }
        catch
        {
        }
    }

    [Fact]
    public void ExpertRoutingProfile_DecayTopKAndBinaryPersistence_RoundTripsAccurately()
    {
        var profile = new ExpertRoutingProfile(numLayers: 2, numExperts: 16, decayFactor: 0.5, adaptWindowSteps: 4);

        // Step 1..3 accumulate score before window decay at step 4
        profile.RecordActivations(layerIdx: 0, [5, 11], [0.8f, 0.4f]);
        profile.RecordActivations(layerIdx: 0, [5], [0.8f]);
        profile.RecordActivations(layerIdx: 1, [3, 9], [0.6f, 0.2f]);
        // Step 4 triggers 0.5 decay across all scores
        profile.RecordActivations(layerIdx: 0, [5, 2], [0.8f, 0.3f]);

        Assert.Equal(4, profile.StepCount);
        Assert.Equal(3, profile.GetTotalActivations(0, 5));
        // (0.8 + 0.8 + 0.8) * 0.5 = 1.2
        Assert.InRange(profile.GetScore(0, 5), 1.199, 1.201);

        Span<int> topLayer0 = stackalloc int[4];
        int written0 = profile.GetTopExpertsForLayer(0, topLayer0);
        Assert.Equal(3, written0);
        Assert.Equal(5, topLayer0[0]);
        Assert.Equal(11, topLayer0[1]);
        Assert.Equal(2, topLayer0[2]);

        Span<(int LayerIndex, int ExpertIndex, double Score)> topPairs = stackalloc (int, int, double)[5];
        int pairCount = profile.GetTopLayerExpertPairs(topPairs);
        Assert.Equal(5, pairCount);
        Assert.Equal((0, 5), (topPairs[0].LayerIndex, topPairs[0].ExpertIndex));
        Assert.Equal((1, 3), (topPairs[1].LayerIndex, topPairs[1].ExpertIndex));

        string path = Path.Combine(_tempDir, "routing.gmoe.profile");
        profile.Save(path);
        Assert.True(File.Exists(path));

        var reloaded = ExpertRoutingProfile.Load(path);
        Assert.Equal(profile.NumLayers, reloaded.NumLayers);
        Assert.Equal(profile.NumExperts, reloaded.NumExperts);
        Assert.Equal(profile.StepCount, reloaded.StepCount);
        Assert.Equal(profile.GetScore(0, 5), reloaded.GetScore(0, 5), precision: 9);
        Assert.Equal(profile.GetTotalActivations(1, 3), reloaded.GetTotalActivations(1, 3));

        Assert.True(ExpertRoutingProfile.TryLoad(path, 2, 16, out var tryLoaded));
        Assert.NotNull(tryLoaded);
        Assert.False(ExpertRoutingProfile.TryLoad(path, 2, 32, out var mismatch));
        Assert.Null(mismatch);
    }

    [Fact]
    public void GpuResidentSlotManager_DecayedFrequencyEviction_And_ProtectedSet_NeverSelfEvicts()
    {
        var profile = new ExpertRoutingProfile(numLayers: 2, numExperts: 8, decayFactor: 0.99, adaptWindowSteps: 100);
        for (int i = 0; i < 6; i++)
            profile.RecordActivations(0, [0], [1.0f]); // Exp 0 very hot
        profile.RecordActivations(0, [1], [1.0f]);     // Exp 1 warm
        profile.RecordActivations(0, [2], [0.2f]);     // Exp 2 coldest

        using var manager = new GpuResidentSlotManager(
            slotCount: 3,
            slotByteSize: 4096,
            allocateVram: false,
            evictionPolicy: SlotEvictionPolicy.DecayedFrequency,
            routingProfile: profile);

        manager.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
        manager.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
        manager.AllocateOrEvictSlot(0, 2, out _, out _, out _, out _);

        // Even though exp 0 was inserted earliest (oldest timestamp), DecayedFrequency must evict exp 2 (lowest score)
        profile.RecordActivations(0, [3], [0.9f]);
        manager.AllocateOrEvictSlot(0, 3, [0, 1], out _, out int evL, out int evE, out _);
        Assert.Equal((0, 2), (evL, evE));
        Assert.True(manager.ContainsSlot(0, 0));
        Assert.True(manager.ContainsSlot(0, 1));
        Assert.True(manager.ContainsSlot(0, 3));

        // Now give expert 2 (cold/evicted) huge surge on layer 1 and verify protected set shields [1] even if [1] is colder than [3]
        // Currently active: 0 (score 6.0), 1 (score 1.0), 3 (score 0.9)
        // Protect [0, 3] so 1 must be evicted when inserting 4
        profile.RecordActivations(0, [4], [0.5f]);
        manager.AllocateOrEvictSlot(0, 4, [0, 3], out _, out evL, out evE, out _);
        Assert.Equal((0, 1), (evL, evE));
        Assert.True(manager.ContainsSlot(0, 3));

        // Promote a non-resident hotter expert (1, 7) via PlanAdaptiveSwaps (beats coldest slot 0:4 with score 0.5)
        for (int i = 0; i < 5; i++)
            profile.RecordActivations(1, [7], [1.0f]);

        Span<(int SlotIndex, int NewLayer, int NewExpert, int EvictedLayer, int EvictedExpert)> swaps = stackalloc (int, int, int, int, int)[2];
        int swapCount = manager.PlanAdaptiveSwaps(profile, maxSwaps: 2, minScoreAdvantageRatio: 1.25, swaps);
        Assert.Equal(1, swapCount);
        Assert.Equal((1, 7), (swaps[0].NewLayer, swaps[0].NewExpert));
        Assert.Equal((0, 4), (swaps[0].EvictedLayer, swaps[0].EvictedExpert));
        Assert.True(manager.ContainsSlot(1, 7));

        // Elastic slot resizing down to 1 and 0, then back up to 3
        Span<int> cleared = stackalloc int[4];
        int invalidated = manager.SetActiveSlotLimit(1, cleared);
        Assert.Equal(2, invalidated);
        Assert.Equal(1, manager.ActiveSlotCount);
        Assert.Equal(1, manager.OccupiedSlotCount);

        invalidated = manager.SetActiveSlotLimit(0);
        Assert.Equal(1, invalidated);
        Assert.Equal(0, manager.ActiveSlotCount);
        Assert.Equal(0, manager.OccupiedSlotCount);

        manager.SetActiveSlotLimit(3);
        Assert.Equal(3, manager.ActiveSlotCount);
        Assert.True(manager.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _));
    }

    [Fact]
    public void WddmVramGuard_CalculatesWindowsCliffReserve_And_RecommendsSafeSlotCounts()
    {
        const ulong GiB = 1024UL * 1024 * 1024;
        const ulong MiB = 1024UL * 1024;

        Assert.Equal(384UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(3UL * GiB, isWindowsWddm: true));
        Assert.Equal(640UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(6UL * GiB, isWindowsWddm: true));
        Assert.Equal(896UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(8UL * GiB, isWindowsWddm: true));
        Assert.Equal(1024UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(16UL * GiB, isWindowsWddm: true));
        Assert.Equal(1536UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(24UL * GiB, isWindowsWddm: true));
        Assert.Equal(256UL * MiB, WddmVramGuard.CalculateSafetyReserveBytes(16UL * GiB, isWindowsWddm: false));

        ulong slotSize = 64UL * MiB;
        // 8 GiB card with 896 MiB WDDM reserve => safe ceiling is 7296 MiB.
        // At 7500 MiB current usage, cliff risk is active and ~4 slots must be dropped.
        var cliffSnap = WddmVramGuard.EvaluateHeadroom(
            dedicatedVramBytes: 8UL * GiB,
            osBudgetBytes: 8UL * GiB,
            currentUsageBytes: 7500UL * MiB,
            isWindowsWddm: true);

        Assert.True(cliffSnap.IsAtCliffRisk);
        Assert.Equal(0UL, cliffSnap.SafeAvailableBytes);
        int shrunk = WddmVramGuard.ComputeRecommendedSlotCount(cliffSnap, slotSize, currentActiveSlots: 64, maxSlotCapacity: 64, minSlotFloor: 4);
        Assert.InRange(shrunk, 58, 61);

        // Plenty of free VRAM allows expanding back toward max capacity
        var healthySnap = WddmVramGuard.EvaluateHeadroom(
            dedicatedVramBytes: 8UL * GiB,
            osBudgetBytes: 8UL * GiB,
            currentUsageBytes: 4096UL * MiB,
            isWindowsWddm: true);

        Assert.False(healthySnap.IsAtCliffRisk);
        int grown = WddmVramGuard.ComputeRecommendedSlotCount(healthySnap, slotSize, currentActiveSlots: 32, maxSlotCapacity: 64, minSlotFloor: 4);
        Assert.Equal(64, grown);
    }

    [Fact]
    public unsafe void TieredVirtualMemoryManager_ExclusiveHostRamBudget_EnforcesZeroDuplicationWithVram()
    {
        string filePath = Path.Combine(_tempDir, "exclusive_budget.gmoe");
        const int layers = 2;
        const int experts = 8;
        MmapWeightBank.CreateFile(filePath, layers, experts, expertRawBytes: 2048, pageAlignment: 4096,
            initializer: (l, e, span) => span.Fill((byte)((l << 4) + e + 1)));

        var profile = new ExpertRoutingProfile(layers, experts);
        // Rank order: (0,0)>(0,1)>(0,2) [VRAM top 3] > (1,0)>(1,1)>(1,2) [Host RAM next 3] > remaining cold
        for (int i = 0; i < 10; i++) profile.RecordActivations(0, [0], [1.0f]);
        for (int i = 0; i < 9; i++) profile.RecordActivations(0, [1], [1.0f]);
        for (int i = 0; i < 8; i++) profile.RecordActivations(0, [2], [1.0f]);
        for (int i = 0; i < 6; i++) profile.RecordActivations(1, [0], [1.0f]);
        for (int i = 0; i < 5; i++) profile.RecordActivations(1, [1], [1.0f]);
        for (int i = 0; i < 4; i++) profile.RecordActivations(1, [2], [1.0f]);

        using var bank = new MmapWeightBank(filePath);
        var options = new TieredMemoryOptions
        {
            VramSlotCount = 3,
            PinnedBufferCount = 2,
            PreferCuda = false,
            HostRamResidentBudgetBytes = 3L * bank.ExpertStrideBytes,
            ExclusiveHostRamResidency = true,
            EvictionPolicy = SlotEvictionPolicy.DecayedFrequency,
            RoutingProfile = profile,
            MaxConcurrentStoragePageFaults = 2
        };

        using var manager = new TieredVirtualMemoryManager(bank, options);
        Assert.Equal(3, manager.MaxHostRamResidentSlots);

        void AssertExclusiveInvariant()
        {
            Assert.InRange(manager.HostRamResidentCount, 0, manager.MaxHostRamResidentSlots);
            for (int l = 0; l < layers; l++)
            {
                for (int e = 0; e < experts; e++)
                {
                    Assert.False(
                        manager.VramManager.ContainsSlot(l, e) && manager.IsHostRamResident(l, e),
                        $"Expert ({l},{e}) duplicated across VRAM and Host RAM tiers.");
                }
            }
        }

        // 1. Warm tiers strictly without overlap
        manager.WarmFromProfile(profile);
        AssertExclusiveInvariant();
        Assert.True(manager.VramManager.ContainsSlot(0, 0));
        Assert.True(manager.VramManager.ContainsSlot(0, 1));
        Assert.True(manager.VramManager.ContainsSlot(0, 2));
        Assert.True(manager.IsHostRamResident(1, 0));
        Assert.True(manager.IsHostRamResident(1, 1));
        Assert.True(manager.IsHostRamResident(1, 2));

        // 2. Host-RAM resident hit returns MemoryTier.HostRam
        var hostHit = manager.AccessExpert(1, 0, targetGpu: false);
        Assert.Equal(MemoryTier.HostRam, hostHit.Tier);
        Assert.True(hostHit.IsPinned);
        Assert.Equal((byte)((1 << 4) + 1), hostHit.AsHostSpan()[0]);

        // 3. Cold Tier-3 expert returns MemoryTier.Storage on first page fault, then MemoryTier.HostRam once promoted
        var storageMiss = manager.AccessExpert(1, 7, targetGpu: false);
        Assert.Equal(MemoryTier.Storage, storageMiss.Tier);
        AssertExclusiveInvariant();
        Assert.True(manager.IsHostRamResident(1, 7));

        var hostWarmHit = manager.AccessExpert(1, 7, targetGpu: false);
        Assert.Equal(MemoryTier.HostRam, hostWarmHit.Tier);

        // 4. Surge score on (1, 7) and promote via RunAdaptivePass -> evicted VRAM expert demotes to Host RAM
        for (int i = 0; i < 12; i++)
            profile.RecordActivations(1, [7], [1.0f]);
        int swaps = manager.RunAdaptivePass(maxSwaps: 2, minScoreAdvantageRatio: 1.1);
        Assert.True(swaps >= 1);
        Assert.True(manager.VramManager.ContainsSlot(1, 7));
        AssertExclusiveInvariant();

        var tel = manager.GetTelemetry();
        Assert.Equal(3, tel.ActiveVramSlots);
        Assert.True(tel.AdaptiveSwapCount >= 1);
        Assert.Equal(3, tel.MaxHostRamResidentSlots);
    }
}
