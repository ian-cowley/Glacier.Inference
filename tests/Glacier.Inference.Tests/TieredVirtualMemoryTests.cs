namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Memory;
using Glacier.Inference.Scheduling;
using Xunit;

public sealed class TieredVirtualMemoryTests : IDisposable
{
    private readonly string _tempDir;

    public TieredVirtualMemoryTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierTieringTests_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, true);
            }
        }
        catch
        {
            // Ignore cleanup failure in temp directory
        }
    }

    #region MmapWeightBank Tests

    [Fact]
    public void ComputeAlignedStride_AlignsToPageBoundaries()
    {
        // 4KB alignment
        Assert.Equal(4096, MmapWeightBank.ComputeAlignedStride(100, 4096));
        Assert.Equal(4096, MmapWeightBank.ComputeAlignedStride(4096, 4096));
        Assert.Equal(8192, MmapWeightBank.ComputeAlignedStride(4097, 4096));

        // 2MB huge page alignment
        long hugePage = 2 * 1024 * 1024;
        Assert.Equal(hugePage, MmapWeightBank.ComputeAlignedStride(hugePage - 1, hugePage));
        Assert.Equal(hugePage, MmapWeightBank.ComputeAlignedStride(hugePage, hugePage));
        Assert.Equal(2 * hugePage, MmapWeightBank.ComputeAlignedStride(hugePage + 1, hugePage));
    }

    [Fact]
    public void ComputeAlignedStride_InvalidParameters_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.ComputeAlignedStride(0, 4096));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.ComputeAlignedStride(-10, 4096));
        Assert.Throws<ArgumentException>(() => MmapWeightBank.ComputeAlignedStride(100, 3000)); // Non power-of-two
    }

    [Fact]
    public unsafe void MmapWeightBank_CreateAndRead_PreservesDataAndAlignment()
    {
        string filePath = Path.Combine(_tempDir, "model.gmoe");
        int layers = 2;
        int experts = 4;
        long expertSize = 512; // bytes

        MmapWeightBank.CreateFile(filePath, layers, experts, expertSize, quantType: 1, pageAlignment: 4096,
            initializer: (l, e, span) =>
            {
                byte fillVal = (byte)(l * 10 + e);
                span.Fill(fillVal);
            });

        using var bank = new MmapWeightBank(filePath);

        Assert.Equal(layers, bank.NumLayers);
        Assert.Equal(experts, bank.NumExperts);
        Assert.Equal(expertSize, bank.ExpertRawBytes);
        Assert.Equal(4096, bank.ExpertStrideBytes);
        Assert.Equal(4096, bank.PageAlignment);

        // Verify each expert data matches initializer
        for (int l = 0; l < layers; l++)
        {
            for (int e = 0; e < experts; e++)
            {
                var span = bank.GetExpertSpan(l, e);
                Assert.Equal(expertSize, span.Length);
                byte expected = (byte)(l * 10 + e);
                for (int i = 0; i < span.Length; i++)
                {
                    Assert.Equal(expected, span[i]);
                }

                // Verify pointer alignment: offset from base should be page-aligned
                byte* ptr = bank.GetExpertPointer(l, e);
                long offset = ptr - bank.BasePointer;
                Assert.Equal(0, offset % 4096);
            }
        }
    }

    [Fact]
    public unsafe void MmapWeightBank_OutOfBounds_Throws()
    {
        string filePath = Path.Combine(_tempDir, "bounds.gmoe");
        MmapWeightBank.CreateFile(filePath, 2, 2, 1024);

        using var bank = new MmapWeightBank(filePath);
        Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertPointer(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertPointer(2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertPointer(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertPointer(0, 2));
    }

    [Fact]
    public void MmapWeightBank_PrefetchAndLock_ExecutesWithoutError()
    {
        string filePath = Path.Combine(_tempDir, "prefetch.gmoe");
        MmapWeightBank.CreateFile(filePath, 1, 4, 2048);

        using var bank = new MmapWeightBank(filePath);
        bank.PrefetchExpertPages(0, 0);
        bank.PrefetchRange(0, new[] { 1, 2 });

        bool locked = bank.LockExpertPages(0, 0);
        if (locked)
        {
            bank.UnlockExpertPages(0, 0);
        }
    }

    [Fact]
    public unsafe void MmapWeightBank_Disposed_ThrowsOnAccess()
    {
        string filePath = Path.Combine(_tempDir, "disposed.gmoe");
        MmapWeightBank.CreateFile(filePath, 1, 1, 1024);

        var bank = new MmapWeightBank(filePath);
        bank.Dispose();

        Assert.Throws<ObjectDisposedException>(() => bank.GetExpertPointer(0, 0));
    }

    #endregion

    #region PinnedHostBufferPool Tests

    [Fact]
    public unsafe void PinnedHostBufferPool_RentAndReturn_ReusesBuffers()
    {
        using var pool = new PinnedHostBufferPool(bufferSizeBytes: 4096, maxBuffers: 4, preferCuda: false);

        var buf1 = pool.Rent();
        Assert.True(buf1.Pointer != null);
        Assert.Equal((nuint)4096, buf1.SizeBytes);
        Assert.Equal(1, pool.TotalAllocatedCount);
        Assert.Equal(0, pool.FreeBufferCount);

        // Write and read buffer
        var span = buf1.AsSpan();
        span.Fill(0xAB);
        Assert.Equal(0xAB, span[0]);
        Assert.Equal(0xAB, span[4095]);

        // Return buffer to pool
        buf1.Dispose();
        Assert.Equal(1, pool.FreeBufferCount);

        // Next rent reuses the same buffer
        var buf2 = pool.Rent();
        Assert.Equal(1, pool.TotalAllocatedCount);
        Assert.Equal(0, pool.FreeBufferCount);
        buf2.Dispose();
    }

    [Fact]
    public unsafe void PinnedHostBufferPool_ConcurrentRent_SucceedsUpToCapacity()
    {
        using var pool = new PinnedHostBufferPool(bufferSizeBytes: 2048, maxBuffers: 8, preferCuda: false);

        var buffers = new PinnedHostBufferPool.PinnedBuffer[8];
        for (int i = 0; i < 8; i++)
        {
            buffers[i] = pool.Rent();
            Assert.True(buffers[i].Pointer != null);
        }

        Assert.Equal(8, pool.TotalAllocatedCount);
        Assert.Equal(0, pool.FreeBufferCount);

        for (int i = 0; i < 8; i++)
        {
            buffers[i].Dispose();
        }

        Assert.Equal(8, pool.FreeBufferCount);
    }

    #endregion

    #region GpuResidentSlotManager Tests

    [Fact]
    public void GpuResidentSlotManager_AllocateAndHit_TracksStats()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 1024, allocateVram: false);

        Assert.Equal(4, slotManager.SlotCount);
        Assert.Equal(0, slotManager.OccupiedSlotCount);

        // First access: cache miss
        Assert.False(slotManager.TryGetSlot(layerIdx: 0, expertIdx: 1, out _, out _));
        Assert.Equal(1, slotManager.MissCount);
        Assert.Equal(0, slotManager.HitCount);

        // Allocate slot
        bool allocated = slotManager.AllocateOrEvictSlot(0, 1, out int slotIdx, out _, out _, out IntPtr devPtr);
        Assert.True(allocated);
        Assert.InRange(slotIdx, 0, 3);
        Assert.NotEqual(IntPtr.Zero, devPtr);
        Assert.Equal(1, slotManager.OccupiedSlotCount);

        // Second access: cache hit
        Assert.True(slotManager.TryGetSlot(0, 1, out int hitSlot, out IntPtr hitPtr));
        Assert.Equal(slotIdx, hitSlot);
        Assert.Equal(devPtr, hitPtr);
        Assert.Equal(1, slotManager.HitCount);
        Assert.Equal(0.5, slotManager.HitRatio);
    }

    [Fact]
    public void GpuResidentSlotManager_Eviction_EvictsLRUUnpinned()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 1024, allocateVram: false);

        // Allocate slot 0: (layer 0, exp 0)
        slotManager.AllocateOrEvictSlot(0, 0, out int slot0, out _, out _, out _);
        Thread.Sleep(5); // Ensure distinct timestamps

        // Allocate slot 1: (layer 0, exp 1)
        slotManager.AllocateOrEvictSlot(0, 1, out int slot1, out _, out _, out _);
        Thread.Sleep(5);

        // Touch slot 0 to make slot 1 the oldest (LRU)
        slotManager.TryGetSlot(0, 0, out _, out _);

        // Allocate new expert (layer 1, exp 0): should evict slot 1 (exp 1)
        bool evicted = slotManager.AllocateOrEvictSlot(1, 0, out int newSlot, out int evictLayer, out int evictExp, out _);
        Assert.True(evicted);
        Assert.Equal(0, evictLayer);
        Assert.Equal(1, evictExp);
        Assert.Equal(slot1, newSlot);
        Assert.Equal(1, slotManager.EvictionCount);

        // Slot 0 (layer 0, exp 0) must still be resident
        Assert.True(slotManager.TryGetSlot(0, 0, out _, out _));
        // Slot 1 (layer 0, exp 1) was evicted
        Assert.False(slotManager.TryGetSlot(0, 1, out _, out _));
        // New expert (layer 1, exp 0) is resident
        Assert.True(slotManager.TryGetSlot(1, 0, out _, out _));
    }

    [Fact]
    public void GpuResidentSlotManager_Pinning_ProtectsFromEviction()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 1024, allocateVram: false);

        slotManager.AllocateOrEvictSlot(0, 0, out int s0, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(0, 1, out int s1, out _, out _, out _);

        // Pin slot 0 (oldest)
        Assert.True(slotManager.PinExpert(0, 0));

        // Attempting to allocate third expert must evict slot 1, NOT the pinned slot 0
        slotManager.AllocateOrEvictSlot(1, 0, out int newSlot, out int evictL, out int evictE, out _);
        Assert.Equal(0, evictL);
        Assert.Equal(1, evictE); // Exp 1 was evicted
        Assert.Equal(s1, newSlot);

        // Pinned expert 0 is still resident
        Assert.True(slotManager.TryGetSlot(0, 0, out _, out _));

        // Unpin expert 0
        Assert.True(slotManager.UnpinExpert(0, 0));
    }

    #endregion

    #region TieredVirtualMemoryManager Tests

    [Fact]
    public unsafe void TieredVirtualMemoryManager_AccessHostAndGpu_ResolvesTiersCorrectly()
    {
        string filePath = Path.Combine(_tempDir, "tiered.gmoe");
        MmapWeightBank.CreateFile(filePath, 2, 4, 1024, pageAlignment: 4096);

        using var bank = new MmapWeightBank(filePath);
        using var manager = new TieredVirtualMemoryManager(bank, vramSlotCount: 2, pinnedBufferCount: 2, preferCuda: false);

        // Access for CPU: Tier should be HostRam
        var cpuDesc = manager.AccessExpert(layerIdx: 0, expertIdx: 2, targetGpu: false);
        Assert.Equal(MemoryTier.HostRam, cpuDesc.Tier);
        Assert.True(cpuDesc.HostAddress != null);
        Assert.Equal(IntPtr.Zero, cpuDesc.DeviceAddress);
        Assert.Equal(1024, cpuDesc.ByteSize);
        Assert.Equal(-1, cpuDesc.SlotIndex);

        // Access for GPU: Tier should be Vram
        var gpuDesc = manager.AccessExpert(layerIdx: 0, expertIdx: 2, targetGpu: true);
        Assert.Equal(MemoryTier.Vram, gpuDesc.Tier);
        Assert.NotEqual(IntPtr.Zero, gpuDesc.DeviceAddress);
        Assert.InRange(gpuDesc.SlotIndex, 0, 1);

        // Telemetry
        var telemetry = manager.GetTelemetry();
        Assert.Equal(2, telemetry.TotalVramSlots);
        Assert.Equal(1, telemetry.OccupiedVramSlots);
    }

    [Fact]
    public async Task TieredVirtualMemoryManager_SpeculativePrefetch_LoadsPagesAsync()
    {
        string filePath = Path.Combine(_tempDir, "speculative.gmoe");
        MmapWeightBank.CreateFile(filePath, 2, 8, 1024, pageAlignment: 4096);

        using var bank = new MmapWeightBank(filePath);
        using var manager = new TieredVirtualMemoryManager(bank, vramSlotCount: 4, pinnedBufferCount: 2, preferCuda: false);

        // Issue prefetch for next layer speculative experts
        int[] nextExperts = new[] { 1, 3, 5 };
        await manager.PrefetchSpeculativeExpertsAsync(layerIdx: 1, nextExperts, prefetchToVram: true);

        // All 3 prefetched experts should now be resident in VRAM
        Assert.True(manager.VramManager.TryGetSlot(1, 1, out _, out _));
        Assert.True(manager.VramManager.TryGetSlot(1, 3, out _, out _));
        Assert.True(manager.VramManager.TryGetSlot(1, 5, out _, out _));
    }

    #endregion

    #region BandwidthCoScheduler Tests

    [Fact]
    public void BandwidthCoScheduler_QStarFormula_MatchesTheoreticalValues()
    {
        // Host Laptop params: Bp = 12.5, Bh = 65.0, Bvram = 272.0
        // Ratio = (1/65) / ((1/12.5 + 1/272) + 1/65) = 0.01538 / (0.08367 + 0.01538) = 0.01538 / 0.09905 = 0.155
        var laptopScheduler = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);

        // For m = 6 missing experts: round(6 * 0.155) = 1 expert on GPU, 5 on CPU
        int qLaptop = laptopScheduler.CalculateOptimalGpuExpertCount(6);
        Assert.Equal(1, qLaptop);

        // Beast Desktop params: Bp = 13.0, Bh = 32.0, Bvram = 360.0
        // Ratio = (1/32) / ((1/13 + 1/360) + 1/32) = 0.03125 / (0.0797 + 0.03125) = 0.03125 / 0.11095 = 0.2816
        // For m = 6 missing experts:
        // Latency at q=1: max(1 * 0.0797, 5 * 0.03125) = 0.15625 s (156.25 ms)
        // Latency at q=2: max(2 * 0.0797, 4 * 0.03125) = 0.15940 s (159.40 ms)
        // Discrete candidate evaluation selects q* = 1 (saving 3.15 ms over continuous rounded q=2)
        var beastScheduler = new BandwidthCoScheduler(initialPcieGBps: 13.0, initialCpuGBps: 32.0, vramGBps: 360.0);
        int qBeast = beastScheduler.CalculateOptimalGpuExpertCount(6);
        Assert.Equal(1, qBeast);
    }

    [Fact]
    public void BandwidthCoScheduler_EdgeCases_HandledGracefully()
    {
        var scheduler = new BandwidthCoScheduler(10.0, 50.0, 200.0);

        Assert.Equal(0, scheduler.CalculateOptimalGpuExpertCount(0));
        Assert.Equal(0, scheduler.CalculateOptimalGpuExpertCount(-5));

        // Single missing expert
        int q1 = scheduler.CalculateOptimalGpuExpertCount(1);
        Assert.InRange(q1, 0, 1);
    }

    [Fact]
    public void BandwidthCoScheduler_Schedule_PartitionsExpertsAccurately()
    {
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);
        using var slotManager = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 1024, allocateVram: false);

        // Pre-populate expert 1 in VRAM
        slotManager.AllocateOrEvictSlot(layerIdx: 0, expertIdx: 1, out _, out _, out _, out _);

        // Active experts for layer 0: [1, 2, 3, 4, 5, 6, 7] (7 total: 1 resident hit, 6 missing)
        int[] active = new[] { 1, 2, 3, 4, 5, 6, 7 };
        long expertSize = 50 * 1024 * 1024; // 50MB

        var decision = scheduler.Schedule(layerIdx: 0, active, slotManager, expertSize);

        Assert.Equal(0, decision.LayerIndex);
        Assert.Equal(7, decision.TotalActivated);
        Assert.Equal(6, decision.MissingCount);

        // Expert 1 should be in resident hits
        Assert.Single(decision.GpuResidentHits);
        Assert.Equal(1, decision.GpuResidentHits[0]);

        // Missing 6 partitioned with q* = 1
        Assert.Equal(1, decision.QStar);
        Assert.Single(decision.GpuTransfers);
        Assert.Equal(5, decision.CpuExecutions.Count);

        // Latencies predicted should be positive
        Assert.True(decision.PredictedGpuLatencyMs > 0);
        Assert.True(decision.PredictedCpuLatencyMs > 0);
        Assert.Equal(Math.Max(decision.PredictedGpuLatencyMs, decision.PredictedCpuLatencyMs), decision.PredictedTotalLatencyMs);
    }

    [Fact]
    public void BandwidthCoScheduler_TelemetryEWMA_UpdatesCorrectly()
    {
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 10.0, initialCpuGBps: 50.0, vramGBps: 200.0, ewmaAlpha: 0.5);

        // Record a 1GB transfer in 50ms = 20 GB/s observed
        long oneGB = 1024 * 1024 * 1024;
        scheduler.RecordPcieTransfer(oneGB, 50.0);

        // Expected EWMA: 0.5 * 20.0 + 0.5 * 10.0 = 15.0 GB/s
        Assert.InRange(scheduler.PcieBandwidthGBps, 14.9, 15.1);

        // Record CPU processing 2GB in 25ms = 80 GB/s observed
        scheduler.RecordCpuExecution(2 * oneGB, 25.0);

        // Expected EWMA: 0.5 * 80.0 + 0.5 * 50.0 = 65.0 GB/s
        Assert.InRange(scheduler.CpuBandwidthGBps, 64.9, 65.1);

        var telemetry = scheduler.GetTelemetry();
        Assert.Equal(1, telemetry.PcieSamplesRecorded);
        Assert.Equal(1, telemetry.CpuSamplesRecorded);
        Assert.Equal(1.0, telemetry.TotalPcieTransferredGB);
        Assert.Equal(2.0, telemetry.TotalCpuProcessedGB);
    }

    [Fact]
    public void BandwidthCoScheduler_PredictLatencies_ShowsCoSchedulingGain()
    {
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);
        long expertSize = 50 * 1024 * 1024; // 50MB
        int missing = 6;

        var (pureGpu, pureCpu, coScheduled, qStar) = scheduler.PredictLatencies(missing, expertSize);

        Assert.Equal(1, qStar);
        // Co-scheduled should be strictly less than pure GPU and at least as good as pure CPU
        Assert.True(coScheduled < pureGpu, $"Co-scheduled ({coScheduled}ms) should be faster than Pure GPU ({pureGpu}ms)");
        Assert.True(coScheduled <= pureCpu, $"Co-scheduled ({coScheduled}ms) should be <= Pure CPU ({pureCpu}ms)");
    }

    #endregion
}
