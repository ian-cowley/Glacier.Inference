namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Memory;
using Xunit;

/// <summary>
/// Adversarial stress tests challenging MmapWeightBank and TieredVirtualMemoryManager.
/// Tests 4KB page alignment, memory boundaries, pointer slicing safety,
/// multi-threaded concurrency, slot eviction under pressure, and resource cleanup.
/// </summary>
public sealed class MmapTieringAdversarialTests : IDisposable
{
    private readonly string _tempDir;

    public MmapTieringAdversarialTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierAdversarialMmap_" + Guid.NewGuid().ToString("N"));
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
            // Ignore temp cleanup errors
        }
    }

    #region 1. 4KB Page Alignment & Arbitrary Byte Size Challenges

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(4095)]
    [InlineData(4096)]
    [InlineData(4097)]
    [InlineData(8191)]
    [InlineData(8192)]
    [InlineData(8193)]
    [InlineData(65535)]
    [InlineData(65536)]
    [InlineData(65537)]
    [InlineData(1048575)]
    [InlineData(1048576)]
    [InlineData(1048577)]
    public unsafe void PageAlignment_ArbitraryRawByteSizes_EnforcesStrict4KBAlignment(long rawBytes)
    {
        string filePath = Path.Combine(_tempDir, $"align_{rawBytes}.gmoe");
        int layers = 2;
        int experts = 3;

        MmapWeightBank.CreateFile(filePath, layers, experts, rawBytes, pageAlignment: 4096);

        using var bank = new MmapWeightBank(filePath);

        Assert.Equal(4096, bank.PageAlignment);
        Assert.True(bank.DataOffset % 4096 == 0, $"DataOffset {bank.DataOffset} must be 4KB aligned");
        Assert.True(bank.ExpertStrideBytes % 4096 == 0, $"Stride {bank.ExpertStrideBytes} must be 4KB aligned");

        byte* basePtr = bank.BasePointer;
        // Verify OS view accessor is at least 4KB aligned (Win32 MapViewOfFile gives 64KB alignment)
        Assert.True(((nuint)basePtr) % 4096 == 0, $"Base pointer 0x{(nuint)basePtr:X} is not 4KB aligned");

        for (int l = 0; l < layers; l++)
        {
            for (int e = 0; e < experts; e++)
            {
                byte* ptr = bank.GetExpertPointer(l, e);
                long fileOffset = ptr - basePtr;

                // Check relative offset from base
                Assert.True(fileOffset % 4096 == 0, $"File offset {fileOffset} for ({l},{e}) is not 4KB aligned");

                // Check absolute virtual address alignment
                Assert.True(((nuint)ptr) % 4096 == 0, $"Absolute address 0x{(nuint)ptr:X} for ({l},{e}) is not 4KB aligned");

                // Verify pointer is within mapped length
                Assert.True(fileOffset + rawBytes <= bank.FileLength);
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-4096)]
    [InlineData(3000)] // Non power-of-two
    [InlineData(5000)]
    public void PageAlignment_InvalidAlignment_ThrowsArgumentException(long badAlignment)
    {
        Assert.Throws<ArgumentException>(() => MmapWeightBank.ComputeAlignedStride(100, badAlignment));

        string filePath = Path.Combine(_tempDir, "bad_align.gmoe");
        Assert.Throws<ArgumentException>(() => MmapWeightBank.CreateFile(filePath, 1, 1, 100, pageAlignment: badAlignment));
    }

    #endregion

    #region 2. Pointer Slicing & Buffer Overrun Verification

    [Fact]
    public unsafe void PointerSlicing_SentinelDataIntegrity_NoCrossExpertCorruption()
    {
        string filePath = Path.Combine(_tempDir, "sentinel.gmoe");
        int layers = 3;
        int experts = 5;
        long expertSize = 1337; // Prime number of bytes to force non-aligned stride padding

        MmapWeightBank.CreateFile(filePath, layers, experts, expertSize, pageAlignment: 4096,
            initializer: (l, e, span) =>
            {
                // Write unique pattern to each expert
                byte val = (byte)((l * 37 + e * 19) & 0xFF);
                span.Fill(val);
                // Mark head and tail specifically
                span[0] = 0xAA;
                span[^1] = 0x55;
            });

        using var bank = new MmapWeightBank(filePath);
        Assert.Equal(4096, bank.ExpertStrideBytes); // Stride must be padded from 1337 to 4096

        for (int l = 0; l < layers; l++)
        {
            for (int e = 0; e < experts; e++)
            {
                var span = bank.GetExpertSpan(l, e);
                Assert.Equal(expertSize, span.Length);
                Assert.Equal(0xAA, span[0]);
                Assert.Equal(0x55, span[^1]);

                byte expectedVal = (byte)((l * 37 + e * 19) & 0xFF);
                for (int i = 1; i < span.Length - 1; i++)
                {
                    Assert.Equal(expectedVal, span[i]);
                }

                // Verify accessing beyond span throws via safe indexing
                bool threw = false;
                try
                {
                    _ = span[span.Length];
                }
                catch (IndexOutOfRangeException)
                {
                    threw = true;
                }
                Assert.True(threw);
            }
        }
    }

    #endregion

    #region 3. Boundary Indices & Out-Of-Bounds Stress

    [Fact]
    public void BoundaryIndices_ExtremeIndices_SafelyThrows()
    {
        string filePath = Path.Combine(_tempDir, "bounds_stress.gmoe");
        int layers = 2;
        int experts = 4;
        MmapWeightBank.CreateFile(filePath, layers, experts, 1024, pageAlignment: 4096);

        using var bank = new MmapWeightBank(filePath);

        int[] badLayers = { -1, -100, int.MinValue, 2, 3, 100, int.MaxValue };
        int[] badExperts = { -1, -100, int.MinValue, 4, 5, 100, int.MaxValue };

        foreach (var bl in badLayers)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertSpan(bl, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.PrefetchExpertPages(bl, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.LockExpertPages(bl, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.UnlockExpertPages(bl, 0));
        }

        foreach (var be in badExperts)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.GetExpertSpan(0, be));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.PrefetchExpertPages(0, be));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.LockExpertPages(0, be));
            Assert.Throws<ArgumentOutOfRangeException>(() => bank.UnlockExpertPages(0, be));
        }
    }

    [Fact]
    public void BoundaryIndices_ZeroOrNegativeDimensions_ThrowsArgumentOutOfRangeException()
    {
        string filePath = Path.Combine(_tempDir, "invalid_dims.gmoe");

        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, 0, 4, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, -1, 4, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, 2, 0, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, 2, -1, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, 2, 4, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => MmapWeightBank.CreateFile(filePath, 2, 4, -100));
    }

    #endregion

    #region 4. Multi-Threaded Concurrency Stress & High Slot Eviction

    [Fact]
    public void Concurrency_ParallelAccessExpert_NoDeadlockOrMemoryFault()
    {
        string filePath = Path.Combine(_tempDir, "concurrent_stress.gmoe");
        int layers = 4;
        int experts = 8; // 32 experts total
        long expertSize = 4096;

        MmapWeightBank.CreateFile(filePath, layers, experts, expertSize, pageAlignment: 4096,
            initializer: (l, e, span) =>
            {
                byte val = (byte)(l * 10 + e);
                span.Fill(val);
            });

        using var bank = new MmapWeightBank(filePath);
        // Constrain VRAM slots to 4 so 32 experts constantly force eviction thrashing
        using var manager = new TieredVirtualMemoryManager(bank, vramSlotCount: 4, pinnedBufferCount: 4, preferCuda: false);

        int threadCount = 16;
        int iterationsPerThread = 200;
        int totalExceptions = 0;

        Parallel.For(0, threadCount, t =>
        {
            var random = new Random(t * 1000 + 42);
            for (int i = 0; i < iterationsPerThread; i++)
            {
                int l = random.Next(0, layers);
                int e = random.Next(0, experts);
                bool targetGpu = (i % 2 == 0);

                try
                {
                    var desc = manager.AccessExpert(l, e, targetGpu);
                    if (targetGpu)
                    {
                        Assert.Equal(MemoryTier.Vram, desc.Tier);
                        Assert.InRange(desc.SlotIndex, 0, 3);
                    }
                    else
                    {
                        Assert.Equal(MemoryTier.HostRam, desc.Tier);
                        var span = desc.AsHostSpan();
                        Assert.Equal(expertSize, span.Length);
                        byte expectedVal = (byte)(l * 10 + e);
                        Assert.Equal(expectedVal, span[0]);
                    }
                }
                catch (Exception)
                {
                    Interlocked.Increment(ref totalExceptions);
                }
            }
        });

        Assert.Equal(0, totalExceptions);

        var telemetry = manager.GetTelemetry();
        Assert.Equal(4, telemetry.TotalVramSlots);
        Assert.Equal(4, telemetry.OccupiedVramSlots);
        Assert.True(telemetry.VramHitCount > 0, "Expected VRAM cache hits under concurrency");
        Assert.True(telemetry.VramMissCount > 0, "Expected VRAM cache misses under concurrency");
        Assert.True(telemetry.VramEvictionCount > 0, "Expected VRAM evictions under slot pressure");
    }

    [Fact]
    public void Concurrency_PinnedExpertProtection_NeverEvictedUnderPressure()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 3, slotByteSize: 1024, allocateVram: false);

        // Populate slot 0 with (0, 0) and pin it
        slotManager.AllocateOrEvictSlot(0, 0, out int pinnedSlot, out _, out _, out _);
        Assert.True(slotManager.PinExpert(0, 0));

        // Hammer the remaining 2 slots across 8 threads with 10 competing experts
        int threadCount = 8;
        int iterations = 100;

        Parallel.For(0, threadCount, t =>
        {
            var rand = new Random(t * 77);
            for (int i = 0; i < iterations; i++)
            {
                int l = 1; // layer 1
                int e = rand.Next(1, 10); // experts 1..9
                slotManager.AllocateOrEvictSlot(l, e, out _, out _, out _, out _);
            }
        });

        // Verify pinned expert (0, 0) is STILL resident in VRAM and was never evicted!
        Assert.True(slotManager.TryGetSlot(0, 0, out int residentSlot, out _));
        Assert.Equal(pinnedSlot, residentSlot);
    }

    [Fact]
    public void SlotManager_AllSlotsPinned_ThrowsGracefullyWhenEvictionImpossible()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 1024, allocateVram: false);

        slotManager.AllocateOrEvictSlot(0, 0, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);

        // Pin both slots
        Assert.True(slotManager.PinExpert(0, 0));
        Assert.True(slotManager.PinExpert(0, 1));

        // Attempting to allocate 3rd expert must throw InvalidOperationException
        var ex = Assert.Throws<InvalidOperationException>(() =>
            slotManager.AllocateOrEvictSlot(0, 2, out _, out _, out _, out _));

        Assert.Contains("All GPU resident expert slots are pinned", ex.Message);
    }

    #endregion

    #region 5. Lifecycle, Double-Dispose & File Cleanup

    [Fact]
    public void Lifecycle_DoubleDispose_NoException()
    {
        string filePath = Path.Combine(_tempDir, "double_disp.gmoe");
        MmapWeightBank.CreateFile(filePath, 1, 2, 1024, pageAlignment: 4096);

        var bank = new MmapWeightBank(filePath);
        var manager = new TieredVirtualMemoryManager(bank, vramSlotCount: 2, pinnedBufferCount: 2, preferCuda: false);

        // First dispose
        manager.Dispose();
        // Second dispose should be idempotent
        manager.Dispose();
        bank.Dispose();
    }

    [Fact]
    public void Lifecycle_RepeatedCreateAndDispose_ReleasesFileLocksCleanly()
    {
        string filePath = Path.Combine(_tempDir, "rapid_cycle.gmoe");

        for (int cycle = 0; cycle < 10; cycle++)
        {
            MmapWeightBank.CreateFile(filePath, 2, 2, 2048, pageAlignment: 4096);

            using (var bank = new MmapWeightBank(filePath))
            using (var manager = new TieredVirtualMemoryManager(bank, 2, 2, false))
            {
                var desc = manager.AccessExpert(0, 1, false);
                Assert.Equal(MemoryTier.HostRam, desc.Tier);
            }

            // Immediately delete the file to verify all MMF and view handles were fully closed
            File.Delete(filePath);
            Assert.False(File.Exists(filePath));
        }
    }

    #endregion
}
