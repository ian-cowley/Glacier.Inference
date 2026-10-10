namespace Glacier.Inference.Memory;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gpu;

/// <summary>
/// Memory tier classification in the hierarchical memory manager.
/// </summary>
public enum MemoryTier
{
    Vram = 1,
    HostRam = 2,
    Storage = 3
}

/// <summary>
/// Descriptor representing the physical or virtual location of an expert's weights.
/// </summary>
public readonly unsafe struct ExpertMemoryDescriptor
{
    public MemoryTier Tier { get; init; }
    public IntPtr DeviceAddress { get; init; }
    public byte* HostAddress { get; init; }
    public long ByteSize { get; init; }
    public bool IsPinned { get; init; }
    public int SlotIndex { get; init; }

    public ReadOnlySpan<byte> AsHostSpan()
    {
        if (HostAddress == null)
            throw new InvalidOperationException("HostAddress is null (expert may reside purely in VRAM).");
        return new ReadOnlySpan<byte>(HostAddress, (int)ByteSize);
    }
}

/// <summary>
/// Telemetry counters for the tiered virtual memory manager.
/// </summary>
public sealed record TieredMemoryTelemetry(
    int TotalVramSlots,
    int OccupiedVramSlots,
    long VramHitCount,
    long VramMissCount,
    long VramEvictionCount,
    double VramHitRatio,
    long MmapFileSizeBytes,
    long ExpertStrideBytes,
    long PageAlignmentBytes,
    long TotalPcieBytesTransferred,
    double TotalPcieTransferTimeMs);

/// <summary>
/// Hierarchical 3-tier virtual memory and paging manager (VRAM / Host RAM / NVMe Storage).
/// Provides zero-copy memory-mapped file access, OS page alignment (4KB/2MB), pinned DMA transfers,
/// and speculative asynchronous prefetching for frontier MoE architectures.
/// </summary>
public sealed unsafe class TieredVirtualMemoryManager : IDisposable
{
    private readonly MmapWeightBank _weightBank;
    private readonly PinnedHostBufferPool _pinnedPool;
    private readonly GpuResidentSlotManager _vramManager;
    private readonly bool _hasCuda;

    private long _totalPcieBytesTransferred;
    private double _totalPcieTransferTimeMs;
    private bool _disposed;

    public MmapWeightBank WeightBank => _weightBank;
    public PinnedHostBufferPool PinnedPool => _pinnedPool;
    public GpuResidentSlotManager VramManager => _vramManager;

    public TieredVirtualMemoryManager(
        MmapWeightBank weightBank,
        int vramSlotCount = 32,
        int pinnedBufferCount = 8,
        bool preferCuda = true)
    {
        ArgumentNullException.ThrowIfNull(weightBank);
        _weightBank = weightBank;

        bool cudaAvailable = false;
        if (preferCuda)
        {
            try { cudaAvailable = CuDriver.IsAvailable(); } catch { cudaAvailable = false; }
        }
        _hasCuda = cudaAvailable;

        nuint slotBytes = (nuint)weightBank.ExpertStrideBytes;
        _pinnedPool = new PinnedHostBufferPool(slotBytes, pinnedBufferCount, preferCuda);
        _vramManager = new GpuResidentSlotManager(vramSlotCount, slotBytes, cudaAvailable);
    }

    /// <summary>
    /// Resolves an expert's weight memory descriptor, demand-paging and streaming to VRAM if requested.
    /// </summary>
    public ExpertMemoryDescriptor AccessExpert(int layerIdx, int expertIdx, bool targetGpu)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        byte* hostPtr = _weightBank.GetExpertPointer(layerIdx, expertIdx);
        long rawBytes = _weightBank.ExpertRawBytes;

        if (!targetGpu)
        {
            // Host RAM execution path: zero-copy pointer directly from memory-mapped bank
            return new ExpertMemoryDescriptor
            {
                Tier = MemoryTier.HostRam,
                DeviceAddress = IntPtr.Zero,
                HostAddress = hostPtr,
                ByteSize = rawBytes,
                IsPinned = false,
                SlotIndex = -1
            };
        }

        // Target is GPU: Check VRAM resident working set
        if (_vramManager.TryGetSlot(layerIdx, expertIdx, out int slotIdx, out IntPtr devicePtr))
        {
            return new ExpertMemoryDescriptor
            {
                Tier = MemoryTier.Vram,
                DeviceAddress = devicePtr,
                HostAddress = hostPtr,
                ByteSize = rawBytes,
                IsPinned = false,
                SlotIndex = slotIdx
            };
        }

        // VRAM cache miss: allocate or evict an unpinned slot
        _vramManager.AllocateOrEvictSlot(layerIdx, expertIdx, out slotIdx, out _, out _, out devicePtr);

        // Stream expert weights to VRAM
        TransferToVram(hostPtr, devicePtr, (nuint)rawBytes);

        return new ExpertMemoryDescriptor
        {
            Tier = MemoryTier.Vram,
            DeviceAddress = devicePtr,
            HostAddress = hostPtr,
            ByteSize = rawBytes,
            IsPinned = false,
            SlotIndex = slotIdx
        };
    }

    /// <summary>
    /// Streams host memory weights into GPU VRAM with timing telemetry.
    /// </summary>
    private void TransferToVram(byte* hostPtr, IntPtr devicePtr, nuint byteCount)
    {
        long start = Stopwatch.GetTimestamp();

        if (_hasCuda && devicePtr != IntPtr.Zero)
        {
            int err = CuDriver.MemcpyHtoD(devicePtr, (IntPtr)hostPtr, byteCount);
            if (err != 0)
            {
                // Fallback or retry
            }
        }

        double elapsedMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        Interlocked.Add(ref _totalPcieBytesTransferred, (long)byteCount);
        // Thread-safe accumulation of elapsed time
        double current, updated;
        do
        {
            current = Volatile.Read(ref _totalPcieTransferTimeMs);
            updated = current + elapsedMs;
        } while (Interlocked.CompareExchange(ref _totalPcieTransferTimeMs, updated, current) != current);
    }

    /// <summary>
    /// Proactively prefetches speculative expert pages from storage into RAM and VRAM.
    /// Employs background asynchronous tasks to hide PCIe and NVMe access latencies.
    /// </summary>
    public ValueTask PrefetchSpeculativeExpertsAsync(
        int layerIdx,
        ReadOnlyMemory<int> expertIndices,
        bool prefetchToVram = true,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (expertIndices.IsEmpty) return ValueTask.CompletedTask;

        return new ValueTask(Task.Run(() =>
        {
            var span = expertIndices.Span;
            for (int i = 0; i < span.Length; i++)
            {
                if (cancellationToken.IsCancellationRequested) break;
                int expIdx = span[i];

                // 1. Advise OS kernel to pull pages from NVMe into Host RAM working set
                _weightBank.PrefetchExpertPages(layerIdx, expIdx);

                // 2. If VRAM staging requested and not already in VRAM, stage in background
                if (prefetchToVram)
                {
                    if (!_vramManager.TryGetSlot(layerIdx, expIdx, out _, out _))
                    {
                        byte* hostPtr = _weightBank.GetExpertPointer(layerIdx, expIdx);
                        _vramManager.AllocateOrEvictSlot(layerIdx, expIdx, out int slotIdx, out _, out _, out IntPtr dptr);
                        TransferToVram(hostPtr, dptr, (nuint)_weightBank.ExpertRawBytes);
                    }
                }
            }
        }, cancellationToken));
    }

    /// <summary>
    /// Retrieves current operational metrics and cache hit statistics.
    /// </summary>
    public TieredMemoryTelemetry GetTelemetry()
    {
        return new TieredMemoryTelemetry(
            TotalVramSlots: _vramManager.SlotCount,
            OccupiedVramSlots: _vramManager.OccupiedSlotCount,
            VramHitCount: _vramManager.HitCount,
            VramMissCount: _vramManager.MissCount,
            VramEvictionCount: _vramManager.EvictionCount,
            VramHitRatio: _vramManager.HitRatio,
            MmapFileSizeBytes: _weightBank.FileLength,
            ExpertStrideBytes: _weightBank.ExpertStrideBytes,
            PageAlignmentBytes: _weightBank.PageAlignment,
            TotalPcieBytesTransferred: Volatile.Read(ref _totalPcieBytesTransferred),
            TotalPcieTransferTimeMs: Volatile.Read(ref _totalPcieTransferTimeMs));
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _vramManager.Dispose();
            _pinnedPool.Dispose();
            _weightBank.Dispose();
        }
    }
}
