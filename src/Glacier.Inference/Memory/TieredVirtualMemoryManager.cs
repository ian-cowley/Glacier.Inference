namespace Glacier.Inference.Memory;

using System;
using System.Collections.Generic;
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
/// Configuration options for <see cref="TieredVirtualMemoryManager"/> supporting exclusive Tier-2 Host-RAM budgets,
/// decayed-frequency VRAM eviction, and NVMe storage page-fault concurrency gating.
/// </summary>
public sealed class TieredMemoryOptions
{
    public int VramSlotCount { get; set; } = 32;
    public int PinnedBufferCount { get; set; } = 8;
    public bool PreferCuda { get; set; } = true;
    public long HostRamResidentBudgetBytes { get; set; } = 0;
    public bool ExclusiveHostRamResidency { get; set; } = true;
    public SlotEvictionPolicy EvictionPolicy { get; set; } = SlotEvictionPolicy.DecayedFrequency;
    public ExpertRoutingProfile? RoutingProfile { get; set; }
    public int MaxConcurrentStoragePageFaults { get; set; } = 2;

    public static TieredMemoryOptions FromHostBudgetGiB(double budgetGiB, int vramSlotCount = 32)
    {
        double clampedGiB = double.IsFinite(budgetGiB) && budgetGiB > 0.0 ? budgetGiB : 0.0;
        long bytes = (long)Math.Min(long.MaxValue / 2.0, clampedGiB * (1024.0 * 1024.0 * 1024.0));
        return new TieredMemoryOptions
        {
            VramSlotCount = vramSlotCount,
            HostRamResidentBudgetBytes = bytes
        };
    }
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
    double TotalPcieTransferTimeMs)
{
    public int ActiveVramSlots { get; init; }
    public int HostRamResidentCount { get; init; }
    public int MaxHostRamResidentSlots { get; init; }
    public long AdaptiveSwapCount { get; init; }
}

/// <summary>
/// Hierarchical 3-tier virtual memory and paging manager (VRAM / Exclusive Host RAM / NVMe Storage).
/// Provides zero-copy memory-mapped file access, OS page alignment (4KB/2MB), pinned DMA transfers,
/// decayed-frequency residency management, WDDM cliff elasticity, and storage page-fault concurrency gating.
/// </summary>
public sealed unsafe class TieredVirtualMemoryManager : IDisposable
{
    private readonly MmapWeightBank _weightBank;
    private readonly PinnedHostBufferPool _pinnedPool;
    private readonly GpuResidentSlotManager _vramManager;
    private readonly bool _hasCuda;

    private readonly int _maxHostRamSlots;
    private readonly bool _exclusiveHostRam;
    private readonly Dictionary<(int Layer, int Expert), long> _hostRamResidents = new();
    private readonly HashSet<(int Layer, int Expert)> _osLockedPages = new();
    private readonly Lock _tierLock = new();
    private readonly SemaphoreSlim _storageFaultGate;
    private ExpertRoutingProfile? _routingProfile;

    private long _totalPcieBytesTransferred;
    private double _totalPcieTransferTimeMs;
    private bool _disposed;

    public MmapWeightBank WeightBank => _weightBank;
    public PinnedHostBufferPool PinnedPool => _pinnedPool;
    public GpuResidentSlotManager VramManager => _vramManager;
    public ExpertRoutingProfile? RoutingProfile => Volatile.Read(ref _routingProfile);
    public int MaxHostRamResidentSlots => _maxHostRamSlots;
    public bool ExclusiveHostRamResidency => _exclusiveHostRam;

    public int HostRamResidentCount
    {
        get
        {
            lock (_tierLock)
            {
                return _hostRamResidents.Count;
            }
        }
    }

    public TieredVirtualMemoryManager(
        MmapWeightBank weightBank,
        int vramSlotCount = 32,
        int pinnedBufferCount = 8,
        bool preferCuda = true)
        : this(weightBank, new TieredMemoryOptions
        {
            VramSlotCount = vramSlotCount,
            PinnedBufferCount = pinnedBufferCount,
            PreferCuda = preferCuda,
            HostRamResidentBudgetBytes = 0,
            ExclusiveHostRamResidency = true,
            EvictionPolicy = SlotEvictionPolicy.Lru
        })
    {
    }

    public TieredVirtualMemoryManager(MmapWeightBank weightBank, TieredMemoryOptions options)
    {
        ArgumentNullException.ThrowIfNull(weightBank);
        ArgumentNullException.ThrowIfNull(options);

        _weightBank = weightBank;
        _routingProfile = options.RoutingProfile;
        _exclusiveHostRam = options.ExclusiveHostRamResidency;

        bool cudaAvailable = false;
        if (options.PreferCuda)
        {
            try { cudaAvailable = CuDriver.IsAvailable(); } catch { cudaAvailable = false; }
        }
        _hasCuda = cudaAvailable;

        nuint slotBytes = (nuint)weightBank.ExpertStrideBytes;
        _pinnedPool = new PinnedHostBufferPool(slotBytes, Math.Max(1, options.PinnedBufferCount), options.PreferCuda);
        _vramManager = new GpuResidentSlotManager(
            options.VramSlotCount,
            slotBytes,
            cudaAvailable,
            options.EvictionPolicy,
            options.RoutingProfile);

        _maxHostRamSlots = options.HostRamResidentBudgetBytes > 0
            ? Math.Max(1, (int)Math.Min(int.MaxValue, options.HostRamResidentBudgetBytes / Math.Max(1L, weightBank.ExpertStrideBytes)))
            : 0;

        int maxFaultConcurrency = Math.Max(1, options.MaxConcurrentStoragePageFaults);
        _storageFaultGate = new SemaphoreSlim(maxFaultConcurrency, maxFaultConcurrency);
    }

    public bool IsHostRamResident(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_tierLock)
        {
            return _hostRamResidents.ContainsKey((layerIdx, expertIdx));
        }
    }

    /// <summary>
    /// Seeds Tier-1 VRAM and exclusive Tier-2 Host-RAM tiers from empirical routing scores in <paramref name="profile"/>.
    /// </summary>
    public void WarmFromProfile(ExpertRoutingProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ObjectDisposedException.ThrowIf(_disposed, this);

        Volatile.Write(ref _routingProfile, profile);
        _vramManager.ConfigurePolicy(SlotEvictionPolicy.DecayedFrequency, profile);

        int vramTarget = _vramManager.ActiveSlotCount;
        int totalWanted = vramTarget + _maxHostRamSlots;
        if (totalWanted <= 0)
            return;

        int maxPossible = Math.Min(profile.NumLayers * profile.NumExperts, totalWanted * 2);
        var pairs = new (int LayerIndex, int ExpertIndex, double Score)[maxPossible];
        int pairCount = profile.GetTopLayerExpertPairs(pairs, includeZeroScores: true);

        int placedVram = 0;
        for (int i = 0; i < pairCount && placedVram < vramTarget; i++)
        {
            var (l, e, _) = pairs[i];
            if ((uint)l >= (uint)_weightBank.NumLayers || (uint)e >= (uint)_weightBank.NumExperts)
                continue;

            bool allocated = _vramManager.AllocateOrEvictSlot(l, e, out _, out _, out _, out IntPtr dptr);
            if (allocated)
            {
                byte* hostPtr = _weightBank.GetExpertPointer(l, e);
                TransferToVram(hostPtr, dptr, (nuint)_weightBank.ExpertRawBytes);
            }

            if (_exclusiveHostRam)
            {
                lock (_tierLock)
                {
                    RemoveFromHostRamLocked(l, e);
                }
            }

            placedVram++;
        }

        if (_maxHostRamSlots <= 0)
            return;

        lock (_tierLock)
        {
            int placedHost = 0;
            for (int i = 0; i < pairCount && placedHost < _maxHostRamSlots; i++)
            {
                var (l, e, _) = pairs[i];
                if ((uint)l >= (uint)_weightBank.NumLayers || (uint)e >= (uint)_weightBank.NumExperts)
                    continue;
                if (_exclusiveHostRam && _vramManager.ContainsSlot(l, e))
                    continue;

                PromoteToHostRamLocked(l, e);
                placedHost++;
            }
        }
    }

    /// <summary>
    /// Executes an adaptive hot/cold VRAM slot swap pass based on the active routing profile,
    /// demoting evicted experts into the exclusive Tier-2 Host-RAM budget when configured.
    /// </summary>
    public int RunAdaptivePass(int maxSwaps = 4, double minScoreAdvantageRatio = 1.25)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var profile = Volatile.Read(ref _routingProfile);
        if (profile == null || maxSwaps <= 0 || _vramManager.ActiveSlotCount <= 0)
            return 0;

        var rented = maxSwaps > 64
            ? new (int SlotIndex, int NewLayer, int NewExpert, int EvictedLayer, int EvictedExpert)[maxSwaps]
            : null;
        Span<(int SlotIndex, int NewLayer, int NewExpert, int EvictedLayer, int EvictedExpert)> swaps =
            rented ?? stackalloc (int, int, int, int, int)[maxSwaps];

        int count = _vramManager.PlanAdaptiveSwaps(profile, maxSwaps, minScoreAdvantageRatio, swaps);
        for (int i = 0; i < count; i++)
        {
            var s = swaps[i];
            byte* hostPtr = _weightBank.GetExpertPointer(s.NewLayer, s.NewExpert);
            IntPtr devPtr = _vramManager.GetSlotDevicePointer(s.SlotIndex);
            TransferToVram(hostPtr, devPtr, (nuint)_weightBank.ExpertRawBytes);

            if (_exclusiveHostRam)
            {
                lock (_tierLock)
                {
                    RemoveFromHostRamLocked(s.NewLayer, s.NewExpert);
                    if (_maxHostRamSlots > 0 && s.EvictedLayer >= 0 && s.EvictedExpert >= 0)
                    {
                        PromoteToHostRamLocked(s.EvictedLayer, s.EvictedExpert);
                    }
                }
            }
        }

        return count;
    }

    /// <summary>
    /// Resizes the active Tier-1 VRAM slot limit elastically.
    /// </summary>
    public int ResizeElasticVramSlots(int targetActiveSlots)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _vramManager.SetActiveSlotLimit(targetActiveSlots);
        return _vramManager.ActiveSlotCount;
    }

    /// <summary>
    /// Applies a WDDM headroom snapshot to shrink/expand active VRAM expert slots safely away from the 0-free-VRAM cliff.
    /// </summary>
    public int ApplyVramHeadroomGuard(VramHeadroomSnapshot snapshot, int minSlotFloor = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int recommended = WddmVramGuard.ComputeRecommendedSlotCount(
            snapshot,
            (ulong)_weightBank.ExpertStrideBytes,
            _vramManager.ActiveSlotCount,
            _vramManager.SlotCount,
            minSlotFloor);

        return ResizeElasticVramSlots(recommended);
    }

    /// <summary>
    /// Resolves an expert's weight memory descriptor, demand-paging and streaming to VRAM or Tier-2 Host RAM as requested.
    /// </summary>
    public ExpertMemoryDescriptor AccessExpert(int layerIdx, int expertIdx, bool targetGpu)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        byte* hostPtr = _weightBank.GetExpertPointer(layerIdx, expertIdx);
        long rawBytes = _weightBank.ExpertRawBytes;

        if (!targetGpu)
        {
            if (_maxHostRamSlots <= 0)
            {
                // Unbounded Host RAM execution path: zero-copy pointer directly from memory-mapped bank
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

            bool inHostRam;
            lock (_tierLock)
            {
                if (_hostRamResidents.ContainsKey((layerIdx, expertIdx)))
                {
                    _hostRamResidents[(layerIdx, expertIdx)] = Stopwatch.GetTimestamp();
                    inHostRam = true;
                }
                else
                {
                    inHostRam = false;
                }
            }

            if (inHostRam)
            {
                return new ExpertMemoryDescriptor
                {
                    Tier = MemoryTier.HostRam,
                    DeviceAddress = IntPtr.Zero,
                    HostAddress = hostPtr,
                    ByteSize = rawBytes,
                    IsPinned = true,
                    SlotIndex = -1
                };
            }

            // Tier-3 cold storage fault: throttle concurrent NVMe page faults and promote into Tier-2 Host RAM
            _storageFaultGate.Wait();
            try
            {
                _weightBank.PrefetchExpertPages(layerIdx, expertIdx);
            }
            finally
            {
                _storageFaultGate.Release();
            }

            lock (_tierLock)
            {
                if (!_exclusiveHostRam || !_vramManager.ContainsSlot(layerIdx, expertIdx))
                {
                    PromoteToHostRamLocked(layerIdx, expertIdx);
                }
            }

            return new ExpertMemoryDescriptor
            {
                Tier = MemoryTier.Storage,
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
            if (_exclusiveHostRam)
            {
                lock (_tierLock)
                {
                    RemoveFromHostRamLocked(layerIdx, expertIdx);
                }
            }

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
        _vramManager.AllocateOrEvictSlot(layerIdx, expertIdx, out slotIdx, out int evL, out int evE, out devicePtr);

        // Stream expert weights to VRAM
        TransferToVram(hostPtr, devicePtr, (nuint)rawBytes);

        if (_exclusiveHostRam)
        {
            lock (_tierLock)
            {
                RemoveFromHostRamLocked(layerIdx, expertIdx);
                if (_maxHostRamSlots > 0 && evL >= 0 && evE >= 0 && !_vramManager.ContainsSlot(evL, evE))
                {
                    PromoteToHostRamLocked(evL, evE);
                }
            }
        }

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

    private void RemoveFromHostRamLocked(int layer, int expert)
    {
        var key = (layer, expert);
        _hostRamResidents.Remove(key);
        if (_osLockedPages.Remove(key))
        {
            try { _weightBank.UnlockExpertPages(layer, expert); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[TieredVirtualMemoryManager] UnlockExpertPages failed for L{layer}E{expert}: {ex.Message}");
            }
        }
    }

    private void PromoteToHostRamLocked(int layer, int expert)
    {
        if (_maxHostRamSlots <= 0)
            return;

        if (_exclusiveHostRam && _vramManager.ContainsSlot(layer, expert))
            return;

        var key = (layer, expert);
        if (_hostRamResidents.ContainsKey(key))
        {
            _hostRamResidents[key] = Stopwatch.GetTimestamp();
            return;
        }

        if (_hostRamResidents.Count >= _maxHostRamSlots)
        {
            (int Layer, int Expert) victimKey = default;
            bool hasVictim = false;
            double lowestScore = double.MaxValue;
            long oldestTs = long.MaxValue;
            var profile = Volatile.Read(ref _routingProfile);

            foreach (var kvp in _hostRamResidents)
            {
                double sc = profile != null ? profile.GetScore(kvp.Key.Layer, kvp.Key.Expert) : 0.0;
                bool worse = !hasVictim ||
                             sc < lowestScore - 1e-9 ||
                             (Math.Abs(sc - lowestScore) <= 1e-9 && kvp.Value < oldestTs);
                if (worse)
                {
                    hasVictim = true;
                    victimKey = kvp.Key;
                    lowestScore = sc;
                    oldestTs = kvp.Value;
                }
            }

            if (hasVictim)
            {
                RemoveFromHostRamLocked(victimKey.Layer, victimKey.Expert);
            }
        }

        _hostRamResidents[key] = Stopwatch.GetTimestamp();
        try
        {
            _weightBank.PrefetchExpertPages(layer, expert);
            if (_weightBank.LockExpertPages(layer, expert))
            {
                _osLockedPages.Add(key);
            }
        }
        catch
        {
            // OS VirtualLock quotas vary; logical Host-RAM residency tracking remains intact
        }
    }

    /// <summary>
    /// Streams host memory weights into GPU VRAM with timing telemetry.
    /// </summary>
    private void TransferToVram(byte* hostPtr, IntPtr devicePtr, nuint byteCount)
    {
        long start = Stopwatch.GetTimestamp();

        if (_hasCuda && devicePtr != IntPtr.Zero)
        {
            CopyHostToCudaDevice(devicePtr, (IntPtr)hostPtr, byteCount);
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

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void CopyHostToCudaDevice(IntPtr devicePtr, IntPtr hostPtr, nuint byteCount)
    {
        _ = CuDriver.MemcpyHtoD(devicePtr, hostPtr, byteCount);
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

                // 1. Advise OS kernel to pull pages from NVMe into Host RAM working set (gated against thrashing)
                _storageFaultGate.Wait(cancellationToken);
                try
                {
                    _weightBank.PrefetchExpertPages(layerIdx, expIdx);
                }
                finally
                {
                    _storageFaultGate.Release();
                }

                // 2. If VRAM staging requested and not already in VRAM, stage in background
                if (prefetchToVram && _vramManager.ActiveSlotCount > 0)
                {
                    if (!_vramManager.TryGetSlot(layerIdx, expIdx, out _, out _))
                    {
                        byte* hostPtr = _weightBank.GetExpertPointer(layerIdx, expIdx);
                        _vramManager.AllocateOrEvictSlot(layerIdx, expIdx, out _, out int evL, out int evE, out IntPtr dptr);
                        TransferToVram(hostPtr, dptr, (nuint)_weightBank.ExpertRawBytes);

                        if (_exclusiveHostRam)
                        {
                            lock (_tierLock)
                            {
                                RemoveFromHostRamLocked(layerIdx, expIdx);
                                if (_maxHostRamSlots > 0 && evL >= 0 && evE >= 0 && !_vramManager.ContainsSlot(evL, evE))
                                {
                                    PromoteToHostRamLocked(evL, evE);
                                }
                            }
                        }
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
            TotalPcieTransferTimeMs: Volatile.Read(ref _totalPcieTransferTimeMs))
        {
            ActiveVramSlots = _vramManager.ActiveSlotCount,
            HostRamResidentCount = HostRamResidentCount,
            MaxHostRamResidentSlots = _maxHostRamSlots,
            AdaptiveSwapCount = _vramManager.AdaptiveSwapCount
        };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            lock (_tierLock)
            {
                foreach (var key in _osLockedPages)
                {
                    try { _weightBank.UnlockExpertPages(key.Layer, key.Expert); }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[TieredVirtualMemoryManager] UnlockExpertPages Dispose failed for L{key.Layer}E{key.Expert}: {ex.Message}");
                    }
                }
                _osLockedPages.Clear();
                _hostRamResidents.Clear();
            }

            _vramManager.Dispose();
            _pinnedPool.Dispose();
            _weightBank.Dispose();
            _storageFaultGate.Dispose();
        }
    }
}
