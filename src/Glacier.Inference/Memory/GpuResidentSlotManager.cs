namespace Glacier.Inference.Memory;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Glacier.Inference.Gpu;

/// <summary>
/// Eviction policy used by <see cref="GpuResidentSlotManager"/> when all active slots are occupied.
/// </summary>
public enum SlotEvictionPolicy
{
    Lru = 0,
    DecayedFrequency = 1
}

/// <summary>
/// Manages Tier 1 GPU VRAM resident expert slots using configurable LRU or decayed-frequency eviction
/// with pinned expert support and elastic active slot limits.
/// Eliminates runtime VRAM fragmentation and OutOfMemoryException by enforcing a fixed working-set slot pool.
/// </summary>
public sealed class GpuResidentSlotManager : IDisposable
{
    public struct ExpertSlot
    {
        public int SlotIndex;
        public int LayerIndex;
        public int ExpertIndex;
        public long LastAccessTimestamp;
        public long AccessCount;
        public bool IsPinned;
        public IntPtr DevicePointer;
        public bool IsOccupied => LayerIndex >= 0 && ExpertIndex >= 0;
    }

    private readonly int _slotCount;
    private readonly nuint _slotByteSize;
    private readonly bool _allocateVram;
    private readonly ExpertSlot[] _slots;
    private readonly Lock _lock = new();

    private int _activeSlotLimit;
    private SlotEvictionPolicy _evictionPolicy;
    private ExpertRoutingProfile? _routingProfile;

    private long _hitCount;
    private long _missCount;
    private long _evictionCount;
    private long _adaptiveSwapCount;
    private bool _disposed;

    public int SlotCount => _slotCount;
    public int ActiveSlotCount
    {
        get
        {
            lock (_lock)
            {
                return _activeSlotLimit;
            }
        }
    }
    public nuint SlotByteSize => _slotByteSize;
    public SlotEvictionPolicy EvictionPolicy
    {
        get
        {
            lock (_lock)
            {
                return _evictionPolicy;
            }
        }
    }
    public ExpertRoutingProfile? RoutingProfile
    {
        get
        {
            lock (_lock)
            {
                return _routingProfile;
            }
        }
    }
    public long HitCount => Volatile.Read(ref _hitCount);
    public long MissCount => Volatile.Read(ref _missCount);
    public long EvictionCount => Volatile.Read(ref _evictionCount);
    public long AdaptiveSwapCount => Volatile.Read(ref _adaptiveSwapCount);
    public double HitRatio
    {
        get
        {
            long hits = HitCount;
            long total = hits + MissCount;
            return total == 0 ? 0.0 : (double)hits / total;
        }
    }

    public int OccupiedSlotCount
    {
        get
        {
            lock (_lock)
            {
                int count = 0;
                for (int i = 0; i < _activeSlotLimit; i++)
                {
                    if (_slots[i].IsOccupied) count++;
                }
                return count;
            }
        }
    }

    /// <summary>
    /// Gets the number of active unpinned slots in VRAM that can be safely evicted or allocated.
    /// </summary>
    public int UnpinnedSlotCount
    {
        get
        {
            lock (_lock)
            {
                int count = 0;
                for (int i = 0; i < _activeSlotLimit; i++)
                {
                    if (!_slots[i].IsPinned) count++;
                }
                return count;
            }
        }
    }

    public GpuResidentSlotManager(int slotCount, nuint slotByteSize, bool allocateVram = true)
        : this(slotCount, slotByteSize, allocateVram, SlotEvictionPolicy.Lru, routingProfile: null, initialActiveSlots: slotCount)
    {
    }

    public GpuResidentSlotManager(
        int slotCount,
        nuint slotByteSize,
        bool allocateVram,
        SlotEvictionPolicy evictionPolicy,
        ExpertRoutingProfile? routingProfile = null,
        int initialActiveSlots = -1)
    {
        if (slotCount <= 0) throw new ArgumentOutOfRangeException(nameof(slotCount));
        if (slotByteSize == 0) throw new ArgumentOutOfRangeException(nameof(slotByteSize));

        _slotCount = slotCount;
        _slotByteSize = slotByteSize;
        _slots = new ExpertSlot[slotCount];
        _evictionPolicy = evictionPolicy;
        _routingProfile = routingProfile;
        _activeSlotLimit = initialActiveSlots >= 0
            ? Math.Clamp(initialActiveSlots, 0, slotCount)
            : slotCount;

        bool vramAllocated = false;
        if (allocateVram && CuDriver.IsAvailable())
        {
            vramAllocated = TryAllocateCudaSlots(_slots, slotCount, slotByteSize);
        }

        _allocateVram = vramAllocated;

        // Initialize unallocated slots with simulated pointer or placeholder
        for (int i = 0; i < slotCount; i++)
        {
            if (_slots[i].DevicePointer == IntPtr.Zero)
            {
                _slots[i] = new ExpertSlot
                {
                    SlotIndex = i,
                    LayerIndex = -1,
                    ExpertIndex = -1,
                    DevicePointer = (IntPtr)(0x10000000 + ((long)i * (long)slotByteSize))
                };
            }
        }
    }

    /// <summary>
    /// Configures the runtime eviction policy and optional decayed-frequency profile.
    /// </summary>
    public void ConfigurePolicy(SlotEvictionPolicy policy, ExpertRoutingProfile? routingProfile = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            _evictionPolicy = policy;
            if (routingProfile != null || policy == SlotEvictionPolicy.Lru)
            {
                _routingProfile = routingProfile;
            }
        }
    }

    /// <summary>
    /// Adjusts the active slot window <c>[0 .. newActiveSlotLimit - 1]</c>, clearing any occupied
    /// slots outside the shrunk boundary and returning how many occupied slots were invalidated.
    /// </summary>
    public int SetActiveSlotLimit(int newActiveSlotLimit, Span<int> invalidatedSlots = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int clamped = Math.Clamp(newActiveSlotLimit, 0, _slotCount);
        int clearedCount = 0;

        lock (_lock)
        {
            if (clamped < _activeSlotLimit)
            {
                for (int i = clamped; i < _activeSlotLimit; i++)
                {
                    ref var slot = ref _slots[i];
                    if (slot.IsOccupied)
                    {
                        if (clearedCount < invalidatedSlots.Length)
                        {
                            invalidatedSlots[clearedCount] = i;
                        }
                        clearedCount++;
                    }
                    slot.LayerIndex = -1;
                    slot.ExpertIndex = -1;
                    slot.IsPinned = false;
                    slot.AccessCount = 0;
                    slot.LastAccessTimestamp = 0;
                }
            }
            _activeSlotLimit = clamped;
        }

        return clearedCount;
    }

    /// <summary>
    /// Checks whether the specified expert is currently resident without mutating hit/miss telemetry.
    /// </summary>
    public bool ContainsSlot(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref readonly var slot = ref _slots[i];
                if (slot.LayerIndex == layerIdx && slot.ExpertIndex == expertIdx)
                    return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Returns the device pointer bound to <paramref name="slotIdx"/> without mutating hit/miss telemetry.
    /// </summary>
    public IntPtr GetSlotDevicePointer(int slotIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)slotIdx >= (uint)_slots.Length)
            throw new ArgumentOutOfRangeException(nameof(slotIdx));

        lock (_lock)
        {
            return _slots[slotIdx].DevicePointer;
        }
    }

    /// <summary>
    /// Checks whether the specified expert is currently resident in VRAM.
    /// Updates LRU timestamp on cache hit.
    /// </summary>
    public bool TryGetSlot(int layerIdx, int expertIdx, out int slotIdx, out IntPtr devicePtr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (slot.LayerIndex == layerIdx && slot.ExpertIndex == expertIdx)
                {
                    slot.LastAccessTimestamp = Stopwatch.GetTimestamp();
                    slot.AccessCount++;
                    Interlocked.Increment(ref _hitCount);
                    slotIdx = i;
                    devicePtr = slot.DevicePointer;
                    return true;
                }
            }
        }

        Interlocked.Increment(ref _missCount);
        slotIdx = -1;
        devicePtr = IntPtr.Zero;
        return false;
    }

    /// <summary>
    /// Computes the number of unpinned slots available for new expert transfers in the specified layer,
    /// ensuring that slots currently holding resident hits for that layer are preserved and not evicted.
    /// </summary>
    public int GetAvailableTransferSlots(int layerIdx, IReadOnlyList<int> residentHits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            int available = 0;
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref readonly var slot = ref _slots[i];
                if (slot.IsPinned) continue;

                if (slot.LayerIndex == layerIdx)
                {
                    bool isResidentHit = false;
                    for (int j = 0; j < residentHits.Count; j++)
                    {
                        if (slot.ExpertIndex == residentHits[j])
                        {
                            isResidentHit = true;
                            break;
                        }
                    }
                    if (isResidentHit) continue;
                }

                available++;
            }
            return available;
        }
    }

    /// <summary>
    /// Span-based overload of GetAvailableTransferSlots for zero-allocation callers.
    /// </summary>
    public int GetAvailableTransferSlots(int layerIdx, ReadOnlySpan<int> residentHits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            int available = 0;
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref readonly var slot = ref _slots[i];
                if (slot.IsPinned) continue;

                if (slot.LayerIndex == layerIdx)
                {
                    bool isResidentHit = false;
                    for (int j = 0; j < residentHits.Length; j++)
                    {
                        if (slot.ExpertIndex == residentHits[j])
                        {
                            isResidentHit = true;
                            break;
                        }
                    }
                    if (isResidentHit) continue;
                }

                available++;
            }
            return available;
        }
    }

    /// <summary>
    /// Allocates an empty slot or evicts an unpinned slot according to the configured eviction policy.
    /// </summary>
    public bool AllocateOrEvictSlot(int layerIdx, int expertIdx, out int slotIdx, out int evictedLayer, out int evictedExpert, out IntPtr devicePtr)
        => AllocateOrEvictSlot(layerIdx, expertIdx, ReadOnlySpan<int>.Empty, out slotIdx, out evictedLayer, out evictedExpert, out devicePtr);

    /// <summary>
    /// Allocates an empty slot or evicts an unpinned slot, strictly preserving any experts in
    /// <paramref name="protectedLayerExperts"/> for <paramref name="layerIdx"/>.
    /// </summary>
    public bool AllocateOrEvictSlot(
        int layerIdx,
        int expertIdx,
        ReadOnlySpan<int> protectedLayerExperts,
        out int slotIdx,
        out int evictedLayer,
        out int evictedExpert,
        out IntPtr devicePtr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (_activeSlotLimit <= 0)
            {
                throw new InvalidOperationException("Active GPU resident slot limit is 0.");
            }

            // 1. Check if already resident
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (slot.LayerIndex == layerIdx && slot.ExpertIndex == expertIdx)
                {
                    slot.LastAccessTimestamp = Stopwatch.GetTimestamp();
                    slot.AccessCount++;
                    slotIdx = i;
                    evictedLayer = -1;
                    evictedExpert = -1;
                    devicePtr = slot.DevicePointer;
                    return false; // Already existed
                }
            }

            // 2. Search for empty slot
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (!slot.IsOccupied)
                {
                    slot.LayerIndex = layerIdx;
                    slot.ExpertIndex = expertIdx;
                    slot.LastAccessTimestamp = Stopwatch.GetTimestamp();
                    slot.AccessCount = 1;
                    slot.IsPinned = false;
                    slotIdx = i;
                    evictedLayer = -1;
                    evictedExpert = -1;
                    devicePtr = slot.DevicePointer;
                    return true;
                }
            }

            // 3. Find victim unpinned & unprotected candidate to evict
            int victimIdx = FindEvictionVictimLocked(layerIdx, protectedLayerExperts, out _);

            if (victimIdx < 0)
            {
                throw new InvalidOperationException("All GPU resident expert slots are pinned. Cannot evict.");
            }

            ref var evictSlot = ref _slots[victimIdx];
            evictedLayer = evictSlot.LayerIndex;
            evictedExpert = evictSlot.ExpertIndex;

            evictSlot.LayerIndex = layerIdx;
            evictSlot.ExpertIndex = expertIdx;
            evictSlot.LastAccessTimestamp = Stopwatch.GetTimestamp();
            evictSlot.AccessCount = 1;
            evictSlot.IsPinned = false;

            Interlocked.Increment(ref _evictionCount);
            slotIdx = victimIdx;
            devicePtr = evictSlot.DevicePointer;
            return true;
        }
    }

    /// <summary>
    /// Identifies the coldest unpinned slot (excluding <paramref name="protectedLayerExperts"/> on
    /// <paramref name="excludeLayerIdx"/>) under the current eviction policy.
    /// </summary>
    public bool TryGetColdestUnpinnedSlot(
        int excludeLayerIdx,
        ReadOnlySpan<int> protectedLayerExperts,
        out int slotIdx,
        out int layerIdx,
        out int expertIdx,
        out double score)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            if (_activeSlotLimit <= 0)
            {
                slotIdx = -1;
                layerIdx = -1;
                expertIdx = -1;
                score = 0.0;
                return false;
            }

            int victimIdx = FindEvictionVictimLocked(excludeLayerIdx, protectedLayerExperts, out double victimScore, forceFrequencyScore: true);
            if (victimIdx < 0)
            {
                slotIdx = -1;
                layerIdx = -1;
                expertIdx = -1;
                score = 0.0;
                return false;
            }

            ref readonly var s = ref _slots[victimIdx];
            slotIdx = victimIdx;
            layerIdx = s.LayerIndex;
            expertIdx = s.ExpertIndex;
            score = victimScore;
            return true;
        }
    }

    /// <summary>
    /// Plans and commits up to <paramref name="maxSwaps"/> hot/cold slot replacements using empirical
    /// scores from <paramref name="profile"/> when non-resident experts exceed the coldest resident slot
    /// by <paramref name="minScoreAdvantageRatio"/>.
    /// </summary>
    public int PlanAdaptiveSwaps(
        ExpertRoutingProfile profile,
        int maxSwaps,
        double minScoreAdvantageRatio,
        Span<(int SlotIndex, int NewLayer, int NewExpert, int EvictedLayer, int EvictedExpert)> swaps)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (maxSwaps <= 0 || swaps.IsEmpty)
            return 0;

        int limit = Math.Min(maxSwaps, swaps.Length);
        int swapsWritten = 0;

        lock (_lock)
        {
            if (_activeSlotLimit <= 0)
                return 0;

            int candidateCapacity = Math.Min(profile.NumLayers * profile.NumExperts, _activeSlotLimit + limit * 4);
            if (candidateCapacity <= 0)
                return 0;

            var rentedCandidates = candidateCapacity > 128
                ? new (int LayerIndex, int ExpertIndex, double Score)[candidateCapacity]
                : null;
            Span<(int LayerIndex, int ExpertIndex, double Score)> candidates = rentedCandidates ?? stackalloc (int, int, double)[candidateCapacity];

            int candidateCount = profile.GetTopLayerExpertPairs(candidates, includeZeroScores: false);
            if (candidateCount <= 0)
                return 0;

            Span<bool> slotTouched = _activeSlotLimit <= 512
                ? stackalloc bool[_activeSlotLimit]
                : new bool[_activeSlotLimit];

            double advantage = Math.Max(1.01, minScoreAdvantageRatio);

            for (int c = 0; c < candidateCount && swapsWritten < limit; c++)
            {
                var (candLayer, candExp, candScore) = candidates[c];
                if (candScore <= 0.0)
                    break;

                bool alreadyResident = false;
                int emptyIdx = -1;
                for (int i = 0; i < _activeSlotLimit; i++)
                {
                    ref readonly var slot = ref _slots[i];
                    if (slot.LayerIndex == candLayer && slot.ExpertIndex == candExp)
                    {
                        alreadyResident = true;
                        break;
                    }
                    if (!slot.IsOccupied && emptyIdx < 0 && !slotTouched[i])
                    {
                        emptyIdx = i;
                    }
                }

                if (alreadyResident)
                    continue;

                if (emptyIdx >= 0)
                {
                    ref var targetSlot = ref _slots[emptyIdx];
                    targetSlot.LayerIndex = candLayer;
                    targetSlot.ExpertIndex = candExp;
                    targetSlot.LastAccessTimestamp = Stopwatch.GetTimestamp();
                    targetSlot.AccessCount = 1;
                    targetSlot.IsPinned = false;

                    slotTouched[emptyIdx] = true;
                    swaps[swapsWritten++] = (emptyIdx, candLayer, candExp, -1, -1);
                    Interlocked.Increment(ref _adaptiveSwapCount);
                    continue;
                }

                // Find coldest unpinned, untouched resident slot
                int coldestIdx = -1;
                double coldestScore = double.MaxValue;
                long oldestTs = long.MaxValue;

                for (int i = 0; i < _activeSlotLimit; i++)
                {
                    if (slotTouched[i]) continue;
                    ref readonly var slot = ref _slots[i];
                    if (slot.IsPinned) continue;

                    double sc = slot.IsOccupied ? profile.GetScore(slot.LayerIndex, slot.ExpertIndex) : 0.0;
                    bool isColder = sc < coldestScore - 1e-9 ||
                                    (Math.Abs(sc - coldestScore) <= 1e-9 && slot.LastAccessTimestamp < oldestTs);
                    if (isColder)
                    {
                        coldestScore = sc;
                        oldestTs = slot.LastAccessTimestamp;
                        coldestIdx = i;
                    }
                }

                if (coldestIdx < 0)
                    break;

                double threshold = Math.Max(0.05, coldestScore * advantage);
                if (candScore < threshold)
                    break;

                ref var replaceSlot = ref _slots[coldestIdx];
                int evLayer = replaceSlot.LayerIndex;
                int evExp = replaceSlot.ExpertIndex;

                replaceSlot.LayerIndex = candLayer;
                replaceSlot.ExpertIndex = candExp;
                replaceSlot.LastAccessTimestamp = Stopwatch.GetTimestamp();
                replaceSlot.AccessCount = 1;
                replaceSlot.IsPinned = false;

                slotTouched[coldestIdx] = true;
                swaps[swapsWritten++] = (coldestIdx, candLayer, candExp, evLayer, evExp);
                Interlocked.Increment(ref _adaptiveSwapCount);
                Interlocked.Increment(ref _evictionCount);
            }
        }

        return swapsWritten;
    }

    private int FindEvictionVictimLocked(
        int layerIdx,
        ReadOnlySpan<int> protectedLayerExperts,
        out double victimScore,
        bool forceFrequencyScore = false)
    {
        int victimIdx = -1;
        long oldestTime = long.MaxValue;
        double lowestScore = double.MaxValue;
        bool useFreq = forceFrequencyScore || _evictionPolicy == SlotEvictionPolicy.DecayedFrequency;

        for (int i = 0; i < _activeSlotLimit; i++)
        {
            ref readonly var slot = ref _slots[i];
            if (slot.IsPinned) continue;
            if (layerIdx >= 0 && slot.LayerIndex == layerIdx && protectedLayerExperts.Contains(slot.ExpertIndex))
                continue;

            if (!useFreq)
            {
                if (slot.LastAccessTimestamp < oldestTime)
                {
                    oldestTime = slot.LastAccessTimestamp;
                    victimIdx = i;
                }
            }
            else
            {
                double sc = (_routingProfile != null && slot.IsOccupied)
                    ? _routingProfile.GetScore(slot.LayerIndex, slot.ExpertIndex)
                    : (double)slot.AccessCount;

                bool worse = sc < lowestScore - 1e-9 ||
                             (Math.Abs(sc - lowestScore) <= 1e-9 && slot.LastAccessTimestamp < oldestTime);
                if (worse)
                {
                    lowestScore = sc;
                    oldestTime = slot.LastAccessTimestamp;
                    victimIdx = i;
                }
            }
        }

        victimScore = victimIdx >= 0 && useFreq ? lowestScore : 0.0;
        return victimIdx;
    }

    /// <summary>
    /// Pins an expert slot in VRAM, exempting it from eviction.
    /// </summary>
    public bool PinExpert(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (slot.LayerIndex == layerIdx && slot.ExpertIndex == expertIdx)
                {
                    slot.IsPinned = true;
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Unpins an expert slot in VRAM, allowing normal eviction.
    /// </summary>
    public bool UnpinExpert(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (slot.LayerIndex == layerIdx && slot.ExpertIndex == expertIdx)
                {
                    slot.IsPinned = false;
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>
    /// Flushes all non-pinned active slots from VRAM.
    /// </summary>
    public void ClearNonPinned()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _activeSlotLimit; i++)
            {
                ref var slot = ref _slots[i];
                if (!slot.IsPinned)
                {
                    slot.LayerIndex = -1;
                    slot.ExpertIndex = -1;
                    slot.AccessCount = 0;
                }
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_allocateVram)
            {
                FreeCudaSlots(_slots);
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool TryAllocateCudaSlots(ExpertSlot[] slots, int slotCount, nuint slotByteSize)
    {
        try
        {
            for (int i = 0; i < slotCount; i++)
            {
                int err = CuDriver.MemAlloc(out IntPtr dptr, slotByteSize);
                if (err == 0 && dptr != IntPtr.Zero)
                {
                    slots[i] = new ExpertSlot
                    {
                        SlotIndex = i,
                        LayerIndex = -1,
                        ExpertIndex = -1,
                        DevicePointer = dptr
                    };
                }
                else
                {
                    break;
                }
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void FreeCudaSlots(ExpertSlot[] slots)
    {
        for (int i = 0; i < slots.Length; i++)
        {
            if (slots[i].DevicePointer != IntPtr.Zero)
            {
                try { CuDriver.MemFree(slots[i].DevicePointer); } catch { }
                slots[i].DevicePointer = IntPtr.Zero;
            }
        }
    }
}
