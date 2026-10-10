namespace Glacier.Inference.Memory;

using System;
using System.Diagnostics;
using System.Threading;
using Glacier.Inference.Gpu;

/// <summary>
/// Manages Tier 1 GPU VRAM resident expert slots using an LRU eviction policy with pinned expert support.
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

    private long _hitCount;
    private long _missCount;
    private long _evictionCount;
    private bool _disposed;

    public int SlotCount => _slotCount;
    public nuint SlotByteSize => _slotByteSize;
    public long HitCount => Volatile.Read(ref _hitCount);
    public long MissCount => Volatile.Read(ref _missCount);
    public long EvictionCount => Volatile.Read(ref _evictionCount);
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
                for (int i = 0; i < _slots.Length; i++)
                {
                    if (_slots[i].IsOccupied) count++;
                }
                return count;
            }
        }
    }

    /// <summary>
    /// Gets the number of unpinned slots in VRAM that can be safely evicted or allocated.
    /// </summary>
    public int UnpinnedSlotCount
    {
        get
        {
            lock (_lock)
            {
                int count = 0;
                for (int i = 0; i < _slots.Length; i++)
                {
                    if (!_slots[i].IsPinned) count++;
                }
                return count;
            }
        }
    }

    public GpuResidentSlotManager(int slotCount, nuint slotByteSize, bool allocateVram = true)
    {
        if (slotCount <= 0) throw new ArgumentOutOfRangeException(nameof(slotCount));
        if (slotByteSize == 0) throw new ArgumentOutOfRangeException(nameof(slotByteSize));

        _slotCount = slotCount;
        _slotByteSize = slotByteSize;
        _slots = new ExpertSlot[slotCount];

        bool vramAllocated = false;
        if (allocateVram && CuDriver.IsAvailable())
        {
            try
            {
                for (int i = 0; i < slotCount; i++)
                {
                    int err = CuDriver.MemAlloc(out IntPtr dptr, slotByteSize);
                    if (err == 0 && dptr != IntPtr.Zero)
                    {
                        _slots[i] = new ExpertSlot
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
                vramAllocated = true;
            }
            catch
            {
                vramAllocated = false;
            }
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
    /// Checks whether the specified expert is currently resident in VRAM.
    /// Updates LRU timestamp on cache hit.
    /// </summary>
    public bool TryGetSlot(int layerIdx, int expertIdx, out int slotIdx, out IntPtr devicePtr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _slots.Length; i++)
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
            for (int i = 0; i < _slots.Length; i++)
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
            for (int i = 0; i < _slots.Length; i++)
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
    /// Allocates an empty slot or evicts the least recently used unpinned slot to house the new expert.
    /// </summary>
    public bool AllocateOrEvictSlot(int layerIdx, int expertIdx, out int slotIdx, out int evictedLayer, out int evictedExpert, out IntPtr devicePtr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            // 1. Check if already resident
            for (int i = 0; i < _slots.Length; i++)
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
            for (int i = 0; i < _slots.Length; i++)
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

            // 3. Find LRU unpinned candidate to evict
            int lruIdx = -1;
            long oldestTime = long.MaxValue;

            for (int i = 0; i < _slots.Length; i++)
            {
                ref var slot = ref _slots[i];
                if (!slot.IsPinned && slot.LastAccessTimestamp < oldestTime)
                {
                    oldestTime = slot.LastAccessTimestamp;
                    lruIdx = i;
                }
            }

            if (lruIdx < 0)
            {
                // All slots are pinned; cannot evict
                throw new InvalidOperationException("All GPU resident expert slots are pinned. Cannot evict.");
            }

            ref var evictSlot = ref _slots[lruIdx];
            evictedLayer = evictSlot.LayerIndex;
            evictedExpert = evictSlot.ExpertIndex;

            evictSlot.LayerIndex = layerIdx;
            evictSlot.ExpertIndex = expertIdx;
            evictSlot.LastAccessTimestamp = Stopwatch.GetTimestamp();
            evictSlot.AccessCount = 1;
            evictSlot.IsPinned = false;

            Interlocked.Increment(ref _evictionCount);
            slotIdx = lruIdx;
            devicePtr = evictSlot.DevicePointer;
            return true;
        }
    }

    /// <summary>
    /// Pins an expert slot in VRAM, exempting it from LRU eviction.
    /// </summary>
    public bool PinExpert(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _slots.Length; i++)
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
    /// Unpins an expert slot in VRAM, allowing normal LRU eviction.
    /// </summary>
    public bool UnpinExpert(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _slots.Length; i++)
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
    /// Flushes all non-pinned slots from VRAM.
    /// </summary>
    public void ClearNonPinned()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        lock (_lock)
        {
            for (int i = 0; i < _slots.Length; i++)
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
                for (int i = 0; i < _slots.Length; i++)
                {
                    if (_slots[i].DevicePointer != IntPtr.Zero)
                    {
                        try { CuDriver.MemFree(_slots[i].DevicePointer); } catch { }
                        _slots[i].DevicePointer = IntPtr.Zero;
                    }
                }
            }
        }
    }
}
