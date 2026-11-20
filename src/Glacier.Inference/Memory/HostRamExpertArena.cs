namespace Glacier.Inference.Memory;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Model;
using Microsoft.Win32.SafeHandles;

/// <summary>
/// Allocates page-aligned unmanaged host-RAM slabs sized to live physical RAM headroom and
/// sequentially bulk-loads MoE routed expert tensors (<c>ffn_*_exps.weight</c>), repointing
/// <see cref="LayerWeights"/> pointers directly into pinned/owned DRAM so OS <c>mmap</c> standby
/// eviction cannot trigger NVMe page faults during token generation.
/// </summary>
public sealed unsafe class HostRamExpertArena : IDisposable
{
    private const int PageSize = 4096;
    private const int FileFlagNoBuffering = 0x20000000;
    private const int ReadChunkBytes = 16 * 1024 * 1024; // 16 MiB sequential I/O chunks
    private const ulong UnbufferedThresholdBytes = 4UL * 1024 * 1024;

    private readonly List<IntPtr> _slabs = new();
    private readonly bool[] _residentLayerMask;
    private bool _disposed;

    public int TotalMoeLayers { get; }
    public int ResidentMoeLayers { get; }
    public ulong AllocatedBytes { get; }
    public double LoadSeconds { get; }
    public double LoadBandwidthGBps => LoadSeconds > 0.0 ? (AllocatedBytes / 1e9) / LoadSeconds : 0.0;
    public bool UsedUnbufferedIo { get; }

    /// <summary>
    /// Returns <see langword="true"/> when <paramref name="layerIdx"/>'s routed expert tensors
    /// are pinned inside this host-RAM arena.
    /// </summary>
    public bool IsLayerResident(int layerIdx)
        => (uint)layerIdx < (uint)_residentLayerMask.Length && _residentLayerMask[layerIdx];

    private HostRamExpertArena(
        List<IntPtr> slabs,
        int totalMoeLayers,
        int residentMoeLayers,
        ulong allocatedBytes,
        double loadSeconds,
        bool usedUnbufferedIo,
        bool[] residentLayerMask)
    {
        _slabs = slabs;
        TotalMoeLayers = totalMoeLayers;
        ResidentMoeLayers = residentMoeLayers;
        AllocatedBytes = allocatedBytes;
        LoadSeconds = loadSeconds;
        UsedUnbufferedIo = usedUnbufferedIo;
        _residentLayerMask = residentLayerMask;
    }

    internal static HostRamExpertArena CreateSyntheticForTesting(
        int totalMoeLayers,
        bool[] residentLayerMask,
        ulong allocatedBytes = 0,
        double loadSeconds = 0.001)
    {
        ArgumentNullException.ThrowIfNull(residentLayerMask);
        if (totalMoeLayers < 0)
            throw new ArgumentOutOfRangeException(nameof(totalMoeLayers));

        var maskCopy = (bool[])residentLayerMask.Clone();
        int residentCount = 0;
        for (int i = 0; i < maskCopy.Length; i++)
        {
            if (maskCopy[i]) residentCount++;
        }

        return new HostRamExpertArena(
            new List<IntPtr>(),
            totalMoeLayers,
            residentCount,
            allocatedBytes,
            loadSeconds,
            usedUnbufferedIo: false,
            maskCopy);
    }

    private enum ExpertTensorSlot
    {
        Gate,
        Up,
        Down,
        GateUp
    }

    private readonly record struct TensorCopySpec(
        LayerWeights Layer,
        ExpertTensorSlot Slot,
        string FilePath,
        long FileOffset,
        ulong ByteSize);

    /// <summary>
    /// Attempts to load as many MoE layers' routed expert tensors as fit inside <paramref name="budgetBytes"/>
    /// into owned unmanaged host RAM, updating <see cref="LayerWeights"/> pointers in place.
    /// Returns <c>null</c> when the model has no MoE layers, budget is 0, or not even one layer fits.
    /// </summary>
    public static HostRamExpertArena? TryAttach(ModelWeights weights, ulong budgetBytes, bool preferUnbufferedIo = true)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!weights.IsMoe || budgetBytes == 0)
            return null;

        int totalMoeLayers = 0;
        var layerPlans = new List<(int LayerIdx, List<TensorCopySpec> Specs)>(weights.BlockCount);

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw == null || !lw.IsMoe)
                continue;

            totalMoeLayers++;
            var layerSpecs = new List<TensorCopySpec>(3);
            TryCollectTensor(weights, lw, l, $"blk.{l}.ffn_gate_exps.weight", ExpertTensorSlot.Gate, lw.FfnGateExpsWeight != null, layerSpecs);
            TryCollectTensor(weights, lw, l, $"blk.{l}.ffn_up_exps.weight", ExpertTensorSlot.Up, lw.FfnUpExpsWeight != null, layerSpecs);
            TryCollectTensor(weights, lw, l, $"blk.{l}.ffn_down_exps.weight", ExpertTensorSlot.Down, lw.FfnDownExpsWeight != null, layerSpecs);
            TryCollectTensor(weights, lw, l, $"blk.{l}.ffn_gate_up_exps.weight", ExpertTensorSlot.GateUp, lw.FfnGateUpExpsWeight != null, layerSpecs);

            if (layerSpecs.Count > 0)
            {
                layerPlans.Add((l, layerSpecs));
            }
        }

        if (totalMoeLayers == 0 || layerPlans.Count == 0)
            return null;

        ulong remainingBudget = budgetBytes;
        int residentLayers = 0;
        bool[] residentLayerMask = new bool[weights.BlockCount];
        var selected = new List<TensorCopySpec>();

        for (int i = 0; i < layerPlans.Count; i++)
        {
            var (layerIdx, specs) = layerPlans[i];
            ulong layerBytes = 0;
            for (int j = 0; j < specs.Count; j++)
            {
                layerBytes += AlignUpPage(specs[j].ByteSize + PageSize);
            }

            if (layerBytes > remainingBudget)
                break;

            remainingBudget -= layerBytes;
            residentLayers++;
            residentLayerMask[layerIdx] = true;
            selected.AddRange(specs);
        }

        if (residentLayers == 0 || selected.Count == 0)
            return null;

        long startTs = Stopwatch.GetTimestamp();
        var slabs = new List<IntPtr>(selected.Count);
        var handles = new Dictionary<string, (SafeFileHandle? Unbuffered, SafeFileHandle Buffered, long FileLength)>(StringComparer.OrdinalIgnoreCase);
        ulong totalAllocated = 0;
        bool anyUnbuffered = false;

        try
        {
            for (int i = 0; i < selected.Count; i++)
            {
                var spec = selected[i];
                if (!handles.TryGetValue(spec.FilePath, out var hSet))
                {
                    long fileLen = new FileInfo(spec.FilePath).Length;
                    SafeFileHandle? unbuffered = null;
                    if (OperatingSystem.IsWindows() && preferUnbufferedIo)
                    {
                        try
                        {
                            unbuffered = File.OpenHandle(
                                spec.FilePath,
                                FileMode.Open,
                                FileAccess.Read,
                                FileShare.ReadWrite | FileShare.Delete,
                                (FileOptions)FileFlagNoBuffering | FileOptions.SequentialScan);
                        }
                        catch
                        {
                            unbuffered = null;
                        }
                    }

                    var buffered = File.OpenHandle(
                        spec.FilePath,
                        FileMode.Open,
                        FileAccess.Read,
                        FileShare.ReadWrite | FileShare.Delete,
                        FileOptions.SequentialScan);

                    hSet = (unbuffered, buffered, fileLen);
                    handles[spec.FilePath] = hSet;
                }

                bool useUnbuffered = hSet.Unbuffered != null && spec.ByteSize >= UnbufferedThresholdBytes;
                long prefixBytes = useUnbuffered ? (spec.FileOffset & (PageSize - 1)) : 0L;
                long alignedStart = spec.FileOffset - prefixBytes;
                ulong rawPlusPrefix = (ulong)prefixBytes + spec.ByteSize;
                ulong allocBytes = AlignUpPage(rawPlusPrefix);

                byte* slab = (byte*)NativeMemory.AlignedAlloc((nuint)allocBytes, PageSize);
                slabs.Add((IntPtr)slab);
                totalAllocated += allocBytes;

                ReadTensorIntoSlab(
                    hSet.Unbuffered,
                    hSet.Buffered,
                    hSet.FileLength,
                    slab,
                    allocBytes,
                    alignedStart,
                    spec.FileOffset,
                    spec.ByteSize,
                    useUnbuffered,
                    out bool usedUnbuffered);

                if (usedUnbuffered)
                {
                    anyUnbuffered = true;
                }

                byte* dataPtr = slab + prefixBytes;
                switch (spec.Slot)
                {
                    case ExpertTensorSlot.Gate:
                        spec.Layer.FfnGateExpsWeight = dataPtr;
                        break;
                    case ExpertTensorSlot.Up:
                        spec.Layer.FfnUpExpsWeight = dataPtr;
                        break;
                    case ExpertTensorSlot.Down:
                        spec.Layer.FfnDownExpsWeight = dataPtr;
                        break;
                    case ExpertTensorSlot.GateUp:
                        spec.Layer.FfnGateUpExpsWeight = dataPtr;
                        break;
                }
            }
        }
        catch
        {
            foreach (IntPtr ptr in slabs)
            {
                if (ptr != IntPtr.Zero) NativeMemory.AlignedFree((void*)ptr);
            }
            throw;
        }
        finally
        {
            foreach (var pair in handles.Values)
            {
                pair.Unbuffered?.Dispose();
                pair.Buffered.Dispose();
            }
        }

        double elapsedSec = Stopwatch.GetElapsedTime(startTs).TotalSeconds;
        var arena = new HostRamExpertArena(slabs, totalMoeLayers, residentLayers, totalAllocated, elapsedSec, anyUnbuffered, residentLayerMask);
        GlacierDiagnostics.LogInformation(
            $"[HostRamExpertArena] Loaded {residentLayers}/{totalMoeLayers} MoE layers ({totalAllocated / (1024.0 * 1024.0):F1} MiB) in {elapsedSec:F2}s ({arena.LoadBandwidthGBps:F2} GB/s, Unbuffered={anyUnbuffered})");
        return arena;
    }

    private static void TryCollectTensor(
        ModelWeights weights,
        LayerWeights layer,
        int layerIdx,
        string tensorName,
        ExpertTensorSlot slot,
        bool presentOnLayer,
        List<TensorCopySpec> dst)
    {
        if (!presentOnLayer) return;
        if (!weights.Gguf.TryGetTensorFileLocation(tensorName, out string path, out long offset, out ulong size) || size == 0)
            return;

        if (!File.Exists(path)) return;
        dst.Add(new TensorCopySpec(layer, slot, path, offset, size));
    }

    private static void ReadTensorIntoSlab(
        SafeFileHandle? unbufferedHandle,
        SafeFileHandle bufferedHandle,
        long fileLength,
        byte* slab,
        ulong slabBytes,
        long alignedStart,
        long exactFileOffset,
        ulong exactByteSize,
        bool useUnbuffered,
        out bool actuallyUsedUnbuffered)
    {
        actuallyUsedUnbuffered = false;

        if (useUnbuffered && unbufferedHandle != null)
        {
            try
            {
                long endOffset = exactFileOffset + (long)exactByteSize;
                long fullSectorLimit = fileLength & ~(long)(PageSize - 1);
                long unbufferedEnd = Math.Min(fullSectorLimit, AlignUpPageSigned(endOffset));

                long cursor = alignedStart;
                ulong dstOffset = 0;
                while (cursor < unbufferedEnd)
                {
                    int toRead = (int)Math.Min(ReadChunkBytes, unbufferedEnd - cursor);
                    var span = new Span<byte>(slab + dstOffset, toRead);
                    int totalRead = 0;
                    while (totalRead < toRead)
                    {
                        int n = RandomAccess.Read(unbufferedHandle, span.Slice(totalRead), cursor + totalRead);
                        if (n <= 0)
                            throw new EndOfStreamException($"Unexpected EOF during unbuffered read at offset {cursor + totalRead}.");
                        totalRead += n;
                    }
                    cursor += toRead;
                    dstOffset += (ulong)toRead;
                }

                if (endOffset > unbufferedEnd)
                {
                    int rem = (int)(endOffset - unbufferedEnd);
                    ReadExact(bufferedHandle, new Span<byte>(slab + dstOffset, rem), unbufferedEnd);
                }

                actuallyUsedUnbuffered = dstOffset > 0;
                return;
            }
            catch
            {
                // Fall back to sequential buffered Read on filesystems rejecting NO_BUFFERING
            }
        }

        ulong remaining = exactByteSize;
        ulong dstBaseOffset = (ulong)(exactFileOffset - alignedStart);
        ulong offset = 0;
        long filePos = exactFileOffset;
        while (remaining > 0)
        {
            int chunk = (int)Math.Min((ulong)ReadChunkBytes, remaining);
            ReadExact(bufferedHandle, new Span<byte>(slab + dstBaseOffset + offset, chunk), filePos);
            offset += (ulong)chunk;
            filePos += chunk;
            remaining -= (ulong)chunk;
        }
    }

    private static void ReadExact(SafeFileHandle handle, Span<byte> dst, long fileOffset)
    {
        int read = 0;
        while (read < dst.Length)
        {
            int n = RandomAccess.Read(handle, dst.Slice(read), fileOffset + read);
            if (n <= 0)
                throw new EndOfStreamException($"Unexpected EOF reading GGUF tensor bytes at offset {fileOffset + read}.");
            read += n;
        }
    }

    private static ulong AlignUpPage(ulong bytes)
    {
        ulong mask = PageSize - 1;
        return (bytes + mask) & ~mask;
    }

    private static long AlignUpPageSigned(long bytes)
    {
        long mask = PageSize - 1;
        return (bytes + mask) & ~mask;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _slabs.Count; i++)
        {
            IntPtr ptr = _slabs[i];
            if (ptr != IntPtr.Zero)
            {
                NativeMemory.AlignedFree((void*)ptr);
            }
        }
        _slabs.Clear();
        GC.SuppressFinalize(this);
    }

    ~HostRamExpertArena()
    {
        Dispose();
    }
}
