namespace Glacier.Inference.Memory;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Microsoft.Win32.SafeHandles;

/// <summary>
/// Configuration parameters and deterministic hash tables for Qwen3.8-Flash-Next Per-Layer N-gram Embeddings (PLE).
/// </summary>
public sealed class PleHashConfig
{
    public static readonly ulong[] DefaultMultipliers =
    [
        23703573157769UL,
        20109073645365UL,
        8052911324071UL
    ];

    public static readonly long[] DefaultHeadVocabSizes =
    [
        20000003L, 20000023L, 20000033L, 20000047L,
        20000059L, 20000063L, 20000069L, 20000077L,
        20000081L, 20000093L, 20000107L, 20000147L,
        20000153L, 20000159L, 20000161L, 20000171L
    ];

    public static readonly long[] DefaultHeadOffsets =
    [
        0L,         20000003L,  40000026L,  60000059L,
        80000106L,  100000165L, 120000228L, 140000297L,
        160000374L, 180000455L, 200000548L, 220000655L,
        240000802L, 260000955L, 280001114L, 300001275L
    ];

    public static PleHashConfig DefaultQwen4Exp { get; } = new();

    public int NgramSize { get; init; } = 3;
    public int HeadsPerNgram { get; init; } = 8;
    public int HeadDim { get; init; } = 160;
    public int ConvKernel { get; init; } = 4;
    public int EosTokenId { get; init; } = 248044;
    public int ImageTokenId { get; init; } = -1;
    public ulong[] Multipliers { get; init; } = DefaultMultipliers;
    public long[] HeadVocabSizes { get; init; } = DefaultHeadVocabSizes;
    public long[] HeadOffsets { get; init; } = DefaultHeadOffsets;

    public int TotalHeads => (NgramSize - 1) * HeadsPerNgram;
    public int EmbeddingDim => TotalHeads * HeadDim;

    public long TotalRows
    {
        get
        {
            long sum = 0;
            for (int i = 0; i < HeadVocabSizes.Length; i++)
            {
                sum += HeadVocabSizes[i];
            }
            return sum;
        }
    }
}

/// <summary>
/// Implements the exact Strata/Qwen3.8-Flash-Next 64-bit multiply/XOR rolling n-gram hash
/// with forward EOS boundary cut propagation.
/// </summary>
public static class PleNgramHasher
{
    /// <summary>
    /// Computes the N PLE table row indices for <paramref name="token"/> given its oldest-first predecessors
    /// (e.g. [t_{-2}, t_{-1}]). Missing predecessors may be omitted or passed as negative IDs (&lt; 0).
    /// </summary>
    public static void ComputeNgramRows(
        int token,
        ReadOnlySpan<int> prevOldestFirst,
        PleHashConfig config,
        Span<long> rowIndicesOut)
    {
        ArgumentNullException.ThrowIfNull(config);
        int nHeads = config.TotalHeads;
        if (rowIndicesOut.Length < nHeads)
            throw new ArgumentException($"Expected at least {nHeads} output row slots, got {rowIndicesOut.Length}.", nameof(rowIndicesOut));
        if (config.Multipliers.Length < config.NgramSize)
            throw new ArgumentException("Multipliers length is smaller than NgramSize.", nameof(config));
        if (config.HeadVocabSizes.Length < nHeads || config.HeadOffsets.Length < nHeads)
            throw new ArgumentException("Head table metadata is smaller than TotalHeads.", nameof(config));

        Span<int> ctx = config.NgramSize <= 32 ? stackalloc int[config.NgramSize] : new int[config.NgramSize];

        // Current token never cuts its own predecessors, even when token == EosTokenId.
        ctx[0] = (config.ImageTokenId >= 0 && token == config.ImageTokenId) ? 0 : token;
        bool cut = false;
        for (int s = 1; s < config.NgramSize; s++)
        {
            int prevIdx = prevOldestFirst.Length - s;
            int t = prevIdx >= 0 ? prevOldestFirst[prevIdx] : -1;
            if (t < 0 || t == config.EosTokenId)
            {
                cut = true;
            }
            else if (config.ImageTokenId >= 0 && t == config.ImageTokenId)
            {
                t = 0;
            }
            ctx[s] = cut ? config.EosTokenId : t;
        }

        for (int n = 2; n <= config.NgramSize; n++)
        {
            ulong mixed = unchecked((ulong)(long)ctx[0] * config.Multipliers[0]);
            for (int j = 1; j < n; j++)
            {
                mixed ^= unchecked((ulong)(long)ctx[j] * config.Multipliers[j]);
            }

            int baseHead = (n - 2) * config.HeadsPerNgram;
            for (int k = 0; k < config.HeadsPerNgram; k++)
            {
                int h = baseHead + k;
                ulong mod = (ulong)config.HeadVocabSizes[h];
                rowIndicesOut[h] = config.HeadOffsets[h] + (long)(mixed % mod);
            }
        }
    }
}

/// <summary>
/// Row storage formats supported by <see cref="DirectIoNgramTable"/>.
/// </summary>
public enum PleRowFormat
{
    IQ4_NL,
    Fp8E4M3,
    Q8_0,
    Q4_0,
    Q5_0,
    BF16,
    F16,
    F32
}

/// <summary>
/// I/O transport mode for reading PLE rows (supports A/B benchmarking Direct I/O vs OS page-cache mmap,
/// or <see cref="Auto"/> selection based on live physical RAM headroom).
/// </summary>
public enum PleIoMode
{
    DirectIo = 0,
    MemoryMapped = 1,
    Auto = 2
}

/// <summary>
/// Represents an in-flight asynchronous PLE table gather operation overlapped with early model computation.
/// </summary>
public struct PleLookupTicket
{
    internal DirectIoNgramTable.TicketState? State;
    internal int Generation;
    internal int RequestedCount;

    public readonly bool IsValid => State != null;
}

/// <summary>
/// Unbuffered 4 KiB-aligned NVMe Direct-I/O (with bounded row cache and split async Issue/Collect)
/// and memory-mapped A/B tier for Qwen3.8-Flash-Next Per-Layer N-gram Embedding tables.
/// </summary>
public sealed unsafe class DirectIoNgramTable : IDisposable
{
    private const int PageSize = 4096;
    private const int FileFlagNoBuffering = 0x20000000;

    private static readonly float[] Fp8E4M3Lut = BuildFp8E4M3Lut();

    private readonly string _filePath;
    private readonly long _tableByteOffset;
    private readonly long _totalRows;
    private readonly int _headDim;
    private readonly int _rowBytes;
    private readonly PleRowFormat _format;
    private readonly PleIoMode _mode;
    private readonly PleHashConfig _hashConfig;
    private readonly long _fileLength;

    private readonly SafeFileHandle? _directHandle;
    private readonly SafeFileHandle? _fallbackHandle;
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _mmapBasePtr;

    private readonly PleRowCache _rowCache;
    private readonly ConcurrentBag<TicketState> _ticketPool = new();

    private long _cacheHits;
    private long _cacheMisses;
    private long _physicalPagesIssued;
    private int _nextGeneration = 1;
    private bool _disposed;

    public string FilePath => _filePath;
    public long TableByteOffset => _tableByteOffset;
    public long TotalRows => _totalRows;
    public int HeadDim => _headDim;
    public int RowBytes => _rowBytes;
    public PleRowFormat Format => _format;
    public PleIoMode IoMode => _mode;
    public PleHashConfig HashConfig => _hashConfig;
    public bool IsUnbufferedHandleActive => _directHandle != null && !_directHandle.IsInvalid && !_directHandle.IsClosed;

    public long CacheHits => Interlocked.Read(ref _cacheHits);
    public long CacheMisses => Interlocked.Read(ref _cacheMisses);
    public long PhysicalPagesIssued => Interlocked.Read(ref _physicalPagesIssued);

    public DirectIoNgramTable(
        string filePath,
        long tableByteOffset,
        long totalRows,
        int headDim,
        PleRowFormat format,
        PleIoMode mode = PleIoMode.DirectIo,
        int rowCacheCapacity = 4096,
        PleHashConfig? hashConfig = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must be non-empty.", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"PLE table file not found: {filePath}", filePath);
        if (headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDim));
        if (totalRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalRows));

        _filePath = filePath;
        _tableByteOffset = tableByteOffset;
        _totalRows = totalRows;
        _headDim = headDim;
        _format = format;
        _hashConfig = hashConfig ?? PleHashConfig.DefaultQwen4Exp;
        _rowBytes = GetRowBytes(format, headDim);
        _fileLength = new FileInfo(filePath).Length;

        long requiredEnd = _tableByteOffset + _totalRows * _rowBytes;
        if (_tableByteOffset < 0 || requiredEnd > _fileLength)
            throw new InvalidDataException($"PLE table bounds [{_tableByteOffset}..{requiredEnd}] exceed file length {_fileLength}.");

        _mode = mode == PleIoMode.Auto
            ? ResolveIoMode(
                PleIoMode.Auto,
                _totalRows * _rowBytes,
                Hardware.HardwareRuntimeProfiler.CaptureSnapshot().AvailablePhysicalRamBytes)
            : mode;

        _rowCache = new PleRowCache(Math.Max(64, rowCacheCapacity), headDim);

        if (_mode == PleIoMode.DirectIo)
        {
            if (OperatingSystem.IsWindows())
            {
                try
                {
                    _directHandle = File.OpenHandle(
                        filePath,
                        FileMode.Open,
                        FileAccess.Read,
                        FileShare.ReadWrite | FileShare.Delete,
                        FileOptions.Asynchronous | (FileOptions)FileFlagNoBuffering);
                }
                catch
                {
                    _directHandle = null;
                }
            }

            _fallbackHandle = File.OpenHandle(
                filePath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete,
                FileOptions.Asynchronous | FileOptions.RandomAccess);
        }
        else
        {
            _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* basePtr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            _mmapBasePtr = basePtr;
        }
    }

    public static PleIoMode ResolveIoMode(
        PleIoMode requested,
        long totalBytesToMap,
        ulong availablePhysicalRamBytes,
        ulong reserveBytes = 4UL * 1024 * 1024 * 1024)
    {
        if (requested != PleIoMode.Auto)
            return requested;

        ulong bytesNeeded = (ulong)Math.Max(0L, totalBytesToMap);
        return availablePhysicalRamBytes > reserveBytes &&
               bytesNeeded <= availablePhysicalRamBytes - reserveBytes
            ? PleIoMode.MemoryMapped
            : PleIoMode.DirectIo;
    }

    public static PleRowFormat FromGgufType(GgufType type) => type switch
    {
        GgufType.IQ4_NL => PleRowFormat.IQ4_NL,
        GgufType.Q8_0 => PleRowFormat.Q8_0,
        GgufType.Q4_0 => PleRowFormat.Q4_0,
        GgufType.Q5_0 => PleRowFormat.Q5_0,
        GgufType.BF16 => PleRowFormat.BF16,
        GgufType.F16 => PleRowFormat.F16,
        GgufType.F32 => PleRowFormat.F32,
        _ => throw new NotSupportedException($"Unsupported GGUF type for PLE table: {type}.")
    };

    public static int GetRowBytes(PleRowFormat format, int headDim)
    {
        if (format is PleRowFormat.IQ4_NL or PleRowFormat.Q8_0 or PleRowFormat.Q4_0 or PleRowFormat.Q5_0 && headDim % 32 != 0)
            throw new ArgumentException($"Block format {format} requires headDim divisible by 32 (got {headDim}).", nameof(headDim));

        return format switch
        {
            PleRowFormat.IQ4_NL => (headDim / 32) * sizeof(BlockIQ4_NL),
            PleRowFormat.Fp8E4M3 => 2 + headDim,
            PleRowFormat.Q8_0 => (headDim / 32) * sizeof(BlockQ8_0),
            PleRowFormat.Q4_0 => (headDim / 32) * sizeof(BlockQ4_0),
            PleRowFormat.Q5_0 => (headDim / 32) * sizeof(BlockQ5_0),
            PleRowFormat.BF16 => headDim * 2,
            PleRowFormat.F16 => headDim * 2,
            PleRowFormat.F32 => headDim * 4,
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
    }

    /// <summary>
    /// Issues asynchronous PLE row lookups for the given token and oldest-first predecessors.
    /// Returns immediately so SSD I/O overlaps embedding extraction and early layers.
    /// </summary>
    public PleLookupTicket IssueToken(int token, ReadOnlySpan<int> prevOldestFirst)
    {
        int nHeads = _hashConfig.TotalHeads;
        Span<long> rows = nHeads <= 64 ? stackalloc long[nHeads] : new long[nHeads];
        PleNgramHasher.ComputeNgramRows(token, prevOldestFirst, _hashConfig, rows);
        return Issue(rows);
    }

    /// <summary>
    /// Synchronous convenience helper equivalent to <see cref="IssueToken"/> followed by <see cref="Collect"/>.
    /// </summary>
    public void GatherToken(int token, ReadOnlySpan<int> prevOldestFirst, Span<float> destination)
    {
        var ticket = IssueToken(token, prevOldestFirst);
        Collect(ticket, destination);
    }

    /// <summary>
    /// Issues asynchronous 4 KiB-aligned page reads (or mmap prefetches) for a span of PLE row indices.
    /// </summary>
    public PleLookupTicket Issue(ReadOnlySpan<long> rowIndices)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int count = rowIndices.Length;
        if (count == 0)
            return default;

        var state = RentTicketState();
        state.ResetForCount(count, _headDim);
        int gen = Interlocked.Increment(ref _nextGeneration);
        state.ActiveGeneration = gen;

        int hits = 0;
        int misses = 0;
        state.MissCount = 0;
        state.PageCount = 0;

        for (int i = 0; i < count; i++)
        {
            long rowIdx = rowIndices[i];
            if ((ulong)rowIdx >= (ulong)_totalRows)
                throw new ArgumentOutOfRangeException(nameof(rowIndices), $"Row index {rowIdx} out of range [0, {_totalRows}).");

            state.RequestedRows[i] = rowIdx;
            if (_rowCache.TryCopyRow(rowIdx, state.StagingPtr + (long)i * _headDim))
            {
                state.IsHit[i] = true;
                hits++;
            }
            else
            {
                state.IsHit[i] = false;
                state.AddMiss(i, rowIdx);
                misses++;

                long byteStart = _tableByteOffset + rowIdx * _rowBytes;
                long byteEnd = byteStart + _rowBytes;
                long firstPage = byteStart >> 12;
                long lastPage = (byteEnd - 1) >> 12;
                for (long p = firstPage; p <= lastPage; p++)
                {
                    state.AddPage(p);
                }
            }
        }

        if (hits > 0) Interlocked.Add(ref _cacheHits, hits);
        if (misses > 0) Interlocked.Add(ref _cacheMisses, misses);

        if (misses > 0)
        {
            state.DeduplicateAndSortPages();
            Interlocked.Add(ref _physicalPagesIssued, state.PageCount);

            if (_mode == PleIoMode.MemoryMapped)
            {
                PrefetchSortedPages(state.Pages, state.PageCount);
            }
            else
            {
                StartAlignedReads(state);
            }
        }

        return new PleLookupTicket
        {
            State = state,
            Generation = gen,
            RequestedCount = count
        };
    }

    /// <summary>
    /// Waits for an issued lookup ticket, dequantizes any cache-missed rows, populates the row cache,
    /// and writes all <c>RequestedCount * HeadDim</c> floats into <paramref name="destination"/>.
    /// </summary>
    public void Collect(PleLookupTicket ticket, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!ticket.IsValid || ticket.State == null)
            return;

        var state = ticket.State;
        if (state.ActiveGeneration != ticket.Generation || state.ActiveGeneration == 0)
            throw new InvalidOperationException("PleLookupTicket has already been collected or invalidated.");

        int expectedFloats = ticket.RequestedCount * _headDim;
        if (destination.Length < expectedFloats)
            throw new ArgumentException($"Destination span too small ({destination.Length} < {expectedFloats}).", nameof(destination));

        try
        {
            state.WaitForIo();

            if (state.MissCount > 0)
            {
                for (int m = 0; m < state.MissCount; m++)
                {
                    int origIdx = state.MissOrigIndex[m];
                    long rowIdx = state.MissRowIndex[m];
                    long byteStart = _tableByteOffset + rowIdx * _rowBytes;
                    byte* rawPtr;

                    if (_mode == PleIoMode.MemoryMapped)
                    {
                        rawPtr = _mmapBasePtr + byteStart;
                    }
                    else
                    {
                        long firstPage = byteStart >> 12;
                        int pageSlot = Array.BinarySearch(state.Pages, 0, state.PageCount, firstPage);
                        int offInPage = (int)(byteStart & (PageSize - 1));
                        rawPtr = state.BouncePtr + (long)pageSlot * PageSize + offInPage;
                    }

                    float* dstRow = state.StagingPtr + (long)origIdx * _headDim;
                    DequantizeRow(_format, rawPtr, dstRow, _headDim);
                    _rowCache.Insert(rowIdx, dstRow);
                }
            }

            fixed (float* pDst = destination)
            {
                ulong copyBytes = (ulong)expectedFloats * sizeof(float);
                Buffer.MemoryCopy(state.StagingPtr, pDst, copyBytes, copyBytes);
            }
        }
        finally
        {
            state.ClearIoHandles();
            state.ActiveGeneration = 0;
            _ticketPool.Add(state);
        }
    }

    private void PrefetchSortedPages(long[] pages, int count)
    {
        int i = 0;
        while (i < count)
        {
            long startPage = pages[i];
            int run = 1;
            while (i + run < count && pages[i + run] == startPage + run)
            {
                run++;
            }

            long offset = startPage << 12;
            nuint bytes = (nuint)Math.Min((long)run * PageSize, _fileLength - offset);
            VirtualMemoryInterop.PrefetchMemory(_mmapBasePtr + offset, bytes);
            i += run;
        }
    }

    private void StartAlignedReads(TicketState state)
    {
        state.EnsureBounceCapacity(state.PageCount * PageSize);
        long alignedFileLimit = _fileLength & ~(long)(PageSize - 1);

        int idx = 0;
        while (idx < state.PageCount)
        {
            long startPage = state.Pages[idx];
            int runPages = 1;
            while (idx + runPages < state.PageCount && state.Pages[idx + runPages] == startPage + runPages)
            {
                runPages++;
            }

            long extentOffset = startPage << 12;
            int extentBytes = runPages * PageSize;

            if (IsUnbufferedHandleActive && extentOffset + extentBytes <= alignedFileLimit)
            {
                Memory<byte> slice = state.GetBounceSlice(idx * PageSize, extentBytes);
                state.EnqueueRead(RandomAccess.ReadAsync(_directHandle!, slice, extentOffset));
            }
            else if (IsUnbufferedHandleActive && runPages > 1 && extentOffset + (runPages - 1) * PageSize <= alignedFileLimit)
            {
                int alignedBytes = (runPages - 1) * PageSize;
                Memory<byte> alignedSlice = state.GetBounceSlice(idx * PageSize, alignedBytes);
                state.EnqueueRead(RandomAccess.ReadAsync(_directHandle!, alignedSlice, extentOffset));

                int tailOffsetSlot = (idx + runPages - 1) * PageSize;
                long tailFileOffset = extentOffset + alignedBytes;
                int tailBytes = (int)Math.Min(PageSize, _fileLength - tailFileOffset);
                Memory<byte> tailSlice = state.GetBounceSlice(tailOffsetSlot, tailBytes);
                state.EnqueueRead(RandomAccess.ReadAsync(_fallbackHandle!, tailSlice, tailFileOffset));
            }
            else
            {
                int clampedBytes = (int)Math.Min((long)extentBytes, _fileLength - extentOffset);
                Memory<byte> slice = state.GetBounceSlice(idx * PageSize, clampedBytes);
                state.EnqueueRead(RandomAccess.ReadAsync(_fallbackHandle!, slice, extentOffset));
            }

            idx += runPages;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeRow(PleRowFormat format, byte* src, float* dst, int headDim)
    {
        switch (format)
        {
            case PleRowFormat.IQ4_NL:
                QuantKernels.DequantizeIQ4_NL((BlockIQ4_NL*)src, dst, headDim);
                break;
            case PleRowFormat.Fp8E4M3:
            {
                float scale = (float)*(Half*)src;
                byte* q = src + 2;
                fixed (float* lut = Fp8E4M3Lut)
                {
                    for (int i = 0; i < headDim; i++)
                    {
                        dst[i] = scale * lut[q[i]];
                    }
                }
                break;
            }
            case PleRowFormat.Q8_0:
            {
                int nb = headDim / 32;
                BlockQ8_0* blocks = (BlockQ8_0*)src;
                for (int b = 0; b < nb; b++)
                {
                    float d = (float)blocks[b].Delta;
                    sbyte* qs = blocks[b].Qs;
                    for (int l = 0; l < 32; l++)
                    {
                        dst[b * 32 + l] = d * qs[l];
                    }
                }
                break;
            }
            case PleRowFormat.Q4_0:
            {
                int nb = headDim / 32;
                BlockQ4_0* blocks = (BlockQ4_0*)src;
                for (int b = 0; b < nb; b++)
                {
                    float d = (float)blocks[b].Delta;
                    byte* qs = blocks[b].Qs;
                    for (int l = 0; l < 16; l++)
                    {
                        byte v = qs[l];
                        dst[b * 32 + l] = ((v & 0x0F) - 8) * d;
                        dst[b * 32 + l + 16] = ((v >> 4) - 8) * d;
                    }
                }
                break;
            }
            case PleRowFormat.Q5_0:
                QuantKernels.DequantizeQ5_0((BlockQ5_0*)src, dst, headDim);
                break;
            case PleRowFormat.BF16:
                QuantKernels.DequantizeBF16((ushort*)src, dst, headDim);
                break;
            case PleRowFormat.F16:
            {
                Half* h = (Half*)src;
                for (int i = 0; i < headDim; i++)
                {
                    dst[i] = (float)h[i];
                }
                break;
            }
            case PleRowFormat.F32:
                Buffer.MemoryCopy(src, dst, (ulong)headDim * 4, (ulong)headDim * 4);
                break;
            default:
                throw new NotSupportedException($"Unsupported PLE row format: {format}.");
        }
    }

    /// <summary>
    /// Quantizes a float row into the compact 2 + headDim byte FP8 E4M3 format (Half row scale + uint8 E4M3 codes).
    /// </summary>
    public static void QuantizeFp8E4M3Row(ReadOnlySpan<float> src, Span<byte> dst)
    {
        int headDim = src.Length;
        if (dst.Length < 2 + headDim)
            throw new ArgumentException($"Destination too small (expected {2 + headDim} bytes).", nameof(dst));

        float maxAbs = 0f;
        for (int i = 0; i < headDim; i++)
        {
            float a = MathF.Abs(src[i]);
            if (a > maxAbs) maxAbs = a;
        }

        float scaleVal = maxAbs > 0f ? Math.Max(maxAbs / 448.0f, 1e-5f) : 1.0f;
        Half rowScale = (Half)scaleVal;
        float actualScale = (float)rowScale;
        MemoryMarshal.Write(dst, in rowScale);

        for (int i = 0; i < headDim; i++)
        {
            float scaled = actualScale > 0f ? src[i] / actualScale : 0f;
            int bestCode = 0;
            float bestErr = float.MaxValue;
            for (int c = 0; c < 256; c++)
            {
                if ((c & 0x7F) == 0x7F) continue; // skip E4M3 NaN encodings
                float err = MathF.Abs(Fp8E4M3Lut[c] - scaled);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestCode = c;
                }
            }
            dst[2 + i] = (byte)bestCode;
        }
    }

    private static void QuantizeQ8_0Row(float* src, BlockQ8_0* dst, int headDim)
    {
        int blockCount = (headDim + 31) / 32;
        for (int b = 0; b < blockCount; b++)
        {
            int offset = b * 32;
            float maxAbs = 0.0f;
            for (int j = 0; j < 32; j++)
            {
                int idx = offset + j;
                if (idx >= headDim) break;
                float a = MathF.Abs(src[idx]);
                if (a > maxAbs) maxAbs = a;
            }

            Half d = (Half)(maxAbs / 127.0f);
            float inv = (float)d != 0.0f ? 1.0f / (float)d : 0.0f;
            BlockQ8_0* block = dst + b;
            block->Delta = d;
            for (int j = 0; j < 32; j++)
            {
                int idx = offset + j;
                float val = idx < headDim ? src[idx] : 0.0f;
                block->Qs[j] = (sbyte)Math.Clamp((int)MathF.Round(val * inv), -127, 127);
            }
        }
    }

    /// <summary>
    /// Helper that writes a synthetic row-major quantized PLE table file on disk padded to 4096-byte page boundaries
    /// for benchmarking and deterministic Direct-I/O verification tests.
    /// </summary>
    public static void CreateSyntheticFile(
        string path,
        long rowCount,
        int headDim,
        PleRowFormat format,
        Func<long, int, float> valueGenerator,
        int headerPaddingBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(valueGenerator);
        int rowBytes = GetRowBytes(format, headDim);
        long totalBytes = headerPaddingBytes + rowCount * rowBytes;
        long alignedBytes = (totalBytes + PageSize - 1) & ~(long)(PageSize - 1);

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        if (headerPaddingBytes > 0)
        {
            fs.Write(new byte[headerPaddingBytes]);
        }

        float[] floats = new float[headDim];
        byte[] encoded = new byte[rowBytes];

        fixed (float* pFloats = floats)
        fixed (byte* pEnc = encoded)
        {
            for (long r = 0; r < rowCount; r++)
            {
                for (int c = 0; c < headDim; c++)
                {
                    floats[c] = valueGenerator(r, c);
                }

                switch (format)
                {
                    case PleRowFormat.IQ4_NL:
                        QuantKernels.QuantizeIQ4_NL(pFloats, (BlockIQ4_NL*)pEnc, headDim);
                        break;
                    case PleRowFormat.Fp8E4M3:
                        QuantizeFp8E4M3Row(floats, encoded);
                        break;
                    case PleRowFormat.Q8_0:
                        QuantizeQ8_0Row(pFloats, (BlockQ8_0*)pEnc, headDim);
                        break;
                    case PleRowFormat.BF16:
                        for (int c = 0; c < headDim; c++)
                        {
                            uint bits = BitConverter.SingleToUInt32Bits(floats[c]);
                            ((ushort*)pEnc)[c] = (ushort)(bits >> 16);
                        }
                        break;
                    case PleRowFormat.F16:
                        for (int c = 0; c < headDim; c++)
                        {
                            ((Half*)pEnc)[c] = (Half)floats[c];
                        }
                        break;
                    case PleRowFormat.F32:
                        Buffer.MemoryCopy(pFloats, pEnc, (ulong)headDim * 4, (ulong)headDim * 4);
                        break;
                    default:
                        throw new NotSupportedException($"CreateSyntheticFile does not support {format}.");
                }

                fs.Write(encoded);
            }
        }

        if (alignedBytes > totalBytes)
        {
            fs.Write(new byte[alignedBytes - totalBytes]);
        }
    }

    private TicketState RentTicketState()
    {
        return _ticketPool.TryTake(out var existing) ? existing : new TicketState();
    }

    private static float[] BuildFp8E4M3Lut()
    {
        var lut = new float[256];
        for (int b = 0; b < 256; b++)
        {
            float sign = (b & 0x80) != 0 ? -1.0f : 1.0f;
            int exp = (b >> 3) & 0x0F;
            int mant = b & 0x07;
            if (exp == 15 && mant == 7)
            {
                lut[b] = 0f;
            }
            else if (exp == 0)
            {
                lut[b] = sign * (mant / 8.0f) * (1.0f / 64.0f);
            }
            else
            {
                lut[b] = sign * (1.0f + mant / 8.0f) * MathF.ScaleB(1.0f, exp - 7);
            }
        }
        return lut;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _directHandle?.Dispose();
        _fallbackHandle?.Dispose();

        if (_mmapBasePtr != null)
        {
            try { _accessor?.SafeMemoryMappedViewHandle.ReleasePointer(); }
            catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
            {
                System.Diagnostics.Trace.TraceWarning($"[DirectIoNgramTable] ReleasePointer failed: {ex.Message}");
            }
            _mmapBasePtr = null;
        }
        _accessor?.Dispose();
        _mmf?.Dispose();

        while (_ticketPool.TryTake(out var state))
        {
            state.Dispose();
        }
        _rowCache.Dispose();
    }

    internal sealed unsafe class TicketState : IDisposable
    {
        public int ActiveGeneration;
        public long[] RequestedRows = Array.Empty<long>();
        public bool[] IsHit = Array.Empty<bool>();
        public int[] MissOrigIndex = Array.Empty<int>();
        public long[] MissRowIndex = Array.Empty<long>();
        public int MissCount;

        public long[] Pages = Array.Empty<long>();
        public int PageCount;

        private float* _stagingPtr;
        private int _stagingFloatCapacity;
        private AlignedPageMemoryManager? _bounceManager;

        private bool _hasSingleRead;
        private ValueTask<int> _singleRead;
        private List<Task>? _multiReads;

        public float* StagingPtr => _stagingPtr;
        public byte* BouncePtr => _bounceManager != null ? _bounceManager.Pointer : null;

        public void ResetForCount(int count, int headDim)
        {
            if (RequestedRows.Length < count)
            {
                RequestedRows = new long[count];
                IsHit = new bool[count];
                MissOrigIndex = new int[count];
                MissRowIndex = new long[count];
            }

            int neededFloats = count * headDim;
            if (_stagingFloatCapacity < neededFloats)
            {
                if (_stagingPtr != null) NativeMemory.Free(_stagingPtr);
                _stagingPtr = (float*)NativeMemory.Alloc((nuint)neededFloats * sizeof(float));
                _stagingFloatCapacity = neededFloats;
            }

            ClearIoHandles();
        }

        public void AddMiss(int origIdx, long rowIdx)
        {
            MissOrigIndex[MissCount] = origIdx;
            MissRowIndex[MissCount] = rowIdx;
            MissCount++;
        }

        public void AddPage(long page)
        {
            if (PageCount == Pages.Length)
            {
                int newLen = Math.Max(32, Pages.Length * 2);
                Array.Resize(ref Pages, newLen);
            }
            Pages[PageCount++] = page;
        }

        public void DeduplicateAndSortPages()
        {
            if (PageCount <= 1) return;
            Array.Sort(Pages, 0, PageCount);
            int unique = 1;
            for (int i = 1; i < PageCount; i++)
            {
                if (Pages[i] != Pages[unique - 1])
                {
                    Pages[unique++] = Pages[i];
                }
            }
            PageCount = unique;
        }

        public void EnsureBounceCapacity(int bytes)
        {
            if (_bounceManager == null || _bounceManager.ByteCapacity < bytes)
            {
                ((IDisposable?)_bounceManager)?.Dispose();
                _bounceManager = new AlignedPageMemoryManager(bytes);
            }
        }

        public Memory<byte> GetBounceSlice(int byteOffset, int length)
        {
            return _bounceManager!.Memory.Slice(byteOffset, length);
        }

        public void EnqueueRead(ValueTask<int> readTask)
        {
            if (!_hasSingleRead && _multiReads == null)
            {
                _singleRead = readTask;
                _hasSingleRead = true;
            }
            else
            {
                if (_multiReads == null)
                {
                    _multiReads = new List<Task>(4);
                }
                if (_hasSingleRead)
                {
                    _multiReads.Add(_singleRead.AsTask());
                    _singleRead = default;
                    _hasSingleRead = false;
                }
                _multiReads.Add(readTask.AsTask());
            }
        }

        public void WaitForIo()
        {
            if (_hasSingleRead)
            {
                _singleRead.AsTask().GetAwaiter().GetResult();
                _singleRead = default;
                _hasSingleRead = false;
            }
            else if (_multiReads != null && _multiReads.Count > 0)
            {
                Task.WhenAll(_multiReads).GetAwaiter().GetResult();
            }
        }

        public void ClearIoHandles()
        {
            _hasSingleRead = false;
            _singleRead = default;
            _multiReads?.Clear();
        }

        public void Dispose()
        {
            ClearIoHandles();
            if (_stagingPtr != null)
            {
                NativeMemory.Free(_stagingPtr);
                _stagingPtr = null;
                _stagingFloatCapacity = 0;
            }
            ((IDisposable?)_bounceManager)?.Dispose();
            _bounceManager = null;
        }
    }

    private sealed unsafe class AlignedPageMemoryManager : MemoryManager<byte>
    {
        private byte* _ptr;
        private readonly int _length;

        public AlignedPageMemoryManager(int byteCapacity)
        {
            _length = (byteCapacity + PageSize - 1) & ~(PageSize - 1);
            _ptr = (byte*)NativeMemory.AlignedAlloc((nuint)_length, PageSize);
        }

        public byte* Pointer => _ptr;
        public int ByteCapacity => _length;

        public override Span<byte> GetSpan()
        {
            ObjectDisposedException.ThrowIf(_ptr == null, this);
            return new Span<byte>(_ptr, _length);
        }

        public override MemoryHandle Pin(int elementIndex = 0)
        {
            ObjectDisposedException.ThrowIf(_ptr == null, this);
            return new MemoryHandle(_ptr + elementIndex);
        }

        public override void Unpin()
        {
        }

        protected override void Dispose(bool disposing)
        {
            if (_ptr != null)
            {
                NativeMemory.AlignedFree(_ptr);
                _ptr = null;
            }
        }
    }

    private sealed unsafe class PleRowCache : IDisposable
    {
        private readonly Lock _lock = new();
        private readonly int _capacity;
        private readonly int _headDim;
        private readonly float* _slab;
        private readonly long[] _slotRows;
        private readonly bool[] _referenced;
        private readonly Dictionary<long, int> _index;
        private int _clockHand;
        private int _count;
        private bool _disposed;

        public PleRowCache(int capacity, int headDim)
        {
            _capacity = capacity;
            _headDim = headDim;
            _slab = (float*)NativeMemory.AlignedAlloc((nuint)capacity * (nuint)headDim * sizeof(float), 64);
            _slotRows = new long[capacity];
            Array.Fill(_slotRows, -1L);
            _referenced = new bool[capacity];
            _index = new Dictionary<long, int>(capacity);
        }

        public bool TryCopyRow(long rowIdx, float* dst)
        {
            lock (_lock)
            {
                if (_disposed || !_index.TryGetValue(rowIdx, out int slot))
                    return false;

                _referenced[slot] = true;
                ulong bytes = (ulong)_headDim * sizeof(float);
                Buffer.MemoryCopy(_slab + (long)slot * _headDim, dst, bytes, bytes);
                return true;
            }
        }

        public void Insert(long rowIdx, float* src)
        {
            lock (_lock)
            {
                if (_disposed) return;

                if (_index.TryGetValue(rowIdx, out int existingSlot))
                {
                    _referenced[existingSlot] = true;
                    return;
                }

                int slot;
                if (_count < _capacity)
                {
                    slot = _count++;
                }
                else
                {
                    while (_referenced[_clockHand])
                    {
                        _referenced[_clockHand] = false;
                        _clockHand = (_clockHand + 1) % _capacity;
                    }

                    slot = _clockHand;
                    _clockHand = (_clockHand + 1) % _capacity;

                    long evicted = _slotRows[slot];
                    if (evicted >= 0)
                    {
                        _index.Remove(evicted);
                    }
                }

                _slotRows[slot] = rowIdx;
                _referenced[slot] = true;
                _index[rowIdx] = slot;

                ulong bytes = (ulong)_headDim * sizeof(float);
                Buffer.MemoryCopy(src, _slab + (long)slot * _headDim, bytes, bytes);
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _index.Clear();
                if (_slab != null)
                {
                    NativeMemory.AlignedFree(_slab);
                }
            }
        }
    }
}
