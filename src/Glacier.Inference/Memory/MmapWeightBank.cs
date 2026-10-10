namespace Glacier.Inference.Memory;

using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Contiguous memory-mapped expert weight bank adhering to strict 4KB/2MB OS page boundary alignments.
/// Enables zero-copy demand paging from NVMe storage directly into unmanaged memory space.
/// </summary>
public sealed unsafe class MmapWeightBank : IDisposable
{
    public const long PageSize4KB = 4096;
    public const long PageSize2MB = 2 * 1024 * 1024;
    public const uint GmoeMagic = 0x474D4F45; // "GMOE" Little-Endian
    public const uint CurrentVersion = 1;

    [StructLayout(LayoutKind.Explicit, Size = 64)]
    public struct GmoeHeader
    {
        [FieldOffset(0)] public uint Magic;
        [FieldOffset(4)] public uint Version;
        [FieldOffset(8)] public int NumLayers;
        [FieldOffset(12)] public int NumExperts;
        [FieldOffset(16)] public long ExpertRawBytes;
        [FieldOffset(24)] public long PageAlignment;
        [FieldOffset(32)] public long ExpertStrideBytes;
        [FieldOffset(40)] public int QuantType;
        [FieldOffset(44)] public int HeaderSize;
        [FieldOffset(48)] public long CreationTimestampUnixMs;
        [FieldOffset(56)] public ulong Checksum;
    }

    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _basePointer;
    private readonly long _fileLength;
    private readonly int _numLayers;
    private readonly int _numExperts;
    private readonly long _expertRawBytes;
    private readonly long _expertStrideBytes;
    private readonly long _pageAlignment;
    private readonly long _dataOffset;
    private bool _disposed;

    public int NumLayers => _numLayers;
    public int NumExperts => _numExperts;
    public long ExpertRawBytes => _expertRawBytes;
    public long ExpertStrideBytes => _expertStrideBytes;
    public long PageAlignment => _pageAlignment;
    public long DataOffset => _dataOffset;
    public long FileLength => _fileLength;
    public byte* BasePointer => _basePointer;

    /// <summary>
    /// Computes the page-aligned stride for an expert block.
    /// </summary>
    public static long ComputeAlignedStride(long rawBytes, long pageAlignment = PageSize4KB)
    {
        if (rawBytes <= 0) throw new ArgumentOutOfRangeException(nameof(rawBytes));
        if (pageAlignment <= 0 || (pageAlignment & (pageAlignment - 1)) != 0)
            throw new ArgumentException("Page alignment must be a power of two.", nameof(pageAlignment));

        long rem = rawBytes % pageAlignment;
        return rem == 0 ? rawBytes : rawBytes + (pageAlignment - rem);
    }

    /// <summary>
    /// Opens an existing memory-mapped weight bank from disk.
    /// If file begins with GMOE header, metadata is parsed automatically; otherwise explicit parameters are applied.
    /// </summary>
    public MmapWeightBank(string filePath, int numLayers = 0, int numExperts = 0, long expertByteSize = 0, long pageAlignment = PageSize4KB)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Expert weight bank file not found.", filePath);

        var fi = new FileInfo(filePath);
        _fileLength = fi.Length;
        if (_fileLength < sizeof(GmoeHeader))
            throw new InvalidDataException("File is too small to contain a valid weight bank.");

        _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);

        // Check if file has GMOE header
        GmoeHeader* header = (GmoeHeader*)_basePointer;
        if (header->Magic == GmoeMagic)
        {
            _numLayers = header->NumLayers;
            _numExperts = header->NumExperts;
            _expertRawBytes = header->ExpertRawBytes;
            _pageAlignment = header->PageAlignment > 0 ? header->PageAlignment : PageSize4KB;
            _expertStrideBytes = header->ExpertStrideBytes;
            _dataOffset = header->HeaderSize > 0 ? header->HeaderSize : sizeof(GmoeHeader);
            // Align data offset to page boundary
            long rem = _dataOffset % _pageAlignment;
            if (rem != 0) _dataOffset += (_pageAlignment - rem);
        }
        else
        {
            if (numLayers <= 0 || numExperts <= 0 || expertByteSize <= 0)
                throw new ArgumentException("Non-GMOE file requires explicit numLayers, numExperts, and expertByteSize.");

            _numLayers = numLayers;
            _numExperts = numExperts;
            _expertRawBytes = expertByteSize;
            _pageAlignment = pageAlignment;
            _expertStrideBytes = ComputeAlignedStride(expertByteSize, pageAlignment);
            _dataOffset = 0;
        }
    }

    /// <summary>
    /// Creates and initializes a new contiguous .gmoe file on disk with page alignment.
    /// </summary>
    public static void CreateFile(
        string filePath,
        int numLayers,
        int numExperts,
        long expertRawBytes,
        int quantType = 0,
        long pageAlignment = PageSize4KB,
        Action<int, int, Span<byte>>? initializer = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (expertRawBytes <= 0) throw new ArgumentOutOfRangeException(nameof(expertRawBytes));

        long stride = ComputeAlignedStride(expertRawBytes, pageAlignment);

        // Aligned header offset
        long dataOffset = sizeof(GmoeHeader);
        long rem = dataOffset % pageAlignment;
        if (rem != 0) dataOffset += (pageAlignment - rem);

        long totalSize = dataOffset + ((long)numLayers * numExperts * stride);

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        using (var fs = new FileStream(filePath, FileMode.Create, FileAccess.ReadWrite, FileShare.None))
        {
            fs.SetLength(totalSize);

            // Write header
            GmoeHeader header = new()
            {
                Magic = GmoeMagic,
                Version = CurrentVersion,
                NumLayers = numLayers,
                NumExperts = numExperts,
                ExpertRawBytes = expertRawBytes,
                PageAlignment = pageAlignment,
                ExpertStrideBytes = stride,
                QuantType = quantType,
                HeaderSize = (int)dataOffset,
                CreationTimestampUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                Checksum = 0
            };

            Span<byte> headerBytes = new byte[sizeof(GmoeHeader)];
            MemoryMarshal.Write(headerBytes, in header);
            fs.Write(headerBytes);
        }

        // Populate expert data if initializer provided
        if (initializer != null)
        {
            using var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.ReadWrite);
            using var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.ReadWrite);
            byte* basePtr = null;
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
            try
            {
                byte* dataBase = basePtr + dataOffset;
                for (int l = 0; l < numLayers; l++)
                {
                    for (int e = 0; e < numExperts; e++)
                    {
                        long offset = ((long)l * numExperts + e) * stride;
                        byte* expertPtr = dataBase + offset;
                        Span<byte> span = new(expertPtr, (int)expertRawBytes);
                        initializer(l, e, span);
                    }
                }
            }
            finally
            {
                if (basePtr != null)
                    accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }
        }
    }

    /// <summary>
    /// Gets raw unmanaged pointer to the requested expert's weights.
    /// Offset calculation is guaranteed to be aligned to OS page boundary.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public byte* GetExpertPointer(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layerIdx >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIdx), layerIdx, $"Layer index must be 0..{_numLayers - 1}.");
        if ((uint)expertIdx >= (uint)_numExperts)
            throw new ArgumentOutOfRangeException(nameof(expertIdx), expertIdx, $"Expert index must be 0..{_numExperts - 1}.");

        long offset = _dataOffset + (((long)layerIdx * _numExperts + expertIdx) * _expertStrideBytes);
        if (offset + _expertRawBytes > _fileLength)
            throw new InvalidOperationException("Expert offset exceeds mapped file length.");

        return _basePointer + offset;
    }

    /// <summary>
    /// Gets a ReadOnlySpan over the expert's raw weight bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<byte> GetExpertSpan(int layerIdx, int expertIdx)
    {
        byte* ptr = GetExpertPointer(layerIdx, expertIdx);
        return new ReadOnlySpan<byte>(ptr, (int)_expertRawBytes);
    }

    /// <summary>
    /// Advises OS kernel to prefetch physical pages for speculative expert execution into RAM.
    /// </summary>
    public void PrefetchExpertPages(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        byte* ptr = GetExpertPointer(layerIdx, expertIdx);
        VirtualMemoryInterop.PrefetchMemory(ptr, (nuint)_expertStrideBytes);
    }

    /// <summary>
    /// Proactively prefetches a batch of speculative experts across OS page cache.
    /// </summary>
    public void PrefetchRange(int layerIdx, ReadOnlySpan<int> expertIndices)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int i = 0; i < expertIndices.Length; i++)
        {
            PrefetchExpertPages(layerIdx, expertIndices[i]);
        }
    }

    /// <summary>
    /// Locks the physical memory pages backing the specified expert in host RAM.
    /// </summary>
    public bool LockExpertPages(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        byte* ptr = GetExpertPointer(layerIdx, expertIdx);
        return VirtualMemoryInterop.LockMemory(ptr, (nuint)_expertStrideBytes);
    }

    /// <summary>
    /// Unlocks the physical memory pages backing the specified expert.
    /// </summary>
    public bool UnlockExpertPages(int layerIdx, int expertIdx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        byte* ptr = GetExpertPointer(layerIdx, expertIdx);
        return VirtualMemoryInterop.UnlockMemory(ptr, (nuint)_expertStrideBytes);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_basePointer != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _basePointer = null;
            }
            _accessor.Dispose();
            _mmf.Dispose();
        }
    }
}
