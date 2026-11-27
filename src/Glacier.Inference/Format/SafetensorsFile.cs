namespace Glacier.Inference.Format;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text.Json;

/// <summary>
/// Tensor metadata within a Safetensors container.
/// </summary>
public sealed record SafetensorsTensorInfo(
    string Name,
    string Dtype,
    long[] Shape,
    ulong StartOffset,
    ulong EndOffset)
{
    public ulong ByteSize => EndOffset - StartOffset;
    public long ElementCount
    {
        get
        {
            if (Shape.Length == 0) return 0;
            long count = 1;
            foreach (var d in Shape) count *= d;
            return count;
        }
    }
}

/// <summary>
/// High-performance zero-copy memory-mapped Hugging Face Safetensors reader.
/// Maps multi-gigabyte models directly into virtual address space with zero memory copying.
/// </summary>
public sealed unsafe class SafetensorsFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _basePointer;
    private readonly long _fileLength;
    private bool _disposed;

    public string FilePath { get; }
    public ulong HeaderSize { get; }
    public ulong DataOffset => 8 + HeaderSize;
    public byte* BasePointer => _basePointer;

    public Dictionary<string, SafetensorsTensorInfo> Tensors { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);

    private SafetensorsFile(string filePath, MemoryMappedFile mmf, MemoryMappedViewAccessor accessor, byte* basePointer, ulong headerSize, long fileLength)
    {
        FilePath = filePath;
        _mmf = mmf;
        _accessor = accessor;
        _basePointer = basePointer;
        HeaderSize = headerSize;
        _fileLength = fileLength;
    }

    public static SafetensorsFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Safetensors file not found: {filePath}", filePath);

        var fileInfo = new FileInfo(filePath);
        if (fileInfo.Length < 8)
            throw new InvalidDataException($"Safetensors file is truncated (length {fileInfo.Length} < 8 bytes).");

        var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        byte* ptr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);

        const ulong MaxHeaderSize = 256UL * 1024 * 1024;
        ulong headerSize = *(ulong*)ptr;
        if (headerSize > MaxHeaderSize || headerSize > (ulong)(fileInfo.Length - 8))
        {
            accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            accessor.Dispose();
            mmf.Dispose();
            throw new InvalidDataException($"Invalid Safetensors header size: {headerSize} bytes (file length {fileInfo.Length}).");
        }

        var file = new SafetensorsFile(filePath, mmf, accessor, ptr, headerSize, fileInfo.Length);
        try
        {
            file.ParseHeader();
            return file;
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    private void ParseHeader()
    {
        // Read header JSON string
        byte[] headerBytes = new byte[HeaderSize];
        System.Runtime.InteropServices.Marshal.Copy((IntPtr)(_basePointer + 8), headerBytes, 0, (int)HeaderSize);

        using var doc = JsonDocument.Parse(headerBytes);
        var root = doc.RootElement;
        ulong dataStart = DataOffset;
        ulong totalFileBytes = (ulong)_fileLength;

        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
            {
                foreach (var meta in prop.Value.EnumerateObject())
                {
                    Metadata[meta.Name] = meta.Value.GetString() ?? "";
                }
                continue;
            }

            var elem = prop.Value;
            string dtype = elem.GetProperty("dtype").GetString() ?? "F32";

            var shapeArr = elem.GetProperty("shape");
            long[] shape = new long[shapeArr.GetArrayLength()];
            int sIdx = 0;
            foreach (var s in shapeArr.EnumerateArray())
            {
                long dim = s.GetInt64();
                if (dim < 0)
                    throw new InvalidDataException($"Invalid negative dimension {dim} in Safetensors tensor '{prop.Name}'.");
                shape[sIdx++] = dim;
            }

            var offsetsArr = elem.GetProperty("data_offsets");
            if (offsetsArr.GetArrayLength() != 2)
                throw new InvalidDataException($"Invalid data_offsets array length in Safetensors tensor '{prop.Name}'.");

            ulong start = offsetsArr[0].GetUInt64();
            ulong end = offsetsArr[1].GetUInt64();
            if (end < start || end > totalFileBytes - dataStart)
            {
                throw new InvalidDataException($"Safetensors tensor '{prop.Name}' data_offsets [{start}, {end}] extend beyond file bounds.");
            }

            Tensors[prop.Name] = new SafetensorsTensorInfo(prop.Name, dtype, shape, start, end);
        }
    }

    /// <summary>
    /// Gets raw byte pointer to tensor data.
    /// </summary>
    public byte* GetTensorPointer(string tensorName)
    {
        if (!Tensors.TryGetValue(tensorName, out var info))
            throw new KeyNotFoundException($"Tensor '{tensorName}' not found in Safetensors file.");

        return _basePointer + DataOffset + info.StartOffset;
    }

    /// <summary>
    /// Checks if a tensor exists in the file.
    /// </summary>
    public bool ContainsTensor(string tensorName) => Tensors.ContainsKey(tensorName);

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_basePointer != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _basePointer = null;
            }
            _accessor.Dispose();
            _mmf.Dispose();
            _disposed = true;
        }
    }
}
