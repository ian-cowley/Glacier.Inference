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
            checked
            {
                foreach (var d in Shape) count *= d;
            }
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
    private const int MaxTensorRank = 16;

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

    private static long ResolveFileLength(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        long length = fileInfo.Length;
        if (length == 0 || fileInfo.LinkTarget != null)
        {
            try
            {
                using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                length = fs.Length;
            }
            catch
            {
                if (fileInfo.LinkTarget != null && File.Exists(fileInfo.LinkTarget))
                {
                    length = new FileInfo(fileInfo.LinkTarget).Length;
                }
            }
        }
        return length;
    }

    public static SafetensorsFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Safetensors file not found: {filePath}", filePath);

        long fileLength = ResolveFileLength(filePath);
        if (fileLength < 8)
            throw new InvalidDataException($"Safetensors file is truncated (length {fileLength} < 8 bytes).");

        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        byte* ptr = null;

        var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        try
        {
            mmf = MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        }
        catch
        {
            stream.Dispose();
            throw;
        }

        try
        {
            accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);

            const ulong MaxHeaderSize = 256UL * 1024 * 1024;
            ulong headerSize = *(ulong*)ptr;
            if (headerSize > MaxHeaderSize || headerSize > (ulong)(fileLength - 8))
            {
                throw new InvalidDataException($"Invalid Safetensors header size: {headerSize} bytes (file length {fileLength}).");
            }

            var file = new SafetensorsFile(filePath, mmf, accessor, ptr, headerSize, fileLength);
            mmf = null;
            accessor = null;
            ptr = null;
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
        catch
        {
            if (ptr != null && accessor != null)
            {
                try { accessor.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
            }
            accessor?.Dispose();
            mmf?.Dispose();
            throw;
        }
    }

    private void ParseHeader()
    {
        var headerSpan = new ReadOnlySpan<byte>(_basePointer + 8, checked((int)HeaderSize));
        var reader = new Utf8JsonReader(headerSpan);

        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            throw new InvalidDataException("Invalid Safetensors header JSON: expected root object.");

        ulong dataStart = DataOffset;
        ulong totalFileBytes = (ulong)_fileLength;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
                break;

            if (reader.TokenType != JsonTokenType.PropertyName)
                throw new InvalidDataException("Invalid Safetensors header JSON: expected property name.");

            string propName = reader.GetString() ?? "";
            if (!reader.Read())
                throw new InvalidDataException($"Unexpected end of Safetensors header JSON at '{propName}'.");

            if (propName == "__metadata__")
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    reader.Skip();
                    continue;
                }
                while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
                {
                    if (reader.TokenType != JsonTokenType.PropertyName)
                        throw new InvalidDataException("Invalid Safetensors __metadata__ entry.");
                    string metaKey = reader.GetString() ?? "";
                    if (!reader.Read())
                        throw new InvalidDataException($"Unexpected end of Safetensors __metadata__ at '{metaKey}'.");
                    Metadata[metaKey] = reader.TokenType == JsonTokenType.String ? (reader.GetString() ?? "") : "";
                    if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                        reader.Skip();
                }
                continue;
            }

            if (reader.TokenType != JsonTokenType.StartObject)
                throw new InvalidDataException($"Invalid Safetensors tensor entry '{propName}': expected object.");

            string? dtype = null;
            var shapeList = new List<long>(4);
            bool hasShape = false;
            ulong start = 0;
            ulong end = 0;
            int offsetCount = 0;
            bool hasOffsets = false;

            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new InvalidDataException($"Invalid Safetensors tensor entry '{propName}'.");

                string field = reader.GetString() ?? "";
                if (!reader.Read())
                    throw new InvalidDataException($"Unexpected end of Safetensors tensor '{propName}' at '{field}'.");

                switch (field)
                {
                    case "dtype":
                        dtype = reader.GetString() ?? "F32";
                        break;
                    case "shape":
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new InvalidDataException($"Invalid shape array in Safetensors tensor '{propName}'.");
                        hasShape = true;
                        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                        {
                            if (shapeList.Count >= MaxTensorRank)
                                throw new InvalidDataException($"Safetensors tensor '{propName}' exceeds maximum rank ({MaxTensorRank}).");
                            long dim = reader.GetInt64();
                            if (dim < 0)
                                throw new InvalidDataException($"Invalid negative dimension {dim} in Safetensors tensor '{propName}'.");
                            shapeList.Add(dim);
                        }
                        break;
                    case "data_offsets":
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new InvalidDataException($"Invalid data_offsets array in Safetensors tensor '{propName}'.");
                        hasOffsets = true;
                        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                        {
                            ulong val = reader.GetUInt64();
                            if (offsetCount == 0) start = val;
                            else if (offsetCount == 1) end = val;
                            offsetCount++;
                        }
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }

            if (dtype == null || !hasShape || !hasOffsets || offsetCount != 2)
                throw new InvalidDataException($"Safetensors tensor '{propName}' is missing required metadata or has invalid data_offsets.");

            if (end < start || end > totalFileBytes - dataStart)
            {
                throw new InvalidDataException($"Safetensors tensor '{propName}' data_offsets [{start}, {end}] extend beyond file bounds.");
            }

            long[] shape = shapeList.ToArray();
            ulong elemCount = 1UL;
            try
            {
                checked
                {
                    for (int i = 0; i < shape.Length; i++)
                    {
                        elemCount *= (ulong)shape[i];
                    }
                }
            }
            catch (OverflowException ex)
            {
                throw new InvalidDataException($"Safetensors tensor '{propName}' shape dimensions overflow element count.", ex);
            }

            if (TryGetDtypeByteSize(dtype, out ulong dtypeBytes))
            {
                ulong expectedBytes;
                try
                {
                    expectedBytes = checked(elemCount * dtypeBytes);
                }
                catch (OverflowException ex)
                {
                    throw new InvalidDataException($"Safetensors tensor '{propName}' byte size overflows.", ex);
                }

                ulong spanBytes = end - start;
                if (spanBytes != expectedBytes)
                {
                    throw new InvalidDataException(
                        $"Safetensors tensor '{propName}' byte size mismatch: data_offsets span {spanBytes} bytes, expected {expectedBytes} bytes for {dtype}.");
                }
            }

            Tensors[propName] = new SafetensorsTensorInfo(propName, dtype, shape, start, end);
        }
    }

    private static bool TryGetDtypeByteSize(string dtype, out ulong bytes)
    {
        bytes = dtype switch
        {
            "F64" or "I64" or "U64" => 8UL,
            "F32" or "I32" or "U32" => 4UL,
            "F16" or "BF16" or "I16" or "U16" => 2UL,
            "F8_E4M3" or "F8_E5M2" or "I8" or "U8" or "BOOL" => 1UL,
            _ => 0UL
        };
        return bytes > 0UL;
    }

    /// <summary>
    /// Gets raw byte pointer to tensor data.
    /// </summary>
    public byte* GetTensorPointer(string tensorName)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
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
                try { _accessor.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
                _basePointer = null;
            }
            _accessor.Dispose();
            _mmf.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
