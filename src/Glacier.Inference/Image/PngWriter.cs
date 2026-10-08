namespace Glacier.Inference.Image;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;

/// <summary>
/// Pure C# .NET 10 Portable Network Graphics (PNG) 24-bit RGB serializer.
/// Implements standard W3C PNG specification with zero external dependencies and zero-allocation chunk streaming.
/// </summary>
public static class PngWriter
{
    private static readonly uint[] CrcTable = InitializeCrcTable();

    public static void SavePng24(string filePath, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        // 1. Standard PNG Signature (8 bytes)
        ReadOnlySpan<byte> pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        bw.Write(pngSignature);

        // 2. IHDR Chunk (Image Header)
        Span<byte> ihdrData = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdrData.Slice(0, 4), (uint)width);
        BinaryPrimitives.WriteUInt32BigEndian(ihdrData.Slice(4, 4), (uint)height);
        ihdrData[8] = 8; // Bit depth: 8 bits per channel
        ihdrData[9] = 2; // Color type: 2 (RGB truecolor)
        ihdrData[10] = 0; // Compression method: 0 (deflate)
        ihdrData[11] = 0; // Filter method: 0 (standard adaptive)
        ihdrData[12] = 0; // Interlace method: 0 (non-interlaced)
        WriteChunk(bw, "IHDR"u8, ihdrData);

        // 3. IDAT Chunk (ZLib Compressed Scanlines with filter byte 0)
        int rowLength = width * 3;
        int filteredRowLen = 1 + rowLength;
        byte[] rentedRow = ArrayPool<byte>.Shared.Rent(filteredRowLen);
        try
        {
            Span<byte> filteredRow = rentedRow.AsSpan(0, filteredRowLen);
            filteredRow[0] = 0; // Filter type: None

            using var compressedMs = new MemoryStream();
            using (var zlib = new ZLibStream(compressedMs, CompressionLevel.Optimal, leaveOpen: true))
            {
                for (int y = 0; y < height; y++)
                {
                    int srcOffset = y * rowLength;
                    rgbPixels.Slice(srcOffset, rowLength).CopyTo(filteredRow.Slice(1));
                    zlib.Write(filteredRow);
                }
            }

            ReadOnlySpan<byte> compressedSpan = compressedMs.TryGetBuffer(out ArraySegment<byte> seg)
                ? seg.AsSpan()
                : compressedMs.ToArray();

            WriteChunk(bw, "IDAT"u8, compressedSpan);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rentedRow);
        }

        // 4. IEND Chunk (Image End)
        WriteChunk(bw, "IEND"u8, ReadOnlySpan<byte>.Empty);
    }

    private static void WriteChunk(BinaryWriter bw, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        uint length = (uint)data.Length;
        Span<byte> lenBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(lenBytes, length);
        bw.Write(lenBytes);

        bw.Write(type);

        if (!data.IsEmpty)
        {
            bw.Write(data);
        }

        uint crc = 0xFFFFFFFF;
        crc = UpdateCrc(crc, type);
        if (!data.IsEmpty)
        {
            crc = UpdateCrc(crc, data);
        }
        crc ^= 0xFFFFFFFF;

        Span<byte> crcBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crcBytes, crc);
        bw.Write(crcBytes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint UpdateCrc(uint crc, ReadOnlySpan<byte> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            crc = (crc >> 8) ^ CrcTable[(crc ^ data[i]) & 0xFF];
        }
        return crc;
    }

    private static uint[] InitializeCrcTable()
    {
        uint[] table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                if ((c & 1) != 0)
                    c = 0xEDB88320 ^ (c >> 1);
                else
                    c >>= 1;
            }
            table[n] = c;
        }
        return table;
    }
}
