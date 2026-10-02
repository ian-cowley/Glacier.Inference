namespace Glacier.Inference.Image;

using System;
using System.IO;
using System.IO.Compression;

/// <summary>
/// Pure C# .NET 10 Portable Network Graphics (PNG) 24-bit RGB serializer.
/// Implements standard W3C PNG specification with zero external dependencies.
/// </summary>
public static class PngWriter
{
    private static readonly uint[] CrcTable = InitializeCrcTable();

    public static void SavePng24(string filePath, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        // 1. Standard PNG Signature (8 bytes)
        bw.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        // 2. IHDR Chunk (Image Header)
        byte[] ihdrData = new byte[13];
        WriteBigEndian(ihdrData, 0, (uint)width);
        WriteBigEndian(ihdrData, 4, (uint)height);
        ihdrData[8] = 8; // Bit depth: 8 bits per channel
        ihdrData[9] = 2; // Color type: 2 (RGB truecolor)
        ihdrData[10] = 0; // Compression method: 0 (deflate)
        ihdrData[11] = 0; // Filter method: 0 (standard adaptive)
        ihdrData[12] = 0; // Interlace method: 0 (non-interlaced)
        WriteChunk(bw, "IHDR", ihdrData);

        // 3. IDAT Chunk (ZLib Compressed Scanlines with filter byte 0)
        using (var compressedMs = new MemoryStream())
        {
            using (var zlib = new ZLibStream(compressedMs, CompressionLevel.Optimal, leaveOpen: true))
            {
                int rowLength = width * 3;
                byte[] filteredRow = new byte[1 + rowLength];
                filteredRow[0] = 0; // Filter type: None

                for (int y = 0; y < height; y++)
                {
                    int srcOffset = y * rowLength;
                    rgbPixels.Slice(srcOffset, rowLength).CopyTo(filteredRow.AsSpan(1));
                    zlib.Write(filteredRow, 0, filteredRow.Length);
                }
            }

            WriteChunk(bw, "IDAT", compressedMs.ToArray());
        }

        // 4. IEND Chunk (Image End)
        WriteChunk(bw, "IEND", Array.Empty<byte>());
    }

    private static void WriteChunk(BinaryWriter bw, string typeStr, byte[] data)
    {
        uint length = (uint)data.Length;
        byte[] typeBytes = System.Text.Encoding.ASCII.GetBytes(typeStr);

        // Write length (big-endian)
        bw.Write(SwapEndian(length));

        // Write type
        bw.Write(typeBytes);

        // Write data
        if (data.Length > 0)
        {
            bw.Write(data);
        }

        // Calculate CRC32 over type + data
        uint crc = 0xFFFFFFFF;
        for (int i = 0; i < typeBytes.Length; i++)
        {
            crc = (crc >> 8) ^ CrcTable[(crc ^ typeBytes[i]) & 0xFF];
        }
        for (int i = 0; i < data.Length; i++)
        {
            crc = (crc >> 8) ^ CrcTable[(crc ^ data[i]) & 0xFF];
        }
        crc ^= 0xFFFFFFFF;

        // Write CRC (big-endian)
        bw.Write(SwapEndian(crc));
    }

    private static void WriteBigEndian(byte[] buffer, int offset, uint value)
    {
        buffer[offset]     = (byte)((value >> 24) & 0xFF);
        buffer[offset + 1] = (byte)((value >> 16) & 0xFF);
        buffer[offset + 2] = (byte)((value >> 8) & 0xFF);
        buffer[offset + 3] = (byte)(value & 0xFF);
    }

    private static uint SwapEndian(uint value)
    {
        return ((value & 0x000000FF) << 24) |
               ((value & 0x0000FF00) << 8) |
               ((value & 0x00FF0000) >> 8) |
               ((value & 0xFF000000) >> 24);
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
