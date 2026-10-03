namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

/// <summary>
/// Pure C# .NET 10 Animated Portable Network Graphics (APNG) Serializer.
/// Supports 24-bit lossless RGB per frame, animation control (acTL), frame control (fcTL),
/// and frame data (fdAT) chunks with zero external dependencies.
/// </summary>
public static class ApngWriter
{
    private static readonly uint[] CrcTable = InitializeCrcTable();

    /// <summary>
    /// Saves a sequence of 24-bit RGB frames as an Animated PNG (APNG) file.
    /// </summary>
    /// <param name="filePath">Target .apng or .png file path.</param>
    /// <param name="frames">List of frame byte arrays (each width * height * 3 in RGB format).</param>
    /// <param name="width">Frame pixel width.</param>
    /// <param name="height">Frame pixel height.</param>
    /// <param name="fps">Frames per second playback rate.</param>
    public static void SaveApng(string filePath, IReadOnlyList<byte[]> frames, int width, int height, int fps = 8)
    {
        if (frames == null || frames.Count == 0)
        {
            throw new ArgumentException("Frames collection must contain at least one frame.", nameof(frames));
        }

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
        ihdrData[10] = 0; // Deflate
        ihdrData[11] = 0; // Standard adaptive filtering
        ihdrData[12] = 0; // Non-interlaced
        WriteChunk(bw, "IHDR", ihdrData);

        // 3. acTL Chunk (Animation Control)
        // num_frames (4 bytes), num_plays (4 bytes: 0 = loop indefinitely)
        byte[] actlData = new byte[8];
        WriteBigEndian(actlData, 0, (uint)frames.Count);
        WriteBigEndian(actlData, 4, 0); // 0 = infinite loop
        WriteChunk(bw, "acTL", actlData);

        uint sequenceNumber = 0;

        for (int i = 0; i < frames.Count; i++)
        {
            // 4. fcTL Chunk (Frame Control)
            // sequence_number (4)
            // width (4), height (4), x_offset (4), y_offset (4)
            // delay_num (2), delay_den (2)
            // dispose_op (1), blend_op (1)
            byte[] fctlData = new byte[26];
            WriteBigEndian(fctlData, 0, sequenceNumber++);
            WriteBigEndian(fctlData, 4, (uint)width);
            WriteBigEndian(fctlData, 8, (uint)height);
            WriteBigEndian(fctlData, 12, 0); // x_offset = 0
            WriteBigEndian(fctlData, 16, 0); // y_offset = 0

            ushort delayNum = 1;
            ushort delayDen = (ushort)Math.Clamp(fps, 1, 1000);
            fctlData[20] = (byte)((delayNum >> 8) & 0xFF);
            fctlData[21] = (byte)(delayNum & 0xFF);
            fctlData[22] = (byte)((delayDen >> 8) & 0xFF);
            fctlData[23] = (byte)(delayDen & 0xFF);

            fctlData[24] = 0; // APNG_DISPOSE_OP_NONE
            fctlData[25] = 0; // APNG_BLEND_OP_SOURCE
            WriteChunk(bw, "fcTL", fctlData);

            // Compress scanlines with ZLib
            byte[] compressedScanlines = CompressRgbScanlines(frames[i], width, height);

            if (i == 0)
            {
                // Frame 0 uses standard IDAT chunk
                WriteChunk(bw, "IDAT", compressedScanlines);
            }
            else
            {
                // Frames 1..N use fdAT chunk with sequence number prefix
                byte[] fdatData = new byte[4 + compressedScanlines.Length];
                WriteBigEndian(fdatData, 0, sequenceNumber++);
                Buffer.BlockCopy(compressedScanlines, 0, fdatData, 4, compressedScanlines.Length);
                WriteChunk(bw, "fdAT", fdatData);
            }
        }

        // 5. IEND Chunk
        WriteChunk(bw, "IEND", Array.Empty<byte>());
    }

    private static byte[] CompressRgbScanlines(byte[] rgbPixels, int width, int height)
    {
        using var ms = new MemoryStream();
        using (var zlib = new ZLibStream(ms, CompressionLevel.Optimal, leaveOpen: true))
        {
            int rowLength = width * 3;
            byte[] filteredRow = new byte[1 + rowLength];
            filteredRow[0] = 0; // Filter type: None

            for (int y = 0; y < height; y++)
            {
                int srcOffset = y * rowLength;
                Buffer.BlockCopy(rgbPixels, srcOffset, filteredRow, 1, rowLength);
                zlib.Write(filteredRow, 0, filteredRow.Length);
            }
        }
        return ms.ToArray();
    }

    private static void WriteChunk(BinaryWriter bw, string typeStr, byte[] data)
    {
        uint length = (uint)data.Length;
        byte[] typeBytes = System.Text.Encoding.ASCII.GetBytes(typeStr);

        bw.Write(SwapEndian(length));
        bw.Write(typeBytes);

        if (data.Length > 0)
        {
            bw.Write(data);
        }

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
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
            {
                if ((c & 1) != 0)
                    c = 0xEDB88320 ^ (c >> 1);
                else
                    c >>= 1;
            }
            table[i] = c;
        }
        return table;
    }
}
