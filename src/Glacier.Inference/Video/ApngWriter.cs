namespace Glacier.Inference.Video;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;

/// <summary>
/// Pure C# .NET 10 Animated Portable Network Graphics (APNG) Serializer.
/// Supports 24-bit lossless RGB per frame, animation control (acTL), frame control (fcTL),
/// and frame data (fdAT) chunks with zero external dependencies and zero-allocation chunk streaming.
/// </summary>
public static class ApngWriter
{
    private static readonly uint[] CrcTable = InitializeCrcTable();

    /// <summary>
    /// Saves a sequence of 24-bit RGB frames as an Animated PNG (APNG) file.
    /// </summary>
    public static void SaveApng(string filePath, IReadOnlyList<byte[]> frames, int width, int height, int fps = 8)
    {
        if (frames == null || frames.Count == 0)
        {
            throw new ArgumentException("Frames collection must contain at least one frame.", nameof(frames));
        }

        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        // 1. Standard PNG Signature (8 bytes)
        ReadOnlySpan<byte> pngSig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        bw.Write(pngSig);

        // 2. IHDR Chunk (Image Header)
        Span<byte> ihdrData = stackalloc byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdrData.Slice(0, 4), (uint)width);
        BinaryPrimitives.WriteUInt32BigEndian(ihdrData.Slice(4, 4), (uint)height);
        ihdrData[8] = 8; // Bit depth: 8 bits per channel
        ihdrData[9] = 2; // Color type: 2 (RGB truecolor)
        ihdrData[10] = 0; // Deflate
        ihdrData[11] = 0; // Standard adaptive filtering
        ihdrData[12] = 0; // Non-interlaced
        WriteChunk(bw, "IHDR"u8, ihdrData);

        // 3. acTL Chunk (Animation Control)
        // num_frames (4 bytes), num_plays (4 bytes: 0 = loop indefinitely)
        Span<byte> actlData = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32BigEndian(actlData.Slice(0, 4), (uint)frames.Count);
        BinaryPrimitives.WriteUInt32BigEndian(actlData.Slice(4, 4), 0); // 0 = infinite loop
        WriteChunk(bw, "acTL"u8, actlData);

        uint sequenceNumber = 0;
        int rowLength = width * 3;
        int filteredRowLen = 1 + rowLength;

        // Rent reusable row buffer across all frames in animation
        byte[] rentedRow = ArrayPool<byte>.Shared.Rent(filteredRowLen);
        try
        {
            Span<byte> filteredRow = rentedRow.AsSpan(0, filteredRowLen);
            filteredRow[0] = 0; // Filter type: None

            Span<byte> fctlData = stackalloc byte[26];

            for (int i = 0; i < frames.Count; i++)
            {
                // 4. fcTL Chunk (Frame Control - 26 bytes)
                // sequence_number (4)
                // width (4), height (4), x_offset (4), y_offset (4)
                // delay_num (2), delay_den (2)
                // dispose_op (1), blend_op (1)
                BinaryPrimitives.WriteUInt32BigEndian(fctlData.Slice(0, 4), sequenceNumber++);
                BinaryPrimitives.WriteUInt32BigEndian(fctlData.Slice(4, 4), (uint)width);
                BinaryPrimitives.WriteUInt32BigEndian(fctlData.Slice(8, 4), (uint)height);
                BinaryPrimitives.WriteUInt32BigEndian(fctlData.Slice(12, 4), 0); // x_offset = 0
                BinaryPrimitives.WriteUInt32BigEndian(fctlData.Slice(16, 4), 0); // y_offset = 0

                ushort delayNum = 1;
                ushort delayDen = (ushort)Math.Clamp(fps, 1, 1000);
                fctlData[20] = (byte)((delayNum >> 8) & 0xFF);
                fctlData[21] = (byte)(delayNum & 0xFF);
                fctlData[22] = (byte)((delayDen >> 8) & 0xFF);
                fctlData[23] = (byte)(delayDen & 0xFF);

                fctlData[24] = 0; // APNG_DISPOSE_OP_NONE
                fctlData[25] = 0; // APNG_BLEND_OP_SOURCE
                WriteChunk(bw, "fcTL"u8, fctlData);

                // Compress scanlines with ZLib into MemoryStream
                using var compressedMs = new MemoryStream();
                using (var zlib = new ZLibStream(compressedMs, CompressionLevel.Optimal, leaveOpen: true))
                {
                    ReadOnlySpan<byte> framePixels = frames[i];
                    for (int y = 0; y < height; y++)
                    {
                        int srcOffset = y * rowLength;
                        framePixels.Slice(srcOffset, rowLength).CopyTo(filteredRow.Slice(1));
                        zlib.Write(filteredRow);
                    }
                }

                ReadOnlySpan<byte> compressedSpan = compressedMs.TryGetBuffer(out ArraySegment<byte> seg)
                    ? seg.AsSpan()
                    : compressedMs.ToArray();

                if (i == 0)
                {
                    // Frame 0 uses standard IDAT chunk
                    WriteChunk(bw, "IDAT"u8, compressedSpan);
                }
                else
                {
                    // Frames 1..N use fdAT chunk with sequence number prefix (written incrementally)
                    WriteFdatChunk(bw, sequenceNumber++, compressedSpan);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rentedRow);
        }

        // 5. IEND Chunk
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

    private static void WriteFdatChunk(BinaryWriter bw, uint seqNum, ReadOnlySpan<byte> compressedData)
    {
        uint chunkLen = (uint)(4 + compressedData.Length);
        Span<byte> lenBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(lenBytes, chunkLen);
        bw.Write(lenBytes);

        bw.Write("fdAT"u8);

        Span<byte> seqBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(seqBytes, seqNum);
        bw.Write(seqBytes);

        bw.Write(compressedData);

        uint crc = 0xFFFFFFFF;
        crc = UpdateCrc(crc, "fdAT"u8);
        crc = UpdateCrc(crc, seqBytes);
        crc = UpdateCrc(crc, compressedData);
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
