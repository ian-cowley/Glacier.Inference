namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Pure C# .NET 10 Audio Video Interleave (AVI) Container Serializer.
/// Encodes raw 24-bit RGB frames into a standard RIFF/AVI container stream with
/// index tables (idx1), compatible with standard desktop and mobile media players.
/// </summary>
public static class AviWriter
{
    public static void SaveAvi(string filePath, IReadOnlyList<byte[]> frames, int width, int height, int fps = 8)
    {
        if (frames == null || frames.Count == 0)
        {
            throw new ArgumentException("Frames collection must contain at least one frame.", nameof(frames));
        }

        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        int rowStride = (width * 3 + 3) & ~3; // 4-byte aligned scanline
        int frameDataSize = rowStride * height;
        uint microSecPerFrame = (uint)(1_000_000 / Math.Clamp(fps, 1, 120));

        // 1. RIFF Header placeholder
        bw.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
        long riffLengthPos = fs.Position;
        bw.Write((uint)0); // Will update later
        bw.Write(System.Text.Encoding.ASCII.GetBytes("AVI "));

        // 2. Main Header LIST 'hdrl'
        WriteList(bw, "hdrl", () =>
        {
            // 'avih' chunk (Main AVI Header - 56 bytes)
            WriteChunk(bw, "avih", () =>
            {
                bw.Write(microSecPerFrame);                       // dwMicroSecPerFrame
                bw.Write((uint)(frameDataSize * fps));             // dwMaxBytesPerSec
                bw.Write((uint)0);                                 // dwPaddingGranularity
                bw.Write((uint)0x800);                             // dwFlags (AVIF_ISINTERLEAVED)
                bw.Write((uint)frames.Count);                      // dwTotalFrames
                bw.Write((uint)0);                                 // dwInitialFrames
                bw.Write((uint)1);                                 // dwStreams
                bw.Write((uint)frameDataSize);                     // dwSuggestedBufferSize
                bw.Write((uint)width);                             // dwWidth
                bw.Write((uint)height);                            // dwHeight
                bw.Write((uint)0); bw.Write((uint)0);              // dwReserved
                bw.Write((uint)0); bw.Write((uint)0);
            });

            // 'strl' LIST (Video Stream)
            WriteList(bw, "strl", () =>
            {
                // 'strh' (Stream Header - 56 bytes)
                WriteChunk(bw, "strh", () =>
                {
                    bw.Write(System.Text.Encoding.ASCII.GetBytes("vids")); // fccType
                    bw.Write(System.Text.Encoding.ASCII.GetBytes("DIB ")); // fccHandler
                    bw.Write((uint)0);                                     // dwFlags
                    bw.Write((ushort)0);                                   // wPriority
                    bw.Write((ushort)0);                                   // wLanguage
                    bw.Write((uint)0);                                     // dwInitialFrames
                    bw.Write((uint)1);                                     // dwScale
                    bw.Write((uint)fps);                                   // dwRate
                    bw.Write((uint)0);                                     // dwStart
                    bw.Write((uint)frames.Count);                          // dwLength
                    bw.Write((uint)frameDataSize);                         // dwSuggestedBufferSize
                    bw.Write((uint)0);                                     // dwQuality
                    bw.Write((uint)0);                                     // dwSampleSize
                    bw.Write((short)0); bw.Write((short)0);                // rcFrame left, top
                    bw.Write((short)width); bw.Write((short)height);       // rcFrame right, bottom
                });

                // 'strf' (Stream Format: BITMAPINFOHEADER - 40 bytes)
                WriteChunk(bw, "strf", () =>
                {
                    bw.Write((uint)40);              // biSize
                    bw.Write((int)width);            // biWidth
                    bw.Write((int)height);           // biHeight (positive = bottom-up DIB)
                    bw.Write((ushort)1);             // biPlanes
                    bw.Write((ushort)24);            // biBitCount
                    bw.Write((uint)0);               // biCompression (BI_RGB uncompressed)
                    bw.Write((uint)frameDataSize);   // biSizeImage
                    bw.Write((int)0); bw.Write((int)0); // biXPelsPerMeter, biYPelsPerMeter
                    bw.Write((uint)0); bw.Write((uint)0); // biClrUsed, biClrImportant
                });
            });
        });

        // 3. 'movi' LIST (Video Frame Data)
        var indexEntries = new List<(uint offset, uint size)>(frames.Count);
        long moviBasePos = 0;

        WriteList(bw, "movi", () =>
        {
            moviBasePos = fs.Position; // Offset reference for idx1

            byte[] dibRow = new byte[rowStride];

            for (int f = 0; f < frames.Count; f++)
            {
                long chunkStart = fs.Position;
                uint relativeOffset = (uint)(chunkStart - moviBasePos);

                bw.Write(System.Text.Encoding.ASCII.GetBytes("00dc"));
                bw.Write((uint)frameDataSize);

                byte[] rgb = frames[f];

                // Write bottom-up BGR scanlines
                for (int y = height - 1; y >= 0; y--)
                {
                    int srcRow = y * width * 3;
                    for (int x = 0; x < width; x++)
                    {
                        byte r = rgb[srcRow + x * 3 + 0];
                        byte g = rgb[srcRow + x * 3 + 1];
                        byte b = rgb[srcRow + x * 3 + 2];

                        dibRow[x * 3 + 0] = b; // Blue
                        dibRow[x * 3 + 1] = g; // Green
                        dibRow[x * 3 + 2] = r; // Red
                    }

                    // Pad remaining bytes to 4-byte boundary
                    for (int p = width * 3; p < rowStride; p++)
                    {
                        dibRow[p] = 0;
                    }

                    bw.Write(dibRow, 0, rowStride);
                }

                indexEntries.Add((relativeOffset, (uint)frameDataSize));
            }
        });

        // 4. 'idx1' Chunk (AVI Index)
        WriteChunk(bw, "idx1", () =>
        {
            byte[] ckid = System.Text.Encoding.ASCII.GetBytes("00dc");
            uint flags = 0x10; // AVIIF_KEYFRAME

            foreach (var (offset, size) in indexEntries)
            {
                bw.Write(ckid);
                bw.Write(flags);
                bw.Write(offset);
                bw.Write(size);
            }
        });

        // 5. Update RIFF File Length
        long totalLength = fs.Length;
        fs.Seek(riffLengthPos, SeekOrigin.Begin);
        bw.Write((uint)(totalLength - 8));
    }

    private static void WriteList(BinaryWriter bw, string fourCC, Action writeContents)
    {
        bw.Write(System.Text.Encoding.ASCII.GetBytes("LIST"));
        long sizePos = bw.BaseStream.Position;
        bw.Write((uint)0); // Placeholder
        bw.Write(System.Text.Encoding.ASCII.GetBytes(fourCC));

        long contentStart = bw.BaseStream.Position;
        writeContents();
        long contentEnd = bw.BaseStream.Position;

        long listLength = contentEnd - contentStart + 4; // Including fourCC
        bw.BaseStream.Seek(sizePos, SeekOrigin.Begin);
        bw.Write((uint)listLength);
        bw.BaseStream.Seek(contentEnd, SeekOrigin.Begin);
    }

    private static void WriteChunk(BinaryWriter bw, string fourCC, Action writeContents)
    {
        bw.Write(System.Text.Encoding.ASCII.GetBytes(fourCC));
        long sizePos = bw.BaseStream.Position;
        bw.Write((uint)0); // Placeholder

        long contentStart = bw.BaseStream.Position;
        writeContents();
        long contentEnd = bw.BaseStream.Position;

        long chunkLength = contentEnd - contentStart;
        bw.BaseStream.Seek(sizePos, SeekOrigin.Begin);
        bw.Write((uint)chunkLength);
        bw.BaseStream.Seek(contentEnd, SeekOrigin.Begin);

        // Word alignment padding (if odd byte count)
        if (chunkLength % 2 != 0)
        {
            bw.Write((byte)0);
        }
    }
}
