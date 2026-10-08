namespace Glacier.Inference.Video;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Media;

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
        bw.Write("RIFF"u8);
        long riffLengthPos = fs.Position;
        bw.Write((uint)0); // Will update later
        bw.Write("AVI "u8);

        // 2. Main Header LIST 'hdrl'
        using (RiffScope.StartList(fs, bw, "hdrl"u8))
        {
            // 'avih' chunk (Main AVI Header - 56 bytes)
            using (RiffScope.StartChunk(fs, bw, "avih"u8))
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
            }

            // 'strl' LIST (Video Stream)
            using (RiffScope.StartList(fs, bw, "strl"u8))
            {
                // 'strh' (Stream Header - 56 bytes)
                using (RiffScope.StartChunk(fs, bw, "strh"u8))
                {
                    bw.Write("vids"u8);                                    // fccType
                    bw.Write("DIB "u8);                                    // fccHandler
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
                }

                // 'strf' (Stream Format: BITMAPINFOHEADER - 40 bytes)
                using (RiffScope.StartChunk(fs, bw, "strf"u8))
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
                }
            }
        }

        // 3. 'movi' LIST (Video Frame Data)
        var indexEntries = ArrayPool<(uint offset, uint size)>.Shared.Rent(frames.Count);
        try
        {
            long moviBasePos;
            using (RiffScope.StartList(fs, bw, "movi"u8))
            {
                moviBasePos = fs.Position; // Offset reference for idx1

                byte[] rentedFrame = ArrayPool<byte>.Shared.Rent(frameDataSize);
                try
                {
                    Span<byte> frameSpan = rentedFrame.AsSpan(0, frameDataSize);

                    for (int f = 0; f < frames.Count; f++)
                    {
                        long chunkStart = fs.Position;
                        uint relativeOffset = (uint)(chunkStart - moviBasePos);

                        bw.Write("00dc"u8);
                        bw.Write((uint)frameDataSize);

                        MediaKernels.ConvertFrameBottomUp(frames[f], frameSpan, width, height, rowStride);
                        fs.Write(frameSpan);

                        indexEntries[f] = (relativeOffset, (uint)frameDataSize);
                    }
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(rentedFrame);
                }
            }

            // 4. 'idx1' Chunk (AVI Index)
            using (RiffScope.StartChunk(fs, bw, "idx1"u8))
            {
                uint flags = 0x10; // AVIIF_KEYFRAME

                for (int f = 0; f < frames.Count; f++)
                {
                    bw.Write("00dc"u8);
                    bw.Write(flags);
                    bw.Write(indexEntries[f].offset);
                    bw.Write(indexEntries[f].size);
                }
            }
        }
        finally
        {
            ArrayPool<(uint offset, uint size)>.Shared.Return(indexEntries);
        }

        // 5. Update RIFF File Length
        long totalLength = fs.Length;
        fs.Seek(riffLengthPos, SeekOrigin.Begin);
        bw.Write((uint)(totalLength - 8));
    }

    private ref struct RiffScope
    {
        private readonly Stream _stream;
        private readonly BinaryWriter _bw;
        private readonly long _sizePosition;
        private readonly long _contentStartPosition;
        private readonly bool _isList;

        public static RiffScope StartList(Stream stream, BinaryWriter bw, ReadOnlySpan<byte> listType)
        {
            bw.Write("LIST"u8);
            long sizePos = stream.Position;
            bw.Write((uint)0); // Placeholder
            bw.Write(listType);
            return new RiffScope(stream, bw, sizePos, stream.Position, isList: true);
        }

        public static RiffScope StartChunk(Stream stream, BinaryWriter bw, ReadOnlySpan<byte> chunkType)
        {
            bw.Write(chunkType);
            long sizePos = stream.Position;
            bw.Write((uint)0); // Placeholder
            return new RiffScope(stream, bw, sizePos, stream.Position, isList: false);
        }

        private RiffScope(Stream stream, BinaryWriter bw, long sizePosition, long contentStartPosition, bool isList)
        {
            _stream = stream;
            _bw = bw;
            _sizePosition = sizePosition;
            _contentStartPosition = contentStartPosition;
            _isList = isList;
        }

        public void Dispose()
        {
            long contentEnd = _stream.Position;
            long length = contentEnd - _contentStartPosition;
            if (_isList)
            {
                length += 4; // list length includes the 4-byte list type
            }

            _stream.Seek(_sizePosition, SeekOrigin.Begin);
            _bw.Write((uint)length);
            _stream.Seek(contentEnd, SeekOrigin.Begin);

            if (!_isList && (length % 2 != 0))
            {
                _bw.Write((byte)0);
            }
        }
    }
}
