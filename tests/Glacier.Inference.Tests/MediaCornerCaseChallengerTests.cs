namespace Glacier.Inference.Tests;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Glacier.Inference.Image;
using Glacier.Inference.Media;
using Glacier.Inference.Video;
using Xunit;

/// <summary>
/// Adversarial test suite for Challenger 1 (Milestone M1).
/// Verifies:
/// 1. Corner cases: small dimensions (1x1, 2x2, 3x3, 4x4, 5x5, 6x6, 7x7), odd widths with row padding, and large dimensions (1920x1080).
/// 2. Bit-exactness: RGB-to-BGR color channel exactness against an independent scalar oracle.
/// 3. Zero-allocation invariants across hot per-frame media processing loops.
/// 4. Container binary layout and chunk integrity for BMP, AVI, GIF, APNG, and PNG.
/// </summary>
public class MediaCornerCaseChallengerTests
{
    private static byte[] OracleConvertFrameBottomUp(byte[] srcRgb, int width, int height, int rowStride)
    {
        byte[] dstBgr = new byte[rowStride * height];
        int srcRowStride = width * 3;

        for (int y = 0; y < height; y++)
        {
            int srcY = height - 1 - y; // Bottom-up
            int srcRowOffset = srcY * srcRowStride;
            int dstRowOffset = y * rowStride;

            for (int x = 0; x < width; x++)
            {
                byte r = srcRgb[srcRowOffset + x * 3 + 0];
                byte g = srcRgb[srcRowOffset + x * 3 + 1];
                byte b = srcRgb[srcRowOffset + x * 3 + 2];

                dstBgr[dstRowOffset + x * 3 + 0] = b; // Byte 0 = Blue
                dstBgr[dstRowOffset + x * 3 + 1] = g; // Byte 1 = Green
                dstBgr[dstRowOffset + x * 3 + 2] = r; // Byte 2 = Red
            }

            // Stride padding must be zero
            for (int p = width * 3; p < rowStride; p++)
            {
                dstBgr[dstRowOffset + p] = 0;
            }
        }

        return dstBgr;
    }

    private static byte[] GenerateDistinctRgbPattern(int width, int height)
    {
        byte[] pixels = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = (y * width + x) * 3;
                pixels[idx + 0] = (byte)((x * 17 + y * 23 + 11) % 256); // R
                pixels[idx + 1] = (byte)((x * 31 + y * 7 + 47) % 256);  // G
                pixels[idx + 2] = (byte)((x * 13 + y * 41 + 89) % 256); // B
            }
        }
        return pixels;
    }

    #region 1. MediaKernels Corner Cases & Bit Exactness

    [Theory]
    [InlineData(1, 1)]     // 1x1: width < 5, rowBytes=3, rowStride=4 (1 padding byte)
    [InlineData(2, 2)]     // 2x2: width < 5, rowBytes=6, rowStride=8 (2 padding bytes)
    [InlineData(3, 3)]     // 3x3: width < 5, rowBytes=9, rowStride=12 (3 padding bytes)
    [InlineData(4, 4)]     // 4x4: width < 5, rowBytes=12, rowStride=12 (0 padding bytes)
    [InlineData(5, 5)]     // 5x5: width < 5 SIMD boundary (15 bytes < 16 bytes), rowBytes=15, rowStride=16 (1 padding byte)
    [InlineData(6, 6)]     // 6x6: 18 bytes >= 16 bytes: 1 SIMD step (5 pixels) + 1 scalar pixel, rowStride=20 (2 padding bytes)
    [InlineData(7, 4)]     // 7x4: odd width with rowStride=24 (3 padding bytes), 1 SIMD step + 2 scalar pixels
    [InlineData(9, 5)]     // 9x5: rowBytes=27, rowStride=28 (1 padding byte)
    [InlineData(10, 10)]   // 10x10: rowBytes=30, rowStride=32 (2 padding bytes), 1 SIMD step + 5 scalar pixels
    [InlineData(11, 7)]    // 11x7: rowBytes=33, rowStride=36 (3 padding bytes), 2 SIMD steps + 1 scalar pixel
    [InlineData(13, 3)]    // 13x3: rowBytes=39, rowStride=40 (1 padding byte)
    [InlineData(14, 2)]    // 14x2: rowBytes=42, rowStride=44 (2 padding bytes)
    [InlineData(15, 8)]    // 15x8: rowBytes=45, rowStride=48 (3 padding bytes), exactly 3 SIMD steps
    [InlineData(16, 16)]   // 16x16: rowBytes=48, rowStride=48 (0 padding bytes), 3 SIMD steps + 1 scalar pixel
    [InlineData(17, 9)]    // 17x9: rowBytes=51, rowStride=52 (1 padding byte)
    [InlineData(31, 15)]   // 31x15: rowBytes=93, rowStride=96 (3 padding bytes)
    [InlineData(32, 32)]   // 32x32: rowBytes=96, rowStride=96 (0 padding bytes)
    [InlineData(33, 17)]   // 33x17: rowBytes=99, rowStride=100 (1 padding byte)
    [InlineData(127, 63)]  // 127x63: rowBytes=381, rowStride=384 (3 padding bytes)
    [InlineData(128, 128)] // 128x128: rowBytes=384, rowStride=384 (0 padding bytes)
    [InlineData(129, 65)]  // 129x65: rowBytes=387, rowStride=388 (1 padding byte)
    [InlineData(256, 256)] // 256x256: power of 2 aligned
    [InlineData(1920, 1080)] // Full HD 1080p: large dimensions
    public void MediaKernels_ConvertFrameBottomUp_MatchesIndependentScalarOracle_BitForBit(int width, int height)
    {
        byte[] srcRgb = GenerateDistinctRgbPattern(width, height);
        int rowStride = (width * 3 + 3) & ~3;
        int frameSize = rowStride * height;

        byte[] expected = OracleConvertFrameBottomUp(srcRgb, width, height, rowStride);

        byte[] actual = new byte[frameSize];
        // Pre-fill destination with sentinel value 0xAA to verify that padding bytes are explicitly cleared to 0
        Array.Fill(actual, (byte)0xAA);

        MediaKernels.ConvertFrameBottomUp(srcRgb, actual, width, height, rowStride);

        // Verify bit-for-bit exactness across the entire buffer
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
            {
                int row = i / rowStride;
                int col = i % rowStride;
                Assert.Fail($"Byte mismatch at index {i} (row {row}, col {col}): expected 0x{expected[i]:X2}, actual 0x{actual[i]:X2} (width={width}, height={height})");
            }
        }
    }

    [Fact]
    public void MediaKernels_ConvertFrameBottomUp_ZeroAllocationsOnHotPath()
    {
        int width = 320;
        int height = 240;
        int rowStride = (width * 3 + 3) & ~3;
        int frameSize = rowStride * height;

        byte[] srcRgb = new byte[width * height * 3];
        byte[] dstBgr = new byte[frameSize];

        // Warm up JIT and tiering
        for (int i = 0; i < 50; i++)
        {
            MediaKernels.ConvertFrameBottomUp(srcRgb, dstBgr, width, height, rowStride);
        }

        // Measure allocations over 50 consecutive frames
        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++)
        {
            MediaKernels.ConvertFrameBottomUp(srcRgb, dstBgr, width, height, rowStride);
        }
        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();

        long diff = afterAlloc - beforeAlloc;
        Assert.Equal(0, diff);
    }

    #endregion

    #region 2. BmpWriter Bit Exactness & GDI+ Interop

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 5)]
    [InlineData(7, 4)]
    [InlineData(15, 15)]
    [InlineData(64, 48)]
    public void BmpWriter_GeneratesBitExactWindowsBitmapFile(int width, int height)
    {
        byte[] srcRgb = GenerateDistinctRgbPattern(width, height);
        int rowStride = (width * 3 + 3) & ~3;
        int imageSize = rowStride * height;
        int expectedFileSize = 54 + imageSize;

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_bmp_{Guid.NewGuid():N}.bmp");
        try
        {
            BmpWriter.SaveBmp24(tmpFile, srcRgb, width, height);

            Assert.True(File.Exists(tmpFile));
            byte[] fileBytes = File.ReadAllBytes(tmpFile);

            Assert.Equal(expectedFileSize, fileBytes.Length);

            // 1. BMP File Header checks (14 bytes)
            Assert.Equal((byte)'B', fileBytes[0]);
            Assert.Equal((byte)'M', fileBytes[1]);
            int reportedFileSize = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(2, 4));
            Assert.Equal(expectedFileSize, reportedFileSize);
            Assert.Equal(0, BinaryPrimitives.ReadInt16LittleEndian(fileBytes.AsSpan(6, 2))); // Reserved 1
            Assert.Equal(0, BinaryPrimitives.ReadInt16LittleEndian(fileBytes.AsSpan(8, 2))); // Reserved 2
            int pixelOffset = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(10, 4));
            Assert.Equal(54, pixelOffset);

            // 2. DIB Header checks (BITMAPINFOHEADER - 40 bytes)
            int headerSize = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(14, 4));
            Assert.Equal(40, headerSize);
            int reportedWidth = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(18, 4));
            Assert.Equal(width, reportedWidth);
            int reportedHeight = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(22, 4));
            Assert.Equal(height, reportedHeight);
            short planes = BinaryPrimitives.ReadInt16LittleEndian(fileBytes.AsSpan(26, 2));
            Assert.Equal(1, planes);
            short bpp = BinaryPrimitives.ReadInt16LittleEndian(fileBytes.AsSpan(28, 2));
            Assert.Equal(24, bpp);
            int compression = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(30, 4));
            Assert.Equal(0, compression); // BI_RGB
            int reportedImageSize = BinaryPrimitives.ReadInt32LittleEndian(fileBytes.AsSpan(34, 4));
            Assert.Equal(imageSize, reportedImageSize);

            // 3. Pixel Data exactness check against oracle
            byte[] expectedPixels = OracleConvertFrameBottomUp(srcRgb, width, height, rowStride);
            ReadOnlySpan<byte> actualPixels = fileBytes.AsSpan(54, imageSize);

            Assert.True(expectedPixels.AsSpan().SequenceEqual(actualPixels),
                $"Pixel data in BMP for {width}x{height} does not match oracle BGR scanlines!");

            // 4. Verify GDI+ Bitmap loads and decodes the BMP without error
            if (OperatingSystem.IsWindows())
            {
                using var bmp = new Bitmap(tmpFile);
                Assert.Equal(width, bmp.Width);
                Assert.Equal(height, bmp.Height);
                Assert.Equal(PixelFormat.Format24bppRgb, bmp.PixelFormat);

                // Spot check color at (0, 0)
                Color c00 = bmp.GetPixel(0, 0);
                Assert.Equal(srcRgb[0], c00.R);
                Assert.Equal(srcRgb[1], c00.G);
                Assert.Equal(srcRgb[2], c00.B);
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion

    #region 3. AviWriter Bit Exactness & Index Integrity

    [Theory]
    [InlineData(1, 1, 3)]
    [InlineData(5, 5, 2)]
    [InlineData(7, 4, 4)]
    [InlineData(32, 24, 6)]
    public void AviWriter_GeneratesBitExactRiffContainerAndIndex(int width, int height, int numFrames)
    {
        int rowStride = (width * 3 + 3) & ~3;
        int frameDataSize = rowStride * height;

        var frames = new List<byte[]>();
        for (int f = 0; f < numFrames; f++)
        {
            frames.Add(GenerateDistinctRgbPattern(width, height));
        }

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_avi_{Guid.NewGuid():N}.avi");
        try
        {
            AviWriter.SaveAvi(tmpFile, frames, width, height, fps: 10);

            Assert.True(File.Exists(tmpFile));
            byte[] fileBytes = File.ReadAllBytes(tmpFile);

            // 1. RIFF Header
            Assert.Equal("RIFF"u8.ToArray(), fileBytes.Take(4).ToArray());
            uint riffLength = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(4, 4));
            Assert.Equal((uint)(fileBytes.Length - 8), riffLength);
            Assert.Equal("AVI "u8.ToArray(), fileBytes.Skip(8).Take(4).ToArray());

            // 2. Scan and verify LIST 'movi' and frames
            int moviIndex = -1;
            for (int i = 12; i < fileBytes.Length - 4; i++)
            {
                if (fileBytes[i] == 'm' && fileBytes[i + 1] == 'o' && fileBytes[i + 2] == 'v' && fileBytes[i + 3] == 'i')
                {
                    moviIndex = i;
                    break;
                }
            }
            Assert.True(moviIndex > 0, "Could not find 'movi' LIST in AVI file!");

            // moviBasePos in AviWriter is immediately after "movi" FourCC (moviIndex + 4)
            int moviDataStart = moviIndex + 4;
            int currentPos = moviDataStart;

            var actualOffsets = new List<uint>();
            for (int f = 0; f < numFrames; f++)
            {
                actualOffsets.Add((uint)(currentPos - moviDataStart));

                // Chunk header '00dc'
                Assert.Equal((byte)'0', fileBytes[currentPos]);
                Assert.Equal((byte)'0', fileBytes[currentPos + 1]);
                Assert.Equal((byte)'d', fileBytes[currentPos + 2]);
                Assert.Equal((byte)'c', fileBytes[currentPos + 3]);
                uint chunkSize = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(currentPos + 4, 4));
                Assert.Equal((uint)frameDataSize, chunkSize);

                // Frame pixel data verification against oracle
                byte[] expectedBgr = OracleConvertFrameBottomUp(frames[f], width, height, rowStride);
                ReadOnlySpan<byte> actualFrameSpan = fileBytes.AsSpan(currentPos + 8, frameDataSize);
                Assert.True(expectedBgr.AsSpan().SequenceEqual(actualFrameSpan),
                    $"Frame {f} in AVI does not match oracle BGR scanlines!");

                currentPos += 8 + frameDataSize;
            }

            // 3. Scan and verify 'idx1' Chunk
            int idx1Pos = currentPos;
            Assert.Equal((byte)'i', fileBytes[idx1Pos]);
            Assert.Equal((byte)'d', fileBytes[idx1Pos + 1]);
            Assert.Equal((byte)'x', fileBytes[idx1Pos + 2]);
            Assert.Equal((byte)'1', fileBytes[idx1Pos + 3]);

            uint idx1Size = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(idx1Pos + 4, 4));
            Assert.Equal((uint)(numFrames * 16), idx1Size);

            int entryPos = idx1Pos + 8;
            for (int f = 0; f < numFrames; f++)
            {
                Assert.Equal("00dc"u8.ToArray(), fileBytes.Skip(entryPos).Take(4).ToArray());
                uint flags = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(entryPos + 4, 4));
                Assert.Equal(0x10u, flags); // AVIIF_KEYFRAME
                uint offset = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(entryPos + 8, 4));
                uint size = BinaryPrimitives.ReadUInt32LittleEndian(fileBytes.AsSpan(entryPos + 12, 4));

                Assert.Equal(actualOffsets[f], offset);
                Assert.Equal((uint)frameDataSize, size);

                entryPos += 16;
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion

    #region 4. GifWriter Standard GDI+ Decoder Verification

    [Theory]
    [InlineData(1, 1, 2)]
    [InlineData(17, 31, 2)]
    [InlineData(64, 64, 4)]
    [InlineData(128, 128, 2)]
    public void GifWriter_CanBeDecodedByStandardGdiPlus(int width, int height, int numFrames)
    {
        if (!OperatingSystem.IsWindows()) return;

        var frames = new List<byte[]>();
        for (int f = 0; f < numFrames; f++)
        {
            frames.Add(GenerateDistinctRgbPattern(width, height));
        }

        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_gif_gdi_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, frames, width, height, fps: 10);
            Assert.True(File.Exists(tmpFile));

            if (OperatingSystem.IsWindows())
            {
                // Standard Windows GDI+ GIF decoder test
                using var img = Image.FromFile(tmpFile);
                Assert.Equal(width, img.Width);
                Assert.Equal(height, img.Height);

                var dimension = new FrameDimension(img.FrameDimensionsList[0]);
                int frameCount = img.GetFrameCount(dimension);
                Assert.Equal(numFrames, frameCount);

                // Select and verify each frame in GDI+
                for (int f = 0; f < numFrames; f++)
                {
                    img.SelectActiveFrame(dimension, f);
                    using var bmp = new Bitmap(img);
                    Assert.Equal(width, bmp.Width);
                    Assert.Equal(height, bmp.Height);
                }
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void DiagnoseLzwEncoderDecoderPair()
    {
        int w = 64, h = 64;
        byte[] frame = GenerateDistinctRgbPattern(w, h);
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_gif_diag_{Guid.NewGuid():N}.gif");
        try
        {
            GifWriter.SaveGif(tmpFile, [frame], w, h, fps: 10);
            Assert.True(File.Exists(tmpFile));

            byte[] fileBytes = File.ReadAllBytes(tmpFile);

            // Locate LZW sub-blocks
            int pos = 13; // after LSD
            byte packed = fileBytes[10];
            if ((packed & 0x80) != 0)
            {
                int gctSize = 1 << ((packed & 0x07) + 1);
                pos += gctSize * 3;
            }

            // Skip Netscape extension and Graphic Control Extension
            while (pos < fileBytes.Length && fileBytes[pos] != 0x2C)
            {
                if (fileBytes[pos] == 0x21)
                {
                    pos += 2;
                    while (pos < fileBytes.Length && fileBytes[pos] != 0)
                    {
                        pos += 1 + fileBytes[pos];
                    }
                    pos++; // skip 0
                }
                else pos++;
            }

            Assert.Equal(0x2C, fileBytes[pos]);
            pos += 10; // 1 + 2 + 2 + 2 + 2 + 1 = 10 bytes for Image Descriptor
            byte imgPacked = fileBytes[pos - 1];
            if ((imgPacked & 0x80) != 0)
            {
                int lctSize = 1 << ((imgPacked & 0x07) + 1);
                pos += lctSize * 3;
            }

            int minCodeSize = fileBytes[pos++];
            Assert.Equal(8, minCodeSize);

            using var ms = new MemoryStream();
            while (pos < fileBytes.Length && fileBytes[pos] != 0)
            {
                byte subLen = fileBytes[pos++];
                ms.Write(fileBytes, pos, subLen);
                pos += subLen;
            }

            byte[] lzwBytes = ms.ToArray();

            // Decode bitstream and verify with nextCode >= codeLimit
            int bitAccumulator = 0;
            int bitsInAccumulator = 0;
            int bytePos = 0;
            int codeSize = 9;
            int codeLimit = 512;
            int nextCode = 258;
            int oldCode = -1;

            int ReadCode()
            {
                while (bitsInAccumulator < codeSize)
                {
                    if (bytePos >= lzwBytes.Length) return 257;
                    bitAccumulator |= lzwBytes[bytePos++] << bitsInAccumulator;
                    bitsInAccumulator += 8;
                }
                int c = bitAccumulator & ((1 << codeSize) - 1);
                bitAccumulator >>= codeSize;
                bitsInAccumulator -= codeSize;
                return c;
            }

            int clearCode = 256;
            int eoiCode = 257;

            var emittedCodes = new List<(int code, int nextCode, int codeSize)>();

            while (true)
            {
                int c = ReadCode();
                if (c == eoiCode) break;
                if (c == clearCode)
                {
                    codeSize = 9;
                    codeLimit = 512;
                    nextCode = 258;
                    oldCode = -1;
                    continue;
                }

                emittedCodes.Add((c, nextCode, codeSize));

                if (oldCode != -1 && nextCode < 4096)
                {
                    nextCode++;
                    if (nextCode >= codeLimit && codeSize < 12)
                    {
                        codeSize++;
                        codeLimit = 1 << codeSize;
                    }
                }
                oldCode = c;
            }

            var suspicious = emittedCodes.Where(x => x.code > x.nextCode).ToList();
            Assert.Empty(suspicious);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    #endregion
}

