namespace Glacier.Inference.Media;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// Hardware-accelerated SIMD media processing and pixel format conversion kernels.
/// Provides vectorized 24-bit RGB to BGR transposition and bottom-up scanline operations.
/// </summary>
public static class MediaKernels
{
    // Shuffles 5 pixels (15 bytes): R0,G0,B0, R1,G1,B1, R2,G2,B2, R3,G3,B3, R4,G4,B4
    // into: B0,G0,R0, B1,G1,R1, B2,G2,R2, B3,G3,B3, B4,G4,R4
    // Trailing 16th byte is untouched and overwritten on the next 15-byte iteration.
    private static readonly Vector128<byte> RgbToBgrShuffleMask15 = Vector128.Create(
        (byte)2, (byte)1, (byte)0,
        (byte)5, (byte)4, (byte)3,
        (byte)8, (byte)7, (byte)6,
        (byte)11, (byte)10, (byte)9,
        (byte)14, (byte)13, (byte)12,
        (byte)15
    );

    /// <summary>
    /// Converts a single scanline from 24-bit RGB to 24-bit BGR with SIMD acceleration (15 bytes/step)
    /// and 4-byte row padding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ConvertRgbToBgrRow(ReadOnlySpan<byte> srcRow, Span<byte> dstRow, int width, int rowStride)
    {
        int rowBytes = width * 3;
        int x = 0;

        if (Vector128.IsHardwareAccelerated && rowBytes >= 16)
        {
            // Process 5 pixels (15 bytes) per SIMD step
            int limit = rowBytes - 16;
            while (x <= limit)
            {
                var v = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(srcRow.Slice(x)));
                var shuffled = Vector128.Shuffle(v, RgbToBgrShuffleMask15);
                shuffled.StoreUnsafe(ref MemoryMarshal.GetReference(dstRow.Slice(x)));
                x += 15;
            }
        }

        // Process remaining pixels in scalar fallback
        int pixelIndex = x / 3;
        while (pixelIndex < width)
        {
            int srcIdx = pixelIndex * 3;
            byte r = srcRow[srcIdx + 0];
            byte g = srcRow[srcIdx + 1];
            byte b = srcRow[srcIdx + 2];

            dstRow[srcIdx + 0] = b;
            dstRow[srcIdx + 1] = g;
            dstRow[srcIdx + 2] = r;
            pixelIndex++;
        }

        // Zero-pad stride to 4-byte boundary
        for (int p = rowBytes; p < rowStride; p++)
        {
            dstRow[p] = 0;
        }
    }

    /// <summary>
    /// Converts a full frame from top-down RGB to bottom-up BGR scanlines in dstBgrFrame.
    /// </summary>
    public static void ConvertFrameBottomUp(
        ReadOnlySpan<byte> srcRgb,
        Span<byte> dstBgrFrame,
        int width,
        int height,
        int rowStride)
    {
        int srcRowStride = width * 3;
        for (int y = 0; y < height; y++)
        {
            int srcY = height - 1 - y; // Bottom-up
            ReadOnlySpan<byte> srcRow = srcRgb.Slice(srcY * srcRowStride, srcRowStride);
            Span<byte> dstRow = dstBgrFrame.Slice(y * rowStride, rowStride);
            ConvertRgbToBgrRow(srcRow, dstRow, width, rowStride);
        }
    }
}
