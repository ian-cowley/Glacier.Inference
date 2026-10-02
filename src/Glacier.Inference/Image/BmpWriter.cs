namespace Glacier.Inference.Image;

using System;
using System.IO;

/// <summary>
/// Pure C# uncompressed 24-bit Windows Bitmap (BMP) writer.
/// Generates standard image files directly from RGB/BGR byte spans with zero dependencies.
/// </summary>
public static class BmpWriter
{
    public static void SaveBmp24(string filePath, ReadOnlySpan<byte> rgbPixels, int width, int height)
    {
        int rowStride = (width * 3 + 3) & ~3; // Align rows to 4-byte boundaries
        int imageSize = rowStride * height;
        int fileSize = 54 + imageSize;

        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        // 1. BMP File Header (14 bytes)
        bw.Write((byte)'B');
        bw.Write((byte)'M');
        bw.Write(fileSize);
        bw.Write((short)0); // Reserved 1
        bw.Write((short)0); // Reserved 2
        bw.Write(54);       // Pixel data offset

        // 2. DIB Header (BITMAPINFOHEADER, 40 bytes)
        bw.Write(40);        // Header size
        bw.Write(width);     // Image width
        bw.Write(-height);   // Negative height for top-to-bottom pixel order
        bw.Write((short)1);  // Color planes
        bw.Write((short)24); // Bits per pixel (24-bit RGB)
        bw.Write(0);         // Compression (0 = BI_RGB uncompressed)
        bw.Write(imageSize); // Image byte size
        bw.Write(2835);      // Horizontal resolution (pixels/meter, ~72 DPI)
        bw.Write(2835);      // Vertical resolution
        bw.Write(0);         // Colors in palette
        bw.Write(0);         // Important colors

        // 3. Pixel Data (RGB -> BGR with 4-byte row padding)
        byte[] rowBuffer = new byte[rowStride];

        for (int y = 0; y < height; y++)
        {
            int srcRowOffset = y * width * 3;
            for (int x = 0; x < width; x++)
            {
                int srcIdx = srcRowOffset + x * 3;
                int dstIdx = x * 3;

                byte r = rgbPixels[srcIdx];
                byte g = rgbPixels[srcIdx + 1];
                byte b = rgbPixels[srcIdx + 2];

                rowBuffer[dstIdx] = b;     // Blue
                rowBuffer[dstIdx + 1] = g; // Green
                rowBuffer[dstIdx + 2] = r; // Red
            }

            // Pad remainder of row with 0s
            for (int p = width * 3; p < rowStride; p++)
            {
                rowBuffer[p] = 0;
            }

            bw.Write(rowBuffer);
        }
    }
}
