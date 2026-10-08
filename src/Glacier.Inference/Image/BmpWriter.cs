namespace Glacier.Inference.Image;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.IO;
using Glacier.Inference.Media;

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

        // 1. BMP File Header (14 bytes) + DIB Header (40 bytes) emitted via stackalloc buffer
        Span<byte> header = stackalloc byte[54];
        header[0] = (byte)'B';
        header[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(2, 4), fileSize);
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(6, 2), 0); // Reserved 1
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(8, 2), 0); // Reserved 2
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(10, 4), 54); // Pixel data offset

        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(14, 4), 40); // Header size
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(18, 4), width);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(22, 4), height); // Positive height for bottom-up DIB
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(26, 2), 1);  // Color planes
        BinaryPrimitives.WriteInt16LittleEndian(header.Slice(28, 2), 24); // Bits per pixel (24-bit RGB)
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(30, 4), 0);  // Compression (0 = BI_RGB uncompressed)
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(34, 4), imageSize);
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(38, 4), 2835); // Pixels/meter horizontal (~72 DPI)
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(42, 4), 2835); // Pixels/meter vertical
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(46, 4), 0); // Colors in palette
        BinaryPrimitives.WriteInt32LittleEndian(header.Slice(50, 4), 0); // Important colors

        fs.Write(header);

        // 2. Pixel Data (RGB -> BGR with 4-byte row padding, bottom-to-top scanlines)
        byte[] rented = ArrayPool<byte>.Shared.Rent(imageSize);
        try
        {
            Span<byte> dstFrame = rented.AsSpan(0, imageSize);
            MediaKernels.ConvertFrameBottomUp(rgbPixels, dstFrame, width, height, rowStride);
            fs.Write(dstFrame);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }
}
