namespace Glacier.Inference.Image;

using System;
using System.IO;
using System.IO.Compression;

/// <summary>
/// Pure C# .NET 10 zero-dependency image decoder for 24-bit/32-bit PNG and standard BMP images.
/// </summary>
public static class ImageDecoder
{
    /// <summary>
    /// Loads and decodes an image file into 24-bit uncompressed RGB pixels [width * height * 3].
    /// </summary>
    public static (byte[] rgbPixels, int width, int height) Load(string filePath)
    {
        byte[] fileBytes = File.ReadAllBytes(filePath);

        if (fileBytes.Length >= 2 && fileBytes[0] == 'B' && fileBytes[1] == 'M')
        {
            return DecodeBmp(fileBytes);
        }
        else if (fileBytes.Length >= 8 && fileBytes[0] == 0x89 && fileBytes[1] == 0x50 && fileBytes[2] == 0x4E && fileBytes[3] == 0x47)
        {
            return DecodePng(fileBytes);
        }

        throw new NotSupportedException($"Unsupported image format: {Path.GetExtension(filePath)}. Supported: PNG, BMP.");
    }

    /// <summary>
    /// Decodes a W3C-standard 24-bit or 32-bit PNG image with ZLib IDAT decompression and Paeth filter reconstruction.
    /// </summary>
    public static (byte[] rgbPixels, int width, int height) DecodePng(ReadOnlySpan<byte> pngBytes)
    {
        if (pngBytes.Length < 33 || pngBytes[0] != 0x89 || pngBytes[1] != 0x50)
        {
            throw new InvalidDataException("Invalid PNG signature.");
        }

        int width = (pngBytes[16] << 24) | (pngBytes[17] << 16) | (pngBytes[18] << 8) | pngBytes[19];
        int height = (pngBytes[20] << 24) | (pngBytes[21] << 16) | (pngBytes[22] << 8) | pngBytes[23];
        byte bitDepth = pngBytes[24];
        byte colorType = pngBytes[25]; // 2 = RGB (3 bpp), 6 = RGBA (4 bpp), 0 = Grayscale (1 bpp)

        if (bitDepth != 8)
        {
            throw new NotSupportedException($"Only 8-bit PNG bit depth is supported (found {bitDepth}).");
        }

        int bytesPerPixel = colorType switch
        {
            2 => 3, // RGB
            6 => 4, // RGBA
            0 => 1, // Grayscale
            _ => throw new NotSupportedException($"PNG color type {colorType} not supported.")
        };

        // Concatenate all IDAT compressed chunks
        using var compressedMs = new MemoryStream();
        int offset = 8;
        while (offset + 8 <= pngBytes.Length)
        {
            int chunkLen = (pngBytes[offset] << 24) | (pngBytes[offset + 1] << 16) | (pngBytes[offset + 2] << 8) | pngBytes[offset + 3];
            string chunkType = System.Text.Encoding.ASCII.GetString(pngBytes.Slice(offset + 4, 4));
            offset += 8;

            if (chunkType == "IDAT")
            {
                compressedMs.Write(pngBytes.Slice(offset, chunkLen));
            }
            else if (chunkType == "IEND")
            {
                break;
            }

            offset += chunkLen + 4; // Skip data + 4 bytes CRC
        }

        compressedMs.Position = 0;

        // Decompress raw scanlines
        int rowBytes = width * bytesPerPixel;
        byte[] decompressed = new byte[height * (1 + rowBytes)];
        using (var zlib = new ZLibStream(compressedMs, CompressionMode.Decompress))
        {
            int read = 0;
            while (read < decompressed.Length)
            {
                int n = zlib.Read(decompressed, read, decompressed.Length - read);
                if (n == 0) break;
                read += n;
            }
        }

        // Apply reverse PNG scanline filters (None, Sub, Up, Average, Paeth)
        byte[] uncompressedScanlines = new byte[height * rowBytes];
        byte[] prevRow = new byte[rowBytes];
        byte[] currRow = new byte[rowBytes];

        for (int y = 0; y < height; y++)
        {
            int srcOffset = y * (1 + rowBytes);
            byte filterType = decompressed[srcOffset];
            ReadOnlySpan<byte> rawRow = decompressed.AsSpan(srcOffset + 1, rowBytes);

            for (int i = 0; i < rowBytes; i++)
            {
                byte raw = rawRow[i];
                byte a = (i >= bytesPerPixel) ? currRow[i - bytesPerPixel] : (byte)0; // Left
                byte b = prevRow[i];                                                  // Up
                byte c = (i >= bytesPerPixel) ? prevRow[i - bytesPerPixel] : (byte)0; // Up-Left

                byte filtered = filterType switch
                {
                    0 => raw,                                     // None
                    1 => (byte)(raw + a),                         // Sub
                    2 => (byte)(raw + b),                         // Up
                    3 => (byte)(raw + (a + b) / 2),               // Average
                    4 => (byte)(raw + PaethPredictor(a, b, c)),   // Paeth
                    _ => raw
                };

                currRow[i] = filtered;
            }

            currRow.CopyTo(uncompressedScanlines.AsSpan(y * rowBytes, rowBytes));
            Array.Copy(currRow, prevRow, rowBytes);
        }

        // Convert to standard 24-bit RGB
        byte[] rgbPixels = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int srcIdx = y * rowBytes + x * bytesPerPixel;
                int dstIdx = (y * width + x) * 3;

                if (colorType == 6) // RGBA -> RGB
                {
                    rgbPixels[dstIdx] = uncompressedScanlines[srcIdx];
                    rgbPixels[dstIdx + 1] = uncompressedScanlines[srcIdx + 1];
                    rgbPixels[dstIdx + 2] = uncompressedScanlines[srcIdx + 2];
                }
                else if (colorType == 2) // RGB -> RGB
                {
                    rgbPixels[dstIdx] = uncompressedScanlines[srcIdx];
                    rgbPixels[dstIdx + 1] = uncompressedScanlines[srcIdx + 1];
                    rgbPixels[dstIdx + 2] = uncompressedScanlines[srcIdx + 2];
                }
                else if (colorType == 0) // Grayscale -> RGB
                {
                    byte gray = uncompressedScanlines[srcIdx];
                    rgbPixels[dstIdx] = gray;
                    rgbPixels[dstIdx + 1] = gray;
                    rgbPixels[dstIdx + 2] = gray;
                }
            }
        }

        return (rgbPixels, width, height);
    }

    /// <summary>
    /// Decodes a W3C-standard 32-bit RGBA PNG image with transparency.
    /// </summary>
    public static (byte[] rgbaPixels, int width, int height) DecodePngRgba(ReadOnlySpan<byte> pngBytes)
    {
        if (pngBytes.Length < 33 || pngBytes[0] != 0x89 || pngBytes[1] != 0x50)
        {
            throw new InvalidDataException("Invalid PNG signature.");
        }

        int width = (pngBytes[16] << 24) | (pngBytes[17] << 16) | (pngBytes[18] << 8) | pngBytes[19];
        int height = (pngBytes[20] << 24) | (pngBytes[21] << 16) | (pngBytes[22] << 8) | pngBytes[23];
        byte bitDepth = pngBytes[24];
        byte colorType = pngBytes[25];

        if (bitDepth != 8 || (colorType != 6 && colorType != 2))
        {
            throw new NotSupportedException($"Unsupported PNG bitDepth={bitDepth}, colorType={colorType}.");
        }

        int bytesPerPixel = (colorType == 6) ? 4 : 3;

        // Collect all IDAT chunk payloads
        using var compressedMs = new MemoryStream();
        int offset = 8;
        while (offset + 8 <= pngBytes.Length)
        {
            int chunkLen = (pngBytes[offset] << 24) | (pngBytes[offset + 1] << 16) | (pngBytes[offset + 2] << 8) | pngBytes[offset + 3];
            string chunkType = System.Text.Encoding.ASCII.GetString(pngBytes.Slice(offset + 4, 4));

            if (chunkType == "IDAT")
            {
                compressedMs.Write(pngBytes.Slice(offset + 8, chunkLen));
            }
            else if (chunkType == "IEND")
            {
                break;
            }

            offset += 12 + chunkLen;
        }

        compressedMs.Position = 0;

        // Decompress raw scanlines
        int rowBytes = width * bytesPerPixel;
        byte[] decompressed = new byte[height * (1 + rowBytes)];
        using (var zlib = new ZLibStream(compressedMs, CompressionMode.Decompress))
        {
            int read = 0;
            while (read < decompressed.Length)
            {
                int n = zlib.Read(decompressed, read, decompressed.Length - read);
                if (n == 0) break;
                read += n;
            }
        }

        byte[] uncompressedScanlines = new byte[height * rowBytes];
        byte[] prevRow = new byte[rowBytes];
        byte[] currRow = new byte[rowBytes];

        for (int y = 0; y < height; y++)
        {
            int srcOffset = y * (1 + rowBytes);
            byte filterType = decompressed[srcOffset];
            ReadOnlySpan<byte> rawRow = decompressed.AsSpan(srcOffset + 1, rowBytes);

            for (int i = 0; i < rowBytes; i++)
            {
                byte raw = rawRow[i];
                byte a = (i >= bytesPerPixel) ? currRow[i - bytesPerPixel] : (byte)0;
                byte b = prevRow[i];
                byte c = (i >= bytesPerPixel) ? prevRow[i - bytesPerPixel] : (byte)0;

                byte filtered = filterType switch
                {
                    0 => raw,
                    1 => (byte)(raw + a),
                    2 => (byte)(raw + b),
                    3 => (byte)(raw + (a + b) / 2),
                    4 => (byte)(raw + PaethPredictor(a, b, c)),
                    _ => raw
                };

                currRow[i] = filtered;
            }

            currRow.CopyTo(uncompressedScanlines.AsSpan(y * rowBytes, rowBytes));
            Array.Copy(currRow, prevRow, rowBytes);
        }

        byte[] rgbaPixels = new byte[width * height * 4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int srcIdx = y * rowBytes + x * bytesPerPixel;
                int dstIdx = (y * width + x) * 4;

                if (colorType == 6)
                {
                    rgbaPixels[dstIdx] = uncompressedScanlines[srcIdx];
                    rgbaPixels[dstIdx + 1] = uncompressedScanlines[srcIdx + 1];
                    rgbaPixels[dstIdx + 2] = uncompressedScanlines[srcIdx + 2];
                    rgbaPixels[dstIdx + 3] = uncompressedScanlines[srcIdx + 3];
                }
                else
                {
                    rgbaPixels[dstIdx] = uncompressedScanlines[srcIdx];
                    rgbaPixels[dstIdx + 1] = uncompressedScanlines[srcIdx + 1];
                    rgbaPixels[dstIdx + 2] = uncompressedScanlines[srcIdx + 2];
                    rgbaPixels[dstIdx + 3] = 255;
                }
            }
        }

        return (rgbaPixels, width, height);
    }

    /// <summary>
    /// Decodes an uncompressed standard 24-bit Windows Bitmap (.bmp).
    /// </summary>
    public static (byte[] rgbPixels, int width, int height) DecodeBmp(ReadOnlySpan<byte> bmpBytes)
    {
        if (bmpBytes.Length < 54 || bmpBytes[0] != 'B' || bmpBytes[1] != 'M')
        {
            throw new InvalidDataException("Invalid BMP signature.");
        }

        int pixelOffset = BitConverter.ToInt32(bmpBytes.Slice(10, 4));
        int width = BitConverter.ToInt32(bmpBytes.Slice(18, 4));
        int height = Math.Abs(BitConverter.ToInt32(bmpBytes.Slice(22, 4)));
        short bpp = BitConverter.ToInt16(bmpBytes.Slice(28, 2));

        if (bpp != 24 && bpp != 32)
        {
            throw new NotSupportedException($"Only 24-bit and 32-bit BMP files supported (found {bpp} bpp).");
        }

        int bytesPerPixel = bpp / 8;
        int rowStride = (width * bytesPerPixel + 3) & ~3; // 4-byte aligned scanlines
        byte[] rgbPixels = new byte[width * height * 3];

        for (int y = 0; y < height; y++)
        {
            // Windows BMP is stored bottom-up
            int bmpY = height - 1 - y;
            int srcRow = pixelOffset + bmpY * rowStride;
            int dstRow = y * width * 3;

            for (int x = 0; x < width; x++)
            {
                int srcIdx = srcRow + x * bytesPerPixel;
                int dstIdx = dstRow + x * 3;

                // BMP stores pixels as BGR / BGRA
                byte b = bmpBytes[srcIdx];
                byte g = bmpBytes[srcIdx + 1];
                byte r = bmpBytes[srcIdx + 2];

                rgbPixels[dstIdx] = r;
                rgbPixels[dstIdx + 1] = g;
                rgbPixels[dstIdx + 2] = b;
            }
        }

        return (rgbPixels, width, height);
    }

    private static byte PaethPredictor(byte a, byte b, byte c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a);
        int pb = Math.Abs(p - b);
        int pc = Math.Abs(p - c);

        if (pa <= pb && pa <= pc) return a;
        if (pb <= pc) return b;
        return c;
    }
}
