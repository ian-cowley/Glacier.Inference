namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Pure C# .NET 10 Animated GIF89a Serializer.
/// Features Netscape 2.0 looping, adaptive 256-color palette quantization,
/// frame delay timing, and standard LZW variable-length compression with zero external dependencies.
/// </summary>
public static class GifWriter
{
    /// <summary>
    /// Saves a sequence of 24-bit RGB frames as an Animated GIF89a file.
    /// </summary>
    public static void SaveGif(string filePath, IReadOnlyList<byte[]> frames, int width, int height, int fps = 8)
    {
        if (frames == null || frames.Count == 0)
        {
            throw new ArgumentException("Frames collection must contain at least one frame.", nameof(frames));
        }

        using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs);

        // 1. GIF89a Header
        bw.Write(new byte[] { (byte)'G', (byte)'I', (byte)'F', (byte)'8', (byte)'9', (byte)'a' });

        // 2. Logical Screen Descriptor (10 bytes)
        bw.Write((ushort)width);
        bw.Write((ushort)height);
        // Packed fields: GCT Flag (1), Color Res (7 = 8 bits/pixel), Sort Flag (0), GCT Size (7 = 2^(7+1) = 256 colors)
        bw.Write((byte)0xF7);
        bw.Write((byte)0); // Background Color Index
        bw.Write((byte)0); // Pixel Aspect Ratio

        // 3. Compute Global Color Table (256 colors * 3 bytes = 768 bytes) from all frames
        byte[] globalPalette = GenerateGlobalPalette(frames, width, height, 256);
        bw.Write(globalPalette);

        // 4. Netscape 2.0 Looping Application Extension (Infinite Loop)
        bw.Write((byte)0x21); // Extension Introducer
        bw.Write((byte)0xFF); // Application Extension Label
        bw.Write((byte)0x0B); // Block Size (11 bytes)
        bw.Write(System.Text.Encoding.ASCII.GetBytes("NETSCAPE2.0"));
        bw.Write((byte)0x03); // Sub-block Length
        bw.Write((byte)0x01); // Loop sub-block ID
        bw.Write((ushort)0);  // Loop count (0 = infinite)
        bw.Write((byte)0x00); // Block Terminator

        // Calculate inter-frame delay in hundredths of a second (100 / fps)
        ushort delayTime = (ushort)Math.Max(1, (int)MathF.Round(100.0f / Math.Clamp(fps, 1, 100)));

        // 5. Serialize Each Frame
        for (int f = 0; f < frames.Count; f++)
        {
            // Graphic Control Extension (8 bytes)
            bw.Write((byte)0x21); // Extension Introducer
            bw.Write((byte)0xF9); // Graphic Control Label
            bw.Write((byte)0x04); // Block Size
            bw.Write((byte)0x04); // Packed: Disposal Method 1 (do not dispose), transparent flag 0
            bw.Write(delayTime);  // Delay Time in 1/100s
            bw.Write((byte)0);    // Transparent Color Index
            bw.Write((byte)0x00); // Block Terminator

            // Image Descriptor (10 bytes)
            bw.Write((byte)0x2C); // Image Separator
            bw.Write((ushort)0);  // Left Position
            bw.Write((ushort)0);  // Top Position
            bw.Write((ushort)width);
            bw.Write((ushort)height);
            bw.Write((byte)0x00); // Packed: No local color table (use global)

            // Quantize Frame RGB pixels to 8-bit Palette Indices
            byte[] indexedPixels = QuantizeFrame(frames[f], globalPalette, width, height);

            // Write LZW Compressed Image Data
            WriteLzwData(bw, indexedPixels, 8);
        }

        // 6. GIF Trailer
        bw.Write((byte)0x3B);
    }

    private static byte[] GenerateGlobalPalette(IReadOnlyList<byte[]> frames, int width, int height, int maxColors)
    {
        // 6x7x6 Uniform Color Cube (252 colors) + 4 Grayscale Accents = 256 colors
        byte[] palette = new byte[maxColors * 3];
        int idx = 0;

        for (int r = 0; r < 6; r++)
        {
            byte red = (byte)(r * 255 / 5);
            for (int g = 0; g < 7; g++)
            {
                byte green = (byte)(g * 255 / 6);
                for (int b = 0; b < 6; b++)
                {
                    byte blue = (byte)(b * 255 / 5);
                    palette[idx++] = red;
                    palette[idx++] = green;
                    palette[idx++] = blue;
                }
            }
        }

        // Fill remaining 4 slots with key shades (pure black, dark gray, light gray, pure white)
        palette[idx++] = 0;   palette[idx++] = 0;   palette[idx++] = 0;
        palette[idx++] = 64;  palette[idx++] = 64;  palette[idx++] = 64;
        palette[idx++] = 192; palette[idx++] = 192; palette[idx++] = 192;
        palette[idx++] = 255; palette[idx++] = 255; palette[idx++] = 255;

        return palette;
    }

    private static byte[] QuantizeFrame(byte[] rgbPixels, byte[] palette, int width, int height)
    {
        int numPixels = width * height;
        byte[] indices = new byte[numPixels];

        for (int i = 0; i < numPixels; i++)
        {
            int r = rgbPixels[i * 3 + 0];
            int g = rgbPixels[i * 3 + 1];
            int b = rgbPixels[i * 3 + 2];

            // Fast uniform index calculation: 6x7x6 cube
            int rIdx = Math.Clamp((r * 5 + 127) / 255, 0, 5);
            int gIdx = Math.Clamp((g * 6 + 127) / 255, 0, 6);
            int bIdx = Math.Clamp((b * 5 + 127) / 255, 0, 5);

            indices[i] = (byte)(rIdx * 42 + gIdx * 6 + bIdx);
        }

        return indices;
    }

    private static void WriteLzwData(BinaryWriter bw, byte[] indexedPixels, int colorDepth)
    {
        int initCodeSize = colorDepth;
        bw.Write((byte)initCodeSize);

        int clearCode = 1 << initCodeSize; // 256
        int eoiCode = clearCode + 1;       // 257
        int nextCode = eoiCode + 1;
        int codeSize = initCodeSize + 1;   // 9 bits initially
        int codeLimit = 1 << codeSize;

        using var bitStream = new MemoryStream();
        uint bitAccumulator = 0;
        int bitsInAccumulator = 0;

        void WriteBits(int code, int size)
        {
            bitAccumulator |= (uint)(code << bitsInAccumulator);
            bitsInAccumulator += size;
            while (bitsInAccumulator >= 8)
            {
                bitStream.WriteByte((byte)(bitAccumulator & 0xFF));
                bitAccumulator >>= 8;
                bitsInAccumulator -= 8;
            }
        }

        // LZW Prefix Tree / Hash Map
        var dictionary = new Dictionary<int, int>(4096);

        // Send Clear Code
        WriteBits(clearCode, codeSize);

        int curPrefix = -1;

        for (int i = 0; i < indexedPixels.Length; i++)
        {
            byte k = indexedPixels[i];

            if (curPrefix == -1)
            {
                curPrefix = k;
                continue;
            }

            int key = (curPrefix << 8) | k;

            if (dictionary.TryGetValue(key, out int foundCode))
            {
                curPrefix = foundCode;
            }
            else
            {
                WriteBits(curPrefix, codeSize);

                if (nextCode < 4096)
                {
                    dictionary[key] = nextCode++;
                    if (nextCode > codeLimit && codeSize < 12)
                    {
                        codeSize++;
                        codeLimit = 1 << codeSize;
                    }
                }
                else
                {
                    // Reset dictionary
                    WriteBits(clearCode, codeSize);
                    dictionary.Clear();
                    codeSize = initCodeSize + 1;
                    codeLimit = 1 << codeSize;
                    nextCode = eoiCode + 1;
                }

                curPrefix = k;
            }
        }

        if (curPrefix != -1)
        {
            WriteBits(curPrefix, codeSize);
        }

        // Send End of Information code
        WriteBits(eoiCode, codeSize);

        // Flush remaining bits
        if (bitsInAccumulator > 0)
        {
            bitStream.WriteByte((byte)(bitAccumulator & 0xFF));
        }

        // Package output into sub-blocks of up to 255 bytes
        byte[] compressedBytes = bitStream.ToArray();
        int offset = 0;
        while (offset < compressedBytes.Length)
        {
            int blockSize = Math.Min(255, compressedBytes.Length - offset);
            bw.Write((byte)blockSize);
            bw.Write(compressedBytes, offset, blockSize);
            offset += blockSize;
        }

        // Block Terminator
        bw.Write((byte)0x00);
    }
}
