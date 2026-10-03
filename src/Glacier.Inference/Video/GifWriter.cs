namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Pure C# .NET 10 Animated GIF89a Serializer.
/// Features Netscape 2.0 looping, adaptive Median-Cut 256-color palette quantization,
/// Floyd-Steinberg error diffusion dithering, and standard LZW variable-length compression.
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

        // 3. Compute Adaptive Median-Cut 256-Color Global Palette from video frames
        byte[] globalPalette = GenerateAdaptivePalette(frames, 256);
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

        // Pre-build 15-bit color lookup table for fast Floyd-Steinberg dithering
        var colorLookup = new byte[32768];
        Array.Fill(colorLookup, (byte)0xFF);

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

            // Quantize Frame RGB pixels with Floyd-Steinberg Error Diffusion Dithering
            byte[] indexedPixels = QuantizeFrameDithered(frames[f], globalPalette, width, height, colorLookup);

            // Write LZW Compressed Image Data
            WriteLzwData(bw, indexedPixels, 8);
        }

        // 6. GIF Trailer
        bw.Write((byte)0x3B);
    }

    /// <summary>
    /// Generates an adaptive 256-color palette tailored to the exact color distribution of the frames using Median-Cut.
    /// </summary>
    private static byte[] GenerateAdaptivePalette(IReadOnlyList<byte[]> frames, int maxColors = 256)
    {
        // 1. Sample up to 16,384 representative RGB pixels across frames
        const int maxSamples = 16384;
        var samples = new List<(byte r, byte g, byte b)>(maxSamples);
        int totalPixelsAcrossFrames = frames.Count * (frames[0].Length / 3);
        int stride = Math.Max(1, totalPixelsAcrossFrames / maxSamples);

        int sampleCounter = 0;
        foreach (var frame in frames)
        {
            int numPixels = frame.Length / 3;
            for (int i = 0; i < numPixels; i++)
            {
                if ((sampleCounter++ % stride) == 0 && samples.Count < maxSamples)
                {
                    samples.Add((frame[i * 3 + 0], frame[i * 3 + 1], frame[i * 3 + 2]));
                }
            }
        }

        if (samples.Count == 0)
        {
            samples.Add((0, 0, 0));
        }

        // 2. Median-Cut Bounding Box Splitting
        var boxes = new List<ColorBox>(maxColors)
        {
            new ColorBox(0, samples.Count, samples)
        };

        while (boxes.Count < maxColors)
        {
            // Find box with greatest range along its longest axis
            int bestBoxIdx = -1;
            int maxRange = -1;

            for (int i = 0; i < boxes.Count; i++)
            {
                if (boxes[i].Count > 1 && boxes[i].MaxSpread > maxRange)
                {
                    maxRange = boxes[i].MaxSpread;
                    bestBoxIdx = i;
                }
            }

            if (bestBoxIdx == -1 || maxRange <= 0) break; // Cannot split further

            var boxToSplit = boxes[bestBoxIdx];
            int axis = boxToSplit.LongestAxis; // 0=R, 1=G, 2=B

            // Sort samples within this box along the longest axis
            if (axis == 0)
                samples.Sort(boxToSplit.Start, boxToSplit.Count, Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.r.CompareTo(b.r)));
            else if (axis == 1)
                samples.Sort(boxToSplit.Start, boxToSplit.Count, Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.g.CompareTo(b.g)));
            else
                samples.Sort(boxToSplit.Start, boxToSplit.Count, Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.b.CompareTo(b.b)));

            // Split at median
            int half = boxToSplit.Count / 2;
            var boxA = new ColorBox(boxToSplit.Start, half, samples);
            var boxB = new ColorBox(boxToSplit.Start + half, boxToSplit.Count - half, samples);

            boxes[bestBoxIdx] = boxA;
            boxes.Add(boxB);
        }

        // 3. Compute centroid color for each box
        byte[] palette = new byte[maxColors * 3];
        for (int i = 0; i < boxes.Count && i < maxColors; i++)
        {
            var (avgR, avgG, avgB) = boxes[i].GetAverageColor(samples);
            palette[i * 3 + 0] = avgR;
            palette[i * 3 + 1] = avgG;
            palette[i * 3 + 2] = avgB;
        }

        // Pad remaining slots if image has fewer unique colors
        for (int i = boxes.Count; i < maxColors; i++)
        {
            byte shade = (byte)((i - boxes.Count) * 255 / Math.Max(1, maxColors - boxes.Count));
            palette[i * 3 + 0] = shade;
            palette[i * 3 + 1] = shade;
            palette[i * 3 + 2] = shade;
        }

        return palette;
    }

    private struct ColorBox
    {
        public int Start;
        public int Count;
        public byte MinR, MaxR;
        public byte MinG, MaxG;
        public byte MinB, MaxB;

        public ColorBox(int start, int count, List<(byte r, byte g, byte b)> samples)
        {
            Start = start;
            Count = count;
            MinR = MinG = MinB = 255;
            MaxR = MaxG = MaxB = 0;

            for (int i = start; i < start + count; i++)
            {
                var s = samples[i];
                if (s.r < MinR) MinR = s.r;
                if (s.r > MaxR) MaxR = s.r;
                if (s.g < MinG) MinG = s.g;
                if (s.g > MaxG) MaxG = s.g;
                if (s.b < MinB) MinB = s.b;
                if (s.b > MaxB) MaxB = s.b;
            }
        }

        public int MaxSpread => Math.Max(MaxR - MinR, Math.Max(MaxG - MinG, MaxB - MinB));

        public int LongestAxis
        {
            get
            {
                int rRange = MaxR - MinR;
                int gRange = MaxG - MinG;
                int bRange = MaxB - MinB;
                if (rRange >= gRange && rRange >= bRange) return 0;
                if (gRange >= rRange && gRange >= bRange) return 1;
                return 2;
            }
        }

        public (byte r, byte g, byte b) GetAverageColor(List<(byte r, byte g, byte b)> samples)
        {
            if (Count == 0) return (0, 0, 0);
            long sumR = 0, sumG = 0, sumB = 0;
            for (int i = Start; i < Start + Count; i++)
            {
                sumR += samples[i].r;
                sumG += samples[i].g;
                sumB += samples[i].b;
            }
            return ((byte)(sumR / Count), (byte)(sumG / Count), (byte)(sumB / Count));
        }
    }

    /// <summary>
    /// Quantizes a 24-bit RGB frame to 8-bit palette indices with Floyd-Steinberg error diffusion dithering.
    /// Eliminates color banding, posterization, and harsh transitions.
    /// </summary>
    private static byte[] QuantizeFrameDithered(
        byte[] rgbPixels,
        byte[] palette,
        int width,
        int height,
        byte[] colorLookup)
    {
        int numPixels = width * height;
        byte[] indices = new byte[numPixels];

        // Floyd-Steinberg error diffusion row buffers (current scanline and next scanline)
        float[] errCurr = new float[(width + 2) * 3];
        float[] errNext = new float[(width + 2) * 3];

        for (int y = 0; y < height; y++)
        {
            Array.Clear(errNext, 0, errNext.Length);
            int rowOffset = y * width * 3;

            for (int x = 0; x < width; x++)
            {
                int srcIdx = rowOffset + x * 3;
                int errIdx = (x + 1) * 3;

                // Add diffused error to incoming pixel
                float rF = Math.Clamp(rgbPixels[srcIdx + 0] + errCurr[errIdx + 0], 0f, 255f);
                float gF = Math.Clamp(rgbPixels[srcIdx + 1] + errCurr[errIdx + 1], 0f, 255f);
                float bF = Math.Clamp(rgbPixels[srcIdx + 2] + errCurr[errIdx + 2], 0f, 255f);

                byte r = (byte)MathF.Round(rF);
                byte g = (byte)MathF.Round(gF);
                byte b = (byte)MathF.Round(bF);

                // Fast nearest color lookup using 15-bit cache (32x32x32)
                int key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
                byte palIdx = colorLookup[key];

                if (palIdx == 0xFF)
                {
                    // Compute nearest color by squared Euclidean distance
                    int bestIdx = 0;
                    int bestDist = int.MaxValue;

                    for (int p = 0; p < 256; p++)
                    {
                        int dr = r - palette[p * 3 + 0];
                        int dg = g - palette[p * 3 + 1];
                        int db = b - palette[p * 3 + 2];
                        int dist = dr * dr + dg * dg + db * db;
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = p;
                            if (dist == 0) break;
                        }
                    }

                    palIdx = (byte)bestIdx;
                    colorLookup[key] = palIdx;
                }

                indices[y * width + x] = palIdx;

                // Quantization error
                float errR = rF - palette[palIdx * 3 + 0];
                float errG = gF - palette[palIdx * 3 + 1];
                float errB = bF - palette[palIdx * 3 + 2];

                // Diffuse error via Floyd-Steinberg kernel:
                // (x+1, y): 7/16
                errCurr[(x + 2) * 3 + 0] += errR * 0.4375f;
                errCurr[(x + 2) * 3 + 1] += errG * 0.4375f;
                errCurr[(x + 2) * 3 + 2] += errB * 0.4375f;

                // (x-1, y+1): 3/16
                errNext[x * 3 + 0] += errR * 0.1875f;
                errNext[x * 3 + 1] += errG * 0.1875f;
                errNext[x * 3 + 2] += errB * 0.1875f;

                // (x, y+1): 5/16
                errNext[(x + 1) * 3 + 0] += errR * 0.3125f;
                errNext[(x + 1) * 3 + 1] += errG * 0.3125f;
                errNext[(x + 1) * 3 + 2] += errB * 0.3125f;

                // (x+1, y+1): 1/16
                errNext[(x + 2) * 3 + 0] += errR * 0.0625f;
                errNext[(x + 2) * 3 + 1] += errG * 0.0625f;
                errNext[(x + 2) * 3 + 2] += errB * 0.0625f;
            }

            // Swap error buffers for next line
            (errCurr, errNext) = (errNext, errCurr);
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
