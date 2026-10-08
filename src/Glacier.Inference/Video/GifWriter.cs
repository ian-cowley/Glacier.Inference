namespace Glacier.Inference.Video;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

/// <summary>
/// Pure C# .NET 10 Animated GIF89a Serializer.
/// Features Netscape 2.0 looping, adaptive Median-Cut 256-color palette quantization,
/// Floyd-Steinberg error diffusion dithering, SIMD 15-bit color table generation,
/// and zero-allocation open-addressing LZW variable-length compression.
/// </summary>
public static class GifWriter
{
    private const int LzwTableSize = 5003;

    private static readonly IComparer<(byte r, byte g, byte b)> SortByR =
        Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.r.CompareTo(b.r));
    private static readonly IComparer<(byte r, byte g, byte b)> SortByG =
        Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.g.CompareTo(b.g));
    private static readonly IComparer<(byte r, byte g, byte b)> SortByB =
        Comparer<(byte r, byte g, byte b)>.Create((a, b) => a.b.CompareTo(b.b));

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
        bw.Write("GIF89a"u8);

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
        bw.Write("NETSCAPE2.0"u8);
        bw.Write((byte)0x03); // Sub-block Length
        bw.Write((byte)0x01); // Loop sub-block ID
        bw.Write((ushort)0);  // Loop count (0 = infinite)
        bw.Write((byte)0x00); // Block Terminator

        // Calculate inter-frame delay in hundredths of a second (100 / fps)
        ushort delayTime = (ushort)Math.Max(1, (int)MathF.Round(100.0f / Math.Clamp(fps, 1, 100)));

        // Transpose 256 RGB colors into contiguous int arrays for SoA SIMD vectorization
        int[] palR = new int[256];
        int[] palG = new int[256];
        int[] palB = new int[256];
        for (int c = 0; c < 256; c++)
        {
            palR[c] = globalPalette[c * 3 + 0];
            palG[c] = globalPalette[c * 3 + 1];
            palB[c] = globalPalette[c * 3 + 2];
        }

        // Pre-build full 15-bit color lookup table for fast lock-free Floyd-Steinberg dithering using SIMD
        var colorLookup = new byte[32768];
        Parallel.For(0, 32768, idx =>
        {
            int r = ((idx >> 10) & 0x1F) * 255 / 31;
            int g = ((idx >> 5) & 0x1F) * 255 / 31;
            int b = (idx & 0x1F) * 255 / 31;

            int bestIdx = 0;
            int bestDist = int.MaxValue;

            if (Vector256.IsHardwareAccelerated)
            {
                var vr = Vector256.Create(r);
                var vg = Vector256.Create(g);
                var vb = Vector256.Create(b);
                var wR = Vector256.Create(3);
                var wG = Vector256.Create(4);
                var wB = Vector256.Create(2);

                for (int c = 0; c < 256; c += 8)
                {
                    var pr = Vector256.LoadUnsafe(ref palR[c]);
                    var pg = Vector256.LoadUnsafe(ref palG[c]);
                    var pb = Vector256.LoadUnsafe(ref palB[c]);

                    var dr = vr - pr;
                    var dg = vg - pg;
                    var db = vb - pb;

                    var dist = dr * dr * wR + dg * dg * wG + db * db * wB;

                    if (Vector256.LessThanAny(dist, Vector256.Create(bestDist)))
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            int d = dist.GetElement(i);
                            if (d < bestDist)
                            {
                                bestDist = d;
                                bestIdx = c + i;
                                if (d == 0) goto Found;
                            }
                        }
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                var vr = Vector128.Create(r);
                var vg = Vector128.Create(g);
                var vb = Vector128.Create(b);
                var wR = Vector128.Create(3);
                var wG = Vector128.Create(4);
                var wB = Vector128.Create(2);

                for (int c = 0; c < 256; c += 4)
                {
                    var pr = Vector128.LoadUnsafe(ref palR[c]);
                    var pg = Vector128.LoadUnsafe(ref palG[c]);
                    var pb = Vector128.LoadUnsafe(ref palB[c]);

                    var dr = vr - pr;
                    var dg = vg - pg;
                    var db = vb - pb;

                    var dist = dr * dr * wR + dg * dg * wG + db * db * wB;

                    if (Vector128.LessThanAny(dist, Vector128.Create(bestDist)))
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            int d = dist.GetElement(i);
                            if (d < bestDist)
                            {
                                bestDist = d;
                                bestIdx = c + i;
                                if (d == 0) goto Found;
                            }
                        }
                    }
                }
            }
            else
            {
                for (int c = 0; c < 256; c++)
                {
                    int dr = r - palR[c];
                    int dg = g - palG[c];
                    int db = b - palB[c];
                    int dist = dr * dr * 3 + dg * dg * 4 + db * db * 2;
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx = c;
                        if (dist == 0) break;
                    }
                }
            }

        Found:
            colorLookup[idx] = (byte)bestIdx;
        });

        // Parallelize frame quantization across all CPU cores
        byte[][] indexedFrames = new byte[frames.Count][];
        Parallel.For(0, frames.Count, f =>
        {
            indexedFrames[f] = QuantizeFrameDithered(frames[f], globalPalette, width, height, colorLookup);
        });

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

            // Write LZW Compressed Image Data
            WriteLzwData(bw, indexedFrames[f], 8);
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
                samples.Sort(boxToSplit.Start, boxToSplit.Count, SortByR);
            else if (axis == 1)
                samples.Sort(boxToSplit.Start, boxToSplit.Count, SortByG);
            else
                samples.Sort(boxToSplit.Start, boxToSplit.Count, SortByB);

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

        var bitWriter = new LzwBitWriter(bw, stackalloc byte[256]);

        int[] hashKeys = ArrayPool<int>.Shared.Rent(LzwTableSize);
        int[] hashCodes = ArrayPool<int>.Shared.Rent(LzwTableSize);
        hashKeys.AsSpan(0, LzwTableSize).Fill(-1);

        try
        {
            // Send Clear Code
            bitWriter.WriteBits(clearCode, codeSize);

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
                int h = (int)((uint)key % LzwTableSize);
                bool found = false;

                while (hashKeys[h] != -1)
                {
                    if (hashKeys[h] == key)
                    {
                        curPrefix = hashCodes[h];
                        found = true;
                        break;
                    }
                    h++;
                    if (h >= LzwTableSize) h = 0;
                }

                if (!found)
                {
                    bitWriter.WriteBits(curPrefix, codeSize);

                    if (nextCode < 4096)
                    {
                        hashKeys[h] = key;
                        hashCodes[h] = nextCode++;
                        if (nextCode > codeLimit && codeSize < 12)
                        {
                            codeSize++;
                            codeLimit = 1 << codeSize;
                        }
                    }
                    else
                    {
                        // Reset dictionary
                        bitWriter.WriteBits(clearCode, codeSize);
                        hashKeys.AsSpan(0, LzwTableSize).Fill(-1);
                        codeSize = initCodeSize + 1;
                        codeLimit = 1 << codeSize;
                        nextCode = eoiCode + 1;
                    }

                    curPrefix = k;
                }
            }

            if (curPrefix != -1)
            {
                bitWriter.WriteBits(curPrefix, codeSize);
            }

            // Send End of Information code
            bitWriter.WriteBits(eoiCode, codeSize);

            // Flush remaining bits and sub-block
            bitWriter.Flush();

            // Block Terminator
            bw.Write((byte)0x00);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(hashKeys);
            ArrayPool<int>.Shared.Return(hashCodes);
        }
    }

    private ref struct LzwBitWriter
    {
        private readonly BinaryWriter _bw;
        private readonly Span<byte> _subBlock;
        private int _subBlockLen;
        private uint _bitAccumulator;
        private int _bitsInAccumulator;

        public LzwBitWriter(BinaryWriter bw, Span<byte> subBlock)
        {
            _bw = bw;
            _subBlock = subBlock;
            _subBlockLen = 0;
            _bitAccumulator = 0;
            _bitsInAccumulator = 0;
        }

        public void WriteBits(int code, int size)
        {
            _bitAccumulator |= (uint)(code << _bitsInAccumulator);
            _bitsInAccumulator += size;
            while (_bitsInAccumulator >= 8)
            {
                _subBlock[1 + _subBlockLen++] = (byte)(_bitAccumulator & 0xFF);
                if (_subBlockLen == 255)
                {
                    _subBlock[0] = 255;
                    _bw.Write(_subBlock);
                    _subBlockLen = 0;
                }
                _bitAccumulator >>= 8;
                _bitsInAccumulator -= 8;
            }
        }

        public void Flush()
        {
            if (_bitsInAccumulator > 0)
            {
                _subBlock[1 + _subBlockLen++] = (byte)(_bitAccumulator & 0xFF);
                if (_subBlockLen == 255)
                {
                    _subBlock[0] = 255;
                    _bw.Write(_subBlock);
                    _subBlockLen = 0;
                }
            }

            if (_subBlockLen > 0)
            {
                _subBlock[0] = (byte)_subBlockLen;
                _bw.Write(_subBlock.Slice(0, 1 + _subBlockLen));
                _subBlockLen = 0;
            }
        }
    }
}
