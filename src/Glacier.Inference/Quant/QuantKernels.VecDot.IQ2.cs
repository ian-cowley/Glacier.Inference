namespace Glacier.Inference.Quant;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static unsafe partial class QuantKernels
{
    internal static readonly ulong[] Iq2xxsGrid = new ulong[256]
    {
        0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
        0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
        0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
        0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
        0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
        0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
        0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
        0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
        0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
        0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
        0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
        0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
        0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
        0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
        0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
        0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
        0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
        0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
        0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
        0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
        0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
        0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
        0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
        0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
        0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
        0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
        0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
        0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
        0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
        0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
        0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
        0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
        0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
        0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
        0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
        0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
        0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
        0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
        0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
        0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
        0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
        0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
        0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
        0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
        0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
        0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
        0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
        0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
        0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
        0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
        0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
        0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
        0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
        0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
        0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
        0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
        0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
        0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
        0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
        0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
        0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
        0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
        0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
        0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
    };

    internal static readonly byte[] KSignsIq2Xs = new byte[128]
    {
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
        144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
        160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
        48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
        192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
        80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
        96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
    };

    /// <summary>
    /// 512-entry E8 sphere lattice codebook for IQ2_XS (extends <see cref="Iq2xxsGrid"/>).
    /// </summary>
    internal static readonly ulong[] Iq2xsGrid = BuildIq2Grid(512);

    /// <summary>
    /// 1024-entry E8 sphere lattice codebook for IQ2_S (extends <see cref="Iq2xsGrid"/>).
    /// </summary>
    internal static readonly ulong[] Iq2sGrid = BuildIq2Grid(1024);

    private static ulong[] BuildIq2Grid(int targetCount)
    {
        // E8 shifted-lattice magnitudes in the positive orthant (levels 0..3 -> {8, 25, 43, 61})
        // obeying the even coordinate-sum parity rule: (sum_j level[j]) % 2 == 0.
        ReadOnlySpan<byte> levels = [8, 25, 43, 61];
        var seen = new HashSet<ulong>(targetCount);
        var grid = new ulong[targetCount];

        int count = Math.Min(Iq2xxsGrid.Length, targetCount);
        for (int i = 0; i < count; i++)
        {
            grid[i] = Iq2xxsGrid[i];
            seen.Add(Iq2xxsGrid[i]);
        }

        if (targetCount <= Iq2xxsGrid.Length)
            return grid;

        var candidates = new List<(ulong Packed, uint NormSq)>(32768);
        for (int idx = 0; idx < 65536; idx++)
        {
            int l0 = (idx >> 0) & 3;
            int l1 = (idx >> 2) & 3;
            int l2 = (idx >> 4) & 3;
            int l3 = (idx >> 6) & 3;
            int l4 = (idx >> 8) & 3;
            int l5 = (idx >> 10) & 3;
            int l6 = (idx >> 12) & 3;
            int l7 = (idx >> 14) & 3;

            if (((l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7) & 1) != 0)
                continue;

            byte m0 = levels[l0], m1 = levels[l1], m2 = levels[l2], m3 = levels[l3];
            byte m4 = levels[l4], m5 = levels[l5], m6 = levels[l6], m7 = levels[l7];

            ulong packed = (ulong)m0
                | ((ulong)m1 << 8)
                | ((ulong)m2 << 16)
                | ((ulong)m3 << 24)
                | ((ulong)m4 << 32)
                | ((ulong)m5 << 40)
                | ((ulong)m6 << 48)
                | ((ulong)m7 << 56);

            if (seen.Contains(packed))
                continue;

            uint normSq = (uint)(m0 * m0 + m1 * m1 + m2 * m2 + m3 * m3 + m4 * m4 + m5 * m5 + m6 * m6 + m7 * m7);
            candidates.Add((packed, normSq));
        }

        candidates.Sort(static (a, b) =>
        {
            int c = a.NormSq.CompareTo(b.NormSq);
            return c != 0 ? c : a.Packed.CompareTo(b.Packed);
        });

        for (int i = 0; i < candidates.Count && count < targetCount; i++)
        {
            grid[count++] = candidates[i].Packed;
        }

        return grid;
    }

    /// <summary>
    /// Computes dot product between an IQ2_XXS quantized row and a float vector x of length k.
    /// Super-block size is 256 elements in 66 bytes (2.0625 bits per weight).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ2_XXS(BlockIQ2_XXS* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        fixed (ulong* pGrid = Iq2xxsGrid)
        fixed (byte* pSigns = KSignsIq2Xs)
        {
            uint* aux32 = stackalloc uint[2];
            byte* aux8 = (byte*)aux32;

            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                ushort* qs = row[i].Qs;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    aux32[0] = *(uint*)(qs + 4 * ib32);
                    aux32[1] = *(uint*)(qs + 4 * ib32 + 2);

                    float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

                    for (int l = 0; l < 4; ++l)
                    {
                        byte* grid = (byte*)(pGrid + aux8[l]);
                        byte signs = pSigns[(aux32[1] >> (7 * l)) & 127];

                        float subSum = 0f;
                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            subSum += x[j] * (grid[j] * sign);
                        }

                        totalSum += db * subSum;
                        x += 8;
                    }
                }
            }
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes an IQ2_XXS quantized row to float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeIQ2_XXS(BlockIQ2_XXS* src, float* dst, int k)
    {
        int nb = k / QK_K;

        fixed (ulong* pGrid = Iq2xxsGrid)
        fixed (byte* pSigns = KSignsIq2Xs)
        {
            uint* aux32 = stackalloc uint[2];
            byte* aux8 = (byte*)aux32;

            for (int i = 0; i < nb; i++)
            {
                float d = (float)src[i].Delta;
                ushort* qs = src[i].Qs;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    aux32[0] = *(uint*)(qs + 4 * ib32);
                    aux32[1] = *(uint*)(qs + 4 * ib32 + 2);

                    float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;

                    for (int l = 0; l < 4; ++l)
                    {
                        byte* grid = (byte*)(pGrid + aux8[l]);
                        byte signs = pSigns[(aux32[1] >> (7 * l)) & 127];

                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            dst[j] = db * grid[j] * sign;
                        }
                        dst += 8;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes dot product between an IQ2_XS quantized row (256 elements in 74 bytes) and float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ2_XS(BlockIQ2_XS* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        fixed (ulong* pGrid = Iq2xsGrid)
        fixed (byte* pSigns = KSignsIq2Xs)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                ushort* qs = row[i].Qs;
                byte* scales = row[i].Scales;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    byte sc = scales[ib32];
                    for (int l = 0; l < 4; ++l)
                    {
                        ushort q = qs[4 * ib32 + l];
                        int ls = (l < 2) ? (sc & 0x0F) : (sc >> 4);
                        float db = d * (0.5f + ls) * 0.25f;
                        byte* grid = (byte*)(pGrid + (q & 511));
                        byte signs = pSigns[q >> 9];

                        float subSum = 0f;
                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            subSum += x[j] * (grid[j] * sign);
                        }

                        totalSum += db * subSum;
                        x += 8;
                    }
                }
            }
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes an IQ2_XS quantized row (256 elements in 74 bytes) to float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeIQ2_XS(BlockIQ2_XS* src, float* dst, int k)
    {
        int nb = k / QK_K;

        fixed (ulong* pGrid = Iq2xsGrid)
        fixed (byte* pSigns = KSignsIq2Xs)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)src[i].Delta;
                ushort* qs = src[i].Qs;
                byte* scales = src[i].Scales;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    byte sc = scales[ib32];
                    for (int l = 0; l < 4; ++l)
                    {
                        ushort q = qs[4 * ib32 + l];
                        int ls = (l < 2) ? (sc & 0x0F) : (sc >> 4);
                        float db = d * (0.5f + ls) * 0.25f;
                        byte* grid = (byte*)(pGrid + (q & 511));
                        byte signs = pSigns[q >> 9];

                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            dst[j] = db * grid[j] * sign;
                        }
                        dst += 8;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes dot product between an IQ2_S quantized row (256 elements in 82 bytes) and float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ2_S(BlockIQ2_S* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        fixed (ulong* pGrid = Iq2sGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                byte* qs = row[i].Qs;
                byte* qh = row[i].Qh;
                byte* scales = row[i].Scales;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    byte sc = scales[ib32];
                    byte qhByte = qh[ib32];
                    for (int l = 0; l < 4; ++l)
                    {
                        int gridIdx = qs[4 * ib32 + l] | (((qhByte >> (2 * l)) & 3) << 8);
                        byte signs = qs[32 + 4 * ib32 + l];
                        int ls = (l < 2) ? (sc & 0x0F) : (sc >> 4);
                        float db = d * (0.5f + ls) * 0.25f;
                        byte* grid = (byte*)(pGrid + gridIdx);

                        float subSum = 0f;
                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            subSum += x[j] * (grid[j] * sign);
                        }

                        totalSum += db * subSum;
                        x += 8;
                    }
                }
            }
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes an IQ2_S quantized row (256 elements in 82 bytes) to float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeIQ2_S(BlockIQ2_S* src, float* dst, int k)
    {
        int nb = k / QK_K;

        fixed (ulong* pGrid = Iq2sGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)src[i].Delta;
                byte* qs = src[i].Qs;
                byte* qh = src[i].Qh;
                byte* scales = src[i].Scales;

                for (int ib32 = 0; ib32 < 8; ++ib32)
                {
                    byte sc = scales[ib32];
                    byte qhByte = qh[ib32];
                    for (int l = 0; l < 4; ++l)
                    {
                        int gridIdx = qs[4 * ib32 + l] | (((qhByte >> (2 * l)) & 3) << 8);
                        byte signs = qs[32 + 4 * ib32 + l];
                        int ls = (l < 2) ? (sc & 0x0F) : (sc >> 4);
                        float db = d * (0.5f + ls) * 0.25f;
                        byte* grid = (byte*)(pGrid + gridIdx);

                        for (int j = 0; j < 8; ++j)
                        {
                            float sign = ((signs >> j) & 1) != 0 ? -1f : 1f;
                            dst[j] = db * grid[j] * sign;
                        }
                        dst += 8;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Quantizes a float row of length k into IQ2_XS super-blocks (256 elements in 74 bytes).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeIQ2_XS(float* src, BlockIQ2_XS* dst, int k)
    {
        int nb = k / QK_K;
        fixed (ulong* pGrid = Iq2xsGrid)
        fixed (byte* pSigns = KSignsIq2Xs)
        {
            for (int i = 0; i < nb; i++)
            {
                float* x = src + i * QK_K;
                float maxAbs = 0f;
                for (int j = 0; j < QK_K; j++)
                {
                    float a = MathF.Abs(x[j]);
                    if (a > maxAbs) maxAbs = a;
                }

                float d = maxAbs / (43.0f * 15.5f * 0.25f);
                dst[i].Delta = (Half)d;
                float dActual = (float)dst[i].Delta;

                for (int ib32 = 0; ib32 < 8; ib32++)
                {
                    int ls0 = ComputeIq2SubScale(x + ib32 * 32, 16, dActual, maxGridMag: 43.0f);
                    int ls1 = ComputeIq2SubScale(x + ib32 * 32 + 16, 16, dActual, maxGridMag: 43.0f);
                    dst[i].Scales[ib32] = (byte)((ls0 & 0xF) | ((ls1 & 0xF) << 4));

                    for (int l = 0; l < 4; l++)
                    {
                        float* xg = x + ib32 * 32 + l * 8;
                        int ls = (l < 2) ? ls0 : ls1;
                        float db = dActual * (0.5f + ls) * 0.25f;
                        float invDb = db > 1e-12f ? 1.0f / db : 0f;

                        int rawSigns = 0;
                        float minMag = float.MaxValue;
                        int minIdx = 0;
                        for (int j = 0; j < 8; j++)
                        {
                            float v = xg[j];
                            if (v < 0f) rawSigns |= (1 << j);
                            float a = MathF.Abs(v);
                            if (a < minMag) { minMag = a; minIdx = j; }
                        }

                        int signIdx = rawSigns & 127;
                        if (pSigns[signIdx] != rawSigns && minIdx < 7)
                        {
                            // Match parity by flipping the smallest-magnitude coordinate when beneficial
                            signIdx = (rawSigns ^ (1 << minIdx)) & 127;
                        }
                        byte usedSigns = pSigns[signIdx];

                        int bestGrid = 0;
                        float bestErr = float.MaxValue;
                        for (int g = 0; g < Iq2xsGrid.Length; g++)
                        {
                            byte* pg = (byte*)(pGrid + g);
                            float err = 0f;
                            for (int j = 0; j < 8; j++)
                            {
                                float qv = pg[j] * (((usedSigns >> j) & 1) != 0 ? -1f : 1f);
                                float diff = qv - (xg[j] * invDb);
                                err += diff * diff;
                            }
                            if (err < bestErr)
                            {
                                bestErr = err;
                                bestGrid = g;
                            }
                        }

                        dst[i].Qs[4 * ib32 + l] = (ushort)((bestGrid & 511) | ((signIdx & 127) << 9));
                    }
                }
            }
        }
    }

    /// <summary>
    /// Quantizes a float row of length k into IQ2_S super-blocks (256 elements in 82 bytes).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeIQ2_S(float* src, BlockIQ2_S* dst, int k)
    {
        int nb = k / QK_K;
        fixed (ulong* pGrid = Iq2sGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                float* x = src + i * QK_K;
                float maxAbs = 0f;
                for (int j = 0; j < QK_K; j++)
                {
                    float a = MathF.Abs(x[j]);
                    if (a > maxAbs) maxAbs = a;
                }

                float d = maxAbs / (43.0f * 15.5f * 0.25f);
                dst[i].Delta = (Half)d;
                float dActual = (float)dst[i].Delta;

                for (int ib32 = 0; ib32 < 8; ib32++)
                {
                    int ls0 = ComputeIq2SubScale(x + ib32 * 32, 16, dActual, maxGridMag: 43.0f);
                    int ls1 = ComputeIq2SubScale(x + ib32 * 32 + 16, 16, dActual, maxGridMag: 43.0f);
                    dst[i].Scales[ib32] = (byte)((ls0 & 0xF) | ((ls1 & 0xF) << 4));
                    byte qhByte = 0;

                    for (int l = 0; l < 4; l++)
                    {
                        float* xg = x + ib32 * 32 + l * 8;
                        int ls = (l < 2) ? ls0 : ls1;
                        float db = dActual * (0.5f + ls) * 0.25f;
                        float invDb = db > 1e-12f ? 1.0f / db : 0f;

                        byte signs = 0;
                        for (int j = 0; j < 8; j++)
                        {
                            if (xg[j] < 0f) signs |= (byte)(1 << j);
                        }

                        int bestGrid = 0;
                        float bestErr = float.MaxValue;
                        for (int g = 0; g < Iq2sGrid.Length; g++)
                        {
                            byte* pg = (byte*)(pGrid + g);
                            float err = 0f;
                            for (int j = 0; j < 8; j++)
                            {
                                float qv = pg[j];
                                float diff = qv - (MathF.Abs(xg[j]) * invDb);
                                err += diff * diff;
                            }
                            if (err < bestErr)
                            {
                                bestErr = err;
                                bestGrid = g;
                            }
                        }

                        dst[i].Qs[4 * ib32 + l] = (byte)(bestGrid & 0xFF);
                        dst[i].Qs[32 + 4 * ib32 + l] = signs;
                        qhByte |= (byte)(((bestGrid >> 8) & 3) << (2 * l));
                    }

                    dst[i].Qh[ib32] = qhByte;
                }
            }
        }
    }

    private static int ComputeIq2SubScale(float* x, int count, float d, float maxGridMag = 43.0f)
    {
        if (d <= 1e-12f) return 0;
        float subMax = 0f;
        for (int j = 0; j < count; j++)
        {
            float a = MathF.Abs(x[j]);
            if (a > subMax) subMax = a;
        }
        return Math.Clamp((int)MathF.Round((subMax / (maxGridMag * d * 0.25f)) - 0.5f), 0, 15);
    }
}
