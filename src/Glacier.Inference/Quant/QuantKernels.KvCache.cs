namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

public static unsafe partial class QuantKernels
{
    private static readonly float[] s_fp8E4M3Lut = BuildFp8E4M3Lut();

    private static float[] BuildFp8E4M3Lut()
    {
        var lut = new float[256];
        for (int c = 0; c < 256; c++)
        {
            int sign = (c >> 7) & 1;
            int exp = (c >> 3) & 0xF;
            int mant = c & 0x7;

            if (exp == 15 && mant == 7)
            {
                lut[c] = 0f; // E4M3 NaN encoding
                continue;
            }

            float mag;
            if (exp == 0)
            {
                mag = (mant / 8.0f) * MathF.Pow(2.0f, -6);
            }
            else
            {
                mag = (1.0f + mant / 8.0f) * MathF.Pow(2.0f, exp - 7);
            }

            lut[c] = sign != 0 ? -mag : mag;
        }
        return lut;
    }

    /// <summary>
    /// Quantizes <paramref name="k"/> single-precision floats into packed 32-element <see cref="BlockQ4_0"/> blocks.
    /// Tail elements beyond <paramref name="k"/> in a partial final block are zero-padded.
    /// Uses symmetric zero-point 8 with <c>Delta = (Half)(amax / 7.0f)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeQ4_0(float* src, BlockQ4_0* dst, int k)
    {
        int nb = (k + QK4_0 - 1) / QK4_0;
        for (int b = 0; b < nb; b++)
        {
            int baseIdx = b * QK4_0;
            int remain = k - baseIdx;
            int count = remain < QK4_0 ? remain : QK4_0;

            float amax = 0f;
            for (int j = 0; j < count; j++)
            {
                float v = src[baseIdx + j];
                float a = float.IsFinite(v) ? MathF.Abs(v) : (float.IsInfinity(v) ? 458528.0f : 0f);
                if (a > amax) amax = a;
            }
            if (amax > 458528.0f) amax = 458528.0f;

            Half dh = (Half)(amax / 7.0f);
            float dQuant = (float)dh;
            if (!float.IsFinite(dQuant))
            {
                dh = Half.MaxValue;
                dQuant = (float)dh;
            }
            float invD = dQuant > 0f ? 1.0f / dQuant : 0f;

            BlockQ4_0* blk = dst + b;
            blk->Delta = dh;

            for (int l = 0; l < 16; l++)
            {
                float v0 = l < count ? src[baseIdx + l] : 0f;
                float v1 = (l + 16) < count ? src[baseIdx + l + 16] : 0f;
                int q0 = float.IsNaN(v0) ? 8 : (float.IsPositiveInfinity(v0) ? 15 : (float.IsNegativeInfinity(v0) ? 0 : (dQuant > 0f ? Math.Clamp((int)MathF.Round(v0 * invD) + 8, 0, 15) : 8)));
                int q1 = float.IsNaN(v1) ? 8 : (float.IsPositiveInfinity(v1) ? 15 : (float.IsNegativeInfinity(v1) ? 0 : (dQuant > 0f ? Math.Clamp((int)MathF.Round(v1 * invD) + 8, 0, 15) : 8)));
                blk->Qs[l] = (byte)(q0 | (q1 << 4));
            }
        }
    }

    /// <summary>
    /// Dequantizes <paramref name="k"/> elements from <see cref="BlockQ4_0"/> blocks into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeQ4_0(BlockQ4_0* src, float* dst, int k)
    {
        int nb = (k + QK4_0 - 1) / QK4_0;
        for (int b = 0; b < nb; b++)
        {
            float d = (float)src[b].Delta;
            byte* qs = src[b].Qs;
            int baseIdx = b * QK4_0;
            int remain = k - baseIdx;

            if (remain >= QK4_0)
            {
                for (int l = 0; l < 16; l++)
                {
                    byte v = qs[l];
                    dst[baseIdx + l] = ((v & 0x0F) - 8) * d;
                    dst[baseIdx + l + 16] = ((v >> 4) - 8) * d;
                }
            }
            else
            {
                for (int l = 0; l < 16; l++)
                {
                    byte v = qs[l];
                    if (l < remain)
                        dst[baseIdx + l] = ((v & 0x0F) - 8) * d;
                    if (l + 16 < remain)
                        dst[baseIdx + l + 16] = ((v >> 4) - 8) * d;
                }
            }
        }
    }

    /// <summary>
    /// Computes the dot product between a <see cref="BlockQ4_0"/> row and a float vector of arbitrary length <paramref name="k"/>.
    /// Dispatches SIMD <see cref="VecDotQ4_0"/> for full 32-element blocks and handles partial tail blocks cleanly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ4_0Any(BlockQ4_0* row, float* x, int k)
    {
        int fullElems = k & ~31;
        float sum = fullElems > 0 ? VecDotQ4_0(row, x, fullElems) : 0f;

        int tail = k - fullElems;
        if (tail > 0)
        {
            int tailBlock = fullElems >> 5;
            float d = (float)row[tailBlock].Delta;
            byte* q = row[tailBlock].Qs;
            float* xt = x + fullElems;
            float blockSum = 0f;
            for (int l = 0; l < 16; l++)
            {
                byte v = q[l];
                if (l < tail)
                    blockSum += ((v & 0x0F) - 8) * xt[l];
                if (l + 16 < tail)
                    blockSum += ((v >> 4) - 8) * xt[l + 16];
            }
            sum += d * blockSum;
        }

        return sum;
    }

    /// <summary>
    /// Accumulates <c>dst[i] += weight * dequant(src[i])</c> for <paramref name="k"/> elements stored in <see cref="BlockQ4_0"/> blocks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateScaledQ4_0(BlockQ4_0* src, float weight, float* dst, int k)
    {
        if (weight == 0f) return;

        int fullBlocks = k >> 5;
        for (int b = 0; b < fullBlocks; b++)
        {
            float wd = weight * (float)src[b].Delta;
            byte* q = src[b].Qs;
            float* outPtr = dst + (b << 5);

            if (Vector512.IsHardwareAccelerated)
            {
                var vwd = Vector512.Create(wd);
                var vqLow = Vector512.Create(
                    (float)((q[0] & 0x0F) - 8), (float)((q[1] & 0x0F) - 8), (float)((q[2] & 0x0F) - 8), (float)((q[3] & 0x0F) - 8),
                    (float)((q[4] & 0x0F) - 8), (float)((q[5] & 0x0F) - 8), (float)((q[6] & 0x0F) - 8), (float)((q[7] & 0x0F) - 8),
                    (float)((q[8] & 0x0F) - 8), (float)((q[9] & 0x0F) - 8), (float)((q[10] & 0x0F) - 8), (float)((q[11] & 0x0F) - 8),
                    (float)((q[12] & 0x0F) - 8), (float)((q[13] & 0x0F) - 8), (float)((q[14] & 0x0F) - 8), (float)((q[15] & 0x0F) - 8));
                var vqHigh = Vector512.Create(
                    (float)((q[0] >> 4) - 8), (float)((q[1] >> 4) - 8), (float)((q[2] >> 4) - 8), (float)((q[3] >> 4) - 8),
                    (float)((q[4] >> 4) - 8), (float)((q[5] >> 4) - 8), (float)((q[6] >> 4) - 8), (float)((q[7] >> 4) - 8),
                    (float)((q[8] >> 4) - 8), (float)((q[9] >> 4) - 8), (float)((q[10] >> 4) - 8), (float)((q[11] >> 4) - 8),
                    (float)((q[12] >> 4) - 8), (float)((q[13] >> 4) - 8), (float)((q[14] >> 4) - 8), (float)((q[15] >> 4) - 8));

                Vector512.FusedMultiplyAdd(vwd, vqLow, Vector512.Load(outPtr)).Store(outPtr);
                Vector512.FusedMultiplyAdd(vwd, vqHigh, Vector512.Load(outPtr + 16)).Store(outPtr + 16);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                var vwd = Vector256.Create(wd);
                var vqLow0 = Vector256.Create(
                    (float)((q[0] & 0x0F) - 8), (float)((q[1] & 0x0F) - 8), (float)((q[2] & 0x0F) - 8), (float)((q[3] & 0x0F) - 8),
                    (float)((q[4] & 0x0F) - 8), (float)((q[5] & 0x0F) - 8), (float)((q[6] & 0x0F) - 8), (float)((q[7] & 0x0F) - 8));
                var vqLow1 = Vector256.Create(
                    (float)((q[8] & 0x0F) - 8), (float)((q[9] & 0x0F) - 8), (float)((q[10] & 0x0F) - 8), (float)((q[11] & 0x0F) - 8),
                    (float)((q[12] & 0x0F) - 8), (float)((q[13] & 0x0F) - 8), (float)((q[14] & 0x0F) - 8), (float)((q[15] & 0x0F) - 8));
                var vqHigh0 = Vector256.Create(
                    (float)((q[0] >> 4) - 8), (float)((q[1] >> 4) - 8), (float)((q[2] >> 4) - 8), (float)((q[3] >> 4) - 8),
                    (float)((q[4] >> 4) - 8), (float)((q[5] >> 4) - 8), (float)((q[6] >> 4) - 8), (float)((q[7] >> 4) - 8));
                var vqHigh1 = Vector256.Create(
                    (float)((q[8] >> 4) - 8), (float)((q[9] >> 4) - 8), (float)((q[10] >> 4) - 8), (float)((q[11] >> 4) - 8),
                    (float)((q[12] >> 4) - 8), (float)((q[13] >> 4) - 8), (float)((q[14] >> 4) - 8), (float)((q[15] >> 4) - 8));

                (Vector256.Load(outPtr) + vwd * vqLow0).Store(outPtr);
                (Vector256.Load(outPtr + 8) + vwd * vqLow1).Store(outPtr + 8);
                (Vector256.Load(outPtr + 16) + vwd * vqHigh0).Store(outPtr + 16);
                (Vector256.Load(outPtr + 24) + vwd * vqHigh1).Store(outPtr + 24);
            }
            else
            {
                for (int l = 0; l < 16; l++)
                {
                    byte v = q[l];
                    outPtr[l] += wd * ((v & 0x0F) - 8);
                    outPtr[l + 16] += wd * ((v >> 4) - 8);
                }
            }
        }

        int fullElems = fullBlocks << 5;
        int tail = k - fullElems;
        if (tail > 0)
        {
            float wd = weight * (float)src[fullBlocks].Delta;
            byte* q = src[fullBlocks].Qs;
            float* outPtr = dst + fullElems;
            for (int l = 0; l < 16; l++)
            {
                byte v = q[l];
                if (l < tail)
                    outPtr[l] += wd * ((v & 0x0F) - 8);
                if (l + 16 < tail)
                    outPtr[l + 16] += wd * ((v >> 4) - 8);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeF16(float* src, Half* dst, int k)
    {
        for (int i = 0; i < k; i++)
        {
            dst[i] = (Half)src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeF16(Half* src, float* dst, int k)
    {
        for (int i = 0; i < k; i++)
        {
            dst[i] = (float)src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateScaledF16(Half* src, float weight, float* dst, int k)
    {
        if (weight == 0f) return;

        int i = 0;
        if (Vector256.IsHardwareAccelerated && k >= 8)
        {
            var vw = Vector256.Create(weight);
            int limit = k - 8;
            for (; i <= limit; i += 8)
            {
                var vv = Vector256.Create(
                    (float)src[i + 0], (float)src[i + 1], (float)src[i + 2], (float)src[i + 3],
                    (float)src[i + 4], (float)src[i + 5], (float)src[i + 6], (float)src[i + 7]);
                var vo = Vector256.Load(dst + i);
                (vo + vw * vv).Store(dst + i);
            }
        }
        for (; i < k; i++)
        {
            dst[i] += weight * (float)src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte FloatToFp8E4M3(float x)
    {
        if (float.IsNaN(x))
            return 0;

        uint bits = BitConverter.SingleToUInt32Bits(x);
        byte sign = (byte)((bits >> 24) & 0x80u);
        uint absBits = bits & 0x7FFFFFFFu;

        if (absBits >= 0x43E00000u) // >= 448.0f
            return (byte)(sign | 0x7Eu);

        int exp32 = (int)(absBits >> 23);
        int unbiasedExp = exp32 - 127;
        if (unbiasedExp < -10)
            return sign;

        if (unbiasedExp < -6)
        {
            float mag = BitConverter.UInt32BitsToSingle(absBits);
            int q = (int)MathF.Round(mag * 512.0f);
            if (q <= 0) return sign;
            if (q >= 8) return (byte)(sign | 0x08u);
            return (byte)(sign | (byte)q);
        }

        int e4 = unbiasedExp + 7;
        uint mant32 = absBits & 0x7FFFFFu;
        uint m3 = mant32 >> 20;
        uint rem = mant32 & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u) != 0u))
        {
            m3++;
            if (m3 == 8u)
            {
                m3 = 0u;
                e4++;
            }
        }

        if (e4 > 15 || (e4 == 15 && m3 >= 7u))
            return (byte)(sign | 0x7Eu);

        return (byte)(sign | (byte)((e4 << 3) | (int)m3));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeFp8E4M3(float* src, byte* dst, int k)
    {
        for (int i = 0; i < k; i++)
        {
            dst[i] = FloatToFp8E4M3(src[i]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeFp8E4M3(byte* src, float* dst, int k)
    {
        fixed (float* lut = s_fp8E4M3Lut)
        {
            for (int i = 0; i < k; i++)
            {
                dst[i] = lut[src[i]];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotFp8E4M3(byte* row, float* x, int k)
    {
        float sum = 0f;
        fixed (float* lut = s_fp8E4M3Lut)
        {
            int i = 0;
            if (Vector256.IsHardwareAccelerated && k >= 8)
            {
                var acc = Vector256<float>.Zero;
                int limit = k - 8;
                for (; i <= limit; i += 8)
                {
                    var vk = Vector256.Create(
                        lut[row[i + 0]], lut[row[i + 1]], lut[row[i + 2]], lut[row[i + 3]],
                        lut[row[i + 4]], lut[row[i + 5]], lut[row[i + 6]], lut[row[i + 7]]);
                    var vx = Vector256.Load(x + i);
                    acc += vk * vx;
                }
                sum += Vector256.Sum(acc);
            }
            for (; i < k; i++)
            {
                sum += lut[row[i]] * x[i];
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateScaledFp8E4M3(byte* src, float weight, float* dst, int k)
    {
        if (weight == 0f) return;

        fixed (float* lut = s_fp8E4M3Lut)
        {
            int i = 0;
            if (Vector256.IsHardwareAccelerated && k >= 8)
            {
                var vw = Vector256.Create(weight);
                int limit = k - 8;
                for (; i <= limit; i += 8)
                {
                    var vv = Vector256.Create(
                        lut[src[i + 0]], lut[src[i + 1]], lut[src[i + 2]], lut[src[i + 3]],
                        lut[src[i + 4]], lut[src[i + 5]], lut[src[i + 6]], lut[src[i + 7]]);
                    var vo = Vector256.Load(dst + i);
                    (vo + vw * vv).Store(dst + i);
                }
            }
            for (; i < k; i++)
            {
                dst[i] += weight * lut[src[i]];
            }
        }
    }
}
