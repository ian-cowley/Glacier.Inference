namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

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

    /// <summary>
    /// Quantizes <paramref name="k"/> single-precision floats into symmetric 32-element <see cref="BlockQ8_0"/> blocks.
    /// Tail elements beyond <paramref name="k"/> in a partial final block are zero-padded.
    /// Uses <c>Delta = (Half)(amax / 127.0f)</c> and symmetric clamping <c>[-127, 127]</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeQ8_0(float* src, BlockQ8_0* dst, int k)
    {
        if (k <= 0) return;

        const float MaxQ8Abs = 8319008.0f; // Half.MaxValue (65504f) * 127f
        int nb = (k + QK8_0 - 1) / QK8_0;

        for (int b = 0; b < nb; b++)
        {
            int baseIdx = b * QK8_0;
            int remain = k - baseIdx;
            int count = remain < QK8_0 ? remain : QK8_0;
            BlockQ8_0* blk = dst + b;

            if (count == QK8_0)
            {
                if (Vector512.IsHardwareAccelerated)
                {
                    var v0 = Vector512.Load(src + baseIdx);
                    var v1 = Vector512.Load(src + baseIdx + 16);
                    var a0 = Vector512.Abs(v0);
                    var a1 = Vector512.Abs(v1);
                    var vCap = Vector512.Create(MaxQ8Abs);

                    if (Vector512.LessThanOrEqualAll(a0, vCap) && Vector512.LessThanOrEqualAll(a1, vCap))
                    {
                        var m512 = Vector512.Max(a0, a1);
                        var m256 = Vector256.Max(m512.GetLower(), m512.GetUpper());
                        var m128 = Vector128.Max(m256.GetLower(), m256.GetUpper());
                        float amax = MathF.Max(MathF.Max(m128[0], m128[1]), MathF.Max(m128[2], m128[3]));

                        Half dh = (Half)(amax / 127.0f);
                        float dQuant = (float)dh;
                        if (!float.IsFinite(dQuant))
                        {
                            dh = Half.MaxValue;
                            dQuant = (float)dh;
                        }
                        float invD = dQuant > 0f ? 1.0f / dQuant : 0f;
                        blk->Delta = dh;

                        var vInv = Vector512.Create(invD);
                        var vMin127 = Vector512.Create(-127.0f);
                        var vMax127 = Vector512.Create(127.0f);

                        var qf0 = Vector512.Max(vMin127, Vector512.Min(vMax127, Vector512.Round(v0 * vInv)));
                        var qf1 = Vector512.Max(vMin127, Vector512.Min(vMax127, Vector512.Round(v1 * vInv)));
                        var qi0 = Vector512.ConvertToInt32(qf0);
                        var qi1 = Vector512.ConvertToInt32(qf1);

                        if (Avx512F.IsSupported)
                        {
                            Avx512F.ConvertToVector128SByteWithSaturation(qi0).Store(blk->Qs);
                            Avx512F.ConvertToVector128SByteWithSaturation(qi1).Store(blk->Qs + 16);
                        }
                        else
                        {
                            var qs16 = Vector512.Narrow(qi0, qi1);
                            Vector256.Narrow(qs16.GetLower(), qs16.GetUpper()).Store(blk->Qs);
                        }
                        continue;
                    }
                }
                else if (Vector256.IsHardwareAccelerated)
                {
                    var v0 = Vector256.Load(src + baseIdx);
                    var v1 = Vector256.Load(src + baseIdx + 8);
                    var v2 = Vector256.Load(src + baseIdx + 16);
                    var v3 = Vector256.Load(src + baseIdx + 24);
                    var a0 = Vector256.Abs(v0);
                    var a1 = Vector256.Abs(v1);
                    var a2 = Vector256.Abs(v2);
                    var a3 = Vector256.Abs(v3);
                    var vCap = Vector256.Create(MaxQ8Abs);

                    if (Vector256.LessThanOrEqualAll(a0, vCap) &&
                        Vector256.LessThanOrEqualAll(a1, vCap) &&
                        Vector256.LessThanOrEqualAll(a2, vCap) &&
                        Vector256.LessThanOrEqualAll(a3, vCap))
                    {
                        var m256 = Vector256.Max(Vector256.Max(a0, a1), Vector256.Max(a2, a3));
                        var m128 = Vector128.Max(m256.GetLower(), m256.GetUpper());
                        float amax = MathF.Max(MathF.Max(m128[0], m128[1]), MathF.Max(m128[2], m128[3]));

                        Half dh = (Half)(amax / 127.0f);
                        float dQuant = (float)dh;
                        if (!float.IsFinite(dQuant))
                        {
                            dh = Half.MaxValue;
                            dQuant = (float)dh;
                        }
                        float invD = dQuant > 0f ? 1.0f / dQuant : 0f;
                        blk->Delta = dh;

                        var vInv = Vector256.Create(invD);
                        var vMin127 = Vector256.Create(-127.0f);
                        var vMax127 = Vector256.Create(127.0f);

                        var qi0 = Vector256.ConvertToInt32(Vector256.Max(vMin127, Vector256.Min(vMax127, Vector256.Round(v0 * vInv))));
                        var qi1 = Vector256.ConvertToInt32(Vector256.Max(vMin127, Vector256.Min(vMax127, Vector256.Round(v1 * vInv))));
                        var qi2 = Vector256.ConvertToInt32(Vector256.Max(vMin127, Vector256.Min(vMax127, Vector256.Round(v2 * vInv))));
                        var qi3 = Vector256.ConvertToInt32(Vector256.Max(vMin127, Vector256.Min(vMax127, Vector256.Round(v3 * vInv))));

                        var qs01 = Vector256.Narrow(qi0, qi1);
                        var qs23 = Vector256.Narrow(qi2, qi3);
                        Vector256.Narrow(qs01, qs23).Store(blk->Qs);
                        continue;
                    }
                }
                else if (AdvSimd.IsSupported)
                {
                    var v0 = Vector128.Load(src + baseIdx);
                    var v1 = Vector128.Load(src + baseIdx + 4);
                    var v2 = Vector128.Load(src + baseIdx + 8);
                    var v3 = Vector128.Load(src + baseIdx + 12);
                    var v4 = Vector128.Load(src + baseIdx + 16);
                    var v5 = Vector128.Load(src + baseIdx + 20);
                    var v6 = Vector128.Load(src + baseIdx + 24);
                    var v7 = Vector128.Load(src + baseIdx + 28);

                    var a0 = AdvSimd.Abs(v0);
                    var a1 = AdvSimd.Abs(v1);
                    var a2 = AdvSimd.Abs(v2);
                    var a3 = AdvSimd.Abs(v3);
                    var a4 = AdvSimd.Abs(v4);
                    var a5 = AdvSimd.Abs(v5);
                    var a6 = AdvSimd.Abs(v6);
                    var a7 = AdvSimd.Abs(v7);
                    var vCap = Vector128.Create(MaxQ8Abs);

                    if (Vector128.LessThanOrEqualAll(a0, vCap) &&
                        Vector128.LessThanOrEqualAll(a1, vCap) &&
                        Vector128.LessThanOrEqualAll(a2, vCap) &&
                        Vector128.LessThanOrEqualAll(a3, vCap) &&
                        Vector128.LessThanOrEqualAll(a4, vCap) &&
                        Vector128.LessThanOrEqualAll(a5, vCap) &&
                        Vector128.LessThanOrEqualAll(a6, vCap) &&
                        Vector128.LessThanOrEqualAll(a7, vCap))
                    {
                        var m128 = Vector128.Max(
                            Vector128.Max(Vector128.Max(a0, a1), Vector128.Max(a2, a3)),
                            Vector128.Max(Vector128.Max(a4, a5), Vector128.Max(a6, a7)));
                        float amax = AdvSimd.Arm64.IsSupported
                            ? AdvSimd.Arm64.MaxAcross(m128).ToScalar()
                            : MathF.Max(MathF.Max(m128[0], m128[1]), MathF.Max(m128[2], m128[3]));

                        Half dh = (Half)(amax / 127.0f);
                        float dQuant = (float)dh;
                        if (!float.IsFinite(dQuant))
                        {
                            dh = Half.MaxValue;
                            dQuant = (float)dh;
                        }
                        float invD = dQuant > 0f ? 1.0f / dQuant : 0f;
                        blk->Delta = dh;

                        var vInv = Vector128.Create(invD);
                        var vMin127 = Vector128.Create(-127.0f);
                        var vMax127 = Vector128.Create(127.0f);

                        var qi0 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v0 * vInv))));
                        var qi1 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v1 * vInv))));
                        var qi2 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v2 * vInv))));
                        var qi3 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v3 * vInv))));
                        var qi4 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v4 * vInv))));
                        var qi5 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v5 * vInv))));
                        var qi6 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v6 * vInv))));
                        var qi7 = Vector128.ConvertToInt32(Vector128.Max(vMin127, Vector128.Min(vMax127, AdvSimd.RoundToNearest(v7 * vInv))));

                        var qs01 = Vector128.Narrow(qi0, qi1);
                        var qs23 = Vector128.Narrow(qi2, qi3);
                        var qs45 = Vector128.Narrow(qi4, qi5);
                        var qs67 = Vector128.Narrow(qi6, qi7);

                        Vector128.Narrow(qs01, qs23).Store(blk->Qs);
                        Vector128.Narrow(qs45, qs67).Store(blk->Qs + 16);
                        continue;
                    }
                }
            }

            // Scalar fallback (also handles partial tail and non-finite/extreme values)
            float scalarAmax = 0f;
            for (int j = 0; j < count; j++)
            {
                float v = src[baseIdx + j];
                float a = float.IsFinite(v) ? MathF.Abs(v) : (float.IsInfinity(v) ? MaxQ8Abs : 0f);
                if (a > scalarAmax) scalarAmax = a;
            }
            if (scalarAmax > MaxQ8Abs) scalarAmax = MaxQ8Abs;

            Half scalarDh = (Half)(scalarAmax / 127.0f);
            float scalarDQuant = (float)scalarDh;
            if (!float.IsFinite(scalarDQuant))
            {
                scalarDh = Half.MaxValue;
                scalarDQuant = (float)scalarDh;
            }
            float scalarInvD = scalarDQuant > 0f ? 1.0f / scalarDQuant : 0f;
            blk->Delta = scalarDh;

            for (int j = 0; j < QK8_0; j++)
            {
                if (j >= count)
                {
                    blk->Qs[j] = 0;
                    continue;
                }

                float v = src[baseIdx + j];
                if (float.IsNaN(v))
                {
                    blk->Qs[j] = 0;
                }
                else if (float.IsPositiveInfinity(v))
                {
                    blk->Qs[j] = 127;
                }
                else if (float.IsNegativeInfinity(v))
                {
                    blk->Qs[j] = -127;
                }
                else
                {
                    blk->Qs[j] = (sbyte)(scalarDQuant > 0f ? Math.Clamp((int)MathF.Round(v * scalarInvD), -127, 127) : 0);
                }
            }
        }
    }

    /// <summary>
    /// Dequantizes <paramref name="k"/> elements from <see cref="BlockQ8_0"/> blocks into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeQ8_0(BlockQ8_0* src, float* dst, int k)
    {
        if (k <= 0) return;

        int nb = (k + QK8_0 - 1) / QK8_0;
        for (int b = 0; b < nb; b++)
        {
            float d = (float)src[b].Delta;
            sbyte* qs = src[b].Qs;
            int baseIdx = b * QK8_0;
            int remain = k - baseIdx;
            float* outPtr = dst + baseIdx;

            if (remain >= QK8_0)
            {
                if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
                {
                    var vd = Vector512.Create(d);
                    var q0 = Vector128.Load(qs);
                    var q1 = Vector128.Load(qs + 16);
                    var vf0 = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(q0)) * vd;
                    var vf1 = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(q1)) * vd;
                    vf0.Store(outPtr);
                    vf1.Store(outPtr + 16);
                }
                else if (Avx2.IsSupported && Sse2.IsSupported)
                {
                    var vd = Vector256.Create(d);
                    var qLo = Vector128.Load(qs);
                    var qHi = Vector128.Load(qs + 16);
                    var qLoUpper = Sse2.ShiftRightLogical128BitLane(qLo, 8);
                    var qHiUpper = Sse2.ShiftRightLogical128BitLane(qHi, 8);

                    (Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qLo)) * vd).Store(outPtr);
                    (Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qLoUpper)) * vd).Store(outPtr + 8);
                    (Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qHi)) * vd).Store(outPtr + 16);
                    (Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qHiUpper)) * vd).Store(outPtr + 24);
                }
                else if (AdvSimd.IsSupported)
                {
                    var vd = Vector128.Create(d);
                    var qLo = Vector128.Load(qs);
                    var qHi = Vector128.Load(qs + 16);

                    var s0 = AdvSimd.SignExtendWideningLower(qLo.GetLower());
                    var s1 = AdvSimd.SignExtendWideningUpper(qLo);
                    var s2 = AdvSimd.SignExtendWideningLower(qHi.GetLower());
                    var s3 = AdvSimd.SignExtendWideningUpper(qHi);

                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s0.GetLower())) * vd).Store(outPtr);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s0)) * vd).Store(outPtr + 4);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s1.GetLower())) * vd).Store(outPtr + 8);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s1)) * vd).Store(outPtr + 12);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s2.GetLower())) * vd).Store(outPtr + 16);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s2)) * vd).Store(outPtr + 20);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s3.GetLower())) * vd).Store(outPtr + 24);
                    (AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s3)) * vd).Store(outPtr + 28);
                }
                else
                {
                    for (int l = 0; l < QK8_0; l++)
                    {
                        outPtr[l] = d * qs[l];
                    }
                }
            }
            else
            {
                for (int l = 0; l < remain; l++)
                {
                    outPtr[l] = d * qs[l];
                }
            }
        }
    }

    /// <summary>
    /// Computes the dot product between a <see cref="BlockQ8_0"/> row and a float vector of arbitrary length <paramref name="k"/>.
    /// Dispatches SIMD <see cref="VecDotQ8_0"/> for full 32-element blocks and handles partial tail blocks cleanly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ8_0Any(BlockQ8_0* row, float* x, int k)
    {
        if (k <= 0) return 0f;

        int fullElems = k & ~31;
        float sum = fullElems > 0 ? VecDotQ8_0(row, x, fullElems) : 0f;

        int tail = k - fullElems;
        if (tail > 0)
        {
            int tailBlock = fullElems >> 5;
            float d = (float)row[tailBlock].Delta;
            sbyte* q = row[tailBlock].Qs;
            float* xt = x + fullElems;
            float blockSum = 0f;
            for (int l = 0; l < tail; l++)
            {
                blockSum += q[l] * xt[l];
            }
            sum += d * blockSum;
        }

        return sum;
    }

    /// <summary>
    /// Accumulates <c>dst[i] += weight * dequant(src[i])</c> for <paramref name="k"/> elements stored in <see cref="BlockQ8_0"/> blocks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AccumulateScaledQ8_0(BlockQ8_0* src, float weight, float* dst, int k)
    {
        if (k <= 0 || weight == 0f) return;

        int fullBlocks = k >> 5;
        for (int b = 0; b < fullBlocks; b++)
        {
            float wd = weight * (float)src[b].Delta;
            sbyte* qs = src[b].Qs;
            float* outPtr = dst + (b << 5);

            if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
            {
                var vwd = Vector512.Create(wd);
                var q0 = Vector128.Load(qs);
                var q1 = Vector128.Load(qs + 16);
                var vf0 = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(q0));
                var vf1 = Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(q1));

                Vector512.FusedMultiplyAdd(vwd, vf0, Vector512.Load(outPtr)).Store(outPtr);
                Vector512.FusedMultiplyAdd(vwd, vf1, Vector512.Load(outPtr + 16)).Store(outPtr + 16);
            }
            else if (Avx2.IsSupported && Sse2.IsSupported)
            {
                var vwd = Vector256.Create(wd);
                var qLo = Vector128.Load(qs);
                var qHi = Vector128.Load(qs + 16);
                var qLoUpper = Sse2.ShiftRightLogical128BitLane(qLo, 8);
                var qHiUpper = Sse2.ShiftRightLogical128BitLane(qHi, 8);

                var vf0 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qLo));
                var vf1 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qLoUpper));
                var vf2 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qHi));
                var vf3 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(qHiUpper));

                if (Fma.IsSupported)
                {
                    Fma.MultiplyAdd(vwd, vf0, Vector256.Load(outPtr)).Store(outPtr);
                    Fma.MultiplyAdd(vwd, vf1, Vector256.Load(outPtr + 8)).Store(outPtr + 8);
                    Fma.MultiplyAdd(vwd, vf2, Vector256.Load(outPtr + 16)).Store(outPtr + 16);
                    Fma.MultiplyAdd(vwd, vf3, Vector256.Load(outPtr + 24)).Store(outPtr + 24);
                }
                else
                {
                    (Vector256.Load(outPtr) + vwd * vf0).Store(outPtr);
                    (Vector256.Load(outPtr + 8) + vwd * vf1).Store(outPtr + 8);
                    (Vector256.Load(outPtr + 16) + vwd * vf2).Store(outPtr + 16);
                    (Vector256.Load(outPtr + 24) + vwd * vf3).Store(outPtr + 24);
                }
            }
            else if (AdvSimd.IsSupported)
            {
                var vwd = Vector128.Create(wd);
                var qLo = Vector128.Load(qs);
                var qHi = Vector128.Load(qs + 16);

                var s0 = AdvSimd.SignExtendWideningLower(qLo.GetLower());
                var s1 = AdvSimd.SignExtendWideningUpper(qLo);
                var s2 = AdvSimd.SignExtendWideningLower(qHi.GetLower());
                var s3 = AdvSimd.SignExtendWideningUpper(qHi);

                var vf0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s0.GetLower()));
                var vf1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s0));
                var vf2 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s1.GetLower()));
                var vf3 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s1));
                var vf4 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s2.GetLower()));
                var vf5 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s2));
                var vf6 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(s3.GetLower()));
                var vf7 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(s3));

                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr), vwd, vf0).Store(outPtr);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 4), vwd, vf1).Store(outPtr + 4);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 8), vwd, vf2).Store(outPtr + 8);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 12), vwd, vf3).Store(outPtr + 12);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 16), vwd, vf4).Store(outPtr + 16);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 20), vwd, vf5).Store(outPtr + 20);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 24), vwd, vf6).Store(outPtr + 24);
                AdvSimd.FusedMultiplyAdd(Vector128.Load(outPtr + 28), vwd, vf7).Store(outPtr + 28);
            }
            else
            {
                for (int l = 0; l < QK8_0; l++)
                {
                    outPtr[l] += wd * qs[l];
                }
            }
        }

        int fullElems = fullBlocks << 5;
        int tail = k - fullElems;
        if (tail > 0)
        {
            float wd = weight * (float)src[fullBlocks].Delta;
            sbyte* qs = src[fullBlocks].Qs;
            float* outPtr = dst + fullElems;
            for (int l = 0; l < tail; l++)
            {
                outPtr[l] += wd * qs[l];
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
