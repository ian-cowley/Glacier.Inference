namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Hardware SIMD accelerated quantization kernels, dot products, and transformer activation functions.
/// Written in pure C# .NET 10 with zero-allocation hot paths.
/// </summary>
public static unsafe class QuantKernels
{
    private const int QK_K = 256;
    private const int QK8_0 = 32;
    private const int QK4_0 = 32;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    /// <summary>
    /// Dequantizes a row of Q4_K blocks into 32-bit floating point array.
    /// </summary>
    public static void DequantizeQ4_K(BlockQ4_K* src, float* dst, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)src[i].Delta;
            float min = (float)src[i].DeltaMin;
            byte* scales = src[i].Scales;
            byte* q = src[i].Qs;

            int is_idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(is_idx + 0, scales, out byte sc0, out byte m0);
                float d1 = d * sc0;
                float m1 = min * m0;

                GetScaleMinK4(is_idx + 1, scales, out byte sc1, out byte m1_val);
                float d2 = d * sc1;
                float m2 = min * m1_val;

                for (int l = 0; l < 32; ++l)
                {
                    *dst++ = d1 * (q[l] & 0x0F) - m1;
                }
                for (int l = 0; l < 32; ++l)
                {
                    *dst++ = d2 * (q[l] >> 4) - m2;
                }

                q += 32;
                is_idx += 2;
            }
        }
    }

    /// <summary>
    /// Computes dot product between a Q4_K quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ4_K(BlockQ4_K* row, float* x, int k)
    {
        int nb = k / QK_K;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float min = (float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* q = row[i].Qs;

            int is_idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(is_idx + 0, scales, out byte sc0, out byte m0);
                float d1 = d * sc0;
                float m1 = min * m0;

                GetScaleMinK4(is_idx + 1, scales, out byte sc1, out byte m1_val);
                float d2 = d * sc1;
                float m2 = min * m1_val;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vSubSum1 = Vector256<float>.Zero;
                    var vSubMin1 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l);
                        var vq = Vector256.Create(
                            (float)(q[l + 0] & 0x0F), (float)(q[l + 1] & 0x0F), (float)(q[l + 2] & 0x0F), (float)(q[l + 3] & 0x0F),
                            (float)(q[l + 4] & 0x0F), (float)(q[l + 5] & 0x0F), (float)(q[l + 6] & 0x0F), (float)(q[l + 7] & 0x0F));
                        vSubSum1 += vx * vq;
                        vSubMin1 += vx;
                    }
                    sum += (d1 * Vector256.Sum(vSubSum1)) - (m1 * Vector256.Sum(vSubMin1));
                    x += 32;

                    var vSubSum2 = Vector256<float>.Zero;
                    var vSubMin2 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l);
                        var vq = Vector256.Create(
                            (float)(q[l + 0] >> 4), (float)(q[l + 1] >> 4), (float)(q[l + 2] >> 4), (float)(q[l + 3] >> 4),
                            (float)(q[l + 4] >> 4), (float)(q[l + 5] >> 4), (float)(q[l + 6] >> 4), (float)(q[l + 7] >> 4));
                        vSubSum2 += vx * vq;
                        vSubMin2 += vx;
                    }
                    sum += (d2 * Vector256.Sum(vSubSum2)) - (m2 * Vector256.Sum(vSubMin2));
                    x += 32;
                }
                else
                {
                    float subSum1 = 0f;
                    float subMin1 = 0f;
                    for (int l = 0; l < 32; ++l)
                    {
                        float val_x = x[l];
                        subSum1 += (q[l] & 0x0F) * val_x;
                        subMin1 += val_x;
                    }
                    sum += (d1 * subSum1) - (m1 * subMin1);
                    x += 32;

                    float subSum2 = 0f;
                    float subMin2 = 0f;
                    for (int l = 0; l < 32; ++l)
                    {
                        float val_x = x[l];
                        subSum2 += (q[l] >> 4) * val_x;
                        subMin2 += val_x;
                    }
                    sum += (d2 * subSum2) - (m2 * subMin2);
                    x += 32;
                }

                q += 32;
                is_idx += 2;
            }
        }

        return sum;
    }

    /// <summary>
    /// Dequantizes a row of Q6_K blocks into 32-bit floating point array.
    /// </summary>
    public static void DequantizeQ6_K(BlockQ6_K* src, float* dst, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)src[i].Delta;
            byte* ql = src[i].Ql;
            byte* qh = src[i].Qh;
            sbyte* sc = src[i].Scales;

            for (int n = 0; n < QK_K; n += 128)
            {
                for (int l = 0; l < 32; ++l)
                {
                    int is_idx = l / 16;
                    sbyte q1 = (sbyte)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4) - 32);
                    sbyte q2 = (sbyte)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4) - 32);
                    sbyte q3 = (sbyte)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32);
                    sbyte q4 = (sbyte)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32);

                    dst[l + 0] = d * sc[is_idx + 0] * q1;
                    dst[l + 32] = d * sc[is_idx + 2] * q2;
                    dst[l + 64] = d * sc[is_idx + 4] * q3;
                    dst[l + 96] = d * sc[is_idx + 6] * q4;
                }

                dst += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    /// <summary>
    /// Computes dot product between a Q6_K quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ6_K(BlockQ6_K* row, float* x, int k)
    {
        int nb = k / QK_K;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            byte* ql = row[i].Ql;
            byte* qh = row[i].Qh;
            sbyte* sc = row[i].Scales;

            for (int n = 0; n < QK_K; n += 128)
            {
                float sub0 = 0f, sub1 = 0f, sub2 = 0f, sub3 = 0f;
                float sub4 = 0f, sub5 = 0f, sub6 = 0f, sub7 = 0f;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vSub0 = Vector256<float>.Zero;
                    var vSub2 = Vector256<float>.Zero;
                    var vSub4 = Vector256<float>.Zero;
                    var vSub6 = Vector256<float>.Zero;

                    for (int l = 0; l < 16; l += 8)
                    {
                        var vx0 = Vector256.Load(x + l + 0);
                        var vx2 = Vector256.Load(x + l + 32);
                        var vx4 = Vector256.Load(x + l + 64);
                        var vx6 = Vector256.Load(x + l + 96);

                        var vq1 = Vector256.Create(
                            (float)(((ql[l + 0] & 0x0F) | (((qh[l + 0] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] & 0x0F) | (((qh[l + 1] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] & 0x0F) | (((qh[l + 2] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] & 0x0F) | (((qh[l + 3] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 4] & 0x0F) | (((qh[l + 4] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 5] & 0x0F) | (((qh[l + 5] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 6] & 0x0F) | (((qh[l + 6] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 7] & 0x0F) | (((qh[l + 7] >> 0) & 3) << 4)) - 32));

                        var vq2 = Vector256.Create(
                            (float)(((ql[l + 32] & 0x0F) | (((qh[l + 0] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] & 0x0F) | (((qh[l + 1] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] & 0x0F) | (((qh[l + 2] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] & 0x0F) | (((qh[l + 3] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 36] & 0x0F) | (((qh[l + 4] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 37] & 0x0F) | (((qh[l + 5] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 38] & 0x0F) | (((qh[l + 6] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 39] & 0x0F) | (((qh[l + 7] >> 2) & 3) << 4)) - 32));

                        var vq3 = Vector256.Create(
                            (float)(((ql[l + 0] >> 4) | (((qh[l + 0] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] >> 4) | (((qh[l + 1] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] >> 4) | (((qh[l + 2] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] >> 4) | (((qh[l + 3] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 4] >> 4) | (((qh[l + 4] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 5] >> 4) | (((qh[l + 5] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 6] >> 4) | (((qh[l + 6] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 7] >> 4) | (((qh[l + 7] >> 4) & 3) << 4)) - 32));

                        var vq4 = Vector256.Create(
                            (float)(((ql[l + 32] >> 4) | (((qh[l + 0] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] >> 4) | (((qh[l + 1] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] >> 4) | (((qh[l + 2] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] >> 4) | (((qh[l + 3] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 36] >> 4) | (((qh[l + 4] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 37] >> 4) | (((qh[l + 5] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 38] >> 4) | (((qh[l + 6] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 39] >> 4) | (((qh[l + 7] >> 6) & 3) << 4)) - 32));

                        vSub0 += vq1 * vx0;
                        vSub2 += vq2 * vx2;
                        vSub4 += vq3 * vx4;
                        vSub6 += vq4 * vx6;
                    }
                    sub0 = Vector256.Sum(vSub0);
                    sub2 = Vector256.Sum(vSub2);
                    sub4 = Vector256.Sum(vSub4);
                    sub6 = Vector256.Sum(vSub6);

                    var vSub1 = Vector256<float>.Zero;
                    var vSub3 = Vector256<float>.Zero;
                    var vSub5 = Vector256<float>.Zero;
                    var vSub7 = Vector256<float>.Zero;

                    for (int l = 16; l < 32; l += 8)
                    {
                        var vx0 = Vector256.Load(x + l + 0);
                        var vx2 = Vector256.Load(x + l + 32);
                        var vx4 = Vector256.Load(x + l + 64);
                        var vx6 = Vector256.Load(x + l + 96);

                        var vq1 = Vector256.Create(
                            (float)(((ql[l + 0] & 0x0F) | (((qh[l + 0] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] & 0x0F) | (((qh[l + 1] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] & 0x0F) | (((qh[l + 2] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] & 0x0F) | (((qh[l + 3] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 4] & 0x0F) | (((qh[l + 4] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 5] & 0x0F) | (((qh[l + 5] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 6] & 0x0F) | (((qh[l + 6] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 7] & 0x0F) | (((qh[l + 7] >> 0) & 3) << 4)) - 32));

                        var vq2 = Vector256.Create(
                            (float)(((ql[l + 32] & 0x0F) | (((qh[l + 0] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] & 0x0F) | (((qh[l + 1] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] & 0x0F) | (((qh[l + 2] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] & 0x0F) | (((qh[l + 3] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 36] & 0x0F) | (((qh[l + 4] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 37] & 0x0F) | (((qh[l + 5] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 38] & 0x0F) | (((qh[l + 6] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 39] & 0x0F) | (((qh[l + 7] >> 2) & 3) << 4)) - 32));

                        var vq3 = Vector256.Create(
                            (float)(((ql[l + 0] >> 4) | (((qh[l + 0] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] >> 4) | (((qh[l + 1] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] >> 4) | (((qh[l + 2] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] >> 4) | (((qh[l + 3] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 4] >> 4) | (((qh[l + 4] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 5] >> 4) | (((qh[l + 5] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 6] >> 4) | (((qh[l + 6] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 7] >> 4) | (((qh[l + 7] >> 4) & 3) << 4)) - 32));

                        var vq4 = Vector256.Create(
                            (float)(((ql[l + 32] >> 4) | (((qh[l + 0] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] >> 4) | (((qh[l + 1] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] >> 4) | (((qh[l + 2] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] >> 4) | (((qh[l + 3] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 36] >> 4) | (((qh[l + 4] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 37] >> 4) | (((qh[l + 5] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 38] >> 4) | (((qh[l + 6] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 39] >> 4) | (((qh[l + 7] >> 6) & 3) << 4)) - 32));

                        vSub1 += vq1 * vx0;
                        vSub3 += vq2 * vx2;
                        vSub5 += vq3 * vx4;
                        vSub7 += vq4 * vx6;
                    }
                    sub1 = Vector256.Sum(vSub1);
                    sub3 = Vector256.Sum(vSub3);
                    sub5 = Vector256.Sum(vSub5);
                    sub7 = Vector256.Sum(vSub7);
                }
                else
                {
                    for (int l = 0; l < 16; ++l)
                    {
                        int q1 = (ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4) - 32;
                        int q2 = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4) - 32;
                        int q3 = (ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32;
                        int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32;

                        sub0 += q1 * x[l + 0];
                        sub2 += q2 * x[l + 32];
                        sub4 += q3 * x[l + 64];
                        sub6 += q4 * x[l + 96];
                    }

                    for (int l = 16; l < 32; ++l)
                    {
                        int q1 = (ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4) - 32;
                        int q2 = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4) - 32;
                        int q3 = (ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32;
                        int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32;

                        sub1 += q1 * x[l + 0];
                        sub3 += q2 * x[l + 32];
                        sub5 += q3 * x[l + 64];
                        sub7 += q4 * x[l + 96];
                    }
                }

                sum += d * (
                    sc[0] * sub0 + sc[1] * sub1 +
                    sc[2] * sub2 + sc[3] * sub3 +
                    sc[4] * sub4 + sc[5] * sub5 +
                    sc[6] * sub6 + sc[7] * sub7
                );

                x += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }

        return sum;
    }

    /// <summary>
    /// Computes dot product between a Q8_0 quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ8_0(BlockQ8_0* row, float* x, int k)
    {
        int nb = k / QK8_0;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            sbyte* q = row[i].Qs;
            float blockSum = 0f;

            if (Vector256.IsHardwareAccelerated)
            {
                for (int l = 0; l < 32; l += 8)
                {
                    var vx = Vector256.Load(x + l);
                    var vq = Vector256.Create(
                        (float)q[l + 0], (float)q[l + 1], (float)q[l + 2], (float)q[l + 3],
                        (float)q[l + 4], (float)q[l + 5], (float)q[l + 6], (float)q[l + 7]);
                    blockSum += Vector256.Dot(vx, vq);
                }
            }
            else
            {
                for (int l = 0; l < 32; ++l)
                {
                    blockSum += q[l] * x[l];
                }
            }

            sum += d * blockSum;
            x += 32;
        }

        return sum;
    }

    /// <summary>
    /// Computes dot product between a Q4_0 quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ4_0(BlockQ4_0* row, float* x, int k)
    {
        int nb = k / QK4_0;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            byte* q = row[i].Qs;
            float blockSum = 0f;

            for (int l = 0; l < 16; ++l)
            {
                int q0 = (q[l] & 0x0F) - 8;
                int q1 = (q[l] >> 4) - 8;
                blockSum += q0 * x[l] + q1 * x[l + 16];
            }

            sum += d * blockSum;
            x += 32;
        }

        return sum;
    }

    /// <summary>
    /// Computes dot product between FP16 row and float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotF16(Half* row, float* x, int k)
    {
        float sum = 0f;
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vecLimit = k - 8;
            for (; i <= vecLimit; i += 8)
            {
                var vx = Vector256.Load(x + i);
                var vr = Vector256.Create(
                    (float)row[i + 0], (float)row[i + 1], (float)row[i + 2], (float)row[i + 3],
                    (float)row[i + 4], (float)row[i + 5], (float)row[i + 6], (float)row[i + 7]);
                sum += Vector256.Dot(vx, vr);
            }
        }
        for (; i < k; i++)
        {
            sum += (float)row[i] * x[i];
        }
        return sum;
    }

    /// <summary>
    /// Computes dot product between FP32 row and float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotF32(float* row, float* x, int k)
    {
        float sum = 0f;
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vecLimit = k - 8;
            var acc = Vector256<float>.Zero;
            for (; i <= vecLimit; i += 8)
            {
                var vx = Vector256.Load(x + i);
                var vr = Vector256.Load(row + i);
                acc += vx * vr;
            }
            sum = Vector256.Sum(acc);
        }
        for (; i < k; i++)
        {
            sum += row[i] * x[i];
        }
        return sum;
    }

    /// <summary>
    /// Multiplies a quantized matrix W [nRows, nCols] by vector x [nCols] producing y [nRows].
    /// Automatically multithreads row calculations across all available CPU cores.
    /// </summary>
    public static void MatVecMul(GgufType type, byte* weightData, float* x, float* y, int nCols, int nRows)
    {
        int rowBytes = (int)GgufTypes.GetRowBytes(type, nCols);

        Parallel.For(0, nRows, i =>
        {
            byte* rowPtr = weightData + (long)i * rowBytes;
            float dot = type switch
            {
                GgufType.Q4_K => VecDotQ4_K((BlockQ4_K*)rowPtr, x, nCols),
                GgufType.Q6_K => VecDotQ6_K((BlockQ6_K*)rowPtr, x, nCols),
                GgufType.Q8_0 => VecDotQ8_0((BlockQ8_0*)rowPtr, x, nCols),
                GgufType.Q4_0 => VecDotQ4_0((BlockQ4_0*)rowPtr, x, nCols),
                GgufType.F16 => VecDotF16((Half*)rowPtr, x, nCols),
                GgufType.F32 => VecDotF32((float*)rowPtr, x, nCols),
                _ => throw new NotSupportedException($"Quantization type {type} is not supported in hardware GEMV kernels.")
            };
            y[i] = dot;
        });
    }

    /// <summary>
    /// Root Mean Square Normalization: dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RMSNorm(float* x, float* weight, float* dst, int size, float eps)
    {
        float sumSq = 0f;
        int i = 0;

        if (Vector256.IsHardwareAccelerated)
        {
            int vecLimit = size - 8;
            var acc = Vector256<float>.Zero;
            for (; i <= vecLimit; i += 8)
            {
                var v = Vector256.Load(x + i);
                acc += v * v;
            }
            sumSq = Vector256.Sum(acc);
        }

        for (; i < size; i++)
        {
            float v = x[i];
            sumSq += v * v;
        }

        float rms = 1.0f / MathF.Sqrt((sumSq / size) + eps);

        i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var vRms = Vector256.Create(rms);
            int vecLimit = size - 8;
            for (; i <= vecLimit; i += 8)
            {
                var vx = Vector256.Load(x + i);
                var vw = Vector256.Load(weight + i);
                var vout = vx * vRms * vw;
                vout.Store(dst + i);
            }
        }

        for (; i < size; i++)
        {
            dst[i] = x[i] * rms * weight[i];
        }
    }

    /// <summary>
    /// Applies Rotary Position Embedding (RoPE NeOX style) in-place to Q and K tensors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RoPE(
        float* q, float* k,
        int nHeadsQ, int nHeadsKv,
        int headDim, int pos,
        float freqBase, float freqScale = 1.0f)
    {
        int halfDim = headDim / 2;

        Span<float> cosTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];

        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(freqBase, (float)(2 * i) / headDim);
            float theta = pos * freq * freqScale;
            cosTable[i] = MathF.Cos(theta);
            sinTable[i] = MathF.Sin(theta);
        }

        fixed (float* pCos = cosTable, pSin = sinTable)
        {
            // Apply to Q heads
            for (int h = 0; h < nHeadsQ; h++)
            {
                float* head = q + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }

            // Apply to K heads
            for (int h = 0; h < nHeadsKv; h++)
            {
                float* head = k + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }
        }
    }

    /// <summary>
    /// SwiGLU activation function: dst[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SwiGLU(float* gate, float* up, float* dst, int size)
    {
        for (int i = 0; i < size; i++)
        {
            float g = gate[i];
            float silu = g / (1.0f + MathF.Exp(-g));
            dst[i] = silu * up[i];
        }
    }

    /// <summary>
    /// In-place numerically stable Softmax over a vector of logits.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Softmax(float* x, int size)
    {
        if (size <= 0) return;

        float maxVal = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > maxVal) maxVal = x[i];
        }

        float sumExp = 0f;
        for (int i = 0; i < size; i++)
        {
            float expVal = MathF.Exp(x[i] - maxVal);
            x[i] = expVal;
            sumExp += expVal;
        }

        float invSum = 1.0f / sumExp;
        for (int i = 0; i < size; i++)
        {
            x[i] *= invSum;
        }
    }

    /// <summary>
    /// Copies an embedding row for token into destination float buffer.
    /// Supports Q4_K, Q6_K, Q8_0, F16, and F32 embeddings.
    /// </summary>
    public static void ExtractEmbedding(GgufType type, byte* embdData, int tokenId, float* dst, int embeddingDim)
    {
        int rowBytes = (int)GgufTypes.GetRowBytes(type, embeddingDim);
        byte* rowPtr = embdData + (long)tokenId * rowBytes;

        switch (type)
        {
            case GgufType.Q4_K:
                DequantizeQ4_K((BlockQ4_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q6_K:
                DequantizeQ6_K((BlockQ6_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q8_0:
                for (int i = 0; i < embeddingDim / 32; i++)
                {
                    BlockQ8_0* b = (BlockQ8_0*)(rowPtr + i * sizeof(BlockQ8_0));
                    float d = (float)b->Delta;
                    for (int l = 0; l < 32; l++)
                    {
                        dst[i * 32 + l] = d * b->Qs[l];
                    }
                }
                break;
            case GgufType.F16:
                Half* h = (Half*)rowPtr;
                for (int i = 0; i < embeddingDim; i++) dst[i] = (float)h[i];
                break;
            case GgufType.F32:
                Buffer.MemoryCopy(rowPtr, dst, embeddingDim * 4, embeddingDim * 4);
                break;
            default:
                throw new NotSupportedException($"Embedding quantization type {type} not supported.");
        }
    }
}
