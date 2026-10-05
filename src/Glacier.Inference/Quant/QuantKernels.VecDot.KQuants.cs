namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

public static unsafe partial class QuantKernels
{
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

            if (Vector512.IsHardwareAccelerated)
            {
                var acc512 = Vector512<float>.Zero;
                for (int l = 0; l < 32; l += 16)
                {
                    var vx = Vector512.Load(x + l);
                    var vq = Vector512.Create(
                        (float)q[l + 0], (float)q[l + 1], (float)q[l + 2], (float)q[l + 3],
                        (float)q[l + 4], (float)q[l + 5], (float)q[l + 6], (float)q[l + 7],
                        (float)q[l + 8], (float)q[l + 9], (float)q[l + 10], (float)q[l + 11],
                        (float)q[l + 12], (float)q[l + 13], (float)q[l + 14], (float)q[l + 15]);
                    acc512 = Vector512.FusedMultiplyAdd(vx, vq, acc512);
                }
                blockSum = Vector512.Sum(acc512);
            }
            else if (AdvSimd.IsSupported)
            {
                var acc0 = Vector128<float>.Zero;
                var acc1 = Vector128<float>.Zero;
                for (int l = 0; l < 32; l += 8)
                {
                    var vx0 = Vector128.Load(x + l);
                    var vx1 = Vector128.Load(x + l + 4);
                    var vq0 = Vector128.Create((float)q[l + 0], (float)q[l + 1], (float)q[l + 2], (float)q[l + 3]);
                    var vq1 = Vector128.Create((float)q[l + 4], (float)q[l + 5], (float)q[l + 6], (float)q[l + 7]);
                    acc0 = AdvSimd.FusedMultiplyAdd(acc0, vx0, vq0);
                    acc1 = AdvSimd.FusedMultiplyAdd(acc1, vx1, vq1);
                }
                blockSum = Vector128.Sum(acc0 + acc1);
            }
            else if (Vector256.IsHardwareAccelerated)
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
        if (Vector512.IsHardwareAccelerated && k >= 16)
        {
            var acc0 = Vector512<float>.Zero;
            var acc1 = Vector512<float>.Zero;
            for (; i <= k - 32; i += 32)
            {
                acc0 = Vector512.FusedMultiplyAdd(Vector512.Load(x + i), Vector512.Load(row + i), acc0);
                acc1 = Vector512.FusedMultiplyAdd(Vector512.Load(x + i + 16), Vector512.Load(row + i + 16), acc1);
            }
            if (i <= k - 16)
            {
                acc0 = Vector512.FusedMultiplyAdd(Vector512.Load(x + i), Vector512.Load(row + i), acc0);
                i += 16;
            }
            sum = Vector512.Sum(acc0 + acc1);
        }
        if (Vector256.IsHardwareAccelerated && (k - i) >= 8)
        {
            var acc = Vector256<float>.Zero;
            for (; i <= k - 8; i += 8)
            {
                acc = Vector256.FusedMultiplyAdd(Vector256.Load(x + i), Vector256.Load(row + i), acc);
            }
            sum += Vector256.Sum(acc);
        }
        for (; i < k; i++)
        {
            sum += row[i] * x[i];
        }
        return sum;
    }

    /// <summary>
    /// Computes dot product between a Q5_K quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ5_K(BlockQ5_K* row, float* x, int k)
    {
        int nb = k / QK_K;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float min = (float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* ql = row[i].Qs;
            byte* qh = row[i].Qh;

            int is_idx = 0;
            byte u1 = 1;
            byte u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(is_idx + 0, scales, out byte sc0, out byte m0);
                float d1 = d * sc0;
                float m1 = min * m0;

                GetScaleMinK4(is_idx + 1, scales, out byte sc1, out byte m1_val);
                float d2 = d * sc1;
                float m2 = min * m1_val;

                float sub0 = 0f;
                float sub1 = 0f;

                if (Vector512.IsHardwareAccelerated)
                {
                    var vd1 = Vector512.Create(d1);
                    var vm1 = Vector512.Create(m1);
                    var vAcc0 = Vector512<float>.Zero;
                    for (int l = 0; l < 32; l += 16)
                    {
                        var vx = Vector512.Load(x + l);
                        var vq = Vector512.Create(
                            (float)((ql[l + 0] & 0x0F) + ((qh[l + 0] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] & 0x0F) + ((qh[l + 1] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] & 0x0F) + ((qh[l + 2] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] & 0x0F) + ((qh[l + 3] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 4] & 0x0F) + ((qh[l + 4] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 5] & 0x0F) + ((qh[l + 5] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 6] & 0x0F) + ((qh[l + 6] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 7] & 0x0F) + ((qh[l + 7] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 8] & 0x0F) + ((qh[l + 8] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 9] & 0x0F) + ((qh[l + 9] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 10] & 0x0F) + ((qh[l + 10] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 11] & 0x0F) + ((qh[l + 11] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 12] & 0x0F) + ((qh[l + 12] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 13] & 0x0F) + ((qh[l + 13] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 14] & 0x0F) + ((qh[l + 14] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 15] & 0x0F) + ((qh[l + 15] & u1) != 0 ? 16 : 0)));
                        var vVal = Vector512.FusedMultiplyAdd(vd1, vq, -vm1);
                        vAcc0 = Vector512.FusedMultiplyAdd(vVal, vx, vAcc0);
                    }
                    sub0 = Vector512.Sum(vAcc0);

                    var vd2 = Vector512.Create(d2);
                    var vm2 = Vector512.Create(m2);
                    var vAcc1 = Vector512<float>.Zero;
                    for (int l = 0; l < 32; l += 16)
                    {
                        var vx = Vector512.Load(x + l + 32);
                        var vq = Vector512.Create(
                            (float)((ql[l + 0] >> 4) + ((qh[l + 0] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] >> 4) + ((qh[l + 1] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] >> 4) + ((qh[l + 2] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] >> 4) + ((qh[l + 3] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 4] >> 4) + ((qh[l + 4] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 5] >> 4) + ((qh[l + 5] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 6] >> 4) + ((qh[l + 6] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 7] >> 4) + ((qh[l + 7] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 8] >> 4) + ((qh[l + 8] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 9] >> 4) + ((qh[l + 9] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 10] >> 4) + ((qh[l + 10] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 11] >> 4) + ((qh[l + 11] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 12] >> 4) + ((qh[l + 12] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 13] >> 4) + ((qh[l + 13] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 14] >> 4) + ((qh[l + 14] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 15] >> 4) + ((qh[l + 15] & u2) != 0 ? 16 : 0)));
                        var vVal = Vector512.FusedMultiplyAdd(vd2, vq, -vm2);
                        vAcc1 = Vector512.FusedMultiplyAdd(vVal, vx, vAcc1);
                    }
                    sub1 = Vector512.Sum(vAcc1);
                }
                else if (AdvSimd.IsSupported)
                {
                    var vd1 = Vector128.Create(d1);
                    var vm1 = Vector128.Create(m1);
                    var vAcc0 = Vector128<float>.Zero;
                    for (int l = 0; l < 32; l += 4)
                    {
                        var vx = Vector128.Load(x + l);
                        var vq = Vector128.Create(
                            (float)((ql[l + 0] & 0x0F) + ((qh[l + 0] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] & 0x0F) + ((qh[l + 1] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] & 0x0F) + ((qh[l + 2] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] & 0x0F) + ((qh[l + 3] & u1) != 0 ? 16 : 0)));
                        var vVal = AdvSimd.FusedMultiplyAdd(-vm1, vd1, vq);
                        vAcc0 = AdvSimd.FusedMultiplyAdd(vAcc0, vVal, vx);
                    }
                    sub0 = Vector128.Sum(vAcc0);

                    var vd2 = Vector128.Create(d2);
                    var vm2 = Vector128.Create(m2);
                    var vAcc1 = Vector128<float>.Zero;
                    for (int l = 0; l < 32; l += 4)
                    {
                        var vx = Vector128.Load(x + l + 32);
                        var vq = Vector128.Create(
                            (float)((ql[l + 0] >> 4) + ((qh[l + 0] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] >> 4) + ((qh[l + 1] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] >> 4) + ((qh[l + 2] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] >> 4) + ((qh[l + 3] & u2) != 0 ? 16 : 0)));
                        var vVal = AdvSimd.FusedMultiplyAdd(-vm2, vd2, vq);
                        vAcc1 = AdvSimd.FusedMultiplyAdd(vAcc1, vVal, vx);
                    }
                    sub1 = Vector128.Sum(vAcc1);
                }
                else if (Vector256.IsHardwareAccelerated)
                {
                    var vd1 = Vector256.Create(d1);
                    var vm1 = Vector256.Create(m1);
                    var vAcc0 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l);
                        var vq = Vector256.Create(
                            (float)((ql[l + 0] & 0x0F) + ((qh[l + 0] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] & 0x0F) + ((qh[l + 1] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] & 0x0F) + ((qh[l + 2] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] & 0x0F) + ((qh[l + 3] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 4] & 0x0F) + ((qh[l + 4] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 5] & 0x0F) + ((qh[l + 5] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 6] & 0x0F) + ((qh[l + 6] & u1) != 0 ? 16 : 0)),
                            (float)((ql[l + 7] & 0x0F) + ((qh[l + 7] & u1) != 0 ? 16 : 0)));
                        var vVal = (vd1 * vq) - vm1;
                        vAcc0 += vVal * vx;
                    }
                    sub0 = Vector256.Sum(vAcc0);

                    var vd2 = Vector256.Create(d2);
                    var vm2 = Vector256.Create(m2);
                    var vAcc1 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l + 32);
                        var vq = Vector256.Create(
                            (float)((ql[l + 0] >> 4) + ((qh[l + 0] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 1] >> 4) + ((qh[l + 1] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 2] >> 4) + ((qh[l + 2] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 3] >> 4) + ((qh[l + 3] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 4] >> 4) + ((qh[l + 4] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 5] >> 4) + ((qh[l + 5] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 6] >> 4) + ((qh[l + 6] & u2) != 0 ? 16 : 0)),
                            (float)((ql[l + 7] >> 4) + ((qh[l + 7] & u2) != 0 ? 16 : 0)));
                        var vVal = (vd2 * vq) - vm2;
                        vAcc1 += vVal * vx;
                    }
                    sub1 = Vector256.Sum(vAcc1);
                }
                else
                {
                    for (int l = 0; l < 32; ++l)
                    {
                        int q0 = (ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                        sub0 += (d1 * q0 - m1) * x[l];
                    }
                    for (int l = 0; l < 32; ++l)
                    {
                        int q1 = (ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                        sub1 += (d2 * q1 - m2) * x[l + 32];
                    }
                }

                sum += sub0 + sub1;
                x += 64;
                ql += 32;
                is_idx += 2;
                u1 = (byte)(u1 << 2);
                u2 = (byte)(u2 << 2);
            }
        }

        return sum;
    }

    /// <summary>
    /// Computes dot product between a Q3_K quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ3_K(BlockQ3_K* row, float* x, int k)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0F0F0F0F;

        int nb = k / QK_K;
        float sum = 0f;
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;

        for (int i = 0; i < nb; i++)
        {
            float d_all = (float)row[i].Delta;
            byte* q = row[i].Qs;
            byte* hm = row[i].Hmask;
            byte m = 1;

            Buffer.MemoryCopy(row[i].Scales, aux, 16, 12);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            int is_idx = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; ++j)
                {
                    float dl0 = d_all * (scales[is_idx++] - 32);
                    float sub0 = 0f;
                    for (int l = 0; l < 16; ++l)
                    {
                        int q0 = (sbyte)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) != 0 ? 0 : 4);
                        sub0 += q0 * x[l];
                    }
                    sum += dl0 * sub0;
                    x += 16;

                    float dl1 = d_all * (scales[is_idx++] - 32);
                    float sub1 = 0f;
                    for (int l = 0; l < 16; ++l)
                    {
                        int q1 = (sbyte)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4);
                        sub1 += q1 * x[l];
                    }
                    sum += dl1 * sub1;
                    x += 16;

                    shift += 2;
                    m = (byte)(m << 1);
                }
                q += 32;
            }
        }

        return sum;
    }

    private static readonly float[] E2M1Table = [
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    /// <summary>
    /// Computes dot product between an MXFP4 quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotMXFP4(BlockMXFP4* row, float* x, int k)
    {
        int nb = k / 32;
        float sum = 0f;

        fixed (float* lut = E2M1Table)
        {
            for (int i = 0; i < nb; i++)
            {
                float scale = MathF.ScaleB(1.0f, row[i].Scale - 127);
                byte* q = row[i].Qs;
                float blockSum = 0f;

                for (int l = 0; l < 16; ++l)
                {
                    int v0 = q[l] & 0x0F;
                    int v1 = (q[l] >> 4) & 0x0F;
                    blockSum += lut[v0] * x[l] + lut[v1] * x[l + 16];
                }

                sum += scale * blockSum;
                x += 32;
            }
        }

        return sum;
    }

    /// <summary>
    /// Non-linear 16-entry codebook for IQ4_NL and IQ4_XS (kvalues_iq4nl).
    /// </summary>
    public static readonly float[] KValuesIq4Nl =
    [
        -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f,
           1f,   13f,  25f,  38f,  53f,  69f,  89f, 113f
    ];

    /// <summary>
    /// Computes dot product between an IQ4_XS quantized row and a float vector x of length k.
    /// Super-block size is 256 elements in 136 bytes (8 sub-blocks of 32).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ4_XS(BlockIQ4_XS* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        fixed (float* kValues = KValuesIq4Nl)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                ushort scalesH = row[i].ScalesH;
                byte* scalesL = row[i].ScalesL;
                byte* qs = row[i].Qs;

                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scalesH >> (2 * ib)) & 3) << 4);
                    float dl = d * (ls - 32);

                    float subSum = 0f;
                    for (int j = 0; j < 16; j++)
                    {
                        byte q = qs[j];
                        subSum += x[j] * kValues[q & 0x0F] + x[j + 16] * kValues[q >> 4];
                    }

                    totalSum += dl * subSum;
                    x += 32;
                    qs += 16;
                }
            }
        }

        return totalSum;
    }
}
