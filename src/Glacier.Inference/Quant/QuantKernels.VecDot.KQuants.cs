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
                var q32 = Vector256.Load(q);
                var vq0 = WidenSByteToFloat512(q32.GetLower());
                var vq1 = WidenSByteToFloat512(q32.GetUpper());
                var acc512 = Vector512.FusedMultiplyAdd(Vector512.Load(x), vq0, Vector512.Load(x + 16) * vq1);
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
    /// Hardware-accelerated with Vector512, AdvSimd, Vector256, and scalar fallback.
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

            if (Vector512.IsHardwareAccelerated)
            {
                var raw = Vector128.Load(q);
                var mask0F = Vector128.Create((byte)0x0F);
                var lo = raw & mask0F;
                var hi = Vector128.ShiftRightLogical(raw.AsUInt16(), 4).AsByte() & mask0F;
                var v8 = Vector512.Create(8.0f);
                var vqLow = WidenUInt8ToFloat512(lo) - v8;
                var vqHigh = WidenUInt8ToFloat512(hi) - v8;
                var vxLow = Vector512.Load(x);
                var vxHigh = Vector512.Load(x + 16);
                var vAcc = Vector512.FusedMultiplyAdd(vxLow, vqLow, vxHigh * vqHigh);
                blockSum = Vector512.Sum(vAcc);
            }
            else if (AdvSimd.IsSupported)
            {
                var vAcc = Vector128<float>.Zero;
                for (int l = 0; l < 16; l += 4)
                {
                    var vxLow = Vector128.Load(x + l);
                    var vxHigh = Vector128.Load(x + l + 16);
                    var vqLow = Vector128.Create(
                        (float)((q[l + 0] & 0x0F) - 8), (float)((q[l + 1] & 0x0F) - 8),
                        (float)((q[l + 2] & 0x0F) - 8), (float)((q[l + 3] & 0x0F) - 8));
                    var vqHigh = Vector128.Create(
                        (float)((q[l + 0] >> 4) - 8), (float)((q[l + 1] >> 4) - 8),
                        (float)((q[l + 2] >> 4) - 8), (float)((q[l + 3] >> 4) - 8));
                    vAcc = AdvSimd.FusedMultiplyAdd(vAcc, vxLow, vqLow);
                    vAcc = AdvSimd.FusedMultiplyAdd(vAcc, vxHigh, vqHigh);
                }
                blockSum = Vector128.Sum(vAcc);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
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

                var vxLow0 = Vector256.Load(x + 0);
                var vxLow1 = Vector256.Load(x + 8);
                var vxHigh0 = Vector256.Load(x + 16);
                var vxHigh1 = Vector256.Load(x + 24);

                blockSum = Vector256.Dot(vxLow0, vqLow0) + Vector256.Dot(vxLow1, vqLow1) +
                           Vector256.Dot(vxHigh0, vqHigh0) + Vector256.Dot(vxHigh1, vqHigh1);
            }
            else
            {
                for (int l = 0; l < 16; ++l)
                {
                    int q0 = (q[l] & 0x0F) - 8;
                    int q1 = (q[l] >> 4) - 8;
                    blockSum += q0 * x[l] + q1 * x[l + 16];
                }
            }

            sum += d * blockSum;
            x += 32;
        }

        return sum;
    }

    /// <summary>
    /// Computes dot product between FP16 row and float vector x of length k.
    /// Hardware-accelerated with Vector512, Vector256 (Widen / ConvertToSingle), and scalar fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotF16(Half* row, float* x, int k)
    {
        float sum = 0f;
        int i = 0;

        if (Vector256.IsHardwareAccelerated && k >= 16)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            int limit16 = k - 16;
            for (; i <= limit16; i += 16)
            {
                var vr0 = Vector256.Create(
                    (float)row[i + 0], (float)row[i + 1], (float)row[i + 2], (float)row[i + 3],
                    (float)row[i + 4], (float)row[i + 5], (float)row[i + 6], (float)row[i + 7]);
                var vr1 = Vector256.Create(
                    (float)row[i + 8], (float)row[i + 9], (float)row[i + 10], (float)row[i + 11],
                    (float)row[i + 12], (float)row[i + 13], (float)row[i + 14], (float)row[i + 15]);

                var vx0 = Vector256.Load(x + i);
                var vx1 = Vector256.Load(x + i + 8);

                if (Fma.IsSupported)
                {
                    acc0 = Fma.MultiplyAdd(vx0, vr0, acc0);
                    acc1 = Fma.MultiplyAdd(vx1, vr1, acc1);
                }
                else
                {
                    acc0 += vx0 * vr0;
                    acc1 += vx1 * vr1;
                }
            }
            sum += Vector256.Sum(acc0 + acc1);
        }

        if (Vector256.IsHardwareAccelerated && (k - i) >= 8)
        {
            var vr = Vector256.Create(
                (float)row[i + 0], (float)row[i + 1], (float)row[i + 2], (float)row[i + 3],
                (float)row[i + 4], (float)row[i + 5], (float)row[i + 6], (float)row[i + 7]);
            var vx = Vector256.Load(x + i);
            sum += Vector256.Dot(vx, vr);
            i += 8;
        }

        for (; i < k; i++)
        {
            sum += (float)row[i] * x[i];
        }
        return sum;
    }

    /// <summary>
    /// Computes dot product between IEEE BF16 row and float vector x of length k.
    /// Hardware-accelerated with AVX2/FMA BF16 widening + scalar fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotBF16(ushort* row, float* x, int k)
    {
        float sum = 0f;
        int i = 0;

        if (Avx2.IsSupported && k >= 16)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            int limit16 = k - 16;
            for (; i <= limit16; i += 16)
            {
                var vr0 = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Vector128.Load(row + i)), 16).AsSingle();
                var vr1 = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Vector128.Load(row + i + 8)), 16).AsSingle();
                var vx0 = Vector256.Load(x + i);
                var vx1 = Vector256.Load(x + i + 8);
                if (Fma.IsSupported)
                {
                    acc0 = Fma.MultiplyAdd(vx0, vr0, acc0);
                    acc1 = Fma.MultiplyAdd(vx1, vr1, acc1);
                }
                else
                {
                    acc0 += vx0 * vr0;
                    acc1 += vx1 * vr1;
                }
            }
            sum += Vector256.Sum(acc0 + acc1);
        }

        if (Avx2.IsSupported && (k - i) >= 8)
        {
            var vr = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Vector128.Load(row + i)), 16).AsSingle();
            var vx = Vector256.Load(x + i);
            sum += Vector256.Dot(vx, vr);
            i += 8;
        }

        for (; i < k; i++)
        {
            sum += BitConverter.UInt32BitsToSingle((uint)row[i] << 16) * x[i];
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

    private static readonly float[] Ue4M3Table = BuildUe4M3Table();

    private static float[] BuildUe4M3Table()
    {
        var table = new float[256];
        for (int x = 0; x < 256; x++)
        {
            if (x == 0 || x == 0x7F)
            {
                table[x] = 0.0f;
                continue;
            }
            int exp = (x >> 3) & 0xF;
            int man = x & 0x7;
            table[x] = exp == 0
                ? MathF.ScaleB((float)man, -9)
                : MathF.ScaleB(1.0f + man / 8.0f, exp - 7);
        }
        return table;
    }

    private static readonly float[] Ue4M3HalfTable = BuildUe4M3HalfTable();

    private static float[] BuildUe4M3HalfTable()
    {
        var table = new float[256];
        for (int x = 0; x < 256; x++)
        {
            table[x] = Ue4M3Table[x] * 0.5f;
        }
        return table;
    }

    /// <summary>
    /// Computes dot product between an NVFP4 (Type 40) quantized row and a float vector x of length k.
    /// 64 elements per block = 4 sub-blocks of 16 with per-sub-block UE4M3 scales.
    /// Hardware-accelerated with AVX-512 / AVX2+FMA PSHUFB nibble LUT decoding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotNVFP4(BlockNVFP4* row, float* x, int k)
    {
        int nb = k / 64;

        if (Avx512F.IsSupported && Avx2.IsSupported)
        {
            var vLut128 = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12).AsByte();
            var vLut256 = Vector256.Create(vLut128, vLut128);
            var vMask256 = Vector256.Create((byte)0x0F);
            var vAcc0 = Vector512<float>.Zero;
            var vAcc1 = Vector512<float>.Zero;

            fixed (float* halfScaleLut = Ue4M3HalfTable)
            {
                for (int i = 0; i < nb; i++)
                {
                    byte* scales = row[i].Scales;
                    var raw = Vector256.Load(row[i].Qs);
                    var lo = Avx2.And(raw, vMask256);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), vMask256);
                    var wLo = Avx2.Shuffle(vLut256, lo).AsSByte();
                    var wHi = Avx2.Shuffle(vLut256, hi).AsSByte();

                    var s0_s2 = Avx2.UnpackLow(wLo.AsUInt64(), wHi.AsUInt64()).AsSByte();
                    var s1_s3 = Avx2.UnpackHigh(wLo.AsUInt64(), wHi.AsUInt64()).AsSByte();

                    var wf0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s0_s2.GetLower()));
                    var wf1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s1_s3.GetLower()));
                    var wf2 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s0_s2.GetUpper()));
                    var wf3 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s1_s3.GetUpper()));

                    vAcc0 = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf0, Vector512.Create(halfScaleLut[scales[0]])), Vector512.Load(x + 0), vAcc0);
                    vAcc1 = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf1, Vector512.Create(halfScaleLut[scales[1]])), Vector512.Load(x + 16), vAcc1);
                    vAcc0 = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf2, Vector512.Create(halfScaleLut[scales[2]])), Vector512.Load(x + 32), vAcc0);
                    vAcc1 = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf3, Vector512.Create(halfScaleLut[scales[3]])), Vector512.Load(x + 48), vAcc1);
                    x += 64;
                }
            }

            return Vector512.Sum(vAcc0 + vAcc1);
        }

        if (Avx512F.IsSupported && Ssse3.IsSupported)
        {
            var vLut = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12).AsByte();
            var vMask = Vector128.Create((byte)0x0F);
            var vAcc = Vector512<float>.Zero;

            fixed (float* halfScaleLut = Ue4M3HalfTable)
            {
                for (int i = 0; i < nb; i++)
                {
                    byte* scales = row[i].Scales;
                    byte* qs = row[i].Qs;
                    for (int s = 0; s < 4; s++)
                    {
                        float scale = halfScaleLut[scales[s]];
                        ulong packed8 = *(ulong*)(qs + s * 8);
                        var raw = Vector128.CreateScalarUnsafe(packed8).AsByte();
                        var lo = Sse2.And(raw, vMask);
                        var hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), vMask);
                        var idx16 = Sse2.UnpackLow(lo.AsUInt64(), hi.AsUInt64()).AsByte();
                        var w16 = Ssse3.Shuffle(vLut, idx16).AsSByte();
                        var wf16 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(w16));
                        var vx16 = Vector512.Load(x + s * 16);
                        vAcc = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf16, Vector512.Create(scale)), vx16, vAcc);
                    }
                    x += 64;
                }
            }

            return Vector512.Sum(vAcc);
        }

        if (Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported)
        {
            var vLut = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12).AsByte();
            var vMask = Vector128.Create((byte)0x0F);
            var vAccLo = Vector256<float>.Zero;
            var vAccHi = Vector256<float>.Zero;

            fixed (float* halfScaleLut = Ue4M3HalfTable)
            {
                for (int i = 0; i < nb; i++)
                {
                    byte* scales = row[i].Scales;
                    byte* qs = row[i].Qs;
                    for (int s = 0; s < 4; s++)
                    {
                        var vScale = Vector256.Create(halfScaleLut[scales[s]]);
                        ulong packed8 = *(ulong*)(qs + s * 8);
                        var raw = Vector128.CreateScalarUnsafe(packed8).AsByte();
                        var lo = Sse2.And(raw, vMask);
                        var hi = Sse2.And(Sse2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), vMask);
                        var idx16 = Sse2.UnpackLow(lo.AsUInt64(), hi.AsUInt64()).AsByte();
                        var w16 = Ssse3.Shuffle(vLut, idx16).AsSByte();

                        var wfLo = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(w16));
                        var wfHi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(w16, 8)));

                        float* xSub = x + s * 16;
                        vAccLo = Fma.MultiplyAdd(Avx.Multiply(wfLo, vScale), Vector256.Load(xSub), vAccLo);
                        vAccHi = Fma.MultiplyAdd(Avx.Multiply(wfHi, vScale), Vector256.Load(xSub + 8), vAccHi);
                    }
                    x += 64;
                }
            }

            return Vector256.Sum(Avx.Add(vAccLo, vAccHi));
        }

        float sum = 0f;
        fixed (float* lut = E2M1Table)
        fixed (float* scaleLut = Ue4M3Table)
        {
            for (int i = 0; i < nb; i++)
            {
                byte* scales = row[i].Scales;
                byte* qs = row[i].Qs;
                for (int s = 0; s < 4; s++)
                {
                    float scale = scaleLut[scales[s]];
                    byte* qSub = qs + s * 8;
                    float* xSub = x + s * 16;
                    float subSum = 0f;

                    for (int j = 0; j < 8; j++)
                    {
                        byte packed = qSub[j];
                        subSum += lut[packed & 0x0F] * xSub[j] + lut[(packed >> 4) & 0x0F] * xSub[j + 8];
                    }

                    sum += scale * subSum;
                }
                x += 64;
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateNvfp4BlockAvx512(
        BlockNVFP4* blk,
        float* halfScaleLut,
        Vector256<byte> vLut256,
        Vector256<byte> vMask256,
        Vector512<float> vx0,
        Vector512<float> vx1,
        Vector512<float> vx2,
        Vector512<float> vx3,
        ref Vector512<float> acc)
    {
        byte* scales = blk->Scales;
        var raw = Vector256.Load(blk->Qs);
        var lo = Avx2.And(raw, vMask256);
        var hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), vMask256);
        var wLo = Avx2.Shuffle(vLut256, lo).AsSByte();
        var wHi = Avx2.Shuffle(vLut256, hi).AsSByte();

        var s0_s2 = Avx2.UnpackLow(wLo.AsUInt64(), wHi.AsUInt64()).AsSByte();
        var s1_s3 = Avx2.UnpackHigh(wLo.AsUInt64(), wHi.AsUInt64()).AsSByte();

        var wf0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s0_s2.GetLower()));
        var wf1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s1_s3.GetLower()));
        var wf2 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s0_s2.GetUpper()));
        var wf3 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(s1_s3.GetUpper()));

        acc = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf0, Vector512.Create(halfScaleLut[scales[0]])), vx0, acc);
        acc = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf1, Vector512.Create(halfScaleLut[scales[1]])), vx1, acc);
        acc = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf2, Vector512.Create(halfScaleLut[scales[2]])), vx2, acc);
        acc = Avx512F.FusedMultiplyAdd(Avx512F.Multiply(wf3, Vector512.Create(halfScaleLut[scales[3]])), vx3, acc);
    }

    /// <summary>
    /// Computes 4 concurrent NVFP4 dot products against a shared vector x, reusing loaded x registers 4x.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void VecDotNVFP4x4(
        BlockNVFP4* r0,
        BlockNVFP4* r1,
        BlockNVFP4* r2,
        BlockNVFP4* r3,
        float* x,
        int k,
        out float s0,
        out float s1,
        out float s2,
        out float s3)
    {
        if (Avx512F.IsSupported && Avx2.IsSupported)
        {
            int nb = k / 64;
            var vLut128 = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12).AsByte();
            var vLut256 = Vector256.Create(vLut128, vLut128);
            var vMask256 = Vector256.Create((byte)0x0F);

            var acc0 = Vector512<float>.Zero;
            var acc1 = Vector512<float>.Zero;
            var acc2 = Vector512<float>.Zero;
            var acc3 = Vector512<float>.Zero;

            fixed (float* halfScaleLut = Ue4M3HalfTable)
            {
                for (int i = 0; i < nb; i++)
                {
                    var vx0 = Vector512.Load(x + 0);
                    var vx1 = Vector512.Load(x + 16);
                    var vx2 = Vector512.Load(x + 32);
                    var vx3 = Vector512.Load(x + 48);

                    AccumulateNvfp4BlockAvx512(r0 + i, halfScaleLut, vLut256, vMask256, vx0, vx1, vx2, vx3, ref acc0);
                    AccumulateNvfp4BlockAvx512(r1 + i, halfScaleLut, vLut256, vMask256, vx0, vx1, vx2, vx3, ref acc1);
                    AccumulateNvfp4BlockAvx512(r2 + i, halfScaleLut, vLut256, vMask256, vx0, vx1, vx2, vx3, ref acc2);
                    AccumulateNvfp4BlockAvx512(r3 + i, halfScaleLut, vLut256, vMask256, vx0, vx1, vx2, vx3, ref acc3);
                    x += 64;
                }
            }

            s0 = Vector512.Sum(acc0);
            s1 = Vector512.Sum(acc1);
            s2 = Vector512.Sum(acc2);
            s3 = Vector512.Sum(acc3);
            return;
        }

        s0 = VecDotNVFP4(r0, x, k);
        s1 = VecDotNVFP4(r1, x, k);
        s2 = VecDotNVFP4(r2, x, k);
        s3 = VecDotNVFP4(r3, x, k);
    }

    public static bool IsNvfp4VnniSupported
        => Avx512BW.IsSupported && Avx512F.IsSupported && Avx2.IsSupported;

    /// <summary>
    /// Quantizes an FP32 activation vector x[0..k) into BlockQ8NVFP4 blocks aligned for AVX-512 VNNI VPDPBUSD.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeXForNVFP4(float* x, BlockQ8NVFP4* dst, int k)
    {
        int nb = k / 64;
        for (int i = 0; i < nb; i++)
        {
            BlockQ8NVFP4* blk = dst + i;
            for (int b = 0; b < 16; b++) blk->Bias[b] = 0.0f;

            for (int s = 0; s < 4; s++)
            {
                float* xs = x + i * 64 + s * 16;
                float amax = 0.0f;
                for (int j = 0; j < 16; j++)
                {
                    float av = MathF.Abs(xs[j]);
                    if (av > amax) amax = av;
                }

                float dx = amax / 127.0f;
                float invDx = amax > 1e-30f ? 127.0f / amax : 0.0f;
                int sumQ = 0;
                for (int j = 0; j < 8; j++)
                {
                    int qLo = Math.Clamp((int)MathF.Round(xs[j] * invDx), -127, 127);
                    int qHi = Math.Clamp((int)MathF.Round(xs[j + 8] * invDx), -127, 127);
                    blk->Qs[s * 8 + j] = (sbyte)qLo;
                    blk->Qs[32 + s * 8 + j] = (sbyte)qHi;
                    sumQ += qLo + qHi;
                }

                int dOff = s * 2;
                blk->Dx[dOff] = dx;
                blk->Dx[dOff + 1] = dx;
                blk->Dx[8 + dOff] = dx;
                blk->Dx[8 + dOff + 1] = dx;
                blk->Bias[dOff] = 12.0f * sumQ;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateNvfp4BlockVnni(
        BlockNVFP4* blk,
        float* halfScaleLut,
        Vector512<byte> vLutU8,
        Vector256<byte> vMask256,
        Vector512<short> vOnes16,
        Vector512<sbyte> vQx,
        Vector512<float> vDx,
        Vector512<float> vBias,
        ref Vector512<float> acc)
    {
        byte* scales = blk->Scales;
        var raw = Vector256.Load(blk->Qs);
        var lo = Avx2.And(raw, vMask256);
        var hi = Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), vMask256);
        var w64 = Avx512BW.Shuffle(vLutU8, Vector512.Create(lo, hi));
        var prod16 = Avx512BW.MultiplyAddAdjacent(w64, vQx);
        var dotI32 = Avx512BW.MultiplyAddAdjacent(prod16, vOnes16);
        var dotF = Avx512F.Subtract(Avx512F.ConvertToVector512Single(dotI32), vBias);

        var vW128 = Vector128.Create(
            halfScaleLut[scales[0]],
            halfScaleLut[scales[1]],
            halfScaleLut[scales[2]],
            halfScaleLut[scales[3]]);
        var vW256 = Vector256.Create(Sse.UnpackLow(vW128, vW128), Sse.UnpackHigh(vW128, vW128));
        var vScale = Avx512F.Multiply(Vector512.Create(vW256, vW256), vDx);
        acc = Avx512F.FusedMultiplyAdd(dotF, vScale, acc);
    }

    /// <summary>
    /// Computes 4 concurrent NVFP4 dot products using AVX-512 BW (VPSHUFB + VPMADDUBSW + VPMADDWD) against pre-quantized Q8 activations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void VecDotNVFP4x4Q8(
        BlockNVFP4* r0,
        BlockNVFP4* r1,
        BlockNVFP4* r2,
        BlockNVFP4* r3,
        BlockQ8NVFP4* qx,
        int k,
        out float s0,
        out float s1,
        out float s2,
        out float s3)
    {
        int nb = k / 64;
        var vLut128 = Vector128.Create((byte)12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
        var vLut256 = Vector256.Create(vLut128, vLut128);
        var vLutU8 = Vector512.Create(vLut256, vLut256);
        var vMask256 = Vector256.Create((byte)0x0F);
        var vOnes16 = Vector512.Create((short)1);

        var acc0 = Vector512<float>.Zero;
        var acc1 = Vector512<float>.Zero;
        var acc2 = Vector512<float>.Zero;
        var acc3 = Vector512<float>.Zero;

        fixed (float* halfScaleLut = Ue4M3HalfTable)
        {
            for (int i = 0; i < nb; i++)
            {
                BlockQ8NVFP4* q = qx + i;
                var vQx = Vector512.Load(q->Qs);
                var vDx = Vector512.Load(q->Dx);
                var vBias = Vector512.Load(q->Bias);

                AccumulateNvfp4BlockVnni(r0 + i, halfScaleLut, vLutU8, vMask256, vOnes16, vQx, vDx, vBias, ref acc0);
                AccumulateNvfp4BlockVnni(r1 + i, halfScaleLut, vLutU8, vMask256, vOnes16, vQx, vDx, vBias, ref acc1);
                AccumulateNvfp4BlockVnni(r2 + i, halfScaleLut, vLutU8, vMask256, vOnes16, vQx, vDx, vBias, ref acc2);
                AccumulateNvfp4BlockVnni(r3 + i, halfScaleLut, vLutU8, vMask256, vOnes16, vQx, vDx, vBias, ref acc3);
            }
        }

        s0 = Vector512.Sum(acc0);
        s1 = Vector512.Sum(acc1);
        s2 = Vector512.Sum(acc2);
        s3 = Vector512.Sum(acc3);
    }

    /// <summary>
    /// Computes a single NVFP4 dot product using AVX-512 BW against pre-quantized Q8 activations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotNVFP4Q8(BlockNVFP4* r0, BlockQ8NVFP4* qx, int k)
    {
        int nb = k / 64;
        var vLut128 = Vector128.Create((byte)12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
        var vLut256 = Vector256.Create(vLut128, vLut128);
        var vLutU8 = Vector512.Create(vLut256, vLut256);
        var vMask256 = Vector256.Create((byte)0x0F);
        var vOnes16 = Vector512.Create((short)1);

        var acc0 = Vector512<float>.Zero;
        fixed (float* halfScaleLut = Ue4M3HalfTable)
        {
            for (int i = 0; i < nb; i++)
            {
                BlockQ8NVFP4* q = qx + i;
                var vQx = Vector512.Load(q->Qs);
                var vDx = Vector512.Load(q->Dx);
                var vBias = Vector512.Load(q->Bias);

                AccumulateNvfp4BlockVnni(r0 + i, halfScaleLut, vLutU8, vMask256, vOnes16, vQx, vDx, vBias, ref acc0);
            }
        }

        return Vector512.Sum(acc0);
    }

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
