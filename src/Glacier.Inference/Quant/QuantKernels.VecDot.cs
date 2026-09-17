namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

public static unsafe partial class QuantKernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ4_K(BlockQ4_K* row, float* x, int k) => VecDotQ4_K(row, x, null, k);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ4_K(BlockQ4_K* row, float* x, float* xSums, int k)
    {
        int nb = k / QK_K;
        float sum = 0f;
        var vmaskLow = Vector128.Create((byte)0x0F);
        int sumIdx = 0;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float min = (float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* q = row[i].Qs;

            int is_idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0, m0, sc1, m1_val;
                if (is_idx < 4)
                {
                    sc0 = (byte)(scales[is_idx] & 63);
                    m0 = (byte)(scales[is_idx + 4] & 63);
                    sc1 = (byte)(scales[is_idx + 1] & 63);
                    m1_val = (byte)(scales[is_idx + 5] & 63);
                }
                else
                {
                    sc0 = (byte)((scales[is_idx + 4] & 0x0F) | ((scales[is_idx - 4] >> 6) << 4));
                    m0 = (byte)((scales[is_idx + 4] >> 4) | ((scales[is_idx] >> 6) << 4));
                    sc1 = (byte)((scales[is_idx + 5] & 0x0F) | ((scales[is_idx - 3] >> 6) << 4));
                    m1_val = (byte)((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4));
                }

                float d1 = d * sc0;
                float m1 = min * m0;
                float d2 = d * sc1;
                float m2 = min * m1_val;

                if (Avx2.IsSupported)
                {
                    var vSubSum1 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l);
                        ulong q64 = *(ulong*)(q + l);
                        var v8 = Vector128.CreateScalar(q64).As<ulong, byte>();
                        var vLow = Avx2.And(v8, vmaskLow);
                        var vInt = Avx2.ConvertToVector256Int32(vLow);
                        var vFloat = Vector256.ConvertToSingle(vInt);
                        if (Fma.IsSupported)
                            vSubSum1 = Fma.MultiplyAdd(vx, vFloat, vSubSum1);
                        else
                            vSubSum1 += vx * vFloat;
                    }
                    float subMin1 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d1 * Vector256.Sum(vSubSum1)) - (m1 * subMin1);
                    x += 32;

                    var vSubSum2 = Vector256<float>.Zero;
                    for (int l = 0; l < 32; l += 8)
                    {
                        var vx = Vector256.Load(x + l);
                        ulong q64 = *(ulong*)(q + l);
                        var v8 = Vector128.CreateScalar(q64).As<ulong, byte>();
                        var vShift = Avx2.ShiftRightLogical(v8.As<byte, int>(), 4).As<int, byte>();
                        var vHigh = Avx2.And(vShift, vmaskLow);
                        var vInt = Avx2.ConvertToVector256Int32(vHigh);
                        var vFloat = Vector256.ConvertToSingle(vInt);
                        if (Fma.IsSupported)
                            vSubSum2 = Fma.MultiplyAdd(vx, vFloat, vSubSum2);
                        else
                            vSubSum2 += vx * vFloat;
                    }
                    float subMin2 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d2 * Vector256.Sum(vSubSum2)) - (m2 * subMin2);
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
                    float s1 = xSums != null ? xSums[sumIdx++] : subMin1;
                    sum += (d1 * subSum1) - (m1 * s1);
                    x += 32;

                    float subSum2 = 0f;
                    float subMin2 = 0f;
                    for (int l = 0; l < 32; ++l)
                    {
                        float val_x = x[l];
                        subSum2 += (q[l] >> 4) * val_x;
                        subMin2 += val_x;
                    }
                    float s2 = xSums != null ? xSums[sumIdx++] : subMin2;
                    sum += (d2 * subSum2) - (m2 * s2);
                    x += 32;
                }

                q += 32;
                is_idx += 2;
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ComputeChunk32Sum(float* x)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var v0 = Vector256.Load(x);
            var v1 = Vector256.Load(x + 8);
            var v2 = Vector256.Load(x + 16);
            var v3 = Vector256.Load(x + 24);
            return Vector256.Sum((v0 + v1) + (v2 + v3));
        }
        float s = 0;
        for (int l = 0; l < 32; l++) s += x[l];
        return s;
    }

    /// <summary>
    /// Computes 32-element chunk sums of vector x across all chunks (nCols / 32).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ComputeBlockSums32(float* x, float* xSums, int nCols)
    {
        int nChunks = nCols / 32;
        for (int c = 0; c < nChunks; c++)
        {
            xSums[c] = ComputeChunk32Sum(x + c * 32);
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

}
