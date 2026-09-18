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

                if (Vector512.IsHardwareAccelerated)
                {
                    var vSubSum1 = Vector512<float>.Zero;
                    for (int l = 0; l < 32; l += 16)
                    {
                        var vx = Vector512.Load(x + l);
                        var vq = Vector512.Create(
                            (float)(q[l + 0] & 0x0F), (float)(q[l + 1] & 0x0F), (float)(q[l + 2] & 0x0F), (float)(q[l + 3] & 0x0F),
                            (float)(q[l + 4] & 0x0F), (float)(q[l + 5] & 0x0F), (float)(q[l + 6] & 0x0F), (float)(q[l + 7] & 0x0F),
                            (float)(q[l + 8] & 0x0F), (float)(q[l + 9] & 0x0F), (float)(q[l + 10] & 0x0F), (float)(q[l + 11] & 0x0F),
                            (float)(q[l + 12] & 0x0F), (float)(q[l + 13] & 0x0F), (float)(q[l + 14] & 0x0F), (float)(q[l + 15] & 0x0F));
                        vSubSum1 = Vector512.FusedMultiplyAdd(vx, vq, vSubSum1);
                    }
                    float subMin1 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d1 * Vector512.Sum(vSubSum1)) - (m1 * subMin1);
                    x += 32;

                    var vSubSum2 = Vector512<float>.Zero;
                    for (int l = 0; l < 32; l += 16)
                    {
                        var vx = Vector512.Load(x + l);
                        var vq = Vector512.Create(
                            (float)(q[l + 0] >> 4), (float)(q[l + 1] >> 4), (float)(q[l + 2] >> 4), (float)(q[l + 3] >> 4),
                            (float)(q[l + 4] >> 4), (float)(q[l + 5] >> 4), (float)(q[l + 6] >> 4), (float)(q[l + 7] >> 4),
                            (float)(q[l + 8] >> 4), (float)(q[l + 9] >> 4), (float)(q[l + 10] >> 4), (float)(q[l + 11] >> 4),
                            (float)(q[l + 12] >> 4), (float)(q[l + 13] >> 4), (float)(q[l + 14] >> 4), (float)(q[l + 15] >> 4));
                        vSubSum2 = Vector512.FusedMultiplyAdd(vx, vq, vSubSum2);
                    }
                    float subMin2 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d2 * Vector512.Sum(vSubSum2)) - (m2 * subMin2);
                    x += 32;
                }
                else if (AdvSimd.IsSupported)
                {
                    var vSubSum1 = Vector128<float>.Zero;
                    for (int l = 0; l < 32; l += 4)
                    {
                        var vx = Vector128.Load(x + l);
                        var vq = Vector128.Create(
                            (float)(q[l + 0] & 0x0F), (float)(q[l + 1] & 0x0F),
                            (float)(q[l + 2] & 0x0F), (float)(q[l + 3] & 0x0F));
                        vSubSum1 = AdvSimd.FusedMultiplyAdd(vSubSum1, vx, vq);
                    }
                    float subMin1 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d1 * Vector128.Sum(vSubSum1)) - (m1 * subMin1);
                    x += 32;

                    var vSubSum2 = Vector128<float>.Zero;
                    for (int l = 0; l < 32; l += 4)
                    {
                        var vx = Vector128.Load(x + l);
                        var vq = Vector128.Create(
                            (float)(q[l + 0] >> 4), (float)(q[l + 1] >> 4),
                            (float)(q[l + 2] >> 4), (float)(q[l + 3] >> 4));
                        vSubSum2 = AdvSimd.FusedMultiplyAdd(vSubSum2, vx, vq);
                    }
                    float subMin2 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d2 * Vector128.Sum(vSubSum2)) - (m2 * subMin2);
                    x += 32;
                }
                else if (Avx2.IsSupported)
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
        if (Vector512.IsHardwareAccelerated)
        {
            var v0 = Vector512.Load(x);
            var v1 = Vector512.Load(x + 16);
            return Vector512.Sum(v0 + v1);
        }
        if (AdvSimd.IsSupported)
        {
            var v0 = Vector128.Load(x);
            var v1 = Vector128.Load(x + 4);
            var v2 = Vector128.Load(x + 8);
            var v3 = Vector128.Load(x + 12);
            var v4 = Vector128.Load(x + 16);
            var v5 = Vector128.Load(x + 20);
            var v6 = Vector128.Load(x + 24);
            var v7 = Vector128.Load(x + 28);
            return Vector128.Sum(((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7)));
        }
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

                if (Vector512.IsHardwareAccelerated)
                {
                    var vx0 = Vector512.Load(x + 0);
                    var vx2 = Vector512.Load(x + 32);
                    var vx4 = Vector512.Load(x + 64);
                    var vx6 = Vector512.Load(x + 96);

                    var vq1 = Vector512.Create(
                        (float)(((ql[0] & 0x0F) | (((qh[0] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[1] & 0x0F) | (((qh[1] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[2] & 0x0F) | (((qh[2] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[3] & 0x0F) | (((qh[3] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[4] & 0x0F) | (((qh[4] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[5] & 0x0F) | (((qh[5] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[6] & 0x0F) | (((qh[6] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[7] & 0x0F) | (((qh[7] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[8] & 0x0F) | (((qh[8] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[9] & 0x0F) | (((qh[9] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[10] & 0x0F) | (((qh[10] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[11] & 0x0F) | (((qh[11] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[12] & 0x0F) | (((qh[12] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[13] & 0x0F) | (((qh[13] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[14] & 0x0F) | (((qh[14] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[15] & 0x0F) | (((qh[15] >> 0) & 3) << 4)) - 32));

                    var vq2 = Vector512.Create(
                        (float)(((ql[32] & 0x0F) | (((qh[0] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[33] & 0x0F) | (((qh[1] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[34] & 0x0F) | (((qh[2] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[35] & 0x0F) | (((qh[3] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[36] & 0x0F) | (((qh[4] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[37] & 0x0F) | (((qh[5] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[38] & 0x0F) | (((qh[6] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[39] & 0x0F) | (((qh[7] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[40] & 0x0F) | (((qh[8] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[41] & 0x0F) | (((qh[9] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[42] & 0x0F) | (((qh[10] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[43] & 0x0F) | (((qh[11] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[44] & 0x0F) | (((qh[12] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[45] & 0x0F) | (((qh[13] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[46] & 0x0F) | (((qh[14] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[47] & 0x0F) | (((qh[15] >> 2) & 3) << 4)) - 32));

                    var vq3 = Vector512.Create(
                        (float)(((ql[0] >> 4) | (((qh[0] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[1] >> 4) | (((qh[1] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[2] >> 4) | (((qh[2] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[3] >> 4) | (((qh[3] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[4] >> 4) | (((qh[4] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[5] >> 4) | (((qh[5] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[6] >> 4) | (((qh[6] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[7] >> 4) | (((qh[7] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[8] >> 4) | (((qh[8] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[9] >> 4) | (((qh[9] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[10] >> 4) | (((qh[10] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[11] >> 4) | (((qh[11] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[12] >> 4) | (((qh[12] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[13] >> 4) | (((qh[13] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[14] >> 4) | (((qh[14] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[15] >> 4) | (((qh[15] >> 4) & 3) << 4)) - 32));

                    var vq4 = Vector512.Create(
                        (float)(((ql[32] >> 4) | (((qh[0] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[33] >> 4) | (((qh[1] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[34] >> 4) | (((qh[2] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[35] >> 4) | (((qh[3] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[36] >> 4) | (((qh[4] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[37] >> 4) | (((qh[5] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[38] >> 4) | (((qh[6] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[39] >> 4) | (((qh[7] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[40] >> 4) | (((qh[8] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[41] >> 4) | (((qh[9] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[42] >> 4) | (((qh[10] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[43] >> 4) | (((qh[11] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[44] >> 4) | (((qh[12] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[45] >> 4) | (((qh[13] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[46] >> 4) | (((qh[14] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[47] >> 4) | (((qh[15] >> 6) & 3) << 4)) - 32));

                    sub0 = Vector512.Sum(vx0 * vq1);
                    sub2 = Vector512.Sum(vx2 * vq2);
                    sub4 = Vector512.Sum(vx4 * vq3);
                    sub6 = Vector512.Sum(vx6 * vq4);

                    var vx1_512 = Vector512.Load(x + 16);
                    var vx3_512 = Vector512.Load(x + 48);
                    var vx5_512 = Vector512.Load(x + 80);
                    var vx7_512 = Vector512.Load(x + 112);

                    var vq1_b = Vector512.Create(
                        (float)(((ql[16] & 0x0F) | (((qh[16] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[17] & 0x0F) | (((qh[17] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[18] & 0x0F) | (((qh[18] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[19] & 0x0F) | (((qh[19] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[20] & 0x0F) | (((qh[20] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[21] & 0x0F) | (((qh[21] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[22] & 0x0F) | (((qh[22] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[23] & 0x0F) | (((qh[23] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[24] & 0x0F) | (((qh[24] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[25] & 0x0F) | (((qh[25] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[26] & 0x0F) | (((qh[26] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[27] & 0x0F) | (((qh[27] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[28] & 0x0F) | (((qh[28] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[29] & 0x0F) | (((qh[29] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[30] & 0x0F) | (((qh[30] >> 0) & 3) << 4)) - 32),
                        (float)(((ql[31] & 0x0F) | (((qh[31] >> 0) & 3) << 4)) - 32));

                    var vq2_b = Vector512.Create(
                        (float)(((ql[48] & 0x0F) | (((qh[16] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[49] & 0x0F) | (((qh[17] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[50] & 0x0F) | (((qh[18] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[51] & 0x0F) | (((qh[19] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[52] & 0x0F) | (((qh[20] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[53] & 0x0F) | (((qh[21] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[54] & 0x0F) | (((qh[22] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[55] & 0x0F) | (((qh[23] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[56] & 0x0F) | (((qh[24] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[57] & 0x0F) | (((qh[25] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[58] & 0x0F) | (((qh[26] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[59] & 0x0F) | (((qh[27] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[60] & 0x0F) | (((qh[28] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[61] & 0x0F) | (((qh[29] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[62] & 0x0F) | (((qh[30] >> 2) & 3) << 4)) - 32),
                        (float)(((ql[63] & 0x0F) | (((qh[31] >> 2) & 3) << 4)) - 32));

                    var vq3_b = Vector512.Create(
                        (float)(((ql[16] >> 4) | (((qh[16] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[17] >> 4) | (((qh[17] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[18] >> 4) | (((qh[18] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[19] >> 4) | (((qh[19] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[20] >> 4) | (((qh[20] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[21] >> 4) | (((qh[21] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[22] >> 4) | (((qh[22] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[23] >> 4) | (((qh[23] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[24] >> 4) | (((qh[24] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[25] >> 4) | (((qh[25] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[26] >> 4) | (((qh[26] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[27] >> 4) | (((qh[27] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[28] >> 4) | (((qh[28] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[29] >> 4) | (((qh[29] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[30] >> 4) | (((qh[30] >> 4) & 3) << 4)) - 32),
                        (float)(((ql[31] >> 4) | (((qh[31] >> 4) & 3) << 4)) - 32));

                    var vq4_b = Vector512.Create(
                        (float)(((ql[48] >> 4) | (((qh[16] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[49] >> 4) | (((qh[17] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[50] >> 4) | (((qh[18] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[51] >> 4) | (((qh[19] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[52] >> 4) | (((qh[20] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[53] >> 4) | (((qh[21] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[54] >> 4) | (((qh[22] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[55] >> 4) | (((qh[23] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[56] >> 4) | (((qh[24] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[57] >> 4) | (((qh[25] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[58] >> 4) | (((qh[26] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[59] >> 4) | (((qh[27] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[60] >> 4) | (((qh[28] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[61] >> 4) | (((qh[29] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[62] >> 4) | (((qh[30] >> 6) & 3) << 4)) - 32),
                        (float)(((ql[63] >> 4) | (((qh[31] >> 6) & 3) << 4)) - 32));

                    sub1 = Vector512.Sum(vx1_512 * vq1_b);
                    sub3 = Vector512.Sum(vx3_512 * vq2_b);
                    sub5 = Vector512.Sum(vx5_512 * vq3_b);
                    sub7 = Vector512.Sum(vx7_512 * vq4_b);
                }
                else if (AdvSimd.IsSupported)
                {
                    for (int l = 0; l < 16; l += 4)
                    {
                        var vx0 = Vector128.Load(x + l + 0);
                        var vx2 = Vector128.Load(x + l + 32);
                        var vx4 = Vector128.Load(x + l + 64);
                        var vx6 = Vector128.Load(x + l + 96);

                        var vq1 = Vector128.Create(
                            (float)(((ql[l + 0] & 0x0F) | (((qh[l + 0] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] & 0x0F) | (((qh[l + 1] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] & 0x0F) | (((qh[l + 2] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] & 0x0F) | (((qh[l + 3] >> 0) & 3) << 4)) - 32));

                        var vq2 = Vector128.Create(
                            (float)(((ql[l + 32] & 0x0F) | (((qh[l + 0] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] & 0x0F) | (((qh[l + 1] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] & 0x0F) | (((qh[l + 2] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] & 0x0F) | (((qh[l + 3] >> 2) & 3) << 4)) - 32));

                        var vq3 = Vector128.Create(
                            (float)(((ql[l + 0] >> 4) | (((qh[l + 0] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] >> 4) | (((qh[l + 1] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] >> 4) | (((qh[l + 2] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] >> 4) | (((qh[l + 3] >> 4) & 3) << 4)) - 32));

                        var vq4 = Vector128.Create(
                            (float)(((ql[l + 32] >> 4) | (((qh[l + 0] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] >> 4) | (((qh[l + 1] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] >> 4) | (((qh[l + 2] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] >> 4) | (((qh[l + 3] >> 6) & 3) << 4)) - 32));

                        sub0 += Vector128.Sum(vx0 * vq1);
                        sub2 += Vector128.Sum(vx2 * vq2);
                        sub4 += Vector128.Sum(vx4 * vq3);
                        sub6 += Vector128.Sum(vx6 * vq4);
                    }
                    for (int l = 16; l < 32; l += 4)
                    {
                        var vx0 = Vector128.Load(x + l + 0);
                        var vx2 = Vector128.Load(x + l + 32);
                        var vx4 = Vector128.Load(x + l + 64);
                        var vx6 = Vector128.Load(x + l + 96);

                        var vq1 = Vector128.Create(
                            (float)(((ql[l + 0] & 0x0F) | (((qh[l + 0] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] & 0x0F) | (((qh[l + 1] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] & 0x0F) | (((qh[l + 2] >> 0) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] & 0x0F) | (((qh[l + 3] >> 0) & 3) << 4)) - 32));

                        var vq2 = Vector128.Create(
                            (float)(((ql[l + 32] & 0x0F) | (((qh[l + 0] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] & 0x0F) | (((qh[l + 1] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] & 0x0F) | (((qh[l + 2] >> 2) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] & 0x0F) | (((qh[l + 3] >> 2) & 3) << 4)) - 32));

                        var vq3 = Vector128.Create(
                            (float)(((ql[l + 0] >> 4) | (((qh[l + 0] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 1] >> 4) | (((qh[l + 1] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 2] >> 4) | (((qh[l + 2] >> 4) & 3) << 4)) - 32),
                            (float)(((ql[l + 3] >> 4) | (((qh[l + 3] >> 4) & 3) << 4)) - 32));

                        var vq4 = Vector128.Create(
                            (float)(((ql[l + 32] >> 4) | (((qh[l + 0] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 33] >> 4) | (((qh[l + 1] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 34] >> 4) | (((qh[l + 2] >> 6) & 3) << 4)) - 32),
                            (float)(((ql[l + 35] >> 4) | (((qh[l + 3] >> 6) & 3) << 4)) - 32));

                        sub1 += Vector128.Sum(vx0 * vq1);
                        sub3 += Vector128.Sum(vx2 * vq2);
                        sub5 += Vector128.Sum(vx4 * vq3);
                        sub7 += Vector128.Sum(vx6 * vq4);
                    }
                }
                else if (Vector256.IsHardwareAccelerated)
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
                        int q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                        sub0 += q1 * x[l + 0];
                        sub2 += q2 * x[l + 32];
                        sub4 += q3 * x[l + 64];
                        sub6 += q4 * x[l + 96];
                    }

                    for (int l = 16; l < 32; ++l)
                    {
                        int q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

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

}
