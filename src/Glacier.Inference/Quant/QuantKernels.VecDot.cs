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
                    var raw32 = Vector256.Load(q);
                    var mask15 = Vector256.Create((byte)0x0F);
                    var lo32 = raw32 & mask15;
                    var hi32 = Vector256.ShiftRightLogical(raw32.AsUInt16(), 4).AsByte() & mask15;

                    var vqLo0 = WidenUInt8ToFloat512(lo32.GetLower());
                    var vqLo1 = WidenUInt8ToFloat512(lo32.GetUpper());
                    var vSubSum1 = Vector512.FusedMultiplyAdd(Vector512.Load(x), vqLo0, Vector512.Load(x + 16) * vqLo1);
                    float subMin1 = xSums != null ? xSums[sumIdx++] : ComputeChunk32Sum(x);
                    sum += (d1 * Vector512.Sum(vSubSum1)) - (m1 * subMin1);
                    x += 32;

                    var vqHi0 = WidenUInt8ToFloat512(hi32.GetLower());
                    var vqHi1 = WidenUInt8ToFloat512(hi32.GetUpper());
                    var vSubSum2 = Vector512.FusedMultiplyAdd(Vector512.Load(x), vqHi0, Vector512.Load(x + 16) * vqHi1);
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
    private static Vector512<float> WidenUInt8ToFloat512(Vector128<byte> b)
    {
        if (Avx512F.IsSupported)
        {
            return Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(b));
        }

        var (u16Lo, u16Hi) = Vector128.Widen(b);
        var (u32_0, u32_1) = Vector128.Widen(u16Lo);
        var (u32_2, u32_3) = Vector128.Widen(u16Hi);
        return Vector512.ConvertToSingle(Vector512.Create(
            Vector256.Create(u32_0, u32_1),
            Vector256.Create(u32_2, u32_3)).AsInt32());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> WidenSByteToFloat512(Vector128<sbyte> b)
    {
        if (Avx512F.IsSupported)
        {
            return Vector512.ConvertToSingle(Avx512F.ConvertToVector512Int32(b));
        }

        var (i16Lo, i16Hi) = Vector128.Widen(b);
        var (i32_0, i32_1) = Vector128.Widen(i16Lo);
        var (i32_2, i32_3) = Vector128.Widen(i16Hi);
        return Vector512.ConvertToSingle(Vector512.Create(
            Vector256.Create(i32_0, i32_1),
            Vector256.Create(i32_2, i32_3)));
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
                    var ql0 = Vector256.Load(ql);
                    var ql1 = Vector256.Load(ql + 32);
                    var qh0 = Vector256.Load(qh);
                    var mask0F = Vector256.Create((byte)0x0F);
                    var mask03 = Vector256.Create((byte)0x03);
                    var v32 = Vector512.Create(32.0f);

                    var ql0Lo = ql0 & mask0F;
                    var ql1Lo = ql1 & mask0F;
                    var ql0Hi = Vector256.ShiftRightLogical(ql0.AsUInt16(), 4).AsByte() & mask0F;
                    var ql1Hi = Vector256.ShiftRightLogical(ql1.AsUInt16(), 4).AsByte() & mask0F;

                    var qhSh0 = (qh0 & mask03) << 4;
                    var qhSh2 = (Vector256.ShiftRightLogical(qh0.AsUInt16(), 2).AsByte() & mask03) << 4;
                    var qhSh4 = (Vector256.ShiftRightLogical(qh0.AsUInt16(), 4).AsByte() & mask03) << 4;
                    var qhSh6 = (Vector256.ShiftRightLogical(qh0.AsUInt16(), 6).AsByte() & mask03) << 4;

                    var q1Bytes = ql0Lo | qhSh0;
                    var q2Bytes = ql1Lo | qhSh2;
                    var q3Bytes = ql0Hi | qhSh4;
                    var q4Bytes = ql1Hi | qhSh6;

                    var vq1 = WidenUInt8ToFloat512(q1Bytes.GetLower()) - v32;
                    var vq2 = WidenUInt8ToFloat512(q2Bytes.GetLower()) - v32;
                    var vq3 = WidenUInt8ToFloat512(q3Bytes.GetLower()) - v32;
                    var vq4 = WidenUInt8ToFloat512(q4Bytes.GetLower()) - v32;

                    sub0 = Vector512.Sum(Vector512.Load(x + 0) * vq1);
                    sub2 = Vector512.Sum(Vector512.Load(x + 32) * vq2);
                    sub4 = Vector512.Sum(Vector512.Load(x + 64) * vq3);
                    sub6 = Vector512.Sum(Vector512.Load(x + 96) * vq4);

                    var vq1_b = WidenUInt8ToFloat512(q1Bytes.GetUpper()) - v32;
                    var vq2_b = WidenUInt8ToFloat512(q2Bytes.GetUpper()) - v32;
                    var vq3_b = WidenUInt8ToFloat512(q3Bytes.GetUpper()) - v32;
                    var vq4_b = WidenUInt8ToFloat512(q4Bytes.GetUpper()) - v32;

                    sub1 = Vector512.Sum(Vector512.Load(x + 16) * vq1_b);
                    sub3 = Vector512.Sum(Vector512.Load(x + 48) * vq2_b);
                    sub5 = Vector512.Sum(Vector512.Load(x + 80) * vq3_b);
                    sub7 = Vector512.Sum(Vector512.Load(x + 112) * vq4_b);
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

}
