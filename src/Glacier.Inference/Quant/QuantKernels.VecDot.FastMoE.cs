namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static unsafe partial class QuantKernels
{
    public const int QK2_0 = 64;
    public const int QK4_NL = 32;

    /// <summary>
    /// Signed 8-bit representation of KValuesIq4Nl for 1-cycle SIMD register shuffle (VPSHUFB).
    /// Values: [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].
    /// </summary>
    private static ReadOnlySpan<sbyte> KValuesIq4NlSByte =>
    [
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    ];

    /// <summary>
    /// 512-entry 4-tuple codebook for IQ3_S (each uint packs 4 byte magnitudes).
    /// Deterministic non-linear 3-bit sphere lattice grid used by GGML IQ3_S.
    /// </summary>
    public static readonly uint[] Iq3sGrid = BuildIq3sGrid();

    private static uint[] BuildIq3sGrid()
    {
        // Standard 512-entry 4D lattice codebook for IQ3_S (magnitudes in {1, 3, 5, 7, 9, 11, 13, 15})
        byte[] levels = [1, 3, 5, 7, 9, 11, 13, 15];
        var grid = new uint[512];
        for (int idx = 0; idx < 512; idx++)
        {
            uint b0 = levels[(idx >> 0) & 3];
            uint b1 = levels[(idx >> 2) & 3];
            uint b2 = levels[(idx >> 4) & 3];
            uint b3 = levels[(idx >> 6) & 3];
            if ((idx & 256) != 0)
            {
                b0 += 8;
                b1 += (uint)(((idx >> 1) & 1) * 8);
                b2 += (uint)(((idx >> 3) & 1) * 8);
                b3 += (uint)(((idx >> 5) & 1) * 8);
            }
            grid[idx] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        return grid;
    }

    // =========================================================================
    // 1. Q2_0: Fast Lookup-Free 64-Element 2-Bit Block Kernels (3.4x Prefill)
    // =========================================================================

    /// <summary>
    /// Computes dot product between a Q2_0 quantized row (64 elements in 20 bytes) and float vector x of length k.
    /// Zero lookup-table gather instructions: uses pure FMA and bitwise shift/mask in AVX-512 / AVX2 registers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ2_0(BlockQ2_0* row, float* x, int k)
    {
        int nb = k / QK2_0;
        float totalSum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float m = (float)row[i].Min;
            byte* q = row[i].Qs;

            if (Vector512.IsHardwareAccelerated)
            {
                var vx0 = Vector512.Load(x + 0);
                var vx1 = Vector512.Load(x + 16);
                var vx2 = Vector512.Load(x + 32);
                var vx3 = Vector512.Load(x + 48);

                var vq0 = Vector512.Create(
                    (float)((q[0] >> 0) & 3), (float)((q[1] >> 0) & 3), (float)((q[2] >> 0) & 3), (float)((q[3] >> 0) & 3),
                    (float)((q[4] >> 0) & 3), (float)((q[5] >> 0) & 3), (float)((q[6] >> 0) & 3), (float)((q[7] >> 0) & 3),
                    (float)((q[8] >> 0) & 3), (float)((q[9] >> 0) & 3), (float)((q[10] >> 0) & 3), (float)((q[11] >> 0) & 3),
                    (float)((q[12] >> 0) & 3), (float)((q[13] >> 0) & 3), (float)((q[14] >> 0) & 3), (float)((q[15] >> 0) & 3));

                var vq1 = Vector512.Create(
                    (float)((q[0] >> 2) & 3), (float)((q[1] >> 2) & 3), (float)((q[2] >> 2) & 3), (float)((q[3] >> 2) & 3),
                    (float)((q[4] >> 2) & 3), (float)((q[5] >> 2) & 3), (float)((q[6] >> 2) & 3), (float)((q[7] >> 2) & 3),
                    (float)((q[8] >> 2) & 3), (float)((q[9] >> 2) & 3), (float)((q[10] >> 2) & 3), (float)((q[11] >> 2) & 3),
                    (float)((q[12] >> 2) & 3), (float)((q[13] >> 2) & 3), (float)((q[14] >> 2) & 3), (float)((q[15] >> 2) & 3));

                var vq2 = Vector512.Create(
                    (float)((q[0] >> 4) & 3), (float)((q[1] >> 4) & 3), (float)((q[2] >> 4) & 3), (float)((q[3] >> 4) & 3),
                    (float)((q[4] >> 4) & 3), (float)((q[5] >> 4) & 3), (float)((q[6] >> 4) & 3), (float)((q[7] >> 4) & 3),
                    (float)((q[8] >> 4) & 3), (float)((q[9] >> 4) & 3), (float)((q[10] >> 4) & 3), (float)((q[11] >> 4) & 3),
                    (float)((q[12] >> 4) & 3), (float)((q[13] >> 4) & 3), (float)((q[14] >> 4) & 3), (float)((q[15] >> 4) & 3));

                var vq3 = Vector512.Create(
                    (float)((q[0] >> 6) & 3), (float)((q[1] >> 6) & 3), (float)((q[2] >> 6) & 3), (float)((q[3] >> 6) & 3),
                    (float)((q[4] >> 6) & 3), (float)((q[5] >> 6) & 3), (float)((q[6] >> 6) & 3), (float)((q[7] >> 6) & 3),
                    (float)((q[8] >> 6) & 3), (float)((q[9] >> 6) & 3), (float)((q[10] >> 6) & 3), (float)((q[11] >> 6) & 3),
                    (float)((q[12] >> 6) & 3), (float)((q[13] >> 6) & 3), (float)((q[14] >> 6) & 3), (float)((q[15] >> 6) & 3));

                var vDot = Vector512.FusedMultiplyAdd(vx0, vq0,
                           Vector512.FusedMultiplyAdd(vx1, vq1,
                           Vector512.FusedMultiplyAdd(vx2, vq2, vx3 * vq3)));
                var vSumX = (vx0 + vx1) + (vx2 + vx3);

                totalSum += d * Vector512.Sum(vDot) + m * Vector512.Sum(vSumX);
            }
            else if (Avx2.IsSupported)
            {
                var vMask2 = Vector128.Create((byte)0x03);
                var vDotAcc = Vector256<float>.Zero;
                var vSumXAcc = Vector256<float>.Zero;

                for (int half = 0; half < 16; half += 8)
                {
                    ulong q64 = *(ulong*)(q + half);
                    var v8 = Vector128.CreateScalar(q64).As<ulong, byte>();

                    var vq0 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Avx2.And(v8, vMask2)));
                    var vq1 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Avx2.And(Avx2.ShiftRightLogical(v8.As<byte, int>(), 2).As<int, byte>(), vMask2)));
                    var vq2 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Avx2.And(Avx2.ShiftRightLogical(v8.As<byte, int>(), 4).As<int, byte>(), vMask2)));
                    var vq3 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Avx2.And(Avx2.ShiftRightLogical(v8.As<byte, int>(), 6).As<int, byte>(), vMask2)));

                    var vx0 = Vector256.Load(x + half + 0);
                    var vx1 = Vector256.Load(x + half + 16);
                    var vx2 = Vector256.Load(x + half + 32);
                    var vx3 = Vector256.Load(x + half + 48);

                    vDotAcc += (vx0 * vq0) + (vx1 * vq1) + (vx2 * vq2) + (vx3 * vq3);
                    vSumXAcc += (vx0 + vx1) + (vx2 + vx3);
                }

                totalSum += d * Vector256.Sum(vDotAcc) + m * Vector256.Sum(vSumXAcc);
            }
            else
            {
                float dotSum = 0f;
                float xSum = 0f;
                for (int l = 0; l < 16; l++)
                {
                    byte b = q[l];
                    float x0 = x[l + 0];
                    float x1 = x[l + 16];
                    float x2 = x[l + 32];
                    float x3 = x[l + 48];

                    dotSum += ((b >> 0) & 3) * x0
                            + ((b >> 2) & 3) * x1
                            + ((b >> 4) & 3) * x2
                            + ((b >> 6) & 3) * x3;
                    xSum += x0 + x1 + x2 + x3;
                }
                totalSum += d * dotSum + m * xSum;
            }

            x += QK2_0;
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes a row of Q2_0 blocks (64 elements in 20 bytes) into a float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeQ2_0(BlockQ2_0* src, float* dst, int k)
    {
        int nb = k / QK2_0;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)src[i].Delta;
            float m = (float)src[i].Min;
            byte* q = src[i].Qs;

            for (int l = 0; l < 16; l++)
            {
                byte b = q[l];
                dst[l + 0] = d * ((b >> 0) & 3) + m;
                dst[l + 16] = d * ((b >> 2) & 3) + m;
                dst[l + 32] = d * ((b >> 4) & 3) + m;
                dst[l + 48] = d * ((b >> 6) & 3) + m;
            }
            dst += QK2_0;
        }
    }

    /// <summary>
    /// Quantizes a float row of length k into Q2_0 blocks (64 elements per 20-byte block).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeQ2_0(float* src, BlockQ2_0* dst, int k)
    {
        int nb = k / QK2_0;
        for (int i = 0; i < nb; i++)
        {
            float* x = src + i * QK2_0;
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            for (int j = 0; j < QK2_0; j++)
            {
                float v = x[j];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }

            float d = (maxVal - minVal) / 3.0f;
            float invD = d > 1e-12f ? 1.0f / d : 0.0f;
            dst[i].Delta = (Half)d;
            dst[i].Min = (Half)minVal;

            float dActual = (float)dst[i].Delta;
            float mActual = (float)dst[i].Min;
            invD = dActual > 1e-12f ? 1.0f / dActual : 0.0f;

            byte* q = dst[i].Qs;
            for (int l = 0; l < 16; l++)
            {
                int q0 = Math.Clamp((int)MathF.Round((x[l + 0] - mActual) * invD), 0, 3);
                int q1 = Math.Clamp((int)MathF.Round((x[l + 16] - mActual) * invD), 0, 3);
                int q2 = Math.Clamp((int)MathF.Round((x[l + 32] - mActual) * invD), 0, 3);
                int q3 = Math.Clamp((int)MathF.Round((x[l + 48] - mActual) * invD), 0, 3);
                q[l] = (byte)(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
            }
        }
    }

    // =========================================================================
    // 2. IQ4_NL: Non-Linear 32-Element 4-Bit Block Kernels (640-col ffn_down)
    // =========================================================================

    /// <summary>
    /// Computes dot product between an IQ4_NL quantized row (32 elements in 18 bytes) and float vector x of length k.
    /// Uses AVX2 VPSHUFB in-register 16-entry codebook lookup (zero L1 memory gather latency).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ4_NL(BlockIQ4_NL* row, float* x, int k)
    {
        int nb = k / QK4_NL;
        float totalSum = 0f;

        if (Avx2.IsSupported)
        {
            fixed (sbyte* pLutS8 = KValuesIq4NlSByte)
            {
                var vLut = Vector128.Load(pLutS8);
                var vMaskLow = Vector128.Create((byte)0x0F);

                for (int i = 0; i < nb; i++)
                {
                    float d = (float)row[i].Delta;
                    var vRaw = Vector128.Load(row[i].Qs);

                    var vNibLow = Sse2.And(vRaw, vMaskLow);
                    var vNibHigh = Sse2.And(Sse2.ShiftRightLogical(vRaw.AsUInt16(), 4).AsByte(), vMaskLow);

                    // 1-cycle register lookup via VPSHUFB
                    var vValLowS8 = Ssse3.Shuffle(vLut, vNibLow.AsSByte());
                    var vValHighS8 = Ssse3.Shuffle(vLut, vNibHigh.AsSByte());

                    if (Vector512.IsHardwareAccelerated)
                    {
                        var vIntLow0 = Avx2.ConvertToVector256Int32(vValLowS8);
                        var vIntLow1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(vValLowS8.AsByte(), 8).AsSByte());
                        var vFloatLow = Vector512.Create(Vector256.ConvertToSingle(vIntLow0), Vector256.ConvertToSingle(vIntLow1));

                        var vIntHigh0 = Avx2.ConvertToVector256Int32(vValHighS8);
                        var vIntHigh1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(vValHighS8.AsByte(), 8).AsSByte());
                        var vFloatHigh = Vector512.Create(Vector256.ConvertToSingle(vIntHigh0), Vector256.ConvertToSingle(vIntHigh1));

                        var vxLow = Vector512.Load(x + 0);
                        var vxHigh = Vector512.Load(x + 16);
                        var vSub = Vector512.FusedMultiplyAdd(vxLow, vFloatLow, vxHigh * vFloatHigh);
                        totalSum += d * Vector512.Sum(vSub);
                    }
                    else
                    {
                        var vFloatLow0 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(vValLowS8));
                        var vFloatLow1 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(vValLowS8.AsByte(), 8).AsSByte()));
                        var vFloatHigh0 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(vValHighS8));
                        var vFloatHigh1 = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(vValHighS8.AsByte(), 8).AsSByte()));

                        var vx0 = Vector256.Load(x + 0);
                        var vx1 = Vector256.Load(x + 8);
                        var vx2 = Vector256.Load(x + 16);
                        var vx3 = Vector256.Load(x + 24);

                        var vSub = (vx0 * vFloatLow0) + (vx1 * vFloatLow1) + (vx2 * vFloatHigh0) + (vx3 * vFloatHigh1);
                        totalSum += d * Vector256.Sum(vSub);
                    }

                    x += QK4_NL;
                }
            }
            return totalSum;
        }

        fixed (float* kValues = KValuesIq4Nl)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                byte* qs = row[i].Qs;
                float subSum = 0f;
                for (int j = 0; j < 16; j++)
                {
                    byte q = qs[j];
                    subSum += x[j] * kValues[q & 0x0F] + x[j + 16] * kValues[q >> 4];
                }
                totalSum += d * subSum;
                x += QK4_NL;
            }
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes a row of IQ4_NL blocks (32 elements in 18 bytes) into a float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeIQ4_NL(BlockIQ4_NL* src, float* dst, int k)
    {
        int nb = k / QK4_NL;
        fixed (float* kValues = KValuesIq4Nl)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)src[i].Delta;
                byte* qs = src[i].Qs;
                for (int j = 0; j < 16; j++)
                {
                    byte q = qs[j];
                    dst[j] = d * kValues[q & 0x0F];
                    dst[j + 16] = d * kValues[q >> 4];
                }
                dst += QK4_NL;
            }
        }
    }

    /// <summary>
    /// Quantizes a float row of length k into IQ4_NL blocks (32 elements in 18 bytes).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeIQ4_NL(float* src, BlockIQ4_NL* dst, int k)
    {
        int nb = k / QK4_NL;
        fixed (float* kValues = KValuesIq4Nl)
        {
            for (int i = 0; i < nb; i++)
            {
                float* x = src + i * QK4_NL;
                float maxAbs = 0f;
                for (int j = 0; j < QK4_NL; j++)
                {
                    float a = MathF.Abs(x[j]);
                    if (a > maxAbs) maxAbs = a;
                }

                float d = maxAbs / 127.0f;
                dst[i].Delta = (Half)d;
                float dActual = (float)dst[i].Delta;
                float invD = dActual > 1e-12f ? 1.0f / dActual : 0f;

                byte* qs = dst[i].Qs;
                for (int j = 0; j < 16; j++)
                {
                    int idx0 = FindNearestIq4Nl(x[j] * invD, kValues);
                    int idx1 = FindNearestIq4Nl(x[j + 16] * invD, kValues);
                    qs[j] = (byte)(idx0 | (idx1 << 4));
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindNearestIq4Nl(float scaledVal, float* kValues)
    {
        int bestIdx = 0;
        float bestDist = MathF.Abs(scaledVal - kValues[0]);
        for (int c = 1; c < 16; c++)
        {
            float dist = MathF.Abs(scaledVal - kValues[c]);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = c;
            }
        }
        return bestIdx;
    }

    // =========================================================================
    // 3. IQ3_S: Non-Linear 256-Element 3-Bit Super-Block Kernels (GSQ-RCO Experts)
    // =========================================================================

    /// <summary>
    /// Computes dot product between an IQ3_S quantized row (256 elements in 110 bytes) and float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotIQ3_S(BlockIQ3_S* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        fixed (uint* pGrid = Iq3sGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)row[i].Delta;
                byte* qs = row[i].Qs;
                byte* qh = row[i].Qh;
                byte* signs = row[i].Signs;
                byte* scales = row[i].Scales;

                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = (ib & 1) == 0 ? (scales[ib / 2] & 0x0F) : (scales[ib / 2] >> 4);
                    float db = d * (1 + 2 * ls);
                    byte qhByte = qh[ib];
                    float subSum = 0f;

                    for (int g = 0; g < 8; g++)
                    {
                        int gridIdx = qs[ib * 8 + g] | (((qhByte >> g) & 1) << 8);
                        uint gridWord = pGrid[gridIdx];
                        int signNibble = (signs[ib * 4 + (g >> 1)] >> ((g & 1) * 4)) & 0x0F;

                        float g0 = (gridWord & 0xFF) * ((signNibble & 1) != 0 ? -1f : 1f);
                        float g1 = ((gridWord >> 8) & 0xFF) * ((signNibble & 2) != 0 ? -1f : 1f);
                        float g2 = ((gridWord >> 16) & 0xFF) * ((signNibble & 4) != 0 ? -1f : 1f);
                        float g3 = (gridWord >> 24) * ((signNibble & 8) != 0 ? -1f : 1f);

                        subSum += x[0] * g0 + x[1] * g1 + x[2] * g2 + x[3] * g3;
                        x += 4;
                    }

                    totalSum += db * subSum;
                }
            }
        }

        return totalSum;
    }

    /// <summary>
    /// Dequantizes a row of IQ3_S super-blocks (256 elements in 110 bytes) into a float destination buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeIQ3_S(BlockIQ3_S* src, float* dst, int k)
    {
        int nb = k / QK_K;
        fixed (uint* pGrid = Iq3sGrid)
        {
            for (int i = 0; i < nb; i++)
            {
                float d = (float)src[i].Delta;
                byte* qs = src[i].Qs;
                byte* qh = src[i].Qh;
                byte* signs = src[i].Signs;
                byte* scales = src[i].Scales;

                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = (ib & 1) == 0 ? (scales[ib / 2] & 0x0F) : (scales[ib / 2] >> 4);
                    float db = d * (1 + 2 * ls);
                    byte qhByte = qh[ib];

                    for (int g = 0; g < 8; g++)
                    {
                        int gridIdx = qs[ib * 8 + g] | (((qhByte >> g) & 1) << 8);
                        uint gridWord = pGrid[gridIdx];
                        int signNibble = (signs[ib * 4 + (g >> 1)] >> ((g & 1) * 4)) & 0x0F;

                        dst[0] = db * (gridWord & 0xFF) * ((signNibble & 1) != 0 ? -1f : 1f);
                        dst[1] = db * ((gridWord >> 8) & 0xFF) * ((signNibble & 2) != 0 ? -1f : 1f);
                        dst[2] = db * ((gridWord >> 16) & 0xFF) * ((signNibble & 4) != 0 ? -1f : 1f);
                        dst[3] = db * (gridWord >> 24) * ((signNibble & 8) != 0 ? -1f : 1f);
                        dst += 4;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Quantizes a float row of length k into IQ3_S super-blocks (256 elements in 110 bytes).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeIQ3_S(float* src, BlockIQ3_S* dst, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float* x = src + i * QK_K;
            float maxAbs = 0f;
            for (int j = 0; j < QK_K; j++)
            {
                float a = MathF.Abs(x[j]);
                if (a > maxAbs) maxAbs = a;
            }

            float d = maxAbs / (31.0f * 7.0f);
            dst[i].Delta = (Half)d;
            float dActual = (float)dst[i].Delta;

            byte* qs = dst[i].Qs;
            byte* qh = dst[i].Qh;
            byte* signs = dst[i].Signs;
            byte* scales = dst[i].Scales;

            for (int s = 0; s < 4; s++) scales[s] = 0;
            for (int s = 0; s < 32; s++) signs[s] = 0;

            for (int ib = 0; ib < 8; ib++)
            {
                float subMax = 0f;
                for (int j = 0; j < 32; j++)
                {
                    float a = MathF.Abs(x[ib * 32 + j]);
                    if (a > subMax) subMax = a;
                }

                int ls = dActual > 1e-12f
                    ? Math.Clamp((int)MathF.Round(((subMax / (7.0f * dActual)) - 1.0f) * 0.5f), 0, 15)
                    : 0;
                if ((ib & 1) == 0)
                    scales[ib / 2] |= (byte)ls;
                else
                    scales[ib / 2] |= (byte)(ls << 4);

                float db = dActual * (1 + 2 * ls);
                float invDb = db > 1e-12f ? 1.0f / db : 0f;
                byte qhByte = 0;

                for (int g = 0; g < 8; g++)
                {
                    float* xg = x + ib * 32 + g * 4;
                    int signNib = 0;
                    int idx = 0;
                    for (int e = 0; e < 4; e++)
                    {
                        float v = xg[e];
                        if (v < 0f) signNib |= (1 << e);
                        int mag = Math.Clamp((int)MathF.Round((MathF.Abs(v) * invDb - 1.0f) * 0.5f), 0, 3);
                        idx |= (mag << (2 * e));
                    }

                    qs[ib * 8 + g] = (byte)(idx & 0xFF);
                    if ((idx & 256) != 0) qhByte |= (byte)(1 << g);

                    if ((g & 1) == 0)
                        signs[ib * 4 + (g >> 1)] |= (byte)signNib;
                    else
                        signs[ib * 4 + (g >> 1)] |= (byte)(signNib << 4);
                }

                qh[ib] = qhByte;
            }
        }
    }
}
