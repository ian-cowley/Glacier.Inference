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
    /// Dequantizes a row of Q5_K blocks into 32-bit floating point array.
    /// </summary>
    public static void DequantizeQ5_K(BlockQ5_K* src, float* dst, int k)
    {
        int nb = k / QK_K;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)src[i].Delta;
            float min = (float)src[i].DeltaMin;
            byte* scales = src[i].Scales;
            byte* ql = src[i].Qs;
            byte* qh = src[i].Qh;

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

                for (int l = 0; l < 32; ++l)
                {
                    int q0 = (ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                    *dst++ = d1 * q0 - m1;
                }
                for (int l = 0; l < 32; ++l)
                {
                    int q1 = (ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                    *dst++ = d2 * q1 - m2;
                }

                ql += 32;
                is_idx += 2;
                u1 = (byte)(u1 << 2);
                u2 = (byte)(u2 << 2);
            }
        }
    }

    /// <summary>
    /// Dequantizes a row of Q3_K blocks into 32-bit floating point array.
    /// </summary>
    public static void DequantizeQ3_K(BlockQ3_K* src, float* dst, int k)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0F0F0F0F;

        int nb = k / QK_K;
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;

        for (int i = 0; i < nb; i++)
        {
            float d_all = (float)src[i].Delta;
            byte* q = src[i].Qs;
            byte* hm = src[i].Hmask;
            byte m = 1;

            Buffer.MemoryCopy(src[i].Scales, aux, 16, 12);
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
                    for (int l = 0; l < 16; ++l)
                    {
                        int q0 = (sbyte)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) != 0 ? 0 : 4);
                        *dst++ = dl0 * q0;
                    }

                    float dl1 = d_all * (scales[is_idx++] - 32);
                    for (int l = 0; l < 16; ++l)
                    {
                        int q1 = (sbyte)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4);
                        *dst++ = dl1 * q1;
                    }

                    shift += 2;
                    m = (byte)(m << 1);
                }
                q += 32;
            }
        }
    }

    /// <summary>
    /// Dequantizes a row of MXFP4 blocks into 32-bit floating point array.
    /// </summary>
    public static void DequantizeMXFP4(BlockMXFP4* src, float* dst, int k)
    {
        int nb = k / 32;
        fixed (float* lut = E2M1Table)
        {
            for (int i = 0; i < nb; i++)
            {
                float scale = MathF.ScaleB(1.0f, src[i].Scale - 127);
                byte* q = src[i].Qs;
                for (int l = 0; l < 16; ++l)
                {
                    dst[l] = scale * lut[q[l] & 0x0F];
                    dst[l + 16] = scale * lut[(q[l] >> 4) & 0x0F];
                }
                dst += 32;
            }
        }
    }

    /// <summary>
    /// Computes dot product between a Q4_K quantized row and a float vector x of length k.
    /// Uses vectorized AVX2/FMA nibble extraction and precalculated block sums.
    /// </summary>


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
                    sbyte q1 = (sbyte)(((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    sbyte q2 = (sbyte)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    sbyte q3 = (sbyte)(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    sbyte q4 = (sbyte)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);

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

}
