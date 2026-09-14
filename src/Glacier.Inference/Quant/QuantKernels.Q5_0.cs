namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;

public static unsafe partial class QuantKernels
{
    /// <summary>
    /// Computes dot product between a Q5_0 quantized row and a float vector x of length k.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ5_0(BlockQ5_0* row, float* x, int k)
    {
        int nb = k / QK5_0;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            uint qh = row[i].Qh;
            byte* q = row[i].Qs;
            float blockSum = 0f;

            for (int l = 0; l < 16; ++l)
            {
                int h0 = (int)((qh >> l) & 1) << 4;
                int h1 = (int)((qh >> (l + 16)) & 1) << 4;
                int q0 = ((q[l] & 0x0F) | h0) - 16;
                int q1 = ((q[l] >> 4) | h1) - 16;
                blockSum += q0 * x[l] + q1 * x[l + 16];
            }

            sum += d * blockSum;
            x += 32;
        }

        return sum;
    }

    /// <summary>
    /// Dequantizes a row of Q5_0 blocks into 32-bit floating point array.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeQ5_0(BlockQ5_0* src, float* dst, int k)
    {
        int nb = k / QK5_0;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)src[i].Delta;
            uint qh = src[i].Qh;
            byte* q = src[i].Qs;

            for (int l = 0; l < 16; ++l)
            {
                int h0 = (int)((qh >> l) & 1) << 4;
                int h1 = (int)((qh >> (l + 16)) & 1) << 4;
                int q0 = ((q[l] & 0x0F) | h0) - 16;
                int q1 = ((q[l] >> 4) | h1) - 16;
                dst[l] = d * q0;
                dst[l + 16] = d * q1;
            }
            dst += 32;
        }
    }
}
