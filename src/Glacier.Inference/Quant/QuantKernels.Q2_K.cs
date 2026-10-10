namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

public static unsafe partial class QuantKernels
{
    /// <summary>
    /// Computes dot product between a Q2_K quantized row and a float vector x of length k.
    /// Super-block size = 256 elements in 84 bytes (~2.625 bits per weight).
    /// Used in advanced variable quantization (GEMQ / BitsMoE) for ultra-compact cold experts.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotQ2_K(BlockQ2_K* row, float* x, int k)
    {
        int nb = k / QK_K;
        float sum = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float min = (float)row[i].DeltaMin;
            byte* q = row[i].Qs;
            byte* sc = row[i].Scales;

            int is_idx = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; ++j)
                {
                    byte scaleByte0 = sc[is_idx++];
                    float dl0 = d * (scaleByte0 & 0x0F);
                    float ml0 = min * (scaleByte0 >> 4);

                    float sub0 = 0f;
                    for (int l = 0; l < 16; ++l)
                    {
                        int qVal = (q[l + 0] >> shift) & 3;
                        sub0 += (dl0 * qVal - ml0) * x[l];
                    }
                    sum += sub0;
                    x += 16;

                    byte scaleByte1 = sc[is_idx++];
                    float dl1 = d * (scaleByte1 & 0x0F);
                    float ml1 = min * (scaleByte1 >> 4);

                    float sub1 = 0f;
                    for (int l = 0; l < 16; ++l)
                    {
                        int qVal = (q[l + 16] >> shift) & 3;
                        sub1 += (dl1 * qVal - ml1) * x[l];
                    }
                    sum += sub1;
                    x += 16;

                    shift += 2;
                }
                q += 32;
            }
        }

        return sum;
    }
}
