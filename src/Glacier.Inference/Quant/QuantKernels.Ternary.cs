namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Glacier.Inference.Gguf;

public static unsafe partial class QuantKernels
{
    /// <summary>
    /// Quantizes an unquantized FP32 weight array into 1.58-bit Ternary super-blocks (BlockTernary256).
    /// Uses BitNet b1.58 absolute-mean threshold scaling:
    /// d = mean(|W|), threshold = 0.5 * d.
    /// Weights are mapped to {-1, 0, +1} encoded as 2 bits (00=0, 01=+1, 10=-1, 11=0).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeTernary256(float* src, BlockTernary256* dst, int k)
    {
        int nb = k / QK_K;
        for (int b = 0; b < nb; b++)
        {
            float* srcBlock = src + b * QK_K;

            // 1. Calculate absolute mean scale d = (1 / 256) * sum(|w_i|)
            float absSum = 0f;
            for (int i = 0; i < QK_K; i++)
            {
                absSum += MathF.Abs(srcBlock[i]);
            }
            float d = absSum / QK_K;
            if (d < 1e-8f) d = 1e-8f;

            dst[b].Delta = (Half)d;
            dst[b].Pad = 0;

            float threshold = 0.5f * d;

            // 2. Quantize 256 weights into 64 bytes (4 weights per byte)
            for (int j = 0; j < 64; j++)
            {
                float w0 = srcBlock[j * 4 + 0];
                float w1 = srcBlock[j * 4 + 1];
                float w2 = srcBlock[j * 4 + 2];
                float w3 = srcBlock[j * 4 + 3];

                byte c0 = (byte)(w0 > threshold ? 1 : (w0 < -threshold ? 2 : 0));
                byte c1 = (byte)(w1 > threshold ? 1 : (w1 < -threshold ? 2 : 0));
                byte c2 = (byte)(w2 > threshold ? 1 : (w2 < -threshold ? 2 : 0));
                byte c3 = (byte)(w3 > threshold ? 1 : (w3 < -threshold ? 2 : 0));

                dst[b].Qs[j] = (byte)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
            }
        }
    }

    /// <summary>
    /// Span overload for QuantizeTernary256.
    /// </summary>
    public static void QuantizeTernary256(ReadOnlySpan<float> src, Span<BlockTernary256> dst)
    {
        fixed (float* pSrc = src)
        fixed (BlockTernary256* pDst = dst)
        {
            QuantizeTernary256(pSrc, pDst, src.Length);
        }
    }

    /// <summary>
    /// Dequantizes 1.58-bit Ternary super-blocks back into an FP32 array.
    /// Uses branchless sign reconstruction: val = ((code &amp; 1) - ((code &gt;&gt; 1) &amp; 1)) * d.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeTernary256(BlockTernary256* src, float* dst, int k)
    {
        int nb = k / QK_K;
        for (int b = 0; b < nb; b++)
        {
            float d = (float)src[b].Delta;
            float* dstBlock = dst + b * QK_K;

            for (int j = 0; j < 64; j++)
            {
                byte q = src[b].Qs[j];
                int c0 = q & 3;
                int c1 = (q >> 2) & 3;
                int c2 = (q >> 4) & 3;
                int c3 = (q >> 6) & 3;

                dstBlock[j * 4 + 0] = ((c0 & 1) - ((c0 >> 1) & 1)) * d;
                dstBlock[j * 4 + 1] = ((c1 & 1) - ((c1 >> 1) & 1)) * d;
                dstBlock[j * 4 + 2] = ((c2 & 1) - ((c2 >> 1) & 1)) * d;
                dstBlock[j * 4 + 3] = ((c3 & 1) - ((c3 >> 1) & 1)) * d;
            }
        }
    }

    /// <summary>
    /// Span overload for DequantizeTernary256.
    /// </summary>
    public static void DequantizeTernary256(ReadOnlySpan<BlockTernary256> src, Span<float> dst)
    {
        fixed (BlockTernary256* pSrc = src)
        fixed (float* pDst = dst)
        {
            DequantizeTernary256(pSrc, pDst, dst.Length);
        }
    }

    /// <summary>
    /// Multiplication-free vector dot product for 1.58-bit Ternary weights.
    /// The inner loop performs purely additions and subtractions of activation values (x),
    /// requiring only 1 scalar multiplication per super-block of 256 weights.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float VecDotTernary256(BlockTernary256* row, float* x, int k)
    {
        int nb = k / QK_K;
        float totalSum = 0f;

        for (int b = 0; b < nb; b++)
        {
            float d = (float)row[b].Delta;
            float blockAcc = 0f;
            float* xBlock = x + b * QK_K;

            // Unroll 64 bytes (256 weights)
            for (int j = 0; j < 64; j++)
            {
                byte q = row[b].Qs[j];
                int c0 = q & 3;
                int c1 = (q >> 2) & 3;
                int c2 = (q >> 4) & 3;
                int c3 = (q >> 6) & 3;

                // Branchless multiplication-free accumulation
                if (c0 == 1) blockAcc += xBlock[j * 4 + 0];
                else if (c0 == 2) blockAcc -= xBlock[j * 4 + 0];

                if (c1 == 1) blockAcc += xBlock[j * 4 + 1];
                else if (c1 == 2) blockAcc -= xBlock[j * 4 + 1];

                if (c2 == 1) blockAcc += xBlock[j * 4 + 2];
                else if (c2 == 2) blockAcc -= xBlock[j * 4 + 2];

                if (c3 == 1) blockAcc += xBlock[j * 4 + 3];
                else if (c3 == 2) blockAcc -= xBlock[j * 4 + 3];
            }

            totalSum += blockAcc * d;
        }

        return totalSum;
    }
}
