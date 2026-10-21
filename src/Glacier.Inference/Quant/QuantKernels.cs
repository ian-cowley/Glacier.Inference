namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Hardware SIMD accelerated quantization kernels, dot products, and transformer activation functions.
/// Written in pure C# .NET 10 with zero-allocation hot paths.
/// </summary>
public static unsafe partial class QuantKernels
{
    private const int QK_K = 256;
    private const int QK8_0 = 32;
    private const int QK4_0 = 32;
    private const int QK5_0 = 32;


    /// <summary>
    /// Copies an embedding row for token into destination float buffer.
    /// Supports Q4_K, Q6_K, Q8_0, F16, and F32 embeddings.
    /// </summary>
    public static void ExtractEmbedding(GgufType type, byte* embdData, int tokenId, float* dst, int embeddingDim)
    {
        int rowBytes = (int)GgufTypes.GetRowBytes(type, embeddingDim);
        byte* rowPtr = embdData + (long)tokenId * rowBytes;

        switch (type)
        {
            case GgufType.Q2_K:
                DequantizeQ2_K((BlockQ2_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q2_0:
                DequantizeQ2_0((BlockQ2_0*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q3_K:
                DequantizeQ3_K((BlockQ3_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.IQ2_XXS:
                DequantizeIQ2_XXS((BlockIQ2_XXS*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.IQ3_S:
                DequantizeIQ3_S((BlockIQ3_S*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.IQ4_NL:
                DequantizeIQ4_NL((BlockIQ4_NL*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.IQ4_XS:
                DequantizeIQ4_XS((BlockIQ4_XS*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Ternary_1_58:
            case GgufType.PQ2_0:
                DequantizeTernary256((BlockTernary256*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q4_0:
                for (int i = 0; i < embeddingDim / 32; i++)
                {
                    BlockQ4_0* b = (BlockQ4_0*)(rowPtr + i * sizeof(BlockQ4_0));
                    float d = (float)b->Delta;
                    for (int l = 0; l < 16; l++)
                    {
                        byte v = b->Qs[l];
                        dst[i * 32 + l] = ((v & 0x0F) - 8) * d;
                        dst[i * 32 + l + 16] = ((v >> 4) - 8) * d;
                    }
                }
                break;
            case GgufType.Q4_K:
                DequantizeQ4_K((BlockQ4_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q5_0:
                DequantizeQ5_0((BlockQ5_0*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q5_K:
                DequantizeQ5_K((BlockQ5_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q6_K:
                DequantizeQ6_K((BlockQ6_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.MXFP4:
                DequantizeMXFP4((BlockMXFP4*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q8_0:
                for (int i = 0; i < embeddingDim / 32; i++)
                {
                    BlockQ8_0* b = (BlockQ8_0*)(rowPtr + i * sizeof(BlockQ8_0));
                    float d = (float)b->Delta;
                    for (int l = 0; l < 32; l++)
                    {
                        dst[i * 32 + l] = d * b->Qs[l];
                    }
                }
                break;
            case GgufType.F16:
                Half* h = (Half*)rowPtr;
                for (int i = 0; i < embeddingDim; i++) dst[i] = (float)h[i];
                break;
            case GgufType.F32:
                Buffer.MemoryCopy(rowPtr, dst, embeddingDim * 4, embeddingDim * 4);
                break;
            default:
                throw new NotSupportedException($"Embedding quantization type {type} not supported.");
        }
    }
}
