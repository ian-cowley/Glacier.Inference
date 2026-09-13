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
            case GgufType.Q3_K:
                DequantizeQ3_K((BlockQ3_K*)rowPtr, dst, embeddingDim);
                break;
            case GgufType.Q4_K:
                DequantizeQ4_K((BlockQ4_K*)rowPtr, dst, embeddingDim);
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
