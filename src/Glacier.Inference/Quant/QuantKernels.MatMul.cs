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
    /// <summary>
    /// Multiplies a quantized matrix W [nRows, nCols] by vector x [nCols] producing y [nRows].
    /// Automatically multithreads row calculations across all available CPU cores.
    /// </summary>
    public static void MatVecMul(GgufType type, byte* weightData, float* x, float* y, int nCols, int nRows)
        => MatVecMul(type, weightData, x, y, nCols, nRows, null);

    /// <summary>
    /// Multiplies a quantized matrix W [nRows, nCols] by vector x [nCols] producing y [nRows] using precomputed block sums.
    /// </summary>
    public static void MatVecMul(GgufType type, byte* weightData, float* x, float* y, int nCols, int nRows, float* xSums)
    {
        if (type == GgufType.Q4_K && xSums == null && nCols >= 32)
        {
            int nChunks = nCols / 32;
            float* localSums = stackalloc float[nChunks];
            ComputeBlockSums32(x, localSums, nCols);
            xSums = localSums;
        }

        int rowBytes = (int)GgufTypes.GetRowBytes(type, nCols);
        int threads = Environment.ProcessorCount;

        if (nRows < threads * 2)
        {
            for (int i = 0; i < nRows; i++)
            {
                byte* rowPtr = weightData + (long)i * rowBytes;
                y[i] = ComputeDot(type, rowPtr, x, xSums, nCols);
            }
            return;
        }

        int rowsPerThread = (nRows + threads - 1) / threads;
        Parallel.For(0, threads, t =>
        {
            int startRow = t * rowsPerThread;
            int endRow = Math.Min(startRow + rowsPerThread, nRows);
            for (int i = startRow; i < endRow; i++)
            {
                byte* rowPtr = weightData + (long)i * rowBytes;
                y[i] = ComputeDot(type, rowPtr, x, xSums, nCols);
            }
        });
    }

    /// <summary>
    /// Multiplies a quantized matrix W [nRows, nCols] by a batch of vectors xBatch [batchSize, nCols]
    /// producing yBatch [batchSize, nRows].
    /// Weights are streamed from memory once and reused across all vectors in the batch.
    /// </summary>
    public static void MatMulBatch(
        GgufType type,
        byte* weightData,
        float* xBatch,
        float* yBatch,
        int nCols,
        int nRows,
        int batchSize,
        float* xSumsBatch = null)
    {
        if (batchSize == 1)
        {
            MatVecMul(type, weightData, xBatch, yBatch, nCols, nRows, xSumsBatch);
            return;
        }

        int rowBytes = (int)GgufTypes.GetRowBytes(type, nCols);
        int threads = Environment.ProcessorCount;
        int sumsStride = nCols / 32;

        if (nRows < threads * 2)
        {
            for (int r = 0; r < nRows; r++)
            {
                byte* rowPtr = weightData + (long)r * rowBytes;
                for (int b = 0; b < batchSize; b++)
                {
                    float* x = xBatch + b * nCols;
                    float* xSums = xSumsBatch != null ? xSumsBatch + b * sumsStride : null;
                    yBatch[b * nRows + r] = ComputeDot(type, rowPtr, x, xSums, nCols);
                }
            }
            return;
        }

        int rowsPerThread = (nRows + threads - 1) / threads;
        Parallel.For(0, threads, t =>
        {
            int startRow = t * rowsPerThread;
            int endRow = Math.Min(startRow + rowsPerThread, nRows);
            for (int r = startRow; r < endRow; r++)
            {
                byte* rowPtr = weightData + (long)r * rowBytes;
                for (int b = 0; b < batchSize; b++)
                {
                    float* x = xBatch + b * nCols;
                    float* xSums = xSumsBatch != null ? xSumsBatch + b * sumsStride : null;
                    yBatch[b * nRows + r] = ComputeDot(type, rowPtr, x, xSums, nCols);
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeDot(GgufType type, byte* rowPtr, float* x, float* xSums, int nCols)
    {
        return type switch
        {
            GgufType.Q4_K => VecDotQ4_K((BlockQ4_K*)rowPtr, x, xSums, nCols),
            GgufType.Q6_K => VecDotQ6_K((BlockQ6_K*)rowPtr, x, nCols),
            GgufType.Q8_0 => VecDotQ8_0((BlockQ8_0*)rowPtr, x, nCols),
            GgufType.Q4_0 => VecDotQ4_0((BlockQ4_0*)rowPtr, x, nCols),
            GgufType.Q5_0 => VecDotQ5_0((BlockQ5_0*)rowPtr, x, nCols),
            GgufType.Q5_K => VecDotQ5_K((BlockQ5_K*)rowPtr, x, nCols),
            GgufType.Q3_K => VecDotQ3_K((BlockQ3_K*)rowPtr, x, nCols),
            GgufType.MXFP4 => VecDotMXFP4((BlockMXFP4*)rowPtr, x, nCols),
            GgufType.F16 => VecDotF16((Half*)rowPtr, x, nCols),
            GgufType.F32 => VecDotF32((float*)rowPtr, x, nCols),
            _ => throw new NotSupportedException($"Quantization type {type} is not supported in hardware GEMV kernels.")
        };
    }

}
