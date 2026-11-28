namespace Glacier.Inference.Quant;

using System;
using System.Diagnostics;
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
        if (GgufTypes.IsTrellis(type))
        {
            if (TrellisWeightRegistry.TryGetBanded(weightData, out var banded) && banded != null)
            {
                TrellisKernels.MatVecMulBanded(banded, x, y);
                return;
            }
            if (!TrellisWeightRegistry.TryGet(weightData, out var trellis) || trellis == null)
                throw new InvalidOperationException($"NariQ trellis weight ({type}) was not registered in TrellisWeightRegistry.");
            TrellisKernels.MatVecMul(trellis, x, y);
            return;
        }

        Debug.Assert(nCols > 0 && (type is GgufType.F32 or GgufType.F16 or GgufType.BF16 || nCols % 32 == 0), $"Matrix columns ({nCols}) must be a positive multiple of 32 for quantized SIMD dispatch.");
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
        if (GgufTypes.IsTrellis(type))
        {
            if (TrellisWeightRegistry.TryGetBanded(weightData, out var banded) && banded != null)
            {
                for (int b = 0; b < batchSize; b++)
                    TrellisKernels.MatVecMulBanded(banded, xBatch + (long)b * nCols, yBatch + (long)b * nRows);
                return;
            }
            if (!TrellisWeightRegistry.TryGet(weightData, out var trellis) || trellis == null)
                throw new InvalidOperationException($"NariQ trellis weight ({type}) was not registered in TrellisWeightRegistry.");
            for (int b = 0; b < batchSize; b++)
                TrellisKernels.MatVecMul(trellis, xBatch + (long)b * nCols, yBatch + (long)b * nRows);
            return;
        }

        Debug.Assert(nCols > 0 && (type is GgufType.F32 or GgufType.F16 or GgufType.BF16 || nCols % 32 == 0), $"Matrix columns ({nCols}) must be a positive multiple of 32 for quantized SIMD dispatch.");
        if (batchSize == 1)
        {
            MatVecMul(type, weightData, xBatch, yBatch, nCols, nRows, xSumsBatch);
            return;
        }

        int rowBytes = (int)GgufTypes.GetRowBytes(type, nCols);
        int threads = Environment.ProcessorCount;
        int sumsStride = nCols / 32;

        bool allocatedSums = false;
        if (type == GgufType.Q4_K && xSumsBatch == null && nCols >= 32)
        {
            xSumsBatch = (float*)NativeMemory.AlignedAlloc((nuint)((long)batchSize * sumsStride * sizeof(float)), 64);
            allocatedSums = true;
            for (int b = 0; b < batchSize; b++)
            {
                ComputeBlockSums32(xBatch + (long)b * nCols, xSumsBatch + (long)b * sumsStride, nCols);
            }
        }

        try
        {
            if (nRows < threads * 2)
            {
                MatMulBatchRowRange(type, weightData, xBatch, yBatch, xSumsBatch, rowBytes, nCols, nRows, batchSize, sumsStride, 0, nRows);
                return;
            }

            int rowsPerThread = (nRows + threads - 1) / threads;
            Parallel.For(0, threads, t =>
            {
                int startRow = t * rowsPerThread;
                int endRow = Math.Min(startRow + rowsPerThread, nRows);
                MatMulBatchRowRange(type, weightData, xBatch, yBatch, xSumsBatch, rowBytes, nCols, nRows, batchSize, sumsStride, startRow, endRow);
            });
        }
        finally
        {
            if (allocatedSums && xSumsBatch != null)
            {
                NativeMemory.AlignedFree(xSumsBatch);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MatMulBatchRowRange(
        GgufType type,
        byte* weightData,
        float* xBatch,
        float* yBatch,
        float* xSumsBatch,
        int rowBytes,
        int nCols,
        int nRows,
        int batchSize,
        int sumsStride,
        int startRow,
        int endRow)
    {
        for (int r = startRow; r < endRow; r++)
        {
            byte* rowPtr = weightData + (long)r * rowBytes;
            int b = 0;

            if (batchSize >= 4 && SupportsTiled4(type))
            {
                for (; b <= batchSize - 4; b += 4)
                {
                    float* x0 = xBatch + (long)(b + 0) * nCols;
                    float* x1 = xBatch + (long)(b + 1) * nCols;
                    float* x2 = xBatch + (long)(b + 2) * nCols;
                    float* x3 = xBatch + (long)(b + 3) * nCols;
                    float* s0 = xSumsBatch != null ? xSumsBatch + (long)(b + 0) * sumsStride : null;
                    float* s1 = xSumsBatch != null ? xSumsBatch + (long)(b + 1) * sumsStride : null;
                    float* s2 = xSumsBatch != null ? xSumsBatch + (long)(b + 2) * sumsStride : null;
                    float* s3 = xSumsBatch != null ? xSumsBatch + (long)(b + 3) * sumsStride : null;

                    ComputeDot4(
                        type, rowPtr, x0, x1, x2, x3, s0, s1, s2, s3, nCols,
                        out float y0, out float y1, out float y2, out float y3);

                    yBatch[(long)(b + 0) * nRows + r] = y0;
                    yBatch[(long)(b + 1) * nRows + r] = y1;
                    yBatch[(long)(b + 2) * nRows + r] = y2;
                    yBatch[(long)(b + 3) * nRows + r] = y3;
                }
            }

            for (; b < batchSize; b++)
            {
                float* x = xBatch + (long)b * nCols;
                float* xSums = xSumsBatch != null ? xSumsBatch + (long)b * sumsStride : null;
                yBatch[(long)b * nRows + r] = ComputeDot(type, rowPtr, x, xSums, nCols);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool SupportsTiled4(GgufType type) =>
        type is GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.Q4_0 or GgufType.F16 or GgufType.F32;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeDot4(
        GgufType type,
        byte* rowPtr,
        float* x0, float* x1, float* x2, float* x3,
        float* s0, float* s1, float* s2, float* s3,
        int nCols,
        out float y0, out float y1, out float y2, out float y3)
    {
        switch (type)
        {
            case GgufType.Q4_K:
                VecDot4Q4_K((BlockQ4_K*)rowPtr, x0, x1, x2, x3, s0, s1, s2, s3, nCols, out y0, out y1, out y2, out y3);
                break;
            case GgufType.Q6_K:
                VecDot4Q6_K((BlockQ6_K*)rowPtr, x0, x1, x2, x3, nCols, out y0, out y1, out y2, out y3);
                break;
            case GgufType.Q8_0:
                VecDot4Q8_0((BlockQ8_0*)rowPtr, x0, x1, x2, x3, nCols, out y0, out y1, out y2, out y3);
                break;
            case GgufType.Q4_0:
                VecDot4Q4_0((BlockQ4_0*)rowPtr, x0, x1, x2, x3, nCols, out y0, out y1, out y2, out y3);
                break;
            case GgufType.F16:
                VecDot4F16((Half*)rowPtr, x0, x1, x2, x3, nCols, out y0, out y1, out y2, out y3);
                break;
            case GgufType.F32:
                VecDot4F32((float*)rowPtr, x0, x1, x2, x3, nCols, out y0, out y1, out y2, out y3);
                break;
            default:
                y0 = ComputeDot(type, rowPtr, x0, s0, nCols);
                y1 = ComputeDot(type, rowPtr, x1, s1, nCols);
                y2 = ComputeDot(type, rowPtr, x2, s2, nCols);
                y3 = ComputeDot(type, rowPtr, x3, s3, nCols);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4F32(
        float* w, float* x0, float* x1, float* x2, float* x3, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int c = 0;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;
        if (Vector512.IsHardwareAccelerated)
        {
            var a0 = Vector512<float>.Zero;
            var a1 = Vector512<float>.Zero;
            var a2 = Vector512<float>.Zero;
            var a3 = Vector512<float>.Zero;
            for (; c <= k - 16; c += 16)
            {
                var vw = Vector512.Load(w + c);
                a0 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(x0 + c), a0);
                a1 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(x1 + c), a1);
                a2 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(x2 + c), a2);
                a3 = Vector512.FusedMultiplyAdd(vw, Vector512.Load(x3 + c), a3);
            }
            r0 = Vector512.Sum(a0);
            r1 = Vector512.Sum(a1);
            r2 = Vector512.Sum(a2);
            r3 = Vector512.Sum(a3);
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = Vector256<float>.Zero;
            var a2 = Vector256<float>.Zero;
            var a3 = Vector256<float>.Zero;
            for (; c <= k - 8; c += 8)
            {
                var vw = Vector256.Load(w + c);
                a0 += vw * Vector256.Load(x0 + c);
                a1 += vw * Vector256.Load(x1 + c);
                a2 += vw * Vector256.Load(x2 + c);
                a3 += vw * Vector256.Load(x3 + c);
            }
            r0 = Vector256.Sum(a0);
            r1 = Vector256.Sum(a1);
            r2 = Vector256.Sum(a2);
            r3 = Vector256.Sum(a3);
        }
        for (; c < k; c++)
        {
            float wv = w[c];
            r0 += wv * x0[c];
            r1 += wv * x1[c];
            r2 += wv * x2[c];
            r3 += wv * x3[c];
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4F16(
        Half* w, float* x0, float* x1, float* x2, float* x3, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int c = 0;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;
        if (Vector256.IsHardwareAccelerated)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = Vector256<float>.Zero;
            var a2 = Vector256<float>.Zero;
            var a3 = Vector256<float>.Zero;
            for (; c <= k - 8; c += 8)
            {
                var vw = Vector256.Create(
                    (float)w[c + 0], (float)w[c + 1], (float)w[c + 2], (float)w[c + 3],
                    (float)w[c + 4], (float)w[c + 5], (float)w[c + 6], (float)w[c + 7]);
                a0 = Fma.IsSupported ? Fma.MultiplyAdd(vw, Vector256.Load(x0 + c), a0) : a0 + vw * Vector256.Load(x0 + c);
                a1 = Fma.IsSupported ? Fma.MultiplyAdd(vw, Vector256.Load(x1 + c), a1) : a1 + vw * Vector256.Load(x1 + c);
                a2 = Fma.IsSupported ? Fma.MultiplyAdd(vw, Vector256.Load(x2 + c), a2) : a2 + vw * Vector256.Load(x2 + c);
                a3 = Fma.IsSupported ? Fma.MultiplyAdd(vw, Vector256.Load(x3 + c), a3) : a3 + vw * Vector256.Load(x3 + c);
            }
            r0 = Vector256.Sum(a0);
            r1 = Vector256.Sum(a1);
            r2 = Vector256.Sum(a2);
            r3 = Vector256.Sum(a3);
        }
        for (; c < k; c++)
        {
            float wv = (float)w[c];
            r0 += wv * x0[c];
            r1 += wv * x1[c];
            r2 += wv * x2[c];
            r3 += wv * x3[c];
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4Q8_0(
        BlockQ8_0* row, float* x0, float* x1, float* x2, float* x3, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int nb = k / QK8_0;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            sbyte* q = row[i].Qs;
            float b0 = 0f, b1 = 0f, b2 = 0f, b3 = 0f;

            if (Vector512.IsHardwareAccelerated)
            {
                var q32 = Vector256.Load(q);
                var vq0 = WidenSByteToFloat512(q32.GetLower());
                var vq1 = WidenSByteToFloat512(q32.GetUpper());
                b0 = Vector512.Sum(Vector512.FusedMultiplyAdd(vq0, Vector512.Load(x0), vq1 * Vector512.Load(x0 + 16)));
                b1 = Vector512.Sum(Vector512.FusedMultiplyAdd(vq0, Vector512.Load(x1), vq1 * Vector512.Load(x1 + 16)));
                b2 = Vector512.Sum(Vector512.FusedMultiplyAdd(vq0, Vector512.Load(x2), vq1 * Vector512.Load(x2 + 16)));
                b3 = Vector512.Sum(Vector512.FusedMultiplyAdd(vq0, Vector512.Load(x3), vq1 * Vector512.Load(x3 + 16)));
            }
            else if (Avx2.IsSupported)
            {
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;
                for (int l = 0; l < 32; l += 8)
                {
                    var vq = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Vector128.CreateScalar(*(ulong*)(q + l)).As<ulong, sbyte>()));
                    a0 = Fma.IsSupported ? Fma.MultiplyAdd(vq, Vector256.Load(x0 + l), a0) : a0 + vq * Vector256.Load(x0 + l);
                    a1 = Fma.IsSupported ? Fma.MultiplyAdd(vq, Vector256.Load(x1 + l), a1) : a1 + vq * Vector256.Load(x1 + l);
                    a2 = Fma.IsSupported ? Fma.MultiplyAdd(vq, Vector256.Load(x2 + l), a2) : a2 + vq * Vector256.Load(x2 + l);
                    a3 = Fma.IsSupported ? Fma.MultiplyAdd(vq, Vector256.Load(x3 + l), a3) : a3 + vq * Vector256.Load(x3 + l);
                }
                b0 = Vector256.Sum(a0);
                b1 = Vector256.Sum(a1);
                b2 = Vector256.Sum(a2);
                b3 = Vector256.Sum(a3);
            }
            else
            {
                for (int l = 0; l < 32; l++)
                {
                    float qv = q[l];
                    b0 += qv * x0[l];
                    b1 += qv * x1[l];
                    b2 += qv * x2[l];
                    b3 += qv * x3[l];
                }
            }

            r0 += d * b0;
            r1 += d * b1;
            r2 += d * b2;
            r3 += d * b3;
            x0 += 32; x1 += 32; x2 += 32; x3 += 32;
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4Q4_0(
        BlockQ4_0* row, float* x0, float* x1, float* x2, float* x3, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int nb = k / QK4_0;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;
        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            byte* q = row[i].Qs;
            float b0 = 0f, b1 = 0f, b2 = 0f, b3 = 0f;

            if (Vector512.IsHardwareAccelerated)
            {
                var raw = Vector128.Load(q);
                var mask0F = Vector128.Create((byte)0x0F);
                var v8 = Vector512.Create(8.0f);
                var vqLo = WidenUInt8ToFloat512(raw & mask0F) - v8;
                var vqHi = WidenUInt8ToFloat512(Vector128.ShiftRightLogical(raw.AsUInt16(), 4).AsByte() & mask0F) - v8;

                b0 = Vector512.Sum(Vector512.FusedMultiplyAdd(vqLo, Vector512.Load(x0), vqHi * Vector512.Load(x0 + 16)));
                b1 = Vector512.Sum(Vector512.FusedMultiplyAdd(vqLo, Vector512.Load(x1), vqHi * Vector512.Load(x1 + 16)));
                b2 = Vector512.Sum(Vector512.FusedMultiplyAdd(vqLo, Vector512.Load(x2), vqHi * Vector512.Load(x2 + 16)));
                b3 = Vector512.Sum(Vector512.FusedMultiplyAdd(vqLo, Vector512.Load(x3), vqHi * Vector512.Load(x3 + 16)));
            }
            else
            {
                for (int l = 0; l < 16; l++)
                {
                    float qLo = (q[l] & 0x0F) - 8;
                    float qHi = (q[l] >> 4) - 8;
                    b0 += qLo * x0[l] + qHi * x0[l + 16];
                    b1 += qLo * x1[l] + qHi * x1[l + 16];
                    b2 += qLo * x2[l] + qHi * x2[l + 16];
                    b3 += qLo * x3[l] + qHi * x3[l + 16];
                }
            }

            r0 += d * b0;
            r1 += d * b1;
            r2 += d * b2;
            r3 += d * b3;
            x0 += 32; x1 += 32; x2 += 32; x3 += 32;
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4Q4_K(
        BlockQ4_K* row,
        float* x0, float* x1, float* x2, float* x3,
        float* xSums0, float* xSums1, float* xSums2, float* xSums3,
        int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int nb = k / QK_K;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;
        int sumIdx = 0;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            float min = (float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* q = row[i].Qs;

            int isIdx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0, m0, sc1, m1Val;
                if (isIdx < 4)
                {
                    sc0 = (byte)(scales[isIdx] & 63);
                    m0 = (byte)(scales[isIdx + 4] & 63);
                    sc1 = (byte)(scales[isIdx + 1] & 63);
                    m1Val = (byte)(scales[isIdx + 5] & 63);
                }
                else
                {
                    sc0 = (byte)((scales[isIdx + 4] & 0x0F) | ((scales[isIdx - 4] >> 6) << 4));
                    m0 = (byte)((scales[isIdx + 4] >> 4) | ((scales[isIdx] >> 6) << 4));
                    sc1 = (byte)((scales[isIdx + 5] & 0x0F) | ((scales[isIdx - 3] >> 6) << 4));
                    m1Val = (byte)((scales[isIdx + 5] >> 4) | ((scales[isIdx + 1] >> 6) << 4));
                }

                float d1 = d * sc0;
                float m1 = min * m0;
                float d2 = d * sc1;
                float m2 = min * m1Val;

                if (Vector512.IsHardwareAccelerated)
                {
                    var raw32 = Vector256.Load(q);
                    var mask15 = Vector256.Create((byte)0x0F);
                    var lo32 = raw32 & mask15;
                    var hi32 = Vector256.ShiftRightLogical(raw32.AsUInt16(), 4).AsByte() & mask15;

                    var vqLo0 = WidenUInt8ToFloat512(lo32.GetLower());
                    var vqLo1 = WidenUInt8ToFloat512(lo32.GetUpper());
                    int sLo = sumIdx++;
                    r0 += d1 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x0), vqLo0, Vector512.Load(x0 + 16) * vqLo1)) - m1 * (xSums0 != null ? xSums0[sLo] : ComputeChunk32Sum(x0));
                    r1 += d1 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x1), vqLo0, Vector512.Load(x1 + 16) * vqLo1)) - m1 * (xSums1 != null ? xSums1[sLo] : ComputeChunk32Sum(x1));
                    r2 += d1 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x2), vqLo0, Vector512.Load(x2 + 16) * vqLo1)) - m1 * (xSums2 != null ? xSums2[sLo] : ComputeChunk32Sum(x2));
                    r3 += d1 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x3), vqLo0, Vector512.Load(x3 + 16) * vqLo1)) - m1 * (xSums3 != null ? xSums3[sLo] : ComputeChunk32Sum(x3));
                    x0 += 32; x1 += 32; x2 += 32; x3 += 32;

                    var vqHi0 = WidenUInt8ToFloat512(hi32.GetLower());
                    var vqHi1 = WidenUInt8ToFloat512(hi32.GetUpper());
                    int sHi = sumIdx++;
                    r0 += d2 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x0), vqHi0, Vector512.Load(x0 + 16) * vqHi1)) - m2 * (xSums0 != null ? xSums0[sHi] : ComputeChunk32Sum(x0));
                    r1 += d2 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x1), vqHi0, Vector512.Load(x1 + 16) * vqHi1)) - m2 * (xSums1 != null ? xSums1[sHi] : ComputeChunk32Sum(x1));
                    r2 += d2 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x2), vqHi0, Vector512.Load(x2 + 16) * vqHi1)) - m2 * (xSums2 != null ? xSums2[sHi] : ComputeChunk32Sum(x2));
                    r3 += d2 * Vector512.Sum(Vector512.FusedMultiplyAdd(Vector512.Load(x3), vqHi0, Vector512.Load(x3 + 16) * vqHi1)) - m2 * (xSums3 != null ? xSums3[sHi] : ComputeChunk32Sum(x3));
                    x0 += 32; x1 += 32; x2 += 32; x3 += 32;
                }
                else
                {
                    float ss0 = 0f, ss1 = 0f, ss2 = 0f, ss3 = 0f;
                    for (int l = 0; l < 32; l++)
                    {
                        float qv = q[l] & 0x0F;
                        ss0 += qv * x0[l];
                        ss1 += qv * x1[l];
                        ss2 += qv * x2[l];
                        ss3 += qv * x3[l];
                    }
                    int sLo = sumIdx++;
                    r0 += d1 * ss0 - m1 * (xSums0 != null ? xSums0[sLo] : ComputeChunk32Sum(x0));
                    r1 += d1 * ss1 - m1 * (xSums1 != null ? xSums1[sLo] : ComputeChunk32Sum(x1));
                    r2 += d1 * ss2 - m1 * (xSums2 != null ? xSums2[sLo] : ComputeChunk32Sum(x2));
                    r3 += d1 * ss3 - m1 * (xSums3 != null ? xSums3[sLo] : ComputeChunk32Sum(x3));
                    x0 += 32; x1 += 32; x2 += 32; x3 += 32;

                    ss0 = 0f; ss1 = 0f; ss2 = 0f; ss3 = 0f;
                    for (int l = 0; l < 32; l++)
                    {
                        float qv = q[l] >> 4;
                        ss0 += qv * x0[l];
                        ss1 += qv * x1[l];
                        ss2 += qv * x2[l];
                        ss3 += qv * x3[l];
                    }
                    int sHi = sumIdx++;
                    r0 += d2 * ss0 - m2 * (xSums0 != null ? xSums0[sHi] : ComputeChunk32Sum(x0));
                    r1 += d2 * ss1 - m2 * (xSums1 != null ? xSums1[sHi] : ComputeChunk32Sum(x1));
                    r2 += d2 * ss2 - m2 * (xSums2 != null ? xSums2[sHi] : ComputeChunk32Sum(x2));
                    r3 += d2 * ss3 - m2 * (xSums3 != null ? xSums3[sHi] : ComputeChunk32Sum(x3));
                    x0 += 32; x1 += 32; x2 += 32; x3 += 32;
                }

                q += 32;
                isIdx += 2;
            }
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDot4Q6_K(
        BlockQ6_K* row, float* x0, float* x1, float* x2, float* x3, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        int nb = k / QK_K;
        float r0 = 0f, r1 = 0f, r2 = 0f, r3 = 0f;

        for (int i = 0; i < nb; i++)
        {
            float d = (float)row[i].Delta;
            byte* ql = row[i].Ql;
            byte* qh = row[i].Qh;
            sbyte* sc = row[i].Scales;

            for (int n = 0; n < QK_K; n += 128)
            {
                if (Vector512.IsHardwareAccelerated)
                {
                    var ql0 = Vector256.Load(ql);
                    var ql1 = Vector256.Load(ql + 32);
                    var qh0 = Vector256.Load(qh);
                    var mask0F = Vector256.Create((byte)0x0F);
                    var mask03 = Vector256.Create((byte)0x03);
                    var v32 = Vector512.Create(32.0f);

                    var q1Bytes = (ql0 & mask0F) | ((qh0 & mask03) << 4);
                    var q2Bytes = (ql1 & mask0F) | ((Vector256.ShiftRightLogical(qh0.AsUInt16(), 2).AsByte() & mask03) << 4);
                    var q3Bytes = (Vector256.ShiftRightLogical(ql0.AsUInt16(), 4).AsByte() & mask0F) | ((Vector256.ShiftRightLogical(qh0.AsUInt16(), 4).AsByte() & mask03) << 4);
                    var q4Bytes = (Vector256.ShiftRightLogical(ql1.AsUInt16(), 4).AsByte() & mask0F) | ((Vector256.ShiftRightLogical(qh0.AsUInt16(), 6).AsByte() & mask03) << 4);

                    var vq1a = WidenUInt8ToFloat512(q1Bytes.GetLower()) - v32;
                    var vq1b = WidenUInt8ToFloat512(q1Bytes.GetUpper()) - v32;
                    var vq2a = WidenUInt8ToFloat512(q2Bytes.GetLower()) - v32;
                    var vq2b = WidenUInt8ToFloat512(q2Bytes.GetUpper()) - v32;
                    var vq3a = WidenUInt8ToFloat512(q3Bytes.GetLower()) - v32;
                    var vq3b = WidenUInt8ToFloat512(q3Bytes.GetUpper()) - v32;
                    var vq4a = WidenUInt8ToFloat512(q4Bytes.GetLower()) - v32;
                    var vq4b = WidenUInt8ToFloat512(q4Bytes.GetUpper()) - v32;

                    float ds0 = d * sc[0], ds1 = d * sc[1], ds2 = d * sc[2], ds3 = d * sc[3];
                    float ds4 = d * sc[4], ds5 = d * sc[5], ds6 = d * sc[6], ds7 = d * sc[7];

                    r0 += ds0 * Vector512.Sum(Vector512.Load(x0 + 0) * vq1a) + ds1 * Vector512.Sum(Vector512.Load(x0 + 16) * vq1b)
                        + ds2 * Vector512.Sum(Vector512.Load(x0 + 32) * vq2a) + ds3 * Vector512.Sum(Vector512.Load(x0 + 48) * vq2b)
                        + ds4 * Vector512.Sum(Vector512.Load(x0 + 64) * vq3a) + ds5 * Vector512.Sum(Vector512.Load(x0 + 80) * vq3b)
                        + ds6 * Vector512.Sum(Vector512.Load(x0 + 96) * vq4a) + ds7 * Vector512.Sum(Vector512.Load(x0 + 112) * vq4b);

                    r1 += ds0 * Vector512.Sum(Vector512.Load(x1 + 0) * vq1a) + ds1 * Vector512.Sum(Vector512.Load(x1 + 16) * vq1b)
                        + ds2 * Vector512.Sum(Vector512.Load(x1 + 32) * vq2a) + ds3 * Vector512.Sum(Vector512.Load(x1 + 48) * vq2b)
                        + ds4 * Vector512.Sum(Vector512.Load(x1 + 64) * vq3a) + ds5 * Vector512.Sum(Vector512.Load(x1 + 80) * vq3b)
                        + ds6 * Vector512.Sum(Vector512.Load(x1 + 96) * vq4a) + ds7 * Vector512.Sum(Vector512.Load(x1 + 112) * vq4b);

                    r2 += ds0 * Vector512.Sum(Vector512.Load(x2 + 0) * vq1a) + ds1 * Vector512.Sum(Vector512.Load(x2 + 16) * vq1b)
                        + ds2 * Vector512.Sum(Vector512.Load(x2 + 32) * vq2a) + ds3 * Vector512.Sum(Vector512.Load(x2 + 48) * vq2b)
                        + ds4 * Vector512.Sum(Vector512.Load(x2 + 64) * vq3a) + ds5 * Vector512.Sum(Vector512.Load(x2 + 80) * vq3b)
                        + ds6 * Vector512.Sum(Vector512.Load(x2 + 96) * vq4a) + ds7 * Vector512.Sum(Vector512.Load(x2 + 112) * vq4b);

                    r3 += ds0 * Vector512.Sum(Vector512.Load(x3 + 0) * vq1a) + ds1 * Vector512.Sum(Vector512.Load(x3 + 16) * vq1b)
                        + ds2 * Vector512.Sum(Vector512.Load(x3 + 32) * vq2a) + ds3 * Vector512.Sum(Vector512.Load(x3 + 48) * vq2b)
                        + ds4 * Vector512.Sum(Vector512.Load(x3 + 64) * vq3a) + ds5 * Vector512.Sum(Vector512.Load(x3 + 80) * vq3b)
                        + ds6 * Vector512.Sum(Vector512.Load(x3 + 96) * vq4a) + ds7 * Vector512.Sum(Vector512.Load(x3 + 112) * vq4b);
                }
                else
                {
                    float s0a = 0f, s0b = 0f, s0c = 0f, s0d = 0f, s0e = 0f, s0f = 0f, s0g = 0f, s0h = 0f;
                    float s1a = 0f, s1b = 0f, s1c = 0f, s1d = 0f, s1e = 0f, s1f = 0f, s1g = 0f, s1h = 0f;
                    float s2a = 0f, s2b = 0f, s2c = 0f, s2d = 0f, s2e = 0f, s2f = 0f, s2g = 0f, s2h = 0f;
                    float s3a = 0f, s3b = 0f, s3c = 0f, s3d = 0f, s3e = 0f, s3f = 0f, s3g = 0f, s3h = 0f;
                    for (int l = 0; l < 16; l++)
                    {
                        float q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        float q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        float q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        float q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        float q1b = ((ql[l + 16] & 0x0F) | (((qh[l + 16] >> 0) & 3) << 4)) - 32;
                        float q2b = ((ql[l + 48] & 0x0F) | (((qh[l + 16] >> 2) & 3) << 4)) - 32;
                        float q3b = ((ql[l + 16] >> 4) | (((qh[l + 16] >> 4) & 3) << 4)) - 32;
                        float q4b = ((ql[l + 48] >> 4) | (((qh[l + 16] >> 6) & 3) << 4)) - 32;

                        s0a += q1 * x0[l]; s0b += q1b * x0[l + 16]; s0c += q2 * x0[l + 32]; s0d += q2b * x0[l + 48];
                        s0e += q3 * x0[l + 64]; s0f += q3b * x0[l + 80]; s0g += q4 * x0[l + 96]; s0h += q4b * x0[l + 112];

                        s1a += q1 * x1[l]; s1b += q1b * x1[l + 16]; s1c += q2 * x1[l + 32]; s1d += q2b * x1[l + 48];
                        s1e += q3 * x1[l + 64]; s1f += q3b * x1[l + 80]; s1g += q4 * x1[l + 96]; s1h += q4b * x1[l + 112];

                        s2a += q1 * x2[l]; s2b += q1b * x2[l + 16]; s2c += q2 * x2[l + 32]; s2d += q2b * x2[l + 48];
                        s2e += q3 * x2[l + 64]; s2f += q3b * x2[l + 80]; s2g += q4 * x2[l + 96]; s2h += q4b * x2[l + 112];

                        s3a += q1 * x3[l]; s3b += q1b * x3[l + 16]; s3c += q2 * x3[l + 32]; s3d += q2b * x3[l + 48];
                        s3e += q3 * x3[l + 64]; s3f += q3b * x3[l + 80]; s3g += q4 * x3[l + 96]; s3h += q4b * x3[l + 112];
                    }
                    r0 += d * (sc[0] * s0a + sc[1] * s0b + sc[2] * s0c + sc[3] * s0d + sc[4] * s0e + sc[5] * s0f + sc[6] * s0g + sc[7] * s0h);
                    r1 += d * (sc[0] * s1a + sc[1] * s1b + sc[2] * s1c + sc[3] * s1d + sc[4] * s1e + sc[5] * s1f + sc[6] * s1g + sc[7] * s1h);
                    r2 += d * (sc[0] * s2a + sc[1] * s2b + sc[2] * s2c + sc[3] * s2d + sc[4] * s2e + sc[5] * s2f + sc[6] * s2g + sc[7] * s2h);
                    r3 += d * (sc[0] * s3a + sc[1] * s3b + sc[2] * s3c + sc[3] * s3d + sc[4] * s3e + sc[5] * s3f + sc[6] * s3g + sc[7] * s3h);
                }

                x0 += 128; x1 += 128; x2 += 128; x3 += 128;
                ql += 64; qh += 32; sc += 8;
            }
        }
        y0 = r0; y1 = r1; y2 = r2; y3 = r3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float ComputeDot(GgufType type, byte* rowPtr, float* x, float* xSums, int nCols)
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
            GgufType.Q2_K => VecDotQ2_K((BlockQ2_K*)rowPtr, x, nCols),
            GgufType.Q2_0 => VecDotQ2_0((BlockQ2_0*)rowPtr, x, nCols),
            GgufType.MXFP4 => VecDotMXFP4((BlockMXFP4*)rowPtr, x, nCols),
            GgufType.NVFP4 => VecDotNVFP4((BlockNVFP4*)rowPtr, x, nCols),
            GgufType.GgmlQ2_0 => VecDotGgmlQ2_0((BlockGgmlQ2_0*)rowPtr, x, nCols),
            GgufType.IQ4_XS => VecDotIQ4_XS((BlockIQ4_XS*)rowPtr, x, nCols),
            GgufType.IQ4_NL => VecDotIQ4_NL((BlockIQ4_NL*)rowPtr, x, nCols),
            GgufType.IQ3_S => VecDotIQ3_S((BlockIQ3_S*)rowPtr, x, nCols),
            GgufType.IQ3_XXS => VecDotIQ3_XXS((BlockIQ3_XXS*)rowPtr, x, nCols),
            GgufType.IQ2_XXS => VecDotIQ2_XXS((BlockIQ2_XXS*)rowPtr, x, nCols),
            GgufType.IQ2_XS => VecDotIQ2_XS((BlockIQ2_XS*)rowPtr, x, nCols),
            GgufType.IQ2_S => VecDotIQ2_S((BlockIQ2_S*)rowPtr, x, nCols),
            GgufType.BF16 => VecDotBF16((ushort*)rowPtr, x, nCols),
            GgufType.F16 => VecDotF16((Half*)rowPtr, x, nCols),
            GgufType.F32 => VecDotF32((float*)rowPtr, x, nCols),
            _ => throw new NotSupportedException($"Quantization type {type} is not supported in hardware GEMV kernels.")
        };
    }

}
