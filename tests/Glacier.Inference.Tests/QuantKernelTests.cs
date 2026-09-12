namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Quant;
using Xunit;

public unsafe class QuantKernelTests
{
    [Fact]
    public void RMSNorm_NormalizesToUnitVariance()
    {
        const int size = 64;
        float[] x = new float[size];
        float[] weight = new float[size];
        float[] dst = new float[size];

        for (int i = 0; i < size; i++)
        {
            x[i] = (i % 5) - 2.0f; // varied inputs
            weight[i] = 1.0f;     // unit weights
        }

        fixed (float* pX = x, pW = weight, pDst = dst)
        {
            QuantKernels.RMSNorm(pX, pW, pDst, size, 1e-5f);
        }

        // Verify that sum(dst^2) / size ~= 1.0
        float sumSq = 0f;
        for (int i = 0; i < size; i++) sumSq += dst[i] * dst[i];
        float meanSq = sumSq / size;

        Assert.InRange(meanSq, 0.99f, 1.01f);
    }

    [Fact]
    public void Softmax_SumsToOneAndPreservesOrder()
    {
        float[] logits = [1.0f, 5.0f, 2.0f, -1.0f, 3.0f];
        fixed (float* p = logits)
        {
            QuantKernels.Softmax(p, logits.Length);
        }

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.True(logits[i] >= 0f);
            sum += logits[i];
        }

        Assert.InRange(sum, 0.999f, 1.001f);
        Assert.True(logits[1] > logits[4]); // index 1 had 5.0, index 4 had 3.0
    }

    [Fact]
    public void SwiGLU_ComputesAccurately()
    {
        float[] gate = [0.0f, 2.0f, -2.0f];
        float[] up = [1.0f, 3.0f, 2.0f];
        float[] dst = new float[3];

        fixed (float* pG = gate, pU = up, pDst = dst)
        {
            QuantKernels.SwiGLU(pG, pU, pDst, 3);
        }

        // silu(0) * 1 = 0 * 1 = 0
        Assert.Equal(0.0f, dst[0], 0.001f);
        // silu(2) * 3 = 2 / (1 + exp(-2)) * 3 ~= 1.761594 * 3 = 5.28478
        float expected = (2.0f / (1.0f + MathF.Exp(-2.0f))) * 3.0f;
        Assert.Equal(expected, dst[1], 0.001f);
    }

    [Fact]
    public void RoPE_RotatesVectorsOrthogonally()
    {
        const int headDim = 128;
        float[] q = new float[headDim];
        float[] k = new float[headDim];

        for (int i = 0; i < headDim; i++)
        {
            q[i] = 1.0f;
            k[i] = 1.0f;
        }

        float normBefore = 0f;
        for (int i = 0; i < headDim; i++) normBefore += q[i] * q[i];

        fixed (float* pQ = q, pK = k)
        {
            QuantKernels.RoPE(pQ, pK, nHeadsQ: 1, nHeadsKv: 1, headDim: headDim, pos: 5, freqBase: 10000.0f);
        }

        float normAfter = 0f;
        for (int i = 0; i < headDim; i++) normAfter += q[i] * q[i];

        // Rotation is an orthogonal isometry, so vector L2 norm must be preserved
        Assert.Equal(normBefore, normAfter, 0.01f);
    }

    [Fact]
    public void VecDotF32_ComputesExactDotProduct()
    {
        float[] a = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f];
        float[] b = [2.0f, 0.5f, 1.0f, -1.0f, 2.0f, 0.0f, 1.0f, 2.0f];
        // 2 + 1 + 3 - 4 + 10 + 0 + 7 + 16 = 35

        fixed (float* pA = a, pB = b)
        {
            float dot = QuantKernels.VecDotF32(pA, pB, a.Length);
            Assert.Equal(35.0f, dot, 0.001f);
        }
    }

    [Fact]
    public void ComputeBlockSums32_ComputesExactChunkSums()
    {
        const int nCols = 64;
        float[] x = new float[nCols];
        for (int i = 0; i < nCols; i++) x[i] = i + 1;

        float[] sums = new float[nCols / 32];
        fixed (float* pX = x, pSums = sums)
        {
            QuantKernels.ComputeBlockSums32(pX, pSums, nCols);
        }

        // Sum of 1..32 is (32 * 33) / 2 = 528
        // Sum of 33..64 is 528 + 32 * 32 = 1552
        Assert.Equal(528f, sums[0], 0.001f);
        Assert.Equal(1552f, sums[1], 0.001f);
    }

    [Fact]
    public void MatMulBatch_MatchesMatVecMul()
    {
        const int nCols = 256;
        const int nRows = 8;
        const int batchSize = 4;

        // Allocate a dummy Q4_K row
        int rowBytes = (int)Glacier.Inference.Gguf.GgufTypes.GetRowBytes(Glacier.Inference.Gguf.GgufType.Q4_K, nCols);
        byte[] weights = new byte[rowBytes * nRows];
        new Random(42).NextBytes(weights);

        float[] xBatch = new float[batchSize * nCols];
        for (int i = 0; i < xBatch.Length; i++) xBatch[i] = (i % 17) - 8.0f;

        float[] yBatch = new float[batchSize * nRows];
        float[] ySingle = new float[batchSize * nRows];

        fixed (byte* pW = weights)
        fixed (float* pX = xBatch, pYBatch = yBatch, pYSingle = ySingle)
        {
            // Compute via MatMulBatch
            QuantKernels.MatMulBatch(Glacier.Inference.Gguf.GgufType.Q4_K, pW, pX, pYBatch, nCols, nRows, batchSize);

            // Compute individually via MatVecMul
            for (int b = 0; b < batchSize; b++)
            {
                QuantKernels.MatVecMul(Glacier.Inference.Gguf.GgufType.Q4_K, pW, pX + b * nCols, pYSingle + b * nRows, nCols, nRows);
            }
        }

        for (int i = 0; i < yBatch.Length; i++)
        {
            Assert.Equal(ySingle[i], yBatch[i], 0.0001f);
        }
    }
}

