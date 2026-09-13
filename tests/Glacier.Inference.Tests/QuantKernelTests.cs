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

    [Fact]
    public void RouterTopK_SelectsTopK_And_NormalizesToUnitSum()
    {
        const int dim = 64;
        const int expertCount = 8;
        const int topK = 2;

        float[] x = new float[dim];
        for (int i = 0; i < dim; i++) x[i] = 1.0f;

        // Create router weights where expert 3 has highest dot product, expert 6 has second highest
        float[] weights = new float[expertCount * dim];
        for (int i = 0; i < dim; i++)
        {
            weights[3 * dim + i] = 2.0f; // Dot = 128
            weights[6 * dim + i] = 1.5f; // Dot = 96
            weights[1 * dim + i] = 0.5f; // Dot = 32
        }

        int[] selectedIndices = new int[topK];
        float[] selectedWeights = new float[topK];

        fixed (float* pX = x, pW = weights, pWeightsOut = selectedWeights)
        fixed (int* pIdxOut = selectedIndices)
        {
            QuantKernels.RouterTopK(pX, pW, null, dim, expertCount, topK, pIdxOut, pWeightsOut);
        }

        // Top-2 should be expert 3 and expert 6
        Assert.Equal(3, selectedIndices[0]);
        Assert.Equal(6, selectedIndices[1]);

        // Weights should sum to 1.0
        float sum = selectedWeights[0] + selectedWeights[1];
        Assert.InRange(sum, 0.999f, 1.001f);
        Assert.True(selectedWeights[0] > selectedWeights[1]);
    }

    [Fact]
    public void RouterTopK_IncorporatesBias_Correctly()
    {
        const int dim = 32;
        const int expertCount = 4;
        const int topK = 1;

        float[] x = new float[dim];
        float[] weights = new float[expertCount * dim]; // all zeros -> dot = 0
        float[] bias = [1.0f, 5.0f, 2.0f, 0.0f]; // expert 1 has highest bias

        int[] selectedIndices = new int[topK];
        float[] selectedWeights = new float[topK];

        fixed (float* pX = x, pW = weights, pB = bias, pWeightsOut = selectedWeights)
        fixed (int* pIdxOut = selectedIndices)
        {
            QuantKernels.RouterTopK(pX, pW, pB, dim, expertCount, topK, pIdxOut, pWeightsOut);
        }

        Assert.Equal(1, selectedIndices[0]);
        Assert.Equal(1.0f, selectedWeights[0], 0.001f);
    }

    [Fact]
    public void VecDotQ5_K_MatchesDequantizedDotProduct()
    {
        const int k = 256;
        BlockQ5_K block = new BlockQ5_K();
        block.Delta = (Half)1.5f;
        block.DeltaMin = (Half)0.5f;

        for (int i = 0; i < 12; i++) block.Scales[i] = (byte)(i * 5 + 1);
        for (int i = 0; i < 32; i++) block.Qh[i] = (byte)(i % 255);
        for (int i = 0; i < 128; i++) block.Qs[i] = (byte)((i * 17) & 0xFF);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 7) - 3.0f;

        float[] dequant = new float[k];
        float expectedDot = 0f;

        BlockQ5_K* pBlock = &block;
        fixed (float* pDst = dequant, pX = x)
        {
            QuantKernels.DequantizeQ5_K(pBlock, pDst, k);
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotQ5_K(pBlock, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.01f);
        }
    }

    [Fact]
    public void VecDotQ3_K_MatchesDequantizedDotProduct()
    {
        const int k = 256;
        BlockQ3_K block = new BlockQ3_K();
        block.Delta = (Half)0.75f;

        for (int i = 0; i < 32; i++) block.Hmask[i] = (byte)(i * 13 + 7);
        for (int i = 0; i < 64; i++) block.Qs[i] = (byte)(i * 23);
        for (int i = 0; i < 12; i++) block.Scales[i] = (byte)(i * 11 + 3);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 9) - 4.0f;

        float[] dequant = new float[k];
        float expectedDot = 0f;

        BlockQ3_K* pBlock3 = &block;
        fixed (float* pDst = dequant, pX = x)
        {
            QuantKernels.DequantizeQ3_K(pBlock3, pDst, k);
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotQ3_K(pBlock3, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.01f);
        }
    }

    [Fact]
    public void VecDotMXFP4_MatchesDequantizedDotProduct()
    {
        const int k = 64; // 2 blocks
        BlockMXFP4[] blocks = new BlockMXFP4[2];
        blocks[0].Scale = 129; // 2^(129-127) = 2^2 = 4.0
        blocks[1].Scale = 125; // 2^(125-127) = 2^(-2) = 0.25

        for (int i = 0; i < 16; i++)
        {
            blocks[0].Qs[i] = (byte)(i | ((15 - i) << 4));
            blocks[1].Qs[i] = (byte)(((i * 3) & 0x0F) | (((i * 7) & 0x0F) << 4));
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 5) - 2.0f;

        float[] dequant = new float[k];
        float expectedDot = 0f;

        fixed (BlockMXFP4* pBlocks = blocks)
        fixed (float* pDst = dequant, pX = x)
        {
            QuantKernels.DequantizeMXFP4(pBlocks, pDst, k);
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotMXFP4(pBlocks, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.001f);
        }
    }
}

