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

    [Fact]
    public void RoPEMla_RotatesOnlyPeSliceAndPreservesNopeSlice()
    {
        const int nHeads = 2;
        const int qHeadDim = 192;
        const int qkNopeDim = 128;
        const int qkRopeDim = 64;

        float[] q = new float[nHeads * qHeadDim];
        float[] kRope = new float[qkRopeDim];

        // Fill with distinct values
        for (int i = 0; i < q.Length; i++) q[i] = (float)(i + 1) * 0.01f;
        for (int i = 0; i < kRope.Length; i++) kRope[i] = (float)(i + 10) * 0.01f;

        float[] qBefore = (float[])q.Clone();
        float[] kBefore = (float[])kRope.Clone();

        fixed (float* pQ = q, pK = kRope)
        {
            QuantKernels.RoPEMla(pQ, pK, nHeads, qHeadDim, qkNopeDim, qkRopeDim, pos: 5, freqBase: 10000.0f);
        }

        // 1. Verify q_nope (0..127) for both heads is completely untouched
        for (int h = 0; h < nHeads; h++)
        {
            for (int i = 0; i < qkNopeDim; i++)
            {
                Assert.Equal(qBefore[h * qHeadDim + i], q[h * qHeadDim + i]);
            }
        }

        // 2. Verify q_pe (128..191) has rotated (values changed) but L2 norm preserved
        for (int h = 0; h < nHeads; h++)
        {
            float normBefore = 0f, normAfter = 0f;
            for (int i = 0; i < qkRopeDim; i++)
            {
                float b = qBefore[h * qHeadDim + qkNopeDim + i];
                float a = q[h * qHeadDim + qkNopeDim + i];
                normBefore += b * b;
                normAfter += a * a;
            }
            Assert.NotEqual(qBefore[h * qHeadDim + qkNopeDim], q[h * qHeadDim + qkNopeDim]);
            Assert.True(Math.Abs(normBefore - normAfter) / normBefore < 1e-4f, $"Norm mismatch: before={normBefore}, after={normAfter}");
        }

        // 3. Verify kRope has rotated and L2 norm preserved
        float kNormBefore = 0f, kNormAfter = 0f;
        for (int i = 0; i < qkRopeDim; i++)
        {
            kNormBefore += kBefore[i] * kBefore[i];
            kNormAfter += kRope[i] * kRope[i];
        }
        Assert.NotEqual(kBefore[0], kRope[0]);
        Assert.True(Math.Abs(kNormBefore - kNormAfter) / kNormBefore < 1e-4f, $"KNorm mismatch: before={kNormBefore}, after={kNormAfter}");
    }

    [Fact]
    public void PrecomputeYarnFrequencies_InterpolatesSmoothly()
    {
        float[] freqs = new float[32];
        QuantKernels.PrecomputeYarnFrequencies(freqs, dim: 64, freqBase: 10000f, scalingFactor: 40f, originalCtx: 4096);

        // Frequencies should be strictly positive and decreasing with index
        for (int i = 0; i < 32; i++)
        {
            Assert.True(freqs[i] > 0f);
            if (i > 0)
            {
                Assert.True(freqs[i] <= freqs[i - 1]);
            }
        }
    }

    [Fact]
    public void Q5_0_DequantizeAndVecDot_MatchesExpected()
    {
        BlockQ5_0 block = new BlockQ5_0();
        block.Delta = (Half)2.0f;
        // bit 0 set (element 0) and bit 16 set (element 16)
        block.Qh = (1u << 0) | (1u << 16);
        // element 0: lower nibble 5 | 16 = 21 -> 21 - 16 = 5. Value = 5 * 2 = 10.
        // element 16: upper nibble 9 | 16 = 25 -> 25 - 16 = 9. Value = 9 * 2 = 18.
        block.Qs[0] = (byte)(5 | (9 << 4));

        float[] dequant = new float[32];
        float[] x = new float[32];
        x[0] = 1.0f;
        x[16] = 2.0f;

        BlockQ5_0* pBlock = &block;
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeQ5_0(pBlock, pDequant, 32);
            float dot = QuantKernels.VecDotQ5_0(pBlock, pX, 32);

            Assert.Equal(10.0f, dequant[0], 1e-4f);
            Assert.Equal(18.0f, dequant[16], 1e-4f);
            Assert.Equal(46.0f, dot, 1e-4f);
        }
    }

    [Fact]
    public void IQ4_XS_DequantizeAndVecDot_MatchesExpected()
    {
        BlockIQ4_XS block = new BlockIQ4_XS();
        block.Delta = (Half)1.0f;
        // For ib = 0: scalesL[0] low nibble = 1, scalesH low 2 bits = 2 -> (2 << 4) | 1 = 33.
        // ls = 33, so (ls - 32) = 1.0f.
        block.ScalesL[0] = 1;
        block.ScalesH = 2;
        // qs[0] = 8 | (9 << 4) -> low nibble index 8 (val = 1.0f), high nibble index 9 (val = 13.0f)
        block.Qs[0] = (byte)(8 | (9 << 4));

        float[] dequant = new float[256];
        float[] x = new float[256];
        x[0] = 2.0f;
        x[16] = 3.0f;

        BlockIQ4_XS* pBlock = &block;
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeIQ4_XS(pBlock, pDequant, 256);
            float dot = QuantKernels.VecDotIQ4_XS(pBlock, pX, 256);

            // dequant[0] = 1.0 * 1.0 = 1.0
            // dequant[16] = 1.0 * 13.0 = 13.0
            Assert.Equal(1.0f, dequant[0], 1e-4f);
            Assert.Equal(13.0f, dequant[16], 1e-4f);

            // dot = 2.0 * 1.0 + 3.0 * 13.0 = 41.0
            Assert.Equal(41.0f, dot, 1e-4f);
        }
    }

    [Fact]
    public void VecDotQ4_0_MatchesExpected_And_Dequantized()
    {
        const int k = 64; // 2 blocks of 32
        BlockQ4_0[] blocks = new BlockQ4_0[2];
        blocks[0].Delta = (Half)1.5f;
        blocks[1].Delta = (Half)2.0f;

        for (int l = 0; l < 16; l++)
        {
            blocks[0].Qs[l] = (byte)((l & 0x0F) | (((15 - l) & 0x0F) << 4));
            blocks[1].Qs[l] = (byte)(((l * 3) & 0x0F) | (((l * 7) & 0x0F) << 4));
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 5) - 2.0f;

        // Compute expected scalar dot
        float expectedDot = 0f;
        for (int b = 0; b < 2; b++)
        {
            float d = (float)blocks[b].Delta;
            float blockSum = 0f;
            for (int l = 0; l < 16; l++)
            {
                int q0 = (blocks[b].Qs[l] & 0x0F) - 8;
                int q1 = (blocks[b].Qs[l] >> 4) - 8;
                blockSum += q0 * x[b * 32 + l] + q1 * x[b * 32 + l + 16];
            }
            expectedDot += d * blockSum;
        }

        fixed (BlockQ4_0* pB = blocks)
        fixed (float* pX = x)
        {
            float actualDot = QuantKernels.VecDotQ4_0(pB, pX, k);
            Assert.Equal(expectedDot, actualDot, 1e-3f);
        }
    }

    [Fact]
    public void VecDotF16_MatchesReference()
    {
        const int k = 64;
        Half[] h = new Half[k];
        float[] x = new float[k];
        float expectedDot = 0f;

        for (int i = 0; i < k; i++)
        {
            float f = (i * 0.25f) - 4.0f;
            h[i] = (Half)f;
            x[i] = (i % 7) * 0.5f;
            expectedDot += (float)h[i] * x[i];
        }

        fixed (Half* pH = h)
        fixed (float* pX = x)
        {
            float actualDot = QuantKernels.VecDotF16(pH, pX, k);
            Assert.Equal(expectedDot, actualDot, 1e-3f);
        }
    }

    [Fact]
    public void IQ2_XXS_DequantizeAndVecDot_MatchesExpected()
    {
        const int k = 256;
        BlockIQ2_XXS block = new BlockIQ2_XXS();
        block.Delta = (Half)1.0f;

        // Populate Qs with deterministic values
        for (int i = 0; i < 32; i++)
        {
            block.Qs[i] = (ushort)(i * 1337 + 42);
        }

        float[] dequant = new float[k];
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 7) * 0.25f - 0.75f;

        BlockIQ2_XXS* pBlock = &block;
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeIQ2_XXS(pBlock, pDequant, k);
            float actualDot = QuantKernels.VecDotIQ2_XXS(pBlock, pX, k);

            float expectedDot = 0f;
            for (int i = 0; i < k; i++)
            {
                expectedDot += dequant[i] * x[i];
            }

            Assert.NotEqual(0f, expectedDot);
            Assert.Equal(expectedDot, actualDot, 1e-3f);
        }
    }

    [Fact]
    public void Q2_0_DequantizeAndVecDot_MatchesExpected()
    {
        const int k = 128; // 2 blocks of 64
        BlockQ2_0[] blocks = new BlockQ2_0[2];
        blocks[0].Delta = (Half)1.5f;
        blocks[0].Min = (Half)(-0.5f);
        for (int l = 0; l < 16; l++)
        {
            blocks[0].Qs[l] = (byte)((l * 17) & 0xFF);
        }

        blocks[1].Delta = (Half)2.0f;
        blocks[1].Min = (Half)0.25f;
        for (int l = 0; l < 16; l++)
        {
            blocks[1].Qs[l] = (byte)((l * 31 + 7) & 0xFF);
        }

        float[] dequant = new float[k];
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 7) - 3.0f;

        fixed (BlockQ2_0* pBlocks = blocks)
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeQ2_0(pBlocks, pDequant, k);

            // 1. Verify dequantization against expected formula
            for (int b = 0; b < 2; b++)
            {
                float d = (float)blocks[b].Delta;
                float m = (float)blocks[b].Min;
                for (int l = 0; l < 16; l++)
                {
                    byte qByte = blocks[b].Qs[l];
                    float q0 = (float)((qByte >> 0) & 3);
                    float q1 = (float)((qByte >> 2) & 3);
                    float q2 = (float)((qByte >> 4) & 3);
                    float q3 = (float)((qByte >> 6) & 3);

                    Assert.Equal(d * q0 + m, dequant[b * 64 + l + 0], 1e-4f);
                    Assert.Equal(d * q1 + m, dequant[b * 64 + l + 16], 1e-4f);
                    Assert.Equal(d * q2 + m, dequant[b * 64 + l + 32], 1e-4f);
                    Assert.Equal(d * q3 + m, dequant[b * 64 + l + 48], 1e-4f);
                }
            }

            // 2. Verify VecDot matches scalar sum of (dequant * x)
            float expectedDot = 0f;
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotQ2_0(pBlocks, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.01f);
        }

        // 3. Test Quantize -> Dequantize round-trip error bounds
        float[] original = new float[k];
        for (int i = 0; i < k; i++) original[i] = (float)Math.Sin(i * 0.1) * 6.0f;

        BlockQ2_0[] qBlocks = new BlockQ2_0[k / 64];
        float[] reconstructed = new float[k];
        fixed (float* pOrig = original, pRecon = reconstructed)
        fixed (BlockQ2_0* pQ = qBlocks)
        {
            QuantKernels.QuantizeQ2_0(pOrig, pQ, k);
            QuantKernels.DequantizeQ2_0(pQ, pRecon, k);
        }

        for (int b = 0; b < k / 64; b++)
        {
            float d = (float)qBlocks[b].Delta;
            float maxAllowedError = (d * 0.5f) + 0.05f;
            for (int j = 0; j < 64; j++)
            {
                int idx = b * 64 + j;
                float err = MathF.Abs(original[idx] - reconstructed[idx]);
                Assert.True(err <= maxAllowedError,
                    $"Q2_0 round-trip error too large at {idx}: err={err}, maxAllowed={maxAllowedError}");
            }
        }
    }

    [Fact]
    public void IQ4_NL_DequantizeAndVecDot_MatchesExpected()
    {
        const int k = 64; // 2 blocks of 32
        BlockIQ4_NL[] blocks = new BlockIQ4_NL[2];
        blocks[0].Delta = (Half)0.5f;
        for (int l = 0; l < 16; l++)
        {
            blocks[0].Qs[l] = (byte)(l | ((15 - l) << 4));
        }

        blocks[1].Delta = (Half)1.25f;
        for (int l = 0; l < 16; l++)
        {
            blocks[1].Qs[l] = (byte)(((l * 3) & 0x0F) | (((l * 7 + 1) & 0x0F) << 4));
        }

        float[] dequant = new float[k];
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 5) - 2.0f;

        float[] kValues = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

        fixed (BlockIQ4_NL* pBlocks = blocks)
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeIQ4_NL(pBlocks, pDequant, k);

            // 1. Verify dequantization against expected non-linear codebook lookup
            for (int b = 0; b < 2; b++)
            {
                float d = (float)blocks[b].Delta;
                for (int l = 0; l < 16; l++)
                {
                    byte q = blocks[b].Qs[l];
                    float expLow = d * kValues[q & 0x0F];
                    float expHigh = d * kValues[q >> 4];

                    Assert.Equal(expLow, dequant[b * 32 + l], 1e-4f);
                    Assert.Equal(expHigh, dequant[b * 32 + l + 16], 1e-4f);
                }
            }

            // 2. Verify VecDot matches scalar sum of (dequant * x)
            float expectedDot = 0f;
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotIQ4_NL(pBlocks, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.01f);
        }

        // 3. Test Quantize -> Dequantize round-trip error bounds
        float[] original = new float[k];
        for (int i = 0; i < k; i++) original[i] = (float)Math.Cos(i * 0.2) * 10.0f;

        BlockIQ4_NL[] qBlocks = new BlockIQ4_NL[k / 32];
        float[] reconstructed = new float[k];
        fixed (float* pOrig = original, pRecon = reconstructed)
        fixed (BlockIQ4_NL* pQ = qBlocks)
        {
            QuantKernels.QuantizeIQ4_NL(pOrig, pQ, k);
            QuantKernels.DequantizeIQ4_NL(pQ, pRecon, k);
        }

        for (int i = 0; i < k; i++)
        {
            int b = i / 32;
            float d = (float)qBlocks[b].Delta;
            float maxError = 15.0f * d + 0.1f;
            float err = MathF.Abs(original[i] - reconstructed[i]);
            Assert.True(err <= maxError,
                $"IQ4_NL roundtrip error at {i}: orig={original[i]}, recon={reconstructed[i]}, err={err}, maxErr={maxError}");
        }
    }

    [Fact]
    public void IQ3_S_DequantizeAndVecDot_MatchesExpected()
    {
        const int k = 256; // 1 super-block
        BlockIQ3_S block = new BlockIQ3_S();
        block.Delta = (Half)0.25f;

        for (int i = 0; i < 4; i++) block.Scales[i] = (byte)(0x21 * (i + 1));
        for (int i = 0; i < 64; i++) block.Qs[i] = (byte)(i * 7 + 3);
        for (int i = 0; i < 8; i++) block.Qh[i] = (byte)(i * 13);
        for (int i = 0; i < 32; i++) block.Signs[i] = (byte)(i * 19 + 5);

        float[] dequant = new float[k];
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (i % 7) * 0.5f - 1.5f;

        BlockIQ3_S* pBlock = &block;
        fixed (float* pDequant = dequant, pX = x)
        {
            QuantKernels.DequantizeIQ3_S(pBlock, pDequant, k);

            // 1. Verify against 4D sphere lattice reconstruction
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = (ib & 1) == 0 ? (block.Scales[ib / 2] & 0x0F) : (block.Scales[ib / 2] >> 4);
                float db = (float)block.Delta * (1 + 2 * ls);
                byte qhByte = block.Qh[ib];

                for (int g = 0; g < 8; g++)
                {
                    int gridIdx = block.Qs[ib * 8 + g] | (((qhByte >> g) & 1) << 8);
                    uint gridWord = QuantKernels.Iq3sGrid[gridIdx];
                    int signNibble = (block.Signs[ib * 4 + (g >> 1)] >> ((g & 1) * 4)) & 0x0F;

                    float g0 = db * (gridWord & 0xFF) * ((signNibble & 1) != 0 ? -1f : 1f);
                    float g1 = db * ((gridWord >> 8) & 0xFF) * ((signNibble & 2) != 0 ? -1f : 1f);
                    float g2 = db * ((gridWord >> 16) & 0xFF) * ((signNibble & 4) != 0 ? -1f : 1f);
                    float g3 = db * (gridWord >> 24) * ((signNibble & 8) != 0 ? -1f : 1f);

                    int baseIdx = ib * 32 + g * 4;
                    Assert.Equal(g0, dequant[baseIdx + 0], 1e-4f);
                    Assert.Equal(g1, dequant[baseIdx + 1], 1e-4f);
                    Assert.Equal(g2, dequant[baseIdx + 2], 1e-4f);
                    Assert.Equal(g3, dequant[baseIdx + 3], 1e-4f);
                }
            }

            // 2. Verify VecDot matches scalar sum of (dequant * x)
            float expectedDot = 0f;
            for (int i = 0; i < k; i++) expectedDot += dequant[i] * x[i];

            float actualDot = QuantKernels.VecDotIQ3_S(pBlock, pX, k);
            Assert.Equal(expectedDot, actualDot, 0.01f);
        }

        // 3. Test Quantize -> Dequantize round-trip
        float[] original = new float[k];
        for (int i = 0; i < k; i++) original[i] = (float)Math.Sin(i * 0.15) * 8.0f;

        BlockIQ3_S[] qBlocks = new BlockIQ3_S[1];
        float[] reconstructed = new float[k];
        fixed (float* pOrig = original, pRecon = reconstructed)
        fixed (BlockIQ3_S* pQ = qBlocks)
        {
            QuantKernels.QuantizeIQ3_S(pOrig, pQ, k);
            QuantKernels.DequantizeIQ3_S(pQ, pRecon, k);
        }

        float dotOrigRecon = 0f, dotOrigOrig = 0f;
        for (int i = 0; i < k; i++)
        {
            Assert.False(float.IsNaN(reconstructed[i]));
            dotOrigRecon += original[i] * reconstructed[i];
            dotOrigOrig += original[i] * original[i];
        }

        float cosSim = dotOrigRecon / dotOrigOrig;
        Assert.True(cosSim > 0.90f, $"IQ3_S cosine similarity: {cosSim}");
    }

    [Theory]
    [InlineData(Glacier.Inference.Gguf.GgufType.Q2_0)]
    [InlineData(Glacier.Inference.Gguf.GgufType.IQ4_NL)]
    [InlineData(Glacier.Inference.Gguf.GgufType.IQ3_S)]
    public void MatMulBatch_Supports_FastMoE_QuantTypes(Glacier.Inference.Gguf.GgufType quantType)
    {
        const int nCols = 256;
        const int nRows = 8;
        const int batchSize = 4;

        int rowBytes = (int)Glacier.Inference.Gguf.GgufTypes.GetRowBytes(quantType, nCols);
        byte[] weights = new byte[rowBytes * nRows];
        new Random(42 + (int)quantType).NextBytes(weights);

        float[] xBatch = new float[batchSize * nCols];
        for (int i = 0; i < xBatch.Length; i++) xBatch[i] = (i % 17) - 8.0f;

        float[] yBatch = new float[batchSize * nRows];
        float[] ySingle = new float[batchSize * nRows];

        fixed (byte* pW = weights)
        fixed (float* pX = xBatch, pYBatch = yBatch, pYSingle = ySingle)
        {
            // Compute via MatMulBatch
            QuantKernels.MatMulBatch(quantType, pW, pX, pYBatch, nCols, nRows, batchSize);

            // Compute individually via MatVecMul
            for (int b = 0; b < batchSize; b++)
            {
                QuantKernels.MatVecMul(quantType, pW, pX + b * nCols, pYSingle + b * nRows, nCols, nRows);
            }
        }

        for (int i = 0; i < yBatch.Length; i++)
        {
            Assert.Equal(ySingle[i], yBatch[i], 0.0001f);
        }
    }

    [Fact]
    public void ComputeYarnMscale_MatchesDeepSeekSpecification()
    {
        // For scaling factor <= 1.0, mscale must always be 1.0 (identity)
        Assert.Equal(1.0f, QuantKernels.ComputeYarnMscale(1.0f, 1.0f, 0.0f, 1.0f));
        Assert.Equal(1.0f, QuantKernels.ComputeYarnMscale(0.5f, 1.0f, 0.0f, 1.0f));

        // For scaling factor = 31.25 (1M context / 32k context) with mscale = 1.0
        // Expected: 0.1 * 1.0 * ln(31.25) + 1.0 = 0.1 * 3.4420 + 1.0 = 1.3442
        float scale = 31.25f;
        float expected = 0.1f * MathF.Log(scale) + 1.0f;
        float actual = QuantKernels.ComputeYarnMscale(scale, mscale: 1.0f, mscaleAllDim: 0.0f, attnFactor: 1.0f);
        Assert.Equal(expected, actual, 0.0001f);
    }

    [Fact]
    public void PrecomputeYarnFrequencies_InterpolatesHighDimensionsAndExtrapolatesLowDimensions()
    {
        const int dim = 128;
        const int halfDim = dim / 2;
        float[] invFreq = new float[halfDim];

        // 1M context config
        QuantKernels.PrecomputeYarnFrequencies(
            invFreq.AsSpan(),
            dim: dim,
            freqBase: 1000000.0f,
            scalingFactor: 31.25f,
            originalCtx: 32768,
            betaFast: 32.0f,
            betaSlow: 1.0f);

        // Low dimensions (high frequency, i=0) should be purely extrapolated: 1 / base^0 = 1.0
        Assert.Equal(1.0f, invFreq[0], 0.0001f);

        // Higher dimensions should be strictly decreasing positive frequencies
        for (int i = 1; i < halfDim; i++)
        {
            Assert.True(invFreq[i] > 0f, $"invFreq[{i}] was {invFreq[i]} <= 0");
            Assert.True(invFreq[i] < invFreq[i - 1], $"invFreq[{i}] was not strictly decreasing: {invFreq[i]} >= {invFreq[i - 1]}");
        }
    }
}


