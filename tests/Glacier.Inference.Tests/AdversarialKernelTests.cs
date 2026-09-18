namespace Glacier.Inference.Tests;

using System;
using System.Reflection;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Xunit;

public unsafe class AdversarialKernelTests
{
    private static void OracleGetScaleMinK4(int j, byte* q, out int sc, out int m)
    {
        if (j < 4)
        {
            sc = q[j] & 63;
            m = q[j + 4] & 63;
        }
        else
        {
            sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }

    private static double OracleDotQ4_K(BlockQ4_K* row, float* x, int k)
    {
        int nb = k / 256;
        double sum = 0.0;
        int xIdx = 0;

        for (int i = 0; i < nb; i++)
        {
            double d = (double)(float)row[i].Delta;
            double min = (double)(float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* q = row[i].Qs;

            int is_idx = 0;
            for (int j = 0; j < 4; j++)
            {
                OracleGetScaleMinK4(is_idx + 0, scales, out int sc0, out int m0);
                OracleGetScaleMinK4(is_idx + 1, scales, out int sc1, out int m1);

                double d1 = d * sc0;
                double min1 = min * m0;
                double d2 = d * sc1;
                double min2 = min * m1;

                for (int l = 0; l < 32; l++)
                {
                    double val = d1 * (q[32 * j + l] & 0x0F) - min1;
                    sum += val * (double)x[xIdx++];
                }
                for (int l = 0; l < 32; l++)
                {
                    double val = d2 * (q[32 * j + l] >> 4) - min2;
                    sum += val * (double)x[xIdx++];
                }

                is_idx += 2;
            }
        }
        return sum;
    }

    private static double OracleDotQ5_K(BlockQ5_K* row, float* x, int k)
    {
        int nb = k / 256;
        double sum = 0.0;
        int xIdx = 0;

        for (int i = 0; i < nb; i++)
        {
            double d = (double)(float)row[i].Delta;
            double min = (double)(float)row[i].DeltaMin;
            byte* scales = row[i].Scales;
            byte* ql = row[i].Qs;
            byte* qh = row[i].Qh;

            int is_idx = 0;
            for (int j = 0; j < 4; j++)
            {
                OracleGetScaleMinK4(is_idx + 0, scales, out int sc0, out int m0);
                OracleGetScaleMinK4(is_idx + 1, scales, out int sc1, out int m1);

                double d1 = d * sc0;
                double min1 = min * m0;
                double d2 = d * sc1;
                double min2 = min * m1;

                byte u1 = (byte)(1 << (2 * j));
                byte u2 = (byte)(2 << (2 * j));

                for (int l = 0; l < 32; l++)
                {
                    int qVal = (ql[32 * j + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                    double val = d1 * qVal - min1;
                    sum += val * (double)x[xIdx++];
                }
                for (int l = 0; l < 32; l++)
                {
                    int qVal = (ql[32 * j + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                    double val = d2 * qVal - min2;
                    sum += val * (double)x[xIdx++];
                }

                is_idx += 2;
            }
        }
        return sum;
    }

    private static double OracleDotQ6_K(BlockQ6_K* row, float* x, int k)
    {
        int nb = k / 256;
        double sum = 0.0;
        int xIdx = 0;

        for (int i = 0; i < nb; i++)
        {
            double d = (double)(float)row[i].Delta;
            byte* ql = row[i].Ql;
            byte* qh = row[i].Qh;
            sbyte* sc = row[i].Scales;

            for (int n = 0; n < 256; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int is_idx = l / 16;
                    int q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum += (d * sc[is_idx + 0] * q1) * (double)x[xIdx + l + 0];
                    sum += (d * sc[is_idx + 2] * q2) * (double)x[xIdx + l + 32];
                    sum += (d * sc[is_idx + 4] * q3) * (double)x[xIdx + l + 64];
                    sum += (d * sc[is_idx + 6] * q4) * (double)x[xIdx + l + 96];
                }

                xIdx += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return sum;
    }

    private static double OracleDotQ8_0(BlockQ8_0* row, float* x, int k)
    {
        int nb = k / 32;
        double sum = 0.0;
        for (int i = 0; i < nb; i++)
        {
            double d = (double)(float)row[i].Delta;
            sbyte* q = row[i].Qs;
            for (int l = 0; l < 32; l++)
            {
                sum += (d * q[l]) * (double)x[i * 32 + l];
            }
        }
        return sum;
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void VecDotQ4_K_MatchesIndependentDoublePrecisionOracle_AcrossBoundaries(int k)
    {
        int nb = k / 256;
        var blocks = new BlockQ4_K[nb];
        var rng = new Random(12345 + k);

        for (int i = 0; i < nb; i++)
        {
            blocks[i].Delta = (Half)(float)(rng.NextDouble() * 2.0 + 0.1);
            blocks[i].DeltaMin = (Half)(float)(rng.NextDouble() * 1.0 + 0.05);
            for (int s = 0; s < 12; s++) blocks[i].Scales[s] = (byte)rng.Next(0, 256);
            for (int q = 0; q < 128; q++) blocks[i].Qs[q] = (byte)rng.Next(0, 256);
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

        float[] xSums = new float[k / 32];

        fixed (BlockQ4_K* pBlocks = blocks)
        fixed (float* pX = x, pSums = xSums)
        {
            QuantKernels.ComputeBlockSums32(pX, pSums, k);

            double oracleDot = OracleDotQ4_K(pBlocks, pX, k);
            float kernelDotNullSums = QuantKernels.VecDotQ4_K(pBlocks, pX, null, k);
            float kernelDotWithSums = QuantKernels.VecDotQ4_K(pBlocks, pX, pSums, k);

            float diffNull = MathF.Abs((float)oracleDot - kernelDotNullSums);
            float diffSums = MathF.Abs((float)oracleDot - kernelDotWithSums);

            float relErrorNull = diffNull / MathF.Max(1.0f, MathF.Abs((float)oracleDot));
            float relErrorSums = diffSums / MathF.Max(1.0f, MathF.Abs((float)oracleDot));

            Assert.True(relErrorNull < 1e-3f, $"Q4_K (null sums, k={k}): relError={relErrorNull}, oracle={oracleDot}, actual={kernelDotNullSums}");
            Assert.True(relErrorSums < 1e-3f, $"Q4_K (with sums, k={k}): relError={relErrorSums}, oracle={oracleDot}, actual={kernelDotWithSums}");
            Assert.Equal(kernelDotNullSums, kernelDotWithSums, 0.01f);
        }
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void VecDotQ5_K_MatchesIndependentDoublePrecisionOracle_AcrossBoundaries(int k)
    {
        int nb = k / 256;
        var blocks = new BlockQ5_K[nb];
        var rng = new Random(54321 + k);

        for (int i = 0; i < nb; i++)
        {
            blocks[i].Delta = (Half)(float)(rng.NextDouble() * 2.5 + 0.2);
            blocks[i].DeltaMin = (Half)(float)(rng.NextDouble() * 1.2 + 0.1);
            for (int s = 0; s < 12; s++) blocks[i].Scales[s] = (byte)rng.Next(0, 256);
            for (int h = 0; h < 32; h++) blocks[i].Qh[h] = (byte)rng.Next(0, 256);
            for (int q = 0; q < 128; q++) blocks[i].Qs[q] = (byte)rng.Next(0, 256);
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 6.0 - 3.0);

        fixed (BlockQ5_K* pBlocks = blocks)
        fixed (float* pX = x)
        {
            double oracleDot = OracleDotQ5_K(pBlocks, pX, k);
            float kernelDot = QuantKernels.VecDotQ5_K(pBlocks, pX, k);

            float diff = MathF.Abs((float)oracleDot - kernelDot);
            float relError = diff / MathF.Max(1.0f, MathF.Abs((float)oracleDot));

            Assert.True(relError < 1e-3f, $"Q5_K (k={k}): relError={relError}, oracle={oracleDot}, actual={kernelDot}");
        }
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void VecDotQ6_K_MatchesIndependentDoublePrecisionOracle_AcrossBoundaries(int k)
    {
        int nb = k / 256;
        var blocks = new BlockQ6_K[nb];
        var rng = new Random(67890 + k);

        for (int i = 0; i < nb; i++)
        {
            blocks[i].Delta = (Half)(float)(rng.NextDouble() * 1.8 + 0.15);
            for (int q = 0; q < 128; q++) blocks[i].Ql[q] = (byte)rng.Next(0, 256);
            for (int h = 0; h < 64; h++) blocks[i].Qh[h] = (byte)rng.Next(0, 256);
            for (int s = 0; s < 16; s++) blocks[i].Scales[s] = (sbyte)rng.Next(-32, 32);
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        fixed (BlockQ6_K* pBlocks = blocks)
        fixed (float* pX = x)
        {
            double oracleDot = OracleDotQ6_K(pBlocks, pX, k);
            float kernelDot = QuantKernels.VecDotQ6_K(pBlocks, pX, k);

            float diff = MathF.Abs((float)oracleDot - kernelDot);
            float relError = diff / MathF.Max(1.0f, MathF.Abs((float)oracleDot));

            Assert.True(relError < 1e-3f, $"Q6_K (k={k}): relError={relError}, oracle={oracleDot}, actual={kernelDot}");
        }
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    [InlineData(4096)]
    public void VecDotQ8_0_MatchesIndependentDoublePrecisionOracle_AcrossBoundaries(int k)
    {
        int nb = k / 32;
        var blocks = new BlockQ8_0[nb];
        var rng = new Random(88888 + k);

        for (int i = 0; i < nb; i++)
        {
            blocks[i].Delta = (Half)(float)(rng.NextDouble() * 3.0 + 0.05);
            for (int q = 0; q < 32; q++) blocks[i].Qs[q] = (sbyte)rng.Next(-128, 127);
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 5.0 - 2.5);

        fixed (BlockQ8_0* pBlocks = blocks)
        fixed (float* pX = x)
        {
            double oracleDot = OracleDotQ8_0(pBlocks, pX, k);
            float kernelDot = QuantKernels.VecDotQ8_0(pBlocks, pX, k);

            float diff = MathF.Abs((float)oracleDot - kernelDot);
            float relError = diff / MathF.Max(1.0f, MathF.Abs((float)oracleDot));

            Assert.True(relError < 1e-3f, $"Q8_0 (k={k}): relError={relError}, oracle={oracleDot}, actual={kernelDot}");
        }
    }

    [Fact]
    public void Qwen2Model_SequentialDecodeThreshold_PreventsThreadPoolSchedulingSpikes()
    {
        var thresholdField = typeof(Qwen2Model).GetField("ParallelAttentionSeqThreshold", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(thresholdField);
        int threshold = (int)thresholdField.GetValue(null)!;

        Assert.Equal(256, threshold);

        var method = typeof(Qwen2Model).GetMethod("ComputeAttentionToken", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);

        int testPosBefore = 100;
        int testPosAfter = 300;

        Assert.True(testPosBefore < threshold, "pos < 256 must use sequential decode");
        Assert.True(testPosAfter >= threshold, "pos >= 256 engages chunked Parallel.For");
    }
}
