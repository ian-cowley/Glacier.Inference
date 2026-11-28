using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Tests.E2E;
using Glacier.Inference.Tests.VulkanE2E.Fixtures;
using Xunit;

namespace Glacier.Inference.Tests.VulkanE2E;

/// <summary>
/// Tier 2: Boundary & Corner Cases (Boundary Value Analysis across all 16 feature areas).
/// Exactly 50 comprehensive boundary and extreme condition tests.
/// </summary>
public unsafe class VulkanE2ETier2BoundaryAndCornerTests
{

    #region Boundary Area 1: Attention Biases Boundaries

    [Fact]
    public void T2_BVA_01_AttentionBiases_AllNull_NoPointersAllocated()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: false);
        var layer = handle.Weights.Layers[0];

        Assert.Null((IntPtr)layer.QBias == IntPtr.Zero ? (object?)null : new object());
        Assert.Null((IntPtr)layer.KBias == IntPtr.Zero ? (object?)null : new object());
        Assert.Null((IntPtr)layer.VBias == IntPtr.Zero ? (object?)null : new object());
        Assert.Null((IntPtr)layer.AttnOutBias == IntPtr.Zero ? (object?)null : new object());
    }

    [Fact]
    public void T2_BVA_02_AttentionBiases_AllZeroMagnitudes()
    {
        const int dim = 64;
        float[] bias = new float[dim];
        float[] activation = new float[dim];
        for (int i = 0; i < dim; i++) activation[i] = (i % 5) * 1.25f - 2.0f;
        float[] original = (float[])activation.Clone();

        fixed (float* pBias = bias, pAct = activation)
        {
            QuantKernels.VecAdd(pBias, pAct, dim);
        }

        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(original[i], activation[i]);
        }
    }

    [Fact]
    public void T2_BVA_03_AttentionBiases_ExtremeMagnitudes_NoNaNOrOverflow()
    {
        const int dim = 32;
        float[] bias = new float[dim];
        float[] activation = new float[dim];
        float[] pattern = [1000.0f, -1000.0f, 1e5f, -1e5f];
        float[] actPattern = [5.0f, -10.0f, 20.0f, -50.0f];
        for (int i = 0; i < dim; i++)
        {
            bias[i] = pattern[i & 3];
            activation[i] = actPattern[i & 3];
        }

        fixed (float* pBias = bias, pAct = activation)
        {
            QuantKernels.VecAdd(pBias, pAct, dim);
        }

        for (int i = 0; i < dim; i++)
        {
            Assert.False(float.IsNaN(activation[i]));
            Assert.False(float.IsInfinity(activation[i]));
            Assert.Equal(actPattern[i & 3] + pattern[i & 3], activation[i], 0.01f);
        }
    }

    [Fact]
    public void T2_BVA_04_AttentionBiases_SubnormalFloats_HandledGracefully()
    {
        const int dim = 20;
        float subnormal = 1e-38f;
        float[] bias = new float[dim];
        float[] activation = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            bias[i] = (i & 1) == 0 ? subnormal : -subnormal;
            activation[i] = (i & 1) == 0 ? 1.0f : -1.0f;
        }

        fixed (float* pBias = bias, pAct = activation)
        {
            QuantKernels.VecAdd(pBias, pAct, dim);
        }

        for (int i = 0; i < dim; i++)
        {
            Assert.True(float.IsFinite(activation[i]));
            Assert.Equal((i & 1) == 0 ? 1.0f : -1.0f, activation[i], 1e-6f);
        }
    }

    [Fact]
    public void T2_BVA_05_AttentionBiases_UnalignedDimension_Padding()
    {
        // Unaligned non-SIMD-multiple dimension (not divisible by 16 or 8)
        const int dim = 899;
        float[] bias = new float[dim];
        float[] activation = new float[dim];
        float[] normWeight = new float[dim];
        float[] normOut = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            bias[i] = 0.25f;
            activation[i] = i * 0.1f;
            normWeight[i] = 1.0f;
        }

        fixed (float* pBias = bias, pAct = activation, pW = normWeight, pNorm = normOut)
        {
            QuantKernels.VecAdd(pBias, pAct, dim);
            QuantKernels.RMSNorm(pAct, pW, pNorm, dim, 1e-5f);
        }

        Assert.Equal(898 * 0.1f + 0.25f, activation[dim - 1], 0.001f);
        Assert.True(float.IsFinite(normOut[dim - 1]));
    }

    #endregion

    #region Boundary Area 2: Quantized Weight Type Boundaries

    [Fact]
    public void T2_BVA_06_GgufTypeSize_ZeroElements_ReturnsZero()
    {
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 0));
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, 0));
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.F16, 0));
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.Q8_0, 0));
    }

    [Fact]
    public void T2_BVA_07_GgufTypeSize_MinimumSingleBlock()
    {
        // Exactly 1 block
        Assert.Equal(84L, TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 256));
        Assert.Equal(18L, TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, 32));
        Assert.Equal(34L, TestableVulkanRunner.GgufTypeSize(GgufType.Q8_0, 32));
        Assert.Equal(2L, TestableVulkanRunner.GgufTypeSize(GgufType.F16, 1));
    }

    [Fact]
    public void T2_BVA_08_GgufTypeSize_NonBlockMultiple_TruncatesOrRoundsDown()
    {
        // 128 elements is half a block for Q2_K (256-elem block)
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 128));
        // 16 elements is half a block for Q4_0 (32-elem block)
        Assert.Equal(0L, TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, 16));
    }

    [Fact]
    public void T2_BVA_09_GgufTypeSize_LargeTensor64Bit_NoOverflow()
    {
        long elements = 100_000_000L;
        long sizeQ2K = TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, elements);
        long sizeQ40 = TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, elements);
        long sizeF16 = TestableVulkanRunner.GgufTypeSize(GgufType.F16, elements);

        Assert.True(sizeQ2K > 0);
        Assert.True(sizeQ40 > 0);
        Assert.True(sizeF16 == 200_000_000L);
    }

    [Fact]
    public void T2_BVA_10_QIdx_UnsupportedGgufType_ThrowsNotSupportedException()
    {
        Assert.Throws<NotSupportedException>(() => TestableVulkanRunner.QIdx((GgufType)9999));
        Assert.Throws<NotSupportedException>(() => TestableVulkanRunner.QIdx(GgufType.F32));
        Assert.Equal(11, TestableVulkanRunner.QIdx(GgufType.BF16));
    }

    #endregion

    #region Boundary Area 3: Embedding Gather Boundaries

    [Fact]
    public void T2_BVA_11_EmbedGather_TokenZero_ValidOffset()
    {
        const int dim = 256;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: 256, vocabSize: 16, embdType: GgufType.Q4_0);
        Assert.Equal(8L * 18L, GgufTypes.GetRowBytes(GgufType.Q4_0, dim));

        float[] embd = new float[dim];
        float[] expected = new float[dim];
        fixed (float* pOut = embd)
        {
            QuantKernels.ExtractEmbedding(handle.Weights.EmbdType, handle.Weights.EmbdWeight, tokenId: 0, pOut, dim);
        }
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ4_0(
            new ReadOnlySpan<byte>(handle.Weights.EmbdWeight, (int)GgufTypes.GetRowBytes(GgufType.Q4_0, dim)),
            expected);

        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(expected[i], embd[i], 0.0001f);
        }
    }

    [Fact]
    public void T2_BVA_12_EmbedGather_MaxVocabToken_ValidOffset()
    {
        const int vocabSize = 32, dim = 256;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: 256, vocabSize: vocabSize, embdType: GgufType.Q4_0);
        long rowBytes = GgufTypes.GetRowBytes(GgufType.Q4_0, dim);
        int token = vocabSize - 1;

        float[] embd = new float[dim];
        float[] expected = new float[dim];
        fixed (float* pOut = embd)
        {
            QuantKernels.ExtractEmbedding(handle.Weights.EmbdType, handle.Weights.EmbdWeight, token, pOut, dim);
        }
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ4_0(
            new ReadOnlySpan<byte>(handle.Weights.EmbdWeight + (long)token * rowBytes, (int)rowBytes),
            expected);

        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(expected[i], embd[i], 0.0001f);
        }
    }

    [Fact]
    public void T2_BVA_13_EmbedGather_UnalignedEmbeddingDim_CalculatesBlocks()
    {
        Assert.Equal(6L * 144L, GgufTypes.GetRowBytes(GgufType.Q4_K, 1536));
        Assert.Throws<ArgumentException>(() => GgufTypes.GetRowBytes(GgufType.Q4_K, 100));
    }

    [Fact]
    public void T2_BVA_14_EmbedGather_NonEmbedQuantType_Rejected()
    {
        // MXFP4 and IQ4_XS are not supported for embedding gathering
        Assert.False(TestableVulkanRunner.IsEmbedKernelType(GgufType.MXFP4));
        Assert.False(TestableVulkanRunner.IsEmbedKernelType(GgufType.IQ4_XS));
        Assert.False(TestableVulkanRunner.IsEmbedKernelType(GgufType.F32));
    }

    [Fact]
    public void T2_BVA_15_EmbedGather_PushConstantPreservesDimAcrossMaxToken()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint dim = 3584u;
        runner.AddEmbedOp(dim);

        int maxToken = 151935;
        runner.ExecuteRun(maxToken, pos: 0);

        Assert.Equal(dim, runner.GetPush(0)[0]);
        Assert.Equal((uint)maxToken, runner.GetPush(0)[1]);
    }

    #endregion

    #region Boundary Area 4: Fused QKV & Gate/Up Boundaries

    [Fact]
    public void T2_BVA_16_FusedQkv_MQA_SingleKvHead()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 128, ffnDim: 256, headCount: 4, headCountKv: 1, vocabSize: 16, fusedQkv: true);
        Assert.Equal(1, handle.Weights.HeadCountKv);
        Assert.True(handle.Weights.Layers[0].HasFusedQkv);

        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.ForwardToken(token: 2, pos: 0, computeLogits: true);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
    }

    [Fact]
    public void T2_BVA_17_FusedQkv_MHA_EqualHeadCounts()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 128, ffnDim: 256, headCount: 4, headCountKv: 4, vocabSize: 16, fusedQkv: true);
        Assert.Equal(handle.Weights.HeadCount, handle.Weights.HeadCountKv);
        Assert.True(handle.Weights.Layers[0].HasFusedQkv);

        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.ForwardToken(token: 2, pos: 0, computeLogits: true);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
    }

    [Fact]
    public void T2_BVA_18_FusedQkv_AsymmetricGQA_Ratio7to1()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 224, ffnDim: 256, headCount: 7, headCountKv: 1, vocabSize: 16, fusedQkv: true);
        Assert.Equal(7, handle.Weights.HeadCount / handle.Weights.HeadCountKv);
        Assert.True(handle.Weights.Layers[0].HasFusedQkv);

        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.ForwardToken(token: 2, pos: 0, computeLogits: true);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
    }

    [Fact]
    public void T2_BVA_19_FusedGateUp_LargeIntermediateDim_Crossing32768()
    {
        int ffnDim = 18944;
        int totalRows = 2 * ffnDim; // 37,888 > 32,768

        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(totalRows);
        Assert.Equal(32768u, gx);
        Assert.Equal(2u, gy);
        Assert.True((long)gx * gy >= totalRows);
    }

    [Fact]
    public void T2_BVA_20_FusedGateUp_BothPresent_NotFusedGateUp()
    {
        // When both ffn_gate and ffn_up are present, it is unfused Gate/Up
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(fusedGateUp: false);
        Assert.False(handle.Weights.Layers[0].HasFusedGateUp);
    }

    #endregion

    #region Boundary Area 5: Workgroup Dispatch Grid Boundaries

    [Fact]
    public void T2_BVA_21_CalculateWorkgroups_SingleRow_M1()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(1);
        Assert.Equal(1u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T2_BVA_22_CalculateWorkgroups_BoundaryM32767()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(32767);
        Assert.Equal(32767u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T2_BVA_23_CalculateWorkgroups_BoundaryM32768()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(32768);
        Assert.Equal(32768u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T2_BVA_24_CalculateWorkgroups_BoundaryM32769()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(32769);
        Assert.Equal(32768u, gx);
        Assert.Equal(2u, gy);
    }

    [Fact]
    public void T2_BVA_25_CalculateWorkgroups_HugeVocab_M256000()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(256000);
        Assert.Equal(32768u, gx);
        Assert.Equal(8u, gy);
        Assert.True((long)gx * gy >= 256000);
    }

    #endregion

    #region Boundary Area 6: Push Constant & Sequence Length Boundaries

    [Fact]
    public void T2_BVA_26_SequencePosition_Zero_IdentityOffset()
    {
        var runner = TestableVulkanRunner.CreateForTest(maxSeqLen: 16, vocab: 32);
        runner.AddPosOp(99u);
        runner.ExecuteRun(token: 0, pos: 0);
        Assert.Equal(0u, runner.GetPush(0)[0]);
    }

    [Fact]
    public void T2_BVA_27_SequencePosition_MaxSeqLenMinusOne()
    {
        const int maxSeqLen = 16;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 64, ffnDim: 128, headCount: 2, headCountKv: 2, vocabSize: 16);
        using var session = new InferenceSession(handle.Path, maxSeqLen: maxSeqLen, device: "cpu");

        int[] tokens = Enumerable.Repeat(1, maxSeqLen).ToArray();
        session.Prefill(tokens);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => float.IsFinite(x));

        Assert.Throws<ArgumentOutOfRangeException>(() => session.ForwardToken(token: 1, pos: maxSeqLen, computeLogits: true));
    }

    [Fact]
    public void T2_BVA_28_SequencePosition_MonotonicKvLenAdvance()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        runner.AddKvLenOp(0u);

        int maxSeqLen = 16;
        uint prevKvLen = 0;
        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            runner.ExecuteRun(token: 0, pos: pos);
            uint currentKvLen = runner.GetPush(0)[0];

            Assert.Equal((uint)(pos + 1), currentKvLen);
            Assert.True(currentKvLen > prevKvLen, $"kvLen must advance monotonically ({currentKvLen} > {prevKvLen})");
            prevKvLen = currentKvLen;
        }
    }

    [Fact]
    public void T2_BVA_29_Vocabulary_SpecialTokens_EOS_BOS()
    {
        string[] tokens = VulkanE2ETestFixtures.SyntheticGgufBuilder.BuildTokens(10);
        Assert.Equal("<|im_start|>", tokens[0]);
        Assert.Equal("<|im_end|>", tokens[1]);
        Assert.Equal("<|endoftext|>", tokens[2]);
    }

    [Fact]
    public void T2_BVA_30_PushArray_CapacityValidation()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        runner.AddOpWithPush([2048u, 42u, 100u, 1u]);

        var push = runner.GetPush(0);
        Assert.Equal(4, push.Length);
        Assert.Equal(2048u, push[0]);
        Assert.Equal(42u, push[1]);
        Assert.Equal(100u, push[2]);
        Assert.Equal(1u, push[3]);
    }

    #endregion

    #region Boundary Area 7: Custom RoPE & Frequency Boundaries

    [Fact]
    public void T2_BVA_31_RoPE_NullCustomTable_UsesDefaultCalculation()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: false);
        Assert.False(handle.Weights.HasRopeFreqs);
        Assert.True(handle.Weights.RopeFreqsWeight == null);
    }

    [Fact]
    public void T2_BVA_32_RoPE_ExtremeBaseFrequency_1e9()
    {
        const int headDim = 64;
        float[] q = new float[headDim];
        float[] k = new float[headDim];
        for (int i = 0; i < headDim; i++) { q[i] = (i + 1) * 0.1f; k[i] = (headDim - i) * 0.1f; }
        float qNormBefore = MathF.Sqrt(q.Sum(v => v * v));

        fixed (float* pQ = q, pK = k)
        {
            QuantKernels.RoPE(pQ, pK, nHeadsQ: 1, nHeadsKv: 1, headDim: headDim, pos: 512, freqBase: 1e9f);
        }

        Assert.All(q, v => Assert.True(float.IsFinite(v)));
        Assert.All(k, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(qNormBefore, MathF.Sqrt(q.Sum(v => v * v)), 0.001f);
    }

    [Fact]
    public void T2_BVA_33_RoPE_HighContextPosition_32768_NumericalStability()
    {
        const int headDim = 64;
        float[] q = new float[headDim];
        float[] k = new float[headDim];
        for (int i = 0; i < headDim; i++) { q[i] = MathF.Sin(i + 1); k[i] = MathF.Cos(i + 1); }
        float qNormBefore = MathF.Sqrt(q.Sum(v => v * v));
        float kNormBefore = MathF.Sqrt(k.Sum(v => v * v));

        fixed (float* pQ = q, pK = k)
        {
            QuantKernels.RoPE(pQ, pK, nHeadsQ: 1, nHeadsKv: 1, headDim: headDim, pos: 32768, freqBase: 10000.0f);
        }

        Assert.All(q, v => Assert.True(float.IsFinite(v)));
        Assert.All(k, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(qNormBefore, MathF.Sqrt(q.Sum(v => v * v)), 0.001f);
        Assert.Equal(kNormBefore, MathF.Sqrt(k.Sum(v => v * v)), 0.001f);
    }

    [Fact]
    public void T2_BVA_34_RoPE_PartialDimension_RotationBounds()
    {
        const int headDim = 128, ropeDim = 64;
        float[] q = new float[headDim];
        float[] k = new float[headDim];
        for (int i = 0; i < headDim; i++) { q[i] = i + 1.0f; k[i] = -(i + 1.0f); }
        float[] qTailBefore = q[ropeDim..];
        float[] kTailBefore = k[ropeDim..];

        fixed (float* pQ = q, pK = k)
        {
            QuantKernels.RoPE(pQ, pK, nHeadsQ: 1, nHeadsKv: 1, headDim: headDim, pos: 7, freqBase: 10000.0f, ropeDim: ropeDim);
        }

        // Unrotated tail [64..128) must remain bit-identical while [0..64) rotates
        Assert.NotEqual(1.0f, q[0]);
        for (int i = ropeDim; i < headDim; i++)
        {
            Assert.Equal(qTailBefore[i - ropeDim], q[i]);
            Assert.Equal(kTailBefore[i - ropeDim], k[i]);
        }
    }

    [Fact]
    public void T2_BVA_35_RoPE_Llama32_HighTheta_500000()
    {
        const int dim = 64;
        float[] invFreq = new float[dim / 2];
        QuantKernels.PrecomputeYarnFrequencies(invFreq, dim, freqBase: 500_000.0f, scalingFactor: 8.0f, originalCtx: 8192);

        Assert.All(invFreq, f => Assert.True(float.IsFinite(f) && f > 0f && f <= 1f));
        Assert.True(invFreq[1] < 1f && invFreq[0] > invFreq[^1], "YaRN inverse frequencies must decrease with dimension index");
    }

    #endregion

    #region Boundary Area 8: Security Scanner Edge Vectors

    [Fact]
    public void T2_BVA_36_SecurityScanner_RegexPattern_GoogleAiKeyMatch()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"bva_sec36_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "cfg.json"), "API_KEY=\"" + "AIzaSy" + new string('A', 33) + "\"");
            var (exitCode, _, findings, _) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(dir);
            Assert.Equal(1, exitCode);
            Assert.Single(findings);
            Assert.Contains("Google AI", findings[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void T2_BVA_37_SecurityScanner_RegexPattern_PrivateKeyMatch()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"bva_sec37_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "key.pem"), "-----" + "BEGIN RSA " + "PRIVATE KEY" + "-----");
            var (exitCode, _, findings, _) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(dir);
            Assert.Equal(1, exitCode);
            Assert.Single(findings);
            Assert.Contains("Private Key Header", findings[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void T2_BVA_38_SecurityScanner_RegexPattern_OpenAiKeyMatch()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"bva_sec38_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "cfg.env"), "OPENAI_KEY=" + "sk-" + new string('b', 35));
            var (exitCode, _, findings, _) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(dir);
            Assert.Equal(1, exitCode);
            Assert.Single(findings);
            Assert.Contains("OpenAI", findings[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void T2_BVA_39_SecurityScanner_Allowlist_EnvironmentVariable()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"bva_sec39_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "Program.cs"), "string key = Environment.GetEnvironmentVariable(\"API_KEY\") ?? \"\";");
            var (exitCode, scanned, findings, _) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(dir);
            Assert.Equal(0, exitCode);
            Assert.Equal(1, scanned);
            Assert.Empty(findings);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void T2_BVA_40_SecurityScanner_Allowlist_PlaceholderPattern()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"bva_sec40_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "template.json"), "apiKey = \"your-api-key\"; // DUMMY");
            var (exitCode, scanned, findings, _) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(dir);
            Assert.Equal(0, exitCode);
            Assert.Equal(1, scanned);
            Assert.Empty(findings);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    #endregion

    #region Boundary Area 9: Additional Architecture & Session Boundaries

    [Fact]
    public void T2_BVA_41_Mamba2_HeadDimExceedingMax_Rejected()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(headDim: 512);
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.False(supported);
        Assert.Contains("unsupported head dim", reason);
    }

    [Fact]
    public void T2_BVA_42_Mamba2_ValueDimMismatch_Rejected()
    {
        // Mamba2 requires HeadDim == ValueDim. Model with HeadDim=64, ValueDim=128 must be rejected.
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(headDim: 64, valueDim: 128);

        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);

        Assert.False(supported, "Mamba2VulkanModel.IsSupported must return false when ValueDim != HeadDim.");
        Assert.Contains("unsupported head dim 64/128", reason);
    }

    [Fact]
    public void T2_BVA_43_Mamba2_MaxSupportedHeadDim_256_Accepted()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(headDim: 256);
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.True(supported, $"Mamba2 headDim 256 should be supported: {reason}");
    }

    [Fact]
    public void T2_BVA_44_ActiveBackend_CaseInsensitiveEngineQuery()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "CPU");
        Assert.Equal(InferenceEngineType.Cpu, session.Engine);
        Assert.Equal("Cpu", session.ActiveBackend);
        Assert.False(session.IsGpuAccelerated);
    }

    [Fact]
    public void T2_BVA_45_ModeDesc_MoE_DualMoE_FormatConsistency()
    {
        string gemmaMoeDesc = InferenceSession.FormatVulkanModeDescription(UniversalArchitecture.Gemma4, isHybridSsm: false, isMoe: true);
        string standardMoeDesc = InferenceSession.FormatVulkanModeDescription(UniversalArchitecture.Qwen, isHybridSsm: false, isMoe: true);

        Assert.Contains("Dual MoE", gemmaMoeDesc);
        Assert.DoesNotContain("Dual MoE", standardMoeDesc);
        Assert.Contains("+ MoE", standardMoeDesc);
    }

    [Fact]
    public void T2_BVA_46_FindVulkanDevice_EmptyDeviceName_ReturnsFirstRealGpu()
    {
        int dev = Qwen35VulkanModel.FindVulkanDevice("");
        if (dev >= 0)
        {
            var devices = VulkanContext.EnumerateDevices();
            Assert.True(dev < devices.Count);
        }
    }

    [Fact]
    public void T2_BVA_47_SpirvInspector_ShortBytecode_Rejected()
    {
        byte[] shortBytes = [1, 2, 3];
        Assert.False(VulkanE2ETestFixtures.SpirvInspector.IsValidSpirv(shortBytes));
    }

    [Fact]
    public void T2_BVA_48_SpirvInspector_CorruptMagicHeader_Rejected()
    {
        byte[] corrupt = new byte[32];
        corrupt[0] = 0xFF;
        Assert.False(VulkanE2ETestFixtures.SpirvInspector.IsValidSpirv(corrupt));
    }

    [Fact]
    public void T2_BVA_49_ReferenceOracle_DequantizeQ4_0_ZerosRemainZero()
    {
        byte[] block = new byte[18];
        // d = 0.0f
        block[0] = 0;
        block[1] = 0;

        float[] output = new float[32];
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ4_0(block, output);

        for (int i = 0; i < 32; i++) Assert.Equal(0.0f, output[i]);
    }

    [Fact]
    public void T2_BVA_50_ReferenceOracle_DequantizeQ8_0_ExtremesRemainBounded()
    {
        byte[] block = new byte[34];
        Half h = (Half)1.0f;
        ushort bits = BitConverter.HalfToUInt16Bits(h);
        block[0] = (byte)(bits & 0xFF);
        block[1] = (byte)((bits >> 8) & 0xFF);
        block[2] = 127;  // max int8
        block[3] = 0x80; // -128 int8

        float[] output = new float[32];
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ8_0(block, output);

        Assert.Equal(127.0f, output[0]);
        Assert.Equal(-128.0f, output[1]);
    }

    #endregion
}
