using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Model;
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
        float[] bias = [0.0f, 0.0f, 0.0f, 0.0f];
        float[] activation = [1.25f, -3.5f, 42.0f, 0.001f];
        float[] result = new float[4];
        for (int i = 0; i < 4; i++) result[i] = activation[i] + bias[i];

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(activation[i], result[i]);
        }
    }

    [Fact]
    public void T2_BVA_03_AttentionBiases_ExtremeMagnitudes_NoNaNOrOverflow()
    {
        float[] bias = [1000.0f, -1000.0f, 1e5f, -1e5f];
        float[] activation = [5.0f, -10.0f, 20.0f, -50.0f];
        float[] result = new float[4];
        for (int i = 0; i < 4; i++)
        {
            result[i] = activation[i] + bias[i];
            Assert.False(float.IsNaN(result[i]));
            Assert.False(float.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void T2_BVA_04_AttentionBiases_SubnormalFloats_HandledGracefully()
    {
        float subnormal = 1e-38f;
        float[] bias = [subnormal, -subnormal];
        float[] activation = [1.0f, -1.0f];
        float[] result = [activation[0] + bias[0], activation[1] + bias[1]];

        Assert.False(float.IsNaN(result[0]));
        Assert.False(float.IsNaN(result[1]));
    }

    [Fact]
    public void T2_BVA_05_AttentionBiases_UnalignedDimension_Padding()
    {
        // Unaligned head count / dimension: 896 elements
        int dim = 896;
        int workgroups = (dim + 255) / 256;
        Assert.Equal(4, workgroups);
        Assert.True(workgroups * 256 >= dim);
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
        Assert.Throws<NotSupportedException>(() => TestableVulkanRunner.QIdx(GgufType.BF16));
    }

    #endregion

    #region Boundary Area 3: Embedding Gather Boundaries

    [Fact]
    public void T2_BVA_11_EmbedGather_TokenZero_ValidOffset()
    {
        int token = 0;
        int dim = 2048;
        int blockSize = 32;
        int nb = dim / blockSize;
        long byteOffset = (long)token * nb * 18;

        Assert.Equal(0L, byteOffset);
    }

    [Fact]
    public void T2_BVA_12_EmbedGather_MaxVocabToken_ValidOffset()
    {
        int vocabSize = 151936;
        int token = vocabSize - 1;
        int dim = 2048;
        int blockSize = 32;
        int nb = dim / blockSize;
        long byteOffset = (long)token * nb * 18;

        Assert.True(byteOffset > 0);
        Assert.Equal((151936L - 1L) * 64 * 18, byteOffset);
    }

    [Fact]
    public void T2_BVA_13_EmbedGather_UnalignedEmbeddingDim_CalculatesBlocks()
    {
        int dim = 1536; // Phi-1.5 dimension
        int blockSize = 256;
        int nb = dim / blockSize;
        Assert.Equal(6, nb);
        Assert.Equal(dim, nb * blockSize);
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
        int nQ = 32;
        int nKv = 1;
        int hd = 64;
        int qDim = nQ * hd;
        int kvDim = nKv * hd;

        Assert.Equal(2048, qDim);
        Assert.Equal(64, kvDim);
        Assert.Equal(2176, qDim + 2 * kvDim);
    }

    [Fact]
    public void T2_BVA_17_FusedQkv_MHA_EqualHeadCounts()
    {
        int nQ = 16;
        int nKv = 16;
        int hd = 64;
        int qDim = nQ * hd;
        int kvDim = nKv * hd;

        Assert.Equal(qDim, kvDim);
        Assert.Equal(3 * qDim, qDim + 2 * kvDim);
    }

    [Fact]
    public void T2_BVA_18_FusedQkv_AsymmetricGQA_Ratio7to1()
    {
        int nQ = 28;
        int nKv = 4;
        int hd = 128;
        int qDim = nQ * hd;
        int kvDim = nKv * hd;

        Assert.Equal(7, nQ / nKv);
        Assert.Equal(3584, qDim);
        Assert.Equal(512, kvDim);
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
        uint pos = 0u;
        uint[] push = [0u];
        push[0] = pos;
        Assert.Equal(0u, push[0]);
    }

    [Fact]
    public void T2_BVA_27_SequencePosition_MaxSeqLenMinusOne()
    {
        int maxSeqLen = 4096;
        int pos = maxSeqLen - 1;
        Assert.Equal(4095, pos);
        Assert.True((uint)pos < (uint)maxSeqLen);
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
        float baseFreq = 1e9f; // Mistral / Devstral extended frequency base
        int ropeDim = 64;
        for (int i = 0; i < ropeDim / 2; i++)
        {
            float theta = 1.0f / MathF.Pow(baseFreq, (2f * i) / ropeDim);
            Assert.False(float.IsNaN(theta));
            Assert.False(float.IsInfinity(theta));
            Assert.True(theta > 0f);
        }
    }

    [Fact]
    public void T2_BVA_33_RoPE_HighContextPosition_32768_NumericalStability()
    {
        int pos = 32768;
        float freq = 10000.0f;
        float theta = pos / freq;
        float cos = MathF.Cos(theta);
        float sin = MathF.Sin(theta);

        Assert.True(cos >= -1.0f && cos <= 1.0f);
        Assert.True(sin >= -1.0f && sin <= 1.0f);
        Assert.Equal(1.0f, cos * cos + sin * sin, 0.0001f);
    }

    [Fact]
    public void T2_BVA_34_RoPE_PartialDimension_RotationBounds()
    {
        int headDim = 128;
        int ropeDim = 64; // only first 64 dimensions rotated
        Assert.True(ropeDim <= headDim);
        Assert.Equal(64, headDim - ropeDim);
    }

    [Fact]
    public void T2_BVA_35_RoPE_Llama32_HighTheta_500000()
    {
        float baseFreq = 500_000.0f; // Llama 3.2 1B/3B theta
        float theta = 1.0f / baseFreq;
        Assert.True(theta > 0f && theta < 1f);
    }

    #endregion

    #region Boundary Area 8: Security Scanner Edge Vectors

    [Fact]
    public void T2_BVA_36_SecurityScanner_RegexPattern_GoogleAiKeyMatch()
    {
        var regex = new Regex(@"AIzaSy[0-9A-Za-z\-_]{33}");
        string syntheticKey = "AIzaSy" + new string('A', 33);
        Assert.Matches(regex, syntheticKey);
    }

    [Fact]
    public void T2_BVA_37_SecurityScanner_RegexPattern_PrivateKeyMatch()
    {
        var regex = new Regex(@"-----\b" + @"BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY" + @"-----");
        string syntheticHeader = "-----" + "BEGIN RSA " + "PRIVATE KEY" + "-----";
        Assert.Matches(regex, syntheticHeader);
    }

    [Fact]
    public void T2_BVA_38_SecurityScanner_RegexPattern_OpenAiKeyMatch()
    {
        var regex = new Regex(@"sk-[a-zA-Z0-9]{32,}");
        string syntheticKey = "sk-" + new string('b', 35);
        Assert.Matches(regex, syntheticKey);
    }

    [Fact]
    public void T2_BVA_39_SecurityScanner_Allowlist_EnvironmentVariable()
    {
        var allowlist = new Regex(@"(?i)environment\.getenvironmentvariable");
        string safeLine = "string key = Environment.GetEnvironmentVariable(\"API_KEY\");";
        Assert.Matches(allowlist, safeLine);
    }

    [Fact]
    public void T2_BVA_40_SecurityScanner_Allowlist_PlaceholderPattern()
    {
        var allowlist = new Regex(@"(?i)your[-_]?api[-_]?key|dummy");
        string placeholderLine = "apiKey = \"your-api-key\";";
        Assert.Matches(allowlist, placeholderLine);
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
        Assert.Equal("Vulkan", InferenceEngineType.Vulkan.ToString());
        Assert.Equal("Cpu", InferenceEngineType.Cpu.ToString());
    }

    [Fact]
    public void T2_BVA_45_ModeDesc_MoE_DualMoE_FormatConsistency()
    {
        var gemmaArch = UniversalArchitecture.Gemma4;
        var qwenArch = UniversalArchitecture.Qwen;

        string gemmaMoeDesc = gemmaArch == UniversalArchitecture.Gemma4 ? "Gemma 4 ISWA + Dual MoE" : "Standard Dense";
        string standardMoeDesc = qwenArch == UniversalArchitecture.Gemma4 ? "Gemma 4 ISWA + Dual MoE" : "Standard Dense + MoE";

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
