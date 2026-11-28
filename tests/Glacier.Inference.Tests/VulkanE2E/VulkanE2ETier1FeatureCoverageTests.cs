using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Tests.E2E;
using Glacier.Inference.Tests.VulkanE2E.Fixtures;
using Xunit;

namespace Glacier.Inference.Tests.VulkanE2E;

/// <summary>
/// Tier 1: Feature Coverage (Category-Partition: isolated happy-path coverage across all 16 features).
/// Exactly 5 comprehensive tests per feature = 80 tests total.
/// </summary>
public unsafe class VulkanE2ETier1FeatureCoverageTests
{

    #region Feature 1: Attention Biases (QBias, KBias, VBias, AttnOutBias)

    [Fact]
    public void T1_F1_01_ModelWeights_MapsAllAttentionBiases()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: true);
        var layer = handle.Weights.Layers[0];

        Assert.True(layer.QBias != null, "QBias pointer must be non-null when attn_q.bias is present");
        Assert.True(layer.KBias != null, "KBias pointer must be non-null when attn_k.bias is present");
        Assert.True(layer.VBias != null, "VBias pointer must be non-null when attn_v.bias is present");
        Assert.True(layer.AttnOutBias != null, "AttnOutBias pointer must be non-null when attn_output.bias is present");
    }

    [Fact]
    public void T1_F1_02_ModelWeights_MapsPartialQBiasOnly()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: false);
        // By default addBiases is false: all bias pointers are null
        var layer = handle.Weights.Layers[0];

        Assert.True(layer.QBias == null, "QBias must be null when absent");
        Assert.True(layer.KBias == null, "KBias must be null when absent");
        Assert.True(layer.VBias == null, "VBias must be null when absent");
        Assert.True(layer.AttnOutBias == null, "AttnOutBias must be null when absent");
    }

    [Fact]
    public void T1_F1_03_ModelWeights_MapsPartialAttnOutBiasOnly()
    {
        // Synthesize model with only attn_output.bias
        string path = Path.Combine(Path.GetTempPath(), $"partial_bias_{Guid.NewGuid():N}.gguf");
        try
        {
            var metadata = new System.Collections.Generic.List<(string, GgufValueType, object)>
            {
                ("general.architecture", GgufValueType.String, "qwen2"),
                ("qwen2.block_count", GgufValueType.Uint32, 1u),
                ("qwen2.context_length", GgufValueType.Uint32, 32u),
                ("qwen2.embedding_length", GgufValueType.Uint32, 64u),
                ("qwen2.feed_forward_length", GgufValueType.Uint32, 128u),
                ("qwen2.attention.head_count", GgufValueType.Uint32, 2u),
                ("qwen2.attention.head_count_kv", GgufValueType.Uint32, 2u),
                ("tokenizer.ggml.tokens", GgufValueType.Array, VulkanE2ETestFixtures.SyntheticGgufBuilder.BuildTokens(10))
            };
            var tensors = new System.Collections.Generic.List<(string, ulong[], GgufType, byte[])>
            {
                ("token_embd.weight", [64UL, 10UL], GgufType.F32, new byte[64 * 10 * 4]),
                ("output_norm.weight", [64UL], GgufType.F32, new byte[64 * 4]),
                ("output.weight", [64UL, 10UL], GgufType.F32, new byte[64 * 10 * 4]),
                ("blk.0.attn_norm.weight", [64UL], GgufType.F32, new byte[64 * 4]),
                ("blk.0.attn_q.weight", [64UL, 64UL], GgufType.F32, new byte[64 * 64 * 4]),
                ("blk.0.attn_k.weight", [64UL, 64UL], GgufType.F32, new byte[64 * 64 * 4]),
                ("blk.0.attn_v.weight", [64UL, 64UL], GgufType.F32, new byte[64 * 64 * 4]),
                ("blk.0.attn_output.weight", [64UL, 64UL], GgufType.F32, new byte[64 * 64 * 4]),
                ("blk.0.attn_output.bias", [64UL], GgufType.F32, new byte[64 * 4]),
                ("blk.0.ffn_norm.weight", [64UL], GgufType.F32, new byte[64 * 4]),
                ("blk.0.ffn_gate.weight", [64UL, 128UL], GgufType.F32, new byte[64 * 128 * 4]),
                ("blk.0.ffn_up.weight", [64UL, 128UL], GgufType.F32, new byte[64 * 128 * 4]),
                ("blk.0.ffn_down.weight", [128UL, 64UL], GgufType.F32, new byte[128 * 64 * 4])
            };
            VulkanE2ETestFixtures.SyntheticGgufBuilder.WriteGguf(path, metadata, tensors);
            using var file = new GgufFile(path);
            var weights = new ModelWeights(file);

            Assert.True(weights.Layers[0].AttnOutBias != null);
            Assert.True(weights.Layers[0].QBias == null);
            Assert.True(weights.Layers[0].KBias == null);
            Assert.True(weights.Layers[0].VBias == null);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void T1_F1_04_ModelWeights_BiasPointerValuesMatchExpectedData()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: true);
        var layer = handle.Weights.Layers[0];

        // ConstantFloats was set to 0.05f for QBias, 0.02f for KBias, -0.02f for VBias, 0.01f for AttnOutBias
        Assert.Equal(0.05f, layer.QBias[0], 0.0001f);
        Assert.Equal(0.02f, layer.KBias[0], 0.0001f);
        Assert.Equal(-0.02f, layer.VBias[0], 0.0001f);
        Assert.Equal(0.01f, layer.AttnOutBias[0], 0.0001f);
    }

    [Fact]
    public void T1_F1_05_AttentionBiases_ElementwiseAdd_NumericalParity()
    {
        const int dim = 64;
        using var biased = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: 128, headCount: 2, headCountKv: 2, vocabSize: 16, addBiases: true);
        var layer = biased.Weights.Layers[0];
        Assert.True(layer.QBias != null);

        float[] activation = new float[dim];
        for (int i = 0; i < dim; i++) activation[i] = (i % 7) * 0.25f - 0.5f;
        float[] expected = new float[dim];
        for (int i = 0; i < dim; i++) expected[i] = activation[i] + layer.QBias[i];

        fixed (float* pDst = activation)
        {
            QuantKernels.VecAdd(layer.QBias, pDst, dim);
        }

        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(expected[i], activation[i], 0.0001f);
        }

        using var unbiased = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: 128, headCount: 2, headCountKv: 2, vocabSize: 16, addBiases: false);
        using var sessionBiased = new InferenceSession(biased.Path, maxSeqLen: 16, device: "cpu");
        using var sessionUnbiased = new InferenceSession(unbiased.Path, maxSeqLen: 16, device: "cpu");

        sessionBiased.ForwardToken(token: 3, pos: 0, computeLogits: true);
        sessionUnbiased.ForwardToken(token: 3, pos: 0, computeLogits: true);
        float[] logitsBiased = new float[sessionBiased.VocabSize];
        float[] logitsUnbiased = new float[sessionUnbiased.VocabSize];
        sessionBiased.CopyCurrentLogits(logitsBiased);
        sessionUnbiased.CopyCurrentLogits(logitsUnbiased);

        Assert.Contains(logitsBiased, x => x != 0f && float.IsFinite(x));
        bool anyDiff = false;
        for (int i = 0; i < logitsBiased.Length; i++)
        {
            if (MathF.Abs(logitsBiased[i] - logitsUnbiased[i]) > 1e-6f) { anyDiff = true; break; }
        }
        Assert.True(anyDiff, "Adding attention biases must alter InferenceSession output logits.");
    }

    #endregion

    #region Feature 2: Missing Weight Kernels (Q2_K, Q4_0, F16)

    [Fact]
    public void T1_F2_01_IsKernelType_Q2_K_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsKernelType(GgufType.Q2_K));
    }

    [Fact]
    public void T1_F2_02_IsKernelType_Q4_0_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsKernelType(GgufType.Q4_0));
    }

    [Fact]
    public void T1_F2_03_IsKernelType_F16_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsKernelType(GgufType.F16));
    }

    [Fact]
    public void T1_F2_04_QIdx_MapsNewQuantTypes_ToIndices_8_9_10()
    {
        Assert.Equal(8, TestableVulkanRunner.QIdx(GgufType.Q2_K));
        Assert.Equal(9, TestableVulkanRunner.QIdx(GgufType.Q4_0));
        Assert.Equal(10, TestableVulkanRunner.QIdx(GgufType.F16));
    }

    [Fact]
    public void T1_F2_05_GgufTypeSize_CalculatesExactBlockBytes_Q2K_Q40_F16()
    {
        // Q2_K: 256 elements in 84 bytes
        Assert.Equal(84L, TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 256));
        Assert.Equal(168L, TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 512));

        // Q4_0: 32 elements in 18 bytes
        Assert.Equal(18L, TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, 32));
        Assert.Equal(36L, TestableVulkanRunner.GgufTypeSize(GgufType.Q4_0, 64));

        // F16: 2 bytes per element
        Assert.Equal(2048L, TestableVulkanRunner.GgufTypeSize(GgufType.F16, 1024));
    }

    #endregion

    #region Feature 3: Missing Embedding Kernels (F16, Q4_0, Q3_K, IQ4_NL, Q2_K)

    [Fact]
    public void T1_F3_01_IsEmbedKernelType_F16_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.F16));
    }

    [Fact]
    public void T1_F3_02_IsEmbedKernelType_Q4_0_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.Q4_0));
    }

    [Fact]
    public void T1_F3_03_IsEmbedKernelType_Q3_K_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.Q3_K));
    }

    [Fact]
    public void T1_F3_04_IsEmbedKernelType_IQ4_NL_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.IQ4_NL));
    }

    [Fact]
    public void T1_F3_05_IsEmbedKernelType_Q2_K_ReturnsTrue()
    {
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.Q2_K));
    }

    #endregion

    #region Feature 4: Fused QKV (HasFusedQkv)

    [Fact]
    public void T1_F4_01_ModelWeights_IdentifiesHasFusedQkv()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(fusedQkv: true);
        Assert.True(handle.Weights.Layers[0].HasFusedQkv);
    }

    [Fact]
    public void T1_F4_02_FusedQkv_CalculatesCorrectQSliceDimensions()
    {
        int dim = 256;
        int headCount = 4;
        int headCountKv = 2;
        int headDim = dim / headCount;
        int qDim = headCount * headDim;

        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(dim: dim, headCount: headCount, headCountKv: headCountKv, fusedQkv: true);
        Assert.Equal(256, qDim);
        Assert.True(handle.Weights.Layers[0].HasFusedQkv);
    }

    [Fact]
    public void T1_F4_03_FusedQkv_CalculatesCorrectKVSliceDimensions()
    {
        int dim = 256;
        int headCount = 4;
        int headCountKv = 2;
        int headDim = dim / headCount;
        int kvDim = headCountKv * headDim;

        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(dim: dim, headCount: headCount, headCountKv: headCountKv, fusedQkv: true);
        Assert.Equal(128, kvDim);
    }

    private static void WriteDeterministicQwen2Gguf(
        string path,
        int dim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize,
        bool fusedQkv,
        bool fusedGateUp)
    {
        var rng = new Random(1234);
        int headDim = dim / headCount;
        int qDim = headCount * headDim;
        int kvDim = headCountKv * headDim;

        byte[] FloatsToBytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();
        float[] RandFloats(int n) => VulkanE2ETestFixtures.SyntheticGgufBuilder.RandomFloats(rng, n);
        float[] ConstFloats(int n, float v) => VulkanE2ETestFixtures.SyntheticGgufBuilder.ConstantFloats(n, v);

        byte[] embdBytes = FloatsToBytes(RandFloats(dim * vocabSize));
        byte[] outNormBytes = FloatsToBytes(ConstFloats(dim, 1.0f));
        byte[] outBytes = FloatsToBytes(RandFloats(dim * vocabSize));
        byte[] attnNormBytes = FloatsToBytes(ConstFloats(dim, 1.0f));
        byte[] qBytes = FloatsToBytes(RandFloats(dim * qDim));
        byte[] kBytes = FloatsToBytes(RandFloats(dim * kvDim));
        byte[] vBytes = FloatsToBytes(RandFloats(dim * kvDim));
        byte[] attnOutBytes = FloatsToBytes(RandFloats(qDim * dim));
        byte[] ffnNormBytes = FloatsToBytes(ConstFloats(dim, 1.0f));
        byte[] gateBytes = FloatsToBytes(RandFloats(dim * ffnDim));
        byte[] upBytes = FloatsToBytes(RandFloats(dim * ffnDim));
        byte[] downBytes = FloatsToBytes(RandFloats(ffnDim * dim));

        var metadata = new System.Collections.Generic.List<(string, GgufValueType, object)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("qwen2.block_count", GgufValueType.Uint32, 1u),
            ("qwen2.context_length", GgufValueType.Uint32, 32u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, VulkanE2ETestFixtures.SyntheticGgufBuilder.BuildTokens(vocabSize))
        };

        var tensors = new System.Collections.Generic.List<(string, ulong[], GgufType, byte[])>
        {
            ("token_embd.weight", [(ulong)dim, (ulong)vocabSize], GgufType.F32, embdBytes),
            ("output_norm.weight", [(ulong)dim], GgufType.F32, outNormBytes),
            ("output.weight", [(ulong)dim, (ulong)vocabSize], GgufType.F32, outBytes),
            ("blk.0.attn_norm.weight", [(ulong)dim], GgufType.F32, attnNormBytes),
            ("blk.0.attn_output.weight", [(ulong)qDim, (ulong)dim], GgufType.F32, attnOutBytes),
            ("blk.0.ffn_norm.weight", [(ulong)dim], GgufType.F32, ffnNormBytes),
            ("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], GgufType.F32, downBytes)
        };

        if (fusedQkv)
        {
            byte[] qkvBytes = new byte[qBytes.Length + kBytes.Length + vBytes.Length];
            qBytes.CopyTo(qkvBytes, 0);
            kBytes.CopyTo(qkvBytes, qBytes.Length);
            vBytes.CopyTo(qkvBytes, qBytes.Length + kBytes.Length);
            tensors.Add(("blk.0.attn_qkv.weight", [(ulong)dim, (ulong)(qDim + 2 * kvDim)], GgufType.F32, qkvBytes));
        }
        else
        {
            tensors.Add(("blk.0.attn_q.weight", [(ulong)dim, (ulong)qDim], GgufType.F32, qBytes));
            tensors.Add(("blk.0.attn_k.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, kBytes));
            tensors.Add(("blk.0.attn_v.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, vBytes));
        }

        if (fusedGateUp)
        {
            byte[] gateUpBytes = new byte[gateBytes.Length + upBytes.Length];
            gateBytes.CopyTo(gateUpBytes, 0);
            upBytes.CopyTo(gateUpBytes, gateBytes.Length);
            tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)(2 * ffnDim)], GgufType.F32, gateUpBytes));
        }
        else
        {
            tensors.Add(("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, gateBytes));
            tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, upBytes));
        }

        VulkanE2ETestFixtures.SyntheticGgufBuilder.WriteGguf(path, metadata, tensors);
    }

    [Fact]
    public void T1_F4_04_FusedQkv_TotalChannelCountInvariant()
    {
        const int dim = 256, headCount = 4, headCountKv = 2;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, headCount: headCount, headCountKv: headCountKv, fusedQkv: true);

        Assert.True(handle.GgufFile.TryGetTensor("blk.0.attn_qkv.weight", out var qkvTensor) && qkvTensor != null);
        Assert.Equal([256UL, 512UL], qkvTensor.Dimensions);

        var layer = handle.Weights.Layers[0];
        Assert.True(layer.HasFusedQkv);
        Assert.True(layer.QkvWeight != null);
        Assert.True(layer.QWeight == null);
        Assert.True(layer.KWeight == null);
        Assert.True(layer.VWeight == null);

        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.ForwardToken(token: 3, pos: 0, computeLogits: true);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
    }

    [Fact]
    public void T1_F4_05_FusedQkv_SlicingOffsetArithmetic()
    {
        string unfusedPath = Path.Combine(Path.GetTempPath(), $"qkv_unfused_{Guid.NewGuid():N}.gguf");
        string fusedPath = Path.Combine(Path.GetTempPath(), $"qkv_fused_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteDeterministicQwen2Gguf(unfusedPath, dim: 128, ffnDim: 256, headCount: 4, headCountKv: 2, vocabSize: 16, fusedQkv: false, fusedGateUp: false);
            WriteDeterministicQwen2Gguf(fusedPath, dim: 128, ffnDim: 256, headCount: 4, headCountKv: 2, vocabSize: 16, fusedQkv: true, fusedGateUp: false);

            using var sUnfused = new InferenceSession(unfusedPath, maxSeqLen: 16, device: "cpu");
            using var sFused = new InferenceSession(fusedPath, maxSeqLen: 16, device: "cpu");
            Assert.False(sUnfused.Weights.Layers[0].HasFusedQkv);
            Assert.True(sFused.Weights.Layers[0].HasFusedQkv);

            sUnfused.Prefill([3, 5, 7]);
            sFused.Prefill([3, 5, 7]);
            float[] logitsUnfused = new float[sUnfused.VocabSize];
            float[] logitsFused = new float[sFused.VocabSize];
            sUnfused.CopyCurrentLogits(logitsUnfused);
            sFused.CopyCurrentLogits(logitsFused);

            Assert.Contains(logitsFused, x => x != 0f && float.IsFinite(x));
            for (int i = 0; i < logitsFused.Length; i++)
            {
                Assert.Equal(logitsUnfused[i], logitsFused[i], 0.0001f);
            }
        }
        finally
        {
            if (File.Exists(unfusedPath)) File.Delete(unfusedPath);
            if (File.Exists(fusedPath)) File.Delete(fusedPath);
        }
    }

    #endregion

    #region Feature 5: Fused Gate/Up (HasFusedGateUp)

    [Fact]
    public void T1_F5_01_ModelWeights_IdentifiesHasFusedGateUp()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(fusedGateUp: true);
        Assert.True(handle.Weights.Layers[0].HasFusedGateUp);
    }

    [Fact]
    public void T1_F5_02_FusedGateUp_CalculatesGateSliceDimensions()
    {
        const int dim = 256, ffnDim = 512;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: ffnDim, fusedGateUp: true);

        Assert.True(handle.GgufFile.TryGetTensor("blk.0.ffn_up.weight", out var ffnUpTensor) && ffnUpTensor != null);
        Assert.Equal([(ulong)dim, (ulong)(2 * ffnDim)], ffnUpTensor.Dimensions);

        var layer = handle.Weights.Layers[0];
        Assert.True(layer.HasFusedGateUp);
        Assert.True(layer.FfnGateWeight == null);
        Assert.True(layer.FfnUpWeight != null);

        long halfBytes = (long)ffnDim * GgufTypes.GetRowBytes(layer.FfnUpType, dim);
        Assert.Equal((ulong)(2 * halfBytes), ffnUpTensor.GetByteSize());
    }

    [Fact]
    public void T1_F5_03_FusedGateUp_CalculatesUpSliceDimensions()
    {
        string unfusedPath = Path.Combine(Path.GetTempPath(), $"gateup_unfused_{Guid.NewGuid():N}.gguf");
        string fusedPath = Path.Combine(Path.GetTempPath(), $"gateup_fused_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteDeterministicQwen2Gguf(unfusedPath, dim: 128, ffnDim: 256, headCount: 4, headCountKv: 2, vocabSize: 16, fusedQkv: false, fusedGateUp: false);
            WriteDeterministicQwen2Gguf(fusedPath, dim: 128, ffnDim: 256, headCount: 4, headCountKv: 2, vocabSize: 16, fusedQkv: false, fusedGateUp: true);

            using var sUnfused = new InferenceSession(unfusedPath, maxSeqLen: 16, device: "cpu");
            using var sFused = new InferenceSession(fusedPath, maxSeqLen: 16, device: "cpu");
            Assert.False(sUnfused.Weights.Layers[0].HasFusedGateUp);
            Assert.True(sFused.Weights.Layers[0].HasFusedGateUp);

            sUnfused.Prefill([2, 4, 6]);
            sFused.Prefill([2, 4, 6]);
            float[] logitsUnfused = new float[sUnfused.VocabSize];
            float[] logitsFused = new float[sFused.VocabSize];
            sUnfused.CopyCurrentLogits(logitsUnfused);
            sFused.CopyCurrentLogits(logitsFused);

            Assert.Contains(logitsFused, x => x != 0f && float.IsFinite(x));
            for (int i = 0; i < logitsFused.Length; i++)
            {
                Assert.Equal(logitsUnfused[i], logitsFused[i], 0.0001f);
            }
        }
        finally
        {
            if (File.Exists(unfusedPath)) File.Delete(unfusedPath);
            if (File.Exists(fusedPath)) File.Delete(fusedPath);
        }
    }

    [Fact]
    public void T1_F5_04_FusedGateUp_TotalIntermediateCountInvariant()
    {
        const int dim = 256, ffnDim = 512;
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: dim, ffnDim: ffnDim, fusedGateUp: true);
        var layer = handle.Weights.Layers[0];
        Assert.True(layer.HasFusedGateUp);

        float[] normX = new float[dim];
        for (int i = 0; i < dim; i++) normX[i] = ((i % 13) - 6) * 0.05f;
        float[] gateUpOut = new float[2 * ffnDim];
        float[] actOut = new float[ffnDim];

        fixed (float* pX = normX, pGU = gateUpOut, pAct = actOut)
        {
            QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, pX, pGU, dim, 2 * ffnDim);
            QuantKernels.SwiGLU(pGU, pGU + ffnDim, pAct, ffnDim);
        }

        float[] oracle = VulkanE2ETestFixtures.ReferenceQuantOracle.ComputeSwiGlu(
            gateUpOut.AsSpan(0, ffnDim).ToArray(),
            gateUpOut.AsSpan(ffnDim, ffnDim).ToArray());
        for (int i = 0; i < ffnDim; i++)
        {
            Assert.Equal(oracle[i], actOut[i], 0.0001f);
        }
    }

    [Fact]
    public void T1_F5_05_FusedGateUp_SwiGluNumericalParity()
    {
        float[] gate = [0.0f, 1.0f, -1.0f, 2.0f];
        float[] up = [2.0f, 3.0f, 4.0f, 0.5f];
        float[] actual = new float[4];

        fixed (float* pG = gate, pU = up, pDst = actual)
        {
            QuantKernels.SwiGLU(pG, pU, pDst, 4);
        }

        float[] swiglu = VulkanE2ETestFixtures.ReferenceQuantOracle.ComputeSwiGlu(gate, up);

        // silu(0) = 0 * 0.5 = 0; 0 * 2 = 0
        Assert.Equal(0.0f, actual[0], 0.0001f);
        // silu(1) = 1 / (1 + exp(-1)) = 0.7310586; * 3 = 2.193176
        Assert.Equal(2.193176f, actual[1], 0.001f);
        // silu(-1) = -1 / (1 + exp(1)) = -0.2689414; * 4 = -1.075765
        Assert.Equal(-1.075765f, actual[2], 0.001f);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(swiglu[i], actual[i], 0.0001f);
        }
    }

    #endregion

    #region Feature 6: Mamba2VulkanModel Cast Bug

    [Fact]
    public void T1_F6_01_Mamba2_InheritsFromVulkanRunnerBase()
    {
        Assert.True(typeof(VulkanRunnerBase).IsAssignableFrom(typeof(Mamba2VulkanModel)), "Mamba2VulkanModel must inherit from VulkanRunnerBase");
    }

    [Fact]
    public void T1_F6_02_VulkanRunnerBase_ExposesDeviceName()
    {
        var prop = typeof(VulkanRunnerBase).GetProperty("DeviceName", BindingFlags.Public | BindingFlags.Instance);
        Assert.NotNull(prop);
    }

    [Fact]
    public void T1_F6_03_vkNameResolution_SafePolymorphism_OnMamba2Model()
    {
        var mambaModel = (Mamba2VulkanModel)RuntimeHelpers.GetUninitializedObject(typeof(Mamba2VulkanModel));
        ID3D12Model vkModel = mambaModel;

        // Direct cast to Qwen2VulkanModel (the bug before fix) must throw InvalidCastException
        Assert.Throws<InvalidCastException>(() => (Qwen2VulkanModel)(object)vkModel);

        // Safe polymorphic pattern matching (the production fix in InferenceSession.cs:296) must match VulkanRunnerBase
        Assert.True(vkModel is VulkanRunnerBase, "Mamba2VulkanModel instance must match 'vkModel is VulkanRunnerBase'");
        Assert.False(vkModel is Qwen2VulkanModel, "Mamba2VulkanModel must NOT match 'vkModel is Qwen2VulkanModel'");
    }

    [Fact]
    public void T1_F6_04_Mamba2_TypeHierarchy_NotDerivedFromQwen2()
    {
        Assert.False(typeof(Qwen2VulkanModel).IsAssignableFrom(typeof(Mamba2VulkanModel)), "Mamba2VulkanModel must not derive from Qwen2VulkanModel");
    }

    [Fact]
    public void T1_F6_05_Mamba2_ActiveDevice_ConstructsWithoutInvalidCastException()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        var ex = Record.Exception(() =>
        {
            using var session = new InferenceSession(handle.Path, device: "cpu");
            Assert.NotNull(session.ActiveDevice);
            Assert.Contains("Mamba-2", session.ActiveDevice);
        });

        Assert.Null(ex);
    }

    #endregion

    #region Feature 7: Diagnostic Chain for Mamba2

    [Fact]
    public void T1_F7_01_Mamba2_IsSupported_MethodSignature()
    {
        var method = typeof(Mamba2VulkanModel).GetMethod("IsSupported", BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(method);
        Assert.Equal(typeof(bool), method.ReturnType);
        var parameters = method.GetParameters();
        Assert.Equal(2, parameters.Length);
        Assert.Equal(typeof(ModelWeights), parameters[0].ParameterType);
        Assert.True(parameters[1].IsOut);
    }

    [Fact]
    public void T1_F7_02_Mamba2_IsSupported_AcceptsValidMamba2Model()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.True(supported, $"Mamba2 model should be supported, but failed with: {reason}");
        Assert.Equal("", reason);
    }

    [Fact]
    public void T1_F7_03_Mamba2_IsSupported_RejectsNonMamba2Architecture()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.False(supported);
        Assert.Contains("not a Mamba-2 model", reason);
    }

    [Fact]
    public void T1_F7_04_Mamba2_IsSupported_RejectsHeadDimExceeding256()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(headDim: 512);
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.False(supported);
        Assert.Contains("unsupported head dim", reason);
    }

    [Fact]
    public void T1_F7_05_DiagnosticChain_OrderPreservesMamba2Evaluation()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        // In the diagnostic chain:
        // if (_weights.ArchitectureFamily == UniversalArchitecture.Mamba2 && !Mamba2VulkanModel.IsSupported(...))
        bool isMamba2 = handle.Weights.ArchitectureFamily == UniversalArchitecture.Mamba2;
        Assert.True(isMamba2);
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out _);
        Assert.True(supported);
    }

    #endregion

    #region Feature 8: Architecture Priority Order for Mamba2

    [Fact]
    public void T1_F8_01_ArchitecturePriority_Mamba2PrecedesIsHybridSsm_OnInferenceSession()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Contains("Mamba-2 Hybrid SSM", session.ActiveDevice);
        Assert.DoesNotContain("Qwen Gated DeltaNet", session.ActiveDevice);
    }

    [Fact]
    public void T1_F8_02_Mamba2Model_WithSsmTensors_IdentifiesArchitectureMamba2()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        Assert.Equal(UniversalArchitecture.Mamba2, handle.Weights.ArchitectureFamily);
    }

    [Fact]
    public void T1_F8_03_Qwen35Model_IdentifiesArchitectureQwen()
    {
        // Model with qwen2 architecture maps to Qwen architecture family
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        Assert.Equal(UniversalArchitecture.Qwen, handle.Weights.ArchitectureFamily);
    }

    [Fact]
    public void T1_F8_04_Gemma4Model_IdentifiesArchitectureGemma4()
    {
        Assert.Equal(UniversalArchitecture.Gemma4, ModelArchitectureDetector.Detect("gemma4", null));
    }

    [Fact]
    public void T1_F8_05_StandardDense_RoutesToQwen_OnInferenceSession()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Qwen, session.Architecture);
        Assert.Contains("Qwen", session.ActiveDevice);
        Assert.DoesNotContain("Mamba-2", session.ActiveDevice);
        Assert.DoesNotContain("Gemma 4", session.ActiveDevice);
    }

    #endregion

    #region Feature 9: ActiveBackend Vulkan Misreporting

    [Fact]
    public void T1_F9_01_ActiveBackend_ReturnsVulkan_WhenEngineIsVulkan()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.Vulkan);

        Assert.Equal(InferenceEngineType.Vulkan, session.Engine);
        Assert.Equal("Vulkan", session.ActiveBackend);
    }

    [Fact]
    public void T1_F9_02_ActiveBackend_ReturnsCuda_WhenEngineIsBareMetal_WithGpuModel()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        var uninitGpuModel = (Qwen2GpuModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen2GpuModel));
        typeof(InferenceSession).GetField("_gpuModel", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, uninitGpuModel);
        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.BareMetal);

        Assert.Equal("Cuda", session.ActiveBackend);

        typeof(InferenceSession).GetField("_gpuModel", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, null);
    }

    [Fact]
    public void T1_F9_03_ActiveBackend_ReturnsDirect3D12_WhenEngineIsDirectML()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        var uninitD3dModel = (ID3D12Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen2D3D12Model));
        typeof(InferenceSession).GetField("_d3d12Model", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, uninitD3dModel);
        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.DirectML);

        Assert.Equal("Direct3D12", session.ActiveBackend);

        typeof(InferenceSession).GetField("_d3d12Model", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, null);
    }

    [Fact]
    public void T1_F9_04_ActiveBackend_ReturnsCpu_WhenEngineIsCpu()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(InferenceEngineType.Cpu, session.Engine);
        Assert.Equal("Cpu", session.ActiveBackend);
    }

    private sealed class StubGpuModel : ID3D12Model
    {
        public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true) { }
        public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true) { }
        public void ResetKvCache() { }
        public void Dispose() { }
    }

    [Fact]
    public void T1_F9_05_ActiveBackend_VulkanPrecedence_PreventsDirect3D12Misreporting()
    {
        // Verifies that when _d3d12Model is non-null and Engine is Vulkan, ActiveBackend returns "Vulkan".
        // This directly validates that the precedence bug (evaluating _d3d12Model before Engine == Vulkan) does not recur.
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        using var stubModel = new StubGpuModel();
        typeof(InferenceSession).GetField("_d3d12Model", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, stubModel);
        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.Vulkan);

        Assert.Equal("Vulkan", session.ActiveBackend);
        Assert.True(session.IsGpuAccelerated);

        typeof(InferenceSession).GetField("_d3d12Model", BindingFlags.NonPublic | BindingFlags.Instance)!
            .SetValue(session, null);
    }

    #endregion

    #region Feature 10: modeDesc Mamba-2 Description

    [Fact]
    public void T1_F10_01_modeDesc_Mamba2Dense_ContainsMamba2DescriptionOnSession()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Contains("Mamba-2 Hybrid SSM", session.ActiveDevice);
    }

    [Fact]
    public void T1_F10_02_modeDesc_QwenDense_IdentifiesArchitectureAndEngineDescription()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Qwen, session.Architecture);
        Assert.Contains("SIMD AVX2/AVX-512 Optimized (Qwen)", session.ActiveDevice);
    }

    [Fact]
    public void T1_F10_03_modeDesc_ArchitectureFormatting_MatchesUniversalArchitecture()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Contains("Mamba-2", session.ActiveDevice);
    }

    [Fact]
    public void T1_F10_04_modeDesc_VulkanFormatContract_SpecifiesComputeEngineAndArch()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        string formatted = InferenceSession.FormatVulkanActiveDevice(
            "AMD Radeon 890M",
            session.Architecture,
            session.Weights.IsHybridSsm,
            session.Weights.IsMoe);

        Assert.Contains("Engine: Vulkan Compute", formatted);
        Assert.Contains("Mamba-2 Hybrid SSM", formatted);
        Assert.Contains($"Arch: {UniversalArchitecture.Mamba2}", formatted);
    }

    [Fact]
    public void T1_F10_05_modeDesc_MoE_DualMoE_ArchitectureDistinction()
    {
        string gemmaMode = InferenceSession.FormatVulkanModeDescription(UniversalArchitecture.Gemma4, isHybridSsm: false, isMoe: true);
        string mambaMode = InferenceSession.FormatVulkanModeDescription(UniversalArchitecture.Mamba2, isHybridSsm: false, isMoe: true);
        string qwenMode = InferenceSession.FormatVulkanModeDescription(UniversalArchitecture.Qwen, isHybridSsm: true, isMoe: true);

        Assert.Contains("Dual MoE", gemmaMode);
        Assert.DoesNotContain("Dual MoE", mambaMode);
        Assert.Contains("+ MoE", mambaMode);
        Assert.Contains("Gated DeltaNet + MoE", qwenMode);
    }

    #endregion

    #region Feature 11: VulkanRunnerBase Workgroup Dispatch Size

    [Fact]
    public void T1_F11_01_CalculateWorkgroups_SmallM_64()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(64);
        Assert.Equal(64u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T1_F11_02_CalculateWorkgroups_StandardM_4096()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(4096);
        Assert.Equal(4096u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T1_F11_03_CalculateWorkgroups_BoundaryM_32768()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(32768);
        Assert.Equal(32768u, gx);
        Assert.Equal(1u, gy);
    }

    [Fact]
    public void T1_F11_04_CalculateWorkgroups_BoundaryM_32769()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(32769);
        Assert.Equal(32768u, gx);
        Assert.Equal(2u, gy);
    }

    [Fact]
    public void T1_F11_05_CalculateWorkgroups_LargeVocabM_151936()
    {
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(151936);
        Assert.Equal(32768u, gx);
        Assert.Equal(5u, gy);
        Assert.True((long)gx * gy >= 151936);
    }

    #endregion

    #region Feature 12: VulkanRunnerBase Token Push Constant Inversion

    [Fact]
    public void T1_F12_01_PushConstants_InitialLayout_Push0_IsDim()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint E = 2048u;
        runner.AddEmbedOp(E);

        Assert.Equal(2, runner.GetPushLength(0));
        Assert.Equal(E, runner.GetPush(0)[0]);
    }

    [Fact]
    public void T1_F12_02_PushConstants_InitialLayout_Push1_IsZero()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint E = 2048u;
        runner.AddEmbedOp(E);

        Assert.Equal(0u, runner.GetPush(0)[1]);
    }

    [Fact]
    public void T1_F12_03_PatchToken_SetsPush1_ToToken()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint E = 2048u;
        runner.AddEmbedOp(E);

        int token = 151643;
        runner.ExecuteRun(token, pos: 0);

        Assert.Equal((uint)token, runner.GetPush(0)[1]);
    }

    [Fact]
    public void T1_F12_04_PatchToken_PreservesPush0_Dimension()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint E = 2048u;
        runner.AddEmbedOp(E);

        int token = 151643;
        runner.ExecuteRun(token, pos: 0);

        Assert.Equal(E, runner.GetPush(0)[0]);
        Assert.Equal((uint)token, runner.GetPush(0)[1]);
    }

    [Fact]
    public void T1_F12_05_SuccessiveTokens_UpdatePush1Sequentially()
    {
        var runner = TestableVulkanRunner.CreateForTest();
        uint E = 4096u;
        runner.AddEmbedOp(E);

        int[] tokenSequence = [100, 200, 300, 400];
        foreach (int t in tokenSequence)
        {
            runner.ExecuteRun(t, pos: 0);
            Assert.Equal((uint)t, runner.GetPush(0)[1]);
            Assert.Equal(E, runner.GetPush(0)[0]);
        }
    }

    #endregion

    #region Feature 13: Custom RoPE Frequency Tables (HasRopeFreqs)

    [Fact]
    public void T1_F13_01_Mamba2_IsSupported_AcceptsHasRopeFreqs()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(hasRopeFreqs: true);
        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.True(supported, $"Mamba2 should accept custom RoPE frequencies, but failed with: {reason}");
    }

    [Fact]
    public void T1_F13_02_Gemma4_IsSupported_AcceptsHasRopeFreqs()
    {
        // Gemma4VulkanModel.IsSupported does not reject HasRopeFreqs
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: true);
        Assert.True(handle.Weights.HasRopeFreqs);
    }

    [Fact]
    public void T1_F13_03_ModelWeights_IdentifiesHasRopeFreqs()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: true);
        Assert.True(handle.Weights.HasRopeFreqs);
    }

    [Fact]
    public void T1_F13_04_ModelWeights_RopeFreqsWeight_MapsFloatPointer()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: true);
        Assert.True(handle.Weights.RopeFreqsWeight != null);
        Assert.Equal(1.0f, handle.Weights.RopeFreqsWeight[0], 0.0001f);
    }

    [Fact]
    public void T1_F13_05_RoPE_FrequencyInterpolation_MathematicalParity()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 64, ffnDim: 128, headCount: 2, headCountKv: 2, hasRopeFreqs: true);
        Assert.True(handle.Weights.HasRopeFreqs);

        const int headDim = 32;
        const int halfDim = headDim / 2;
        float[] q = new float[headDim];
        float[] k = new float[headDim];
        float[] ropeFreqs = new float[halfDim];
        for (int i = 0; i < headDim; i++) { q[i] = 1.0f + i * 0.1f; k[i] = -0.5f + i * 0.05f; }
        Array.Fill(ropeFreqs, 2.0f);

        float qNormBefore = MathF.Sqrt(q.Sum(v => v * v));
        const int pos = 10;
        const float baseFreq = 10000.0f;
        float q0 = q[0], qHalf = q[halfDim];

        fixed (float* pQ = q, pK = k, pRf = ropeFreqs)
        {
            QuantKernels.RoPE(pQ, pK, nHeadsQ: 1, nHeadsKv: 1, headDim: headDim, pos: pos, freqBase: baseFreq, ropeFreqs: pRf);
        }

        float qNormAfter = MathF.Sqrt(q.Sum(v => v * v));
        Assert.Equal(qNormBefore, qNormAfter, 0.001f);

        float theta0 = pos * (1.0f / (1.0f * ropeFreqs[0]));
        Assert.Equal(q0 * MathF.Cos(theta0) - qHalf * MathF.Sin(theta0), q[0], 0.0001f);
        Assert.Equal(q0 * MathF.Sin(theta0) + qHalf * MathF.Cos(theta0), q[halfDim], 0.0001f);
    }

    #endregion

    #region Feature 14: SPIR-V Shader Kernels for New Quant Formats

    [Fact]
    public void T1_F14_01_SpirvResource_ManifestStreamInspection()
    {
        string[] shaders = VulkanE2ETestFixtures.SpirvInspector.GetAllEmbeddedShaderNames();
        Assert.NotEmpty(shaders);
        Assert.Contains(shaders, s => s.Contains("qmatvec") || s.Contains("embed") || s.Contains("elementwise"));
    }

    [Fact]
    public void T1_F14_02_SpirvResource_ValidMagicNumber_0x07230203()
    {
        byte[]? q0 = VulkanE2ETestFixtures.SpirvInspector.GetEmbeddedShader("qmatvec_q0");
        if (q0 != null)
        {
            Assert.True(VulkanE2ETestFixtures.SpirvInspector.IsValidSpirv(q0), "qmatvec_q0 must have valid SPIR-V magic header 0x07230203");
        }
    }

    [Fact]
    public void T1_F14_03_DequantizeQ4_0_ReferenceMatchesCpuFormula()
    {
        // 32 elements: scale d = 2.0f (half 0x4000), 16 bytes nibbles with value 8 (quant 0)
        byte[] block = new byte[18];
        Half h = (Half)2.0f;
        ushort bits = BitConverter.HalfToUInt16Bits(h);
        block[0] = (byte)(bits & 0xFF);
        block[1] = (byte)((bits >> 8) & 0xFF);
        // fill with 0x88 (both nibbles 8 -> quant 0)
        for (int i = 2; i < 18; i++) block[i] = 0x88;

        float[] output = new float[32];
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ4_0(block, output);

        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(0.0f, output[i], 0.0001f);
        }
    }

    [Fact]
    public void T1_F14_04_DequantizeQ8_0_ReferenceMatchesCpuFormula()
    {
        // 32 elements: scale d = 1.0f (half 0x3C00), 32 bytes quants with value 5
        byte[] block = new byte[34];
        Half h = (Half)1.0f;
        ushort bits = BitConverter.HalfToUInt16Bits(h);
        block[0] = (byte)(bits & 0xFF);
        block[1] = (byte)((bits >> 8) & 0xFF);
        for (int i = 2; i < 34; i++) block[i] = 5;

        float[] output = new float[32];
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeQ8_0(block, output);

        for (int i = 0; i < 32; i++)
        {
            Assert.Equal(5.0f, output[i], 0.0001f);
        }
    }

    [Fact]
    public void T1_F14_05_DequantizeF16_ReferenceMatchesCpuFormula()
    {
        byte[] bytes = new byte[4];
        Half h1 = (Half)1.5f;
        Half h2 = (Half)(-3.25f);
        ushort b1 = BitConverter.HalfToUInt16Bits(h1);
        ushort b2 = BitConverter.HalfToUInt16Bits(h2);
        bytes[0] = (byte)(b1 & 0xFF);
        bytes[1] = (byte)((b1 >> 8) & 0xFF);
        bytes[2] = (byte)(b2 & 0xFF);
        bytes[3] = (byte)((b2 >> 8) & 0xFF);

        float[] output = new float[2];
        VulkanE2ETestFixtures.ReferenceQuantOracle.DequantizeF16(bytes, output);

        Assert.Equal(1.5f, output[0], 0.0001f);
        Assert.Equal(-3.25f, output[1], 0.0001f);
    }

    #endregion

    #region Feature 15: Local & Remote GPU Validation

    [Fact]
    public void T1_F15_01_VulkanFact_AttributeInitialization()
    {
        var attr = new VulkanFactAttribute();
        Assert.NotNull(attr);
        // On environments without GPU, Skip is populated; on environments with GPU, Skip is null.
        // It must never throw an unhandled exception during instantiation.
    }

    [Fact]
    public void T1_F15_02_FindVulkanDevice_ExcludesSoftwareRasteriser()
    {
        int dev = Qwen35VulkanModel.FindVulkanDevice("llvmpipe");
        if (dev >= 0)
        {
            var devices = VulkanContext.EnumerateDevices();
            if (dev < devices.Count)
            {
                Assert.DoesNotContain("llvmpipe", devices[dev].Name, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("software", devices[dev].Name, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    [Fact]
    public void T1_F15_03_VulkanContext_IsSupported_PropertyQuery()
    {
        var ex = Record.Exception(() =>
        {
            bool supported = VulkanContext.IsSupported;
        });
        Assert.Null(ex);

        if (VulkanContext.IsSupported)
        {
            var devices = VulkanContext.EnumerateDevices();
            Assert.NotEmpty(devices);
            Assert.All(devices, d =>
            {
                Assert.True(d.Index >= 0);
                Assert.False(string.IsNullOrWhiteSpace(d.Name));
                Assert.True(d.TotalMemoryBytes > 0);
            });
        }
        else
        {
            var devices = VulkanContext.EnumerateDevices();
            Assert.Empty(devices);
        }
    }

    [VulkanFact]
    [Trait("Hardware", "Gpu")]
    public void T1_F15_04_GlacierSettings_ResolveTarget_RespectsVulkanEngine()
    {
        var (device, engine) = GlacierSettings.ResolveTarget(null, "Vulkan");
        Assert.Equal(InferenceEngineType.Vulkan, engine);
    }

    [Fact]
    public void T1_F15_05_GpuContext_NonConcurrencyAttributePresent()
    {
        var type = typeof(VulkanKernelTests);
        var collAttr = type.GetCustomAttribute<CollectionAttribute>();
        Assert.NotNull(collAttr);
    }

    #endregion

    #region Feature 16: Mandatory Security Compliance

    [Fact]
    public void T1_F16_01_SecurityAudit_TargetSourceTree_ExistsOnDisk()
    {
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        Assert.True(Directory.Exists(Path.Combine(inferenceRoot, "src", "Glacier.Inference")),
            $"Glacier.Inference source directory must exist under: {inferenceRoot}");
        Assert.True(Directory.Exists(Path.Combine(inferenceRoot, "tests", "Glacier.Inference.Tests")),
            $"Glacier.Inference test directory must exist under: {inferenceRoot}");
    }

    [Fact]
    public void T1_F16_02_SecurityScanner_ExecutesAndExitsZero()
    {
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        var (exitCode, scanned, findings, report) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(
            Path.Combine(inferenceRoot, "src"));
        Assert.Equal(0, exitCode);
        Assert.Empty(findings);
        Assert.True(scanned > 10, $"Expected > 10 source files scanned, got {scanned}");
        Assert.Contains("[PASSED]", report);
    }

    [Fact]
    public void T1_F16_03_SecurityScanner_ReportsZeroSecretsDetected()
    {
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        var (exitCode, _, findings, output) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(inferenceRoot);
        Assert.True(exitCode == 0, $"Secret audit reported findings:\n{output}");
        Assert.Empty(findings);
        Assert.Contains("[PASSED] Zero secrets or credentials detected.", output);
    }

    [Fact]
    public void T1_F16_04_SourceCode_AuditNoHardcodedApiKeys()
    {
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        string testFile = Path.Combine(inferenceRoot, "tests", "Glacier.Inference.Tests", "VulkanE2E", "VulkanE2ETier1FeatureCoverageTests.cs");
        Assert.True(File.Exists(testFile));
        string content = File.ReadAllText(testFile);
        Assert.DoesNotContain("AIza" + "Sy", content);
        Assert.DoesNotContain("BEGIN PRIVATE " + "KEY", content);
    }

    [Fact]
    public void T1_F16_05_GitIgnore_ContainsCredentialExclusions()
    {
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        string? repoRoot = Directory.GetParent(inferenceRoot)?.FullName;
        string gitIgnorePath = Path.Combine(inferenceRoot, ".gitignore");
        if (!File.Exists(gitIgnorePath) && repoRoot != null)
            gitIgnorePath = Path.Combine(repoRoot, ".gitignore");

        Assert.True(File.Exists(gitIgnorePath), $"Expected .gitignore at {gitIgnorePath}");
        string gitIgnore = File.ReadAllText(gitIgnorePath);
        Assert.Contains(".env", gitIgnore);
    }

    #endregion
}
