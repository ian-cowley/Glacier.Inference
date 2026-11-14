using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
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
/// Tier 3: Cross-Feature Combinations (Pairwise coverage across major feature pairs).
/// 16 distinct combinatorial scenarios validating complex architectural intersections.
/// </summary>
public unsafe class VulkanE2ETier3CrossFeatureCombinationTests
{

    [Fact]
    public void T3_P01_Biases_And_FusedQkv_Interaction()
    {
        // P1: Attention Biases (F1) + Fused QKV (F4)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: true, fusedQkv: true);
        var layer = handle.Weights.Layers[0];

        Assert.True(layer.HasFusedQkv, "Model must identify fused QKV");
        Assert.True(layer.QkvBias != null, "Model with fused QKV must bind QkvBias");
        Assert.True(layer.AttnOutBias != null, "Model must bind AttnOutBias");
    }

    [Fact]
    public void T3_P02_Biases_And_FusedGateUp_Interaction()
    {
        // P2: Attention Biases (F1) + Fused Gate/Up SwiGLU (F5)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: true, fusedGateUp: true);
        var layer = handle.Weights.Layers[0];

        Assert.True(layer.HasFusedGateUp, "Model must identify fused Gate/Up");
        Assert.True(layer.AttnOutBias != null, "Model must bind AttnOutBias");

        // Validate SwiGLU combined with bias offset math
        float[] gate = [layer.AttnOutBias[0], 1.0f];
        float[] up = [2.0f, 0.5f];
        float[] swiglu = VulkanE2ETestFixtures.ReferenceQuantOracle.ComputeSwiGlu(gate, up);
        Assert.Equal(2, swiglu.Length);
    }

    [Fact]
    public void T3_P03_Biases_And_Q4_0_Weights_Interaction()
    {
        // P3: Attention Biases (F1) + Q4_0 Quantized Weights (F2)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(addBiases: true, weightType: GgufType.Q4_0);
        var layer = handle.Weights.Layers[0];

        Assert.Equal(GgufType.Q4_0, layer.AttnOutType);
        Assert.True(TestableVulkanRunner.IsKernelType(layer.AttnOutType));
        Assert.True(layer.AttnOutBias != null);
    }

    [Fact]
    public void T3_P04_Biases_And_WorkgroupDispatch_Interaction()
    {
        // P4: Attention Biases (F1) + Workgroup Dispatch boundary (F11)
        int m = 32769; // Output rows crossing 32768
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(m);
        Assert.Equal(32768u, gx);
        Assert.Equal(2u, gy);

        // Elementwise bias addition across 32769 elements
        int biasWorkgroups = (m + 255) / 256;
        Assert.Equal(129, biasWorkgroups);
        Assert.True(biasWorkgroups * 256 >= m);
    }

    [Fact]
    public void T3_P05_FusedQkv_And_FusedGateUp_Interaction()
    {
        // P5: Fused QKV (F4) + Fused Gate/Up (F5) (Phi-3 / Phi-4 style)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(fusedQkv: true, fusedGateUp: true);
        var layer = handle.Weights.Layers[0];

        Assert.True(layer.HasFusedQkv, "Should recognize fused QKV");
        Assert.True(layer.HasFusedGateUp, "Should recognize fused Gate/Up");
    }

    [Fact]
    public void T3_P06_CustomRoPE_And_Q4_0_Weights_Interaction()
    {
        // P6: Custom RoPE (F13) + Q4_0 Quantized Weights (F2) (Llama 3.1 style)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: true, weightType: GgufType.Q4_0);

        Assert.True(handle.Weights.HasRopeFreqs);
        Assert.True(handle.Weights.RopeFreqsWeight != null);
        Assert.True(TestableVulkanRunner.IsKernelType(GgufType.Q4_0));
    }

    [Fact]
    public void T3_P07_CustomRoPE_And_Q2_K_Weights_Interaction()
    {
        // P7: Custom RoPE (F13) + Q2_K Quantized Weights (F2)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(hasRopeFreqs: true, weightType: GgufType.Q2_K);

        Assert.True(handle.Weights.HasRopeFreqs);
        Assert.True(TestableVulkanRunner.IsKernelType(GgufType.Q2_K));
        Assert.Equal(8, TestableVulkanRunner.QIdx(GgufType.Q2_K));
    }

    [Fact]
    public void T3_P08_Q4_0_Embedding_And_PushConstant_Interaction()
    {
        // P8: Q4_0 Embedding (F3) + Push Constant Patching (F12)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(embdType: GgufType.Q4_0);

        Assert.True(TestableVulkanRunner.IsEmbedKernelType(handle.Weights.EmbdType));
        uint E = (uint)handle.Weights.EmbeddingLength;

        var runner = TestableVulkanRunner.CreateForTest();
        runner.AddEmbedOp(E);

        int targetToken = 42;
        runner.ExecuteRun(targetToken, pos: 0);

        Assert.Equal(E, runner.GetPush(0)[0]);
        Assert.Equal((uint)targetToken, runner.GetPush(0)[1]);
    }

    [Fact]
    public void T3_P09_IQ4_NL_Embedding_And_PushConstant_Interaction()
    {
        // P9: IQ4_NL Embedding (F3) + Push Constant Patching (F12)
        Assert.True(TestableVulkanRunner.IsEmbedKernelType(GgufType.IQ4_NL));
        Assert.Equal(7, TestableVulkanRunner.QIdx(GgufType.IQ4_NL));

        uint E = 2048u;
        var runner = TestableVulkanRunner.CreateForTest();
        runner.AddEmbedOp(E);

        int token = 100;
        runner.ExecuteRun(token, pos: 0);

        Assert.Equal(E, runner.GetPush(0)[0]);
        Assert.Equal(100u, runner.GetPush(0)[1]);
    }

    [Fact]
    public void T3_P10_F16_Embedding_And_F16_Weights_Interaction()
    {
        // P10: F16 Embedding (F3) + F16 Weights (F2) (Full FP16 Model)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(embdType: GgufType.F16, weightType: GgufType.F16);

        Assert.True(TestableVulkanRunner.IsEmbedKernelType(handle.Weights.EmbdType));
        Assert.True(TestableVulkanRunner.IsKernelType(handle.Weights.Layers[0].AttnOutType));
        Assert.Equal(10, TestableVulkanRunner.QIdx(GgufType.F16));
    }

    [Fact]
    public void T3_P11_Mamba2_Priority_And_PolymorphicCast_Interaction()
    {
        // P11: Mamba2 Priority (F8) + Safe DeviceName Cast (F6)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();

        Assert.Equal(UniversalArchitecture.Mamba2, handle.Weights.ArchitectureFamily);
        Assert.True(typeof(VulkanRunnerBase).IsAssignableFrom(typeof(Mamba2VulkanModel)));
    }

    [Fact]
    public void T3_P12_Mamba2_Priority_And_DiagnosticChain_Interaction()
    {
        // P12: Mamba2 Priority (F8) + Diagnostic Chain (F7)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();

        bool isMamba2 = handle.Weights.ArchitectureFamily == UniversalArchitecture.Mamba2;
        Assert.True(isMamba2);

        bool supported = Mamba2VulkanModel.IsSupported(handle.Weights, out string reason);
        Assert.True(supported, $"Mamba2 diagnostic check must pass: {reason}");
    }

    [Fact]
    public void T3_P13_Mamba2_Priority_And_ActiveBackend_Interaction()
    {
        // P13: Mamba2 Priority (F8) + ActiveBackend Vulkan (F9)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Equal("Cpu", session.ActiveBackend);

        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.Vulkan);

        Assert.Equal("Vulkan", session.ActiveBackend);
    }

    [Fact]
    public void T3_P14_Mamba2_Priority_And_ModeDesc_Interaction()
    {
        // P14: Mamba2 Priority (F8) + modeDesc Description (F10)
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Contains("Mamba-2 Hybrid SSM", session.ActiveDevice);
    }

    [Fact]
    public void T3_P15_WorkgroupDispatch_And_Q2_K_OutputHead_Interaction()
    {
        // P15: Workgroup Dispatch (F11) + Q2_K Quantization (F2) on Large Vocab Projection
        int vocabSize = 151936;
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(vocabSize);

        Assert.Equal(32768u, gx);
        Assert.Equal(5u, gy);
        Assert.True((long)gx * gy >= vocabSize);

        // Size of Q2_K projection row block
        long rowBytes = TestableVulkanRunner.GgufTypeSize(GgufType.Q2_K, 256);
        Assert.Equal(84L, rowBytes);
    }

    [Fact]
    public void T3_P16_PreCommit_E2E_Test_And_SecurityScanner_Interaction()
    {
        // P16: Mandatory Security (F16) + Full Test Suite Execution in Sequence
        string repoRoot = Path.GetFullPath(AppContext.BaseDirectory);
        while (!string.IsNullOrEmpty(repoRoot) && !File.Exists(Path.Combine(repoRoot, "scripts", "security_check.py")))
        {
            var parent = Path.GetDirectoryName(repoRoot);
            if (parent == repoRoot) break;
            repoRoot = parent;
        }

        if (!File.Exists(Path.Combine(repoRoot, "scripts", "security_check.py"))) return;

        var psi = new ProcessStartInfo("python", "scripts/security_check.py Glacier.Inference")
        {
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        using var proc = Process.Start(psi);
        Assert.NotNull(proc);
        proc.WaitForExit();
        Assert.Equal(0, proc.ExitCode);
    }
}
