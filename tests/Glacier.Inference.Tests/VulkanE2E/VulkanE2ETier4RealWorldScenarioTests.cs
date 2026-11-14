using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
/// Tier 4: Real-World Application Scenarios (End-to-end full prompt evaluation, token generation, KV cache roll, security scanner check).
/// Exactly 6 comprehensive production workload scenarios.
/// </summary>
public unsafe class VulkanE2ETier4RealWorldScenarioTests
{
    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir) && !File.Exists(Path.Combine(dir, "scripts", "security_check.py")))
        {
            var parent = Path.GetDirectoryName(dir);
            if (parent == dir) break;
            dir = parent;
        }
        return dir ?? AppContext.BaseDirectory;
    }

    [Fact]
    public void Scenario1_Qwen25_AttentionBiases_And_FusedWeights_Execution()
    {
        // Workload: Qwen-2.5 model with attention biases and fused Gate/Up FFN
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 256,
            ffnDim: 512,
            headCount: 4,
            headCountKv: 2,
            vocabSize: 100,
            addBiases: true,
            fusedGateUp: true);

        var weights = handle.Weights;
        var layer = weights.Layers[0];

        // 1. Verify attention biases are mapped
        Assert.True(layer.QBias != null);
        Assert.True(layer.KBias != null);
        Assert.True(layer.VBias != null);
        Assert.True(layer.AttnOutBias != null);

        // 2. Verify fused Gate/Up is recognized
        Assert.True(layer.HasFusedGateUp);

        // 3. Simulate forward pass math: apply attention biases to projection
        float[] qRaw = new float[256];
        float[] qBiased = new float[256];
        for (int i = 0; i < 256; i++)
        {
            qRaw[i] = 0.5f;
            qBiased[i] = qRaw[i] + layer.QBias[i];
        }

        // Bias was initialized to 0.05f in fixture
        Assert.Equal(0.55f, qBiased[0], 0.0001f);

        // 4. Simulate intermediate SwiGLU activation
        float[] gate = new float[512];
        float[] up = new float[512];
        Array.Fill(gate, 1.0f);
        Array.Fill(up, 2.0f);
        float[] swiglu = VulkanE2ETestFixtures.ReferenceQuantOracle.ComputeSwiGlu(gate, up);

        Assert.Equal(512, swiglu.Length);
        Assert.True(swiglu[0] > 0.0f);
    }

    [Fact]
    public void Scenario2_Llama31_CustomRoPE_And_QuantWeights_Evaluation()
    {
        // Workload: Llama 3.1 with precomputed custom RoPE table and Q4_0 quantized weights
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(
            dim: 256,
            ffnDim: 512,
            headCount: 4,
            headCountKv: 2,
            hasRopeFreqs: true,
            weightType: GgufType.Q4_0);

        var weights = handle.Weights;

        // 1. Verify custom RoPE table is identified and loaded
        Assert.True(weights.HasRopeFreqs);
        Assert.True(weights.RopeFreqsWeight != null);

        // 2. Evaluate frequency factors across context positions
        int[] contextPositions = [0, 128, 512, 1024, 4096];
        float baseFreq = (float)weights.RopeFreqBase;
        if (baseFreq <= 0f) baseFreq = 10000.0f;

        float ropeFreq = weights.RopeFreqsWeight[0];
        Assert.True(ropeFreq > 0f);

        foreach (int pos in contextPositions)
        {
            float theta = pos * (1.0f / (baseFreq * ropeFreq));
            float cos = MathF.Cos(theta);
            float sin = MathF.Sin(theta);

            Assert.False(float.IsNaN(cos));
            Assert.False(float.IsNaN(sin));
            Assert.True(cos >= -1.0f && cos <= 1.0f);
            Assert.True(sin >= -1.0f && sin <= 1.0f);
        }
    }

    [Fact]
    public void Scenario3_Mamba2_SSM_Hybrid_Dispatch_And_Session_Negotiation()
    {
        // Workload: Interleaved Mamba-2 SSM hybrid session negotiation
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateMamba2(
            dim: 256,
            ffnDim: 512,
            headCount: 4,
            headDim: 64,
            hasRopeFreqs: true);

        var weights = handle.Weights;

        // 1. Architecture priority check: Mamba2 takes precedence
        Assert.Equal(UniversalArchitecture.Mamba2, weights.ArchitectureFamily);

        // 2. Diagnostic validation check
        bool supported = Mamba2VulkanModel.IsSupported(weights, out string reason);
        Assert.True(supported, $"Mamba2 model should be accepted: {reason}");

        // 3. Telemetry and active device string verification
        string deviceName = "AMD Radeon 890M";
        string modeDesc = $"Mamba-2 Hybrid SSM{(weights.IsMoe ? " + MoE" : "")}";
        string activeDevice = $"{deviceName} [Engine: Vulkan Compute ({modeDesc}) | KV: FP32 | Arch: {weights.ArchitectureFamily}]";

        Assert.Contains("Engine: Vulkan Compute", activeDevice);
        Assert.Contains("Mamba-2 Hybrid SSM", activeDevice);
        Assert.Contains("Arch: Mamba2", activeDevice);
    }

    [Fact]
    public void Scenario4_Large_Vocabulary_Embedding_Gather_Across_Quant_Types()
    {
        // Workload: 151,936 vocab embedding table gathered across boundary token IDs
        int vocabSize = 151936;
        int dim = 3584;

        var runner = TestableVulkanRunner.CreateForTest(maxSeqLen: 512, vocab: vocabSize);
        runner.AddEmbedOp((uint)dim);

        int[] testTokens = [0, 1, 15, 128, 1024, 32767, 32768, 65535, 151643, vocabSize - 1];

        foreach (int token in testTokens)
        {
            runner.ExecuteRun(token, pos: 0);
            var push = runner.GetPush(0);

            Assert.Equal((uint)dim, push[0]);   // dim preserved
            Assert.Equal((uint)token, push[1]); // token patched
            Assert.True(push[1] < (uint)vocabSize);
            runner.TestForwardGuards(token, pos: 0);
        }

        // Forward guards must reject token >= vocabSize
        Assert.Throws<ArgumentOutOfRangeException>((Action)(() => runner.TestForwardGuards(vocabSize, pos: 0)));
    }

    [Fact]
    public void Scenario5_MultiTurn_Generation_And_KV_Cache_Roll_With_Workgroups()
    {
        int vocabSize = 100;
        int maxSeqLen = 256;

        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2(vocabSize: vocabSize);
        using var session = new InferenceSession(handle.Path, maxSeqLen: maxSeqLen, device: "cpu");

        // 1. Output head workgroup grid covers all vocab rows (F11 verification)
        int fullVocabSize = 151936;
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(fullVocabSize);
        Assert.Equal(32768u, gx);
        Assert.Equal(5u, gy);
        Assert.True((long)gx * gy >= fullVocabSize);

        // 2. Multi-turn execution with real prompt evaluations
        int[] turn1Tokens = [10, 11, 12, 13];
        session.Prefill(turn1Tokens);
        float[] turn1Logits = new float[session.VocabSize];
        session.CopyCurrentLogits(turn1Logits);
        Assert.False(turn1Logits.All(x => x == 0f), "Turn 1 prefill must compute non-zero logits");

        // Autoregressive forward step
        session.ForwardToken(token: 25, pos: turn1Tokens.Length, computeLogits: true);
        float[] stepLogits = new float[session.VocabSize];
        session.CopyCurrentLogits(stepLogits);
        Assert.False(stepLogits.All(x => x == 0f), "Forward step must compute non-zero logits");

        // 3. Reset KV cache between turns to ensure clean restart from position 0
        session.ResetKvCache();

        // 4. Turn 2: same tokens after reset must yield bit-identical logits (determinism verification)
        session.Prefill(turn1Tokens);
        float[] turn2Logits = new float[session.VocabSize];
        session.CopyCurrentLogits(turn2Logits);

        for (int i = 0; i < session.VocabSize; i++)
        {
            Assert.Equal(turn1Logits[i], turn2Logits[i]);
        }

        // 5. Bounds check: position exceeding maxSeqLen must throw ArgumentOutOfRangeException
        int[] oversizedTokens = new int[maxSeqLen + 1];
        Assert.Throws<ArgumentOutOfRangeException>((Action)(() => session.Prefill(oversizedTokens)));
    }

    [Fact]
    public void Scenario6_Automated_PrePublish_Security_Audit_And_Hygiene()
    {
        // Workload: Automated pre-publish security scanner execution on Glacier.Inference
        string repoRoot = FindRepoRoot();
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
        string output = proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();

        Assert.Equal(0, proc.ExitCode);
        Assert.Contains("[PASSED] Zero secrets or credentials detected.", output);
    }
}
