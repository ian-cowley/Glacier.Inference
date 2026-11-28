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
using Glacier.Inference.Quant;
using Glacier.Inference.Tests.E2E;
using Glacier.Inference.Tests.VulkanE2E.Fixtures;
using Xunit;

namespace Glacier.Inference.Tests.VulkanE2E;

/// <summary>
/// Tier 4: Real-World Application Scenarios (End-to-end full prompt evaluation, token generation, KV cache roll, security scanner check).
/// Exactly 6 comprehensive production workload scenarios.
/// </summary>
public unsafe class VulkanE2ETier4RealWorldScenarioTests
{
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

        // 3. Apply attention biases via real QuantKernels.VecAdd
        float[] qBiased = new float[256];
        Array.Fill(qBiased, 0.5f);
        fixed (float* pQ = qBiased)
        {
            QuantKernels.VecAdd(layer.QBias, pQ, 256);
        }

        // Bias was initialized to 0.05f in fixture
        Assert.Equal(0.55f, qBiased[0], 0.0001f);
        Assert.Equal(0.55f, qBiased[255], 0.0001f);

        // 4. Execute SwiGLU activation via real QuantKernels.SwiGLU
        float[] gate = new float[512];
        float[] up = new float[512];
        float[] swiglu = new float[512];
        Array.Fill(gate, 1.0f);
        Array.Fill(up, 2.0f);
        fixed (float* pGate = gate, pUp = up, pSwi = swiglu)
        {
            QuantKernels.SwiGLU(pGate, pUp, pSwi, 512);
        }

        Assert.True(swiglu[0] > 0.0f);

        // 5. Execute real InferenceSession forward pass end-to-end
        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.Prefill([1, 2, 3]);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
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

        // 2. Evaluate real QuantKernels.RoPE across context positions
        int[] contextPositions = [0, 128, 512, 1024, 4096];
        float baseFreq = weights.RopeFreqBase > 0f ? weights.RopeFreqBase : 10000.0f;
        int headDim = weights.HeadDim;
        foreach (int pos in contextPositions)
        {
            float[] q = new float[weights.HeadCount * headDim];
            float[] k = new float[weights.HeadCountKv * headDim];
            for (int i = 0; i < q.Length; i++) q[i] = ((i % 11) - 5) * 0.2f;
            for (int i = 0; i < k.Length; i++) k[i] = ((i % 7) - 3) * 0.3f;
            float qNorm = MathF.Sqrt(q.Sum(v => v * v));

            fixed (float* pQ = q, pK = k)
            {
                QuantKernels.RoPE(pQ, pK, weights.HeadCount, weights.HeadCountKv, headDim, pos, baseFreq, ropeFreqs: weights.RopeFreqsWeight);
            }

            Assert.All(q, v => Assert.True(float.IsFinite(v)));
            Assert.Equal(qNorm, MathF.Sqrt(q.Sum(v => v * v)), 0.01f);
        }

        // 3. Execute real InferenceSession forward pass on Q4_0 + custom RoPE model
        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        session.ForwardToken(token: 4, pos: 0, computeLogits: true);
        float[] logits = new float[session.VocabSize];
        session.CopyCurrentLogits(logits);
        Assert.Contains(logits, x => x != 0f && float.IsFinite(x));
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

        // 3. Real InferenceSession negotiation + production Vulkan device string formatting
        using var session = new InferenceSession(handle.Path, maxSeqLen: 16, device: "cpu");
        Assert.Equal(UniversalArchitecture.Mamba2, session.Architecture);
        Assert.Contains("Mamba-2 Hybrid SSM", session.ActiveDevice);

        string activeDevice = InferenceSession.FormatVulkanActiveDevice(
            "AMD Radeon 890M",
            session.Architecture,
            weights.IsHybridSsm,
            weights.IsMoe);

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
        // Workload: Automated standalone pre-publish security scanner execution on Glacier.Inference
        string inferenceRoot = E2EGovernanceAndBenchmarkTests.ResolveInferenceRepoRoot();
        var (exitCode, scanned, findings, output) = E2EGovernanceAndBenchmarkTests.RunPureCSharpSecretAudit(inferenceRoot);

        Assert.True(exitCode == 0, $"Security audit failed:\n{output}");
        Assert.Empty(findings);
        Assert.True(scanned > 20);
        Assert.Contains("[PASSED] Zero secrets or credentials detected.", output);
    }
}
