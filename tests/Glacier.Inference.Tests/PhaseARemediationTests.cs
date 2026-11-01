namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Audio;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Image.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Vortice.Direct3D12;
using Xunit;

/// <summary>
/// Phase A (v1.3.2) deterministic regression suite verifying:
/// 1. Multi-turn state reset parity across Qwen3HybridModel (SSM), Gemma4Model (ISWA), and InferenceSession.
/// 2. KV cache / position bounds validation and sliding-window ring-buffer overflow prevention.
/// 3. CUDA head_dim generalization (64, 256) and non-contiguous PagedAttention (single + batch, symmetric + asymmetric V dim).
/// 4. Direct3D 12 HLSL attention shader head_dim generalization (64, 256) and bounds guards.
/// 5. Exact Leviathan et al. non-greedy speculative rejection sampling, recurrent SSM rollback, and split 4-byte UTF-8 streaming.
/// 6. Batched CPU prefill LoRA parity (standard + fused Gate/Up), WAV multi-chunk parsing, and fail-fast stub guards.
/// </summary>
[Collection("SequentialGpu")]
public class PhaseARemediationTests
{
    [Fact]
    public void Qwen3HybridModel_ResetState_RestoresExactMultiTurnParity()
    {
        string ggufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen35HybridGguf(ggufPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);

            using var gguf = GgufFile.Open(ggufPath);
            var weights = new ModelWeights(gguf);
            using var model = new Qwen3HybridModel(weights, maxSeqLen: 32);
            using var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 32);

            float[] turn1Pos0 = new float[weights.VocabSize];
            float[] turn1Pos2 = new float[weights.VocabSize];
            float[] pollutedPos0 = new float[weights.VocabSize];
            float[] turn2Pos0 = new float[weights.VocabSize];
            float[] turn2Pos2 = new float[weights.VocabSize];

            // Turn 1: advance through positions 0, 1, 2
            model.Forward(10, 0, kvCache, turn1Pos0, computeLogits: true);
            model.Forward(11, 1, kvCache, Span<float>.Empty, computeLogits: false);
            model.Forward(12, 2, kvCache, turn1Pos2, computeLogits: true);

            // Resetting only KV cache without ResetState() leaves _ssmCache polluted -> pos=0 diverges
            kvCache.Reset();
            model.Forward(10, 0, kvCache, pollutedPos0, computeLogits: true);
            Assert.True(MaxAbsDiff(turn1Pos0, pollutedPos0) > 1e-5f,
                "Expected polluted SSM state to alter Turn 2 logits when ResetState() is omitted.");

            // Full reset (KV cache + recurrent SSM state) must restore bit-identical multi-turn outputs
            kvCache.Reset();
            model.ResetState();
            model.Forward(10, 0, kvCache, turn2Pos0, computeLogits: true);
            model.Forward(11, 1, kvCache, Span<float>.Empty, computeLogits: false);
            model.Forward(12, 2, kvCache, turn2Pos2, computeLogits: true);

            Assert.Equal(0f, MaxAbsDiff(turn1Pos0, turn2Pos0));
            Assert.Equal(0f, MaxAbsDiff(turn1Pos2, turn2Pos2));
        }
        finally
        {
            DeleteIfExists(ggufPath);
        }

        // Also verify InferenceSession.ResetKvCache() routes through _cpuModel.ResetState()
        string sessionGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen35HybridGguf(sessionGgufPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);
            using var session = new InferenceSession(sessionGgufPath, maxSeqLen: 32, device: "cpu");

            float[] sTurn1 = new float[session.VocabSize];
            float[] sTurn2 = new float[session.VocabSize];

            session.Prefill(new[] { 5, 6, 7 });
            session.CopyCurrentLogits(sTurn1);

            session.ResetKvCache();
            session.Prefill(new[] { 5, 6, 7 });
            session.CopyCurrentLogits(sTurn2);

            Assert.Equal(0f, MaxAbsDiff(sTurn1, sTurn2));
        }
        finally
        {
            DeleteIfExists(sessionGgufPath);
        }
    }

    [Fact]
    public unsafe void Gemma4Model_ResetState_ClearsInternalGemmaKvCacheAndRestoresParity()
    {
        string ggufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticGemma4Gguf(ggufPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 2, vocabSize: 288);

            using var gguf = GgufFile.Open(ggufPath);
            var weights = new ModelWeights(gguf);
            using var model = new Gemma4Model(weights, maxSeqLen: 16);
            using var dummyKv = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 16);

            float[] turn1 = new float[weights.VocabSize];
            float[] turn2 = new float[weights.VocabSize];

            model.Forward(15, 0, dummyKv, Span<float>.Empty, computeLogits: false);
            model.Forward(16, 1, dummyKv, turn1, computeLogits: true);

            // Confirm internal Gemma4KvCache holds non-zero K/V state after Turn 1
            float nonZeroSum = 0f;
            float* kPtr = model.GemmaCache.GetKeyPtr(0, 0, 1);
            for (int d = 0; d < 256; d++) nonZeroSum += MathF.Abs(kPtr[d]);
            Assert.True(nonZeroSum > 0f);

            // ResetState() must zero internal Gemma4KvCache and restore exact Turn 1 parity
            model.ResetState();
            float afterResetSum = 0f;
            for (int d = 0; d < 256; d++) afterResetSum += MathF.Abs(kPtr[d]);
            Assert.Equal(0f, afterResetSum);

            model.Forward(15, 0, dummyKv, Span<float>.Empty, computeLogits: false);
            model.Forward(16, 1, dummyKv, turn2, computeLogits: true);
            Assert.Equal(0f, MaxAbsDiff(turn1, turn2));
        }
        finally
        {
            DeleteIfExists(ggufPath);
        }
    }

    [Fact]
    public unsafe void KvCacheAndModels_OutOfBoundsPosition_ThrowsArgumentOutOfRangeException()
    {
        using (var gemmaKv = new Gemma4KvCache(layers: 1, maxSeqLen: 8, swaPattern: new[] { true }, headsKvPattern: new[] { 2 }, defaultHeadDim: 256))
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => { _ = (IntPtr)gemmaKv.GetKeyPtr(0, 0, -1); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { _ = (IntPtr)gemmaKv.GetKeyPtr(0, 0, 8); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { _ = (IntPtr)gemmaKv.GetValPtr(0, 0, -1); });
            Assert.Throws<ArgumentOutOfRangeException>(() => { _ = (IntPtr)gemmaKv.GetValPtr(0, 0, 8); });
        }

        string qwenPath = CreateTempPath(".gguf");
        string hybridPath = CreateTempPath(".gguf");
        string gemmaPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(qwenPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            WriteSyntheticQwen35HybridGguf(hybridPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);
            WriteSyntheticGemma4Gguf(gemmaPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 2, vocabSize: 288);

            using (var gguf = GgufFile.Open(qwenPath))
            {
                var w = new ModelWeights(gguf);
                using var qwen = new Qwen2Model(w, maxSeqLen: 8);
                using var kv = new KVCache(w.BlockCount, w.HeadCountKv, w.HeadDim, maxSeqLen: 8, isRingBuffer: false);
                float[] logits = new float[w.VocabSize];

                Assert.Throws<ArgumentOutOfRangeException>(() => qwen.Forward(1, -1, kv, logits));
                Assert.Throws<ArgumentOutOfRangeException>(() => qwen.Forward(1, 8, kv, logits));
                Assert.Throws<ArgumentOutOfRangeException>(() => qwen.ForwardBatch(new[] { 1, 2, 3 }, startPos: 7, logits, kv));
            }

            using (var gguf = GgufFile.Open(hybridPath))
            {
                var w = new ModelWeights(gguf);
                using var hybrid = new Qwen3HybridModel(w, maxSeqLen: 8);
                using var kv = new KVCache(w.BlockCount, w.HeadCountKv, w.HeadDim, maxSeqLen: 8, isRingBuffer: false);
                float[] logits = new float[w.VocabSize];

                Assert.Throws<ArgumentOutOfRangeException>(() => hybrid.Forward(1, -1, kv, logits));
                Assert.Throws<ArgumentOutOfRangeException>(() => hybrid.Forward(1, 8, kv, logits));
                Assert.Throws<ArgumentOutOfRangeException>(() => hybrid.ForwardBatch(new[] { 1, 2 }, startPos: 7, logits, kv));
            }

            using (var gguf = GgufFile.Open(gemmaPath))
            {
                var w = new ModelWeights(gguf);
                using var gemma = new Gemma4Model(w, maxSeqLen: 8);
                using var kv = new KVCache(w.BlockCount, w.HeadCountKv, w.HeadDim, maxSeqLen: 8);
                float[] logits = new float[w.VocabSize];

                Assert.Throws<ArgumentOutOfRangeException>(() => gemma.Forward(1, -1, kv, logits));
                Assert.Throws<ArgumentOutOfRangeException>(() => gemma.Forward(1, 8, kv, logits));
            }

            using (var session = new InferenceSession(qwenPath, maxSeqLen: 8, device: "cpu"))
            {
                int[] preds = new int[3];
                Assert.Throws<ArgumentOutOfRangeException>(() => session.ForwardToken(1, -1, computeLogits: false));
                Assert.Throws<ArgumentOutOfRangeException>(() => session.ForwardToken(1, 8, computeLogits: false));
                Assert.Throws<ArgumentOutOfRangeException>(() => session.VerifyBatch(new[] { 1, 2, 3 }, startPos: 7, preds));
            }
        }
        finally
        {
            DeleteIfExists(qwenPath);
            DeleteIfExists(hybridPath);
            DeleteIfExists(gemmaPath);
        }
    }

    [Fact]
    public void RingBufferSlidingWindow_ExceedingMaxSeqLen_DoesNotOverflowScores()
    {
        string qwenPath = CreateTempPath(".gguf");
        string hybridPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(qwenPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            WriteSyntheticQwen35HybridGguf(hybridPath, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);

            using (var gguf = GgufFile.Open(qwenPath))
            {
                var w = new ModelWeights(gguf);
                using var qwen = new Qwen2Model(w, maxSeqLen: 8);
                using var ringKv = new KVCache(w.BlockCount, w.HeadCountKv, w.HeadDim, maxSeqLen: 8, isRingBuffer: true);
                float[] logits = new float[w.VocabSize];

                for (int pos = 0; pos < 25; pos++)
                {
                    qwen.Forward((pos % 20) + 1, pos, ringKv, logits, computeLogits: pos == 24);
                }

                for (int i = 0; i < logits.Length; i++)
                {
                    Assert.True(float.IsFinite(logits[i]), $"Non-finite logit at index {i} after ring-buffer wraparound.");
                }
            }

            using (var gguf = GgufFile.Open(hybridPath))
            {
                var w = new ModelWeights(gguf);
                using var hybrid = new Qwen3HybridModel(w, maxSeqLen: 8);
                using var ringKv = new KVCache(w.BlockCount, w.HeadCountKv, w.HeadDim, maxSeqLen: 8, isRingBuffer: true);
                float[] logits = new float[w.VocabSize];

                for (int pos = 0; pos < 25; pos++)
                {
                    hybrid.Forward((pos % 20) + 1, pos, ringKv, logits, computeLogits: pos == 24);
                }

                for (int i = 0; i < logits.Length; i++)
                {
                    Assert.True(float.IsFinite(logits[i]), $"Non-finite hybrid logit at index {i} after ring-buffer wraparound.");
                }
            }
        }
        finally
        {
            DeleteIfExists(qwenPath);
            DeleteIfExists(hybridPath);
        }
    }

    [CudaFact(false)]
    public void CudaAttentionGqa_HeadDim64And256_MatchesCpuOracle()
    {
        using var gpu = new GpuContext(0);
        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnF32, module, "attention_gqa_kernel"), "attention_gqa_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreF16, module, "kv_cache_store_f16"), "kv_cache_store_f16");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnF16, module, "attention_gqa_f16"), "attention_gqa_f16");

            foreach (int headDim in new[] { 64, 256 })
            {
                VerifyCudaContiguousAttention(gpu, fnAttnF32, fnStoreF16, fnAttnF16, nHeadsQ: 4, nHeadsKv: 2, headDim: headDim, maxSeqLen: 16, pos: 10);
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [CudaFact(false)]
    public void CudaPagedAttention_SingleAndBatch_NonContiguousBlocks_MatchesCpuOracle()
    {
        using var gpu = new GpuContext(0);
        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreSingle, module, "paged_kv_cache_store_f16_kernel"), "paged_kv_cache_store_f16_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreBatch, module, "paged_kv_cache_store_batch_f16_kernel"), "paged_kv_cache_store_batch_f16_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnSingle, module, "paged_attention_gqa_f16_kernel"), "paged_attention_gqa_f16_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnBatch, module, "paged_attention_gqa_batch_f16_kernel"), "paged_attention_gqa_batch_f16_kernel");

            // Test symmetric (headDim=128, vHeadDim=128) and asymmetric MLA-style (headDim=256, vHeadDim=128)
            VerifyCudaPagedAttention(gpu, fnStoreSingle, fnStoreBatch, fnAttnSingle, fnAttnBatch, headDim: 128, vHeadDim: 128);
            VerifyCudaPagedAttention(gpu, fnStoreSingle, fnStoreBatch, fnAttnSingle, fnAttnBatch, headDim: 256, vHeadDim: 128);
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    public void D3D12AttentionShaders_HeadDim64And256_MatchesCpuOracle()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        RunD3D12AttentionShadersCore();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunD3D12AttentionShadersCore()
    {
        if (!D3D12Context.IsSupported)
        {
            return;
        }

        using var ctx = new D3D12Context();
        foreach (int headDim in new[] { 64, 256 })
        {
            VerifyD3D12AttentionGqa(ctx, headDim, useFp16: false);
            VerifyD3D12AttentionGqa(ctx, headDim, useFp16: true);
            VerifyD3D12AttentionBatch(ctx, headDim);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_NonGreedyRejectionSampling_MatchesTargetDistribution()
    {
        // Printable ASCII byte tokens in GPT-2 byte vocab: 'A'=65 (prefix), 'B'=66, 'C'=67, 'D'=68.
        // Target distribution p on {'B','C','D'} = [0.10, 0.70, 0.20]
        // Skewed draft distribution q on {'B','C','D'} = [0.80, 0.10, 0.10]
        // Under exact Leviathan et al. rejection + residual sampling:
        //   P(accept 'B') = 0.80 * min(1, 0.10/0.80) = 0.10
        //   P(accept 'C') = 0.10 * min(1, 0.70/0.10) = 0.10
        //   P(accept 'D') = 0.10 * min(1, 0.20/0.10) = 0.10
        //   P(reject)     = 0.70; residual norm(max(0, p - q)) -> P('C'|rej)=6/7, P('D'|rej)=1/7
        //   Exact marginal: P('B')=0.10, P('C')=0.70, P('D')=0.20!
        const int vocabSize = 96;
        float[] targetProbs = new float[vocabSize];
        targetProbs['B'] = 0.10f;
        targetProbs['C'] = 0.70f;
        targetProbs['D'] = 0.20f;

        float[] draftProbs = new float[vocabSize];
        draftProbs['B'] = 0.80f;
        draftProbs['C'] = 0.10f;
        draftProbs['D'] = 0.10f;

        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: vocabSize, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            int countB = 0, countC = 0, countD = 0;
            const int trials = 1200;

            for (int t = 0; t < trials; t++)
            {
                using var target = new FixedDistributionSpeculativeTarget(tokenizer, targetProbs, firstToken: 'A', seed: 10_000 + t);
                using var draft = new FixedDistributionDraftProvider(draftProbs, fallbackToken: 'B');
                using var engine = new SpeculativeEngine(target, draft, seed: 50_000 + t);

                var opts = new SpeculativeOptions
                {
                    MaxTokens = 2, // Token 0 = 'A' from prefill, Token 1 = speculative draft+verify step
                    MaxDraftTokens = 1,
                    Sampling = new SamplingOptions
                    {
                        Temperature = 1.0f,
                        TopK = 0,
                        TopP = 1.0f,
                        RepetitionPenalty = 1.0f
                    }
                };

                var res = await engine.GenerateAsync("A", opts, formatChat: false);
                Assert.Equal(1, res.SpeculativeMetrics.DraftTokensProposed);
                Assert.Equal(2, res.Text.Length);
                Assert.Equal('A', res.Text[0]);

                switch (res.Text[1])
                {
                    case 'B': countB++; break;
                    case 'C': countC++; break;
                    case 'D': countD++; break;
                    default: Assert.Fail($"Unexpected sampled character '{res.Text[1]}' outside target support."); break;
                }
            }

            double freqB = countB / (double)trials;
            double freqC = countC / (double)trials;
            double freqD = countD / (double)trials;

            Assert.InRange(freqB, 0.10 - 0.05, 0.10 + 0.05);
            Assert.InRange(freqC, 0.70 - 0.055, 0.70 + 0.055);
            Assert.InRange(freqD, 0.20 - 0.05, 0.20 + 0.05);
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_RejectedDrafts_RollsBackRecurrentSsmState()
    {
        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 32, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);
            string promptText = tokenizer.Decode(new[] { 1 });

            // Draft proposes [4, 9, 10], while target deterministically predicts:
            // prefill([1]) -> firstToken = 3
            // verifyBatch([3, 4, 9, 10] at basePos=1):
            //   pos 1 (tok 3) -> 4 (accepts draft[0]=4!)
            //   pos 2 (tok 4) -> 5 (rejects draft[1]=9! bonus token = 5)
            //   pos 3 (tok 9) -> 6 (rejected lookahead that polluted recurrent state during VerifyBatch!)
            //   pos 4 (tok 10)-> 7 (rejected lookahead that polluted recurrent state during VerifyBatch!)
            // After rollback + replay of [3 at pos 1, 4 at pos 2], recurrent state MUST match serial execution of [1@0, 3@1, 4@2]!
            using var target = new StatefulRecurrentMockTarget(tokenizer);
            using var draft = new FixedSequenceDraftProvider(new[] { 4, 9, 10 });
            using var engine = new SpeculativeEngine(target, draft, seed: 42);

            var opts = new SpeculativeOptions
            {
                MaxTokens = 3, // firstToken(3), acceptedDraft(4), bonus(5) -> done!
                MaxDraftTokens = 3,
                Sampling = new SamplingOptions { Temperature = 0.0f, TopK = 1 }
            };

            var result = await engine.GenerateAsync(promptText, opts, formatChat: false);
            Assert.Equal(3, result.SpeculativeMetrics.DraftTokensProposed);
            Assert.Equal(1, result.SpeculativeMetrics.DraftTokensAccepted);
            Assert.True(target.RestoreCallCount >= 1, "Expected RestoreRecurrentState to be invoked when drafts were rejected.");

            // Compute expected recurrent state for clean serial sequence: prompt 1@0, pending 3@1, accepted 4@2
            long expectedState = 0;
            expectedState = StatefulRecurrentMockTarget.StepState(expectedState, token: 1, pos: 0);
            expectedState = StatefulRecurrentMockTarget.StepState(expectedState, token: 3, pos: 1);
            expectedState = StatefulRecurrentMockTarget.StepState(expectedState, token: 4, pos: 2);

            Assert.Equal(expectedState, target.RecurrentState);
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_SplitFourByteUtf8Emoji_StreamsWithoutReplacementChars()
    {
        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            // Rocket emoji "🚀" = UTF-8 [0xF0, 0x9F, 0x9A, 0x80] = 4 single-byte GPT-2 BPE tokens
            int[] emojiTokens = tokenizer.Encode("🚀");
            Assert.Equal(4, emojiTokens.Length);
            Assert.Single(tokenizer.GetTokenBytes(emojiTokens[0]));
            Assert.Equal(0xF0, tokenizer.GetTokenBytes(emojiTokens[0])[0]);
            Assert.Equal(0x9F, tokenizer.GetTokenBytes(emojiTokens[1])[0]);
            Assert.Equal(0x9A, tokenizer.GetTokenBytes(emojiTokens[2])[0]);
            Assert.Equal(0x80, tokenizer.GetTokenBytes(emojiTokens[3])[0]);

            string promptText = "A"; // single ASCII byte token

            // Stream 2 rockets ("🚀🚀" = 8 byte-tokens): first via accepted drafts, second via serial fallback
            int[] targetSequence =
            {
                emojiTokens[0], emojiTokens[1], emojiTokens[2], emojiTokens[3],
                emojiTokens[0], emojiTokens[1], emojiTokens[2], emojiTokens[3]
            };

            using var target = new ScriptedSequenceTarget(tokenizer, targetSequence);
            using var draft = new FirstStepOnlyDraftProvider(new[] { emojiTokens[1], emojiTokens[2], emojiTokens[3] });
            using var engine = new SpeculativeEngine(target, draft, seed: 7);

            var streamedChunks = new List<string>();
            var opts = new SpeculativeOptions
            {
                MaxTokens = 8,
                MaxDraftTokens = 3,
                Sampling = new SamplingOptions { Temperature = 0.0f, TopK = 1 }
            };

            var res = await engine.GenerateAsync(promptText, opts, formatChat: false, onToken: chunk => streamedChunks.Add(chunk));

            string concatenated = string.Concat(streamedChunks);
            Assert.Equal("🚀🚀", res.Text);
            Assert.Equal("🚀🚀", concatenated);
            Assert.DoesNotContain('\uFFFD', concatenated);
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public void CpuBatchedPrefill_WithAttachedLora_MatchesSequentialForward_StandardAndFusedGateUp()
    {
        foreach (bool fusedGateUp in new[] { false, true })
        {
            string ggufPath = CreateTempPath(".gguf");
            string loraPath = CreateTempPath(".lora");
            try
            {
                const int dim = 64;
                const int ffnDim = 128;
                const int headCount = 2;
                const int headCountKv = 1;
                const int vocabSize = 288;

                WriteSyntheticQwen2Gguf(ggufPath, dim, ffnDim, headCount, headCountKv, vocabSize, fusedGateUp);
                WriteSyntheticLoraFile(loraPath, layerCount: 1, rank: 4, alpha: 8.0f, dim: dim, qDim: 64, kvDim: 32, ffnDim: ffnDim);

                var lora = LoraAdapterWeights.Load(loraPath);
                int[] prompt = { 3, 7, 11, 19, 23 };

                using var gguf = GgufFile.Open(ggufPath);
                var weights = new ModelWeights(gguf);

                float[] baselineBatchLogits = new float[vocabSize];
                float[] loraSequentialLogits = new float[vocabSize];
                float[] loraBatchLogits = new float[vocabSize];

                // 1. Baseline without LoRA
                using (var baseModel = new Qwen2Model(weights, maxSeqLen: 32))
                using (var baseKv = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 32))
                {
                    baseModel.ForwardBatch(prompt, startPos: 0, baselineBatchLogits, baseKv);
                }

                // 2. Sequential Forward with LoRA
                using (var seqModel = new Qwen2Model(weights, maxSeqLen: 32))
                using (var seqKv = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 32))
                {
                    seqModel.LoraWeights = lora;
                    for (int i = 0; i < prompt.Length; i++)
                    {
                        bool last = i == prompt.Length - 1;
                        seqModel.Forward(prompt[i], i, seqKv, last ? loraSequentialLogits : Span<float>.Empty, computeLogits: last);
                    }
                }

                // 3. Batched ForwardBatch with LoRA
                using (var batchModel = new Qwen2Model(weights, maxSeqLen: 32))
                using (var batchKv = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 32))
                {
                    batchModel.LoraWeights = lora;
                    batchModel.ForwardBatch(prompt, startPos: 0, loraBatchLogits, batchKv);
                }

                // Confirm LoRA materially shifted logits from baseline AND batch matches sequential
                Assert.True(MaxAbsDiff(baselineBatchLogits, loraBatchLogits) > 1e-3f,
                    $"Expected attached LoRA to shift batched logits (fusedGateUp={fusedGateUp}).");
                Assert.True(MaxAbsDiff(loraSequentialLogits, loraBatchLogits) < 1e-4f,
                    $"Expected batched LoRA logits to match sequential Forward (fusedGateUp={fusedGateUp}, maxDiff={MaxAbsDiff(loraSequentialLogits, loraBatchLogits)}).");
            }
            finally
            {
                DeleteIfExists(ggufPath);
                DeleteIfExists(loraPath);
            }
        }
    }

    [Fact]
    public void WavWriter_ReadMonoFloat_SkipsIntermediateRiffChunksAccurately()
    {
        float[] expected = { -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f };
        byte[] wavWithExtraChunks = BuildWavWithExtraChunks(expected, sampleRate: 22050);

        float[] parsed = WavWriter.ReadMonoFloat(wavWithExtraChunks, out int parsedSampleRate);
        Assert.Equal(22050, parsedSampleRate);
        Assert.Equal(expected.Length, parsed.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - parsed[i]) < 1e-3f,
                $"WAV sample mismatch at {i}: expected {expected[i]}, got {parsed[i]}");
        }
    }

    [Fact]
    public void DiffusionGgufPipeline_UnsupportedArchAndMissingCompanions_FailFast()
    {
        string sd3Path = CreateTempPath(".gguf");
        string fluxPath = CreateTempPath(".gguf");
        string missingCompanion = CreateTempPath(".missing");
        try
        {
            WriteMinimalDiffusionGguf(sd3Path, "sd3");
            WriteMinimalDiffusionGguf(fluxPath, "flux");

            using (var sd3Model = DiffusionGgufModel.Open(sd3Path))
            using (var sd3Pipe = new DiffusionGgufPipeline(sd3Model, vaeSafetensorsPath: missingCompanion, clipSafetensorsPath: missingCompanion, t5GgufPath: missingCompanion))
            {
                Assert.Throws<NotSupportedException>(() => sd3Pipe.Generate("a test prompt", width: 64, height: 64, numSteps: 1));
            }

            using (var fluxModel = DiffusionGgufModel.Open(fluxPath))
            using (var fluxPipe = new DiffusionGgufPipeline(fluxModel, vaeSafetensorsPath: missingCompanion, clipSafetensorsPath: missingCompanion, t5GgufPath: missingCompanion))
            {
                Assert.Throws<FileNotFoundException>(() => fluxPipe.Generate("a test prompt", width: 64, height: 64, numSteps: 1));
            }
        }
        finally
        {
            DeleteIfExists(sd3Path);
            DeleteIfExists(fluxPath);
        }
    }

    // =========================================================================
    // Follow-Up Adversarial Review Regression Suite
    // =========================================================================

    [CudaFact(false)]
    public void CudaPagedAttention_NegativePhysBlock_SkipsUnmappedBlocksSafely()
    {
        using var gpu = new GpuContext(0);
        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreSingle, module, "paged_kv_cache_store_f16_kernel"), "paged_kv_cache_store_f16_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnSingle, module, "paged_attention_gqa_f16_kernel"), "paged_attention_gqa_f16_kernel");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnBatch, module, "paged_attention_gqa_batch_f16_kernel"), "paged_attention_gqa_batch_f16_kernel");

            VerifyCudaPagedAttentionNegativePhysBlock(gpu, fnStoreSingle, fnAttnSingle, fnAttnBatch);
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [CudaFact(false)]
    public async Task CudaGpuContext_MakeCurrent_SurvivesAsyncThreadHop()
    {
        using var gpu = new GpuContext(0);
        float[] expected = { 1.25f, -3.5f, 7.75f, 42.0f, -0.125f, 99.5f };
        IntPtr dBuf = gpu.AllocateDevice((nuint)(expected.Length * sizeof(float)));
        try
        {
            var tcs = new TaskCompletionSource<float[]>(TaskCreationOptions.RunContinuationsAsynchronously);
            var bgThread = new System.Threading.Thread(() =>
            {
                try
                {
                    // Fresh OS thread has NULL CUDA TLS context until MakeCurrent() binds it!
                    gpu.MakeCurrent();
                    CopyFloatsToCuda(gpu, dBuf, expected);
                    gpu.Synchronize();
                    tcs.SetResult(CopyFloatsFromCuda(gpu, dBuf, expected.Length));
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
            })
            {
                IsBackground = true
            };
            bgThread.Start();

            float[] actual = await tcs.Task;
            Assert.Equal(expected, actual);
        }
        finally
        {
            gpu.MakeCurrent();
            gpu.FreeDevice(dBuf);
        }
    }

    [Fact]
    public void D3D12_RmsNormHeads_HeadDim256_MatchesCpuOracle()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        RunD3D12RmsNormHeadsCore();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunD3D12RmsNormHeadsCore()
    {
        if (!D3D12Context.IsSupported)
        {
            return;
        }

        using var ctx = new D3D12Context();
        const int nHeads = 4;
        const int headDim = 256;
        const float eps = 1e-5f;

        var rng = new Random(2026);
        float[] x = RandomFloats(rng, nHeads * headDim, -2.0f, 2.0f);
        float[] weight = RandomFloats(rng, headDim, 0.5f, 1.5f);
        float[] expected = new float[x.Length];

        for (int h = 0; h < nHeads; h++)
        {
            int offset = h * headDim;
            float sumSq = 0f;
            for (int d = 0; d < headDim; d++)
            {
                sumSq += x[offset + d] * x[offset + d];
            }
            float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + eps);
            for (int d = 0; d < headDim; d++)
            {
                expected[offset + d] = x[offset + d] * invRms * weight[d];
            }
        }

        using var dX = ctx.CreateDeviceBuffer((ulong)(x.Length * sizeof(float)));
        using var dW = ctx.CreateDeviceBuffer((ulong)(weight.Length * sizeof(float)));
        ctx.CopyToDevice(dX, MemoryMarshal.AsBytes(x.AsSpan()));
        ctx.CopyToDevice(dW, MemoryMarshal.AsBytes(weight.AsSpan()));

        ctx.BeginCommands();
        ctx.DispatchRmsNormHeads(dX, dW, nHeads, headDim, eps);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[x.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actual.AsSpan()), dX, (ulong)(actual.Length * sizeof(float)));
        Assert.True(MaxAbsDiff(expected, actual) < 1e-4f,
            $"D3D12 RmsNormHeads (headDim=256) mismatch: maxDiff={MaxAbsDiff(expected, actual)}");
    }

    [Fact]
    public void D3D12_MoeRouterSoftmaxTopK_NumExperts300_MatchesCpuOracle()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        RunD3D12MoeRouterSoftmaxTopKCore();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunD3D12MoeRouterSoftmaxTopKCore()
    {
        if (!D3D12Context.IsSupported)
        {
            return;
        }

        using var ctx = new D3D12Context();
        const int numExperts = 300; // > 256: exercises strided 512-entry LDS router!
        const int topK = 4;
        const int batchSize = 2;

        var rng = new Random(3001);
        float[] logits = RandomFloats(rng, batchSize * numExperts, -1.0f, 1.0f);
        // Guarantee winning experts include indices >= 256
        logits[0 * numExperts + 295] = 8.5f;
        logits[0 * numExperts + 280] = 7.2f;
        logits[0 * numExperts + 150] = 6.1f;
        logits[0 * numExperts + 12] = 5.4f;

        logits[1 * numExperts + 299] = 9.0f;
        logits[1 * numExperts + 260] = 7.8f;
        logits[1 * numExperts + 256] = 6.5f;
        logits[1 * numExperts + 5] = 5.0f;

        using var dLogits = ctx.CreateDeviceBuffer((ulong)(logits.Length * sizeof(float)));
        using var dIndices = ctx.CreateDeviceBuffer((ulong)(batchSize * topK * sizeof(uint)));
        using var dWeights = ctx.CreateDeviceBuffer((ulong)(batchSize * topK * sizeof(float)));
        ctx.CopyToDevice(dLogits, MemoryMarshal.AsBytes(logits.AsSpan()));

        ctx.BeginCommands();
        ctx.DispatchMoeRouterSoftmaxTopK(dLogits, dIndices, dWeights, numExperts, topK, normTopK: true, batchSize: batchSize);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        uint[] actualIndices = new uint[batchSize * topK];
        float[] actualWeights = new float[batchSize * topK];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actualIndices.AsSpan()), dIndices, (ulong)(actualIndices.Length * sizeof(uint)));
        ctx.CopyToHost(MemoryMarshal.AsBytes(actualWeights.AsSpan()), dWeights, (ulong)(actualWeights.Length * sizeof(float)));

        Assert.Equal(new uint[] { 295, 280, 150, 12, 299, 260, 256, 5 }, actualIndices);

        for (int b = 0; b < batchSize; b++)
        {
            float maxL = float.NegativeInfinity;
            for (int e = 0; e < numExperts; e++)
            {
                if (logits[b * numExperts + e] > maxL) maxL = logits[b * numExperts + e];
            }
            float sumExp = 0f;
            float[] probs = new float[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                probs[e] = MathF.Exp(logits[b * numExperts + e] - maxL);
                sumExp += probs[e];
            }
            for (int e = 0; e < numExperts; e++) probs[e] /= sumExp;

            float topSum = 0f;
            for (int k = 0; k < topK; k++) topSum += probs[actualIndices[b * topK + k]];

            for (int k = 0; k < topK; k++)
            {
                float expectedW = probs[actualIndices[b * topK + k]] / topSum;
                Assert.True(MathF.Abs(expectedW - actualWeights[b * topK + k]) < 1e-4f,
                    $"MoE weight mismatch at b={b}, k={k}: expected {expectedW}, got {actualWeights[b * topK + k]}");
            }
        }
    }

    [Fact]
    public void D3D12_RepetitionPenalty_DeduplicatesAndIgnoresOutOfBoundsTokens()
    {
        if (!OperatingSystem.IsWindows())
        {
            return;
        }

        RunD3D12RepetitionPenaltyCore();
    }

    [SupportedOSPlatform("windows")]
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunD3D12RepetitionPenaltyCore()
    {
        if (!D3D12Context.IsSupported)
        {
            return;
        }

        using var ctx = new D3D12Context();
        float[] initialLogits =
        {
            1.0f, -1.0f, 4.0f, 6.0f, -2.0f, -3.0f, 0.5f, 2.5f,
            -0.5f, 1.5f, 3.0f, -4.0f, 0.25f, -0.25f, 5.0f, -5.0f
        };

        using var dLogits = ctx.CreateDeviceBuffer((ulong)(initialLogits.Length * sizeof(float)));
        ctx.CopyToDevice(dLogits, MemoryMarshal.AsBytes(initialLogits.AsSpan()));

        // Duplicate tokens 2 (x3) and 5 (x2), plus out-of-bounds tokens -5 and 999999
        int[] recent = { -5, 2, 2, 2, 5, 5, 999999 };
        ctx.BeginCommands();
        ctx.DispatchRepetitionPenalty(dLogits, vocabSize: initialLogits.Length, recentTokens: recent, penalty: 2.0f);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[initialLogits.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actual.AsSpan()), dLogits, (ulong)(actual.Length * sizeof(float)));

        // Positive logit 4.0f penalized once -> 2.0f (NOT 4.0f / 8.0f = 0.5f!)
        Assert.Equal(2.0f, actual[2], precision: 4);
        // Negative logit -3.0f penalized once -> -6.0f (NOT -3.0f * 4.0f = -12.0f!)
        Assert.Equal(-6.0f, actual[5], precision: 4);
        // Untouched tokens remain exact
        Assert.Equal(6.0f, actual[3], precision: 4);
        Assert.Equal(1.0f, actual[0], precision: 4);

        // Also exercise raw HLSL shader vocab_size guard directly with un-filtered negative & huge indices
        VerifyRawD3D12RepetitionPenaltyShaderBoundsGuard(ctx);
    }

    [Fact]
    public void ModelDraftProvider_ExactPrefixTracking_PreventsCollisionAndAvoidsRedundantPrefill()
    {
        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            // 1. Cross-prompt collision test: [10, 20] + 1 drafted token (99) => cached=[10, 20, 99].
            // Next call passes [30, 40, 99] (same length=3 and same final token=99, but divergent prefix!).
            using (var target = new PrefixTrackingMockTarget(tokenizer, canOverwriteKvFromPrefix: true, nextTokenSequence: new[] { 99, 77, 88 }))
            using (var provider = new ModelDraftProvider(target, ownsSession: false))
            {
                Span<int> drafts = stackalloc int[4];
                int n1 = provider.Draft(new[] { 10, 20 }, maxDraftTokens: 1, drafts);
                Assert.Equal(1, n1);
                Assert.Equal(99, drafts[0]);
                Assert.Equal(1, target.PrefillCount);

                int n2 = provider.Draft(new[] { 30, 40, 99 }, maxDraftTokens: 1, drafts);
                Assert.Equal(1, n2);
                Assert.Equal(2, target.PrefillCount); // Must re-prefill, NOT collide!
                Assert.Equal(new[] { 30, 40, 99 }, target.LastPrefilledTokens);
            }

            // 2. 100% draft acceptance + bonus token AND partial acceptance on causal overwriteable KV cache => ZERO re-prefills!
            using (var target = new PrefixTrackingMockTarget(tokenizer, canOverwriteKvFromPrefix: true, nextTokenSequence: new[] { 5, 6, 11, 12, 21, 22 }))
            using (var provider = new ModelDraftProvider(target, ownsSession: false))
            {
                Span<int> drafts = stackalloc int[4];

                // Step A: initial prompt [10, 20], drafts [5, 6] -> cached=[10, 20, 5, 6]
                int nA = provider.Draft(new[] { 10, 20 }, maxDraftTokens: 2, drafts);
                Assert.Equal(2, nA);
                Assert.Equal(5, drafts[0]);
                Assert.Equal(6, drafts[1]);
                Assert.Equal(1, target.PrefillCount);

                target.ForwardedSteps.Clear();

                // Step B: 100% accepted + bonus token 9 -> context=[10, 20, 5, 6, 9]
                // Must NOT re-prefill; must forward bonus token 9 at pos=4, then draft [11, 12]
                int nB = provider.Draft(new[] { 10, 20, 5, 6, 9 }, maxDraftTokens: 2, drafts);
                Assert.Equal(2, nB);
                Assert.Equal(11, drafts[0]);
                Assert.Equal(12, drafts[1]);
                Assert.Equal(1, target.PrefillCount);
                Assert.Equal((Token: 9, Pos: 4), target.ForwardedSteps[0]);

                target.ForwardedSteps.Clear();

                // Step C: partial acceptance (11 accepted, 12 rejected & replaced by 19) -> context=[10, 20, 5, 6, 9, 11, 19]
                // Because CanOverwriteKvFromPrefix == true, must overwrite from pos=6 without re-prefilling!
                int nC = provider.Draft(new[] { 10, 20, 5, 6, 9, 11, 19 }, maxDraftTokens: 2, drafts);
                Assert.Equal(2, nC);
                Assert.Equal(21, drafts[0]);
                Assert.Equal(22, drafts[1]);
                Assert.Equal(1, target.PrefillCount);
                Assert.Equal((Token: 19, Pos: 6), target.ForwardedSteps[0]);

                // Step D: Reset() clears cached prefix and resets session KV cache
                int resetsBefore = target.ResetCount;
                provider.Reset();
                Assert.Equal(resetsBefore + 1, target.ResetCount);
            }
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_MaxTokensZero_EmitsNothing()
    {
        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            using var target = new PrefixTrackingMockTarget(tokenizer, canOverwriteKvFromPrefix: true, nextTokenSequence: new[] { 65, 66 });
            using var draft = new FixedSequenceDraftProvider(new[] { 66 });
            using var engine = new SpeculativeEngine(target, draft, seed: 1);

            var chunks = new List<string>();
            var res = await engine.GenerateAsync(
                "Hello",
                new SpeculativeOptions { MaxTokens = 0, MaxDraftTokens = 4 },
                formatChat: false,
                onToken: c => chunks.Add(c));

            Assert.Equal(string.Empty, res.Text);
            Assert.Equal(0, res.Metrics.GeneratedTokens);
            Assert.Equal("length", res.FinishReason);
            Assert.Empty(chunks);
            Assert.Equal(0, target.PrefillCount);
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_ZeroDraftProb_SamplesExactTargetProbabilityWithoutDoubleCounting()
    {
        // Target distribution p on {'B'=0.40, 'C'=0.60}.
        // Degenerate draft provider proposes 'B' while reporting q = all zeros (e.g. underflow / OOV fallback).
        // Accepting when qCandidate <= 0 && pCandidate > 0 would output 'B' 100% of the time instead of 40%!
        const int vocabSize = 96;
        float[] targetProbs = new float[vocabSize];
        targetProbs['B'] = 0.40f;
        targetProbs['C'] = 0.60f;

        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            using var target = new FixedDistributionSpeculativeTarget(tokenizer, targetProbs, firstToken: 'A', seed: 123);
            using var draft = new ZeroProbabilityCandidateDraftProvider(candidateToken: 'B');
            using var engine = new SpeculativeEngine(target, draft, seed: 98765);

            var opts = new SpeculativeOptions
            {
                MaxTokens = 2,
                MaxDraftTokens = 1,
                Sampling = new SamplingOptions { Temperature = 1.0f, TopK = 0, TopP = 1.0f }
            };

            const int trials = 800;
            int countB = 0;
            int countC = 0;
            for (int t = 0; t < trials; t++)
            {
                var res = await engine.GenerateAsync("X", opts, formatChat: false);
                Assert.Equal(2, res.Text.Length);
                if (res.Text[1] == 'B') countB++;
                else if (res.Text[1] == 'C') countC++;
            }

            Assert.Equal(trials, countB + countC);
            Assert.InRange(countB / (double)trials, 0.34, 0.46);
            Assert.InRange(countC / (double)trials, 0.54, 0.66);
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_UnsnapshottableRecurrentOrRingBufferTarget_DisablesDrafting()
    {
        string tokGgufPath = CreateTempPath(".gguf");
        try
        {
            WriteSyntheticQwen2Gguf(tokGgufPath, dim: 32, ffnDim: 64, headCount: 1, headCountKv: 1, vocabSize: 288, fusedGateUp: false);
            using var gguf = GgufFile.Open(tokGgufPath);
            var tokenizer = new BpeTokenizer(gguf);

            foreach ((bool ringBuffer, bool unsnapshottable) in new[] { (true, false), (false, true) })
            {
                using var target = new UnsnapshottableOrRingMockTarget(tokenizer, ringBuffer, unsnapshottable, new[] { (int)'B', 'C', 'D', 'E' });
                using var draft = new CountingDraftProvider(new[] { (int)'Z', 'Z' });
                using var engine = new SpeculativeEngine(target, draft, seed: 42);

                var res = await engine.GenerateAsync(
                    "A",
                    new SpeculativeOptions
                    {
                        MaxTokens = 4,
                        MaxDraftTokens = 3,
                        Sampling = new SamplingOptions { Temperature = 0.0f, TopK = 1 }
                    },
                    formatChat: false);

                Assert.Equal("BCDE", res.Text);
                Assert.Equal(0, res.SpeculativeMetrics.DraftTokensProposed);
                Assert.Equal(0, draft.CallCount);
            }
        }
        finally
        {
            DeleteIfExists(tokGgufPath);
        }
    }

    [Fact]
    public async Task CliVisionQuery_WithoutSyntheticDemoFlag_ReturnsNonZeroWithActionableMessage()
    {
        string rawImgPath = CreateTempPath(".raw");
        var previousLogger = GlacierDiagnostics.Logger;
        try
        {
            File.WriteAllBytes(rawImgPath, new byte[448 * 448 * 3]);

            // 1. Without --synthetic-demo: must return exit code 1 with actionable message
            int exitWithoutFlag = await Glacier.Inference.Cli.Program.Main(new[] { "vision", "query", rawImgPath, "What is in this image?" });
            Assert.Equal(1, exitWithoutFlag);

            // 2. With --synthetic-demo: succeeds cleanly (exit code 0) without NotSupportedException
            int exitWithFlag = await Glacier.Inference.Cli.Program.Main(new[] { "vision", "query", rawImgPath, "What is in this image?", "--synthetic-demo" });
            Assert.Equal(0, exitWithFlag);
        }
        finally
        {
            GlacierDiagnostics.Logger = previousLogger;
            DeleteIfExists(rawImgPath);
        }
    }

    // =========================================================================
    // GPU / D3D12 Verification Helpers
    // =========================================================================

    private static unsafe void VerifyCudaContiguousAttention(
        GpuContext gpu,
        IntPtr fnAttnF32,
        IntPtr fnStoreF16,
        IntPtr fnAttnF16,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        int maxSeqLen,
        int pos)
    {
        var rng = new Random(1234 + headDim);
        float attnScale = 1.0f / MathF.Sqrt(headDim);
        uint blockSize = (uint)Math.Min(256, Math.Max(32, ((headDim + 31) / 32) * 32));

        float[] q = RandomFloats(rng, nHeadsQ * headDim, -0.5f, 0.5f);
        float[] kCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);
        float[] vCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);

        float[] expectedF32 = ComputeCpuGqaReference(q, kCache, vCache, nHeadsQ, nHeadsKv, headDim, headDim, maxSeqLen, pos, attnScale, quantizeKvFp16: false);
        float[] expectedF16 = ComputeCpuGqaReference(q, kCache, vCache, nHeadsQ, nHeadsKv, headDim, headDim, maxSeqLen, pos, attnScale, quantizeKvFp16: true);

        IntPtr dQ = gpu.AllocateDevice((nuint)(q.Length * sizeof(float)));
        IntPtr dK32 = gpu.AllocateDevice((nuint)(kCache.Length * sizeof(float)));
        IntPtr dV32 = gpu.AllocateDevice((nuint)(vCache.Length * sizeof(float)));
        IntPtr dK16 = gpu.AllocateDevice((nuint)(kCache.Length * sizeof(ushort)));
        IntPtr dV16 = gpu.AllocateDevice((nuint)(vCache.Length * sizeof(ushort)));
        IntPtr dStepK = gpu.AllocateDevice((nuint)(nHeadsKv * headDim * sizeof(float)));
        IntPtr dStepV = gpu.AllocateDevice((nuint)(nHeadsKv * headDim * sizeof(float)));
        IntPtr dOut = gpu.AllocateDevice((nuint)(nHeadsQ * headDim * sizeof(float)));

        try
        {
            CopyFloatsToCuda(gpu, dQ, q);
            CopyFloatsToCuda(gpu, dK32, kCache);
            CopyFloatsToCuda(gpu, dV32, vCache);

            // 1. FP32 attention_gqa_kernel
            IntPtr dScores = IntPtr.Zero;
            void** args32 = stackalloc void*[11];
            args32[0] = &dQ;
            args32[1] = &dK32;
            args32[2] = &dV32;
            args32[3] = &dOut;
            args32[4] = &dScores;
            args32[5] = &nHeadsQ;
            args32[6] = &nHeadsKv;
            args32[7] = &headDim;
            args32[8] = &maxSeqLen;
            args32[9] = &pos;
            args32[10] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(fnAttnF32, (uint)nHeadsQ, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)args32, IntPtr.Zero), "attention_gqa_kernel");
            gpu.Synchronize();

            float[] actualF32 = CopyFloatsFromCuda(gpu, dOut, nHeadsQ * headDim);
            Assert.True(MaxAbsDiff(expectedF32, actualF32) < 1e-4f,
                $"CUDA FP32 attention mismatch for headDim={headDim}: maxDiff={MaxAbsDiff(expectedF32, actualF32)}");

            // 2. Store FP16 KV cache via kv_cache_store_f16 and run attention_gqa_f16
            float[] stepK = new float[nHeadsKv * headDim];
            float[] stepV = new float[nHeadsKv * headDim];
            int tPos = 0;
            void** storeArgs = stackalloc void*[8];
            storeArgs[0] = &dK16;
            storeArgs[1] = &dV16;
            storeArgs[2] = &dStepK;
            storeArgs[3] = &dStepV;
            storeArgs[4] = &nHeadsKv;
            storeArgs[5] = &headDim;
            storeArgs[6] = &maxSeqLen;
            storeArgs[7] = &tPos;

            int totalKv = nHeadsKv * headDim;
            uint storeGrid = (uint)((totalKv + 255) / 256);

            for (int t = 0; t <= pos; t++)
            {
                for (int hKv = 0; hKv < nHeadsKv; hKv++)
                {
                    int srcBase = (hKv * maxSeqLen + t) * headDim;
                    Array.Copy(kCache, srcBase, stepK, hKv * headDim, headDim);
                    Array.Copy(vCache, srcBase, stepV, hKv * headDim, headDim);
                }
                CopyFloatsToCuda(gpu, dStepK, stepK);
                CopyFloatsToCuda(gpu, dStepV, stepV);

                tPos = t;
                CuDriver.Check(CuDriver.LaunchKernel(fnStoreF16, storeGrid, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeArgs, IntPtr.Zero), "kv_cache_store_f16");
            }

            void** args16 = stackalloc void*[11];
            args16[0] = &dQ;
            args16[1] = &dK16;
            args16[2] = &dV16;
            args16[3] = &dOut;
            args16[4] = &dScores;
            args16[5] = &nHeadsQ;
            args16[6] = &nHeadsKv;
            args16[7] = &headDim;
            args16[8] = &maxSeqLen;
            args16[9] = &pos;
            args16[10] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(fnAttnF16, (uint)nHeadsQ, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)args16, IntPtr.Zero), "attention_gqa_f16");
            gpu.Synchronize();

            float[] actualF16 = CopyFloatsFromCuda(gpu, dOut, nHeadsQ * headDim);
            Assert.True(MaxAbsDiff(expectedF16, actualF16) < 2e-3f,
                $"CUDA FP16 attention mismatch for headDim={headDim}: maxDiff={MaxAbsDiff(expectedF16, actualF16)}");
        }
        finally
        {
            gpu.FreeDevice(dQ);
            gpu.FreeDevice(dK32);
            gpu.FreeDevice(dV32);
            gpu.FreeDevice(dK16);
            gpu.FreeDevice(dV16);
            gpu.FreeDevice(dStepK);
            gpu.FreeDevice(dStepV);
            gpu.FreeDevice(dOut);
        }
    }

    private static unsafe void VerifyCudaPagedAttention(
        GpuContext gpu,
        IntPtr fnStoreSingle,
        IntPtr fnStoreBatch,
        IntPtr fnAttnSingle,
        IntPtr fnAttnBatch,
        int headDim,
        int vHeadDim)
    {
        var rng = new Random(777 + headDim * 3 + vHeadDim);
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int kvBlockSize = 8;
        const int numPhysicalBlocks = 10;
        const int numLayers = 2;
        const int modelLayer = 1;
        const int prefillLen = 14; // spans logical blocks 0 (8 tokens) and 1 (6 tokens)
        const int decodePos = 14;  // logical block 1, offset 6
        const int totalSeq = 19;   // batch verify of 4 tokens at startPos=15 -> positions 15, 16, 17, 18 (crosses into logical block 2!)
        int[] blockTable = { 5, 2, 7 }; // deliberately non-contiguous physical blocks!

        float attnScale = 1.0f / MathF.Sqrt(headDim);
        uint attnThreads = (uint)Math.Min(256, Math.Max(32, ((Math.Max(headDim, vHeadDim) + 31) / 32) * 32));

        float[] allK = RandomFloats(rng, totalSeq * nHeadsKv * headDim, -0.5f, 0.5f);
        float[] allV = RandomFloats(rng, totalSeq * nHeadsKv * vHeadDim, -0.5f, 0.5f);
        float[] decodeQ = RandomFloats(rng, nHeadsQ * headDim, -0.5f, 0.5f);
        const int verifyBatchSize = 4;
        const int verifyStartPos = 15;
        float[] batchQ = RandomFloats(rng, verifyBatchSize * nHeadsQ * headDim, -0.5f, 0.5f);

        // Build contiguous [nHeadsKv, totalSeq, dim] views for the CPU reference oracle
        float[] kCacheRef = new float[nHeadsKv * totalSeq * headDim];
        float[] vCacheRef = new float[nHeadsKv * totalSeq * vHeadDim];
        for (int t = 0; t < totalSeq; t++)
        {
            for (int h = 0; h < nHeadsKv; h++)
            {
                Array.Copy(allK, (t * nHeadsKv + h) * headDim, kCacheRef, (h * totalSeq + t) * headDim, headDim);
                Array.Copy(allV, (t * nHeadsKv + h) * vHeadDim, vCacheRef, (h * totalSeq + t) * vHeadDim, vHeadDim);
            }
        }

        IntPtr dKPool = gpu.AllocateDevice((nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * headDim * sizeof(ushort)));
        IntPtr dVPool = gpu.AllocateDevice((nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * vHeadDim * sizeof(ushort)));
        IntPtr dBlockTable = gpu.AllocateDevice((nuint)(blockTable.Length * sizeof(int)));
        IntPtr dKBatch = gpu.AllocateDevice((nuint)(totalSeq * nHeadsKv * headDim * sizeof(float)));
        IntPtr dVBatch = gpu.AllocateDevice((nuint)(totalSeq * nHeadsKv * vHeadDim * sizeof(float)));
        IntPtr dQSingle = gpu.AllocateDevice((nuint)(decodeQ.Length * sizeof(float)));
        IntPtr dOutSingle = gpu.AllocateDevice((nuint)(nHeadsQ * vHeadDim * sizeof(float)));
        IntPtr dQBatch = gpu.AllocateDevice((nuint)(batchQ.Length * sizeof(float)));
        IntPtr dOutBatch = gpu.AllocateDevice((nuint)(verifyBatchSize * nHeadsQ * vHeadDim * sizeof(float)));

        try
        {
            CuDriver.Check(CuDriver.MemsetD8(dKPool, 0, (nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * headDim * sizeof(ushort))), "MemsetD8(dKPool)");
            CuDriver.Check(CuDriver.MemsetD8(dVPool, 0, (nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * vHeadDim * sizeof(ushort))), "MemsetD8(dVPool)");
            CopyIntsToCuda(gpu, dBlockTable, blockTable);

            // 1. Store prefill [0..13] via paged_kv_cache_store_batch_f16_kernel
            float[] prefillK = new float[prefillLen * nHeadsKv * headDim];
            float[] prefillV = new float[prefillLen * nHeadsKv * vHeadDim];
            Array.Copy(allK, 0, prefillK, 0, prefillK.Length);
            Array.Copy(allV, 0, prefillV, 0, prefillV.Length);
            CopyFloatsToCuda(gpu, dKBatch, prefillK);
            CopyFloatsToCuda(gpu, dVBatch, prefillV);

            int startZero = 0;
            int batchLen = prefillLen;
            int layer = modelLayer;
            int layers = numLayers;
            int nKv = nHeadsKv;
            int hDim = headDim;
            int vDim = vHeadDim;
            int blkSize = kvBlockSize;
            int maxDim = Math.Max(headDim, vHeadDim);
            uint storeBatchGrid = (uint)((prefillLen * nHeadsKv * maxDim + 255) / 256);

            void** storeBatchArgs = stackalloc void*[13];
            storeBatchArgs[0] = &dKBatch;
            storeBatchArgs[1] = &dVBatch;
            storeBatchArgs[2] = &dKPool;
            storeBatchArgs[3] = &dVPool;
            storeBatchArgs[4] = &dBlockTable;
            storeBatchArgs[5] = &startZero;
            storeBatchArgs[6] = &batchLen;
            storeBatchArgs[7] = &layer;
            storeBatchArgs[8] = &layers;
            storeBatchArgs[9] = &nKv;
            storeBatchArgs[10] = &hDim;
            storeBatchArgs[11] = &vDim;
            storeBatchArgs[12] = &blkSize;
            CuDriver.Check(CuDriver.LaunchKernel(fnStoreBatch, storeBatchGrid, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeBatchArgs, IntPtr.Zero), "paged_kv_cache_store_batch_f16_kernel");

            // 2. Store single decode token at pos=14 via paged_kv_cache_store_f16_kernel
            float[] singleK = new float[nHeadsKv * headDim];
            float[] singleV = new float[nHeadsKv * vHeadDim];
            Array.Copy(allK, decodePos * nHeadsKv * headDim, singleK, 0, singleK.Length);
            Array.Copy(allV, decodePos * nHeadsKv * vHeadDim, singleV, 0, singleV.Length);
            CopyFloatsToCuda(gpu, dKBatch, singleK);
            CopyFloatsToCuda(gpu, dVBatch, singleV);

            int singlePos = decodePos;
            uint storeSingleGrid = (uint)((nHeadsKv * maxDim + 255) / 256);
            void** storeSingleArgs = stackalloc void*[12];
            storeSingleArgs[0] = &dKBatch;
            storeSingleArgs[1] = &dVBatch;
            storeSingleArgs[2] = &dKPool;
            storeSingleArgs[3] = &dVPool;
            storeSingleArgs[4] = &dBlockTable;
            storeSingleArgs[5] = &singlePos;
            storeSingleArgs[6] = &layer;
            storeSingleArgs[7] = &layers;
            storeSingleArgs[8] = &nKv;
            storeSingleArgs[9] = &hDim;
            storeSingleArgs[10] = &vDim;
            storeSingleArgs[11] = &blkSize;
            CuDriver.Check(CuDriver.LaunchKernel(fnStoreSingle, storeSingleGrid, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeSingleArgs, IntPtr.Zero), "paged_kv_cache_store_f16_kernel");

            // 3. Verify single-token paged_attention_gqa_f16_kernel at pos=14
            CopyFloatsToCuda(gpu, dQSingle, decodeQ);
            int nQ = nHeadsQ;
            void** attnSingleArgs = stackalloc void*[14];
            attnSingleArgs[0] = &dQSingle;
            attnSingleArgs[1] = &dKPool;
            attnSingleArgs[2] = &dVPool;
            attnSingleArgs[3] = &dOutSingle;
            attnSingleArgs[4] = &dBlockTable;
            attnSingleArgs[5] = &singlePos;
            attnSingleArgs[6] = &layer;
            attnSingleArgs[7] = &layers;
            attnSingleArgs[8] = &nQ;
            attnSingleArgs[9] = &nKv;
            attnSingleArgs[10] = &hDim;
            attnSingleArgs[11] = &vDim;
            attnSingleArgs[12] = &blkSize;
            attnSingleArgs[13] = &attnScale;
            CuDriver.Check(CuDriver.LaunchKernel(fnAttnSingle, (uint)nHeadsQ, 1, 1, attnThreads, 1, 1, 0, IntPtr.Zero, (IntPtr)attnSingleArgs, IntPtr.Zero), "paged_attention_gqa_f16_kernel");
            gpu.Synchronize();

            float[] expectedSingle = ComputeCpuGqaReference(decodeQ, kCacheRef, vCacheRef, nHeadsQ, nHeadsKv, headDim, vHeadDim, totalSeq, decodePos, attnScale, quantizeKvFp16: true);
            float[] actualSingle = CopyFloatsFromCuda(gpu, dOutSingle, nHeadsQ * vHeadDim);
            Assert.True(MaxAbsDiff(expectedSingle, actualSingle) < 2e-3f,
                $"CUDA PagedAttention single mismatch (headDim={headDim}, vHeadDim={vHeadDim}): maxDiff={MaxAbsDiff(expectedSingle, actualSingle)}");

            // 4. Store batch [15..18] (crossing into physical block 7) and verify paged_attention_gqa_batch_f16_kernel
            float[] verifyK = new float[verifyBatchSize * nHeadsKv * headDim];
            float[] verifyV = new float[verifyBatchSize * nHeadsKv * vHeadDim];
            Array.Copy(allK, verifyStartPos * nHeadsKv * headDim, verifyK, 0, verifyK.Length);
            Array.Copy(allV, verifyStartPos * nHeadsKv * vHeadDim, verifyV, 0, verifyV.Length);
            CopyFloatsToCuda(gpu, dKBatch, verifyK);
            CopyFloatsToCuda(gpu, dVBatch, verifyV);

            int vStart = verifyStartPos;
            int vBatch = verifyBatchSize;
            uint verifyStoreGrid = (uint)((verifyBatchSize * nHeadsKv * maxDim + 255) / 256);
            storeBatchArgs[5] = &vStart;
            storeBatchArgs[6] = &vBatch;
            CuDriver.Check(CuDriver.LaunchKernel(fnStoreBatch, verifyStoreGrid, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeBatchArgs, IntPtr.Zero), "paged_kv_cache_store_batch_f16_kernel(verify)");

            CopyFloatsToCuda(gpu, dQBatch, batchQ);
            void** attnBatchArgs = stackalloc void*[15];
            attnBatchArgs[0] = &dQBatch;
            attnBatchArgs[1] = &dKPool;
            attnBatchArgs[2] = &dVPool;
            attnBatchArgs[3] = &dOutBatch;
            attnBatchArgs[4] = &dBlockTable;
            attnBatchArgs[5] = &vStart;
            attnBatchArgs[6] = &vBatch;
            attnBatchArgs[7] = &layer;
            attnBatchArgs[8] = &layers;
            attnBatchArgs[9] = &nQ;
            attnBatchArgs[10] = &nKv;
            attnBatchArgs[11] = &hDim;
            attnBatchArgs[12] = &vDim;
            attnBatchArgs[13] = &blkSize;
            attnBatchArgs[14] = &attnScale;
            CuDriver.Check(CuDriver.LaunchKernel(fnAttnBatch, (uint)nHeadsQ, (uint)verifyBatchSize, 1, attnThreads, 1, 1, 0, IntPtr.Zero, (IntPtr)attnBatchArgs, IntPtr.Zero), "paged_attention_gqa_batch_f16_kernel");
            gpu.Synchronize();

            float[] actualBatch = CopyFloatsFromCuda(gpu, dOutBatch, verifyBatchSize * nHeadsQ * vHeadDim);
            for (int b = 0; b < verifyBatchSize; b++)
            {
                float[] qSlice = new float[nHeadsQ * headDim];
                Array.Copy(batchQ, b * nHeadsQ * headDim, qSlice, 0, qSlice.Length);
                float[] expSlice = ComputeCpuGqaReference(qSlice, kCacheRef, vCacheRef, nHeadsQ, nHeadsKv, headDim, vHeadDim, totalSeq, verifyStartPos + b, attnScale, quantizeKvFp16: true);
                float[] actSlice = new float[nHeadsQ * vHeadDim];
                Array.Copy(actualBatch, b * nHeadsQ * vHeadDim, actSlice, 0, actSlice.Length);

                Assert.True(MaxAbsDiff(expSlice, actSlice) < 2e-3f,
                    $"CUDA PagedAttention batch mismatch at b={b} (headDim={headDim}, vHeadDim={vHeadDim}): maxDiff={MaxAbsDiff(expSlice, actSlice)}");
            }
        }
        finally
        {
            gpu.FreeDevice(dKPool);
            gpu.FreeDevice(dVPool);
            gpu.FreeDevice(dBlockTable);
            gpu.FreeDevice(dKBatch);
            gpu.FreeDevice(dVBatch);
            gpu.FreeDevice(dQSingle);
            gpu.FreeDevice(dOutSingle);
            gpu.FreeDevice(dQBatch);
            gpu.FreeDevice(dOutBatch);
        }
    }

    private static unsafe void VerifyD3D12AttentionGqa(D3D12Context ctx, int headDim, bool useFp16)
    {
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int maxSeqLen = 12;
        const int pos = 9;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        var rng = new Random(4321 + headDim + (useFp16 ? 1 : 0));
        float[] q = RandomFloats(rng, nHeadsQ * headDim, -0.5f, 0.5f);
        float[] kCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);
        float[] vCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);

        float[] expected = ComputeCpuGqaReference(q, kCache, vCache, nHeadsQ, nHeadsKv, headDim, headDim, maxSeqLen, pos, attnScale, quantizeKvFp16: useFp16);

        var rootParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var rootSig = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rootParams));
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(useFp16 ? D3D12Shaders.AttentionGqaF16 : D3D12Shaders.AttentionGqa));

        using var dQ = ctx.CreateDeviceBuffer((ulong)(q.Length * sizeof(float)));
        using var dK = ctx.CreateDeviceBuffer((ulong)(useFp16 ? kCache.Length * sizeof(ushort) : kCache.Length * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(useFp16 ? vCache.Length * sizeof(ushort) : vCache.Length * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(q.Length * sizeof(float)));

        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(q.AsSpan()));
        if (useFp16)
        {
            uint[] kPacked = PackFp16Pairs(kCache);
            uint[] vPacked = PackFp16Pairs(vCache);
            ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kPacked.AsSpan()));
            ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vPacked.AsSpan()));
        }
        else
        {
            ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kCache.AsSpan()));
            ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vCache.AsSpan()));
        }

        ctx.BeginCommands();
        ctx.CommandList.SetComputeRootSignature(rootSig);
        ctx.CommandList.SetPipelineState(pso);

        uint* cbuf = stackalloc uint[6];
        cbuf[0] = (uint)nHeadsQ;
        cbuf[1] = (uint)nHeadsKv;
        cbuf[2] = (uint)headDim;
        cbuf[3] = (uint)maxSeqLen;
        cbuf[4] = (uint)pos;
        cbuf[5] = BitConverter.SingleToUInt32Bits(attnScale);
        ctx.CommandList.SetComputeRoot32BitConstants(0, 6, (IntPtr)cbuf, 0);
        ctx.CommandList.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        ctx.CommandList.Dispatch((uint)nHeadsQ, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[q.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actual.AsSpan()), dOut, (ulong)(actual.Length * sizeof(float)));
        Assert.True(MaxAbsDiff(expected, actual) < 2e-3f,
            $"D3D12 AttentionGqa (fp16={useFp16}, headDim={headDim}) mismatch: maxDiff={MaxAbsDiff(expected, actual)}");

        // Verify out-of-bounds pos >= maxSeqLen guard safely no-ops without modifying dOut
        ctx.ZeroDeviceBuffer(dOut);
        ctx.BeginCommands();
        ctx.CommandList.SetComputeRootSignature(rootSig);
        ctx.CommandList.SetPipelineState(pso);
        cbuf[4] = (uint)maxSeqLen; // out-of-bounds!
        ctx.CommandList.SetComputeRoot32BitConstants(0, 6, (IntPtr)cbuf, 0);
        ctx.CommandList.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        ctx.CommandList.Dispatch((uint)nHeadsQ, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] zeroCheck = new float[q.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(zeroCheck.AsSpan()), dOut, (ulong)(zeroCheck.Length * sizeof(float)));
        for (int i = 0; i < zeroCheck.Length; i++)
        {
            Assert.Equal(0f, zeroCheck[i]);
        }
    }

    private static unsafe void VerifyD3D12AttentionBatch(D3D12Context ctx, int headDim)
    {
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int maxSeqLen = 12;
        const int startPos = 5;
        const int batchSize = 4;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        var rng = new Random(8765 + headDim);
        float[] qBatch = RandomFloats(rng, batchSize * nHeadsQ * headDim, -0.5f, 0.5f);
        float[] kCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);
        float[] vCache = RandomFloats(rng, nHeadsKv * maxSeqLen * headDim, -0.5f, 0.5f);

        var rootParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var rootSig = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rootParams));
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.AttentionBatch));

        using var dQ = ctx.CreateDeviceBuffer((ulong)(qBatch.Length * sizeof(float)));
        using var dK = ctx.CreateDeviceBuffer((ulong)(kCache.Length * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(vCache.Length * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(qBatch.Length * sizeof(float)));

        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qBatch.AsSpan()));
        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kCache.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vCache.AsSpan()));

        ctx.BeginCommands();
        ctx.CommandList.SetComputeRootSignature(rootSig);
        ctx.CommandList.SetPipelineState(pso);

        uint* cbuf = stackalloc uint[7];
        cbuf[0] = (uint)nHeadsQ;
        cbuf[1] = (uint)nHeadsKv;
        cbuf[2] = (uint)headDim;
        cbuf[3] = (uint)maxSeqLen;
        cbuf[4] = (uint)startPos;
        cbuf[5] = BitConverter.SingleToUInt32Bits(attnScale);
        cbuf[6] = (uint)batchSize;
        ctx.CommandList.SetComputeRoot32BitConstants(0, 7, (IntPtr)cbuf, 0);
        ctx.CommandList.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        ctx.CommandList.Dispatch((uint)nHeadsQ, (uint)batchSize, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actualBatch = new float[qBatch.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actualBatch.AsSpan()), dOut, (ulong)(actualBatch.Length * sizeof(float)));

        for (int b = 0; b < batchSize; b++)
        {
            float[] qSlice = new float[nHeadsQ * headDim];
            Array.Copy(qBatch, b * nHeadsQ * headDim, qSlice, 0, qSlice.Length);
            float[] expected = ComputeCpuGqaReference(qSlice, kCache, vCache, nHeadsQ, nHeadsKv, headDim, headDim, maxSeqLen, startPos + b, attnScale, quantizeKvFp16: false);
            float[] actual = new float[nHeadsQ * headDim];
            Array.Copy(actualBatch, b * nHeadsQ * headDim, actual, 0, actual.Length);

            Assert.True(MaxAbsDiff(expected, actual) < 2e-3f,
                $"D3D12 AttentionBatch mismatch at b={b}, headDim={headDim}: maxDiff={MaxAbsDiff(expected, actual)}");
        }
    }

    private static float[] ComputeCpuGqaReference(
        float[] q,
        float[] kCache,
        float[] vCache,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        int vHeadDim,
        int maxSeqLen,
        int pos,
        float attnScale,
        bool quantizeKvFp16)
    {
        float[] output = new float[nHeadsQ * vHeadDim];
        int groupSize = nHeadsQ / nHeadsKv;

        for (int h = 0; h < nHeadsQ; h++)
        {
            int hKv = h / groupSize;
            float[] scores = new float[pos + 1];
            float maxScore = float.NegativeInfinity;

            for (int t = 0; t <= pos; t++)
            {
                float dot = 0f;
                int kBase = (hKv * maxSeqLen + t) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float kd = quantizeKvFp16 ? (float)(Half)kCache[kBase + d] : kCache[kBase + d];
                    dot += q[h * headDim + d] * kd;
                }
                float s = dot * attnScale;
                scores[t] = s;
                if (s > maxScore) maxScore = s;
            }

            float sumExp = 0f;
            for (int t = 0; t <= pos; t++)
            {
                scores[t] = MathF.Exp(scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int t = 0; t <= pos; t++)
            {
                float w = scores[t] / sumExp;
                int vBase = (hKv * maxSeqLen + t) * vHeadDim;
                for (int d = 0; d < vHeadDim; d++)
                {
                    float vd = quantizeKvFp16 ? (float)(Half)vCache[vBase + d] : vCache[vBase + d];
                    output[h * vHeadDim + d] += w * vd;
                }
            }
        }
        return output;
    }

    // =========================================================================
    // Speculative Decoding Mock Harnesses
    // =========================================================================

    private sealed class FixedDistributionSpeculativeTarget : ISpeculativeTarget
    {
        private readonly float[] _logits;
        private readonly Sampler _sampler;
        private readonly int _firstToken;
        private int _generatedCount;

        public BpeTokenizer Tokenizer { get; }
        public int MaxSeqLen => 64;
        public int VocabSize => _logits.Length;
        public bool SupportsLogitVerification => true;

        public FixedDistributionSpeculativeTarget(BpeTokenizer tokenizer, float[] targetProbs, int firstToken, int seed)
        {
            Tokenizer = tokenizer;
            _firstToken = firstToken;
            _sampler = new Sampler(seed);
            _logits = new float[targetProbs.Length];
            for (int i = 0; i < targetProbs.Length; i++)
            {
                _logits[i] = targetProbs[i] > 0f ? MathF.Log(targetProbs[i]) : -1e20f;
            }
        }

        public void ResetKvCache() => _generatedCount = 0;
        public void Prefill(ReadOnlySpan<int> promptTokens) => _generatedCount = 0;

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            if (_generatedCount == 0)
            {
                _generatedCount++;
                return _firstToken; // Deterministic first token so token #2 always goes through speculative Draft+Verify!
            }
            float[] copy = (float[])_logits.Clone();
            return _sampler.Sample(copy, options);
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            for (int i = 0; i < tokens.Length; i++) predictedTokens[i] = _firstToken;
        }

        public void ForwardToken(int token, int pos, bool computeLogits) { }

        public void CopyCurrentLogits(Span<float> destination) => _logits.AsSpan().CopyTo(destination);

        public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                _logits.AsSpan().CopyTo(batchLogitsOut.Slice(i * _logits.Length, _logits.Length));
            }
        }

        public void Dispose() { }
    }

    private sealed class FixedDistributionDraftProvider : IDraftProvider
    {
        private readonly float[] _draftProbs;
        private readonly int _fallbackToken;

        public FixedDistributionDraftProvider(float[] draftProbs, int fallbackToken)
        {
            _draftProbs = draftProbs;
            _fallbackToken = fallbackToken;
        }

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            draftTokens[0] = _fallbackToken;
            return 1;
        }

        public int DraftWithProbabilities(
            ReadOnlySpan<int> tokens,
            ReadOnlySpan<int> recentTokens,
            int maxDraftTokens,
            SamplingOptions sampling,
            Sampler sampler,
            int targetVocabSize,
            Span<int> draftTokens,
            Span<float> draftProbs)
        {
            draftProbs.Slice(0, targetVocabSize).Clear();
            for (int i = 0; i < _draftProbs.Length && i < targetVocabSize; i++)
            {
                draftProbs[i] = _draftProbs[i];
            }
            draftTokens[0] = Sampler.SampleFromDistribution(draftProbs.Slice(0, targetVocabSize), sampler.Rng);
            return 1;
        }

        public void Dispose() { }
    }

    private sealed class StatefulRecurrentMockTarget : ISpeculativeTarget
    {
        public BpeTokenizer Tokenizer { get; }
        public int MaxSeqLen => 64;
        public long StateSnapshotBytes => sizeof(long);
        public long RecurrentState { get; private set; }
        public int RestoreCallCount { get; private set; }

        public StatefulRecurrentMockTarget(BpeTokenizer tokenizer) => Tokenizer = tokenizer;

        public static long StepState(long prev, int token, int pos) => prev * 31L + token + pos * 7L;

        public void ResetKvCache() => RecurrentState = 0;

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            RecurrentState = 0;
            for (int i = 0; i < promptTokens.Length; i++)
            {
                RecurrentState = StepState(RecurrentState, promptTokens[i], i);
            }
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default) => 3;

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                RecurrentState = StepState(RecurrentState, tokens[i], startPos + i);
                predictedTokens[i] = 4 + i; // i=0 -> 4 (matches draft[0]=4), i=1 -> 5 (rejects draft[1]=9)
            }
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            RecurrentState = StepState(RecurrentState, token, pos);
        }

        public void SaveRecurrentState(Span<byte> destination)
        {
            long s = RecurrentState;
            MemoryMarshal.Write(destination, in s);
        }

        public void RestoreRecurrentState(ReadOnlySpan<byte> snapshot)
        {
            RecurrentState = MemoryMarshal.Read<long>(snapshot);
            RestoreCallCount++;
        }

        public void Dispose() { }
    }

    private sealed class FixedSequenceDraftProvider : IDraftProvider
    {
        private readonly int[] _drafts;
        public FixedSequenceDraftProvider(int[] drafts) => _drafts = drafts;

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            int n = Math.Min(maxDraftTokens, _drafts.Length);
            _drafts.AsSpan(0, n).CopyTo(draftTokens);
            return n;
        }

        public void Dispose() { }
    }

    private sealed class ScriptedSequenceTarget : ISpeculativeTarget
    {
        private readonly int[] _script;
        private int _promptLen;
        private int _cursor;

        public BpeTokenizer Tokenizer { get; }
        public int MaxSeqLen => 64;

        public ScriptedSequenceTarget(BpeTokenizer tokenizer, int[] script)
        {
            Tokenizer = tokenizer;
            _script = script;
        }

        public void ResetKvCache()
        {
            _promptLen = 0;
            _cursor = 0;
        }

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            _promptLen = promptTokens.Length;
            _cursor = 0;
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            return _cursor < _script.Length ? _script[_cursor++] : Tokenizer.EosTokenId;
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                int idx = (startPos - _promptLen) + i + 1;
                predictedTokens[i] = idx < _script.Length ? _script[idx] : Tokenizer.EosTokenId;
            }
            _cursor = Math.Min(_script.Length, (startPos - _promptLen) + tokens.Length + 1);
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            _cursor = (pos - _promptLen) + 1;
        }

        public void Dispose() { }
    }

    private sealed class FirstStepOnlyDraftProvider : IDraftProvider
    {
        private readonly int[] _firstDrafts;
        private bool _used;

        public FirstStepOnlyDraftProvider(int[] firstDrafts) => _firstDrafts = firstDrafts;

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            if (_used) return 0; // Force serial fallback on subsequent steps!
            _used = true;
            int n = Math.Min(maxDraftTokens, _firstDrafts.Length);
            _firstDrafts.AsSpan(0, n).CopyTo(draftTokens);
            return n;
        }

        public void Dispose() { }
    }

    private static unsafe void VerifyCudaPagedAttentionNegativePhysBlock(
        GpuContext gpu,
        IntPtr fnStoreSingle,
        IntPtr fnAttnSingle,
        IntPtr fnAttnBatch)
    {
        var rng = new Random(9911);
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int headDim = 128;
        const int vHeadDim = 128;
        const int kvBlockSize = 8;
        const int numPhysicalBlocks = 4;
        const int numLayers = 1;
        const int modelLayer = 0;

        // Logical block 0 -> phys block 2; logical block 1 -> -1 (unmapped!)
        int[] blockTable = { 2, -1 };
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        float[] validK = RandomFloats(rng, kvBlockSize * nHeadsKv * headDim, -0.5f, 0.5f);
        float[] validV = RandomFloats(rng, kvBlockSize * nHeadsKv * vHeadDim, -0.5f, 0.5f);
        float[] qSingle = RandomFloats(rng, nHeadsQ * headDim, -0.5f, 0.5f);
        const int batchSize = 2;
        float[] qBatch = RandomFloats(rng, batchSize * nHeadsQ * headDim, -0.5f, 0.5f);

        float[] kCacheRef = new float[nHeadsKv * kvBlockSize * headDim];
        float[] vCacheRef = new float[nHeadsKv * kvBlockSize * vHeadDim];
        for (int t = 0; t < kvBlockSize; t++)
        {
            for (int h = 0; h < nHeadsKv; h++)
            {
                Array.Copy(validK, (t * nHeadsKv + h) * headDim, kCacheRef, (h * kvBlockSize + t) * headDim, headDim);
                Array.Copy(validV, (t * nHeadsKv + h) * vHeadDim, vCacheRef, (h * kvBlockSize + t) * vHeadDim, vHeadDim);
            }
        }

        IntPtr dKPool = gpu.AllocateDevice((nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * headDim * sizeof(ushort)));
        IntPtr dVPool = gpu.AllocateDevice((nuint)(numPhysicalBlocks * numLayers * kvBlockSize * nHeadsKv * vHeadDim * sizeof(ushort)));
        IntPtr dBlockTable = gpu.AllocateDevice((nuint)(blockTable.Length * sizeof(int)));
        IntPtr dStepK = gpu.AllocateDevice((nuint)(nHeadsKv * headDim * sizeof(float)));
        IntPtr dStepV = gpu.AllocateDevice((nuint)(nHeadsKv * vHeadDim * sizeof(float)));
        IntPtr dQSingle = gpu.AllocateDevice((nuint)(qSingle.Length * sizeof(float)));
        IntPtr dOutSingle = gpu.AllocateDevice((nuint)(nHeadsQ * vHeadDim * sizeof(float)));
        IntPtr dQBatch = gpu.AllocateDevice((nuint)(qBatch.Length * sizeof(float)));
        IntPtr dOutBatch = gpu.AllocateDevice((nuint)(batchSize * nHeadsQ * vHeadDim * sizeof(float)));

        try
        {
            CopyIntsToCuda(gpu, dBlockTable, blockTable);

            int layer = modelLayer;
            int layers = numLayers;
            int nQ = nHeadsQ;
            int nKv = nHeadsKv;
            int hDim = headDim;
            int vDim = vHeadDim;
            int blkSize = kvBlockSize;
            int singlePos = 0;
            uint storeGrid = (uint)((nHeadsKv * headDim + 255) / 256);

            void** storeArgs = stackalloc void*[12];
            storeArgs[0] = &dStepK;
            storeArgs[1] = &dStepV;
            storeArgs[2] = &dKPool;
            storeArgs[3] = &dVPool;
            storeArgs[4] = &dBlockTable;
            storeArgs[5] = &singlePos;
            storeArgs[6] = &layer;
            storeArgs[7] = &layers;
            storeArgs[8] = &nKv;
            storeArgs[9] = &hDim;
            storeArgs[10] = &vDim;
            storeArgs[11] = &blkSize;

            float[] stepK = new float[nHeadsKv * headDim];
            float[] stepV = new float[nHeadsKv * vHeadDim];
            for (int t = 0; t < kvBlockSize; t++)
            {
                Array.Copy(validK, t * nHeadsKv * headDim, stepK, 0, stepK.Length);
                Array.Copy(validV, t * nHeadsKv * vHeadDim, stepV, 0, stepV.Length);
                CopyFloatsToCuda(gpu, dStepK, stepK);
                CopyFloatsToCuda(gpu, dStepV, stepV);
                singlePos = t;
                CuDriver.Check(CuDriver.LaunchKernel(fnStoreSingle, storeGrid, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeArgs, IntPtr.Zero), "paged_kv_cache_store_f16_kernel");
            }

            // Query at pos=11 (logical block 1 has phys_block == -1!): must safely skip positions 8..11 and match block 0 (0..7)
            CopyFloatsToCuda(gpu, dQSingle, qSingle);
            int queryPos = 11;
            void** attnSingleArgs = stackalloc void*[14];
            attnSingleArgs[0] = &dQSingle;
            attnSingleArgs[1] = &dKPool;
            attnSingleArgs[2] = &dVPool;
            attnSingleArgs[3] = &dOutSingle;
            attnSingleArgs[4] = &dBlockTable;
            attnSingleArgs[5] = &queryPos;
            attnSingleArgs[6] = &layer;
            attnSingleArgs[7] = &layers;
            attnSingleArgs[8] = &nQ;
            attnSingleArgs[9] = &nKv;
            attnSingleArgs[10] = &hDim;
            attnSingleArgs[11] = &vDim;
            attnSingleArgs[12] = &blkSize;
            attnSingleArgs[13] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(fnAttnSingle, (uint)nHeadsQ, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)attnSingleArgs, IntPtr.Zero), "paged_attention_gqa_f16_kernel(neg_block)");
            gpu.Synchronize();

            float[] expectedSingle = ComputeCpuGqaReference(qSingle, kCacheRef, vCacheRef, nHeadsQ, nHeadsKv, headDim, vHeadDim, kvBlockSize, kvBlockSize - 1, attnScale, quantizeKvFp16: true);
            float[] actualSingle = CopyFloatsFromCuda(gpu, dOutSingle, nHeadsQ * vHeadDim);
            Assert.True(MaxAbsDiff(expectedSingle, actualSingle) < 2e-3f,
                $"Negative phys_block single PagedAttention mismatch: maxDiff={MaxAbsDiff(expectedSingle, actualSingle)}");

            // Batch query at startPos=8, batchSize=2 (cur_pos=8, 9 in unmapped logical block 1)
            CopyFloatsToCuda(gpu, dQBatch, qBatch);
            int startPos = 8;
            int bSize = batchSize;
            void** attnBatchArgs = stackalloc void*[15];
            attnBatchArgs[0] = &dQBatch;
            attnBatchArgs[1] = &dKPool;
            attnBatchArgs[2] = &dVPool;
            attnBatchArgs[3] = &dOutBatch;
            attnBatchArgs[4] = &dBlockTable;
            attnBatchArgs[5] = &startPos;
            attnBatchArgs[6] = &bSize;
            attnBatchArgs[7] = &layer;
            attnBatchArgs[8] = &layers;
            attnBatchArgs[9] = &nQ;
            attnBatchArgs[10] = &nKv;
            attnBatchArgs[11] = &hDim;
            attnBatchArgs[12] = &vDim;
            attnBatchArgs[13] = &blkSize;
            attnBatchArgs[14] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(fnAttnBatch, (uint)nHeadsQ, (uint)batchSize, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)attnBatchArgs, IntPtr.Zero), "paged_attention_gqa_batch_f16_kernel(neg_block)");
            gpu.Synchronize();

            float[] actualBatch = CopyFloatsFromCuda(gpu, dOutBatch, batchSize * nHeadsQ * vHeadDim);
            for (int b = 0; b < batchSize; b++)
            {
                float[] qSlice = new float[nHeadsQ * headDim];
                Array.Copy(qBatch, b * qSlice.Length, qSlice, 0, qSlice.Length);
                float[] expectedB = ComputeCpuGqaReference(qSlice, kCacheRef, vCacheRef, nHeadsQ, nHeadsKv, headDim, vHeadDim, kvBlockSize, kvBlockSize - 1, attnScale, quantizeKvFp16: true);
                float[] actualB = new float[nHeadsQ * vHeadDim];
                Array.Copy(actualBatch, b * actualB.Length, actualB, 0, actualB.Length);
                Assert.True(MaxAbsDiff(expectedB, actualB) < 2e-3f,
                    $"Negative phys_block batch PagedAttention mismatch at b={b}: maxDiff={MaxAbsDiff(expectedB, actualB)}");
            }
        }
        finally
        {
            gpu.FreeDevice(dKPool);
            gpu.FreeDevice(dVPool);
            gpu.FreeDevice(dBlockTable);
            gpu.FreeDevice(dStepK);
            gpu.FreeDevice(dStepV);
            gpu.FreeDevice(dQSingle);
            gpu.FreeDevice(dOutSingle);
            gpu.FreeDevice(dQBatch);
            gpu.FreeDevice(dOutBatch);
        }
    }

    private static unsafe void VerifyRawD3D12RepetitionPenaltyShaderBoundsGuard(D3D12Context ctx)
    {
        float[] logits = { 10.0f, -4.0f, 8.0f, 6.0f, 2.0f, -2.0f, 1.0f, 3.0f };
        int[] rawIndicesWithOob = { -99, 3, 8, 999999 }; // only token 3 is within [0, 8)!

        var repParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        using var sig = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, repParams));
        using var pso = ctx.CreatePipelineState(sig, ctx.CompileShader(D3D12Shaders.RepetitionPenalty));
        using var dLogits = ctx.CreateDeviceBuffer((ulong)(logits.Length * sizeof(float)));
        using var dTokens = ctx.CreateDeviceBuffer((ulong)(rawIndicesWithOob.Length * sizeof(int)));

        ctx.CopyToDevice(dLogits, MemoryMarshal.AsBytes(logits.AsSpan()));
        ctx.CopyToDevice(dTokens, MemoryMarshal.AsBytes(rawIndicesWithOob.AsSpan()));

        ctx.BeginCommands();
        ctx.CommandList.SetComputeRootSignature(sig);
        ctx.CommandList.SetComputeRootUnorderedAccessView(2, dLogits.GPUVirtualAddress);
        ctx.CommandList.SetComputeRootShaderResourceView(1, dTokens.GPUVirtualAddress);
        ctx.CommandList.SetPipelineState(pso);

        uint* consts = stackalloc uint[3];
        consts[0] = (uint)rawIndicesWithOob.Length;
        consts[1] = (uint)logits.Length;
        consts[2] = BitConverter.SingleToUInt32Bits(2.0f);
        ctx.CommandList.SetComputeRoot32BitConstants(0, 3, (IntPtr)consts, 0);
        ctx.CommandList.Dispatch(1, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[logits.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(actual.AsSpan()), dLogits, (ulong)(actual.Length * sizeof(float)));
        Assert.Equal(3.0f, actual[3], precision: 4); // 6.0f / 2.0f
        Assert.Equal(10.0f, actual[0], precision: 4);
        Assert.Equal(3.0f, actual[7], precision: 4);
    }

    private sealed class PrefixTrackingMockTarget : ISpeculativeTarget
    {
        private readonly int[] _sequence;
        private int _seqIdx;

        public BpeTokenizer Tokenizer { get; }
        public int MaxSeqLen => 64;
        public bool CanOverwriteKvFromPrefix { get; }
        public int ResetCount { get; private set; }
        public int PrefillCount { get; private set; }
        public int[] LastPrefilledTokens { get; private set; } = Array.Empty<int>();
        public List<(int Token, int Pos)> ForwardedSteps { get; } = new();

        public PrefixTrackingMockTarget(BpeTokenizer tokenizer, bool canOverwriteKvFromPrefix, int[] nextTokenSequence)
        {
            Tokenizer = tokenizer;
            CanOverwriteKvFromPrefix = canOverwriteKvFromPrefix;
            _sequence = nextTokenSequence;
        }

        public void ResetKvCache() => ResetCount++;

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            PrefillCount++;
            LastPrefilledTokens = promptTokens.ToArray();
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            int tok = _sequence[_seqIdx % _sequence.Length];
            _seqIdx++;
            return tok;
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            for (int i = 0; i < tokens.Length; i++) predictedTokens[i] = _sequence[0];
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            ForwardedSteps.Add((token, pos));
        }

        public void Dispose() { }
    }

    private sealed class ZeroProbabilityCandidateDraftProvider : IDraftProvider
    {
        private readonly int _candidateToken;

        public ZeroProbabilityCandidateDraftProvider(int candidateToken) => _candidateToken = candidateToken;

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            draftTokens[0] = _candidateToken;
            return 1;
        }

        public int DraftWithProbabilities(
            ReadOnlySpan<int> tokens,
            ReadOnlySpan<int> recentTokens,
            int maxDraftTokens,
            SamplingOptions sampling,
            Sampler sampler,
            int targetVocabSize,
            Span<int> draftTokens,
            Span<float> draftProbs)
        {
            draftProbs.Slice(0, targetVocabSize).Clear(); // q(x) = 0 everywhere!
            draftTokens[0] = _candidateToken;
            return 1;
        }

        public void Dispose() { }
    }

    private sealed class UnsnapshottableOrRingMockTarget : ISpeculativeTarget
    {
        private readonly int[] _tokens;
        private int _idx;

        public BpeTokenizer Tokenizer { get; }
        public int MaxSeqLen => 64;
        public bool IsRingBufferEnabled { get; }
        public bool HasUnsnapshottableRecurrentState { get; }

        public UnsnapshottableOrRingMockTarget(BpeTokenizer tokenizer, bool ringBuffer, bool unsnapshottable, int[] tokens)
        {
            Tokenizer = tokenizer;
            IsRingBufferEnabled = ringBuffer;
            HasUnsnapshottableRecurrentState = unsnapshottable;
            _tokens = tokens;
        }

        public void ResetKvCache() => _idx = 0;
        public void Prefill(ReadOnlySpan<int> promptTokens) => _idx = 0;

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            return _idx < _tokens.Length ? _tokens[_idx++] : Tokenizer.EosTokenId;
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
        {
            throw new InvalidOperationException("Speculative batch verification must not be called when drafting is disabled!");
        }

        public void ForwardToken(int token, int pos, bool computeLogits) { }
        public void Dispose() { }
    }

    private sealed class CountingDraftProvider : IDraftProvider
    {
        private readonly int[] _drafts;
        public int CallCount { get; private set; }

        public CountingDraftProvider(int[] drafts) => _drafts = drafts;

        public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
        {
            CallCount++;
            int n = Math.Min(maxDraftTokens, _drafts.Length);
            _drafts.AsSpan(0, n).CopyTo(draftTokens);
            return n;
        }

        public void Dispose() { }
    }

    // =========================================================================
    // Synthetic GGUF / LoRA / WAV Builders
    // =========================================================================

    private static void WriteSyntheticQwen2Gguf(
        string path,
        int dim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize,
        bool fusedGateUp)
    {
        int headDim = dim / headCount;
        int qDim = headCount * headDim;
        int kvDim = headCountKv * headDim;
        var rng = new Random(101);

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("qwen2.block_count", GgufValueType.Uint32, 1u),
            ("qwen2.context_length", GgufValueType.Uint32, 64u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, BuildGpt2ByteVocab(vocabSize))
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),
            ("output_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.attn_q.weight", new[] { (ulong)dim, (ulong)qDim }, RandomFloats(rng, dim * qDim, -0.2f, 0.2f)),
            ("blk.0.attn_k.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)),
            ("blk.0.attn_v.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)),
            ("blk.0.attn_output.weight", new[] { (ulong)qDim, (ulong)dim }, RandomFloats(rng, qDim * dim, -0.2f, 0.2f)),
            ("blk.0.ffn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, RandomFloats(rng, ffnDim * dim, -0.2f, 0.2f))
        };

        if (fusedGateUp)
        {
            tensors.Add(("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)(2 * ffnDim) }, RandomFloats(rng, dim * 2 * ffnDim, -0.2f, 0.2f)));
        }
        else
        {
            tensors.Add(("blk.0.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)));
            tensors.Add(("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)));
        }

        WriteF32Gguf(path, metadata, tensors);
    }

    private static void WriteSyntheticQwen35HybridGguf(
        string path,
        int dim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize)
    {
        int headDim = dim / headCount;
        int qDim = headCount * headDim;
        int kvDim = headCountKv * headDim;
        const int ssmConvKernel = 4;
        const int ssmStateSize = 32;
        const int ssmGroupCount = 1;
        const int ssmHeads = 2;
        const int ssmInnerSize = 64; // ssmHeads * ssmStateSize
        const int gdnConvChannels = (ssmGroupCount * 2 + ssmHeads) * ssmStateSize; // 128

        var rng = new Random(202);
        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen35"),
            ("qwen35.block_count", GgufValueType.Uint32, 2u),
            ("qwen35.context_length", GgufValueType.Uint32, 64u),
            ("qwen35.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen35.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen35.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("qwen35.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("qwen35.full_attention_interval", GgufValueType.Uint32, 2u),
            ("qwen35.ssm.conv_kernel", GgufValueType.Uint32, (uint)ssmConvKernel),
            ("qwen35.ssm.state_size", GgufValueType.Uint32, (uint)ssmStateSize),
            ("qwen35.ssm.group_count", GgufValueType.Uint32, (uint)ssmGroupCount),
            ("qwen35.ssm.time_step_rank", GgufValueType.Uint32, (uint)ssmHeads),
            ("qwen35.ssm.inner_size", GgufValueType.Uint32, (uint)ssmInnerSize),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, BuildGpt2ByteVocab(vocabSize))
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),
            ("output_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),

            // Layer 0: Gated DeltaNet (SSM) layer
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.attn_qkv.weight", new[] { (ulong)dim, (ulong)gdnConvChannels }, RandomFloats(rng, dim * gdnConvChannels, -0.25f, 0.25f)),
            ("blk.0.attn_gate.weight", new[] { (ulong)dim, (ulong)ssmInnerSize }, RandomFloats(rng, dim * ssmInnerSize, -0.25f, 0.25f)),
            ("blk.0.ssm_conv1d.weight", new[] { (ulong)ssmConvKernel, (ulong)gdnConvChannels }, RandomFloats(rng, ssmConvKernel * gdnConvChannels, 0.1f, 0.3f)),
            ("blk.0.ssm_a", new[] { (ulong)ssmHeads }, new[] { -0.35f, -0.50f }),
            ("blk.0.ssm_alpha.weight", new[] { (ulong)dim, (ulong)ssmHeads }, RandomFloats(rng, dim * ssmHeads, -0.2f, 0.2f)),
            ("blk.0.ssm_beta.weight", new[] { (ulong)dim, (ulong)ssmHeads }, RandomFloats(rng, dim * ssmHeads, -0.2f, 0.2f)),
            ("blk.0.ssm_dt.bias", new[] { (ulong)ssmHeads }, new[] { 0.1f, 0.2f }),
            ("blk.0.ssm_norm.weight", new[] { (ulong)ssmStateSize }, ConstantFloats(ssmStateSize, 1.0f)),
            ("blk.0.ssm_out.weight", new[] { (ulong)ssmInnerSize, (ulong)dim }, RandomFloats(rng, ssmInnerSize * dim, -0.25f, 0.25f)),
            ("blk.0.ffn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)),
            ("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)),
            ("blk.0.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, RandomFloats(rng, ffnDim * dim, -0.2f, 0.2f)),

            // Layer 1: Interleaved standard full-attention layer
            ("blk.1.attn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.1.attn_q.weight", new[] { (ulong)dim, (ulong)qDim }, RandomFloats(rng, dim * qDim, -0.2f, 0.2f)),
            ("blk.1.attn_k.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)),
            ("blk.1.attn_v.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)),
            ("blk.1.attn_output.weight", new[] { (ulong)qDim, (ulong)dim }, RandomFloats(rng, qDim * dim, -0.2f, 0.2f)),
            ("blk.1.ffn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.1.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)),
            ("blk.1.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)),
            ("blk.1.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, RandomFloats(rng, ffnDim * dim, -0.2f, 0.2f))
        };

        WriteF32Gguf(path, metadata, tensors);
    }

    private static void WriteSyntheticGemma4Gguf(
        string path,
        int dim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize)
    {
        const int swaHeadDim = 256;
        int qDim = headCount * swaHeadDim;
        int kvDim = headCountKv * swaHeadDim;
        var rng = new Random(303);

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "gemma4"),
            ("gemma4.block_count", GgufValueType.Uint32, 1u),
            ("gemma4.context_length", GgufValueType.Uint32, 64u),
            ("gemma4.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("gemma4.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("gemma4.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("gemma4.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("gemma4.attention.key_length", GgufValueType.Uint32, (uint)swaHeadDim),
            ("gemma4.attention.sliding_window", GgufValueType.Uint32, 16u),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, BuildGpt2ByteVocab(vocabSize))
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),
            ("output_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, RandomFloats(rng, dim * vocabSize, -0.2f, 0.2f)),
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.attn_q.weight", new[] { (ulong)dim, (ulong)qDim }, RandomFloats(rng, dim * qDim, -0.15f, 0.15f)),
            ("blk.0.attn_q_norm.weight", new[] { (ulong)swaHeadDim }, ConstantFloats(swaHeadDim, 1.0f)),
            ("blk.0.attn_k.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.15f, 0.15f)),
            ("blk.0.attn_k_norm.weight", new[] { (ulong)swaHeadDim }, ConstantFloats(swaHeadDim, 1.0f)),
            ("blk.0.attn_v.weight", new[] { (ulong)dim, (ulong)kvDim }, RandomFloats(rng, dim * kvDim, -0.15f, 0.15f)),
            ("blk.0.attn_output.weight", new[] { (ulong)qDim, (ulong)dim }, RandomFloats(rng, qDim * dim, -0.15f, 0.15f)),
            ("blk.0.post_attention_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.ffn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.15f, 0.15f)),
            ("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, RandomFloats(rng, dim * ffnDim, -0.15f, 0.15f)),
            ("blk.0.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, RandomFloats(rng, ffnDim * dim, -0.15f, 0.15f)),
            ("blk.0.post_ffw_norm_1.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.post_ffw_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f))
        };

        WriteF32Gguf(path, metadata, tensors);
    }

    private static void WriteMinimalDiffusionGguf(string path, string architecture)
    {
        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, architecture)
        };
        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("dummy.weight", new[] { 4UL }, new[] { 1f, 2f, 3f, 4f })
        };
        WriteF32Gguf(path, metadata, tensors);
    }

    private static void WriteF32Gguf(
        string path,
        List<(string Key, GgufValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, float[] Data)> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x46554747U); // 'GGUF'
        bw.Write(3U);          // Version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, val) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case GgufValueType.String:
                    WriteGgufString(bw, (string)val);
                    break;
                case GgufValueType.Uint32:
                    bw.Write((uint)val);
                    break;
                case GgufValueType.Float32:
                    bw.Write((float)val);
                    break;
                case GgufValueType.Array:
                    var arr = (string[])val;
                    bw.Write((uint)GgufValueType.String);
                    bw.Write((ulong)arr.Length);
                    for (int i = 0; i < arr.Length; i++)
                    {
                        WriteGgufString(bw, arr[i]);
                    }
                    break;
                default:
                    throw new NotSupportedException($"Unsupported synthetic GGUF metadata type {type}");
            }
        }

        ulong runningOffset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++) bw.Write(dims[d]);
            bw.Write((uint)GgufType.F32);
            bw.Write(runningOffset);

            ulong byteLen = (ulong)data.Length * sizeof(float);
            runningOffset = AlignUp(runningOffset + byteLen, 32UL);
        }

        long headerEnd = fs.Position;
        long alignedStart = (long)AlignUp((ulong)headerEnd, 32UL);
        while (fs.Position < alignedStart) bw.Write((byte)0);

        for (int t = 0; t < tensors.Count; t++)
        {
            float[] data = tensors[t].Data;
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
            long alignedNext = (long)AlignUp((ulong)fs.Position - (ulong)alignedStart, 32UL) + alignedStart;
            while (fs.Position < alignedNext) bw.Write((byte)0);
        }
    }

    private static void WriteSyntheticLoraFile(
        string path,
        int layerCount,
        int rank,
        float alpha,
        int dim,
        int qDim,
        int kvDim,
        int ffnDim)
    {
        var rng = new Random(404);
        var specs = new (string Proj, int InDim, int OutDim)[]
        {
            ("attn_q", dim, qDim),
            ("attn_k", dim, kvDim),
            ("attn_v", dim, kvDim),
            ("attn_output", qDim, dim),
            ("ffn_gate", dim, ffnDim),
            ("ffn_up", dim, ffnDim),
            ("ffn_down", ffnDim, dim)
        };

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x41524F4CU); // 'LORA'
        bw.Write(1U);          // Version 1
        bw.Write(layerCount);
        bw.Write(rank);
        bw.Write(alpha);
        bw.Write(layerCount * specs.Length * 2);

        for (int l = 0; l < layerCount; l++)
        {
            foreach (var (proj, inDim, outDim) in specs)
            {
                float[] a = RandomFloats(rng, inDim * rank, -0.3f, 0.3f);
                bw.Write($"blk.{l}.{proj}.lora_a.weight");
                bw.Write(inDim);
                bw.Write(rank);
                bw.Write(MemoryMarshal.AsBytes(a.AsSpan()));

                float[] b = RandomFloats(rng, rank * outDim, -0.3f, 0.3f);
                bw.Write($"blk.{l}.{proj}.lora_b.weight");
                bw.Write(rank);
                bw.Write(outDim);
                bw.Write(MemoryMarshal.AsBytes(b.AsSpan()));
            }
        }
    }

    private static byte[] BuildWavWithExtraChunks(float[] samples, int sampleRate)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII);

        byte[] junkPayload = new byte[28];
        byte[] listPayload = new byte[13]; // odd length -> requires 1 pad byte per RIFF spec!
        int dataBytes = samples.Length * sizeof(short);

        int riffSize = 4
            + (8 + 16)                          // fmt
            + (8 + junkPayload.Length)          // JUNK
            + (8 + listPayload.Length + 1)      // LIST + pad byte
            + (8 + dataBytes);                  // data

        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(riffSize);
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));

        // fmt chunk (PCM 16-bit mono)
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16);
        bw.Write((short)1); // PCM
        bw.Write((short)1); // Mono
        bw.Write(sampleRate);
        bw.Write(sampleRate * 2);
        bw.Write((short)2);
        bw.Write((short)16);

        // Intermediate JUNK chunk
        bw.Write(Encoding.ASCII.GetBytes("JUNK"));
        bw.Write(junkPayload.Length);
        bw.Write(junkPayload);

        // Odd-length LIST chunk + 1 RIFF alignment pad byte
        bw.Write(Encoding.ASCII.GetBytes("LIST"));
        bw.Write(listPayload.Length);
        bw.Write(listPayload);
        bw.Write((byte)0);

        // Actual audio data chunk
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(dataBytes);
        for (int i = 0; i < samples.Length; i++)
        {
            short pcm = (short)Math.Clamp((int)MathF.Round(samples[i] * 32767f), short.MinValue, short.MaxValue);
            bw.Write(pcm);
        }

        return ms.ToArray();
    }

    private static string[] BuildGpt2ByteVocab(int vocabSize)
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var byteToToken = new string[256];
        for (int i = 0; i < 256; i++)
        {
            byteToToken[bs[i]] = ((char)cs[i]).ToString();
        }

        string[] vocab = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i == vocabSize - 1) vocab[i] = "<|endoftext|>";
            else if (i == vocabSize - 2) vocab[i] = "<|im_start|>";
            else if (i < 256) vocab[i] = byteToToken[i];
            else vocab[i] = $"<|extra_{i}|>";
        }
        return vocab;
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        ulong rem = value % alignment;
        return rem == 0 ? value : value + (alignment - rem);
    }

    private static float[] RandomFloats(Random rng, int count, float min, float max)
    {
        float[] arr = new float[count];
        float scale = max - min;
        for (int i = 0; i < count; i++) arr[i] = min + (float)rng.NextDouble() * scale;
        return arr;
    }

    private static float[] ConstantFloats(int count, float value)
    {
        float[] arr = new float[count];
        Array.Fill(arr, value);
        return arr;
    }

    private static uint[] PackFp16Pairs(float[] src)
    {
        uint[] packed = new uint[src.Length / 2];
        for (int i = 0; i < packed.Length; i++)
        {
            ushort lo = BitConverter.HalfToUInt16Bits((Half)src[i * 2]);
            ushort hi = BitConverter.HalfToUInt16Bits((Half)src[i * 2 + 1]);
            packed[i] = lo | ((uint)hi << 16);
        }
        return packed;
    }

    private static unsafe void CopyFloatsToCuda(GpuContext gpu, IntPtr dDst, float[] src)
    {
        fixed (float* p = src)
        {
            gpu.CopyToDevice(dDst, (IntPtr)p, (nuint)(src.Length * sizeof(float)));
        }
    }

    private static unsafe void CopyIntsToCuda(GpuContext gpu, IntPtr dDst, int[] src)
    {
        fixed (int* p = src)
        {
            gpu.CopyToDevice(dDst, (IntPtr)p, (nuint)(src.Length * sizeof(int)));
        }
    }

    private static unsafe float[] CopyFloatsFromCuda(GpuContext gpu, IntPtr dSrc, int count)
    {
        float[] dst = new float[count];
        fixed (float* p = dst)
        {
            gpu.CopyToHost((IntPtr)p, dSrc, (nuint)(count * sizeof(float)));
        }
        return dst;
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (float.IsNaN(d) || d > max) max = d;
        }
        return max;
    }

    private static string CreateTempPath(string ext) =>
        Path.Combine(Path.GetTempPath(), $"glacier_phase_a_{Guid.NewGuid():N}{ext}");

    private static void DeleteIfExists(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch { }
    }
}
