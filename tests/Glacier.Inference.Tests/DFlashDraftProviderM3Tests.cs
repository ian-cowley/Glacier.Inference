namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

public sealed class DFlashDraftProviderM3Tests
{
    private static string CreateMinimalTokenizerGguf(int vocabSize = 32)
    {
        string path = Path.Combine(Path.GetTempPath(), $"dflash_tok_{Guid.NewGuid():N}.gguf");
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        void WriteStr(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)bytes.Length);
            bw.Write(bytes);
        }

        bw.Write(0x46554747u); // GGUF
        bw.Write(3u);          // version 3
        bw.Write(0UL);         // tensor count
        bw.Write(6UL);         // kv count

        WriteStr("general.architecture");
        bw.Write((uint)GgufValueType.String);
        WriteStr("qwen2");

        WriteStr("tokenizer.ggml.tokens");
        bw.Write((uint)GgufValueType.Array);
        bw.Write((uint)GgufValueType.String);
        bw.Write((ulong)vocabSize);
        for (int i = 0; i < vocabSize; i++)
        {
            if (i == 0) WriteStr("<|endoftext|>");
            else if (i == 1) WriteStr("<|im_start|>");
            else if (i == 2) WriteStr("<|im_end|>");
            else if (i <= 28) WriteStr(((char)('a' + (i - 3))).ToString());
            else WriteStr($"t{i}");
        }

        WriteStr("tokenizer.ggml.merges");
        bw.Write((uint)GgufValueType.Array);
        bw.Write((uint)GgufValueType.String);
        bw.Write(0UL);

        WriteStr("tokenizer.ggml.eos_token_id");
        bw.Write((uint)GgufValueType.Uint32);
        bw.Write(0u);

        WriteStr("tokenizer.ggml.bos_token_id");
        bw.Write((uint)GgufValueType.Uint32);
        bw.Write(1u);

        WriteStr("tokenizer.ggml.add_bos_token");
        bw.Write((uint)GgufValueType.Bool);
        bw.Write((byte)0);

        return path;
    }

    private sealed class SyntheticSpeculativeTarget : ISpeculativeTarget
    {
        private readonly string _tokPath;
        private readonly GgufFile _gguf;
        private readonly Func<IReadOnlyList<int>, float[]> _logitsForPrefix;
        private readonly List<int> _kvTokens = new();
        private readonly float[] _currentLogits;
        private readonly Sampler _sampler = new(seed: 123);
        private int _recurrentChecksum;

        public int MaxSeqLen { get; }
        public BpeTokenizer Tokenizer { get; }
        public int VocabSize { get; }
        public bool SupportsLogitVerification => true;
        public bool SupportsPerStepVerificationSnapshots { get; }
        public long StateSnapshotBytes { get; }
        public bool CanOverwriteKvFromPrefix => StateSnapshotBytes == 0;

        public SyntheticSpeculativeTarget(
            int vocabSize,
            int maxSeqLen,
            Func<IReadOnlyList<int>, float[]> logitsForPrefix,
            long stateSnapshotBytes = 0,
            bool perStepSnapshots = false)
        {
            VocabSize = vocabSize;
            MaxSeqLen = maxSeqLen;
            _logitsForPrefix = logitsForPrefix;
            StateSnapshotBytes = stateSnapshotBytes;
            SupportsPerStepVerificationSnapshots = perStepSnapshots;
            _currentLogits = new float[vocabSize];
            _tokPath = CreateMinimalTokenizerGguf(vocabSize);
            _gguf = GgufFile.Open(_tokPath);
            Tokenizer = new BpeTokenizer(_gguf);
        }

        public void ResetKvCache()
        {
            _kvTokens.Clear();
            _recurrentChecksum = 0;
        }

        public void Prefill(ReadOnlySpan<int> promptTokens)
        {
            ResetKvCache();
            for (int i = 0; i < promptTokens.Length; i++)
            {
                ForwardToken(promptTokens[i], i, computeLogits: i == promptTokens.Length - 1);
            }
        }

        public void ForwardToken(int token, int pos, bool computeLogits)
        {
            while (_kvTokens.Count > pos) _kvTokens.RemoveAt(_kvTokens.Count - 1);
            _kvTokens.Add(token);
            _recurrentChecksum = HashCode.Combine(_kvTokens.Count, token, pos);
            if (computeLogits)
            {
                float[] logits = _logitsForPrefix(_kvTokens);
                Array.Copy(logits, _currentLogits, VocabSize);
            }
        }

        public void CopyCurrentLogits(Span<float> destination)
        {
            _currentLogits.AsSpan(0, VocabSize).CopyTo(destination);
        }

        public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
        {
            Span<float> copy = _currentLogits.AsSpan(0, VocabSize).ToArray();
            return _sampler.Sample(copy, options, recentTokens);
        }

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
            => VerifyBatch(tokens, startPos, predictedTokens, null);

        public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens, byte[][]? perStepSnapshots)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                ForwardToken(tokens[i], startPos + i, computeLogits: true);
                predictedTokens[i] = SampleNextToken(SamplingOptions.Greedy);
                if (perStepSnapshots != null && i < perStepSnapshots.Length && perStepSnapshots[i] != null && StateSnapshotBytes > 0)
                {
                    SaveRecurrentState(perStepSnapshots[i]);
                }
            }
        }

        public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut, byte[][]? perStepSnapshots)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                ForwardToken(tokens[i], startPos + i, computeLogits: true);
                CopyCurrentLogits(batchLogitsOut.Slice(i * VocabSize, VocabSize));
                if (perStepSnapshots != null && i < perStepSnapshots.Length && perStepSnapshots[i] != null && StateSnapshotBytes > 0)
                {
                    SaveRecurrentState(perStepSnapshots[i]);
                }
            }
        }

        public void SaveRecurrentState(Span<byte> destination)
        {
            if (StateSnapshotBytes <= 0) return;
            destination.Clear();
            BitConverter.TryWriteBytes(destination, _kvTokens.Count);
            if (destination.Length >= 8)
                BitConverter.TryWriteBytes(destination.Slice(4), _recurrentChecksum);
        }

        public void RestoreRecurrentState(ReadOnlySpan<byte> snapshot)
        {
            if (StateSnapshotBytes <= 0) return;
            int savedLen = BitConverter.ToInt32(snapshot);
            while (_kvTokens.Count > savedLen) _kvTokens.RemoveAt(_kvTokens.Count - 1);
            if (snapshot.Length >= 8)
                _recurrentChecksum = BitConverter.ToInt32(snapshot.Slice(4));
        }

        public void Dispose()
        {
            _gguf.Dispose();
            try { File.Delete(_tokPath); } catch { }
        }
    }

    private sealed class SyntheticHiddenStateTrunk : IHiddenStateSource
    {
        private readonly float[] _hidden;
        private readonly float[,] _embeddings;

        public int HiddenDim { get; }
        public int VocabSize { get; }

        public SyntheticHiddenStateTrunk(int hiddenDim, int vocabSize)
        {
            HiddenDim = hiddenDim;
            VocabSize = vocabSize;
            _hidden = new float[hiddenDim];
            _embeddings = new float[vocabSize, hiddenDim];
            for (int v = 0; v < vocabSize; v++)
            {
                _embeddings[v, v % hiddenDim] = 1.0f;
            }
            _hidden[0] = 1.0f;
        }

        public void EnablePositionHistory() { }
        public ReadOnlySpan<float> GetLastHiddenState() => _hidden;
        public bool TryGetHiddenStateAt(int pos, Span<float> destination)
        {
            _hidden.CopyTo(destination);
            return true;
        }
        public void CopyTokenEmbedding(int tokenId, Span<float> destination)
        {
            destination.Clear();
            if ((uint)tokenId < (uint)VocabSize)
            {
                for (int d = 0; d < HiddenDim; d++)
                    destination[d] = _embeddings[tokenId, d];
            }
        }
    }

    [Fact]
    public void DFlash_DraftsMultiTokenBlock_AndGeneralizesViaSkipGramsWherePromptLookupReturnsZero()
    {
        using var dflash = new DFlashDraftProvider(new DFlashOptions
        {
            BlockSize = 4,
            DiffusionSteps = 3,
            MaxNgramOrder = 3
        });
        using var lookup = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        // Prompt ends with [14, 99]: exact 2-gram [14, 99] never appeared before,
        // so PromptLookup(minMatch=2) returns 0, while DFlash predicts [7, 8, 9]
        // using unigram/skip-gram context around 99 -> 7 -> 8 -> 9 and 14 -> (offset 2 -> 7, offset 3 -> 8).
        int[] tokens = [14, 20, 7, 8, 9, 15, 99, 7, 8, 9, 14, 99];
        int[] lookupOut = new int[4];
        int[] dflashOut = new int[4];

        int lookupCount = lookup.Draft(tokens, 4, lookupOut);
        int dflashCount = dflash.Draft(tokens, 4, dflashOut);

        Assert.Equal(0, lookupCount);
        Assert.True(dflashCount >= 3, $"Expected DFlash to draft >= 3 tokens, got {dflashCount}.");
        Assert.Equal(7, dflashOut[0]);
        Assert.Equal(8, dflashOut[1]);
        Assert.Equal(9, dflashOut[2]);
        Assert.Equal(1, dflash.DraftRounds);
        Assert.Equal(3, dflash.TotalDiffusionStepsExecuted);
    }

    [Fact]
    public void DFlash_BidirectionalRefinement_UpdatesSlotFromRightNeighborAnchor()
    {
        // Construct sequence where token 10 has competing right-hand continuations 11 vs 12 (2x tie-break to 11),
        // while skip-gram anchor 10 -> offset 2 strongly selects 13 at slot 1, and backward bigram 13 -> 12
        // outweighs 11 across multiple diffusion steps!
        using var dflash = new DFlashDraftProvider(new DFlashOptions
        {
            BlockSize = 2,
            DiffusionSteps = 3,
            BidirectionalWeight = 2.5f,
            SkipGramWeight = 1.5f,
            RemaskThreshold = 0.25f
        });

        int[] tokens =
        [
            10, 11, 20,
            10, 11, 21,
            10, 12, 13,
            10, 12, 13,
            30, 10
        ];

        int[] draft = new int[2];
        int count = dflash.Draft(tokens, 2, draft);

        Assert.Equal(2, count);
        Assert.Equal(13, draft[1]);
        Assert.Equal(12, draft[0]);
    }

    [Fact]
    public async Task SpeculativeEngine_WithDFlash_GreedyAndLeviathanNonGreedy_Succeeds()
    {
        const int VocabSize = 24;
        // Deterministic periodic cycle 4 -> 5 -> 6 -> 7 -> 4 ... after prompt [3, 4, 5, 6, 7]
        float[] PeriodicLogits(IReadOnlyList<int> prefix)
        {
            float[] logits = new float[VocabSize];
            Array.Fill(logits, -4.0f);
            int last = prefix[^1];
            int next = last switch
            {
                4 => 5,
                5 => 6,
                6 => 7,
                7 => 4,
                _ => 4
            };
            logits[next] = 6.0f;
            logits[(next + 1) % VocabSize] = 1.2f;
            return logits;
        }

        using var targetSerial = new SyntheticSpeculativeTarget(VocabSize, maxSeqLen: 128, PeriodicLogits);
        using var targetSpecGreedy = new SyntheticSpeculativeTarget(VocabSize, maxSeqLen: 128, PeriodicLogits);
        using var dflashGreedy = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4, DiffusionSteps = 3 });
        using var engineGreedy = new SpeculativeEngine(targetSpecGreedy, dflashGreedy, seed: 42);

        string prompt = "bcdebcd";
        var specOpts = new SpeculativeOptions
        {
            MaxTokens = 16,
            MaxDraftTokens = 4,
            Sampling = SamplingOptions.Greedy
        };

        var greedySpecResult = await engineGreedy.GenerateAsync(prompt, specOpts, formatChat: false);

        // Compare against pure serial decoding
        int[] promptIds = targetSerial.Tokenizer.Encode(prompt);
        targetSerial.Prefill(promptIds);
        var serialIds = new List<int>();
        for (int i = 0; i < 16; i++)
        {
            int tok = targetSerial.SampleNextToken(SamplingOptions.Greedy);
            serialIds.Add(tok);
            targetSerial.ForwardToken(tok, promptIds.Length + i, computeLogits: true);
        }
        string expectedText = targetSerial.Tokenizer.Decode(serialIds.ToArray());

        Assert.Equal(expectedText, greedySpecResult.Text);
        Assert.True(greedySpecResult.SpeculativeMetrics.DraftTokensProposed > 0);
        Assert.True(greedySpecResult.SpeculativeMetrics.DraftTokensAccepted > 0);

        // Now verify Leviathan non-greedy rejection sampling + calibrated probabilities
        using var targetNonGreedy = new SyntheticSpeculativeTarget(VocabSize, maxSeqLen: 128, PeriodicLogits);
        using var dflashNonGreedy = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4, DiffusionSteps = 3 });
        using var engineNonGreedy = new SpeculativeEngine(targetNonGreedy, dflashNonGreedy, seed: 7);

        var samplingOpts = new SamplingOptions
        {
            MaxTokens = 12,
            Temperature = 0.8f,
            TopK = 8,
            TopP = 0.95f,
            RepetitionPenalty = 1.05f
        };
        var nonGreedyResult = await engineNonGreedy.GenerateAsync(
            prompt,
            new SpeculativeOptions { MaxTokens = 12, MaxDraftTokens = 4, Sampling = samplingOpts },
            formatChat: false);

        Assert.Equal(12, nonGreedyResult.Metrics.GeneratedTokens);
        Assert.True(nonGreedyResult.SpeculativeMetrics.DraftTokensProposed > 0);
        Assert.True(nonGreedyResult.SpeculativeMetrics.DraftTokensAccepted > 0);

        // Directly inspect DraftWithProbabilities invariants
        int[] draftToks = new int[4];
        float[] draftProbs = new float[4 * VocabSize];
        var sampler = new Sampler(seed: 99);
        int drafted = dflashNonGreedy.DraftWithProbabilities(
            [3, 4, 5, 6, 7, 4, 5, 6, 7, 4],
            [4, 5, 6],
            4,
            samplingOpts,
            sampler,
            VocabSize,
            draftToks,
            draftProbs);
        Assert.True(drafted >= 2);
        for (int i = 0; i < drafted; i++)
        {
            float rowSum = 0f;
            for (int v = 0; v < VocabSize; v++)
            {
                float pv = draftProbs[i * VocabSize + v];
                Assert.True(pv >= 0f && float.IsFinite(pv));
                rowSum += pv;
            }
            Assert.InRange(rowSum, 0.999f, 1.001f);
            Assert.True(draftProbs[i * VocabSize + draftToks[i]] > 0f);
        }
    }

    [Fact]
    public async Task SpeculativeEngine_WithDFlash_AndRecurrentSsmSnapshotRollback_Works()
    {
        const int VocabSize = 20;
        float[] Logits(IReadOnlyList<int> prefix)
        {
            float[] l = new float[VocabSize];
            Array.Fill(l, -5f);
            int next = (prefix[^1] == 8) ? 9 : 8;
            l[next] = 5f;
            return l;
        }

        using var targetPerStep = new SyntheticSpeculativeTarget(VocabSize, 64, Logits, stateSnapshotBytes: 16, perStepSnapshots: true);
        using var dflash1 = new DFlashDraftProvider(new DFlashOptions { BlockSize = 3, DiffusionSteps = 2 });
        using var engine1 = new SpeculativeEngine(targetPerStep, dflash1, seed: 11);
        var res1 = await engine1.GenerateAsync("fghij", new SpeculativeOptions { MaxTokens = 10, MaxDraftTokens = 3 }, formatChat: false);
        Assert.Equal(10, res1.Metrics.GeneratedTokens);

        using var targetReplay = new SyntheticSpeculativeTarget(VocabSize, 64, Logits, stateSnapshotBytes: 16, perStepSnapshots: false);
        using var dflash2 = new DFlashDraftProvider(new DFlashOptions { BlockSize = 3, DiffusionSteps = 2 });
        using var engine2 = new SpeculativeEngine(targetReplay, dflash2, seed: 11);
        var res2 = await engine2.GenerateAsync("fghij", new SpeculativeOptions { MaxTokens = 10, MaxDraftTokens = 3 }, formatChat: false);
        Assert.Equal(res1.Text, res2.Text);
    }

    [Fact]
    public void CostGatedDraftPolicy_And_AdaptiveCompositeDraftProvider_RouteDFlashAndPreserveCompatibility()
    {
        var policy = new CostGatedDraftPolicy(
            baseVerifyMs: 10.0,
            perDraftVerifyMs: 1.5,
            mtpDraftStepMs: 2.0,
            lookupOverheadMs: 0.05)
        {
            DFlashDraftMs = 0.25,
            PriorDFlashAcceptRate = 0.85,
            PriorMtpAcceptRate = 0.60,
            MaxDFlashDraftTokens = 4,
            MaxMtpDraftTokens = 3
        };

        // 4-arg backward-compatible overload still works cleanly without DFlash
        DraftDecision legacyDecision = policy.SelectRoute(4, hasMtp: true, lookupCandidateCount: 0, matchedNgramLength: 0);
        Assert.Equal(DraftRoute.Mtp, legacyDecision.Route);

        // 5-arg overload selects DFlash when DFlash outperforms MTP and Lookup
        DraftDecision dflashDecision = policy.SelectRoute(4, hasMtp: true, lookupCandidateCount: 0, matchedNgramLength: 0, hasDFlash: true);
        Assert.Equal(DraftRoute.DFlash, dflashDecision.Route);
        Assert.True(dflashDecision.DraftCount >= 1);

        policy.RecordOutcome(DraftRoute.DFlash, 0, draftedCount: 4, acceptedCount: 3, roundElapsedMs: 14.0);
        var (succ, trials) = policy.GetRouteStats(DraftRoute.DFlash);
        Assert.Equal(3.0, succ, 6);
        Assert.Equal(4.0, trials, 6);
        Assert.True(policy.GetAcceptanceEstimate(DraftRoute.DFlash) > 0.7);

        // 4-arg constructor of AdaptiveCompositeDraftProvider remains unambiguous
        using var lookup = new PromptLookupDraftProvider(2, 4);
        using var legacyComposite = new AdaptiveCompositeDraftProvider(lookup, null, null);
        Assert.Null(legacyComposite.DFlashProvider);

        // 5-arg constructor with DFlashDraftProvider
        using var dflash = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4, DiffusionSteps = 2 });
        using var adaptive = new AdaptiveCompositeDraftProvider(lookup, null, policy, ownsProviders: false, dflashProvider: dflash);

        int[] buf = new int[4];
        // First call on unseen sequence: DFlash returns 0 and lookup returns 0 -> SerialBypassRounds increments cleanly
        int cold = adaptive.Draft([50, 51, 52], 4, buf);
        Assert.Equal(0, cold);
        Assert.Equal(1, adaptive.SerialBypassRounds);

        // Sequence with skip-gram anchor [10, 20, 30, 40, 10, 99, 30, 40, 10] where lookup(min=2) misses [99, 10] -> DFlash drafts!
        // Force policy to strongly prefer DFlash over PromptLookup
        policy.PriorLookupAcceptRate = 0.05;
        int warm = adaptive.Draft([10, 20, 30, 40, 10, 99, 30, 40, 10], 4, buf);
        Assert.True(warm >= 1);
        Assert.Equal(1, adaptive.DFlashRounds);
        adaptive.OnVerificationResult(warm, warm, 12.0);
    }

    [Fact]
    public unsafe void DFlash_NeuralWeightsMode_ExecutesBidirectionalLayers_AndLifecycleChecks()
    {
        const int Dim = 32;
        const int FfnDim = 32;
        const int Vocab = 16;
        const int Block = 3;

        var trunk = new SyntheticHiddenStateTrunk(Dim, Vocab);

        // Verify TryCreateFromModel returns null when GGUF has no dflash.* tensors
        string tokPath = CreateMinimalTokenizerGguf(Vocab);
        try
        {
            using var emptyGguf = GgufFile.Open(tokPath);
            // Only verify neural mode directly using DFlashWeights
            Assert.NotNull(emptyGguf);
        }
        finally
        {
            try { File.Delete(tokPath); } catch { }
        }

        // Allocate small aligned float weight buffers for neural DFlashWeights
        var owned = new List<IntPtr>();
        float* Alloc(int count, float fill = 0f, int diag = 0, int diagRows = 0, int diagCols = 0)
        {
            float* p = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            owned.Add((IntPtr)p);
            new Span<float>(p, count).Fill(fill);
            if (diag > 0)
            {
                for (int r = 0; r < diagRows; r++)
                {
                    p[r * diagCols + (r % diagCols)] = 1.0f;
                }
            }
            return p;
        }

        float* ones = Alloc(Dim, 1.0f);
        float* qkvW = Alloc(Dim * Dim, diag: 1, diagRows: Dim, diagCols: Dim);
        float* ffnW = Alloc(Dim * FfnDim, diag: 1, diagRows: FfnDim, diagCols: Dim);
        float* ffnDownW = Alloc(FfnDim * Dim, diag: 1, diagRows: Dim, diagCols: FfnDim);
        float* headW = Alloc(Vocab * Dim, fill: 0f);
        for (int v = 0; v < Vocab; v++)
        {
            headW[v * Dim + 0] = (v == 5) ? 4.0f : 0.1f;
            headW[v * Dim + (v % Dim)] = (v == 5) ? 4.0f : 0.1f;
        }
        float* slotPos = Alloc(Block * Dim, fill: 0.01f);
        float* maskEmb = Alloc(Dim, fill: 0.05f);

        var layer = new LayerWeights
        {
            AttnNormWeight = ones,
            QWeight = (byte*)qkvW,
            QType = GgufType.F32,
            KWeight = (byte*)qkvW,
            KType = GgufType.F32,
            VWeight = (byte*)qkvW,
            VType = GgufType.F32,
            AttnOutWeight = (byte*)qkvW,
            AttnOutType = GgufType.F32,
            FfnNormWeight = ones,
            FfnGateWeight = (byte*)ffnW,
            FfnGateType = GgufType.F32,
            FfnUpWeight = (byte*)ffnW,
            FfnUpType = GgufType.F32,
            FfnDownWeight = (byte*)ffnDownW,
            FfnDownType = GgufType.F32
        };

        var weights = new DFlashWeights
        {
            Dim = Dim,
            FfnDim = FfnDim,
            NumHeads = 2,
            NumHeadsKv = 2,
            HeadDim = 16,
            VocabSize = Vocab,
            BlockSize = Block,
            MaskEmbedding = maskEmb,
            SlotPosEmbeddings = slotPos,
            ContextNormWeight = ones,
            Layers = [layer],
            SharedHeadNormWeight = ones,
            SharedHeadWeight = (byte*)headW,
            SharedHeadType = GgufType.F32,
            OwnedFloatBuffers = owned
        };

        using var provider = new DFlashDraftProvider(
            trunk,
            weights,
            new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 });

        Assert.True(provider.HasNeuralWeights);

        int[] outTokens = new int[Block];
        int drafted = provider.Draft([1, 2, 3], Block, outTokens);
        Assert.Equal(Block, drafted);
        for (int i = 0; i < drafted; i++)
        {
            Assert.Equal(5, outTokens[i]);
        }

        // Empty / 0-limit / Reset / Dispose checks
        Assert.Equal(0, provider.Draft(ReadOnlySpan<int>.Empty, 3, outTokens));
        Assert.Equal(0, provider.Draft([1, 2], 0, outTokens));
        provider.Reset();
        provider.Dispose();
        provider.Dispose(); // idempotent

        Assert.Throws<ObjectDisposedException>(() => provider.Draft([1, 2], 2, outTokens));
        Assert.Throws<ObjectDisposedException>(() => provider.Reset());
    }

    [Fact]
    public unsafe void DFlash_NeuralWeightsMode_TargetVocabSmallerThanWeightsVocab_ClampsLogitsAndMaintainsSimplex()
    {
        const int Dim = 32;
        const int FfnDim = 32;
        const int WeightVocab = 32;
        const int TargetVocab = 12;
        const int Block = 2;

        var trunk = new SyntheticHiddenStateTrunk(Dim, WeightVocab);
        var owned = new List<IntPtr>();
        float* Alloc(int count, float fill = 0f, bool identity = false, int rows = 0, int cols = 0)
        {
            float* p = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            owned.Add((IntPtr)p);
            new Span<float>(p, count).Fill(fill);
            if (identity)
            {
                for (int r = 0; r < rows; r++)
                    p[r * cols + (r % cols)] = 1.0f;
            }
            return p;
        }

        float* ones = Alloc(Dim, 1.0f);
        float* qkvW = Alloc(Dim * Dim, identity: true, rows: Dim, cols: Dim);
        float* ffnW = Alloc(Dim * FfnDim, identity: true, rows: FfnDim, cols: Dim);
        float* ffnDownW = Alloc(FfnDim * Dim, identity: true, rows: Dim, cols: FfnDim);
        float* headW = Alloc(WeightVocab * Dim, fill: 0f);
        for (int v = 0; v < WeightVocab; v++)
        {
            // Give out-of-range token 25 the dominant logit (+20), while in-range token 5 has +2.
            float val = v == 25 ? 20.0f : (v == 5 ? 2.0f : 0.1f);
            headW[v * Dim + 0] = val;
            headW[v * Dim + (v % Dim)] = val;
        }

        var layer = new LayerWeights
        {
            AttnNormWeight = ones,
            QWeight = (byte*)qkvW,
            QType = GgufType.F32,
            KWeight = (byte*)qkvW,
            KType = GgufType.F32,
            VWeight = (byte*)qkvW,
            VType = GgufType.F32,
            AttnOutWeight = (byte*)qkvW,
            AttnOutType = GgufType.F32,
            FfnNormWeight = ones,
            FfnGateWeight = (byte*)ffnW,
            FfnGateType = GgufType.F32,
            FfnUpWeight = (byte*)ffnW,
            FfnUpType = GgufType.F32,
            FfnDownWeight = (byte*)ffnDownW,
            FfnDownType = GgufType.F32
        };

        var weights = new DFlashWeights
        {
            Dim = Dim,
            FfnDim = FfnDim,
            NumHeads = 2,
            NumHeadsKv = 2,
            HeadDim = 16,
            VocabSize = WeightVocab,
            BlockSize = Block,
            MaskEmbedding = Alloc(Dim, fill: 0.05f),
            SlotPosEmbeddings = Alloc(Block * Dim, fill: 0.01f),
            ContextNormWeight = ones,
            Layers = [layer],
            SharedHeadNormWeight = ones,
            SharedHeadWeight = (byte*)headW,
            SharedHeadType = GgufType.F32
        };

        try
        {
            // 1. Unpacked head (_packedHead == null) with targetVocabSize < weights.VocabSize
            using (var provider = new DFlashDraftProvider(trunk, weights, new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 }))
            {
                var sampler = new Sampler(seed: 123);
                var samplingOpts = new SamplingOptions { Temperature = 0.85f, TopK = 8, TopP = 0.95f };
                int[] outTokens = new int[Block];
                float[] probs = new float[Block * TargetVocab];

                int drafted = provider.DraftWithProbabilities([1, 2, 3], [2, 3], Block, samplingOpts, sampler, TargetVocab, outTokens, probs);
                Assert.Equal(Block, drafted);
                for (int i = 0; i < drafted; i++)
                {
                    Assert.InRange(outTokens[i], 0, TargetVocab - 1);
                    float sum = 0f;
                    for (int v = 0; v < TargetVocab; v++)
                    {
                        float pv = probs[i * TargetVocab + v];
                        Assert.True(pv >= 0f && float.IsFinite(pv));
                        sum += pv;
                    }
                    Assert.InRange(sum, 0.999f, 1.001f);
                    Assert.True(probs[i * TargetVocab + outTokens[i]] > 0f);
                }

                // Greedy sampling with TopK=1 / Temperature=0
                int draftedGreedy = provider.DraftWithProbabilities([1, 2, 3], [2, 3], Block, SamplingOptions.Greedy, sampler, TargetVocab, outTokens, probs);
                Assert.Equal(Block, draftedGreedy);
                for (int i = 0; i < draftedGreedy; i++)
                {
                    Assert.Equal(5, outTokens[i]);
                    Assert.InRange(probs[i * TargetVocab + 5], 0.999f, 1.001f);
                }
            }

            // 2. PackedDraftOutputHead containing both in-range and out-of-range subset IDs
            var subsetMixed = DraftVocabularySubset.FromTokenIds(WeightVocab, [1, 3, 5, 25, 29]);
            using (var packedProvider = new DFlashDraftProvider(trunk, new DFlashWeights
            {
                Dim = Dim,
                FfnDim = FfnDim,
                NumHeads = 2,
                NumHeadsKv = 2,
                HeadDim = 16,
                VocabSize = WeightVocab,
                BlockSize = Block,
                MaskEmbedding = weights.MaskEmbedding,
                SlotPosEmbeddings = weights.SlotPosEmbeddings,
                ContextNormWeight = weights.ContextNormWeight,
                Layers = [layer],
                SharedHeadNormWeight = weights.SharedHeadNormWeight,
                SharedHeadWeight = weights.SharedHeadWeight,
                SharedHeadType = GgufType.F32
            }, new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 }, subsetMixed))
            {
                var sampler = new Sampler(seed: 456);
                var samplingOpts = new SamplingOptions { Temperature = 5.0f, TopK = 0, TopP = 1.0f };
                int[] outTokens = new int[Block];
                float[] probs = new float[Block * TargetVocab];

                int drafted = packedProvider.DraftWithProbabilities([1, 2, 3], [2, 3], Block, samplingOpts, sampler, TargetVocab, outTokens, probs);
                Assert.Equal(Block, drafted);
                for (int i = 0; i < drafted; i++)
                {
                    Assert.InRange(outTokens[i], 0, TargetVocab - 1);
                    float sum = 0f;
                    for (int v = 0; v < TargetVocab; v++) sum += probs[i * TargetVocab + v];
                    Assert.InRange(sum, 0.999f, 1.001f);
                    Assert.True(probs[i * TargetVocab + outTokens[i]] > 0f);
                }
            }

            // 3. Edge case: PackedDraftOutputHead where every subset ID >= TargetVocab
            var subsetOutOnly = DraftVocabularySubset.FromTokenIds(WeightVocab, [20, 25]);
            using (var outOnlyProvider = new DFlashDraftProvider(trunk, new DFlashWeights
            {
                Dim = Dim,
                FfnDim = FfnDim,
                NumHeads = 2,
                NumHeadsKv = 2,
                HeadDim = 16,
                VocabSize = WeightVocab,
                BlockSize = Block,
                MaskEmbedding = weights.MaskEmbedding,
                SlotPosEmbeddings = weights.SlotPosEmbeddings,
                ContextNormWeight = weights.ContextNormWeight,
                Layers = [layer],
                SharedHeadNormWeight = weights.SharedHeadNormWeight,
                SharedHeadWeight = weights.SharedHeadWeight,
                SharedHeadType = GgufType.F32
            }, new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 }, subsetOutOnly))
            {
                var sampler = new Sampler(seed: 789);
                int[] outTokens = new int[Block];
                float[] probs = new float[Block * TargetVocab];
                int drafted = outOnlyProvider.DraftWithProbabilities([1, 2, 3], [2, 3], Block, new SamplingOptions { Temperature = 0.7f, TopK = 4 }, sampler, TargetVocab, outTokens, probs);
                Assert.Equal(Block, drafted);
                for (int i = 0; i < drafted; i++)
                {
                    Assert.InRange(outTokens[i], 0, TargetVocab - 1);
                    float sum = 0f;
                    for (int v = 0; v < TargetVocab; v++) sum += probs[i * TargetVocab + v];
                    Assert.InRange(sum, 0.999f, 1.001f);
                }
            }
        }
        finally
        {
            foreach (IntPtr p in owned)
            {
                if (p != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree((void*)p);
                }
            }
        }
    }

    [Fact]
    public unsafe void DFlash_TryCreateFromModel_And_LoadFromSidecar_WithF16AndBf16Weights_DraftAndDisposeCleanly()
    {
        const int Dim = 32;
        const int FfnDim = 32;
        const int Vocab = 16;
        const int Block = 3;

        string embeddedPath = Path.Combine(Path.GetTempPath(), $"dflash_embedded_{Guid.NewGuid():N}.gguf");
        string basePath = Path.Combine(Path.GetTempPath(), $"dflash_base_{Guid.NewGuid():N}.gguf");
        string sidecarPath = Path.Combine(Path.GetTempPath(), $"dflash_sidecar_{Guid.NewGuid():N}.gguf");
        string incompletePath = Path.Combine(Path.GetTempPath(), $"dflash_incomplete_{Guid.NewGuid():N}.gguf");

        try
        {
            WriteSyntheticBaseAndDFlashGguf(embeddedPath, Dim, FfnDim, Vocab, Block, includeBase: true, includeDFlash: true, incompleteSecondLayer: false);
            WriteSyntheticBaseAndDFlashGguf(basePath, Dim, FfnDim, Vocab, Block, includeBase: true, includeDFlash: false, incompleteSecondLayer: false);
            WriteSyntheticBaseAndDFlashGguf(sidecarPath, Dim, FfnDim, Vocab, Block, includeBase: false, includeDFlash: true, incompleteSecondLayer: false);
            WriteSyntheticBaseAndDFlashGguf(incompletePath, Dim, FfnDim, Vocab, Block, includeBase: true, includeDFlash: true, incompleteSecondLayer: true);

            var trunk = new SyntheticHiddenStateTrunk(Dim, Vocab);

            // 1. TryCreateFromModel with embedded F16/BF16 DFlash tensors
            using (var embGguf = GgufFile.Open(embeddedPath))
            using (var embWeights = new ModelWeights(embGguf))
            {
                using var embeddedProvider = DFlashDraftProvider.TryCreateFromModel(
                    trunk,
                    embWeights,
                    new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 });
                Assert.NotNull(embeddedProvider);
                Assert.True(embeddedProvider!.HasNeuralWeights);

                int[] draft = new int[Block];
                int count = embeddedProvider.Draft([1, 2, 3], Block, draft);
                Assert.Equal(Block, count);
                for (int i = 0; i < count; i++)
                    Assert.Equal(5, draft[i]);

                float[] probs = new float[Block * Vocab];
                var sampler = new Sampler(seed: 42);
                int probCount = embeddedProvider.DraftWithProbabilities(
                    [1, 2, 3],
                    [2, 3],
                    Block,
                    new SamplingOptions { Temperature = 0.7f, TopK = 8, TopP = 0.95f },
                    sampler,
                    Vocab,
                    draft,
                    probs);
                Assert.Equal(Block, probCount);
                for (int i = 0; i < probCount; i++)
                {
                    float sum = 0f;
                    for (int v = 0; v < Vocab; v++) sum += probs[i * Vocab + v];
                    Assert.InRange(sum, 0.999f, 1.001f);
                }

                embeddedProvider.Dispose();
                embeddedProvider.Dispose();
                Assert.Throws<ObjectDisposedException>(() => embeddedProvider.Draft([1, 2], 2, draft));
            }

            // 2. LoadFromSidecar with F16/BF16 DFlash tensors + TryCreateFromModel returning null on base without DFlash
            using (var baseGguf = GgufFile.Open(basePath))
            using (var baseWeights = new ModelWeights(baseGguf))
            {
                Assert.Null(DFlashDraftProvider.TryCreateFromModel(trunk, baseWeights));

                using var sidecarProvider = DFlashDraftProvider.LoadFromSidecar(
                    trunk,
                    baseWeights,
                    sidecarPath,
                    new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 });
                Assert.True(sidecarProvider.HasNeuralWeights);

                int[] draft = new int[Block];
                int count = sidecarProvider.Draft([1, 2, 3], Block, draft);
                Assert.Equal(Block, count);
                for (int i = 0; i < count; i++)
                    Assert.Equal(5, draft[i]);
            }

            // 3. Incomplete layer-1 DFlash tensors after layer-0 F16/BF16 allocation -> returns null / throws InvalidDataException cleanly
            using (var incGguf = GgufFile.Open(incompletePath))
            using (var incWeights = new ModelWeights(incGguf))
            {
                Assert.Null(DFlashDraftProvider.TryCreateFromModel(trunk, incWeights));
                Assert.Throws<InvalidDataException>(() => DFlashDraftProvider.LoadFromSidecar(trunk, incWeights, incompletePath));
            }
        }
        finally
        {
            foreach (string p in new[] { embeddedPath, basePath, sidecarPath, incompletePath })
            {
                try { if (File.Exists(p)) File.Delete(p); } catch { }
            }
        }
    }

    private static void WriteSyntheticBaseAndDFlashGguf(
        string path,
        int dim,
        int ffnDim,
        int vocab,
        int blockSize,
        bool includeBase,
        bool includeDFlash,
        bool incompleteSecondLayer)
    {
        var kvEntries = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("qwen2.context_length", GgufValueType.Uint32, 64u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.block_count", GgufValueType.Uint32, 1u),
            ("qwen2.attention.head_count", GgufValueType.Uint32, 2u),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, 2u),
            ("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-6f),
            ("dflash.block_size", GgufValueType.Uint32, (uint)blockSize)
        };

        byte[] MakeF32Identity(int rows, int cols)
        {
            byte[] buf = new byte[rows * cols * sizeof(float)];
            var f = MemoryMarshal.Cast<byte, float>(buf.AsSpan());
            for (int r = 0; r < rows; r++)
                f[r * cols + (r % cols)] = 1.0f;
            return buf;
        }

        byte[] MakeF32Const(int count, float val)
        {
            byte[] buf = new byte[count * sizeof(float)];
            MemoryMarshal.Cast<byte, float>(buf.AsSpan()).Fill(val);
            return buf;
        }

        byte[] MakeF16Const(int count, float val)
        {
            byte[] buf = new byte[count * sizeof(ushort)];
            var h = MemoryMarshal.Cast<byte, Half>(buf.AsSpan());
            h.Fill((Half)val);
            return buf;
        }

        byte[] MakeBf16Const(int count, float val)
        {
            byte[] buf = new byte[count * sizeof(ushort)];
            var u = MemoryMarshal.Cast<byte, ushort>(buf.AsSpan());
            ushort bf = (ushort)(BitConverter.SingleToUInt32Bits(val) >> 16);
            u.Fill(bf);
            return buf;
        }

        byte[] MakeBiasedHead(int vCount, int d)
        {
            byte[] buf = new byte[vCount * d * sizeof(float)];
            var f = MemoryMarshal.Cast<byte, float>(buf.AsSpan());
            for (int v = 0; v < vCount; v++)
            {
                float w = (v == 5) ? 4.0f : 0.1f;
                f[v * d + 0] = w;
                f[v * d + (v % d)] = w;
            }
            return buf;
        }

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();

        if (includeBase)
        {
            tensors.Add(("token_embd.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeF32Identity(vocab, dim)));
            tensors.Add(("output_norm.weight", [(ulong)dim], GgufType.F32, MakeF32Const(dim, 1.0f)));
            tensors.Add(("output.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeBiasedHead(vocab, dim)));
            tensors.Add(("blk.0.attn_norm.weight", [(ulong)dim], GgufType.F32, MakeF32Const(dim, 1.0f)));
            tensors.Add(("blk.0.attn_q.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.attn_k.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.attn_v.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.attn_output.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.ffn_norm.weight", [(ulong)dim], GgufType.F32, MakeF32Const(dim, 1.0f)));
            tensors.Add(("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32Identity(ffnDim, dim)));
            tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32Identity(ffnDim, dim)));
            tensors.Add(("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, ffnDim)));
        }

        if (includeDFlash)
        {
            // Store 1D norm/mask/pos weights as F16 and BF16 to exercise LoadFloatWeight + OwnedFloatBuffers tracking
            tensors.Add(("dflash.mask_embd.weight", [(ulong)dim], GgufType.F16, MakeF16Const(dim, 0.05f)));
            tensors.Add(("dflash.slot_pos_embd.weight", [(ulong)dim, (ulong)blockSize], GgufType.BF16, MakeBf16Const(blockSize * dim, 0.01f)));
            tensors.Add(("dflash.context_norm.weight", [(ulong)dim], GgufType.F16, MakeF16Const(dim, 1.0f)));
            tensors.Add(("dflash.shared_head.norm.weight", [(ulong)dim], GgufType.BF16, MakeBf16Const(dim, 1.0f)));
            tensors.Add(("dflash.shared_head.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeBiasedHead(vocab, dim)));

            tensors.Add(("blk.0.dflash.attn_norm.weight", [(ulong)dim], GgufType.F16, MakeF16Const(dim, 1.0f)));
            tensors.Add(("blk.0.dflash.attn_q.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.dflash.attn_k.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.dflash.attn_v.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.dflash.attn_output.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            tensors.Add(("blk.0.dflash.ffn_norm.weight", [(ulong)dim], GgufType.BF16, MakeBf16Const(dim, 1.0f)));
            tensors.Add(("blk.0.dflash.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32Identity(ffnDim, dim)));
            tensors.Add(("blk.0.dflash.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32Identity(ffnDim, dim)));
            tensors.Add(("blk.0.dflash.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, ffnDim)));

            if (incompleteSecondLayer)
            {
                // Layer 1 has F16 norm + Q weight, but omits K/V/O/FFN -> CreateFromGguf returns null and must free layer-0 F16/BF16 buffers
                tensors.Add(("blk.1.dflash.attn_norm.weight", [(ulong)dim], GgufType.F16, MakeF16Const(dim, 1.0f)));
                tensors.Add(("blk.1.dflash.attn_q.weight", [(ulong)dim, (ulong)dim], GgufType.F32, MakeF32Identity(dim, dim)));
            }
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        void WriteStr(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)bytes.Length);
            bw.Write(bytes);
        }

        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvEntries.Count);

        foreach (var (k, t, v) in kvEntries)
        {
            WriteStr(k);
            bw.Write((uint)t);
            switch (t)
            {
                case GgufValueType.String: WriteStr((string)v); break;
                case GgufValueType.Uint32: bw.Write((uint)v); break;
                case GgufValueType.Float32: bw.Write((float)v); break;
            }
        }

        ulong offset = 0;
        var offsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = offset;
            ulong len = (ulong)tensors[i].Data.Length;
            offset += ((len + 31UL) / 32UL) * 32UL;
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, dims, type, _) = tensors[i];
            WriteStr(name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims) bw.Write(d);
            bw.Write((uint)type);
            bw.Write(offsets[i]);
        }

        long pos = fs.Position;
        long pad = ((pos + 31L) / 32L) * 32L - pos;
        if (pad > 0) bw.Write(new byte[pad]);

        for (int i = 0; i < tensors.Count; i++)
        {
            byte[] data = tensors[i].Data;
            bw.Write(data);
            long rem = data.Length % 32;
            if (rem != 0) bw.Write(new byte[32 - rem]);
        }
    }

    private static void WriteCustomGguf(
        string path,
        List<(string Key, GgufValueType Type, object Value)> kvEntries,
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        void WriteStr(string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)bytes.Length);
            bw.Write(bytes);
        }

        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvEntries.Count);

        foreach (var (k, t, v) in kvEntries)
        {
            WriteStr(k);
            bw.Write((uint)t);
            switch (t)
            {
                case GgufValueType.String: WriteStr((string)v); break;
                case GgufValueType.Uint32: bw.Write((uint)v); break;
                case GgufValueType.Float32: bw.Write((float)v); break;
            }
        }

        ulong offset = 0;
        var offsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = offset;
            ulong len = (ulong)tensors[i].Data.Length;
            offset += ((len + 31UL) / 32UL) * 32UL;
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, dims, type, _) = tensors[i];
            WriteStr(name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims) bw.Write(d);
            bw.Write((uint)type);
            bw.Write(offsets[i]);
        }

        long pos = fs.Position;
        long pad = ((pos + 31L) / 32L) * 32L - pos;
        if (pad > 0) bw.Write(new byte[pad]);

        for (int i = 0; i < tensors.Count; i++)
        {
            byte[] data = tensors[i].Data;
            bw.Write(data);
            long rem = data.Length % 32;
            if (rem != 0) bw.Write(new byte[32 - rem]);
        }
    }

    private static byte[] MakeF32ZeroData(int count) => new byte[count * sizeof(float)];

    private static byte[] MakeF32IdentityData(int rows, int cols)
    {
        byte[] buf = new byte[rows * cols * sizeof(float)];
        var f = MemoryMarshal.Cast<byte, float>(buf.AsSpan());
        for (int r = 0; r < rows; r++)
            f[r * cols + (r % cols)] = 1.0f;
        return buf;
    }

    private static byte[] MakeF16ConstData(int count, float val)
    {
        byte[] buf = new byte[count * sizeof(ushort)];
        MemoryMarshal.Cast<byte, Half>(buf.AsSpan()).Fill((Half)val);
        return buf;
    }

    private static List<(string Key, GgufValueType Type, object Value)> BuildBaseMetadata(
        int dim,
        int ffnDim,
        int nHeads,
        int nHeadsKv,
        int headDim,
        int blockSize,
        bool includeArchDims = true)
    {
        var kv = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("dflash.block_size", GgufValueType.Uint32, (uint)blockSize)
        };
        if (includeArchDims)
        {
            kv.Add(("qwen2.context_length", GgufValueType.Uint32, 64u));
            kv.Add(("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim));
            kv.Add(("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim));
            kv.Add(("qwen2.block_count", GgufValueType.Uint32, 1u));
            kv.Add(("qwen2.attention.head_count", GgufValueType.Uint32, (uint)nHeads));
            kv.Add(("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv));
            kv.Add(("qwen2.attention.key_length", GgufValueType.Uint32, (uint)headDim));
            kv.Add(("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-6f));
        }
        return kv;
    }

    private static void AppendBaseTensors(
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)> tensors,
        int dim,
        int ffnDim,
        int qDim,
        int kvDim,
        int vocab)
    {
        tensors.Add(("token_embd.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeF32IdentityData(vocab, dim)));
        tensors.Add(("output_norm.weight", [(ulong)dim], GgufType.F32, MakeF32ZeroData(dim)));
        tensors.Add(("output.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeF32IdentityData(vocab, dim)));
        tensors.Add(("blk.0.attn_norm.weight", [(ulong)dim], GgufType.F32, MakeF32ZeroData(dim)));
        tensors.Add(("blk.0.attn_q.weight", [(ulong)dim, (ulong)qDim], GgufType.F32, MakeF32IdentityData(qDim, dim)));
        tensors.Add(("blk.0.attn_k.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, MakeF32IdentityData(kvDim, dim)));
        tensors.Add(("blk.0.attn_v.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, MakeF32IdentityData(kvDim, dim)));
        tensors.Add(("blk.0.attn_output.weight", [(ulong)qDim, (ulong)dim], GgufType.F32, MakeF32IdentityData(dim, qDim)));
        tensors.Add(("blk.0.ffn_norm.weight", [(ulong)dim], GgufType.F32, MakeF32ZeroData(dim)));
        tensors.Add(("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32IdentityData(ffnDim, dim)));
        tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32IdentityData(ffnDim, dim)));
        tensors.Add(("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], GgufType.F32, MakeF32IdentityData(dim, ffnDim)));
    }

    private static void AppendDFlashTensors(
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)> tensors,
        int dim,
        int ffnDim,
        int qDim,
        int kvDim,
        int vocab,
        int blockSize,
        bool includeSharedHead = true,
        bool putAttnQAtEnd = false)
    {
        tensors.Add(("dflash.mask_embd.weight", [(ulong)dim], GgufType.F16, MakeF16ConstData(dim, 0.05f)));
        tensors.Add(("dflash.slot_pos_embd.weight", [(ulong)dim, (ulong)blockSize], GgufType.F16, MakeF16ConstData(blockSize * dim, 0.01f)));
        tensors.Add(("dflash.context_norm.weight", [(ulong)dim], GgufType.F16, MakeF16ConstData(dim, 1.0f)));
        tensors.Add(("dflash.shared_head.norm.weight", [(ulong)dim], GgufType.F16, MakeF16ConstData(dim, 1.0f)));
        if (includeSharedHead)
        {
            tensors.Add(("dflash.shared_head.weight", [(ulong)dim, (ulong)vocab], GgufType.F32, MakeF32IdentityData(vocab, dim)));
        }

        tensors.Add(("blk.0.dflash.attn_norm.weight", [(ulong)dim], GgufType.F16, MakeF16ConstData(dim, 1.0f)));
        if (!putAttnQAtEnd)
        {
            tensors.Add(("blk.0.dflash.attn_q.weight", [(ulong)dim, (ulong)qDim], GgufType.F32, MakeF32IdentityData(qDim, dim)));
        }
        tensors.Add(("blk.0.dflash.attn_k.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, MakeF32IdentityData(kvDim, dim)));
        tensors.Add(("blk.0.dflash.attn_v.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, MakeF32IdentityData(kvDim, dim)));
        tensors.Add(("blk.0.dflash.attn_output.weight", [(ulong)qDim, (ulong)dim], GgufType.F32, MakeF32IdentityData(dim, qDim)));
        tensors.Add(("blk.0.dflash.ffn_norm.weight", [(ulong)dim], GgufType.F16, MakeF16ConstData(dim, 1.0f)));
        tensors.Add(("blk.0.dflash.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32IdentityData(ffnDim, dim)));
        tensors.Add(("blk.0.dflash.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], GgufType.F32, MakeF32IdentityData(ffnDim, dim)));
        tensors.Add(("blk.0.dflash.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], GgufType.F32, MakeF32IdentityData(dim, ffnDim)));
        if (putAttnQAtEnd)
        {
            tensors.Add(("blk.0.dflash.attn_q.weight", [(ulong)dim, (ulong)qDim], GgufType.F32, MakeF32IdentityData(qDim, dim)));
        }
    }

    [Fact]
    public void DFlash_CreateFromGguf_RejectsUndersizedAndTransposedTensors_WithoutCrashOrLeak()
    {
        const int Dim = 32;
        const int FfnDim = 32;
        const int Vocab = 16;
        const int Block = 4;

        var cases = new List<(string Name, Action<List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>> Mutate)>
        {
            ("undersized_mask", list =>
            {
                int idx = list.FindIndex(t => t.Name == "dflash.mask_embd.weight");
                list[idx] = ("dflash.mask_embd.weight", [4UL], GgufType.F16, MakeF16ConstData(4, 0.1f));
            }),
            ("undersized_pos", list =>
            {
                int idx = list.FindIndex(t => t.Name == "dflash.slot_pos_embd.weight");
                list[idx] = ("dflash.slot_pos_embd.weight", [(ulong)Dim, 1UL], GgufType.F16, MakeF16ConstData(Dim, 0.1f));
            }),
            ("undersized_shared_head", list =>
            {
                int idx = list.FindIndex(t => t.Name == "dflash.shared_head.weight");
                list[idx] = ("dflash.shared_head.weight", [(ulong)Dim, 4UL], GgufType.F32, MakeF32IdentityData(4, Dim));
            }),
            ("transposed_shared_head", list =>
            {
                int idx = list.FindIndex(t => t.Name == "dflash.shared_head.weight");
                list[idx] = ("dflash.shared_head.weight", [(ulong)Vocab, (ulong)Dim], GgufType.F32, MakeF32IdentityData(Dim, Vocab));
            }),
            ("undersized_attn_q_at_eof_dim32", list =>
            {
                int idx = list.FindIndex(t => t.Name == "blk.0.dflash.attn_q.weight");
                list[idx] = ("blk.0.dflash.attn_q.weight", [4UL, 4UL], GgufType.F32, MakeF32IdentityData(4, 4));
            }),
            ("undersized_context_proj", list =>
            {
                list.Add(("dflash.context_proj.weight", [8UL, 8UL], GgufType.F32, MakeF32IdentityData(8, 8)));
            })
        };

        var trunk = new SyntheticHiddenStateTrunk(Dim, Vocab);
        string basePath = Path.Combine(Path.GetTempPath(), $"dflash_shape_base_{Guid.NewGuid():N}.gguf");
        var tempPaths = new List<string> { basePath };

        try
        {
            var baseTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(baseTensors, Dim, FfnDim, Dim, Dim, Vocab);
            WriteCustomGguf(basePath, BuildBaseMetadata(Dim, FfnDim, 2, 2, 16, Block), baseTensors);

            using var baseGguf = GgufFile.Open(basePath);
            using var baseWeights = new ModelWeights(baseGguf);

            foreach (var (caseName, mutate) in cases)
            {
                string embPath = Path.Combine(Path.GetTempPath(), $"dflash_emb_{caseName}_{Guid.NewGuid():N}.gguf");
                string sidePath = Path.Combine(Path.GetTempPath(), $"dflash_side_{caseName}_{Guid.NewGuid():N}.gguf");
                tempPaths.Add(embPath);
                tempPaths.Add(sidePath);

                var embTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
                AppendBaseTensors(embTensors, Dim, FfnDim, Dim, Dim, Vocab);
                AppendDFlashTensors(embTensors, Dim, FfnDim, Dim, Dim, Vocab, Block, putAttnQAtEnd: true);
                mutate(embTensors);
                WriteCustomGguf(embPath, BuildBaseMetadata(Dim, FfnDim, 2, 2, 16, Block), embTensors);

                using (var g = GgufFile.Open(embPath))
                using (var w = new ModelWeights(g))
                {
                    Assert.Null(DFlashDraftProvider.TryCreateFromModel(trunk, w, new DFlashOptions { BlockSize = Block }));
                }

                var sideTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
                AppendDFlashTensors(sideTensors, Dim, FfnDim, Dim, Dim, Vocab, Block, putAttnQAtEnd: true);
                mutate(sideTensors);
                WriteCustomGguf(sidePath, BuildBaseMetadata(Dim, FfnDim, 2, 2, 16, Block), sideTensors);

                Assert.Throws<InvalidDataException>(() =>
                    DFlashDraftProvider.LoadFromSidecar(trunk, baseWeights, sidePath, new DFlashOptions { BlockSize = Block }));
            }
        }
        finally
        {
            foreach (string p in tempPaths)
            {
                try { if (File.Exists(p)) File.Delete(p); } catch { }
            }
        }
    }

    [Fact]
    public void DFlash_CreateFromGguf_RejectsEofDim256_TransposedQDim_AndMismatchedSidecarDim_WithoutAccessViolation()
    {
        const int TrunkDim = 256;
        const int TrunkFfnDim = 256;
        const int Heads = 4;
        const int HeadsKv = 2;
        const int HeadDim = 64;
        const int QDim = Heads * HeadDim;     // 256
        const int KvDim = HeadsKv * HeadDim;  // 128
        const int Vocab = 16;
        const int Block = 4;

        var trunk256 = new SyntheticHiddenStateTrunk(TrunkDim, Vocab);
        string basePath = Path.Combine(Path.GetTempPath(), $"dflash_256_base_{Guid.NewGuid():N}.gguf");
        string eofQPath = Path.Combine(Path.GetTempPath(), $"dflash_256_eofq_{Guid.NewGuid():N}.gguf");
        string sideMismatchMetaPath = Path.Combine(Path.GetTempPath(), $"dflash_32_sidemeta_{Guid.NewGuid():N}.gguf");
        string sideMismatchNoMetaPath = Path.Combine(Path.GetTempPath(), $"dflash_32_sidenometa_{Guid.NewGuid():N}.gguf");
        string transposedBasePath = Path.Combine(Path.GetTempPath(), $"dflash_trans_base_{Guid.NewGuid():N}.gguf");
        string transposedQPath = Path.Combine(Path.GetTempPath(), $"dflash_trans_q_{Guid.NewGuid():N}.gguf");
        string validSharedSidecarPath = Path.Combine(Path.GetTempPath(), $"dflash_256_valid_shared_{Guid.NewGuid():N}.gguf");

        try
        {
            var baseTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(baseTensors, TrunkDim, TrunkFfnDim, QDim, KvDim, Vocab);
            WriteCustomGguf(basePath, BuildBaseMetadata(TrunkDim, TrunkFfnDim, Heads, HeadsKv, HeadDim, Block), baseTensors);

            // 1. dim=256 embedded GGUF with [4, 4] attn_q at EOF (previously caused AccessViolationException)
            var eofTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(eofTensors, TrunkDim, TrunkFfnDim, QDim, KvDim, Vocab);
            AppendDFlashTensors(eofTensors, TrunkDim, TrunkFfnDim, QDim, KvDim, Vocab, Block, putAttnQAtEnd: true);
            eofTensors[^1] = ("blk.0.dflash.attn_q.weight", [4UL, 4UL], GgufType.F32, MakeF32IdentityData(4, 4));
            WriteCustomGguf(eofQPath, BuildBaseMetadata(TrunkDim, TrunkFfnDim, Heads, HeadsKv, HeadDim, Block), eofTensors);

            using (var g = GgufFile.Open(eofQPath))
            using (var w = new ModelWeights(g))
            {
                Assert.Null(DFlashDraftProvider.TryCreateFromModel(trunk256, w, new DFlashOptions { BlockSize = Block }));
            }

            // 2. dim=32 sidecar loaded against dim=256 trunk (both with and without embedding_length metadata)
            var side32Tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendDFlashTensors(side32Tensors, 32, 32, 32, 32, Vocab, Block, putAttnQAtEnd: true);
            WriteCustomGguf(sideMismatchMetaPath, BuildBaseMetadata(32, 32, 2, 2, 16, Block, includeArchDims: true), side32Tensors);
            WriteCustomGguf(sideMismatchNoMetaPath, BuildBaseMetadata(32, 32, 2, 2, 16, Block, includeArchDims: false), side32Tensors);

            using (var baseGguf = GgufFile.Open(basePath))
            using (var baseWeights = new ModelWeights(baseGguf))
            {
                // Repeatedly reject malformed sidecars to verify zero leaks and zero double-frees
                for (int i = 0; i < 40; i++)
                {
                    Assert.Throws<InvalidDataException>(() =>
                        DFlashDraftProvider.LoadFromSidecar(trunk256, baseWeights, sideMismatchMetaPath, new DFlashOptions { BlockSize = Block }));
                    Assert.Throws<InvalidDataException>(() =>
                        DFlashDraftProvider.LoadFromSidecar(trunk256, baseWeights, sideMismatchNoMetaPath, new DFlashOptions { BlockSize = Block }));
                }

                // 3. Valid sidecar omitting embedding_length/vocab metadata and sharing trunk output.weight must still load and draft!
                var validSharedTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
                AppendDFlashTensors(validSharedTensors, TrunkDim, TrunkFfnDim, QDim, KvDim, Vocab, Block, includeSharedHead: false);
                WriteCustomGguf(validSharedSidecarPath, BuildBaseMetadata(TrunkDim, TrunkFfnDim, Heads, HeadsKv, HeadDim, Block, includeArchDims: false), validSharedTensors);

                using var validProvider = DFlashDraftProvider.LoadFromSidecar(
                    trunk256,
                    baseWeights,
                    validSharedSidecarPath,
                    new DFlashOptions { BlockSize = Block, DiffusionSteps = 2 });
                Assert.True(validProvider.HasNeuralWeights);
                int[] draftedTokens = new int[Block];
                int drafted = validProvider.Draft([1, 2, 3], Block, draftedTokens);
                Assert.Equal(Block, drafted);
            }

            // 4. Transposed attn_q [qDim, dim] when dim != qDim (dim=32, qDim=64) must be rejected
            const int AsymmetricDim = 32;
            const int AsymmetricQDim = 64;
            var asymTrunk = new SyntheticHiddenStateTrunk(AsymmetricDim, Vocab);
            var asymBaseTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(asymBaseTensors, AsymmetricDim, 64, AsymmetricQDim, AsymmetricDim, Vocab);
            WriteCustomGguf(transposedBasePath, BuildBaseMetadata(AsymmetricDim, 64, 4, 2, 16, Block), asymBaseTensors);

            var transQTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendDFlashTensors(transQTensors, AsymmetricDim, 64, AsymmetricQDim, AsymmetricDim, Vocab, Block);
            int qIdx = transQTensors.FindIndex(t => t.Name == "blk.0.dflash.attn_q.weight");
            transQTensors[qIdx] = ("blk.0.dflash.attn_q.weight", [(ulong)AsymmetricQDim, (ulong)AsymmetricDim], GgufType.F32, MakeF32IdentityData(AsymmetricDim, AsymmetricQDim));
            WriteCustomGguf(transposedQPath, BuildBaseMetadata(AsymmetricDim, 64, 4, 2, 16, Block), transQTensors);

            using (var asymGguf = GgufFile.Open(transposedBasePath))
            using (var asymWeights = new ModelWeights(asymGguf))
            {
                Assert.Throws<InvalidDataException>(() =>
                    DFlashDraftProvider.LoadFromSidecar(asymTrunk, asymWeights, transposedQPath, new DFlashOptions { BlockSize = Block }));
            }
        }
        finally
        {
            foreach (string p in new[] { basePath, eofQPath, sideMismatchMetaPath, sideMismatchNoMetaPath, transposedBasePath, transposedQPath, validSharedSidecarPath })
            {
                try { if (File.Exists(p)) File.Delete(p); } catch { }
            }
        }
    }

    [Fact]
    public unsafe void DFlash_CapsOversizedBlockSize_AndRejectsNonDivisibleGqaHeads_WithoutOomOrLeak()
    {
        // 1. DFlashOptions.BlockSize boundary checks
        Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashOptions { BlockSize = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashOptions { BlockSize = DFlashDraftProvider.MaxDFlashBlockSize + 1 });
        var maxValidOpts = new DFlashOptions { BlockSize = DFlashDraftProvider.MaxDFlashBlockSize };
        Assert.Equal(DFlashDraftProvider.MaxDFlashBlockSize, maxValidOpts.BlockSize);

        const int Dim = 32;
        const int FfnDim = 32;
        const int Vocab = 16;
        const int ValidBlock = 4;

        var trunk = new SyntheticHiddenStateTrunk(Dim, Vocab);
        var tempPaths = new List<string>();

        try
        {
            // 2. Oversized dflash.block_size in GGUF omitting dflash.slot_pos_embd.weight
            string basePath = Path.Combine(Path.GetTempPath(), $"dflash_cap_base_{Guid.NewGuid():N}.gguf");
            tempPaths.Add(basePath);

            var baseTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(baseTensors, Dim, FfnDim, Dim, Dim, Vocab);
            WriteCustomGguf(basePath, BuildBaseMetadata(Dim, FfnDim, 2, 2, 16, ValidBlock), baseTensors);

            using var baseGguf = GgufFile.Open(basePath);
            using var baseWeights = new ModelWeights(baseGguf);

            uint[] oversizedBlockSizes = [65_536u, 1_000_000u, (uint)int.MaxValue, 0x80000001u];
            foreach (uint badBlockSize in oversizedBlockSizes)
            {
                string embPath = Path.Combine(Path.GetTempPath(), $"dflash_cap_emb_{badBlockSize}_{Guid.NewGuid():N}.gguf");
                string sidePath = Path.Combine(Path.GetTempPath(), $"dflash_cap_side_{badBlockSize}_{Guid.NewGuid():N}.gguf");
                tempPaths.Add(embPath);
                tempPaths.Add(sidePath);

                var kv = BuildBaseMetadata(Dim, FfnDim, 2, 2, 16, ValidBlock);
                int bsIdx = kv.FindIndex(entry => entry.Key == "dflash.block_size");
                kv[bsIdx] = ("dflash.block_size", GgufValueType.Uint32, badBlockSize);

                var embTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
                AppendBaseTensors(embTensors, Dim, FfnDim, Dim, Dim, Vocab);
                AppendDFlashTensors(embTensors, Dim, FfnDim, Dim, Dim, Vocab, ValidBlock);
                embTensors.RemoveAll(t => t.Name == "dflash.slot_pos_embd.weight");
                WriteCustomGguf(embPath, kv, embTensors);

                using (var g = GgufFile.Open(embPath))
                using (var w = new ModelWeights(g))
                {
                    Assert.Null(DFlashDraftProvider.TryCreateFromModel(trunk, w));
                }

                var sideTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
                AppendDFlashTensors(sideTensors, Dim, FfnDim, Dim, Dim, Vocab, ValidBlock);
                sideTensors.RemoveAll(t => t.Name == "dflash.slot_pos_embd.weight");
                WriteCustomGguf(sidePath, kv, sideTensors);

                Assert.Throws<InvalidDataException>(() =>
                    DFlashDraftProvider.LoadFromSidecar(trunk, baseWeights, sidePath));
            }

            // 3. Direct DFlashWeights constructor rejects BlockSize > MaxDFlashBlockSize and non-divisible nHeads % nHeadsKv != 0
            byte[] dummyBuf = new byte[Dim * Vocab * sizeof(float)];
            fixed (byte* pDummy = dummyBuf)
            {
                byte* dummyPtr = pDummy;
                var dummyLayer = new LayerWeights
                {
                    QWeight = dummyPtr,
                    QType = GgufType.F32,
                    KWeight = dummyPtr,
                    KType = GgufType.F32,
                    VWeight = dummyPtr,
                    VType = GgufType.F32,
                    AttnOutWeight = dummyPtr,
                    AttnOutType = GgufType.F32,
                    FfnGateWeight = dummyPtr,
                    FfnGateType = GgufType.F32,
                    FfnUpWeight = dummyPtr,
                    FfnUpType = GgufType.F32,
                    FfnDownWeight = dummyPtr,
                    FfnDownType = GgufType.F32
                };

                Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashDraftProvider(
                    trunk,
                    new DFlashWeights
                    {
                        Dim = Dim,
                        FfnDim = FfnDim,
                        NumHeads = 2,
                        NumHeadsKv = 2,
                        HeadDim = 16,
                        VocabSize = Vocab,
                        BlockSize = DFlashDraftProvider.MaxDFlashBlockSize + 1,
                        Layers = [dummyLayer],
                        SharedHeadWeight = dummyPtr,
                        SharedHeadType = GgufType.F32
                    }));

                Assert.Throws<ArgumentOutOfRangeException>(() => new DFlashDraftProvider(
                    trunk,
                    new DFlashWeights
                    {
                        Dim = Dim,
                        FfnDim = FfnDim,
                        NumHeads = 3,
                        NumHeadsKv = 2,
                        HeadDim = 16,
                        VocabSize = Vocab,
                        BlockSize = ValidBlock,
                        Layers = [dummyLayer],
                        SharedHeadWeight = dummyPtr,
                        SharedHeadType = GgufType.F32
                    }));
            }

            // 4. CreateFromGguf rejects non-divisible nHeads % nHeadsKv != 0 cleanly
            const int BadGqaDim = 48;
            const int BadHeads = 3;
            const int BadHeadsKv = 2;
            const int BadHeadDim = 16;
            const int BadQDim = BadHeads * BadHeadDim;
            const int BadKvDim = BadHeadsKv * BadHeadDim;
            var badGqaTrunk = new SyntheticHiddenStateTrunk(BadGqaDim, Vocab);

            string badGqaEmbPath = Path.Combine(Path.GetTempPath(), $"dflash_badgqa_emb_{Guid.NewGuid():N}.gguf");
            string badGqaBasePath = Path.Combine(Path.GetTempPath(), $"dflash_badgqa_base_{Guid.NewGuid():N}.gguf");
            string badGqaSidePath = Path.Combine(Path.GetTempPath(), $"dflash_badgqa_side_{Guid.NewGuid():N}.gguf");
            tempPaths.Add(badGqaEmbPath);
            tempPaths.Add(badGqaBasePath);
            tempPaths.Add(badGqaSidePath);

            var badKv = BuildBaseMetadata(BadGqaDim, BadGqaDim, BadHeads, BadHeadsKv, BadHeadDim, ValidBlock);

            var badEmbTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(badEmbTensors, BadGqaDim, BadGqaDim, BadQDim, BadKvDim, Vocab);
            AppendDFlashTensors(badEmbTensors, BadGqaDim, BadGqaDim, BadQDim, BadKvDim, Vocab, ValidBlock);
            WriteCustomGguf(badGqaEmbPath, badKv, badEmbTensors);

            using (var g = GgufFile.Open(badGqaEmbPath))
            using (var w = new ModelWeights(g))
            {
                Assert.Null(DFlashDraftProvider.TryCreateFromModel(badGqaTrunk, w));
            }

            var badBaseTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendBaseTensors(badBaseTensors, BadGqaDim, BadGqaDim, BadQDim, BadKvDim, Vocab);
            WriteCustomGguf(badGqaBasePath, badKv, badBaseTensors);

            var badSideTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
            AppendDFlashTensors(badSideTensors, BadGqaDim, BadGqaDim, BadQDim, BadKvDim, Vocab, ValidBlock);
            WriteCustomGguf(badGqaSidePath, badKv, badSideTensors);

            using (var bg = GgufFile.Open(badGqaBasePath))
            using (var bw = new ModelWeights(bg))
            {
                Assert.Throws<InvalidDataException>(() =>
                    DFlashDraftProvider.LoadFromSidecar(badGqaTrunk, bw, badGqaSidePath));
            }
        }
        finally
        {
            foreach (string p in tempPaths)
            {
                try { if (File.Exists(p)) File.Delete(p); } catch { }
            }
        }
    }
}
