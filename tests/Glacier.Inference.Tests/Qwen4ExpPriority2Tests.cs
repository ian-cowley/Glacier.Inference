namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

[Collection("SequentialGpu")]
public sealed unsafe class Qwen4ExpPriority2Tests : IDisposable
{
    private readonly string _tempDir;

    public Qwen4ExpPriority2Tests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierQwen4ExpP2_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, recursive: true);
            }
        }
        catch
        {
        }
    }

    [Fact]
    public void PleNgramHasher_Follows64BitMultiplyXorAndForwardEosCutInvariants()
    {
        var cfg = new PleHashConfig
        {
            NgramSize = 3,
            HeadsPerNgram = 4,
            HeadDim = 32,
            ConvKernel = 4,
            EosTokenId = 99,
            ImageTokenId = 88,
            Multipliers = [11UL, 23UL, 37UL],
            HeadVocabSizes = [101L, 103L, 107L, 109L, 113L, 127L, 131L, 137L],
            HeadOffsets = [0L, 101L, 204L, 311L, 420L, 533L, 660L, 791L]
        };

        long[] rowsA = new long[8];
        long[] rowsB = new long[8];

        // 1. Token 0 is a valid predecessor and must NOT cut older token 7
        PleNgramHasher.ComputeNgramRows(token: 5, prevOldestFirst: [7, 0], cfg, rowsA);
        ulong biMix = unchecked((5UL * 11UL) ^ (0UL * 23UL));
        ulong triMix = unchecked(biMix ^ (7UL * 37UL));
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(cfg.HeadOffsets[k] + (long)(biMix % (ulong)cfg.HeadVocabSizes[k]), rowsA[k]);
            Assert.Equal(cfg.HeadOffsets[4 + k] + (long)(triMix % (ulong)cfg.HeadVocabSizes[4 + k]), rowsA[4 + k]);
        }

        // 2. Predecessor EOS or < 0 propagates cut forward across older positions
        PleNgramHasher.ComputeNgramRows(token: 5, prevOldestFirst: [7, 99], cfg, rowsA);
        PleNgramHasher.ComputeNgramRows(token: 5, prevOldestFirst: [12, -1], cfg, rowsB);
        ulong cutBi = unchecked((5UL * 11UL) ^ (99UL * 23UL));
        ulong cutTri = unchecked(cutBi ^ (99UL * 37UL));
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(rowsA[k], rowsB[k]);
            Assert.Equal(rowsA[4 + k], rowsB[4 + k]);
            Assert.Equal(cfg.HeadOffsets[k] + (long)(cutBi % (ulong)cfg.HeadVocabSizes[k]), rowsA[k]);
            Assert.Equal(cfg.HeadOffsets[4 + k] + (long)(cutTri % (ulong)cfg.HeadVocabSizes[4 + k]), rowsA[4 + k]);
        }

        // 3. Current token == EOS or ImageTokenId does NOT cut valid predecessors; ImageTokenId substitutes 0
        PleNgramHasher.ComputeNgramRows(token: 99, prevOldestFirst: [7, 13], cfg, rowsA);
        ulong eosCurBi = unchecked((99UL * 11UL) ^ (13UL * 23UL));
        ulong eosCurTri = unchecked(eosCurBi ^ (7UL * 37UL));
        Assert.Equal(cfg.HeadOffsets[0] + (long)(eosCurBi % (ulong)cfg.HeadVocabSizes[0]), rowsA[0]);
        Assert.Equal(cfg.HeadOffsets[4] + (long)(eosCurTri % (ulong)cfg.HeadVocabSizes[4]), rowsA[4]);

        PleNgramHasher.ComputeNgramRows(token: 5, prevOldestFirst: [7, 88], cfg, rowsA);
        PleNgramHasher.ComputeNgramRows(token: 5, prevOldestFirst: [7, 0], cfg, rowsB);
        Assert.Equal(rowsB, rowsA);
    }

    [Theory]
    [InlineData(PleRowFormat.IQ4_NL)]
    [InlineData(PleRowFormat.Fp8E4M3)]
    [InlineData(PleRowFormat.BF16)]
    [InlineData(PleRowFormat.Q8_0)]
    public void DirectIoNgramTable_AllFormatsStraddledPagesAndCaching_MatchMemoryMapped(PleRowFormat format)
    {
        const int headDim = 64;
        const int rowCount = 300;
        const int padding = 777; // Forces non-page-aligned offset so multiple rows straddle 4 KiB page boundaries

        string path = Path.Combine(_tempDir, $"ple_{format}.bin");
        DirectIoNgramTable.CreateSyntheticFile(
            path,
            rowCount,
            headDim,
            format,
            (row, col) => MathF.Sin((row * 0.13f) + (col * 0.07f)) * 0.8f,
            padding);

        var hashCfg = new PleHashConfig
        {
            NgramSize = 3,
            HeadsPerNgram = 2,
            HeadDim = headDim,
            ConvKernel = 4,
            EosTokenId = 250,
            Multipliers = [17UL, 31UL, 53UL],
            HeadVocabSizes = [71L, 73L, 79L, 77L],
            HeadOffsets = [0L, 71L, 144L, 223L]
        };

        using var direct = new DirectIoNgramTable(path, padding, rowCount, headDim, format, PleIoMode.DirectIo, rowCacheCapacity: 128, hashCfg);
        using var mapped = new DirectIoNgramTable(path, padding, rowCount, headDim, format, PleIoMode.MemoryMapped, rowCacheCapacity: 128, hashCfg);

        if (OperatingSystem.IsWindows())
        {
            Assert.True(direct.IsUnbufferedHandleActive);
        }

        // Compute which row straddles page 0 -> page 1 (4096 boundary)
        int rowBytes = DirectIoNgramTable.GetRowBytes(format, headDim);
        long straddleRow = (4096L - padding) / rowBytes;
        long[] lookupRows = [0, straddleRow, straddleRow, straddleRow + 1, rowCount - 1, straddleRow - 1];

        float[] directBuf = new float[lookupRows.Length * headDim];
        float[] mappedBuf = new float[lookupRows.Length * headDim];

        var tDirect = direct.Issue(lookupRows);
        var tMapped = mapped.Issue(lookupRows);
        direct.Collect(tDirect, directBuf);
        mapped.Collect(tMapped, mappedBuf);

        Assert.Equal(mappedBuf, directBuf);
        Assert.Equal(lookupRows.Length, direct.CacheMisses);
        Assert.Equal(0, direct.CacheHits);
        Assert.True(direct.PhysicalPagesIssued >= 2);

        // Repeat lookup -> all rows served from Clock row cache (0 additional misses / physical pages)
        long pagesBefore = direct.PhysicalPagesIssued;
        var tDirect2 = direct.Issue(lookupRows);
        direct.Collect(tDirect2, directBuf);
        Assert.Equal(lookupRows.Length, direct.CacheHits);
        Assert.Equal(pagesBefore, direct.PhysicalPagesIssued);

        // Token gather parity via split IssueToken + Collect
        float[] tokDirect = new float[hashCfg.EmbeddingDim];
        float[] tokMapped = new float[hashCfg.EmbeddingDim];
        var td = direct.IssueToken(token: 11, prevOldestFirst: [3, 9]);
        var tm = mapped.IssueToken(token: 11, prevOldestFirst: [3, 9]);
        direct.Collect(td, tokDirect);
        mapped.Collect(tm, tokMapped);
        Assert.Equal(tokMapped, tokDirect);
    }

    [Fact]
    public void HyperConnectionStack_And_PleLayerProcessor_VerifyDividedSiluGateAndSnapshotRollback()
    {
        const int hcCount = 4;
        const int dim = 8;
        const int lowRank = 4;
        const int hcDim = hcCount * dim;

        using var hc = new HyperConnectionStack(hcCount, dim, lowRank, rmsEps: 1e-6f, maxBatchSize: 2);

        float* wNorm = AllocFloats(hcDim, 1.0f);
        float* wDown = AllocFloats(lowRank * hcDim, 0.25f);
        float* wUp = AllocFloats(hcDim * lowRank, 0.5f);
        float* wInject = AllocFloats(hcCount * hcDim, 0f);
        float* embed = AllocFloats(dim, 0f);
        float* mixed = AllocFloats(dim, 0f);
        float* inject = AllocFloats(hcCount, 0f);
        float* blockOut = AllocFloats(dim, 0.75f);

        try
        {
            for (int d = 0; d < dim; d++) embed[d] = 0.5f + 0.1f * d;

            hc.InitTokenFromEmbedding(embed);
            float* r = hc.GetResidualTokenPtr();

            // Inject logits with wInject == 0 => inject[c] = 0 => gr_write weight = 2 * sigmoid(0) = 1.0
            hc.Read(r, wNorm, GgufType.F32, (byte*)wDown, GgufType.F32, (byte*)wUp, GgufType.F32, (byte*)wInject, mixed, inject);

            // Verify mixed manually with exact silu((xn @ wDown) / hcCount)
            float sumSq = 0f;
            for (int d = 0; d < dim; d++) sumSq += embed[d] * embed[d];
            float rms = 1.0f / MathF.Sqrt((sumSq / dim) + 1e-6f);

            float dotSum = 0f;
            for (int c = 0; c < hcCount; c++)
            {
                for (int d = 0; d < dim; d++)
                {
                    dotSum += (embed[d] * rms) * 0.25f;
                }
            }
            float zPreDiv = dotSum / hcCount; // Divided by hcCount BEFORE SiLU!
            float lo = zPreDiv / (1.0f + MathF.Exp(-zPreDiv));
            float upVal = lowRank * lo * 0.5f;
            float gate = 1.0f / (1.0f + MathF.Exp(-upVal));
            for (int d = 0; d < dim; d++)
            {
                float expected = (embed[d] * rms) * gate;
                Assert.InRange(mixed[d], expected - 1e-5f, expected + 1e-5f);
            }

            hc.Write(r, blockOut, inject);
            for (int c = 0; c < hcCount; c++)
            {
                for (int d = 0; d < dim; d++)
                {
                    Assert.InRange(r[c * dim + d], embed[d] + blockOut[d] - 1e-5f, embed[d] + blockOut[d] + 1e-5f);
                }
            }

            // Test PleLayerProcessor normalized history + SaveState/RestoreState rollback
            var hashCfg = new PleHashConfig
            {
                NgramSize = 3,
                HeadsPerNgram = 2,
                HeadDim = 8,
                ConvKernel = 4,
                EosTokenId = 99,
                Multipliers = [3UL, 5UL, 7UL],
                HeadVocabSizes = [11L, 13L, 17L, 19L],
                HeadOffsets = [0L, 11L, 24L, 41L]
            };

            using var ple = new PleLayerProcessor(pleLayerCount: 1, hcCount, dim, hashCfg, rmsEps: 1e-6f);
            Assert.Equal(9, ple.HistorySlots);

            float* pKeyW = AllocFloats(hcDim * hashCfg.EmbeddingDim, 0.05f);
            float* pValW = AllocFloats(dim * hashCfg.EmbeddingDim, 0.1f);
            float* pNorm = AllocFloats(hcDim, 1.0f);
            float* pConvW = AllocFloats(hcDim * 4, 0.2f);
            float* pEmb = AllocFloats(hashCfg.EmbeddingDim, 0.3f);
            float[] rCopyBefore = new float[hcDim];
            float[] rAfterSpec = new float[hcDim];
            float[] rAfterReplay = new float[hcDim];

            try
            {
                var pw = new PleLayerWeights
                {
                    PleKeyType = GgufType.F32,
                    PleKeyWeight = (byte*)pKeyW,
                    PleValueType = GgufType.F32,
                    PleValueWeight = (byte*)pValW,
                    PleNormKey = pNorm,
                    PleNormQuery = pNorm,
                    PleNormConv = pNorm,
                    PleConv1dWeight = pConvW
                };

                ple.ExecuteToken(0, r, pEmb, pw);
                ple.PushToken(10);

                byte[] snapshot = new byte[ple.StateSnapshotBytes];
                ple.SaveState(snapshot);
                new ReadOnlySpan<float>(r, hcDim).CopyTo(rCopyBefore);

                // Run speculative token 20 + 21 (mutates conv history + rolling prevTokens)
                ple.ExecuteToken(0, r, pEmb, pw);
                ple.PushToken(20);
                ple.ExecuteToken(0, r, pEmb, pw);
                ple.PushToken(21);
                new ReadOnlySpan<float>(r, hcDim).CopyTo(rAfterSpec);

                // Roll back and replay exact same sequence
                ple.RestoreState(snapshot);
                fixed (float* pReplay = rCopyBefore)
                {
                    ple.ExecuteToken(0, pReplay, pEmb, pw);
                    ple.PushToken(20);
                    ple.ExecuteToken(0, pReplay, pEmb, pw);
                    ple.PushToken(21);
                    new ReadOnlySpan<float>(pReplay, hcDim).CopyTo(rAfterReplay);
                }

                Assert.Equal(rAfterSpec, rAfterReplay);
            }
            finally
            {
                NativeMemory.AlignedFree(pKeyW);
                NativeMemory.AlignedFree(pValW);
                NativeMemory.AlignedFree(pNorm);
                NativeMemory.AlignedFree(pConvW);
                NativeMemory.AlignedFree(pEmb);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(wNorm);
            NativeMemory.AlignedFree(wDown);
            NativeMemory.AlignedFree(wUp);
            NativeMemory.AlignedFree(wInject);
            NativeMemory.AlignedFree(embed);
            NativeMemory.AlignedFree(mixed);
            NativeMemory.AlignedFree(inject);
            NativeMemory.AlignedFree(blockOut);
        }
    }

    [Fact]
    public void QsaSparseAttentionIndexer_IncompleteTailBiasPerHeadReluTieBreakAndRollback()
    {
        const int dim = 8;
        const int idxHeads = 2;
        const int idxKeyDim = 8;
        const int blockSize = 4;
        const int topK = 4; // MaxSelectionWidth = 7

        using var qsa = new QsaSparseAttentionIndexer(
            dim,
            idxHeads,
            idxKeyDim,
            blockSize,
            topK,
            rotaryDim: 4,
            ropeFreqBase: 10000.0f,
            rmsEps: 1e-6f,
            maxSeqLen: 64);

        float* qProjW = AllocFloats(idxHeads * idxKeyDim * dim, 0f);
        float* kProjW = AllocFloats(idxKeyDim * dim, 0f);
        float* normOne = AllocFloats(idxKeyDim, 1.0f);
        float* hidden = AllocFloats(dim, 0f);

        try
        {
            // Identity-like k_proj (k[d] = hidden[d])
            for (int d = 0; d < idxKeyDim; d++) kProjW[d * dim + d] = 1.0f;

            // 1. Equal-score tie-break: zero queries => all complete blocks score 0, incomplete tail scores +1e9f.
            var zeroWeights = new QsaIndexerWeights
            {
                QProjType = GgufType.F32,
                QProjWeight = (byte*)qProjW,
                KProjType = GgufType.F32,
                KProjWeight = (byte*)kProjW,
                QNormWeight = normOne,
                KNormWeight = normOne
            };

            byte[] snapshotAt7 = new byte[qsa.StateSnapshotBytes];
            int* selIds = null;
            int selCount = 0;

            for (int p = 0; p < 11; p++)
            {
                for (int d = 0; d < dim; d++) hidden[d] = (p == d) ? 1.0f : 0.1f;
                qsa.RecordKeyAndSelect(p, hidden, in zeroWeights, out selIds, out selCount);
                if (p == 7)
                {
                    qsa.SaveState(snapshotAt7);
                }
            }

            // nKv = 11: complete blocks [0..3] (cb=0), [4..7] (cb=1); incomplete tail [8..10]
            Assert.Equal(7, selCount);
            // Tie-break must keep smaller cell indices [0,1,2,3] ahead of [4,5,6,7], plus protected tail [8,9,10], sorted ascending!
            int[] actualIds = new ReadOnlySpan<int>(selIds, selCount).ToArray();
            Assert.Equal([0, 1, 2, 3, 8, 9, 10], actualIds);

            // Tail scores have +1e9f protection relative to complete blocks
            ReadOnlySpan<float> tailScores = qsa.LastScores;
            Assert.Equal(11, tailScores.Length);
            Assert.True(tailScores[8] >= 1e8f && tailScores[9] >= 1e8f && tailScores[10] >= 1e8f);
            Assert.Equal(0f, tailScores[0]);

            // 2. Rollback check from pos=7 replaying 8..10 yields identical selected IDs and scores
            float[] scoresBeforeRollback = tailScores.ToArray();
            qsa.RestoreState(snapshotAt7);
            for (int p = 8; p < 11; p++)
            {
                for (int d = 0; d < dim; d++) hidden[d] = (p == d) ? 1.0f : 0.1f;
                qsa.RecordKeyAndSelect(p, hidden, in zeroWeights, out selIds, out selCount);
            }
            Assert.Equal(actualIds, new ReadOnlySpan<int>(selIds, selCount).ToArray());
            Assert.Equal(scoresBeforeRollback, qsa.LastScores.ToArray());

            // 3. Per-head ReLU before sum: construct Head 0 strongly positive, Head 1 strongly negative
            // Since ReLU runs per head before summing, a negative Head 1 dot product does NOT cancel Head 0!
            qsa.Reset();
            for (int d = 0; d < idxKeyDim; d++)
            {
                qProjW[(0 * idxKeyDim + d) * dim + d] = +1.0f; // Head 0 aligns positively
                qProjW[(1 * idxKeyDim + d) * dim + d] = -1.0f; // Head 1 aligns negatively
            }
            for (int d = 0; d < dim; d++) hidden[d] = 1.0f;
            for (int p = 0; p < 9; p++)
            {
                qsa.RecordKeyAndSelect(p, hidden, in zeroWeights, out selIds, out selCount);
            }

            // Pooled block 0 and query at constant positive hidden[d]=1.0: Head 0 dot > 0 while Head 1 dot < 0 -> sum > 0 (would be <= 0 if summed before ReLU)
            Assert.True(qsa.LastScores[0] > 1.0f, $"Expected positive per-head ReLU score, got {qsa.LastScores[0]}");
        }
        finally
        {
            NativeMemory.AlignedFree(qProjW);
            NativeMemory.AlignedFree(kProjW);
            NativeMemory.AlignedFree(normOne);
            NativeMemory.AlignedFree(hidden);
        }
    }

    [Fact]
    public void CostGatedDraftPolicy_TracksTestedTrialsAndAdaptsRouteOnRejections()
    {
        var policy = new CostGatedDraftPolicy(
            baseVerifyMs: 10.0,
            perDraftVerifyMs: 2.0,
            mtpDraftStepMs: 1.5,
            lookupOverheadMs: 0.05,
            priorMtpAcceptRate: 0.70,
            priorLookupAcceptRate: 0.72,
            decayFactor: 0.90,
            maxMtpDraftTokens: 3,
            maxLookupDraftTokens: 6);

        // Initial choice with lookup candidate available should favor PromptLookup (lower overhead)
        DraftDecision initial = policy.SelectRoute(maxDraftTokens: 4, hasMtp: true, lookupCandidateCount: 4, matchedNgramLength: 3);
        Assert.Equal(DraftRoute.PromptLookup, initial.Route);

        // Verify trial accounting: K=4, A=0 => tested = min(4, 0+1) = 1
        policy.RecordOutcome(DraftRoute.PromptLookup, matchedNgramLength: 3, draftedCount: 4, acceptedCount: 0, roundElapsedMs: 12.0);
        var (s1, t1) = policy.GetRouteStats(DraftRoute.PromptLookup, 3);
        Assert.Equal(0.0, s1, precision: 6);
        Assert.Equal(1.0, t1, precision: 6);

        // K=4, A=2 => tested = min(4, 2+1) = 3
        policy.RecordOutcome(DraftRoute.PromptLookup, matchedNgramLength: 4, draftedCount: 4, acceptedCount: 2, roundElapsedMs: 12.0);
        var (s2, t2) = policy.GetRouteStats(DraftRoute.PromptLookup, 4);
        Assert.Equal(2.0, s2, precision: 6);
        Assert.Equal(3.0, t2, precision: 6);

        // K=4, A=4 => tested = min(4, 4+1) = 4
        policy.RecordOutcome(DraftRoute.PromptLookup, matchedNgramLength: 2, draftedCount: 4, acceptedCount: 4, roundElapsedMs: 12.0);
        var (s3, t3) = policy.GetRouteStats(DraftRoute.PromptLookup, 2);
        Assert.Equal(4.0, s3, precision: 6);
        Assert.Equal(4.0, t3, precision: 6);

        // Reject lookup repeatedly at bucket 3 until policy switches to MTP
        for (int i = 0; i < 6; i++)
        {
            policy.RecordOutcome(DraftRoute.PromptLookup, matchedNgramLength: 3, draftedCount: 4, acceptedCount: 0, roundElapsedMs: 12.0);
        }

        DraftDecision afterLookupFail = policy.SelectRoute(maxDraftTokens: 4, hasMtp: true, lookupCandidateCount: 4, matchedNgramLength: 3);
        Assert.Equal(DraftRoute.Mtp, afterLookupFail.Route);

        // Now reject MTP repeatedly until policy falls back to Serial (DraftCount == 0)
        for (int i = 0; i < 12; i++)
        {
            policy.RecordOutcome(DraftRoute.Mtp, matchedNgramLength: 0, draftedCount: 3, acceptedCount: 0, roundElapsedMs: 15.0);
        }

        DraftDecision fallback = policy.SelectRoute(maxDraftTokens: 4, hasMtp: true, lookupCandidateCount: 4, matchedNgramLength: 3);
        Assert.Equal(DraftRoute.Serial, fallback.Route);
        Assert.Equal(0, fallback.DraftCount);
    }

    [Fact]
    public async Task EndToEnd_SyntheticQwen4ExpGguf_DirectIoParity_DraftVocab_AndSpeculativeMtp()
    {
        string ggufPath = Path.Combine(_tempDir, "synthetic_qwen4exp.gguf");
        BuildSyntheticQwen4ExpGguf(ggufPath);

        // 1. Metadata, GGUF parsing, and trailing MTP stripping
        using (var gguf = GgufFile.Open(ggufPath))
        {
            Assert.Equal("qwen4exp", gguf.Architecture);
            Assert.Equal(3, gguf.BlockCount); // GGUF reports 3 blocks including trailing blk.2.nextn.*
            var weights = new ModelWeights(gguf);

            Assert.Equal(UniversalArchitecture.HybridSsm, weights.ArchitectureFamily);
            Assert.Equal(2, weights.BlockCount); // Trunk strips trailing MTP block 2
            Assert.Equal(2, weights.EmbeddedMtpBlockIndex);
            Assert.Equal(4, weights.HyperConnectionCount);
            Assert.Equal(4, weights.HyperConnectionLowRank);
            Assert.True(weights.Layers[0].IsGdn);
            Assert.True(weights.Layers[0].HasPle);
            Assert.True(weights.Layers[0].HasHyperConnections);
            Assert.True(weights.Layers[1].IsQsa);
            Assert.True(weights.Layers[1].IsMoe);
            Assert.True(weights.Layers[1].HasHyperConnections);

            // 2. DirectIo vs MemoryMapped PLE parity inside full Qwen3HybridModel forward + snapshot rollback
            using var modelDirect = new Qwen3HybridModel(weights, maxSeqLen: 64, pleIoMode: PleIoMode.DirectIo);
            using var modelMmap = new Qwen3HybridModel(weights, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
            using var kvDirect = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 64, weights.ValueDim);
            using var kvMmap = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 64, weights.ValueDim);

            if (OperatingSystem.IsWindows())
            {
                Assert.True(modelDirect.PleTable!.IsUnbufferedHandleActive);
            }

            float[] logitsDirect = new float[weights.VocabSize];
            float[] logitsMmap = new float[weights.VocabSize];
            int[] seq = [1, 2, 3, 4];

            for (int p = 0; p < seq.Length; p++)
            {
                bool isLast = p == seq.Length - 1;
                modelDirect.Forward(seq[p], p, kvDirect, isLast ? logitsDirect : Span<float>.Empty, computeLogits: isLast);
                modelMmap.Forward(seq[p], p, kvMmap, isLast ? logitsMmap : Span<float>.Empty, computeLogits: isLast);
            }

            for (int v = 0; v < weights.VocabSize; v++)
            {
                Assert.True(float.IsFinite(logitsDirect[v]));
                Assert.InRange(logitsDirect[v], logitsMmap[v] - 1e-5f, logitsMmap[v] + 1e-5f);
            }

            // Test snapshot rollback across GDN + PLE + QSA state
            byte[] snap = new byte[modelDirect.StateSnapshotBytes];
            modelDirect.SaveRecurrentState(snap);

            float[] logitsBeforeRollback = new float[weights.VocabSize];
            modelDirect.Forward(5, 4, kvDirect, logitsBeforeRollback, computeLogits: true);
            modelDirect.Forward(6, 5, kvDirect, logitsBeforeRollback, computeLogits: true);

            modelDirect.RestoreRecurrentState(snap);
            float[] logitsAfterRollback = new float[weights.VocabSize];
            modelDirect.Forward(5, 4, kvDirect, logitsAfterRollback, computeLogits: true);
            modelDirect.Forward(6, 5, kvDirect, logitsAfterRollback, computeLogits: true);

            for (int v = 0; v < weights.VocabSize; v++)
            {
                Assert.InRange(logitsAfterRollback[v], logitsBeforeRollback[v] - 1e-5f, logitsBeforeRollback[v] + 1e-5f);
            }

            // 3. DraftVocabularySubset language filtering, binary persistence, and PackedDraftOutputHead parity
            var tokenizer = new BpeTokenizer(gguf);
            var subset = DraftVocabularySubset.FromLanguageFilter(tokenizer);
            Assert.True(subset.SubsetSize < weights.VocabSize);
            Assert.True(tokenizer.TryGetTokenId("你好", out int cjkId));
            Assert.False(subset.ContainsTokenId(cjkId));
            Assert.True(tokenizer.TryGetTokenId("the", out int asciiId));
            Assert.True(subset.ContainsTokenId(asciiId));
            Assert.True(subset.ContainsTokenId(tokenizer.EosTokenId));

            string subsetPath = Path.Combine(_tempDir, "draft_vocab.gdvb");
            subset.SaveBinary(subsetPath);
            DraftVocabularySubset loadedSubset = DraftVocabularySubset.LoadBinary(subsetPath);
            Assert.Equal(subset.SubsetTokenIds.ToArray(), loadedSubset.SubsetTokenIds.ToArray());

            using var packedHead = new PackedDraftOutputHead(weights.OutType, weights.OutWeight, weights.EmbeddingLength, weights.VocabSize, loadedSubset);
            float* hiddenPtr = AllocFloats(weights.EmbeddingLength, 0f);
            try
            {
                for (int d = 0; d < weights.EmbeddingLength; d++) hiddenPtr[d] = MathF.Cos(d * 0.25f);
                float[] fullHeadLogits = new float[weights.VocabSize];
                float[] subHeadLogits = new float[loadedSubset.SubsetSize];
                fixed (float* pFull = fullHeadLogits)
                {
                    QuantKernels.MatVecMul(weights.OutType, weights.OutWeight, hiddenPtr, pFull, weights.EmbeddingLength, weights.VocabSize);
                }
                packedHead.ComputeSubsetLogits(hiddenPtr, subHeadLogits);
                for (int s = 0; s < loadedSubset.SubsetSize; s++)
                {
                    int fullIdx = loadedSubset.MapSubsetIndexToTokenId(s);
                    Assert.InRange(subHeadLogits[s], fullHeadLogits[fullIdx] - 1e-5f, fullHeadLogits[fullIdx] + 1e-5f);
                }
            }
            finally
            {
                NativeMemory.AlignedFree(hiddenPtr);
            }
        }

        // 4. End-to-end InferenceSession + MtpDraftProvider + AdaptiveCompositeDraftProvider + SpeculativeEngine parity
        using var baselineSession = new InferenceSession(ggufPath, maxSeqLen: 64, engine: InferenceEngineType.Cpu);
        Assert.IsType<Qwen3HybridModel>(baselineSession.CpuModel);
        using var baselineEngine = new SpeculativeEngine(baselineSession, draftProvider: new PromptLookupDraftProvider(), seed: 42);
        var baselineResult = await baselineEngine.GenerateAsync(
            "the quick fox",
            new SpeculativeOptions
            {
                MaxTokens = 10,
                MaxDraftTokens = 0,
                Sampling = SamplingOptions.Greedy
            },
            formatChat: false);

        using var mtpSession = new InferenceSession(ggufPath, maxSeqLen: 64, engine: InferenceEngineType.Cpu);
        var trunk = Assert.IsAssignableFrom<CpuModelBase>(mtpSession.CpuModel);
        DraftVocabularySubset langSubset = DraftVocabularySubset.FromLanguageFilter(mtpSession.Tokenizer);
        using MtpDraftProvider? mtpDrafter = MtpDraftProvider.TryCreateFromModel(trunk, langSubset);
        Assert.NotNull(mtpDrafter);

        using var specMtpEngine = new SpeculativeEngine(mtpSession, mtpDrafter, seed: 42);
        var mtpResult = await specMtpEngine.GenerateAsync(
            "the quick fox",
            new SpeculativeOptions
            {
                MaxTokens = 10,
                MaxDraftTokens = 3,
                Sampling = SamplingOptions.Greedy
            },
            formatChat: false);

        Assert.Equal(baselineResult.Text, mtpResult.Text);
        Assert.Equal(baselineResult.FinishReason, mtpResult.FinishReason);
        Assert.Equal(10, baselineResult.Metrics.GeneratedTokens);
        Assert.Equal(10, mtpResult.Metrics.GeneratedTokens);
        Assert.True(
            mtpResult.SpeculativeMetrics.SpeculativeSteps > 0,
            $"SpecSteps=0! promptToks={mtpResult.Metrics.PromptTokens}, genToks={mtpResult.Metrics.GeneratedTokens}, finish={mtpResult.FinishReason}, serial={mtpResult.SpeculativeMetrics.SerialFallbackSteps}, proposed={mtpResult.SpeculativeMetrics.DraftTokensProposed}, snapBytes={mtpSession.StateSnapshotBytes}, unsnap={mtpSession.HasUnsnapshottableRecurrentState}, ring={mtpSession.IsRingBufferEnabled}");
        Assert.True(mtpResult.SpeculativeMetrics.DraftTokensProposed > 0);

        // Adaptive composite (PromptLookup + MTP + CostGatedDraftPolicy) stochastic execution check
        var adaptivePolicy = new CostGatedDraftPolicy(maxMtpDraftTokens: 2, maxLookupDraftTokens: 4);
        using var adaptiveDrafter = new AdaptiveCompositeDraftProvider(
            new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4),
            mtpDrafter,
            adaptivePolicy,
            ownsProviders: false);

        using var stochasticEngine = new SpeculativeEngine(mtpSession, adaptiveDrafter, seed: 123);
        var stochasticResult = await stochasticEngine.GenerateAsync(
            "the quick fox",
            new SpeculativeOptions
            {
                MaxTokens = 8,
                MaxDraftTokens = 2,
                Sampling = new SamplingOptions { Temperature = 0.8f, TopK = 8, TopP = 0.9f, RepetitionPenalty = 1.05f }
            },
            formatChat: false);

        Assert.Equal(8, stochasticResult.Metrics.GeneratedTokens);
        Assert.True(stochasticResult.SpeculativeMetrics.SpeculativeSteps > 0);
        Assert.True(adaptiveDrafter.MtpRounds + adaptiveDrafter.PromptLookupRounds > 0);
    }

    [Fact]
    public void RealQwen4ExpLayout_MergedTensors_NullOutputNorm_SigmoidGdnGate_And_SharedShexpDim_ExecutesCleanly()
    {
        string ggufPath = Path.Combine(_tempDir, "qwen4exp_real_layout.gguf");
        BuildRealLayoutSyntheticQwen4ExpGguf(ggufPath);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        Assert.True(weights.OutNormWeight == null);
        Assert.Equal(48, weights.SharedExpertFeedForwardLength);
        Assert.Equal(4, weights.HeadCount);
        Assert.Equal(2, weights.HeadCountKv);
        Assert.Equal(-1, weights.EmbeddedMtpBlockIndex);
        Assert.Equal(2, weights.BlockCount);

        Assert.True(weights.Layers[0].IsGdn);
        Assert.True(weights.Layers[0].HasPle);
        Assert.True(weights.Layers[0].HasHyperConnections);
        Assert.True(weights.Layers[0].IsMoe);

        Assert.False(weights.Layers[1].IsGdn);
        Assert.True(weights.Layers[1].HasQGate);
        Assert.True(weights.Layers[1].IsQsa);
        Assert.True(weights.Layers[1].IsMoe);
        Assert.True(weights.Layers[1].HasHyperConnections);

        using var modelDirect = new Qwen3HybridModel(weights, maxSeqLen: 64, pleIoMode: PleIoMode.DirectIo);
        using var modelMmap = new Qwen3HybridModel(weights, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
        using var kvDirect = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 64, weights.ValueDim);
        using var kvMmap = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 64, weights.ValueDim);

        float[] logitsDirect = new float[weights.VocabSize];
        float[] logitsMmap = new float[weights.VocabSize];
        int[] seq = [1, 2, 3, 4];

        for (int p = 0; p < seq.Length; p++)
        {
            bool isLast = p == seq.Length - 1;
            modelDirect.Forward(seq[p], p, kvDirect, isLast ? logitsDirect : Span<float>.Empty, computeLogits: isLast);
            modelMmap.Forward(seq[p], p, kvMmap, isLast ? logitsMmap : Span<float>.Empty, computeLogits: isLast);
        }

        for (int v = 0; v < weights.VocabSize; v++)
        {
            Assert.True(float.IsFinite(logitsDirect[v]));
            Assert.InRange(logitsDirect[v], logitsMmap[v] - 1e-5f, logitsMmap[v] + 1e-5f);
        }
    }

    [Fact]
    public async Task DetachedSidecarMtp_MultiStreamHcAndQsa_PreservesExactGreedyParity()
    {
        string trunkPath = Path.Combine(_tempDir, "qwen4exp_trunk_detached.gguf");
        string sidecarPath = Path.Combine(_tempDir, "qwen4exp_mtp_sidecar.gguf");
        BuildRealLayoutSyntheticQwen4ExpGguf(trunkPath);
        BuildDetachedMtpSidecarGguf(sidecarPath);

        using var baselineSession = new InferenceSession(trunkPath, maxSeqLen: 64, engine: InferenceEngineType.Cpu);
        Assert.IsType<Qwen3HybridModel>(baselineSession.CpuModel);
        Assert.Null(MtpDraftProvider.TryCreateFromModel((CpuModelBase)baselineSession.CpuModel!));

        using var baselineEngine = new SpeculativeEngine(baselineSession, draftProvider: new PromptLookupDraftProvider(), seed: 42);
        var baselineResult = await baselineEngine.GenerateAsync(
            "the quick fox",
            new SpeculativeOptions
            {
                MaxTokens = 10,
                MaxDraftTokens = 0,
                Sampling = SamplingOptions.Greedy
            },
            formatChat: false);

        using var mtpSession = new InferenceSession(trunkPath, maxSeqLen: 64, engine: InferenceEngineType.Cpu);
        var trunk = Assert.IsAssignableFrom<CpuModelBase>(mtpSession.CpuModel);
        using MtpDraftProvider sidecarMtp = MtpDraftProvider.LoadFromSidecar(trunk, sidecarPath);

        using var specEngine = new SpeculativeEngine(mtpSession, sidecarMtp, seed: 42);
        var specResult = await specEngine.GenerateAsync(
            "the quick fox",
            new SpeculativeOptions
            {
                MaxTokens = 10,
                MaxDraftTokens = 3,
                Sampling = SamplingOptions.Greedy
            },
            formatChat: false);

        Assert.Equal(baselineResult.Text, specResult.Text);
        Assert.Equal(baselineResult.FinishReason, specResult.FinishReason);
        Assert.Equal(10, baselineResult.Metrics.GeneratedTokens);
        Assert.Equal(10, specResult.Metrics.GeneratedTokens);
        Assert.True(specResult.SpeculativeMetrics.SpeculativeSteps > 0);
        Assert.True(specResult.SpeculativeMetrics.DraftTokensProposed > 0);
    }

    [Fact]
    public void UpstreamGgmlQ2_0_Type42_And_Q5_0_GetByteSize_MatchExactDequantAndMatVecMul()
    {
        Assert.Equal(18, sizeof(BlockGgmlQ2_0));
        Assert.Equal(180L, GgufTypes.GetRowBytes(GgufType.GgmlQ2_0, 640));

        var pleQ5Tensor = new GgufTensorInfo
        {
            Name = "test.q5_0",
            DimensionsCount = 2,
            Dimensions = [320, 10240],
            Type = GgufType.Q5_0,
            Offset = 0
        };
        Assert.Equal(2_252_800UL, pleQ5Tensor.GetByteSize());

        var hugeQ2Tensor = new GgufTensorInfo
        {
            Name = "test.huge_ggml_q2_0",
            DimensionsCount = 2,
            Dimensions = [640, 4_000_000],
            Type = GgufType.GgmlQ2_0,
            Offset = 0
        };
        Assert.Equal((640UL * 4_000_000UL / 64UL) * 18UL, hugeQ2Tensor.GetByteSize());

        const int nCols = 640; // 10 blocks per row
        const int nRows = 64;
        int blocksPerRow = nCols / 64;

        BlockGgmlQ2_0* weights = (BlockGgmlQ2_0*)NativeMemory.AlignedAlloc((nuint)((long)nRows * blocksPerRow * sizeof(BlockGgmlQ2_0)), 64);
        float* x = AllocFloats(nCols, 0f);
        float* y = AllocFloats(nRows, 0f);
        float* dequantRow = AllocFloats(nCols, 0f);

        try
        {
            uint state = 0x1234567u;
            for (int c = 0; c < nCols; c++)
            {
                state = unchecked(state * 1664525u + 1013904223u);
                x[c] = ((state & 0xFFFFu) / 32768.0f) - 1.0f;
            }

            for (int r = 0; r < nRows; r++)
            {
                for (int b = 0; b < blocksPerRow; b++)
                {
                    BlockGgmlQ2_0* blk = &weights[r * blocksPerRow + b];
                    state = unchecked(state * 1664525u + 1013904223u);
                    blk->Delta = (Half)((((state & 0xFFu) / 255.0f) - 0.5f) * 0.4f);
                    for (int qb = 0; qb < 16; qb++)
                    {
                        state = unchecked(state * 1664525u + 1013904223u);
                        blk->Qs[qb] = (byte)(state & 0xFFu);
                    }
                }
            }

            // Explicitly verify consecutive packing [4*b + 0..3] on block 0
            weights[0].Delta = (Half)0.5f;
            weights[0].Qs[0] = 0b11_10_01_00; // j=0:-1, j=1:0, j=2:+1, j=3:+2

            QuantKernels.DequantizeGgmlQ2_0(weights, dequantRow, nCols);
            Assert.Equal(-0.5f, dequantRow[0], 4);
            Assert.Equal(0.0f, dequantRow[1], 4);
            Assert.Equal(0.5f, dequantRow[2], 4);
            Assert.Equal(1.0f, dequantRow[3], 4);

            QuantKernels.ExtractEmbedding(GgufType.GgmlQ2_0, (byte*)weights, 0, dequantRow, nCols);
            Assert.Equal(1.0f, dequantRow[3], 4);

            QuantKernels.MatVecMul(GgufType.GgmlQ2_0, (byte*)weights, x, y, nCols, nRows);

            for (int r = 0; r < nRows; r++)
            {
                QuantKernels.DequantizeGgmlQ2_0(weights + (long)r * blocksPerRow, dequantRow, nCols);
                float expected = 0f;
                for (int c = 0; c < nCols; c++)
                {
                    expected += dequantRow[c] * x[c];
                }
                Assert.InRange(y[r], expected - 1e-4f, expected + 1e-4f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(y);
            NativeMemory.AlignedFree(dequantRow);
        }
    }

    [Fact]
    public void HardwareRuntimeProfiler_CapturesLiveSnapshot_And_RecommendsMoePolicy()
    {
        var snap = HardwareRuntimeProfiler.CaptureSnapshot();
        Assert.True(snap.LogicalCores > 0);
        Assert.True(snap.TotalPhysicalRamBytes > 0);
        Assert.True(snap.MeasuredCpuSimdBandwidthGBps > 0.0);
        Assert.False(string.IsNullOrWhiteSpace(snap.ActiveSimdTier));
        Assert.False(string.IsNullOrWhiteSpace(HardwareRuntimeProfiler.FormatSummaryLine(snap)));

        string modelPath = Path.Combine(_tempDir, "profiler_moe.gguf");
        BuildRealLayoutSyntheticQwen4ExpGguf(modelPath);
        using var gguf = GgufFile.Open(modelPath);
        var weights = new ModelWeights(gguf);

        var moePolicy = HardwareRuntimeProfiler.RecommendPolicy(snap, weights);
        var densePolicy = HardwareRuntimeProfiler.RecommendPolicy(snap, weights: null);

        Assert.Equal(2, moePolicy.RecommendedMtpDraftTokens);
        Assert.Equal(4, densePolicy.RecommendedMtpDraftTokens);
        Assert.Equal(PleIoMode.MemoryMapped, moePolicy.RecommendedPleIoMode);
        Assert.Equal(PleIoMode.DirectIo, DirectIoNgramTable.ResolveIoMode(PleIoMode.Auto, 40L * 1024 * 1024 * 1024, 32UL * 1024 * 1024 * 1024));
        Assert.Equal(PleIoMode.MemoryMapped, DirectIoNgramTable.ResolveIoMode(PleIoMode.Auto, 4L * 1024 * 1024 * 1024, 32UL * 1024 * 1024 * 1024));
        Assert.NotNull(HardwareRuntimeProfiler.CreateCalibratedCoScheduler(snap));
    }

    [Fact]
    public void BatchedQwen4Exp_VerifyBatch_And_PerStepSnapshots_MatchSequentialExact()
    {
        string modelPath = Path.Combine(_tempDir, "qwen4exp_batched_verify.gguf");
        BuildRealLayoutSyntheticQwen4ExpGguf(modelPath);

        using var ggufSeq = GgufFile.Open(modelPath);
        var weightsSeq = new ModelWeights(ggufSeq);
        using var seqModel = new Qwen3HybridModel(weightsSeq, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
        using var seqKv = new KVCache(weightsSeq.BlockCount, weightsSeq.HeadCountKv, weightsSeq.HeadDim, maxSeqLen: 64, weightsSeq.ValueDim);

        using var ggufBatch = GgufFile.Open(modelPath);
        var weightsBatch = new ModelWeights(ggufBatch);
        using var batchModel = new Qwen3HybridModel(weightsBatch, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
        using var batchKv = new KVCache(weightsBatch.BlockCount, weightsBatch.HeadCountKv, weightsBatch.HeadDim, maxSeqLen: 64, weightsBatch.ValueDim);

        Assert.True(batchModel.SupportsBatchVerification);
        Assert.True(batchModel.SupportsPerStepVerificationSnapshots);

        int vocabSize = weightsSeq.VocabSize;

        // 1. Batched prompt prefill exact parity vs sequential Forward
        int[] prompt = [1, 2];
        float[] seqPrefillLogits = new float[vocabSize];
        float[] batchPrefillLogits = new float[vocabSize];
        seqModel.Forward(prompt[0], 0, seqKv, Span<float>.Empty, computeLogits: false);
        seqModel.Forward(prompt[1], 1, seqKv, seqPrefillLogits, computeLogits: true);
        batchModel.ForwardBatch(prompt, 0, batchPrefillLogits, batchKv, computeLogits: true);
        for (int v = 0; v < vocabSize; v++)
        {
            Assert.InRange(batchPrefillLogits[v], seqPrefillLogits[v] - 1e-5f, seqPrefillLogits[v] + 1e-5f);
        }

        // 2. VerifyBatchLogits + per-step recurrent snapshots exact parity across all rejection points
        seqModel.ResetState();
        seqKv.Reset();
        batchModel.ResetState();
        batchKv.Reset();

        seqModel.Forward(1, 0, seqKv, Span<float>.Empty, computeLogits: false);
        batchModel.Forward(1, 0, batchKv, Span<float>.Empty, computeLogits: false);

        int[] verifyTokens = [2, 3, 4, 5];
        int snapBytes = checked((int)seqModel.StateSnapshotBytes);
        byte[][] refSnaps = new byte[verifyTokens.Length][];
        float[] refVerifyLogits = new float[verifyTokens.Length * vocabSize];
        int[] refGreedy = new int[verifyTokens.Length];

        for (int i = 0; i < verifyTokens.Length; i++)
        {
            Span<float> dstLogits = refVerifyLogits.AsSpan(i * vocabSize, vocabSize);
            seqModel.Forward(verifyTokens[i], 1 + i, seqKv, dstLogits, computeLogits: true);
            refSnaps[i] = new byte[snapBytes];
            seqModel.SaveRecurrentState(refSnaps[i]);

            int bestIdx = 0;
            float bestVal = dstLogits[0];
            for (int v = 1; v < vocabSize; v++)
            {
                if (dstLogits[v] > bestVal)
                {
                    bestVal = dstLogits[v];
                    bestIdx = v;
                }
            }
            refGreedy[i] = bestIdx;
        }

        byte[][] stepSnapshots =
        [
            new byte[snapBytes],
            new byte[snapBytes],
            new byte[snapBytes]
        ];
        float[] batchVerifyLogits = new float[verifyTokens.Length * vocabSize];
        batchModel.VerifyBatchLogits(verifyTokens, startPos: 1, batchVerifyLogits, vocabSize, batchKv, stepSnapshots);

        for (int i = 0; i < batchVerifyLogits.Length; i++)
        {
            Assert.True(float.IsFinite(batchVerifyLogits[i]));
            Assert.InRange(batchVerifyLogits[i], refVerifyLogits[i] - 1e-5f, refVerifyLogits[i] + 1e-5f);
        }

        // Check VerifyBatch greedy outputs too
        batchModel.ResetState();
        batchKv.Reset();
        batchModel.Forward(1, 0, batchKv, Span<float>.Empty, computeLogits: false);
        int[] batchGreedy = new int[verifyTokens.Length];
        batchModel.VerifyBatch(verifyTokens, startPos: 1, batchGreedy, batchKv, stepSnapshots);
        Assert.Equal(refGreedy, batchGreedy);

        // Verify restoring EVERY intermediate stepSnapshot[i] yields exact same next-token logits as sequential Forward
        float[] nextSeqLogits = new float[vocabSize];
        float[] nextBatchLogits = new float[vocabSize];
        for (int i = 0; i < stepSnapshots.Length; i++)
        {
            seqModel.RestoreRecurrentState(refSnaps[i]);
            batchModel.RestoreRecurrentState(stepSnapshots[i]);

            int followTok = 6 + i;
            int followPos = 1 + i + 1;
            seqModel.Forward(followTok, followPos, seqKv, nextSeqLogits, computeLogits: true);
            batchModel.Forward(followTok, followPos, batchKv, nextBatchLogits, computeLogits: true);

            for (int v = 0; v < vocabSize; v++)
            {
                Assert.InRange(nextBatchLogits[v], nextSeqLogits[v] - 1e-5f, nextSeqLogits[v] + 1e-5f);
            }
        }

        // Verify MoeTailSkipMinRank produces finite verify outputs cleanly
        batchModel.MoeTailSkipMinRank = 2;
        batchModel.ResetState();
        batchKv.Reset();
        batchModel.Forward(1, 0, batchKv, Span<float>.Empty, computeLogits: false);
        batchModel.VerifyBatch(verifyTokens, startPos: 1, batchGreedy, batchKv, stepSnapshots);
        Assert.All(batchGreedy, tok => Assert.InRange(tok, 0, vocabSize - 1));
    }

    [Fact]
    public void D3D12_GemvGgmlQ2_0_ExpertAndBatchGemm_MatchCpuExact()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        const int kCols = 128; // 2 blocks per row
        const int mRows = 8;
        const int batchSize = 5;

        float[] rawW = new float[mRows * kCols];
        float[] x = new float[batchSize * kCols];
        for (int i = 0; i < rawW.Length; i++) rawW[i] = MathF.Sin(i * 0.173f) * 0.6f;
        for (int i = 0; i < x.Length; i++) x[i] = MathF.Cos(i * 0.091f) * 0.8f;

        long rowBytes = GgufTypes.GetRowBytes(GgufType.GgmlQ2_0, kCols);
        byte* qW = (byte*)NativeMemory.AlignedAlloc((nuint)(mRows * rowBytes), 64);
        float[] cpuGemv = new float[mRows];
        float[] cpuBatch = new float[batchSize * mRows];

        try
        {
            fixed (float* pRaw = rawW, pX = x, pCpuGemv = cpuGemv, pCpuBatch = cpuBatch)
            {
                for (int r = 0; r < mRows; r++)
                {
                    QuantKernels.QuantizeGgmlQ2_0(pRaw + r * kCols, (BlockGgmlQ2_0*)(qW + r * rowBytes), kCols);
                }
                QuantKernels.MatVecMul(GgufType.GgmlQ2_0, qW, pX, pCpuGemv, kCols, mRows);
                QuantKernels.MatMulBatch(GgufType.GgmlQ2_0, qW, pX, pCpuBatch, kCols, mRows, batchSize);
            }

            // 1. Validate alignment, MakeExpertGemvShader compilation, and GemvGgmlQ2_0 parity
            Assert.Equal((ulong)(mRows * (kCols / 64) * 20), D3D12TensorAlign.GetAlignedTensorBytes(GgufType.GgmlQ2_0, mRows, kCols));
            var expertHlsl = D3D12Shaders.MakeExpertGemvShader(D3D12Shaders.GemvGgmlQ2_0);
            Assert.False(ctx.CompileShader(expertHlsl).IsEmpty);

            int allocBatch = Math.Max(32, batchSize);
            using var dW = D3D12TensorAlign.UploadAlignedTensor(ctx, GgufType.GgmlQ2_0, (IntPtr)qW, mRows, kCols);
            using var dX = ctx.CreateDeviceBuffer((ulong)(allocBatch * kCols * sizeof(float)));
            using var dYGemv = ctx.CreateDeviceBuffer((ulong)(mRows * sizeof(float)));
            using var dYBatch = ctx.CreateDeviceBuffer((ulong)(allocBatch * mRows * sizeof(float)));

            var gemvRootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
                Vortice.Direct3D12.RootSignatureFlags.None,
                new Vortice.Direct3D12.RootParameter[]
                {
                    new(new Vortice.Direct3D12.RootConstants(0, 0, 6), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All)
                }));
            using var _sigGemv = gemvRootSig;
            using var psoGemv = ctx.CreatePipelineState(gemvRootSig, ctx.CompileShader(D3D12Shaders.GemvGgmlQ2_0));

            fixed (float* pX = x)
            {
                ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(batchSize * kCols * sizeof(float)));
            }

            ctx.BeginCommands();
            var cmd = ctx.CommandList;
            cmd.SetComputeRootSignature(gemvRootSig);
            cmd.SetPipelineState(psoGemv);
            uint* cGemv = stackalloc uint[6];
            cGemv[0] = kCols;
            cGemv[1] = mRows;
            cGemv[2] = 0;
            cGemv[3] = 0;
            cGemv[4] = 1;
            cGemv[5] = 0;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)cGemv, 0);
            cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
            cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(5, dYGemv.GPUVirtualAddress);
            cmd.Dispatch((mRows + 3) / 4, 1, 1);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            float[] gpuGemv = new float[mRows];
            fixed (float* pGpu = gpuGemv)
            {
                ctx.CopyToHost((IntPtr)pGpu, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.InRange(gpuGemv[r], cpuGemv[r] - 1e-3f, cpuGemv[r] + 1e-3f);
            }

            // 2. Validate GemmGgmlQ2_0Batch parity
            using var gemmRootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
                Vortice.Direct3D12.RootSignatureFlags.None,
                new Vortice.Direct3D12.RootParameter[]
                {
                    new(new Vortice.Direct3D12.RootConstants(0, 0, 6), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                    new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All)
                }));
            using var psoGemm = ctx.CreatePipelineState(gemmRootSig, ctx.CompileShader(D3D12Shaders.GemmGgmlQ2_0Batch));

            ctx.BeginCommands();
            cmd = ctx.CommandList;
            cmd.SetComputeRootSignature(gemmRootSig);
            cmd.SetPipelineState(psoGemm);
            cGemv[0] = kCols;
            cGemv[1] = mRows;
            cGemv[2] = batchSize;
            cGemv[3] = 0;
            cGemv[4] = 0;
            cGemv[5] = 1;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)cGemv, 0);
            cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
            cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(5, dYBatch.GPUVirtualAddress);
            cmd.Dispatch((mRows + 3) / 4, (batchSize + 31) / 32, 1);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            float[] gpuBatch = new float[batchSize * mRows];
            fixed (float* pGpu = gpuBatch)
            {
                ctx.CopyToHost((IntPtr)pGpu, dYBatch, (ulong)(batchSize * mRows * sizeof(float)));
            }
            for (int i = 0; i < gpuBatch.Length; i++)
            {
                Assert.InRange(gpuBatch[i], cpuBatch[i] - 1e-3f, cpuBatch[i] + 1e-3f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(qW);
        }
    }

    private static float* AllocFloats(long count, float fill)
    {
        float* ptr = (float*)NativeMemory.AlignedAlloc((nuint)(Math.Max(1L, count) * sizeof(float)), 64);
        for (long i = 0; i < count; i++) ptr[i] = fill;
        return ptr;
    }

    private sealed class GgufTensorDef
    {
        public required string Name { get; init; }
        public required ulong[] Dims { get; init; }
        public required GgufType Type { get; init; }
        public required byte[] Data { get; init; }
        public long Offset { get; set; }
    }

    private static void BuildSyntheticQwen4ExpGguf(string path)
    {
        const int dim = 32;
        const int ffnDim = 64;
        const int vocabSize = 48;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int headDim = 8; // qDim=32, kvDim=16
        const int ssmConvKernel = 4;
        const int ssmStateSize = 4;
        const int ssmGroupCount = 2;
        const int ssmTimeStepRank = 4;
        const int ssmInnerSize = 16;
        const int gdnConvChannels = (ssmGroupCount * 2 + ssmTimeStepRank) * ssmStateSize; // 32
        const int hcCount = 4;
        const int hcLowRank = 4;
        const int hcDim = hcCount * dim; // 128
        const int idxHeads = 2;
        const int idxKeyDim = 16;
        const int idxTopK = 4;
        const int pleNgram = 3;
        const int pleHeadsPerNgram = 4;
        const int pleHeadDim = 32;
        const int pleTotalHeads = (pleNgram - 1) * pleHeadsPerNgram; // 8
        const int pleEmbDim = pleTotalHeads * pleHeadDim; // 256
        long[] pleVocabSizes = [17, 19, 23, 29, 31, 37, 41, 43]; // 240 rows
        long[] pleOffsets = [0, 17, 36, 59, 88, 119, 156, 197];
        long pleTotalRows = 240;
        const int eosId = 47;

        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen4exp"),
            ("general.alignment", 4U, 32U),
            ("qwen4exp.block_count", 4U, 3U), // blk 0 (GDN+PLE), blk 1 (QSA+MoE), blk 2 (NextN MTP)
            ("qwen4exp.embedding_length", 4U, (uint)dim),
            ("qwen4exp.feed_forward_length", 4U, (uint)ffnDim),
            ("qwen4exp.context_length", 4U, 128U),
            ("qwen4exp.attention.head_count", 4U, (uint)nHeads),
            ("qwen4exp.attention.head_count_kv", 4U, (uint)nHeadsKv),
            ("qwen4exp.attention.key_length", 4U, (uint)headDim),
            ("qwen4exp.attention.value_length", 4U, (uint)headDim),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen4exp.rope.dimension_count", 4U, (uint)headDim),
            ("qwen4exp.rope.freq_base", 6U, 10000.0f),
            ("qwen4exp.ssm.conv_kernel", 4U, (uint)ssmConvKernel),
            ("qwen4exp.ssm.state_size", 4U, (uint)ssmStateSize),
            ("qwen4exp.ssm.group_count", 4U, (uint)ssmGroupCount),
            ("qwen4exp.ssm.time_step_rank", 4U, (uint)ssmTimeStepRank),
            ("qwen4exp.ssm.inner_size", 4U, (uint)ssmInnerSize),
            ("qwen4exp.expert_count", 4U, 4U),
            ("qwen4exp.expert_used_count", 4U, 2U),
            ("qwen4exp.expert_feed_forward_length", 4U, 32U),
            ("qwen4exp.hyper_connection.count", 4U, (uint)hcCount),
            ("qwen4exp.hyper_connection.low_rank", 4U, (uint)hcLowRank),
            ("qwen4exp.attention.indexer.head_count", 4U, (uint)idxHeads),
            ("qwen4exp.attention.indexer.key_length", 4U, (uint)idxKeyDim),
            ("qwen4exp.attention.indexer.top_k", 4U, (uint)idxTopK),
            ("qwen4exp.attention.compress_ratios", 9U, (4U, new object[] { 0U, 2U, 0U })),
            ("qwen4exp.ple.layers", 9U, (4U, new object[] { 0U })),
            ("qwen4exp.ple.ngram_size", 4U, (uint)pleNgram),
            ("qwen4exp.ple.heads_per_ngram", 4U, (uint)pleHeadsPerNgram),
            ("qwen4exp.ple.conv_kernel", 4U, 4U),
            ("qwen4exp.ple.eos_token_id", 4U, (uint)eosId),
            ("qwen4exp.embedding_length_per_layer_input", 4U, (uint)pleHeadDim),
            ("qwen4exp.ple.layer_multipliers", 9U, (10U, new object[] { 11UL, 23UL, 37UL })),
            ("qwen4exp.ple.head_offsets", 9U, (10U, new object[] { 0UL, 17UL, 36UL, 59UL, 88UL, 119UL, 156UL, 197UL })),
            ("qwen4exp.ple.head_vocab_sizes", 9U, (10U, new object[] { 17UL, 19UL, 23UL, 29UL, 31UL, 37UL, 41UL, 43UL }))
        };

        // Build deterministic 48-token BPE vocabulary mixing ASCII & CJK tokens
        var tokens = new List<object>(vocabSize)
        {
            "<unk>", "the", " quick", " fox", " brown", " jumps", " over", " lazy",
            " dog", " alpha", " beta", " gamma", " delta", " one", " two", " three",
            "t", "h", "e", "\u0120", "q", "u", "i", "c",
            "k", "f", "o", "x", "a", "b", "d", "g",
            " ", ".", ",", "!", "?", "\n", "0", "1",
            "2", "3", "4", "你好", "世界", "中文", "<|im_end|>", "<|endoftext|>"
        };
        kv.Add(("tokenizer.ggml.tokens", 9U, (8U, tokens.ToArray())));
        kv.Add(("tokenizer.ggml.bos_token_id", 4U, 0U));
        kv.Add(("tokenizer.ggml.eos_token_id", 4U, (uint)eosId));
        kv.Add(("tokenizer.ggml.merges", 9U, (8U, Array.Empty<object>())));

        var tensors = new List<GgufTensorDef>();
        int seed = 101;

        void AddF32(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 4];
            Span<float> fSpan = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                fSpan[(int)i] = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.F32, Data = bytes });
        }

        void AddOneNorm(string name, int len, bool asBf16 = false)
        {
            if (asBf16)
            {
                byte[] bytes = new byte[len * 2];
                Span<ushort> uSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
                uint oneBits = BitConverter.SingleToUInt32Bits(1.0f);
                ushort bfOne = (ushort)(oneBits >> 16);
                uSpan.Fill(bfOne);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.BF16, Data = bytes });
            }
            else
            {
                byte[] bytes = new byte[len * 4];
                MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).Fill(1.0f);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.F32, Data = bytes });
            }
        }

        void AddBf16Matrix(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 2];
            Span<ushort> uSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                float val = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
                uSpan[(int)i] = (ushort)(BitConverter.SingleToUInt32Bits(val) >> 16);
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.BF16, Data = bytes });
        }

        // Core trunk embeddings & output HC head
        AddF32("token_embd.weight", dim, vocabSize);
        AddOneNorm("output_norm.weight", dim, asBf16: true);
        AddOneNorm("output_hc_norm.weight", hcDim, asBf16: true);
        AddF32("output_hc_down.weight", hcDim, hcLowRank);
        AddF32("output_hc_up.weight", hcLowRank, hcDim);
        AddF32("output.weight", dim, vocabSize);
        MemoryMarshal.Cast<byte, float>(tensors[^1].Data.AsSpan()).Slice(46 * dim, 2 * dim).Clear();
        AddBf16Matrix("per_layer_token_embd.weight", pleHeadDim, (ulong)pleTotalRows);

        // Layer 0: GDN + PLE + HC + Dense FFN
        AddOneNorm("blk.0.hc_attn_norm.weight", hcDim, asBf16: true);
        AddF32("blk.0.hc_attn_down.weight", hcDim, hcLowRank);
        AddF32("blk.0.hc_attn_up.weight", hcLowRank, hcDim);
        AddF32("blk.0.hc_attn_inject.weight", hcDim, hcCount);
        AddOneNorm("blk.0.hc_ffn_norm.weight", hcDim);
        AddF32("blk.0.hc_ffn_down.weight", hcDim, hcLowRank);
        AddF32("blk.0.hc_ffn_up.weight", hcLowRank, hcDim);
        AddF32("blk.0.hc_ffn_inject.weight", hcDim, hcCount);

        AddBf16Matrix("blk.0.ple_key.weight", pleEmbDim, hcDim);
        AddBf16Matrix("blk.0.ple_value.weight", pleEmbDim, dim);
        AddOneNorm("blk.0.ple_norm_key.weight", hcDim, asBf16: true);
        AddOneNorm("blk.0.ple_norm_query.weight", hcDim, asBf16: true);
        AddOneNorm("blk.0.ple_norm_conv.weight", hcDim, asBf16: true);
        AddF32("blk.0.ple_conv1d.weight", 4, hcDim);

        AddF32("blk.0.attn_qkv.weight", dim, gdnConvChannels);
        AddF32("blk.0.attn_gate.weight", dim, ssmInnerSize);
        AddF32("blk.0.ssm_conv1d.weight", ssmConvKernel, gdnConvChannels);
        AddBf16Matrix("blk.0.ssm_a", (ulong)ssmTimeStepRank);
        AddF32("blk.0.ssm_alpha.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_beta.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_dt.bias", (ulong)ssmTimeStepRank);
        AddOneNorm("blk.0.ssm_norm.weight", ssmStateSize, asBf16: true);
        AddF32("blk.0.ssm_out.weight", ssmInnerSize, dim);

        AddF32("blk.0.ffn_gate.weight", dim, ffnDim);
        AddF32("blk.0.ffn_up.weight", dim, ffnDim);
        AddF32("blk.0.ffn_down.weight", ffnDim, dim);

        // Layer 1: QSA + HC + Fused MoE (ffn_gate_up_exps) + Shared Expert
        AddOneNorm("blk.1.hc_attn_norm.weight", hcDim);
        AddF32("blk.1.hc_attn_down.weight", hcDim, hcLowRank);
        AddF32("blk.1.hc_attn_up.weight", hcLowRank, hcDim);
        AddF32("blk.1.hc_attn_inject.weight", hcDim, hcCount);
        AddOneNorm("blk.1.hc_ffn_norm.weight", hcDim);
        AddF32("blk.1.hc_ffn_down.weight", hcDim, hcLowRank);
        AddF32("blk.1.hc_ffn_up.weight", hcLowRank, hcDim);
        AddF32("blk.1.hc_ffn_inject.weight", hcDim, hcCount);

        AddF32("blk.1.attn_q.weight", dim, nHeads * headDim);
        AddF32("blk.1.attn_k.weight", dim, nHeadsKv * headDim);
        AddF32("blk.1.attn_v.weight", dim, nHeadsKv * headDim);
        AddF32("blk.1.attn_output.weight", nHeads * headDim, dim);
        AddOneNorm("blk.1.attn_q_norm.weight", headDim, asBf16: true);
        AddOneNorm("blk.1.attn_k_norm.weight", headDim, asBf16: true);

        AddF32("blk.1.indexer.q_proj.weight", dim, idxHeads * idxKeyDim);
        AddF32("blk.1.indexer.k_proj.weight", dim, idxKeyDim);
        AddOneNorm("blk.1.indexer.q_norm.weight", idxKeyDim, asBf16: true);
        AddOneNorm("blk.1.indexer.k_norm.weight", idxKeyDim, asBf16: true);

        AddBf16Matrix("blk.1.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.1.ffn_gate_up_exps.weight", dim, 64, 4);
        AddF32("blk.1.ffn_down_exps.weight", 32, dim, 4);
        AddF32("blk.1.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.1.ffn_gate_shexp.weight", dim, 32);
        AddF32("blk.1.ffn_up_shexp.weight", dim, 32);
        AddF32("blk.1.ffn_down_shexp.weight", 32, dim);

        // Layer 2: Trailing MTP / NextN block
        AddOneNorm("blk.2.nextn.enorm.weight", dim, asBf16: true);
        AddOneNorm("blk.2.nextn.hnorm.weight", dim, asBf16: true);
        AddF32("blk.2.nextn.eh_proj.weight", 2 * dim, dim);
        AddOneNorm("blk.2.nextn.shared_head_norm.weight", dim, asBf16: true);
        AddF32("blk.2.nextn.shared_head_head.weight", dim, vocabSize);
        MemoryMarshal.Cast<byte, float>(tensors[^1].Data.AsSpan()).Slice(46 * dim, 2 * dim).Clear();
        AddOneNorm("blk.2.attn_norm.weight", dim);
        AddF32("blk.2.attn_q.weight", dim, nHeads * headDim);
        AddF32("blk.2.attn_k.weight", dim, nHeadsKv * headDim);
        AddF32("blk.2.attn_v.weight", dim, nHeadsKv * headDim);
        AddF32("blk.2.attn_output.weight", nHeads * headDim, dim);
        AddOneNorm("blk.2.attn_q_norm.weight", headDim);
        AddOneNorm("blk.2.attn_k_norm.weight", headDim);
        AddOneNorm("blk.2.ffn_norm.weight", dim);
        AddF32("blk.2.ffn_gate.weight", dim, ffnDim);
        AddF32("blk.2.ffn_up.weight", dim, ffnDim);
        AddF32("blk.2.ffn_down.weight", ffnDim, dim);

        WriteGgufFile(path, kv, tensors);
    }

    private static List<object> BuildSyntheticQwen4ExpTokens(int vocabSize)
    {
        var tokens = new List<object>(vocabSize)
        {
            "<unk>", "the", " quick", " fox", " brown", " jumps", " over", " lazy",
            " dog", " alpha", " beta", " gamma", " delta", " one", " two", " three",
            "t", "h", "e", "\u0120", "q", "u", "i", "c",
            "k", "f", "o", "x", "a", "b", "d", "g",
            " ", ".", ",", "!", "?", "\n", "0", "1",
            "2", "3", "4", "你好", "世界", "中文", "<|im_end|>", "<|endoftext|>"
        };
        return tokens;
    }

    private static void BuildRealLayoutSyntheticQwen4ExpGguf(string path)
    {
        const int dim = 32;
        const int expFfnDim = 32;
        const int sharedFfnDim = 48;
        const int vocabSize = 48;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int headDim = 8; // qDim=32, kvDim=16
        const int ssmConvKernel = 4;
        const int ssmStateSize = 4;
        const int ssmGroupCount = 2;
        const int ssmTimeStepRank = 4;
        const int ssmInnerSize = 16;
        const int gdnConvChannels = (ssmGroupCount * 2 + ssmTimeStepRank) * ssmStateSize; // 32
        const int hcCount = 4;
        const int hcLowRank = 4;
        const int hcDim = hcCount * dim; // 128
        const int idxHeads = 2;
        const int idxKeyDim = 16;
        const int idxTopK = 4;
        const int pleNgram = 3;
        const int pleHeadsPerNgram = 4;
        const int pleHeadDim = 32;
        const long pleTotalRows = 240;
        const int eosId = 47;

        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen4exp"),
            ("general.alignment", 4U, 32U),
            ("qwen4exp.block_count", 4U, 2U),
            ("qwen4exp.embedding_length", 4U, (uint)dim),
            // Intentionally omit qwen4exp.feed_forward_length to exercise SharedExpertFeedForwardLength path
            ("qwen4exp.context_length", 4U, 128U),
            ("qwen4exp.attention.head_count", 9U, (4U, new object[] { 0U, (uint)nHeads })),
            ("qwen4exp.attention.head_count_kv", 9U, (4U, new object[] { 0U, (uint)nHeadsKv })),
            ("qwen4exp.attention.key_length", 4U, (uint)headDim),
            ("qwen4exp.attention.value_length", 4U, (uint)headDim),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen4exp.rope.dimension_count", 4U, (uint)headDim),
            ("qwen4exp.rope.freq_base", 6U, 10000.0f),
            ("qwen4exp.ssm.conv_kernel", 4U, (uint)ssmConvKernel),
            ("qwen4exp.ssm.state_size", 4U, (uint)ssmStateSize),
            ("qwen4exp.ssm.group_count", 4U, (uint)ssmGroupCount),
            ("qwen4exp.ssm.time_step_rank", 4U, (uint)ssmTimeStepRank),
            ("qwen4exp.ssm.inner_size", 4U, (uint)ssmInnerSize),
            ("qwen4exp.expert_count", 4U, 4U),
            ("qwen4exp.expert_used_count", 4U, 2U),
            ("qwen4exp.expert_feed_forward_length", 4U, (uint)expFfnDim),
            ("qwen4exp.expert_shared_feed_forward_length", 4U, (uint)sharedFfnDim),
            ("qwen4exp.hyper_connection.count", 4U, (uint)hcCount),
            ("qwen4exp.hyper_connection.low_rank", 4U, (uint)hcLowRank),
            ("qwen4exp.attention.indexer.head_count", 4U, (uint)idxHeads),
            ("qwen4exp.attention.indexer.key_length", 4U, (uint)idxKeyDim),
            ("qwen4exp.attention.indexer.top_k", 4U, (uint)idxTopK),
            ("qwen4exp.attention.compress_ratios", 9U, (4U, new object[] { 0U, 2U })),
            ("qwen4exp.ple.layers", 9U, (4U, new object[] { 0U })),
            ("qwen4exp.ple.ngram_size", 4U, (uint)pleNgram),
            ("qwen4exp.ple.heads_per_ngram", 4U, (uint)pleHeadsPerNgram),
            ("qwen4exp.ple.conv_kernel", 4U, 4U),
            ("qwen4exp.ple.eos_token_id", 4U, (uint)eosId),
            ("qwen4exp.embedding_length_per_layer_input", 4U, (uint)pleHeadDim),
            ("qwen4exp.ple.layer_multipliers", 9U, (10U, new object[] { 11UL, 23UL, 37UL })),
            ("qwen4exp.ple.head_offsets", 9U, (10U, new object[] { 0UL, 17UL, 36UL, 59UL, 88UL, 119UL, 156UL, 197UL })),
            ("qwen4exp.ple.head_vocab_sizes", 9U, (10U, new object[] { 17UL, 19UL, 23UL, 29UL, 31UL, 37UL, 41UL, 43UL })),
            ("tokenizer.ggml.tokens", 9U, (8U, BuildSyntheticQwen4ExpTokens(vocabSize).ToArray())),
            ("tokenizer.ggml.bos_token_id", 4U, 0U),
            ("tokenizer.ggml.eos_token_id", 4U, (uint)eosId),
            ("tokenizer.ggml.merges", 9U, (8U, Array.Empty<object>()))
        };

        var tensors = new List<GgufTensorDef>();
        int seed = 313;

        void AddF32(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 4];
            Span<float> fSpan = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                fSpan[(int)i] = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.F32, Data = bytes });
        }

        void AddOneNorm(string name, int len, bool asBf16 = false)
        {
            if (asBf16)
            {
                byte[] bytes = new byte[len * 2];
                ushort bfOne = (ushort)(BitConverter.SingleToUInt32Bits(1.0f) >> 16);
                MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan()).Fill(bfOne);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.BF16, Data = bytes });
            }
            else
            {
                byte[] bytes = new byte[len * 4];
                MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).Fill(1.0f);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.F32, Data = bytes });
            }
        }

        void AddBf16Matrix(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 2];
            Span<ushort> uSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                float val = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
                uSpan[(int)i] = (ushort)(BitConverter.SingleToUInt32Bits(val) >> 16);
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.BF16, Data = bytes });
        }

        // Core trunk embeddings & hc_head collapse WITHOUT output_norm.weight
        AddF32("token_embd.weight", dim, vocabSize);
        AddOneNorm("hc_head_norm.weight", hcDim, asBf16: true);
        AddF32("hc_head_down.weight", hcDim, hcLowRank);
        AddF32("hc_head_up.weight", hcLowRank, hcDim);
        AddF32("output.weight", dim, vocabSize);
        MemoryMarshal.Cast<byte, float>(tensors[^1].Data.AsSpan()).Slice(46 * dim, 2 * dim).Clear();
        AddBf16Matrix("per_layer_token_embd.weight", pleHeadDim, (ulong)pleTotalRows);

        // Layer 0: GDN + PLE + HC (merged down_inject) + MoE with larger shared expert
        AddOneNorm("blk.0.hc_attn_norm.weight", hcDim, asBf16: true);
        AddF32("blk.0.hc_attn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.0.hc_attn_up.weight", hcLowRank, hcDim);
        AddOneNorm("blk.0.hc_ffn_norm.weight", hcDim, asBf16: true);
        AddF32("blk.0.hc_ffn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.0.hc_ffn_up.weight", hcLowRank, hcDim);

        int pleEmbDim = (pleNgram - 1) * pleHeadsPerNgram * pleHeadDim;
        AddBf16Matrix("blk.0.ple_key.weight", (ulong)pleEmbDim, hcDim);
        AddBf16Matrix("blk.0.ple_value.weight", (ulong)pleEmbDim, dim);
        AddOneNorm("blk.0.ple_norm_key.weight", hcDim, asBf16: true);
        AddOneNorm("blk.0.ple_norm_query.weight", hcDim, asBf16: true);
        AddOneNorm("blk.0.ple_norm_conv.weight", hcDim, asBf16: true);
        AddF32("blk.0.ple_conv1d.weight", 4, hcDim);

        // Merged attn_qkv_gate & ssm_beta_alpha
        AddF32("blk.0.attn_qkv_gate.weight", dim, gdnConvChannels + ssmInnerSize);
        AddF32("blk.0.ssm_conv1d.weight", ssmConvKernel, gdnConvChannels);
        AddBf16Matrix("blk.0.ssm_a", (ulong)ssmTimeStepRank);
        AddF32("blk.0.ssm_beta_alpha.weight", dim, 2 * ssmTimeStepRank);
        AddF32("blk.0.ssm_dt.bias", (ulong)ssmTimeStepRank);
        AddOneNorm("blk.0.ssm_norm.weight", ssmStateSize, asBf16: true);
        AddF32("blk.0.ssm_out.weight", ssmInnerSize, dim);

        AddBf16Matrix("blk.0.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.0.ffn_gate_up_exps.weight", dim, 2 * expFfnDim, 4);
        AddF32("blk.0.ffn_down_exps.weight", expFfnDim, dim, 4);
        AddF32("blk.0.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.0.ffn_gate_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_up_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_down_shexp.weight", sharedFfnDim, dim);

        // Layer 1: QSA + HC (merged down_inject) + Merged Q-gated attn_qkv + MoE with 48-wide shared expert
        AddOneNorm("blk.1.hc_attn_norm.weight", hcDim);
        AddF32("blk.1.hc_attn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.1.hc_attn_up.weight", hcLowRank, hcDim);
        AddOneNorm("blk.1.hc_ffn_norm.weight", hcDim);
        AddF32("blk.1.hc_ffn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.1.hc_ffn_up.weight", hcLowRank, hcDim);

        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        AddF32("blk.1.attn_qkv.weight", dim, (ulong)(2 * qDim + 2 * kvDim));
        AddF32("blk.1.attn_output.weight", (ulong)qDim, dim);
        AddOneNorm("blk.1.attn_q_norm.weight", headDim, asBf16: true);
        AddOneNorm("blk.1.attn_k_norm.weight", headDim, asBf16: true);

        AddF32("blk.1.indexer.q_proj.weight", dim, idxHeads * idxKeyDim);
        AddF32("blk.1.indexer.k_proj.weight", dim, idxKeyDim);
        AddOneNorm("blk.1.indexer.q_norm.weight", idxKeyDim, asBf16: true);
        AddOneNorm("blk.1.indexer.k_norm.weight", idxKeyDim, asBf16: true);

        AddBf16Matrix("blk.1.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.1.ffn_gate_up_exps.weight", dim, 2 * expFfnDim, 4);
        AddF32("blk.1.ffn_down_exps.weight", expFfnDim, dim, 4);
        AddF32("blk.1.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.1.ffn_gate_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.1.ffn_up_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.1.ffn_down_shexp.weight", sharedFfnDim, dim);

        WriteGgufFile(path, kv, tensors);
    }

    private static void BuildDetachedMtpSidecarGguf(string path)
    {
        const int dim = 32;
        const int expFfnDim = 32;
        const int sharedFfnDim = 48;
        const int vocabSize = 48;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int headDim = 8;
        const int hcCount = 4;
        const int hcLowRank = 4;
        const int hcDim = hcCount * dim; // 128
        const int idxHeads = 2;
        const int idxKeyDim = 16;
        const int idxTopK = 4;
        const int eosId = 47;

        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen4exp"),
            ("general.alignment", 4U, 32U),
            ("qwen4exp.block_count", 4U, 1U),
            ("qwen4exp.embedding_length", 4U, (uint)dim),
            ("qwen4exp.context_length", 4U, 128U),
            ("qwen4exp.attention.head_count", 4U, (uint)nHeads),
            ("qwen4exp.attention.head_count_kv", 4U, (uint)nHeadsKv),
            ("qwen4exp.attention.key_length", 4U, (uint)headDim),
            ("qwen4exp.attention.value_length", 4U, (uint)headDim),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen4exp.rope.dimension_count", 4U, (uint)headDim),
            ("qwen4exp.rope.freq_base", 6U, 10000.0f),
            ("qwen4exp.expert_count", 4U, 4U),
            ("qwen4exp.expert_used_count", 4U, 2U),
            ("qwen4exp.expert_feed_forward_length", 4U, (uint)expFfnDim),
            ("qwen4exp.expert_shared_feed_forward_length", 4U, (uint)sharedFfnDim),
            ("qwen4exp.hyper_connection.count", 4U, (uint)hcCount),
            ("qwen4exp.hyper_connection.low_rank", 4U, (uint)hcLowRank),
            ("qwen4exp.attention.indexer.head_count", 4U, (uint)idxHeads),
            ("qwen4exp.attention.indexer.key_length", 4U, (uint)idxKeyDim),
            ("qwen4exp.attention.indexer.top_k", 4U, (uint)idxTopK),
            ("qwen4exp.attention.compress_ratios", 9U, (4U, new object[] { 4U })),
            ("tokenizer.ggml.bos_token_id", 4U, 0U),
            ("tokenizer.ggml.eos_token_id", 4U, (uint)eosId)
        };

        var tensors = new List<GgufTensorDef>();
        int seed = 707;

        void AddF32(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 4];
            Span<float> fSpan = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                fSpan[(int)i] = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.F32, Data = bytes });
        }

        void AddOneNorm(string name, int len, bool asBf16 = false)
        {
            if (asBf16)
            {
                byte[] bytes = new byte[len * 2];
                ushort bfOne = (ushort)(BitConverter.SingleToUInt32Bits(1.0f) >> 16);
                MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan()).Fill(bfOne);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.BF16, Data = bytes });
            }
            else
            {
                byte[] bytes = new byte[len * 4];
                MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).Fill(1.0f);
                tensors.Add(new GgufTensorDef { Name = name, Dims = [(ulong)len], Type = GgufType.F32, Data = bytes });
            }
        }

        void AddBf16Matrix(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 2];
            Span<ushort> uSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                float val = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.08f;
                uSpan[(int)i] = (ushort)(BitConverter.SingleToUInt32Bits(val) >> 16);
            }
            tensors.Add(new GgufTensorDef { Name = name, Dims = dims, Type = GgufType.BF16, Data = bytes });
        }

        // Detached sidecar top-level tensors (no shared_head_* or output_norm.weight)
        AddF32("token_embd.weight", dim, vocabSize);
        AddOneNorm("output_hc_norm.weight", hcDim, asBf16: true);
        AddF32("output_hc_down.weight", hcDim, hcLowRank);
        AddF32("output_hc_up.weight", hcLowRank, hcDim);
        AddF32("output.weight", dim, vocabSize);
        MemoryMarshal.Cast<byte, float>(tensors[^1].Data.AsSpan()).Slice(46 * dim, 2 * dim).Clear();

        // NextN multi-stream block
        AddOneNorm("blk.0.nextn.enorm.weight", dim, asBf16: true);
        AddOneNorm("blk.0.nextn.hnorm.weight", hcDim, asBf16: true);
        AddF32("blk.0.nextn.eh_proj.weight", 2 * dim, dim);

        AddOneNorm("blk.0.hc_attn_norm.weight", hcDim, asBf16: true);
        AddF32("blk.0.hc_attn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.0.hc_attn_up.weight", hcLowRank, hcDim);
        AddOneNorm("blk.0.hc_ffn_norm.weight", hcDim, asBf16: true);
        AddF32("blk.0.hc_ffn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddF32("blk.0.hc_ffn_up.weight", hcLowRank, hcDim);

        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        AddF32("blk.0.attn_qkv.weight", dim, (ulong)(2 * qDim + 2 * kvDim));
        AddF32("blk.0.attn_output.weight", (ulong)qDim, dim);
        AddOneNorm("blk.0.attn_q_norm.weight", headDim, asBf16: true);
        AddOneNorm("blk.0.attn_k_norm.weight", headDim, asBf16: true);

        AddF32("blk.0.indexer.q_proj.weight", dim, idxHeads * idxKeyDim);
        AddF32("blk.0.indexer.k_proj.weight", dim, idxKeyDim);
        AddOneNorm("blk.0.indexer.q_norm.weight", idxKeyDim, asBf16: true);
        AddOneNorm("blk.0.indexer.k_norm.weight", idxKeyDim, asBf16: true);

        AddBf16Matrix("blk.0.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.0.ffn_gate_up_exps.weight", dim, 2 * expFfnDim, 4);
        AddF32("blk.0.ffn_down_exps.weight", expFfnDim, dim, 4);
        AddF32("blk.0.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.0.ffn_gate_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_up_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_down_shexp.weight", sharedFfnDim, dim);

        WriteGgufFile(path, kv, tensors);
    }

    private static void WriteGgufFile(
        string path,
        List<(string Key, uint Type, object Value)> kv,
        List<GgufTensorDef> tensors)
    {
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offset = (offset + 31L) & ~31L;
            tensors[i].Offset = offset;
            offset += tensors[i].Data.Length;
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        bw.Write(0x46554747U); // 'GGUF'
        bw.Write(3U);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kv.Count);

        static void WriteStr(BinaryWriter w, string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        static void WriteVal(BinaryWriter w, uint type, object val)
        {
            switch (type)
            {
                case 4U: w.Write((uint)val); break;
                case 5U: w.Write((int)val); break;
                case 6U: w.Write((float)val); break;
                case 8U: WriteStr(w, (string)val); break;
                case 10U: w.Write((ulong)val); break;
                case 9U:
                {
                    var (elemType, arr) = ((uint, object[]))val;
                    w.Write(elemType);
                    w.Write((ulong)arr.Length);
                    for (int j = 0; j < arr.Length; j++) WriteVal(w, elemType, arr[j]);
                    break;
                }
                default:
                    throw new NotSupportedException($"Test GGUF writer unsupported type {type}.");
            }
        }

        foreach (var (key, type, value) in kv)
        {
            WriteStr(bw, key);
            bw.Write(type);
            WriteVal(bw, type, value);
        }

        foreach (var t in tensors)
        {
            WriteStr(bw, t.Name);
            bw.Write((uint)t.Dims.Length);
            foreach (ulong d in t.Dims) bw.Write(d);
            bw.Write((uint)t.Type);
            bw.Write((ulong)t.Offset);
        }

        long pos = fs.Position;
        long dataStart = (pos + 31L) & ~31L;
        if (dataStart > pos)
        {
            bw.Write(new byte[dataStart - pos]);
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            long targetPos = dataStart + tensors[i].Offset;
            if (fs.Position < targetPos)
            {
                bw.Write(new byte[targetPos - fs.Position]);
            }
            bw.Write(tensors[i].Data);
        }
    }
}
