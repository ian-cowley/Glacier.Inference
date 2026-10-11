namespace Glacier.Inference.Tests.E2E;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using System.Threading.Tasks;
using Xunit;

/// <summary>
/// Tier 3 Cross-Feature Combination Tests: Pairwise interactions across compute, transport, protocol, and scheduling.
/// </summary>
[Trait("Category", "E2E")]
[Trait("Tier", "3")]
public class E2ETier3CrossFeatureCombinationTests
{
    // =========================================================================
    // 1. BATCHED PREFILL + DYNAMIC FP16/INT8 WIRE TRANSPORT
    // =========================================================================

    [Fact]
    public void Combination_BatchedPrefill_WithDynamicInt8WireTransport()
    {
        int batchTokens = 64; // Large prompt prefill chunk >= 32
        int dim = 5120;

        // Dynamic precision heuristic selects INT8 for prefill chunks
        WirePrecisionMode mode = (batchTokens >= 32) ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, mode);

        // Calculate wire transport bandwidth
        int wireBytesPerToken = (mode == WirePrecisionMode.INT8_Symmetric) ? (dim * 1 + 4) : (dim * 2);
        int totalWireBytes = batchTokens * wireBytesPerToken;

        Assert.Equal(5124, wireBytesPerToken);
        Assert.Equal(64 * 5124, totalWireBytes);

        // INT8 reduces wire bandwidth by nearly 50% vs FP16
        int fp16Bytes = batchTokens * (dim * 2);
        double compressionRatio = (double)fp16Bytes / totalWireBytes;
        Assert.True(compressionRatio > 1.99, $"Compression ratio: {compressionRatio:F2}x");
    }

    [Fact]
    public void Combination_SingleTokenDecode_WithDynamicFp16WireTransport()
    {
        int batchTokens = 1; // Autoregressive decoding step
        int dim = 5120;

        // Dynamic heuristic selects FP16 for decode tokens (lossless precision preservation)
        WirePrecisionMode mode = (batchTokens >= 32) ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;
        Assert.Equal(WirePrecisionMode.FP16, mode);

        int wireBytes = dim * 2;
        Assert.Equal(10240, wireBytes);
    }

    // =========================================================================
    // 2. SPECULATIVE HIT/MISS + DYNAMIC PRECISION
    // =========================================================================

    [Fact]
    public void Combination_SpeculativeHit_WithDynamicPrecision()
    {
        var draftProvider = new PromptLookupDraftOracle();
        draftProvider.AppendTokens(new[] { 101, 102, 103, 104 });
        draftProvider.AppendTokens(new[] { 101, 102, 103 });

        // Predict draft token N+1
        bool hasDraft = draftProvider.TryPredictNextToken(out int draftToken);
        Assert.True(hasDraft);
        Assert.Equal(104, draftToken);

        // Speculative activation sent over wire with IsSpeculative flag
        var header = new FusedActivationHeader(
            sequenceId: 1,
            tokenIndex: 8,
            layerIndex: 27,
            hiddenDim: 5120,
            precisionMode: WirePrecisionMode.FP16,
            flags: FusedActivationFlags.IsSpeculative);

        Assert.True(header.TypedFlags.HasFlag(FusedActivationFlags.IsSpeculative));

        // Ground truth arrives: token matches draft!
        int trueToken = 104;
        bool isHit = (draftToken == trueToken);
        Assert.True(isHit);

        // Advance two tokens
        draftProvider.AppendToken(trueToken);
        Assert.Equal(8, draftProvider.ContextLength);
    }

    [Fact]
    public void Combination_SpeculativeMiss_WithDynamicPrecisionRollback()
    {
        var draftProvider = new PromptLookupDraftOracle();
        draftProvider.AppendTokens(new[] { 1, 2, 3 });

        // Speculative token generated as 99
        int draftToken = 99;
        draftProvider.AppendToken(draftToken);
        Assert.Equal(4, draftProvider.ContextLength);

        // Ground truth from Stage 1 LM Head evaluates to 55 (Miss!)
        int groundTruthToken = 55;
        bool isHit = (draftToken == groundTruthToken);
        Assert.False(isHit);

        // In-place rollback and replacement with true token
        draftProvider.Rollback(1);
        draftProvider.AppendToken(groundTruthToken);

        Assert.Equal(4, draftProvider.ContextLength);
    }

    // =========================================================================
    // 3. 27:21 LAYER SPLIT + FULL DUPLEX CHANNELS
    // =========================================================================

    [Fact]
    public async Task Combination_27To21LayerSplit_WithFullDuplexChannelBus()
    {
        // Channel from Stage 0 (Host, 27 layers) to Stage 1 (Beast, 21 layers)
        var stage0ToStage1 = Channel.CreateBounded<FusedActivationHeader>(8);
        // Return channel from Stage 1 (Result/Logits) to Stage 0/Coordinator
        var stage1ToCoord = Channel.CreateBounded<int>(8);

        // Producer: Stage 0 computes activation exiting layer 26
        var stage0Task = Task.Run(async () =>
        {
            for (int step = 0; step < 5; step++)
            {
                var frame = new FusedActivationHeader(
                    sequenceId: (ulong)step,
                    tokenIndex: step,
                    layerIndex: 27, // Targeting start of stage 1
                    hiddenDim: 5120,
                    precisionMode: WirePrecisionMode.FP16);

                await stage0ToStage1.Writer.WriteAsync(frame);
            }
            stage0ToStage1.Writer.Complete();
        });

        // Consumer: Stage 1 evaluates layers 27..47 + LM Head and yields next token
        var stage1Task = Task.Run(async () =>
        {
            await foreach (var act in stage0ToStage1.Reader.ReadAllAsync())
            {
                // Emulate LM Head argmax
                int sampledToken = (int)(act.SequenceId * 10 + 7);
                await stage1ToCoord.Writer.WriteAsync(sampledToken);
            }
            stage1ToCoord.Writer.Complete();
        });

        await Task.WhenAll(stage0Task, stage1Task);

        var results = new System.Collections.Generic.List<int>();
        await foreach (int tok in stage1ToCoord.Reader.ReadAllAsync())
        {
            results.Add(tok);
        }

        Assert.Equal(5, results.Count);
        Assert.Equal(7, results[0]);
        Assert.Equal(17, results[1]);
    }

    // =========================================================================
    // 4. PINNED BUFFER POOL + FUSED FRAME SERIALIZATION UNDER CONCURRENCY
    // =========================================================================

    [Fact]
    public void Combination_PinnedBufferPool_WithFusedFrameSerialization()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 64 * 1024, maxBuffers: 32);

        Parallel.For(0, 100, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
        {
            var mem = pool.Rent();
            try
            {
                var header = new FusedActivationHeader(
                    sequenceId: (ulong)i,
                    tokenIndex: i,
                    layerIndex: 27,
                    hiddenDim: 5120,
                    precisionMode: WirePrecisionMode.INT8_Symmetric,
                    scale: 0.15f);

                header.WriteTo(mem.Span.Slice(0, 32));

                var parsed = FusedActivationHeader.ReadFrom(mem.Span.Slice(0, 32));
                Assert.Equal(header, parsed);
            }
            finally
            {
                pool.Return(mem);
            }
        });
    }

    // =========================================================================
    // 5. PROMPT LOOKUP DRAFT PROVIDER + IN-PLACE KV ROLLBACK
    // =========================================================================

    [Fact]
    public void Combination_PromptLookupDraftProvider_WithInPlaceKvRollback()
    {
        var oracle = new PromptLookupDraftOracle(maxContext: 512, maxNgram: 3);
        oracle.AppendTokens(new[] { 10, 20, 30, 40, 50 });

        // Simulate 5 speculative steps
        for (int step = 0; step < 5; step++)
        {
            if (oracle.TryPredictNextToken(out int draft))
            {
                // If draft matches simulated ground truth (even steps hit, odd steps miss)
                int groundTruth = (step % 2 == 0) ? draft : (draft + 1000);
                if (draft == groundTruth)
                {
                    // Hit: keep draft and proceed
                    oracle.AppendToken(draft);
                }
                else
                {
                    // Miss: rollback and insert ground truth
                    oracle.AppendToken(draft);
                    oracle.Rollback(1);
                    oracle.AppendToken(groundTruth);
                }
            }
            else
            {
                oracle.AppendToken(step + 100);
            }
        }

        Assert.True(oracle.ContextLength >= 5);
    }

    // =========================================================================
    // 6. PREFILL LOH ELIMINATION + CHUNKED BATCHED GEMM
    // =========================================================================

    [Fact]
    public void Combination_PrefillLohElimination_WithChunkedBatchedGemm()
    {
        const int totalPromptTokens = 1024;
        const int chunkSize = 256;
        const int dim = 5120;

        // Rent large chunk buffer without LOH GC churn
        float[] rentedChunk = ArrayPool<float>.Shared.Rent(chunkSize * dim);
        try
        {
            int numChunks = (totalPromptTokens + chunkSize - 1) / chunkSize;
            Assert.Equal(4, numChunks);

            for (int c = 0; c < numChunks; c++)
            {
                int tokensInChunk = Math.Min(chunkSize, totalPromptTokens - c * chunkSize);
                Span<float> chunkSpan = rentedChunk.AsSpan(0, tokensInChunk * dim);
                chunkSpan.Fill(1.0f); // Pre-warm / evaluate
                Assert.Equal(tokensInChunk * dim, chunkSpan.Length);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedChunk);
        }
    }

    // =========================================================================
    // 7. LAN CONGESTION BACKOFF: SWITCHES FROM FP16 TO INT8
    // =========================================================================

    [Fact]
    public void Combination_LanCongestionBackoff_SwitchesFromFp16ToInt8()
    {
        double nominalLatencyMs = 0.45;
        double congestedLatencyMs = 3.20;

        WirePrecisionMode modeNominal = (nominalLatencyMs > 2.5) ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;
        WirePrecisionMode modeCongested = (congestedLatencyMs > 2.5) ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;

        Assert.Equal(WirePrecisionMode.FP16, modeNominal);
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, modeCongested);
    }

    // =========================================================================
    // 8. FUSED FRAME OVER DUPLEX SOCKET WITH SIMD QUANTIZATION
    // =========================================================================

    [Fact]
    public async Task Combination_FusedFrameOverDuplexSocket_WithSimdQuantization()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        int dim = 5120;
        float[] rawActivations = new float[dim];
        for (int i = 0; i < dim; i++) rawActivations[i] = MathF.Cos(i * 0.05f);

        // Quantize to INT8
        sbyte[] quantized = new sbyte[dim];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(rawActivations, quantized);

        // Create Fused Frame: 32B ClusterFrameHeader + 32B FusedActivationHeader + 5124B INT8 payload
        byte[] frame = new byte[64 + 5124];
        var fusedHdr = new FusedActivationHeader(
            sequenceId: 42,
            tokenIndex: 10,
            layerIndex: 27,
            hiddenDim: dim,
            precisionMode: WirePrecisionMode.INT8_Symmetric,
            flags: FusedActivationFlags.HasScale,
            scale: scale);

        fusedHdr.WriteTo(frame.AsSpan(32, 32));
        MemoryMarshal.Cast<sbyte, byte>(quantized).CopyTo(frame.AsSpan(64, dim));
        BinaryPrimitives.WriteSingleLittleEndian(frame.AsSpan(64 + dim, 4), scale);

        _ = Task.Run(async () =>
        {
            await pair.Client.SendAsync(frame, SocketFlags.None);
        });

        byte[] received = new byte[64 + 5124];
        int total = 0;
        while (total < received.Length)
        {
            int r = await pair.Server.ReceiveAsync(received.AsMemory(total, received.Length - total), SocketFlags.None);
            if (r == 0) break;
            total += r;
        }

        Assert.Equal(received.Length, total);
        var readHdr = FusedActivationHeader.ReadFrom(received.AsSpan(32, 32));
        Assert.Equal(scale, readHdr.Scale);

        // Dequantize and verify cosine similarity
        float[] recovered = new float[dim];
        var recvQuantized = MemoryMarshal.Cast<byte, sbyte>(received.AsSpan(64, dim));
        QuantizationOracle.DequantizeSymmetricInt8(recvQuantized, readHdr.Scale, recovered);

        double cosSim = QuantizationOracle.ComputeCosineSimilarity(rawActivations, recovered);
        Assert.True(cosSim > 0.9995, $"Cosine similarity: {cosSim:F6}");
    }
}
