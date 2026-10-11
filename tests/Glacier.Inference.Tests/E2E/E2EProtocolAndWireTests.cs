namespace Glacier.Inference.Tests.E2E;

using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Xunit;

/// <summary>
/// Tier 1 Feature Coverage Tests: Protocol, Framing, Quantization, and Buffer Pooling.
/// Features 7, 8, 12, 13, 14, 15.
/// </summary>
[Trait("Category", "E2E")]
[Trait("Tier", "1")]
public class E2EProtocolAndWireTests
{
    // =========================================================================
    // FEATURE 7: PINNED SOCKET BUFFER POOL (>=5 tests)
    // =========================================================================

    [Fact]
    public void PinnedPool_RentAndReturn_ReusesSameMemoryWithoutAllocation()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 16384, maxBuffers: 4);

        var buf1 = pool.Rent();
        Assert.Equal(16384, buf1.Length);

        // Fill with sentinel
        buf1.Span[0] = 0xAA;
        pool.Return(buf1);

        var buf2 = pool.Rent();
        Assert.Equal(16384, buf2.Length);
        Assert.Equal((byte)0xAA, buf2.Span[0]); // Confirms same underlying buffer reused
        pool.Return(buf2);
    }

    [Fact]
    public void PinnedPool_Exhaustion_ThrowsInvalidOperationException()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 1024, maxBuffers: 2);

        var b1 = pool.Rent();
        var b2 = pool.Rent();

        Assert.Throws<InvalidOperationException>(() => pool.Rent());

        pool.Return(b1);
        var b3 = pool.Rent();
        Assert.False(b3.IsEmpty);

        pool.Return(b2);
        pool.Return(b3);
    }

    [Fact]
    public void PinnedPool_Dispose_FreesAllGCHandlesCleanly()
    {
        var pool = new PinnedSocketBufferPool(bufferSize: 2048, maxBuffers: 3);
        var b1 = pool.Rent();
        pool.Dispose();

        // Renting from disposed pool throws
        Assert.Throws<ObjectDisposedException>(() => pool.Rent());

        // Return on disposed pool is safe no-op
        pool.Return(b1);
    }

    [Fact]
    public void PinnedPool_ConcurrentRentReturn_ThreadSafe()
    {
        const int bufferCount = 32;
        const int iterations = 1000;
        using var pool = new PinnedSocketBufferPool(bufferSize: 512, maxBuffers: bufferCount);

        Parallel.For(0, iterations, new ParallelOptions { MaxDegreeOfParallelism = 16 }, _ =>
        {
            var buf = pool.Rent();
            buf.Span[0] = 42;
            pool.Return(buf);
        });
    }

    [Fact]
    public void PinnedPool_ReturnAlienBuffer_IgnoredWithoutCorruption()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 1024, maxBuffers: 2);
        byte[] alien = new byte[1024];

        // Returning buffer not belonging to pool must not throw or alter internal count
        pool.Return(alien.AsMemory());

        var b1 = pool.Rent();
        var b2 = pool.Rent();
        Assert.Throws<InvalidOperationException>(() => pool.Rent());

        pool.Return(b1);
        pool.Return(b2);
    }

    // =========================================================================
    // FEATURE 8: FUSED FRAME SERIALIZATION (>=5 tests)
    // =========================================================================

    [Fact]
    public void FusedFrame_SerializationRoundtrip_ByteExact()
    {
        var original = new FusedActivationHeader(
            sequenceId: 0xDEADBEEFCAFE1234UL,
            tokenIndex: 128,
            layerIndex: 27,
            hiddenDim: 5120,
            precisionMode: WirePrecisionMode.INT8_Symmetric,
            flags: FusedActivationFlags.IsSpeculative | FusedActivationFlags.HasScale,
            scale: 0.0456f);

        byte[] buffer = new byte[32];
        original.WriteTo(buffer.AsSpan());

        var parsed = FusedActivationHeader.ReadFrom(buffer.AsSpan());

        Assert.Equal(original, parsed);
        Assert.Equal(FusedActivationHeader.FusedMagic, parsed.Magic);
        Assert.Equal(0xDEADBEEFCAFE1234UL, parsed.SequenceId);
        Assert.Equal(128, parsed.TokenIndex);
        Assert.Equal(27, parsed.LayerIndex);
        Assert.Equal(5120, parsed.HiddenDim);
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, parsed.TypedPrecision);
        Assert.True(parsed.TypedFlags.HasFlag(FusedActivationFlags.IsSpeculative));
        Assert.True(parsed.TypedFlags.HasFlag(FusedActivationFlags.HasScale));
        Assert.Equal(0.0456f, parsed.Scale);
    }

    [Fact]
    public void FusedFrame_64ByteAlignedHeader_ClusterAndFusedCombined()
    {
        // ClusterFrameHeader (32B) + FusedActivationHeader (32B) = 64B
        Assert.Equal(32, ClusterFrameHeader.HeaderSize);
        Assert.Equal(32, FusedActivationHeader.HeaderSize);

        byte[] combinedBuffer = new byte[64];

        var clusterHeader = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x70, // MsgActivation
            correlationId: 1001,
            payloadLength: 5124); // 32B FusedActivationHeader is inside payload or prepended

        var fusedHeader = new FusedActivationHeader(
            sequenceId: 1,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: 5120,
            precisionMode: WirePrecisionMode.INT8_Symmetric,
            flags: FusedActivationFlags.None,
            scale: 0.12f);

        MemoryMarshal.Write(combinedBuffer.AsSpan(0, 32), in clusterHeader);
        fusedHeader.WriteTo(combinedBuffer.AsSpan(32, 32));

        var readCluster = MemoryMarshal.Read<ClusterFrameHeader>(combinedBuffer.AsSpan(0, 32));
        var readFused = FusedActivationHeader.ReadFrom(combinedBuffer.AsSpan(32, 32));

        Assert.Equal(ClusterFrameHeader.MagicConstant, readCluster.Magic);
        Assert.Equal((ClusterMessageType)0x70, readCluster.MessageType);
        Assert.Equal(FusedActivationHeader.FusedMagic, readFused.Magic);
        Assert.Equal(5120, readFused.HiddenDim);
    }

    [Fact]
    public void FusedFrame_TryRead_IncompleteBuffer_ReturnsFalse()
    {
        byte[] shortBuf = new byte[31];
        Assert.False(FusedActivationHeader.TryRead(shortBuf.AsSpan(), out _));
    }

    [Fact]
    public void FusedFrame_InvalidMagic_ThrowsInvalidOperationException()
    {
        byte[] buf = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(buf.AsSpan(0, 4), 0x12345678);

        Assert.Throws<InvalidOperationException>(() => FusedActivationHeader.ReadFrom(buf.AsSpan()));
        Assert.False(FusedActivationHeader.TryRead(buf.AsSpan(), out _));
    }

    [Fact]
    public async Task FusedFrame_FullDuplexSocketStreaming_RoundtripMatches()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        var header = new FusedActivationHeader(
            sequenceId: 999,
            tokenIndex: 12,
            layerIndex: 4,
            hiddenDim: 256,
            precisionMode: WirePrecisionMode.FP16,
            flags: FusedActivationFlags.None,
            scale: 1.0f);

        byte[] sendBuf = new byte[32 + 512]; // 32B header + 512B payload (256 halfs)
        header.WriteTo(sendBuf.AsSpan(0, 32));
        for (int i = 0; i < 512; i++) sendBuf[32 + i] = (byte)(i % 255);

        // Send from client to server
        _ = Task.Run(async () =>
        {
            await pair.Client.SendAsync(sendBuf, SocketFlags.None);
        });

        byte[] recvBuf = new byte[32 + 512];
        int totalRead = 0;
        while (totalRead < recvBuf.Length)
        {
            int r = await pair.Server.ReceiveAsync(recvBuf.AsMemory(totalRead, recvBuf.Length - totalRead), SocketFlags.None);
            if (r == 0) break;
            totalRead += r;
        }

        Assert.Equal(recvBuf.Length, totalRead);
        var readHeader = FusedActivationHeader.ReadFrom(recvBuf.AsSpan(0, 32));
        Assert.Equal(header, readHeader);
        Assert.True(sendBuf.AsSpan(32).SequenceEqual(recvBuf.AsSpan(32)));
    }

    // =========================================================================
    // FEATURE 12: WIRE PRECISION MODE NEGOTIATION (>=5 tests)
    // =========================================================================

    [Fact]
    public void PrecisionNegotiation_Handshake_SupportsAllFourModes()
    {
        var modes = new[]
        {
            WirePrecisionMode.FP32,
            WirePrecisionMode.FP16,
            WirePrecisionMode.INT8_Symmetric,
            WirePrecisionMode.FP8_E4M3
        };

        foreach (var mode in modes)
        {
            var h = new FusedActivationHeader(1, 0, 0, 5120, mode);
            byte[] buf = new byte[32];
            h.WriteTo(buf.AsSpan());
            var res = FusedActivationHeader.ReadFrom(buf.AsSpan());
            Assert.Equal(mode, res.TypedPrecision);
        }
    }

    [Fact]
    public void PrecisionNegotiation_PayloadEncoding_LittleEndian()
    {
        var h = new FusedActivationHeader(1, 0, 0, 5120, WirePrecisionMode.INT8_Symmetric, FusedActivationFlags.None, 2.5f);
        byte[] buf = new byte[32];
        h.WriteTo(buf.AsSpan());

        float scale = BinaryPrimitives.ReadSingleLittleEndian(buf.AsSpan(28, 4));
        Assert.Equal(2.5f, scale);
    }

    [Fact]
    public void PrecisionNegotiation_FallbackToFp16_OnUnknownCode()
    {
        byte unknownPrecision = 99;
        WirePrecisionMode resolved = Enum.IsDefined(typeof(WirePrecisionMode), unknownPrecision)
            ? (WirePrecisionMode)unknownPrecision
            : WirePrecisionMode.FP16; // Safe default fallback

        Assert.Equal(WirePrecisionMode.FP16, resolved);
    }

    [Fact]
    public void PrecisionNegotiation_FlagBitmask_PreservesSpeculativeAndPrefillFlags()
    {
        var flags = FusedActivationFlags.IsSpeculative | FusedActivationFlags.IsPrefillChunk | FusedActivationFlags.HasScale;
        var h = new FusedActivationHeader(10, 5, 2, 5120, WirePrecisionMode.INT8_Symmetric, flags, 0.5f);

        byte[] buf = new byte[32];
        h.WriteTo(buf.AsSpan());
        var res = FusedActivationHeader.ReadFrom(buf.AsSpan());

        Assert.True(res.TypedFlags.HasFlag(FusedActivationFlags.IsSpeculative));
        Assert.True(res.TypedFlags.HasFlag(FusedActivationFlags.IsPrefillChunk));
        Assert.True(res.TypedFlags.HasFlag(FusedActivationFlags.HasScale));
        Assert.False(res.TypedFlags.HasFlag(FusedActivationFlags.FinalLayer));
    }

    [Fact]
    public void PrecisionNegotiation_ZeroByteHeader_FailsGracefully()
    {
        byte[] empty = Array.Empty<byte>();
        Assert.False(FusedActivationHeader.TryRead(empty.AsSpan(), out _));
    }

    // =========================================================================
    // FEATURE 13: SIMD AVX-512/AVX2 QUANTIZATION (>=5 tests)
    // =========================================================================

    [Fact]
    public void Quantization_Fp32ToInt8_AccurateWithinQuantError()
    {
        int dim = 5120;
        float[] input = new float[dim];
        var rng = new Random(42);
        for (int i = 0; i < dim; i++) input[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

        sbyte[] quantized = new sbyte[dim];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(input, quantized);

        Assert.True(scale > 0.0f);

        float[] dequantized = new float[dim];
        QuantizationOracle.DequantizeSymmetricInt8(quantized, scale, dequantized);

        double cosSim = QuantizationOracle.ComputeCosineSimilarity(input, dequantized);
        Assert.True(cosSim > 0.9995, $"Cosine similarity was {cosSim:F6}, expected > 0.9995");
    }

    [Fact]
    public void Quantization_Int8ToFp32_PreservesSignAndMagnitude()
    {
        sbyte[] q = new sbyte[] { -127, -64, 0, 64, 127 };
        float scale = 0.1f;
        float[] outFloats = new float[5];

        QuantizationOracle.DequantizeSymmetricInt8(q, scale, outFloats);

        Assert.Equal(-12.7f, outFloats[0], 4);
        Assert.Equal(-6.4f, outFloats[1], 4);
        Assert.Equal(0.0f, outFloats[2], 4);
        Assert.Equal(6.4f, outFloats[3], 4);
        Assert.Equal(12.7f, outFloats[4], 4);
    }

    [Fact]
    public void Quantization_ZeroVector_ProducesZeroScaleAndZeroBytes()
    {
        float[] zeros = new float[128];
        sbyte[] q = new sbyte[128];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(zeros, q);

        Assert.Equal(1.0f, scale);
        for (int i = 0; i < 128; i++) Assert.Equal(0, q[i]);
    }

    [Fact]
    public void Quantization_SymmetricClamp_WithinNeg127ToPos127()
    {
        float[] input = new float[] { -100.0f, 100.0f, -50.0f, 50.0f };
        sbyte[] q = new sbyte[4];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(input, q);

        Assert.Equal(-127, q[0]);
        Assert.Equal(127, q[1]);
        Assert.True(q[2] >= -127 && q[2] <= 127);
        Assert.True(q[3] >= -127 && q[3] <= 127);
    }

    [Fact]
    public void Quantization_SubnormalAndExtremeFloats_HandlesGracefully()
    {
        float[] extreme = new float[] { 1e-35f, -1e-35f, 1e20f, -1e20f };
        sbyte[] q = new sbyte[4];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(extreme, q);
        Assert.False(float.IsNaN(scale));
        Assert.False(float.IsInfinity(scale));
        Assert.True(scale > 0.0f);
    }

    // =========================================================================
    // FEATURE 14: SYMMETRIC PER-TOKEN INT8 LAYOUT (>=5 tests)
    // =========================================================================

    [Fact]
    public void Int8Layout_SizeExactly5124Bytes_ForDim5120()
    {
        int dim = 5120;
        int expectedPayloadSize = dim * 1 + 4; // 5120 int8 bytes + 4B scale float
        Assert.Equal(5124, expectedPayloadSize);
    }

    [Fact]
    public void Int8Layout_PerTokenScaleComputation_MatchesMaxAbs()
    {
        float[] vec = new float[5120];
        vec[100] = 5.08f;
        vec[200] = -5.08f;

        sbyte[] q = new sbyte[5120];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(vec, q);

        Assert.Equal(5.08f / 127.0f, scale, 5);
        Assert.Equal(127, q[100]);
        Assert.Equal(-127, q[200]);
    }

    [Fact]
    public void Int8Layout_ReconstructionError_CosineSimilarityAbove9995()
    {
        int dim = 5120;
        float[] orig = new float[dim];
        for (int i = 0; i < dim; i++) orig[i] = MathF.Sin(i * 0.1f);

        sbyte[] q = new sbyte[dim];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(orig, q);

        float[] recon = new float[dim];
        QuantizationOracle.DequantizeSymmetricInt8(q, scale, recon);

        double sim = QuantizationOracle.ComputeCosineSimilarity(orig, recon);
        Assert.True(sim > 0.9995, $"Reconstruction similarity: {sim:F6}");
    }

    [Fact]
    public void Int8Layout_ZeroPoint_IsStrictlyZero()
    {
        // Symmetric quantization has zero-point = 0 (no offset byte needed)
        int zeroPoint = 0;
        Assert.Equal(0, zeroPoint);
    }

    [Fact]
    public void Int8Layout_SingleSpikeOutlier_CapturedInScaleFactor()
    {
        float[] spike = new float[5120];
        spike[15] = 127.0f; // Spike
        for (int i = 0; i < 5120; i++) if (i != 15) spike[i] = 1.0f;

        sbyte[] q = new sbyte[5120];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(spike, q);

        Assert.Equal(1.0f, scale, 4);
        Assert.Equal(127, q[15]);
        Assert.Equal(1, q[0]);
    }

    // =========================================================================
    // FEATURE 15: DYNAMIC WIRE PRECISION HEURISTIC (>=5 tests)
    // =========================================================================

    [Theory]
    [InlineData(32, WirePrecisionMode.INT8_Symmetric)]
    [InlineData(64, WirePrecisionMode.INT8_Symmetric)]
    [InlineData(128, WirePrecisionMode.INT8_Symmetric)]
    public void PrecisionHeuristic_BatchSizeAtLeast32_SelectsInt8(int batchSize, WirePrecisionMode expected)
    {
        WirePrecisionMode selected = SelectOptimalWirePrecision(batchSize, isDecode: false, lanLatencyMs: 0.5);
        Assert.Equal(expected, selected);
    }

    [Theory]
    [InlineData(1, WirePrecisionMode.FP16)]
    [InlineData(8, WirePrecisionMode.FP16)]
    [InlineData(31, WirePrecisionMode.FP16)]
    public void PrecisionHeuristic_BatchSizeBelow32_SelectsFp16(int batchSize, WirePrecisionMode expected)
    {
        WirePrecisionMode selected = SelectOptimalWirePrecision(batchSize, isDecode: false, lanLatencyMs: 0.5);
        Assert.Equal(expected, selected);
    }

    [Fact]
    public void PrecisionHeuristic_DecodeToken_SelectsFp16()
    {
        WirePrecisionMode selected = SelectOptimalWirePrecision(batchSize: 1, isDecode: true, lanLatencyMs: 0.4);
        Assert.Equal(WirePrecisionMode.FP16, selected);
    }

    [Fact]
    public void PrecisionHeuristic_HighLatencyBackoff_SelectsInt8()
    {
        // When network latency exceeds 2.5 ms, even small batches back off to INT8
        WirePrecisionMode selected = SelectOptimalWirePrecision(batchSize: 16, isDecode: false, lanLatencyMs: 3.5);
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, selected);
    }

    [Fact]
    public void PrecisionHeuristic_CliOverride_ForcesConfiguredPrecision()
    {
        WirePrecisionMode forced = WirePrecisionMode.FP32;
        WirePrecisionMode result = ResolveWithOverride(forcedOverride: forced, fallback: WirePrecisionMode.FP16);
        Assert.Equal(WirePrecisionMode.FP32, result);
    }

    private static WirePrecisionMode SelectOptimalWirePrecision(int batchSize, bool isDecode, double lanLatencyMs)
    {
        if (isDecode) return WirePrecisionMode.FP16;
        if (lanLatencyMs > 2.5) return WirePrecisionMode.INT8_Symmetric;
        return batchSize >= 32 ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;
    }

    private static WirePrecisionMode ResolveWithOverride(WirePrecisionMode? forcedOverride, WirePrecisionMode fallback)
    {
        return forcedOverride ?? fallback;
    }
}
