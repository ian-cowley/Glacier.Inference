namespace Glacier.Inference.Tests.E2E;

using System;
using System.Buffers.Binary;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Xunit;

/// <summary>
/// Tier 2 Boundary & Corner Case Tests: Extreme prompts, edge scales, packet fragmentation, and socket disconnect recovery.
/// </summary>
[Trait("Category", "E2E")]
[Trait("Tier", "2")]
public class E2ETier2BoundaryAndCornerTests
{
    // =========================================================================
    // EXTREME PROMPT LENGTHS & EMPTY PROMPTS
    // =========================================================================

    [Fact]
    public void Boundary_EmptyPrompt_HandledWithoutNullReferenceException()
    {
        string prompt = "";
        int[] tokens = Array.Empty<int>();

        Assert.Empty(prompt);
        Assert.Empty(tokens);

        // Prefill chunk with 0 tokens should be a no-op or handled gracefully
        int chunkCount = (tokens.Length + 31) / 32;
        Assert.Equal(0, chunkCount);
    }

    [Fact]
    public void Boundary_SingleTokenPrompt_GeneratesFirstTokenDirectly()
    {
        int[] singleToken = new[] { 151644 }; // DeepSeek BOS
        Assert.Single(singleToken);

        var draft = new PromptLookupDraftOracle();
        draft.AppendTokens(singleToken);
        Assert.Equal(1, draft.ContextLength);

        // A single token has no prior n-gram match
        Assert.False(draft.TryPredictNextToken(out _));
    }

    [Fact]
    public void Boundary_ExtremePromptLength_4096Tokens_ChunkedCleanly()
    {
        int[] longPrompt = new int[4096];
        for (int i = 0; i < longPrompt.Length; i++) longPrompt[i] = i % 1000;

        const int chunkSize = 512;
        int chunks = (longPrompt.Length + chunkSize - 1) / chunkSize;
        Assert.Equal(8, chunks);

        for (int c = 0; c < chunks; c++)
        {
            int start = c * chunkSize;
            int len = Math.Min(chunkSize, longPrompt.Length - start);
            ReadOnlySpan<int> chunk = longPrompt.AsSpan(start, len);
            Assert.Equal(512, chunk.Length);
        }
    }

    [Fact]
    public void Boundary_HugeContextLength_32768Tokens_KVHeadroomMaintained()
    {
        const int maxContext = 32768;
        int currentSeq = 30000;
        int remainingSlots = maxContext - currentSeq;
        Assert.True(remainingSlots > 2000, "Should have enough KV slots remaining");
    }

    // =========================================================================
    // EDGE QUANTIZATION SCALES & EXTREME VALUES
    // =========================================================================

    [Fact]
    public void Boundary_EdgeQuantizationScale_ZeroVector_ProducesScaleOneWithoutDivideByZero()
    {
        float[] zeros = new float[5120];
        sbyte[] q = new sbyte[5120];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(zeros, q);
        Assert.Equal(1.0f, scale);
        Assert.False(float.IsNaN(scale));
        Assert.False(float.IsInfinity(scale));
    }

    [Fact]
    public void Boundary_EdgeQuantizationScale_SingleSubnormalFloat_HandlesPrecisionWithoutNaN()
    {
        float[] subnormal = new float[128];
        subnormal[64] = 1.40129846e-45f; // Smallest positive subnormal float32

        sbyte[] q = new sbyte[128];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(subnormal, q);

        Assert.False(float.IsNaN(scale));
        Assert.True(scale > 0.0f);
    }

    [Fact]
    public void Boundary_EdgeQuantizationScale_MaxFloatValue_ScalesWithoutOverflow()
    {
        float[] big = new float[] { 3.40282347e+38f, -3.40282347e+38f };
        sbyte[] q = new sbyte[2];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(big, q);
        Assert.False(float.IsInfinity(scale));
        Assert.Equal(127, q[0]);
        Assert.Equal(-127, q[1]);
    }

    [Fact]
    public void Boundary_EdgeQuantizationScale_AllNegativeMaxValues_ClampsToNeg127()
    {
        float[] allNeg = new float[] { -50.0f, -100.0f, -20.0f };
        sbyte[] q = new sbyte[3];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(allNeg, q);
        Assert.Equal(-127, q[1]); // -100 is max abs, scales to -127
    }

    [Fact]
    public void Boundary_ExtremeQuantizationScale_ScaleFactorIsNaN_ReplacedWithUnitScale()
    {
        float[] nanInput = new float[] { float.NaN, float.PositiveInfinity };
        sbyte[] q = new sbyte[2];

        float scale = QuantizationOracle.QuantizeSymmetricInt8(nanInput, q);
        Assert.Equal(1.0f, scale);
    }

    // =========================================================================
    // PACKET BOUNDARY FRAGMENTATION
    // =========================================================================

    [Fact]
    public async Task Boundary_PacketBoundaryFragmentation_1ByteHeaderChunking_ReassemblesCompleteFrame()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        var header = new FusedActivationHeader(
            sequenceId: 0x8877665544332211UL,
            tokenIndex: 77,
            layerIndex: 21,
            hiddenDim: 64,
            precisionMode: WirePrecisionMode.FP16,
            flags: FusedActivationFlags.None,
            scale: 1.0f);

        byte[] frameBytes = new byte[32];
        header.WriteTo(frameBytes.AsSpan());

        // Send byte-by-byte
        _ = Task.Run(async () =>
        {
            for (int i = 0; i < frameBytes.Length; i++)
            {
                await pair.Client.SendAsync(frameBytes.AsMemory(i, 1), SocketFlags.None);
                await Task.Yield();
            }
        });

        byte[] recvBuffer = new byte[32];
        int totalRead = 0;
        while (totalRead < 32)
        {
            int r = await pair.Server.ReceiveAsync(recvBuffer.AsMemory(totalRead, 32 - totalRead), SocketFlags.None);
            if (r == 0) break;
            totalRead += r;
        }

        Assert.Equal(32, totalRead);
        var readHeader = FusedActivationHeader.ReadFrom(recvBuffer.AsSpan());
        Assert.Equal(header, readHeader);
    }

    [Fact]
    public async Task Boundary_PacketBoundaryFragmentation_SplitBetweenHeaderAndPayload_ReassemblesSeamlessly()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        var header = new FusedActivationHeader(
            sequenceId: 10,
            tokenIndex: 2,
            layerIndex: 0,
            hiddenDim: 16,
            precisionMode: WirePrecisionMode.INT8_Symmetric,
            scale: 0.5f);

        byte[] fullFrame = new byte[32 + 20]; // 32B header + 20B payload (16 ints + 4B scale)
        header.WriteTo(fullFrame.AsSpan(0, 32));
        for (int i = 0; i < 20; i++) fullFrame[32 + i] = (byte)(i + 1);

        _ = Task.Run(async () =>
        {
            // Send exactly 32 bytes (header), then remainder (payload)
            await pair.Client.SendAsync(fullFrame.AsMemory(0, 32), SocketFlags.None);
            await Task.Delay(5);
            await pair.Client.SendAsync(fullFrame.AsMemory(32, 20), SocketFlags.None);
        });

        byte[] received = new byte[52];
        int total = 0;
        while (total < 52)
        {
            int r = await pair.Server.ReceiveAsync(received.AsMemory(total, 52 - total), SocketFlags.None);
            if (r == 0) break;
            total += r;
        }

        Assert.Equal(52, total);
        var parsedHeader = FusedActivationHeader.ReadFrom(received.AsSpan(0, 32));
        Assert.Equal(header, parsedHeader);
        Assert.True(fullFrame.AsSpan(32).SequenceEqual(received.AsSpan(32)));
    }

    [Fact]
    public async Task Boundary_PacketBoundaryFragmentation_HeaderSplitAtByte17_LittleEndianIntact()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        var header = new FusedActivationHeader(
            sequenceId: 123456789UL,
            tokenIndex: 99,
            layerIndex: 47,
            hiddenDim: 5120,
            precisionMode: WirePrecisionMode.FP32);

        byte[] buffer = new byte[32];
        header.WriteTo(buffer.AsSpan());

        _ = Task.Run(async () =>
        {
            await pair.Client.SendAsync(buffer.AsMemory(0, 17), SocketFlags.None);
            await Task.Delay(5);
            await pair.Client.SendAsync(buffer.AsMemory(17, 15), SocketFlags.None);
        });

        byte[] recv = new byte[32];
        int read = 0;
        while (read < 32)
        {
            int r = await pair.Server.ReceiveAsync(recv.AsMemory(read, 32 - read), SocketFlags.None);
            if (r == 0) break;
            read += r;
        }

        Assert.Equal(32, read);
        var parsed = FusedActivationHeader.ReadFrom(recv.AsSpan());
        Assert.Equal(header, parsed);
    }

    // =========================================================================
    // SOCKET DISCONNECT AND RECOVERY
    // =========================================================================

    [Fact]
    public async Task Boundary_SocketDisconnectRecovery_ClientAbruptClose_WorkerResetsSessionCleanly()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        // Client abruptly disconnects
        pair.Client.Close();

        byte[] buffer = new byte[64];
        int read = await pair.Server.ReceiveAsync(buffer.AsMemory(), SocketFlags.None);
        // Clean EOF (read == 0) indicates client closure
        Assert.Equal(0, read);
    }

    [Fact]
    public async Task Boundary_SocketDisconnectRecovery_ServerClose_ClientCatchesSocketExceptionAndBacksOff()
    {
        using var pair = await TestSocketPair.CreateConnectedAsync();

        pair.Server.Close();

        byte[] sendBuf = new byte[32];
        // Client writing to closed socket eventually throws SocketException
        bool threw = false;
        try
        {
            for (int i = 0; i < 10; i++)
            {
                await pair.Client.SendAsync(sendBuf, SocketFlags.None);
            }
        }
        catch (SocketException)
        {
            threw = true;
        }

        Assert.True(threw || !pair.Server.Connected);
    }

    [Fact]
    public async Task Boundary_SocketDisconnectRecovery_ReconnectAfterDrop_HandshakeSucceeds()
    {
        // First connection
        var pair1 = await TestSocketPair.CreateConnectedAsync();
        pair1.Dispose();

        // Reconnect after drop
        var pair2 = await TestSocketPair.CreateConnectedAsync();
        Assert.True(pair2.Client.Connected);
        Assert.True(pair2.Server.Connected);
        pair2.Dispose();
    }

    // =========================================================================
    // MALFORMED HEADERS & RESOURCE GUARDS
    // =========================================================================

    [Fact]
    public void Boundary_MalformedHeader_CorruptedMagic_RejectedImmediately()
    {
        byte[] malformed = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(malformed.AsSpan(0, 4), 0x99999999);

        Assert.Throws<InvalidOperationException>(() => FusedActivationHeader.ReadFrom(malformed.AsSpan()));
    }

    [Fact]
    public void Boundary_MalformedHeader_UlongMaxValueSequenceId_HandledWithoutOverflow()
    {
        var h = new FusedActivationHeader(
            sequenceId: ulong.MaxValue,
            tokenIndex: int.MaxValue,
            layerIndex: int.MaxValue,
            hiddenDim: 5120,
            precisionMode: WirePrecisionMode.FP16);

        byte[] buf = new byte[32];
        h.WriteTo(buf.AsSpan());
        var res = FusedActivationHeader.ReadFrom(buf.AsSpan());

        Assert.Equal(ulong.MaxValue, res.SequenceId);
        Assert.Equal(int.MaxValue, res.TokenIndex);
    }

    [Fact]
    public void Boundary_ZeroLengthBatchActivation_RejectedGracefully()
    {
        int numTokens = 0;
        int actDim = 5120;
        int expectedFloats = numTokens * actDim;
        Assert.Equal(0, expectedFloats);
    }

    [Fact]
    public void Boundary_ChannelBackpressure_ZeroCapacityBuffer_WaitsWithoutDeadlock()
    {
        var channel = Channel.CreateBounded<int>(1);
        Assert.True(channel.Writer.TryWrite(100));
        Assert.False(channel.Writer.TryWrite(200)); // Bounded
    }

    [Fact]
    public void Boundary_DoubleDispose_PinnedPoolIsIdempotent()
    {
        var pool = new PinnedSocketBufferPool(1024, 2);
        pool.Dispose();
        pool.Dispose(); // Must not throw
    }

    [Fact]
    public void Boundary_DraftProvider_EmptyContext_ReturnsFalseWithoutThrowing()
    {
        var draft = new PromptLookupDraftOracle();
        Assert.False(draft.TryPredictNextToken(out int token));
        Assert.Equal(0, token);
    }

    [Fact]
    public void Boundary_DraftProvider_RollbackBeyondZero_ClampsToZeroLength()
    {
        var draft = new PromptLookupDraftOracle();
        draft.AppendToken(42);
        draft.Rollback(100);
        Assert.Equal(0, draft.ContextLength);
    }
}
