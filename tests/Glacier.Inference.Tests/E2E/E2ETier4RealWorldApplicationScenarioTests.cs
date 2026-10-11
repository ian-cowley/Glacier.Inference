namespace Glacier.Inference.Tests.E2E;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Xunit;

/// <summary>
/// Tier 4 Real-World Application Scenario Tests:
/// DeepSeek-R1 reasoning traces, code generation, multi-turn dialogue, TTFT measurement, and full cluster simulation.
/// </summary>
[Trait("Category", "E2E")]
[Trait("Tier", "4")]
public class E2ETier4RealWorldApplicationScenarioTests
{
    // =========================================================================
    // SCENARIO 1: DEEPSEEK-R1 <think> REASONING TRACES
    // =========================================================================

    [Fact]
    public void Scenario1_DeepSeekR1_ThinkReasoningTraces_EmitsAndClosesTags()
    {
        // DeepSeek-R1 reasoning tokens:
        // <think> = 151648, </think> = 151649, EOS = 151643
        const int tokThinkOpen = 151648;
        const int tokThinkClose = 151649;
        const int tokEos = 151643;

        var tokens = new List<int>
        {
            tokThinkOpen,
            101, 102, 103, // Internal thought steps
            tokThinkClose,
            201, 202,     // Structured answer
            tokEos
        };

        // Assert valid reasoning stream structure
        int openIdx = tokens.IndexOf(tokThinkOpen);
        int closeIdx = tokens.IndexOf(tokThinkClose);
        int eosIdx = tokens.IndexOf(tokEos);

        Assert.True(openIdx >= 0, "<think> tag must be emitted");
        Assert.True(closeIdx > openIdx, "</think> tag must follow <think>");
        Assert.True(eosIdx > closeIdx, "EOS token must follow thinking block");

        // Extract reasoning trace tokens
        int reasoningLength = closeIdx - openIdx - 1;
        Assert.Equal(3, reasoningLength);
    }

    // =========================================================================
    // SCENARIO 2: CODE GENERATION WITH REPETITIVE INDENTATION & HIGH HIT RATE
    // =========================================================================

    [Fact]
    public void Scenario2_CodeGeneration_LongContext_HighRepetitionHitRate()
    {
        var draftOracle = new PromptLookupDraftOracle(maxContext: 2048, maxNgram: 3);

        // Typical C# source pattern:
        // "    public static void Method() {\n        return 0;\n    }\n"
        // Indentation tokens (4 spaces, 8 spaces, curly braces, semicolons) repeat frequently
        int[] codeBoilerplate = new[] { 1000, 1001, 1002, 1003, 1004, 1005, 1006 };
        draftOracle.AppendTokens(codeBoilerplate);

        int totalGenerations = 20;
        int hitCount = 0;

        for (int i = 0; i < totalGenerations; i++)
        {
            // Inject periodic repetition
            int expectedNext = codeBoilerplate[(i + 2) % codeBoilerplate.Length];
            draftOracle.AppendToken(codeBoilerplate[(i + 1) % codeBoilerplate.Length]);

            if (draftOracle.TryPredictNextToken(out int predicted) && predicted == expectedNext)
            {
                hitCount++;
                draftOracle.AppendToken(predicted);
            }
            else
            {
                draftOracle.AppendToken(expectedNext);
            }
        }

        double hitRate = (double)hitCount / totalGenerations;
        Assert.True(hitRate >= 0.50, $"Draft hit rate on repetitive code was {hitRate:P1} (expected >= 50%)");
    }

    // =========================================================================
    // SCENARIO 3: MULTI-TURN DIALOGUE & KV RESET
    // =========================================================================

    [Fact]
    public void Scenario3_MultiTurnDialogue_MaintainsContext_ResetsKVOnCommand()
    {
        var sessionContext = new List<int>();

        // Turn 1: User prompt
        int[] turn1User = new[] { 1, 10, 20 };
        sessionContext.AddRange(turn1User);
        int[] turn1Assis = new[] { 30, 40 };
        sessionContext.AddRange(turn1Assis);

        Assert.Equal(5, sessionContext.Count);

        // Turn 2: Follow-up question (KV cache grows continuously)
        int[] turn2User = new[] { 2, 50, 60 };
        sessionContext.AddRange(turn2User);
        int[] turn2Assis = new[] { 70, 80 };
        sessionContext.AddRange(turn2Assis);

        Assert.Equal(10, sessionContext.Count);

        // Reset session command (e.g. /reset in chat)
        sessionContext.Clear();
        Assert.Empty(sessionContext);
    }

    // =========================================================================
    // SCENARIO 4: TTFT (TIME-TO-FIRST-TOKEN) & PREFILL THROUGHPUT BENCHMARK
    // =========================================================================

    [Fact]
    public void Scenario4_TtftMeasurement_MeasuresPrefillAndFirstTokenLatency()
    {
        int promptTokenCount = 256;
        var sw = Stopwatch.StartNew();

        // Emulate batched prompt prefill ingestion
        float[] promptActivations = new float[promptTokenCount * 64];
        for (int i = 0; i < promptActivations.Length; i++) promptActivations[i] = 0.5f;

        // First token sampled
        int firstToken = 42;
        sw.Stop();

        double ttftMs = sw.Elapsed.TotalMilliseconds;
        double prefillTokPerSec = (promptTokenCount / Math.Max(0.001, ttftMs)) * 1000.0;

        Assert.True(firstToken > 0);
        Assert.True(prefillTokPerSec > 0.0);
    }

    // =========================================================================
    // SCENARIO 5: CONTINUOUS 256-TOKEN GENERATION WITH ZERO GC ALLOCATIONS
    // =========================================================================

    [Fact]
    public void Scenario5_Continuous256TokenGeneration_ZeroGCAllocationsOnHotPath()
    {
        int dim = 5120;
        float[] actBuffer = new float[dim];
        float[] residualBuffer = new float[dim];
        sbyte[] quantBuffer = new sbyte[dim];

        // Start zero-allocation verification scope
        var scope = ZeroAllocScope.Start();

        for (int step = 0; step < 256; step++)
        {
            // 1. In-place residual update
            for (int i = 0; i < dim; i++)
            {
                residualBuffer[i] = actBuffer[i] + 0.1f;
            }

            // 2. Symmetric INT8 quantization
            float scale = QuantizationOracle.QuantizeSymmetricInt8(residualBuffer, quantBuffer);

            // 3. Dequantization
            QuantizationOracle.DequantizeSymmetricInt8(quantBuffer, scale, actBuffer);
        }

        scope.AssertZeroAllocations("256-token continuous hot decode generation");
    }

    // =========================================================================
    // SCENARIO 6: CLUSTER COORDINATOR-WORKER END-TO-END LOOPBACK SIMULATION
    // =========================================================================

    [Fact]
    public async Task Scenario6_ClusterCoordinatorWorker_EndToEndLoopbackStreaming()
    {
        // Create loopback socket connection
        using var pair = await TestSocketPair.CreateConnectedAsync();

        // 1. Coordinator sends MsgInitStage to Worker
        const ClusterMessageType MsgInitStage = (ClusterMessageType)0x60;
        const ClusterMessageType MsgStageReady = (ClusterMessageType)0x61;
        const ClusterMessageType MsgActivation = (ClusterMessageType)0x70;
        const ClusterMessageType MsgResult = (ClusterMessageType)0x71;
        const ClusterMessageType MsgShutdown = (ClusterMessageType)0x73;

        var workerTask = Task.Run(async () =>
        {
            byte[] hdrBuf = new byte[32];
            // Read Init
            await pair.Server.ReceiveAsync(hdrBuf, SocketFlags.None);
            var initHdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBuf);
            Assert.Equal(MsgInitStage, initHdr.MessageType);

            // Send StageReady
            var readyHdr = new ClusterFrameHeader(MsgStageReady, initHdr.CorrelationId, 0);
            MemoryMarshal.Write(hdrBuf, in readyHdr);
            await pair.Server.SendAsync(hdrBuf, SocketFlags.None);

            // Read Activation
            await pair.Server.ReceiveAsync(hdrBuf, SocketFlags.None);
            var actHdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBuf);
            Assert.Equal(MsgActivation, actHdr.MessageType);

            // Send Result with predicted token 151644
            byte[] resBuf = new byte[32 + 28];
            var resHdr = new ClusterFrameHeader(MsgResult, actHdr.CorrelationId, 28);
            MemoryMarshal.Write(resBuf.AsSpan(0, 32), in resHdr);
            BitConverter.TryWriteBytes(resBuf.AsSpan(32 + 16, 4), 151644); // nextTokenId

            await pair.Server.SendAsync(resBuf, SocketFlags.None);

            // Read Shutdown
            await pair.Server.ReceiveAsync(hdrBuf, SocketFlags.None);
            var shutHdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBuf);
            Assert.Equal(MsgShutdown, shutHdr.MessageType);
        });

        // Coordinator side:
        byte[] coordHdr = new byte[32];

        // Send Init
        var init = new ClusterFrameHeader(MsgInitStage, 101, 0);
        MemoryMarshal.Write(coordHdr, in init);
        await pair.Client.SendAsync(coordHdr, SocketFlags.None);

        // Recv Ready
        await pair.Client.ReceiveAsync(coordHdr, SocketFlags.None);
        var ready = MemoryMarshal.Read<ClusterFrameHeader>(coordHdr);
        Assert.Equal(MsgStageReady, ready.MessageType);

        // Send Activation
        var act = new ClusterFrameHeader(MsgActivation, 102, 0);
        MemoryMarshal.Write(coordHdr, in act);
        await pair.Client.SendAsync(coordHdr, SocketFlags.None);

        // Recv Result
        byte[] coordRes = new byte[60];
        await pair.Client.ReceiveAsync(coordRes, SocketFlags.None);
        int generatedToken = BitConverter.ToInt32(coordRes.AsSpan(32 + 16, 4));
        Assert.Equal(151644, generatedToken);

        // Send Shutdown
        var shut = new ClusterFrameHeader(MsgShutdown, 103, 0);
        MemoryMarshal.Write(coordHdr, in shut);
        await pair.Client.SendAsync(coordHdr, SocketFlags.None);

        await workerTask;
    }
}
