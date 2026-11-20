namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Inference.Cli;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Xunit;

public sealed class SleepMemoryServeTests : IDisposable
{
    private readonly string _tempDir;

    public SleepMemoryServeTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierSleepServe_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
        }
    }

    [Fact]
    public async Task ConsolidateMessagesAsync_64KHaystack_PacksWithinBudgetAndPreservesExactNeedle()
    {
        using var sleep = new SleepMemoryService(tokenBudget: 400);

        string haystack = Build64KMultiDomainHaystack(
            needle: "[SECURITY_DIRECTIVE] Emergency failover endpoint uses signing token OMEGA_HELIO_9482 and standby port 19443 in region north_edge_7.",
            targetChars: 65_000);
        Assert.True(haystack.Length >= 64_000, $"Expected >= 64K chars, got {haystack.Length}");

        var input = new List<(string Role, string Content)>
        {
            ("system", "Answer concisely using the provided evidence."),
            ("user", $"{haystack}\n\nQuestion: What emergency signing token and standby port are specified for north_edge_7?")
        };

        var consolidated = await sleep.ConsolidateMessagesAsync(input);

        Assert.Equal(2, consolidated.Count);
        Assert.Equal("system", consolidated[0].Role);
        Assert.Equal("user", consolidated[1].Role);

        string userPrompt = consolidated[1].Content;
        Assert.StartsWith("Consolidated Context:", userPrompt);
        Assert.Contains("OMEGA_HELIO_9482", userPrompt);
        Assert.Contains("19443", userPrompt);
        Assert.Contains("Question:", userPrompt);
        Assert.DoesNotContain("[cluster/", userPrompt);
        Assert.True(userPrompt.Length < 4_000, $"Expected compact consolidated prompt, got {userPrompt.Length} chars.");

        SleepStatusResponse status = sleep.GetStatus();
        Assert.True(status.Enabled);
        Assert.Equal(400, status.TokenBudget);
        Assert.True(status.SleepCycles >= 1, $"Expected >= 1 sleep cycle on large haystack ingest, got {status.SleepCycles}");
        Assert.True(status.LastPackedTokens is > 0 and <= 400, $"PackedTokens out of budget: {status.LastPackedTokens}");
        Assert.True(status.RawChunks >= 50);

        // Re-sending identical haystack must deduplicate raw chunks.
        int chunksBeforeRepeat = status.RawChunks;
        await sleep.ConsolidateMessagesAsync(input);
        Assert.Equal(chunksBeforeRepeat, sleep.GetStatus().RawChunks);
    }

    [Fact]
    public async Task ConsolidateMessagesAsync_ShortPrompt_PassesThroughUnmodified()
    {
        using var sleep = new SleepMemoryService(tokenBudget: 400);
        var input = new List<(string Role, string Content)>
        {
            ("system", "You are a helpful assistant."),
            ("user", "Write a compact lock-free ring buffer in C#.")
        };

        var output = await sleep.ConsolidateMessagesAsync(input);

        Assert.Equal(2, output.Count);
        Assert.Equal("Write a compact lock-free ring buffer in C#.", output[1].Content);
        Assert.Equal(0, sleep.GetStatus().SleepCycles);
    }

    [Fact]
    public async Task RecordTurnCompletion_StripsThinkTagsBeforeEpisodicStorage()
    {
        using var sleep = new SleepMemoryService(tokenBudget: 400);
        sleep.RecordTurnCompletion(
            "What port is primary?",
            "<think>noisy reasoning token SECRET_THOUGHT_999</think>\n\nThe primary port is 18443.");

        await sleep.TriggerSleepAsync();
        var status = sleep.GetStatus();
        Assert.True(status.RawChunks >= 2);
        Assert.True(status.SleepCycles >= 1);

        // Now ask in a second multi-turn request with extra filler history to verify retrieval excludes <think> noise.
        var msgs = new List<(string Role, string Content)>();
        for (int i = 0; i < 40; i++)
        {
            msgs.Add(("user", $"Routine telemetry heartbeat status check iteration {i} reported nominal latency."));
            msgs.Add(("assistant", $"Acknowledged nominal telemetry iteration {i} with baseline throughput counters."));
        }
        msgs.Add(("user", "Which primary port was stated earlier?"));

        var consolidated = await sleep.ConsolidateMessagesAsync(msgs);
        string userPrompt = consolidated[^1].Content;
        Assert.Contains("18443", userPrompt);
        Assert.DoesNotContain("SECRET_THOUGHT_999", userPrompt);
    }

    [Fact]
    public async Task HttpEndpoints_SleepStatusAndTrigger_WorkOverKestrel()
    {
        string ggufPath = Path.Combine(_tempDir, "sleep_endpoint_test.gguf");
        WriteMinimalQwen2Gguf(ggufPath);

        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu");
        await using var sleep = new SleepMemoryService(tokenBudget: 400);

        // Disabled status check on separate app instance without sleepMemory
        int disabledPort = GetAvailablePort();
        await using (var disabledApp = Program.CreateServeApp(session, ggufPath, port: disabledPort))
        {
            await disabledApp.StartAsync();
            try
            {
                using var disabledClient = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{disabledPort}") };
                var stResp = await disabledClient.GetFromJsonAsync<SleepStatusResponse>("/api/sleep/status");
                Assert.NotNull(stResp);
                Assert.False(stResp!.Enabled);
                var trigResp = await disabledClient.PostAsync("/api/sleep/trigger", content: null);
                Assert.Equal(HttpStatusCode.BadRequest, trigResp.StatusCode);
            }
            finally
            {
                await disabledApp.StopAsync();
            }
        }

        // Enabled sleep memory endpoints
        await using var app = Program.CreateServeApp(session, ggufPath, port: port, sleepMemory: sleep);
        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var initial = await client.GetFromJsonAsync<SleepStatusResponse>("/api/sleep/status");
            Assert.NotNull(initial);
            Assert.True(initial!.Enabled);
            Assert.Equal(400, initial.TokenBudget);
            Assert.Equal(0, initial.SleepCycles);

            sleep.RecordTurnCompletion("Remember calibration constant GAMMA_4412.", "Stored GAMMA_4412 cleanly.");

            var triggerHttp = await client.PostAsync("/api/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, triggerHttp.StatusCode);

            var triggered = await triggerHttp.Content.ReadFromJsonAsync<SleepStatusResponse>();
            Assert.NotNull(triggered);
            Assert.True(triggered!.Enabled);
            Assert.True(triggered.RawChunks >= 2);
            Assert.True(triggered.SleepCycles >= 1);

            var polled = await client.GetFromJsonAsync<SleepStatusResponse>("/api/sleep/status");
            Assert.NotNull(polled);
            Assert.Equal(triggered.SleepCycles, polled!.SleepCycles);
        }
        finally
        {
            await app.StopAsync();
        }
    }

    private static string Build64KMultiDomainHaystack(string needle, int targetChars)
    {
        var sb = new StringBuilder(targetChars + 1024);
        int i = 0;
        bool needleInserted = false;
        while (sb.Length < targetChars)
        {
            if (!needleInserted && sb.Length >= targetChars / 2)
            {
                sb.AppendLine();
                sb.AppendLine(needle);
                sb.AppendLine();
                needleInserted = true;
            }

            sb.Append("[TELEMETRY node_")
              .Append((i % 19) + 1)
              .Append("] Routine scheduled metric sample seq=")
              .Append(i)
              .Append(" cpu_pct=")
              .Append(12 + (i % 45))
              .Append(" mem_mb=")
              .Append(4096 + (i % 512))
              .Append(" queue_depth=")
              .Append(i % 7)
              .AppendLine(" status=NOMINAL heartbeat=OK.");
            if (i % 12 == 0)
            {
                sb.AppendLine();
            }
            i++;
        }

        if (!needleInserted)
        {
            sb.AppendLine(needle);
        }

        return sb.ToString();
    }

    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static void WriteMinimalQwen2Gguf(string path)
    {
        const int dim = 64;
        const int ffnDim = 128;
        const int headCount = 2;
        const int headCountKv = 1;
        const int headDim = dim / headCount;
        const int vocabSize = 288;

        string[] vocab = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            vocab[i] = i == vocabSize - 1 ? "<|endoftext|>" : i == vocabSize - 2 ? "<|im_start|>" : $"tok_{i}";
        }

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
            ("tokenizer.ggml.tokens", GgufValueType.Array, vocab)
        };

        float[] ones = new float[dim];
        Array.Fill(ones, 1.0f);
        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, new float[dim * vocabSize]),
            ("output_norm.weight", new[] { (ulong)dim }, ones),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, new float[dim * vocabSize]),
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ones),
            ("blk.0.attn_q.weight", new[] { (ulong)dim, (ulong)(headCount * headDim) }, new float[dim * headCount * headDim]),
            ("blk.0.attn_k.weight", new[] { (ulong)dim, (ulong)(headCountKv * headDim) }, new float[dim * headCountKv * headDim]),
            ("blk.0.attn_v.weight", new[] { (ulong)dim, (ulong)(headCountKv * headDim) }, new float[dim * headCountKv * headDim]),
            ("blk.0.attn_output.weight", new[] { (ulong)(headCount * headDim), (ulong)dim }, new float[headCount * headDim * dim]),
            ("blk.0.ffn_norm.weight", new[] { (ulong)dim }, ones),
            ("blk.0.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, new float[dim * ffnDim]),
            ("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, new float[dim * ffnDim]),
            ("blk.0.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, new float[ffnDim * dim])
        };

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);
        bw.Write(0x46554747U);
        bw.Write(3U);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, val) in metadata)
        {
            WriteStr(bw, key);
            bw.Write((uint)type);
            if (type == GgufValueType.String) WriteStr(bw, (string)val);
            else if (type == GgufValueType.Uint32) bw.Write((uint)val);
            else if (type == GgufValueType.Array)
            {
                var arr = (string[])val;
                bw.Write((uint)GgufValueType.String);
                bw.Write((ulong)arr.Length);
                foreach (var s in arr) WriteStr(bw, s);
            }
        }

        ulong offset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            WriteStr(bw, name);
            bw.Write((uint)dims.Length);
            foreach (var d in dims) bw.Write(d);
            bw.Write((uint)GgufType.F32);
            bw.Write(offset);
            ulong len = (ulong)data.Length * sizeof(float);
            offset = (offset + len + 31UL) & ~31UL;
        }

        long alignedStart = ((fs.Position + 31L) & ~31L);
        while (fs.Position < alignedStart) bw.Write((byte)0);
        foreach (var (_, _, data) in tensors)
        {
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
            long nextAligned = ((fs.Position - alignedStart + 31L) & ~31L) + alignedStart;
            while (fs.Position < nextAligned) bw.Write((byte)0);
        }
    }

    private static void WriteStr(BinaryWriter bw, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)b.Length);
        bw.Write(b);
    }
}
