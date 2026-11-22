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
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Glacier.Inference.Cli;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Serve.Daemons;
using Glacier.Serve.Serialization;
using Xunit;

[Collection("ConsoleTests")]
public sealed class CliOptionsAndServingTests : IDisposable
{
    private readonly string _tempDir;

    public CliOptionsAndServingTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierInfCliM5b_" + Guid.NewGuid().ToString("N"));
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

    [Theory]
    [InlineData("auto", KvCachePrecision.Auto)]
    [InlineData("fp32", KvCachePrecision.Fp32)]
    [InlineData("fp16", KvCachePrecision.Fp16)]
    [InlineData("fp8", KvCachePrecision.Fp8)]
    [InlineData("q8_0", KvCachePrecision.Q8_0)]
    [InlineData("q4_0", KvCachePrecision.Q4_0)]
    public void TryParseKvPrecisionArg_AcceptsAllSupportedModes(string raw, KvCachePrecision expected)
    {
        bool ok = Program.TryParseKvPrecisionArg(raw, out var precision, out var error);
        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal(expected, precision);
    }

    [Fact]
    public void TryParseKvPrecisionArg_RejectsUnknownValue_AndListsQ8_0()
    {
        bool ok = Program.TryParseKvPrecisionArg("int2_invalid", out _, out var error);
        Assert.False(ok);
        Assert.NotNull(error);
        Assert.Contains("Q8_0", error);
        Assert.Contains("Q4_0", error);
    }

    [Fact]
    public void TryParseLoraSpec_SupportsNamedFileUnnamedFileDirectoryAndRejectsInvalid()
    {
        string fileLora = Path.Combine(_tempDir, "coder_adapter.bin");
        WriteMinimalLoraBinary(fileLora);

        string dirLora = Path.Combine(_tempDir, "roslyn_dir");
        Directory.CreateDirectory(dirLora);
        WriteMinimalLoraBinary(Path.Combine(dirLora, "adapter_model.bin"));

        Assert.True(Program.TryParseLoraSpec($"custom={fileLora}", out var n1, out var p1, out var e1));
        Assert.Null(e1);
        Assert.Equal("custom", n1);
        Assert.Equal(fileLora, p1);

        Assert.True(Program.TryParseLoraSpec(fileLora, out var n2, out var p2, out var e2));
        Assert.Null(e2);
        Assert.Equal("coder_adapter", n2);
        Assert.Equal(fileLora, p2);

        Assert.True(Program.TryParseLoraSpec($"roslyn={dirLora}", out var n3, out var p3, out var e3));
        Assert.Null(e3);
        Assert.Equal("roslyn", n3);
        Assert.Equal(Path.Combine(dirLora, "adapter_model.bin"), p3);

        Assert.False(Program.TryParseLoraSpec($"={fileLora}", out _, out _, out var eEmptyName));
        Assert.Contains("empty", eEmptyName!, StringComparison.OrdinalIgnoreCase);

        Assert.False(Program.TryParseLoraSpec($"bad:name={fileLora}", out _, out _, out var eColon));
        Assert.Contains(":", eColon!);

        Assert.False(Program.TryParseLoraSpec(Path.Combine(_tempDir, "missing.bin"), out _, out _, out var eMissing));
        Assert.Contains("not found", eMissing!, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("bounce", "bounce")]
    [InlineData("serve", "--lora")]
    [InlineData("serve", "q8_0")]
    [InlineData("serve", "/v1/sleep/status")]
    [InlineData("run", "--lora")]
    [InlineData("eval-kl", "q8_0")]
    public async Task Program_HelpCommands_ContainM5bFlagsAndRoutes(string command, string expectedSubstring)
    {
        var outWriter = new StringWriter();
        var origOut = Console.Out;
        Console.SetOut(outWriter);
        try
        {
            int code = await Program.Main([command, "--help"]);
            Assert.Equal(0, code);
            Assert.Contains(expectedSubstring, outWriter.ToString(), StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Console.SetOut(origOut);
        }
    }

    [Fact]
    public async Task Program_InvalidFlags_ReturnExitCodeOne()
    {
        string dummyGguf = Path.Combine(_tempDir, "dummy.gguf");
        string loraA = Path.Combine(_tempDir, "a.bin");
        string loraB = Path.Combine(_tempDir, "b.bin");
        WriteMinimalLoraBinary(loraA);
        WriteMinimalLoraBinary(loraB);

        var errWriter = new StringWriter();
        var outWriter = new StringWriter();
        var origErr = Console.Error;
        var origOut = Console.Out;
        Console.SetError(errWriter);
        Console.SetOut(outWriter);
        try
        {
            int c1 = await Program.Main(["serve", dummyGguf, "--kv-precision", "bogus"]);
            Assert.Equal(1, c1);
            Assert.Contains("Q8_0", errWriter.ToString());

            errWriter.GetStringBuilder().Clear();
            int c2 = await Program.Main(["serve", dummyGguf, "--lora", Path.Combine(_tempDir, "missing.lora")]);
            Assert.Equal(1, c2);
            Assert.Contains("not found", errWriter.ToString(), StringComparison.OrdinalIgnoreCase);

            int c3 = await Program.Main(["serve", dummyGguf, "--lora"]);
            Assert.Equal(1, c3);

            int c4 = await Program.Main(["bench", dummyGguf, "--kv-precision", "bogus"]);
            Assert.Equal(1, c4);

            int c5 = await Program.Main(["eval-kl", dummyGguf, "--ref", dummyGguf, "--kv-precision", "bogus"]);
            Assert.Equal(1, c5);

            int c6 = await Program.Main(["run", dummyGguf, "--lora", $"one={loraA}", "--lora", $"two={loraB}"]);
            Assert.Equal(1, c6);

            int c7 = await Program.Main(["bounce", "--port", "99999"]);
            Assert.Equal(1, c7);

            errWriter.GetStringBuilder().Clear();
            int c8 = await Program.Main(["serve", dummyGguf, "--unknown-bogus-flag"]);
            Assert.Equal(1, c8);
            Assert.Contains("Unknown option", errWriter.ToString(), StringComparison.OrdinalIgnoreCase);

            errWriter.GetStringBuilder().Clear();
            int c9 = await Program.Main(["serve", dummyGguf, "--max-think-tokens=-2"]);
            Assert.Equal(1, c9);
            Assert.Contains("--max-think-tokens", errWriter.ToString(), StringComparison.OrdinalIgnoreCase);

            // --max-think-tokens=2 parses cleanly and proceeds to model existence check
            errWriter.GetStringBuilder().Clear();
            int c10 = await Program.Main(["serve", dummyGguf, "--max-think-tokens=2"]);
            Assert.Equal(1, c10);
            Assert.Contains("Model file not found", errWriter.ToString(), StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Console.SetError(origErr);
            Console.SetOut(origOut);
        }
    }

    [Fact]
    public async Task PathBServe_IsolatesBaseVsLoraRequests_RejectsNegativeThinkBudget_AndHonorsEnableThinking()
    {
        string ggufPath = Path.Combine(_tempDir, "qwen2_pathb_test.gguf");
        WriteDistinctBaseVsLoraQwen2Gguf(ggufPath);

        string loraPath = Path.Combine(_tempDir, "roslyn.bin");
        WriteStrongLoraBinary(loraPath);

        string expectedBearer = "pathb-unit-" + Guid.NewGuid().ToString("N");
        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu);
        session.AttachLora(loraPath);

        await using var app = Program.CreateServeApp(
            session,
            ggufPath,
            apiKey: expectedBearer,
            host: "127.0.0.1",
            port: port,
            attachedLoraName: "roslyn");

        Assert.Null(session.CpuModel!.LoraWeights);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
            client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", expectedBearer);

            // 1. Base request (no LoRA)
            var baseResp1 = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                max_tokens = 2,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, baseResp1.StatusCode);
            var baseJson1 = await baseResp1.Content.ReadFromJsonAsync<JsonElement>();
            string baseText1 = baseJson1.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.Null(session.CpuModel.LoraWeights);

            // 2. LoRA request via model:<adapter>
            var loraResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test:roslyn",
                max_tokens = 2,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, loraResp.StatusCode);
            var loraJson = await loraResp.Content.ReadFromJsonAsync<JsonElement>();
            string loraText = loraJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.NotEqual(baseText1, loraText);
            Assert.Null(session.CpuModel.LoraWeights);

            // 3. Subsequent base request must NOT be contaminated by prior LoRA request
            var baseResp2 = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                max_tokens = 2,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, baseResp2.StatusCode);
            var baseJson2 = await baseResp2.Content.ReadFromJsonAsync<JsonElement>();
            string baseText2 = baseJson2.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.Equal(baseText1, baseText2);

            // 4. LoRA request via explicit lora field
            var loraFieldResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                lora = "roslyn",
                max_tokens = 2,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, loraFieldResp.StatusCode);
            var loraFieldJson = await loraFieldResp.Content.ReadFromJsonAsync<JsonElement>();
            string loraFieldText = loraFieldJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.Equal(loraText, loraFieldText);

            // 5. max_think_tokens: -2 -> HTTP 400 Bad Request
            var badThinkResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                max_think_tokens = -2,
                max_tokens = 2,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.BadRequest, badThinkResp.StatusCode);
            var badThinkJson = await badThinkResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("invalid_request_error", badThinkJson.GetProperty("error").GetProperty("type").GetString());

            // 6. chat_template_kwargs.enable_thinking changes prompt token count (<think> prefix)
            var thinkFalseResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                max_tokens = 1,
                chat_template_kwargs = new { enable_thinking = false },
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, thinkFalseResp.StatusCode);
            int promptTokensNoThink = (await thinkFalseResp.Content.ReadFromJsonAsync<JsonElement>())
                .GetProperty("usage").GetProperty("prompt_tokens").GetInt32();

            var thinkTrueResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_test",
                max_tokens = 1,
                chat_template_kwargs = new { enable_thinking = true },
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, thinkTrueResp.StatusCode);
            int promptTokensWithThink = (await thinkTrueResp.Content.ReadFromJsonAsync<JsonElement>())
                .GetProperty("usage").GetProperty("prompt_tokens").GetInt32();

            Assert.True(promptTokensWithThink > promptTokensNoThink,
                $"Expected enable_thinking=true prompt tokens ({promptTokensWithThink}) > enable_thinking=false ({promptTokensNoThink}).");
        }
        finally
        {
            await app.StopAsync();
        }
    }

    private sealed class FaultAfterFirstTokenChunkStream : Stream
    {
        private readonly Stream _inner;
        private readonly Func<int, bool> _shouldFaultOnWrite;
        private readonly Action _onFaultPointReached;
        private readonly Task _allowFaultTask;
        private int _writeCount;

        public FaultAfterFirstTokenChunkStream(
            Stream inner,
            Func<int, bool> shouldFaultOnWrite,
            Action onFaultPointReached,
            Task allowFaultTask)
        {
            _inner = inner;
            _shouldFaultOnWrite = shouldFaultOnWrite;
            _onFaultPointReached = onFaultPointReached;
            _allowFaultTask = allowFaultTask;
        }

        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => _inner.CanSeek;
        public override bool CanWrite => _inner.CanWrite;
        public override long Length => _inner.Length;
        public override long Position
        {
            get => _inner.Position;
            set => _inner.Position = value;
        }
        public override void Flush() => _inner.Flush();
        public override Task FlushAsync(CancellationToken cancellationToken) => _inner.FlushAsync(cancellationToken);
        public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => _inner.Seek(offset, origin);
        public override void SetLength(long value) => _inner.SetLength(value);

        public override void Write(byte[] buffer, int offset, int count)
        {
            int writeIndex = Interlocked.Increment(ref _writeCount);
            if (_shouldFaultOnWrite(writeIndex))
            {
                _onFaultPointReached();
                _allowFaultTask.Wait(TimeSpan.FromSeconds(10));
                throw new IOException("Simulated broken pipe during streaming.");
            }
            _inner.Write(buffer, offset, count);
        }

        public override async Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
        {
            int writeIndex = Interlocked.Increment(ref _writeCount);
            if (_shouldFaultOnWrite(writeIndex))
            {
                _onFaultPointReached();
                await _allowFaultTask.WaitAsync(TimeSpan.FromSeconds(10), cancellationToken);
                throw new IOException("Simulated broken pipe during streaming.");
            }
            await _inner.WriteAsync(buffer.AsMemory(offset, count), cancellationToken);
        }

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            int writeIndex = Interlocked.Increment(ref _writeCount);
            if (_shouldFaultOnWrite(writeIndex))
            {
                _onFaultPointReached();
                await _allowFaultTask.WaitAsync(TimeSpan.FromSeconds(10), cancellationToken);
                throw new IOException("Simulated broken pipe during streaming.");
            }
            await _inner.WriteAsync(buffer, cancellationToken);
        }
    }

    [Fact]
    public async Task PathBServe_StreamingFaultMidGeneration_CancelsAndAwaitsGenTaskBeforeReleasingLockAndRestoringLora()
    {
        string ggufPath = Path.Combine(_tempDir, "qwen2_pathb_fault.gguf");
        WriteDistinctBaseVsLoraQwen2Gguf(ggufPath, contextLength: 512u);

        string loraPath = Path.Combine(_tempDir, "roslyn.bin");
        WriteStrongLoraBinary(loraPath);

        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 512, device: "cpu", engine: InferenceEngineType.Cpu);
        session.AttachLora(loraPath);

        var write3Reached = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var allowFault = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        bool? loraWasActiveAtWrite3 = null;
        int injectFaultOnRequest = 1;

        var builder = WebApplication.CreateBuilder();
        var app = builder.Build();
        app.Urls.Add($"http://127.0.0.1:{port}");
        app.Use(async (ctx, next) =>
        {
            if (ctx.Request.Path == "/v1/chat/completions" && Interlocked.CompareExchange(ref injectFaultOnRequest, 0, 1) == 1)
            {
                var origBody = ctx.Response.Body;
                ctx.Response.Body = new FaultAfterFirstTokenChunkStream(
                    origBody,
                    writeIdx => writeIdx == 3,
                    () =>
                    {
                        loraWasActiveAtWrite3 = session.CpuModel?.LoraWeights != null;
                        write3Reached.TrySetResult();
                    },
                    allowFault.Task);
            }
            await next();
        });
        Program.ConfigureServeEndpoints(app, session, ggufPath, apiKey: null, attachedLoraName: "roslyn");

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. Start streaming LoRA request with max_tokens=256 (> 128 BoundedChannel capacity)
            var streamReq = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
            {
                Content = JsonContent.Create(new
                {
                    model = "qwen2_pathb_fault:roslyn",
                    stream = true,
                    max_tokens = 256,
                    temperature = 0.0f,
                    messages = new[] { new { role = "user", content = "hello" } }
                })
            };
            var req1Task = Task.Run(async () =>
            {
                try
                {
                    using var resp = await client.SendAsync(streamReq, HttpCompletionOption.ResponseHeadersRead);
                    using var s = await resp.Content.ReadAsStreamAsync();
                    byte[] buf = new byte[4096];
                    while (await s.ReadAsync(buf) > 0) { }
                }
                catch
                {
                    // Expected when FaultAfterFirstTokenChunkStream aborts mid-stream.
                }
            });

            // Wait deterministically until write #3 is paused inside FaultAfterFirstTokenChunkStream
            await write3Reached.Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.True(loraWasActiveAtWrite3);

            // 2. Launch concurrent base-model request; it MUST wait for s_inferenceLock while req1 is paused
            var req2Task = client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_fault",
                stream = false,
                max_tokens = 4,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            await Task.Yield();
            Assert.False(req2Task.IsCompleted);

            // 3. Unblock write #3 -> throws IOException without RequestAborted; finally must cancel genCts & await genTask
            allowFault.TrySetResult();

            using var req2Resp = await req2Task.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(HttpStatusCode.OK, req2Resp.StatusCode);
            var req2Json = await req2Resp.Content.ReadFromJsonAsync<JsonElement>();
            string req2Content = req2Json.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.Contains("tok_1", req2Content, StringComparison.Ordinal);
            Assert.DoesNotContain("tok_2", req2Content, StringComparison.Ordinal);
            Assert.Null(session.CpuModel!.LoraWeights);

            await req1Task.WaitAsync(TimeSpan.FromSeconds(10));

            // 4. Subsequent LoRA request still works cleanly and clears LoraWeights afterwards
            using var loraFollowUp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_fault:roslyn",
                stream = false,
                max_tokens = 4,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, loraFollowUp.StatusCode);
            var loraJson = await loraFollowUp.Content.ReadFromJsonAsync<JsonElement>();
            string loraContent = loraJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
            Assert.Contains("tok_2", loraContent, StringComparison.Ordinal);
            Assert.Null(session.CpuModel.LoraWeights);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task PathBServe_RealKestrelMidStreamAbort_IsolatesBaseRequestsAcrossAllThreeStreamingEndpoints()
    {
        string ggufPath = Path.Combine(_tempDir, "qwen2_pathb_abort.gguf");
        WriteDistinctBaseVsLoraQwen2Gguf(ggufPath, contextLength: 512u);

        string loraPath = Path.Combine(_tempDir, "roslyn.bin");
        WriteStrongLoraBinary(loraPath);

        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 512, device: "cpu", engine: InferenceEngineType.Cpu);
        session.AttachLora(loraPath);

        await using var app = Program.CreateServeApp(
            session,
            ggufPath,
            apiKey: null,
            host: "127.0.0.1",
            port: port,
            attachedLoraName: "roslyn");

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            async Task AbortStreamAndVerifyBaseAsync(string endpoint, object payload)
            {
                using var streamCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                using var req = new HttpRequestMessage(HttpMethod.Post, endpoint)
                {
                    Content = JsonContent.Create(payload)
                };
                using (var resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, streamCts.Token))
                {
                    Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
                    await using var bodyStream = await resp.Content.ReadAsStreamAsync(streamCts.Token);
                    using var reader = new StreamReader(bodyStream, Encoding.UTF8);
                    bool sawLoraToken = false;
                    while (true)
                    {
                        string? line = await reader.ReadLineAsync(streamCts.Token);
                        if (line == null) break;
                        if (line.Contains("tok_2", StringComparison.Ordinal))
                        {
                            sawLoraToken = true;
                            break;
                        }
                    }
                    Assert.True(sawLoraToken, $"Expected LoRA token 'tok_2' before abort on {endpoint}.");
                    streamCts.Cancel();
                }

                // Immediately follow with a base-model request; it must return only tok_1 (no LoRA leak)
                using var baseResp = await client.PostAsJsonAsync("/v1/chat/completions", new
                {
                    model = "qwen2_pathb_abort",
                    stream = false,
                    max_tokens = 4,
                    temperature = 0.0f,
                    messages = new[] { new { role = "user", content = "hello" } }
                });
                Assert.Equal(HttpStatusCode.OK, baseResp.StatusCode);
                var baseJson = await baseResp.Content.ReadFromJsonAsync<JsonElement>();
                string baseText = baseJson.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
                Assert.Contains("tok_1", baseText, StringComparison.Ordinal);
                Assert.DoesNotContain("tok_2", baseText, StringComparison.Ordinal);
                Assert.Null(session.CpuModel!.LoraWeights);
            }

            await AbortStreamAndVerifyBaseAsync("/v1/chat/completions", new
            {
                model = "qwen2_pathb_abort:roslyn",
                stream = true,
                max_tokens = 256,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });

            await AbortStreamAndVerifyBaseAsync("/api/generate", new
            {
                model = "qwen2_pathb_abort:roslyn",
                stream = true,
                max_tokens = 256,
                temperature = 0.0f,
                prompt = "hello"
            });

            await AbortStreamAndVerifyBaseAsync("/api/chat", new
            {
                model = "qwen2_pathb_abort",
                lora = "roslyn",
                stream = true,
                max_tokens = 256,
                temperature = 0.0f,
                messages = new[] { new { role = "user", content = "hello" } }
            });
        }
        finally
        {
            await app.StopAsync();
        }
    }

    [Fact]
    public async Task PathBServe_ThinkBudgetPrecedence_MatchesPathA_AndValidatesAllFieldsAndNestedOptions()
    {
        // 1. Direct ParseSamplingOptions precedence parity: max_think_tokens > max_thinking_tokens > reasoning_budget
        using (var doc1 = JsonDocument.Parse("""{"max_think_tokens": 1, "max_thinking_tokens": 10, "reasoning_budget": 20, "max_tokens": 64}"""))
        {
            Assert.Equal(1, Program.ParseSamplingOptions(doc1.RootElement).MaxReasoningTokens);
        }
        using (var doc2 = JsonDocument.Parse("""{"reasoning_budget": 1, "max_thinking_tokens": 3, "max_think_tokens": 10, "max_tokens": 64}"""))
        {
            Assert.Equal(10, Program.ParseSamplingOptions(doc2.RootElement).MaxReasoningTokens);
        }
        using (var doc3 = JsonDocument.Parse("""{"reasoning_budget": 1, "max_thinking_tokens": 3, "max_tokens": 64}"""))
        {
            Assert.Equal(3, Program.ParseSamplingOptions(doc3.RootElement).MaxReasoningTokens);
        }

        // 2. HTTP 400 validation across multi-field and nested options combinations
        string ggufPath = Path.Combine(_tempDir, "qwen2_pathb_budget.gguf");
        WriteDistinctBaseVsLoraQwen2Gguf(ggufPath);

        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu);
        await using var app = Program.CreateServeApp(session, ggufPath, apiKey: null, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            string[] invalidPayloads =
            [
                """{"model":"qwen2_pathb_budget","reasoning_budget":5,"max_think_tokens":-2,"messages":[{"role":"user","content":"hi"}]}""",
                """{"model":"qwen2_pathb_budget","max_think_tokens":5,"reasoning_budget":-2,"messages":[{"role":"user","content":"hi"}]}""",
                """{"model":"qwen2_pathb_budget","max_think_tokens":5,"options":{"max_think_tokens":-2},"messages":[{"role":"user","content":"hi"}]}"""
            ];

            foreach (string payload in invalidPayloads)
            {
                using var content = new StringContent(payload, Encoding.UTF8, "application/json");
                using var resp = await client.PostAsync("/v1/chat/completions", content);
                Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
                var errJson = await resp.Content.ReadFromJsonAsync<JsonElement>();
                Assert.Equal("invalid_request_error", errJson.GetProperty("error").GetProperty("type").GetString());
            }
        }
        finally
        {
            await app.StopAsync();
        }
    }

    [Fact]
    public void ResolveDefaultHarnessPath_FindsCrossPlatformBinaryAcrossBaseDirAndRepoTree()
    {
        string exeName = OperatingSystem.IsWindows() ? "Glacier.EvaluationHarness.exe" : "Glacier.EvaluationHarness";
        string fakeRepo = Path.Combine(_tempDir, "fake_repo");
        string binDir = Path.Combine(fakeRepo, "tests", "Glacier.EvaluationHarness", "bin", "Release", "net10.0");
        Directory.CreateDirectory(binDir);
        string fakeExe = Path.Combine(binDir, exeName);
        File.WriteAllBytes(fakeExe, [0x4D, 0x5A]);

        string unrelatedCwd = Path.Combine(_tempDir, "unrelated_cwd");
        Directory.CreateDirectory(unrelatedCwd);

        string nestedAppDir = Path.Combine(fakeRepo, "Glacier.Inference", "bin", "Release", "net10.0");
        Directory.CreateDirectory(nestedAppDir);

        string? resolved = Program.ResolveDefaultHarnessPath(unrelatedCwd, nestedAppDir);
        Assert.Equal(Path.GetFullPath(fakeExe), resolved);

        string? missing = Program.ResolveDefaultHarnessPath(unrelatedCwd, unrelatedCwd);
        Assert.Null(missing);
    }

    [Fact]
    public async Task CreateServeApp_ExposesV1ModelsWithBaseFirst_V1SleepCompat_AndLora404Validation()
    {
        string ggufPath = Path.Combine(_tempDir, "qwen2_cli_bridge.gguf");
        WriteMinimalQwen2Gguf(ggufPath);

        string loraPath = Path.Combine(_tempDir, "roslyn.bin");
        WriteMinimalLoraBinary(loraPath);

        int port = GetAvailablePort();
        using var session = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu);
        session.AttachLora(loraPath);
        await using var sleep = new SleepMemoryService(tokenBudget: 400);

        await using var app = Program.CreateServeApp(
            session,
            ggufPath,
            apiKey: null,
            host: "127.0.0.1",
            port: port,
            sleepMemory: sleep,
            attachedLoraName: "roslyn");

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var modelsResp = await client.GetFromJsonAsync<JsonElement>("/v1/models");
            var data = modelsResp.GetProperty("data");
            Assert.Equal(2, data.GetArrayLength());
            Assert.Equal("qwen2_cli_bridge", data[0].GetProperty("id").GetString());
            Assert.Equal("qwen2_cli_bridge:roslyn", data[1].GetProperty("id").GetString());

            // Glacier.Studio /v1/sleep/* compatibility
            var v1Status = await client.GetFromJsonAsync<SleepStatusResponse>("/v1/sleep/status");
            Assert.NotNull(v1Status);
            Assert.True(v1Status!.Enabled);

            sleep.RecordTurnCompletion("remember port 19001", "port 19001 stored");
            var v1TriggerResp = await client.PostAsync("/v1/sleep/trigger", content: null);
            Assert.Equal(HttpStatusCode.OK, v1TriggerResp.StatusCode);
            var v1Trigger = await v1TriggerResp.Content.ReadFromJsonAsync<SleepTriggerResponse>();
            Assert.NotNull(v1Trigger);
            Assert.True(v1Trigger!.Triggered);

            // Unknown LoRA returns 404 with model_not_found
            var badLoraResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_cli_bridge:unknown_adapter",
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.NotFound, badLoraResp.StatusCode);
            var badLoraJson = await badLoraResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("model_not_found", badLoraJson.GetProperty("error").GetProperty("type").GetString());

            // Conflicting model:<unknown> + lora:<attached> returns 400 Bad Request
            var conflictResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_cli_bridge:unknown_x",
                lora = "roslyn",
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.BadRequest, conflictResp.StatusCode);
            var conflictJson = await conflictResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("invalid_request_error", conflictJson.GetProperty("error").GetProperty("type").GetString());

            // Invalid empty segment in model:adapter returns 400 Bad Request
            var emptySegResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_cli_bridge:",
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.BadRequest, emptySegResp.StatusCode);

            // Matching model:<attached> + lora:<attached> succeeds with 200 OK
            var matchResp = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "qwen2_cli_bridge:roslyn",
                lora = "roslyn",
                max_tokens = 1,
                messages = new[] { new { role = "user", content = "hello" } }
            });
            Assert.Equal(HttpStatusCode.OK, matchResp.StatusCode);
        }
        finally
        {
            await app.StopAsync();
        }
    }

    [Fact]
    public void ResolvePathBLoraRequest_ExemptsExactModelNameWithColon_AndDetectsConflicts()
    {
        var exactColon = Program.ResolvePathBLoraRequest("qwen2:7b", null, "qwen2:7b", attachedLoraName: null);
        Assert.Null(exactColon.ErrorStatusCode);
        Assert.Null(exactColon.SelectedLora);

        var exactColonWithAttached = Program.ResolvePathBLoraRequest("qwen2:7b", null, "qwen2:7b", attachedLoraName: "roslyn");
        Assert.Null(exactColonWithAttached.ErrorStatusCode);
        Assert.Null(exactColonWithAttached.SelectedLora);

        var multiColon = Program.ResolvePathBLoraRequest("qwen2:7b:roslyn", null, "qwen2:7b", attachedLoraName: "roslyn");
        Assert.Null(multiColon.ErrorStatusCode);
        Assert.Equal("roslyn", multiColon.SelectedLora);

        var conflict = Program.ResolvePathBLoraRequest("qwen2:7b:unknown_x", "roslyn", "qwen2:7b", attachedLoraName: "roslyn");
        Assert.Equal(400, conflict.ErrorStatusCode);
        Assert.Equal("invalid_request_error", conflict.ErrorType);

        var unknown = Program.ResolvePathBLoraRequest("qwen2:7b:unknown_x", null, "qwen2:7b", attachedLoraName: "roslyn");
        Assert.Equal(404, unknown.ErrorStatusCode);
        Assert.Equal("model_not_found", unknown.ErrorType);
    }

    [Fact]
    public async Task Program_TrailingValueFlags_ReturnExitCodeOneWithErrorMessage()
    {
        string dummyGguf = Path.Combine(_tempDir, "dummy.gguf");
        var errWriter = new StringWriter();
        var origErr = Console.Error;
        Console.SetError(errWriter);
        try
        {
            string[][] trailingCases =
            [
                ["serve", dummyGguf, "--port"],
                ["serve", dummyGguf, "--host"],
                ["serve", dummyGguf, "--device"],
                ["serve", dummyGguf, "--engine"],
                ["serve", dummyGguf, "--split"],
                ["serve", dummyGguf, "-c"],
                ["serve", dummyGguf, "--max-think-tokens"],
                ["serve", dummyGguf, "--sleep-budget"],
                ["serve", dummyGguf, "--api-key"],
                ["serve", dummyGguf, "--blocks"],
                ["serve", dummyGguf, "-b"],
                ["bench", dummyGguf, "-n"],
                ["bench", dummyGguf, "-p"],
                ["bench", dummyGguf, "-c"],
                ["bench", dummyGguf, "--device"],
                ["bench", dummyGguf, "--host-arena-gib"],
                ["run", dummyGguf, "--device"],
                ["run", dummyGguf, "--temp"],
                ["run", dummyGguf, "--top-p"],
                ["eval-kl", dummyGguf, "--ref", dummyGguf, "--threshold"],
                ["eval-kl", dummyGguf, "--ref", dummyGguf, "-p"],
                ["eval-kl", dummyGguf, "--ref", dummyGguf, "--kv-precision"]
            ];

            foreach (var argv in trailingCases)
            {
                errWriter.GetStringBuilder().Clear();
                int code = await Program.Main(argv);
                Assert.Equal(1, code);
                Assert.Contains($"Option '{argv[^1]}' requires a value.", errWriter.ToString());
            }
        }
        finally
        {
            Console.SetError(origErr);
        }
    }

    [Fact]
    public async Task Program_DeviceAutoWithLora_DefaultsToCpu()
    {
        var (autoDev, autoEng) = Program.ResolveSingleLoraTarget(hasLora: true, device: "auto", engine: InferenceEngineType.Auto);
        Assert.Equal("cpu", autoDev);
        Assert.Equal(InferenceEngineType.Cpu, autoEng);

        var (upperDev, upperEng) = Program.ResolveSingleLoraTarget(hasLora: true, device: "AUTO", engine: InferenceEngineType.Auto);
        Assert.Equal("cpu", upperDev);
        Assert.Equal(InferenceEngineType.Cpu, upperEng);

        var (explicitDev, explicitEng) = Program.ResolveSingleLoraTarget(hasLora: true, device: "nvidia-rtx-4060", engine: InferenceEngineType.Auto);
        Assert.Equal("nvidia-rtx-4060", explicitDev);
        Assert.Equal(InferenceEngineType.Auto, explicitEng);

        string ggufPath = Path.Combine(_tempDir, "run_auto_lora.gguf");
        WriteMinimalQwen2Gguf(ggufPath);
        string loraPath = Path.Combine(_tempDir, "roslyn.bin");
        WriteMinimalLoraBinary(loraPath);

        var outWriter = new StringWriter();
        var origOut = Console.Out;
        Console.SetOut(outWriter);
        try
        {
            int code = await Program.Main(["run", ggufPath, "hi", "-n", "1", "--device", "auto", "--lora", $"roslyn={loraPath}"]);
            Assert.Equal(0, code);
            Assert.Contains("Attached Pure C# LoRA Adapter: roslyn", outWriter.ToString());
        }
        finally
        {
            Console.SetOut(origOut);
        }
    }

    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static void WriteMinimalLoraBinary(string path)
    {
        const int dim = 64;
        const int rank = 4;
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);
        bw.Write(0x41524F4CU); // 'LORA'
        bw.Write(1U);
        bw.Write(1); // layerCount
        bw.Write(rank);
        bw.Write(8.0f); // alpha
        bw.Write(2); // tensorCount

        bw.Write("blk.0.attn_q.lora_a.weight");
        bw.Write(dim);
        bw.Write(rank);
        for (int i = 0; i < dim * rank; i++) bw.Write(0.01f);

        bw.Write("blk.0.attn_q.lora_b.weight");
        bw.Write(rank);
        bw.Write(dim);
        for (int i = 0; i < rank * dim; i++) bw.Write(0.02f);
    }

    private static void WriteStrongLoraBinary(string path)
    {
        const int dim = 64;
        const int kvDim = 32;
        const int rank = 4;
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);
        bw.Write(0x41524F4CU); // 'LORA'
        bw.Write(1U);
        bw.Write(1); // layerCount
        bw.Write(rank);
        bw.Write(32.0f); // alpha
        bw.Write(2); // tensorCount

        bw.Write("blk.0.attn_v.lora_a.weight");
        bw.Write(dim);
        bw.Write(rank);
        for (int i = 0; i < dim; i++)
            for (int r = 0; r < rank; r++)
                bw.Write(i == 0 ? 1.0f : 0.0f);

        bw.Write("blk.0.attn_v.lora_b.weight");
        bw.Write(rank);
        bw.Write(kvDim);
        for (int r = 0; r < rank; r++)
            for (int c = 0; c < kvDim; c++)
                bw.Write(c == 1 ? 1.0f : 0.0f);
    }

    private static void WriteDistinctBaseVsLoraQwen2Gguf(string path, uint contextLength = 64u)
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
            vocab[i] = i == vocabSize - 1 ? "<|endoftext|>"
                : i == vocabSize - 2 ? "<|im_start|>"
                : i == vocabSize - 3 ? "<think>"
                : i == vocabSize - 4 ? "</think>"
                : i == vocabSize - 5 ? "<｜User｜>"
                : i == vocabSize - 6 ? "<｜Assistant｜>"
                : i == vocabSize - 7 ? "<｜end of sentence｜>"
                : $"tok_{i}";
        }

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("qwen2.block_count", GgufValueType.Uint32, 1u),
            ("qwen2.context_length", GgufValueType.Uint32, contextLength),
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

        float[] tokEmbd = new float[dim * vocabSize];
        for (int t = 0; t < vocabSize; t++)
            tokEmbd[t * dim + 0] = 1.0f;

        float[] attnOut = new float[headCount * headDim * dim];
        for (int i = 0; i < headDim; i++)
            attnOut[i * dim + i] = 1.0f;

        float[] outWeight = new float[dim * vocabSize];
        // Dim 0 favors tok_1 (base); Dim 1 favors tok_2 (LoRA)
        outWeight[1 * dim + 0] = 5.0f;
        outWeight[2 * dim + 1] = 10.0f;

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, tokEmbd),
            ("output_norm.weight", new[] { (ulong)dim }, ones),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, outWeight),
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ones),
            ("blk.0.attn_q.weight", new[] { (ulong)dim, (ulong)(headCount * headDim) }, new float[dim * headCount * headDim]),
            ("blk.0.attn_k.weight", new[] { (ulong)dim, (ulong)(headCountKv * headDim) }, new float[dim * headCountKv * headDim]),
            ("blk.0.attn_v.weight", new[] { (ulong)dim, (ulong)(headCountKv * headDim) }, new float[dim * headCountKv * headDim]),
            ("blk.0.attn_output.weight", new[] { (ulong)(headCount * headDim), (ulong)dim }, attnOut),
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
