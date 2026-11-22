using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Bounce;
using Xunit;

namespace Glacier.Inference.Tests;

public sealed class PeerBounceTests
{
    private sealed class StubHttpMessageHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, string?, HttpResponseMessage> _handler;

        public StubHttpMessageHandler(Func<HttpRequestMessage, string?, HttpResponseMessage> handler)
        {
            _handler = handler;
        }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string? body = request.Content != null ? await request.Content.ReadAsStringAsync(cancellationToken) : null;
            return _handler(request, body);
        }
    }

    private sealed class FakeBounceVerifier : IBounceVerifier
    {
        private readonly Func<BounceTaskKind, string, BounceVerificationResult> _fn;
        public int CallCount { get; private set; }

        public FakeBounceVerifier(Func<BounceTaskKind, string, BounceVerificationResult> fn)
        {
            _fn = fn;
        }

        public Task<BounceVerificationResult> VerifyAsync(BounceTaskKind kind, string promptText, string completionText, CancellationToken ct)
        {
            CallCount++;
            return Task.FromResult(_fn(kind, completionText));
        }
    }

    private static HttpResponseMessage JsonResp(object payload) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
        };

    [Fact]
    public void ClassifyRequest_BypassesSleepMemoryAndDetectsCodeTasks()
    {
        var memMsgs = new ChatMessage[]
        {
            new() { Role = "system", Content = "CONSOLIDATED EPISODIC MEMORY:\nFact 1: port=8080" },
            new() { Role = "user", Content = "What port is configured?" }
        };
        Assert.Equal(BounceTaskKind.Bypass, PeerBounceService.ClassifyRequest(memMsgs, 256));

        Assert.Equal(BounceTaskKind.CSharp, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Write a C# Vector256 dot product." }], 1024));
        Assert.Equal(BounceTaskKind.Hlsl, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Write an HLSL cs_6_6 compute shader." }], 1024));
        Assert.Equal(BounceTaskKind.HtmlJs, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Build a single-file HTML5 JavaScript app." }], 2048));
        Assert.Equal(BounceTaskKind.Python, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Write a Python function def solve(x):" }], 1024));
        Assert.Equal(BounceTaskKind.ToolCall, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Emit <tool_call> JSON." }], 256));
        Assert.Equal(BounceTaskKind.FinalAnswer, PeerBounceService.ClassifyRequest([new() { Role = "user", Content = "Compute 7*8 and end with FINAL_ANSWER: <int>" }], 384));
    }

    [Fact]
    public async Task CompleteAsync_BypassesCriticAndVerifier_ForSleepMemoryPrompts()
    {
        int criticCalls = 0;
        var handler = new StubHttpMessageHandler((req, _) =>
        {
            if (req.RequestUri!.Host == "critic.local")
            {
                criticCalls++;
                throw new InvalidOperationException("Critic should not be called on bypassed requests.");
            }
            if (req.RequestUri.AbsolutePath == "/v1/models")
                return JsonResp(new { data = new[] { new { id = "glacier-14b-v3" } } });
            return JsonResp(new
            {
                choices = new[] { new { message = new { content = "Port 8080." }, finish_reason = "stop" } },
                usage = new { prompt_tokens = 40, completion_tokens = 4 }
            });
        });

        var verifier = new FakeBounceVerifier((_, _) => new BounceVerificationResult(true, null));
        using var http = new HttpClient(handler);
        var svc = new PeerBounceService(http, verifier, "http://primary.local:18080", "http://critic.local:8080", maxRepairs: 2);

        var msgs = new ChatMessage[]
        {
            new() { Role = "system", Content = "CONSOLIDATED EPISODIC MEMORY: port is 8080." },
            new() { Role = "user", Content = "What is the port?" }
        };

        var res = await svc.CompleteAsync(msgs, null, 0.0, 128, CancellationToken.None);
        Assert.True(res.Bypassed);
        Assert.False(res.CriticAssisted);
        Assert.Equal(0, criticCalls);
        Assert.Equal(0, verifier.CallCount);
        Assert.Equal("Port 8080.", res.Content);
    }

    [Fact]
    public async Task CompleteAsync_ConsultsCriticAndAutoRepairsCompilerFailure()
    {
        int primaryCalls = 0;
        int criticCalls = 0;
        var handler = new StubHttpMessageHandler((req, _) =>
        {
            if (req.RequestUri!.AbsolutePath == "/v1/models")
            {
                string id = req.RequestUri.Host == "critic.local" ? "qwen3.6-35b" : "glacier-14b-v3";
                return JsonResp(new { data = new[] { new { id } } });
            }
            if (req.RequestUri.Host == "critic.local")
            {
                criticCalls++;
                return JsonResp(new
                {
                    choices = new[] { new { message = new { content = "- Pin ReadOnlySpan<float> using fixed (float* p = span)." }, finish_reason = "stop" } },
                    usage = new { prompt_tokens = 20, completion_tokens = 15 }
                });
            }
            primaryCalls++;
            string reply = primaryCalls == 1
                ? "```csharp\n// broken code with CS1503\n```"
                : "```csharp\n// fixed code with fixed(float* p = a)\n```";
            return JsonResp(new
            {
                choices = new[] { new { message = new { content = reply }, finish_reason = "stop" } },
                usage = new { prompt_tokens = 50, completion_tokens = 25 }
            });
        });

        var verifier = new FakeBounceVerifier((_, text) =>
            text.Contains("CS1503", StringComparison.Ordinal)
                ? new BounceVerificationResult(false, "CS1503: cannot convert ReadOnlySpan<float> to float*")
                : new BounceVerificationResult(true, null));

        using var http = new HttpClient(handler);
        var svc = new PeerBounceService(http, verifier, "http://primary.local:18080", "http://critic.local:8080", maxRepairs: 2);

        var msgs = new ChatMessage[]
        {
            new() { Role = "user", Content = "Write a C# .NET 10 Vector256 SIMD dot product over ReadOnlySpan<float>." }
        };

        var res = await svc.CompleteAsync(msgs, null, 0.0, 1024, CancellationToken.None);
        Assert.False(res.Bypassed);
        Assert.True(res.CriticAssisted);
        Assert.Equal(1, res.RepairsUsed);
        Assert.Equal(1, criticCalls);
        Assert.Equal(2, primaryCalls);
        Assert.Contains("fixed(float* p = a)", res.Content, StringComparison.Ordinal);

        var status = svc.GetStatus();
        Assert.Equal(1, status.TotalRequests);
        Assert.Equal(1, status.CriticAssistedRequests);
        Assert.Equal(1, status.RepairAttempts);
        Assert.Equal(1, status.RepairsSucceeded);
    }

    [Fact]
    public async Task CompleteAsync_GracefullyFallsBack_WhenCriticIsOffline()
    {
        var handler = new StubHttpMessageHandler((req, _) =>
        {
            if (req.RequestUri!.Host == "critic.local")
                throw new HttpRequestException("Simulated .200 ethernet disconnect");
            if (req.RequestUri.AbsolutePath == "/v1/models")
                return JsonResp(new { data = new[] { new { id = "glacier-14b-v3" } } });
            return JsonResp(new
            {
                choices = new[] { new { message = new { content = "```hlsl\n[numthreads(64,1,1)] void CSMain() {}\n```" }, finish_reason = "stop" } },
                usage = new { prompt_tokens = 30, completion_tokens = 20 }
            });
        });

        var verifier = new FakeBounceVerifier((_, _) => new BounceVerificationResult(true, null));
        using var http = new HttpClient(handler);
        var svc = new PeerBounceService(http, verifier, "http://primary.local:18080", "http://critic.local:8080", maxRepairs: 2);

        var res = await svc.CompleteAsync([new() { Role = "user", Content = "Write an HLSL cs_6_6 shader." }], null, 0.0, 512, CancellationToken.None);
        Assert.False(res.CriticAssisted);
        Assert.Equal(0, res.RepairsUsed);
        Assert.Equal(1, svc.GetStatus().CriticFallbacks);
        Assert.Equal(1, svc.GetStatus().VerifierFirstPassOk);
    }

    [Fact]
    public async Task LocalProcessBounceVerifier_ValidatesToolCallAndFinalAnswer()
    {
        var verifier = new LocalProcessBounceVerifier(harnessExePath: null);

        var okTc = await verifier.VerifyAsync(BounceTaskKind.ToolCall, "use <tool_call>", "<tool_call>{\"name\":\"scan\",\"arguments\":{\"path\":\".\"}}</tool_call>", CancellationToken.None);
        Assert.True(okTc.Success);

        var badTc = await verifier.VerifyAsync(BounceTaskKind.ToolCall, "use <tool_call>", "{\"name\":\"scan\",\"arguments\":{}}", CancellationToken.None);
        Assert.False(badTc.Success);

        var okFa = await verifier.VerifyAsync(BounceTaskKind.FinalAnswer, "FINAL_ANSWER:", "Step 1: 6*7=42\nFINAL_ANSWER: 42", CancellationToken.None);
        Assert.True(okFa.Success);

        var badFa = await verifier.VerifyAsync(BounceTaskKind.FinalAnswer, "FINAL_ANSWER:", "Step 1: expanding integral...", CancellationToken.None);
        Assert.False(badFa.Success);
    }
}
