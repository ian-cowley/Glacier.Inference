namespace Glacier.Inference.Tests;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Xunit;

public sealed class NpuSidecarDiscoveryCachingTests
{
    private sealed class StubDiscoveryHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        public ConcurrentDictionary<string, int> HitCounts { get; } = new(StringComparer.OrdinalIgnoreCase);
        public int TotalHits => _totalHits;
        private int _totalHits;

        public StubDiscoveryHandler(Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string key = request.RequestUri?.ToString() ?? string.Empty;
            int hit = HitCounts.AddOrUpdate(key, 1, static (_, c) => c + 1);
            Interlocked.Increment(ref _totalHits);
            return await _responder(request, hit, cancellationToken).ConfigureAwait(false);
        }
    }

    private static HttpResponseMessage ModelsJsonResponse(params string[] modelIds)
    {
        var sb = new StringBuilder("{\"object\":\"list\",\"data\":[");
        for (int i = 0; i < modelIds.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append("{\"id\":\"").Append(modelIds[i]).Append("\"}");
        }
        sb.Append("]}");
        return new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(sb.ToString(), Encoding.UTF8, "application/json")
        };
    }

    [Fact]
    public async Task ProbeAsync_ParallelProbing_PreservesPriorityOrder_EvenWhenLowerPriorityRespondsFaster()
    {
        const string firstUrl = "http://127.0.0.1:59001";
        const string secondUrl = "http://127.0.0.1:59002";
        const string thirdUrl = "http://127.0.0.1:59003";

        using var handler = new StubDiscoveryHandler(async (req, _, ct) =>
        {
            string uri = req.RequestUri!.ToString();
            if (uri.StartsWith(firstUrl, StringComparison.OrdinalIgnoreCase))
            {
                await Task.Delay(50, ct).ConfigureAwait(false);
                return ModelsJsonResponse("flm-qwen3");
            }
            if (uri.StartsWith(secondUrl, StringComparison.OrdinalIgnoreCase))
            {
                await Task.Delay(1, ct).ConfigureAwait(false);
                return ModelsJsonResponse("lemonade-gemma");
            }
            return ModelsJsonResponse("custom-model");
        });
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (firstUrl, NpuSidecarRuntime.FastFlowLM),
            (secondUrl, NpuSidecarRuntime.LemonadeServer),
            (thirdUrl, NpuSidecarRuntime.CustomOpenAi)
        };

        NpuSidecarEndpoint? result = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: client,
            candidates: candidates,
            perEndpointTimeout: TimeSpan.FromSeconds(2));

        Assert.NotNull(result);
        Assert.Equal(firstUrl, result.BaseUrl);
        Assert.Equal(NpuSidecarRuntime.FastFlowLM, result.Runtime);
        Assert.Equal(["flm-qwen3"], result.Models);
        Assert.Equal("flm-qwen3", result.PrimaryModel);
    }

    [Fact]
    public async Task ProbeAsync_ParallelProbing_FallsBackToSecondCandidate_WhenFirstFails()
    {
        const string deadUrl = "http://127.0.0.1:59011";
        const string aliveUrl = "http://127.0.0.1:59012";

        using var handler = new StubDiscoveryHandler((req, _, _) =>
        {
            if (req.RequestUri!.ToString().StartsWith(deadUrl, StringComparison.OrdinalIgnoreCase))
                throw new HttpRequestException("Connection refused");

            return Task.FromResult(ModelsJsonResponse("lemonade-phi4"));
        });
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (deadUrl, NpuSidecarRuntime.FastFlowLM),
            (aliveUrl, NpuSidecarRuntime.LemonadeServer)
        };

        NpuSidecarEndpoint? result = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: client,
            candidates: candidates,
            perEndpointTimeout: TimeSpan.FromSeconds(2));

        Assert.NotNull(result);
        Assert.Equal(aliveUrl, result.BaseUrl);
        Assert.Equal(NpuSidecarRuntime.LemonadeServer, result.Runtime);
        Assert.Equal("lemonade-phi4", result.PrimaryModel);
    }

    [Fact]
    public async Task ProbeAsync_ParallelProbing_AllOffline_RunsConcurrently()
    {
        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            ("http://127.0.0.1:59021", NpuSidecarRuntime.FastFlowLM),
            ("http://127.0.0.1:59022", NpuSidecarRuntime.LemonadeServer),
            ("http://127.0.0.1:59023", NpuSidecarRuntime.LemonadeServer),
            ("http://127.0.0.1:59024", NpuSidecarRuntime.CustomOpenAi)
        };

        int inFlight = 0;
        int maxInFlight = 0;
        var allArrived = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        using var handler = new StubDiscoveryHandler(async (_, _, ct) =>
        {
            int current = Interlocked.Increment(ref inFlight);
            int observedMax;
            do
            {
                observedMax = Volatile.Read(ref maxInFlight);
                if (current <= observedMax) break;
            } while (Interlocked.CompareExchange(ref maxInFlight, current, observedMax) != observedMax);

            if (current == candidates.Length)
            {
                allArrived.TrySetResult();
            }

            try
            {
                await allArrived.Task.WaitAsync(TimeSpan.FromSeconds(5), ct).ConfigureAwait(false);
                return new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
            }
            finally
            {
                Interlocked.Decrement(ref inFlight);
            }
        });
        using var client = new HttpClient(handler);

        NpuSidecarEndpoint? result = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: client,
            candidates: candidates,
            perEndpointTimeout: TimeSpan.FromSeconds(10));

        Assert.Null(result);
        Assert.Equal(4, handler.TotalHits);
        Assert.Equal(4, maxInFlight);
        Assert.True(allArrived.Task.IsCompletedSuccessfully, "All 4 candidate probes should be simultaneously in-flight.");
    }

    [Fact]
    public async Task ProbeAsync_RejectsNonOpenAiHtmlAndInvalidUri()
    {
        using var handler = new StubDiscoveryHandler((req, _, _) =>
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("<html><body>Not an OpenAI server</body></html>", Encoding.UTF8, "text/html")
            });
        });
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            ("not a valid uri", NpuSidecarRuntime.CustomOpenAi),
            ("http://127.0.0.1:59031", NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? result = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: client,
            candidates: candidates,
            perEndpointTimeout: TimeSpan.FromSeconds(1));

        Assert.Null(result);
    }

    [Fact]
    public async Task ProbeAsync_PreservesCallerCancellation()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            await NpuSidecarDiscovery.ProbeAsync(
                candidates: [("http://127.0.0.1:59035", NpuSidecarRuntime.FastFlowLM)],
                cancellationToken: cts.Token));
    }

    [Fact]
    public async Task ProbeCachedAsync_PositiveHit_AvoidsDuplicateProbesWithinTtl()
    {
        const string endpoint = "http://127.0.0.1:59041";
        using var handler = new StubDiscoveryHandler((_, _, _) =>
            Task.FromResult(ModelsJsonResponse("embeddinggemma")));
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? first = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);
        NpuSidecarEndpoint? second = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);

        Assert.NotNull(first);
        Assert.Same(first, second);
        Assert.Equal(1, handler.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_ExpiresAfterTtl_AndReprobes()
    {
        const string endpoint = "http://127.0.0.1:59051";
        using var handler = new StubDiscoveryHandler((_, hit, _) =>
            Task.FromResult(ModelsJsonResponse($"model-v{hit}")));
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? first = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromMilliseconds(25),
            httpClient: client,
            candidates: candidates);

        await Task.Delay(120);

        NpuSidecarEndpoint? second = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromMilliseconds(25),
            httpClient: client,
            candidates: candidates);

        Assert.NotNull(first);
        Assert.NotNull(second);
        Assert.Equal("model-v1", first.PrimaryModel);
        Assert.Equal("model-v2", second.PrimaryModel);
        Assert.Equal(2, handler.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_NegativeResult_IsCachedWithinTtl()
    {
        const string endpoint = "http://127.0.0.1:59061";
        using var handler = new StubDiscoveryHandler((_, _, _) =>
            throw new HttpRequestException("Offline"));
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? first = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);
        NpuSidecarEndpoint? second = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);

        Assert.Null(first);
        Assert.Null(second);
        Assert.Equal(1, handler.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_InvalidateCache_ForcesFreshProbe()
    {
        const string endpoint = "http://127.0.0.1:59071";
        using var handler = new StubDiscoveryHandler((_, hit, _) =>
            Task.FromResult(ModelsJsonResponse($"qwen3-{hit}")));
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? first = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);
        Assert.Equal("qwen3-1", first?.PrimaryModel);

        NpuSidecarDiscovery.InvalidateCache();

        NpuSidecarEndpoint? second = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.FromSeconds(30),
            httpClient: client,
            candidates: candidates);
        Assert.Equal("qwen3-2", second?.PrimaryModel);
        Assert.Equal(2, handler.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_ConcurrentStampede_CoalescesIntoSingleProbeRound()
    {
        const string endpoint = "http://127.0.0.1:59081";
        using var handler = new StubDiscoveryHandler(async (_, _, ct) =>
        {
            await Task.Delay(45, ct).ConfigureAwait(false);
            return ModelsJsonResponse("coalesced-model");
        });
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        var tasks = new Task<NpuSidecarEndpoint?>[8];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = NpuSidecarDiscovery.ProbeCachedAsync(
                cacheTtl: TimeSpan.FromSeconds(30),
                httpClient: client,
                candidates: candidates);
        }

        NpuSidecarEndpoint?[] results = await Task.WhenAll(tasks);

        Assert.Equal(1, handler.TotalHits);
        foreach (var r in results)
        {
            Assert.NotNull(r);
            Assert.Equal("coalesced-model", r.PrimaryModel);
        }
    }

    [Fact]
    public async Task ProbeCachedAsync_ZeroTtl_BypassesCache()
    {
        const string endpoint = "http://127.0.0.1:59091";
        using var handler = new StubDiscoveryHandler((_, hit, _) =>
            Task.FromResult(ModelsJsonResponse($"nocache-{hit}")));
        using var client = new HttpClient(handler);

        var candidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            (endpoint, NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? first = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.Zero,
            httpClient: client,
            candidates: candidates);
        NpuSidecarEndpoint? second = await NpuSidecarDiscovery.ProbeCachedAsync(
            cacheTtl: TimeSpan.Zero,
            httpClient: client,
            candidates: candidates);

        Assert.Equal("nocache-1", first?.PrimaryModel);
        Assert.Equal("nocache-2", second?.PrimaryModel);
        Assert.Equal(2, handler.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_IsolatesDifferentClientsAndCandidateLists()
    {
        const string endpointA = "http://127.0.0.1:59101";
        const string endpointB = "http://127.0.0.1:59102";

        using var handler1 = new StubDiscoveryHandler((req, _, _) =>
            Task.FromResult(ModelsJsonResponse(req.RequestUri!.Host + ":" + req.RequestUri.Port)));
        using var handler2 = new StubDiscoveryHandler((_, _, _) =>
            Task.FromResult(ModelsJsonResponse("client2-model")));

        using var client1 = new HttpClient(handler1);
        using var client2 = new HttpClient(handler2);

        NpuSidecarEndpoint? a1 = await NpuSidecarDiscovery.ProbeCachedAsync(
            TimeSpan.FromSeconds(30),
            client1,
            [(endpointA, NpuSidecarRuntime.FastFlowLM)]);
        NpuSidecarEndpoint? b1 = await NpuSidecarDiscovery.ProbeCachedAsync(
            TimeSpan.FromSeconds(30),
            client1,
            [(endpointB, NpuSidecarRuntime.LemonadeServer)]);
        NpuSidecarEndpoint? a2 = await NpuSidecarDiscovery.ProbeCachedAsync(
            TimeSpan.FromSeconds(30),
            client2,
            [(endpointA, NpuSidecarRuntime.FastFlowLM)]);

        Assert.Equal("127.0.0.1:59101", a1?.PrimaryModel);
        Assert.Equal("127.0.0.1:59102", b1?.PrimaryModel);
        Assert.Equal("client2-model", a2?.PrimaryModel);
        Assert.Equal(2, handler1.TotalHits);
        Assert.Equal(1, handler2.TotalHits);
    }

    [Fact]
    public async Task ProbeCachedAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            await NpuSidecarDiscovery.ProbeCachedAsync(
                cacheTtl: TimeSpan.FromSeconds(5),
                candidates: [("http://127.0.0.1:59111", NpuSidecarRuntime.FastFlowLM)],
                cancellationToken: cts.Token));
    }

    [Fact]
    public void DefaultCacheTtl_IsFiveSeconds()
    {
        Assert.Equal(TimeSpan.FromSeconds(5), NpuSidecarDiscovery.DefaultCacheTtl);
    }

    [Fact]
    public async Task ProbeAsync_RealHttpClient_SkipsNonHttpSchemesAndFallsBackCleanly()
    {
        var invalidOnly = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            ("ftp://127.0.0.1:52625", NpuSidecarRuntime.CustomOpenAi),
            ("localhost:52625", NpuSidecarRuntime.CustomOpenAi),
            ("htt://127.0.0.1:52625", NpuSidecarRuntime.CustomOpenAi),
            ("file:///C:/nonexistent", NpuSidecarRuntime.CustomOpenAi)
        };

        // Must not throw NotSupportedException on real HttpClient (both null and caller-supplied)
        NpuSidecarEndpoint? nullResult = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: null,
            candidates: invalidOnly,
            perEndpointTimeout: TimeSpan.FromMilliseconds(300));
        Assert.Null(nullResult);

        using var explicitRealClient = new HttpClient();
        NpuSidecarEndpoint? explicitClientResult = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: explicitRealClient,
            candidates: invalidOnly,
            perEndpointTimeout: TimeSpan.FromMilliseconds(300));
        Assert.Null(explicitClientResult);

        // Also verify fallback to a valid loopback HTTP endpoint after skipping invalid schemes
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        using var serverCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task serverTask = ServeSingleModelsResponseAsync(listener, "{\"data\":[{\"id\":\"flm-fallback\"}]}", serverCts.Token);

        var mixedCandidates = new (string BaseUrl, NpuSidecarRuntime Runtime)[]
        {
            ("ftp://127.0.0.1:52625", NpuSidecarRuntime.CustomOpenAi),
            ("localhost:52625", NpuSidecarRuntime.CustomOpenAi),
            ("htt://127.0.0.1:52625", NpuSidecarRuntime.CustomOpenAi),
            ("file:///C:/nonexistent", NpuSidecarRuntime.CustomOpenAi),
            ($"http://127.0.0.1:{port}", NpuSidecarRuntime.FastFlowLM)
        };

        NpuSidecarEndpoint? fallbackResult = await NpuSidecarDiscovery.ProbeAsync(
            httpClient: null,
            candidates: mixedCandidates,
            perEndpointTimeout: TimeSpan.FromSeconds(2));

        await serverTask;

        Assert.NotNull(fallbackResult);
        Assert.Equal($"http://127.0.0.1:{port}", fallbackResult.BaseUrl);
        Assert.Equal("flm-fallback", fallbackResult.PrimaryModel);
    }

    [Fact]
    public void GetDefaultCandidateEndpoints_ClassifiesRuntimeByExactPort()
    {
        var port8001 = NpuSidecarDiscovery.GetDefaultCandidateEndpoints("http://127.0.0.1:8001");
        Assert.Equal(NpuSidecarRuntime.CustomOpenAi, port8001[0].Runtime);

        var pathWithPortSubstring = NpuSidecarDiscovery.GetDefaultCandidateEndpoints("http://127.0.0.1:9000/proxy:8000");
        Assert.Equal(NpuSidecarRuntime.CustomOpenAi, pathWithPortSubstring[0].Runtime);

        var flm = NpuSidecarDiscovery.GetDefaultCandidateEndpoints("http://127.0.0.1:52625/v1");
        Assert.Equal(NpuSidecarRuntime.FastFlowLM, flm[0].Runtime);

        var lemonade = NpuSidecarDiscovery.GetDefaultCandidateEndpoints("http://127.0.0.1:8000/v1");
        Assert.Equal(NpuSidecarRuntime.LemonadeServer, lemonade[0].Runtime);
    }

    private static async Task ServeSingleModelsResponseAsync(TcpListener listener, string jsonBody, CancellationToken ct)
    {
        using TcpClient client = await listener.AcceptTcpClientAsync(ct).ConfigureAwait(false);
        using NetworkStream stream = client.GetStream();
        var buffer = new byte[2048];
        int totalRead = 0;
        while (totalRead < buffer.Length)
        {
            int read = await stream.ReadAsync(buffer.AsMemory(totalRead), ct).ConfigureAwait(false);
            if (read == 0) break;
            totalRead += read;
            if (Encoding.ASCII.GetString(buffer, 0, totalRead).Contains("\r\n\r\n", StringComparison.Ordinal))
                break;
        }

        byte[] bodyBytes = Encoding.UTF8.GetBytes(jsonBody);
        string headers = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {bodyBytes.Length}\r\nConnection: close\r\n\r\n";
        byte[] headerBytes = Encoding.ASCII.GetBytes(headers);
        await stream.WriteAsync(headerBytes, ct).ConfigureAwait(false);
        await stream.WriteAsync(bodyBytes, ct).ConfigureAwait(false);
        await stream.FlushAsync(ct).ConfigureAwait(false);
    }
}
