namespace Glacier.Inference.Tests;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Xunit;

[Collection("SequentialGpu")]
public sealed class Tier5M7Phase2HardeningTests
{
    private sealed class StubDiscoveryHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        private int _hits;
        public int Hits => Volatile.Read(ref _hits);

        public StubDiscoveryHandler(Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            int hit = Interlocked.Increment(ref _hits);
            return _responder(request, hit, cancellationToken);
        }
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    [Fact]
    public async Task NpuSidecarDiscovery_InvalidateMidFlight_StaleProbeDoesNotPoisonFreshCache()
    {
        NpuSidecarDiscovery.InvalidateCache();
        try
        {
            var firstProbeStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var releaseFirstProbe = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            int generation = 0;

            using var handler = new StubDiscoveryHandler(async (req, hit, ct) =>
            {
                if (req.RequestUri?.Host == "127.0.0.1" && req.RequestUri.Port == 52625)
                {
                    if (Volatile.Read(ref generation) == 0)
                    {
                        firstProbeStarted.TrySetResult();
                        await releaseFirstProbe.Task.WaitAsync(ct).ConfigureAwait(false);
                        return Json(HttpStatusCode.OK, """{"data":[{"id":"stale-model-gen0"}]}""");
                    }
                    return Json(HttpStatusCode.OK, """{"data":[{"id":"fresh-model-gen1"}]}""");
                }
                return Json(HttpStatusCode.NotFound, "{}");
            });

            using var http = new HttpClient(handler);
            var candidates = new (string, NpuSidecarRuntime)[]
            {
                ("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM)
            };

            Task<NpuSidecarEndpoint?> staleTask = NpuSidecarDiscovery.ProbeCachedAsync(
                TimeSpan.FromMinutes(1), http, candidates, TimeSpan.FromSeconds(5));

            await firstProbeStarted.Task;
            Interlocked.Exchange(ref generation, 1);
            NpuSidecarDiscovery.InvalidateCache();

            NpuSidecarEndpoint? freshResult = await NpuSidecarDiscovery.ProbeCachedAsync(
                TimeSpan.FromMinutes(1), http, candidates, TimeSpan.FromSeconds(5));

            releaseFirstProbe.TrySetResult();
            await staleTask;

            NpuSidecarEndpoint? cachedAfter = await NpuSidecarDiscovery.ProbeCachedAsync(
                TimeSpan.FromMinutes(1), http, candidates, TimeSpan.FromSeconds(5));

            Assert.NotNull(freshResult);
            Assert.Contains("fresh-model-gen1", freshResult!.Models);
            Assert.NotNull(cachedAfter);
            Assert.Contains("fresh-model-gen1", cachedAfter!.Models);
            Assert.DoesNotContain("stale-model-gen0", cachedAfter.Models);
        }
        finally
        {
            NpuSidecarDiscovery.InvalidateCache();
        }
    }

    [Fact]
    public async Task NpuSidecarDiscovery_PartialCancellationAndPriorityFallback_SucceedsForNonCancelledWaiter()
    {
        NpuSidecarDiscovery.InvalidateCache();
        try
        {
            var probeEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var releaseProbe = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

            using var handler = new StubDiscoveryHandler(async (req, _, ct) =>
            {
                string hostPort = $"{req.RequestUri?.Host}:{req.RequestUri?.Port}";
                if (hostPort == "127.0.0.1:52625")
                {
                    probeEntered.TrySetResult();
                    await releaseProbe.Task.WaitAsync(ct).ConfigureAwait(false);
                    return Json(HttpStatusCode.InternalServerError, """{"error":"500"}""");
                }
                if (hostPort == "127.0.0.1:8000")
                {
                    return Json(HttpStatusCode.OK, """{"data":null}"""); // null data rejected cleanly
                }
                if (hostPort == "127.0.0.1:13305")
                {
                    return Json(HttpStatusCode.OK, "{not-valid-json"); // malformed JSON
                }
                if (hostPort == "127.0.0.1:19999")
                {
                    return Json(HttpStatusCode.OK, """{"data":[{"id":"priority4-npu"}]}""");
                }
                return Json(HttpStatusCode.NotFound, "{}");
            });

            using var http = new HttpClient(handler);
            var candidates = new (string, NpuSidecarRuntime)[]
            {
                ("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM),
                ("http://127.0.0.1:8000", NpuSidecarRuntime.LemonadeServer),
                ("http://127.0.0.1:13305", NpuSidecarRuntime.LemonadeServer),
                ("http://127.0.0.1:19999", NpuSidecarRuntime.CustomOpenAi)
            };

            using var cts1 = new CancellationTokenSource();
            Task<NpuSidecarEndpoint?> waiter1 = NpuSidecarDiscovery.ProbeCachedAsync(
                TimeSpan.FromMinutes(1), http, candidates, TimeSpan.FromSeconds(5), cts1.Token);

            await probeEntered.Task;
            Task<NpuSidecarEndpoint?> waiter2 = NpuSidecarDiscovery.ProbeCachedAsync(
                TimeSpan.FromMinutes(1), http, candidates, TimeSpan.FromSeconds(5), CancellationToken.None);

            cts1.Cancel();
            releaseProbe.TrySetResult();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiter1);
            NpuSidecarEndpoint? result2 = await waiter2;

            Assert.NotNull(result2);
            Assert.Equal("http://127.0.0.1:19999", result2!.BaseUrl);
            Assert.Contains("priority4-npu", result2.Models);

            // Also verify empty {"data":[]} is handled cleanly with 0 models without null/index crashes
            using var emptyHandler = new StubDiscoveryHandler((_, _, _) =>
                Task.FromResult(Json(HttpStatusCode.OK, """{"data":[]}""")));
            using var emptyHttp = new HttpClient(emptyHandler);
            var emptyEp = await NpuSidecarDiscovery.ProbeSingleAsync(emptyHttp, "http://127.0.0.1:8000", NpuSidecarRuntime.LemonadeServer, TimeSpan.FromSeconds(2));
            Assert.NotNull(emptyEp);
            Assert.Empty(emptyEp!.Models);
        }
        finally
        {
            NpuSidecarDiscovery.InvalidateCache();
        }
    }

    [Fact]
    public async Task NpuSidecarDiscovery_NegativePerEndpointTimeout_ClampsWithoutUnhandledCrash()
    {
        NpuSidecarDiscovery.InvalidateCache();
        try
        {
            using var handler = new StubDiscoveryHandler((_, _, _) =>
                Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"id":"ok"}]}""")));
            using var http = new HttpClient(handler);
            var candidates = new (string, NpuSidecarRuntime)[]
            {
                ("http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM)
            };

            // Negative timeout must not throw ArgumentOutOfRangeException inside CancellationTokenSource.CancelAfter
            _ = await NpuSidecarDiscovery.ProbeAsync(http, candidates, TimeSpan.FromMilliseconds(-500));
            _ = await NpuSidecarDiscovery.ProbeSingleAsync(http, "http://127.0.0.1:52625", NpuSidecarRuntime.FastFlowLM, TimeSpan.FromSeconds(-1));
            _ = await NpuSidecarDiscovery.ProbeCachedAsync(TimeSpan.FromSeconds(1), http, candidates, TimeSpan.FromMilliseconds(-10));
        }
        finally
        {
            NpuSidecarDiscovery.InvalidateCache();
        }
    }

    [Fact]
    public void DeviceManager_ParseNpuHardwareId_EdgeCasesAndZeroRam_SafeAndAccurate()
    {
        Assert.Null(DeviceManager.ParseNpuHardwareId(null, 0, 0));
        Assert.Null(DeviceManager.ParseNpuHardwareId("   ", 16UL << 30, 0));
        Assert.Null(DeviceManager.ParseNpuHardwareId("PCI\\VEN_9999&DEV_0000", 16UL << 30, 0));

        // AMD XDNA 2 (DEV_17F0) with 0 RAM -> 0 SharedVramBytes without overflow/crash
        var xdna2Zero = DeviceManager.ParseNpuHardwareId("PCI\\VEN_1022&DEV_17F0&SUBSYS_0000", 0UL, 0);
        Assert.NotNull(xdna2Zero);
        Assert.True(xdna2Zero!.IsNpu);
        Assert.Equal(0UL, xdna2Zero.SharedVramBytes);

        // AMD XDNA 1 (DEV_1502) and Intel AI Boost (VEN_8086&DEV_7D1D)
        var xdna1 = DeviceManager.ParseNpuHardwareId("PCI\\VEN_1022&DEV_1502", 32UL << 30, 1);
        Assert.NotNull(xdna1);
        Assert.True(xdna1!.IsNpu);
        Assert.Equal(16UL << 30, xdna1.SharedVramBytes);

        var intel = DeviceManager.ParseNpuHardwareId("PCI\\VEN_8086&DEV_AD1D", 32UL << 30, 2);
        Assert.NotNull(intel);
        Assert.True(intel!.IsNpu);
        Assert.Equal(GpuVendor.Intel, intel.Vendor);

        // Smoke-check GetNpuDevices enumeration does not throw
        var npus = DeviceManager.GetNpuDevices();
        Assert.NotNull(npus);
    }

    [Theory]
    [InlineData("http://127.0.0.1:52625/v1?api-version=1", "http://127.0.0.1:52625?api-version=1", "http://127.0.0.1:52625/v1/models?api-version=1")]
    [InlineData("http://127.0.0.1:52625?api-version=1", "http://127.0.0.1:52625?api-version=1", "http://127.0.0.1:52625/v1/models?api-version=1")]
    [InlineData("http://127.0.0.1:52625/v1/?api-version=1#frag", "http://127.0.0.1:52625?api-version=1#frag", "http://127.0.0.1:52625/v1/models?api-version=1#frag")]
    public async Task NpuSidecarDiscovery_QueryAndFragmentEndpoints_ProbesModelsBeforeQueryAndPreservesSuffix(
        string inputUrl,
        string expectedNormalizedBaseUrl,
        string expectedProbedUrl)
    {
        Uri? capturedUri = null;
        using var handler = new StubDiscoveryHandler((req, _, _) =>
        {
            capturedUri = req.RequestUri;
            return Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"id":"qwen3-npu"}]}"""));
        });
        using var http = new HttpClient(handler);

        NpuSidecarEndpoint? endpoint = await NpuSidecarDiscovery.ProbeSingleAsync(
            http,
            inputUrl,
            NpuSidecarRuntime.FastFlowLM,
            TimeSpan.FromSeconds(2));

        Assert.NotNull(endpoint);
        Assert.Equal(expectedNormalizedBaseUrl, endpoint!.BaseUrl);
        Assert.Equal("qwen3-npu", endpoint.PrimaryModel);
        Assert.Equal(expectedProbedUrl, capturedUri?.ToString());
    }
}
