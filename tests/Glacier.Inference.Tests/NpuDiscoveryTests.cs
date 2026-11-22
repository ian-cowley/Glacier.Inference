namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Config;
using Glacier.Inference.Hardware;
using Xunit;

[Collection("SequentialGpu")]
public sealed class NpuDiscoveryTests
{
    [Fact]
    public void ParseNpuHardwareId_AmdXdna2_32GbRam_Reflects50PercentCapAndGuidance()
    {
        ulong ram32Gb = 32UL * 1024 * 1024 * 1024;
        var dev = DeviceManager.ParseNpuHardwareId(
            @"PCI\VEN_1022&DEV_17F0&SUBSYS_20CF1043&REV_10\4&23445453&0&0142",
            ram32Gb,
            index: 2);

        Assert.NotNull(dev);
        Assert.Equal("amd-xdna2", dev.Id);
        Assert.True(dev.IsNpu);
        Assert.Equal(GpuVendor.Amd, dev.Vendor);
        Assert.Equal(16UL * 1024 * 1024 * 1024, dev.SharedVramBytes);
        Assert.Equal(InferenceEngineType.NpuSidecar, dev.RecommendedEngine);
        Assert.Contains(InferenceEngineType.NpuSidecar, dev.SupportedEngines);
        Assert.Contains(">=64 GB RAM", dev.SafetyNotes, StringComparison.Ordinal);
    }

    [Fact]
    public void ParseNpuHardwareId_AmdXdna2_64GbRam_Allows35BMoeGuidance()
    {
        ulong ram64Gb = 64UL * 1024 * 1024 * 1024;
        var dev = DeviceManager.ParseNpuHardwareId(@"PCI\VEN_1022&DEV_17F0", ram64Gb, index: 1);

        Assert.NotNull(dev);
        Assert.Equal("amd-xdna2", dev.Id);
        Assert.True(dev.IsNpu);
        Assert.Equal(32UL * 1024 * 1024 * 1024, dev.SharedVramBytes);
        Assert.Contains("35B sparse MoE", dev.SafetyNotes, StringComparison.Ordinal);
    }

    [Fact]
    public void ParseNpuHardwareId_AmdXdna1AndIntelNpu_ParsedCleanly()
    {
        ulong ram32Gb = 32UL * 1024 * 1024 * 1024;

        var xdna1 = DeviceManager.ParseNpuHardwareId(@"PCI\VEN_1022&DEV_1502", ram32Gb, index: 1);
        Assert.NotNull(xdna1);
        Assert.Equal("amd-xdna1", xdna1.Id);
        Assert.True(xdna1.IsNpu);

        var intel = DeviceManager.ParseNpuHardwareId(@"PCI\VEN_8086&DEV_7D1D", ram32Gb, index: 2);
        Assert.NotNull(intel);
        Assert.Equal("intel-npu", intel.Id);
        Assert.Equal(GpuVendor.Intel, intel.Vendor);
        Assert.True(intel.IsNpu);

        Assert.Null(DeviceManager.ParseNpuHardwareId(@"PCI\VEN_10DE&DEV_28E0", ram32Gb, index: 0));
        Assert.Null(DeviceManager.ParseNpuHardwareId(null, ram32Gb, index: 0));
    }

    [Fact]
    public void GetOptimalDevice_NeverSelectsNpuForInProcessGguf()
    {
        var optimal = DeviceManager.GetOptimalDevice();
        Assert.False(optimal.IsNpu);
    }

    [Fact]
    public void ResolveDevice_WhenNpuPresent_ResolvesByKeywordsAndValidatesNpuSidecarEngine()
    {
        var npus = DeviceManager.GetNpuDevices();
        if (npus.Count == 0)
            return;

        var resolved = DeviceManager.ResolveDevice("npu");
        Assert.True(resolved.IsNpu);

        var (targetDev, targetEng) = GlacierSettings.ResolveTarget("npu", "npusidecar");
        Assert.True(targetDev.IsNpu);
        Assert.Equal(InferenceEngineType.NpuSidecar, targetEng);
    }

    [Fact]
    public async Task NpuSidecarDiscovery_ProbeAsync_DiscoversLoopbackFastFlowLmServer()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Task serverTask = Task.Run(async () =>
        {
            using var client = await listener.AcceptTcpClientAsync(cts.Token);
            using var stream = client.GetStream();
            byte[] reqBuf = new byte[1024];
            _ = await stream.ReadAsync(reqBuf, cts.Token);

            const string json = """{"object":"list","data":[{"id":"qwen3.6-35b-a3b-npu2"},{"id":"embeddinggemma"}]}""";
            string httpResp = $"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {Encoding.UTF8.GetByteCount(json)}\r\nConnection: close\r\n\r\n{json}";
            byte[] respBytes = Encoding.UTF8.GetBytes(httpResp);
            await stream.WriteAsync(respBytes, cts.Token);
            await stream.FlushAsync(cts.Token);
        }, cts.Token);

        using var http = new HttpClient();
        var candidates = new List<(string BaseUrl, NpuSidecarRuntime Runtime)>
        {
            ($"http://127.0.0.1:{port}", NpuSidecarRuntime.FastFlowLM)
        };

        var found = await NpuSidecarDiscovery.ProbeAsync(
            http,
            candidates,
            perEndpointTimeout: TimeSpan.FromSeconds(2),
            cancellationToken: cts.Token);

        await serverTask;

        Assert.NotNull(found);
        Assert.Equal($"http://127.0.0.1:{port}", found.BaseUrl);
        Assert.Equal(NpuSidecarRuntime.FastFlowLM, found.Runtime);
        Assert.Equal("qwen3.6-35b-a3b-npu2", found.PrimaryModel);
        Assert.Contains("embeddinggemma", found.Models);
    }

    [Fact]
    public async Task NpuSidecarDiscovery_ProbeAsync_ReturnsNullWhenUnreachable()
    {
        // Grab an unused ephemeral port then immediately close it so connection is refused cleanly
        int unusedPort;
        using (var probe = new TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            unusedPort = ((IPEndPoint)probe.LocalEndpoint).Port;
        }

        using var http = new HttpClient();
        var found = await NpuSidecarDiscovery.ProbeAsync(
            http,
            [($"http://127.0.0.1:{unusedPort}", NpuSidecarRuntime.FastFlowLM)],
            perEndpointTimeout: TimeSpan.FromMilliseconds(250));

        Assert.Null(found);
    }
}
