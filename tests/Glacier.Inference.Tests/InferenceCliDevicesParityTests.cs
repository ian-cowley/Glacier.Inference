namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Cli;
using Glacier.Inference.Hardware;
using Xunit;

public sealed class InferenceCliDevicesParityTests
{
    private static DeviceInfo MakeCpu() => new()
    {
        Id = "cpu",
        Index = 99,
        Name = "AMD Ryzen AI 9 HX 370",
        Vendor = GpuVendor.Cpu,
        DedicatedVramBytes = 0,
        SharedVramBytes = 64UL << 30,
        IsDisplayDevice = false,
        IsNpu = false,
        SupportedEngines = [InferenceEngineType.Cpu],
        RecommendedEngine = InferenceEngineType.Cpu,
        SafetyNotes = "Host CPU AVX2/AVX-512"
    };

    private static DeviceInfo MakeIntegratedGpu() => new()
    {
        Id = "amd-890m",
        Index = 0,
        Name = "AMD Radeon 890M Graphics",
        Vendor = GpuVendor.Amd,
        DedicatedVramBytes = 512UL << 20,
        SharedVramBytes = 16UL << 30,
        IsDisplayDevice = true,
        IsNpu = false,
        SupportedEngines = [InferenceEngineType.Vulkan, InferenceEngineType.DirectML],
        RecommendedEngine = InferenceEngineType.Vulkan,
        SafetyNotes = "Vulkan CoopMat / WMMA"
    };

    private static DeviceInfo MakeDiscreteGpu() => new()
    {
        Id = "nvidia-rtx-4060",
        Index = 1,
        Name = "NVIDIA GeForce RTX 4060",
        Vendor = GpuVendor.Nvidia,
        DedicatedVramBytes = 8UL << 30,
        SharedVramBytes = 16UL << 30,
        IsDisplayDevice = true,
        IsNpu = false,
        SupportedEngines = [InferenceEngineType.BareMetal, InferenceEngineType.Vulkan],
        RecommendedEngine = InferenceEngineType.BareMetal,
        SafetyNotes = "Native CUDA SASS"
    };

    private static DeviceInfo MakeNpu() => new()
    {
        Id = "amd-xdna2",
        Index = 2,
        Name = "AMD Ryzen AI NPU (XDNA 2)",
        Vendor = GpuVendor.Amd,
        DedicatedVramBytes = 0,
        SharedVramBytes = (ulong)(15.5 * 1024 * 1024 * 1024),
        IsDisplayDevice = false,
        IsNpu = true,
        SupportedEngines = [InferenceEngineType.NpuSidecar],
        RecommendedEngine = InferenceEngineType.NpuSidecar,
        SafetyNotes = "AMD XDNA 2 NPU via OpenAI-compatible local NPU sidecar."
    };

    [Fact]
    public void FormatDeviceMemory_FormatsCpuNpuUnifiedAndDiscreteVram()
    {
        Assert.Equal("64.0 GB RAM", Program.FormatDeviceMemory(MakeCpu()));
        Assert.Equal("15.5 GB NPU Cap", Program.FormatDeviceMemory(MakeNpu()));
        Assert.Equal("16.0 GB Unified", Program.FormatDeviceMemory(MakeIntegratedGpu()));
        Assert.Equal("8.0 GB VRAM", Program.FormatDeviceMemory(MakeDiscreteGpu()));
    }

    [Fact]
    public void FormatEngineName_FormatsNpuSidecarAndExistingEngines()
    {
        Assert.Equal("NPU Sidecar (FLM / Lemonade)", Program.FormatEngineName(InferenceEngineType.NpuSidecar));
        Assert.Equal("Native Driver (SASS/HIP)", Program.FormatEngineName(InferenceEngineType.BareMetal));
        Assert.Equal("Vulkan (CoopMat / WMMA)", Program.FormatEngineName(InferenceEngineType.Vulkan));
        Assert.Equal("DirectML", Program.FormatEngineName(InferenceEngineType.DirectML));
        Assert.Equal("Cpu", Program.FormatEngineName(InferenceEngineType.Cpu));
    }

    [Fact]
    public void WriteDevicesReport_WithNpuDevice_EmitsCapEngineAndSidecarHints()
    {
        var dgpu = MakeDiscreteGpu();
        var npu = MakeNpu();
        var cpu = MakeCpu();

        using var writer = new StringWriter();
        Program.WriteDevicesReport([dgpu, npu, cpu], dgpu, InferenceEngineType.BareMetal, writer, useColors: false);
        string text = writer.ToString();

        Assert.Contains("15.5 GB NPU Cap", text, StringComparison.Ordinal);
        Assert.Contains("NPU Sidecar (FLM / Lemonade)", text, StringComparison.Ordinal);
        Assert.Contains("[ACTIVE: Native Driver (SASS/HIP)]", text, StringComparison.Ordinal);
        Assert.Contains("* NPU Sidecar detected: run 'flm serve <model>' (FastFlowLM) or 'lemonade-server' and offload critic/RAG:", text, StringComparison.Ordinal);
        Assert.Contains("glacier bounce --primary http://127.0.0.1:11434 --critic auto-npu", text, StringComparison.Ordinal);
    }

    [Fact]
    public void WriteDevicesReport_WithoutNpuDevice_OmitsNpuSidecarHint()
    {
        var dgpu = MakeDiscreteGpu();
        var cpu = MakeCpu();

        using var writer = new StringWriter();
        Program.WriteDevicesReport([dgpu, cpu], dgpu, InferenceEngineType.BareMetal, writer, useColors: false);
        string text = writer.ToString();

        Assert.DoesNotContain("NPU Cap", text, StringComparison.Ordinal);
        Assert.DoesNotContain("NPU Sidecar detected", text, StringComparison.Ordinal);
        Assert.DoesNotContain("flm serve", text, StringComparison.Ordinal);
    }
}
