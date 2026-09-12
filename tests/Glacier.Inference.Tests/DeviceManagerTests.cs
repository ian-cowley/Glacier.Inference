namespace Glacier.Inference.Tests;

using System;
using System.Linq;
using Glacier.Inference.Config;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Xunit;

public class DeviceManagerTests
{
    [Fact]
    public void GetDevices_ReturnsAtLeastHostCpu()
    {
        var devices = DeviceManager.GetDevices();
        Assert.NotEmpty(devices);

        var cpu = devices.FirstOrDefault(d => d.Vendor == GpuVendor.Cpu);
        Assert.NotNull(cpu);
        Assert.Equal("cpu", cpu.Id);
        Assert.Contains(InferenceEngineType.Cpu, cpu.SupportedEngines);
    }

    [Fact]
    public void ResolveDevice_Auto_ResolvesOptimalDevice()
    {
        var device = DeviceManager.ResolveDevice("auto");
        Assert.NotNull(device);

        var optimal = DeviceManager.GetOptimalDevice();
        Assert.Equal(optimal.Id, device.Id);
    }

    [Fact]
    public void ResolveDevice_CpuKeyword_ResolvesCpu()
    {
        var device = DeviceManager.ResolveDevice("cpu");
        Assert.NotNull(device);
        Assert.Equal(GpuVendor.Cpu, device.Vendor);
    }

    [Fact]
    public void ResolveDevice_FuzzyMatching_FindsExpectedHardware()
    {
        var devices = DeviceManager.GetDevices();

        if (devices.Any(d => d.Vendor == GpuVendor.Nvidia))
        {
            var dev = DeviceManager.ResolveDevice("nvidia");
            Assert.Equal(GpuVendor.Nvidia, dev.Vendor);

            var nDev = devices.First(d => d.Vendor == GpuVendor.Nvidia);
            string snippet = nDev.Name.Contains("RTX", StringComparison.OrdinalIgnoreCase) ? "RTX" : "nvidia";
            var dev2 = DeviceManager.ResolveDevice(snippet);
            Assert.Equal(GpuVendor.Nvidia, dev2.Vendor);

            var dev3 = DeviceManager.ResolveDevice("baremetal");
            Assert.Equal(GpuVendor.Nvidia, dev3.Vendor);

            var dev4 = DeviceManager.ResolveDevice("sass");
            Assert.Equal(GpuVendor.Nvidia, dev4.Vendor);
        }

        if (devices.Any(d => d.Vendor == GpuVendor.Amd))
        {
            var dev = DeviceManager.ResolveDevice("amd");
            Assert.Equal(GpuVendor.Amd, dev.Vendor);

            var aDev = devices.First(d => d.Vendor == GpuVendor.Amd);
            string snippet = aDev.Name.Contains("Radeon", StringComparison.OrdinalIgnoreCase) ? "Radeon" : "amd";
            var dev2 = DeviceManager.ResolveDevice(snippet);
            Assert.Equal(GpuVendor.Amd, dev2.Vendor);
        }

        if (devices.Any(d => d.Vendor == GpuVendor.Intel))
        {
            var dev = DeviceManager.ResolveDevice("intel");
            Assert.Equal(GpuVendor.Intel, dev.Vendor);
        }
    }

    [Fact]
    public void ResolveDevice_InvalidName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => DeviceManager.ResolveDevice("nonexistent_quantum_chip_9999"));
    }

    [Fact]
    public void SafeEngineMatrix_AmdRejectsBareMetal()
    {
        var devices = DeviceManager.GetDevices();
        var amd = devices.FirstOrDefault(d => d.Vendor == GpuVendor.Amd);

        if (amd != null)
        {
            Assert.Contains(InferenceEngineType.DirectML, amd.SupportedEngines);
            Assert.DoesNotContain(InferenceEngineType.BareMetal, amd.SupportedEngines);
            Assert.DoesNotContain(InferenceEngineType.Cpu, amd.SupportedEngines);

            bool safe = GlacierSettings.ValidateSafety(amd, InferenceEngineType.BareMetal, out string? error);
            Assert.False(safe);
            Assert.NotNull(error);
            Assert.Contains("Safe engines for this device", error);
        }
    }

    [Fact]
    public void SafeEngineMatrix_NvidiaAllowsBareMetalAndDirectML()
    {
        var devices = DeviceManager.GetDevices();
        var nvidia = devices.FirstOrDefault(d => d.Vendor == GpuVendor.Nvidia);

        if (nvidia != null)
        {
            Assert.Contains(InferenceEngineType.BareMetal, nvidia.SupportedEngines);
            Assert.Contains(InferenceEngineType.DirectML, nvidia.SupportedEngines);
            Assert.DoesNotContain(InferenceEngineType.Cpu, nvidia.SupportedEngines);

            bool safeBm = GlacierSettings.ValidateSafety(nvidia, InferenceEngineType.BareMetal, out string? errBm);
            Assert.True(safeBm);
            Assert.Null(errBm);

            bool safeDml = GlacierSettings.ValidateSafety(nvidia, InferenceEngineType.DirectML, out string? errDml);
            Assert.True(safeDml);
            Assert.Null(errDml);
        }
    }

    [Fact]
    public void ResolveTarget_ValidTarget_Succeeds()
    {
        var (device, engine) = GlacierSettings.ResolveTarget("cpu", "cpu");
        Assert.Equal(GpuVendor.Cpu, device.Vendor);
        Assert.Equal(InferenceEngineType.Cpu, engine);
    }

    [Fact]
    public void ResolveTarget_AliasesForBareMetal_AllResolveCorrectly()
    {
        var devices = DeviceManager.GetDevices();
        if (devices.Any(d => d.Vendor == GpuVendor.Nvidia))
        {
            var (_, e1) = GlacierSettings.ResolveTarget("nvidia", "baremetal");
            Assert.Equal(InferenceEngineType.BareMetal, e1);

            var (_, e2) = GlacierSettings.ResolveTarget("nvidia", "sass");
            Assert.Equal(InferenceEngineType.BareMetal, e2);

            var (_, e3) = GlacierSettings.ResolveTarget("nvidia", "cuda");
            Assert.Equal(InferenceEngineType.BareMetal, e3);
        }
    }

    [Fact]
    public void ResolveTarget_UnsafeTarget_ThrowsInvalidOperationException()
    {
        var devices = DeviceManager.GetDevices();
        if (devices.Any(d => d.Vendor == GpuVendor.Amd))
        {
            Assert.Throws<InvalidOperationException>(() => GlacierSettings.ResolveTarget("amd", "baremetal"));
        }
    }

    [Fact]
    public void TestGpuDiagnostics()
    {
        if (!GpuContext.IsSupported) return;
        using var gpu = new GpuContext();
        byte[] cubin = KernelCompiler.GetOrCompileKernels("sm_89");
        int res = CuDriver.ModuleLoadData(out IntPtr module, cubin);
        Assert.True(res == 0, $"ModuleLoadData failed with code: {res}");

        // Also test retrieving a kernel function
        int funcRes = CuDriver.ModuleGetFunction(out IntPtr fn, module, "gemv_q4_k_fast");
        Assert.True(funcRes == 0, $"ModuleGetFunction gemv_q4_k_fast failed with code: {funcRes}");
        Assert.NotEqual(IntPtr.Zero, fn);
    }

    [Fact]
    public void MultiVendorSupport_DirectMlAvailableOnWindows()
    {
        if (!OperatingSystem.IsWindows()) return;
        // Verify DirectML.dll and d3d12.dll exist in Windows System32 for universal AMD/Intel/NVIDIA GPU compute
        bool dml = System.IO.File.Exists(System.IO.Path.Combine(Environment.SystemDirectory, "DirectML.dll"));
        bool d3d = System.IO.File.Exists(System.IO.Path.Combine(Environment.SystemDirectory, "d3d12.dll"));
        Assert.True(dml, "Windows 10/11 DirectML.dll must be present in System32");
        Assert.True(d3d, "Windows 10/11 d3d12.dll must be present in System32");
    }

    [Fact]
    public void MultiVendorSupport_DeviceManagerRecognizesOptimalDevice()
    {
        var optimal = DeviceManager.GetOptimalDevice();
        Assert.NotNull(optimal);
        Assert.False(string.IsNullOrWhiteSpace(optimal.Name));
        Assert.False(string.IsNullOrWhiteSpace(optimal.Id));
        Assert.NotEmpty(optimal.SupportedEngines);
    }
}
