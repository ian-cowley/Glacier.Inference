namespace Glacier.Inference.Tests;

using System;
using System.Linq;
using Glacier.Inference.Config;
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

            var dev2 = DeviceManager.ResolveDevice("4060");
            Assert.Equal(GpuVendor.Nvidia, dev2.Vendor);
        }

        if (devices.Any(d => d.Vendor == GpuVendor.Amd))
        {
            var dev = DeviceManager.ResolveDevice("amd");
            Assert.Equal(GpuVendor.Amd, dev.Vendor);

            var dev2 = DeviceManager.ResolveDevice("890m");
            Assert.Equal(GpuVendor.Amd, dev2.Vendor);
        }
    }

    [Fact]
    public void ResolveDevice_InvalidName_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => DeviceManager.ResolveDevice("nonexistent_quantum_chip_9999"));
    }

    [Fact]
    public void SafeEngineMatrix_AmdRejectsCuda()
    {
        var devices = DeviceManager.GetDevices();
        var amd = devices.FirstOrDefault(d => d.Vendor == GpuVendor.Amd);

        if (amd != null)
        {
            Assert.Contains(InferenceEngineType.DirectML, amd.SupportedEngines);
            Assert.Contains(InferenceEngineType.Cpu, amd.SupportedEngines);
            Assert.DoesNotContain(InferenceEngineType.Cuda, amd.SupportedEngines);

            bool safe = GlacierSettings.ValidateSafety(amd, InferenceEngineType.Cuda, out string? error);
            Assert.False(safe);
            Assert.NotNull(error);
            Assert.Contains("Safe engines for this device", error);
        }
    }

    [Fact]
    public void SafeEngineMatrix_NvidiaAllowsCudaAndDirectML()
    {
        var devices = DeviceManager.GetDevices();
        var nvidia = devices.FirstOrDefault(d => d.Vendor == GpuVendor.Nvidia);

        if (nvidia != null)
        {
            Assert.Contains(InferenceEngineType.Cuda, nvidia.SupportedEngines);
            Assert.Contains(InferenceEngineType.DirectML, nvidia.SupportedEngines);
            Assert.Contains(InferenceEngineType.Cpu, nvidia.SupportedEngines);

            bool safeCuda = GlacierSettings.ValidateSafety(nvidia, InferenceEngineType.Cuda, out string? errCuda);
            Assert.True(safeCuda);
            Assert.Null(errCuda);

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
    public void ResolveTarget_UnsafeTarget_ThrowsInvalidOperationException()
    {
        var devices = DeviceManager.GetDevices();
        if (devices.Any(d => d.Vendor == GpuVendor.Amd))
        {
            Assert.Throws<InvalidOperationException>(() => GlacierSettings.ResolveTarget("amd", "cuda"));
        }
    }
}
