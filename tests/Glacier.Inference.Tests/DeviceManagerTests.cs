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

            var dev3 = DeviceManager.ResolveDevice("baremetal");
            Assert.Equal(GpuVendor.Nvidia, dev3.Vendor);

            var dev4 = DeviceManager.ResolveDevice("sass");
            Assert.Equal(GpuVendor.Nvidia, dev4.Vendor);
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
}
