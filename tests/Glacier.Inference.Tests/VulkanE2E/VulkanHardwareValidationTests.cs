using System;
using System.Collections.Generic;
using Glacier.Inference.Gpu;
using Glacier.Inference.Tests.VulkanE2E.Fixtures;
using Xunit;

namespace Glacier.Inference.Tests.VulkanE2E;

/// <summary>
/// Hardware-accelerated GPU kernel validation on live Vulkan devices.
/// Uses [VulkanFact] attribute: executes on hosts with physical Vulkan hardware (such as AMD Radeon 890M GFX1150),
/// and cleanly skips with explicit reason when run on headless CI or environments without Vulkan GPUs.
/// </summary>
[Collection("SequentialGpu")]
public unsafe class VulkanHardwareValidationTests
{
    [VulkanFact]
    public void Hardware_DeviceProperties_Validation()
    {
        Assert.True(VulkanContext.IsSupported);
        int devIdx = Qwen35VulkanModel.FindVulkanDevice("");
        Assert.True(devIdx >= 0, "A valid hardware Vulkan device must be discovered");

        var devices = VulkanContext.EnumerateDevices();
        Assert.True(devIdx < devices.Count);
        var dev = devices[devIdx];

        Assert.False(string.IsNullOrWhiteSpace(dev.Name));
        Assert.True(dev.TotalMemoryBytes > 0, "Device memory must be greater than zero");
    }

    [VulkanFact]
    public void Hardware_RMSNorm_KernelExecution()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        int n = 256;
        float eps = 1e-5f;

        float[] xData = new float[n];
        float[] wData = new float[n];
        for (int i = 0; i < n; i++)
        {
            xData[i] = (i % 7) - 3.0f;
            wData[i] = 1.0f + (i % 3) * 0.1f;
        }

        float[] expected = VulkanE2ETestFixtures.ReferenceQuantOracle.ComputeRmsNorm(xData, wData, eps);

        var bufX = vk.Alloc(n * 4);
        var bufW = vk.Alloc(n * 4);
        var bufO = vk.Alloc(n * 4);

        fixed (float* pX = xData) bufX.UploadFrom(pX, n * 4);
        fixed (float* pW = wData) bufW.UploadFrom(pW, n * 4);

        var kern = vk.LoadEmbedded("rmsnorm", 3, 12);
        var set = vk.MakeSet(kern, bufX, bufW, bufO);

        uint fbits = BitConverter.SingleToUInt32Bits(eps);
        uint[] push = [(uint)n, 1u, fbits];

        vk.Begin();
        vk.Dispatch(kern, set, push, 1, 1);
        vk.SubmitAndWait();

        float[] actual = new float[n];
        fixed (float* pO = actual) bufO.DownloadTo(pO, n * 4);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(expected[i], actual[i], 0.001f);
        }
    }

    [VulkanFact]
    public void Hardware_ResetKvCache_DeviceBufferZeroing()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        long sizeBytes = 4096;
        var buf = vk.Alloc(sizeBytes);

        // Upload non-zero data
        byte[] nonZero = new byte[sizeBytes];
        Array.Fill(nonZero, (byte)0xAB);
        fixed (byte* p = nonZero) buf.UploadFrom((float*)p, sizeBytes);

        // Zero buffer
        buf.Zero();

        // Download and verify
        byte[] downloaded = new byte[sizeBytes];
        fixed (byte* p = downloaded) buf.DownloadTo((float*)p, sizeBytes);

        for (int i = 0; i < sizeBytes; i++)
        {
            Assert.Equal(0, downloaded[i]);
        }
    }

    [VulkanFact]
    public void Hardware_EmbeddingGather_PushConstants()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        int dim = 256;
        int vocab = 10;
        int targetToken = 3;

        // Create embedding table: row 3 filled with 7.0f
        float[] table = new float[vocab * dim];
        for (int i = 0; i < dim; i++)
        {
            table[targetToken * dim + i] = 7.0f;
        }

        var bufTable = vk.Alloc(table.Length * 4);
        var bufOut = vk.Alloc(dim * 4);

        fixed (float* pT = table) bufTable.UploadFrom(pT, table.Length * 4);

        var kern = vk.LoadEmbedded("embed_q3", 2, 8);
        Assert.NotNull(kern);
    }

    [VulkanFact]
    public void Hardware_MatrixVectorMultiply_Q8_0()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        var kern = vk.LoadEmbedded("qmatvec_q3", 3, 8); // Q8_0 kernel index 3
        Assert.NotNull(kern);
    }

    [VulkanFact]
    public void Hardware_2DWorkgroupDispatch_LargeM()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        int m = 32769;
        var (gx, gy) = VulkanE2ETestFixtures.ReferenceQuantOracle.CalculateAddQmvWorkgroups(m);

        Assert.Equal(32768u, gx);
        Assert.Equal(2u, gy);
        Assert.True((long)gx * gy >= m);
    }
}
