namespace Glacier.Inference.Tests;

using System;
using System.Runtime.InteropServices;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Vortice.Direct3D12;
using Xunit;

public sealed class D3D12FactAttribute : FactAttribute
{
    public D3D12FactAttribute()
    {
        if (!OperatingSystem.IsWindows())
            Skip = "Direct3D 12 is Windows-only.";
        else if (!D3D12Context.IsSupported)
            Skip = "No Direct3D 12 hardware adapter available.";
    }
}

[Trait("Hardware", "Gpu")]
[Collection("SequentialGpu")]
public unsafe class HardeningM34GpuTests
{
    [VulkanFact]
    public void VulkanCompute_DescriptorPoolChaining_And_GBufZero_WorkAndDisposeCleanly()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        // Verify GBuf.Zero() clears both host-visible and device-only buffers cleanly
        var hostBuf = vk.Alloc(256);
        float[] pattern = new float[64];
        Array.Fill(pattern, 42.5f);
        hostBuf.Upload(pattern);

        hostBuf.Zero();

        float[] readBack = new float[64];
        fixed (float* pOut = readBack)
            hostBuf.DownloadTo(pOut, 256);
        foreach (float val in readBack)
            Assert.Equal(0.0f, val);

        // Also test GBuf.Zero() inside an active command recording + on AllocDeviceOnly
        var devOnly = vk.AllocDeviceOnly(256);
        vk.Begin();
        devOnly.Zero();
        vk.SubmitAndWait();

        // Exhaust initial 8192 descriptor pool and verify pool chaining grows without exception
        var kernel = vk.LoadEmbedded("qmatvec_q3", 3, 8);
        Assert.True(vk.KernelCount >= 1);
        Assert.Equal(1, vk.PoolCount);

        for (int i = 0; i < 8210; i++)
        {
            _ = vk.MakeSet(kernel, hostBuf, hostBuf, hostBuf);
        }
        Assert.True(vk.PoolCount >= 2, $"Expected descriptor pool chaining (>=2 pools), actual PoolCount={vk.PoolCount}");

        vk.Dispose();
        Assert.Equal(0, vk.PoolCount);
        Assert.Equal(0, vk.KernelCount);
    }

    [VulkanFact]
    public void VulkanCompute_SubmitAndWaitWithoutBegin_ThrowsInvalidOperationException()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        Assert.Throws<InvalidOperationException>(() => vk.SubmitAndWait());

        var buf = vk.Alloc(64);
        float[] pattern = new float[16];
        Array.Fill(pattern, 7.0f);
        buf.Upload(pattern);

        vk.Begin();
        buf.Zero();
        vk.SubmitAndWait();

        float[] readBack = new float[16];
        fixed (float* pOut = readBack)
            buf.DownloadTo(pOut, 64);
        Assert.All(readBack, v => Assert.Equal(0.0f, v));
    }

    [VulkanFact]
    public void VulkanCompute_ZeroBufferDuringRecording_PreservesUnrelatedWriteHazards()
    {
        const int N = 64;
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);

        var kSwi = vk.LoadEmbedded("elementwise_swiglu", 3, 4, writeBindingMask: 1u << 2);
        var kCopy = vk.LoadEmbedded("copy_f32", 2, 12, writeBindingMask: 1u << 1);
        var kAdd = vk.LoadEmbedded("elementwise_add", 3, 4, writeBindingMask: 1u << 0);

        float[] ones = new float[N];
        float[] twos = new float[N];
        float[] sentinel = new float[N];
        Array.Fill(ones, 1.0f);
        Array.Fill(twos, 2.0f);
        Array.Fill(sentinel, -999.0f);

        var bufOne = vk.Alloc(N * 4L); bufOne.Upload(ones);
        var bufTwo = vk.Alloc(N * 4L); bufTwo.Upload(twos);
        var bufA = vk.Alloc(N * 4L); bufA.Upload(sentinel);
        var bufB = vk.Alloc(N * 4L); bufB.Upload(sentinel);
        var bufOut = vk.Alloc(N * 4L); bufOut.Upload(sentinel);
        var bufOutB = vk.Alloc(N * 4L); bufOutB.Upload(sentinel);

        IntPtr setWriteA = vk.MakeSet(kSwi, bufOne, bufTwo, bufA);
        IntPtr setWriteB = vk.MakeSet(kSwi, bufOne, bufTwo, bufB);
        IntPtr setCopyA = vk.MakeSet(kCopy, bufA, bufOut);
        IntPtr setCopyB = vk.MakeSet(kCopy, bufB, bufOutB);
        IntPtr setAddB = vk.MakeSet(kAdd, bufOut, bufB, bufB);

        uint gx = (uint)((N + 255) / 256);
        uint[] nPush = [(uint)N];
        uint[] copyPush = [(uint)N, 0u, 0u];

        vk.Begin();
        // 1. Write buffer A via compute shader
        vk.Dispatch(kSwi, setWriteA, nPush, gx);
        // 2. Write buffer B via compute shader
        vk.Dispatch(kSwi, setWriteB, nPush, gx);
        // 3. Re-zero buffer B mid-recording (must preserve A's pending write-hazard tracking!)
        bufB.Zero();
        // 4. Read A (ReadIds-only) into bufOut, read B into bufOutB, and accumulate B into bufOut
        vk.Dispatch(kCopy, setCopyA, copyPush, gx);
        vk.Dispatch(kCopy, setCopyB, copyPush, gx);
        vk.Dispatch(kAdd, setAddB, nPush, gx);
        vk.SubmitAndWait();

        float expectedSwiglu = (1.0f / (1.0f + MathF.Exp(-1.0f))) * 2.0f;
        float[] outA = new float[N];
        float[] outB = new float[N];
        float[] outSum = new float[N];
        float[] outCopyB = new float[N];
        fixed (float* pA = outA, pB = outB, pSum = outSum, pCopyB = outCopyB)
        {
            bufA.DownloadTo(pA, N * 4L);
            bufB.DownloadTo(pB, N * 4L);
            bufOut.DownloadTo(pSum, N * 4L);
            bufOutB.DownloadTo(pCopyB, N * 4L);
        }

        for (int i = 0; i < N; i++)
        {
            Assert.Equal(expectedSwiglu, outA[i], 0.0001f);
            Assert.Equal(0.0f, outB[i]);
            Assert.Equal(0.0f, outCopyB[i]);
            Assert.Equal(expectedSwiglu, outSum[i], 0.0001f);
        }
        Assert.True(vk.LastBarrierCount >= 4, $"Expected pre/post Zero barriers + compute/host barriers, got {vk.LastBarrierCount}");
    }

    [D3D12Fact]
    public void D3D12Context_UmaCheckAndPooledReadback_SucceedAcrossGrowingBuffers()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var d3d = new D3D12Context();
        Assert.True(d3d.TotalAccessibleMemory > 0);
        // Accessing IsUnifiedMemoryArchitecture verifies Feature.Architecture1 query succeeded
        _ = d3d.IsUnifiedMemoryArchitecture;

        ulong[] sizes = [256UL, 64UL * 1024UL, 512UL * 1024UL];
        foreach (ulong sz in sizes)
        {
            using var devBuf = d3d.CreateDeviceBuffer(sz);
            byte[] src = new byte[sz];
            for (int i = 0; i < src.Length; i++) src[i] = (byte)((i * 13 + 7) & 0xFF);
            d3d.CopyToDevice(devBuf, src);

            byte[] dst = new byte[sz];
            d3d.CopyToHost(dst, devBuf, sz);
            Assert.Equal(src[0], dst[0]);
            Assert.Equal(src[^1], dst[^1]);

            d3d.ZeroDeviceBuffer(devBuf, ResourceStates.Common);
            d3d.CopyToHost(dst, devBuf, sz);
            Assert.Equal(0, dst[0]);
            Assert.Equal(0, dst[^1]);
        }
    }

    [Fact]
    public void VulkanContext_OutOfRangeDeviceOrdinal_ThrowsCleanlyWithoutNativeCrash()
    {
        if (!VulkanContext.IsSupported) return;
        Assert.ThrowsAny<Exception>(() => new VulkanContext(99999));
    }
}
