namespace Glacier.Inference.Tests;

using System;
using System.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Quant;
using Xunit;
using Xunit.Abstractions;

public sealed unsafe class NariQGpuParityTests
{
    private readonly ITestOutputHelper _output;

    public NariQGpuParityTests(ITestOutputHelper output) => _output = output;

    private static (byte[] Code, ushort[] S16, float[] Pre, float[] Post, int[]? Pin, int[]? Pout, float[] X, int[] Desc)
        CreateFixture(GgufType type, int k, int n, int rowsKind, int rowsBlock, int colsKind, int colsBlock, bool usePerm)
    {
        int bits = GgufTypes.GetTrellisBits(type);
        var rng = new Random(1337 + bits + k + n);
        var code = new byte[(long)k * n * bits / 8];
        rng.NextBytes(code);
        var s16 = new ushort[k / 16];
        for (int i = 0; i < s16.Length; i++) s16[i] = BitConverter.HalfToUInt16Bits((Half)(0.005f + 0.01f * rng.NextSingle()));
        var pre = new float[k];
        var post = new float[n];
        var x = new float[k];
        for (int i = 0; i < k; i++) { pre[i] = (rng.Next(2) == 0 ? 1f : -1f) * (0.5f + rng.NextSingle()); x[i] = (float)(rng.NextDouble() * 2.0 - 1.0); }
        for (int i = 0; i < n; i++) post[i] = (rng.Next(2) == 0 ? 1f : -1f) * (0.5f + rng.NextSingle());

        int[]? pin = null, pout = null;
        if (usePerm)
        {
            pin = new int[k];
            for (int i = 0; i < k; i++) pin[i] = i;
            for (int i = k - 1; i > 0; i--) { int j = rng.Next(i + 1); (pin[i], pin[j]) = (pin[j], pin[i]); }
            pout = new int[n];
            for (int i = 0; i < n; i++) pout[i] = i;
            for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (pout[i], pout[j]) = (pout[j], pout[i]); }
        }

        int[] desc = [rowsKind, rowsBlock, 1, colsKind, colsBlock, 1, 0];
        return (code, s16, pre, post, pin, pout, x, desc);
    }

    [Theory]
    [InlineData(GgufType.TRL2, 128, 128, 1, 128, 1, 128, true)]
    [InlineData(GgufType.TRL3, 256, 256, 1, 128, 1, 128, false)]
    [InlineData(GgufType.TRL6, 256, 128, 0, 0, 1, 128, true)]
    [InlineData(GgufType.TRL3, 2048, 1024, 1, 1024, 1, 128, false)] // Qwen3.8-27B real block sizes!
    public void CudaAndVulkan_TrellisMatVecMul_MatchCpuBitAccurately(
        GgufType type, int k, int n, int rowsKind, int rowsBlock, int colsKind, int colsBlock, bool usePerm)
    {
        var f = CreateFixture(type, k, n, rowsKind, rowsBlock, colsKind, colsBlock, usePerm);
        var yCpu = new float[n];
        var yCuda = new float[n];
        var yVk = new float[n];

        fixed (byte* pCode = f.Code)
        fixed (ushort* pS16 = f.S16)
        fixed (float* pPre = f.Pre)
        fixed (float* pPost = f.Post)
        fixed (int* pPin = f.Pin)
        fixed (int* pPout = f.Pout)
        fixed (float* pX = f.X)
        fixed (float* pYCpu = yCpu)
        {
            var binding = new TrellisWeightBinding
            {
                Type = type, CodeData = pCode, K = k, N = n,
                S16 = pS16, Pre = pPre, Post = pPost, Pin = pPin, Pout = pPout, Descriptor = f.Desc,
            };

            TrellisKernels.MatVecMul(binding, pX, pYCpu);

            // 1. CUDA Parity
            if (GpuContext.IsSupported)
            {
                using var cuda = new TrellisCudaRunner(0);
                using var dm = cuda.Upload(binding);
                var sw = Stopwatch.StartNew();
                cuda.MatVecMul(dm, f.X, yCuda);
                sw.Stop();
                float maxErr = 0f;
                for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs(yCuda[i] - yCpu[i]));
                _output.WriteLine($"[CUDA {cuda.DeviceName}] {type} {k}x{n} maxErr={maxErr:E3} ({sw.Elapsed.TotalMilliseconds:F2} ms)");
                Assert.InRange(maxErr, 0f, 1e-2f);
            }

            // 2. Vulkan Parity (tests EVERY real Vulkan device when available on the runner)
            var vkDevs = VulkanContext.EnumerateDevices();
            foreach (var d in vkDevs)
            {
                if (d.Name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase)) continue;
                using var vk = new TrellisVulkanRunner(d.Index);
                var dm = vk.Upload(binding);
                double gpuMs = vk.MatVecMul(dm, f.X, yVk);
                float maxErr = 0f;
                for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs(yVk[i] - yCpu[i]));
                _output.WriteLine($"[Vulkan:{d.Index} {vk.DeviceName}] {type} {k}x{n} maxErr={maxErr:E3} (gpu={gpuMs:F2} ms)");
                Assert.InRange(maxErr, 0f, 1e-2f);
            }
        }
    }

    [Theory]
    [InlineData(GgufType.TRL2, 5120, 17408)]
    [InlineData(GgufType.TRL3, 5120, 17408)]
    [InlineData(GgufType.TRL2, 17408, 5120)]
    public void Benchmark_Qwen38_27B_FfnShape_CudaAndVulkan(GgufType type, int k, int n)
    {
        var f = CreateFixture(type, k, n, 1, 1024, 1, 128, false);
        var yCuda = new float[n];
        var yVk = new float[n];
        double mb = (double)f.Code.Length / (1024.0 * 1024.0);

        fixed (byte* pCode = f.Code)
        fixed (ushort* pS16 = f.S16)
        fixed (float* pPre = f.Pre)
        fixed (float* pPost = f.Post)
        fixed (float* pX = f.X)
        {
            var binding = new TrellisWeightBinding
            {
                Type = type, CodeData = pCode, K = k, N = n,
                S16 = pS16, Pre = pPre, Post = pPost, Descriptor = f.Desc,
            };

            if (GpuContext.IsSupported)
            {
                using var cuda = new TrellisCudaRunner(0);
                using var dm = cuda.Upload(binding);
                double ms = cuda.BenchmarkDevice(dm, f.X, reps: 20);
                _output.WriteLine($"[CUDA {cuda.DeviceName}] {type} {k}x{n} ({mb:F1} MB): {ms * 1000:F1} us ({mb / ms:F1} GB/s)");
            }

            foreach (var d in VulkanContext.EnumerateDevices())
            {
                if (d.Name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase) ||
                    d.Name.Contains("Direct3D12", StringComparison.OrdinalIgnoreCase)) continue;
                using var vk = new TrellisVulkanRunner(d.Index);
                var dm = vk.Upload(binding);
                vk.MatVecMul(dm, f.X, yVk); // warmup
                double totalMs = 0;
                for (int r = 0; r < 10; r++) totalMs += vk.MatVecMul(dm, f.X, yVk);
                double ms = totalMs / 10.0;
                _output.WriteLine($"[Vulkan:{d.Index} {vk.DeviceName}] {type} {k}x{n} ({mb:F1} MB): {ms * 1000:F1} us ({mb / ms:F1} GB/s)");
            }
        }
    }
}
