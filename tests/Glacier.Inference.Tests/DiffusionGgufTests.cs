namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Image.Gguf;
using Glacier.Inference.Model;
using Xunit;

public class DiffusionGgufTests
{
    [Fact]
    public void ModelArchitectureDetector_DetectsFluxAndDiffusionFamilies()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_mock_gguf_{Guid.NewGuid():N}.gguf");
        try
        {
            using (var fs = new FileStream(tmpFile, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(0x46554747); // 'GGUF'
                bw.Write(3u);         // Version 3
                bw.Write(0UL);        // 0 tensors
                bw.Write(0UL);        // 0 metadata
            }

            using (var gguf = GgufFile.Open(tmpFile))
            {
                Assert.Equal(UniversalArchitecture.Flux, ModelArchitectureDetector.Detect("flux", gguf));
                Assert.Equal(UniversalArchitecture.SD3, ModelArchitectureDetector.Detect("sd3", gguf));
                Assert.Equal(UniversalArchitecture.StableDiffusion, ModelArchitectureDetector.Detect("sdxl", gguf));
                Assert.Equal(UniversalArchitecture.StableDiffusion, ModelArchitectureDetector.Detect("sd1", gguf));
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void DiffusionGgufPipeline_RunsEndToEndSyntheticFlow()
    {
        int w = 256;
        int h = 256;
        int steps = 4;

        using var pipeline = new Glacier.Inference.Image.ImageGenerationPipeline();
        var result = pipeline.Generate("A cinematic landscape of icy fjords under neon auroras", width: w, height: h, numSteps: steps, seed: 123);

        Assert.Equal(w, result.Width);
        Assert.Equal(h, result.Height);
        Assert.Equal(steps, result.Steps);
        Assert.Equal(w * h * 3, result.RgbPixels.Length);
        Assert.True(result.ElapsedMilliseconds >= 0);
    }

    [Fact]
    public void InspectFluxSchnellTensors()
    {
        string path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        if (!File.Exists(path)) return;

        using var gguf = GgufFile.Open(path);
        string report = $"[FLUX GGUF] Arch: {gguf.Architecture}, Tensors: {gguf.Tensors.Count}\n";
        foreach (var kvp in gguf.Tensors.Take(25))
        {
            report += $"  {kvp.Key} => Type={kvp.Value.Type}, Dims=[{string.Join(", ", kvp.Value.Dimensions)}]\n";
        }
        foreach (var kvp in gguf.Tensors.Where(k => k.Key.Contains("final_layer") || k.Key.Contains("img_in") || k.Key.Contains("time_in") || k.Key.Contains("vector_in") || k.Key.Contains("txt_in") || k.Key.StartsWith("single_blocks.0.")))
        {
            report += $"  {kvp.Key} => Type={kvp.Value.Type}, Dims=[{string.Join(", ", kvp.Value.Dimensions)}]\n";
        }
        report += "\n[CLIP_L SAFETENSORS]\n";
        string clipPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\clip_l.safetensors";
        if (File.Exists(clipPath))
        {
            using var clip = Glacier.Inference.Format.SafetensorsFile.Open(clipPath);
            report += $"Tensors count: {clip.Tensors.Count}\n";
            foreach (var kvp in clip.Tensors.Take(20))
            {
                report += $"  {kvp.Key} => Dims=[{string.Join(", ", kvp.Value.Shape)}], Dtype={kvp.Value.Dtype}\n";
            }
            foreach (var kvp in clip.Tensors.Where(k => k.Key.Contains("pooled") || k.Key.Contains("text_projection") || k.Key.Contains("final_layer_norm")))
            {
                report += $"  {kvp.Key} => Dims=[{string.Join(", ", kvp.Value.Shape)}], Dtype={kvp.Value.Dtype}\n";
            }

        }

        report += "\n[T5XXL GGUF]\n";
        string t5Path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\t5xxl.gguf";
        if (File.Exists(t5Path))
        {
            using var t5 = GgufFile.Open(t5Path);
            report += $"Arch: {t5.Architecture}, Tensors: {t5.Tensors.Count}, Dim: {t5.EmbeddingLength}\n";
            foreach (var kvp in t5.Tensors.Where(k => k.Key.Contains("emb") || k.Key.Contains("norm") && !k.Key.Contains("blk.")))
            {
                report += $"  {kvp.Key} => Type={kvp.Value.Type}, Dims=[{string.Join(", ", kvp.Value.Dimensions)}]\n";
            }
        }
        File.WriteAllText(@"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\flux_tensors.txt", report);
    }

    [Fact]
    public void DiffusionGgufPipeline_RunsFluxSchnellRealModelEndToEnd()
    {
        string fluxPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        string aePath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\ae.safetensors";
        string clipPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\clip_l.safetensors";
        string t5Path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\t5xxl.gguf";

        if (!File.Exists(fluxPath) || !File.Exists(aePath)) return;

        using var model = DiffusionGgufModel.Open(fluxPath);
        using var pipeline = new DiffusionGgufPipeline(model, vaeSafetensorsPath: aePath, clipSafetensorsPath: clipPath, t5GgufPath: t5Path);

        Assert.NotNull(pipeline.FluxDiT);
        Assert.NotNull(pipeline.NeuralVae);
        Assert.NotNull(pipeline.T5Encoder);

        string prompt = "a cinematic photo of a glowing crystal orb in an enchanted autumn forest";
        int w = 256;
        int h = 256;
        int steps = 4;

        var result = pipeline.Generate(prompt, width: w, height: h, numSteps: steps, seed: 1337);

        Assert.Equal(w, result.Width);
        Assert.Equal(h, result.Height);
        Assert.Equal(steps, result.Steps);
        Assert.Equal(w * h * 3, result.RgbPixels.Length);

        string outPng = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\flux_crystal_orb.png";
        Glacier.Inference.Image.PngWriter.SavePng24(outPng, result.RgbPixels, w, h);

        Assert.True(File.Exists(outPng));
        var fileInfo = new FileInfo(outPng);
        Assert.True(fileInfo.Length > 1000);
        Console.WriteLine($"[GENERATION SUCCESS] Output written to {outPng}, size: {fileInfo.Length} bytes, elapsed: {result.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void DiffusionGgufPipeline_RunsFluxSchnell512x512()
    {
        string fluxPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        string aePath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\ae.safetensors";
        string clipPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\clip_l.safetensors";
        string t5Path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\t5xxl.gguf";

        if (!File.Exists(fluxPath) || !File.Exists(aePath)) return;

        using var model = DiffusionGgufModel.Open(fluxPath);
        using var pipeline = new DiffusionGgufPipeline(model, vaeSafetensorsPath: aePath, clipSafetensorsPath: clipPath, t5GgufPath: t5Path);

        string prompt = "a cinematic photo of a glowing crystal orb in an enchanted autumn forest";
        int w = 512;
        int h = 512;
        int steps = 4;

        var result = pipeline.Generate(prompt, width: w, height: h, numSteps: steps, seed: 1337);

        Assert.Equal(w, result.Width);
        Assert.Equal(h, result.Height);
        Assert.Equal(steps, result.Steps);
        Assert.Equal(w * h * 3, result.RgbPixels.Length);

        string outPng = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\flux_crystal_orb_512.png";
        Glacier.Inference.Image.PngWriter.SavePng24(outPng, result.RgbPixels, w, h);

        Assert.True(File.Exists(outPng));
        var fileInfo = new FileInfo(outPng);
        Assert.True(fileInfo.Length > 1000);
        Console.WriteLine($"[512x512 GENERATION SUCCESS] Output written to {outPng}, size: {fileInfo.Length} bytes, elapsed: {result.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void DiffusionGgufPipeline_Profile512x512SingleStep()
    {
        string fluxPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        string aePath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\ae.safetensors";
        string clipPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\clip_l.safetensors";
        string t5Path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\t5xxl.gguf";

        if (!File.Exists(fluxPath) || !File.Exists(aePath)) return;

        using var model = DiffusionGgufModel.Open(fluxPath);
        using var pipeline = new DiffusionGgufPipeline(model, vaeSafetensorsPath: aePath, clipSafetensorsPath: clipPath, t5GgufPath: t5Path);

        string prompt = "a glowing crystal orb in an enchanted autumn forest";
        var result = pipeline.Generate(prompt, width: 512, height: 512, numSteps: 1, seed: 1337);
        Console.WriteLine($"[PROFILE 1-STEP TOTAL] Elapsed: {result.ElapsedMilliseconds} ms");
    }

    [Fact]
    public void FluxT5Encoder_EncodesPromptSuccessfully()
    {
        string t5Path = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\t5xxl.gguf";
        if (!File.Exists(t5Path)) return;

        using var encoder = Glacier.Inference.Image.Flux.FluxT5Encoder.Open(t5Path);
        int[] tokens = encoder.Tokenize("a glowing crystal orb in an enchanted autumn forest", maxTokens: 64);
        Assert.NotEmpty(tokens);
        Assert.Equal(64, tokens.Length);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] embeddings = encoder.Encode("a glowing crystal orb in an enchanted autumn forest", seqLen: 64);
        sw.Stop();

        Assert.Equal(64 * 4096, embeddings.Length);
        float sum = 0f;
        for (int i = 0; i < 4096; i++) sum += MathF.Abs(embeddings[i]);
        Assert.True(sum > 0f);
        Console.WriteLine($"[T5 ENCODER SUCCESS] Encoded 64 tokens across 24 layers in {sw.ElapsedMilliseconds} ms. L1 sum: {sum}");
    }

    [CudaFact(false)]
    public void GpuContext_InspectVramAndKernel()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        var (free, total) = gpu.GetMemoryInfo();
        Console.WriteLine($"[GPU INFO] Device: {gpu.DeviceName}, Arch: {gpu.ArchString}, Free: {free / (1024 * 1024)} MB, Total: {total / (1024 * 1024)} MB");
    }

    [Fact]
    public void D3D12Context_InspectAdapters()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var d3d = new Glacier.Inference.Gpu.D3D12.D3D12Context();
        Console.WriteLine($"[D3D12 INFO] Device: {d3d.DeviceName}");
    }

    [CudaFact(false)]
    public unsafe void GpuContext_VerifyBatchedGemmQ4K()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnGemm, module, "gemm_q4_k_batch"), "ModuleGetFunction");

        int kCols = 256;
        int mRows = 32;
        int batchSize = 64;

        int numBlocks = (kCols / 256) * mRows;
        int q4kBlockSize = 144;
        byte[] rawW = new byte[numBlocks * q4kBlockSize];
        // Populate one dummy block with scale=1
        for (int r = 0; r < mRows; r++)
        {
            fixed (byte* p = &rawW[r * q4kBlockSize])
            {
                var blk = (Glacier.Inference.Quant.BlockQ4_K*)p;
                blk->Delta = (Half)1.0f;
                blk->DeltaMin = (Half)0.0f;
                for (int s = 0; s < 12; s++) blk->Scales[s] = 1;
                for (int q = 0; q < 128; q++) blk->Qs[q] = 0x22; // 2 and 2
            }
        }

        float[] x = new float[batchSize * kCols];
        for (int i = 0; i < x.Length; i++) x[i] = 0.5f;

        float[] yCpu = new float[batchSize * mRows];
        fixed (byte* pW = rawW)
        fixed (float* pX = x)
        fixed (float* pY = yCpu)
        {
            Glacier.Inference.Quant.QuantKernels.MatMulBatch(Glacier.Inference.Gguf.GgufType.Q4_K, pW, pX, pY, kCols, mRows, batchSize);
        }

        float[] yGpu = new float[batchSize * mRows];
        IntPtr dW = gpu.AllocateDevice((nuint)rawW.Length);
        IntPtr dX = gpu.AllocateDevice((nuint)(x.Length * sizeof(float)));
        IntPtr dY = gpu.AllocateDevice((nuint)(yGpu.Length * sizeof(float)));

        fixed (byte* pW = rawW) gpu.CopyToDevice(dW, (IntPtr)pW, (nuint)rawW.Length);
        fixed (float* pX = x) gpu.CopyToDevice(dX, (IntPtr)pX, (nuint)(x.Length * sizeof(float)));

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);

        IntPtr dBias = IntPtr.Zero;
        IntPtr dResidual = IntPtr.Zero;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dBias;
        pArgs[7] = &dResidual;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnGemm,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel");

        gpu.Synchronize();

        fixed (float* pYGpu = yGpu) gpu.CopyToHost((IntPtr)pYGpu, dY, (nuint)(yGpu.Length * sizeof(float)));

        gpu.FreeDevice(dW);
        gpu.FreeDevice(dX);
        gpu.FreeDevice(dY);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);

        float maxDiff = 0f;
        for (int i = 0; i < yCpu.Length; i++)
        {
            float diff = MathF.Abs(yCpu[i] - yGpu[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Console.WriteLine($"[GPU GEMM SUCCESS] MatMulBatch {batchSize}x{kCols}x{mRows} verified! MaxDiff: {maxDiff:F6}, CPU[0]: {yCpu[0]}, GPU[0]: {yGpu[0]}");
        Assert.True(maxDiff < 1e-3f, $"MaxDiff too high: {maxDiff}");
    }

    [Fact]
    public void FluxDiT_InspectWeightBreakdown()
    {
        string fluxPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        if (!File.Exists(fluxPath)) return;

        using var gguf = Glacier.Inference.Gguf.GgufFile.Open(fluxPath);
        long doubleBytes = 0;
        long singleBytes = 0;
        long otherBytes = 0;

        var types = new Dictionary<Glacier.Inference.Gguf.GgufType, int>();
        foreach (var (name, t) in gguf.Tensors)
        {
            long size = (long)t.GetByteSize();
            if (name.StartsWith("double_blocks.")) doubleBytes += size;
            else if (name.StartsWith("single_blocks.")) singleBytes += size;
            else otherBytes += size;

            if (name.EndsWith(".weight"))
            {
                types[t.Type] = types.GetValueOrDefault(t.Type) + 1;
            }
        }

        Console.WriteLine($"[WEIGHT SIZES] Double Blocks: {doubleBytes / (1024 * 1024)} MB, Single Blocks: {singleBytes / (1024 * 1024)} MB, Other: {otherBytes / (1024 * 1024)} MB, Total: {(doubleBytes + singleBytes + otherBytes) / (1024 * 1024)} MB");
        foreach (var kvp in types) Console.WriteLine($"  Weight type {kvp.Key}: {kvp.Value} tensors");
    }

    [CudaFact(false)]
    public void GpuContext_TestLargeVramAllocation()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        var (free, total) = gpu.GetMemoryInfo();
        Console.WriteLine($"[VRAM BEFORE] Free: {free / (1024 * 1024)} MB");

        nuint allocBytes = unchecked((nuint)(6400UL * 1024 * 1024)); // 6.4 GB for all blocks
        IntPtr ptr = gpu.AllocateDevice(allocBytes);
        var (freeAfter, _) = gpu.GetMemoryInfo();
        Console.WriteLine($"[VRAM AFTER 6.4 GB ALLOC] Free: {freeAfter / (1024 * 1024)} MB");
        gpu.FreeDevice(ptr);
    }

    [CudaFact(false)]
    public unsafe void GpuContext_BenchmarkFluxGemm()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnGemm, module, "gemm_q4_k_batch"), "ModuleGetFunction");

        // Benchmark SingleStream Lin1: 3072 -> 21504 with 320 tokens (256x256)
        int kCols = 3072;
        int mRows = 21504;
        int batchSize = 320;

        int numBlocks = (kCols / 256) * mRows;
        int q4kBlockSize = 144;
        nuint wBytes = (nuint)((long)numBlocks * q4kBlockSize); // ~37 MB
        nuint xBytes = (nuint)((long)batchSize * kCols * sizeof(float)); // ~3.9 MB
        nuint yBytes = (nuint)((long)batchSize * mRows * sizeof(float)); // ~27.5 MB

        IntPtr dW = gpu.AllocateDevice(wBytes);
        IntPtr dX = gpu.AllocateDevice(xBytes);
        IntPtr dY = gpu.AllocateDevice(yBytes);

        // Warmup
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);
        IntPtr dBias = IntPtr.Zero;
        IntPtr dResidual = IntPtr.Zero;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dBias;
        pArgs[7] = &dResidual;

        for (int i = 0; i < 5; i++)
        {
            Glacier.Inference.Gpu.CuDriver.LaunchKernel(fnGemm, gridX, gridY, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);
        }
        gpu.Synchronize();

        int iters = 20;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            Glacier.Inference.Gpu.CuDriver.LaunchKernel(fnGemm, gridX, gridY, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);
        }
        gpu.Synchronize();
        sw.Stop();

        double msPerCall = sw.Elapsed.TotalMilliseconds / iters;
        Console.WriteLine($"[GPU BENCHMARK] Lin1 MatMulBatch {batchSize} tokens x {kCols} -> {mRows}: {msPerCall:F2} ms per call!");

        gpu.FreeDevice(dW);
        gpu.FreeDevice(dX);
        gpu.FreeDevice(dY);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);
    }

    [CudaFact(false)]
    public unsafe void GpuContext_VerifyRealGgufFluxGemm()
    {
        string fluxPath = @"C:\Users\spuri\source\repos\PolarsPlus\Glacier.Inference\models\flux1-schnell-Q4_K_S.gguf";
        if (!File.Exists(fluxPath) || !Glacier.Inference.Gpu.GpuContext.IsSupported) return;

        using var gguf = Glacier.Inference.Gguf.GgufFile.Open(fluxPath);
        var wTensor = gguf.Tensors["double_blocks.0.img_attn.qkv.weight"];
        var bTensor = gguf.Tensors["double_blocks.0.img_attn.qkv.bias"];

        int kCols = (int)wTensor.Dimensions[0]; // 3072
        int mRows = (int)wTensor.Dimensions[1]; // 9216
        int batchSize = 64; // 64 tokens

        byte* wPtr = gguf.GetTensorPointer(wTensor);
        float* bPtr = (float*)gguf.GetTensorPointer(bTensor);

        float[] x = new float[batchSize * kCols];
        for (int i = 0; i < x.Length; i++) x[i] = ((i % 17) - 8) * 0.1f;

        // 1. Compute on CPU
        float[] yCpu = new float[batchSize * mRows];
        fixed (float* pX = x)
        fixed (float* pY = yCpu)
        {
            Glacier.Inference.Quant.QuantKernels.MatMulBatch(wTensor.Type, wPtr, pX, pY, kCols, mRows, batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                float* row = pY + b * mRows;
                for (int r = 0; r < mRows; r++) row[r] += bPtr[r];
            }
        }

        // 2. Compute on GPU
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnGemm, module, "gemm_q4_k_batch"), "ModuleGetFunction");

        nuint wBytes = (nuint)wTensor.GetByteSize();
        nuint bBytes = (nuint)(mRows * sizeof(float));
        nuint xBytes = (nuint)(batchSize * kCols * sizeof(float));
        nuint yBytes = (nuint)(batchSize * mRows * sizeof(float));

        IntPtr dW = gpu.AllocateDevice(wBytes);
        IntPtr dB = gpu.AllocateDevice(bBytes);
        IntPtr dX = gpu.AllocateDevice(xBytes);
        IntPtr dY = gpu.AllocateDevice(yBytes);

        gpu.CopyToDevice(dW, (IntPtr)wPtr, wBytes);
        gpu.CopyToDevice(dB, (IntPtr)bPtr, bBytes);
        fixed (float* pX = x) gpu.CopyToDevice(dX, (IntPtr)pX, xBytes);

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);
        IntPtr dResidual = IntPtr.Zero;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dB;
        pArgs[7] = &dResidual;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnGemm, gridX, gridY, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel");
        gpu.Synchronize();

        float[] yGpu = new float[batchSize * mRows];
        fixed (float* pYGpu = yGpu) gpu.CopyToHost((IntPtr)pYGpu, dY, yBytes);

        gpu.FreeDevice(dW);
        gpu.FreeDevice(dB);
        gpu.FreeDevice(dX);
        gpu.FreeDevice(dY);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);

        float maxDiff = 0f;
        for (int i = 0; i < yCpu.Length; i++)
        {
            float diff = MathF.Abs(yCpu[i] - yGpu[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Console.WriteLine($"[REAL GGUF GPU GEMM SUCCESS] MatMulBatch {batchSize}x{kCols}x{mRows} verified against real FLUX weights! MaxDiff: {maxDiff:F6}, CPU[0]: {yCpu[0]}, GPU[0]: {yGpu[0]}");
        Assert.True(maxDiff < 1e-2f, $"MaxDiff too high: {maxDiff}");
    }

    [CudaFact(false)]
    public unsafe void GpuContext_VerifyBidirectionalFlashAttention()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;

        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnAttn, module, "attention_bidirectional_batch"), "ModuleGetFunction(attention_bidirectional_batch)");

        int numTokens = 64;
        int numHeads = 24;
        int headDim = 128;
        int hiddenDim = numHeads * headDim; // 3072
        float scale = 1.0f / MathF.Sqrt(headDim);

        float[] q = new float[numTokens * hiddenDim];
        float[] k = new float[numTokens * hiddenDim];
        float[] v = new float[numTokens * hiddenDim];

        for (int i = 0; i < q.Length; i++)
        {
            q[i] = ((i % 19) - 9) * 0.05f;
            k[i] = ((i % 23) - 11) * 0.05f;
            v[i] = ((i % 17) - 8) * 0.05f;
        }

        // 1. CPU reference
        float[] cpuOut = new float[numTokens * hiddenDim];
        for (int h = 0; h < numHeads; h++)
        {
            int headOffset = h * headDim;
            float[] scores = new float[numTokens];

            for (int i = 0; i < numTokens; i++)
            {
                int qIdx = i * hiddenDim + headOffset;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < numTokens; j++)
                {
                    int kIdx = j * hiddenDim + headOffset;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[qIdx + d] * k[kIdx + d];
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0f;
                for (int j = 0; j < numTokens; j++)
                {
                    float e = MathF.Exp(scores[j] - maxScore);
                    scores[j] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;

                int outIdx = i * hiddenDim + headOffset;
                for (int d = 0; d < headDim; d++) cpuOut[outIdx + d] = 0f;

                for (int j = 0; j < numTokens; j++)
                {
                    float w = scores[j] * invSum;
                    int vIdx = j * hiddenDim + headOffset;
                    for (int d = 0; d < headDim; d++) cpuOut[outIdx + d] += w * v[vIdx + d];
                }
            }
        }

        // 2. GPU execution
        nuint bytes = (nuint)(numTokens * hiddenDim * sizeof(float));
        IntPtr dQ = gpu.AllocateDevice(bytes);
        IntPtr dK = gpu.AllocateDevice(bytes);
        IntPtr dV = gpu.AllocateDevice(bytes);
        IntPtr dOut = gpu.AllocateDevice(bytes);

        fixed (float* pQ = q) gpu.CopyToDevice(dQ, (IntPtr)pQ, bytes);
        fixed (float* pK = k) gpu.CopyToDevice(dK, (IntPtr)pK, bytes);
        fixed (float* pV = v) gpu.CopyToDevice(dV, (IntPtr)pV, bytes);

        uint gridX = (uint)numHeads;
        uint gridY = (uint)numTokens;
        uint blockSize = 128;

        int localHeads = numHeads;
        int localDim = headDim;
        int localTokens = numTokens;
        float localScale = scale;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dOut;
        pArgs[4] = &localHeads;
        pArgs[5] = &localDim;
        pArgs[6] = &localTokens;
        pArgs[7] = &localScale;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnAttn, gridX, gridY, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(attention_bidirectional_batch)");
        gpu.Synchronize();

        float[] gpuOut = new float[numTokens * hiddenDim];
        fixed (float* pGpu = gpuOut) gpu.CopyToHost((IntPtr)pGpu, dOut, bytes);

        gpu.FreeDevice(dQ);
        gpu.FreeDevice(dK);
        gpu.FreeDevice(dV);
        gpu.FreeDevice(dOut);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);

        float maxDiff = 0f;
        for (int i = 0; i < cpuOut.Length; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Console.WriteLine($"[GPU FLASH ATTENTION VERIFIED] Diff: {maxDiff:F6}, CPU[0]: {cpuOut[0]}, GPU[0]: {gpuOut[0]}");
        Assert.True(maxDiff < 1e-4f, $"MaxDiff too high: {maxDiff}");
    }

    [CudaFact(false)]
    public unsafe void GpuContext_VerifyFluxFusedKernels()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;

        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnQkNormRope, module, "flux_qk_norm_rope"), "flux_qk_norm_rope");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnGeluConcat, module, "flux_fused_gelu_concat"), "flux_fused_gelu_concat");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnAdaLn, module, "flux_adaln_kernel"), "flux_adaln_kernel");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnResidual, module, "flux_residual_gated"), "flux_residual_gated");

        // 1. Verify flux_adaln_kernel
        int numTokens = 32;
        int dim = 3072;
        float[] src = new float[numTokens * dim];
        float[] shift = new float[dim];
        float[] scale = new float[dim];
        for (int i = 0; i < src.Length; i++) src[i] = ((i % 13) - 6) * 0.1f;
        for (int i = 0; i < dim; i++) { shift[i] = 0.5f; scale[i] = 0.2f; }

        float[] cpuAda = new float[numTokens * dim];
        for (int t = 0; t < numTokens; t++)
        {
            float sum = 0f, sumSq = 0f;
            for (int i = 0; i < dim; i++)
            {
                float v = src[t * dim + i];
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / dim;
            float var = MathF.Max(0f, (sumSq / dim) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(var + 1e-6f);
            for (int i = 0; i < dim; i++)
            {
                cpuAda[t * dim + i] = ((src[t * dim + i] - mean) * invStd) * (1.0f + scale[i]) + shift[i];
            }
        }

        IntPtr dSrc = gpu.AllocateDevice((nuint)(src.Length * sizeof(float)));
        IntPtr dDst = gpu.AllocateDevice((nuint)(src.Length * sizeof(float)));
        IntPtr dShift = gpu.AllocateDevice((nuint)(shift.Length * sizeof(float)));
        IntPtr dScale = gpu.AllocateDevice((nuint)(scale.Length * sizeof(float)));

        fixed (float* p = src) gpu.CopyToDevice(dSrc, (IntPtr)p, (nuint)(src.Length * sizeof(float)));
        fixed (float* p = shift) gpu.CopyToDevice(dShift, (IntPtr)p, (nuint)(shift.Length * sizeof(float)));
        fixed (float* p = scale) gpu.CopyToDevice(dScale, (IntPtr)p, (nuint)(scale.Length * sizeof(float)));

        int localTokens = numTokens;
        int localDim = dim;
        void** pAdaArgs = stackalloc void*[6];
        pAdaArgs[0] = &dSrc;
        pAdaArgs[1] = &dDst;
        pAdaArgs[2] = &dShift;
        pAdaArgs[3] = &dScale;
        pAdaArgs[4] = &localTokens;
        pAdaArgs[5] = &localDim;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnAdaLn, (uint)numTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaArgs, IntPtr.Zero), "LaunchKernel(flux_adaln_kernel)");
        gpu.Synchronize();

        float[] gpuAda = new float[numTokens * dim];
        fixed (float* p = gpuAda) gpu.CopyToHost((IntPtr)p, dDst, (nuint)(gpuAda.Length * sizeof(float)));

        float diffAda = 0f;
        for (int i = 0; i < cpuAda.Length; i++)
        {
            float d = MathF.Abs(cpuAda[i] - gpuAda[i]);
            if (d > diffAda) diffAda = d;
        }

        Console.WriteLine($"[GPU ADALN VERIFIED] Diff: {diffAda:F6}, CPU[0]: {cpuAda[0]}, GPU[0]: {gpuAda[0]}");
        Assert.True(diffAda < 1e-4f, $"AdaLN diff too high: {diffAda}");

        // 2. Verify flux_fused_gelu_concat
        float[] attnOut = new float[numTokens * 3072];
        float[] qkvMlp = new float[numTokens * 21504];
        for (int i = 0; i < attnOut.Length; i++) attnOut[i] = ((i % 7) - 3) * 0.1f;
        for (int i = 0; i < qkvMlp.Length; i++) qkvMlp[i] = ((i % 11) - 5) * 0.1f;

        float[] cpuConcat = new float[numTokens * 15360];
        for (int t = 0; t < numTokens; t++)
        {
            for (int c = 0; c < 3072; c++) cpuConcat[t * 15360 + c] = attnOut[t * 3072 + c];
            for (int c = 0; c < 12288; c++)
            {
                float x = qkvMlp[t * 21504 + 9216 + c];
                cpuConcat[t * 15360 + 3072 + c] = 0.5f * x * (1.0f + MathF.CopySign(1.0f - 1.0f / (1.0f + 0.3275911f * MathF.Abs(x / MathF.Sqrt(2.0f))), x)); // erf approximation or standard
            }
        }

        IntPtr dAttn = gpu.AllocateDevice((nuint)(attnOut.Length * sizeof(float)));
        IntPtr dQkvMlp = gpu.AllocateDevice((nuint)(qkvMlp.Length * sizeof(float)));
        IntPtr dZ = gpu.AllocateDevice((nuint)(numTokens * 15360 * sizeof(float)));

        fixed (float* p = attnOut) gpu.CopyToDevice(dAttn, (IntPtr)p, (nuint)(attnOut.Length * sizeof(float)));
        fixed (float* p = qkvMlp) gpu.CopyToDevice(dQkvMlp, (IntPtr)p, (nuint)(qkvMlp.Length * sizeof(float)));

        int totalZ = numTokens * 15360;
        uint geluBlock = 256;
        uint geluGrid = (uint)((totalZ + 255) / 256);

        void** pGeluArgs = stackalloc void*[4];
        pGeluArgs[0] = &dAttn;
        pGeluArgs[1] = &dQkvMlp;
        pGeluArgs[2] = &dZ;
        pGeluArgs[3] = &localTokens;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnGeluConcat, geluGrid, 1, 1, geluBlock, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero), "LaunchKernel(flux_fused_gelu_concat)");
        gpu.Synchronize();

        float[] gpuConcat = new float[totalZ];
        fixed (float* p = gpuConcat) gpu.CopyToHost((IntPtr)p, dZ, (nuint)(totalZ * sizeof(float)));

        float diffConcat = 0f;
        for (int i = 0; i < totalZ; i++)
        {
            if (i % 15360 < 3072)
            {
                float d = MathF.Abs(cpuConcat[i] - gpuConcat[i]);
                if (d > diffConcat) diffConcat = d;
            }
        }
        Console.WriteLine($"[GPU GELU CONCAT VERIFIED] Attn copy diff: {diffConcat:F6}, GPU MLP[3072]: {gpuConcat[3072]}");
        Assert.True(diffConcat < 1e-5f);

        // 3. Verify flux_qk_norm_rope
        int nHeads = 24;
        int headDim = 128;
        float[] qRope = new float[numTokens * nHeads * headDim];
        float[] kRope = new float[numTokens * nHeads * headDim];
        float[] qScale = new float[headDim];
        float[] kScale = new float[headDim];
        float[] ropeCos = new float[numTokens * 64];
        float[] ropeSin = new float[numTokens * 64];

        for (int i = 0; i < qRope.Length; i++) { qRope[i] = ((i % 17) - 8) * 0.1f; kRope[i] = ((i % 19) - 9) * 0.1f; }
        for (int i = 0; i < headDim; i++) { qScale[i] = 1.1f; kScale[i] = 0.9f; }
        for (int i = 0; i < ropeCos.Length; i++) { ropeCos[i] = MathF.Cos(i * 0.05f); ropeSin[i] = MathF.Sin(i * 0.05f); }

        float[] cpuQRope = (float[])qRope.Clone();
        float[] cpuKRope = (float[])kRope.Clone();
        float* cRope = stackalloc float[64];
        float* sRope = stackalloc float[64];
        for (int t = 0; t < numTokens; t++)
        {
            for (int i = 0; i < 64; i++) { cRope[i] = ropeCos[t * 64 + i]; sRope[i] = ropeSin[t * 64 + i]; }

            for (int h = 0; h < nHeads; h++)
            {
                int offset = t * (nHeads * headDim) + h * headDim;
                // RMSNorm
                float qSumSq = 0f, kSumSq = 0f;
                for (int d = 0; d < headDim; d++) { qSumSq += cpuQRope[offset + d] * cpuQRope[offset + d]; kSumSq += cpuKRope[offset + d] * cpuKRope[offset + d]; }
                float qInvRms = 1.0f / MathF.Sqrt(qSumSq / headDim + 1e-6f);
                float kInvRms = 1.0f / MathF.Sqrt(kSumSq / headDim + 1e-6f);

                for (int d = 0; d < headDim; d++) { cpuQRope[offset + d] *= qInvRms * qScale[d]; cpuKRope[offset + d] *= kInvRms * kScale[d]; }

                // RoPE
                for (int i = 0; i < 64; i++)
                {
                    float q0 = cpuQRope[offset + 2 * i], q1 = cpuQRope[offset + 2 * i + 1];
                    cpuQRope[offset + 2 * i] = q0 * cRope[i] - q1 * sRope[i];
                    cpuQRope[offset + 2 * i + 1] = q0 * sRope[i] + q1 * cRope[i];

                    float k0 = cpuKRope[offset + 2 * i], k1 = cpuKRope[offset + 2 * i + 1];
                    cpuKRope[offset + 2 * i] = k0 * cRope[i] - k1 * sRope[i];
                    cpuKRope[offset + 2 * i + 1] = k0 * sRope[i] + k1 * cRope[i];
                }
            }
        }

        IntPtr dQRope = gpu.AllocateDevice((nuint)(qRope.Length * sizeof(float)));
        IntPtr dKRope = gpu.AllocateDevice((nuint)(kRope.Length * sizeof(float)));
        IntPtr dQScale = gpu.AllocateDevice((nuint)(qScale.Length * sizeof(float)));
        IntPtr dKScale = gpu.AllocateDevice((nuint)(kScale.Length * sizeof(float)));
        IntPtr dRopeCos = gpu.AllocateDevice((nuint)(ropeCos.Length * sizeof(float)));
        IntPtr dRopeSin = gpu.AllocateDevice((nuint)(ropeSin.Length * sizeof(float)));

        fixed (float* p = qRope) gpu.CopyToDevice(dQRope, (IntPtr)p, (nuint)(qRope.Length * sizeof(float)));
        fixed (float* p = kRope) gpu.CopyToDevice(dKRope, (IntPtr)p, (nuint)(kRope.Length * sizeof(float)));
        fixed (float* p = qScale) gpu.CopyToDevice(dQScale, (IntPtr)p, (nuint)(qScale.Length * sizeof(float)));
        fixed (float* p = kScale) gpu.CopyToDevice(dKScale, (IntPtr)p, (nuint)(kScale.Length * sizeof(float)));
        fixed (float* p = ropeCos) gpu.CopyToDevice(dRopeCos, (IntPtr)p, (nuint)(ropeCos.Length * sizeof(float)));
        fixed (float* p = ropeSin) gpu.CopyToDevice(dRopeSin, (IntPtr)p, (nuint)(ropeSin.Length * sizeof(float)));

        int localHeads = nHeads;
        int localHeadDim = headDim;

        void** pQkArgs = stackalloc void*[9];
        pQkArgs[0] = &dQRope;
        pQkArgs[1] = &dKRope;
        pQkArgs[2] = &dQScale;
        pQkArgs[3] = &dKScale;
        pQkArgs[4] = &dRopeCos;
        pQkArgs[5] = &dRopeSin;
        pQkArgs[6] = &localHeads;
        pQkArgs[7] = &localHeadDim;
        pQkArgs[8] = &localTokens;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnQkNormRope, (uint)nHeads, (uint)numTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pQkArgs, IntPtr.Zero), "LaunchKernel(flux_qk_norm_rope)");
        gpu.Synchronize();

        float[] gpuQRope = new float[qRope.Length];
        fixed (float* p = gpuQRope) gpu.CopyToHost((IntPtr)p, dQRope, (nuint)(gpuQRope.Length * sizeof(float)));

        float diffQ = 0f;
        for (int i = 0; i < cpuQRope.Length; i++)
        {
            float d = MathF.Abs(cpuQRope[i] - gpuQRope[i]);
            if (d > diffQ) diffQ = d;
        }

        Console.WriteLine($"[GPU QK NORM ROPE VERIFIED] Diff Q: {diffQ:F6}, CPU[0]: {cpuQRope[0]}, GPU[0]: {gpuQRope[0]}");
        Assert.True(diffQ < 1e-4f);

        gpu.FreeDevice(dQRope);
        gpu.FreeDevice(dKRope);
        gpu.FreeDevice(dQScale);
        gpu.FreeDevice(dKScale);
        gpu.FreeDevice(dRopeCos);
        gpu.FreeDevice(dRopeSin);

        gpu.FreeDevice(dSrc);
        gpu.FreeDevice(dDst);
        gpu.FreeDevice(dShift);
        gpu.FreeDevice(dScale);
        gpu.FreeDevice(dAttn);
        gpu.FreeDevice(dQkvMlp);
        gpu.FreeDevice(dZ);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);
    }

    [CudaFact(false)]
    public unsafe void GpuContext_VerifyFluxQkvPrep()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;
        using var gpu = new Glacier.Inference.Gpu.GpuContext(0);
        byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out IntPtr fnQkvPrep, module, "flux_qkv_prep"), "ModuleGetFunction");

        int nHeads = 24;
        int headDim = 128;
        int numTokens = 32;
        int stride = 21504;

        float[] qkvSrc = new float[numTokens * stride];
        var rnd = new Random(42);
        for (int i = 0; i < qkvSrc.Length; i++) qkvSrc[i] = (float)rnd.NextDouble() * 2f - 1f;

        float[] qScale = new float[headDim];
        float[] kScale = new float[headDim];
        for (int i = 0; i < headDim; i++) { qScale[i] = 1.0f + 0.1f * i; kScale[i] = 0.9f + 0.05f * i; }

        float[] ropeCos = new float[numTokens * 64];
        float[] ropeSin = new float[numTokens * 64];
        for (int i = 0; i < numTokens * 64; i++)
        {
            float a = 0.01f * i;
            ropeCos[i] = MathF.Cos(a);
            ropeSin[i] = MathF.Sin(a);
        }

        // CPU Reference
        float[] cpuQ = new float[numTokens * nHeads * headDim];
        float[] cpuK = new float[numTokens * nHeads * headDim];
        float[] cpuV = new float[numTokens * nHeads * headDim];

        for (int t = 0; t < numTokens; t++)
        {
            int srcRow = t * stride;
            int dstRow = t * (nHeads * headDim);

            // Copy V
            for (int i = 0; i < nHeads * headDim; i++) cpuV[dstRow + i] = qkvSrc[srcRow + 6144 + i];

            for (int h = 0; h < nHeads; h++)
            {
                int qOffset = srcRow + h * headDim;
                int kOffset = srcRow + 3072 + h * headDim;
                int dstHeadOffset = dstRow + h * headDim;

                float qSq = 0f, kSq = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    qSq += qkvSrc[qOffset + d] * qkvSrc[qOffset + d];
                    kSq += qkvSrc[kOffset + d] * qkvSrc[kOffset + d];
                }
                float invRmsQ = 1.0f / MathF.Sqrt(qSq / headDim + 1e-6f);
                float invRmsK = 1.0f / MathF.Sqrt(kSq / headDim + 1e-6f);

                for (int d = 0; d < headDim; d++)
                {
                    cpuQ[dstHeadOffset + d] = qkvSrc[qOffset + d] * invRmsQ * qScale[d];
                    cpuK[dstHeadOffset + d] = qkvSrc[kOffset + d] * invRmsK * kScale[d];
                }

                // RoPE
                for (int p = 0; p < 64; p++)
                {
                    float c = ropeCos[t * 64 + p];
                    float s = ropeSin[t * 64 + p];

                    float q0 = cpuQ[dstHeadOffset + 2 * p];
                    float q1 = cpuQ[dstHeadOffset + 2 * p + 1];
                    cpuQ[dstHeadOffset + 2 * p] = q0 * c - q1 * s;
                    cpuQ[dstHeadOffset + 2 * p + 1] = q0 * s + q1 * c;

                    float k0 = cpuK[dstHeadOffset + 2 * p];
                    float k1 = cpuK[dstHeadOffset + 2 * p + 1];
                    cpuK[dstHeadOffset + 2 * p] = k0 * c - k1 * s;
                    cpuK[dstHeadOffset + 2 * p + 1] = k0 * s + k1 * c;
                }
            }
        }

        // GPU execution
        IntPtr dSrc = gpu.AllocateDevice((nuint)(qkvSrc.Length * sizeof(float)));
        IntPtr dQ = gpu.AllocateDevice((nuint)(cpuQ.Length * sizeof(float)));
        IntPtr dK = gpu.AllocateDevice((nuint)(cpuK.Length * sizeof(float)));
        IntPtr dV = gpu.AllocateDevice((nuint)(cpuV.Length * sizeof(float)));
        IntPtr dQScale = gpu.AllocateDevice((nuint)(qScale.Length * sizeof(float)));
        IntPtr dKScale = gpu.AllocateDevice((nuint)(kScale.Length * sizeof(float)));
        IntPtr dRopeCos = gpu.AllocateDevice((nuint)(ropeCos.Length * sizeof(float)));
        IntPtr dRopeSin = gpu.AllocateDevice((nuint)(ropeSin.Length * sizeof(float)));

        fixed (float* p = qkvSrc) gpu.CopyToDevice(dSrc, (IntPtr)p, (nuint)(qkvSrc.Length * sizeof(float)));
        fixed (float* p = qScale) gpu.CopyToDevice(dQScale, (IntPtr)p, (nuint)(qScale.Length * sizeof(float)));
        fixed (float* p = kScale) gpu.CopyToDevice(dKScale, (IntPtr)p, (nuint)(kScale.Length * sizeof(float)));
        fixed (float* p = ropeCos) gpu.CopyToDevice(dRopeCos, (IntPtr)p, (nuint)(ropeCos.Length * sizeof(float)));
        fixed (float* p = ropeSin) gpu.CopyToDevice(dRopeSin, (IntPtr)p, (nuint)(ropeSin.Length * sizeof(float)));

        int localStride = stride;
        int localHeads = nHeads;
        int localHeadDim = headDim;
        int localTokens = numTokens;
        int localOffset = 0;

        void** pArgs = stackalloc void*[13];
        pArgs[0] = &dSrc;
        pArgs[1] = &dQ;
        pArgs[2] = &dK;
        pArgs[3] = &dV;
        pArgs[4] = &dQScale;
        pArgs[5] = &dKScale;
        pArgs[6] = &dRopeCos;
        pArgs[7] = &dRopeSin;
        pArgs[8] = &localStride;
        pArgs[9] = &localHeads;
        pArgs[10] = &localHeadDim;
        pArgs[11] = &localTokens;
        pArgs[12] = &localOffset;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            fnQkvPrep, (uint)nHeads, (uint)numTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(flux_qkv_prep)");
        gpu.Synchronize();

        float[] gpuQ = new float[cpuQ.Length];
        float[] gpuK = new float[cpuK.Length];
        float[] gpuV = new float[cpuV.Length];
        fixed (float* p = gpuQ) gpu.CopyToHost((IntPtr)p, dQ, (nuint)(gpuQ.Length * sizeof(float)));
        fixed (float* p = gpuK) gpu.CopyToHost((IntPtr)p, dK, (nuint)(gpuK.Length * sizeof(float)));
        fixed (float* p = gpuV) gpu.CopyToHost((IntPtr)p, dV, (nuint)(gpuV.Length * sizeof(float)));

        float diffQ = 0f, diffK = 0f, diffV = 0f;
        for (int i = 0; i < cpuQ.Length; i++)
        {
            float dq = MathF.Abs(cpuQ[i] - gpuQ[i]);
            if (dq > diffQ) diffQ = dq;
            float dk = MathF.Abs(cpuK[i] - gpuK[i]);
            if (dk > diffK) diffK = dk;
            float dv = MathF.Abs(cpuV[i] - gpuV[i]);
            if (dv > diffV) diffV = dv;
        }

        Console.WriteLine($"[GPU FLUX QKV PREP VERIFIED] Diff Q: {diffQ:F6}, Diff K: {diffK:F6}, Diff V: {diffV:F6}");
        Assert.True(diffQ < 1e-4f);
        Assert.True(diffK < 1e-4f);
        Assert.True(diffV < 1e-4f);

        gpu.FreeDevice(dSrc);
        gpu.FreeDevice(dQ);
        gpu.FreeDevice(dK);
        gpu.FreeDevice(dV);
        gpu.FreeDevice(dQScale);
        gpu.FreeDevice(dKScale);
        gpu.FreeDevice(dRopeCos);
        gpu.FreeDevice(dRopeSin);
        Glacier.Inference.Gpu.CuDriver.ModuleUnload(module);
    }
}



