namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Xunit;
using Xunit.Abstractions;

public unsafe class GpuKernelValidationTests
{
    private readonly ITestOutputHelper _output;
    private const string ModelPath = @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";

    public GpuKernelValidationTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void InspectLayer0TypesAndCompareCpuVsGpu()
    {
        if (!File.Exists(ModelPath))
        {
            _output.WriteLine("Model file not found, skipping.");
            return;
        }

        using var gguf = GgufFile.Open(ModelPath);
        _output.WriteLine($"Arch: {gguf.Architecture}, layers: {gguf.BlockCount}, dim: {gguf.EmbeddingLength}");

        var qInfo = gguf.Tensors["blk.0.attn_q.weight"];
        var kInfo = gguf.Tensors["blk.0.attn_k.weight"];
        var vInfo = gguf.Tensors["blk.0.attn_v.weight"];
        var outAttnInfo = gguf.Tensors["blk.0.attn_output.weight"];
        var gateInfo = gguf.Tensors["blk.0.ffn_gate.weight"];
        var upInfo = gguf.Tensors["blk.0.ffn_up.weight"];
        var downInfo = gguf.Tensors["blk.0.ffn_down.weight"];
        var lmHeadInfo = gguf.Tensors["output.weight"];

        _output.WriteLine($"Q type: {qInfo.Type}, dims: [{string.Join(", ", qInfo.Dimensions)}]");
        _output.WriteLine($"K type: {kInfo.Type}, dims: [{string.Join(", ", kInfo.Dimensions)}]");
        _output.WriteLine($"V type: {vInfo.Type}, dims: [{string.Join(", ", vInfo.Dimensions)}]");
        _output.WriteLine($"AttnOut type: {outAttnInfo.Type}, dims: [{string.Join(", ", outAttnInfo.Dimensions)}]");
        _output.WriteLine($"Gate type: {gateInfo.Type}, dims: [{string.Join(", ", gateInfo.Dimensions)}]");
        _output.WriteLine($"Up type: {upInfo.Type}, dims: [{string.Join(", ", upInfo.Dimensions)}]");
        _output.WriteLine($"Down type: {downInfo.Type}, dims: [{string.Join(", ", downInfo.Dimensions)}]");
        _output.WriteLine($"LM Head type: {lmHeadInfo.Type}, dims: [{string.Join(", ", lmHeadInfo.Dimensions)}]");

        if (!GpuContext.IsSupported)
        {
            _output.WriteLine("GPU not supported, skipping GPU comparison.");
            return;
        }

        using var gpu = new GpuContext();
        byte[] cubin = KernelCompiler.GetOrCompileKernels("sm_89");
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ4K, module, "gemv_q4_k"), "ModuleGetFunction");

        // Compare first 4 rows of Q matrix
        int dim = (int)qInfo.Dimensions[0]; // 3584
        int mRows = 4;
        float[] x = new float[dim];
        for (int i = 0; i < dim; i++) x[i] = 0.1f * ((i % 7) - 3);

        float[] yCpu = new float[mRows];
        BlockQ4_K* pW = (BlockQ4_K*)gguf.GetTensorPointer(qInfo);
        int nb = dim / 256;

        fixed (float* pX = x)
        {
            for (int r = 0; r < mRows; r++)
            {
                yCpu[r] = QuantKernels.VecDotQ4_K(pW + (nuint)r * (nuint)nb, pX, dim);
            }
        }

        // Run on GPU
        IntPtr dX = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dY = gpu.AllocateDevice((nuint)(mRows * sizeof(float)));
        nuint wBytes = (nuint)(mRows * nb * sizeof(BlockQ4_K));
        IntPtr dW = gpu.AllocateDevice(wBytes);

        fixed (float* pX = x)
        {
            gpu.CopyToDevice(dX, (IntPtr)pX, (nuint)(dim * sizeof(float)));
            gpu.CopyToDevice(dW, (IntPtr)pW, wBytes);
        }

        // Launch kernel
        void* pDY = &dY;
        void* pDX = &dX;
        void* pDW = &dW;
        int kCols = dim;
        void* pK = &kCols;
        void* pM = &mRows;
        void*[] args = [pDY, pDX, pDW, pK, pM];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                fnGemvQ4K,
                1, 1, 1,
                128, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel");
        }

        float[] yGpu = new float[mRows];
        fixed (float* pYGpu = yGpu)
        {
            gpu.CopyToHost((IntPtr)pYGpu, dY, (nuint)(mRows * sizeof(float)));
        }

        for (int r = 0; r < mRows; r++)
        {
            _output.WriteLine($"[Q4_K] Row {r}: CPU = {yCpu[r]:F6}, GPU = {yGpu[r]:F6}, Diff = {Math.Abs(yCpu[r] - yGpu[r]):F6}");
        }

        gpu.FreeDevice(dX);
        gpu.FreeDevice(dY);
        gpu.FreeDevice(dW);

        // Now test Q6_K on V weight
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ6K, module, "gemv_q6_k"), "ModuleGetFunction(gemv_q6_k)");
        BlockQ6_K* pWV = (BlockQ6_K*)gguf.GetTensorPointer(vInfo);
        float[] yCpuV = new float[mRows];
        fixed (float* pX = x)
        {
            for (int r = 0; r < mRows; r++)
            {
                yCpuV[r] = QuantKernels.VecDotQ6_K(pWV + (nuint)r * (nuint)nb, pX, dim);
            }
        }

        dX = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        dY = gpu.AllocateDevice((nuint)(mRows * sizeof(float)));
        nuint wVBytes = (nuint)(mRows * nb * sizeof(BlockQ6_K));
        dW = gpu.AllocateDevice(wVBytes);

        fixed (float* pX = x)
        {
            gpu.CopyToDevice(dX, (IntPtr)pX, (nuint)(dim * sizeof(float)));
            gpu.CopyToDevice(dW, (IntPtr)pWV, wVBytes);
        }

        pDY = &dY;
        pDX = &dX;
        pDW = &dW;
        pK = &kCols;
        pM = &mRows;
        args = [pDY, pDX, pDW, pK, pM];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                fnGemvQ6K,
                1, 1, 1,
                128, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(gemv_q6_k)");
        }

        float[] yGpuV = new float[mRows];
        fixed (float* pYGpuV = yGpuV)
        {
            gpu.CopyToHost((IntPtr)pYGpuV, dY, (nuint)(mRows * sizeof(float)));
        }

        for (int r = 0; r < mRows; r++)
        {
            _output.WriteLine($"[Q6_K] Row {r}: CPU = {yCpuV[r]:F6}, GPU = {yGpuV[r]:F6}, Diff = {Math.Abs(yCpuV[r] - yGpuV[r]):F6}");
        }

        gpu.FreeDevice(dX);
        gpu.FreeDevice(dY);
        gpu.FreeDevice(dW);

        // Test RMSNorm
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnRmsNorm, module, "rms_norm_kernel"), "ModuleGetFunction(rms_norm_kernel)");
        float[] normWeight = new float[dim];
        for (int i = 0; i < dim; i++) normWeight[i] = 1.0f;
        float[] xNormCpu = new float[dim];
        fixed (float* pX = x, pNormW = normWeight, pDst = xNormCpu)
        {
            QuantKernels.RMSNorm(pX, pNormW, pDst, dim, 1e-5f);
        }

        IntPtr dNormIn = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dNormW = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dNormOut = gpu.AllocateDevice((nuint)(dim * sizeof(float)));

        fixed (float* pX = x, pNormW = normWeight)
        {
            gpu.CopyToDevice(dNormIn, (IntPtr)pX, (nuint)(dim * sizeof(float)));
            gpu.CopyToDevice(dNormW, (IntPtr)pNormW, (nuint)(dim * sizeof(float)));
        }

        float eps = 1e-5f;
        void* pDIn = &dNormIn;
        void* pDWn = &dNormW;
        void* pDOut = &dNormOut;
        int nSize = dim;
        void* pSize = &nSize;
        void* pEps = &eps;
        void*[] normArgs = [pDIn, pDWn, pDOut, pSize, pEps];

        fixed (void** pArgs = normArgs)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                fnRmsNorm,
                1, 1, 1,
                256, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(rms_norm)");
        }

        float[] xNormGpu = new float[dim];
        fixed (float* pGpuNorm = xNormGpu)
        {
            gpu.CopyToHost((IntPtr)pGpuNorm, dNormOut, (nuint)(dim * sizeof(float)));
        }

        float maxNormDiff = 0f;
        for (int i = 0; i < dim; i++)
        {
            float d = Math.Abs(xNormCpu[i] - xNormGpu[i]);
            if (d > maxNormDiff) maxNormDiff = d;
        }
        _output.WriteLine($"[RMSNorm] CPU[0]={xNormCpu[0]:F6}, GPU[0]={xNormGpu[0]:F6}, MaxDiff={maxNormDiff:F6}");

        gpu.FreeDevice(dNormIn);
        gpu.FreeDevice(dNormW);
        gpu.FreeDevice(dNormOut);
    }

    [Fact]
    public void TestGemvQ4KAligned()
    {
        if (!File.Exists(ModelPath) || !GpuContext.IsSupported) return;

        using var gguf = GgufFile.Open(ModelPath);
        var qInfo = gguf.Tensors["blk.0.attn_q.weight"];
        int dim = (int)qInfo.Dimensions[0];
        int mRows = 128; // Test 128 rows
        int nb = dim / 256;

        using var gpu = new GpuContext();
        byte[] cubin = KernelCompiler.GetOrCompileKernels("sm_89");
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ4K, module, "gemv_q4_k"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ4KAligned, module, "gemv_q4_k_aligned"), "ModuleGetFunction");

        BlockQ4_K* pW = (BlockQ4_K*)gguf.GetTensorPointer(qInfo);
        float[] x = new float[dim];
        for (int i = 0; i < dim; i++) x[i] = 0.05f * ((i % 11) - 5);

        // Repack into aligned buffers
        byte[] qsAligned = new byte[mRows * nb * 128];
        Half[] scalesAligned = new Half[mRows * nb * 16]; // 8 half2 = 16 Half per block

        BlockQ4_K* pSrc = pW;
        fixed (byte* pQs = qsAligned)
        fixed (Half* pSc = scalesAligned)
        {
            int totalBlocks = mRows * nb;
            for (int b = 0; b < totalBlocks; b++)
            {
                BlockQ4_K* blk = pSrc + b;
                byte* dstQsBlk = pQs + b * 128;
                Half* dstScBlk = pSc + b * 16;

                // Copy 128 bytes qs
                Buffer.MemoryCopy(blk->Qs, dstQsBlk, 128, 128);

                // Unpack scales & mins into FP16 half2
                float d = (float)blk->Delta;
                float dmin = (float)blk->DeltaMin;
                byte* sc = blk->Scales;

                for (int s = 0; s < 8; s++)
                {
                    byte sc_val, m_val;
                    if (s < 4)
                    {
                        sc_val = (byte)(sc[s] & 63);
                        m_val = (byte)(sc[s + 4] & 63);
                    }
                    else
                    {
                        sc_val = (byte)((sc[s + 4] & 0x0F) | ((sc[s - 4] >> 6) << 4));
                        m_val = (byte)((sc[s + 4] >> 4) | ((sc[s] >> 6) << 4));
                    }

                    dstScBlk[s * 2 + 0] = (Half)(d * sc_val);
                    dstScBlk[s * 2 + 1] = (Half)(dmin * m_val);
                }
            }
        }

        // Upload to GPU
        IntPtr dX = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dYStd = gpu.AllocateDevice((nuint)(mRows * sizeof(float)));
        IntPtr dYAln = gpu.AllocateDevice((nuint)(mRows * sizeof(float)));
        IntPtr dWStd = gpu.AllocateDevice((nuint)(mRows * nb * sizeof(BlockQ4_K)));
        IntPtr dWQs = gpu.AllocateDevice((nuint)qsAligned.Length);
        IntPtr dWScales = gpu.AllocateDevice((nuint)(scalesAligned.Length * sizeof(Half)));

        fixed (float* pX = x) gpu.CopyToDevice(dX, (IntPtr)pX, (nuint)(dim * sizeof(float)));
        gpu.CopyToDevice(dWStd, (IntPtr)pW, (nuint)(mRows * nb * sizeof(BlockQ4_K)));
        fixed (byte* pQs = qsAligned) gpu.CopyToDevice(dWQs, (IntPtr)pQs, (nuint)qsAligned.Length);
        fixed (Half* pSc = scalesAligned) gpu.CopyToDevice(dWScales, (IntPtr)pSc, (nuint)(scalesAligned.Length * sizeof(Half)));

        // Launch standard
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + 3) / 4);
        int kCols = dim; int m_rows = mRows;
        void* pDYS = &dYStd; void* pDX = &dX; void* pDWS = &dWStd; void* pK = &kCols; void* pM = &m_rows;
        void*[] argsStd = [pDYS, pDX, pDWS, pK, pM];
        fixed (void** pArgs = argsStd) CuDriver.LaunchKernel(fnGemvQ4K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        // Launch aligned
        void* pDYA = &dYAln; void* pDWQs = &dWQs; void* pDWSc = &dWScales;
        void*[] argsAln = [pDYA, pDX, pDWQs, pDWSc, pK, pM];
        fixed (void** pArgs = argsAln) CuDriver.LaunchKernel(fnGemvQ4KAligned, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        float[] yStd = new float[mRows];
        float[] yAln = new float[mRows];
        fixed (float* p1 = yStd, p2 = yAln)
        {
            gpu.CopyToHost((IntPtr)p1, dYStd, (nuint)(mRows * sizeof(float)));
            gpu.CopyToHost((IntPtr)p2, dYAln, (nuint)(mRows * sizeof(float)));
        }

        float maxDiff = 0f;
        for (int i = 0; i < mRows; i++)
        {
            float diff = Math.Abs(yStd[i] - yAln[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        _output.WriteLine($"[TestGemvQ4KAligned] MaxDiff across {mRows} rows: {maxDiff:F6}");
        Assert.True(maxDiff < 0.001f, $"MaxDiff too large: {maxDiff}");

        // Now benchmark 1,000 iterations of both!
        int iters = 1000;
        CuDriver.CtxSynchronize();

        var swStd = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            fixed (void** pArgs = argsStd) CuDriver.LaunchKernel(fnGemvQ4K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);
        }
        CuDriver.CtxSynchronize();
        swStd.Stop();

        var swAln = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            fixed (void** pArgs = argsAln) CuDriver.LaunchKernel(fnGemvQ4KAligned, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);
        }
        CuDriver.CtxSynchronize();
        swAln.Stop();

        _output.WriteLine($"[Benchmark 128 rows x 3584 cols]: Standard = {swStd.Elapsed.TotalMilliseconds / iters * 1000.0:F2} µs | Aligned = {swAln.Elapsed.TotalMilliseconds / iters * 1000.0:F2} µs ({swStd.Elapsed.TotalMilliseconds / swAln.Elapsed.TotalMilliseconds:F2}x faster!)");

        gpu.FreeDevice(dX); gpu.FreeDevice(dYStd); gpu.FreeDevice(dYAln);
        gpu.FreeDevice(dWStd); gpu.FreeDevice(dWQs); gpu.FreeDevice(dWScales);
    }

    [Fact]
    public void ProfileForwardBreakdown()
    {
        if (!File.Exists(ModelPath) || !GpuContext.IsSupported) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext();
        using var gpuModel = new Qwen2GpuModel(gpu, weights);
        using var kv = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 128);

        float[] logits = new float[weights.VocabSize];

        // Warmup
        gpuModel.Forward(9707, 0, kv, logits);
        CuDriver.CtxSynchronize();

        // Benchmark with and without computeLogits
        var swLayers = System.Diagnostics.Stopwatch.StartNew();
        int iters = 10;
        for (int i = 0; i < iters; i++)
        {
            gpuModel.Forward(9707, i + 1, kv, logits, computeLogits: false);
        }
        CuDriver.CtxSynchronize();
        swLayers.Stop();

        var swAll = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            gpuModel.Forward(9707, i + 1, kv, logits, computeLogits: true);
        }
        CuDriver.CtxSynchronize();
        swAll.Stop();

        double msLayers = swLayers.Elapsed.TotalMilliseconds / iters;
        double msTotal = swAll.Elapsed.TotalMilliseconds / iters;
        double msLmHead = msTotal - msLayers;
        _output.WriteLine($"[Profile] Total: {msTotal:F2} ms ({1000.0/msTotal:F1} t/s) | 28 Layers: {msLayers:F2} ms ({1000.0/msLayers:F1} t/s) | LM Head: {msLmHead:F2} ms");
    }

    [Fact]
    public void DiagnoseLayerByLayer()
    {
        if (!File.Exists(ModelPath) || !GpuContext.IsSupported) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext();
        using var cpuModel = new Qwen2Model(weights);
        using var gpuModel = new Qwen2GpuModel(gpu, weights);

        int dim = weights.EmbeddingLength;
        int ffnDim = weights.FeedForwardLength;
        int qDim = weights.HeadCount * weights.HeadDim;
        int kvDim = weights.HeadCountKv * weights.HeadDim;

        int token = 9707;
        float[] hCpuX = new float[dim];
        float[] hCpuAttnNormX = new float[dim];
        float[] hCpuFfnNormX = new float[dim];
        float[] hCpuQ = new float[qDim];
        float[] hCpuK = new float[kvDim];
        float[] hCpuV = new float[kvDim];
        float[] hCpuAttnProj = new float[dim];
        float[] hCpuGate = new float[ffnDim];
        float[] hCpuUp = new float[ffnDim];
        float[] hCpuFfnAct = new float[ffnDim];
        float[] hCpuFfnOut = new float[dim];

        // 1. CPU Step
        fixed (float* pX = hCpuX, pAttnNormX = hCpuAttnNormX, pFfnNormX = hCpuFfnNormX, pQ = hCpuQ, pKey = hCpuK, pV = hCpuV,
                      pAttnProj = hCpuAttnProj, pGate = hCpuGate, pUp = hCpuUp, pAct = hCpuFfnAct, pFfnOut = hCpuFfnOut)
        {
            QuantKernels.ExtractEmbedding(weights.EmbdType, weights.EmbdWeight, token, pX, dim);
            var l0 = weights.Layers[0];

            QuantKernels.RMSNorm(pX, l0.AttnNormWeight, pAttnNormX, dim, weights.RmsNormEps);
            QuantKernels.MatVecMul(l0.QType, l0.QWeight, pAttnNormX, pQ, dim, qDim);
            for (int i = 0; i < qDim; i++) pQ[i] += l0.QBias[i];

            QuantKernels.MatVecMul(l0.KType, l0.KWeight, pAttnNormX, pKey, dim, kvDim);
            for (int i = 0; i < kvDim; i++) pKey[i] += l0.KBias[i];

            QuantKernels.MatVecMul(l0.VType, l0.VWeight, pAttnNormX, pV, dim, kvDim);
            for (int i = 0; i < kvDim; i++) pV[i] += l0.VBias[i];

            // FFN
            QuantKernels.RMSNorm(pX, l0.FfnNormWeight, pFfnNormX, dim, weights.RmsNormEps);
            QuantKernels.MatVecMul(l0.FfnGateType, l0.FfnGateWeight, pFfnNormX, pGate, dim, ffnDim);
            QuantKernels.MatVecMul(l0.FfnUpType, l0.FfnUpWeight, pFfnNormX, pUp, dim, ffnDim);
            QuantKernels.SwiGLU(pGate, pUp, pAct, ffnDim);
            QuantKernels.MatVecMul(l0.FfnDownType, l0.FfnDownWeight, pAct, pFfnOut, ffnDim, dim);
        }

        // 2. GPU Step - run forward on gpuModel but intercept buffers via reflection or test pointers
        byte[] cubin = KernelCompiler.GetOrCompileKernels("sm_89");
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ4K, module, "gemv_q4_k"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnGemvQ6K, module, "gemv_q6_k"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnRmsNorm, module, "rms_norm_kernel"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnSwiglu, module, "swiglu_kernel"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAddBias, module, "add_bias_kernel"), "ModuleGetFunction");
        CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnVecAdd, module, "vec_add_kernel"), "ModuleGetFunction");

        var lw0 = weights.Layers[0];
        // Allocate device buffers
        IntPtr dX = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dNormX = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        IntPtr dQ = gpu.AllocateDevice((nuint)(qDim * sizeof(float)));
        IntPtr dAttnNorm = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        gpu.CopyToDevice(dAttnNorm, (IntPtr)lw0.AttnNormWeight, (nuint)(dim * sizeof(float)));

        nuint qBytes = (nuint)GgufTypes.GetRowBytes(lw0.QType, dim) * (nuint)qDim;
        IntPtr dQW = gpu.AllocateDevice(qBytes);
        gpu.CopyToDevice(dQW, (IntPtr)lw0.QWeight, qBytes);
        IntPtr dQBias = gpu.AllocateDevice((nuint)(qDim * sizeof(float)));
        gpu.CopyToDevice(dQBias, (IntPtr)lw0.QBias, (nuint)(qDim * sizeof(float)));

        // Copy embedding
        fixed (float* pX = hCpuX) gpu.CopyToDevice(dX, (IntPtr)pX, (nuint)(dim * sizeof(float)));

        // Run RMSNorm
        float eps = weights.RmsNormEps;
        void* pDIn = &dX; void* pDWn = &dAttnNorm; void* pDOut = &dNormX; int nSize = dim; void* pSize = &nSize; void* pEps = &eps;
        void*[] normArgs = [pDIn, pDWn, pDOut, pSize, pEps];
        fixed (void** pArgs = normArgs) CuDriver.LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        float[] hGpuNormX = new float[dim];
        fixed (float* p = hGpuNormX) gpu.CopyToHost((IntPtr)p, dNormX, (nuint)(dim * sizeof(float)));
        float normDiff = 0f;
        for (int i = 0; i < dim; i++) normDiff = Math.Max(normDiff, Math.Abs(hCpuAttnNormX[i] - hGpuNormX[i]));
        _output.WriteLine($"Step 1 [Attn RMSNorm] MaxDiff: {normDiff:F6}");

        // Run Q GEMV
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((qDim + numWarps - 1) / numWarps);
        void* pDY = &dQ; void* pDX = &dNormX; void* pDW = &dQW; int kCols = dim; void* pK = &kCols; int mRows = qDim; void* pM = &mRows;
        void*[] gemvArgs = [pDY, pDX, pDW, pK, pM];
        fixed (void** pArgs = gemvArgs) CuDriver.LaunchKernel(fnGemvQ4K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        // Run Add Bias
        gridSize = (uint)((qDim + 255) / 256);
        void* pBias = &dQBias;
        void*[] biasArgs = [pDY, pBias, pM];
        fixed (void** pArgs = biasArgs) CuDriver.LaunchKernel(fnAddBias, gridSize, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        float[] hGpuQ = new float[qDim];
        fixed (float* p = hGpuQ) gpu.CopyToHost((IntPtr)p, dQ, (nuint)(qDim * sizeof(float)));
        float qDiff = 0f;
        for (int i = 0; i < qDim; i++) qDiff = Math.Max(qDiff, Math.Abs(hCpuQ[i] - hGpuQ[i]));
        _output.WriteLine($"Step 2 [Q GEMV + Bias] MaxDiff: {qDiff:F6}");

        // Now test FFN SwiGLU + Down
        IntPtr dFfnNorm = gpu.AllocateDevice((nuint)(dim * sizeof(float)));
        gpu.CopyToDevice(dFfnNorm, (IntPtr)lw0.FfnNormWeight, (nuint)(dim * sizeof(float)));
        void* pDFfnNorm = &dFfnNorm;
        void*[] ffnNormArgs = [pDIn, pDFfnNorm, pDOut, pSize, pEps];
        fixed (void** pArgs = ffnNormArgs) CuDriver.LaunchKernel(fnRmsNorm, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        nuint gateBytes = (nuint)GgufTypes.GetRowBytes(lw0.FfnGateType, dim) * (nuint)ffnDim;
        IntPtr dGateW = gpu.AllocateDevice(gateBytes);
        gpu.CopyToDevice(dGateW, (IntPtr)lw0.FfnGateWeight, gateBytes);
        nuint upBytes = (nuint)GgufTypes.GetRowBytes(lw0.FfnUpType, dim) * (nuint)ffnDim;
        IntPtr dUpW = gpu.AllocateDevice(upBytes);
        gpu.CopyToDevice(dUpW, (IntPtr)lw0.FfnUpWeight, upBytes);
        nuint downBytes = (nuint)GgufTypes.GetRowBytes(lw0.FfnDownType, ffnDim) * (nuint)dim;
        IntPtr dDownW = gpu.AllocateDevice(downBytes);
        gpu.CopyToDevice(dDownW, (IntPtr)lw0.FfnDownWeight, downBytes);

        IntPtr dGate = gpu.AllocateDevice((nuint)(ffnDim * sizeof(float)));
        IntPtr dUp = gpu.AllocateDevice((nuint)(ffnDim * sizeof(float)));
        IntPtr dAct = gpu.AllocateDevice((nuint)(ffnDim * sizeof(float)));
        IntPtr dFfnOut = gpu.AllocateDevice((nuint)(dim * sizeof(float)));

        // Gate GEMV
        gridSize = (uint)((ffnDim + numWarps - 1) / numWarps);
        void* pDGate = &dGate; mRows = ffnDim;
        void*[] gateArgs = [pDGate, pDX, &dGateW, pK, pM];
        fixed (void** pArgs = gateArgs) CuDriver.LaunchKernel(fnGemvQ4K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        // Up GEMV
        void* pDUp = &dUp;
        void*[] upArgs = [pDUp, pDX, &dUpW, pK, pM];
        fixed (void** pArgs = upArgs) CuDriver.LaunchKernel(fnGemvQ4K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        // SwiGLU
        gridSize = (uint)((ffnDim + 255) / 256);
        void* pDAct = &dAct;
        void*[] swigluArgs = [pDGate, pDUp, pDAct, pM];
        fixed (void** pArgs = swigluArgs) CuDriver.LaunchKernel(fnSwiglu, gridSize, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        // Down GEMV (Q6_K)
        gridSize = (uint)((dim + numWarps - 1) / numWarps);
        void* pDFfnOut = &dFfnOut; int kColsFfn = ffnDim; void* pKFfn = &kColsFfn; int mRowsDim = dim; void* pMDim = &mRowsDim;
        void*[] downArgs = [pDFfnOut, pDAct, &dDownW, pKFfn, pMDim];
        fixed (void** pArgs = downArgs) CuDriver.LaunchKernel(fnGemvQ6K, gridSize, 1, 1, blockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero);

        float[] hGpuFfnOut = new float[dim];
        fixed (float* p = hGpuFfnOut) gpu.CopyToHost((IntPtr)p, dFfnOut, (nuint)(dim * sizeof(float)));
        float ffnDiff = 0f;
        for (int i = 0; i < dim; i++) ffnDiff = Math.Max(ffnDiff, Math.Abs(hCpuFfnOut[i] - hGpuFfnOut[i]));
        _output.WriteLine($"Step 3 [FFN Down GEMV] MaxDiff: {ffnDiff:F6}");
    }

    [Fact]
    public void TestForwardPassCpuVsGpu()
    {
        if (!File.Exists(ModelPath) || !GpuContext.IsSupported)
        {
            _output.WriteLine("Model or GPU not available, skipping.");
            return;
        }

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext();
        using var cpuModel = new Qwen2Model(weights);
        using var gpuModel = new Qwen2GpuModel(gpu, weights);

        using var kvCpu = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 128);
        using var kvGpu = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 128);

        float[] logitsCpu = new float[weights.VocabSize];
        float[] logitsGpu = new float[weights.VocabSize];

        int token = 9707; // "Hello"

        cpuModel.Forward(token, 0, kvCpu, logitsCpu);
        gpuModel.Forward(token, 0, kvGpu, logitsGpu);

        // Find argmax for CPU
        int bestCpuToken = 0;
        float bestCpuLogit = float.MinValue;
        for (int i = 0; i < weights.VocabSize; i++)
        {
            if (logitsCpu[i] > bestCpuLogit)
            {
                bestCpuLogit = logitsCpu[i];
                bestCpuToken = i;
            }
        }

        // Find argmax for GPU
        int bestGpuToken = 0;
        float bestGpuLogit = float.MinValue;
        for (int i = 0; i < weights.VocabSize; i++)
        {
            if (logitsGpu[i] > bestGpuLogit)
            {
                bestGpuLogit = logitsGpu[i];
                bestGpuToken = i;
            }
        }

        _output.WriteLine($"CPU Best Token: {bestCpuToken} (logit={bestCpuLogit:F4})");
        _output.WriteLine($"GPU Best Token: {bestGpuToken} (logit={bestGpuLogit:F4})");

        float maxDiff = 0f;
        for (int i = 0; i < weights.VocabSize; i++)
        {
            float d = Math.Abs(logitsCpu[i] - logitsGpu[i]);
            if (d > maxDiff) maxDiff = d;
        }
        _output.WriteLine($"Max Logits Diff: {maxDiff:F4}");
    }
}
