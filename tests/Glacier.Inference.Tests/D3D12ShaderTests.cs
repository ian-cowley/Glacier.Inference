namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.D3DCompiler;
using Xunit;

public class D3D12ShaderTests
{
    [Fact]
    public void All_D3D12_HLSL_Shaders_Compile_Successfully()
    {
        if (!OperatingSystem.IsWindows()) return;

        var gemvQ4K = Compiler.Compile(D3D12Shaders.GemvQ4K, "main", "gemv_q4_k.hlsl", "cs_5_0");
        Assert.False(gemvQ4K.IsEmpty);

        var gemvQ6K = Compiler.Compile(D3D12Shaders.GemvQ6K, "main", "gemv_q6_k.hlsl", "cs_5_0");
        Assert.False(gemvQ6K.IsEmpty);

        var gemvFp32 = Compiler.Compile(D3D12Shaders.GemvFp32, "main", "gemv_fp32.hlsl", "cs_5_0");
        Assert.False(gemvFp32.IsEmpty);

        var rmsNorm = Compiler.Compile(D3D12Shaders.RmsNorm, "main", "rms_norm.hlsl", "cs_5_0");
        Assert.False(rmsNorm.IsEmpty);

        var swiglu = Compiler.Compile(D3D12Shaders.SwiGLU, "main", "swiglu.hlsl", "cs_5_0");
        Assert.False(swiglu.IsEmpty);

        var vecAdd = Compiler.Compile(D3D12Shaders.VecAdd, "main", "vec_add.hlsl", "cs_5_0");
        Assert.False(vecAdd.IsEmpty);

        var rope = Compiler.Compile(D3D12Shaders.RoPE, "main", "rope.hlsl", "cs_5_0");
        Assert.False(rope.IsEmpty);

        var kvCache = Compiler.Compile(D3D12Shaders.KvCacheStore, "main", "kv_store.hlsl", "cs_5_0");
        Assert.False(kvCache.IsEmpty);

        var attnGqa = Compiler.Compile(D3D12Shaders.AttentionGqa, "main", "attention_gqa.hlsl", "cs_5_0");
        Assert.False(attnGqa.IsEmpty);

        var argmax = Compiler.Compile(D3D12Shaders.Argmax, "main", "argmax.hlsl", "cs_5_0");
        Assert.False(argmax.IsEmpty);

        var gemmQ4KBatch = Compiler.Compile(D3D12Shaders.GemmQ4KBatch, "main", "gemm_q4_k_batch.hlsl", "cs_5_0");
        Assert.False(gemmQ4KBatch.IsEmpty);

        var gemmQ6KBatch = Compiler.Compile(D3D12Shaders.GemmQ6KBatch, "main", "gemm_q6_k_batch.hlsl", "cs_5_0");
        Assert.False(gemmQ6KBatch.IsEmpty);

        var gemmFp32Batch = Compiler.Compile(D3D12Shaders.GemmFp32Batch, "main", "gemm_fp32_batch.hlsl", "cs_5_0");
        Assert.False(gemmFp32Batch.IsEmpty);

        var rmsNormBatch = Compiler.Compile(D3D12Shaders.RmsNormBatch, "main", "rms_norm_batch.hlsl", "cs_5_0");
        Assert.False(rmsNormBatch.IsEmpty);

        var ropeBatch = Compiler.Compile(D3D12Shaders.RoPEBatch, "main", "rope_batch.hlsl", "cs_5_0");
        Assert.False(ropeBatch.IsEmpty);

        var kvCacheBatch = Compiler.Compile(D3D12Shaders.KvCacheStoreBatch, "main", "kv_store_batch.hlsl", "cs_5_0");
        Assert.False(kvCacheBatch.IsEmpty);

        var attnBatch = Compiler.Compile(D3D12Shaders.AttentionBatch, "main", "attention_batch.hlsl", "cs_5_0");
        Assert.False(attnBatch.IsEmpty);
    }

    [Fact]
    public unsafe void D3D12_GemvQ4K_Matches_CPU()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 1;

        // Create 1 BlockQ4_K
        var block = new Glacier.Inference.Quant.BlockQ4_K();
        block.Delta = (Half)1.0f;
        block.DeltaMin = (Half)0.5f;

        // Set scales
        for (int i = 0; i < 12; i++) block.Scales[i] = (byte)(i + 1);

        // Set qs
        for (int i = 0; i < 128; i++) block.Scales[i % 12] = 10;
        for (int i = 0; i < 128; i++) block.Qs[i] = (byte)((i & 0x0F) | (((i + 1) & 0x0F) << 4));

        float[] x = new float[kCols];
        for (int i = 0; i < kCols; i++) x[i] = 1.0f;

        // Compute on CPU
        float cpuResult;
        fixed (float* pX = x)
        {
            cpuResult = Glacier.Inference.Quant.QuantKernels.VecDotQ4_K(&block, pX, kCols);
        }

        var rootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
            Vortice.Direct3D12.RootSignatureFlags.None,
            new Vortice.Direct3D12.RootParameter[]
            {
                new(new Vortice.Direct3D12.RootConstants(0, 0, 5), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All)
            }));
        var psoGemv = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemvQ4K));

        using var dW = ctx.CreateDeviceBuffer(144);
        using var dX = ctx.CreateDeviceBuffer((ulong)(kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer(sizeof(float));

        fixed (float* pX = x)
        {
            BlockQ4_K* pB = &block;
            ctx.CopyToDevice(dW, (IntPtr)pB, 144);
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(kCols * sizeof(float)));
        }

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(psoGemv);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = 0;
        pConsts[3] = 0;
        pConsts[4] = 1; // has_y
        cmd.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, 0);
        cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, 0);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch(1, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float gpuResult = 0;
        ctx.CopyToHost((IntPtr)(&gpuResult), dY, sizeof(float));

        Assert.Equal(cpuResult, gpuResult, 0.01f);
    }

    [Fact]
    public unsafe void D3D12_GemvQ6K_Matches_CPU()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 1;

        var block = new BlockQ6_K();
        block.Delta = (Half)0.25f;

        for (int i = 0; i < 128; i++) block.Ql[i] = (byte)(i * 3);
        for (int i = 0; i < 64; i++) block.Qh[i] = (byte)(i % 4);
        for (int i = 0; i < 16; i++) block.Scales[i] = (sbyte)((i % 5) - 2);

        float[] x = new float[kCols];
        for (int i = 0; i < kCols; i++) x[i] = 1.0f;

        float cpuResult;
        fixed (float* pX = x)
        {
            cpuResult = QuantKernels.VecDotQ6_K(&block, pX, kCols);
        }

        // Align block to 212 bytes
        byte[] aligned = new byte[212];
        BlockQ6_K* pB = &block;
        fixed (byte* pDst = aligned)
        {
            Buffer.MemoryCopy((void*)pB, (void*)pDst, 210, 210);
            pDst[210] = 0;
            pDst[211] = 0;
        }

        var rootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
            Vortice.Direct3D12.RootSignatureFlags.None,
            new Vortice.Direct3D12.RootParameter[]
            {
                new(new Vortice.Direct3D12.RootConstants(0, 0, 5), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All)
            }));
        var psoGemv = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemvQ6K));

        using var dW = ctx.CreateDeviceBuffer(212);
        using var dX = ctx.CreateDeviceBuffer((ulong)(kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer(sizeof(float));

        fixed (float* pX = x)
        {
            ctx.CopyToDevice(dW, aligned);
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(kCols * sizeof(float)));
        }

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(psoGemv);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = 0;
        pConsts[3] = 0;
        pConsts[4] = 1; // has_y
        cmd.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, 0);
        cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, 0);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch(1, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float gpuResult = 0;
        ctx.CopyToHost((IntPtr)(&gpuResult), dY, sizeof(float));

        Assert.Equal(cpuResult, gpuResult, 0.05f);
    }

    [Fact]
    public unsafe void D3D12_GemmQ4KBatch_Matches_CPU()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int batchSize = 5;

        // Create 4 BlockQ4_K (one per row)
        var blocks = new BlockQ4_K[mRows];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(1.0f + r * 0.2f);
            blocks[r].DeltaMin = (Half)(0.5f + r * 0.1f);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(10 + r);
            for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(((i + r) & 0x0F) | ((((i + r) + 1) & 0x0F) << 4));
        }

        // Create input X for batch
        float[] x = new float[batchSize * kCols];
        for (int t = 0; t < batchSize; t++)
        {
            for (int i = 0; i < kCols; i++)
            {
                x[t * kCols + i] = (t + 1) * 0.1f + (i % 8) * 0.02f;
            }
        }

        // Compute expected CPU results [batchSize, mRows]
        float[] cpuResults = new float[batchSize * mRows];
        fixed (BlockQ4_K* pB = blocks)
        fixed (float* pX = x)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* xToken = pX + t * kCols;
                for (int r = 0; r < mRows; r++)
                {
                    cpuResults[t * mRows + r] = QuantKernels.VecDotQ4_K(pB + r, xToken, kCols);
                }
            }
        }

        var rootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
            Vortice.Direct3D12.RootSignatureFlags.None,
            new Vortice.Direct3D12.RootParameter[]
            {
                new(new Vortice.Direct3D12.RootConstants(0, 0, 6), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All)
            }));
        var psoGemm = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
        using var dX = ctx.CreateDeviceBuffer((ulong)(batchSize * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(batchSize * mRows * sizeof(float)));

        fixed (BlockQ4_K* pB = blocks)
        fixed (float* pX = x)
        {
            ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(batchSize * kCols * sizeof(float)));
        }

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(psoGemm);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = 0; // has_bias
        pConsts[4] = 0; // has_residual
        pConsts[5] = 1; // has_y
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, 0);
        cmd.SetComputeRootShaderResourceView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, 0);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch((uint)(mRows + 3) / 4, (uint)(batchSize + 31) / 32, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] gpuResults = new float[batchSize * mRows];
        fixed (float* pGpu = gpuResults)
        {
            ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(batchSize * mRows * sizeof(float)));
        }

        for (int t = 0; t < batchSize; t++)
        {
            for (int r = 0; r < mRows; r++)
            {
                int idx = t * mRows + r;
                Assert.Equal(cpuResults[idx], gpuResults[idx], 0.05f);
            }
        }
    }

    [Fact]
    public unsafe void D3D12_GemmQ6KBatch_Matches_CPU()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int batchSize = 5;

        // Create 4 BlockQ6_K (aligned to 212 bytes each)
        var blocks = new BlockQ6_K[mRows];
        byte[] aligned = new byte[mRows * 212];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(0.25f + r * 0.05f);
            for (int i = 0; i < 128; i++) blocks[r].Ql[i] = (byte)((i * 3 + r) & 0xFF);
            for (int i = 0; i < 64; i++) blocks[r].Qh[i] = (byte)((i + r) % 4);
            for (int i = 0; i < 16; i++) blocks[r].Scales[i] = (sbyte)(((i + r) % 5) - 2);

            fixed (BlockQ6_K* pB = &blocks[r])
            fixed (byte* pDst = &aligned[r * 212])
            {
                Buffer.MemoryCopy((void*)pB, (void*)pDst, 210, 210);
                pDst[210] = 0;
                pDst[211] = 0;
            }
        }

        float[] x = new float[batchSize * kCols];
        for (int t = 0; t < batchSize; t++)
        {
            for (int i = 0; i < kCols; i++)
            {
                x[t * kCols + i] = (t + 1) * 0.1f + (i % 8) * 0.02f;
            }
        }

        float[] cpuResults = new float[batchSize * mRows];
        fixed (BlockQ6_K* pB = blocks)
        fixed (float* pX = x)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* xToken = pX + t * kCols;
                for (int r = 0; r < mRows; r++)
                {
                    cpuResults[t * mRows + r] = QuantKernels.VecDotQ6_K(pB + r, xToken, kCols);
                }
            }
        }

        var rootSig = ctx.CreateRootSignature(new Vortice.Direct3D12.RootSignatureDescription(
            Vortice.Direct3D12.RootSignatureFlags.None,
            new Vortice.Direct3D12.RootParameter[]
            {
                new(new Vortice.Direct3D12.RootConstants(0, 0, 6), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.ShaderResourceView, new Vortice.Direct3D12.RootDescriptor(2, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(0, 0), Vortice.Direct3D12.ShaderVisibility.All),
                new(Vortice.Direct3D12.RootParameterType.UnorderedAccessView, new Vortice.Direct3D12.RootDescriptor(1, 0), Vortice.Direct3D12.ShaderVisibility.All)
            }));
        var psoGemm = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)aligned.Length);
        using var dX = ctx.CreateDeviceBuffer((ulong)(batchSize * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(batchSize * mRows * sizeof(float)));

        ctx.CopyToDevice(dW, aligned);
        fixed (float* pX = x)
        {
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(batchSize * kCols * sizeof(float)));
        }

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(psoGemm);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = 0; // has_bias
        pConsts[4] = 0; // has_residual
        pConsts[5] = 1; // has_y
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, 0);
        cmd.SetComputeRootShaderResourceView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, 0);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch((uint)(mRows + 3) / 4, (uint)(batchSize + 31) / 32, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] gpuResults = new float[batchSize * mRows];
        fixed (float* pGpu = gpuResults)
        {
            ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(batchSize * mRows * sizeof(float)));
        }

        for (int t = 0; t < batchSize; t++)
        {
            for (int r = 0; r < mRows; r++)
            {
                int idx = t * mRows + r;
                Assert.Equal(cpuResults[idx], gpuResults[idx], 0.05f);
            }
        }
    }

    [Fact]
    public void D3D12_Qwen2_ForwardPass_Matches_CPU()
    {
        if (!OperatingSystem.IsWindows()) return;
        string modelPath = @"C:\Users\spuri\.ollama\models\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4";
        if (!File.Exists(modelPath)) return;

        using var gguf = GgufFile.Open(modelPath);
        var weights = new ModelWeights(gguf);

        // Run CPU forward pass
        var cpuModel = new Qwen2Model(weights, 128);
        using var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 128);
        float[] cpuLogits = new float[weights.VocabSize];
        cpuModel.Forward(token: 9707, pos: 0, kvCache, cpuLogits.AsSpan(), computeLogits: true);

        // Run GPU forward pass
        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen2D3D12Model(ctx, weights, 128);
        float[] gpuLogits = new float[weights.VocabSize];
        gpuModel.Forward(token: 9707, pos: 0, gpuLogits.AsSpan(), computeLogits: true);

        Console.WriteLine($"Model Arch Details: Dim={weights.EmbeddingLength}, FfnDim={weights.FeedForwardLength}, Blocks={weights.BlockCount}, Heads={weights.HeadCount}, HeadsKv={weights.HeadCountKv}, HeadDim={weights.HeadDim}");
        Console.WriteLine($"Layer 0 Types: Q={weights.Layers[0].QType}, K={weights.Layers[0].KType}, V={weights.Layers[0].VType}, AttnOut={weights.Layers[0].AttnOutType}, FfnGate={weights.Layers[0].FfnGateType}, FfnUp={weights.Layers[0].FfnUpType}, FfnDown={weights.Layers[0].FfnDownType}, Out={weights.OutType}");

        // Compare top token
        int cpuTop = 0; float cpuMax = cpuLogits[0];
        for (int i = 1; i < weights.VocabSize; i++)
            if (cpuLogits[i] > cpuMax) { cpuMax = cpuLogits[i]; cpuTop = i; }

        int gpuTop = 0; float gpuMax = gpuLogits[0];
        for (int i = 1; i < weights.VocabSize; i++)
            if (gpuLogits[i] > gpuMax) { gpuMax = gpuLogits[i]; gpuTop = i; }

        Console.WriteLine($"CPU Top: {cpuTop} (val={cpuMax}), GPU Top: {gpuTop} (val={gpuMax})");
        Assert.Equal(cpuTop, gpuTop);
    }

    [Fact]
    public unsafe void D3D12_Profile_ForwardPass_Timing()
    {
        if (!OperatingSystem.IsWindows()) return;
        string modelPath = @"C:\Users\spuri\.ollama\models\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4";
        if (!File.Exists(modelPath)) return;

        using var gguf = GgufFile.Open(modelPath);
        var weights = new ModelWeights(gguf);

        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen2D3D12Model(ctx, weights, 128);
        float[] gpuLogits = new float[weights.VocabSize];

        // Warmup
        gpuModel.Forward(token: 9707, pos: 0, gpuLogits.AsSpan(), computeLogits: true);

        // Measure with computeLogits = false (no LM head, no logits readback)
        var swNoLogits = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 1; i <= 10; i++)
        {
            gpuModel.Forward(token: 9707, pos: i, Span<float>.Empty, computeLogits: false);
        }
        swNoLogits.Stop();

        // Measure with computeLogits = true (with LM head and logits readback)
        var swWithLogits = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 11; i <= 20; i++)
        {
            gpuModel.Forward(token: 9707, pos: i, gpuLogits.AsSpan(), computeLogits: true);
        }
        swWithLogits.Stop();

        Console.WriteLine($"[GPU PROFILE NO LOGITS] 10 tokens took {swNoLogits.ElapsedMilliseconds} ms: {swNoLogits.Elapsed.TotalMilliseconds / 10.0:F2} ms/token ({1000.0 / (swNoLogits.Elapsed.TotalMilliseconds / 10.0):F2} tok/s)");
        Console.WriteLine($"[GPU PROFILE WITH LOGITS] 10 tokens took {swWithLogits.ElapsedMilliseconds} ms: {swWithLogits.Elapsed.TotalMilliseconds / 10.0:F2} ms/token ({1000.0 / (swWithLogits.Elapsed.TotalMilliseconds / 10.0):F2} tok/s)");
        Console.WriteLine($"[GPU PROFILE TIMINGS BREAKDOWN] Last token: CPU Record = {gpuModel.LastTimings.RecordMs:F2} ms, GPU Exec + Wait = {gpuModel.LastTimings.GpuMs:F2} ms");
        double lmHeadMs = (swWithLogits.Elapsed.TotalMilliseconds - swNoLogits.Elapsed.TotalMilliseconds) / 10.0;
        Console.WriteLine($"[GPU PROFILE LM HEAD] LM Head + Readback overhead: {lmHeadMs:F2} ms/token");
    }

    [Fact]
    public unsafe void D3D12_ForwardBatch_Matches_Sequential()
    {
        if (!OperatingSystem.IsWindows()) return;
        string modelPath = @"C:\Users\spuri\.ollama\models\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4";
        if (!File.Exists(modelPath)) return;

        using var gguf = GgufFile.Open(modelPath);
        var weights = new ModelWeights(gguf);

        int[] prompt = [9707, 311, 279, 14816, 374]; // 5 tokens

        // 1. Sequential run
        using var ctx1 = new D3D12Context();
        using var modelSeq = new Qwen2D3D12Model(ctx1, weights, 128);
        float[] logitsSeq = new float[weights.VocabSize];
        for (int i = 0; i < prompt.Length; i++)
        {
            bool isLast = (i == prompt.Length - 1);
            modelSeq.Forward(prompt[i], i, isLast ? logitsSeq.AsSpan() : Span<float>.Empty, isLast);
        }

        // 2. Batched run
        using var ctx2 = new D3D12Context();
        using var modelBatch = new Qwen2D3D12Model(ctx2, weights, 128);
        float[] logitsBatch = new float[weights.VocabSize];
        modelBatch.ForwardBatch(prompt, 0, logitsBatch.AsSpan(), computeLogits: true);

        // Compare top tokens
        int topSeq = 0; float maxSeq = logitsSeq[0];
        for (int i = 1; i < weights.VocabSize; i++)
            if (logitsSeq[i] > maxSeq) { maxSeq = logitsSeq[i]; topSeq = i; }

        int topBatch = 0; float maxBatch = logitsBatch[0];
        for (int i = 1; i < weights.VocabSize; i++)
            if (logitsBatch[i] > maxBatch) { maxBatch = logitsBatch[i]; topBatch = i; }

        Console.WriteLine($"Sequential Top: {topSeq} ({maxSeq:F4}), Batched Top: {topBatch} ({maxBatch:F4})");
        Assert.Equal(topSeq, topBatch);
        Assert.Equal(maxSeq, maxBatch, 0.01f);
    }

    [Fact]
    public unsafe void D3D12_Profile_ForwardBatch_Kernels()
    {
        if (!OperatingSystem.IsWindows()) return;
        string modelPath = @"C:\Users\spuri\.ollama\models\blobs\sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4";
        if (!File.Exists(modelPath)) return;

        using var gguf = GgufFile.Open(modelPath);
        var weights = new ModelWeights(gguf);

        int[] prompt = new int[30];
        for (int i = 0; i < 30; i++) prompt[i] = 9707 + i;

        using var ctx = new D3D12Context();
        using var model = new Qwen2D3D12Model(ctx, weights, 128);
        float[] logits = new float[weights.VocabSize];

        // Warmup
        model.ForwardBatch(prompt.AsSpan(0, 5), 0, logits.AsSpan(), computeLogits: true);

        // Run 3 times to get accurate GPU execution time
        for (int run = 0; run < 3; run++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            model.ForwardBatch(prompt, 0, logits.AsSpan(), computeLogits: true);
            sw.Stop();
            Console.WriteLine($"[RUN {run + 1}] 30 tokens evaluated in {sw.ElapsedMilliseconds} ms ({30.0 / sw.Elapsed.TotalSeconds:F1} tok/s)");
        }
    }
}
