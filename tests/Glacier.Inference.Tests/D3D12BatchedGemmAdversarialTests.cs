namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;
using Xunit;

[Collection("SequentialGpu")]
public unsafe class D3D12BatchedGemmAdversarialTests
{
    private static readonly int[] BoundaryBatchSizes = [1, 2, 7, 16, 31, 32, 33, 64, 127, 128, 255, 256];

    private static ID3D12RootSignature CreateGemmBatchRootSignature(D3D12Context ctx)
    {
        return ctx.CreateRootSignature(new RootSignatureDescription(
            RootSignatureFlags.None,
            new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
                new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: x
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: residual
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)  // u1: y
            }));
    }

    private static ID3D12RootSignature CreateGemvRootSignature(D3D12Context ctx)
    {
        return ctx.CreateRootSignature(new RootSignatureDescription(
            RootSignatureFlags.None,
            new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
                new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: x
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: residual
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)  // u2: y
            }));
    }

    private static void DispatchGemmBatch(
        D3D12Context ctx,
        ID3D12RootSignature rootSig,
        ID3D12PipelineState pso,
        ID3D12Resource dW,
        ID3D12Resource dX,
        ID3D12Resource dY,
        int kCols,
        int mRows,
        int batchSize)
    {
        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = 0; // has_bias
        pConsts[4] = 0; // has_residual
        pConsts[5] = 1; // has_y
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch((uint)(mRows + 3) / 4, (uint)(batchSize + 31) / 32, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
    }

    private static void DispatchGemv(
        D3D12Context ctx,
        ID3D12RootSignature rootSig,
        ID3D12PipelineState pso,
        ID3D12Resource dW,
        ID3D12Resource dX,
        ID3D12Resource dY,
        int kCols,
        int mRows)
    {
        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = 0; // has_bias
        pConsts[3] = 0; // has_residual
        pConsts[4] = 1; // has_y
        pConsts[5] = 0; // row_offset
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

        cmd.Dispatch((uint)(mRows + 3) / 4, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
    }

    // =========================================================================
    // 1. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (Q4_K)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_Q4K()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;

        var blocks = new BlockQ4_K[mRows];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(1.2f + r * 0.25f);
            blocks[r].DeltaMin = (Half)(0.4f + r * 0.1f);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 4 + 1 + r);
            for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(((i * 7) + r * 3) & 0xFF);
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = (float)Math.Sin(t * 0.1 + c * 0.05);
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        fixed (BlockQ4_K* pB = blocks)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ4_K(pB + r, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.05f,
                            $"Q4_K Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 2. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (Q6_K)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_Q6K()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;

        var blocks = new BlockQ6_K[mRows];
        byte[] aligned = new byte[mRows * 212];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(0.35f + r * 0.05f);
            for (int i = 0; i < 128; i++) blocks[r].Ql[i] = (byte)((i * 3 + r) & 0xFF);
            for (int i = 0; i < 64; i++) blocks[r].Qh[i] = (byte)((i + r) % 4);
            for (int i = 0; i < 16; i++) blocks[r].Scales[i] = (sbyte)(((i + r) % 5) - 2);

            fixed (BlockQ6_K* pB = &blocks[r])
            fixed (byte* pDst = &aligned[r * 212])
            {
                Buffer.MemoryCopy(pB, pDst, 210, 210);
                pDst[210] = 0;
                pDst[211] = 0;
            }
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = (float)Math.Cos(t * 0.08 + c * 0.04);
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 212));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        ctx.CopyToDevice(dW, aligned);
        fixed (BlockQ6_K* pB = blocks)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ6_K(pB + r, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.05f,
                            $"Q6_K Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 3. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (Q8_0)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_Q8_0()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;
        int nb = kCols / 32; // 8 blocks of 32 per row

        var blocks = new BlockQ8_0[mRows * nb];
        byte[] aligned = new byte[mRows * nb * 36];
        for (int r = 0; r < mRows; r++)
        {
            for (int b = 0; b < nb; b++)
            {
                int idx = r * nb + b;
                blocks[idx].Delta = (Half)(0.2f + idx * 0.01f);
                for (int i = 0; i < 32; i++) blocks[idx].Qs[i] = (sbyte)(((i * 5) + idx) % 127 - 64);

                fixed (BlockQ8_0* pB = &blocks[idx])
                fixed (byte* pDst = &aligned[idx * 36])
                {
                    *(ushort*)pDst = *(ushort*)(&pB->Delta);
                    pDst[2] = 0;
                    pDst[3] = 0;
                    Buffer.MemoryCopy(pB->Qs, pDst + 4, 32, 32);
                }
            }
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = (float)((c % 7) - 3) * 0.1f + (t * 0.01f);
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ8_0Batch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * nb * 36));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        ctx.CopyToDevice(dW, aligned);
        fixed (BlockQ8_0* pB = blocks)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ8_0(pB + r * nb, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.05f,
                            $"Q8_0 Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 4. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (FP32)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_FP32()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;

        float[] W = new float[mRows * kCols];
        for (int r = 0; r < mRows; r++)
        {
            for (int c = 0; c < kCols; c++)
            {
                W[r * kCols + c] = ((r * 17 + c) % 23 - 11) * 0.05f;
            }
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = ((t * 7 + c * 3) % 19 - 9) * 0.04f;
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmFp32Batch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * kCols * sizeof(float)));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        fixed (float* pW = W)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dW, (IntPtr)pW, (ulong)(mRows * kCols * sizeof(float)));
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = 0.0f;
                        for (int c = 0; c < kCols; c++)
                        {
                            cpuExpected += W[r * kCols + c] * allX[t * kCols + c];
                        }
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.01f,
                            $"FP32 Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 5. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (Q3_K)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_Q3K()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;

        var blocks = new BlockQ3_K[mRows];
        byte[] aligned = new byte[mRows * 112];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(0.75f + r * 0.1f);
            for (int i = 0; i < 32; i++) blocks[r].Hmask[i] = (byte)(i * 13 + 7 + r);
            for (int i = 0; i < 64; i++) blocks[r].Qs[i] = (byte)(i * 23 + r);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 11 + 3 + r);

            fixed (BlockQ3_K* pB = &blocks[r])
            fixed (byte* pDst = &aligned[r * 112])
            {
                Buffer.MemoryCopy(pB, pDst, 110, 110);
                pDst[110] = 0;
                pDst[111] = 0;
            }
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = ((c % 9) - 4.0f) * 0.05f + (t * 0.005f);
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ3KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 112));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        ctx.CopyToDevice(dW, aligned);
        fixed (BlockQ3_K* pB = blocks)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ3_K(pB + r, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.05f,
                            $"Q3_K Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 6. BATCH SIZE BOUNDARIES & CPU ORACLE PARITY (Q5_K)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_BatchSweep_Q5K()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        const int maxB = 256;

        var blocks = new BlockQ5_K[mRows];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(1.5f + r * 0.2f);
            blocks[r].DeltaMin = (Half)(0.5f + r * 0.1f);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 5 + 1 + r);
            for (int i = 0; i < 32; i++) blocks[r].Qh[i] = (byte)((i + r) % 255);
            for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(((i * 17) + r) & 0xFF);
        }

        float[] allX = new float[maxB * kCols];
        for (int t = 0; t < maxB; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                allX[t * kCols + c] = (t + 1) * 0.05f + (c % 8) * 0.02f;
            }
        }

        using var rootSig = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.GemmQ5KBatch));

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 176));
        using var dX = ctx.CreateDeviceBuffer((ulong)(maxB * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(maxB * mRows * sizeof(float)));

        fixed (BlockQ5_K* pB = blocks)
        fixed (float* pX = allX)
        {
            ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 176));
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(maxB * kCols * sizeof(float)));

            foreach (int B in BoundaryBatchSizes)
            {
                DispatchGemmBatch(ctx, rootSig, pso, dW, dX, dY, kCols, mRows, B);

                float[] gpuOut = new float[B * mRows];
                fixed (float* pGpu = gpuOut)
                {
                    ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));
                }

                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ5_K(pB + r, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        float tol = Math.Max(0.1f, MathF.Abs(cpuExpected) * 0.001f);
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= tol,
                            $"Q5_K Parity Mismatch at B={B}, token={t}, row={r}: cpu={cpuExpected}, gpu={gpuActual}");
                    }
                }
            }
        }
    }

    // =========================================================================
    // 7. NUMERICAL PARITY: Batched GEMM (B=1) vs Single-Token GEMV (All 6 Types)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_GEMV_Parity_AllQuantTypes()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;

        using var sigBatch = CreateGemmBatchRootSignature(ctx);
        using var sigGemv = CreateGemvRootSignature(ctx);

        float[] x = new float[kCols];
        for (int i = 0; i < kCols; i++) x[i] = (float)Math.Sin(i * 0.1);

        using var dX = ctx.CreateDeviceBuffer((ulong)(kCols * sizeof(float)));
        using var dYBatch = ctx.CreateDeviceBuffer((ulong)(mRows * sizeof(float)));
        using var dYGemv = ctx.CreateDeviceBuffer((ulong)(mRows * sizeof(float)));

        fixed (float* pX = x)
        {
            ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(kCols * sizeof(float)));
        }

        // Test 1: Q4_K
        {
            var blocks = new BlockQ4_K[mRows];
            for (int r = 0; r < mRows; r++)
            {
                blocks[r].Delta = (Half)(1.2f + r * 0.1f);
                blocks[r].DeltaMin = (Half)(0.3f + r * 0.05f);
                for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 3 + r);
                for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(((i * 13) + r) & 0xFF);
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
            fixed (BlockQ4_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvQ4K));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"Q4_K GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }

        // Test 2: Q6_K
        {
            var blocks = new BlockQ6_K[mRows];
            byte[] aligned = new byte[mRows * 212];
            for (int r = 0; r < mRows; r++)
            {
                blocks[r].Delta = (Half)(0.4f + r * 0.05f);
                for (int i = 0; i < 128; i++) blocks[r].Ql[i] = (byte)((i * 3 + r) & 0xFF);
                for (int i = 0; i < 64; i++) blocks[r].Qh[i] = (byte)((i + r) % 4);
                for (int i = 0; i < 16; i++) blocks[r].Scales[i] = (sbyte)(((i + r) % 5) - 2);

                fixed (BlockQ6_K* pB = &blocks[r])
                fixed (byte* pDst = &aligned[r * 212])
                {
                    Buffer.MemoryCopy(pB, pDst, 210, 210);
                    pDst[210] = 0;
                    pDst[211] = 0;
                }
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 212));
            ctx.CopyToDevice(dW, aligned);

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvQ6K));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"Q6_K GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }

        // Test 3: Q8_0
        {
            int nb = kCols / 32;
            var blocks = new BlockQ8_0[mRows * nb];
            byte[] aligned = new byte[mRows * nb * 36];
            for (int r = 0; r < mRows; r++)
            {
                for (int b = 0; b < nb; b++)
                {
                    int idx = r * nb + b;
                    blocks[idx].Delta = (Half)(0.2f + idx * 0.01f);
                    for (int i = 0; i < 32; i++) blocks[idx].Qs[i] = (sbyte)(((i * 5) + idx) % 127 - 64);

                    fixed (BlockQ8_0* pB = &blocks[idx])
                    fixed (byte* pDst = &aligned[idx * 36])
                    {
                        *(ushort*)pDst = *(ushort*)(&pB->Delta);
                        pDst[2] = 0;
                        pDst[3] = 0;
                        Buffer.MemoryCopy(pB->Qs, pDst + 4, 32, 32);
                    }
                }
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * nb * 36));
            ctx.CopyToDevice(dW, aligned);

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ8_0Batch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvQ8_0));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"Q8_0 GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }

        // Test 4: FP32
        {
            float[] W = new float[mRows * kCols];
            for (int r = 0; r < mRows; r++)
            {
                for (int c = 0; c < kCols; c++)
                {
                    W[r * kCols + c] = ((r * 11 + c) % 17 - 8) * 0.05f;
                }
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * kCols * sizeof(float)));
            fixed (float* pW = W) ctx.CopyToDevice(dW, (IntPtr)pW, (ulong)(mRows * kCols * sizeof(float)));

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmFp32Batch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvFp32));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"FP32 GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }

        // Test 5: Q3_K
        {
            var blocks = new BlockQ3_K[mRows];
            byte[] aligned = new byte[mRows * 112];
            for (int r = 0; r < mRows; r++)
            {
                blocks[r].Delta = (Half)(0.75f + r * 0.1f);
                for (int i = 0; i < 32; i++) blocks[r].Hmask[i] = (byte)(i * 13 + 7 + r);
                for (int i = 0; i < 64; i++) blocks[r].Qs[i] = (byte)(i * 23 + r);
                for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 11 + 3 + r);

                fixed (BlockQ3_K* pB = &blocks[r])
                fixed (byte* pDst = &aligned[r * 112])
                {
                    Buffer.MemoryCopy(pB, pDst, 110, 110);
                    pDst[110] = 0;
                    pDst[111] = 0;
                }
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 112));
            ctx.CopyToDevice(dW, aligned);

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ3KBatch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvQ3K));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"Q3_K GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }

        // Test 6: Q5_K
        {
            var blocks = new BlockQ5_K[mRows];
            for (int r = 0; r < mRows; r++)
            {
                blocks[r].Delta = (Half)(1.5f + r * 0.2f);
                blocks[r].DeltaMin = (Half)(0.5f + r * 0.1f);
                for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i * 5 + 1 + r);
                for (int i = 0; i < 32; i++) blocks[r].Qh[i] = (byte)((i + r) % 255);
                for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(((i * 17) + r) & 0xFF);
            }
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 176));
            fixed (BlockQ5_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 176));

            using var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ5KBatch));
            using var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(D3D12Shaders.GemvQ5K));

            DispatchGemmBatch(ctx, sigBatch, psoBatch, dW, dX, dYBatch, kCols, mRows, 1);
            DispatchGemv(ctx, sigGemv, psoGemv, dW, dX, dYGemv, kCols, mRows);

            float[] resBatch = new float[mRows];
            float[] resGemv = new float[mRows];
            fixed (float* pb = resBatch, pg = resGemv)
            {
                ctx.CopyToHost((IntPtr)pb, dYBatch, (ulong)(mRows * sizeof(float)));
                ctx.CopyToHost((IntPtr)pg, dYGemv, (ulong)(mRows * sizeof(float)));
            }
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(resBatch[r] - resGemv[r]) <= 0.01f,
                    $"Q5_K GEMV parity mismatch at row {r}: batch={resBatch[r]}, gemv={resGemv[r]}");
            }
        }
    }

    // =========================================================================
    // 8. EDGE CASE: Zero Blocks across all formats must yield exact 0.0f
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_EdgeCase_ZeroBlocks()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int B = 7;

        using var sigBatch = CreateGemmBatchRootSignature(ctx);

        float[] x = new float[B * kCols];
        for (int i = 0; i < x.Length; i++) x[i] = 1.0f;

        using var dX = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));
        fixed (float* pX = x) ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(B * kCols * sizeof(float)));

        // Test Q4_K all zeros
        {
            byte[] zeroBytes = new byte[mRows * 144];
            using var dW = ctx.CreateDeviceBuffer((ulong)zeroBytes.Length);
            ctx.CopyToDevice(dW, zeroBytes);

            using var pso = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
            DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

            float[] gpuOut = new float[B * mRows];
            fixed (float* pGpu = gpuOut) ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));

            for (int i = 0; i < gpuOut.Length; i++)
            {
                Assert.Equal(0.0f, gpuOut[i]);
            }
        }

        // Test FP32 all zeros
        {
            byte[] zeroBytes = new byte[mRows * kCols * sizeof(float)];
            using var dW = ctx.CreateDeviceBuffer((ulong)zeroBytes.Length);
            ctx.CopyToDevice(dW, zeroBytes);

            using var pso = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmFp32Batch));
            DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

            float[] gpuOut = new float[B * mRows];
            fixed (float* pGpu = gpuOut) ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));

            for (int i = 0; i < gpuOut.Length; i++)
            {
                Assert.Equal(0.0f, gpuOut[i]);
            }
        }
    }

    // =========================================================================
    // 9. EDGE CASE: Subnormal scales and extreme dynamic ranges
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_EdgeCase_SubnormalScales()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int B = 5;

        using var sigBatch = CreateGemmBatchRootSignature(ctx);

        // 1. Subnormal Half scale in Q4_K
        var blocks = new BlockQ4_K[mRows];
        for (int r = 0; r < mRows; r++)
        {
            // Smallest positive subnormal Half is 2^-24 ~= 5.96e-8
            blocks[r].Delta = BitConverter.UInt16BitsToHalf(0x0001);
            blocks[r].DeltaMin = BitConverter.UInt16BitsToHalf(0x0002);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i + 1);
            for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)(i % 255);
        }

        float[] x = new float[B * kCols];
        for (int i = 0; i < x.Length; i++) x[i] = 1.0f;

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
        using var dX = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));

        fixed (BlockQ4_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));
        fixed (float* pX = x) ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(B * kCols * sizeof(float)));

        using var pso = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
        DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

        float[] gpuOut = new float[B * mRows];
        fixed (float* pGpu = gpuOut) ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));

        for (int i = 0; i < gpuOut.Length; i++)
        {
            Assert.False(float.IsNaN(gpuOut[i]), $"Subnormal produced NaN at index {i}");
            Assert.False(float.IsInfinity(gpuOut[i]), $"Subnormal produced Inf at index {i}");
        }
    }

    [Fact]
    public void D3D12_Adversarial_EdgeCase_ExtremeDynamicRange()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int B = 5;

        using var sigBatch = CreateGemmBatchRootSignature(ctx);

        // Extreme dynamic range in FP32
        float[] extremeW = new float[mRows * kCols];
        float[] extremeX = new float[B * kCols];
        for (int i = 0; i < extremeW.Length; i++) extremeW[i] = (i % 2 == 0) ? 1e4f : -1e4f;
        for (int i = 0; i < extremeX.Length; i++) extremeX[i] = (i % 2 == 0) ? 1e-4f : 1e-4f;

        using var dWExt = ctx.CreateDeviceBuffer((ulong)(mRows * kCols * sizeof(float)));
        using var dXExt = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
        using var dYExt = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));

        fixed (float* pEW = extremeW, pEX = extremeX)
        {
            ctx.CopyToDevice(dWExt, (IntPtr)pEW, (ulong)(mRows * kCols * sizeof(float)));
            ctx.CopyToDevice(dXExt, (IntPtr)pEX, (ulong)(B * kCols * sizeof(float)));
        }

        using var psoFp32 = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmFp32Batch));
        DispatchGemmBatch(ctx, sigBatch, psoFp32, dWExt, dXExt, dYExt, kCols, mRows, B);

        float[] gpuExtOut = new float[B * mRows];
        fixed (float* pGpu = gpuExtOut) ctx.CopyToHost((IntPtr)pGpu, dYExt, (ulong)(B * mRows * sizeof(float)));

        for (int i = 0; i < gpuExtOut.Length; i++)
        {
            Assert.False(float.IsNaN(gpuExtOut[i]), $"Extreme dynamic range produced NaN at index {i}");
            Assert.False(float.IsInfinity(gpuExtOut[i]), $"Extreme dynamic range produced Inf at index {i}");
        }
    }

    // =========================================================================
    // 10. EDGE CASE: LDS NaN / Inf Token Isolation (No cross-token pollution)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_EdgeCase_LDS_NaN_Inf_Isolation()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;
        int mRows = 4;
        int B = 4; // Tokens 0, 1, 2, 3

        using var sigBatch = CreateGemmBatchRootSignature(ctx);

        var blocks = new BlockQ4_K[mRows];
        for (int r = 0; r < mRows; r++)
        {
            blocks[r].Delta = (Half)(1.0f + r * 0.1f);
            blocks[r].DeltaMin = (Half)(0.2f + r * 0.05f);
            for (int i = 0; i < 12; i++) blocks[r].Scales[i] = (byte)(i + 1);
            for (int i = 0; i < 128; i++) blocks[r].Qs[i] = (byte)((i * 11) & 0xFF);
        }

        float[] x = new float[B * kCols];
        // Token 0: Contains Poison (NaN and +Inf)
        for (int c = 0; c < kCols; c++)
        {
            x[0 * kCols + c] = (c % 2 == 0) ? float.NaN : float.PositiveInfinity;
        }
        // Tokens 1, 2, 3: Clean, normal activations
        for (int t = 1; t < B; t++)
        {
            for (int c = 0; c < kCols; c++)
            {
                x[t * kCols + c] = (t * 0.2f) + (c * 0.01f);
            }
        }

        using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
        using var dX = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
        using var dY = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));

        fixed (BlockQ4_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));
        fixed (float* pX = x) ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(B * kCols * sizeof(float)));

        using var pso = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
        DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

        float[] gpuOut = new float[B * mRows];
        fixed (float* pGpu = gpuOut) ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));

        // Verify Token 0 received NaN/Inf (poisoned as expected)
        for (int r = 0; r < mRows; r++)
        {
            Assert.True(float.IsNaN(gpuOut[0 * mRows + r]) || float.IsInfinity(gpuOut[0 * mRows + r]),
                "Token 0 was expected to reflect poison NaN/Inf");
        }

        // Verify Tokens 1, 2, 3 were 100% ISOLATED and match CPU oracle perfectly!
        fixed (BlockQ4_K* pB = blocks)
        fixed (float* pX = x)
        {
            for (int t = 1; t < B; t++)
            {
                float* xToken = pX + t * kCols;
                for (int r = 0; r < mRows; r++)
                {
                    float expected = QuantKernels.VecDotQ4_K(pB + r, xToken, kCols);
                    float actual = gpuOut[t * mRows + r];
                    Assert.False(float.IsNaN(actual), $"Token {t} row {r} was corrupted by Token 0 NaN!");
                    Assert.False(float.IsInfinity(actual), $"Token {t} row {r} was corrupted by Token 0 Inf!");
                    Assert.True(MathF.Abs(expected - actual) <= 0.05f,
                        $"Token {t} row {r} numerical mismatch: expected={expected}, actual={actual}");
                }
            }
        }
    }

    // =========================================================================
    // 11. MEMORY SAFETY: Exact unpadded buffer bounds (No OOB overrun / MMU faults)
    // =========================================================================
    [Fact]
    public void D3D12_Adversarial_MemorySafety_ExactBufferBounds_NoOverrun()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        int kCols = 256;

        using var sigBatch = CreateGemmBatchRootSignature(ctx);
        using var pso = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));

        // Subcase A: Extreme minimal allocation (B=1, M=1)
        {
            int mRows = 1;
            int B = 1;
            var blocks = new BlockQ4_K[mRows];
            blocks[0].Delta = (Half)1.0f;
            blocks[0].DeltaMin = (Half)0.2f;

            float[] x = new float[B * kCols];
            for (int i = 0; i < kCols; i++) x[i] = 1.0f;

            // Strict exact buffer sizes: dW = 144B, dX = 1024B, dY = 4B
            using var dW = ctx.CreateDeviceBuffer(144);
            using var dX = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
            using var dY = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));

            fixed (BlockQ4_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, 144);
            fixed (float* pX = x) ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(B * kCols * sizeof(float)));

            DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

            float gpuResult = 0f;
            ctx.CopyToHost((IntPtr)(&gpuResult), dY, sizeof(float));

            fixed (BlockQ4_K* pB = blocks)
            fixed (float* pX = x)
            {
                float cpuExpected = QuantKernels.VecDotQ4_K(pB, pX, kCols);
                Assert.True(MathF.Abs(cpuExpected - gpuResult) <= 0.05f);
            }
        }

        // Subcase B: Non-power-of-two, non-warp aligned (B=7, M=3)
        {
            int mRows = 3;
            int B = 7;
            var blocks = new BlockQ4_K[mRows];
            for (int r = 0; r < mRows; r++)
            {
                blocks[r].Delta = (Half)(1.1f + r * 0.1f);
                blocks[r].DeltaMin = (Half)(0.3f + r * 0.05f);
            }

            float[] x = new float[B * kCols];
            for (int i = 0; i < x.Length; i++) x[i] = (i % 7) * 0.1f;

            // Strict exact buffer sizes: dW = 3 * 144 = 432B, dX = 7 * 256 * 4 = 7168B, dY = 7 * 3 * 4 = 84B
            using var dW = ctx.CreateDeviceBuffer((ulong)(mRows * 144));
            using var dX = ctx.CreateDeviceBuffer((ulong)(B * kCols * sizeof(float)));
            using var dY = ctx.CreateDeviceBuffer((ulong)(B * mRows * sizeof(float)));

            fixed (BlockQ4_K* pB = blocks) ctx.CopyToDevice(dW, (IntPtr)pB, (ulong)(mRows * 144));
            fixed (float* pX = x) ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(B * kCols * sizeof(float)));

            DispatchGemmBatch(ctx, sigBatch, pso, dW, dX, dY, kCols, mRows, B);

            float[] gpuOut = new float[B * mRows];
            fixed (float* pGpu = gpuOut) ctx.CopyToHost((IntPtr)pGpu, dY, (ulong)(B * mRows * sizeof(float)));

            fixed (BlockQ4_K* pB = blocks)
            fixed (float* pX = x)
            {
                for (int t = 0; t < B; t++)
                {
                    float* xToken = pX + t * kCols;
                    for (int r = 0; r < mRows; r++)
                    {
                        float cpuExpected = QuantKernels.VecDotQ4_K(pB + r, xToken, kCols);
                        float gpuActual = gpuOut[t * mRows + r];
                        Assert.True(MathF.Abs(cpuExpected - gpuActual) <= 0.05f);
                    }
                }
            }
        }
    }
}
