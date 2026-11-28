using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Vortice.Direct3D12;
using Xunit;

namespace Glacier.Inference.Tests;

[Collection("SequentialGpu")]
[Trait("Tier", "Tier5")]
public sealed unsafe class Tier5AdversarialHardeningTests
{
    // =========================================================================
    // 1. BUG-CATCHING REGRESSION TESTS (B1..B7 + D3D12 odd head_dim & +Inf)
    // =========================================================================

    [Fact]
    public void T5_B1_CpuKvCache_Q4_0_OutlierAboveHalfMaxTimes7_DoesNotPoisonFiniteBlockToNaN()
    {
        const int dim = 32;
        using var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: dim, maxSeqLen: 4, precision: KvCachePrecision.Q4_0);

        float[] k = new float[dim];
        Array.Fill(k, 1.0f);
        k[0] = 500_000.0f;

        cache.Store(0, 0, k, k);

        float[] readBack = new float[dim];
        cache.ReadKey(0, 0, 0, readBack);
        Assert.All(readBack, v => Assert.True(float.IsFinite(v), $"Expected finite Q4_0 dequantized value, got {v}."));
        Assert.True(readBack[0] > 400_000.0f, $"Expected positive clamped max value at index 0, got {readBack[0]}.");

        float[] q = new float[dim];
        Array.Fill(q, 1.0f);
        float[] acc = new float[dim];
        fixed (float* pQ = q, pAcc = acc)
        {
            float dot = cache.DotKey(0, 0, 0, pQ);
            Assert.True(float.IsFinite(dot));
            cache.AccumulateValue(0, 0, 0, 1.0f, pAcc);
            Assert.All(acc, v => Assert.True(float.IsFinite(v)));
        }
    }

    [CudaFact]
    [Trait("Hardware", "Gpu")]
    public void T5_B1_Cuda_KvCacheStoreQ4_0_OutlierAboveHalfMaxTimes7_DoesNotPoisonToNaN()
    {
        const int headDim = 32;
        using var gpu = new GpuContext(0);
        gpu.MakeCurrent();
        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStore, module, "kv_cache_store_q4_0"), "GetStore");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttn, module, "attention_gqa_q4_0"), "GetAttn");

            nuint kvBytes = (nuint)sizeof(BlockQ4_0);
            nuint vecBytes = (nuint)(headDim * sizeof(float));
            IntPtr dKC = gpu.AllocateDevice(kvBytes);
            IntPtr dVC = gpu.AllocateDevice(kvBytes);
            IntPtr dK = gpu.AllocateDevice(vecBytes);
            IntPtr dV = gpu.AllocateDevice(vecBytes);
            IntPtr dQ = gpu.AllocateDevice(vecBytes);
            IntPtr dOut = gpu.AllocateDevice(vecBytes);
            IntPtr dScores = gpu.AllocateDevice(sizeof(float));

            try
            {
                float[] kVec = new float[headDim];
                Array.Fill(kVec, 1.0f);
                kVec[0] = 500_000.0f;
                float[] qVec = new float[headDim];
                Array.Fill(qVec, 1.0f);

                fixed (float* pK = kVec, pQ = qVec)
                {
                    gpu.CopyToDevice(dK, (IntPtr)pK, vecBytes);
                    gpu.CopyToDevice(dV, (IntPtr)pK, vecBytes);
                    gpu.CopyToDevice(dQ, (IntPtr)pQ, vecBytes);
                }

                int nh = 1, hd = headDim, ms = 1, pos = 0;
                void** pStore = stackalloc void*[8];
                pStore[0] = &dKC; pStore[1] = &dVC; pStore[2] = &dK; pStore[3] = &dV;
                pStore[4] = &nh; pStore[5] = &hd; pStore[6] = &ms; pStore[7] = &pos;
                CuDriver.Check(CuDriver.LaunchKernel(fnStore, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pStore, IntPtr.Zero), "Store");

                float scale = 1.0f / MathF.Sqrt(headDim);
                void** pAttn = stackalloc void*[11];
                pAttn[0] = &dQ; pAttn[1] = &dKC; pAttn[2] = &dVC; pAttn[3] = &dOut; pAttn[4] = &dScores;
                pAttn[5] = &nh; pAttn[6] = &nh; pAttn[7] = &hd; pAttn[8] = &ms; pAttn[9] = &pos; pAttn[10] = &scale;
                CuDriver.Check(CuDriver.LaunchKernel(fnAttn, 1, 1, 1, 64, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttn, IntPtr.Zero), "Attn");
                gpu.Synchronize();

                float[] hostOut = new float[headDim];
                fixed (float* pOut = hostOut)
                {
                    gpu.CopyToHost((IntPtr)pOut, dOut, vecBytes);
                }
                Assert.All(hostOut, v => Assert.True(float.IsFinite(v)));
            }
            finally
            {
                gpu.FreeDevice(dKC); gpu.FreeDevice(dVC); gpu.FreeDevice(dK); gpu.FreeDevice(dV);
                gpu.FreeDevice(dQ); gpu.FreeDevice(dOut); gpu.FreeDevice(dScores);
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void T5_B1_D3D12_KvCacheStoreQ4_0_PositiveInfinity_DoesNotPoisonToNaN()
    {
        if (!D3D12Context.IsSupported)
            return;

        const int headDim = 32;
        const int nHeads = 1;
        const int maxSeq = 1;
        using var ctx = new D3D12Context(-1);

        ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeads, headDim, maxSeq, KvCachePrecision.Q4_0);
        float[] kVec = new float[headDim];
        Array.Fill(kVec, 1.0f);
        kVec[0] = float.PositiveInfinity;
        float[] qVec = new float[headDim];
        Array.Fill(qVec, 1.0f);

        using var dKCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dVCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dK = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dQ = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));

        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kVec.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(kVec.AsSpan()));
        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qVec.AsSpan()));

        var storeParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigStore = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeParams));
        using var psoStore = ctx.CreatePipelineState(sigStore, ctx.CompileShader(D3D12Shaders.KvCacheStoreQ4_0));

        var attnParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigAttn = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        using var psoAttn = ctx.CreatePipelineState(sigAttn, ctx.CompileShader(D3D12Shaders.AttentionGqaQ4_0));

        uint* sConsts = stackalloc uint[4];
        sConsts[0] = nHeads; sConsts[1] = headDim; sConsts[2] = maxSeq; sConsts[3] = 0u;
        uint* aConsts = stackalloc uint[6];
        aConsts[0] = nHeads; aConsts[1] = nHeads; aConsts[2] = headDim; aConsts[3] = maxSeq; aConsts[4] = 0u;
        *(float*)&aConsts[5] = 1.0f / MathF.Sqrt(headDim);

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(sigStore);
        cmd.SetPipelineState(psoStore);
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)sConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
        cmd.Dispatch(1, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        cmd.SetComputeRootSignature(sigAttn);
        cmd.SetPipelineState(psoAttn);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)aConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        cmd.Dispatch(1, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] hostOut = new float[headDim];
        ctx.CopyToHost(MemoryMarshal.AsBytes(hostOut.AsSpan()), dOut, (ulong)(headDim * sizeof(float)));
        Assert.All(hostOut, v => Assert.True(float.IsFinite(v), $"Expected finite D3D12 Q4_0 attention output, got {v}."));
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void T5_B2_D3D12_KvCacheStoreF16_OddHeadDim33_PreservesFinalElementWithoutTruncation()
    {
        if (!D3D12Context.IsSupported)
            return;

        const int headDim = 33;
        const int nHeads = 1;
        const int maxSeq = 1;
        using var ctx = new D3D12Context(-1);

        ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeads, headDim, maxSeq, KvCachePrecision.Fp16);
        int paddedPairs = (headDim + 1) / 2;
        Assert.True(kvBytes >= (ulong)(paddedPairs * sizeof(uint)));

        float[] kVec = new float[headDim];
        float[] vVec = new float[headDim];
        for (int i = 0; i < headDim; i++)
        {
            kVec[i] = 0.0f;
            vVec[i] = 0.0f;
        }
        kVec[32] = 7.0f;
        vVec[32] = 11.0f;

        float[] qVec = new float[headDim];
        qVec[32] = 1.0f;

        using var dKCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dVCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dK = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dQ = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));

        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kVec.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vVec.AsSpan()));
        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qVec.AsSpan()));

        var storeParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigStore = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeParams));
        using var psoStore = ctx.CreatePipelineState(sigStore, ctx.CompileShader(D3D12Shaders.KvCacheStoreF16));

        var attnParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigAttn = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        using var psoAttn = ctx.CreatePipelineState(sigAttn, ctx.CompileShader(D3D12Shaders.AttentionGqaF16));

        uint* sConsts = stackalloc uint[4];
        sConsts[0] = nHeads; sConsts[1] = headDim; sConsts[2] = maxSeq; sConsts[3] = 0u;
        uint* aConsts = stackalloc uint[6];
        aConsts[0] = nHeads; aConsts[1] = nHeads; aConsts[2] = headDim; aConsts[3] = maxSeq; aConsts[4] = 0u;
        *(float*)&aConsts[5] = 1.0f / MathF.Sqrt(headDim);

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(sigStore);
        cmd.SetPipelineState(psoStore);
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)sConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
        cmd.Dispatch(((uint)paddedPairs + 127u) / 128u, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        cmd.SetComputeRootSignature(sigAttn);
        cmd.SetPipelineState(psoAttn);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)aConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        cmd.Dispatch(1, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        uint[] kCacheRaw = new uint[paddedPairs];
        ctx.CopyToHost(MemoryMarshal.AsBytes(kCacheRaw.AsSpan()), dKCache, (ulong)(paddedPairs * sizeof(uint)));
        Half storedK32 = BitConverter.UInt16BitsToHalf((ushort)(kCacheRaw[16] & 0xFFFF));
        Assert.Equal(7.0f, (float)storedK32, precision: 2);

        float[] hostOut = new float[headDim];
        ctx.CopyToHost(MemoryMarshal.AsBytes(hostOut.AsSpan()), dOut, (ulong)(headDim * sizeof(float)));
        Assert.Equal(11.0f, hostOut[32], precision: 2);
    }

    [Theory]
    [Trait("Hardware", "Gpu")]
    [InlineData("Gemma4SwaAttentionF16")]
    [InlineData("Qwen3AttentionF16")]
    [InlineData("Qwen3SparseAttentionF16")]
    public void T5_B2_D3D12_F16AttentionReaders_OddHeadDim33_AtPos1_MatchStoreStride(string shaderName)
    {
        if (!D3D12Context.IsSupported)
            return;

        const int headDim = 33, nHeadsQ = 2, nHeadsKv = 1, maxSeq = 4, pos = 1;
        using var ctx = new D3D12Context(-1);

        ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeadsKv, headDim, maxSeq, KvCachePrecision.Fp16);
        using var dKC = ctx.CreateDeviceBuffer(kvBytes);
        using var dVC = ctx.CreateDeviceBuffer(kvBytes);
        byte[] zeros = new byte[kvBytes];
        ctx.CopyToDevice(dKC, zeros);
        ctx.CopyToDevice(dVC, zeros);

        using var dK = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
        using var dQ = ctx.CreateDeviceBuffer((ulong)(nHeadsQ * headDim * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(nHeadsQ * headDim * sizeof(float)));
        using var dSel = ctx.CreateDeviceBuffer(sizeof(uint) * 2);

        float[] kVec = new float[headDim];
        float[] vVec = new float[headDim];
        float[] qVec = new float[nHeadsQ * headDim];
        kVec[32] = 3.0f;
        vVec[32] = 15.0f;
        qVec[32] = 1.0f;
        qVec[headDim + 32] = 1.0f;
        uint[] selIds = { 0u, 1u };

        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kVec.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vVec.AsSpan()));
        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qVec.AsSpan()));
        ctx.CopyToDevice(dSel, MemoryMarshal.AsBytes(selIds.AsSpan()));

        var sParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
        };
        using var sigS = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sParams));
        using var psoS = ctx.CreatePipelineState(sigS, ctx.CompileShader(D3D12Shaders.KvCacheStoreF16));

        bool isGemma4 = shaderName == "Gemma4SwaAttentionF16";
        bool isSparse = shaderName == "Qwen3SparseAttentionF16";
        string attnShader = shaderName switch
        {
            "Gemma4SwaAttentionF16" => D3D12Shaders.Gemma4SwaAttentionF16,
            "Qwen3AttentionF16" => D3D12Shaders.Qwen3AttentionF16,
            _ => D3D12Shaders.Qwen3SparseAttentionF16
        };

        RootParameter[] aParams = isGemma4
            ? new RootParameter[]
            {
                new(new RootConstants(0, 0, 8), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
            }
            : isSparse
            ? new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
            }
            : new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
            };
        using var sigA = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, aParams));
        using var psoA = ctx.CreatePipelineState(sigA, ctx.CompileShader(attnShader));

        uint* sc = stackalloc uint[4];
        sc[0] = nHeadsKv; sc[1] = headDim; sc[2] = maxSeq; sc[3] = pos;
        float scale = 1.0f / MathF.Sqrt(headDim);

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(sigS);
        cmd.SetPipelineState(psoS);
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)sc, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKC.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVC.GPUVirtualAddress);
        cmd.Dispatch(1, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        cmd.SetComputeRootSignature(sigA);
        cmd.SetPipelineState(psoA);
        if (isGemma4)
        {
            uint* ac = stackalloc uint[8];
            ac[0] = nHeadsQ; ac[1] = nHeadsKv; ac[2] = headDim; ac[3] = maxSeq; ac[4] = pos;
            *(float*)&ac[5] = scale; ac[6] = 0u; *(float*)&ac[7] = 0.0f;
            cmd.SetComputeRoot32BitConstants(0, 8, (IntPtr)ac, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dKC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dVC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        }
        else if (isSparse)
        {
            uint* ac = stackalloc uint[6];
            ac[0] = nHeadsQ; ac[1] = nHeadsKv; ac[2] = headDim; ac[3] = maxSeq; ac[4] = 2u;
            *(float*)&ac[5] = scale;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)ac, 0);
            cmd.SetComputeRootShaderResourceView(1, dSel.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dKC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dVC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(5, dOut.GPUVirtualAddress);
        }
        else
        {
            uint* ac = stackalloc uint[6];
            ac[0] = nHeadsQ; ac[1] = nHeadsKv; ac[2] = headDim; ac[3] = maxSeq; ac[4] = pos;
            *(float*)&ac[5] = scale;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)ac, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dKC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dVC.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        }
        cmd.Dispatch(nHeadsQ, 1, 1);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] hostOut = new float[nHeadsQ * headDim];
        ctx.CopyToHost(MemoryMarshal.AsBytes(hostOut.AsSpan()), dOut, (ulong)(hostOut.Length * sizeof(float)));
        float w0 = MathF.Exp(0f);
        float w1 = MathF.Exp(3.0f * scale);
        float expected = (w1 * 15.0f) / (w0 + w1);
        Assert.Equal(expected, hostOut[32], precision: 1);
        Assert.Equal(expected, hostOut[headDim + 32], precision: 1);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void T5_B3_Qwen3HybridD3D12Model_MissingSsmConv1dOnSplitStage_ReportsGlobalModelLayerIndex()
    {
        if (!D3D12Context.IsSupported)
            return;

        string path = Path.Combine(Path.GetTempPath(), $"t5_q3h_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteSyntheticQwen3HybridGguf(path, layers: 4);
            using var gguf = GgufFile.Open(path);
            using var weights = new ModelWeights(gguf);

            typeof(LayerWeights)
                .GetField("<SsmConv1dWeight>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
                .SetValue(weights.Layers[2], Pointer.Box(null, typeof(float*)));

            using var ctx = new D3D12Context(-1);
            var ex = Assert.Throws<InvalidDataException>(() =>
                new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen: 16, startLayer: 2, layerCount: 2, isLastStage: true));

            Assert.Contains("layer 2", ex.Message);
        }
        finally
        {
            TryDelete(path);
        }
    }

    [Fact]
    public void T5_B4_PipelineSessionAndGemma4_AcceptExplicitVramReserveBytesParameter()
    {
        var createMethod = typeof(Glacier.Inference.Pipeline.PipelineSession).GetMethod(
            nameof(Glacier.Inference.Pipeline.PipelineSession.Create),
            BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(createMethod);
        Assert.Contains(createMethod!.GetParameters(), p => p.Name == "explicitVramReserveBytes" && p.ParameterType == typeof(ulong?));

        var gemmaCtor = typeof(Gemma4D3D12Model).GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .FirstOrDefault(c => c.GetParameters().Any(p => p.Name == "explicitVramReserveBytes" && p.ParameterType == typeof(ulong?)));
        Assert.NotNull(gemmaCtor);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public async Task T5_B5_TieredMoERunner_GenerateStreamAsync_ClampsAtMaxSequenceLengthWithoutThrowing()
    {
        string path = Path.Combine(Path.GetTempPath(), $"t5_moe_seq_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteSyntheticMoeGguf(path, blockCount: 1, dim: 32, ffnDim: 32, expertFfnDim: 32, nHeads: 2, nHeadsKv: 1, vocabSize: 16, expertCount: 4, expertUsedCount: 2);
            using var runner = new TieredMoERunner(path, new TieredRunnerOptions { MaxSequenceLength = 4, VramReserveMb = 0 });

            int emitted = 0;
            await foreach (string _ in runner.GenerateStreamAsync("a b c", maxNewTokens: 5))
            {
                emitted++;
            }

            Assert.InRange(emitted, 0, 2);
        }
        finally
        {
            TryDelete(path);
        }
    }

    [Fact]
    public void T5_B6_DFlashOptions_NonNegativeMaskTokenId_IsRejectedOrNeverEmittedAsDraftToken()
    {
        bool rejected = false;
        DFlashOptions? opts = null;
        try
        {
            opts = new DFlashOptions { BlockSize = 4, MaskTokenId = 5, MinEmitConfidence = 0f };
        }
        catch (ArgumentOutOfRangeException)
        {
            rejected = true;
        }

        if (!rejected)
        {
            using var provider = new DFlashDraftProvider(opts!);
            int[] draft = new int[4];
            int count = provider.Draft(new[] { 10, 11, 10 }, maxDraftTokens: 4, draft);
            for (int i = 0; i < count; i++)
            {
                Assert.NotEqual(5, draft[i]);
            }
        }
    }

    [Fact]
    public void T5_B7_KlDivergenceVerifier_EvaluateProbabilities_NonFiniteCandidate_ReturnsNonFiniteResult()
    {
        float[] validProbs = { 0.5f, 0.25f, 0.25f, 0.0f };
        float[] nanProbs = { 0.5f, float.NaN, 0.25f, 0.25f };

        KlDivergenceResult res = KlDivergenceVerifier.EvaluateProbabilities(validProbs, nanProbs, vocabSize: 4);
        Assert.False(res.Passed);
        Assert.True(res.HasNonFiniteQuantizedValues);
    }

    // =========================================================================
    // 2. HARDENING COVERAGE TESTS
    // =========================================================================

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(33)]
    [InlineData(40)]
    public void T5_R1_CpuKvCache_Q4_0_TinyAndOddHeadDims_AndSlidingWindow_AreFiniteAndIsolated(int headDim)
    {
        using var cache = new KVCache(layers: 1, nHeadsKv: 2, headDim: headDim, maxSeqLen: 12, slidingWindow: 4, precision: KvCachePrecision.Q4_0);
        float[] k = new float[2 * headDim];
        float[] v = new float[2 * headDim];
        float[] q = new float[headDim];

        for (int p = 0; p < 7; p++)
        {
            for (int i = 0; i < k.Length; i++)
            {
                k[i] = MathF.Sin(p + i + 1);
                v[i] = MathF.Cos(p + i + 1);
            }
            cache.Store(0, p, k, v);
        }

        Array.Fill(q, 0.5f);
        float[] kr = new float[headDim];
        float[] vr = new float[headDim];
        float[] acc = new float[headDim];

        cache.ReadKey(0, 1, 6, kr);
        cache.ReadValue(0, 1, 6, vr);
        Assert.All(kr, x => Assert.True(float.IsFinite(x)));
        Assert.All(vr, x => Assert.True(float.IsFinite(x)));

        fixed (float* pQ = q, pAcc = acc)
        {
            float dot = cache.DotKey(0, 1, 6, pQ);
            Assert.True(float.IsFinite(dot));
            cache.AccumulateValue(0, 1, 6, 1.0f, pAcc);
            Assert.All(acc, x => Assert.True(float.IsFinite(x)));
        }
    }

    [Fact]
    public void T5_R1_CpuKvCache_Q4_0_PostDisposeOperations_ThrowObjectDisposedException()
    {
        var cache = new KVCache(1, 1, 32, 4, precision: KvCachePrecision.Q4_0);
        cache.Dispose();
        cache.Dispose();

        Assert.Throws<ObjectDisposedException>(() => cache.Store(0, 0, new float[32], new float[32]));
        Assert.Throws<ObjectDisposedException>(() => cache.ReadKey(0, 0, 0, new float[32]));
        Assert.Throws<ObjectDisposedException>(() => cache.ReadValue(0, 0, 0, new float[32]));
    }

    [Fact]
    public void T5_R2_PlanLayerBatch_AllSlotsPinned_AndDuplicateExpertWithinSingleToken_FallsBackToCpuCleanly()
    {
        using var slots = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 4096, allocateVram: false);
        slots.AllocateOrEvictSlot(1, 0, out _, out _, out _, out _);
        slots.AllocateOrEvictSlot(1, 1, out _, out _, out _, out _);
        Assert.True(slots.PinExpert(1, 0));
        Assert.True(slots.PinExpert(1, 1));

        var splitter = new TieredMoEMissSplitter();
        int[] dupIdx = { 2, 2, 3, 3 };
        float[] dupW = { 0.25f, 0.25f, 0.25f, 0.25f };

        var split = splitter.PlanLayerBatch(
            layerIdx: 0,
            batchSize: 1,
            topK: 4,
            numExperts: 8,
            normTopK: true,
            isSpeculativeDraftBatch: true,
            routeTailSkipMinRank: 2,
            policy: TieringExecutionPolicy.GpuSharedMemory,
            batchExpertIndices: dupIdx,
            batchExpertWeights: dupW,
            routingProfile: new ExpertRoutingProfile(2, 8),
            slotManager: slots,
            coScheduler: new BandwidthCoScheduler(),
            hostArena: null);

        Assert.Equal(0, split.PcieTransferUniqueCount);
        Assert.Equal(1, split.CpuExecuteUniqueCount);
        Assert.Equal(2, split.SkippedRoutedSlots);
        Assert.Equal(0.5f, dupW[0], 3);
        Assert.Equal(0.5f, dupW[1], 3);
        Assert.Equal(-1, dupIdx[2]);
        Assert.Equal(-1, dupIdx[3]);
    }

    [Fact]
    public void T5_R3_DFlashDraftProvider_ConcurrentDraftFeedbackAndReset_IsThreadSafe()
    {
        using var provider = new DFlashDraftProvider(new DFlashOptions { BlockSize = 4, EnableAdaptiveConfidence = true });
        int[] prompt = { 1, 2, 3, 1, 2, 3, 1, 2 };

        Parallel.For(0, 64, i =>
        {
            int[] buf = new int[4];
            int emitted = provider.Draft(prompt, maxDraftTokens: 4, buf);
            Assert.InRange(emitted, 0, 4);
            provider.OnVerificationResult(4, i % 5, 1.0);
            if (i % 17 == 0)
            {
                provider.Reset();
            }
        });

        provider.Dispose();
        provider.Dispose();
        Assert.Throws<ObjectDisposedException>(() => provider.Draft(prompt, 4, new int[4]));
    }

    [Fact]
    public void T5_R3_UnslothDynamic31_MatVecZeroRows_MatMulZeroBatch_AndNonFiniteInput_BehavePredictably()
    {
        const int k = 256;
        var blk = new BlockIQ2_XS[1];
        float[] x = new float[k];
        float[] y = new float[4];

        fixed (float* px = x, py = y)
        fixed (BlockIQ2_XS* pb = blk)
        {
            QuantKernels.MatVecMul(GgufType.IQ2_XS, (byte*)pb, px, py, nCols: k, nRows: 0);
            QuantKernels.MatMulBatch(GgufType.IQ2_XS, (byte*)pb, px, py, nCols: k, nRows: 1, batchSize: 0);

            float[] infSrc = new float[k];
            Array.Fill(infSrc, 1.0f);
            infSrc[0] = float.PositiveInfinity;
            float[] dq = new float[k];
            fixed (float* pInf = infSrc, pDq = dq)
            {
                QuantKernels.QuantizeIQ2_XS(pInf, pb, k);
                QuantKernels.DequantizeIQ2_XS(pb, pDq, k);
            }
            Assert.True(float.IsPositiveInfinity((float)blk[0].Delta));
            Assert.All(dq, v => Assert.True(float.IsInfinity(v)));
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { }
    }

    private static void WriteSyntheticQwen3HybridGguf(string path, int layers)
    {
        const int dim = 64, ffnDim = 64, nHeads = 2, nHeadsKv = 1, vocab = 16;
        const int ssmHeadV = 2, ssmHeadK = 2, ssmKeyDim = 8, ssmValDim = 8, convK = 4;
        int ssmCh = ssmKeyDim * 2 + ssmValDim;
        var tensors = new List<(string Name, GgufType Type, ulong[] Dims, byte[] Data)>();

        void AddF32(string name, params ulong[] dims)
        {
            ulong elems = 1;
            foreach (var d in dims) elems *= d;
            byte[] data = new byte[elems * 4];
            float[] f = new float[elems];
            Array.Fill(f, 0.01f);
            Buffer.BlockCopy(f, 0, data, 0, data.Length);
            tensors.Add((name, GgufType.F32, dims, data));
        }

        AddF32("token_embd.weight", (ulong)dim, (ulong)vocab);
        AddF32("output_norm.weight", (ulong)dim);
        AddF32("output.weight", (ulong)dim, (ulong)vocab);

        for (int l = 0; l < layers; l++)
        {
            AddF32($"blk.{l}.attn_norm.weight", (ulong)dim);
            AddF32($"blk.{l}.post_attention_norm.weight", (ulong)dim);
            AddF32($"blk.{l}.ffn_gate.weight", (ulong)dim, (ulong)ffnDim);
            AddF32($"blk.{l}.ffn_up.weight", (ulong)dim, (ulong)ffnDim);
            AddF32($"blk.{l}.ffn_down.weight", (ulong)ffnDim, (ulong)dim);
            AddF32($"blk.{l}.attn_qkv.weight", (ulong)dim, (ulong)ssmCh);
            AddF32($"blk.{l}.attn_gate.weight", (ulong)dim, (ulong)ssmValDim);
            AddF32($"blk.{l}.ssm_a", (ulong)ssmHeadV);
            AddF32($"blk.{l}.ssm_dt.bias", (ulong)ssmHeadV);
            AddF32($"blk.{l}.ssm_beta.weight", (ulong)dim, (ulong)ssmHeadV);
            AddF32($"blk.{l}.ssm_alpha.weight", (ulong)dim, (ulong)ssmHeadV);
            AddF32($"blk.{l}.ssm_norm.weight", (ulong)(ssmValDim / ssmHeadV));
            AddF32($"blk.{l}.ssm_out.weight", (ulong)ssmValDim, (ulong)dim);
            AddF32($"blk.{l}.ssm_conv1d.weight", (ulong)convK, (ulong)ssmCh);
        }

        var kvs = new (string K, GgufValueType T, object V)[]
        {
            ("general.architecture", GgufValueType.String, "qwen35"),
            ("qwen35.block_count", GgufValueType.Uint32, (uint)layers),
            ("qwen35.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen35.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen35.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen35.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen35.attention.key_length", GgufValueType.Uint32, (uint)(dim / nHeads)),
            ("qwen35.context_length", GgufValueType.Uint32, 64u),
            ("qwen35.ssm.conv_kernel", GgufValueType.Uint32, (uint)convK),
            ("qwen35.ssm.state_size", GgufValueType.Uint32, (uint)(ssmKeyDim / ssmHeadK)),
            ("qwen35.ssm.group_count", GgufValueType.Uint32, (uint)ssmHeadK),
            ("qwen35.ssm.time_step_rank", GgufValueType.Uint32, (uint)ssmHeadV),
            ("qwen35.ssm.inner_size", GgufValueType.Uint32, (uint)ssmValDim),
            ("qwen35.full_attention_interval", GgufValueType.Uint32, 99u),
            ("tokenizer.ggml.tokens", GgufValueType.Array, Enumerable.Range(0, vocab).Select(i => $"t{i}").ToArray()),
            ("tokenizer.ggml.merges", GgufValueType.Array, Array.Empty<string>()),
        };
        WriteGgufFile(path, kvs, tensors);
    }

    private static void WriteSyntheticMoeGguf(string path, int blockCount, int dim, int ffnDim, int expertFfnDim, int nHeads, int nHeadsKv, int vocabSize, int expertCount, int expertUsedCount)
    {
        var rng = new Random(42);
        var tensors = new List<(string Name, GgufType Type, ulong[] Dims, byte[] Data)>();

        void AddF32(string name, params ulong[] dims)
        {
            ulong elems = 1;
            foreach (var d in dims) elems *= d;
            float[] f = new float[elems];
            for (int i = 0; i < f.Length; i++) f[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            byte[] data = new byte[elems * 4];
            Buffer.BlockCopy(f, 0, data, 0, data.Length);
            tensors.Add((name, GgufType.F32, dims, data));
        }

        AddF32("token_embd.weight", (ulong)dim, (ulong)vocabSize);
        AddF32("output_norm.weight", (ulong)dim);
        AddF32("output.weight", (ulong)dim, (ulong)vocabSize);
        int headDim = dim / nHeads;

        for (int l = 0; l < blockCount; l++)
        {
            AddF32($"blk.{l}.attn_norm.weight", (ulong)dim);
            AddF32($"blk.{l}.attn_q.weight", (ulong)dim, (ulong)(nHeads * headDim));
            AddF32($"blk.{l}.attn_k.weight", (ulong)dim, (ulong)(nHeadsKv * headDim));
            AddF32($"blk.{l}.attn_v.weight", (ulong)dim, (ulong)(nHeadsKv * headDim));
            AddF32($"blk.{l}.attn_output.weight", (ulong)(nHeads * headDim), (ulong)dim);
            AddF32($"blk.{l}.ffn_norm.weight", (ulong)dim);
            AddF32($"blk.{l}.ffn_gate_inp.weight", (ulong)dim, (ulong)expertCount);
            AddF32($"blk.{l}.ffn_gate_exps.weight", (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount);
            AddF32($"blk.{l}.ffn_up_exps.weight", (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount);
            AddF32($"blk.{l}.ffn_down_exps.weight", (ulong)expertFfnDim, (ulong)dim, (ulong)expertCount);
        }

        var kvs = new (string K, GgufValueType T, object V)[]
        {
            ("general.architecture", GgufValueType.String, "qwen2moe"),
            ("qwen2moe.block_count", GgufValueType.Uint32, (uint)blockCount),
            ("qwen2moe.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2moe.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2moe.expert_feed_forward_length", GgufValueType.Uint32, (uint)expertFfnDim),
            ("qwen2moe.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen2moe.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen2moe.attention.key_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2moe.context_length", GgufValueType.Uint32, 64u),
            ("qwen2moe.expert_count", GgufValueType.Uint32, (uint)expertCount),
            ("qwen2moe.expert_used_count", GgufValueType.Uint32, (uint)expertUsedCount),
            ("tokenizer.ggml.tokens", GgufValueType.Array, Enumerable.Range(0, vocabSize).Select(i => i switch { 0 => "<|endoftext|>", 1 => "a", 2 => "b", 3 => "c", 4 => "d", _ => $"t{i}" }).ToArray()),
            ("tokenizer.ggml.merges", GgufValueType.Array, Array.Empty<string>()),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, 0u),
        };
        WriteGgufFile(path, kvs, tensors);
    }

    private static void WriteGgufFile(string path, (string K, GgufValueType T, object V)[] kvs, List<(string Name, GgufType Type, ulong[] Dims, byte[] Data)> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
        void WriteStr(string s) { byte[] b = Encoding.UTF8.GetBytes(s); bw.Write((ulong)b.Length); bw.Write(b); }
        void Align32() { long m = fs.Position % 32; if (m != 0) bw.Write(new byte[32 - m]); }

        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvs.Length);

        foreach (var (k, t, v) in kvs)
        {
            WriteStr(k);
            bw.Write((uint)t);
            if (t == GgufValueType.String) WriteStr((string)v);
            else if (t == GgufValueType.Uint32) bw.Write((uint)v);
            else if (t == GgufValueType.Array)
            {
                var arr = (string[])v;
                bw.Write((uint)GgufValueType.String);
                bw.Write((ulong)arr.Length);
                foreach (var s in arr) WriteStr(s);
            }
        }

        ulong off = 0;
        foreach (var t in tensors)
        {
            WriteStr(t.Name);
            bw.Write((uint)t.Dims.Length);
            foreach (var d in t.Dims) bw.Write(d);
            bw.Write((uint)t.Type);
            bw.Write(off);
            ulong sz = (ulong)t.Data.Length;
            off += sz;
            if (off % 32 != 0) off += 32 - (off % 32);
        }

        Align32();
        foreach (var t in tensors) { bw.Write(t.Data); Align32(); }
    }
}
