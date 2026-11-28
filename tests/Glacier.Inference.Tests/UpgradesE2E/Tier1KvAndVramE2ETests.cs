using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Vortice.Direct3D12;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

public sealed unsafe class Tier1CpuKvAndVramE2ETests
{
    // =========================================================================
    // F1: CPU 4-Bit KV Cache (T1-F1-01 .. T1-F1-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F1_01_KvCachePrecision_TryParse_And_DisplayName_AreAccurate()
    {
        Assert.True(KvCachePrecisionExtensions.TryParse("q4_0", out var p1));
        Assert.Equal(KvCachePrecision.Q4_0, p1);

        Assert.True(KvCachePrecisionExtensions.TryParse("Q4_0", out var p2));
        Assert.Equal(KvCachePrecision.Q4_0, p2);

        Assert.True(KvCachePrecisionExtensions.TryParse("int4", out var p3));
        Assert.Equal(KvCachePrecision.Q4_0, p3);

        Assert.True(KvCachePrecisionExtensions.TryParse("fp16", out var p4));
        Assert.Equal(KvCachePrecision.Fp16, p4);

        Assert.True(KvCachePrecisionExtensions.TryParse("auto", out var p5));
        Assert.Equal(KvCachePrecision.Auto, p5);

        Assert.False(KvCachePrecisionExtensions.TryParse("q99_invalid", out _));
        Assert.Equal("Q4_0", KvCachePrecision.Q4_0.ToDisplayName());
        Assert.Equal("Fp16", KvCachePrecision.Fp16.ToDisplayName());
        Assert.Equal("Fp32", KvCachePrecision.Fp32.ToDisplayName());
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F1_02_CpuKvCache_Q4_0_ReducesFootprintByApprox2xVsFp8And4xVsFp16()
    {
        const int layers = 2;
        const int headsKv = 4;
        const int headDim = 64;
        const int maxSeq = 128;

        using var q4 = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Q4_0);
        using var f8 = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp8);
        using var f16 = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp16);
        using var f32 = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp32);

        long expectedQ4PerBuffer = (long)layers * headsKv * maxSeq * ((headDim + 31) / 32) * sizeof(BlockQ4_0);
        Assert.Equal(expectedQ4PerBuffer, q4.KeyBufferBytes);
        Assert.Equal(expectedQ4PerBuffer, q4.ValueBufferBytes);
        Assert.Equal(expectedQ4PerBuffer * 2, q4.TotalBytes);

        double ratioVsFp32 = (double)q4.TotalBytes / f32.TotalBytes;
        double ratioVsFp16 = (double)q4.TotalBytes / f16.TotalBytes;
        double ratioVsFp8 = (double)q4.TotalBytes / f8.TotalBytes;
        Assert.True(ratioVsFp32 <= 0.15, $"Expected Q4_0/Fp32 <= 0.15, got {ratioVsFp32:F4}");
        Assert.True(ratioVsFp16 <= 0.3125, $"Expected Q4_0/Fp16 <= 0.3125, got {ratioVsFp16:F4}");
        Assert.True(ratioVsFp8 <= 0.625, $"Expected Q4_0/Fp8 <= 0.625, got {ratioVsFp8:F4}");

        // Fixed byte budget token capacity: >= 1.6x Fp8 and >= 3.2x Fp16
        const long fixedByteBudget = 512L * 1024L;
        long perTokenQ4 = (long)layers * 2L * headsKv * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Q4_0);
        long perTokenF8 = (long)layers * 2L * headsKv * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Fp8);
        long perTokenF16 = (long)layers * 2L * headsKv * KVCache.GetSlotByteSize(headDim, KvCachePrecision.Fp16);

        long capQ4 = fixedByteBudget / perTokenQ4;
        long capF8 = fixedByteBudget / perTokenF8;
        long capF16 = fixedByteBudget / perTokenF16;

        Assert.True(capQ4 >= capF8 * 1.6, $"Expected Q4_0 token capacity ({capQ4}) >= 1.6x Fp8 ({capF8})");
        Assert.True(capQ4 >= capF16 * 3.2, $"Expected Q4_0 token capacity ({capQ4}) >= 3.2x Fp16 ({capF16})");
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F1_03_CpuKvCache_Q4_0_StoreAndReadRoundTrip_PreservesCosineSimilarity()
    {
        const int layers = 2;
        const int headsKv = 2;
        const int headDim = 64;
        const int maxSeq = 16;

        using var cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Q4_0);
        var (k, v, _) = Fx1Vectors.GenerateOrthogonalVectors(nHeadsQ: 4, headsKv, headDim, seed: 11);

        cache.Store(layer: 1, pos: 5, k, v);

        float[] kRead = new float[headDim];
        float[] vRead = new float[headDim];
        for (int h = 0; h < headsKv; h++)
        {
            cache.ReadKey(1, h, 5, kRead);
            cache.ReadValue(1, h, 5, vRead);

            float cosK = Fx1Vectors.CosineSimilarity(k.AsSpan(h * headDim, headDim), kRead);
            float cosV = Fx1Vectors.CosineSimilarity(v.AsSpan(h * headDim, headDim), vRead);
            Assert.True(cosK > 0.96f, $"Head {h} K cosine similarity {cosK:F4} <= 0.96");
            Assert.True(cosV > 0.96f, $"Head {h} V cosine similarity {cosV:F4} <= 0.96");
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F1_04_CpuKvCache_Q4_0_DotKeyAndAccumulateValue_MatchFp32AnalyticalReference()
    {
        const int headsKv = 2;
        const int headDim = 64;
        using var q4Cache = new KVCache(1, headsKv, headDim, 16, precision: KvCachePrecision.Q4_0);
        using var fp32Cache = new KVCache(1, headsKv, headDim, 16, precision: KvCachePrecision.Fp32);

        var (k0, v0, q) = Fx1Vectors.GenerateOrthogonalVectors(2, headsKv, headDim, seed: 23);
        var (k1, v1, _) = Fx1Vectors.GenerateOrthogonalVectors(2, headsKv, headDim, seed: 29);

        q4Cache.Store(0, 0, k0, v0);
        q4Cache.Store(0, 1, k1, v1);
        fp32Cache.Store(0, 0, k0, v0);
        fp32Cache.Store(0, 1, k1, v1);

        float[] outQ4 = new float[headDim];
        float[] outFp32 = new float[headDim];

        fixed (float* pQ = q, pOutQ4 = outQ4, pOutFp32 = outFp32)
        {
            for (int h = 0; h < headsKv; h++)
            {
                float dotQ4_0 = q4Cache.DotKey(0, h, 0, pQ + h * headDim);
                float dotFp32_0 = fp32Cache.DotKey(0, h, 0, pQ + h * headDim);
                Assert.True(MathF.Abs(dotQ4_0 - dotFp32_0) < 0.5f,
                    $"Head {h} DotKey diff too large: q4={dotQ4_0:F4}, fp32={dotFp32_0:F4}");

                Array.Clear(outQ4);
                Array.Clear(outFp32);
                q4Cache.AccumulateValue(0, h, 0, 0.7f, pOutQ4);
                q4Cache.AccumulateValue(0, h, 1, 0.3f, pOutQ4);
                fp32Cache.AccumulateValue(0, h, 0, 0.7f, pOutFp32);
                fp32Cache.AccumulateValue(0, h, 1, 0.3f, pOutFp32);

                float cos = Fx1Vectors.CosineSimilarity(outFp32, outQ4);
                Assert.True(cos > 0.96f, $"Head {h} AccumulateValue cosine {cos:F4} <= 0.96");
            }
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F1_05_CpuKvCache_Q4_0_RingBufferWraparound_OverwritesCleanly()
    {
        const int headsKv = 1;
        const int headDim = 64;
        const int maxSeq = 16;

        using var cache = new KVCache(1, headsKv, headDim, maxSeq, isRingBuffer: true, precision: KvCachePrecision.Q4_0);
        float[][] keys = new float[24][];
        float[][] vals = new float[24][];

        for (int pos = 0; pos < 24; pos++)
        {
            var (k, v, _) = Fx1Vectors.GenerateOrthogonalVectors(1, headsKv, headDim, seed: 500 + pos);
            keys[pos] = k;
            vals[pos] = v;
            cache.Store(0, pos, k, v);
        }

        Assert.Equal(4, cache.GetPhysicalSlot(20));

        float[] kRead = new float[headDim];
        float[] vRead = new float[headDim];
        for (int pos = 8; pos < 24; pos++)
        {
            cache.ReadKey(0, 0, pos, kRead);
            cache.ReadValue(0, 0, pos, vRead);
            float cosK = Fx1Vectors.CosineSimilarity(keys[pos], kRead);
            float cosV = Fx1Vectors.CosineSimilarity(vals[pos], vRead);
            Assert.True(cosK > 0.96f, $"Ring pos {pos} K cosine {cosK:F4} <= 0.96");
            Assert.True(cosV > 0.96f, $"Ring pos {pos} V cosine {cosV:F4} <= 0.96");
        }
    }

    // =========================================================================
    // F4: VramReserveMb = 0 Override (CPU / Guard / TieredMoERunner) (T1-F4-01 .. T1-F4-03)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F4_01_WddmVramGuard_ResolveExplicitReserveBytes_ExactMappings()
    {
        Assert.Null(WddmVramGuard.ResolveExplicitReserveBytes(null));
        Assert.Equal(0UL, WddmVramGuard.ResolveExplicitReserveBytes(0));
        Assert.Equal(512UL * 1024UL * 1024UL, WddmVramGuard.ResolveExplicitReserveBytes(512));
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F4_02_WddmVramGuard_ZeroReserveOverride_RemovesSafetyFloorAndIncreasesHeadroom()
    {
        const ulong dedicated = 4UL * 1024 * 1024 * 1024;
        const ulong availableVramBytes = 3800UL * 1024 * 1024;
        const ulong osMinReserveFloor = 350UL * 1024 * 1024;

        ulong defaultReserve = WddmVramGuard.ResolveModelSafetyReserveBytes(
            dedicated, isWindowsWddm: true, isUnifiedMemory: false, explicitReserveBytes: null);
        Assert.True(defaultReserve >= osMinReserveFloor);

        ulong zeroReserve = WddmVramGuard.ResolveModelSafetyReserveBytes(
            dedicated, isWindowsWddm: true, isUnifiedMemory: false, explicitReserveBytes: 0UL);
        Assert.Equal(0UL, zeroReserve);

        var defaultHeadroom = WddmVramGuard.EvaluateHeadroom(dedicated, availableVramBytes, currentUsageBytes: 0UL, isWindowsWddm: true, explicitReserveBytes: null);
        var zeroHeadroom = WddmVramGuard.EvaluateHeadroom(dedicated, availableVramBytes, currentUsageBytes: 0UL, isWindowsWddm: true, explicitReserveBytes: 0UL);

        Assert.True(defaultHeadroom.SafetyReserveBytes >= osMinReserveFloor);
        Assert.Equal(0UL, zeroHeadroom.SafetyReserveBytes);
        Assert.Equal(availableVramBytes, zeroHeadroom.SafeAvailableBytes);
        Assert.True(defaultHeadroom.SafeAvailableBytes <= availableVramBytes - osMinReserveFloor);
        Assert.True(zeroHeadroom.SafeAvailableBytes > defaultHeadroom.SafeAvailableBytes);
        Assert.Equal(defaultHeadroom.SafeAvailableBytes + defaultHeadroom.SafetyReserveBytes, zeroHeadroom.SafeAvailableBytes);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier1")]
    public void T1_F4_03_TieredMoERunner_VramReserveMbZero_WiresEffectiveReserveZero()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 4, topK: 2, dim: 32, expertFfnDim: 32);
        var profile = new ExpertRoutingProfile(1, 4);
        var coScheduler = new BandwidthCoScheduler();
        var options = new TieredRunnerOptions { VramReserveMb = 0 };

        using var runner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, profile, coScheduler, slotManager: null, hostArena: null, options: options, lmHeadWeights: moe.LmHead);

        Assert.Equal(0UL, runner.EffectiveVramReserveBytes);
        var headroom = runner.QueryVramHeadroom();
        Assert.Equal(0UL, headroom.SafetyReserveBytes);

        // Verify reclaimed reserve strictly expands computed expert VRAM slot budget vs null reserve
        const ulong dedicated = 4UL * 1024 * 1024 * 1024;
        const ulong availableVramBytes = 3800UL * 1024 * 1024;
        const ulong currentUsage = 3200UL * 1024 * 1024;
        const ulong slotBytes = 64UL * 1024 * 1024;

        var defaultSnap = WddmVramGuard.EvaluateHeadroom(dedicated, availableVramBytes, currentUsage, isWindowsWddm: true, explicitReserveBytes: null);
        var zeroSnap = WddmVramGuard.EvaluateHeadroom(dedicated, availableVramBytes, currentUsage, isWindowsWddm: true, explicitReserveBytes: 0UL);
        Assert.True(defaultSnap.SafetyReserveBytes >= 350_000_000UL);
        Assert.Equal(0UL, zeroSnap.SafetyReserveBytes);
        Assert.True(zeroSnap.SafeAvailableBytes > defaultSnap.SafeAvailableBytes);

        int defaultSlots = WddmVramGuard.ComputeRecommendedSlotCount(defaultSnap, slotBytes, 0, maxSlotCapacity: 32, minSlotFloor: 0);
        int zeroSlots = WddmVramGuard.ComputeRecommendedSlotCount(zeroSnap, slotBytes, 0, maxSlotCapacity: 32, minSlotFloor: 0);
        Assert.True(zeroSlots > defaultSlots, $"Expected zeroSlots ({zeroSlots}) > defaultSlots ({defaultSlots}).");
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier1")]
[Trait("Hardware", "Gpu")]
public sealed unsafe class Tier1CudaKvCacheE2ETests
{
    // =========================================================================
    // F2: CUDA 4-Bit KV Cache (T1-F2-01 .. T1-F2-05)
    // =========================================================================

    [UpgradeCudaFact]
    public void T1_F2_01_CudaKvStoreAndAttentionQ4_0_Kernels_MatchCpuReference()
    {
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int headDim = 64;
        const int maxSeqLen = 16;
        const int seqSteps = 4;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        using var gpu = new GpuContext(0);
        gpu.MakeCurrent();

        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStore, module, "kv_cache_store_q4_0"), "GetFunc(kv_cache_store_q4_0)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttn, module, "attention_gqa_q4_0"), "GetFunc(attention_gqa_q4_0)");

            int blocksPerHead = (headDim + 31) / 32;
            nuint kvCacheBytes = (nuint)((long)nHeadsKv * maxSeqLen * blocksPerHead * sizeof(BlockQ4_0));
            nuint kvVecBytes = (nuint)(nHeadsKv * headDim * sizeof(float));
            nuint qOutBytes = (nuint)(nHeadsQ * headDim * sizeof(float));

            IntPtr dKCache = gpu.AllocateDevice(kvCacheBytes);
            IntPtr dVCache = gpu.AllocateDevice(kvCacheBytes);
            IntPtr dK = gpu.AllocateDevice(kvVecBytes);
            IntPtr dV = gpu.AllocateDevice(kvVecBytes);
            IntPtr dQ = gpu.AllocateDevice(qOutBytes);
            IntPtr dOut = gpu.AllocateDevice(qOutBytes);
            IntPtr dScores = gpu.AllocateDevice((nuint)(nHeadsQ * maxSeqLen * sizeof(float)));

            try
            {
                CuDriver.Check(CuDriver.MemsetD8(dKCache, 0, kvCacheBytes), "MemsetD8(K)");
                CuDriver.Check(CuDriver.MemsetD8(dVCache, 0, kvCacheBytes), "MemsetD8(V)");

                float[] kSeq = new float[seqSteps * nHeadsKv * headDim];
                float[] vSeq = new float[seqSteps * nHeadsKv * headDim];
                float[] query = null!;
                void** storeParams = stackalloc void*[8];

                for (int pos = 0; pos < seqSteps; pos++)
                {
                    var (k, v, q) = Fx1Vectors.GenerateOrthogonalVectors(nHeadsQ, nHeadsKv, headDim, seed: 200 + pos);
                    Array.Copy(k, 0, kSeq, pos * nHeadsKv * headDim, k.Length);
                    Array.Copy(v, 0, vSeq, pos * nHeadsKv * headDim, v.Length);
                    query = q;

                    fixed (float* pK = k, pV = v)
                    {
                        gpu.CopyToDevice(dK, (IntPtr)pK, kvVecBytes);
                        gpu.CopyToDevice(dV, (IntPtr)pV, kvVecBytes);
                    }

                    int totalBlocks = nHeadsKv * blocksPerHead;
                    uint gridStore = (uint)((totalBlocks + 7) / 8);
                    IntPtr pKC = dKCache, pVC = dVCache, pKd = dK, pVd = dV;
                    int nhKv = nHeadsKv, hd = headDim, ms = maxSeqLen, p = pos;
                    storeParams[0] = &pKC;
                    storeParams[1] = &pVC;
                    storeParams[2] = &pKd;
                    storeParams[3] = &pVd;
                    storeParams[4] = &nhKv;
                    storeParams[5] = &hd;
                    storeParams[6] = &ms;
                    storeParams[7] = &p;
                    CuDriver.Check(CuDriver.LaunchKernel(fnStore, gridStore, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeParams, IntPtr.Zero), "Launch(store_q4_0)");
                }

                fixed (float* pQ = query)
                {
                    gpu.CopyToDevice(dQ, (IntPtr)pQ, qOutBytes);
                }

                uint blockAttn = (uint)Math.Clamp(((headDim + 31) / 32) * 32, 32, 256);
                IntPtr pQd = dQ, pKC2 = dKCache, pVC2 = dVCache, pOutD = dOut, pScoresD = dScores;
                int nhQ2 = nHeadsQ, nhKv2 = nHeadsKv, hd2 = headDim, ms2 = maxSeqLen, lastPos = seqSteps - 1;
                float scale2 = attnScale;
                void** attnParams = stackalloc void*[11];
                attnParams[0] = &pQd;
                attnParams[1] = &pKC2;
                attnParams[2] = &pVC2;
                attnParams[3] = &pOutD;
                attnParams[4] = &pScoresD;
                attnParams[5] = &nhQ2;
                attnParams[6] = &nhKv2;
                attnParams[7] = &hd2;
                attnParams[8] = &ms2;
                attnParams[9] = &lastPos;
                attnParams[10] = &scale2;
                CuDriver.Check(CuDriver.LaunchKernel(fnAttn, (uint)nHeadsQ, 1, 1, blockAttn, 1, 1, 0, IntPtr.Zero, (IntPtr)attnParams, IntPtr.Zero), "Launch(attention_q4_0)");
                gpu.Synchronize();

                float[] gpuOut = new float[nHeadsQ * headDim];
                fixed (float* pOut = gpuOut)
                {
                    gpu.CopyToHost((IntPtr)pOut, dOut, qOutBytes);
                }

                float[] cpuRef = Fx1Vectors.ComputeSoftAttentionReference(query, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, seqSteps, attnScale);
                float cos = Fx1Vectors.CosineSimilarity(cpuRef, gpuOut);
                Assert.True(cos > 0.95f, $"CUDA Q4_0 GQA attention cosine {cos:F4} <= 0.95");
            }
            finally
            {
                gpu.FreeDevice(dKCache);
                gpu.FreeDevice(dVCache);
                gpu.FreeDevice(dK);
                gpu.FreeDevice(dV);
                gpu.FreeDevice(dQ);
                gpu.FreeDevice(dOut);
                gpu.FreeDevice(dScores);
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [UpgradeCudaFact]
    public void T1_F2_02_CudaKvStoreQ4_0_RoundTrip_MatchesCpuKvCacheWithin1e3()
    {
        const int nHeadsKv = 2;
        const int headDim = 64; // 2 blocks of 32: Block 0 ramp [-7..+7]*0.25, Block 1 zeros (FX-1.2b)
        const int maxSeqLen = 8;
        const int pos = 2;

        float[] kVec = new float[nHeadsKv * headDim];
        float[] vVec = new float[nHeadsKv * headDim];
        for (int h = 0; h < nHeadsKv; h++)
        {
            for (int d = 0; d < 32; d++)
            {
                int ramp = (d % 15) - 7;
                kVec[h * headDim + d] = ramp * 0.25f;
                vVec[h * headDim + d] = -ramp * 0.25f;
            }
            // Block 1 [32..63] intentionally left 0.0f (zero-scale block)
        }

        using var cpuCache = new KVCache(1, nHeadsKv, headDim, maxSeqLen, precision: KvCachePrecision.Q4_0);
        cpuCache.Store(0, pos, kVec, vVec);

        using var gpu = new GpuContext(0);
        gpu.MakeCurrent();

        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStore, module, "kv_cache_store_q4_0"), "GetFunc(kv_cache_store_q4_0)");

            int blocksPerHead = (headDim + 31) / 32;
            nuint kvCacheBytes = (nuint)((long)nHeadsKv * maxSeqLen * blocksPerHead * sizeof(BlockQ4_0));
            nuint kvVecBytes = (nuint)(nHeadsKv * headDim * sizeof(float));

            IntPtr dKCache = gpu.AllocateDevice(kvCacheBytes);
            IntPtr dVCache = gpu.AllocateDevice(kvCacheBytes);
            IntPtr dK = gpu.AllocateDevice(kvVecBytes);
            IntPtr dV = gpu.AllocateDevice(kvVecBytes);

            try
            {
                CuDriver.Check(CuDriver.MemsetD8(dKCache, 0, kvCacheBytes), "MemsetD8(K)");
                CuDriver.Check(CuDriver.MemsetD8(dVCache, 0, kvCacheBytes), "MemsetD8(V)");

                fixed (float* pK = kVec, pV = vVec)
                {
                    gpu.CopyToDevice(dK, (IntPtr)pK, kvVecBytes);
                    gpu.CopyToDevice(dV, (IntPtr)pV, kvVecBytes);
                }

                int totalBlocks = nHeadsKv * blocksPerHead;
                uint gridStore = (uint)((totalBlocks + 7) / 8);
                IntPtr pKC = dKCache, pVC = dVCache, pKd = dK, pVd = dV;
                int nhKv = nHeadsKv, hd = headDim, ms = maxSeqLen, p = pos;
                void** storeParams = stackalloc void*[8];
                storeParams[0] = &pKC; storeParams[1] = &pVC; storeParams[2] = &pKd; storeParams[3] = &pVd;
                storeParams[4] = &nhKv; storeParams[5] = &hd; storeParams[6] = &ms; storeParams[7] = &p;
                CuDriver.Check(CuDriver.LaunchKernel(fnStore, gridStore, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeParams, IntPtr.Zero), "Launch(store_q4_0)");
                gpu.Synchronize();

                byte[] hostKCache = new byte[kvCacheBytes];
                byte[] hostVCache = new byte[kvCacheBytes];
                fixed (byte* pHostK = hostKCache, pHostV = hostVCache)
                {
                    gpu.CopyToHost((IntPtr)pHostK, dKCache, kvCacheBytes);
                    gpu.CopyToHost((IntPtr)pHostV, dVCache, kvCacheBytes);
                }

                float[] cpuK = new float[headDim];
                float[] cpuV = new float[headDim];
                float[] cudaK = new float[headDim];
                float[] cudaV = new float[headDim];

                fixed (byte* pHostK = hostKCache, pHostV = hostVCache)
                fixed (float* pCudaK = cudaK, pCudaV = cudaV)
                {
                    for (int h = 0; h < nHeadsKv; h++)
                    {
                        cpuCache.ReadKey(0, h, pos, cpuK);
                        cpuCache.ReadValue(0, h, pos, cpuV);

                        int blockOffset = ((h * maxSeqLen) + pos) * blocksPerHead;
                        QuantKernels.DequantizeQ4_0(((BlockQ4_0*)pHostK) + blockOffset, pCudaK, headDim);
                        QuantKernels.DequantizeQ4_0(((BlockQ4_0*)pHostV) + blockOffset, pCudaV, headDim);

                        double l2K = 0.0;
                        double l2V = 0.0;
                        for (int d = 0; d < headDim; d++)
                        {
                            double dk = cpuK[d] - cudaK[d];
                            double dv = cpuV[d] - cudaV[d];
                            l2K += dk * dk;
                            l2V += dv * dv;
                        }
                        l2K = Math.Sqrt(l2K);
                        l2V = Math.Sqrt(l2V);

                        Assert.True(l2K < 1e-3, $"Head {h} CUDA vs CPU Q4_0 K L2 {l2K:E3} >= 1e-3");
                        Assert.True(l2V < 1e-3, $"Head {h} CUDA vs CPU Q4_0 V L2 {l2V:E3} >= 1e-3");
                    }
                }
            }
            finally
            {
                gpu.FreeDevice(dKCache);
                gpu.FreeDevice(dVCache);
                gpu.FreeDevice(dK);
                gpu.FreeDevice(dV);
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [UpgradeCudaFact]
    public void T1_F2_03_Qwen2GpuModel_Q4_0_ForwardAndReset_MatchesFp32LogitsDistribution()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_cuda_fwd.gguf"), dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);

        using var q4Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0);
        using var f32Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Fp32);

        float[] q4Logits = new float[weights.VocabSize];
        float[] f32Logits = new float[weights.VocabSize];
        int[] prompt = { 1, 5, 11, 3 };

        for (int pos = 0; pos < prompt.Length; pos++)
        {
            q4Model.Forward(prompt[pos], pos, null, q4Logits, computeLogits: pos == prompt.Length - 1);
            f32Model.Forward(prompt[pos], pos, null, f32Logits, computeLogits: pos == prompt.Length - 1);
        }

        foreach (float l in q4Logits) Assert.True(float.IsFinite(l), "Non-finite CUDA Q4_0 logit.");
        float cos = Fx1Vectors.CosineSimilarity(f32Logits, q4Logits);
        Assert.True(cos > 0.95f, $"CUDA Q4_0 vs FP32 forward logits cosine {cos:F4} <= 0.95");

        q4Model.ResetKvCache();
        float[] afterReset = new float[weights.VocabSize];
        float[] freshRun = new float[weights.VocabSize];
        q4Model.Forward(prompt[0], 0, null, afterReset, computeLogits: true);
        f32Model.ResetKvCache();
        f32Model.Forward(prompt[0], 0, null, freshRun, computeLogits: true);
        Assert.True(Fx1Vectors.CosineSimilarity(freshRun, afterReset) > 0.95f);
    }

    [UpgradeCudaFact]
    public void T1_F2_04_Qwen2GpuModel_Q4_0_AllocatesExpectedPackedKvBytesVsFp8AndFp16()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_cuda_mem.gguf"), dim: 256, ffnDim: 256, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);

        using var q4Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Q4_0);
        using var f8Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp8);
        using var f16Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16);

        Assert.Equal(KvCachePrecision.Q4_0, q4Model.KvPrecision);
        Assert.Equal(KvCachePrecision.Fp8, f8Model.KvPrecision);
        Assert.Equal(KvCachePrecision.Fp16, f16Model.KvPrecision);

        long headDim = weights.HeadDim;
        long expectedLayerBytes = weights.HeadCountKv * 64L * ((headDim + 31) / 32) * 18L;
        Assert.Equal(expectedLayerBytes, (long)q4Model.KvCacheBytesPerLayer);
        Assert.Equal(expectedLayerBytes * 2 * weights.BlockCount, q4Model.TotalKvCacheBytes);

        double ratioVsFp8 = (double)q4Model.TotalKvCacheBytes / f8Model.TotalKvCacheBytes;
        double ratioVsFp16 = (double)q4Model.TotalKvCacheBytes / f16Model.TotalKvCacheBytes;
        Assert.True(ratioVsFp8 <= 0.625, $"Expected CUDA Q4_0/Fp8 <= 0.625, got {ratioVsFp8:F4}");
        Assert.True(ratioVsFp16 <= 0.3125, $"Expected CUDA Q4_0/Fp16 <= 0.3125, got {ratioVsFp16:F4}");
    }

    [UpgradeCudaFact]
    public void T1_F2_05_InferenceSession_And_ForwardBatch_Cuda_Q4_0_EndToEndPrefillAndDecode()
    {
        var nv = DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor == GpuVendor.Nvidia && d.SupportedEngines.Contains(InferenceEngineType.BareMetal));
        Assert.NotNull(nv);

        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_cuda_session.gguf"), dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32);

        // 1. Direct Qwen2GpuModel ForwardBatch vs sequential Forward parity under Q4_0
        using (var gguf = GgufFile.Open(ggufPath))
        {
            var weights = new ModelWeights(gguf);
            using var gpu = new GpuContext(0);
            using var model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0);
            using var f16Model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Fp16);

            int[] tokens = { 2, 7, 14, 9 };
            float[] seqLogits = new float[weights.VocabSize];
            float[] f16Logits = new float[weights.VocabSize];
            for (int i = 0; i < tokens.Length; i++)
            {
                model.Forward(tokens[i], i, null, seqLogits, computeLogits: i == tokens.Length - 1);
                f16Model.Forward(tokens[i], i, null, f16Logits, computeLogits: i == tokens.Length - 1);
            }

            model.ResetKvCache();
            float[] batchLogits = new float[weights.VocabSize];
            model.ForwardBatch(tokens, startPos: 0, batchLogits, computeLogits: true);

            foreach (float l in batchLogits) Assert.True(float.IsFinite(l));
            float cos = Fx1Vectors.CosineSimilarity(seqLogits, batchLogits);
            Assert.True(cos > 0.98f, $"CUDA ForwardBatch vs sequential Q4_0 cosine {cos:F4} <= 0.98");
            Assert.Equal(Glacier.Inference.Evaluation.KlDivergenceVerifier.ArgMax(f16Logits), Glacier.Inference.Evaluation.KlDivergenceVerifier.ArgMax(batchLogits));
        }

        // 2. End-to-end InferenceSession prefill + decode on CUDA Q4_0
        using var session = new InferenceSession(
            ggufPath,
            maxSeqLen: 32,
            device: nv.Id,
            engine: InferenceEngineType.BareMetal,
            kvPrecision: KvCachePrecision.Q4_0);

        Assert.Equal("Cuda", session.ActiveBackend);
        Assert.Equal(KvCachePrecision.Q4_0, session.KvPrecision);

        session.Prefill(new[] { 1, 4, 9 });
        session.ForwardToken(12, 3, computeLogits: true);

        float[] logits = new float[session.VocabSize];
        ((ISpeculativeTarget)session).CopyCurrentLogits(logits);
        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier1")]
[Trait("Hardware", "Gpu")]
public sealed unsafe class Tier1D3D12KvAndVramE2ETests
{
    // =========================================================================
    // F3: Direct3D 12 4-Bit KV Cache (T1-F3-01 .. T1-F3-05)
    // + F4 GPU Verification (T1-F4-04 .. T1-F4-05)
    // =========================================================================

    [UpgradeD3D12Fact]
    public void T1_F3_01_D3D12KvStoreAndAttentionQ4_0_Shaders_MatchCpuReference()
    {
        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int headDim = 64;
        const int maxSeqLen = 16;
        const int seqSteps = 4;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        using var ctx = new D3D12Context();
        uint nb = (headDim + 31u) / 32u;
        uint numPairs = (nb + 1u) / 2u;
        uint rowUints = nb * 4u + numPairs;
        ulong kvCacheBytes = (ulong)nHeadsKv * maxSeqLen * rowUints * sizeof(uint);
        ulong kvVecBytes = (ulong)(nHeadsKv * headDim * sizeof(float));
        ulong qOutBytes = (ulong)(nHeadsQ * headDim * sizeof(float));

        var storeParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigStore = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeParams));
        using var psoStore = ctx.CreatePipelineState(sigStore, ctx.CompileShader(D3D12Shaders.KvCacheStoreQ4_0));

        var attnParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigAttn = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        using var psoAttn = ctx.CreatePipelineState(sigAttn, ctx.CompileShader(D3D12Shaders.AttentionGqaQ4_0));

        using var dKCache = ctx.CreateDeviceBuffer(kvCacheBytes);
        using var dVCache = ctx.CreateDeviceBuffer(kvCacheBytes);
        using var dK = ctx.CreateDeviceBuffer(kvVecBytes);
        using var dV = ctx.CreateDeviceBuffer(kvVecBytes);
        using var dQ = ctx.CreateDeviceBuffer(qOutBytes);
        using var dOut = ctx.CreateDeviceBuffer(qOutBytes);

        float[] kSeq = new float[seqSteps * nHeadsKv * headDim];
        float[] vSeq = new float[seqSteps * nHeadsKv * headDim];
        float[] query = null!;

        for (int pos = 0; pos < seqSteps; pos++)
        {
            var (k, v, q) = Fx1Vectors.GenerateOrthogonalVectors(nHeadsQ, nHeadsKv, headDim, seed: 300 + pos);
            Array.Copy(k, 0, kSeq, pos * nHeadsKv * headDim, k.Length);
            Array.Copy(v, 0, vSeq, pos * nHeadsKv * headDim, v.Length);
            query = q;

            fixed (float* pK = k, pV = v)
            {
                ctx.CopyToDevice(dK, (IntPtr)pK, kvVecBytes);
                ctx.CopyToDevice(dV, (IntPtr)pV, kvVecBytes);
            }

            ctx.BeginCommands();
            var cmd = ctx.CommandList;
            cmd.ResourceBarrierTransition(dK, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(dV, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(dKCache, ResourceStates.Common, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(dVCache, ResourceStates.Common, ResourceStates.UnorderedAccess);

            cmd.SetComputeRootSignature(sigStore);
            cmd.SetPipelineState(psoStore);
            cmd.SetComputeRoot32BitConstant(0, (uint)nHeadsKv, 0);
            cmd.SetComputeRoot32BitConstant(0, (uint)headDim, 1);
            cmd.SetComputeRoot32BitConstant(0, (uint)maxSeqLen, 2);
            cmd.SetComputeRoot32BitConstant(0, (uint)pos, 3);
            cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeadsKv * numPairs, 1, 1);

            cmd.ResourceBarrierTransition(dK, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmd.ResourceBarrierTransition(dV, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmd.ResourceBarrierTransition(dKCache, ResourceStates.UnorderedAccess, ResourceStates.Common);
            cmd.ResourceBarrierTransition(dVCache, ResourceStates.UnorderedAccess, ResourceStates.Common);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();
        }

        fixed (float* pQ = query)
        {
            ctx.CopyToDevice(dQ, (IntPtr)pQ, qOutBytes);
        }

        ctx.BeginCommands();
        var cmdA = ctx.CommandList;
        cmdA.ResourceBarrierTransition(dQ, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmdA.ResourceBarrierTransition(dKCache, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmdA.ResourceBarrierTransition(dVCache, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmdA.ResourceBarrierTransition(dOut, ResourceStates.Common, ResourceStates.UnorderedAccess);

        cmdA.SetComputeRootSignature(sigAttn);
        cmdA.SetPipelineState(psoAttn);
        cmdA.SetComputeRoot32BitConstant(0, (uint)nHeadsQ, 0);
        cmdA.SetComputeRoot32BitConstant(0, (uint)nHeadsKv, 1);
        cmdA.SetComputeRoot32BitConstant(0, (uint)headDim, 2);
        cmdA.SetComputeRoot32BitConstant(0, (uint)maxSeqLen, 3);
        cmdA.SetComputeRoot32BitConstant(0, (uint)(seqSteps - 1), 4);
        cmdA.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(attnScale), 5);
        cmdA.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmdA.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmdA.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmdA.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        cmdA.Dispatch((uint)nHeadsQ, 1, 1);

        cmdA.ResourceBarrierTransition(dQ, ResourceStates.UnorderedAccess, ResourceStates.Common);
        cmdA.ResourceBarrierTransition(dKCache, ResourceStates.UnorderedAccess, ResourceStates.Common);
        cmdA.ResourceBarrierTransition(dVCache, ResourceStates.UnorderedAccess, ResourceStates.Common);
        cmdA.ResourceBarrierTransition(dOut, ResourceStates.UnorderedAccess, ResourceStates.Common);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] gpuOut = new float[nHeadsQ * headDim];
        fixed (float* pOut = gpuOut)
        {
            ctx.CopyToHost((IntPtr)pOut, dOut, qOutBytes);
        }

        float[] cpuRef = Fx1Vectors.ComputeSoftAttentionReference(query, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, seqSteps, attnScale);
        float cos = Fx1Vectors.CosineSimilarity(cpuRef, gpuOut);
        Assert.True(cos > 0.95f, $"D3D12 Q4_0 GQA attention cosine {cos:F4} <= 0.95");
    }

    [UpgradeD3D12Fact]
    public void T1_F3_02_Qwen2D3D12Model_Q4_0_AllocatesExactPackedKvBytes()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_mem.gguf"), dim: 256, ffnDim: 256, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 64);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var q4Model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Q4_0);
        using var f16Model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16);

        Assert.Equal(KvCachePrecision.Q4_0, q4Model.KvPrecision);
        ulong expected = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(weights.HeadCountKv, weights.HeadDim, 64, KvCachePrecision.Q4_0);
        Assert.Equal(expected, q4Model.KvCacheBytesPerLayer);
        Assert.Equal((long)expected * 2 * weights.BlockCount, q4Model.TotalKvCacheBytes);

        // Reference Fp8 byte size (1 byte/elem via KVCache.GetSlotByteSize) since D3D12 promotes Fp8 -> Fp16
        long fp8RefTotalBytes = (long)weights.BlockCount * 2L * weights.HeadCountKv * 64L * KVCache.GetSlotByteSize(weights.HeadDim, KvCachePrecision.Fp8);
        double ratioVsFp16 = (double)q4Model.TotalKvCacheBytes / f16Model.TotalKvCacheBytes;
        double ratioVsFp8 = (double)q4Model.TotalKvCacheBytes / fp8RefTotalBytes;
        Assert.True(ratioVsFp16 <= 0.3125, $"Expected D3D12 Q4_0/Fp16 <= 0.3125, got {ratioVsFp16:F4}");
        Assert.True(ratioVsFp8 <= 0.625, $"Expected D3D12 Q4_0/Fp8 <= 0.625, got {ratioVsFp8:F4}");
    }

    [UpgradeD3D12Fact]
    public void T1_F3_03_Qwen2D3D12Model_Q4_0_ForwardAndBatchMatchFp32()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_fwd.gguf"), dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var q4Model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0);
        using var f32Model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Fp32);

        int[] prompt = { 1, 6, 13, 4 };
        float[] q4Logits = new float[weights.VocabSize];
        float[] f32Logits = new float[weights.VocabSize];

        for (int pos = 0; pos < prompt.Length; pos++)
        {
            q4Model.Forward(prompt[pos], pos, q4Logits, computeLogits: pos == prompt.Length - 1);
            f32Model.Forward(prompt[pos], pos, f32Logits, computeLogits: pos == prompt.Length - 1);
        }

        Assert.All(q4Logits, x => Assert.True(float.IsFinite(x)));
        float cos = Fx1Vectors.CosineSimilarity(f32Logits, q4Logits);
        Assert.True(cos > 0.95f, $"D3D12 Q4_0 vs FP32 logits cosine {cos:F4} <= 0.95");

        q4Model.ResetKvCache();
        float[] batchLogits = new float[weights.VocabSize];
        q4Model.ForwardBatch(prompt, 0, batchLogits, computeLogits: true);
        Assert.True(Fx1Vectors.CosineSimilarity(q4Logits, batchLogits) > 0.98f);
    }

    [UpgradeD3D12Fact]
    public void T1_F3_04_InferenceSession_D3D12_Q4_0_EndToEndPrefillAndDecode()
    {
        var d3dDev = DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor == GpuVendor.Amd)
                  ?? DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor != GpuVendor.Cpu);
        Assert.NotNull(d3dDev);

        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_session.gguf"), dim: 64, ffnDim: 128, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32);

        using var session = new InferenceSession(
            ggufPath,
            maxSeqLen: 32,
            device: d3dDev.Id,
            engine: InferenceEngineType.DirectML,
            kvPrecision: KvCachePrecision.Q4_0);

        Assert.Equal("Direct3D12", session.ActiveBackend);
        Assert.Equal(KvCachePrecision.Q4_0, session.KvPrecision);

        session.Prefill(new[] { 2, 5, 8 });
        session.ForwardToken(10, 3, computeLogits: true);

        float[] logits = new float[session.VocabSize];
        ((ISpeculativeTarget)session).CopyCurrentLogits(logits);
        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
    }

    [UpgradeD3D12Fact]
    public void T1_F3_05_D3D12_MultiStage_Qwen3Hybrid_And_Gemma4BatchQ4_0_ShadersCompileAndExecute()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_stage.gguf"), dim: 256, ffnDim: 256, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 32);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        // Verify batched + Gemma4 Q4_0 KV HLSL shaders compile cleanly on DX12
        Assert.True(ctx.CompileShader(D3D12Shaders.KvCacheStoreBatchQ4_0).Length > 0);
        Assert.True(ctx.CompileShader(D3D12Shaders.AttentionBatchQ4_0).Length > 0);
        Assert.True(ctx.CompileShader(D3D12Shaders.Gemma4SwaAttentionQ4_0).Length > 0);
        Assert.True(ctx.CompileShader(D3D12Shaders.Qwen3AttentionQ4_0).Length > 0);

        // Verify Qwen3HybridD3D12Model honors KvCachePrecision.Q4_0 propagation without falling back to Fp16
        long hybridQ4TotalBytes;
        long hybridF16TotalBytes;
        using (var ctxHybridQ4 = new D3D12Context())
        using (var hybridQ4 = new Qwen3HybridD3D12Model(ctxHybridQ4, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0, explicitVramReserveBytes: 0UL))
        {
            Assert.Equal(KvCachePrecision.Q4_0, hybridQ4.KvPrecision);
            Assert.Equal(0UL, hybridQ4.EffectiveVramReserveBytes);
            Assert.Equal(Qwen2D3D12Model.ComputeD3D12KvLayerBytes(weights.HeadCountKv, weights.HeadDim, 32, KvCachePrecision.Q4_0), hybridQ4.KvCacheBytesPerLayer);
            hybridQ4TotalBytes = hybridQ4.TotalKvCacheBytes;

            float[] hybridLogits = new float[weights.VocabSize];
            hybridQ4.Forward(token: 3, pos: 0, hybridLogits, computeLogits: true);
            Assert.All(hybridLogits, x => Assert.True(float.IsFinite(x)));
        }

        using (var ctxHybridF16 = new D3D12Context())
        using (var hybridF16 = new Qwen3HybridD3D12Model(ctxHybridF16, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Fp16, explicitVramReserveBytes: 0UL))
        {
            hybridF16TotalBytes = hybridF16.TotalKvCacheBytes;
        }
        Assert.True((double)hybridQ4TotalBytes / hybridF16TotalBytes <= 0.3125);

        // Verify pipeline stage Q4_0 KV forwarding across split stages
        using var stage0 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0, startLayer: 0, layerCount: 1, isLastStage: false);
        using var stage1 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q4_0, startLayer: 1, layerCount: 1, isLastStage: true);

        float[] hidden = new float[weights.EmbeddingLength];
        float[] logits = new float[weights.VocabSize];
        stage0.ForwardStage(token: 3, pos: 0, inputX: default, outputX: hidden, logits: default, computeLogits: false);
        stage1.ForwardStage(token: 3, pos: 0, inputX: hidden, outputX: default, logits: logits, computeLogits: true);

        Assert.All(hidden, x => Assert.True(float.IsFinite(x)));
        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
    }

    [UpgradeD3D12Fact]
    public void T1_F4_04_D3D12Context_QueryVramHeadroom_ZeroReserveOverrideEliminatesFloor()
    {
        using var ctx = new D3D12Context();
        var defaultHeadroom = ctx.QueryVramHeadroom(explicitReserveBytes: null);
        var zeroHeadroom = ctx.QueryVramHeadroom(explicitReserveBytes: 0UL);

        Assert.True(defaultHeadroom.SafetyReserveBytes >= 350_000_000UL,
            $"Expected default safety reserve >= 350MB on WDDM, got {defaultHeadroom.SafetyReserveBytes}");
        Assert.Equal(0UL, zeroHeadroom.SafetyReserveBytes);
        Assert.True(zeroHeadroom.SafeAvailableBytes > defaultHeadroom.SafeAvailableBytes,
            $"Expected zero-reserve headroom ({zeroHeadroom.SafeAvailableBytes}) > default ({defaultHeadroom.SafeAvailableBytes})");

        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_res_cmp.gguf"), dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 16);
        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);

        using var defaultModel = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0, explicitVramReserveBytes: null);
        using var zeroModel = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0, explicitVramReserveBytes: 0UL);
        Assert.True(defaultModel.EffectiveVramReserveBytes >= 350_000_000UL);
        Assert.Equal(0UL, zeroModel.EffectiveVramReserveBytes);
    }

    [UpgradeD3D12Fact]
    public void T1_F4_05_Qwen2D3D12Model_And_InferenceSession_RespectVramReserveMbZero()
    {
        var d3dDev = DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor == GpuVendor.Amd)
                  ?? DeviceManager.GetDevices().FirstOrDefault(d => d.Vendor != GpuVendor.Cpu);
        Assert.NotNull(d3dDev);

        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("tiny_d3d12_vram0.gguf"), dim: 64, ffnDim: 128, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 32, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0, explicitVramReserveBytes: 0UL);
        Assert.Equal(0UL, model.EffectiveVramReserveBytes);

        using var session = new InferenceSession(
            ggufPath,
            maxSeqLen: 16,
            device: d3dDev.Id,
            engine: InferenceEngineType.DirectML,
            kvPrecision: KvCachePrecision.Q4_0,
            vramReserveMb: 0);
        Assert.NotNull(session.D3D12Model);
        Assert.Equal(0UL, ((Qwen2D3D12Model)session.D3D12Model!).EffectiveVramReserveBytes);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void T1_F4_06_D3D12TieredMoERunner_VramReserveMbZero_WiresEffectiveReserveZero()
    {
        if (!OperatingSystem.IsWindows() || !D3D12Context.IsSupported)
            return;

        using var tmp = new TempDirScope();
        string moePath = UpgradeScenarioFixtures.BuildTinyQwen2MoeModel(
            tmp.GetFilePath("t1_f4_03_moe.gguf"),
            dim: 64, ffnDim: 64, expertFfnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, numExperts: 4, topK: 2, vocabSize: 32, maxSeqLen: 16);

        using var d3dDefault = new TieredMoERunner(moePath, new TieredRunnerOptions { VramReserveMb = null, MaxSequenceLength = 16 });
        using var d3dZero = new TieredMoERunner(moePath, new TieredRunnerOptions { VramReserveMb = 0, MaxSequenceLength = 16 });

        Assert.True(d3dDefault.EffectiveVramReserveBytes >= 350_000_000UL);
        Assert.Equal(0UL, d3dZero.EffectiveVramReserveBytes);
        Assert.True(d3dZero.QueryVramHeadroom().SafeAvailableBytes > d3dDefault.QueryVramHeadroom().SafeAvailableBytes);
    }
}
