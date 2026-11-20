using System;
using System.IO;
using System.Linq;
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

public sealed unsafe class Tier2CpuKvAndVramBoundaryE2ETests
{
    // =========================================================================
    // F1 Boundaries: CPU 4-Bit KV Cache (T2-F1-01 .. T2-F1-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F1_01_CpuKvCache_Q4_0_InvalidBoundsAndShortSpans_ThrowExpectedExceptions()
    {
        using var cache = new KVCache(layers: 1, nHeadsKv: 2, headDim: 64, maxSeqLen: 8, precision: KvCachePrecision.Q4_0);

        float[] tooShort = new float[64]; // Needs 2 * 64 = 128
        float[] valid = new float[128];

        Assert.Throws<ArgumentException>(() => cache.Store(0, 0, tooShort, valid));
        Assert.Throws<ArgumentException>(() => cache.Store(0, 0, valid, tooShort));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Store(0, 8, valid, valid));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Store(0, -1, valid, valid));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetPhysicalSlot(8));
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F1_02_CpuKvCache_Q4_0_NonMultipleOf32HeadDim_PadsCleanlyWithoutOverrun()
    {
        const int headsKv = 2;
        const int headDim = 48; // Not divisible by 32 -> pads to 2 Q4_0 blocks (36 bytes/slot)
        const int maxSeq = 8;

        Assert.Equal(2 * sizeof(BlockQ4_0), KVCache.GetSlotByteSize(headDim, KvCachePrecision.Q4_0));

        using var cache = new KVCache(1, headsKv, headDim, maxSeq, precision: KvCachePrecision.Q4_0);
        var (k, v, q) = Fx1Vectors.GenerateOrthogonalVectors(2, headsKv, headDim, seed: 2102);

        cache.Store(0, 3, k, v);

        float[] kRead = new float[headDim];
        float[] vRead = new float[headDim];
        for (int h = 0; h < headsKv; h++)
        {
            cache.ReadKey(0, h, 3, kRead);
            cache.ReadValue(0, h, 3, vRead);

            Assert.All(kRead, x => Assert.True(float.IsFinite(x)));
            Assert.All(vRead, x => Assert.True(float.IsFinite(x)));
            Assert.True(Fx1Vectors.CosineSimilarity(k.AsSpan(h * headDim, headDim), kRead) > 0.96f);
            Assert.True(Fx1Vectors.CosineSimilarity(v.AsSpan(h * headDim, headDim), vRead) > 0.96f);
        }

        float[] outAcc = new float[headDim];
        fixed (float* pQ = q, pOut = outAcc)
        {
            float dot = cache.DotKey(0, 0, 3, pQ);
            Assert.True(float.IsFinite(dot));
            cache.AccumulateValue(0, 0, 3, 1.0f, pOut);
            Assert.True(Fx1Vectors.CosineSimilarity(v.AsSpan(0, headDim), outAcc) > 0.96f);
        }
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F1_03_CpuKvCache_Q4_0_ZeroAndExtremeMagnitudeVectors_RemainFinite()
    {
        const int headDim = 64;
        using var cache = new KVCache(1, 1, headDim, 8, precision: KvCachePrecision.Q4_0);

        // 1. All-zero vector (scale == 0 path)
        float[] zeros = new float[headDim];
        cache.Store(0, 0, zeros, zeros);

        float[] readZero = new float[headDim];
        cache.ReadKey(0, 0, 0, readZero);
        Assert.All(readZero, x => Assert.Equal(0.0f, x));

        float[] ones = Fx1Vectors.CreateOnes(headDim, 1.0f);
        fixed (float* pOnes = ones)
        {
            Assert.Equal(0.0f, cache.DotKey(0, 0, 0, pOnes));
        }

        // 2. Large 65000f vector (within Half scale limit after / 7.0f -> 9285.7f <= 65504f)
        float[] large = new float[headDim];
        for (int i = 0; i < headDim; i++) large[i] = (i % 2 == 0 ? 1f : -1f) * (65000.0f * (i + 1) / headDim);
        cache.Store(0, 1, large, large);

        float[] readLarge = new float[headDim];
        cache.ReadKey(0, 0, 1, readLarge);
        Assert.All(readLarge, x => Assert.True(float.IsFinite(x)));
        Assert.True(Fx1Vectors.CosineSimilarity(large, readLarge) > 0.96f);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F1_04_CpuKvCache_Q4_0_ResetAndDoubleDispose_ZeroBuffersAndCleanUpSafely()
    {
        var cache = new KVCache(2, 2, 64, 8, precision: KvCachePrecision.Q4_0);
        var (k, v, q) = Fx1Vectors.GenerateOrthogonalVectors(2, 2, 64, seed: 2104);

        cache.Store(0, 2, k, v);
        cache.Reset();

        float[] read = new float[64];
        cache.ReadKey(0, 0, 2, read);
        Assert.All(read, x => Assert.Equal(0.0f, x));

        fixed (float* pQ = q)
        {
            Assert.Equal(0.0f, cache.DotKey(0, 0, 2, pQ));
        }

        // Re-store after Reset
        cache.Store(0, 2, k, v);
        cache.ReadKey(0, 0, 2, read);
        Assert.True(Fx1Vectors.CosineSimilarity(k.AsSpan(0, 64), read) > 0.96f);

        cache.Dispose();
        cache.Dispose(); // Double-dispose must be idempotent
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F1_05_CpuKvCache_Q4_0_AsymmetricKeyAndValueHeadDims_IsolatesBuffers()
    {
        const int headsKv = 2;
        const int kDim = 96;
        const int vDim = 64;
        using var cache = new KVCache(1, headsKv, kDim, maxSeqLen: 8, vHeadDim: vDim, precision: KvCachePrecision.Q4_0);

        Assert.NotEqual(cache.KeyBufferBytes, cache.ValueBufferBytes);
        Assert.Equal(headsKv * 8L * 3 * sizeof(BlockQ4_0), cache.KeyBufferBytes);
        Assert.Equal(headsKv * 8L * 2 * sizeof(BlockQ4_0), cache.ValueBufferBytes);

        var (k, _, _) = Fx1Vectors.GenerateOrthogonalVectors(2, headsKv, kDim, seed: 2105);
        var (_, v, _) = Fx1Vectors.GenerateOrthogonalVectors(2, headsKv, vDim, seed: 2106);

        cache.Store(0, 1, k, v);

        float[] kRead = new float[kDim];
        float[] vRead = new float[vDim];
        cache.ReadKey(0, 1, 1, kRead);
        cache.ReadValue(0, 1, 1, vRead);

        Assert.True(Fx1Vectors.CosineSimilarity(k.AsSpan(kDim, kDim), kRead) > 0.96f);
        Assert.True(Fx1Vectors.CosineSimilarity(v.AsSpan(vDim, vDim), vRead) > 0.96f);
    }

    // =========================================================================
    // F4 Boundaries: VramReserveMb Guard & Clamping (T2-F4-01 .. T2-F4-05)
    // =========================================================================

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F4_01_WddmVramGuard_And_InferenceSession_NegativeReserve_ThrowArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => WddmVramGuard.ResolveExplicitReserveBytes(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => WddmVramGuard.ResolveExplicitReserveBytes(int.MinValue));

        using var tmp = new TempDirScope();
        string path = SyntheticGgufBuilder.BuildTinyQwen2Model(tmp.GetFilePath("neg_vram.gguf"), dim: 64, ffnDim: 64, nLayers: 1);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new InferenceSession(path, maxSeqLen: 16, device: "cpu", engine: InferenceEngineType.Cpu, vramReserveMb: -5));
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F4_02_WddmVramGuard_EvaluateHeadroom_NeverUnderflowsWhenUsageExceedsBudget()
    {
        const ulong dedicated = 8UL * 1024 * 1024 * 1024;
        const ulong osBudget = 4UL * 1024 * 1024 * 1024;
        const ulong excessiveUsage = 6UL * 1024 * 1024 * 1024;

        var headroomDefault = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, excessiveUsage, isWindowsWddm: true, explicitReserveBytes: null);
        Assert.Equal(0UL, headroomDefault.SafeAvailableBytes);

        var headroomHugeReserve = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, currentUsageBytes: 0, isWindowsWddm: true, explicitReserveBytes: 16UL * 1024 * 1024 * 1024);
        Assert.Equal(0UL, headroomHugeReserve.SafeAvailableBytes);

        var headroomZeroDed = WddmVramGuard.EvaluateHeadroom(0UL, 0UL, 0UL, isWindowsWddm: true, explicitReserveBytes: 0UL);
        Assert.Equal(0UL, headroomZeroDed.SafetyReserveBytes);
        Assert.Equal(0UL, headroomZeroDed.SafeAvailableBytes);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F4_03_WddmVramGuard_ComputeRecommendedSlotCount_HandlesZeroBudgetAndExplicitZeroReserve()
    {
        const ulong dedicated = 8UL * 1024 * 1024 * 1024;
        const ulong osBudget = 7_500_000_000UL;
        const ulong used = 6_000_000_000UL;
        const ulong slotBytes = 64UL * 1024 * 1024;

        var snapDefault = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, used, isWindowsWddm: true, explicitReserveBytes: null);
        var snapZero = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, used, isWindowsWddm: true, explicitReserveBytes: 0UL);
        var snapExhausted = WddmVramGuard.EvaluateHeadroom(dedicated, osBudget, osBudget + 1000, isWindowsWddm: true, explicitReserveBytes: 0UL);

        int slotsDefault = WddmVramGuard.ComputeRecommendedSlotCount(snapDefault, slotBytes, currentActiveSlots: 0, maxSlotCapacity: 128, minSlotFloor: 0);
        int slotsZeroReserve = WddmVramGuard.ComputeRecommendedSlotCount(snapZero, slotBytes, currentActiveSlots: 0, maxSlotCapacity: 128, minSlotFloor: 0);

        Assert.True(slotsZeroReserve > slotsDefault,
            $"Zero reserve slots ({slotsZeroReserve}) must exceed default reserve slots ({slotsDefault}).");

        int slotsExhausted = WddmVramGuard.ComputeRecommendedSlotCount(snapExhausted, slotBytes, currentActiveSlots: 0, maxSlotCapacity: 128, minSlotFloor: 0);
        Assert.Equal(0, slotsExhausted);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F4_04_TieredRunnerOptions_NegativeVramReserveMb_ThrowsOnHeadroomQuery_AndLargeReserveClampsSafely()
    {
        using var moe = new SyntheticMoeFixture(numLayers: 1, numExperts: 4, topK: 2, dim: 32, expertFfnDim: 32);
        var negOpts = new TieredRunnerOptions { VramReserveMb = -1 };
        using var negRunner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 4), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: negOpts, lmHeadWeights: moe.LmHead);
        Assert.Throws<ArgumentOutOfRangeException>(() => negRunner.QueryVramHeadroom());

        var hugeOpts = new TieredRunnerOptions { VramReserveMb = 100_000 }; // ~100 GB
        using var hugeRunner = new TieredMoERunner(
            moe.Layers, moe.Dim, moe.ExpertFfnDim, moe.NumExperts, moe.TopK, moe.VocabSize,
            normTopK: true, new ExpertRoutingProfile(1, 4), new BandwidthCoScheduler(),
            slotManager: null, hostArena: null, options: hugeOpts, lmHeadWeights: moe.LmHead);

        Assert.Equal(100_000UL * 1024 * 1024, hugeRunner.EffectiveVramReserveBytes);
        var headroom = hugeRunner.QueryVramHeadroom();
        Assert.Equal(0UL, headroom.SafeAvailableBytes);
    }

    [Fact]
    [Trait("Suite", "UpgradesE2E")]
    [Trait("Tier", "Tier2")]
    public void T2_F4_05_WddmVramGuard_UnifiedAndNonWddmPaths_RespectExplicitZeroVsDefault()
    {
        const ulong unifiedMem = 16UL * 1024 * 1024 * 1024;
        ulong defaultUnified = WddmVramGuard.ResolveModelSafetyReserveBytes(unifiedMem, isWindowsWddm: true, isUnifiedMemory: true, explicitReserveBytes: null);
        ulong zeroUnified = WddmVramGuard.ResolveModelSafetyReserveBytes(unifiedMem, isWindowsWddm: true, isUnifiedMemory: true, explicitReserveBytes: 0UL);
        ulong nonWddmDefault = WddmVramGuard.ResolveModelSafetyReserveBytes(unifiedMem, isWindowsWddm: false, isUnifiedMemory: false, explicitReserveBytes: null);
        ulong nonWddmZero = WddmVramGuard.ResolveModelSafetyReserveBytes(unifiedMem, isWindowsWddm: false, isUnifiedMemory: false, explicitReserveBytes: 0UL);

        Assert.True(defaultUnified > 0UL);
        Assert.Equal(0UL, zeroUnified);
        Assert.True(nonWddmDefault >= 0UL);
        Assert.Equal(0UL, nonWddmZero);
    }
}

internal static class Fx1VectorExtensions
{
    extension(Fx1Vectors)
    {
        public static float[] CreateOnes(int length, float val)
        {
            float[] arr = new float[length];
            Array.Fill(arr, val);
            return arr;
        }
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier2")]
[Trait("Hardware", "Gpu")]
public sealed unsafe class Tier2CudaKvBoundaryE2ETests
{
    // =========================================================================
    // F2 Boundaries: CUDA 4-Bit KV Cache (T2-F2-01 .. T2-F2-05)
    // =========================================================================

    [UpgradeCudaFact]
    public void T2_F2_01_CudaKvQ4_0_ZeroVectorsAndOutOfRangePos_StayFiniteAndSafe()
    {
        const int nHeadsQ = 2;
        const int nHeadsKv = 2;
        const int headDim = 64;
        const int maxSeqLen = 8;

        using var gpu = new GpuContext(0);
        gpu.MakeCurrent();
        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStore, module, "kv_cache_store_q4_0"), "GetStore");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttn, module, "attention_gqa_q4_0"), "GetAttn");

            int blocksPerHead = (headDim + 31) / 32;
            nuint kvBytes = (nuint)((long)nHeadsKv * maxSeqLen * blocksPerHead * sizeof(BlockQ4_0));
            nuint kvVecBytes = (nuint)(nHeadsKv * headDim * sizeof(float));
            nuint qOutBytes = (nuint)(nHeadsQ * headDim * sizeof(float));

            IntPtr dKCache = gpu.AllocateDevice(kvBytes);
            IntPtr dVCache = gpu.AllocateDevice(kvBytes);
            IntPtr dK = gpu.AllocateDevice(kvVecBytes);
            IntPtr dV = gpu.AllocateDevice(kvVecBytes);
            IntPtr dQ = gpu.AllocateDevice(qOutBytes);
            IntPtr dOut = gpu.AllocateDevice(qOutBytes);
            IntPtr dScores = gpu.AllocateDevice((nuint)(nHeadsQ * maxSeqLen * sizeof(float)));

            try
            {
                CuDriver.Check(CuDriver.MemsetD8(dKCache, 0, kvBytes), "ZeroKCache");
                CuDriver.Check(CuDriver.MemsetD8(dVCache, 0, kvBytes), "ZeroVCache");
                CuDriver.Check(CuDriver.MemsetD8(dK, 0, kvVecBytes), "ZeroK");
                CuDriver.Check(CuDriver.MemsetD8(dV, 0, kvVecBytes), "ZeroV");
                CuDriver.Check(CuDriver.MemsetD8(dQ, 0, qOutBytes), "ZeroQ");

                IntPtr pKC = dKCache, pVC = dVCache, pKd = dK, pVd = dV;
                int nhKv = nHeadsKv, hd = headDim, ms = maxSeqLen, pos = 0;
                void** pStore = stackalloc void*[8];
                pStore[0] = &pKC; pStore[1] = &pVC; pStore[2] = &pKd; pStore[3] = &pVd;
                pStore[4] = &nhKv; pStore[5] = &hd; pStore[6] = &ms; pStore[7] = &pos;
                CuDriver.Check(CuDriver.LaunchKernel(fnStore, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pStore, IntPtr.Zero), "StoreZero");

                // Out-of-bounds store (pos = maxSeqLen) should early-return safely
                pos = maxSeqLen;
                CuDriver.Check(CuDriver.LaunchKernel(fnStore, 1, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pStore, IntPtr.Zero), "StoreOOB");

                IntPtr pQd = dQ, pOutD = dOut, pScoresD = dScores;
                int nhQ = nHeadsQ, pos0 = 0;
                float scale = 1.0f / MathF.Sqrt(headDim);
                void** pAttn = stackalloc void*[11];
                pAttn[0] = &pQd; pAttn[1] = &pKC; pAttn[2] = &pVC; pAttn[3] = &pOutD; pAttn[4] = &pScoresD;
                pAttn[5] = &nhQ; pAttn[6] = &nhKv; pAttn[7] = &hd; pAttn[8] = &ms; pAttn[9] = &pos0; pAttn[10] = &scale;
                CuDriver.Check(CuDriver.LaunchKernel(fnAttn, (uint)nHeadsQ, 1, 1, 64, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttn, IntPtr.Zero), "AttnZero");
                gpu.Synchronize();

                float[] hostOut = new float[nHeadsQ * headDim];
                fixed (float* pHostOut = hostOut)
                {
                    gpu.CopyToHost((IntPtr)pHostOut, dOut, qOutBytes);
                }

                Assert.All(hostOut, v => Assert.Equal(0.0f, v));
            }
            finally
            {
                gpu.FreeDevice(dKCache); gpu.FreeDevice(dVCache);
                gpu.FreeDevice(dK); gpu.FreeDevice(dV);
                gpu.FreeDevice(dQ); gpu.FreeDevice(dOut); gpu.FreeDevice(dScores);
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [UpgradeCudaFact]
    public void T2_F2_02_Qwen2GpuModel_Q4_0_HighGqaRatioAndNon32HeadDim_ForwardMatchesFp32()
    {
        using var tmp = new TempDirScope();
        // High GQA 8:1 with headDim = 48 (dim = 384, nHeads = 8, nHeadsKv = 1 -> headDim = 48, non-multiple of 32)
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("cuda_gqa8_hd48.gguf"),
            dim: 384, ffnDim: 128, nLayers: 1, nHeads: 8, nHeadsKv: 1, vocabSize: 16, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);

        using var q4 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0);
        using var f32 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp32);

        float[] q4Logits = new float[weights.VocabSize];
        float[] f32Logits = new float[weights.VocabSize];
        q4.Forward(1, 0, null, q4Logits, computeLogits: false);
        q4.Forward(4, 1, null, q4Logits, computeLogits: true);
        f32.Forward(1, 0, null, f32Logits, computeLogits: false);
        f32.Forward(4, 1, null, f32Logits, computeLogits: true);

        Assert.All(q4Logits, x => Assert.True(float.IsFinite(x)));
        Assert.True(Fx1Vectors.CosineSimilarity(f32Logits, q4Logits) > 0.95f);
    }

    [UpgradeCudaFact]
    public void T2_F2_03_Qwen2GpuModel_AutoPrecisionThreshold_And_ExplicitQ4OverrideAt4097()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("cuda_auto_seq.gguf"),
            dim: 64, ffnDim: 64, nLayers: 1, nHeads: 2, nHeadsKv: 1, vocabSize: 16, maxSeqLen: 4097);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);

        using var autoShort = new Qwen2GpuModel(gpu, weights, maxSeqLen: 4096, kvPrecision: KvCachePrecision.Auto);
        using var autoLong = new Qwen2GpuModel(gpu, weights, maxSeqLen: 4097, kvPrecision: KvCachePrecision.Auto);
        using var q4Long = new Qwen2GpuModel(gpu, weights, maxSeqLen: 4097, kvPrecision: KvCachePrecision.Q4_0);

        Assert.Equal(KvCachePrecision.Fp16, autoShort.KvPrecision);
        Assert.Equal(KvCachePrecision.Fp8, autoLong.KvPrecision);
        Assert.Equal(KvCachePrecision.Q4_0, q4Long.KvPrecision);
        Assert.True(q4Long.TotalKvCacheBytes < autoLong.TotalKvCacheBytes);
    }

    [UpgradeCudaFact]
    public void T2_F2_04_Qwen2GpuModel_OutOfRangePos_ThrowsCleanlyWithoutPoisoningContext()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("cuda_pos_oob.gguf"),
            dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 16, maxSeqLen: 8);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);
        using var model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 8, kvPrecision: KvCachePrecision.Q4_0);

        float[] logits = new float[weights.VocabSize];
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(1, pos: -1, null, logits));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(1, pos: 8, null, logits));

        model.Forward(1, pos: 0, null, logits, computeLogits: true);
        Assert.All(logits, x => Assert.True(float.IsFinite(x)));
    }

    [UpgradeCudaFact]
    public void T2_F2_05_Qwen2GpuModel_Q4_0_ResetKvCache_ProducesBitIdenticalReplay()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("cuda_reset_det.gguf"),
            dim: 64, ffnDim: 64, nLayers: 2, nHeads: 4, nHeadsKv: 2, vocabSize: 16, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var gpu = new GpuContext(0);
        using var model = new Qwen2GpuModel(gpu, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0);

        float[] run1 = new float[weights.VocabSize];
        float[] run2 = new float[weights.VocabSize];

        model.Forward(2, 0, null, run1, false);
        model.Forward(5, 1, null, run1, true);

        // Pollute cache at pos 2, then reset
        model.Forward(9, 2, null, run2, false);
        model.ResetKvCache();

        model.Forward(2, 0, null, run2, false);
        model.Forward(5, 1, null, run2, true);

        Assert.True(Fx1Vectors.CosineSimilarity(run1, run2) > 0.9999f);
        for (int i = 0; i < run1.Length; i++)
        {
            Assert.True(MathF.Abs(run1[i] - run2[i]) < 1e-4f);
        }
    }
}

[Collection("SequentialGpu")]
[Trait("Suite", "UpgradesE2E")]
[Trait("Tier", "Tier2")]
[Trait("Hardware", "Gpu")]
public sealed unsafe class Tier2D3D12KvBoundaryE2ETests
{
    // =========================================================================
    // F3 Boundaries: Direct3D 12 4-Bit KV Cache (T2-F3-01 .. T2-F3-05)
    // =========================================================================

    [UpgradeD3D12Fact]
    public void T2_F3_01_D3D12_ComputeKvLayerBytes_BoundariesAndZeroVectorsRemainFinite()
    {
        // Check split-pair row packing for multiple headDims including non-32-aligned (48) and odd-block (96)
        Assert.Equal((ulong)(2 * 16 * (1 * 4 + 1) * 4), Qwen2D3D12Model.ComputeD3D12KvLayerBytes(2, 32, 16, KvCachePrecision.Q4_0));
        Assert.Equal((ulong)(2 * 16 * (2 * 4 + 1) * 4), Qwen2D3D12Model.ComputeD3D12KvLayerBytes(2, 48, 16, KvCachePrecision.Q4_0));
        Assert.Equal((ulong)(2 * 16 * (3 * 4 + 2) * 4), Qwen2D3D12Model.ComputeD3D12KvLayerBytes(2, 96, 16, KvCachePrecision.Q4_0));

        // Direct shader execution with zero K/V/Q produces finite 0.0f output
        using var ctx = new D3D12Context();
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

        const int headDim = 96; // 3 blocks (odd nb)
        ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(1, headDim, 8, KvCachePrecision.Q4_0);
        ulong vecBytes = headDim * sizeof(float);

        using var dKCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dVCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dZero = ctx.CreateDeviceBuffer(vecBytes);
        using var dOut = ctx.CreateDeviceBuffer(vecBytes);

        byte[] zeros = new byte[vecBytes];
        fixed (byte* pZ = zeros) ctx.CopyToDevice(dZero, (IntPtr)pZ, vecBytes);

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.ResourceBarrierTransition(dZero, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(dKCache, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(dVCache, ResourceStates.Common, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(dOut, ResourceStates.Common, ResourceStates.UnorderedAccess);

        cmd.SetComputeRootSignature(sigStore);
        cmd.SetPipelineState(psoStore);
        cmd.SetComputeRoot32BitConstant(0, 1u, 0);
        cmd.SetComputeRoot32BitConstant(0, (uint)headDim, 1);
        cmd.SetComputeRoot32BitConstant(0, 8u, 2);
        cmd.SetComputeRoot32BitConstant(0, 0u, 3);
        cmd.SetComputeRootUnorderedAccessView(1, dZero.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dZero.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
        cmd.Dispatch(2, 1, 1); // numPairs = (3+1)/2 = 2
        cmd.ResourceBarrierUnorderedAccessView(null!);

        cmd.SetComputeRootSignature(sigAttn);
        cmd.SetPipelineState(psoAttn);
        cmd.SetComputeRoot32BitConstant(0, 1u, 0);
        cmd.SetComputeRoot32BitConstant(0, 1u, 1);
        cmd.SetComputeRoot32BitConstant(0, (uint)headDim, 2);
        cmd.SetComputeRoot32BitConstant(0, 8u, 3);
        cmd.SetComputeRoot32BitConstant(0, 0u, 4);
        cmd.SetComputeRoot32BitConstant(0, BitConverter.SingleToUInt32Bits(1.0f / MathF.Sqrt(headDim)), 5);
        cmd.SetComputeRootUnorderedAccessView(1, dZero.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);
        cmd.Dispatch(1, 1, 1);

        cmd.ResourceBarrierTransition(dOut, ResourceStates.UnorderedAccess, ResourceStates.Common);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] hostOut = new float[headDim];
        fixed (float* pOut = hostOut) ctx.CopyToHost((IntPtr)pOut, dOut, vecBytes);
        Assert.All(hostOut, x => Assert.Equal(0.0f, x));
    }

    [UpgradeD3D12Fact]
    public void T2_F3_02_Qwen2D3D12Model_Q4_0_OddBlockAndNon32HeadDim_ForwardMatchesFp32()
    {
        using var tmp = new TempDirScope();
        // dim=192, nHeads=2, nHeadsKv=1 => headDim=96 (3 Q4_0 blocks, odd nb)
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("d3d12_hd96.gguf"),
            dim: 192, ffnDim: 128, nLayers: 1, nHeads: 2, nHeadsKv: 1, vocabSize: 16, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var q4 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0);
        using var f32 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp32);

        float[] q4Logits = new float[weights.VocabSize];
        float[] f32Logits = new float[weights.VocabSize];
        q4.Forward(1, 0, q4Logits, computeLogits: false);
        q4.Forward(3, 1, q4Logits, computeLogits: true);
        f32.Forward(1, 0, f32Logits, computeLogits: false);
        f32.Forward(3, 1, f32Logits, computeLogits: true);

        Assert.All(q4Logits, x => Assert.True(float.IsFinite(x)));
        Assert.True(Fx1Vectors.CosineSimilarity(f32Logits, q4Logits) > 0.95f);
    }

    [UpgradeD3D12Fact]
    public void T2_F3_03_Qwen2D3D12Model_OutOfRangePosAndResetReplay_AreSafeAndDeterministic()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("d3d12_oob_reset.gguf"),
            dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 16, maxSeqLen: 8);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();
        using var model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 8, kvPrecision: KvCachePrecision.Q4_0);

        float[] run1 = new float[weights.VocabSize];
        float[] run2 = new float[weights.VocabSize];

        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(1, pos: -1, run1));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Forward(1, pos: 8, run1));

        model.Forward(2, 0, run1, false);
        model.Forward(4, 1, run1, true);

        model.ResetKvCache();
        model.Forward(2, 0, run2, false);
        model.Forward(4, 1, run2, true);

        for (int i = 0; i < run1.Length; i++)
        {
            Assert.True(MathF.Abs(run1[i] - run2[i]) < 1e-4f);
        }
    }

    [UpgradeD3D12Fact]
    public void T2_F3_04_Qwen2D3D12Model_InvalidKvPrecisionEnum99_FallsBackToFp32Safely()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("d3d12_invalid_kv.gguf"),
            dim: 64, ffnDim: 64, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 16, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var invalidModel = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: (KvCachePrecision)99);
        using var fp32Model = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp32);

        Assert.Equal(fp32Model.KvCacheBytesPerLayer, invalidModel.KvCacheBytesPerLayer);

        float[] invLogits = new float[weights.VocabSize];
        float[] fp32Logits = new float[weights.VocabSize];
        invalidModel.Forward(3, 0, invLogits, computeLogits: true);
        fp32Model.Forward(3, 0, fp32Logits, computeLogits: true);

        Assert.All(invLogits, x => Assert.True(float.IsFinite(x)));
        for (int i = 0; i < invLogits.Length; i++)
        {
            Assert.True(MathF.Abs(invLogits[i] - fp32Logits[i]) < 1e-4f);
        }
    }

    [UpgradeD3D12Fact]
    public void T2_F3_05_Qwen2D3D12Model_AutoAndFp8NormalizeToFp16_WhileQ4RetainsCompactPacking()
    {
        using var tmp = new TempDirScope();
        string ggufPath = SyntheticGgufBuilder.BuildTinyQwen2Model(
            tmp.GetFilePath("d3d12_norm_prec.gguf"),
            dim: 256, ffnDim: 256, nLayers: 1, nHeads: 4, nHeadsKv: 2, vocabSize: 16, maxSeqLen: 16);

        using var gguf = GgufFile.Open(ggufPath);
        var weights = new ModelWeights(gguf);
        using var ctx = new D3D12Context();

        using var mAuto = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Auto);
        using var mFp8 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Fp8);
        using var mQ4 = new Qwen2D3D12Model(ctx, weights, maxSeqLen: 16, kvPrecision: KvCachePrecision.Q4_0);

        Assert.Equal(KvCachePrecision.Fp16, mAuto.KvPrecision);
        Assert.Equal(KvCachePrecision.Fp16, mFp8.KvPrecision);
        Assert.Equal(KvCachePrecision.Q4_0, mQ4.KvPrecision);
        Assert.Equal(mAuto.TotalKvCacheBytes, mFp8.TotalKvCacheBytes);
        Assert.True(mQ4.TotalKvCacheBytes < mAuto.TotalKvCacheBytes / 3);
    }
}
