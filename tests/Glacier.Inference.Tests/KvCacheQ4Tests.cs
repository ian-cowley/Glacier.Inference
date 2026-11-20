namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;
using Xunit;

[CollectionDefinition("M1GpuSerial", DisableParallelization = true)]
public class M1GpuSerialCollection { }

[Collection("M1GpuSerial")]
public unsafe class KvCacheQ4Tests
{
    [Fact]
    public void KvCachePrecision_EnumAndExtensions_ParseAndDisplayCorrectly()
    {
        Assert.Equal(4, (int)KvCachePrecision.Q4_0);
        Assert.Equal(KvCachePrecision.Q4_0, KvCachePrecision.Int4);
        Assert.Equal("Q4_0", KvCachePrecision.Q4_0.ToDisplayName());
        Assert.Equal("Q4_0", KvCachePrecision.Int4.ToDisplayName());
        Assert.Equal("Fp8", KvCachePrecision.Fp8.ToDisplayName());
        Assert.Equal("Fp16", KvCachePrecision.Fp16.ToDisplayName());
        Assert.Equal("Fp32", KvCachePrecision.Fp32.ToDisplayName());
        Assert.Equal("Auto", KvCachePrecision.Auto.ToDisplayName());

        Assert.True(KvCachePrecisionExtensions.TryParse("auto", out var p)); Assert.Equal(KvCachePrecision.Auto, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("FP32", out p)); Assert.Equal(KvCachePrecision.Fp32, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("f32", out p)); Assert.Equal(KvCachePrecision.Fp32, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("fp16", out p)); Assert.Equal(KvCachePrecision.Fp16, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("F16", out p)); Assert.Equal(KvCachePrecision.Fp16, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("fp8", out p)); Assert.Equal(KvCachePrecision.Fp8, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("q8_0", out p)); Assert.Equal(KvCachePrecision.Fp8, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("e4m3", out p)); Assert.Equal(KvCachePrecision.Fp8, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("q4_0", out p)); Assert.Equal(KvCachePrecision.Q4_0, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("q4", out p)); Assert.Equal(KvCachePrecision.Q4_0, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("int4", out p)); Assert.Equal(KvCachePrecision.Int4, p);

        Assert.False(KvCachePrecisionExtensions.TryParse(null, out _));
        Assert.False(KvCachePrecisionExtensions.TryParse("", out _));
        Assert.False(KvCachePrecisionExtensions.TryParse("unknown_fmt", out _));
    }

    [Theory]
    [InlineData(64)]
    [InlineData(80)]
    [InlineData(128)]
    [InlineData(256)]
    public void CpuKvCache_Q4_0_And_Int4_ByteCapacity_And_RoundTripAccuracy(int headDim)
    {
        const int layers = 2;
        const int headsKv = 4;
        const int maxSeq = 128;

        using var fp32Cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp32);
        using var fp16Cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp16);
        using var fp8Cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Fp8);
        using var q4Cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Q4_0);
        using var int4Cache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Int4);
        using var autoCache = new KVCache(layers, headsKv, headDim, maxSeq, precision: KvCachePrecision.Auto);

        Assert.Equal(KvCachePrecision.Fp32, autoCache.Precision);
        Assert.Equal(KvCachePrecision.Q4_0, q4Cache.Precision);
        Assert.Equal(q4Cache.TotalBytes, int4Cache.TotalBytes);
        Assert.Equal(q4Cache.TotalBytes, q4Cache.GetByteSize());

        // 18 bytes/32 elements = 0.5625 B/elem (for aligned dims >= 64: ~1.78x over Fp8, ~3.56x over Fp16, ~7.11x over Fp32;
        // for headDim=80, blocks=3 => 54B vs 80B Fp8 (1.48x) and 160B Fp16 (2.96x))
        if (headDim % 32 == 0)
        {
            Assert.True(q4Cache.TotalBytes * 1.75 <= fp8Cache.TotalBytes, $"Q4_0 ({q4Cache.TotalBytes}) should be ~2x smaller than Fp8 ({fp8Cache.TotalBytes})");
            Assert.True(q4Cache.TotalBytes * 3.5 <= fp16Cache.TotalBytes, $"Q4_0 ({q4Cache.TotalBytes}) should be ~4x smaller than Fp16 ({fp16Cache.TotalBytes})");
            Assert.True(q4Cache.TotalBytes * 7.0 <= fp32Cache.TotalBytes, $"Q4_0 ({q4Cache.TotalBytes}) should be ~7x+ smaller than Fp32 ({fp32Cache.TotalBytes})");
        }
        else
        {
            Assert.True(q4Cache.TotalBytes < fp8Cache.TotalBytes);
            Assert.True(q4Cache.TotalBytes * 2.9 <= fp16Cache.TotalBytes);
        }

        // Direct float* pointer access must throw for non-FP32 caches
        Assert.Throws<InvalidOperationException>(() => q4Cache.GetKeyPtr(0, 0, 0));
        Assert.Throws<InvalidOperationException>(() => q4Cache.GetValuePtr(0, 0, 0));
        Assert.Throws<InvalidOperationException>(() => fp8Cache.GetKeyPtr(0, 0, 0));
        Assert.Throws<InvalidOperationException>(() => fp16Cache.GetValuePtr(0, 0, 0));

        var rng = new Random(1337 + headDim);
        var kVec = new float[headsKv * headDim];
        var vVec = new float[headsKv * headDim];
        var qVec = new float[headDim];

        for (int pos = 0; pos < 6; pos++)
        {
            for (int i = 0; i < kVec.Length; i++)
            {
                kVec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                vVec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            fp32Cache.Store(1, pos, kVec, vVec);
            q4Cache.Store(1, pos, kVec, vVec);
            fp8Cache.Store(1, pos, kVec, vVec);
            fp16Cache.Store(1, pos, kVec, vVec);
        }

        for (int i = 0; i < headDim; i++)
        {
            qVec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var readK = new float[headDim];
        var refK = new float[headDim];
        var outFp32 = new float[headDim];
        var outQ4 = new float[headDim];

        fixed (float* pQ = qVec)
        fixed (float* pOut32 = outFp32)
        fixed (float* pOutQ4 = outQ4)
        {
            for (int h = 0; h < headsKv; h++)
            {
                Array.Clear(outFp32);
                Array.Clear(outQ4);

                for (int pos = 0; pos < 6; pos++)
                {
                    fp32Cache.ReadKey(1, h, pos, refK);
                    q4Cache.ReadKey(1, h, pos, readK);
                    Assert.True(CosineSimilarity(refK, readK) > 0.985f);

                    float dotRef = fp32Cache.DotKey(1, h, pos, pQ);
                    float dotQ4 = q4Cache.DotKey(1, h, pos, pQ);
                    float dotFp8 = fp8Cache.DotKey(1, h, pos, pQ);
                    float dotFp16 = fp16Cache.DotKey(1, h, pos, pQ);

                    Assert.True(MathF.Abs(dotRef - dotFp16) < 0.05f, $"FP16 dot mismatch: ref={dotRef}, fp16={dotFp16}");
                    Assert.True(MathF.Abs(dotRef - dotFp8) < 0.05f * MathF.Sqrt(headDim), $"FP8 dot mismatch: ref={dotRef}, fp8={dotFp8}");
                    Assert.True(MathF.Abs(dotRef - dotQ4) < 0.20f * MathF.Sqrt(headDim), $"Q4 dot mismatch: ref={dotRef}, q4={dotQ4}");

                    float w = 1.0f / 6.0f;
                    fp32Cache.AccumulateValue(1, h, pos, w, pOut32);
                    q4Cache.AccumulateValue(1, h, pos, w, pOutQ4);
                }

                Assert.True(CosineSimilarity(outFp32, outQ4) > 0.985f);
            }
        }
    }

    [Fact]
    public void CpuKvCache_RingBuffer_Q4_0_WrapsCorrectly()
    {
        const int headDim = 64;
        using var cache = new KVCache(layers: 1, nHeadsKv: 2, headDim: headDim, maxSeqLen: 16, slidingWindow: 4, isRingBuffer: true, precision: KvCachePrecision.Q4_0);

        var k = new float[2 * headDim];
        var v = new float[2 * headDim];
        for (int pos = 0; pos < 6; pos++)
        {
            Array.Fill(k, pos + 1.0f);
            Array.Fill(v, -(pos + 1.0f));
            cache.Store(0, pos, k, v);
        }

        // Position 5 overwrote slot 1 (5 % 4 == 1)
        var readBack = new float[headDim];
        cache.ReadKey(0, 0, 1, readBack);
        Assert.All(readBack, x => Assert.InRange(x, 5.5f, 6.5f));
    }

    [Fact]
    public void CpuQwen2Model_SyntheticForward_Q4_0_Matches_Fp32()
    {
        string tempGguf = CreateSyntheticQwen2Gguf(dim: 128, ffnDim: 128, nHeads: 2, nHeadsKv: 1, layers: 2, vocab: 64, useQ8Weights: true);
        try
        {
            using var gguf = GgufFile.Open(tempGguf);
            using var weights = new ModelWeights(gguf);

            using var modelFp32 = new Qwen2Model(weights, maxSeqLen: 64);
            using var modelQ4 = new Qwen2Model(weights, maxSeqLen: 64);
            using var kvFp32 = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 64, precision: KvCachePrecision.Fp32);
            using var kvQ4 = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, 64, precision: KvCachePrecision.Q4_0);

            Assert.True(kvQ4.TotalBytes * 7.0 <= kvFp32.TotalBytes);

            var logits32 = new float[weights.VocabSize];
            var logitsQ4 = new float[weights.VocabSize];
            int[] prompt = [1, 5, 12, 7];

            for (int i = 0; i < prompt.Length; i++)
            {
                bool last = i == prompt.Length - 1;
                modelFp32.Forward(prompt[i], i, kvFp32, logits32, computeLogits: last);
                modelQ4.Forward(prompt[i], i, kvQ4, logitsQ4, computeLogits: last);
            }

            float cos = CosineSimilarity(logits32, logitsQ4);
            Assert.True(cos > 0.99f, $"Expected cosine similarity > 0.99, got {cos}");
        }
        finally
        {
            TryDeleteFile(tempGguf);
        }
    }

    [Fact]
    public void CudaKernels_Q4_0_StoreAndAttention_MatchReference_WhenCudaAvailable()
    {
        try
        {
            if (!GpuContext.IsSupported)
                return;
        }
        catch (DllNotFoundException)
        {
            return;
        }
        catch (TypeInitializationException)
        {
            return;
        }

        using var gpu = new GpuContext(0);
        gpu.MakeCurrent();

        byte[] cubin = KernelCompiler.GetOrCompileKernels(gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out IntPtr module, cubin), "ModuleLoadData");
        try
        {
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreQ4, module, "kv_cache_store_q4_0"), "GetFn(kv_cache_store_q4_0)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnQ4, module, "attention_gqa_q4_0"), "GetFn(attention_gqa_q4_0)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnStoreBatchQ4, module, "kv_cache_store_batch_q4_0"), "GetFn(kv_cache_store_batch_q4_0)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out IntPtr fnAttnBatchQ4, module, "attention_gqa_batch_q4_0"), "GetFn(attention_gqa_batch_q4_0)");

            void** storeBatchArgs = stackalloc void*[9];
            void** attnBatchArgs = stackalloc void*[12];
            void** storeArgs = stackalloc void*[8];
            void** attnArgs = stackalloc void*[11];

            foreach (int headDim in new[] { 64, 80, 128 })
            {
                int nHeadsQ = 4;
                int nHeadsKv = 2;
                int maxSeq = 16;
                int seqLen = 5;
                int blocksPerHead = (headDim + 31) / 32;
                nuint cacheBytes = (nuint)((long)nHeadsKv * maxSeq * blocksPerHead * 18L);
                float attnScale = 1.0f / MathF.Sqrt(headDim);

                var rng = new Random(42 + headDim);
                var kSeq = new float[seqLen * nHeadsKv * headDim];
                var vSeq = new float[seqLen * nHeadsKv * headDim];
                var qBatch = new float[seqLen * nHeadsQ * headDim];
                for (int i = 0; i < kSeq.Length; i++)
                {
                    kSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                    vSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }
                for (int i = 0; i < qBatch.Length; i++)
                {
                    qBatch[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }

                IntPtr dKCache = gpu.AllocateDevice(cacheBytes);
                IntPtr dVCache = gpu.AllocateDevice(cacheBytes);
                IntPtr dK = gpu.AllocateDevice((nuint)(kSeq.Length * sizeof(float)));
                IntPtr dV = gpu.AllocateDevice((nuint)(vSeq.Length * sizeof(float)));
                IntPtr dQ = gpu.AllocateDevice((nuint)(qBatch.Length * sizeof(float)));
                IntPtr dOutSingle = gpu.AllocateDevice((nuint)(nHeadsQ * headDim * sizeof(float)));
                IntPtr dOutBatch = gpu.AllocateDevice((nuint)(qBatch.Length * sizeof(float)));
                try
                {
                    CuDriver.Check(CuDriver.MemsetD8(dKCache, 0, cacheBytes), "MemsetD8");
                    CuDriver.Check(CuDriver.MemsetD8(dVCache, 0, cacheBytes), "MemsetD8");

                    fixed (float* pK = kSeq)
                    fixed (float* pV = vSeq)
                    fixed (float* pQ = qBatch)
                    {
                        gpu.CopyToDevice(dK, (IntPtr)pK, (nuint)(kSeq.Length * sizeof(float)));
                        gpu.CopyToDevice(dV, (IntPtr)pV, (nuint)(vSeq.Length * sizeof(float)));
                        gpu.CopyToDevice(dQ, (IntPtr)pQ, (nuint)(qBatch.Length * sizeof(float)));
                    }

                    // 1. Batch store
                    int totalBatchBlocks = seqLen * nHeadsKv * blocksPerHead;
                    uint gridStoreBatch = (uint)((totalBatchBlocks + 7) / 8);
                    int startPos = 0;
                    int bs = seqLen;
                    storeBatchArgs[0] = &dKCache;
                    storeBatchArgs[1] = &dVCache;
                    storeBatchArgs[2] = &dK;
                    storeBatchArgs[3] = &dV;
                    storeBatchArgs[4] = &nHeadsKv;
                    storeBatchArgs[5] = &headDim;
                    storeBatchArgs[6] = &maxSeq;
                    storeBatchArgs[7] = &startPos;
                    storeBatchArgs[8] = &bs;
                    CuDriver.Check(CuDriver.LaunchKernel(fnStoreBatchQ4, gridStoreBatch, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeBatchArgs, IntPtr.Zero), "LaunchStoreBatchQ4");

                    // 2. Batch attention
                    uint attnBlock = (uint)Math.Min(256, Math.Max(32, ((headDim + 31) / 32) * 32));
                    IntPtr dScores = IntPtr.Zero;
                    attnBatchArgs[0] = &dQ;
                    attnBatchArgs[1] = &dKCache;
                    attnBatchArgs[2] = &dVCache;
                    attnBatchArgs[3] = &dOutBatch;
                    attnBatchArgs[4] = &dScores;
                    attnBatchArgs[5] = &nHeadsQ;
                    attnBatchArgs[6] = &nHeadsKv;
                    attnBatchArgs[7] = &headDim;
                    attnBatchArgs[8] = &maxSeq;
                    attnBatchArgs[9] = &startPos;
                    attnBatchArgs[10] = &bs;
                    attnBatchArgs[11] = &attnScale;
                    CuDriver.Check(CuDriver.LaunchKernel(fnAttnBatchQ4, (uint)nHeadsQ, (uint)seqLen, 1, attnBlock, 1, 1, 0, IntPtr.Zero, (IntPtr)attnBatchArgs, IntPtr.Zero), "LaunchAttnBatchQ4");

                    // 3. Also verify single-token store + single-token attention at last pos
                    int lastPos = seqLen - 1;
                    IntPtr dKLast = dK + lastPos * nHeadsKv * headDim * sizeof(float);
                    IntPtr dVLast = dV + lastPos * nHeadsKv * headDim * sizeof(float);
                    IntPtr dQLast = dQ + lastPos * nHeadsQ * headDim * sizeof(float);
                    uint gridStoreSingle = (uint)((nHeadsKv * blocksPerHead + 7) / 8);
                    storeArgs[0] = &dKCache;
                    storeArgs[1] = &dVCache;
                    storeArgs[2] = &dKLast;
                    storeArgs[3] = &dVLast;
                    storeArgs[4] = &nHeadsKv;
                    storeArgs[5] = &headDim;
                    storeArgs[6] = &maxSeq;
                    storeArgs[7] = &lastPos;
                    CuDriver.Check(CuDriver.LaunchKernel(fnStoreQ4, gridStoreSingle, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)storeArgs, IntPtr.Zero), "LaunchStoreQ4");

                    attnArgs[0] = &dQLast;
                    attnArgs[1] = &dKCache;
                    attnArgs[2] = &dVCache;
                    attnArgs[3] = &dOutSingle;
                    attnArgs[4] = &dScores;
                    attnArgs[5] = &nHeadsQ;
                    attnArgs[6] = &nHeadsKv;
                    attnArgs[7] = &headDim;
                    attnArgs[8] = &maxSeq;
                    attnArgs[9] = &lastPos;
                    attnArgs[10] = &attnScale;
                    CuDriver.Check(CuDriver.LaunchKernel(fnAttnQ4, (uint)nHeadsQ, 1, 1, attnBlock, 1, 1, 0, IntPtr.Zero, (IntPtr)attnArgs, IntPtr.Zero), "LaunchAttnQ4");

                    gpu.Synchronize();

                    var hostOutBatch = new float[qBatch.Length];
                    var hostOutSingle = new float[nHeadsQ * headDim];
                    fixed (float* pOutB = hostOutBatch)
                    fixed (float* pOutS = hostOutSingle)
                    {
                        gpu.CopyToHost((IntPtr)pOutB, dOutBatch, (nuint)(hostOutBatch.Length * sizeof(float)));
                        gpu.CopyToHost((IntPtr)pOutS, dOutSingle, (nuint)(hostOutSingle.Length * sizeof(float)));
                    }

                    var refOut = ComputeCpuReferenceAttention(qBatch, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, seqLen, attnScale);
                    Assert.True(CosineSimilarity(refOut, hostOutBatch) > 0.985f);

                    var lastRef = refOut.AsSpan(lastPos * nHeadsQ * headDim, nHeadsQ * headDim).ToArray();
                    Assert.True(CosineSimilarity(lastRef, hostOutSingle) > 0.985f);
                }
                finally
                {
                    gpu.FreeDevice(dKCache);
                    gpu.FreeDevice(dVCache);
                    gpu.FreeDevice(dK);
                    gpu.FreeDevice(dV);
                    gpu.FreeDevice(dQ);
                    gpu.FreeDevice(dOutSingle);
                    gpu.FreeDevice(dOutBatch);
                }
            }
        }
        finally
        {
            CuDriver.ModuleUnload(module);
        }
    }

    [Fact]
    public void CudaQwen2GpuModel_Q4_0_MemoryReduction_And_ForwardAccuracy()
    {
        if (!GpuContext.IsSupported)
            return;

        string tempGguf = CreateSyntheticQwen2Gguf(dim: 128, ffnDim: 128, nHeads: 2, nHeadsKv: 1, layers: 2, vocab: 64, useQ8Weights: true);
        try
        {
            using var gpu = new GpuContext(0);
            using var gguf = GgufFile.Open(tempGguf);
            using var weights = new ModelWeights(gguf);

            using var mFp32 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Fp32);
            using var mFp16 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Fp16);
            using var mFp8 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Fp8);
            using var mQ4 = new Qwen2GpuModel(gpu, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Q4_0);

            Assert.Equal(KvCachePrecision.Q4_0, mQ4.KvPrecision);
            Assert.True(mQ4.TotalKvCacheBytes * 1.75 <= mFp8.TotalKvCacheBytes);
            Assert.True(mQ4.TotalKvCacheBytes * 3.5 <= mFp16.TotalKvCacheBytes);
            Assert.True(mQ4.TotalKvCacheBytes * 7.0 <= mFp32.TotalKvCacheBytes);

            var logits32 = new float[weights.VocabSize];
            var logitsQ4 = new float[weights.VocabSize];
            int[] tokens = [1, 3, 7, 11];

            mFp32.ForwardBatch(tokens, 0, logits32);
            mQ4.ForwardBatch(tokens, 0, logitsQ4);
            Assert.True(CosineSimilarity(logits32, logitsQ4) > 0.99f);

            mFp32.Forward(19, tokens.Length, logits: logits32);
            mQ4.Forward(19, tokens.Length, logits: logitsQ4);
            Assert.True(CosineSimilarity(logits32, logitsQ4) > 0.99f);
        }
        finally
        {
            TryDeleteFile(tempGguf);
        }
    }

    [Fact]
    public void D3D12Shaders_CompileAndExecute_Q4_0_Attention_WhenD3D12Supported()
    {
        if (!D3D12Context.IsSupported)
            return;

        using var ctx = new D3D12Context(-1);

        // Verify all new HLSL shaders compile cleanly under cs_5_0
        Assert.False(ctx.CompileShader(D3D12Shaders.KvCacheStoreQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.KvCacheStoreBatchQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.AttentionGqaQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.AttentionBatchQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Qwen3AttentionF16).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Qwen3AttentionQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Qwen3SparseAttentionF16).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Qwen3SparseAttentionQ4_0).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Gemma4SwaAttentionF16).IsEmpty);
        Assert.False(ctx.CompileShader(D3D12Shaders.Gemma4SwaAttentionQ4_0).IsEmpty);

        uint* sbC = stackalloc uint[5];
        uint* abC = stackalloc uint[7];
        uint* ssC = stackalloc uint[4];
        uint* asC = stackalloc uint[6];

        // Execute real D3D12 Q4_0 single-token & batch store + attention
        foreach (int headDim in new[] { 64, 80, 128, 256 })
        {
            const int nHeadsQ = 4;
            const int nHeadsKv = 2;
            const int maxSeq = 16;
            const int seqLen = 5;
            float attnScale = 1.0f / MathF.Sqrt(headDim);

            ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeadsKv, headDim, maxSeq, KvCachePrecision.Q4_0);
            uint nb = (uint)((headDim + 31) / 32);
            uint numPairs = (nb + 1) / 2;

            var rng = new Random(700 + headDim);
            var kSeq = new float[seqLen * nHeadsKv * headDim];
            var vSeq = new float[seqLen * nHeadsKv * headDim];
            var qBatch = new float[seqLen * nHeadsQ * headDim];
            for (int i = 0; i < kSeq.Length; i++)
            {
                kSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                vSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            for (int i = 0; i < qBatch.Length; i++)
            {
                qBatch[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            using var dKCache = ctx.CreateDeviceBuffer(kvBytes);
            using var dVCache = ctx.CreateDeviceBuffer(kvBytes);
            using var dK = ctx.CreateDeviceBuffer((ulong)(kSeq.Length * sizeof(float)));
            using var dV = ctx.CreateDeviceBuffer((ulong)(vSeq.Length * sizeof(float)));
            using var dQ = ctx.CreateDeviceBuffer((ulong)(qBatch.Length * sizeof(float)));
            using var dOutBatch = ctx.CreateDeviceBuffer((ulong)(qBatch.Length * sizeof(float)));
            using var dOutSingle = ctx.CreateDeviceBuffer((ulong)(nHeadsQ * headDim * sizeof(float)));

            ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kSeq.AsSpan()));
            ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vSeq.AsSpan()));
            ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qBatch.AsSpan()));

            var storeBatchParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 5), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigStoreBatch = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeBatchParams));
            using var psoStoreBatch = ctx.CreatePipelineState(sigStoreBatch, ctx.CompileShader(D3D12Shaders.KvCacheStoreBatchQ4_0));

            var attnBatchParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 7), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigAttnBatch = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnBatchParams));
            using var psoAttnBatch = ctx.CreatePipelineState(sigAttnBatch, ctx.CompileShader(D3D12Shaders.AttentionBatchQ4_0));

            var storeSingleParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigStoreSingle = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeSingleParams));
            using var psoStoreSingle = ctx.CreatePipelineState(sigStoreSingle, ctx.CompileShader(D3D12Shaders.KvCacheStoreQ4_0));

            var attnSingleParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            using var sigAttnSingle = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnSingleParams));
            using var psoAttnSingle = ctx.CreatePipelineState(sigAttnSingle, ctx.CompileShader(D3D12Shaders.AttentionGqaQ4_0));

            ctx.BeginCommands();
            var cmd = ctx.CommandList;

            sbC[0] = (uint)nHeadsKv;
            sbC[1] = (uint)headDim;
            sbC[2] = (uint)maxSeq;
            sbC[3] = 0u;
            sbC[4] = (uint)seqLen;
            cmd.SetComputeRootSignature(sigStoreBatch);
            cmd.SetPipelineState(psoStoreBatch);
            cmd.SetComputeRoot32BitConstants(0, 5, (IntPtr)sbC, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeadsKv * numPairs, (uint)seqLen, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            abC[0] = (uint)nHeadsQ;
            abC[1] = (uint)nHeadsKv;
            abC[2] = (uint)headDim;
            abC[3] = (uint)maxSeq;
            abC[4] = 0u;
            *(float*)&abC[5] = attnScale;
            abC[6] = (uint)seqLen;
            cmd.SetComputeRootSignature(sigAttnBatch);
            cmd.SetPipelineState(psoAttnBatch);
            cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)abC, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dOutBatch.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeadsQ, (uint)seqLen, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            uint lastPos = (uint)(seqLen - 1);
            ulong kvOff = lastPos * (ulong)(nHeadsKv * headDim * sizeof(float));
            ulong qOff = lastPos * (ulong)(nHeadsQ * headDim * sizeof(float));

            ssC[0] = (uint)nHeadsKv;
            ssC[1] = (uint)headDim;
            ssC[2] = (uint)maxSeq;
            ssC[3] = lastPos;
            cmd.SetComputeRootSignature(sigStoreSingle);
            cmd.SetPipelineState(psoStoreSingle);
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)ssC, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress + kvOff);
            cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress + kvOff);
            cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeadsKv * numPairs, 1, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            asC[0] = (uint)nHeadsQ;
            asC[1] = (uint)nHeadsKv;
            asC[2] = (uint)headDim;
            asC[3] = (uint)maxSeq;
            asC[4] = lastPos;
            *(float*)&asC[5] = attnScale;
            cmd.SetComputeRootSignature(sigAttnSingle);
            cmd.SetPipelineState(psoAttnSingle);
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)asC, 0);
            cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress + qOff);
            cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, dOutSingle.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeadsQ, 1, 1);

            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            var hostOutBatch = new float[qBatch.Length];
            var hostOutSingle = new float[nHeadsQ * headDim];
            ctx.CopyToHost(MemoryMarshal.AsBytes(hostOutBatch.AsSpan()), dOutBatch, (ulong)(hostOutBatch.Length * sizeof(float)));
            ctx.CopyToHost(MemoryMarshal.AsBytes(hostOutSingle.AsSpan()), dOutSingle, (ulong)(hostOutSingle.Length * sizeof(float)));

            var refOut = ComputeCpuReferenceAttention(qBatch, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, seqLen, attnScale);
            Assert.True(CosineSimilarity(refOut, hostOutBatch) > 0.985f);

            var lastRef = refOut.AsSpan((int)lastPos * nHeadsQ * headDim, nHeadsQ * headDim).ToArray();
            Assert.True(CosineSimilarity(lastRef, hostOutSingle) > 0.985f);
        }
    }

    [Fact]
    public void D3D12Qwen2Model_Q4_0_MemoryReduction_And_ForwardAccuracy()
    {
        if (!D3D12Context.IsSupported)
            return;

        string tempGguf = CreateSyntheticQwen2Gguf(dim: 128, ffnDim: 128, nHeads: 2, nHeadsKv: 1, layers: 2, vocab: 64, useQ8Weights: true);
        try
        {
            using var gguf = GgufFile.Open(tempGguf);
            using var weights = new ModelWeights(gguf);

            ulong fp32Bytes;
            ulong fp16Bytes;
            ulong q4Bytes;
            var logits32 = new float[weights.VocabSize];
            var logitsQ4 = new float[weights.VocabSize];
            var logits32Single = new float[weights.VocabSize];
            var logitsQ4Single = new float[weights.VocabSize];
            int[] tokens = [1, 4, 9, 15];

            using (var ctxFp16 = new D3D12Context(-1))
            using (var mFp16 = new Qwen2D3D12Model(ctxFp16, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Fp16))
            {
                fp16Bytes = (ulong)mFp16.TotalKvCacheBytes;
            }

            using (var ctxFp32 = new D3D12Context(-1))
            using (var mFp32 = new Qwen2D3D12Model(ctxFp32, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Fp32))
            {
                fp32Bytes = (ulong)mFp32.TotalKvCacheBytes;
                mFp32.ForwardBatch(tokens, 0, logits32);
                mFp32.Forward(21, tokens.Length, logits32Single);
            }

            using (var ctxQ4 = new D3D12Context(-1))
            using (var mQ4 = new Qwen2D3D12Model(ctxQ4, weights, maxSeqLen: 128, kvPrecision: KvCachePrecision.Q4_0))
            {
                Assert.Equal(KvCachePrecision.Q4_0, mQ4.KvPrecision);
                q4Bytes = (ulong)mQ4.TotalKvCacheBytes;
                mQ4.ForwardBatch(tokens, 0, logitsQ4);
                mQ4.Forward(21, tokens.Length, logitsQ4Single);
            }

            Assert.True(q4Bytes * 3.5 <= fp16Bytes, $"Q4_0 ({q4Bytes}) vs Fp16 ({fp16Bytes})");
            Assert.True(q4Bytes * 7.0 <= fp32Bytes, $"Q4_0 ({q4Bytes}) vs Fp32 ({fp32Bytes})");

            float cosBatch = CosineSimilarity(logits32, logitsQ4);
            Assert.True(cosBatch > 0.98f, $"D3D12 batch cosine similarity too low: {cosBatch}");

            float cosSingle = CosineSimilarity(logits32Single, logitsQ4Single);
            Assert.True(cosSingle > 0.98f, $"D3D12 single-token cosine similarity too low: {cosSingle}");
        }
        finally
        {
            TryDeleteFile(tempGguf);
        }
    }

    [Fact]
    public void D3D12Shaders_Execute_Qwen3_Sparse_And_Gemma4Swa_Q4_0_WhenD3D12Supported()
    {
        if (!D3D12Context.IsSupported)
            return;

        using var ctx = new D3D12Context(-1);

        const int nHeadsQ = 4;
        const int nHeadsKv = 2;
        const int headDim = 256;
        const int maxSeq = 8;
        const int seqLen = 5;
        float attnScale = 1.0f / MathF.Sqrt(headDim);

        ulong kvBytes = Qwen2D3D12Model.ComputeD3D12KvLayerBytes(nHeadsKv, headDim, maxSeq, KvCachePrecision.Q4_0);
        uint nb = (uint)((headDim + 31) / 32);
        uint numPairs = (nb + 1) / 2;

        var rng = new Random(909);
        var kSeq = new float[seqLen * nHeadsKv * headDim];
        var vSeq = new float[seqLen * nHeadsKv * headDim];
        var qVec = new float[nHeadsQ * headDim];
        for (int i = 0; i < kSeq.Length; i++)
        {
            kSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vSeq[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < qVec.Length; i++)
        {
            qVec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        using var dKCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dVCache = ctx.CreateDeviceBuffer(kvBytes);
        using var dK = ctx.CreateDeviceBuffer((ulong)(kSeq.Length * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(vSeq.Length * sizeof(float)));
        using var dQ = ctx.CreateDeviceBuffer((ulong)(qVec.Length * sizeof(float)));
        using var dOutQ3 = ctx.CreateDeviceBuffer((ulong)(qVec.Length * sizeof(float)));
        using var dOutSparse = ctx.CreateDeviceBuffer((ulong)(qVec.Length * sizeof(float)));
        using var dOutGemma = ctx.CreateDeviceBuffer((ulong)(qVec.Length * sizeof(float)));

        uint[] selectedIds = [0u, 2u, 4u];
        using var dSelected = ctx.CreateDeviceBuffer((ulong)(selectedIds.Length * sizeof(uint)));

        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes(kSeq.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes(vSeq.AsSpan()));
        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes(qVec.AsSpan()));
        ctx.CopyToDevice(dSelected, MemoryMarshal.AsBytes(selectedIds.AsSpan()));

        var storeBatchParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 5), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigStore = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, storeBatchParams));
        using var psoStore = ctx.CreatePipelineState(sigStore, ctx.CompileShader(D3D12Shaders.KvCacheStoreBatchQ4_0));

        var attnParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigQ3 = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        using var psoQ3 = ctx.CreatePipelineState(sigQ3, ctx.CompileShader(D3D12Shaders.Qwen3AttentionQ4_0));

        var sparseParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigSparse = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sparseParams));
        using var psoSparse = ctx.CreatePipelineState(sigSparse, ctx.CompileShader(D3D12Shaders.Qwen3SparseAttentionQ4_0));

        var gemmaParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        using var sigGemma = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemmaParams));
        using var psoGemma = ctx.CreatePipelineState(sigGemma, ctx.CompileShader(D3D12Shaders.Gemma4SwaAttentionQ4_0));

        uint* sbC = stackalloc uint[5];
        uint* q3C = stackalloc uint[6];
        uint* spC = stackalloc uint[6];
        uint* gmC = stackalloc uint[7];

        ctx.BeginCommands();
        var cmd = ctx.CommandList;

        sbC[0] = (uint)nHeadsKv; sbC[1] = (uint)headDim; sbC[2] = (uint)maxSeq; sbC[3] = 0u; sbC[4] = (uint)seqLen;
        cmd.SetComputeRootSignature(sigStore);
        cmd.SetPipelineState(psoStore);
        cmd.SetComputeRoot32BitConstants(0, 5, (IntPtr)sbC, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
        cmd.Dispatch((uint)nHeadsKv * numPairs, (uint)seqLen, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        q3C[0] = (uint)nHeadsQ; q3C[1] = (uint)nHeadsKv; q3C[2] = (uint)headDim; q3C[3] = (uint)maxSeq; q3C[4] = (uint)(seqLen - 1);
        *(float*)&q3C[5] = attnScale;
        cmd.SetComputeRootSignature(sigQ3);
        cmd.SetPipelineState(psoQ3);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)q3C, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOutQ3.GPUVirtualAddress);
        cmd.Dispatch((uint)nHeadsQ, 1, 1);

        spC[0] = (uint)nHeadsQ; spC[1] = (uint)nHeadsKv; spC[2] = (uint)headDim; spC[3] = (uint)maxSeq; spC[4] = (uint)selectedIds.Length;
        *(float*)&spC[5] = attnScale;
        cmd.SetComputeRootSignature(sigSparse);
        cmd.SetPipelineState(psoSparse);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)spC, 0);
        cmd.SetComputeRootShaderResourceView(1, dSelected.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, dOutSparse.GPUVirtualAddress);
        cmd.Dispatch((uint)nHeadsQ, 1, 1);

        gmC[0] = (uint)nHeadsQ; gmC[1] = (uint)nHeadsKv; gmC[2] = (uint)headDim; gmC[3] = (uint)maxSeq; gmC[4] = (uint)(seqLen - 1);
        *(float*)&gmC[5] = attnScale;
        gmC[6] = 3u; // sliding_window = 3 -> positions [2, 3, 4]
        cmd.SetComputeRootSignature(sigGemma);
        cmd.SetPipelineState(psoGemma);
        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)gmC, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dKCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dVCache.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOutGemma.GPUVirtualAddress);
        cmd.Dispatch((uint)nHeadsQ, 1, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        var outQ3 = new float[qVec.Length];
        var outSparse = new float[qVec.Length];
        var outGemma = new float[qVec.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes(outQ3.AsSpan()), dOutQ3, (ulong)(outQ3.Length * sizeof(float)));
        ctx.CopyToHost(MemoryMarshal.AsBytes(outSparse.AsSpan()), dOutSparse, (ulong)(outSparse.Length * sizeof(float)));
        ctx.CopyToHost(MemoryMarshal.AsBytes(outGemma.AsSpan()), dOutGemma, (ulong)(outGemma.Length * sizeof(float)));

        var refFull = ComputeCpuReferenceAttentionSubset(qVec, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, [0, 1, 2, 3, 4], attnScale);
        var refSparse = ComputeCpuReferenceAttentionSubset(qVec, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, [0, 2, 4], attnScale);
        var refSwa = ComputeCpuReferenceAttentionSubset(qVec, kSeq, vSeq, nHeadsQ, nHeadsKv, headDim, [2, 3, 4], attnScale);

        Assert.True(CosineSimilarity(refFull, outQ3) > 0.985f);
        Assert.True(CosineSimilarity(refSparse, outSparse) > 0.985f);
        Assert.True(CosineSimilarity(refSwa, outGemma) > 0.985f);
    }

    private static float[] ComputeCpuReferenceAttentionSubset(
        float[] qVec,
        float[] kSeq,
        float[] vSeq,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        int[] positions,
        float attnScale)
    {
        var output = new float[nHeadsQ * headDim];
        int groupSize = nHeadsQ / nHeadsKv;
        var scores = new float[positions.Length];

        for (int h = 0; h < nHeadsQ; h++)
        {
            int hKv = h / groupSize;
            int qBase = h * headDim;

            float maxS = float.NegativeInfinity;
            for (int i = 0; i < positions.Length; i++)
            {
                int kBase = (positions[i] * nHeadsKv + hKv) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += qVec[qBase + d] * kSeq[kBase + d];
                scores[i] = dot * attnScale;
                if (scores[i] > maxS) maxS = scores[i];
            }

            float sumExp = 0f;
            for (int i = 0; i < positions.Length; i++)
            {
                scores[i] = MathF.Exp(scores[i] - maxS);
                sumExp += scores[i];
            }

            for (int i = 0; i < positions.Length; i++)
            {
                float w = scores[i] / sumExp;
                int vBase = (positions[i] * nHeadsKv + hKv) * headDim;
                for (int d = 0; d < headDim; d++) output[qBase + d] += w * vSeq[vBase + d];
            }
        }
        return output;
    }

    private static float[] ComputeCpuReferenceAttention(
        float[] qBatch,
        float[] kSeq,
        float[] vSeq,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        int seqLen,
        float attnScale)
    {
        var output = new float[seqLen * nHeadsQ * headDim];
        int groupSize = nHeadsQ / nHeadsKv;
        var scores = new float[seqLen];

        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int h = 0; h < nHeadsQ; h++)
            {
                int hKv = h / groupSize;
                int qBase = (pos * nHeadsQ + h) * headDim;

                float maxS = float.NegativeInfinity;
                for (int t = 0; t <= pos; t++)
                {
                    int kBase = (t * nHeadsKv + hKv) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += qBatch[qBase + d] * kSeq[kBase + d];
                    }
                    scores[t] = dot * attnScale;
                    if (scores[t] > maxS) maxS = scores[t];
                }

                float sumExp = 0f;
                for (int t = 0; t <= pos; t++)
                {
                    scores[t] = MathF.Exp(scores[t] - maxS);
                    sumExp += scores[t];
                }

                int outBase = (pos * nHeadsQ + h) * headDim;
                for (int t = 0; t <= pos; t++)
                {
                    float w = scores[t] / sumExp;
                    int vBase = (t * nHeadsKv + hKv) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        output[outBase + d] += w * vSeq[vBase + d];
                    }
                }
            }
        }
        return output;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom > 0 ? (float)(dot / denom) : 0f;
    }

    internal static string CreateSyntheticQwen2Gguf(
        int dim,
        int ffnDim,
        int nHeads,
        int nHeadsKv,
        int layers,
        int vocab,
        int expertCount = 0,
        int expertUsed = 0,
        int expertFfnDim = 0,
        bool useQ8Weights = false)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_m1_test_{Guid.NewGuid():N}.gguf");
        var rng = new Random(2026);
        int headDim = dim / nHeads;
        int kvDim = nHeadsKv * headDim;

        var meta = new List<(string Key, uint Type, byte[] Data)>
        {
            ("general.architecture", 8u, EncodeString("qwen2")),
            ("qwen2.block_count", 4u, BitConverter.GetBytes((uint)layers)),
            ("qwen2.context_length", 4u, BitConverter.GetBytes(256u)),
            ("qwen2.embedding_length", 4u, BitConverter.GetBytes((uint)dim)),
            ("qwen2.feed_forward_length", 4u, BitConverter.GetBytes((uint)ffnDim)),
            ("qwen2.attention.head_count", 4u, BitConverter.GetBytes((uint)nHeads)),
            ("qwen2.attention.head_count_kv", 4u, BitConverter.GetBytes((uint)nHeadsKv)),
            ("qwen2.rope.freq_base", 6u, BitConverter.GetBytes(10000.0f)),
            ("qwen2.attention.layer_norm_rms_epsilon", 6u, BitConverter.GetBytes(1e-5f)),
            ("tokenizer.ggml.tokens", 9u, EncodeStringArray(vocab))
        };

        if (expertCount > 0)
        {
            meta.Add(("qwen2.expert_count", 4u, BitConverter.GetBytes((uint)expertCount)));
            meta.Add(("qwen2.expert_used_count", 4u, BitConverter.GetBytes((uint)expertUsed)));
            meta.Add(("qwen2.expert_feed_forward_length", 4u, BitConverter.GetBytes((uint)expertFfnDim)));
        }

        var tensors = new List<(string Name, GgufType Type, ulong[] Dims, byte[] Bytes)>();
        void AddF32(string name, params ulong[] dims)
        {
            ulong count = 1;
            foreach (ulong d in dims) count *= d;
            var raw = new byte[count * 4];
            var fSpan = MemoryMarshal.Cast<byte, float>(raw.AsSpan());
            bool isNorm = name.Contains("norm");
            for (int i = 0; i < fSpan.Length; i++)
            {
                fSpan[i] = isNorm ? 1.0f : (float)((rng.NextDouble() * 0.2) - 0.1);
            }
            tensors.Add((name, GgufType.F32, dims, raw));
        }

        void AddMat(string name, int cols, int rows, int third = 1)
        {
            if (!useQ8Weights || (cols % 32) != 0)
            {
                if (third > 1) AddF32(name, (ulong)cols, (ulong)rows, (ulong)third);
                else AddF32(name, (ulong)cols, (ulong)rows);
                return;
            }

            int totalRows = rows * third;
            int nb = cols / 32;
            var raw = new byte[totalRows * nb * 34];
            for (int r = 0; r < totalRows; r++)
            {
                for (int b = 0; b < nb; b++)
                {
                    int blkOff = (r * nb + b) * 34;
                    float scale = 0.01f;
                    Half hScale = (Half)scale;
                    MemoryMarshal.Write(raw.AsSpan(blkOff, 2), in hScale);
                    for (int j = 0; j < 32; j++)
                    {
                        raw[blkOff + 2 + j] = (byte)(sbyte)rng.Next(-12, 13);
                    }
                }
            }
            ulong[] dims = third > 1 ? [(ulong)cols, (ulong)rows, (ulong)third] : [(ulong)cols, (ulong)rows];
            tensors.Add((name, GgufType.Q8_0, dims, raw));
        }

        AddF32("token_embd.weight", (ulong)dim, (ulong)vocab);
        AddF32("output_norm.weight", (ulong)dim);
        AddMat("output.weight", dim, vocab);

        for (int l = 0; l < layers; l++)
        {
            AddF32($"blk.{l}.attn_norm.weight", (ulong)dim);
            AddMat($"blk.{l}.attn_q.weight", dim, nHeads * headDim);
            AddMat($"blk.{l}.attn_k.weight", dim, kvDim);
            AddMat($"blk.{l}.attn_v.weight", dim, kvDim);
            AddMat($"blk.{l}.attn_output.weight", nHeads * headDim, dim);
            AddF32($"blk.{l}.ffn_norm.weight", (ulong)dim);
            AddMat($"blk.{l}.ffn_gate.weight", dim, ffnDim);
            AddMat($"blk.{l}.ffn_up.weight", dim, ffnDim);
            AddMat($"blk.{l}.ffn_down.weight", ffnDim, dim);

            if (expertCount > 0)
            {
                AddF32($"blk.{l}.ffn_gate_inp.weight", (ulong)dim, (ulong)expertCount);
                AddF32($"blk.{l}.ffn_gate_exps.weight", (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount);
                AddF32($"blk.{l}.ffn_up_exps.weight", (ulong)dim, (ulong)expertFfnDim, (ulong)expertCount);
                AddF32($"blk.{l}.ffn_down_exps.weight", (ulong)expertFfnDim, (ulong)dim, (ulong)expertCount);
            }
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // Version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)meta.Count);

        foreach (var (key, type, data) in meta)
        {
            WriteGgufString(bw, key);
            bw.Write(type);
            bw.Write(data);
        }

        ulong offset = 0;
        foreach (var (name, tType, dims, bytes) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims) bw.Write(d);
            bw.Write((uint)tType);
            bw.Write(offset);
            offset += (ulong)((bytes.Length + 31) & ~31);
        }

        long pos = fs.Position;
        long pad = (32 - (pos % 32)) % 32;
        for (int i = 0; i < pad; i++) bw.Write((byte)0);

        foreach (var (_, _, _, bytes) in tensors)
        {
            bw.Write(bytes);
            int tPad = (32 - (bytes.Length % 32)) % 32;
            for (int i = 0; i < tPad; i++) bw.Write((byte)0);
        }

        return path;
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)utf8.Length);
        bw.Write(utf8);
    }

    private static byte[] EncodeString(string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s);
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write((ulong)utf8.Length);
        bw.Write(utf8);
        return ms.ToArray();
    }

    private static byte[] EncodeStringArray(int count)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        bw.Write(8u); // string element type
        bw.Write((ulong)count);
        for (int i = 0; i < count; i++)
        {
            string tok = i == 0 ? "<unk>" : i == 1 ? "<s>" : i == 2 ? "</s>" : $"t{i}";
            WriteGgufString(bw, tok);
        }
        return ms.ToArray();
    }

    internal static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch { }
    }
}
