using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Xunit;

namespace Glacier.Inference.Tests;

[Collection("SequentialGpu")]
public unsafe class KvCacheQ8AndCudaPagedTests
{
    [Fact]
    public void KvCachePrecision_EnumParsingAndDisplay_MatchContract()
    {
        Assert.Equal(0, (int)KvCachePrecision.Auto);
        Assert.Equal(1, (int)KvCachePrecision.Fp32);
        Assert.Equal(2, (int)KvCachePrecision.Fp16);
        Assert.Equal(3, (int)KvCachePrecision.Fp8);
        Assert.Equal(4, (int)KvCachePrecision.Q4_0);
        Assert.Equal(4, (int)KvCachePrecision.Int4);
        Assert.Equal(5, (int)KvCachePrecision.Q8_0);

        Assert.True(KvCachePrecisionExtensions.TryParse("q8_0", out var p));
        Assert.Equal(KvCachePrecision.Q8_0, p);
        Assert.True(KvCachePrecisionExtensions.TryParse(" Q8 ", out p));
        Assert.Equal(KvCachePrecision.Q8_0, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("int8", out p));
        Assert.Equal(KvCachePrecision.Q8_0, p);

        Assert.True(KvCachePrecisionExtensions.TryParse("fp8", out p));
        Assert.Equal(KvCachePrecision.Fp8, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("fp8_e4m3", out p));
        Assert.Equal(KvCachePrecision.Fp8, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("e4m3", out p));
        Assert.Equal(KvCachePrecision.Fp8, p);

        Assert.True(KvCachePrecisionExtensions.TryParse("q4_0", out p));
        Assert.Equal(KvCachePrecision.Q4_0, p);
        Assert.True(KvCachePrecisionExtensions.TryParse("int4", out p));
        Assert.Equal(KvCachePrecision.Q4_0, p);

        Assert.False(KvCachePrecisionExtensions.TryParse(null, out p));
        Assert.False(KvCachePrecisionExtensions.TryParse("bogus", out p));

        Assert.Equal("Q8_0", KvCachePrecision.Q8_0.ToDisplayName());
        Assert.Equal("Fp8", KvCachePrecision.Fp8.ToDisplayName());
        Assert.Equal("Q4_0", KvCachePrecision.Q4_0.ToDisplayName());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(80)]
    [InlineData(128)]
    public void QuantKernelsQ8_0_RoundTripDotAndAccumulate_MatchScalarReference(int k)
    {
        var rng = new Random(1000 + k);
        var src = new float[k];
        var query = new float[k];
        for (int i = 0; i < k; i++)
        {
            src[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        int numBlocks = (k + 31) / 32;
        var blocks = new BlockQ8_0[numBlocks];
        var dequant = new float[k];
        var accActual = new float[k];
        var accExpected = new float[k];
        for (int i = 0; i < k; i++)
        {
            accActual[i] = 0.5f * i;
            accExpected[i] = 0.5f * i;
        }

        float dotActual;
        fixed (float* pSrc = src, pQuery = query, pDeq = dequant, pAcc = accActual)
        fixed (BlockQ8_0* pBlk = blocks)
        {
            QuantKernels.QuantizeQ8_0(pSrc, pBlk, k);
            QuantKernels.DequantizeQ8_0(pBlk, pDeq, k);
            dotActual = QuantKernels.VecDotQ8_0Any(pBlk, pQuery, k);

            // weight == 0f must be a no-op
            QuantKernels.AccumulateScaledQ8_0(pBlk, 0f, pAcc, k);
            for (int i = 0; i < k; i++) Assert.Equal(accExpected[i], pAcc[i]);

            QuantKernels.AccumulateScaledQ8_0(pBlk, 0.375f, pAcc, k);
        }

        float dotExpected = 0f;
        for (int i = 0; i < k; i++)
        {
            Assert.InRange(MathF.Abs(dequant[i] - src[i]), 0f, 0.03f);
            dotExpected += dequant[i] * query[i];
            accExpected[i] += 0.375f * dequant[i];
        }

        Assert.InRange(MathF.Abs(dotActual - dotExpected), 0f, 1e-4f);
        for (int i = 0; i < k; i++)
        {
            Assert.InRange(MathF.Abs(accActual[i] - accExpected[i]), 0f, 1e-4f);
        }
    }

    [Fact]
    public void QuantKernelsQ8_0_EdgeCases_HandleZeroNaNInfinityAndOverflow()
    {
        const int k = 32;
        var block = new BlockQ8_0();
        var zeros = new float[k];
        var deq = new float[k];

        fixed (float* pz = zeros, pd = deq)
        {
            QuantKernels.QuantizeQ8_0(pz, &block, k);
            Assert.Equal(0f, (float)block.Delta);
            QuantKernels.DequantizeQ8_0(&block, pd, k);
            for (int i = 0; i < k; i++) Assert.Equal(0f, deq[i]);
        }

        var special = new float[k];
        special[0] = float.NaN;
        special[1] = float.PositiveInfinity;
        special[2] = float.NegativeInfinity;
        special[3] = 20_000_000.0f; // > MaxQ8Abs (8319008.0f)
        special[4] = -20_000_000.0f;

        fixed (float* ps = special, pd = deq)
        {
            QuantKernels.QuantizeQ8_0(ps, &block, k);
            Assert.True(float.IsFinite((float)block.Delta));
            QuantKernels.DequantizeQ8_0(&block, pd, k);
            Assert.Equal(0f, deq[0]);
            Assert.True(float.IsFinite(deq[1]) && deq[1] > 0f);
            Assert.True(float.IsFinite(deq[2]) && deq[2] < 0f);
            Assert.True(float.IsFinite(deq[3]) && deq[3] > 0f);
            Assert.True(float.IsFinite(deq[4]) && deq[4] < 0f);
        }
    }

    [Fact]
    public void CpuKvCaches_Q8_0_RoundTripDotAndAccumulate_WorkForKVCacheAndGemma4()
    {
        Assert.Equal(136, KVCache.GetSlotByteSize(128, KvCachePrecision.Q8_0));
        Assert.Equal(68, KVCache.GetSlotByteSize(40, KvCachePrecision.Q8_0));

        const int layers = 2;
        const int headsKv = 2;
        const int headDim = 40; // non-32 multiple to exercise partial tail
        const int vHeadDim = 36;
        using var cache = new KVCache(layers, headsKv, headDim, maxSeqLen: 8, vHeadDim: vHeadDim, precision: KvCachePrecision.Q8_0);
        Assert.Equal(KvCachePrecision.Q8_0, cache.Precision);

        var kSrc = new float[headsKv * headDim];
        var vSrc = new float[headsKv * vHeadDim];
        var query = new float[headDim];
        var kRead = new float[headDim];
        var vRead = new float[vHeadDim];
        var outHead = new float[vHeadDim];

        for (int i = 0; i < kSrc.Length; i++) kSrc[i] = MathF.Sin(i * 0.17f);
        for (int i = 0; i < vSrc.Length; i++) vSrc[i] = MathF.Cos(i * 0.23f);
        for (int i = 0; i < query.Length; i++) query[i] = (i % 5 - 2) * 0.3f;

        cache.Store(1, 3, kSrc, vSrc);
        cache.ReadKey(1, 1, 3, kRead);
        cache.ReadValue(1, 1, 3, vRead);

        float dotActual;
        fixed (float* pq = query, pOut = outHead)
        {
            dotActual = cache.DotKey(1, 1, 3, pq);
            cache.AccumulateValue(1, 1, 3, 0.5f, pOut);
        }

        float dotExpected = 0f;
        for (int d = 0; d < headDim; d++)
        {
            Assert.InRange(MathF.Abs(kRead[d] - kSrc[headDim + d]), 0f, 0.02f);
            dotExpected += kRead[d] * query[d];
        }
        Assert.InRange(MathF.Abs(dotActual - dotExpected), 0f, 1e-4f);

        for (int d = 0; d < vHeadDim; d++)
        {
            Assert.InRange(MathF.Abs(vRead[d] - vSrc[vHeadDim + d]), 0f, 0.02f);
            Assert.InRange(MathF.Abs(outHead[d] - 0.5f * vRead[d]), 0f, 1e-4f);
        }

        cache.Reset();
        Assert.False(cache.IsPositionValid(1, 3));
        Assert.Throws<InvalidOperationException>(() => cache.ReadKey(1, 1, 3, kRead));

        // Gemma4KvCache Q8_0 verification
        using var gemmaCache = new Gemma4KvCache(
            layers: 2,
            maxSeqLen: 4,
            swaPattern: new[] { true, false },
            headsKvPattern: new[] { 2, 1 },
            defaultHeadDim: 40,
            precision: KvCachePrecision.Q8_0,
            headDimsPattern: new[] { 40, 40 });

        var gk = new float[40];
        var gv = new float[40];
        var gq = new float[40];
        var gout = new float[40];
        for (int d = 0; d < 40; d++)
        {
            gk[d] = 0.25f * MathF.Sin(d);
            gv[d] = 0.5f * MathF.Cos(d);
            gq[d] = 0.1f * (d + 1);
        }

        float gDot;
        fixed (float* pgk = gk, pgv = gv, pgq = gq, pgout = gout)
        {
            gemmaCache.StoreHead(0, 1, 2, pgk, pgv);
            gDot = gemmaCache.DotKey(0, 1, 2, pgq);
            gemmaCache.AccumulateValue(0, 1, 2, 0.75f, pgout);
        }

        float expectedGDot = 0f;
        for (int d = 0; d < 40; d++) expectedGDot += gk[d] * gq[d];
        Assert.InRange(MathF.Abs(gDot - expectedGDot), 0f, 0.05f);
        for (int d = 0; d < 40; d++)
        {
            Assert.InRange(MathF.Abs(gout[d] - 0.75f * gv[d]), 0f, 0.01f);
        }
    }

    [Fact]
    public void Qwen2GpuModel_PagedPoolSizingHelpers_ReflectExpectedCompressionRatios()
    {
        Assert.Equal(512, Qwen2GpuModel.GetPagedKvSlotBytes(128, KvCachePrecision.Fp32));
        Assert.Equal(256, Qwen2GpuModel.GetPagedKvSlotBytes(128, KvCachePrecision.Fp16));
        Assert.Equal(256, Qwen2GpuModel.GetPagedKvSlotBytes(128, KvCachePrecision.Auto));
        Assert.Equal(136, Qwen2GpuModel.GetPagedKvSlotBytes(128, KvCachePrecision.Q8_0));
        Assert.Equal(72, Qwen2GpuModel.GetPagedKvSlotBytes(128, KvCachePrecision.Q4_0));

        var (fp16K, fp16V) = Qwen2GpuModel.GetPagedPoolByteSizes(16, 4, 2, 128, 128, 16, KvCachePrecision.Fp16);
        var (q8K, q8V) = Qwen2GpuModel.GetPagedPoolByteSizes(16, 4, 2, 128, 128, 16, KvCachePrecision.Q8_0);
        var (q4K, q4V) = Qwen2GpuModel.GetPagedPoolByteSizes(16, 4, 2, 128, 128, 16, KvCachePrecision.Q4_0);

        Assert.Equal(fp16K, fp16V);
        Assert.Equal(q8K, q8V);
        Assert.Equal(q4K, q4V);
        Assert.InRange((double)fp16K / q8K, 1.88, 1.89); // 256 / 136 = 1.882x vs Fp16 (3.765x vs Fp32)
        Assert.InRange((double)fp16K / q4K, 3.55, 3.56); // 256 / 72 = 3.556x vs Fp16 (7.111x vs Fp32)
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void Qwen2GpuModel_CudaPagedFp16Q8AndQ4_SingleAndBatchedMatchCpuFp32()
    {
        if (!CuDriver.IsAvailable() || !GpuContext.IsSupported)
        {
            return;
        }

        RunCudaPagedFp16Q8AndQ4Test();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RunCudaPagedFp16Q8AndQ4Test()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_cuda_paged_").FullName;
        try
        {
            const int layers = 2;
            const int dim = 256;
            const int ffnDim = 512;
            const int nHeads = 2;
            const int nHeadsKv = 1;
            const int headDim = 128; // 4 blocks/head -> exercises multi-block Q4_0/Q8_0 warps
            const int vocab = 32;
            const int blockSize = 4;
            const int totalBlocks = 4;

            string ggufPath = Path.Combine(tempDir, "tiny_qwen2.gguf");
            WriteTinyQwen2Gguf(ggufPath, layers, dim, ffnDim, nHeads, nHeadsKv, headDim, vocab);

            using var gguf = GgufFile.Open(ggufPath);
            using var weights = new ModelWeights(gguf);
            using var gpu = new GpuContext(0);

            // Contiguous Q8_0 must throw NotSupportedException rather than silently falling back to FP32
            Assert.Throws<NotSupportedException>(() =>
                new Qwen2GpuModel(gpu, weights, maxSeqLen: 32, kvPrecision: KvCachePrecision.Q8_0, allocateContiguousKvCache: true));

            // Paged Auto must resolve to Fp16
            using var model = new Qwen2GpuModel(
                gpu,
                weights,
                maxSeqLen: 64,
                kvPrecision: KvCachePrecision.Auto,
                allocateContiguousKvCache: false);
            Assert.Equal(KvCachePrecision.Fp16, model.KvPrecision);

            // Non-identity physical block table to verify block indirection across block boundaries (pos 0..5)
            int[] blockTable = { 2, 0, 1, 3 };
            IntPtr dBlockTable = gpu.AllocateDevice((nuint)(blockTable.Length * sizeof(int)));
            try
            {
                fixed (int* pBt = blockTable)
                {
                    gpu.CopyToDevice(dBlockTable, (IntPtr)pBt, (nuint)(blockTable.Length * sizeof(int)));
                }

                // CPU FP32 reference across 4 prefill tokens + 2 single-token steps
                using var cpuCache = new KVCache(layers, nHeadsKv, headDim, maxSeqLen: 16, precision: KvCachePrecision.Fp32);
                using var cpuModel = new Qwen2Model(weights, maxSeqLen: 16);
                int[] prompt = { 1, 5, 9, 2 };
                var cpuLogitsAfterPrefill = new float[vocab];
                var cpuLogitsAfterToken4 = new float[vocab];
                var cpuLogitsAfterToken5 = new float[vocab];

                for (int i = 0; i < prompt.Length; i++)
                {
                    bool last = i == prompt.Length - 1;
                    cpuModel.Forward(prompt[i], i, cpuCache, last ? cpuLogitsAfterPrefill : Span<float>.Empty, computeLogits: last);
                }
                cpuModel.Forward(7, 4, cpuCache, cpuLogitsAfterToken4, computeLogits: true);
                cpuModel.Forward(11, 5, cpuCache, cpuLogitsAfterToken5, computeLogits: true);

                // Verify Fp16, Q8_0, and Q4_0 paged GPU paths
                foreach (var precision in new[] { KvCachePrecision.Fp16, KvCachePrecision.Q8_0, KvCachePrecision.Q4_0 })
                {
                    var (kBytes, vBytes) = Qwen2GpuModel.GetPagedPoolByteSizes(
                        totalBlocks, layers, nHeadsKv, headDim, headDim, blockSize, precision);
                    IntPtr dKPool = gpu.AllocateDevice(kBytes);
                    IntPtr dVPool = gpu.AllocateDevice(vBytes);
                    try
                    {
                        CuDriver.Check(CuDriver.MemsetD8(dKPool, 0, kBytes), "MemsetD8(dKPool)");
                        CuDriver.Check(CuDriver.MemsetD8(dVPool, 0, vBytes), "MemsetD8(dVPool)");

                        var gpuPrefillLogits = new float[vocab];
                        var gpuToken4Logits = new float[vocab];
                        var gpuToken5Logits = new float[vocab];

                        model.ForwardPagedBatch(prompt, 0, dKPool, dVPool, dBlockTable, blockSize, precision, gpuPrefillLogits, computeLogits: true);
                        model.ForwardPagedToken(7, 4, dKPool, dVPool, dBlockTable, blockSize, precision, gpuToken4Logits, computeLogits: true);
                        model.ForwardPagedToken(11, 5, dKPool, dVPool, dBlockTable, blockSize, precision, gpuToken5Logits, computeLogits: true);

                        float minCosine = precision switch
                        {
                            KvCachePrecision.Fp16 => 0.9999f,
                            KvCachePrecision.Q8_0 => 0.9995f,
                            KvCachePrecision.Q4_0 => 0.990f,
                            _ => 0.99f
                        };

                        Assert.True(CosineSimilarity(cpuLogitsAfterPrefill, gpuPrefillLogits) >= minCosine,
                            $"Prefill cosine similarity too low for {precision}: {CosineSimilarity(cpuLogitsAfterPrefill, gpuPrefillLogits)}");
                        Assert.True(CosineSimilarity(cpuLogitsAfterToken4, gpuToken4Logits) >= minCosine,
                            $"Token4 cosine similarity too low for {precision}: {CosineSimilarity(cpuLogitsAfterToken4, gpuToken4Logits)}");
                        Assert.True(CosineSimilarity(cpuLogitsAfterToken5, gpuToken5Logits) >= minCosine,
                            $"Token5 cosine similarity too low for {precision}: {CosineSimilarity(cpuLogitsAfterToken5, gpuToken5Logits)}");
                    }
                    finally
                    {
                        gpu.FreeDevice(dKPool);
                        gpu.FreeDevice(dVPool);
                    }
                }

                // Also verify 8-arg default overloads with a model constructed with KvCachePrecision.Q8_0
                using var q8DefaultModel = new Qwen2GpuModel(
                    gpu,
                    weights,
                    maxSeqLen: 64,
                    kvPrecision: KvCachePrecision.Q8_0,
                    allocateContiguousKvCache: false);
                Assert.Equal(KvCachePrecision.Q8_0, q8DefaultModel.KvPrecision);

                var (q8KBytes, q8VBytes) = Qwen2GpuModel.GetPagedPoolByteSizes(
                    totalBlocks, layers, nHeadsKv, headDim, headDim, blockSize, KvCachePrecision.Q8_0);
                IntPtr dQ8K = gpu.AllocateDevice(q8KBytes);
                IntPtr dQ8V = gpu.AllocateDevice(q8VBytes);
                try
                {
                    var logits8Arg = new float[vocab];
                    q8DefaultModel.ForwardPagedBatch(prompt, 0, dQ8K, dQ8V, dBlockTable, blockSize, Span<float>.Empty, computeLogits: false);
                    q8DefaultModel.ForwardPagedToken(7, 4, dQ8K, dQ8V, dBlockTable, blockSize, logits8Arg, computeLogits: true);
                    Assert.True(CosineSimilarity(cpuLogitsAfterToken4, logits8Arg) >= 0.9995f);
                }
                finally
                {
                    gpu.FreeDevice(dQ8K);
                    gpu.FreeDevice(dQ8V);
                }
            }
            finally
            {
                gpu.FreeDevice(dBlockTable);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
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

    private static void WriteTinyQwen2Gguf(
        string path,
        int layers,
        int dim,
        int ffnDim,
        int nHeads,
        int nHeadsKv,
        int headDim,
        int vocab)
    {
        var kvMeta = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("qwen2.block_count", GgufValueType.Uint32, (uint)layers),
            ("qwen2.context_length", GgufValueType.Uint32, 64u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen2.attention.key_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2.attention.value_length", GgufValueType.Uint32, (uint)headDim),
            ("qwen2.rope.freq_base", GgufValueType.Float32, 10000.0f),
            ("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("tokenizer.ggml.tokens.length", GgufValueType.Uint32, (uint)vocab)
        };

        var rng = new Random(42);
        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>();

        float[] MakeWeights(int count, float scale = 0.05f)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
            {
                arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
            }
            return arr;
        }

        float[] MakeOnes(int count)
        {
            var arr = new float[count];
            Array.Fill(arr, 1.0f);
            return arr;
        }

        tensors.Add(("token_embd.weight", new ulong[] { (ulong)dim, (ulong)vocab }, MakeWeights(dim * vocab, 0.2f)));
        tensors.Add(("output_norm.weight", new ulong[] { (ulong)dim }, MakeOnes(dim)));
        tensors.Add(("output.weight", new ulong[] { (ulong)dim, (ulong)vocab }, MakeWeights(dim * vocab, 0.1f)));

        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        for (int l = 0; l < layers; l++)
        {
            tensors.Add(($"blk.{l}.attn_norm.weight", new ulong[] { (ulong)dim }, MakeOnes(dim)));
            tensors.Add(($"blk.{l}.attn_q.weight", new ulong[] { (ulong)dim, (ulong)qDim }, MakeWeights(dim * qDim)));
            tensors.Add(($"blk.{l}.attn_k.weight", new ulong[] { (ulong)dim, (ulong)kvDim }, MakeWeights(dim * kvDim)));
            tensors.Add(($"blk.{l}.attn_v.weight", new ulong[] { (ulong)dim, (ulong)kvDim }, MakeWeights(dim * kvDim)));
            tensors.Add(($"blk.{l}.attn_output.weight", new ulong[] { (ulong)qDim, (ulong)dim }, MakeWeights(qDim * dim)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", new ulong[] { (ulong)dim }, MakeOnes(dim)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", new ulong[] { (ulong)dim, (ulong)ffnDim }, MakeWeights(dim * ffnDim)));
            tensors.Add(($"blk.{l}.ffn_up.weight", new ulong[] { (ulong)dim, (ulong)ffnDim }, MakeWeights(dim * ffnDim)));
            tensors.Add(($"blk.{l}.ffn_down.weight", new ulong[] { (ulong)ffnDim, (ulong)dim }, MakeWeights(ffnDim * dim)));
        }

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x46554747u); // 'GGUF'
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvMeta.Count);

        static void WriteGgufString(BinaryWriter w, string s)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }

        foreach (var (key, type, value) in kvMeta)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case GgufValueType.String:
                    WriteGgufString(bw, (string)value);
                    break;
                case GgufValueType.Uint32:
                    bw.Write((uint)value);
                    break;
                case GgufValueType.Float32:
                    bw.Write((float)value);
                    break;
                default:
                    throw new NotSupportedException();
            }
        }

        ulong runningOffset = 0;
        const ulong alignment = 32;
        foreach (var (name, dims, data) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims) bw.Write(d);
            bw.Write((uint)GgufType.F32);
            bw.Write(runningOffset);

            ulong byteSize = (ulong)data.Length * sizeof(float);
            ulong padded = ((byteSize + alignment - 1) / alignment) * alignment;
            runningOffset += padded;
        }

        long pos = fs.Position;
        long padHeader = (long)((alignment - ((ulong)pos % alignment)) % alignment);
        for (long i = 0; i < padHeader; i++) bw.Write((byte)0);

        foreach (var (_, _, data) in tensors)
        {
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
            ulong byteSize = (ulong)data.Length * sizeof(float);
            ulong pad = ((alignment - (byteSize % alignment)) % alignment);
            for (ulong i = 0; i < pad; i++) bw.Write((byte)0);
        }
    }
}
