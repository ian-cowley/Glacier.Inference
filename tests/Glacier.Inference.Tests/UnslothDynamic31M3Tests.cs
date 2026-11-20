using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Xunit;

namespace Glacier.Inference.Tests;

[Collection("SequentialGpu")]
public unsafe class UnslothDynamic31M3Tests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                if (File.Exists(path)) File.Delete(path);
            }
            catch
            {
                // Best effort cleanup for temp GGUF test files
            }
        }
    }

    private string CreateTempPath(string name)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_m3_{name}_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public void GgufTypes_And_D3D12TensorAlign_MatchUnslothDynamic31Specifications()
    {
        Assert.Equal(50L, GgufTypes.GetRowBytes(GgufType.IQ1_S, 256));
        Assert.Equal(56L, GgufTypes.GetRowBytes(GgufType.IQ1_M, 256));
        Assert.Equal(66L, GgufTypes.GetRowBytes(GgufType.IQ2_XXS, 256));
        Assert.Equal(74L, GgufTypes.GetRowBytes(GgufType.IQ2_XS, 256));
        Assert.Equal(82L, GgufTypes.GetRowBytes(GgufType.IQ2_S, 256));
        Assert.Equal(98L, GgufTypes.GetRowBytes(GgufType.IQ3_XXS, 256));
        Assert.Equal(110L, GgufTypes.GetRowBytes(GgufType.IQ3_S, 256));
        Assert.Equal(18L, GgufTypes.GetRowBytes(GgufType.IQ4_NL, 32));
        Assert.Equal(136L, GgufTypes.GetRowBytes(GgufType.IQ4_XS, 256));

        static ulong TensorBytes(GgufType t, ulong count) =>
            new GgufTensorInfo { Name = "w", DimensionsCount = 1, Dimensions = new[] { count }, Type = t, Offset = 0 }.GetByteSize();

        Assert.Equal(50UL, TensorBytes(GgufType.IQ1_S, 256));
        Assert.Equal(56UL, TensorBytes(GgufType.IQ1_M, 256));
        Assert.Equal(74UL, TensorBytes(GgufType.IQ2_XS, 256));
        Assert.Equal(82UL, TensorBytes(GgufType.IQ2_S, 256));
        Assert.Equal(98UL, TensorBytes(GgufType.IQ3_XXS, 256));

        Assert.Equal(50, sizeof(BlockIQ1_S));
        Assert.Equal(56, sizeof(BlockIQ1_M));
        Assert.Equal(66, sizeof(BlockIQ2_XXS));
        Assert.Equal(74, sizeof(BlockIQ2_XS));
        Assert.Equal(82, sizeof(BlockIQ2_S));
        Assert.Equal(98, sizeof(BlockIQ3_XXS));

        // Verify 4-byte HLSL padding rules for 2 mod 4 GGUF block strides
        Assert.Equal(76UL, D3D12TensorAlign.GetAlignedTensorBytes(GgufType.IQ2_XS, rows: 1, cols: 256));
        Assert.Equal(84UL, D3D12TensorAlign.GetAlignedTensorBytes(GgufType.IQ2_S, rows: 1, cols: 256));
        Assert.Equal(100UL, D3D12TensorAlign.GetAlignedTensorBytes(GgufType.IQ3_XXS, rows: 1, cols: 256));
        Assert.Equal(52UL, D3D12TensorAlign.GetAlignedTensorBytes(GgufType.IQ1_S, rows: 1, cols: 256));
        Assert.Equal(56UL, D3D12TensorAlign.GetAlignedTensorBytes(GgufType.IQ1_M, rows: 1, cols: 256));
    }

    [Fact]
    public void QuantKernels_Iq2Xs_Iq2S_Iq3Xxs_DequantizeVecDotExtractAndMatMul_MatchReference()
    {
        const int k = 512; // 2 blocks of 256
        const int rows = 3;
        const int batch = 2;

        var rng = new Random(20261120);
        var src = new float[rows * k];
        var act = new float[batch * k];
        for (int i = 0; i < src.Length; i++)
            src[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * (0.4f + 0.6f * ((i % 17) / 16f));
        for (int i = 0; i < act.Length; i++)
            act[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        VerifyQuantType(GgufType.IQ2_XS, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ2_S, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ3_XXS, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ2_XXS, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ3_S, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ4_NL, rows, k, batch, src, act, tolerance: 2e-3f);
        VerifyQuantType(GgufType.IQ4_XS, rows, k, batch, src, act, tolerance: 2e-3f);
    }

    private static void VerifyQuantType(
        GgufType type, int rows, int k, int batch, float[] src, float[] act, float tolerance)
    {
        int rowBytes = checked((int)GgufTypes.GetRowBytes(type, k));
        byte[] rawWeights = new byte[rows * rowBytes];

        fixed (float* pSrc = src)
        fixed (byte* pRaw = rawWeights)
        {
            for (int r = 0; r < rows; r++)
            {
                EncodeQuantizedRow(type, pSrc + r * k, pRaw + r * rowBytes, k, rowSeed: r + 1);
            }
        }

        float[] dequantized = new float[rows * k];
        float[] embedRow = new float[k];
        float[] matVecOut = new float[rows];
        float[] matBatchOut = new float[batch * rows];

        fixed (byte* pRaw = rawWeights)
        fixed (float* pDequant = dequantized)
        fixed (float* pAct = act)
        fixed (float* pEmbed = embedRow)
        fixed (float* pMvOut = matVecOut)
        fixed (float* pBatchOut = matBatchOut)
        {
            for (int r = 0; r < rows; r++)
            {
                byte* rowPtr = pRaw + r * rowBytes;
                DequantizeByType(type, rowPtr, pDequant + r * k, k);

                float refDot = 0f;
                for (int i = 0; i < k; i++)
                {
                    Assert.True(float.IsFinite(dequantized[r * k + i]));
                    refDot += dequantized[r * k + i] * act[i];
                }

                float directDot = VecDotByType(type, rowPtr, pAct, k);
                Assert.InRange(MathF.Abs(directDot - refDot), 0f, tolerance);

                float matVecRow = 0f;
                QuantKernels.MatVecMul(type, rowPtr, pAct, &matVecRow, k, 1);
                Assert.InRange(MathF.Abs(matVecRow - refDot), 0f, tolerance);
            }

            QuantKernels.ExtractEmbedding(type, pRaw, tokenId: 1, pEmbed, k);
            for (int i = 0; i < k; i++)
                Assert.InRange(MathF.Abs(embedRow[i] - dequantized[k + i]), 0f, 1e-5f);

            QuantKernels.MatVecMul(type, pRaw, pAct, pMvOut, k, rows);
            for (int r = 0; r < rows; r++)
            {
                float refDot = 0f;
                for (int i = 0; i < k; i++)
                    refDot += dequantized[r * k + i] * act[i];
                Assert.InRange(MathF.Abs(matVecOut[r] - refDot), 0f, tolerance);
            }

            QuantKernels.MatMulBatch(type, pRaw, pAct, pBatchOut, k, rows, batch);
            for (int b = 0; b < batch; b++)
            {
                for (int r = 0; r < rows; r++)
                {
                    float refDot = 0f;
                    for (int i = 0; i < k; i++)
                        refDot += dequantized[r * k + i] * act[b * k + i];
                    Assert.InRange(MathF.Abs(matBatchOut[b * rows + r] - refDot), 0f, tolerance);
                }
            }
        }
    }

    [Fact]
    public void SyntheticUdIq2Xs_MixedTensorGguf_RunsOnCpuD3D12AndCuda_WithVulkanGuard()
    {
        // UD-IQ2_XS mixed profile combining IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
        // IQ4_NL, IQ4_XS, Q2_K, Q4_0, Q5_0, Q8_0, BF16, F16, and F32 tensors.
        string path = CreateTempPath("ud_iq2_xs");
        var layer0Types = new LayerTensorTypes(
            Q: GgufType.IQ2_XXS,
            K: GgufType.IQ2_XS,
            V: GgufType.IQ2_S,
            Out: GgufType.IQ3_XXS,
            FfnGate: GgufType.IQ3_S,
            FfnUp: GgufType.IQ4_XS,
            FfnDown: GgufType.Q4_0);
        var layer1Types = new LayerTensorTypes(
            Q: GgufType.IQ2_XS,
            K: GgufType.Q2_K,
            V: GgufType.Q5_0,
            Out: GgufType.F16,
            FfnGate: GgufType.IQ2_S,
            FfnUp: GgufType.IQ3_XXS,
            FfnDown: GgufType.Q8_0);

        WriteMixedQwen2Gguf(
            path,
            embedType: GgufType.IQ4_NL,
            outputType: GgufType.BF16,
            new[] { layer0Types, layer1Types },
            seed: 42);

        using (var gguf = GgufFile.Open(path))
        {
            var weights = new ModelWeights(gguf);
            Assert.False(Qwen2VulkanModel.IsSupported(weights, out string reason));
            Assert.False(string.IsNullOrWhiteSpace(reason));
        }

        VerifyCrossBackendParity(path);
    }

    [Fact]
    public void SyntheticUdIq3_And_UdIq4Xs_MixedTensorGgufs_RunOnCpuD3D12AndCuda()
    {
        string udIq3Path = CreateTempPath("ud_iq3");
        WriteMixedQwen2Gguf(
            udIq3Path,
            embedType: GgufType.IQ4_NL,
            outputType: GgufType.F16,
            new[]
            {
                new LayerTensorTypes(
                    Q: GgufType.IQ3_XXS,
                    K: GgufType.IQ3_S,
                    V: GgufType.IQ3_XXS,
                    Out: GgufType.IQ4_NL,
                    FfnGate: GgufType.IQ3_S,
                    FfnUp: GgufType.IQ4_XS,
                    FfnDown: GgufType.Q8_0)
            },
            seed: 108);
        VerifyCrossBackendParity(udIq3Path);

        string udIq4XsPath = CreateTempPath("ud_iq4_xs");
        WriteMixedQwen2Gguf(
            udIq4XsPath,
            embedType: GgufType.Q4_0,
            outputType: GgufType.BF16,
            new[]
            {
                new LayerTensorTypes(
                    Q: GgufType.IQ4_XS,
                    K: GgufType.IQ4_NL,
                    V: GgufType.IQ3_S,
                    Out: GgufType.IQ4_XS,
                    FfnGate: GgufType.IQ4_NL,
                    FfnUp: GgufType.IQ4_XS,
                    FfnDown: GgufType.BF16)
            },
            seed: 215);
        VerifyCrossBackendParity(udIq4XsPath);
    }

    [Fact]
    public void D3D12Context_EnsureExpertGemvPipelines_CompilesAllUd31ShadersAtRuntime()
    {
        if (!D3D12Context.IsSupported)
            return;

        using var ctx = new D3D12Context();
        GgufType[] types =
        {
            GgufType.IQ2_XXS,
            GgufType.IQ2_XS,
            GgufType.IQ2_S,
            GgufType.IQ3_XXS,
            GgufType.IQ3_S,
            GgufType.IQ4_NL,
            GgufType.IQ4_XS
        };

        foreach (GgufType type in types)
        {
            ctx.EnsureExpertGemvPipelines(type);
        }
    }

    [Fact]
    public void QuantizeIQ2_XS_WithMaxGridMag43_PreservesMixedScaleAndHighDynamicRangeCosine()
    {
        const int k = 768;
        var rng = new Random(9988);

        static double EvalRoundTrip(float[] src, out float worstDotErr)
        {
            int len = src.Length;
            float[] dequant = new float[len];
            float[] act = new float[len];
            var actRng = new Random(12345);
            for (int i = 0; i < len; i++)
                act[i] = (float)(actRng.NextDouble() * 2.0 - 1.0);

            byte[] encoded = new byte[GgufTypes.GetRowBytes(GgufType.IQ2_XS, len)];
            float vecDot;
            fixed (float* pSrc = src, pDeq = dequant, pAct = act)
            fixed (byte* pEnc = encoded)
            {
                QuantKernels.QuantizeIQ2_XS(pSrc, (BlockIQ2_XS*)pEnc, len);
                QuantKernels.DequantizeIQ2_XS((BlockIQ2_XS*)pEnc, pDeq, len);
                vecDot = QuantKernels.VecDotIQ2_XS((BlockIQ2_XS*)pEnc, pAct, len);
            }

            double dot = 0, normA = 0, normB = 0, refDot = 0;
            for (int i = 0; i < len; i++)
            {
                Assert.True(float.IsFinite(dequant[i]));
                dot += src[i] * (double)dequant[i];
                normA += src[i] * (double)src[i];
                normB += dequant[i] * (double)dequant[i];
                refDot += dequant[i] * (double)act[i];
            }
            worstDotErr = MathF.Abs(vecDot - (float)refDot);
            return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        }

        // 1. Gaussian N(0, 1)
        float[] gaussian = new float[k];
        for (int i = 0; i < k; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            gaussian[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
        }
        double cosGauss = EvalRoundTrip(gaussian, out float errGauss);
        Assert.True(cosGauss >= 0.94, $"Expected Gaussian IQ2_XS cosine >= 0.94 with maxGridMag=43, got {cosGauss:F4}");
        Assert.InRange(errGauss, 0f, 5e-3f);

        // 2. Mixed sub-block scales across 16-element sub-blocks
        float[] mixed = new float[k];
        for (int i = 0; i < k; i++)
        {
            float subScale = ((i / 16) % 4) switch { 0 => 0.25f, 1 => 1.0f, 2 => 2.5f, _ => 0.5f };
            mixed[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * subScale;
        }
        double cosMixed = EvalRoundTrip(mixed, out float errMixed);
        Assert.True(cosMixed >= 0.94, $"Expected MixedScale IQ2_XS cosine >= 0.94 with maxGridMag=43, got {cosMixed:F4}");
        Assert.InRange(errMixed, 0f, 5e-3f);

        // 3. Realistic IQ2_XS 43/8-ratio high dynamic range spikes
        float[] spikes = new float[k];
        for (int i = 0; i < k; i++)
        {
            float baseVal = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.25f;
            spikes[i] = (i % 19 == 0) ? (baseVal >= 0 ? 1.35f : -1.35f) : baseVal;
        }
        double cosSpikes = EvalRoundTrip(spikes, out float errSpikes);
        Assert.True(cosSpikes >= 0.90, $"Expected 43/8 spike IQ2_XS cosine >= 0.90, got {cosSpikes:F4}");
        Assert.InRange(errSpikes, 0f, 5e-3f);
    }

    [Fact]
    public void Gemma4D3D12Model_AlignedMoeExpertSliceOffsets_MatchUploadAlignedTensor_AndExecuteOnD3D12()
    {
        const int dim = 256;
        const int sharedFfnDim = 256;
        const int expertFfnDim = 256;
        const int expertCount = 6;
        const int expertUsedCount = 4;

        // 1. Verify D3D12TensorAlign.GetAlignedTensorBytes differs from unaligned GetRowBytes
        // for all padded UD 3.1 + F16/BF16 MoE expert tensor types.
        GgufType[] paddedTypes =
        {
            GgufType.IQ2_XXS,
            GgufType.IQ2_XS,
            GgufType.IQ2_S,
            GgufType.IQ3_XXS,
            GgufType.IQ3_S,
            GgufType.IQ4_NL,
            GgufType.Q4_0,
            GgufType.Q5_0,
            GgufType.Q8_0,
            GgufType.Q3_K,
            GgufType.Q6_K,
            GgufType.MXFP4,
            GgufType.F16,
            GgufType.BF16
        };

        foreach (GgufType t in paddedTypes)
        {
            ulong alignedGateUp = D3D12TensorAlign.GetAlignedTensorBytes(t, 2 * expertFfnDim, dim);
            ulong unalignedGateUp = (ulong)(2 * expertFfnDim) * (ulong)GgufTypes.GetRowBytes(t, dim);
            Assert.True(alignedGateUp > unalignedGateUp, $"Expected aligned slice bytes > unaligned for {t} GateUp.");

            ulong alignedDown = D3D12TensorAlign.GetAlignedTensorBytes(t, dim, expertFfnDim);
            ulong unalignedDown = (ulong)dim * (ulong)GgufTypes.GetRowBytes(t, expertFfnDim);
            Assert.True(alignedDown > unalignedDown, $"Expected aligned slice bytes > unaligned for {t} Down.");
        }

        if (!D3D12Context.IsSupported)
            return;

        // 2. Build paired synthetic Gemma4 MoE GGUFs (quantized UD 3.1 experts vs exact F32 dequantized experts,
        // plus an up-half-negated F32 GGUF to verify up [expertFfnDim..2*expertFfnDim) sensitivity)
        // with expScale = 0.35f, attn_q_norm/attn_k_norm, and routing toward experts {0, 1, 3, 5}.
        string quantGemma4Path = CreateTempPath("gemma4_moe_ud31");
        string refGemma4Path = CreateTempPath("gemma4_moe_f32ref");
        string negUpGemma4Path = CreateTempPath("gemma4_moe_f32negup");
        WriteSyntheticGemma4MoePair(
            quantGemma4Path,
            refGemma4Path,
            negUpGemma4Path,
            dim,
            sharedFfnDim,
            expertFfnDim,
            expertCount,
            expertUsedCount,
            gateUpExpsType: GgufType.IQ2_XS,
            downExpsType: GgufType.IQ3_XXS,
            seed: 777);

        // 2a. Verify D3D12 Quantized vs D3D12 F32 Reference parity
        using (var quantGguf = GgufFile.Open(quantGemma4Path))
        using (var quantWeights = new ModelWeights(quantGguf))
        using (var refGguf = GgufFile.Open(refGemma4Path))
        using (var refWeights = new ModelWeights(refGguf))
        using (var ctx = new D3D12Context())
        using (var quantModel = new Gemma4D3D12Model(ctx, quantWeights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16))
        using (var refModel = new Gemma4D3D12Model(ctx, refWeights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16))
        {
            float[] quantLogits = new float[quantWeights.VocabSize];
            float[] refLogits = new float[refWeights.VocabSize];
            quantModel.Forward(token: 1, pos: 0, quantLogits, computeLogits: true);
            refModel.Forward(token: 1, pos: 0, refLogits, computeLogits: true);

            AssertAllFinite(quantLogits);
            AssertAllFinite(refLogits);
            AssertLogitClose(refLogits, quantLogits, minCosine: 0.999f, maxAbsDiff: 0.05f);
        }

        // 2b. Verify up-half sensitivity on D3D12: negating [expertFfnDim .. 2*expertFfnDim) must change logits significantly
        using (var refGguf = GgufFile.Open(refGemma4Path))
        using (var refWeights = new ModelWeights(refGguf))
        using (var negUpGguf = GgufFile.Open(negUpGemma4Path))
        using (var negUpWeights = new ModelWeights(negUpGguf))
        using (var ctx = new D3D12Context())
        using (var refModel = new Gemma4D3D12Model(ctx, refWeights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16))
        using (var negUpModel = new Gemma4D3D12Model(ctx, negUpWeights, maxSeqLen: 64, kvPrecision: KvCachePrecision.Fp16))
        {
            float[] refLogits = new float[refWeights.VocabSize];
            float[] negUpLogits = new float[negUpWeights.VocabSize];
            refModel.Forward(token: 1, pos: 0, refLogits, computeLogits: true);
            negUpModel.Forward(token: 1, pos: 0, negUpLogits, computeLogits: true);

            AssertAllFinite(refLogits);
            AssertAllFinite(negUpLogits);
            float maxUpDiff = 0f;
            for (int i = 0; i < refLogits.Length; i++)
                maxUpDiff = MathF.Max(maxUpDiff, MathF.Abs(refLogits[i] - negUpLogits[i]));
            Assert.True(maxUpDiff > 1.0f, $"Expected negating up-half of ffn_gate_up_exps to change D3D12 logits by > 1.0, got {maxUpDiff:F4}");
        }

        // 2c. Verify DirectML (Gemma4D3D12Model) vs CPU (Gemma4Model) parity >= 0.999 across Fp16, Fp32, and Q4_0
        // on BOTH ForwardToken (positions 0..3) and Prefill (4-token VerifyBatchLogits).
        int[] prompt = { 1, 5, 11, 23 };
        KvCachePrecision[] precisions = { KvCachePrecision.Fp16, KvCachePrecision.Fp32, KvCachePrecision.Q4_0 };
        foreach (KvCachePrecision kvPrec in precisions)
        {
            using var cpu = new InferenceSession(quantGemma4Path, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu, kvPrecision: kvPrec);
            using var d3d = new InferenceSession(quantGemma4Path, maxSeqLen: 64, device: null, engine: InferenceEngineType.DirectML, kvPrecision: kvPrec);

            int vocab = ((ISpeculativeTarget)cpu).VocabSize;
            float[] cpuSingle = new float[vocab];
            float[] d3dSingle = new float[vocab];
            for (int i = 0; i < prompt.Length; i++)
            {
                cpu.ForwardToken(prompt[i], i, computeLogits: true);
                d3d.ForwardToken(prompt[i], i, computeLogits: true);
                cpu.CopyCurrentLogits(cpuSingle);
                d3d.CopyCurrentLogits(d3dSingle);
                AssertAllFinite(cpuSingle);
                AssertAllFinite(d3dSingle);
                float maxDiff = kvPrec == KvCachePrecision.Q4_0 ? 0.35f : 0.05f;
                AssertLogitClose(cpuSingle, d3dSingle, minCosine: 0.999f, maxAbsDiff: maxDiff);
            }

            cpu.ResetKvCache();
            d3d.ResetKvCache();
            float[] cpuBatch = new float[prompt.Length * vocab];
            float[] d3dBatch = new float[prompt.Length * vocab];
            cpu.VerifyBatchLogits(prompt, startPos: 0, cpuBatch);
            d3d.VerifyBatchLogits(prompt, startPos: 0, d3dBatch);
            AssertAllFinite(cpuBatch);
            AssertAllFinite(d3dBatch);
            float batchMaxDiff = kvPrec == KvCachePrecision.Q4_0 ? 0.35f : 0.05f;
            AssertLogitClose(cpuBatch, d3dBatch, minCosine: 0.999f, maxAbsDiff: batchMaxDiff);
        }
    }

    private static void WriteSyntheticGemma4MoePair(
        string quantPath,
        string refPath,
        string negUpPath,
        int dim,
        int sharedFfnDim,
        int expertFfnDim,
        int expertCount,
        int expertUsedCount,
        GgufType gateUpExpsType,
        GgufType downExpsType,
        int seed)
    {
        const int vocab = 32;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        int headDim = dim / nHeads; // 64
        int kvDim = nHeadsKv * headDim; // 128

        var rng = new Random(seed);
        var tokens = new string[vocab];
        tokens[0] = "<unk>";
        tokens[1] = "<s>";
        tokens[2] = "</s>";
        for (int i = 3; i < vocab; i++)
            tokens[i] = i <= 28 ? ((char)('a' + (i - 3))).ToString() : ((char)('0' + (i - 29))).ToString();

        var kvEntries = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "gemma4"),
            ("general.name", GgufValueType.String, "synthetic-gemma4-moe-ud31"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("tokenizer.ggml.model", GgufValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", GgufValueType.Array, tokens),
            ("tokenizer.ggml.merges", GgufValueType.Array, Array.Empty<string>()),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, 1u),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, 2u),
            ("gemma4.context_length", GgufValueType.Uint32, 128u),
            ("gemma4.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("gemma4.feed_forward_length", GgufValueType.Uint32, (uint)sharedFfnDim),
            ("gemma4.expert_feed_forward_length", GgufValueType.Uint32, (uint)expertFfnDim),
            ("gemma4.expert_count", GgufValueType.Uint32, (uint)expertCount),
            ("gemma4.expert_used_count", GgufValueType.Uint32, (uint)expertUsedCount),
            ("gemma4.block_count", GgufValueType.Uint32, 1u),
            ("gemma4.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("gemma4.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("gemma4.attention.key_length", GgufValueType.Uint32, (uint)headDim),
            ("gemma4.attention.sliding_window", GgufValueType.Uint32, 64u),
            ("gemma4.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("gemma4.rope.freq_base", GgufValueType.Float32, 10000.0f),
            ("gemma4.rope.freq_base_swa", GgufValueType.Float32, 10000.0f),
            ("gemma4.final_logit_softcapping", GgufValueType.Float32, 30.0f),
            ("gemma4.attention.logit_softcapping", GgufValueType.Float32, 50.0f)
        };

        var commonTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>
        {
            CreateMatrixTensor("token_embd.weight", dim, vocab, GgufType.F32, rng, scale: 0.3f),
            CreateNormTensor("output_norm.weight", dim),
            CreateMatrixTensor("output.weight", dim, vocab, GgufType.F32, rng, scale: 0.3f),
            CreateNormTensor("blk.0.attn_norm.weight", dim),
            CreateMatrixTensor("blk.0.attn_q.weight", dim, dim, GgufType.F32, rng, scale: 0.2f),
            CreateMatrixTensor("blk.0.attn_k.weight", dim, kvDim, GgufType.F32, rng, scale: 0.2f),
            CreateMatrixTensor("blk.0.attn_v.weight", dim, kvDim, GgufType.F32, rng, scale: 0.2f),
            CreateNormTensor("blk.0.attn_q_norm.weight", headDim),
            CreateNormTensor("blk.0.attn_k_norm.weight", headDim),
            CreateMatrixTensor("blk.0.attn_output.weight", dim, dim, GgufType.F32, rng, scale: 0.2f),
            CreateNormTensor("blk.0.post_attention_norm.weight", dim),
            CreateNormTensor("blk.0.ffn_norm.weight", dim),
            CreateMatrixTensor("blk.0.ffn_gate.weight", dim, sharedFfnDim, GgufType.F32, rng, scale: 0.2f),
            CreateMatrixTensor("blk.0.ffn_up.weight", dim, sharedFfnDim, GgufType.F32, rng, scale: 0.2f),
            CreateMatrixTensor("blk.0.ffn_down.weight", sharedFfnDim, dim, GgufType.F32, rng, scale: 0.2f),
            CreateNormTensor("blk.0.post_ffw_norm_1.weight", dim),
            CreateNormTensor("blk.0.pre_ffw_norm_2.weight", dim),
            CreateNormTensor("blk.0.post_ffw_norm_2.weight", dim),
            CreateNormTensor("blk.0.post_ffw_norm.weight", dim),
            CreateNormTensor("blk.0.ffn_gate_inp.scale", dim)
        };

        // Bias router strongly toward experts {0, 1, 3, 5} to force non-zero slice offsets.
        byte[] routerData = new byte[expertCount * dim * sizeof(float)];
        var routerF = MemoryMarshal.Cast<byte, float>(routerData.AsSpan());
        for (int e = 0; e < expertCount; e++)
        {
            float val = (e == 0 || e == 1 || e == 3 || e == 5) ? 0.5f : -0.5f;
            for (int d = 0; d < dim; d++)
                routerF[e * dim + d] = val * (1.0f + 0.05f * ((d + e) % 7));
        }
        commonTensors.Add(("blk.0.ffn_gate_inp.weight", [(ulong)dim, (ulong)expertCount], GgufType.F32, routerData));

        byte[] expScales = new byte[expertCount * sizeof(float)];
        MemoryMarshal.Cast<byte, float>(expScales.AsSpan()).Fill(1.0f);
        commonTensors.Add(("blk.0.ffn_down_exps.scale", [(ulong)expertCount], GgufType.F32, expScales));

        var gateUpQuant = CreateMatrixTensor("blk.0.ffn_gate_up_exps.weight", dim, expertCount * 2 * expertFfnDim, gateUpExpsType, rng, scale: 0.35f);
        var downQuant = CreateMatrixTensor("blk.0.ffn_down_exps.weight", expertFfnDim, expertCount * dim, downExpsType, rng, scale: 0.35f);

        // Dequantize the exact quantized expert weights into F32 for the reference GGUF
        byte[] gateUpRefBytes = new byte[expertCount * 2 * expertFfnDim * dim * sizeof(float)];
        byte[] downRefBytes = new byte[expertCount * dim * expertFfnDim * sizeof(float)];
        int gateUpRowBytes = checked((int)GgufTypes.GetRowBytes(gateUpExpsType, dim));
        int downRowBytes = checked((int)GgufTypes.GetRowBytes(downExpsType, expertFfnDim));

        fixed (byte* pGuQ = gateUpQuant.Data, pGuR = gateUpRefBytes, pDnQ = downQuant.Data, pDnR = downRefBytes)
        {
            for (int r = 0; r < expertCount * 2 * expertFfnDim; r++)
                DequantizeByType(gateUpExpsType, pGuQ + (long)r * gateUpRowBytes, (float*)pGuR + (long)r * dim, dim);
            for (int r = 0; r < expertCount * dim; r++)
                DequantizeByType(downExpsType, pDnQ + (long)r * downRowBytes, (float*)pDnR + (long)r * expertFfnDim, expertFfnDim);
        }

        byte[] gateUpNegUpBytes = (byte[])gateUpRefBytes.Clone();
        Span<float> negUpFloats = MemoryMarshal.Cast<byte, float>(gateUpNegUpBytes.AsSpan());
        for (int e = 0; e < expertCount; e++)
        {
            int upRowStart = (e * 2 * expertFfnDim + expertFfnDim) * dim;
            int upRowEnd = (e * 2 * expertFfnDim + 2 * expertFfnDim) * dim;
            for (int idx = upRowStart; idx < upRowEnd; idx++)
                negUpFloats[idx] = -negUpFloats[idx];
        }

        var quantTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>(commonTensors)
        {
            ("blk.0.ffn_gate_up_exps.weight", [(ulong)dim, (ulong)(2 * expertFfnDim), (ulong)expertCount], gateUpExpsType, gateUpQuant.Data),
            ("blk.0.ffn_down_exps.weight", [(ulong)expertFfnDim, (ulong)dim, (ulong)expertCount], downExpsType, downQuant.Data)
        };

        var refTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>(commonTensors)
        {
            ("blk.0.ffn_gate_up_exps.weight", [(ulong)dim, (ulong)(2 * expertFfnDim), (ulong)expertCount], GgufType.F32, gateUpRefBytes),
            ("blk.0.ffn_down_exps.weight", [(ulong)expertFfnDim, (ulong)dim, (ulong)expertCount], GgufType.F32, downRefBytes)
        };

        var negUpTensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>(commonTensors)
        {
            ("blk.0.ffn_gate_up_exps.weight", [(ulong)dim, (ulong)(2 * expertFfnDim), (ulong)expertCount], GgufType.F32, gateUpNegUpBytes),
            ("blk.0.ffn_down_exps.weight", [(ulong)expertFfnDim, (ulong)dim, (ulong)expertCount], GgufType.F32, downRefBytes)
        };

        WriteGgufFile(quantPath, kvEntries, quantTensors);
        WriteGgufFile(refPath, kvEntries, refTensors);
        WriteGgufFile(negUpPath, kvEntries, negUpTensors);
    }

    private static void VerifyCrossBackendParity(string ggufPath)
    {
        int[] prompt = { 1, 5, 11, 23 };
        float[] cpuSingleLogits;
        float[] cpuBatchLogits;
        int vocab;

        using (var cpu = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu))
        {
            vocab = ((ISpeculativeTarget)cpu).VocabSize;
            cpuSingleLogits = new float[vocab];
            cpuBatchLogits = new float[prompt.Length * vocab];

            for (int i = 0; i < prompt.Length; i++)
                cpu.ForwardToken(prompt[i], i, computeLogits: i == prompt.Length - 1);
            cpu.CopyCurrentLogits(cpuSingleLogits);

            cpu.ResetKvCache();
            cpu.VerifyBatchLogits(prompt, startPos: 0, cpuBatchLogits);
        }

        AssertAllFinite(cpuSingleLogits);
        AssertAllFinite(cpuBatchLogits);
        float[] cpuLastBatch = cpuBatchLogits.AsSpan((prompt.Length - 1) * vocab, vocab).ToArray();
        AssertLogitClose(cpuSingleLogits, cpuLastBatch, minCosine: 0.9999f, maxAbsDiff: 1e-2f);

        if (D3D12Context.IsSupported)
        {
            using var d3d = new InferenceSession(ggufPath, maxSeqLen: 64, device: null, engine: InferenceEngineType.DirectML);
            float[] d3dSingle = new float[vocab];
            float[] d3dBatch = new float[prompt.Length * vocab];
            for (int i = 0; i < prompt.Length; i++)
                d3d.ForwardToken(prompt[i], i, computeLogits: i == prompt.Length - 1);
            d3d.CopyCurrentLogits(d3dSingle);

            d3d.ResetKvCache();
            d3d.VerifyBatchLogits(prompt, startPos: 0, d3dBatch);
            AssertAllFinite(d3dSingle);
            AssertAllFinite(d3dBatch);
            AssertLogitClose(cpuSingleLogits, d3dSingle, minCosine: 0.995f, maxAbsDiff: 0.15f);
            AssertLogitClose(cpuLastBatch, d3dBatch.AsSpan((prompt.Length - 1) * vocab, vocab).ToArray(), minCosine: 0.995f, maxAbsDiff: 0.15f);
        }

        bool hasNvidiaCuda = DeviceManager.GetDevices().Any(
            d => d.Vendor == GpuVendor.Nvidia && d.SupportedEngines.Contains(InferenceEngineType.BareMetal));
        if (hasNvidiaCuda && GpuContext.IsSupported)
        {
            using var cuda = new InferenceSession(ggufPath, maxSeqLen: 64, device: "nvidia", engine: InferenceEngineType.BareMetal);
            float[] cudaSingle = new float[vocab];
            float[] cudaBatch = new float[prompt.Length * vocab];
            for (int i = 0; i < prompt.Length; i++)
                cuda.ForwardToken(prompt[i], i, computeLogits: i == prompt.Length - 1);
            cuda.CopyCurrentLogits(cudaSingle);

            cuda.ResetKvCache();
            cuda.VerifyBatchLogits(prompt, startPos: 0, cudaBatch);
            AssertAllFinite(cudaSingle);
            AssertAllFinite(cudaBatch);
            AssertLogitClose(cpuSingleLogits, cudaSingle, minCosine: 0.995f, maxAbsDiff: 0.15f);
            AssertLogitClose(cpuLastBatch, cudaBatch.AsSpan((prompt.Length - 1) * vocab, vocab).ToArray(), minCosine: 0.995f, maxAbsDiff: 0.15f);
        }
    }

    private static void AssertAllFinite(float[] values)
    {
        float maxAbs = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            Assert.True(float.IsFinite(values[i]), $"Non-finite value at index {i}: {values[i]}");
            maxAbs = MathF.Max(maxAbs, MathF.Abs(values[i]));
        }
        Assert.True(maxAbs > 1e-4f, $"Expected non-degenerate logits, got maxAbs={maxAbs}");
    }

    private static void AssertLogitClose(float[] expected, float[] actual, float minCosine, float maxAbsDiff)
    {
        Assert.Equal(expected.Length, actual.Length);
        double dot = 0, normA = 0, normB = 0;
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            dot += expected[i] * (double)actual[i];
            normA += expected[i] * (double)expected[i];
            normB += actual[i] * (double)actual[i];
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }

        double cosine = (normA > 0 && normB > 0) ? dot / (Math.Sqrt(normA) * Math.Sqrt(normB)) : 0;
        Assert.True(cosine >= minCosine, $"Cosine similarity {cosine:F6} < {minCosine:F6}, worstDiff={worst:F6}");
        Assert.True(worst <= maxAbsDiff, $"Worst logit diff {worst:F6} > {maxAbsDiff:F6}, cosine={cosine:F6}");
    }

    internal readonly record struct LayerTensorTypes(
        GgufType Q,
        GgufType K,
        GgufType V,
        GgufType Out,
        GgufType FfnGate,
        GgufType FfnUp,
        GgufType FfnDown);

    internal static void WriteMixedQwen2Gguf(
        string path,
        GgufType embedType,
        GgufType outputType,
        LayerTensorTypes[] layers,
        int seed)
    {
        const int dim = 256;
        const int ffnDim = 256;
        const int vocab = 32;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        int headDim = dim / nHeads; // 64
        int kvDim = nHeadsKv * headDim; // 128

        var rng = new Random(seed);
        var tokens = new string[vocab];
        tokens[0] = "<unk>";
        tokens[1] = "<s>";
        tokens[2] = "</s>";
        for (int i = 3; i < vocab; i++)
            tokens[i] = i <= 28 ? ((char)('a' + (i - 3))).ToString() : ((char)('0' + (i - 29))).ToString();

        var kvEntries = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.name", GgufValueType.String, "synthetic-ud31-qwen2"),
            ("general.alignment", GgufValueType.Uint32, 32u),
            ("tokenizer.ggml.model", GgufValueType.String, "gpt2"),
            ("tokenizer.ggml.tokens", GgufValueType.Array, tokens),
            ("tokenizer.ggml.merges", GgufValueType.Array, Array.Empty<string>()),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, 1u),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, 2u),
            ("qwen2.context_length", GgufValueType.Uint32, 256u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.block_count", GgufValueType.Uint32, (uint)layers.Length),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)nHeads),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)nHeadsKv),
            ("qwen2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("qwen2.rope.freq_base", GgufValueType.Float32, 10000.0f)
        };

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();

        tensors.Add(CreateMatrixTensor("token_embd.weight", dim, vocab, embedType, rng, scale: 0.35f));
        tensors.Add(CreateNormTensor("output_norm.weight", dim));
        tensors.Add(CreateMatrixTensor("output.weight", dim, vocab, outputType, rng, scale: 0.35f));

        for (int l = 0; l < layers.Length; l++)
        {
            LayerTensorTypes lt = layers[l];
            string prefix = $"blk.{l}";
            tensors.Add(CreateNormTensor($"{prefix}.attn_norm.weight", dim));
            tensors.Add(CreateMatrixTensor($"{prefix}.attn_q.weight", dim, dim, lt.Q, rng, scale: 0.25f));
            tensors.Add(CreateMatrixTensor($"{prefix}.attn_k.weight", dim, kvDim, lt.K, rng, scale: 0.25f));
            tensors.Add(CreateMatrixTensor($"{prefix}.attn_v.weight", dim, kvDim, lt.V, rng, scale: 0.25f));
            tensors.Add(CreateMatrixTensor($"{prefix}.attn_output.weight", dim, dim, lt.Out, rng, scale: 0.25f));
            tensors.Add(CreateNormTensor($"{prefix}.ffn_norm.weight", dim));
            tensors.Add(CreateMatrixTensor($"{prefix}.ffn_gate.weight", dim, ffnDim, lt.FfnGate, rng, scale: 0.25f));
            tensors.Add(CreateMatrixTensor($"{prefix}.ffn_up.weight", dim, ffnDim, lt.FfnUp, rng, scale: 0.25f));
            tensors.Add(CreateMatrixTensor($"{prefix}.ffn_down.weight", ffnDim, dim, lt.FfnDown, rng, scale: 0.25f));
        }

        WriteGgufFile(path, kvEntries, tensors);
    }

    private static (string Name, ulong[] Dims, GgufType Type, byte[] Data) CreateNormTensor(string name, int dim)
    {
        byte[] data = new byte[dim * sizeof(float)];
        var floats = MemoryMarshal.Cast<byte, float>(data.AsSpan());
        for (int i = 0; i < dim; i++)
            floats[i] = 1.0f + 0.05f * MathF.Sin(i * 0.1f);
        return (name, new ulong[] { (ulong)dim }, GgufType.F32, data);
    }

    private static (string Name, ulong[] Dims, GgufType Type, byte[] Data) CreateMatrixTensor(
        string name, int cols, int rows, GgufType type, Random rng, float scale)
    {
        float[] src = new float[rows * cols];
        for (int i = 0; i < src.Length; i++)
            src[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;

        int rowBytes = checked((int)GgufTypes.GetRowBytes(type, cols));
        byte[] data = new byte[rows * rowBytes];

        fixed (float* pSrc = src)
        fixed (byte* pData = data)
        {
            for (int r = 0; r < rows; r++)
                EncodeQuantizedRow(type, pSrc + r * cols, pData + r * rowBytes, cols, rowSeed: r + rng.Next(1, 1000));
        }

        return (name, new ulong[] { (ulong)cols, (ulong)rows }, type, data);
    }

    private static void EncodeQuantizedRow(GgufType type, float* src, byte* dst, int k, int rowSeed)
    {
        switch (type)
        {
            case GgufType.F32:
                new ReadOnlySpan<float>(src, k).CopyTo(new Span<float>(dst, k));
                break;

            case GgufType.F16:
                QuantKernels.QuantizeF16(src, (Half*)dst, k);
                break;

            case GgufType.BF16:
                ushort* bfDst = (ushort*)dst;
                for (int i = 0; i < k; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(src[i]);
                    bfDst[i] = (ushort)(bits >> 16);
                }
                break;

            case GgufType.Q4_0:
                QuantKernels.QuantizeQ4_0(src, (BlockQ4_0*)dst, k);
                break;

            case GgufType.Q8_0:
            {
                int nb = k / 32;
                var blocks = (BlockQ8_0*)dst;
                for (int b = 0; b < nb; b++)
                {
                    float maxAbs = 0f;
                    for (int j = 0; j < 32; j++)
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(src[b * 32 + j]));
                    float d = maxAbs / 127.0f;
                    float invD = d > 0f ? 1.0f / d : 0f;
                    blocks[b].Delta = (Half)d;
                    for (int j = 0; j < 32; j++)
                        blocks[b].Qs[j] = (sbyte)Math.Clamp((int)MathF.Round(src[b * 32 + j] * invD), -127, 127);
                }
                break;
            }

            case GgufType.Q5_0:
            {
                int nb = k / 32;
                var blocks = (BlockQ5_0*)dst;
                for (int b = 0; b < nb; b++)
                {
                    float min = float.MaxValue;
                    float maxAbs = 0f;
                    for (int j = 0; j < 32; j++)
                    {
                        float v = src[b * 32 + j];
                        min = MathF.Min(min, v);
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(v));
                    }
                    float d = MathF.Max(maxAbs / 15.0f, 1e-6f);
                    float invD = 1.0f / d;
                    blocks[b].Delta = (Half)d;
                    uint qh = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        int q0 = Math.Clamp((int)MathF.Round(src[b * 32 + j] * invD) + 16, 0, 31);
                        int q1 = Math.Clamp((int)MathF.Round(src[b * 32 + j + 16] * invD) + 16, 0, 31);
                        blocks[b].Qs[j] = (byte)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                        if ((q0 & 0x10) != 0) qh |= (1u << j);
                        if ((q1 & 0x10) != 0) qh |= (1u << (j + 16));
                    }
                    blocks[b].Qh = qh;
                }
                break;
            }

            case GgufType.Q2_K:
            {
                int nb = k / 256;
                var blocks = (BlockQ2_K*)dst;
                for (int b = 0; b < nb; b++)
                {
                    float min = float.MaxValue, max = float.MinValue;
                    for (int j = 0; j < 256; j++)
                    {
                        min = MathF.Min(min, src[b * 256 + j]);
                        max = MathF.Max(max, src[b * 256 + j]);
                    }
                    float span = MathF.Max(max - min, 1e-5f);
                    blocks[b].Delta = (Half)(span / (3.0f * 15.0f));
                    blocks[b].DeltaMin = (Half)(MathF.Abs(min) / 15.0f);
                    float scaleVal = (float)blocks[b].Delta * 15.0f;
                    float minVal = (float)blocks[b].DeltaMin * 15.0f;
                    float invS = scaleVal > 1e-8f ? 1.0f / scaleVal : 0f;
                    for (int s = 0; s < 16; s++)
                        blocks[b].Scales[s] = 0xFF;

                    for (int n = 0; n < 2; n++)
                    {
                        byte* qOut = blocks[b].Qs + n * 32;
                        float* xIn = src + b * 256 + n * 128;
                        for (int l = 0; l < 32; l++)
                        {
                            int q0 = Math.Clamp((int)MathF.Round((xIn[l] + minVal) * invS), 0, 3);
                            int q1 = Math.Clamp((int)MathF.Round((xIn[l + 32] + minVal) * invS), 0, 3);
                            int q2 = Math.Clamp((int)MathF.Round((xIn[l + 64] + minVal) * invS), 0, 3);
                            int q3 = Math.Clamp((int)MathF.Round((xIn[l + 96] + minVal) * invS), 0, 3);
                            qOut[l] = (byte)(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
                        }
                    }
                }
                break;
            }

            case GgufType.IQ2_XXS:
            {
                int nb = k / 256;
                var blocks = (BlockIQ2_XXS*)dst;
                for (int b = 0; b < nb; b++)
                {
                    float maxAbs = 0f;
                    for (int j = 0; j < 256; j++)
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(src[b * 256 + j]));
                    blocks[b].Delta = (Half)MathF.Max(maxAbs / 8.0f, 1e-4f);
                    byte* qBytes = (byte*)blocks[b].Qs;
                    for (int g = 0; g < 32; g++)
                    {
                        byte gridIdx = (byte)((rowSeed * 31 + b * 13 + g * 7) & 0xFF);
                        qBytes[g] = gridIdx;
                    }
                    // Scales/signs packed into Qs[32..63]: keep scales modest and sign indices valid
                    for (int ib = 0; ib < 8; ib++)
                    {
                        uint signs = 0x152A3C49u ^ (uint)((rowSeed + ib) * 0x1020304u);
                        uint scale = (uint)((ib + rowSeed) % 15);
                        uint aux = (signs & 0x0FFFFFFFu) | (scale << 28);
                        int baseByte = 32 + ib * 4;
                        qBytes[baseByte + 0] = (byte)(aux & 0xFF);
                        qBytes[baseByte + 1] = (byte)((aux >> 8) & 0xFF);
                        qBytes[baseByte + 2] = (byte)((aux >> 16) & 0xFF);
                        qBytes[baseByte + 3] = (byte)((aux >> 24) & 0xFF);
                    }
                }
                break;
            }

            case GgufType.IQ2_XS:
                QuantKernels.QuantizeIQ2_XS(src, (BlockIQ2_XS*)dst, k);
                break;

            case GgufType.IQ2_S:
                QuantKernels.QuantizeIQ2_S(src, (BlockIQ2_S*)dst, k);
                break;

            case GgufType.IQ3_XXS:
                QuantKernels.QuantizeIQ3_XXS(src, (BlockIQ3_XXS*)dst, k);
                break;

            case GgufType.IQ3_S:
                QuantKernels.QuantizeIQ3_S(src, (BlockIQ3_S*)dst, k);
                break;

            case GgufType.IQ4_NL:
                QuantKernels.QuantizeIQ4_NL(src, (BlockIQ4_NL*)dst, k);
                break;

            case GgufType.IQ4_XS:
            {
                int nb = k / 256;
                var blocks = (BlockIQ4_XS*)dst;
                for (int b = 0; b < nb; b++)
                {
                    float maxAbs = 0f;
                    for (int j = 0; j < 256; j++)
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(src[b * 256 + j]));
                    float d = MathF.Max(maxAbs / 127.0f, 1e-5f);
                    blocks[b].Delta = (Half)d;
                    blocks[b].ScalesH = 0x5555;
                    for (int s = 0; s < 4; s++)
                        blocks[b].ScalesL[s] = 0x44;
                    float scale = d * 12.0f;
                    for (int ib = 0; ib < 8; ib++)
                    {
                        for (int j = 0; j < 16; j++)
                        {
                            float v0 = src[b * 256 + ib * 32 + j] / scale;
                            float v1 = src[b * 256 + ib * 32 + j + 16] / scale;
                            int q0 = Math.Clamp((int)MathF.Round(v0 + 7.5f), 0, 15);
                            int q1 = Math.Clamp((int)MathF.Round(v1 + 7.5f), 0, 15);
                            blocks[b].Qs[ib * 16 + j] = (byte)(q0 | (q1 << 4));
                        }
                    }
                }
                break;
            }

            default:
                throw new NotSupportedException($"Test helper does not support GgufType {type}");
        }
    }

    private static void DequantizeByType(GgufType type, byte* src, float* dst, int k)
    {
        switch (type)
        {
            case GgufType.IQ2_XXS: QuantKernels.DequantizeIQ2_XXS((BlockIQ2_XXS*)src, dst, k); break;
            case GgufType.IQ2_XS: QuantKernels.DequantizeIQ2_XS((BlockIQ2_XS*)src, dst, k); break;
            case GgufType.IQ2_S: QuantKernels.DequantizeIQ2_S((BlockIQ2_S*)src, dst, k); break;
            case GgufType.IQ3_XXS: QuantKernels.DequantizeIQ3_XXS((BlockIQ3_XXS*)src, dst, k); break;
            case GgufType.IQ3_S: QuantKernels.DequantizeIQ3_S((BlockIQ3_S*)src, dst, k); break;
            case GgufType.IQ4_NL: QuantKernels.DequantizeIQ4_NL((BlockIQ4_NL*)src, dst, k); break;
            case GgufType.IQ4_XS: QuantKernels.DequantizeIQ4_XS((BlockIQ4_XS*)src, dst, k); break;
            default: throw new NotSupportedException(type.ToString());
        }
    }

    private static float VecDotByType(GgufType type, byte* row, float* x, int k) => type switch
    {
        GgufType.IQ2_XXS => QuantKernels.VecDotIQ2_XXS((BlockIQ2_XXS*)row, x, k),
        GgufType.IQ2_XS => QuantKernels.VecDotIQ2_XS((BlockIQ2_XS*)row, x, k),
        GgufType.IQ2_S => QuantKernels.VecDotIQ2_S((BlockIQ2_S*)row, x, k),
        GgufType.IQ3_XXS => QuantKernels.VecDotIQ3_XXS((BlockIQ3_XXS*)row, x, k),
        GgufType.IQ3_S => QuantKernels.VecDotIQ3_S((BlockIQ3_S*)row, x, k),
        GgufType.IQ4_NL => QuantKernels.VecDotIQ4_NL((BlockIQ4_NL*)row, x, k),
        GgufType.IQ4_XS => QuantKernels.VecDotIQ4_XS((BlockIQ4_XS*)row, x, k),
        _ => throw new NotSupportedException(type.ToString())
    };

    private static void WriteGgufFile(
        string path,
        List<(string Key, GgufValueType Type, object Value)> kvEntries,
        List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kvEntries.Count);

        foreach (var (key, type, val) in kvEntries)
        {
            WriteString(bw, key);
            bw.Write((uint)type);
            WriteMetadataValue(bw, type, val);
        }

        ulong currentOffset = 0;
        var alignedOffsets = new ulong[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            alignedOffsets[i] = currentOffset;
            ulong len = (ulong)tensors[i].Data.Length;
            currentOffset += ((len + 31UL) / 32UL) * 32UL;
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, dims, type, _) = tensors[i];
            WriteString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims)
                bw.Write(d);
            bw.Write((uint)type);
            bw.Write(alignedOffsets[i]);
        }

        long pos = fs.Position;
        long padHeader = ((pos + 31L) / 32L) * 32L - pos;
        if (padHeader > 0)
            bw.Write(new byte[padHeader]);

        for (int i = 0; i < tensors.Count; i++)
        {
            byte[] data = tensors[i].Data;
            bw.Write(data);
            long rem = data.Length % 32;
            if (rem != 0)
                bw.Write(new byte[32 - rem]);
        }
    }

    private static void WriteString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static void WriteMetadataValue(BinaryWriter bw, GgufValueType type, object val)
    {
        switch (type)
        {
            case GgufValueType.String:
                WriteString(bw, (string)val);
                break;
            case GgufValueType.Uint32:
                bw.Write((uint)val);
                break;
            case GgufValueType.Float32:
                bw.Write((float)val);
                break;
            case GgufValueType.Array:
                string[] arr = (string[])val;
                bw.Write((uint)GgufValueType.String);
                bw.Write((ulong)arr.Length);
                foreach (string s in arr)
                    WriteString(bw, s);
                break;
            default:
                throw new NotSupportedException($"Metadata type {type} not supported in test writer.");
        }
    }
}
