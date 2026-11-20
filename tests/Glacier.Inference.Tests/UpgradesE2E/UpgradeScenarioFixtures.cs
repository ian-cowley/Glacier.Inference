using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

namespace Glacier.Inference.Tests.UpgradesE2E;

internal static unsafe class UpgradeScenarioFixtures
{
    public static float[] CreateDeterministicFloatMatrix(int rows, int cols, float scale = 0.08f, int seed = 1234)
    {
        var rng = new Random(seed);
        float[] mat = new float[rows * cols];
        for (int i = 0; i < mat.Length; i++)
        {
            mat[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        }
        return mat;
    }

    public static byte[] FloatMatrixToBytes(float[] mat)
    {
        byte[] bytes = new byte[mat.Length * sizeof(float)];
        MemoryMarshal.AsBytes(mat.AsSpan()).CopyTo(bytes);
        return bytes;
    }

    public static byte[] QuantizeF32ToQ8_0(float[] src, int rows, int cols)
    {
        int blocksPerRow = cols / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 34];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = r * cols + b * 32;
                int dstOff = (r * blocksPerRow + b) * 34;

                float amax = 0f;
                for (int i = 0; i < 32; i++)
                {
                    float av = MathF.Abs(src[srcOff + i]);
                    if (av > amax) amax = av;
                }

                Half d = (Half)(amax / 127.0f);
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(dstOff, 2), d);
                float df = (float)d;
                float inv = df > 0f ? 1.0f / df : 0f;

                for (int i = 0; i < 32; i++)
                {
                    int q = Math.Clamp((int)MathF.Round(src[srcOff + i] * inv), -127, 127);
                    bytes[dstOff + 2 + i] = unchecked((byte)(sbyte)q);
                }
            }
        }

        return bytes;
    }

    public static byte[] QuantizeF32ToIq(float[] src, int rows, int cols, GgufType targetType)
    {
        if (targetType == GgufType.Q8_0)
            return QuantizeF32ToQ8_0(src, rows, cols);
        if (targetType == GgufType.F32)
            return FloatMatrixToBytes(src);

        int rowBytes = (int)GgufTypes.GetRowBytes(targetType, cols);
        byte[] dst = new byte[rows * rowBytes];

        fixed (float* pSrc = src)
        fixed (byte* pDst = dst)
        {
            for (int r = 0; r < rows; r++)
            {
                float* srcRow = pSrc + (long)r * cols;
                byte* dstRow = pDst + (long)r * rowBytes;

                switch (targetType)
                {
                    case GgufType.IQ4_NL:
                        QuantKernels.QuantizeIQ4_NL(srcRow, (BlockIQ4_NL*)dstRow, cols);
                        break;
                    case GgufType.IQ4_XS:
                        QuantizeRowToIQ4_XS(srcRow, (BlockIQ4_XS*)dstRow, cols);
                        break;
                    case GgufType.IQ3_S:
                        QuantKernels.QuantizeIQ3_S(srcRow, (BlockIQ3_S*)dstRow, cols);
                        break;
                    case GgufType.IQ3_XXS:
                        QuantKernels.QuantizeIQ3_XXS(srcRow, (BlockIQ3_XXS*)dstRow, cols);
                        break;
                    case GgufType.IQ2_XXS:
                        QuantizeRowToIQ2_XXS(srcRow, (BlockIQ2_XXS*)dstRow, cols);
                        break;
                    case GgufType.IQ2_XS:
                        QuantKernels.QuantizeIQ2_XS(srcRow, (BlockIQ2_XS*)dstRow, cols);
                        break;
                    case GgufType.IQ2_S:
                        QuantKernels.QuantizeIQ2_S(srcRow, (BlockIQ2_S*)dstRow, cols);
                        break;
                    default:
                        throw new NotSupportedException($"QuantizeF32ToIq does not support {targetType}.");
                }
            }
        }

        return dst;
    }

    private static void QuantizeRowToIQ4_XS(float* src, BlockIQ4_XS* dst, int cols)
    {
        const int QK_K = 256;
        int nb = cols / QK_K;
        float[] kValues = QuantKernels.KValuesIq4Nl;
        Span<float> subMax = stackalloc float[8];

        for (int i = 0; i < nb; i++)
        {
            float* x = src + i * QK_K;
            float maxSubMax = 0f;
            for (int ib = 0; ib < 8; ib++)
            {
                float sm = 0f;
                float* xb = x + ib * 32;
                for (int j = 0; j < 32; j++)
                {
                    float a = MathF.Abs(xb[j]);
                    if (a > sm) sm = a;
                }
                subMax[ib] = sm;
                if (sm > maxSubMax) maxSubMax = sm;
            }

            float d = maxSubMax / (31.0f * 127.0f);
            dst[i].Delta = (Half)d;
            float dActual = (float)dst[i].Delta;

            ushort scalesH = 0;
            for (int k = 0; k < 4; k++) dst[i].ScalesL[k] = 0;

            for (int ib = 0; ib < 8; ib++)
            {
                int ls;
                if (dActual <= 1e-12f || subMax[ib] <= 0f)
                {
                    ls = 32;
                }
                else
                {
                    float desiredDl = subMax[ib] / 127.0f;
                    ls = Math.Clamp(32 + (int)MathF.Round(desiredDl / dActual), 33, 63);
                }

                dst[i].ScalesL[ib / 2] |= (byte)((ls & 0x0F) << (4 * (ib % 2)));
                scalesH |= (ushort)(((ls >> 4) & 3) << (2 * ib));

                float dl = dActual * (ls - 32);
                float invDl = MathF.Abs(dl) > 1e-12f ? 1.0f / dl : 0f;
                float* xb = x + ib * 32;
                byte* qs = dst[i].Qs + ib * 16;

                for (int j = 0; j < 16; j++)
                {
                    int q0 = FindNearestIq4Nl(xb[j] * invDl, kValues);
                    int q1 = FindNearestIq4Nl(xb[j + 16] * invDl, kValues);
                    qs[j] = (byte)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }
            }

            dst[i].ScalesH = scalesH;
        }
    }

    private static int FindNearestIq4Nl(float scaledVal, float[] kValues)
    {
        int best = 0;
        float bestErr = MathF.Abs(scaledVal - kValues[0]);
        for (int idx = 1; idx < 16; idx++)
        {
            float err = MathF.Abs(scaledVal - kValues[idx]);
            if (err < bestErr)
            {
                bestErr = err;
                best = idx;
            }
        }
        return best;
    }

    private static void QuantizeRowToIQ2_XXS(float* src, BlockIQ2_XXS* dst, int cols)
    {
        const int QK_K = 256;
        int nb = cols / QK_K;
        fixed (ulong* pGrid = QuantKernels.Iq2xxsGrid)
        fixed (byte* pSigns = QuantKernels.KSignsIq2Xs)
        {
            for (int i = 0; i < nb; i++)
            {
                float* x = src + i * QK_K;
                float maxAbs = 0f;
                for (int j = 0; j < QK_K; j++)
                {
                    float a = MathF.Abs(x[j]);
                    if (a > maxAbs) maxAbs = a;
                }

                float d = maxAbs / (43.0f * 15.5f * 0.25f);
                dst[i].Delta = (Half)d;
                float dActual = (float)dst[i].Delta;
                ushort* qs = dst[i].Qs;

                for (int ib32 = 0; ib32 < 8; ib32++)
                {
                    float subMax = 0f;
                    for (int j = 0; j < 32; j++)
                    {
                        float a = MathF.Abs(x[ib32 * 32 + j]);
                        if (a > subMax) subMax = a;
                    }
                    int ls = dActual > 1e-12f
                        ? Math.Clamp((int)MathF.Round((subMax / (43.0f * dActual * 0.25f)) - 0.5f), 0, 15)
                        : 0;

                    float db = dActual * (0.5f + ls) * 0.25f;
                    float invDb = db > 1e-12f ? 1.0f / db : 0f;

                    uint aux0 = 0;
                    uint aux1 = (uint)ls << 28;

                    for (int l = 0; l < 4; l++)
                    {
                        float* xg = x + ib32 * 32 + l * 8;
                        int rawSigns = 0;
                        float minMag = float.MaxValue;
                        int minIdx = 0;
                        for (int j = 0; j < 8; j++)
                        {
                            float v = xg[j];
                            if (v < 0f) rawSigns |= (1 << j);
                            float a = MathF.Abs(v);
                            if (a < minMag) { minMag = a; minIdx = j; }
                        }

                        int signIdx = rawSigns & 127;
                        if (pSigns[signIdx] != rawSigns && minIdx < 7)
                        {
                            signIdx = (rawSigns ^ (1 << minIdx)) & 127;
                        }
                        byte usedSigns = pSigns[signIdx];

                        int bestGrid = 0;
                        float bestErr = float.MaxValue;
                        for (int g = 0; g < 256; g++)
                        {
                            byte* pg = (byte*)(pGrid + g);
                            float err = 0f;
                            for (int j = 0; j < 8; j++)
                            {
                                float qv = pg[j] * (((usedSigns >> j) & 1) != 0 ? -1f : 1f);
                                float diff = qv - (xg[j] * invDb);
                                err += diff * diff;
                            }
                            if (err < bestErr)
                            {
                                bestErr = err;
                                bestGrid = g;
                            }
                        }

                        aux0 |= (uint)(bestGrid & 0xFF) << (8 * l);
                        aux1 |= (uint)(signIdx & 127) << (7 * l);
                    }

                    *(uint*)(qs + 4 * ib32) = aux0;
                    *(uint*)(qs + 4 * ib32 + 2) = aux1;
                }
            }
        }
    }

    /// <summary>
    /// Builds a mixed-tensor Unsloth Dynamic 3.1 GGUF model. When <paramref name="sharedWeightsSeed"/> is
    /// shared between a Q8_0 reference model and a UD quantized model, both models encode the exact same underlying
    /// FP32 weights so KL divergence stays faithfully low; when <paramref name="corruptWeights"/> is true, output/FFN
    /// tensors are scrambled to simulate a broken quantization.
    /// </summary>
    public static string BuildMixedUdQwen2Model(
        string path,
        string profileName,
        int dim = 256,
        int ffnDim = 256,
        int nLayers = 1,
        int nHeads = 4,
        int nHeadsKv = 2,
        int vocabSize = 32,
        int maxSeqLen = 32,
        int sharedWeightsSeed = 4242,
        bool useQ8ReferenceOnly = false,
        bool corruptWeights = false)
    {
        var b = new SyntheticGgufBuilder();
        b.AddString("general.architecture", "qwen2");
        b.AddString("general.name", $"synthetic-ud31-{profileName}");
        b.AddUInt32("qwen2.context_length", (uint)maxSeqLen);
        b.AddUInt32("qwen2.embedding_length", (uint)dim);
        b.AddUInt32("qwen2.feed_forward_length", (uint)ffnDim);
        b.AddUInt32("qwen2.block_count", (uint)nLayers);
        b.AddUInt32("qwen2.attention.head_count", (uint)nHeads);
        b.AddUInt32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        b.AddFloat32("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        b.AddFloat32("qwen2.rope.freq_base", 10000.0f);

        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++)
            tokens.Add(i < 28 ? ((char)('a' + i - 2)).ToString() : $"tok{i}");
        b.AddString("tokenizer.ggml.model", "gpt2");
        b.AddStringArray("tokenizer.ggml.tokens", tokens);
        b.AddStringArray("tokenizer.ggml.merges", Array.Empty<string>());
        b.AddUInt32("tokenizer.ggml.bos_token_id", 0);
        b.AddUInt32("tokenizer.ggml.eos_token_id", 1);

        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        float[] embdF32 = CreateDeterministicFloatMatrix(vocabSize, dim, 0.03f, sharedWeightsSeed + 1);
        float[] outF32 = CreateDeterministicFloatMatrix(vocabSize, dim, 0.03f, sharedWeightsSeed + 2);
        for (int t = 0; t < Math.Min(vocabSize, dim); t++)
        {
            embdF32[t * dim + t] += 2.5f;
            int nextTok = 2 + ((t + 1) % Math.Max(1, vocabSize - 2));
            outF32[nextTok * dim + t] += 2.5f;
        }
        if (corruptWeights)
        {
            for (int i = 0; i < outF32.Length; i++)
            {
                outF32[i] = -outF32[(outF32.Length - 1) - i] * 4.5f;
            }
        }

        b.AddTensor2D("token_embd.weight", vocabSize, dim, GgufType.F32, FloatMatrixToBytes(embdF32));
        b.AddTensor1D("output_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
        b.AddTensor2D("output.weight", vocabSize, dim, GgufType.F32, FloatMatrixToBytes(outF32));

        (GgufType qkvType, GgufType oType, GgufType gateUpType, GgufType downType) = useQ8ReferenceOnly
            ? (GgufType.Q8_0, GgufType.Q8_0, GgufType.Q8_0, GgufType.Q8_0)
            : profileName switch
            {
                "UD-IQ2_XS" => (GgufType.IQ3_XXS, GgufType.IQ2_XS, GgufType.IQ2_XXS, GgufType.IQ3_S),
                "UD-IQ3"    => (GgufType.IQ3_S,   GgufType.IQ3_XXS, GgufType.IQ3_S,  GgufType.IQ4_XS),
                "UD-IQ4_XS" => (GgufType.IQ4_XS,  GgufType.IQ4_NL, GgufType.IQ4_XS, GgufType.Q8_0),
                _           => (GgufType.IQ4_XS,  GgufType.IQ3_S,  GgufType.IQ4_NL, GgufType.Q8_0)
            };

        for (int l = 0; l < nLayers; l++)
        {
            int ls = sharedWeightsSeed + (l + 1) * 100;
            b.AddTensor1D($"blk.{l}.attn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            b.AddTensor1D($"blk.{l}.ffn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));

            byte[] Encode(int rows, int cols, int s, GgufType t)
            {
                float[] w = CreateDeterministicFloatMatrix(rows, cols, 0.03f, s);
                if (corruptWeights)
                {
                    for (int i = 0; i < w.Length; i++) w[i] = -w[i] * 3.0f;
                }
                return QuantizeF32ToIq(w, rows, cols, t);
            }

            b.AddTensor2D($"blk.{l}.attn_q.weight", qDim, dim, qkvType, Encode(qDim, dim, ls + 1, qkvType));
            b.AddTensor2D($"blk.{l}.attn_k.weight", kvDim, dim, qkvType, Encode(kvDim, dim, ls + 2, qkvType));
            b.AddTensor2D($"blk.{l}.attn_v.weight", kvDim, dim, qkvType, Encode(kvDim, dim, ls + 3, qkvType));
            b.AddTensor2D($"blk.{l}.attn_output.weight", dim, qDim, oType, Encode(dim, qDim, ls + 4, oType));
            b.AddTensor2D($"blk.{l}.ffn_gate.weight", ffnDim, dim, gateUpType, Encode(ffnDim, dim, ls + 5, gateUpType));
            b.AddTensor2D($"blk.{l}.ffn_up.weight", ffnDim, dim, gateUpType, Encode(ffnDim, dim, ls + 6, gateUpType));
            b.AddTensor2D($"blk.{l}.ffn_down.weight", dim, ffnDim, downType, Encode(dim, ffnDim, ls + 7, downType));
        }

        b.WriteToFile(path);
        return path;
    }

    /// <summary>
    /// Builds a valid synthetic Qwen2 MoE GGUF model loadable by <see cref="TieredMoERunner"/>
    /// on both CPU and Direct3D 12 backends.
    /// </summary>
    public static string BuildTinyQwen2MoeModel(
        string path,
        int dim = 64,
        int ffnDim = 64,
        int expertFfnDim = 64,
        int nLayers = 2,
        int nHeads = 4,
        int nHeadsKv = 2,
        int numExperts = 8,
        int topK = 4,
        int vocabSize = 32,
        int maxSeqLen = 64,
        bool normTopK = true,
        int seed = 5150)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        var metadata = new List<(string Key, Action<BinaryWriter> WriteVal)>();
        void AddStr(string k, string v) => metadata.Add((k, bw => { bw.Write(8u); WriteGgufStr(bw, v); }));
        void AddU32(string k, uint v) => metadata.Add((k, bw => { bw.Write(4u); bw.Write(v); }));
        void AddF32(string k, float v) => metadata.Add((k, bw => { bw.Write(6u); bw.Write(v); }));
        void AddBool(string k, bool v) => metadata.Add((k, bw => { bw.Write(7u); bw.Write((byte)(v ? 1 : 0)); }));
        void AddStrArr(string k, IReadOnlyList<string> vals) => metadata.Add((k, bw =>
        {
            bw.Write(9u);
            bw.Write(8u);
            bw.Write((ulong)vals.Count);
            foreach (var s in vals) WriteGgufStr(bw, s);
        }));

        AddStr("general.architecture", "qwen2");
        AddStr("general.name", "synthetic-qwen2-moe-e2e");
        AddU32("qwen2.context_length", (uint)maxSeqLen);
        AddU32("qwen2.embedding_length", (uint)dim);
        AddU32("qwen2.feed_forward_length", (uint)ffnDim);
        AddU32("qwen2.expert_feed_forward_length", (uint)expertFfnDim);
        AddU32("qwen2.expert_count", (uint)numExperts);
        AddU32("qwen2.expert_used_count", (uint)topK);
        AddBool("qwen2.expert_weights_norm", normTopK);
        AddU32("qwen2.block_count", (uint)nLayers);
        AddU32("qwen2.attention.head_count", (uint)nHeads);
        AddU32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        AddFloat32Meta(metadata, "qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        AddF32("qwen2.rope.freq_base", 10000.0f);

        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++)
            tokens.Add(i < 28 ? ((char)('a' + i - 2)).ToString() : $"t{i}");
        AddStr("tokenizer.ggml.model", "gpt2");
        AddStrArr("tokenizer.ggml.tokens", tokens);
        AddStrArr("tokenizer.ggml.merges", Array.Empty<string>());
        AddU32("tokenizer.ggml.bos_token_id", 0);
        AddU32("tokenizer.ggml.eos_token_id", 1);

        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] Data)>();
        void AddT1(string name, int len, byte[] data)
            => tensors.Add((name, new[] { (ulong)len }, GgufType.F32, data));
        void AddT2(string name, int rows, int cols, byte[] data)
            => tensors.Add((name, new[] { (ulong)cols, (ulong)rows }, GgufType.F32, data));
        void AddT3(string name, int experts, int rows, int cols, byte[] data)
            => tensors.Add((name, new[] { (ulong)cols, (ulong)rows, (ulong)experts }, GgufType.F32, data));

        AddT2("token_embd.weight", vocabSize, dim, FxQuantData.CreateF32MatrixBytes(vocabSize, dim, 0.1f, seed + 1));
        AddT1("output_norm.weight", dim, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
        AddT2("output.weight", vocabSize, dim, FxQuantData.CreateF32MatrixBytes(vocabSize, dim, 0.1f, seed + 2));

        for (int l = 0; l < nLayers; l++)
        {
            int ls = seed + (l + 1) * 100;
            AddT1($"blk.{l}.attn_norm.weight", dim, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            AddT2($"blk.{l}.attn_q.weight", qDim, dim, FxQuantData.CreateF32MatrixBytes(qDim, dim, 0.08f, ls + 1));
            AddT2($"blk.{l}.attn_k.weight", kvDim, dim, FxQuantData.CreateF32MatrixBytes(kvDim, dim, 0.08f, ls + 2));
            AddT2($"blk.{l}.attn_v.weight", kvDim, dim, FxQuantData.CreateF32MatrixBytes(kvDim, dim, 0.08f, ls + 3));
            AddT2($"blk.{l}.attn_output.weight", dim, qDim, FxQuantData.CreateF32MatrixBytes(dim, qDim, 0.08f, ls + 4));
            AddT1($"blk.{l}.ffn_norm.weight", dim, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));

            AddT2($"blk.{l}.ffn_gate_inp.weight", numExperts, dim, FxQuantData.CreateF32MatrixBytes(numExperts, dim, 0.08f, ls + 5));
            AddT3($"blk.{l}.ffn_gate_exps.weight", numExperts, expertFfnDim, dim, FxQuantData.CreateF32MatrixBytes(numExperts * expertFfnDim, dim, 0.05f, ls + 6));
            AddT3($"blk.{l}.ffn_up_exps.weight", numExperts, expertFfnDim, dim, FxQuantData.CreateF32MatrixBytes(numExperts * expertFfnDim, dim, 0.05f, ls + 7));
            AddT3($"blk.{l}.ffn_down_exps.weight", numExperts, dim, expertFfnDim, FxQuantData.CreateF32MatrixBytes(numExperts * dim, expertFfnDim, 0.05f, ls + 8));
        }

        w.Write(0x46554747u); // "GGUF"
        w.Write(3u);          // v3
        w.Write((ulong)tensors.Count);
        w.Write((ulong)metadata.Count);

        foreach (var (k, writeVal) in metadata)
        {
            WriteGgufStr(w, k);
            writeVal(w);
        }

        ulong curOffset = 0;
        foreach (var t in tensors)
        {
            WriteGgufStr(w, t.Name);
            w.Write((uint)t.Dims.Length);
            foreach (ulong d in t.Dims) w.Write(d);
            w.Write((uint)t.Type);
            w.Write(curOffset);
            curOffset += AlignUp32((ulong)t.Data.Length);
        }

        long headerEnd = fs.Position;
        long dataStart = (long)AlignUp32((ulong)headerEnd);
        for (long i = headerEnd; i < dataStart; i++) w.Write((byte)0);

        foreach (var t in tensors)
        {
            w.Write(t.Data);
            int pad = (int)(AlignUp32((ulong)t.Data.Length) - (ulong)t.Data.Length);
            for (int p = 0; p < pad; p++) w.Write((byte)0);
        }

        return path;
    }

    /// <summary>
    /// Builds a dense tiny Qwen2 GGUF with non-EOS output bias and well-separated logit margins so
    /// multi-step generation and KV-precision KL comparisons never prematurely stop at token 1 (&lt;eos&gt;)
    /// or flip random near-zero ties.
    /// </summary>
    public static string BuildTinyDenseWithClearMargins(
        string path,
        int dim = 64,
        int ffnDim = 128,
        int nLayers = 2,
        int nHeads = 4,
        int nHeadsKv = 2,
        int vocabSize = 32,
        int maxSeqLen = 64,
        int seed = 77)
    {
        var b = new SyntheticGgufBuilder();
        b.AddString("general.architecture", "qwen2");
        b.AddString("general.name", "synthetic-qwen2-clear-margins");
        b.AddUInt32("qwen2.context_length", (uint)maxSeqLen);
        b.AddUInt32("qwen2.embedding_length", (uint)dim);
        b.AddUInt32("qwen2.feed_forward_length", (uint)ffnDim);
        b.AddUInt32("qwen2.block_count", (uint)nLayers);
        b.AddUInt32("qwen2.attention.head_count", (uint)nHeads);
        b.AddUInt32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        b.AddFloat32("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        b.AddFloat32("qwen2.rope.freq_base", 10000.0f);

        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++)
            tokens.Add(i < 28 ? ((char)('a' + i - 2)).ToString() : $"t{i}");
        b.AddString("tokenizer.ggml.model", "gpt2");
        b.AddStringArray("tokenizer.ggml.tokens", tokens);
        b.AddStringArray("tokenizer.ggml.merges", Array.Empty<string>());
        b.AddUInt32("tokenizer.ggml.bos_token_id", 0);
        b.AddUInt32("tokenizer.ggml.eos_token_id", 1);

        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        float[] embd = CreateDeterministicFloatMatrix(vocabSize, dim, 0.08f, seed + 1);
        float[] outW = CreateDeterministicFloatMatrix(vocabSize, dim, 0.08f, seed + 2);
        for (int t = 0; t < vocabSize; t++)
        {
            if (t >= 2 && t < dim)
            {
                embd[t * dim + t] += 1.5f;
                int nextTok = 2 + ((t - 1) % (vocabSize - 2));
                outW[nextTok * dim + t] += 1.5f;
            }
        }
        // Zero out <bos>/<eos> rows and ensure opposing rows exist so neither <bos> nor <eos> can ever be argmax
        Array.Clear(outW, 0 * dim, dim);
        Array.Clear(outW, 1 * dim, dim);
        outW[2 * dim + 0] += 1.0f;
        outW[3 * dim + 0] -= 1.0f;
        outW[4 * dim + 1] += 1.0f;
        outW[5 * dim + 1] -= 1.0f;

        b.AddTensor2D("token_embd.weight", vocabSize, dim, GgufType.F32, FloatMatrixToBytes(embd));
        b.AddTensor1D("output_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
        b.AddTensor2D("output.weight", vocabSize, dim, GgufType.F32, FloatMatrixToBytes(outW));

        for (int l = 0; l < nLayers; l++)
        {
            int ls = seed + (l + 1) * 100;
            b.AddTensor1D($"blk.{l}.attn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            b.AddTensor2D($"blk.{l}.attn_q.weight", qDim, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(qDim, dim, 0.1f, ls + 1));
            b.AddTensor2D($"blk.{l}.attn_k.weight", kvDim, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(kvDim, dim, 0.1f, ls + 2));
            b.AddTensor2D($"blk.{l}.attn_v.weight", kvDim, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(kvDim, dim, 0.1f, ls + 3));
            b.AddTensor2D($"blk.{l}.attn_output.weight", dim, qDim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(dim, qDim, 0.1f, ls + 4));
            b.AddTensor1D($"blk.{l}.ffn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            b.AddTensor2D($"blk.{l}.ffn_gate.weight", ffnDim, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(ffnDim, dim, 0.1f, ls + 5));
            b.AddTensor2D($"blk.{l}.ffn_up.weight", ffnDim, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(ffnDim, dim, 0.1f, ls + 6));
            b.AddTensor2D($"blk.{l}.ffn_down.weight", dim, ffnDim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(dim, ffnDim, 0.1f, ls + 7));
        }

        b.WriteToFile(path);
        return path;
    }

    public static void WriteVersionedEmptyGguf(string path, uint version)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var w = new BinaryWriter(fs);
        w.Write(0x46554747u);
        w.Write(version);
        w.Write(0UL);
        w.Write(0UL);
    }

    public static string WriteUnalignedOrUnknownDFlashSidecarGguf(
        string path,
        int cols = 100,
        int rows = 64,
        GgufType tensorType = GgufType.IQ2_XS,
        uint dflashBlockSize = 4)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        // Compute payload length safely even if tensorType is unknown (fallback count * 2)
        ulong elemCount = (ulong)cols * (ulong)rows;
        ulong payloadBytes = tensorType == GgufType.IQ2_XS
            ? (elemCount / 256UL) * 74UL
            : elemCount * 2UL;

        w.Write(0x46554747u);
        w.Write(3u);
        w.Write(1UL); // 1 tensor
        w.Write(2UL); // 2 metadata entries

        WriteGgufStr(w, "general.architecture");
        w.Write(8u);
        WriteGgufStr(w, "qwen2");

        WriteGgufStr(w, "dflash.block_size");
        w.Write(4u);
        w.Write(dflashBlockSize);

        WriteGgufStr(w, "blk.0.dflash.attn_q.weight");
        w.Write(2u);
        w.Write((ulong)cols);
        w.Write((ulong)rows);
        w.Write((uint)tensorType);
        w.Write(0UL);

        long headerEnd = fs.Position;
        long dataStart = (long)AlignUp32((ulong)headerEnd);
        for (long i = headerEnd; i < dataStart; i++) w.Write((byte)0);
        for (ulong i = 0; i < payloadBytes; i++) w.Write((byte)0);

        return path;
    }

    /// <summary>
    /// Builds a valid neural DFlash sidecar GGUF loadable by <see cref="DFlashDraftProvider.LoadFromSidecar"/>.
    /// Slot position embeddings and the shared head are configured deterministically so slot <c>j</c>
    /// predicts <paramref name="slotTargetTokens"/>[j] with high confidence across diffusion steps.
    /// </summary>
    public static string BuildPositiveDFlashSidecarGguf(
        string path,
        int dim = 64,
        int ffnDim = 64,
        int nHeads = 4,
        int nHeadsKv = 2,
        int vocabSize = 32,
        int blockSize = 4,
        int[]? slotTargetTokens = null)
    {
        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        var b = new SyntheticGgufBuilder();
        b.AddString("general.architecture", "qwen2");
        b.AddUInt32("qwen2.embedding_length", (uint)dim);
        b.AddUInt32("dflash.block_size", (uint)blockSize);

        float[] onesDim = new float[dim];
        Array.Fill(onesDim, 1.0f);

        float[] slotPos = new float[blockSize * dim];
        for (int j = 0; j < blockSize; j++)
        {
            slotPos[j * dim + (j % dim)] = 6.0f;
        }

        float[] sharedHead = new float[vocabSize * dim];
        for (int j = 0; j < blockSize; j++)
        {
            int targetTok = (slotTargetTokens != null && j < slotTargetTokens.Length)
                ? slotTargetTokens[j]
                : (5 + j) % vocabSize;
            sharedHead[targetTok * dim + (j % dim)] = 12.0f;
        }

        b.AddTensor1D("dflash.mask_embd.weight", dim, GgufType.F32, new byte[dim * sizeof(float)]);
        b.AddTensor2D("dflash.slot_pos_embd.weight", blockSize, dim, GgufType.F32, FloatMatrixToBytes(slotPos));
        b.AddTensor1D("dflash.context_norm.weight", dim, GgufType.F32, FloatMatrixToBytes(onesDim));
        b.AddTensor2D("dflash.context_proj.weight", dim, dim, GgufType.F32, new byte[dim * dim * sizeof(float)]);
        b.AddTensor1D("dflash.shared_head.norm.weight", dim, GgufType.F32, FloatMatrixToBytes(onesDim));
        b.AddTensor2D("dflash.shared_head.weight", vocabSize, dim, GgufType.F32, FloatMatrixToBytes(sharedHead));

        b.AddTensor1D("blk.0.dflash.attn_norm.weight", dim, GgufType.F32, FloatMatrixToBytes(onesDim));
        b.AddTensor2D("blk.0.dflash.attn_q.weight", qDim, dim, GgufType.F32, new byte[qDim * dim * sizeof(float)]);
        b.AddTensor2D("blk.0.dflash.attn_k.weight", kvDim, dim, GgufType.F32, new byte[kvDim * dim * sizeof(float)]);
        b.AddTensor2D("blk.0.dflash.attn_v.weight", kvDim, dim, GgufType.F32, new byte[kvDim * dim * sizeof(float)]);
        b.AddTensor2D("blk.0.dflash.attn_output.weight", dim, qDim, GgufType.F32, new byte[dim * qDim * sizeof(float)]);
        b.AddTensor1D("blk.0.dflash.ffn_norm.weight", dim, GgufType.F32, FloatMatrixToBytes(onesDim));
        b.AddTensor2D("blk.0.dflash.ffn_gate.weight", ffnDim, dim, GgufType.F32, new byte[ffnDim * dim * sizeof(float)]);
        b.AddTensor2D("blk.0.dflash.ffn_up.weight", ffnDim, dim, GgufType.F32, new byte[ffnDim * dim * sizeof(float)]);
        b.AddTensor2D("blk.0.dflash.ffn_down.weight", dim, ffnDim, GgufType.F32, new byte[dim * ffnDim * sizeof(float)]);

        b.WriteToFile(path);
        return path;
    }

    private static void AddFloat32Meta(List<(string Key, Action<BinaryWriter> WriteVal)> metadata, string k, float v)
        => metadata.Add((k, bw => { bw.Write(6u); bw.Write(v); }));

    private static void WriteGgufStr(BinaryWriter w, string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)utf8.Length);
        w.Write(utf8);
    }

    private static ulong AlignUp32(ulong v) => (v + 31UL) & ~31UL;
}

internal sealed class ScriptedEosSpeculativeTarget : ISpeculativeTarget, IHiddenStateSource
{
    private readonly TempDirScope _tmpScope;
    private readonly GgufFile _gguf;
    private readonly int[] _scriptedPredictions;
    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly float[] _hidden;
    private readonly float[] _logits;
    private int _stepIndex;

    public BpeTokenizer Tokenizer { get; }
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => 64;
    public bool SupportsLogitVerification => true;
    public bool IsRingBufferEnabled => false;
    public bool CanOverwriteKvFromPrefix => true;
    public long StateSnapshotBytes => 0;
    public bool HasUnsnapshottableRecurrentState => false;
    public int HiddenDim => _dim;

    public ScriptedEosSpeculativeTarget(int[] scriptedPredictions, int vocabSize = 32, int dim = 32)
    {
        _scriptedPredictions = scriptedPredictions;
        _vocabSize = vocabSize;
        _dim = dim;
        _hidden = new float[dim];
        _logits = new float[vocabSize];

        _tmpScope = new TempDirScope();
        var builder = new SyntheticGgufBuilder();
        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++)
            tokens.Add(i < 28 ? ((char)('a' + i - 2)).ToString() : $"t{i}");
        builder.AddString("tokenizer.ggml.model", "gpt2");
        builder.AddStringArray("tokenizer.ggml.tokens", tokens);
        builder.AddStringArray("tokenizer.ggml.merges", Array.Empty<string>());
        builder.AddUInt32("tokenizer.ggml.bos_token_id", 0);
        builder.AddUInt32("tokenizer.ggml.eos_token_id", 1);
        string tokPath = _tmpScope.GetFilePath("scripted_tok.gguf");
        builder.WriteToFile(tokPath);
        _gguf = GgufFile.Open(tokPath);
        Tokenizer = new BpeTokenizer(_gguf);
    }

    public void ResetKvCache() => _stepIndex = 0;

    public void Prefill(ReadOnlySpan<int> tokens) => _stepIndex = 0;

    public void ForwardToken(int token, int pos, bool computeLogits = true) => _stepIndex++;

    public void CopyCurrentLogits(Span<float> destination)
    {
        destination.Slice(0, _vocabSize).Fill(-5.0f);
        int pred = _scriptedPredictions[Math.Clamp(_stepIndex, 0, _scriptedPredictions.Length - 1)];
        destination[pred] = 10.0f;
    }

    public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        CopyCurrentLogits(_logits);
        int best = 0;
        for (int i = 1; i < _vocabSize; i++)
        {
            if (_logits[i] > _logits[best]) best = i;
        }
        return best;
    }

    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        for (int t = 0; t < tokens.Length; t++)
        {
            int idx = Math.Clamp(_stepIndex + 1 + t, 0, _scriptedPredictions.Length - 1);
            predictedTokens[t] = _scriptedPredictions[idx];
        }
        _stepIndex += tokens.Length;
    }

    public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
    {
        for (int t = 0; t < tokens.Length; t++)
        {
            var row = batchLogitsOut.Slice(t * _vocabSize, _vocabSize);
            row.Fill(-5.0f);
            int idx = Math.Clamp(_stepIndex + 1 + t, 0, _scriptedPredictions.Length - 1);
            row[_scriptedPredictions[idx]] = 10.0f;
        }
        _stepIndex += tokens.Length;
    }

    public ReadOnlySpan<float> GetLastHiddenState() => _hidden;
    public void CopyTokenEmbedding(int tokenId, Span<float> destination) => destination.Clear();
    public void EnablePositionHistory() { }
    public bool TryGetHiddenStateAt(int position, Span<float> destination) => false;

    public void Dispose()
    {
        _gguf.Dispose();
        _tmpScope.Dispose();
    }
}
