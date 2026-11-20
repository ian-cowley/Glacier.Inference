using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests.UpgradesE2E;

[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
internal sealed class UpgradeCudaFactAttribute : FactAttribute
{
    public UpgradeCudaFactAttribute()
    {
        try
        {
            if (!GpuContext.IsSupported)
            {
                Skip = "CUDA GPU not available on this host.";
            }
        }
        catch (Exception ex)
        {
            Skip = $"CUDA GPU probe failed: {ex.Message}";
        }
    }
}

[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
internal sealed class UpgradeD3D12FactAttribute : FactAttribute
{
    public UpgradeD3D12FactAttribute()
    {
        if (!OperatingSystem.IsWindows())
        {
            Skip = "D3D12 requires Windows.";
            return;
        }

        try
        {
            if (!D3D12Context.IsSupported)
            {
                Skip = "D3D12 GPU not available on this host.";
            }
        }
        catch (Exception ex)
        {
            Skip = $"D3D12 GPU probe failed: {ex.Message}";
        }
    }
}

internal sealed class TempDirScope : IDisposable
{
    public string DirPath { get; }

    public TempDirScope()
    {
        DirPath = Path.Combine(Path.GetTempPath(), "GlacierUpgradesE2E_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(DirPath);
    }

    public string GetFilePath(string fileName) => Path.Combine(DirPath, fileName);

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(DirPath))
            {
                Directory.Delete(DirPath, recursive: true);
            }
        }
        catch
        {
            // Best-effort cleanup of temp artifacts
        }
    }
}

internal static unsafe class Fx1Vectors
{
    public static (float[] K, float[] V, float[] Q) GenerateOrthogonalVectors(int nHeadsQ, int nHeadsKv, int headDim, int seed = 101)
    {
        var rng = new Random(seed);
        float[] k = new float[nHeadsKv * headDim];
        float[] v = new float[nHeadsKv * headDim];
        float[] q = new float[nHeadsQ * headDim];

        for (int h = 0; h < nHeadsKv; h++)
        {
            float normK = 0f;
            float normV = 0f;
            for (int d = 0; d < headDim; d++)
            {
                float kv = (float)(rng.NextDouble() * 2.0 - 1.0);
                float vv = (float)(rng.NextDouble() * 2.0 - 1.0);
                k[h * headDim + d] = kv;
                v[h * headDim + d] = vv;
                normK += kv * kv;
                normV += vv * vv;
            }

            float invK = normK > 0f ? 1.0f / MathF.Sqrt(normK) : 1.0f;
            float invV = normV > 0f ? 1.0f / MathF.Sqrt(normV) : 1.0f;
            for (int d = 0; d < headDim; d++)
            {
                k[h * headDim + d] *= invK * 2.0f;
                v[h * headDim + d] *= invV * 2.0f;
            }
        }

        for (int h = 0; h < nHeadsQ; h++)
        {
            float normQ = 0f;
            for (int d = 0; d < headDim; d++)
            {
                float qv = (float)(rng.NextDouble() * 2.0 - 1.0);
                q[h * headDim + d] = qv;
                normQ += qv * qv;
            }
            float invQ = normQ > 0f ? 1.0f / MathF.Sqrt(normQ) : 1.0f;
            for (int d = 0; d < headDim; d++)
            {
                q[h * headDim + d] *= invQ * 2.0f;
            }
        }

        return (k, v, q);
    }

    public static float[] GenerateRampVector(int length, float start = 0.0f, float step = 0.01f)
    {
        float[] vec = new float[length];
        for (int i = 0; i < length; i++)
        {
            vec[i] = start + i * step;
        }
        return vec;
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have equal length.");
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 && nb == 0) return 1.0f;
        if (na == 0 || nb == 0) return 0.0f;
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }

    public static float[] ComputeSoftAttentionReference(
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> kSeq,
        ReadOnlySpan<float> vSeq,
        int nHeadsQ,
        int nHeadsKv,
        int headDim,
        int seqLen,
        float attnScale)
    {
        float[] output = new float[nHeadsQ * headDim];
        int groupSize = nHeadsQ / nHeadsKv;
        double[] scores = new double[seqLen];

        for (int h = 0; h < nHeadsQ; h++)
        {
            int hKv = h / groupSize;
            int qOff = h * headDim;
            double maxScore = double.NegativeInfinity;

            for (int t = 0; t < seqLen; t++)
            {
                int kvOff = (t * nHeadsKv + hKv) * headDim;
                double dot = 0.0;
                for (int d = 0; d < headDim; d++)
                {
                    dot += (double)q[qOff + d] * kSeq[kvOff + d];
                }
                double s = dot * attnScale;
                scores[t] = s;
                if (s > maxScore) maxScore = s;
            }

            double sumExp = 0.0;
            for (int t = 0; t < seqLen; t++)
            {
                scores[t] = Math.Exp(scores[t] - maxScore);
                sumExp += scores[t];
            }

            for (int d = 0; d < headDim; d++)
            {
                double acc = 0.0;
                for (int t = 0; t < seqLen; t++)
                {
                    int kvOff = (t * nHeadsKv + hKv) * headDim;
                    acc += (scores[t] / sumExp) * vSeq[kvOff + d];
                }
                output[qOff + d] = (float)acc;
            }
        }

        return output;
    }
}

internal static unsafe class FxQuantData
{
    public static byte[] CreateValidIqPayload(GgufType type, int rows, int cols, int seed = 303)
    {
        var rng = new Random(seed);
        int rowBytes = (int)GgufTypes.GetRowBytes(type, cols);
        byte[] data = new byte[rows * rowBytes];
        int blockElems = type == GgufType.IQ4_NL ? 32 : 256;
        int blocksPerRow = cols / blockElems;
        int bytesPerBlock = (int)GgufTypes.GetRowBytes(type, blockElems);

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int offset = r * rowBytes + b * bytesPerBlock;
                Half d = (Half)(0.02f + (float)(rng.NextDouble() * 0.06));
                BinaryPrimitives.WriteHalfLittleEndian(data.AsSpan(offset, 2), d);

                switch (type)
                {
                    case GgufType.IQ2_XXS:
                    {
                        // 2B d + 8 * 8B (aux0=4 grid indices, aux1=4*7 sign bits + 4-bit scale)
                        for (int i = 0; i < 8; i++)
                        {
                            int gBase = offset + 2 + i * 8;
                            data[gBase + 0] = (byte)rng.Next(0, 256);
                            data[gBase + 1] = (byte)rng.Next(0, 256);
                            data[gBase + 2] = (byte)rng.Next(0, 256);
                            data[gBase + 3] = (byte)rng.Next(0, 256);
                            uint aux1 = (uint)rng.Next(0, 1 << 28) | ((uint)rng.Next(1, 8) << 28);
                            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(gBase + 4, 4), aux1);
                        }
                        break;
                    }
                    case GgufType.IQ2_XS:
                    {
                        // 2B d + 32*2B qs + 8B scales (total 74)
                        for (int i = 0; i < 32; i++)
                        {
                            ushort gridIdx = (ushort)rng.Next(0, 512);
                            ushort signIdx = (ushort)rng.Next(0, 128);
                            ushort packed = (ushort)(gridIdx | (signIdx << 9));
                            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 2 + i * 2, 2), packed);
                        }
                        for (int i = 0; i < 8; i++)
                        {
                            data[offset + 66 + i] = (byte)(rng.Next(1, 8) | (rng.Next(1, 8) << 4));
                        }
                        break;
                    }
                    case GgufType.IQ2_S:
                    {
                        // 2B d + 32B q_lo + 32B signs + 8B qh + 8B scales (total 82)
                        for (int i = 0; i < 64; i++) data[offset + 2 + i] = (byte)rng.Next(0, 256);
                        for (int i = 0; i < 8; i++) data[offset + 66 + i] = (byte)rng.Next(0, 256);
                        for (int i = 0; i < 8; i++) data[offset + 74 + i] = (byte)(rng.Next(1, 8) | (rng.Next(1, 8) << 4));
                        break;
                    }
                    case GgufType.IQ3_XXS:
                    {
                        // 2B d + 64B grid indices + 32B aux32 (total 98)
                        for (int i = 0; i < 64; i++) data[offset + 2 + i] = (byte)rng.Next(0, 256);
                        for (int i = 0; i < 8; i++)
                        {
                            uint aux = (uint)rng.Next(0, 1 << 28) | ((uint)rng.Next(1, 8) << 28);
                            BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 66 + i * 4, 4), aux);
                        }
                        break;
                    }
                    case GgufType.IQ3_S:
                    {
                        // 2B d + 64B qs + 8B qh + 32B signs + 4B scales (total 110)
                        for (int i = 0; i < 104; i++) data[offset + 2 + i] = (byte)rng.Next(0, 256);
                        for (int i = 0; i < 4; i++) data[offset + 106 + i] = (byte)(rng.Next(1, 8) | (rng.Next(1, 8) << 4));
                        break;
                    }
                    case GgufType.IQ4_NL:
                    {
                        // 2B d + 16B qs (total 18)
                        for (int i = 0; i < 16; i++) data[offset + 2 + i] = (byte)rng.Next(0, 256);
                        break;
                    }
                    case GgufType.IQ4_XS:
                    {
                        // 2B d + 2B scales_h + 4B scales_l + 128B qs (total 136)
                        ushort scalesH = (ushort)rng.Next(0, 65536);
                        uint scalesL = (uint)rng.Next(0, int.MaxValue);
                        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 2, 2), scalesH);
                        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 4, 4), scalesL);
                        for (int i = 0; i < 128; i++) data[offset + 8 + i] = (byte)rng.Next(0, 256);
                        break;
                    }
                    default:
                        throw new NotSupportedException($"Unsupported synthetic IQ payload type: {type}");
                }
            }
        }

        return data;
    }

    public static byte[] CreateF32MatrixBytes(int rows, int cols, float scale = 0.05f, int seed = 505)
    {
        var rng = new Random(seed);
        byte[] bytes = new byte[rows * cols * sizeof(float)];
        var floats = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
        for (int i = 0; i < floats.Length; i++)
        {
            floats[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        }
        return bytes;
    }

    public static byte[] CreateOnesVectorBytes(int len, float value = 1.0f)
    {
        byte[] bytes = new byte[len * sizeof(float)];
        var floats = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
        floats.Fill(value);
        return bytes;
    }

    public static byte[] CreateQ8_0MatrixBytes(int rows, int cols, float scale = 0.05f, int seed = 606)
    {
        var rng = new Random(seed);
        int blocksPerRow = cols / 32;
        byte[] bytes = new byte[rows * blocksPerRow * 34];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = (r * blocksPerRow + b) * 34;
                float amax = 0f;
                float[] vals = new float[32];
                for (int i = 0; i < 32; i++)
                {
                    float v = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
                    vals[i] = v;
                    float av = MathF.Abs(v);
                    if (av > amax) amax = av;
                }
                Half d = (Half)(amax / 127.0f);
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(off, 2), d);
                float df = (float)d;
                float inv = df > 0f ? 1.0f / df : 0f;
                for (int i = 0; i < 32; i++)
                {
                    int q = Math.Clamp((int)MathF.Round(vals[i] * inv), -127, 127);
                    bytes[off + 2 + i] = unchecked((byte)(sbyte)q);
                }
            }
        }

        return bytes;
    }
}

internal sealed class SyntheticGgufBuilder
{
    private const uint GgufMagic = 0x46554747; // "GGUF" LE
    private const uint GgufVersion = 3;
    private const int DefaultAlignment = 32;

    private sealed record KvEntry(string Key, Action<BinaryWriter> WriteValue);
    private sealed record TensorEntry(string Name, ulong[] Dimensions, GgufType Type, byte[] Data);

    private readonly List<KvEntry> _metadata = new();
    private readonly List<TensorEntry> _tensors = new();

    public void AddString(string key, string value)
    {
        _metadata.Add(new KvEntry(key, w =>
        {
            w.Write(8u); // STRING
            WriteGgufString(w, value);
        }));
    }

    public void AddUInt32(string key, uint value)
    {
        _metadata.Add(new KvEntry(key, w =>
        {
            w.Write(4u); // UINT32
            w.Write(value);
        }));
    }

    public void AddFloat32(string key, float value)
    {
        _metadata.Add(new KvEntry(key, w =>
        {
            w.Write(6u); // FLOAT32
            w.Write(value);
        }));
    }

    public void AddBool(string key, bool value)
    {
        _metadata.Add(new KvEntry(key, w =>
        {
            w.Write(7u); // BOOL
            w.Write((byte)(value ? 1 : 0));
        }));
    }

    public void AddStringArray(string key, IReadOnlyList<string> values)
    {
        _metadata.Add(new KvEntry(key, w =>
        {
            w.Write(9u); // ARRAY
            w.Write(8u); // STRING element
            w.Write((ulong)values.Count);
            for (int i = 0; i < values.Count; i++)
                WriteGgufString(w, values[i]);
        }));
    }

    public void AddTensor1D(string name, int len, GgufType type, byte[] data)
        => _tensors.Add(new TensorEntry(name, new[] { (ulong)len }, type, data));

    public void AddTensor2D(string name, int rows, int cols, GgufType type, byte[] data)
        => _tensors.Add(new TensorEntry(name, new[] { (ulong)cols, (ulong)rows }, type, data));

    public void WriteToFile(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(GgufMagic);
        w.Write(GgufVersion);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)_metadata.Count);

        foreach (var kv in _metadata)
        {
            WriteGgufString(w, kv.Key);
            kv.WriteValue(w);
        }

        ulong currentDataOffset = 0;
        foreach (var t in _tensors)
        {
            WriteGgufString(w, t.Name);
            w.Write((uint)t.Dimensions.Length);
            foreach (ulong d in t.Dimensions) w.Write(d);
            w.Write((uint)t.Type);
            w.Write(currentDataOffset);
            ulong paddedLen = AlignUp((ulong)t.Data.Length, DefaultAlignment);
            currentDataOffset += paddedLen;
        }

        long headerEnd = fs.Position;
        long dataStart = (long)AlignUp((ulong)headerEnd, DefaultAlignment);
        WriteZeros(w, (int)(dataStart - headerEnd));

        foreach (var t in _tensors)
        {
            w.Write(t.Data);
            int pad = (int)(AlignUp((ulong)t.Data.Length, DefaultAlignment) - (ulong)t.Data.Length);
            if (pad > 0) WriteZeros(w, pad);
        }
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)utf8.Length);
        w.Write(utf8);
    }

    private static void WriteZeros(BinaryWriter w, int count)
    {
        for (int i = 0; i < count; i++) w.Write((byte)0);
    }

    private static ulong AlignUp(ulong value, ulong alignment)
        => (value + alignment - 1) / alignment * alignment;

    public static string BuildTinyQwen2Model(
        string path,
        int dim = 64,
        int ffnDim = 128,
        int nLayers = 2,
        int nHeads = 4,
        int nHeadsKv = 2,
        int vocabSize = 32,
        int maxSeqLen = 64,
        GgufType linearType = GgufType.F32,
        int seed = 77)
    {
        var b = new SyntheticGgufBuilder();
        b.AddString("general.architecture", "qwen2");
        b.AddString("general.name", "synthetic-qwen2-e2e");
        b.AddUInt32("qwen2.context_length", (uint)maxSeqLen);
        b.AddUInt32("qwen2.embedding_length", (uint)dim);
        b.AddUInt32("qwen2.feed_forward_length", (uint)ffnDim);
        b.AddUInt32("qwen2.block_count", (uint)nLayers);
        b.AddUInt32("qwen2.attention.head_count", (uint)nHeads);
        b.AddUInt32("qwen2.attention.head_count_kv", (uint)nHeadsKv);
        b.AddFloat32("qwen2.attention.layer_norm_rms_epsilon", 1e-5f);
        b.AddFloat32("qwen2.rope.freq_base", 10000.0f);

        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++) tokens.Add($"tok{i}");
        b.AddString("tokenizer.ggml.model", "gpt2");
        b.AddStringArray("tokenizer.ggml.tokens", tokens);
        b.AddStringArray("tokenizer.ggml.merges", Array.Empty<string>());
        b.AddUInt32("tokenizer.ggml.bos_token_id", 0);
        b.AddUInt32("tokenizer.ggml.eos_token_id", 1);

        int headDim = dim / nHeads;
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        b.AddTensor2D("token_embd.weight", vocabSize, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(vocabSize, dim, 0.1f, seed + 1));
        b.AddTensor1D("output_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
        b.AddTensor2D("output.weight", vocabSize, dim, GgufType.F32, FxQuantData.CreateF32MatrixBytes(vocabSize, dim, 0.1f, seed + 2));

        byte[] MakeMatrix(int rows, int cols, int s) => linearType switch
        {
            GgufType.F32 => FxQuantData.CreateF32MatrixBytes(rows, cols, 0.08f, s),
            GgufType.Q8_0 => FxQuantData.CreateQ8_0MatrixBytes(rows, cols, 0.08f, s),
            GgufType.IQ2_XXS or GgufType.IQ2_XS or GgufType.IQ2_S or GgufType.IQ3_XXS or GgufType.IQ3_S or GgufType.IQ4_NL or GgufType.IQ4_XS
                => FxQuantData.CreateValidIqPayload(linearType, rows, cols, s),
            _ => throw new NotSupportedException($"Synthetic linear type {linearType} not configured.")
        };

        for (int l = 0; l < nLayers; l++)
        {
            int ls = seed + (l + 1) * 100;
            b.AddTensor1D($"blk.{l}.attn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            b.AddTensor2D($"blk.{l}.attn_q.weight", qDim, dim, linearType, MakeMatrix(qDim, dim, ls + 1));
            b.AddTensor2D($"blk.{l}.attn_k.weight", kvDim, dim, linearType, MakeMatrix(kvDim, dim, ls + 2));
            b.AddTensor2D($"blk.{l}.attn_v.weight", kvDim, dim, linearType, MakeMatrix(kvDim, dim, ls + 3));
            b.AddTensor2D($"blk.{l}.attn_output.weight", dim, qDim, linearType, MakeMatrix(dim, qDim, ls + 4));
            b.AddTensor1D($"blk.{l}.ffn_norm.weight", dim, GgufType.F32, FxQuantData.CreateOnesVectorBytes(dim, 1.0f));
            b.AddTensor2D($"blk.{l}.ffn_gate.weight", ffnDim, dim, linearType, MakeMatrix(ffnDim, dim, ls + 5));
            b.AddTensor2D($"blk.{l}.ffn_up.weight", ffnDim, dim, linearType, MakeMatrix(ffnDim, dim, ls + 6));
            b.AddTensor2D($"blk.{l}.ffn_down.weight", dim, ffnDim, linearType, MakeMatrix(dim, ffnDim, ls + 7));
        }

        b.WriteToFile(path);
        return path;
    }

    public static void WriteInvalidMagicGguf(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(fs);
        w.Write(0x44414221u); // "!BAD"
        w.Write(3u);
        w.Write(0UL);
        w.Write(0UL);
    }

    public static void WriteTruncatedHeaderGguf(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(fs);
        w.Write(GgufMagic);
        w.Write(3u);
        w.Write(1UL);
        // Truncated before metadata count and tensor info
    }

    public static void WriteTruncatedTensorPayloadGguf(string path)
    {
        var b = new SyntheticGgufBuilder();
        b.AddString("general.architecture", "qwen2");
        b.AddTensor1D("token_embd.weight", 64, GgufType.F32, FxQuantData.CreateOnesVectorBytes(64, 1.0f));
        b.WriteToFile(path);

        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite);
        fs.SetLength(Math.Max(0, fs.Length - 64));
    }
}

internal sealed unsafe class SyntheticMoeFixture : IDisposable
{
    private readonly List<IntPtr> _allocs = new();
    public LayerWeights[] Layers { get; }
    public float[] LmHead { get; }
    public int Dim { get; }
    public int ExpertFfnDim { get; }
    public int NumExperts { get; }
    public int TopK { get; }
    public int VocabSize { get; }

    public SyntheticMoeFixture(
        int numLayers = 2,
        int numExperts = 8,
        int topK = 2,
        int dim = 32,
        int expertFfnDim = 64,
        int vocabSize = 32,
        GgufType expertType = GgufType.F32,
        int seed = 707,
        bool analyticalDiagonal = false)
    {
        Dim = dim;
        ExpertFfnDim = expertFfnDim;
        NumExperts = numExperts;
        TopK = topK;
        VocabSize = vocabSize;

        Layers = new LayerWeights[numLayers];
        var rng = new Random(seed);

        for (int l = 0; l < numLayers; l++)
        {
            long gateUpRowBytes = GgufTypes.GetRowBytes(expertType, dim);
            long downRowBytes = GgufTypes.GetRowBytes(expertType, expertFfnDim);

            nuint gateBytes = (nuint)((long)numExperts * expertFfnDim * gateUpRowBytes);
            nuint downBytes = (nuint)((long)numExperts * dim * downRowBytes);

            byte* pGate;
            byte* pUp;
            byte* pDown;
            float* pRouter = (float*)AllocBytes((nuint)(numExperts * dim * sizeof(float)));

            if (analyticalDiagonal)
            {
                if (expertType != GgufType.F32)
                    throw new NotSupportedException("Analytical diagonal mode requires GgufType.F32.");

                pGate = (byte*)AllocBytes(gateBytes);
                pUp = (byte*)AllocBytes(gateBytes);
                pDown = (byte*)AllocBytes(downBytes);

                float* fg = (float*)pGate;
                float* fu = (float*)pUp;
                float* fd = (float*)pDown;

                int diagLen = Math.Min(dim, expertFfnDim);
                for (int e = 0; e < numExperts; e++)
                {
                    float scale = (e + 1) * 0.125f;
                    long gateBase = (long)e * expertFfnDim * dim;
                    long downBase = (long)e * dim * expertFfnDim;
                    for (int i = 0; i < diagLen; i++)
                    {
                        fg[gateBase + i * dim + i] = 1.0f;
                        fu[gateBase + i * dim + i] = 1.0f;
                        fd[downBase + i * expertFfnDim + i] = scale;
                    }

                    // Distinct deterministic router scores favoring higher-index experts for positive ramp inputs
                    for (int d = 0; d < dim; d++)
                    {
                        pRouter[e * dim + d] = (e + 1) * 0.02f;
                    }
                }
            }
            else
            {
                pGate = AllocAndFillWeights(gateBytes, numExperts * expertFfnDim, dim, expertType, rng);
                pUp = AllocAndFillWeights(gateBytes, numExperts * expertFfnDim, dim, expertType, rng);
                pDown = AllocAndFillWeights(downBytes, numExperts * dim, expertFfnDim, expertType, rng);

                for (int i = 0; i < numExperts * dim; i++)
                    pRouter[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.05f;
            }

            float* pNorm = (float*)AllocBytes((nuint)(dim * sizeof(float)));
            for (int i = 0; i < dim; i++) pNorm[i] = 1.0f;

            Layers[l] = new LayerWeights
            {
                IsMoe = true,
                AttnOutWeight = null,
                AttnOutType = GgufType.F32,
                FfnNormWeight = pNorm,
                FfnGateInpWeight = pRouter,
                FfnGateExpsWeight = pGate,
                FfnUpExpsWeight = pUp,
                FfnDownExpsWeight = pDown,
                FfnGateExpsType = expertType,
                FfnUpExpsType = expertType,
                FfnDownExpsType = expertType
            };
        }

        LmHead = new float[vocabSize * dim];
        for (int i = 0; i < LmHead.Length; i++)
            LmHead[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.05f;
    }

    public static float[] ComputeAnalyticalDiagonalLayerOutput(
        LayerWeights layer,
        ReadOnlySpan<float> normXBatch,
        int batchSize,
        int dim,
        int numExperts,
        int topK,
        bool normTopK)
    {
        float[] expected = new float[batchSize * dim];
        int[] indices = new int[topK];
        float[] weights = new float[topK];

        fixed (float* pX = normXBatch)
        fixed (int* pIdx = indices)
        fixed (float* pW = weights)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* x = pX + (long)t * dim;
                QuantKernels.RouterTopK(x, layer.FfnGateInpWeight, layer.FfnGateInpBias, dim, numExperts, topK, pIdx, pW, normTopK);

                double weightedScaleSum = 0.0;
                for (int k = 0; k < topK; k++)
                {
                    int e = indices[k];
                    float w = weights[k];
                    weightedScaleSum += w * ((e + 1) * 0.125f);
                }

                for (int d = 0; d < dim; d++)
                {
                    float xd = x[d];
                    float silu = xd / (1.0f + MathF.Exp(-xd));
                    expected[t * dim + d] = (float)(weightedScaleSum * silu * xd);
                }
            }
        }

        return expected;
    }

    private IntPtr AllocBytes(nuint bytes)
    {
        void* ptr = NativeMemory.AllocZeroed(bytes);
        _allocs.Add((IntPtr)ptr);
        return (IntPtr)ptr;
    }

    private byte* AllocAndFillWeights(nuint totalBytes, int rows, int cols, GgufType type, Random rng)
    {
        byte* ptr = (byte*)AllocBytes(totalBytes);
        if (type == GgufType.F32)
        {
            float* fp = (float*)ptr;
            int count = rows * cols;
            for (int i = 0; i < count; i++)
                fp[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.04f;
        }
        else if (type == GgufType.Q8_0)
        {
            byte[] q = FxQuantData.CreateQ8_0MatrixBytes(rows, cols, 0.04f, rng.Next());
            Marshal.Copy(q, 0, (IntPtr)ptr, q.Length);
        }
        else
        {
            throw new NotSupportedException($"SyntheticMoE expert type {type} not configured.");
        }
        return ptr;
    }

    public void Dispose()
    {
        foreach (var ptr in _allocs)
        {
            NativeMemory.Free((void*)ptr);
        }
        _allocs.Clear();
    }
}

internal static class Fx2Routing
{
    public static (int[] Indices, float[] Weights) BuildRankNormalizedBatch(int batchSize, int topK, int numExperts, int startOffset = 0)
    {
        int[] indices = new int[batchSize * topK];
        float[] weights = new float[batchSize * topK];

        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0f;
            for (int k = 0; k < topK; k++)
            {
                indices[b * topK + k] = (startOffset + b + k) % numExperts;
                float w = MathF.Pow(0.5f, k + 1);
                weights[b * topK + k] = w;
                sum += w;
            }
            for (int k = 0; k < topK; k++)
            {
                weights[b * topK + k] /= sum;
            }
        }

        return (indices, weights);
    }

    public static (int[] Indices, float[] Weights) BuildSkewedRoutingBatch(int batchSize, int topK, int hotExpert0 = 0, int hotExpert1 = 1)
    {
        int[] indices = new int[batchSize * topK];
        float[] weights = new float[batchSize * topK];

        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0f;
            for (int k = 0; k < topK; k++)
            {
                indices[b * topK + k] = (k % 2 == 0) ? hotExpert0 : hotExpert1;
                float w = (k == 0) ? 0.8f : 0.2f / Math.Max(1, topK - 1);
                weights[b * topK + k] = w;
                sum += w;
            }
            for (int k = 0; k < topK; k++) weights[b * topK + k] /= sum;
        }

        return (indices, weights);
    }
}

internal sealed class SyntheticSpeculativeTarget : ISpeculativeTarget, IHiddenStateSource
{
    private readonly TempDirScope _tmpScope;
    private readonly GgufFile _gguf;
    private readonly int _vocabSize;
    private readonly int _dim;
    private readonly int _maxSeqLen;
    private readonly float[] _lastHiddenState;
    private readonly Dictionary<int, float[]> _history = new();
    private readonly float[] _biasLogits;
    private readonly float[] _logitsScratch;
    private readonly float _temperatureBias;
    private readonly bool _invertLogits;
    private readonly Sampler _sampler = new(1337);
    private bool _trackHistory;
    private int _lastToken;
    private int _currentPos;

    public BpeTokenizer Tokenizer { get; }
    public int VocabSize => _vocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public bool SupportsLogitVerification => true;
    public bool IsRingBufferEnabled => false;
    public bool CanOverwriteKvFromPrefix => true;
    public long StateSnapshotBytes => 0;
    public bool HasUnsnapshottableRecurrentState => false;
    public int HiddenDim => _dim;

    public SyntheticSpeculativeTarget(int vocabSize = 32, int dim = 32, int maxSeqLen = 256, float perturbation = 0.0f, bool invertLogits = false)
    {
        _vocabSize = vocabSize;
        _dim = dim;
        _maxSeqLen = maxSeqLen;
        _lastHiddenState = new float[dim];
        _biasLogits = new float[vocabSize];
        _logitsScratch = new float[vocabSize];
        _temperatureBias = perturbation;
        _invertLogits = invertLogits;
        for (int v = 0; v < vocabSize; v++)
        {
            _biasLogits[v] = -0.1f * (v % 8);
        }

        _tmpScope = new TempDirScope();
        var builder = new SyntheticGgufBuilder();
        var tokens = new List<string>(vocabSize) { "<bos>", "<eos>" };
        for (int i = 2; i < vocabSize; i++)
        {
            tokens.Add(i < 28 ? ((char)('a' + i - 2)).ToString() : $"t{i}");
        }
        builder.AddString("tokenizer.ggml.model", "gpt2");
        builder.AddStringArray("tokenizer.ggml.tokens", tokens);
        builder.AddStringArray("tokenizer.ggml.merges", Array.Empty<string>());
        builder.AddUInt32("tokenizer.ggml.bos_token_id", 0);
        builder.AddUInt32("tokenizer.ggml.eos_token_id", 1);
        string tokPath = _tmpScope.GetFilePath("tok_only.gguf");
        builder.WriteToFile(tokPath);
        _gguf = GgufFile.Open(tokPath);
        Tokenizer = new BpeTokenizer(_gguf);
    }

    public void ResetKvCache()
    {
        _currentPos = 0;
        _lastToken = 0;
        Array.Clear(_lastHiddenState);
        _history.Clear();
    }

    public void Prefill(ReadOnlySpan<int> tokens)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            ForwardToken(tokens[i], i, computeLogits: i == tokens.Length - 1);
        }
    }

    public void ForwardToken(int token, int pos, bool computeLogits = true)
    {
        _lastToken = token;
        _currentPos = pos;
        for (int d = 0; d < _dim; d++)
        {
            _lastHiddenState[d] = MathF.Sin((token + 1) * 0.13f + d * 0.07f + pos * 0.01f);
        }
        if (_trackHistory)
        {
            _history[pos] = (float[])_lastHiddenState.Clone();
        }
    }

    public void CopyCurrentLogits(Span<float> destination)
    {
        ComputeLogitsForToken(_lastToken, destination);
    }

    public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        CopyCurrentLogits(_logitsScratch);
        return _sampler.Sample(_logitsScratch, options, recentTokens);
    }

    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        for (int t = 0; t < tokens.Length; t++)
        {
            ForwardToken(tokens[t], startPos + t, computeLogits: true);
            ComputeLogitsForToken(tokens[t], _logitsScratch);
            predictedTokens[t] = _sampler.Sample(_logitsScratch, SamplingOptions.Greedy);
        }
    }

    public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
    {
        for (int t = 0; t < tokens.Length; t++)
        {
            ForwardToken(tokens[t], startPos + t, computeLogits: true);
            ComputeLogitsForToken(tokens[t], batchLogitsOut.Slice(t * _vocabSize, _vocabSize));
        }
    }

    public ReadOnlySpan<float> GetLastHiddenState() => _lastHiddenState;

    public void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        int len = Math.Min(destination.Length, _dim);
        for (int d = 0; d < len; d++)
        {
            destination[d] = MathF.Cos((tokenId + 1) * 0.11f + d * 0.05f);
        }
    }

    public void EnablePositionHistory() => _trackHistory = true;

    public bool TryGetHiddenStateAt(int position, Span<float> destination)
    {
        if (!_history.TryGetValue(position, out var vec) || destination.Length < _dim)
            return false;
        vec.CopyTo(destination);
        return true;
    }

    private void ComputeLogitsForToken(int seedToken, Span<float> dst)
    {
        int favored = 2 + (Math.Abs(seedToken + 1) % Math.Max(1, _vocabSize - 2));
        for (int v = 0; v < _vocabSize; v++)
        {
            float baseLogit = _biasLogits[v] + (v == favored ? 4.0f : 0.0f);
            float val = baseLogit + _temperatureBias * ((v % 3) - 1);
            dst[v] = _invertLogits ? -val : val;
        }
    }

    public void Dispose()
    {
        _gguf.Dispose();
        _tmpScope.Dispose();
    }
}

internal static class Fx4Kl
{
    public static double ComputeAnalyticalKl(ReadOnlySpan<float> pLogits, ReadOnlySpan<float> qLogits, float temperature = 1.0f)
    {
        double[] p = Softmax(pLogits, temperature);
        double[] q = Softmax(qLogits, temperature);
        double kl = 0.0;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 1e-12)
            {
                double qi = Math.Max(q[i], 1e-12);
                kl += p[i] * Math.Log(p[i] / qi);
            }
        }
        return kl;
    }

    private static double[] Softmax(ReadOnlySpan<float> logits, float temperature)
    {
        double invT = 1.0 / Math.Max(temperature, 1e-5f);
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = logits[i] * invT;
            if (v > max) max = v;
        }
        double[] probs = new double[logits.Length];
        double sum = 0.0;
        for (int i = 0; i < logits.Length; i++)
        {
            double e = Math.Exp(logits[i] * invT - max);
            probs[i] = e;
            sum += e;
        }
        for (int i = 0; i < logits.Length; i++) probs[i] /= sum;
        return probs;
    }

    public static float[] GenerateDeterministicLogits(int vocabSize, int seed, float amplitude = 3.0f)
    {
        var rng = new Random(seed);
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            logits[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * amplitude;
        }
        return logits;
    }
}
