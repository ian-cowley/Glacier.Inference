namespace Glacier.Inference.Model;

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// Post-Epilogue nonlinear/linear low-rank adapter for Inkling-2-Mini's final hidden state -> lm_head logits:
/// rawLogits += (Alpha / Rank) * B^T @ act(A @ hidden), applied immediately before FinalLogitSoftcapping.
/// Matrix A is stored row-major [Rank, EmbdDim] and B is stored row-major [Rank, VocabSize] for contiguous SIMD access.
/// </summary>
public sealed class Inkling2HeadLoraAdapter
{
    public const uint Magic = 0x324B4E49; // "INK2"
    public const uint FormatVersion = 1;

    public int EmbdDim { get; }
    public int VocabSize { get; }
    public int Rank { get; }
    public float Alpha { get; }
    public bool UseSiLU { get; }
    public float Scaling => Rank > 0 ? Alpha / Rank : 0f;

    /// <summary>Row-major [Rank, EmbdDim].</summary>
    public float[] A { get; }

    /// <summary>Row-major [Rank, VocabSize]. Zero-initialized for exact identity at step 0.</summary>
    public float[] B { get; }

    public Inkling2HeadLoraAdapter(int embdDim, int vocabSize, int rank, float alpha, bool useSiLU = true, int seed = 42)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embdDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);
        if (!float.IsFinite(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be finite.");

        EmbdDim = embdDim;
        VocabSize = vocabSize;
        Rank = rank;
        Alpha = alpha;
        UseSiLU = useSiLU;

        A = new float[rank * embdDim];
        B = new float[rank * vocabSize];

        // Kaiming uniform initialization for A; B stays strictly zero so initial delta == 0
        var rng = new Random(seed);
        float bound = MathF.Sqrt(6.0f / embdDim) * 0.5f;
        for (int i = 0; i < A.Length; i++)
            A[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * bound;
    }

    private Inkling2HeadLoraAdapter(int embdDim, int vocabSize, int rank, float alpha, bool useSiLU, float[] a, float[] b)
    {
        EmbdDim = embdDim;
        VocabSize = vocabSize;
        Rank = rank;
        Alpha = alpha;
        UseSiLU = useSiLU;
        A = a;
        B = b;
    }

    /// <summary>
    /// Applies (Alpha / Rank) * B^T @ act(A @ hidden) directly onto rawLogits (before FinalLogitSoftcapping).
    /// Zero heap allocations when Rank &lt;= 256.
    /// </summary>
    public unsafe void ApplyDelta(ReadOnlySpan<float> hidden, Span<float> rawLogits)
    {
        int embd = Math.Min(EmbdDim, hidden.Length);
        int vocab = Math.Min(VocabSize, rawLogits.Length);
        if (embd <= 0 || vocab <= 0 || Rank <= 0) return;

        Span<float> act = Rank <= 256 ? stackalloc float[Rank] : new float[Rank];
        float scale = Scaling;

        fixed (float* pH = hidden)
        fixed (float* pA = A)
        fixed (float* pB = B)
        fixed (float* pOut = rawLogits)
        {
            for (int r = 0; r < Rank; r++)
            {
                float u = DotSimd(pA + r * EmbdDim, pH, embd);
                if (UseSiLU)
                {
                    float sig = 1.0f / (1.0f + MathF.Exp(-u));
                    act[r] = u * sig * scale;
                }
                else
                {
                    act[r] = u * scale;
                }
            }

            for (int r = 0; r < Rank; r++)
            {
                float w = act[r];
                if (w == 0f) continue;
                AxpySimd(pOut, pB + r * VocabSize, w, vocab);
            }
        }
    }

    /// <summary>Applies Inkling-2 FinalLogitSoftcapping in-place: cap * tanh(x / cap).</summary>
    public static void ApplySoftcapInPlace(Span<float> logits, float cap)
    {
        if (cap <= 0f) return;
        float invCap = 1.0f / cap;
        for (int i = 0; i < logits.Length; i++)
            logits[i] = cap * MathF.Tanh(logits[i] * invCap);
    }

    /// <summary>Saves the adapter to a compact binary file.</summary>
    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);
        bw.Write(Magic);
        bw.Write(FormatVersion);
        bw.Write(EmbdDim);
        bw.Write(VocabSize);
        bw.Write(Rank);
        bw.Write(Alpha);
        bw.Write((byte)(UseSiLU ? 1 : 0));
        bw.Flush();

        fs.Write(MemoryMarshal.AsBytes(A.AsSpan()));
        fs.Write(MemoryMarshal.AsBytes(B.AsSpan()));
    }

    /// <summary>Loads an Inkling-2 head LoRA adapter from a binary file.</summary>
    public static Inkling2HeadLoraAdapter Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Inkling-2 head LoRA adapter not found: {path}");

        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        uint magic = br.ReadUInt32();
        if (magic != Magic)
            throw new InvalidDataException($"Invalid Inkling-2 LoRA magic: 0x{magic:X8}");
        uint version = br.ReadUInt32();
        if (version != FormatVersion)
            throw new InvalidDataException($"Unsupported Inkling-2 LoRA version: {version}");

        int embdDim = br.ReadInt32();
        int vocabSize = br.ReadInt32();
        int rank = br.ReadInt32();
        float alpha = br.ReadSingle();
        bool useSiLU = br.ReadByte() != 0;

        if (embdDim <= 0 || vocabSize <= 0 || rank <= 0 || !float.IsFinite(alpha))
            throw new InvalidDataException($"Corrupt Inkling-2 LoRA header: embd={embdDim}, vocab={vocabSize}, rank={rank}, alpha={alpha}");

        float[] a = new float[checked(rank * embdDim)];
        float[] b = new float[checked(rank * vocabSize)];
        fs.ReadExactly(MemoryMarshal.AsBytes(a.AsSpan()));
        fs.ReadExactly(MemoryMarshal.AsBytes(b.AsSpan()));

        return new Inkling2HeadLoraAdapter(embdDim, vocabSize, rank, alpha, useSiLU, a, b);
    }

    public static unsafe float DotSimd(float* a, float* b, int len)
    {
        int i = 0;
        float sum = 0f;
        if (Vector512.IsHardwareAccelerated && len >= 16)
        {
            Vector512<float> acc = Vector512<float>.Zero;
            for (; i <= len - 16; i += 16)
                acc = Vector512.Add(acc, Vector512.Multiply(Vector512.Load(a + i), Vector512.Load(b + i)));
            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated && len >= 8)
        {
            Vector256<float> acc = Vector256<float>.Zero;
            for (; i <= len - 8; i += 8)
                acc = Vector256.Add(acc, Vector256.Multiply(Vector256.Load(a + i), Vector256.Load(b + i)));
            sum = Vector256.Sum(acc);
        }
        for (; i < len; i++)
            sum += a[i] * b[i];
        return sum;
    }

    public static unsafe void AxpySimd(float* dst, float* src, float scale, int len)
    {
        int i = 0;
        if (Vector512.IsHardwareAccelerated && len >= 16)
        {
            Vector512<float> vs = Vector512.Create(scale);
            for (; i <= len - 16; i += 16)
            {
                Vector512<float> vd = Vector512.Load(dst + i);
                Vector512<float> vx = Vector512.Load(src + i);
                Vector512.Store(Vector512.Add(vd, Vector512.Multiply(vx, vs)), dst + i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= 8)
        {
            Vector256<float> vs = Vector256.Create(scale);
            for (; i <= len - 8; i += 8)
            {
                Vector256<float> vd = Vector256.Load(dst + i);
                Vector256<float> vx = Vector256.Load(src + i);
                Vector256.Store(Vector256.Add(vd, Vector256.Multiply(vx, vs)), dst + i);
            }
        }
        for (; i < len; i++)
            dst[i] += src[i] * scale;
    }
}
