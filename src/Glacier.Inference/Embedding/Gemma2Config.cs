namespace Glacier.Inference.Embedding;

using System;
using System.Runtime.CompilerServices;
using Glacier.Inference.Gguf;

/// <summary>Hyper-parameters of a <c>gemma-embedding2</c> GGUF, validated against tensor shapes.</summary>
public sealed class Gemma2Config
{
    public int Hidden { get; init; }
    public int Layers { get; init; }
    public int Heads { get; init; }
    public int FfnDim { get; init; }
    public int PleDim { get; init; }
    public int OutDim { get; init; }
    public int Vocab { get; init; }
    public float Eps { get; init; }
    public float RopeBase { get; init; }
    public float RopeBaseSwa { get; init; }
    /// <summary>Sliding layers attend to |i-j| &lt;= HalfWindow (symmetric window = sliding_window / 2 each side).</summary>
    public int HalfWindow { get; init; }
    public required bool[] IsSwa { get; init; }
    public required int[] HeadDim { get; init; }
    public required int[] KvHeads { get; init; }

    public static Gemma2Config FromGguf(GgufFile gguf)
    {
        const string a = EmbeddingGemma2Model.ArchitectureName;
        if (gguf.GetMetadataBool($"{a}.attention.causal", false))
            throw new NotSupportedException("Causal gemma-embedding2 variants are not supported.");

        int layers = (int)gguf.GetMetadataUInt32($"{a}.block_count");
        int headDimFull = (int)gguf.GetMetadataUInt32($"{a}.attention.key_length");
        int headDimSwa = (int)gguf.GetMetadataUInt32($"{a}.attention.key_length_swa", (uint)headDimFull);
        var swa = gguf.SlidingWindowPattern ?? throw new InvalidOperationException("Missing sliding_window_pattern.");
        var kv = gguf.HeadCountKvPattern ?? throw new InvalidOperationException("Missing head_count_kv array.");
        if (swa.Length < layers || kv.Length < layers) throw new InvalidOperationException("Per-layer metadata arrays are shorter than block_count.");

        var cfg = new Gemma2Config
        {
            Hidden = (int)gguf.GetMetadataUInt32($"{a}.embedding_length"),
            Layers = layers,
            Heads = (int)gguf.GetMetadataUInt32($"{a}.attention.head_count"),
            FfnDim = (int)gguf.GetMetadataUInt32($"{a}.feed_forward_length"),
            PleDim = (int)gguf.GetMetadataUInt32($"{a}.embedding_length_per_layer_input"),
            Eps = gguf.GetMetadataSingle($"{a}.attention.layer_norm_rms_epsilon", 1e-6f),
            RopeBase = gguf.GetMetadataSingle($"{a}.rope.freq_base", 1_000_000f),
            RopeBaseSwa = gguf.GetMetadataSingle($"{a}.rope.freq_base_swa", 10_000f),
            HalfWindow = (int)gguf.GetMetadataUInt32($"{a}.attention.sliding_window", 1024) / 2,
            IsSwa = swa[..layers],
            HeadDim = new int[layers],
            KvHeads = kv[..layers],
            OutDim = 0,
            Vocab = 0,
        };
        for (int l = 0; l < layers; l++) cfg.HeadDim[l] = swa[l] ? headDimSwa : headDimFull;

        int outDim = (int)gguf.GetMetadataUInt32($"{a}.embedding_length_out", (uint)cfg.Hidden);
        int vocab = (int)gguf.Tensors["token_embd.weight"].Dimensions[1];
        return new Gemma2Config
        {
            Hidden = cfg.Hidden, Layers = cfg.Layers, Heads = cfg.Heads, FfnDim = cfg.FfnDim, PleDim = cfg.PleDim,
            Eps = cfg.Eps, RopeBase = cfg.RopeBase, RopeBaseSwa = cfg.RopeBaseSwa, HalfWindow = cfg.HalfWindow,
            IsSwa = cfg.IsSwa, HeadDim = cfg.HeadDim, KvHeads = cfg.KvHeads, OutDim = outDim, Vocab = vocab,
        };
    }
}

/// <summary>Raw GGUF tensor access shared by backends (GGML type ids: 0=F32, 1=F16, 8=Q8_0, 30=BF16).</summary>
internal static unsafe class Gemma2Tensors
{
    public const uint F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30;

    public static GgufTensorInfo Require(GgufFile gguf, string name, params ulong[] expectedDims)
    {
        if (!gguf.Tensors.TryGetValue(name, out var info))
            throw new InvalidOperationException($"Tensor '{name}' not found in GGUF.");
        if (info.Dimensions.Length < expectedDims.Length)
            throw new InvalidOperationException($"Tensor '{name}' has unexpected rank {info.Dimensions.Length}.");
        for (int i = 0; i < expectedDims.Length; i++)
        {
            if (info.Dimensions[i] != expectedDims[i])
                throw new InvalidOperationException($"Tensor '{name}' dim {i} is {info.Dimensions[i]}, expected {expectedDims[i]}.");
        }
        return info;
    }

    public static long RowBytes(uint type, int count) => type switch
    {
        F32 => (long)count * 4,
        F16 or BF16 => (long)count * 2,
        Q8_0 => (long)count / 32 * 34,
        _ => throw new NotSupportedException($"GGML tensor type {type} is not supported for gemma-embedding2."),
    };

    public static void DequantRow(byte* baseData, uint type, int row, int count, float* dst)
    {
        byte* src = baseData + row * RowBytes(type, count);
        switch (type)
        {
            case F32:
                Buffer.MemoryCopy(src, dst, count * 4L, count * 4L);
                break;
            case F16:
                for (int i = 0; i < count; i++) dst[i] = (float)Unsafe.ReadUnaligned<Half>(src + i * 2);
                break;
            case BF16:
                for (int i = 0; i < count; i++)
                    dst[i] = BitConverter.Int32BitsToSingle(Unsafe.ReadUnaligned<ushort>(src + i * 2) << 16);
                break;
            case Q8_0:
                for (int b = 0; b < count / 32; b++)
                {
                    byte* blk = src + b * 34;
                    float scale = (float)Unsafe.ReadUnaligned<Half>(blk);
                    sbyte* q = (sbyte*)(blk + 2);
                    for (int i = 0; i < 32; i++) dst[b * 32 + i] = q[i] * scale;
                }
                break;
            default:
                throw new NotSupportedException($"GGML tensor type {type} is not supported for gemma-embedding2.");
        }
    }

    public static float[] LoadVector(GgufFile gguf, string name, int length)
    {
        var info = Require(gguf, name, (ulong)length);
        var result = new float[length];
        fixed (float* p = result) DequantRow(gguf.GetTensorPointer(info), (uint)info.Type, 0, length, p);
        return result;
    }
}

/// <summary>Compute backend executing the encoder stack. Implementations must be thread-safe for concurrent <see cref="ForwardHidden"/> calls or serialise internally.</summary>
public interface IGemma2Backend : IDisposable
{
    /// <summary>Human-readable device / engine description.</summary>
    string Name { get; }

    /// <summary>Final <c>output_norm</c>'d hidden states, shape [tokens, hidden], row-major.</summary>
    float[] ForwardHidden(ReadOnlySpan<int> tokens);

    /// <summary>Mean over tokens of the final-norm'd hidden states ([hidden]).</summary>
    float[] ForwardPooled(ReadOnlySpan<int> tokens);

    /// <summary>Like <see cref="ForwardPooled"/> but from caller-built input embeddings [n, hidden] (already scaled; used to inject vision/audio soft tokens).</summary>
    float[] ForwardPooledEmbeds(float[] inputEmbeddings, int tokenCount);
}

