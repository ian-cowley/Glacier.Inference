namespace Glacier.Inference.Gguf;

using System;

/// <summary>
/// GGUF metadata value types as defined in the GGUF v2/v3 specification.
/// </summary>
public enum GgufValueType : uint
{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12
}

/// <summary>
/// GGML tensor data quantization types.
/// </summary>
public enum GgufType : uint
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    Q2_0 = 36,
    MXFP4 = 39,
    NVFP4 = 40,
    GgmlQ2_0 = 42,
    F8_E4M3 = 43,
    TRL1 = 44,
    TRL2 = 45,
    TRL3 = 46,
    TRL4 = 47,
    TRL5 = 48,
    TRL6 = 49,
    Ternary_1_58 = 100,
    PQ2_0 = 142
}

/// <summary>
/// Helper utilities for GGUF types and block strides.
/// </summary>
public static class GgufTypes
{
    public static bool IsTrellis(GgufType type) =>
        type is >= GgufType.TRL1 and <= GgufType.TRL6;

    public static int GetTrellisBits(GgufType type) =>
        IsTrellis(type) ? (int)type - (int)GgufType.TRL1 + 1 : 0;

    public static bool TryGetBlockInfo(GgufType type, out int blockSize, out int blockBytes)
    {
        switch (type)
        {
            case GgufType.F32:
                blockSize = 1; blockBytes = 4; return true;
            case GgufType.F16:
            case GgufType.BF16:
                blockSize = 1; blockBytes = 2; return true;
            case GgufType.Q8_0:
                blockSize = 32; blockBytes = 34; return true;
            case GgufType.Q4_0:
            case GgufType.IQ4_NL:
                blockSize = 32; blockBytes = 18; return true;
            case GgufType.Q5_0:
                blockSize = 32; blockBytes = 22; return true;
            case GgufType.Q5_1:
                blockSize = 32; blockBytes = 24; return true;
            case GgufType.MXFP4:
                blockSize = 32; blockBytes = 17; return true;
            case GgufType.NVFP4:
                blockSize = 64; blockBytes = 36; return true;
            case GgufType.Q2_0:
                blockSize = 64; blockBytes = 20; return true;
            case GgufType.GgmlQ2_0:
                blockSize = 64; blockBytes = 18; return true;
            case GgufType.TRL1:
                blockSize = 16; blockBytes = 2; return true;
            case GgufType.TRL2:
                blockSize = 16; blockBytes = 4; return true;
            case GgufType.TRL3:
                blockSize = 16; blockBytes = 6; return true;
            case GgufType.TRL4:
                blockSize = 16; blockBytes = 8; return true;
            case GgufType.TRL5:
                blockSize = 16; blockBytes = 10; return true;
            case GgufType.TRL6:
                blockSize = 16; blockBytes = 12; return true;
            case GgufType.Q4_K:
                blockSize = 256; blockBytes = 144; return true;
            case GgufType.Q6_K:
                blockSize = 256; blockBytes = 210; return true;
            case GgufType.Q3_K:
                blockSize = 256; blockBytes = 110; return true;
            case GgufType.Q5_K:
                blockSize = 256; blockBytes = 176; return true;
            case GgufType.Q2_K:
                blockSize = 256; blockBytes = 84; return true;
            case GgufType.IQ1_S:
                blockSize = 256; blockBytes = 50; return true;
            case GgufType.IQ1_M:
                blockSize = 256; blockBytes = 56; return true;
            case GgufType.IQ2_XXS:
                blockSize = 256; blockBytes = 66; return true;
            case GgufType.IQ2_XS:
                blockSize = 256; blockBytes = 74; return true;
            case GgufType.IQ2_S:
                blockSize = 256; blockBytes = 82; return true;
            case GgufType.IQ3_XXS:
                blockSize = 256; blockBytes = 98; return true;
            case GgufType.IQ3_S:
                blockSize = 256; blockBytes = 110; return true;
            case GgufType.IQ4_XS:
                blockSize = 256; blockBytes = 136; return true;
            case GgufType.Ternary_1_58:
            case GgufType.PQ2_0:
                blockSize = 256; blockBytes = 68; return true;
            default:
                blockSize = 1; blockBytes = 2; return false;
        }
    }

    public static long GetRowBytes(GgufType type, int count)
    {
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), count, "Element count must be non-negative.");

        if (!TryGetBlockInfo(type, out int blockSize, out int blockBytes))
            return (long)count * 2;

        if (blockSize > 1 && (count % blockSize) != 0)
            throw new ArgumentException($"Element count ({count}) must be a multiple of block size ({blockSize}) for {type}.", nameof(count));

        return ((long)count / blockSize) * blockBytes;
    }
}

/// <summary>
/// Describes a single tensor stored inside the GGUF model file.
/// </summary>
public sealed class GgufTensorInfo
{
    public required string Name { get; init; }
    public required uint DimensionsCount { get; init; }
    public required ulong[] Dimensions { get; init; }
    public required GgufType Type { get; init; }
    public required ulong Offset { get; init; }

    public ulong ElementCount
    {
        get
        {
            if (Dimensions.Length == 0) return 0;
            ulong total = 1;
            for (int i = 0; i < Dimensions.Length; i++) total *= Dimensions[i];
            return total;
        }
    }

    public ulong GetByteSize()
    {
        ulong count = ElementCount;
        return Type switch
        {
            GgufType.F32 => count * 4,
            GgufType.F16 or GgufType.BF16 => count * 2,
            GgufType.Q8_0 => (count / 32) * 34,
            GgufType.Q4_0 => (count / 32) * 18,
            GgufType.Q5_0 => (count / 32) * 22,
            GgufType.Q5_1 => (count / 32) * 24,
            GgufType.Q4_K => (count / 256) * 144, // 256 elements in 144 bytes
            GgufType.Q6_K => (count / 256) * 210, // 256 elements in 210 bytes
            GgufType.Q3_K => (count / 256) * 110, // 256 elements in 110 bytes
            GgufType.Q5_K => (count / 256) * 176, // 256 elements in 176 bytes
            GgufType.Q2_K => (count / 256) * 84,  // 256 elements in 84 bytes
            GgufType.IQ1_S => (count / 256) * 50, // 256 elements in 50 bytes
            GgufType.IQ1_M => (count / 256) * 56, // 256 elements in 56 bytes
            GgufType.IQ2_XXS => (count / 256) * 66, // 256 elements in 66 bytes (2.0625 b/w)
            GgufType.IQ2_XS => (count / 256) * 74, // 256 elements in 74 bytes (2.3125 b/w)
            GgufType.IQ2_S => (count / 256) * 82,  // 256 elements in 82 bytes (2.5625 b/w)
            GgufType.IQ3_XXS => (count / 256) * 98, // 256 elements in 98 bytes (3.0625 b/w)
            GgufType.IQ3_S => (count / 256) * 110, // 256 elements in 110 bytes (3.4375 b/w)
            GgufType.IQ4_XS => (count / 256) * 136, // 256 elements in 136 bytes
            GgufType.IQ4_NL => (count / 32) * 18,  // 32 elements in 18 bytes
            GgufType.Q2_0 => (count / 64) * 20,    // 64 elements in 20 bytes (2.50 b/w fast MoE block)
            GgufType.MXFP4 => (count / 32) * 17,  // 32 elements in 17 bytes (E8M0 + 32x E2M1 FP4)
            GgufType.GgmlQ2_0 => (count / 64) * 18, // 64 elements in 18 bytes (upstream GGML_TYPE_Q2_0 = 42, 2.25 b/w)
            GgufType.TRL1 => (count / 16) * 2,  // NariQ trellis 1 bpw (16x16 tiles, 2 bytes per 16 elements)
            GgufType.TRL2 => (count / 16) * 4,  // NariQ trellis 2 bpw
            GgufType.TRL3 => (count / 16) * 6,  // NariQ trellis 3 bpw
            GgufType.TRL4 => (count / 16) * 8,  // NariQ trellis 4 bpw
            GgufType.TRL5 => (count / 16) * 10, // NariQ trellis 5 bpw
            GgufType.TRL6 => (count / 16) * 12, // NariQ trellis 6 bpw
            GgufType.Ternary_1_58 or GgufType.PQ2_0 => (count / 256) * 68, // 256 elements in 68 bytes (2.125 b/w multiplication-free ternary / Prism PQ2_0)
            _ => count * 2 // conservative fallback
        };
    }

    public override string ToString() =>
        $"{Name} [{string.Join(", ", Dimensions)}] ({Type}, {GetByteSize():N0} bytes, offset: {Offset:N0})";
}
