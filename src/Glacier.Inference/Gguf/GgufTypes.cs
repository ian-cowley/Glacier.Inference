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
    BF16 = 24,
    Q4_0_4_4 = 28,
    Q4_0_4_8 = 29,
    Q4_0_8_8 = 30,
    MXFP4 = 39
}

/// <summary>
/// Helper utilities for GGUF types and block strides.
/// </summary>
public static class GgufTypes
{
    public static long GetRowBytes(GgufType type, int count) => type switch
    {
        GgufType.F32 => (long)count * 4,
        GgufType.F16 or GgufType.BF16 => (long)count * 2,
        GgufType.Q8_0 => ((long)count / 32) * 34,
        GgufType.Q4_0 => ((long)count / 32) * 18,
        GgufType.Q4_K => ((long)count / 256) * 144,
        GgufType.Q6_K => ((long)count / 256) * 210,
        GgufType.Q3_K => ((long)count / 256) * 110,
        GgufType.Q5_K => ((long)count / 256) * 176,
        GgufType.MXFP4 => ((long)count / 32) * 17,
        _ => (long)count * 2
    };
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
            GgufType.Q4_K => (count / 256) * 144, // 256 elements in 144 bytes
            GgufType.Q6_K => (count / 256) * 210, // 256 elements in 210 bytes
            GgufType.Q3_K => (count / 256) * 110, // 256 elements in 110 bytes
            GgufType.Q5_K => (count / 256) * 176, // 256 elements in 176 bytes
            GgufType.MXFP4 => (count / 32) * 17,  // 32 elements in 17 bytes (E8M0 + 32x E2M1 FP4)
            _ => count * 2 // conservative fallback
        };
    }

    public override string ToString() =>
        $"{Name} [{string.Join(", ", Dimensions)}] ({Type}, {GetByteSize():N0} bytes, offset: {Offset:N0})";
}
