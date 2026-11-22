namespace Glacier.Inference.Memory;

/// <summary>
/// Hardware precision format for in-VRAM and in-memory Key-Value (KV) cache storage.
/// <para>
/// On GPU backends:
/// <list type="bullet">
/// <item><description><b>CUDA (Bare-Metal)</b>: Supports <see cref="Auto"/>, <see cref="Fp32"/>, <see cref="Fp16"/> (2x VRAM savings), <see cref="Fp8"/> (4x VRAM savings), <see cref="Q8_0"/> (~3.76x vs FP32, ~1.88x vs FP16 in paged KV), and <see cref="Q4_0"/> / <see cref="Int4"/> (~7.1x vs FP32, ~3.56x vs FP16, ~1.78x vs FP8) with native CUBIN attention kernels.</description></item>
/// <item><description><b>Direct3D 12</b>: Supports <see cref="Auto"/>, <see cref="Fp32"/>, <see cref="Fp16"/> (2x VRAM savings), and <see cref="Q4_0"/> / <see cref="Int4"/> (~3.56x vs FP16, ~1.78x vs FP8) via HLSL compute shaders.</description></item>
/// </list>
/// </para>
/// <para>
/// On CPU backends (<c>ICpuModel</c>):
/// Supports <see cref="Fp32"/> (default when <see cref="Auto"/> is specified), <see cref="Fp16"/>, <see cref="Fp8"/>,
/// symmetric 8-bit <see cref="Q8_0"/> (34-byte 32-element blocks, <c>BlockQ8_0</c>), and packed 4-bit <see cref="Q4_0"/> / <see cref="Int4"/>
/// using 18-byte 32-element blocks (<c>BlockQ4_0</c>) for ~2x context capacity over <see cref="Fp8"/> and ~4x over <see cref="Fp16"/>.
/// </para>
/// </summary>
public enum KvCachePrecision
{
    /// <summary>
    /// Automatically selects optimal precision based on context length and hardware capacity:
    /// - For CUDA GPU backends with maxSeqLen &lt;= 4096: FP16 (2x VRAM reduction vs FP32); &gt; 4096: FP8 (or FP16 in paged pool mode).
    /// - For Direct3D 12 GPU backends: FP16.
    /// - For CPU backends: Evaluated as FP32 for maximum SIMD throughput.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// 32-bit single-precision float (4 bytes per element). Standard precision with maximum dynamic range across all backends.
    /// </summary>
    Fp32 = 1,

    /// <summary>
    /// 16-bit half-precision float (2 bytes per element, 2x compression vs FP32).
    /// Supported across CPU, CUDA, and Direct3D 12 backends.
    /// </summary>
    Fp16 = 2,

    /// <summary>
    /// 8-bit floating point e4m3 (1 byte per element, 4x compression vs FP32, 2x vs FP16).
    /// Supported on CPU and CUDA Ada Lovelace (sm_89+) silicon.
    /// </summary>
    Fp8 = 3,

    /// <summary>
    /// Packed 4-bit block quantization (32-element blocks with FP16 scale, 18 bytes per 32 elements = 0.5625 bytes/element).
    /// Provides ~2x context capacity over <see cref="Fp8"/> and ~4x over <see cref="Fp16"/> across CPU, CUDA, and Direct3D 12.
    /// </summary>
    Q4_0 = 4,

    /// <summary>
    /// Alias for <see cref="Q4_0"/> (packed 4-bit integer block KV-cache storage).
    /// </summary>
    Int4 = Q4_0,

    /// <summary>
    /// Symmetric 8-bit block quantization (32-element blocks with FP16 scale, 34 bytes per 32 elements = 1.0625 bytes/element).
    /// Provides ~3.76x compression vs FP32 and ~1.88x vs FP16 while preserving near-lossless attention accuracy.
    /// </summary>
    Q8_0 = 5
}

/// <summary>
/// AOT-safe parsing and display helpers for <see cref="KvCachePrecision"/>.
/// </summary>
public static class KvCachePrecisionExtensions
{
    /// <summary>
    /// Parses a KV-cache precision identifier without reflection.
    /// Accepts <c>auto</c>, <c>fp32</c>/<c>f32</c>, <c>fp16</c>/<c>f16</c>, <c>fp8</c>/<c>fp8_e4m3</c>/<c>e4m3</c>,
    /// <c>q8_0</c>/<c>q8</c>/<c>int8</c>, and <c>q4_0</c>/<c>q4</c>/<c>int4</c> (case-insensitive).
    /// </summary>
    public static bool TryParse(string? text, out KvCachePrecision precision)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            precision = KvCachePrecision.Auto;
            return false;
        }

        switch (text.Trim().ToLowerInvariant())
        {
            case "auto":
                precision = KvCachePrecision.Auto;
                return true;
            case "fp32":
            case "f32":
                precision = KvCachePrecision.Fp32;
                return true;
            case "fp16":
            case "f16":
                precision = KvCachePrecision.Fp16;
                return true;
            case "fp8":
            case "fp8_e4m3":
            case "e4m3":
                precision = KvCachePrecision.Fp8;
                return true;
            case "q8_0":
            case "q8":
            case "int8":
                precision = KvCachePrecision.Q8_0;
                return true;
            case "q4_0":
            case "q4":
            case "int4":
                precision = KvCachePrecision.Q4_0;
                return true;
            default:
                precision = KvCachePrecision.Auto;
                return false;
        }
    }

    /// <summary>
    /// Returns a canonical display name for <paramref name="precision"/> (always returning <c>"Q4_0"</c> for value 4).
    /// </summary>
    public static string ToDisplayName(this KvCachePrecision precision) => precision switch
    {
        KvCachePrecision.Auto => "Auto",
        KvCachePrecision.Fp32 => "Fp32",
        KvCachePrecision.Fp16 => "Fp16",
        KvCachePrecision.Fp8 => "Fp8",
        KvCachePrecision.Q4_0 => "Q4_0",
        KvCachePrecision.Q8_0 => "Q8_0",
        _ => precision.ToString()
    };
}
