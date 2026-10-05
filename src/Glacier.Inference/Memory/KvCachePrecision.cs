namespace Glacier.Inference.Memory;

/// <summary>
/// Hardware precision format for in-VRAM and in-memory Key-Value (KV) cache storage.
/// <para>
/// On GPU backends:
/// <list type="bullet">
/// <item><description><b>CUDA (Bare-Metal)</b>: Supports <see cref="Auto"/>, <see cref="Fp32"/>, <see cref="Fp16"/> (2x VRAM savings), and <see cref="Fp8"/> (4x VRAM savings) with native PTX/CUBIN attention kernels.</description></item>
/// <item><description><b>Direct3D 12</b>: Supports <see cref="Auto"/>, <see cref="Fp32"/>, and <see cref="Fp16"/> (2x VRAM savings via packed half2 HLSL compute shaders).</description></item>
/// </list>
/// </para>
/// <para>
/// On CPU backends (<c>ICpuModel</c>):
/// KV cache storage and attention computation always execute in uncompressed IEEE 754 FP32 format (<see cref="Fp32"/>).
/// This design maximizes vector arithmetic throughput across hardware SIMD instruction sets (AVX-512, AVX2, ARM NEON)
/// and avoids runtime FP16/FP8 unpack decompression overhead on every token generation step, since CPU inference is
/// compute- and cache-bound rather than raw VRAM bandwidth bound.
/// </para>
/// </summary>
public enum KvCachePrecision
{
    /// <summary>
    /// Automatically selects optimal precision based on context length and hardware capacity:
    /// - For GPU backends with maxSeqLen &lt;= 4096: FP16 (lossless, 2x VRAM reduction vs FP32).
    /// - For GPU backends with maxSeqLen &gt; 4096: FP8 (4x VRAM reduction on supported silicon such as Ada Lovelace sm_89 and AMD RDNA 3/4 WMMA).
    /// - For CPU backends: Evaluated as FP32 for maximum SIMD throughput.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// 32-bit single-precision float (4 bytes per element). Standard precision with maximum dynamic range across all backends.
    /// </summary>
    Fp32 = 1,

    /// <summary>
    /// 16-bit half-precision float (2 bytes per element, 2x VRAM compression).
    /// Hardware accelerated on CUDA and Direct3D 12 GPUs.
    /// </summary>
    Fp16 = 2,

    /// <summary>
    /// 8-bit floating point e4m3 (1 byte per element, 4x VRAM compression).
    /// Hardware accelerated on NVIDIA Ada Lovelace (sm_89+) and AMD RDNA 3/4 WMMA silicon.
    /// </summary>
    Fp8 = 3
}
