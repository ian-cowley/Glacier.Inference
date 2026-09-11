namespace Glacier.Inference.Memory;

/// <summary>
/// Hardware precision format for in-VRAM Key-Value (KV) cache storage.
/// </summary>
public enum KvCachePrecision
{
    /// <summary>
    /// Automatically selects optimal precision based on context length and GPU VRAM capacity:
    /// - For maxSeqLen &lt;= 4096: FP16 (lossless, 2x VRAM reduction vs FP32)
    /// - For maxSeqLen &gt; 4096: FP8 (4x VRAM reduction, native Ada Lovelace sm_89 silicon)
    /// </summary>
    Auto = 0,

    /// <summary>
    /// 32-bit single-precision float (4 bytes per element).
    /// </summary>
    Fp32 = 1,

    /// <summary>
    /// 16-bit half-precision float (2 bytes per element, 2x compression, lossless).
    /// </summary>
    Fp16 = 2,

    /// <summary>
    /// 8-bit floating point e4m3 (1 byte per element, 4x compression, native Ada Lovelace instructions).
    /// </summary>
    Fp8 = 3
}
