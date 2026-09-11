namespace Glacier.Inference.Hardware;

using System.Text.Json.Serialization;

/// <summary>
/// Execution engine options for model inference.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum InferenceEngineType
{
    /// <summary>
    /// Automatically select the highest-performing safe engine for the target device.
    /// </summary>
    Auto,

    /// <summary>
    /// Bare-Metal NVIDIA CUDA driver engine (nvcuda.dll). Optimal for NVIDIA GPUs.
    /// </summary>
    Cuda,

    /// <summary>
    /// Microsoft DirectML / DirectX 12 Compute engine (directml.dll / d3d12.dll).
    /// Safe cooperative engine for AMD Radeon, Intel, and NVIDIA GPUs on Windows.
    /// </summary>
    DirectML,

    /// <summary>
    /// SIMD-vectorized CPU execution engine (AVX-512 / AVX2 / ARM Neon).
    /// </summary>
    Cpu
}
