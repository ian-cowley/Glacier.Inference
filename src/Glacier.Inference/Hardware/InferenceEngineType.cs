namespace Glacier.Inference.Hardware;

using System;
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
    /// Pure C# Bare-Metal SASS engine. Bypasses the CUDA Toolkit and cudart64.dll runtime,
    /// directly driving NVIDIA streaming multiprocessors with sub-microsecond latency.
    /// </summary>
    BareMetal,

    /// <summary>
    /// Microsoft DirectML / DirectX 12 Compute engine (directml.dll / d3d12.dll).
    /// Safe cooperative engine for AMD Radeon, Intel, and NVIDIA GPUs on Windows.
    /// </summary>
    DirectML,

    /// <summary>
    /// Universal Vulkan 1.3+ compute engine using VK_KHR_cooperative_matrix.
    /// Zero-dependency hardware tensor execution for AMD, Intel, and NVIDIA across Windows and Linux.
    /// </summary>
    Vulkan,

    /// <summary>
    /// SIMD-vectorized CPU execution engine (AVX-512 / AVX2 / ARM Neon).
    /// </summary>
    Cpu,

    /// <summary>
    /// Backward-compatible alias for BareMetal.
    /// </summary>
    [Obsolete("Use BareMetal instead. Glacier does not use the CUDA Toolkit runtime.")]
    Cuda = BareMetal
}
