namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gpu;
using Xunit;

/// <summary>
/// Custom xUnit fact attribute that automatically skips integration tests when
/// NVIDIA CUDA drivers or hardware are not present in the runtime environment (e.g. CI runners).
/// </summary>
public sealed class CudaFactAttribute : FactAttribute
{
    private const string ModelPath = @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";

    public CudaFactAttribute()
    {
        if (!CuDriver.IsAvailable())
        {
            Skip = "NVIDIA CUDA driver library is not available in the current environment.";
        }
        else if (!GpuContext.IsSupported)
        {
            Skip = "No CUDA-capable GPU hardware device detected.";
        }
        else if (!File.Exists(ModelPath))
        {
            Skip = $"Benchmark model file not found at '{ModelPath}'.";
        }
    }
}
