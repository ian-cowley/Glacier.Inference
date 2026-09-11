namespace Glacier.Inference.Hardware;

using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

/// <summary>
/// Hardware vendor classifications.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum GpuVendor
{
    Nvidia,
    Amd,
    Intel,
    Microsoft,
    Cpu,
    Unknown
}

/// <summary>
/// Metadata and capabilities for a compute accelerator or host processor.
/// </summary>
public sealed record DeviceInfo
{
    public required string Id { get; init; }
    public required int Index { get; init; }
    public required string Name { get; init; }
    public required GpuVendor Vendor { get; init; }
    public required ulong DedicatedVramBytes { get; init; }
    public required ulong SharedVramBytes { get; init; }
    public required bool IsDisplayDevice { get; init; }
    public required IReadOnlyList<InferenceEngineType> SupportedEngines { get; init; }
    public required InferenceEngineType RecommendedEngine { get; init; }
    public required string SafetyNotes { get; init; }

    public double DedicatedVramGb => DedicatedVramBytes / (1024.0 * 1024.0 * 1024.0);
    public double SharedVramGb => SharedVramBytes / (1024.0 * 1024.0 * 1024.0);

    public bool IsSafeEngine(InferenceEngineType engine)
    {
        if (engine == InferenceEngineType.Auto) return true;
        foreach (var supported in SupportedEngines)
        {
            if (supported == engine) return true;
        }
        return false;
    }
}
