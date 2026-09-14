namespace Glacier.Inference.Gpu;

using System;
using Glacier.Inference.Gguf;

/// <summary>
/// GPU-accelerated transformer layer weights residing directly in GPU VRAM.
/// </summary>
public sealed class GpuLayerWeights
{
    public IntPtr AttnNormWeight { get; init; }
    public IntPtr QWeight { get; init; }
    public IntPtr QBias { get; init; }
    public IntPtr KWeight { get; init; }
    public IntPtr KBias { get; init; }
    public IntPtr VWeight { get; init; }
    public IntPtr VBias { get; init; }
    public IntPtr AttnOutWeight { get; init; }
    public IntPtr AttnQNormWeight { get; init; }
    public IntPtr AttnKNormWeight { get; init; }

    public IntPtr FfnNormWeight { get; init; }
    public IntPtr FfnGateWeight { get; init; }
    public IntPtr FfnUpWeight { get; init; }
    public IntPtr FfnDownWeight { get; init; }

    public GgufType QType { get; init; }
    public GgufType KType { get; init; }
    public GgufType VType { get; init; }
    public GgufType AttnOutType { get; init; }
    public GgufType FfnGateType { get; init; }
    public GgufType FfnUpType { get; init; }
    public GgufType FfnDownType { get; init; }
}
