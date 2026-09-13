namespace Glacier.Inference.Gpu.D3D12;

using System;
using Glacier.Inference.Gguf;
using Vortice.Direct3D12;

/// <summary>
/// Direct3D 12 GPU transformer layer weights residing in high-speed GPU device memory.
/// </summary>
public sealed class D3D12LayerWeights : IDisposable
{
    public required ID3D12Resource AttnNormWeight { get; init; }
    public required ID3D12Resource QWeight { get; init; }
    public ID3D12Resource? QBias { get; init; }
    public required ID3D12Resource KWeight { get; init; }
    public ID3D12Resource? KBias { get; init; }
    public required ID3D12Resource VWeight { get; init; }
    public ID3D12Resource? VBias { get; init; }
    public required ID3D12Resource AttnOutWeight { get; init; }

    // Optional QK-Norm & Sinks
    public ID3D12Resource? AttnQNormWeight { get; init; }
    public ID3D12Resource? AttnKNormWeight { get; init; }
    public ID3D12Resource? AttnSinksWeight { get; init; }

    public required ID3D12Resource FfnNormWeight { get; init; }

    // Dense FFN
    public ID3D12Resource? FfnGateWeight { get; init; }
    public ID3D12Resource? FfnUpWeight { get; init; }
    public ID3D12Resource? FfnDownWeight { get; init; }

    public required GgufType QType { get; init; }
    public required GgufType KType { get; init; }
    public required GgufType VType { get; init; }
    public required GgufType AttnOutType { get; init; }
    public GgufType FfnGateType { get; init; }
    public GgufType FfnUpType { get; init; }
    public GgufType FfnDownType { get; init; }

    // MoE Router & Experts
    public bool IsMoe { get; init; }
    public ID3D12Resource? FfnGateInpWeight { get; init; }
    public ID3D12Resource? FfnGateInpBias { get; init; }
    public ID3D12Resource? FfnGateExpsWeight { get; init; }
    public ID3D12Resource? FfnUpExpsWeight { get; init; }
    public ID3D12Resource? FfnDownExpsWeight { get; init; }
    public GgufType FfnGateExpsType { get; init; }
    public GgufType FfnUpExpsType { get; init; }
    public GgufType FfnDownExpsType { get; init; }

    // Shared Experts (DeepSeek, ERNIE)
    public ID3D12Resource? FfnGateShexpWeight { get; init; }
    public ID3D12Resource? FfnUpShexpWeight { get; init; }
    public ID3D12Resource? FfnDownShexpWeight { get; init; }
    public GgufType FfnGateShexpType { get; init; }
    public GgufType FfnUpShexpType { get; init; }
    public GgufType FfnDownShexpType { get; init; }

    public void Dispose()
    {
        AttnNormWeight?.Dispose();
        QWeight?.Dispose();
        QBias?.Dispose();
        KWeight?.Dispose();
        KBias?.Dispose();
        VWeight?.Dispose();
        VBias?.Dispose();
        AttnOutWeight?.Dispose();
        AttnQNormWeight?.Dispose();
        AttnKNormWeight?.Dispose();
        AttnSinksWeight?.Dispose();
        FfnNormWeight?.Dispose();
        FfnGateWeight?.Dispose();
        FfnUpWeight?.Dispose();
        FfnDownWeight?.Dispose();
        FfnGateInpWeight?.Dispose();
        FfnGateInpBias?.Dispose();
        FfnGateExpsWeight?.Dispose();
        FfnUpExpsWeight?.Dispose();
        FfnDownExpsWeight?.Dispose();
        FfnGateShexpWeight?.Dispose();
        FfnUpShexpWeight?.Dispose();
        FfnDownShexpWeight?.Dispose();
    }
}
