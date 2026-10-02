namespace Glacier.Inference.Gpu.D3D12;

using System;

/// <summary>
/// Common interface for Direct3D 12 GPU compute LLM inference models.
/// </summary>
public interface ID3D12Model : IDisposable
{
    void Forward(int token, int pos, Span<float> logits, bool computeLogits = true);
    void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true);
}
