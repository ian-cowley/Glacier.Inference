namespace Glacier.Inference.Gpu.D3D12;

using System;

/// <summary>
/// Common interface for Direct3D 12 GPU compute LLM inference models.
/// </summary>
public interface ID3D12Model : IDisposable
{
    void Forward(int token, int pos, Span<float> logits, bool computeLogits = true);
    void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true);
    void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits)
        => Forward(token, pos, logits, computeLogits);
    void ForwardBatchStage(ReadOnlySpan<int> tokens, int startPos, ReadOnlySpan<float> inputXBatch, Span<float> outputXBatch, Span<float> logits, bool computeLogits)
    {
        int dim = tokens.Length > 0 ? Math.Max(inputXBatch.Length, outputXBatch.Length) / tokens.Length : 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            bool isLast = i == tokens.Length - 1;
            var inSlice = !inputXBatch.IsEmpty && dim > 0 ? inputXBatch.Slice(i * dim, dim) : ReadOnlySpan<float>.Empty;
            var outSlice = !outputXBatch.IsEmpty && dim > 0 ? outputXBatch.Slice(i * dim, dim) : Span<float>.Empty;
            ForwardStage(tokens[i], startPos + i, inSlice, outSlice, isLast && computeLogits ? logits : Span<float>.Empty, isLast && computeLogits);
        }
    }
    void ResetKvCache() { }
}
