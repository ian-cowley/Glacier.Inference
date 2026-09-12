namespace Glacier.Inference.Engine;

using System;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Target model interface for speculative decoding verification.
/// Implemented by <see cref="InferenceSession"/> and test harnesses.
/// </summary>
public interface ISpeculativeTarget : IDisposable
{
    int MaxSeqLen { get; }
    BpeTokenizer Tokenizer { get; }

    void ResetKvCache();
    void Prefill(ReadOnlySpan<int> promptTokens);
    int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default);
    void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens);
    void ForwardToken(int token, int pos, bool computeLogits);
}
