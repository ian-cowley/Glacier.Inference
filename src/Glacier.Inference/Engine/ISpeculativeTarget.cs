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

    int VocabSize => Tokenizer.VocabSize;
    bool SupportsLogitVerification => false;
    bool IsRingBufferEnabled => false;
    long StateSnapshotBytes => 0;

    /// <summary>
    /// True when the target holds recurrent/SSM state that cannot be saved and restored on draft rejection.
    /// </summary>
    bool HasUnsnapshottableRecurrentState => false;

    /// <summary>
    /// True when the target uses a pure positional causal KV cache where stale entries at positions &gt;= pos
    /// are causally masked out and safely overwritten without requiring a full <see cref="Prefill"/>.
    /// </summary>
    bool CanOverwriteKvFromPrefix => false;

    void ResetKvCache();
    void Prefill(ReadOnlySpan<int> promptTokens);
    int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default);
    void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens);
    void ForwardToken(int token, int pos, bool computeLogits);

    void CopyCurrentLogits(Span<float> destination) => throw new NotSupportedException("Target does not expose raw logits.");

    void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
    {
        int vocabSize = VocabSize;
        for (int i = 0; i < tokens.Length; i++)
        {
            ForwardToken(tokens[i], startPos + i, computeLogits: true);
            CopyCurrentLogits(batchLogitsOut.Slice(i * vocabSize, vocabSize));
        }
    }

    void SaveRecurrentState(Span<byte> destination) { }
    void RestoreRecurrentState(ReadOnlySpan<byte> snapshot) { }
}
