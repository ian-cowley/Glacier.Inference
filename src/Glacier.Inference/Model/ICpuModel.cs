namespace Glacier.Inference.Model;

using System;
using Glacier.Inference.Engine;
using Glacier.Inference.Memory;

/// <summary>
/// Universal interface for CPU SIMD-accelerated model execution runtime engines.
/// Provides zero-allocation single-token decode and batched prompt prefill.
/// </summary>
public interface ICpuModel : IDisposable
{
    ModelWeights Weights { get; }
    int MaxSeqLen { get; }
    int StartLayer { get; }
    int LayerCount { get; }
    bool IsLastStage { get; }
    LoraAdapterWeights? LoraWeights { get; set; }

    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// Writes logits of size VocabSize into destination buffer if computeLogits is true.
    /// </summary>
    void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true);

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null);

    /// <summary>
    /// Executes forward pass for an entire sequence of prompt tokens in batched chunks.
    /// </summary>
    void ForwardBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> logits,
        KVCache kvCache,
        bool computeLogits = true);

    /// <summary>
    /// Executes layers assigned to this stage for a batch of tokens.
    /// </summary>
    void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null);

    /// <summary>
    /// Prefills the KV cache with prompt tokens without computing final output logits.
    /// </summary>
    void PrefillPrompt(ReadOnlySpan<int> tokens, KVCache kvCache);

    /// <summary>
    /// Resets any internal model state (such as recurrent SSM/GDN caches or model-owned KV caches)
    /// between independent prompts or conversations.
    /// </summary>
    void ResetState();

    /// <summary>
    /// Byte size required to snapshot recurrent state (0 for pure attention models).
    /// </summary>
    long StateSnapshotBytes => 0;

    /// <summary>
    /// Saves internal recurrent state into <paramref name="destination"/> for speculative rollback.
    /// </summary>
    void SaveRecurrentState(Span<byte> destination) { }

    /// <summary>
    /// Restores internal recurrent state from <paramref name="source"/> after speculative draft rejection.
    /// </summary>
    void RestoreRecurrentState(ReadOnlySpan<byte> source) { }

    /// <summary>
    /// Minimum sorted expert rank eligible for speculative/verify MoE tail skipping (0 disables skipping).
    /// When set above 0, experts at ranks &gt;= <see cref="MoeTailSkipMinRank"/> whose normalized
    /// routing weight falls below a small fraction of the top-1 weight are skipped during batched verify.
    /// </summary>
    int MoeTailSkipMinRank { get => 0; set { } }

    /// <summary>
    /// True if the model implements batched multi-token speculative verification (<see cref="VerifyBatch"/>
    /// and <see cref="VerifyBatchLogits"/>).
    /// </summary>
    bool SupportsBatchVerification => false;

    /// <summary>
    /// True if <see cref="VerifyBatch"/> and <see cref="VerifyBatchLogits"/> capture exact post-token
    /// recurrent state snapshots into <c>perStepSnapshots</c>, eliminating rejection rollback replay.
    /// </summary>
    bool SupportsPerStepVerificationSnapshots => false;

    /// <summary>
    /// Executes batched multi-token verification and outputs the greedy argmax token ID for each token step.
    /// When <paramref name="perStepSnapshots"/> is provided, step <c>t</c> (<c>0 &lt;= t &lt; perStepSnapshots.Length</c>)
    /// receives the exact post-token recurrent state snapshot needed if token <c>t + 1</c> is rejected.
    /// </summary>
    void VerifyBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<int> greedyTokens,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
        => throw new NotSupportedException("VerifyBatch is not implemented for this model.");

    /// <summary>
    /// Executes batched multi-token verification and writes each step's vocabulary logits into
    /// <paramref name="logitsOut"/> with row stride <paramref name="vocabStride"/>.
    /// When <paramref name="perStepSnapshots"/> is provided, step <c>t</c> (<c>0 &lt;= t &lt; perStepSnapshots.Length</c>)
    /// receives the exact post-token recurrent state snapshot needed if token <c>t + 1</c> is rejected.
    /// </summary>
    void VerifyBatchLogits(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> logitsOut,
        int vocabStride,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
        => throw new NotSupportedException("VerifyBatchLogits is not implemented for this model.");

    /// <summary>
    /// Extracts a normalized embedding vector for the given token sequence.
    /// </summary>
    void ExtractEmbedding(
        ReadOnlySpan<int> tokens,
        Span<float> destination,
        KVCache kvCache,
        PoolingStrategy strategy = PoolingStrategy.LastToken);
}
