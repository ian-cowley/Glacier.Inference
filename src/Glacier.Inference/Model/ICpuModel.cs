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
    /// Extracts a normalized embedding vector for the given token sequence.
    /// </summary>
    void ExtractEmbedding(
        ReadOnlySpan<int> tokens,
        Span<float> destination,
        KVCache kvCache,
        PoolingStrategy strategy = PoolingStrategy.LastToken);
}
