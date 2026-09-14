namespace Glacier.Inference.Pipeline;

using System;
using Glacier.Inference.Hardware;
using Glacier.Inference.Sampling;

/// <summary>
/// Defines an executable pipeline stage in a heterogeneous multi-device LLM pipeline.
/// Each stage is responsible for executing a contiguous slice of transformer layers [StartLayer, EndLayer).
/// </summary>
public interface IPipelineStage : IDisposable
{
    /// <summary>
    /// Index of the first transformer block managed by this stage (inclusive).
    /// </summary>
    int StartLayer { get; }

    /// <summary>
    /// Number of transformer blocks managed by this stage.
    /// </summary>
    int LayerCount { get; }

    /// <summary>
    /// Index of the first transformer block beyond this stage (exclusive).
    /// </summary>
    int EndLayer => StartLayer + LayerCount;

    /// <summary>
    /// Hardware device hosting this stage.
    /// </summary>
    DeviceInfo Device { get; }

    /// <summary>
    /// Acceleration engine driving this stage.
    /// </summary>
    InferenceEngineType Engine { get; }

    /// <summary>
    /// True if this stage is the initial stage responsible for input token embedding lookup.
    /// </summary>
    bool IsFirstStage { get; }

    /// <summary>
    /// True if this stage is the final stage responsible for final RMSNorm and LM head projection.
    /// </summary>
    bool IsLastStage { get; }

    /// <summary>
    /// Executes the layers assigned to this stage for a single token forward step.
    /// </summary>
    void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits);

    /// <summary>
    /// Executes batched prefill for the layers assigned to this stage.
    /// </summary>
    void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits);

    /// <summary>
    /// Samples next token on-device if supported (e.g. In-VRAM GPU Argmax), or returns -1 for CPU fallback sampling.
    /// </summary>
    int SampleToken(SamplingOptions options, ReadOnlySpan<int> recentTokens);

    /// <summary>
    /// Resets layer-local KV caches on this stage.
    /// </summary>
    void ResetKvCache();
}
