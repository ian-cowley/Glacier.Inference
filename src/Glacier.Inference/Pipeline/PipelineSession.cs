namespace Glacier.Inference.Pipeline;

using System;
using System.Collections.Generic;
using System.Text;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;

/// <summary>
/// Multi-stage pipeline parallelism session coordinating forward execution across heterogeneous silicon.
/// Passes activation vectors between discrete GPU, integrated GPU, and CPU stages with sub-microsecond latency.
/// </summary>
public sealed class PipelineSession : IDisposable
{
    private readonly IPipelineStage[] _stages;
    private readonly int _dim;
    private readonly int _vocabSize;
    private float[] _x0;
    private float[] _x1;
    private float[] _xBatch0;
    private float[] _xBatch1;
    private readonly Sampler _sampler = new();
    private bool _disposed;

    public IReadOnlyList<IPipelineStage> Stages => _stages;
    public int StageCount => _stages.Length;
    public int StartLayer => _stages[0].StartLayer;
    public int EndLayer => _stages[^1].EndLayer;
    public int TotalLayers => EndLayer - StartLayer;
    public string TopologyDescription { get; }

    public PipelineSession(IPipelineStage[] stages, int dim, int vocabSize)
    {
        if (stages == null || stages.Length == 0)
            throw new ArgumentException("Stages cannot be null or empty.", nameof(stages));

        _stages = stages;
        _dim = dim;
        _vocabSize = vocabSize;

        _x0 = new float[_dim];
        _x1 = new float[_dim];
        _xBatch0 = new float[_dim * 32];
        _xBatch1 = new float[_dim * 32];

        var sb = new StringBuilder();
        sb.Append($"Pipeline [{_stages.Length} Stages | {TotalLayers} Layers]: ");
        for (int i = 0; i < _stages.Length; i++)
        {
            if (i > 0) sb.Append(" -> ");
            sb.Append($"{_stages[i].Device.Name} (L{_stages[i].StartLayer}..{_stages[i].EndLayer - 1}, {_stages[i].Engine})");
        }
        TopologyDescription = sb.ToString();
    }

    /// <summary>
    /// Factory creating a configured PipelineSession from parsed stage specifications.
    /// </summary>
    public static PipelineSession Create(
        ModelWeights weights,
        List<PipelineStageSpec> specs,
        int maxSeqLen = 4096,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto)
    {
        if (specs == null || specs.Count == 0)
            throw new ArgumentException("Pipeline stage specifications cannot be null or empty.", nameof(specs));

        var stages = new IPipelineStage[specs.Count];
        for (int i = 0; i < specs.Count; i++)
        {
            try
            {
                var spec = specs[i];
                bool isFirst = (i == 0);
                bool isLast = (i == specs.Count - 1);

                if (spec.Engine == InferenceEngineType.BareMetal && spec.Device.Vendor == GpuVendor.Nvidia)
                {
                    stages[i] = new CudaPipelineStage(
                        spec.Device,
                        weights,
                        startLayer: spec.StartLayer,
                        layerCount: spec.LayerCount,
                        isFirstStage: isFirst,
                        isLastStage: isLast,
                        maxSeqLen: maxSeqLen,
                        kvPrecision: kvPrecision);
                }
                else if ((spec.Engine == InferenceEngineType.DirectML || spec.Engine == InferenceEngineType.BareMetal) &&
                         (spec.Device.Vendor == GpuVendor.Amd || spec.Device.Vendor == GpuVendor.Intel) &&
                         OperatingSystem.IsWindows())
                {
                    stages[i] = new D3D12PipelineStage(
                        spec.Device,
                        weights,
                        startLayer: spec.StartLayer,
                        layerCount: spec.LayerCount,
                        isFirstStage: isFirst,
                        isLastStage: isLast,
                        maxSeqLen: maxSeqLen);
                }
                else
                {
                    stages[i] = new CpuPipelineStage(
                        spec.Device,
                        weights,
                        startLayer: spec.StartLayer,
                        layerCount: spec.LayerCount,
                        isFirstStage: isFirst,
                        isLastStage: isLast,
                        maxSeqLen: maxSeqLen);
                }
            }
            catch
            {
                for (int j = 0; j < i; j++)
                {
                    stages[j]?.Dispose();
                }
                throw;
            }
        }

        return new PipelineSession(stages, weights.EmbeddingLength, weights.VocabSize);
    }

    /// <summary>
    /// Executes a single token forward step across all pipeline stages.
    /// </summary>
    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        if (_stages.Length == 1)
        {
            _stages[0].ForwardStage(token, pos, default, default, logits, computeLogits);
            return;
        }

        Span<float> curIn = Span<float>.Empty;
        Span<float> curOut = _x0.AsSpan();

        for (int i = 0; i < _stages.Length; i++)
        {
            var stage = _stages[i];
            bool isFirst = (i == 0);
            bool isLast = (i == _stages.Length - 1);

            if (isFirst)
            {
                stage.ForwardStage(token, pos, default, curOut, default, computeLogits: false);
                curIn = curOut;
                curOut = _x1.AsSpan();
            }
            else if (isLast)
            {
                stage.ForwardStage(token, pos, curIn, default, logits, computeLogits);
            }
            else
            {
                stage.ForwardStage(token, pos, curIn, curOut, default, computeLogits: false);
                var temp = curIn;
                curIn = curOut;
                curOut = temp;
            }
        }
    }

    /// <summary>
    /// Executes batched prefill across all pipeline stages.
    /// </summary>
    public void ForwardBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> logits,
        bool computeLogits = true)
    {
        if (tokens.IsEmpty) return;
        int numTokens = tokens.Length;

        if (_stages.Length == 1)
        {
            _stages[0].ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits);
            return;
        }

        int requiredElements = numTokens * _dim;
        EnsureBatchCapacity(requiredElements);

        Span<float> curIn = Span<float>.Empty;
        Span<float> curOut = _xBatch0.AsSpan(0, requiredElements);

        for (int i = 0; i < _stages.Length; i++)
        {
            var stage = _stages[i];
            bool isFirst = (i == 0);
            bool isLast = (i == _stages.Length - 1);

            if (isFirst)
            {
                stage.ForwardBatchStage(tokens, startPos, default, curOut, default, computeLogits: false);
                curIn = curOut;
                curOut = _xBatch1.AsSpan(0, requiredElements);
            }
            else if (isLast)
            {
                stage.ForwardBatchStage(default, startPos, curIn, default, logits, computeLogits);
            }
            else
            {
                stage.ForwardBatchStage(default, startPos, curIn, curOut, default, computeLogits: false);
                var temp = curIn;
                curIn = curOut;
                curOut = temp;
            }
        }
    }

    /// <summary>
    /// Samples the next token using the final stage's hardware reducer or fallback CPU sampler.
    /// </summary>
    public int SampleToken(
        SamplingOptions options,
        ReadOnlySpan<int> recentTokens = default,
        Span<float> logits = default,
        Sampler? sampler = null)
    {
        var lastStage = _stages[^1];
        int sampled = lastStage.SampleToken(options, recentTokens);
        if (sampled >= 0)
            return sampled;

        var s = sampler ?? _sampler;
        return s.Sample(logits, options, recentTokens);
    }

    /// <summary>
    /// Resets layer-local KV caches on all pipeline stages.
    /// </summary>
    public void ResetKvCache()
    {
        for (int i = 0; i < _stages.Length; i++)
        {
            _stages[i].ResetKvCache();
        }
    }

    private void EnsureBatchCapacity(int requiredElements)
    {
        if (_xBatch0.Length < requiredElements)
        {
            _xBatch0 = new float[requiredElements];
            _xBatch1 = new float[requiredElements];
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            for (int i = 0; i < _stages.Length; i++)
            {
                _stages[i].Dispose();
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
