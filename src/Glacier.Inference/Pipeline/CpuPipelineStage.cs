namespace Glacier.Inference.Pipeline;

using System;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;

/// <summary>
/// CPU SIMD pipeline stage executing transformer layers via multi-threaded AVX2/AVX-512 kernels.
/// </summary>
public sealed class CpuPipelineStage : IPipelineStage
{
    private readonly Qwen2Model _model;
    private readonly KVCache _kvCache;
    private readonly DeviceInfo _device;
    private readonly InferenceEngineType _engine;
    private readonly bool _isFirstStage;
    private readonly bool _isLastStage;
    private bool _disposed;

    public int StartLayer => _model.StartLayer;
    public int LayerCount => _model.LayerCount;
    public DeviceInfo Device => _device;
    public InferenceEngineType Engine => _engine;
    public bool IsFirstStage => _isFirstStage;
    public bool IsLastStage => _isLastStage;
    public Qwen2Model Model => _model;

    public CpuPipelineStage(
        DeviceInfo device,
        ModelWeights weights,
        int startLayer,
        int layerCount,
        bool isFirstStage,
        bool isLastStage,
        int maxSeqLen = 4096)
    {
        _device = device;
        _engine = InferenceEngineType.Cpu;
        _isFirstStage = isFirstStage;
        _isLastStage = isLastStage;
        int resolvedLayers = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        _kvCache = new KVCache(resolvedLayers, weights.HeadCountKv, weights.HeadDim, maxSeqLen, weights.ValueDim);
        _model = new Qwen2Model(
            weights,
            maxSeqLen: maxSeqLen,
            startLayer: startLayer,
            layerCount: resolvedLayers,
            isLastStage: isLastStage);
    }

    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits)
    {
        _model.ForwardStage(token, pos, inputX, outputX, logits, computeLogits, _kvCache);
    }

    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        _model.ForwardBatchStage(tokens, startPos, inputXBatch, outputXBatch, logits, computeLogits, _kvCache);
    }

    public int SampleToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        return -1; // Pipeline coordinator falls back to CPU Sampler using output logits
    }

    public void ResetKvCache()
    {
        _kvCache.Reset();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _model.Dispose();
            _kvCache.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
