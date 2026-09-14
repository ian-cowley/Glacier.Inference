namespace Glacier.Inference.Pipeline;

using System;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;

/// <summary>
/// NVIDIA CUDA pipeline stage executing transformer layers directly via SASS / nvcuda.dll.
/// </summary>
public sealed class CudaPipelineStage : IPipelineStage
{
    private readonly GpuContext _gpu;
    private readonly Qwen2GpuModel _model;
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
    public Qwen2GpuModel Model => _model;

    public CudaPipelineStage(
        DeviceInfo device,
        ModelWeights weights,
        int startLayer,
        int layerCount,
        bool isFirstStage,
        bool isLastStage,
        int maxSeqLen = 4096,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto)
    {
        _device = device;
        _engine = InferenceEngineType.BareMetal;
        _isFirstStage = isFirstStage;
        _isLastStage = isLastStage;
        _gpu = new GpuContext(device.Index);
        _model = new Qwen2GpuModel(
            _gpu,
            weights,
            maxSeqLen: maxSeqLen,
            kvPrecision: kvPrecision,
            startLayer: startLayer,
            layerCount: layerCount,
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
        _model.ForwardStage(token, pos, inputX, outputX, logits, computeLogits);
    }

    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        _model.ForwardBatchStage(tokens, startPos, inputXBatch, outputXBatch, logits, computeLogits);
    }

    public int SampleToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        return _isLastStage ? _model.SampleToken(options, recentTokens) : -1;
    }

    public void ResetKvCache()
    {
        _model.ResetKvCache();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _model.Dispose();
            _gpu.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
