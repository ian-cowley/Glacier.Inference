namespace Glacier.Inference.Pipeline;

using System;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;

/// <summary>
/// Direct3D 12 pipeline stage executing transformer layers on AMD Radeon or Intel GPUs via HLSL Wave32 compute.
/// </summary>
public sealed class D3D12PipelineStage : IPipelineStage
{
    private readonly D3D12Context _ctx;
    private readonly Qwen2D3D12Model _model;
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
    public Qwen2D3D12Model Model => _model;

    public D3D12PipelineStage(
        DeviceInfo device,
        ModelWeights weights,
        int startLayer,
        int layerCount,
        bool isFirstStage,
        bool isLastStage,
        int maxSeqLen = 4096)
    {
        _device = device;
        _engine = InferenceEngineType.DirectML;
        _isFirstStage = isFirstStage;
        _isLastStage = isLastStage;
        _ctx = new D3D12Context(device.Index);
        _model = new Qwen2D3D12Model(
            _ctx,
            weights,
            maxSeqLen: maxSeqLen,
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
            _ctx.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
