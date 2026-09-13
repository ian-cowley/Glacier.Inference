namespace Glacier.Inference.Engine;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Config;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Target computing hardware for inference execution.
/// </summary>
public enum InferenceDevice
{
    Auto,
    Gpu,
    Cpu
}

/// <summary>
/// Telemetry metrics for a generation session.
/// </summary>
public sealed record GenerationMetrics
{
    public int PromptTokens { get; init; }
    public int GeneratedTokens { get; init; }
    public TimeSpan PromptEvalDuration { get; init; }
    public TimeSpan GenerationDuration { get; init; }
    public TimeSpan TotalDuration { get; init; }

    public double PromptTokensPerSecond =>
        PromptEvalDuration.TotalSeconds > 0 ? PromptTokens / PromptEvalDuration.TotalSeconds : 0;

    public double GenerationTokensPerSecond =>
        GenerationDuration.TotalSeconds > 0 ? GeneratedTokens / GenerationDuration.TotalSeconds : 0;
}

/// <summary>
/// Result of an autoregressive inference generation.
/// </summary>
public sealed record GenerationResult
{
    public required string Text { get; init; }
    public required GenerationMetrics Metrics { get; init; }
    public required string FinishReason { get; init; }
}

/// <summary>
/// High-performance LLM generation session managing KV cache, model forward passes, and token streaming.
/// Supports both Bare-Metal GPU execution on NVIDIA GPUs and multi-threaded SIMD CPU fallback.
/// </summary>
public sealed class InferenceSession : IDisposable, ISpeculativeTarget
{
    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly GpuContext? _gpu;
    private readonly Qwen2GpuModel? _gpuModel;
    private readonly Qwen2D3D12Model? _d3d12Model;
    private readonly Qwen2Model? _cpuModel;
    private readonly KVCache? _kvCache;
    private readonly int _maxSeqLen;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;
    private readonly float[] _logits;
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public KVCache? KVCache => _kvCache;
    public DeviceInfo Device { get; }
    public InferenceEngineType Engine { get; }
    public string ActiveDevice { get; }
    public bool IsGpuAccelerated => _gpuModel != null || _d3d12Model != null;
    public KvCachePrecision KvPrecision => _gpuModel?.KvPrecision ?? KvCachePrecision.Fp32;
    public Qwen2GpuModel? GpuModel => _gpuModel;
    public Qwen2D3D12Model? D3D12Model => _d3d12Model;
    public Qwen2Model? CpuModel => _cpuModel;
    public int MaxSeqLen => _maxSeqLen;

    public InferenceSession(
        string modelPath,
        int maxSeqLen = 4096,
        string? device = null,
        InferenceEngineType engine = InferenceEngineType.Auto,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto)
    {
        _gguf = GgufFile.Open(modelPath);
        _weights = new ModelWeights(_gguf);
        _maxSeqLen = maxSeqLen;
        _tokenizer = new BpeTokenizer(_gguf);
        _sampler = new Sampler();
        _logits = new float[_weights.VocabSize];

        var (targetDevice, targetEngine) = GlacierSettings.ResolveTarget(device, engine != InferenceEngineType.Auto ? engine.ToString() : null);
        Device = targetDevice;
        Engine = targetEngine;

        if (targetEngine == InferenceEngineType.BareMetal && GpuContext.IsSupported && targetDevice.Vendor == GpuVendor.Nvidia)
        {
            try
            {
                _gpu = new GpuContext(targetDevice.Index);
                _gpuModel = new Qwen2GpuModel(_gpu, _weights, maxSeqLen, kvPrecision);
                _kvCache = null; // GPU maintains all KV states in device VRAM
                ActiveDevice = $"{targetDevice.Name} [Engine: Pure C# Bare-Metal SASS | KV: {_gpuModel.KvPrecision}]";
            }
            catch (Exception ex)
            {
                var settings = GlacierSettings.Load();
                if (!settings.FallbackToCpu)
                    throw new InvalidOperationException($"Failed to initialize Bare-Metal SASS inference on {targetDevice.Name}: {ex.Message}", ex);

                _gpu?.Dispose();
                _gpu = null;
                _gpuModel = null;
                _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen);
                _cpuModel = new Qwen2Model(_weights, maxSeqLen);
                ActiveDevice = $"{DeviceManager.ResolveDevice("cpu").Name} [Fallback from Bare-Metal]";
            }
        }
        else if ((targetEngine == InferenceEngineType.BareMetal || targetEngine == InferenceEngineType.DirectML) &&
                 targetDevice.Vendor == GpuVendor.Amd && OperatingSystem.IsWindows())
        {
            try
            {
                var d3dCtx = new D3D12Context(targetDevice.Index);
                _d3d12Model = new Qwen2D3D12Model(d3dCtx, _weights, maxSeqLen);
                _kvCache = null; // GPU maintains all KV states in device VRAM
                ActiveDevice = $"{targetDevice.Name} [Engine: Bare-Metal DirectX 12 Compute (HLSL Wave32) | KV: FP32]";
            }
            catch (Exception ex)
            {
                var settings = GlacierSettings.Load();
                if (!settings.FallbackToCpu)
                    throw new InvalidOperationException($"Failed to initialize Direct3D 12 Compute inference on {targetDevice.Name}: {ex.Message}", ex);

                _d3d12Model?.Dispose();
                _d3d12Model = null;
                _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen);
                _cpuModel = new Qwen2Model(_weights, maxSeqLen);
                ActiveDevice = $"{DeviceManager.ResolveDevice("cpu").Name} [Fallback from Direct3D 12]";
            }
        }
        else
        {
            _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen);
            _cpuModel = new Qwen2Model(_weights, maxSeqLen);
            ActiveDevice = $"{targetDevice.Name} [Engine: SIMD AVX2 Optimized (Batched GEMM)]";
        }
    }

    public InferenceSession(string modelPath, int maxSeqLen, InferenceDevice device, KvCachePrecision kvPrecision = KvCachePrecision.Auto)
        : this(modelPath, maxSeqLen, device switch
        {
            InferenceDevice.Gpu => "gpu",
            InferenceDevice.Cpu => "cpu",
            _ => "auto"
        }, device switch
        {
            InferenceDevice.Gpu => InferenceEngineType.BareMetal,
            InferenceDevice.Cpu => InferenceEngineType.Cpu,
            _ => InferenceEngineType.Auto
        }, kvPrecision: kvPrecision)
    {
    }

    /// <summary>
    /// Generates text autoregressively with real-time token streaming callback.
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(
        string prompt,
        SamplingOptions? options = null,
        bool formatChat = true,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        options ??= new SamplingOptions();

        // 1. Format and encode prompt
        string formattedPrompt = formatChat ? _tokenizer.FormatChatML(prompt) : prompt;
        int[] promptTokens = _tokenizer.Encode(formattedPrompt);

        if (promptTokens.Length == 0)
        {
            return new GenerationResult
            {
                Text = "",
                Metrics = new GenerationMetrics(),
                FinishReason = "empty_prompt"
            };
        }

        // Reset KV cache if active
        _kvCache?.Reset();

        var totalStopwatch = Stopwatch.StartNew();
        var promptStopwatch = Stopwatch.StartNew();

        // 2. Prefill prompt tokens
        if (_gpuModel != null)
        {
            _gpuModel.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_d3d12Model != null)
        {
            _d3d12Model.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_cpuModel != null && _kvCache != null)
        {
            _cpuModel.ForwardBatch(promptTokens, 0, _logits.AsSpan(), _kvCache, computeLogits: true);
        }
        else
        {
            for (int i = 0; i < promptTokens.Length - 1; i++)
            {
                ct.ThrowIfCancellationRequested();
                ForwardToken(promptTokens[i], i, computeLogits: false);
            }

            // Forward last prompt token to get first logits
            int lastPromptIdx = promptTokens.Length - 1;
            ForwardToken(promptTokens[lastPromptIdx], lastPromptIdx, computeLogits: true);
        }
        promptStopwatch.Stop();

        // 3. Autoregressive token generation loop
        var genStopwatch = Stopwatch.StartNew();
        var recentTokens = new List<int>(options.MaxTokens + 16);
        var responseSb = new StringBuilder();
        string finishReason = "length";

        int maxSeq = _gpuModel != null ? _maxSeqLen : (_d3d12Model != null ? _maxSeqLen : (_kvCache?.MaxSeqLen ?? _maxSeqLen));
        int currentPos = promptTokens.Length;
        for (int step = 0; step < options.MaxTokens && currentPos < maxSeq; step++)
        {
            ct.ThrowIfCancellationRequested();

            // Sample next token (pure GPU reduction in ~3 us for GPU model; CPU sampler for CPU model)
            int nextToken = (_gpuModel != null && step > 0)
                ? _gpuModel.SampleToken(options, CollectionsMarshal.AsSpan(recentTokens))
                : _sampler.Sample(_logits.AsSpan(), options, CollectionsMarshal.AsSpan(recentTokens));
            recentTokens.Add(nextToken);

            // Check for stop tokens
            if (_tokenizer.IsStopToken(nextToken) || nextToken == _tokenizer.EosTokenId || nextToken == 151645 || nextToken == 151643)
            {
                finishReason = $"stop({nextToken})";
                break;
            }

            // Decode token piece
            string piece = _tokenizer.DecodeToken(nextToken);
            responseSb.Append(piece);
            onToken?.Invoke(piece);

            // Forward next token
            ForwardToken(nextToken, currentPos, computeLogits: true);
            currentPos++;

            // Yield control briefly
            if ((step & 15) == 0)
            {
                await Task.Yield();
            }
        }

        genStopwatch.Stop();
        totalStopwatch.Stop();

        return new GenerationResult
        {
            Text = responseSb.ToString(),
            FinishReason = finishReason,
            Metrics = new GenerationMetrics
            {
                PromptTokens = promptTokens.Length,
                GeneratedTokens = recentTokens.Count,
                PromptEvalDuration = promptStopwatch.Elapsed,
                GenerationDuration = genStopwatch.Elapsed,
                TotalDuration = totalStopwatch.Elapsed
            }
        };
    }

    /// <summary>
    /// Resets the key-value cache.
    /// </summary>
    public void ResetKvCache()
    {
        _kvCache?.Reset();
    }

    /// <summary>
    /// Evaluates prompt tokens and computes initial logits.
    /// </summary>
    public void Prefill(ReadOnlySpan<int> promptTokens)
    {
        if (promptTokens.IsEmpty) return;
        ResetKvCache();

        if (_gpuModel != null)
        {
            _gpuModel.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_d3d12Model != null)
        {
            _d3d12Model.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_cpuModel != null && _kvCache != null)
        {
            _cpuModel.ForwardBatch(promptTokens, 0, _logits.AsSpan(), _kvCache, computeLogits: true);
        }
        else
        {
            for (int i = 0; i < promptTokens.Length - 1; i++)
            {
                ForwardToken(promptTokens[i], i, computeLogits: false);
            }
            ForwardToken(promptTokens[^1], promptTokens.Length - 1, computeLogits: true);
        }
    }

    /// <summary>
    /// Samples the next token using the active sampler or GPU reduction kernel.
    /// </summary>
    public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        return _gpuModel != null
            ? _gpuModel.SampleToken(options, recentTokens)
            : _sampler.Sample(_logits.AsSpan(), options, recentTokens);
    }

    /// <summary>
    /// Executes batched candidate token verification.
    /// In GPU mode, evaluates all candidate positions in a single batched transformer pass.
    /// In CPU mode, verifies candidates with cached attention.
    /// </summary>
    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        if (tokens.IsEmpty) return;

        if (_gpuModel != null)
        {
            _gpuModel.VerifyBatch(tokens, startPos, predictedTokens);
        }
        else
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                ForwardToken(tokens[t], startPos + t, computeLogits: true);
                predictedTokens[t] = _sampler.Sample(_logits.AsSpan(), SamplingOptions.Greedy);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ForwardToken(int token, int pos, bool computeLogits)
    {
        if (_gpuModel != null)
        {
            _gpuModel.Forward(token, pos, null, Span<float>.Empty, computeLogits);
        }
        else if (_d3d12Model != null)
        {
            _d3d12Model.Forward(token, pos, computeLogits ? _logits.AsSpan() : Span<float>.Empty, computeLogits);
        }
        else
        {
            _cpuModel!.Forward(token, pos, _kvCache!, _logits.AsSpan(), computeLogits);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gpuModel?.Dispose();
            _gpu?.Dispose();
            _d3d12Model?.Dispose();
            _cpuModel?.Dispose();
            _kvCache?.Dispose();
            _gguf.Dispose();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~InferenceSession()
    {
        Dispose();
    }
}
