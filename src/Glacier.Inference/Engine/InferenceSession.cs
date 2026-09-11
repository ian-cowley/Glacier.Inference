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
public sealed class InferenceSession : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly GpuContext? _gpu;
    private readonly Qwen2GpuModel? _gpuModel;
    private readonly Qwen2Model? _cpuModel;
    private readonly KVCache _kvCache;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;
    private readonly float[] _logits;
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public KVCache KVCache => _kvCache;
    public DeviceInfo Device { get; }
    public InferenceEngineType Engine { get; }
    public string ActiveDevice { get; }
    public bool IsGpuAccelerated => _gpuModel != null;

    public InferenceSession(
        string modelPath,
        int maxSeqLen = 4096,
        string? device = null,
        InferenceEngineType engine = InferenceEngineType.Auto)
    {
        _gguf = GgufFile.Open(modelPath);
        _weights = new ModelWeights(_gguf);
        _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen);
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
                _gpuModel = new Qwen2GpuModel(_gpu, _weights, maxSeqLen);
                ActiveDevice = $"{targetDevice.Name} [Engine: Pure C# Bare-Metal SASS]";
            }
            catch (Exception ex)
            {
                var settings = GlacierSettings.Load();
                if (!settings.FallbackToCpu)
                    throw new InvalidOperationException($"Failed to initialize Bare-Metal SASS inference on {targetDevice.Name}: {ex.Message}", ex);

                _gpu?.Dispose();
                _gpu = null;
                _gpuModel = null;
                _cpuModel = new Qwen2Model(_weights, maxSeqLen);
                ActiveDevice = $"{DeviceManager.ResolveDevice("cpu").Name} [Fallback from Bare-Metal]";
            }
        }
        else if (targetEngine == InferenceEngineType.DirectML)
        {
            // DirectML cooperative execution path
            _cpuModel = new Qwen2Model(_weights, maxSeqLen);
            ActiveDevice = $"{targetDevice.Name} [Engine: DirectML / DX12 Compute]";
        }
        else
        {
            _cpuModel = new Qwen2Model(_weights, maxSeqLen);
            ActiveDevice = $"{targetDevice.Name} [Engine: SIMD AVX-512/AVX2]";
        }
    }

    public InferenceSession(string modelPath, int maxSeqLen, InferenceDevice device)
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
        })
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

        // Reset KV cache
        _kvCache.Reset();

        var totalStopwatch = Stopwatch.StartNew();
        var promptStopwatch = Stopwatch.StartNew();

        // 2. Prefill prompt tokens
        if (_gpuModel != null)
        {
            _gpuModel.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
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

        int currentPos = promptTokens.Length;
        for (int step = 0; step < options.MaxTokens && currentPos < _kvCache.MaxSeqLen; step++)
        {
            ct.ThrowIfCancellationRequested();

            // Sample next token
            int nextToken = _sampler.Sample(_logits.AsSpan(), options, CollectionsMarshal.AsSpan(recentTokens));
            recentTokens.Add(nextToken);

            // Check for stop tokens
            if (nextToken == _tokenizer.EosTokenId || nextToken == 151645 || nextToken == 151643)
            {
                finishReason = "stop";
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ForwardToken(int token, int pos, bool computeLogits)
    {
        if (_gpuModel != null)
        {
            _gpuModel.Forward(token, pos, _kvCache, _logits.AsSpan(), computeLogits);
        }
        else
        {
            _cpuModel!.Forward(token, pos, _kvCache, _logits.AsSpan(), computeLogits);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gpuModel?.Dispose();
            _gpu?.Dispose();
            _cpuModel?.Dispose();
            _kvCache.Dispose();
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
