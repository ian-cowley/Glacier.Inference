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
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Pipeline;
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
    private readonly ID3D12Model? _d3d12Model;
    private readonly ICpuModel? _cpuModel;
    private readonly KVCache? _kvCache;
    private readonly PipelineSession? _pipelineSession;
    private readonly int _maxSeqLen;
    private readonly BpeTokenizer _tokenizer;
    private readonly Sampler _sampler;
    private readonly float[] _logits;
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public ModelWeights Weights => _weights;
    public BpeTokenizer Tokenizer => _tokenizer;
    public KVCache? KVCache => _kvCache;
    public DeviceInfo Device { get; private set; }
    public InferenceEngineType Engine { get; private set; }
    public string ActiveDevice { get; private set; }
    public string ActiveBackend =>
        _gpuModel != null ? "Cuda" :
        _d3d12Model != null ? "Direct3D12" :
        (_pipelineSession != null && _pipelineSession.Stages.Count > 0
            ? (_pipelineSession.Stages[0].Engine == InferenceEngineType.BareMetal ? "Cuda"
               : _pipelineSession.Stages[0].Engine == InferenceEngineType.DirectML ? "Direct3D12"
               : "Cpu")
            : "Cpu");
    public Exception? FallbackException { get; private set; }
    public bool IsGpuAccelerated => _gpuModel != null || _d3d12Model != null || _pipelineSession != null;
    public bool IsPipelineAccelerated => _pipelineSession != null;
    public PipelineSession? PipelineSession => _pipelineSession;
    public KvCachePrecision KvPrecision => _gpuModel?.KvPrecision ?? KvCachePrecision.Fp32;
    public Qwen2GpuModel? GpuModel => _gpuModel;
    public ID3D12Model? D3D12Model => _d3d12Model;
    public ICpuModel? CpuModel => _cpuModel;
    public UniversalArchitecture Architecture => _weights.ArchitectureFamily;
    public int MaxSeqLen => _maxSeqLen;

    public InferenceSession(
        string modelPath,
        int maxSeqLen = 4096,
        string? device = null,
        InferenceEngineType engine = InferenceEngineType.Auto,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto,
        string? split = null)
    {
        _gguf = GgufFile.Open(modelPath);
        if (Embedding.EmbeddingGemma2Model.IsSupported(_gguf))
        {
            _gguf.Dispose();
            throw new NotSupportedException(
                "'gemma-embedding2' is an encoder-only embedding model. Use Glacier.Inference.Embedding.EmbeddingGemma2Model instead of InferenceSession.");
        }
        _weights = new ModelWeights(_gguf);
        _maxSeqLen = maxSeqLen;
        _tokenizer = new BpeTokenizer(_gguf);
        _sampler = new Sampler();
        _logits = new float[_weights.VocabSize];

        string? effectiveSplit = split;
        if (effectiveSplit == null && device != null && (device.Contains(',') || device.Contains(':') || device.Equals("auto", StringComparison.OrdinalIgnoreCase)))
        {
            effectiveSplit = device;
        }

        var stageSpecs = !string.IsNullOrEmpty(effectiveSplit)
            ? PipelineSplitConfig.ResolveStages(effectiveSplit, _weights)
            : null;

        if (stageSpecs != null && stageSpecs.Count > 1)
        {
            _pipelineSession = PipelineSession.Create(_weights, stageSpecs, maxSeqLen, kvPrecision);
            _kvCache = null;
            ActiveDevice = _pipelineSession.TopologyDescription;
            Device = stageSpecs[0].Device;
            Engine = stageSpecs[0].Engine;
        }
        else
        {
            var (targetDevice, targetEngine) = GlacierSettings.ResolveTarget(device, engine != InferenceEngineType.Auto ? engine.ToString() : null);
            Device = targetDevice;
            Engine = targetEngine;

            if (!_weights.IsHybridSsm && !_weights.IsMoe && !_weights.IsMla && !_weights.Layers[0].HasFusedQkv && targetEngine == InferenceEngineType.BareMetal && GpuContext.IsSupported && targetDevice.Vendor == GpuVendor.Nvidia)
            {
                try
                {
                    _gpu = new GpuContext(targetDevice.Index);
                    _gpuModel = new Qwen2GpuModel(_gpu, _weights, maxSeqLen, kvPrecision);
                    _kvCache = null; // GPU maintains all KV states in device VRAM
                    ActiveDevice = $"{targetDevice.Name} [Engine: Pure C# Bare-Metal SASS | KV: {_gpuModel.KvPrecision} | Arch: {_weights.ArchitectureFamily}]";
                }
                catch (Exception ex)
                {
                    GlacierDiagnostics.LogWarning($"GPU init failed: {ex.GetType().FullName}: {ex.Message}", ex);
                    FallbackException = ex;
                    var settings = GlacierSettings.Load();
                    if (!settings.FallbackToCpu)
                        throw new InvalidOperationException($"Failed to initialize Bare-Metal SASS inference on {targetDevice.Name}: {ex.Message}", ex);

                    _gpu?.Dispose();
                    _gpu = null;
                    _gpuModel = null;
                    _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen, _weights.ValueDim);
                    _cpuModel = CpuModelFactory.Create(_weights, maxSeqLen);
                    Device = DeviceManager.ResolveDevice("cpu");
                    Engine = InferenceEngineType.Cpu;
                    ActiveDevice = $"{Device.Name} [Fallback from Bare-Metal | Arch: {_weights.ArchitectureFamily}]";
                }
            }
#if !ANDROID
            else if (!_weights.IsMla && (_weights.IsHybridSsm || !_weights.Layers[0].HasFusedQkv) && (targetEngine == InferenceEngineType.BareMetal || targetEngine == InferenceEngineType.DirectML) &&
                     (targetDevice.Vendor == GpuVendor.Amd || targetDevice.Vendor == GpuVendor.Nvidia || targetDevice.Vendor == GpuVendor.Intel) && OperatingSystem.IsWindows())
            {
                try
                {
                    var d3dCtx = new D3D12Context(targetDevice.Index);
                    if (_weights.ArchitectureFamily == UniversalArchitecture.Gemma4)
                    {
                        _d3d12Model = new Gemma4D3D12Model(d3dCtx, _weights, maxSeqLen);
                        ActiveDevice = $"{targetDevice.Name} [Engine: Bare-Metal DirectX 12 Compute (Google Gemma 4 ISWA + Dual MoE) | KV: FP32 | Arch: Gemma 4]";
                    }
                    else if (_weights.IsHybridSsm || _weights.ArchitectureFamily == UniversalArchitecture.HybridSsm)
                    {
                        _d3d12Model = new Qwen3HybridD3D12Model(d3dCtx, _weights, maxSeqLen);
                        ActiveDevice = $"{targetDevice.Name} [Engine: Bare-Metal DirectX 12 Compute (Qwen 3.5/3.6 Gated DeltaNet SSM) | KV: FP32 | Arch: HybridSsm]";
                    }
                    else
                    {
                        _d3d12Model = new Qwen2D3D12Model(d3dCtx, _weights, maxSeqLen);
                        ActiveDevice = $"{targetDevice.Name} [Engine: Bare-Metal DirectX 12 Compute (HLSL Wave32{(_weights.IsMoe ? " MoE" : "")}) | KV: FP32 | Arch: {_weights.ArchitectureFamily}]";
                    }
                    _kvCache = null; // GPU maintains all KV states in device VRAM
                }
                catch (Exception ex)
                {
                    GlacierDiagnostics.LogWarning($"GPU init failed: {ex.GetType().FullName}: {ex.Message}", ex);
                    FallbackException = ex;
                    var settings = GlacierSettings.Load();
                    if (!settings.FallbackToCpu)
                        throw new InvalidOperationException($"Failed to initialize Direct3D 12 Compute inference on {targetDevice.Name}: {ex.Message}", ex);

                    _d3d12Model?.Dispose();
                    _d3d12Model = null;
                    _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen, _weights.ValueDim);
                    _cpuModel = CpuModelFactory.Create(_weights, maxSeqLen);
                    Device = DeviceManager.ResolveDevice("cpu");
                    Engine = InferenceEngineType.Cpu;
                    ActiveDevice = $"{Device.Name} [Fallback from Direct3D 12 | Arch: {_weights.ArchitectureFamily}]";
                }
            }
#endif
            else
            {
                var settings = GlacierSettings.Load();
                if (!settings.FallbackToCpu && targetDevice.Vendor != GpuVendor.Cpu && targetEngine != InferenceEngineType.Cpu)
                {
                    string reason = _weights.IsHybridSsm
                        ? $"Model uses Qwen 3.5/3.6 Gated DeltaNet (Hybrid SSM) architecture which executes via high-performance multi-threaded SIMD AVX2/AVX-512 engine. Specify '--device cpu' or enable CPU fallback."
                        : _weights.ArchitectureFamily == UniversalArchitecture.Gemma4
                            ? $"Model uses Google Gemma 4 (ISWA + Dual MoE) architecture which executes via high-performance multi-threaded SIMD AVX2/AVX-512 engine. Specify '--device cpu' or enable CPU fallback."
                            : targetDevice.Vendor switch
                            {
                                GpuVendor.Amd when !OperatingSystem.IsWindows() =>
                                    $"GPU hardware acceleration on {targetDevice.Name} currently requires Direct3D 12 on Windows. On Linux, please specify '--device cpu'.",
                                GpuVendor.Nvidia =>
                                    $"Bare-metal SASS engine is not available for {targetDevice.Name} or requires an unsupported model architecture variant.",
                                _ =>
                                    $"No compatible GPU hardware engine available for device '{targetDevice.Name}' under engine '{targetEngine}' on this operating system."
                            };
                    throw new InvalidOperationException($"Cannot run on {targetDevice.Name} with engine '{targetEngine}': {reason}");
                }

                _kvCache = new KVCache(_weights.BlockCount, _weights.HeadCountKv, _weights.HeadDim, maxSeqLen, _weights.ValueDim);
                _cpuModel = CpuModelFactory.Create(_weights, maxSeqLen);
                var cpuDevice = DeviceManager.ResolveDevice("cpu");
                if (targetDevice.Vendor != GpuVendor.Cpu || targetEngine != InferenceEngineType.Cpu)
                {
                    Device = cpuDevice;
                    Engine = InferenceEngineType.Cpu;
                }
                string cpuDeviceName = cpuDevice.Name;
                string prefix = targetDevice.Vendor != GpuVendor.Cpu
                    ? $"{cpuDeviceName} [Fallback from {targetDevice.Name} | "
                    : $"{targetDevice.Name} [";
                ActiveDevice = _weights.ArchitectureFamily == UniversalArchitecture.Gemma4
                    ? $"{prefix}Engine: Multi-threaded SIMD AVX2/AVX-512 (Gemma 4 ISWA + Dual MoE)]"
                    : _weights.IsMla
                        ? $"{prefix}Engine: Multi-threaded SIMD AVX2/AVX-512 (DeepSeek MLA)]"
                        : _weights.ArchitectureFamily == UniversalArchitecture.HybridSsm
                            ? $"{prefix}Engine: Multi-threaded SIMD AVX2/AVX-512 (Qwen Gated DeltaNet / Hybrid SSM)]"
                            : _weights.Layers[0].HasFusedQkv
                                ? $"{prefix}Engine: Multi-threaded SIMD AVX2/AVX-512 (Microsoft Phi Fused QKV/SwiGLU)]"
                                : _weights.IsMoe
                                    ? $"{prefix}Engine: Multi-threaded SIMD AVX2/AVX-512 MoE]"
                                    : $"{prefix}Engine: SIMD AVX2/AVX-512 Optimized ({_weights.ArchitectureFamily})]";
            }
        }
    }

    public InferenceSession(string modelPath, int maxSeqLen, InferenceDevice device, KvCachePrecision kvPrecision = KvCachePrecision.Auto, string? split = null)
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
        }, kvPrecision: kvPrecision, split: split)
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
        if (_pipelineSession != null)
            _pipelineSession.ResetKvCache();
        else
            _kvCache?.Reset();

        var totalStopwatch = Stopwatch.StartNew();
        var promptStopwatch = Stopwatch.StartNew();

        // 2. Prefill prompt tokens
        if (_pipelineSession != null)
        {
            _pipelineSession.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_gpuModel != null)
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

            // Sample next token
            int nextToken;
            if (_pipelineSession != null)
            {
                nextToken = _pipelineSession.SampleToken(options, CollectionsMarshal.AsSpan(recentTokens), _logits.AsSpan(), _sampler);
            }
            else if (_gpuModel != null && step > 0)
            {
                nextToken = _gpuModel.SampleToken(options, CollectionsMarshal.AsSpan(recentTokens));
            }
            else
            {
                nextToken = _sampler.Sample(_logits.AsSpan(), options, CollectionsMarshal.AsSpan(recentTokens));
            }
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
        if (_pipelineSession != null)
            _pipelineSession.ResetKvCache();
        else
            _kvCache?.Reset();
    }

    /// <summary>
    /// Evaluates prompt tokens and computes initial logits.
    /// </summary>
    public void Prefill(ReadOnlySpan<int> promptTokens)
    {
        if (promptTokens.IsEmpty) return;
        ResetKvCache();

        if (_pipelineSession != null)
        {
            _pipelineSession.ForwardBatch(promptTokens, 0, _logits.AsSpan(), computeLogits: true);
        }
        else if (_gpuModel != null)
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
        if (_pipelineSession != null)
            return _pipelineSession.SampleToken(options, recentTokens, _logits.AsSpan(), _sampler);

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

        if (_pipelineSession != null)
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                ForwardToken(tokens[t], startPos + t, computeLogits: true);
                predictedTokens[t] = _pipelineSession.SampleToken(SamplingOptions.Greedy, default, _logits.AsSpan(), _sampler);
            }
        }
        else if (_gpuModel != null)
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
        if (_pipelineSession != null)
        {
            _pipelineSession.Forward(token, pos, computeLogits ? _logits.AsSpan() : Span<float>.Empty, computeLogits);
        }
        else if (_gpuModel != null)
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

    public void AttachLora(string loraPath)
    {
        var lora = LoraAdapterWeights.Load(loraPath);
        if (_cpuModel != null)
        {
            _cpuModel.LoraWeights = lora;
        }
    }

    /// <summary>
    /// Extracts a normalized vector embedding for the given input text.
    /// Runs directly through the in-process hardware engine with zero IPC serialization.
    /// </summary>
    public void ExtractEmbedding(
        ReadOnlySpan<char> text,
        Span<float> destination,
        PoolingStrategy strategy = PoolingStrategy.LastToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (text.IsEmpty)
        {
            destination.Slice(0, Math.Min(destination.Length, _weights.EmbeddingLength)).Clear();
            return;
        }

        int[] tokens = _tokenizer.Encode(text.ToString());
        ExtractEmbedding(tokens, destination, strategy);
    }

    /// <summary>
    /// Extracts a normalized vector embedding for pre-tokenized inputs.
    /// </summary>
    public void ExtractEmbedding(
        ReadOnlySpan<int> tokens,
        Span<float> destination,
        PoolingStrategy strategy = PoolingStrategy.LastToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_pipelineSession != null)
            _pipelineSession.ResetKvCache();
        else
            _kvCache?.Reset();

        if (_gpuModel != null)
        {
            _gpuModel.ExtractEmbedding(tokens, destination, strategy);
        }
        else if (_cpuModel != null && _kvCache != null)
        {
            _cpuModel.ExtractEmbedding(tokens, destination, _kvCache, strategy);
        }
        else
        {
            throw new NotSupportedException("Embedding extraction is not supported on the active engine configuration.");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _pipelineSession?.Dispose();
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
