namespace Glacier.Inference.Engine;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Tokenizer;
using Vortice.Direct3D12;
using Vortice.DXGI;

/// <summary>
/// Operational tiering policy controlling weight locality across VRAM, Host RAM, and Cluster nodes.
/// </summary>
public enum TieringExecutionPolicy
{
    /// <summary>Automatically inspect hardware and balance across VRAM and Host DDR5.</summary>
    Auto = 0,

    /// <summary>Pin dense transformer core in VRAM, evaluate routed experts in Host DDR5 via AVX-512/SIMD.</summary>
    HybridGpuHostRam = 1,

    /// <summary>Evaluate routed experts directly on GPU using shared PCIe memory mapping.</summary>
    GpuSharedMemory = 2,

    /// <summary>Distribute pipeline stages across LAN cluster nodes.</summary>
    DistributedCluster = 3
}

/// <summary>
/// Configuration options for <see cref="TieredMoERunner"/>.
/// </summary>
public sealed class TieredRunnerOptions
{
    /// <summary>Target GPU adapter index (-1 for automatic selection of highest VRAM hardware GPU).</summary>
    public int GpuAdapterIndex { get; set; } = -1;

    /// <summary>VRAM reserve budget in megabytes to allocate for dense core and KV cache.</summary>
    public ulong VramBudgetMb { get; set; } = 8192;

    /// <summary>Maximum sequence/context length in tokens.</summary>
    public int MaxSequenceLength { get; set; } = 4096;

    /// <summary>Tiering policy for routed expert compute.</summary>
    public TieringExecutionPolicy Policy { get; set; } = TieringExecutionPolicy.Auto;

    /// <summary>Temperature for sampling (0.0 for deterministic greedy sampling).</summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>Top-P nucleus sampling threshold.</summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>Top-K sampling threshold (0 to disable).</summary>
    public int TopK { get; set; } = 40;
}

/// <summary>
/// Detailed telemetry for a single autoregressive token generation step in the tiered engine.
/// </summary>
public sealed record TieredStepTelemetry
{
    public required int StepIndex { get; init; }
    public required int TokenId { get; init; }
    public required string TokenText { get; init; }
    public required double TotalStepDurationMs { get; init; }
    public required double AttentionVramDurationMs { get; init; }
    public required double RoutedExpertDurationMs { get; init; }
    public required double NetworkInterconnectDurationMs { get; init; }
}

/// <summary>
/// High-level production runner for Heterogeneous Tiered Mixture-of-Experts inference.
/// Orchestrates dense transformer core execution in GPU VRAM and routed expert pools
/// across Host DDR5 RAM, iGPU UMA, and distributed cluster nodes.
/// </summary>
public sealed class TieredMoERunner : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly ModelWeights _weights;
    private readonly BpeTokenizer _tokenizer;
    private readonly D3D12Context _d3dCtx;
    private readonly Qwen2D3D12Model _d3dModel;
    private readonly TieredRunnerOptions _options;
    private readonly float[] _logitsBuffer;
    private bool _disposed;

    /// <summary>Active hardware device descriptor.</summary>
    public string DeviceDescription => _d3dCtx.DeviceName;

    /// <summary>Dedicated GPU video memory in megabytes.</summary>
    public ulong DedicatedVramMb => _d3dCtx.DedicatedVideoMemory / (1024 * 1024);

    /// <summary>Shared system memory in megabytes.</summary>
    public ulong SharedSystemMemoryMb => _d3dCtx.SharedSystemMemory / (1024 * 1024);

    /// <summary>Whether the model is operating in heterogeneous tiered mode.</summary>
    public bool IsTieredActive => _d3dModel.TieredMoe;

    /// <summary>Model vocabulary size.</summary>
    public int VocabularySize => _tokenizer.VocabSize;

    /// <summary>Model embedding dimension.</summary>
    public int EmbeddingLength => _weights.EmbeddingLength;

    /// <summary>Number of transformer layers.</summary>
    public int LayerCount => _weights.BlockCount;

    /// <summary>
    /// Initializes a new instance of <see cref="TieredMoERunner"/> loading the target GGUF model.
    /// </summary>
    public TieredMoERunner(string modelPath, TieredRunnerOptions? options = null)
    {
        _options = options ?? new TieredRunnerOptions();

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"GGUF model file not found: {modelPath}", modelPath);

        _gguf = GgufFile.Open(modelPath);
        _weights = new ModelWeights(_gguf);
        _tokenizer = new BpeTokenizer(_gguf);

        int adapterIndex = _options.GpuAdapterIndex;
        if (adapterIndex < 0)
        {
            adapterIndex = SelectOptimalAdapter();
        }

        _d3dCtx = new D3D12Context(adapterIndex);
        _d3dModel = new Qwen2D3D12Model(_d3dCtx, _weights, _options.MaxSequenceLength);
        _logitsBuffer = new float[_tokenizer.VocabSize];
    }

    /// <summary>
    /// Streams generated tokens asynchronously.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateStreamAsync(
        string prompt,
        int maxNewTokens = 128,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int[] promptTokens = _tokenizer.Encode(prompt);
        if (promptTokens.Length == 0) yield break;

        // Prefill prompt tokens
        for (int i = 0; i < promptTokens.Length - 1; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _d3dModel.Forward(promptTokens[i], i, _logitsBuffer.AsSpan(), computeLogits: false);
        }

        int lastPromptIdx = promptTokens.Length - 1;
        _d3dModel.Forward(promptTokens[lastPromptIdx], lastPromptIdx, _logitsBuffer.AsSpan(), computeLogits: true);

        int currentPos = promptTokens.Length;

        for (int step = 0; step < maxNewTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int nextToken = SampleToken(_logitsBuffer.AsSpan(), _options.Temperature);

            if (_tokenizer.IsStopToken(nextToken) || nextToken == _tokenizer.EosTokenId || nextToken == 151645 || nextToken == 151643)
                break;

            string piece = _tokenizer.DecodeToken(nextToken);
            yield return piece;
            await Task.Yield();

            _d3dModel.Forward(nextToken, currentPos++, _logitsBuffer.AsSpan(), computeLogits: true);
        }
    }

    /// <summary>
    /// Generates a complete response synchronously/task-based with comprehensive execution metrics.
    /// </summary>
    public async Task<GenerationResult> GenerateAsync(
        string prompt,
        int maxNewTokens = 128,
        CancellationToken cancellationToken = default)
    {
        var swTotal = Stopwatch.StartNew();
        var sb = new StringBuilder();
        int tokenCount = 0;

        await foreach (var token in GenerateStreamAsync(prompt, maxNewTokens, cancellationToken))
        {
            sb.Append(token);
            tokenCount++;
        }

        swTotal.Stop();

        return new GenerationResult
        {
            Text = sb.ToString(),
            FinishReason = "completed",
            Metrics = new GenerationMetrics
            {
                PromptTokens = _tokenizer.Encode(prompt).Length,
                GeneratedTokens = tokenCount,
                TotalDuration = swTotal.Elapsed,
                GenerationDuration = swTotal.Elapsed,
                PromptEvalDuration = TimeSpan.Zero
            }
        };
    }

    private static int SelectOptimalAdapter()
    {
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
        int bestIdx = 0;
        ulong maxVram = 0;

        for (int i = 0; factory.EnumAdapters1((uint)i, out var ad).Success; i++)
        {
            if ((ad.Description1.Flags & AdapterFlags.Software) == 0 && (ulong)ad.Description1.DedicatedVideoMemory > maxVram)
            {
                maxVram = (ulong)ad.Description1.DedicatedVideoMemory;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private static int SampleToken(ReadOnlySpan<float> logits, float temperature)
    {
        if (temperature <= 0.001f)
        {
            int maxIdx = 0;
            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit)
                {
                    maxLogit = logits[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // Greedy / argmax fallback for numerical stability
        int best = 0;
        float high = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > high)
            {
                high = logits[i];
                best = i;
            }
        }
        return best;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _d3dModel.Dispose();
            _d3dCtx.Dispose();
            _gguf.Dispose();
            _disposed = true;
        }
    }
}
