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
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Scheduling;
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
    private int _routeTailSkipMinRank;

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

    /// <summary>
    /// Optional OS/WDDM VRAM safety reserve in megabytes.
    /// Null uses automatic WDDM guard sizing; 0 explicitly disables the safety reserve without clamping.
    /// </summary>
    public int? VramReserveMb { get; set; } = null;

    /// <summary>KV-cache quantization precision (default: Auto).</summary>
    public KvCachePrecision KvPrecision { get; set; } = KvCachePrecision.Auto;

    /// <summary>
    /// Opt-in minimum 0-based router rank threshold for skipping VRAM-missed routed experts
    /// that rank in the bottom tail for every token in a speculative draft batch.
    /// Default is 0 (disabled).
    /// </summary>
    public int RouteTailSkipMinRank
    {
        get => _routeTailSkipMinRank;
        set => _routeTailSkipMinRank = value < 0 ? 0 : value;
    }

    /// <summary>Optional host-RAM expert arena budget in bytes (null uses hardware-recommended budget).</summary>
    public ulong? HostArenaBudgetBytes { get; set; } = null;

    /// <summary>Optional VRAM expert cache slot count override (null uses WDDM-safe headroom sizing).</summary>
    public int? VramExpertSlotCount { get; set; } = null;

    /// <summary>Optional file path to load/save the <see cref="ExpertRoutingProfile"/>.</summary>
    public string? RoutingProfilePath { get; set; } = null;

    /// <summary>Optional co-scheduler override (null creates a hardware-calibrated scheduler).</summary>
    public BandwidthCoScheduler? CoScheduler { get; set; } = null;

    /// <summary>Optional callback invoked with <see cref="TieredStepTelemetry"/> on every executed step/batch.</summary>
    public Action<TieredStepTelemetry>? OnStepTelemetry { get; set; } = null;
}

/// <summary>
/// Detailed telemetry for a single autoregressive token generation step or speculative batch in the tiered engine.
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

    public double ExpertsSkippedPercent { get; init; }
    public int TotalRoutedExperts { get; init; }
    public int SkippedRoutedExperts { get; init; }
    public int VramHitExperts { get; init; }
    public int PcieTransferredExperts { get; init; }
    public int CpuExecutedExperts { get; init; }
    public TieringExecutionPolicy ResolvedPolicy { get; init; }
}

/// <summary>
/// High-level production runner for Heterogeneous Tiered Mixture-of-Experts inference.
/// Orchestrates dense transformer core execution in GPU VRAM and routed expert pools
/// across Host DDR5 RAM (<see cref="HostRamExpertArena"/>), PCIe shared memory, and
/// adaptive <see cref="ExpertRoutingProfile"/> / <see cref="BandwidthCoScheduler"/> splitting.
/// </summary>
public sealed unsafe class TieredMoERunner : ISpeculativeTarget
{
    private const ulong SlotSubAlignmentBytes = 256UL;

    private readonly GgufFile? _gguf;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly D3D12Context? _d3dCtx;
    private readonly Qwen2D3D12Model? _d3dModel;
    private readonly Qwen2Model? _cpuModel;
    private readonly KVCache? _kvCache;
    private readonly TieredRunnerOptions _options;
    private readonly float[] _logitsBuffer;
    private readonly Sampler _sampler = new();

    private readonly ExpertRoutingProfile _routingProfile;
    private readonly HostRamExpertArena? _hostArena;
    private readonly BandwidthCoScheduler _coScheduler;
    private GpuResidentSlotManager? _slotManager;
    private readonly TieredMoEMissSplitter _missSplitter = new();
    private readonly TieringExecutionPolicy _resolvedPolicy;
    private readonly List<TieredStepTelemetry> _telemetryHistory = new();
    private readonly IReadOnlyList<TieredStepTelemetry> _telemetryHistoryReadOnly;
    private TieredStepTelemetry? _lastStepTelemetry;

    // D3D12 tiered MoE VRAM slot pool & PCIe Upload staging
    private readonly List<ID3D12Resource> _dVramSlotSegments = new();
    private int _slotsPerSegment;
    private ulong _maxAlignedSlotBytes;
    private ulong[]? _layerGateAlignedBytes;
    private ulong[]? _layerUpAlignedBytes;
    private ulong[]? _layerDownAlignedBytes;
    private GgufType[]? _layerGateGpuType;
    private GgufType[]? _layerUpGpuType;
    private GgufType[]? _layerDownGpuType;
    private bool[]? _layerHasFusedExps;
    private bool[]? _layerHasBiases;
    private ID3D12Resource? _uploadExpertPcieStaging;
    private byte* _pUploadExpertPcieStaging;
    private int _pcieStagingCapacity;
    private Qwen2D3D12Model.ExpertCpuWorkspace[]? _cpuWorkspaces;
    private float[]? _cpuNormXSums;

    // Synthetic/in-memory MoE execution state (used by internal test constructor)
    private readonly LayerWeights[]? _syntheticLayers;
    private readonly float[]? _syntheticLmHead;
    private readonly int _syntheticDim;
    private readonly int _syntheticExpertFfnDim;
    private readonly int _syntheticNumExperts;
    private readonly int _syntheticTopK;
    private readonly int _syntheticVocabSize;
    private readonly bool _syntheticNormTopK;
    private float[]? _syntheticNormScratch;
    private float[]? _syntheticNormSums;
    private float[]? _syntheticOutScratch;

    // Step telemetry accumulators across layers
    private int _stepCounter;
    private int _stepTotalRouted;
    private int _stepSkippedRouted;
    private int _stepVramHits;
    private int _stepPcieTransferred;
    private int _stepCpuExecuted;
    private double _stepRoutedExpertMs;
    private bool _disposed;

    /// <summary>Active hardware device descriptor.</summary>
    public string DeviceDescription => _d3dCtx?.DeviceName ?? "CPU-TieredMoE";

    /// <summary>Dedicated GPU video memory in megabytes.</summary>
    public ulong DedicatedVramMb => _d3dCtx != null ? _d3dCtx.DedicatedVideoMemory / (1024 * 1024) : 0UL;

    /// <summary>Shared system memory in megabytes.</summary>
    public ulong SharedSystemMemoryMb => _d3dCtx != null ? _d3dCtx.SharedSystemMemory / (1024 * 1024) : 0UL;

    /// <summary>Whether the model is operating in heterogeneous tiered mode.</summary>
    public bool IsTieredActive => _d3dModel?.TieredMoe ?? true;

    /// <summary>Model vocabulary size.</summary>
    public int VocabularySize => _tokenizer?.VocabSize ?? _syntheticVocabSize;

    /// <summary>Model embedding dimension.</summary>
    public int EmbeddingLength => _weights?.EmbeddingLength ?? _syntheticDim;

    /// <summary>Number of transformer layers.</summary>
    public int LayerCount => _weights?.BlockCount ?? _syntheticLayers?.Length ?? 0;

    /// <summary>Resolved KV-cache precision in VRAM.</summary>
    public KvCachePrecision KvPrecision =>
        _d3dModel?.KvPrecision
        ?? _kvCache?.Precision
        ?? KvCachePrecision.Fp32;

    /// <summary>Effective VRAM safety reserve in bytes honored by the underlying D3D12 model.</summary>
    public ulong EffectiveVramReserveBytes =>
        _d3dModel?.EffectiveVramReserveBytes
        ?? (WddmVramGuard.ResolveExplicitReserveBytes(_options.VramReserveMb) ?? 0UL);

    /// <summary>Empirical routing frequency profile shared across MoE layers.</summary>
    public ExpertRoutingProfile RoutingProfile => _routingProfile;

    /// <summary>Pinned Host-RAM expert arena (null when non-MoE or zero host budget).</summary>
    public HostRamExpertArena? HostArena => _weights?.ExpertArena ?? _hostArena;

    /// <summary>Dynamic bandwidth-adaptive CPU/PCIe co-scheduler.</summary>
    public BandwidthCoScheduler CoScheduler => _coScheduler;

    /// <summary>GPU VRAM expert slot cache manager (null when 0 VRAM expert slots are configured).</summary>
    public GpuResidentSlotManager? SlotManager => _slotManager;

    /// <summary>Per-batch MoE miss splitter &amp; speculative tail-skip planner.</summary>
    public TieredMoEMissSplitter MissSplitter => _missSplitter;

    /// <summary>Effective tiering policy resolved for this runner.</summary>
    public TieringExecutionPolicy ResolvedPolicy => _resolvedPolicy;

    /// <summary>Most recently emitted step telemetry, if any.</summary>
    public TieredStepTelemetry? LastStepTelemetry => _lastStepTelemetry;

    /// <summary>Chronological read-only view of all emitted step telemetry records.</summary>
    public IReadOnlyList<TieredStepTelemetry> TelemetryHistory => _telemetryHistoryReadOnly;

    // ISpeculativeTarget implementation
    public int MaxSeqLen => _options.MaxSequenceLength;
    public BpeTokenizer Tokenizer => _tokenizer ?? throw new NotSupportedException("Synthetic TieredMoERunner does not expose a BpeTokenizer.");
    public int VocabSize => VocabularySize;
    public bool SupportsLogitVerification => true;
    public bool CanOverwriteKvFromPrefix => _weights == null || !_weights.IsHybridSsm;

    /// <summary>Queries live WDDM VRAM headroom honoring <see cref="TieredRunnerOptions.VramReserveMb"/>.</summary>
    public VramHeadroomSnapshot QueryVramHeadroom()
    {
        ulong? explicitReserve = WddmVramGuard.ResolveExplicitReserveBytes(_options.VramReserveMb);
        if (_d3dCtx != null)
        {
            return _d3dCtx.QueryVramHeadroom(explicitReserve);
        }

        return WddmVramGuard.EvaluateHeadroom(
            0UL,
            0UL,
            0UL,
            OperatingSystem.IsWindows(),
            isUnifiedMemory: false,
            explicitReserveBytes: explicitReserve);
    }

    /// <summary>
    /// Initializes a new instance of <see cref="TieredMoERunner"/> loading the target GGUF model.
    /// </summary>
    public TieredMoERunner(string modelPath, TieredRunnerOptions? options = null)
    {
        _options = options ?? new TieredRunnerOptions();
        _telemetryHistoryReadOnly = _telemetryHistory.AsReadOnly();
        if (_options.Policy == TieringExecutionPolicy.DistributedCluster)
            throw new NotSupportedException("DistributedCluster policy requires cluster node endpoints.");

        ulong? explicitReserveBytes = WddmVramGuard.ResolveExplicitReserveBytes(_options.VramReserveMb);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"GGUF model file not found: {modelPath}", modelPath);

        _gguf = GgufFile.Open(modelPath);
        _weights = new ModelWeights(_gguf);
        _tokenizer = new BpeTokenizer(_gguf);
        _logitsBuffer = new float[_tokenizer.VocabSize];

        _resolvedPolicy = _options.Policy;

        var hwSnapshot = HardwareRuntimeProfiler.CaptureSnapshot();
        var hwPolicy = HardwareRuntimeProfiler.RecommendPolicy(hwSnapshot, _weights);
        _coScheduler = _options.CoScheduler ?? HardwareRuntimeProfiler.CreateCalibratedCoScheduler(hwSnapshot);

        int layerCount = Math.Max(1, _weights.BlockCount);
        int expertCount = Math.Max(1, _weights.ExpertCount);

        if (!string.IsNullOrWhiteSpace(_options.RoutingProfilePath) &&
            ExpertRoutingProfile.TryLoad(_options.RoutingProfilePath, layerCount, expertCount, out var loadedProfile) &&
            loadedProfile != null)
        {
            _routingProfile = loadedProfile;
        }
        else
        {
            _routingProfile = new ExpertRoutingProfile(layerCount, expertCount);
        }

        if (_weights.IsMoe)
        {
            ulong arenaBudget = _options.HostArenaBudgetBytes ?? hwPolicy.RecommendedHostArenaBytes;
            if (arenaBudget > 0)
            {
                _hostArena = _weights.AttachExpertArena(arenaBudget, preferUnbufferedIo: true);
            }
        }

        bool useD3D12 = D3D12Context.IsSupported && _options.GpuAdapterIndex >= -1;
        if (useD3D12)
        {
            int adapterIndex = _options.GpuAdapterIndex;
            if (adapterIndex < 0)
            {
                adapterIndex = SelectOptimalAdapter();
            }

            _d3dCtx = new D3D12Context(adapterIndex);

            // Always keep tiered MoE enabled in TieredMoERunner so TieredMoEMissSplitter,
            // ExpertRoutingProfile, RouteTailSkipMinRank, and TieredStepTelemetry execute
            // consistently even under default Auto options and for fused GateUp models.
            bool? forceTiered = _weights.IsMoe ? true : null;

            _d3dModel = new Qwen2D3D12Model(
                _d3dCtx,
                _weights,
                _options.MaxSequenceLength,
                kvPrecision: _options.KvPrecision,
                tieredMoe: forceTiered,
                explicitVramReserveBytes: explicitReserveBytes);

            if (_weights.IsMoe && _d3dModel.TieredMoe)
            {
                InitializeD3D12TieredExpertResources();
                _d3dModel.TieredMoeBatchHandler = HandleD3D12TieredMoeLayerBatch;
            }
        }
        else
        {
            _cpuModel = new Qwen2Model(_weights, _options.MaxSequenceLength);
            _kvCache = new KVCache(
                _weights.BlockCount,
                _weights.HeadCountKv,
                _weights.HeadDim,
                _options.MaxSequenceLength,
                _weights.ValueDim,
                precision: _options.KvPrecision);

            if (_weights.IsMoe)
            {
                LayerWeights? firstMoeLayer = null;
                for (int ml = 0; ml < _weights.BlockCount; ml++)
                {
                    if (_weights.Layers[ml] is { IsMoe: true } candidate)
                    {
                        firstMoeLayer = candidate;
                        break;
                    }
                }

                int topK = Math.Max(1, _weights.ExpertUsedCount);
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int dim = _weights.EmbeddingLength;
                _cpuWorkspaces = new Qwen2D3D12Model.ExpertCpuWorkspace[topK];
                for (int i = 0; i < topK; i++)
                {
                    _cpuWorkspaces[i] = new Qwen2D3D12Model.ExpertCpuWorkspace(expertFfnDim, dim);
                }
                _cpuNormXSums = new float[(dim + 31) / 32];

                if (firstMoeLayer != null && (_options.VramExpertSlotCount ?? 0) > 0)
                {
                    long expBytes = TieredMoEMissSplitter.EstimateExpertByteSize(
                        firstMoeLayer,
                        dim,
                        expertFfnDim);
                    _slotManager = new GpuResidentSlotManager(
                        _options.VramExpertSlotCount!.Value,
                        (nuint)Math.Max(256L, expBytes),
                        allocateVram: false,
                        SlotEvictionPolicy.DecayedFrequency,
                        _routingProfile);
                }

                _cpuModel.TieredMoeBatchHandler = HandleCpuTieredMoeLayerBatch;
            }
        }
    }

    /// <summary>
    /// Internal synthetic/in-memory constructor for deterministic zero-GGUF/zero-GPU unit verification
    /// using genuine <see cref="QuantKernels"/> MoE math, <see cref="ExpertRoutingProfile"/>,
    /// <see cref="HostRamExpertArena"/>, <see cref="GpuResidentSlotManager"/>, and <see cref="BandwidthCoScheduler"/>.
    /// </summary>
    internal TieredMoERunner(
        LayerWeights[] layers,
        int dim,
        int expertFfnDim,
        int numExperts,
        int topK,
        int vocabSize,
        bool normTopK,
        ExpertRoutingProfile routingProfile,
        BandwidthCoScheduler coScheduler,
        GpuResidentSlotManager? slotManager,
        HostRamExpertArena? hostArena,
        TieredRunnerOptions? options = null,
        float[]? lmHeadWeights = null,
        BpeTokenizer? tokenizer = null)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(routingProfile);
        ArgumentNullException.ThrowIfNull(coScheduler);

        _options = options ?? new TieredRunnerOptions();
        _telemetryHistoryReadOnly = _telemetryHistory.AsReadOnly();
        if (_options.Policy == TieringExecutionPolicy.DistributedCluster)
            throw new NotSupportedException("DistributedCluster policy requires cluster node endpoints.");

        _resolvedPolicy = _options.Policy;
        _syntheticLayers = layers;
        _syntheticDim = dim;
        _syntheticExpertFfnDim = expertFfnDim;
        _syntheticNumExperts = numExperts;
        _syntheticTopK = topK;
        _syntheticVocabSize = vocabSize;
        _syntheticNormTopK = normTopK;
        _syntheticLmHead = lmHeadWeights;
        _tokenizer = tokenizer;

        _routingProfile = routingProfile;
        _coScheduler = coScheduler;
        _slotManager = slotManager;
        _hostArena = hostArena;
        _logitsBuffer = new float[Math.Max(1, vocabSize)];

        _cpuWorkspaces = new Qwen2D3D12Model.ExpertCpuWorkspace[Math.Max(1, topK)];
        for (int i = 0; i < _cpuWorkspaces.Length; i++)
        {
            _cpuWorkspaces[i] = new Qwen2D3D12Model.ExpertCpuWorkspace(expertFfnDim, dim);
        }
        _cpuNormXSums = new float[(dim + 31) / 32];
        _syntheticNormScratch = new float[dim];
        _syntheticNormSums = new float[(dim + 31) / 32];
        _syntheticOutScratch = new float[dim];
    }

    /// <summary>
    /// Executes a single MoE layer across a batch of <paramref name="batchSize"/> token hidden states
    /// using genuine router scoring, <see cref="ExpertRoutingProfile"/> updates, VRAM slot probing,
    /// opt-in <see cref="TieredRunnerOptions.RouteTailSkipMinRank"/>, dynamic CPU/PCIe miss splitting,
    /// and <see cref="QuantKernels"/> expert matrix multiplications.
    /// </summary>
    internal BatchLayerMissSplitResult ExecuteSyntheticMoELayerBatch(
        int layerIdx,
        ReadOnlySpan<float> normXBatch,
        Span<float> ffnOutBatch,
        int batchSize,
        bool isSpeculativeDraftBatch)
        => ExecuteSyntheticMoELayerBatchCore(layerIdx, normXBatch, ffnOutBatch, batchSize, isSpeculativeDraftBatch, emitTelemetry: true);

    private BatchLayerMissSplitResult ExecuteSyntheticMoELayerBatchCore(
        int layerIdx,
        ReadOnlySpan<float> normXBatch,
        Span<float> ffnOutBatch,
        int batchSize,
        bool isSpeculativeDraftBatch,
        bool emitTelemetry)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticLayers == null)
            throw new InvalidOperationException("ExecuteSyntheticMoELayerBatch requires a synthetic TieredMoERunner instance.");
        if ((uint)layerIdx >= (uint)_syntheticLayers.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIdx));
        if (batchSize <= 0)
            return default;
        if (normXBatch.Length < batchSize * _syntheticDim)
            throw new ArgumentException("normXBatch too small.", nameof(normXBatch));
        if (ffnOutBatch.Length < batchSize * _syntheticDim)
            throw new ArgumentException("ffnOutBatch too small.", nameof(ffnOutBatch));

        if (emitTelemetry)
        {
            ResetStepTelemetryAccumulators();
        }
        long startTs = Stopwatch.GetTimestamp();

        var lw = _syntheticLayers[layerIdx];
        int dim = _syntheticDim;
        int expertFfnDim = _syntheticExpertFfnDim;
        int numExperts = _syntheticNumExperts;
        int topK = _syntheticTopK;

        int totalSlots = checked(batchSize * topK);
        int[] indices = new int[totalSlots];
        float[] weights = new float[totalSlots];

        BatchLayerMissSplitResult split;
        fixed (float* pNormX = normXBatch, pOut = ffnOutBatch)
        fixed (int* pIdx = indices)
        fixed (float* pW = weights)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.RouterTopK(
                    pNormX + (long)t * dim,
                    lw.FfnGateInpWeight,
                    lw.FfnGateInpBias,
                    dim,
                    numExperts,
                    topK,
                    pIdx + (long)t * topK,
                    pW + (long)t * topK,
                    _syntheticNormTopK);
            }

            split = ExecutePlannedMoELayerCore(
                layerIdx,
                lw,
                dim,
                expertFfnDim,
                numExperts,
                topK,
                _syntheticNormTopK,
                batchSize,
                isSpeculativeDraftBatch,
                indices.AsSpan(),
                weights.AsSpan(),
                pNormX,
                pOut);
        }

        if (emitTelemetry)
        {
            AccumulateLayerSplit(in split);
            double elapsedMs = Stopwatch.GetElapsedTime(startTs).TotalMilliseconds;
            EmitStepTelemetry(0, string.Empty, elapsedMs, attentionMs: 0.0);
        }

        return split;
    }

    /// <summary>
    /// Executes an end-to-end synthetic multi-layer batch (e.g. speculative draft verification or prefill)
    /// across all configured MoE layers, computing output logits and emitting <see cref="TieredStepTelemetry"/>.
    /// </summary>
    internal void EvaluateSyntheticBatch(
        ReadOnlySpan<float> inputHiddenBatch,
        int batchSize,
        bool isSpeculativeDraftBatch,
        Span<float> batchLogitsOut,
        int primaryTokenId = 0,
        string primaryTokenText = "")
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_syntheticLayers == null)
            throw new InvalidOperationException("EvaluateSyntheticBatch requires a synthetic TieredMoERunner instance.");
        if (batchSize <= 0)
            return;

        int dim = _syntheticDim;
        if (inputHiddenBatch.Length < batchSize * dim)
            throw new ArgumentException("inputHiddenBatch too small.", nameof(inputHiddenBatch));

        ResetStepTelemetryAccumulators();
        long stepStartTs = Stopwatch.GetTimestamp();

        float[] residual = new float[batchSize * dim];
        float[] normBuf = new float[batchSize * dim];
        float[] ffnOut = new float[batchSize * dim];
        inputHiddenBatch.Slice(0, batchSize * dim).CopyTo(residual);

        for (int l = 0; l < _syntheticLayers.Length; l++)
        {
            var lw = _syntheticLayers[l];
            fixed (float* pRes = residual, pNorm = normBuf)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    if (lw.FfnNormWeight != null)
                    {
                        QuantKernels.RMSNorm(pRes + (long)t * dim, lw.FfnNormWeight, pNorm + (long)t * dim, dim, 1e-5f);
                    }
                    else
                    {
                        new ReadOnlySpan<float>(pRes + (long)t * dim, dim).CopyTo(new Span<float>(pNorm + (long)t * dim, dim));
                    }
                }
            }

            if (lw.IsMoe)
            {
                var split = ExecuteSyntheticMoELayerBatchCore(l, normBuf, ffnOut, batchSize, isSpeculativeDraftBatch, emitTelemetry: false);
                AccumulateLayerSplit(in split);
            }
            else
            {
                ffnOut.AsSpan().Clear();
            }

            for (int i = 0; i < batchSize * dim; i++)
            {
                residual[i] += ffnOut[i];
            }
        }

        if (!batchLogitsOut.IsEmpty && _syntheticLmHead != null)
        {
            int vocabSize = _syntheticVocabSize;
            fixed (float* pRes = residual, pLm = _syntheticLmHead, pLogits = batchLogitsOut)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    float* x = pRes + (long)t * dim;
                    float* dst = pLogits + (long)t * vocabSize;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        dst[v] = QuantKernels.VecDotF32(pLm + (long)v * dim, x, dim);
                    }
                }
            }

            batchLogitsOut.Slice((batchSize - 1) * _syntheticVocabSize, _syntheticVocabSize)
                .CopyTo(_logitsBuffer);
        }

        double totalMs = Stopwatch.GetElapsedTime(stepStartTs).TotalMilliseconds;
        EmitStepTelemetry(primaryTokenId, primaryTokenText, totalMs, attentionMs: 0.0);
    }

    private BatchLayerMissSplitResult ExecutePlannedMoELayerCore(
        int layerIdx,
        LayerWeights lw,
        int dim,
        int expertFfnDim,
        int numExperts,
        int topK,
        bool normTopK,
        int batchSize,
        bool isSpeculativeDraftBatch,
        Span<int> indices,
        Span<float> weights,
        float* pNormXBatch,
        float* pOutBatch)
    {
        long layerStartTs = Stopwatch.GetTimestamp();

        var split = _missSplitter.PlanLayerBatch(
            layerIdx,
            batchSize,
            topK,
            numExperts,
            normTopK,
            isSpeculativeDraftBatch,
            _options.RouteTailSkipMinRank,
            _resolvedPolicy,
            indices,
            weights,
            _routingProfile,
            _slotManager,
            _coScheduler,
            HostArena);

        long expertBytes = TieredMoEMissSplitter.EstimateExpertByteSize(lw, dim, expertFfnDim);

        // Execute PCIe-promoted / GPU-shared-memory experts and record measured timings
        if (split.PcieTransferUniqueCount > 0)
        {
            long pcieStart = Stopwatch.GetTimestamp();
            // Simulate/perform host-to-shared-memory staging read across each promoted unique expert
            long pcieBytes = split.PcieTransferUniqueCount * expertBytes;
            double pcieMs = Math.Max(0.001, Stopwatch.GetElapsedTime(pcieStart).TotalMilliseconds);
            _coScheduler.RecordPcieTransfer(pcieBytes, pcieMs);
        }

        // Execute all non-skipped expert activations genuinely via QuantKernels
        long cpuSw = Stopwatch.GetTimestamp();
        long gpuExecStart = Stopwatch.GetTimestamp();
        var ws = _cpuWorkspaces![0];
        fixed (float* pNormXSums = _cpuNormXSums!, pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* normX = pNormXBatch + (long)t * dim;
                float* outVec = pOutBatch + (long)t * dim;
                new Span<float>(outVec, dim).Clear();
                QuantKernels.ComputeBlockSums32(normX, pNormXSums, dim);

                int baseOff = t * topK;
                for (int k = 0; k < topK; k++)
                {
                    int expIdx = indices[baseOff + k];
                    float w = weights[baseOff + k];
                    if (expIdx < 0 || w == 0f)
                        continue;

                    TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                        lw,
                        expIdx,
                        w,
                        normX,
                        pNormXSums,
                        outVec,
                        dim,
                        expertFfnDim,
                        pGate,
                        pUp,
                        pAct,
                        pActSums,
                        pDownOut);
                }
            }
        }

        if (split.VramHitUniqueCount + split.PcieTransferUniqueCount > 0)
        {
            double gpuMs = Math.Max(0.001, Stopwatch.GetElapsedTime(gpuExecStart).TotalMilliseconds);
            _coScheduler.RecordGpuExecution((split.VramHitUniqueCount + split.PcieTransferUniqueCount) * expertBytes, gpuMs);
        }

        if (split.CpuExecuteUniqueCount > 0)
        {
            double cpuMs = Math.Max(0.001, Stopwatch.GetElapsedTime(cpuSw).TotalMilliseconds);
            _coScheduler.RecordCpuExecution(split.CpuExecuteUniqueCount * expertBytes, cpuMs);
        }

        _stepRoutedExpertMs += Stopwatch.GetElapsedTime(layerStartTs).TotalMilliseconds;
        return split;
    }

    private void InitializeD3D12TieredExpertResources()
    {
        if (_d3dCtx == null || _weights == null || !_weights.IsMoe)
            return;

        int expertFfnDim = _weights.ExpertFeedForwardLength;
        int numExperts = _weights.ExpertCount;
        int dim = _weights.EmbeddingLength;
        int topK = Math.Max(1, _weights.ExpertUsedCount);

        _layerGateAlignedBytes = new ulong[_weights.BlockCount];
        _layerUpAlignedBytes = new ulong[_weights.BlockCount];
        _layerDownAlignedBytes = new ulong[_weights.BlockCount];
        _layerGateGpuType = new GgufType[_weights.BlockCount];
        _layerUpGpuType = new GgufType[_weights.BlockCount];
        _layerDownGpuType = new GgufType[_weights.BlockCount];
        _layerHasFusedExps = new bool[_weights.BlockCount];
        _layerHasBiases = new bool[_weights.BlockCount];

        int totalMoeLayers = 0;
        ulong maxSlotBytes = 0;

        for (int ml = 0; ml < _weights.BlockCount; ml++)
        {
            var lw = _weights.Layers[ml];
            if (lw == null || !lw.IsMoe) continue;

            totalMoeLayers++;
            bool fused = lw.FfnGateUpExpsWeight != null && lw.FfnGateExpsWeight == null;
            _layerHasFusedExps[ml] = fused;
            _layerHasBiases[ml] = lw.FfnGateExpsBias != null || lw.FfnUpExpsBias != null || lw.FfnDownExpsBias != null;

            GgufType rawGateType = fused ? lw.FfnGateUpExpsType : lw.FfnGateExpsType;
            GgufType rawUpType = fused ? lw.FfnGateUpExpsType : lw.FfnUpExpsType;
            GgufType rawDownType = lw.FfnDownExpsType;

            _layerGateGpuType[ml] = rawGateType is GgufType.BF16 or GgufType.F16 ? GgufType.F32 : rawGateType;
            _layerUpGpuType[ml] = rawUpType is GgufType.BF16 or GgufType.F16 ? GgufType.F32 : rawUpType;
            _layerDownGpuType[ml] = rawDownType is GgufType.BF16 or GgufType.F16 ? GgufType.F32 : rawDownType;

            ulong gateAligned = AlignUp256(D3D12TensorAlign.GetAlignedTensorBytes(rawGateType, expertFfnDim, dim));
            ulong upAligned = AlignUp256(D3D12TensorAlign.GetAlignedTensorBytes(rawUpType, expertFfnDim, dim));
            ulong downAligned = AlignUp256(D3D12TensorAlign.GetAlignedTensorBytes(rawDownType, dim, expertFfnDim));

            _layerGateAlignedBytes[ml] = gateAligned;
            _layerUpAlignedBytes[ml] = upAligned;
            _layerDownAlignedBytes[ml] = downAligned;

            ulong layerSlot = gateAligned + upAligned + downAligned;
            if (layerSlot > maxSlotBytes) maxSlotBytes = layerSlot;
        }

        _maxAlignedSlotBytes = maxSlotBytes;
        _cpuWorkspaces = new Qwen2D3D12Model.ExpertCpuWorkspace[topK];
        for (int i = 0; i < topK; i++)
        {
            _cpuWorkspaces[i] = new Qwen2D3D12Model.ExpertCpuWorkspace(expertFfnDim, dim);
        }
        _cpuNormXSums = new float[(dim + 31) / 32];

        if (totalMoeLayers == 0 || maxSlotBytes == 0)
            return;

        int maxSlots = checked(totalMoeLayers * numExperts);
        int slotCapacity;

        if (_options.VramExpertSlotCount.HasValue)
        {
            slotCapacity = Math.Clamp(_options.VramExpertSlotCount.Value, 0, maxSlots);
        }
        else
        {
            var snap = QueryVramHeadroom();
            ulong budgetCap = checked(_options.VramBudgetMb * 1024UL * 1024UL);
            ulong usable = Math.Min(snap.SafeAvailableBytes, budgetCap);
            var boundedSnap = new VramHeadroomSnapshot(
                snap.DedicatedVramBytes,
                snap.OsBudgetBytes,
                snap.CurrentUsageBytes,
                snap.SafetyReserveBytes,
                usable,
                snap.IsAtCliffRisk);

            slotCapacity = WddmVramGuard.ComputeRecommendedSlotCount(
                boundedSnap,
                maxSlotBytes,
                currentActiveSlots: 0,
                maxSlotCapacity: maxSlots,
                minSlotFloor: 0);
        }

        if (slotCapacity > 0)
        {
            const ulong segmentTargetBytes = 256UL * 1024 * 1024;
            _slotsPerSegment = Math.Max(1, (int)(segmentTargetBytes / maxSlotBytes));
            int segmentCount = (slotCapacity + _slotsPerSegment - 1) / _slotsPerSegment;
            int rem = slotCapacity;
            for (int s = 0; s < segmentCount; s++)
            {
                int segSlots = Math.Min(_slotsPerSegment, rem);
                _dVramSlotSegments.Add(_d3dCtx.CreateDeviceBuffer((ulong)segSlots * maxSlotBytes));
                rem -= segSlots;
            }

            _slotManager = new GpuResidentSlotManager(
                slotCapacity,
                (nuint)maxSlotBytes,
                allocateVram: false,
                SlotEvictionPolicy.DecayedFrequency,
                _routingProfile);
        }

        _pcieStagingCapacity = Math.Max(topK, Math.Min(32, Math.Max(topK, slotCapacity)));
        _uploadExpertPcieStaging = _d3dCtx.CreateUploadBuffer((ulong)_pcieStagingCapacity * maxSlotBytes);
        void* pStg = null;
        _uploadExpertPcieStaging.Map(0, null, &pStg);
        _pUploadExpertPcieStaging = (byte*)pStg;
    }

    private ulong GetVramSlotGpuAddress(int slotIdx)
    {
        int segIdx = slotIdx / _slotsPerSegment;
        int localIdx = slotIdx % _slotsPerSegment;
        return _dVramSlotSegments[segIdx].GPUVirtualAddress + (ulong)localIdx * _maxAlignedSlotBytes;
    }

    private ulong StageExpertIntoPcieBuffer(LayerWeights lw, int modelLayer, int expertIdx, int expertFfnDim, int dim, byte* dstPtr)
    {
        bool fused = _layerHasFusedExps![modelLayer];
        ulong gateAligned = _layerGateAlignedBytes![modelLayer];
        ulong upAligned = _layerUpAlignedBytes![modelLayer];
        ulong downAligned = _layerDownAlignedBytes![modelLayer];

        byte* srcGate;
        byte* srcUp;
        GgufType rawGateType;
        GgufType rawUpType;

        if (fused)
        {
            rawGateType = lw.FfnGateUpExpsType;
            rawUpType = lw.FfnGateUpExpsType;
            long halfBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(rawGateType, dim);
            srcGate = lw.FfnGateUpExpsWeight + (long)expertIdx * (halfBytes * 2L);
            srcUp = srcGate + halfBytes;
        }
        else
        {
            rawGateType = lw.FfnGateExpsType;
            rawUpType = lw.FfnUpExpsType;
            srcGate = lw.FfnGateExpsWeight + (long)expertIdx * (long)expertFfnDim * GgufTypes.GetRowBytes(rawGateType, dim);
            srcUp = lw.FfnUpExpsWeight + (long)expertIdx * (long)expertFfnDim * GgufTypes.GetRowBytes(rawUpType, dim);
        }

        byte* srcDown = lw.FfnDownExpsWeight + (long)expertIdx * (long)dim * GgufTypes.GetRowBytes(lw.FfnDownExpsType, expertFfnDim);

        D3D12TensorAlign.AlignTensorToBuffer(rawGateType, srcGate, dstPtr, expertFfnDim, dim);
        D3D12TensorAlign.AlignTensorToBuffer(rawUpType, srcUp, dstPtr + gateAligned, expertFfnDim, dim);
        D3D12TensorAlign.AlignTensorToBuffer(lw.FfnDownExpsType, srcDown, dstPtr + gateAligned + upAligned, dim, expertFfnDim);

        return gateAligned + upAligned + downAligned;
    }

    private bool HandleD3D12TieredMoeLayerBatch(
        int modelLayer,
        int chunkSize,
        int topK,
        bool allowTailSkip,
        int* batchTopKIndices,
        float* batchTopKWeights,
        float* batchNormXHostPtr,
        float* cpuOutHostPtr,
        ID3D12Resource dNormXSingle,
        ID3D12Resource dNormXBatch,
        ID3D12Resource dExpertGate,
        ID3D12Resource dExpertUp,
        ID3D12Resource dExpertAct,
        ID3D12Resource dExpertDownOut,
        ID3D12Resource dFfnOutSingle,
        ID3D12Resource dFfnOutBatch,
        out bool cpuOutHasContribution)
    {
        cpuOutHasContribution = false;
        if (_weights == null || _d3dCtx == null || _d3dModel == null)
            return false;

        long layerStart = Stopwatch.GetTimestamp();
        var lwHost = _weights.Layers[modelLayer];
        int dim = _weights.EmbeddingLength;
        int expertFfnDim = _weights.ExpertFeedForwardLength;
        int numExperts = _weights.ExpertCount;
        int totalSlots = checked(chunkSize * topK);

        bool layerSupportsGpu = _pUploadExpertPcieStaging != null && !_layerHasBiases![modelLayer];
        var split = _missSplitter.PlanLayerBatch(
            modelLayer,
            chunkSize,
            topK,
            numExperts,
            _weights.NormTopK,
            allowTailSkip,
            _options.RouteTailSkipMinRank,
            layerSupportsGpu ? _resolvedPolicy : TieringExecutionPolicy.HybridGpuHostRam,
            new Span<int>(batchTopKIndices, totalSlots),
            new Span<float>(batchTopKWeights, totalSlots),
            _routingProfile,
            layerSupportsGpu ? _slotManager : null,
            _coScheduler,
            HostArena,
            maxPcieTransfers: layerSupportsGpu ? (_pcieStagingCapacity > 0 ? _pcieStagingCapacity : int.MaxValue) : 0);

        AccumulateLayerSplit(in split);

        bool gpuSupportedForLayer =
            layerSupportsGpu &&
            (split.VramHitUniqueCount > 0 || split.PcieTransferUniqueCount > 0);

        bool gpuInitializedOut = false;

        if (gpuSupportedForLayer)
        {
            var planned = _missSplitter.PlannedExperts;
            var pcieSw = split.PcieTransferUniqueCount > 0 ? Stopwatch.StartNew() : null;

            int stagingCursor = 0;
            // Track Upload-heap VA per expert when _slotManager is null (pure GpuSharedMemory PCIe direct SRV)
            Span<(int ExpertIdx, int SlotIdx, ulong PcieUploadOffset, ulong CopyBytes)> pcieEntries =
                split.PcieTransferUniqueCount <= 64
                    ? stackalloc (int, int, ulong, ulong)[split.PcieTransferUniqueCount]
                    : new (int, int, ulong, ulong)[split.PcieTransferUniqueCount];
            int pcieCount = 0;

            for (int u = 0; u < planned.Length; u++)
            {
                ref readonly var ue = ref planned[u];
                if (ue.Disposition == ExpertExecutionDisposition.PcieTransfer && stagingCursor < _pcieStagingCapacity)
                {
                    ulong stgOffset = (ulong)stagingCursor * _maxAlignedSlotBytes;
                    ulong copyBytes = StageExpertIntoPcieBuffer(
                        lwHost,
                        modelLayer,
                        ue.ExpertIndex,
                        expertFfnDim,
                        dim,
                        _pUploadExpertPcieStaging + stgOffset);
                    pcieEntries[pcieCount++] = (ue.ExpertIndex, ue.VramSlotIndex, stgOffset, copyBytes);
                    stagingCursor++;
                }
            }
            pcieSw?.Stop();

            long gpuSwStart = Stopwatch.GetTimestamp();
            _d3dCtx.BeginCommands();
            var cmd = _d3dCtx.CommandList;

            // Promote staged experts into VRAM slots if slot pool is active
            if (_dVramSlotSegments.Count > 0 && pcieCount > 0)
            {
                Span<bool> segTouched = _dVramSlotSegments.Count <= 256
                    ? stackalloc bool[_dVramSlotSegments.Count]
                    : new bool[_dVramSlotSegments.Count];

                for (int i = 0; i < pcieCount; i++)
                {
                    int slotIdx = pcieEntries[i].SlotIdx;
                    if (slotIdx < 0) continue;
                    int seg = slotIdx / _slotsPerSegment;
                    ulong segOff = (ulong)(slotIdx % _slotsPerSegment) * _maxAlignedSlotBytes;
                    if (!segTouched[seg])
                    {
                        cmd.ResourceBarrierTransition(_dVramSlotSegments[seg], ResourceStates.Common, ResourceStates.CopyDest);
                        segTouched[seg] = true;
                    }
                    cmd.CopyBufferRegion(_dVramSlotSegments[seg], segOff, _uploadExpertPcieStaging!, pcieEntries[i].PcieUploadOffset, pcieEntries[i].CopyBytes);
                }

                for (int s = 0; s < segTouched.Length; s++)
                {
                    if (segTouched[s])
                    {
                        cmd.ResourceBarrierTransition(_dVramSlotSegments[s], ResourceStates.CopyDest, ResourceStates.Common);
                    }
                }
            }

            GgufType gateType = _layerGateGpuType![modelLayer];
            GgufType upType = _layerUpGpuType![modelLayer];
            GgufType downType = _layerDownGpuType![modelLayer];
            ulong gateAligned = _layerGateAlignedBytes![modelLayer];
            ulong upAligned = _layerUpAlignedBytes![modelLayer];

            if (chunkSize == 1)
            {
                bool ffnInit = false;
                for (int k = 0; k < topK; k++)
                {
                    int expIdx = batchTopKIndices[k];
                    float w = batchTopKWeights[k];
                    if (expIdx < 0 || w == 0f) continue;

                    ulong va = ResolveExpertGpuVa(
                        _missSplitter,
                        _dVramSlotSegments,
                        _slotsPerSegment,
                        _maxAlignedSlotBytes,
                        _uploadExpertPcieStaging,
                        pcieEntries.Slice(0, pcieCount),
                        expIdx);
                    if (va == 0UL) continue;

                    _d3dModel.DispatchGemvExternal(cmd, gateType, dExpertGate, dNormXSingle, va, dim, expertFfnDim);
                    _d3dModel.DispatchGemvExternal(cmd, upType, dExpertUp, dNormXSingle, va + gateAligned, dim, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    _d3dModel.DispatchSwigluExternal(cmd, dExpertGate, dExpertUp, dExpertAct, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    _d3dModel.DispatchGemvExternal(cmd, downType, dExpertDownOut, dExpertAct, va + gateAligned + upAligned, expertFfnDim, dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    _d3dModel.DispatchVecAddWeightedExternal(cmd, dExpertDownOut, dFfnOutSingle, w, dim, ffnInit ? 1u : 0u);
                    cmd.ResourceBarrierUnorderedAccessView(dFfnOutSingle);
                    ffnInit = true;
                }
                gpuInitializedOut = ffnInit;
            }
            else
            {
                ulong vecBytes = (ulong)dim * sizeof(float);
                bool anyTokenGpu = false;

                cmd.ResourceBarrierTransition(dNormXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.ResourceBarrierTransition(dNormXSingle, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                cmd.ResourceBarrierTransition(dFfnOutBatch, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);

                for (int t = 0; t < chunkSize; t++)
                {
                    ulong tokenOff = (ulong)t * vecBytes;
                    cmd.CopyBufferRegion(dNormXSingle, 0, dNormXBatch, tokenOff, vecBytes);
                    cmd.ResourceBarrierTransition(dNormXSingle, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                    bool tokenInit = false;
                    int baseOff = t * topK;
                    for (int k = 0; k < topK; k++)
                    {
                        int expIdx = batchTopKIndices[baseOff + k];
                        float w = batchTopKWeights[baseOff + k];
                        if (expIdx < 0 || w == 0f) continue;

                        ulong va = ResolveExpertGpuVa(
                            _missSplitter,
                            _dVramSlotSegments,
                            _slotsPerSegment,
                            _maxAlignedSlotBytes,
                            _uploadExpertPcieStaging,
                            pcieEntries.Slice(0, pcieCount),
                            expIdx);
                        if (va == 0UL) continue;

                        _d3dModel.DispatchGemvExternal(cmd, gateType, dExpertGate, dNormXSingle, va, dim, expertFfnDim);
                        _d3dModel.DispatchGemvExternal(cmd, upType, dExpertUp, dNormXSingle, va + gateAligned, dim, expertFfnDim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        _d3dModel.DispatchSwigluExternal(cmd, dExpertGate, dExpertUp, dExpertAct, expertFfnDim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        _d3dModel.DispatchGemvExternal(cmd, downType, dExpertDownOut, dExpertAct, va + gateAligned + upAligned, expertFfnDim, dim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        _d3dModel.DispatchVecAddWeightedExternal(cmd, dExpertDownOut, dFfnOutSingle, w, dim, tokenInit ? 1u : 0u);
                        cmd.ResourceBarrierUnorderedAccessView(dFfnOutSingle);
                        tokenInit = true;
                    }

                    if (!tokenInit)
                    {
                        _d3dModel.DispatchVecAddWeightedExternal(cmd, dFfnOutSingle, dFfnOutSingle, 0f, dim, accumulate: 0);
                        cmd.ResourceBarrierUnorderedAccessView(dFfnOutSingle);
                    }
                    else
                    {
                        anyTokenGpu = true;
                    }

                    cmd.ResourceBarrierTransition(dFfnOutSingle, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(dFfnOutBatch, tokenOff, dFfnOutSingle, 0, vecBytes);
                    cmd.ResourceBarrierTransition(dFfnOutSingle, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(dNormXSingle, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                }

                cmd.ResourceBarrierTransition(dNormXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(dNormXSingle, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                cmd.ResourceBarrierTransition(dFfnOutBatch, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                gpuInitializedOut = anyTokenGpu;
            }

            _d3dCtx.EndCommandsAndExecute();
            _d3dCtx.Synchronize();
            double gpuMs = Math.Max(0.001, Stopwatch.GetElapsedTime(gpuSwStart).TotalMilliseconds);

            long singleSlotBytes = (long)_maxAlignedSlotBytes;
            if (pcieCount > 0 && pcieSw != null)
            {
                _coScheduler.RecordPcieTransfer(pcieCount * singleSlotBytes, Math.Max(0.001, pcieSw.Elapsed.TotalMilliseconds));
            }
            if (gpuInitializedOut)
            {
                _coScheduler.RecordGpuExecution((split.VramHitUniqueCount + pcieCount) * singleSlotBytes, gpuMs);
            }
        }

        // Execute CPU-assigned experts (or all active experts if GPU couldn't execute this layer)
        bool needCpuPass = !gpuInitializedOut || split.CpuExecuteUniqueCount > 0;
        if (needCpuPass)
        {
            long cpuSwStart = Stopwatch.GetTimestamp();
            var ws = _cpuWorkspaces![0];
            fixed (float* pNormXSums = _cpuNormXSums!, pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
            {
                for (int t = 0; t < chunkSize; t++)
                {
                    float* normX = batchNormXHostPtr + (long)t * dim;
                    float* dstOut = cpuOutHostPtr + (long)t * dim;
                    new Span<float>(dstOut, dim).Clear();
                    QuantKernels.ComputeBlockSums32(normX, pNormXSums, dim);

                    int baseOff = t * topK;
                    for (int k = 0; k < topK; k++)
                    {
                        int expIdx = batchTopKIndices[baseOff + k];
                        float w = batchTopKWeights[baseOff + k];
                        if (expIdx < 0 || w == 0f)
                            continue;

                        if (gpuInitializedOut &&
                            _missSplitter.TryGetPlannedExpert(expIdx, out var planned) &&
                            planned.Disposition != ExpertExecutionDisposition.CpuExecute)
                        {
                            continue;
                        }

                        TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                            lwHost,
                            expIdx,
                            w,
                            normX,
                            pNormXSums,
                            dstOut,
                            dim,
                            expertFfnDim,
                            pGate,
                            pUp,
                            pAct,
                            pActSums,
                            pDownOut);
                        cpuOutHasContribution = true;
                    }
                }
            }

            if (cpuOutHasContribution)
            {
                double cpuMs = Math.Max(0.001, Stopwatch.GetElapsedTime(cpuSwStart).TotalMilliseconds);
                int cpuExperts = gpuInitializedOut ? split.CpuExecuteUniqueCount : (split.VramHitUniqueCount + split.PcieTransferUniqueCount + split.CpuExecuteUniqueCount);
                _coScheduler.RecordCpuExecution(Math.Max(1, cpuExperts) * (long)_maxAlignedSlotBytes, cpuMs);
            }
        }

        _stepRoutedExpertMs += Stopwatch.GetElapsedTime(layerStart).TotalMilliseconds;
        return gpuInitializedOut;
    }

    private void HandleCpuTieredMoeLayerBatch(
        int modelLayer,
        int batchSize,
        int topK,
        bool allowTailSkip,
        int* batchTopKIndices,
        float* batchTopKWeights,
        float* normXBatch,
        float* normXSumBatch,
        float* ffnOutBatch)
    {
        if (_weights == null)
            return;

        long layerStart = Stopwatch.GetTimestamp();
        var lwHost = _weights.Layers[modelLayer];
        int dim = _weights.EmbeddingLength;
        int expertFfnDim = _weights.ExpertFeedForwardLength;
        int numExperts = _weights.ExpertCount;
        int normXChunks = Math.Max(1, (dim + 31) / 32);
        int totalSlots = checked(batchSize * topK);

        var split = _missSplitter.PlanLayerBatch(
            modelLayer,
            batchSize,
            topK,
            numExperts,
            _weights.NormTopK,
            allowTailSkip,
            _options.RouteTailSkipMinRank,
            _resolvedPolicy,
            new Span<int>(batchTopKIndices, totalSlots),
            new Span<float>(batchTopKWeights, totalSlots),
            _routingProfile,
            _slotManager,
            _coScheduler,
            HostArena,
            maxPcieTransfers: 0);

        AccumulateLayerSplit(in split);

        long cpuSwStart = Stopwatch.GetTimestamp();
        var ws = _cpuWorkspaces![0];
        bool executedAny = false;

        fixed (float* pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* normX = normXBatch + (long)t * dim;
                float* normXSums = normXSumBatch + (long)t * normXChunks;
                float* outVec = ffnOutBatch + (long)t * dim;
                new Span<float>(outVec, dim).Clear();

                int baseOff = t * topK;
                for (int k = 0; k < topK; k++)
                {
                    int expIdx = batchTopKIndices[baseOff + k];
                    float w = batchTopKWeights[baseOff + k];
                    if (expIdx < 0 || w == 0f)
                        continue;

                    TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                        lwHost,
                        expIdx,
                        w,
                        normX,
                        normXSums,
                        outVec,
                        dim,
                        expertFfnDim,
                        pGate,
                        pUp,
                        pAct,
                        pActSums,
                        pDownOut);
                    executedAny = true;
                }
            }
        }

        if (executedAny)
        {
            double cpuMs = Math.Max(0.001, Stopwatch.GetElapsedTime(cpuSwStart).TotalMilliseconds);
            long expertBytes = TieredMoEMissSplitter.EstimateExpertByteSize(lwHost, dim, expertFfnDim);
            _coScheduler.RecordCpuExecution(Math.Max(1, split.CpuExecuteUniqueCount) * expertBytes, cpuMs);
        }

        _stepRoutedExpertMs += Stopwatch.GetElapsedTime(layerStart).TotalMilliseconds;
    }

    private void AccumulateLayerSplit(in BatchLayerMissSplitResult split)
    {
        _stepTotalRouted += split.TotalRoutedSlots;
        _stepSkippedRouted += split.SkippedRoutedSlots;
        _stepVramHits += split.VramHitUniqueCount;
        _stepPcieTransferred += split.PcieTransferUniqueCount;
        _stepCpuExecuted += split.CpuExecuteUniqueCount;
    }

    private void ResetStepTelemetryAccumulators()
    {
        _stepTotalRouted = 0;
        _stepSkippedRouted = 0;
        _stepVramHits = 0;
        _stepPcieTransferred = 0;
        _stepCpuExecuted = 0;
        _stepRoutedExpertMs = 0.0;
    }

    private TieredStepTelemetry EmitStepTelemetry(int tokenId, string tokenText, double totalMs, double attentionMs)
    {
        double skippedPct = _stepTotalRouted > 0
            ? (100.0 * _stepSkippedRouted) / _stepTotalRouted
            : 0.0;

        var telemetry = new TieredStepTelemetry
        {
            StepIndex = _stepCounter++,
            TokenId = tokenId,
            TokenText = tokenText,
            TotalStepDurationMs = totalMs,
            AttentionVramDurationMs = attentionMs,
            RoutedExpertDurationMs = _stepRoutedExpertMs,
            NetworkInterconnectDurationMs = 0.0,
            ExpertsSkippedPercent = skippedPct,
            TotalRoutedExperts = _stepTotalRouted,
            SkippedRoutedExperts = _stepSkippedRouted,
            VramHitExperts = _stepVramHits,
            PcieTransferredExperts = _stepPcieTransferred,
            CpuExecutedExperts = _stepCpuExecuted,
            ResolvedPolicy = _resolvedPolicy
        };

        _lastStepTelemetry = telemetry;
        _telemetryHistory.Add(telemetry);
        _options.OnStepTelemetry?.Invoke(telemetry);

        GlacierDiagnostics.LogInformation(
            $"[TieredMoERunner] Step={telemetry.StepIndex} Policy={telemetry.ResolvedPolicy} Hits={telemetry.VramHitExperts} PCIe={telemetry.PcieTransferredExperts} CPU={telemetry.CpuExecutedExperts} Skipped={telemetry.SkippedRoutedExperts}/{telemetry.TotalRoutedExperts} ({telemetry.ExpertsSkippedPercent:F1}% experts skipped) StepMs={telemetry.TotalStepDurationMs:F2}");

        return telemetry;
    }

    public void ResetKvCache()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _d3dModel?.ResetKvCache();
        _kvCache?.Reset();
    }

    public void Prefill(ReadOnlySpan<int> promptTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (promptTokens.IsEmpty) return;

        ResetStepTelemetryAccumulators();
        long startTs = Stopwatch.GetTimestamp();

        if (_d3dModel != null)
        {
            _d3dModel.ForwardBatch(promptTokens, 0, _logitsBuffer.AsSpan(), computeLogits: true);
        }
        else if (_cpuModel != null)
        {
            _cpuModel.ForwardBatch(promptTokens, 0, _logitsBuffer.AsSpan(), _kvCache!, computeLogits: true);
        }

        double elapsedMs = Stopwatch.GetElapsedTime(startTs).TotalMilliseconds;
        int lastTok = promptTokens[^1];
        string text = _tokenizer != null ? _tokenizer.DecodeToken(lastTok) : string.Empty;
        EmitStepTelemetry(lastTok, text, elapsedMs, Math.Max(0.0, elapsedMs - _stepRoutedExpertMs));
    }

    public void ForwardToken(int token, int pos, bool computeLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ResetStepTelemetryAccumulators();
        long startTs = Stopwatch.GetTimestamp();

        if (_d3dModel != null)
        {
            _d3dModel.Forward(token, pos, _logitsBuffer.AsSpan(), computeLogits);
        }
        else if (_cpuModel != null)
        {
            _cpuModel.Forward(token, pos, _kvCache!, _logitsBuffer.AsSpan(), computeLogits);
        }

        double elapsedMs = Stopwatch.GetElapsedTime(startTs).TotalMilliseconds;
        string text = _tokenizer != null ? _tokenizer.DecodeToken(token) : string.Empty;
        EmitStepTelemetry(token, text, elapsedMs, Math.Max(0.0, elapsedMs - _stepRoutedExpertMs));
    }

    public void CopyCurrentLogits(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int vocab = VocabularySize;
        if (destination.Length < vocab)
            throw new ArgumentException($"Destination length {destination.Length} < VocabSize {vocab}.", nameof(destination));

        _logitsBuffer.AsSpan(0, vocab).CopyTo(destination);
    }

    public int SampleNextToken(SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _sampler.Sample(_logitsBuffer.AsSpan(), options, recentTokens);
    }

    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return;
        int vocabSize = VocabularySize;
        if (predictedTokens.Length < tokens.Length)
            throw new ArgumentException($"predictedTokens length {predictedTokens.Length} < tokens length {tokens.Length}.", nameof(predictedTokens));

        float[] flatLogits = new float[tokens.Length * vocabSize];
        VerifyBatchLogits(tokens, startPos, flatLogits);

        for (int t = 0; t < tokens.Length; t++)
        {
            ReadOnlySpan<float> row = flatLogits.AsSpan(t * vocabSize, vocabSize);
            int bestId = 0;
            float bestVal = row[0];
            for (int v = 1; v < row.Length; v++)
            {
                if (row[v] > bestVal)
                {
                    bestVal = row[v];
                    bestId = v;
                }
            }
            predictedTokens[t] = bestId;
        }
    }

    public void VerifyBatchLogits(ReadOnlySpan<int> tokens, int startPos, Span<float> batchLogitsOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return;
        int vocabSize = VocabularySize;
        if (batchLogitsOut.Length < tokens.Length * vocabSize)
            throw new ArgumentException("batchLogitsOut buffer is too small.", nameof(batchLogitsOut));

        ResetStepTelemetryAccumulators();
        long startTs = Stopwatch.GetTimestamp();

        if (_d3dModel != null)
        {
            _d3dModel.VerifyBatchLogits(tokens, startPos, batchLogitsOut, vocabSize, allowTailSkip: true);
        }
        else if (_cpuModel != null)
        {
            _cpuModel.VerifyBatchLogits(tokens, startPos, batchLogitsOut, vocabSize, _kvCache);
        }

        batchLogitsOut.Slice((tokens.Length - 1) * vocabSize, vocabSize).CopyTo(_logitsBuffer);
        double elapsedMs = Stopwatch.GetElapsedTime(startTs).TotalMilliseconds;
        int lastTok = tokens[^1];
        string text = _tokenizer != null ? _tokenizer.DecodeToken(lastTok) : string.Empty;
        EmitStepTelemetry(lastTok, text, elapsedMs, Math.Max(0.0, elapsedMs - _stepRoutedExpertMs));
    }

    /// <summary>
    /// Streams generated tokens asynchronously while emitting <see cref="TieredStepTelemetry"/>.
    /// </summary>
    public IAsyncEnumerable<string> GenerateStreamAsync(
        string prompt,
        int maxNewTokens = 128,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_tokenizer == null)
            throw new InvalidOperationException("GenerateStreamAsync requires a tokenizer-backed GGUF model.");

        return GenerateStreamCoreAsync(prompt, maxNewTokens, cancellationToken);
    }

    private async IAsyncEnumerable<string> GenerateStreamCoreAsync(
        string prompt,
        int maxNewTokens,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int[] promptTokens = _tokenizer!.Encode(prompt);
        if (promptTokens.Length == 0 || _options.MaxSequenceLength <= 0) yield break;
        if (promptTokens.Length > _options.MaxSequenceLength)
        {
            promptTokens = promptTokens[.._options.MaxSequenceLength];
        }

        Prefill(promptTokens);
        int currentPos = promptTokens.Length;

        var samplingOpts = new SamplingOptions
        {
            Temperature = _options.Temperature,
            TopP = _options.TopP,
            TopK = _options.TopK,
            RepetitionPenalty = 1.0f
        };

        for (int step = 0; step < maxNewTokens && currentPos < _options.MaxSequenceLength; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int nextToken = SampleToken(_logitsBuffer.AsSpan(), samplingOpts);

            if (_tokenizer.IsStopToken(nextToken) || nextToken == _tokenizer.EosTokenId || nextToken == 151645 || nextToken == 151643)
                break;

            string piece = _tokenizer.DecodeToken(nextToken);
            yield return piece;
            await Task.Yield();

            if (currentPos >= _options.MaxSequenceLength)
                break;

            ForwardToken(nextToken, currentPos++, computeLogits: true);
        }
    }

    /// <summary>
    /// Generates a complete response asynchronously with comprehensive execution metrics.
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
                PromptTokens = _tokenizer != null ? _tokenizer.Encode(prompt).Length : 0,
                GeneratedTokens = tokenCount,
                TotalDuration = swTotal.Elapsed,
                GenerationDuration = swTotal.Elapsed,
                PromptEvalDuration = TimeSpan.Zero
            }
        };
    }

    /// <summary>
    /// Persists the current <see cref="ExpertRoutingProfile"/> to <paramref name="filePath"/>
    /// (or <see cref="TieredRunnerOptions.RoutingProfilePath"/>).
    /// </summary>
    public void SaveRoutingProfile(string? filePath = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        string? target = filePath ?? _options.RoutingProfilePath;
        if (string.IsNullOrWhiteSpace(target))
            throw new InvalidOperationException("No routing profile path specified.");

        _routingProfile.Save(target);
    }

    private static ulong ResolveExpertGpuVa(
        TieredMoEMissSplitter missSplitter,
        List<ID3D12Resource> slotSegments,
        int slotsPerSegment,
        ulong maxAlignedSlotBytes,
        ID3D12Resource? uploadStaging,
        ReadOnlySpan<(int ExpertIdx, int SlotIdx, ulong PcieUploadOffset, ulong CopyBytes)> pcieEntries,
        int expIdx)
    {
        if (missSplitter.TryGetPlannedExpert(expIdx, out var pe))
        {
            if (pe.VramSlotIndex >= 0 && slotSegments.Count > 0)
            {
                int segIdx = pe.VramSlotIndex / slotsPerSegment;
                int localIdx = pe.VramSlotIndex % slotsPerSegment;
                return slotSegments[segIdx].GPUVirtualAddress + (ulong)localIdx * maxAlignedSlotBytes;
            }

            if (uploadStaging != null)
            {
                for (int i = 0; i < pcieEntries.Length; i++)
                {
                    if (pcieEntries[i].ExpertIdx == expIdx)
                    {
                        return uploadStaging.GPUVirtualAddress + pcieEntries[i].PcieUploadOffset;
                    }
                }
            }
        }
        return 0UL;
    }

    private int SampleToken(Span<float> logits, SamplingOptions samplingOptions)
    {
        if (samplingOptions.Temperature <= Sampler.GreedyTemperatureThreshold)
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

        return _sampler.Sample(logits, samplingOptions);
    }

    private static ulong ComputeTotalExpertBytes(ModelWeights weights)
    {
        if (!weights.IsMoe) return 0UL;
        ulong total = 0UL;
        int dim = weights.EmbeddingLength;
        int expertFfnDim = weights.ExpertFeedForwardLength;
        int numExperts = weights.ExpertCount;

        for (int l = 0; l < weights.BlockCount; l++)
        {
            var lw = weights.Layers[l];
            if (lw == null || !lw.IsMoe) continue;
            long one = TieredMoEMissSplitter.EstimateExpertByteSize(lw, dim, expertFfnDim);
            total += (ulong)numExperts * (ulong)one;
        }
        return total;
    }

    private static ulong AlignUp256(ulong bytes) =>
        (bytes + SlotSubAlignmentBytes - 1UL) & ~(SlotSubAlignmentBytes - 1UL);

    private static int SelectOptimalAdapter()
    {
        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
        int bestIdx = 0;
        ulong maxVram = 0;

        for (int i = 0; factory.EnumAdapters1((uint)i, out var ad).Success; i++)
        {
            using (ad)
            {
                if ((ad.Description1.Flags & AdapterFlags.Software) == 0 && (ulong)ad.Description1.DedicatedVideoMemory > maxVram)
                {
                    maxVram = (ulong)ad.Description1.DedicatedVideoMemory;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;

            _d3dCtx?.Synchronize();

            for (int i = 0; i < _dVramSlotSegments.Count; i++)
            {
                _dVramSlotSegments[i]?.Dispose();
            }
            _dVramSlotSegments.Clear();

            if (_uploadExpertPcieStaging != null)
            {
                _uploadExpertPcieStaging.Unmap(0);
                _uploadExpertPcieStaging.Dispose();
                _uploadExpertPcieStaging = null;
                _pUploadExpertPcieStaging = null;
            }

            _d3dModel?.Dispose();
            _d3dCtx?.Dispose();
            _cpuModel?.Dispose();
            _kvCache?.Dispose();
            _slotManager?.Dispose();
            _weights?.Dispose();
            _gguf?.Dispose();
        }
    }
}
