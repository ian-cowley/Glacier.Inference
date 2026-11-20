namespace Glacier.Inference.Engine;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;

/// <summary>
/// Configuration parameters for DFlash-style parallel block-diffusion speculative drafting.
/// </summary>
public sealed record DFlashOptions
{
    private int _blockSize = 4;
    private int _diffusionSteps = 3;
    private int _maxNgramOrder = 3;
    private int _maxCandidatesPerSlot = 24;
    private float _remaskThreshold = 0.35f;
    private float _minEmitConfidence;
    private float _diffusionTemperature = 0.75f;
    private int _maxTransitionKeys = 16384;

    /// <summary>Target parallel draft block size <c>B</c> (1..1024, default: 4).</summary>
    public int BlockSize
    {
        get => _blockSize;
        init
        {
            if (value < 1 || value > DFlashDraftProvider.MaxDFlashBlockSize)
                throw new ArgumentOutOfRangeException(nameof(value), value, $"BlockSize must be in [1, {DFlashDraftProvider.MaxDFlashBlockSize}].");
            _blockSize = value;
        }
    }

    /// <summary>Number of iterative bidirectional mask-refinement steps <c>T</c> (&gt;= 1, default: 3).</summary>
    public int DiffusionSteps
    {
        get => _diffusionSteps;
        init
        {
            if (value < 1) throw new ArgumentOutOfRangeException(nameof(value), value, "DiffusionSteps must be >= 1.");
            _diffusionSteps = value;
        }
    }

    /// <summary>Maximum forward n-gram order used by context-conditioned transitions (1..4, default: 3).</summary>
    public int MaxNgramOrder
    {
        get => _maxNgramOrder;
        init
        {
            if (value < 1 || value > 4) throw new ArgumentOutOfRangeException(nameof(value), value, "MaxNgramOrder must be in [1, 4].");
            _maxNgramOrder = value;
        }
    }

    /// <summary>Maximum candidate tokens evaluated per slot during parallel refinement (default: 24).</summary>
    public int MaxCandidatesPerSlot
    {
        get => _maxCandidatesPerSlot;
        init
        {
            if (value < 1) throw new ArgumentOutOfRangeException(nameof(value), value, "MaxCandidatesPerSlot must be >= 1.");
            _maxCandidatesPerSlot = value;
        }
    }

    /// <summary>Confidence threshold in [0, 1] below which previously unmasked slots are remasked between steps.</summary>
    public float RemaskThreshold
    {
        get => _remaskThreshold;
        init
        {
            if (float.IsNaN(value) || value < 0f || value > 1f)
                throw new ArgumentOutOfRangeException(nameof(value), value, "RemaskThreshold must be in [0, 1].");
            _remaskThreshold = value;
        }
    }

    /// <summary>Minimum final confidence in [0, 1] required to emit slots after position 0.</summary>
    public float MinEmitConfidence
    {
        get => _minEmitConfidence;
        init
        {
            if (float.IsNaN(value) || value < 0f || value > 1f)
                throw new ArgumentOutOfRangeException(nameof(value), value, "MinEmitConfidence must be in [0, 1].");
            _minEmitConfidence = value;
        }
    }

    /// <summary>Weight applied to right-to-left bidirectional neighbor transitions during refinement.</summary>
    public float BidirectionalWeight { get; init; } = 0.65f;

    /// <summary>Weight applied to multi-offset skip-gram anchor transitions during refinement.</summary>
    public float SkipGramWeight { get; init; } = 0.55f;

    /// <summary>Temperature (&gt; 0) for candidate softmax scoring during diffusion steps.</summary>
    public float DiffusionTemperature
    {
        get => _diffusionTemperature;
        init
        {
            if (!float.IsFinite(value) || value <= 0f)
                throw new ArgumentOutOfRangeException(nameof(value), value, "DiffusionTemperature must be finite and > 0.");
            _diffusionTemperature = value;
        }
    }

    /// <summary>When true, adapts <see cref="DFlashDraftProvider.EffectiveBlockSize"/> from online acceptance rates.</summary>
    public bool EnableAdaptiveConfidence { get; init; } = true;

    private int _maskTokenId = -1;

    /// <summary>Sentinel token ID representing a masked slot inside a diffusion block (default: -1, must be negative).</summary>
    public int MaskTokenId
    {
        get => _maskTokenId;
        init
        {
            if (value >= 0)
                throw new ArgumentOutOfRangeException(nameof(value), value, "MaskTokenId must be negative (< 0).");
            _maskTokenId = value;
        }
    }

    /// <summary>Upper bound on distinct transition table keys retained for online context conditioning.</summary>
    public int MaxTransitionKeys
    {
        get => _maxTransitionKeys;
        init
        {
            if (value < 64) throw new ArgumentOutOfRangeException(nameof(value), value, "MaxTransitionKeys must be >= 64.");
            _maxTransitionKeys = value;
        }
    }
}

/// <summary>
/// Weight descriptor for optional neural block-diffusion DFlash layers (embedded or sidecar GGUF).
/// </summary>
public sealed unsafe class DFlashWeights
{
    public int Dim { get; init; }
    public int FfnDim { get; init; }
    public int NumHeads { get; init; }
    public int NumHeadsKv { get; init; }
    public int HeadDim { get; init; }
    public int VocabSize { get; init; }
    public int BlockSize { get; init; } = 4;
    public float RmsNormEps { get; init; } = 1e-6f;

    public float* MaskEmbedding { get; init; }
    public float* SlotPosEmbeddings { get; init; }
    public float* ContextNormWeight { get; init; }
    public byte* ContextProjWeight { get; init; }
    public GgufType ContextProjType { get; init; }

    public required LayerWeights[] Layers { get; init; }

    public float* SharedHeadNormWeight { get; init; }
    public byte* SharedHeadWeight { get; init; }
    public GgufType SharedHeadType { get; init; }

    internal GgufFile? OwnedSidecarGguf { get; init; }
    internal List<IntPtr>? OwnedFloatBuffers { get; init; }
}

/// <summary>
/// DFlash-style parallel block-diffusion speculative draft provider (<see cref="IDraftProvider"/>).
/// <para>
/// Operates in two seamlessly integrated modes:
/// 1. <b>Zero-GGUF Context-Conditioned Block Diffusion</b>: builds online forward n-gram, backward bidirectional
///    bigram, and multi-offset skip-gram transition stores over the context (plus optional <see cref="IHiddenStateSource"/>
///    embedding dot-product bias) and runs <c>T</c> parallel bidirectional mask-refinement steps over a <c>B</c>-slot block
///    with cosine unmasking and confidence-based remasking.
/// 2. <b>Neural Block Diffusion</b>: when optional <see cref="DFlashWeights"/> are provided (embedded or sidecar GGUF),
///    executes non-causal bidirectional self-attention across the <c>B</c> masked/unmasked block slots conditioned on the
///    trunk hidden state.
/// </para>
/// </summary>
public sealed unsafe class DFlashDraftProvider : IDraftProvider
{
    internal const int MaxDFlashBlockSize = 1024;
    private const int BucketCapacity = 16;

    private unsafe struct CandidateBucket
    {
        public int Count;
        public fixed int Tokens[BucketCapacity];
        public fixed float Weights[BucketCapacity];

        public void Add(int tokenId, float delta)
        {
            for (int i = 0; i < Count; i++)
            {
                if (Tokens[i] == tokenId)
                {
                    Weights[i] += delta;
                    // Keep partially ordered (bubble up if larger)
                    while (i > 0 && Weights[i] > Weights[i - 1])
                    {
                        int tmpTok = Tokens[i - 1];
                        float tmpW = Weights[i - 1];
                        Tokens[i - 1] = Tokens[i];
                        Weights[i - 1] = Weights[i];
                        Tokens[i] = tmpTok;
                        Weights[i] = tmpW;
                        i--;
                    }
                    return;
                }
            }

            if (Count < BucketCapacity)
            {
                Tokens[Count] = tokenId;
                Weights[Count] = delta;
                int idx = Count;
                Count++;
                while (idx > 0 && Weights[idx] > Weights[idx - 1])
                {
                    int tmpTok = Tokens[idx - 1];
                    float tmpW = Weights[idx - 1];
                    Tokens[idx - 1] = Tokens[idx];
                    Weights[idx - 1] = Weights[idx];
                    Tokens[idx] = tmpTok;
                    Weights[idx] = tmpW;
                    idx--;
                }
                return;
            }

            // Replace lowest-weight entry if new weight exceeds half of min
            int last = BucketCapacity - 1;
            if (delta > Weights[last])
            {
                Tokens[last] = tokenId;
                Weights[last] = delta;
                int idx = last;
                while (idx > 0 && Weights[idx] > Weights[idx - 1])
                {
                    int tmpTok = Tokens[idx - 1];
                    float tmpW = Weights[idx - 1];
                    Tokens[idx - 1] = Tokens[idx];
                    Weights[idx - 1] = Weights[idx];
                    Tokens[idx] = tmpTok;
                    Weights[idx] = tmpW;
                    idx--;
                }
            }
        }
    }

    private readonly Lock _syncLock = new();
    private readonly IHiddenStateSource? _trunk;
    private readonly DFlashWeights? _weights;
    private readonly PackedDraftOutputHead? _packedHead;

    // Context-conditioned online transition tables
    private readonly List<int> _syncedTokens = new();
    private readonly Dictionary<ulong, CandidateBucket> _transitions = new(512);

    // Preallocated scratch for zero-allocation hot diffusion rounds
    private int[] _blockTokens = Array.Empty<int>();
    private int[] _prevBlockTokens = Array.Empty<int>();
    private float[] _blockConf = Array.Empty<float>();
    private int[] _slotBest = Array.Empty<int>();
    private float[] _slotConf = Array.Empty<float>();
    private int[] _slotOrder = Array.Empty<int>();
    private int[] _slotCandCount = Array.Empty<int>();
    private int[] _slotCandIds = Array.Empty<int>();
    private float[] _slotCandScores = Array.Empty<float>();
    private float[] _slotCandProbs = Array.Empty<float>();
    private int[] _recentScratch = Array.Empty<int>();
    private float[] _embScratch = Array.Empty<float>();

    // Neural DFlash scratch buffers (only allocated when _weights != null)
    private float* _nBlockHidden;
    private float* _nNormX;
    private float* _nNormSums;
    private float* _nQ;
    private float* _nK;
    private float* _nV;
    private float* _nScores;
    private float* _nAttnOut;
    private float* _nAttnSums;
    private float* _nAttnProj;
    private float* _nGate;
    private float* _nUp;
    private float* _nFfnAct;
    private float* _nFfnSums;
    private float* _nFfnOut;
    private float* _nLogits;
    private float* _nCtxProj;
    private float[]? _neuralProbsScratch;

    private bool _disposed;

    public DFlashOptions Options { get; }
    public int BlockSize => Options.BlockSize;
    public int DiffusionSteps => Options.DiffusionSteps;
    public bool HasNeuralWeights => _weights != null;
    public int EffectiveBlockSize { get; private set; }
    public long DraftRounds { get; private set; }
    public long TotalDiffusionStepsExecuted { get; private set; }
    public double SmoothedAcceptanceRate { get; private set; } = 0.65;
    public PackedDraftOutputHead? PackedOutputHead => _packedHead;
    public DraftVocabularySubset? VocabularySubset => _packedHead?.Subset;

    public DFlashDraftProvider(DFlashOptions? options = null)
    {
        Options = options ?? new DFlashOptions();
        if (Options.BlockSize < 1 || Options.BlockSize > MaxDFlashBlockSize)
            throw new ArgumentOutOfRangeException(nameof(options), Options.BlockSize, $"DFlashOptions.BlockSize must be in [1, {MaxDFlashBlockSize}].");
        EffectiveBlockSize = Options.BlockSize;
        EnsureBlockScratch(Options.BlockSize, Options.MaxCandidatesPerSlot);
    }

    public DFlashDraftProvider(
        IHiddenStateSource trunk,
        DFlashWeights? weights = null,
        DFlashOptions? options = null,
        DraftVocabularySubset? subset = null)
    {
        _trunk = trunk ?? throw new ArgumentNullException(nameof(trunk));
        _weights = weights;
        Options = options ?? new DFlashOptions();
        if (Options.BlockSize < 1 || Options.BlockSize > MaxDFlashBlockSize)
            throw new ArgumentOutOfRangeException(nameof(options), Options.BlockSize, $"DFlashOptions.BlockSize must be in [1, {MaxDFlashBlockSize}].");
        EffectiveBlockSize = Options.BlockSize;

        _trunk.EnablePositionHistory();
        if (_trunk.HiddenDim > 0)
        {
            _embScratch = new float[_trunk.HiddenDim];
        }

        if (_weights != null)
        {
            if (_weights.Dim <= 0) throw new ArgumentOutOfRangeException(nameof(weights), "DFlashWeights.Dim must be positive.");
            if (_weights.VocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(weights), "DFlashWeights.VocabSize must be positive.");
            if (_weights.BlockSize <= 0 || _weights.BlockSize > MaxDFlashBlockSize)
                throw new ArgumentOutOfRangeException(nameof(weights), _weights.BlockSize, $"DFlashWeights.BlockSize must be in [1, {MaxDFlashBlockSize}].");
            if (_weights.SharedHeadWeight == null) throw new ArgumentException("DFlashWeights.SharedHeadWeight must not be null.", nameof(weights));
            if (_weights.Layers == null) throw new ArgumentException("DFlashWeights.Layers must not be null.", nameof(weights));

            int maxSlots = Math.Max(Options.BlockSize, _weights.BlockSize);
            int dim = _weights.Dim;
            int nHeads = Math.Max(1, _weights.NumHeads);
            int nHeadsKv = Math.Max(1, _weights.NumHeadsKv);
            if (nHeads < nHeadsKv || (nHeads % nHeadsKv) != 0)
                throw new ArgumentOutOfRangeException(nameof(weights), "DFlashWeights.NumHeads must be >= NumHeadsKv and divisible by NumHeadsKv.");
            int headDim = _weights.HeadDim > 0 ? _weights.HeadDim : (dim / nHeads);
            if (headDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(weights), "DFlashWeights.HeadDim must be positive.");
            long qDimLong = (long)nHeads * headDim;
            long kvDimLong = (long)nHeadsKv * headDim;
            if (qDimLong <= 0 || qDimLong > int.MaxValue || kvDimLong <= 0 || kvDimLong > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(weights), "DFlashWeights attention dimensions overflow.");
            int qDim = (int)qDimLong;
            int kvDim = (int)kvDimLong;
            int ffnDim = _weights.FfnDim > 0 ? _weights.FfnDim : dim;

            for (int l = 0; l < _weights.Layers.Length; l++)
            {
                LayerWeights? layer = _weights.Layers[l];
                if (layer == null ||
                    layer.QWeight == null ||
                    layer.KWeight == null ||
                    layer.VWeight == null ||
                    layer.AttnOutWeight == null ||
                    layer.FfnGateWeight == null ||
                    layer.FfnUpWeight == null ||
                    layer.FfnDownWeight == null)
                {
                    throw new ArgumentException($"DFlashWeights.Layers[{l}] contains null required weight pointers.", nameof(weights));
                }
            }

            try
            {
                if (subset != null)
                {
                    _packedHead = new PackedDraftOutputHead(
                        _weights.SharedHeadType,
                        _weights.SharedHeadWeight,
                        dim,
                        _weights.VocabSize,
                        subset);
                }

                _nBlockHidden = AllocFloats((long)maxSlots * dim);
                _nNormX = AllocFloats(dim);
                _nNormSums = AllocFloats(Math.Max(1, (dim + 31) / 32));
                _nQ = AllocFloats((long)maxSlots * qDim);
                _nK = AllocFloats((long)maxSlots * kvDim);
                _nV = AllocFloats((long)maxSlots * kvDim);
                _nScores = AllocFloats((long)maxSlots * maxSlots);
                _nAttnOut = AllocFloats(qDim);
                _nAttnSums = AllocFloats(Math.Max(1, (qDim + 31) / 32));
                _nAttnProj = AllocFloats(dim);
                _nGate = AllocFloats(ffnDim);
                _nUp = AllocFloats(ffnDim);
                _nFfnAct = AllocFloats(ffnDim);
                _nFfnSums = AllocFloats(Math.Max(1, (ffnDim + 31) / 32));
                _nFfnOut = AllocFloats(dim);
                _nCtxProj = AllocFloats(dim);
                _nLogits = subset == null ? AllocFloats(_weights.VocabSize) : null;
            }
            catch
            {
                _packedHead?.Dispose();
                _packedHead = null;
                FreeAligned(ref _nBlockHidden);
                FreeAligned(ref _nNormX);
                FreeAligned(ref _nNormSums);
                FreeAligned(ref _nQ);
                FreeAligned(ref _nK);
                FreeAligned(ref _nV);
                FreeAligned(ref _nScores);
                FreeAligned(ref _nAttnOut);
                FreeAligned(ref _nAttnSums);
                FreeAligned(ref _nAttnProj);
                FreeAligned(ref _nGate);
                FreeAligned(ref _nUp);
                FreeAligned(ref _nFfnAct);
                FreeAligned(ref _nFfnSums);
                FreeAligned(ref _nFfnOut);
                FreeAligned(ref _nLogits);
                FreeAligned(ref _nCtxProj);
                throw;
            }
        }

        EnsureBlockScratch(Options.BlockSize, Options.MaxCandidatesPerSlot);
    }

    public static DFlashDraftProvider CreateContextConditioned(
        IHiddenStateSource trunk,
        DFlashOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(trunk);
        return new DFlashDraftProvider(trunk, weights: null, options: options);
    }

    public static DFlashDraftProvider? TryCreateFromModel(
        IHiddenStateSource trunk,
        ModelWeights weights,
        DFlashOptions? options = null,
        DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunk);
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Gguf == null) return null;

        var ownedFloatBuffers = new List<IntPtr>();
        return CreateFromGguf(trunk, weights, weights.Gguf, isSidecar: false, ownedFloatBuffers, options, subset);
    }

    public static DFlashDraftProvider LoadFromSidecar(
        IHiddenStateSource trunk,
        ModelWeights weights,
        string sidecarGgufPath,
        DFlashOptions? options = null,
        DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunk);
        ArgumentNullException.ThrowIfNull(weights);
        if (string.IsNullOrWhiteSpace(sidecarGgufPath))
            throw new ArgumentException("DFlash sidecar GGUF path must not be empty.", nameof(sidecarGgufPath));

        GgufFile sidecarGguf = GgufFile.Open(sidecarGgufPath);
        List<IntPtr> ownedFloatBuffers = new();
        try
        {
            DFlashDraftProvider? provider = CreateFromGguf(
                trunk,
                weights,
                sidecarGguf,
                isSidecar: true,
                ownedFloatBuffers,
                options,
                subset);
            if (provider == null)
            {
                throw new InvalidDataException($"DFlash sidecar '{sidecarGgufPath}' does not contain valid DFlash block-diffusion tensors.");
            }
            return provider;
        }
        catch
        {
            foreach (IntPtr ptr in ownedFloatBuffers)
            {
                if (ptr != IntPtr.Zero) NativeMemory.AlignedFree((void*)ptr);
            }
            sidecarGguf.Dispose();
            throw;
        }
    }

    public void Reset()
    {
        lock (_syncLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _syncedTokens.Clear();
            _transitions.Clear();
        }
    }

    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
    {
        lock (_syncLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return DraftCore(
                tokens,
                recentTokens: default,
                maxDraftTokens,
                withProbs: false,
                sampling: null,
                sampler: null,
                targetVocabSize: 0,
                draftTokens,
                draftProbs: default);
        }
    }

    public int DraftWithProbabilities(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        SamplingOptions sampling,
        Sampler sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentNullException.ThrowIfNull(sampler);

        lock (_syncLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (targetVocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetVocabSize), targetVocabSize, "targetVocabSize must be positive.");

            return DraftCore(
                tokens,
                recentTokens,
                maxDraftTokens,
                withProbs: true,
                sampling,
                sampler,
                targetVocabSize,
                draftTokens,
                draftProbs);
        }
    }

    public void OnVerificationResult(int draftedCount, int acceptedCount, double roundElapsedMs)
    {
        lock (_syncLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (draftedCount <= 0) return;

            int clampedAccepted = Math.Clamp(acceptedCount, 0, draftedCount);
            int tested = Math.Min(draftedCount, clampedAccepted + 1);
            double roundRate = (double)clampedAccepted / Math.Max(1, tested);
            SmoothedAcceptanceRate = (0.85 * SmoothedAcceptanceRate) + (0.15 * roundRate);

            if (Options.EnableAdaptiveConfidence)
            {
                int minBlock = Math.Min(2, Options.BlockSize);
                if (SmoothedAcceptanceRate < 0.25 && EffectiveBlockSize > minBlock)
                {
                    EffectiveBlockSize--;
                }
                else if (SmoothedAcceptanceRate >= 0.55 && EffectiveBlockSize < Options.BlockSize)
                {
                    EffectiveBlockSize++;
                }
            }
        }
    }

    private int DraftCore(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        bool withProbs,
        SamplingOptions? sampling,
        Sampler? sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        if (tokens.IsEmpty || maxDraftTokens <= 0 || draftTokens.IsEmpty)
            return 0;

        int blockLimit = _weights != null ? Math.Min(EffectiveBlockSize, _weights.BlockSize) : EffectiveBlockSize;
        int blockSize = Math.Min(blockLimit, Math.Min(maxDraftTokens, draftTokens.Length));
        if (blockSize <= 0)
            return 0;

        if (withProbs && draftProbs.Length < blockSize * targetVocabSize)
            throw new ArgumentException("draftProbs buffer is too small.", nameof(draftProbs));

        EnsureBlockScratch(blockSize, Options.MaxCandidatesPerSlot);

        int steps = Options.DiffusionSteps;
        if (_weights != null && _trunk != null)
        {
            return DraftNeuralCore(
                tokens,
                recentTokens,
                blockSize,
                steps,
                withProbs,
                sampling,
                sampler,
                targetVocabSize,
                draftTokens,
                draftProbs);
        }

        SyncTransitions(tokens);
        if (_transitions.Count == 0)
            return 0;

        int maskId = Options.MaskTokenId;
        for (int j = 0; j < blockSize; j++)
        {
            _blockTokens[j] = maskId;
            _blockConf[j] = 0f;
            _slotBest[j] = maskId;
            _slotConf[j] = 0f;
            _slotCandCount[j] = 0;
        }

        ReadOnlySpan<float> trunkHidden = ReadOnlySpan<float>.Empty;
        if (_trunk != null && _trunk.HiddenDim > 0)
        {
            trunkHidden = _trunk.GetLastHiddenState();
        }

        int maxCands = Options.MaxCandidatesPerSlot;
        float temp = Options.DiffusionTemperature;

        for (int step = 0; step < steps; step++)
        {
            Array.Copy(_blockTokens, _prevBlockTokens, blockSize);

            for (int j = 0; j < blockSize; j++)
            {
                int slotBase = j * maxCands;
                int candCount = 0;

                // 1. Left causal context (from prompt tail or unmasked left slots in block)
                int left1 = ResolveContextToken(tokens, _prevBlockTokens, j - 1);
                int left2 = ResolveContextToken(tokens, _prevBlockTokens, j - 2);
                int left3 = ResolveContextToken(tokens, _prevBlockTokens, j - 3);

                if (Options.MaxNgramOrder >= 3 && left3 >= 0 && left2 >= 0 && left1 >= 0)
                {
                    AddFromBucket(MakeTrigramKey(left3, left2, left1), 2.5f, targetVocabSize, maxCands, slotBase, ref candCount);
                }
                if (Options.MaxNgramOrder >= 2 && left2 >= 0 && left1 >= 0)
                {
                    AddFromBucket(MakeBigramKey(left2, left1), 1.8f, targetVocabSize, maxCands, slotBase, ref candCount);
                }
                if (left1 >= 0)
                {
                    AddFromBucket(MakeUnigramKey(left1), 1.0f, targetVocabSize, maxCands, slotBase, ref candCount);
                }

                // 2. Right bidirectional context from unmasked slot j+1
                if (j + 1 < blockSize && _prevBlockTokens[j + 1] >= 0)
                {
                    AddFromBucket(
                        MakeBackwardBigramKey(_prevBlockTokens[j + 1]),
                        Options.BidirectionalWeight * 1.6f,
                        targetVocabSize,
                        maxCands,
                        slotBase,
                        ref candCount);
                }

                // 3. Multi-offset skip-gram anchors from prompt tail and earlier unmasked block slots
                int anchor1 = tokens[^1];
                if (j >= 1)
                {
                    AddFromBucket(
                        MakeSkipGramKey(anchor1, j + 1),
                        Options.SkipGramWeight * 1.3f,
                        targetVocabSize,
                        maxCands,
                        slotBase,
                        ref candCount);
                }
                if (tokens.Length >= 2)
                {
                    AddFromBucket(
                        MakeSkipGramKey(tokens[^2], j + 2),
                        Options.SkipGramWeight * 1.0f,
                        targetVocabSize,
                        maxCands,
                        slotBase,
                        ref candCount);
                }
                for (int m = 0; m < j - 1; m++)
                {
                    int unmasked = _prevBlockTokens[m];
                    if (unmasked >= 0)
                    {
                        AddFromBucket(
                            MakeSkipGramKey(unmasked, j - m),
                            Options.SkipGramWeight * 0.9f,
                            targetVocabSize,
                            maxCands,
                            slotBase,
                            ref candCount);
                    }
                }

                if (candCount == 0)
                {
                    _slotCandCount[j] = 0;
                    _slotBest[j] = maskId;
                    _slotConf[j] = 0f;
                    continue;
                }

                // 4. Self-consistency bonus & optional lightweight trunk embedding dot-product bias
                int prevTok = _prevBlockTokens[j];
                bool useTrunkBias = !trunkHidden.IsEmpty && _trunk != null && candCount <= 16 && _embScratch.Length >= _trunk.HiddenDim;
                int hDim = useTrunkBias ? _trunk!.HiddenDim : 0;

                float maxScore = float.NegativeInfinity;
                for (int c = 0; c < candCount; c++)
                {
                    int idx = slotBase + c;
                    int tok = _slotCandIds[idx];
                    float s = _slotCandScores[idx];
                    if (prevTok >= 0 && tok == prevTok)
                    {
                        s += 0.25f;
                    }
                    if (useTrunkBias)
                    {
                        _trunk!.CopyTokenEmbedding(tok, _embScratch.AsSpan(0, hDim));
                        float dot = 0f;
                        for (int d = 0; d < hDim; d++)
                        {
                            dot += trunkHidden[d] * _embScratch[d];
                        }
                        s += 0.1f * MathF.Tanh(dot / MathF.Max(1f, MathF.Sqrt(hDim)));
                    }
                    _slotCandScores[idx] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0f;
                int bestIdx = 0;
                float bestProb = 0f;
                for (int c = 0; c < candCount; c++)
                {
                    int idx = slotBase + c;
                    float e = MathF.Exp((_slotCandScores[idx] - maxScore) / temp);
                    _slotCandProbs[idx] = e;
                    sumExp += e;
                }

                if (sumExp > 0f && float.IsFinite(sumExp))
                {
                    float invSum = 1f / sumExp;
                    for (int c = 0; c < candCount; c++)
                    {
                        int idx = slotBase + c;
                        float p = _slotCandProbs[idx] * invSum;
                        _slotCandProbs[idx] = p;
                        if (p > bestProb)
                        {
                            bestProb = p;
                            bestIdx = c;
                        }
                    }
                }
                else
                {
                    _slotCandProbs[slotBase] = 1f;
                    bestIdx = 0;
                    bestProb = 1f;
                }

                _slotCandCount[j] = candCount;
                _slotBest[j] = _slotCandIds[slotBase + bestIdx];
                _slotConf[j] = bestProb;
            }

            if (step < steps - 1)
            {
                // Remask previously unmasked slots whose confidence dropped below RemaskThreshold
                for (int j = 0; j < blockSize; j++)
                {
                    if (_blockTokens[j] >= 0 && _slotConf[j] < Options.RemaskThreshold)
                    {
                        _blockTokens[j] = maskId;
                        _blockConf[j] = 0f;
                    }
                }

                // Cosine schedule target number of unmasked slots in [1, B]
                int targetUnmasked = Math.Clamp(
                    (int)MathF.Ceiling(blockSize * (1f - MathF.Cos(MathF.PI * (step + 1) / (2f * steps)))),
                    1,
                    blockSize);

                for (int j = 0; j < blockSize; j++) _slotOrder[j] = j;
                // Insertion sort slot indices by descending confidence
                for (int i = 1; i < blockSize; i++)
                {
                    int key = _slotOrder[i];
                    float keyConf = _slotConf[key];
                    int k = i - 1;
                    while (k >= 0 && _slotConf[_slotOrder[k]] < keyConf)
                    {
                        _slotOrder[k + 1] = _slotOrder[k];
                        k--;
                    }
                    _slotOrder[k + 1] = key;
                }

                int unmaskedNow = 0;
                for (int j = 0; j < blockSize; j++)
                {
                    if (_blockTokens[j] >= 0) unmaskedNow++;
                }

                for (int r = 0; r < blockSize && unmaskedNow < targetUnmasked; r++)
                {
                    int slot = _slotOrder[r];
                    if (_blockTokens[slot] < 0 && _slotBest[slot] >= 0 && _slotConf[slot] > 0f)
                    {
                        _blockTokens[slot] = _slotBest[slot];
                        _blockConf[slot] = _slotConf[slot];
                        unmaskedNow++;
                    }
                }
            }
            else
            {
                // Final refinement step: commit best valid prediction for every slot
                for (int j = 0; j < blockSize; j++)
                {
                    if (_slotBest[j] >= 0 && _slotConf[j] > 0f)
                    {
                        _blockTokens[j] = _slotBest[j];
                        _blockConf[j] = _slotConf[j];
                    }
                }
            }
        }

        // Emit contiguous prefix starting at slot 0
        int emitted = 0;
        for (int j = 0; j < blockSize; j++)
        {
            if (_blockTokens[j] < 0) break;
            if (j > 0 && _blockConf[j] < Options.MinEmitConfidence) break;
            emitted++;
        }

        if (emitted <= 0)
            return 0;

        if (!withProbs)
        {
            for (int i = 0; i < emitted; i++)
            {
                draftTokens[i] = _blockTokens[i];
            }
            DraftRounds++;
            TotalDiffusionStepsExecuted += steps;
            return emitted;
        }

        // Calibrate normalized draft probabilities per emitted slot for Leviathan rejection sampling
        draftProbs.Slice(0, emitted * targetVocabSize).Clear();
        bool isGreedy = sampling!.Temperature <= Sampler.GreedyTemperatureThreshold || sampling.TopK == 1;

        for (int i = 0; i < emitted; i++)
        {
            int candCount = _slotCandCount[i];
            int slotBase = i * maxCands;
            Span<float> qRow = draftProbs.Slice(i * targetVocabSize, targetVocabSize);

            if (candCount <= 0)
            {
                int tok = _blockTokens[i];
                if ((uint)tok >= (uint)targetVocabSize)
                    return i;
                draftTokens[i] = tok;
                qRow[tok] = 1.0f;
                continue;
            }

            // Copy raw slot scores and apply repetition penalty over recentTokens + already drafted tokens
            int totalRecent = recentTokens.Length + i;
            if (_recentScratch.Length < totalRecent)
            {
                _recentScratch = new int[Math.Max(totalRecent, 64)];
            }
            if (!recentTokens.IsEmpty)
            {
                recentTokens.CopyTo(_recentScratch);
            }
            for (int d = 0; d < i; d++)
            {
                _recentScratch[recentTokens.Length + d] = draftTokens[d];
            }
            ReadOnlySpan<int> history = _recentScratch.AsSpan(0, totalRecent);

            // Score candidates with RepetitionPenalty
            for (int c = 0; c < candCount; c++)
            {
                int tok = _slotCandIds[slotBase + c];
                float score = _slotCandScores[slotBase + c];
                if (sampling.RepetitionPenalty != 1.0f && ContainsToken(history, tok))
                {
                    score = score > 0f ? score / sampling.RepetitionPenalty : score * sampling.RepetitionPenalty;
                }
                _slotCandProbs[slotBase + c] = score;
            }

            if (isGreedy)
            {
                int bestIdx = 0;
                float bestScore = _slotCandProbs[slotBase];
                for (int c = 1; c < candCount; c++)
                {
                    float s = _slotCandProbs[slotBase + c];
                    if (s > bestScore)
                    {
                        bestScore = s;
                        bestIdx = c;
                    }
                }
                int chosenTok = _slotCandIds[slotBase + bestIdx];
                draftTokens[i] = chosenTok;
                qRow[chosenTok] = 1.0f;
                continue;
            }

            // Sort candidate indices by penalized score descending for TopK + TopP
            for (int c = 0; c < candCount; c++) _slotOrder[c] = c;
            for (int a = 1; a < candCount; a++)
            {
                int key = _slotOrder[a];
                float keyS = _slotCandProbs[slotBase + key];
                int b = a - 1;
                while (b >= 0 && _slotCandProbs[slotBase + _slotOrder[b]] < keyS)
                {
                    _slotOrder[b + 1] = _slotOrder[b];
                    b--;
                }
                _slotOrder[b + 1] = key;
            }

            int k = sampling.TopK <= 0 ? candCount : Math.Min(sampling.TopK, candCount);
            float invTemp = 1.0f / sampling.Temperature;
            float maxLogit = _slotCandProbs[slotBase + _slotOrder[0]] * invTemp;
            float sumExp = 0f;
            for (int r = 0; r < k; r++)
            {
                int c = _slotOrder[r];
                float e = MathF.Exp((_slotCandProbs[slotBase + c] * invTemp) - maxLogit);
                _slotCandProbs[slotBase + c] = e;
                sumExp += e;
            }

            if (!(sumExp > 0f) || !float.IsFinite(sumExp))
            {
                int fallbackTok = _slotCandIds[slotBase + _slotOrder[0]];
                draftTokens[i] = fallbackTok;
                qRow[fallbackTok] = 1.0f;
                continue;
            }

            float invSum = 1f / sumExp;
            float cumProb = 0f;
            int activeCount = 0;
            for (int r = 0; r < k; r++)
            {
                int c = _slotOrder[r];
                float prob = _slotCandProbs[slotBase + c] * invSum;
                _slotCandProbs[slotBase + c] = prob;
                cumProb += prob;
                activeCount++;
                if (cumProb >= sampling.TopP) break;
            }

            float invTrunc = cumProb > 0f && float.IsFinite(cumProb) ? 1f / cumProb : 1f;
            double draw = sampler!.Rng.NextDouble();
            double acc = 0.0;
            int sampledTok = _slotCandIds[slotBase + _slotOrder[0]];
            bool picked = false;

            for (int r = 0; r < activeCount; r++)
            {
                int c = _slotOrder[r];
                float p = _slotCandProbs[slotBase + c] * invTrunc;
                int tok = _slotCandIds[slotBase + c];
                if ((uint)tok < (uint)targetVocabSize)
                {
                    qRow[tok] = p;
                    acc += p;
                    if (!picked && draw <= acc)
                    {
                        sampledTok = tok;
                        picked = true;
                    }
                }
            }

            if (qRow[sampledTok] <= 0f)
            {
                sampledTok = _slotCandIds[slotBase + _slotOrder[0]];
                if (qRow[sampledTok] <= 0f)
                {
                    qRow.Clear();
                    qRow[sampledTok] = 1.0f;
                }
            }

            draftTokens[i] = sampledTok;
        }

        DraftRounds++;
        TotalDiffusionStepsExecuted += steps;
        return emitted;
    }

    private int DraftNeuralCore(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int blockSize,
        int steps,
        bool withProbs,
        SamplingOptions? sampling,
        Sampler? sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        DFlashWeights w = _weights!;
        int dim = w.Dim;
        int nHeads = Math.Max(1, w.NumHeads);
        int nHeadsKv = Math.Max(1, w.NumHeadsKv);
        int headDim = w.HeadDim > 0 ? w.HeadDim : (dim / nHeads);
        int groupSize = Math.Max(1, nHeads / nHeadsKv);
        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        int ffnDim = w.FfnDim > 0 ? w.FfnDim : dim;
        float attnScale = 1.0f / MathF.Sqrt(headDim);
        float rmsEps = w.RmsNormEps > 0f ? w.RmsNormEps : 1e-6f;

        ReadOnlySpan<float> latest = _trunk!.GetLastHiddenState();
        Span<float> ctxSpan = new Span<float>(_nNormX, dim);
        ctxSpan.Clear();
        if (!latest.IsEmpty)
        {
            latest.Slice(0, Math.Min(dim, latest.Length)).CopyTo(ctxSpan);
        }

        if (w.ContextNormWeight != null)
        {
            QuantKernels.RMSNorm(_nNormX, w.ContextNormWeight, _nCtxProj, dim, rmsEps);
        }
        else
        {
            Buffer.MemoryCopy(_nNormX, _nCtxProj, (ulong)dim * sizeof(float), (ulong)dim * sizeof(float));
        }

        if (w.ContextProjWeight != null)
        {
            float* sums = ComputeSumsIfAligned(_nCtxProj, dim, _nNormSums);
            QuantKernels.MatVecMul(w.ContextProjType, w.ContextProjWeight, _nCtxProj, _nAttnProj, dim, dim, sums);
            Buffer.MemoryCopy(_nAttnProj, _nCtxProj, (ulong)dim * sizeof(float), (ulong)dim * sizeof(float));
        }

        int maskId = Options.MaskTokenId;
        for (int j = 0; j < blockSize; j++)
        {
            _blockTokens[j] = maskId;
            _blockConf[j] = 0f;
            _slotBest[j] = maskId;
            _slotConf[j] = 0f;
        }

        if (_neuralProbsScratch == null || _neuralProbsScratch.Length < blockSize * w.VocabSize)
        {
            _neuralProbsScratch = new float[blockSize * w.VocabSize];
        }

        for (int step = 0; step < steps; step++)
        {
            // Build per-slot initial hidden states
            for (int j = 0; j < blockSize; j++)
            {
                float* dst = _nBlockHidden + (long)j * dim;
                Buffer.MemoryCopy(_nCtxProj, dst, (ulong)dim * sizeof(float), (ulong)dim * sizeof(float));

                if (w.SlotPosEmbeddings != null && j < w.BlockSize)
                {
                    QuantKernels.VecAdd(w.SlotPosEmbeddings + (long)j * dim, dst, dim);
                }

                if (_blockTokens[j] >= 0)
                {
                    _trunk.CopyTokenEmbedding(_blockTokens[j], _embScratch.AsSpan(0, Math.Min(dim, _embScratch.Length)));
                    fixed (float* pEmb = _embScratch)
                    {
                        QuantKernels.VecAdd(pEmb, dst, Math.Min(dim, _embScratch.Length));
                    }
                }
                else if (w.MaskEmbedding != null)
                {
                    QuantKernels.VecAdd(w.MaskEmbedding, dst, dim);
                }
            }

            // Bidirectional (non-causal) block-diffusion layers
            for (int l = 0; l < w.Layers.Length; l++)
            {
                LayerWeights layer = w.Layers[l];

                for (int j = 0; j < blockSize; j++)
                {
                    float* hSlot = _nBlockHidden + (long)j * dim;
                    QuantKernels.RMSNorm(hSlot, layer.AttnNormWeight, _nNormX, dim, rmsEps);
                    float* normSums = ComputeSumsIfAligned(_nNormX, dim, _nNormSums);
                    QuantKernels.MatVecMul(layer.QType, layer.QWeight, _nNormX, _nQ + (long)j * qDim, dim, qDim, normSums);
                    QuantKernels.MatVecMul(layer.KType, layer.KWeight, _nNormX, _nK + (long)j * kvDim, dim, kvDim, normSums);
                    QuantKernels.MatVecMul(layer.VType, layer.VWeight, _nNormX, _nV + (long)j * kvDim, dim, kvDim, normSums);
                }

                // Non-causal full attention across all slots j in [0..blockSize-1]
                for (int j = 0; j < blockSize; j++)
                {
                    for (int h = 0; h < nHeads; h++)
                    {
                        int hKv = h / groupSize;
                        float* qHead = _nQ + (long)j * qDim + h * headDim;
                        float* scores = _nScores + (long)j * blockSize;

                        for (int m = 0; m < blockSize; m++)
                        {
                            float* kHead = _nK + (long)m * kvDim + hKv * headDim;
                            scores[m] = QuantKernels.VecDotF32(qHead, kHead, headDim) * attnScale;
                        }

                        QuantKernels.Softmax(scores, blockSize);

                        float* outHead = _nAttnOut + h * headDim;
                        for (int d = 0; d < headDim; d++) outHead[d] = 0f;
                        for (int m = 0; m < blockSize; m++)
                        {
                            float* vHead = _nV + (long)m * kvDim + hKv * headDim;
                            QuantKernels.VecAddWeighted(vHead, outHead, scores[m], headDim);
                        }
                    }

                    float* attnSums = ComputeSumsIfAligned(_nAttnOut, qDim, _nAttnSums);
                    QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _nAttnOut, _nAttnProj, qDim, dim, attnSums);
                    float* hSlot = _nBlockHidden + (long)j * dim;
                    QuantKernels.VecAdd(_nAttnProj, hSlot, dim);

                    QuantKernels.RMSNorm(hSlot, layer.FfnNormWeight, _nNormX, dim, rmsEps);
                    float* ffnInSums = ComputeSumsIfAligned(_nNormX, dim, _nNormSums);
                    QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _nNormX, _nGate, dim, ffnDim, ffnInSums);
                    QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _nNormX, _nUp, dim, ffnDim, ffnInSums);
                    QuantKernels.SwiGLU(_nGate, _nUp, _nFfnAct, ffnDim);
                    float* ffnOutSums = ComputeSumsIfAligned(_nFfnAct, ffnDim, _nFfnSums);
                    QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _nFfnAct, _nFfnOut, ffnDim, dim, ffnOutSums);
                    QuantKernels.VecAdd(_nFfnOut, hSlot, dim);
                }
            }

            // Head prediction per slot
            int effectiveVocab = (targetVocabSize > 0 && targetVocabSize < w.VocabSize) ? targetVocabSize : w.VocabSize;
            for (int j = 0; j < blockSize; j++)
            {
                float* hSlot = _nBlockHidden + (long)j * dim;
                if (w.SharedHeadNormWeight != null)
                {
                    QuantKernels.RMSNorm(hSlot, w.SharedHeadNormWeight, _nNormX, dim, rmsEps);
                }
                else
                {
                    Buffer.MemoryCopy(hSlot, _nNormX, (ulong)dim * sizeof(float), (ulong)dim * sizeof(float));
                }

                Span<float> slotProbs = _neuralProbsScratch.AsSpan(j * w.VocabSize, w.VocabSize);
                if (_packedHead != null)
                {
                    if (withProbs && step == steps - 1)
                    {
                        _slotBest[j] = _packedHead.SampleWithProbabilities(_nNormX, sampling!, sampler!, recentTokens, slotProbs, effectiveVocab);
                        _slotConf[j] = slotProbs[_slotBest[j]];
                    }
                    else
                    {
                        _slotBest[j] = _packedHead.SampleGreedy(_nNormX, effectiveVocab);
                        _slotConf[j] = 0.85f;
                    }
                }
                else
                {
                    float* headSums = ComputeSumsIfAligned(_nNormX, dim, _nNormSums);
                    QuantKernels.MatVecMul(w.SharedHeadType, w.SharedHeadWeight, _nNormX, _nLogits, dim, w.VocabSize, headSums);
                    for (int v = effectiveVocab; v < w.VocabSize; v++)
                    {
                        _nLogits[v] = float.NegativeInfinity;
                    }

                    int bestId = 0;
                    float bestVal = float.NegativeInfinity;
                    for (int v = 0; v < effectiveVocab; v++)
                    {
                        float val = _nLogits[v];
                        if (val > bestVal)
                        {
                            bestVal = val;
                            bestId = v;
                        }
                    }

                    // Compute peak softmax confidence
                    float sumExp = 0f;
                    for (int v = 0; v < effectiveVocab; v++)
                    {
                        sumExp += MathF.Exp(_nLogits[v] - bestVal);
                    }
                    _slotBest[j] = bestId;
                    _slotConf[j] = sumExp > 0f && float.IsFinite(sumExp) ? (1.0f / sumExp) : 1.0f;

                    if (withProbs && step == steps - 1)
                    {
                        slotProbs.Clear();
                        Sampler.ComputeFilteredDistribution(
                            new ReadOnlySpan<float>(_nLogits, effectiveVocab),
                            sampling!,
                            recentTokens,
                            slotProbs.Slice(0, effectiveVocab));
                        _slotBest[j] = Sampler.SampleFromDistribution(slotProbs.Slice(0, effectiveVocab), sampler!.Rng);
                        _slotConf[j] = slotProbs[_slotBest[j]];
                    }
                }
            }

            if (step < steps - 1)
            {
                for (int j = 0; j < blockSize; j++)
                {
                    if (_blockTokens[j] >= 0 && _slotConf[j] < Options.RemaskThreshold)
                    {
                        _blockTokens[j] = maskId;
                        _blockConf[j] = 0f;
                    }
                }

                int targetUnmasked = Math.Clamp(
                    (int)MathF.Ceiling(blockSize * (1f - MathF.Cos(MathF.PI * (step + 1) / (2f * steps)))),
                    1,
                    blockSize);

                for (int j = 0; j < blockSize; j++) _slotOrder[j] = j;
                for (int i = 1; i < blockSize; i++)
                {
                    int key = _slotOrder[i];
                    float keyConf = _slotConf[key];
                    int k = i - 1;
                    while (k >= 0 && _slotConf[_slotOrder[k]] < keyConf)
                    {
                        _slotOrder[k + 1] = _slotOrder[k];
                        k--;
                    }
                    _slotOrder[k + 1] = key;
                }

                int unmaskedNow = 0;
                for (int j = 0; j < blockSize; j++)
                {
                    if (_blockTokens[j] >= 0) unmaskedNow++;
                }

                for (int r = 0; r < blockSize && unmaskedNow < targetUnmasked; r++)
                {
                    int slot = _slotOrder[r];
                    if (_blockTokens[slot] < 0 && _slotBest[slot] >= 0)
                    {
                        _blockTokens[slot] = _slotBest[slot];
                        _blockConf[slot] = _slotConf[slot];
                        unmaskedNow++;
                    }
                }
            }
            else
            {
                for (int j = 0; j < blockSize; j++)
                {
                    if (_slotBest[j] >= 0)
                    {
                        _blockTokens[j] = _slotBest[j];
                        _blockConf[j] = _slotConf[j];
                    }
                }
            }
        }

        int emitted = 0;
        for (int j = 0; j < blockSize; j++)
        {
            if (_blockTokens[j] < 0) break;
            if (j > 0 && _blockConf[j] < Options.MinEmitConfidence) break;
            draftTokens[j] = _blockTokens[j];
            if (withProbs)
            {
                Span<float> qRow = draftProbs.Slice(j * targetVocabSize, targetVocabSize);
                qRow.Clear();
                int copyLen = Math.Min(targetVocabSize, w.VocabSize);
                _neuralProbsScratch.AsSpan(j * w.VocabSize, copyLen).CopyTo(qRow);
                if ((uint)draftTokens[j] < (uint)targetVocabSize && qRow[draftTokens[j]] <= 0f)
                {
                    qRow.Clear();
                    qRow[draftTokens[j]] = 1.0f;
                }
            }
            emitted++;
        }

        DraftRounds++;
        TotalDiffusionStepsExecuted += steps;
        return emitted;
    }

    private void SyncTransitions(ReadOnlySpan<int> tokens)
    {
        int lcp = 0;
        int minLen = Math.Min(_syncedTokens.Count, tokens.Length);
        while (lcp < minLen && _syncedTokens[lcp] == tokens[lcp])
        {
            lcp++;
        }

        if (lcp < _syncedTokens.Count)
        {
            _syncedTokens.Clear();
            _transitions.Clear();
            lcp = 0;
        }

        int maxSkip = Math.Max(2, Options.BlockSize + 1);
        for (int p = lcp; p < tokens.Length; p++)
        {
            int cur = tokens[p];
            if (p >= 1)
            {
                int prev1 = tokens[p - 1];
                RecordTransition(MakeUnigramKey(prev1), cur, 1.0f);
                RecordTransition(MakeBackwardBigramKey(cur), prev1, 1.0f);
            }
            if (p >= 2 && Options.MaxNgramOrder >= 2)
            {
                RecordTransition(MakeBigramKey(tokens[p - 2], tokens[p - 1]), cur, 1.0f);
            }
            if (p >= 3 && Options.MaxNgramOrder >= 3)
            {
                RecordTransition(MakeTrigramKey(tokens[p - 3], tokens[p - 2], tokens[p - 1]), cur, 1.0f);
            }
            for (int d = 2; d <= maxSkip && p - d >= 0; d++)
            {
                RecordTransition(MakeSkipGramKey(tokens[p - d], d), cur, 1.0f / (1.0f + 0.15f * (d - 1)));
            }

            _syncedTokens.Add(cur);
        }
    }

    private void RecordTransition(ulong key, int targetToken, float weight)
    {
        if (targetToken < 0) return;
        ref CandidateBucket bucket = ref CollectionsMarshal.GetValueRefOrAddDefault(_transitions, key, out bool exists);
        if (!exists && _transitions.Count > Options.MaxTransitionKeys)
        {
            _transitions.Remove(key);
            return;
        }
        bucket.Add(targetToken, weight);
    }

    private void AddFromBucket(
        ulong key,
        float scale,
        int targetVocabSize,
        int maxCands,
        int slotBase,
        ref int candCount)
    {
        if (!_transitions.TryGetValue(key, out CandidateBucket bucket))
            return;

        for (int i = 0; i < bucket.Count; i++)
        {
            int tok = bucket.Tokens[i];
            if (tok < 0 || (targetVocabSize > 0 && tok >= targetVocabSize))
                continue;

            float w = bucket.Weights[i] * scale;

            int existing = -1;
            for (int c = 0; c < candCount; c++)
            {
                if (_slotCandIds[slotBase + c] == tok)
                {
                    existing = c;
                    break;
                }
            }

            if (existing >= 0)
            {
                _slotCandScores[slotBase + existing] += w;
            }
            else if (candCount < maxCands)
            {
                _slotCandIds[slotBase + candCount] = tok;
                _slotCandScores[slotBase + candCount] = w;
                candCount++;
            }
            else
            {
                int minIdx = 0;
                float minScore = _slotCandScores[slotBase];
                for (int c = 1; c < candCount; c++)
                {
                    if (_slotCandScores[slotBase + c] < minScore)
                    {
                        minScore = _slotCandScores[slotBase + c];
                        minIdx = c;
                    }
                }
                if (w > minScore)
                {
                    _slotCandIds[slotBase + minIdx] = tok;
                    _slotCandScores[slotBase + minIdx] = w;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ResolveContextToken(ReadOnlySpan<int> tokens, int[] prevBlock, int blockOffset)
    {
        if (blockOffset >= 0)
        {
            return blockOffset < prevBlock.Length ? prevBlock[blockOffset] : -1;
        }
        int idx = tokens.Length + blockOffset;
        return (uint)idx < (uint)tokens.Length ? tokens[idx] : -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ContainsToken(ReadOnlySpan<int> history, int token)
    {
        for (int i = 0; i < history.Length; i++)
        {
            if (history[i] == token) return true;
        }
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MakeUnigramKey(int a) => (1UL << 60) | (uint)a;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MakeBigramKey(int a, int b)
    {
        ulong h = ((ulong)(uint)a * 0x9E3779B97F4A7C15UL) ^ ((ulong)(uint)b * 0xBF58476D1CE4E5B9UL);
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9UL;
        h ^= h >> 27;
        return (2UL << 60) | (h & 0x0FFF_FFFF_FFFF_FFFFUL);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MakeTrigramKey(int a, int b, int c)
    {
        ulong h = ((ulong)(uint)a * 0x9E3779B97F4A7C15UL) ^ ((ulong)(uint)b * 0xBF58476D1CE4E5B9UL) ^ ((ulong)(uint)c * 0x94D049BB133111EBUL);
        h ^= h >> 30;
        h *= 0x94D049BB133111EBUL;
        h ^= h >> 27;
        return (3UL << 60) | (h & 0x0FFF_FFFF_FFFF_FFFFUL);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MakeBackwardBigramKey(int nextTok) => (4UL << 60) | (uint)nextTok;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MakeSkipGramKey(int anchor, int offset) =>
        (5UL << 60) | ((ulong)(uint)offset << 32) | (uint)anchor;

    private void EnsureBlockScratch(int blockSize, int maxCands)
    {
        if (_blockTokens.Length < blockSize)
        {
            _blockTokens = new int[blockSize];
            _prevBlockTokens = new int[blockSize];
            _blockConf = new float[blockSize];
            _slotBest = new int[blockSize];
            _slotConf = new float[blockSize];
            _slotCandCount = new int[blockSize];
        }
        int orderCap = Math.Max(blockSize, maxCands);
        if (_slotOrder.Length < orderCap)
        {
            _slotOrder = new int[orderCap];
        }
        int totalCands = blockSize * maxCands;
        if (_slotCandIds.Length < totalCands)
        {
            _slotCandIds = new int[totalCands];
            _slotCandScores = new float[totalCands];
            _slotCandProbs = new float[totalCands];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float* ComputeSumsIfAligned(float* x, int len, float* sumsBuf)
    {
        if (len < 32) return null;
        QuantKernels.ComputeBlockSums32(x, sumsBuf, len);
        return sumsBuf;
    }

    private static DFlashDraftProvider? CreateFromGguf(
        IHiddenStateSource trunk,
        ModelWeights weights,
        GgufFile gguf,
        bool isSidecar,
        List<IntPtr>? ownedFloatBuffers,
        DFlashOptions? options,
        DraftVocabularySubset? subset)
    {
        bool hasDFlash = gguf.Tensors.ContainsKey("dflash.mask_embd.weight") ||
                         gguf.Tensors.ContainsKey("blk.0.dflash.attn_q.weight") ||
                         gguf.Tensors.ContainsKey("dflash.blk.0.attn_q.weight");
        if (!hasDFlash)
        {
            return null;
        }

        List<IntPtr> owned = ownedFloatBuffers ?? new List<IntPtr>();
        int initialOwnedCount = owned.Count;
        bool committed = false;

        try
        {
            int dim = weights.EmbeddingLength;
            int ffnDim = weights.FeedForwardLength;
            int nHeads = weights.HeadCount;
            int nHeadsKv = weights.HeadCountKv;
            int headDim = weights.HeadDim;
            int vocabSize = weights.VocabSize;
            if (dim <= 0 || ffnDim <= 0 || nHeads <= 0 || nHeadsKv <= 0 || nHeads < nHeadsKv || (nHeads % nHeadsKv) != 0 || headDim <= 0 || vocabSize <= 0)
                return null;

            long qDimLong = (long)nHeads * headDim;
            long kvDimLong = (long)nHeadsKv * headDim;
            if (qDimLong <= 0 || qDimLong > int.MaxValue || kvDimLong <= 0 || kvDimLong > int.MaxValue)
                return null;
            int qDim = (int)qDimLong;
            int kvDim = (int)kvDimLong;

            if (trunk.HiddenDim > 0 && trunk.HiddenDim != dim)
                return null;

            uint rawBlockSize = gguf.GetMetadataUInt32("dflash.block_size", (uint)(options?.BlockSize ?? 4));
            if (rawBlockSize == 0 || rawBlockSize > MaxDFlashBlockSize)
                return null;
            int blockSize = (int)rawBlockSize;

            uint declaredEmbed = gguf.GetMetadataUInt32($"{gguf.Architecture}.embedding_length", 0);
            if (declaredEmbed == 0)
                declaredEmbed = gguf.GetMetadataUInt32("dflash.embedding_length", 0);
            if (declaredEmbed > 0 && declaredEmbed != (uint)dim)
                return null;

            uint declaredVocab = gguf.GetMetadataUInt32("tokenizer.ggml.tokens.length", 0);
            if (declaredVocab > 0 && declaredVocab != (uint)vocabSize)
                return null;

            // Validate global / shared tensors up front before allocating any unmanaged float buffers.
            byte* headPtr = weights.OutWeight;
            GgufType headType = weights.OutType;
            float* headNormPtr = weights.OutNormWeight;

            GgufTensorInfo? shInfo = null;
            if (TryGetAnyTensor(gguf, out shInfo, "dflash.shared_head.weight", "output.weight") && shInfo != null)
            {
                if (!HasMatrixShape(shInfo, dim, vocabSize, allowLargerOutRows: true))
                    return null;
                headPtr = gguf.GetTensorPointer(shInfo);
                headType = shInfo.Type;
            }
            else
            {
                if (headPtr == null || !IsSupportedMatVecType(headType, dim))
                    return null;
                if (weights.Gguf != null &&
                    TryGetAnyTensor(weights.Gguf, out var baseHeadInfo, "output.weight", "token_embd.weight") &&
                    baseHeadInfo != null &&
                    !HasMatrixShape(baseHeadInfo, dim, vocabSize, allowLargerOutRows: true))
                {
                    return null;
                }
            }

            if (TryGetAnyTensor(gguf, out var shNormInfo, "dflash.shared_head.norm.weight", "output_norm.weight") && shNormInfo != null)
            {
                if (!HasVectorShape(shNormInfo, dim))
                    return null;
            }
            else
            {
                shNormInfo = null;
            }

            TryGetAnyTensor(gguf, out var maskInfo, "dflash.mask_embd.weight");
            if (maskInfo != null && !HasVectorShape(maskInfo, dim))
                return null;

            TryGetAnyTensor(gguf, out var posInfo, "dflash.slot_pos_embd.weight");
            if (posInfo != null && !HasSlotPosShape(posInfo, dim, blockSize))
                return null;

            TryGetAnyTensor(gguf, out var ctxNormInfo, "dflash.context_norm.weight");
            if (ctxNormInfo != null && !HasVectorShape(ctxNormInfo, dim))
                return null;

            TryGetAnyTensor(gguf, out var ctxProjInfo, "dflash.context_proj.weight");
            if (ctxProjInfo != null && !HasMatrixShape(ctxProjInfo, dim, dim))
                return null;

            var layerSpecs = new List<(
                GgufTensorInfo Q, GgufTensorInfo K, GgufTensorInfo V, GgufTensorInfo O,
                GgufTensorInfo? An, GgufTensorInfo? Fn,
                GgufTensorInfo Fg, GgufTensorInfo Fu, GgufTensorInfo Fd)>();

            for (int i = 0; i < 16; i++)
            {
                if (!TryGetAnyTensor(gguf, out var qInfo, $"blk.{i}.dflash.attn_q.weight", $"dflash.blk.{i}.attn_q.weight") || qInfo == null)
                    break;

                TryGetAnyTensor(gguf, out var kInfo, $"blk.{i}.dflash.attn_k.weight", $"dflash.blk.{i}.attn_k.weight");
                TryGetAnyTensor(gguf, out var vInfo, $"blk.{i}.dflash.attn_v.weight", $"dflash.blk.{i}.attn_v.weight");
                TryGetAnyTensor(gguf, out var oInfo, $"blk.{i}.dflash.attn_output.weight", $"dflash.blk.{i}.attn_output.weight");
                TryGetAnyTensor(gguf, out var anInfo, $"blk.{i}.dflash.attn_norm.weight", $"dflash.blk.{i}.attn_norm.weight");
                TryGetAnyTensor(gguf, out var fnInfo, $"blk.{i}.dflash.ffn_norm.weight", $"dflash.blk.{i}.ffn_norm.weight");
                TryGetAnyTensor(gguf, out var fgInfo, $"blk.{i}.dflash.ffn_gate.weight", $"dflash.blk.{i}.ffn_gate.weight");
                TryGetAnyTensor(gguf, out var fuInfo, $"blk.{i}.dflash.ffn_up.weight", $"dflash.blk.{i}.ffn_up.weight");
                TryGetAnyTensor(gguf, out var fdInfo, $"blk.{i}.dflash.ffn_down.weight", $"dflash.blk.{i}.ffn_down.weight");

                if (kInfo == null || vInfo == null || oInfo == null || fgInfo == null || fuInfo == null || fdInfo == null)
                    return null;

                if (!HasMatrixShape(qInfo, dim, qDim) ||
                    !HasMatrixShape(kInfo, dim, kvDim) ||
                    !HasMatrixShape(vInfo, dim, kvDim) ||
                    !HasMatrixShape(oInfo, qDim, dim) ||
                    !HasMatrixShape(fgInfo, dim, ffnDim) ||
                    !HasMatrixShape(fuInfo, dim, ffnDim) ||
                    !HasMatrixShape(fdInfo, ffnDim, dim))
                {
                    return null;
                }

                if (anInfo != null && !HasVectorShape(anInfo, dim))
                    return null;
                if (fnInfo != null && !HasVectorShape(fnInfo, dim))
                    return null;

                layerSpecs.Add((qInfo, kInfo, vInfo, oInfo, anInfo, fnInfo, fgInfo, fuInfo, fdInfo));
            }

            if (layerSpecs.Count == 0)
                return null;

            var layerList = new List<LayerWeights>(layerSpecs.Count);
            foreach (var spec in layerSpecs)
            {
                layerList.Add(new LayerWeights
                {
                    QWeight = gguf.GetTensorPointer(spec.Q),
                    QType = spec.Q.Type,
                    KWeight = gguf.GetTensorPointer(spec.K),
                    KType = spec.K.Type,
                    VWeight = gguf.GetTensorPointer(spec.V),
                    VType = spec.V.Type,
                    AttnOutWeight = gguf.GetTensorPointer(spec.O),
                    AttnOutType = spec.O.Type,
                    AttnNormWeight = spec.An != null ? LoadFloatWeight(gguf, spec.An, owned) : null,
                    FfnNormWeight = spec.Fn != null ? LoadFloatWeight(gguf, spec.Fn, owned) : null,
                    FfnGateWeight = gguf.GetTensorPointer(spec.Fg),
                    FfnGateType = spec.Fg.Type,
                    FfnUpWeight = gguf.GetTensorPointer(spec.Fu),
                    FfnUpType = spec.Fu.Type,
                    FfnDownWeight = gguf.GetTensorPointer(spec.Fd),
                    FfnDownType = spec.Fd.Type
                });
            }

            if (shNormInfo != null)
            {
                headNormPtr = LoadFloatWeight(gguf, shNormInfo, owned);
            }

            var dfWeights = new DFlashWeights
            {
                Dim = dim,
                FfnDim = ffnDim,
                NumHeads = nHeads,
                NumHeadsKv = nHeadsKv,
                HeadDim = headDim,
                VocabSize = vocabSize,
                BlockSize = blockSize,
                RmsNormEps = weights.RmsNormEps,
                MaskEmbedding = maskInfo != null ? LoadFloatWeight(gguf, maskInfo, owned) : null,
                SlotPosEmbeddings = posInfo != null ? LoadFloatWeight(gguf, posInfo, owned) : null,
                ContextNormWeight = ctxNormInfo != null ? LoadFloatWeight(gguf, ctxNormInfo, owned) : null,
                ContextProjWeight = ctxProjInfo != null ? gguf.GetTensorPointer(ctxProjInfo) : null,
                ContextProjType = ctxProjInfo?.Type ?? GgufType.F32,
                Layers = layerList.ToArray(),
                SharedHeadNormWeight = headNormPtr,
                SharedHeadWeight = headPtr,
                SharedHeadType = headType,
                OwnedSidecarGguf = isSidecar ? gguf : null,
                OwnedFloatBuffers = owned
            };

            var provider = new DFlashDraftProvider(trunk, dfWeights, options, subset);
            committed = true;
            return provider;
        }
        finally
        {
            if (!committed)
            {
                for (int idx = initialOwnedCount; idx < owned.Count; idx++)
                {
                    IntPtr ptr = owned[idx];
                    if (ptr != IntPtr.Zero) NativeMemory.AlignedFree((void*)ptr);
                }
                if (owned.Count > initialOwnedCount)
                {
                    owned.RemoveRange(initialOwnedCount, owned.Count - initialOwnedCount);
                }
            }
        }
    }

    private static bool TryGetCheckedElementCount(GgufTensorInfo info, out ulong elementCount)
    {
        elementCount = 0;
        if (info.Dimensions == null || info.Dimensions.Length == 0)
            return false;

        ulong total = 1;
        for (int i = 0; i < info.Dimensions.Length; i++)
        {
            ulong d = info.Dimensions[i];
            if (d == 0)
                return false;
            if (total > ulong.MaxValue / d)
                return false;
            total *= d;
        }

        elementCount = total;
        return true;
    }

    private static bool IsSupportedMatVecType(GgufType type, int inCols)
    {
        if (inCols <= 0) return false;
        return type switch
        {
            GgufType.F32 or GgufType.F16 or GgufType.BF16 => true,
            GgufType.Q4_0 or GgufType.Q5_0 or GgufType.Q8_0 or GgufType.IQ4_NL or GgufType.MXFP4 => (inCols % 32) == 0,
            GgufType.Q2_0 or GgufType.GgmlQ2_0 => (inCols % 64) == 0,
            GgufType.Q2_K or GgufType.Q3_K or GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K
                or GgufType.IQ2_XXS or GgufType.IQ2_XS or GgufType.IQ2_S
                or GgufType.IQ3_XXS or GgufType.IQ3_S or GgufType.IQ4_XS => (inCols % 256) == 0,
            _ => false
        };
    }

    private static bool IsSupportedFloatLoadType(GgufType type, int rowWidth)
    {
        if (rowWidth <= 0) return false;
        return type switch
        {
            GgufType.F32 or GgufType.F16 or GgufType.BF16 => true,
            GgufType.Q4_0 or GgufType.Q5_0 or GgufType.Q8_0 or GgufType.IQ4_NL or GgufType.MXFP4 => (rowWidth % 32) == 0,
            GgufType.Q2_0 or GgufType.GgmlQ2_0 => (rowWidth % 64) == 0,
            GgufType.Q2_K or GgufType.Q3_K or GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K
                or GgufType.IQ2_XXS or GgufType.IQ2_XS or GgufType.IQ2_S
                or GgufType.IQ3_XXS or GgufType.IQ3_S or GgufType.IQ4_XS
                or GgufType.Ternary_1_58 or GgufType.PQ2_0 => (rowWidth % 256) == 0,
            _ => false
        };
    }

    private static bool HasVectorShape(GgufTensorInfo? info, int expectedLength)
    {
        if (info == null || expectedLength <= 0)
            return false;
        if (!TryGetCheckedElementCount(info, out ulong count) || count < (ulong)expectedLength || count > int.MaxValue)
            return false;

        if (info.Dimensions.Length == 1)
        {
            if (info.Dimensions[0] < (ulong)expectedLength || info.Dimensions[0] > int.MaxValue)
                return false;
            return IsSupportedFloatLoadType(info.Type, (int)info.Dimensions[0]);
        }

        if (info.Dimensions.Length == 2)
        {
            if (info.Dimensions[0] != (ulong)expectedLength || info.Dimensions[1] < 1)
                return false;
            return IsSupportedFloatLoadType(info.Type, expectedLength);
        }

        return false;
    }

    private static bool HasSlotPosShape(GgufTensorInfo? info, int dim, int blockSize)
    {
        if (info == null || dim <= 0 || blockSize <= 0)
            return false;
        if (!TryGetCheckedElementCount(info, out ulong count) || count > int.MaxValue)
            return false;

        ulong required = (ulong)dim * (ulong)blockSize;
        if (count < required)
            return false;

        if (info.Dimensions.Length == 2)
        {
            if (info.Dimensions[0] != (ulong)dim || info.Dimensions[1] < (ulong)blockSize)
                return false;
            return IsSupportedFloatLoadType(info.Type, dim);
        }

        if (info.Dimensions.Length == 1)
        {
            if (info.Dimensions[0] < required || info.Dimensions[0] > int.MaxValue)
                return false;
            return IsSupportedFloatLoadType(info.Type, (int)info.Dimensions[0]);
        }

        return false;
    }

    private static bool HasMatrixShape(
        GgufTensorInfo? info,
        int expectedInCols,
        int expectedOutRows,
        bool allowLargerOutRows = false)
    {
        if (info == null || expectedInCols <= 0 || expectedOutRows <= 0)
            return false;
        if (!TryGetCheckedElementCount(info, out ulong count))
            return false;
        if (!IsSupportedMatVecType(info.Type, expectedInCols))
            return false;

        ulong required = (ulong)expectedInCols * (ulong)expectedOutRows;
        if (count < required)
            return false;

        if (info.Dimensions.Length == 2)
        {
            if (info.Dimensions[0] != (ulong)expectedInCols)
                return false;
            return allowLargerOutRows
                ? info.Dimensions[1] >= (ulong)expectedOutRows
                : info.Dimensions[1] == (ulong)expectedOutRows;
        }

        return false;
    }

    private static bool TryGetAnyTensor(GgufFile gguf, out GgufTensorInfo? info, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
        {
            if (gguf.TryGetTensor(names[i], out info) && info != null)
                return true;
        }
        info = null;
        return false;
    }

    private static float* LoadFloatWeight(GgufFile gguf, GgufTensorInfo info, List<IntPtr>? ownedFloatBuffers)
    {
        if (info.Type == GgufType.F32)
            return (float*)gguf.GetTensorPointer(info);

        long count = (long)info.ElementCount;
        float* buf = AllocFloats(count);
        ownedFloatBuffers?.Add((IntPtr)buf);
        byte* raw = gguf.GetTensorPointer(info);
        if (info.Type == GgufType.F16)
        {
            Half* h = (Half*)raw;
            for (long i = 0; i < count; i++) buf[i] = (float)h[i];
        }
        else if (info.Type == GgufType.BF16)
        {
            ushort* b = (ushort*)raw;
            for (long i = 0; i < count; i++) buf[i] = BitConverter.UInt32BitsToSingle((uint)b[i] << 16);
        }
        else
        {
            int rowWidth = info.Dimensions.Length > 0 ? (int)info.Dimensions[0] : (int)count;
            int rows = rowWidth > 0 ? (int)(count / rowWidth) : 1;
            for (int r = 0; r < rows; r++)
            {
                QuantKernels.ExtractEmbedding(info.Type, raw, r, buf + (long)r * rowWidth, rowWidth);
            }
        }
        return buf;
    }

    private static float* AllocFloats(long count)
    {
        nuint byteCount = checked((nuint)checked(Math.Max(1L, count) * sizeof(float)));
        return (float*)NativeMemory.AlignedAlloc(byteCount, 64);
    }

    private static void FreeAligned(ref float* ptr)
    {
        if (ptr != null)
        {
            NativeMemory.AlignedFree(ptr);
            ptr = null;
        }
    }

    public void Dispose()
    {
        lock (_syncLock)
        {
            if (_disposed) return;
            _disposed = true;

            _packedHead?.Dispose();
            if (_weights?.OwnedFloatBuffers != null)
            {
                foreach (IntPtr ptr in _weights.OwnedFloatBuffers)
                {
                    if (ptr != IntPtr.Zero) NativeMemory.AlignedFree((void*)ptr);
                }
                _weights.OwnedFloatBuffers.Clear();
            }
            _weights?.OwnedSidecarGguf?.Dispose();

            FreeAligned(ref _nBlockHidden);
            FreeAligned(ref _nNormX);
            FreeAligned(ref _nNormSums);
            FreeAligned(ref _nQ);
            FreeAligned(ref _nK);
            FreeAligned(ref _nV);
            FreeAligned(ref _nScores);
            FreeAligned(ref _nAttnOut);
            FreeAligned(ref _nAttnSums);
            FreeAligned(ref _nAttnProj);
            FreeAligned(ref _nGate);
            FreeAligned(ref _nUp);
            FreeAligned(ref _nFfnAct);
            FreeAligned(ref _nFfnSums);
            FreeAligned(ref _nFfnOut);
            FreeAligned(ref _nLogits);
            FreeAligned(ref _nCtxProj);
        }
    }
}
