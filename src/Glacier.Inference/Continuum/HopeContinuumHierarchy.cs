namespace Glacier.Inference.Continuum;

using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Configuration descriptor for a single tier in a <see cref="HopeContinuumHierarchy"/>.
/// </summary>
public readonly record struct HopeTierConfig(
    ContinuumFrequency Frequency,
    int ClockStride,
    float RetentionRateLambda,
    float MomentumDecayTheta,
    float LearningRateEta,
    float RoutingWeight);

/// <summary>
/// HOPE (Hierarchical Optimizing Processing Ensemble) multi-frequency continuum memory hierarchy.
/// Coordinates multiple <see cref="TitansContinuumMemoryBlock"/> modules operating at distinct clock frequencies
/// (from fast per-token updates to surprise-gated slow continuum blocks) with zero-allocation hot paths.
/// </summary>
public sealed unsafe class HopeContinuumHierarchy : IDisposable
{
    private const nuint CacheLineAlignment = 64;
    private const int SnapshotHeaderBytes = 16; // 8-byte global clock + 4-byte tier count + 4-byte dimension

    private readonly int _dimension;
    private readonly TitansContinuumMemoryBlock[] _tiers;
    private readonly float[] _routingWeights;
    private readonly long _totalSnapshotBytes;
    private float _surpriseThreshold;
    private long _globalTokenClock;
    private float* _tierScratchPtr;
    private bool _disposed;

    public int Dimension => _dimension;
    public int TierCount => _tiers.Length;
    public long GlobalTokenClock => _globalTokenClock;

    public float SurpriseThreshold
    {
        get => _surpriseThreshold;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _surpriseThreshold = value;
        }
    }

    public long StateSnapshotBytes => _totalSnapshotBytes;

    public HopeContinuumHierarchy(
        int dimension,
        ReadOnlySpan<HopeTierConfig> tierConfigs,
        float surpriseThreshold = 0.65f)
    {
        if (dimension <= 0 || dimension > ushort.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Dimension must be between 1 and 65535.");
        }
        if (tierConfigs.IsEmpty)
        {
            throw new ArgumentException("At least one continuum tier configuration must be provided.", nameof(tierConfigs));
        }
        ArgumentOutOfRangeException.ThrowIfNegative(surpriseThreshold);

        _dimension = dimension;
        _surpriseThreshold = surpriseThreshold;
        _tiers = new TitansContinuumMemoryBlock[tierConfigs.Length];
        _routingWeights = new float[tierConfigs.Length];

        long snapshotAccum = SnapshotHeaderBytes;
        for (int i = 0; i < tierConfigs.Length; i++)
        {
            HopeTierConfig cfg = tierConfigs[i];
            var block = new TitansContinuumMemoryBlock(
                dimension,
                cfg.Frequency,
                cfg.ClockStride,
                cfg.RetentionRateLambda,
                cfg.MomentumDecayTheta,
                cfg.LearningRateEta);
            _tiers[i] = block;
            _routingWeights[i] = cfg.RoutingWeight;
            snapshotAccum += block.StateSnapshotBytes;
        }

        _totalSnapshotBytes = snapshotAccum;
        nuint scratchBytes = (nuint)(((long)dimension * sizeof(float) + 63) & ~63L);
        _tierScratchPtr = (float*)NativeMemory.AlignedAlloc(scratchBytes, CacheLineAlignment);
        NativeMemory.Clear(_tierScratchPtr, scratchBytes);
    }

    /// <summary>
    /// Creates a standard 4-tier HOPE continuum hierarchy spanning <c>omega_0</c> (1 token),
    /// <c>omega_1</c> (64 tokens), <c>omega_2</c> (4,096 tokens), and <c>omega_3</c> (65,536 tokens).
    /// </summary>
    public static HopeContinuumHierarchy CreateDefault4Tier(int dimension, float surpriseThreshold = 0.65f)
    {
        ReadOnlySpan<HopeTierConfig> defaults =
        [
            new HopeTierConfig(ContinuumFrequency.PerToken,     ClockStride: 1,     RetentionRateLambda: 0.995f,   MomentumDecayTheta: 0.40f, LearningRateEta: 0.50f, RoutingWeight: 0.10f),
            new HopeTierConfig(ContinuumFrequency.ShortWindow,  ClockStride: 64,    RetentionRateLambda: 0.999f,   MomentumDecayTheta: 0.60f, LearningRateEta: 0.65f, RoutingWeight: 0.15f),
            new HopeTierConfig(ContinuumFrequency.Session,      ClockStride: 4096,  RetentionRateLambda: 0.9995f,  MomentumDecayTheta: 0.70f, LearningRateEta: 0.75f, RoutingWeight: 0.25f),
            new HopeTierConfig(ContinuumFrequency.Consolidated, ClockStride: 65536, RetentionRateLambda: 0.99999f, MomentumDecayTheta: 0.20f, LearningRateEta: 0.95f, RoutingWeight: 0.50f)
        ];
        return new HopeContinuumHierarchy(dimension, defaults, surpriseThreshold);
    }

    /// <summary>
    /// Gets the <see cref="TitansContinuumMemoryBlock"/> at the specified tier index.
    /// </summary>
    public TitansContinuumMemoryBlock GetTier(int tierIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)tierIndex >= (uint)_tiers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(tierIndex), tierIndex, $"Tier index must be between 0 and {_tiers.Length - 1}.");
        }
        return _tiers[tierIndex];
    }

    /// <summary>
    /// Evaluates the multi-frequency continuum query <c>y = sum_l pi_l * (M^{(l)} * query)</c> using configured routing weights.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<float> query, Span<float> output)
    {
        Forward(query, _routingWeights, output);
    }

    /// <summary>
    /// Evaluates the multi-frequency continuum query <c>y = sum_l weights[l] * (M^{(l)} * query)</c> with zero heap allocations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<float> query, ReadOnlySpan<float> tierWeights, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (query.Length < _dimension)
        {
            throw new ArgumentException($"Query length ({query.Length}) must be at least Dimension ({_dimension}).", nameof(query));
        }
        if (tierWeights.Length < _tiers.Length)
        {
            throw new ArgumentException($"Tier weights length ({tierWeights.Length}) must be at least TierCount ({_tiers.Length}).", nameof(tierWeights));
        }
        if (output.Length < _dimension)
        {
            throw new ArgumentException($"Output length ({output.Length}) must be at least Dimension ({_dimension}).", nameof(output));
        }

        output.Slice(0, _dimension).Clear();
        Span<float> tierScratch = new Span<float>(_tierScratchPtr, _dimension);

        fixed (float* pOut = output)
        {
            for (int l = 0; l < _tiers.Length; l++)
            {
                float w = tierWeights[l];
                if (MathF.Abs(w) < 1e-12f) continue;

                _tiers[l].Forward(query, tierScratch);
                AccumulateScaledSimd(pOut, _tierScratchPtr, w, _dimension);
            }
        }
    }

    /// <summary>
    /// Advances the continuum clock by 1 token, updating the fast per-token tier and conditionally updating
    /// slower continuum tiers when either their clock stride triggers or the surprise metric exceeds <see cref="SurpriseThreshold"/>.
    /// Returns the measured surprise metric from the primary fast tier.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public float Step(ReadOnlySpan<float> key, ReadOnlySpan<float> value, float? explicitSurprise = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (key.Length < _dimension)
        {
            throw new ArgumentException($"Key length ({key.Length}) must be at least Dimension ({_dimension}).", nameof(key));
        }
        if (value.Length < _dimension)
        {
            throw new ArgumentException($"Value length ({value.Length}) must be at least Dimension ({_dimension}).", nameof(value));
        }

        _globalTokenClock++;

        // Always update Tier 0 (fastest clock)
        float measuredSurprise = _tiers[0].UpdateAssociativeState(key, value);
        float effectiveSurprise = explicitSurprise ?? measuredSurprise;
        bool highSalienceEvent = effectiveSurprise >= _surpriseThreshold;

        for (int l = 1; l < _tiers.Length; l++)
        {
            TitansContinuumMemoryBlock tier = _tiers[l];
            bool clockTick = (_globalTokenClock % tier.ClockStride) == 0;

            if (clockTick || highSalienceEvent)
            {
                // For the slowest consolidated tier, modulate learning rate eta by salience so background clock ticks
                // cause minimal perturbation while high-surprise needles are strongly encoded.
                if (l == _tiers.Length - 1 && !highSalienceEvent)
                {
                    float backgroundEta = tier.LearningRateEta * 0.02f;
                    tier.UpdateAssociativeState(key, value, etaOverride: backgroundEta);
                }
                else
                {
                    tier.UpdateAssociativeState(key, value);
                }
            }
        }

        return effectiveSurprise;
    }

    /// <summary>
    /// Synaptic Pruning: Resets volatile fast memory tiers up to <paramref name="maxTierIndexToPrune"/> (inclusive)
    /// while leaving slower consolidated tiers intact.
    /// </summary>
    public void PruneVolatileTiers(int maxTierIndexToPrune = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)maxTierIndexToPrune >= (uint)_tiers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTierIndexToPrune), maxTierIndexToPrune, $"Tier index must be between 0 and {_tiers.Length - 1}.");
        }

        for (int l = 0; l <= maxTierIndexToPrune; l++)
        {
            _tiers[l].ResetVolatileState();
        }
    }

    /// <summary>
    /// Resets all tiers and the global token clock to zero.
    /// </summary>
    public void ResetAll()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int l = 0; l < _tiers.Length; l++)
        {
            _tiers[l].ResetVolatileState();
        }
        _globalTokenClock = 0;
    }

    /// <summary>
    /// Serializes the entire multi-tier continuum state into <paramref name="destination"/> for checkpointing or speculative rollback.
    /// </summary>
    public void SaveState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < _totalSnapshotBytes)
        {
            throw new ArgumentException($"Destination length ({destination.Length}) is smaller than StateSnapshotBytes ({_totalSnapshotBytes}).", nameof(destination));
        }

        BinaryPrimitives.WriteInt64LittleEndian(destination.Slice(0, 8), _globalTokenClock);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(8, 4), _tiers.Length);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(12, 4), _dimension);

        int offset = SnapshotHeaderBytes;
        for (int l = 0; l < _tiers.Length; l++)
        {
            int tierBytes = (int)_tiers[l].StateSnapshotBytes;
            _tiers[l].SaveState(destination.Slice(offset, tierBytes));
            offset += tierBytes;
        }
    }

    /// <summary>
    /// Restores the entire multi-tier continuum state from <paramref name="source"/>.
    /// </summary>
    public void RestoreState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length < _totalSnapshotBytes)
        {
            throw new ArgumentException($"Source length ({source.Length}) is smaller than StateSnapshotBytes ({_totalSnapshotBytes}).", nameof(source));
        }

        long savedClock = BinaryPrimitives.ReadInt64LittleEndian(source.Slice(0, 8));
        int savedTierCount = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(8, 4));
        int savedDim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));

        if (savedTierCount != _tiers.Length || savedDim != _dimension)
        {
            throw new InvalidOperationException($"Snapshot topology (Tiers={savedTierCount}, Dim={savedDim}) does not match hierarchy (Tiers={_tiers.Length}, Dim={_dimension}).");
        }

        int offset = SnapshotHeaderBytes;
        for (int l = 0; l < _tiers.Length; l++)
        {
            int tierBytes = (int)_tiers[l].StateSnapshotBytes;
            _tiers[l].RestoreState(source.Slice(offset, tierBytes));
            offset += tierBytes;
        }

        _globalTokenClock = savedClock;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateScaledSimd(float* dst, float* src, float scale, int dim)
    {
        int j = 0;
        if (Avx512F.IsSupported && dim >= 16)
        {
            Vector512<float> vScale = Vector512.Create(scale);
            for (; j <= dim - 16; j += 16)
            {
                Vector512<float> vd = Avx512F.LoadVector512(dst + j);
                Vector512<float> vs = Avx512F.LoadVector512(src + j);
                Avx512F.Store(dst + j, Avx512F.FusedMultiplyAdd(vs, vScale, vd));
            }
        }

        if (Avx2.IsSupported && j <= dim - 8)
        {
            Vector256<float> vScale = Vector256.Create(scale);
            for (; j <= dim - 8; j += 8)
            {
                Vector256<float> vd = Avx.LoadVector256(dst + j);
                Vector256<float> vs = Avx.LoadVector256(src + j);
                Vector256<float> res = Fma.IsSupported ? Fma.MultiplyAdd(vs, vScale, vd) : Avx.Add(vd, Avx.Multiply(vs, vScale));
                Avx.Store(dst + j, res);
            }
        }

        for (; j < dim; j++)
        {
            dst[j] += scale * src[j];
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            for (int l = 0; l < _tiers.Length; l++)
            {
                _tiers[l]?.Dispose();
            }
            if (_tierScratchPtr != null)
            {
                NativeMemory.AlignedFree(_tierScratchPtr);
                _tierScratchPtr = null;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~HopeContinuumHierarchy()
    {
        Dispose();
    }
}
