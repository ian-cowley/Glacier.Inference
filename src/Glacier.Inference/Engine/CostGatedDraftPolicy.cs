namespace Glacier.Inference.Engine;

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using Glacier.Inference.Sampling;

/// <summary>
/// Selected speculative execution route from <see cref="CostGatedDraftPolicy"/>.
/// </summary>
public enum DraftRoute
{
    Serial = 0,
    Mtp = 1,
    PromptLookup = 2,
    DFlash = 3
}

/// <summary>
/// Routing decision emitted by <see cref="CostGatedDraftPolicy.SelectRoute(int, bool, int, int)"/>.
/// </summary>
public readonly record struct DraftDecision(DraftRoute Route, int DraftCount, double ExpectedTokensPerMs);

/// <summary>
/// Cost-model speculative draft policy that online-estimates per-route acceptance rates with Beta priors
/// and exponential decay, trading expected committed tokens <c>E(q, k) = sum_{i=0}^{k} q^i</c> against
/// draft + target batch verification latency to maximize expected tokens/ms.
/// </summary>
public sealed class CostGatedDraftPolicy
{
    private const int MaxLookupBuckets = 8;
    private const double PriorPseudocount = 2.0;

    private struct RouteStats
    {
        public double Successes;
        public double Trials;
    }

    private RouteStats _mtpStats;
    private RouteStats _dflashStats;
    private readonly RouteStats[] _lookupBuckets = new RouteStats[MaxLookupBuckets + 1];

    public double BaseVerifyMs { get; set; }
    public double PerDraftVerifyMs { get; set; }
    public double MtpDraftStepMs { get; set; }
    public double LookupOverheadMs { get; set; }
    public double DFlashDraftMs { get; set; } = 0.35;
    public double PriorMtpAcceptRate { get; set; }
    public double PriorLookupAcceptRate { get; set; }
    public double PriorDFlashAcceptRate { get; set; } = 0.68;
    public double DecayFactor { get; set; }
    public int MaxMtpDraftTokens { get; set; }
    public int MaxLookupDraftTokens { get; set; }
    public int MaxDFlashDraftTokens { get; set; } = 8;
    public bool EnableLatencyAdaptation { get; set; }

    public CostGatedDraftPolicy(
        double baseVerifyMs = 10.0,
        double perDraftVerifyMs = 1.8,
        double mtpDraftStepMs = 1.2,
        double lookupOverheadMs = 0.05,
        double priorMtpAcceptRate = 0.72,
        double priorLookupAcceptRate = 0.65,
        double decayFactor = 0.95,
        int maxMtpDraftTokens = 3,
        int maxLookupDraftTokens = 8,
        bool enableLatencyAdaptation = false)
    {
        if (baseVerifyMs <= 0.0) throw new ArgumentOutOfRangeException(nameof(baseVerifyMs));
        if (perDraftVerifyMs < 0.0) throw new ArgumentOutOfRangeException(nameof(perDraftVerifyMs));
        if (mtpDraftStepMs < 0.0) throw new ArgumentOutOfRangeException(nameof(mtpDraftStepMs));
        if (lookupOverheadMs < 0.0) throw new ArgumentOutOfRangeException(nameof(lookupOverheadMs));
        if (priorMtpAcceptRate < 0.0 || priorMtpAcceptRate > 1.0) throw new ArgumentOutOfRangeException(nameof(priorMtpAcceptRate));
        if (priorLookupAcceptRate < 0.0 || priorLookupAcceptRate > 1.0) throw new ArgumentOutOfRangeException(nameof(priorLookupAcceptRate));
        if (decayFactor <= 0.0 || decayFactor > 1.0) throw new ArgumentOutOfRangeException(nameof(decayFactor));
        if (maxMtpDraftTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxMtpDraftTokens));
        if (maxLookupDraftTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxLookupDraftTokens));

        BaseVerifyMs = baseVerifyMs;
        PerDraftVerifyMs = perDraftVerifyMs;
        MtpDraftStepMs = mtpDraftStepMs;
        LookupOverheadMs = lookupOverheadMs;
        PriorMtpAcceptRate = priorMtpAcceptRate;
        PriorLookupAcceptRate = priorLookupAcceptRate;
        DecayFactor = decayFactor;
        MaxMtpDraftTokens = maxMtpDraftTokens;
        MaxLookupDraftTokens = maxLookupDraftTokens;
        EnableLatencyAdaptation = enableLatencyAdaptation;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ComputeExpectedCommittedTokens(double acceptRate, int draftCount)
    {
        if (draftCount <= 0) return 1.0;
        double q = Math.Clamp(acceptRate, 0.0, 1.0);
        double expected = 1.0;
        double qPow = 1.0;
        for (int i = 1; i <= draftCount; i++)
        {
            qPow *= q;
            expected += qPow;
        }
        return expected;
    }

    public double GetAcceptanceEstimate(DraftRoute route, int matchedNgramLength = 0)
    {
        if (route == DraftRoute.Mtp)
        {
            return (PriorMtpAcceptRate * PriorPseudocount + _mtpStats.Successes) /
                   (PriorPseudocount + _mtpStats.Trials);
        }

        if (route == DraftRoute.PromptLookup)
        {
            int bucket = ClampBucket(matchedNgramLength);
            RouteStats s = _lookupBuckets[bucket];
            return (PriorLookupAcceptRate * PriorPseudocount + s.Successes) /
                   (PriorPseudocount + s.Trials);
        }

        if (route == DraftRoute.DFlash)
        {
            return (PriorDFlashAcceptRate * PriorPseudocount + _dflashStats.Successes) /
                   (PriorPseudocount + _dflashStats.Trials);
        }

        return 0.0;
    }

    public (double Successes, double Trials) GetRouteStats(DraftRoute route, int matchedNgramLength = 0) => route switch
    {
        DraftRoute.Mtp => (_mtpStats.Successes, _mtpStats.Trials),
        DraftRoute.PromptLookup => (_lookupBuckets[ClampBucket(matchedNgramLength)].Successes, _lookupBuckets[ClampBucket(matchedNgramLength)].Trials),
        DraftRoute.DFlash => (_dflashStats.Successes, _dflashStats.Trials),
        _ => (0.0, 0.0)
    };

    /// <summary>
    /// Chooses the highest-throughput route and optimal draft horizon <c>k</c>.
    /// Falls back to <see cref="DraftRoute.Serial"/> (<c>DraftCount = 0</c>) if neither speculative path
    /// strictly beats unassisted single-token decoding <c>1.0 / BaseVerifyMs</c>.
    /// </summary>
    public DraftDecision SelectRoute(
        int maxDraftTokens,
        bool hasMtp,
        int lookupCandidateCount,
        int matchedNgramLength)
        => SelectRoute(maxDraftTokens, hasMtp, lookupCandidateCount, matchedNgramLength, hasDFlash: false, dflashCandidateCount: 0);

    public DraftDecision SelectRoute(
        int maxDraftTokens,
        bool hasMtp,
        int lookupCandidateCount,
        int matchedNgramLength,
        bool hasDFlash,
        int dflashCandidateCount = 0)
    {
        double serialRate = 1.0 / Math.Max(0.001, BaseVerifyMs);
        DraftRoute bestRoute = DraftRoute.Serial;
        int bestK = 0;
        double bestRate = serialRate;

        if (maxDraftTokens <= 0)
        {
            return new DraftDecision(DraftRoute.Serial, 0, serialRate);
        }

        if (lookupCandidateCount > 0 && MaxLookupDraftTokens > 0)
        {
            double qLkp = GetAcceptanceEstimate(DraftRoute.PromptLookup, matchedNgramLength);
            int maxKLkp = Math.Min(lookupCandidateCount, Math.Min(maxDraftTokens, MaxLookupDraftTokens));
            for (int k = 1; k <= maxKLkp; k++)
            {
                double cost = BaseVerifyMs + LookupOverheadMs + (k * PerDraftVerifyMs);
                double rate = ComputeExpectedCommittedTokens(qLkp, k) / Math.Max(0.001, cost);
                if (rate > bestRate)
                {
                    bestRate = rate;
                    bestRoute = DraftRoute.PromptLookup;
                    bestK = k;
                }
            }
        }

        if (hasMtp && MaxMtpDraftTokens > 0)
        {
            double qMtp = GetAcceptanceEstimate(DraftRoute.Mtp);
            int maxKMtp = Math.Min(maxDraftTokens, MaxMtpDraftTokens);
            for (int k = 1; k <= maxKMtp; k++)
            {
                double cost = BaseVerifyMs + (k * (MtpDraftStepMs + PerDraftVerifyMs));
                double rate = ComputeExpectedCommittedTokens(qMtp, k) / Math.Max(0.001, cost);
                if (rate > bestRate)
                {
                    bestRate = rate;
                    bestRoute = DraftRoute.Mtp;
                    bestK = k;
                }
            }
        }

        if (hasDFlash && MaxDFlashDraftTokens > 0)
        {
            double qDFlash = GetAcceptanceEstimate(DraftRoute.DFlash);
            int maxKDFlash = Math.Min(maxDraftTokens, MaxDFlashDraftTokens);
            if (dflashCandidateCount > 0)
            {
                maxKDFlash = Math.Min(maxKDFlash, dflashCandidateCount);
            }
            for (int k = 1; k <= maxKDFlash; k++)
            {
                double cost = BaseVerifyMs + DFlashDraftMs + (k * PerDraftVerifyMs);
                double rate = ComputeExpectedCommittedTokens(qDFlash, k) / Math.Max(0.001, cost);
                if (rate > bestRate)
                {
                    bestRate = rate;
                    bestRoute = DraftRoute.DFlash;
                    bestK = k;
                }
            }
        }

        return new DraftDecision(bestRoute, bestK, bestRate);
    }

    /// <summary>
    /// Updates decayed trial counts using verified outcomes: in a round proposing <paramref name="draftedCount"/>
    /// tokens with <paramref name="acceptedCount"/> matches, <c>tested = min(draftedCount, acceptedCount + 1)</c>
    /// draft positions were tested.
    /// </summary>
    public void RecordOutcome(
        DraftRoute route,
        int matchedNgramLength,
        int draftedCount,
        int acceptedCount,
        double roundElapsedMs)
    {
        if (route == DraftRoute.Serial || draftedCount <= 0)
        {
            return;
        }

        int clampedAccepted = Math.Clamp(acceptedCount, 0, draftedCount);
        int tested = Math.Min(draftedCount, clampedAccepted + 1);

        if (route == DraftRoute.Mtp)
        {
            _mtpStats.Successes = (_mtpStats.Successes * DecayFactor) + clampedAccepted;
            _mtpStats.Trials = (_mtpStats.Trials * DecayFactor) + tested;
        }
        else if (route == DraftRoute.PromptLookup)
        {
            int bucket = ClampBucket(matchedNgramLength);
            ref RouteStats s = ref _lookupBuckets[bucket];
            s.Successes = (s.Successes * DecayFactor) + clampedAccepted;
            s.Trials = (s.Trials * DecayFactor) + tested;
        }
        else if (route == DraftRoute.DFlash)
        {
            _dflashStats.Successes = (_dflashStats.Successes * DecayFactor) + clampedAccepted;
            _dflashStats.Trials = (_dflashStats.Trials * DecayFactor) + tested;
        }

        if (EnableLatencyAdaptation && roundElapsedMs > 0.0)
        {
            double extraDrafts = draftedCount;
            double perDraftObserved = Math.Clamp((roundElapsedMs - BaseVerifyMs) / extraDrafts, 0.01, BaseVerifyMs * 2.0);
            PerDraftVerifyMs = (0.85 * PerDraftVerifyMs) + (0.15 * perDraftObserved);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ClampBucket(int matchedNgramLength) =>
        matchedNgramLength <= 1 ? 1 : (matchedNgramLength >= MaxLookupBuckets ? MaxLookupBuckets : matchedNgramLength);
}

/// <summary>
/// Composite speculative drafter coupling zero-cost Prompt Lookup with an optional native <see cref="MtpDraftProvider"/>
/// under online throughput control from a <see cref="CostGatedDraftPolicy"/>.
/// </summary>
public sealed class AdaptiveCompositeDraftProvider : IDraftProvider
{
    private readonly PromptLookupDraftProvider? _lookupProvider;
    private readonly MtpDraftProvider? _mtpProvider;
    private readonly DFlashDraftProvider? _dflashProvider;
    private readonly bool _ownsProviders;
    private int[] _probeScratch = new int[32];
    private int _lastMatchedNgramLength;
    private DraftDecision _lastDecision;
    private bool _disposed;

    public PromptLookupDraftProvider? LookupProvider => _lookupProvider;
    public MtpDraftProvider? MtpProvider => _mtpProvider;
    public DFlashDraftProvider? DFlashProvider => _dflashProvider;
    public CostGatedDraftPolicy Policy { get; }
    public DraftDecision LastDecision => _lastDecision;

    public int PromptLookupRounds { get; private set; }
    public int MtpRounds { get; private set; }
    public int DFlashRounds { get; private set; }
    public int SerialBypassRounds { get; private set; }

    public AdaptiveCompositeDraftProvider(
        PromptLookupDraftProvider? lookupProvider,
        MtpDraftProvider? mtpProvider,
        CostGatedDraftPolicy? policy = null,
        bool ownsProviders = false,
        DFlashDraftProvider? dflashProvider = null)
    {
        if (lookupProvider == null && mtpProvider == null && dflashProvider == null)
            throw new ArgumentException("At least one of LookupProvider, MtpProvider, or DFlashProvider must be provided.");

        _lookupProvider = lookupProvider;
        _mtpProvider = mtpProvider;
        _dflashProvider = dflashProvider;
        Policy = policy ?? new CostGatedDraftPolicy();
        _ownsProviders = ownsProviders;
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ((IDraftProvider?)_lookupProvider)?.Reset();
        _mtpProvider?.Reset();
        _dflashProvider?.Reset();
        _lastMatchedNgramLength = 0;
        _lastDecision = default;
    }

    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxDraftTokens <= 0 || draftTokens.IsEmpty)
            return 0;

        int maxOut = Math.Min(maxDraftTokens, draftTokens.Length);
        EnsureProbeCapacity(maxOut);

        int lookupCount = 0;
        int matchLen = 0;
        if (_lookupProvider != null)
        {
            lookupCount = _lookupProvider.DraftWithMatchLength(tokens, maxOut, _probeScratch.AsSpan(0, maxOut), out matchLen);
        }

        _lastMatchedNgramLength = matchLen;
        _lastDecision = Policy.SelectRoute(maxOut, _mtpProvider != null, lookupCount, matchLen, hasDFlash: _dflashProvider != null, dflashCandidateCount: _dflashProvider?.EffectiveBlockSize ?? 0);

        switch (_lastDecision.Route)
        {
            case DraftRoute.PromptLookup:
            {
                int chosen = Math.Min(_lastDecision.DraftCount, Math.Min(lookupCount, draftTokens.Length));
                if (chosen <= 0)
                {
                    SerialBypassRounds++;
                    return 0;
                }
                _probeScratch.AsSpan(0, chosen).CopyTo(draftTokens);
                PromptLookupRounds++;
                return chosen;
            }
            case DraftRoute.Mtp when _mtpProvider != null:
            {
                int k = Math.Min(_lastDecision.DraftCount, draftTokens.Length);
                int drafted = k > 0 ? _mtpProvider.Draft(tokens, k, draftTokens) : 0;
                if (drafted <= 0)
                {
                    SerialBypassRounds++;
                    return 0;
                }
                MtpRounds++;
                return drafted;
            }
            case DraftRoute.DFlash when _dflashProvider != null:
            {
                int k = Math.Min(_lastDecision.DraftCount, draftTokens.Length);
                int drafted = k > 0 ? _dflashProvider.Draft(tokens, k, draftTokens) : 0;
                if (drafted > 0)
                {
                    DFlashRounds++;
                    return drafted;
                }
                if (_mtpProvider != null && k > 0)
                {
                    drafted = _mtpProvider.Draft(tokens, Math.Min(k, Policy.MaxMtpDraftTokens), draftTokens);
                    if (drafted > 0)
                    {
                        _lastDecision = _lastDecision with { Route = DraftRoute.Mtp, DraftCount = drafted };
                        MtpRounds++;
                        return drafted;
                    }
                }
                if (lookupCount > 0)
                {
                    int chosen = Math.Min(lookupCount, Math.Min(maxOut, Policy.MaxLookupDraftTokens));
                    if (chosen > 0)
                    {
                        _probeScratch.AsSpan(0, chosen).CopyTo(draftTokens);
                        _lastDecision = _lastDecision with { Route = DraftRoute.PromptLookup, DraftCount = chosen };
                        PromptLookupRounds++;
                        return chosen;
                    }
                }
                SerialBypassRounds++;
                return 0;
            }
            default:
                SerialBypassRounds++;
                return 0;
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
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (maxDraftTokens <= 0 || draftTokens.IsEmpty)
            return 0;

        int maxOut = Math.Min(maxDraftTokens, draftTokens.Length);
        EnsureProbeCapacity(maxOut);

        int lookupCount = 0;
        int matchLen = 0;
        if (_lookupProvider != null)
        {
            lookupCount = _lookupProvider.DraftWithMatchLength(tokens, maxOut, _probeScratch.AsSpan(0, maxOut), out matchLen);
        }

        _lastMatchedNgramLength = matchLen;
        _lastDecision = Policy.SelectRoute(maxOut, _mtpProvider != null, lookupCount, matchLen, hasDFlash: _dflashProvider != null, dflashCandidateCount: _dflashProvider?.EffectiveBlockSize ?? 0);

        switch (_lastDecision.Route)
        {
            case DraftRoute.PromptLookup:
            {
                int chosen = Math.Min(_lastDecision.DraftCount, Math.Min(lookupCount, draftTokens.Length));
                if (chosen <= 0)
                {
                    SerialBypassRounds++;
                    return 0;
                }
                _probeScratch.AsSpan(0, chosen).CopyTo(draftTokens);
                draftProbs.Slice(0, chosen * targetVocabSize).Clear();
                for (int i = 0; i < chosen; i++)
                {
                    int tok = draftTokens[i];
                    if ((uint)tok < (uint)targetVocabSize)
                    {
                        draftProbs[i * targetVocabSize + tok] = 1.0f;
                    }
                }
                PromptLookupRounds++;
                return chosen;
            }
            case DraftRoute.Mtp when _mtpProvider != null:
            {
                int k = Math.Min(_lastDecision.DraftCount, draftTokens.Length);
                int drafted = k > 0
                    ? _mtpProvider.DraftWithProbabilities(tokens, recentTokens, k, sampling, sampler, targetVocabSize, draftTokens, draftProbs)
                    : 0;
                if (drafted <= 0)
                {
                    SerialBypassRounds++;
                    return 0;
                }
                MtpRounds++;
                return drafted;
            }
            case DraftRoute.DFlash when _dflashProvider != null:
            {
                int k = Math.Min(_lastDecision.DraftCount, draftTokens.Length);
                int drafted = k > 0
                    ? _dflashProvider.DraftWithProbabilities(tokens, recentTokens, k, sampling, sampler, targetVocabSize, draftTokens, draftProbs)
                    : 0;
                if (drafted > 0)
                {
                    DFlashRounds++;
                    return drafted;
                }
                if (_mtpProvider != null && k > 0)
                {
                    drafted = _mtpProvider.DraftWithProbabilities(tokens, recentTokens, Math.Min(k, Policy.MaxMtpDraftTokens), sampling, sampler, targetVocabSize, draftTokens, draftProbs);
                    if (drafted > 0)
                    {
                        _lastDecision = _lastDecision with { Route = DraftRoute.Mtp, DraftCount = drafted };
                        MtpRounds++;
                        return drafted;
                    }
                }
                if (lookupCount > 0)
                {
                    int chosen = Math.Min(lookupCount, Math.Min(maxOut, Policy.MaxLookupDraftTokens));
                    if (chosen > 0)
                    {
                        _probeScratch.AsSpan(0, chosen).CopyTo(draftTokens);
                        draftProbs.Slice(0, chosen * targetVocabSize).Clear();
                        for (int i = 0; i < chosen; i++)
                        {
                            int tok = draftTokens[i];
                            if ((uint)tok < (uint)targetVocabSize)
                            {
                                draftProbs[i * targetVocabSize + tok] = 1.0f;
                            }
                        }
                        _lastDecision = _lastDecision with { Route = DraftRoute.PromptLookup, DraftCount = chosen };
                        PromptLookupRounds++;
                        return chosen;
                    }
                }
                SerialBypassRounds++;
                return 0;
            }
            default:
                SerialBypassRounds++;
                return 0;
        }
    }

    public void OnVerificationResult(int draftedCount, int acceptedCount, double roundElapsedMs)
    {
        Policy.RecordOutcome(_lastDecision.Route, _lastMatchedNgramLength, draftedCount, acceptedCount, roundElapsedMs);
        if (_lastDecision.Route == DraftRoute.DFlash)
        {
            _dflashProvider?.OnVerificationResult(draftedCount, acceptedCount, roundElapsedMs);
        }
    }

    private void EnsureProbeCapacity(int needed)
    {
        if (_probeScratch.Length < needed)
        {
            _probeScratch = new int[needed];
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_ownsProviders)
        {
            _lookupProvider?.Dispose();
            _mtpProvider?.Dispose();
            _dflashProvider?.Dispose();
        }
    }
}
