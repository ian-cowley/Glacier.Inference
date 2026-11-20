namespace Glacier.Inference.Engine;

using System;
using System.Threading;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;

/// <summary>
/// Execution disposition assigned to a unique routed expert within a batched MoE layer step.
/// </summary>
public enum ExpertExecutionDisposition
{
    VramHit = 0,
    PcieTransfer = 1,
    CpuExecute = 2,
    Skipped = 3
}

/// <summary>
/// Per-batch deduplicated plan record for a single routed expert in a layer.
/// </summary>
public struct PlannedUniqueExpert
{
    public int ExpertIndex;
    public int MinRankInBatch;
    public int BatchOccurrenceCount;
    public float TotalWeightInBatch;
    public double ProfileScore;
    public int VramSlotIndex;
    public ExpertExecutionDisposition Disposition;
}

/// <summary>
/// Summary telemetry produced by <see cref="TieredMoEMissSplitter.PlanLayerBatch"/> for one MoE layer batch.
/// </summary>
public readonly record struct BatchLayerMissSplitResult
{
    public int LayerIndex { get; init; }
    public int BatchSize { get; init; }
    public int TopK { get; init; }
    public int TotalRoutedSlots { get; init; }
    public int SkippedRoutedSlots { get; init; }
    public double ExpertsSkippedPercent => TotalRoutedSlots > 0 ? (100.0 * SkippedRoutedSlots) / TotalRoutedSlots : 0.0;
    public int VramHitUniqueCount { get; init; }
    public int PcieTransferUniqueCount { get; init; }
    public int CpuExecuteUniqueCount { get; init; }
    public int VramHitRoutedSlots { get; init; }
    public int PcieTransferRoutedSlots { get; init; }
    public int CpuExecuteRoutedSlots { get; init; }
    public TieringExecutionPolicy ResolvedPolicy { get; init; }
    public bool IsHostArenaLayerResident { get; init; }
}

/// <summary>
/// AOT-safe, zero-LINQ per-batch MoE expert deduplicator, VRAM cache prober,
/// speculative draft tail-skip filter, and dynamic CPU/PCIe miss splitter.
/// </summary>
public sealed class TieredMoEMissSplitter
{
    private readonly Lock _lock = new();
    private int[] _expertToUnique = Array.Empty<int>();
    private PlannedUniqueExpert[] _unique = Array.Empty<PlannedUniqueExpert>();
    private int[] _missOrder = Array.Empty<int>();
    private int[] _protectedExperts = Array.Empty<int>();
    private int _lastUniqueCount;
    private int _lastNumExperts;

    /// <summary>
    /// Deduplicated planned experts from the most recent <see cref="PlanLayerBatch"/> call (single-stream per runner).
    /// </summary>
    public ReadOnlySpan<PlannedUniqueExpert> PlannedExperts => _unique.AsSpan(0, _lastUniqueCount);

    /// <summary>
    /// Looks up the disposition and VRAM slot assigned to <paramref name="expertIdx"/> in the last planned batch.
    /// </summary>
    public bool TryGetPlannedExpert(int expertIdx, out PlannedUniqueExpert planned)
    {
        lock (_lock)
        {
            if ((uint)expertIdx < (uint)_lastNumExperts && _expertToUnique[expertIdx] >= 0)
            {
                planned = _unique[_expertToUnique[expertIdx]];
                return true;
            }
            planned = default;
            return false;
        }
    }

    /// <summary>
    /// Plans a single MoE layer batch of <paramref name="batchSize"/> tokens with <paramref name="topK"/>
    /// routed experts per token (ordered in descending router probability rank <c>0..topK-1</c>).
    /// </summary>
    public BatchLayerMissSplitResult PlanLayerBatch(
        int layerIdx,
        int batchSize,
        int topK,
        int numExperts,
        bool normTopK,
        bool isSpeculativeDraftBatch,
        int routeTailSkipMinRank,
        TieringExecutionPolicy policy,
        Span<int> batchExpertIndices,
        Span<float> batchExpertWeights,
        ExpertRoutingProfile? routingProfile,
        GpuResidentSlotManager? slotManager,
        BandwidthCoScheduler? coScheduler,
        HostRamExpertArena? hostArena,
        int maxPcieTransfers = int.MaxValue)
    {
        if (batchSize <= 0 || topK <= 0 || numExperts <= 0)
        {
            return new BatchLayerMissSplitResult
            {
                LayerIndex = layerIdx,
                BatchSize = batchSize,
                TopK = topK,
                ResolvedPolicy = policy,
                IsHostArenaLayerResident = hostArena != null && hostArena.IsLayerResident(layerIdx)
            };
        }

        int totalSlots = checked(batchSize * topK);
        if (batchExpertIndices.Length < totalSlots)
            throw new ArgumentException("batchExpertIndices buffer is smaller than batchSize * topK.", nameof(batchExpertIndices));
        if (batchExpertWeights.Length < totalSlots)
            throw new ArgumentException("batchExpertWeights buffer is smaller than batchSize * topK.", nameof(batchExpertWeights));

        lock (_lock)
        {
            // Reset prior batch lookup table BEFORE EnsureCapacity, which may reallocate _unique.
            for (int u = 0; u < _lastUniqueCount; u++)
            {
                int prevExp = _unique[u].ExpertIndex;
                if ((uint)prevExp < (uint)_expertToUnique.Length)
                {
                    _expertToUnique[prevExp] = -1;
                }
            }
            _lastUniqueCount = 0;
            _lastNumExperts = numExperts;

            EnsureCapacity(numExperts, totalSlots);

            // 1. Record activations into ExpertRoutingProfile per token
            if (routingProfile != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    int baseOff = t * topK;
                    routingProfile.RecordActivations(
                        layerIdx,
                        batchExpertIndices.Slice(baseOff, topK),
                        batchExpertWeights.Slice(baseOff, topK));
                }
            }

            // 2. Deduplicate unique experts across all tokens in the batch
            int uniqueCount = 0;
            int totalRoutedSlots = 0;

            for (int t = 0; t < batchSize; t++)
            {
                int baseOff = t * topK;
                for (int k = 0; k < topK; k++)
                {
                    int expIdx = batchExpertIndices[baseOff + k];
                    if ((uint)expIdx >= (uint)numExperts)
                        continue;

                    totalRoutedSlots++;
                    float w = batchExpertWeights[baseOff + k];
                    int pos = _expertToUnique[expIdx];

                    if (pos < 0)
                    {
                        pos = uniqueCount++;
                        _expertToUnique[expIdx] = pos;
                        _unique[pos] = new PlannedUniqueExpert
                        {
                            ExpertIndex = expIdx,
                            MinRankInBatch = k,
                            BatchOccurrenceCount = 1,
                            TotalWeightInBatch = w,
                            ProfileScore = routingProfile != null ? routingProfile.GetScore(layerIdx, expIdx) : 0.0,
                            VramSlotIndex = -1,
                            Disposition = ExpertExecutionDisposition.CpuExecute
                        };
                    }
                    else
                    {
                        ref var entry = ref _unique[pos];
                        if (k < entry.MinRankInBatch)
                        {
                            entry.MinRankInBatch = k;
                        }
                        entry.BatchOccurrenceCount++;
                        entry.TotalWeightInBatch += w;
                    }
                }
            }

            _lastUniqueCount = uniqueCount;

            // 3. Probe GpuResidentSlotManager once per unique expert & apply opt-in RouteTailSkipMinRank
            int protectedCount = 0;
            int vramHitUnique = 0;
            int vramHitSlots = 0;
            int activeMissCount = 0;
            int skippedSlots = 0;
            bool tailSkipEnabled = isSpeculativeDraftBatch && routeTailSkipMinRank > 0 && routeTailSkipMinRank < topK;

            for (int u = 0; u < uniqueCount; u++)
            {
                ref var ue = ref _unique[u];
                int slotIdx = -1;
                bool hit = slotManager != null &&
                           slotManager.ActiveSlotCount > 0 &&
                           slotManager.TryGetSlot(layerIdx, ue.ExpertIndex, out slotIdx, out _);

                if (hit)
                {
                    ue.Disposition = ExpertExecutionDisposition.VramHit;
                    ue.VramSlotIndex = slotIdx;
                    _protectedExperts[protectedCount++] = ue.ExpertIndex;
                    vramHitUnique++;
                    vramHitSlots += ue.BatchOccurrenceCount;
                }
                else if (tailSkipEnabled && ue.MinRankInBatch >= routeTailSkipMinRank)
                {
                    // Only skip VRAM-missed experts that appear in the bottom tail for EVERY token in the batch
                    ue.Disposition = ExpertExecutionDisposition.Skipped;
                    ue.VramSlotIndex = -1;
                    skippedSlots += ue.BatchOccurrenceCount;
                }
                else
                {
                    _missOrder[activeMissCount++] = u;
                }
            }

            // 4. Mask skipped slots and renormalize kept weights per affected token
            if (skippedSlots > 0)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    int baseOff = t * topK;
                    float keptSum = 0f;
                    bool skippedInToken = false;

                    for (int k = 0; k < topK; k++)
                    {
                        int expIdx = batchExpertIndices[baseOff + k];
                        if ((uint)expIdx >= (uint)numExperts)
                            continue;

                        int uPos = _expertToUnique[expIdx];
                        if (_unique[uPos].Disposition == ExpertExecutionDisposition.Skipped)
                        {
                            batchExpertIndices[baseOff + k] = -1;
                            batchExpertWeights[baseOff + k] = 0f;
                            skippedInToken = true;
                        }
                        else
                        {
                            keptSum += batchExpertWeights[baseOff + k];
                        }
                    }

                    if (skippedInToken && normTopK && keptSum > 0f)
                    {
                        float inv = 1.0f / keptSum;
                        for (int k = 0; k < topK; k++)
                        {
                            if (batchExpertIndices[baseOff + k] >= 0)
                            {
                                batchExpertWeights[baseOff + k] *= inv;
                            }
                        }
                    }
                }
            }

            // 5. Sort active misses descending by ProfileScore, BatchOccurrenceCount, TotalWeightInBatch (AOT-safe insertion sort)
            for (int i = 1; i < activeMissCount; i++)
            {
                int keyU = _missOrder[i];
                ref readonly var key = ref _unique[keyU];
                int j = i - 1;
                while (j >= 0)
                {
                    ref readonly var prev = ref _unique[_missOrder[j]];
                    if (!IsMissHigherPriority(in key, in prev))
                        break;

                    _missOrder[j + 1] = _missOrder[j];
                    j--;
                }
                _missOrder[j + 1] = keyU;
            }

            // 6. Compute q* split across HybridGpuHostRam, GpuSharedMemory, and Auto
            bool isHostResident = hostArena != null && hostArena.IsLayerResident(layerIdx);
            int availSlots = slotManager != null
                ? slotManager.GetAvailableTransferSlots(layerIdx, _protectedExperts.AsSpan(0, protectedCount))
                : 0;
            int pcieCap = Math.Max(0, maxPcieTransfers);

            int qStar = ComputePolicyQStar(
                policy,
                activeMissCount,
                vramHitUnique,
                availSlots,
                pcieCap,
                slotManager != null,
                isHostResident,
                hostArena != null,
                coScheduler);

            // 7. Assign dispositions & promote qStar misses into VRAM slots (protecting intra-batch hits & promotions)
            int pcieUnique = 0;
            int pcieSlots = 0;
            int cpuUnique = 0;
            int cpuSlots = 0;

            for (int i = 0; i < activeMissCount; i++)
            {
                ref var ue = ref _unique[_missOrder[i]];
                if (i < qStar)
                {
                    ue.Disposition = ExpertExecutionDisposition.PcieTransfer;
                    if (slotManager != null && slotManager.ActiveSlotCount > 0)
                    {
                        slotManager.AllocateOrEvictSlot(
                            layerIdx,
                            ue.ExpertIndex,
                            _protectedExperts.AsSpan(0, protectedCount),
                            out int dstSlot,
                            out _,
                            out _,
                            out _);
                        ue.VramSlotIndex = dstSlot;
                        _protectedExperts[protectedCount++] = ue.ExpertIndex;
                    }
                    else
                    {
                        ue.VramSlotIndex = -1;
                    }
                    pcieUnique++;
                    pcieSlots += ue.BatchOccurrenceCount;
                }
                else
                {
                    ue.Disposition = ExpertExecutionDisposition.CpuExecute;
                    ue.VramSlotIndex = -1;
                    cpuUnique++;
                    cpuSlots += ue.BatchOccurrenceCount;
                }
            }

            return new BatchLayerMissSplitResult
            {
                LayerIndex = layerIdx,
                BatchSize = batchSize,
                TopK = topK,
                TotalRoutedSlots = totalRoutedSlots,
                SkippedRoutedSlots = skippedSlots,
                VramHitUniqueCount = vramHitUnique,
                PcieTransferUniqueCount = pcieUnique,
                CpuExecuteUniqueCount = cpuUnique,
                VramHitRoutedSlots = vramHitSlots,
                PcieTransferRoutedSlots = pcieSlots,
                CpuExecuteRoutedSlots = cpuSlots,
                ResolvedPolicy = policy,
                IsHostArenaLayerResident = isHostResident
            };
        }
    }

    private static bool IsMissHigherPriority(in PlannedUniqueExpert candidate, in PlannedUniqueExpert existing)
    {
        double cScore = QuantizeScore(candidate.ProfileScore);
        double eScore = QuantizeScore(existing.ProfileScore);
        if (cScore != eScore)
            return cScore > eScore;

        if (candidate.BatchOccurrenceCount != existing.BatchOccurrenceCount)
            return candidate.BatchOccurrenceCount > existing.BatchOccurrenceCount;

        float cWeight = QuantizeWeight(candidate.TotalWeightInBatch);
        float eWeight = QuantizeWeight(existing.TotalWeightInBatch);
        if (cWeight != eWeight)
            return cWeight > eWeight;

        return candidate.ExpertIndex < existing.ExpertIndex;
    }

    private static double QuantizeScore(double score) =>
        double.IsFinite(score) && Math.Abs(score) <= 1e15 ? Math.Round(score, 9) : score;

    private static float QuantizeWeight(float weight) =>
        float.IsFinite(weight) && MathF.Abs(weight) <= 1e7f ? MathF.Round(weight, 5) : weight;

    private static int ComputePolicyQStar(
        TieringExecutionPolicy policy,
        int activeMissCount,
        int vramHitCount,
        int availSlots,
        int maxPcieTransfers,
        bool hasSlotManager,
        bool isHostArenaResident,
        bool hasHostArena,
        BandwidthCoScheduler? coScheduler)
    {
        if (activeMissCount <= 0 || maxPcieTransfers <= 0)
            return 0;

        int slotBound = hasSlotManager ? Math.Min(activeMissCount, availSlots) : activeMissCount;
        int upperBound = Math.Min(slotBound, maxPcieTransfers);
        if (upperBound <= 0)
            return 0;

        switch (policy)
        {
            case TieringExecutionPolicy.HybridGpuHostRam:
                // Routed misses evaluate in Host DDR5 via HostRamExpertArena; VRAM hits stay on GPU.
                return 0;

            case TieringExecutionPolicy.GpuSharedMemory:
                // Stream misses over PCIe / shared memory up to slot & staging capacity; excess overflows to CPU.
                return upperBound;

            case TieringExecutionPolicy.Auto:
            case TieringExecutionPolicy.DistributedCluster:
            default:
                if (coScheduler == null)
                    return 0;

                int qOptimal;
                if (!hasHostArena || isHostArenaResident)
                {
                    qOptimal = coScheduler.CalculateOptimalGpuExpertCount(activeMissCount, vramHitCount);
                }
                else
                {
                    // Layer spilled out of HostRamExpertArena budget (cold mmap risk): discount CPU bandwidth
                    // so the equilibrium shifts misses toward PCIe / GPU shared memory.
                    qOptimal = CalculateColdMmapOptimalGpuCount(coScheduler, activeMissCount, vramHitCount);
                }

                return Math.Clamp(qOptimal, 0, upperBound);
        }
    }

    private static int CalculateColdMmapOptimalGpuCount(
        BandwidthCoScheduler coScheduler,
        int missingCount,
        int residentHitCount)
    {
        double bp = coScheduler.PcieBandwidthGBps;
        double bh = Math.Min(coScheduler.CpuBandwidthGBps * 0.25, 4.0);
        double bv = coScheduler.VramBandwidthGBps;

        if (!double.IsFinite(bp) || bp <= 0.0) bp = 12.5;
        if (!double.IsFinite(bh) || bh <= 0.0) bh = 4.0;
        if (!double.IsFinite(bv) || bv <= 0.0) bv = 272.0;

        double alpha = (1.0 / bp) + (1.0 / bv);
        double invBh = 1.0 / bh;
        double gamma = 1.0 / bv;

        double target = (missingCount * invBh - residentHitCount * gamma) / (alpha + invBh);
        if (target <= 0.0)
            return 0;

        int qFloor = Math.Clamp((int)Math.Floor(target), 0, missingCount);
        int qCeil = Math.Min(missingCount, qFloor + 1);
        double latFloor = Math.Max(residentHitCount * gamma + qFloor * alpha, (missingCount - qFloor) * invBh);
        double latCeil = Math.Max(residentHitCount * gamma + qCeil * alpha, (missingCount - qCeil) * invBh);
        return latFloor <= latCeil ? qFloor : qCeil;
    }

    private void EnsureCapacity(int numExperts, int totalSlots)
    {
        if (_expertToUnique.Length < numExperts)
        {
            int newSize = Math.Max(numExperts, _expertToUnique.Length == 0 ? 64 : _expertToUnique.Length * 2);
            var nextMap = new int[newSize];
            Array.Fill(nextMap, -1);
            _expertToUnique = nextMap;
        }

        if (_unique.Length < totalSlots)
        {
            int newCap = Math.Max(totalSlots, _unique.Length == 0 ? 32 : _unique.Length * 2);
            _unique = new PlannedUniqueExpert[newCap];
            _missOrder = new int[newCap];
            _protectedExperts = new int[newCap];
        }
    }

    /// <summary>
    /// Computes the byte footprint of one routed expert across Gate, Up, and Down projections.
    /// </summary>
    public static unsafe long EstimateExpertByteSize(LayerWeights lw, int dim, int expertFfnDim)
    {
        ArgumentNullException.ThrowIfNull(lw);
        bool fused = lw.FfnGateUpExpsWeight != null && lw.FfnGateExpsWeight == null;
        if (fused)
        {
            long gateUpBytes = (long)(2 * expertFfnDim) * GgufTypes.GetRowBytes(lw.FfnGateUpExpsType, dim);
            long downBytes = (long)dim * GgufTypes.GetRowBytes(lw.FfnDownExpsType, expertFfnDim);
            return Math.Max(1L, gateUpBytes + downBytes);
        }

        long gBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(lw.FfnGateExpsType, dim);
        long uBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(lw.FfnUpExpsType, dim);
        long dBytes = (long)dim * GgufTypes.GetRowBytes(lw.FfnDownExpsType, expertFfnDim);
        return Math.Max(1L, gBytes + uBytes + dBytes);
    }

    /// <summary>
    /// Executes a single routed expert FFN (<c>SwiGLU(Gate(x), Up(x)) -> Down</c>) on CPU using
    /// <see cref="QuantKernels"/> and accumulates the weighted result into <paramref name="dstOut"/>.
    /// </summary>
    public static unsafe void ExecuteCpuSingleExpert(
        LayerWeights lw,
        int expertIdx,
        float weight,
        float* normX,
        float* normXSums,
        float* dstOut,
        int dim,
        int expertFfnDim,
        float* gateScratch,
        float* upScratch,
        float* actScratch,
        float* actSumsScratch,
        float* downOutScratch)
    {
        if (expertIdx < 0 || weight == 0f)
            return;

        bool fused = lw.FfnGateUpExpsWeight != null && lw.FfnGateExpsWeight == null;
        GgufType gateType = fused ? lw.FfnGateUpExpsType : lw.FfnGateExpsType;
        GgufType upType = fused ? lw.FfnGateUpExpsType : lw.FfnUpExpsType;
        GgufType downType = lw.FfnDownExpsType;

        long gateRowBytes = GgufTypes.GetRowBytes(gateType, dim);
        long halfFusedBytes = (long)expertFfnDim * gateRowBytes;
        long fusedSliceBytes = halfFusedBytes * 2L;
        long gateSliceBytes = (long)expertFfnDim * gateRowBytes;
        long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(upType, dim);
        long downSliceBytes = (long)dim * GgufTypes.GetRowBytes(downType, expertFfnDim);

        byte* expGate = fused
            ? lw.FfnGateUpExpsWeight + (long)expertIdx * fusedSliceBytes
            : lw.FfnGateExpsWeight + (long)expertIdx * gateSliceBytes;
        byte* expUp = fused
            ? expGate + halfFusedBytes
            : lw.FfnUpExpsWeight + (long)expertIdx * upSliceBytes;
        byte* expDown = lw.FfnDownExpsWeight + (long)expertIdx * downSliceBytes;

        QuantKernels.MatVecMul(gateType, expGate, normX, gateScratch, dim, expertFfnDim, normXSums);
        if (lw.FfnGateExpsBias != null)
            QuantKernels.VecAdd(lw.FfnGateExpsBias + (long)expertIdx * expertFfnDim, gateScratch, expertFfnDim);

        QuantKernels.MatVecMul(upType, expUp, normX, upScratch, dim, expertFfnDim, normXSums);
        if (lw.FfnUpExpsBias != null)
            QuantKernels.VecAdd(lw.FfnUpExpsBias + (long)expertIdx * expertFfnDim, upScratch, expertFfnDim);

        QuantKernels.SwiGLU(gateScratch, upScratch, actScratch, expertFfnDim);
        QuantKernels.ComputeBlockSums32(actScratch, actSumsScratch, expertFfnDim);

        QuantKernels.MatVecMul(downType, expDown, actScratch, downOutScratch, expertFfnDim, dim, actSumsScratch);
        if (lw.FfnDownExpsBias != null)
            QuantKernels.VecAdd(lw.FfnDownExpsBias + (long)expertIdx * dim, downOutScratch, dim);

        QuantKernels.VecAddWeighted(downOutScratch, dstOut, weight, dim);
    }
}
