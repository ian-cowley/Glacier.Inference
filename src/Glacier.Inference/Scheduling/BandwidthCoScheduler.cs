namespace Glacier.Inference.Scheduling;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using Glacier.Inference.Memory;

/// <summary>
/// Dynamic bandwidth-adaptive CPU/GPU co-scheduling policy engine and profiler.
/// Implements the q* equilibrium formula to dynamically partition sparse MoE expert execution
/// between GPU PCIe bus transfer and CPU SIMD (AVX-512 / AVX2) memory bandwidth.
/// </summary>
public sealed class BandwidthCoScheduler
{
    private double _pcieBandwidthGBps;   // Bp: Effective PCIe transfer bandwidth
    private double _cpuBandwidthGBps;    // Bh: Effective host memory read bandwidth
    private double _vramBandwidthGBps;   // Bvram: Internal GPU VRAM memory bandwidth
    private readonly double _ewmaAlpha;  // Moving average smoothing factor

    private long _pcieSamplesRecorded;
    private long _cpuSamplesRecorded;
    private long _totalPcieBytes;
    private long _totalCpuBytes;
    private readonly Lock _telemetryLock = new();

    public double PcieBandwidthGBps => Volatile.Read(ref _pcieBandwidthGBps);
    public double CpuBandwidthGBps => Volatile.Read(ref _cpuBandwidthGBps);
    public double VramBandwidthGBps => Volatile.Read(ref _vramBandwidthGBps);
    public double EwmaAlpha => _ewmaAlpha;

    /// <summary>
    /// Initializes a new co-scheduler profiler instance with baseline hardware capabilities.
    /// Default values correspond to Node 1 (Host Laptop: Ryzen AI 9 Zen 5 + RTX 4060).
    /// </summary>
    public BandwidthCoScheduler(
        double initialPcieGBps = 12.5,
        double initialCpuGBps = 65.0,
        double vramGBps = 272.0,
        double ewmaAlpha = 0.15)
    {
        if (!double.IsFinite(initialPcieGBps) || initialPcieGBps <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialPcieGBps), "Initial PCIe bandwidth must be finite and positive.");
        if (!double.IsFinite(initialCpuGBps) || initialCpuGBps <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialCpuGBps), "Initial CPU bandwidth must be finite and positive.");
        if (!double.IsFinite(vramGBps) || vramGBps <= 0)
            throw new ArgumentOutOfRangeException(nameof(vramGBps), "VRAM bandwidth must be finite and positive.");
        if (!double.IsFinite(ewmaAlpha))
            throw new ArgumentOutOfRangeException(nameof(ewmaAlpha), "EWMA alpha must be finite.");

        _pcieBandwidthGBps = initialPcieGBps;
        _cpuBandwidthGBps = initialCpuGBps;
        _vramBandwidthGBps = vramGBps;
        _ewmaAlpha = Math.Clamp(ewmaAlpha, 0.01, 1.0);
    }

    /// <summary>
    /// Computes the optimal number of missing experts (q*) to stream to GPU over PCIe,
    /// accounting for pre-existing GPU compute load from resident VRAM hits.
    /// The remaining (m - q*) experts are executed concurrently on CPU SIMD.
    /// Evaluates discrete integer candidates (qFloor and qCeil) against exact latency predictions
    /// Math.Max(r * gamma + q * alpha, (m - q) * invBh) to strictly guarantee minimum step latency.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int CalculateOptimalGpuExpertCount(int missingCount, int residentHitCount = 0)
    {
        if (missingCount <= 0) return 0;
        residentHitCount = Math.Max(0, residentHitCount);

        double bp = PcieBandwidthGBps;
        double bh = CpuBandwidthGBps;
        double bv = VramBandwidthGBps;

        // Guard against non-finite or non-positive bandwidth telemetry
        if (!double.IsFinite(bp) || bp <= 0) bp = 12.5;
        if (!double.IsFinite(bh) || bh <= 0) bh = 65.0;
        if (!double.IsFinite(bv) || bv <= 0) bv = 272.0;

        double alpha = (1.0 / bp) + (1.0 / bv); // GPU transfer + compute per expert (s/GB)
        double invBh = 1.0 / bh;                 // CPU compute per expert (s/GB)
        double gamma = 1.0 / bv;                 // GPU resident hit compute per expert (s/GB)

        // Continuous equilibrium accounting for baseline resident GPU work:
        // residentHitCount * gamma + q * alpha = (missingCount - q) * invBh
        // q * (alpha + invBh) = missingCount * invBh - residentHitCount * gamma
        double continuousTarget = (missingCount * invBh - residentHitCount * gamma) / (alpha + invBh);

        // If resident GPU workload exceeds CPU capacity, route all missing experts to CPU
        if (continuousTarget <= 0.0)
        {
            return 0;
        }

        int qFloor = (int)Math.Floor(continuousTarget);
        qFloor = Math.Clamp(qFloor, 0, missingCount);

        int qCeil = Math.Min(missingCount, qFloor + 1);

        // Exact discrete latency evaluation for both integer candidates
        double latFloor = Math.Max(residentHitCount * gamma + qFloor * alpha, (missingCount - qFloor) * invBh);
        double latCeil = Math.Max(residentHitCount * gamma + qCeil * alpha, (missingCount - qCeil) * invBh);

        // Strict discrete minimum selection (favor qFloor on tie to minimize PCIe transfer volume)
        int qStar = latFloor <= latCeil ? qFloor : qCeil;
        return Math.Clamp(qStar, 0, missingCount);
    }

    /// <summary>
    /// Partitions an MoE layer's activated experts into GPU resident hits, GPU PCIe transfers, and CPU SIMD executions.
    /// </summary>
    public CoSchedulingDecision Schedule(
        int layerIdx,
        ReadOnlySpan<int> activatedExperts,
        GpuResidentSlotManager? slotManager,
        long expertByteSize)
    {
        if (expertByteSize <= 0) throw new ArgumentOutOfRangeException(nameof(expertByteSize));

        var residentHits = new List<int>();
        var missingExperts = new List<int>();

        for (int i = 0; i < activatedExperts.Length; i++)
        {
            int expIdx = activatedExperts[i];
            if (slotManager != null && slotManager.TryGetSlot(layerIdx, expIdx, out _, out _))
            {
                residentHits.Add(expIdx);
            }
            else
            {
                missingExperts.Add(expIdx);
            }
        }

        int m = missingExperts.Count;
        int r = residentHits.Count;

        // 1. Calculate optimal transfer count offsetting for resident GPU compute load
        int qStar = CalculateOptimalGpuExpertCount(m, r);

        // 2. Clamp transfers against available unpinned VRAM slots to prevent intra-layer eviction thrashing
        if (slotManager != null)
        {
            int availableSlots = slotManager.GetAvailableTransferSlots(layerIdx, residentHits);
            qStar = Math.Min(qStar, availableSlots);
        }

        var gpuTransfers = new List<int>(qStar);
        var cpuExecutions = new List<int>(m - qStar);

        for (int i = 0; i < m; i++)
        {
            if (i < qStar)
                gpuTransfers.Add(missingExperts[i]);
            else
                cpuExecutions.Add(missingExperts[i]);
        }

        // Predict execution latencies (in milliseconds)
        double expertBytesGB = expertByteSize / (1024.0 * 1024.0 * 1024.0);
        double bp = PcieBandwidthGBps;
        double bh = CpuBandwidthGBps;
        double bv = VramBandwidthGBps;

        double residentGpuTimeMs = residentHits.Count * (expertBytesGB / bv) * 1000.0;
        double transferGpuTimeMs = qStar * ((expertBytesGB / bp) + (expertBytesGB / bv)) * 1000.0;
        double predictedGpuMs = residentGpuTimeMs + transferGpuTimeMs;

        double predictedCpuMs = (m - qStar) * (expertBytesGB / bh) * 1000.0;
        double predictedTotalMs = Math.Max(predictedGpuMs, predictedCpuMs);

        return new CoSchedulingDecision
        {
            LayerIndex = layerIdx,
            GpuResidentHits = residentHits,
            GpuTransfers = gpuTransfers,
            CpuExecutions = cpuExecutions,
            PredictedGpuLatencyMs = predictedGpuMs,
            PredictedCpuLatencyMs = predictedCpuMs,
            PredictedTotalLatencyMs = predictedTotalMs
        };
    }

    /// <summary>
    /// Predicts pure GPU, pure CPU, and co-scheduled step latency across varying missing expert counts.
    /// </summary>
    public (double PureGpuMs, double PureCpuMs, double CoScheduledMs, int QStar) PredictLatencies(int missingCount, long expertByteSize, int residentCount = 0)
    {
        if (missingCount <= 0 || expertByteSize <= 0)
            return (0.0, 0.0, 0.0, 0);

        double expertBytesGB = expertByteSize / (1024.0 * 1024.0 * 1024.0);
        double bp = PcieBandwidthGBps;
        double bh = CpuBandwidthGBps;
        double bv = VramBandwidthGBps;

        int qStar = CalculateOptimalGpuExpertCount(missingCount, residentCount);

        double residentGpuMs = residentCount * (expertBytesGB / bv) * 1000.0;
        double pureGpuMs = residentGpuMs + missingCount * ((expertBytesGB / bp) + (expertBytesGB / bv)) * 1000.0;
        double pureCpuMs = missingCount * (expertBytesGB / bh) * 1000.0;

        double coGpuMs = residentGpuMs + qStar * ((expertBytesGB / bp) + (expertBytesGB / bv)) * 1000.0;
        double coCpuMs = (missingCount - qStar) * (expertBytesGB / bh) * 1000.0;
        double coScheduledMs = Math.Max(coGpuMs, coCpuMs);

        return (pureGpuMs, pureCpuMs, coScheduledMs, qStar);
    }

    /// <summary>
    /// Records measured empirical PCIe transfer telemetry and updates effective bandwidth via EWMA.
    /// </summary>
    public void RecordPcieTransfer(long bytesTransferred, double elapsedMilliseconds)
    {
        if (bytesTransferred <= 0 || !double.IsFinite(elapsedMilliseconds) || elapsedMilliseconds <= 0.0) return;

        double elapsedSeconds = elapsedMilliseconds / 1000.0;
        double bytesGigabytes = bytesTransferred / (1024.0 * 1024.0 * 1024.0);
        double observedGBps = bytesGigabytes / elapsedSeconds;

        if (!double.IsFinite(observedGBps) || observedGBps <= 0.0) return;

        lock (_telemetryLock)
        {
            _pcieBandwidthGBps = (_ewmaAlpha * observedGBps) + ((1.0 - _ewmaAlpha) * _pcieBandwidthGBps);
            _pcieSamplesRecorded++;
            _totalPcieBytes += bytesTransferred;
        }
    }

    /// <summary>
    /// Records measured empirical CPU SIMD execution telemetry and updates effective bandwidth via EWMA.
    /// </summary>
    public void RecordCpuExecution(long expertBytesProcessed, double elapsedMilliseconds)
    {
        if (expertBytesProcessed <= 0 || !double.IsFinite(elapsedMilliseconds) || elapsedMilliseconds <= 0.0) return;

        double elapsedSeconds = elapsedMilliseconds / 1000.0;
        double bytesGigabytes = expertBytesProcessed / (1024.0 * 1024.0 * 1024.0);
        double observedGBps = bytesGigabytes / elapsedSeconds;

        if (!double.IsFinite(observedGBps) || observedGBps <= 0.0) return;

        lock (_telemetryLock)
        {
            _cpuBandwidthGBps = (_ewmaAlpha * observedGBps) + ((1.0 - _ewmaAlpha) * _cpuBandwidthGBps);
            _cpuSamplesRecorded++;
            _totalCpuBytes += expertBytesProcessed;
        }
    }

    /// <summary>
    /// Records measured empirical GPU execution telemetry and updates internal VRAM bandwidth via EWMA.
    /// </summary>
    public void RecordGpuExecution(long expertBytesProcessed, double elapsedMilliseconds)
    {
        if (expertBytesProcessed <= 0 || !double.IsFinite(elapsedMilliseconds) || elapsedMilliseconds <= 0.0) return;

        double elapsedSeconds = elapsedMilliseconds / 1000.0;
        double bytesGigabytes = expertBytesProcessed / (1024.0 * 1024.0 * 1024.0);
        double observedGBps = bytesGigabytes / elapsedSeconds;

        if (!double.IsFinite(observedGBps) || observedGBps <= 0.0) return;

        lock (_telemetryLock)
        {
            _vramBandwidthGBps = (_ewmaAlpha * observedGBps) + ((1.0 - _ewmaAlpha) * _vramBandwidthGBps);
        }
    }

    /// <summary>
    /// Gets current profiler telemetry snapshot.
    /// </summary>
    public BandwidthTelemetry GetTelemetry()
    {
        lock (_telemetryLock)
        {
            return new BandwidthTelemetry(
                EffectivePcieBandwidthGBps: _pcieBandwidthGBps,
                EffectiveCpuBandwidthGBps: _cpuBandwidthGBps,
                VramBandwidthGBps: _vramBandwidthGBps,
                PcieSamplesRecorded: _pcieSamplesRecorded,
                CpuSamplesRecorded: _cpuSamplesRecorded,
                TotalPcieTransferredGB: _totalPcieBytes / (1024.0 * 1024.0 * 1024.0),
                TotalCpuProcessedGB: _totalCpuBytes / (1024.0 * 1024.0 * 1024.0));
        }
    }
}
