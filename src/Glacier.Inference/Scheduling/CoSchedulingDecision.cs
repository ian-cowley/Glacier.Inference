namespace Glacier.Inference.Scheduling;

using System;
using System.Collections.Generic;

/// <summary>
/// Partitioned execution plan produced by the dynamic bandwidth-adaptive co-scheduler for an MoE layer.
/// Balances missing expert computation between GPU PCIe DMA transfer and CPU SIMD memory bus execution.
/// </summary>
public sealed class CoSchedulingDecision
{
    public int LayerIndex { get; init; }

    /// <summary>
    /// Expert indices already resident in GPU VRAM (no PCIe transfer needed).
    /// </summary>
    public IReadOnlyList<int> GpuResidentHits { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Missing expert indices selected to be transferred over PCIe and executed on GPU (q* experts).
    /// </summary>
    public IReadOnlyList<int> GpuTransfers { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Missing expert indices selected to be executed directly in host RAM via CPU SIMD (m - q* experts).
    /// </summary>
    public IReadOnlyList<int> CpuExecutions { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Predicted execution time on GPU (including PCIe transfer and VRAM compute).
    /// </summary>
    public double PredictedGpuLatencyMs { get; init; }

    /// <summary>
    /// Predicted execution time on CPU (evaluating m - q* experts in host RAM).
    /// </summary>
    public double PredictedCpuLatencyMs { get; init; }

    /// <summary>
    /// Overall predicted co-scheduled layer step latency: max(PredictedGpuLatencyMs, PredictedCpuLatencyMs).
    /// </summary>
    public double PredictedTotalLatencyMs { get; init; }

    public int TotalActivated => GpuResidentHits.Count + GpuTransfers.Count + CpuExecutions.Count;
    public int MissingCount => GpuTransfers.Count + CpuExecutions.Count;
    public int QStar => GpuTransfers.Count;
}
