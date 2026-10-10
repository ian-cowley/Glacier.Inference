namespace Glacier.Inference.Scheduling;

/// <summary>
/// Operational hardware bandwidth and execution telemetry recorded by the co-scheduler profiler.
/// </summary>
public sealed record BandwidthTelemetry(
    double EffectivePcieBandwidthGBps,
    double EffectiveCpuBandwidthGBps,
    double VramBandwidthGBps,
    long PcieSamplesRecorded,
    long CpuSamplesRecorded,
    double TotalPcieTransferredGB,
    double TotalCpuProcessedGB);
