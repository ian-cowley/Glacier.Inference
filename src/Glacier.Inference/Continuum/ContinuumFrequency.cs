namespace Glacier.Inference.Continuum;

using System.Runtime.InteropServices;

/// <summary>
/// Update frequency level in the HOPE (Hierarchical Optimizing Processing Ensemble) continuum memory hierarchy.
/// </summary>
public enum ContinuumFrequency : byte
{
    /// <summary>
    /// Fast working memory updated every token (omega_0: stride = 1).
    /// </summary>
    PerToken = 0,

    /// <summary>
    /// Short-window episodic memory updated every block (omega_1: default stride = 64 or 4,096).
    /// </summary>
    ShortWindow = 1,

    /// <summary>
    /// Session-level associative memory updated at intermediate intervals (omega_2: default stride = 4,096 or 65,536).
    /// </summary>
    Session = 2,

    /// <summary>
    /// Slow consolidated continuum memory updated on high-surprise events or macro-epochs (omega_3).
    /// </summary>
    Consolidated = 3
}

/// <summary>
/// Bitflags describing the lifecycle state of an <see cref="AssociativeMemoryBlockHeader"/>.
/// </summary>
[Flags]
public enum ContinuumStateFlags : byte
{
    None = 0,
    Active = 1 << 0,
    Sleeping = 1 << 1,
    Pruning = 1 << 2,
    Frozen = 1 << 3
}

/// <summary>
/// Fixed 64-byte cache-line aligned header for a HOPE/Titans linear associative memory block.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64)]
public unsafe struct AssociativeMemoryBlockHeader
{
    [FieldOffset(0)] public ContinuumFrequency Frequency;
    [FieldOffset(1)] public ContinuumStateFlags StateFlags;
    [FieldOffset(2)] public ushort Dimension;
    [FieldOffset(4)] public int ClockStride;
    [FieldOffset(8)] public long SequenceTokenClock;
    [FieldOffset(16)] public float RetentionRateLambda;
    [FieldOffset(20)] public float MomentumDecayTheta;
    [FieldOffset(24)] public float LearningRateEta;
    [FieldOffset(28)] public float LastSurpriseMetric;
    [FieldOffset(32)] public ulong MemoryPointer;
    [FieldOffset(40)] public ulong MomentumPointer;
    [FieldOffset(48)] public ulong AdjointMemoryPointer;
    [FieldOffset(56)] public ulong AdjointMomentumPointer;
}
