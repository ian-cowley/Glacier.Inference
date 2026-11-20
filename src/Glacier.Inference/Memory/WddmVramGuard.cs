namespace Glacier.Inference.Memory;

using System;

/// <summary>
/// Point-in-time snapshot of GPU VRAM usage and safe headroom relative to the Windows WDDM cliff threshold.
/// </summary>
public readonly record struct VramHeadroomSnapshot(
    ulong DedicatedVramBytes,
    ulong OsBudgetBytes,
    ulong CurrentUsageBytes,
    ulong SafetyReserveBytes,
    ulong SafeAvailableBytes,
    bool IsAtCliffRisk);

/// <summary>
/// Prevents silent WDDM system-memory fallback cliffs (when free local VRAM hits 0 MiB on Windows)
/// by enforcing a dynamic safety reserve and recommending elastic expert-cache slot limits.
/// </summary>
public static class WddmVramGuard
{
    private const ulong GiB = 1024UL * 1024 * 1024;
    private const ulong MiB = 1024UL * 1024;

    /// <summary>
    /// Resolves an optional VRAM reserve in megabytes to explicit bytes, validating non-negative values.
    /// Returns <see langword="null"/> when <paramref name="vramReserveMb"/> is <see langword="null"/>,
    /// and <c>0UL</c> when <paramref name="vramReserveMb"/> is <c>0</c>.
    /// </summary>
    public static ulong? ResolveExplicitReserveBytes(int? vramReserveMb)
    {
        if (!vramReserveMb.HasValue)
            return null;

        if (vramReserveMb.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(vramReserveMb), vramReserveMb.Value, "VRAM reserve in MB must be non-negative.");

        return checked((ulong)vramReserveMb.Value) * MiB;
    }

    /// <summary>
    /// Resolves the safety reserve bytes used for model residency and MoE tiering decisions.
    /// When <paramref name="explicitReserveBytes"/> is provided (including <c>0UL</c>), it is returned verbatim without clamping.
    /// </summary>
    public static ulong ResolveModelSafetyReserveBytes(
        ulong dedicatedVramBytes,
        bool isWindowsWddm,
        bool isUnifiedMemory = false,
        ulong? explicitReserveBytes = null,
        ulong autoMinFloorBytes = 350_000_000UL)
    {
        if (explicitReserveBytes.HasValue)
            return explicitReserveBytes.Value;

        return Math.Max(
            autoMinFloorBytes,
            CalculateSafetyReserveBytes(dedicatedVramBytes, isWindowsWddm, isUnifiedMemory));
    }

    /// <summary>
    /// Calculates the recommended free VRAM safety reserve for desktop/driver headroom.
    /// </summary>
    public static ulong CalculateSafetyReserveBytes(
        ulong dedicatedVramBytes,
        bool isWindowsWddm,
        bool isUnifiedMemory = false)
    {
        if (dedicatedVramBytes == 0)
            return 0UL;

        if (!isWindowsWddm || isUnifiedMemory)
            return Math.Min(256UL * MiB, dedicatedVramBytes / 16);

        if (dedicatedVramBytes < 4UL * GiB)
            return 384UL * MiB;
        if (dedicatedVramBytes < 8UL * GiB)
            return 640UL * MiB;
        if (dedicatedVramBytes < 16UL * GiB)
            return 896UL * MiB;

        return Math.Clamp(dedicatedVramBytes / 16, 1024UL * MiB, 1536UL * MiB);
    }

    /// <summary>
    /// Evaluates current VRAM headroom against OS budget and WDDM cliff risk.
    /// </summary>
    public static VramHeadroomSnapshot EvaluateHeadroom(
        ulong dedicatedVramBytes,
        ulong osBudgetBytes,
        ulong currentUsageBytes,
        bool isWindowsWddm,
        bool isUnifiedMemory = false,
        ulong? explicitReserveBytes = null)
    {
        ulong effectiveBudget = osBudgetBytes > 0
            ? (dedicatedVramBytes > 0 ? Math.Min(dedicatedVramBytes, osBudgetBytes) : osBudgetBytes)
            : dedicatedVramBytes;

        ulong reserve = explicitReserveBytes
            ?? CalculateSafetyReserveBytes(effectiveBudget, isWindowsWddm, isUnifiedMemory);

        ulong safeCeiling = effectiveBudget > reserve ? effectiveBudget - reserve : 0UL;
        ulong safeAvailable = safeCeiling > currentUsageBytes ? safeCeiling - currentUsageBytes : 0UL;
        bool cliffRisk = effectiveBudget > 0 && currentUsageBytes >= safeCeiling;

        return new VramHeadroomSnapshot(
            DedicatedVramBytes: dedicatedVramBytes,
            OsBudgetBytes: effectiveBudget,
            CurrentUsageBytes: currentUsageBytes,
            SafetyReserveBytes: reserve,
            SafeAvailableBytes: safeAvailable,
            IsAtCliffRisk: cliffRisk);
    }

    /// <summary>
    /// Computes an elastic active slot count that stays safely below the WDDM cliff ceiling.
    /// </summary>
    public static int ComputeRecommendedSlotCount(
        VramHeadroomSnapshot snapshot,
        ulong slotByteSize,
        int currentActiveSlots,
        int maxSlotCapacity,
        int minSlotFloor = 1)
    {
        if (slotByteSize == 0 || maxSlotCapacity <= 0)
            return 0;

        minSlotFloor = Math.Clamp(minSlotFloor, 0, maxSlotCapacity);
        currentActiveSlots = Math.Clamp(currentActiveSlots, 0, maxSlotCapacity);

        ulong safeCeiling = snapshot.OsBudgetBytes > snapshot.SafetyReserveBytes
            ? snapshot.OsBudgetBytes - snapshot.SafetyReserveBytes
            : 0UL;

        if (snapshot.IsAtCliffRisk)
        {
            ulong excess = snapshot.CurrentUsageBytes > safeCeiling
                ? snapshot.CurrentUsageBytes - safeCeiling
                : 0UL;
            int slotsToDrop = excess > 0
                ? (int)((excess + slotByteSize - 1UL) / slotByteSize)
                : 1;
            return Math.Clamp(currentActiveSlots - Math.Max(1, slotsToDrop), minSlotFloor, maxSlotCapacity);
        }

        int extraSlots = (int)Math.Min(int.MaxValue, snapshot.SafeAvailableBytes / slotByteSize);
        return Math.Clamp(currentActiveSlots + extraSlots, minSlotFloor, maxSlotCapacity);
    }
}
