namespace Glacier.Inference.Hardware;

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
#if !ANDROID
using Vortice.DXGI;
#endif

/// <summary>
/// Live snapshot of physical hardware capabilities, free memory headroom, and measured SIMD bandwidth
/// captured on the active machine at inference startup.
/// </summary>
public sealed record HardwareRuntimeSnapshot
{
    public required string OsDescription { get; init; }
    public required string CpuName { get; init; }
    public required int LogicalCores { get; init; }
    public required bool HasAvx512F { get; init; }
    public required bool HasAvx2 { get; init; }
    public required bool HasFma { get; init; }
    public required bool HasArmAdvSimd { get; init; }
    public required string ActiveSimdTier { get; init; }
    public required ulong TotalPhysicalRamBytes { get; init; }
    public required ulong AvailablePhysicalRamBytes { get; init; }
    public required DeviceInfo ActiveDevice { get; init; }
    public required bool IsIntegratedUma { get; init; }
    public required ulong LiveVramBudgetBytes { get; init; }
    public required ulong LiveVramFreeBytes { get; init; }
    public required double MeasuredCpuSimdBandwidthGBps { get; init; }

    public double TotalRamGb => TotalPhysicalRamBytes / (1024.0 * 1024.0 * 1024.0);
    public double AvailableRamGb => AvailablePhysicalRamBytes / (1024.0 * 1024.0 * 1024.0);
    public double LiveVramBudgetGb => LiveVramBudgetBytes / (1024.0 * 1024.0 * 1024.0);
    public double LiveVramFreeGb => LiveVramFreeBytes / (1024.0 * 1024.0 * 1024.0);
}

/// <summary>
/// Hardware-calibrated execution defaults derived from a <see cref="HardwareRuntimeSnapshot"/>.
/// </summary>
public sealed record HardwareExecutionPolicy
{
    public required int RecommendedMtpDraftTokens { get; init; }
    public required PleIoMode RecommendedPleIoMode { get; init; }
    public required ulong RecommendedHostArenaBytes { get; init; }
    public required double InitialCpuBandwidthGBps { get; init; }
    public required double InitialPcieBandwidthGBps { get; init; }
    public required double InitialVramBandwidthGBps { get; init; }
}

/// <summary>
/// Probes live CPU SIMD ISA support, host physical RAM availability, GPU VRAM headroom, and
/// multi-threaded SIMD memory bandwidth so Glacier adapts cleanly to the active machine.
/// </summary>
public static unsafe class HardwareRuntimeProfiler
{
    private const ulong GiB = 1024UL * 1024 * 1024;
    private const ulong MiB = 1024UL * 1024;

    private static readonly Lock _probeLock = new();
    private static double _cachedCpuBandwidthGBps;

    public static HardwareRuntimeSnapshot CaptureSnapshot(DeviceInfo? targetDevice = null, bool forceReprobe = false)
    {
        DeviceInfo cpuDev = ResolveCpuDevice();
        DeviceInfo activeDev = targetDevice ?? DeviceManager.GetOptimalDevice();

        bool hasAvx512F = Vector512.IsHardwareAccelerated || Avx512F.IsSupported;
        bool hasAvx2 = Vector256.IsHardwareAccelerated || Avx2.IsSupported;
        bool hasFma = Fma.IsSupported;
        bool hasNeon = AdvSimd.IsSupported;

        string simdTier = hasAvx512F
            ? "AVX-512"
            : (hasAvx2 ? (hasFma ? "AVX2+FMA" : "AVX2") : (hasNeon ? "ARM-NEON" : "Scalar"));

        (ulong totalRam, ulong availRam) = QueryHostPhysicalMemory();
        bool isUma = activeDev.Vendor != GpuVendor.Cpu &&
                     activeDev.DedicatedVramBytes < 3UL * GiB &&
                     activeDev.SharedVramBytes >= 2UL * GiB;

        (ulong vramBudget, ulong vramFree) = QueryLiveVramHeadroom(activeDev, isUma);
        double measuredBw = MeasureOrGetCpuSimdBandwidthGBps(forceReprobe);

        return new HardwareRuntimeSnapshot
        {
            OsDescription = RuntimeInformation.OSDescription.Trim(),
            CpuName = cpuDev.Name,
            LogicalCores = Environment.ProcessorCount,
            HasAvx512F = hasAvx512F,
            HasAvx2 = hasAvx2,
            HasFma = hasFma,
            HasArmAdvSimd = hasNeon,
            ActiveSimdTier = simdTier,
            TotalPhysicalRamBytes = totalRam,
            AvailablePhysicalRamBytes = availRam,
            ActiveDevice = activeDev,
            IsIntegratedUma = isUma,
            LiveVramBudgetBytes = vramBudget,
            LiveVramFreeBytes = vramFree,
            MeasuredCpuSimdBandwidthGBps = measuredBw
        };
    }

    public static ulong EstimateTotalModelAndPleBytes(ModelWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        ulong totalBytes = 0;
        var tensorList = weights.Gguf.TensorList;
        for (int i = 0; i < tensorList.Count; i++)
        {
            totalBytes += tensorList[i].GetByteSize();
        }

        if (totalBytes == 0)
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(weights.Gguf.FilePath) && File.Exists(weights.Gguf.FilePath))
                {
                    totalBytes += (ulong)Math.Max(0L, new FileInfo(weights.Gguf.FilePath).Length);
                }
            }
            catch
            {
                // Keep conservative estimate if metadata file stat fails
            }
        }

        try
        {
            string? plePath = weights.Gguf.PleTableFilePath;
            if (!weights.Gguf.Tensors.ContainsKey("per_layer_token_embd.weight") &&
                !string.IsNullOrWhiteSpace(plePath) &&
                !string.Equals(plePath, weights.Gguf.FilePath, StringComparison.OrdinalIgnoreCase) &&
                File.Exists(plePath))
            {
                totalBytes += (ulong)Math.Max(0L, new FileInfo(plePath).Length);
            }
        }
        catch
        {
            // Ignore external PLE stat errors
        }

        return totalBytes;
    }

    public static HardwareExecutionPolicy RecommendPolicy(HardwareRuntimeSnapshot snapshot, ModelWeights? weights = null)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        // MoE models pay an expert-fanout bandwidth tax on every speculative draft token, making K=2 optimal;
        // dense models benefit cleanly from deeper K=4 speculative verification rounds.
        int draftTokens = weights?.IsMoe == true ? 2 : 4;

        // Reserve at least 4 GiB (or 1/8 of RAM, capped cleanly on <16 GiB hosts) for OS / foreground apps
        ulong margin = Math.Max(4UL * GiB, snapshot.TotalPhysicalRamBytes / 8UL);
        if (snapshot.TotalPhysicalRamBytes < 16UL * GiB)
        {
            margin = Math.Min(margin, Math.Max(1UL * GiB, snapshot.AvailablePhysicalRamBytes / 2UL));
        }

        ulong safeHostArena = snapshot.AvailablePhysicalRamBytes > margin
            ? snapshot.AvailablePhysicalRamBytes - margin
            : snapshot.AvailablePhysicalRamBytes / 2UL;

        PleIoMode recommendedPleIo = PleIoMode.DirectIo;
        if (weights != null)
        {
            ulong totalModelAndPle = EstimateTotalModelAndPleBytes(weights);
            long mapBytes = (long)Math.Min((ulong)long.MaxValue, totalModelAndPle);
            recommendedPleIo = DirectIoNgramTable.ResolveIoMode(
                PleIoMode.Auto,
                mapBytes,
                snapshot.AvailablePhysicalRamBytes,
                margin);
        }

        double cpuBw = Math.Max(5.0, snapshot.MeasuredCpuSimdBandwidthGBps);
        double pcieBw = snapshot.IsIntegratedUma ? Math.Max(40.0, cpuBw * 4.0) : 14.0;
        double vramGb = snapshot.ActiveDevice.DedicatedVramGb;
        double vramBw = snapshot.IsIntegratedUma || snapshot.ActiveDevice.Vendor == GpuVendor.Cpu
            ? Math.Max(80.0, cpuBw)
            : (vramGb >= 20.0 ? 800.0 : (vramGb >= 11.0 ? 360.0 : 256.0));

        return new HardwareExecutionPolicy
        {
            RecommendedMtpDraftTokens = draftTokens,
            RecommendedPleIoMode = recommendedPleIo,
            RecommendedHostArenaBytes = safeHostArena,
            InitialCpuBandwidthGBps = cpuBw,
            InitialPcieBandwidthGBps = pcieBw,
            InitialVramBandwidthGBps = vramBw
        };
    }

    public static BandwidthCoScheduler CreateCalibratedCoScheduler(HardwareRuntimeSnapshot snapshot)
    {
        var policy = RecommendPolicy(snapshot);
        return new BandwidthCoScheduler(
            initialPcieGBps: policy.InitialPcieBandwidthGBps,
            initialCpuGBps: policy.InitialCpuBandwidthGBps,
            vramGBps: policy.InitialVramBandwidthGBps);
    }

    public static string FormatSummaryLine(HardwareRuntimeSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        string devTag = snapshot.ActiveDevice.Vendor == GpuVendor.Cpu
            ? "Device=CPU"
            : snapshot.IsIntegratedUma
                ? $"Device={snapshot.ActiveDevice.Name} (UMA SharedRAM={snapshot.LiveVramFreeGb:F1} GB free)"
                : $"Device={snapshot.ActiveDevice.Name} (VRAM Free={snapshot.LiveVramFreeGb:F1}/{snapshot.LiveVramBudgetGb:F1} GB)";

        return $"SIMD={snapshot.ActiveSimdTier} ({snapshot.LogicalCores}T) | FreeRAM={snapshot.AvailableRamGb:F1}/{snapshot.TotalRamGb:F1} GB | MeasuredRAM={snapshot.MeasuredCpuSimdBandwidthGBps:F1} GB/s | {devTag}";
    }

    private static DeviceInfo ResolveCpuDevice()
    {
        var devices = DeviceManager.GetDevices();
        for (int i = 0; i < devices.Count; i++)
        {
            if (devices[i].Vendor == GpuVendor.Cpu)
                return devices[i];
        }
        return DeviceManager.ResolveDevice("cpu");
    }

    private static double MeasureOrGetCpuSimdBandwidthGBps(bool forceReprobe)
    {
        lock (_probeLock)
        {
            if (!forceReprobe && _cachedCpuBandwidthGBps > 0.0)
                return _cachedCpuBandwidthGBps;

            double measured = RunStreamingBandwidthProbe();
            _cachedCpuBandwidthGBps = measured;
            return measured;
        }
    }

    private static double RunStreamingBandwidthProbe()
    {
        const int totalFloats = 8 * 1024 * 1024; // 32 MiB split into two disjoint 16 MiB vectors
        const int halfFloats = totalFloats / 2;
        const int vectorAlignedFloats = 64;

        float* buf = (float*)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
        try
        {
            for (int i = 0; i < totalFloats; i++)
            {
                buf[i] = ((i & 255) - 128) * 0.00390625f + 0.001f;
            }

            float* a = buf;
            float* b = buf + halfFloats;
            int workers = Math.Max(1, Environment.ProcessorCount);
            int chunk = (halfFloats / workers / vectorAlignedFloats) * vectorAlignedFloats;
            if (chunk < vectorAlignedFloats)
            {
                workers = 1;
                chunk = halfFloats;
            }

            double sink = 0.0;

            void ExecutePass()
            {
                double partialSum = 0.0;
                if (workers == 1)
                {
                    partialSum = QuantKernels.VecDotF32(a, b, halfFloats);
                }
                else
                {
                    nint addrA = (nint)a;
                    nint addrB = (nint)b;
                    Parallel.For(0, workers, () => 0.0, (t, _, local) =>
                    {
                        int start = t * chunk;
                        int len = (t == workers - 1) ? (halfFloats - start) : chunk;
                        int alignedLen = (len / 16) * 16;
                        if (alignedLen > 0)
                        {
                            local += QuantKernels.VecDotF32((float*)addrA + start, (float*)addrB + start, alignedLen);
                        }
                        return local;
                    }, local =>
                    {
                        double init, computed;
                        do
                        {
                            init = Volatile.Read(ref partialSum);
                            computed = init + local;
                        }
                        while (Interlocked.CompareExchange(ref partialSum, computed, init) != init);
                    });
                }
                sink += partialSum;
            }

            // Warm up JIT + threads once, then take the faster of 2 full 32 MiB streaming passes
            ExecutePass();
            double bestSec = double.MaxValue;
            double totalBytesRead = (double)totalFloats * sizeof(float);

            for (int pass = 0; pass < 2; pass++)
            {
                long startTs = Stopwatch.GetTimestamp();
                ExecutePass();
                double elapsedSec = Stopwatch.GetElapsedTime(startTs).TotalSeconds;
                if (elapsedSec > 0.0 && elapsedSec < bestSec)
                    bestSec = elapsedSec;
            }

            if (double.IsNaN(sink))
                return 10.0;

            double gbps = bestSec > 0.0 && bestSec < double.MaxValue
                ? (totalBytesRead / 1e9) / bestSec
                : 10.0;

            return double.IsFinite(gbps) && gbps > 0.0 ? Math.Clamp(gbps, 1.0, 2000.0) : 10.0;
        }
        finally
        {
            NativeMemory.AlignedFree(buf);
        }
    }

    private static (ulong TotalBytes, ulong AvailableBytes) QueryHostPhysicalMemory()
    {
#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            var mem = new MEMORYSTATUSEX { dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>() };
            if (GlobalMemoryStatusEx(ref mem))
            {
                return (mem.ullTotalPhys, mem.ullAvailPhys);
            }
        }
        else
#endif
        if (OperatingSystem.IsLinux() && File.Exists("/proc/meminfo"))
        {
            try
            {
                ulong totalKb = 0;
                ulong availKb = 0;
                ulong freeKb = 0;
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (line.StartsWith("MemTotal:", StringComparison.Ordinal))
                        totalKb = ParseProcMeminfoKb(line);
                    else if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                        availKb = ParseProcMeminfoKb(line);
                    else if (line.StartsWith("MemFree:", StringComparison.Ordinal))
                        freeKb = ParseProcMeminfoKb(line);
                }

                if (totalKb > 0)
                {
                    ulong effectiveAvailKb = availKb > 0 ? availKb : freeKb;
                    return (totalKb * 1024UL, effectiveAvailKb * 1024UL);
                }
            }
            catch
            {
                // Fall back to GC memory info below
            }
        }

        var gcInfo = GC.GetGCMemoryInfo();
        ulong fallbackTotal = (ulong)Math.Max(0L, gcInfo.TotalAvailableMemoryBytes);
        return (fallbackTotal, fallbackTotal);
    }

    private static ulong ParseProcMeminfoKb(string line)
    {
        int colon = line.IndexOf(':');
        if (colon < 0) return 0UL;
        ReadOnlySpan<char> span = line.AsSpan(colon + 1).Trim();
        int space = span.IndexOfAny(' ', '\t');
        if (space > 0) span = span.Slice(0, space);
        return ulong.TryParse(span, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong val) ? val : 0UL;
    }

    private static (ulong BudgetBytes, ulong FreeBytes) QueryLiveVramHeadroom(DeviceInfo dev, bool isUma)
    {
        if (dev.Vendor == GpuVendor.Cpu)
            return (0UL, 0UL);

        ulong budget = dev.DedicatedVramBytes;
        ulong free = dev.DedicatedVramBytes;

#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            try
            {
                using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
                int hwIdx = 0;
                for (uint i = 0; factory.EnumAdapters1(i, out IDXGIAdapter1 adapter).Success; i++)
                {
                    try
                    {
                        var desc = adapter.Description1;
                        if ((desc.Flags & AdapterFlags.Software) != 0)
                            continue;

                        if (hwIdx == dev.Index)
                        {
                            ulong ded = (ulong)desc.DedicatedVideoMemory;
                            budget = ded;
                            free = ded;

                            using var adapter3 = adapter.QueryInterfaceOrNull<IDXGIAdapter3>();
                            if (adapter3 != null)
                            {
                                var vmi = adapter3.QueryVideoMemoryInfo(0u, MemorySegmentGroup.Local);
                                var snap = WddmVramGuard.EvaluateHeadroom(
                                    ded,
                                    vmi.Budget,
                                    vmi.CurrentUsage,
                                    isWindowsWddm: true,
                                    isUnifiedMemory: isUma);
                                budget = snap.OsBudgetBytes;
                                free = snap.SafeAvailableBytes;
                            }
                            break;
                        }
                        hwIdx++;
                    }
                    finally
                    {
                        adapter.Dispose();
                    }
                }
            }
            catch
            {
                // Keep static DXGI totals if live VideoMemoryInfo query is unavailable
            }
        }
#endif

        return (budget, free);
    }

#if !ANDROID
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    private struct MEMORYSTATUSEX
    {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
    }
#endif
}
