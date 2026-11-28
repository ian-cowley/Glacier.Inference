namespace Glacier.Inference.Hardware;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text.RegularExpressions;
using Glacier.Inference.Gpu;

/// <summary>
/// Discovers and manages physical compute devices (NVIDIA, AMD, Intel, Host CPU)
/// using pure zero-dependency Windows DXGI and CUDA driver P/Invoke.
/// Evaluates and enforces safe driver engine options per device.
/// </summary>
public static class DeviceManager
{
    private static readonly Lazy<List<DeviceInfo>> _devices = new(EnumerateDevices);

    public static IReadOnlyList<DeviceInfo> GetDevices() => _devices.Value;

    /// <summary>
    /// Returns all detected Neural Processing Units (NPUs), such as AMD XDNA 2 / XDNA 1 or Intel AI Boost.
    /// </summary>
    public static IReadOnlyList<DeviceInfo> GetNpuDevices()
    {
        var npus = new List<DeviceInfo>();
        foreach (var dev in GetDevices())
        {
            if (dev.IsNpu)
                npus.Add(dev);
        }
        return npus;
    }

    public static DeviceInfo GetOptimalDevice()
    {
        var list = GetDevices();
        // 1. Prefer NVIDIA discrete GPU with Bare-Metal SASS if available
        foreach (var dev in list)
        {
            if (!dev.IsNpu && dev.Vendor == GpuVendor.Nvidia && dev.SupportedEngines.Contains(InferenceEngineType.BareMetal))
                return dev;
        }

        // 2. Prefer AMD GPU (Ryzen iGPU or Radeon dGPU)
        foreach (var dev in list)
        {
            if (!dev.IsNpu && dev.Vendor == GpuVendor.Amd)
                return dev;
        }

        // 3. Prefer Intel GPU (Arc discrete or Core Ultra Xe integrated)
        foreach (var dev in list)
        {
            if (!dev.IsNpu && dev.Vendor == GpuVendor.Intel)
                return dev;
        }

        // 4. Prefer any other GPU
        foreach (var dev in list)
        {
            if (!dev.IsNpu && dev.Vendor is not GpuVendor.Cpu)
                return dev;
        }

        // 5. Fall back to Host CPU
        return list[^1];
    }

    /// <summary>
    /// Resolves a user-supplied device identifier (slug, index, or name substring) to a DeviceInfo.
    /// </summary>
    public static DeviceInfo ResolveDevice(string? query)
    {
        var list = GetDevices();
        if (string.IsNullOrWhiteSpace(query) || query.Equals("auto", StringComparison.OrdinalIgnoreCase))
            return GetOptimalDevice();

        // 1. Exact ID match
        foreach (var dev in list)
        {
            if (dev.Id.Equals(query, StringComparison.OrdinalIgnoreCase))
                return dev;
        }

        // 2. Numeric index match
        if (int.TryParse(query, out int idx))
        {
            foreach (var dev in list)
            {
                if (dev.Index == idx)
                    return dev;
            }
        }

        // 3. "gpu:N" format
        if (query.StartsWith("gpu:", StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(query.AsSpan(4), out int gpuIdx))
        {
            foreach (var dev in list)
            {
                if (dev.Index == gpuIdx)
                    return dev;
            }
        }

        // 4. Fuzzy match on device name / vendor
        string q = query.ToLowerInvariant();
        foreach (var dev in list)
        {
            string name = dev.Name.ToLowerInvariant();
            if (name.Contains(q))
                return dev;
        }

        // Vendor keyword shortcuts
        if (q is "npu" or "xdna" or "xdna2" or "xdna1" or "amd-xdna2" or "amd-xdna1" or "ryzen-ai" or "flm" or "fastflowlm" or "lemonade" or "npusidecar")
        {
            foreach (var dev in list)
                if (dev.IsNpu) return dev;
        }
        if (q is "nvidia" or "geforce" or "rtx" or "gtx" or "cuda" or "baremetal" or "sass")
        {
            foreach (var dev in list)
                if (!dev.IsNpu && dev.Vendor == GpuVendor.Nvidia) return dev;
        }
        if (q is "amd" or "radeon" or "rdna" or "890m" or "880m" or "780m" or "760m" or "680m" or "660m" or "apu" or "igpu" or "rx")
        {
            foreach (var dev in list)
                if (!dev.IsNpu && dev.Vendor == GpuVendor.Amd) return dev;
        }
        if (q is "intel" or "arc" or "xe" or "iris" or "uhd" or "battlemage" or "alchemist")
        {
            foreach (var dev in list)
                if (!dev.IsNpu && dev.Vendor == GpuVendor.Intel) return dev;
        }
        if (q is "cpu" or "host" or "simd" or "ryzen" or "core")
        {
            foreach (var dev in list)
                if (dev.Vendor == GpuVendor.Cpu) return dev;
        }

        throw new ArgumentException($"No compute device matching '{query}' was found. Run 'glacier devices' to inspect available hardware.");
    }

    private static List<DeviceInfo> EnumerateDevices()
    {
        var results = new List<DeviceInfo>();

#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            try
            {
                EnumerateDxgiAdapters(results);
            }
            catch
            {
                // Fallback if DXGI fails
            }
        }
#endif

        // If DXGI found no discrete GPUs or if on Linux, run native direct driver enumeration
        if (results.Count == 0)
        {
            try
            {
                EnumerateNativeGpuDrivers(results);
            }
            catch
            {
                // Fallback if native driver discovery fails
            }
        }

        // Detect Neural Processing Units (AMD XDNA 2 / XDNA 1, Intel AI Boost) via zero-dependency OS APIs
        try
        {
            EnumerateNpuAccelerators(results);
        }
        catch
        {
            // Ignore NPU discovery errors on restricted environments
        }

        // Always add Host CPU
        results.Add(GetHostCpuDevice(results.Count));
        return results;
    }

    private static unsafe void EnumerateNativeGpuDrivers(List<DeviceInfo> results)
    {
        // 1. AMD ROCm / HIP devices
        if (HipDriver.IsAvailable())
        {
            try
            {
                if (HipDriver.Init(0) == 0 && HipDriver.DeviceGetCount(out int hipCount) == 0)
                {
                    for (int i = 0; i < hipCount; i++)
                    {
                        if (HipDriver.DeviceGet(out int dev, i) == 0)
                        {
                            string name = HipDriver.GetDeviceName(dev);
                            ulong mem = 0;
                            if (HipDriver.DeviceTotalMem(out nuint bytes, dev) == 0)
                                mem = (ulong)bytes;

                            string slug = GenerateSlug(name, GpuVendor.Amd, results.Count);
                            var engines = new List<InferenceEngineType> { InferenceEngineType.BareMetal };
                            if (VulkanDriver.IsAvailable())
                                engines.Add(InferenceEngineType.Vulkan);
                            if (OperatingSystem.IsWindows())
                                engines.Add(InferenceEngineType.DirectML);
                            engines.Add(InferenceEngineType.Cpu);

                            results.Add(new DeviceInfo
                            {
                                Id = slug,
                                Index = results.Count,
                                Name = name,
                                Vendor = GpuVendor.Amd,
                                DedicatedVramBytes = mem,
                                SharedVramBytes = 0,
                                IsDisplayDevice = false,
                                SupportedEngines = engines,
                                RecommendedEngine = InferenceEngineType.BareMetal,
                                SafetyNotes = "AMD ROCm / HIP driver engine. Direct HSA / KFD kernel execution via AQL queues & hardware doorbells."
                            });
                        }
                    }
                }
            }
            catch
            {
                // Ignore HIP failure
            }
        }

        // 2. NVIDIA CUDA devices
        if (CuDriver.IsAvailable())
        {
            try
            {
                if (CuDriver.Init(0) == 0 && CuDriver.DeviceGetCount(out int cuCount) == 0)
                {
                    for (int i = 0; i < cuCount; i++)
                    {
                        if (CuDriver.DeviceGet(out int dev, i) == 0)
                        {
                            string name = CuDriver.GetDeviceName(dev);
                            ulong mem = 0;
                            if (CuDriver.DeviceTotalMem(out nuint bytes, dev) == 0)
                                mem = (ulong)bytes;

                            string slug = GenerateSlug(name, GpuVendor.Nvidia, results.Count);
                            var engines = new List<InferenceEngineType> { InferenceEngineType.BareMetal };
                            if (VulkanDriver.IsAvailable())
                                engines.Add(InferenceEngineType.Vulkan);
                            if (OperatingSystem.IsWindows())
                                engines.Add(InferenceEngineType.DirectML);
                            engines.Add(InferenceEngineType.Cpu);

                            results.Add(new DeviceInfo
                            {
                                Id = slug,
                                Index = results.Count,
                                Name = name,
                                Vendor = GpuVendor.Nvidia,
                                DedicatedVramBytes = mem,
                                SharedVramBytes = 0,
                                IsDisplayDevice = false,
                                SupportedEngines = engines,
                                RecommendedEngine = InferenceEngineType.BareMetal,
                                SafetyNotes = $"NVIDIA CUDA Bare-Metal driver engine via {(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so")}."
                            });
                        }
                    }
                }
            }
            catch
            {
                // Ignore CUDA failure on Linux
            }
        }

        // 3. Vulkan devices (Cross-vendor AMD, Intel, NVIDIA on Linux & Windows)
        if (VulkanDriver.IsAvailable())
        {
            try
            {
                var appInfo = new VulkanDriver.VkApplicationInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    apiVersion = VulkanDriver.MakeVersion(1, 2, 0)
                };
                var createInfo = new VulkanDriver.VkInstanceCreateInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    pApplicationInfo = new IntPtr(&appInfo)
                };
                if (VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out IntPtr inst) == 0 && inst != IntPtr.Zero)
                {
                    uint devCount = 0;
                    VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, null);
                    if (devCount > 0)
                    {
                        var devHandles = new IntPtr[devCount];
                        VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, devHandles);
                        for (int i = 0; i < devCount; i++)
                        {
                            VulkanDriver.GetPhysicalDeviceProperties(devHandles[i], out var props);
                            // Skip CPU / llvmpipe / software rasterizers (VK_PHYSICAL_DEVICE_TYPE_CPU = 4)
                            if (props.deviceType == 4) continue;

                            string name = props.deviceName;
                            if (string.IsNullOrWhiteSpace(name) || name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase))
                                continue;

                            // Avoid duplicate devices already discovered by HIP, CUDA, or DXGI
                            if (results.Exists(d => d.Name.Equals(name, StringComparison.OrdinalIgnoreCase)))
                                continue;

                            var vendor = props.vendorID switch
                            {
                                0x10DE => GpuVendor.Nvidia,
                                0x1002 => GpuVendor.Amd,
                                0x8086 => GpuVendor.Intel,
                                _ => GpuVendor.Unknown
                            };

                            VulkanDriver.GetPhysicalDeviceMemoryProperties(devHandles[i], out var memProps);
                            ulong vramBytes = 0;
                            var heaps = (VulkanDriver.VkMemoryHeap*)memProps.memoryHeaps;
                            for (uint h = 0; h < memProps.memoryHeapCount; h++)
                            {
                                if ((heaps[h].flags & 1) != 0)
                                    vramBytes += heaps[h].size;
                            }
                            if (vramBytes == 0)
                            {
                                for (uint h = 0; h < memProps.memoryHeapCount; h++)
                                    vramBytes += heaps[h].size;
                            }

                            string slug = GenerateSlug(name, vendor, results.Count);
                            var engines = new List<InferenceEngineType> { InferenceEngineType.Vulkan, InferenceEngineType.BareMetal, InferenceEngineType.Cpu };

                            results.Add(new DeviceInfo
                            {
                                Id = slug,
                                Index = results.Count,
                                Name = name,
                                Vendor = vendor,
                                DedicatedVramBytes = vramBytes,
                                SharedVramBytes = 0,
                                IsDisplayDevice = false,
                                SupportedEngines = engines,
                                RecommendedEngine = InferenceEngineType.Vulkan,
                                SafetyNotes = $"Vulkan compute accelerator (API {VulkanDriver.VersionMajor(props.apiVersion)}.{VulkanDriver.VersionMinor(props.apiVersion)})."
                            });
                        }
                    }
                    VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
                }
            }
            catch
            {
                // Ignore Vulkan enumeration failure
            }
        }
    }

#if !ANDROID
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static void EnumerateDxgiAdapters(List<DeviceInfo> results)
    {
        var factoryGuid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387"); // IID_IDXGIFactory1
        int hr = CreateDXGIFactory1(ref factoryGuid, out IntPtr factoryPtr);
        if (hr != 0 || factoryPtr == IntPtr.Zero) return;

        try
        {
            var factory = (IDXGIFactory1)Marshal.GetObjectForIUnknown(factoryPtr);
            uint index = 0;

            while (true)
            {
                hr = factory.EnumAdapters1(index, out IDXGIAdapter1 adapter);
                if (hr != 0 || adapter == null) break;

                try
                {
                    hr = adapter.GetDesc1(out DXGI_ADAPTER_DESC1 desc);
                    if (hr == 0)
                    {
                        // Skip software rasterizer (DXGI_ADAPTER_FLAG_SOFTWARE = 2)
                        bool isSoftware = (desc.Flags & 2) != 0;
                        if (!isSoftware)
                        {
                            // Check if this adapter is driving a display monitor
                            bool isDisplay = adapter.EnumOutputs(0, out IntPtr pOutput) == 0;
                            if (pOutput != IntPtr.Zero)
                            {
                                Marshal.Release(pOutput);
                            }

                            var vendor = desc.VendorId switch
                            {
                                0x10DE => GpuVendor.Nvidia,
                                0x1002 => GpuVendor.Amd,
                                0x8086 => GpuVendor.Intel,
                                0x1414 => GpuVendor.Microsoft,
                                _ => GpuVendor.Unknown
                            };

                            string slug = GenerateSlug(desc.Description, vendor, (int)index);
                            var supportedEngines = new List<InferenceEngineType>();
                            InferenceEngineType recommendedEngine;
                            string safetyNotes;

                            if (vendor == GpuVendor.Nvidia)
                            {
                                bool bareMetalAvail = GpuContext.IsSupported;
                                if (bareMetalAvail)
                                    supportedEngines.Add(InferenceEngineType.BareMetal);
                                if (VulkanDriver.IsAvailable())
                                    supportedEngines.Add(InferenceEngineType.Vulkan);
                                supportedEngines.Add(InferenceEngineType.DirectML);
                                supportedEngines.Add(InferenceEngineType.Cpu);
                                recommendedEngine = bareMetalAvail ? InferenceEngineType.BareMetal : (VulkanDriver.IsAvailable() ? InferenceEngineType.Vulkan : InferenceEngineType.DirectML);
                                safetyNotes = "Bare-metal GPU driver engine. Bypasses CUDA Toolkit & cudart64.dll runtime. Direct3D 12, Vulkan & DirectML also supported.";
                            }
                            else if (vendor == GpuVendor.Amd)
                            {
                                bool hipAvail = HipDriver.IsAvailable();
                                bool vkAvail = VulkanDriver.IsAvailable();

                                if (hipAvail)
                                    supportedEngines.Add(InferenceEngineType.BareMetal);
                                if (vkAvail)
                                    supportedEngines.Add(InferenceEngineType.Vulkan);
                                supportedEngines.Add(InferenceEngineType.DirectML);
                                supportedEngines.Add(InferenceEngineType.Cpu);

                                if (hipAvail)
                                {
                                    recommendedEngine = InferenceEngineType.BareMetal;
                                    safetyNotes = "AMD ROCm / HIP driver engine (amdhip64.dll). Vulkan Hardware Tensor and DirectML also supported.";
                                }
                                else if (vkAvail)
                                {
                                    recommendedEngine = InferenceEngineType.Vulkan;
                                    safetyNotes = "Vulkan Hardware Tensor engine (VK_KHR_cooperative_matrix). DirectML also supported.";
                                }
                                else
                                {
                                    recommendedEngine = InferenceEngineType.DirectML;
                                    safetyNotes = "AMD Radeon / Ryzen APU: Pure C# Direct3D 12 Compute engine (HLSL Wave32) across unified system memory.";
                                }
                            }
                            else if (vendor == GpuVendor.Intel)
                            {
                                if (VulkanDriver.IsAvailable())
                                    supportedEngines.Add(InferenceEngineType.Vulkan);
                                supportedEngines.Add(InferenceEngineType.DirectML);
                                supportedEngines.Add(InferenceEngineType.Cpu);
                                recommendedEngine = VulkanDriver.IsAvailable() ? InferenceEngineType.Vulkan : InferenceEngineType.DirectML;
                                safetyNotes = "Intel Arc / Core Ultra Xe GPU: Vulkan Cooperative Matrix & DirectML / DX12 Compute engine supported.";
                            }
                            else
                            {
                                if (VulkanDriver.IsAvailable())
                                    supportedEngines.Add(InferenceEngineType.Vulkan);
                                supportedEngines.Add(InferenceEngineType.DirectML);
                                supportedEngines.Add(InferenceEngineType.Cpu);
                                recommendedEngine = VulkanDriver.IsAvailable() ? InferenceEngineType.Vulkan : InferenceEngineType.DirectML;
                                safetyNotes = "Vulkan Cooperative Matrix & DirectML / DX12 Compute engine supported.";
                            }

                            results.Add(new DeviceInfo
                            {
                                Id = slug,
                                Index = (int)index,
                                Name = desc.Description.Trim(),
                                Vendor = vendor,
                                DedicatedVramBytes = (ulong)desc.DedicatedVideoMemory,
                                SharedVramBytes = (ulong)desc.SharedSystemMemory,
                                IsDisplayDevice = isDisplay,
                                SupportedEngines = supportedEngines,
                                RecommendedEngine = recommendedEngine,
                                SafetyNotes = safetyNotes
                            });
                        }
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(adapter);
                }

                index++;
            }
        }
        finally
        {
            Marshal.Release(factoryPtr);
        }
    }
#endif

    private static DeviceInfo GetHostCpuDevice(int index)
    {
        string cpuName = "Host CPU";
#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            try
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                var val = key?.GetValue("ProcessorNameString") as string;
                if (!string.IsNullOrWhiteSpace(val))
                    cpuName = val.Trim();
            }
            catch
            {
                // Fallback if registry read fails
            }
        }
#endif
        if (cpuName == "Host CPU")
        {
            cpuName = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "Host CPU";
        }

        ulong totalRam = GetPhysicalMemoryBytes();
        string simdSupport = Vector512.IsHardwareAccelerated ? "AVX-512" : (Vector256.IsHardwareAccelerated ? "AVX2" : "SIMD");

        return new DeviceInfo
        {
            Id = "cpu",
            Index = index,
            Name = $"{cpuName} ({Environment.ProcessorCount} Threads, {simdSupport})",
            Vendor = GpuVendor.Cpu,
            DedicatedVramBytes = 0,
            SharedVramBytes = totalRam,
            IsDisplayDevice = false,
            SupportedEngines = [InferenceEngineType.Cpu],
            RecommendedEngine = InferenceEngineType.Cpu,
            SafetyNotes = $"Host multi-threaded CPU execution using SIMD {simdSupport} hardware intrinsics."
        };
    }

    private static string GenerateSlug(string description, GpuVendor vendor, int index)
    {
        string clean = description.ToLowerInvariant();
        if (vendor == GpuVendor.Nvidia)
        {
            if (clean.Contains("5090")) return "nvidia-rtx-5090";
            if (clean.Contains("5080")) return "nvidia-rtx-5080";
            if (clean.Contains("5070")) return "nvidia-rtx-5070";
            if (clean.Contains("4090")) return "nvidia-rtx-4090";
            if (clean.Contains("4080")) return "nvidia-rtx-4080";
            if (clean.Contains("4070")) return "nvidia-rtx-4070";
            if (clean.Contains("4060")) return "nvidia-rtx-4060";
            if (clean.Contains("4050")) return "nvidia-rtx-4050";
            if (clean.Contains("3090")) return "nvidia-rtx-3090";
            if (clean.Contains("3080")) return "nvidia-rtx-3080";
            if (clean.Contains("3070")) return "nvidia-rtx-3070";
            if (clean.Contains("3060")) return "nvidia-rtx-3060";
            if (clean.Contains("3050")) return "nvidia-rtx-3050";
            if (clean.Contains("2080")) return "nvidia-rtx-2080";
            if (clean.Contains("2070")) return "nvidia-rtx-2070";
            if (clean.Contains("2060")) return "nvidia-rtx-2060";
            if (clean.Contains("1660")) return "nvidia-gtx-1660";
            if (clean.Contains("1650")) return "nvidia-gtx-1650";
            return $"nvidia-gpu-{index}";
        }
        if (vendor == GpuVendor.Amd)
        {
            if (clean.Contains("890m")) return "amd-890m";
            if (clean.Contains("880m")) return "amd-880m";
            if (clean.Contains("780m")) return "amd-780m";
            if (clean.Contains("760m")) return "amd-760m";
            if (clean.Contains("680m")) return "amd-680m";
            if (clean.Contains("660m")) return "amd-660m";
            if (clean.Contains("7900")) return "amd-rx-7900";
            if (clean.Contains("7800")) return "amd-rx-7800";
            if (clean.Contains("7700")) return "amd-rx-7700";
            if (clean.Contains("7600")) return "amd-rx-7600";
            if (clean.Contains("6900") || clean.Contains("6800")) return "amd-rx-6800";
            if (clean.Contains("6700") || clean.Contains("6600")) return "amd-rx-6600";
            if (clean.Contains("radeon") && (clean.Contains("graphics") || clean.Contains("tm"))) return "amd-radeon-graphics";
            return $"amd-gpu-{index}";
        }
        if (vendor == GpuVendor.Intel)
        {
            if (clean.Contains("b580")) return "intel-arc-b580";
            if (clean.Contains("b570")) return "intel-arc-b570";
            if (clean.Contains("a770")) return "intel-arc-a770";
            if (clean.Contains("a750")) return "intel-arc-a750";
            if (clean.Contains("a580")) return "intel-arc-a580";
            if (clean.Contains("a380")) return "intel-arc-a380";
            if (clean.Contains("arc")) return "intel-arc";
            if (clean.Contains("iris")) return "intel-iris";
            if (clean.Contains("uhd")) return "intel-uhd";
            if (clean.Contains("xe")) return "intel-xe";
            return $"intel-gpu-{index}";
        }
        return $"gpu-{index}";
    }

    private static ulong GetPhysicalMemoryBytes()
    {
#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            var memStatus = new MEMORYSTATUSEX();
            memStatus.dwLength = (uint)Marshal.SizeOf<MEMORYSTATUSEX>();
            if (GlobalMemoryStatusEx(ref memStatus))
            {
                return memStatus.ullTotalPhys;
            }
        }
#endif
        return (ulong)GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
    }

    // =========================================================================
    // Zero-Dependency NPU Discovery (Windows cfgmgr32 / Linux /sys/class/accel)
    // =========================================================================

#if !ANDROID
    private const uint CR_SUCCESS = 0;
    private const uint CM_GETIDLIST_FILTER_CLASS = 0x00000200;
    private const uint CM_GETIDLIST_FILTER_PRESENT = 0x00000100;
    private const string GUID_DEVCLASS_COMPUTEACCELERATOR = "{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}";
#endif

    private static void EnumerateNpuAccelerators(List<DeviceInfo> results)
    {
        ulong totalRam = GetPhysicalMemoryBytes();

#if !ANDROID
        if (OperatingSystem.IsWindows())
        {
            uint flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
            if (CM_Get_Device_ID_List_SizeW(out uint length, GUID_DEVCLASS_COMPUTEACCELERATOR, flags) == CR_SUCCESS && length > 1)
            {
                char[] buffer = new char[length];
                if (CM_Get_Device_ID_ListW(GUID_DEVCLASS_COMPUTEACCELERATOR, buffer, length, flags) == CR_SUCCESS)
                {
                    int start = 0;
                    while (start < buffer.Length)
                    {
                        int end = Array.IndexOf(buffer, '\0', start);
                        if (end <= start) break;

                        string instanceId = new string(buffer, start, end - start);
                        start = end + 1;

                        var npu = ParseNpuHardwareId(instanceId, totalRam, results.Count);
                        if (npu != null && !results.Exists(d => d.Id.Equals(npu.Id, StringComparison.OrdinalIgnoreCase)))
                        {
                            results.Add(npu);
                        }
                    }
                }
            }
        }
        else
#endif
        if (OperatingSystem.IsLinux())
        {
            const string accelRoot = "/sys/class/accel";
            if (!Directory.Exists(accelRoot))
                return;

            foreach (string accelDir in Directory.EnumerateDirectories(accelRoot, "accel*"))
            {
                try
                {
                    string vendorPath = Path.Combine(accelDir, "device", "vendor");
                    string devicePath = Path.Combine(accelDir, "device", "device");
                    if (!File.Exists(vendorPath) || !File.Exists(devicePath))
                        continue;

                    string vendorHex = File.ReadAllText(vendorPath).Trim().Replace("0x", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant();
                    string deviceHex = File.ReadAllText(devicePath).Trim().Replace("0x", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant();
                    string syntheticId = $"PCI\\VEN_{vendorHex}&DEV_{deviceHex}";

                    var npu = ParseNpuHardwareId(syntheticId, totalRam, results.Count);
                    if (npu != null && !results.Exists(d => d.Id.Equals(npu.Id, StringComparison.OrdinalIgnoreCase)))
                    {
                        results.Add(npu);
                    }
                }
                catch
                {
                    // Skip unreadable sysfs entries
                }
            }
        }
    }

    /// <summary>
    /// Parses a Windows PnP instance ID or Linux sysfs PCI vendor/device tuple into a <see cref="DeviceInfo"/>
    /// for supported Neural Processing Units (AMD XDNA 2 / XDNA 1, Intel AI Boost).
    /// </summary>
    internal static DeviceInfo? ParseNpuHardwareId(string? hardwareOrInstanceId, ulong totalPhysicalRamBytes, int index)
    {
        if (string.IsNullOrWhiteSpace(hardwareOrInstanceId))
            return null;

        string upper = hardwareOrInstanceId.ToUpperInvariant();
        // Windows WDDM/MCDM caps NPU shared system memory at ~50% of total physical RAM
        ulong npuSharedCapBytes = totalPhysicalRamBytes / 2;
        double capGb = npuSharedCapBytes / (1024.0 * 1024.0 * 1024.0);

        if (upper.Contains("VEN_1022", StringComparison.Ordinal))
        {
            if (upper.Contains("DEV_17F0", StringComparison.Ordinal))
            {
                string capacityHint = capGb >= 28.0
                    ? $"~{capGb:F1} GB NPU cap supports up to 35B sparse MoE (e.g. Qwen3.6-35B-A3B Q4NX), 1B-14B LLMs, embeddings, and Whisper."
                    : $"~{capGb:F1} GB NPU cap (50% of RAM) supports 1B-14B LLMs, EmbeddingGemma, and Whisper; 35B MoE full-context requires >=64 GB RAM.";

                return new DeviceInfo
                {
                    Id = "amd-xdna2",
                    Index = index,
                    Name = "AMD XDNA 2 NPU (Ryzen AI 50 TOPS)",
                    Vendor = GpuVendor.Amd,
                    DedicatedVramBytes = 0,
                    SharedVramBytes = npuSharedCapBytes,
                    IsDisplayDevice = false,
                    IsNpu = true,
                    SupportedEngines = [InferenceEngineType.NpuSidecar],
                    RecommendedEngine = InferenceEngineType.NpuSidecar,
                    SafetyNotes = $"AMD XDNA 2 NPU (PCI 1022:17F0) via FastFlowLM (flm serve) or Lemonade Server OpenAI sidecar. {capacityHint}"
                };
            }

            if (upper.Contains("DEV_1502", StringComparison.Ordinal))
            {
                return new DeviceInfo
                {
                    Id = "amd-xdna1",
                    Index = index,
                    Name = "AMD XDNA 1 NPU (Ryzen AI)",
                    Vendor = GpuVendor.Amd,
                    DedicatedVramBytes = 0,
                    SharedVramBytes = npuSharedCapBytes,
                    IsDisplayDevice = false,
                    IsNpu = true,
                    SupportedEngines = [InferenceEngineType.NpuSidecar],
                    RecommendedEngine = InferenceEngineType.NpuSidecar,
                    SafetyNotes = $"AMD XDNA 1 NPU (PCI 1022:1502, ~{capGb:F1} GB shared cap). Note: FastFlowLM Q4NX requires XDNA 2; use Lemonade/ONNX sidecar for XDNA 1."
                };
            }
        }

        if (upper.Contains("VEN_8086", StringComparison.Ordinal) &&
            (upper.Contains("DEV_7D1D", StringComparison.Ordinal) ||
             upper.Contains("DEV_AD1D", StringComparison.Ordinal) ||
             upper.Contains("DEV_643E", StringComparison.Ordinal) ||
             upper.Contains("DEV_B03E", StringComparison.Ordinal)))
        {
            return new DeviceInfo
            {
                Id = "intel-npu",
                Index = index,
                Name = "Intel AI Boost NPU",
                Vendor = GpuVendor.Intel,
                DedicatedVramBytes = 0,
                SharedVramBytes = npuSharedCapBytes,
                IsDisplayDevice = false,
                IsNpu = true,
                SupportedEngines = [InferenceEngineType.NpuSidecar],
                RecommendedEngine = InferenceEngineType.NpuSidecar,
                SafetyNotes = $"Intel AI Boost NPU (~{capGb:F1} GB shared cap) via OpenAI-compatible local NPU sidecar."
            };
        }

        return null;
    }

#if !ANDROID
    // =========================================================================
    // DXGI & Win32 Interop
    // =========================================================================

    [DllImport("cfgmgr32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern uint CM_Get_Device_ID_List_SizeW(out uint pulLen, string pszFilter, uint ulFlags);

    [DllImport("cfgmgr32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern uint CM_Get_Device_ID_ListW(string pszFilter, [Out] char[] buffer, uint bufferLen, uint ulFlags);

    [DllImport("dxgi.dll", ExactSpelling = true)]
    private static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr ppFactory);

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

    [ComImport]
    [Guid("770aae78-f26f-4dba-a829-253c83d1b387")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIFactory1
    {
        int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        int SetPrivateDataInterface(ref Guid Name, [MarshalAs(UnmanagedType.IUnknown)] object pUnknown);
        int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        int GetParent(ref Guid riid, out IntPtr ppParent);
        int EnumAdapters(uint Adapter, out IntPtr ppAdapter);
        int MakeWindowAssociation(IntPtr WindowHandle, uint Flags);
        int GetWindowAssociation(out IntPtr pWindowHandle);
        int CreateSwapChain(IntPtr pDevice, IntPtr pDesc, out IntPtr ppSwapChain);
        int CreateSoftwareAdapter(IntPtr Module, out IntPtr ppAdapter);

        [PreserveSig]
        int EnumAdapters1(uint Adapter, out IDXGIAdapter1 ppAdapter);
        [PreserveSig]
        int IsCurrent();
    }

    [ComImport]
    [Guid("29038f61-3839-4626-91fd-086879011a05")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IDXGIAdapter1
    {
        int SetPrivateData(ref Guid Name, uint DataSize, IntPtr pData);
        int SetPrivateDataInterface(ref Guid Name, [MarshalAs(UnmanagedType.IUnknown)] object pUnknown);
        int GetPrivateData(ref Guid Name, ref uint pDataSize, IntPtr pData);
        int GetParent(ref Guid riid, out IntPtr ppParent);

        [PreserveSig]
        int EnumOutputs(uint Output, out IntPtr ppOutput);
        [PreserveSig]
        int GetDesc(IntPtr pDesc);
        [PreserveSig]
        int CheckInterfaceSupport(ref Guid InterfaceName, out long pUMDVersion);

        [PreserveSig]
        int GetDesc1(out DXGI_ADAPTER_DESC1 pDesc);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct DXGI_ADAPTER_DESC1
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string Description;
        public uint VendorId;
        public uint DeviceId;
        public uint SubSysId;
        public uint Revision;
        public nuint DedicatedVideoMemory;
        public nuint DedicatedSystemMemory;
        public nuint SharedSystemMemory;
        public long AdapterLuid;
        public uint Flags;
    }
#endif
}
