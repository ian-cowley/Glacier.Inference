namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Universal Vulkan 1.3+ compute context.
/// Manages instance creation, physical device selection, compute queue creation, and memory management.
/// Designed for zero-dependency cross-vendor execution on AMD, Intel, and NVIDIA across Windows and Linux.
/// </summary>
public sealed unsafe class VulkanContext : IDisposable
{
    private IntPtr _instance;
    private IntPtr _physicalDevice;
    private IntPtr _device;
    private IntPtr _computeQueue;
    private uint _computeQueueIndex;
    private IntPtr _commandPool;
    private IntPtr _commandBuffer;
    private VulkanDriver.VkPhysicalDeviceMemoryProperties _memProps;
    private bool _disposed;

    public string DeviceName { get; }
    public ulong TotalVramBytes { get; }
    public int DeviceOrdinal { get; }
    public bool HasCooperativeMatrix { get; }
    /// <summary>
    /// True when the Vulkan 1.3 <c>shaderIntegerDotProduct</c> feature (<c>VK_KHR_shader_integer_dot_product</c>,
    /// mapping to 1-cycle RDNA 3.5 <c>v_dot4_i32_i8</c> and NVIDIA <c>IDP4A</c>) is supported and enabled.
    /// </summary>
    public bool SupportsIntegerDotProduct { get; }
    /// <summary>
    /// True when compute shaders support Vulkan 1.1 subgroup shuffle operations (<c>GL_KHR_shader_subgroup_shuffle</c>)
    /// with a power-of-two subgroup size up to 128.
    /// </summary>
    public bool SupportsSubgroupShuffle { get; }
    public uint SubgroupSize { get; }
    /// <summary>
    /// True on discrete GPUs where pure DEVICE_LOCAL VRAM (e.g. 8-24 GB GDDR6) is much larger than
    /// the host-visible DEVICE_LOCAL BAR (typically 256 MB). False on unified-memory APUs (e.g. Strix Point).
    /// </summary>
    public bool PreferDeviceLocalStaging { get; }
    public IntPtr DeviceHandle => _device;
    public IntPtr InstanceHandle => _instance;
    public IntPtr ComputeQueueHandle => _computeQueue;
    public IntPtr CommandBufferHandle => _commandBuffer;
    public IntPtr CommandPoolHandle => _commandPool;

    private static readonly Lazy<bool> _isSupported = new(() =>
    {
        try
        {
            if (!VulkanDriver.IsAvailable()) return false;
            var appInfo = new VulkanDriver.VkApplicationInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                apiVersion = VulkanDriver.MakeVersion(1, 1, 0)
            };
            var createInfo = new VulkanDriver.VkInstanceCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo = new IntPtr(&appInfo)
            };
            int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out IntPtr inst);
            if (res != 0 || inst == IntPtr.Zero) return false;

            try
            {
                uint devCount = 0;
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, null) != 0 || devCount == 0)
                    return false;

                var devices = new IntPtr[devCount];
                if (VulkanDriver.EnumeratePhysicalDevices(inst, ref devCount, devices) != 0)
                    return false;

                for (uint i = 0; i < devCount; i++)
                {
                    VulkanDriver.GetPhysicalDeviceProperties(devices[i], out var props);
                    if (!IsSoftwareDevice(in props))
                    {
                        return true;
                    }
                }
                return false;
            }
            finally
            {
                VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
            }
        }
        catch
        {
            return false;
        }
    });

    public static bool IsSupported => _isSupported.Value;

    private static bool IsSoftwareDevice(in VulkanDriver.VkPhysicalDeviceProperties props)
    {
        if (props.deviceType == VulkanDriver.VK_PHYSICAL_DEVICE_TYPE_CPU) return true;
        string name = props.deviceName;
        return string.IsNullOrWhiteSpace(name) ||
               name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase) ||
               name.Contains("lavapipe", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Enumerates all available Vulkan physical devices, their device names, and accessible memory heaps.
    /// Works uniformly across Windows, Linux, and Android.
    /// </summary>
    public static List<(int Index, string Name, ulong TotalMemoryBytes)> EnumerateDevices()
    {
        var list = new List<(int Index, string Name, ulong TotalMemoryBytes)>();
        if (!VulkanDriver.IsAvailable()) return list;

        try
        {
            var appInfo = new VulkanDriver.VkApplicationInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
                apiVersion = VulkanDriver.MakeVersion(1, 1, 0)
            };
            var createInfo = new VulkanDriver.VkInstanceCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo = new IntPtr(&appInfo)
            };
            int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out IntPtr instance);
            if (res != 0 || instance == IntPtr.Zero) return list;

            try
            {
                uint devCount = 0;
                VulkanDriver.EnumeratePhysicalDevices(instance, ref devCount, null);
                if (devCount == 0) return list;

                var devices = new IntPtr[devCount];
                VulkanDriver.EnumeratePhysicalDevices(instance, ref devCount, devices);

                int hwOrdinal = 0;
                for (uint i = 0; i < devCount; i++)
                {
                    VulkanDriver.GetPhysicalDeviceProperties(devices[i], out var props);
                    if (IsSoftwareDevice(in props))
                    {
                        continue;
                    }
                    VulkanDriver.GetPhysicalDeviceMemoryProperties(devices[i], out var memProps);
                    ulong totalMem = 0;
                    var heaps = (VulkanDriver.VkMemoryHeap*)memProps.memoryHeaps;
                    for (uint h = 0; h < memProps.memoryHeapCount; h++)
                    {
                        totalMem += heaps[h].size;
                    }
                    list.Add((hwOrdinal++, props.deviceName, totalMem));
                }
            }
            finally
            {
                VulkanDriver.DestroyInstance(instance, IntPtr.Zero);
            }
        }
        catch
        {
            // Defensive: return empty list on environments where Vulkan ICD loader fails
        }
        return list;
    }

    public VulkanContext(int deviceOrdinal = 0)
    {
        if (!VulkanDriver.IsAvailable())
            throw new PlatformNotSupportedException("Vulkan driver (vulkan-1.dll / libvulkan.so) is not installed.");

        // 1. Create Vulkan Instance (prefer Vulkan 1.3 for integer dot product; fall back cleanly to 1.1)
        uint requestedApiVersion = VulkanDriver.MakeVersion(1, 3, 0);
        var appInfo = new VulkanDriver.VkApplicationInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName = Marshal.StringToHGlobalAnsi("Glacier.Inference"),
            applicationVersion = VulkanDriver.MakeVersion(1, 0, 0),
            pEngineName = Marshal.StringToHGlobalAnsi("GlacierEngine"),
            engineVersion = VulkanDriver.MakeVersion(1, 0, 0),
            apiVersion = requestedApiVersion
        };

        var createInfo = new VulkanDriver.VkInstanceCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo = new IntPtr(&appInfo)
        };

        int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out _instance);
        if (res != 0 || _instance == IntPtr.Zero)
        {
            requestedApiVersion = VulkanDriver.MakeVersion(1, 1, 0);
            appInfo.apiVersion = requestedApiVersion;
            res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out _instance);
        }
        Marshal.FreeHGlobal(appInfo.pApplicationName);
        Marshal.FreeHGlobal(appInfo.pEngineName);

        if (res != 0 || _instance == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create Vulkan instance: code {res}");

        try
        {
            // 2. Enumerate Physical Devices
            uint devCount = 0;
            VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, null);
            if (devCount == 0)
                throw new InvalidOperationException("No Vulkan physical devices found on this system.");

            var devices = new IntPtr[devCount];
            VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, devices);

            var hwDevices = new List<IntPtr>();
            for (int i = 0; i < devCount; i++)
            {
                VulkanDriver.GetPhysicalDeviceProperties(devices[i], out var candidateProps);
                if (!IsSoftwareDevice(in candidateProps))
                {
                    hwDevices.Add(devices[i]);
                }
            }

            if (hwDevices.Count == 0)
            {
                throw new PlatformNotSupportedException("No hardware Vulkan GPU physical devices found (software rasterizers skipped).");
            }
            if ((uint)deviceOrdinal >= (uint)hwDevices.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(deviceOrdinal), deviceOrdinal,
                    $"Vulkan hardware device ordinal must be between 0 and {hwDevices.Count - 1}.");
            }
            DeviceOrdinal = deviceOrdinal;
            _physicalDevice = hwDevices[DeviceOrdinal];

        VulkanDriver.GetPhysicalDeviceProperties(_physicalDevice, out var props);
        DeviceName = props.deviceName;

        // Query Vulkan 1.1 subgroup shuffle capabilities and Vulkan 1.3 integer dot product support
        uint vk11 = VulkanDriver.MakeVersion(1, 1, 0);
        uint vk13 = VulkanDriver.MakeVersion(1, 3, 0);
        if (props.apiVersion >= vk11)
        {
            try
            {
                var sgProps = new VulkanDriver.VkPhysicalDeviceSubgroupProperties
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                    pNext = IntPtr.Zero
                };
                var props2 = new VulkanDriver.VkPhysicalDeviceProperties2
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    pNext = new IntPtr(&sgProps)
                };
                VulkanDriver.GetPhysicalDeviceProperties2(_physicalDevice, ref props2);
                SubgroupSize = sgProps.subgroupSize;
                const uint requiredSgOps = VulkanDriver.VK_SUBGROUP_FEATURE_BASIC_BIT | VulkanDriver.VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
                SupportsSubgroupShuffle =
                    (sgProps.supportedStages & VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                    (sgProps.supportedOperations & requiredSgOps) == requiredSgOps &&
                    sgProps.subgroupSize >= 1 &&
                    sgProps.subgroupSize <= 128 &&
                    (sgProps.subgroupSize & (sgProps.subgroupSize - 1)) == 0;
            }
            catch
            {
                SupportsSubgroupShuffle = false;
            }
        }

        bool supportsIntDot = false;
        if (requestedApiVersion >= vk13 && props.apiVersion >= vk13)
        {
            try
            {
                var dotFeat = new VulkanDriver.VkPhysicalDeviceShaderIntegerDotProductFeatures
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
                    pNext = IntPtr.Zero,
                    shaderIntegerDotProduct = 0
                };
                var feat2 = new VulkanDriver.VkPhysicalDeviceFeatures2
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                    pNext = new IntPtr(&dotFeat)
                };
                VulkanDriver.GetPhysicalDeviceFeatures2(_physicalDevice, ref feat2);
                supportsIntDot = dotFeat.shaderIntegerDotProduct != 0;
            }
            catch
            {
                supportsIntDot = false;
            }
        }

        VulkanDriver.GetPhysicalDeviceMemoryProperties(_physicalDevice, out _memProps);
        ulong totalMem = 0;
        ulong maxDeviceLocalHeap = 0;
        ulong maxDeviceLocalHostVisibleHeap = 0;
        fixed (byte* pHeaps = _memProps.memoryHeaps)
        fixed (byte* pTypes = _memProps.memoryTypes)
        {
            var heaps = (VulkanDriver.VkMemoryHeap*)pHeaps;
            var types = (VulkanDriver.VkMemoryType*)pTypes;
            for (uint h = 0; h < _memProps.memoryHeapCount; h++)
            {
                totalMem += heaps[h].size;
            }
            for (uint t = 0; t < _memProps.memoryTypeCount; t++)
            {
                uint flags = types[t].propertyFlags;
                uint hi = types[t].heapIndex;
                if (hi >= _memProps.memoryHeapCount) continue;
                ulong hSize = heaps[hi].size;
                if ((flags & VulkanDriver.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
                {
                    if (hSize > maxDeviceLocalHeap) maxDeviceLocalHeap = hSize;
                    if ((flags & VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && hSize > maxDeviceLocalHostVisibleHeap)
                        maxDeviceLocalHostVisibleHeap = hSize;
                }
            }
        }
        TotalVramBytes = totalMem;
        PreferDeviceLocalStaging = maxDeviceLocalHeap > 0 && maxDeviceLocalHeap > maxDeviceLocalHostVisibleHeap * 2;

        // 3. Find Compute Queue Family
        uint qfCount = 0;
        VulkanDriver.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref qfCount, null);
        var qfProps = new VulkanDriver.VkQueueFamilyProperties[qfCount];
        VulkanDriver.GetPhysicalDeviceQueueFamilyProperties(_physicalDevice, ref qfCount, qfProps);

        int computeIndex = -1;
        for (uint i = 0; i < qfCount; i++)
        {
            if ((qfProps[i].queueFlags & VulkanDriver.VK_QUEUE_COMPUTE_BIT) != 0)
            {
                computeIndex = (int)i;
                break;
            }
        }

        if (computeIndex < 0)
            throw new InvalidOperationException($"No compute queue family found on {DeviceName}");

        _computeQueueIndex = (uint)computeIndex;

        // 4. Create Logical Device with Compute Queue
        float priority = 1.0f;
        var queueInfo = new VulkanDriver.VkDeviceQueueCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = _computeQueueIndex,
            queueCount = 1,
            pQueuePriorities = new IntPtr(&priority)
        };

        var enableDotFeat = new VulkanDriver.VkPhysicalDeviceShaderIntegerDotProductFeatures
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
            pNext = IntPtr.Zero,
            shaderIntegerDotProduct = 1
        };

        var devInfo = new VulkanDriver.VkDeviceCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext = supportsIntDot ? new IntPtr(&enableDotFeat) : IntPtr.Zero,
            queueCreateInfoCount = 1,
            pQueueCreateInfos = new IntPtr(&queueInfo)
        };

        int devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
        if ((devRes != 0 || _device == IntPtr.Zero) && supportsIntDot)
        {
            supportsIntDot = false;
            devInfo.pNext = IntPtr.Zero;
            devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
        }
        if (devRes != 0 || _device == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create Vulkan logical device on {DeviceName}: code {devRes}");

        SupportsIntegerDotProduct = supportsIntDot;

        VulkanDriver.GetDeviceQueue(_device, _computeQueueIndex, 0, out _computeQueue);

        // 5. Create Compute Command Pool
        var poolInfo = new VulkanDriver.VkCommandPoolCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex = _computeQueueIndex,
            flags = 2 // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        };

        int poolRes = VulkanDriver.CreateCommandPool(_device, ref poolInfo, IntPtr.Zero, out _commandPool);
        if (poolRes != 0)
            throw new InvalidOperationException($"Failed to create Vulkan command pool: code {poolRes}");

        // 6. Allocate Primary Command Buffer
        var allocInfo = new VulkanDriver.VkCommandBufferAllocateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = _commandPool,
            level = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
            commandBufferCount = 1
        };

        var cmdBufs = new IntPtr[1];
        int cmdRes = VulkanDriver.AllocateCommandBuffers(_device, ref allocInfo, cmdBufs);
        if (cmdRes != 0)
            throw new InvalidOperationException($"Failed to allocate Vulkan command buffer: code {cmdRes}");

            _commandBuffer = cmdBufs[0];
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal static int ResolveHardwareDeviceOrdinal(ReadOnlySpan<(bool IsSoftware, string Name)> devices, int requestedOrdinal)
    {
        if (requestedOrdinal < 0) return -1;
        int hwIndex = 0;
        for (int i = 0; i < devices.Length; i++)
        {
            if (devices[i].IsSoftware) continue;
            if (hwIndex == requestedOrdinal) return i;
            hwIndex++;
        }
        return -1;
    }

    public bool TryFindMemoryType(uint typeFilter, uint properties, uint avoidProperties, out uint memTypeIndex)
    {
        fixed (byte* pTypes = _memProps.memoryTypes)
        {
            var types = (VulkanDriver.VkMemoryType*)pTypes;
            for (uint i = 0; i < _memProps.memoryTypeCount; i++)
            {
                if ((typeFilter & (1u << (int)i)) != 0)
                {
                    uint flags = types[i].propertyFlags;
                    if ((flags & properties) == properties && (flags & avoidProperties) == 0)
                    {
                        memTypeIndex = i;
                        return true;
                    }
                }
            }
        }
        memTypeIndex = 0;
        return false;
    }

    public uint FindMemoryType(uint typeFilter, uint properties)
    {
        if (TryFindMemoryType(typeFilter, properties, 0u, out uint idx))
        {
            return idx;
        }
        throw new InvalidOperationException($"Failed to find suitable memory type for flags {properties}");
    }

    public void CreateBuffer(ulong size, uint usage, uint memProperties, out IntPtr buffer, out IntPtr memory)
        => CreateBufferPreferred(size, usage, out buffer, out memory, (memProperties, 0u));

    public void CreateBufferPreferred(
        ulong size,
        uint usage,
        out IntPtr buffer,
        out IntPtr memory,
        params ReadOnlySpan<(uint Required, uint Avoid)> preferences)
    {
        var bufInfo = new VulkanDriver.VkBufferCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size = size,
            usage = usage,
            sharingMode = 0 // VK_SHARING_MODE_EXCLUSIVE
        };

        int res = VulkanDriver.CreateBuffer(_device, ref bufInfo, IntPtr.Zero, out buffer);
        if (res != 0)
            throw new InvalidOperationException($"vkCreateBuffer failed for {size} bytes: code {res}");

        VulkanDriver.GetBufferMemoryRequirements(_device, buffer, out var memReqs);

        memory = IntPtr.Zero;
        int lastAllocRes = -1;
        bool matchedAny = false;
        uint testedMask = 0u;

        foreach (var (req, avoid) in preferences)
        {
            if (!TryFindMemoryType(memReqs.memoryTypeBits, req, avoid, out uint memTypeIndex))
            {
                continue;
            }

            uint bit = 1u << (int)memTypeIndex;
            if ((testedMask & bit) != 0u)
            {
                continue;
            }
            testedMask |= bit;
            matchedAny = true;

            var allocInfo = new VulkanDriver.VkMemoryAllocateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = memReqs.size,
                memoryTypeIndex = memTypeIndex
            };

            lastAllocRes = VulkanDriver.AllocateMemory(_device, ref allocInfo, IntPtr.Zero, out memory);
            if (lastAllocRes == 0)
            {
                break;
            }
            memory = IntPtr.Zero;
        }

        if (memory == IntPtr.Zero)
        {
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
            if (!matchedAny)
            {
                throw new InvalidOperationException($"Failed to find suitable memory type for buffer ({size} bytes).");
            }
            throw new InvalidOperationException($"vkAllocateMemory failed for {memReqs.size} bytes: code {lastAllocRes}");
        }

        int bindRes = VulkanDriver.BindBufferMemory(_device, buffer, memory, 0);
        if (bindRes != 0)
        {
            VulkanDriver.FreeMemory(_device, memory, IntPtr.Zero);
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
            throw new InvalidOperationException($"vkBindBufferMemory failed: code {bindRes}");
        }
    }

    public void DestroyBuffer(IntPtr buffer, IntPtr memory)
    {
        if (buffer != IntPtr.Zero)
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
        if (memory != IntPtr.Zero)
            VulkanDriver.FreeMemory(_device, memory, IntPtr.Zero);
    }

    public void CopyToBuffer(IntPtr memory, IntPtr srcHost, ulong bytes)
    {
        int mapRes = VulkanDriver.MapMemory(_device, memory, 0, bytes, 0, out IntPtr mapped);
        if (mapRes != 0)
            throw new InvalidOperationException($"vkMapMemory failed: code {mapRes}");

        Buffer.MemoryCopy((void*)srcHost, (void*)mapped, bytes, bytes);
        VulkanDriver.UnmapMemory(_device, memory);
    }

    public void CopyFromBuffer(IntPtr dstHost, IntPtr memory, ulong bytes)
    {
        int mapRes = VulkanDriver.MapMemory(_device, memory, 0, bytes, 0, out IntPtr mapped);
        if (mapRes != 0)
            throw new InvalidOperationException($"vkMapMemory failed: code {mapRes}");

        Buffer.MemoryCopy((void*)mapped, (void*)dstHost, bytes, bytes);
        VulkanDriver.UnmapMemory(_device, memory);
    }

    public void Synchronize()
    {
        if (_device != IntPtr.Zero)
        {
            VulkanDriver.DeviceWaitIdle(_device);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_device != IntPtr.Zero)
            {
                VulkanDriver.DeviceWaitIdle(_device);
                if (_commandPool != IntPtr.Zero)
                {
                    VulkanDriver.DestroyCommandPool(_device, _commandPool, IntPtr.Zero);
                    _commandPool = IntPtr.Zero;
                }
                VulkanDriver.DestroyDevice(_device, IntPtr.Zero);
                _device = IntPtr.Zero;
            }

            if (_instance != IntPtr.Zero)
            {
                VulkanDriver.DestroyInstance(_instance, IntPtr.Zero);
                _instance = IntPtr.Zero;
            }

            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~VulkanContext()
    {
        Dispose();
    }
}
