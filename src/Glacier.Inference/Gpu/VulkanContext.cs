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
    public IntPtr DeviceHandle => _device;
    public IntPtr InstanceHandle => _instance;

    public static bool IsSupported
    {
        get
        {
            try
            {
                if (!VulkanDriver.IsAvailable()) return false;
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
                int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out IntPtr inst);
                if (res == 0 && inst != IntPtr.Zero)
                {
                    VulkanDriver.DestroyInstance(inst, IntPtr.Zero);
                    return true;
                }
                return false;
            }
            catch
            {
                return false;
            }
        }
    }

    public VulkanContext(int deviceOrdinal = 0)
    {
        if (!VulkanDriver.IsAvailable())
            throw new PlatformNotSupportedException("Vulkan driver (vulkan-1.dll / libvulkan.so.1) is not installed.");

        // 1. Create Vulkan Instance
        var appInfo = new VulkanDriver.VkApplicationInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName = Marshal.StringToHGlobalAnsi("Glacier.Inference"),
            applicationVersion = VulkanDriver.MakeVersion(1, 0, 0),
            pEngineName = Marshal.StringToHGlobalAnsi("GlacierEngine"),
            engineVersion = VulkanDriver.MakeVersion(1, 0, 0),
            apiVersion = VulkanDriver.MakeVersion(1, 2, 0)
        };

        var createInfo = new VulkanDriver.VkInstanceCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo = new IntPtr(&appInfo)
        };

        int res = VulkanDriver.CreateInstance(ref createInfo, IntPtr.Zero, out _instance);
        Marshal.FreeHGlobal(appInfo.pApplicationName);
        Marshal.FreeHGlobal(appInfo.pEngineName);

        if (res != 0 || _instance == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create Vulkan instance: code {res}");

        // 2. Enumerate Physical Devices
        uint devCount = 0;
        VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, null);
        if (devCount == 0)
            throw new InvalidOperationException("No Vulkan physical devices found on this system.");

        var devices = new IntPtr[devCount];
        VulkanDriver.EnumeratePhysicalDevices(_instance, ref devCount, devices);

        DeviceOrdinal = Math.Clamp(deviceOrdinal, 0, (int)devCount - 1);
        _physicalDevice = devices[DeviceOrdinal];

        VulkanDriver.GetPhysicalDeviceProperties(_physicalDevice, out var props);
        DeviceName = props.deviceName;

        VulkanDriver.GetPhysicalDeviceMemoryProperties(_physicalDevice, out _memProps);
        ulong totalMem = 0;
        fixed (byte* pHeaps = _memProps.memoryHeaps)
        {
            var heaps = (VulkanDriver.VkMemoryHeap*)pHeaps;
            for (uint h = 0; h < _memProps.memoryHeapCount; h++)
            {
                totalMem += heaps[h].size;
            }
        }
        TotalVramBytes = totalMem;

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

        var devInfo = new VulkanDriver.VkDeviceCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount = 1,
            pQueueCreateInfos = new IntPtr(&queueInfo)
        };

        int devRes = VulkanDriver.CreateDevice(_physicalDevice, ref devInfo, IntPtr.Zero, out _device);
        if (devRes != 0 || _device == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create Vulkan logical device on {DeviceName}: code {devRes}");

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

    public uint FindMemoryType(uint typeFilter, uint properties)
    {
        fixed (byte* pTypes = _memProps.memoryTypes)
        {
            var types = (VulkanDriver.VkMemoryType*)pTypes;
            for (uint i = 0; i < _memProps.memoryTypeCount; i++)
            {
                if ((typeFilter & (1u << (int)i)) != 0)
                {
                    if ((types[i].propertyFlags & properties) == properties)
                    {
                        return i;
                    }
                }
            }
        }
        throw new InvalidOperationException($"Failed to find suitable memory type for flags {properties}");
    }

    public void CreateBuffer(ulong size, uint usage, uint memProperties, out IntPtr buffer, out IntPtr memory)
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
        uint memTypeIndex = FindMemoryType(memReqs.memoryTypeBits, memProperties);

        var allocInfo = new VulkanDriver.VkMemoryAllocateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize = memReqs.size,
            memoryTypeIndex = memTypeIndex
        };

        int allocRes = VulkanDriver.AllocateMemory(_device, ref allocInfo, IntPtr.Zero, out memory);
        if (allocRes != 0)
        {
            VulkanDriver.DestroyBuffer(_device, buffer, IntPtr.Zero);
            throw new InvalidOperationException($"vkAllocateMemory failed for {memReqs.size} bytes: code {allocRes}");
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
