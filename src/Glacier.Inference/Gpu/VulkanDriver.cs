namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native Vulkan driver (vulkan-1.dll on Windows, libvulkan.so.1 on Linux).
/// Enables universal cross-vendor compute acceleration and checks for VK_KHR_cooperative_matrix hardware tensor support.
/// </summary>
public static unsafe class VulkanDriver
{
    private const string VulkanLib = "vulkan-1.dll";

    public const int VK_SUCCESS = 0;
    public const int VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
    public const int VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
    public const int VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
    public const int VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
    public const int VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
    public const int VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
    public const int VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
    public const int VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
    public const int VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
    public const int VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
    public const int VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
    public const int VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
    public const int VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
    public const int VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
    public const int VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 = 1000059000;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506000;

    public const uint VK_QUEUE_COMPUTE_BIT = 0x00000002;
    public const uint VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
    public const uint VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
    public const uint VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    public const uint VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;

    public const uint VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    public const int VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
    public const int VK_PIPELINE_BIND_POINT_COMPUTE = 1;
    public const uint VK_ACCESS_SHADER_READ_BIT = 0x00000020;
    public const uint VK_ACCESS_SHADER_WRITE_BIT = 0x00000040;
    public const uint VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800;

    static VulkanDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.TryLoad("vulkan-1.dll", out IntPtr handle) && handle != IntPtr.Zero;
            }
            else
            {
                return (NativeLibrary.TryLoad("libvulkan.so", out IntPtr handle) ||
                        NativeLibrary.TryLoad("libvulkan.so.1", out handle)) && handle != IntPtr.Zero;
            }
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [StructLayout(LayoutKind.Sequential)]
    public struct VkApplicationInfo
    {
        public int sType;
        public IntPtr pNext;
        public IntPtr pApplicationName;
        public uint applicationVersion;
        public IntPtr pEngineName;
        public uint engineVersion;
        public uint apiVersion;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkInstanceCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public IntPtr pApplicationInfo;
        public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceProperties
    {
        public uint apiVersion;
        public uint driverVersion;
        public uint vendorID;
        public uint deviceID;
        public uint deviceType;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string deviceName;
        public fixed byte pipelineCacheUUID[16];
        public VkPhysicalDeviceLimits limits;
        public VkPhysicalDeviceSparseProperties sparseProperties;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceLimits
    {
        public uint maxImageDimension1D;
        public uint maxImageDimension2D;
        public uint maxImageDimension3D;
        public uint maxImageDimensionCube;
        public uint maxImageArrayLayers;
        public uint maxTexelBufferElements;
        public uint maxUniformBufferRange;
        public ulong maxStorageBufferRange;
        public ulong maxPushConstantsSize;
        public ulong maxMemoryAllocationCount;
        public ulong maxSamplerAllocationCount;
        public ulong bufferImageGranularity;
        public ulong sparseAddressSpaceSize;
        public uint maxBoundDescriptorSets;
        public uint maxPerStageDescriptorSamplers;
        public uint maxPerStageDescriptorUniformBuffers;
        public uint maxPerStageDescriptorStorageBuffers;
        public uint maxPerStageDescriptorSampledImages;
        public uint maxPerStageDescriptorStorageImages;
        public uint maxPerStageDescriptorInputAttachments;
        public uint maxPerStageResources;
        public uint maxDescriptorSetSamplers;
        public uint maxDescriptorSetUniformBuffers;
        public uint maxDescriptorSetUniformBuffersDynamic;
        public uint maxDescriptorSetStorageBuffers;
        public uint maxDescriptorSetStorageBuffersDynamic;
        public uint maxDescriptorSetSampledImages;
        public uint maxDescriptorSetStorageImages;
        public uint maxDescriptorSetInputAttachments;
        public uint maxVertexInputAttributes;
        public uint maxVertexInputBindings;
        public uint maxVertexInputAttributeOffset;
        public uint maxVertexInputBindingStride;
        public uint maxVertexOutputComponents;
        public uint maxTessellationGenerationLevel;
        public uint maxTessellationPatchSize;
        public uint maxTessellationControlPerVertexInputComponents;
        public uint maxTessellationControlPerVertexOutputComponents;
        public uint maxTessellationControlPerPatchOutputComponents;
        public uint maxTessellationControlTotalOutputComponents;
        public uint maxTessellationEvaluationInputComponents;
        public uint maxTessellationEvaluationOutputComponents;
        public uint maxGeometryShaderInvocations;
        public uint maxGeometryInputComponents;
        public uint maxGeometryOutputComponents;
        public uint maxGeometryOutputVertices;
        public uint maxGeometryTotalOutputComponents;
        public uint maxFragmentInputComponents;
        public uint maxFragmentOutputAttachments;
        public uint maxFragmentDualSrcAttachments;
        public uint maxFragmentCombinedOutputResources;
        public uint maxComputeSharedMemorySize;
        public fixed uint maxComputeWorkGroupCount[3];
        public uint maxComputeWorkGroupInvocations;
        public fixed uint maxComputeWorkGroupSize[3];
        public uint subPixelPrecisionBits;
        public uint subTexelPrecisionBits;
        public uint mipmapPrecisionBits;
        public uint maxDrawIndexedIndexValue;
        public uint maxDrawIndirectCount;
        public float maxSamplerLodBias;
        public float maxSamplerAnisotropy;
        public uint maxViewports;
        public fixed uint maxViewportDimensions[2];
        public fixed float viewportBoundsRange[2];
        public uint viewportSubPixelBits;
        public nuint minMemoryMapAlignment;
        public ulong minTexelBufferOffsetAlignment;
        public ulong minUniformBufferOffsetAlignment;
        public ulong minStorageBufferOffsetAlignment;
        public int minTexelOffset;
        public uint maxTexelOffset;
        public int minTexelGatherOffset;
        public uint maxTexelGatherOffset;
        public float minInterpolationOffset;
        public float maxInterpolationOffset;
        public uint subPixelInterpolationOffsetBits;
        public uint maxFramebufferWidth;
        public uint maxFramebufferHeight;
        public uint maxFramebufferLayers;
        public uint framebufferColorSampleCounts;
        public uint framebufferDepthSampleCounts;
        public uint framebufferStencilSampleCounts;
        public uint framebufferNoAttachmentsSampleCounts;
        public uint maxColorAttachments;
        public uint sampledImageColorSampleCounts;
        public uint sampledImageIntegerSampleCounts;
        public uint sampledImageDepthSampleCounts;
        public uint sampledImageStencilSampleCounts;
        public uint storageImageSampleCounts;
        public uint maxSampleMaskWords;
        public uint timestampComputeAndGraphics;
        public float timestampPeriod;
        public uint maxClipDistances;
        public uint maxCullDistances;
        public uint maxCombinedClipAndCullDistances;
        public uint discreteQueuePriorities;
        public fixed float pointSizeRange[2];
        public fixed float lineWidthRange[2];
        public float pointSizeGranularity;
        public float lineWidthGranularity;
        public uint strictLines;
        public uint standardSampleLocations;
        public ulong optimalBufferCopyOffsetAlignment;
        public ulong optimalBufferCopyRowPitchAlignment;
        public ulong nonCoherentAtomSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSparseProperties
    {
        public uint residencyStandard2DBlockShape;
        public uint residencyStandard2DMultisampleBlockShape;
        public uint residencyStandard3DBlockShape;
        public uint residencyAlignedMipSize;
        public uint residencyNonResidentStrict;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkQueueFamilyProperties
    {
        public uint queueFlags;
        public uint queueCount;
        public uint timestampValidBits;
        public uint minImageTransferGranularityWidth;
        public uint minImageTransferGranularityHeight;
        public uint minImageTransferGranularityDepth;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryType
    {
        public uint propertyFlags;
        public uint heapIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryHeap
    {
        public ulong size;
        public uint flags;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceMemoryProperties
    {
        public uint memoryTypeCount;
        public fixed byte memoryTypes[256]; // 32 * sizeof(VkMemoryType)
        public uint memoryHeapCount;
        public fixed byte memoryHeaps[256]; // 16 * sizeof(VkMemoryHeap)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDeviceQueueCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint queueFamilyIndex;
        public uint queueCount;
        public IntPtr pQueuePriorities;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDeviceCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint queueCreateInfoCount;
        public IntPtr pQueueCreateInfos;
        public uint enabledLayerCount;
        public IntPtr ppEnabledLayerNames;
        public uint enabledExtensionCount;
        public IntPtr ppEnabledExtensionNames;
        public IntPtr pEnabledFeatures;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkBufferCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public ulong size;
        public uint usage;
        public int sharingMode;
        public uint queueFamilyIndexCount;
        public IntPtr pQueueFamilyIndices;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryRequirements
    {
        public ulong size;
        public ulong alignment;
        public uint memoryTypeBits;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryAllocateInfo
    {
        public int sType;
        public IntPtr pNext;
        public ulong allocationSize;
        public uint memoryTypeIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandPoolCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint queueFamilyIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandBufferAllocateInfo
    {
        public int sType;
        public IntPtr pNext;
        public IntPtr commandPool;
        public int level;
        public uint commandBufferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkCommandBufferBeginInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public IntPtr pInheritanceInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkSubmitInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint waitSemaphoreCount;
        public IntPtr pWaitSemaphores;
        public IntPtr pWaitDstStageMask;
        public uint commandBufferCount;
        public IntPtr pCommandBuffers;
        public uint signalSemaphoreCount;
        public IntPtr pSignalSemaphores;
    }

    [DllImport(VulkanLib, EntryPoint = "vkCreateInstance")]
    public static extern int CreateInstance(ref VkInstanceCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pInstance);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyInstance")]
    public static extern void DestroyInstance(IntPtr instance, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkEnumeratePhysicalDevices")]
    public static extern int EnumeratePhysicalDevices(IntPtr instance, ref uint pPhysicalDeviceCount, [Out] IntPtr[]? pPhysicalDevices);

    [DllImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceProperties")]
    public static extern void GetPhysicalDeviceProperties(IntPtr physicalDevice, out VkPhysicalDeviceProperties pProperties);

    [DllImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
    public static extern void GetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, [Out] VkQueueFamilyProperties[]? pQueueFamilyProperties);

    [DllImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
    public static extern void GetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, out VkPhysicalDeviceMemoryProperties pMemoryProperties);

    [DllImport(VulkanLib, EntryPoint = "vkCreateDevice")]
    public static extern int CreateDevice(IntPtr physicalDevice, ref VkDeviceCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pDevice);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyDevice")]
    public static extern void DestroyDevice(IntPtr device, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkGetDeviceQueue")]
    public static extern void GetDeviceQueue(IntPtr device, uint queueFamilyIndex, uint queueIndex, out IntPtr pQueue);

    [DllImport(VulkanLib, EntryPoint = "vkDeviceWaitIdle")]
    public static extern int DeviceWaitIdle(IntPtr device);

    [DllImport(VulkanLib, EntryPoint = "vkQueueWaitIdle")]
    public static extern int QueueWaitIdle(IntPtr queue);

    [DllImport(VulkanLib, EntryPoint = "vkCreateBuffer")]
    public static extern int CreateBuffer(IntPtr device, ref VkBufferCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pBuffer);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyBuffer")]
    public static extern void DestroyBuffer(IntPtr device, IntPtr buffer, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkGetBufferMemoryRequirements")]
    public static extern void GetBufferMemoryRequirements(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);

    [DllImport(VulkanLib, EntryPoint = "vkAllocateMemory")]
    public static extern int AllocateMemory(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, out IntPtr pMemory);

    [DllImport(VulkanLib, EntryPoint = "vkFreeMemory")]
    public static extern void FreeMemory(IntPtr device, IntPtr memory, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkBindBufferMemory")]
    public static extern int BindBufferMemory(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);

    [DllImport(VulkanLib, EntryPoint = "vkMapMemory")]
    public static extern int MapMemory(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, out IntPtr ppData);

    [DllImport(VulkanLib, EntryPoint = "vkUnmapMemory")]
    public static extern void UnmapMemory(IntPtr device, IntPtr memory);

    [DllImport(VulkanLib, EntryPoint = "vkCreateCommandPool")]
    public static extern int CreateCommandPool(IntPtr device, ref VkCommandPoolCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pCommandPool);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyCommandPool")]
    public static extern void DestroyCommandPool(IntPtr device, IntPtr commandPool, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkAllocateCommandBuffers")]
    public static extern int AllocateCommandBuffers(IntPtr device, ref VkCommandBufferAllocateInfo pAllocateInfo, [Out] IntPtr[] pCommandBuffers);

    [DllImport(VulkanLib, EntryPoint = "vkBeginCommandBuffer")]
    public static extern int BeginCommandBuffer(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);

    [DllImport(VulkanLib, EntryPoint = "vkEndCommandBuffer")]
    public static extern int EndCommandBuffer(IntPtr commandBuffer);

    [StructLayout(LayoutKind.Sequential)]
    public struct VkShaderModuleCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public nuint codeSize;
        public uint* pCode;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetLayoutBinding
    {
        public uint binding;
        public int descriptorType;
        public uint descriptorCount;
        public uint stageFlags;
        public IntPtr pImmutableSamplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetLayoutCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint bindingCount;
        public IntPtr pBindings;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPushConstantRange
    {
        public uint stageFlags;
        public uint offset;
        public uint size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPipelineLayoutCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint setLayoutCount;
        public IntPtr pSetLayouts;
        public uint pushConstantRangeCount;
        public IntPtr pPushConstantRanges;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPipelineShaderStageCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint stage;
        public IntPtr module;
        public IntPtr pName;
        public IntPtr pSpecializationInfo;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkComputePipelineCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public VkPipelineShaderStageCreateInfo stage;
        public IntPtr layout;
        public IntPtr basePipelineHandle;
        public int basePipelineIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorPoolSize
    {
        public int type;
        public uint descriptorCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorPoolCreateInfo
    {
        public int sType;
        public IntPtr pNext;
        public uint flags;
        public uint maxSets;
        public uint poolSizeCount;
        public IntPtr pPoolSizes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorSetAllocateInfo
    {
        public int sType;
        public IntPtr pNext;
        public IntPtr descriptorPool;
        public uint descriptorSetCount;
        public IntPtr pSetLayouts;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkDescriptorBufferInfo
    {
        public IntPtr buffer;
        public ulong offset;
        public ulong range;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkWriteDescriptorSet
    {
        public int sType;
        public IntPtr pNext;
        public IntPtr dstSet;
        public uint dstBinding;
        public uint dstArrayElement;
        public uint descriptorCount;
        public int descriptorType;
        public IntPtr pImageInfo;
        public IntPtr pBufferInfo;
        public IntPtr pTexelBufferView;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkMemoryBarrier
    {
        public int sType;
        public IntPtr pNext;
        public uint srcAccessMask;
        public uint dstAccessMask;
    }

    [DllImport(VulkanLib, EntryPoint = "vkCreateShaderModule")]
    public static extern int CreateShaderModule(IntPtr device, ref VkShaderModuleCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pShaderModule);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyShaderModule")]
    public static extern void DestroyShaderModule(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkCreateDescriptorSetLayout")]
    public static extern int CreateDescriptorSetLayout(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pSetLayout);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyDescriptorSetLayout")]
    public static extern void DestroyDescriptorSetLayout(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkCreatePipelineLayout")]
    public static extern int CreatePipelineLayout(IntPtr device, ref VkPipelineLayoutCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pPipelineLayout);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyPipelineLayout")]
    public static extern void DestroyPipelineLayout(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkCreateDescriptorPool")]
    public static extern int CreateDescriptorPool(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pDescriptorPool);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyDescriptorPool")]
    public static extern void DestroyDescriptorPool(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkAllocateDescriptorSets")]
    public static extern int AllocateDescriptorSets(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr* pDescriptorSets);

    [DllImport(VulkanLib, EntryPoint = "vkUpdateDescriptorSets")]
    public static extern void UpdateDescriptorSets(IntPtr device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

    [DllImport(VulkanLib, EntryPoint = "vkCreateComputePipelines")]
    public static extern int CreateComputePipelines(IntPtr device, IntPtr pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, IntPtr pAllocator, IntPtr* pPipelines);

    [DllImport(VulkanLib, EntryPoint = "vkDestroyPipeline")]
    public static extern void DestroyPipeline(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

    [DllImport(VulkanLib, EntryPoint = "vkCmdBindPipeline")]
    public static extern void CmdBindPipeline(IntPtr commandBuffer, int pipelineBindPoint, IntPtr pipeline);

    [DllImport(VulkanLib, EntryPoint = "vkCmdBindDescriptorSets")]
    public static extern void CmdBindDescriptorSets(IntPtr commandBuffer, int pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr* pDescriptorSets, uint dynamicOffsetCount, uint* pDynamicOffsets);

    [DllImport(VulkanLib, EntryPoint = "vkCmdPushConstants")]
    public static extern void CmdPushConstants(IntPtr commandBuffer, IntPtr layout, uint stageFlags, uint offset, uint size, void* pValues);

    [DllImport(VulkanLib, EntryPoint = "vkCmdDispatch")]
    public static extern void CmdDispatch(IntPtr commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [DllImport(VulkanLib, EntryPoint = "vkCmdPipelineBarrier")]
    public static extern void CmdPipelineBarrier(IntPtr commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags, uint memoryBarrierCount, VkMemoryBarrier* pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

    [DllImport(VulkanLib, EntryPoint = "vkResetCommandBuffer")]
    public static extern int ResetCommandBuffer(IntPtr commandBuffer, uint flags);

    [DllImport(VulkanLib, EntryPoint = "vkQueueSubmit")]
    public static extern int QueueSubmit(IntPtr queue, uint submitCount, ref VkSubmitInfo pSubmits, IntPtr fence);

    public static uint MakeVersion(uint major, uint minor, uint patch) => (major << 22) | (minor << 12) | patch;
    public static uint VersionMajor(uint version) => version >> 22;
    public static uint VersionMinor(uint version) => (version >> 12) & 0x3FF;
    public static uint VersionPatch(uint version) => version & 0xFFF;

    public static void Check(int res, string op)
    {
        if (res != VK_SUCCESS)
        {
            throw new InvalidOperationException($"Vulkan Error during '{op}': code {res}");
        }
    }
}
