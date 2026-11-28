namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native Vulkan driver (vulkan-1.dll on Windows, libvulkan.so.1 on Linux).
/// Enables universal cross-vendor compute acceleration and checks for VK_KHR_cooperative_matrix hardware tensor support.
/// </summary>
public static unsafe partial class VulkanDriver
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
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES = 1000280000;
    public const int VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506000;

    public const uint VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001;
    public const uint VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010;

    public const uint VK_QUEUE_COMPUTE_BIT = 0x00000002;
    public const uint VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
    public const uint VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
    public const uint VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
    public const uint VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
    public const uint VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    public const uint VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;

    public const uint VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
    public const int VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    public const int VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
    public const uint VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD = 0x00000080;
    public const int VK_PIPELINE_BIND_POINT_COMPUTE = 1;
    public const uint VK_ACCESS_SHADER_READ_BIT = 0x00000020;
    public const uint VK_ACCESS_SHADER_WRITE_BIT = 0x00000040;
    public const uint VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000;
    public const uint VK_ACCESS_HOST_READ_BIT = 0x00002000;
    public const uint VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800;
    public const uint VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000;
    public const uint VK_PIPELINE_STAGE_HOST_BIT = 0x00004000;
    public const int VK_ERROR_OUT_OF_POOL_MEMORY = -1000069000;
    public const int VK_ERROR_FRAGMENTED_POOL = -12;
    public const ulong VK_WHOLE_SIZE = ~0UL;

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

    public const uint VK_PHYSICAL_DEVICE_TYPE_CPU = 4;

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceProperties
    {
        public uint apiVersion;
        public uint driverVersion;
        public uint vendorID;
        public uint deviceID;
        public uint deviceType;
        public fixed byte deviceNameRaw[256];
        public fixed byte pipelineCacheUUID[16];
        public VkPhysicalDeviceLimits limits;
        public VkPhysicalDeviceSparseProperties sparseProperties;

        public string deviceName
        {
            get
            {
                fixed (byte* p = deviceNameRaw)
                {
                    return Marshal.PtrToStringUTF8((IntPtr)p) ?? string.Empty;
                }
            }
        }
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
        public uint maxStorageBufferRange;
        public uint maxPushConstantsSize;
        public uint maxMemoryAllocationCount;
        public uint maxSamplerAllocationCount;
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
    public struct VkPhysicalDeviceProperties2
    {
        public int sType;
        public IntPtr pNext;
        public VkPhysicalDeviceProperties properties;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceSubgroupProperties
    {
        public int sType;
        public IntPtr pNext;
        public uint subgroupSize;
        public uint supportedStages;
        public uint supportedOperations;
        public uint quadOperationsInAllStages;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceFeatures2
    {
        public int sType;
        public IntPtr pNext;
        public fixed uint features[55];
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VkPhysicalDeviceShaderIntegerDotProductFeatures
    {
        public int sType;
        public IntPtr pNext;
        public uint shaderIntegerDotProduct;
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

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateInstance")]
    public static partial int CreateInstance(ref VkInstanceCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pInstance);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyInstance")]
    public static partial void DestroyInstance(IntPtr instance, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkEnumeratePhysicalDevices")]
    public static partial int EnumeratePhysicalDevices(IntPtr instance, ref uint pPhysicalDeviceCount, [Out] IntPtr[]? pPhysicalDevices);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceProperties")]
    public static partial void GetPhysicalDeviceProperties(IntPtr physicalDevice, out VkPhysicalDeviceProperties pProperties);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceProperties2")]
    public static partial void GetPhysicalDeviceProperties2(IntPtr physicalDevice, ref VkPhysicalDeviceProperties2 pProperties);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceFeatures2")]
    public static partial void GetPhysicalDeviceFeatures2(IntPtr physicalDevice, ref VkPhysicalDeviceFeatures2 pFeatures);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceQueueFamilyProperties")]
    public static partial void GetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint pQueueFamilyPropertyCount, [Out] VkQueueFamilyProperties[]? pQueueFamilyProperties);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetPhysicalDeviceMemoryProperties")]
    public static partial void GetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, out VkPhysicalDeviceMemoryProperties pMemoryProperties);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateDevice")]
    public static partial int CreateDevice(IntPtr physicalDevice, ref VkDeviceCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pDevice);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyDevice")]
    public static partial void DestroyDevice(IntPtr device, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetDeviceQueue")]
    public static partial void GetDeviceQueue(IntPtr device, uint queueFamilyIndex, uint queueIndex, out IntPtr pQueue);

    [LibraryImport(VulkanLib, EntryPoint = "vkDeviceWaitIdle")]
    public static partial int DeviceWaitIdle(IntPtr device);

    [LibraryImport(VulkanLib, EntryPoint = "vkQueueWaitIdle")]
    public static partial int QueueWaitIdle(IntPtr queue);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateBuffer")]
    public static partial int CreateBuffer(IntPtr device, ref VkBufferCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pBuffer);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyBuffer")]
    public static partial void DestroyBuffer(IntPtr device, IntPtr buffer, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkGetBufferMemoryRequirements")]
    public static partial void GetBufferMemoryRequirements(IntPtr device, IntPtr buffer, out VkMemoryRequirements pMemoryRequirements);

    [LibraryImport(VulkanLib, EntryPoint = "vkAllocateMemory")]
    public static partial int AllocateMemory(IntPtr device, ref VkMemoryAllocateInfo pAllocateInfo, IntPtr pAllocator, out IntPtr pMemory);

    [LibraryImport(VulkanLib, EntryPoint = "vkFreeMemory")]
    public static partial void FreeMemory(IntPtr device, IntPtr memory, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkBindBufferMemory")]
    public static partial int BindBufferMemory(IntPtr device, IntPtr buffer, IntPtr memory, ulong memoryOffset);

    [LibraryImport(VulkanLib, EntryPoint = "vkMapMemory")]
    public static partial int MapMemory(IntPtr device, IntPtr memory, ulong offset, ulong size, uint flags, out IntPtr ppData);

    [LibraryImport(VulkanLib, EntryPoint = "vkUnmapMemory")]
    public static partial void UnmapMemory(IntPtr device, IntPtr memory);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateCommandPool")]
    public static partial int CreateCommandPool(IntPtr device, ref VkCommandPoolCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pCommandPool);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyCommandPool")]
    public static partial void DestroyCommandPool(IntPtr device, IntPtr commandPool, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkAllocateCommandBuffers")]
    public static partial int AllocateCommandBuffers(IntPtr device, ref VkCommandBufferAllocateInfo pAllocateInfo, [Out] IntPtr[] pCommandBuffers);

    [LibraryImport(VulkanLib, EntryPoint = "vkBeginCommandBuffer")]
    public static partial int BeginCommandBuffer(IntPtr commandBuffer, ref VkCommandBufferBeginInfo pBeginInfo);

    [LibraryImport(VulkanLib, EntryPoint = "vkEndCommandBuffer")]
    public static partial int EndCommandBuffer(IntPtr commandBuffer);

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

    [StructLayout(LayoutKind.Sequential)]
    public struct VkBufferCopy
    {
        public ulong srcOffset;
        public ulong dstOffset;
        public ulong size;
    }

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateShaderModule")]
    public static partial int CreateShaderModule(IntPtr device, ref VkShaderModuleCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pShaderModule);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyShaderModule")]
    public static partial void DestroyShaderModule(IntPtr device, IntPtr shaderModule, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateDescriptorSetLayout")]
    public static partial int CreateDescriptorSetLayout(IntPtr device, ref VkDescriptorSetLayoutCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pSetLayout);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyDescriptorSetLayout")]
    public static partial void DestroyDescriptorSetLayout(IntPtr device, IntPtr descriptorSetLayout, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreatePipelineLayout")]
    public static partial int CreatePipelineLayout(IntPtr device, ref VkPipelineLayoutCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pPipelineLayout);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyPipelineLayout")]
    public static partial void DestroyPipelineLayout(IntPtr device, IntPtr pipelineLayout, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateDescriptorPool")]
    public static partial int CreateDescriptorPool(IntPtr device, ref VkDescriptorPoolCreateInfo pCreateInfo, IntPtr pAllocator, out IntPtr pDescriptorPool);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyDescriptorPool")]
    public static partial void DestroyDescriptorPool(IntPtr device, IntPtr descriptorPool, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkAllocateDescriptorSets")]
    public static partial int AllocateDescriptorSets(IntPtr device, ref VkDescriptorSetAllocateInfo pAllocateInfo, IntPtr* pDescriptorSets);

    [LibraryImport(VulkanLib, EntryPoint = "vkUpdateDescriptorSets")]
    public static partial void UpdateDescriptorSets(IntPtr device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, IntPtr pDescriptorCopies);

    [LibraryImport(VulkanLib, EntryPoint = "vkCreateComputePipelines")]
    public static partial int CreateComputePipelines(IntPtr device, IntPtr pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, IntPtr pAllocator, IntPtr* pPipelines);

    [LibraryImport(VulkanLib, EntryPoint = "vkDestroyPipeline")]
    public static partial void DestroyPipeline(IntPtr device, IntPtr pipeline, IntPtr pAllocator);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdBindPipeline")]
    public static partial void CmdBindPipeline(IntPtr commandBuffer, int pipelineBindPoint, IntPtr pipeline);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdBindDescriptorSets")]
    public static partial void CmdBindDescriptorSets(IntPtr commandBuffer, int pipelineBindPoint, IntPtr layout, uint firstSet, uint descriptorSetCount, IntPtr* pDescriptorSets, uint dynamicOffsetCount, uint* pDynamicOffsets);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdPushConstants")]
    public static partial void CmdPushConstants(IntPtr commandBuffer, IntPtr layout, uint stageFlags, uint offset, uint size, void* pValues);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdDispatch")]
    public static partial void CmdDispatch(IntPtr commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdCopyBuffer")]
    public static partial void CmdCopyBuffer(IntPtr commandBuffer, IntPtr srcBuffer, IntPtr dstBuffer, uint regionCount, VkBufferCopy* pRegions);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdFillBuffer")]
    public static partial void CmdFillBuffer(IntPtr commandBuffer, IntPtr dstBuffer, ulong dstOffset, ulong size, uint data);

    [LibraryImport(VulkanLib, EntryPoint = "vkCmdPipelineBarrier")]
    public static partial void CmdPipelineBarrier(IntPtr commandBuffer, uint srcStageMask, uint dstStageMask, uint dependencyFlags, uint memoryBarrierCount, VkMemoryBarrier* pMemoryBarriers, uint bufferMemoryBarrierCount, IntPtr pBufferMemoryBarriers, uint imageMemoryBarrierCount, IntPtr pImageMemoryBarriers);

    [LibraryImport(VulkanLib, EntryPoint = "vkResetCommandBuffer")]
    public static partial int ResetCommandBuffer(IntPtr commandBuffer, uint flags);

    [LibraryImport(VulkanLib, EntryPoint = "vkQueueSubmit")]
    public static partial int QueueSubmit(IntPtr queue, uint submitCount, ref VkSubmitInfo pSubmits, IntPtr fence);

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
