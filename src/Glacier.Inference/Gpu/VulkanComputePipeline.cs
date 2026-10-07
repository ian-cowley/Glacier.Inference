namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Helper for creating and dispatching Vulkan SPIR-V compute pipelines.
/// Works uniformly across Windows, Linux, and Android (Qualcomm Adreno, ARM Mali, NVIDIA, AMD, Intel).
/// </summary>
public static unsafe class VulkanComputePipeline
{
    public static IntPtr CreateShaderModule(IntPtr device, ReadOnlySpan<byte> spirv)
    {
        fixed (byte* pCode = spirv)
        {
            var createInfo = new VulkanDriver.VkShaderModuleCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize = (nuint)spirv.Length,
                pCode = (uint*)pCode
            };

            int res = VulkanDriver.CreateShaderModule(device, ref createInfo, IntPtr.Zero, out IntPtr module);
            if (res != VulkanDriver.VK_SUCCESS || module == IntPtr.Zero)
                throw new InvalidOperationException($"vkCreateShaderModule failed: code {res}");
            return module;
        }
    }

    public static void CreatePipeline(
        IntPtr device,
        ReadOnlySpan<byte> spirv,
        uint storageBufferCount,
        uint pushConstantBytes,
        out IntPtr pipeline,
        out IntPtr pipelineLayout,
        out IntPtr descSetLayout)
    {
        // 1. Create Descriptor Set Layout
        var bindings = stackalloc VulkanDriver.VkDescriptorSetLayoutBinding[(int)storageBufferCount];
        for (uint i = 0; i < storageBufferCount; i++)
        {
            bindings[i] = new VulkanDriver.VkDescriptorSetLayoutBinding
            {
                binding = i,
                descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT
            };
        }

        var dslInfo = new VulkanDriver.VkDescriptorSetLayoutCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = storageBufferCount,
            pBindings = (IntPtr)bindings
        };

        int res = VulkanDriver.CreateDescriptorSetLayout(device, ref dslInfo, IntPtr.Zero, out descSetLayout);
        if (res != VulkanDriver.VK_SUCCESS)
            throw new InvalidOperationException($"vkCreateDescriptorSetLayout failed: code {res}");

        // 2. Create Pipeline Layout
        IntPtr dsl = descSetLayout;
        var pcr = new VulkanDriver.VkPushConstantRange
        {
            stageFlags = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
            offset = 0,
            size = pushConstantBytes
        };

        var plInfo = new VulkanDriver.VkPipelineLayoutCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = (IntPtr)(&dsl),
            pushConstantRangeCount = pushConstantBytes > 0 ? 1u : 0u,
            pPushConstantRanges = pushConstantBytes > 0 ? (IntPtr)(&pcr) : IntPtr.Zero
        };

        res = VulkanDriver.CreatePipelineLayout(device, ref plInfo, IntPtr.Zero, out pipelineLayout);
        if (res != VulkanDriver.VK_SUCCESS)
            throw new InvalidOperationException($"vkCreatePipelineLayout failed: code {res}");

        // 3. Create Shader Module
        IntPtr shaderModule = CreateShaderModule(device, spirv);

        // 4. Create Compute Pipeline
        byte[] entryName = "main\0"u8.ToArray();
        fixed (byte* pEntry = entryName)
        {
            var stageInfo = new VulkanDriver.VkPipelineShaderStageCreateInfo
            {
                sType = 18, // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
                stage = VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT,
                module = shaderModule,
                pName = (IntPtr)pEntry
            };

            var cpInfo = new VulkanDriver.VkComputePipelineCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage = stageInfo,
                layout = pipelineLayout
            };

            IntPtr pip = IntPtr.Zero;
            res = VulkanDriver.CreateComputePipelines(device, IntPtr.Zero, 1, &cpInfo, IntPtr.Zero, &pip);
            VulkanDriver.DestroyShaderModule(device, shaderModule, IntPtr.Zero);

            if (res != VulkanDriver.VK_SUCCESS || pip == IntPtr.Zero)
                throw new InvalidOperationException($"vkCreateComputePipelines failed: code {res}");

            pipeline = pip;
        }
    }
}
