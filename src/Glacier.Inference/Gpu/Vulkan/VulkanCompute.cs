namespace Glacier.Inference.Gpu;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;

/// <summary>
/// Thin helper over <see cref="VulkanContext"/>/<see cref="VulkanDriver"/> for compute-kernel based inference runners:
/// host-visible storage buffers (unified-memory APUs), embedded SPIR-V kernels, descriptor sets and command recording.
/// </summary>
internal sealed unsafe class VulkanCompute : IDisposable
{
    public readonly VulkanContext Ctx;
    private readonly IntPtr _pool;
    private readonly List<(IntPtr Buf, IntPtr Mem)> _bufs = new();
    private bool _disposed;

    public sealed class GBuf
    {
        public IntPtr Buf, Mem;
        public ulong Size;
        internal VulkanCompute Owner = null!;

        public void UploadFrom(void* src, long bytes) => Owner.Ctx.CopyToBuffer(Mem, (IntPtr)src, (ulong)bytes);
        public void Upload(float[] d) { fixed (float* p = d) UploadFrom(p, d.Length * 4L); }
        public void DownloadTo(void* dst, long bytes) => Owner.Ctx.CopyFromBuffer((IntPtr)dst, Mem, (ulong)bytes);
        public void Zero() { var z = new byte[Size]; fixed (byte* p = z) UploadFrom(p, (long)Size); }
    }

    public sealed class Kernel
    {
        public IntPtr Pipe, Layout, Dsl;
        public int NBuf;
        public uint PushBytes;
    }

    public VulkanCompute(VulkanContext ctx)
    {
        Ctx = ctx;
        var poolSize = new VulkanDriver.VkDescriptorPoolSize { type = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount = 65536 };
        var poolInfo = new VulkanDriver.VkDescriptorPoolCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets = 8192,
            poolSizeCount = 1,
            pPoolSizes = (IntPtr)(&poolSize)
        };
        Check(VulkanDriver.CreateDescriptorPool(ctx.DeviceHandle, ref poolInfo, IntPtr.Zero, out _pool), "vkCreateDescriptorPool");
    }

    public static void Check(int res, string what)
    {
        if (res != 0) throw new InvalidOperationException($"{what} failed: {res}");
    }

    /// <summary>Allocates a host-visible, host-coherent storage buffer (device-visible memory on unified-memory APUs).</summary>
    public GBuf Alloc(long bytes)
    {
        uint props = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        Ctx.CreateBuffer((ulong)Math.Max(bytes, 16), VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, props, out var b, out var m);
        _bufs.Add((b, m));
        return new GBuf { Buf = b, Mem = m, Size = (ulong)Math.Max(bytes, 16), Owner = this };
    }

    /// <summary>Allocates a buffer and fills it from host memory.</summary>
    public GBuf AllocFrom(void* src, long bytes)
    {
        var g = Alloc(bytes);
        g.UploadFrom(src, bytes);
        return g;
    }

    /// <summary>Loads a kernel from the SPIR-V embedded in this assembly (Gpu/Vulkan/Shaders/&lt;name&gt;.spv).</summary>
    public Kernel LoadEmbedded(string name, int nBuf, uint pushBytes)
    {
        using Stream? s = typeof(VulkanCompute).Assembly.GetManifestResourceStream($"vkshader.{name}.spv");
        if (s == null) throw new FileNotFoundException($"Embedded Vulkan shader '{name}' was not found.");
        byte[] spirv = new byte[s.Length];
        s.ReadExactly(spirv);
        var k = new Kernel { NBuf = nBuf, PushBytes = pushBytes };
        VulkanComputePipeline.CreatePipeline(Ctx.DeviceHandle, spirv, (uint)nBuf, pushBytes, out k.Pipe, out k.Layout, out k.Dsl);
        return k;
    }

    public IntPtr MakeSet(Kernel k, params GBuf[] bufs)
    {
        if (bufs.Length != k.NBuf) throw new ArgumentException($"Kernel expects {k.NBuf} buffers, got {bufs.Length}.");
        IntPtr dsl = k.Dsl;
        var alloc = new VulkanDriver.VkDescriptorSetAllocateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = _pool,
            descriptorSetCount = 1,
            pSetLayouts = (IntPtr)(&dsl)
        };
        IntPtr set = IntPtr.Zero;
        Check(VulkanDriver.AllocateDescriptorSets(Ctx.DeviceHandle, ref alloc, &set), "vkAllocateDescriptorSets");
        var infos = stackalloc VulkanDriver.VkDescriptorBufferInfo[bufs.Length];
        var writes = stackalloc VulkanDriver.VkWriteDescriptorSet[bufs.Length];
        for (int i = 0; i < bufs.Length; i++)
        {
            infos[i] = new() { buffer = bufs[i].Buf, offset = 0, range = bufs[i].Size };
            writes[i] = new VulkanDriver.VkWriteDescriptorSet
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet = set,
                dstBinding = (uint)i,
                descriptorCount = 1,
                descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo = (IntPtr)(&infos[i])
            };
        }
        VulkanDriver.UpdateDescriptorSets(Ctx.DeviceHandle, (uint)bufs.Length, writes, 0, IntPtr.Zero);
        return set;
    }

    private IntPtr Cmd => Ctx.CommandBufferHandle;

    public void Begin()
    {
        VulkanDriver.ResetCommandBuffer(Cmd, 0);
        var begin = new VulkanDriver.VkCommandBufferBeginInfo { sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        Check(VulkanDriver.BeginCommandBuffer(Cmd, ref begin), "vkBeginCommandBuffer");
    }

    public void Dispatch(Kernel k, IntPtr set, ReadOnlySpan<uint> push, uint gx, uint gy = 1, uint gz = 1)
    {
        VulkanDriver.CmdBindPipeline(Cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, k.Pipe);
        IntPtr s = set;
        VulkanDriver.CmdBindDescriptorSets(Cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, k.Layout, 0, 1, &s, 0, null);
        if (k.PushBytes > 0)
        {
            uint* c = stackalloc uint[(int)(k.PushBytes / 4)];
            for (int i = 0; i < k.PushBytes / 4; i++) c[i] = i < push.Length ? push[i] : 0;
            VulkanDriver.CmdPushConstants(Cmd, k.Layout, VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT, 0, k.PushBytes, c);
        }
        VulkanDriver.CmdDispatch(Cmd, gx, gy, gz);
        var barrier = new VulkanDriver.VkMemoryBarrier
        {
            sType = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
            srcAccessMask = VulkanDriver.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask = VulkanDriver.VK_ACCESS_SHADER_READ_BIT | VulkanDriver.VK_ACCESS_SHADER_WRITE_BIT
        };
        VulkanDriver.CmdPipelineBarrier(Cmd, VulkanDriver.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VulkanDriver.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, IntPtr.Zero, 0, IntPtr.Zero);
    }

    /// <summary>Ends the command buffer, submits it and waits for completion.</summary>
    public double SubmitAndWait()
    {
        IntPtr cb = Cmd;
        Check(VulkanDriver.EndCommandBuffer(cb), "vkEndCommandBuffer");
        var submit = new VulkanDriver.VkSubmitInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = (IntPtr)(&cb)
        };
        long t0 = Stopwatch.GetTimestamp();
        Check(VulkanDriver.QueueSubmit(Ctx.ComputeQueueHandle, 1, ref submit, IntPtr.Zero), "vkQueueSubmit");
        Ctx.Synchronize();
        return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
    }

    public static uint F(float v) => BitConverter.SingleToUInt32Bits(v);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Ctx.Synchronize();
        foreach (var (b, m) in _bufs) Ctx.DestroyBuffer(b, m);
        _bufs.Clear();
    }
}
