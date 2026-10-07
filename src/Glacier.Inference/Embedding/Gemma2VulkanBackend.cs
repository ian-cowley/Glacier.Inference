namespace Glacier.Inference.Embedding;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;

/// <summary>
/// Vulkan compute backend for the Gemma 2 encoder.
/// Runs on any Vulkan 1.1+ GPU: Qualcomm Adreno on Android, ARM Mali, NVIDIA, AMD, Intel.
/// Q8_0 weights are dequantised on the fly in tiled compute shaders; all forward passes remain on the GPU.
/// </summary>
public sealed unsafe class Gemma2VulkanBackend : IGemma2Backend
{
    internal sealed class VulkanBuffer : IDisposable
    {
        public IntPtr Handle;
        public IntPtr Memory;
        public ulong Size;
        public IntPtr Mapped;
        private readonly VulkanContext _ctx;
        private readonly bool _ownsMemory;

        public VulkanBuffer(VulkanContext ctx, IntPtr handle, IntPtr memory, ulong size, IntPtr mapped, bool ownsMemory = true)
        {
            _ctx = ctx;
            Handle = handle;
            Memory = memory;
            Size = size;
            Mapped = mapped;
            _ownsMemory = ownsMemory;
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                if (_ownsMemory)
                    _ctx.DestroyBuffer(Handle, Memory);
                else
                    VulkanDriver.DestroyBuffer(_ctx.DeviceHandle, Handle, IntPtr.Zero);
                Handle = IntPtr.Zero;
            }
        }
    }

    internal sealed class VulkanMemoryArena : IDisposable
    {
        private readonly VulkanContext _ctx;
        private readonly IntPtr _device;
        private readonly List<IntPtr> _memories = [];
        private readonly List<IntPtr> _mappedBases = [];
        private IntPtr _currentMem;
        private IntPtr _currentMapped;
        private ulong _currentCapacity;
        private ulong _currentOffset;
        private readonly uint _memTypeIndex;

        public VulkanMemoryArena(VulkanContext ctx, ulong initialChunkSize = 384 * 1024 * 1024)
        {
            _ctx = ctx;
            _device = ctx.DeviceHandle;
            uint memProps = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            _memTypeIndex = ctx.FindMemoryType(0xFFFFFFFF, memProps);
            AllocChunk(initialChunkSize);
        }

        private void AllocChunk(ulong size)
        {
            var allocInfo = new VulkanDriver.VkMemoryAllocateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize = size,
                memoryTypeIndex = _memTypeIndex
            };
            int res = VulkanDriver.AllocateMemory(_device, ref allocInfo, IntPtr.Zero, out _currentMem);
            if (res != 0) throw new InvalidOperationException($"Vulkan arena failed to allocate {size} bytes: code {res}");

            res = VulkanDriver.MapMemory(_device, _currentMem, 0, size, 0, out _currentMapped);
            if (res != 0) throw new InvalidOperationException($"Vulkan arena failed to map memory: code {res}");

            _memories.Add(_currentMem);
            _mappedBases.Add(_currentMapped);
            _currentCapacity = size;
            _currentOffset = 0;
        }

        public VulkanBuffer CreateBuffer(ulong size, uint usage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        {
            var bufInfo = new VulkanDriver.VkBufferCreateInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size = size,
                usage = usage,
                sharingMode = 0
            };
            int res = VulkanDriver.CreateBuffer(_device, ref bufInfo, IntPtr.Zero, out IntPtr buffer);
            if (res != 0) throw new InvalidOperationException($"vkCreateBuffer failed for {size} bytes: code {res}");

            VulkanDriver.GetBufferMemoryRequirements(_device, buffer, out var reqs);
            ulong alignedOffset = (_currentOffset + reqs.alignment - 1) & ~(reqs.alignment - 1);
            if (alignedOffset + reqs.size > _currentCapacity)
            {
                ulong nextChunk = Math.Max(reqs.size, 128 * 1024 * 1024);
                AllocChunk(nextChunk);
                alignedOffset = 0;
            }

            res = VulkanDriver.BindBufferMemory(_device, buffer, _currentMem, alignedOffset);
            if (res != 0) throw new InvalidOperationException($"vkBindBufferMemory failed: code {res}");

            IntPtr mappedPtr = (IntPtr)((byte*)_currentMapped + alignedOffset);
            _currentOffset = alignedOffset + reqs.size;
            return new VulkanBuffer(_ctx, buffer, _currentMem, size, mappedPtr, ownsMemory: false);
        }

        public void Dispose()
        {
            foreach (var m in _memories)
            {
                VulkanDriver.UnmapMemory(_device, m);
                VulkanDriver.FreeMemory(_device, m, IntPtr.Zero);
            }
            _memories.Clear();
            _mappedBases.Clear();
        }
    }

    private sealed class GpuLinear : IDisposable
    {
        public VulkanBuffer? Quants;   // Q8: uint[rows*in/4]
        public VulkanBuffer? Scales;   // Q8: float[rows*in/32]
        public VulkanBuffer? F32;      // float weights
        public int In, Out;
        public bool IsQ8 => Quants != null;
        public void Dispose() { Quants?.Dispose(); Scales?.Dispose(); F32?.Dispose(); }
    }

    private sealed class GpuLayer
    {
        public required GpuLinear Q, K, V, O, Gate, Up, Down, PleGate, PleProj;
        public required VulkanBuffer AttnNorm, QNorm, KNorm, AttnPostNorm, FfnNorm, FfnPostNorm, PleNorm;
        public required VulkanBuffer RopeInvFreq;
        public float OutScale;

        // Pre-recorded descriptor sets (updated only on sequence length change)
        public IntPtr SetAttnNorm, SetQ, SetK, SetV, SetQNorm, SetKNorm, SetVNorm;
        public IntPtr SetRopeQ, SetRopeK, SetAttn, SetO, SetAttnAddNorm;
        public IntPtr SetFfnNorm, SetGate, SetUp, SetGeluMul1, SetDown, SetFfnAddNorm;
        public IntPtr SetPleIn, SetPleInNorm, SetPleGate, SetGeluMul2, SetPleProj, SetPleAddNorm;
    }

    private readonly VulkanContext _ctx;
    private readonly VulkanMemoryArena _arena;
    private readonly GgufFile _gguf;
    private readonly Gemma2Config _cfg;
    private readonly List<IDisposable> _owned = [];
    private readonly GpuLayer[] _layers;
    private readonly GpuLinear _perLayerProj;
    private readonly VulkanBuffer _perLayerProjNorm, _outputNorm, _dummyBuf;
    private readonly object _gate = new();

    private IntPtr _pipGemmQ8, _layoutGemmQ8, _dslGemmQ8;
    private IntPtr _pipGemmF32, _layoutGemmF32, _dslGemmF32;
    private IntPtr _pipRms, _layoutRms, _dslRms;
    private IntPtr _pipRope, _layoutRope, _dslRope;
    private IntPtr _pipAttn, _layoutAttn, _dslAttn;
    private IntPtr _pipGeluMul, _layoutGeluMul, _dslGeluMul;
    private IntPtr _pipAddNorm, _layoutAddNorm, _dslAddNorm;
    private IntPtr _pipMean, _layoutMean, _dslMean;

    private IntPtr _descPool;
    private IntPtr _setOutputNorm, _setMean;

    private int _capacity;
    private VulkanBuffer? _x0, _x, _normed, _tmp, _q, _k, _v, _attnOut, _ffnGate, _ffnUp, _pleIn, _pleG, _pooled;
    private bool _disposed;

    public string Name => $"Vulkan compute on {_ctx.DeviceName} (Q8_0 on-the-fly dequant)";

    public Gemma2VulkanBackend(GgufFile gguf, Gemma2Config cfg, int deviceOrdinal = 0)
    {
        NativeDriverResolver.EnsureRegistered();
        _gguf = gguf;
        _cfg = cfg;
        _ctx = new VulkanContext(deviceOrdinal);
        _arena = new VulkanMemoryArena(_ctx);
        _owned.Add(_arena);
        _owned.Add(_ctx);

        try
        {
            InitPipelines();
            _dummyBuf = Track(_arena.CreateBuffer(256));

            int H = cfg.Hidden;
            _perLayerProj = LoadLinear("per_layer_model_proj.weight", H, cfg.PleDim * cfg.Layers, forceF32: true);
            _perLayerProjNorm = UploadVector("per_layer_proj_norm.weight", cfg.PleDim);
            _outputNorm = UploadVector("output_norm.weight", H);

            _layers = new GpuLayer[cfg.Layers];
            for (int l = 0; l < cfg.Layers; l++)
            {
                int hd = cfg.HeadDim[l], kv = cfg.KvHeads[l];
                string p = $"blk.{l}.";
                var invFreq = new float[hd / 2];
                double theta = cfg.IsSwa[l] ? cfg.RopeBaseSwa : cfg.RopeBase;
                for (int i = 0; i < invFreq.Length; i++) invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / hd));

                _layers[l] = new GpuLayer
                {
                    Q = LoadLinear(p + "attn_q.weight", H, cfg.Heads * hd),
                    K = LoadLinear(p + "attn_k.weight", H, kv * hd),
                    V = LoadLinear(p + "attn_v.weight", H, kv * hd),
                    O = LoadLinear(p + "attn_output.weight", cfg.Heads * hd, H),
                    Gate = LoadLinear(p + "ffn_gate.weight", H, cfg.FfnDim),
                    Up = LoadLinear(p + "ffn_up.weight", H, cfg.FfnDim),
                    Down = LoadLinear(p + "ffn_down.weight", cfg.FfnDim, H),
                    PleGate = LoadLinear(p + "inp_gate.weight", H, cfg.PleDim),
                    PleProj = LoadLinear(p + "proj.weight", cfg.PleDim, H),
                    AttnNorm = UploadVector(p + "attn_norm.weight", H),
                    QNorm = UploadVector(p + "attn_q_norm.weight", hd),
                    KNorm = UploadVector(p + "attn_k_norm.weight", hd),
                    AttnPostNorm = UploadVector(p + "post_attention_norm.weight", H),
                    FfnNorm = UploadVector(p + "ffn_norm.weight", H),
                    FfnPostNorm = UploadVector(p + "post_ffw_norm.weight", H),
                    PleNorm = UploadVector(p + "post_norm.weight", H),
                    RopeInvFreq = UploadFloats(invFreq),
                    OutScale = Gemma2Tensors.LoadVector(gguf, p + "layer_output_scale.weight", 1)[0],
                };
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private void InitPipelines()
    {
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GemmQ8Spirv, 4, 64, out _pipGemmQ8, out _layoutGemmQ8, out _dslGemmQ8);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GemmF32Spirv, 3, 64, out _pipGemmF32, out _layoutGemmF32, out _dslGemmF32);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.RmsNormSpirv, 3, 64, out _pipRms, out _layoutRms, out _dslRms);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.RopeSpirv, 2, 64, out _pipRope, out _layoutRope, out _dslRope);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.AttentionSpirv, 4, 64, out _pipAttn, out _layoutAttn, out _dslAttn);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GeluMulSpirv, 2, 64, out _pipGeluMul, out _layoutGeluMul, out _dslGeluMul);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.AddNormSpirv, 3, 64, out _pipAddNorm, out _layoutAddNorm, out _dslAddNorm);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.MeanPoolSpirv, 2, 64, out _pipMean, out _layoutMean, out _dslMean);
    }

    private T Track<T>(T obj) where T : IDisposable { _owned.Add(obj); return obj; }

    private VulkanBuffer UploadFloats(float[] data)
    {
        var buf = Track(_arena.CreateBuffer((ulong)Math.Max(256, data.Length * 4)));
        fixed (float* p = data) Buffer.MemoryCopy(p, (void*)buf.Mapped, (ulong)data.Length * 4, (ulong)data.Length * 4);
        return buf;
    }

    private VulkanBuffer UploadVector(string name, int len) => UploadFloats(Gemma2Tensors.LoadVector(_gguf, name, len));

    private GpuLinear LoadLinear(string name, int inDim, int outDim, bool forceF32 = false)
    {
        var info = Gemma2Tensors.Require(_gguf, name, (ulong)inDim, (ulong)outDim);
        uint type = (uint)info.Type;
        byte* src = _gguf.GetTensorPointer(info);
        var lin = new GpuLinear { In = inDim, Out = outDim };
        _owned.Add(lin);

        if (type == Gemma2Tensors.Q8_0 && !forceF32 && inDim % 32 == 0)
        {
            int blocks = inDim / 32;
            var quants = new sbyte[(long)outDim * inDim];
            var scales = new float[(long)outDim * blocks];
            for (int r = 0; r < outDim; r++)
            {
                byte* row = src + r * (long)blocks * 34;
                for (int b = 0; b < blocks; b++)
                {
                    byte* blk = row + b * 34;
                    scales[(long)r * blocks + b] = (float)Unsafe.ReadUnaligned<Half>(blk);
                    for (int i = 0; i < 32; i++) quants[(long)r * inDim + b * 32 + i] = (sbyte)blk[2 + i];
                }
            }
            lin.Quants = Track(_arena.CreateBuffer((ulong)quants.Length));
            lin.Scales = Track(_arena.CreateBuffer((ulong)(scales.Length * 4)));
            fixed (sbyte* pq = quants) Buffer.MemoryCopy(pq, (void*)lin.Quants.Mapped, (ulong)quants.Length, (ulong)quants.Length);
            fixed (float* ps = scales) Buffer.MemoryCopy(ps, (void*)lin.Scales.Mapped, (ulong)scales.Length * 4, (ulong)scales.Length * 4);
        }
        else
        {
            var w = new float[(long)outDim * inDim];
            fixed (float* pw = w)
                for (int r = 0; r < outDim; r++) Gemma2Tensors.DequantRow(src, type, r, inDim, pw + (long)r * inDim);
            lin.F32 = Track(_arena.CreateBuffer((ulong)w.Length * 4));
            fixed (float* pw = w) Buffer.MemoryCopy(pw, (void*)lin.F32.Mapped, (ulong)w.Length * 4, (ulong)w.Length * 4);
        }
        return lin;
    }

    private void EnsureCapacity(int n)
    {
        if (n <= _capacity) return;
        FreeActivations();
        int cap = ((n + 255) / 256 + 1) * 256;
        int H = _cfg.Hidden, P = _cfg.PleDim, F = _cfg.FfnDim;
        int maxQ = 0, maxKv = 0;
        for (int l = 0; l < _cfg.Layers; l++)
        {
            maxQ = Math.Max(maxQ, _cfg.Heads * _cfg.HeadDim[l]);
            maxKv = Math.Max(maxKv, _cfg.KvHeads[l] * _cfg.HeadDim[l]);
        }

        VulkanBuffer Buf(long floats) => _arena.CreateBuffer((ulong)Math.Max(256, floats * 4));
        _x0 = Buf((long)cap * H); _x = Buf((long)cap * H); _normed = Buf((long)cap * H); _tmp = Buf((long)cap * H);
        _q = Buf((long)cap * maxQ); _k = Buf((long)cap * maxKv); _v = Buf((long)cap * maxKv); _attnOut = Buf((long)cap * maxQ);
        _ffnGate = Buf((long)cap * F); _ffnUp = Buf((long)cap * F);
        _pleIn = Buf((long)cap * P); _pleG = Buf((long)cap * P); _pooled = Buf(H);
        _capacity = cap;

        CreateDescriptorSets();
    }

    private void FreeActivations()
    {
        foreach (var r in new[] { _x0, _x, _normed, _tmp, _q, _k, _v, _attnOut, _ffnGate, _ffnUp, _pleIn, _pleG, _pooled })
            r?.Dispose();
        _x0 = _x = _normed = _tmp = _q = _k = _v = _attnOut = _ffnGate = _ffnUp = _pleIn = _pleG = _pooled = null;
        _capacity = 0;
    }

    private IntPtr AllocSet(IntPtr layout)
    {
        IntPtr dsl = layout;
        var allocInfo = new VulkanDriver.VkDescriptorSetAllocateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = _descPool,
            descriptorSetCount = 1,
            pSetLayouts = (IntPtr)(&dsl)
        };
        IntPtr set = IntPtr.Zero;
        int res = VulkanDriver.AllocateDescriptorSets(_ctx.DeviceHandle, ref allocInfo, &set);
        if (res != 0) throw new InvalidOperationException($"vkAllocateDescriptorSets failed: {res}");
        return set;
    }

    private void UpdateSet(IntPtr set, VulkanBuffer b0, VulkanBuffer b1)
    {
        var bufInfos = stackalloc VulkanDriver.VkDescriptorBufferInfo[2];
        bufInfos[0] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b0.Handle, offset = 0, range = b0.Size };
        bufInfos[1] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b1.Handle, offset = 0, range = b1.Size };

        var writes = stackalloc VulkanDriver.VkWriteDescriptorSet[2];
        writes[0] = new VulkanDriver.VkWriteDescriptorSet
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = set, dstBinding = 0, descriptorCount = 1,
            descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo = (IntPtr)(&bufInfos[0])
        };
        writes[1] = new VulkanDriver.VkWriteDescriptorSet
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet = set, dstBinding = 1, descriptorCount = 1,
            descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo = (IntPtr)(&bufInfos[1])
        };
        VulkanDriver.UpdateDescriptorSets(_ctx.DeviceHandle, 2, writes, 0, IntPtr.Zero);
    }

    private void UpdateSet(IntPtr set, VulkanBuffer b0, VulkanBuffer b1, VulkanBuffer b2)
    {
        var bufInfos = stackalloc VulkanDriver.VkDescriptorBufferInfo[3];
        bufInfos[0] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b0.Handle, offset = 0, range = b0.Size };
        bufInfos[1] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b1.Handle, offset = 0, range = b1.Size };
        bufInfos[2] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b2.Handle, offset = 0, range = b2.Size };

        var writes = stackalloc VulkanDriver.VkWriteDescriptorSet[3];
        for (uint i = 0; i < 3; i++)
        {
            writes[i] = new VulkanDriver.VkWriteDescriptorSet
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet = set, dstBinding = i, descriptorCount = 1,
                descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo = (IntPtr)(&bufInfos[i])
            };
        }
        VulkanDriver.UpdateDescriptorSets(_ctx.DeviceHandle, 3, writes, 0, IntPtr.Zero);
    }

    private void UpdateSet(IntPtr set, VulkanBuffer b0, VulkanBuffer b1, VulkanBuffer b2, VulkanBuffer b3)
    {
        var bufInfos = stackalloc VulkanDriver.VkDescriptorBufferInfo[4];
        bufInfos[0] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b0.Handle, offset = 0, range = b0.Size };
        bufInfos[1] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b1.Handle, offset = 0, range = b1.Size };
        bufInfos[2] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b2.Handle, offset = 0, range = b2.Size };
        bufInfos[3] = new VulkanDriver.VkDescriptorBufferInfo { buffer = b3.Handle, offset = 0, range = b3.Size };

        var writes = stackalloc VulkanDriver.VkWriteDescriptorSet[4];
        for (uint i = 0; i < 4; i++)
        {
            writes[i] = new VulkanDriver.VkWriteDescriptorSet
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet = set, dstBinding = i, descriptorCount = 1,
                descriptorType = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo = (IntPtr)(&bufInfos[i])
            };
        }
        VulkanDriver.UpdateDescriptorSets(_ctx.DeviceHandle, 4, writes, 0, IntPtr.Zero);
    }

    private void CreateDescriptorSets()
    {
        if (_descPool != IntPtr.Zero)
        {
            VulkanDriver.DestroyDescriptorPool(_ctx.DeviceHandle, _descPool, IntPtr.Zero);
            _descPool = IntPtr.Zero;
        }

        var poolSize = new VulkanDriver.VkDescriptorPoolSize
        {
            type = VulkanDriver.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount = 8192
        };
        var poolInfo = new VulkanDriver.VkDescriptorPoolCreateInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets = 2048,
            poolSizeCount = 1,
            pPoolSizes = (IntPtr)(&poolSize)
        };
        int res = VulkanDriver.CreateDescriptorPool(_ctx.DeviceHandle, ref poolInfo, IntPtr.Zero, out _descPool);
        if (res != 0) throw new InvalidOperationException($"vkCreateDescriptorPool failed: {res}");

        for (int l = 0; l < _cfg.Layers; l++)
        {
            var L = _layers[l];
            L.SetAttnNorm = AllocSet(_dslRms);
            UpdateSet(L.SetAttnNorm, _x!, L.AttnNorm, _normed!);

            L.SetQ = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetQ, L.Q.Quants!, L.Q.Scales!, _normed!, _q!);

            L.SetK = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetK, L.K.Quants!, L.K.Scales!, _normed!, _k!);

            L.SetV = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetV, L.V.Quants!, L.V.Scales!, _normed!, _v!);

            L.SetQNorm = AllocSet(_dslRms);
            UpdateSet(L.SetQNorm, _q!, L.QNorm, _q!);

            L.SetKNorm = AllocSet(_dslRms);
            UpdateSet(L.SetKNorm, _k!, L.KNorm, _k!);

            L.SetVNorm = AllocSet(_dslRms);
            UpdateSet(L.SetVNorm, _v!, _dummyBuf, _v!);

            L.SetRopeQ = AllocSet(_dslRope);
            UpdateSet(L.SetRopeQ, _q!, L.RopeInvFreq);

            L.SetRopeK = AllocSet(_dslRope);
            UpdateSet(L.SetRopeK, _k!, L.RopeInvFreq);

            L.SetAttn = AllocSet(_dslAttn);
            UpdateSet(L.SetAttn, _q!, _k!, _v!, _attnOut!);

            L.SetO = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetO, L.O.Quants!, L.O.Scales!, _attnOut!, _tmp!);

            L.SetAttnAddNorm = AllocSet(_dslAddNorm);
            UpdateSet(L.SetAttnAddNorm, _x!, _tmp!, L.AttnPostNorm);

            L.SetFfnNorm = AllocSet(_dslRms);
            UpdateSet(L.SetFfnNorm, _x!, L.FfnNorm, _normed!);

            L.SetGate = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetGate, L.Gate.Quants!, L.Gate.Scales!, _normed!, _ffnGate!);

            L.SetUp = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetUp, L.Up.Quants!, L.Up.Scales!, _normed!, _ffnUp!);

            L.SetGeluMul1 = AllocSet(_dslGeluMul);
            UpdateSet(L.SetGeluMul1, _ffnGate!, _ffnUp!);

            L.SetDown = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetDown, L.Down.Quants!, L.Down.Scales!, _ffnGate!, _tmp!);

            L.SetFfnAddNorm = AllocSet(_dslAddNorm);
            UpdateSet(L.SetFfnAddNorm, _x!, _tmp!, L.FfnPostNorm);

            L.SetPleIn = AllocSet(_dslGemmF32);
            UpdateSet(L.SetPleIn, _perLayerProj.F32!, _x0!, _pleIn!);

            L.SetPleInNorm = AllocSet(_dslRms);
            UpdateSet(L.SetPleInNorm, _pleIn!, _perLayerProjNorm, _pleIn!);

            L.SetPleGate = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetPleGate, L.PleGate.Quants!, L.PleGate.Scales!, _x!, _pleG!);

            L.SetGeluMul2 = AllocSet(_dslGeluMul);
            UpdateSet(L.SetGeluMul2, _pleG!, _pleIn!);

            L.SetPleProj = AllocSet(_dslGemmQ8);
            UpdateSet(L.SetPleProj, L.PleProj.Quants!, L.PleProj.Scales!, _pleG!, _tmp!);

            L.SetPleAddNorm = AllocSet(_dslAddNorm);
            UpdateSet(L.SetPleAddNorm, _x!, _tmp!, L.PleNorm);
        }

        _setOutputNorm = AllocSet(_dslRms);
        UpdateSet(_setOutputNorm, _x!, _outputNorm, _x!);

        _setMean = AllocSet(_dslMean);
        UpdateSet(_setMean, _x!, _pooled!);
    }

    private IntPtr Cmd => _ctx.CommandBufferHandle;

    private void Push(IntPtr layout, ReadOnlySpan<uint> consts)
    {
        uint* c = stackalloc uint[16];
        for (int i = 0; i < 16; i++) c[i] = i < consts.Length ? consts[i] : 0;
        VulkanDriver.CmdPushConstants(Cmd, layout, VulkanDriver.VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, c);
    }

    private void Barrier()
    {
        var memBarrier = new VulkanDriver.VkMemoryBarrier
        {
            sType = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
            srcAccessMask = VulkanDriver.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask = VulkanDriver.VK_ACCESS_SHADER_READ_BIT | VulkanDriver.VK_ACCESS_SHADER_WRITE_BIT
        };
        VulkanDriver.CmdPipelineBarrier(
            Cmd,
            VulkanDriver.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VulkanDriver.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memBarrier,
            0, IntPtr.Zero,
            0, IntPtr.Zero);
    }

    private static (uint X, uint Y) Grid1D(long count)
    {
        long groups = (count + 255) / 256;
        return ((uint)Math.Min(groups, 65535), (uint)((groups + 65534) / 65535));
    }

    private static uint AsUint(float v) => BitConverter.SingleToUInt32Bits(v);

    private void Dispatch(IntPtr pipeline, IntPtr layout, IntPtr descSet, ReadOnlySpan<uint> consts, uint gx, uint gy = 1, uint gz = 1)
    {
        VulkanDriver.CmdBindPipeline(Cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        IntPtr s = descSet;
        VulkanDriver.CmdBindDescriptorSets(Cmd, VulkanDriver.VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &s, 0, null);
        Push(layout, consts);
        VulkanDriver.CmdDispatch(Cmd, gx, gy, gz);
        Barrier();
    }

    private void Flush()
    {
        VulkanDriver.EndCommandBuffer(Cmd);
        IntPtr cb = Cmd;
        var submitInfo = new VulkanDriver.VkSubmitInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = (IntPtr)(&cb)
        };
        VulkanDriver.QueueSubmit(_ctx.ComputeQueueHandle, 1, ref submitInfo, IntPtr.Zero);
        _ctx.Synchronize();

        var beginInfo = new VulkanDriver.VkCommandBufferBeginInfo
        {
            sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        VulkanDriver.BeginCommandBuffer(Cmd, ref beginInfo);
    }

    // ------------------------------------------------------------------------------ forward

    public float[] ForwardPooled(ReadOnlySpan<int> tokens) => Run(Gemma2Embedding.Gather(_gguf, _cfg, tokens), tokens.Length, pooled: true);
    public float[] ForwardHidden(ReadOnlySpan<int> tokens) => Run(Gemma2Embedding.Gather(_gguf, _cfg, tokens), tokens.Length, pooled: false);
    public float[] ForwardPooledEmbeds(float[] inputEmbeddings, int tokenCount) => Run(inputEmbeddings, tokenCount, pooled: true);

    private float[] Run(float[] x0, int n, bool pooled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (n == 0) throw new ArgumentException("At least one token is required.", nameof(n));

        lock (_gate)
        {
            EnsureCapacity(n);

            fixed (float* p = x0)
            {
                Buffer.MemoryCopy(p, (void*)_x0!.Mapped, (ulong)x0.Length * 4, (ulong)x0.Length * 4);
                Buffer.MemoryCopy(p, (void*)_x!.Mapped, (ulong)x0.Length * 4, (ulong)x0.Length * 4);
            }

            int H = _cfg.Hidden, P = _cfg.PleDim, F = _cfg.FfnDim, heads = _cfg.Heads;

            var beginInfo = new VulkanDriver.VkCommandBufferBeginInfo
            {
                sType = VulkanDriver.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            };
            VulkanDriver.BeginCommandBuffer(Cmd, ref beginInfo);

            for (int l = 0; l < _cfg.Layers; l++)
            {
                var L = _layers[l];
                int hd = _cfg.HeadDim[l], kv = _cfg.KvHeads[l];

                // 1. Attention Norm
                Dispatch(_pipRms, _layoutRms, L.SetAttnNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(1f), 1u], (uint)n);

                // 2. Q, K, V Projections
                uint qgx = (uint)((heads * hd + 15) / 16), qgy = (uint)((n + 15) / 16);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetQ, [(uint)n, (uint)H, (uint)(heads * hd), 0u], qgx, qgy);

                uint kvgx = (uint)((kv * hd + 15) / 16);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetK, [(uint)n, (uint)H, (uint)(kv * hd), 0u], kvgx, qgy);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetV, [(uint)n, (uint)H, (uint)(kv * hd), 0u], kvgx, qgy);

                // 3. Q/K/V Norms
                Dispatch(_pipRms, _layoutRms, L.SetQNorm, [(uint)hd, (uint)(n * heads), AsUint(_cfg.Eps), AsUint(1f), 1u], (uint)(n * heads));
                Dispatch(_pipRms, _layoutRms, L.SetKNorm, [(uint)hd, (uint)(n * kv), AsUint(_cfg.Eps), AsUint(1f), 1u], (uint)(n * kv));
                Dispatch(_pipRms, _layoutRms, L.SetVNorm, [(uint)hd, (uint)(n * kv), AsUint(_cfg.Eps), AsUint(1f), 0u], (uint)(n * kv));

                // 4. RoPE
                { var (gx, gy) = Grid1D((long)n * heads * (hd / 2)); Dispatch(_pipRope, _layoutRope, L.SetRopeQ, [(uint)hd, (uint)heads, (uint)(n * heads)], gx, gy); }
                { var (gx, gy) = Grid1D((long)n * kv * (hd / 2)); Dispatch(_pipRope, _layoutRope, L.SetRopeK, [(uint)hd, (uint)kv, (uint)(n * kv)], gx, gy); }

                // 5. Attention
                int window = _cfg.IsSwa[l] ? _cfg.HalfWindow : -1;
                uint blocks = (uint)((n + 7) / 8);
                uint chunk = (uint)Math.Max(1, 24576 / Math.Max(1, (long)heads * Math.Max(1, (window < 0 ? n : Math.Min(n, 2 * window + 1))) / 64));
                chunk = Math.Min(chunk, 64);
                for (uint b0 = 0; b0 < blocks; b0 += chunk)
                {
                    Dispatch(_pipAttn, _layoutAttn, L.SetAttn, [(uint)n, (uint)heads, (uint)kv, (uint)hd, unchecked((uint)window), b0], Math.Min(chunk, blocks - b0), (uint)heads);
                    if (blocks > chunk) Flush();
                }

                // 6. Out Projection + Post-Attn AddNorm
                uint ogx = (uint)((H + 15) / 16);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetO, [(uint)n, (uint)(heads * hd), (uint)H, 0u], ogx, qgy);
                Dispatch(_pipAddNorm, _layoutAddNorm, L.SetAttnAddNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(1f)], (uint)n);

                // 7. FFN Norm + Gate/Up + GeluMul + Down + Post-FFN AddNorm
                Dispatch(_pipRms, _layoutRms, L.SetFfnNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(1f), 1u], (uint)n);
                uint fgx = (uint)((F + 15) / 16);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetGate, [(uint)n, (uint)H, (uint)F, 0u], fgx, qgy);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetUp, [(uint)n, (uint)H, (uint)F, 0u], fgx, qgy);
                { var (gx, gy) = Grid1D((long)n * F); Dispatch(_pipGeluMul, _layoutGeluMul, L.SetGeluMul1, [(uint)((long)n * F)], gx, gy); }
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetDown, [(uint)n, (uint)F, (uint)H, 0u], ogx, qgy);
                Dispatch(_pipAddNorm, _layoutAddNorm, L.SetFfnAddNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(1f)], (uint)n);

                // 8. PLE Projection + Norm + Gate + Mul + Proj + AddNorm
                uint pgx = (uint)((P + 15) / 16);
                Dispatch(_pipGemmF32, _layoutGemmF32, L.SetPleIn, [(uint)n, (uint)H, (uint)P, (uint)(l * P)], pgx, qgy);
                Dispatch(_pipRms, _layoutRms, L.SetPleInNorm, [(uint)P, (uint)n, AsUint(_cfg.Eps), AsUint(1f / MathF.Sqrt(H)), 1u], (uint)n);
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetPleGate, [(uint)n, (uint)H, (uint)P, 0u], pgx, qgy);
                { var (gx, gy) = Grid1D((long)n * P); Dispatch(_pipGeluMul, _layoutGeluMul, L.SetGeluMul2, [(uint)((long)n * P)], gx, gy); }
                Dispatch(_pipGemmQ8, _layoutGemmQ8, L.SetPleProj, [(uint)n, (uint)P, (uint)H, 0u], ogx, qgy);
                Dispatch(_pipAddNorm, _layoutAddNorm, L.SetPleAddNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(L.OutScale)], (uint)n);

                Flush(); // Short submission per layer for guaranteed OS watchdog / TDR safety
            }

            // Output Norm
            Dispatch(_pipRms, _layoutRms, _setOutputNorm, [(uint)H, (uint)n, AsUint(_cfg.Eps), AsUint(1f), 1u], (uint)n);

            float[] result;
            if (pooled)
            {
                Dispatch(_pipMean, _layoutMean, _setMean, [(uint)H, (uint)n], (uint)((H + 255) / 256));
                VulkanDriver.EndCommandBuffer(Cmd);
                IntPtr cb = Cmd;
                var submitInfo = new VulkanDriver.VkSubmitInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount = 1,
                    pCommandBuffers = (IntPtr)(&cb)
                };
                VulkanDriver.QueueSubmit(_ctx.ComputeQueueHandle, 1, ref submitInfo, IntPtr.Zero);
                _ctx.Synchronize();

                result = new float[H];
                fixed (float* pr = result) Buffer.MemoryCopy((void*)_pooled!.Mapped, pr, (ulong)H * 4, (ulong)H * 4);
            }
            else
            {
                VulkanDriver.EndCommandBuffer(Cmd);
                IntPtr cb = Cmd;
                var submitInfo = new VulkanDriver.VkSubmitInfo
                {
                    sType = VulkanDriver.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount = 1,
                    pCommandBuffers = (IntPtr)(&cb)
                };
                VulkanDriver.QueueSubmit(_ctx.ComputeQueueHandle, 1, ref submitInfo, IntPtr.Zero);
                _ctx.Synchronize();

                result = new float[(long)n * H];
                fixed (float* pr = result) Buffer.MemoryCopy((void*)_x!.Mapped, pr, (ulong)result.Length * 4, (ulong)result.Length * 4);
            }

            return result;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            FreeActivations();

            if (_descPool != IntPtr.Zero)
            {
                VulkanDriver.DestroyDescriptorPool(_ctx.DeviceHandle, _descPool, IntPtr.Zero);
                _descPool = IntPtr.Zero;
            }

            void DestroyPip(IntPtr pip, IntPtr lay, IntPtr dsl)
            {
                if (pip != IntPtr.Zero) VulkanDriver.DestroyPipeline(_ctx.DeviceHandle, pip, IntPtr.Zero);
                if (lay != IntPtr.Zero) VulkanDriver.DestroyPipelineLayout(_ctx.DeviceHandle, lay, IntPtr.Zero);
                if (dsl != IntPtr.Zero) VulkanDriver.DestroyDescriptorSetLayout(_ctx.DeviceHandle, dsl, IntPtr.Zero);
            }

            DestroyPip(_pipGemmQ8, _layoutGemmQ8, _dslGemmQ8);
            DestroyPip(_pipGemmF32, _layoutGemmF32, _dslGemmF32);
            DestroyPip(_pipRms, _layoutRms, _dslRms);
            DestroyPip(_pipRope, _layoutRope, _dslRope);
            DestroyPip(_pipAttn, _layoutAttn, _dslAttn);
            DestroyPip(_pipGeluMul, _layoutGeluMul, _dslGeluMul);
            DestroyPip(_pipAddNorm, _layoutAddNorm, _dslAddNorm);
            DestroyPip(_pipMean, _layoutMean, _dslMean);

            for (int i = _owned.Count - 1; i >= 0; i--)
            {
                try { _owned[i].Dispose(); } catch { }
            }
            _owned.Clear();
        }
    }
}
