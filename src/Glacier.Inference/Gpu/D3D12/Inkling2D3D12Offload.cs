#if !ANDROID
namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Vortice.Direct3D12;

/// <summary>
/// Universal Direct3D 12 Compute GPU Offload for Inkling 2 Mini.
/// Works natively on BOTH NVIDIA (RTX 4060 / 4090 / 5090) and AMD (Radeon 890M RDNA 3.5 UMA).
///
/// Pinning Priority (Maximum tok/s per GB of VRAM):
///   1. OutputWeight (lm_head, ~378 MB) — runs once per decode token, eliminates huge CPU scan.
///   2. All 24 physical layers of Attention Projections (Wq, Wk, Wv, AttnRel, AttnGate, Wo, ~681 MB)
///      — covers ALL 48 logical passes! Wq+Wk+Wv+AttnRel+AttnGate run in ONE fused command list!
///   3. Looped-Core-First NVFP4 MoE Experts (blk.6..17 first = 3x VRAM leverage), then Prelude/Epilogue.
///
/// Execution Optimizations:
///   - Persistent permanently-mapped Upload and Readback buffers (ZERO per-call D3D12 allocations).
///   - Batched MoE prefill (up to MaxBatch=16 tokens in ONE command list, grouped by expert for L2 hits).
///   - Strict D3D12 resource state transitions + dedicated SRV/UAV dummy buffers (zero AMD RDNA aliasing).
/// </summary>
public sealed unsafe class Inkling2D3D12Offload : IDisposable
{
    public const int MaxMoeBatch = 16;

    private readonly D3D12Context _ctx;
    private readonly Inkling2Weights _w;
    private readonly bool _ownsContext;

    // NVFP4 MoE & Q6_K LM head pipelines (zero-aliasing dedicated shaders)
    private readonly ID3D12RootSignature _sigNvfp4;
    private readonly ID3D12PipelineState _psoNvfp4;
    private readonly ID3D12PipelineState _psoQ6K;
    private readonly ID3D12RootSignature _sigReduce;
    private readonly ID3D12PipelineState _psoSituBatch;
    private readonly ID3D12PipelineState _psoWeightedSum;

    // Generic GEMV pipelines for OutputWeight (Q6_K/Q8_0/etc.) and Attention (Q8_0/IQ4_XS/etc.)
    private readonly ID3D12RootSignature _sigGenericGemv;
    private readonly Dictionary<GgufType, ID3D12PipelineState> _psoGenericGemv = new();
    private readonly ID3D12Resource _dummySrvBuf; // permanently NonPixelShaderResource (for t1: bias)
    private readonly ID3D12Resource _dummyUavBuf; // permanently UnorderedAccess (for u1: residual)

    // OutputWeight (lm_head) GPU buffer
    private readonly ID3D12Resource? _outputWeightGpu;
    private readonly ID3D12Resource? _logitsBuf;
    private readonly ID3D12Resource? _logitsReadbackBuf;
    private float* _pLogitsReadback;
    private readonly ulong _logitsBytes;

    // Attention projection GPU buffers (per physical layer)
    private readonly ID3D12Resource?[] _wqGpu;
    private readonly ID3D12Resource?[] _wkGpu;
    private readonly ID3D12Resource?[] _wvGpu;
    private readonly ID3D12Resource?[] _relGpu;
    private readonly ID3D12Resource?[] _gateGpu;
    private readonly ID3D12Resource?[] _woGpu;
    private readonly bool[] _attnPinned;

    // Attention scratch buffers (reusable, permanently UnorderedAccess)
    private readonly ID3D12Resource _in0Buf;
    private readonly ID3D12Resource _in1Buf;
    private readonly ID3D12Resource _in2Buf;
    private readonly ID3D12Resource _qBuf;
    private readonly ID3D12Resource _kBuf;
    private readonly ID3D12Resource _vBuf;
    private readonly ID3D12Resource _rBuf;
    private readonly ID3D12Resource _gateOutBuf;
    private readonly ID3D12Resource _attnOutInBuf;
    private readonly ID3D12Resource _attnProjOutBuf;
    private readonly ID3D12Resource _attnUploadBuf;
    private readonly ID3D12Resource _attnReadbackBuf;
    private float* _pAttnUpload;
    private float* _pAttnReadback;
    private readonly int _maxQDim;
    private readonly int _maxKvDim;
    private readonly int _maxRDim;

    // MoE expert GPU buffers (per physical layer)
    private readonly ID3D12Resource?[] _gateUpExpsGpu;
    private readonly ID3D12Resource?[] _downExpsGpu;
    private readonly ulong[] _gateUpExpBytes;
    private readonly ulong[] _downExpBytes;
    private readonly bool[] _layerPinned;

    // MoE reusable device scratch buffers (sized for MaxMoeBatch=16 tokens!)
    private readonly ID3D12Resource _xBuf;     // [MaxMoeBatch * embd]
    private readonly ID3D12Resource _metaBuf;  // [MaxMoeBatch * 2 * topK]
    private readonly ID3D12Resource _guBuf;    // [MaxMoeBatch * topK * 2 * ffExp]
    private readonly ID3D12Resource _hidBuf;   // [MaxMoeBatch * topK * ffExp]
    private readonly ID3D12Resource _dnBuf;    // [MaxMoeBatch * topK * embd]
    private readonly ID3D12Resource _outBuf;   // [MaxMoeBatch * embd]

    // Persistent mapped CPU<->GPU MoE staging buffers
    private readonly ID3D12Resource _uploadBuf;
    private readonly ID3D12Resource _readbackBuf;
    private float* _pUpload;
    private float* _pReadback;
    private readonly ulong _maxXBytes;
    private readonly ulong _maxMetaBytes;
    private readonly ulong _uploadMetaOffsetBytes;

    // Preallocated expert grouping list for batched MoE
    private readonly List<(int TokenIdx, int SlotIdx)>[] _expertBuckets;

    private bool _disposed;

    public string DeviceName => _ctx.DeviceName;
    public bool OutputHeadPinned { get; }
    public int PinnedAttnLayers { get; }
    public int PinnedPhysicalLayers { get; }
    public int PinnedLogicalPasses { get; }
    public ulong PinnedVramBytes { get; }

    public Inkling2D3D12Offload(D3D12Context ctx, Inkling2Weights weights, bool ownsContext = true, ulong? explicitVramReserveBytes = null)
    {
        _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
        _w = weights ?? throw new ArgumentNullException(nameof(weights));
        _ownsContext = ownsContext;

        _expertBuckets = new List<(int, int)>[_w.ExpertCount];
        for (int e = 0; e < _w.ExpertCount; e++) _expertBuckets[e] = new List<(int, int)>(4);

        // 1. Compile NVFP4 MoE shaders
        var nvfp4Params = new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0: W
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: x
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: y
        };
        _sigNvfp4 = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, nvfp4Params));
        _psoNvfp4 = _ctx.CreatePipelineState(_sigNvfp4, _ctx.CompileShader(D3D12Shaders.Inkling2GemvNVFP4));
        _psoQ6K = _ctx.CreatePipelineState(_sigNvfp4, _ctx.CompileShader(D3D12Shaders.Inkling2GemvQ6K));

        var reduceParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: meta
        };
        _sigReduce = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, reduceParams));
        _psoSituBatch = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2SituBatch));
        _psoWeightedSum = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2WeightedSum8));

        // 2. Generic GEMV root signature (for OutputWeight & Attention projections)
        var genericParams = new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0: W
            new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),  // t1: bias (SRV dummy)
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: x
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: residual (UAV dummy)
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: y
        };
        _sigGenericGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, genericParams));

        _dummySrvBuf = _ctx.CreateDeviceBuffer(256UL, ResourceFlags.None);
        _dummyUavBuf = _ctx.CreateDeviceBuffer(256UL, ResourceFlags.AllowUnorderedAccess);

        // 3. Allocate MoE scratch buffers (for up to MaxMoeBatch=16 tokens)
        int embd = _w.EmbeddingDim;
        int ffExp = _w.ExpertFfnDim;
        int topK = _w.ExpertUsedCount;

        _maxXBytes = (ulong)(MaxMoeBatch * embd) * 4UL;
        _maxMetaBytes = (ulong)(MaxMoeBatch * 2 * topK) * 4UL;
        _uploadMetaOffsetBytes = (_maxXBytes + 255UL) & ~255UL;

        _xBuf = _ctx.CreateDeviceBuffer(_maxXBytes);
        _metaBuf = _ctx.CreateDeviceBuffer(Math.Max(256UL, _maxMetaBytes));
        _guBuf = _ctx.CreateDeviceBuffer((ulong)(MaxMoeBatch * topK * 2 * ffExp) * 4UL);
        _hidBuf = _ctx.CreateDeviceBuffer((ulong)(MaxMoeBatch * topK * ffExp) * 4UL);
        _dnBuf = _ctx.CreateDeviceBuffer((ulong)(MaxMoeBatch * topK * embd) * 4UL);
        _outBuf = _ctx.CreateDeviceBuffer(_maxXBytes);

        _uploadBuf = _ctx.CreateUploadBuffer(_uploadMetaOffsetBytes + Math.Max(256UL, _maxMetaBytes));
        _readbackBuf = _ctx.CreateReadbackBuffer(_maxXBytes);
        _pUpload = MapPersistent(_uploadBuf);
        _pReadback = MapPersistent(_readbackBuf);

        // 4. Allocate Attention scratch buffers
        int maxQ = 0, maxKv = 0, maxR = 0;
        for (int ip = 0; ip < _w.PhysicalLayerCount; ip++)
        {
            var l = _w.PhysicalLayers[ip];
            maxQ = Math.Max(maxQ, l.HeadCount * l.HeadDim);
            maxKv = Math.Max(maxKv, l.HeadCountKv * l.HeadDim);
            maxR = Math.Max(maxR, l.HeadCount * _w.RelativeLength);
        }
        _maxQDim = maxQ;
        _maxKvDim = maxKv;
        _maxRDim = maxR;

        _in0Buf = _ctx.CreateDeviceBuffer((ulong)embd * 4UL);
        _in1Buf = _ctx.CreateDeviceBuffer((ulong)embd * 4UL);
        _in2Buf = _ctx.CreateDeviceBuffer((ulong)embd * 4UL);
        _qBuf = _ctx.CreateDeviceBuffer((ulong)maxQ * 4UL);
        _kBuf = _ctx.CreateDeviceBuffer((ulong)maxKv * 4UL);
        _vBuf = _ctx.CreateDeviceBuffer((ulong)maxKv * 4UL);
        _rBuf = _ctx.CreateDeviceBuffer((ulong)maxR * 4UL);
        _gateOutBuf = _ctx.CreateDeviceBuffer((ulong)maxQ * 4UL);
        _attnOutInBuf = _ctx.CreateDeviceBuffer((ulong)maxQ * 4UL);
        _attnProjOutBuf = _ctx.CreateDeviceBuffer((ulong)embd * 4UL);

        ulong attnUpBytes = (ulong)Math.Max(3 * embd, maxQ) * 4UL;
        ulong attnRdBytes = (ulong)(maxQ + maxKv + maxKv + maxR + maxQ) * 4UL;
        _attnUploadBuf = _ctx.CreateUploadBuffer(attnUpBytes);
        _attnReadbackBuf = _ctx.CreateReadbackBuffer(attnRdBytes);
        _pAttnUpload = MapPersistent(_attnUploadBuf);
        _pAttnReadback = MapPersistent(_attnReadbackBuf);

        // Transition all scratch buffers once from Common -> target states
        _ctx.BeginCommands();
        var initCmd = _ctx.CommandList;
        initCmd.ResourceBarrierTransition(_dummySrvBuf, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
        initCmd.ResourceBarrierTransition(_dummyUavBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_xBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_metaBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_guBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_hidBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_dnBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_outBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_in0Buf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_in1Buf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_in2Buf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_qBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_kBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_vBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_rBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_gateOutBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_attnOutInBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        initCmd.ResourceBarrierTransition(_attnProjOutBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        int nPhys = _w.PhysicalLayerCount;
        _wqGpu = new ID3D12Resource?[nPhys];
        _wkGpu = new ID3D12Resource?[nPhys];
        _wvGpu = new ID3D12Resource?[nPhys];
        _relGpu = new ID3D12Resource?[nPhys];
        _gateGpu = new ID3D12Resource?[nPhys];
        _woGpu = new ID3D12Resource?[nPhys];
        _attnPinned = new bool[nPhys];

        _gateUpExpsGpu = new ID3D12Resource?[nPhys];
        _downExpsGpu = new ID3D12Resource?[nPhys];
        _gateUpExpBytes = new ulong[nPhys];
        _downExpBytes = new ulong[nPhys];
        _layerPinned = new bool[nPhys];

        // 5. Determine safe VRAM budget (NVIDIA discrete + AMD 890M UMA)
        var headroom = _ctx.QueryVramHeadroom(explicitVramReserveBytes);
        ulong safeBudget;
        bool isUma = _ctx.IsUnifiedMemoryArchitecture || _ctx.DeviceName.Contains("Radeon", StringComparison.OrdinalIgnoreCase);
        if (isUma)
        {
            ulong umaBudget = Math.Min((ulong)(_ctx.TotalAccessibleMemory * 0.55), 15UL * 1024 * 1024 * 1024);
            ulong localBudget = headroom.SafeAvailableBytes > 512UL * 1024 * 1024
                ? headroom.SafeAvailableBytes - 512UL * 1024 * 1024
                : 0UL;
            safeBudget = Math.Max(localBudget, umaBudget);
        }
        else if (headroom.SafeAvailableBytes > 256UL * 1024 * 1024)
        {
            safeBudget = headroom.SafeAvailableBytes - 256UL * 1024 * 1024;
        }
        else
        {
            safeBudget = _ctx.DedicatedVideoMemory > 1_200_000_000UL ? _ctx.DedicatedVideoMemory - 1_200_000_000UL : 0UL;
        }

        ulong usedVram = 0;

        // 6. PRIORITY 1: Pin OutputWeight (lm_head, ~378 MB)
        bool allowLm = Environment.GetEnvironmentVariable("GLACIER_NO_LM") != "1";
        bool allowAttn = Environment.GetEnvironmentVariable("GLACIER_NO_ATTN") != "1";
        if (allowLm && _w.OutputWeight != null && IsGenericGemvSupported(_w.OutputWeightType))
        {
            ulong lmBytes = GetGpuAlignedBytes(_w.OutputWeightType, _w.VocabSizeUnpadded, embd);
            if (usedVram + lmBytes <= safeBudget)
            {
                try
                {
                    EnsureGenericGemvPso(_w.OutputWeightType);
                    _outputWeightGpu = UploadQuantizedWeight(_w.OutputWeightType, (IntPtr)_w.OutputWeight, _w.VocabSizeUnpadded, embd);
                    _logitsBytes = (ulong)_w.VocabSizeUnpadded * 4UL;
                    _logitsBuf = _ctx.CreateDeviceBuffer(_logitsBytes);
                    _logitsReadbackBuf = _ctx.CreateReadbackBuffer(_logitsBytes);
                    _pLogitsReadback = MapPersistent(_logitsReadbackBuf);

                    _ctx.BeginCommands();
                    _ctx.CommandList.ResourceBarrierTransition(_logitsBuf, ResourceStates.Common, ResourceStates.UnorderedAccess);
                    _ctx.EndCommandsAndExecute();
                    _ctx.Synchronize();

                    usedVram += lmBytes;
                    OutputHeadPinned = true;
                }
                catch
                {
                    _outputWeightGpu?.Dispose(); _outputWeightGpu = null;
                    _logitsBuf?.Dispose(); _logitsBuf = null;
                    if (_pLogitsReadback != null && _logitsReadbackBuf != null) { _logitsReadbackBuf.Unmap(0); _pLogitsReadback = null; }
                    _logitsReadbackBuf?.Dispose(); _logitsReadbackBuf = null;
                }
            }
        }

        // 7. PRIORITY 2: Pin ALL 24 physical layers of Attention Projections (~681 MB total = ALL 48 logical passes!)
        int pinnedAttn = 0;
        var pinOrder = new List<int>(nPhys);
        for (int i = _w.CoreStart; i < _w.CoreEnd && i < nPhys; i++) pinOrder.Add(i);
        for (int i = 0; i < _w.CoreStart && i < nPhys; i++) pinOrder.Add(i);
        for (int i = _w.CoreEnd; i < nPhys; i++) pinOrder.Add(i);

        foreach (int ip in pinOrder)
        {
            if (!allowAttn) break;
            var l = _w.PhysicalLayers[ip];
            if (!IsGenericGemvSupported(l.WqType) || !IsGenericGemvSupported(l.WkType) ||
                !IsGenericGemvSupported(l.WvType) || !IsGenericGemvSupported(l.AttnRelType) ||
                !IsGenericGemvSupported(l.AttnGateType) || !IsGenericGemvSupported(l.WoType))
                continue;

            int qDim = l.HeadCount * l.HeadDim;
            int kvDim = l.HeadCountKv * l.HeadDim;
            int rDim = l.HeadCount * _w.RelativeLength;

            ulong bWq = GetGpuAlignedBytes(l.WqType, qDim, embd);
            ulong bWk = GetGpuAlignedBytes(l.WkType, kvDim, embd);
            ulong bWv = GetGpuAlignedBytes(l.WvType, kvDim, embd);
            ulong bRel = GetGpuAlignedBytes(l.AttnRelType, rDim, embd);
            ulong bGate = GetGpuAlignedBytes(l.AttnGateType, qDim, embd);
            ulong bWo = GetGpuAlignedBytes(l.WoType, embd, qDim);
            ulong totalAttn = bWq + bWk + bWv + bRel + bGate + bWo;

            if (usedVram + totalAttn > safeBudget)
                continue;

            ID3D12Resource? wq = null, wk = null, wv = null, rel = null, gate = null, wo = null;
            try
            {
                EnsureGenericGemvPso(l.WqType);
                EnsureGenericGemvPso(l.WkType);
                EnsureGenericGemvPso(l.WvType);
                EnsureGenericGemvPso(l.AttnRelType);
                EnsureGenericGemvPso(l.AttnGateType);
                EnsureGenericGemvPso(l.WoType);

                wq = UploadQuantizedWeight(l.WqType, (IntPtr)l.Wq, qDim, embd);
                wk = UploadQuantizedWeight(l.WkType, (IntPtr)l.Wk, kvDim, embd);
                wv = UploadQuantizedWeight(l.WvType, (IntPtr)l.Wv, kvDim, embd);
                rel = UploadQuantizedWeight(l.AttnRelType, (IntPtr)l.AttnRel, rDim, embd);
                gate = UploadQuantizedWeight(l.AttnGateType, (IntPtr)l.AttnGate, qDim, embd);
                wo = UploadQuantizedWeight(l.WoType, (IntPtr)l.Wo, embd, qDim);

                _wqGpu[ip] = wq;
                _wkGpu[ip] = wk;
                _wvGpu[ip] = wv;
                _relGpu[ip] = rel;
                _gateGpu[ip] = gate;
                _woGpu[ip] = wo;
                _attnPinned[ip] = true;

                usedVram += totalAttn;
                pinnedAttn++;
            }
            catch
            {
                wq?.Dispose(); wk?.Dispose(); wv?.Dispose();
                rel?.Dispose(); gate?.Dispose(); wo?.Dispose();
                break;
            }
        }
        PinnedAttnLayers = pinnedAttn;

        // 8. PRIORITY 3: Looped-Core-First NVFP4 MoE Experts (3x VRAM leverage!)
        int pinnedPhys = 0;
        int pinnedLogical = 0;

        foreach (int ip in pinOrder)
        {
            var layer = _w.PhysicalLayers[ip];
            if (layer.FfnGateUpExpsType != GgufType.NVFP4 || layer.FfnDownExpsType != GgufType.NVFP4)
                continue;

            ulong guSingleExp = (ulong)(2 * ffExp) * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, embd);
            ulong dnSingleExp = (ulong)embd * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, ffExp);
            ulong guTotal = guSingleExp * (ulong)_w.ExpertCount;
            ulong dnTotal = dnSingleExp * (ulong)_w.ExpertCount;
            ulong layerBytes = guTotal + dnTotal;

            if (usedVram + layerBytes > safeBudget)
                continue;

            ID3D12Resource? guGpu = null;
            ID3D12Resource? dnGpu = null;
            try
            {
                guGpu = UploadStaticWeight((IntPtr)layer.FfnGateUpExps, guTotal);
                dnGpu = UploadStaticWeight((IntPtr)layer.FfnDownExps, dnTotal);

                _gateUpExpsGpu[ip] = guGpu;
                _downExpsGpu[ip] = dnGpu;
                _gateUpExpBytes[ip] = guSingleExp;
                _downExpBytes[ip] = dnSingleExp;
                _layerPinned[ip] = true;

                usedVram += layerBytes;
                pinnedPhys++;
                pinnedLogical += (ip >= _w.CoreStart && ip < _w.CoreEnd) ? _w.NumLoops : 1;
            }
            catch
            {
                guGpu?.Dispose();
                dnGpu?.Dispose();
                break;
            }
        }

        PinnedPhysicalLayers = pinnedPhys;
        PinnedLogicalPasses = pinnedLogical;
        PinnedVramBytes = usedVram;
    }

    private float* MapPersistent(ID3D12Resource res)
    {
        void* p = null;
        var hr = res.Map(0, null, &p);
        if (!hr.Success || p == null)
            throw new InvalidOperationException($"Failed to map persistent D3D12 buffer: {hr}");
        return (float*)p;
    }

    private static ulong GetGpuAlignedBytes(GgufType type, int rows, int cols)
    {
        return type switch
        {
            GgufType.Q8_0 => (ulong)rows * (ulong)(cols / 32) * 36UL,
            GgufType.Q4_0 => (ulong)rows * (ulong)(cols / 32) * 20UL,
            GgufType.Q5_0 => (ulong)rows * (ulong)(cols / 32) * 24UL,
            GgufType.Q6_K => (ulong)rows * (ulong)(cols / 256) * 212UL,
            GgufType.Q3_K => (ulong)rows * (ulong)(cols / 256) * 112UL,
            _ => (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols),
        };
    }

    private ID3D12Resource UploadQuantizedWeight(GgufType type, IntPtr hostPtr, int rows, int cols)
    {
        ulong alignedBytes = GetGpuAlignedBytes(type, rows, cols);
        ulong rawBytes = (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
        if (alignedBytes == rawBytes)
            return UploadStaticWeight(hostPtr, rawBytes);

        // Repack 2-byte-unaligned GGUF blocks to 4-byte-aligned D3D12 shader layout
        byte* padded = (byte*)NativeMemory.Alloc((nuint)alignedBytes);
        try
        {
            byte* src = (byte*)hostPtr;
            if (type == GgufType.Q8_0)
            {
                long nBlocks = (long)rows * (cols / 32);
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 34;
                    byte* d = padded + b * 36;
                    *(ushort*)d = *(ushort*)s;
                    *(ushort*)(d + 2) = 0;
                    Buffer.MemoryCopy(s + 2, d + 4, 32, 32);
                }
            }
            else if (type == GgufType.Q6_K)
            {
                long nBlocks = (long)rows * (cols / 256);
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 210;
                    byte* d = padded + b * 212;
                    Buffer.MemoryCopy(s, d, 210, 210);
                    *(ushort*)(d + 210) = 0;
                }
            }
            else if (type == GgufType.Q4_0)
            {
                long nBlocks = (long)rows * (cols / 32);
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 18;
                    byte* d = padded + b * 20;
                    *(ushort*)d = *(ushort*)s;
                    *(ushort*)(d + 2) = 0;
                    Buffer.MemoryCopy(s + 2, d + 4, 16, 16);
                }
            }
            else if (type == GgufType.Q5_0)
            {
                long nBlocks = (long)rows * (cols / 32);
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 22;
                    byte* d = padded + b * 24;
                    *(ushort*)d = *(ushort*)s;
                    *(ushort*)(d + 2) = 0;
                    Buffer.MemoryCopy(s + 2, d + 4, 20, 20);
                }
            }
            else if (type == GgufType.Q3_K)
            {
                long nBlocks = (long)rows * (cols / 256);
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 110;
                    byte* d = padded + b * 112;
                    Buffer.MemoryCopy(s, d, 110, 110);
                    *(ushort*)(d + 110) = 0;
                }
            }
            return UploadStaticWeight((IntPtr)padded, alignedBytes);
        }
        finally
        {
            NativeMemory.Free(padded);
        }
    }

    private ID3D12Resource UploadStaticWeight(IntPtr hostPtr, ulong sizeBytes)
    {
        var gpu = _ctx.CreateDeviceBuffer(sizeBytes, ResourceFlags.None);
        _ctx.CopyToDevice(gpu, hostPtr, sizeBytes);
        _ctx.BeginCommands();
        _ctx.CommandList.ResourceBarrierTransition(gpu, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        return gpu;
    }

    private static bool IsGenericGemvSupported(GgufType type) => type switch
    {
        GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q3_K or
        GgufType.Q4_0 or GgufType.Q5_0 or GgufType.Q8_0 or GgufType.Q2_K or
        GgufType.IQ4_NL or GgufType.IQ4_XS or GgufType.IQ3_XXS or GgufType.IQ3_S or
        GgufType.IQ2_XXS or GgufType.IQ2_XS or GgufType.IQ2_S or
        GgufType.MXFP4 or GgufType.NVFP4 or GgufType.BF16 or GgufType.F32 => true,
        _ => false
    };

    private ID3D12PipelineState EnsureGenericGemvPso(GgufType type)
    {
        if (_psoGenericGemv.TryGetValue(type, out var pso)) return pso;

        string shader = type switch
        {
            GgufType.Q4_K => D3D12Shaders.GemvQ4K,
            GgufType.Q5_K => D3D12Shaders.GemvQ5K,
            GgufType.Q6_K => D3D12Shaders.GemvQ6K,
            GgufType.Q3_K => D3D12Shaders.GemvQ3K,
            GgufType.Q4_0 => D3D12Shaders.GemvQ4_0,
            GgufType.Q5_0 => D3D12Shaders.GemvQ5_0,
            GgufType.Q8_0 => D3D12Shaders.GemvQ8_0,
            GgufType.Q2_K => D3D12Shaders.GemvQ2_K,
            GgufType.IQ4_NL => D3D12Shaders.GemvIQ4_NL,
            GgufType.IQ4_XS => D3D12Shaders.GemvIQ4_XS,
            GgufType.IQ3_XXS => D3D12Shaders.GemvIQ3_XXS,
            GgufType.IQ3_S => D3D12Shaders.GemvIQ3_S,
            GgufType.IQ2_XXS => D3D12Shaders.GemvIQ2_XXS,
            GgufType.IQ2_XS => D3D12Shaders.GemvIQ2_XS,
            GgufType.IQ2_S => D3D12Shaders.GemvIQ2_S,
            GgufType.MXFP4 => D3D12Shaders.GemvMXFP4,
            GgufType.NVFP4 => D3D12Shaders.GemvNVFP4,
            GgufType.BF16 => D3D12Shaders.GemvBf16,
            GgufType.F32 => D3D12Shaders.GemvFp32,
            _ => throw new NotSupportedException($"Unsupported GEMV type {type}")
        };
        pso = _ctx.CreatePipelineState(_sigGenericGemv, _ctx.CompileShader(shader));
        _psoGenericGemv[type] = pso;
        return pso;
    }

    private void DispatchGenericGemv(ID3D12GraphicsCommandList cmd, GgufType type, ID3D12Resource wGpu, ID3D12Resource xBuf, ID3D12Resource yBuf, int kCols, int mRows)
    {
        cmd.SetComputeRootSignature(_sigGenericGemv);
        cmd.SetPipelineState(_psoGenericGemv[type]);
        cmd.SetComputeRootShaderResourceView(1, wGpu.GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, _dummySrvBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, xBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _dummyUavBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, yBuf.GPUVirtualAddress);

        uint* c = stackalloc uint[6];
        c[0] = (uint)kCols;
        c[1] = (uint)mRows;
        c[2] = 0u; // has_bias
        c[3] = 0u; // has_residual
        c[4] = 1u; // has_y
        c[5] = 0u; // row_offset

        const int maxRowsPerDispatch = D3D12TensorAlign.GemvChunkRows;
        int rowOff = 0;
        while (rowOff < mRows)
        {
            int rowsThis = Math.Min(mRows - rowOff, maxRowsPerDispatch);
            c[5] = (uint)rowOff;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
            cmd.Dispatch((uint)((rowsThis + 3) / 4), 1, 1);
            rowOff += rowsThis;
        }
    }

    public bool IsAttnPinned(int physicalLayer) => (uint)physicalLayer < (uint)_attnPinned.Length && _attnPinned[physicalLayer];
    public bool IsLayerPinned(int physicalLayer) => (uint)physicalLayer < (uint)_layerPinned.Length && _layerPinned[physicalLayer];

    /// <summary>
    /// Executes OutputWeight (lm_head) on GPU in ONE command list + ONE fence wait.
    /// </summary>
    public void ExecuteOutputHead(float[] normX, Span<float> logits, int vocabOut)
    {
        int embd = _w.EmbeddingDim;
        ulong xBytes = (ulong)embd * 4UL;
        fixed (float* pX = normX)
        {
            Buffer.MemoryCopy(pX, _pUpload, xBytes, xBytes);
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;
        cmd.ResourceBarrierTransition(_xBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_xBuf, 0, _uploadBuf, 0, xBytes);
        cmd.ResourceBarrierTransition(_xBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        if (_w.OutputWeightType == GgufType.Q6_K)
        {
            cmd.SetComputeRootSignature(_sigNvfp4);
            cmd.SetPipelineState(_psoQ6K);
            cmd.SetComputeRootShaderResourceView(1, _outputWeightGpu!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _xBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _logitsBuf!.GPUVirtualAddress);

            uint* c = stackalloc uint[4];
            c[0] = (uint)embd;
            c[1] = (uint)vocabOut;
            c[2] = 0u;
            int rowOff = 0;
            while (rowOff < vocabOut)
            {
                int rowsThis = Math.Min(vocabOut - rowOff, D3D12TensorAlign.GemvChunkRows);
                c[3] = (uint)rowOff;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
                cmd.Dispatch((uint)((rowsThis + 15) / 16), 1, 1);
                rowOff += rowsThis;
            }
        }
        else
        {
            DispatchGenericGemv(cmd, _w.OutputWeightType, _outputWeightGpu!, _xBuf, _logitsBuf!, embd, vocabOut);
        }
        _ctx.UavBarrier(_logitsBuf!);

        ulong outBytes = (ulong)vocabOut * 4UL;
        cmd.ResourceBarrierTransition(_logitsBuf!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.CopyBufferRegion(_logitsReadbackBuf!, 0, _logitsBuf!, 0, outBytes);
        cmd.ResourceBarrierTransition(_logitsBuf!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        fixed (float* pDst = logits)
        {
            Buffer.MemoryCopy(_pLogitsReadback, pDst, outBytes, outBytes);
        }
    }

    /// <summary>
    /// Executes ALL 5 attention input projections (Wq, Wk, Wv, AttnRel, AttnGate) in ONE fused D3D12 command list!
    /// </summary>
    public void ExecuteAttnQkvrg(int physicalLayer, float[] in0, float[] in1, float[] in2, float[] q, float[] k, float[] v, float[] r, float[] gate)
    {
        var l = _w.PhysicalLayers[physicalLayer];
        int embd = _w.EmbeddingDim;
        int qDim = l.HeadCount * l.HeadDim;
        int kvDim = l.HeadCountKv * l.HeadDim;
        int rDim = l.HeadCount * _w.RelativeLength;
        ulong eBytes = (ulong)embd * 4UL;

        fixed (float* p0 = in0, p1 = in1, p2 = in2)
        {
            Buffer.MemoryCopy(p0, _pAttnUpload, eBytes, eBytes);
            Buffer.MemoryCopy(p1, _pAttnUpload + embd, eBytes, eBytes);
            Buffer.MemoryCopy(p2, _pAttnUpload + 2 * embd, eBytes, eBytes);
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;
        cmd.ResourceBarrierTransition(_in0Buf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.ResourceBarrierTransition(_in1Buf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.ResourceBarrierTransition(_in2Buf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_in0Buf, 0, _attnUploadBuf, 0, eBytes);
        cmd.CopyBufferRegion(_in1Buf, 0, _attnUploadBuf, eBytes, eBytes);
        cmd.CopyBufferRegion(_in2Buf, 0, _attnUploadBuf, 2 * eBytes, eBytes);
        cmd.ResourceBarrierTransition(_in0Buf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_in1Buf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_in2Buf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        // Dispatch all 5 independent projections back-to-back with ZERO barriers between them!
        DispatchGenericGemv(cmd, l.WqType, _wqGpu[physicalLayer]!, _in0Buf, _qBuf, embd, qDim);
        DispatchGenericGemv(cmd, l.WkType, _wkGpu[physicalLayer]!, _in1Buf, _kBuf, embd, kvDim);
        DispatchGenericGemv(cmd, l.WvType, _wvGpu[physicalLayer]!, _in2Buf, _vBuf, embd, kvDim);
        DispatchGenericGemv(cmd, l.AttnRelType, _relGpu[physicalLayer]!, _in0Buf, _rBuf, embd, rDim);
        DispatchGenericGemv(cmd, l.AttnGateType, _gateGpu[physicalLayer]!, _in0Buf, _gateOutBuf, embd, qDim);

        ulong qBytes = (ulong)qDim * 4UL;
        ulong kvBytes = (ulong)kvDim * 4UL;
        ulong rBytes = (ulong)rDim * 4UL;

        cmd.ResourceBarrierTransition(_qBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_kBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_vBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_rBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_gateOutBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);

        ulong off = 0;
        cmd.CopyBufferRegion(_attnReadbackBuf, off, _qBuf, 0, qBytes); off += qBytes;
        cmd.CopyBufferRegion(_attnReadbackBuf, off, _kBuf, 0, kvBytes); off += kvBytes;
        cmd.CopyBufferRegion(_attnReadbackBuf, off, _vBuf, 0, kvBytes); off += kvBytes;
        cmd.CopyBufferRegion(_attnReadbackBuf, off, _rBuf, 0, rBytes); off += rBytes;
        cmd.CopyBufferRegion(_attnReadbackBuf, off, _gateOutBuf, 0, qBytes);

        cmd.ResourceBarrierTransition(_qBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_kBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_vBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_rBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_gateOutBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        fixed (float* pq = q, pk = k, pv = v, pr = r, pg = gate)
        {
            float* src = _pAttnReadback;
            Buffer.MemoryCopy(src, pq, qBytes, qBytes); src += qDim;
            Buffer.MemoryCopy(src, pk, kvBytes, kvBytes); src += kvDim;
            Buffer.MemoryCopy(src, pv, kvBytes, kvBytes); src += kvDim;
            Buffer.MemoryCopy(src, pr, rBytes, rBytes); src += rDim;
            Buffer.MemoryCopy(src, pg, qBytes, qBytes);
        }
    }

    /// <summary>
    /// Executes Wo attention output projection on GPU in ONE command list + ONE fence wait.
    /// </summary>
    public void ExecuteAttnWo(int physicalLayer, float[] attnOut, float[] attnProj)
    {
        var l = _w.PhysicalLayers[physicalLayer];
        int embd = _w.EmbeddingDim;
        int qDim = l.HeadCount * l.HeadDim;
        ulong qBytes = (ulong)qDim * 4UL;
        ulong eBytes = (ulong)embd * 4UL;

        fixed (float* pIn = attnOut)
        {
            Buffer.MemoryCopy(pIn, _pAttnUpload, qBytes, qBytes);
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;
        cmd.ResourceBarrierTransition(_attnOutInBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_attnOutInBuf, 0, _attnUploadBuf, 0, qBytes);
        cmd.ResourceBarrierTransition(_attnOutInBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        DispatchGenericGemv(cmd, l.WoType, _woGpu[physicalLayer]!, _attnOutInBuf, _attnProjOutBuf, qDim, embd);

        cmd.ResourceBarrierTransition(_attnProjOutBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.CopyBufferRegion(_attnReadbackBuf, 0, _attnProjOutBuf, 0, eBytes);
        cmd.ResourceBarrierTransition(_attnProjOutBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        fixed (float* pOut = attnProj)
        {
            Buffer.MemoryCopy(_pAttnReadback, pOut, eBytes, eBytes);
        }
    }

    /// <summary>
    /// Executes all top-K active NVFP4 experts for a single token in ONE command list + ONE fence wait.
    /// </summary>
    public void ExecuteMoeLayer(int physicalLayer, float[] cur, int[] activeIds, float[] activeWeights, float[] moeOut)
        => ExecuteMoeLayerBatch(physicalLayer, cur, 1, activeIds, activeWeights, moeOut);

    /// <summary>
    /// Executes all top-K active NVFP4 experts for a batch of tokens (1..MaxMoeBatch) in ONE command list
    /// + ONE fence wait, grouped by expert ID for maximum GPU L2 cache reuse across tokens!
    /// </summary>
    public void ExecuteMoeLayerBatch(int physicalLayer, float[] curBatch, int batchLen, int[] activeIdsFlat, float[] activeWeightsFlat, float[] moeOutBatch)
    {
        var layer = _w.PhysicalLayers[physicalLayer];
        int embd = _w.EmbeddingDim;
        int ffExp = _w.ExpertFfnDim;
        int topK = _w.ExpertUsedCount;
        int totalPairs = batchLen * topK;

        var guGpu = _gateUpExpsGpu[physicalLayer]!;
        var dnGpu = _downExpsGpu[physicalLayer]!;
        ulong guStride = _gateUpExpBytes[physicalLayer];
        ulong dnStride = _downExpBytes[physicalLayer];

        ulong xBytes = (ulong)(batchLen * embd) * 4UL;
        ulong metaBytes = (ulong)(2 * totalPairs) * 4UL;

        // 1. Copy batch inputs + per-(token, slot) scales & weights into permanently mapped upload buffer
        fixed (float* pCur = curBatch)
        {
            Buffer.MemoryCopy(pCur, _pUpload, xBytes, xBytes);
        }

        float* pMeta = (float*)((byte*)_pUpload + _uploadMetaOffsetBytes);
        for (int e = 0; e < _w.ExpertCount; e++) _expertBuckets[e].Clear();

        for (int t = 0; t < batchLen; t++)
        {
            int baseIdx = t * topK;
            for (int k = 0; k < topK; k++)
            {
                int idx = baseIdx + k;
                int expId = activeIdsFlat[idx];
                float guScale = layer.FfnGateUpExpsScale != null ? layer.FfnGateUpExpsScale[expId] : 1.0f;
                float dnScale = layer.FfnDownExpsScale != null ? layer.FfnDownExpsScale[expId] : 1.0f;
                pMeta[idx] = guScale;
                pMeta[totalPairs + idx] = activeWeightsFlat[idx] * dnScale;
                _expertBuckets[expId].Add((t, k));
            }
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // 2. Upload xBatch and meta on GPU queue
        cmd.ResourceBarrierTransition(_xBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_xBuf, 0, _uploadBuf, 0, xBytes);
        cmd.CopyBufferRegion(_metaBuf, 0, _uploadBuf, _uploadMetaOffsetBytes, metaBytes);
        cmd.ResourceBarrierTransition(_xBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        uint* consts = stackalloc uint[4];

        // 3. Parallel Gate+Up NVFP4 GEMVs grouped by expert ID (L2 cache reuse across tokens!)
        cmd.SetComputeRootSignature(_sigNvfp4);
        cmd.SetPipelineState(_psoNvfp4);
        cmd.SetComputeRootUnorderedAccessView(2, _xBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _guBuf.GPUVirtualAddress);

        uint guGroups = (uint)((2 * ffExp + 3) / 4);
        consts[0] = (uint)embd;
        consts[1] = (uint)(2 * ffExp);
        for (int expId = 0; expId < _w.ExpertCount; expId++)
        {
            var bucket = _expertBuckets[expId];
            if (bucket.Count == 0) continue;
            cmd.SetComputeRootShaderResourceView(1, guGpu.GPUVirtualAddress + (ulong)expId * guStride);
            for (int bi = 0; bi < bucket.Count; bi++)
            {
                var (t, k) = bucket[bi];
                int pairIdx = t * topK + k;
                consts[2] = (uint)(t * embd);
                consts[3] = (uint)(pairIdx * 2 * ffExp);
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)consts, 0);
                cmd.Dispatch(guGroups, 1, 1);
            }
        }
        _ctx.UavBarrier(_guBuf);

        // 4. Fused SiTU activation across all batchLen * topK pairs in 1 dispatch
        cmd.SetComputeRootSignature(_sigReduce);
        cmd.SetPipelineState(_psoSituBatch);
        cmd.SetComputeRootUnorderedAccessView(1, _guBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
        consts[0] = (uint)ffExp;
        consts[1] = (uint)totalPairs;
        consts[2] = BitConverter.SingleToUInt32Bits(_w.SituBeta);
        consts[3] = BitConverter.SingleToUInt32Bits(_w.SituLinearBeta);
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)consts, 0);
        cmd.Dispatch((uint)((totalPairs * ffExp + 255) / 256), 1, 1);
        _ctx.UavBarrier(_hidBuf);

        // 5. Parallel Down NVFP4 GEMVs grouped by expert ID (L2 cache reuse!)
        cmd.SetComputeRootSignature(_sigNvfp4);
        cmd.SetPipelineState(_psoNvfp4);
        cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _dnBuf.GPUVirtualAddress);

        uint dnGroups = (uint)((embd + 3) / 4);
        consts[0] = (uint)ffExp;
        consts[1] = (uint)embd;
        for (int expId = 0; expId < _w.ExpertCount; expId++)
        {
            var bucket = _expertBuckets[expId];
            if (bucket.Count == 0) continue;
            cmd.SetComputeRootShaderResourceView(1, dnGpu.GPUVirtualAddress + (ulong)expId * dnStride);
            for (int bi = 0; bi < bucket.Count; bi++)
            {
                var (t, k) = bucket[bi];
                int pairIdx = t * topK + k;
                consts[2] = (uint)(pairIdx * ffExp);
                consts[3] = (uint)(pairIdx * embd);
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)consts, 0);
                cmd.Dispatch(dnGroups, 1, 1);
            }
        }
        _ctx.UavBarrier(_dnBuf);

        // 6. Fused WeightedSum8 across all batchLen tokens in 1 dispatch
        cmd.SetComputeRootSignature(_sigReduce);
        cmd.SetPipelineState(_psoWeightedSum);
        cmd.SetComputeRootUnorderedAccessView(1, _dnBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _outBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
        consts[0] = (uint)embd;
        consts[1] = (uint)topK;
        consts[2] = (uint)totalPairs; // weights_offset in meta
        consts[3] = (uint)batchLen;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)consts, 0);
        cmd.Dispatch((uint)((embd + 255) / 256), (uint)batchLen, 1);
        _ctx.UavBarrier(_outBuf);

        // 7. Copy result to persistent readback buffer in the SAME command list
        cmd.ResourceBarrierTransition(_outBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.CopyBufferRegion(_readbackBuf, 0, _outBuf, 0, xBytes);
        cmd.ResourceBarrierTransition(_outBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        // 8. Copy result from permanently mapped readback buffer
        fixed (float* pOut = moeOutBatch)
        {
            Buffer.MemoryCopy(_pReadback, pOut, xBytes, xBytes);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_pUpload != null) { _uploadBuf.Unmap(0); _pUpload = null; }
        if (_pReadback != null) { _readbackBuf.Unmap(0); _pReadback = null; }
        if (_pAttnUpload != null) { _attnUploadBuf.Unmap(0); _pAttnUpload = null; }
        if (_pAttnReadback != null) { _attnReadbackBuf.Unmap(0); _pAttnReadback = null; }
        if (_pLogitsReadback != null && _logitsReadbackBuf != null) { _logitsReadbackBuf.Unmap(0); _pLogitsReadback = null; }

        _outputWeightGpu?.Dispose();
        _logitsBuf?.Dispose();
        _logitsReadbackBuf?.Dispose();

        for (int i = 0; i < _wqGpu.Length; i++)
        {
            _wqGpu[i]?.Dispose();
            _wkGpu[i]?.Dispose();
            _wvGpu[i]?.Dispose();
            _relGpu[i]?.Dispose();
            _gateGpu[i]?.Dispose();
            _woGpu[i]?.Dispose();
            _gateUpExpsGpu[i]?.Dispose();
            _downExpsGpu[i]?.Dispose();
        }

        foreach (var pso in _psoGenericGemv.Values) pso.Dispose();
        _psoGenericGemv.Clear();

        _dummySrvBuf.Dispose();
        _dummyUavBuf.Dispose();
        _in0Buf.Dispose();
        _in1Buf.Dispose();
        _in2Buf.Dispose();
        _qBuf.Dispose();
        _kBuf.Dispose();
        _vBuf.Dispose();
        _rBuf.Dispose();
        _gateOutBuf.Dispose();
        _attnOutInBuf.Dispose();
        _attnProjOutBuf.Dispose();
        _attnUploadBuf.Dispose();
        _attnReadbackBuf.Dispose();

        _uploadBuf.Dispose();
        _readbackBuf.Dispose();
        _xBuf.Dispose();
        _metaBuf.Dispose();
        _guBuf.Dispose();
        _hidBuf.Dispose();
        _dnBuf.Dispose();
        _outBuf.Dispose();
        _psoNvfp4.Dispose();
        _psoQ6K.Dispose();
        _psoSituBatch.Dispose();
        _psoWeightedSum.Dispose();
        _sigNvfp4.Dispose();
        _sigReduce.Dispose();
        _sigGenericGemv.Dispose();
        if (_ownsContext) _ctx.Dispose();
    }
}
#endif
