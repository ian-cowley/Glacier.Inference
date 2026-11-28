#if !ANDROID
namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;

/// <summary>
/// 100% GPU-Resident Direct3D 12 Compute Model for Thinking Machines Lab's Inkling 2 Mini.
/// Works natively on BOTH NVIDIA (RTX 4060 / 4090 / 5090) and AMD (Radeon 890M RDNA 3.5 UMA).
///
/// Key Speed Architecture:
///   - ALL small ops (RmsNorm, Block-Attention Mix, Cross-Pass LoopControl, Short-Conv,
///     Per-Head Clip/Norm, Online FlashAttention w/ FP16 KV Cache + Attention Sinks,
///     Router Top-K, Indirect GPU-Indexed NVFP4 MoE, SiTU, WeightedSum, Softcap) execute
///     purely on GPU in a SINGLE D3D12 command list per token.
///   - On AMD Radeon 890M (32GB UMA): ALL 24 physical layers (48/48 logical passes) + KV cache
///     + lm_head stay GPU-resident -> LITERALLY 1 Synchronize() per token, ZERO PCIe/CPU syncs!
///   - On NVIDIA RTX 4060 (8GB VRAM): lm_head + Attn 24/24 + KV cache + 9-10 Core MoE layers
///     stay GPU-resident; only unpinned MoE layers seamlessly fall back to 24-thread AVX-512 CPU.
/// </summary>
public sealed unsafe partial class Inkling2D3D12Model : ICpuModel
{
    public const int MaxPrefillBatch = 16;
    private readonly D3D12Context _ctx;
    private readonly Inkling2Weights _w;
    private readonly int _maxSeqLen;
    private readonly int _embd;
    private readonly int _ffExp;
    private readonly int _topK;
    private readonly int _nPhys;
    private readonly int _nLogical;
    private readonly bool _ownsContext;

    // Root signatures & PSOs
    private readonly ID3D12RootSignature _sigGemv;      // b0(4), t0, u0, u1
    private readonly ID3D12PipelineState _psoQ8_0;
    private readonly ID3D12PipelineState _psoQ6K;
    private readonly ID3D12PipelineState _psoQ5K;
    private readonly ID3D12PipelineState _psoF32;

    private readonly ID3D12RootSignature _sigNvfp4Ind;  // b0(6), t0, u0, u1, u2
    private readonly ID3D12PipelineState _psoNvfp4Ind;

    private readonly ID3D12RootSignature _sigRmsNorm;   // b0(6), t0, u0, u1
    private readonly ID3D12PipelineState _psoRmsNorm;

    private readonly ID3D12RootSignature _sigMix;       // b0(6), t0, u0, u1, u2
    private readonly ID3D12PipelineState _psoMix;

    private readonly ID3D12RootSignature _sigLoopBlend; // b0(4), u0, u1, u2, u3
    private readonly ID3D12PipelineState _psoLoopBlend;

    private readonly ID3D12RootSignature _sigShortConv; // b0(4), t0, u0, u1, u2
    private readonly ID3D12PipelineState _psoShortConv;

    private readonly ID3D12RootSignature _sigHeadNorm;  // b0(8), t0, u0, u1
    private readonly ID3D12PipelineState _psoHeadNorm;

    private readonly ID3D12RootSignature _sigAttn;      // b0(8), t0, u0..u6
    private readonly ID3D12PipelineState _psoAttn;

    private readonly ID3D12RootSignature _sigRouter;    // b0(6), t0, t1, t2, u0, u1, u2
    private readonly ID3D12PipelineState _psoRouter;

    private readonly ID3D12RootSignature _sigReduce;    // b0(4), u0, u1, u2
    private readonly ID3D12PipelineState _psoSituBatch;
    private readonly ID3D12PipelineState _psoWeightedSum;
    private readonly ID3D12PipelineState _psoVecAdd;
    private readonly ID3D12PipelineState _psoUavCopy;

    private readonly ID3D12RootSignature _sigSoftcap;   // b0(4), u0
    private readonly ID3D12PipelineState _psoSoftcap;

    // Dummy SRV/UAV (strict NVIDIA + AMD compliance)
    private readonly ID3D12Resource _dummySrv;
    private readonly ID3D12Resource _dummyUav;

    // Global GPU weights
    private readonly ID3D12Resource _tokenEmbdNormGpu;
    private readonly ID3D12Resource _outputResQueryGpu;
    private readonly ID3D12Resource _outputNormGpu;
    private readonly ID3D12Resource _outputWeightGpu;
    private readonly ID3D12Resource? _loopResQueryGpu;
    private readonly ID3D12Resource? _loopKvResQueryGpu;

    // Per-physical-layer GPU weights
    private sealed class LayerGpu
    {
        public ID3D12Resource? AttnNorm;
        public ID3D12Resource? AttnQInpNorm;
        public ID3D12Resource? AttnKInpNorm;
        public ID3D12Resource? AttnVInpNorm;
        public ID3D12Resource? FfnNorm;
        public ID3D12Resource? AttnResQuery;
        public ID3D12Resource? FfnResQuery;

        public ID3D12Resource Wq = null!;
        public ID3D12Resource Wk = null!;
        public ID3D12Resource Wv = null!;
        public ID3D12Resource AttnRel = null!;
        public ID3D12Resource AttnGate = null!;
        public ID3D12Resource Wo = null!;

        public ID3D12Resource? AttnQClip;
        public ID3D12Resource AttnSinkProj = null!;
        public ID3D12Resource AttnRelBias = null!;
        public ID3D12Resource? AttnRelClip;

        public ID3D12Resource AttnKConv = null!;
        public ID3D12Resource AttnVConv = null!;
        public ID3D12Resource AttnPostConv = null!;
        public ID3D12Resource FfnPostConv = null!;
        public ID3D12Resource? AttnKConvProj;
        public ID3D12Resource? AttnVConvProj;

        public ID3D12Resource FfnGateInp = null!;
        public ID3D12Resource? ExpProbsBias;
        public ID3D12Resource? GuScales;
        public ID3D12Resource? DnScales;

        public ID3D12Resource? LoopMsgQuery;
        public ID3D12Resource?[] LoopCtrlDown = new ID3D12Resource?[4];
        public ID3D12Resource?[] LoopCtrlGain = new ID3D12Resource?[4];
        public ID3D12Resource?[] LoopCtrlForget = new ID3D12Resource?[4];

        // MoE experts (optional if VRAM limited on 8GB dGPU)
        public ID3D12Resource? GateUpExps;
        public ID3D12Resource? DownExps;
        public uint GateUpExpStrideBytes;
        public uint DownExpStrideBytes;
        public bool MoePinned;
    }

    private readonly LayerGpu[] _layers;
    private readonly List<ID3D12Resource> _allResources = new();

    // Contiguous GPU FP16 KV Cache + Short-Conv History
    private readonly ID3D12Resource _kCacheFp16;
    private readonly ID3D12Resource _vCacheFp16;
    private readonly uint[] _kvLayerBaseU32;
    private readonly int[] _kvCap;

    private readonly ID3D12Resource _kConvHistBuf;
    private readonly ID3D12Resource _vConvHistBuf;
    private readonly ID3D12Resource _attnPostHistBuf;
    private readonly ID3D12Resource _ffnPostHistBuf;
    private readonly uint[] _kvHistBaseFloats;
    private readonly uint[] _embdHistBaseFloats;

    // GPU Scratch activation buffers (all permanently UnorderedAccess)
    private readonly ID3D12Resource _curBuf;
    private readonly ID3D12Resource _preludeBuf;
    private readonly ID3D12Resource _inpSaBuf;
    private readonly ID3D12Resource _ffnInpBuf;
    private readonly ID3D12Resource _blocksBuf;        // [16 * embd]
    private readonly ID3D12Resource _preludeBlocksBuf; // [16 * embd]
    private readonly ID3D12Resource _bankBuf;          // [16 * embd]
    private readonly ID3D12Resource _bankOutBuf;       // [embd]
    private readonly ID3D12Resource _msgBuf;           // [4 * embd]
    private readonly ID3D12Resource _mixOutBuf;        // [4 * embd]
    private readonly ID3D12Resource _in0Buf;
    private readonly ID3D12Resource _in1Buf;
    private readonly ID3D12Resource _in2Buf;
    private readonly ID3D12Resource _ctrlCatBuf;       // [2 * embd]
    private readonly ID3D12Resource _ctrlGateBuf;      // [GateRank]
    private readonly ID3D12Resource _ctrlGainBuf;      // [embd]
    private readonly ID3D12Resource _ctrlForgetBuf;    // [embd]
    private readonly ID3D12Resource _qBuf;
    private readonly ID3D12Resource _kBuf;
    private readonly ID3D12Resource _vBuf;
    private readonly ID3D12Resource _rBuf;
    private readonly ID3D12Resource _gateBuf;
    private readonly ID3D12Resource _sinksBuf;
    private readonly ID3D12Resource _kDynBuf;
    private readonly ID3D12Resource _vDynBuf;
    private readonly ID3D12Resource _attnOutBuf;
    private readonly ID3D12Resource _attnProjBuf;
    private readonly ID3D12Resource _routerLogitsBuf;
    private readonly ID3D12Resource _topkIdsBuf;       // [topK] uint32
    private readonly ID3D12Resource _metaBuf;          // [2 * topK] float
    private readonly ID3D12Resource _guBuf;            // [topK * 2 * ffExp]
    private readonly ID3D12Resource _hidBuf;           // [topK * ffExp]
    private readonly ID3D12Resource _dnBuf;            // [topK * embd]
    private readonly ID3D12Resource _moeOutBuf;        // [embd]
    private readonly ID3D12Resource _logitsBuf;        // [VocabSizeUnpadded]

    // Persistent mapped CPU<->GPU buffers
    private readonly ID3D12Resource _uploadBuf;        // [embd]
    private readonly ID3D12Resource _readbackBuf;      // [VocabSizeUnpadded]
    private readonly ID3D12Resource _cpuMoeReadbackBuf;// [embd + topK + 2*topK] (for 8GB dGPU fallback)
    private float* _pUpload;
    private float* _pReadback;
    private byte* _pCpuMoeReadback;

    // CPU fallback scratch (only used when unpinned MoE layers fall back to AVX-512 CPU)
    private readonly float[] _cpuCur;
    private readonly float[] _cpuMoeOut;
    private readonly float[][] _cpuExpHidden;
    private readonly int[] _cpuActiveIds;
    private readonly float[] _cpuActiveWeights;
    private readonly BlockQ8NVFP4[] _cpuQ8Cur;
    private readonly BlockQ8NVFP4[][] _cpuQ8ExpHidden;
    private readonly bool _useVnniMoe;
    private float[]? _cpuBatchCur;
    private float[]? _cpuBatchMoeOut;
    private float[]? _cpuBatchPairDn;
    private BlockQ8NVFP4[]? _cpuBatchQ8Cur;

    // Reusable <= 16 MB batched staging for multi-turn prefix checkpoints (ShortConv histories + SWA KV rings)
    private const ulong CkptBatchCapBytes = 16UL * 1024UL * 1024UL;
    private readonly record struct CkptBatchItem(ID3D12Resource Resource, ulong ResourceOffsetBytes, ulong BatchOffsetBytes, int ByteCount);
    private sealed class CkptBatch(CkptBatchItem[] items, ID3D12Resource[] uniqueResources, int hostOffsetBytes, int totalBytes)
    {
        public CkptBatchItem[] Items { get; } = items;
        public ID3D12Resource[] UniqueResources { get; } = uniqueResources;
        public int HostOffsetBytes { get; } = hostOffsetBytes;
        public int TotalBytes { get; } = totalBytes;
    }
    private CkptBatch[]? _ckptBatches;
    private int _ckptMaxBatchBytes;
    private byte[]? _ckptHostBuf;
    private ID3D12Resource? _ckptReadbackBuf;
    private ID3D12Resource? _ckptUploadBuf;
    private byte* _pCkptReadback;
    private byte* _pCkptUpload;
    private bool _hasCheckpoint;

    private bool _disposed;

    public ModelWeights Weights { get; }
    public string DeviceName => _ctx.DeviceName;
    public int MaxSeqLen => _maxSeqLen;
    public int StartLayer => 0;
    public int LayerCount => _nLogical;
    public bool IsLastStage => true;
    public LoraAdapterWeights? LoraWeights { get; set; }

    /// <summary>Optional post-Epilogue Inkling-2 head LoRA adapter applied before FinalLogitSoftcapping.</summary>
    public Inkling2HeadLoraAdapter? HeadLora { get; set; }

    /// <summary>Final logit softcapping constant from GGUF metadata.</summary>
    public float FinalLogitSoftcapping => _w.FinalLogitSoftcapping;

    /// <summary>Embedding dimension (hidden size).</summary>
    public int EmbeddingDim => _embd;

    public int VocabSize => _w.VocabSizeUnpadded;
    public int PinnedMoePhysicalLayers { get; }
    public int PinnedMoeLogicalPasses { get; }
    public ulong PinnedVramBytes { get; }
    public int OverflowMoePhysicalLayers { get; }
    public int OverflowMoeLogicalPasses { get; }
    public ulong OverflowUploadBytes { get; }
    public int TotalGpuMoePhysicalLayers => PinnedMoePhysicalLayers + OverflowMoePhysicalLayers;
    public int TotalGpuMoeLogicalPasses => PinnedMoeLogicalPasses + OverflowMoeLogicalPasses;
    public bool IsFullyGpuResident => TotalGpuMoePhysicalLayers == _nPhys;

    public Inkling2D3D12Model(D3D12Context ctx, ModelWeights weights, int maxSeqLen, bool ownsContext = true, ulong? explicitVramReserveBytes = null)
    {
        _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
        Weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _w = weights.Inkling2 ?? throw new InvalidOperationException("Inkling2Weights not loaded.");
        _maxSeqLen = maxSeqLen;
        _embd = _w.EmbeddingDim;
        _ffExp = _w.ExpertFfnDim;
        _topK = _w.ExpertUsedCount;
        _nPhys = _w.PhysicalLayerCount;
        _nLogical = _w.LogicalLayerCount;
        _ownsContext = ownsContext;

        _cpuCur = new float[_embd];
        _cpuMoeOut = new float[_embd];
        _cpuActiveIds = new int[_topK];
        _cpuActiveWeights = new float[_topK];
        _cpuExpHidden = new float[_topK][];
        _cpuQ8Cur = new BlockQ8NVFP4[Math.Max(1, _embd / 64)];
        _cpuQ8ExpHidden = new BlockQ8NVFP4[_topK][];
        for (int k = 0; k < _topK; k++)
        {
            _cpuExpHidden[k] = new float[_ffExp];
            _cpuQ8ExpHidden[k] = new BlockQ8NVFP4[Math.Max(1, _ffExp / 64)];
        }
        _useVnniMoe = QuantKernels.IsNvfp4VnniSupported
            && (_embd % 64 == 0)
            && (_ffExp % 64 == 0)
            && Environment.GetEnvironmentVariable("GLACIER_NO_VNNI_MOE") != "1";

        // 1. Compile all dedicated zero-aliasing Inkling 2 HLSL shaders
        _sigGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
        }));
        _psoQ8_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemvQ8_0));
        _psoQ6K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemvQ6K));
        _psoQ5K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemvQ5K));
        _psoF32 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemvF32));

        _sigNvfp4Ind = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2
        }));
        _psoNvfp4Ind = _ctx.CreatePipelineState(_sigNvfp4Ind, _ctx.CompileShader(D3D12Shaders.Inkling2GemvNVFP4Indirect));

        _sigRmsNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
        }));
        _psoRmsNorm = _ctx.CreatePipelineState(_sigRmsNorm, _ctx.CompileShader(D3D12Shaders.Inkling2RmsNorm));

        _sigMix = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2
        }));
        _psoMix = _ctx.CreatePipelineState(_sigMix, _ctx.CompileShader(D3D12Shaders.Inkling2Mix));

        _sigLoopBlend = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All), // u3
        }));
        _psoLoopBlend = _ctx.CreatePipelineState(_sigLoopBlend, _ctx.CompileShader(D3D12Shaders.Inkling2LoopCtrlBlend));

        _sigShortConv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2
        }));
        _psoShortConv = _ctx.CreatePipelineState(_sigShortConv, _ctx.CompileShader(D3D12Shaders.Inkling2ShortConv));

        _sigHeadNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 8), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
        }));
        _psoHeadNorm = _ctx.CreatePipelineState(_sigHeadNorm, _ctx.CompileShader(D3D12Shaders.Inkling2HeadNormClipKv));

        _sigAttn = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 8), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0: rel_bias
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: q
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: r
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: sinks
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All), // u3: gate
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(4, 0), ShaderVisibility.All), // u4: k_cache
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(5, 0), ShaderVisibility.All), // u5: v_cache
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(6, 0), ShaderVisibility.All), // u6: attn_out
        }));
        _psoAttn = _ctx.CreatePipelineState(_sigAttn, _ctx.CompileShader(D3D12Shaders.Inkling2AttnDecode));

        _sigRouter = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),  // t0: exp_bias
            new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),  // t1: gu_scales
            new(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All),  // t2: dn_scales
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: logits
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: topk_ids
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: meta
        }));
        _psoRouter = _ctx.CreatePipelineState(_sigRouter, _ctx.CompileShader(D3D12Shaders.Inkling2RouterTopK));

        _sigReduce = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2
        }));
        _psoSituBatch = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2SituBatch));
        _psoWeightedSum = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2WeightedSum8));
        _psoVecAdd = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2VecAdd));
        _psoUavCopy = _ctx.CreatePipelineState(_sigReduce, _ctx.CompileShader(D3D12Shaders.Inkling2UavCopy));

        _sigSoftcap = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0
        }));
        _psoSoftcap = _ctx.CreatePipelineState(_sigSoftcap, _ctx.CompileShader(D3D12Shaders.Inkling2Softcap));

        // 2. Allocate scratch & dummy buffers
        _dummySrv = Track(_ctx.CreateDeviceBuffer(256UL, ResourceFlags.None));
        _dummyUav = Track(_ctx.CreateDeviceBuffer(256UL, ResourceFlags.AllowUnorderedAccess));

        int maxQ = 0, maxKv = 0, maxR = 0, maxHead = 0;
        for (int ip = 0; ip < _nPhys; ip++)
        {
            var l = _w.PhysicalLayers[ip];
            maxQ = Math.Max(maxQ, l.HeadCount * l.HeadDim);
            maxKv = Math.Max(maxKv, l.HeadCountKv * l.HeadDim);
            maxR = Math.Max(maxR, l.HeadCount * _w.RelativeLength);
            maxHead = Math.Max(maxHead, l.HeadCount);
        }

        ulong embdBytesInit = (ulong)_embd * 4UL;
        ulong bEmbdBytes = (ulong)MaxPrefillBatch * embdBytesInit;
        _curBuf = CreateUav(bEmbdBytes);
        _preludeBuf = CreateUav(bEmbdBytes);
        _inpSaBuf = CreateUav(bEmbdBytes);
        _ffnInpBuf = CreateUav(bEmbdBytes);
        _blocksBuf = CreateUav(16UL * bEmbdBytes);
        _preludeBlocksBuf = CreateUav(16UL * bEmbdBytes);
        _bankBuf = CreateUav(16UL * bEmbdBytes);
        _bankOutBuf = CreateUav(bEmbdBytes);
        _msgBuf = CreateUav(4UL * bEmbdBytes);
        _mixOutBuf = CreateUav(4UL * bEmbdBytes);
        _in0Buf = CreateUav(bEmbdBytes);
        _in1Buf = CreateUav(bEmbdBytes);
        _in2Buf = CreateUav(bEmbdBytes);
        _ctrlCatBuf = CreateUav(2UL * bEmbdBytes);
        _ctrlGateBuf = CreateUav((ulong)(MaxPrefillBatch * _w.GateRank) * 4UL);
        _ctrlGainBuf = CreateUav(bEmbdBytes);
        _ctrlForgetBuf = CreateUav(bEmbdBytes);
        _qBuf = CreateUav((ulong)(MaxPrefillBatch * maxQ) * 4UL);
        _kBuf = CreateUav((ulong)(MaxPrefillBatch * maxKv) * 4UL);
        _vBuf = CreateUav((ulong)(MaxPrefillBatch * maxKv) * 4UL);
        _rBuf = CreateUav((ulong)(MaxPrefillBatch * maxR) * 4UL);
        _gateBuf = CreateUav((ulong)(MaxPrefillBatch * maxQ) * 4UL);
        _sinksBuf = CreateUav((ulong)(MaxPrefillBatch * maxHead) * 4UL);
        _kDynBuf = CreateUav((ulong)(MaxPrefillBatch * _w.ShortConvLCache * maxHead) * 4UL);
        _vDynBuf = CreateUav((ulong)(MaxPrefillBatch * _w.ShortConvLCache * maxHead) * 4UL);
        _attnOutBuf = CreateUav((ulong)(MaxPrefillBatch * maxQ) * 4UL);
        _attnProjBuf = CreateUav(bEmbdBytes);
        _routerLogitsBuf = CreateUav((ulong)(MaxPrefillBatch * _w.ExpertCount) * 4UL);
        _topkIdsBuf = CreateUav(Math.Max(256UL, (ulong)(MaxPrefillBatch * _topK) * 4UL));
        _metaBuf = CreateUav(Math.Max(256UL, (ulong)(MaxPrefillBatch * 2 * _topK) * 4UL));
        _guBuf = CreateUav((ulong)(MaxPrefillBatch * _topK * 2 * _ffExp) * 4UL);
        _hidBuf = CreateUav((ulong)(MaxPrefillBatch * _topK * _ffExp) * 4UL);
        _dnBuf = CreateUav((ulong)(MaxPrefillBatch * _topK * _embd) * 4UL);
        _moeOutBuf = CreateUav(bEmbdBytes);
        _logitsBuf = CreateUav((ulong)_w.VocabSizeUnpadded * 4UL);

        // 3. Allocate contiguous GPU FP16 KV Cache + Short-Conv history across all 48 logical layers
        _kvLayerBaseU32 = new uint[_nLogical];
        _kvCap = new int[_nLogical];
        _kvHistBaseFloats = new uint[_nLogical];
        _embdHistBaseFloats = new uint[_nLogical];

        ulong totalKvU32 = 0;
        ulong totalKvHistFloats = 0;
        ulong totalEmbdHistFloats = 0;
        int nHist = _w.ShortConvLCache - 1; // 3

        for (int il = 0; il < _nLogical; il++)
        {
            int ip = PhysicalLayerOf(il);
            var l = _w.PhysicalLayers[ip];
            int cap = l.IsSwa ? Math.Min(_w.SlidingWindow, _maxSeqLen) : _maxSeqLen;
            int kvDim = l.HeadCountKv * l.HeadDim;
            _kvCap[il] = cap;
            _kvLayerBaseU32[il] = (uint)totalKvU32;
            totalKvU32 += (ulong)cap * (ulong)(kvDim / 2);

            _kvHistBaseFloats[il] = (uint)totalKvHistFloats;
            totalKvHistFloats += (ulong)(nHist * kvDim);

            _embdHistBaseFloats[il] = (uint)totalEmbdHistFloats;
            totalEmbdHistFloats += (ulong)(nHist * _embd);
        }

        ulong kvBytes = totalKvU32 * 4UL;
        _kCacheFp16 = CreateUav(kvBytes);
        _vCacheFp16 = CreateUav(kvBytes);
        _kConvHistBuf = CreateUav(totalKvHistFloats * 4UL);
        _vConvHistBuf = CreateUav(totalKvHistFloats * 4UL);
        _attnPostHistBuf = CreateUav(totalEmbdHistFloats * 4UL);
        _ffnPostHistBuf = CreateUav(totalEmbdHistFloats * 4UL);

        // Persistent mapped staging buffers (sized for MaxPrefillBatch=16 tokens!)
        _uploadBuf = Track(_ctx.CreateUploadBuffer(bEmbdBytes));
        _readbackBuf = Track(_ctx.CreateReadbackBuffer((ulong)_w.VocabSizeUnpadded * 4UL));
        ulong cpuMoeRdBytes = bEmbdBytes + (ulong)(MaxPrefillBatch * _topK) * 4UL + (ulong)(MaxPrefillBatch * 2 * _topK) * 4UL;
        _cpuMoeReadbackBuf = Track(_ctx.CreateReadbackBuffer(cpuMoeRdBytes));

        _pUpload = (float*)MapPersistent(_uploadBuf);
        _pReadback = (float*)MapPersistent(_readbackBuf);
        _pCpuMoeReadback = (byte*)MapPersistent(_cpuMoeReadbackBuf);

        // Transition all scratch & KV buffers to initial states
        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;
        cmd.ResourceBarrierTransition(_dummySrv, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
        foreach (var r in _allResources)
        {
            if (r == _dummySrv || r == _uploadBuf || r == _readbackBuf || r == _cpuMoeReadbackBuf) continue;
            cmd.ResourceBarrierTransition(r, ResourceStates.Common, ResourceStates.UnorderedAccess);
        }
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        // 4. Determine safe VRAM / UMA budget
        var headroom = _ctx.QueryVramHeadroom(explicitVramReserveBytes);
        ulong safeBudget;
        bool isUma = _ctx.IsUnifiedMemoryArchitecture;
        if (isUma)
        {
            // On AMD Radeon 890M (32GB UMA), TotalAccessibleMemory is ~16.5GB; allow up to 15.5GB so ALL 24 physical layers fit!
            ulong umaBudget = Math.Min((ulong)(_ctx.TotalAccessibleMemory * 0.94), 15_600_000_000UL);
            ulong localBudget = headroom.SafeAvailableBytes > 512UL * 1024 * 1024
                ? headroom.SafeAvailableBytes - 512UL * 1024 * 1024
                : 0UL;
            safeBudget = Math.Max(localBudget, umaBudget);
        }
        else if (headroom.SafeAvailableBytes > 0)
        {
            // Strictly respect WddmVramGuard on discrete GPUs so Windows never hits the WDDM VRAM eviction cliff
            safeBudget = headroom.SafeAvailableBytes;
        }
        else
        {
            safeBudget = _ctx.DedicatedVideoMemory > 1_200_000_000UL ? _ctx.DedicatedVideoMemory - 1_200_000_000UL : 0UL;
        }

        ulong usedVram = kvBytes * 2UL;

        // 5. Upload global weights (TokenEmbdNorm, OutputResQuery, OutputNorm, OutputWeight Q6_K, LoopResQuery)
        _tokenEmbdNormGpu = UploadF32(_w.TokenEmbdNorm, _embd, ref usedVram);
        _outputResQueryGpu = UploadF32(_w.OutputResQuery, _embd, ref usedVram);
        _outputNormGpu = UploadF32(_w.OutputNorm, _embd, ref usedVram);
        _outputWeightGpu = UploadQuantized(_w.OutputWeightType, (IntPtr)_w.OutputWeight, _w.VocabSizeUnpadded, _embd, ref usedVram);

        int coreLen = _w.CoreEnd - _w.CoreStart;
        if (_w.LoopResQuery != null)
            _loopResQueryGpu = UploadF32(_w.LoopResQuery, 2 * coreLen * _embd, ref usedVram);
        if (_w.LoopKvResQuery != null)
            _loopKvResQueryGpu = UploadF32(_w.LoopKvResQuery, 2 * coreLen * _embd, ref usedVram);

        // 6. Upload all 24 physical layers' Attention + Small F32/BF16 weights (~755 MB total!)
        _layers = new LayerGpu[_nPhys];
        for (int ip = 0; ip < _nPhys; ip++)
        {
            var l = _w.PhysicalLayers[ip];
            var lg = new LayerGpu();
            int qDim = l.HeadCount * l.HeadDim;
            int kvDim = l.HeadCountKv * l.HeadDim;
            int rDim = l.HeadCount * _w.RelativeLength;

            if (l.AttnNorm != null) lg.AttnNorm = UploadF32(l.AttnNorm, _embd, ref usedVram);
            if (l.AttnQInpNorm != null) lg.AttnQInpNorm = UploadF32(l.AttnQInpNorm, _embd, ref usedVram);
            if (l.AttnKInpNorm != null) lg.AttnKInpNorm = UploadF32(l.AttnKInpNorm, _embd, ref usedVram);
            if (l.AttnVInpNorm != null) lg.AttnVInpNorm = UploadF32(l.AttnVInpNorm, _embd, ref usedVram);
            if (l.FfnNorm != null) lg.FfnNorm = UploadF32(l.FfnNorm, _embd, ref usedVram);
            if (l.AttnResQuery != null) lg.AttnResQuery = UploadF32(l.AttnResQuery, (l.IsSwa ? 1 : 3) * _embd, ref usedVram);
            if (l.FfnResQuery != null) lg.FfnResQuery = UploadF32(l.FfnResQuery, _embd, ref usedVram);

            lg.Wq = UploadQuantized(l.WqType, (IntPtr)l.Wq, qDim, _embd, ref usedVram);
            lg.Wk = UploadQuantized(l.WkType, (IntPtr)l.Wk, kvDim, _embd, ref usedVram);
            lg.Wv = UploadQuantized(l.WvType, (IntPtr)l.Wv, kvDim, _embd, ref usedVram);
            lg.AttnRel = UploadQuantized(l.AttnRelType, (IntPtr)l.AttnRel, rDim, _embd, ref usedVram);
            lg.AttnGate = UploadQuantized(l.AttnGateType, (IntPtr)l.AttnGate, qDim, _embd, ref usedVram);
            lg.Wo = UploadQuantized(l.WoType, (IntPtr)l.Wo, _embd, qDim, ref usedVram);

            if (l.AttnQClip != null) lg.AttnQClip = UploadF32(l.AttnQClip, qDim, ref usedVram);
            lg.AttnSinkProj = UploadSmallAsF32(l.AttnSinkProjType, l.AttnSinkProj, l.HeadCount, _embd, ref usedVram);
            lg.AttnRelBias = UploadF32(l.AttnRelBias, _w.RelativeLength * l.RelExtent, ref usedVram);
            if (l.AttnRelClip != null) lg.AttnRelClip = UploadF32(l.AttnRelClip, rDim, ref usedVram);

            lg.AttnKConv = UploadF32(l.AttnKConv, _w.ShortConvLCache * kvDim, ref usedVram);
            lg.AttnVConv = UploadF32(l.AttnVConv, _w.ShortConvLCache * kvDim, ref usedVram);
            lg.AttnPostConv = UploadF32(l.AttnPostConv, _w.ShortConvLCache * _embd, ref usedVram);
            lg.FfnPostConv = UploadF32(l.FfnPostConv, _w.ShortConvLCache * _embd, ref usedVram);

            if (!l.IsSwa && l.AttnKConvProj != null && l.AttnVConvProj != null)
            {
                int dynDim = _w.ShortConvLCache * l.HeadCountKv;
                lg.AttnKConvProj = UploadSmallAsF32(l.AttnKConvProjType, l.AttnKConvProj, dynDim, _embd, ref usedVram);
                lg.AttnVConvProj = UploadSmallAsF32(l.AttnVConvProjType, l.AttnVConvProj, dynDim, _embd, ref usedVram);
            }

            lg.FfnGateInp = UploadSmallAsF32(l.FfnGateInpType, l.FfnGateInp, _w.ExpertCount, _embd, ref usedVram);
            if (l.ExpProbsBias != null) lg.ExpProbsBias = UploadF32(l.ExpProbsBias, _w.ExpertCount, ref usedVram);
            if (l.FfnGateUpExpsScale != null) lg.GuScales = UploadF32(l.FfnGateUpExpsScale, _w.ExpertCount, ref usedVram);
            if (l.FfnDownExpsScale != null) lg.DnScales = UploadF32(l.FfnDownExpsScale, _w.ExpertCount, ref usedVram);

            if (l.HasLoopGates)
            {
                if (l.LoopMsgQuery != null) lg.LoopMsgQuery = UploadF32(l.LoopMsgQuery, 4 * _embd, ref usedVram);
                for (int c = 0; c < 4; c++)
                {
                    lg.LoopCtrlDown[c] = UploadSmallAsF32(l.LoopCtrlDownType[c], l.LoopCtrlDown[c], _w.GateRank, 2 * _embd, ref usedVram);
                    lg.LoopCtrlGain[c] = UploadSmallAsF32(l.LoopCtrlGainType[c], l.LoopCtrlGain[c], _embd, _w.GateRank, ref usedVram);
                    lg.LoopCtrlForget[c] = UploadSmallAsF32(l.LoopCtrlForgetType[c], l.LoopCtrlForget[c], _embd, _w.GateRank, ref usedVram);
                }
            }

            _layers[ip] = lg;
        }

        // 7. Looped-Core-First NVFP4 MoE Pinning (blk.6..17 FIRST = 3x leverage, then Prelude/Epilogue!)
        var pinOrder = new List<int>(_nPhys);
        for (int i = _w.CoreStart; i < _w.CoreEnd && i < _nPhys; i++) pinOrder.Add(i);
        for (int i = 0; i < _w.CoreStart && i < _nPhys; i++) pinOrder.Add(i);
        for (int i = _w.CoreEnd; i < _nPhys; i++) pinOrder.Add(i);

        // Pass 1: Pin maximum MoE layers into high-bandwidth HeapType.Default VRAM within safeBudget
        int pinnedPhys = 0;
        int pinnedLogical = 0;
        foreach (int ip in pinOrder)
        {
            var l = _w.PhysicalLayers[ip];
            if (l.FfnGateUpExpsType != GgufType.NVFP4 || l.FfnDownExpsType != GgufType.NVFP4)
                continue;

            ulong guSingle = (ulong)(2 * _ffExp) * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, _embd);
            ulong dnSingle = (ulong)_embd * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, _ffExp);
            ulong guTotal = guSingle * (ulong)_w.ExpertCount;
            ulong dnTotal = dnSingle * (ulong)_w.ExpertCount;
            ulong layerBytes = guTotal + dnTotal;

            if (usedVram + layerBytes > safeBudget)
                continue;

            ID3D12Resource? guGpu = null;
            ID3D12Resource? dnGpu = null;
            try
            {
                guGpu = UploadStaticRaw((IntPtr)l.FfnGateUpExps, guTotal);
                dnGpu = UploadStaticRaw((IntPtr)l.FfnDownExps, dnTotal);
                _layers[ip].GateUpExps = guGpu;
                _layers[ip].DownExps = dnGpu;
                _layers[ip].GateUpExpStrideBytes = (uint)guSingle;
                _layers[ip].DownExpStrideBytes = (uint)dnSingle;
                _layers[ip].MoePinned = true;

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

        // Pass 2: Zero-copy HeapType.Upload (ResourceStates.GenericRead = direct GPU SRV) overflow for remaining MoE layers.
        // Eliminates the remaining WDDM/DXGI default-heap commit cap on AMD UMA (24/24 GPU, 1 sync/tok!)
        // and eliminates 24 CPU↔GPU sync fences on discrete GPUs (A/B selectable via GLACIER_UPLOAD_OVERFLOW=0|1).
        string? ovEnv = Environment.GetEnvironmentVariable("GLACIER_UPLOAD_OVERFLOW");
        bool allowUploadOverflow = string.IsNullOrWhiteSpace(ovEnv)
            ? isUma
            : ovEnv != "0" && !ovEnv.Equals("false", StringComparison.OrdinalIgnoreCase);

        int overflowPhys = 0;
        int overflowLogical = 0;
        ulong overflowBytes = 0;
        if (allowUploadOverflow)
        {
            ulong maxTotalUma = Math.Min((ulong)(_ctx.TotalAccessibleMemory * 0.95), 16_000_000_000UL);
            ulong uploadBudget = isUma
                ? (maxTotalUma > usedVram ? maxTotalUma - usedVram : 0UL)
                : (ulong)(_ctx.SharedSystemMemory * 0.78);

            foreach (int ip in pinOrder)
            {
                if (_layers[ip].MoePinned) continue;
                var l = _w.PhysicalLayers[ip];
                if (l.FfnGateUpExpsType != GgufType.NVFP4 || l.FfnDownExpsType != GgufType.NVFP4)
                    continue;

                ulong guSingle = (ulong)(2 * _ffExp) * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, _embd);
                ulong dnSingle = (ulong)_embd * (ulong)GgufTypes.GetRowBytes(GgufType.NVFP4, _ffExp);
                ulong guTotal = guSingle * (ulong)_w.ExpertCount;
                ulong dnTotal = dnSingle * (ulong)_w.ExpertCount;
                ulong layerBytes = guTotal + dnTotal;

                if (overflowBytes + layerBytes > uploadBudget)
                    continue;

                ID3D12Resource? guUp = null;
                ID3D12Resource? dnUp = null;
                try
                {
                    guUp = UploadStaticUploadHeap((IntPtr)l.FfnGateUpExps, guTotal);
                    dnUp = UploadStaticUploadHeap((IntPtr)l.FfnDownExps, dnTotal);
                    _layers[ip].GateUpExps = guUp;
                    _layers[ip].DownExps = dnUp;
                    _layers[ip].GateUpExpStrideBytes = (uint)guSingle;
                    _layers[ip].DownExpStrideBytes = (uint)dnSingle;
                    _layers[ip].MoePinned = true;

                    overflowBytes += layerBytes;
                    overflowPhys++;
                    overflowLogical += (ip >= _w.CoreStart && ip < _w.CoreEnd) ? _w.NumLoops : 1;
                }
                catch
                {
                    guUp?.Dispose();
                    dnUp?.Dispose();
                    break;
                }
            }
        }

        PinnedMoePhysicalLayers = pinnedPhys;
        PinnedMoeLogicalPasses = pinnedLogical;
        PinnedVramBytes = usedVram;
        OverflowMoePhysicalLayers = overflowPhys;
        OverflowMoeLogicalPasses = overflowLogical;
        OverflowUploadBytes = overflowBytes;

        InitBatchPipelines();
        ResetState();
    }

    private ID3D12Resource Track(ID3D12Resource r) { _allResources.Add(r); return r; }
    private ID3D12Resource CreateUav(ulong bytes) => Track(_ctx.CreateDeviceBuffer(Math.Max(256UL, bytes), ResourceFlags.AllowUnorderedAccess));

    private int PhysicalLayerOf(int il)
        => il < _nPhys ? il : _w.CoreStart + (il - _nPhys) % (_w.CoreEnd - _w.CoreStart);

    private void* MapPersistent(ID3D12Resource res)
    {
        void* p = null;
        var hr = res.Map(0, null, &p);
        if (!hr.Success || p == null) throw new InvalidOperationException($"Map failed: {hr}");
        return p;
    }

    private ID3D12Resource UploadStaticRaw(IntPtr hostPtr, ulong sizeBytes)
    {
        var gpu = _ctx.CreateDeviceBuffer(Math.Max(256UL, sizeBytes), ResourceFlags.None);
        try
        {
            _ctx.CopyToDevice(gpu, hostPtr, sizeBytes);
            _ctx.BeginCommands();
            _ctx.CommandList.ResourceBarrierTransition(gpu, ResourceStates.Common, ResourceStates.NonPixelShaderResource);
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            return Track(gpu);
        }
        catch
        {
            gpu.Dispose();
            throw;
        }
    }

    private ID3D12Resource UploadStaticUploadHeap(IntPtr hostPtr, ulong sizeBytes)
    {
        var buf = _ctx.CreateUploadBuffer(Math.Max(256UL, sizeBytes));
        try
        {
            void* pDst = null;
            var hr = buf.Map(0, null, &pDst);
            if (!hr.Success || pDst == null)
                throw new InvalidOperationException($"UploadStaticUploadHeap Map failed: {hr}");
            try
            {
                Buffer.MemoryCopy((void*)hostPtr, pDst, sizeBytes, sizeBytes);
            }
            finally
            {
                buf.Unmap(0);
            }
            // Permanently ResourceStates.GenericRead -> valid SRV directly without any ResourceBarrierTransition!
            return Track(buf);
        }
        catch
        {
            buf.Dispose();
            throw;
        }
    }

    private ID3D12Resource UploadF32(float* ptr, int count, ref ulong usedVram)
    {
        ulong bytes = (ulong)count * 4UL;
        usedVram += bytes;
        return UploadStaticRaw((IntPtr)ptr, bytes);
    }

    private ID3D12Resource UploadSmallAsF32(GgufType type, byte* ptr, int rows, int cols, ref ulong usedVram)
    {
        int total = rows * cols;
        if (type == GgufType.F32)
            return UploadF32((float*)ptr, total, ref usedVram);

        // Convert tiny BF16/F16/quantized control/sink/router tensors to F32 once at load time
        float[] tmp = new float[total];
        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        fixed (float* pDst = tmp)
        {
            for (int r = 0; r < rows; r++)
                QuantKernels.ExtractEmbedding(type, ptr, r, pDst + r * cols, cols);
            return UploadF32(pDst, total, ref usedVram);
        }
    }

    private ID3D12Resource UploadQuantized(GgufType type, IntPtr hostPtr, int rows, int cols, ref ulong usedVram)
    {
        if (type == GgufType.Q8_0)
        {
            long nBlocks = (long)rows * (cols / 32);
            ulong alignedBytes = (ulong)nBlocks * 36UL;
            byte* padded = (byte*)NativeMemory.Alloc((nuint)alignedBytes);
            try
            {
                byte* src = (byte*)hostPtr;
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 34;
                    byte* d = padded + b * 36;
                    *(ushort*)d = *(ushort*)s;
                    *(ushort*)(d + 2) = 0;
                    Buffer.MemoryCopy(s + 2, d + 4, 32, 32);
                }
                usedVram += alignedBytes;
                return UploadStaticRaw((IntPtr)padded, alignedBytes);
            }
            finally { NativeMemory.Free(padded); }
        }
        if (type == GgufType.Q6_K)
        {
            long nBlocks = (long)rows * (cols / 256);
            ulong alignedBytes = (ulong)nBlocks * 212UL;
            byte* padded = (byte*)NativeMemory.Alloc((nuint)alignedBytes);
            try
            {
                byte* src = (byte*)hostPtr;
                for (long b = 0; b < nBlocks; b++)
                {
                    byte* s = src + b * 210;
                    byte* d = padded + b * 212;
                    Buffer.MemoryCopy(s, d, 210, 210);
                    *(ushort*)(d + 210) = 0;
                }
                usedVram += alignedBytes;
                return UploadStaticRaw((IntPtr)padded, alignedBytes);
            }
            finally { NativeMemory.Free(padded); }
        }
        ulong rawBytes = (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
        usedVram += rawBytes;
        return UploadStaticRaw(hostPtr, rawBytes);
    }

    public void ResetState()
    {
        _hasCheckpoint = false;
        _ctx.ZeroDeviceBuffer(_kCacheFp16, ResourceStates.UnorderedAccess);
        _ctx.ZeroDeviceBuffer(_vCacheFp16, ResourceStates.UnorderedAccess);
        _ctx.ZeroDeviceBuffer(_kConvHistBuf, ResourceStates.UnorderedAccess);
        _ctx.ZeroDeviceBuffer(_vConvHistBuf, ResourceStates.UnorderedAccess);
        _ctx.ZeroDeviceBuffer(_attnPostHistBuf, ResourceStates.UnorderedAccess);
        _ctx.ZeroDeviceBuffer(_ffnPostHistBuf, ResourceStates.UnorderedAccess);
    }

    public void PrefillPrompt(ReadOnlySpan<int> tokens, KVCache kvCache)
    {
        if (tokens.IsEmpty) return;
        if (tokens.Length == 1)
        {
            Forward(tokens[0], 0, kvCache, Span<float>.Empty, computeLogits: false);
            return;
        }
        ExecuteBatchPrefill(tokens, 0, Span<float>.Empty, computeLogits: false);
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        if (tokens.IsEmpty) return;
        if (tokens.Length == 1)
        {
            Forward(tokens[0], startPos, kvCache, logits, computeLogits);
            return;
        }
        ExecuteBatchPrefill(tokens, startPos, logits, computeLogits);
    }

    public void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits, KVCache? kvCache = null)
        => Forward(token, pos, kvCache!, logits, computeLogits);

    public void ForwardBatchStage(ReadOnlySpan<int> tokens, int startPos, ReadOnlySpan<float> inputXBatch, Span<float> outputXBatch, Span<float> logits, bool computeLogits, KVCache? kvCache = null)
        => ForwardBatch(tokens, startPos, logits, kvCache!, computeLogits);

    public void ExtractEmbedding(ReadOnlySpan<int> tokens, Span<float> destination, KVCache kvCache, PoolingStrategy strategy = PoolingStrategy.LastToken)
    {
        ResetState();
        destination.Clear();
        for (int i = 0; i < tokens.Length; i++)
            Forward(tokens[i], i, kvCache, Span<float>.Empty, computeLogits: false);
    }

    /// <summary>
    /// Full GPU-Resident Forward Pass!
    /// On AMD Radeon 890M (all 24 layers pinned): literally ONE command list + ONE Synchronize() per token!
    /// On NVIDIA RTX 4060 (9-10 core layers pinned): flushes ONLY on unpinned MoE layers.
    /// </summary>
    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        if ((uint)token >= (uint)_w.VocabSizePadded) token = 0;

        // 1. Extract 9 KB token embedding on CPU -> permanently mapped upload buffer
        QuantKernels.ExtractEmbedding(_w.TokenEmbdType, _w.TokenEmbd, token, _pUpload, _embd);
        float tau = 1.0f + _w.LogScaleAlpha * MathF.Log(MathF.Max(1.0f, (pos + 1.0f) / _w.LogScaleFloor));

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Upload 9 KB embedding into _curBuf & apply TokenEmbdNorm on GPU
        ulong embdBytes = (ulong)_embd * 4UL;
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_curBuf, 0, _uploadBuf, 0, embdBytes);
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        DispatchRmsNorm(cmd, _tokenEmbdNormGpu, _curBuf, _curBuf, _embd, _w.RmsNormEps, 0, 0, hasWeight: true);
        _ctx.UavBarrier(_curBuf);

        int blocksCount = 0;

        // Prelude: physical layers 0 .. CoreStart-1
        for (int il = 0; il < _w.CoreStart; il++)
        {
            ExecuteLayerGpu(ref cmd, il, il, pos, tau, ref blocksCount, hasMessages: false);
        }

        // Save prelude output & blocks on GPU (zero CPU involvement!)
        int preludeBlocksCount = blocksCount;
        GpuCopy(cmd, _preludeBuf, 0, _curBuf, 0, embdBytes, barrier: false);
        if (preludeBlocksCount > 0)
            GpuCopy(cmd, _preludeBlocksBuf, 0, _blocksBuf, 0, (ulong)preludeBlocksCount * embdBytes, barrier: false);
        _ctx.UavBarrier();

        // 3-Pass Looped Core
        int coreLen = _w.CoreEnd - _w.CoreStart;
        int bankCount = 0;

        for (int pass = 0; pass < _w.NumLoops; pass++)
        {
            if (pass > 0)
            {
                int bankStart = preludeBlocksCount + 1;
                bankCount = Math.Max(0, blocksCount - bankStart);
                if (bankCount > 0)
                    GpuCopy(cmd, _bankBuf, 0, _blocksBuf, (ulong)bankStart * embdBytes, (ulong)bankCount * embdBytes, barrier: false);
                GpuCopy(cmd, _bankOutBuf, 0, _curBuf, 0, embdBytes, barrier: false);
                _ctx.UavBarrier();

                GpuCopy(cmd, _curBuf, 0, _preludeBuf, 0, embdBytes, barrier: false);
                if (preludeBlocksCount > 0)
                    GpuCopy(cmd, _blocksBuf, 0, _preludeBlocksBuf, 0, (ulong)preludeBlocksCount * embdBytes, barrier: false);
                _ctx.UavBarrier();
                blocksCount = preludeBlocksCount;
            }

            for (int ip = _w.CoreStart; ip < _w.CoreEnd; ip++)
            {
                int il = pass == 0 ? ip : _nPhys + (pass - 1) * coreLen + (ip - _w.CoreStart);
                if (pass > 0)
                {
                    ComputeCrossPassMessagesGpu(cmd, ip, bankCount);
                }
                ExecuteLayerGpu(ref cmd, il, ip, pos, tau, ref blocksCount, hasMessages: pass > 0);
            }
        }

        // Epilogue: physical layers CoreEnd .. PhysicalLayerCount-1
        for (int il = _w.CoreEnd; il < _nPhys; il++)
        {
            ExecuteLayerGpu(ref cmd, il, il, pos, tau, ref blocksCount, hasMessages: false);
        }

        if (computeLogits && !logits.IsEmpty)
        {
            int vocabOut = Math.Min(_w.VocabSizeUnpadded, logits.Length);
            // Output head: Mix(blocks, cur, OutputResQuery) -> OutputNorm -> Q6_K lm_head -> Softcap
            DispatchMix(cmd, _outputResQueryGpu, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, 0, 0);
            _ctx.UavBarrier(_mixOutBuf);

            DispatchRmsNorm(cmd, _outputNormGpu, _mixOutBuf, _curBuf, _embd, _w.RmsNormEps, 0, 0, hasWeight: true);
            _ctx.UavBarrier(_curBuf);

            // Q6_K lm_head
            cmd.SetComputeRootSignature(_sigGemv);
            cmd.SetPipelineState(_psoQ6K);
            cmd.SetComputeRootShaderResourceView(1, _outputWeightGpu.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _curBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _logitsBuf.GPUVirtualAddress);

            uint* c = stackalloc uint[4];
            c[0] = (uint)_embd;
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
            _ctx.UavBarrier(_logitsBuf);

            bool useHeadLora = HeadLora != null;
            if (!useHeadLora && _w.FinalLogitSoftcapping > 0.0f)
            {
                cmd.SetComputeRootSignature(_sigSoftcap);
                cmd.SetPipelineState(_psoSoftcap);
                cmd.SetComputeRootUnorderedAccessView(1, _logitsBuf.GPUVirtualAddress);
                c[0] = (uint)vocabOut;
                c[1] = BitConverter.SingleToUInt32Bits(_w.FinalLogitSoftcapping);
                c[2] = BitConverter.SingleToUInt32Bits(1.0f / _w.FinalLogitSoftcapping);
                c[3] = 0u;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
                cmd.Dispatch((uint)((vocabOut + 255) / 256), 1, 1);
                _ctx.UavBarrier(_logitsBuf);
            }

            // Copy logits (and post-OutputNorm hidden state if HeadLora is active) in the SAME command list
            ulong outBytes = (ulong)vocabOut * 4UL;
            cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_readbackBuf, 0, _logitsBuf, 0, outBytes);
            cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            if (useHeadLora)
            {
                cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_cpuMoeReadbackBuf, 0, _curBuf, 0, (ulong)_embd * 4UL);
                cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            }

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            fixed (float* pDst = logits)
            {
                Buffer.MemoryCopy(_pReadback, pDst, outBytes, outBytes);
            }

            if (useHeadLora)
            {
                HeadLora!.ApplyDelta(new ReadOnlySpan<float>((float*)_pCpuMoeReadback, _embd), logits.Slice(0, vocabOut));
                Inkling2HeadLoraAdapter.ApplySoftcapInPlace(logits.Slice(0, vocabOut), _w.FinalLogitSoftcapping);
            }
        }
        else
        {
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
        }
    }

    /// <summary>
    /// Computes frozen GPU lm_head raw logits (before FinalLogitSoftcapping) for a post-OutputNorm hidden vector.
    /// Used by Glacier.Tune's Inkling2HeadLoraTrainer to compute exact base logits without re-running the 48-layer trunk.
    /// </summary>
    public void ComputeLmHeadRawLogits(ReadOnlySpan<float> hidden, Span<float> outRawLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (hidden.Length < _embd)
            throw new ArgumentException($"Hidden vector length {hidden.Length} < {_embd}.", nameof(hidden));
        int vocabOut = Math.Min(_w.VocabSizeUnpadded, outRawLogits.Length);
        if (vocabOut <= 0) return;

        ulong embdBytes = (ulong)_embd * 4UL;
        fixed (float* pH = hidden)
        {
            Buffer.MemoryCopy(pH, _pUpload, embdBytes, embdBytes);
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_curBuf, 0, _uploadBuf, 0, embdBytes);
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        cmd.SetComputeRootSignature(_sigGemv);
        cmd.SetPipelineState(_psoQ6K);
        cmd.SetComputeRootShaderResourceView(1, _outputWeightGpu.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _curBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _logitsBuf.GPUVirtualAddress);

        uint* c = stackalloc uint[4];
        c[0] = (uint)_embd;
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
        _ctx.UavBarrier(_logitsBuf);

        ulong outBytes = (ulong)vocabOut * 4UL;
        cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.CopyBufferRegion(_readbackBuf, 0, _logitsBuf, 0, outBytes);
        cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        fixed (float* pDst = outRawLogits)
        {
            Buffer.MemoryCopy(_pReadback, pDst, outBytes, outBytes);
        }
    }

    private void DebugDumpCur(ref ID3D12GraphicsCommandList cmd, string tag, ulong embdBytes)
    {
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.CopyBufferRegion(_readbackBuf, 0, _curBuf, 0, embdBytes);
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        var rem = _ctx.Device.DeviceRemovedReason;
        Console.Error.WriteLine($"[GPU {tag}] rem=0x{rem.Code:X8} cur[0..3]={_pReadback[0]:F4}, {_pReadback[1]:F4}, {_pReadback[2]:F4}, {_pReadback[3]:F4}");
        _ctx.BeginCommands();
        cmd = _ctx.CommandList;
    }

    private void ComputeCrossPassMessagesGpu(ID3D12GraphicsCommandList cmd, int ip, int bankCount)
    {
        var lg = _layers[ip];
        if (lg.LoopMsgQuery != null)
        {
            DispatchMix(cmd, lg.LoopMsgQuery, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 4, 0, 0, 0);
            _ctx.UavBarrier(_msgBuf);
            return;
        }

        int cIdx = ip - _w.CoreStart;
        if (_loopResQueryGpu != null && _loopKvResQueryGpu != null)
        {
            DispatchMix(cmd, _loopResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx) * _embd, 0 * _embd, 0);
            DispatchMix(cmd, _loopKvResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx) * _embd, 1 * _embd, 0);
            DispatchMix(cmd, _loopKvResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx + 1) * _embd, 2 * _embd, 0);
            DispatchMix(cmd, _loopResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx + 1) * _embd, 3 * _embd, 0);
            _ctx.UavBarrier(_msgBuf);
        }
    }

    private void ApplyLoopControlGpu(ID3D12GraphicsCommandList cmd, ID3D12Resource targetBuf, int msgOffsetFloats, LayerGpu lg, int ctrl)
    {
        // RMSNorm(targetBuf, 1e-6) -> _ctrlCatBuf[0..embd), RMSNorm(_msgBuf+msgOffset, 1e-6) -> _ctrlCatBuf[embd..2*embd)
        DispatchRmsNorm(cmd, _dummySrv, targetBuf, _ctrlCatBuf, _embd, 1e-6f, 0, 0, hasWeight: false);
        DispatchRmsNorm(cmd, _dummySrv, _msgBuf, _ctrlCatBuf, _embd, 1e-6f, msgOffsetFloats, _embd, hasWeight: false);
        _ctx.UavBarrier(_ctrlCatBuf);

        // Down -> Gain & Forget
        DispatchGemvF32(cmd, lg.LoopCtrlDown[ctrl]!, _ctrlCatBuf, _ctrlGateBuf, 2 * _embd, _w.GateRank);
        _ctx.UavBarrier(_ctrlGateBuf);

        DispatchGemvF32(cmd, lg.LoopCtrlGain[ctrl]!, _ctrlGateBuf, _ctrlGainBuf, _w.GateRank, _embd);
        DispatchGemvF32(cmd, lg.LoopCtrlForget[ctrl]!, _ctrlGateBuf, _ctrlForgetBuf, _w.GateRank, _embd);
        _ctx.UavBarrier();

        // Blend: target = (1 - tanh(forget)) * target + tanh(gain) * msg
        cmd.SetComputeRootSignature(_sigLoopBlend);
        cmd.SetPipelineState(_psoLoopBlend);
        cmd.SetComputeRootUnorderedAccessView(1, targetBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _msgBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _ctrlGainBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _ctrlForgetBuf.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)_embd;
        c[1] = (uint)msgOffsetFloats;
        c[2] = 0u;
        c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((_embd + 255) / 256), 1, 1);
        _ctx.UavBarrier(targetBuf);
    }

    private void ExecuteLayerGpu(ref ID3D12GraphicsCommandList cmd, int il, int ip, int pos, float tau, ref int blocksCount, bool hasMessages)
    {
        var l = _w.PhysicalLayers[ip];
        var lg = _layers[ip];
        bool isSwa = l.IsSwa;
        int nHead = l.HeadCount;
        int nHeadKv = l.HeadCountKv;
        int headDim = l.HeadDim;
        int qDim = nHead * headDim;
        int kvDim = nHeadKv * headDim;
        int nRel = _w.RelativeLength;
        int rDim = nRel * nHead;
        ulong embdBytes = (ulong)_embd * 4UL;

        // Save inpSa = cur
        GpuCopy(cmd, _inpSaBuf, 0, _curBuf, 0, embdBytes);

        // 1. Attention inputs via Block-Attention Residual Mix + LoopControl + RMSNorm (zero redundant copies!)
        ID3D12Resource kInBuf = _in1Buf;
        ID3D12Resource vInBuf = _in2Buf;

        if (isSwa && !hasMessages)
        {
            if (lg.AttnResQuery != null)
            {
                DispatchMix(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, 0, 0);
                _ctx.UavBarrier(_mixOutBuf);
                DispatchRmsNorm(cmd, lg.AttnNorm!, _mixOutBuf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
            }
            else
            {
                DispatchRmsNorm(cmd, lg.AttnNorm!, _curBuf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
            }
            _ctx.UavBarrier(_in0Buf);
            kInBuf = _in0Buf;
            vInBuf = _in0Buf;
        }
        else if (!isSwa && !hasMessages)
        {
            if (lg.AttnResQuery != null)
            {
                DispatchMix(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 3, 0, 0, 0);
                _ctx.UavBarrier(_mixOutBuf);
                DispatchRmsNorm(cmd, lg.AttnQInpNorm!, _mixOutBuf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnKInpNorm!, _mixOutBuf, _in1Buf, _embd, _w.RmsNormEps, _embd, 0, true);
                DispatchRmsNorm(cmd, lg.AttnVInpNorm!, _mixOutBuf, _in2Buf, _embd, _w.RmsNormEps, 2 * _embd, 0, true);
            }
            else
            {
                DispatchRmsNorm(cmd, lg.AttnQInpNorm!, _curBuf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnKInpNorm!, _curBuf, _in1Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnVInpNorm!, _curBuf, _in2Buf, _embd, _w.RmsNormEps, 0, 0, true);
            }
            _ctx.UavBarrier();
        }
        else
        {
            if (lg.AttnResQuery != null)
            {
                int nQ = isSwa ? 1 : 3;
                DispatchMix(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, nQ, 0, 0, 0);
                _ctx.UavBarrier(_mixOutBuf);
                GpuCopy(cmd, _in0Buf, 0, _mixOutBuf, 0, embdBytes, barrier: false);
                GpuCopy(cmd, _in1Buf, 0, _mixOutBuf, isSwa ? 0UL : embdBytes, embdBytes, barrier: false);
                GpuCopy(cmd, _in2Buf, 0, _mixOutBuf, isSwa ? 0UL : 2UL * embdBytes, embdBytes, barrier: false);
            }
            else
            {
                GpuCopy(cmd, _in0Buf, 0, _curBuf, 0, embdBytes, barrier: false);
                GpuCopy(cmd, _in1Buf, 0, _curBuf, 0, embdBytes, barrier: false);
                GpuCopy(cmd, _in2Buf, 0, _curBuf, 0, embdBytes, barrier: false);
            }
            _ctx.UavBarrier();

            ApplyLoopControlGpu(cmd, _in0Buf, 0 * _embd, lg, 0);
            ApplyLoopControlGpu(cmd, _in1Buf, 1 * _embd, lg, 1);
            ApplyLoopControlGpu(cmd, _in2Buf, 2 * _embd, lg, 2);

            if (isSwa)
            {
                DispatchRmsNorm(cmd, lg.AttnNorm!, _in0Buf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnNorm!, _in1Buf, _in1Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnNorm!, _in2Buf, _in2Buf, _embd, _w.RmsNormEps, 0, 0, true);
            }
            else
            {
                DispatchRmsNorm(cmd, lg.AttnQInpNorm!, _in0Buf, _in0Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnKInpNorm!, _in1Buf, _in1Buf, _embd, _w.RmsNormEps, 0, 0, true);
                DispatchRmsNorm(cmd, lg.AttnVInpNorm!, _in2Buf, _in2Buf, _embd, _w.RmsNormEps, 0, 0, true);
            }
            _ctx.UavBarrier();
        }

        // 2. Parallel Attention Projections: Wq, Wk, Wv, AttnRel, AttnGate, Sinks (+ DynConv on Global) — ZERO intermediate barriers!
        DispatchGemvByType(cmd, l.WqType, lg.Wq, _in0Buf, _qBuf, _embd, qDim);
        DispatchGemvByType(cmd, l.WkType, lg.Wk, kInBuf, _kBuf, _embd, kvDim);
        DispatchGemvByType(cmd, l.WvType, lg.Wv, vInBuf, _vBuf, _embd, kvDim);
        DispatchGemvByType(cmd, l.AttnRelType, lg.AttnRel, _in0Buf, _rBuf, _embd, rDim);
        DispatchGemvByType(cmd, l.AttnGateType, lg.AttnGate, _in0Buf, _gateBuf, _embd, qDim);
        DispatchGemvF32(cmd, lg.AttnSinkProj, _in0Buf, _sinksBuf, _embd, nHead);

        bool hasDyn = !isSwa && lg.AttnKConvProj != null && lg.AttnVConvProj != null;
        if (hasDyn)
        {
            int dynDim = _w.ShortConvLCache * nHeadKv;
            DispatchGemvF32(cmd, lg.AttnKConvProj!, kInBuf, _kDynBuf, _embd, dynDim);
            DispatchGemvF32(cmd, lg.AttnVConvProj!, vInBuf, _vDynBuf, _embd, dynDim);
        }
        _ctx.UavBarrier(); // single global UAV barrier for all 6-8 parallel projections

        // Short-Conv on K & V
        DispatchShortConv(cmd, lg.AttnKConv, _kBuf, _kConvHistBuf, hasDyn ? _kDynBuf : _dummyUav, kvDim, _kvHistBaseFloats[il], hasDyn, headDim);
        DispatchShortConv(cmd, lg.AttnVConv, _vBuf, _vConvHistBuf, hasDyn ? _vDynBuf : _dummyUav, kvDim, _kvHistBaseFloats[il], hasDyn, headDim);
        _ctx.UavBarrier();

        // Per-head L2 Clip (Q, K, R) + V RMSNorm + tau scaling + FP16 KV cache write
        int cap = _kvCap[il];
        uint kvSlotU32 = _kvLayerBaseU32[il] + (uint)(pos % cap) * (uint)(kvDim / 2);
        float qTau = isSwa ? 1.0f : tau;

        DispatchHeadNormClipKv(cmd, lg.AttnQClip, _qBuf, _dummyUav, headDim, nHead, mode: 0, _w.QkClip, qTau, writeKv: false, 0);
        DispatchHeadNormClipKv(cmd, null, _kBuf, _kCacheFp16, headDim, nHeadKv, mode: 0, _w.QkClip, 1.0f, writeKv: true, kvSlotU32);
        DispatchHeadNormClipKv(cmd, null, _vBuf, _vCacheFp16, headDim, nHeadKv, mode: 1, _w.RmsNormEps, 1.0f, writeKv: true, kvSlotU32);
        DispatchHeadNormClipKv(cmd, lg.AttnRelClip, _rBuf, _dummyUav, nRel, nHead, mode: 0, _w.RelativeClip, qTau, writeKv: false, 0);
        _ctx.UavBarrier();

        // 3. Online FlashAttention-style Relative Positional Attention + Sinks + Sigmoid Output Gate (24 parallel heads)
        int startT = isSwa ? Math.Max(0, pos - _w.SlidingWindow + 1) : 0;
        cmd.SetComputeRootSignature(_sigAttn);
        cmd.SetPipelineState(_psoAttn);
        cmd.SetComputeRootShaderResourceView(1, lg.AttnRelBias.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _qBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _rBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _sinksBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, _gateBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(6, _kCacheFp16.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(7, _vCacheFp16.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(8, _attnOutBuf.GPUVirtualAddress);

        uint* ac = stackalloc uint[8];
        ac[0] = (uint)pos;
        ac[1] = (uint)startT;
        ac[2] = (uint)cap;
        ac[3] = (uint)(kvDim / 2);
        ac[4] = (uint)(nHead / nHeadKv);
        ac[5] = (uint)l.RelExtent;
        ac[6] = (uint)nRel;
        ac[7] = _kvLayerBaseU32[il];
        cmd.SetComputeRoot32BitConstants(0, 8, (IntPtr)ac, 0);
        cmd.Dispatch((uint)nHead, 1, 1);
        _ctx.UavBarrier(_attnOutBuf);

        // Wo projection + AttnPostConv
        DispatchGemvByType(cmd, l.WoType, lg.Wo, _attnOutBuf, _attnProjBuf, qDim, _embd);
        _ctx.UavBarrier(_attnProjBuf);

        DispatchShortConv(cmd, lg.AttnPostConv, _attnProjBuf, _attnPostHistBuf, _dummyUav, _embd, _embdHistBaseFloats[il], false, 0);
        _ctx.UavBarrier(_attnProjBuf);

        // Block residual stream boundary
        if (ip % _w.AttnResBlockSize == 0)
        {
            GpuCopy(cmd, _blocksBuf, (ulong)blocksCount * embdBytes, _inpSaBuf, 0, embdBytes, barrier: false);
            blocksCount++;
            GpuCopy(cmd, _curBuf, 0, _attnProjBuf, 0, embdBytes, barrier: false);
            _ctx.UavBarrier();
        }
        else
        {
            DispatchVecAdd(cmd, _inpSaBuf, _attnProjBuf, _curBuf, _embd);
            _ctx.UavBarrier(_curBuf);
        }

        // Save ffnInp = cur
        GpuCopy(cmd, _ffnInpBuf, 0, _curBuf, 0, embdBytes);

        // 4. MoE FFN: Mix -> LoopCtrl -> FfnNorm -> Router -> TopK -> 8 Experts -> FfnPostConv -> Residual Add
        DispatchMix(cmd, lg.FfnResQuery!, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, 0, 0);
        _ctx.UavBarrier(_mixOutBuf);

        if (hasMessages)
        {
            GpuCopy(cmd, _curBuf, 0, _mixOutBuf, 0, embdBytes);
            ApplyLoopControlGpu(cmd, _curBuf, 3 * _embd, lg, 3);
            DispatchRmsNorm(cmd, lg.FfnNorm!, _curBuf, _curBuf, _embd, _w.RmsNormEps, 0, 0, true);
        }
        else
        {
            DispatchRmsNorm(cmd, lg.FfnNorm!, _mixOutBuf, _curBuf, _embd, _w.RmsNormEps, 0, 0, true);
        }
        _ctx.UavBarrier(_curBuf);

        // Router GEMV + Top-8 selection directly on GPU
        DispatchGemvF32(cmd, lg.FfnGateInp, _curBuf, _routerLogitsBuf, _embd, _w.ExpertCount);
        _ctx.UavBarrier(_routerLogitsBuf);

        cmd.SetComputeRootSignature(_sigRouter);
        cmd.SetPipelineState(_psoRouter);
        cmd.SetComputeRootShaderResourceView(1, (lg.ExpProbsBias ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, (lg.GuScales ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(3, (lg.DnScales ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _routerLogitsBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, _topkIdsBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(6, _metaBuf.GPUVirtualAddress);

        uint* rc = stackalloc uint[6];
        rc[0] = (uint)_w.ExpertCount;
        rc[1] = (uint)_topK;
        rc[2] = BitConverter.SingleToUInt32Bits(_w.ExpertWeightsScale);
        rc[3] = lg.ExpProbsBias != null ? 1u : 0u;
        rc[4] = lg.GuScales != null ? 1u : 0u;
        rc[5] = lg.DnScales != null ? 1u : 0u;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)rc, 0);
        cmd.Dispatch(1, 1, 1);
        _ctx.UavBarrier();

        if (lg.MoePinned)
        {
            // 100% GPU-RESIDENT 2D-BATCHED INDIRECT NVFP4 MoE (ALL 8 experts in ONE GateUp + ONE Down dispatch!)
            cmd.SetComputeRootSignature(_sigNvfp4Ind);
            cmd.SetPipelineState(_psoNvfp4Ind);
            cmd.SetComputeRootShaderResourceView(1, lg.GateUpExps!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _curBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _guBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _topkIdsBuf.GPUVirtualAddress);

            uint guGroups = (uint)((2 * _ffExp + 15) / 16);
            uint* mc = stackalloc uint[6];
            mc[0] = (uint)_embd;
            mc[1] = (uint)(2 * _ffExp);
            mc[2] = 0u; // x_stride = 0 (all 8 experts read the same _curBuf[0..embd))
            mc[3] = (uint)(2 * _ffExp); // y_stride
            mc[4] = lg.GateUpExpStrideBytes;
            mc[5] = 0u;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)mc, 0);
            cmd.Dispatch(guGroups, (uint)_topK, 1);
            _ctx.UavBarrier(_guBuf);

            // SiTU across all 8 experts in 1 dispatch
            cmd.SetComputeRootSignature(_sigReduce);
            cmd.SetPipelineState(_psoSituBatch);
            cmd.SetComputeRootUnorderedAccessView(1, _guBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
            uint* uc = stackalloc uint[4];
            uc[0] = (uint)_ffExp;
            uc[1] = (uint)_topK;
            uc[2] = BitConverter.SingleToUInt32Bits(_w.SituBeta);
            uc[3] = BitConverter.SingleToUInt32Bits(_w.SituLinearBeta);
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)uc, 0);
            cmd.Dispatch((uint)((_topK * _ffExp + 255) / 256), 1, 1);
            _ctx.UavBarrier(_hidBuf);

            // Down NVFP4 across all 8 experts in ONE 2D dispatch!
            cmd.SetComputeRootSignature(_sigNvfp4Ind);
            cmd.SetPipelineState(_psoNvfp4Ind);
            cmd.SetComputeRootShaderResourceView(1, lg.DownExps!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _dnBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _topkIdsBuf.GPUVirtualAddress);

            uint dnGroups = (uint)((_embd + 15) / 16);
            mc[0] = (uint)_ffExp;
            mc[1] = (uint)_embd;
            mc[2] = (uint)_ffExp; // x_stride
            mc[3] = (uint)_embd;  // y_stride
            mc[4] = lg.DownExpStrideBytes;
            mc[5] = 0u;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)mc, 0);
            cmd.Dispatch(dnGroups, (uint)_topK, 1);
            _ctx.UavBarrier(_dnBuf);

            // WeightedSum8 into _moeOutBuf
            cmd.SetComputeRootSignature(_sigReduce);
            cmd.SetPipelineState(_psoWeightedSum);
            cmd.SetComputeRootUnorderedAccessView(1, _dnBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _moeOutBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
            uc[0] = (uint)_embd;
            uc[1] = (uint)_topK;
            uc[2] = (uint)_topK;
            uc[3] = 1u;
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)uc, 0);
            cmd.Dispatch((uint)((_embd + 255) / 256), 1, 1);
            _ctx.UavBarrier(_moeOutBuf);
        }
        else
        {
            // Seamless AVX-512 CPU fallback ONLY for unpinned MoE layers (8GB dGPU)
            ulong topkBytes = (ulong)_topK * 4UL;
            ulong metaBytes = (ulong)(2 * _topK) * 4UL;

            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.ResourceBarrierTransition(_topkIdsBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, 0, _curBuf, 0, embdBytes);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, embdBytes, _topkIdsBuf, 0, topkBytes);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, embdBytes + topkBytes, _metaBuf, 0, metaBytes);
            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(_topkIdsBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            fixed (float* pCur = _cpuCur)
                Buffer.MemoryCopy(_pCpuMoeReadback, pCur, embdBytes, embdBytes);
            uint* pIds = (uint*)(_pCpuMoeReadback + embdBytes);
            float* pMeta = (float*)(_pCpuMoeReadback + embdBytes + topkBytes);
            for (int k = 0; k < _topK; k++)
            {
                int eid = (int)pIds[k];
                if ((uint)eid >= (uint)_w.ExpertCount)
                {
                    var rem = _ctx.Device.DeviceRemovedReason;
                    throw new InvalidOperationException($"Invalid MoE expert id {eid} (ExpertCount={_w.ExpertCount}, k={k}, DeviceRemoved=0x{rem.Code:X8}).");
                }
                _cpuActiveIds[k] = eid;
                _cpuActiveWeights[k] = pMeta[_topK + k]; // already includes weight * dnScale!
            }

            ExecuteCpuMoeFallback(l);

            fixed (float* pOut = _cpuMoeOut)
                Buffer.MemoryCopy(pOut, _pUpload, embdBytes, embdBytes);

            _ctx.BeginCommands();
            cmd = _ctx.CommandList;
            cmd.ResourceBarrierTransition(_moeOutBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_moeOutBuf, 0, _uploadBuf, 0, embdBytes);
            cmd.ResourceBarrierTransition(_moeOutBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        }

        // FfnPostConv + Residual Add
        DispatchShortConv(cmd, lg.FfnPostConv, _moeOutBuf, _ffnPostHistBuf, _dummyUav, _embd, _embdHistBaseFloats[il], false, 0);
        _ctx.UavBarrier(_moeOutBuf);

        DispatchVecAdd(cmd, _ffnInpBuf, _moeOutBuf, _curBuf, _embd);
        _ctx.UavBarrier(_curBuf);
    }

    private void ExecuteCpuMoeFallback(Inkling2LayerWeights layer)
    {
        int embd = _embd;
        int ffExp = _ffExp;
        int topK = _topK;
        long guRowBytes = GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, embd);
        long dnRowBytes = GgufTypes.GetRowBytes(layer.FfnDownExpsType, ffExp);
        long guExpBytes = (long)(2 * ffExp) * guRowBytes;
        long dnExpBytes = (long)embd * dnRowBytes;
        float beta = _w.SituBeta;
        float invBeta = 1.0f / beta;
        float lb = _w.SituLinearBeta;
        float invLb = lb > 0.0f ? 1.0f / lb : 0.0f;

        int guChunks = Math.Max(1, (Environment.ProcessorCount + topK - 1) / topK);
        int guChunkSize = (ffExp + guChunks - 1) / guChunks;
        nint guBase = (nint)layer.FfnGateUpExps;
        nint guScaleBase = (nint)layer.FfnGateUpExpsScale;
        var guType = layer.FfnGateUpExpsType;
        var dnType = layer.FfnDownExpsType;
        bool useVnniGu = _useVnniMoe && guType == GgufType.NVFP4;
        bool useVnniDn = _useVnniMoe && dnType == GgufType.NVFP4;

        fixed (float* pX = _cpuCur)
        fixed (BlockQ8NVFP4* pQ8X = _cpuQ8Cur)
        {
            if (useVnniGu)
            {
                QuantKernels.QuantizeXForNVFP4(pX, pQ8X, embd);
            }

            nint xAddr = (nint)pX;
            nint q8XAddr = (nint)pQ8X;
            Parallel.For(0, topK * guChunks, taskIdx =>
            {
                int k = taskIdx / guChunks;
                int chunk = taskIdx - k * guChunks;
                int jStart = chunk * guChunkSize;
                int jEnd = Math.Min(ffExp, jStart + guChunkSize);
                if (jStart >= jEnd) return;

                int expId = _cpuActiveIds[k];
                float guScale = guScaleBase != 0 ? ((float*)guScaleBase)[expId] : 1.0f;
                byte* gRow = (byte*)guBase + expId * guExpBytes + jStart * guRowBytes;
                byte* uRow = (byte*)guBase + expId * guExpBytes + (ffExp + jStart) * guRowBytes;
                float[] hid = _cpuExpHidden[k];
                int j = jStart;

                if (guType == GgufType.NVFP4)
                {
                    if (useVnniGu)
                    {
                        for (; j + 1 < jEnd; j += 2)
                        {
                            QuantKernels.VecDotNVFP4x4Q8(
                                (BlockNVFP4*)gRow,
                                (BlockNVFP4*)(gRow + guRowBytes),
                                (BlockNVFP4*)uRow,
                                (BlockNVFP4*)(uRow + guRowBytes),
                                (BlockQ8NVFP4*)q8XAddr,
                                embd,
                                out float g0,
                                out float g1,
                                out float u0,
                                out float u1);
                            g0 *= guScale; g1 *= guScale;
                            u0 *= guScale; u1 *= guScale;
                            float act0 = beta * MathF.Tanh(g0 * invBeta) / (1.0f + MathF.Exp(-g0));
                            float act1 = beta * MathF.Tanh(g1 * invBeta) / (1.0f + MathF.Exp(-g1));
                            if (lb > 0.0f)
                            {
                                u0 = lb * MathF.Tanh(u0 * invLb);
                                u1 = lb * MathF.Tanh(u1 * invLb);
                            }
                            hid[j] = act0 * u0;
                            hid[j + 1] = act1 * u1;
                            gRow += 2 * guRowBytes;
                            uRow += 2 * guRowBytes;
                        }
                        for (; j < jEnd; j++)
                        {
                            float g = QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)gRow, (BlockQ8NVFP4*)q8XAddr, embd) * guScale;
                            float u = QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)uRow, (BlockQ8NVFP4*)q8XAddr, embd) * guScale;
                            float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                            if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                            hid[j] = act * u;
                            gRow += guRowBytes;
                            uRow += guRowBytes;
                        }
                    }
                    else
                    {
                        for (; j + 1 < jEnd; j += 2)
                        {
                            QuantKernels.VecDotNVFP4x4(
                                (BlockNVFP4*)gRow,
                                (BlockNVFP4*)(gRow + guRowBytes),
                                (BlockNVFP4*)uRow,
                                (BlockNVFP4*)(uRow + guRowBytes),
                                (float*)xAddr,
                                embd,
                                out float g0,
                                out float g1,
                                out float u0,
                                out float u1);
                            g0 *= guScale; g1 *= guScale;
                            u0 *= guScale; u1 *= guScale;
                            float act0 = beta * MathF.Tanh(g0 * invBeta) / (1.0f + MathF.Exp(-g0));
                            float act1 = beta * MathF.Tanh(g1 * invBeta) / (1.0f + MathF.Exp(-g1));
                            if (lb > 0.0f)
                            {
                                u0 = lb * MathF.Tanh(u0 * invLb);
                                u1 = lb * MathF.Tanh(u1 * invLb);
                            }
                            hid[j] = act0 * u0;
                            hid[j + 1] = act1 * u1;
                            gRow += 2 * guRowBytes;
                            uRow += 2 * guRowBytes;
                        }
                    }
                }

                for (; j < jEnd; j++)
                {
                    float g = QuantKernels.ComputeDot(guType, gRow, (float*)xAddr, null, embd) * guScale;
                    float u = QuantKernels.ComputeDot(guType, uRow, (float*)xAddr, null, embd) * guScale;
                    float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                    if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                    hid[j] = act * u;
                    gRow += guRowBytes;
                    uRow += guRowBytes;
                }
            });
        }

        if (useVnniDn)
        {
            for (int k = 0; k < topK; k++)
            {
                fixed (float* pHid = _cpuExpHidden[k])
                fixed (BlockQ8NVFP4* pQ8Hid = _cpuQ8ExpHidden[k])
                {
                    QuantKernels.QuantizeXForNVFP4(pHid, pQ8Hid, ffExp);
                }
            }
        }

        int nTiles = Math.Max(8, Environment.ProcessorCount);
        int tileSize = (embd + nTiles - 1) / nTiles;
        nint dnBase = (nint)layer.FfnDownExps;

        fixed (float* pOut = _cpuMoeOut)
        {
            nint outAddr = (nint)pOut;
            Parallel.For(0, nTiles, tile =>
            {
                int dStart = tile * tileSize;
                int dEnd = Math.Min(embd, dStart + tileSize);
                if (dStart >= dEnd) return;

                float* dst = (float*)outAddr;
                for (int d = dStart; d < dEnd; d++) dst[d] = 0.0f;

                for (int k = 0; k < topK; k++)
                {
                    int expId = _cpuActiveIds[k];
                    float w = _cpuActiveWeights[k];
                    byte* dnRow = (byte*)dnBase + expId * dnExpBytes + dStart * dnRowBytes;
                    int d = dStart;

                    if (useVnniDn)
                    {
                        fixed (BlockQ8NVFP4* q8HidPtr = _cpuQ8ExpHidden[k])
                        {
                            for (; d + 3 < dEnd; d += 4)
                            {
                                QuantKernels.VecDotNVFP4x4Q8(
                                    (BlockNVFP4*)dnRow,
                                    (BlockNVFP4*)(dnRow + dnRowBytes),
                                    (BlockNVFP4*)(dnRow + 2 * dnRowBytes),
                                    (BlockNVFP4*)(dnRow + 3 * dnRowBytes),
                                    q8HidPtr,
                                    ffExp,
                                    out float r0,
                                    out float r1,
                                    out float r2,
                                    out float r3);
                                dst[d + 0] += w * r0;
                                dst[d + 1] += w * r1;
                                dst[d + 2] += w * r2;
                                dst[d + 3] += w * r3;
                                dnRow += 4 * dnRowBytes;
                            }
                            for (; d < dEnd; d++)
                            {
                                dst[d] += w * QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)dnRow, q8HidPtr, ffExp);
                                dnRow += dnRowBytes;
                            }
                        }
                    }
                    else
                    {
                        fixed (float* hidPtr = _cpuExpHidden[k])
                        {
                            if (dnType == GgufType.NVFP4)
                            {
                                for (; d + 3 < dEnd; d += 4)
                                {
                                    QuantKernels.VecDotNVFP4x4(
                                        (BlockNVFP4*)dnRow,
                                        (BlockNVFP4*)(dnRow + dnRowBytes),
                                        (BlockNVFP4*)(dnRow + 2 * dnRowBytes),
                                        (BlockNVFP4*)(dnRow + 3 * dnRowBytes),
                                        hidPtr,
                                        ffExp,
                                        out float r0,
                                        out float r1,
                                        out float r2,
                                        out float r3);
                                    dst[d + 0] += w * r0;
                                    dst[d + 1] += w * r1;
                                    dst[d + 2] += w * r2;
                                    dst[d + 3] += w * r3;
                                    dnRow += 4 * dnRowBytes;
                                }
                            }

                            for (; d < dEnd; d++)
                            {
                                dst[d] += w * QuantKernels.ComputeDot(dnType, dnRow, hidPtr, null, ffExp);
                                dnRow += dnRowBytes;
                            }
                        }
                    }
                }
            });
        }
    }

    // ---- Low-level D3D12 Dispatch Helpers ----

    private void GpuCopy(ID3D12GraphicsCommandList cmd, ID3D12Resource dst, ulong dstOffBytes, ID3D12Resource src, ulong srcOffBytes, ulong bytes, bool barrier = true)
    {
        uint count = (uint)(bytes >> 2);
        if (count == 0u) return;
        cmd.SetComputeRootSignature(_sigReduce);
        cmd.SetPipelineState(_psoUavCopy);
        cmd.SetComputeRootUnorderedAccessView(1, src.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dst.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _dummyUav.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = count;
        c[1] = (uint)(srcOffBytes >> 2);
        c[2] = (uint)(dstOffBytes >> 2);
        c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((count + 255u) / 256u, 1, 1);
        if (barrier) _ctx.UavBarrier(dst);
    }

    private void DispatchGemvByType(ID3D12GraphicsCommandList cmd, GgufType type, ID3D12Resource w, ID3D12Resource x, ID3D12Resource y, int kCols, int mRows)
    {
        var (pso, rowsPerGroup) = type switch
        {
            GgufType.Q8_0 => (_psoQ8_0, 16),
            GgufType.Q6_K => (_psoQ6K, 16),
            GgufType.Q5_K => (_psoQ5K, 8),
            GgufType.F32 => (_psoF32, 16),
            _ => throw new NotSupportedException($"Inkling2D3D12Model does not support attention quant type {type}.")
        };
        cmd.SetComputeRootSignature(_sigGemv);
        cmd.SetPipelineState(pso);
        cmd.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, y.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)kCols; c[1] = (uint)mRows; c[2] = 0u; c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((mRows + rowsPerGroup - 1) / rowsPerGroup), 1, 1);
    }

    private void DispatchGemvF32(ID3D12GraphicsCommandList cmd, ID3D12Resource w, ID3D12Resource x, ID3D12Resource y, int kCols, int mRows)
    {
        cmd.SetComputeRootSignature(_sigGemv);
        cmd.SetPipelineState(_psoF32);
        cmd.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, y.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)kCols; c[1] = (uint)mRows; c[2] = 0u; c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((mRows + 15) / 16), 1, 1);
    }

    private void DispatchRmsNorm(ID3D12GraphicsCommandList cmd, ID3D12Resource w, ID3D12Resource src, ID3D12Resource dst, int dim, float eps, int inOff, int outOff, bool hasWeight)
    {
        cmd.SetComputeRootSignature(_sigRmsNorm);
        cmd.SetPipelineState(_psoRmsNorm);
        cmd.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, src.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);
        uint* c = stackalloc uint[6];
        c[0] = (uint)dim;
        c[1] = BitConverter.SingleToUInt32Bits(eps);
        c[2] = (uint)inOff;
        c[3] = (uint)outOff;
        c[4] = hasWeight ? 1u : 0u;
        c[5] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
        cmd.Dispatch(1, 1, 1);
    }

    private void DispatchMix(ID3D12GraphicsCommandList cmd, ID3D12Resource queries, ID3D12Resource blocksBuf, ID3D12Resource curBuf, ID3D12Resource outBuf, int nBlocks, int nQueries, int qBaseFloats, int outBaseFloats, int blocksStartIdx)
    {
        cmd.SetComputeRootSignature(_sigMix);
        cmd.SetPipelineState(_psoMix);
        cmd.SetComputeRootShaderResourceView(1, queries.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, blocksBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, curBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, outBuf.GPUVirtualAddress);
        uint* c = stackalloc uint[6];
        c[0] = (uint)_embd;
        c[1] = (uint)nBlocks;
        c[2] = (uint)qBaseFloats;
        c[3] = (uint)outBaseFloats;
        c[4] = (uint)blocksStartIdx;
        c[5] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
        cmd.Dispatch((uint)nQueries, 1, 1);
    }

    private void DispatchShortConv(ID3D12GraphicsCommandList cmd, ID3D12Resource convW, ID3D12Resource cur, ID3D12Resource hist, ID3D12Resource dyn, int channels, uint histBaseFloats, bool hasDyn, int headDim)
    {
        cmd.SetComputeRootSignature(_sigShortConv);
        cmd.SetPipelineState(_psoShortConv);
        cmd.SetComputeRootShaderResourceView(1, convW.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, cur.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, hist.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dyn.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)channels;
        c[1] = histBaseFloats;
        c[2] = hasDyn ? 1u : 0u;
        c[3] = (uint)headDim;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((channels + 255) / 256), 1, 1);
    }

    private void DispatchHeadNormClipKv(ID3D12GraphicsCommandList cmd, ID3D12Resource? scale, ID3D12Resource buf, ID3D12Resource kvFp16, int dim, int nHeads, uint mode, float threshOrEps, float tau, bool writeKv, uint kvSlotU32)
    {
        cmd.SetComputeRootSignature(_sigHeadNorm);
        cmd.SetPipelineState(_psoHeadNorm);
        cmd.SetComputeRootShaderResourceView(1, (scale ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, buf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, kvFp16.GPUVirtualAddress);
        uint* c = stackalloc uint[8];
        c[0] = (uint)dim;
        c[1] = mode;
        c[2] = scale != null ? 1u : 0u;
        c[3] = BitConverter.SingleToUInt32Bits(threshOrEps);
        c[4] = BitConverter.SingleToUInt32Bits(tau);
        c[5] = writeKv ? 1u : 0u;
        c[6] = kvSlotU32;
        c[7] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 8, (IntPtr)c, 0);
        cmd.Dispatch((uint)nHeads, 1, 1);
    }

    private void DispatchVecAdd(ID3D12GraphicsCommandList cmd, ID3D12Resource a, ID3D12Resource b, ID3D12Resource dst, int count)
    {
        cmd.SetComputeRootSignature(_sigReduce);
        cmd.SetPipelineState(_psoVecAdd);
        cmd.SetComputeRootUnorderedAccessView(1, a.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, b.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)count; c[1] = 0u; c[2] = 0u; c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((count + 255) / 256), 1, 1);
    }

    /// <summary>Gets whether a prefix state checkpoint is currently held in host memory.</summary>
    public bool HasPrefixCheckpoint => _hasCheckpoint;

    /// <summary>
    /// Captures all Short-Conv history buffers and SWA KV ring slices from GPU VRAM into host memory
    /// so subsequent turns extending this prefix can restore exact recurrent/ring state without re-prefilling.
    /// </summary>
    public void SavePrefixCheckpoint()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureCheckpointSetup();
        if (_ckptBatches!.Length == 0)
        {
            _hasCheckpoint = true;
            return;
        }
        if (_ckptReadbackBuf == null)
        {
            _ckptReadbackBuf = Track(_ctx.CreateReadbackBuffer((ulong)Math.Max(256, _ckptMaxBatchBytes)));
            _pCkptReadback = (byte*)MapPersistent(_ckptReadbackBuf);
        }

        for (int bi = 0; bi < _ckptBatches.Length; bi++)
        {
            var batch = _ckptBatches[bi];
            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;
            foreach (var res in batch.UniqueResources)
                cmd.ResourceBarrierTransition(res, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            foreach (var item in batch.Items)
                cmd.CopyBufferRegion(_ckptReadbackBuf, item.BatchOffsetBytes, item.Resource, item.ResourceOffsetBytes, (ulong)item.ByteCount);
            foreach (var res in batch.UniqueResources)
                cmd.ResourceBarrierTransition(res, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            new ReadOnlySpan<byte>(_pCkptReadback, batch.TotalBytes).CopyTo(_ckptHostBuf.AsSpan(batch.HostOffsetBytes, batch.TotalBytes));
        }

        _hasCheckpoint = true;
    }

    /// <summary>
    /// Restores the previously saved Short-Conv history buffers and SWA KV ring slices back to GPU VRAM.
    /// Non-SWA full-attention KV slots in [0, prefixLen) remain untouched by subsequent generation.
    /// </summary>
    public void RestorePrefixCheckpoint()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_hasCheckpoint || _ckptBatches == null || _ckptHostBuf == null)
            throw new InvalidOperationException("No prefix checkpoint has been saved.");
        if (_ckptBatches.Length == 0) return;

        if (_ckptUploadBuf == null)
        {
            _ckptUploadBuf = Track(_ctx.CreateUploadBuffer((ulong)Math.Max(256, _ckptMaxBatchBytes)));
            _pCkptUpload = (byte*)MapPersistent(_ckptUploadBuf);
        }

        for (int bi = 0; bi < _ckptBatches.Length; bi++)
        {
            var batch = _ckptBatches[bi];
            _ckptHostBuf.AsSpan(batch.HostOffsetBytes, batch.TotalBytes).CopyTo(new Span<byte>(_pCkptUpload, batch.TotalBytes));

            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;
            foreach (var res in batch.UniqueResources)
                cmd.ResourceBarrierTransition(res, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
            foreach (var item in batch.Items)
                cmd.CopyBufferRegion(item.Resource, item.ResourceOffsetBytes, _ckptUploadBuf, item.BatchOffsetBytes, (ulong)item.ByteCount);
            foreach (var res in batch.UniqueResources)
                cmd.ResourceBarrierTransition(res, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
        }
    }

    private void EnsureCheckpointSetup()
    {
        if (_ckptBatches != null) return;

        var rawSlices = new List<(ID3D12Resource Res, ulong Off, ulong Len)>(64);
        AddSliceChunked(rawSlices, _kConvHistBuf, 0UL, _kConvHistBuf.Description.Width);
        AddSliceChunked(rawSlices, _vConvHistBuf, 0UL, _vConvHistBuf.Description.Width);
        AddSliceChunked(rawSlices, _attnPostHistBuf, 0UL, _attnPostHistBuf.Description.Width);
        AddSliceChunked(rawSlices, _ffnPostHistBuf, 0UL, _ffnPostHistBuf.Description.Width);

        for (int il = 0; il < _nLogical; il++)
        {
            var phys = _w.PhysicalLayers[PhysicalLayerOf(il)];
            if (!phys.IsSwa) continue;
            ulong sliceBytes = (ulong)_kvCap[il] * (ulong)phys.HeadCountKv * (ulong)phys.HeadDim * 2UL;
            if (sliceBytes == 0UL) continue;
            ulong offsetBytes = (ulong)_kvLayerBaseU32[il] * 4UL;
            AddSliceChunked(rawSlices, _kCacheFp16, offsetBytes, sliceBytes);
            AddSliceChunked(rawSlices, _vCacheFp16, offsetBytes, sliceBytes);
        }

        var batches = new List<CkptBatch>();
        var items = new List<CkptBatchItem>();
        var unique = new List<ID3D12Resource>(6);
        ulong batchBytes = 0UL;
        int hostTotal = 0;
        int batchHostStart = 0;
        int maxBatch = 0;

        foreach (var s in rawSlices)
        {
            if (batchBytes + s.Len > CkptBatchCapBytes && items.Count > 0)
            {
                int bLen = checked((int)batchBytes);
                batches.Add(new CkptBatch(items.ToArray(), unique.ToArray(), batchHostStart, bLen));
                if (bLen > maxBatch) maxBatch = bLen;
                items.Clear();
                unique.Clear();
                batchHostStart = hostTotal;
                batchBytes = 0UL;
            }

            if (!unique.Contains(s.Res)) unique.Add(s.Res);
            items.Add(new CkptBatchItem(s.Res, s.Off, batchBytes, checked((int)s.Len)));
            batchBytes += s.Len;
            hostTotal = checked(hostTotal + (int)s.Len);
        }

        if (items.Count > 0)
        {
            int bLen = checked((int)batchBytes);
            batches.Add(new CkptBatch(items.ToArray(), unique.ToArray(), batchHostStart, bLen));
            if (bLen > maxBatch) maxBatch = bLen;
        }

        _ckptBatches = batches.ToArray();
        _ckptMaxBatchBytes = maxBatch;
        _ckptHostBuf = new byte[hostTotal];
    }

    private static void AddSliceChunked(List<(ID3D12Resource Res, ulong Off, ulong Len)> dst, ID3D12Resource res, ulong off, ulong len)
    {
        ulong done = 0UL;
        while (done < len)
        {
            ulong chunk = Math.Min(len - done, CkptBatchCapBytes);
            dst.Add((res, off + done, chunk));
            done += chunk;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_pUpload != null) { _uploadBuf.Unmap(0); _pUpload = null; }
        if (_pReadback != null) { _readbackBuf.Unmap(0); _pReadback = null; }
        if (_pCpuMoeReadback != null) { _cpuMoeReadbackBuf.Unmap(0); _pCpuMoeReadback = null; }
        if (_pCkptReadback != null) { _ckptReadbackBuf!.Unmap(0); _pCkptReadback = null; }
        if (_pCkptUpload != null) { _ckptUploadBuf!.Unmap(0); _pCkptUpload = null; }

        foreach (var r in _allResources) r.Dispose();
        _allResources.Clear();

        _psoQ8_0.Dispose(); _psoQ6K.Dispose(); _psoQ5K.Dispose(); _psoF32.Dispose(); _sigGemv.Dispose();
        _psoNvfp4Ind.Dispose(); _sigNvfp4Ind.Dispose();
        _psoRmsNorm.Dispose(); _sigRmsNorm.Dispose();
        _psoMix.Dispose(); _sigMix.Dispose();
        _psoLoopBlend.Dispose(); _sigLoopBlend.Dispose();
        _psoShortConv.Dispose(); _sigShortConv.Dispose();
        _psoHeadNorm.Dispose(); _sigHeadNorm.Dispose();
        _psoAttn.Dispose(); _sigAttn.Dispose();
        _psoRouter.Dispose(); _sigRouter.Dispose();
        _psoSituBatch.Dispose(); _psoWeightedSum.Dispose(); _psoVecAdd.Dispose(); _psoUavCopy.Dispose(); _sigReduce.Dispose();
        _psoSoftcap.Dispose(); _sigSoftcap.Dispose();
        DisposeBatchPipelines();

        if (_ownsContext) _ctx.Dispose();
    }
}
#endif
