namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;

/// <summary>
/// Direct3D 12 GPU inference runtime for hybrid SSM / attention models.
/// Accelerates Gated DeltaNet (linear attention SSM with recurrent associative memory)
/// interleaved with full quadratic multi-head attention on Direct3D 12 hardware.
/// </summary>
public sealed unsafe class Qwen3HybridD3D12Model : ID3D12Model
{
    private readonly D3D12Context _ctx;
    private readonly ModelWeights _weights;
    private readonly int _maxSeqLen;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _vocabSize;
    private readonly int _nHeads;
    private readonly int _headsKv;
    private readonly int _headDim;
    private readonly int _ropeDim;
    private readonly int _ssmConvKernel;
    private readonly int _ssmStateDim;
    private readonly int _ssmGroupCount;
    private readonly int _ssmHeads;
    private readonly int _ssmInnerSize;
    private readonly int _gdnConvChannels;

    private readonly List<D3D12LayerWeights> _layerWeights = new();

    // Direct3D 12 Root Signatures & Pipeline State Objects
    private ID3D12RootSignature _sigRmsNorm = null!;
    private ID3D12PipelineState _psoRmsNorm = null!;

    private ID3D12RootSignature _sigRmsNormHeads = null!;
    private ID3D12PipelineState _psoRmsNormHeads = null!;

    private ID3D12RootSignature _sigGemv = null!;
    private ID3D12PipelineState _psoGemvQ4K = null!;
    private ID3D12PipelineState _psoGemvQ5K = null!;
    private ID3D12PipelineState _psoGemvQ6K = null!;
    private ID3D12PipelineState _psoGemvQ3K = null!;
    private ID3D12PipelineState _psoGemvQ8_0 = null!;
    private ID3D12PipelineState _psoGemvQ2_K = null!;
    private ID3D12PipelineState _psoGemvQ2_0 = null!;
    private ID3D12PipelineState _psoGemvIQ2_XXS = null!;
    private ID3D12PipelineState _psoGemvIQ4_NL = null!;
    private ID3D12PipelineState _psoGemvIQ3_S = null!;
    private ID3D12PipelineState _psoGemvTernary = null!;
    private ID3D12PipelineState _psoGemvFp32 = null!;

    private ID3D12RootSignature _sigSwiglu = null!;
    private ID3D12PipelineState _psoSwiglu = null!;

    private ID3D12RootSignature _sigVecAdd = null!;
    private ID3D12PipelineState _psoVecAdd = null!;

    private ID3D12RootSignature _sigVecAddWeighted = null!;
    private ID3D12PipelineState _psoVecAddWeighted = null!;

    private ID3D12RootSignature _sigRope = null!;
    private ID3D12PipelineState _psoRope = null!;

    private ID3D12RootSignature _sigKvStore = null!;
    private ID3D12PipelineState _psoKvStore = null!;

    private ID3D12RootSignature _sigFlashAttn = null!;
    private ID3D12PipelineState _psoFlashAttn = null!;

    private ID3D12RootSignature _sigSsmConv1d = null!;
    private ID3D12PipelineState _psoSsmConv1d = null!;

    private ID3D12RootSignature _sigL2NormQK = null!;
    private ID3D12PipelineState _psoL2NormQK = null!;

    private ID3D12RootSignature _sigDeltaNet = null!;
    private ID3D12PipelineState _psoDeltaNet = null!;

    private ID3D12RootSignature _sigSsmGateSilu = null!;
    private ID3D12PipelineState _psoSsmGateSilu = null!;

    private ID3D12RootSignature _sigQGateSplit = null!;
    private ID3D12PipelineState _psoQGateSplit = null!;

    private ID3D12RootSignature _sigAttnOutGate = null!;
    private ID3D12PipelineState _psoAttnOutGate = null!;

    // Batched Root Signatures & Pipeline State Objects
    private ID3D12RootSignature _sigGemmBatch = null!;
    private ID3D12PipelineState _psoGemmQ4KBatch = null!;
    private ID3D12PipelineState _psoGemmQ6KBatch = null!;
    private ID3D12PipelineState _psoGemmQ8_0Batch = null!;
    private ID3D12PipelineState _psoGemmFp32Batch = null!;
    private ID3D12PipelineState _psoGemmQ3KBatch = null!;
    private ID3D12PipelineState _psoGemmQ5KBatch = null!;
    private ID3D12PipelineState _psoGemmQ2_KBatch = null!;
    private ID3D12PipelineState _psoGemmQ2_0Batch = null!;
    private ID3D12PipelineState _psoGemmIQ2_XXSBatch = null!;
    private ID3D12PipelineState _psoGemmIQ4_NLBatch = null!;
    private ID3D12PipelineState _psoGemmIQ3_SBatch = null!;
    private ID3D12PipelineState _psoGemmTernaryBatch = null!;

    private ID3D12RootSignature _sigRmsNormBatch = null!;
    private ID3D12PipelineState _psoRmsNormBatch = null!;

    // GPU Scratch Buffers
    private ID3D12Resource _dX = null!;
    private ID3D12Resource _dNormX = null!;
    private ID3D12Resource _dQ = null!;
    private ID3D12Resource _dQGate = null!;
    private ID3D12Resource _dK = null!;
    private ID3D12Resource _dV = null!;
    private ID3D12Resource _dQFull = null!;
    private ID3D12Resource _dAttnOut = null!;
    private ID3D12Resource _dAttnProjOut = null!;

    // SSM Gated DeltaNet GPU Buffers
    private ID3D12Resource _dGdnQkv = null!;
    private ID3D12Resource _dGdnConvOut = null!;
    private ID3D12Resource _dGdnZ = null!;
    private ID3D12Resource _dGdnA = null!;
    private ID3D12Resource _dGdnB = null!;
    private ID3D12Resource _dGdnY = null!;

    // Per-layer SSM State Buffers
    private readonly ID3D12Resource[] _dConvState;
    private readonly ID3D12Resource[] _dSsmState;

    // Interleaved Full Attention per-layer KV cache
    private readonly ID3D12Resource[] _dKeyCache;
    private readonly ID3D12Resource[] _dValCache;

    // MLP Buffers
    private ID3D12Resource _dMlpNorm = null!;
    private ID3D12Resource _dMlpGate = null!;
    private ID3D12Resource _dMlpUp = null!;
    private ID3D12Resource _dMlpAct = null!;
    private ID3D12Resource _dMlpOut = null!;

    // MoE Scratch Buffers (GPU-First + Tiered Overflow)
    private readonly bool _tieredMoe;
    private ID3D12Resource? _dRouterLogits;
    private ID3D12Resource? _readbackRouterLogits;
    private float* _pReadbackRouterLogits;
    private ID3D12Resource? _dExpertGate;
    private ID3D12Resource? _dExpertUp;
    private ID3D12Resource? _dExpertAct;
    private ID3D12Resource? _dExpertDownOut;
    private ID3D12Resource? _dFfnOut;
    private ID3D12Resource? _dShexpGate;
    private ID3D12Resource? _dShexpUp;
    private ID3D12Resource? _dShexpAct;
    private ID3D12Resource? _readbackNormX;
    private float* _pReadbackNormX;
    private ID3D12Resource? _uploadExpertOut;
    private float* _pUploadExpertOut;
    private float[]? _normXSums;
    private Qwen2D3D12Model.ExpertCpuWorkspace[]? _expertWorkspaces;

    // In-VRAM MoE Router and Shared Expert Buffers
    private ID3D12Resource? _dTopKIndices;
    private ID3D12Resource? _dTopKWeights;
    private ID3D12Resource? _dShexpGateScalar;
    private readonly Dictionary<int, ID3D12Resource> _dShexpGateWeights = new();

    // Chunked Batched Prefill Buffers
    private int _allocatedBatchCapacity;
    private ID3D12Resource? _dXBatch;
    private ID3D12Resource? _dNormXBatch;
    private ID3D12Resource? _uploadEmbeddingBatch;
    private float* _pUploadEmbeddingBatch;

    private ID3D12Resource? _dQBatch;
    private ID3D12Resource? _dQGateBatch;
    private ID3D12Resource? _dQFullBatch;
    private ID3D12Resource? _dKBatch;
    private ID3D12Resource? _dVBatch;
    private ID3D12Resource? _dAttnOutBatch;

    private ID3D12Resource? _dGdnQkvBatch;
    private ID3D12Resource? _dGdnConvOutBatch;
    private ID3D12Resource? _dGdnZBatch;
    private ID3D12Resource? _dGdnABatch;
    private ID3D12Resource? _dGdnBBatch;
    private ID3D12Resource? _dGdnYBatch;
    private ID3D12Resource? _dAttnProjOutBatch;

    private ID3D12Resource? _dMlpNormBatch;
    private ID3D12Resource? _dMlpGateBatch;
    private ID3D12Resource? _dMlpUpBatch;
    private ID3D12Resource? _dMlpActBatch;
    private ID3D12Resource? _dMlpOutBatch;

    private ID3D12Resource? _dRouterLogitsBatch;
    private ID3D12Resource? _dTopKIndicesBatch;
    private ID3D12Resource? _dTopKWeightsBatch;

    // Output Head Buffers
    private ID3D12Resource? _dOutNormWeight;
    private ID3D12Resource _dOutNorm = null!;
    private ID3D12Resource? _dOutWeight;
    private ID3D12Resource _dLogits = null!;

    private ID3D12Resource _uploadEmbedding = null!;
    private float* _pUploadEmbedding;
    private ID3D12Resource? _readbackActivation;
    private float* _pReadbackActivation;
    private ID3D12Resource _readbackLogits = null!;
    private float* _pReadbackLogits;

    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int VocabSize => _vocabSize;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }
    public bool TieredMoe => _tieredMoe;

    public Qwen3HybridD3D12Model(
        D3D12Context ctx,
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true,
        bool? tieredMoe = null)
    {
        _ctx = ctx;
        _weights = weights;
        _maxSeqLen = maxSeqLen;
        StartLayer = startLayer;
        LayerCount = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        IsLastStage = isLastStage;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _vocabSize = weights.VocabSize;
        _nHeads = weights.HeadCount;
        _headsKv = weights.HeadCountKv;
        _headDim = weights.HeadDim;
        _ropeDim = weights.RopeDimensionCount > 0 ? weights.RopeDimensionCount : _headDim;

        _ssmConvKernel = weights.SsmConvKernel > 0 ? weights.SsmConvKernel : 4;
        _ssmStateDim = weights.SsmStateSize > 0 ? weights.SsmStateSize : 128;
        _ssmGroupCount = weights.SsmGroupCount > 0 ? weights.SsmGroupCount : 16;
        _ssmHeads = weights.SsmTimeStepRank > 0 ? weights.SsmTimeStepRank : 32;
        _ssmInnerSize = weights.SsmInnerSize > 0 ? weights.SsmInnerSize : 4096;
        _gdnConvChannels = (_ssmGroupCount * 2 + _ssmHeads) * _ssmStateDim;

        if (tieredMoe.HasValue)
        {
            _tieredMoe = tieredMoe.Value && weights.IsMoe;
        }
        else if (weights.IsMoe && LayerCount > 0)
        {
            int expertFfnDim = weights.ExpertFeedForwardLength;
            int numExperts = weights.ExpertCount;
            long gateSlice = (long)expertFfnDim * GgufTypes.GetRowBytes(weights.Layers[startLayer].FfnGateExpsType, _dim);
            long upSlice = (long)expertFfnDim * GgufTypes.GetRowBytes(weights.Layers[startLayer].FfnUpExpsType, _dim);
            long downSlice = (long)_dim * GgufTypes.GetRowBytes(weights.Layers[startLayer].FfnDownExpsType, expertFfnDim);
            ulong totalExpertBytes = (ulong)(LayerCount * (long)numExperts * (gateSlice + upSlice + downSlice));

            ulong vram = ctx.TotalAccessibleMemory;
            _tieredMoe = vram > 0 && totalExpertBytes + 350_000_000UL > vram;
        }
        else
        {
            _tieredMoe = false;
        }

        string archLabel = !string.IsNullOrWhiteSpace(weights.Gguf.Architecture) ? weights.Gguf.Architecture : weights.ArchitectureFamily.ToString();
        Glacier.Inference.Diagnostics.GlacierDiagnostics.LogInformation($"[Direct3D 12 {archLabel}] Arch: '{weights.Gguf.Architecture}', Layers: {StartLayer}..{StartLayer + LayerCount - 1}, Heads: {_nHeads}/{_headsKv}, HeadDim: {_headDim}, RopeDim: {_ropeDim}, SsmHeads: {_ssmHeads}, ConvChannels: {_gdnConvChannels}, IsMoe: {weights.IsMoe}, TieredMoe: {_tieredMoe}");

        _dConvState = new ID3D12Resource[LayerCount];
        _dSsmState = new ID3D12Resource[LayerCount];
        _dKeyCache = new ID3D12Resource[LayerCount];
        _dValCache = new ID3D12Resource[LayerCount];

        InitPipelines();
        AllocateBuffers();
        UploadWeights();
    }

    private void InitPipelines()
    {
        // 1. GEMV Root Signature & PSOs
        var gemvParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
        _psoGemvQ4K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ4K));
        _psoGemvQ5K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ5K));
        _psoGemvQ6K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ6K));
        _psoGemvQ3K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ3K));
        _psoGemvQ8_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ8_0));
        _psoGemvQ2_K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ2_K));
        _psoGemvQ2_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ2_0));
        _psoGemvIQ2_XXS = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ2_XXS));
        _psoGemvIQ4_NL = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ4_NL));
        _psoGemvIQ3_S = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ3_S));
        _psoGemvTernary = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvTernary));
        _psoGemvFp32 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvFp32));

        // 2. RMSNorm
        var rmsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsParams));
        _psoRmsNorm = _ctx.CreatePipelineState(_sigRmsNorm, _ctx.CompileShader(D3D12Shaders.RmsNorm));

        // 3. RMSNorm per Head (QK-Norm)
        var rmsHeadsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigRmsNormHeads = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsHeadsParams));
        _psoRmsNormHeads = _ctx.CreatePipelineState(_sigRmsNormHeads, _ctx.CompileShader(D3D12Shaders.Qwen3RmsNormHeads));

        // 4. SwiGLU
        var swigluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigSwiglu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, swigluParams));
        _psoSwiglu = _ctx.CreatePipelineState(_sigSwiglu, _ctx.CompileShader(D3D12Shaders.SwiGLU));

        // 5. VecAdd & VecAddWeighted
        var vecAddParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAdd = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddParams));
        _psoVecAdd = _ctx.CreatePipelineState(_sigVecAdd, _ctx.CompileShader(D3D12Shaders.VecAdd));

        var vecAddWeightedParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAddWeighted = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddWeightedParams));
        _psoVecAddWeighted = _ctx.CreatePipelineState(_sigVecAddWeighted, _ctx.CompileShader(D3D12Shaders.VecAddWeighted));

        // 6. RoPE (Partial Rotary Position Embedding)
        var ropeParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRope = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeParams));
        _psoRope = _ctx.CreatePipelineState(_sigRope, _ctx.CompileShader(D3D12Shaders.Qwen3RoPE));

        // 7. KV Store
        var kvStoreParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigKvStore = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, kvStoreParams));
        _psoKvStore = _ctx.CreatePipelineState(_sigKvStore, _ctx.CompileShader(D3D12Shaders.KvCacheStore));

        // 8. FlashAttention / AttentionGqa (256-wide)
        var attnParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigFlashAttn = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        _psoFlashAttn = _ctx.CreatePipelineState(_sigFlashAttn, _ctx.CompileShader(D3D12Shaders.Qwen3Attention));

        // 9. SSM Conv1d Pipeline
        var sigConv1dParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigSsmConv1d = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigConv1dParams));
        _psoSsmConv1d = _ctx.CreatePipelineState(_sigSsmConv1d, _ctx.CompileShader(D3D12Shaders.Qwen3SsmConv1d));

        // 10. SSM L2 Norm QK Pipeline
        var sigL2NormParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigL2NormQK = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigL2NormParams));
        _psoL2NormQK = _ctx.CreatePipelineState(_sigL2NormQK, _ctx.CompileShader(D3D12Shaders.Qwen3L2NormQK));

        // 11. SSM DeltaNet Update Pipeline
        var sigDeltaNetParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(4, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(5, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(6, 0), ShaderVisibility.All)
        };
        _sigDeltaNet = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigDeltaNetParams));
        _psoDeltaNet = _ctx.CreatePipelineState(_sigDeltaNet, _ctx.CompileShader(D3D12Shaders.Qwen3DeltaNetUpdate));

        // 12. SSM Gate SiLU Pipeline
        var sigGateSiluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigSsmGateSilu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigGateSiluParams));
        _psoSsmGateSilu = _ctx.CreatePipelineState(_sigSsmGateSilu, _ctx.CompileShader(D3D12Shaders.Qwen3SsmGateSilu));

        // 13. Q-Gate Split Pipeline (u0: q_full, u1: q_out, u2: q_gate_out)
        var sigQGateParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigQGateSplit = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigQGateParams));
        _psoQGateSplit = _ctx.CreatePipelineState(_sigQGateSplit, _ctx.CompileShader(D3D12Shaders.Qwen3QGateSplit));

        // 14. Post-Attention Q-Gate Pipeline (u0: attn_out, u1: q_gate)
        var sigAttnOutGateParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigAttnOutGate = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, sigAttnOutGateParams));
        _psoAttnOutGate = _ctx.CreatePipelineState(_sigAttnOutGate, _ctx.CompileShader(D3D12Shaders.Qwen3AttnOutGate));

        // 15. Batched GEMM Root Signature: (Params b0, W t0, bias t1, x u2, residual u0, y u1)
        var gemmBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: x
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: residual
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)  // u1: y
        };

        _sigGemmBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemmBatchParams));
        _psoGemmQ4KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
        _psoGemmQ6KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));
        _psoGemmQ8_0Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ8_0Batch));
        _psoGemmFp32Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmFp32Batch));
        _psoGemmQ3KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ3KBatch));
        _psoGemmQ5KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ5KBatch));
        _psoGemmQ2_KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ2_KBatch));
        _psoGemmQ2_0Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ2_0Batch));
        _psoGemmIQ2_XXSBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmIQ2_XXSBatch));
        _psoGemmIQ4_NLBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmIQ4_NLBatch));
        _psoGemmIQ3_SBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmIQ3_SBatch));
        _psoGemmTernaryBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmTernaryBatch));

        // 16. Batched RMSNorm Root Signature: (Params b0, weight t0, x u0, dst u1)
        var rmsNormBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNormBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsNormBatchParams));
        _psoRmsNormBatch = _ctx.CreatePipelineState(_sigRmsNormBatch, _ctx.CompileShader(D3D12Shaders.RmsNormBatch));
    }

    private void AllocateBuffers()
    {
        int qDim = _nHeads * _headDim;
        int kvDim = _headsKv * _headDim;

        _dX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dNormX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dQ = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
        _dQGate = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
        _dK = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dV = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dQFull = _ctx.CreateDeviceBuffer((ulong)(2 * qDim * sizeof(float)));
        _dAttnOut = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
        _dAttnProjOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // SSM Gated DeltaNet Buffers
        _dGdnQkv = _ctx.CreateDeviceBuffer((ulong)(_gdnConvChannels * sizeof(float)));
        _dGdnConvOut = _ctx.CreateDeviceBuffer((ulong)(_gdnConvChannels * sizeof(float)));
        _dGdnZ = _ctx.CreateDeviceBuffer((ulong)(_ssmInnerSize * sizeof(float)));
        _dGdnA = _ctx.CreateDeviceBuffer((ulong)(_ssmHeads * sizeof(float)));
        _dGdnB = _ctx.CreateDeviceBuffer((ulong)(_ssmHeads * sizeof(float)));
        _dGdnY = _ctx.CreateDeviceBuffer((ulong)(_ssmInnerSize * sizeof(float)));

        // Per-layer states
        for (int l = 0; l < LayerCount; l++)
        {
            var lw = _weights.Layers[StartLayer + l];
            if (lw.IsGdn)
            {
                _dConvState[l] = _ctx.CreateDeviceBuffer((ulong)(_gdnConvChannels * 3 * sizeof(float)));
                _dSsmState[l] = _ctx.CreateDeviceBuffer((ulong)(_ssmHeads * _ssmStateDim * _ssmStateDim * sizeof(float)));
            }
            else
            {
                ulong kvLayerBytes = (ulong)(_headsKv * _headDim * _maxSeqLen * sizeof(float));
                _dKeyCache[l] = _ctx.CreateDeviceBuffer(kvLayerBytes);
                _dValCache[l] = _ctx.CreateDeviceBuffer(kvLayerBytes);
            }
        }

        // MLP Buffers
        int mlpDim = Math.Max(1, _ffnDim);
        _dMlpNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dMlpGate = _ctx.CreateDeviceBuffer((ulong)(mlpDim * sizeof(float)));
        _dMlpUp = _ctx.CreateDeviceBuffer((ulong)(mlpDim * sizeof(float)));
        _dMlpAct = _ctx.CreateDeviceBuffer((ulong)(mlpDim * sizeof(float)));
        _dMlpOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // MoE Buffers
        if (_weights.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFeedForwardLength;
            int numExperts = _weights.ExpertCount;

            _dRouterLogits = _ctx.CreateDeviceBuffer((ulong)(numExperts * sizeof(float)));
            _readbackRouterLogits = _ctx.CreateReadbackBuffer((ulong)(numExperts * sizeof(float)));
            void* pRbRouter = null;
            _readbackRouterLogits.Map(0, null, &pRbRouter);
            _pReadbackRouterLogits = (float*)pRbRouter;

            _dExpertGate = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertUp = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertAct = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertDownOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _dFfnOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

            int shexpFfnDim = expertFfnDim * 2;
            if (_weights.Gguf.TryGetTensor($"blk.{StartLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                shexpFfnDim = (int)tShexp.Dimensions[1];

            _dShexpGate = _ctx.CreateDeviceBuffer((ulong)(shexpFfnDim * sizeof(float)));
            _dShexpUp = _ctx.CreateDeviceBuffer((ulong)(shexpFfnDim * sizeof(float)));
            _dShexpAct = _ctx.CreateDeviceBuffer((ulong)(shexpFfnDim * sizeof(float)));

            if (_tieredMoe)
            {
                _readbackNormX = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
                void* pRbNormX = null;
                _readbackNormX.Map(0, null, &pRbNormX);
                _pReadbackNormX = (float*)pRbNormX;

                _uploadExpertOut = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
                void* pUpExpOut = null;
                _uploadExpertOut.Map(0, null, &pUpExpOut);
                _pUploadExpertOut = (float*)pUpExpOut;

                _normXSums = new float[(_dim + 31) / 32];
                _expertWorkspaces = [new Qwen2D3D12Model.ExpertCpuWorkspace(expertFfnDim, _dim)];
            }
        }

        // Head and Output
        _dOutNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dLogits = _ctx.CreateDeviceBuffer((ulong)(_vocabSize * sizeof(float)));

        _uploadEmbedding = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbedding.Map(0, null, &pUpload);
        _pUploadEmbedding = (float*)pUpload;

        if (!IsLastStage)
        {
            _readbackActivation = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
            void* pRbAct = null;
            _readbackActivation.Map(0, null, &pRbAct);
            _pReadbackActivation = (float*)pRbAct;
        }

        _readbackLogits = _ctx.CreateReadbackBuffer((ulong)(_vocabSize * sizeof(float)));
        void* pRbLogits = null;
        _readbackLogits.Map(0, null, &pRbLogits);
        _pReadbackLogits = (float*)pRbLogits;
    }

    private void UploadWeights()
    {
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            if (l % 8 == 0 || l == LayerCount - 1)
            {
                string archLabel = !string.IsNullOrWhiteSpace(_weights.Gguf.Architecture) ? _weights.Gguf.Architecture : _weights.ArchitectureFamily.ToString();
                Glacier.Inference.Diagnostics.GlacierDiagnostics.LogInformation($"[D3D12 GPU] Uploading {archLabel} weights: layer {modelLayer + 1}/{_weights.BlockCount}...");
            }

            var lw = _weights.Layers[modelLayer];
            bool isGdn = lw.IsGdn;

            var dAttnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (ulong)(_dim * sizeof(float)));

            ID3D12Resource? dQ = null;
            ID3D12Resource? dK = null;
            ID3D12Resource? dV = null;
            ID3D12Resource? dAttnOut = null;
            ID3D12Resource? dAttnQNorm = null;
            ID3D12Resource? dAttnKNorm = null;
            ID3D12Resource? dQkv = null;
            ID3D12Resource? dAttnGate = null;
            ID3D12Resource? dSsmConv1d = null;
            ID3D12Resource? dSsmAlpha = null;
            ID3D12Resource? dSsmBeta = null;
            ID3D12Resource? dSsmDtBias = null;
            ID3D12Resource? dSsmA = null;
            ID3D12Resource? dSsmNorm = null;
            ID3D12Resource? dSsmOut = null;

            if (isGdn)
            {
                // Gated DeltaNet layer
                dQkv = UploadTensor(lw.QkvType, (IntPtr)lw.QkvWeight, _gdnConvChannels, _dim);
                dAttnGate = UploadTensor(lw.AttnGateType, (IntPtr)lw.AttnGateWeight, _ssmInnerSize, _dim);

                dSsmConv1d = _ctx.CreateDeviceBuffer((ulong)(_gdnConvChannels * _ssmConvKernel * sizeof(float)));
                _ctx.CopyToDevice(dSsmConv1d, (IntPtr)lw.SsmConv1dWeight, (ulong)(_gdnConvChannels * _ssmConvKernel * sizeof(float)));

                dSsmAlpha = UploadTensor(lw.SsmAlphaType, (IntPtr)lw.SsmAlphaWeight, _ssmHeads, _dim);
                dSsmBeta = UploadTensor(lw.SsmBetaType, (IntPtr)lw.SsmBetaWeight, _ssmHeads, _dim);

                if (lw.SsmDtBias != null)
                {
                    dSsmDtBias = _ctx.CreateDeviceBuffer((ulong)(_ssmHeads * sizeof(float)));
                    _ctx.CopyToDevice(dSsmDtBias, (IntPtr)lw.SsmDtBias, (ulong)(_ssmHeads * sizeof(float)));
                }

                if (lw.SsmAWeight != null)
                {
                    dSsmA = _ctx.CreateDeviceBuffer((ulong)(_ssmHeads * sizeof(float)));
                    _ctx.CopyToDevice(dSsmA, (IntPtr)lw.SsmAWeight, (ulong)(_ssmHeads * sizeof(float)));
                }

                if (lw.SsmNormWeight != null)
                {
                    dSsmNorm = _ctx.CreateDeviceBuffer((ulong)(_ssmStateDim * sizeof(float)));
                    _ctx.CopyToDevice(dSsmNorm, (IntPtr)lw.SsmNormWeight, (ulong)(_ssmStateDim * sizeof(float)));
                }

                dSsmOut = UploadTensor(lw.SsmOutType, (IntPtr)lw.SsmOutWeight, _dim, _ssmInnerSize);
            }
            else
            {
                // Standard full attention layer
                int qDim = _nHeads * _headDim;
                int kvDim = _headsKv * _headDim;
                int qTargetRows = lw.HasQGate ? 2 * qDim : qDim;

                dQ = UploadTensor(lw.QType, (IntPtr)lw.QWeight, qTargetRows, _dim);
                dK = UploadTensor(lw.KType, (IntPtr)lw.KWeight, kvDim, _dim);
                dV = UploadTensor(lw.VType, (IntPtr)lw.VWeight, kvDim, _dim);
                dAttnOut = UploadTensor(lw.AttnOutType, (IntPtr)lw.AttnOutWeight, _dim, qDim);

                if (lw.AttnQNormWeight != null)
                {
                    dAttnQNorm = _ctx.CreateDeviceBuffer((ulong)(_headDim * sizeof(float)));
                    _ctx.CopyToDevice(dAttnQNorm, (IntPtr)lw.AttnQNormWeight, (ulong)(_headDim * sizeof(float)));
                }

                if (lw.AttnKNormWeight != null)
                {
                    dAttnKNorm = _ctx.CreateDeviceBuffer((ulong)(_headDim * sizeof(float)));
                    _ctx.CopyToDevice(dAttnKNorm, (IntPtr)lw.AttnKNormWeight, (ulong)(_headDim * sizeof(float)));
                }
            }

            // FFN
            var dFfnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (ulong)(_dim * sizeof(float)));

            ID3D12Resource? dFfnGate = null;
            ID3D12Resource? dFfnUp = null;
            ID3D12Resource? dFfnDown = null;
            ID3D12Resource? dFfnGateInp = null;
            ID3D12Resource? dFfnGateInpBias = null;
            ID3D12Resource? dFfnGateExps = null;
            ID3D12Resource? dFfnUpExps = null;
            ID3D12Resource? dFfnDownExps = null;
            ID3D12Resource? dFfnGateShexp = null;
            ID3D12Resource? dFfnUpShexp = null;
            ID3D12Resource? dFfnDownShexp = null;

            if (lw.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;

                dFfnGateInp = UploadTensor(GgufType.F32, (IntPtr)lw.FfnGateInpWeight, numExperts, _dim);
                if (lw.FfnGateInpBias != null)
                {
                    dFfnGateInpBias = _ctx.CreateDeviceBuffer((ulong)(numExperts * sizeof(float)));
                    _ctx.CopyToDevice(dFfnGateInpBias, (IntPtr)lw.FfnGateInpBias, (ulong)(numExperts * sizeof(float)));
                }

                if (!_tieredMoe)
                {
                    dFfnGateExps = UploadTensor(lw.FfnGateExpsType, (IntPtr)lw.FfnGateExpsWeight, numExperts * expertFfnDim, _dim);
                    dFfnUpExps = UploadTensor(lw.FfnUpExpsType, (IntPtr)lw.FfnUpExpsWeight, numExperts * expertFfnDim, _dim);
                    dFfnDownExps = UploadTensor(lw.FfnDownExpsType, (IntPtr)lw.FfnDownExpsWeight, numExperts * _dim, expertFfnDim);
                }

                if (lw.FfnGateShexpWeight != null)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        shexpFfnDim = (int)tShexp.Dimensions[1];

                    dFfnGateShexp = UploadTensor(lw.FfnGateShexpType, (IntPtr)lw.FfnGateShexpWeight, shexpFfnDim, _dim);
                    dFfnUpShexp = UploadTensor(lw.FfnUpShexpType, (IntPtr)lw.FfnUpShexpWeight, shexpFfnDim, _dim);
                    dFfnDownShexp = UploadTensor(lw.FfnDownShexpType, (IntPtr)lw.FfnDownShexpWeight, _dim, shexpFfnDim);
                }
            }
            else
            {
                dFfnGate = UploadTensor(lw.FfnGateType, (IntPtr)lw.FfnGateWeight, _ffnDim, _dim);
                dFfnUp = UploadTensor(lw.FfnUpType, (IntPtr)lw.FfnUpWeight, _ffnDim, _dim);
                dFfnDown = UploadTensor(lw.FfnDownType, (IntPtr)lw.FfnDownWeight, _dim, _ffnDim);
            }

            _layerWeights.Add(new D3D12LayerWeights
            {
                IsGdn = isGdn,
                HasQGate = lw.HasQGate,
                AttnNormWeight = dAttnNorm,
                QWeight = dQ ?? _ctx.DummyBuffer,
                QType = NormalizeGpuType(lw.QType),
                KWeight = dK ?? _ctx.DummyBuffer,
                KType = NormalizeGpuType(lw.KType),
                VWeight = dV ?? _ctx.DummyBuffer,
                VType = NormalizeGpuType(lw.VType),
                AttnOutWeight = dAttnOut ?? _ctx.DummyBuffer,
                AttnOutType = NormalizeGpuType(lw.AttnOutType),
                AttnQNormWeight = dAttnQNorm,
                AttnKNormWeight = dAttnKNorm,
                QkvWeight = dQkv,
                QkvType = NormalizeGpuType(lw.QkvType),
                AttnGateWeight = dAttnGate,
                AttnGateType = NormalizeGpuType(lw.AttnGateType),
                SsmConv1dWeight = dSsmConv1d,
                SsmAlphaWeight = dSsmAlpha,
                SsmAlphaType = NormalizeGpuType(lw.SsmAlphaType),
                SsmBetaWeight = dSsmBeta,
                SsmBetaType = NormalizeGpuType(lw.SsmBetaType),
                SsmDtBias = dSsmDtBias,
                SsmAWeight = dSsmA,
                SsmNormWeight = dSsmNorm,
                SsmOutWeight = dSsmOut,
                SsmOutType = NormalizeGpuType(lw.SsmOutType),
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnGateType = NormalizeGpuType(lw.FfnGateType),
                FfnUpWeight = dFfnUp,
                FfnUpType = NormalizeGpuType(lw.FfnUpType),
                FfnDownWeight = dFfnDown,
                FfnDownType = NormalizeGpuType(lw.FfnDownType),
                IsMoe = lw.IsMoe,
                FfnGateInpWeight = dFfnGateInp,
                FfnGateInpBias = dFfnGateInpBias,
                FfnGateExpsWeight = dFfnGateExps,
                FfnUpExpsWeight = dFfnUpExps,
                FfnDownExpsWeight = dFfnDownExps,
                FfnGateExpsType = NormalizeGpuType(lw.FfnGateExpsType),
                FfnUpExpsType = NormalizeGpuType(lw.FfnUpExpsType),
                FfnDownExpsType = NormalizeGpuType(lw.FfnDownExpsType),
                FfnGateShexpWeight = dFfnGateShexp,
                FfnUpShexpWeight = dFfnUpShexp,
                FfnDownShexpWeight = dFfnDownShexp,
                FfnGateShexpType = NormalizeGpuType(lw.FfnGateShexpType),
                FfnUpShexpType = NormalizeGpuType(lw.FfnUpShexpType),
                FfnDownShexpType = NormalizeGpuType(lw.FfnDownShexpType)
            });
        }

        // LM Head (only needed on last stage)
        if (IsLastStage)
        {
            _dOutNormWeight = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (ulong)(_dim * sizeof(float)));
            _dOutWeight = UploadTensor(_weights.OutType, (IntPtr)_weights.OutWeight, _vocabSize, _dim);
        }
    }

    private static GgufType NormalizeGpuType(GgufType t) => t == GgufType.BF16 ? GgufType.F32 : t;

    private ID3D12Resource UploadTensor(GgufType type, IntPtr hostPtr, int rows, int cols)
    {
        return D3D12TensorAlign.UploadAlignedTensor(_ctx, type, hostPtr, rows, cols);
    }

    private void DispatchGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource y,
        ID3D12Resource x,
        ulong wGpuVirtualAddress,
        int k_cols,
        int m_rows)
        => DispatchGemvBias(cmdList, type, y, x, wGpuVirtualAddress, k_cols, m_rows, null);

    private void DispatchGemvBias(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource y,
        ID3D12Resource x,
        ulong wGpuVirtualAddress,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null)
    {
        cmdList.SetComputeRootSignature(_sigGemv);
        var pso = type switch
        {
            GgufType.Q4_K => _psoGemvQ4K,
            GgufType.Q5_K => _psoGemvQ5K,
            GgufType.Q6_K => _psoGemvQ6K,
            GgufType.Q3_K => _psoGemvQ3K,
            GgufType.Q8_0 => _psoGemvQ8_0,
            GgufType.Q2_K => _psoGemvQ2_K,
            GgufType.Q2_0 => _psoGemvQ2_0,
            GgufType.IQ2_XXS => _psoGemvIQ2_XXS,
            GgufType.IQ4_NL => _psoGemvIQ4_NL,
            GgufType.IQ3_S => _psoGemvIQ3_S,
            GgufType.Ternary_1_58 or GgufType.PQ2_0 => _psoGemvTernary,
            _ => _psoGemvFp32
        };
        cmdList.SetPipelineState(pso);

        cmdList.SetComputeRootShaderResourceView(1, wGpuVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y.GPUVirtualAddress);

        const int chunkSize = D3D12TensorAlign.GemvChunkRows;
        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)k_cols;
        pConsts[1] = (uint)m_rows;
        pConsts[2] = bias != null ? 1u : 0u; // has_bias
        pConsts[3] = 0u; // has_residual
        pConsts[4] = 1u; // has_y

        for (int r = 0; r < m_rows; r += chunkSize)
        {
            int currentChunk = Math.Min(m_rows - r, chunkSize);
            pConsts[5] = (uint)r;
            cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

            cmdList.Dispatch(((uint)currentChunk + 3) / 4, 1, 1);
        }
    }

    private void DispatchRmsNorm(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource x,
        ID3D12Resource? gamma,
        ID3D12Resource y,
        int size,
        float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNorm);
        cmdList.SetPipelineState(_psoRmsNorm);

        uint* pConsts = stackalloc uint[2];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;

        cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConsts, 0);
        cmdList.SetComputeRootShaderResourceView(1, gamma?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, y.GPUVirtualAddress);

        cmdList.Dispatch(1, 1, 1);
    }

    private void DispatchRmsNormHeads(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource x,
        ID3D12Resource weight,
        int nHeads,
        int headDim,
        float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNormHeads);
        cmdList.SetPipelineState(_psoRmsNormHeads);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)headDim;
        pConsts[1] = (uint)nHeads;
        *(float*)(&pConsts[2]) = eps;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);

        cmdList.Dispatch((uint)nHeads, 1, 1);
    }

    private void DispatchVecAdd(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource a,
        ID3D12Resource b_accum,
        int size)
    {
        cmdList.SetComputeRootSignature(_sigVecAdd);
        cmdList.SetPipelineState(_psoVecAdd);

        uint* pConsts = stackalloc uint[1];
        pConsts[0] = (uint)size;

        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)pConsts, 0);
        cmdList.SetComputeRootUnorderedAccessView(1, a.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, b_accum.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchVecAddWeighted(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource a,
        ID3D12Resource b_accum,
        float weight,
        int size,
        uint accumulate)
    {
        cmdList.SetComputeRootSignature(_sigVecAddWeighted);
        cmdList.SetPipelineState(_psoVecAddWeighted);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = weight;
        pConsts[2] = accumulate;

        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);
        cmdList.SetComputeRootUnorderedAccessView(1, a.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, b_accum.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchSwiGLU(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource gate,
        ID3D12Resource up,
        ID3D12Resource output,
        int size)
    {
        cmdList.SetComputeRootSignature(_sigSwiglu);
        cmdList.SetPipelineState(_psoSwiglu);

        uint* pConsts = stackalloc uint[1];
        pConsts[0] = (uint)size;

        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)pConsts, 0);
        cmdList.SetComputeRootUnorderedAccessView(1, gate.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, up.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, output.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits)
        => ForwardStage(token, pos, default, default, logits, computeLogits, -1);

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true, int maxLayers = -1)
        => ForwardStage(token, pos, default, default, logits, computeLogits, maxLayers);

    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits)
        => ForwardStage(token, pos, inputX, outputX, logits, computeLogits, -1);

    private void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        int maxLayers)
    {
        if (StartLayer == 0)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);
        }
        else
        {
            fixed (float* pInput = inputX)
            {
                Buffer.MemoryCopy(pInput, _pUploadEmbedding, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }

        uint* ropeConsts = stackalloc uint[7];
        uint* kvConsts = stackalloc uint[4];
        uint* faConsts = stackalloc uint[6];
        uint* convConsts = stackalloc uint[1];
        uint* l2Consts = stackalloc uint[2];
        uint* dnConsts = stackalloc uint[4];
        uint* gsConsts = stackalloc uint[4];
        uint* qgConsts = stackalloc uint[2];

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding / input activation into _dX
        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        int qDim = _nHeads * _headDim;
        int kvDim = _headsKv * _headDim;

        int layersToRun = maxLayers >= 0 ? Math.Min(maxLayers, LayerCount) : LayerCount;
        for (int l = 0; l < layersToRun; l++)
        {
            var lw = _layerWeights[l];

            // 1. Attention / GDN Pre-Norm
            DispatchRmsNorm(cmd, _dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (lw.IsGdn)
            {
                // -------------------------------------------------------------
                // Gated DeltaNet (SSM Linear Attention)
                // -------------------------------------------------------------
                DispatchGemv(cmd, lw.QkvType, _dGdnQkv, _dNormX, lw.QkvWeight!.GPUVirtualAddress, _dim, _gdnConvChannels);
                DispatchGemv(cmd, lw.AttnGateType, _dGdnZ, _dNormX, lw.AttnGateWeight!.GPUVirtualAddress, _dim, _ssmInnerSize);
                DispatchGemv(cmd, lw.SsmAlphaType, _dGdnA, _dNormX, lw.SsmAlphaWeight!.GPUVirtualAddress, _dim, _ssmHeads);
                DispatchGemv(cmd, lw.SsmBetaType, _dGdnB, _dNormX, lw.SsmBetaWeight!.GPUVirtualAddress, _dim, _ssmHeads);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Causal 1D Depthwise Conv
                cmd.SetComputeRootSignature(_sigSsmConv1d);
                cmd.SetPipelineState(_psoSsmConv1d);
                convConsts[0] = (uint)_gdnConvChannels;
                cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)convConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dGdnQkv.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dConvState[l].GPUVirtualAddress);
                cmd.SetComputeRootShaderResourceView(3, lw.SsmConv1dWeight!.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(4, _dGdnConvOut.GPUVirtualAddress);
                cmd.Dispatch(((uint)_gdnConvChannels + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // L2 Normalization of Q and K per group
                cmd.SetComputeRootSignature(_sigL2NormQK);
                cmd.SetPipelineState(_psoL2NormQK);
                l2Consts[0] = (uint)_ssmGroupCount;
                l2Consts[1] = (uint)_ssmStateDim;
                cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)l2Consts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dGdnConvOut.GPUVirtualAddress);
                ulong kByteOffset = (ulong)(_ssmGroupCount * _ssmStateDim * sizeof(float));
                cmd.SetComputeRootUnorderedAccessView(2, _dGdnConvOut.GPUVirtualAddress + kByteOffset);
                cmd.Dispatch((uint)_ssmGroupCount, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Recurrent Associative Memory State Update
                cmd.SetComputeRootSignature(_sigDeltaNet);
                cmd.SetPipelineState(_psoDeltaNet);
                dnConsts[0] = (uint)_ssmHeads;
                dnConsts[1] = (uint)_ssmStateDim;
                dnConsts[2] = (uint)_ssmGroupCount;
                dnConsts[3] = lw.SsmDtBias != null ? 1u : 0u;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)dnConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dGdnConvOut.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dGdnConvOut.GPUVirtualAddress + kByteOffset);
                ulong vByteOffset = (ulong)(2 * _ssmGroupCount * _ssmStateDim * sizeof(float));
                cmd.SetComputeRootUnorderedAccessView(3, _dGdnConvOut.GPUVirtualAddress + vByteOffset);
                cmd.SetComputeRootUnorderedAccessView(4, _dGdnA.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(5, _dGdnB.GPUVirtualAddress);
                cmd.SetComputeRootShaderResourceView(6, lw.SsmDtBias?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                cmd.SetComputeRootShaderResourceView(7, lw.SsmAWeight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(8, _dSsmState[l].GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(9, _dGdnY.GPUVirtualAddress);
                cmd.Dispatch((uint)_ssmHeads, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Fused Head RMSNorm + SiLU gating on Z
                cmd.SetComputeRootSignature(_sigSsmGateSilu);
                cmd.SetPipelineState(_psoSsmGateSilu);
                gsConsts[0] = (uint)_ssmHeads;
                gsConsts[1] = (uint)_ssmStateDim;
                *(float*)(&gsConsts[2]) = _weights.RmsNormEps;
                gsConsts[3] = lw.SsmNormWeight != null ? 1u : 0u;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)gsConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dGdnY.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dGdnZ.GPUVirtualAddress);
                cmd.SetComputeRootShaderResourceView(3, lw.SsmNormWeight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                cmd.Dispatch((uint)_ssmHeads, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Output projection back to residual stream
                DispatchGemv(cmd, lw.SsmOutType, _dAttnProjOut, _dGdnY, lw.SsmOutWeight!.GPUVirtualAddress, _ssmInnerSize, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
            else
            {
                // -------------------------------------------------------------
                // Interleaved Full Attention Layer
                // -------------------------------------------------------------
                if (lw.HasQGate)
                {
                    DispatchGemv(cmd, lw.QType, _dQFull, _dNormX, lw.QWeight.GPUVirtualAddress, _dim, 2 * qDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // Split into Q and QGate
                    cmd.SetComputeRootSignature(_sigQGateSplit);
                    cmd.SetPipelineState(_psoQGateSplit);
                    qgConsts[0] = (uint)_nHeads;
                    qgConsts[1] = (uint)_headDim;
                    cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)qgConsts, 0);
                    cmd.SetComputeRootUnorderedAccessView(1, _dQFull.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(2, _dQ.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(3, _dQGate.GPUVirtualAddress);
                    cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }
                else
                {
                    DispatchGemv(cmd, lw.QType, _dQ, _dNormX, lw.QWeight.GPUVirtualAddress, _dim, qDim);
                }

                DispatchGemv(cmd, lw.KType, _dK, _dNormX, lw.KWeight.GPUVirtualAddress, _dim, kvDim);
                DispatchGemv(cmd, lw.VType, _dV, _dNormX, lw.VWeight.GPUVirtualAddress, _dim, kvDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Optional QK-Norm
                if (lw.AttnQNormWeight != null)
                {
                    DispatchRmsNormHeads(cmd, _dQ, lw.AttnQNormWeight, _nHeads, _headDim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }
                if (lw.AttnKNormWeight != null)
                {
                    DispatchRmsNormHeads(cmd, _dK, lw.AttnKNormWeight, _headsKv, _headDim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // Partial RoPE (using _ropeDim, e.g. 64 out of 256)
                cmd.SetComputeRootSignature(_sigRope);
                cmd.SetPipelineState(_psoRope);
                ropeConsts[0] = (uint)_nHeads;
                ropeConsts[1] = (uint)_headsKv;
                ropeConsts[2] = (uint)_headDim;
                ropeConsts[3] = (uint)_ropeDim;
                ropeConsts[4] = (uint)pos;
                *(float*)(&ropeConsts[5]) = _weights.RopeFreqBase;
                *(float*)(&ropeConsts[6]) = 1.0f;
                cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)ropeConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dK.GPUVirtualAddress);
                uint totalPairs = (uint)((_nHeads + _headsKv) * (_ropeDim / 2));
                cmd.Dispatch((totalPairs + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // KV Store
                cmd.SetComputeRootSignature(_sigKvStore);
                cmd.SetPipelineState(_psoKvStore);
                kvConsts[0] = (uint)_headsKv;
                kvConsts[1] = (uint)_headDim;
                kvConsts[2] = (uint)_maxSeqLen;
                kvConsts[3] = (uint)pos;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)kvConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dK.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dV.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(3, _dKeyCache[l].GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(4, _dValCache[l].GPUVirtualAddress);
                cmd.Dispatch(((uint)kvDim + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // FlashAttention / AttentionGqa (256-wide)
                cmd.SetComputeRootSignature(_sigFlashAttn);
                cmd.SetPipelineState(_psoFlashAttn);
                faConsts[0] = (uint)_nHeads;
                faConsts[1] = (uint)_headsKv;
                faConsts[2] = (uint)_headDim;
                faConsts[3] = (uint)_maxSeqLen;
                faConsts[4] = (uint)pos;
                *(float*)(&faConsts[5]) = 1.0f / MathF.Sqrt(_headDim);
                cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)faConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dKeyCache[l].GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(3, _dValCache[l].GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(4, _dAttnOut.GPUVirtualAddress);
                cmd.Dispatch((uint)_nHeads, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Post-Attention Q-Gate if present
                if (lw.HasQGate)
                {
                    cmd.SetComputeRootSignature(_sigAttnOutGate);
                    cmd.SetPipelineState(_psoAttnOutGate);
                    uint totalQ = (uint)qDim;
                    cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&totalQ), 0);
                    cmd.SetComputeRootUnorderedAccessView(1, _dAttnOut.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(2, _dQGate.GPUVirtualAddress);
                    cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // Output projection
                DispatchGemv(cmd, lw.AttnOutType, _dAttnProjOut, _dAttnOut, lw.AttnOutWeight.GPUVirtualAddress, qDim, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }

            // Residual Add: _dX = _dX + _dAttnProjOut
            DispatchVecAdd(cmd, _dAttnProjOut, _dX, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // -------------------------------------------------------------
            // Feed-Forward Network (Dense SwiGLU or GPU-First Sparse MoE)
            // -------------------------------------------------------------
            DispatchRmsNorm(cmd, _dX, lw.FfnNormWeight, _dMlpNorm, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (lw.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;
                int modelLayer = StartLayer + l;
                var layerHost = _weights.Layers[modelLayer];

                // 1. Router GEMV: compute logits for all experts into _dRouterLogits
                DispatchGemvBias(cmd, GgufType.F32, _dRouterLogits!, _dMlpNorm, lw.FfnGateInpWeight!.GPUVirtualAddress, _dim, numExperts, lw.FfnGateInpBias);
                cmd.ResourceBarrierUnorderedAccessView(_dRouterLogits!);

                _dTopKIndices ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(uint)));
                _dTopKWeights ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(float)));

                bool hasShexp = (lw.FfnGateShexpWeight != null);
                bool hasShexpGate = hasShexp && layerHost.FfnGateInpShexpWeight != null;
                if (hasShexpGate && _dShexpGateScalar == null)
                    _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));

                // 2. GPU Softmax and Top-K directly in VRAM (zero host sync)
                _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogits!, _dTopKIndices, _dTopKWeights, numExperts, topK, _weights.NormTopK);
                cmd.ResourceBarrierUnorderedAccessView(_dTopKIndices);
                cmd.ResourceBarrierUnorderedAccessView(_dTopKWeights);

                // 3. Routed Experts Execution via Indirected GEMV (Zero-Stall)
                ulong gateSliceBytes = lw.FfnGateExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnGateExpsType, expertFfnDim, _dim) : 0;
                ulong upSliceBytes = lw.FfnUpExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnUpExpsType, expertFfnDim, _dim) : 0;
                ulong downSliceBytes = lw.FfnDownExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnDownExpsType, _dim, expertFfnDim) : 0;

                for (int k = 0; k < topK; k++)
                {
                    _ctx.DispatchExpertGemv(cmd, lw.FfnGateExpsType, _dExpertGate!, _dMlpNorm, lw.FfnGateExpsWeight!, _dTopKIndices, k, gateSliceBytes, _dim, expertFfnDim);
                    _ctx.DispatchExpertGemv(cmd, lw.FfnUpExpsType, _dExpertUp!, _dMlpNorm, lw.FfnUpExpsWeight!, _dTopKIndices, k, upSliceBytes, _dim, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchSwiGLU(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    _ctx.DispatchExpertGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut!, _dExpertAct!, lw.FfnDownExpsWeight!, _dTopKIndices, k, downSliceBytes, expertFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dTopKWeights, k, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // 4. Shared Expert if present
                if (hasShexp)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        shexpFfnDim = (int)tShexp.Dimensions[1];

                    DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate!, _dMlpNorm, lw.FfnGateShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
                    DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp!, _dMlpNorm, lw.FfnUpShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchSwiGLU(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut!, _dShexpAct!, lw.FfnDownShexpWeight!.GPUVirtualAddress, shexpFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    if (hasShexpGate)
                    {
                        var dShexpGateW = GetOrCreateShexpGateWeight(modelLayer, layerHost.FfnGateInpShexpWeight);
                        _ctx.DispatchMoeSharedExpertGate(cmd, _dMlpNorm, dShexpGateW, _dShexpGateScalar!, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar!);

                        _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar!, 0, _dim, accumulate: 1);
                    }
                    else
                    {
                        DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                    }
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // 5. Residual connection: _dX += _dFfnOut
                DispatchVecAdd(cmd, _dFfnOut!, _dX, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
            else
            {
                DispatchGemv(cmd, lw.FfnGateType, _dMlpGate, _dMlpNorm, lw.FfnGateWeight!.GPUVirtualAddress, _dim, _ffnDim);
                DispatchGemv(cmd, lw.FfnUpType, _dMlpUp, _dMlpNorm, lw.FfnUpWeight!.GPUVirtualAddress, _dim, _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchSwiGLU(cmd, _dMlpGate, _dMlpUp, _dMlpAct, _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchGemv(cmd, lw.FfnDownType, _dMlpOut, _dMlpAct, lw.FfnDownWeight!.GPUVirtualAddress, _ffnDim, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Residual Add: _dX = _dX + _dMlpOut
                DispatchVecAdd(cmd, _dMlpOut, _dX, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
        }

        // Output Head or Intermediate Stage Readback
        if (IsLastStage)
        {
            if (computeLogits && logits.Length > 0)
            {
                DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dOutNorm, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchGemv(cmd, _weights.OutType, _dLogits, _dOutNorm, _dOutWeight!.GPUVirtualAddress, _dim, _vocabSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_vocabSize * sizeof(float)));
                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();

                fixed (float* pLogits = logits)
                {
                    Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_vocabSize * sizeof(float)), (ulong)(_vocabSize * sizeof(float)));
                }
            }
            else
            {
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();
            }
        }
        else
        {
            if (!outputX.IsEmpty && _readbackActivation != null)
            {
                cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackActivation, 0, _dX, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.CopySource, ResourceStates.Common);
            }
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            if (!outputX.IsEmpty && _pReadbackActivation != null)
            {
                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_pReadbackActivation, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
            }
        }
    }

    private void ComputeTieredRoutedExpertsCpu(
        int modelLayer,
        int expertFfnDim,
        int topK,
        int* selectedIndices,
        float* selectedWeights)
    {
        var lwHost = _weights.Layers[modelLayer];
        long gateSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(lwHost.FfnGateExpsType, _dim);
        long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(lwHost.FfnUpExpsType, _dim);
        long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(lwHost.FfnDownExpsType, expertFfnDim);

        fixed (float* pNormXSums = _normXSums)
        {
            QuantKernels.ComputeBlockSums32(_pReadbackNormX, pNormXSums, _dim);
            new Span<float>(_pUploadExpertOut, _dim).Clear();

            var ws = _expertWorkspaces![0];
            fixed (float* pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
            {
                for (int k = 0; k < topK; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    byte* expGateWeight = lwHost.FfnGateExpsWeight + (long)expertIdx * gateSliceBytes;
                    byte* expUpWeight = lwHost.FfnUpExpsWeight + (long)expertIdx * upSliceBytes;
                    byte* expDownWeight = lwHost.FfnDownExpsWeight + (long)expertIdx * downSliceBytes;

                    QuantKernels.MatVecMul(lwHost.FfnGateExpsType, expGateWeight, _pReadbackNormX, pGate, _dim, expertFfnDim, pNormXSums);
                    if (lwHost.FfnGateExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnGateExpsBias + (long)expertIdx * expertFfnDim, pGate, expertFfnDim);

                    QuantKernels.MatVecMul(lwHost.FfnUpExpsType, expUpWeight, _pReadbackNormX, pUp, _dim, expertFfnDim, pNormXSums);
                    if (lwHost.FfnUpExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnUpExpsBias + (long)expertIdx * expertFfnDim, pUp, expertFfnDim);

                    QuantKernels.SwiGLU(pGate, pUp, pAct, expertFfnDim);
                    QuantKernels.ComputeBlockSums32(pAct, pActSums, expertFfnDim);

                    QuantKernels.MatVecMul(lwHost.FfnDownExpsType, expDownWeight, pAct, pDownOut, expertFfnDim, _dim, pActSums);
                    if (lwHost.FfnDownExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnDownExpsBias + (long)expertIdx * _dim, pDownOut, _dim);

                    QuantKernels.VecAddWeighted(pDownOut, _pUploadExpertOut, weight, _dim);
                }
            }
        }
    }

    private ID3D12Resource GetOrCreateShexpGateWeight(int layerIdx, float* pWeight)
    {
        if (!_dShexpGateWeights.TryGetValue(layerIdx, out var dBuf))
        {
            dBuf = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dBuf, (IntPtr)pWeight, (ulong)(_dim * sizeof(float)));
            _dShexpGateWeights[layerIdx] = dBuf;
        }
        return dBuf;
    }

    private void EnsureBatchBuffers(int chunkCapacity)
    {
        if (_allocatedBatchCapacity >= chunkCapacity)
            return;

        int newCap = Math.Max(chunkCapacity, Math.Max(_allocatedBatchCapacity * 2, 32));

        _dXBatch?.Dispose();
        _dNormXBatch?.Dispose();
        if (_uploadEmbeddingBatch != null)
        {
            _uploadEmbeddingBatch.Unmap(0);
            _uploadEmbeddingBatch.Dispose();
            _uploadEmbeddingBatch = null;
            _pUploadEmbeddingBatch = null;
        }
        _dQBatch?.Dispose();
        _dQGateBatch?.Dispose();
        _dQFullBatch?.Dispose();
        _dKBatch?.Dispose();
        _dVBatch?.Dispose();
        _dAttnOutBatch?.Dispose();
        _dGdnQkvBatch?.Dispose();
        _dGdnConvOutBatch?.Dispose();
        _dGdnZBatch?.Dispose();
        _dGdnABatch?.Dispose();
        _dGdnBBatch?.Dispose();
        _dGdnYBatch?.Dispose();
        _dAttnProjOutBatch?.Dispose();
        _dMlpNormBatch?.Dispose();
        _dMlpGateBatch?.Dispose();
        _dMlpUpBatch?.Dispose();
        _dMlpActBatch?.Dispose();
        _dMlpOutBatch?.Dispose();
        _dRouterLogitsBatch?.Dispose();
        _dTopKIndicesBatch?.Dispose();
        _dTopKWeightsBatch?.Dispose();

        int qDim = _nHeads * _headDim;
        int kvDim = _headsKv * _headDim;
        int mlpDim = Math.Max(1, _ffnDim);

        _dXBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _dim * sizeof(float)));
        _dNormXBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _dim * sizeof(float)));

        _uploadEmbeddingBatch = _ctx.CreateUploadBuffer((ulong)(newCap * _dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbeddingBatch.Map(0, null, &pUpload);
        _pUploadEmbeddingBatch = (float*)pUpload;

        _dQBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * qDim * sizeof(float)));
        _dQGateBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * qDim * sizeof(float)));
        _dQFullBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * 2 * qDim * sizeof(float)));
        _dKBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * kvDim * sizeof(float)));
        _dVBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * kvDim * sizeof(float)));
        _dAttnOutBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * qDim * sizeof(float)));

        _dGdnQkvBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _gdnConvChannels * sizeof(float)));
        _dGdnConvOutBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _gdnConvChannels * sizeof(float)));
        _dGdnZBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _ssmInnerSize * sizeof(float)));
        _dGdnABatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _ssmHeads * sizeof(float)));
        _dGdnBBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _ssmHeads * sizeof(float)));
        _dGdnYBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _ssmInnerSize * sizeof(float)));
        _dAttnProjOutBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _dim * sizeof(float)));

        _dMlpNormBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _dim * sizeof(float)));
        _dMlpGateBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * mlpDim * sizeof(float)));
        _dMlpUpBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * mlpDim * sizeof(float)));
        _dMlpActBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * mlpDim * sizeof(float)));
        _dMlpOutBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * _dim * sizeof(float)));

        if (_weights.IsMoe)
        {
            int numExperts = _weights.ExpertCount;
            int topK = _weights.ExpertUsedCount;
            _dRouterLogitsBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * numExperts * sizeof(float)));
            _dTopKIndicesBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * topK * sizeof(uint)));
            _dTopKWeightsBatch = _ctx.CreateDeviceBuffer((ulong)(newCap * topK * sizeof(float)));
        }

        _allocatedBatchCapacity = newCap;
    }

    private void DispatchGemmBatch(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ulong wGpuVirtualAddress,
        int kCols,
        int mRows,
        int batchSize,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
    {
        cmdList.SetComputeRootSignature(_sigGemmBatch);
        if (type == GgufType.Q4_K)
            cmdList.SetPipelineState(_psoGemmQ4KBatch);
        else if (type == GgufType.Q6_K)
            cmdList.SetPipelineState(_psoGemmQ6KBatch);
        else if (type == GgufType.Q8_0)
            cmdList.SetPipelineState(_psoGemmQ8_0Batch);
        else if (type == GgufType.F32)
            cmdList.SetPipelineState(_psoGemmFp32Batch);
        else if (type == GgufType.Q3_K)
            cmdList.SetPipelineState(_psoGemmQ3KBatch);
        else if (type == GgufType.Q5_K)
            cmdList.SetPipelineState(_psoGemmQ5KBatch);
        else if (type == GgufType.Q2_K)
            cmdList.SetPipelineState(_psoGemmQ2_KBatch);
        else if (type == GgufType.Q2_0)
            cmdList.SetPipelineState(_psoGemmQ2_0Batch);
        else if (type == GgufType.IQ2_XXS)
            cmdList.SetPipelineState(_psoGemmIQ2_XXSBatch);
        else if (type == GgufType.IQ4_NL)
            cmdList.SetPipelineState(_psoGemmIQ4_NLBatch);
        else if (type == GgufType.IQ3_S)
            cmdList.SetPipelineState(_psoGemmIQ3_SBatch);
        else if (type is GgufType.Ternary_1_58 or GgufType.PQ2_0)
            cmdList.SetPipelineState(_psoGemmTernaryBatch);
        else
            throw new NotSupportedException($"Direct3D 12 batch GEMM does not support tensor quantization type {type}.");

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = (uint)(bias != null ? 1 : 0);
        pConsts[4] = (uint)(residual != null ? 1 : 0);
        pConsts[5] = (uint)(y != null ? 1 : 0);
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, wGpuVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias != null ? bias.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual != null ? residual.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y != null ? y.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);

        bool is8TokenTile = type is GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.F32 or GgufType.Q3_K or GgufType.Q5_K or GgufType.Ternary_1_58 or GgufType.PQ2_0;
        uint groupsY = is8TokenTile ? (uint)(batchSize + 7) / 8 : (uint)(batchSize + 31) / 32;
        cmdList.Dispatch((uint)(mRows + 3) / 4, groupsY, 1);
    }

    private void DispatchRmsNormBatch(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource x,
        ID3D12Resource weight,
        ID3D12Resource dst,
        int size,
        int batchSize,
        float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNormBatch);
        cmdList.SetPipelineState(_psoRmsNormBatch);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;
        pConsts[2] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch((uint)batchSize, 1, 1);
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits);
    }

    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        int totalTokens = !tokens.IsEmpty ? tokens.Length : (inputXBatch.Length / _dim);
        if (totalTokens == 0) return;

        if (totalTokens == 1)
        {
            ForwardStage(
                !tokens.IsEmpty ? tokens[0] : 0,
                startPos,
                inputXBatch,
                outputXBatch,
                logits,
                computeLogits);
            return;
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _headsKv * _headDim;
        int maxChunk = 32;
        EnsureBatchBuffers(maxChunk);

        uint* convConsts = stackalloc uint[1];
        uint* l2Consts = stackalloc uint[2];
        uint* dnConsts = stackalloc uint[4];
        uint* gsConsts = stackalloc uint[4];
        uint* qgConsts = stackalloc uint[2];
        uint* ropeConsts = stackalloc uint[7];
        uint* kvConsts = stackalloc uint[4];
        uint* faConsts = stackalloc uint[6];

        int offset = 0;
        while (offset < totalTokens)
        {
            int chunkSize = Math.Min(maxChunk, totalTokens - offset);
            if (chunkSize > 8 && (chunkSize % 8) == 1)
            {
                chunkSize -= 1;
            }
            bool isLastChunk = (offset + chunkSize == totalTokens);
            int chunkStartPos = startPos + offset;

            if (chunkSize == 1)
            {
                ForwardStage(
                    !tokens.IsEmpty ? tokens[offset] : 0,
                    chunkStartPos,
                    !inputXBatch.IsEmpty ? inputXBatch.Slice(offset * _dim, _dim) : default,
                    !outputXBatch.IsEmpty ? outputXBatch.Slice(offset * _dim, _dim) : default,
                    logits,
                    isLastChunk && computeLogits);
                offset += 1;
                continue;
            }

            // 1. Vectorized host embedding extraction or intermediate activation copy for the chunk
            if (StartLayer == 0)
            {
                fixed (int* pTokens = tokens)
                {
                    for (int t = 0; t < chunkSize; t++)
                    {
                        QuantKernels.ExtractEmbedding(
                            _weights.EmbdType,
                            _weights.EmbdWeight,
                            pTokens[offset + t],
                            _pUploadEmbeddingBatch + t * _dim,
                            _dim);
                    }
                }
            }
            else
            {
                fixed (float* pInput = inputXBatch)
                {
                    Buffer.MemoryCopy(
                        pInput + (long)offset * _dim,
                        _pUploadEmbeddingBatch,
                        (ulong)(chunkSize * _dim * sizeof(float)),
                        (ulong)(chunkSize * _dim * sizeof(float)));
                }
            }

            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;

            // Copy entire chunk of embeddings/activations into _dXBatch
            cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dXBatch!, 0, _uploadEmbeddingBatch!, 0, (ulong)(chunkSize * _dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

            for (int l = 0; l < LayerCount; l++)
            {
                var lw = _layerWeights[l];

                // Attention / GDN Pre-Norm
                DispatchRmsNormBatch(cmd, _dXBatch!, lw.AttnNormWeight, _dNormXBatch!, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(_dNormXBatch!);

                if (lw.IsGdn)
                {
                    // -------------------------------------------------------------
                    // Gated DeltaNet (SSM Linear Attention)
                    // -------------------------------------------------------------
                    DispatchGemmBatch(cmd, lw.QkvType, _dGdnQkvBatch!, _dNormXBatch!, lw.QkvWeight!.GPUVirtualAddress, _dim, _gdnConvChannels, chunkSize);
                    DispatchGemmBatch(cmd, lw.AttnGateType, _dGdnZBatch!, _dNormXBatch!, lw.AttnGateWeight!.GPUVirtualAddress, _dim, _ssmInnerSize, chunkSize);
                    DispatchGemmBatch(cmd, lw.SsmAlphaType, _dGdnABatch!, _dNormXBatch!, lw.SsmAlphaWeight!.GPUVirtualAddress, _dim, _ssmHeads, chunkSize);
                    DispatchGemmBatch(cmd, lw.SsmBetaType, _dGdnBBatch!, _dNormXBatch!, lw.SsmBetaWeight!.GPUVirtualAddress, _dim, _ssmHeads, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dGdnQkvBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dGdnZBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dGdnABatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dGdnBBatch!);

                    for (int t = 0; t < chunkSize; t++)
                    {
                        ulong qkvOffset = (ulong)(t * _gdnConvChannels * sizeof(float));
                        ulong convOutOffset = (ulong)(t * _gdnConvChannels * sizeof(float));
                        ulong kByteOffset = (ulong)(_ssmGroupCount * _ssmStateDim * sizeof(float));
                        ulong vByteOffset = (ulong)(2 * _ssmGroupCount * _ssmStateDim * sizeof(float));
                        ulong aOffset = (ulong)(t * _ssmHeads * sizeof(float));
                        ulong bOffset = (ulong)(t * _ssmHeads * sizeof(float));
                        ulong yOffset = (ulong)(t * _ssmInnerSize * sizeof(float));
                        ulong zOffset = (ulong)(t * _ssmInnerSize * sizeof(float));

                        // Causal 1D Depthwise Conv
                        cmd.SetComputeRootSignature(_sigSsmConv1d);
                        cmd.SetPipelineState(_psoSsmConv1d);
                        convConsts[0] = (uint)_gdnConvChannels;
                        cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)convConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dGdnQkvBatch!.GPUVirtualAddress + qkvOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dConvState[l].GPUVirtualAddress);
                        cmd.SetComputeRootShaderResourceView(3, lw.SsmConv1dWeight!.GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(4, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset);
                        cmd.Dispatch(((uint)_gdnConvChannels + 255) / 256, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dGdnConvOutBatch!);

                        // L2 Normalization of Q and K per group
                        cmd.SetComputeRootSignature(_sigL2NormQK);
                        cmd.SetPipelineState(_psoL2NormQK);
                        l2Consts[0] = (uint)_ssmGroupCount;
                        l2Consts[1] = (uint)_ssmStateDim;
                        cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)l2Consts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset + kByteOffset);
                        cmd.Dispatch((uint)_ssmGroupCount, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dGdnConvOutBatch!);

                        // Recurrent Associative Memory State Update
                        cmd.SetComputeRootSignature(_sigDeltaNet);
                        cmd.SetPipelineState(_psoDeltaNet);
                        dnConsts[0] = (uint)_ssmHeads;
                        dnConsts[1] = (uint)_ssmStateDim;
                        dnConsts[2] = (uint)_ssmGroupCount;
                        dnConsts[3] = lw.SsmDtBias != null ? 1u : 0u;
                        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)dnConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset + kByteOffset);
                        cmd.SetComputeRootUnorderedAccessView(3, _dGdnConvOutBatch!.GPUVirtualAddress + convOutOffset + vByteOffset);
                        cmd.SetComputeRootUnorderedAccessView(4, _dGdnABatch!.GPUVirtualAddress + aOffset);
                        cmd.SetComputeRootUnorderedAccessView(5, _dGdnBBatch!.GPUVirtualAddress + bOffset);
                        cmd.SetComputeRootShaderResourceView(6, lw.SsmDtBias?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                        cmd.SetComputeRootShaderResourceView(7, lw.SsmAWeight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(8, _dSsmState[l].GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(9, _dGdnYBatch!.GPUVirtualAddress + yOffset);
                        cmd.Dispatch((uint)_ssmHeads, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dGdnYBatch!);

                        // Fused Head RMSNorm + SiLU gating on Z
                        cmd.SetComputeRootSignature(_sigSsmGateSilu);
                        cmd.SetPipelineState(_psoSsmGateSilu);
                        gsConsts[0] = (uint)_ssmHeads;
                        gsConsts[1] = (uint)_ssmStateDim;
                        *(float*)(&gsConsts[2]) = _weights.RmsNormEps;
                        gsConsts[3] = lw.SsmNormWeight != null ? 1u : 0u;
                        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)gsConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dGdnYBatch!.GPUVirtualAddress + yOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dGdnZBatch!.GPUVirtualAddress + zOffset);
                        cmd.SetComputeRootShaderResourceView(3, lw.SsmNormWeight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                        cmd.Dispatch((uint)_ssmHeads, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dGdnYBatch!);
                    }

                    // Output projection with fused residual addition
                    DispatchGemmBatch(cmd, lw.SsmOutType, null, _dGdnYBatch!, lw.SsmOutWeight!.GPUVirtualAddress, _ssmInnerSize, _dim, chunkSize, residual: _dXBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dXBatch!);
                }
                else
                {
                    // -------------------------------------------------------------
                    // Interleaved Full Attention Layer
                    // -------------------------------------------------------------
                    if (lw.HasQGate)
                    {
                        DispatchGemmBatch(cmd, lw.QType, _dQFullBatch!, _dNormXBatch!, lw.QWeight.GPUVirtualAddress, _dim, 2 * qDim, chunkSize);
                        cmd.ResourceBarrierUnorderedAccessView(_dQFullBatch!);

                        cmd.SetComputeRootSignature(_sigQGateSplit);
                        cmd.SetPipelineState(_psoQGateSplit);
                        qgConsts[0] = (uint)_nHeads;
                        qgConsts[1] = (uint)_headDim;
                        for (int t = 0; t < chunkSize; t++)
                        {
                            cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)qgConsts, 0);
                            cmd.SetComputeRootUnorderedAccessView(1, _dQFullBatch!.GPUVirtualAddress + (ulong)(t * 2 * qDim * sizeof(float)));
                            cmd.SetComputeRootUnorderedAccessView(2, _dQBatch!.GPUVirtualAddress + (ulong)(t * qDim * sizeof(float)));
                            cmd.SetComputeRootUnorderedAccessView(3, _dQGateBatch!.GPUVirtualAddress + (ulong)(t * qDim * sizeof(float)));
                            cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
                        }
                        cmd.ResourceBarrierUnorderedAccessView(_dQBatch!);
                        cmd.ResourceBarrierUnorderedAccessView(_dQGateBatch!);
                    }
                    else
                    {
                        DispatchGemmBatch(cmd, lw.QType, _dQBatch!, _dNormXBatch!, lw.QWeight.GPUVirtualAddress, _dim, qDim, chunkSize);
                        cmd.ResourceBarrierUnorderedAccessView(_dQBatch!);
                    }

                    DispatchGemmBatch(cmd, lw.KType, _dKBatch!, _dNormXBatch!, lw.KWeight.GPUVirtualAddress, _dim, kvDim, chunkSize);
                    DispatchGemmBatch(cmd, lw.VType, _dVBatch!, _dNormXBatch!, lw.VWeight.GPUVirtualAddress, _dim, kvDim, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dKBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dVBatch!);

                    if (lw.AttnQNormWeight != null)
                    {
                        DispatchRmsNormHeads(cmd, _dQBatch!, lw.AttnQNormWeight, chunkSize * _nHeads, _headDim, _weights.RmsNormEps);
                        cmd.ResourceBarrierUnorderedAccessView(_dQBatch!);
                    }
                    if (lw.AttnKNormWeight != null)
                    {
                        DispatchRmsNormHeads(cmd, _dKBatch!, lw.AttnKNormWeight, chunkSize * _headsKv, _headDim, _weights.RmsNormEps);
                        cmd.ResourceBarrierUnorderedAccessView(_dKBatch!);
                    }

                    for (int t = 0; t < chunkSize; t++)
                    {
                        int pos = chunkStartPos + t;
                        ulong qOffset = (ulong)(t * qDim * sizeof(float));
                        ulong kvOffset = (ulong)(t * kvDim * sizeof(float));

                        // Partial RoPE
                        cmd.SetComputeRootSignature(_sigRope);
                        cmd.SetPipelineState(_psoRope);
                        ropeConsts[0] = (uint)_nHeads;
                        ropeConsts[1] = (uint)_headsKv;
                        ropeConsts[2] = (uint)_headDim;
                        ropeConsts[3] = (uint)_ropeDim;
                        ropeConsts[4] = (uint)pos;
                        *(float*)(&ropeConsts[5]) = _weights.RopeFreqBase;
                        *(float*)(&ropeConsts[6]) = 1.0f;
                        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)ropeConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dQBatch!.GPUVirtualAddress + qOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dKBatch!.GPUVirtualAddress + kvOffset);
                        uint totalPairs = (uint)((_nHeads + _headsKv) * (_ropeDim / 2));
                        cmd.Dispatch((totalPairs + 255) / 256, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dKBatch!);

                        // KV Store
                        cmd.SetComputeRootSignature(_sigKvStore);
                        cmd.SetPipelineState(_psoKvStore);
                        kvConsts[0] = (uint)_headsKv;
                        kvConsts[1] = (uint)_headDim;
                        kvConsts[2] = (uint)_maxSeqLen;
                        kvConsts[3] = (uint)pos;
                        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)kvConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dKBatch!.GPUVirtualAddress + kvOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dVBatch!.GPUVirtualAddress + kvOffset);
                        cmd.SetComputeRootUnorderedAccessView(3, _dKeyCache[l].GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(4, _dValCache[l].GPUVirtualAddress);
                        cmd.Dispatch(((uint)kvDim + 255) / 256, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dKeyCache[l]);
                        cmd.ResourceBarrierUnorderedAccessView(_dValCache[l]);

                        // FlashAttention
                        cmd.SetComputeRootSignature(_sigFlashAttn);
                        cmd.SetPipelineState(_psoFlashAttn);
                        faConsts[0] = (uint)_nHeads;
                        faConsts[1] = (uint)_headsKv;
                        faConsts[2] = (uint)_headDim;
                        faConsts[3] = (uint)_maxSeqLen;
                        faConsts[4] = (uint)pos;
                        *(float*)(&faConsts[5]) = 1.0f / MathF.Sqrt(_headDim);
                        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)faConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dQBatch!.GPUVirtualAddress + qOffset);
                        cmd.SetComputeRootUnorderedAccessView(2, _dKeyCache[l].GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(3, _dValCache[l].GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(4, _dAttnOutBatch!.GPUVirtualAddress + qOffset);
                        cmd.Dispatch((uint)_nHeads, 1, 1);
                        cmd.ResourceBarrierUnorderedAccessView(_dAttnOutBatch!);

                        if (lw.HasQGate)
                        {
                            cmd.SetComputeRootSignature(_sigAttnOutGate);
                            cmd.SetPipelineState(_psoAttnOutGate);
                            uint totalQ = (uint)qDim;
                            cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&totalQ), 0);
                            cmd.SetComputeRootUnorderedAccessView(1, _dAttnOutBatch!.GPUVirtualAddress + qOffset);
                            cmd.SetComputeRootUnorderedAccessView(2, _dQGateBatch!.GPUVirtualAddress + qOffset);
                            cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
                            cmd.ResourceBarrierUnorderedAccessView(_dAttnOutBatch!);
                        }
                    }

                    // AttnOut projection with fused residual addition
                    DispatchGemmBatch(cmd, lw.AttnOutType, null, _dAttnOutBatch!, lw.AttnOutWeight.GPUVirtualAddress, qDim, _dim, chunkSize, residual: _dXBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dXBatch!);
                }

                // -------------------------------------------------------------
                // Feed-Forward Network (Dense SwiGLU or Sparse MoE)
                // -------------------------------------------------------------
                DispatchRmsNormBatch(cmd, _dXBatch!, lw.FfnNormWeight, _dMlpNormBatch!, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(_dMlpNormBatch!);

                if (!lw.IsMoe)
                {
                    DispatchGemmBatch(cmd, lw.FfnGateType, _dMlpGateBatch!, _dMlpNormBatch!, lw.FfnGateWeight!.GPUVirtualAddress, _dim, _ffnDim, chunkSize);
                    DispatchGemmBatch(cmd, lw.FfnUpType, _dMlpUpBatch!, _dMlpNormBatch!, lw.FfnUpWeight!.GPUVirtualAddress, _dim, _ffnDim, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dMlpGateBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dMlpUpBatch!);

                    DispatchSwiGLU(cmd, _dMlpGateBatch!, _dMlpUpBatch!, _dMlpActBatch!, chunkSize * _ffnDim);
                    cmd.ResourceBarrierUnorderedAccessView(_dMlpActBatch!);

                    DispatchGemmBatch(cmd, lw.FfnDownType, null, _dMlpActBatch!, lw.FfnDownWeight!.GPUVirtualAddress, _ffnDim, _dim, chunkSize, residual: _dXBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dXBatch!);
                }
                else
                {
                    int expertFfnDim = _weights.ExpertFeedForwardLength;
                    int numExperts = _weights.ExpertCount;
                    int topK = _weights.ExpertUsedCount;
                    int modelLayer = StartLayer + l;
                    var layerHost = _weights.Layers[modelLayer];

                    DispatchGemmBatch(cmd, GgufType.F32, _dRouterLogitsBatch!, _dMlpNormBatch!, lw.FfnGateInpWeight!.GPUVirtualAddress, _dim, numExperts, chunkSize, lw.FfnGateInpBias);
                    cmd.ResourceBarrierUnorderedAccessView(_dRouterLogitsBatch!);

                    _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogitsBatch!, _dTopKIndicesBatch!, _dTopKWeightsBatch!, numExperts, topK, _weights.NormTopK, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKIndicesBatch!);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKWeightsBatch!);

                    ulong gateSliceBytes = lw.FfnGateExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnGateExpsType, expertFfnDim, _dim) : 0;
                    ulong upSliceBytes = lw.FfnUpExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnUpExpsType, expertFfnDim, _dim) : 0;
                    ulong downSliceBytes = lw.FfnDownExpsWeight != null ? D3D12TensorAlign.GetAlignedTensorBytes(lw.FfnDownExpsType, _dim, expertFfnDim) : 0;

                    bool hasShexp = (lw.FfnGateShexpWeight != null);
                    bool hasShexpGate = hasShexp && layerHost.FfnGateInpShexpWeight != null;
                    if (hasShexpGate && _dShexpGateScalar == null)
                        _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));

                    for (int t = 0; t < chunkSize; t++)
                    {
                        ulong normOffset = (ulong)(t * _dim * sizeof(float));
                        ulong xOffset = (ulong)(t * _dim * sizeof(float));

                        cmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dMlpNorm, 0, _dMlpNormBatch!, normOffset, (ulong)(_dim * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                        for (int k = 0; k < topK; k++)
                        {
                            int tokenKIndex = t * topK + k;
                            _ctx.DispatchExpertGemv(cmd, lw.FfnGateExpsType, _dExpertGate!, _dMlpNorm, lw.FfnGateExpsWeight!, _dTopKIndicesBatch!, tokenKIndex, gateSliceBytes, _dim, expertFfnDim);
                            _ctx.DispatchExpertGemv(cmd, lw.FfnUpExpsType, _dExpertUp!, _dMlpNorm, lw.FfnUpExpsWeight!, _dTopKIndicesBatch!, tokenKIndex, upSliceBytes, _dim, expertFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchSwiGLU(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            _ctx.DispatchExpertGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut!, _dExpertAct!, lw.FfnDownExpsWeight!, _dTopKIndicesBatch!, tokenKIndex, downSliceBytes, expertFfnDim, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dTopKWeightsBatch!, tokenKIndex, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                            cmd.ResourceBarrierUnorderedAccessView(null!);
                        }

                        if (hasShexp)
                        {
                            int shexpFfnDim = expertFfnDim * 2;
                            if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                                shexpFfnDim = (int)tShexp.Dimensions[1];

                            DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate!, _dMlpNorm, lw.FfnGateShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
                            DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp!, _dMlpNorm, lw.FfnUpShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchSwiGLU(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut!, _dShexpAct!, lw.FfnDownShexpWeight!.GPUVirtualAddress, shexpFfnDim, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            if (hasShexpGate)
                            {
                                var dShexpGateW = GetOrCreateShexpGateWeight(modelLayer, layerHost.FfnGateInpShexpWeight);
                                _ctx.DispatchMoeSharedExpertGate(cmd, _dMlpNorm, dShexpGateW, _dShexpGateScalar!, _dim);
                                cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar!);

                                _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar!, 0, _dim, accumulate: 1);
                            }
                            else
                            {
                                DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                            }
                            cmd.ResourceBarrierUnorderedAccessView(null!);
                        }

                        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dX, 0, _dXBatch!, xOffset, (ulong)(_dim * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                        DispatchVecAdd(cmd, _dFfnOut!, _dX, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dX);

                        cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dXBatch!, xOffset, _dX, 0, (ulong)(_dim * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopySource, ResourceStates.Common);
                    }
                }
            }

            if (IsLastStage)
            {
                if (computeLogits && isLastChunk)
                {
                    ulong lastTokenOffset = (ulong)((chunkSize - 1) * _dim * sizeof(float));
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
                    cmd.CopyBufferRegion(_dX, 0, _dXBatch!, lastTokenOffset, (ulong)(_dim * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                    DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dOutNorm, _dim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(_dOutNorm);

                    DispatchGemv(cmd, _weights.OutType, _dLogits, _dOutNorm, _dOutWeight!.GPUVirtualAddress, _dim, _vocabSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dLogits);

                    if (!logits.IsEmpty)
                    {
                        cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_vocabSize * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
                    }
                    else
                    {
                        cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.Common);
                    }
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.Common);
                }
            }
            else
            {
                if (!outputXBatch.IsEmpty && _readbackActivation != null)
                {
                    cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    ulong totalBytes = (ulong)(chunkSize * _dim * sizeof(float));
                    cmd.CopyBufferRegion(_readbackActivation, 0, _dXBatch!, 0, totalBytes);
                    cmd.ResourceBarrierTransition(_dXBatch!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                }
            }

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            if (IsLastStage && computeLogits && isLastChunk && !logits.IsEmpty)
            {
                fixed (float* pDst = logits)
                {
                    Buffer.MemoryCopy(_pReadbackLogits, pDst, (ulong)(_vocabSize * sizeof(float)), (ulong)(_vocabSize * sizeof(float)));
                }
            }
            else if (!IsLastStage && !outputXBatch.IsEmpty && _pReadbackActivation != null)
            {
                fixed (float* pOutput = outputXBatch)
                {
                    Buffer.MemoryCopy(
                        _pReadbackActivation,
                        pOutput + (long)offset * _dim,
                        (ulong)(chunkSize * _dim * sizeof(float)),
                        (ulong)(chunkSize * _dim * sizeof(float)));
                }
            }

            offset += chunkSize;
        }
    }

    public void ResetKvCache()
    {
        // Zero out recurrent SSM states (conv1d state and associative memory state) across GDN layers
        int convFloats = _gdnConvChannels * 3;
        int ssmFloats = _ssmHeads * _ssmStateDim * _ssmStateDim;
        int maxFloats = Math.Max(convFloats, ssmFloats);
        if (maxFloats <= 0) return;

        float[] zeros = new float[maxFloats];
        fixed (float* pZeros = zeros)
        {
            for (int l = 0; l < LayerCount; l++)
            {
                if (_dConvState[l] != null)
                {
                    _ctx.CopyToDevice(_dConvState[l], (IntPtr)pZeros, (ulong)(convFloats * sizeof(float)));
                }
                if (_dSsmState[l] != null)
                {
                    _ctx.CopyToDevice(_dSsmState[l], (IntPtr)pZeros, (ulong)(ssmFloats * sizeof(float)));
                }
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _ctx.Synchronize();

            foreach (var lw in _layerWeights) lw.Dispose();
            foreach (var res in _dConvState) res?.Dispose();
            foreach (var res in _dSsmState) res?.Dispose();
            foreach (var res in _dKeyCache) res?.Dispose();
            foreach (var res in _dValCache) res?.Dispose();

            _dX?.Dispose();
            _dNormX?.Dispose();
            _dQ?.Dispose();
            _dQGate?.Dispose();
            _dK?.Dispose();
            _dV?.Dispose();
            _dQFull?.Dispose();
            _dAttnOut?.Dispose();
            _dAttnProjOut?.Dispose();

            _dGdnQkv?.Dispose();
            _dGdnConvOut?.Dispose();
            _dGdnZ?.Dispose();
            _dGdnA?.Dispose();
            _dGdnB?.Dispose();
            _dGdnY?.Dispose();

            _dMlpNorm?.Dispose();
            _dMlpGate?.Dispose();
            _dMlpUp?.Dispose();
            _dMlpAct?.Dispose();
            _dMlpOut?.Dispose();

            if (_readbackRouterLogits != null)
            {
                _readbackRouterLogits.Unmap(0);
                _readbackRouterLogits.Dispose();
            }
            if (_readbackNormX != null)
            {
                _readbackNormX.Unmap(0);
                _readbackNormX.Dispose();
            }
            if (_uploadExpertOut != null)
            {
                _uploadExpertOut.Unmap(0);
                _uploadExpertOut.Dispose();
            }
            _dRouterLogits?.Dispose();
            _dExpertGate?.Dispose();
            _dExpertUp?.Dispose();
            _dExpertAct?.Dispose();
            _dExpertDownOut?.Dispose();
            _dFfnOut?.Dispose();
            _dShexpGate?.Dispose();
            _dShexpUp?.Dispose();
            _dShexpAct?.Dispose();
            _dTopKIndices?.Dispose();
            _dTopKWeights?.Dispose();
            _dShexpGateScalar?.Dispose();
            foreach (var res in _dShexpGateWeights.Values)
                res?.Dispose();
            _dShexpGateWeights.Clear();

            _dXBatch?.Dispose();
            _dNormXBatch?.Dispose();
            if (_uploadEmbeddingBatch != null)
            {
                _uploadEmbeddingBatch.Unmap(0);
                _uploadEmbeddingBatch.Dispose();
            }
            _dQBatch?.Dispose();
            _dQGateBatch?.Dispose();
            _dQFullBatch?.Dispose();
            _dKBatch?.Dispose();
            _dVBatch?.Dispose();
            _dAttnOutBatch?.Dispose();
            _dGdnQkvBatch?.Dispose();
            _dGdnConvOutBatch?.Dispose();
            _dGdnZBatch?.Dispose();
            _dGdnABatch?.Dispose();
            _dGdnBBatch?.Dispose();
            _dGdnYBatch?.Dispose();
            _dAttnProjOutBatch?.Dispose();
            _dMlpNormBatch?.Dispose();
            _dMlpGateBatch?.Dispose();
            _dMlpUpBatch?.Dispose();
            _dMlpActBatch?.Dispose();
            _dMlpOutBatch?.Dispose();
            _dRouterLogitsBatch?.Dispose();
            _dTopKIndicesBatch?.Dispose();
            _dTopKWeightsBatch?.Dispose();

            _dOutNormWeight?.Dispose();
            _dOutNorm?.Dispose();
            _dOutWeight?.Dispose();
            _dLogits?.Dispose();

            _uploadEmbedding?.Unmap(0);
            _uploadEmbedding?.Dispose();
            if (_readbackActivation != null)
            {
                _readbackActivation.Unmap(0);
                _readbackActivation.Dispose();
            }
            _readbackLogits?.Unmap(0);
            _readbackLogits?.Dispose();

            _psoRmsNorm?.Dispose();
            _sigRmsNorm?.Dispose();

            _psoRmsNormHeads?.Dispose();
            _sigRmsNormHeads?.Dispose();

            _psoGemvQ4K?.Dispose();
            _psoGemvQ5K?.Dispose();
            _psoGemvQ6K?.Dispose();
            _psoGemvQ3K?.Dispose();
            _psoGemvQ8_0?.Dispose();
            _psoGemvQ2_K?.Dispose();
            _psoGemvQ2_0?.Dispose();
            _psoGemvIQ2_XXS?.Dispose();
            _psoGemvIQ4_NL?.Dispose();
            _psoGemvIQ3_S?.Dispose();
            _psoGemvTernary?.Dispose();
            _psoGemvFp32?.Dispose();
            _sigGemv?.Dispose();

            _psoSwiglu?.Dispose();
            _sigSwiglu?.Dispose();

            _psoVecAdd?.Dispose();
            _sigVecAdd?.Dispose();

            _psoVecAddWeighted?.Dispose();
            _sigVecAddWeighted?.Dispose();

            _psoRope?.Dispose();
            _sigRope?.Dispose();

            _psoKvStore?.Dispose();
            _sigKvStore?.Dispose();

            _psoFlashAttn?.Dispose();
            _sigFlashAttn?.Dispose();

            _psoSsmConv1d?.Dispose();
            _sigSsmConv1d?.Dispose();

            _psoL2NormQK?.Dispose();
            _sigL2NormQK?.Dispose();

            _psoDeltaNet?.Dispose();
            _sigDeltaNet?.Dispose();

            _psoSsmGateSilu?.Dispose();
            _sigSsmGateSilu?.Dispose();

            _psoQGateSplit?.Dispose();
            _sigQGateSplit?.Dispose();

            _psoAttnOutGate?.Dispose();
            _sigAttnOutGate?.Dispose();

            _psoGemmQ4KBatch?.Dispose();
            _psoGemmQ6KBatch?.Dispose();
            _psoGemmQ8_0Batch?.Dispose();
            _psoGemmFp32Batch?.Dispose();
            _psoGemmQ3KBatch?.Dispose();
            _psoGemmQ5KBatch?.Dispose();
            _psoGemmQ2_KBatch?.Dispose();
            _psoGemmQ2_0Batch?.Dispose();
            _psoGemmIQ2_XXSBatch?.Dispose();
            _psoGemmIQ4_NLBatch?.Dispose();
            _psoGemmIQ3_SBatch?.Dispose();
            _psoGemmTernaryBatch?.Dispose();
            _sigGemmBatch?.Dispose();

            _psoRmsNormBatch?.Dispose();
            _sigRmsNormBatch?.Dispose();

            _ctx.Dispose();
        }
    }
}
