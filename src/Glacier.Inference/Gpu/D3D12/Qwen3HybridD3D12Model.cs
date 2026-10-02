namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;

/// <summary>
/// World-class Direct3D 12 GPU inference runtime for Qwen 3.5 / 3.6 Hybrid models.
/// Accelerates Gated DeltaNet (linear attention SSM with recurrent associative memory)
/// interleaved with full quadratic multi-head attention (3:1 ratio) on AMD Radeon / Windows GPUs.
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
    private ID3D12PipelineState _psoGemvFp32 = null!;

    private ID3D12RootSignature _sigSwiglu = null!;
    private ID3D12PipelineState _psoSwiglu = null!;

    private ID3D12RootSignature _sigVecAdd = null!;
    private ID3D12PipelineState _psoVecAdd = null!;

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

    // Output Head Buffers
    private ID3D12Resource _dOutNormWeight = null!;
    private ID3D12Resource _dOutNorm = null!;
    private ID3D12Resource _dOutWeight = null!;
    private ID3D12Resource _dLogits = null!;

    private ID3D12Resource _uploadEmbedding = null!;
    private float* _pUploadEmbedding;
    private ID3D12Resource _readbackLogits = null!;
    private float* _pReadbackLogits;

    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int VocabSize => _vocabSize;

    public Qwen3HybridD3D12Model(D3D12Context ctx, ModelWeights weights, int maxSeqLen = 4096)
    {
        _ctx = ctx;
        _weights = weights;
        _maxSeqLen = maxSeqLen;
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

        Console.WriteLine($"[Qwen 3.5 D3D12] Arch: '{weights.Gguf.Architecture}', Heads: {_nHeads}/{_headsKv}, HeadDim: {_headDim}, RopeDim: {_ropeDim}, SsmHeads: {_ssmHeads}, ConvChannels: {_gdnConvChannels}");

        _dConvState = new ID3D12Resource[weights.BlockCount];
        _dSsmState = new ID3D12Resource[weights.BlockCount];
        _dKeyCache = new ID3D12Resource[weights.BlockCount];
        _dValCache = new ID3D12Resource[weights.BlockCount];

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

        // 5. VecAdd
        var vecAddParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAdd = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddParams));
        _psoVecAdd = _ctx.CreatePipelineState(_sigVecAdd, _ctx.CompileShader(D3D12Shaders.VecAdd));

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
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];
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
        _dMlpNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dMlpGate = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dMlpUp = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dMlpAct = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dMlpOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // Head and Output
        _dOutNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dLogits = _ctx.CreateDeviceBuffer((ulong)(_vocabSize * sizeof(float)));

        _uploadEmbedding = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbedding.Map(0, null, &pUpload);
        _pUploadEmbedding = (float*)pUpload;

        _readbackLogits = _ctx.CreateReadbackBuffer((ulong)(_vocabSize * sizeof(float)));
        void* pRbLogits = null;
        _readbackLogits.Map(0, null, &pRbLogits);
        _pReadbackLogits = (float*)pRbLogits;
    }

    private void UploadWeights()
    {
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            if (l % 8 == 0 || l == _weights.BlockCount - 1)
            {
                Console.WriteLine($"[D3D12 GPU] Uploading Qwen 3.5 weights: layer {l + 1}/{_weights.BlockCount}...");
            }

            var lw = _weights.Layers[l];
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

            var dFfnGate = UploadTensor(lw.FfnGateType, (IntPtr)lw.FfnGateWeight, _ffnDim, _dim);
            var dFfnUp = UploadTensor(lw.FfnUpType, (IntPtr)lw.FfnUpWeight, _ffnDim, _dim);
            var dFfnDown = UploadTensor(lw.FfnDownType, (IntPtr)lw.FfnDownWeight, _dim, _ffnDim);

            _layerWeights.Add(new D3D12LayerWeights
            {
                IsGdn = isGdn,
                HasQGate = lw.HasQGate,
                AttnNormWeight = dAttnNorm,
                QWeight = dQ ?? _ctx.DummyBuffer,
                QType = lw.QType,
                KWeight = dK ?? _ctx.DummyBuffer,
                KType = lw.KType,
                VWeight = dV ?? _ctx.DummyBuffer,
                VType = lw.VType,
                AttnOutWeight = dAttnOut ?? _ctx.DummyBuffer,
                AttnOutType = lw.AttnOutType,
                AttnQNormWeight = dAttnQNorm,
                AttnKNormWeight = dAttnKNorm,
                QkvWeight = dQkv,
                QkvType = lw.QkvType,
                AttnGateWeight = dAttnGate,
                AttnGateType = lw.AttnGateType,
                SsmConv1dWeight = dSsmConv1d,
                SsmAlphaWeight = dSsmAlpha,
                SsmAlphaType = lw.SsmAlphaType,
                SsmBetaWeight = dSsmBeta,
                SsmBetaType = lw.SsmBetaType,
                SsmDtBias = dSsmDtBias,
                SsmAWeight = dSsmA,
                SsmNormWeight = dSsmNorm,
                SsmOutWeight = dSsmOut,
                SsmOutType = lw.SsmOutType,
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnGateType = lw.FfnGateType,
                FfnUpWeight = dFfnUp,
                FfnUpType = lw.FfnUpType,
                FfnDownWeight = dFfnDown,
                FfnDownType = lw.FfnDownType,
            });
        }

        // LM Head
        _dOutNormWeight = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _ctx.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (ulong)(_dim * sizeof(float)));
        _dOutWeight = UploadTensor(_weights.OutType, (IntPtr)_weights.OutWeight, _vocabSize, _dim);
    }

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
    {
        cmdList.SetComputeRootSignature(_sigGemv);
        var pso = type switch
        {
            GgufType.Q4_K => _psoGemvQ4K,
            GgufType.Q5_K => _psoGemvQ5K,
            GgufType.Q6_K => _psoGemvQ6K,
            GgufType.Q3_K => _psoGemvQ3K,
            GgufType.Q8_0 => _psoGemvQ8_0,
            _ => _psoGemvFp32
        };
        cmdList.SetPipelineState(pso);

        cmdList.SetComputeRootShaderResourceView(1, wGpuVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y.GPUVirtualAddress);

        const int chunkSize = D3D12TensorAlign.GemvChunkRows;
        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)k_cols;
        pConsts[1] = (uint)m_rows;
        pConsts[2] = 0u; // has_bias
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
        => Forward(token, pos, logits, computeLogits, -1);

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true, int maxLayers = -1)
    {
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);

        uint* ropeConsts = stackalloc uint[7];
        uint* kvConsts = stackalloc uint[4];
        uint* faConsts = stackalloc uint[6];
        uint* convConsts = stackalloc uint[1];
        uint* l2Consts = stackalloc uint[2];
        uint* dnConsts = stackalloc uint[4];
        uint* gsConsts = stackalloc uint[4];
        uint* qgConsts = stackalloc uint[2];

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding into _dX
        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        int qDim = _nHeads * _headDim;
        int kvDim = _headsKv * _headDim;

        int layersToRun = maxLayers >= 0 ? Math.Min(maxLayers, _weights.BlockCount) : _weights.BlockCount;
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
                // Projections
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
            // Feed-Forward Network (SwiGLU)
            // -------------------------------------------------------------
            DispatchRmsNorm(cmd, _dX, lw.FfnNormWeight, _dMlpNorm, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

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

        // Output Head
        if (computeLogits && logits.Length > 0)
        {
            DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dOutNorm, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGemv(cmd, _weights.OutType, _dLogits, _dOutNorm, _dOutWeight.GPUVirtualAddress, _dim, _vocabSize);
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

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool isLast = i == tokens.Length - 1;
            Forward(tokens[i], startPos + i, isLast && computeLogits ? logits : Span<float>.Empty, isLast && computeLogits);
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

            _dOutNormWeight?.Dispose();
            _dOutNorm?.Dispose();
            _dOutWeight?.Dispose();
            _dLogits?.Dispose();

            _uploadEmbedding?.Unmap(0);
            _uploadEmbedding?.Dispose();
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
            _psoGemvFp32?.Dispose();
            _sigGemv?.Dispose();

            _psoSwiglu?.Dispose();
            _sigSwiglu?.Dispose();

            _psoVecAdd?.Dispose();
            _sigVecAdd?.Dispose();

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

            _ctx.Dispose();
        }
    }
}
