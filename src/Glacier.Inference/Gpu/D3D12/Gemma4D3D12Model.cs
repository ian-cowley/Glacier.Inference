namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

/// <summary>
/// High-performance Direct3D 12 GPU inference runtime for Google Gemma 4 models.
/// Implements native HLSL compute shaders for Interleaved Sliding Window Attention (ISWA),
/// dual Shared+MoE FFN, GeLU-GLU activations, and logit softcapping directly on DirectX 12 hardware.
/// </summary>
public sealed unsafe partial class Gemma4D3D12Model : ID3D12Model
{
    private readonly D3D12Context _ctx;
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _sharedFfnDim;
    private readonly int _expertFfnDim;
    private readonly int _expertCount;
    private readonly int _expertUsedCount;
    private readonly int _slidingWindow;
    private readonly float _finalLogitSoftcapping;
    private readonly float _embeddingScale;
    private readonly int _maxSeqLen;

    // Direct3D 12 Pipelines & Root Signatures
    private ID3D12RootSignature _sigGemv = null!;
    private ID3D12PipelineState _psoGemvQ4K = null!;
    private ID3D12PipelineState _psoGemvQ5K = null!;
    private ID3D12PipelineState _psoGemvQ6K = null!;
    private ID3D12PipelineState _psoGemvQ3K = null!;
    private ID3D12PipelineState _psoGemvQ8_0 = null!;
    private ID3D12PipelineState _psoGemvFp32 = null!;

    private ID3D12RootSignature _sigRmsNorm = null!;
    private ID3D12PipelineState _psoRmsNorm = null!;

    private ID3D12RootSignature _sigRmsNormHeads = null!;
    private ID3D12PipelineState _psoRmsNormHeads = null!;

    private ID3D12RootSignature _sigGeluGLU = null!;
    private ID3D12PipelineState _psoGeluGLU = null!;

    private ID3D12RootSignature _sigVecAdd = null!;
    private ID3D12PipelineState _psoVecAdd = null!;

    private ID3D12RootSignature _sigVecAddWeighted = null!;
    private ID3D12PipelineState _psoVecAddWeighted = null!;

    private ID3D12RootSignature _sigRope = null!;
    private ID3D12PipelineState _psoRope = null!;

    private ID3D12RootSignature _sigKvStore = null!;
    private ID3D12PipelineState _psoKvStore = null!;

    private ID3D12RootSignature _sigSwaAttention = null!;
    private ID3D12PipelineState _psoSwaAttention = null!;

    private ID3D12RootSignature _sigScaleAndMul = null!;
    private ID3D12PipelineState _psoScaleAndMul = null!;

    private ID3D12RootSignature _sigSoftcap = null!;
    private ID3D12PipelineState _psoSoftcap = null!;

    private ID3D12RootSignature _sigLayerOutScale = null!;
    private ID3D12PipelineState _psoLayerOutScale = null!;

    // GPU Scratch Buffers
    private ID3D12Resource _dX = null!;
    private ID3D12Resource _dAttnIn = null!;
    private ID3D12Resource _dQ = null!;
    private ID3D12Resource _dK = null!;
    private ID3D12Resource _dV = null!;
    private ID3D12Resource _dAttnOut = null!;
    private ID3D12Resource _dAttnProjOut = null!;
    private ID3D12Resource _dAttnOutResidual = null!;

    // Shared MLP Scratch Buffers
    private ID3D12Resource _dMlpNorm = null!;
    private ID3D12Resource _dSharedGate = null!;
    private ID3D12Resource _dSharedUp = null!;
    private ID3D12Resource _dMlpAct = null!;
    private ID3D12Resource _dMlpOut = null!;

    // MoE Scratch Buffers
    private ID3D12Resource _dMoeNorm = null!;
    private ID3D12Resource _dRouterIn = null!;
    private ID3D12Resource _dRouterLogits = null!;
    private ID3D12Resource _readbackRouterLogits = null!;
    private float* _pReadbackRouterLogits;
    private ID3D12Resource _dExpertGateUp = null!;
    private ID3D12Resource _dExpertAct = null!;
    private ID3D12Resource _dExpertDownOut = null!;
    private ID3D12Resource _dMoeOut = null!;

    // Combined FFN Buffer
    private ID3D12Resource _dCombinedFfn = null!;

    // Output Scratch Buffers
    private ID3D12Resource _dOutNorm = null!;
    private ID3D12Resource _dLogits = null!;
    private ID3D12Resource _readbackLogits = null!;
    private float* _pReadbackLogits;

    // Persistent Upload buffer for token embedding
    private ID3D12Resource _uploadEmbedding = null!;
    private float* _pUploadEmbedding;

    // Per-layer GPU KV Cache (SWA: 8 heads x 256 dim; Dense: 2 heads x 512 dim)
    private readonly ID3D12Resource[] _dKeyCache;
    private readonly ID3D12Resource[] _dValCache;
    private readonly D3D12LayerWeights[] _layerWeights;

    private ID3D12Resource? _dOutNormWeight;
    private ID3D12Resource? _dOutWeight;
    private bool _disposed;

    public D3D12Context Context => _ctx;
    public ModelWeights Weights => _weights;

    public Gemma4D3D12Model(D3D12Context ctx, ModelWeights weights, int maxSeqLen = 4096)
    {
        _ctx = ctx;
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _sharedFfnDim = weights.FeedForwardLength;
        _expertFfnDim = weights.ExpertFeedForwardLength > 0 ? weights.ExpertFeedForwardLength : 704;
        _expertCount = weights.ExpertCount > 0 ? weights.ExpertCount : 128;
        _expertUsedCount = weights.ExpertUsedCount > 0 ? weights.ExpertUsedCount : 8;
        _slidingWindow = weights.SlidingWindow > 0 ? weights.SlidingWindow : 1024;
        _finalLogitSoftcapping = weights.FinalLogitSoftcapping;
        _embeddingScale = MathF.Sqrt(_dim);
        _maxSeqLen = maxSeqLen;

        _dKeyCache = new ID3D12Resource[weights.BlockCount];
        _dValCache = new ID3D12Resource[weights.BlockCount];
        _layerWeights = new D3D12LayerWeights[weights.BlockCount];

        InitPipelines();
        InitScratchBuffers();
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

        // 2b. RMSNorm per-head
        var rmsHeadsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigRmsNormHeads = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsHeadsParams));
        _psoRmsNormHeads = _ctx.CreatePipelineState(_sigRmsNormHeads, _ctx.CompileShader(D3D12Shaders.RmsNormHeads));

        // 3. GeLU-GLU
        var geluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigGeluGLU = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, geluParams));
        _psoGeluGLU = _ctx.CreatePipelineState(_sigGeluGLU, _ctx.CompileShader(D3D12Shaders.Gemma4GeluGLU));

        // 4. VecAdd & VecAddWeighted
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

        // 5. RoPE
        var ropeParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRope = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeParams));
        _psoRope = _ctx.CreatePipelineState(_sigRope, _ctx.CompileShader(D3D12Shaders.RoPE));

        // 6. KV Store
        var kvStoreParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigKvStore = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, kvStoreParams));
        _psoKvStore = _ctx.CreatePipelineState(_sigKvStore, _ctx.CompileShader(D3D12Shaders.Gemma4KvCacheStore));

        // 7. SWA Attention
        var swaParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigSwaAttention = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, swaParams));
        _psoSwaAttention = _ctx.CreatePipelineState(_sigSwaAttention, _ctx.CompileShader(D3D12Shaders.Gemma4SwaAttention));

        // 8. ScaleAndMul
        var scaleAndMulParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigScaleAndMul = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, scaleAndMulParams));
        _psoScaleAndMul = _ctx.CreatePipelineState(_sigScaleAndMul, _ctx.CompileShader(D3D12Shaders.Gemma4ScaleAndMul));

        // 9. Softcap
        var softcapParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigSoftcap = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, softcapParams));
        _psoSoftcap = _ctx.CreatePipelineState(_sigSoftcap, _ctx.CompileShader(D3D12Shaders.Gemma4SoftcapLogits));

        // 10. Layer Output Scale
        var layerOutScaleParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigLayerOutScale = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, layerOutScaleParams));
        _psoLayerOutScale = _ctx.CreatePipelineState(_sigLayerOutScale, _ctx.CompileShader(D3D12Shaders.Gemma4LayerOutputScale));
    }

    private void InitScratchBuffers()
    {
        int maxQDim = 16 * 512; // 8192
        int maxKvDim = 8 * 256; // 2048

        _dX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dAttnIn = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dQ = _ctx.CreateDeviceBuffer((ulong)(maxQDim * sizeof(float)));
        _dK = _ctx.CreateDeviceBuffer((ulong)(maxKvDim * sizeof(float)));
        _dV = _ctx.CreateDeviceBuffer((ulong)(maxKvDim * sizeof(float)));
        _dAttnOut = _ctx.CreateDeviceBuffer((ulong)(maxQDim * sizeof(float)));
        _dAttnProjOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dAttnOutResidual = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // Shared MLP buffers
        _dMlpNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dSharedGate = _ctx.CreateDeviceBuffer((ulong)(_sharedFfnDim * sizeof(float)));
        _dSharedUp = _ctx.CreateDeviceBuffer((ulong)(_sharedFfnDim * sizeof(float)));
        _dMlpAct = _ctx.CreateDeviceBuffer((ulong)(_sharedFfnDim * sizeof(float)));
        _dMlpOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // MoE buffers
        if (_expertCount > 0)
        {
            _dMoeNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _dRouterIn = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _dRouterLogits = _ctx.CreateDeviceBuffer((ulong)(_expertCount * sizeof(float)));
            _readbackRouterLogits = _ctx.CreateReadbackBuffer((ulong)(_expertCount * sizeof(float)));
            void* pRbRLogits = null;
            _readbackRouterLogits.Map(0, null, &pRbRLogits);
            _pReadbackRouterLogits = (float*)pRbRLogits;

            _dExpertGateUp = _ctx.CreateDeviceBuffer((ulong)(2 * _expertFfnDim * sizeof(float)));
            _dExpertAct = _ctx.CreateDeviceBuffer((ulong)(_expertFfnDim * sizeof(float)));
            _dExpertDownOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _dMoeOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        }

        // Combined FFN buffer
        _dCombinedFfn = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        // Output buffers
        _dOutNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dLogits = _ctx.CreateDeviceBuffer((ulong)(_weights.VocabSize * sizeof(float)));
        _readbackLogits = _ctx.CreateReadbackBuffer((ulong)(_weights.VocabSize * sizeof(float)));
        void* pRbLogits = null;
        _readbackLogits.Map(0, null, &pRbLogits);
        _pReadbackLogits = (float*)pRbLogits;

        _uploadEmbedding = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbedding.Map(0, null, &pUpload);
        _pUploadEmbedding = (float*)pUpload;

        // KV Cache per layer (sized for alternating SWA vs Dense layers)
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var layer = _weights.Layers[l];
            int headsKv = layer.HeadsKv;
            int headDim = layer.HeadDim;
            ulong kvBytes = (ulong)((long)headsKv * _maxSeqLen * headDim * sizeof(float));

            _dKeyCache[l] = _ctx.CreateDeviceBuffer(kvBytes);
            _dValCache[l] = _ctx.CreateDeviceBuffer(kvBytes);
        }
    }

    private void UploadWeights()
    {
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            if (l % 5 == 0 || l == _weights.BlockCount - 1)
            {
                Console.WriteLine($"[D3D12 GPU] Uploading weights: layer {l + 1}/{_weights.BlockCount}...");
            }
            var lw = _weights.Layers[l];
            bool isSwa = lw.IsSwa;
            int headDim = lw.HeadDim;
            int headsKv = lw.HeadsKv;
            int nHeads = _weights.HeadCount;
            int totalQDim = nHeads * headDim;
            int totalKvDim = headsKv * headDim;

            var dAttnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (ulong)(_dim * sizeof(float)));

            var dQ = UploadTensor(lw.QType, (IntPtr)lw.QWeight, totalQDim, _dim);
            var dK = UploadTensor(lw.KType, (IntPtr)lw.KWeight, totalKvDim, _dim);
            ID3D12Resource? dV = lw.VWeight != null ? UploadTensor(lw.VType, (IntPtr)lw.VWeight, totalKvDim, _dim) : null;
            var dAttnOut = UploadTensor(lw.AttnOutType, (IntPtr)lw.AttnOutWeight, _dim, totalQDim);

            ID3D12Resource? dAttnQNorm = null;
            if (lw.AttnQNormWeight != null)
            {
                dAttnQNorm = _ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
                _ctx.CopyToDevice(dAttnQNorm, (IntPtr)lw.AttnQNormWeight, (ulong)(headDim * sizeof(float)));
            }

            ID3D12Resource? dAttnKNorm = null;
            if (lw.AttnKNormWeight != null)
            {
                dAttnKNorm = _ctx.CreateDeviceBuffer((ulong)(headDim * sizeof(float)));
                _ctx.CopyToDevice(dAttnKNorm, (IntPtr)lw.AttnKNormWeight, (ulong)(headDim * sizeof(float)));
            }

            ID3D12Resource? dAttnPostNorm = null;
            if (lw.AttnPostNormWeight != null)
            {
                dAttnPostNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dAttnPostNorm, (IntPtr)lw.AttnPostNormWeight, (ulong)(_dim * sizeof(float)));
            }

            // Shared MLP weights
            var dFfnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (ulong)(_dim * sizeof(float)));

            var dFfnGate = UploadTensor(lw.FfnGateType, (IntPtr)lw.FfnGateWeight, _sharedFfnDim, _dim);
            var dFfnUp = UploadTensor(lw.FfnUpType, (IntPtr)lw.FfnUpWeight, _sharedFfnDim, _dim);
            var dFfnDown = UploadTensor(lw.FfnDownType, (IntPtr)lw.FfnDownWeight, _dim, _sharedFfnDim);

            ID3D12Resource? dFfnPostNorm1 = null;
            if (lw.FfnPostNorm1Weight != null)
            {
                dFfnPostNorm1 = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dFfnPostNorm1, (IntPtr)lw.FfnPostNorm1Weight, (ulong)(_dim * sizeof(float)));
            }

            // MoE weights
            ID3D12Resource? dFfnPreNorm2 = null;
            if (lw.FfnPreNorm2Weight != null)
            {
                dFfnPreNorm2 = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dFfnPreNorm2, (IntPtr)lw.FfnPreNorm2Weight, (ulong)(_dim * sizeof(float)));
            }

            ID3D12Resource? dFfnGateInpScale = null;
            if (lw.FfnGateInpScaleWeight != null)
            {
                dFfnGateInpScale = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dFfnGateInpScale, (IntPtr)lw.FfnGateInpScaleWeight, (ulong)(_dim * sizeof(float)));
            }

            bool isMoe = _expertCount > 0 && lw.FfnGateInpWeight != null;
            ID3D12Resource? dFfnGateInp = null;
            ID3D12Resource? dFfnGateUpExps = null;
            ID3D12Resource? dFfnDownExps = null;
            ID3D12Resource? dFfnDownExpsScale = null;

            if (isMoe)
            {
                dFfnGateInp = UploadTensor(GgufType.F32, (IntPtr)lw.FfnGateInpWeight, _expertCount, _dim);
                dFfnGateUpExps = UploadTensor(lw.FfnGateUpExpsType, (IntPtr)lw.FfnGateUpExpsWeight, _expertCount * 2 * _expertFfnDim, _dim);
                dFfnDownExps = UploadTensor(lw.FfnDownExpsType, (IntPtr)lw.FfnDownExpsWeight, _expertCount * _dim, _expertFfnDim);

                if (lw.FfnDownExpsScaleWeight != null)
                {
                    dFfnDownExpsScale = _ctx.CreateDeviceBuffer((ulong)(_expertCount * sizeof(float)));
                    _ctx.CopyToDevice(dFfnDownExpsScale, (IntPtr)lw.FfnDownExpsScaleWeight, (ulong)(_expertCount * sizeof(float)));
                }
            }

            ID3D12Resource? dFfnPostNorm2 = null;
            if (lw.FfnPostNorm2Weight != null)
            {
                dFfnPostNorm2 = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dFfnPostNorm2, (IntPtr)lw.FfnPostNorm2Weight, (ulong)(_dim * sizeof(float)));
            }

            // Post-FFN combined norm & layer scale
            ID3D12Resource? dFfnPostNorm = null;
            if (lw.FfnPostNormWeight != null)
            {
                dFfnPostNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
                _ctx.CopyToDevice(dFfnPostNorm, (IntPtr)lw.FfnPostNormWeight, (ulong)(_dim * sizeof(float)));
            }

            ID3D12Resource? dLayerOutScale = null;
            if (lw.LayerOutputScaleWeight != null)
            {
                dLayerOutScale = _ctx.CreateDeviceBuffer(sizeof(float));
                _ctx.CopyToDevice(dLayerOutScale, (IntPtr)lw.LayerOutputScaleWeight, sizeof(float));
            }

            _layerWeights[l] = new D3D12LayerWeights
            {
                AttnNormWeight = dAttnNorm,
                QWeight = dQ,
                KWeight = dK,
                VWeight = dV ?? dK,
                AttnOutWeight = dAttnOut,
                AttnQNormWeight = dAttnQNorm,
                AttnKNormWeight = dAttnKNorm,
                AttnPostNormWeight = dAttnPostNorm,
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnUpWeight = dFfnUp,
                FfnDownWeight = dFfnDown,
                FfnPostNorm1Weight = dFfnPostNorm1,
                FfnPreNorm2Weight = dFfnPreNorm2,
                FfnGateInpScaleWeight = dFfnGateInpScale,
                FfnGateInpWeight = dFfnGateInp ?? _ctx.DummyBuffer,
                FfnGateUpExpsWeight = dFfnGateUpExps ?? _ctx.DummyBuffer,
                FfnGateUpExpsType = lw.FfnGateUpExpsType,
                FfnDownExpsWeight = dFfnDownExps ?? _ctx.DummyBuffer,
                FfnDownExpsType = lw.FfnDownExpsType,
                FfnDownExpsScaleWeight = dFfnDownExpsScale,
                FfnPostNorm2Weight = dFfnPostNorm2,
                FfnPostNormWeight = dFfnPostNorm,
                LayerOutputScaleWeight = dLayerOutScale,
                QType = lw.QType,
                KType = lw.KType,
                VType = lw.VType,
                AttnOutType = lw.AttnOutType,
                FfnGateType = lw.FfnGateType,
                FfnUpType = lw.FfnUpType,
                FfnDownType = lw.FfnDownType,
                IsMoe = isMoe
            };
        }

        // Final output norm and LM head
        _dOutNormWeight = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _ctx.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (ulong)(_dim * sizeof(float)));

        _dOutWeight = UploadTensor(_weights.OutType, (IntPtr)_weights.OutWeight, _weights.VocabSize, _dim);
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
            GgufType.F32 => _psoGemvFp32,
            _ => throw new NotSupportedException($"Direct3D 12 GEMV does not support tensor quantization type {type}.")
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

    private void DispatchRmsNorm(ID3D12GraphicsCommandList cmdList, ID3D12Resource x, ID3D12Resource? weight, ID3D12Resource dst, int size, float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNorm);
        cmdList.SetPipelineState(_psoRmsNorm);

        uint* pConsts = stackalloc uint[2];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;
        cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(1, 1, 1);
    }

    private void DispatchGeluGLU(ID3D12GraphicsCommandList cmdList, ID3D12Resource gate, ID3D12Resource up, ID3D12Resource dst, int size)
    {
        cmdList.SetComputeRootSignature(_sigGeluGLU);
        cmdList.SetPipelineState(_psoGeluGLU);

        uint pConst = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&pConst), 0);

        cmdList.SetComputeRootUnorderedAccessView(1, gate.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, up.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchVecAdd(ID3D12GraphicsCommandList cmdList, ID3D12Resource b, ID3D12Resource a, int size)
    {
        cmdList.SetComputeRootSignature(_sigVecAdd);
        cmdList.SetPipelineState(_psoVecAdd);

        uint pConst = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&pConst), 0);

        cmdList.SetComputeRootUnorderedAccessView(1, b.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, a.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchVecAddWeighted(ID3D12GraphicsCommandList cmdList, ID3D12Resource b, ID3D12Resource a, float weight, int size, uint accumulate)
    {
        cmdList.SetComputeRootSignature(_sigVecAddWeighted);
        cmdList.SetPipelineState(_psoVecAddWeighted);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = weight;
        pConsts[2] = accumulate;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, b.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, a.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        // 1. Extract token embedding and scale by sqrt(dim)
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);
        for (int i = 0; i < _dim; i++)
        {
            _pUploadEmbedding[i] *= _embeddingScale;
        }

        int maxTopK = _expertUsedCount > 0 ? _expertUsedCount : 8;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        uint* ropeConsts = stackalloc uint[6];
        uint* kvConsts = stackalloc uint[4];
        uint* swaConsts = stackalloc uint[7];
        uint* smConsts = stackalloc uint[3];
        uint* lConsts = stackalloc uint[2];

        // 2. Record Command List
        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding into _dX
        cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
        cmd.ResourceBarrierUnorderedAccessView(null!);

        // 3. Execute Transformer Layers
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _layerWeights[l];
            var lMeta = _weights.Layers[l];
            bool isSwa = lMeta.IsSwa;
            int headDim = lMeta.HeadDim;
            int headsKv = lMeta.HeadsKv;
            int nHeads = _weights.HeadCount;
            int totalQDim = nHeads * headDim;
            int totalKvDim = headsKv * headDim;

            // Attention pre-norm: RMSNorm(_dX) -> _dAttnIn
            DispatchRmsNorm(cmd, _dX, lw.AttnNormWeight, _dAttnIn, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Q & K Projections
            DispatchGemv(cmd, lw.QType, _dQ, _dAttnIn, lw.QWeight.GPUVirtualAddress, _dim, totalQDim);
            DispatchGemv(cmd, lw.KType, _dK, _dAttnIn, lw.KWeight.GPUVirtualAddress, _dim, totalKvDim);

            // V Projection (or copy from K if absent)
            if (lMeta.VWeight != null)
            {
                DispatchGemv(cmd, lw.VType, _dV, _dAttnIn, lw.VWeight.GPUVirtualAddress, _dim, totalKvDim);
            }
            else
            {
                cmd.CopyBufferRegion(_dV, 0, _dK, 0, (ulong)(totalKvDim * sizeof(float)));
            }
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // RoPE on Q and K
            float freqBase = isSwa ? _weights.RopeFreqBaseSwa : _weights.RopeFreqBase;
            cmd.SetComputeRootSignature(_sigRope);
            cmd.SetPipelineState(_psoRope);
            ropeConsts[0] = (uint)nHeads;
            ropeConsts[1] = (uint)headsKv;
            ropeConsts[2] = (uint)headDim;
            ropeConsts[3] = (uint)pos;
            *(float*)(&ropeConsts[4]) = freqBase;
            *(float*)(&ropeConsts[5]) = 1.0f;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)ropeConsts, 0);
            cmd.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _dK.GPUVirtualAddress);
            cmd.Dispatch(((uint)((nHeads + headsKv) * (headDim / 2)) + 255) / 256, 1, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Store K & V to per-layer KV cache
            cmd.SetComputeRootSignature(_sigKvStore);
            cmd.SetPipelineState(_psoKvStore);
            kvConsts[0] = (uint)headsKv;
            kvConsts[1] = (uint)headDim;
            kvConsts[2] = (uint)_maxSeqLen;
            kvConsts[3] = (uint)pos;
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)kvConsts, 0);
            cmd.SetComputeRootUnorderedAccessView(1, _dK.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _dV.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _dKeyCache[l].GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _dValCache[l].GPUVirtualAddress);
            cmd.Dispatch(((uint)totalKvDim + 255) / 256, 1, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // SWA Attention
            cmd.SetComputeRootSignature(_sigSwaAttention);
            cmd.SetPipelineState(_psoSwaAttention);
            swaConsts[0] = (uint)nHeads;
            swaConsts[1] = (uint)headsKv;
            swaConsts[2] = (uint)headDim;
            swaConsts[3] = (uint)_maxSeqLen;
            swaConsts[4] = (uint)pos;
            *(float*)(&swaConsts[5]) = 1.0f; // attn_scale = 1.0 for Gemma 4
            swaConsts[6] = isSwa ? (uint)_slidingWindow : 0u;
            cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)swaConsts, 0);
            cmd.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _dKeyCache[l].GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _dValCache[l].GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _dAttnOut.GPUVirtualAddress);
            cmd.Dispatch((uint)nHeads, 1, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Attention Output projection: _dAttnOut * AttnOutWeight -> _dAttnProjOut
            DispatchGemv(cmd, lw.AttnOutType, _dAttnProjOut, _dAttnOut, lw.AttnOutWeight.GPUVirtualAddress, totalQDim, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Post-attention norm: RMSNorm(_dAttnProjOut, AttnPostNorm) -> _dAttnProjOut
            DispatchRmsNorm(cmd, _dAttnProjOut, lw.AttnPostNormWeight, _dAttnProjOut, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Residual connection: _dAttnOutResidual = _dX + _dAttnProjOut
            cmd.CopyBufferRegion(_dAttnOutResidual, 0, _dX, 0, (ulong)(_dim * sizeof(float)));
            cmd.ResourceBarrierUnorderedAccessView(null!);
            DispatchVecAdd(cmd, _dAttnProjOut, _dAttnOutResidual, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // -----------------------------------------------------------------
            // Dual FFN: Part 1 - Shared MLP
            // -----------------------------------------------------------------
            DispatchRmsNorm(cmd, _dAttnOutResidual, lw.FfnNormWeight, _dMlpNorm, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGemv(cmd, lw.FfnGateType, _dSharedGate, _dMlpNorm, lw.FfnGateWeight!.GPUVirtualAddress, _dim, _sharedFfnDim);
            DispatchGemv(cmd, lw.FfnUpType, _dSharedUp, _dMlpNorm, lw.FfnUpWeight!.GPUVirtualAddress, _dim, _sharedFfnDim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGeluGLU(cmd, _dSharedGate, _dSharedUp, _dMlpAct, _sharedFfnDim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGemv(cmd, lw.FfnDownType, _dMlpOut, _dMlpAct, lw.FfnDownWeight!.GPUVirtualAddress, _sharedFfnDim, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchRmsNorm(cmd, _dMlpOut, lw.FfnPostNorm1Weight, _dMlpOut, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // -----------------------------------------------------------------
            // Dual FFN: Part 2 - Sparse MoE (128 Experts, Top-8 routing)
            // -----------------------------------------------------------------
            if (lw.IsMoe)
            {
                DispatchRmsNorm(cmd, _dAttnOutResidual, lw.FfnPreNorm2Weight, _dMoeNorm, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Custom Router In: RMSNorm(attn_out, null) / sqrt(dim) * FfnGateInpScale
                DispatchRmsNorm(cmd, _dAttnOutResidual, null, _dRouterIn, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                cmd.SetComputeRootSignature(_sigScaleAndMul);
                cmd.SetPipelineState(_psoScaleAndMul);
                smConsts[0] = (uint)_dim;
                *(float*)(&smConsts[1]) = 1.0f / _embeddingScale;
                smConsts[2] = lw.FfnGateInpScaleWeight != null ? 1u : 0u;
                cmd.SetComputeRoot32BitConstants(0, 3, (IntPtr)smConsts, 0);
                cmd.SetComputeRootShaderResourceView(1, lw.FfnGateInpScaleWeight?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(2, _dRouterIn.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(3, _dRouterIn.GPUVirtualAddress);
                cmd.Dispatch(((uint)_dim + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Router GEMV: _dRouterIn * FfnGateInpWeight -> _dRouterLogits
                DispatchGemv(cmd, GgufType.F32, _dRouterLogits, _dRouterIn, lw.FfnGateInpWeight!.GPUVirtualAddress, _dim, _expertCount);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Readback router logits to CPU host to select Top-K
                cmd.CopyBufferRegion(_readbackRouterLogits, 0, _dRouterLogits, 0, (ulong)(_expertCount * sizeof(float)));
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();

                // CPU Top-K Softmax selection
                QuantKernels.SoftmaxTopK(_pReadbackRouterLogits, _expertCount, _expertUsedCount, selectedIndices, selectedWeights, normTopK: true);

                // Re-open command list for active expert dispatches
                _ctx.BeginCommands();
                cmd = _ctx.CommandList;

                ulong gateUpSliceBytes = (ulong)(2 * _expertFfnDim) * (ulong)GgufTypes.GetRowBytes(lw.FfnGateUpExpsType, _dim);
                ulong downSliceBytes = (ulong)_dim * (ulong)GgufTypes.GetRowBytes(lw.FfnDownExpsType, _expertFfnDim);

                for (int k = 0; k < _expertUsedCount; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    ulong expGateUpAddr = lw.FfnGateUpExpsWeight!.GPUVirtualAddress + (ulong)expertIdx * gateUpSliceBytes;
                    ulong expDownAddr = lw.FfnDownExpsWeight!.GPUVirtualAddress + (ulong)expertIdx * downSliceBytes;

                    // Fused Gate-Up GEMV
                    DispatchGemv(cmd, lw.FfnGateUpExpsType, _dExpertGateUp, _dMoeNorm, expGateUpAddr, _dim, 2 * _expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // GeLU-GLU
                    DispatchGeluGLU(cmd, _dExpertGateUp, _dExpertGateUp, _dExpertAct, _expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // Down GEMV
                    DispatchGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut, _dExpertAct, expDownAddr, _expertFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // Expert scaling
                    float expScale = lMeta.FfnDownExpsScaleWeight != null ? lMeta.FfnDownExpsScaleWeight[expertIdx] : 1.0f;
                    float finalScale = expScale * weight;

                    DispatchVecAddWeighted(cmd, _dExpertDownOut, _dMoeOut, finalScale, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // Post-norm MoE output
                DispatchRmsNorm(cmd, _dMoeOut, lw.FfnPostNorm2Weight, _dMoeOut, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // -----------------------------------------------------------------
                // Dual FFN: Part 3 - Combine Shared MLP + Sparse MoE
                // -----------------------------------------------------------------
                cmd.CopyBufferRegion(_dCombinedFfn, 0, _dMlpOut, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierUnorderedAccessView(null!);
                DispatchVecAdd(cmd, _dMoeOut, _dCombinedFfn, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
            else
            {
                cmd.CopyBufferRegion(_dCombinedFfn, 0, _dMlpOut, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }

            // Post-FFN norm
            DispatchRmsNorm(cmd, _dCombinedFfn, lw.FfnPostNormWeight, _dCombinedFfn, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Residual add: _dX = _dAttnOutResidual + _dCombinedFfn
            cmd.CopyBufferRegion(_dX, 0, _dAttnOutResidual, 0, (ulong)(_dim * sizeof(float)));
            cmd.ResourceBarrierUnorderedAccessView(null!);
            DispatchVecAdd(cmd, _dCombinedFfn, _dX, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Layer output scaling
            if (lMeta.LayerOutputScaleWeight != null)
            {
                float lScale = lMeta.LayerOutputScaleWeight[0];
                cmd.SetComputeRootSignature(_sigLayerOutScale);
                cmd.SetPipelineState(_psoLayerOutScale);
                lConsts[0] = (uint)_dim;
                *(float*)(&lConsts[1]) = lScale;
                cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)lConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dX.GPUVirtualAddress);
                cmd.Dispatch(((uint)_dim + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
        }

        // 4. Final Output Normalization & Logits
        if (computeLogits && logits.Length > 0)
        {
            DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dOutNorm, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGemv(cmd, _weights.OutType, _dLogits, _dOutNorm, _dOutWeight!.GPUVirtualAddress, _dim, _weights.VocabSize);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Final Logit Softcapping
            if (_finalLogitSoftcapping > 0f)
            {
                cmd.SetComputeRootSignature(_sigSoftcap);
                cmd.SetPipelineState(_psoSoftcap);
                uint* scConsts = stackalloc uint[2];
                scConsts[0] = (uint)_weights.VocabSize;
                *(float*)(&scConsts[1]) = _finalLogitSoftcapping;
                cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)scConsts, 0);
                cmd.SetComputeRootUnorderedAccessView(1, _dLogits.GPUVirtualAddress);
                cmd.Dispatch(((uint)_weights.VocabSize + 255) / 256, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }

            // Readback logits to host
            cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            fixed (float* pLogits = logits)
            {
                Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
            }
        }
        else
        {
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
        }
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool isLast = (i == tokens.Length - 1);
            Forward(tokens[i], startPos + i, isLast && computeLogits ? logits : Span<float>.Empty, isLast && computeLogits);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _uploadEmbedding?.Dispose();
            _dX?.Dispose();
            _dAttnIn?.Dispose();
            _dQ?.Dispose();
            _dK?.Dispose();
            _dV?.Dispose();
            _dAttnOut?.Dispose();
            _dAttnProjOut?.Dispose();
            _dAttnOutResidual?.Dispose();

            _dMlpNorm?.Dispose();
            _dSharedGate?.Dispose();
            _dSharedUp?.Dispose();
            _dMlpAct?.Dispose();
            _dMlpOut?.Dispose();

            _dMoeNorm?.Dispose();
            _dRouterIn?.Dispose();
            _dRouterLogits?.Dispose();
            _readbackRouterLogits?.Dispose();
            _dExpertGateUp?.Dispose();
            _dExpertAct?.Dispose();
            _dExpertDownOut?.Dispose();
            _dMoeOut?.Dispose();
            _dCombinedFfn?.Dispose();

            _dOutNorm?.Dispose();
            _dLogits?.Dispose();
            _readbackLogits?.Dispose();

            _dOutNormWeight?.Dispose();
            _dOutWeight?.Dispose();

            for (int l = 0; l < _dKeyCache.Length; l++)
            {
                _dKeyCache[l]?.Dispose();
                _dValCache[l]?.Dispose();
                _layerWeights[l]?.Dispose();
            }

            _sigGemv?.Dispose();
            _psoGemvQ4K?.Dispose();
            _psoGemvQ5K?.Dispose();
            _psoGemvQ6K?.Dispose();
            _psoGemvQ3K?.Dispose();
            _psoGemvQ8_0?.Dispose();
            _psoGemvFp32?.Dispose();

            _sigRmsNorm?.Dispose();
            _psoRmsNorm?.Dispose();
            _sigRmsNormHeads?.Dispose();
            _psoRmsNormHeads?.Dispose();
            _sigGeluGLU?.Dispose();
            _psoGeluGLU?.Dispose();
            _sigVecAdd?.Dispose();
            _psoVecAdd?.Dispose();
            _sigVecAddWeighted?.Dispose();
            _psoVecAddWeighted?.Dispose();
            _sigRope?.Dispose();
            _psoRope?.Dispose();
            _sigKvStore?.Dispose();
            _psoKvStore?.Dispose();
            _sigSwaAttention?.Dispose();
            _psoSwaAttention?.Dispose();
            _sigScaleAndMul?.Dispose();
            _psoScaleAndMul?.Dispose();
            _sigSoftcap?.Dispose();
            _psoSoftcap?.Dispose();
            _sigLayerOutScale?.Dispose();
            _psoLayerOutScale?.Dispose();

            _ctx?.Dispose();
            _disposed = true;
        }
    }
}
