namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

/// <summary>
/// Bare-metal Direct3D 12 Compute transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on AMD Radeon (Wave32 RDNA 2/3/3.5) and DirectX 12 hardware with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe partial class Qwen2D3D12Model : ID3D12Model
{
    private readonly D3D12Context _ctx;
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _headDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _groupSize;
    private readonly float _attnScale;
    private readonly int _maxSeqLen;

    // Pipelines & Root Signatures
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

    private ID3D12RootSignature _sigAttention = null!;
    private ID3D12PipelineState _psoAttention = null!;

    private ID3D12RootSignature _sigArgmax = null!;
    private ID3D12PipelineState _psoArgmax = null!;

    // Batched Pipelines
    private ID3D12RootSignature _sigGemmBatch = null!;
    private ID3D12PipelineState _psoGemmQ4KBatch = null!;
    private ID3D12PipelineState _psoGemmQ6KBatch = null!;
    private ID3D12PipelineState _psoGemmQ8_0Batch = null!;
    private ID3D12PipelineState _psoGemmFp32Batch = null!;

    private ID3D12RootSignature _sigRmsNormBatch = null!;
    private ID3D12PipelineState _psoRmsNormBatch = null!;

    private ID3D12RootSignature _sigRopeBatch = null!;
    private ID3D12PipelineState _psoRopeBatch = null!;

    private ID3D12RootSignature _sigKvStoreBatch = null!;
    private ID3D12PipelineState _psoKvStoreBatch = null!;

    private ID3D12RootSignature _sigAttentionBatch = null!;
    private ID3D12PipelineState _psoAttentionBatch = null!;

    // GPU Scratch Buffers (Single Token)
    private ID3D12Resource _dX = null!;
    private ID3D12Resource _dNormX = null!;
    private ID3D12Resource _dQ = null!;
    private ID3D12Resource _dK = null!;
    private ID3D12Resource _dV = null!;
    private ID3D12Resource _dAttnOut = null!;
    private ID3D12Resource _dGate = null!;
    private ID3D12Resource _dUp = null!;
    private ID3D12Resource _dFfnAct = null!;
    private ID3D12Resource _dLogits = null!;
    private ID3D12Resource _dBestToken = null!;
    private ID3D12Resource _dBestLogit = null!;

    // MoE Scratch Buffers
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

    // Batched GPU Scratch Buffers
    private ID3D12Resource _dXBatch = null!;
    private ID3D12Resource _dNormXBatch = null!;
    private ID3D12Resource _dQBatch = null!;
    private ID3D12Resource _dKBatch = null!;
    private ID3D12Resource _dVBatch = null!;
    private ID3D12Resource _dAttnOutBatch = null!;
    private ID3D12Resource _dGateBatch = null!;
    private ID3D12Resource _dUpBatch = null!;
    private ID3D12Resource _dFfnActBatch = null!;

    // GPU KV Cache per layer [n_heads_kv, max_seq_len, head_dim]
    private readonly ID3D12Resource[] _dKeyCache;
    private readonly ID3D12Resource[] _dValCache;

    // Model Weights in GPU VRAM
    private ID3D12Resource? _dOutNormWeight;
    private ID3D12Resource? _dOutWeight;
    private readonly D3D12LayerWeights[] _layerWeights;

    private readonly float[] _hX;
    private ID3D12Resource _uploadEmbedding = null!;
    private float* _pUploadEmbedding;
    private const int MaxBatchChunk = 64;
    private ID3D12Resource _uploadEmbeddingBatch = null!;
    private float* _pUploadEmbeddingBatch;

    // Device-resident In-VRAM Token Embedding Cache (Default Heap, zero-PCIe CopyBufferRegion)
    public const int EmbdCacheCapacity = 4096;
    private ID3D12Resource? _dEmbdCache;
    private int[]? _embdCacheTokens;
    private int _embdCacheHead;
    private Dictionary<int, int>? _embdTokenToSlot;
    public int EmbdCacheHitCount { get; internal set; }
    public int EmbdCacheMissCount { get; internal set; }

    private ID3D12Resource? _readbackActivation;
    private float* _pReadbackActivation;
    private ID3D12Resource? _readbackActivationBatch;
    private float* _pReadbackActivationBatch;
    private ID3D12Resource _readbackLogits = null!;
    private float* _pReadbackLogits;
    private float[]? _hostLogits;
    private readonly Sampling.Sampler _sampler = new();
    private bool _disposed;

    public D3D12Context Context => _ctx;
    public ModelWeights Weights => _weights;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }
    public (double RecordMs, double GpuMs) LastTimings { get; private set; }
    public Glacier.Inference.Memory.KvCachePrecision KvPrecision { get; }

    public Qwen2D3D12Model(
        D3D12Context ctx,
        ModelWeights weights,
        int maxSeqLen = 4096,
        Glacier.Inference.Memory.KvCachePrecision kvPrecision = Glacier.Inference.Memory.KvCachePrecision.Auto,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true)
    {
        _ctx = ctx;
        _weights = weights;
        StartLayer = startLayer;
        LayerCount = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        IsLastStage = isLastStage;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _maxSeqLen = maxSeqLen;
        KvPrecision = kvPrecision == Glacier.Inference.Memory.KvCachePrecision.Auto
            ? Glacier.Inference.Memory.KvCachePrecision.Fp16
            : kvPrecision;

        _hX = new float[_dim];
        _dKeyCache = new ID3D12Resource[LayerCount];
        _dValCache = new ID3D12Resource[LayerCount];
        _layerWeights = new D3D12LayerWeights[LayerCount];

        InitPipelines();
        InitScratchBuffers();
        UploadWeights();
    }

    private void InitScratchBuffers()
    {
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int maxAttnOut = Math.Max(_dim, qDim);

        _dX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dNormX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dQ = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
        _dK = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dV = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dAttnOut = _ctx.CreateDeviceBuffer((ulong)(maxAttnOut * sizeof(float)));
        _dGate = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dUp = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dFfnAct = _ctx.CreateDeviceBuffer((ulong)(_ffnDim * sizeof(float)));
        _dLogits = _ctx.CreateDeviceBuffer((ulong)(_weights.VocabSize * sizeof(float)));
        _dBestToken = _ctx.CreateDeviceBuffer(sizeof(int));
        _dBestLogit = _ctx.CreateDeviceBuffer(sizeof(float));

        // Batched scratch buffers for prefill
        _dXBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
        _dNormXBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
        _dQBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * qDim * sizeof(float)));
        _dKBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * kvDim * sizeof(float)));
        _dVBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * kvDim * sizeof(float)));
        _dAttnOutBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * maxAttnOut * sizeof(float)));
        _dGateBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));
        _dUpBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));
        _dFfnActBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));

        // Persistent Upload and Readback buffers (avoids reallocating D3D12 resources per token)
        _uploadEmbedding = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbedding.Map(0, null, &pUpload);
        _pUploadEmbedding = (float*)pUpload;

        if (StartLayer == 0)
        {
            ulong cacheBytes = (ulong)(EmbdCacheCapacity * _dim * sizeof(float));
            _dEmbdCache = _ctx.CreateDeviceBuffer(cacheBytes);
            _embdCacheTokens = new int[EmbdCacheCapacity];
            Array.Fill(_embdCacheTokens, -1);
            _embdTokenToSlot = new Dictionary<int, int>(EmbdCacheCapacity);
        }

        _uploadEmbeddingBatch = _ctx.CreateUploadBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
        void* pUploadBatch = null;
        _uploadEmbeddingBatch.Map(0, null, &pUploadBatch);
        _pUploadEmbeddingBatch = (float*)pUploadBatch;

        _readbackLogits = _ctx.CreateReadbackBuffer((ulong)(_weights.VocabSize * sizeof(float)));
        void* pReadback = null;
        _readbackLogits.Map(0, null, &pReadback);
        _pReadbackLogits = (float*)pReadback;

        if (!IsLastStage)
        {
            _readbackActivation = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
            void* pRbAct = null;
            _readbackActivation.Map(0, null, &pRbAct);
            _pReadbackActivation = (float*)pRbAct;

            _readbackActivationBatch = _ctx.CreateReadbackBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
            void* pRbBatch = null;
            _readbackActivationBatch.Map(0, null, &pRbBatch);
            _pReadbackActivationBatch = (float*)pRbBatch;
        }

        // KV Cache per layer
        ulong elemSize = (ulong)(KvPrecision == Glacier.Inference.Memory.KvCachePrecision.Fp16 ? sizeof(short) : sizeof(float));
        ulong kvBytes = (ulong)((long)_nHeadsKv * _maxSeqLen * _headDim) * elemSize;
        for (int l = 0; l < LayerCount; l++)
        {
            _dKeyCache[l] = _ctx.CreateDeviceBuffer(kvBytes);
            _dValCache[l] = _ctx.CreateDeviceBuffer(kvBytes);
        }

        if (_weights.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFeedForwardLength;
            int numExperts = _weights.ExpertCount;

            _dRouterLogits = _ctx.CreateDeviceBuffer((ulong)(numExperts * sizeof(float)));
            _readbackRouterLogits = _ctx.CreateReadbackBuffer((ulong)(numExperts * sizeof(float)));
            void* pReadbackRouter = null;
            _readbackRouterLogits.Map(0, null, &pReadbackRouter);
            _pReadbackRouterLogits = (float*)pReadbackRouter;

            _dExpertGate = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertUp = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertAct = _ctx.CreateDeviceBuffer((ulong)(expertFfnDim * sizeof(float)));
            _dExpertDownOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _dFfnOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

            int maxShexpFfnDim = expertFfnDim * 2;
            bool hasShexp = false;
            for (int l = 0; l < LayerCount; l++)
            {
                int modelLayer = StartLayer + l;
                if (_weights.Layers[modelLayer].FfnGateShexpWeight != null)
                {
                    hasShexp = true;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                    {
                        if ((int)tShexp.Dimensions[1] > maxShexpFfnDim)
                            maxShexpFfnDim = (int)tShexp.Dimensions[1];
                    }
                }
            }
            if (hasShexp)
            {
                _dShexpGate = _ctx.CreateDeviceBuffer((ulong)(maxShexpFfnDim * sizeof(float)));
                _dShexpUp = _ctx.CreateDeviceBuffer((ulong)(maxShexpFfnDim * sizeof(float)));
                _dShexpAct = _ctx.CreateDeviceBuffer((ulong)(maxShexpFfnDim * sizeof(float)));
            }
        }
    }

    private static ulong GetTensorSliceBytes(GgufType type, int rows, int cols) =>
        D3D12TensorAlign.GetAlignedTensorBytes(type, rows, cols);

    private ID3D12Resource UploadTensor(GgufType type, IntPtr hostPtr, int rows, int cols)
    {
        return D3D12TensorAlign.UploadAlignedTensor(_ctx, type, hostPtr, rows, cols);
    }

    private void UploadWeights()
    {
        GlacierDiagnostics.LogInformation($">> Uploading model weights to {_ctx.DeviceName} via Direct3D 12 Compute (Layers {StartLayer}..{StartLayer + LayerCount - 1})...");
        var sw = Stopwatch.StartNew();

        if (IsLastStage)
        {
            _dOutNormWeight = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (ulong)(_dim * sizeof(float)));

            _dOutWeight = UploadTensor(_weights.OutType, (IntPtr)_weights.OutWeight, _weights.VocabSize, _dim);
        }

        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var lw = _weights.Layers[modelLayer];
            int qDim = _nHeads * _headDim;
            int kvDim = _nHeadsKv * _headDim;

            var dAttnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (ulong)(_dim * sizeof(float)));

            var dQ = UploadTensor(lw.QType, (IntPtr)lw.QWeight, qDim, _dim);
            ID3D12Resource? dQBias = null;
            if (lw.QBias != null)
            {
                dQBias = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
                _ctx.CopyToDevice(dQBias, (IntPtr)lw.QBias, (ulong)(qDim * sizeof(float)));
            }

            var dK = UploadTensor(lw.KType, (IntPtr)lw.KWeight, kvDim, _dim);
            ID3D12Resource? dKBias = null;
            if (lw.KBias != null)
            {
                dKBias = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
                _ctx.CopyToDevice(dKBias, (IntPtr)lw.KBias, (ulong)(kvDim * sizeof(float)));
            }

            var dV = UploadTensor(lw.VType, (IntPtr)lw.VWeight, kvDim, _dim);
            ID3D12Resource? dVBias = null;
            if (lw.VBias != null)
            {
                dVBias = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
                _ctx.CopyToDevice(dVBias, (IntPtr)lw.VBias, (ulong)(kvDim * sizeof(float)));
            }

            // Optional QK-Norm
            ID3D12Resource? dAttnQNorm = null;
            if (lw.AttnQNormWeight != null)
            {
                dAttnQNorm = _ctx.CreateDeviceBuffer((ulong)(_headDim * sizeof(float)));
                _ctx.CopyToDevice(dAttnQNorm, (IntPtr)lw.AttnQNormWeight, (ulong)(_headDim * sizeof(float)));
            }

            ID3D12Resource? dAttnKNorm = null;
            if (lw.AttnKNormWeight != null)
            {
                dAttnKNorm = _ctx.CreateDeviceBuffer((ulong)(_headDim * sizeof(float)));
                _ctx.CopyToDevice(dAttnKNorm, (IntPtr)lw.AttnKNormWeight, (ulong)(_headDim * sizeof(float)));
            }

            var dAttnOut = UploadTensor(lw.AttnOutType, (IntPtr)lw.AttnOutWeight, _dim, qDim);

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

                dFfnGateExps = UploadTensor(lw.FfnGateExpsType, (IntPtr)lw.FfnGateExpsWeight, numExperts * expertFfnDim, _dim);
                dFfnUpExps = UploadTensor(lw.FfnUpExpsType, (IntPtr)lw.FfnUpExpsWeight, numExperts * expertFfnDim, _dim);
                dFfnDownExps = UploadTensor(lw.FfnDownExpsType, (IntPtr)lw.FfnDownExpsWeight, numExperts * _dim, expertFfnDim);

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

            _layerWeights[l] = new D3D12LayerWeights
            {
                AttnNormWeight = dAttnNorm,
                QWeight = dQ,
                QBias = dQBias,
                KWeight = dK,
                KBias = dKBias,
                VWeight = dV,
                VBias = dVBias,
                AttnQNormWeight = dAttnQNorm,
                AttnKNormWeight = dAttnKNorm,
                AttnOutWeight = dAttnOut,
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnUpWeight = dFfnUp,
                FfnDownWeight = dFfnDown,
                IsMoe = lw.IsMoe,
                FfnGateInpWeight = dFfnGateInp,
                FfnGateInpBias = dFfnGateInpBias,
                FfnGateExpsWeight = dFfnGateExps,
                FfnUpExpsWeight = dFfnUpExps,
                FfnDownExpsWeight = dFfnDownExps,
                FfnGateExpsType = lw.FfnGateExpsType,
                FfnUpExpsType = lw.FfnUpExpsType,
                FfnDownExpsType = lw.FfnDownExpsType,
                FfnGateShexpWeight = dFfnGateShexp,
                FfnUpShexpWeight = dFfnUpShexp,
                FfnDownShexpWeight = dFfnDownShexp,
                FfnGateShexpType = lw.FfnGateShexpType,
                FfnUpShexpType = lw.FfnUpShexpType,
                FfnDownShexpType = lw.FfnDownShexpType,
                QType = lw.QType,
                KType = lw.KType,
                VType = lw.VType,
                AttnOutType = lw.AttnOutType,
                FfnGateType = lw.FfnGateType,
                FfnUpType = lw.FfnUpType,
                FfnDownType = lw.FfnDownType
            };
        }

        sw.Stop();
        GlacierDiagnostics.LogInformation($"   Weights uploaded to Direct3D 12 GPU VRAM in {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F2} s)!");
    }
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _ctx.Synchronize();

            _dX?.Dispose();
            _dNormX?.Dispose();
            _dQ?.Dispose();
            _dK?.Dispose();
            _dV?.Dispose();
            _dAttnOut?.Dispose();
            _dGate?.Dispose();
            _dUp?.Dispose();
            _dFfnAct?.Dispose();
            _dLogits?.Dispose();
            _dBestToken?.Dispose();
            _dBestLogit?.Dispose();

            _dXBatch?.Dispose();
            _dNormXBatch?.Dispose();
            _dQBatch?.Dispose();
            _dKBatch?.Dispose();
            _dVBatch?.Dispose();
            _dAttnOutBatch?.Dispose();
            _dGateBatch?.Dispose();
            _dUpBatch?.Dispose();
            _dFfnActBatch?.Dispose();

            if (_dEmbdCache != null)
            {
                _dEmbdCache.Dispose();
                _dEmbdCache = null;
            }
            _embdTokenToSlot?.Clear();

            if (_uploadEmbedding != null)
            {
                _uploadEmbedding.Unmap(0);
                _uploadEmbedding.Dispose();
            }
            if (_uploadEmbeddingBatch != null)
            {
                _uploadEmbeddingBatch.Unmap(0);
                _uploadEmbeddingBatch.Dispose();
            }
            if (_readbackActivation != null)
            {
                _readbackActivation.Unmap(0);
                _readbackActivation.Dispose();
            }
            if (_readbackActivationBatch != null)
            {
                _readbackActivationBatch.Unmap(0);
                _readbackActivationBatch.Dispose();
            }
            if (_readbackLogits != null)
            {
                _readbackLogits.Unmap(0);
                _readbackLogits.Dispose();
            }

            for (int l = 0; l < LayerCount; l++)
            {
                _dKeyCache[l]?.Dispose();
                _dValCache[l]?.Dispose();
                _layerWeights[l]?.Dispose();
            }

            _dOutNormWeight?.Dispose();
            _dOutWeight?.Dispose();

            if (_readbackRouterLogits != null)
            {
                _readbackRouterLogits.Unmap(0);
                _readbackRouterLogits.Dispose();
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

            _psoGemvQ4K?.Dispose();
            _psoGemvQ5K?.Dispose();
            _psoGemvQ6K?.Dispose();
            _psoGemvQ3K?.Dispose();
            _psoGemvQ8_0?.Dispose();
            _psoGemvFp32?.Dispose();
            _sigGemv?.Dispose();

            _psoVecAddWeighted?.Dispose();
            _sigVecAddWeighted?.Dispose();

            _psoRmsNormHeads?.Dispose();
            _sigRmsNormHeads?.Dispose();

            _psoGemmQ4KBatch?.Dispose();
            _psoGemmQ6KBatch?.Dispose();
            _psoGemmQ8_0Batch?.Dispose();
            _psoGemmFp32Batch?.Dispose();
            _sigGemmBatch?.Dispose();

            _psoRmsNorm?.Dispose();
            _sigRmsNorm?.Dispose();

            _psoRmsNormBatch?.Dispose();
            _sigRmsNormBatch?.Dispose();

            _psoSwiglu?.Dispose();
            _sigSwiglu?.Dispose();

            _psoVecAdd?.Dispose();
            _sigVecAdd?.Dispose();

            _psoRope?.Dispose();
            _sigRope?.Dispose();

            _psoRopeBatch?.Dispose();
            _sigRopeBatch?.Dispose();

            _psoKvStore?.Dispose();
            _sigKvStore?.Dispose();

            _psoKvStoreBatch?.Dispose();
            _sigKvStoreBatch?.Dispose();

            _psoAttention?.Dispose();
            _sigAttention?.Dispose();

            _psoAttentionBatch?.Dispose();
            _sigAttentionBatch?.Dispose();

            _psoArgmax?.Dispose();
            _sigArgmax?.Dispose();

            _ctx?.Dispose();
        }
    }

    /// <summary>
    /// Samples next token with support for greedy / temperature / top-P sampling from readback logits.
    /// </summary>
    public int SampleToken(Sampling.SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        if (_pReadbackLogits == null) return -1;
        _hostLogits ??= new float[_weights.VocabSize];
        fixed (float* pLogits = _hostLogits)
        {
            Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
        }
        return _sampler.Sample(_hostLogits.AsSpan(), options, recentTokens);
    }

    /// <summary>
    /// Resets layer-local KV caches on this stage.
    /// </summary>
    public void ResetKvCache()
    {
        // Direct3D 12 KV cache writes directly overwrite entries by token position.
    }
}
