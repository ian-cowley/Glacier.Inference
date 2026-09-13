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
/// Bare-metal Direct3D 12 Compute transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on AMD Radeon (Wave32 RDNA 2/3/3.5) and DirectX 12 hardware with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe partial class Qwen2D3D12Model : IDisposable
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
    private ID3D12Resource _dOutNormWeight = null!;
    private ID3D12Resource _dOutWeight = null!;
    private readonly D3D12LayerWeights[] _layerWeights;

    private readonly float[] _hX;
    private ID3D12Resource _uploadEmbedding = null!;
    private float* _pUploadEmbedding;
    private const int MaxBatchChunk = 64;
    private ID3D12Resource _uploadEmbeddingBatch = null!;
    private float* _pUploadEmbeddingBatch;
    private ID3D12Resource _readbackLogits = null!;
    private float* _pReadbackLogits;
    private bool _disposed;

    public D3D12Context Context => _ctx;
    public ModelWeights Weights => _weights;
    public (double RecordMs, double GpuMs) LastTimings { get; private set; }

    public Qwen2D3D12Model(D3D12Context ctx, ModelWeights weights, int maxSeqLen = 4096)
    {
        _ctx = ctx;
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _maxSeqLen = maxSeqLen;

        _hX = new float[_dim];
        _dKeyCache = new ID3D12Resource[weights.BlockCount];
        _dValCache = new ID3D12Resource[weights.BlockCount];
        _layerWeights = new D3D12LayerWeights[weights.BlockCount];

        InitPipelines();
        InitScratchBuffers();
        UploadWeights();
    }

    private void InitScratchBuffers()
    {
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        _dX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dNormX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dQ = _ctx.CreateDeviceBuffer((ulong)(qDim * sizeof(float)));
        _dK = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dV = _ctx.CreateDeviceBuffer((ulong)(kvDim * sizeof(float)));
        _dAttnOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
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
        _dAttnOutBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
        _dGateBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));
        _dUpBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));
        _dFfnActBatch = _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * _ffnDim * sizeof(float)));

        // Persistent Upload and Readback buffers (avoids reallocating D3D12 resources per token)
        _uploadEmbedding = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUpload = null;
        _uploadEmbedding.Map(0, null, &pUpload);
        _pUploadEmbedding = (float*)pUpload;

        _uploadEmbeddingBatch = _ctx.CreateUploadBuffer((ulong)(MaxBatchChunk * _dim * sizeof(float)));
        void* pUploadBatch = null;
        _uploadEmbeddingBatch.Map(0, null, &pUploadBatch);
        _pUploadEmbeddingBatch = (float*)pUploadBatch;

        _readbackLogits = _ctx.CreateReadbackBuffer((ulong)(_weights.VocabSize * sizeof(float)));
        void* pReadback = null;
        _readbackLogits.Map(0, null, &pReadback);
        _pReadbackLogits = (float*)pReadback;

        // KV Cache per layer
        ulong kvBytes = (ulong)((long)_nHeadsKv * _maxSeqLen * _headDim * sizeof(float));
        for (int l = 0; l < _weights.BlockCount; l++)
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
            for (int l = 0; l < _weights.BlockCount; l++)
            {
                if (_weights.Layers[l].FfnGateShexpWeight != null)
                {
                    hasShexp = true;
                    if (_weights.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
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

    private static byte[] AlignQ3K(ReadOnlySpan<byte> rawQ3K, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 112);
        fixed (byte* pSrc = rawQ3K, pDst = aligned)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;
            int numThreads = Math.Max(1, Environment.ProcessorCount);
            int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
            Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, totalBlocks);
                byte* pS = (byte*)srcAddr;
                byte* pD = (byte*)dstAddr;
                for (int b = start; b < end; b++)
                {
                    byte* srcBlk = pS + (long)b * 110;
                    byte* dstBlk = pD + (long)b * 112;
                    Buffer.MemoryCopy(srcBlk, dstBlk, 110, 110);
                    dstBlk[110] = 0;
                    dstBlk[111] = 0;
                }
            });
        }
        return aligned;
    }

    private static ulong GetTensorSliceBytes(GgufType type, int rows, int cols)
    {
        if (type == GgufType.Q3_K)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 112UL;
        }
        if (type == GgufType.Q6_K)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 212UL;
        }
        if (type == GgufType.Q8_0)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 36UL;
        }
        return (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
    }

    private static byte[] AlignQ6K(ReadOnlySpan<byte> rawQ6K, int totalBlocks)
    {
        byte[] aligned = new byte[totalBlocks * 212];
        fixed (byte* pSrc = rawQ6K, pDst = aligned)
        {
            for (int b = 0; b < totalBlocks; b++)
            {
                byte* srcBlk = pSrc + b * 210;
                byte* dstBlk = pDst + b * 212;
                Buffer.MemoryCopy(srcBlk, dstBlk, 210, 210);
                dstBlk[210] = 0;
                dstBlk[211] = 0;
            }
        }
        return aligned;
    }

    private static byte[] AlignQ8_0(ReadOnlySpan<byte> rawQ8_0, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 36);
        fixed (byte* pSrc = rawQ8_0, pDst = aligned)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;
            int numThreads = Math.Max(1, Environment.ProcessorCount);
            int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
            Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, totalBlocks);
                byte* pS = (byte*)srcAddr;
                byte* pD = (byte*)dstAddr;
                for (int b = start; b < end; b++)
                {
                    byte* srcBlk = pS + (long)b * 34;
                    byte* dstBlk = pD + (long)b * 36;
                    *(ushort*)dstBlk = *(ushort*)srcBlk;
                    dstBlk[2] = 0;
                    dstBlk[3] = 0;
                    Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, 32, 32);
                }
            });
        }
        return aligned;
    }

    private ID3D12Resource UploadTensor(GgufType type, IntPtr pData, int rows, int cols)
    {
        if (type != GgufType.Q4_K && type != GgufType.Q5_K && type != GgufType.Q6_K && type != GgufType.Q3_K && type != GgufType.Q8_0 && type != GgufType.F32 && type != GgufType.F16)
        {
            throw new NotSupportedException($"Direct3D 12 compute engine does not yet support tensor quantization type {type}. Supported GPU types: Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, F32, F16.");
        }

        if (type == GgufType.Q3_K)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            ReadOnlySpan<byte> raw = new ReadOnlySpan<byte>((void*)pData, totalBlocks * 110);
            byte[] aligned = AlignQ3K(raw, totalBlocks);
            var buf = _ctx.CreateDeviceBuffer((ulong)aligned.Length);
            _ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        else if (type == GgufType.Q6_K)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            ReadOnlySpan<byte> raw = new ReadOnlySpan<byte>((void*)pData, totalBlocks * 210);
            byte[] aligned = AlignQ6K(raw, totalBlocks);
            var buf = _ctx.CreateDeviceBuffer((ulong)aligned.Length);
            _ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        else if (type == GgufType.Q8_0)
        {
            int nb = cols / 32;
            int totalBlocks = rows * nb;
            ReadOnlySpan<byte> raw = new ReadOnlySpan<byte>((void*)pData, totalBlocks * 34);
            byte[] aligned = AlignQ8_0(raw, totalBlocks);
            var buf = _ctx.CreateDeviceBuffer((ulong)aligned.Length);
            _ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        else
        {
            long rowBytes = GgufTypes.GetRowBytes(type, cols);
            ulong totalBytes = (ulong)rowBytes * (ulong)rows;
            var buf = _ctx.CreateDeviceBuffer(totalBytes);
            _ctx.CopyToDevice(buf, pData, totalBytes);
            return buf;
        }
    }

    private void UploadWeights()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($">> Uploading model weights to {_ctx.DeviceName} via Direct3D 12 Compute...");
        var sw = Stopwatch.StartNew();

        _dOutNormWeight = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _ctx.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (ulong)(_dim * sizeof(float)));

        _dOutWeight = UploadTensor(_weights.OutType, (IntPtr)_weights.OutWeight, _weights.VocabSize, _dim);

        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];
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

            var dAttnOut = UploadTensor(lw.AttnOutType, (IntPtr)lw.AttnOutWeight, _dim, _dim);

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
                    if (_weights.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
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
        Console.WriteLine($"   Weights uploaded to Direct3D 12 GPU VRAM in {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F2} s)!");
        Console.ResetColor();
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
            if (_readbackLogits != null)
            {
                _readbackLogits.Unmap(0);
                _readbackLogits.Dispose();
            }

            for (int l = 0; l < _weights.BlockCount; l++)
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
}
