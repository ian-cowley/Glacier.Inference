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
/// Direct3D 12 GPU transformer layer weights residing in high-speed GPU device memory.
/// </summary>
public sealed class D3D12LayerWeights : IDisposable
{
    public required ID3D12Resource AttnNormWeight { get; init; }
    public required ID3D12Resource QWeight { get; init; }
    public ID3D12Resource? QBias { get; init; }
    public required ID3D12Resource KWeight { get; init; }
    public ID3D12Resource? KBias { get; init; }
    public required ID3D12Resource VWeight { get; init; }
    public ID3D12Resource? VBias { get; init; }
    public required ID3D12Resource AttnOutWeight { get; init; }

    public required ID3D12Resource FfnNormWeight { get; init; }
    public required ID3D12Resource FfnGateWeight { get; init; }
    public required ID3D12Resource FfnUpWeight { get; init; }
    public required ID3D12Resource FfnDownWeight { get; init; }

    public required GgufType QType { get; init; }
    public required GgufType KType { get; init; }
    public required GgufType VType { get; init; }
    public required GgufType AttnOutType { get; init; }
    public required GgufType FfnGateType { get; init; }
    public required GgufType FfnUpType { get; init; }
    public required GgufType FfnDownType { get; init; }

    public void Dispose()
    {
        AttnNormWeight?.Dispose();
        QWeight?.Dispose();
        QBias?.Dispose();
        KWeight?.Dispose();
        KBias?.Dispose();
        VWeight?.Dispose();
        VBias?.Dispose();
        AttnOutWeight?.Dispose();
        FfnNormWeight?.Dispose();
        FfnGateWeight?.Dispose();
        FfnUpWeight?.Dispose();
        FfnDownWeight?.Dispose();
    }
}

/// <summary>
/// Bare-metal Direct3D 12 Compute transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on AMD Radeon (Wave32 RDNA 2/3/3.5) and DirectX 12 hardware with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe class Qwen2D3D12Model : IDisposable
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
    private ID3D12PipelineState _psoGemvQ6K = null!;
    private ID3D12PipelineState _psoGemvFp32 = null!;

    private ID3D12RootSignature _sigRmsNorm = null!;
    private ID3D12PipelineState _psoRmsNorm = null!;

    private ID3D12RootSignature _sigSwiglu = null!;
    private ID3D12PipelineState _psoSwiglu = null!;

    private ID3D12RootSignature _sigVecAdd = null!;
    private ID3D12PipelineState _psoVecAdd = null!;

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

    private void InitPipelines()
    {
        // 1. GEMV Root Signature: (Params b0, W t0, bias t1, x u0, residual u1, y u2)
        var gemvParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
        _psoGemvQ4K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ4K));
        _psoGemvQ6K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ6K));
        _psoGemvFp32 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvFp32));

        // 2. RMSNorm Root Signature: (Params b0, weight t0, x u0, dst u1)
        var rmsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsParams));
        _psoRmsNorm = _ctx.CreatePipelineState(_sigRmsNorm, _ctx.CompileShader(D3D12Shaders.RmsNorm));

        // 3. SwiGLU Root Signature: (Params b0, gate u0, up u1, dst u2)
        var swigluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigSwiglu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, swigluParams));
        _psoSwiglu = _ctx.CreatePipelineState(_sigSwiglu, _ctx.CompileShader(D3D12Shaders.SwiGLU));

        // 4. VecAdd Root Signature: (Params b0, b u0, a u1)
        var vecAddParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAdd = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddParams));
        _psoVecAdd = _ctx.CreatePipelineState(_sigVecAdd, _ctx.CompileShader(D3D12Shaders.VecAdd));

        // 5. RoPE Root Signature: (Params b0, q u0, k u1)
        var ropeParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRope = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeParams));
        _psoRope = _ctx.CreatePipelineState(_sigRope, _ctx.CompileShader(D3D12Shaders.RoPE));

        // 6. KvStore Root Signature: (Params b0, k u0, v u1, k_cache u2, v_cache u3)
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

        // 7. Attention GQA Root Signature: (Params b0, q u0, k_cache u1, v_cache u2, attn_out u3)
        var attnParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigAttention = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        _psoAttention = _ctx.CreatePipelineState(_sigAttention, _ctx.CompileShader(D3D12Shaders.AttentionGqa));

        // 8. Argmax Root Signature: (Params b0, logits u0, best_token u1, best_logit u2)
        var argmaxParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigArgmax = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, argmaxParams));
        _psoArgmax = _ctx.CreatePipelineState(_sigArgmax, _ctx.CompileShader(D3D12Shaders.Argmax));

        // 9. Batched GEMM Root Signature: (Params b0, W t0, bias t1, x t2, residual u0, y u1)
        var gemmBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All), // t2: x (SRV for L1 cache)
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: residual
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)  // u1: y
        };

        _sigGemmBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemmBatchParams));
        _psoGemmQ4KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
        _psoGemmQ6KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));
        _psoGemmFp32Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmFp32Batch));

        // 10. Batched RMSNorm Root Signature: (Params b0, weight t0, x u0, dst u1)
        var rmsNormBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNormBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsNormBatchParams));
        _psoRmsNormBatch = _ctx.CreatePipelineState(_sigRmsNormBatch, _ctx.CompileShader(D3D12Shaders.RmsNormBatch));

        // 11. Batched RoPE Root Signature: (Params b0, q u0, k u1)
        var ropeBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRopeBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeBatchParams));
        _psoRopeBatch = _ctx.CreatePipelineState(_sigRopeBatch, _ctx.CompileShader(D3D12Shaders.RoPEBatch));

        // 12. Batched KvStore Root Signature: (Params b0, k u0, v u1, k_cache u2, v_cache u3)
        var kvStoreBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigKvStoreBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, kvStoreBatchParams));
        _psoKvStoreBatch = _ctx.CreatePipelineState(_sigKvStoreBatch, _ctx.CompileShader(D3D12Shaders.KvCacheStoreBatch));

        // 13. Batched Attention Root Signature: (Params b0, q u0, k_cache u1, v_cache u2, attn_out u3)
        var attnBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigAttentionBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnBatchParams));
        _psoAttentionBatch = _ctx.CreatePipelineState(_sigAttentionBatch, _ctx.CompileShader(D3D12Shaders.AttentionBatch));
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

    private ID3D12Resource UploadTensor(GgufType type, IntPtr pData, int rows, int cols)
    {
        if (type == GgufType.Q6_K)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            ReadOnlySpan<byte> raw = new ReadOnlySpan<byte>((void*)pData, totalBlocks * 210);
            byte[] aligned = AlignQ6K(raw, totalBlocks);
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

            var dAttnOut = UploadTensor(lw.AttnOutType, (IntPtr)lw.AttnOutWeight, _dim, _dim);

            var dFfnNorm = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
            _ctx.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (ulong)(_dim * sizeof(float)));

            var dFfnGate = UploadTensor(lw.FfnGateType, (IntPtr)lw.FfnGateWeight, _ffnDim, _dim);
            var dFfnUp = UploadTensor(lw.FfnUpType, (IntPtr)lw.FfnUpWeight, _ffnDim, _dim);
            var dFfnDown = UploadTensor(lw.FfnDownType, (IntPtr)lw.FfnDownWeight, _dim, _ffnDim);

            _layerWeights[l] = new D3D12LayerWeights
            {
                AttnNormWeight = dAttnNorm,
                QWeight = dQ,
                QBias = dQBias,
                KWeight = dK,
                KBias = dKBias,
                VWeight = dV,
                VBias = dVBias,
                AttnOutWeight = dAttnOut,
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnUpWeight = dFfnUp,
                FfnDownWeight = dFfnDown,
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

    private void DispatchRmsNorm(ID3D12GraphicsCommandList cmdList, ID3D12Resource x, ID3D12Resource weight, ID3D12Resource dst, int size, float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNorm);
        cmdList.SetPipelineState(_psoRmsNorm);

        uint* pConsts = stackalloc uint[2];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;
        cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(1, 1, 1);
    }

    private void DispatchGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ID3D12Resource W,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
    {
        cmdList.SetComputeRootSignature(_sigGemv);
        var pso = type switch
        {
            GgufType.Q4_K => _psoGemvQ4K,
            GgufType.Q6_K => _psoGemvQ6K,
            _ => _psoGemvFp32
        };
        cmdList.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)k_cols;
        pConsts[1] = (uint)m_rows;
        pConsts[2] = bias != null ? 1u : 0u;
        pConsts[3] = residual != null ? 1u : 0u;
        pConsts[4] = y != null ? 1u : 0u;
        cmdList.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, W.GPUVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias?.GPUVirtualAddress ?? 0);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual?.GPUVirtualAddress ?? 0);
        cmdList.SetComputeRootUnorderedAccessView(5, y?.GPUVirtualAddress ?? 0);

        cmdList.Dispatch(((uint)m_rows + 3) / 4, 1, 1);
    }

    private void DispatchRope(ID3D12GraphicsCommandList cmdList, ID3D12Resource q, ID3D12Resource k, int pos)
    {
        cmdList.SetComputeRootSignature(_sigRope);
        cmdList.SetPipelineState(_psoRope);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)pos;
        *(float*)(&pConsts[4]) = _weights.RopeFreqBase;
        *(float*)(&pConsts[5]) = 1.0f;
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, q.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, k.GPUVirtualAddress);

        uint totalHalf = (uint)((_nHeads + _nHeadsKv) * (_headDim / 2));
        cmdList.Dispatch((totalHalf + 255) / 256, 1, 1);
    }

    private void DispatchKvStore(ID3D12GraphicsCommandList cmdList, int layerIdx, int pos)
    {
        cmdList.SetComputeRootSignature(_sigKvStore);
        cmdList.SetPipelineState(_psoKvStore);

        uint* pConsts = stackalloc uint[4];
        pConsts[0] = (uint)_nHeadsKv;
        pConsts[1] = (uint)_headDim;
        pConsts[2] = (uint)_maxSeqLen;
        pConsts[3] = (uint)pos;
        cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dK.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dV.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dValCache[layerIdx].GPUVirtualAddress);

        uint total = (uint)(_nHeadsKv * _headDim);
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchAttention(ID3D12GraphicsCommandList cmdList, int layerIdx, int pos)
    {
        cmdList.SetComputeRootSignature(_sigAttention);
        cmdList.SetPipelineState(_psoAttention);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)_maxSeqLen;
        pConsts[4] = (uint)pos;
        *(float*)(&pConsts[5]) = _attnScale;
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dValCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dAttnOut.GPUVirtualAddress);

        cmdList.Dispatch((uint)_nHeads, 1, 1);
    }

    private void DispatchSwiglu(ID3D12GraphicsCommandList cmdList, ID3D12Resource gate, ID3D12Resource up, ID3D12Resource dst, int size)
    {
        cmdList.SetComputeRootSignature(_sigSwiglu);
        cmdList.SetPipelineState(_psoSwiglu);

        uint* pConsts = stackalloc uint[1];
        pConsts[0] = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, gate.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, up.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchGemmBatch(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ID3D12Resource w,
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
        else
            cmdList.SetPipelineState(_psoGemmFp32Batch);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = (uint)(bias != null ? 1 : 0);
        pConsts[4] = (uint)(residual != null ? 1 : 0);
        pConsts[5] = (uint)(y != null ? 1 : 0);
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias != null ? bias.GPUVirtualAddress : 0);
        cmdList.SetComputeRootShaderResourceView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual != null ? residual.GPUVirtualAddress : 0);
        cmdList.SetComputeRootUnorderedAccessView(5, y != null ? y.GPUVirtualAddress : 0);


        cmdList.Dispatch((uint)(mRows + 3) / 4, (uint)(batchSize + 31) / 32, 1);
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

    private void DispatchRopeBatch(ID3D12GraphicsCommandList cmdList, ID3D12Resource q, ID3D12Resource k, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigRopeBatch);
        cmdList.SetPipelineState(_psoRopeBatch);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)startPos;
        *(float*)(&pConsts[4]) = _weights.RopeFreqBase;
        *(float*)(&pConsts[5]) = 1.0f;
        pConsts[6] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, q.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, k.GPUVirtualAddress);

        uint halfDim = (uint)(_headDim / 2);
        uint totalHalfPerToken = (uint)((_nHeads + _nHeadsKv) * halfDim);
        uint total = (uint)batchSize * totalHalfPerToken;
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchKvStoreBatch(ID3D12GraphicsCommandList cmdList, int layerIdx, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigKvStoreBatch);
        cmdList.SetPipelineState(_psoKvStoreBatch);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)_nHeadsKv;
        pConsts[1] = (uint)_headDim;
        pConsts[2] = (uint)_maxSeqLen;
        pConsts[3] = (uint)startPos;
        pConsts[4] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dKBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dVBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dValCache[layerIdx].GPUVirtualAddress);

        uint total = (uint)(batchSize * _nHeadsKv * _headDim);
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchAttentionBatch(ID3D12GraphicsCommandList cmdList, int layerIdx, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigAttentionBatch);
        cmdList.SetPipelineState(_psoAttentionBatch);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)_maxSeqLen;
        pConsts[4] = (uint)startPos;
        *(float*)(&pConsts[5]) = _attnScale;
        pConsts[6] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dQBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dValCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dAttnOutBatch.GPUVirtualAddress);

        cmdList.Dispatch((uint)_nHeads, (uint)batchSize, 1);
    }

    /// <summary>
    /// Executes full transformer forward pass directly on Direct3D 12 GPU.
    /// Zero CPU synchronization during layer execution.
    /// </summary>
    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        // 1. Extract embedding directly into persistently mapped upload buffer
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Record full transformer graph into command list
        var swRec = Stopwatch.StartNew();
        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding from upload buffer into _dX inside the same command list
        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);

        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _layerWeights[l];

            // Attention pre-norm
            DispatchRmsNorm(cmd, _dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Q, K, V projections
            DispatchGemv(cmd, lw.QType, _dQ, _dNormX, lw.QWeight, _dim, qDim, lw.QBias);
            DispatchGemv(cmd, lw.KType, _dK, _dNormX, lw.KWeight, _dim, kvDim, lw.KBias);
            DispatchGemv(cmd, lw.VType, _dV, _dNormX, lw.VWeight, _dim, kvDim, lw.VBias);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // RoPE
            DispatchRope(cmd, _dQ, _dK, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // KV Cache Store
            DispatchKvStore(cmd, l, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // GQA Attention
            DispatchAttention(cmd, l, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Attn Out projection with fused residual addition (_dX += attnOut)
            DispatchGemv(cmd, lw.AttnOutType, null, _dAttnOut, lw.AttnOutWeight, _dim, _dim, null, residual: _dX);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // FFN pre-norm
            DispatchRmsNorm(cmd, _dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // FFN Gate and Up projections
            DispatchGemv(cmd, lw.FfnGateType, _dGate, _dNormX, lw.FfnGateWeight, _dim, _ffnDim);
            DispatchGemv(cmd, lw.FfnUpType, _dUp, _dNormX, lw.FfnUpWeight, _dim, _ffnDim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // SwiGLU activation
            DispatchSwiglu(cmd, _dGate, _dUp, _dFfnAct, _ffnDim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // FFN Down projection with fused residual addition (_dX += ffnDown)
            DispatchGemv(cmd, lw.FfnDownType, null, _dFfnAct, lw.FfnDownWeight, _ffnDim, _dim, null, residual: _dX);
            cmd.ResourceBarrierUnorderedAccessView(null!);
        }

        if (computeLogits)
        {
            // Final RMSNorm
            DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Output projection (LM Head)
            DispatchGemv(cmd, _weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (!logits.IsEmpty)
            {
                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
            }
        }
        swRec.Stop();

        var swGpu = Stopwatch.StartNew();
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        swGpu.Stop();

        LastTimings = (swRec.Elapsed.TotalMilliseconds, swGpu.Elapsed.TotalMilliseconds);

        // 3. Read back logits if requested
        if (computeLogits && !logits.IsEmpty)
        {
            fixed (float* pLogits = logits)
            {
                Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
            }
        }
    }

    /// <summary>
    /// Batched prompt prefill execution on Direct3D 12 GPU.
    /// Pipelined with register-tiled Batched GEMM (weights streamed ONCE per chunk).
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        if (tokens.IsEmpty) return;

        if (tokens.Length == 1)
        {
            Forward(tokens[0], startPos, logits, computeLogits);
            return;
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        for (int offset = 0; offset < tokens.Length; offset += MaxBatchChunk)
        {
            int chunkSize = Math.Min(MaxBatchChunk, tokens.Length - offset);
            bool isLastChunk = (offset + chunkSize == tokens.Length);
            int chunkStartPos = startPos + offset;

            // 1. Vectorized host embedding extraction for the chunk
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

            var swRec = Stopwatch.StartNew();
            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;

            // Copy entire chunk of embeddings into _dXBatch
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dXBatch, 0, _uploadEmbeddingBatch, 0, (ulong)(chunkSize * _dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopyDest, ResourceStates.Common);

            // Execute all 28 layers across all tokens in the chunk simultaneously!
            // Weights for each layer are read from VRAM ONCE per chunk!
            for (int l = 0; l < _weights.BlockCount; l++)
            {
                var lw = _layerWeights[l];

                // Attention pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.AttnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Q, K, V projections (weights read ONCE into registers)
                DispatchGemmBatch(cmd, lw.QType, _dQBatch, _dNormXBatch, lw.QWeight, _dim, qDim, chunkSize, lw.QBias);
                DispatchGemmBatch(cmd, lw.KType, _dKBatch, _dNormXBatch, lw.KWeight, _dim, kvDim, chunkSize, lw.KBias);
                DispatchGemmBatch(cmd, lw.VType, _dVBatch, _dNormXBatch, lw.VWeight, _dim, kvDim, chunkSize, lw.VBias);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched RoPE
                DispatchRopeBatch(cmd, _dQBatch, _dKBatch, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched KV Cache Store
                DispatchKvStoreBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Attention GQA
                DispatchAttentionBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Attn Out with fused residual addition (_dXBatch += attnOut)
                DispatchGemmBatch(cmd, lw.AttnOutType, null, _dAttnOutBatch, lw.AttnOutWeight, _dim, _dim, chunkSize, null, residual: _dXBatch);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // FFN pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.FfnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched FFN Gate and Up projections
                DispatchGemmBatch(cmd, lw.FfnGateType, _dGateBatch, _dNormXBatch, lw.FfnGateWeight, _dim, _ffnDim, chunkSize);
                DispatchGemmBatch(cmd, lw.FfnUpType, _dUpBatch, _dNormXBatch, lw.FfnUpWeight, _dim, _ffnDim, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // SwiGLU activation across all tokens
                DispatchSwiglu(cmd, _dGateBatch, _dUpBatch, _dFfnActBatch, chunkSize * _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched FFN Down projection with fused residual addition (_dXBatch += ffnDown)
                DispatchGemmBatch(cmd, lw.FfnDownType, null, _dFfnActBatch, lw.FfnDownWeight, _ffnDim, _dim, chunkSize, null, residual: _dXBatch);
                cmd.ResourceBarrierUnorderedAccessView(null!);


            }

            if (isLastChunk && computeLogits)
            {
                // Copy the final token's hidden state into _dX for LM Head evaluation
                ulong lastTokenOffset = (ulong)((chunkSize - 1) * _dim * sizeof(float));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(_dX, 0, _dXBatch, lastTokenOffset, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);

                // Final RMSNorm on final token
                DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Output projection (LM Head)
                DispatchGemv(cmd, _weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                if (!logits.IsEmpty)
                {
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
                }
            }
            swRec.Stop();
            var swGpu = Stopwatch.StartNew();
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            swGpu.Stop();
            LastTimings = (swRec.Elapsed.TotalMilliseconds, swGpu.Elapsed.TotalMilliseconds);


            if (isLastChunk && computeLogits && !logits.IsEmpty)
            {
                fixed (float* pLogits = logits)
                {
                    Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
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

            _psoGemvQ4K?.Dispose();
            _psoGemvQ6K?.Dispose();
            _psoGemvFp32?.Dispose();
            _sigGemv?.Dispose();

            _psoGemmQ4KBatch?.Dispose();
            _psoGemmQ6KBatch?.Dispose();
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
