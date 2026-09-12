namespace Glacier.Inference.Gpu;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// GPU-accelerated transformer layer weights residing directly in GPU VRAM.
/// </summary>
public sealed class GpuLayerWeights
{
    public IntPtr AttnNormWeight { get; init; }
    public IntPtr QWeight { get; init; }
    public IntPtr QBias { get; init; }
    public IntPtr KWeight { get; init; }
    public IntPtr KBias { get; init; }
    public IntPtr VWeight { get; init; }
    public IntPtr VBias { get; init; }
    public IntPtr AttnOutWeight { get; init; }

    public IntPtr FfnNormWeight { get; init; }
    public IntPtr FfnGateWeight { get; init; }
    public IntPtr FfnUpWeight { get; init; }
    public IntPtr FfnDownWeight { get; init; }

    public GgufType QType { get; init; }
    public GgufType KType { get; init; }
    public GgufType VType { get; init; }
    public GgufType AttnOutType { get; init; }
    public GgufType FfnGateType { get; init; }
    public GgufType FfnUpType { get; init; }
    public GgufType FfnDownType { get; init; }
}

/// <summary>
/// High-performance bare-metal GPU transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on NVIDIA hardware via nvcuda.dll with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe class Qwen2GpuModel : IDisposable
{
    private readonly GpuContext _gpu;
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _headDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _groupSize;
    private readonly float _attnScale;
    private readonly int _maxSeqLen;

    private IntPtr _module;
    private IntPtr _fnGemvQ4K;
    private IntPtr _fnGemvQ6K;
    private IntPtr _fnSwigluFused;
    private IntPtr _fnRmsNorm;
    private IntPtr _fnSwiglu;
    private IntPtr _fnAddBias;
    private IntPtr _fnVecAdd;
    private IntPtr _fnRope;
    private IntPtr _fnKvCacheStore;
    private IntPtr _fnKvCacheStoreF16;
    private IntPtr _fnKvCacheStoreFp8;
    private IntPtr _fnAttentionGqa;
    private IntPtr _fnAttentionGqaF16;
    private IntPtr _fnAttentionGqaFp8;

    // Batched Prefill Kernels & Buffers (batchSize <= 32)
    public const int MaxBatchSize = 32;
    private IntPtr _fnGemmQ4KBatch;
    private IntPtr _fnGemmQ6KBatch;
    private IntPtr _fnGemmSwigluBatch;
    private IntPtr _fnRmsNormBatch;
    private IntPtr _fnAddBiasBatch;
    private IntPtr _fnVecAddBatch;
    private IntPtr _fnRopeBatch;
    private IntPtr _fnKvCacheStoreBatch;
    private IntPtr _fnKvCacheStoreBatchF16;
    private IntPtr _fnKvCacheStoreBatchFp8;
    private IntPtr _fnAttentionGqaBatch;
    private IntPtr _fnAttentionGqaBatchF16;
    private IntPtr _fnAttentionGqaBatchFp8;
    private IntPtr _fnArgmax;
    private IntPtr _fnRepetitionPenalty;

    public KvCachePrecision KvPrecision { get; }

    // CUDA Non-Blocking Streams and Events for Concurrent Projections
    private IntPtr _streamK;
    private IntPtr _streamV;
    private IntPtr _eventNormDone;
    private IntPtr _eventKDone;
    private IntPtr _eventVDone;

    // GPU VRAM Weights
    private IntPtr _dOutNormWeight;
    private IntPtr _dOutWeight;
    private readonly GpuLayerWeights[] _layerWeights;

    // GPU VRAM Intermediate Buffers
    private IntPtr _dX;
    private IntPtr _dNormX;
    private IntPtr _dQ;
    private IntPtr _dK;
    private IntPtr _dV;
    private IntPtr _dAttnOut;
    private IntPtr _dAttnProj;
    private IntPtr _dGate;
    private IntPtr _dUp;
    private IntPtr _dFfnAct;
    private IntPtr _dFfnOut;
    private IntPtr _dLogits;
    private IntPtr _dScoresBuf;
    private IntPtr _dBestToken;
    private IntPtr _dBestLogit;
    private IntPtr _dRecentTokens;
    private float[]? _hostLogits;
    private readonly Sampling.Sampler _sampler = new();

    // GPU VRAM Batch Scratch Buffers
    private IntPtr _dXBatch;
    private IntPtr _dNormXBatch;
    private IntPtr _dQBatch;
    private IntPtr _dKBatch;
    private IntPtr _dVBatch;
    private IntPtr _dAttnOutBatch;
    private IntPtr _dAttnProjBatch;
    private IntPtr _dFfnActBatch;
    private IntPtr _dFfnOutBatch;
    private IntPtr _dScoresBufBatch;
    private float[] _hXBatch;

    // GPU VRAM KV Cache
    private readonly IntPtr[] _dKeyCache;
    private readonly IntPtr[] _dValCache;

    // Host buffer for input embedding
    private float[] _hX;

    private bool _disposed;

    public GpuContext Context => _gpu;
    public ModelWeights Weights => _weights;

    public Qwen2GpuModel(GpuContext gpu, ModelWeights weights, int maxSeqLen = 4096, KvCachePrecision kvPrecision = KvCachePrecision.Auto)
    {
        _gpu = gpu;
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _maxSeqLen = maxSeqLen;

        if (kvPrecision == KvCachePrecision.Auto)
        {
            // Adaptive resolution based on sequence length and 8GB laptop VRAM limits:
            // maxSeqLen <= 4096: FP16 (lossless, 235 MB @ 4k, halves attention bandwidth)
            // maxSeqLen > 4096: FP8 (4x compression, 470 MB @ 16k, enables up to 32k on 8 GB GPU)
            KvPrecision = maxSeqLen <= 4096 ? KvCachePrecision.Fp16 : KvCachePrecision.Fp8;
        }
        else
        {
            KvPrecision = kvPrecision;
        }

        // 1. Load compiled CUBIN kernel module for target GPU architecture
        byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), $"cuModuleLoadData(kernels.cubin, {_gpu.ArchString})");

        // 2. Retrieve kernel function handles (using fast vectorized kernels)
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ4K, _module, "gemv_q4_k_fast"), "ModuleGetFunction(gemv_q4_k_fast)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ6K, _module, "gemv_q6_k_fast"), "ModuleGetFunction(gemv_q6_k_fast)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwigluFused, _module, "gemv_q4_k_swiglu_fused"), "ModuleGetFunction(gemv_q4_k_swiglu_fused)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNorm, _module, "rms_norm_kernel"), "ModuleGetFunction(rms_norm_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "swiglu_kernel"), "ModuleGetFunction(swiglu_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddBias, _module, "add_bias_kernel"), "ModuleGetFunction(add_bias_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _module, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRope, _module, "rope_kernel"), "ModuleGetFunction(rope_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStore, _module, "kv_cache_store_kernel"), "ModuleGetFunction(kv_cache_store_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreF16, _module, "kv_cache_store_f16"), "ModuleGetFunction(kv_cache_store_f16)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreFp8, _module, "kv_cache_store_fp8"), "ModuleGetFunction(kv_cache_store_fp8)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqa, _module, "attention_gqa_kernel"), "ModuleGetFunction(attention_gqa_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaF16, _module, "attention_gqa_f16"), "ModuleGetFunction(attention_gqa_f16)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaFp8, _module, "attention_gqa_fp8"), "ModuleGetFunction(attention_gqa_fp8)");

        // 2b. Retrieve batched prefill kernels
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _module, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBatch, _module, "gemm_q6_k_batch"), "ModuleGetFunction(gemm_q6_k_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmSwigluBatch, _module, "gemm_q4_k_swiglu_batch"), "ModuleGetFunction(gemm_q4_k_swiglu_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatch, _module, "rms_norm_batch"), "ModuleGetFunction(rms_norm_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddBiasBatch, _module, "add_bias_batch"), "ModuleGetFunction(add_bias_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAddBatch, _module, "vec_add_batch"), "ModuleGetFunction(vec_add_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRopeBatch, _module, "rope_batch"), "ModuleGetFunction(rope_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatch, _module, "kv_cache_store_batch"), "ModuleGetFunction(kv_cache_store_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatchF16, _module, "kv_cache_store_batch_f16"), "ModuleGetFunction(kv_cache_store_batch_f16)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatchFp8, _module, "kv_cache_store_batch_fp8"), "ModuleGetFunction(kv_cache_store_batch_fp8)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatch, _module, "attention_gqa_batch"), "ModuleGetFunction(attention_gqa_batch)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatchF16, _module, "attention_gqa_batch_f16"), "ModuleGetFunction(attention_gqa_batch_f16)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatchFp8, _module, "attention_gqa_batch_fp8"), "ModuleGetFunction(attention_gqa_batch_fp8)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnArgmax, _module, "argmax_kernel"), "ModuleGetFunction(argmax_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRepetitionPenalty, _module, "apply_repetition_penalty_kernel"), "ModuleGetFunction(apply_repetition_penalty_kernel)");

        // 2c. Create non-blocking streams and events for concurrent Q/K/V projections
        CuDriver.Check(CuDriver.StreamCreate(out _streamK, 1), "StreamCreate(streamK)");
        CuDriver.Check(CuDriver.StreamCreate(out _streamV, 1), "StreamCreate(streamV)");
        CuDriver.Check(CuDriver.EventCreate(out _eventNormDone, 0), "EventCreate(eventNormDone)");
        CuDriver.Check(CuDriver.EventCreate(out _eventKDone, 0), "EventCreate(eventKDone)");
        CuDriver.Check(CuDriver.EventCreate(out _eventVDone, 0), "EventCreate(eventVDone)");

        // 3. Allocate GPU VRAM scratch buffers (single token)
        _dX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dNormX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dQ = _gpu.AllocateDevice((nuint)(_nHeads * _headDim * sizeof(float)));
        _dK = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _dV = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _dAttnOut = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dAttnProj = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dGate = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dUp = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dFfnAct = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dFfnOut = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dLogits = _gpu.AllocateDevice((nuint)(_weights.VocabSize * sizeof(float)));
        _dScoresBuf = _gpu.AllocateDevice((nuint)((long)_nHeads * _maxSeqLen * sizeof(float)));
        _dBestToken = _gpu.AllocateDevice((nuint)sizeof(int));
        _dBestLogit = _gpu.AllocateDevice((nuint)sizeof(float));
        _dRecentTokens = _gpu.AllocateDevice((nuint)(1024 * sizeof(int)));

        // 3b. Allocate GPU VRAM batch scratch buffers (prefill batch <= 32)
        _dXBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _dNormXBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _dQBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeads * _headDim * sizeof(float)));
        _dKBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
        _dVBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
        _dAttnOutBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _dAttnProjBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _dFfnActBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _ffnDim * sizeof(float)));
        _dFfnOutBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
        _dScoresBufBatch = _gpu.AllocateDevice((nuint)((long)MaxBatchSize * _nHeads * _maxSeqLen * sizeof(float)));
        _hXBatch = new float[MaxBatchSize * _dim];

        // 4. Allocate GPU VRAM KV Cache per layer with precision-specific byte footprint
        _dKeyCache = new IntPtr[_weights.BlockCount];
        _dValCache = new IntPtr[_weights.BlockCount];
        int elementBytes = KvPrecision switch
        {
            KvCachePrecision.Fp8 => 1,
            KvCachePrecision.Fp16 => 2,
            _ => 4
        };
        nuint kvBytes = (nuint)((long)_nHeadsKv * _maxSeqLen * _headDim * elementBytes);
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            _dKeyCache[l] = _gpu.AllocateDevice(kvBytes);
            _dValCache[l] = _gpu.AllocateDevice(kvBytes);
        }

        _hX = new float[_dim];

        // 4. Upload model weights into GPU VRAM
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($">> Uploading 4.68 GB model weights into {gpu.DeviceName} VRAM...");
        var sw = Stopwatch.StartNew();

        _dOutNormWeight = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _gpu.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (nuint)(_dim * sizeof(float)));

        nuint outBytes = (nuint)GgufTypes.GetRowBytes(_weights.OutType, _dim) * (nuint)_weights.VocabSize;
        _dOutWeight = _gpu.AllocateDevice(outBytes);
        _gpu.CopyToDevice(_dOutWeight, (IntPtr)_weights.OutWeight, outBytes);

        _layerWeights = new GpuLayerWeights[_weights.BlockCount];
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];

            nuint qDim = (nuint)(_nHeads * _headDim);
            nuint kvDim = (nuint)(_nHeadsKv * _headDim);

            nuint qBytes = (nuint)GgufTypes.GetRowBytes(lw.QType, _dim) * qDim;
            nuint kBytes = (nuint)GgufTypes.GetRowBytes(lw.KType, _dim) * kvDim;
            nuint vBytes = (nuint)GgufTypes.GetRowBytes(lw.VType, _dim) * kvDim;
            nuint attnOutBytes = (nuint)GgufTypes.GetRowBytes(lw.AttnOutType, _dim) * (nuint)_dim;

            nuint ffnGateBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnGateType, _dim) * (nuint)_ffnDim;
            nuint ffnUpBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnUpType, _dim) * (nuint)_ffnDim;
            nuint ffnDownBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnDownType, _ffnDim) * (nuint)_dim;

            IntPtr dAttnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _gpu.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (nuint)(_dim * sizeof(float)));

            IntPtr dQ = _gpu.AllocateDevice(qBytes);
            _gpu.CopyToDevice(dQ, (IntPtr)lw.QWeight, qBytes);
            IntPtr dQBias = _gpu.AllocateDevice(qDim * sizeof(float));
            _gpu.CopyToDevice(dQBias, (IntPtr)lw.QBias, qDim * sizeof(float));

            IntPtr dK = _gpu.AllocateDevice(kBytes);
            _gpu.CopyToDevice(dK, (IntPtr)lw.KWeight, kBytes);
            IntPtr dKBias = _gpu.AllocateDevice(kvDim * sizeof(float));
            _gpu.CopyToDevice(dKBias, (IntPtr)lw.KBias, kvDim * sizeof(float));

            IntPtr dV = _gpu.AllocateDevice(vBytes);
            _gpu.CopyToDevice(dV, (IntPtr)lw.VWeight, vBytes);
            IntPtr dVBias = _gpu.AllocateDevice(kvDim * sizeof(float));
            _gpu.CopyToDevice(dVBias, (IntPtr)lw.VBias, kvDim * sizeof(float));

            IntPtr dAttnOut = _gpu.AllocateDevice(attnOutBytes);
            _gpu.CopyToDevice(dAttnOut, (IntPtr)lw.AttnOutWeight, attnOutBytes);

            IntPtr dFfnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _gpu.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (nuint)(_dim * sizeof(float)));

            IntPtr dFfnGate = _gpu.AllocateDevice(ffnGateBytes);
            _gpu.CopyToDevice(dFfnGate, (IntPtr)lw.FfnGateWeight, ffnGateBytes);

            IntPtr dFfnUp = _gpu.AllocateDevice(ffnUpBytes);
            _gpu.CopyToDevice(dFfnUp, (IntPtr)lw.FfnUpWeight, ffnUpBytes);

            IntPtr dFfnDown = _gpu.AllocateDevice(ffnDownBytes);
            _gpu.CopyToDevice(dFfnDown, (IntPtr)lw.FfnDownWeight, ffnDownBytes);

            _layerWeights[l] = new GpuLayerWeights
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
        Console.WriteLine($"   Weights uploaded to GPU VRAM in {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F2} s)!");
        Console.ResetColor();
    }

    /// <summary>
    /// Executes transformer forward pass directly on NVIDIA GPU with zero intermediate PCIe round-trips.
    /// RoPE, GQA Attention, RMSNorm, GEMV, SwiGLU, and residual additions all execute in-VRAM.
    /// </summary>
    public void Forward(int token, int pos, KVCache? kvCache = null, Span<float> logits = default, bool computeLogits = true)
    {
        // 1. Extract embedding into host buffer and copy to GPU
        fixed (float* pX = _hX)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, pX, _dim);
            _gpu.CopyToDevice(_dX, (IntPtr)pX, (nuint)(_dim * sizeof(float)));
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Transformer layers (100% inside GPU VRAM)
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _layerWeights[l];

            // Attention pre-norm in VRAM
            LaunchRmsNorm(_dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Wait on GPU for RMSNorm output to be ready before K and V read it on their streams
            CuDriver.EventRecord(_eventNormDone, IntPtr.Zero);
            CuDriver.StreamWaitEvent(_streamK, _eventNormDone, 0);
            CuDriver.StreamWaitEvent(_streamV, _eventNormDone, 0);

            // Q, K, V projections on GPU (fused bias addition; K and V launched concurrently on dedicated CUDA streams)
            LaunchGemv(lw.QType, _dQ, _dNormX, lw.QWeight, _dim, qDim, lw.QBias);
            LaunchGemv(lw.KType, _dK, _dNormX, lw.KWeight, _dim, kvDim, lw.KBias, hStream: _streamK);
            LaunchGemv(lw.VType, _dV, _dNormX, lw.VWeight, _dim, kvDim, lw.VBias, hStream: _streamV);

            // Instruct default stream to wait on GPU for K and V completion prior to RoPE and KV cache storage
            CuDriver.EventRecord(_eventKDone, _streamK);
            CuDriver.EventRecord(_eventVDone, _streamV);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventKDone, 0);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventVDone, 0);

            // Rotary Position Embedding (RoPE) directly in VRAM
            LaunchRope(_dQ, _dK, pos);

            // Store in GPU VRAM KV Cache
            LaunchKvStore(l, pos);

            // Multi-Head / Grouped Query Attention (GQA) in VRAM
            LaunchAttention(l, pos);

            // Attention out projection (fused residual accumulation directly into _dX: _dX += attnProj)
            LaunchGemv(lw.AttnOutType, IntPtr.Zero, _dAttnOut, lw.AttnOutWeight, _dim, _dim, IntPtr.Zero, _dX);

            // FFN pre-norm
            LaunchRmsNorm(_dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // SwiGLU FFN projections on GPU (Fused kernel computes Gate + Up in registers and applies SiLU)
            if (lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
            {
                LaunchSwigluFused(_dFfnAct, _dNormX, lw.FfnGateWeight, lw.FfnUpWeight, _dim, _ffnDim);
            }
            else
            {
                LaunchGemv(lw.FfnGateType, _dGate, _dNormX, lw.FfnGateWeight, _dim, _ffnDim);
                LaunchGemv(lw.FfnUpType, _dUp, _dNormX, lw.FfnUpWeight, _dim, _ffnDim);
                LaunchSwiglu(_dGate, _dUp, _dFfnAct, _ffnDim);
            }

            // FFN down projection (fused residual accumulation directly into _dX: _dX += ffnDown)
            LaunchGemv(lw.FfnDownType, IntPtr.Zero, _dFfnAct, lw.FfnDownWeight, _ffnDim, _dim, IntPtr.Zero, _dX);
        }

        if (computeLogits)
        {
            // Final RMSNorm
            LaunchRmsNorm(_dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Output projection (LM Head) on GPU
            LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);

            // Copy logits from GPU to CPU (skipped when logits buffer is empty for pure GPU sampling)
            if (!logits.IsEmpty)
            {
                fixed (float* pLogits = logits)
                {
                    _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
                }
            }
        }
    }

    /// <summary>
    /// Executes batched transformer prompt prefill directly on NVIDIA GPU.
    /// Weights are loaded once from VRAM and multiplied across up to MaxBatchSize tokens simultaneously.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        int offset = 0;
        while (offset < tokens.Length)
        {
            int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
            bool isLastChunk = (offset + batchSize == tokens.Length);
            ForwardBatchChunk(
                tokens.Slice(offset, batchSize),
                startPos + offset,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                isLastChunk && computeLogits);
            offset += batchSize;
        }
    }

    /// <summary>
    /// Executes batched speculative verification directly on GPU.
    /// Evaluates all candidate tokens in a single batched transformer pass,
    /// computes LM Head and fast GPU argmax for each position,
    /// and writes the predicted next token for each candidate position.
    /// </summary>
    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        int batchSize = tokens.Length;
        if (batchSize == 0) return;
        if (batchSize > MaxBatchSize)
            throw new ArgumentException($"Batch size {batchSize} exceeds MaxBatchSize {MaxBatchSize}");

        // Forward through all transformer layers in batch
        ForwardBatchChunk(tokens, startPos, Span<float>.Empty, computeLogits: false);

        // For each token position in the batch, compute RMSNorm + LM Head + GPU argmax
        for (int t = 0; t < batchSize; t++)
        {
            IntPtr dXt = _dXBatch + t * _dim * sizeof(float);
            LaunchRmsNorm(dXt, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
            predictedTokens[t] = SampleGreedy();
        }
    }

    /// <summary>
    /// Executes fast GPU argmax reduction directly across the 152K vocab logits in ~3 microseconds.
    /// Transfers only 4 bytes (the sampled token ID) across PCIe, bypassing 608 KB host copies.
    /// </summary>
    public int SampleGreedy(ReadOnlySpan<int> recentTokens = default, float repetitionPenalty = 1.0f)
    {
        IntPtr dLogits = _dLogits;
        IntPtr dBestToken = _dBestToken;
        IntPtr dBestLogit = _dBestLogit;
        IntPtr dRecentTokens = _dRecentTokens;

        if (repetitionPenalty != 1.0f && !recentTokens.IsEmpty)
        {
            int count = Math.Min(recentTokens.Length, 1024);
            fixed (int* pRecent = recentTokens)
            {
                _gpu.CopyToDevice(dRecentTokens, (IntPtr)pRecent, (nuint)(count * sizeof(int)));
            }

            void** pPenArgs = stackalloc void*[4];
            pPenArgs[0] = &dLogits;
            pPenArgs[1] = &dRecentTokens;
            pPenArgs[2] = &count;
            pPenArgs[3] = &repetitionPenalty;

            uint blockSize = 64;
            uint gridSize = (uint)((count + (int)blockSize - 1) / (int)blockSize);

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnRepetitionPenalty,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pPenArgs,
                IntPtr.Zero), "LaunchKernel(apply_repetition_penalty)");
        }

        int vocabSize = _weights.VocabSize;
        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dLogits;
        pArgs[1] = &vocabSize;
        pArgs[2] = &dBestToken;
        pArgs[3] = &dBestLogit;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnArgmax,
            1, 1, 1,
            512, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(argmax)");

        int bestToken = 0;
        _gpu.CopyToHost((IntPtr)(&bestToken), dBestToken, (nuint)sizeof(int));
        return bestToken;
    }

    /// <summary>
    /// Samples next token with support for greedy GPU reduction and CPU temperature / top-P sampling.
    /// </summary>
    public int SampleToken(Sampling.SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        if (options.Temperature <= 0.001f || options.TopK == 1)
        {
            return SampleGreedy(recentTokens, options.RepetitionPenalty);
        }

        _hostLogits ??= new float[_weights.VocabSize];
        fixed (float* pLogits = _hostLogits)
        {
            _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
        }

        return _sampler.Sample(_hostLogits.AsSpan(), options, recentTokens);
    }

    private void ForwardBatchChunk(ReadOnlySpan<int> chunkTokens, int chunkStartPos, Span<float> logits, bool computeLogits)
    {
        int batchSize = chunkTokens.Length;
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 1. Extract embeddings into host batch buffer and copy to GPU
        fixed (float* pXBatch = _hXBatch)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], pXBatch + t * _dim, _dim);
            }
            _gpu.CopyToDevice(_dXBatch, (IntPtr)pXBatch, (nuint)(batchSize * _dim * sizeof(float)));
        }

        // 2. Transformer layers in batch
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _layerWeights[l];

            LaunchRmsNormBatch(_dXBatch, lw.AttnNormWeight, _dNormXBatch, _dim, _weights.RmsNormEps, batchSize);
            CuDriver.EventRecord(_eventNormDone, IntPtr.Zero);
            CuDriver.StreamWaitEvent(_streamK, _eventNormDone, 0);
            CuDriver.StreamWaitEvent(_streamV, _eventNormDone, 0);

            LaunchGemmBatch(lw.QType, _dQBatch, _dNormXBatch, lw.QWeight, _dim, qDim, batchSize, lw.QBias);
            LaunchGemmBatch(lw.KType, _dKBatch, _dNormXBatch, lw.KWeight, _dim, kvDim, batchSize, lw.KBias, hStream: _streamK);
            LaunchGemmBatch(lw.VType, _dVBatch, _dNormXBatch, lw.VWeight, _dim, kvDim, batchSize, lw.VBias, hStream: _streamV);
            CuDriver.EventRecord(_eventKDone, _streamK);
            CuDriver.EventRecord(_eventVDone, _streamV);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventKDone, 0);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventVDone, 0);

            LaunchRopeBatch(_dQBatch, _dKBatch, chunkStartPos, batchSize);
            LaunchKvCacheStoreBatch(l, chunkStartPos, batchSize);

            LaunchAttentionBatch(l, chunkStartPos, batchSize);

            LaunchGemmBatch(lw.AttnOutType, IntPtr.Zero, _dAttnOutBatch, lw.AttnOutWeight, _dim, _dim, batchSize, IntPtr.Zero, _dXBatch);

            LaunchRmsNormBatch(_dXBatch, lw.FfnNormWeight, _dNormXBatch, _dim, _weights.RmsNormEps, batchSize);

            if (lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
            {
                LaunchSwigluBatch(_dFfnActBatch, _dNormXBatch, lw.FfnGateWeight, lw.FfnUpWeight, _dim, _ffnDim, batchSize);
            }
            else
            {
                for (int t = 0; t < batchSize; t++)
                {
                    IntPtr dNorm = _dNormXBatch + t * _dim * sizeof(float);
                    IntPtr dAct = _dFfnActBatch + t * _ffnDim * sizeof(float);
                    LaunchGemv(lw.FfnGateType, _dGate, dNorm, lw.FfnGateWeight, _dim, _ffnDim);
                    LaunchGemv(lw.FfnUpType, _dUp, dNorm, lw.FfnUpWeight, _dim, _ffnDim);
                    LaunchSwiglu(_dGate, _dUp, dAct, _ffnDim);
                }
            }

            LaunchGemmBatch(lw.FfnDownType, IntPtr.Zero, _dFfnActBatch, lw.FfnDownWeight, _ffnDim, _dim, batchSize, IntPtr.Zero, _dXBatch);
        }

        if (computeLogits)
        {
            // We only need logits for the last token in the batch
            int lastTokenIdx = batchSize - 1;
            IntPtr dXLast = _dXBatch + lastTokenIdx * _dim * sizeof(float);

            // Final RMSNorm
            LaunchRmsNorm(dXLast, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Output projection (LM Head) on GPU
            LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);

            // Copy logits from GPU to CPU
            fixed (float* pLogits = logits)
            {
                _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
            }
        }
    }

    private void LaunchRope(IntPtr dQ, IntPtr dK, int pos)
    {
        int halfDim = _headDim / 2;
        int totalHalf = (_nHeads + _nHeadsKv) * halfDim;
        uint blockSize = 256;
        uint gridSize = (uint)((totalHalf + (int)blockSize - 1) / (int)blockSize);

        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        float freqBase = _weights.RopeFreqBase;
        float freqScale = 1.0f;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &pos;
        pArgs[6] = &freqBase;
        pArgs[7] = &freqScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRope,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope)");
    }

    private void LaunchKvStore(int layer, int pos)
    {
        int total = _nHeadsKv * _headDim;
        uint blockSize = 256;
        uint gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        IntPtr dK = _dK;
        IntPtr dV = _dV;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dKCache;
        pArgs[1] = &dVCache;
        pArgs[2] = &dK;
        pArgs[3] = &dV;
        pArgs[4] = &nHeadsKv;
        pArgs[5] = &headDim;
        pArgs[6] = &maxSeq;
        pArgs[7] = &pos;

        IntPtr fnKv = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnKvCacheStoreF16,
            KvCachePrecision.Fp8 => _fnKvCacheStoreFp8,
            _ => _fnKvCacheStore
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKv,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store)");
    }

    private void LaunchAttention(int layer, int pos) => LaunchAttention(layer, pos, _dQ, _dAttnOut);

    private void LaunchAttention(int layer, int pos, IntPtr dQ, IntPtr dOut)
    {
        uint gridSize = (uint)_nHeads;
        uint blockSize = (uint)_headDim;

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dScores = _dScoresBuf;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[11];
        pArgs[0] = &dQ;
        pArgs[1] = &dKCache;
        pArgs[2] = &dVCache;
        pArgs[3] = &dOut;
        pArgs[4] = &dScores;
        pArgs[5] = &nHeadsQ;
        pArgs[6] = &nHeadsKv;
        pArgs[7] = &headDim;
        pArgs[8] = &maxSeq;
        pArgs[9] = &pos;
        pArgs[10] = &attnScale;

        IntPtr fnAttn = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnAttentionGqaF16,
            KvCachePrecision.Fp8 => _fnAttentionGqaFp8,
            _ => _fnAttentionGqa
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa)");
    }

    private void LaunchAttentionBatch(int layer, int chunkStartPos, int batchSize)
    {
        uint gridX = (uint)_nHeads;
        uint gridY = (uint)batchSize;
        uint blockSize = (uint)_headDim;

        IntPtr dQ = _dQBatch;
        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dOut = _dAttnOutBatch;
        IntPtr dScores = _dScoresBufBatch;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[12];
        pArgs[0] = &dQ;
        pArgs[1] = &dKCache;
        pArgs[2] = &dVCache;
        pArgs[3] = &dOut;
        pArgs[4] = &dScores;
        pArgs[5] = &nHeadsQ;
        pArgs[6] = &nHeadsKv;
        pArgs[7] = &headDim;
        pArgs[8] = &maxSeq;
        pArgs[9] = &chunkStartPos;
        pArgs[10] = &batchSize;
        pArgs[11] = &attnScale;

        IntPtr fnAttnBatch = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnAttentionGqaBatchF16,
            KvCachePrecision.Fp8 => _fnAttentionGqaBatchFp8,
            _ => _fnAttentionGqaBatch
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttnBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa_batch)");
    }

    private void LaunchGemv(
        GgufType type,
        IntPtr dY,
        IntPtr dX,
        IntPtr dW,
        int kCols,
        int mRows,
        IntPtr dBias = default,
        IntPtr dResidual = default,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemvQ4K,
            GgufType.Q6_K => _fnGemvQ6K,
            _ => throw new NotSupportedException($"GPU GEMV does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &dBias;
        pArgs[6] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemv)");
    }

    private void LaunchSwigluFused(IntPtr dDst, IntPtr dX, IntPtr dWGate, IntPtr dWUp, int kCols, int mRows)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dDst;
        pArgs[1] = &dX;
        pArgs[2] = &dWGate;
        pArgs[3] = &dWUp;
        pArgs[4] = &kCols;
        pArgs[5] = &mRows;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwigluFused,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemv_q4_k_swiglu_fused)");
    }

    private void LaunchRmsNorm(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps)
    {
        uint blockSize = 256;
        uint gridSize = 1;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &size;
        pArgs[4] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNorm,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm)");
    }

    private void LaunchSwiglu(IntPtr dGate, IntPtr dUp, IntPtr dDst, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dGate;
        pArgs[1] = &dUp;
        pArgs[2] = &dDst;
        pArgs[3] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwiglu,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(swiglu)");
    }

    private void LaunchAddBias(IntPtr dY, IntPtr dBias, int size, IntPtr hStream = default)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dY;
        pArgs[1] = &dBias;
        pArgs[2] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAddBias,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(add_bias)");
    }

    private void LaunchVecAdd(IntPtr dA, IntPtr dB, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dA;
        pArgs[1] = &dB;
        pArgs[2] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAdd,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add)");
    }

    private void LaunchGemmBatch(
        GgufType type,
        IntPtr dY,
        IntPtr dX,
        IntPtr dW,
        int kCols,
        int mRows,
        int batchSize,
        IntPtr dBias = default,
        IntPtr dResidual = default,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBatch,
            GgufType.Q6_K => _fnGemmQ6KBatch,
            _ => throw new NotSupportedException($"GPU batch GEMM does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = 1;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dBias;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_batch)");
    }

    private void LaunchSwigluBatch(IntPtr dDst, IntPtr dX, IntPtr dWGate, IntPtr dWUp, int kCols, int mRows, int batchSize)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = 1;

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dDst;
        pArgs[1] = &dX;
        pArgs[2] = &dWGate;
        pArgs[3] = &dWUp;
        pArgs[4] = &kCols;
        pArgs[5] = &mRows;
        pArgs[6] = &batchSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmSwigluBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q4_k_swiglu_batch)");
    }

    private void LaunchRmsNormBatch(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps, int batchSize)
    {
        uint blockSize = 256;
        uint gridSize = (uint)batchSize;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &size;
        pArgs[4] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_batch)");
    }

    private void LaunchAddBiasBatch(IntPtr dY, IntPtr dBias, int size, int batchSize, IntPtr hStream = default)
    {
        int totalElements = size * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dY;
        pArgs[1] = &dBias;
        pArgs[2] = &size;
        pArgs[3] = &totalElements;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAddBiasBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(add_bias_batch)");
    }

    private void LaunchVecAddBatch(IntPtr dA, IntPtr dB, int totalElements)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dA;
        pArgs[1] = &dB;
        pArgs[2] = &totalElements;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAddBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add_batch)");
    }

    private void LaunchRopeBatch(IntPtr dQ, IntPtr dK, int startPos, int batchSize)
    {
        int halfDim = _headDim / 2;
        int totalHalf = (_nHeads + _nHeadsKv) * halfDim;
        int totalAll = totalHalf * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((totalAll + (int)blockSize - 1) / (int)blockSize);

        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        float freqBase = _weights.RopeFreqBase;
        float freqScale = 1.0f;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &freqBase;
        pArgs[8] = &freqScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRopeBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_batch)");
    }

    private void LaunchKvCacheStoreBatch(int layer, int startPos, int batchSize)
    {
        int total = _nHeadsKv * _headDim * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dK = _dKBatch;
        IntPtr dV = _dVBatch;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dKCache;
        pArgs[1] = &dVCache;
        pArgs[2] = &dK;
        pArgs[3] = &dV;
        pArgs[4] = &nHeadsKv;
        pArgs[5] = &headDim;
        pArgs[6] = &maxSeq;
        pArgs[7] = &startPos;
        pArgs[8] = &batchSize;

        IntPtr fnKvBatch = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnKvCacheStoreBatchF16,
            KvCachePrecision.Fp8 => _fnKvCacheStoreBatchFp8,
            _ => _fnKvCacheStoreBatch
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKvBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store_batch)");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_eventNormDone != IntPtr.Zero) CuDriver.EventDestroy(_eventNormDone);
            if (_eventKDone != IntPtr.Zero) CuDriver.EventDestroy(_eventKDone);
            if (_eventVDone != IntPtr.Zero) CuDriver.EventDestroy(_eventVDone);
            if (_streamK != IntPtr.Zero) CuDriver.StreamDestroy(_streamK);
            if (_streamV != IntPtr.Zero) CuDriver.StreamDestroy(_streamV);

            _gpu.FreeDevice(_dX);
            _gpu.FreeDevice(_dNormX);
            _gpu.FreeDevice(_dQ);
            _gpu.FreeDevice(_dK);
            _gpu.FreeDevice(_dV);
            _gpu.FreeDevice(_dAttnOut);
            _gpu.FreeDevice(_dAttnProj);
            _gpu.FreeDevice(_dGate);
            _gpu.FreeDevice(_dUp);
            _gpu.FreeDevice(_dFfnAct);
            _gpu.FreeDevice(_dFfnOut);
            _gpu.FreeDevice(_dLogits);
            _gpu.FreeDevice(_dScoresBuf);
            _gpu.FreeDevice(_dBestToken);
            _gpu.FreeDevice(_dBestLogit);
            _gpu.FreeDevice(_dRecentTokens);
            _gpu.FreeDevice(_dOutNormWeight);
            _gpu.FreeDevice(_dOutWeight);

            _gpu.FreeDevice(_dXBatch);
            _gpu.FreeDevice(_dNormXBatch);
            _gpu.FreeDevice(_dQBatch);
            _gpu.FreeDevice(_dKBatch);
            _gpu.FreeDevice(_dVBatch);
            _gpu.FreeDevice(_dAttnOutBatch);
            _gpu.FreeDevice(_dAttnProjBatch);
            _gpu.FreeDevice(_dFfnActBatch);
            _gpu.FreeDevice(_dFfnOutBatch);
            _gpu.FreeDevice(_dScoresBufBatch);

            if (_dKeyCache != null)
            {
                for (int l = 0; l < _dKeyCache.Length; l++)
                {
                    if (_dKeyCache[l] != IntPtr.Zero) _gpu.FreeDevice(_dKeyCache[l]);
                    if (_dValCache[l] != IntPtr.Zero) _gpu.FreeDevice(_dValCache[l]);
                }
            }

            if (_layerWeights != null)
            {
                for (int l = 0; l < _layerWeights.Length; l++)
                {
                    var lw = _layerWeights[l];
                    if (lw != null)
                    {
                        _gpu.FreeDevice(lw.AttnNormWeight);
                        _gpu.FreeDevice(lw.QWeight);
                        _gpu.FreeDevice(lw.QBias);
                        _gpu.FreeDevice(lw.KWeight);
                        _gpu.FreeDevice(lw.KBias);
                        _gpu.FreeDevice(lw.VWeight);
                        _gpu.FreeDevice(lw.VBias);
                        _gpu.FreeDevice(lw.AttnOutWeight);
                        _gpu.FreeDevice(lw.FfnNormWeight);
                        _gpu.FreeDevice(lw.FfnGateWeight);
                        _gpu.FreeDevice(lw.FfnUpWeight);
                        _gpu.FreeDevice(lw.FfnDownWeight);
                    }
                }
            }

            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~Qwen2GpuModel()
    {
        Dispose();
    }
}
