namespace Glacier.Inference.Gpu;

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// High-performance bare-metal GPU transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on NVIDIA hardware via nvcuda.dll with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe partial class Qwen2GpuModel : IDisposable
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
    private IntPtr _fnGemvQ5K;
    private IntPtr _fnGemvQ3K;
    private IntPtr _fnGemvQ6K;
    private IntPtr _fnGemvQ8_0;
    private IntPtr _fnGemvFp32;
    private IntPtr _fnSwigluFused;
    private IntPtr _fnRmsNorm;
    private IntPtr _fnSwiglu;
    private IntPtr _fnAddBias;
    private IntPtr _fnVecAdd;
    private IntPtr _fnVecAddWeighted;
    private IntPtr _fnRope;
    private IntPtr _fnKvCacheStore;
    private IntPtr _fnKvCacheStoreF16;
    private IntPtr _fnKvCacheStoreFp8;
    private IntPtr _fnAttentionGqa;
    private IntPtr _fnAttentionGqaF16;
    private IntPtr _fnAttentionGqaFp8;
    private IntPtr _fnRmsNormHeads;

    // Batched Prefill Kernels & Buffers (batchSize <= 32)
    public const int MaxBatchSize = 32;
    private IntPtr _fnGemmQ4KBatch;
    private IntPtr _fnGemmQ6KBatch;
    private IntPtr _fnGemmQ8_0Batch;
    private IntPtr _fnGemmSwigluBatch;
    private IntPtr _fnRmsNormBatch;
    private IntPtr _fnRmsNormBatchHeads;
    private IntPtr _fnAddBiasBatch;
    private IntPtr _fnVecAddBatch;
    private IntPtr _fnRopeBatch;
    private IntPtr _fnKvCacheStoreBatch;
    private IntPtr _fnKvCacheStoreBatchF16;
    private IntPtr _fnKvCacheStoreBatchFp8;
    private IntPtr _fnAttentionGqaBatch;
    private IntPtr _fnAttentionGqaBatchF16;
    private IntPtr _fnAttentionGqaBatchFp8;
    private IntPtr _fnPagedKvStoreF16;
    private IntPtr _fnPagedKvStoreBatchF16;
    private IntPtr _fnPagedAttnGqaF16;
    private IntPtr _fnPagedAttnGqaBatchF16;
    private IntPtr _fnArgmax;
    private IntPtr _fnRepetitionPenalty;

    public KvCachePrecision KvPrecision { get; }
    private readonly bool _allocateContiguousKvCache;
    private readonly nuint _kvBytesPerLayer;

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
    private float[] _hXBatch;

    // GPU VRAM KV Cache
    private readonly IntPtr[] _dKeyCache;
    private readonly IntPtr[] _dValCache;

    // Host buffer for input embedding
    private float[] _hX;

    // VRAM Token Embedding Cache (eliminates host-to-device PCIe copy per token)
    private const int EmbdCacheCapacity = 4096;
    private IntPtr _dEmbdCache = IntPtr.Zero;
    private int[]? _embdCacheTokens;
    private int _embdCacheHead = 0;
    private System.Collections.Generic.Dictionary<int, int>? _embdTokenToSlot;
    private IntPtr _hXPinned = IntPtr.Zero;

    private bool _disposed;

    public GpuContext Context => _gpu;
    public ModelWeights Weights => _weights;
    public int StartLayer { get; }
    public int LayerCount { get; }
    public bool IsLastStage { get; }

    public void ResetKvCache()
    {
        if (_disposed || !_allocateContiguousKvCache || _kvBytesPerLayer == 0) return;
        _gpu.MakeCurrent();
        for (int l = 0; l < LayerCount; l++)
        {
            if (_dKeyCache[l] != IntPtr.Zero)
                CuDriver.Check(CuDriver.MemsetD8(_dKeyCache[l], 0, _kvBytesPerLayer), "MemsetD8(KeyCache)");
            if (_dValCache[l] != IntPtr.Zero)
                CuDriver.Check(CuDriver.MemsetD8(_dValCache[l], 0, _kvBytesPerLayer), "MemsetD8(ValCache)");
        }
    }

    public Qwen2GpuModel(
        GpuContext gpu,
        ModelWeights weights,
        int maxSeqLen = 4096,
        KvCachePrecision kvPrecision = KvCachePrecision.Auto,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true,
        bool allocateContiguousKvCache = true)
    {
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(weights);

        _gpu = gpu;
        _weights = weights;
        StartLayer = startLayer;
        LayerCount = layerCount < 0 ? weights.BlockCount - startLayer : layerCount;
        IsLastStage = isLastStage;
        _allocateContiguousKvCache = allocateContiguousKvCache;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        if (weights.ValueDim != _headDim)
        {
            throw new NotSupportedException(
                $"Qwen2GpuModel requires symmetric K/V head dimensions (HeadDim={_headDim}, ValueDim={weights.ValueDim}).");
        }

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

        int elementBytes = KvPrecision switch
        {
            KvCachePrecision.Fp8 => 1,
            KvCachePrecision.Fp16 => 2,
            _ => 4
        };
        nuint kvBytes = _allocateContiguousKvCache
            ? (nuint)((long)_nHeadsKv * _maxSeqLen * _headDim * elementBytes)
            : 0;
        _kvBytesPerLayer = kvBytes;

        _hX = new float[_dim];
        _hXBatch = new float[MaxBatchSize * _dim];
        _dKeyCache = new IntPtr[LayerCount];
        _dValCache = new IntPtr[LayerCount];
        _layerWeights = new GpuLayerWeights[LayerCount];

        try
        {
            _gpu.MakeCurrent();

            // 1. Load compiled CUBIN kernel module for target GPU architecture
            byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
            CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), $"cuModuleLoadData(kernels.cubin, {_gpu.ArchString})");

            // 2. Retrieve kernel function handles (using fast vectorized kernels)
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ4K, _module, "gemv_q4_k_fast"), "ModuleGetFunction(gemv_q4_k_fast)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ5K, _module, "gemv_q5_k"), "ModuleGetFunction(gemv_q5_k)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ3K, _module, "gemv_q3_k"), "ModuleGetFunction(gemv_q3_k)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ6K, _module, "gemv_q6_k_fast"), "ModuleGetFunction(gemv_q6_k_fast)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ8_0, _module, "gemv_q8_0"), "ModuleGetFunction(gemv_q8_0)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvFp32, _module, "gemv_fp32"), "ModuleGetFunction(gemv_fp32)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwigluFused, _module, "gemv_q4_k_swiglu_fused"), "ModuleGetFunction(gemv_q4_k_swiglu_fused)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNorm, _module, "rms_norm_kernel"), "ModuleGetFunction(rms_norm_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "swiglu_kernel"), "ModuleGetFunction(swiglu_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddBias, _module, "add_bias_kernel"), "ModuleGetFunction(add_bias_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _module, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAddWeighted, _module, "vec_add_weighted_kernel"), "ModuleGetFunction(vec_add_weighted_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRope, _module, "rope_kernel"), "ModuleGetFunction(rope_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStore, _module, "kv_cache_store_kernel"), "ModuleGetFunction(kv_cache_store_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreF16, _module, "kv_cache_store_f16"), "ModuleGetFunction(kv_cache_store_f16)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreFp8, _module, "kv_cache_store_fp8"), "ModuleGetFunction(kv_cache_store_fp8)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqa, _module, "attention_gqa_kernel"), "ModuleGetFunction(attention_gqa_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaF16, _module, "attention_gqa_f16"), "ModuleGetFunction(attention_gqa_f16)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaFp8, _module, "attention_gqa_fp8"), "ModuleGetFunction(attention_gqa_fp8)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormHeads, _module, "rms_norm_heads_kernel"), "ModuleGetFunction(rms_norm_heads_kernel)");

            // 2b. Retrieve batched prefill kernels
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _module, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBatch, _module, "gemm_q6_k_batch"), "ModuleGetFunction(gemm_q6_k_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ8_0Batch, _module, "gemm_q8_0_batch"), "ModuleGetFunction(gemm_q8_0_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmSwigluBatch, _module, "gemm_q4_k_swiglu_batch"), "ModuleGetFunction(gemm_q4_k_swiglu_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatch, _module, "rms_norm_batch"), "ModuleGetFunction(rms_norm_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatchHeads, _module, "rms_norm_batch_heads_kernel"), "ModuleGetFunction(rms_norm_batch_heads_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddBiasBatch, _module, "add_bias_batch"), "ModuleGetFunction(add_bias_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAddBatch, _module, "vec_add_batch"), "ModuleGetFunction(vec_add_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRopeBatch, _module, "rope_batch"), "ModuleGetFunction(rope_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatch, _module, "kv_cache_store_batch"), "ModuleGetFunction(kv_cache_store_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatchF16, _module, "kv_cache_store_batch_f16"), "ModuleGetFunction(kv_cache_store_batch_f16)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStoreBatchFp8, _module, "kv_cache_store_batch_fp8"), "ModuleGetFunction(kv_cache_store_batch_fp8)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatch, _module, "attention_gqa_batch"), "ModuleGetFunction(attention_gqa_batch)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatchF16, _module, "attention_gqa_batch_f16"), "ModuleGetFunction(attention_gqa_batch_f16)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqaBatchFp8, _module, "attention_gqa_batch_fp8"), "ModuleGetFunction(attention_gqa_batch_fp8)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPagedKvStoreF16, _module, "paged_kv_cache_store_f16_kernel"), "ModuleGetFunction(paged_kv_cache_store_f16_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPagedKvStoreBatchF16, _module, "paged_kv_cache_store_batch_f16_kernel"), "ModuleGetFunction(paged_kv_cache_store_batch_f16_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPagedAttnGqaF16, _module, "paged_attention_gqa_f16_kernel"), "ModuleGetFunction(paged_attention_gqa_f16_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPagedAttnGqaBatchF16, _module, "paged_attention_gqa_batch_f16_kernel"), "ModuleGetFunction(paged_attention_gqa_batch_f16_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnArgmax, _module, "argmax_kernel"), "ModuleGetFunction(argmax_kernel)");
            CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRepetitionPenalty, _module, "apply_repetition_penalty_kernel"), "ModuleGetFunction(apply_repetition_penalty_kernel)");

            // 2c. Create non-blocking streams and events for concurrent Q/K/V projections
            CuDriver.Check(CuDriver.StreamCreate(out _streamK, 1), "StreamCreate(streamK)");
            CuDriver.Check(CuDriver.StreamCreate(out _streamV, 1), "StreamCreate(streamV)");
            CuDriver.Check(CuDriver.EventCreate(out _eventNormDone, 0), "EventCreate(eventNormDone)");
            CuDriver.Check(CuDriver.EventCreate(out _eventKDone, 0), "EventCreate(eventKDone)");
            CuDriver.Check(CuDriver.EventCreate(out _eventVDone, 0), "EventCreate(eventVDone)");

            // 3. Allocate GPU VRAM scratch buffers (single token)
            int maxAttnOut = Math.Max(_dim, _nHeads * _headDim);
            _dX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _dNormX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _dQ = _gpu.AllocateDevice((nuint)(_nHeads * _headDim * sizeof(float)));
            _dK = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
            _dV = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
            _dAttnOut = _gpu.AllocateDevice((nuint)(maxAttnOut * sizeof(float)));
            _dAttnProj = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _dGate = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
            _dUp = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
            _dFfnAct = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
            _dFfnOut = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _dLogits = _gpu.AllocateDevice((nuint)(_weights.VocabSize * sizeof(float)));
            _dBestToken = _gpu.AllocateDevice((nuint)sizeof(int));
            _dBestLogit = _gpu.AllocateDevice((nuint)sizeof(float));
            _dRecentTokens = _gpu.AllocateDevice((nuint)(1024 * sizeof(int)));

            // 3b. Allocate GPU VRAM batch scratch buffers (prefill batch <= 32)
            _dXBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
            _dNormXBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
            _dQBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeads * _headDim * sizeof(float)));
            _dKBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
            _dVBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _nHeadsKv * _headDim * sizeof(float)));
            _dAttnOutBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
            _dAttnProjBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));
            _dFfnActBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _ffnDim * sizeof(float)));
            _dFfnOutBatch = _gpu.AllocateDevice((nuint)(MaxBatchSize * _dim * sizeof(float)));

            // 4. Allocate GPU VRAM KV Cache per layer with precision-specific byte footprint (skipped in PagedAttention pool mode)
            if (_allocateContiguousKvCache)
            {
                for (int l = 0; l < LayerCount; l++)
                {
                    _dKeyCache[l] = _gpu.AllocateDevice(kvBytes);
                    _dValCache[l] = _gpu.AllocateDevice(kvBytes);
                }
            }

            // Allocate VRAM token embedding cache (4096 tokens * dim * 4 bytes)
            if (StartLayer == 0)
            {
                nuint cacheBytes = (nuint)(EmbdCacheCapacity * _dim * sizeof(float));
                if (CuDriver.MemAlloc(out _dEmbdCache, cacheBytes) == 0)
                {
                    _embdCacheTokens = new int[EmbdCacheCapacity];
                    Array.Fill(_embdCacheTokens, -1);
                    _embdTokenToSlot = new System.Collections.Generic.Dictionary<int, int>(EmbdCacheCapacity);
                }
                CuDriver.MemHostAlloc(out _hXPinned, (nuint)(_dim * sizeof(float)), 0);
            }

            // 5. Upload model weights into GPU VRAM
            double modelGb = (double)new FileInfo(weights.Gguf.FilePath).Length / (1024 * 1024 * 1024);
            GlacierDiagnostics.LogInformation($">> Uploading {modelGb:F2} GB model weights into {gpu.DeviceName} VRAM (Layers {StartLayer}..{StartLayer + LayerCount - 1})...");
            var sw = Stopwatch.StartNew();

            if (IsLastStage)
            {
                _dOutNormWeight = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
                _gpu.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (nuint)(_dim * sizeof(float)));

                nuint outBytes = (nuint)GgufTypes.GetRowBytes(_weights.OutType, _dim) * (nuint)_weights.VocabSize;
                _dOutWeight = _gpu.AllocateDevice(outBytes);
                _gpu.CopyToDevice(_dOutWeight, (IntPtr)_weights.OutWeight, outBytes);
            }

            for (int l = 0; l < LayerCount; l++)
            {
                int modelLayer = StartLayer + l;
                var lw = _weights.Layers[modelLayer];

                nuint qDim = (nuint)(_nHeads * _headDim);
                nuint kvDim = (nuint)(_nHeadsKv * _headDim);

                nuint qBytes = (nuint)GgufTypes.GetRowBytes(lw.QType, _dim) * qDim;
                nuint kBytes = (nuint)GgufTypes.GetRowBytes(lw.KType, _dim) * kvDim;
                nuint vBytes = (nuint)GgufTypes.GetRowBytes(lw.VType, _dim) * kvDim;
                nuint attnOutBytes = (nuint)GgufTypes.GetRowBytes(lw.AttnOutType, (int)qDim) * (nuint)_dim;

                nuint ffnGateBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnGateType, _dim) * (nuint)_ffnDim;
                nuint ffnUpBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnUpType, _dim) * (nuint)_ffnDim;
                nuint ffnDownBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnDownType, _ffnDim) * (nuint)_dim;

                IntPtr dAttnNorm = IntPtr.Zero;
                IntPtr dQ = IntPtr.Zero;
                IntPtr dQBias = IntPtr.Zero;
                IntPtr dK = IntPtr.Zero;
                IntPtr dKBias = IntPtr.Zero;
                IntPtr dV = IntPtr.Zero;
                IntPtr dVBias = IntPtr.Zero;
                IntPtr dAttnQNorm = IntPtr.Zero;
                IntPtr dAttnKNorm = IntPtr.Zero;
                IntPtr dAttnOut = IntPtr.Zero;
                IntPtr dFfnNorm = IntPtr.Zero;
                IntPtr dFfnGate = IntPtr.Zero;
                IntPtr dFfnUp = IntPtr.Zero;
                IntPtr dFfnDown = IntPtr.Zero;

                try
                {
                    dAttnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
                    _gpu.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (nuint)(_dim * sizeof(float)));

                    dQ = _gpu.AllocateDevice(qBytes);
                    _gpu.CopyToDevice(dQ, (IntPtr)lw.QWeight, qBytes);
                    if (lw.QBias != null)
                    {
                        dQBias = _gpu.AllocateDevice(qDim * sizeof(float));
                        _gpu.CopyToDevice(dQBias, (IntPtr)lw.QBias, qDim * sizeof(float));
                    }

                    dK = _gpu.AllocateDevice(kBytes);
                    _gpu.CopyToDevice(dK, (IntPtr)lw.KWeight, kBytes);
                    if (lw.KBias != null)
                    {
                        dKBias = _gpu.AllocateDevice(kvDim * sizeof(float));
                        _gpu.CopyToDevice(dKBias, (IntPtr)lw.KBias, kvDim * sizeof(float));
                    }

                    dV = _gpu.AllocateDevice(vBytes);
                    _gpu.CopyToDevice(dV, (IntPtr)lw.VWeight, vBytes);
                    if (lw.VBias != null)
                    {
                        dVBias = _gpu.AllocateDevice(kvDim * sizeof(float));
                        _gpu.CopyToDevice(dVBias, (IntPtr)lw.VBias, kvDim * sizeof(float));
                    }

                    if (lw.AttnQNormWeight != null)
                    {
                        nuint qNormBytes = (nuint)(_headDim * sizeof(float));
                        dAttnQNorm = _gpu.AllocateDevice(qNormBytes);
                        _gpu.CopyToDevice(dAttnQNorm, (IntPtr)lw.AttnQNormWeight, qNormBytes);
                    }

                    if (lw.AttnKNormWeight != null)
                    {
                        nuint kNormBytes = (nuint)(_headDim * sizeof(float));
                        dAttnKNorm = _gpu.AllocateDevice(kNormBytes);
                        _gpu.CopyToDevice(dAttnKNorm, (IntPtr)lw.AttnKNormWeight, kNormBytes);
                    }

                    dAttnOut = _gpu.AllocateDevice(attnOutBytes);
                    _gpu.CopyToDevice(dAttnOut, (IntPtr)lw.AttnOutWeight, attnOutBytes);

                    dFfnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
                    _gpu.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (nuint)(_dim * sizeof(float)));

                    dFfnGate = _gpu.AllocateDevice(ffnGateBytes);
                    _gpu.CopyToDevice(dFfnGate, (IntPtr)lw.FfnGateWeight, ffnGateBytes);

                    dFfnUp = _gpu.AllocateDevice(ffnUpBytes);
                    _gpu.CopyToDevice(dFfnUp, (IntPtr)lw.FfnUpWeight, ffnUpBytes);

                    dFfnDown = _gpu.AllocateDevice(ffnDownBytes);
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
                        AttnQNormWeight = dAttnQNorm,
                        AttnKNormWeight = dAttnKNorm,
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
                catch
                {
                    _gpu.FreeDevice(dAttnNorm);
                    _gpu.FreeDevice(dQ);
                    _gpu.FreeDevice(dQBias);
                    _gpu.FreeDevice(dK);
                    _gpu.FreeDevice(dKBias);
                    _gpu.FreeDevice(dV);
                    _gpu.FreeDevice(dVBias);
                    _gpu.FreeDevice(dAttnQNorm);
                    _gpu.FreeDevice(dAttnKNorm);
                    _gpu.FreeDevice(dAttnOut);
                    _gpu.FreeDevice(dFfnNorm);
                    _gpu.FreeDevice(dFfnGate);
                    _gpu.FreeDevice(dFfnUp);
                    _gpu.FreeDevice(dFfnDown);
                    throw;
                }
            }

            sw.Stop();
            GlacierDiagnostics.LogInformation($"   Weights uploaded to GPU VRAM in {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F2} s)!");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_gpu != null && _gpu.Handle != IntPtr.Zero)
            {
                CuDriver.CtxSetCurrent(_gpu.Handle);
            }

            if (_eventNormDone != IntPtr.Zero) { CuDriver.EventDestroy(_eventNormDone); _eventNormDone = IntPtr.Zero; }
            if (_eventKDone != IntPtr.Zero) { CuDriver.EventDestroy(_eventKDone); _eventKDone = IntPtr.Zero; }
            if (_eventVDone != IntPtr.Zero) { CuDriver.EventDestroy(_eventVDone); _eventVDone = IntPtr.Zero; }
            if (_streamK != IntPtr.Zero) { CuDriver.StreamDestroy(_streamK); _streamK = IntPtr.Zero; }
            if (_streamV != IntPtr.Zero) { CuDriver.StreamDestroy(_streamV); _streamV = IntPtr.Zero; }

            if (_gpu != null && _gpu.Handle != IntPtr.Zero)
            {
                _gpu.FreeDevice(_dX); _dX = IntPtr.Zero;
                _gpu.FreeDevice(_dNormX); _dNormX = IntPtr.Zero;
                _gpu.FreeDevice(_dQ); _dQ = IntPtr.Zero;
                _gpu.FreeDevice(_dK); _dK = IntPtr.Zero;
                _gpu.FreeDevice(_dV); _dV = IntPtr.Zero;
                _gpu.FreeDevice(_dAttnOut); _dAttnOut = IntPtr.Zero;
                _gpu.FreeDevice(_dAttnProj); _dAttnProj = IntPtr.Zero;
                _gpu.FreeDevice(_dGate); _dGate = IntPtr.Zero;
                _gpu.FreeDevice(_dUp); _dUp = IntPtr.Zero;
                _gpu.FreeDevice(_dFfnAct); _dFfnAct = IntPtr.Zero;
                _gpu.FreeDevice(_dFfnOut); _dFfnOut = IntPtr.Zero;
                _gpu.FreeDevice(_dLogits); _dLogits = IntPtr.Zero;
                _gpu.FreeDevice(_dBestToken); _dBestToken = IntPtr.Zero;
                _gpu.FreeDevice(_dBestLogit); _dBestLogit = IntPtr.Zero;
                _gpu.FreeDevice(_dRecentTokens); _dRecentTokens = IntPtr.Zero;
                _gpu.FreeDevice(_dOutNormWeight); _dOutNormWeight = IntPtr.Zero;
                _gpu.FreeDevice(_dOutWeight); _dOutWeight = IntPtr.Zero;

                _gpu.FreeDevice(_dXBatch); _dXBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dNormXBatch); _dNormXBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dQBatch); _dQBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dKBatch); _dKBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dVBatch); _dVBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dAttnOutBatch); _dAttnOutBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dAttnProjBatch); _dAttnProjBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dFfnActBatch); _dFfnActBatch = IntPtr.Zero;
                _gpu.FreeDevice(_dFfnOutBatch); _dFfnOutBatch = IntPtr.Zero;

                if (_dEmbdCache != IntPtr.Zero)
                {
                    _gpu.FreeDevice(_dEmbdCache);
                    _dEmbdCache = IntPtr.Zero;
                }

                if (_dKeyCache != null)
                {
                    for (int l = 0; l < _dKeyCache.Length; l++)
                    {
                        if (_dKeyCache[l] != IntPtr.Zero) { _gpu.FreeDevice(_dKeyCache[l]); _dKeyCache[l] = IntPtr.Zero; }
                        if (_dValCache[l] != IntPtr.Zero) { _gpu.FreeDevice(_dValCache[l]); _dValCache[l] = IntPtr.Zero; }
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
                            if (lw.QBias != IntPtr.Zero) _gpu.FreeDevice(lw.QBias);
                            _gpu.FreeDevice(lw.KWeight);
                            if (lw.KBias != IntPtr.Zero) _gpu.FreeDevice(lw.KBias);
                            _gpu.FreeDevice(lw.VWeight);
                            if (lw.VBias != IntPtr.Zero) _gpu.FreeDevice(lw.VBias);
                            _gpu.FreeDevice(lw.AttnOutWeight);
                            if (lw.AttnQNormWeight != IntPtr.Zero) _gpu.FreeDevice(lw.AttnQNormWeight);
                            if (lw.AttnKNormWeight != IntPtr.Zero) _gpu.FreeDevice(lw.AttnKNormWeight);
                            _gpu.FreeDevice(lw.FfnNormWeight);
                            _gpu.FreeDevice(lw.FfnGateWeight);
                            _gpu.FreeDevice(lw.FfnUpWeight);
                            _gpu.FreeDevice(lw.FfnDownWeight);
                            _layerWeights[l] = null!;
                        }
                    }
                }
            }

            if (_hXPinned != IntPtr.Zero)
            {
                CuDriver.MemFreeHost(_hXPinned);
                _hXPinned = IntPtr.Zero;
            }

            if (_module != IntPtr.Zero)
            {
                CuDriver.ModuleUnload(_module);
                _module = IntPtr.Zero;
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
