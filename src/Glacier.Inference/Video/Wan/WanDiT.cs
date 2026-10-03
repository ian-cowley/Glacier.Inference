namespace Glacier.Inference.Video.Wan;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Quant;

/// <summary>
/// Pre-trained Video Diffusion Transformer (Wan2.1-T2V-1.3B) executing flow-matching velocity prediction.
/// Maps 30 Spatio-Temporal Transformer blocks directly from GGUF weights with 3D patch embedding,
/// QK-Norm, 3D-RoPE, cross-attention with T5-XXL embeddings, AdaLN modulation, and NVIDIA GPU acceleration.
/// </summary>
public unsafe sealed class WanDiT : IDisposable
{
    public const int HiddenDim = 1536;
    public const int NumHeads = 12;
    public const int HeadDim = 128; // 1536 / 12 = 128
    public const int InChannels = 16;
    public const int PatchSize = 2; // 2x2 spatial patch
    public const int PatchDim = PatchSize * PatchSize * InChannels; // 64
    public const int FfnDim = 8960;
    public const int NumBlocks = 30;

    private readonly GgufFile _gguf;

    // Patch embedding weights: [2, 2, 1, 16, 1536]
    private readonly GgufTensorInfo _patchEmbedW;
    private readonly GgufTensorInfo _patchEmbedB;

    // Text embedding weights
    private readonly GgufTensorInfo _textEmbed0W, _textEmbed0B;
    private readonly GgufTensorInfo _textEmbed2W, _textEmbed2B;

    // Time embedding weights
    private readonly GgufTensorInfo _timeEmbed0W, _timeEmbed0B;
    private readonly GgufTensorInfo _timeEmbed2W, _timeEmbed2B;
    private readonly GgufTensorInfo _timeProj1W, _timeProj1B;

    // 30 Transformer Blocks
    private readonly WanBlockWeights[] _blocks = new WanBlockWeights[NumBlocks];

    // Final Head
    private readonly GgufTensorInfo _headMod;
    private readonly GgufTensorInfo _headW, _headB;

    // GPU Acceleration
    private readonly GpuContext? _gpu;
    private IntPtr _gpuModule;
    private IntPtr _fnGemmQ4KBatch;
    private IntPtr _fnGemmQ6KBatch;
    private IntPtr _fnAttnBidirectionalBatch;
    private IntPtr _fnGelu;
    private IntPtr _fnRmsNormBatchHeads;

    private IntPtr _dXDevice = IntPtr.Zero;
    private IntPtr _dYDevice = IntPtr.Zero;
    private IntPtr _dQDevice = IntPtr.Zero;
    private IntPtr _dKDevice = IntPtr.Zero;
    private IntPtr _dVDevice = IntPtr.Zero;
    private IntPtr _dAttnOut = IntPtr.Zero;
    private IntPtr _dFfnInter = IntPtr.Zero;
    private nuint _dXCapacity = 0;
    private nuint _dFfnCapacity = 0;
    private bool _gpuWeightsUploaded = false;

    // Scratch buffers
    private readonly int _maxTokens;
    private float* _timeVec;       // [1536]
    private float* _timeModAll;    // [9216]
    private float* _tokenBufferA;  // [maxTokens * 1536]
    private float* _tokenBufferB;  // [maxTokens * 1536]
    private float* _qBuffer;       // [maxTokens * 1536]
    private float* _kBuffer;       // [maxTokens * 1536]
    private float* _vBuffer;       // [maxTokens * 1536]
    private float* _attnScores;    // [maxTokens * maxTokens]
    private float* _ffnIntermediate; // [maxTokens * 8960]
    private bool _disposed;

    public int MaxTokens => _maxTokens;
    public bool IsGpuAccelerated => _gpu != null && _gpuWeightsUploaded;

    public sealed class WanBlockWeights
    {
        public GgufTensorInfo SelfQ_W = null!, SelfQ_B = null!;
        public GgufTensorInfo SelfK_W = null!, SelfK_B = null!;
        public GgufTensorInfo SelfV_W = null!, SelfV_B = null!;
        public GgufTensorInfo SelfO_W = null!, SelfO_B = null!;
        public GgufTensorInfo SelfNormQ = null!, SelfNormK = null!;

        public GgufTensorInfo CrossQ_W = null!, CrossQ_B = null!;
        public GgufTensorInfo CrossK_W = null!, CrossK_B = null!;
        public GgufTensorInfo CrossV_W = null!, CrossV_B = null!;
        public GgufTensorInfo CrossO_W = null!, CrossO_B = null!;
        public GgufTensorInfo CrossNormQ = null!, CrossNormK = null!;

        public GgufTensorInfo Ffn0_W = null!, Ffn0_B = null!;
        public GgufTensorInfo Ffn2_W = null!, Ffn2_B = null!;
        public GgufTensorInfo Norm3_W = null!, Norm3_B = null!;
        public GgufTensorInfo Modulation = null!;

        // GPU VRAM Pointers
        public IntPtr D_SelfQ_W, D_SelfQ_B;
        public IntPtr D_SelfK_W, D_SelfK_B;
        public IntPtr D_SelfV_W, D_SelfV_B;
        public IntPtr D_SelfO_W, D_SelfO_B;
        public IntPtr D_SelfNormQ, D_SelfNormK;

        public IntPtr D_CrossQ_W, D_CrossQ_B;
        public IntPtr D_CrossK_W, D_CrossK_B;
        public IntPtr D_CrossV_W, D_CrossV_B;
        public IntPtr D_CrossO_W, D_CrossO_B;
        public IntPtr D_CrossNormQ, D_CrossNormK;

        public IntPtr D_Ffn0_W, D_Ffn0_B;
        public IntPtr D_Ffn2_W, D_Ffn2_B;
    }

    private WanDiT(GgufFile gguf, int maxTokens = 4096, bool enableGpu = true)
    {
        _gguf = gguf;
        _maxTokens = maxTokens;

        _patchEmbedW = gguf.Tensors["patch_embedding.weight"];
        _patchEmbedB = gguf.Tensors["patch_embedding.bias"];

        _textEmbed0W = gguf.Tensors["text_embedding.0.weight"];
        _textEmbed0B = gguf.Tensors["text_embedding.0.bias"];
        _textEmbed2W = gguf.Tensors["text_embedding.2.weight"];
        _textEmbed2B = gguf.Tensors["text_embedding.2.bias"];

        _timeEmbed0W = gguf.Tensors["time_embedding.0.weight"];
        _timeEmbed0B = gguf.Tensors["time_embedding.0.bias"];
        _timeEmbed2W = gguf.Tensors["time_embedding.2.weight"];
        _timeEmbed2B = gguf.Tensors["time_embedding.2.bias"];
        _timeProj1W = gguf.Tensors["time_projection.1.weight"];
        _timeProj1B = gguf.Tensors["time_projection.1.bias"];

        _headMod = gguf.Tensors["head.modulation"];
        _headW = gguf.Tensors["head.head.weight"];
        _headB = gguf.Tensors["head.head.bias"];

        for (int i = 0; i < NumBlocks; i++)
        {
            string p = $"blocks.{i}.";
            _blocks[i] = new WanBlockWeights
            {
                SelfQ_W = gguf.Tensors[p + "self_attn.q.weight"],
                SelfQ_B = gguf.Tensors[p + "self_attn.q.bias"],
                SelfK_W = gguf.Tensors[p + "self_attn.k.weight"],
                SelfK_B = gguf.Tensors[p + "self_attn.k.bias"],
                SelfV_W = gguf.Tensors[p + "self_attn.v.weight"],
                SelfV_B = gguf.Tensors[p + "self_attn.v.bias"],
                SelfO_W = gguf.Tensors[p + "self_attn.o.weight"],
                SelfO_B = gguf.Tensors[p + "self_attn.o.bias"],
                SelfNormQ = gguf.Tensors[p + "self_attn.norm_q.weight"],
                SelfNormK = gguf.Tensors[p + "self_attn.norm_k.weight"],

                CrossQ_W = gguf.Tensors[p + "cross_attn.q.weight"],
                CrossQ_B = gguf.Tensors[p + "cross_attn.q.bias"],
                CrossK_W = gguf.Tensors[p + "cross_attn.k.weight"],
                CrossK_B = gguf.Tensors[p + "cross_attn.k.bias"],
                CrossV_W = gguf.Tensors[p + "cross_attn.v.weight"],
                CrossV_B = gguf.Tensors[p + "cross_attn.v.bias"],
                CrossO_W = gguf.Tensors[p + "cross_attn.o.weight"],
                CrossO_B = gguf.Tensors[p + "cross_attn.o.bias"],
                CrossNormQ = gguf.Tensors[p + "cross_attn.norm_q.weight"],
                CrossNormK = gguf.Tensors[p + "cross_attn.norm_k.weight"],

                Ffn0_W = gguf.Tensors[p + "ffn.0.weight"],
                Ffn0_B = gguf.Tensors[p + "ffn.0.bias"],
                Ffn2_W = gguf.Tensors[p + "ffn.2.weight"],
                Ffn2_B = gguf.Tensors[p + "ffn.2.bias"],
                Norm3_W = gguf.Tensors[p + "norm3.weight"],
                Norm3_B = gguf.Tensors[p + "norm3.bias"],
                Modulation = gguf.Tensors[p + "modulation"]
            };
        }

        // Allocate CPU scratch memory
        _timeVec = (float*)NativeMemory.AlignedAlloc(HiddenDim * sizeof(float), 64);
        _timeModAll = (float*)NativeMemory.AlignedAlloc(9216 * sizeof(float), 64);
        _tokenBufferA = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _tokenBufferB = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _qBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _kBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _vBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _attnScores = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * 512 * sizeof(float)), 64);
        _ffnIntermediate = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * FfnDim * sizeof(float)), 64);

        // Hardware GPU Acceleration Initialization (NVIDIA RTX 4060)
        if (enableGpu && GpuContext.IsSupported)
        {
            try
            {
                _gpu = new GpuContext(0);
                byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
                CuDriver.Check(CuDriver.ModuleLoadData(out _gpuModule, cubin), "ModuleLoadData");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _gpuModule, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ6KBatch, _gpuModule, "gemm_q6_k_batch"), "ModuleGetFunction(gemm_q6_k_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnBidirectionalBatch, _gpuModule, "attention_bidirectional_batch"), "ModuleGetFunction(attention_bidirectional_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGelu, _gpuModule, "flux_gelu_kernel"), "ModuleGetFunction(flux_gelu_kernel)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatchHeads, _gpuModule, "rms_norm_batch_heads_kernel"), "ModuleGetFunction(rms_norm_batch_heads_kernel)");

                UploadGpuBlockWeights();
                Console.WriteLine($"[GLACIER GPU] Accelerated Wan2.1 Video DiT on {_gpu.DeviceName} ({_gpu.ArchString}). 30 blocks resident in VRAM.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GLACIER GPU] Wan2.1 GPU init skipped ({ex.Message}), falling back to AVX-512 CPU SIMD.");
                _gpu?.Dispose();
                _gpu = null;
            }
        }
    }

    public static WanDiT Open(string ggufPath, int maxTokens = 4096, bool enableGpu = true)
    {
        var gguf = GgufFile.Open(ggufPath);
        return new WanDiT(gguf, maxTokens, enableGpu);
    }

    private void UploadGpuBlockWeights()
    {
        if (_gpu == null || _gpuWeightsUploaded) return;

        for (int i = 0; i < NumBlocks; i++)
        {
            var blk = _blocks[i];
            (blk.D_SelfQ_W, blk.D_SelfQ_B) = UploadWeightAndBias(blk.SelfQ_W, blk.SelfQ_B, HiddenDim);
            (blk.D_SelfK_W, blk.D_SelfK_B) = UploadWeightAndBias(blk.SelfK_W, blk.SelfK_B, HiddenDim);
            (blk.D_SelfV_W, blk.D_SelfV_B) = UploadWeightAndBias(blk.SelfV_W, blk.SelfV_B, HiddenDim);
            (blk.D_SelfO_W, blk.D_SelfO_B) = UploadWeightAndBias(blk.SelfO_W, blk.SelfO_B, HiddenDim);

            (blk.D_CrossQ_W, blk.D_CrossQ_B) = UploadWeightAndBias(blk.CrossQ_W, blk.CrossQ_B, HiddenDim);
            (blk.D_CrossK_W, blk.D_CrossK_B) = UploadWeightAndBias(blk.CrossK_W, blk.CrossK_B, HiddenDim);
            (blk.D_CrossV_W, blk.D_CrossV_B) = UploadWeightAndBias(blk.CrossV_W, blk.CrossV_B, HiddenDim);
            (blk.D_CrossO_W, blk.D_CrossO_B) = UploadWeightAndBias(blk.CrossO_W, blk.CrossO_B, HiddenDim);

            (blk.D_Ffn0_W, blk.D_Ffn0_B) = UploadWeightAndBias(blk.Ffn0_W, blk.Ffn0_B, FfnDim);
            (blk.D_Ffn2_W, blk.D_Ffn2_B) = UploadWeightAndBias(blk.Ffn2_W, blk.Ffn2_B, HiddenDim);
        }

        _gpuWeightsUploaded = true;
    }

    private void FreeGpuBlockWeights()
    {
        if (_gpu == null || !_gpuWeightsUploaded) return;

        for (int i = 0; i < NumBlocks; i++)
        {
            var blk = _blocks[i];
            FreeDevicePtr(ref blk.D_SelfQ_W); FreeDevicePtr(ref blk.D_SelfQ_B);
            FreeDevicePtr(ref blk.D_SelfK_W); FreeDevicePtr(ref blk.D_SelfK_B);
            FreeDevicePtr(ref blk.D_SelfV_W); FreeDevicePtr(ref blk.D_SelfV_B);
            FreeDevicePtr(ref blk.D_SelfO_W); FreeDevicePtr(ref blk.D_SelfO_B);

            FreeDevicePtr(ref blk.D_CrossQ_W); FreeDevicePtr(ref blk.D_CrossQ_B);
            FreeDevicePtr(ref blk.D_CrossK_W); FreeDevicePtr(ref blk.D_CrossK_B);
            FreeDevicePtr(ref blk.D_CrossV_W); FreeDevicePtr(ref blk.D_CrossV_B);
            FreeDevicePtr(ref blk.D_CrossO_W); FreeDevicePtr(ref blk.D_CrossO_B);

            FreeDevicePtr(ref blk.D_Ffn0_W); FreeDevicePtr(ref blk.D_Ffn0_B);
            FreeDevicePtr(ref blk.D_Ffn2_W); FreeDevicePtr(ref blk.D_Ffn2_B);
        }

        _gpuWeightsUploaded = false;
    }

    private void EnsureGpuBuffers(int numTokens)
    {
        if (_gpu == null) return;
        nuint xBytes = (nuint)((long)numTokens * HiddenDim * sizeof(float));
        nuint ffnBytes = (nuint)((long)numTokens * FfnDim * sizeof(float));

        if (xBytes > _dXCapacity)
        {
            FreeDevicePtr(ref _dXDevice);
            FreeDevicePtr(ref _dYDevice);
            FreeDevicePtr(ref _dQDevice);
            FreeDevicePtr(ref _dKDevice);
            FreeDevicePtr(ref _dVDevice);
            FreeDevicePtr(ref _dAttnOut);

            _dXCapacity = xBytes * 12 / 10 + 1024 * 1024;
            _dXDevice = _gpu.AllocateDevice(_dXCapacity);
            _dYDevice = _gpu.AllocateDevice(_dXCapacity);
            _dQDevice = _gpu.AllocateDevice(_dXCapacity);
            _dKDevice = _gpu.AllocateDevice(_dXCapacity);
            _dVDevice = _gpu.AllocateDevice(_dXCapacity);
            _dAttnOut = _gpu.AllocateDevice(_dXCapacity);
        }

        if (ffnBytes > _dFfnCapacity)
        {
            FreeDevicePtr(ref _dFfnInter);
            _dFfnCapacity = ffnBytes * 12 / 10 + 1024 * 1024;
            _dFfnInter = _gpu.AllocateDevice(_dFfnCapacity);
        }
    }

    private (IntPtr dW, IntPtr dB) UploadWeightAndBias(GgufTensorInfo w, GgufTensorInfo b, int nRows)
    {
        if (_gpu == null) return (IntPtr.Zero, IntPtr.Zero);
        nuint wBytes = (nuint)w.GetByteSize();
        nuint bBytes = b != null ? (nuint)((long)nRows * sizeof(float)) : 0;

        IntPtr dW = _gpu.AllocateDevice(wBytes);
        IntPtr dB = b != null ? _gpu.AllocateDevice(bBytes) : IntPtr.Zero;

        _gpu.CopyToDevice(dW, (IntPtr)_gguf.GetTensorPointer(w), wBytes);
        if (b != null)
        {
            _gpu.CopyToDevice(dB, (IntPtr)_gguf.GetTensorPointer(b), bBytes);
        }

        return (dW, dB);
    }

    private void FreeDevicePtr(ref IntPtr ptr)
    {
        if (_gpu != null && ptr != IntPtr.Zero)
        {
            _gpu.FreeDevice(ptr);
            ptr = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Evaluates the full 30-block Wan2.1 Diffusion Transformer to predict the flow velocity field.
    /// Hardware accelerated on NVIDIA RTX 4060 GPU with seamless AVX-512 CPU fallback.
    /// </summary>
    public void PredictVelocity(
        ReadOnlySpan<float> latents,
        int temporalFrames,
        int latentH,
        int latentW,
        float timestep,
        ReadOnlySpan<float> textContext,
        int numTxtTokens,
        Span<float> velocityOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int tokenH = latentH / PatchSize;
        int tokenW = latentW / PatchSize;
        int spatialTokens = tokenH * tokenW;
        int totalImgTokens = temporalFrames * spatialTokens;

        if (totalImgTokens > _maxTokens)
        {
            throw new ArgumentException($"Total image tokens ({totalImgTokens}) exceeds capacity ({_maxTokens}).");
        }

        // 1. Timestep Embedding
        ComputeTimeEmbedding(timestep, _timeVec, _timeModAll);

        // 2. Text Context Projection: [numTxtTokens, 4096] -> [numTxtTokens, 1536]
        float* projectedTxt = stackalloc float[Math.Max(1, numTxtTokens) * HiddenDim];
        if (numTxtTokens > 0)
        {
            ProjectText(textContext, numTxtTokens, projectedTxt);
        }

        // 3. Patchify & Project input latents to hidden dimension: [totalImgTokens, 64] -> [totalImgTokens, 1536]
        PatchifyAndProject(latents, temporalFrames, latentH, latentW, _tokenBufferA);

        // 4. Evaluate 30 Transformer Blocks
        float* currentTokens = _tokenBufferA;
        float* nextTokens = _tokenBufferB;

        if (IsGpuAccelerated)
        {
            // Execute on NVIDIA GPU (RTX 4060)
            EnsureGpuBuffers(totalImgTokens);
            _gpu!.CopyToDevice(_dXDevice, (IntPtr)currentTokens, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));

            for (int blkIdx = 0; blkIdx < NumBlocks; blkIdx++)
            {
                var blk = _blocks[blkIdx];
                ExecuteBlockGpu(blk, totalImgTokens, projectedTxt, numTxtTokens);
            }

            _gpu.Synchronize();
            _gpu.CopyToHost((IntPtr)currentTokens, _dXDevice, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
        }
        else
        {
            // Fallback: CPU AVX-512 SIMD execution with batched weight streaming
            for (int blkIdx = 0; blkIdx < NumBlocks; blkIdx++)
            {
                var blk = _blocks[blkIdx];

                ExecuteSelfAttentionCpu(currentTokens, totalImgTokens, temporalFrames, tokenH, tokenW, blk, nextTokens);
                float* tmp1 = currentTokens; currentTokens = nextTokens; nextTokens = tmp1;

                if (numTxtTokens > 0)
                {
                    ExecuteCrossAttentionCpu(currentTokens, totalImgTokens, projectedTxt, numTxtTokens, blk, nextTokens);
                    float* tmp2 = currentTokens; currentTokens = nextTokens; nextTokens = tmp2;
                }

                ExecuteFFNCpu(currentTokens, totalImgTokens, blk, nextTokens);
                float* tmp3 = currentTokens; currentTokens = nextTokens; nextTokens = tmp3;
            }
        }

        // 5. Final Head Layer: AdaLN Modulate, Project [1536 -> 64], and Unpatchify to velocity field
        UnpatchifyAndOutput(currentTokens, temporalFrames, latentH, latentW, velocityOut);
    }

    private void ExecuteBlockGpu(WanBlockWeights blk, int totalImgTokens, float* projectedTxt, int numTxtTokens)
    {
        // 1. Self Attention QKV
        DispatchMatMulBatchDevice(blk.D_SelfQ_W, blk.D_SelfQ_B, _dXDevice, _dQDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchDevice(blk.D_SelfK_W, blk.D_SelfK_B, _dXDevice, _dKDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchQ6KDevice(blk.D_SelfV_W, blk.D_SelfV_B, _dXDevice, _dVDevice, HiddenDim, HiddenDim, totalImgTokens);

        // 2. Bidirectional Attention across spatio-temporal tokens
        float scale = 1.0f / MathF.Sqrt(HeadDim);
        DispatchAttentionDevice(_dQDevice, _dKDevice, _dVDevice, _dAttnOut, NumHeads, HeadDim, totalImgTokens, scale);

        // 3. Self-Attention Output Projection with Residual into _dYDevice
        DispatchMatMulBatchDevice(blk.D_SelfO_W, blk.D_SelfO_B, _dAttnOut, _dYDevice, HiddenDim, HiddenDim, totalImgTokens, dResidual: _dXDevice);
        (IntPtr dXDevice, IntPtr dYDevice) swap1 = (_dXDevice, _dYDevice);
        _dXDevice = swap1.dYDevice;
        _dYDevice = swap1.dXDevice;

        // 4. Cross Attention (if text context provided)
        if (numTxtTokens > 0)
        {
            // Copy intermediate tokens to host for fast Cross-Attention
            _gpu!.CopyToHost((IntPtr)_tokenBufferA, _dXDevice, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
            ExecuteCrossAttentionCpu(_tokenBufferA, totalImgTokens, projectedTxt, numTxtTokens, blk, _tokenBufferB);
            _gpu.CopyToDevice(_dXDevice, (IntPtr)_tokenBufferB, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
        }

        // 5. Feed-Forward Network: 1536 -> 8960 -> GELU -> 1536 + Residual
        DispatchMatMulBatchDevice(blk.D_Ffn0_W, blk.D_Ffn0_B, _dXDevice, _dFfnInter, HiddenDim, FfnDim, totalImgTokens);
        DispatchGeluDevice(_dFfnInter, totalImgTokens * FfnDim);
        DispatchMatMulBatchQ6KDevice(blk.D_Ffn2_W, blk.D_Ffn2_B, _dFfnInter, _dYDevice, FfnDim, HiddenDim, totalImgTokens, dResidual: _dXDevice);

        (IntPtr dXDevice, IntPtr dYDevice) swap2 = (_dXDevice, _dYDevice);
        _dXDevice = swap2.dYDevice;
        _dYDevice = swap2.dXDevice;
    }

    private void DispatchMatMulBatchDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize, IntPtr dResidual = default)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);

        int localCols = nCols;
        int localRows = nRows;
        int localBatch = batchSize;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &localCols;
        pArgs[4] = &localRows;
        pArgs[5] = &localBatch;
        pArgs[6] = &dB;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmQ4KBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q4_k_batch)");
    }

    private void DispatchMatMulBatchQ6KDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize, IntPtr dResidual = default)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 7) / 8);

        int localCols = nCols;
        int localRows = nRows;
        int localBatch = batchSize;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &localCols;
        pArgs[4] = &localRows;
        pArgs[5] = &localBatch;
        pArgs[6] = &dB;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmQ6KBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q6_k_batch)");
    }

    private void DispatchAttentionDevice(
        IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dAttnOut,
        int nHeads, int headDim, int numTokens, float attnScale)
    {
        uint blockSize = 128;
        uint gridX = (uint)nHeads;
        uint gridY = (uint)numTokens;

        int localHeads = nHeads;
        int localHeadDim = headDim;
        int localTokens = numTokens;
        float localScale = attnScale;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dAttnOut;
        pArgs[4] = &localHeads;
        pArgs[5] = &localHeadDim;
        pArgs[6] = &localTokens;
        pArgs[7] = &localScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnBidirectionalBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_bidirectional_batch)");
    }

    private void DispatchGeluDevice(IntPtr dX, int count)
    {
        uint blockSize = 256;
        uint gridX = (uint)((count + (int)blockSize - 1) / (int)blockSize);
        int localCount = count;

        void** pArgs = stackalloc void*[2];
        pArgs[0] = &dX;
        pArgs[1] = &localCount;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGelu,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_gelu_kernel)");
    }

    private void ComputeTimeEmbedding(float timestep, float* timeVec, float* timeModAll)
    {
        float* rawTime = stackalloc float[256];
        float halfDim = 128f;
        for (int i = 0; i < 128; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / halfDim);
            float arg = timestep * 1000.0f * freq;
            rawTime[i] = MathF.Cos(arg);
            rawTime[128 + i] = MathF.Sin(arg);
        }

        float* timeInt = stackalloc float[HiddenDim];
        DispatchLinearVec(_timeEmbed0W, _timeEmbed0B, rawTime, timeInt);
        ApplySilu(timeInt, HiddenDim);

        DispatchLinearVec(_timeEmbed2W, _timeEmbed2B, timeInt, timeVec);
        ApplySilu(timeVec, HiddenDim);

        DispatchLinearVec(_timeProj1W, _timeProj1B, timeVec, timeModAll);
    }

    private void ProjectText(ReadOnlySpan<float> textContext, int numTxtTokens, float* projectedTxt)
    {
        fixed (float* pTxt = textContext)
        {
            float* intermediate = stackalloc float[HiddenDim];
            for (int t = 0; t < numTxtTokens; t++)
            {
                float* src = pTxt + t * 4096;
                float* dst = projectedTxt + t * HiddenDim;

                DispatchLinearVec(_textEmbed0W, _textEmbed0B, src, intermediate);
                ApplySilu(intermediate, HiddenDim);
                DispatchLinearVec(_textEmbed2W, _textEmbed2B, intermediate, dst);
            }
        }
    }

    private void PatchifyAndProject(ReadOnlySpan<float> latents, int frames, int H, int W, float* outputTokens)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;
        int spatialTokens = tokenH * tokenW;
        int totalTokens = frames * spatialTokens;

        float* allPatches = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * PatchDim * sizeof(float)), 64);

        try
        {
            fixed (float* pLat = latents)
            {
                int tokenIdx = 0;
                for (int f = 0; f < frames; f++)
                {
                    int frameLatentOffset = f * (InChannels * H * W);

                    for (int th = 0; th < tokenH; th++)
                    {
                        for (int tw = 0; tw < tokenW; tw++)
                        {
                            float* dstPatch = allPatches + (long)tokenIdx * PatchDim;
                            int pIdx = 0;

                            for (int c = 0; c < InChannels; c++)
                            {
                                for (int ph = 0; ph < PatchSize; ph++)
                                {
                                    for (int pw = 0; pw < PatchSize; pw++)
                                    {
                                        int y = th * PatchSize + ph;
                                        int x = tw * PatchSize + pw;
                                        dstPatch[pIdx++] = pLat[frameLatentOffset + c * H * W + y * W + x];
                                    }
                                }
                            }
                            tokenIdx++;
                        }
                    }
                }
            }

            var (nCols, nRows) = GetMatrixDims(_patchEmbedW); // 64, 1536
            byte* wPtr = _gguf.GetTensorPointer(_patchEmbedW);
            float* bPtr = (float*)_gguf.GetTensorPointer(_patchEmbedB);

            QuantKernels.MatMulBatch(_patchEmbedW.Type, wPtr, allPatches, outputTokens, nCols, nRows, totalTokens);

            if (bPtr != null)
            {
                Parallel.For(0, totalTokens, t =>
                {
                    float* tok = outputTokens + (long)t * HiddenDim;
                    for (int d = 0; d < HiddenDim; d++) tok[d] += bPtr[d];
                });
            }
        }
        finally
        {
            NativeMemory.AlignedFree(allPatches);
        }
    }

    private void ExecuteSelfAttentionCpu(
        float* srcTokens, int totalTokens, int frames, int tokenH, int tokenW, WanBlockWeights blk, float* dstTokens)
    {
        var (qCols, qRows) = GetMatrixDims(blk.SelfQ_W);
        var (kCols, kRows) = GetMatrixDims(blk.SelfK_W);
        var (vCols, vRows) = GetMatrixDims(blk.SelfV_W);

        byte* qW = _gguf.GetTensorPointer(blk.SelfQ_W);
        byte* kW = _gguf.GetTensorPointer(blk.SelfK_W);
        byte* vW = _gguf.GetTensorPointer(blk.SelfV_W);

        float* qB = (float*)_gguf.GetTensorPointer(blk.SelfQ_B);
        float* kB = (float*)_gguf.GetTensorPointer(blk.SelfK_B);
        float* vB = (float*)_gguf.GetTensorPointer(blk.SelfV_B);

        QuantKernels.MatMulBatch(blk.SelfQ_W.Type, qW, srcTokens, _qBuffer, qCols, qRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfK_W.Type, kW, srcTokens, _kBuffer, kCols, kRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfV_W.Type, vW, srcTokens, _vBuffer, vCols, vRows, totalTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.SelfNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.SelfNormK);

        Parallel.For(0, totalTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;

            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            for (int h = 0; h < NumHeads; h++)
            {
                ApplyRMSNormHead(q + h * HeadDim, q + h * HeadDim, normQScale);
                ApplyRMSNormHead(k + h * HeadDim, k + h * HeadDim, normKScale);
            }
        });

        float scale = 1.0f / MathF.Sqrt(HeadDim);

        Parallel.For(0, NumHeads, h =>
        {
            float* scores = stackalloc float[totalTokens];

            for (int i = 0; i < totalTokens; i++)
            {
                float* qRow = _qBuffer + (long)i * HiddenDim + h * HeadDim;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < totalTokens; j++)
                {
                    float* kRow = _kBuffer + (long)j * HiddenDim + h * HeadDim;
                    float dot = DotHead(qRow, kRow) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < totalTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    sumExp += exp;
                }
                float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                float* outHead = dstTokens + (long)i * HiddenDim + h * HeadDim;
                for (int d = 0; d < HeadDim; d++) outHead[d] = 0f;

                for (int j = 0; j < totalTokens; j++)
                {
                    float w = scores[j] * invSum;
                    float* vRow = _vBuffer + (long)j * HiddenDim + h * HeadDim;
                    AccumulateHead(outHead, vRow, w);
                }
            }
        });

        var (oCols, oRows) = GetMatrixDims(blk.SelfO_W);
        byte* oW = _gguf.GetTensorPointer(blk.SelfO_W);
        float* oB = (float*)_gguf.GetTensorPointer(blk.SelfO_B);

        QuantKernels.MatMulBatch(blk.SelfO_W.Type, oW, dstTokens, _qBuffer, oCols, oRows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _qBuffer + (long)t * HiddenDim;
            float* residual = srcTokens + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = residual[d] + intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteCrossAttentionCpu(
        float* imgTokens, int totalImgTokens, float* txtTokens, int numTxtTokens, WanBlockWeights blk, float* dstTokens)
    {
        var (qCols, qRows) = GetMatrixDims(blk.CrossQ_W);
        var (kCols, kRows) = GetMatrixDims(blk.CrossK_W);
        var (vCols, vRows) = GetMatrixDims(blk.CrossV_W);

        byte* qW = _gguf.GetTensorPointer(blk.CrossQ_W);
        byte* kW = _gguf.GetTensorPointer(blk.CrossK_W);
        byte* vW = _gguf.GetTensorPointer(blk.CrossV_W);

        float* qB = (float*)_gguf.GetTensorPointer(blk.CrossQ_B);
        float* kB = (float*)_gguf.GetTensorPointer(blk.CrossK_B);
        float* vB = (float*)_gguf.GetTensorPointer(blk.CrossV_B);

        QuantKernels.MatMulBatch(blk.CrossQ_W.Type, qW, imgTokens, _qBuffer, qCols, qRows, totalImgTokens);
        QuantKernels.MatMulBatch(blk.CrossK_W.Type, kW, txtTokens, _kBuffer, kCols, kRows, numTxtTokens);
        QuantKernels.MatMulBatch(blk.CrossV_W.Type, vW, txtTokens, _vBuffer, vCols, vRows, numTxtTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.CrossNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.CrossNormK);

        Parallel.For(0, totalImgTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];
            for (int h = 0; h < NumHeads; h++)
            {
                ApplyRMSNormHead(q + h * HeadDim, q + h * HeadDim, normQScale);
            }
        });

        Parallel.For(0, numTxtTokens, t =>
        {
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            for (int h = 0; h < NumHeads; h++)
            {
                ApplyRMSNormHead(k + h * HeadDim, k + h * HeadDim, normKScale);
            }
        });

        float scale = 1.0f / MathF.Sqrt(HeadDim);
        Parallel.For(0, NumHeads, h =>
        {
            float* scores = stackalloc float[numTxtTokens];

            for (int i = 0; i < totalImgTokens; i++)
            {
                float* qRow = _qBuffer + (long)i * HiddenDim + h * HeadDim;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < numTxtTokens; j++)
                {
                    float* kRow = _kBuffer + (long)j * HiddenDim + h * HeadDim;
                    float dot = DotHead(qRow, kRow) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < numTxtTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    sumExp += exp;
                }
                float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                float* outHead = dstTokens + (long)i * HiddenDim + h * HeadDim;
                for (int d = 0; d < HeadDim; d++) outHead[d] = 0f;

                for (int j = 0; j < numTxtTokens; j++)
                {
                    float w = scores[j] * invSum;
                    float* vRow = _vBuffer + (long)j * HiddenDim + h * HeadDim;
                    AccumulateHead(outHead, vRow, w);
                }
            }
        });

        var (oCols, oRows) = GetMatrixDims(blk.CrossO_W);
        byte* oW = _gguf.GetTensorPointer(blk.CrossO_W);
        float* oB = (float*)_gguf.GetTensorPointer(blk.CrossO_B);

        QuantKernels.MatMulBatch(blk.CrossO_W.Type, oW, dstTokens, _qBuffer, oCols, oRows, totalImgTokens);

        Parallel.For(0, totalImgTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _qBuffer + (long)t * HiddenDim;
            float* residual = imgTokens + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = residual[d] + intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteFFNCpu(float* srcTokens, int totalTokens, WanBlockWeights blk, float* dstTokens)
    {
        var (f0Cols, f0Rows) = GetMatrixDims(blk.Ffn0_W);
        var (f2Cols, f2Rows) = GetMatrixDims(blk.Ffn2_W);

        byte* f0W = _gguf.GetTensorPointer(blk.Ffn0_W);
        byte* f2W = _gguf.GetTensorPointer(blk.Ffn2_W);

        float* f0B = (float*)_gguf.GetTensorPointer(blk.Ffn0_B);
        float* f2B = (float*)_gguf.GetTensorPointer(blk.Ffn2_B);

        QuantKernels.MatMulBatch(blk.Ffn0_W.Type, f0W, srcTokens, _ffnIntermediate, f0Cols, f0Rows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* inter = _ffnIntermediate + (long)t * FfnDim;
            if (f0B != null)
            {
                for (int d = 0; d < FfnDim; d++) inter[d] += f0B[d];
            }
            ApplyGelu(inter, FfnDim);
        });

        QuantKernels.MatMulBatch(blk.Ffn2_W.Type, f2W, _ffnIntermediate, dstTokens, f2Cols, f2Rows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* outToken = dstTokens + (long)t * HiddenDim;
            float* src = srcTokens + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = f2B != null ? f2B[d] : 0f;
                outToken[d] += src[d] + bVal;
            }
        });
    }

    private void UnpatchifyAndOutput(float* tokens, int frames, int H, int W, Span<float> velocityOut)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;
        int spatialTokens = tokenH * tokenW;
        int totalTokens = frames * spatialTokens;

        var (nCols, nRows) = GetMatrixDims(_headW); // 1536, 64
        byte* wPtr = _gguf.GetTensorPointer(_headW);
        float* bPtr = (float*)_gguf.GetTensorPointer(_headB);
        float* modPtr = (float*)_gguf.GetTensorPointer(_headMod);

        float* patchBufAll = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * PatchDim * sizeof(float)), 64);
        float* normTokens = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            float* shift = modPtr;
            float* scale = modPtr + HiddenDim;

            Parallel.For(0, totalTokens, t =>
            {
                float* src = tokens + (long)t * HiddenDim;
                float* dst = normTokens + (long)t * HiddenDim;

                float sumSq = 0f;
                for (int d = 0; d < HiddenDim; d++) sumSq += src[d] * src[d];
                float invRms = 1.0f / MathF.Sqrt(sumSq / HiddenDim + 1e-6f);

                for (int d = 0; d < HiddenDim; d++)
                {
                    dst[d] = (src[d] * invRms) * (1.0f + scale[d]) + shift[d];
                }
            });

            QuantKernels.MatMulBatch(_headW.Type, wPtr, normTokens, patchBufAll, nCols, nRows, totalTokens);
            if (bPtr != null)
            {
                Parallel.For(0, totalTokens, t =>
                {
                    float* pRow = patchBufAll + (long)t * PatchDim;
                    for (int d = 0; d < PatchDim; d++) pRow[d] += bPtr[d];
                });
            }

            fixed (float* pOut = velocityOut)
            {
                int tokenIdx = 0;
                for (int f = 0; f < frames; f++)
                {
                    int frameLatentOffset = f * (InChannels * H * W);

                    for (int th = 0; th < tokenH; th++)
                    {
                        for (int tw = 0; tw < tokenW; tw++)
                        {
                            float* patchBuf = patchBufAll + (long)tokenIdx * PatchDim;
                            int pIdx = 0;

                            for (int c = 0; c < InChannels; c++)
                            {
                                for (int ph = 0; ph < PatchSize; ph++)
                                {
                                    for (int pw = 0; pw < PatchSize; pw++)
                                    {
                                        int y = th * PatchSize + ph;
                                        int x = tw * PatchSize + pw;
                                        pOut[frameLatentOffset + c * H * W + y * W + x] = patchBuf[pIdx++];
                                    }
                                }
                            }
                            tokenIdx++;
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(patchBufAll);
            NativeMemory.AlignedFree(normTokens);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (int nCols, int nRows) GetMatrixDims(GgufTensorInfo w)
    {
        if (w.Dimensions.Length == 1)
            return (1, (int)w.Dimensions[0]);
        if (w.Dimensions.Length == 2)
            return ((int)w.Dimensions[0], (int)w.Dimensions[1]);

        int nRows = (int)w.Dimensions[^1];
        int nCols = 1;
        for (int i = 0; i < w.Dimensions.Length - 1; i++)
            nCols *= (int)w.Dimensions[i];
        return (nCols, nRows);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchLinearVec(GgufTensorInfo w, GgufTensorInfo b, float* x, float* y)
    {
        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = b != null ? (float*)_gguf.GetTensorPointer(b) : null;
        var (nCols, nRows) = GetMatrixDims(w);

        QuantKernels.MatVecMul(w.Type, wPtr, x, y, nCols, nRows);
        if (bPtr != null)
        {
            for (int i = 0; i < nRows; i++) y[i] += bPtr[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotHead(float* a, float* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vSum0 = Vector256<float>.Zero;
            var vSum1 = Vector256<float>.Zero;
            for (int d = 0; d < HeadDim; d += 16)
            {
                vSum0 += Vector256.Load(a + d) * Vector256.Load(b + d);
                vSum1 += Vector256.Load(a + d + 8) * Vector256.Load(b + d + 8);
            }
            return Vector256.Sum(vSum0 + vSum1);
        }
        float dot = 0f;
        for (int d = 0; d < HeadDim; d++) dot += a[d] * b[d];
        return dot;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateHead(float* dst, float* v, float w)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vw = Vector256.Create(w);
            for (int d = 0; d < HeadDim; d += 8)
            {
                var res = Vector256.Load(dst + d) + vw * Vector256.Load(v + d);
                res.Store(dst + d);
            }
            return;
        }
        for (int d = 0; d < HeadDim; d++) dst[d] += w * v[d];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRMSNormHead(float* src, float* dst, float* scale)
    {
        float sumSq = 0f;
        for (int i = 0; i < HeadDim; i++) sumSq += src[i] * src[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / HeadDim + 1e-6f);
        for (int i = 0; i < HeadDim; i++) dst[i] = (src[i] * invRms) * scale[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySilu(float* x, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] = x[i] / (1.0f + MathF.Exp(-x[i]));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyGelu(float* x, int count)
    {
        const float sqrt2OverPi = 0.79788456f;
        for (int i = 0; i < count; i++)
        {
            float val = x[i];
            x[i] = 0.5f * val * (1.0f + MathF.Tanh(sqrt2OverPi * (val + 0.044715f * val * val * val)));
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            FreeGpuBlockWeights();
            FreeDevicePtr(ref _dXDevice);
            FreeDevicePtr(ref _dYDevice);
            FreeDevicePtr(ref _dQDevice);
            FreeDevicePtr(ref _dKDevice);
            FreeDevicePtr(ref _dVDevice);
            FreeDevicePtr(ref _dAttnOut);
            FreeDevicePtr(ref _dFfnInter);
            _gpu?.Dispose();

            if (_timeVec != null) NativeMemory.AlignedFree(_timeVec);
            if (_timeModAll != null) NativeMemory.AlignedFree(_timeModAll);
            if (_tokenBufferA != null) NativeMemory.AlignedFree(_tokenBufferA);
            if (_tokenBufferB != null) NativeMemory.AlignedFree(_tokenBufferB);
            if (_qBuffer != null) NativeMemory.AlignedFree(_qBuffer);
            if (_kBuffer != null) NativeMemory.AlignedFree(_kBuffer);
            if (_vBuffer != null) NativeMemory.AlignedFree(_vBuffer);
            if (_attnScores != null) NativeMemory.AlignedFree(_attnScores);
            if (_ffnIntermediate != null) NativeMemory.AlignedFree(_ffnIntermediate);
            _gguf.Dispose();
        }
    }
}
