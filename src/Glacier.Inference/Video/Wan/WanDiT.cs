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
/// across-head QK-Norm, 3D-RoPE, cross-attention with T5-XXL embeddings, AdaLN modulation,
/// and NVIDIA bare-metal GPU acceleration.
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
    private IntPtr _fnAttnCrossBatch;
    private IntPtr _fnGelu;
    private IntPtr _fnRmsNormBatch;
    private IntPtr _fnLayerNormAffine;
    private IntPtr _fnAdaLn;
    private IntPtr _fnResidualGated;
    private IntPtr _fnVecAdd;

    private IntPtr _dXDevice = IntPtr.Zero;
    private IntPtr _dYDevice = IntPtr.Zero;
    private IntPtr _dQDevice = IntPtr.Zero;
    private IntPtr _dKDevice = IntPtr.Zero;
    private IntPtr _dVDevice = IntPtr.Zero;
    private IntPtr _dAttnOut = IntPtr.Zero;
    private IntPtr _dFfnInter = IntPtr.Zero;
    private IntPtr _dTxtDevice = IntPtr.Zero;

    private IntPtr _dShiftMsa = IntPtr.Zero;
    private IntPtr _dScaleMsa = IntPtr.Zero;
    private IntPtr _dGateMsa = IntPtr.Zero;
    private IntPtr _dCShiftMsa = IntPtr.Zero;
    private IntPtr _dCScaleMsa = IntPtr.Zero;
    private IntPtr _dCGateMsa = IntPtr.Zero;

    private nuint _dXCapacity = 0;
    private nuint _dFfnCapacity = 0;
    private nuint _dTxtCapacity = 0;
    private bool _gpuWeightsUploaded = false;

    // Scratch buffers
    private readonly int _maxTokens;
    private float* _timeVec;          // [1536]
    private float* _timeModAll;       // [9216]
    private float* _tokenBufferA;     // [maxTokens * 1536]
    private float* _tokenBufferB;     // [maxTokens * 1536]
    private float* _tokenBufferC;     // [maxTokens * 1536]
    private float* _qBuffer;          // [maxTokens * 1536]
    private float* _kBuffer;          // [maxTokens * 1536]
    private float* _vBuffer;          // [maxTokens * 1536]
    private float* _ffnIntermediate;  // [maxTokens * 8960]
    private float* _ropeCos;          // [maxTokens * 64]
    private float* _ropeSin;          // [maxTokens * 64]
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
        public IntPtr D_Norm3_W, D_Norm3_B;
    }

    private WanDiT(GgufFile gguf, int maxTokens = 16384, bool enableGpu = true)
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
        _tokenBufferC = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _qBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _kBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _vBuffer = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * HiddenDim * sizeof(float)), 64);
        _ffnIntermediate = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * FfnDim * sizeof(float)), 64);
        _ropeCos = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * 64 * sizeof(float)), 64);
        _ropeSin = (float*)NativeMemory.AlignedAlloc((nuint)(_maxTokens * 64 * sizeof(float)), 64);

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
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnCrossBatch, _gpuModule, "attention_cross_batch"), "ModuleGetFunction(attention_cross_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGelu, _gpuModule, "flux_gelu_kernel"), "ModuleGetFunction(flux_gelu_kernel)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNormBatch, _gpuModule, "rms_norm_batch"), "ModuleGetFunction(rms_norm_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnLayerNormAffine, _gpuModule, "layer_norm_affine_kernel"), "ModuleGetFunction(layer_norm_affine_kernel)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAdaLn, _gpuModule, "flux_adaln_kernel"), "ModuleGetFunction(flux_adaln_kernel)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnResidualGated, _gpuModule, "flux_residual_gated"), "ModuleGetFunction(flux_residual_gated)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _gpuModule, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");

                _dShiftMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dScaleMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dGateMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCShiftMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCScaleMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCGateMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));

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

    public static WanDiT Open(string ggufPath, int maxTokens = 16384, bool enableGpu = true)
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
            (blk.D_SelfNormQ, _) = UploadWeightAndBias(blk.SelfNormQ, null, HiddenDim);
            (blk.D_SelfNormK, _) = UploadWeightAndBias(blk.SelfNormK, null, HiddenDim);

            (blk.D_CrossQ_W, blk.D_CrossQ_B) = UploadWeightAndBias(blk.CrossQ_W, blk.CrossQ_B, HiddenDim);
            (blk.D_CrossK_W, blk.D_CrossK_B) = UploadWeightAndBias(blk.CrossK_W, blk.CrossK_B, HiddenDim);
            (blk.D_CrossV_W, blk.D_CrossV_B) = UploadWeightAndBias(blk.CrossV_W, blk.CrossV_B, HiddenDim);
            (blk.D_CrossO_W, blk.D_CrossO_B) = UploadWeightAndBias(blk.CrossO_W, blk.CrossO_B, HiddenDim);
            (blk.D_CrossNormQ, _) = UploadWeightAndBias(blk.CrossNormQ, null, HiddenDim);
            (blk.D_CrossNormK, _) = UploadWeightAndBias(blk.CrossNormK, null, HiddenDim);

            (blk.D_Ffn0_W, blk.D_Ffn0_B) = UploadWeightAndBias(blk.Ffn0_W, blk.Ffn0_B, FfnDim);
            (blk.D_Ffn2_W, blk.D_Ffn2_B) = UploadWeightAndBias(blk.Ffn2_W, blk.Ffn2_B, HiddenDim);
            (blk.D_Norm3_W, blk.D_Norm3_B) = UploadWeightAndBias(blk.Norm3_W, blk.Norm3_B, HiddenDim);
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
            FreeDevicePtr(ref blk.D_SelfNormQ); FreeDevicePtr(ref blk.D_SelfNormK);

            FreeDevicePtr(ref blk.D_CrossQ_W); FreeDevicePtr(ref blk.D_CrossQ_B);
            FreeDevicePtr(ref blk.D_CrossK_W); FreeDevicePtr(ref blk.D_CrossK_B);
            FreeDevicePtr(ref blk.D_CrossV_W); FreeDevicePtr(ref blk.D_CrossV_B);
            FreeDevicePtr(ref blk.D_CrossO_W); FreeDevicePtr(ref blk.D_CrossO_B);
            FreeDevicePtr(ref blk.D_CrossNormQ); FreeDevicePtr(ref blk.D_CrossNormK);

            FreeDevicePtr(ref blk.D_Ffn0_W); FreeDevicePtr(ref blk.D_Ffn0_B);
            FreeDevicePtr(ref blk.D_Ffn2_W); FreeDevicePtr(ref blk.D_Ffn2_B);
            FreeDevicePtr(ref blk.D_Norm3_W); FreeDevicePtr(ref blk.D_Norm3_B);
        }

        _gpuWeightsUploaded = false;
    }

    private void EnsureGpuBuffers(int numTokens, int numTxtTokens = 0)
    {
        if (_gpu == null) return;
        int maxTokens = Math.Max(numTokens, numTxtTokens);
        nuint xBytes = (nuint)((long)maxTokens * HiddenDim * sizeof(float));
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

        if (numTxtTokens > 0)
        {
            nuint txtBytes = (nuint)((long)numTxtTokens * HiddenDim * sizeof(float));
            if (txtBytes > _dTxtCapacity)
            {
                FreeDevicePtr(ref _dTxtDevice);
                _dTxtCapacity = txtBytes * 12 / 10 + 256 * 1024;
                _dTxtDevice = _gpu.AllocateDevice(_dTxtCapacity);
            }
        }
    }

    private (IntPtr dW, IntPtr dB) UploadWeightAndBias(GgufTensorInfo w, GgufTensorInfo? b, int nRows)
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

        // 1. Precompute 3D RoPE lookup table
        Precompute3DRoPE(temporalFrames, tokenH, tokenW, _ropeCos, _ropeSin);

        // 2. Timestep Embedding
        ComputeTimeEmbedding(timestep, _timeVec, _timeModAll);

        // 3. Text Context Projection: [numTxtTokens, 4096] -> [numTxtTokens, 1536] (GELU approx)
        int txtAllocTokens = Math.Max(1, numTxtTokens);
        float* projectedTxt = (float*)NativeMemory.AlignedAlloc((nuint)(txtAllocTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            if (numTxtTokens > 0)
            {
                ProjectText(textContext, numTxtTokens, projectedTxt);
            }

            // 4. Patchify & Project input latents to hidden dimension: [totalImgTokens, 64] -> [totalImgTokens, 1536]
            PatchifyAndProject(latents, temporalFrames, latentH, latentW, _tokenBufferA);

            // 5. Evaluate 30 Transformer Blocks with AdaLN Modulation, 3D RoPE, and Gating
            float* currentTokens = _tokenBufferA;

            if (IsGpuAccelerated)
            {
                // Execute on NVIDIA GPU (RTX 4060)
                EnsureGpuBuffers(totalImgTokens, numTxtTokens);
                _gpu!.CopyToDevice(_dXDevice, (IntPtr)currentTokens, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
                if (numTxtTokens > 0)
                {
                    _gpu.CopyToDevice(_dTxtDevice, (IntPtr)projectedTxt, (nuint)(numTxtTokens * HiddenDim * sizeof(float)));
                }

                for (int blkIdx = 0; blkIdx < NumBlocks; blkIdx++)
                {
                    var blk = _blocks[blkIdx];
                    ExecuteBlockGpu(blk, totalImgTokens, temporalFrames, tokenH, tokenW, _ropeCos, _ropeSin, projectedTxt, numTxtTokens, _timeModAll);
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
                    ExecuteBlockCpu(currentTokens, totalImgTokens, temporalFrames, tokenH, tokenW, _ropeCos, _ropeSin, projectedTxt, numTxtTokens, blk, _timeModAll);
                }
            }

            // 6. Final Head Layer: AdaLN Modulate (head.modulation + _timeVec), Project [1536 -> 64], and Unpatchify to velocity field
            UnpatchifyAndOutput(currentTokens, temporalFrames, latentH, latentW, velocityOut);
        }
        finally
        {
            NativeMemory.AlignedFree(projectedTxt);
        }
    }

    private void ExecuteBlockGpu(
        WanBlockWeights blk,
        int totalImgTokens,
        int frames,
        int tokenH,
        int tokenW,
        float* ropeCos,
        float* ropeSin,
        float* projectedTxt,
        int numTxtTokens,
        float* timeModAll)
    {
        float* modBase = (float*)_gguf.GetTensorPointer(blk.Modulation);
        float* shift_msa   = stackalloc float[HiddenDim];
        float* scale_msa   = stackalloc float[HiddenDim];
        float* gate_msa    = stackalloc float[HiddenDim];
        float* c_shift_msa = stackalloc float[HiddenDim];
        float* c_scale_msa = stackalloc float[HiddenDim];
        float* c_gate_msa  = stackalloc float[HiddenDim];

        for (int d = 0; d < HiddenDim; d++)
        {
            shift_msa[d]   = modBase[0 * HiddenDim + d] + timeModAll[0 * HiddenDim + d];
            scale_msa[d]   = modBase[1 * HiddenDim + d] + timeModAll[1 * HiddenDim + d];
            gate_msa[d]    = modBase[2 * HiddenDim + d] + timeModAll[2 * HiddenDim + d];
            c_shift_msa[d] = modBase[3 * HiddenDim + d] + timeModAll[3 * HiddenDim + d];
            c_scale_msa[d] = modBase[4 * HiddenDim + d] + timeModAll[4 * HiddenDim + d];
            c_gate_msa[d]  = modBase[5 * HiddenDim + d] + timeModAll[5 * HiddenDim + d];
        }

        // Upload block modulation vectors to GPU scratch
        _gpu!.CopyToDevice(_dShiftMsa, (IntPtr)shift_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dScaleMsa, (IntPtr)scale_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dGateMsa, (IntPtr)gate_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCShiftMsa, (IntPtr)c_shift_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCScaleMsa, (IntPtr)c_scale_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCGateMsa, (IntPtr)c_gate_msa, (nuint)(HiddenDim * sizeof(float)));

        // 1. Self-Attention
        // 1a. AdaLN LayerNorm on GPU: _dXDevice -> _dYDevice
        DispatchAdaLnDevice(_dXDevice, _dYDevice, _dShiftMsa, _dScaleMsa, totalImgTokens, HiddenDim);

        // 1b. QKV MatMuls
        DispatchMatMulBatchDevice(blk.D_SelfQ_W, blk.D_SelfQ_B, _dYDevice, _dQDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchDevice(blk.D_SelfK_W, blk.D_SelfK_B, _dYDevice, _dKDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchQ6KDevice(blk.D_SelfV_W, blk.D_SelfV_B, _dYDevice, _dVDevice, HiddenDim, HiddenDim, totalImgTokens);

        // 1c. Across-heads RMSNorm on Q and K
        DispatchRmsNormBatchDevice(_dQDevice, blk.D_SelfNormQ, _dQDevice, HiddenDim, totalImgTokens);
        DispatchRmsNormBatchDevice(_dKDevice, blk.D_SelfNormK, _dKDevice, HiddenDim, totalImgTokens);

        // 1d. 3D-RoPE rotation on Q and K
        _gpu.CopyToHost((IntPtr)_qBuffer, _dQDevice, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
        _gpu.CopyToHost((IntPtr)_kBuffer, _dKDevice, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
        Apply3DRoPEToTokens(_qBuffer, totalImgTokens, ropeCos, ropeSin);
        Apply3DRoPEToTokens(_kBuffer, totalImgTokens, ropeCos, ropeSin);
        _gpu.CopyToDevice(_dQDevice, (IntPtr)_qBuffer, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dKDevice, (IntPtr)_kBuffer, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));

        // 1e. Bidirectional Attention across spatio-temporal tokens
        float scale = 1.0f / MathF.Sqrt(HeadDim);
        DispatchAttentionDevice(_dQDevice, _dKDevice, _dVDevice, _dAttnOut, NumHeads, HeadDim, totalImgTokens, scale);

        // 1f. Self-Attention Output Projection into _dYDevice
        DispatchMatMulBatchDevice(blk.D_SelfO_W, blk.D_SelfO_B, _dAttnOut, _dYDevice, HiddenDim, HiddenDim, totalImgTokens);

        // 1g. Gated Residual: _dXDevice += _dYDevice * gate_msa
        DispatchResidualGatedDevice(_dXDevice, _dYDevice, _dGateMsa, totalImgTokens * HiddenDim, HiddenDim);

        // 2. Cross-Attention (if prompt text context provided)
        if (numTxtTokens > 0)
        {
            // 2a. Affine LayerNorm: _dXDevice -> _dYDevice using blk.D_Norm3_W and blk.D_Norm3_B
            DispatchLayerNormAffineDevice(_dXDevice, _dYDevice, blk.D_Norm3_W, blk.D_Norm3_B, totalImgTokens, HiddenDim);

            // 2b. Q projection from image tokens (_dYDevice) into _dQDevice [totalImgTokens, HiddenDim]
            DispatchMatMulBatchDevice(blk.D_CrossQ_W, blk.D_CrossQ_B, _dYDevice, _dQDevice, HiddenDim, HiddenDim, totalImgTokens);

            // 2c. K projection from text tokens (_dTxtDevice) into _dKDevice [numTxtTokens, HiddenDim]
            DispatchMatMulBatchDevice(blk.D_CrossK_W, blk.D_CrossK_B, _dTxtDevice, _dKDevice, HiddenDim, HiddenDim, numTxtTokens);

            // 2d. V projection from text tokens (_dTxtDevice) into _dVDevice [numTxtTokens, HiddenDim] (Q6_K)
            DispatchMatMulBatchQ6KDevice(blk.D_CrossV_W, blk.D_CrossV_B, _dTxtDevice, _dVDevice, HiddenDim, HiddenDim, numTxtTokens);

            // 2e. Across-heads RMSNorm on Q and K
            DispatchRmsNormBatchDevice(_dQDevice, blk.D_CrossNormQ, _dQDevice, HiddenDim, totalImgTokens);
            DispatchRmsNormBatchDevice(_dKDevice, blk.D_CrossNormK, _dKDevice, HiddenDim, numTxtTokens);

            // 2f. FlashAttention-2 Cross-Attention: Q [totalImgTokens], K [numTxtTokens], V [numTxtTokens] -> _dAttnOut [totalImgTokens]
            DispatchAttentionCrossDevice(_dQDevice, _dKDevice, _dVDevice, _dAttnOut, NumHeads, HeadDim, totalImgTokens, numTxtTokens, scale);

            // 2g. Output projection into _dYDevice
            DispatchMatMulBatchDevice(blk.D_CrossO_W, blk.D_CrossO_B, _dAttnOut, _dYDevice, HiddenDim, HiddenDim, totalImgTokens);

            // 2h. Residual addition: _dXDevice += _dYDevice
            DispatchVecAddBatchDevice(_dXDevice, _dYDevice, totalImgTokens * HiddenDim);
        }

        // 3. Feed-Forward Network: 1536 -> 8960 -> GELU -> 1536 + Gated Residual
        // 3a. AdaLN LayerNorm: _dXDevice -> _dYDevice using c_shift_msa and c_scale_msa
        DispatchAdaLnDevice(_dXDevice, _dYDevice, _dCShiftMsa, _dCScaleMsa, totalImgTokens, HiddenDim);

        // 3b. FFN0
        DispatchMatMulBatchDevice(blk.D_Ffn0_W, blk.D_Ffn0_B, _dYDevice, _dFfnInter, HiddenDim, FfnDim, totalImgTokens);
        DispatchGeluDevice(_dFfnInter, totalImgTokens * FfnDim);

        // 3c. FFN2 into _dYDevice
        DispatchMatMulBatchQ6KDevice(blk.D_Ffn2_W, blk.D_Ffn2_B, _dFfnInter, _dYDevice, FfnDim, HiddenDim, totalImgTokens);

        // 3d. Gated Residual: _dXDevice += _dYDevice * c_gate_msa
        DispatchResidualGatedDevice(_dXDevice, _dYDevice, _dCGateMsa, totalImgTokens * HiddenDim, HiddenDim);
    }

    private void ExecuteBlockCpu(
        float* currentTokens,
        int totalImgTokens,
        int frames,
        int tokenH,
        int tokenW,
        float* ropeCos,
        float* ropeSin,
        float* projectedTxt,
        int numTxtTokens,
        WanBlockWeights blk,
        float* timeModAll)
    {
        float* modBase = (float*)_gguf.GetTensorPointer(blk.Modulation);
        float* shift_msa   = stackalloc float[HiddenDim];
        float* scale_msa   = stackalloc float[HiddenDim];
        float* gate_msa    = stackalloc float[HiddenDim];
        float* c_shift_msa = stackalloc float[HiddenDim];
        float* c_scale_msa = stackalloc float[HiddenDim];
        float* c_gate_msa  = stackalloc float[HiddenDim];

        for (int d = 0; d < HiddenDim; d++)
        {
            shift_msa[d]   = modBase[0 * HiddenDim + d] + timeModAll[0 * HiddenDim + d];
            scale_msa[d]   = modBase[1 * HiddenDim + d] + timeModAll[1 * HiddenDim + d];
            gate_msa[d]    = modBase[2 * HiddenDim + d] + timeModAll[2 * HiddenDim + d];
            c_shift_msa[d] = modBase[3 * HiddenDim + d] + timeModAll[3 * HiddenDim + d];
            c_scale_msa[d] = modBase[4 * HiddenDim + d] + timeModAll[4 * HiddenDim + d];
            c_gate_msa[d]  = modBase[5 * HiddenDim + d] + timeModAll[5 * HiddenDim + d];
        }

        // 1. Self-Attention
        // 1a. AdaLN LayerNorm: _tokenBufferB = LayerNorm(currentTokens) * (1 + scale_msa) + shift_msa
        ApplyAdaLnCpu(currentTokens, _tokenBufferB, shift_msa, scale_msa, totalImgTokens, HiddenDim);

        // 1b. Self-Attention Q, K, V Projections from _tokenBufferB -> output into _tokenBufferC
        ExecuteSelfAttentionCpu(_tokenBufferB, totalImgTokens, blk, ropeCos, ropeSin, _tokenBufferC);

        // 1c. Gated Residual: currentTokens += _tokenBufferC * gate_msa
        ApplyGatedResidualCpu(currentTokens, _tokenBufferC, gate_msa, totalImgTokens, HiddenDim);

        // 2. Cross-Attention (if prompt text context provided)
        if (numTxtTokens > 0)
        {
            float* norm3W = (float*)_gguf.GetTensorPointer(blk.Norm3_W);
            float* norm3B = (float*)_gguf.GetTensorPointer(blk.Norm3_B);

            // 2a. LayerNorm with affine norm3: _tokenBufferB = LayerNorm(currentTokens) * norm3W + norm3B
            ApplyAffineLayerNormCpu(currentTokens, _tokenBufferB, norm3W, norm3B, totalImgTokens, HiddenDim);

            // 2b. Cross-Attention between _tokenBufferB and projectedTxt -> output into _tokenBufferC
            ExecuteCrossAttentionCpu(_tokenBufferB, totalImgTokens, projectedTxt, numTxtTokens, blk, _tokenBufferC);

            // 2c. Residual: currentTokens += _tokenBufferC
            ApplyResidualCpu(currentTokens, _tokenBufferC, totalImgTokens, HiddenDim);
        }

        // 3. Feed-Forward Network
        // 3a. AdaLN LayerNorm: _tokenBufferB = LayerNorm(currentTokens) * (1 + c_scale_msa) + c_shift_msa
        ApplyAdaLnCpu(currentTokens, _tokenBufferB, c_shift_msa, c_scale_msa, totalImgTokens, HiddenDim);

        // 3b. FFN -> output into _tokenBufferC
        ExecuteFFNCpu(_tokenBufferB, totalImgTokens, blk, _tokenBufferC);

        // 3c. Gated Residual: currentTokens += _tokenBufferC * c_gate_msa
        ApplyGatedResidualCpu(currentTokens, _tokenBufferC, c_gate_msa, totalImgTokens, HiddenDim);
    }

    private void DispatchAdaLnDevice(IntPtr dSrc, IntPtr dDst, IntPtr dShift, IntPtr dScale, int numTokens, int dim)
    {
        uint blockSize = 128;
        uint gridX = (uint)numTokens;

        int localTokens = numTokens;
        int localDim = dim;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dShift;
        pArgs[3] = &dScale;
        pArgs[4] = &localTokens;
        pArgs[5] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAdaLn,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_adaln_kernel)");
    }

    private void DispatchLayerNormAffineDevice(IntPtr dSrc, IntPtr dDst, IntPtr dWeight, IntPtr dBias, int numTokens, int dim)
    {
        uint blockSize = 128;
        uint gridX = (uint)numTokens;

        int localTokens = numTokens;
        int localDim = dim;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dWeight;
        pArgs[3] = &dBias;
        pArgs[4] = &localTokens;
        pArgs[5] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnLayerNormAffine,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(layer_norm_affine_kernel)");
    }

    private void DispatchRmsNormBatchDevice(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, int numTokens, float eps = 1e-6f)
    {
        uint blockSize = 256;
        uint gridX = (uint)numTokens;

        int localSize = size;
        float localEps = eps;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &localSize;
        pArgs[4] = &localEps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBatch,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_batch)");
    }

    private void DispatchResidualGatedDevice(IntPtr dTarget, IntPtr dUpdate, IntPtr dGate, int totalElements, int dim)
    {
        uint blockSize = 256;
        uint gridX = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        int localCount = totalElements;
        int localDim = dim;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dTarget;
        pArgs[1] = &dUpdate;
        pArgs[2] = &dGate;
        pArgs[3] = &localCount;
        pArgs[4] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnResidualGated,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_residual_gated)");
    }

    private void DispatchVecAddBatchDevice(IntPtr dTarget, IntPtr dSource, int count)
    {
        uint blockSize = 256;
        uint gridX = (uint)((count + (int)blockSize - 1) / (int)blockSize);

        int localCount = count;

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dTarget;
        pArgs[1] = &dSource;
        pArgs[2] = &localCount;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAdd,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add_kernel)");
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

    private void DispatchAttentionCrossDevice(
        IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dAttnOut,
        int nHeads, int headDim, int numQTokens, int numKvTokens, float attnScale)
    {
        uint blockSize = 128;
        uint gridX = (uint)nHeads;
        uint gridY = (uint)numQTokens;

        int localHeads = nHeads;
        int localHeadDim = headDim;
        int localQTokens = numQTokens;
        int localKvTokens = numKvTokens;
        float localScale = attnScale;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dAttnOut;
        pArgs[4] = &localHeads;
        pArgs[5] = &localHeadDim;
        pArgs[6] = &localQTokens;
        pArgs[7] = &localKvTokens;
        pArgs[8] = &localScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnCrossBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_cross_batch)");
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

    private void Precompute3DRoPE(int frames, int tokenH, int tokenW, float* ropeCos, float* ropeSin)
    {
        for (int f = 0; f < frames; f++)
        {
            for (int th = 0; th < tokenH; th++)
            {
                for (int tw = 0; tw < tokenW; tw++)
                {
                    int tokIdx = f * (tokenH * tokenW) + th * tokenW + tw;
                    float* cosT = ropeCos + tokIdx * 64;
                    float* sinT = ropeSin + tokIdx * 64;

                    // Wan 2.1 3D-RoPE: HeadDim = 128 (64 frequency pairs)
                    // Split sizes: t_dim = 44 (22 pairs), h_dim = 42 (21 pairs), w_dim = 42 (21 pairs)
                    // Base theta = 10000.0f
                    // 1. Temporal: 22 pairs (t_dim = 44)
                    for (int k = 0; k < 22; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 44.0f);
                        float theta = f * freq;
                        cosT[k] = MathF.Cos(theta);
                        sinT[k] = MathF.Sin(theta);
                    }

                    // 2. Height: 21 pairs (h_dim = 42)
                    for (int k = 0; k < 21; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 42.0f);
                        float theta = th * freq;
                        cosT[22 + k] = MathF.Cos(theta);
                        sinT[22 + k] = MathF.Sin(theta);
                    }

                    // 3. Width: 21 pairs (w_dim = 42)
                    for (int k = 0; k < 21; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 42.0f);
                        float theta = tw * freq;
                        cosT[43 + k] = MathF.Cos(theta);
                        sinT[43 + k] = MathF.Sin(theta);
                    }
                }
            }
        }
    }

    private static void Apply3DRoPEToTokens(float* qk, int totalTokens, float* ropeCos, float* ropeSin)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* cosT = ropeCos + t * 64;
            float* sinT = ropeSin + t * 64;

            for (int h = 0; h < NumHeads; h++)
            {
                float* head = qk + (long)t * HiddenDim + h * HeadDim;

                for (int p = 0; p < 64; p++)
                {
                    float c = cosT[p];
                    float s = sinT[p];

                    float v0 = head[2 * p];
                    float v1 = head[2 * p + 1];

                    head[2 * p]     = v0 * c - v1 * s;
                    head[2 * p + 1] = v0 * s + v1 * c;
                }
            }
        });
    }

    private void ComputeTimeEmbedding(float timestep, float* timeVec, float* timeModAll)
    {
        float* rawTime = stackalloc float[256];
        float halfDim = 128f;
        float tVal = timestep <= 1.0f ? timestep * 1000.0f : timestep;
        for (int i = 0; i < 128; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / halfDim);
            float arg = tVal * freq;
            rawTime[i] = MathF.Cos(arg);
            rawTime[128 + i] = MathF.Sin(arg);
        }

        float* timeInt = stackalloc float[HiddenDim];
        DispatchLinearVec(_timeEmbed0W, _timeEmbed0B, rawTime, timeInt);
        ApplySilu(timeInt, HiddenDim);

        DispatchLinearVec(_timeEmbed2W, _timeEmbed2B, timeInt, timeVec);

        float* siluTimeVec = stackalloc float[HiddenDim];
        for (int i = 0; i < HiddenDim; i++)
        {
            siluTimeVec[i] = timeVec[i] / (1.0f + MathF.Exp(-timeVec[i]));
        }

        DispatchLinearVec(_timeProj1W, _timeProj1B, siluTimeVec, timeModAll);
    }

    private void ProjectText(ReadOnlySpan<float> textContext, int numTxtTokens, float* projectedTxt)
    {
        fixed (float* pTxt = textContext)
        {
            nint pTxtNint = (nint)pTxt;
            nint projTxtNint = (nint)projectedTxt;

            Parallel.For(0, numTxtTokens, t =>
            {
                float* intermediate = stackalloc float[HiddenDim];
                float* src = (float*)pTxtNint + t * 4096;
                float* dst = (float*)projTxtNint + t * HiddenDim;

                DispatchLinearVec(_textEmbed0W, _textEmbed0B, src, intermediate);
                ApplyGelu(intermediate, HiddenDim);
                DispatchLinearVec(_textEmbed2W, _textEmbed2B, intermediate, dst);
            });
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

    private static void ApplyAdaLnCpu(
        float* src, float* dst, float* shift, float* scale, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* s = src + (long)t * dim;
            float* d = dst + (long)t * dim;

            float sum = 0f;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++)
            {
                float v = s[i];
                sum += v;
                sumSq += v * v;
            }

            float mean = sum / dim;
            float var = MathF.Max(0f, (sumSq / dim) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(var + 1e-6f);

            for (int i = 0; i < dim; i++)
            {
                d[i] = ((s[i] - mean) * invStd) * (1.0f + scale[i]) + shift[i];
            }
        });
    }

    private static void ApplyAffineLayerNormCpu(
        float* src, float* dst, float* weight, float* bias, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* s = src + (long)t * dim;
            float* d = dst + (long)t * dim;

            float sum = 0f;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++)
            {
                float v = s[i];
                sum += v;
                sumSq += v * v;
            }

            float mean = sum / dim;
            float var = MathF.Max(0f, (sumSq / dim) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(var + 1e-6f);

            for (int i = 0; i < dim; i++)
            {
                float w = weight != null ? weight[i] : 1.0f;
                float b = bias != null ? bias[i] : 0.0f;
                d[i] = ((s[i] - mean) * invStd) * w + b;
            }
        });
    }

    private static void ApplyGatedResidualCpu(
        float* target, float* update, float* gate, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* tgt = target + (long)t * dim;
            float* upd = update + (long)t * dim;

            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i] * gate[i];
            }
        });
    }

    private static void ApplyResidualCpu(
        float* target, float* update, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* tgt = target + (long)t * dim;
            float* upd = update + (long)t * dim;

            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i];
            }
        });
    }

    private void ExecuteSelfAttentionCpu(
        float* normTokens, int totalTokens, WanBlockWeights blk, float* ropeCos, float* ropeSin, float* dstTokens)
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

        QuantKernels.MatMulBatch(blk.SelfQ_W.Type, qW, normTokens, _qBuffer, qCols, qRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfK_W.Type, kW, normTokens, _kBuffer, kCols, kRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfV_W.Type, vW, normTokens, _vBuffer, vCols, vRows, totalTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.SelfNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.SelfNormK);

        // Across-heads RMSNorm on Q and K
        Parallel.For(0, totalTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;

            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            float sumSqQ = 0f;
            float sumSqK = 0f;
            for (int d = 0; d < HiddenDim; d++)
            {
                sumSqQ += q[d] * q[d];
                sumSqK += k[d] * k[d];
            }
            float invRmsQ = 1.0f / MathF.Sqrt(sumSqQ / HiddenDim + 1e-6f);
            float invRmsK = 1.0f / MathF.Sqrt(sumSqK / HiddenDim + 1e-6f);

            for (int d = 0; d < HiddenDim; d++)
            {
                q[d] = (q[d] * invRmsQ) * normQScale[d];
                k[d] = (k[d] * invRmsK) * normKScale[d];
            }
        });

        // 3D-RoPE
        Apply3DRoPEToTokens(_qBuffer, totalTokens, ropeCos, ropeSin);
        Apply3DRoPEToTokens(_kBuffer, totalTokens, ropeCos, ropeSin);

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

        QuantKernels.MatMulBatch(blk.SelfO_W.Type, oW, dstTokens, _tokenBufferB, oCols, oRows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _tokenBufferB + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteCrossAttentionCpu(
        float* normTokens, int totalImgTokens, float* txtTokens, int numTxtTokens, WanBlockWeights blk, float* dstTokens)
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

        QuantKernels.MatMulBatch(blk.CrossQ_W.Type, qW, normTokens, _qBuffer, qCols, qRows, totalImgTokens);
        QuantKernels.MatMulBatch(blk.CrossK_W.Type, kW, txtTokens, _kBuffer, kCols, kRows, numTxtTokens);
        QuantKernels.MatMulBatch(blk.CrossV_W.Type, vW, txtTokens, _vBuffer, vCols, vRows, numTxtTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.CrossNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.CrossNormK);

        // Across-heads RMSNorm on Cross-Q
        Parallel.For(0, totalImgTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];

            float sumSq = 0f;
            for (int d = 0; d < HiddenDim; d++) sumSq += q[d] * q[d];
            float invRms = 1.0f / MathF.Sqrt(sumSq / HiddenDim + 1e-6f);
            for (int d = 0; d < HiddenDim; d++) q[d] = (q[d] * invRms) * normQScale[d];
        });

        // Across-heads RMSNorm on Cross-K
        Parallel.For(0, numTxtTokens, t =>
        {
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            float sumSq = 0f;
            for (int d = 0; d < HiddenDim; d++) sumSq += k[d] * k[d];
            float invRms = 1.0f / MathF.Sqrt(sumSq / HiddenDim + 1e-6f);
            for (int d = 0; d < HiddenDim; d++) k[d] = (k[d] * invRms) * normKScale[d];
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

        QuantKernels.MatMulBatch(blk.CrossO_W.Type, oW, dstTokens, _tokenBufferB, oCols, oRows, totalImgTokens);

        Parallel.For(0, totalImgTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _tokenBufferB + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteFFNCpu(float* normTokens, int totalTokens, WanBlockWeights blk, float* dstTokens)
    {
        var (f0Cols, f0Rows) = GetMatrixDims(blk.Ffn0_W);
        var (f2Cols, f2Rows) = GetMatrixDims(blk.Ffn2_W);

        byte* f0W = _gguf.GetTensorPointer(blk.Ffn0_W);
        byte* f2W = _gguf.GetTensorPointer(blk.Ffn2_W);

        float* f0B = (float*)_gguf.GetTensorPointer(blk.Ffn0_B);
        float* f2B = (float*)_gguf.GetTensorPointer(blk.Ffn2_B);

        QuantKernels.MatMulBatch(blk.Ffn0_W.Type, f0W, normTokens, _ffnIntermediate, f0Cols, f0Rows, totalTokens);

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

        if (f2B != null)
        {
            Parallel.For(0, totalTokens, t =>
            {
                float* outToken = dstTokens + (long)t * HiddenDim;
                for (int d = 0; d < HiddenDim; d++) outToken[d] += f2B[d];
            });
        }
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
            float* headShift = stackalloc float[HiddenDim];
            float* headScale = stackalloc float[HiddenDim];

            for (int d = 0; d < HiddenDim; d++)
            {
                headShift[d] = modPtr[0 * HiddenDim + d] + _timeVec[d];
                headScale[d] = modPtr[1 * HiddenDim + d] + _timeVec[d];
            }

            // Head LayerNorm + AdaLN modulation
            ApplyAdaLnCpu(tokens, normTokens, headShift, headScale, totalTokens, HiddenDim);

            // Project 1536 -> 64
            QuantKernels.MatMulBatch(_headW.Type, wPtr, normTokens, patchBufAll, nCols, nRows, totalTokens);

            if (bPtr != null)
            {
                Parallel.For(0, totalTokens, t =>
                {
                    float* p = patchBufAll + (long)t * PatchDim;
                    for (int d = 0; d < PatchDim; d++) p[d] += bPtr[d];
                });
            }

            // Unpatchify: 64 output channels are ordered (ph * PatchSize + pw) * InChannels + c
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

                            for (int ph = 0; ph < PatchSize; ph++)
                            {
                                for (int pw = 0; pw < PatchSize; pw++)
                                {
                                    int y = th * PatchSize + ph;
                                    int x = tw * PatchSize + pw;

                                    for (int c = 0; c < InChannels; c++)
                                    {
                                        int pIdx = (ph * PatchSize + pw) * InChannels + c;
                                        pOut[frameLatentOffset + c * H * W + y * W + x] = patchBuf[pIdx];
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
    private void DispatchLinearVec(GgufTensorInfo w, GgufTensorInfo? b, float* x, float* y)
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
            FreeDevicePtr(ref _dTxtDevice);

            FreeDevicePtr(ref _dShiftMsa);
            FreeDevicePtr(ref _dScaleMsa);
            FreeDevicePtr(ref _dGateMsa);
            FreeDevicePtr(ref _dCShiftMsa);
            FreeDevicePtr(ref _dCScaleMsa);
            FreeDevicePtr(ref _dCGateMsa);

            _gpu?.Dispose();

            if (_timeVec != null) NativeMemory.AlignedFree(_timeVec);
            if (_timeModAll != null) NativeMemory.AlignedFree(_timeModAll);
            if (_tokenBufferA != null) NativeMemory.AlignedFree(_tokenBufferA);
            if (_tokenBufferB != null) NativeMemory.AlignedFree(_tokenBufferB);
            if (_tokenBufferC != null) NativeMemory.AlignedFree(_tokenBufferC);
            if (_qBuffer != null) NativeMemory.AlignedFree(_qBuffer);
            if (_kBuffer != null) NativeMemory.AlignedFree(_kBuffer);
            if (_vBuffer != null) NativeMemory.AlignedFree(_vBuffer);
            if (_ffnIntermediate != null) NativeMemory.AlignedFree(_ffnIntermediate);
            if (_ropeCos != null) NativeMemory.AlignedFree(_ropeCos);
            if (_ropeSin != null) NativeMemory.AlignedFree(_ropeSin);
            _gguf.Dispose();
        }
    }
}
