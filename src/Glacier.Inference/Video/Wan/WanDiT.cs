namespace Glacier.Inference.Video.Wan;

using Glacier.Inference.Diagnostics;
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
public sealed unsafe partial class WanDiT : IDisposable
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
    private IntPtr _fnRoPE3D;

    private IntPtr _dXDevice = IntPtr.Zero;
    private IntPtr _dYDevice = IntPtr.Zero;
    private IntPtr _dQDevice = IntPtr.Zero;
    private IntPtr _dKDevice = IntPtr.Zero;
    private IntPtr _dVDevice = IntPtr.Zero;
    private IntPtr _dAttnOut = IntPtr.Zero;
    private IntPtr _dFfnInter = IntPtr.Zero;
    private IntPtr _dTxtDevice = IntPtr.Zero;

    private IntPtr _dRoPECos = IntPtr.Zero;
    private IntPtr _dRoPESin = IntPtr.Zero;

    private IntPtr _dShiftMsa = IntPtr.Zero;
    private IntPtr _dScaleMsa = IntPtr.Zero;
    private IntPtr _dGateMsa = IntPtr.Zero;
    private IntPtr _dCShiftMsa = IntPtr.Zero;
    private IntPtr _dCScaleMsa = IntPtr.Zero;
    private IntPtr _dCGateMsa = IntPtr.Zero;

    private nuint _dXCapacity = 0;
    private nuint _dFfnCapacity = 0;
    private nuint _dTxtCapacity = 0;
    private nuint _dRoPECapacity = 0;
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
    public string ActiveBackend => IsGpuAccelerated ? "Cuda" : "Cpu";

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

        // Hardware GPU Acceleration Initialization
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
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRoPE3D, _gpuModule, "rope_3d_in_vram"), "ModuleGetFunction(rope_3d_in_vram)");

                _dShiftMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dScaleMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dGateMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCShiftMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCScaleMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));
                _dCGateMsa = _gpu.AllocateDevice((nuint)(HiddenDim * sizeof(float)));

                UploadGpuBlockWeights();
                GlacierDiagnostics.LogInformation($"[GLACIER GPU] Accelerated Wan2.1 Video DiT on {_gpu.DeviceName} ({_gpu.ArchString}). {_blocks.Length} blocks resident in VRAM.");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER GPU] Wan2.1 GPU init skipped [{ex.GetType().Name}]: {ex.Message}, falling back to CPU SIMD.", ex);
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

        nuint ropeBytes = (nuint)((long)numTokens * 64 * sizeof(float));
        if (ropeBytes > _dRoPECapacity)
        {
            FreeDevicePtr(ref _dRoPECos);
            FreeDevicePtr(ref _dRoPESin);
            _dRoPECapacity = ropeBytes * 12 / 10 + 256 * 1024;
            _dRoPECos = _gpu.AllocateDevice(_dRoPECapacity);
            _dRoPESin = _gpu.AllocateDevice(_dRoPECapacity);
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
    /// Hardware accelerated on GPU with seamless CPU fallback.
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
                // Execute on GPU
                EnsureGpuBuffers(totalImgTokens, numTxtTokens);
                _gpu!.CopyToDevice(_dXDevice, (IntPtr)currentTokens, (nuint)(totalImgTokens * HiddenDim * sizeof(float)));
                _gpu.CopyToDevice(_dRoPECos, (IntPtr)_ropeCos, (nuint)(totalImgTokens * 64 * sizeof(float)));
                _gpu.CopyToDevice(_dRoPESin, (IntPtr)_ropeSin, (nuint)(totalImgTokens * 64 * sizeof(float)));
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
            FreeDevicePtr(ref _dRoPECos);
            FreeDevicePtr(ref _dRoPESin);

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
