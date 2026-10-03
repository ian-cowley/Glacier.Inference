namespace Glacier.Inference.Image.Flux;

using System;
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
/// Pure C# .NET 10 Diffusion Transformer (DiT) backbone executing the real Flux.1-schnell architecture.
/// Maps 19 DoubleStreamBlocks and 38 SingleStreamBlocks directly from GGUF weights (Q4_K_S)
/// with 3D-RoPE, QK-Norm, AdaLN modulation, and AVX-512 SIMD batch parallelization.
/// </summary>
public unsafe sealed class FluxDiT : IDisposable
{
    public const int HiddenDim = 3072;
    public const int NumHeads = 24;
    public const int HeadDim = 128;
    public const int InChannels = 16;
    public const int PatchSize = 2;
    public const int PatchDim = PatchSize * PatchSize * InChannels; // 64
    public const int DoubleBlocksCount = 19;
    public const int SingleBlocksCount = 38;

    private readonly GgufFile _gguf;

    // Prelude weights
    private readonly GgufTensorInfo _imgInW, _imgInB;
    private readonly GgufTensorInfo _txtInW, _txtInB;
    private readonly GgufTensorInfo _timeIn1W, _timeIn1B, _timeIn2W, _timeIn2B;
    private readonly GgufTensorInfo? _vecIn1W, _vecIn1B, _vecIn2W, _vecIn2B;

    // Double stream blocks
    private readonly DoubleBlockWeights[] _doubleBlocks = new DoubleBlockWeights[DoubleBlocksCount];

    // Single stream blocks
    private readonly SingleBlockWeights[] _singleBlocks = new SingleBlockWeights[SingleBlocksCount];

    // Final layer weights
    private readonly GgufTensorInfo _finalAdaLnW, _finalAdaLnB;
    private readonly GgufTensorInfo _finalLinearW, _finalLinearB;

    // Modulation and vector scratch memory
    private readonly float* _imgModBuffer;    // [18432]
    private readonly float* _txtModBuffer;    // [18432]
    private readonly float* _singleModBuffer; // [9216]
    private readonly float* _finalModBuffer;  // [6144]
    private readonly float* _vecBuffer;       // [3072]
    private readonly float* _siluVecBuffer;   // [3072]

    // Max capacity
    private readonly int _maxTokens;
    private readonly float* _scratchBuffer;
    private readonly nuint _scratchSizeBytes;
    private bool _disposed;

    // GPU Hardware Acceleration
    private readonly GpuContext? _gpu;
    private IntPtr _gpuModule;
    private IntPtr _fnGemmQ4KBatch;
    private IntPtr _fnAttnBidirectionalBatch;
    private IntPtr _fnQkvPrep;
    private IntPtr _fnGeluConcat;
    private IntPtr _fnGelu;
    private IntPtr _fnAdaLn;
    private IntPtr _fnResidualGated;
    private IntPtr _dFinalLinearW = IntPtr.Zero, _dFinalLinearB = IntPtr.Zero;
    private IntPtr _dXDevice = IntPtr.Zero;
    private IntPtr _dYDevice = IntPtr.Zero;
    private nuint _dXDeviceCapacity = 0;
    private nuint _dYDeviceCapacity = 0;

    // GPU In-VRAM Activation Resident Pipeline Buffers
    private IntPtr _dSingleStream = IntPtr.Zero;
    private IntPtr _dNormBuf = IntPtr.Zero;
    private IntPtr _dQkvMlpBuf = IntPtr.Zero;
    private IntPtr _dAttnOut = IntPtr.Zero;
    private IntPtr _dConcatBuf = IntPtr.Zero;
    private IntPtr _dQ = IntPtr.Zero;
    private IntPtr _dK = IntPtr.Zero;
    private IntPtr _dV = IntPtr.Zero;
    private IntPtr _dRopeCos = IntPtr.Zero;
    private IntPtr _dRopeSin = IntPtr.Zero;
    private IntPtr _dSingleMod = IntPtr.Zero;
    private IntPtr _dDoubleModImg = IntPtr.Zero;
    private IntPtr _dDoubleModTxt = IntPtr.Zero;
    private IntPtr _dFinalOut = IntPtr.Zero;
    private int _allocatedGpuTokens = 0;

    private bool _fullGpuMode = false;
    private bool _doubleBlocksLoaded = false;
    private bool _singleBlocksLoaded = false;

    private sealed class DoubleBlockWeights
    {
        public GgufTensorInfo ImgModW = null!, ImgModB = null!;
        public GgufTensorInfo TxtModW = null!, TxtModB = null!;
        public GgufTensorInfo ImgQkvW = null!, ImgQkvB = null!;
        public GgufTensorInfo TxtQkvW = null!, TxtQkvB = null!;
        public GgufTensorInfo ImgQueryNorm = null!, ImgKeyNorm = null!;
        public GgufTensorInfo TxtQueryNorm = null!, TxtKeyNorm = null!;
        public GgufTensorInfo ImgProjW = null!, ImgProjB = null!;
        public GgufTensorInfo TxtProjW = null!, TxtProjB = null!;
        public GgufTensorInfo ImgMlp0W = null!, ImgMlp0B = null!;
        public GgufTensorInfo ImgMlp2W = null!, ImgMlp2B = null!;
        public GgufTensorInfo TxtMlp0W = null!, TxtMlp0B = null!;
        public GgufTensorInfo TxtMlp2W = null!, TxtMlp2B = null!;

        // GPU VRAM Pointers
        public IntPtr D_ImgQkvW, D_ImgQkvB;
        public IntPtr D_TxtQkvW, D_TxtQkvB;
        public IntPtr D_ImgProjW, D_ImgProjB;
        public IntPtr D_TxtProjW, D_TxtProjB;
        public IntPtr D_ImgMlp0W, D_ImgMlp0B;
        public IntPtr D_ImgMlp2W, D_ImgMlp2B;
        public IntPtr D_TxtMlp0W, D_TxtMlp0B;
        public IntPtr D_TxtMlp2W, D_TxtMlp2B;

        public IntPtr D_ImgQueryNorm, D_ImgKeyNorm;
        public IntPtr D_TxtQueryNorm, D_TxtKeyNorm;
    }

    private sealed class SingleBlockWeights
    {
        public GgufTensorInfo ModW = null!, ModB = null!;
        public GgufTensorInfo Lin1W = null!, Lin1B = null!;
        public GgufTensorInfo QueryNorm = null!, KeyNorm = null!;
        public GgufTensorInfo Lin2W = null!, Lin2B = null!;

        // GPU VRAM Pointers
        public IntPtr D_Lin1W, D_Lin1B;
        public IntPtr D_Lin2W, D_Lin2B;
        public IntPtr D_QueryNorm, D_KeyNorm;
    }

    private FluxDiT(GgufFile gguf, int maxTokens = 4160, bool enableGpu = true)
    {
        _gguf = gguf;
        _maxTokens = maxTokens;

        _imgInW = gguf.Tensors["img_in.weight"];
        _imgInB = gguf.Tensors["img_in.bias"];
        _txtInW = gguf.Tensors["txt_in.weight"];
        _txtInB = gguf.Tensors["txt_in.bias"];

        _timeIn1W = gguf.Tensors["time_in.in_layer.weight"];
        _timeIn1B = gguf.Tensors["time_in.in_layer.bias"];
        _timeIn2W = gguf.Tensors["time_in.out_layer.weight"];
        _timeIn2B = gguf.Tensors["time_in.out_layer.bias"];

        gguf.TryGetTensor("vector_in.in_layer.weight", out _vecIn1W);
        gguf.TryGetTensor("vector_in.in_layer.bias", out _vecIn1B);
        gguf.TryGetTensor("vector_in.out_layer.weight", out _vecIn2W);
        gguf.TryGetTensor("vector_in.out_layer.bias", out _vecIn2B);

        for (int i = 0; i < DoubleBlocksCount; i++)
        {
            string p = $"double_blocks.{i}.";
            _doubleBlocks[i] = new DoubleBlockWeights
            {
                ImgModW = gguf.Tensors[p + "img_mod.lin.weight"],
                ImgModB = gguf.Tensors[p + "img_mod.lin.bias"],
                TxtModW = gguf.Tensors[p + "txt_mod.lin.weight"],
                TxtModB = gguf.Tensors[p + "txt_mod.lin.bias"],
                ImgQkvW = gguf.Tensors[p + "img_attn.qkv.weight"],
                ImgQkvB = gguf.Tensors[p + "img_attn.qkv.bias"],
                TxtQkvW = gguf.Tensors[p + "txt_attn.qkv.weight"],
                TxtQkvB = gguf.Tensors[p + "txt_attn.qkv.bias"],
                ImgQueryNorm = gguf.Tensors[p + "img_attn.norm.query_norm.scale"],
                ImgKeyNorm = gguf.Tensors[p + "img_attn.norm.key_norm.scale"],
                TxtQueryNorm = gguf.Tensors[p + "txt_attn.norm.query_norm.scale"],
                TxtKeyNorm = gguf.Tensors[p + "txt_attn.norm.key_norm.scale"],
                ImgProjW = gguf.Tensors[p + "img_attn.proj.weight"],
                ImgProjB = gguf.Tensors[p + "img_attn.proj.bias"],
                TxtProjW = gguf.Tensors[p + "txt_attn.proj.weight"],
                TxtProjB = gguf.Tensors[p + "txt_attn.proj.bias"],
                ImgMlp0W = gguf.Tensors[p + "img_mlp.0.weight"],
                ImgMlp0B = gguf.Tensors[p + "img_mlp.0.bias"],
                ImgMlp2W = gguf.Tensors[p + "img_mlp.2.weight"],
                ImgMlp2B = gguf.Tensors[p + "img_mlp.2.bias"],
                TxtMlp0W = gguf.Tensors[p + "txt_mlp.0.weight"],
                TxtMlp0B = gguf.Tensors[p + "txt_mlp.0.bias"],
                TxtMlp2W = gguf.Tensors[p + "txt_mlp.2.weight"],
                TxtMlp2B = gguf.Tensors[p + "txt_mlp.2.bias"]
            };
        }

        for (int i = 0; i < SingleBlocksCount; i++)
        {
            string p = $"single_blocks.{i}.";
            _singleBlocks[i] = new SingleBlockWeights
            {
                ModW = gguf.Tensors[p + "modulation.lin.weight"],
                ModB = gguf.Tensors[p + "modulation.lin.bias"],
                Lin1W = gguf.Tensors[p + "linear1.weight"],
                Lin1B = gguf.Tensors[p + "linear1.bias"],
                QueryNorm = gguf.Tensors[p + "norm.query_norm.scale"],
                KeyNorm = gguf.Tensors[p + "norm.key_norm.scale"],
                Lin2W = gguf.Tensors[p + "linear2.weight"],
                Lin2B = gguf.Tensors[p + "linear2.bias"]
            };
        }

        _finalAdaLnW = gguf.Tensors["final_layer.adaLN_modulation.1.weight"];
        _finalAdaLnB = gguf.Tensors["final_layer.adaLN_modulation.1.bias"];
        _finalLinearW = gguf.Tensors["final_layer.linear.weight"];
        _finalLinearB = gguf.Tensors["final_layer.linear.bias"];

        // Preallocate modulation buffers
        _imgModBuffer = (float*)NativeMemory.AlignedAlloc(18432 * sizeof(float), 64);
        _txtModBuffer = (float*)NativeMemory.AlignedAlloc(18432 * sizeof(float), 64);
        _singleModBuffer = (float*)NativeMemory.AlignedAlloc(9216 * sizeof(float), 64);
        _finalModBuffer = (float*)NativeMemory.AlignedAlloc(6144 * sizeof(float), 64);
        _vecBuffer = (float*)NativeMemory.AlignedAlloc(HiddenDim * sizeof(float), 64);
        _siluVecBuffer = (float*)NativeMemory.AlignedAlloc(HiddenDim * sizeof(float), 64);

        // Preallocate scratch memory: non-overlapping buffers for 44k floats/token
        _scratchSizeBytes = (nuint)(_maxTokens * 44000 * sizeof(float) + 16 * 1024 * 1024);
        _scratchBuffer = (float*)NativeMemory.AlignedAlloc(_scratchSizeBytes, 64);

        if (enableGpu && GpuContext.IsSupported)
        {
            try
            {
                _gpu = new GpuContext(0);
                byte[] cubin = KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
                CuDriver.Check(CuDriver.ModuleLoadData(out _gpuModule, cubin), "ModuleLoadData");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemmQ4KBatch, _gpuModule, "gemm_q4_k_batch"), "ModuleGetFunction(gemm_q4_k_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttnBidirectionalBatch, _gpuModule, "attention_bidirectional_batch"), "ModuleGetFunction(attention_bidirectional_batch)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnQkvPrep, _gpuModule, "flux_qkv_prep"), "ModuleGetFunction(flux_qkv_prep)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGeluConcat, _gpuModule, "flux_fused_gelu_concat"), "ModuleGetFunction(flux_fused_gelu_concat)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAdaLn, _gpuModule, "flux_adaln_kernel"), "ModuleGetFunction(flux_adaln_kernel)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnResidualGated, _gpuModule, "flux_residual_gated"), "ModuleGetFunction(flux_residual_gated)");
                CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGelu, _gpuModule, "flux_gelu_kernel"), "ModuleGetFunction(flux_gelu_kernel)");

                var (freeBytes, totalBytes) = _gpu.GetMemoryInfo();
                ulong freeMb = freeBytes / (1024 * 1024);
                Console.WriteLine($"[GLACIER GPU] Detected {_gpu.DeviceName} ({_gpu.ArchString}) with {freeMb} MB free VRAM.");

                if (freeMb >= 4800)
                {
                    Console.WriteLine("[GLACIER GPU] Uploading FLUX.1 DiT weights into GPU VRAM for zero-transfer inference...");
                    UploadDoubleBlocks();
                    UploadSingleBlocks();
                    (_dFinalLinearW, _dFinalLinearB) = UploadWeightAndBias(_finalLinearW, _finalLinearB, PatchDim);
                    _fullGpuMode = true;
                    Console.WriteLine("[GLACIER GPU] DiT weights resident in GPU VRAM (Full VRAM Mode).");
                }
                else if (freeMb >= 3700)
                {
                    Console.WriteLine("[GLACIER GPU] Operating in two-phase dynamic GPU offload mode.");
                    _fullGpuMode = false;
                }
                else
                {
                    Console.WriteLine("[GLACIER GPU] Insufficient free VRAM for GPU offload, falling back to CPU SIMD.");
                    _gpu.Dispose();
                    _gpu = null;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[GLACIER GPU] GPU initialization skipped ({ex.Message}), falling back to CPU SIMD.");
                _gpu?.Dispose();
                _gpu = null;
            }
        }
    }

    public static FluxDiT Open(string ggufPath, int maxTokens = 4160, bool enableGpu = true)
    {
        var gguf = GgufFile.Open(ggufPath);
        return new FluxDiT(gguf, maxTokens, enableGpu);
    }

    /// <summary>
    /// Predicts the flow velocity field v_theta(x_t, t, c) using the full 57-block Flux DiT backbone.
    /// </summary>
    public void PredictVelocity(
        ReadOnlySpan<float> latents,
        int latentH,
        int latentW,
        float timestep,
        ReadOnlySpan<float> pooledY,
        ReadOnlySpan<float> contextTxt,
        int numTxtTokens,
        Span<float> outputVelocity)
    {
        int tokenH = latentH / PatchSize;
        int tokenW = latentW / PatchSize;
        int numImgTokens = tokenH * tokenW;
        int numTotalTokens = numTxtTokens + numImgTokens;

        if (numTotalTokens > _maxTokens)
        {
            throw new ArgumentException($"Total tokens ({numTotalTokens}) exceeds allocated capacity ({_maxTokens}).");
        }

        // 1. Timestep & Vector Embeddings -> vec [3072]
        ComputeConditioningVector(timestep, pooledY, _vecBuffer);
        for (int i = 0; i < HiddenDim; i++) _siluVecBuffer[i] = _vecBuffer[i] / (1.0f + MathF.Exp(-_vecBuffer[i]));

        // Scratch pointers: guaranteed disjoint non-overlapping layout
        float* txtTokens = _scratchBuffer;                                          // [numTxtTokens, 3072]
        float* imgTokens = txtTokens + numTxtTokens * HiddenDim;                    // [numImgTokens, 3072]
        float* singleStream = _scratchBuffer;                                       // [numTotalTokens, 3072] (txt followed by img)
        float* normBuf = singleStream + numTotalTokens * HiddenDim;                 // [numTotalTokens, 3072]
        float* qkvBuf = normBuf + numTotalTokens * HiddenDim;                       // [numTotalTokens, 21504]
        float* attnOut = qkvBuf + numTotalTokens * 21504;                           // [numTotalTokens, 15360]

        // 2. Patchify & Project latents -> imgTokens [numImgTokens, 3072]
        PatchifyAndProject(latents, latentH, latentW, imgTokens);

        // 3. Project contextTxt -> txtTokens [numTxtTokens, 3072]
        ProjectTextContext(contextTxt, numTxtTokens, txtTokens);

        // 4. Precompute 3D RoPE cos/sin tables
        int ropeDimHalf = HeadDim / 2; // 64
        float[] ropeCosArr = new float[numTotalTokens * ropeDimHalf];
        float[] ropeSinArr = new float[numTotalTokens * ropeDimHalf];

        fixed (float* ropeCos = ropeCosArr)
        fixed (float* ropeSin = ropeSinArr)
        {
            Compute3DRoPE(numTxtTokens, tokenH, tokenW, ropeCos, ropeSin);

            if (_gpu != null)
            {
                EnsureGpuActivationBuffers(numTotalTokens);
                _gpu.CopyToDevice(_dRopeCos, (IntPtr)ropeCos, (nuint)((long)numTotalTokens * 64 * sizeof(float)));
                _gpu.CopyToDevice(_dRopeSin, (IntPtr)ropeSin, (nuint)((long)numTotalTokens * 64 * sizeof(float)));
            }

            // 5. 19 DoubleStreamBlocks
            var swDouble = System.Diagnostics.Stopwatch.StartNew();
            PrepareDoubleStreamGpu();
            if (_gpu != null && _doubleBlocksLoaded)
            {
                ExecuteDoubleBlocksGpu(txtTokens, imgTokens, numTxtTokens, numImgTokens, ropeCos, ropeSin);
            }
            else
            {
                for (int b = 0; b < DoubleBlocksCount; b++)
                {
                    var blk = _doubleBlocks[b];

                    // Modulation vectors: [3072] -> [18432]
                    DispatchLinearVec(blk.ImgModW, blk.ImgModB, _siluVecBuffer, _imgModBuffer);
                    DispatchLinearVec(blk.TxtModW, blk.TxtModB, _siluVecBuffer, _txtModBuffer);

                    float* imgShift1 = _imgModBuffer;
                    float* imgScale1 = _imgModBuffer + 3072;
                    float* imgGate1 = _imgModBuffer + 6144;

                    float* txtShift1 = _txtModBuffer;
                    float* txtScale1 = _txtModBuffer + 3072;
                    float* txtGate1 = _txtModBuffer + 6144;

                    // Modulate txt & img into normBuf (txt first, img second)
                    ApplyAdaLN(txtTokens, normBuf, txtShift1, txtScale1, numTxtTokens, HiddenDim);
                    ApplyAdaLN(imgTokens, normBuf + numTxtTokens * HiddenDim, imgShift1, imgScale1, numImgTokens, HiddenDim);

                    // QKV projections
                    DispatchMatMulBatch(blk.TxtQkvW, blk.TxtQkvB, blk.D_TxtQkvW, blk.D_TxtQkvB, normBuf, qkvBuf, HiddenDim, 9216, numTxtTokens);
                    DispatchMatMulBatch(blk.ImgQkvW, blk.ImgQkvB, blk.D_ImgQkvW, blk.D_ImgQkvB, normBuf + numTxtTokens * HiddenDim, qkvBuf + numTxtTokens * 9216, HiddenDim, 9216, numImgTokens);

                    // QK-Norm & RoPE & Joint Attention (txt first, img second)
                    JointAttentionDouble(
                        qkvBuf, qkvBuf + numTxtTokens * 9216,
                        blk,
                        ropeCos, ropeSin,
                        numTxtTokens, numImgTokens,
                        attnOut);

                    // Projections & Residual 1 with gate
                    DispatchMatMulBatch(blk.TxtProjW, blk.TxtProjB, blk.D_TxtProjW, blk.D_TxtProjB, attnOut, normBuf, HiddenDim, HiddenDim, numTxtTokens);
                    ApplyResidualGated(txtTokens, normBuf, txtGate1, numTxtTokens, HiddenDim);

                    DispatchMatMulBatch(blk.ImgProjW, blk.ImgProjB, blk.D_ImgProjW, blk.D_ImgProjB, attnOut + numTxtTokens * HiddenDim, normBuf + numTxtTokens * HiddenDim, HiddenDim, HiddenDim, numImgTokens);
                    ApplyResidualGated(imgTokens, normBuf + numTxtTokens * HiddenDim, imgGate1, numImgTokens, HiddenDim);

                    // MLP 2 with gate 2
                    float* txtShift2 = _txtModBuffer + 9216;
                    float* txtScale2 = _txtModBuffer + 12288;
                    float* txtGate2 = _txtModBuffer + 15360;

                    float* imgShift2 = _imgModBuffer + 9216;
                    float* imgScale2 = _imgModBuffer + 12288;
                    float* imgGate2 = _imgModBuffer + 15360;

                    ApplyAdaLN(txtTokens, normBuf, txtShift2, txtScale2, numTxtTokens, HiddenDim);
                    ApplyAdaLN(imgTokens, normBuf + numTxtTokens * HiddenDim, imgShift2, imgScale2, numImgTokens, HiddenDim);

                    // txt mlp
                    DispatchMatMulBatch(blk.TxtMlp0W, blk.TxtMlp0B, blk.D_TxtMlp0W, blk.D_TxtMlp0B, normBuf, qkvBuf, HiddenDim, 12288, numTxtTokens);
                    ApplyGelu(qkvBuf, numTxtTokens * 12288);
                    DispatchMatMulBatch(blk.TxtMlp2W, blk.TxtMlp2B, blk.D_TxtMlp2W, blk.D_TxtMlp2B, qkvBuf, normBuf, 12288, HiddenDim, numTxtTokens);
                    ApplyResidualGated(txtTokens, normBuf, txtGate2, numTxtTokens, HiddenDim);

                    // img mlp
                    DispatchMatMulBatch(blk.ImgMlp0W, blk.ImgMlp0B, blk.D_ImgMlp0W, blk.D_ImgMlp0B, normBuf + numTxtTokens * HiddenDim, qkvBuf + numTxtTokens * 12288, HiddenDim, 12288, numImgTokens);
                    ApplyGelu(qkvBuf + numTxtTokens * 12288, numImgTokens * 12288);
                    DispatchMatMulBatch(blk.ImgMlp2W, blk.ImgMlp2B, blk.D_ImgMlp2W, blk.D_ImgMlp2B, qkvBuf + numTxtTokens * 12288, normBuf + numTxtTokens * HiddenDim, 12288, HiddenDim, numImgTokens);
                    ApplyResidualGated(imgTokens, normBuf + numTxtTokens * HiddenDim, imgGate2, numImgTokens, HiddenDim);
                }
            }
            swDouble.Stop();

            // 6. SingleStreamBlocks (tokens are already contiguous [txt; img] in singleStream = _scratchBuffer)
            var swSingle = System.Diagnostics.Stopwatch.StartNew();
            PrepareSingleStreamGpu();
            if (_gpu != null && _singleBlocksLoaded)
            {
                ExecuteSingleBlocksGpu(singleStream, numTotalTokens, alreadyOnGpu: _doubleBlocksLoaded);
            }
            else
            {
                for (int b = 0; b < SingleBlocksCount; b++)
                {
                    var blk = _singleBlocks[b];

                    // Modulation: [3072] -> [9216] (shift, scale, gate)
                    DispatchLinearVec(blk.ModW, blk.ModB, _siluVecBuffer, _singleModBuffer);
                    float* shift = _singleModBuffer;
                    float* scale = _singleModBuffer + 3072;
                    float* gate = _singleModBuffer + 6144;

                    ApplyAdaLN(singleStream, normBuf, shift, scale, numTotalTokens, HiddenDim);

                    // Fused Linear1: [3072] -> [21504] (QKV 9216 + MLP 12288)
                    DispatchMatMulBatch(blk.Lin1W, blk.Lin1B, blk.D_Lin1W, blk.D_Lin1B, normBuf, qkvBuf, HiddenDim, 21504, numTotalTokens);

                    // Single Stream Attention + MLP
                    SingleAttentionAndMlp(
                        qkvBuf,
                        blk.QueryNorm, blk.KeyNorm,
                        ropeCos, ropeSin,
                        numTotalTokens,
                        attnOut);

                    // Fused Linear2: [15360] -> [3072]
                    DispatchMatMulBatch(blk.Lin2W, blk.Lin2B, blk.D_Lin2W, blk.D_Lin2B, attnOut, normBuf, 15360, HiddenDim, numTotalTokens);

                    // Residual with gate
                    ApplyResidualGated(singleStream, normBuf, gate, numTotalTokens, HiddenDim);
                }
            }
            swSingle.Stop();
            Console.WriteLine($"    [DIT STEP BREAKDOWN] 19 DoubleBlocks: {swDouble.ElapsedMilliseconds} ms, 38 SingleBlocks: {swSingle.ElapsedMilliseconds} ms");

            // 7. Final Layer
            float* patchifiedVel = qkvBuf; // [numImgTokens, 64]
            if (_gpu != null && _dFinalLinearW != IntPtr.Zero && _singleBlocksLoaded)
            {
                ExecuteFinalLayerGpu(numTxtTokens, numImgTokens, patchifiedVel);
            }
            else
            {
                float* finalImgTokens = singleStream + numTxtTokens * HiddenDim;
                DispatchLinearVec(_finalAdaLnW, _finalAdaLnB, _siluVecBuffer, _finalModBuffer);
                float* fShift = _finalModBuffer;
                float* fScale = _finalModBuffer + 3072;

                ApplyAdaLN(finalImgTokens, normBuf, fShift, fScale, numImgTokens, HiddenDim);
                DispatchMatMulBatch(_finalLinearW, _finalLinearB, _dFinalLinearW, _dFinalLinearB, normBuf, patchifiedVel, HiddenDim, PatchDim, numImgTokens);
            }

            // 9. Unpatchify velocity: [numImgTokens, 64] -> outputVelocity [16, latentH, latentW]
            Unpatchify(patchifiedVel, latentH, latentW, outputVelocity);
        }
    }


    private void ComputeConditioningVector(float timestep, ReadOnlySpan<float> pooledY, float* vecOut)
    {
        // 1. Timestep embedding (256 floats)
        float* tSinCos = stackalloc float[256];
        int half = 128;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / half);
            float arg = timestep * 1000.0f * freq;
            tSinCos[i] = MathF.Cos(arg);
            tSinCos[i + half] = MathF.Sin(arg);
        }

        float* tInt = stackalloc float[HiddenDim];
        DispatchLinearVec(_timeIn1W, _timeIn1B, tSinCos, tInt);
        for (int i = 0; i < HiddenDim; i++) tInt[i] = tInt[i] / (1.0f + MathF.Exp(-tInt[i]));
        DispatchLinearVec(_timeIn2W, _timeIn2B, tInt, vecOut);

        // 2. Vector embedding (CLIP pooled y: 768 floats)
        if (_vecIn1W != null && _vecIn2W != null && pooledY.Length >= 768)
        {
            fixed (float* pY = pooledY)
            {
                float* vInt = stackalloc float[HiddenDim];
                DispatchLinearVec(_vecIn1W, _vecIn1B!, pY, vInt);
                for (int i = 0; i < HiddenDim; i++) vInt[i] = vInt[i] / (1.0f + MathF.Exp(-vInt[i]));
                float* vOut = stackalloc float[HiddenDim];
                DispatchLinearVec(_vecIn2W, _vecIn2B!, vInt, vOut);

                for (int i = 0; i < HiddenDim; i++) vecOut[i] += vOut[i];
            }
        }
    }

    private void PatchifyAndProject(ReadOnlySpan<float> latents, int H, int W, float* imgTokens)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;
        int numTokens = tokenH * tokenW;

        // Patchify: rearrange [16, H, W] -> [tokenH * tokenW, 64]
        float* patches = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * PatchDim * sizeof(float)), 64);
        try
        {
            fixed (float* pLat = latents)
            {
                for (int th = 0; th < tokenH; th++)
                {
                    for (int tw = 0; tw < tokenW; tw++)
                    {
                        int tokenIdx = th * tokenW + tw;
                        float* patchRow = patches + tokenIdx * PatchDim;

                        int pIdx = 0;
                        for (int c = 0; c < InChannels; c++)
                        {
                            for (int ph = 0; ph < PatchSize; ph++)
                            {
                                for (int pw = 0; pw < PatchSize; pw++)
                                {
                                    int y = th * PatchSize + ph;
                                    int x = tw * PatchSize + pw;
                                    patchRow[pIdx++] = pLat[c * H * W + y * W + x];
                                }
                            }
                        }
                    }
                }
            }

            // Project: [numTokens, 64] * img_in.weight [64, 3072] -> [numTokens, 3072]
            DispatchMatMulBatch(_imgInW, _imgInB, patches, imgTokens, PatchDim, HiddenDim, numTokens);
        }
        finally
        {
            NativeMemory.AlignedFree(patches);
        }
    }

    private void ProjectTextContext(ReadOnlySpan<float> contextTxt, int numTxtTokens, float* txtTokens)
    {
        if (numTxtTokens <= 0) return;

        fixed (float* pCtx = contextTxt)
        {
            // contextTxt is [numTxtTokens, 4096]. Project: [numTxtTokens, 4096] * txt_in.weight [4096, 3072]
            DispatchMatMulBatch(_txtInW, _txtInB, pCtx, txtTokens, 4096, HiddenDim, numTxtTokens);
        }
    }

    private static void Compute3DRoPE(int numTxtTokens, int tokenH, int tokenW, float* ropeCos, float* ropeSin)
    {
        // axes_dim = {16, 56, 56}, total headDim = 128, halfDim = 64
        int numImgTokens = tokenH * tokenW;
        int numTotalTokens = numTxtTokens + numImgTokens;

        // Axis 0: 16 (half 8), Axis 1: 56 (half 28), Axis 2: 56 (half 28)
        float* omega0 = stackalloc float[8];
        for (int i = 0; i < 8; i++) omega0[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 16.0f);

        float* omega1 = stackalloc float[28];
        for (int i = 0; i < 28; i++) omega1[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 56.0f);

        float* omega2 = stackalloc float[28];
        for (int i = 0; i < 28; i++) omega2[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 56.0f);

        // Text tokens: pos = 0 for all axes -> cos = 1, sin = 0
        for (int t = 0; t < numTxtTokens; t++)
        {
            float* cRow = ropeCos + t * 64;
            float* sRow = ropeSin + t * 64;
            for (int d = 0; d < 64; d++)
            {
                cRow[d] = 1.0f;
                sRow[d] = 0.0f;
            }
        }

        // Image tokens: axis 0 pos = 0, axis 1 pos = row, axis 2 pos = col
        for (int th = 0; th < tokenH; th++)
        {
            for (int tw = 0; tw < tokenW; tw++)
            {
                int imgIdx = th * tokenW + tw;
                int tokenIdx = numTxtTokens + imgIdx;
                float* cRow = ropeCos + tokenIdx * 64;
                float* sRow = ropeSin + tokenIdx * 64;

                // Axis 0 (dim 16, half 8): pos = 0
                for (int d = 0; d < 8; d++)
                {
                    cRow[d] = 1.0f;
                    sRow[d] = 0.0f;
                }
                // Axis 1 (dim 56, half 28): pos = th
                for (int d = 0; d < 28; d++)
                {
                    float angle = th * omega1[d];
                    cRow[8 + d] = MathF.Cos(angle);
                    sRow[8 + d] = MathF.Sin(angle);
                }
                // Axis 2 (dim 56, half 28): pos = tw
                for (int d = 0; d < 28; d++)
                {
                    float angle = tw * omega2[d];
                    cRow[36 + d] = MathF.Cos(angle);
                    sRow[36 + d] = MathF.Sin(angle);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyAdaLN(float* src, float* dst, float* shift, float* scale, int numTokens, int dim)
    {
        for (int t = 0; t < numTokens; t++)
        {
            float* s = src + t * dim;
            float* d = dst + t * dim;

            // Pure LayerNorm: (x - mean) / std (no affine parameters)
            float mean = 0f;
            for (int i = 0; i < dim; i++) mean += s[i];
            mean /= dim;

            float var = 0f;
            for (int i = 0; i < dim; i++)
            {
                float diff = s[i] - mean;
                var += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(var / dim + 1e-6f);

            // Modulate: (1 + scale) * norm(x) + shift
            for (int i = 0; i < dim; i++)
            {
                d[i] = ((s[i] - mean) * invStd) * (1.0f + scale[i]) + shift[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyResidualGated(float* target, float* update, float* gate, int numTokens, int dim)
    {
        for (int t = 0; t < numTokens; t++)
        {
            float* tgt = target + t * dim;
            float* upd = update + t * dim;
            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i] * gate[i];
            }
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

    private void JointAttentionDouble(
        float* txtQkv, float* imgQkv,
        DoubleBlockWeights blk,
        float* ropeCos, float* ropeSin,
        int numTxtTokens, int numImgTokens,
        float* attnOut)
    {
        int numTotalTokens = numTxtTokens + numImgTokens;
        if (_gpu != null && blk.D_TxtQueryNorm != IntPtr.Zero)
        {
            EnsureGpuActivationBuffers(numTotalTokens);

            nuint txtBytes = (nuint)((long)numTxtTokens * 9216 * sizeof(float));
            nuint imgBytes = (nuint)((long)numImgTokens * 9216 * sizeof(float));
            _gpu.CopyToDevice(_dQkvMlpBuf, (IntPtr)txtQkv, txtBytes);
            _gpu.CopyToDevice(_dQkvMlpBuf + (nint)txtBytes, (IntPtr)imgQkv, imgBytes);

            int stride = 9216;
            int nHeads = NumHeads;
            int headDim = HeadDim;
            int txtOffset = 0;
            int imgOffset = numTxtTokens;
            int localNumTxt = numTxtTokens;
            int localNumImg = numImgTokens;
            int localNumTotal = numTotalTokens;

            IntPtr dQkvSrc = _dQkvMlpBuf;
            IntPtr dQ = _dQ;
            IntPtr dK = _dK;
            IntPtr dV = _dV;
            IntPtr dRopeCos = _dRopeCos;
            IntPtr dRopeSin = _dRopeSin;
            IntPtr dTxtQNorm = blk.D_TxtQueryNorm;
            IntPtr dTxtKNorm = blk.D_TxtKeyNorm;
            IntPtr dImgQNorm = blk.D_ImgQueryNorm;
            IntPtr dImgKNorm = blk.D_ImgKeyNorm;
            IntPtr dAttnOut = _dAttnOut;

            // 1. Text QKV Prep: unpack, RMSNorm, RoPE, store into _dQ, _dK, _dV
            void** pTxtArgs = stackalloc void*[13];
            pTxtArgs[0] = &dQkvSrc;
            pTxtArgs[1] = &dQ;
            pTxtArgs[2] = &dK;
            pTxtArgs[3] = &dV;
            pTxtArgs[4] = &dTxtQNorm;
            pTxtArgs[5] = &dTxtKNorm;
            pTxtArgs[6] = &dRopeCos;
            pTxtArgs[7] = &dRopeSin;
            pTxtArgs[8] = &stride;
            pTxtArgs[9] = &nHeads;
            pTxtArgs[10] = &headDim;
            pTxtArgs[11] = &localNumTxt;
            pTxtArgs[12] = &txtOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTxtTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:txt)");

            // 2. Image QKV Prep: unpack, RMSNorm, RoPE, store into _dQ, _dK, _dV
            IntPtr dImgQkvSrc = _dQkvMlpBuf + (nint)txtBytes;
            void** pImgArgs = stackalloc void*[13];
            pImgArgs[0] = &dImgQkvSrc;
            pImgArgs[1] = &dQ;
            pImgArgs[2] = &dK;
            pImgArgs[3] = &dV;
            pImgArgs[4] = &dImgQNorm;
            pImgArgs[5] = &dImgKNorm;
            pImgArgs[6] = &dRopeCos;
            pImgArgs[7] = &dRopeSin;
            pImgArgs[8] = &stride;
            pImgArgs[9] = &nHeads;
            pImgArgs[10] = &headDim;
            pImgArgs[11] = &localNumImg;
            pImgArgs[12] = &imgOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numImgTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:img)");

            // 3. Bidirectional Multi-Head Attention: _dQ, _dK, _dV -> _dAttnOut
            float attnScale = 1.0f / MathF.Sqrt(HeadDim);
            void** pAttnArgs = stackalloc void*[8];
            pAttnArgs[0] = &dQ;
            pAttnArgs[1] = &dK;
            pAttnArgs[2] = &dV;
            pAttnArgs[3] = &dAttnOut;
            pAttnArgs[4] = &nHeads;
            pAttnArgs[5] = &headDim;
            pAttnArgs[6] = &localNumTotal;
            pAttnArgs[7] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBidirectionalBatch, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttnArgs, IntPtr.Zero),
                "LaunchKernel(attention_bidirectional_batch:joint)");

            _gpu.Synchronize();
            _gpu.CopyToHost((IntPtr)attnOut, _dAttnOut, (nuint)((long)numTotalTokens * HiddenDim * sizeof(float)));
            return;
        }

        // CPU Reference Fallback
        float invSqrtHeadDim = 1.0f / MathF.Sqrt(HeadDim);

        float* txtQNormScale = (float*)_gguf.GetTensorPointer(blk.TxtQueryNorm);
        float* txtKNormScale = (float*)_gguf.GetTensorPointer(blk.TxtKeyNorm);
        float* imgQNormScale = (float*)_gguf.GetTensorPointer(blk.ImgQueryNorm);
        float* imgKNormScale = (float*)_gguf.GetTensorPointer(blk.ImgKeyNorm);

        // Pre-normalize Q and K for all tokens and apply RoPE
        float* qAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);
        float* kAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);
        float* vAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            // 1. Unpack text stream (first numTxtTokens)
            for (int t = 0; t < numTxtTokens; t++)
            {
                float* qkv = txtQkv + t * 9216;
                float* qDst = qAll + t * HiddenDim;
                float* kDst = kAll + t * HiddenDim;
                float* vDst = vAll + t * HiddenDim;
                Buffer.MemoryCopy(qkv + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                // QKNorm per head
                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(qkv + h * HeadDim, qDst + h * HeadDim, txtQNormScale);
                    ApplyRMSNormHead(qkv + 3072 + h * HeadDim, kDst + h * HeadDim, txtKNormScale);
                }
            }

            // 2. Unpack image stream (subsequent numImgTokens)
            for (int t = 0; t < numImgTokens; t++)
            {
                int totalIdx = numTxtTokens + t;
                float* qkv = imgQkv + t * 9216;
                float* qDst = qAll + totalIdx * HiddenDim;
                float* kDst = kAll + totalIdx * HiddenDim;
                float* vDst = vAll + totalIdx * HiddenDim;
                Buffer.MemoryCopy(qkv + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                // QKNorm per head
                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(qkv + h * HeadDim, qDst + h * HeadDim, imgQNormScale);
                    ApplyRMSNormHead(qkv + 3072 + h * HeadDim, kDst + h * HeadDim, imgKNormScale);
                }
            }

            // 3. Apply RoPE to Q and K
            for (int t = 0; t < numTotalTokens; t++)
            {
                float* c = ropeCos + t * 64;
                float* s = ropeSin + t * 64;
                for (int h = 0; h < NumHeads; h++)
                {
                    RotateHead(qAll + t * HiddenDim + h * HeadDim, c, s);
                    RotateHead(kAll + t * HiddenDim + h * HeadDim, c, s);
                }
            }

            // 4. Multi-head Attention
            nint qPtr = (nint)qAll;
            nint kPtr = (nint)kAll;
            nint vPtr = (nint)vAll;
            nint outPtr = (nint)attnOut;

            Parallel.For(0, NumHeads, h =>
            {
                float* pQ = (float*)qPtr;
                float* pK = (float*)kPtr;
                float* pV = (float*)vPtr;
                float* pOut = (float*)outPtr;

                int headOffset = h * HeadDim;
                float[] scores = new float[numTotalTokens];

                for (int i = 0; i < numTotalTokens; i++)
                {
                    float* qRow = pQ + i * HiddenDim + headOffset;

                    float maxScore = float.NegativeInfinity;
                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float* kRow = pK + j * HiddenDim + headOffset;
                        float dot = DotHead(qRow, kRow);
                        float sc = dot * invSqrtHeadDim;
                        scores[j] = sc;
                        if (sc > maxScore) maxScore = sc;
                    }

                    // Softmax
                    float sumExp = 0f;
                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float exp = MathF.Exp(scores[j] - maxScore);
                        scores[j] = exp;
                        sumExp += exp;
                    }
                    float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                    // Output
                    float* outRow = pOut + i * HiddenDim + headOffset;
                    for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;

                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float w = scores[j] * invSum;
                        float* vRow = pV + j * HiddenDim + headOffset;
                        AccumulateHead(outRow, vRow, w);
                    }
                }
            });
        }
        finally
        {
            NativeMemory.AlignedFree(qAll);
            NativeMemory.AlignedFree(kAll);
            NativeMemory.AlignedFree(vAll);
        }
    }

    private void ExecuteDoubleBlocksGpu(
        float* txtTokens, float* imgTokens,
        int numTxtTokens, int numImgTokens,
        float* ropeCos, float* ropeSin)
    {
        int numTotalTokens = numTxtTokens + numImgTokens;
        EnsureGpuActivationBuffers(numTotalTokens);

        nuint singleStreamBytes = (nuint)((long)numTotalTokens * HiddenDim * sizeof(float));
        nuint txtBytes = (nuint)((long)numTxtTokens * HiddenDim * sizeof(float));
        nuint imgBytes = (nuint)((long)numImgTokens * HiddenDim * sizeof(float));

        // Upload tokens to GPU once at the start of block 0
        _gpu!.CopyToDevice(_dSingleStream, (IntPtr)txtTokens, singleStreamBytes);

        IntPtr dTxtTokens = _dSingleStream;
        IntPtr dImgTokens = _dSingleStream + (nint)txtBytes;

        IntPtr dNormBufTxt = _dNormBuf;
        IntPtr dNormBufImg = _dNormBuf + (nint)txtBytes;

        nuint txtQkvBytes = (nuint)((long)numTxtTokens * 9216 * sizeof(float));
        IntPtr dTxtQkv = _dQkvMlpBuf;
        IntPtr dImgQkv = _dQkvMlpBuf + (nint)txtQkvBytes;

        int stride = 9216;
        int nHeads = NumHeads;
        int headDim = HeadDim;
        int dim = HiddenDim;
        float attnScale = 1.0f / MathF.Sqrt(HeadDim);

        int localNumTxt = numTxtTokens;
        int localNumImg = numImgTokens;
        int localNumTotal = numTotalTokens;
        int txtOffset = 0;
        int imgOffset = numTxtTokens;

        IntPtr dQ = _dQ;
        IntPtr dK = _dK;
        IntPtr dV = _dV;
        IntPtr dRopeCos = _dRopeCos;
        IntPtr dRopeSin = _dRopeSin;
        IntPtr dAttnOut = _dAttnOut;
        IntPtr dQkvMlpBuf = _dQkvMlpBuf;

        void** pAdaTxtArgs = stackalloc void*[6];
        void** pAdaImgArgs = stackalloc void*[6];
        void** pTxtQkvArgs = stackalloc void*[13];
        void** pImgQkvArgs = stackalloc void*[13];
        void** pAttnArgs = stackalloc void*[8];
        void** pGeluArgs = stackalloc void*[2];
        void** pResTxtArgs = stackalloc void*[5];
        void** pResImgArgs = stackalloc void*[5];

        int totalTxtResElements = numTxtTokens * HiddenDim;
        uint resTxtBlocks = (uint)((totalTxtResElements + 255) / 256);

        int totalImgResElements = numImgTokens * HiddenDim;
        uint resImgBlocks = (uint)((totalImgResElements + 255) / 256);

        for (int b = 0; b < DoubleBlocksCount; b++)
        {
            var blk = _doubleBlocks[b];

            // 1. Modulations computed on CPU, copied to GPU buffers (18432 floats = 72 KB each)
            DispatchLinearVec(blk.ImgModW, blk.ImgModB, _siluVecBuffer, _imgModBuffer);
            DispatchLinearVec(blk.TxtModW, blk.TxtModB, _siluVecBuffer, _txtModBuffer);

            _gpu.CopyToDevice(_dDoubleModImg, (IntPtr)_imgModBuffer, 18432 * sizeof(float));
            _gpu.CopyToDevice(_dDoubleModTxt, (IntPtr)_txtModBuffer, 18432 * sizeof(float));

            IntPtr dTxtShift1 = _dDoubleModTxt;
            IntPtr dTxtScale1 = _dDoubleModTxt + 3072 * sizeof(float);
            IntPtr dTxtGate1  = _dDoubleModTxt + 6144 * sizeof(float);
            IntPtr dTxtShift2 = _dDoubleModTxt + 9216 * sizeof(float);
            IntPtr dTxtScale2 = _dDoubleModTxt + 12288 * sizeof(float);
            IntPtr dTxtGate2  = _dDoubleModTxt + 15360 * sizeof(float);

            IntPtr dImgShift1 = _dDoubleModImg;
            IntPtr dImgScale1 = _dDoubleModImg + 3072 * sizeof(float);
            IntPtr dImgGate1  = _dDoubleModImg + 6144 * sizeof(float);
            IntPtr dImgShift2 = _dDoubleModImg + 9216 * sizeof(float);
            IntPtr dImgScale2 = _dDoubleModImg + 12288 * sizeof(float);
            IntPtr dImgGate2  = _dDoubleModImg + 15360 * sizeof(float);

            // 2. AdaLN 1: txt & img -> normBuf
            pAdaTxtArgs[0] = &dTxtTokens;
            pAdaTxtArgs[1] = &dNormBufTxt;
            pAdaTxtArgs[2] = &dTxtShift1;
            pAdaTxtArgs[3] = &dTxtScale1;
            pAdaTxtArgs[4] = &localNumTxt;
            pAdaTxtArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTxtTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:txt1)");

            pAdaImgArgs[0] = &dImgTokens;
            pAdaImgArgs[1] = &dNormBufImg;
            pAdaImgArgs[2] = &dImgShift1;
            pAdaImgArgs[3] = &dImgScale1;
            pAdaImgArgs[4] = &localNumImg;
            pAdaImgArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:img1)");

            // 3. QKV Projections
            DispatchMatMulBatchDevice(blk.D_TxtQkvW, blk.D_TxtQkvB, dNormBufTxt, dTxtQkv, 3072, 9216, numTxtTokens);
            DispatchMatMulBatchDevice(blk.D_ImgQkvW, blk.D_ImgQkvB, dNormBufImg, dImgQkv, 3072, 9216, numImgTokens);

            // 4. QKV Prep: unpack, RMSNorm, 3D-RoPE into _dQ, _dK, _dV
            IntPtr dTxtQNorm = blk.D_TxtQueryNorm;
            IntPtr dTxtKNorm = blk.D_TxtKeyNorm;
            pTxtQkvArgs[0] = &dTxtQkv;
            pTxtQkvArgs[1] = &dQ;
            pTxtQkvArgs[2] = &dK;
            pTxtQkvArgs[3] = &dV;
            pTxtQkvArgs[4] = &dTxtQNorm;
            pTxtQkvArgs[5] = &dTxtKNorm;
            pTxtQkvArgs[6] = &dRopeCos;
            pTxtQkvArgs[7] = &dRopeSin;
            pTxtQkvArgs[8] = &stride;
            pTxtQkvArgs[9] = &nHeads;
            pTxtQkvArgs[10] = &headDim;
            pTxtQkvArgs[11] = &localNumTxt;
            pTxtQkvArgs[12] = &txtOffset;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTxtTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pTxtQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:txt)");

            IntPtr dImgQNorm = blk.D_ImgQueryNorm;
            IntPtr dImgKNorm = blk.D_ImgKeyNorm;
            pImgQkvArgs[0] = &dImgQkv;
            pImgQkvArgs[1] = &dQ;
            pImgQkvArgs[2] = &dK;
            pImgQkvArgs[3] = &dV;
            pImgQkvArgs[4] = &dImgQNorm;
            pImgQkvArgs[5] = &dImgKNorm;
            pImgQkvArgs[6] = &dRopeCos;
            pImgQkvArgs[7] = &dRopeSin;
            pImgQkvArgs[8] = &stride;
            pImgQkvArgs[9] = &nHeads;
            pImgQkvArgs[10] = &headDim;
            pImgQkvArgs[11] = &localNumImg;
            pImgQkvArgs[12] = &imgOffset;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numImgTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pImgQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:img)");

            // 5. FlashAttention-2 Joint Attention -> _dAttnOut [numTotalTokens, 3072]
            pAttnArgs[0] = &dQ;
            pAttnArgs[1] = &dK;
            pAttnArgs[2] = &dV;
            pAttnArgs[3] = &dAttnOut;
            pAttnArgs[4] = &nHeads;
            pAttnArgs[5] = &headDim;
            pAttnArgs[6] = &localNumTotal;
            pAttnArgs[7] = &attnScale;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBidirectionalBatch, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttnArgs, IntPtr.Zero),
                "LaunchKernel(attention_bidirectional_batch:joint)");

            // 6. Attention Output Projections & Residual 1
            IntPtr dAttnOutImg = _dAttnOut + (nint)txtBytes;
            DispatchMatMulBatchDevice(blk.D_TxtProjW, blk.D_TxtProjB, dAttnOut, dNormBufTxt, 3072, 3072, numTxtTokens);
            pResTxtArgs[0] = &dTxtTokens;
            pResTxtArgs[1] = &dNormBufTxt;
            pResTxtArgs[2] = &dTxtGate1;
            pResTxtArgs[3] = &totalTxtResElements;
            pResTxtArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:txt1)");

            DispatchMatMulBatchDevice(blk.D_ImgProjW, blk.D_ImgProjB, dAttnOutImg, dNormBufImg, 3072, 3072, numImgTokens);
            pResImgArgs[0] = &dImgTokens;
            pResImgArgs[1] = &dNormBufImg;
            pResImgArgs[2] = &dImgGate1;
            pResImgArgs[3] = &totalImgResElements;
            pResImgArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:img1)");

            // 7. AdaLN 2: txt & img -> normBuf
            pAdaTxtArgs[0] = &dTxtTokens;
            pAdaTxtArgs[1] = &dNormBufTxt;
            pAdaTxtArgs[2] = &dTxtShift2;
            pAdaTxtArgs[3] = &dTxtScale2;
            pAdaTxtArgs[4] = &localNumTxt;
            pAdaTxtArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTxtTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:txt2)");

            pAdaImgArgs[0] = &dImgTokens;
            pAdaImgArgs[1] = &dNormBufImg;
            pAdaImgArgs[2] = &dImgShift2;
            pAdaImgArgs[3] = &dImgScale2;
            pAdaImgArgs[4] = &localNumImg;
            pAdaImgArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:img2)");

            // 8. Txt MLP: [3072] -> [12288] -> GeLU -> [3072] + Residual
            DispatchMatMulBatchDevice(blk.D_TxtMlp0W, blk.D_TxtMlp0B, dNormBufTxt, _dQkvMlpBuf, 3072, 12288, numTxtTokens);
            int txtMlpCount = numTxtTokens * 12288;
            uint geluTxtBlocks = (uint)((txtMlpCount + 255) / 256);
            pGeluArgs[0] = &dQkvMlpBuf;
            pGeluArgs[1] = &txtMlpCount;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGelu, geluTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_gelu_kernel:txt)");
            DispatchMatMulBatchDevice(blk.D_TxtMlp2W, blk.D_TxtMlp2B, _dQkvMlpBuf, dNormBufTxt, 12288, 3072, numTxtTokens);
            pResTxtArgs[0] = &dTxtTokens;
            pResTxtArgs[1] = &dNormBufTxt;
            pResTxtArgs[2] = &dTxtGate2;
            pResTxtArgs[3] = &totalTxtResElements;
            pResTxtArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:txt2)");

            // 9. Img MLP: [3072] -> [12288] -> GeLU -> [3072] + Residual
            DispatchMatMulBatchDevice(blk.D_ImgMlp0W, blk.D_ImgMlp0B, dNormBufImg, _dQkvMlpBuf, 3072, 12288, numImgTokens);
            int imgMlpCount = numImgTokens * 12288;
            uint geluImgBlocks = (uint)((imgMlpCount + 255) / 256);
            pGeluArgs[0] = &dQkvMlpBuf;
            pGeluArgs[1] = &imgMlpCount;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGelu, geluImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_gelu_kernel:img)");
            DispatchMatMulBatchDevice(blk.D_ImgMlp2W, blk.D_ImgMlp2B, _dQkvMlpBuf, dNormBufImg, 12288, 3072, numImgTokens);
            pResImgArgs[0] = &dImgTokens;
            pResImgArgs[1] = &dNormBufImg;
            pResImgArgs[2] = &dImgGate2;
            pResImgArgs[3] = &totalImgResElements;
            pResImgArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:img2)");
        }

        if (!_fullGpuMode)
        {
            _gpu!.Synchronize();
            _gpu!.CopyToHost((IntPtr)txtTokens, _dSingleStream, singleStreamBytes);
        }
    }

    private void ExecuteSingleBlocksGpu(float* singleStream, int numTotalTokens, bool alreadyOnGpu = false)
    {
        EnsureGpuActivationBuffers(numTotalTokens);

        nuint singleStreamBytes = (nuint)((long)numTotalTokens * HiddenDim * sizeof(float));
        if (!alreadyOnGpu)
        {
            _gpu!.CopyToDevice(_dSingleStream, (IntPtr)singleStream, singleStreamBytes);
        }

        float attnScale = 1.0f / MathF.Sqrt(HeadDim);
        int stride1 = 21504;
        int nHeads = NumHeads;
        int headDim = HeadDim;
        int dim = HiddenDim;
        int localTotalTokens = numTotalTokens;

        IntPtr dSingleStream = _dSingleStream;
        IntPtr dNormBuf = _dNormBuf;
        IntPtr dQkvMlpBuf = _dQkvMlpBuf;
        IntPtr dAttnOut = _dAttnOut;
        IntPtr dConcatBuf = _dConcatBuf;
        IntPtr dQ = _dQ;
        IntPtr dK = _dK;
        IntPtr dV = _dV;
        IntPtr dRopeCos = _dRopeCos;
        IntPtr dRopeSin = _dRopeSin;
        IntPtr dSingleMod = _dSingleMod;

        void** pAdaArgs = stackalloc void*[6];
        void** pQkvArgs = stackalloc void*[13];
        void** pAttnArgs = stackalloc void*[8];
        void** pGeluArgs = stackalloc void*[4];
        void** pResArgs = stackalloc void*[5];

        for (int b = 0; b < SingleBlocksCount; b++)
        {
            var blk = _singleBlocks[b];

            // 1. Modulation vector on CPU -> copy 36KB to GPU
            DispatchLinearVec(blk.ModW, blk.ModB, _siluVecBuffer, _singleModBuffer);
            _gpu!.CopyToDevice(dSingleMod, (IntPtr)_singleModBuffer, 9216 * sizeof(float));

            IntPtr dShift = dSingleMod;
            IntPtr dScale = dSingleMod + 3072 * sizeof(float);
            IntPtr dGate = dSingleMod + 6144 * sizeof(float);

            // 2. AdaLN: _dSingleStream -> _dNormBuf
            pAdaArgs[0] = &dSingleStream;
            pAdaArgs[1] = &dNormBuf;
            pAdaArgs[2] = &dShift;
            pAdaArgs[3] = &dScale;
            pAdaArgs[4] = &localTotalTokens;
            pAdaArgs[5] = &dim;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTotalTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel)");

            // 3. Fused Linear1: _dNormBuf [N, 3072] -> _dQkvMlpBuf [N, 21504]
            DispatchMatMulBatchDevice(blk.D_Lin1W, blk.D_Lin1B, dNormBuf, dQkvMlpBuf, 3072, 21504, numTotalTokens);

            // 4. Fused QKV Unpack + RMSNorm + 3D-RoPE: _dQkvMlpBuf -> _dQ, _dK, _dV
            int tokenOffset = 0;
            IntPtr dQueryNorm = blk.D_QueryNorm;
            IntPtr dKeyNorm = blk.D_KeyNorm;
            pQkvArgs[0] = &dQkvMlpBuf;
            pQkvArgs[1] = &dQ;
            pQkvArgs[2] = &dK;
            pQkvArgs[3] = &dV;
            pQkvArgs[4] = &dQueryNorm;
            pQkvArgs[5] = &dKeyNorm;
            pQkvArgs[6] = &dRopeCos;
            pQkvArgs[7] = &dRopeSin;
            pQkvArgs[8] = &stride1;
            pQkvArgs[9] = &nHeads;
            pQkvArgs[10] = &headDim;
            pQkvArgs[11] = &localTotalTokens;
            pQkvArgs[12] = &tokenOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:single)");

            // 5. FlashAttention-2: _dQ, _dK, _dV -> _dAttnOut [N, 3072]
            pAttnArgs[0] = &dQ;
            pAttnArgs[1] = &dK;
            pAttnArgs[2] = &dV;
            pAttnArgs[3] = &dAttnOut;
            pAttnArgs[4] = &nHeads;
            pAttnArgs[5] = &headDim;
            pAttnArgs[6] = &localTotalTokens;
            pAttnArgs[7] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBidirectionalBatch, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttnArgs, IntPtr.Zero),
                "LaunchKernel(attention_bidirectional_batch:single)");

            // 6. Fused GELU + Concat: _dAttnOut [N, 3072] + GELU(_dQkvMlpBuf[:, 9216..]) -> _dConcatBuf [N, 15360]
            int totalConcatElements = numTotalTokens * 15360;
            uint concatBlocks = (uint)((totalConcatElements + 255) / 256);
            pGeluArgs[0] = &dAttnOut;
            pGeluArgs[1] = &dQkvMlpBuf;
            pGeluArgs[2] = &dConcatBuf;
            pGeluArgs[3] = &localTotalTokens;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGeluConcat, concatBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_fused_gelu_concat)");

            // 7. Fused Linear2: _dConcatBuf [N, 15360] -> _dNormBuf [N, 3072]
            DispatchMatMulBatchDevice(blk.D_Lin2W, blk.D_Lin2B, dConcatBuf, dNormBuf, 15360, 3072, numTotalTokens);

            // 8. Residual Gated: _dSingleStream += _dNormBuf * gate
            int totalResElements = numTotalTokens * HiddenDim;
            uint resBlocks = (uint)((totalResElements + 255) / 256);
            pResArgs[0] = &dSingleStream;
            pResArgs[1] = &dNormBuf;
            pResArgs[2] = &dGate;
            pResArgs[3] = &totalResElements;
            pResArgs[4] = &dim;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated)");
        }

        // Copy back to host only if final layer is not going to execute on GPU
        if (!_fullGpuMode || _dFinalLinearW == IntPtr.Zero)
        {
            _gpu!.Synchronize();
            _gpu!.CopyToHost((IntPtr)singleStream, _dSingleStream, singleStreamBytes);
        }
    }

    private void ExecuteFinalLayerGpu(int numTxtTokens, int numImgTokens, float* patchifiedVel)
    {
        // 1. Final modulation vector on CPU (6144 floats = 24 KB)
        DispatchLinearVec(_finalAdaLnW, _finalAdaLnB, _siluVecBuffer, _finalModBuffer);
        _gpu!.CopyToDevice(_dSingleMod, (IntPtr)_finalModBuffer, 6144 * sizeof(float));

        IntPtr dShift = _dSingleMod;
        IntPtr dScale = _dSingleMod + 3072 * sizeof(float);

        IntPtr dImgTokens = _dSingleStream + (nint)((long)numTxtTokens * HiddenDim * sizeof(float));
        IntPtr dNormBuf = _dNormBuf;
        int dim = HiddenDim;
        int localImgTokens = numImgTokens;

        void** pAdaArgs = stackalloc void*[6];
        pAdaArgs[0] = &dImgTokens;
        pAdaArgs[1] = &dNormBuf;
        pAdaArgs[2] = &dShift;
        pAdaArgs[3] = &dScale;
        pAdaArgs[4] = &localImgTokens;
        pAdaArgs[5] = &dim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaArgs, IntPtr.Zero),
            "LaunchKernel(flux_adaln_kernel:final)");

        // 2. Final Linear: [numImgTokens, 3072] * [3072, 64] -> [numImgTokens, 64]
        DispatchMatMulBatchDevice(_dFinalLinearW, _dFinalLinearB, dNormBuf, _dFinalOut, 3072, PatchDim, numImgTokens);

        nuint velBytes = (nuint)((long)numImgTokens * PatchDim * sizeof(float));
        _gpu.Synchronize();
        _gpu.CopyToHost((IntPtr)patchifiedVel, _dFinalOut, velBytes);
    }

    private void SingleAttentionAndMlp(
        float* qkvMlp,
        GgufTensorInfo qNorm, GgufTensorInfo kNorm,
        float* ropeCos, float* ropeSin,
        int numTokens,
        float* attnOutMlp)
    {
        float invSqrtHeadDim = 1.0f / MathF.Sqrt(HeadDim);
        float* qNormScale = (float*)_gguf.GetTensorPointer(qNorm);
        float* kNormScale = (float*)_gguf.GetTensorPointer(kNorm);

        float* qAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);
        float* kAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);
        float* vAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            for (int t = 0; t < numTokens; t++)
            {
                float* row = qkvMlp + t * 21504;
                float* qDst = qAll + t * HiddenDim;
                float* kDst = kAll + t * HiddenDim;
                float* vDst = vAll + t * HiddenDim;
                Buffer.MemoryCopy(row + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(row + h * HeadDim, qDst + h * HeadDim, qNormScale);
                    ApplyRMSNormHead(row + 3072 + h * HeadDim, kDst + h * HeadDim, kNormScale);
                }

                // RoPE
                float* c = ropeCos + t * 64;
                float* s = ropeSin + t * 64;
                for (int h = 0; h < NumHeads; h++)
                {
                    RotateHead(qDst + h * HeadDim, c, s);
                    RotateHead(kDst + h * HeadDim, c, s);
                }
            }

            // Multi-head Attention
            nint qPtr = (nint)qAll;
            nint kPtr = (nint)kAll;
            nint vPtr = (nint)vAll;
            nint outPtr = (nint)attnOutMlp;

            Parallel.For(0, NumHeads, h =>
            {
                float* pQ = (float*)qPtr;
                float* pK = (float*)kPtr;
                float* pV = (float*)vPtr;
                float* pOut = (float*)outPtr;

                int headOffset = h * HeadDim;
                float[] scores = new float[numTokens];

                for (int i = 0; i < numTokens; i++)
                {
                    float* qRow = pQ + i * HiddenDim + headOffset;

                    float maxScore = float.NegativeInfinity;
                    for (int j = 0; j < numTokens; j++)
                    {
                        float* kRow = pK + j * HiddenDim + headOffset;
                        float dot = DotHead(qRow, kRow);
                        float sc = dot * invSqrtHeadDim;
                        scores[j] = sc;
                        if (sc > maxScore) maxScore = sc;
                    }

                    float sumExp = 0f;
                    for (int j = 0; j < numTokens; j++)
                    {
                        float exp = MathF.Exp(scores[j] - maxScore);
                        scores[j] = exp;
                        sumExp += exp;
                    }
                    float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                    float* outRow = pOut + i * 15360 + headOffset;
                    for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;

                    for (int j = 0; j < numTokens; j++)
                    {
                        float w = scores[j] * invSum;
                        float* vRow = pV + j * HiddenDim + headOffset;
                        AccumulateHead(outRow, vRow, w);
                    }
                }
            });

            // Fused MLP activation: copy activated MLP (12288) to position [3072..15359]
            nint qkvMlpPtr = (nint)qkvMlp;
            nint attnOutMlpPtr = (nint)attnOutMlp;

            Parallel.For(0, numTokens, t =>
            {
                float* mlpSrc = (float*)qkvMlpPtr + t * 21504 + 9216;
                float* mlpDst = (float*)attnOutMlpPtr + t * 15360 + 3072;
                ApplyGelu(mlpSrc, 12288);
                Buffer.MemoryCopy(mlpSrc, mlpDst, 12288 * sizeof(float), 12288 * sizeof(float));
            });
        }
        finally
        {
            NativeMemory.AlignedFree(qAll);
            NativeMemory.AlignedFree(kAll);
            NativeMemory.AlignedFree(vAll);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotHead(float* a, float* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vSum0 = Vector256<float>.Zero;
            var vSum1 = Vector256<float>.Zero;
            for (int d = 0; d < 128; d += 16)
            {
                vSum0 += Vector256.Load(a + d) * Vector256.Load(b + d);
                vSum1 += Vector256.Load(a + d + 8) * Vector256.Load(b + d + 8);
            }
            return Vector256.Sum(vSum0 + vSum1);
        }
        float dot = 0f;
        for (int d = 0; d < 128; d++) dot += a[d] * b[d];
        return dot;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateHead(float* dst, float* v, float w)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vw = Vector256.Create(w);
            for (int d = 0; d < 128; d += 8)
            {
                var res = Vector256.Load(dst + d) + vw * Vector256.Load(v + d);
                res.Store(dst + d);
            }
            return;
        }
        for (int d = 0; d < 128; d++) dst[d] += w * v[d];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRMSNormHead(float* src, float* dst, float* scale)
    {
        float sumSq = 0f;
        for (int i = 0; i < HeadDim; i++) sumSq += src[i] * src[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / HeadDim + 1e-6f);
        for (int i = 0; i < HeadDim; i++) dst[i] = src[i] * invRms * scale[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHead(float* vec, float* cos, float* sin)
    {
        for (int i = 0; i < 64; i++)
        {
            float v0 = vec[2 * i];
            float v1 = vec[2 * i + 1];
            float c = cos[i];
            float s = sin[i];
            vec[2 * i] = v0 * c - v1 * s;
            vec[2 * i + 1] = v0 * s + v1 * c;
        }
    }

    private void Unpatchify(float* patches, int H, int W, Span<float> output)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;

        fixed (float* pOut = output)
        {
            for (int th = 0; th < tokenH; th++)
            {
                for (int tw = 0; tw < tokenW; tw++)
                {
                    int tokenIdx = th * tokenW + tw;
                    float* patchRow = patches + tokenIdx * PatchDim;

                    int pIdx = 0;
                    for (int c = 0; c < InChannels; c++)
                    {
                        for (int ph = 0; ph < PatchSize; ph++)
                        {
                            for (int pw = 0; pw < PatchSize; pw++)
                            {
                                int y = th * PatchSize + ph;
                                int x = tw * PatchSize + pw;
                                pOut[c * H * W + y * W + x] = patchRow[pIdx++];
                            }
                        }
                    }
                }
            }
        }
    }

    private void DispatchLinearVec(GgufTensorInfo w, GgufTensorInfo b, float* x, float* y)
    {
        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = (float*)_gguf.GetTensorPointer(b);
        int nCols = (int)w.Dimensions[0];
        int nRows = (int)w.Dimensions[1];

        QuantKernels.MatVecMul(w.Type, wPtr, x, y, nCols, nRows);
        for (int i = 0; i < nRows; i++) y[i] += bPtr[i];
    }

    private IntPtr UploadVector(GgufTensorInfo t, int count)
    {
        if (_gpu == null) return IntPtr.Zero;
        nuint bytes = (nuint)((long)count * sizeof(float));
        IntPtr d = _gpu.AllocateDevice(bytes);
        _gpu.CopyToDevice(d, (IntPtr)_gguf.GetTensorPointer(t), bytes);
        return d;
    }

    private (IntPtr dW, IntPtr dB) UploadWeightAndBias(GgufTensorInfo w, GgufTensorInfo b, int nRows)
    {
        if (_gpu == null || w.Type != GgufType.Q4_K) return (IntPtr.Zero, IntPtr.Zero);

        nuint wBytes = (nuint)w.GetByteSize();
        nuint bBytes = (nuint)(nRows * sizeof(float));

        IntPtr dW = _gpu.AllocateDevice(wBytes);
        IntPtr dB = _gpu.AllocateDevice(bBytes);

        _gpu.CopyToDevice(dW, (IntPtr)_gguf.GetTensorPointer(w), wBytes);
        _gpu.CopyToDevice(dB, (IntPtr)_gguf.GetTensorPointer(b), bBytes);

        return (dW, dB);
    }

    private void FreeWeightAndBias(ref IntPtr dW, ref IntPtr dB)
    {
        if (_gpu != null)
        {
            if (dW != IntPtr.Zero) { _gpu.FreeDevice(dW); dW = IntPtr.Zero; }
            if (dB != IntPtr.Zero) { _gpu.FreeDevice(dB); dB = IntPtr.Zero; }
        }
    }

    private void UploadDoubleBlocks()
    {
        if (_gpu == null || _doubleBlocksLoaded) return;
        for (int i = 0; i < DoubleBlocksCount; i++)
        {
            var blk = _doubleBlocks[i];
            (blk.D_ImgQkvW, blk.D_ImgQkvB) = UploadWeightAndBias(blk.ImgQkvW, blk.ImgQkvB, 9216);
            (blk.D_TxtQkvW, blk.D_TxtQkvB) = UploadWeightAndBias(blk.TxtQkvW, blk.TxtQkvB, 9216);
            (blk.D_ImgProjW, blk.D_ImgProjB) = UploadWeightAndBias(blk.ImgProjW, blk.ImgProjB, HiddenDim);
            (blk.D_TxtProjW, blk.D_TxtProjB) = UploadWeightAndBias(blk.TxtProjW, blk.TxtProjB, HiddenDim);
            (blk.D_ImgMlp0W, blk.D_ImgMlp0B) = UploadWeightAndBias(blk.ImgMlp0W, blk.ImgMlp0B, 12288);
            (blk.D_ImgMlp2W, blk.D_ImgMlp2B) = UploadWeightAndBias(blk.ImgMlp2W, blk.ImgMlp2B, HiddenDim);
            (blk.D_TxtMlp0W, blk.D_TxtMlp0B) = UploadWeightAndBias(blk.TxtMlp0W, blk.TxtMlp0B, 12288);
            (blk.D_TxtMlp2W, blk.D_TxtMlp2B) = UploadWeightAndBias(blk.TxtMlp2W, blk.TxtMlp2B, HiddenDim);
            blk.D_ImgQueryNorm = UploadVector(blk.ImgQueryNorm, HeadDim);
            blk.D_ImgKeyNorm = UploadVector(blk.ImgKeyNorm, HeadDim);
            blk.D_TxtQueryNorm = UploadVector(blk.TxtQueryNorm, HeadDim);
            blk.D_TxtKeyNorm = UploadVector(blk.TxtKeyNorm, HeadDim);
        }
        _doubleBlocksLoaded = true;
    }

    private void FreeDoubleBlocks()
    {
        if (_gpu == null || !_doubleBlocksLoaded) return;
        for (int i = 0; i < DoubleBlocksCount; i++)
        {
            var blk = _doubleBlocks[i];
            FreeWeightAndBias(ref blk.D_ImgQkvW, ref blk.D_ImgQkvB);
            FreeWeightAndBias(ref blk.D_TxtQkvW, ref blk.D_TxtQkvB);
            FreeWeightAndBias(ref blk.D_ImgProjW, ref blk.D_ImgProjB);
            FreeWeightAndBias(ref blk.D_TxtProjW, ref blk.D_TxtProjB);
            FreeWeightAndBias(ref blk.D_ImgMlp0W, ref blk.D_ImgMlp0B);
            FreeWeightAndBias(ref blk.D_ImgMlp2W, ref blk.D_ImgMlp2B);
            FreeWeightAndBias(ref blk.D_TxtMlp0W, ref blk.D_TxtMlp0B);
            FreeWeightAndBias(ref blk.D_TxtMlp2W, ref blk.D_TxtMlp2B);
            if (blk.D_ImgQueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_ImgQueryNorm); blk.D_ImgQueryNorm = IntPtr.Zero; }
            if (blk.D_ImgKeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_ImgKeyNorm); blk.D_ImgKeyNorm = IntPtr.Zero; }
            if (blk.D_TxtQueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_TxtQueryNorm); blk.D_TxtQueryNorm = IntPtr.Zero; }
            if (blk.D_TxtKeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_TxtKeyNorm); blk.D_TxtKeyNorm = IntPtr.Zero; }
        }
        _doubleBlocksLoaded = false;
    }

    private void UploadSingleBlocks()
    {
        if (_gpu == null || _singleBlocksLoaded) return;
        for (int i = 0; i < SingleBlocksCount; i++)
        {
            var blk = _singleBlocks[i];
            (blk.D_Lin1W, blk.D_Lin1B) = UploadWeightAndBias(blk.Lin1W, blk.Lin1B, 21504);
            (blk.D_Lin2W, blk.D_Lin2B) = UploadWeightAndBias(blk.Lin2W, blk.Lin2B, HiddenDim);
            blk.D_QueryNorm = UploadVector(blk.QueryNorm, HeadDim);
            blk.D_KeyNorm = UploadVector(blk.KeyNorm, HeadDim);
        }
        _singleBlocksLoaded = true;
    }

    private void FreeSingleBlocks()
    {
        if (_gpu == null || !_singleBlocksLoaded) return;
        for (int i = 0; i < SingleBlocksCount; i++)
        {
            var blk = _singleBlocks[i];
            FreeWeightAndBias(ref blk.D_Lin1W, ref blk.D_Lin1B);
            FreeWeightAndBias(ref blk.D_Lin2W, ref blk.D_Lin2B);
            if (blk.D_QueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_QueryNorm); blk.D_QueryNorm = IntPtr.Zero; }
            if (blk.D_KeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_KeyNorm); blk.D_KeyNorm = IntPtr.Zero; }
        }
        _singleBlocksLoaded = false;
    }

    private void EnsureGpuActivationBuffers(int numTotalTokens)
    {
        if (_gpu == null || _allocatedGpuTokens >= numTotalTokens) return;
        FreeGpuActivationBuffers();

        int n = numTotalTokens;
        _dSingleStream = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dNormBuf = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dQkvMlpBuf = _gpu.AllocateDevice((nuint)((long)n * 21504 * sizeof(float)));
        _dAttnOut = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dConcatBuf = _gpu.AllocateDevice((nuint)((long)n * 15360 * sizeof(float)));
        _dQ = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dK = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dV = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dRopeCos = _gpu.AllocateDevice((nuint)((long)n * 64 * sizeof(float)));
        _dRopeSin = _gpu.AllocateDevice((nuint)((long)n * 64 * sizeof(float)));
        _dSingleMod = _gpu.AllocateDevice((nuint)(9216 * sizeof(float)));
        _dDoubleModImg = _gpu.AllocateDevice((nuint)(18432 * sizeof(float)));
        _dDoubleModTxt = _gpu.AllocateDevice((nuint)(18432 * sizeof(float)));
        _dFinalOut = _gpu.AllocateDevice((nuint)((long)n * PatchDim * sizeof(float)));

        _allocatedGpuTokens = n;
    }

    private void FreeGpuActivationBuffers()
    {
        if (_gpu != null)
        {
            if (_dSingleStream != IntPtr.Zero) { _gpu.FreeDevice(_dSingleStream); _dSingleStream = IntPtr.Zero; }
            if (_dNormBuf != IntPtr.Zero) { _gpu.FreeDevice(_dNormBuf); _dNormBuf = IntPtr.Zero; }
            if (_dQkvMlpBuf != IntPtr.Zero) { _gpu.FreeDevice(_dQkvMlpBuf); _dQkvMlpBuf = IntPtr.Zero; }
            if (_dAttnOut != IntPtr.Zero) { _gpu.FreeDevice(_dAttnOut); _dAttnOut = IntPtr.Zero; }
            if (_dConcatBuf != IntPtr.Zero) { _gpu.FreeDevice(_dConcatBuf); _dConcatBuf = IntPtr.Zero; }
            if (_dQ != IntPtr.Zero) { _gpu.FreeDevice(_dQ); _dQ = IntPtr.Zero; }
            if (_dK != IntPtr.Zero) { _gpu.FreeDevice(_dK); _dK = IntPtr.Zero; }
            if (_dV != IntPtr.Zero) { _gpu.FreeDevice(_dV); _dV = IntPtr.Zero; }
            if (_dRopeCos != IntPtr.Zero) { _gpu.FreeDevice(_dRopeCos); _dRopeCos = IntPtr.Zero; }
            if (_dRopeSin != IntPtr.Zero) { _gpu.FreeDevice(_dRopeSin); _dRopeSin = IntPtr.Zero; }
            if (_dSingleMod != IntPtr.Zero) { _gpu.FreeDevice(_dSingleMod); _dSingleMod = IntPtr.Zero; }
            if (_dDoubleModImg != IntPtr.Zero) { _gpu.FreeDevice(_dDoubleModImg); _dDoubleModImg = IntPtr.Zero; }
            if (_dDoubleModTxt != IntPtr.Zero) { _gpu.FreeDevice(_dDoubleModTxt); _dDoubleModTxt = IntPtr.Zero; }
            if (_dFinalOut != IntPtr.Zero) { _gpu.FreeDevice(_dFinalOut); _dFinalOut = IntPtr.Zero; }
        }
        _allocatedGpuTokens = 0;
    }

    private void PrepareDoubleStreamGpu()
    {
        if (_gpu == null || _fullGpuMode) return;
        FreeSingleBlocks();
        UploadDoubleBlocks();
    }

    private void PrepareSingleStreamGpu()
    {
        if (_gpu == null || _fullGpuMode) return;
        FreeDoubleBlocks();
        UploadSingleBlocks();
    }

    private void EnsureGpuScratchBuffers(int batchSize, int nCols, int nRows)
    {
        nuint neededX = (nuint)((long)batchSize * nCols * sizeof(float));
        nuint neededY = (nuint)((long)batchSize * nRows * sizeof(float));

        if (neededX > _dXDeviceCapacity)
        {
            if (_dXDevice != IntPtr.Zero) _gpu!.FreeDevice(_dXDevice);
            _dXDeviceCapacity = (nuint)(neededX * 12 / 10 + 1024 * 1024);
            _dXDevice = _gpu!.AllocateDevice(_dXDeviceCapacity);
        }

        if (neededY > _dYDeviceCapacity)
        {
            if (_dYDevice != IntPtr.Zero) _gpu!.FreeDevice(_dYDevice);
            _dYDeviceCapacity = (nuint)(neededY * 12 / 10 + 1024 * 1024);
            _dYDevice = _gpu!.AllocateDevice(_dYDeviceCapacity);
        }
    }

    private void DispatchMatMulBatch(GgufTensorInfo w, GgufTensorInfo b, float* xBatch, float* yBatch, int nCols, int nRows, int batchSize)
        => DispatchMatMulBatch(w, b, IntPtr.Zero, IntPtr.Zero, xBatch, yBatch, nCols, nRows, batchSize);

    private void DispatchMatMulBatch(
        GgufTensorInfo w, GgufTensorInfo b,
        IntPtr dW, IntPtr dB,
        float* xBatch, float* yBatch,
        int nCols, int nRows, int batchSize)
    {
        if (_gpu != null && dW != IntPtr.Zero && w.Type == GgufType.Q4_K)
        {
            EnsureGpuScratchBuffers(batchSize, nCols, nRows);

            nuint xBytes = (nuint)((long)batchSize * nCols * sizeof(float));
            nuint yBytes = (nuint)((long)batchSize * nRows * sizeof(float));

            _gpu.CopyToDevice(_dXDevice, (IntPtr)xBatch, xBytes);

            uint blockSize = 128;
            uint numWarps = 4;
            uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
            uint gridY = (uint)((batchSize + 15) / 16);
            IntPtr dResidual = IntPtr.Zero;

            IntPtr dY = _dYDevice;
            IntPtr dX = _dXDevice;
            int localCols = nCols;
            int localRows = nRows;
            int localBatch = batchSize;
            IntPtr localW = dW;
            IntPtr localB = dB;

            void** pArgs = stackalloc void*[8];
            pArgs[0] = &dY;
            pArgs[1] = &dX;
            pArgs[2] = &localW;
            pArgs[3] = &localCols;
            pArgs[4] = &localRows;
            pArgs[5] = &localBatch;
            pArgs[6] = &localB;
            pArgs[7] = &dResidual;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGemmQ4KBatch,
                gridX, gridY, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(gemm_q4_k_batch)");

            _gpu.Synchronize();
            _gpu.CopyToHost((IntPtr)yBatch, _dYDevice, yBytes);
            return;
        }

        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = (float*)_gguf.GetTensorPointer(b);

        QuantKernels.MatMulBatch(w.Type, wPtr, xBatch, yBatch, nCols, nRows, batchSize);

        // Add bias across batch
        nint yPtr = (nint)yBatch;
        nint bVal = (nint)bPtr;

        Parallel.For(0, batchSize, bIdx =>
        {
            float* row = (float*)yPtr + (long)bIdx * nRows;
            float* bArr = (float*)bVal;
            for (int r = 0; r < nRows; r++) row[r] += bArr[r];
        });
    }

    private void DispatchMatMulBatchDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);
        IntPtr dResidual = IntPtr.Zero;

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

    public void Dispose()
    {
        if (!_disposed)
        {
            FreeDoubleBlocks();
            FreeSingleBlocks();
            FreeGpuActivationBuffers();
            FreeWeightAndBias(ref _dFinalLinearW, ref _dFinalLinearB);

            if (_dXDevice != IntPtr.Zero) { _gpu?.FreeDevice(_dXDevice); _dXDevice = IntPtr.Zero; }
            if (_dYDevice != IntPtr.Zero) { _gpu?.FreeDevice(_dYDevice); _dYDevice = IntPtr.Zero; }
            if (_gpuModule != IntPtr.Zero) { CuDriver.ModuleUnload(_gpuModule); _gpuModule = IntPtr.Zero; }
            _gpu?.Dispose();

            if (_scratchBuffer != null) NativeMemory.AlignedFree(_scratchBuffer);
            if (_imgModBuffer != null) NativeMemory.AlignedFree(_imgModBuffer);
            if (_txtModBuffer != null) NativeMemory.AlignedFree(_txtModBuffer);
            if (_singleModBuffer != null) NativeMemory.AlignedFree(_singleModBuffer);
            if (_finalModBuffer != null) NativeMemory.AlignedFree(_finalModBuffer);
            if (_vecBuffer != null) NativeMemory.AlignedFree(_vecBuffer);
            if (_siluVecBuffer != null) NativeMemory.AlignedFree(_siluVecBuffer);

            _gguf.Dispose();
            _disposed = true;
        }
    }
}

