namespace Glacier.Inference.Image.Flux;

using Glacier.Inference.Diagnostics;
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
public sealed unsafe partial class FluxDiT : IDisposable
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
    private bool _gpuWeightsResident => _gpu != null;
    public string ActiveBackend => _gpu != null && _gpuWeightsResident ? "Cuda" : "Cpu";

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
                GlacierDiagnostics.LogInformation($"[GLACIER GPU] Detected {_gpu.DeviceName} ({_gpu.ArchString}) with {freeMb} MB free VRAM.");

                if (freeMb >= 4800)
                {
                    GlacierDiagnostics.LogInformation("[GLACIER GPU] Uploading FLUX.1 DiT weights into GPU VRAM for zero-transfer inference...");
                    UploadDoubleBlocks();
                    UploadSingleBlocks();
                    (_dFinalLinearW, _dFinalLinearB) = UploadWeightAndBias(_finalLinearW, _finalLinearB, PatchDim);
                    _fullGpuMode = true;
                    GlacierDiagnostics.LogInformation("[GLACIER GPU] DiT weights resident in GPU VRAM (Full VRAM Mode).");
                }
                else if (freeMb >= 3700)
                {
                    GlacierDiagnostics.LogInformation("[GLACIER GPU] Operating in two-phase dynamic GPU offload mode.");
                    _fullGpuMode = false;
                }
                else
                {
                    GlacierDiagnostics.LogInformation("[GLACIER GPU] Insufficient free VRAM for GPU offload, falling back to CPU SIMD.");
                    _gpu.Dispose();
                    _gpu = null;
                }
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER GPU] GPU initialization skipped [{ex.GetType().Name}]: {ex.Message}, falling back to CPU SIMD.", ex);
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
            GlacierDiagnostics.LogInformation($"    [DIT STEP BREAKDOWN] {_doubleBlocks.Length} DoubleBlocks: {swDouble.ElapsedMilliseconds} ms, {_singleBlocks.Length} SingleBlocks: {swSingle.ElapsedMilliseconds} ms");

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

