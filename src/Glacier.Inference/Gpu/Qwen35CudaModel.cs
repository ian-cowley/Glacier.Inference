namespace Glacier.Inference.Gpu;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# Bare-Metal CUDA (<c>nvcuda.dll</c> / <c>libcuda.so.1</c>) inference runner for Qwen 3.5 / 3.6 / 3.8
/// Hybrid Gated DeltaNet + Full-Attention models quantized with NariQ Trellis-Coded Quantization (<c>TRL1..TRL6</c>).
/// All 64 layers + 16-band mixed-bitrate LM head run asynchronously on a single CUDA stream with zero host syncs
/// until logits download.
/// </summary>
public sealed unsafe class Qwen35CudaModel : ID3D12Model
{
    private const int HeadDim = 256, StateDim = 128, ConvKernel = 4;

    private sealed class TrellisBandedGpu
    {
        public required TrellisCudaRunner.DeviceMatrix[] Bands;
        public required IntPtr VocabGather;
        public int TotalN;
    }

    private sealed class MatGpu
    {
        public TrellisCudaRunner.DeviceMatrix? Trellis;
        public TrellisBandedGpu? Banded;
    }

    private sealed class LayerGpu
    {
        public bool IsGdn;
        public IntPtr AttnNorm, FfnNorm;
        // GDN
        public MatGpu? Qkv, Gate, SsmOut;
        public IntPtr QkvBias, AlphaW, BetaW, ConvW, ConvState, AW, DtBias, SsmNorm, RecState;
        public nuint ConvStateBytes, RecStateBytes;
        // Full Attention
        public MatGpu? QW, KW, VW, OW;
        public IntPtr QNorm, KNorm, KCache, VCache, QBias, KBias, VBias, OBias;
        // Dense FFN
        public MatGpu? GW, UW, DW;
        public IntPtr UBias;
        public int FfnDim;
    }

    private readonly ModelWeights _w;
    private readonly GpuContext _ctx;
    private readonly TrellisCudaRunner _trl;
    private readonly IntPtr _module;
    private readonly IntPtr _stream;
    private readonly IntPtr _stream2;
    private readonly IntPtr _stream3;
    private readonly IntPtr _evFork;
    private readonly IntPtr _evJoin;
    private readonly IntPtr _evJoin3;
    private readonly IntPtr _evAlphaBeta;
    private readonly List<IntPtr> _allocs = new();
    private readonly List<TrellisCudaRunner.DeviceMatrix> _trlMats = new();

    // Kernels from qwen35.cuh + trellis.cuh
    private readonly IntPtr _fnRms, _fnAddRms, _fnSwiglu, _fnAdd, _fnF32Mv;
    private readonly IntPtr _fnConv, _fnL2, _fnRec, _fnPrep, _fnDec;
    private readonly IntPtr _fnTrlGather;

    // Shared scratch buffers
    private readonly int _dim, _vocab, _maxSeq, _nH, _nKv, _qDim, _kvDim, _nK, _nV, _ch, _inner, _ffn;
    private readonly float _attnScale;
    private readonly IntPtr _dX, _dNorm, _dProj, _dQkv, _dZ, _dAlpha, _dBeta, _dConvOut, _dY, _dGated;
    private readonly IntPtr _dQFull, _dK, _dV, _dQ, _dGate, _dAttnOut, _dScores;
    private readonly IntPtr _dGateOut, _dUpOut, _dAct;
    private readonly IntPtr _dTrlXRot, _dTrlYRot, _dTrlYScaled, _dTrlBandRaw, _dLogits, _dPos;
    private readonly IntPtr _dTrlXRot2, _dTrlYRot2, _dTrlYScaled2;
    private readonly IntPtr _dTrlXRot3, _dTrlYRot3, _dTrlYScaled3;
    private readonly float[] _hostEmb;

    private readonly IntPtr _dOutNorm;
    private readonly MatGpu _outMat;
    private readonly LayerGpu[] _layers;
    private IntPtr _graphExecLogits;
    private IntPtr _graphExecNoLogits;
    private bool _graphDisabled;
    private bool _disposed;

    public string DeviceName => _ctx.DeviceName;
    public long StateSnapshotBytes { get; }

    public static bool IsSupported(ModelWeights w, out string reason)
    {
        if (!w.HasTrellisWeights)
        {
            reason = "Model does not use NariQ trellis weights.";
            return false;
        }
        if (!w.IsHybridSsm)
        {
            reason = "Not a Qwen 3.5/3.6/3.8 Hybrid SSM model.";
            return false;
        }
        if (w.IsMoe)
        {
            reason = "Qwen35CudaModel currently supports dense FFN NariQ models (not MoE).";
            return false;
        }
        if (w.HeadDim != HeadDim || w.SsmStateSize != StateDim || w.SsmConvKernel != ConvKernel)
        {
            reason = $"Unsupported Qwen35 dimensions (headDim={w.HeadDim}, stateDim={w.SsmStateSize}, convKernel={w.SsmConvKernel}).";
            return false;
        }
        if (!GpuContext.IsSupported)
        {
            reason = "CUDA driver not available.";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    public Qwen35CudaModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Qwen35CudaModel cannot execute this model: {why}");

        _w = weights;
        _maxSeq = maxSeqLen;
        _dim = weights.EmbeddingLength;
        _vocab = weights.VocabSize;
        _nH = weights.HeadCount;
        _nKv = weights.HeadCountKv;
        _qDim = _nH * HeadDim;
        _kvDim = _nKv * HeadDim;
        _nK = weights.SsmGroupCount;
        _nV = weights.SsmTimeStepRank;
        _ch = (2 * _nK + _nV) * StateDim;
        _inner = weights.SsmInnerSize;
        _ffn = weights.FeedForwardLength;
        _attnScale = 1f / MathF.Sqrt(HeadDim);
        _hostEmb = new float[_dim];

        _ctx = new GpuContext(deviceOrdinal);
        _ctx.MakeCurrent();
        _trl = new TrellisCudaRunner(_ctx, ownsContext: false);

        CuDriver.Check(CuDriver.StreamCreate(out _stream, 0), "cuStreamCreate");
        CuDriver.Check(CuDriver.StreamCreate(out _stream2, 0), "cuStreamCreate(2)");
        CuDriver.Check(CuDriver.StreamCreate(out _stream3, 0), "cuStreamCreate(3)");
        CuDriver.Check(CuDriver.EventCreate(out _evFork, 2), "cuEventCreate(fork)"); // CU_EVENT_DISABLE_TIMING = 2
        CuDriver.Check(CuDriver.EventCreate(out _evJoin, 2), "cuEventCreate(join)");
        CuDriver.Check(CuDriver.EventCreate(out _evJoin3, 2), "cuEventCreate(join3)");
        CuDriver.Check(CuDriver.EventCreate(out _evAlphaBeta, 2), "cuEventCreate(ab)");

        byte[] cubin = KernelCompiler.GetOrCompileKernels(_ctx.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), "cuModuleLoadData(qwen35)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRms, _module, "q35_rmsnorm"), "get(q35_rmsnorm)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddRms, _module, "q35_add_rmsnorm"), "get(q35_add_rmsnorm)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "q35_swiglu"), "get(q35_swiglu)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAdd, _module, "q35_add_inplace"), "get(q35_add_inplace)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnF32Mv, _module, "q35_f32_matvec"), "get(q35_f32_matvec)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnConv, _module, "q35_gdn_conv"), "get(q35_gdn_conv)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnL2, _module, "q35_gdn_l2"), "get(q35_gdn_l2)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRec, _module, "q35_gdn_rec"), "get(q35_gdn_rec)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPrep, _module, "q35_attn_prep"), "get(q35_attn_prep)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnDec, _module, "q35_attn_decode"), "get(q35_attn_decode)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnTrlGather, _module, "trellis_gather_f32"), "get(trellis_gather_f32)");

        try
        {
            // Shared scratch buffers
            _dX = Alloc((long)_dim * 4);
            _dNorm = Alloc((long)_dim * 4);
            _dProj = Alloc((long)_dim * 4);
            _dQkv = Alloc((long)_ch * 4);
            _dZ = Alloc((long)_inner * 4);
            _dAlpha = Alloc((long)_nV * 4);
            _dBeta = Alloc((long)_nV * 4);
            _dConvOut = Alloc((long)_ch * 4);
            _dY = Alloc((long)_inner * 4);
            _dGated = Alloc((long)_inner * 4);
            _dQFull = Alloc(2L * _qDim * 4);
            _dK = Alloc((long)_kvDim * 4);
            _dV = Alloc((long)_kvDim * 4);
            _dQ = Alloc((long)_qDim * 4);
            _dGate = Alloc((long)_qDim * 4);
            _dAttnOut = Alloc((long)_qDim * 4);
            _dScores = Alloc((long)_nH * _maxSeq * 4);
            _dGateOut = Alloc((long)_ffn * 4);
            _dUpOut = Alloc((long)_ffn * 4);
            _dAct = Alloc((long)_ffn * 4);

            int trlMax = Math.Max(Math.Max(Math.Max(_dim, _ffn), Math.Max(_ch, 2 * _qDim)), 20480);
            _dTrlXRot = Alloc((long)trlMax * 4);
            _dTrlYRot = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlYScaled = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlXRot2 = Alloc((long)trlMax * 4);
            _dTrlYRot2 = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlYScaled2 = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlXRot3 = Alloc((long)trlMax * 4);
            _dTrlYRot3 = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlYScaled3 = Alloc((long)Math.Max(trlMax, _vocab) * 4);
            _dTrlBandRaw = weights.Gguf.TrellisLmHeadBands > 0 ? Alloc((long)_vocab * 4) : IntPtr.Zero;
            _dLogits = Alloc((long)_vocab * 4);
            _dPos = AllocZero(16);

            // Output head
            _dOutNorm = UploadF32(weights.OutNormWeight, _dim);
            _outMat = UploadTrellisMat(weights.OutWeight, weights.OutType);

            // Layers
            int nL = weights.BlockCount;
            _layers = new LayerGpu[nL];
            long snapBytes = 0;
            for (int l = 0; l < nL; l++)
            {
                if (l % 8 == 0 || l == nL - 1)
                    GlacierDiagnostics.LogInformation($"[CUDA GPU] Uploading {weights.Gguf.Architecture} NariQ weights: layer {l + 1}/{nL}...");

                var L = weights.Layers[l];
                var lg = new LayerGpu
                {
                    IsGdn = L.IsGdn,
                    AttnNorm = UploadF32(L.AttnNormWeight, _dim),
                    FfnNorm = UploadF32(L.FfnNormWeight, _dim),
                    FfnDim = _ffn,
                };

                if (L.IsGdn)
                {
                    lg.Qkv = UploadTrellisMat(L.QkvWeight, L.QkvType);
                    if (L.QkvBias != null) lg.QkvBias = UploadF32(L.QkvBias, _ch);
                    lg.Gate = UploadTrellisMat(L.AttnGateWeight, L.AttnGateType);
                    lg.AlphaW = UploadSmallMatrixAsF32(L.SsmAlphaWeight, L.SsmAlphaType, _dim, _nV);
                    lg.BetaW = UploadSmallMatrixAsF32(L.SsmBetaWeight, L.SsmBetaType, _dim, _nV);
                    lg.ConvW = UploadF32(L.SsmConv1dWeight, (long)_ch * ConvKernel);
                    lg.AW = UploadF32(L.SsmAWeight, _nV);
                    lg.DtBias = UploadF32(L.SsmDtBias, _nV);
                    lg.SsmNorm = UploadF32(L.SsmNormWeight, StateDim);
                    lg.SsmOut = UploadTrellisMat(L.SsmOutWeight, L.SsmOutType);

                    lg.ConvStateBytes = (nuint)((long)_ch * 3L * 4);
                    lg.RecStateBytes = (nuint)((long)_nV * StateDim * StateDim * 4);
                    lg.ConvState = AllocZero((long)lg.ConvStateBytes);
                    lg.RecState = AllocZero((long)lg.RecStateBytes);
                    snapBytes += (long)lg.ConvStateBytes + (long)lg.RecStateBytes;
                }
                else
                {
                    lg.QW = UploadTrellisMat(L.QWeight, L.QType);
                    lg.KW = UploadTrellisMat(L.KWeight, L.KType);
                    lg.VW = UploadTrellisMat(L.VWeight, L.VType);
                    lg.OW = UploadTrellisMat(L.AttnOutWeight, L.AttnOutType);
                    lg.QNorm = UploadF32(L.AttnQNormWeight, HeadDim);
                    lg.KNorm = UploadF32(L.AttnKNormWeight, HeadDim);
                    if (L.QBias != null) lg.QBias = UploadF32(L.QBias, 2 * _qDim);
                    if (L.KBias != null) lg.KBias = UploadF32(L.KBias, _kvDim);
                    if (L.VBias != null) lg.VBias = UploadF32(L.VBias, _kvDim);
                    if (L.AttnOutBias != null) lg.OBias = UploadF32(L.AttnOutBias, _dim);
                    lg.KCache = AllocZero((long)_nKv * _maxSeq * HeadDim * 4);
                    lg.VCache = AllocZero((long)_nKv * _maxSeq * HeadDim * 4);
                }

                lg.GW = UploadTrellisMat(L.FfnGateWeight, L.FfnGateType);
                lg.UW = UploadTrellisMat(L.FfnUpWeight, L.FfnUpType);
                lg.DW = UploadTrellisMat(L.FfnDownWeight, L.FfnDownType);
                if (L.FfnUpBias != null) lg.UBias = UploadF32(L.FfnUpBias, _ffn);

                _layers[l] = lg;
            }
            StateSnapshotBytes = snapBytes;
            GlacierDiagnostics.LogInformation($"[CUDA GPU] NariQ Qwen35 ready on {_ctx.DeviceName}: {nL} layers loaded in pure GDDR6 VRAM.");
            if (Environment.GetEnvironmentVariable("GLACIER_CUDA_PROFILE") == "1")
            {
                RunCudaProfileBreakdown();
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private IntPtr Alloc(long bytes)
    {
        IntPtr p = _ctx.AllocateDevice((nuint)Math.Max(bytes, 16));
        _allocs.Add(p);
        return p;
    }

    private IntPtr AllocZero(long bytes)
    {
        IntPtr p = Alloc(bytes);
        CuDriver.Check(CuDriver.MemsetD8(p, 0, (nuint)bytes), "cuMemsetD8");
        return p;
    }

    private IntPtr UploadF32(void* src, long count)
    {
        if (src == null) throw new InvalidOperationException("Missing F32 tensor.");
        nuint bytes = (nuint)(count * 4);
        IntPtr dst = Alloc((long)bytes);
        CuDriver.Check(CuDriver.MemcpyHtoD(dst, (IntPtr)src, bytes), "HtoD(F32)");
        return dst;
    }

    private IntPtr UploadSmallMatrixAsF32(byte* p, GgufType type, int k, int n)
    {
        if (p == null) throw new InvalidOperationException("Missing small matrix tensor.");
        if (type == GgufType.F32) return UploadF32(p, (long)n * k);
        var rows = new float[(long)n * k];
        fixed (float* dst = rows)
        {
            for (int r = 0; r < n; r++)
                QuantKernels.ExtractEmbedding(type, p, r, dst + (long)r * k, k);
            return UploadF32(dst, rows.Length);
        }
    }

    private MatGpu UploadTrellisMat(byte* p, GgufType type)
    {
        if (p == null) throw new InvalidOperationException("Missing trellis weight tensor.");
        if (!GgufTypes.IsTrellis(type))
            throw new NotSupportedException($"Qwen35CudaModel expects NariQ trellis weight, got {type}.");

        if (TrellisWeightRegistry.TryGetBanded(p, out var banded) && banded != null)
        {
            var bands = new TrellisCudaRunner.DeviceMatrix[banded.Bands.Length];
            for (int i = 0; i < banded.Bands.Length; i++)
            {
                bands[i] = _trl.Upload(banded.Bands[i]);
                _trlMats.Add(bands[i]);
            }
            nuint gBytes = (nuint)((long)banded.TotalN * sizeof(int));
            IntPtr dGather = Alloc((long)gBytes);
            CuDriver.Check(CuDriver.MemcpyHtoD(dGather, (IntPtr)banded.VocabGather, gBytes), "HtoD(vocab_gather)");
            return new MatGpu { Banded = new TrellisBandedGpu { Bands = bands, VocabGather = dGather, TotalN = banded.TotalN } };
        }

        if (!TrellisWeightRegistry.TryGet(p, out var binding) || binding == null)
            throw new InvalidOperationException($"NariQ trellis weight ({type}) not found in TrellisWeightRegistry.");

        var dm = _trl.Upload(binding);
        _trlMats.Add(dm);
        return new MatGpu { Trellis = dm };
    }

    // ------------------------------------------------------------------ Kernel Launch Helpers

    private void LaunchRms(IntPtr x, IntPtr w, IntPtr y, int size, int rows)
    {
        float eps = _w.RmsNormEps;
        IntPtr* args = stackalloc IntPtr[6];
        args[0] = (IntPtr)(&x); args[1] = (IntPtr)(&w); args[2] = (IntPtr)(&y);
        args[3] = (IntPtr)(&size); args[4] = (IntPtr)(&rows); args[5] = (IntPtr)(&eps);
        CuDriver.Check(CuDriver.LaunchKernel(_fnRms, (uint)rows, 1, 1, 256, 1, 1, 0, _stream, (IntPtr)args, IntPtr.Zero), "q35_rmsnorm");
    }

    private void LaunchAddRms(IntPtr x, IntPtr residual, IntPtr w, IntPtr y, int size)
    {
        float eps = _w.RmsNormEps;
        IntPtr* args = stackalloc IntPtr[6];
        args[0] = (IntPtr)(&x); args[1] = (IntPtr)(&residual); args[2] = (IntPtr)(&w);
        args[3] = (IntPtr)(&y); args[4] = (IntPtr)(&size); args[5] = (IntPtr)(&eps);
        CuDriver.Check(CuDriver.LaunchKernel(_fnAddRms, 1, 1, 1, 256, 1, 1, 0, _stream, (IntPtr)args, IntPtr.Zero), "q35_add_rmsnorm");
    }

    private void LaunchSwiglu(IntPtr gate, IntPtr up, IntPtr output, int n)
    {
        IntPtr* args = stackalloc IntPtr[4];
        args[0] = (IntPtr)(&gate); args[1] = (IntPtr)(&up); args[2] = (IntPtr)(&output); args[3] = (IntPtr)(&n);
        CuDriver.Check(CuDriver.LaunchKernel(_fnSwiglu, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, _stream, (IntPtr)args, IntPtr.Zero), "q35_swiglu");
    }

    private void LaunchAdd(IntPtr a, IntPtr b, int n)
    {
        IntPtr* args = stackalloc IntPtr[3];
        args[0] = (IntPtr)(&a); args[1] = (IntPtr)(&b); args[2] = (IntPtr)(&n);
        CuDriver.Check(CuDriver.LaunchKernel(_fnAdd, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, _stream, (IntPtr)args, IntPtr.Zero), "q35_add");
    }

    private void LaunchF32Mv(IntPtr w, IntPtr x, IntPtr y, int k, int n, IntPtr stream = default)
    {
        IntPtr st = stream == IntPtr.Zero ? _stream : stream;
        IntPtr* args = stackalloc IntPtr[5];
        args[0] = (IntPtr)(&w); args[1] = (IntPtr)(&x); args[2] = (IntPtr)(&y);
        args[3] = (IntPtr)(&k); args[4] = (IntPtr)(&n);
        CuDriver.Check(CuDriver.LaunchKernel(_fnF32Mv, (uint)((n + 3) / 4), 1, 1, 128, 1, 1, 0, st, (IntPtr)args, IntPtr.Zero), "q35_f32_matvec");
    }

    private void LaunchTrellisMv(MatGpu m, IntPtr x, IntPtr y, int streamIdx = 0)
    {
        IntPtr st = streamIdx == 2 ? _stream3 : (streamIdx == 1 ? _stream2 : _stream);
        IntPtr xRot = streamIdx == 2 ? _dTrlXRot3 : (streamIdx == 1 ? _dTrlXRot2 : _dTrlXRot);
        IntPtr yRot = streamIdx == 2 ? _dTrlYRot3 : (streamIdx == 1 ? _dTrlYRot2 : _dTrlYRot);
        IntPtr ySc = streamIdx == 2 ? _dTrlYScaled3 : (streamIdx == 1 ? _dTrlYScaled2 : _dTrlYScaled);

        if (m.Banded != null)
        {
            // Run 16 LM head bands 3-way concurrently across _stream, _stream2, _stream3 inside the CUDA Graph!
            CuDriver.Check(CuDriver.EventRecord(_evFork, _stream), "fork(lm)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream2, _evFork, 0), "wait(lm2)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream3, _evFork, 0), "wait(lm3)");
            int offset = 0;
            for (int b = 0; b < m.Banded.Bands.Length; b++)
            {
                var band = m.Banded.Bands[b];
                int si = b % 3;
                IntPtr bs = si == 2 ? _stream3 : (si == 1 ? _stream2 : _stream);
                IntPtr bx = si == 2 ? _dTrlXRot3 : (si == 1 ? _dTrlXRot2 : _dTrlXRot);
                IntPtr by = si == 2 ? _dTrlYRot3 : (si == 1 ? _dTrlYRot2 : _dTrlYRot);
                IntPtr bys = band.HasPout ? (si == 2 ? _dTrlYScaled3 : (si == 1 ? _dTrlYScaled2 : _dTrlYScaled)) : _dTrlBandRaw;
                _trl.DispatchDevice(band, x, bx, by, bys, _dTrlBandRaw, outOffset: offset, stream: bs);
                offset += band.N;
            }
            CuDriver.Check(CuDriver.EventRecord(_evJoin, _stream2), "join(lm2)");
            CuDriver.Check(CuDriver.EventRecord(_evJoin3, _stream3), "join(lm3)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin, 0), "waitjoin(lm2)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin3, 0), "waitjoin(lm3)");

            IntPtr src = _dTrlBandRaw, gather = m.Banded.VocabGather, dst = y;
            int totalN = m.Banded.TotalN;
            IntPtr* gArgs = stackalloc IntPtr[4];
            gArgs[0] = (IntPtr)(&src); gArgs[1] = (IntPtr)(&gather); gArgs[2] = (IntPtr)(&dst); gArgs[3] = (IntPtr)(&totalN);
            CuDriver.Check(CuDriver.LaunchKernel(_fnTrlGather, (uint)((totalN + 255) / 256), 1, 1, 256, 1, 1, 0, _stream, (IntPtr)gArgs, IntPtr.Zero), "trellis_vocab_gather");
            return;
        }

        var tm = m.Trellis!;
        IntPtr ys = tm.HasPout ? ySc : y;
        _trl.DispatchDevice(tm, x, xRot, yRot, ys, y, outOffset: 0, stream: st);
    }

    private void LaunchTrellisSwigluMv(MatGpu m, IntPtr gate, IntPtr up, IntPtr y)
    {
        var tm = m.Trellis!;
        IntPtr ys = tm.HasPout ? _dTrlYScaled : y;
        _trl.DispatchDeviceSwiglu(tm, gate, up, _dTrlXRot, _dTrlYRot, ys, y, outOffset: 0, stream: _stream);
    }

    // ------------------------------------------------------------------ Forward

    private void EnqueueForwardPass(bool wantLogits)
    {
        int ropeDim = _w.RopeDimensionCount;
        float eps = _w.RmsNormEps;
        float freqBase = _w.RopeFreqBase;
        IntPtr* a = stackalloc IntPtr[16];

        if (_layers.Length > 0)
            LaunchRms(_dX, _layers[0].AttnNorm, _dNorm, _dim, 1);

        for (int l = 0; l < _layers.Length; l++)
        {
            var L = _layers[l];

            if (L.IsGdn)
            {
                // Overlap Qkv + Conv + L2 (_stream) || AlphaW + BetaW + Gate (_stream2) inside the CUDA Graph!
                // AlphaW & BetaW finish during Qkv, hiding 96 F32 matvecs behind Qkv!
                CuDriver.Check(CuDriver.EventRecord(_evFork, _stream), "fork(gdn)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream2, _evFork, 0), "wait(gdn)");
                LaunchTrellisMv(L.Qkv!, _dNorm, _dQkv, streamIdx: 0);
                LaunchF32Mv(L.AlphaW, _dNorm, _dAlpha, _dim, _nV, _stream2);
                LaunchF32Mv(L.BetaW, _dNorm, _dBeta, _dim, _nV, _stream2);
                CuDriver.Check(CuDriver.EventRecord(_evAlphaBeta, _stream2), "rec(ab)");
                LaunchTrellisMv(L.Gate!, _dNorm, _dZ, streamIdx: 1);
                if (L.QkvBias != IntPtr.Zero) LaunchAdd(_dQkv, L.QkvBias, _ch);

                // gdn_conv
                {
                    IntPtr qkv = _dQkv, cw = L.ConvW, cs = L.ConvState, co = _dConvOut;
                    int ch = _ch;
                    a[0] = (IntPtr)(&qkv); a[1] = (IntPtr)(&cw); a[2] = (IntPtr)(&cs); a[3] = (IntPtr)(&co); a[4] = (IntPtr)(&ch);
                    CuDriver.Check(CuDriver.LaunchKernel(_fnConv, (uint)((ch + 255) / 256), 1, 1, 256, 1, 1, 0, _stream, (IntPtr)a, IntPtr.Zero), "q35_gdn_conv");
                }
                // gdn_l2
                {
                    IntPtr co = _dConvOut;
                    int nK = _nK;
                    a[0] = (IntPtr)(&co); a[1] = (IntPtr)(&nK);
                    CuDriver.Check(CuDriver.LaunchKernel(_fnL2, (uint)nK, 1, 1, 128, 1, 1, 0, _stream, (IntPtr)a, IntPtr.Zero), "q35_gdn_l2");
                }
                // Ensure Alpha & Beta are ready before gdn_rec (already completed during Qkv!)
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evAlphaBeta, 0), "wait(ab)");
                // gdn_rec
                {
                    IntPtr co = _dConvOut, al = _dAlpha, be = _dBeta, dt = L.DtBias, aw = L.AW, st = L.RecState, dy = _dY;
                    int nK = _nK, nV = _nV;
                    a[0] = (IntPtr)(&co); a[1] = (IntPtr)(&al); a[2] = (IntPtr)(&be); a[3] = (IntPtr)(&dt);
                    a[4] = (IntPtr)(&aw); a[5] = (IntPtr)(&st); a[6] = (IntPtr)(&dy); a[7] = (IntPtr)(&nK); a[8] = (IntPtr)(&nV);
                    CuDriver.Check(CuDriver.LaunchKernel(_fnRec, (uint)nV, 1, 1, 128, 1, 1, 0, _stream, (IntPtr)a, IntPtr.Zero), "q35_gdn_rec");
                }

                LaunchRms(_dY, L.SsmNorm, _dY, StateDim, _nV);
                CuDriver.Check(CuDriver.EventRecord(_evJoin, _stream2), "join(gdn)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin, 0), "waitjoin(gdn)");
                // Fused SwiGLU(_dZ, _dY) + Pre-RHT + GEMV + Post-RHT for SsmOut!
                LaunchTrellisSwigluMv(L.SsmOut!, _dZ, _dY, _dProj);
                // Fused Residual Add + FfnNorm!
                LaunchAddRms(_dX, _dProj, L.FfnNorm, _dNorm, _dim);
            }
            else
            {
                // 3-way parallel QW (_stream) || KW (_stream2) || VW (_stream3) inside the CUDA Graph!
                CuDriver.Check(CuDriver.EventRecord(_evFork, _stream), "fork(attn)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream2, _evFork, 0), "wait(attn2)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream3, _evFork, 0), "wait(attn3)");
                LaunchTrellisMv(L.QW!, _dNorm, _dQFull, streamIdx: 0);
                LaunchTrellisMv(L.KW!, _dNorm, _dK, streamIdx: 1);
                LaunchTrellisMv(L.VW!, _dNorm, _dV, streamIdx: 2);
                CuDriver.Check(CuDriver.EventRecord(_evJoin, _stream2), "join(attn2)");
                CuDriver.Check(CuDriver.EventRecord(_evJoin3, _stream3), "join(attn3)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin, 0), "waitjoin(attn2)");
                CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin3, 0), "waitjoin(attn3)");
                if (L.QBias != IntPtr.Zero) LaunchAdd(_dQFull, L.QBias, 2 * _qDim);
                if (L.KBias != IntPtr.Zero) LaunchAdd(_dK, L.KBias, _kvDim);
                if (L.VBias != IntPtr.Zero) LaunchAdd(_dV, L.VBias, _kvDim);

                // attn_prep (reads pos from _dPos)
                {
                    IntPtr qf = _dQFull, kr = _dK, vr = _dV, qn = L.QNorm, kn = L.KNorm, qo = _dQ, go = _dGate, kc = L.KCache, vc = L.VCache, dPos = _dPos;
                    int nQ = _nH, nKv = _nKv, maxS = _maxSeq, rDim = ropeDim;
                    float fEps = eps, fBase = freqBase;
                    a[0] = (IntPtr)(&qf); a[1] = (IntPtr)(&kr); a[2] = (IntPtr)(&vr); a[3] = (IntPtr)(&qn);
                    a[4] = (IntPtr)(&kn); a[5] = (IntPtr)(&qo); a[6] = (IntPtr)(&go); a[7] = (IntPtr)(&kc);
                    a[8] = (IntPtr)(&vc); a[9] = (IntPtr)(&dPos); a[10] = (IntPtr)(&nQ); a[11] = (IntPtr)(&nKv);
                    a[12] = (IntPtr)(&maxS); a[13] = (IntPtr)(&rDim); a[14] = (IntPtr)(&fEps); a[15] = (IntPtr)(&fBase);
                    CuDriver.Check(CuDriver.LaunchKernel(_fnPrep, (uint)(_nH + 2 * _nKv), 1, 1, 256, 1, 1, 0, _stream, (IntPtr)a, IntPtr.Zero), "q35_attn_prep");
                }
                // attn_decode (reads kvLen = *d_pos + 1 from _dPos)
                {
                    IntPtr qo = _dQ, go = _dGate, kc = L.KCache, vc = L.VCache, sc = _dScores, ao = _dAttnOut, dPos = _dPos;
                    int nQ = _nH, nKv = _nKv, maxS = _maxSeq;
                    float scale = _attnScale;
                    a[0] = (IntPtr)(&qo); a[1] = (IntPtr)(&go); a[2] = (IntPtr)(&kc); a[3] = (IntPtr)(&vc);
                    a[4] = (IntPtr)(&sc); a[5] = (IntPtr)(&ao); a[6] = (IntPtr)(&dPos); a[7] = (IntPtr)(&nQ);
                    a[8] = (IntPtr)(&nKv); a[9] = (IntPtr)(&maxS); a[10] = (IntPtr)(&scale);
                    CuDriver.Check(CuDriver.LaunchKernel(_fnDec, (uint)_nH, 1, 1, 256, 1, 1, 0, _stream, (IntPtr)a, IntPtr.Zero), "q35_attn_decode");
                }

                LaunchTrellisMv(L.OW!, _dAttnOut, _dProj);
                if (L.OBias != IntPtr.Zero) LaunchAdd(_dProj, L.OBias, _dim);
                // Fused Residual Add + FfnNorm!
                LaunchAddRms(_dX, _dProj, L.FfnNorm, _dNorm, _dim);
            }

            // Dense SwiGLU FFN: overlap GW (_stream) || UW (_stream2) inside the CUDA Graph
            CuDriver.Check(CuDriver.EventRecord(_evFork, _stream), "fork(ffn)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream2, _evFork, 0), "wait(ffn)");
            LaunchTrellisMv(L.GW!, _dNorm, _dGateOut, streamIdx: 0);
            LaunchTrellisMv(L.UW!, _dNorm, _dUpOut, streamIdx: 1);
            CuDriver.Check(CuDriver.EventRecord(_evJoin, _stream2), "join(ffn)");
            CuDriver.Check(CuDriver.StreamWaitEvent(_stream, _evJoin, 0), "waitjoin(ffn)");
            if (L.UBias != IntPtr.Zero) LaunchAdd(_dUpOut, L.UBias, _ffn);
            // Fused SwiGLU(_dGateOut, _dUpOut) + Pre-RHT + GEMV + Post-RHT for DW!
            LaunchTrellisSwigluMv(L.DW!, _dGateOut, _dUpOut, _dProj);

            // Fused Residual Add + next layer AttnNorm (or final OutNorm)!
            if (l + 1 < _layers.Length)
                LaunchAddRms(_dX, _dProj, _layers[l + 1].AttnNorm, _dNorm, _dim);
            else if (wantLogits)
                LaunchAddRms(_dX, _dProj, _dOutNorm, _dNorm, _dim);
            else
                LaunchAdd(_dX, _dProj, _dim);
        }

        if (wantLogits)
        {
            if (_layers.Length == 0)
                LaunchRms(_dX, _dOutNorm, _dNorm, _dim, 1);
            LaunchTrellisMv(_outMat, _dNorm, _dLogits);
        }
    }

    private IntPtr EnsureGraph(bool wantLogits)
    {
        if (_graphDisabled) return IntPtr.Zero;
        ref IntPtr exec = ref (wantLogits ? ref _graphExecLogits : ref _graphExecNoLogits);
        if (exec != IntPtr.Zero) return exec;

        try
        {
            CuDriver.Check(CuDriver.StreamSynchronize(_stream), "pre-capture sync");
            CuDriver.Check(CuDriver.StreamSynchronize(_stream2), "pre-capture sync2");
            CuDriver.Check(CuDriver.StreamSynchronize(_stream3), "pre-capture sync3");
            CuDriver.Check(CuDriver.StreamBeginCapture(_stream, 0), "cuStreamBeginCapture");
            EnqueueForwardPass(wantLogits);
            CuDriver.Check(CuDriver.StreamEndCapture(_stream, out IntPtr graph), "cuStreamEndCapture");
            try
            {
                CuDriver.Check(CuDriver.GraphInstantiateWithFlags(out exec, graph, 0), "cuGraphInstantiate");
            }
            finally
            {
                CuDriver.GraphDestroy(graph);
            }
            return exec;
        }
        catch (Exception ex)
        {
            GlacierDiagnostics.LogWarning($"[CUDA GPU] CUDA Graph capture unavailable ({ex.Message}); using direct stream launch.");
            _graphDisabled = true;
            return IntPtr.Zero;
        }
    }

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)pos >= (uint)_maxSeq) throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position exceeds MaxSeqLen ({_maxSeq}).");
        if ((uint)token >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(token), token, "Token id outside vocabulary.");
        bool wantLogits = computeLogits && logits.Length >= _vocab;

        _ctx.MakeCurrent();

        // 1. Decode 1 embedding row on CPU (<0.05 ms) and async-upload 20 KB + 4-byte pos
        fixed (float* pEmb = _hostEmb)
        {
            QuantKernels.ExtractEmbedding(_w.EmbdType, _w.EmbdWeight, token, pEmb, _dim);
            CuDriver.Check(CuDriver.MemcpyHtoDAsync(_dX, (IntPtr)pEmb, (nuint)((long)_dim * 4), _stream), "HtoD(emb)");
        }
        int posLocal = pos;
        CuDriver.Check(CuDriver.MemcpyHtoDAsync(_dPos, (IntPtr)(&posLocal), 4, _stream), "HtoD(pos)");

        // 2. Execute all 64 layers (+ LM head) in ONE single CUDA Graph launch (zero WDDM launch overhead!)
        IntPtr graphExec = EnsureGraph(wantLogits);
        if (graphExec != IntPtr.Zero)
        {
            CuDriver.Check(CuDriver.GraphLaunch(graphExec, _stream), "cuGraphLaunch");
        }
        else
        {
            EnqueueForwardPass(wantLogits);
        }

        // 3. Synchronize & download logits
        CuDriver.Check(CuDriver.StreamSynchronize(_stream), "cuStreamSynchronize");
        if (wantLogits)
        {
            fixed (float* pLog = logits)
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)pLog, _dLogits, (nuint)((long)_vocab * 4)), "DtoH(logits)");
        }
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool last = i == tokens.Length - 1;
            Forward(tokens[i], startPos + i, last ? logits : Span<float>.Empty, last && computeLogits);
        }
    }

    public void ResetKvCache()
    {
        _ctx.MakeCurrent();
        foreach (var L in _layers)
        {
            if (L.IsGdn)
            {
                CuDriver.Check(CuDriver.MemsetD8(L.ConvState, 0, L.ConvStateBytes), "reset(convState)");
                CuDriver.Check(CuDriver.MemsetD8(L.RecState, 0, L.RecStateBytes), "reset(recState)");
            }
        }
    }

    private void RunCudaProfileBreakdown()
    {
        // Time each stage individually across 1 full forward pass using CUDA events (5 reps of full graph vs component breakdown)
        CuDriver.Check(CuDriver.EventCreate(out IntPtr ev0, 0), "ev0");
        CuDriver.Check(CuDriver.EventCreate(out IntPtr ev1, 0), "ev1");
        try
        {
            // Warmup graph
            IntPtr g = EnsureGraph(true);
            CuDriver.Check(CuDriver.GraphLaunch(g, _stream), "warmup");
            CuDriver.Check(CuDriver.StreamSynchronize(_stream), "sync");

            CuDriver.Check(CuDriver.EventRecord(ev0, _stream), "rec0");
            for (int r = 0; r < 5; r++) CuDriver.Check(CuDriver.GraphLaunch(g, _stream), "rep");
            CuDriver.Check(CuDriver.EventRecord(ev1, _stream), "rec1");
            CuDriver.Check(CuDriver.EventSynchronize(ev1), "evsync");
            CuDriver.Check(CuDriver.EventElapsedTime(out float totalGraphMs, ev0, ev1), "elapsed");

            // Time LM head alone
            CuDriver.Check(CuDriver.EventRecord(ev0, _stream), "rec0");
            for (int r = 0; r < 5; r++) LaunchTrellisMv(_outMat, _dNorm, _dLogits);
            CuDriver.Check(CuDriver.EventRecord(ev1, _stream), "rec1");
            CuDriver.Check(CuDriver.EventSynchronize(ev1), "evsync");
            CuDriver.Check(CuDriver.EventElapsedTime(out float lmHeadMs, ev0, ev1), "elapsed");

            // Time 1 layer of FFN GW (5120x17408), DW (17408x5120), GDN rec, and 1 full GDN layer vs Attn layer
            var L0 = _layers[0]; // GDN
            var L3 = _layers[3]; // Attn
            float TimeMat(MatGpu m, IntPtr x, IntPtr y)
            {
                CuDriver.EventRecord(ev0, _stream);
                for (int r = 0; r < 20; r++) LaunchTrellisMv(m, x, y);
                CuDriver.EventRecord(ev1, _stream);
                CuDriver.EventSynchronize(ev1);
                CuDriver.EventElapsedTime(out float ms, ev0, ev1);
                return ms / 20f;
            }

            float gwMs = TimeMat(L0.GW!, _dNorm, _dGateOut);
            float dwMs = TimeMat(L0.DW!, _dAct, _dProj);
            float qkvMs = TimeMat(L0.Qkv!, _dNorm, _dQkv);
            float ssmOutMs = TimeMat(L0.SsmOut!, _dGated, _dProj);
            float qMs = TimeMat(L3.QW!, _dNorm, _dQFull);
            float kMs = TimeMat(L3.KW!, _dNorm, _dK);

            GlacierDiagnostics.LogInformation(
                $"[CUDA PROFILE] FullGraph={totalGraphMs / 5f:F2} ms ({5000f / totalGraphMs:F1} tok/s) | LMHead={lmHeadMs / 5f:F2} ms\n" +
                $"  L0.GW(b{L0.GW!.Trellis!.Bits} {L0.GW.Trellis.K}x{L0.GW.Trellis.N})={gwMs * 1000:F1} us | " +
                $"L0.DW(b{L0.DW!.Trellis!.Bits} {L0.DW.Trellis.K}x{L0.DW.Trellis.N})={dwMs * 1000:F1} us | " +
                $"L0.Qkv(b{L0.Qkv!.Trellis!.Bits} {L0.Qkv.Trellis.K}x{L0.Qkv.Trellis.N})={qkvMs * 1000:F1} us | " +
                $"L0.SsmOut(b{L0.SsmOut!.Trellis!.Bits} {L0.SsmOut.Trellis.K}x{L0.SsmOut.Trellis.N})={ssmOutMs * 1000:F1} us\n" +
                $"  L3.QW(b{L3.QW!.Trellis!.Bits} {L3.QW.Trellis.K}x{L3.QW.Trellis.N})={qMs * 1000:F1} us | " +
                $"L3.KW(b{L3.KW!.Trellis!.Bits} {L3.KW.Trellis.K}x{L3.KW.Trellis.N})={kMs * 1000:F1} us");

            ResetKvCache();
        }
        finally
        {
            CuDriver.EventDestroy(ev0);
            CuDriver.EventDestroy(ev1);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            _ctx.MakeCurrent();
            if (_stream != IntPtr.Zero) CuDriver.StreamSynchronize(_stream);
            if (_graphExecLogits != IntPtr.Zero) { CuDriver.GraphExecDestroy(_graphExecLogits); _graphExecLogits = IntPtr.Zero; }
            if (_graphExecNoLogits != IntPtr.Zero) { CuDriver.GraphExecDestroy(_graphExecNoLogits); _graphExecNoLogits = IntPtr.Zero; }
            foreach (var tm in _trlMats) tm.Dispose();
            _trlMats.Clear();
            foreach (var p in _allocs) _ctx.FreeDevice(p);
            _allocs.Clear();
            if (_module != IntPtr.Zero) CuDriver.ModuleUnload(_module);
            if (_stream != IntPtr.Zero) CuDriver.StreamDestroy(_stream);
            if (_stream2 != IntPtr.Zero) CuDriver.StreamDestroy(_stream2);
            if (_stream3 != IntPtr.Zero) CuDriver.StreamDestroy(_stream3);
            if (_evFork != IntPtr.Zero) CuDriver.EventDestroy(_evFork);
            if (_evJoin != IntPtr.Zero) CuDriver.EventDestroy(_evJoin);
            if (_evJoin3 != IntPtr.Zero) CuDriver.EventDestroy(_evJoin3);
            if (_evAlphaBeta != IntPtr.Zero) CuDriver.EventDestroy(_evAlphaBeta);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning($"[Qwen35CudaModel] CUDA event cleanup failed during Dispose: {ex.Message}");
        }
        _trl.Dispose();
        _ctx.Dispose();
    }
}
