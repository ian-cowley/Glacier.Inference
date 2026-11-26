namespace Glacier.Inference.Gpu;

using System;
using System.Collections.Generic;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# CUDA Driver API (<c>nvcuda.dll</c>) runner for NariQ Trellis-Coded Quantization
/// (<c>GGML_TYPE_TRL1..TRL6</c>, <c>nariq/FORMAT.md</c> v1).
/// Executes <c>trellis_pre_rht</c>, <c>trellis_gemv</c> (256 threads per 16x16 output tile row),
/// <c>trellis_post_rht</c>, and <c>trellis_gather_f32</c> directly from embedded <c>kernels.cubin</c>.
/// </summary>
public sealed unsafe class TrellisCudaRunner : IDisposable
{
    private readonly GpuContext _ctx;
    private readonly bool _ownsContext;
    private IntPtr _module;
    private IntPtr _fnPreRht;
    private IntPtr _fnPreRhtSwiglu;
    private IntPtr _fnGemv;
    private IntPtr _fnGemvSplitK;
    private IntPtr _fnGemvSplitK8;
    private IntPtr _fnPostRht;
    private IntPtr _fnGather;
    private bool _disposed;

    public string DeviceName => _ctx.DeviceName;

    public sealed class DeviceMatrix : IDisposable
    {
        private readonly GpuContext _ctx;
        public IntPtr CodeWords, S16, Pre, Post, Pin, Pout;
        public int K, N, Bits;
        public int RowsKind, RowsBlock, ColsKind, ColsBlock;
        public bool HasPin, HasPout;

        internal DeviceMatrix(GpuContext ctx) => _ctx = ctx;

        public void Dispose()
        {
            _ctx.FreeDevice(CodeWords); CodeWords = IntPtr.Zero;
            _ctx.FreeDevice(S16); S16 = IntPtr.Zero;
            _ctx.FreeDevice(Pre); Pre = IntPtr.Zero;
            _ctx.FreeDevice(Post); Post = IntPtr.Zero;
            _ctx.FreeDevice(Pin); Pin = IntPtr.Zero;
            _ctx.FreeDevice(Pout); Pout = IntPtr.Zero;
        }
    }

    public TrellisCudaRunner(int deviceOrdinal = 0) : this(new GpuContext(deviceOrdinal), ownsContext: true) { }

    public TrellisCudaRunner(GpuContext ctx, bool ownsContext = false)
    {
        _ctx = ctx;
        _ownsContext = ownsContext;
        _ctx.MakeCurrent();

        byte[] cubin = KernelCompiler.GetOrCompileKernels(_ctx.ArchString);
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), "cuModuleLoadData(trellis)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPreRht, _module, "trellis_pre_rht"), "cuModuleGetFunction(trellis_pre_rht)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPreRhtSwiglu, _module, "trellis_pre_rht_swiglu"), "cuModuleGetFunction(trellis_pre_rht_swiglu)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemv, _module, "trellis_gemv"), "cuModuleGetFunction(trellis_gemv)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvSplitK, _module, "trellis_gemv_splitk"), "cuModuleGetFunction(trellis_gemv_splitk)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvSplitK8, _module, "trellis_gemv_splitk8"), "cuModuleGetFunction(trellis_gemv_splitk8)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnPostRht, _module, "trellis_post_rht"), "cuModuleGetFunction(trellis_post_rht)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGather, _module, "trellis_gather_f32"), "cuModuleGetFunction(trellis_gather_f32)");
    }

    public DeviceMatrix Upload(TrellisWeightBinding b)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _ctx.MakeCurrent();

        int bits = GgufTypes.GetTrellisBits(b.Type);
        nuint codeBytes = (nuint)((long)b.K * b.N * bits / 8);
        nuint s16Bytes = (nuint)((b.K / 16) * sizeof(ushort));
        nuint preBytes = (nuint)((long)b.K * sizeof(float));
        nuint postBytes = (nuint)((long)b.N * sizeof(float));

        var m = new DeviceMatrix(_ctx)
        {
            K = b.K,
            N = b.N,
            Bits = bits,
            RowsKind = b.Descriptor[0],
            RowsBlock = b.Descriptor[1],
            ColsKind = b.Descriptor[3],
            ColsBlock = b.Descriptor[4],
            HasPin = b.Pin != null,
            HasPout = b.Pout != null,
            CodeWords = _ctx.AllocateDevice(codeBytes),
            S16 = _ctx.AllocateDevice(s16Bytes),
            Pre = _ctx.AllocateDevice(preBytes),
            Post = _ctx.AllocateDevice(postBytes),
            Pin = b.Pin != null ? _ctx.AllocateDevice((nuint)((long)b.K * sizeof(int))) : IntPtr.Zero,
            Pout = b.Pout != null ? _ctx.AllocateDevice((nuint)((long)b.N * sizeof(int))) : IntPtr.Zero,
        };

        CuDriver.Check(CuDriver.MemcpyHtoD(m.CodeWords, (IntPtr)b.CodeData, codeBytes), "HtoD(code)");
        CuDriver.Check(CuDriver.MemcpyHtoD(m.S16, (IntPtr)b.S16, s16Bytes), "HtoD(s16)");
        CuDriver.Check(CuDriver.MemcpyHtoD(m.Pre, (IntPtr)b.Pre, preBytes), "HtoD(pre)");
        CuDriver.Check(CuDriver.MemcpyHtoD(m.Post, (IntPtr)b.Post, postBytes), "HtoD(post)");
        if (b.Pin != null) CuDriver.Check(CuDriver.MemcpyHtoD(m.Pin, (IntPtr)b.Pin, (nuint)((long)b.K * sizeof(int))), "HtoD(pin)");
        if (b.Pout != null) CuDriver.Check(CuDriver.MemcpyHtoD(m.Pout, (IntPtr)b.Pout, (nuint)((long)b.N * sizeof(int))), "HtoD(pout)");
        return m;
    }

    /// <summary>
    /// Executes <c>y = gather_pout(post * T_rows^T(W_rotated * T_cols(pre * gather_pin(x))))</c> on CUDA.
    /// </summary>
    public void MatVecMul(DeviceMatrix m, ReadOnlySpan<float> xHost, Span<float> yHost)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (m.ColsKind > 1 || m.RowsKind > 1)
            throw new NotSupportedException($"CUDA kernel currently supports kind=0 (identity) and kind=1 (block RHT <= 2048); got cols={m.ColsKind}, rows={m.RowsKind}.");

        _ctx.MakeCurrent();
        nuint xBytes = (nuint)((long)m.K * sizeof(float));
        nuint yBytes = (nuint)((long)m.N * sizeof(float));

        IntPtr dX = _ctx.AllocateDevice(xBytes);
        IntPtr dXRot = _ctx.AllocateDevice(xBytes);
        IntPtr dYRot = _ctx.AllocateDevice(yBytes);
        IntPtr dYScaled = _ctx.AllocateDevice(yBytes);
        IntPtr dYFinal = m.HasPout ? _ctx.AllocateDevice(yBytes) : dYScaled;

        try
        {
            fixed (float* px = xHost)
                CuDriver.Check(CuDriver.MemcpyHtoD(dX, (IntPtr)px, xBytes), "HtoD(x)");

            DispatchDevice(m, dX, dXRot, dYRot, dYScaled, dYFinal);
            CuDriver.Check(CuDriver.CtxSynchronize(), "cuCtxSynchronize");

            fixed (float* py = yHost)
                CuDriver.Check(CuDriver.MemcpyDtoH((IntPtr)py, dYFinal, yBytes), "DtoH(y)");
        }
        finally
        {
            _ctx.FreeDevice(dX);
            _ctx.FreeDevice(dXRot);
            _ctx.FreeDevice(dYRot);
            _ctx.FreeDevice(dYScaled);
            if (m.HasPout) _ctx.FreeDevice(dYFinal);
        }
    }

    /// <summary>
    /// Benchmarks pure device-side <see cref="DispatchDevice"/> execution (excluding HtoD/DtoH/alloc)
    /// over <paramref name="reps"/> iterations and returns average milliseconds per call.
    /// </summary>
    public double BenchmarkDevice(DeviceMatrix m, ReadOnlySpan<float> xHost, int reps = 20)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _ctx.MakeCurrent();
        nuint xBytes = (nuint)((long)m.K * sizeof(float));
        nuint yBytes = (nuint)((long)m.N * sizeof(float));

        IntPtr dX = _ctx.AllocateDevice(xBytes);
        IntPtr dXRot = _ctx.AllocateDevice(xBytes);
        IntPtr dYRot = _ctx.AllocateDevice(yBytes);
        IntPtr dYScaled = _ctx.AllocateDevice(yBytes);
        IntPtr dYFinal = m.HasPout ? _ctx.AllocateDevice(yBytes) : dYScaled;
        try
        {
            fixed (float* px = xHost)
                CuDriver.Check(CuDriver.MemcpyHtoD(dX, (IntPtr)px, xBytes), "HtoD(x)");
            DispatchDevice(m, dX, dXRot, dYRot, dYScaled, dYFinal);
            CuDriver.Check(CuDriver.CtxSynchronize(), "warmup");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < reps; r++)
                DispatchDevice(m, dX, dXRot, dYRot, dYScaled, dYFinal);
            CuDriver.Check(CuDriver.CtxSynchronize(), "bench");
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds / reps;
        }
        finally
        {
            _ctx.FreeDevice(dX);
            _ctx.FreeDevice(dXRot);
            _ctx.FreeDevice(dYRot);
            _ctx.FreeDevice(dYScaled);
            if (m.HasPout) _ctx.FreeDevice(dYFinal);
        }
    }

    public void DispatchDevice(DeviceMatrix m, IntPtr dX, IntPtr dXRot, IntPtr dYRot, IntPtr dYScaled, IntPtr dYFinal, int outOffset = 0, IntPtr stream = default)
    {
        // 1. Pre-RHT with fused s16 scaling and dynamic SMEM (512 B for block_len == 128!)
        IntPtr pX = dX, pPre = m.Pre, pPin = m.Pin, pS16 = m.S16, pXRot = dXRot;
        int k = m.K, cKind = m.ColsKind, cBlock = m.ColsBlock, hasPin = m.HasPin ? 1 : 0;
        uint preGrid = cKind == 1 ? (uint)(k / cBlock) : (uint)((k + 255) / 256);
        uint preBlock = (cKind == 1 && cBlock == 128) ? 128u : 256u;
        uint preSmem = cKind == 1 ? (uint)(cBlock * sizeof(float)) : 0u;
        IntPtr* preArgs = stackalloc IntPtr[9];
        preArgs[0] = (IntPtr)(&pX);
        preArgs[1] = (IntPtr)(&pPre);
        preArgs[2] = (IntPtr)(&pPin);
        preArgs[3] = (IntPtr)(&pS16);
        preArgs[4] = (IntPtr)(&pXRot);
        preArgs[5] = (IntPtr)(&k);
        preArgs[6] = (IntPtr)(&cKind);
        preArgs[7] = (IntPtr)(&cBlock);
        preArgs[8] = (IntPtr)(&hasPin);
        CuDriver.Check(CuDriver.LaunchKernel(_fnPreRht, preGrid, 1, 1, preBlock, 1, 1, preSmem, stream, (IntPtr)preArgs, IntPtr.Zero), "launch(trellis_pre_rht)");

        DispatchGemvAndPost(m, dXRot, dYRot, dYScaled, dYFinal, outOffset, stream);
    }

    public void DispatchDeviceSwiglu(DeviceMatrix m, IntPtr dGate, IntPtr dUp, IntPtr dXRot, IntPtr dYRot, IntPtr dYScaled, IntPtr dYFinal, int outOffset = 0, IntPtr stream = default)
    {
        // 1. Fused SwiGLU + Pre-RHT + s16 scaling (eliminates separate q35_swiglu kernel + global round-trip!)
        IntPtr pGate = dGate, pUp = dUp, pPre = m.Pre, pPin = m.Pin, pS16 = m.S16, pXRot = dXRot;
        int k = m.K, cKind = m.ColsKind, cBlock = m.ColsBlock, hasPin = m.HasPin ? 1 : 0;
        uint preGrid = cKind == 1 ? (uint)(k / cBlock) : (uint)((k + 255) / 256);
        uint preBlock = (cKind == 1 && cBlock == 128) ? 128u : 256u;
        uint preSmem = cKind == 1 ? (uint)(cBlock * sizeof(float)) : 0u;
        IntPtr* preArgs = stackalloc IntPtr[10];
        preArgs[0] = (IntPtr)(&pGate);
        preArgs[1] = (IntPtr)(&pUp);
        preArgs[2] = (IntPtr)(&pPre);
        preArgs[3] = (IntPtr)(&pPin);
        preArgs[4] = (IntPtr)(&pS16);
        preArgs[5] = (IntPtr)(&pXRot);
        preArgs[6] = (IntPtr)(&k);
        preArgs[7] = (IntPtr)(&cKind);
        preArgs[8] = (IntPtr)(&cBlock);
        preArgs[9] = (IntPtr)(&hasPin);
        CuDriver.Check(CuDriver.LaunchKernel(_fnPreRhtSwiglu, preGrid, 1, 1, preBlock, 1, 1, preSmem, stream, (IntPtr)preArgs, IntPtr.Zero), "launch(trellis_pre_rht_swiglu)");

        DispatchGemvAndPost(m, dXRot, dYRot, dYScaled, dYFinal, outOffset, stream);
    }

    private void DispatchGemvAndPost(DeviceMatrix m, IntPtr dXRot, IntPtr dYRot, IntPtr dYScaled, IntPtr dYFinal, int outOffset, IntPtr stream)
    {
        // 2. 16x16 Tile GEMV:
        //    - N <= 2048 & kt >= 64: 8-warp Split-K (256T/block) for full SM saturation on KW/VW
        //    - N <= 6144 & kt >= 64: 4-warp Split-K (128T/block) for DW/SsmOut/OW
        //    - N > 6144: 4 warps = 64 output rows/block (128T/block) for GW/UW/Qkv/QW/LM head
        IntPtr pCode = m.CodeWords, pXRot = dXRot, pYRot = dYRot;
        int k = m.K, n = m.N, bits = m.Bits;
        IntPtr* gemvArgs = stackalloc IntPtr[6];
        gemvArgs[0] = (IntPtr)(&pCode);
        gemvArgs[1] = (IntPtr)(&pXRot);
        gemvArgs[2] = (IntPtr)(&pYRot);
        gemvArgs[3] = (IntPtr)(&k);
        gemvArgs[4] = (IntPtr)(&n);
        gemvArgs[5] = (IntPtr)(&bits);
        if (n <= 2048 && (k >> 4) >= 64)
        {
            uint splitGrid = (uint)(n / 16);
            CuDriver.Check(CuDriver.LaunchKernel(_fnGemvSplitK8, splitGrid, 1, 1, 256, 1, 1, 0, stream, (IntPtr)gemvArgs, IntPtr.Zero), "launch(trellis_gemv_splitk8)");
        }
        else if (n <= 6144 && (k >> 4) >= 64)
        {
            uint splitGrid = (uint)(n / 16);
            CuDriver.Check(CuDriver.LaunchKernel(_fnGemvSplitK, splitGrid, 1, 1, 128, 1, 1, 0, stream, (IntPtr)gemvArgs, IntPtr.Zero), "launch(trellis_gemv_splitk)");
        }
        else
        {
            uint gemvGrid = (uint)((n + 63) / 64);
            CuDriver.Check(CuDriver.LaunchKernel(_fnGemv, gemvGrid, 1, 1, 128, 1, 1, 0, stream, (IntPtr)gemvArgs, IntPtr.Zero), "launch(trellis_gemv)");
        }

        // 3. Post-RHT with dynamic SMEM + warp-shuffle FWHT
        IntPtr pPost = m.Post, pYScaled = dYScaled;
        int rKind = m.RowsKind, rBlock = m.RowsBlock;
        int off = m.HasPout ? 0 : outOffset;
        uint postGrid = rKind == 1 ? (uint)(n / rBlock) : (uint)((n + 255) / 256);
        uint postBlock = (rKind == 1 && rBlock == 128) ? 128u : 256u;
        uint postSmem = rKind == 1 ? (uint)(rBlock * sizeof(float)) : 0u;
        IntPtr* postArgs = stackalloc IntPtr[7];
        postArgs[0] = (IntPtr)(&pYRot);
        postArgs[1] = (IntPtr)(&pPost);
        postArgs[2] = (IntPtr)(&pYScaled);
        postArgs[3] = (IntPtr)(&n);
        postArgs[4] = (IntPtr)(&rKind);
        postArgs[5] = (IntPtr)(&rBlock);
        postArgs[6] = (IntPtr)(&off);
        CuDriver.Check(CuDriver.LaunchKernel(_fnPostRht, postGrid, 1, 1, postBlock, 1, 1, postSmem, stream, (IntPtr)postArgs, IntPtr.Zero), "launch(trellis_post_rht)");

        // 4. Optional output gather
        if (m.HasPout)
        {
            IntPtr pPout = m.Pout, pYFin = dYFinal;
            uint gGrid = (uint)((n + 255) / 256);
            IntPtr* gArgs = stackalloc IntPtr[4];
            gArgs[0] = (IntPtr)(&pYScaled);
            gArgs[1] = (IntPtr)(&pPout);
            gArgs[2] = (IntPtr)(&pYFin);
            gArgs[3] = (IntPtr)(&n);
            CuDriver.Check(CuDriver.LaunchKernel(_fnGather, gGrid, 1, 1, 256, 1, 1, 0, stream, (IntPtr)gArgs, IntPtr.Zero), "launch(trellis_gather_f32)");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_module != IntPtr.Zero)
        {
            _ctx.MakeCurrent();
            CuDriver.ModuleUnload(_module);
            _module = IntPtr.Zero;
        }
        if (_ownsContext) _ctx.Dispose();
    }
}
