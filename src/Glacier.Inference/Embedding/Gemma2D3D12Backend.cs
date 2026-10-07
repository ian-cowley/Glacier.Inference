namespace Glacier.Inference.Embedding;

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Vortice.Direct3D12;

/// <summary>
/// Direct3D 12 compute backend for the Gemma 2 encoder. Runs on any DX12 GPU: NVIDIA, AMD, Intel, and Qualcomm
/// Adreno / other ARM GPUs on Windows-on-ARM. Q8_0 weights are re-packed once (int8 quants + FP32 scales) and
/// dequantised on the fly inside a tiled GEMM kernel; the whole forward pass stays on the GPU.
/// </summary>
internal sealed unsafe class Gemma2D3D12Backend : IGemma2Backend
{
    private sealed class GpuLinear : IDisposable
    {
        public ID3D12Resource? Quants;   // Q8: uint[rows*in/4]
        public ID3D12Resource? Scales;   // Q8: float[rows*in/32]
        public ID3D12Resource? F32;      // float weights
        public int In, Out;
        public bool IsQ8 => Quants != null;
        public void Dispose() { Quants?.Dispose(); Scales?.Dispose(); F32?.Dispose(); }
    }

    private sealed class GpuLayer
    {
        public required GpuLinear Q, K, V, O, Gate, Up, Down, PleGate, PleProj;
        public required ID3D12Resource AttnNorm, QNorm, KNorm, AttnPostNorm, FfnNorm, FfnPostNorm, PleNorm;
        public required ID3D12Resource RopeInvFreq;
        public float OutScale;
    }

    private readonly D3D12Context _ctx;
    private readonly GgufFile _gguf;
    private readonly Gemma2Config _cfg;
    private readonly List<IDisposable> _owned = [];
    private readonly GpuLayer[] _layers;
    private readonly GpuLinear _perLayerProj;
    private readonly ID3D12Resource _perLayerProjNorm, _outputNorm;
    private readonly object _gate = new();

    private ID3D12RootSignature _sig = null!;
    private ID3D12PipelineState _psoGemmQ8 = null!, _psoGemmF32 = null!, _psoRms = null!, _psoRope = null!,
        _psoAttn = null!, _psoGeluMul = null!, _psoAddNorm = null!, _psoMean = null!;

    private int _capacity;
    private ID3D12Resource? _x0, _x, _normed, _tmp, _q, _k, _v, _attnOut, _ffnGate, _ffnUp, _pleIn, _pleG, _pooled;
    private bool _disposed;

    public string Name => $"Direct3D 12 compute on {_ctx.DeviceName} (Q8_0 on-the-fly dequant)";

    public Gemma2D3D12Backend(GgufFile gguf, Gemma2Config cfg, int adapterIndex = -1)
    {
        _gguf = gguf;
        _cfg = cfg;
        _ctx = new D3D12Context(adapterIndex);
        try
        {
            InitPipelines();
            int H = cfg.Hidden;
            _perLayerProj = LoadLinear("per_layer_model_proj.weight", H, cfg.PleDim * cfg.Layers, forceF32: true);
            _perLayerProjNorm = UploadVector("per_layer_proj_norm.weight", cfg.PleDim);
            _outputNorm = UploadVector("output_norm.weight", H);

            _layers = new GpuLayer[cfg.Layers];
            for (int l = 0; l < cfg.Layers; l++)
            {
                int hd = cfg.HeadDim[l], kv = cfg.KvHeads[l];
                string p = $"blk.{l}.";
                var invFreq = new float[hd / 2];
                double theta = cfg.IsSwa[l] ? cfg.RopeBaseSwa : cfg.RopeBase;
                for (int i = 0; i < invFreq.Length; i++) invFreq[i] = (float)(1.0 / Math.Pow(theta, 2.0 * i / hd));
                _layers[l] = new GpuLayer
                {
                    Q = LoadLinear(p + "attn_q.weight", H, cfg.Heads * hd),
                    K = LoadLinear(p + "attn_k.weight", H, kv * hd),
                    V = LoadLinear(p + "attn_v.weight", H, kv * hd),
                    O = LoadLinear(p + "attn_output.weight", cfg.Heads * hd, H),
                    Gate = LoadLinear(p + "ffn_gate.weight", H, cfg.FfnDim),
                    Up = LoadLinear(p + "ffn_up.weight", H, cfg.FfnDim),
                    Down = LoadLinear(p + "ffn_down.weight", cfg.FfnDim, H),
                    PleGate = LoadLinear(p + "inp_gate.weight", H, cfg.PleDim),
                    PleProj = LoadLinear(p + "proj.weight", cfg.PleDim, H),
                    AttnNorm = UploadVector(p + "attn_norm.weight", H),
                    QNorm = UploadVector(p + "attn_q_norm.weight", hd),
                    KNorm = UploadVector(p + "attn_k_norm.weight", hd),
                    AttnPostNorm = UploadVector(p + "post_attention_norm.weight", H),
                    FfnNorm = UploadVector(p + "ffn_norm.weight", H),
                    FfnPostNorm = UploadVector(p + "post_ffw_norm.weight", H),
                    PleNorm = UploadVector(p + "post_norm.weight", H),
                    RopeInvFreq = UploadFloats(invFreq),
                    OutScale = Gemma2Tensors.LoadVector(gguf, p + "layer_output_scale.weight", 1)[0],
                };
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // ------------------------------------------------------------------------------ setup

    private void InitPipelines()
    {
        var ps = new List<RootParameter> { new(new RootConstants(0, 0, 16), ShaderVisibility.All) };
        for (uint i = 0; i < 5; i++)
            ps.Add(new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(i, 0), ShaderVisibility.All));
        _sig = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ps.ToArray()));

        ID3D12PipelineState Pso(string hlsl, string define = "") =>
            _ctx.CreatePipelineState(_sig, _ctx.CompileShader(define + "\n" + Gemma2Shaders.Common + "\n" + hlsl));

        _psoGemmQ8 = Pso(Gemma2Shaders.Gemm, "#define Q8W 1\n");
        _psoGemmF32 = Pso(Gemma2Shaders.Gemm);
        _psoRms = Pso(Gemma2Shaders.RmsNorm);
        _psoRope = Pso(Gemma2Shaders.Rope);
        _psoAttn = Pso(Gemma2Shaders.Attention);
        _psoGeluMul = Pso(Gemma2Shaders.GeluMul);
        _psoAddNorm = Pso(Gemma2Shaders.AddNorm);
        _psoMean = Pso(Gemma2Shaders.MeanPool);
    }

    private ID3D12Resource Track(ID3D12Resource r) { _owned.Add(r); return r; }

    private ID3D12Resource UploadFloats(float[] data)
    {
        var buf = Track(_ctx.CreateDeviceBuffer((ulong)Math.Max(256, data.Length * 4)));
        fixed (float* p = data) _ctx.CopyToDevice(buf, (IntPtr)p, (ulong)data.Length * 4);
        return buf;
    }

    private ID3D12Resource UploadVector(string name, int len) => UploadFloats(Gemma2Tensors.LoadVector(_gguf, name, len));

    private GpuLinear LoadLinear(string name, int inDim, int outDim, bool forceF32 = false)
    {
        var info = Gemma2Tensors.Require(_gguf, name, (ulong)inDim, (ulong)outDim);
        uint type = (uint)info.Type;
        byte* src = _gguf.GetTensorPointer(info);
        var lin = new GpuLinear { In = inDim, Out = outDim };
        _owned.Add(lin);

        if (type == Gemma2Tensors.Q8_0 && !forceF32 && inDim % 32 == 0)
        {
            int blocks = inDim / 32;
            var quants = new sbyte[(long)outDim * inDim];
            var scales = new float[(long)outDim * blocks];
            for (int r = 0; r < outDim; r++)
            {
                byte* row = src + r * (long)blocks * 34;
                for (int b = 0; b < blocks; b++)
                {
                    byte* blk = row + b * 34;
                    scales[(long)r * blocks + b] = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(blk);
                    for (int i = 0; i < 32; i++) quants[(long)r * inDim + b * 32 + i] = (sbyte)blk[2 + i];
                }
            }
            lin.Quants = _ctx.CreateDeviceBuffer((ulong)quants.Length);
            lin.Scales = _ctx.CreateDeviceBuffer((ulong)(scales.Length * 4));
            fixed (sbyte* pq = quants) _ctx.CopyToDevice(lin.Quants, (IntPtr)pq, (ulong)quants.Length);
            fixed (float* ps = scales) _ctx.CopyToDevice(lin.Scales, (IntPtr)ps, (ulong)scales.Length * 4);
        }
        else
        {
            var w = new float[(long)outDim * inDim];
            fixed (float* pw = w)
                for (int r = 0; r < outDim; r++) Gemma2Tensors.DequantRow(src, type, r, inDim, pw + (long)r * inDim);
            lin.F32 = _ctx.CreateDeviceBuffer((ulong)w.Length * 4);
            fixed (float* pw = w) _ctx.CopyToDevice(lin.F32, (IntPtr)pw, (ulong)w.Length * 4);
        }
        return lin;
    }

    private void EnsureCapacity(int n)
    {
        if (n <= _capacity) return;
        FreeActivations();
        int cap = ((n + 255) / 256 + 1) * 256; // slack rows: no kernel may fault at an exact allocation boundary
        int H = _cfg.Hidden, P = _cfg.PleDim, F = _cfg.FfnDim;
        int maxQ = 0, maxKv = 0;
        for (int l = 0; l < _cfg.Layers; l++)
        {
            maxQ = Math.Max(maxQ, _cfg.Heads * _cfg.HeadDim[l]);
            maxKv = Math.Max(maxKv, _cfg.KvHeads[l] * _cfg.HeadDim[l]);
        }
        ID3D12Resource Buf(long floats) => _ctx.CreateDeviceBuffer((ulong)Math.Max(256, floats * 4));
        _x0 = Buf((long)cap * H); _x = Buf((long)cap * H); _normed = Buf((long)cap * H); _tmp = Buf((long)cap * H);
        _q = Buf((long)cap * maxQ); _k = Buf((long)cap * maxKv); _v = Buf((long)cap * maxKv); _attnOut = Buf((long)cap * maxQ);
        _ffnGate = Buf((long)cap * F); _ffnUp = Buf((long)cap * F);
        _pleIn = Buf((long)cap * P); _pleG = Buf((long)cap * P); _pooled = Buf(H);
        _capacity = cap;
    }

    private void FreeActivations()
    {
        foreach (var r in new[] { _x0, _x, _normed, _tmp, _q, _k, _v, _attnOut, _ffnGate, _ffnUp, _pleIn, _pleG, _pooled })
            r?.Dispose();
        _x0 = _x = _normed = _tmp = _q = _k = _v = _attnOut = _ffnGate = _ffnUp = _pleIn = _pleG = _pooled = null;
        _capacity = 0;
    }

    // ------------------------------------------------------------------------------ dispatch helpers

    private ID3D12GraphicsCommandList Cmd => _ctx.CommandList;

    private void Dispatch(ID3D12PipelineState pso, ReadOnlySpan<uint> consts, uint gx, uint gy,
        ID3D12Resource? u0 = null, ID3D12Resource? u1 = null, ID3D12Resource? u2 = null, ID3D12Resource? u3 = null, ID3D12Resource? u4 = null)
    {
        var cmd = Cmd;
        cmd.SetComputeRootSignature(_sig);
        cmd.SetPipelineState(pso);
        uint* c = stackalloc uint[16];
        for (int i = 0; i < 16; i++) c[i] = i < consts.Length ? consts[i] : 0;
        cmd.SetComputeRoot32BitConstants(0, 16, (IntPtr)c, 0);
        ulong dummy = _ctx.DummyBuffer.GPUVirtualAddress;
        cmd.SetComputeRootUnorderedAccessView(1, u0?.GPUVirtualAddress ?? dummy);
        cmd.SetComputeRootUnorderedAccessView(2, u1?.GPUVirtualAddress ?? dummy);
        cmd.SetComputeRootUnorderedAccessView(3, u2?.GPUVirtualAddress ?? dummy);
        cmd.SetComputeRootUnorderedAccessView(4, u3?.GPUVirtualAddress ?? dummy);
        cmd.SetComputeRootUnorderedAccessView(5, u4?.GPUVirtualAddress ?? dummy);
        cmd.Dispatch(gx, gy, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
    }

    private static uint F(float v) => BitConverter.SingleToUInt32Bits(v);

    private void Linear(GpuLinear w, ID3D12Resource x, ID3D12Resource y, int n, int rowStart = 0, int rowCount = -1)
    {
        if (rowCount < 0) rowCount = w.Out - rowStart;
        uint[] c = [(uint)n, (uint)w.In, (uint)rowCount, (uint)rowStart];
        uint gx = (uint)((rowCount + 15) / 16), gy = (uint)((n + 15) / 16);
        if (w.IsQ8) Dispatch(_psoGemmQ8, c, gx, gy, w.Quants, w.Scales, x, y);
        else Dispatch(_psoGemmF32, c, gx, gy, w.F32, null, x, y);
    }

    private void RmsNorm(ID3D12Resource x, ID3D12Resource? weight, ID3D12Resource y, int rowLen, int rows, float inScale = 1f)
        => Dispatch(_psoRms, [(uint)rowLen, (uint)rows, F(_cfg.Eps), F(inScale), weight != null ? 1u : 0u], (uint)rows, 1, x, weight, y);

    /// <summary>Splits a 1-D element count into a 2-D grid (D3D12 caps each dimension at 65535 groups of 256 threads).</summary>
    private static (uint X, uint Y) Grid1D(long count)
    {
        long groups = (count + 255) / 256;
        return ((uint)Math.Min(groups, 65535), (uint)((groups + 65534) / 65535));
    }

    private void GeluMul(ID3D12Resource a, ID3D12Resource b, long count)
        { var (gx, gy) = Grid1D(count); Dispatch(_psoGeluMul, [(uint)count], gx, gy, a, b); }

    private void AddNorm(ID3D12Resource x, ID3D12Resource y, ID3D12Resource w, int rowLen, int rows, float scale)
        => Dispatch(_psoAddNorm, [(uint)rowLen, (uint)rows, F(_cfg.Eps), F(scale)], (uint)rows, 1, x, y, w);

    private void Flush()
    {
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        _ctx.BeginCommands();
    }

    // ------------------------------------------------------------------------------ forward

    public float[] ForwardPooled(ReadOnlySpan<int> tokens) => Run(Gemma2Embedding.Gather(_gguf, _cfg, tokens), tokens.Length, pooled: true);
    public float[] ForwardHidden(ReadOnlySpan<int> tokens) => Run(Gemma2Embedding.Gather(_gguf, _cfg, tokens), tokens.Length, pooled: false);
    public float[] ForwardPooledEmbeds(float[] inputEmbeddings, int tokenCount) => Run(inputEmbeddings, tokenCount, pooled: true);

    private float[] Run(float[] x0, int n, bool pooled)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (n == 0) throw new ArgumentException("At least one token is required.", nameof(n));

        lock (_gate)
        {
            EnsureCapacity(n);
            fixed (float* p = x0)
            {
                _ctx.CopyToDevice(_x0!, (IntPtr)p, (ulong)x0.Length * 4);
                _ctx.CopyToDevice(_x!, (IntPtr)p, (ulong)x0.Length * 4);
            }

            int H = _cfg.Hidden, P = _cfg.PleDim, F = _cfg.FfnDim, heads = _cfg.Heads;
            _ctx.BeginCommands();
            for (int l = 0; l < _cfg.Layers; l++)
            {
                var L = _layers[l];
                int hd = _cfg.HeadDim[l], kv = _cfg.KvHeads[l];

                RmsNorm(_x!, L.AttnNorm, _normed!, H, n);
                Linear(L.Q, _normed!, _q!, n);
                Linear(L.K, _normed!, _k!, n);
                Linear(L.V, _normed!, _v!, n);
                RmsNorm(_q!, L.QNorm, _q!, hd, n * heads);
                RmsNorm(_k!, L.KNorm, _k!, hd, n * kv);
                RmsNorm(_v!, null, _v!, hd, n * kv);
                { var (gx, gy) = Grid1D((long)n * heads * (hd / 2)); Dispatch(_psoRope, [(uint)hd, (uint)heads, (uint)(n * heads)], gx, gy, _q, L.RopeInvFreq); }
                { var (gx, gy) = Grid1D((long)n * kv * (hd / 2)); Dispatch(_psoRope, [(uint)hd, (uint)kv, (uint)(n * kv)], gx, gy, _k, L.RopeInvFreq); }

                int window = _cfg.IsSwa[l] ? _cfg.HalfWindow : -1;
                {
                    // Split into query-block chunks, one submission each, so no single dispatch can trip the GPU watchdog (TDR).
                    uint blocks = (uint)((n + 7) / 8);
                    uint chunk = (uint)Math.Max(1, 24576 / Math.Max(1, (long)heads * Math.Max(1, (window < 0 ? n : Math.Min(n, 2 * window + 1))) / 64));
                    chunk = Math.Min(chunk, 64);
                    for (uint b0 = 0; b0 < blocks; b0 += chunk)
                    {
                        Dispatch(_psoAttn, [(uint)n, (uint)heads, (uint)kv, (uint)hd, unchecked((uint)window), b0], Math.Min(chunk, blocks - b0), (uint)heads, _q, _k, _v, _attnOut);
                        if (blocks > chunk) Flush();
                    }
                }

                Linear(L.O, _attnOut!, _tmp!, n);
                AddNorm(_x!, _tmp!, L.AttnPostNorm, H, n, 1f);

                RmsNorm(_x!, L.FfnNorm, _normed!, H, n);
                Linear(L.Gate, _normed!, _ffnGate!, n);
                Linear(L.Up, _normed!, _ffnUp!, n);
                GeluMul(_ffnGate!, _ffnUp!, (long)n * F);
                Linear(L.Down, _ffnGate!, _tmp!, n);
                AddNorm(_x!, _tmp!, L.FfnPostNorm, H, n, 1f);

                Linear(_perLayerProj, _x0!, _pleIn!, n, rowStart: l * P, rowCount: P);
                RmsNorm(_pleIn!, _perLayerProjNorm, _pleIn!, P, n, inScale: 1f / MathF.Sqrt(H));
                Linear(L.PleGate, _x!, _pleG!, n);
                GeluMul(_pleG!, _pleIn!, (long)n * P);
                Linear(L.PleProj, _pleG!, _tmp!, n);
                AddNorm(_x!, _tmp!, L.PleNorm, H, n, L.OutScale);

                Flush(); // keep each submission short (TDR safety on integrated / mobile GPUs)
            }

            RmsNorm(_x!, _outputNorm, _x!, H, n);

            float[] result;
            if (pooled)
            {
                Dispatch(_psoMean, [(uint)H, (uint)n], (uint)((H + 255) / 256), 1, _x, _pooled);
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();
                result = new float[H];
                fixed (float* pr = result) _ctx.CopyToHost((IntPtr)pr, _pooled!, (ulong)H * 4);
            }
            else
            {
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();
                result = new float[(long)n * H];
                fixed (float* pr = result) _ctx.CopyToHost((IntPtr)pr, _x!, (ulong)result.Length * 4);
            }
            return result;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _ctx.Synchronize(); } catch { /* device may already be lost */ }
        FreeActivations();
        foreach (var o in _owned) o.Dispose();
        _owned.Clear();
        _psoGemmQ8?.Dispose(); _psoGemmF32?.Dispose(); _psoRms?.Dispose(); _psoRope?.Dispose();
        _psoAttn?.Dispose(); _psoGeluMul?.Dispose(); _psoAddNorm?.Dispose(); _psoMean?.Dispose();
        _sig?.Dispose();
        _ctx.Dispose();
    }
}

internal static class Gemma2Shaders
{
    /// <summary>Shared declarations: one root signature (16 constants + u0..u4) for every kernel.</summary>
    public const string Common = """
        cbuffer Params : register(b0)
        {
            uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
            uint c8; uint c9; uint c10; uint c11; uint c12; uint c13; uint c14; uint c15;
        };
        static const float SQRT_2_OVER_PI = 0.7978845608028654f;
        float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanh(clamp(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x), -15.0f, 15.0f))); }
        """;

    /// <summary>Tiled GEMM Y[t, o] = sum_k W[rowStart+o, k] * X[t, k]. Q8W: W = packed int8 (u0) + scales (u1); else float W (u0).</summary>
    public const string Gemm = """
        #ifdef Q8W
        RWStructuredBuffer<uint> Wq : register(u0);
        RWStructuredBuffer<float> Ws : register(u1);
        #else
        RWStructuredBuffer<float> Wf : register(u0);
        #endif
        RWStructuredBuffer<float> X : register(u2);
        RWStructuredBuffer<float> Y : register(u3);

        groupshared float Wt[16][33];
        groupshared float Xt[16][33];

        [numthreads(16, 16, 1)]
        void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
        {
            uint n = c0, inDim = c1, rowCount = c2, rowStart = c3;
            uint o0 = gid.x * 16, t0 = gid.y * 16;
            uint lin = tid.y * 16 + tid.x;
            float acc = 0.0f;
            uint blocks = inDim / 32;

            for (uint kb = 0; kb < blocks; kb++)
            {
                // X tile: 16 tokens x 32 -> 512 floats, 2 per thread
                [unroll] for (uint e = 0; e < 2; e++)
                {
                    uint idx = lin + e * 256;
                    uint r = idx / 32, cc = idx % 32;
                    uint t = t0 + r;
                    Xt[r][cc] = (t < n) ? X[t * inDim + kb * 32 + cc] : 0.0f;
                }
        #ifdef Q8W
                // W tile: 16 rows x 8 words (4 int8 each) = 128 threads
                if (lin < 128)
                {
                    uint r = lin / 8, wi = lin % 8;
                    uint o = o0 + r;
                    float4 v = float4(0, 0, 0, 0);
                    if (o < rowCount)
                    {
                        uint wrow = rowStart + o;
                        uint word = Wq[(wrow * inDim + kb * 32) / 4 + wi];
                        float s = Ws[wrow * blocks + kb];
                        int b0 = ((int)(word << 24)) >> 24;
                        int b1 = ((int)(word << 16)) >> 24;
                        int b2 = ((int)(word << 8)) >> 24;
                        int b3 = ((int)word) >> 24;
                        v = float4(b0, b1, b2, b3) * s;
                    }
                    Wt[r][wi * 4 + 0] = v.x; Wt[r][wi * 4 + 1] = v.y; Wt[r][wi * 4 + 2] = v.z; Wt[r][wi * 4 + 3] = v.w;
                }
        #else
                [unroll] for (uint e2 = 0; e2 < 2; e2++)
                {
                    uint idx = lin + e2 * 256;
                    uint r = idx / 32, cc = idx % 32;
                    uint o = o0 + r;
                    Wt[r][cc] = (o < rowCount) ? Wf[(rowStart + o) * inDim + kb * 32 + cc] : 0.0f;
                }
        #endif
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (uint c = 0; c < 32; c++) acc += Wt[tid.x][c] * Xt[tid.y][c];
                GroupMemoryBarrierWithGroupSync();
            }

            uint o = o0 + tid.x, t = t0 + tid.y;
            if (o < rowCount && t < n) Y[t * rowCount + o] = acc;
        }
        """;

    /// <summary>Row RMSNorm (one group per row). c0=len c1=rows c2=eps c3=inScale c4=hasWeight. u0=x u1=weight u2=dst (dst may alias x).</summary>
    public const string RmsNorm = """
        RWStructuredBuffer<float> X : register(u0);
        RWStructuredBuffer<float> W : register(u1);
        RWStructuredBuffer<float> D : register(u2);
        groupshared float red[256];

        [numthreads(256, 1, 1)]
        void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
        {
            uint len = c0; float eps = asfloat(c2); float inScale = asfloat(c3);
            uint row = gid.x; uint baseIdx = row * len;
            float ss = 0.0f;
            for (uint i = tid; i < len; i += 256) { float v = X[baseIdx + i] * inScale; ss += v * v; }
            red[tid] = ss;
            GroupMemoryBarrierWithGroupSync();
            for (uint s = 128; s > 0; s >>= 1)
            {
                if (tid < s) red[tid] += red[tid + s];
                GroupMemoryBarrierWithGroupSync();
            }
            float scale = rsqrt(red[0] / (float)len + eps);
            GroupMemoryBarrierWithGroupSync();
            for (uint j = tid; j < len; j += 256)
            {
                float v = X[baseIdx + j] * inScale * scale;
                if (c4 != 0) v *= W[j];
                D[baseIdx + j] = v;
            }
        }
        """;

    /// <summary>NeoX rotate_half RoPE. c0=hd c1=headsPerToken c2=rows(n*heads). u0=x u1=invFreq[hd/2]. Row r has position r / headsPerToken.</summary>
    public const string Rope = """
        RWStructuredBuffer<float> X : register(u0);
        RWStructuredBuffer<float> InvFreq : register(u1);

        [numthreads(256, 1, 1)]
        void main(uint3 gid : SV_GroupID, uint ltid : SV_GroupIndex)
        {
            uint hd = c0, hpt = c1, rows = c2;
            uint half_ = hd / 2;
            uint idx = (gid.y * 65535 + gid.x) * 256 + ltid;
            if (idx >= rows * half_) return;
            uint row = idx / half_, i = idx % half_;
            uint pos = row / hpt;
            float angle = (float)pos * InvFreq[i];
            float cs, sn; sincos(angle, sn, cs);
            uint b = row * hd;
            float a = X[b + i], bb = X[b + i + half_];
            X[b + i] = a * cs - bb * sn;
            X[b + i + half_] = bb * cs + a * sn;
        }
        """;

    /// <summary>
    /// Bidirectional (optionally windowed) attention, online softmax, scale 1.0.
    /// c0=n c1=heads c2=kvHeads c3=hd c4=halfWindow (-1 = global). u0=q u1=k u2=v u3=out. Group (x=t, y=head), 256 threads.
    /// </summary>
    public const string Attention = """
        RWStructuredBuffer<float> Q : register(u0);
        RWStructuredBuffer<float> K : register(u1);
        RWStructuredBuffer<float> V : register(u2);
        RWStructuredBuffer<float> O : register(u3);

        // 8 queries (consecutive t, same head) per group share every K/V read: 8x less memory traffic.
        groupshared float sq[8 * 512];
        groupshared float sp[8 * 256];
        groupshared float sred[8 * 256];

        [numthreads(256, 1, 1)]
        void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
        {
            int n = (int)c0; uint heads = c1, kvHeads = c2, hd = c3; int win = (int)c4;
            uint t0 = (gid.x + c5) * 8, h = gid.y;
            int nq = min(8, n - (int)t0);
            uint group = heads / kvHeads, kvh = h / group;
            uint qStride = heads * hd, kvStride = kvHeads * hd;

            for (uint idx = tid; idx < 8 * hd; idx += 256)
            {
                uint qi0 = idx / hd, d0 = idx - qi0 * hd;
                sq[qi0 * 512 + d0] = ((int)qi0 < nq) ? Q[(t0 + qi0) * qStride + h * hd + d0] : 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();

            int lo = 0, hi = n - 1;
            if (win >= 0) { lo = max(0, (int)t0 - win); hi = min(n - 1, (int)t0 + nq - 1 + win); }

            float m[8], l[8], acc0[8], acc1[8];
            [unroll] for (int a = 0; a < 8; a++) { m[a] = -1e30f; l[a] = 0.0f; acc0[a] = 0.0f; acc1[a] = 0.0f; }

            for (int start = lo; start <= hi; start += 256)
            {
                int j = start + (int)tid;
                float dsum[8];
                [unroll] for (int b = 0; b < 8; b++) dsum[b] = 0.0f;
                if (j <= hi)
                {
                    uint kb = (uint)j * kvStride + kvh * hd;
                    for (uint d = 0; d < hd; d++)
                    {
                        float kv = K[kb + d];
                        [unroll] for (int c = 0; c < 8; c++) dsum[c] += sq[c * 512 + d] * kv;
                    }
                }
                bool ok[8];
                [unroll] for (int q = 0; q < 8; q++)
                {
                    int tq = (int)t0 + q;
                    ok[q] = (q < nq) && (j <= hi) && (win < 0 || abs(j - tq) <= win);
                    sred[q * 256 + tid] = ok[q] ? dsum[q] : -1e30f;
                }
                GroupMemoryBarrierWithGroupSync();
                for (uint r = 128; r > 0; r >>= 1)
                {
                    if (tid < r) { [unroll] for (int q2 = 0; q2 < 8; q2++) sred[q2 * 256 + tid] = max(sred[q2 * 256 + tid], sred[q2 * 256 + tid + r]); }
                    GroupMemoryBarrierWithGroupSync();
                }
                float corr[8];
                [unroll] for (int q3 = 0; q3 < 8; q3++)
                {
                    float tileMax = sred[q3 * 256];
                    float newM = max(m[q3], tileMax);
                    corr[q3] = exp(m[q3] - newM);
                    m[q3] = newM;
                }
                GroupMemoryBarrierWithGroupSync();
                [unroll] for (int q4 = 0; q4 < 8; q4++)
                {
                    float pj = ok[q4] ? exp(dsum[q4] - m[q4]) : 0.0f;
                    sp[q4 * 256 + tid] = pj;
                    sred[q4 * 256 + tid] = pj;
                }
                GroupMemoryBarrierWithGroupSync();
                for (uint r2 = 128; r2 > 0; r2 >>= 1)
                {
                    if (tid < r2) { [unroll] for (int q5 = 0; q5 < 8; q5++) sred[q5 * 256 + tid] += sred[q5 * 256 + tid + r2]; }
                    GroupMemoryBarrierWithGroupSync();
                }
                [unroll] for (int q6 = 0; q6 < 8; q6++)
                {
                    l[q6] = l[q6] * corr[q6] + sred[q6 * 256];
                    acc0[q6] *= corr[q6]; acc1[q6] *= corr[q6];
                }
                GroupMemoryBarrierWithGroupSync();

                int cnt = min(256, hi - start + 1);
                for (int jj = 0; jj < cnt; jj++)
                {
                    uint vb = (uint)(start + jj) * kvStride + kvh * hd;
                    float v0 = V[vb + tid];
                    float v1 = (hd > 256) ? V[vb + tid + 256] : 0.0f;
                    [unroll] for (int q7 = 0; q7 < 8; q7++)
                    {
                        float p = sp[q7 * 256 + jj];
                        acc0[q7] += p * v0;
                        acc1[q7] += p * v1;
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }

            [unroll] for (int q8 = 0; q8 < 8; q8++)
            {
                if (q8 < nq)
                {
                    float inv = 1.0f / l[q8];
                    uint ob = (t0 + q8) * qStride + h * hd;
                    if (tid < hd) O[ob + tid] = acc0[q8] * inv;
                    if (hd > 256 && tid + 256 < hd) O[ob + tid + 256] = acc1[q8] * inv;
                }
            }
        }
        """;

    /// <summary>a[i] = gelu(a[i]) * b[i]; c0 = count.</summary>
    public const string GeluMul = """
        RWStructuredBuffer<float> A : register(u0);
        RWStructuredBuffer<float> B : register(u1);

        [numthreads(256, 1, 1)]
        void main(uint3 gid : SV_GroupID, uint ltid : SV_GroupIndex)
        {
            uint idx = (gid.y * 65535 + gid.x) * 256 + ltid;
            if (idx >= c0) return;
            A[idx] = gelu_tanh(A[idx]) * B[idx];
        }
        """;

    /// <summary>x[row] = (x[row] + RMSNorm(y[row]) * w) * scale. c0=len c1=rows c2=eps c3=scale. u0=x u1=y u2=w.</summary>
    public const string AddNorm = """
        RWStructuredBuffer<float> X : register(u0);
        RWStructuredBuffer<float> Y : register(u1);
        RWStructuredBuffer<float> W : register(u2);
        groupshared float red[256];

        [numthreads(256, 1, 1)]
        void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
        {
            uint len = c0; float eps = asfloat(c2); float scale = asfloat(c3);
            uint baseIdx = gid.x * len;
            float ss = 0.0f;
            for (uint i = tid; i < len; i += 256) { float v = Y[baseIdx + i]; ss += v * v; }
            red[tid] = ss;
            GroupMemoryBarrierWithGroupSync();
            for (uint s = 128; s > 0; s >>= 1)
            {
                if (tid < s) red[tid] += red[tid + s];
                GroupMemoryBarrierWithGroupSync();
            }
            float rs = rsqrt(red[0] / (float)len + eps);
            for (uint j = tid; j < len; j += 256)
                X[baseIdx + j] = (X[baseIdx + j] + Y[baseIdx + j] * rs * W[j]) * scale;
        }
        """;

    /// <summary>out[d] = mean_t x[t, d]. c0=H c1=n. u0=x u1=out.</summary>
    public const string MeanPool = """
        RWStructuredBuffer<float> X : register(u0);
        RWStructuredBuffer<float> Out : register(u1);

        [numthreads(256, 1, 1)]
        void main(uint3 dtid : SV_DispatchThreadID)
        {
            uint H = c0, n = c1;
            if (dtid.x >= H) return;
            float s = 0.0f;
            for (uint t = 0; t < n; t++) s += X[t * H + dtid.x];
            Out[dtid.x] = s / (float)n;
        }
        """;
}






