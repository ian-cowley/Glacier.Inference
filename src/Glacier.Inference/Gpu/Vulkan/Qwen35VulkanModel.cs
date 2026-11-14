namespace Glacier.Inference.Gpu;

using System;
using System.Collections.Generic;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;

/// <summary>
/// Vulkan compute inference runner for the Qwen 3.5 / 3.6 hybrid family (Gated DeltaNet + interleaved full attention,
/// dense or Mixture-of-Experts FFN). Works on any Vulkan 1.1 GPU (Linux, Windows, Android); the whole forward pass
/// for a token is recorded into one command buffer and only the logits are read back.
/// Supported quant types for weight matrices: Q4_K, Q5_K, Q6_K, Q8_0 (alpha/beta/router/norms are F32).
/// </summary>
public sealed unsafe class Qwen35VulkanModel : ID3D12Model
{
    private const int HeadDim = 256, StateDim = 128, ConvKernel = 4;

    private enum Patch { None, Pos, KvLen, Token }

    private sealed class Op
    {
        public VulkanCompute.Kernel K = null!;
        public IntPtr Set;
        public uint[] Push = Array.Empty<uint>();
        public uint Gx = 1, Gy = 1;
        public Patch Patch;
        public string Name = "";
    }

    private sealed record Mat(VulkanCompute.GBuf Buf, GgufType Type, int K, int N);

    private readonly ModelWeights _w;
    private readonly VulkanContext _ctx;
    private readonly VulkanCompute _vk;
    private readonly int _maxSeq, _dim, _vocab;
    private readonly List<Op> _embedOps = new(), _layerOps = new(), _finalOps = new();
    private readonly List<(VulkanCompute.GBuf Conv, VulkanCompute.GBuf Rec)> _states = new();
    private readonly VulkanCompute.GBuf _gLogits;
    private bool _disposed;

    // Kernels
    private readonly VulkanCompute.Kernel[] _kMv = new VulkanCompute.Kernel[6], _kMoe = new VulkanCompute.Kernel[6], _kEmbed = new VulkanCompute.Kernel[4];
    private readonly VulkanCompute.Kernel _kF32, _kRms, _kSwi, _kAdd, _kConv, _kL2, _kRec, _kTopk, _kComb, _kPrep, _kDec;

    public string DeviceName => _ctx.DeviceName;

    /// <summary>
    /// Maps a Glacier device name (e.g. "AMD Radeon 890M Graphics") to a Vulkan physical-device ordinal.
    /// Returns -1 when no real GPU matches (software rasterisers such as llvmpipe are never selected).
    /// </summary>
    public static int FindVulkanDevice(string glacierDeviceName)
    {
        try
        {
            var devices = VulkanContext.EnumerateDevices();
            int firstReal = -1;
            foreach (var d in devices)
            {
                if (d.Name.Contains("llvmpipe", StringComparison.OrdinalIgnoreCase) || d.Name.Contains("software", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (firstReal < 0) firstReal = d.Index;
                if (!string.IsNullOrEmpty(glacierDeviceName) &&
                    (d.Name.Contains(glacierDeviceName, StringComparison.OrdinalIgnoreCase) ||
                     glacierDeviceName.Contains(d.Name, StringComparison.OrdinalIgnoreCase)))
                    return d.Index;
            }
            return firstReal;
        }
        catch
        {
            return -1;
        }
    }

    /// <summary>Returns true when the Vulkan runner can execute this model; otherwise <paramref name="reason"/> explains why not.</summary>
    public static bool IsSupported(ModelWeights w, out string reason)
    {
        reason = "";
        if (!w.IsHybridSsm) { reason = "not a hybrid Gated DeltaNet model"; return false; }
        if (w.HeadDim != HeadDim || w.ValueDim != HeadDim) { reason = $"attention head dim {w.HeadDim}/{w.ValueDim} != {HeadDim}"; return false; }
        if (w.SsmStateSize != StateDim || w.SsmConvKernel != ConvKernel) { reason = "unsupported SSM state size or conv kernel"; return false; }
        if (w.SsmTimeStepRank * StateDim != w.SsmInnerSize) { reason = "SSM inner size != heads * state dim"; return false; }
        if (w.HasRopeFreqs) { reason = "custom RoPE frequency table"; return false; }
        if (w.HeadCount % w.HeadCountKv != 0) { reason = "head count not divisible by KV head count"; return false; }
        foreach (var l in w.Layers)
        {
            if (l.QBias != null || l.KBias != null || l.VBias != null || l.AttnOutBias != null) { reason = "attention biases"; return false; }
            if (!l.IsGdn && !l.HasQGate) { reason = "full-attention layer without Q gate"; return false; }
            if (!l.IsMoe && l.HasFusedGateUp) { reason = "fused gate/up FFN"; return false; }

            // Weight matrices must use a quantisation type that has a Vulkan kernel. Checked here, before any GPU memory is
            // allocated, so that unsupported files fail fast and fall back to the CPU engine.
            (string Name, bool Present, GgufType T)[] mats =
            {
                ("attn_qkv", l.QkvWeight != null, l.QkvType), ("attn_gate", l.AttnGateWeight != null, l.AttnGateType), ("ssm_out", l.SsmOutWeight != null, l.SsmOutType),
                ("attn_q", l.QWeight != null, l.QType), ("attn_k", l.KWeight != null, l.KType), ("attn_v", l.VWeight != null, l.VType), ("attn_output", l.AttnOutWeight != null, l.AttnOutType),
                ("ffn_gate", l.FfnGateWeight != null, l.FfnGateType), ("ffn_up", l.FfnUpWeight != null, l.FfnUpType), ("ffn_down", l.FfnDownWeight != null, l.FfnDownType),
                ("ffn_gate_exps", l.FfnGateExpsWeight != null, l.FfnGateExpsType), ("ffn_up_exps", l.FfnUpExpsWeight != null, l.FfnUpExpsType), ("ffn_down_exps", l.FfnDownExpsWeight != null, l.FfnDownExpsType),
                ("ffn_gate_shexp", l.FfnGateShexpWeight != null, l.FfnGateShexpType), ("ffn_up_shexp", l.FfnUpShexpWeight != null, l.FfnUpShexpType), ("ffn_down_shexp", l.FfnDownShexpWeight != null, l.FfnDownShexpType),
            };
            foreach (var (name, present, t) in mats)
                if (present && !IsKernelType(t)) { reason = $"tensor '{name}' uses unsupported weight type {t} (supported: Q4_K, Q5_K, Q6_K, Q8_0)"; return false; }
            if (l.IsGdn && !(l.SsmAlphaType == GgufType.F32 || IsKernelType(l.SsmAlphaType)) ||
                l.IsGdn && !(l.SsmBetaType == GgufType.F32 || IsKernelType(l.SsmBetaType)))
            { reason = "ssm_alpha / ssm_beta use an unsupported weight type"; return false; }
        }
        if (!IsEmbedKernelType(w.EmbdType)) { reason = $"token embedding uses unsupported weight type {w.EmbdType}"; return false; }
        if (!IsKernelType(w.OutType)) { reason = $"output head uses unsupported weight type {w.OutType}"; return false; }
        return true;
    }

    private static bool IsEmbedKernelType(GgufType t) => t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0;
    private static bool IsKernelType(GgufType t) => t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.IQ4_XS or GgufType.MXFP4;

    private static int QIdx(GgufType t) => t switch
    {
        GgufType.Q4_K => 0,
        GgufType.Q5_K => 1,
        GgufType.Q6_K => 2,
        GgufType.Q8_0 => 3,
        GgufType.IQ4_XS => 4,
        GgufType.MXFP4 => 5,
        _ => throw new NotSupportedException($"Vulkan runner does not support weight type {t}.")
    };

    public Qwen35VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Vulkan runner cannot execute this model: {why}.");

        _w = weights;
        _maxSeq = maxSeqLen;
        _dim = weights.EmbeddingLength;
        _vocab = weights.VocabSize;
        _ctx = new VulkanContext(deviceOrdinal);
        _vk = new VulkanCompute(_ctx);

        try
        {
            // ---- kernels ----
            for (int t = 0; t < 6; t++)
            {
                _kMv[t] = _vk.LoadEmbedded($"qmatvec_q{t}", 3, 8);
                _kMoe[t] = _vk.LoadEmbedded($"qmatvec_moe_q{t}", 4, 16);
                if (t < 4) _kEmbed[t] = _vk.LoadEmbedded($"embed_q{t}", 2, 8);
            }
            _kF32 = _vk.LoadEmbedded("f32_matvec", 3, 8);
            _kRms = _vk.LoadEmbedded("rmsnorm", 3, 12);
            _kSwi = _vk.LoadEmbedded("elementwise_swiglu", 3, 4);
            _kAdd = _vk.LoadEmbedded("elementwise_add", 3, 4);
            _kConv = _vk.LoadEmbedded("gdn_conv", 4, 4);
            _kL2 = _vk.LoadEmbedded("gdn_l2", 1, 4);
            _kRec = _vk.LoadEmbedded("gdn_rec", 7, 8);
            _kTopk = _vk.LoadEmbedded("moe_topk", 3, 12);
            _kComb = _vk.LoadEmbedded("moe_combine", 6, 12);
            _kPrep = _vk.LoadEmbedded("attn_prep", 9, 28);
            _kDec = _vk.LoadEmbedded("attn_decode", 6, 20);

            _gLogits = _vk.Alloc((long)_vocab * 4);
            Build();
        }
        catch
        {
            _vk.Dispose();
            _ctx.Dispose();
            throw;
        }
    }

    // ------------------------------------------------------------------ plan construction

    private Mat UploadMat(byte* p, GgufType type, int k, int n)
    {
        QIdx(type);
        if (p == null) throw new InvalidOperationException("Missing weight tensor.");
        long bytes = (long)n * GgufTypes.GetRowBytes(type, k);
        return new Mat(_vk.AllocFrom(p, bytes), type, k, n);
    }

    private VulkanCompute.GBuf UploadF32(void* p, long count)
    {
        if (p == null) throw new InvalidOperationException("Missing F32 tensor.");
        return _vk.AllocFrom(p, count * 4);
    }

    /// <summary>
    /// Uploads a small [n rows x k cols] matrix as F32. Quantised sources (some files store the tiny ssm_alpha / ssm_beta
    /// projections as Q8_0 or Q4_K) are dequantised row by row on the host.
    /// </summary>
    private VulkanCompute.GBuf UploadSmallMatrixAsF32(byte* p, GgufType type, int k, int n)
    {
        if (p == null) throw new InvalidOperationException("Missing tensor.");
        if (type == GgufType.F32) return UploadF32(p, (long)n * k);
        var rows = new float[(long)n * k];
        fixed (float* dst = rows)
        {
            for (int r = 0; r < n; r++)
                Glacier.Inference.Quant.QuantKernels.ExtractEmbedding(type, p, r, dst + (long)r * k, k);
        }
        fixed (float* src = rows) return _vk.AllocFrom(src, rows.Length * 4L);
    }

    private Op Mv(Mat m, VulkanCompute.GBuf x, VulkanCompute.GBuf y)
    {
        var k = _kMv[QIdx(m.Type)];
        return new Op
        {
            Name = $"mv_{m.Type} K{m.K} N{m.N}", K = k, Set = _vk.MakeSet(k, m.Buf, x, y),
            Push = new uint[] { (uint)m.K, (uint)m.N },
            Gx = (uint)Math.Min(m.N, 32768), Gy = (uint)((m.N + 32767) / 32768)
        };
    }

    private Op MvF32(VulkanCompute.GBuf w, int k, int n, VulkanCompute.GBuf x, VulkanCompute.GBuf y) => new()
    {
        Name = "mv_F32", K = _kF32, Set = _vk.MakeSet(_kF32, w, x, y), Push = new uint[] { (uint)k, (uint)n }, Gx = (uint)n
    };

    private Op MvMoe(Mat expsAsOne, int k, int n, long sliceBytes, int xStride, VulkanCompute.GBuf x, VulkanCompute.GBuf y, VulkanCompute.GBuf idx, int topK)
    {
        var kern = _kMoe[QIdx(expsAsOne.Type)];
        return new Op
        {
            Name = $"moe_mv_{expsAsOne.Type} K{k} N{n}", K = kern, Set = _vk.MakeSet(kern, expsAsOne.Buf, x, y, idx),
            Push = new uint[] { (uint)k, (uint)n, (uint)sliceBytes, (uint)xStride },
            Gx = (uint)n, Gy = (uint)topK
        };
    }

    private Op Rms(VulkanCompute.GBuf x, VulkanCompute.GBuf w, VulkanCompute.GBuf y, int size, int rows) => new()
    {
        Name = "rmsnorm", K = _kRms, Set = _vk.MakeSet(_kRms, x, w, y),
        Push = new uint[] { (uint)size, (uint)rows, VulkanCompute.F(_w.RmsNormEps) }, Gx = (uint)rows
    };

    private Op Swiglu(VulkanCompute.GBuf gate, VulkanCompute.GBuf up, VulkanCompute.GBuf o, int n) => new()
    {
        Name = "swiglu", K = _kSwi, Set = _vk.MakeSet(_kSwi, gate, up, o), Push = new uint[] { (uint)n }, Gx = (uint)((n + 255) / 256)
    };

    private Op Add(VulkanCompute.GBuf a, VulkanCompute.GBuf b, int n) => new()
    {
        Name = "add", K = _kAdd, Set = _vk.MakeSet(_kAdd, a, b, b), Push = new uint[] { (uint)n }, Gx = (uint)((n + 255) / 256)
    };

    private void Build()
    {
        var w = _w;
        int dim = _dim, nL = w.BlockCount, nH = w.HeadCount, nKv = w.HeadCountKv;
        int qDim = nH * HeadDim, kvDim = nKv * HeadDim;
        int nK = w.SsmGroupCount, nV = w.SsmTimeStepRank, ch = (2 * nK + nV) * StateDim, inner = w.SsmInnerSize;
        int E = w.ExpertCount, topK = Math.Max(1, w.ExpertUsedCount), F = w.ExpertFeedForwardLength, ffn = w.FeedForwardLength;
        int sFmax = F * 2;
        for (int l = 0; l < nL; l++)
            if (w.Layers[l].IsMoe && w.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tt) && tt != null)
                sFmax = Math.Max(sFmax, (int)tt.Dimensions[1]);
        int maxF = Math.Max(Math.Max(F * topK, sFmax), ffn);

        // ---- shared activations / scratch ----
        var gX = _vk.Alloc(dim * 4L); var gNorm = _vk.Alloc(dim * 4L); var gProj = _vk.Alloc(dim * 4L);
        var gQkv = _vk.Alloc(ch * 4L); var gZ = _vk.Alloc(inner * 4L); var gAlpha = _vk.Alloc(nV * 4L); var gBeta = _vk.Alloc(nV * 4L);
        var gConvOut = _vk.Alloc(ch * 4L); var gY = _vk.Alloc(inner * 4L); var gGated = _vk.Alloc(inner * 4L);
        var gQFull = _vk.Alloc(2L * qDim * 4); var gK = _vk.Alloc(kvDim * 4L); var gV = _vk.Alloc(kvDim * 4L);
        var gQ = _vk.Alloc(qDim * 4L); var gGate = _vk.Alloc(qDim * 4L); var gAttnOut = _vk.Alloc(qDim * 4L);
        var gScores = _vk.Alloc((long)nH * _maxSeq * 4);
        var gIdx = _vk.Alloc(topK * 4L); var gTw = _vk.Alloc(topK * 4L);
        var gGateOut = _vk.Alloc(maxF * 4L); var gUpOut = _vk.Alloc(maxF * 4L); var gAct = _vk.Alloc(maxF * 4L);
        var gDownOut = _vk.Alloc((long)topK * dim * 4);
        var gSg = _vk.Alloc(sFmax * 4L); var gSu = _vk.Alloc(sFmax * 4L); var gSAct = _vk.Alloc(sFmax * 4L); var gSd = _vk.Alloc(dim * 4L);
        var gDummy = _vk.Alloc(dim * 4L);
        var gRouterLogits = _vk.Alloc(Math.Max(E, 1) * 4L);

        // ---- output head ----
        // Allocation order matters on unified-memory APUs: the first ~0.9 GB of GPU allocations are placed in the (fast) UMA
        // frame-buffer carve-out; everything after that lands in system-memory GTT, which streams several times slower.
        // The output head is read in full on every token, so it is allocated first; the embedding table (a one-row gather)
        // is allocated last. See docs/VULKAN_BACKEND.md.
        var outNorm = UploadF32(w.OutNormWeight, dim);
        var outMat = UploadMat(w.OutWeight, w.OutType, dim, _vocab);

        // ---- layers ----
        float scale = 1f / MathF.Sqrt(HeadDim);
        for (int l = 0; l < nL; l++)
        {
            var L = w.Layers[l];
            if (l % 4 == 0 || l == nL - 1)
                GlacierDiagnostics.LogInformation($"[Vulkan GPU] Uploading {w.Gguf.Architecture} weights: layer {l + 1}/{nL}...");

            var attnNorm = UploadF32(L.AttnNormWeight, dim);
            var ffnNorm = UploadF32(L.FfnNormWeight, dim);
            _layerOps.Add(Rms(gX, attnNorm, gNorm, dim, 1));

            if (L.IsGdn)
            {
                var qkv = UploadMat(L.QkvWeight, L.QkvType, dim, ch);
                var gate = UploadMat(L.AttnGateWeight, L.AttnGateType, dim, inner);
                var alphaW = UploadSmallMatrixAsF32(L.SsmAlphaWeight, L.SsmAlphaType, dim, nV);
                var betaW = UploadSmallMatrixAsF32(L.SsmBetaWeight, L.SsmBetaType, dim, nV);
                var convW = UploadF32(L.SsmConv1dWeight, (long)ch * ConvKernel);
                var aW = UploadF32(L.SsmAWeight, nV);
                var dtBias = UploadF32(L.SsmDtBias, nV);
                var ssmNorm = UploadF32(L.SsmNormWeight, StateDim);
                var outW = UploadMat(L.SsmOutWeight, L.SsmOutType, inner, dim);

                var convState = _vk.Alloc(ch * 3L * 4); convState.Zero();
                var recState = _vk.Alloc((long)nV * StateDim * StateDim * 4); recState.Zero();
                _states.Add((convState, recState));

                _layerOps.Add(Mv(qkv, gNorm, gQkv));
                _layerOps.Add(Mv(gate, gNorm, gZ));
                _layerOps.Add(MvF32(alphaW, dim, nV, gNorm, gAlpha));
                _layerOps.Add(MvF32(betaW, dim, nV, gNorm, gBeta));
                _layerOps.Add(new Op { Name = "gdn_conv", K = _kConv, Set = _vk.MakeSet(_kConv, gQkv, convW, convState, gConvOut), Push = new uint[] { (uint)ch }, Gx = (uint)((ch + 255) / 256) });
                _layerOps.Add(new Op { Name = "gdn_l2", K = _kL2, Set = _vk.MakeSet(_kL2, gConvOut), Push = new uint[] { (uint)nK }, Gx = (uint)nK });
                _layerOps.Add(new Op { Name = "gdn_rec", K = _kRec, Set = _vk.MakeSet(_kRec, gConvOut, gAlpha, gBeta, dtBias, aW, recState, gY), Push = new uint[] { (uint)nK, (uint)nV }, Gx = (uint)nV });
                _layerOps.Add(Rms(gY, ssmNorm, gY, StateDim, nV));
                _layerOps.Add(Swiglu(gZ, gY, gGated, inner));
                _layerOps.Add(Mv(outW, gGated, gProj));
                _layerOps.Add(Add(gX, gProj, dim));
            }
            else
            {
                var qW = UploadMat(L.QWeight, L.QType, dim, 2 * qDim);
                var kW = UploadMat(L.KWeight, L.KType, dim, kvDim);
                var vW = UploadMat(L.VWeight, L.VType, dim, kvDim);
                var oW = UploadMat(L.AttnOutWeight, L.AttnOutType, qDim, dim);
                var qNorm = UploadF32(L.AttnQNormWeight, HeadDim);
                var kNorm = UploadF32(L.AttnKNormWeight, HeadDim);
                var kCache = _vk.Alloc((long)nKv * _maxSeq * HeadDim * 4);
                var vCache = _vk.Alloc((long)nKv * _maxSeq * HeadDim * 4);

                _layerOps.Add(Mv(qW, gNorm, gQFull));
                _layerOps.Add(Mv(kW, gNorm, gK));
                _layerOps.Add(Mv(vW, gNorm, gV));
                _layerOps.Add(new Op
                {
                    Name = "attn_prep", K = _kPrep, Set = _vk.MakeSet(_kPrep, gQFull, gK, gV, qNorm, kNorm, gQ, gGate, kCache, vCache),
                    Push = new uint[] { 0, (uint)nH, (uint)nKv, (uint)_maxSeq, (uint)w.RopeDimensionCount, VulkanCompute.F(w.RmsNormEps), VulkanCompute.F(w.RopeFreqBase) },
                    Gx = (uint)(nH + 2 * nKv), Patch = Patch.Pos
                });
                _layerOps.Add(new Op
                {
                    Name = "attn_decode", K = _kDec, Set = _vk.MakeSet(_kDec, gQ, gGate, kCache, vCache, gScores, gAttnOut),
                    Push = new uint[] { 0, (uint)nH, (uint)nKv, (uint)_maxSeq, VulkanCompute.F(scale) },
                    Gx = (uint)nH, Patch = Patch.KvLen
                });
                _layerOps.Add(Mv(oW, gAttnOut, gProj));
                _layerOps.Add(Add(gX, gProj, dim));
            }

            // ---- FFN ----
            _layerOps.Add(Rms(gX, ffnNorm, gNorm, dim, 1));
            if (L.IsMoe)
            {
                var router = UploadF32(L.FfnGateInpWeight, (long)E * dim);
                long gateSlice = (long)F * GgufTypes.GetRowBytes(L.FfnGateExpsType, dim);
                long upSlice = (long)F * GgufTypes.GetRowBytes(L.FfnUpExpsType, dim);
                long downSlice = (long)dim * GgufTypes.GetRowBytes(L.FfnDownExpsType, F);
                var gateE = UploadMat(L.FfnGateExpsWeight, L.FfnGateExpsType, dim, E * F);
                var upE = UploadMat(L.FfnUpExpsWeight, L.FfnUpExpsType, dim, E * F);
                var downE = UploadMat(L.FfnDownExpsWeight, L.FfnDownExpsType, F, E * dim);
                if (L.FfnGateExpsBias != null || L.FfnUpExpsBias != null || L.FfnDownExpsBias != null)
                    throw new NotSupportedException("Vulkan runner does not support expert biases.");

                int sF = F * 2;
                if (w.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null) sF = (int)tShexp.Dimensions[1];
                var sgW = UploadMat(L.FfnGateShexpWeight, L.FfnGateShexpType, dim, sF);
                var suW = UploadMat(L.FfnUpShexpWeight, L.FfnUpShexpType, dim, sF);
                var sdW = UploadMat(L.FfnDownShexpWeight, L.FfnDownShexpType, sF, dim);
                bool hasShGate = L.FfnGateInpShexpWeight != null;
                var shGate = hasShGate ? UploadF32(L.FfnGateInpShexpWeight, dim) : gDummy;

                _layerOps.Add(MvF32(router, dim, E, gNorm, gRouterLogits));
                _layerOps.Add(new Op
                {
                    Name = "moe_topk", K = _kTopk, Set = _vk.MakeSet(_kTopk, gRouterLogits, gIdx, gTw),
                    Push = new uint[] { (uint)E, (uint)topK, w.NormTopK ? 1u : 0u }, Gx = 1
                });
                _layerOps.Add(MvMoe(gateE, dim, F, gateSlice, 0, gNorm, gGateOut, gIdx, topK));
                _layerOps.Add(MvMoe(upE, dim, F, upSlice, 0, gNorm, gUpOut, gIdx, topK));
                _layerOps.Add(Swiglu(gGateOut, gUpOut, gAct, topK * F));
                _layerOps.Add(MvMoe(downE, F, dim, downSlice, F, gAct, gDownOut, gIdx, topK));
                _layerOps.Add(Mv(sgW, gNorm, gSg));
                _layerOps.Add(Mv(suW, gNorm, gSu));
                _layerOps.Add(Swiglu(gSg, gSu, gSAct, sF));
                _layerOps.Add(Mv(sdW, gSAct, gSd));
                _layerOps.Add(new Op
                {
                    Name = "moe_combine", K = _kComb, Set = _vk.MakeSet(_kComb, gX, gNorm, shGate, gDownOut, gTw, gSd),
                    Push = new uint[] { (uint)dim, (uint)topK, hasShGate ? 1u : 0u }, Gx = 1
                });
            }
            else
            {
                var gW = UploadMat(L.FfnGateWeight, L.FfnGateType, dim, ffn);
                var uW = UploadMat(L.FfnUpWeight, L.FfnUpType, dim, ffn);
                var dW = UploadMat(L.FfnDownWeight, L.FfnDownType, ffn, dim);
                _layerOps.Add(Mv(gW, gNorm, gGateOut));
                _layerOps.Add(Mv(uW, gNorm, gUpOut));
                _layerOps.Add(Swiglu(gGateOut, gUpOut, gAct, ffn));
                _layerOps.Add(Mv(dW, gAct, gProj));
                _layerOps.Add(Add(gX, gProj, dim));
            }
        }

        // ---- output head ----
        _finalOps.Add(Rms(gX, outNorm, gNorm, dim, 1));
        _finalOps.Add(Mv(outMat, gNorm, _gLogits));

        // ---- embedding (allocated last: only one row is read per token; see the allocation-order note above) ----
        var embdBuf = _vk.AllocFrom(w.EmbdWeight, (long)_vocab * GgufTypes.GetRowBytes(w.EmbdType, dim));
        _embedOps.Add(new Op
        {
            Name = "embed", K = _kEmbed[QIdx(w.EmbdType)], Set = _vk.MakeSet(_kEmbed[QIdx(w.EmbdType)], embdBuf, gX),
            Push = new uint[] { (uint)dim, 0 }, Gx = 1, Patch = Patch.Token
        });
        GlacierDiagnostics.LogInformation($"[Vulkan GPU] Model ready on {_ctx.DeviceName}: {_embedOps.Count + _layerOps.Count + _finalOps.Count} dispatches per token.");
    }

    // ------------------------------------------------------------------ execution

    private void Run(List<Op> ops, int token, int pos)
    {
        for (int i = 0; i < ops.Count; i++)
        {
            var op = ops[i];
            switch (op.Patch)
            {
                case Patch.Pos: op.Push[0] = (uint)pos; break;
                case Patch.KvLen: op.Push[0] = (uint)(pos + 1); break;
                case Patch.Token: op.Push[1] = (uint)token; break;
            }
            _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
        }
    }


    /// <summary>
    /// Runs one token through the model on the GPU. When <paramref name="computeLogits"/> is true and <paramref name="logits"/>
    /// has at least <c>VocabSize</c> elements, the output logits are copied into it; otherwise the output head is skipped.
    /// Not thread-safe: calls must be serialised.
    /// </summary>
    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)pos >= (uint)_maxSeq) throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position exceeds MaxSeqLen ({_maxSeq}).");
        if ((uint)token >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(token), token, "Token id outside vocabulary.");
        bool wantLogits = computeLogits && logits.Length >= _vocab;

        _vk.Begin();
        Run(_embedOps, token, pos);
        Run(_layerOps, token, pos);
        if (wantLogits) Run(_finalOps, token, pos);
        _vk.SubmitAndWait();

        if (wantLogits)
        {
            fixed (float* p = logits) _gLogits.DownloadTo(p, (long)_vocab * 4);
        }
    }

    /// <summary>Processes prompt tokens one at a time (no batched GPU prefill yet); logits are produced for the last token only.</summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool last = i == tokens.Length - 1;
            Forward(tokens[i], startPos + i, last && computeLogits ? logits : Span<float>.Empty, last && computeLogits);
        }
    }

    public void ResetKvCache()
    {
        // KV positions are overwritten as new tokens are processed and attention only reads [0, pos],
        // so only the recurrent / convolution states need clearing.
        foreach (var (conv, rec) in _states)
        {
            conv.Zero();
            rec.Zero();
        }
    }

    /// <summary>
    /// Estimates GPU time per token by kernel category, as a text table. Every op of the plan is replayed <paramref name="reps"/> times
    /// back-to-back in one command buffer and the results are grouped by kernel name. Replays reuse the same pages, so
    /// bandwidth-bound kernels look faster than they do in a real token (cold weights); use the table to see relative
    /// cost, not absolute speed. Call <see cref="Forward"/> at least once first (the MoE ops need valid expert indices).
    /// Recurrent state is reset afterwards.
    /// </summary>
    public string ProfileReport(int reps = 20)
    {
        var totals = new Dictionary<string, (int Count, double Ms)>();
        int pos = Math.Min(16, _maxSeq - 1);
        foreach (var list in new[] { _embedOps, _layerOps, _finalOps })
        {
            foreach (var op in list)
            {
                switch (op.Patch)
                {
                    case Patch.Pos: op.Push[0] = (uint)pos; break;
                    case Patch.KvLen: op.Push[0] = (uint)(pos + 1); break;
                    case Patch.Token: op.Push[1] = 1; break;
                }
                _vk.Begin();
                for (int r = 0; r < reps; r++) _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
                double ms = _vk.SubmitAndWait() / reps;
                totals.TryGetValue(op.Name, out var t);
                totals[op.Name] = (t.Count + 1, t.Ms + ms);
            }
        }
        ResetKvCache();

        var sb = new System.Text.StringBuilder();
        double grand = 0;
        foreach (var kv in totals) grand += kv.Value.Ms;
        sb.AppendLine($"{"kernel",-18}{"calls",7}{"avg us",10}{"total ms",11}{"share",8}");
        foreach (var kv in System.Linq.Enumerable.OrderByDescending(totals, k => k.Value.Ms))
            sb.AppendLine($"{kv.Key,-18}{kv.Value.Count,7}{kv.Value.Ms / kv.Value.Count * 1000,10:F1}{kv.Value.Ms,11:F2}{kv.Value.Ms / grand * 100,7:F1}%");
        sb.AppendLine($"{"TOTAL",-18}{"",7}{"",10}{grand,11:F2}   (estimated ms/token => {1000.0 / grand:F1} tok/s)");
        return sb.ToString();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vk.Dispose();
        _ctx.Dispose();
    }
}
