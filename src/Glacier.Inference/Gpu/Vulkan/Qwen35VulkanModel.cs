namespace Glacier.Inference.Gpu;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// Vulkan compute inference runner for the Qwen 3.5 / 3.6 hybrid family (Gated DeltaNet + interleaved full attention,
/// dense or Mixture-of-Experts FFN). Works on any Vulkan 1.1 GPU (Linux, Windows, Android); the whole forward pass
/// for a token is recorded into one command buffer and only the logits are read back.
/// Supported quant types for weight matrices: Q4_K, Q5_K, Q6_K, Q8_0 (alpha/beta/router/norms are F32).
/// </summary>
public sealed unsafe class Qwen35VulkanModel : ID3D12Model, IHiddenStateSource
{
    private const int HeadDim = 256, StateDim = 128, ConvKernel = 4, BatchMax = 32;

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

    private sealed class TrellisGpuMat
    {
        public required VulkanCompute.GBuf CodeWords, S16, Pre, Post;
        public VulkanCompute.GBuf? Pin, Pout;
        public int K, N, Bits, RowsKind, RowsBlock, ColsKind, ColsBlock;
        public bool HasPin, HasPout;
    }

    private sealed class TrellisGpuBanded
    {
        public required TrellisGpuMat[] Bands;
        public required VulkanCompute.GBuf VocabGather;
        public int TotalN;
    }

    private sealed record Mat(VulkanCompute.GBuf? Buf, GgufType Type, int K, int N, TrellisGpuMat? Trellis = null, TrellisGpuBanded? Banded = null);

    private sealed class BatchLayerMeta
    {
        public VulkanCompute.GBuf AttnNorm = null!, FfnNorm = null!;
        public bool IsGdn, IsMoe;
        public Mat? Qkv, Gate, OutW, QW, KW, VW, OW, GW, UW, DW;
        public VulkanCompute.GBuf? QkvBias, AlphaW, BetaW, ConvW, ConvState, AW, DtBias, SsmNorm, RecState;
        public VulkanCompute.GBuf? QNorm, KNorm, KCache, VCache, QBias, KBias, VBias, OBias, GBias, UBias;
        public int FfnDim, MoeOpsStart, MoeOpsCount;

        // Lazily initialized batched descriptor sets
        public IntPtr BAttnNormSet, BFfnNormSet;
        public IntPtr BQkvSet, BQkvBiasSet, BGateSet, BAlphaSet, BBetaSet, BConvSet, BL2Set, BRecSet, BSsmNormSet, BSwiGdnSet, BOutSet;
        public IntPtr BQSet, BKSet, BVSet, BQBiasSet, BKBiasSet, BVBiasSet, BPrepSet, BDecSet, BOSet, BOBiasSet;
        public IntPtr BGSet, BUSet, BGBiasSet, BUBiasSet, BSwiFfnSet, BDSet;
    }

    private readonly ModelWeights _w;
    private readonly VulkanContext _ctx;
    private readonly VulkanCompute _vk;
    private readonly int _maxSeq, _dim, _vocab;
    private readonly List<Op> _embedOps = new(), _layerOps = new(), _finalOps = new();
    private readonly List<BatchLayerMeta> _batchMeta = new();
    private readonly List<(VulkanCompute.GBuf Conv, VulkanCompute.GBuf Rec)> _states = new();
    private readonly VulkanCompute.GBuf _gLogits;
    private VulkanCompute.GBuf _gX = null!, _gNorm = null!, _embdBuf = null!;
    private int _maxFfn;
    private readonly float[] _lastHidden;
    private float[]? _hiddenHistory;
    private bool[]? _historyValid;
    private bool _historyEnabled;
    private bool _disposed;

    // Kernels
    private readonly VulkanCompute.Kernel[] _kMv = new VulkanCompute.Kernel[12], _kMoe = new VulkanCompute.Kernel[12], _kEmbed = new VulkanCompute.Kernel[12];
    private readonly VulkanCompute.Kernel _kF32, _kRms, _kSwi, _kAdd, _kConv, _kL2, _kRec, _kTopk, _kComb, _kPrep, _kDec;
    private readonly VulkanCompute.Kernel[] _kMm = new VulkanCompute.Kernel[12], _kEmbedB = new VulkanCompute.Kernel[12];
    private readonly VulkanCompute.Kernel _kF32Mm, _kBiasBatch, _kCopy, _kConvB, _kL2B, _kRecB;
    private VulkanCompute.Kernel? _kTrlPre, _kTrlPre128, _kTrlGemv, _kTrlGemvSplitK, _kTrlPost, _kTrlPost128, _kTrlGather;
    private VulkanCompute.GBuf? _gTrlDummyIdx, _gTrlXRot, _gTrlYRot, _gTrlYScaled, _gTrlBandRaw;
    private float[]? _hostTrellisEmb;
    private bool _batchReady;
    private VulkanCompute.GBuf? _bTokens, _bX, _bNorm, _bProj, _bQkv, _bZ, _bAlpha, _bBeta, _bConvOut, _bY, _bGated;
    private VulkanCompute.GBuf? _bQFull, _bK, _bV, _bQ, _bGate, _bAttnOut, _bScores, _bGateOut, _bUpOut, _bAct;
    private IntPtr _bEmbedSet, _bAddProjSet, _bCopyBXToGX, _bCopyBNormToGNorm, _bCopyGXToBX;

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
        if (w.HeadCount % w.HeadCountKv != 0) { reason = "head count not divisible by KV head count"; return false; }
        foreach (var l in w.Layers)
        {
            if (!l.IsGdn && !l.HasQGate) { reason = "full-attention layer without Q gate"; return false; }
            if (!l.IsMoe && l.HasFusedGateUp && !IsKernelType(l.FfnUpType)) { reason = $"fused gate/up FFN uses unsupported weight type {l.FfnUpType}"; return false; }

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

    private static bool IsEmbedKernelType(GgufType t) => GgufTypes.IsTrellis(t) || t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.F16 or GgufType.BF16 or GgufType.Q4_0 or GgufType.Q3_K or GgufType.IQ4_NL or GgufType.Q2_K;
    private static bool IsKernelType(GgufType t) => GgufTypes.IsTrellis(t) || t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.IQ4_XS or GgufType.MXFP4 or GgufType.Q3_K or GgufType.IQ4_NL or GgufType.Q2_K or GgufType.Q4_0 or GgufType.F16 or GgufType.BF16;

    private static int QIdx(GgufType t) => t switch
    {
        GgufType.Q4_K => 0,
        GgufType.Q5_K => 1,
        GgufType.Q6_K => 2,
        GgufType.Q8_0 => 3,
        GgufType.IQ4_XS => 4,
        GgufType.MXFP4 => 5,
        GgufType.Q3_K => 6,
        GgufType.IQ4_NL => 7,
        GgufType.Q2_K => 8,
        GgufType.Q4_0 => 9,
        GgufType.F16 => 10,
        GgufType.BF16 => 11,
        _ => throw new NotSupportedException($"Vulkan runner does not support weight type {t}.")
    };

    private static bool HasEmbeddedShader(string name)
    {
        using Stream? s = typeof(VulkanCompute).Assembly.GetManifestResourceStream($"vkshader.{name}.spv");
        return s != null;
    }

    public Qwen35VulkanModel(ModelWeights weights, int maxSeqLen, int deviceOrdinal = 0)
    {
        if (!IsSupported(weights, out string why))
            throw new NotSupportedException($"Vulkan runner cannot execute this model: {why}.");

        _w = weights;
        _maxSeq = maxSeqLen;
        _dim = weights.EmbeddingLength;
        _vocab = weights.VocabSize;
        _lastHidden = new float[_dim];
        _ctx = new VulkanContext(deviceOrdinal);
        _vk = new VulkanCompute(_ctx);

        try
        {
            // ---- kernels ----
            for (int t = 0; t < 12; t++)
            {
                if (HasEmbeddedShader($"qmatvec_q{t}")) _kMv[t] = _vk.LoadEmbedded($"qmatvec_q{t}", 3, 8, 1u << 2);
                if (HasEmbeddedShader($"qmatvec_moe_q{t}")) _kMoe[t] = _vk.LoadEmbedded($"qmatvec_moe_q{t}", 4, 16, 1u << 2);
                if (HasEmbeddedShader($"embed_q{t}")) _kEmbed[t] = _vk.LoadEmbedded($"embed_q{t}", 2, 8, 1u << 1);
                if (HasEmbeddedShader($"qmatmat_q{t}")) _kMm[t] = _vk.LoadEmbedded($"qmatmat_q{t}", 3, 12, 1u << 2);
                if (HasEmbeddedShader($"embed_batch_q{t}")) _kEmbedB[t] = _vk.LoadEmbedded($"embed_batch_q{t}", 3, 8, 1u << 1);
            }
            _kF32 = _vk.LoadEmbedded("f32_matvec", 3, 8, 1u << 2);
            _kRms = _vk.LoadEmbedded("rmsnorm", 3, 12, 1u << 2);
            _kSwi = _vk.LoadEmbedded("elementwise_swiglu", 3, 4, 1u << 2);
            _kAdd = _vk.LoadEmbedded("elementwise_add", 3, 4, 1u << 0);
            _kConv = _vk.LoadEmbedded("gdn_conv", 4, 4, (1u << 2) | (1u << 3));
            _kL2 = _vk.LoadEmbedded("gdn_l2", 1, 4, 1u << 0);
            _kRec = _vk.LoadEmbedded("gdn_rec", 7, 8, (1u << 5) | (1u << 6));
            _kTopk = _vk.LoadEmbedded("moe_topk", 3, 12, (1u << 1) | (1u << 2));
            _kComb = _vk.LoadEmbedded("moe_combine", 6, 12, 1u << 0);
            _kPrep = _vk.LoadEmbedded("attn_prep", 9, 28, (1u << 5) | (1u << 6) | (1u << 7) | (1u << 8));
            _kDec = _vk.LoadEmbedded("attn_decode", 6, 20, (1u << 4) | (1u << 5));
            _kF32Mm = _vk.LoadEmbedded("f32_matmat", 3, 12, 1u << 2);
            _kBiasBatch = _vk.LoadEmbedded("bias_add_batch", 2, 8, 1u << 0);
            _kCopy = _vk.LoadEmbedded("copy_f32", 2, 12, 1u << 1);
            _kConvB = _vk.LoadEmbedded("gdn_conv_b", 4, 8, (1u << 2) | (1u << 3));
            _kL2B = _vk.LoadEmbedded("gdn_l2_b", 1, 8, 1u << 0);
            _kRecB = _vk.LoadEmbedded("gdn_rec_b", 7, 16, (1u << 5) | (1u << 6));
            if (weights.HasTrellisWeights && HasEmbeddedShader("trellis_gemv"))
            {
                _kTrlPre = _vk.LoadEmbedded("trellis_pre_rht", 5, 16, 1u << 3);
                if (_ctx.SupportsSubgroupShuffle && HasEmbeddedShader("trellis_pre_rht_128"))
                    _kTrlPre128 = _vk.LoadEmbedded("trellis_pre_rht_128", 5, 16, 1u << 3);
                string gemvName = _ctx.SupportsIntegerDotProduct && HasEmbeddedShader("trellis_gemv_dp4a")
                    ? "trellis_gemv_dp4a"
                    : "trellis_gemv";
                _kTrlGemv = _vk.LoadEmbedded(gemvName, 3, 12, 1u << 2);
                string splitKName = _ctx.SupportsIntegerDotProduct && HasEmbeddedShader("trellis_gemv_splitk_dp4a")
                    ? "trellis_gemv_splitk_dp4a"
                    : "trellis_gemv_splitk";
                if (HasEmbeddedShader(splitKName))
                    _kTrlGemvSplitK = _vk.LoadEmbedded(splitKName, 3, 12, 1u << 2);
                _kTrlPost = _vk.LoadEmbedded("trellis_post_rht", 3, 16, 1u << 2);
                if (_ctx.SupportsSubgroupShuffle && HasEmbeddedShader("trellis_post_rht_128"))
                    _kTrlPost128 = _vk.LoadEmbedded("trellis_post_rht_128", 3, 16, 1u << 2);
                _kTrlGather = _vk.LoadEmbedded("trellis_gather_f32", 3, 4, 1u << 2);
                _gTrlDummyIdx = _vk.Alloc(16);
                _gTrlDummyIdx.Zero();
            }

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

    private TrellisGpuMat UploadTrellisBinding(TrellisWeightBinding b)
    {
        int bits = GgufTypes.GetTrellisBits(b.Type);
        long codeBytes = (long)b.K * b.N * bits / 8;
        int s16Count = b.K / 16;
        long s16Bytes = Math.Max(4L, ((s16Count + 1) / 2) * 4L);
        VulkanCompute.GBuf s16Buf;
        if ((long)s16Count * sizeof(ushort) == s16Bytes)
        {
            s16Buf = _vk.AllocDeviceFrom(b.S16, s16Bytes);
        }
        else
        {
            ushort[] padded = new ushort[s16Bytes / 2];
            new ReadOnlySpan<ushort>(b.S16, s16Count).CopyTo(padded);
            fixed (ushort* ps = padded) s16Buf = _vk.AllocDeviceFrom(ps, s16Bytes);
        }
        return new TrellisGpuMat
        {
            K = b.K, N = b.N, Bits = bits,
            RowsKind = b.Descriptor[0], RowsBlock = b.Descriptor[1],
            ColsKind = b.Descriptor[3], ColsBlock = b.Descriptor[4],
            HasPin = b.Pin != null, HasPout = b.Pout != null,
            CodeWords = _vk.AllocDeviceFrom(b.CodeData, codeBytes),
            S16 = s16Buf,
            Pre = _vk.AllocDeviceFrom(b.Pre, (long)b.K * 4),
            Post = _vk.AllocDeviceFrom(b.Post, (long)b.N * 4),
            Pin = b.Pin != null ? _vk.AllocDeviceFrom(b.Pin, (long)b.K * 4) : null,
            Pout = b.Pout != null ? _vk.AllocDeviceFrom(b.Pout, (long)b.N * 4) : null,
        };
    }

    private Mat UploadMat(byte* p, GgufType type, int k, int n)
    {
        if (p == null) throw new InvalidOperationException("Missing weight tensor.");
        if (GgufTypes.IsTrellis(type))
        {
            if (TrellisWeightRegistry.TryGetBanded(p, out var banded) && banded != null)
            {
                var gpuBands = new TrellisGpuMat[banded.Bands.Length];
                for (int i = 0; i < banded.Bands.Length; i++)
                    gpuBands[i] = UploadTrellisBinding(banded.Bands[i]);
                var gatherBuf = _vk.AllocDeviceFrom(banded.VocabGather, (long)banded.TotalN * 4);
                return new Mat(null, type, k, banded.TotalN, Banded: new TrellisGpuBanded
                {
                    Bands = gpuBands,
                    VocabGather = gatherBuf,
                    TotalN = banded.TotalN,
                });
            }
            if (!TrellisWeightRegistry.TryGet(p, out var binding) || binding == null)
                throw new InvalidOperationException($"NariQ trellis weight ({type}) not found in TrellisWeightRegistry.");
            return new Mat(null, type, k, n, Trellis: UploadTrellisBinding(binding));
        }

        QIdx(type);
        long bytes = (long)n * GgufTypes.GetRowBytes(type, k);
        return new Mat(_vk.AllocDeviceFrom(p, bytes), type, k, n);
    }

    private VulkanCompute.GBuf UploadF32(void* p, long count)
    {
        if (p == null) throw new InvalidOperationException("Missing F32 tensor.");
        return _vk.AllocDeviceFrom(p, count * 4);
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
        fixed (float* src = rows) return _vk.AllocDeviceFrom(src, rows.Length * 4L);
    }

    private void AddTrellisMvOps(List<Op> list, TrellisGpuMat tm, VulkanCompute.GBuf x, VulkanCompute.GBuf y, int outOffset = 0)
    {
        bool usePre128 = tm.ColsKind == 1 && tm.ColsBlock == 128 && _kTrlPre128 != null;
        var kPre = usePre128 ? _kTrlPre128! : _kTrlPre!;
        int kt = tm.K >> 4;
        bool useSplitK = tm.N <= 6144 && kt >= 64 && _kTrlGemvSplitK != null;
        var kGemv = useSplitK ? _kTrlGemvSplitK! : _kTrlGemv!;
        bool usePost128 = tm.RowsKind == 1 && tm.RowsBlock == 128 && _kTrlPost128 != null;
        var kPost = usePost128 ? _kTrlPost128! : _kTrlPost!;
        var xRot = _gTrlXRot!;
        var yRot = _gTrlYRot!;
        var yTarget = tm.HasPout ? _gTrlYScaled! : y;

        uint preGrid = tm.ColsKind == 1 ? (uint)(tm.K / tm.ColsBlock) : (uint)((tm.K + 255) / 256);
        uint gemvGrid = useSplitK ? (uint)((tm.N + 15) / 16) : (uint)((tm.N + 63) / 64);
        uint postGrid = tm.RowsKind == 1 ? (uint)(tm.N / tm.RowsBlock) : (uint)((tm.N + 255) / 256);

        list.Add(new Op
        {
            Name = usePre128 ? $"trl_pre128 K{tm.K}" : $"trl_pre K{tm.K}",
            K = kPre,
            Set = _vk.MakeSet(kPre, x, tm.Pre, tm.Pin ?? _gTrlDummyIdx!, xRot, tm.S16),
            Push = new uint[] { (uint)tm.K, (uint)tm.ColsKind, (uint)tm.ColsBlock, tm.HasPin ? 1u : 0u },
            Gx = preGrid
        });
        list.Add(new Op
        {
            Name = useSplitK ? $"trl_gemv_sk b{tm.Bits} K{tm.K} N{tm.N}" : $"trl_gemv b{tm.Bits} K{tm.K} N{tm.N}",
            K = kGemv,
            Set = _vk.MakeSet(kGemv, tm.CodeWords, xRot, yRot),
            Push = new uint[] { (uint)tm.K, (uint)tm.N, (uint)tm.Bits },
            Gx = gemvGrid
        });
        list.Add(new Op
        {
            Name = usePost128 ? $"trl_post128 N{tm.N}" : $"trl_post N{tm.N}", K = kPost, Set = _vk.MakeSet(kPost, yRot, tm.Post, yTarget),
            Push = new uint[] { (uint)tm.N, (uint)tm.RowsKind, (uint)tm.RowsBlock, (uint)(tm.HasPout ? 0 : outOffset) }, Gx = postGrid
        });
        if (tm.HasPout)
        {
            var kGather = _kTrlGather!;
            list.Add(new Op
            {
                Name = $"trl_pout N{tm.N}", K = kGather, Set = _vk.MakeSet(kGather, yTarget, tm.Pout!, y),
                Push = new uint[] { (uint)tm.N }, Gx = (uint)((tm.N + 255) / 256)
            });
        }
    }

    private void AddMv(List<Op> list, Mat m, VulkanCompute.GBuf x, VulkanCompute.GBuf y)
    {
        if (m.Banded != null)
        {
            int offset = 0;
            for (int b = 0; b < m.Banded.Bands.Length; b++)
            {
                var band = m.Banded.Bands[b];
                AddTrellisMvOps(list, band, x, _gTrlBandRaw!, outOffset: offset);
                offset += band.N;
            }
            var kGather = _kTrlGather!;
            list.Add(new Op
            {
                Name = $"trl_vocab_gather N{m.Banded.TotalN}", K = kGather,
                Set = _vk.MakeSet(kGather, _gTrlBandRaw!, m.Banded.VocabGather, y),
                Push = new uint[] { (uint)m.Banded.TotalN }, Gx = (uint)((m.Banded.TotalN + 255) / 256)
            });
            return;
        }
        if (m.Trellis != null)
        {
            AddTrellisMvOps(list, m.Trellis, x, y);
            return;
        }
        list.Add(Mv(m, x, y));
    }

    private Op Mv(Mat m, VulkanCompute.GBuf x, VulkanCompute.GBuf y)
    {
        var k = _kMv[QIdx(m.Type)];
        int groups = (m.N + 3) / 4;
        return new Op
        {
            Name = $"mv_{m.Type} K{m.K} N{m.N}", K = k, Set = _vk.MakeSet(k, m.Buf!, x, y),
            Push = new uint[] { (uint)m.K, (uint)m.N },
            Gx = (uint)Math.Min(groups, 32768), Gy = (uint)((groups + 32767) / 32768)
        };
    }

    private Op MvF32(VulkanCompute.GBuf w, int k, int n, VulkanCompute.GBuf x, VulkanCompute.GBuf y) => new()
    {
        Name = "mv_F32", K = _kF32, Set = _vk.MakeSet(_kF32, w, x, y), Push = new uint[] { (uint)k, (uint)n }, Gx = (uint)((n + 3) / 4)
    };

    private Op MvMoe(Mat expsAsOne, int k, int n, long sliceBytes, int xStride, VulkanCompute.GBuf x, VulkanCompute.GBuf y, VulkanCompute.GBuf idx, int topK)
    {
        var kern = _kMoe[QIdx(expsAsOne.Type)];
        return new Op
        {
            Name = $"moe_mv_{expsAsOne.Type} K{k} N{n}", K = kern, Set = _vk.MakeSet(kern, expsAsOne.Buf, x, y, idx),
            Push = new uint[] { (uint)k, (uint)n, (uint)sliceBytes, (uint)xStride },
            Gx = (uint)((n + 3) / 4), Gy = (uint)topK
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
        _maxFfn = maxF;

        // ---- shared activations / scratch ----
        var gX = _vk.Alloc(dim * 4L); _gX = gX; var gNorm = _vk.AllocDeviceOnly(dim * 4L); _gNorm = gNorm; var gProj = _vk.AllocDeviceOnly(dim * 4L);
        var gQkv = _vk.AllocDeviceOnly(ch * 4L); var gZ = _vk.AllocDeviceOnly(inner * 4L); var gAlpha = _vk.AllocDeviceOnly(nV * 4L); var gBeta = _vk.AllocDeviceOnly(nV * 4L);
        var gConvOut = _vk.AllocDeviceOnly(ch * 4L); var gY = _vk.AllocDeviceOnly(inner * 4L); var gGated = _vk.AllocDeviceOnly(inner * 4L);
        var gQFull = _vk.AllocDeviceOnly(2L * qDim * 4); var gK = _vk.AllocDeviceOnly(kvDim * 4L); var gV = _vk.AllocDeviceOnly(kvDim * 4L);
        var gQ = _vk.AllocDeviceOnly(qDim * 4L); var gGate = _vk.AllocDeviceOnly(qDim * 4L); var gAttnOut = _vk.AllocDeviceOnly(qDim * 4L);
        var gScores = _vk.AllocDeviceOnly((long)nH * _maxSeq * 4);
        var gIdx = _vk.AllocDeviceOnly(topK * 4L); var gTw = _vk.AllocDeviceOnly(topK * 4L);
        var gGateOut = _vk.AllocDeviceOnly(maxF * 4L); var gUpOut = _vk.AllocDeviceOnly(maxF * 4L); var gAct = _vk.AllocDeviceOnly(maxF * 4L);
        var gDownOut = _vk.AllocDeviceOnly((long)topK * dim * 4);
        var gSg = _vk.AllocDeviceOnly(sFmax * 4L); var gSu = _vk.AllocDeviceOnly(sFmax * 4L); var gSAct = _vk.AllocDeviceOnly(sFmax * 4L); var gSd = _vk.AllocDeviceOnly(dim * 4L);
        var gDummy = _vk.AllocDeviceOnly(dim * 4L);
        var gRouterLogits = _vk.AllocDeviceOnly(Math.Max(E, 1) * 4L);
        if (w.HasTrellisWeights)
        {
            int trlMax = Math.Max(Math.Max(Math.Max(dim, maxF), Math.Max(ch, 2 * qDim)), 20480);
            _gTrlXRot = _vk.AllocDeviceOnly(trlMax * 4L);
            _gTrlYRot = _vk.AllocDeviceOnly(Math.Max(trlMax, _vocab) * 4L);
            _gTrlYScaled = _vk.AllocDeviceOnly(Math.Max(trlMax, _vocab) * 4L);
            if (w.Gguf.TrellisLmHeadBands > 0)
                _gTrlBandRaw = _vk.AllocDeviceOnly((long)_vocab * 4L);
        }

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
            var meta = new BatchLayerMeta { AttnNorm = attnNorm, FfnNorm = ffnNorm, IsGdn = L.IsGdn, IsMoe = L.IsMoe, FfnDim = ffn };
            _layerOps.Add(Rms(gX, attnNorm, gNorm, dim, 1));

            if (L.IsGdn)
            {
                if (L.SsmConv1dWeight == null)
                {
                    throw new InvalidDataException($"Missing SsmConv1dWeight for GDN layer {l}.");
                }

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

                meta.Qkv = qkv; meta.Gate = gate; meta.AlphaW = alphaW; meta.BetaW = betaW;
                meta.ConvW = convW; meta.ConvState = convState; meta.AW = aW; meta.DtBias = dtBias;
                meta.SsmNorm = ssmNorm; meta.RecState = recState; meta.OutW = outW;

                AddMv(_layerOps, qkv, gNorm, gQkv);
                if (L.QkvBias != null)
                {
                    var qkvBias = UploadF32(L.QkvBias, ch);
                    meta.QkvBias = qkvBias;
                    _layerOps.Add(Add(gQkv, qkvBias, ch));
                }
                AddMv(_layerOps, gate, gNorm, gZ);
                _layerOps.Add(MvF32(alphaW, dim, nV, gNorm, gAlpha));
                _layerOps.Add(MvF32(betaW, dim, nV, gNorm, gBeta));
                _layerOps.Add(new Op { Name = "gdn_conv", K = _kConv, Set = _vk.MakeSet(_kConv, gQkv, convW, convState, gConvOut), Push = new uint[] { (uint)ch }, Gx = (uint)((ch + 255) / 256) });
                _layerOps.Add(new Op { Name = "gdn_l2", K = _kL2, Set = _vk.MakeSet(_kL2, gConvOut), Push = new uint[] { (uint)nK }, Gx = (uint)nK });
                _layerOps.Add(new Op { Name = "gdn_rec", K = _kRec, Set = _vk.MakeSet(_kRec, gConvOut, gAlpha, gBeta, dtBias, aW, recState, gY), Push = new uint[] { (uint)nK, (uint)nV }, Gx = (uint)nV });
                _layerOps.Add(Rms(gY, ssmNorm, gY, StateDim, nV));
                _layerOps.Add(Swiglu(gZ, gY, gGated, inner));
                AddMv(_layerOps, outW, gGated, gProj);
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
                var kCache = _vk.AllocDeviceOnly((long)nKv * _maxSeq * HeadDim * 4);
                var vCache = _vk.AllocDeviceOnly((long)nKv * _maxSeq * HeadDim * 4);

                meta.QW = qW; meta.KW = kW; meta.VW = vW; meta.OW = oW;
                meta.QNorm = qNorm; meta.KNorm = kNorm; meta.KCache = kCache; meta.VCache = vCache;

                AddMv(_layerOps, qW, gNorm, gQFull);
                AddMv(_layerOps, kW, gNorm, gK);
                AddMv(_layerOps, vW, gNorm, gV);
                if (L.QBias != null)
                {
                    var qBias = UploadF32(L.QBias, 2 * qDim);
                    meta.QBias = qBias;
                    _layerOps.Add(Add(gQFull, qBias, 2 * qDim));
                }
                if (L.KBias != null)
                {
                    var kBias = UploadF32(L.KBias, kvDim);
                    meta.KBias = kBias;
                    _layerOps.Add(Add(gK, kBias, kvDim));
                }
                if (L.VBias != null)
                {
                    var vBias = UploadF32(L.VBias, kvDim);
                    meta.VBias = vBias;
                    _layerOps.Add(Add(gV, vBias, kvDim));
                }
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
                AddMv(_layerOps, oW, gAttnOut, gProj);
                if (L.AttnOutBias != null)
                {
                    var oBias = UploadF32(L.AttnOutBias, dim);
                    meta.OBias = oBias;
                    _layerOps.Add(Add(gProj, oBias, dim));
                }
                _layerOps.Add(Add(gX, gProj, dim));
            }

            // ---- FFN ----
            _layerOps.Add(Rms(gX, ffnNorm, gNorm, dim, 1));
            if (L.IsMoe)
            {
                meta.MoeOpsStart = _layerOps.Count;
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
                meta.MoeOpsCount = _layerOps.Count - meta.MoeOpsStart;
            }
            else
            {
                if (L.HasFusedGateUp)
                {
                    long rowBytes = GgufTypes.GetRowBytes(L.FfnUpType, dim);
                    var gW = UploadMat(L.FfnUpWeight, L.FfnUpType, dim, ffn);
                    var uW = UploadMat(L.FfnUpWeight + (long)ffn * rowBytes, L.FfnUpType, dim, ffn);
                    var dW = UploadMat(L.FfnDownWeight, L.FfnDownType, ffn, dim);
                    meta.GW = gW; meta.UW = uW; meta.DW = dW;
                    AddMv(_layerOps, gW, gNorm, gGateOut);
                    AddMv(_layerOps, uW, gNorm, gUpOut);
                    if (L.FfnUpBias != null)
                    {
                        var gBias = UploadF32(L.FfnUpBias, ffn);
                        _layerOps.Add(Add(gGateOut, gBias, ffn));
                        var uBias = UploadF32(L.FfnUpBias + ffn, ffn);
                        _layerOps.Add(Add(gUpOut, uBias, ffn));
                        meta.GBias = gBias; meta.UBias = uBias;
                    }
                    _layerOps.Add(Swiglu(gGateOut, gUpOut, gAct, ffn));
                    AddMv(_layerOps, dW, gAct, gProj);
                    _layerOps.Add(Add(gX, gProj, dim));
                }
                else
                {
                    var gW = UploadMat(L.FfnGateWeight, L.FfnGateType, dim, ffn);
                    var uW = UploadMat(L.FfnUpWeight, L.FfnUpType, dim, ffn);
                    var dW = UploadMat(L.FfnDownWeight, L.FfnDownType, ffn, dim);
                    meta.GW = gW; meta.UW = uW; meta.DW = dW;
                    AddMv(_layerOps, gW, gNorm, gGateOut);
                    AddMv(_layerOps, uW, gNorm, gUpOut);
                    if (L.FfnUpBias != null)
                    {
                        var gBias = UploadF32(L.FfnUpBias, ffn);
                        _layerOps.Add(Add(gGateOut, gBias, ffn));
                        var uBias = UploadF32(L.FfnUpBias + ffn, ffn);
                        _layerOps.Add(Add(gUpOut, uBias, ffn));
                        meta.GBias = gBias; meta.UBias = uBias;
                    }
                    _layerOps.Add(Swiglu(gGateOut, gUpOut, gAct, ffn));
                    AddMv(_layerOps, dW, gAct, gProj);
                    _layerOps.Add(Add(gX, gProj, dim));
                }
            }
            _batchMeta.Add(meta);
        }

        // ---- output head ----
        _finalOps.Add(Rms(gX, outNorm, gNorm, dim, 1));
        AddMv(_finalOps, outMat, gNorm, _gLogits);

        // ---- embedding (allocated last: only one row is read per token; see the allocation-order note above) ----
        if (GgufTypes.IsTrellis(w.EmbdType))
        {
            // Decode 1 embedding row on CPU (<0.05 ms) and upload 20 KB to _gX per token (saves ~0.8 GB VRAM!)
            _hostTrellisEmb = new float[dim];
        }
        else
        {
            var embdBuf = _vk.AllocDeviceFrom(w.EmbdWeight, (long)_vocab * GgufTypes.GetRowBytes(w.EmbdType, dim));
            _embdBuf = embdBuf;
            _embedOps.Add(new Op
            {
                Name = "embed", K = _kEmbed[QIdx(w.EmbdType)], Set = _vk.MakeSet(_kEmbed[QIdx(w.EmbdType)], embdBuf, gX),
                Push = new uint[] { (uint)dim, 0 }, Gx = 1, Patch = Patch.Token
            });
        }
        GlacierDiagnostics.LogInformation($"[Vulkan GPU] Model ready on {_ctx.DeviceName}: {_embedOps.Count + _layerOps.Count + _finalOps.Count} dispatches per token (staging={_ctx.PreferDeviceLocalStaging}, dp4a={_ctx.SupportsIntegerDotProduct}, sgShfl={_ctx.SupportsSubgroupShuffle}).");
        if (Environment.GetEnvironmentVariable("GLACIER_VK_PROFILE") == "1")
        {
            GlacierDiagnostics.LogInformation("[Vulkan GPU] ProfileReport:\n" + ProfileReport(5));
        }
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


    public int HiddenDim => _dim;
    public int VocabSize => _vocab;
    public bool SupportsPerStepVerificationSnapshots => _states.Count > 0;
    public long StateSnapshotBytes
    {
        get
        {
            long total = 0;
            foreach (var (conv, rec) in _states)
            {
                total += (long)conv.Size + (long)rec.Size;
            }
            return total;
        }
    }

    public ReadOnlySpan<float> GetLastHiddenState() => _lastHidden;

    public void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        if ((uint)tokenId >= (uint)_vocab)
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, "Token id outside vocabulary.");
        if (destination.Length < _dim)
            throw new ArgumentException($"Destination span length {destination.Length} < HiddenDim ({_dim}).", nameof(destination));

        fixed (float* dst = destination)
        {
            Glacier.Inference.Quant.QuantKernels.ExtractEmbedding(_w.EmbdType, _w.EmbdWeight, tokenId, dst, _dim);
        }
    }

    public void EnablePositionHistory()
    {
        _hiddenHistory ??= new float[(long)_maxSeq * _dim];
        _historyValid ??= new bool[_maxSeq];
        _historyEnabled = true;
    }

    public bool TryGetHiddenStateAt(int position, Span<float> destination)
    {
        if (!_historyEnabled || _hiddenHistory == null || _historyValid == null ||
            (uint)position >= (uint)_maxSeq || !_historyValid[position] || destination.Length < _dim)
        {
            return false;
        }

        _hiddenHistory.AsSpan(position * _dim, _dim).CopyTo(destination.Slice(0, _dim));
        return true;
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

        if (_hostTrellisEmb != null)
        {
            fixed (float* dst = _hostTrellisEmb)
            {
                QuantKernels.ExtractEmbedding(_w.EmbdType, _w.EmbdWeight, token, dst, _dim);
                _gX.UploadFrom(dst, (long)_dim * 4);
            }
        }

        _vk.Begin();
        if (_embedOps.Count > 0) Run(_embedOps, token, pos);
        Run(_layerOps, token, pos);
        if (wantLogits) Run(_finalOps, token, pos);
        _vk.SubmitAndWait();

        if (wantLogits)
        {
            fixed (float* p = logits) _gLogits.DownloadTo(p, (long)_vocab * 4);
        }
        if (wantLogits || _historyEnabled)
        {
            fixed (float* pHid = _lastHidden) _gX.DownloadTo(pHid, (long)_dim * 4);
            if (_historyEnabled && _hiddenHistory != null && _historyValid != null)
            {
                _lastHidden.AsSpan().CopyTo(_hiddenHistory.AsSpan(pos * _dim, _dim));
                _historyValid[pos] = true;
            }
        }
    }

    private void DispatchQmm(Mat m, IntPtr set, int b)
    {
        var kern = _kMm[QIdx(m.Type)];
        int groups = (m.N + 3) / 4;
        uint gx = (uint)Math.Min(groups, 32768);
        uint gy = (uint)((b + 7) / 8);
        uint gz = (uint)((groups + 32767) / 32768);
        _vk.Dispatch(kern, set, new uint[] { (uint)m.K, (uint)m.N, (uint)b }, gx, gy, gz);
    }

    private void DispatchBiasBatch(IntPtr set, int len, int b)
    {
        uint gx = (uint)((len * b + 255) / 256);
        _vk.Dispatch(_kBiasBatch, set, new uint[] { (uint)len, (uint)b }, gx);
    }

    private void EnsureBatchState()
    {
        if (_batchReady) return;
        _batchReady = true;

        var w = _w;
        int dim = _dim, nH = w.HeadCount, nKv = w.HeadCountKv;
        int qDim = nH * HeadDim, kvDim = nKv * HeadDim;
        int nK = w.SsmGroupCount, nV = w.SsmTimeStepRank, ch = (2 * nK + nV) * StateDim, inner = w.SsmInnerSize;

        _bTokens = _vk.Alloc(BatchMax * 4L);
        _bX = _vk.AllocDeviceOnly((long)BatchMax * dim * 4L);
        _bNorm = _vk.AllocDeviceOnly((long)BatchMax * dim * 4L);
        _bProj = _vk.AllocDeviceOnly((long)BatchMax * dim * 4L);
        _bQkv = _vk.AllocDeviceOnly((long)BatchMax * ch * 4L);
        _bConvOut = _vk.AllocDeviceOnly((long)BatchMax * ch * 4L);
        _bZ = _vk.AllocDeviceOnly((long)BatchMax * inner * 4L);
        _bAlpha = _vk.AllocDeviceOnly((long)BatchMax * nV * 4L);
        _bBeta = _vk.AllocDeviceOnly((long)BatchMax * nV * 4L);
        _bY = _vk.AllocDeviceOnly((long)BatchMax * inner * 4L);
        _bGated = _vk.AllocDeviceOnly((long)BatchMax * inner * 4L);
        _bQFull = _vk.AllocDeviceOnly((long)BatchMax * 2L * qDim * 4L);
        _bK = _vk.AllocDeviceOnly((long)BatchMax * kvDim * 4L);
        _bV = _vk.AllocDeviceOnly((long)BatchMax * kvDim * 4L);
        _bQ = _vk.AllocDeviceOnly((long)BatchMax * qDim * 4L);
        _bGate = _vk.AllocDeviceOnly((long)BatchMax * qDim * 4L);
        _bAttnOut = _vk.AllocDeviceOnly((long)BatchMax * qDim * 4L);
        _bScores = _vk.AllocDeviceOnly((long)BatchMax * nH * _maxSeq * 4L);
        _bGateOut = _vk.AllocDeviceOnly((long)BatchMax * _maxFfn * 4L);
        _bUpOut = _vk.AllocDeviceOnly((long)BatchMax * _maxFfn * 4L);
        _bAct = _vk.AllocDeviceOnly((long)BatchMax * _maxFfn * 4L);

        _bEmbedSet = _vk.MakeSet(_kEmbedB[QIdx(w.EmbdType)], _embdBuf, _bX, _bTokens);
        _bAddProjSet = _vk.MakeSet(_kAdd, _bX, _bProj, _bProj);
        _bCopyBXToGX = _vk.MakeSet(_kCopy, _bX, _gX);
        _bCopyBNormToGNorm = _vk.MakeSet(_kCopy, _bNorm, _gNorm);
        _bCopyGXToBX = _vk.MakeSet(_kCopy, _gX, _bX);

        foreach (var m in _batchMeta)
        {
            m.BAttnNormSet = _vk.MakeSet(_kRms, _bX, m.AttnNorm, _bNorm);
            m.BFfnNormSet = _vk.MakeSet(_kRms, _bX, m.FfnNorm, _bNorm);

            if (m.IsGdn)
            {
                var kQkv = _kMm[QIdx(m.Qkv!.Type)];
                var kGate = _kMm[QIdx(m.Gate!.Type)];
                var kOut = _kMm[QIdx(m.OutW!.Type)];
                m.BQkvSet = _vk.MakeSet(kQkv, m.Qkv.Buf, _bNorm, _bQkv);
                if (m.QkvBias != null) m.BQkvBiasSet = _vk.MakeSet(_kBiasBatch, _bQkv, m.QkvBias);
                m.BGateSet = _vk.MakeSet(kGate, m.Gate.Buf, _bNorm, _bZ);
                m.BAlphaSet = _vk.MakeSet(_kF32Mm, m.AlphaW!, _bNorm, _bAlpha);
                m.BBetaSet = _vk.MakeSet(_kF32Mm, m.BetaW!, _bNorm, _bBeta);
                m.BConvSet = _vk.MakeSet(_kConvB, _bQkv, m.ConvW!, m.ConvState!, _bConvOut);
                m.BL2Set = _vk.MakeSet(_kL2B, _bConvOut);
                m.BRecSet = _vk.MakeSet(_kRecB, _bConvOut, _bAlpha, _bBeta, m.DtBias!, m.AW!, m.RecState!, _bY);
                m.BSsmNormSet = _vk.MakeSet(_kRms, _bY, m.SsmNorm!, _bY);
                m.BSwiGdnSet = _vk.MakeSet(_kSwi, _bZ, _bY, _bGated);
                m.BOutSet = _vk.MakeSet(kOut, m.OutW.Buf, _bGated, _bProj);
            }
            else
            {
                m.BQSet = _vk.MakeSet(_kMm[QIdx(m.QW!.Type)], m.QW.Buf, _bNorm, _bQFull);
                m.BKSet = _vk.MakeSet(_kMm[QIdx(m.KW!.Type)], m.KW.Buf, _bNorm, _bK);
                m.BVSet = _vk.MakeSet(_kMm[QIdx(m.VW!.Type)], m.VW.Buf, _bNorm, _bV);
                if (m.QBias != null) m.BQBiasSet = _vk.MakeSet(_kBiasBatch, _bQFull, m.QBias);
                if (m.KBias != null) m.BKBiasSet = _vk.MakeSet(_kBiasBatch, _bK, m.KBias);
                if (m.VBias != null) m.BVBiasSet = _vk.MakeSet(_kBiasBatch, _bV, m.VBias);
                m.BPrepSet = _vk.MakeSet(_kPrep, _bQFull, _bK, _bV, m.QNorm!, m.KNorm!, _bQ, _bGate, m.KCache!, m.VCache!);
                m.BDecSet = _vk.MakeSet(_kDec, _bQ, _bGate, m.KCache!, m.VCache!, _bScores, _bAttnOut);
                m.BOSet = _vk.MakeSet(_kMm[QIdx(m.OW!.Type)], m.OW.Buf, _bAttnOut, _bProj);
                if (m.OBias != null) m.BOBiasSet = _vk.MakeSet(_kBiasBatch, _bProj, m.OBias);
            }

            if (!m.IsMoe)
            {
                m.BGSet = _vk.MakeSet(_kMm[QIdx(m.GW!.Type)], m.GW.Buf, _bNorm, _bGateOut);
                m.BUSet = _vk.MakeSet(_kMm[QIdx(m.UW!.Type)], m.UW.Buf, _bNorm, _bUpOut);
                if (m.GBias != null) m.BGBiasSet = _vk.MakeSet(_kBiasBatch, _bGateOut, m.GBias);
                if (m.UBias != null) m.BUBiasSet = _vk.MakeSet(_kBiasBatch, _bUpOut, m.UBias);
                m.BSwiFfnSet = _vk.MakeSet(_kSwi, _bGateOut, _bUpOut, _bAct);
                m.BDSet = _vk.MakeSet(_kMm[QIdx(m.DW!.Type)], m.DW.Buf, _bAct, _bProj);
            }
        }
    }

    /// <summary>
    /// Processes prompt tokens in batched chunks of up to 32 tokens with batched prefill projections,
    /// full-attention prep/decode, and chunked GDN state stepping. Logits are computed for the final token only.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return;
        if ((uint)startPos >= (uint)_maxSeq || tokens.Length > _maxSeq - startPos)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, $"Sequence range [{startPos}, {startPos + tokens.Length}) exceeds MaxSeqLen ({_maxSeq}).");
        for (int i = 0; i < tokens.Length; i++)
        {
            if ((uint)tokens[i] >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(tokens), tokens[i], "Token id outside vocabulary.");
        }

        if (_w.HasTrellisWeights)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                Forward(tokens[i], startPos + i, logits, computeLogits && i == tokens.Length - 1);
            }
            return;
        }

        if (tokens.Length == 1 || _historyEnabled || _kEmbedB[QIdx(_w.EmbdType)] == null)
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                bool last = i == tokens.Length - 1;
                Forward(tokens[i], startPos + i, last && computeLogits ? logits : Span<float>.Empty, last && computeLogits);
            }
            return;
        }

        EnsureBatchState();
        bool wantLogits = computeLogits && logits.Length >= _vocab;

        var w = _w;
        int dim = _dim, nH = w.HeadCount, nKv = w.HeadCountKv;
        int kvDim = nKv * HeadDim, qDim = nH * HeadDim;
        int nK = w.SsmGroupCount, nV = w.SsmTimeStepRank, ch = (2 * nK + nV) * StateDim, inner = w.SsmInnerSize;
        uint epsBits = VulkanCompute.F(w.RmsNormEps);
        uint freqBits = VulkanCompute.F(w.RopeFreqBase);
        uint scaleBits = VulkanCompute.F(1f / MathF.Sqrt(HeadDim));
        uint gxDim = (uint)((dim + 255) / 256);

        uint* tokUpload = stackalloc uint[BatchMax];
        int offset = 0;
        while (offset < tokens.Length)
        {
            int b = Math.Min(BatchMax, tokens.Length - offset);
            int chunkPos = startPos + offset;
            bool isLastChunk = offset + b == tokens.Length;

            for (int i = 0; i < b; i++) tokUpload[i] = (uint)tokens[offset + i];
            _bTokens!.UploadFrom(tokUpload, b * 4L);

            _vk.Begin();
            _vk.Dispatch(_kEmbedB[QIdx(w.EmbdType)], _bEmbedSet, new uint[] { (uint)dim, (uint)b }, (uint)b);

            uint gxBDim = (uint)((b * dim + 255) / 256);
            uint gxCh = (uint)((ch + 255) / 256);
            uint gyB8 = (uint)((b + 7) / 8);

            for (int l = 0; l < _batchMeta.Count; l++)
            {
                var m = _batchMeta[l];
                _vk.Dispatch(_kRms, m.BAttnNormSet, new uint[] { (uint)dim, (uint)b, epsBits }, (uint)b);

                if (m.IsGdn)
                {
                    DispatchQmm(m.Qkv!, m.BQkvSet, b);
                    if (m.QkvBias != null) DispatchBiasBatch(m.BQkvBiasSet, ch, b);
                    DispatchQmm(m.Gate!, m.BGateSet, b);
                    _vk.Dispatch(_kF32Mm, m.BAlphaSet, new uint[] { (uint)dim, (uint)nV, (uint)b }, (uint)((nV + 3) / 4), gyB8);
                    _vk.Dispatch(_kF32Mm, m.BBetaSet, new uint[] { (uint)dim, (uint)nV, (uint)b }, (uint)((nV + 3) / 4), gyB8);

                    for (int t = 0; t < b; t++)
                    {
                        _vk.Dispatch(_kConvB, m.BConvSet, new uint[] { (uint)ch, (uint)t }, gxCh);
                    }
                    _vk.Dispatch(_kL2B, m.BL2Set, new uint[] { (uint)nK, (uint)ch }, (uint)nK, (uint)b);
                    for (int t = 0; t < b; t++)
                    {
                        _vk.Dispatch(_kRecB, m.BRecSet, new uint[] { (uint)nK, (uint)nV, (uint)ch, (uint)t }, (uint)nV);
                    }

                    _vk.Dispatch(_kRms, m.BSsmNormSet, new uint[] { (uint)StateDim, (uint)(b * nV), epsBits }, (uint)(b * nV));
                    _vk.Dispatch(_kSwi, m.BSwiGdnSet, new uint[] { (uint)(b * inner) }, (uint)((b * inner + 255) / 256));
                    DispatchQmm(m.OutW!, m.BOutSet, b);
                    _vk.Dispatch(_kAdd, _bAddProjSet, new uint[] { (uint)(b * dim) }, gxBDim);
                }
                else
                {
                    DispatchQmm(m.QW!, m.BQSet, b);
                    DispatchQmm(m.KW!, m.BKSet, b);
                    DispatchQmm(m.VW!, m.BVSet, b);
                    if (m.QBias != null) DispatchBiasBatch(m.BQBiasSet, 2 * qDim, b);
                    if (m.KBias != null) DispatchBiasBatch(m.BKBiasSet, kvDim, b);
                    if (m.VBias != null) DispatchBiasBatch(m.BVBiasSet, kvDim, b);

                    _vk.Dispatch(
                        _kPrep,
                        m.BPrepSet,
                        new uint[] { (uint)chunkPos, (uint)nH, (uint)nKv, (uint)_maxSeq, (uint)w.RopeDimensionCount, epsBits, freqBits },
                        (uint)(nH + 2 * nKv),
                        (uint)b);
                    _vk.Dispatch(
                        _kDec,
                        m.BDecSet,
                        new uint[] { (uint)(chunkPos + 1), (uint)nH, (uint)nKv, (uint)_maxSeq, scaleBits },
                        (uint)nH,
                        (uint)b);
                    DispatchQmm(m.OW!, m.BOSet, b);
                    if (m.OBias != null) DispatchBiasBatch(m.BOBiasSet, dim, b);
                    _vk.Dispatch(_kAdd, _bAddProjSet, new uint[] { (uint)(b * dim) }, gxBDim);
                }

                // FFN
                _vk.Dispatch(_kRms, m.BFfnNormSet, new uint[] { (uint)dim, (uint)b, epsBits }, (uint)b);
                if (m.IsMoe)
                {
                    for (int t = 0; t < b; t++)
                    {
                        _vk.Dispatch(_kCopy, _bCopyBXToGX, new uint[] { (uint)dim, (uint)(t * dim), 0u }, gxDim);
                        _vk.Dispatch(_kCopy, _bCopyBNormToGNorm, new uint[] { (uint)dim, (uint)(t * dim), 0u }, gxDim);
                        for (int oi = 0; oi < m.MoeOpsCount; oi++)
                        {
                            var op = _layerOps[m.MoeOpsStart + oi];
                            _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
                        }
                        _vk.Dispatch(_kCopy, _bCopyGXToBX, new uint[] { (uint)dim, 0u, (uint)(t * dim) }, gxDim);
                    }
                }
                else
                {
                    DispatchQmm(m.GW!, m.BGSet, b);
                    DispatchQmm(m.UW!, m.BUSet, b);
                    if (m.GBias != null) DispatchBiasBatch(m.BGBiasSet, m.FfnDim, b);
                    if (m.UBias != null) DispatchBiasBatch(m.BUBiasSet, m.FfnDim, b);
                    _vk.Dispatch(_kSwi, m.BSwiFfnSet, new uint[] { (uint)(b * m.FfnDim) }, (uint)((b * m.FfnDim + 255) / 256));
                    DispatchQmm(m.DW!, m.BDSet, b);
                    _vk.Dispatch(_kAdd, _bAddProjSet, new uint[] { (uint)(b * dim) }, gxBDim);
                }

                // Flush periodically on deep models (e.g. 64-layer 27B) so a single vkQueueSubmit never trips AMDGPU's ~2s CS watchdog.
                if ((l & 7) == 7 && l + 1 < _batchMeta.Count)
                {
                    _vk.SubmitAndWait();
                    _vk.Begin();
                }
            }

            if (isLastChunk)
            {
                _vk.Dispatch(_kCopy, _bCopyBXToGX, new uint[] { (uint)dim, (uint)((b - 1) * dim), 0u }, gxDim);
                if (wantLogits)
                {
                    Run(_finalOps, tokens[^1], startPos + tokens.Length - 1);
                }
            }

            _vk.SubmitAndWait();
            offset += b;
        }

        if (wantLogits)
        {
            fixed (float* p = logits) _gLogits.DownloadTo(p, (long)_vocab * 4);
            fixed (float* pHid = _lastHidden) _gX.DownloadTo(pHid, (long)_dim * 4);
        }
    }

    public void SaveRecurrentState(Span<byte> destination)
    {
        long required = StateSnapshotBytes;
        if (destination.Length < required)
            throw new ArgumentException($"Destination span length {destination.Length} < required {required} bytes.", nameof(destination));
        if (required == 0) return;

        long offset = 0;
        fixed (byte* p = destination)
        {
            foreach (var (conv, rec) in _states)
            {
                conv.DownloadTo(p + offset, (long)conv.Size);
                offset += (long)conv.Size;
                rec.DownloadTo(p + offset, (long)rec.Size);
                offset += (long)rec.Size;
            }
        }
    }

    public void RestoreRecurrentState(ReadOnlySpan<byte> snapshot)
    {
        long required = StateSnapshotBytes;
        if (snapshot.Length < required)
            throw new ArgumentException($"Snapshot span length {snapshot.Length} < required {required} bytes.", nameof(snapshot));
        if (required == 0) return;

        long offset = 0;
        fixed (byte* p = snapshot)
        {
            foreach (var (conv, rec) in _states)
            {
                conv.UploadFrom(p + offset, (long)conv.Size);
                offset += (long)conv.Size;
                rec.UploadFrom(p + offset, (long)rec.Size);
                offset += (long)rec.Size;
            }
        }
    }

    public void VerifyBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<int> predictedTokens,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;
        if (predictedTokens.Length < tokens.Length)
            throw new ArgumentException($"predictedTokens length {predictedTokens.Length} < tokens length {tokens.Length}.", nameof(predictedTokens));

        float[] row = new float[_vocab];
        long snapBytes = perStepSnapshots != null ? StateSnapshotBytes : 0L;
        for (int t = 0; t < tokens.Length; t++)
        {
            Forward(tokens[t], startPos + t, row, computeLogits: true);
            int bestId = 0;
            float bestVal = row[0];
            for (int v = 1; v < _vocab; v++)
            {
                if (row[v] > bestVal)
                {
                    bestVal = row[v];
                    bestId = v;
                }
            }
            predictedTokens[t] = bestId;
            if (perStepSnapshots != null && t < perStepSnapshots.Length && perStepSnapshots[t] != null && snapBytes > 0)
            {
                SaveRecurrentState(perStepSnapshots[t]);
            }
        }
    }

    public void VerifyBatchLogits(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> flatLogits,
        int vocabStride,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;
        if (vocabStride < _vocab)
            throw new ArgumentException($"vocabStride ({vocabStride}) must be >= VocabSize ({_vocab}).", nameof(vocabStride));
        if (flatLogits.Length < (tokens.Length - 1) * vocabStride + _vocab)
            throw new ArgumentException("flatLogits buffer is too small.", nameof(flatLogits));

        long snapBytes = perStepSnapshots != null ? StateSnapshotBytes : 0L;
        for (int t = 0; t < tokens.Length; t++)
        {
            Forward(tokens[t], startPos + t, flatLogits.Slice(t * vocabStride, _vocab), computeLogits: true);
            if (perStepSnapshots != null && t < perStepSnapshots.Length && perStepSnapshots[t] != null && snapBytes > 0)
            {
                SaveRecurrentState(perStepSnapshots[t]);
            }
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
        Array.Clear(_lastHidden);
        if (_historyValid != null) Array.Clear(_historyValid);
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
