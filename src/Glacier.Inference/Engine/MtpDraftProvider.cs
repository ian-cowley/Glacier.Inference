namespace Glacier.Inference.Engine;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;

/// <summary>
/// Weight descriptor for an integrated Multi-Token Prediction (MTP / NextN) speculative draft head.
/// Supports both fused <c>eh_proj</c> and split <c>fc_emb</c>+<c>fc_hidden</c>, plus dense or MoE FFN blocks.
/// </summary>
public sealed unsafe class MtpWeights
{
    public int Dim { get; init; }
    public int FfnDim { get; init; }
    public int NumHeads { get; init; }
    public int NumHeadsKv { get; init; }
    public int HeadDim { get; init; }
    public int RopeDim { get; init; }
    public float RopeFreqBase { get; init; } = 10000.0f;
    public float RmsNormEps { get; init; } = 1e-6f;
    public int VocabSize { get; init; }

    public float* EnormWeight { get; init; }
    public float* HnormWeight { get; init; }

    public byte* EhProjWeight { get; init; }
    public GgufType EhProjType { get; init; }
    public byte* FcEmbWeight { get; init; }
    public GgufType FcEmbType { get; init; }
    public byte* FcHiddenWeight { get; init; }
    public GgufType FcHiddenType { get; init; }

    public required LayerWeights Layer { get; init; }

    public float* SharedHeadNormWeight { get; init; }
    public byte* SharedHeadWeight { get; init; }
    public GgufType SharedHeadType { get; init; }

    public int ExpertCount { get; init; }
    public int ExpertUsedCount { get; init; }
    public int ExpertFfnDim { get; init; }
    public int SharedExpertFfnDim { get; init; }
    public bool NormTopK { get; init; }

    // Qwen3.8-Flash-Next (qwen4exp): Multi-stream HC, QSA, and detached sidecar resources
    public bool HnormPerStream { get; init; }
    public int HcCount { get; init; }
    public int HcLowRank { get; init; }
    public float* HcHeadNormWeight { get; init; }
    public byte* HcHeadDownWeight { get; init; }
    public GgufType HcHeadDownType { get; init; }
    public byte* HcHeadUpWeight { get; init; }
    public GgufType HcHeadUpType { get; init; }
    public byte* SidecarTokenEmbdWeight { get; init; }
    public GgufType SidecarTokenEmbdType { get; init; }
    public int IndexerHeadCount { get; init; }
    public int IndexerKeyLength { get; init; }
    public int IndexerTopK { get; init; }
    internal GgufFile? OwnedSidecarGguf { get; init; }
    internal List<IntPtr>? OwnedFloatBuffers { get; init; }
}

/// <summary>
/// Native zero-VRAM-overhead Multi-Token Prediction (MTP) speculative draft provider.
/// Combines trunk hidden states with next token embeddings via <c>enorm</c>/<c>hnorm</c>/<c>eh_proj</c>,
/// executes one causal transformer block with a private KV cache, and projects through either the full
/// shared output head or an optional compact <see cref="DraftVocabularySubset"/>.
/// </summary>
public sealed unsafe class MtpDraftProvider : IDraftProvider
{
    private readonly IHiddenStateSource _trunk;
    private readonly MtpWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly int _groupSize;
    private readonly int _ropeDim;
    private readonly float _ropeFreqBase;
    private readonly float _rmsEps;
    private readonly float _attnScale;
    private readonly int _vocabSize;
    private readonly int _maxSeqLen;

    private readonly KVCache _mtpKvCache;
    private readonly PackedDraftOutputHead? _packedHead;
    private readonly List<int> _syncedTokens = new();
    private int _syncedPairs;

    // Qwen3.8-Flash-Next (qwen4exp): Multi-stream HC & QSA indexer state
    private readonly int _hcCount;
    private readonly HyperConnectionStack? _hcStack;
    private readonly QsaSparseAttentionIndexer? _qsaIndexer;
    private readonly float[]? _qsaTailBackup;
    private float* _hcResidual;
    private float* _hcInject;

    // Preallocated scratch buffers
    private float* _embBuf;
    private float* _embNorm;
    private float* _hidBuf;
    private float* _hidNorm;
    private float* _prevHiddenBuf;
    private float* _ehConcat;
    private float* _ehSums;
    private float* _tmpProj;
    private float* _mtpHidden;
    private float* _normX;
    private float* _normXSums;

    private float* _q;
    private float* _qFull;
    private float* _qGate;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProj;
    private float* _headScores;

    private float* _gate;
    private float* _up;
    private float* _gateUpFused;
    private float* _ffnAct;
    private float* _ffnActSums;
    private float* _ffnOut;
    private float* _expertDownOut;

    private float* _logits;
    private float[]? _fullProbsScratch;
    private bool _disposed;

    public PackedDraftOutputHead? PackedOutputHead => _packedHead;
    public DraftVocabularySubset? VocabularySubset => _packedHead?.Subset;
    public int SyncedPairs => _syncedPairs;

    public MtpDraftProvider(
        IHiddenStateSource trunk,
        MtpWeights weights,
        int maxSeqLen = 4096,
        DraftVocabularySubset? subset = null)
    {
        _trunk = trunk ?? throw new ArgumentNullException(nameof(trunk));
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (weights.Dim <= 0) throw new ArgumentOutOfRangeException(nameof(weights));
        if (weights.VocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(weights));
        if (weights.SharedHeadWeight == null) throw new ArgumentException("SharedHeadWeight pointer must not be null.", nameof(weights));

        _trunk.EnablePositionHistory();

        _dim = weights.Dim;
        _ffnDim = weights.FfnDim;
        _nHeads = Math.Max(1, weights.NumHeads);
        _nHeadsKv = Math.Max(1, weights.NumHeadsKv);
        _headDim = weights.HeadDim > 0 ? weights.HeadDim : (_dim / _nHeads);
        _groupSize = Math.Max(1, _nHeads / _nHeadsKv);
        _ropeDim = weights.RopeDim > 0 ? weights.RopeDim : _headDim;
        _ropeFreqBase = weights.RopeFreqBase > 0f ? weights.RopeFreqBase : 10000.0f;
        _rmsEps = weights.RmsNormEps > 0f ? weights.RmsNormEps : 1e-6f;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _vocabSize = weights.VocabSize;
        _maxSeqLen = maxSeqLen;

        bool useHc = weights.HcCount > 1 && weights.Layer.HasHyperConnections && weights.HcLowRank > 0;
        _hcCount = useHc ? weights.HcCount : 1;
        if (useHc)
        {
            _hcStack = new HyperConnectionStack(_hcCount, _dim, weights.HcLowRank, _rmsEps);
            _hcResidual = AllocFloats((long)_dim * _hcCount);
            _hcInject = AllocFloats(_hcCount);
        }

        if (weights.Layer.IsQsa)
        {
            int idxBlock = weights.Layer.IndexerCompressRatio > 0 ? weights.Layer.IndexerCompressRatio : 4;
            _qsaIndexer = new QsaSparseAttentionIndexer(
                _dim,
                Math.Max(1, weights.IndexerHeadCount),
                Math.Max(1, weights.IndexerKeyLength),
                idxBlock,
                Math.Max(1, weights.IndexerTopK),
                _ropeDim,
                _ropeFreqBase,
                _rmsEps,
                maxSeqLen);
            _qsaTailBackup = new float[_qsaIndexer.TailFloatCount];
        }

        _mtpKvCache = new KVCache(layers: 1, _nHeadsKv, _headDim, maxSeqLen);

        if (subset != null)
        {
            _packedHead = new PackedDraftOutputHead(
                weights.SharedHeadType,
                weights.SharedHeadWeight,
                _dim,
                _vocabSize,
                subset);
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int maxFfn = Math.Max(_ffnDim, Math.Max(weights.ExpertFfnDim, weights.SharedExpertFfnDim));
        if (maxFfn <= 0) maxFfn = _dim;

        _embBuf = AllocFloats(_dim);
        _embNorm = AllocFloats(_dim);
        _hidBuf = AllocFloats((long)_dim * _hcCount);
        _hidNorm = AllocFloats(_dim);
        _prevHiddenBuf = AllocFloats((long)_dim * _hcCount);
        _ehConcat = AllocFloats(2 * _dim);
        _ehSums = AllocFloats(Math.Max(1, ((2 * _dim) + 31) / 32));
        _tmpProj = AllocFloats(_dim);
        _mtpHidden = AllocFloats(_dim);
        _normX = AllocFloats(_dim);
        _normXSums = AllocFloats(Math.Max(1, (_dim + 31) / 32));

        _q = AllocFloats(qDim);
        _qFull = AllocFloats(2 * qDim);
        _qGate = AllocFloats(qDim);
        _k = AllocFloats(kvDim);
        _v = AllocFloats(kvDim);
        _attnOut = AllocFloats(qDim);
        _attnOutSums = AllocFloats(Math.Max(1, (qDim + 31) / 32));
        _attnProj = AllocFloats(_dim);
        _headScores = AllocFloats((long)_nHeads * _maxSeqLen);

        _gate = AllocFloats(maxFfn);
        _up = AllocFloats(maxFfn);
        _gateUpFused = AllocFloats(2 * maxFfn);
        _ffnAct = AllocFloats(maxFfn);
        _ffnActSums = AllocFloats(Math.Max(1, (maxFfn + 31) / 32));
        _ffnOut = AllocFloats(_dim);
        _expertDownOut = AllocFloats(_dim);

        _logits = subset == null ? AllocFloats(_vocabSize) : null;
    }

    private static float* AllocFloats(long count) =>
        (float*)NativeMemory.AlignedAlloc((nuint)(Math.Max(1L, count) * sizeof(float)), 64);

    /// <summary>
    /// Attempts to bind embedded NextN / MTP layer weights directly from a loaded <see cref="CpuModelBase"/>.
    /// Returns <c>null</c> when the GGUF model contains no MTP tensors.
    /// </summary>
    public static MtpDraftProvider? TryCreateFromModel(CpuModelBase trunkModel, DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunkModel);
        return TryCreateFromModel(trunkModel, trunkModel.Weights, trunkModel.MaxSeqLen, subset);
    }

    /// <summary>
    /// Attempts to bind embedded NextN / MTP layer weights directly from any <see cref="IHiddenStateSource"/>
    /// (such as a CPU or Direct3D 12 trunk runtime) and its loaded <see cref="ModelWeights"/>.
    /// Returns <c>null</c> when the GGUF model contains no MTP tensors.
    /// </summary>
    public static MtpDraftProvider? TryCreateFromModel(
        IHiddenStateSource trunk,
        ModelWeights weights,
        int maxSeqLen,
        DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunk);
        ArgumentNullException.ThrowIfNull(weights);

        return CreateFromGguf(trunk, weights, maxSeqLen, weights.Gguf, weights.EmbeddedMtpBlockIndex, isSidecar: false, ownedFloatBuffers: null, subset);
    }

    /// <summary>
    /// Loads a detached NextN / MTP speculative sidecar GGUF (such as <c>mtp-Qwen3.8-Flash-Next-*.gguf</c> or
    /// <c>micro-qwen4exp-realqsa-mtp-*.gguf</c>) without running full trunk <see cref="ModelWeights"/> validation.
    /// </summary>
    public static MtpDraftProvider LoadFromSidecar(
        CpuModelBase trunkModel,
        string sidecarGgufPath,
        DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunkModel);
        return LoadFromSidecar(trunkModel, trunkModel.Weights, trunkModel.MaxSeqLen, sidecarGgufPath, subset);
    }

    /// <summary>
    /// Loads a detached NextN / MTP speculative sidecar GGUF for any <see cref="IHiddenStateSource"/> trunk runtime
    /// (CPU or Direct3D 12 GPU).
    /// </summary>
    public static MtpDraftProvider LoadFromSidecar(
        IHiddenStateSource trunk,
        ModelWeights weights,
        int maxSeqLen,
        string sidecarGgufPath,
        DraftVocabularySubset? subset = null)
    {
        ArgumentNullException.ThrowIfNull(trunk);
        ArgumentNullException.ThrowIfNull(weights);
        if (string.IsNullOrWhiteSpace(sidecarGgufPath))
            throw new ArgumentException("MTP sidecar GGUF path must not be empty.", nameof(sidecarGgufPath));

        GgufFile sidecarGguf = GgufFile.Open(sidecarGgufPath);
        List<IntPtr> ownedFloatBuffers = new();
        try
        {
            int mtpIdx = FindMtpBlockIndex(sidecarGguf);
            MtpDraftProvider? provider = CreateFromGguf(
                trunk,
                weights,
                maxSeqLen,
                sidecarGguf,
                mtpIdx,
                isSidecar: true,
                ownedFloatBuffers,
                subset);
            if (provider == null)
            {
                throw new System.IO.InvalidDataException(
                    $"MTP sidecar '{sidecarGgufPath}' does not contain a valid NextN/MTP block.");
            }
            return provider;
        }
        catch
        {
            foreach (IntPtr ptr in ownedFloatBuffers)
            {
                if (ptr != IntPtr.Zero) NativeMemory.Free((void*)ptr);
            }
            sidecarGguf.Dispose();
            throw;
        }
    }

    private static int FindMtpBlockIndex(GgufFile gguf)
    {
        for (int i = Math.Max(0, gguf.BlockCount - 1); i >= 0; i--)
        {
            if (TryGetAnyTensor(gguf, out _, $"blk.{i}.nextn.eh_proj.weight", $"blk.{i}.nextn.fc_emb.weight"))
            {
                return i;
            }
        }

        foreach (GgufTensorInfo tensor in gguf.TensorList)
        {
            const string prefix = "blk.";
            if (!tensor.Name.StartsWith(prefix, StringComparison.Ordinal))
                continue;

            int dot = tensor.Name.IndexOf('.', prefix.Length);
            if (dot <= prefix.Length)
                continue;

            string suffix = tensor.Name.Substring(dot + 1);
            if ((suffix == "nextn.eh_proj.weight" || suffix == "nextn.fc_emb.weight") &&
                int.TryParse(tensor.Name.AsSpan(prefix.Length, dot - prefix.Length), out int idx))
            {
                return idx;
            }
        }

        return -1;
    }

    private static MtpDraftProvider? CreateFromGguf(
        IHiddenStateSource trunk,
        ModelWeights w,
        int maxSeqLen,
        GgufFile gguf,
        int mtpIdx,
        bool isSidecar,
        List<IntPtr>? ownedFloatBuffers,
        DraftVocabularySubset? subset)
    {
        float* LoadF32(GgufTensorInfo? tensor) =>
            tensor == null
                ? null
                : (ownedFloatBuffers != null
                    ? ModelWeights.DequantizeTensorToFloat32(gguf, tensor, ownedFloatBuffers)
                    : w.LoadFloat32Pointer(tensor));

        string blkPrefix = mtpIdx >= 0 ? $"blk.{mtpIdx}." : string.Empty;
        string nextnPrefix = blkPrefix + "nextn.";

        if (!TryGetAnyTensor(gguf, out var ehProjT, $"{nextnPrefix}eh_proj.weight", "mtp.eh_proj.weight") &&
            !TryGetAnyTensor(gguf, out _, $"{nextnPrefix}fc_emb.weight", "mtp.fc_emb.weight"))
        {
            return null;
        }

        TryGetAnyTensor(gguf, out var fcEmbT, $"{nextnPrefix}fc_emb.weight", "mtp.fc_emb.weight");
        TryGetAnyTensor(gguf, out var fcHidT, $"{nextnPrefix}fc_hidden.weight", "mtp.fc_hidden.weight");

        if (ehProjT == null && (fcEmbT == null || fcHidT == null))
        {
            return null;
        }

        TryGetAnyTensor(gguf, out var enormT, $"{nextnPrefix}enorm.weight", "mtp.enorm.weight");
        TryGetAnyTensor(gguf, out var hnormT, $"{nextnPrefix}hnorm.weight", "mtp.hnorm.weight");
        TryGetAnyTensor(gguf, out var shNormT, $"{nextnPrefix}shared_head_norm.weight", $"{nextnPrefix}shared_head.norm.weight", "mtp.shared_head_norm.weight");
        TryGetAnyTensor(gguf, out var shHeadT, $"{nextnPrefix}shared_head_head.weight", $"{nextnPrefix}shared_head.head.weight", $"{nextnPrefix}shared_head.weight", "mtp.shared_head_head.weight");

        string layerPrefix = mtpIdx >= 0 ? $"blk.{mtpIdx}." : "mtp.blk.0.";
        TryGetAnyTensor(gguf, out var attnNormT, $"{layerPrefix}attn_norm.weight", $"{layerPrefix}input_layernorm.weight");
        TryGetAnyTensor(gguf, out var qT, $"{layerPrefix}attn_q.weight");
        TryGetAnyTensor(gguf, out var qBiasT, $"{layerPrefix}attn_q.bias");
        TryGetAnyTensor(gguf, out var kT, $"{layerPrefix}attn_k.weight");
        TryGetAnyTensor(gguf, out var kBiasT, $"{layerPrefix}attn_k.bias");
        TryGetAnyTensor(gguf, out var vT, $"{layerPrefix}attn_v.weight");
        TryGetAnyTensor(gguf, out var vBiasT, $"{layerPrefix}attn_v.bias");
        TryGetAnyTensor(gguf, out var qkvT, $"{layerPrefix}attn_qkv.weight");
        TryGetAnyTensor(gguf, out var qkvBiasT, $"{layerPrefix}attn_qkv.bias");
        TryGetAnyTensor(gguf, out var outT, $"{layerPrefix}attn_output.weight");
        TryGetAnyTensor(gguf, out var outBiasT, $"{layerPrefix}attn_output.bias");
        TryGetAnyTensor(gguf, out var qNormT, $"{layerPrefix}attn_q_norm.weight");
        TryGetAnyTensor(gguf, out var kNormT, $"{layerPrefix}attn_k_norm.weight");

        if (outT == null || (qkvT == null && (qT == null || kT == null || vT == null)))
        {
            return null;
        }

        TryGetAnyTensor(gguf, out var ffnNormT, $"{layerPrefix}ffn_norm.weight", $"{layerPrefix}post_attention_norm.weight");
        TryGetAnyTensor(gguf, out var ffnGateT, $"{layerPrefix}ffn_gate.weight");
        TryGetAnyTensor(gguf, out var ffnUpT, $"{layerPrefix}ffn_up.weight");
        TryGetAnyTensor(gguf, out var ffnUpBiasT, $"{layerPrefix}ffn_up.bias");
        TryGetAnyTensor(gguf, out var ffnDownT, $"{layerPrefix}ffn_down.weight");

        TryGetAnyTensor(gguf, out var gateInpT, $"{layerPrefix}ffn_gate_inp.weight");
        TryGetAnyTensor(gguf, out var gateInpBiasT, $"{layerPrefix}ffn_gate_inp.bias", $"{layerPrefix}exp_probs_b.bias");
        TryGetAnyTensor(gguf, out var gateExpsT, $"{layerPrefix}ffn_gate_exps.weight");
        TryGetAnyTensor(gguf, out var upExpsT, $"{layerPrefix}ffn_up_exps.weight");
        TryGetAnyTensor(gguf, out var gateUpExpsT, $"{layerPrefix}ffn_gate_up_exps.weight");
        TryGetAnyTensor(gguf, out var downExpsT, $"{layerPrefix}ffn_down_exps.weight");
        TryGetAnyTensor(gguf, out var gateShexpT, $"{layerPrefix}ffn_gate_shexp.weight");
        TryGetAnyTensor(gguf, out var upShexpT, $"{layerPrefix}ffn_up_shexp.weight");
        TryGetAnyTensor(gguf, out var downShexpT, $"{layerPrefix}ffn_down_shexp.weight");
        TryGetAnyTensor(gguf, out var gateInpShexpT, $"{layerPrefix}ffn_gate_inp_shexp.weight");

        bool hasMoe = gateInpT != null && downExpsT != null && (gateUpExpsT != null || (gateExpsT != null && upExpsT != null));
        if (!hasMoe && (ffnUpT == null || ffnDownT == null))
        {
            return null;
        }

        // Qwen3.8-Flash-Next (qwen4exp): Hyper-Connections and QSA indexer inside MTP block
        TryGetAnyTensor(gguf, out var hcAttnNormT, $"{layerPrefix}hc_attn_norm.weight");
        TryGetAnyTensor(gguf, out var hcAttnDownT, $"{layerPrefix}hc_attn_down.weight");
        TryGetAnyTensor(gguf, out var hcAttnUpT, $"{layerPrefix}hc_attn_up.weight");
        TryGetAnyTensor(gguf, out var hcAttnInjT, $"{layerPrefix}hc_attn_inject.weight");
        TryGetAnyTensor(gguf, out var hcAttnDownInjT, $"{layerPrefix}hc_attn_down_inject.weight");
        TryGetAnyTensor(gguf, out var hcFfnNormT, $"{layerPrefix}hc_ffn_norm.weight");
        TryGetAnyTensor(gguf, out var hcFfnDownT, $"{layerPrefix}hc_ffn_down.weight");
        TryGetAnyTensor(gguf, out var hcFfnUpT, $"{layerPrefix}hc_ffn_up.weight");
        TryGetAnyTensor(gguf, out var hcFfnInjT, $"{layerPrefix}hc_ffn_inject.weight");
        TryGetAnyTensor(gguf, out var hcFfnDownInjT, $"{layerPrefix}hc_ffn_down_inject.weight");

        TryGetAnyTensor(gguf, out var idxQProjT, $"{layerPrefix}indexer.q_proj.weight");
        TryGetAnyTensor(gguf, out var idxKProjT, $"{layerPrefix}indexer.k_proj.weight");
        TryGetAnyTensor(gguf, out var idxQNormT, $"{layerPrefix}indexer.q_norm.weight");
        TryGetAnyTensor(gguf, out var idxKNormT, $"{layerPrefix}indexer.k_norm.weight");

        int hcCount = gguf.HyperConnectionCount;
        if (hcCount == 0 && hcAttnInjT != null && hcAttnInjT.Dimensions.Length >= 2)
        {
            hcCount = (int)hcAttnInjT.Dimensions[1];
        }
        if (hcCount == 0 && hcAttnNormT != null)
        {
            if (hcAttnNormT.Dimensions.Length >= 2)
            {
                hcCount = (int)hcAttnNormT.Dimensions[1];
            }
            else if (w.EmbeddingLength > 0 && hcAttnNormT.ElementCount > (ulong)w.EmbeddingLength && hcAttnNormT.ElementCount % (ulong)w.EmbeddingLength == 0)
            {
                hcCount = (int)(hcAttnNormT.ElementCount / (ulong)w.EmbeddingLength);
            }
        }
        if (hcCount == 0) hcCount = w.HyperConnectionCount;

        int hcLowRank = gguf.HyperConnectionLowRank;
        if (hcLowRank == 0 && hcAttnDownT != null && hcAttnDownT.Dimensions.Length >= 2)
        {
            hcLowRank = (int)hcAttnDownT.Dimensions[1];
        }
        if (hcLowRank == 0 && hcCount > 0 && hcAttnDownInjT != null && hcAttnDownInjT.Dimensions.Length >= 2)
        {
            int nLo = (int)hcAttnDownInjT.Dimensions[1] - hcCount;
            if (nLo > 0) hcLowRank = nLo;
        }
        if (hcLowRank == 0) hcLowRank = w.HyperConnectionLowRank;

        byte* hcAttnDownPtr = hcAttnDownT != null ? gguf.GetTensorPointer(hcAttnDownT) : null;
        GgufType hcAttnDownType = hcAttnDownT?.Type ?? GgufType.F32;
        byte* hcAttnInjPtr = hcAttnInjT != null ? gguf.GetTensorPointer(hcAttnInjT) : null;
        GgufType hcAttnInjType = hcAttnInjT?.Type ?? GgufType.F32;
        if (hcAttnDownPtr == null && hcAttnInjPtr == null && hcAttnDownInjT != null && hcAttnDownInjT.Dimensions.Length >= 2 && hcCount > 0)
        {
            int hcDim = (int)hcAttnDownInjT.Dimensions[0];
            int nLo = (int)hcAttnDownInjT.Dimensions[1] - hcCount;
            if (nLo > 0)
            {
                byte* basePtr = gguf.GetTensorPointer(hcAttnDownInjT);
                long rowBytes = GgufTypes.GetRowBytes(hcAttnDownInjT.Type, hcDim);
                hcAttnDownPtr = basePtr;
                hcAttnDownType = hcAttnDownInjT.Type;
                hcAttnInjPtr = basePtr + (long)nLo * rowBytes;
                hcAttnInjType = hcAttnDownInjT.Type;
            }
        }

        byte* hcFfnDownPtr = hcFfnDownT != null ? gguf.GetTensorPointer(hcFfnDownT) : null;
        GgufType hcFfnDownType = hcFfnDownT?.Type ?? GgufType.F32;
        byte* hcFfnInjPtr = hcFfnInjT != null ? gguf.GetTensorPointer(hcFfnInjT) : null;
        GgufType hcFfnInjType = hcFfnInjT?.Type ?? GgufType.F32;
        if (hcFfnDownPtr == null && hcFfnInjPtr == null && hcFfnDownInjT != null && hcFfnDownInjT.Dimensions.Length >= 2 && hcCount > 0)
        {
            int hcDim = (int)hcFfnDownInjT.Dimensions[0];
            int nLo = (int)hcFfnDownInjT.Dimensions[1] - hcCount;
            if (nLo > 0)
            {
                byte* basePtr = gguf.GetTensorPointer(hcFfnDownInjT);
                long rowBytes = GgufTypes.GetRowBytes(hcFfnDownInjT.Type, hcDim);
                hcFfnDownPtr = basePtr;
                hcFfnDownType = hcFfnDownInjT.Type;
                hcFfnInjPtr = basePtr + (long)nLo * rowBytes;
                hcFfnInjType = hcFfnDownInjT.Type;
            }
        }

        int headDim = qNormT != null && qNormT.Dimensions.Length > 0 ? (int)qNormT.Dimensions[0] : (gguf.HeadDim > 0 ? gguf.HeadDim : w.HeadDim);
        int numHeads = outT.Dimensions.Length >= 1 && headDim > 0
            ? Math.Max(1, (int)outT.Dimensions[0] / headDim)
            : (gguf.HeadCount > 0 ? gguf.HeadCount : w.HeadCount);
        int numHeadsKv = kT != null && kT.Dimensions.Length >= 2 && headDim > 0
            ? Math.Max(1, (int)kT.Dimensions[1] / headDim)
            : (gguf.HeadCountKvPattern is { Length: > 0 } patKv && mtpIdx >= 0 && mtpIdx < patKv.Length && patKv[mtpIdx] > 0
                ? patKv[mtpIdx]
                : (gguf.HeadCountKv > 0 ? gguf.HeadCountKv : w.HeadCountKv));

        byte* qkvPtr = qkvT != null ? gguf.GetTensorPointer(qkvT) : null;
        GgufType qkvType = qkvT?.Type ?? GgufType.F32;
        float* qkvBiasPtr = LoadF32(qkvBiasT);
        byte* qPtr = qT != null ? gguf.GetTensorPointer(qT) : null;
        GgufType qType = qT?.Type ?? GgufType.F32;
        float* qBiasPtr = LoadF32(qBiasT);
        byte* kPtr = kT != null ? gguf.GetTensorPointer(kT) : null;
        GgufType kType = kT?.Type ?? GgufType.F32;
        float* kBiasPtr = LoadF32(kBiasT);
        byte* vPtr = vT != null ? gguf.GetTensorPointer(vT) : null;
        GgufType vType = vT?.Type ?? GgufType.F32;
        float* vBiasPtr = LoadF32(vBiasT);

        bool hasQGate = qT != null && qT.Dimensions.Length >= 2 && (int)qT.Dimensions[1] == 2 * numHeads * headDim;
        if (qPtr == null && kPtr == null && vPtr == null && qkvT != null && qkvT.Dimensions.Length >= 2)
        {
            int qDim = numHeads * headDim;
            int kvDim = numHeadsKv * headDim;
            if (qkvT.Dimensions[1] == (ulong)(2 * qDim + 2 * kvDim))
            {
                int inDim = (int)qkvT.Dimensions[0];
                int qRows = 2 * qDim;
                byte* basePtr = gguf.GetTensorPointer(qkvT);
                long rowBytes = GgufTypes.GetRowBytes(qkvT.Type, inDim);
                qPtr = basePtr;
                qType = qkvT.Type;
                hasQGate = true;
                kPtr = basePtr + (long)qRows * rowBytes;
                kType = qkvT.Type;
                vPtr = basePtr + (long)(qRows + kvDim) * rowBytes;
                vType = qkvT.Type;
                qkvPtr = null;
                if (qkvBiasPtr != null)
                {
                    qBiasPtr = qkvBiasPtr;
                    kBiasPtr = qkvBiasPtr + qRows;
                    vBiasPtr = qkvBiasPtr + qRows + kvDim;
                    qkvBiasPtr = null;
                }
            }
        }

        int ffnDim = ffnDownT != null && ffnDownT.Dimensions.Length > 0 ? (int)ffnDownT.Dimensions[0] : w.FeedForwardLength;
        int expFfnDim = downExpsT != null && downExpsT.Dimensions.Length > 0 ? (int)downExpsT.Dimensions[0] : w.ExpertFeedForwardLength;
        int expCount = downExpsT != null && downExpsT.Dimensions.Length >= 3 ? (int)downExpsT.Dimensions[2] : (gguf.ExpertCount > 0 ? gguf.ExpertCount : w.ExpertCount);
        int sharedFfnDim = (int)gguf.GetMetadataUInt32($"{gguf.Architecture}.expert_shared_feed_forward_length",
            gguf.GetMetadataUInt32("expert_shared_feed_forward_length", 0));
        if (sharedFfnDim <= 0 && gateShexpT != null && gateShexpT.Dimensions.Length >= 2)
            sharedFfnDim = (int)gateShexpT.Dimensions[1];
        if (sharedFfnDim <= 0 && upShexpT != null && upShexpT.Dimensions.Length >= 2)
            sharedFfnDim = (int)upShexpT.Dimensions[1];
        if (sharedFfnDim <= 0)
            sharedFfnDim = w.SharedExpertFeedForwardLength;

        int[]? compressRatios = gguf.AttentionCompressRatios ?? w.Gguf.AttentionCompressRatios;
        int idxCompressRatio = compressRatios != null && mtpIdx >= 0 && mtpIdx < compressRatios.Length && compressRatios[mtpIdx] > 0
            ? compressRatios[mtpIdx]
            : 0;
        if (idxCompressRatio == 0 && compressRatios != null)
        {
            for (int r = 0; r < compressRatios.Length; r++)
            {
                if (compressRatios[r] > 0) { idxCompressRatio = compressRatios[r]; break; }
            }
        }
        if (idxCompressRatio == 0 && idxQProjT != null && idxKProjT != null) idxCompressRatio = 4;

        bool hasHcLayer = hcAttnNormT != null && hcAttnDownPtr != null && hcAttnUpT != null;

        var layer = new LayerWeights
        {
            AttnNormWeight = LoadF32(attnNormT),
            AttnNormType = attnNormT?.Type ?? GgufType.F32,
            QkvWeight = qkvPtr,
            QkvType = qkvType,
            QkvBias = qkvBiasPtr,
            QWeight = qPtr,
            QType = qType,
            QBias = qBiasPtr,
            KWeight = kPtr,
            KType = kType,
            KBias = kBiasPtr,
            VWeight = vPtr,
            VType = vType,
            VBias = vBiasPtr,
            AttnOutWeight = gguf.GetTensorPointer(outT),
            AttnOutType = outT.Type,
            AttnOutBias = LoadF32(outBiasT),
            AttnQNormWeight = LoadF32(qNormT),
            AttnKNormWeight = LoadF32(kNormT),
            HasQGate = hasQGate,
            FfnNormWeight = LoadF32(ffnNormT),
            FfnNormType = ffnNormT?.Type ?? GgufType.F32,
            IsMoe = hasMoe,
            FfnGateWeight = ffnGateT != null ? gguf.GetTensorPointer(ffnGateT) : null,
            FfnGateType = ffnGateT?.Type ?? GgufType.F32,
            FfnUpWeight = ffnUpT != null ? gguf.GetTensorPointer(ffnUpT) : null,
            FfnUpType = ffnUpT?.Type ?? GgufType.F32,
            FfnUpBias = LoadF32(ffnUpBiasT),
            FfnDownWeight = ffnDownT != null ? gguf.GetTensorPointer(ffnDownT) : null,
            FfnDownType = ffnDownT?.Type ?? GgufType.F32,
            FfnGateInpWeight = LoadF32(gateInpT),
            FfnGateInpBias = LoadF32(gateInpBiasT),
            FfnGateExpsWeight = gateExpsT != null ? gguf.GetTensorPointer(gateExpsT) : null,
            FfnGateExpsType = gateExpsT?.Type ?? GgufType.F32,
            FfnUpExpsWeight = upExpsT != null ? gguf.GetTensorPointer(upExpsT) : null,
            FfnUpExpsType = upExpsT?.Type ?? GgufType.F32,
            FfnGateUpExpsWeight = gateUpExpsT != null ? gguf.GetTensorPointer(gateUpExpsT) : null,
            FfnGateUpExpsType = gateUpExpsT?.Type ?? GgufType.F32,
            FfnDownExpsWeight = downExpsT != null ? gguf.GetTensorPointer(downExpsT) : null,
            FfnDownExpsType = downExpsT?.Type ?? GgufType.F32,
            FfnGateInpShexpWeight = LoadF32(gateInpShexpT),
            FfnGateShexpWeight = gateShexpT != null ? gguf.GetTensorPointer(gateShexpT) : null,
            FfnGateShexpType = gateShexpT?.Type ?? GgufType.F32,
            FfnUpShexpWeight = upShexpT != null ? gguf.GetTensorPointer(upShexpT) : null,
            FfnUpShexpType = upShexpT?.Type ?? GgufType.F32,
            FfnDownShexpWeight = downShexpT != null ? gguf.GetTensorPointer(downShexpT) : null,
            FfnDownShexpType = downShexpT?.Type ?? GgufType.F32,
            HasHyperConnections = hasHcLayer,
            HcAttnNormWeight = LoadF32(hcAttnNormT),
            HcAttnDownWeight = hcAttnDownPtr,
            HcAttnDownType = hcAttnDownType,
            HcAttnUpWeight = hcAttnUpT != null ? gguf.GetTensorPointer(hcAttnUpT) : null,
            HcAttnUpType = hcAttnUpT?.Type ?? GgufType.F32,
            HcAttnInjectWeight = hcAttnInjPtr,
            HcAttnInjectType = hcAttnInjType,
            HcFfnNormWeight = LoadF32(hcFfnNormT),
            HcFfnDownWeight = hcFfnDownPtr,
            HcFfnDownType = hcFfnDownType,
            HcFfnUpWeight = hcFfnUpT != null ? gguf.GetTensorPointer(hcFfnUpT) : null,
            HcFfnUpType = hcFfnUpT?.Type ?? GgufType.F32,
            HcFfnInjectWeight = hcFfnInjPtr,
            HcFfnInjectType = hcFfnInjType,
            IsQsa = idxQProjT != null && idxKProjT != null,
            IndexerQProjWeight = idxQProjT != null ? gguf.GetTensorPointer(idxQProjT) : null,
            IndexerQProjType = idxQProjT?.Type ?? GgufType.F32,
            IndexerKProjWeight = idxKProjT != null ? gguf.GetTensorPointer(idxKProjT) : null,
            IndexerKProjType = idxKProjT?.Type ?? GgufType.F32,
            IndexerQNormWeight = LoadF32(idxQNormT),
            IndexerKNormWeight = LoadF32(idxKNormT),
            IndexerCompressRatio = idxCompressRatio
        };

        // Resolve output HC collapse weights (MTP-specific first, then sidecar top-level, then trunk)
        TryGetAnyTensor(gguf, out var hcHeadNormT,
            $"{nextnPrefix}hc_head_norm.weight", $"{nextnPrefix}output_hc_norm.weight",
            "output_hc_norm.weight", "hc_head_norm.weight");
        TryGetAnyTensor(gguf, out var hcHeadDownT,
            $"{nextnPrefix}hc_head_down.weight", $"{nextnPrefix}output_hc_down.weight",
            "output_hc_down.weight", "hc_head_down.weight");
        TryGetAnyTensor(gguf, out var hcHeadUpT,
            $"{nextnPrefix}hc_head_up.weight", $"{nextnPrefix}output_hc_up.weight",
            "output_hc_up.weight", "hc_head_up.weight");

        GgufTensorInfo? sideEmbdT = null;
        if (isSidecar)
        {
            gguf.TryGetTensor("token_embd.weight", out sideEmbdT);
        }

        byte* sharedHeadPtr = null;
        GgufType sharedHeadType = GgufType.F32;
        if (shHeadT != null)
        {
            sharedHeadPtr = gguf.GetTensorPointer(shHeadT);
            sharedHeadType = shHeadT.Type;
        }
        else if (isSidecar && gguf.TryGetTensor("output.weight", out var sideOutT) && sideOutT != null)
        {
            sharedHeadPtr = gguf.GetTensorPointer(sideOutT);
            sharedHeadType = sideOutT.Type;
        }
        else if (isSidecar && sideEmbdT != null)
        {
            sharedHeadPtr = gguf.GetTensorPointer(sideEmbdT);
            sharedHeadType = sideEmbdT.Type;
        }
        else
        {
            sharedHeadPtr = w.OutWeight;
            sharedHeadType = w.OutType;
        }
        if (sharedHeadPtr == null) return null;

        float* sharedHeadNormPtr = null;
        if (shNormT != null)
        {
            sharedHeadNormPtr = LoadF32(shNormT);
        }
        else if (isSidecar && gguf.TryGetTensor("output_norm.weight", out var sideOutNormT) && sideOutNormT != null)
        {
            sharedHeadNormPtr = LoadF32(sideOutNormT);
        }
        else if (!hasHcLayer)
        {
            sharedHeadNormPtr = w.OutNormWeight;
        }

        int expertUsed = (int)gguf.GetMetadataUInt32($"{gguf.Architecture}.expert_used_count", (uint)w.ExpertUsedCount);
        bool normTopK = gguf.Metadata.ContainsKey($"{gguf.Architecture}.expert_weights_norm") ? gguf.NormTopK : w.NormTopK;
        bool hnormPerStream = hnormT != null && hcCount > 1 && hnormT.ElementCount >= (ulong)(w.EmbeddingLength * hcCount);

        var mtpWeights = new MtpWeights
        {
            Dim = w.EmbeddingLength,
            FfnDim = ffnDim > 0 ? ffnDim : (expFfnDim > 0 ? expFfnDim : sharedFfnDim),
            NumHeads = numHeads,
            NumHeadsKv = numHeadsKv,
            HeadDim = headDim,
            RopeDim = gguf.RopeDimensionCount > 0 ? gguf.RopeDimensionCount : (w.RopeDimensionCount > 0 ? w.RopeDimensionCount : headDim),
            RopeFreqBase = gguf.Metadata.ContainsKey($"{gguf.Architecture}.rope.freq_base") ? gguf.RopeFreqBase : w.RopeFreqBase,
            RmsNormEps = gguf.Metadata.ContainsKey($"{gguf.Architecture}.attention.layer_norm_rms_epsilon") ? gguf.RmsNormEps : w.RmsNormEps,
            VocabSize = w.VocabSize,
            EnormWeight = LoadF32(enormT),
            HnormWeight = LoadF32(hnormT),
            HnormPerStream = hnormPerStream,
            EhProjWeight = ehProjT != null ? gguf.GetTensorPointer(ehProjT) : null,
            EhProjType = ehProjT?.Type ?? GgufType.F32,
            FcEmbWeight = fcEmbT != null ? gguf.GetTensorPointer(fcEmbT) : null,
            FcEmbType = fcEmbT?.Type ?? GgufType.F32,
            FcHiddenWeight = fcHidT != null ? gguf.GetTensorPointer(fcHidT) : null,
            FcHiddenType = fcHidT?.Type ?? GgufType.F32,
            Layer = layer,
            SharedHeadNormWeight = sharedHeadNormPtr,
            SharedHeadWeight = sharedHeadPtr,
            SharedHeadType = sharedHeadType,
            ExpertCount = expCount,
            ExpertUsedCount = Math.Max(1, expertUsed),
            ExpertFfnDim = expFfnDim,
            SharedExpertFfnDim = sharedFfnDim,
            NormTopK = normTopK,
            HcCount = hcCount,
            HcLowRank = hcLowRank,
            HcHeadNormWeight = hcHeadNormT != null ? LoadF32(hcHeadNormT) : (hasHcLayer ? w.OutputHcNormWeight : null),
            HcHeadDownWeight = hcHeadDownT != null ? gguf.GetTensorPointer(hcHeadDownT) : (hasHcLayer ? w.OutputHcDownWeight : null),
            HcHeadDownType = hcHeadDownT?.Type ?? (hasHcLayer ? w.OutputHcDownType : GgufType.F32),
            HcHeadUpWeight = hcHeadUpT != null ? gguf.GetTensorPointer(hcHeadUpT) : (hasHcLayer ? w.OutputHcUpWeight : null),
            HcHeadUpType = hcHeadUpT?.Type ?? (hasHcLayer ? w.OutputHcUpType : GgufType.F32),
            SidecarTokenEmbdWeight = sideEmbdT != null ? gguf.GetTensorPointer(sideEmbdT) : null,
            SidecarTokenEmbdType = sideEmbdT?.Type ?? GgufType.F32,
            IndexerHeadCount = gguf.IndexerHeadCount > 0 ? gguf.IndexerHeadCount : w.IndexerHeadCount,
            IndexerKeyLength = gguf.IndexerKeyLength > 0 ? gguf.IndexerKeyLength : w.IndexerKeyLength,
            IndexerTopK = gguf.IndexerTopK > 0 ? gguf.IndexerTopK : w.IndexerTopK,
            OwnedSidecarGguf = isSidecar ? gguf : null,
            OwnedFloatBuffers = ownedFloatBuffers
        };

        return new MtpDraftProvider(trunk, mtpWeights, maxSeqLen, subset);
    }

    private static bool TryGetAnyTensor(GgufFile gguf, out GgufTensorInfo? info, params string[] names)
    {
        for (int i = 0; i < names.Length; i++)
        {
            if (gguf.TryGetTensor(names[i], out info) && info != null)
            {
                return true;
            }
        }
        info = null;
        return false;
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _mtpKvCache.Reset();
        _qsaIndexer?.Reset();
        _syncedTokens.Clear();
        _syncedPairs = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void CopyTokenEmbeddingForDraft(int tokenId, Span<float> destination)
    {
        if (_weights.SidecarTokenEmbdWeight != null)
        {
            fixed (float* pDst = destination)
            {
                QuantKernels.ExtractEmbedding(_weights.SidecarTokenEmbdType, _weights.SidecarTokenEmbdWeight, tokenId, pDst, _dim);
            }
            return;
        }
        _trunk.CopyTokenEmbedding(tokenId, destination);
    }

    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens) =>
        DraftCore(tokens, ReadOnlySpan<int>.Empty, maxDraftTokens, sampling: null, sampler: null, targetVocabSize: 0, draftTokens, Span<float>.Empty);

    public int DraftWithProbabilities(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        SamplingOptions sampling,
        Sampler sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs) =>
        DraftCore(tokens, recentTokens, maxDraftTokens, sampling, sampler, targetVocabSize, draftTokens, draftProbs);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int DraftCore(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        SamplingOptions? sampling,
        Sampler? sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.Length < 2 || maxDraftTokens <= 0 || draftTokens.IsEmpty)
            return 0;

        int pairPos = tokens.Length - 2;
        int available = _maxSeqLen - pairPos;
        int count = Math.Min(maxDraftTokens, Math.Min(draftTokens.Length, available));
        if (count <= 0) return 0;

        bool withProbs = sampling != null && sampler != null && targetVocabSize > 0;
        if (withProbs && draftProbs.Length < count * targetVocabSize)
            throw new ArgumentException($"draftProbs buffer too small ({draftProbs.Length} < {count * targetVocabSize}).", nameof(draftProbs));

        SyncPrefix(tokens, pairPos);

        if (_qsaIndexer != null && _qsaTailBackup != null)
        {
            _qsaIndexer.SaveTail(_qsaTailBackup);
        }

        try
        {
            // Step d = 0: fuse trunk hidden state at pairPos with committed next token tokens[pairPos + 1]
            LoadHiddenStateForPair(pairPos);
            CopyTokenEmbeddingForDraft(tokens[pairPos + 1], new Span<float>(_embBuf, _dim));
            ForwardMtpStep(pairPos, _embBuf, _hidBuf);

            int tok0 = SampleHead(withProbs, sampling, sampler, recentTokens, targetVocabSize, draftProbs);
            draftTokens[0] = tok0;

            // Autoregressive multi-step roll-forward at positions pairPos + 1 .. pairPos + count - 1
            float* stepSrc = _hcStack != null ? _hcResidual : _mtpHidden;
            ulong hiddenBytes = (ulong)(_dim * _hcCount) * sizeof(float);
            for (int d = 1; d < count; d++)
            {
                Buffer.MemoryCopy(stepSrc, _prevHiddenBuf, hiddenBytes, hiddenBytes);
                CopyTokenEmbeddingForDraft(draftTokens[d - 1], new Span<float>(_embBuf, _dim));
                ForwardMtpStep(pairPos + d, _embBuf, _prevHiddenBuf);

                Span<float> rowProbs = withProbs ? draftProbs.Slice(d * targetVocabSize, targetVocabSize) : Span<float>.Empty;
                draftTokens[d] = SampleHead(withProbs, sampling, sampler, recentTokens, targetVocabSize, rowProbs);
            }

            return count;
        }
        finally
        {
            if (_qsaIndexer != null && _qsaTailBackup != null)
            {
                _qsaIndexer.RestoreTail(_qsaTailBackup);
            }
        }
    }

    private void SyncPrefix(ReadOnlySpan<int> tokens, int pairPos)
    {
        if (pairPos < _syncedPairs)
        {
            Reset();
        }
        else if (_syncedPairs > 0)
        {
            int checkLen = Math.Min(_syncedPairs + 1, _syncedTokens.Count);
            if (tokens.Length < checkLen)
            {
                Reset();
            }
            else
            {
                for (int i = 0; i < checkLen; i++)
                {
                    if (_syncedTokens[i] != tokens[i])
                    {
                        Reset();
                        break;
                    }
                }
            }
        }

        for (int p = _syncedPairs; p < pairPos; p++)
        {
            LoadHiddenStateForPair(p);
            CopyTokenEmbeddingForDraft(tokens[p + 1], new Span<float>(_embBuf, _dim));
            ForwardMtpStep(p, _embBuf, _hidBuf);
            RecordSyncedPair(tokens, p);
        }
    }

    private void RecordSyncedPair(ReadOnlySpan<int> tokens, int p)
    {
        while (_syncedTokens.Count <= p + 1)
        {
            _syncedTokens.Add(tokens[_syncedTokens.Count]);
        }
        _syncedPairs = p + 1;
    }

    private void LoadHiddenStateForPair(int pos)
    {
        if (_hcStack != null)
        {
            int multiDim = _dim * _hcCount;
            var multiSpan = new Span<float>(_hidBuf, multiDim);
            if (_trunk.TryGetMultiStreamHiddenStateAt(pos, multiSpan))
            {
                return;
            }

            ReadOnlySpan<float> latestMulti = _trunk.GetLastMultiStreamHiddenState();
            if (latestMulti.Length >= multiDim)
            {
                latestMulti.Slice(0, multiDim).CopyTo(multiSpan);
                return;
            }

            ReadOnlySpan<float> fallbackSingle = latestMulti.Length >= _dim ? latestMulti.Slice(0, _dim) : _trunk.GetLastHiddenState().Slice(0, _dim);
            for (int s = 0; s < _hcCount; s++)
            {
                fallbackSingle.CopyTo(multiSpan.Slice(s * _dim, _dim));
            }
            return;
        }

        var hidSpan = new Span<float>(_hidBuf, _dim);
        if (_trunk.TryGetHiddenStateAt(pos, hidSpan))
        {
            return;
        }

        ReadOnlySpan<float> latest = _trunk.GetLastHiddenState();
        latest.Slice(0, _dim).CopyTo(hidSpan);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardMtpStep(int pos, float* embPtr, float* prevHiddenPtr)
    {
        LayerWeights layer = _weights.Layer;
        QuantKernels.RMSNorm(embPtr, _weights.EnormWeight, _embNorm, _dim, _rmsEps);

        if (_hcStack != null)
        {
            ulong halfBytes = (ulong)_dim * sizeof(float);
            Buffer.MemoryCopy(_embNorm, _ehConcat, halfBytes, halfBytes);
            for (int s = 0; s < _hcCount; s++)
            {
                float* srcStream = prevHiddenPtr + (long)s * _dim;
                float* hGamma = _weights.HnormWeight != null
                    ? (_weights.HnormPerStream ? _weights.HnormWeight + (long)s * _dim : _weights.HnormWeight)
                    : null;
                QuantKernels.RMSNorm(srcStream, hGamma, _hidNorm, _dim, _rmsEps);

                float* dstStream = _hcResidual + (long)s * _dim;
                if (_weights.EhProjWeight != null)
                {
                    Buffer.MemoryCopy(_hidNorm, _ehConcat + _dim, halfBytes, halfBytes);
                    float* ehSums = null;
                    if (2 * _dim >= 32)
                    {
                        QuantKernels.ComputeBlockSums32(_ehConcat, _ehSums, 2 * _dim);
                        ehSums = _ehSums;
                    }
                    QuantKernels.MatVecMul(_weights.EhProjType, _weights.EhProjWeight, _ehConcat, dstStream, 2 * _dim, _dim, ehSums);
                }
                else if (_weights.FcEmbWeight != null && _weights.FcHiddenWeight != null)
                {
                    ComputeSumsIfAligned(_embNorm);
                    QuantKernels.MatVecMul(_weights.FcEmbType, _weights.FcEmbWeight, _embNorm, dstStream, _dim, _dim, _dim >= 32 ? _normXSums : null);
                    ComputeSumsIfAligned(_hidNorm);
                    QuantKernels.MatVecMul(_weights.FcHiddenType, _weights.FcHiddenWeight, _hidNorm, _tmpProj, _dim, _dim, _dim >= 32 ? _normXSums : null);
                    QuantKernels.VecAdd(_tmpProj, dstStream, _dim);
                }
                else
                {
                    for (int d = 0; d < _dim; d++)
                    {
                        dstStream[d] = 0.5f * (_embNorm[d] + _hidNorm[d]);
                    }
                }
            }

            _hcStack.Read(
                _hcResidual,
                layer.HcAttnNormWeight,
                layer.HcAttnDownType,
                layer.HcAttnDownWeight,
                layer.HcAttnUpType,
                layer.HcAttnUpWeight,
                layer.HcAttnInjectType,
                layer.HcAttnInjectWeight,
                _normX,
                _hcInject);
        }
        else
        {
            QuantKernels.RMSNorm(prevHiddenPtr, _weights.HnormWeight, _hidNorm, _dim, _rmsEps);

            if (_weights.EhProjWeight != null)
            {
                ulong halfBytes = (ulong)_dim * sizeof(float);
                Buffer.MemoryCopy(_embNorm, _ehConcat, halfBytes, halfBytes);
                Buffer.MemoryCopy(_hidNorm, _ehConcat + _dim, halfBytes, halfBytes);
                float* ehSums = null;
                if (2 * _dim >= 32)
                {
                    QuantKernels.ComputeBlockSums32(_ehConcat, _ehSums, 2 * _dim);
                    ehSums = _ehSums;
                }
                QuantKernels.MatVecMul(_weights.EhProjType, _weights.EhProjWeight, _ehConcat, _mtpHidden, 2 * _dim, _dim, ehSums);
            }
            else if (_weights.FcEmbWeight != null && _weights.FcHiddenWeight != null)
            {
                ComputeSumsIfAligned(_embNorm);
                QuantKernels.MatVecMul(_weights.FcEmbType, _weights.FcEmbWeight, _embNorm, _mtpHidden, _dim, _dim, _dim >= 32 ? _normXSums : null);
                ComputeSumsIfAligned(_hidNorm);
                QuantKernels.MatVecMul(_weights.FcHiddenType, _weights.FcHiddenWeight, _hidNorm, _tmpProj, _dim, _dim, _dim >= 32 ? _normXSums : null);
                QuantKernels.VecAdd(_tmpProj, _mtpHidden, _dim);
            }
            else
            {
                for (int d = 0; d < _dim; d++)
                {
                    _mtpHidden[d] = 0.5f * (_embNorm[d] + _hidNorm[d]);
                }
            }

            QuantKernels.RMSNorm(_mtpHidden, layer.AttnNormWeight, _normX, _dim, _rmsEps);
        }

        // 1. Attention sub-block
        float* normSums = ComputeSumsIfAligned(_normX);

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        if (layer.HasFusedQkv)
        {
            int qkvDim = qDim + 2 * kvDim;
            QuantKernels.MatVecMul(layer.QkvType, layer.QkvWeight, _normX, _qFull, _dim, qkvDim, normSums);
            if (layer.QkvBias != null) QuantKernels.VecAdd(layer.QkvBias, _qFull, qkvDim);
            Buffer.MemoryCopy(_qFull, _q, (ulong)(qDim * sizeof(float)), (ulong)(qDim * sizeof(float)));
            Buffer.MemoryCopy(_qFull + qDim, _k, (ulong)(kvDim * sizeof(float)), (ulong)(kvDim * sizeof(float)));
            Buffer.MemoryCopy(_qFull + qDim + kvDim, _v, (ulong)(kvDim * sizeof(float)), (ulong)(kvDim * sizeof(float)));
        }
        else
        {
            if (layer.HasQGate)
            {
                QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _qFull, _dim, 2 * qDim, normSums);
                if (layer.QBias != null) QuantKernels.VecAdd(layer.QBias, _qFull, 2 * qDim);
                for (int h = 0; h < _nHeads; h++)
                {
                    float* srcHead = _qFull + h * (2 * _headDim);
                    Buffer.MemoryCopy(srcHead, _q + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
                    Buffer.MemoryCopy(srcHead + _headDim, _qGate + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
                }
            }
            else
            {
                QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, normSums);
                if (layer.QBias != null) QuantKernels.VecAdd(layer.QBias, _q, qDim);
            }

            QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, normSums);
            if (layer.KBias != null) QuantKernels.VecAdd(layer.KBias, _k, kvDim);

            QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, normSums);
            if (layer.VBias != null) QuantKernels.VecAdd(layer.VBias, _v, kvDim);
        }

        if (layer.AttnQNormWeight != null)
        {
            for (int h = 0; h < _nHeads; h++)
            {
                float* qHead = _q + h * _headDim;
                QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _rmsEps);
            }
        }
        if (layer.AttnKNormWeight != null)
        {
            for (int h = 0; h < _nHeadsKv; h++)
            {
                float* kHead = _k + h * _headDim;
                QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _rmsEps);
            }
        }

        QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _ropeFreqBase, ropeDim: _ropeDim);
        _mtpKvCache.Store(0, pos, _k, _v);

        int* selectedIds = null;
        int kvCount;
        if (_qsaIndexer != null && layer.IsQsa)
        {
            var qsaWeights = new QsaIndexerWeights
            {
                QProjType = layer.IndexerQProjType,
                QProjWeight = layer.IndexerQProjWeight,
                KProjType = layer.IndexerKProjType,
                KProjWeight = layer.IndexerKProjWeight,
                QNormWeight = layer.IndexerQNormWeight,
                KNormWeight = layer.IndexerKNormWeight
            };
            _qsaIndexer.RecordKeyAndSelect(pos, _normX, in qsaWeights, out selectedIds, out kvCount);
        }
        else
        {
            kvCount = pos + 1;
        }

        if (kvCount <= 0)
        {
            new Span<float>(_attnOut, qDim).Clear();
        }
        else
        {
            for (int h = 0; h < _nHeads; h++)
            {
                int hKv = h / _groupSize;
                float* qHead = _q + h * _headDim;
                float* scores = _headScores + (long)h * _maxSeqLen;

                for (int idx = 0; idx < kvCount; idx++)
                {
                    int t = selectedIds != null ? selectedIds[idx] : idx;
                    scores[idx] = _mtpKvCache.DotKey(0, hKv, t, qHead) * _attnScale;
                }

                QuantKernels.Softmax(scores, kvCount);

                float* outHead = _attnOut + h * _headDim;
                for (int d = 0; d < _headDim; d++) outHead[d] = 0f;

                for (int idx = 0; idx < kvCount; idx++)
                {
                    int t = selectedIds != null ? selectedIds[idx] : idx;
                    _mtpKvCache.AccumulateValue(0, hKv, t, scores[idx], outHead);
                }
            }
        }

        if (layer.HasQGate)
        {
            for (int j = 0; j < qDim; j++)
            {
                _attnOut[j] *= 1.0f / (1.0f + MathF.Exp(-_qGate[j]));
            }
        }

        float* attnSums = null;
        if (qDim >= 32)
        {
            QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
            attnSums = _attnOutSums;
        }
        QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, attnSums);
        if (layer.AttnOutBias != null) QuantKernels.VecAdd(layer.AttnOutBias, _attnProj, _dim);

        if (_hcStack != null)
        {
            _hcStack.Write(_hcResidual, _attnProj, _hcInject);

            _hcStack.Read(
                _hcResidual,
                layer.HcFfnNormWeight,
                layer.HcFfnDownType,
                layer.HcFfnDownWeight,
                layer.HcFfnUpType,
                layer.HcFfnUpWeight,
                layer.HcFfnInjectType,
                layer.HcFfnInjectWeight,
                _normX,
                _hcInject);
            normSums = ComputeSumsIfAligned(_normX);
            ExecuteLayerFfn(layer, normSums);
            _hcStack.Write(_hcResidual, _ffnOut, _hcInject);

            if (_weights.HcHeadDownWeight != null)
            {
                _hcStack.Read(
                    _hcResidual,
                    _weights.HcHeadNormWeight,
                    _weights.HcHeadDownType,
                    _weights.HcHeadDownWeight,
                    _weights.HcHeadUpType,
                    _weights.HcHeadUpWeight,
                    GgufType.F32,
                    null,
                    _mtpHidden,
                    null);
            }
            else
            {
                Buffer.MemoryCopy(_hcResidual, _mtpHidden, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }
        else
        {
            QuantKernels.VecAdd(_attnProj, _mtpHidden, _dim);

            // 2. FFN / MoE sub-block
            QuantKernels.RMSNorm(_mtpHidden, layer.FfnNormWeight, _normX, _dim, _rmsEps);
            normSums = ComputeSumsIfAligned(_normX);
            ExecuteLayerFfn(layer, normSums);
            QuantKernels.VecAdd(_ffnOut, _mtpHidden, _dim);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ExecuteLayerFfn(LayerWeights layer, float* normSums)
    {
        if (layer.IsMoe)
        {
            int expertFfnDim = _weights.ExpertFfnDim > 0 ? _weights.ExpertFfnDim : _ffnDim;
            int numExperts = _weights.ExpertCount;
            int topK = Math.Max(1, _weights.ExpertUsedCount);
            int* selectedIndices = stackalloc int[topK];
            float* selectedWeights = stackalloc float[topK];

            QuantKernels.RouterTopK(_normX, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);
            new Span<float>(_ffnOut, _dim).Clear();

            bool fusedExps = layer.FfnGateUpExpsWeight != null && layer.FfnGateExpsWeight == null;
            long fusedSliceBytes = fusedExps ? (long)(2 * expertFfnDim) * GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, _dim) : 0;
            long gateSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
            long upSliceBytes = fusedExps ? 0 : (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
            long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

            for (int k = 0; k < topK; k++)
            {
                int expertIdx = selectedIndices[k];
                float weight = selectedWeights[k];

                if (fusedExps)
                {
                    byte* expGateUp = layer.FfnGateUpExpsWeight + expertIdx * fusedSliceBytes;
                    QuantKernels.MatVecMul(layer.FfnGateUpExpsType, expGateUp, _normX, _gateUpFused, _dim, 2 * expertFfnDim, normSums);
                    QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + expertFfnDim, _ffnAct, expertFfnDim);
                }
                else
                {
                    QuantKernels.MatVecMul(layer.FfnGateExpsType, layer.FfnGateExpsWeight + expertIdx * gateSliceBytes, _normX, _gate, _dim, expertFfnDim, normSums);
                    QuantKernels.MatVecMul(layer.FfnUpExpsType, layer.FfnUpExpsWeight + expertIdx * upSliceBytes, _normX, _up, _dim, expertFfnDim, normSums);
                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
                }

                float* actSums = null;
                if (expertFfnDim >= 32)
                {
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);
                    actSums = _ffnActSums;
                }
                QuantKernels.MatVecMul(layer.FfnDownExpsType, layer.FfnDownExpsWeight + expertIdx * downSliceBytes, _ffnAct, _expertDownOut, expertFfnDim, _dim, actSums);
                QuantKernels.VecAddWeighted(_expertDownOut, _ffnOut, weight, _dim);
            }

            if (layer.FfnGateShexpWeight != null)
            {
                int sharedFfnDim = _weights.SharedExpertFfnDim > 0 ? _weights.SharedExpertFfnDim : expertFfnDim;
                QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, _normX, _gate, _dim, sharedFfnDim, normSums);
                QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, _normX, _up, _dim, sharedFfnDim, normSums);
                QuantKernels.SwiGLU(_gate, _up, _ffnAct, sharedFfnDim);
                float* actSums = null;
                if (sharedFfnDim >= 32)
                {
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, sharedFfnDim);
                    actSums = _ffnActSums;
                }
                QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, sharedFfnDim, _dim, actSums);
                if (layer.FfnGateInpShexpWeight != null)
                {
                    float g = QuantKernels.VecDotF32(_normX, layer.FfnGateInpShexpWeight, _dim);
                    float gate = 1.0f / (1.0f + MathF.Exp(-g));
                    QuantKernels.VecAddWeighted(_expertDownOut, _ffnOut, gate, _dim);
                }
                else
                {
                    QuantKernels.VecAdd(_expertDownOut, _ffnOut, _dim);
                }
            }
            return;
        }

        if (layer.HasFusedGateUp)
        {
            int gateUpDim = 2 * _ffnDim;
            QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _gateUpFused, _dim, gateUpDim, normSums);
            if (layer.FfnUpBias != null) QuantKernels.VecAdd(layer.FfnUpBias, _gateUpFused, gateUpDim);
            QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + _ffnDim, _ffnAct, _ffnDim);
        }
        else
        {
            QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim, normSums);
            QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim, normSums);
            QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
        }

        float* ffnSums = null;
        if (_ffnDim >= 32)
        {
            QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, _ffnDim);
            ffnSums = _ffnActSums;
        }
        QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, _ffnOut, _ffnDim, _dim, ffnSums);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float* ComputeSumsIfAligned(float* x)
    {
        if (_dim < 32) return null;
        QuantKernels.ComputeBlockSums32(x, _normXSums, _dim);
        return _normXSums;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int SampleHead(
        bool withProbs,
        SamplingOptions? sampling,
        Sampler? sampler,
        ReadOnlySpan<int> recentTokens,
        int targetVocabSize,
        Span<float> draftProbsRow)
    {
        if (_weights.SharedHeadNormWeight != null)
        {
            QuantKernels.RMSNorm(_mtpHidden, _weights.SharedHeadNormWeight, _normX, _dim, _rmsEps);
        }
        else
        {
            Buffer.MemoryCopy(_mtpHidden, _normX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
        }

        if (_packedHead != null)
        {
            if (!withProbs)
            {
                return _packedHead.SampleGreedy(_normX);
            }

            if (targetVocabSize != _vocabSize)
            {
                draftProbsRow.Clear();
                if (_fullProbsScratch == null || _fullProbsScratch.Length < _vocabSize)
                {
                    _fullProbsScratch = new float[_vocabSize];
                }
                int tok = _packedHead.SampleWithProbabilities(_normX, sampling!, sampler!, recentTokens, _fullProbsScratch);
                int copyLen = Math.Min(targetVocabSize, _vocabSize);
                _fullProbsScratch.AsSpan(0, copyLen).CopyTo(draftProbsRow);
                return tok;
            }

            return _packedHead.SampleWithProbabilities(_normX, sampling!, sampler!, recentTokens, draftProbsRow);
        }

        float* normSums = ComputeSumsIfAligned(_normX);
        QuantKernels.MatVecMul(_weights.SharedHeadType, _weights.SharedHeadWeight, _normX, _logits, _dim, _vocabSize, normSums);

        if (!withProbs)
        {
            int bestId = 0;
            float bestLogit = float.NegativeInfinity;
            for (int i = 0; i < _vocabSize; i++)
            {
                float v = _logits[i];
                if (v > bestLogit)
                {
                    bestLogit = v;
                    bestId = i;
                }
            }
            return bestId;
        }

        if (_fullProbsScratch == null || _fullProbsScratch.Length < _vocabSize)
        {
            _fullProbsScratch = new float[_vocabSize];
        }
        Sampler.ComputeFilteredDistribution(new ReadOnlySpan<float>(_logits, _vocabSize), sampling!, recentTokens, _fullProbsScratch);
        draftProbsRow.Clear();
        int len = Math.Min(targetVocabSize, _vocabSize);
        _fullProbsScratch.AsSpan(0, len).CopyTo(draftProbsRow);
        return Sampler.SampleFromDistribution(_fullProbsScratch.AsSpan(0, _vocabSize), sampler!.Rng);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _packedHead?.Dispose();
        _mtpKvCache.Dispose();
        _hcStack?.Dispose();
        _qsaIndexer?.Dispose();
        FreeAligned(ref _hcResidual);
        FreeAligned(ref _hcInject);

        if (_weights.OwnedFloatBuffers != null)
        {
            foreach (IntPtr ptr in _weights.OwnedFloatBuffers)
            {
                if (ptr != IntPtr.Zero) NativeMemory.Free((void*)ptr);
            }
            _weights.OwnedFloatBuffers.Clear();
        }
        _weights.OwnedSidecarGguf?.Dispose();

        FreeAligned(ref _embBuf);
        FreeAligned(ref _embNorm);
        FreeAligned(ref _hidBuf);
        FreeAligned(ref _hidNorm);
        FreeAligned(ref _prevHiddenBuf);
        FreeAligned(ref _ehConcat);
        FreeAligned(ref _ehSums);
        FreeAligned(ref _tmpProj);
        FreeAligned(ref _mtpHidden);
        FreeAligned(ref _normX);
        FreeAligned(ref _normXSums);

        FreeAligned(ref _q);
        FreeAligned(ref _qFull);
        FreeAligned(ref _qGate);
        FreeAligned(ref _k);
        FreeAligned(ref _v);
        FreeAligned(ref _attnOut);
        FreeAligned(ref _attnOutSums);
        FreeAligned(ref _attnProj);
        FreeAligned(ref _headScores);

        FreeAligned(ref _gate);
        FreeAligned(ref _up);
        FreeAligned(ref _gateUpFused);
        FreeAligned(ref _ffnAct);
        FreeAligned(ref _ffnActSums);
        FreeAligned(ref _ffnOut);
        FreeAligned(ref _expertDownOut);

        FreeAligned(ref _logits);
    }

    private static void FreeAligned(ref float* ptr)
    {
        if (ptr != null)
        {
            NativeMemory.AlignedFree(ptr);
            ptr = null;
        }
    }
}
