namespace Glacier.Inference.Model;

using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

/// <summary>
/// Dedicated high-performance CPU inference runtime for Qwen 3.5 / 3.6 Hybrid Models.
/// Implements Gated DeltaNet (linear attention SSM) with recurrent associative memory
/// interleaved with full quadratic multi-head attention (e.g. 3:1 ratio via full_attention_interval).
/// </summary>
public sealed unsafe class Qwen3HybridModel : CpuModelBase
{
    private readonly SsmStateCache _ssmCache;
    private readonly int _ssmConvKernel;
    private readonly int _ssmStateDim;
    private readonly int _ssmGroupCount;
    private readonly int _ssmHeads;
    private readonly int _ssmInnerSize;
    private readonly int _gdnConvChannels;
    private readonly int _groupSize;
    private readonly float _attnScale;

    // Preallocated GDN scratch buffers (single token)
    private float* _gdnQkv;
    private float* _gdnConvOut;
    private float* _gdnZ;
    private float* _gdnA;
    private float* _gdnB;
    private float* _gdnY;
    private float* _gdnYSums;

    // Preallocated standard attention scratch buffers for interleaved full-attention layers
    private float* _q;
    private float* _qFull;
    private float* _qGate;
    private float* _k;
    private float* _v;
    private float* _attnOut;
    private float* _attnOutSums;
    private float* _attnProj;
    private float* _headScores;

    // Batched standard attention scratch buffers
    private float* _qBatch;
    private float* _qFullBatch;
    private float* _qGateBatch;
    private float* _kBatch;
    private float* _vBatch;
    private float* _attnOutBatch;
    private float* _attnOutSumBatch;
    private float* _attnProjBatch;

    // Batched GDN scratch buffers
    private float* _gdnQkvBatch;
    private float* _gdnZBatch;
    private float* _gdnABatch;
    private float* _gdnBBatch;
    private float* _gdnYBatch;
    private float* _gdnYSumBatch;

    // Qwen3.8-Flash-Next (qwen4exp): Hyper-Connections, QSA Sparse Attention, and PLE NVMe Direct-I/O
    private readonly bool _useSigmoidGdnGate;
    private readonly HyperConnectionStack? _hc;
    private readonly int _hcCount;
    private float* _hcInject;
    private float* _lastHcHidden;
    private float* _hcHiddenHistory;
    private readonly PleLayerProcessor? _pleProcessor;
    private readonly int[]? _pleSlotMap;
    private readonly DirectIoNgramTable? _pleTable;
    private readonly bool _ownsPleTable;
    private readonly int _pleEmbDim;
    private float* _pleEmbBuf;
    private float* _pleEmbBatch;
    private readonly QsaSparseAttentionIndexer?[]? _qsaIndexers;
    private readonly int[]? _qsaSnapshotOffsets;

    // Per-step recurrent state capture during batched speculative verification
    private byte[][]? _activePerStepSnapshots;
    private int _activeSnapshotOffset;
    private int _activeSnapshotLimit;

    public override bool SupportsPerStepVerificationSnapshots => true;

    public SsmStateCache SsmCache => _ssmCache;
    public HyperConnectionStack? HyperConnections => _hc;
    public PleLayerProcessor? PleProcessor => _pleProcessor;
    public DirectIoNgramTable? PleTable => _pleTable;

    public override int MultiStreamHiddenDim => _hc != null ? _dim * _hcCount : _dim;

    public override void EnablePositionHistory()
    {
        base.EnablePositionHistory();
        if (_hc != null && _hcHiddenHistory == null)
        {
            long totalFloats = Math.Max(1L, (long)_maxSeqLen * _dim * _hcCount);
            _hcHiddenHistory = (float*)NativeMemory.AllocZeroed((nuint)(totalFloats * sizeof(float)));
        }
    }

    public override ReadOnlySpan<float> GetLastMultiStreamHiddenState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hc != null && _lastHcHidden != null)
        {
            return new ReadOnlySpan<float>(_lastHcHidden, _dim * _hcCount);
        }
        return base.GetLastMultiStreamHiddenState();
    }

    public override bool TryGetMultiStreamHiddenStateAt(int position, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hc != null)
        {
            int multiDim = _dim * _hcCount;
            if (_hcHiddenHistory == null || (uint)position >= (uint)_maxSeqLen || destination.Length < multiDim)
                return false;

            new ReadOnlySpan<float>(_hcHiddenHistory + (long)position * multiDim, multiDim).CopyTo(destination);
            return true;
        }
        return base.TryGetMultiStreamHiddenStateAt(position, destination);
    }

    public Qwen3HybridModel(
        ModelWeights weights,
        int maxSeqLen = 4096,
        int startLayer = 0,
        int layerCount = -1,
        bool isLastStage = true,
        DirectIoNgramTable? pleTable = null,
        PleIoMode pleIoMode = PleIoMode.DirectIo)
        : base(weights, maxSeqLen, startLayer, layerCount, isLastStage)
    {
        _ssmConvKernel = weights.Gguf.SsmConvKernel;
        _ssmStateDim = weights.Gguf.SsmStateSize;
        _ssmGroupCount = weights.Gguf.SsmGroupCount;
        _ssmHeads = weights.Gguf.SsmTimeStepRank;
        _ssmInnerSize = weights.Gguf.SsmInnerSize;
        _gdnConvChannels = (_ssmGroupCount * 2 + _ssmHeads) * _ssmStateDim;

        _groupSize = _nHeadsKv > 0 ? _nHeads / _nHeadsKv : 1;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);

        // 1. Allocate unmanaged recurrent state cache for Gated DeltaNet
        _ssmCache = new SsmStateCache(
            weights.BlockCount,
            _ssmConvKernel,
            _gdnConvChannels,
            _ssmHeads,
            _ssmStateDim);

        // 2. Allocate GDN scratch buffers
        _gdnQkv = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnConvOut = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnZ = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        _gdnA = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnB = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnY = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        int yChunks = (_ssmInnerSize + 31) / 32;
        _gdnYSums = (float*)NativeMemory.AllocZeroed((nuint)(yChunks * sizeof(float)));

        _gdnQkvBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _gdnConvChannels * sizeof(float)));
        _gdnZBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _ssmInnerSize * sizeof(float)));
        _gdnABatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _ssmHeads * sizeof(float)));
        _gdnBBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _ssmHeads * sizeof(float)));
        _gdnYBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _ssmInnerSize * sizeof(float)));
        _gdnYSumBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * yChunks * sizeof(float)));

        // 3. Allocate standard attention buffers for interleaved full-attention layers
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int maxAttnOut = Math.Max(qDim, _nHeads * _vHeadDim);
        int attnOutChunks = (Math.Max(maxAttnOut, _dim) + 31) / 32;

        _q = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _qFull = (float*)NativeMemory.AllocZeroed((nuint)(2 * qDim * sizeof(float)));
        _qGate = (float*)NativeMemory.AllocZeroed((nuint)(qDim * sizeof(float)));
        _k = (float*)NativeMemory.AllocZeroed((nuint)(kvDim * sizeof(float)));
        _v = (float*)NativeMemory.AllocZeroed((nuint)(_nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(maxAttnOut * sizeof(float)));
        _attnOutSums = (float*)NativeMemory.AllocZeroed((nuint)(attnOutChunks * sizeof(float)));
        _attnProj = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));

        long scoreBufferSize = (long)_nHeads * _maxSeqLen * sizeof(float);
        _headScores = (float*)NativeMemory.AllocZeroed((nuint)scoreBufferSize);

        _qBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qDim * sizeof(float)));
        _qFullBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * 2 * qDim * sizeof(float)));
        _qGateBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * qDim * sizeof(float)));
        _kBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * kvDim * sizeof(float)));
        _vBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _nHeadsKv * _vHeadDim * sizeof(float)));
        _attnOutBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * maxAttnOut * sizeof(float)));
        _attnOutSumBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * attnOutChunks * sizeof(float)));
        _attnProjBatch = (float*)NativeMemory.AllocZeroed((nuint)(MaxBatchSize * _dim * sizeof(float)));

        // 4. Initialize Qwen3.8-Flash-Next (qwen4exp) subsystems if present
        _useSigmoidGdnGate = string.Equals(weights.Gguf.Architecture, "qwen4exp", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(weights.Gguf.Architecture, "qwen38", StringComparison.OrdinalIgnoreCase) ||
            weights.HyperConnectionCount > 0;
        if (weights.HyperConnectionCount > 0 && weights.HyperConnectionLowRank > 0)
        {
            _hcCount = weights.HyperConnectionCount;
            _hc = new HyperConnectionStack(
                _hcCount,
                _dim,
                weights.HyperConnectionLowRank,
                weights.RmsNormEps,
                MaxBatchSize);
            _hcInject = (float*)NativeMemory.AllocZeroed((nuint)(_hcCount * sizeof(float)));
            _lastHcHidden = (float*)NativeMemory.AllocZeroed((nuint)(_dim * _hcCount * sizeof(float)));
        }

        int pleLayerCount = 0;
        for (int i = 0; i < weights.BlockCount; i++)
        {
            if (weights.Layers[i] != null && weights.Layers[i].IsGdn && weights.Layers[i].AttnGateWeight != null && weights.Layers[i].SsmConv1dWeight == null)
            {
                throw new InvalidDataException($"Missing SsmConv1dWeight for GDN layer {i}.");
            }
            if (weights.Layers[i] != null && weights.Layers[i].HasPle)
            {
                pleLayerCount++;
            }
            if (weights.Layers[i] != null && weights.Layers[i].IsQsa)
            {
                _qsaIndexers ??= new QsaSparseAttentionIndexer?[weights.BlockCount];
                int idxBlock = weights.Layers[i].IndexerCompressRatio > 0 ? weights.Layers[i].IndexerCompressRatio : 4;
                _qsaIndexers[i] = new QsaSparseAttentionIndexer(
                    _dim,
                    weights.IndexerHeadCount,
                    weights.IndexerKeyLength,
                    idxBlock,
                    weights.IndexerTopK,
                    weights.RopeDimensionCount,
                    weights.RopeFreqBase,
                    weights.RmsNormEps,
                    maxSeqLen);
            }
        }

        if (pleLayerCount > 0 && weights.PleHashConfig != null)
        {
            _pleSlotMap = new int[weights.BlockCount];
            Array.Fill(_pleSlotMap, -1);
            int slot = 0;
            for (int i = 0; i < weights.BlockCount; i++)
            {
                if (weights.Layers[i] != null && weights.Layers[i].HasPle)
                {
                    _pleSlotMap[i] = slot++;
                }
            }

            _pleProcessor = new PleLayerProcessor(
                pleLayerCount,
                Math.Max(1, weights.HyperConnectionCount),
                _dim,
                weights.PleHashConfig,
                weights.RmsNormEps);
            _pleEmbDim = weights.PleHashConfig.EmbeddingDim;
            _pleEmbBuf = (float*)NativeMemory.AllocZeroed((nuint)(_pleEmbDim * sizeof(float)));
            _pleEmbBatch = (float*)NativeMemory.AllocZeroed((nuint)((long)MaxBatchSize * _pleEmbDim * sizeof(float)));

            if (pleTable != null)
            {
                _pleTable = pleTable;
                _ownsPleTable = false;
            }
            else if (weights.Gguf.PleTableFilePath != null &&
                     weights.Gguf.TryGetTensor("per_layer_token_embd.weight", out var pleTensor) &&
                     pleTensor != null)
            {
                if (pleIoMode == PleIoMode.DirectIo)
                {
                    weights.Gguf.DetachExclusivePleShardMapping();
                }

                long totalRows = pleTensor.Dimensions.Length >= 2 ? (long)pleTensor.Dimensions[1] : weights.PleHashConfig.TotalRows;
                int rowDim = pleTensor.Dimensions.Length >= 1 ? (int)pleTensor.Dimensions[0] : weights.PleHashConfig.HeadDim;
                _pleTable = new DirectIoNgramTable(
                    weights.Gguf.PleTableFilePath,
                    weights.Gguf.PleTableByteOffset,
                    totalRows,
                    rowDim,
                    DirectIoNgramTable.FromGgufType(pleTensor.Type),
                    pleIoMode,
                    hashConfig: weights.PleHashConfig);
                _ownsPleTable = true;
            }
        }

        if (_qsaIndexers != null)
        {
            _qsaSnapshotOffsets = new int[_qsaIndexers.Length];
            int qsaOffset = checked((int)_ssmCache.TotalBytes) +
                (_pleProcessor != null ? checked((int)_pleProcessor.StateSnapshotBytes) : 0);
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                if (_qsaIndexers[i] != null)
                {
                    _qsaSnapshotOffsets[i] = qsaOffset;
                    qsaOffset += checked((int)_qsaIndexers[i]!.StateSnapshotBytes);
                }
            }
        }
    }

    public override void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null)
    {
        if (_hc == null && _pleProcessor == null)
        {
            base.ForwardStage(token, pos, inputX, outputX, logits, computeLogits, kvCache);
            return;
        }

        ObjectDisposedException.ThrowIf(_disposed, this);

        // Issue async PLE SSD read BEFORE token embedding lookup + early layers so I/O overlaps compute
        PleLookupTicket pleTicket = default;
        if (StartLayer == 0 && _pleTable != null && _pleProcessor != null)
        {
            pleTicket = _pleTable.IssueToken(token, _pleProcessor.PreviousTokens);
        }

        if (StartLayer == 0)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _x, _dim);
        }
        else
        {
            fixed (float* pInput = inputX)
            {
                Buffer.MemoryCopy(pInput, _x, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }

        float* rPtr = _x;
        if (_hc != null)
        {
            _hc.InitTokenFromEmbedding(_x);
            rPtr = _hc.GetResidualTokenPtr(0);
        }

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        try
        {
            for (int l = 0; l < LayerCount; l++)
            {
                int modelLayer = StartLayer + l;
                var layer = _weights.Layers[modelLayer];

                if (layer.HasPle && _pleProcessor != null)
                {
                    if (pleTicket.IsValid && _pleTable != null)
                    {
                        _pleTable.Collect(pleTicket, new Span<float>(_pleEmbBuf, _pleEmbDim));
                        pleTicket = default;
                    }
                    else
                    {
                        new Span<float>(_pleEmbBuf, _pleEmbDim).Clear();
                    }

                    var pleWeights = new PleLayerWeights
                    {
                        PleKeyType = layer.PleKeyType,
                        PleKeyWeight = layer.PleKeyWeight,
                        PleValueType = layer.PleValueType,
                        PleValueWeight = layer.PleValueWeight,
                        PleNormKey = layer.PleNormKeyWeight,
                        PleNormQuery = layer.PleNormQueryWeight,
                        PleNormConv = layer.PleNormConvWeight,
                        PleConv1dWeight = layer.PleConv1dWeight
                    };
                    _pleProcessor.ExecuteToken(_pleSlotMap![modelLayer], rPtr, _pleEmbBuf, pleWeights);
                }

                // 1. Attention / SSM sub-block
                if (_hc != null && layer.HasHyperConnections)
                {
                    _hc.Read(
                        rPtr,
                        layer.HcAttnNormWeight,
                        layer.HcAttnDownType,
                        layer.HcAttnDownWeight,
                        layer.HcAttnUpType,
                        layer.HcAttnUpWeight,
                        layer.HcAttnInjectType,
                        layer.HcAttnInjectWeight,
                        _normX,
                        _hcInject);
                    QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                    if (layer.IsGdn)
                    {
                        ForwardGdnLayerWithNorm(l, modelLayer, _x, _normX, _normXSums, addResidual: false);
                    }
                    else
                    {
                        ForwardStandardAttentionLayer(l, modelLayer, pos, _x, kvCache, addResidual: false);
                    }

                    _hc.Write(rPtr, _attnProj, _hcInject);
                }
                else
                {
                    QuantKernels.RMSNorm(rPtr, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);
                    QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);
                    if (layer.IsGdn)
                    {
                        ForwardGdnLayerWithNorm(l, modelLayer, rPtr, _normX, _normXSums, addResidual: true);
                    }
                    else
                    {
                        ForwardStandardAttentionLayer(l, modelLayer, pos, rPtr, kvCache, addResidual: true);
                    }
                }

                // 2. FFN / MoE sub-block
                if (_hc != null && layer.HasHyperConnections)
                {
                    _hc.Read(
                        rPtr,
                        layer.HcFfnNormWeight,
                        layer.HcFfnDownType,
                        layer.HcFfnDownWeight,
                        layer.HcFfnUpType,
                        layer.HcFfnUpWeight,
                        layer.HcFfnInjectType,
                        layer.HcFfnInjectWeight,
                        _normX,
                        _hcInject);
                    QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                    ExecuteSingleTokenFfn(modelLayer, _normX, _normXSums, _ffnOut, selectedIndices, selectedWeights);
                    _hc.Write(rPtr, _ffnOut, _hcInject);
                }
                else
                {
                    QuantKernels.RMSNorm(rPtr, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);
                    QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);
                    ExecuteSingleTokenFfn(modelLayer, _normX, _normXSums, _ffnOut, selectedIndices, selectedWeights);
                    AddVector(rPtr, _ffnOut, _dim);
                }
            }
        }
        finally
        {
            if (pleTicket.IsValid && _pleTable != null)
            {
                _pleTable.Collect(pleTicket, new Span<float>(_pleEmbBuf, _pleEmbDim));
            }
        }

        if (StartLayer == 0 && _pleProcessor != null)
        {
            _pleProcessor.PushToken(token);
        }

        if (IsLastStage && _hc != null && _lastHcHidden != null)
        {
            ulong hcBytes = (ulong)_dim * (ulong)_hcCount * sizeof(float);
            Buffer.MemoryCopy(rPtr, _lastHcHidden, hcBytes, hcBytes);
            if (_hcHiddenHistory != null && (uint)pos < (uint)_maxSeqLen)
            {
                Buffer.MemoryCopy(rPtr, _hcHiddenHistory + (long)pos * (_dim * _hcCount), hcBytes, hcBytes);
            }
        }

        // Collapse multi-stream Hyper-Connection stack into _x before output norm & LM head
        if (_hc != null && _weights.OutputHcDownWeight != null)
        {
            _hc.Read(
                rPtr,
                _weights.OutputHcNormWeight,
                _weights.OutputHcDownType,
                _weights.OutputHcDownWeight,
                _weights.OutputHcUpType,
                _weights.OutputHcUpWeight,
                GgufType.F32,
                null,
                _x,
                null);
        }
        else if (_hc != null)
        {
            ulong bytes = (ulong)_dim * sizeof(float);
            Buffer.MemoryCopy(rPtr, _x, bytes, bytes);
        }

        if (IsLastStage)
        {
            RecordHiddenStateForPosition(pos, _x);
            if (computeLogits && logits.Length > 0)
            {
                if (_weights.OutNormWeight != null)
                {
                    QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                }
                else
                {
                    Buffer.MemoryCopy(_x, _normX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                fixed (float* pLogits = logits)
                {
                    QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, pLogits, _dim, _weights.VocabSize, _normXSums);
                }
            }
            else if (outputX.Length >= _dim)
            {
                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
            }
        }
        else if (outputX.Length >= _dim)
        {
            fixed (float* pOut = outputX)
            {
                Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }
    }

    public override void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null)
    {
        if (_hc == null && _pleProcessor == null)
        {
            base.ForwardBatchStage(tokens, startPos, inputXBatch, outputXBatch, logits, computeLogits, kvCache);
            return;
        }

        int totalTokens = !tokens.IsEmpty ? tokens.Length : (inputXBatch.Length / _dim);
        int offset = 0;
        while (offset < totalTokens)
        {
            int batchSize = Math.Min(totalTokens - offset, MaxBatchSize);
            bool isLastChunk = (offset + batchSize == totalTokens);
            var inSlice = !inputXBatch.IsEmpty ? inputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var outSlice = !outputXBatch.IsEmpty ? outputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var tokSlice = !tokens.IsEmpty ? tokens.Slice(offset, batchSize) : default;

            ForwardBatchChunkUnified(
                tokSlice,
                startPos + offset,
                inSlice,
                outSlice,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                Span<float>.Empty,
                0,
                kvCache,
                computeLastLogits: isLastChunk && computeLogits,
                computeAllLogits: false,
                allowTailSkip: false);
            offset += batchSize;
        }
    }

    public override void VerifyBatch(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<int> greedyTokens,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;

        int offset = 0;
        int vocabSize = _weights.VocabSize;
        int snapshotLimit = perStepSnapshots != null ? Math.Min(perStepSnapshots.Length, Math.Max(0, tokens.Length - 1)) : 0;
        _activePerStepSnapshots = perStepSnapshots;
        _activeSnapshotLimit = snapshotLimit;
        try
        {
            while (offset < tokens.Length)
            {
                int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
                _activeSnapshotOffset = offset;
                var tokSlice = tokens.Slice(offset, batchSize);
                var greedySlice = greedyTokens.Slice(offset, batchSize);

                if (_hc == null && _pleProcessor == null)
                {
                    if (LayerCount < _weights.BlockCount && perStepSnapshots != null)
                    {
                        for (int t = 0; t < batchSize; t++)
                        {
                            int snapIdx = offset + t;
                            if ((uint)snapIdx < (uint)snapshotLimit)
                            {
                                SaveRecurrentState(perStepSnapshots[snapIdx]);
                            }
                        }
                    }

                    base.VerifyBatch(tokSlice, startPos + offset, greedySlice, kvCache, null);
                }
                else
                {
                    float* logitsBuf = EnsureVerifyLogitsBuffer(batchSize);
                    ForwardBatchChunkUnified(
                        tokSlice,
                        startPos + offset,
                        default,
                        default,
                        Span<float>.Empty,
                        new Span<float>(logitsBuf, batchSize * vocabSize),
                        vocabSize,
                        kvCache,
                        computeLastLogits: false,
                        computeAllLogits: true,
                        allowTailSkip: true);

                    for (int t = 0; t < batchSize; t++)
                    {
                        greedySlice[t] = ComputeGreedyArgmax(logitsBuf + t * vocabSize, vocabSize);
                    }
                }

                offset += batchSize;
            }
        }
        finally
        {
            _activePerStepSnapshots = null;
            _activeSnapshotOffset = 0;
            _activeSnapshotLimit = 0;
        }
    }

    public override void VerifyBatchLogits(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> logitsOut,
        int vocabStride,
        KVCache? kvCache = null,
        byte[][]? perStepSnapshots = null)
    {
        if (tokens.IsEmpty) return;

        int offset = 0;
        int snapshotLimit = perStepSnapshots != null ? Math.Min(perStepSnapshots.Length, Math.Max(0, tokens.Length - 1)) : 0;
        _activePerStepSnapshots = perStepSnapshots;
        _activeSnapshotLimit = snapshotLimit;
        try
        {
            while (offset < tokens.Length)
            {
                int batchSize = Math.Min(tokens.Length - offset, MaxBatchSize);
                _activeSnapshotOffset = offset;
                var tokSlice = tokens.Slice(offset, batchSize);
                var logitSlice = logitsOut.Slice(offset * vocabStride);

                if (_hc == null && _pleProcessor == null)
                {
                    if (LayerCount < _weights.BlockCount && perStepSnapshots != null)
                    {
                        for (int t = 0; t < batchSize; t++)
                        {
                            int snapIdx = offset + t;
                            if ((uint)snapIdx < (uint)snapshotLimit)
                            {
                                SaveRecurrentState(perStepSnapshots[snapIdx]);
                            }
                        }
                    }

                    base.VerifyBatchLogits(tokSlice, startPos + offset, logitSlice, vocabStride, kvCache, null);
                }
                else
                {
                    ForwardBatchChunkUnified(
                        tokSlice,
                        startPos + offset,
                        default,
                        default,
                        Span<float>.Empty,
                        logitSlice,
                        vocabStride,
                        kvCache,
                        computeLastLogits: false,
                        computeAllLogits: true,
                        allowTailSkip: true);
                }

                offset += batchSize;
            }
        }
        finally
        {
            _activePerStepSnapshots = null;
            _activeSnapshotOffset = 0;
            _activeSnapshotLimit = 0;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardBatchChunkUnified(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> lastTokenLogits,
        Span<float> allTokensLogits,
        int allTokensLogitsStride,
        KVCache? kvCache,
        bool computeLastLogits,
        bool computeAllLogits,
        bool allowTailSkip)
    {
        int batchSize = !chunkTokens.IsEmpty ? chunkTokens.Length : (inputXBatch.Length / _dim);
        if (batchSize <= 0) return;

        ObjectDisposedException.ThrowIf(_disposed, this);

        int normXChunks = Math.Max(1, (_dim + 31) / 32);

        byte[][]? perStepSnapshots = _activePerStepSnapshots;
        int snapshotOffset = _activeSnapshotOffset;
        int snapshotLimit = _activeSnapshotLimit;

        int ssmBytes = checked((int)_ssmCache.TotalBytes);
        int prevCount = _pleProcessor?.NgramPrevCount ?? 0;
        int prevBytes = prevCount * sizeof(int);
        int pleHistBase = ssmBytes + prevBytes;
        int pleLayerHistBytes = _pleProcessor != null ? checked((int)_pleProcessor.LayerHistoryBytes) : 0;

        if (perStepSnapshots != null && LayerCount < _weights.BlockCount)
        {
            for (int t = 0; t < batchSize; t++)
            {
                int snapIdx = snapshotOffset + t;
                if ((uint)snapIdx < (uint)snapshotLimit)
                {
                    SaveRecurrentState(perStepSnapshots[snapIdx]);
                }
            }
        }

        // Issue concurrent PLE NVMe/mmap lookups for every token in the batch before layer 0
        PleLookupTicket[]? pleTickets = null;
        if (StartLayer == 0 && _pleProcessor != null)
        {
            Span<int> prevWindow = prevCount <= 32 ? stackalloc int[prevCount] : new int[prevCount];
            ReadOnlySpan<int> initialPrev = _pleProcessor.PreviousTokens;

            if (perStepSnapshots != null && !chunkTokens.IsEmpty)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    int snapIdx = snapshotOffset + t;
                    if ((uint)snapIdx < (uint)snapshotLimit)
                    {
                        PleLayerProcessor.ComputePrevWindowAfterSteps(
                            initialPrev,
                            chunkTokens,
                            t + 1,
                            MemoryMarshal.Cast<byte, int>(perStepSnapshots[snapIdx].AsSpan(ssmBytes, prevBytes)));
                    }
                }
            }

            if (_pleTable != null && !chunkTokens.IsEmpty)
            {
                pleTickets = new PleLookupTicket[batchSize];
                for (int t = 0; t < batchSize; t++)
                {
                    if (t == 0)
                    {
                        pleTickets[t] = _pleTable.IssueToken(chunkTokens[t], initialPrev);
                    }
                    else
                    {
                        PleLayerProcessor.ComputePrevWindowAfterSteps(initialPrev, chunkTokens, t, prevWindow);
                        pleTickets[t] = _pleTable.IssueToken(chunkTokens[t], prevWindow);
                    }
                }
            }
        }

        if (StartLayer == 0)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], _xBatch + (long)t * _dim, _dim);
            }
        }
        else
        {
            fixed (float* pInput = inputXBatch)
            {
                Buffer.MemoryCopy(pInput, _xBatch, (ulong)((long)batchSize * _dim * sizeof(float)), (ulong)((long)batchSize * _dim * sizeof(float)));
            }
        }

        if (_hc != null)
        {
            for (int t = 0; t < batchSize; t++)
            {
                _hc.InitTokenFromEmbedding(_xBatch + (long)t * _dim, t);
            }
        }

        bool pleCollected = false;
        try
        {
            for (int l = 0; l < LayerCount; l++)
            {
                int modelLayer = StartLayer + l;
                var layer = _weights.Layers[modelLayer];

                if (layer.HasPle && _pleProcessor != null)
                {
                    if (!pleCollected)
                    {
                        for (int t = 0; t < batchSize; t++)
                        {
                            float* pleDst = _pleEmbBatch + (long)t * _pleEmbDim;
                            if (pleTickets != null && pleTickets[t].IsValid && _pleTable != null)
                            {
                                _pleTable.Collect(pleTickets[t], new Span<float>(pleDst, _pleEmbDim));
                                pleTickets[t] = default;
                            }
                            else
                            {
                                new Span<float>(pleDst, _pleEmbDim).Clear();
                            }
                        }
                        pleCollected = true;
                    }

                    var pleWeights = new PleLayerWeights
                    {
                        PleKeyType = layer.PleKeyType,
                        PleKeyWeight = layer.PleKeyWeight,
                        PleValueType = layer.PleValueType,
                        PleValueWeight = layer.PleValueWeight,
                        PleNormKey = layer.PleNormKeyWeight,
                        PleNormQuery = layer.PleNormQueryWeight,
                        PleNormConv = layer.PleNormConvWeight,
                        PleConv1dWeight = layer.PleConv1dWeight
                    };
                    int pleSlot = _pleSlotMap![modelLayer];
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* rPtrT = _hc != null ? _hc.GetResidualTokenPtr(t) : (_xBatch + (long)t * _dim);
                        _pleProcessor.ExecuteToken(pleSlot, rPtrT, _pleEmbBatch + (long)t * _pleEmbDim, pleWeights);
                        int snapIdx = snapshotOffset + t;
                        if (perStepSnapshots != null && (uint)snapIdx < (uint)snapshotLimit)
                        {
                            _pleProcessor.SaveLayerHistory(
                                pleSlot,
                                perStepSnapshots[snapIdx].AsSpan(checked(pleHistBase + pleSlot * pleLayerHistBytes), pleLayerHistBytes));
                        }
                    }
                }

                // 1. Attention / GDN sub-block
                if (_hc != null && layer.HasHyperConnections)
                {
                    _hc.ReadBatch(
                        batchSize,
                        layer.HcAttnNormWeight,
                        layer.HcAttnDownType,
                        layer.HcAttnDownWeight,
                        layer.HcAttnUpType,
                        layer.HcAttnUpWeight,
                        layer.HcAttnInjectType,
                        layer.HcAttnInjectWeight,
                        _normXBatch);
                    for (int t = 0; t < batchSize; t++)
                    {
                        QuantKernels.ComputeBlockSums32(_normXBatch + (long)t * _dim, _normXSumBatch + (long)t * normXChunks, _dim);
                    }

                    if (layer.IsGdn)
                    {
                        ForwardGdnBatchLayer(l, modelLayer, _normXBatch, _normXSumBatch, _attnProjBatch, batchSize, addToXBatch: false, perStepSnapshots, snapshotOffset, snapshotLimit);
                    }
                    else
                    {
                        ForwardStandardAttentionBatchLayer(l, modelLayer, chunkStartPos, batchSize, kvCache, addToXBatch: false, perStepSnapshots, snapshotOffset, snapshotLimit);
                    }

                    _hc.WriteBatch(batchSize, _attnProjBatch);
                }
                else
                {
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* rPtrT = _hc != null ? _hc.GetResidualTokenPtr(t) : (_xBatch + (long)t * _dim);
                        float* normXt = _normXBatch + (long)t * _dim;
                        QuantKernels.RMSNorm(rPtrT, layer.AttnNormWeight, normXt, _dim, _weights.RmsNormEps);
                        QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + (long)t * normXChunks, _dim);
                    }

                    if (layer.IsGdn)
                    {
                        ForwardGdnBatchLayer(l, modelLayer, _normXBatch, _normXSumBatch, _attnProjBatch, batchSize, addToXBatch: false, perStepSnapshots, snapshotOffset, snapshotLimit);
                    }
                    else
                    {
                        ForwardStandardAttentionBatchLayer(l, modelLayer, chunkStartPos, batchSize, kvCache, addToXBatch: false, perStepSnapshots, snapshotOffset, snapshotLimit);
                    }

                    for (int t = 0; t < batchSize; t++)
                    {
                        float* rPtrT = _hc != null ? _hc.GetResidualTokenPtr(t) : (_xBatch + (long)t * _dim);
                        AddVector(rPtrT, _attnProjBatch + (long)t * _dim, _dim);
                    }
                }

                // 2. FFN / MoE sub-block
                if (_hc != null && layer.HasHyperConnections)
                {
                    _hc.ReadBatch(
                        batchSize,
                        layer.HcFfnNormWeight,
                        layer.HcFfnDownType,
                        layer.HcFfnDownWeight,
                        layer.HcFfnUpType,
                        layer.HcFfnUpWeight,
                        layer.HcFfnInjectType,
                        layer.HcFfnInjectWeight,
                        _normXBatch);
                    for (int t = 0; t < batchSize; t++)
                    {
                        QuantKernels.ComputeBlockSums32(_normXBatch + (long)t * _dim, _normXSumBatch + (long)t * normXChunks, _dim);
                    }

                    ComputeBatchChunkFfnOut(modelLayer, batchSize, allowTailSkip);
                    _hc.WriteBatch(batchSize, _ffnOutBatch);
                }
                else
                {
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* rPtrT = _hc != null ? _hc.GetResidualTokenPtr(t) : (_xBatch + (long)t * _dim);
                        float* normXt = _normXBatch + (long)t * _dim;
                        QuantKernels.RMSNorm(rPtrT, layer.FfnNormWeight, normXt, _dim, _weights.RmsNormEps);
                        QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + (long)t * normXChunks, _dim);
                    }

                    ComputeBatchChunkFfnOut(modelLayer, batchSize, allowTailSkip);
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* rPtrT = _hc != null ? _hc.GetResidualTokenPtr(t) : (_xBatch + (long)t * _dim);
                        AddVector(rPtrT, _ffnOutBatch + (long)t * _dim, _dim);
                    }
                }
            }
        }
        finally
        {
            if (pleTickets != null && _pleTable != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    if (pleTickets[t].IsValid)
                    {
                        _pleTable.Collect(pleTickets[t], new Span<float>(_pleEmbBatch + (long)t * _pleEmbDim, _pleEmbDim));
                    }
                }
            }
        }

        if (StartLayer == 0 && _pleProcessor != null && !chunkTokens.IsEmpty)
        {
            for (int t = 0; t < batchSize; t++)
            {
                _pleProcessor.PushToken(chunkTokens[t]);
            }
        }

        if (IsLastStage && _hc != null && _lastHcHidden != null)
        {
            ulong hcBytes = (ulong)_dim * (ulong)_hcCount * sizeof(float);
            if (_hcHiddenHistory != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    int pos = chunkStartPos + t;
                    if ((uint)pos < (uint)_maxSeqLen)
                    {
                        Buffer.MemoryCopy(_hc.GetResidualTokenPtr(t), _hcHiddenHistory + (long)pos * (_dim * _hcCount), hcBytes, hcBytes);
                    }
                }
            }

            Buffer.MemoryCopy(_hc.GetResidualTokenPtr(batchSize - 1), _lastHcHidden, hcBytes, hcBytes);
        }

        if (_hc != null && _weights.OutputHcDownWeight != null)
        {
            _hc.ReadBatch(
                batchSize,
                _weights.OutputHcNormWeight,
                _weights.OutputHcDownType,
                _weights.OutputHcDownWeight,
                _weights.OutputHcUpType,
                _weights.OutputHcUpWeight,
                GgufType.F32,
                null,
                _xBatch);
        }
        else if (_hc != null)
        {
            ulong vecBytes = (ulong)_dim * sizeof(float);
            for (int t = 0; t < batchSize; t++)
            {
                Buffer.MemoryCopy(_hc.GetResidualTokenPtr(t), _xBatch + (long)t * _dim, vecBytes, vecBytes);
            }
        }

        FinalizeBatchChunkOutputs(
            chunkStartPos,
            batchSize,
            outputXBatch,
            lastTokenLogits,
            allTokensLogits,
            allTokensLogitsStride,
            computeLastLogits,
            computeAllLogits);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardAttention(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsGdn)
        {
            ForwardGdnLayer(stageLayer, modelLayer, x);
        }
        else
        {
            ForwardStandardAttentionLayer(stageLayer, modelLayer, pos, x, kvCache);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayer(int stageLayer, int modelLayer, float* x)
    {
        ForwardGdnLayerWithNorm(stageLayer, modelLayer, x, _normX, _normXSums, addResidual: true);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayerWithNorm(int stageLayer, int modelLayer, float* x, float* normX, float* normXSums, bool addResidual = true)
    {
        var layer = _weights.Layers[modelLayer];

        // 1. Compute linear projections from normalized x
        QuantKernels.MatVecMul(layer.QkvType, layer.QkvWeight, normX, _gdnQkv, _dim, _gdnConvChannels, normXSums);
        QuantKernels.MatVecMul(layer.AttnGateType, layer.AttnGateWeight, normX, _gdnZ, _dim, _ssmInnerSize, normXSums);
        QuantKernels.MatVecMul(layer.SsmAlphaType, layer.SsmAlphaWeight, normX, _gdnA, _dim, _ssmHeads, normXSums);
        QuantKernels.MatVecMul(layer.SsmBetaType, layer.SsmBetaWeight, normX, _gdnB, _dim, _ssmHeads, normXSums);

        AdvanceGdnTokenState(layer, modelLayer, _gdnQkv, _gdnA, _gdnB, _gdnZ, _gdnY);

        // 7. Output projection back to residual stream
        QuantKernels.ComputeBlockSums32(_gdnY, _gdnYSums, _ssmInnerSize);
        QuantKernels.MatVecMul(layer.SsmOutType, layer.SsmOutWeight, _gdnY, _attnProj, _ssmInnerSize, _dim, _gdnYSums);

        if (addResidual)
        {
            // Residual connection: x = x + attnProj (projected out)
            AddVector(x, _attnProj, _dim);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void AdvanceGdnTokenState(
        LayerWeights layer,
        int modelLayer,
        float* qkv,
        float* aVec,
        float* bVec,
        float* zVec,
        float* yOut)
    {
        if (layer.SsmConv1dWeight == null)
        {
            throw new InvalidDataException($"Missing SsmConv1dWeight for GDN layer {modelLayer}.");
        }

        // 2. Depthwise 1D Causal Convolution with SiLU activation
        float* convState = _ssmCache.GetConvState(modelLayer);
        int convHistLen = _ssmConvKernel - 1; // 3
        for (int c = 0; c < _gdnConvChannels; c++)
        {
            float* w = layer.SsmConv1dWeight + c * _ssmConvKernel;
            float* h = convState + c * convHistLen;

            float acc = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + qkv[c] * w[3];

            h[0] = h[1];
            h[1] = h[2];
            h[2] = qkv[c];

            float s = 1.0f / (1.0f + MathF.Exp(-acc));
            _gdnConvOut[c] = acc * s;
        }

        // 3. Partition channels into Q, K, V
        int qLen = _ssmGroupCount * _ssmStateDim;
        int kLen = _ssmGroupCount * _ssmStateDim;

        float* qPtr = _gdnConvOut;
        float* kPtr = _gdnConvOut + qLen;
        float* vPtr = _gdnConvOut + qLen + kLen;

        // 4. L2 Normalization of Q and K per group
        float qkScale = 1.0f / MathF.Sqrt(_ssmStateDim);
        for (int g = 0; g < _ssmGroupCount; g++)
        {
            float* qg = qPtr + g * _ssmStateDim;
            float* kg = kPtr + g * _ssmStateDim;

            float sumSqQ = 0f;
            float sumSqK = 0f;
            for (int d = 0; d < _ssmStateDim; d++)
            {
                sumSqQ += qg[d] * qg[d];
                sumSqK += kg[d] * kg[d];
            }

            float invNormQ = sumSqQ > 1e-12f ? (1.0f / MathF.Sqrt(sumSqQ + 1e-6f)) * qkScale : 0f;
            float invNormK = sumSqK > 1e-12f ? 1.0f / MathF.Sqrt(sumSqK + 1e-6f) : 0f;

            for (int d = 0; d < _ssmStateDim; d++)
            {
                qg[d] *= invNormQ;
                kg[d] *= invNormK;
            }
        }

        // 5. Recurrent DeltaNet associative state update across all heads
        int dState = _ssmStateDim;
        float* delta = stackalloc float[dState];

        for (int h = 0; h < _ssmHeads; h++)
        {
            // llama.cpp maps V-head -> Q/K-head by tiling (iq1 = iv1 % nK), not by grouping (h / ratio).
            int g = h % _ssmGroupCount;
            float* qHead = qPtr + g * dState;
            float* kHead = kPtr + g * dState;
            float* vHead = vPtr + h * dState;
            float* sMat = _ssmCache.GetRecurrentState(modelLayer, h);

            float aVal = aVec[h] + (layer.SsmDtBias != null ? layer.SsmDtBias[h] : 0f);
            float dt = aVal > 20.0f ? aVal : MathF.Log(1.0f + MathF.Exp(aVal));
            float aLog = layer.SsmAWeight != null ? layer.SsmAWeight[h] : 0f;
            float gVal = aLog * dt; // In GGUF, ssm_a is already stored as -exp(A_log)
            float decay = MathF.Exp(gVal);

            float bVal = bVec[h];
            float beta = 1.0f / (1.0f + MathF.Exp(-bVal));

            // Step 1 & 2: Decay recurrent state and retrieve memory: kvMem = (decay * S_{t-1}) * k
            for (int i = 0; i < dState; i++)
            {
                float kvMem = 0f;
                float* sRow = sMat + i * dState;
                for (int j = 0; j < dState; j++)
                {
                    float decayed = sRow[j] * decay;
                    sRow[j] = decayed;
                    kvMem += decayed * kHead[j];
                }
                delta[i] = (vHead[i] - kvMem) * beta;
            }

            // Step 3 & 4: S_t = S' + delta * k^T and compute output y_h = S_t * q
            float* yHead = yOut + h * dState;
            for (int i = 0; i < dState; i++)
            {
                float* sRow = sMat + i * dState;
                float dVal = delta[i];
                float dot = 0f;
                for (int j = 0; j < dState; j++)
                {
                    float updated = sRow[j] + dVal * kHead[j];
                    sRow[j] = updated;
                    dot += updated * qHead[j];
                }
                yHead[i] = dot;
            }

            // Step 5: Head RMSNorm
            if (layer.SsmNormWeight != null)
            {
                QuantKernels.RMSNorm(yHead, layer.SsmNormWeight, yHead, dState, _weights.RmsNormEps);
            }
        }

        // 6. Output gating on z: qwen4exp uses sigmoid(z), whereas Qwen3.5 uses silu(z) = z * sigmoid(z)
        if (_useSigmoidGdnGate)
        {
            for (int i = 0; i < _ssmInnerSize; i++)
            {
                float sig = 1.0f / (1.0f + MathF.Exp(-zVec[i]));
                yOut[i] *= sig;
            }
        }
        else
        {
            for (int i = 0; i < _ssmInnerSize; i++)
            {
                float zVal = zVec[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-zVal));
                yOut[i] *= (zVal * sig);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnBatchLayer(
        int stageLayer,
        int modelLayer,
        float* mixedInBatch,
        float* mixedInSumBatch,
        float* outBatch,
        int batchSize,
        bool addToXBatch,
        byte[][]? perStepSnapshots,
        int snapshotTokenOffset,
        int snapshotLimit)
    {
        var layer = _weights.Layers[modelLayer];

        QuantKernels.MatMulBatch(layer.QkvType, layer.QkvWeight, mixedInBatch, _gdnQkvBatch, _dim, _gdnConvChannels, batchSize, mixedInSumBatch);
        QuantKernels.MatMulBatch(layer.AttnGateType, layer.AttnGateWeight, mixedInBatch, _gdnZBatch, _dim, _ssmInnerSize, batchSize, mixedInSumBatch);
        QuantKernels.MatMulBatch(layer.SsmAlphaType, layer.SsmAlphaWeight, mixedInBatch, _gdnABatch, _dim, _ssmHeads, batchSize, mixedInSumBatch);
        QuantKernels.MatMulBatch(layer.SsmBetaType, layer.SsmBetaWeight, mixedInBatch, _gdnBBatch, _dim, _ssmHeads, batchSize, mixedInSumBatch);

        int yChunks = (_ssmInnerSize + 31) / 32;
        int ssmLayerBytes = perStepSnapshots != null ? checked((int)_ssmCache.PerLayerBytes) : 0;
        int ssmByteOff = modelLayer * ssmLayerBytes;

        for (int t = 0; t < batchSize; t++)
        {
            float* yT = _gdnYBatch + (long)t * _ssmInnerSize;
            AdvanceGdnTokenState(
                layer,
                modelLayer,
                _gdnQkvBatch + (long)t * _gdnConvChannels,
                _gdnABatch + (long)t * _ssmHeads,
                _gdnBBatch + (long)t * _ssmHeads,
                _gdnZBatch + (long)t * _ssmInnerSize,
                yT);

            int snapIdx = snapshotTokenOffset + t;
            if (perStepSnapshots != null && (uint)snapIdx < (uint)snapshotLimit)
            {
                _ssmCache.SaveLayerState(modelLayer, perStepSnapshots[snapIdx].AsSpan(ssmByteOff, ssmLayerBytes));
            }

            QuantKernels.ComputeBlockSums32(yT, _gdnYSumBatch + (long)t * yChunks, _ssmInnerSize);
        }

        QuantKernels.MatMulBatch(layer.SsmOutType, layer.SsmOutWeight, _gdnYBatch, outBatch, _ssmInnerSize, _dim, batchSize, _gdnYSumBatch);
        if (addToXBatch)
        {
            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + (long)t * _dim, outBatch + (long)t * _dim, _dim);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardStandardAttentionLayer(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache, bool addResidual = true)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        if (layer.HasQGate)
        {
            QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _qFull, _dim, 2 * qDim, _normXSums);
            if (layer.QBias != null) AddVector(_qFull, layer.QBias, 2 * qDim);

            // Split interleaved heads: for each head h, first headDim is Q, second headDim is Q-Gate
            for (int h = 0; h < _nHeads; h++)
            {
                float* srcHead = _qFull + h * (2 * _headDim);
                Buffer.MemoryCopy(srcHead, _q + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
                Buffer.MemoryCopy(srcHead + _headDim, _qGate + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
            }
        }
        else
        {
            QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
            if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
        }

        QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
        QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

        if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
        if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);

        if (layer.AttnQNormWeight != null)
        {
            for (int h = 0; h < _nHeads; h++)
            {
                float* qHead = _q + h * _headDim;
                QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _weights.RmsNormEps);
            }
        }
        if (layer.AttnKNormWeight != null)
        {
            for (int h = 0; h < _nHeadsKv; h++)
            {
                float* kHead = _k + h * _headDim;
                QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _weights.RmsNormEps);
            }
        }

        QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight, ropeDim: _weights.RopeDimensionCount);
        kvCache?.Store(stageLayer, pos, _k, _v);

        if (kvCache != null)
        {
            ComputeStandardAttentionToken(stageLayer, modelLayer, pos, _normX, _q, _attnOut, kvCache);
        }

        if (layer.HasQGate)
        {
            for (int j = 0; j < qDim; j++)
            {
                float sig = 1.0f / (1.0f + MathF.Exp(-_qGate[j]));
                _attnOut[j] *= sig;
            }
        }

        QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
        QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
        if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

        if (addResidual)
        {
            AddVector(x, _attnProj, _dim);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeStandardAttentionToken(int stageLayer, int modelLayer, int pos, float* normX, float* qHeadBase, float* outHeadBase, KVCache kvCache)
    {
        var layerWeights = _weights.Layers[modelLayer];
        var qsa = _qsaIndexers != null ? _qsaIndexers[modelLayer] : null;
        bool useSparse = qsa != null && layerWeights.IsQsa;

        int* selectedIds = null;
        int startT = 0;
        int kvCount;

        if (useSparse)
        {
            if (pos < 0 || pos >= _maxSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Sequence position {pos} exceeds maxSeqLen {_maxSeqLen}.");
            }

            var qsaWeights = new QsaIndexerWeights
            {
                QProjType = layerWeights.IndexerQProjType,
                QProjWeight = layerWeights.IndexerQProjWeight,
                KProjType = layerWeights.IndexerKProjType,
                KProjWeight = layerWeights.IndexerKProjWeight,
                QNormWeight = layerWeights.IndexerQNormWeight,
                KNormWeight = layerWeights.IndexerKNormWeight
            };
            qsa!.RecordKeyAndSelect(pos, normX, in qsaWeights, out selectedIds, out kvCount);
            if (kvCount <= 0)
            {
                new Span<float>(outHeadBase, _nHeads * _vHeadDim).Clear();
                return;
            }
        }
        else
        {
            bool useWindow = kvCache.IsRingBuffer || kvCache.SlidingWindow > 0;
            int windowSize = useWindow ? Math.Min(_maxSeqLen, kvCache.EffectiveWindowSize) : _maxSeqLen;
            startT = useWindow ? Math.Max(0, pos + 1 - windowSize) : 0;
            kvCount = pos - startT + 1;
            if (pos < 0 || kvCount > _maxSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Sequence position {pos} exceeds maxSeqLen {_maxSeqLen}.");
            }
        }

        for (int h = 0; h < _nHeads; h++)
        {
            int hKv = h / _groupSize;
            float* qHead = qHeadBase + h * _headDim;
            float* scores = _headScores + (long)h * _maxSeqLen;

            for (int idx = 0; idx < kvCount; idx++)
            {
                int t = useSparse ? selectedIds[idx] : startT + idx;
                float dot = kvCache.DotKey(stageLayer, hKv, t, qHead);
                scores[idx] = dot * _attnScale;
            }

            float? sinkLogit = layerWeights.AttnSinksWeight != null ? (float?)layerWeights.AttnSinksWeight[h] : null;
            QuantKernels.Softmax(scores, kvCount, sinkLogit);

            float* outHead = outHeadBase + h * _vHeadDim;
            for (int d = 0; d < _vHeadDim; d++) outHead[d] = 0f;

            for (int idx = 0; idx < kvCount; idx++)
            {
                int t = useSparse ? selectedIds[idx] : startT + idx;
                kvCache.AccumulateValue(stageLayer, hKv, t, scores[idx], outHead);
            }
        }
    }

    public override void ResetState()
    {
        _ssmCache.Reset();
        _pleProcessor?.Reset();
        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                _qsaIndexers[i]?.Reset();
            }
        }
    }

    public override long StateSnapshotBytes
    {
        get
        {
            long bytes = _ssmCache.TotalBytes;
            if (_pleProcessor != null) bytes += _pleProcessor.StateSnapshotBytes;
            if (_qsaIndexers != null)
            {
                for (int i = 0; i < _qsaIndexers.Length; i++)
                {
                    if (_qsaIndexers[i] != null) bytes += _qsaIndexers[i]!.StateSnapshotBytes;
                }
            }
            return bytes;
        }
    }

    public override void SaveRecurrentState(Span<byte> destination)
    {
        int ssmBytes = checked((int)_ssmCache.TotalBytes);
        _ssmCache.SaveState(destination.Slice(0, ssmBytes));
        int offset = ssmBytes;

        if (_pleProcessor != null)
        {
            int pleBytes = checked((int)_pleProcessor.StateSnapshotBytes);
            _pleProcessor.SaveState(destination.Slice(offset, pleBytes));
            offset += pleBytes;
        }

        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                var qsa = _qsaIndexers[i];
                if (qsa == null) continue;
                int qsaBytes = checked((int)qsa.StateSnapshotBytes);
                qsa.SaveState(destination.Slice(offset, qsaBytes));
                offset += qsaBytes;
            }
        }
    }

    public override void RestoreRecurrentState(ReadOnlySpan<byte> source)
    {
        int ssmBytes = checked((int)_ssmCache.TotalBytes);
        _ssmCache.RestoreState(source.Slice(0, ssmBytes));
        int offset = ssmBytes;

        if (_pleProcessor != null)
        {
            int pleBytes = checked((int)_pleProcessor.StateSnapshotBytes);
            _pleProcessor.RestoreState(source.Slice(offset, pleBytes));
            offset += pleBytes;
        }

        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                var qsa = _qsaIndexers[i];
                if (qsa == null) continue;
                int qsaBytes = checked((int)qsa.StateSnapshotBytes);
                qsa.RestoreState(source.Slice(offset, qsaBytes));
                offset += qsaBytes;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardBatchChunkAttention(int stageLayer, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        if (layer.IsGdn)
        {
            ForwardGdnBatchLayer(
                stageLayer,
                modelLayer,
                _normXBatch,
                _normXSumBatch,
                _attnProjBatch,
                batchSize,
                addToXBatch: true,
                _activePerStepSnapshots,
                _activeSnapshotOffset,
                _activeSnapshotLimit);
        }
        else
        {
            ForwardStandardAttentionBatchLayer(
                stageLayer,
                modelLayer,
                chunkStartPos,
                batchSize,
                kvCache,
                addToXBatch: true,
                _activePerStepSnapshots,
                _activeSnapshotOffset,
                _activeSnapshotLimit);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardStandardAttentionBatchLayer(
        int stageLayer,
        int modelLayer,
        int chunkStartPos,
        int batchSize,
        KVCache? kvCache,
        bool addToXBatch,
        byte[][]? perStepSnapshots,
        int snapshotTokenOffset,
        int snapshotLimit)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        if (layer.HasQGate)
        {
            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qFullBatch, _dim, 2 * qDim, batchSize, _normXSumBatch);
            for (int t = 0; t < batchSize; t++)
            {
                float* qFull = _qFullBatch + t * (2 * qDim);
                if (layer.QBias != null) AddVector(qFull, layer.QBias, 2 * qDim);

                float* q = _qBatch + t * qDim;
                float* qGate = _qGateBatch + t * qDim;
                for (int h = 0; h < _nHeads; h++)
                {
                    float* srcHead = qFull + h * (2 * _headDim);
                    Buffer.MemoryCopy(srcHead, q + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
                    Buffer.MemoryCopy(srcHead + _headDim, qGate + h * _headDim, (ulong)(_headDim * sizeof(float)), (ulong)(_headDim * sizeof(float)));
                }
            }
        }
        else
        {
            QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
            if (layer.QBias != null)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    AddVector(_qBatch + t * qDim, layer.QBias, qDim);
                }
            }
        }

        QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumBatch);
        QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumBatch);

        if (layer.KBias != null || layer.VBias != null)
        {
            for (int t = 0; t < batchSize; t++)
            {
                if (layer.KBias != null) AddVector(_kBatch + t * kvDim, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(_vBatch + t * kvDim, layer.VBias, kvDim);
            }
        }

        var qsa = _qsaIndexers != null ? _qsaIndexers[modelLayer] : null;
        int qsaOffset = qsa != null && _qsaSnapshotOffsets != null ? _qsaSnapshotOffsets[modelLayer] : 0;
        int qsaBytes = qsa != null ? checked((int)qsa.StateSnapshotBytes) : 0;

        for (int t = 0; t < batchSize; t++)
        {
            int pos = chunkStartPos + t;
            float* q = _qBatch + t * qDim;
            float* k = _kBatch + t * kvDim;
            float* v = _vBatch + t * kvDim;

            if (layer.AttnQNormWeight != null)
            {
                for (int h = 0; h < _nHeads; h++)
                {
                    float* qHead = q + h * _headDim;
                    QuantKernels.RMSNorm(qHead, layer.AttnQNormWeight, qHead, _headDim, _weights.RmsNormEps);
                }
            }
            if (layer.AttnKNormWeight != null)
            {
                for (int h = 0; h < _nHeadsKv; h++)
                {
                    float* kHead = k + h * _headDim;
                    QuantKernels.RMSNorm(kHead, layer.AttnKNormWeight, kHead, _headDim, _weights.RmsNormEps);
                }
            }

            QuantKernels.RoPE(q, k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight, ropeDim: _weights.RopeDimensionCount);
            kvCache?.Store(stageLayer, pos, k, v);

            if (kvCache != null)
            {
                ComputeStandardAttentionToken(stageLayer, modelLayer, pos, _normXBatch + t * _dim, q, _attnOutBatch + t * qDim, kvCache);

                int snapIdx = snapshotTokenOffset + t;
                if (qsa != null && perStepSnapshots != null && (uint)snapIdx < (uint)snapshotLimit)
                {
                    qsa.SaveState(perStepSnapshots[snapIdx].AsSpan(qsaOffset, qsaBytes));
                }
            }
        }

        if (layer.HasQGate)
        {
            for (int t = 0; t < batchSize; t++)
            {
                float* attnOut = _attnOutBatch + t * qDim;
                float* qGate = _qGateBatch + t * qDim;
                for (int j = 0; j < qDim; j++)
                {
                    float sig = 1.0f / (1.0f + MathF.Exp(-qGate[j]));
                    attnOut[j] *= sig;
                }
            }
        }

        int attnOutChunks = (qDim + 31) / 32;
        for (int t = 0; t < batchSize; t++)
        {
            QuantKernels.ComputeBlockSums32(_attnOutBatch + t * qDim, _attnOutSumBatch + t * attnOutChunks, qDim);
        }
        QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, qDim, _dim, batchSize, _attnOutSumBatch);
        if (layer.AttnOutBias != null)
        {
            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_attnProjBatch + t * _dim, layer.AttnOutBias, _dim);
            }
        }

        if (addToXBatch)
        {
            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
            }
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            _ssmCache?.Dispose();
            _hc?.Dispose();
            FreeIfAllocated(ref _hcInject);
            FreeIfAllocated(ref _lastHcHidden);
            FreeIfAllocated(ref _hcHiddenHistory);
            _pleProcessor?.Dispose();
            FreeIfAllocated(ref _pleEmbBuf);
            FreeIfAllocated(ref _pleEmbBatch);
            if (_ownsPleTable)
            {
                _pleTable?.Dispose();
            }
            if (_qsaIndexers != null)
            {
                for (int i = 0; i < _qsaIndexers.Length; i++)
                {
                    _qsaIndexers[i]?.Dispose();
                }
            }

            FreeIfAllocated(ref _gdnQkv);
            FreeIfAllocated(ref _gdnConvOut);
            FreeIfAllocated(ref _gdnZ);
            FreeIfAllocated(ref _gdnA);
            FreeIfAllocated(ref _gdnB);
            FreeIfAllocated(ref _gdnY);
            FreeIfAllocated(ref _gdnYSums);

            FreeIfAllocated(ref _gdnQkvBatch);
            FreeIfAllocated(ref _gdnZBatch);
            FreeIfAllocated(ref _gdnABatch);
            FreeIfAllocated(ref _gdnBBatch);
            FreeIfAllocated(ref _gdnYBatch);
            FreeIfAllocated(ref _gdnYSumBatch);

            FreeIfAllocated(ref _q);
            FreeIfAllocated(ref _qFull);
            FreeIfAllocated(ref _qGate);
            FreeIfAllocated(ref _k);
            FreeIfAllocated(ref _v);
            FreeIfAllocated(ref _attnOut);
            FreeIfAllocated(ref _attnOutSums);
            FreeIfAllocated(ref _attnProj);
            FreeIfAllocated(ref _headScores);

            FreeIfAllocated(ref _qBatch);
            FreeIfAllocated(ref _qFullBatch);
            FreeIfAllocated(ref _qGateBatch);
            FreeIfAllocated(ref _kBatch);
            FreeIfAllocated(ref _vBatch);
            FreeIfAllocated(ref _attnOutBatch);
            FreeIfAllocated(ref _attnOutSumBatch);
            FreeIfAllocated(ref _attnProjBatch);

            base.Dispose(disposing);
        }
    }
}
