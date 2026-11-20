namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Vortice.Direct3D12;

public sealed unsafe partial class Qwen3HybridD3D12Model : ID3D12Model, IHiddenStateSource
{
    private const ulong HostedSegmentTargetBytes = 64UL * 1024 * 1024;
    private const ulong HostedBatchReserveBytes = 128UL * 1024 * 1024;
    private const ulong SlotSubAlignmentBytes = 256UL;

    // ── qwen4exp Feature Flags & Dimensions ─────────────────────────────
    private bool _useSigmoidGdnGate;
    private bool _hasHc;
    private int _hcCount;
    private int _hcLowRank;
    private int _hcDim;

    // ── qwen4exp HLSL Root Signatures & PSOs ─────────────────────────────
    private ID3D12RootSignature? _sigHcBroadcastInit;
    private ID3D12PipelineState? _psoHcBroadcastInit;
    private ID3D12RootSignature? _sigHcStreamRmsNorm;
    private ID3D12PipelineState? _psoHcStreamRmsNorm;
    private ID3D12RootSignature? _sigHcScaleSilu;
    private ID3D12PipelineState? _psoHcScaleSilu;
    private ID3D12RootSignature? _sigHcGateMean;
    private ID3D12PipelineState? _psoHcGateMean;
    private ID3D12RootSignature? _sigHcWrite;
    private ID3D12PipelineState? _psoHcWrite;
    private ID3D12RootSignature? _sigSparseAttn;
    private ID3D12PipelineState? _psoSparseAttn;

    // ── Hyper-Connection Single & Batched Device/Staging Buffers ─────────
    private ID3D12Resource? _dHcR;
    private ID3D12Resource? _dHcXn;
    private ID3D12Resource? _dHcLo;
    private ID3D12Resource? _dHcUp;
    private ID3D12Resource? _dHcInject;
    private ID3D12Resource? _uploadHcR;
    private float* _pUploadHcR;
    private ID3D12Resource? _readbackHcR;
    private float* _pReadbackHcR;

    private ID3D12Resource? _dHcRBatch;
    private ID3D12Resource? _dHcXnBatch;
    private ID3D12Resource? _dHcLoBatch;
    private ID3D12Resource? _dHcUpBatch;
    private ID3D12Resource? _dHcInjectBatch;
    private ID3D12Resource? _uploadHcRBatch;
    private float* _pUploadHcRBatch;
    private ID3D12Resource? _readbackHcRBatch;
    private float* _pReadbackHcRBatch;
    private ID3D12Resource? _readbackNormXBatch;
    private float* _pReadbackNormXBatch;

    private ID3D12Resource? _dOutputHcNormWeight;
    private ID3D12Resource? _dOutputHcDownWeight;
    private GgufType _outputHcDownType;
    private ID3D12Resource? _dOutputHcUpWeight;
    private GgufType _outputHcUpType;

    // ── PLE (Per-Layer Embedding) Components ─────────────────────────────
    private PleLayerProcessor? _pleProcessor;
    private DirectIoNgramTable? _pleTable;
    private int[]? _pleSlotMap;
    private int _pleEmbDim;
    private float* _pleEmbBuf;
    private float* _pleEmbBatch;

    // ── QSA Sparse Attention Indexers ────────────────────────────────────
    private QsaSparseAttentionIndexer?[]? _qsaIndexers;
    private int[]? _qsaSnapshotOffsets;
    private ID3D12Resource? _dSelectedIds;
    private ID3D12Resource? _uploadSelectedIds;
    private uint* _pUploadSelectedIds;
    private int _selectedIdsCapacity;

    // ── Hosted MoE Segmented VRAM Pool & Co-Scheduling ───────────────────
    private readonly List<ID3D12Resource> _expertSegments = new();
    private int _slotsPerSegment;
    private int _hostedSlotCapacity;
    private int _hostedActiveSlots;
    private GpuResidentSlotManager? _slotManager;
    private ExpertRoutingProfile? _routingProfile;
    private BandwidthCoScheduler? _coScheduler;
    private ulong _maxAlignedSlotBytes;
    private ulong[]? _layerGateAlignedBytes;
    private ulong[]? _layerUpAlignedBytes;
    private ulong[]? _layerDownAlignedBytes;
    private GgufType[]? _layerGateGpuType;
    private GgufType[]? _layerUpGpuType;
    private GgufType[]? _layerDownGpuType;
    private bool[]? _layerHasFusedExps;
    private bool[]? _layerHasExpertBiases;
    private ID3D12Resource? _uploadExpertStaging;
    private byte* _pUploadExpertStaging;
    private int _stagingSlotCapacity;
    private ID3D12Resource? _readbackTopK;
    private byte* _pReadbackTopK;
    private long _gpuExpertHitCount;
    private long _gpuExpertTransferCount;
    private long _cpuExpertOverflowCount;

    // ── Hidden State History, Recurrent Snapshots, & Batched Verify ──────
    private float* _lastHidden;
    private float* _hiddenHistory;
    private float* _lastHcHidden;
    private float* _hcHiddenHistory;
    private byte[][]? _activePerStepSnapshots;
    private int _activeSnapshotOffset;
    private int _activeSnapshotLimit;
    private ID3D12Resource? _readbackHidden;
    private float* _pReadbackHidden;
    private ID3D12Resource? _readbackHiddenBatch;
    private float* _pReadbackHiddenBatch;
    private ID3D12Resource? _dLogitsBatch;
    private ID3D12Resource? _readbackLogitsBatch;
    private float* _pReadbackLogitsBatch;
    private int _allocatedVerifyLogitsCapacity;

    private long _ssmSnapshotBytes;
    private long[]? _ssmLayerSnapshotOffsets;
    private ID3D12Resource? _uploadSsmSnapshot;
    private byte* _pUploadSsmSnapshot;
    private ID3D12Resource? _readbackSsmSnapshot;
    private byte* _pReadbackSsmSnapshot;
    private ID3D12Resource? _readbackStepSsmSnapshot;
    private byte* _pReadbackStepSsmSnapshot;
    private int _stepSnapshotCapacity;

    public DirectIoNgramTable? PleTable => _pleTable;
    public int HostedMoeSlotCapacity => _hostedSlotCapacity;
    public int ActiveHostedMoeSlots => _hostedActiveSlots;
    public long GpuExpertHitCount => Interlocked.Read(ref _gpuExpertHitCount);
    public long GpuExpertTransferCount => Interlocked.Read(ref _gpuExpertTransferCount);
    public long CpuExpertOverflowCount => Interlocked.Read(ref _cpuExpertOverflowCount);
    public double GpuExpertExecutionRatio
    {
        get
        {
            long gpu = GpuExpertHitCount + GpuExpertTransferCount;
            long total = gpu + CpuExpertOverflowCount;
            return total > 0 ? (double)gpu / total : (_weights.IsMoe ? 1.0 : 0.0);
        }
    }

    public bool SupportsPerStepVerificationSnapshots => true;

    public int HiddenDim => _dim;
    public int MultiStreamHiddenDim => _hasHc ? _dim * _hcCount : _dim;

    public long StateSnapshotBytes
    {
        get
        {
            long total = _ssmSnapshotBytes;
            if (_pleProcessor != null) total += _pleProcessor.StateSnapshotBytes;
            if (_qsaIndexers != null)
            {
                for (int i = 0; i < _qsaIndexers.Length; i++)
                {
                    if (_qsaIndexers[i] != null) total += _qsaIndexers[i]!.StateSnapshotBytes;
                }
            }
            return total;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Initialization Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private void InitQwen4ExpFeatures(PleIoMode pleIoMode)
    {
        string arch = _weights.Gguf.Architecture ?? string.Empty;
        _useSigmoidGdnGate =
            string.Equals(arch, "qwen4exp", StringComparison.OrdinalIgnoreCase) ||
            _weights.HyperConnectionCount > 0 ||
            _weights.PleHashConfig != null;

        _hcCount = _weights.HyperConnectionCount;
        _hcLowRank = _weights.HyperConnectionLowRank;
        _hasHc = _hcCount > 0 && _hcLowRank > 0;
        _hcDim = _hasHc ? _hcCount * _dim : _dim;

        // Compute SSM snapshot offsets for Save/RestoreRecurrentState & speculative verification
        ulong convBytes = (ulong)_gdnConvChannels * 3UL * sizeof(float);
        ulong ssmBytes = (ulong)_ssmHeads * (ulong)_ssmStateDim * (ulong)_ssmStateDim * sizeof(float);
        _ssmLayerSnapshotOffsets = new long[LayerCount];
        long offset = 0;
        for (int l = 0; l < LayerCount; l++)
        {
            var lw = _weights.Layers[StartLayer + l];
            if (lw.IsGdn)
            {
                _ssmLayerSnapshotOffsets[l] = offset;
                offset += checked((long)(convBytes + ssmBytes));
            }
            else
            {
                _ssmLayerSnapshotOffsets[l] = -1;
            }
        }
        _ssmSnapshotBytes = offset;

        // QSA sparse attention indexers + PLE layers
        int pleLayerCount = 0;
        for (int i = 0; i < _weights.BlockCount; i++)
        {
            var layer = _weights.Layers[i];
            if (layer == null) continue;
            if (layer.HasPle) pleLayerCount++;
            if (layer.IsQsa && i >= StartLayer && i < StartLayer + LayerCount)
            {
                _qsaIndexers ??= new QsaSparseAttentionIndexer?[_weights.BlockCount];
                int idxBlock = layer.IndexerCompressRatio > 0 ? layer.IndexerCompressRatio : 4;
                _qsaIndexers[i] = new QsaSparseAttentionIndexer(
                    _dim,
                    _weights.IndexerHeadCount > 0 ? _weights.IndexerHeadCount : 4,
                    _weights.IndexerKeyLength > 0 ? _weights.IndexerKeyLength : 128,
                    idxBlock,
                    _weights.IndexerTopK > 0 ? _weights.IndexerTopK : 2048,
                    _ropeDim,
                    _weights.RopeFreqBase,
                    _weights.RmsNormEps,
                    _maxSeqLen);
            }
        }

        if (pleLayerCount > 0 && _weights.PleHashConfig != null)
        {
            _pleSlotMap = new int[_weights.BlockCount];
            Array.Fill(_pleSlotMap, -1);
            int slot = 0;
            for (int i = 0; i < _weights.BlockCount; i++)
            {
                if (_weights.Layers[i]?.HasPle == true)
                {
                    _pleSlotMap[i] = slot++;
                }
            }

            _pleProcessor = new PleLayerProcessor(
                pleLayerCount,
                Math.Max(1, _hcCount),
                _dim,
                _weights.PleHashConfig,
                _weights.RmsNormEps);
            _pleEmbDim = _weights.PleHashConfig.EmbeddingDim;
            _pleEmbBuf = (float*)NativeMemory.AllocZeroed((nuint)(_pleEmbDim * sizeof(float)));
            _pleEmbBatch = (float*)NativeMemory.AllocZeroed((nuint)(32L * _pleEmbDim * sizeof(float)));

            if (_weights.Gguf.PleTableFilePath != null &&
                _weights.Gguf.TryGetTensor("per_layer_token_embd.weight", out var pleTensor) &&
                pleTensor != null)
            {
                PleIoMode effectivePleIo = pleIoMode;
                if (effectivePleIo == PleIoMode.Auto)
                {
                    var snap = HardwareRuntimeProfiler.CaptureSnapshot();
                    effectivePleIo = HardwareRuntimeProfiler.RecommendPolicy(snap, _weights).RecommendedPleIoMode;
                }

                if (effectivePleIo == PleIoMode.DirectIo)
                {
                    _weights.Gguf.DetachExclusivePleShardMapping();
                }

                long totalRows = pleTensor.Dimensions.Length >= 2 ? (long)pleTensor.Dimensions[1] : _weights.PleHashConfig.TotalRows;
                int rowDim = pleTensor.Dimensions.Length >= 1 ? (int)pleTensor.Dimensions[0] : _weights.PleHashConfig.HeadDim;
                _pleTable = new DirectIoNgramTable(
                    _weights.Gguf.PleTableFilePath,
                    _weights.Gguf.PleTableByteOffset,
                    totalRows,
                    rowDim,
                    DirectIoNgramTable.FromGgufType(pleTensor.Type),
                    effectivePleIo,
                    hashConfig: _weights.PleHashConfig);
            }
        }

        if (_qsaIndexers != null)
        {
            _qsaSnapshotOffsets = new int[_qsaIndexers.Length];
            int qsaOff = checked((int)_ssmSnapshotBytes) + (_pleProcessor != null ? checked((int)_pleProcessor.StateSnapshotBytes) : 0);
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                if (_qsaIndexers[i] != null)
                {
                    _qsaSnapshotOffsets[i] = qsaOff;
                    qsaOff += checked((int)_qsaIndexers[i]!.StateSnapshotBytes);
                }
            }
        }
    }

    private void InitQwen4ExpPipelines()
    {
        if (_hasHc)
        {
            // BroadcastInit: b0(3 uints), u0(x), u1(r)
            var bcastParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 3), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
            };
            _sigHcBroadcastInit = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, bcastParams));
            _psoHcBroadcastInit = _ctx.CreatePipelineState(_sigHcBroadcastInit, _ctx.CompileShader(D3D12Shaders.Qwen4HcBroadcastInit));

            // StreamRmsNorm: b0(4 uints), t0(weight), u0(r), u1(xn)
            var streamNormParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
            };
            _sigHcStreamRmsNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, streamNormParams));
            _psoHcStreamRmsNorm = _ctx.CreatePipelineState(_sigHcStreamRmsNorm, _ctx.CompileShader(D3D12Shaders.Qwen4HcStreamRmsNorm));

            // ScaleSilu: b0(2 uints), u0(lo)
            var scaleSiluParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 2), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
            };
            _sigHcScaleSilu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, scaleSiluParams));
            _psoHcScaleSilu = _ctx.CreatePipelineState(_sigHcScaleSilu, _ctx.CompileShader(D3D12Shaders.Qwen4HcScaleSilu));

            // GateMean: b0(4 uints), u0(xn), u1(up), u2(mixed_out)
            var gateMeanParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
            };
            _sigHcGateMean = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gateMeanParams));
            _psoHcGateMean = _ctx.CreatePipelineState(_sigHcGateMean, _ctx.CompileShader(D3D12Shaders.Qwen4HcGateMean));

            // Write: b0(4 uints), u0(r), u1(sub_out), u2(inject)
            var writeParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
            };
            _sigHcWrite = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, writeParams));
            _psoHcWrite = _ctx.CreatePipelineState(_sigHcWrite, _ctx.CompileShader(D3D12Shaders.Qwen4HcWrite));
        }

        if (_qsaIndexers != null)
        {
            // Qwen3SparseAttention: b0(6 uints), t0(selected_ids), u0(q), u1(k_cache), u2(v_cache), u3(attn_out)
            var qsaParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
            };
            string sparseShader = KvPrecision switch
            {
                KvCachePrecision.Q4_0 => D3D12Shaders.Qwen3SparseAttentionQ4_0,
                KvCachePrecision.Fp16 => D3D12Shaders.Qwen3SparseAttentionF16,
                _ => D3D12Shaders.Qwen3SparseAttention
            };
            _sigSparseAttn = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, qsaParams));
            _psoSparseAttn = _ctx.CreatePipelineState(_sigSparseAttn, _ctx.CompileShader(sparseShader));
        }
    }

    private void AllocateQwen4ExpBuffers()
    {
        if (_hasHc)
        {
            _dHcR = _ctx.CreateDeviceBuffer((ulong)(_hcDim * sizeof(float)));
            _dHcXn = _ctx.CreateDeviceBuffer((ulong)(_hcDim * sizeof(float)));
            _dHcLo = _ctx.CreateDeviceBuffer((ulong)(_hcLowRank * sizeof(float)));
            _dHcUp = _ctx.CreateDeviceBuffer((ulong)(_hcDim * sizeof(float)));
            _dHcInject = _ctx.CreateDeviceBuffer((ulong)(_hcCount * sizeof(float)));

            _uploadHcR = _ctx.CreateUploadBuffer((ulong)(_hcDim * sizeof(float)));
            void* pUpHc = null;
            _uploadHcR.Map(0, null, &pUpHc);
            _pUploadHcR = (float*)pUpHc;

            _readbackHcR = _ctx.CreateReadbackBuffer((ulong)(_hcDim * sizeof(float)));
            void* pRbHc = null;
            _readbackHcR.Map(0, null, &pRbHc);
            _pReadbackHcR = (float*)pRbHc;
        }
        else if (_pleProcessor != null)
        {
            // Non-HC with PLE still needs host <-> GPU residual transfer on PLE layers
            _uploadHcR = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
            void* pUpHc = null;
            _uploadHcR.Map(0, null, &pUpHc);
            _pUploadHcR = (float*)pUpHc;

            _readbackHcR = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
            void* pRbHc = null;
            _readbackHcR.Map(0, null, &pRbHc);
            _pReadbackHcR = (float*)pRbHc;
        }

        if (_qsaIndexers != null)
        {
            _selectedIdsCapacity = Math.Max(64, _maxSeqLen);
            ulong bytes = (ulong)_selectedIdsCapacity * sizeof(uint);
            _dSelectedIds = _ctx.CreateDeviceBuffer(bytes);
            _uploadSelectedIds = _ctx.CreateUploadBuffer(bytes);
            void* pSel = null;
            _uploadSelectedIds.Map(0, null, &pSel);
            _pUploadSelectedIds = (uint*)pSel;

            if (_readbackNormX == null)
            {
                _readbackNormX = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
                void* pRbNorm = null;
                _readbackNormX.Map(0, null, &pRbNorm);
                _pReadbackNormX = (float*)pRbNorm;
            }
        }

        _lastHidden = (float*)NativeMemory.AllocZeroed((nuint)(_dim * sizeof(float)));
        _readbackHidden = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
        void* pRbHid = null;
        _readbackHidden.Map(0, null, &pRbHid);
        _pReadbackHidden = (float*)pRbHid;

        if (_hasHc)
        {
            _lastHcHidden = (float*)NativeMemory.AllocZeroed((nuint)(_hcDim * sizeof(float)));
        }

        if (_weights.IsMoe && _tieredMoe)
        {
            int maxTopK = Math.Max(1, _weights.ExpertUsedCount);
            ulong topKBytes = 32UL * (ulong)maxTopK * (sizeof(uint) + sizeof(float));
            _readbackTopK = _ctx.CreateReadbackBuffer(topKBytes);
            void* pRbTopK = null;
            _readbackTopK.Map(0, null, &pRbTopK);
            _pReadbackTopK = (byte*)pRbTopK;
        }
    }

    private void UploadOutputHcWeights()
    {
        if (!IsLastStage || !_hasHc) return;

        if (_weights.OutputHcNormWeight != null)
        {
            _dOutputHcNormWeight = _ctx.CreateDeviceBuffer((ulong)(_hcDim * sizeof(float)));
            _ctx.CopyToDevice(_dOutputHcNormWeight, (IntPtr)_weights.OutputHcNormWeight, (ulong)(_hcDim * sizeof(float)));
        }

        if (_weights.OutputHcDownWeight != null)
        {
            _outputHcDownType = NormalizeHybridGpuType(_weights.OutputHcDownType, _hcDim);
            _dOutputHcDownWeight = UploadHybridTensor(_weights.OutputHcDownType, (IntPtr)_weights.OutputHcDownWeight, _hcLowRank, _hcDim);
        }

        if (_weights.OutputHcUpWeight != null)
        {
            _outputHcUpType = NormalizeHybridGpuType(_weights.OutputHcUpType, _hcLowRank);
            _dOutputHcUpWeight = UploadHybridTensor(_weights.OutputHcUpType, (IntPtr)_weights.OutputHcUpWeight, _hcDim, _hcLowRank);
        }
    }

    private void EnsureQwen4ExpBatchBuffers(int chunkCapacity)
    {
        if (_hasHc)
        {
            _dHcRBatch?.Dispose();
            _dHcXnBatch?.Dispose();
            _dHcLoBatch?.Dispose();
            _dHcUpBatch?.Dispose();
            _dHcInjectBatch?.Dispose();
            if (_uploadHcRBatch != null) { _uploadHcRBatch.Unmap(0); _uploadHcRBatch.Dispose(); }
            if (_readbackHcRBatch != null) { _readbackHcRBatch.Unmap(0); _readbackHcRBatch.Dispose(); }

            _dHcRBatch = _ctx.CreateDeviceBuffer((ulong)(chunkCapacity * _hcDim * sizeof(float)));
            _dHcXnBatch = _ctx.CreateDeviceBuffer((ulong)(chunkCapacity * _hcDim * sizeof(float)));
            _dHcLoBatch = _ctx.CreateDeviceBuffer((ulong)(chunkCapacity * _hcLowRank * sizeof(float)));
            _dHcUpBatch = _ctx.CreateDeviceBuffer((ulong)(chunkCapacity * _hcDim * sizeof(float)));
            _dHcInjectBatch = _ctx.CreateDeviceBuffer((ulong)(chunkCapacity * _hcCount * sizeof(float)));

            _uploadHcRBatch = _ctx.CreateUploadBuffer((ulong)(chunkCapacity * _hcDim * sizeof(float)));
            void* pUp = null;
            _uploadHcRBatch.Map(0, null, &pUp);
            _pUploadHcRBatch = (float*)pUp;

            _readbackHcRBatch = _ctx.CreateReadbackBuffer((ulong)(chunkCapacity * _hcDim * sizeof(float)));
            void* pRb = null;
            _readbackHcRBatch.Map(0, null, &pRb);
            _pReadbackHcRBatch = (float*)pRb;
        }
        else if (_pleProcessor != null)
        {
            if (_uploadHcRBatch != null) { _uploadHcRBatch.Unmap(0); _uploadHcRBatch.Dispose(); }
            if (_readbackHcRBatch != null) { _readbackHcRBatch.Unmap(0); _readbackHcRBatch.Dispose(); }

            _uploadHcRBatch = _ctx.CreateUploadBuffer((ulong)(chunkCapacity * _dim * sizeof(float)));
            void* pUp = null;
            _uploadHcRBatch.Map(0, null, &pUp);
            _pUploadHcRBatch = (float*)pUp;

            _readbackHcRBatch = _ctx.CreateReadbackBuffer((ulong)(chunkCapacity * _dim * sizeof(float)));
            void* pRb = null;
            _readbackHcRBatch.Map(0, null, &pRb);
            _pReadbackHcRBatch = (float*)pRb;
        }

        if (_qsaIndexers != null || (_weights.IsMoe && _tieredMoe))
        {
            if (_readbackNormXBatch != null) { _readbackNormXBatch.Unmap(0); _readbackNormXBatch.Dispose(); }
            _readbackNormXBatch = _ctx.CreateReadbackBuffer((ulong)(chunkCapacity * _dim * sizeof(float)));
            void* pRbNorm = null;
            _readbackNormXBatch.Map(0, null, &pRbNorm);
            _pReadbackNormXBatch = (float*)pRbNorm;

            if (_weights.IsMoe && _tieredMoe && chunkCapacity > 32)
            {
                if (_readbackTopK != null) { _readbackTopK.Unmap(0); _readbackTopK.Dispose(); }
                ulong topKBytes = (ulong)chunkCapacity * (ulong)Math.Max(1, _weights.ExpertUsedCount) * (sizeof(uint) + sizeof(float));
                _readbackTopK = _ctx.CreateReadbackBuffer(topKBytes);
                void* pRbTopK = null;
                _readbackTopK.Map(0, null, &pRbTopK);
                _pReadbackTopK = (byte*)pRbTopK;
            }
        }

        if (_readbackHiddenBatch != null) { _readbackHiddenBatch.Unmap(0); _readbackHiddenBatch.Dispose(); }
        _readbackHiddenBatch = _ctx.CreateReadbackBuffer((ulong)(chunkCapacity * _dim * sizeof(float)));
        void* pRbHid = null;
        _readbackHiddenBatch.Map(0, null, &pRbHid);
        _pReadbackHiddenBatch = (float*)pRbHid;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Compact BF16 / F16 & Split Fused MoE Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private static GgufType NormalizeHybridGpuType(GgufType t, int cols)
    {
        if (t == GgufType.F16) return GgufType.F32;
        if (t == GgufType.BF16) return (cols & 1) == 0 ? GgufType.BF16 : GgufType.F32;
        return t;
    }

    private ID3D12Resource UploadHybridTensor(GgufType type, IntPtr hostPtr, int rows, int cols)
    {
        if (type == GgufType.BF16 && (cols & 1) == 0)
        {
            return D3D12TensorAlign.UploadCompactBf16Tensor(_ctx, hostPtr, rows, cols);
        }
        return D3D12TensorAlign.UploadAlignedTensor(_ctx, type, hostPtr, rows, cols);
    }

    private static ulong GetHybridAlignedBytes(GgufType type, int rows, int cols)
    {
        if (type == GgufType.BF16 && (cols & 1) == 0)
        {
            return (ulong)rows * (ulong)cols * 2UL;
        }
        return D3D12TensorAlign.GetAlignedTensorBytes(type, rows, cols);
    }

    private static ulong AlignHybridTensorToBuffer(GgufType type, byte* srcRaw, byte* dstAligned, int rows, int cols)
    {
        if (type == GgufType.BF16 && (cols & 1) == 0)
        {
            ulong bytes = (ulong)rows * (ulong)cols * 2UL;
            Buffer.MemoryCopy(srcRaw, dstAligned, bytes, bytes);
            return bytes;
        }
        return D3D12TensorAlign.AlignTensorToBuffer(type, srcRaw, dstAligned, rows, cols);
    }

    private void UploadSplitFusedExperts(
        LayerWeights lw,
        int numExperts,
        int expertFfnDim,
        out ID3D12Resource dGateExps,
        out ID3D12Resource dUpExps,
        out GgufType gateGpuType,
        out GgufType upGpuType)
    {
        gateGpuType = NormalizeHybridGpuType(lw.FfnGateUpExpsType, _dim);
        upGpuType = gateGpuType;

        long gateRowBytes = GgufTypes.GetRowBytes(lw.FfnGateUpExpsType, _dim);
        long halfExpertBytes = (long)expertFfnDim * gateRowBytes;
        long fusedSliceBytes = halfExpertBytes * 2L;

        ulong gateSliceAligned = GetHybridAlignedBytes(lw.FfnGateUpExpsType, expertFfnDim, _dim);
        ulong totalGateAligned = gateSliceAligned * (ulong)numExperts;

        byte[] gateAlignedBuf = GC.AllocateUninitializedArray<byte>(checked((int)totalGateAligned));
        byte[] upAlignedBuf = GC.AllocateUninitializedArray<byte>(checked((int)totalGateAligned));

        fixed (byte* pGateDst = gateAlignedBuf, pUpDst = upAlignedBuf)
        {
            for (int e = 0; e < numExperts; e++)
            {
                byte* expertSrc = lw.FfnGateUpExpsWeight + (long)e * fusedSliceBytes;
                AlignHybridTensorToBuffer(lw.FfnGateUpExpsType, expertSrc, pGateDst + (ulong)e * gateSliceAligned, expertFfnDim, _dim);
                AlignHybridTensorToBuffer(lw.FfnGateUpExpsType, expertSrc + halfExpertBytes, pUpDst + (ulong)e * gateSliceAligned, expertFfnDim, _dim);
            }
        }

        dGateExps = _ctx.CreateDeviceBuffer(totalGateAligned);
        _ctx.CopyToDevice(dGateExps, gateAlignedBuf);

        dUpExps = _ctx.CreateDeviceBuffer(totalGateAligned);
        _ctx.CopyToDevice(dUpExps, upAlignedBuf);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Hyper-Connections & QSA Dispatch
    // ═══════════════════════════════════════════════════════════════════════

    private void DispatchHcBroadcastInit(ID3D12GraphicsCommandList cmd, int batchSize, ID3D12Resource xBuf, ID3D12Resource rBuf)
    {
        cmd.SetComputeRootSignature(_sigHcBroadcastInit!);
        cmd.SetPipelineState(_psoHcBroadcastInit!);

        uint* c = stackalloc uint[3];
        c[0] = (uint)batchSize;
        c[1] = (uint)_hcCount;
        c[2] = (uint)_dim;
        cmd.SetComputeRoot32BitConstants(0, 3, (IntPtr)c, 0);
        cmd.SetComputeRootUnorderedAccessView(1, xBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, rBuf.GPUVirtualAddress);

        uint total = (uint)(batchSize * _hcDim);
        cmd.Dispatch((total + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(rBuf);
    }

    private void DispatchHcRead(
        ID3D12GraphicsCommandList cmd,
        int batchSize,
        ID3D12Resource rBuf,
        ID3D12Resource xnBuf,
        ID3D12Resource loBuf,
        ID3D12Resource upBuf,
        ID3D12Resource? injectBuf,
        ID3D12Resource? wNorm,
        GgufType wDownType,
        ID3D12Resource wDown,
        GgufType wUpType,
        ID3D12Resource wUp,
        GgufType wInjectType,
        ID3D12Resource? wInject,
        ID3D12Resource mixedOutBuf)
    {
        float invHc = 1.0f / _hcCount;

        // 1. Per-stream RMSNorm
        cmd.SetComputeRootSignature(_sigHcStreamRmsNorm!);
        cmd.SetPipelineState(_psoHcStreamRmsNorm!);
        uint* normC = stackalloc uint[4];
        normC[0] = (uint)_dim;
        normC[1] = (uint)_hcCount;
        *(float*)&normC[2] = _weights.RmsNormEps;
        normC[3] = wNorm != null ? 1u : 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)normC, 0);
        cmd.SetComputeRootShaderResourceView(1, wNorm?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, rBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, xnBuf.GPUVirtualAddress);
        cmd.Dispatch((uint)(batchSize * _hcCount), 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(xnBuf);

        // 2. Low-rank bottleneck: lo = silu((xn @ wDown) / hc)
        if (batchSize == 1)
            DispatchGemv(cmd, wDownType, loBuf, xnBuf, wDown.GPUVirtualAddress, _hcDim, _hcLowRank);
        else
            DispatchGemmBatch(cmd, wDownType, loBuf, xnBuf, wDown.GPUVirtualAddress, _hcDim, _hcLowRank, batchSize);
        cmd.ResourceBarrierUnorderedAccessView(loBuf);

        cmd.SetComputeRootSignature(_sigHcScaleSilu!);
        cmd.SetPipelineState(_psoHcScaleSilu!);
        uint* siluC = stackalloc uint[2];
        uint totalLo = (uint)(batchSize * _hcLowRank);
        siluC[0] = totalLo;
        *(float*)&siluC[1] = invHc;
        cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)siluC, 0);
        cmd.SetComputeRootUnorderedAccessView(1, loBuf.GPUVirtualAddress);
        cmd.Dispatch((totalLo + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(loBuf);

        // 3. Up projection & gated stream mean -> mixedOutBuf
        if (batchSize == 1)
            DispatchGemv(cmd, wUpType, upBuf, loBuf, wUp.GPUVirtualAddress, _hcLowRank, _hcDim);
        else
            DispatchGemmBatch(cmd, wUpType, upBuf, loBuf, wUp.GPUVirtualAddress, _hcLowRank, _hcDim, batchSize);
        cmd.ResourceBarrierUnorderedAccessView(upBuf);

        cmd.SetComputeRootSignature(_sigHcGateMean!);
        cmd.SetPipelineState(_psoHcGateMean!);
        uint* meanC = stackalloc uint[4];
        meanC[0] = (uint)batchSize;
        meanC[1] = (uint)_hcCount;
        meanC[2] = (uint)_dim;
        *(float*)&meanC[3] = invHc;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)meanC, 0);
        cmd.SetComputeRootUnorderedAccessView(1, xnBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, upBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, mixedOutBuf.GPUVirtualAddress);
        cmd.Dispatch(((uint)(batchSize * _dim) + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(mixedOutBuf);

        // 4. Optional write-back injection logits
        if (wInject != null && injectBuf != null)
        {
            if (batchSize == 1)
                DispatchGemv(cmd, wInjectType, injectBuf, xnBuf, wInject.GPUVirtualAddress, _hcDim, _hcCount);
            else
                DispatchGemmBatch(cmd, wInjectType, injectBuf, xnBuf, wInject.GPUVirtualAddress, _hcDim, _hcCount, batchSize);
            cmd.ResourceBarrierUnorderedAccessView(injectBuf);
        }
    }

    private void DispatchHcWrite(
        ID3D12GraphicsCommandList cmd,
        int batchSize,
        ID3D12Resource rBuf,
        ID3D12Resource subOutBuf,
        ID3D12Resource injectBuf)
    {
        cmd.SetComputeRootSignature(_sigHcWrite!);
        cmd.SetPipelineState(_psoHcWrite!);
        uint* c = stackalloc uint[4];
        c[0] = (uint)batchSize;
        c[1] = (uint)_hcCount;
        c[2] = (uint)_dim;
        *(float*)&c[3] = 1.0f / _hcCount;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.SetComputeRootUnorderedAccessView(1, rBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, subOutBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, injectBuf.GPUVirtualAddress);
        cmd.Dispatch(((uint)(batchSize * _dim) + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(rBuf);
    }

    private void CopyFirstHcStreamToX(ID3D12GraphicsCommandList cmd, int batchSize, ID3D12Resource rBuf, ID3D12Resource xBuf)
    {
        ulong vecBytes = (ulong)_dim * sizeof(float);
        cmd.ResourceBarrierTransition(rBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(xBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        for (int t = 0; t < batchSize; t++)
        {
            cmd.CopyBufferRegion(xBuf, (ulong)t * vecBytes, rBuf, (ulong)(t * _hcDim) * sizeof(float), vecBytes);
        }
        cmd.ResourceBarrierTransition(xBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        cmd.ResourceBarrierTransition(rBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
    }

    private void ApplyOutputHcAndExtractHidden(ID3D12GraphicsCommandList cmd, int batchSize, ID3D12Resource xBuf)
    {
        if (!_hasHc) return;

        ID3D12Resource rBuf = batchSize == 1 ? _dHcR! : _dHcRBatch!;
        if (_dOutputHcDownWeight != null && _dOutputHcUpWeight != null)
        {
            DispatchHcRead(
                cmd,
                batchSize,
                rBuf,
                batchSize == 1 ? _dHcXn! : _dHcXnBatch!,
                batchSize == 1 ? _dHcLo! : _dHcLoBatch!,
                batchSize == 1 ? _dHcUp! : _dHcUpBatch!,
                injectBuf: null,
                _dOutputHcNormWeight,
                _outputHcDownType,
                _dOutputHcDownWeight,
                _outputHcUpType,
                _dOutputHcUpWeight,
                GgufType.F32,
                wInject: null,
                xBuf);
        }
        else
        {
            CopyFirstHcStreamToX(cmd, batchSize, rBuf, xBuf);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PLE Host-Side N-gram Integration
    // ═══════════════════════════════════════════════════════════════════════

    private PleLookupTicket BeginPleTokenOverlap(int token)
    {
        if (StartLayer != 0 || _pleProcessor == null)
            return default;

        return _pleTable != null
            ? _pleTable.IssueToken(token, _pleProcessor.PreviousTokens)
            : default;
    }

    private void ExecutePleLayerOnHostBuffer(
        int modelLayer,
        ref PleLookupTicket ticket,
        float* rHostPtr,
        float* pleEmbPtr)
    {
        if (_pleProcessor == null || _pleSlotMap == null) return;
        int slot = _pleSlotMap[modelLayer];
        if (slot < 0) return;

        if (ticket.IsValid && _pleTable != null)
        {
            _pleTable.Collect(ticket, new Span<float>(pleEmbPtr, _pleEmbDim));
            ticket = default;
        }
        else
        {
            new Span<float>(pleEmbPtr, _pleEmbDim).Clear();
        }

        var layer = _weights.Layers[modelLayer];
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
        _pleProcessor.ExecuteToken(slot, rHostPtr, pleEmbPtr, pleWeights);
    }

    private void DispatchSparseOrFlashAttention(
        ID3D12GraphicsCommandList cmd,
        int stageLayer,
        int pos,
        ulong qGpuVa,
        ulong outGpuVa,
        QsaSparseAttentionIndexer? qsa,
        float* normXHostPtr,
        in QsaIndexerWeights qsaWeights,
        bool syncSelectedIds)
    {
        if (qsa == null)
        {
            DispatchFlashAttentionKernel(cmd, stageLayer, pos, qGpuVa, outGpuVa);
            return;
        }

        qsa.RecordKeyAndSelect(pos, normXHostPtr, in qsaWeights, out int* selectedIds, out int kvCount);
        if (kvCount <= 0)
        {
            // Zero output on empty selection
            DispatchVecAddWeighted(cmd, _dAttnOut, _dAttnOut, 0f, _nHeads * _headDim, accumulate: 0);
            return;
        }

        if (kvCount == pos + 1)
        {
            DispatchFlashAttentionKernel(cmd, stageLayer, pos, qGpuVa, outGpuVa);
            return;
        }

        // Upload selected KV cell indices and execute QSA sparse attention shader
        int copyCount = Math.Min(kvCount, _selectedIdsCapacity);
        for (int i = 0; i < copyCount; i++)
        {
            _pUploadSelectedIds[i] = (uint)selectedIds[i];
        }

        ulong copyBytes = (ulong)copyCount * sizeof(uint);
        cmd.ResourceBarrierTransition(
            _dSelectedIds!,
            syncSelectedIds ? ResourceStates.NonPixelShaderResource : ResourceStates.Common,
            ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_dSelectedIds!, 0, _uploadSelectedIds!, 0, copyBytes);
        cmd.ResourceBarrierTransition(_dSelectedIds!, ResourceStates.CopyDest, ResourceStates.NonPixelShaderResource);

        cmd.SetComputeRootSignature(_sigSparseAttn!);
        cmd.SetPipelineState(_psoSparseAttn!);
        uint* saConsts = stackalloc uint[6];
        saConsts[0] = (uint)_nHeads;
        saConsts[1] = (uint)_headsKv;
        saConsts[2] = (uint)_headDim;
        saConsts[3] = (uint)_maxSeqLen;
        saConsts[4] = (uint)copyCount;
        *(float*)&saConsts[5] = 1.0f / MathF.Sqrt(_headDim);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)saConsts, 0);
        cmd.SetComputeRootShaderResourceView(1, _dSelectedIds!.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, qGpuVa);
        cmd.SetComputeRootUnorderedAccessView(3, _dKeyCache[stageLayer].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _dValCache[stageLayer].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, outGpuVa);
        cmd.Dispatch((uint)_nHeads, 1, 1);
    }

    private void DispatchFlashAttentionKernel(
        ID3D12GraphicsCommandList cmd,
        int stageLayer,
        int pos,
        ulong qGpuVa,
        ulong outGpuVa)
    {
        cmd.SetComputeRootSignature(_sigFlashAttn);
        cmd.SetPipelineState(_psoFlashAttn);
        uint* faConsts = stackalloc uint[6];
        faConsts[0] = (uint)_nHeads;
        faConsts[1] = (uint)_headsKv;
        faConsts[2] = (uint)_headDim;
        faConsts[3] = (uint)_maxSeqLen;
        faConsts[4] = (uint)pos;
        *(float*)&faConsts[5] = 1.0f / MathF.Sqrt(_headDim);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)faConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, qGpuVa);
        cmd.SetComputeRootUnorderedAccessView(2, _dKeyCache[stageLayer].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _dValCache[stageLayer].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, outGpuVa);
        cmd.Dispatch((uint)_nHeads, 1, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Hosted MoE Segmented VRAM Pool & Concurrent CPU+GPU Execution
    // ═══════════════════════════════════════════════════════════════════════

    private static ulong AlignUp256(ulong bytes) =>
        (bytes + SlotSubAlignmentBytes - 1UL) & ~(SlotSubAlignmentBytes - 1UL);

    private void InitHostedMoeSlotPool()
    {
        if (!_tieredMoe || !_weights.IsMoe) return;

        int expertFfnDim = _weights.ExpertFeedForwardLength;
        int numExperts = _weights.ExpertCount;
        int topK = Math.Max(1, _weights.ExpertUsedCount);

        _layerGateAlignedBytes = new ulong[_weights.BlockCount];
        _layerUpAlignedBytes = new ulong[_weights.BlockCount];
        _layerDownAlignedBytes = new ulong[_weights.BlockCount];
        _layerGateGpuType = new GgufType[_weights.BlockCount];
        _layerUpGpuType = new GgufType[_weights.BlockCount];
        _layerDownGpuType = new GgufType[_weights.BlockCount];
        _layerHasFusedExps = new bool[_weights.BlockCount];
        _layerHasExpertBiases = new bool[_weights.BlockCount];

        int totalMoeLayers = 0;
        ulong maxSlotBytes = 0;

        for (int l = 0; l < LayerCount; l++)
        {
            int ml = StartLayer + l;
            var lw = _weights.Layers[ml];
            if (lw == null || !lw.IsMoe) continue;

            totalMoeLayers++;
            bool fused = lw.FfnGateUpExpsWeight != null && lw.FfnGateExpsWeight == null;
            _layerHasFusedExps[ml] = fused;
            _layerHasExpertBiases[ml] = lw.FfnGateExpsBias != null || lw.FfnUpExpsBias != null || lw.FfnDownExpsBias != null;

            GgufType rawGateType = fused ? lw.FfnGateUpExpsType : lw.FfnGateExpsType;
            GgufType rawUpType = fused ? lw.FfnGateUpExpsType : lw.FfnUpExpsType;
            GgufType rawDownType = lw.FfnDownExpsType;

            _layerGateGpuType[ml] = NormalizeHybridGpuType(rawGateType, _dim);
            _layerUpGpuType[ml] = NormalizeHybridGpuType(rawUpType, _dim);
            _layerDownGpuType[ml] = NormalizeHybridGpuType(rawDownType, expertFfnDim);

            ulong gateAligned = AlignUp256(GetHybridAlignedBytes(rawGateType, expertFfnDim, _dim));
            ulong upAligned = AlignUp256(GetHybridAlignedBytes(rawUpType, expertFfnDim, _dim));
            ulong downAligned = AlignUp256(GetHybridAlignedBytes(rawDownType, _dim, expertFfnDim));

            _layerGateAlignedBytes[ml] = gateAligned;
            _layerUpAlignedBytes[ml] = upAligned;
            _layerDownAlignedBytes[ml] = downAligned;

            ulong layerSlot = gateAligned + upAligned + downAligned;
            if (layerSlot > maxSlotBytes) maxSlotBytes = layerSlot;
        }

        _maxAlignedSlotBytes = maxSlotBytes;
        if (totalMoeLayers == 0 || _maxAlignedSlotBytes == 0)
        {
            _hostedSlotCapacity = 0;
            return;
        }

        // Upgrade expert workspaces so topK overflow experts can execute in parallel via Parallel.For
        if (_expertWorkspaces == null || _expertWorkspaces.Length < topK)
        {
            _expertWorkspaces = new Qwen2D3D12Model.ExpertCpuWorkspace[topK];
            for (int i = 0; i < topK; i++)
            {
                _expertWorkspaces[i] = new Qwen2D3D12Model.ExpertCpuWorkspace(expertFfnDim, _dim);
            }
        }

        var snapshot = HardwareRuntimeProfiler.CaptureSnapshot();
        _coScheduler = HardwareRuntimeProfiler.CreateCalibratedCoScheduler(snapshot);
        _routingProfile = new ExpertRoutingProfile(_weights.BlockCount, numExperts);

        VramHeadroomSnapshot vramSnap = _ctx.QueryVramHeadroom(_explicitVramReserveBytes);
        ulong usableVram = vramSnap.SafeAvailableBytes;
        ulong fallbackReserve = _explicitVramReserveBytes ?? (512UL * 1024 * 1024);
        if (usableVram == 0 && _ctx.TotalAccessibleMemory > vramSnap.CurrentUsageBytes + fallbackReserve)
        {
            usableVram = _ctx.TotalAccessibleMemory - vramSnap.CurrentUsageBytes - fallbackReserve;
        }

        ulong headroomAfterBatch = usableVram > HostedBatchReserveBytes ? usableVram - HostedBatchReserveBytes : 0UL;
        var poolSnapshot = new VramHeadroomSnapshot(
            vramSnap.DedicatedVramBytes,
            vramSnap.OsBudgetBytes,
            vramSnap.CurrentUsageBytes,
            vramSnap.SafetyReserveBytes,
            headroomAfterBatch,
            vramSnap.IsAtCliffRisk);

        int maxSlots = checked(totalMoeLayers * numExperts);
        _hostedSlotCapacity = WddmVramGuard.ComputeRecommendedSlotCount(
            poolSnapshot,
            _maxAlignedSlotBytes,
            currentActiveSlots: 0,
            maxSlotCapacity: maxSlots,
            minSlotFloor: 0);

        if (_hostedSlotCapacity == 0)
        {
            GlacierDiagnostics.LogInformation("[Hosted MoE] Insufficient post-trunk VRAM headroom for expert slots; routing experts via CPU HostRamExpertArena.");
            return;
        }

        _slotsPerSegment = Math.Max(1, (int)(HostedSegmentTargetBytes / _maxAlignedSlotBytes));
        int segmentCount = (_hostedSlotCapacity + _slotsPerSegment - 1) / _slotsPerSegment;
        int remainingSlots = _hostedSlotCapacity;
        for (int s = 0; s < segmentCount; s++)
        {
            int segSlots = Math.Min(_slotsPerSegment, remainingSlots);
            _expertSegments.Add(_ctx.CreateDeviceBuffer((ulong)segSlots * _maxAlignedSlotBytes));
            remainingSlots -= segSlots;
        }

        _slotManager = new GpuResidentSlotManager(
            _hostedSlotCapacity,
            (nuint)_maxAlignedSlotBytes,
            allocateVram: false,
            SlotEvictionPolicy.DecayedFrequency,
            _routingProfile);

        _stagingSlotCapacity = Math.Max(topK, Math.Min(32, _hostedSlotCapacity));
        _uploadExpertStaging = _ctx.CreateUploadBuffer((ulong)_stagingSlotCapacity * _maxAlignedSlotBytes);
        void* pStg = null;
        _uploadExpertStaging.Map(0, null, &pStg);
        _pUploadExpertStaging = (byte*)pStg;

        PrewarmHostedMoeSlotsRoundRobin(numExperts, expertFfnDim);

        double residentGiB = ((ulong)_hostedActiveSlots * _maxAlignedSlotBytes) / (1024.0 * 1024 * 1024);
        GlacierDiagnostics.LogInformation(
            $"[Hosted MoE] Initialized {_hostedActiveSlots}/{_hostedSlotCapacity} VRAM expert slots across {_expertSegments.Count} segments ({residentGiB:F2} GiB, slot={_maxAlignedSlotBytes / 1024.0:F0} KB, safeVRAM={headroomAfterBatch / (1024.0 * 1024 * 1024):F2} GiB).");
    }

    private void PrewarmHostedMoeSlotsRoundRobin(int numExperts, int expertFfnDim)
    {
        if (_slotManager == null || _pUploadExpertStaging == null || _hostedSlotCapacity <= 0)
            return;

        int filled = 0;
        var batch = new List<(int SegmentIdx, ulong SegOffset, ulong StagingOffset, ulong CopyBytes)>(_stagingSlotCapacity);

        void FlushPrewarmBatch()
        {
            if (batch.Count == 0) return;

            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;
            Span<bool> segTouched = _expertSegments.Count <= 256 ? stackalloc bool[_expertSegments.Count] : new bool[_expertSegments.Count];

            for (int i = 0; i < batch.Count; i++)
            {
                int seg = batch[i].SegmentIdx;
                if (!segTouched[seg])
                {
                    cmd.ResourceBarrierTransition(_expertSegments[seg], ResourceStates.Common, ResourceStates.CopyDest);
                    segTouched[seg] = true;
                }
                cmd.CopyBufferRegion(_expertSegments[seg], batch[i].SegOffset, _uploadExpertStaging!, batch[i].StagingOffset, batch[i].CopyBytes);
            }

            for (int seg = 0; seg < segTouched.Length; seg++)
            {
                if (segTouched[seg])
                {
                    cmd.ResourceBarrierTransition(_expertSegments[seg], ResourceStates.CopyDest, ResourceStates.Common);
                }
            }

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            batch.Clear();
        }

        for (int exp = 0; exp < numExperts && filled < _hostedSlotCapacity; exp++)
        {
            for (int l = 0; l < LayerCount && filled < _hostedSlotCapacity; l++)
            {
                int ml = StartLayer + l;
                var lw = _weights.Layers[ml];
                if (lw == null || !lw.IsMoe || _layerHasExpertBiases![ml]) continue;

                _slotManager.AllocateOrEvictSlot(ml, exp, ReadOnlySpan<int>.Empty, out int slotIdx, out _, out _, out _);
                int stagingSlot = batch.Count;
                ulong stagingOffset = (ulong)stagingSlot * _maxAlignedSlotBytes;
                ulong slotBytes = StageExpertIntoHostBuffer(lw, ml, exp, expertFfnDim, _pUploadExpertStaging + stagingOffset);

                int segIdx = slotIdx / _slotsPerSegment;
                int localIdx = slotIdx % _slotsPerSegment;
                batch.Add((segIdx, (ulong)localIdx * _maxAlignedSlotBytes, stagingOffset, slotBytes));
                filled++;

                if (batch.Count >= _stagingSlotCapacity)
                {
                    FlushPrewarmBatch();
                }
            }
        }

        FlushPrewarmBatch();
        _hostedActiveSlots = filled;
    }

    private ulong StageExpertIntoHostBuffer(LayerWeights lw, int modelLayer, int expertIdx, int expertFfnDim, byte* dstSlotPtr)
    {
        bool fused = _layerHasFusedExps![modelLayer];
        ulong gateAligned = _layerGateAlignedBytes![modelLayer];
        ulong upAligned = _layerUpAlignedBytes![modelLayer];
        ulong downAligned = _layerDownAlignedBytes![modelLayer];

        byte* srcGate;
        byte* srcUp;
        GgufType rawGateType;
        GgufType rawUpType;

        if (fused)
        {
            rawGateType = lw.FfnGateUpExpsType;
            rawUpType = lw.FfnGateUpExpsType;
            long rowBytes = GgufTypes.GetRowBytes(rawGateType, _dim);
            long halfBytes = (long)expertFfnDim * rowBytes;
            srcGate = lw.FfnGateUpExpsWeight + (long)expertIdx * (halfBytes * 2L);
            srcUp = srcGate + halfBytes;
        }
        else
        {
            rawGateType = lw.FfnGateExpsType;
            rawUpType = lw.FfnUpExpsType;
            long gateBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(rawGateType, _dim);
            long upBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(rawUpType, _dim);
            srcGate = lw.FfnGateExpsWeight + (long)expertIdx * gateBytes;
            srcUp = lw.FfnUpExpsWeight + (long)expertIdx * upBytes;
        }

        long downBytes = (long)_dim * GgufTypes.GetRowBytes(lw.FfnDownExpsType, expertFfnDim);
        byte* srcDown = lw.FfnDownExpsWeight + (long)expertIdx * downBytes;

        AlignHybridTensorToBuffer(rawGateType, srcGate, dstSlotPtr, expertFfnDim, _dim);
        AlignHybridTensorToBuffer(rawUpType, srcUp, dstSlotPtr + gateAligned, expertFfnDim, _dim);
        AlignHybridTensorToBuffer(lw.FfnDownExpsType, srcDown, dstSlotPtr + gateAligned + upAligned, _dim, expertFfnDim);

        return gateAligned + upAligned + downAligned;
    }

    private ulong GetHostedSlotGpuAddress(int slotIdx)
    {
        int segIdx = slotIdx / _slotsPerSegment;
        int localIdx = slotIdx % _slotsPerSegment;
        return _expertSegments[segIdx].GPUVirtualAddress + (ulong)localIdx * _maxAlignedSlotBytes;
    }

    private bool RecordSharedExpertIntoFfnOut(
        ID3D12GraphicsCommandList cmd,
        D3D12LayerWeights lw,
        LayerWeights lwHost,
        int modelLayer,
        int expertFfnDim,
        bool hasShexp,
        bool hasShexpGate)
    {
        if (!hasShexp) return false;

        int shexpFfnDim = ResolveSharedExpertFfnDim(modelLayer, expertFfnDim);
        DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate!, _dMlpNorm, lw.FfnGateShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
        DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp!, _dMlpNorm, lw.FfnUpShexpWeight!.GPUVirtualAddress, _dim, shexpFfnDim);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        DispatchSwiGLU(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut!, _dShexpAct!, lw.FfnDownShexpWeight!.GPUVirtualAddress, shexpFfnDim, _dim);
        cmd.ResourceBarrierUnorderedAccessView(null!);

        if (hasShexpGate)
        {
            if (_dShexpGateScalar == null) _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));
            var dShexpGateW = GetOrCreateShexpGateWeight(modelLayer, lwHost.FfnGateInpShexpWeight);
            _ctx.DispatchMoeSharedExpertGate(cmd, _dMlpNorm, dShexpGateW, _dShexpGateScalar, _dim);
            cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar);
            _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar, 0, _dim, accumulate: 0);
        }
        else
        {
            DispatchVecAddWeighted(cmd, _dExpertDownOut!, _dFfnOut!, 1.0f, _dim, accumulate: 0);
        }
        cmd.ResourceBarrierUnorderedAccessView(_dFfnOut!);
        return true;
    }

    /// <summary>
    /// Executes one token's MoE layer using Hosted MoE (VRAM hits + dynamic PCIe promotions +
    /// concurrent CPU SIMD overflow from <see cref="HostRamExpertArena"/>).
    /// Expects <c>_dMlpNorm</c> populated in active command list, and leaves caller
    /// ready in a newly opened command list with <c>_dFfnOut</c> populated.
    /// </summary>
    private void ExecuteHostedMoeSingleToken(
        D3D12LayerWeights lw,
        int modelLayer,
        int expertFfnDim,
        int numExperts,
        int topK,
        bool hasShexp,
        bool hasShexpGate)
    {
        var lwHost = _weights.Layers[modelLayer];
        var cmd = _ctx.CommandList;

        // 1. Router GEMV + GPU Softmax TopK + Shared Expert inside Phase 1 GPU command list
        DispatchGemvBias(cmd, GgufType.F32, _dRouterLogits!, _dMlpNorm, lw.FfnGateInpWeight!.GPUVirtualAddress, _dim, numExperts, lw.FfnGateInpBias);
        cmd.ResourceBarrierUnorderedAccessView(_dRouterLogits!);

        _dTopKIndices ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(uint)));
        _dTopKWeights ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(float)));
        _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogits!, _dTopKIndices, _dTopKWeights, numExperts, topK, _weights.NormTopK);
        cmd.ResourceBarrierUnorderedAccessView(_dTopKIndices);
        cmd.ResourceBarrierUnorderedAccessView(_dTopKWeights);

        bool ffnOutInitialized = RecordSharedExpertIntoFfnOut(cmd, lw, lwHost, modelLayer, expertFfnDim, hasShexp, hasShexpGate);

        // Copy TopK indices, TopK weights, and normalized input activation back for Phase 2 scheduling
        ulong idxBytes = (ulong)(topK * sizeof(uint));
        ulong wBytes = (ulong)(topK * sizeof(float));
        cmd.ResourceBarrierTransition(_dTopKIndices, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_dTopKWeights, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.UnorderedAccess, ResourceStates.CopySource);

        cmd.CopyBufferRegion(_readbackTopK!, 0, _dTopKIndices, 0, idxBytes);
        cmd.CopyBufferRegion(_readbackTopK!, idxBytes, _dTopKWeights, 0, wBytes);
        cmd.CopyBufferRegion(_readbackNormX!, 0, _dMlpNorm, 0, (ulong)(_dim * sizeof(float)));

        cmd.ResourceBarrierTransition(_dTopKIndices, ResourceStates.CopySource, ResourceStates.Common);
        cmd.ResourceBarrierTransition(_dTopKWeights, ResourceStates.CopySource, ResourceStates.Common);
        cmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.CopySource, ResourceStates.Common);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        int* topIndices = (int*)_pReadbackTopK;
        float* topWeights = (float*)(_pReadbackTopK + idxBytes);
        ExecuteHostedMoeCoScheduledStep(
            lw,
            lwHost,
            modelLayer,
            expertFfnDim,
            topK,
            topIndices,
            topWeights,
            _pReadbackNormX,
            ffnOutInitialized,
            normBatchSourceOffsetBytes: null,
            deferredShexp: false,
            deferredHasShexpGate: false);
    }

    /// <summary>
    /// Executes Hosted MoE across <paramref name="chunkSize"/> tokens whose router TopK indices,
    /// weights, and normalized activations have already been computed into batch buffers.
    /// Leaves a newly opened command list with <c>_dMlpOutBatch</c> populated.
    /// </summary>
    private void ExecuteHostedMoeBatchChunk(
        D3D12LayerWeights lw,
        int modelLayer,
        int expertFfnDim,
        int numExperts,
        int topK,
        int chunkSize,
        bool hasShexp,
        bool hasShexpGate)
    {
        var lwHost = _weights.Layers[modelLayer];
        var cmd = _ctx.CommandList;

        // 1. Batched Router GEMM + Batched GPU Softmax TopK
        DispatchGemmBatch(cmd, GgufType.F32, _dRouterLogitsBatch!, _dMlpNormBatch!, lw.FfnGateInpWeight!.GPUVirtualAddress, _dim, numExperts, chunkSize, lw.FfnGateInpBias);
        cmd.ResourceBarrierUnorderedAccessView(_dRouterLogitsBatch!);

        _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogitsBatch!, _dTopKIndicesBatch!, _dTopKWeightsBatch!, numExperts, topK, _weights.NormTopK, chunkSize);
        cmd.ResourceBarrierUnorderedAccessView(_dTopKIndicesBatch!);
        cmd.ResourceBarrierUnorderedAccessView(_dTopKWeightsBatch!);

        ulong allIdxBytes = (ulong)(chunkSize * topK * sizeof(uint));
        ulong allWBytes = (ulong)(chunkSize * topK * sizeof(float));
        ulong allNormBytes = (ulong)(chunkSize * _dim * sizeof(float));

        cmd.ResourceBarrierTransition(_dTopKIndicesBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_dTopKWeightsBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
        cmd.ResourceBarrierTransition(_dMlpNormBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);

        cmd.CopyBufferRegion(_readbackTopK!, 0, _dTopKIndicesBatch!, 0, allIdxBytes);
        cmd.CopyBufferRegion(_readbackTopK!, allIdxBytes, _dTopKWeightsBatch!, 0, allWBytes);
        cmd.CopyBufferRegion(_readbackNormXBatch!, 0, _dMlpNormBatch!, 0, allNormBytes);

        cmd.ResourceBarrierTransition(_dTopKIndicesBatch!, ResourceStates.CopySource, ResourceStates.Common);
        cmd.ResourceBarrierTransition(_dTopKWeightsBatch!, ResourceStates.CopySource, ResourceStates.Common);
        cmd.ResourceBarrierTransition(_dMlpNormBatch!, ResourceStates.CopySource, ResourceStates.Common);

        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();

        int* allIndices = (int*)_pReadbackTopK;
        float* allWeights = (float*)(_pReadbackTopK + allIdxBytes);
        ulong vecBytes = (ulong)_dim * sizeof(float);

        for (int t = 0; t < chunkSize; t++)
        {
            ulong normSrcOff = (ulong)t * vecBytes;
            ExecuteHostedMoeCoScheduledStep(
                lw,
                lwHost,
                modelLayer,
                expertFfnDim,
                topK,
                allIndices + (long)t * topK,
                allWeights + (long)t * topK,
                _pReadbackNormXBatch + (long)t * _dim,
                ffnOutInitialized: false,
                normBatchSourceOffsetBytes: normSrcOff,
                deferredShexp: hasShexp,
                deferredHasShexpGate: hasShexpGate);

            cmd = _ctx.CommandList;
            cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.ResourceBarrierTransition(_dMlpOutBatch!, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dMlpOutBatch!, normSrcOff, _dFfnOut!, 0, vecBytes);
            cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.CopySource, ResourceStates.Common);

            if (t == chunkSize - 1)
            {
                cmd.ResourceBarrierTransition(_dMlpOutBatch!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
            }
            else
            {
                cmd.ResourceBarrierTransition(_dMlpOutBatch!, ResourceStates.CopyDest, ResourceStates.Common);
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();
            }
        }
    }

    private void ExecuteHostedMoeCoScheduledStep(
        D3D12LayerWeights lw,
        LayerWeights lwHost,
        int modelLayer,
        int expertFfnDim,
        int topK,
        int* topIndices,
        float* topWeights,
        float* normXHostPtr,
        bool ffnOutInitialized,
        ulong? normBatchSourceOffsetBytes,
        bool deferredShexp,
        bool deferredHasShexpGate)
    {
        _routingProfile?.RecordActivations(modelLayer, new ReadOnlySpan<int>(topIndices, topK), new ReadOnlySpan<float>(topWeights, topK));

        Span<int> gpuSlotList = stackalloc int[topK];
        Span<float> gpuWeightList = stackalloc float[topK];
        int gpuHitCount = 0;

        Span<int> protectedExperts = stackalloc int[topK];
        int protectedCount = 0;

        Span<int> missIndices = stackalloc int[topK];
        Span<float> missWeights = stackalloc float[topK];
        int missCount = 0;

        for (int k = 0; k < topK; k++)
        {
            int expIdx = topIndices[k];
            float w = topWeights[k];
            if (_slotManager != null && _hostedSlotCapacity > 0 && _slotManager.TryGetSlot(modelLayer, expIdx, out int slotIdx, out _))
            {
                gpuSlotList[gpuHitCount] = slotIdx;
                gpuWeightList[gpuHitCount] = w;
                gpuHitCount++;
                protectedExperts[protectedCount++] = expIdx;
                Interlocked.Increment(ref _gpuExpertHitCount);
            }
            else
            {
                missIndices[missCount] = expIdx;
                missWeights[missCount] = w;
                missCount++;
            }
        }

        int qStar = 0;
        bool canPromote = _slotManager != null && _hostedSlotCapacity > 0 && !_layerHasExpertBiases![modelLayer] && _pUploadExpertStaging != null;
        if (canPromote && missCount > 0)
        {
            int optimalQ = _coScheduler!.CalculateOptimalGpuExpertCount(missCount, gpuHitCount);
            int availSlots = _slotManager!.GetAvailableTransferSlots(modelLayer, protectedExperts.Slice(0, protectedCount));
            qStar = Math.Min(optimalQ, Math.Min(availSlots, Math.Min(missCount, _stagingSlotCapacity)));
        }

        Span<int> xferSlots = stackalloc int[qStar];
        Span<ulong> xferSegOffsets = stackalloc ulong[qStar];
        Span<int> xferSegIndices = stackalloc int[qStar];
        Span<ulong> xferCopyBytes = stackalloc ulong[qStar];
        Span<float> xferWeights = stackalloc float[qStar];

        var pcieSw = qStar > 0 ? Stopwatch.StartNew() : null;
        for (int m = 0; m < qStar; m++)
        {
            int expIdx = missIndices[m];
            _slotManager!.AllocateOrEvictSlot(
                modelLayer,
                expIdx,
                protectedExperts.Slice(0, protectedCount),
                out int dstSlot,
                out _,
                out _,
                out _);
            protectedExperts[protectedCount++] = expIdx;

            ulong stagingOff = (ulong)m * _maxAlignedSlotBytes;
            ulong slotBytes = StageExpertIntoHostBuffer(lwHost, modelLayer, expIdx, expertFfnDim, _pUploadExpertStaging + stagingOff);

            xferSlots[m] = dstSlot;
            xferSegIndices[m] = dstSlot / _slotsPerSegment;
            xferSegOffsets[m] = (ulong)(dstSlot % _slotsPerSegment) * _maxAlignedSlotBytes;
            xferCopyBytes[m] = slotBytes;
            xferWeights[m] = missWeights[m];
            _hostedActiveSlots = Math.Max(_hostedActiveSlots, _slotManager.OccupiedSlotCount);
            Interlocked.Increment(ref _gpuExpertTransferCount);
        }
        pcieSw?.Stop();

        int cpuCount = missCount - qStar;
        Interlocked.Add(ref _cpuExpertOverflowCount, cpuCount);

        bool normCopied = false;
        void EnsureMlpNormForToken(ID3D12GraphicsCommandList activeCmd)
        {
            if (normCopied || !normBatchSourceOffsetBytes.HasValue) return;
            activeCmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.Common, ResourceStates.CopyDest);
            activeCmd.CopyBufferRegion(_dMlpNorm, 0, _dMlpNormBatch!, normBatchSourceOffsetBytes.Value, (ulong)(_dim * sizeof(float)));
            activeCmd.ResourceBarrierTransition(_dMlpNorm, ResourceStates.CopyDest, ResourceStates.Common);
            normCopied = true;
        }

        // 3. Submit GPU work asynchronously BEFORE launching CPU SIMD
        int totalGpuExperts = gpuHitCount + qStar;
        bool needGpuCommand = totalGpuExperts > 0;
        Stopwatch? gpuSw = null;
        if (needGpuCommand)
        {
            gpuSw = Stopwatch.StartNew();
            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;

            if (totalGpuExperts > 0 || deferredShexp)
            {
                EnsureMlpNormForToken(cmd);
            }

            if (deferredShexp)
            {
                ffnOutInitialized = RecordSharedExpertIntoFfnOut(cmd, lw, lwHost, modelLayer, expertFfnDim, hasShexp: true, deferredHasShexpGate);
            }

            if (qStar > 0)
            {
                Span<bool> segTouched = _expertSegments.Count <= 256 ? stackalloc bool[_expertSegments.Count] : new bool[_expertSegments.Count];
                for (int m = 0; m < qStar; m++)
                {
                    int seg = xferSegIndices[m];
                    if (!segTouched[seg])
                    {
                        cmd.ResourceBarrierTransition(_expertSegments[seg], ResourceStates.Common, ResourceStates.CopyDest);
                        segTouched[seg] = true;
                    }
                    cmd.CopyBufferRegion(
                        _expertSegments[seg],
                        xferSegOffsets[m],
                        _uploadExpertStaging!,
                        (ulong)m * _maxAlignedSlotBytes,
                        xferCopyBytes[m]);
                }
                for (int s = 0; s < segTouched.Length; s++)
                {
                    if (segTouched[s])
                    {
                        cmd.ResourceBarrierTransition(_expertSegments[s], ResourceStates.CopyDest, ResourceStates.Common);
                    }
                }
            }

            GgufType gateType = _layerGateGpuType![modelLayer];
            GgufType upType = _layerUpGpuType![modelLayer];
            GgufType downType = _layerDownGpuType![modelLayer];
            ulong gateAligned = _layerGateAlignedBytes![modelLayer];
            ulong upAligned = _layerUpAlignedBytes![modelLayer];

            void RecordSlotGemv(int slotIdx, float weight)
            {
                ulong slotVa = GetHostedSlotGpuAddress(slotIdx);
                DispatchGemv(cmd, gateType, _dExpertGate!, _dMlpNorm, slotVa, _dim, expertFfnDim);
                DispatchGemv(cmd, upType, _dExpertUp!, _dMlpNorm, slotVa + gateAligned, _dim, expertFfnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchSwiGLU(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchGemv(cmd, downType, _dExpertDownOut!, _dExpertAct!, slotVa + gateAligned + upAligned, expertFfnDim, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                DispatchVecAddWeighted(cmd, _dExpertDownOut!, _dFfnOut!, weight, _dim, ffnOutInitialized ? 1u : 0u);
                cmd.ResourceBarrierUnorderedAccessView(_dFfnOut!);
                ffnOutInitialized = true;
            }

            for (int h = 0; h < gpuHitCount; h++)
            {
                RecordSlotGemv(gpuSlotList[h], gpuWeightList[h]);
            }
            for (int m = 0; m < qStar; m++)
            {
                RecordSlotGemv(xferSlots[m], xferWeights[m]);
            }

            _ctx.EndCommandsAndExecute();
        }

        // 4. Concurrently execute remaining overflow experts on CPU SIMD
        Stopwatch? cpuSw = null;
        if (cpuCount > 0)
        {
            cpuSw = Stopwatch.StartNew();
            ExecuteCpuOverflowExperts(lwHost, modelLayer, expertFfnDim, normXHostPtr, missIndices.Slice(qStar, cpuCount), missWeights.Slice(qStar, cpuCount));
            cpuSw.Stop();
        }

        // 5. Join GPU & merge CPU output into _dFfnOut, leaving a newly opened command list
        if (gpuSw != null)
        {
            _ctx.Synchronize();
            gpuSw.Stop();

            long singleSlotBytes = checked((long)_maxAlignedSlotBytes);
            if (qStar > 0 && pcieSw != null)
            {
                double xferMs = Math.Max(0.01, pcieSw.Elapsed.TotalMilliseconds);
                _coScheduler?.RecordPcieTransfer(qStar * singleSlotBytes, xferMs);
            }
            _coScheduler?.RecordGpuExecution(totalGpuExperts * singleSlotBytes, Math.Max(0.01, gpuSw.Elapsed.TotalMilliseconds));
        }

        if (cpuCount > 0 && cpuSw != null)
        {
            _coScheduler?.RecordCpuExecution(cpuCount * (long)_maxAlignedSlotBytes, Math.Max(0.01, cpuSw.Elapsed.TotalMilliseconds));
        }

        _ctx.BeginCommands();
        var finalCmd = _ctx.CommandList;

        if (!needGpuCommand && deferredShexp)
        {
            EnsureMlpNormForToken(finalCmd);
            ffnOutInitialized = RecordSharedExpertIntoFfnOut(finalCmd, lw, lwHost, modelLayer, expertFfnDim, hasShexp: true, deferredHasShexpGate);
        }

        if (cpuCount > 0)
        {
            ulong vecBytes = (ulong)_dim * sizeof(float);
            finalCmd.ResourceBarrierTransition(_dExpertDownOut!, ResourceStates.Common, ResourceStates.CopyDest);
            finalCmd.CopyBufferRegion(_dExpertDownOut!, 0, _uploadExpertOut!, 0, vecBytes);
            finalCmd.ResourceBarrierTransition(_dExpertDownOut!, ResourceStates.CopyDest, ResourceStates.Common);

            DispatchVecAddWeighted(finalCmd, _dExpertDownOut!, _dFfnOut!, 1.0f, _dim, ffnOutInitialized ? 1u : 0u);
            finalCmd.ResourceBarrierUnorderedAccessView(_dFfnOut!);
        }
        else if (!ffnOutInitialized)
        {
            DispatchVecAddWeighted(finalCmd, _dFfnOut!, _dFfnOut!, 0f, _dim, accumulate: 0);
            finalCmd.ResourceBarrierUnorderedAccessView(_dFfnOut!);
        }
    }

    private void ExecuteCpuOverflowExperts(
        LayerWeights lwHost,
        int modelLayer,
        int expertFfnDim,
        float* normXHostPtr,
        ReadOnlySpan<int> cpuIndices,
        ReadOnlySpan<float> cpuWeights)
    {
        int count = cpuIndices.Length;
        if (count == 0) return;

        bool fused = _layerHasFusedExps![modelLayer];
        GgufType gateType = fused ? lwHost.FfnGateUpExpsType : lwHost.FfnGateExpsType;
        GgufType upType = fused ? lwHost.FfnGateUpExpsType : lwHost.FfnUpExpsType;
        GgufType downType = lwHost.FfnDownExpsType;

        long gateRowBytes = GgufTypes.GetRowBytes(gateType, _dim);
        long halfFusedBytes = (long)expertFfnDim * gateRowBytes;
        long fusedSliceBytes = halfFusedBytes * 2L;
        long gateSliceBytes = (long)expertFfnDim * gateRowBytes;
        long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(upType, _dim);
        long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(downType, expertFfnDim);

        fixed (float* pNormXSums = _normXSums)
        {
            QuantKernels.ComputeBlockSums32(normXHostPtr, pNormXSums, _dim);
            new Span<float>(_pUploadExpertOut, _dim).Clear();

            int[] idxArr = new int[count];
            float[] wtArr = new float[count];
            cpuIndices.CopyTo(idxArr);
            cpuWeights.CopyTo(wtArr);

            nint normPtr = (nint)normXHostPtr;
            nint normSumsPtr = (nint)pNormXSums;
            int dim = _dim;

            void RunOne(int c)
            {
                int expIdx = idxArr[c];
                var ws = _expertWorkspaces![c];
                float* normX = (float*)normPtr;
                float* normSums = (float*)normSumsPtr;

                fixed (float* pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
                {
                    byte* expGate = fused
                        ? lwHost.FfnGateUpExpsWeight + (long)expIdx * fusedSliceBytes
                        : lwHost.FfnGateExpsWeight + (long)expIdx * gateSliceBytes;
                    byte* expUp = fused
                        ? expGate + halfFusedBytes
                        : lwHost.FfnUpExpsWeight + (long)expIdx * upSliceBytes;
                    byte* expDown = lwHost.FfnDownExpsWeight + (long)expIdx * downSliceBytes;

                    QuantKernels.MatVecMul(gateType, expGate, normX, pGate, dim, expertFfnDim, normSums);
                    if (lwHost.FfnGateExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnGateExpsBias + (long)expIdx * expertFfnDim, pGate, expertFfnDim);

                    QuantKernels.MatVecMul(upType, expUp, normX, pUp, dim, expertFfnDim, normSums);
                    if (lwHost.FfnUpExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnUpExpsBias + (long)expIdx * expertFfnDim, pUp, expertFfnDim);

                    QuantKernels.SwiGLU(pGate, pUp, pAct, expertFfnDim);
                    QuantKernels.ComputeBlockSums32(pAct, pActSums, expertFfnDim);

                    QuantKernels.MatVecMul(downType, expDown, pAct, pDownOut, expertFfnDim, dim, pActSums);
                    if (lwHost.FfnDownExpsBias != null)
                        QuantKernels.VecAdd(lwHost.FfnDownExpsBias + (long)expIdx * dim, pDownOut, dim);
                }
            }

            if (count == 1)
            {
                RunOne(0);
            }
            else
            {
                Parallel.For(0, count, RunOne);
            }

            for (int c = 0; c < count; c++)
            {
                fixed (float* pDownOut = _expertWorkspaces![c].DownOut)
                {
                    QuantKernels.VecAddWeighted(pDownOut, _pUploadExpertOut, wtArr[c], _dim);
                }
            }
        }
    }

    private int ResolveSharedExpertFfnDim(int modelLayer, int expertFfnDim)
    {
        if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
        {
            return (int)tShexp.Dimensions[1];
        }
        return _weights.SharedExpertFeedForwardLength > 0
            ? _weights.SharedExpertFeedForwardLength
            : expertFfnDim * 2;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // IHiddenStateSource, State Snapshots, & Speculative Batch Verification
    // ═══════════════════════════════════════════════════════════════════════

    public void EnablePositionHistory()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hiddenHistory == null)
        {
            _hiddenHistory = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxSeqLen * _dim * sizeof(float)));
        }
        if (_hasHc && _hcHiddenHistory == null)
        {
            _hcHiddenHistory = (float*)NativeMemory.AllocZeroed((nuint)((long)_maxSeqLen * _hcDim * sizeof(float)));
        }
    }

    public ReadOnlySpan<float> GetLastHiddenState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<float>(_lastHidden, _dim);
    }

    public bool TryGetHiddenStateAt(int position, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hiddenHistory == null || (uint)position >= (uint)_maxSeqLen || destination.Length < _dim)
            return false;

        new ReadOnlySpan<float>(_hiddenHistory + (long)position * _dim, _dim).CopyTo(destination);
        return true;
    }

    public ReadOnlySpan<float> GetLastMultiStreamHiddenState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hasHc && _lastHcHidden != null)
        {
            return new ReadOnlySpan<float>(_lastHcHidden, _hcDim);
        }
        return GetLastHiddenState();
    }

    public bool TryGetMultiStreamHiddenStateAt(int position, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_hasHc)
        {
            if (_hcHiddenHistory == null || (uint)position >= (uint)_maxSeqLen || destination.Length < _hcDim)
                return false;

            new ReadOnlySpan<float>(_hcHiddenHistory + (long)position * _hcDim, _hcDim).CopyTo(destination);
            return true;
        }
        return TryGetHiddenStateAt(position, destination);
    }

    public void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < _dim)
            throw new ArgumentException($"Destination length {destination.Length} < hidden dim {_dim}.", nameof(destination));

        fixed (float* pDst = destination)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, tokenId, pDst, _dim);
        }
    }

    private void RecordHiddenStatesAfterSingleStep(int pos)
    {
        ulong vecBytes = (ulong)_dim * sizeof(float);
        Buffer.MemoryCopy(_pReadbackHidden, _lastHidden, vecBytes, vecBytes);
        if (_hiddenHistory != null && (uint)pos < (uint)_maxSeqLen)
        {
            Buffer.MemoryCopy(_pReadbackHidden, _hiddenHistory + (long)pos * _dim, vecBytes, vecBytes);
        }

        if (_hasHc && _lastHcHidden != null && _pReadbackHcR != null)
        {
            ulong hcBytes = (ulong)_hcDim * sizeof(float);
            Buffer.MemoryCopy(_pReadbackHcR, _lastHcHidden, hcBytes, hcBytes);
            if (_hcHiddenHistory != null && (uint)pos < (uint)_maxSeqLen)
            {
                Buffer.MemoryCopy(_pReadbackHcR, _hcHiddenHistory + (long)pos * _hcDim, hcBytes, hcBytes);
            }
        }
    }

    private void RecordHiddenStatesAfterBatchChunk(int chunkStartPos, int chunkSize)
    {
        ulong vecBytes = (ulong)_dim * sizeof(float);
        if (_pReadbackHiddenBatch != null && chunkSize > 0)
        {
            Buffer.MemoryCopy(_pReadbackHiddenBatch + (long)(chunkSize - 1) * _dim, _lastHidden, vecBytes, vecBytes);
            if (_hiddenHistory != null)
            {
                for (int t = 0; t < chunkSize; t++)
                {
                    int pos = chunkStartPos + t;
                    if ((uint)pos < (uint)_maxSeqLen)
                    {
                        Buffer.MemoryCopy(_pReadbackHiddenBatch + (long)t * _dim, _hiddenHistory + (long)pos * _dim, vecBytes, vecBytes);
                    }
                }
            }
        }

        if (_hasHc && _lastHcHidden != null && _pReadbackHcRBatch != null && chunkSize > 0)
        {
            ulong hcBytes = (ulong)_hcDim * sizeof(float);
            Buffer.MemoryCopy(_pReadbackHcRBatch + (long)(chunkSize - 1) * _hcDim, _lastHcHidden, hcBytes, hcBytes);
            if (_hcHiddenHistory != null)
            {
                for (int t = 0; t < chunkSize; t++)
                {
                    int pos = chunkStartPos + t;
                    if ((uint)pos < (uint)_maxSeqLen)
                    {
                        Buffer.MemoryCopy(_pReadbackHcRBatch + (long)t * _hcDim, _hcHiddenHistory + (long)pos * _hcDim, hcBytes, hcBytes);
                    }
                }
            }
        }
    }

    private void EnsureSsmSnapshotStagingBuffers()
    {
        if (_ssmSnapshotBytes <= 0) return;

        ulong bytes = (ulong)_ssmSnapshotBytes;
        if (_readbackSsmSnapshot == null)
        {
            _readbackSsmSnapshot = _ctx.CreateReadbackBuffer(bytes);
            void* pRb = null;
            _readbackSsmSnapshot.Map(0, null, &pRb);
            _pReadbackSsmSnapshot = (byte*)pRb;
        }
        if (_uploadSsmSnapshot == null)
        {
            _uploadSsmSnapshot = _ctx.CreateUploadBuffer(bytes);
            void* pUp = null;
            _uploadSsmSnapshot.Map(0, null, &pUp);
            _pUploadSsmSnapshot = (byte*)pUp;
        }
    }

    private void EnsureStepSsmSnapshotBuffer(int stepCapacity)
    {
        if (_ssmSnapshotBytes <= 0 || stepCapacity <= _stepSnapshotCapacity) return;

        if (_readbackStepSsmSnapshot != null)
        {
            _readbackStepSsmSnapshot.Unmap(0);
            _readbackStepSsmSnapshot.Dispose();
        }
        ulong totalBytes = (ulong)stepCapacity * (ulong)_ssmSnapshotBytes;
        _readbackStepSsmSnapshot = _ctx.CreateReadbackBuffer(totalBytes);
        void* pStep = null;
        _readbackStepSsmSnapshot.Map(0, null, &pStep);
        _pReadbackStepSsmSnapshot = (byte*)pStep;
        _stepSnapshotCapacity = stepCapacity;
    }

    public void SaveRecurrentState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long required = StateSnapshotBytes;
        if (required == 0) return;
        if (destination.Length < required)
            throw new ArgumentException($"Destination snapshot buffer too small ({destination.Length} < {required}).", nameof(destination));

        int offset = 0;
        if (_ssmSnapshotBytes > 0)
        {
            EnsureSsmSnapshotStagingBuffers();
            ulong convBytes = (ulong)_gdnConvChannels * 3UL * sizeof(float);
            ulong ssmBytes = (ulong)_ssmHeads * (ulong)_ssmStateDim * (ulong)_ssmStateDim * sizeof(float);

            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;
            for (int l = 0; l < LayerCount; l++)
            {
                long layerOff = _ssmLayerSnapshotOffsets![l];
                if (layerOff < 0) continue;

                cmd.ResourceBarrierTransition(_dConvState[l], ResourceStates.Common, ResourceStates.CopySource);
                cmd.ResourceBarrierTransition(_dSsmState[l], ResourceStates.Common, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackSsmSnapshot!, (ulong)layerOff, _dConvState[l], 0, convBytes);
                cmd.CopyBufferRegion(_readbackSsmSnapshot!, (ulong)layerOff + convBytes, _dSsmState[l], 0, ssmBytes);
                cmd.ResourceBarrierTransition(_dConvState[l], ResourceStates.CopySource, ResourceStates.Common);
                cmd.ResourceBarrierTransition(_dSsmState[l], ResourceStates.CopySource, ResourceStates.Common);
            }
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            int ssmLen = checked((int)_ssmSnapshotBytes);
            new ReadOnlySpan<byte>(_pReadbackSsmSnapshot, ssmLen).CopyTo(destination.Slice(0, ssmLen));
            offset += ssmLen;
        }

        if (_pleProcessor != null)
        {
            int pleLen = checked((int)_pleProcessor.StateSnapshotBytes);
            _pleProcessor.SaveState(destination.Slice(offset, pleLen));
            offset += pleLen;
        }

        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                var qsa = _qsaIndexers[i];
                if (qsa == null) continue;
                int qsaLen = checked((int)qsa.StateSnapshotBytes);
                qsa.SaveState(destination.Slice(offset, qsaLen));
                offset += qsaLen;
            }
        }
    }

    public void RestoreRecurrentState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long required = StateSnapshotBytes;
        if (required == 0) return;
        if (source.Length < required)
            throw new ArgumentException($"Source snapshot buffer too small ({source.Length} < {required}).", nameof(source));

        int offset = 0;
        if (_ssmSnapshotBytes > 0)
        {
            EnsureSsmSnapshotStagingBuffers();
            int ssmLen = checked((int)_ssmSnapshotBytes);
            source.Slice(0, ssmLen).CopyTo(new Span<byte>(_pUploadSsmSnapshot, ssmLen));

            ulong convBytes = (ulong)_gdnConvChannels * 3UL * sizeof(float);
            ulong ssmBytes = (ulong)_ssmHeads * (ulong)_ssmStateDim * (ulong)_ssmStateDim * sizeof(float);

            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;
            for (int l = 0; l < LayerCount; l++)
            {
                long layerOff = _ssmLayerSnapshotOffsets![l];
                if (layerOff < 0) continue;

                cmd.ResourceBarrierTransition(_dConvState[l], ResourceStates.Common, ResourceStates.CopyDest);
                cmd.ResourceBarrierTransition(_dSsmState[l], ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(_dConvState[l], 0, _uploadSsmSnapshot!, (ulong)layerOff, convBytes);
                cmd.CopyBufferRegion(_dSsmState[l], 0, _uploadSsmSnapshot!, (ulong)layerOff + convBytes, ssmBytes);
                cmd.ResourceBarrierTransition(_dConvState[l], ResourceStates.CopyDest, ResourceStates.Common);
                cmd.ResourceBarrierTransition(_dSsmState[l], ResourceStates.CopyDest, ResourceStates.Common);
            }
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            offset += ssmLen;
        }

        if (_pleProcessor != null)
        {
            int pleLen = checked((int)_pleProcessor.StateSnapshotBytes);
            _pleProcessor.RestoreState(source.Slice(offset, pleLen));
            offset += pleLen;
        }

        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++)
            {
                var qsa = _qsaIndexers[i];
                if (qsa == null) continue;
                int qsaLen = checked((int)qsa.StateSnapshotBytes);
                qsa.RestoreState(source.Slice(offset, qsaLen));
                offset += qsaLen;
            }
        }
    }

    private void EnsureVerifyLogitsBuffers(int chunkCap)
    {
        if (!IsLastStage || chunkCap <= _allocatedVerifyLogitsCapacity) return;

        _dLogitsBatch?.Dispose();
        if (_readbackLogitsBatch != null)
        {
            _readbackLogitsBatch.Unmap(0);
            _readbackLogitsBatch.Dispose();
        }

        ulong totalBytes = (ulong)chunkCap * (ulong)_vocabSize * sizeof(float);
        _dLogitsBatch = _ctx.CreateDeviceBuffer(totalBytes);
        _readbackLogitsBatch = _ctx.CreateReadbackBuffer(totalBytes);
        void* pRb = null;
        _readbackLogitsBatch.Map(0, null, &pRb);
        _pReadbackLogitsBatch = (float*)pRb;
        _allocatedVerifyLogitsCapacity = chunkCap;
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

        float[] flatLogits = new float[tokens.Length * _vocabSize];
        VerifyBatchLogits(tokens, startPos, flatLogits, _vocabSize, perStepSnapshots);

        for (int t = 0; t < tokens.Length; t++)
        {
            ReadOnlySpan<float> row = flatLogits.AsSpan(t * _vocabSize, _vocabSize);
            int bestId = 0;
            float bestVal = row[0];
            for (int v = 1; v < row.Length; v++)
            {
                if (row[v] > bestVal)
                {
                    bestVal = row[v];
                    bestId = v;
                }
            }
            predictedTokens[t] = bestId;
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
        if (!IsLastStage)
            throw new InvalidOperationException("VerifyBatchLogits requires IsLastStage == true.");
        if (vocabStride < _vocabSize)
            throw new ArgumentException($"vocabStride ({vocabStride}) must be >= VocabSize ({_vocabSize}).", nameof(vocabStride));
        if (flatLogits.Length < (tokens.Length - 1) * vocabStride + _vocabSize)
            throw new ArgumentException("flatLogits buffer is too small.", nameof(flatLogits));

        // Capture full per-step logits via ForwardBatchStage in verify chunks (max 8 to cap VRAM)
        const int maxVerifyChunk = 8;
        EnsureBatchBuffers(Math.Max(32, maxVerifyChunk));
        EnsureVerifyLogitsBuffers(maxVerifyChunk);
        if (perStepSnapshots != null && perStepSnapshots.Length > 0)
        {
            EnsureStepSsmSnapshotBuffer(maxVerifyChunk);
        }

        _activePerStepSnapshots = perStepSnapshots;
        _activeSnapshotOffset = 0;
        _activeSnapshotLimit = perStepSnapshots?.Length ?? 0;

        try
        {
            int offset = 0;
            while (offset < tokens.Length)
            {
                int chunkSize = Math.Min(maxVerifyChunk, tokens.Length - offset);
                _activeSnapshotOffset = offset;

                ForwardBatchChunkCore(
                    tokens.Slice(offset, chunkSize),
                    startPos + offset,
                    ReadOnlySpan<float>.Empty,
                    Span<float>.Empty,
                    flatLogits.Slice(offset * vocabStride),
                    vocabStride,
                    computeLastLogits: false,
                    computeAllLogits: true);

                offset += chunkSize;
            }
        }
        finally
        {
            _activePerStepSnapshots = null;
            _activeSnapshotOffset = 0;
            _activeSnapshotLimit = 0;
        }
    }

    private void DisposeQwen4Exp()
    {
        foreach (var seg in _expertSegments) seg?.Dispose();
        _expertSegments.Clear();
        _slotManager?.Dispose();

        if (_uploadExpertStaging != null) { _uploadExpertStaging.Unmap(0); _uploadExpertStaging.Dispose(); }
        if (_readbackTopK != null) { _readbackTopK.Unmap(0); _readbackTopK.Dispose(); }

        _dHcR?.Dispose();
        _dHcXn?.Dispose();
        _dHcLo?.Dispose();
        _dHcUp?.Dispose();
        _dHcInject?.Dispose();
        if (_uploadHcR != null) { _uploadHcR.Unmap(0); _uploadHcR.Dispose(); }
        if (_readbackHcR != null) { _readbackHcR.Unmap(0); _readbackHcR.Dispose(); }

        _dHcRBatch?.Dispose();
        _dHcXnBatch?.Dispose();
        _dHcLoBatch?.Dispose();
        _dHcUpBatch?.Dispose();
        _dHcInjectBatch?.Dispose();
        if (_uploadHcRBatch != null) { _uploadHcRBatch.Unmap(0); _uploadHcRBatch.Dispose(); }
        if (_readbackHcRBatch != null) { _readbackHcRBatch.Unmap(0); _readbackHcRBatch.Dispose(); }
        if (_readbackNormXBatch != null) { _readbackNormXBatch.Unmap(0); _readbackNormXBatch.Dispose(); }

        _dOutputHcNormWeight?.Dispose();
        _dOutputHcDownWeight?.Dispose();
        _dOutputHcUpWeight?.Dispose();

        _pleProcessor?.Dispose();
        _pleTable?.Dispose();
        if (_pleEmbBuf != null) { NativeMemory.Free(_pleEmbBuf); _pleEmbBuf = null; }
        if (_pleEmbBatch != null) { NativeMemory.Free(_pleEmbBatch); _pleEmbBatch = null; }

        if (_qsaIndexers != null)
        {
            for (int i = 0; i < _qsaIndexers.Length; i++) _qsaIndexers[i]?.Dispose();
        }
        _dSelectedIds?.Dispose();
        if (_uploadSelectedIds != null) { _uploadSelectedIds.Unmap(0); _uploadSelectedIds.Dispose(); }

        if (_lastHidden != null) { NativeMemory.Free(_lastHidden); _lastHidden = null; }
        if (_hiddenHistory != null) { NativeMemory.Free(_hiddenHistory); _hiddenHistory = null; }
        if (_lastHcHidden != null) { NativeMemory.Free(_lastHcHidden); _lastHcHidden = null; }
        if (_hcHiddenHistory != null) { NativeMemory.Free(_hcHiddenHistory); _hcHiddenHistory = null; }
        if (_readbackHidden != null) { _readbackHidden.Unmap(0); _readbackHidden.Dispose(); }
        if (_readbackHiddenBatch != null) { _readbackHiddenBatch.Unmap(0); _readbackHiddenBatch.Dispose(); }
        _dLogitsBatch?.Dispose();
        if (_readbackLogitsBatch != null) { _readbackLogitsBatch.Unmap(0); _readbackLogitsBatch.Dispose(); }

        if (_uploadSsmSnapshot != null) { _uploadSsmSnapshot.Unmap(0); _uploadSsmSnapshot.Dispose(); }
        if (_readbackSsmSnapshot != null) { _readbackSsmSnapshot.Unmap(0); _readbackSsmSnapshot.Dispose(); }
        if (_readbackStepSsmSnapshot != null) { _readbackStepSsmSnapshot.Unmap(0); _readbackStepSsmSnapshot.Dispose(); }

        _psoHcBroadcastInit?.Dispose();
        _sigHcBroadcastInit?.Dispose();
        _psoHcStreamRmsNorm?.Dispose();
        _sigHcStreamRmsNorm?.Dispose();
        _psoHcScaleSilu?.Dispose();
        _sigHcScaleSilu?.Dispose();
        _psoHcGateMean?.Dispose();
        _sigHcGateMean?.Dispose();
        _psoHcWrite?.Dispose();
        _sigHcWrite?.Dispose();
        _psoSparseAttn?.Dispose();
        _sigSparseAttn?.Dispose();
    }
}
