namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

public sealed unsafe partial class Qwen2D3D12Model
{
    private ID3D12Resource? _dTopKIndices;
    private ID3D12Resource? _dTopKWeights;
    private ID3D12Resource? _dShexpGateScalar;
    private ID3D12Resource? _dTopKIndicesBatch;
    private ID3D12Resource? _dTopKWeightsBatch;
    private readonly Dictionary<int, ID3D12Resource> _dShexpGateWeights = new();

    private ID3D12Resource GetShexpGateWeight(int layerIdx)
    {
        if (!_dShexpGateWeights.TryGetValue(layerIdx, out var dBuf))
            throw new InvalidOperationException($"Shared expert gate weight for layer {layerIdx} was not pre-uploaded.");
        return dBuf;
    }

    /// <summary>
    /// Executes full transformer forward pass directly on Direct3D 12 GPU.
    /// Zero CPU synchronization during layer execution.
    /// </summary>
    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        ForwardStageCore(token, pos, default, default, logits, computeLogits, allowTailSkip: false);
    }

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits)
    {
        ForwardStageCore(token, pos, inputX, outputX, logits, computeLogits, allowTailSkip: false);
    }

    private void ForwardStageCore(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        bool allowTailSkip)
    {
        if ((uint)pos >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Token position {pos} exceeds maximum sequence length {_maxSeqLen}.");

        // 1. Input activation: either extract embedding or copy incoming hidden vector
        bool isCacheHit = false;
        int cachedSlot = -1;
        int missSlot = -1;

        if (StartLayer == 0)
        {
            if (_dEmbdCache != null && _embdTokenToSlot != null && _embdTokenToSlot.TryGetValue(token, out cachedSlot))
            {
                // In-VRAM token embedding lookup: 0 B PCIe copy!
                isCacheHit = true;
                EmbdCacheHitCount++;
            }
            else
            {
                EmbdCacheMissCount++;
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);

                if (_dEmbdCache != null && _embdTokenToSlot != null && _embdCacheTokens != null)
                {
                    missSlot = _embdCacheHead;
                    _embdCacheHead = (_embdCacheHead + 1) % EmbdCacheCapacity;

                    int evictedToken = _embdCacheTokens[missSlot];
                    if (evictedToken >= 0)
                    {
                        _embdTokenToSlot.Remove(evictedToken);
                    }

                    _embdCacheTokens[missSlot] = token;
                    _embdTokenToSlot[token] = missSlot;
                }
            }
        }
        else
        {
            fixed (float* pInput = inputX)
            {
                Buffer.MemoryCopy(pInput, _pUploadEmbedding, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            }
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Record full transformer graph into command list
        long startRec = Stopwatch.GetTimestamp();
        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding / input activation into _dX inside the same command list
        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
        if (isCacheHit)
        {
            // Zero-PCIe In-VRAM copy via CopyBufferRegion from default heap cache to default heap _dX
            cmd.ResourceBarrierTransition(_dEmbdCache!, ResourceStates.Common, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_dX, 0, _dEmbdCache!, (ulong)((long)cachedSlot * _dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dEmbdCache!, ResourceStates.CopySource, ResourceStates.Common);
        }
        else
        {
            cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));

            if (missSlot >= 0 && _dEmbdCache != null)
            {
                // Populate default heap in-VRAM embedding cache directly from upload buffer
                cmd.ResourceBarrierTransition(_dEmbdCache, ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(_dEmbdCache, (ulong)((long)missSlot * _dim * sizeof(float)), _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dEmbdCache, ResourceStates.CopyDest, ResourceStates.Common);
            }
        }
        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        for (int l = 0; l < LayerCount; l++)
        {
            var lw = _layerWeights[l];

            // Attention pre-norm
            DispatchRmsNorm(cmd, _dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Q, K, V projections
            DispatchGemv(cmd, lw.QType, _dQ, _dNormX, lw.QWeight, _dim, qDim, lw.QBias);
            DispatchGemv(cmd, lw.KType, _dK, _dNormX, lw.KWeight, _dim, kvDim, lw.KBias);
            DispatchGemv(cmd, lw.VType, _dV, _dNormX, lw.VWeight, _dim, kvDim, lw.VBias);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Optional QK-Norm (e.g. Qwen3)
            if (lw.AttnQNormWeight != null)
            {
                DispatchRmsNormHeads(cmd, _dQ, lw.AttnQNormWeight, _nHeads, _headDim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
            if (lw.AttnKNormWeight != null)
            {
                DispatchRmsNormHeads(cmd, _dK, lw.AttnKNormWeight, _nHeadsKv, _headDim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }

            // RoPE
            DispatchRope(cmd, _dQ, _dK, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // KV Cache Store
            DispatchKvStore(cmd, l, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // GQA Attention
            DispatchAttention(cmd, l, pos);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Attn Out projection with fused residual addition (_dX += attnOut)
            DispatchGemv(cmd, lw.AttnOutType, null, _dAttnOut, lw.AttnOutWeight, qDim, _dim, null, residual: _dX);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // FFN pre-norm
            DispatchRmsNorm(cmd, _dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (lw.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;

                // 1. Router GEMV: compute logits for all experts into _dRouterLogits
                DispatchGemv(cmd, GgufType.F32, _dRouterLogits!, _dNormX, lw.FfnGateInpWeight!, _dim, numExperts, lw.FfnGateInpBias);
                cmd.ResourceBarrierUnorderedAccessView(_dRouterLogits!);

                _dTopKIndices ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(uint)));
                _dTopKWeights ??= _ctx.CreateDeviceBuffer((ulong)(topK * sizeof(float)));

                int modelLayer = StartLayer + l;
                var layer = _weights.Layers[modelLayer];
                bool hasShexp = (lw.FfnGateShexpWeight != null);
                bool hasShexpGate = hasShexp && layer.FfnGateInpShexpWeight != null;
                if (hasShexpGate && _dShexpGateScalar == null)
                    _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));

                // 2. GPU Softmax and Top-K directly in VRAM (zero host sync)
                _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogits!, _dTopKIndices, _dTopKWeights, numExperts, topK, _weights.NormTopK);
                cmd.ResourceBarrierUnorderedAccessView(_dTopKIndices);
                cmd.ResourceBarrierUnorderedAccessView(_dTopKWeights);

                // 3. Routed Experts Execution: Tiered CPU/PCIe split vs Full-VRAM Indirected GEMV
                if (_tieredMoe)
                {
                    ulong idxBytes = (ulong)(topK * sizeof(uint));
                    ulong wBytes = (ulong)(topK * sizeof(float));
                    ulong vecBytes = (ulong)(_dim * sizeof(float));

                    cmd.ResourceBarrierTransition(_dTopKIndices, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dTopKWeights, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dNormX, ResourceStates.UnorderedAccess, ResourceStates.CopySource);

                    cmd.CopyBufferRegion(_readbackTopK!, 0, _dTopKIndices, 0, idxBytes);
                    cmd.CopyBufferRegion(_readbackTopK!, idxBytes, _dTopKWeights, 0, wBytes);
                    cmd.CopyBufferRegion(_readbackNormX!, 0, _dNormX, 0, vecBytes);

                    cmd.ResourceBarrierTransition(_dTopKIndices, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dTopKWeights, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dNormX, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

                    _ctx.EndCommandsAndExecute();
                    _ctx.Synchronize();

                    int* topIndices = (int*)_pReadbackTopK;
                    float* topWeights = (float*)(_pReadbackTopK + idxBytes);
                    bool gpuInitializedFfnOut = false;
                    bool cpuHasContribution = true;

                    if (TieredMoeBatchHandler != null)
                    {
                        gpuInitializedFfnOut = TieredMoeBatchHandler(
                            modelLayer,
                            1,
                            topK,
                            allowTailSkip,
                            topIndices,
                            topWeights,
                            _pReadbackNormX,
                            _pUploadExpertOut,
                            _dNormX,
                            _dNormXBatch,
                            _dExpertGate!,
                            _dExpertUp!,
                            _dExpertAct!,
                            _dExpertDownOut!,
                            _dFfnOut!,
                            _dFfnOutBatch!,
                            out cpuHasContribution);
                    }
                    else
                    {
                        ComputeTieredRoutedExpertsCpu(l, modelLayer, expertFfnDim, topK, topIndices, topWeights);
                    }

                    _ctx.BeginCommands();
                    cmd = _ctx.CommandList;

                    if (!gpuInitializedFfnOut)
                    {
                        cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dFfnOut!, 0, _uploadExpertOut!, 0, vecBytes);
                        cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                    }
                    else if (cpuHasContribution)
                    {
                        cmd.ResourceBarrierTransition(_dExpertDownOut!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dExpertDownOut!, 0, _uploadExpertOut!, 0, vecBytes);
                        cmd.ResourceBarrierTransition(_dExpertDownOut!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dFfnOut!);
                    }
                }
                else
                {
                    ulong gateSliceBytes = lw.FfnGateExpsWeight != null ? GetTensorSliceBytes(lw.FfnGateExpsType, expertFfnDim, _dim) : 0;
                    ulong upSliceBytes = lw.FfnUpExpsWeight != null ? GetTensorSliceBytes(lw.FfnUpExpsType, expertFfnDim, _dim) : 0;
                    ulong downSliceBytes = lw.FfnDownExpsWeight != null ? GetTensorSliceBytes(lw.FfnDownExpsType, _dim, expertFfnDim) : 0;

                    for (int k = 0; k < topK; k++)
                    {
                        _ctx.DispatchExpertGemv(cmd, lw.FfnGateExpsType, _dExpertGate!, _dNormX, lw.FfnGateExpsWeight!, _dTopKIndices, k, gateSliceBytes, _dim, expertFfnDim);
                        _ctx.DispatchExpertGemv(cmd, lw.FfnUpExpsType, _dExpertUp!, _dNormX, lw.FfnUpExpsWeight!, _dTopKIndices, k, upSliceBytes, _dim, expertFfnDim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        DispatchSwiglu(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        _ctx.DispatchExpertGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut!, _dExpertAct!, lw.FfnDownExpsWeight!, _dTopKIndices, k, downSliceBytes, expertFfnDim, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dTopKWeights, k, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                        cmd.ResourceBarrierUnorderedAccessView(null!);
                    }
                }

                // 4. Shared Expert if present
                if (hasShexp)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        shexpFfnDim = (int)tShexp.Dimensions[1];

                    DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate, _dNormX, lw.FfnGateShexpWeight!, _dim, shexpFfnDim);
                    DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp, _dNormX, lw.FfnUpShexpWeight!, _dim, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchSwiglu(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut, _dShexpAct!, lw.FfnDownShexpWeight!, shexpFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    if (hasShexpGate)
                    {
                        var dShexpGateW = GetShexpGateWeight(modelLayer);
                        _ctx.DispatchMoeSharedExpertGate(cmd, _dNormX, dShexpGateW, _dShexpGateScalar!, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar!);

                        _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar!, 0, _dim, accumulate: 1);
                    }
                    else
                    {
                        DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                    }
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // 5. Residual connection: _dX += _dFfnOut
                DispatchVecAdd(cmd, _dFfnOut!, _dX, _dim);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
            else
            {
                // Dense FFN Gate and Up projections
                DispatchGemv(cmd, lw.FfnGateType, _dGate, _dNormX, lw.FfnGateWeight!, _dim, _ffnDim);
                DispatchGemv(cmd, lw.FfnUpType, _dUp, _dNormX, lw.FfnUpWeight!, _dim, _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // SwiGLU activation
                DispatchSwiglu(cmd, _dGate, _dUp, _dFfnAct, _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // FFN Down projection with fused residual addition (_dX += ffnDown)
                DispatchGemv(cmd, lw.FfnDownType, null, _dFfnAct, lw.FfnDownWeight!, _ffnDim, _dim, null, residual: _dX);
                cmd.ResourceBarrierUnorderedAccessView(null!);
            }
        }

        if (IsLastStage)
        {
            if (computeLogits)
            {
                // Final RMSNorm
                DispatchRmsNorm(cmd, _dX, _dOutNormWeight!, _dNormX, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Output projection (LM Head)
                DispatchGemv(cmd, NormalizeGpuType(_weights.OutType), _dLogits, _dNormX, _dOutWeight!, _dim, _weights.VocabSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                if (!logits.IsEmpty)
                {
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                }
            }
            cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.Common);
        }
        else
        {
            // Intermediate stage: copy _dX to _readbackActivation
            if (!outputX.IsEmpty && _readbackActivation != null)
            {
                cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackActivation, 0, _dX, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.CopySource, ResourceStates.Common);
            }
            else
            {
                cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.Common);
            }
        }
        double recMs = Stopwatch.GetElapsedTime(startRec).TotalMilliseconds;

        long startGpu = Stopwatch.GetTimestamp();
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        double gpuMs = Stopwatch.GetElapsedTime(startGpu).TotalMilliseconds;

        LastTimings = (recMs, gpuMs);

        // 3. Read back logits if last stage, or activation if intermediate stage
        if (IsLastStage)
        {
            if (computeLogits && !logits.IsEmpty)
            {
                fixed (float* pLogits = logits)
                {
                    Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
                }
            }
        }
        else
        {
            if (!outputX.IsEmpty && _pReadbackActivation != null)
            {
                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_pReadbackActivation, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
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

        ulong totalBytes = (ulong)chunkCap * (ulong)_weights.VocabSize * sizeof(float);
        _dLogitsBatch = _ctx.CreateDeviceBuffer(totalBytes);
        _readbackLogitsBatch = _ctx.CreateReadbackBuffer(totalBytes);
        void* pRb = null;
        _readbackLogitsBatch.Map(0, null, &pRb);
        _pReadbackLogitsBatch = (float*)pRb;
        _allocatedVerifyLogitsCapacity = chunkCap;
    }

    /// <summary>
    /// Batched prompt prefill execution on Direct3D 12 GPU.
    /// Pipelined with register-tiled Batched GEMM (weights streamed ONCE per chunk).
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits);
    }

    /// <summary>
    /// Executes batched multi-token speculative verification and writes greedy next-token predictions.
    /// </summary>
    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens, bool allowTailSkip = true)
    {
        if (tokens.IsEmpty) return;
        int vocabSize = _weights.VocabSize;
        if (predictedTokens.Length < tokens.Length)
            throw new ArgumentException($"predictedTokens length {predictedTokens.Length} < tokens length {tokens.Length}.", nameof(predictedTokens));

        float[] flatLogits = new float[tokens.Length * vocabSize];
        VerifyBatchLogits(tokens, startPos, flatLogits, vocabSize, allowTailSkip);

        for (int t = 0; t < tokens.Length; t++)
        {
            ReadOnlySpan<float> row = flatLogits.AsSpan(t * vocabSize, vocabSize);
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

    /// <summary>
    /// Executes batched multi-token speculative verification and writes per-token logits.
    /// </summary>
    public void VerifyBatchLogits(
        ReadOnlySpan<int> tokens,
        int startPos,
        Span<float> flatLogits,
        int vocabStride,
        bool allowTailSkip = true)
    {
        if (tokens.IsEmpty) return;
        if (!IsLastStage)
            throw new InvalidOperationException("VerifyBatchLogits requires IsLastStage == true.");
        int vocabSize = _weights.VocabSize;
        if (vocabStride < vocabSize)
            throw new ArgumentException($"vocabStride ({vocabStride}) must be >= VocabSize ({vocabSize}).", nameof(vocabStride));
        if (flatLogits.Length < (tokens.Length - 1) * vocabStride + vocabSize)
            throw new ArgumentException("flatLogits buffer is too small.", nameof(flatLogits));

        ForwardBatchCore(
            tokens,
            startPos,
            default,
            default,
            Span<float>.Empty,
            flatLogits,
            vocabStride,
            computeLastLogits: false,
            computeAllLogits: true,
            allowTailSkip: allowTailSkip);
    }

    /// <summary>
    /// Executes batched prefill for the layers assigned to this stage.
    /// </summary>
    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        ForwardBatchCore(
            tokens,
            startPos,
            inputXBatch,
            outputXBatch,
            logits,
            Span<float>.Empty,
            0,
            computeLastLogits: computeLogits,
            computeAllLogits: false,
            allowTailSkip: false);
    }

    private void ForwardBatchCore(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        Span<float> allTokensLogits,
        int allTokensLogitsStride,
        bool computeLastLogits,
        bool computeAllLogits,
        bool allowTailSkip)
    {
        int totalTokens = !tokens.IsEmpty ? tokens.Length : (inputXBatch.Length / _dim);
        if (totalTokens == 0) return;
        if (startPos < 0 || startPos + totalTokens > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, $"Batch range [{startPos}, {startPos + totalTokens}) exceeds maximum sequence length {_maxSeqLen}.");

        if (totalTokens == 1)
        {
            Span<float> singleLogits = computeAllLogits
                ? allTokensLogits.Slice(0, _weights.VocabSize)
                : logits;
            ForwardStageCore(
                !tokens.IsEmpty ? tokens[0] : 0,
                startPos,
                inputXBatch,
                outputXBatch,
                singleLogits,
                computeLastLogits || computeAllLogits,
                allowTailSkip);
            return;
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        int topK = _weights.ExpertUsedCount;
        int* selectedIndices = stackalloc int[Math.Max(1, topK)];
        float* selectedWeights = stackalloc float[Math.Max(1, topK)];

        for (int offset = 0; offset < totalTokens;)
        {
            int remaining = totalTokens - offset;
            int chunkSize = Math.Min(MaxBatchChunk, remaining);
            if (!allowTailSkip && chunkSize > 8 && (chunkSize % 8) == 1)
            {
                chunkSize -= 1;
            }
            bool isLastChunk = (offset + chunkSize == totalTokens);
            int chunkStartPos = startPos + offset;

            if (chunkSize == 1)
            {
                Span<float> singleLogits = computeAllLogits
                    ? allTokensLogits.Slice(offset * allTokensLogitsStride, _weights.VocabSize)
                    : logits;
                ForwardStageCore(
                    !tokens.IsEmpty ? tokens[offset] : 0,
                    chunkStartPos,
                    !inputXBatch.IsEmpty ? inputXBatch.Slice(offset * _dim, _dim) : default,
                    !outputXBatch.IsEmpty ? outputXBatch.Slice(offset * _dim, _dim) : default,
                    singleLogits,
                    computeAllLogits || (isLastChunk && computeLastLogits),
                    allowTailSkip);
                offset += 1;
                continue;
            }

            // 1. Vectorized host embedding extraction or intermediate activation copy for the chunk
            if (StartLayer == 0)
            {
                fixed (int* pTokens = tokens)
                {
                    for (int t = 0; t < chunkSize; t++)
                    {
                        QuantKernels.ExtractEmbedding(
                            _weights.EmbdType,
                            _weights.EmbdWeight,
                            pTokens[offset + t],
                            _pUploadEmbeddingBatch + t * _dim,
                            _dim);
                    }
                }
            }
            else
            {
                fixed (float* pInput = inputXBatch)
                {
                    Buffer.MemoryCopy(
                        pInput + (long)offset * _dim,
                        _pUploadEmbeddingBatch,
                        (ulong)(chunkSize * _dim * sizeof(float)),
                        (ulong)(chunkSize * _dim * sizeof(float)));
                }
            }

            long startRec = Stopwatch.GetTimestamp();
            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;

            // Copy entire chunk of embeddings/activations into _dXBatch
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dXBatch, 0, _uploadEmbeddingBatch, 0, (ulong)(chunkSize * _dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

            if (StartLayer == 0 && _dEmbdCache != null && _embdTokenToSlot != null && _embdCacheTokens != null)
            {
                fixed (int* pTokens = tokens)
                {
                    for (int t = 0; t < chunkSize; t++)
                    {
                        int tok = pTokens[offset + t];
                        if (!_embdTokenToSlot.ContainsKey(tok))
                        {
                            int slot = _embdCacheHead;
                            _embdCacheHead = (_embdCacheHead + 1) % EmbdCacheCapacity;
                            int evicted = _embdCacheTokens[slot];
                            if (evicted >= 0) _embdTokenToSlot.Remove(evicted);
                            _embdCacheTokens[slot] = tok;
                            _embdTokenToSlot[tok] = slot;

                            cmd.ResourceBarrierTransition(_dEmbdCache, ResourceStates.Common, ResourceStates.CopyDest);
                            cmd.CopyBufferRegion(
                                _dEmbdCache,
                                (ulong)((long)slot * _dim * sizeof(float)),
                                _uploadEmbeddingBatch,
                                (ulong)((long)t * _dim * sizeof(float)),
                                (ulong)(_dim * sizeof(float)));
                            cmd.ResourceBarrierTransition(_dEmbdCache, ResourceStates.CopyDest, ResourceStates.Common);
                        }
                    }
                }
            }

            // Execute assigned layers across all tokens in the chunk simultaneously
            for (int l = 0; l < LayerCount; l++)
            {
                var lw = _layerWeights[l];

                // 1. Attention pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.AttnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(_dNormXBatch);

                // 2. Batched Q, K, V projections (weights read ONCE into registers)
                DispatchGemmBatch(cmd, lw.QType, _dQBatch, _dNormXBatch, lw.QWeight, _dim, qDim, chunkSize, lw.QBias);
                DispatchGemmBatch(cmd, lw.KType, _dKBatch, _dNormXBatch, lw.KWeight, _dim, kvDim, chunkSize, lw.KBias);
                DispatchGemmBatch(cmd, lw.VType, _dVBatch, _dNormXBatch, lw.VWeight, _dim, kvDim, chunkSize, lw.VBias);
                cmd.ResourceBarrierUnorderedAccessView(_dQBatch);
                cmd.ResourceBarrierUnorderedAccessView(_dKBatch);

                // Optional QK-Norm (e.g. Qwen3 / Qwen3-Coder) across all tokens in the batch
                if (lw.AttnQNormWeight != null)
                {
                    DispatchRmsNormHeads(cmd, _dQBatch, lw.AttnQNormWeight, chunkSize * _nHeads, _headDim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(_dQBatch);
                }
                if (lw.AttnKNormWeight != null)
                {
                    DispatchRmsNormHeads(cmd, _dKBatch, lw.AttnKNormWeight, chunkSize * _nHeadsKv, _headDim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(_dKBatch);
                }

                // 3. Batched RoPE
                DispatchRopeBatch(cmd, _dQBatch, _dKBatch, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(_dKBatch);
                cmd.ResourceBarrierUnorderedAccessView(_dVBatch);

                // 4. Batched KV Cache Store
                DispatchKvStoreBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(_dKeyCache[l]);
                cmd.ResourceBarrierUnorderedAccessView(_dValCache[l]);
                cmd.ResourceBarrierUnorderedAccessView(_dQBatch);

                // 5. Batched Attention GQA
                DispatchAttentionBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(_dAttnOutBatch);

                // 6. Batched Attn Out with fused residual addition (_dXBatch += attnOut)
                DispatchGemmBatch(cmd, lw.AttnOutType, null, _dAttnOutBatch, lw.AttnOutWeight, qDim, _dim, chunkSize, null, residual: _dXBatch);
                cmd.ResourceBarrierUnorderedAccessView(_dXBatch);

                // 7. FFN pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.FfnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(_dNormXBatch);

                if (!lw.IsMoe)
                {
                    // 8. Batched FFN Gate and Up projections
                    DispatchGemmBatch(cmd, lw.FfnGateType, _dGateBatch, _dNormXBatch, lw.FfnGateWeight!, _dim, _ffnDim, chunkSize);
                    DispatchGemmBatch(cmd, lw.FfnUpType, _dUpBatch, _dNormXBatch, lw.FfnUpWeight!, _dim, _ffnDim, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dGateBatch);
                    cmd.ResourceBarrierUnorderedAccessView(_dUpBatch);

                    // 9. SwiGLU activation across all tokens
                    DispatchSwiglu(cmd, _dGateBatch, _dUpBatch, _dFfnActBatch, chunkSize * _ffnDim);
                    cmd.ResourceBarrierUnorderedAccessView(_dFfnActBatch);

                    // 10. Batched FFN Down projection with fused residual addition (_dXBatch += ffnDown)
                    DispatchGemmBatch(cmd, lw.FfnDownType, null, _dFfnActBatch, lw.FfnDownWeight!, _ffnDim, _dim, chunkSize, null, residual: _dXBatch);
                    cmd.ResourceBarrierUnorderedAccessView(_dXBatch);
                }
                else if (_tieredMoe)
                {
                    int expertFfnDim = _weights.ExpertFeedForwardLength;
                    int numExperts = _weights.ExpertCount;
                    int modelLayer = StartLayer + l;
                    var layer = _weights.Layers[modelLayer];
                    bool hasShexp = (lw.FfnGateShexpWeight != null);
                    bool hasShexpGate = hasShexp && layer.FfnGateInpShexpWeight != null;

                    _dTopKIndicesBatch ??= _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * topK * sizeof(uint)));
                    _dTopKWeightsBatch ??= _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * topK * sizeof(float)));
                    if (hasShexpGate && _dShexpGateScalar == null)
                        _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));

                    // Batched router gate projection & Softmax Top-K
                    DispatchGemmBatch(cmd, GgufType.F32, _dRouterLogitsBatch, _dNormXBatch, lw.FfnGateInpWeight!, _dim, numExperts, chunkSize, bias: lw.FfnGateInpBias);
                    cmd.ResourceBarrierUnorderedAccessView(_dRouterLogitsBatch!);

                    _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogitsBatch!, _dTopKIndicesBatch, _dTopKWeightsBatch, numExperts, topK, _weights.NormTopK, batchSize: chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKIndicesBatch);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKWeightsBatch);

                    ulong allIdxBytes = (ulong)(chunkSize * topK * sizeof(uint));
                    ulong allWBytes = (ulong)(chunkSize * topK * sizeof(float));
                    ulong allVecBytes = (ulong)(chunkSize * _dim * sizeof(float));

                    cmd.ResourceBarrierTransition(_dTopKIndicesBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dTopKWeightsBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);

                    cmd.CopyBufferRegion(_readbackTopK!, 0, _dTopKIndicesBatch, 0, allIdxBytes);
                    cmd.CopyBufferRegion(_readbackTopK!, allIdxBytes, _dTopKWeightsBatch, 0, allWBytes);
                    cmd.CopyBufferRegion(_readbackNormXBatch!, 0, _dNormXBatch, 0, allVecBytes);

                    cmd.ResourceBarrierTransition(_dTopKIndicesBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dTopKWeightsBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

                    _ctx.EndCommandsAndExecute();
                    _ctx.Synchronize();

                    int* allIndices = (int*)_pReadbackTopK;
                    float* allWeights = (float*)(_pReadbackTopK + allIdxBytes);
                    bool gpuInitializedFfnOut = false;
                    bool cpuHasContribution = true;

                    if (TieredMoeBatchHandler != null)
                    {
                        gpuInitializedFfnOut = TieredMoeBatchHandler(
                            modelLayer,
                            chunkSize,
                            topK,
                            allowTailSkip,
                            allIndices,
                            allWeights,
                            _pReadbackNormXBatch,
                            _pUploadExpertOutBatch,
                            _dNormX,
                            _dNormXBatch,
                            _dExpertGate!,
                            _dExpertUp!,
                            _dExpertAct!,
                            _dExpertDownOut!,
                            _dFfnOut!,
                            _dFfnOutBatch!,
                            out cpuHasContribution);
                    }
                    else
                    {
                        ComputeTieredRoutedExpertsCpuBatch(
                            modelLayer,
                            expertFfnDim,
                            chunkSize,
                            topK,
                            allIndices,
                            allWeights);
                    }

                    _ctx.BeginCommands();
                    cmd = _ctx.CommandList;

                    if (!gpuInitializedFfnOut)
                    {
                        cmd.ResourceBarrierTransition(_dFfnOutBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dFfnOutBatch!, 0, _uploadExpertOutBatch!, 0, allVecBytes);
                        cmd.ResourceBarrierTransition(_dFfnOutBatch!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                    }
                    else if (cpuHasContribution)
                    {
                        cmd.ResourceBarrierTransition(_dAttnOutBatch, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dAttnOutBatch, 0, _uploadExpertOutBatch!, 0, allVecBytes);
                        cmd.ResourceBarrierTransition(_dAttnOutBatch, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        DispatchVecAdd(cmd, _dAttnOutBatch, _dFfnOutBatch!, chunkSize * _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dFfnOutBatch!);
                    }

                    if (!hasShexp)
                    {
                        DispatchVecAdd(cmd, _dFfnOutBatch!, _dXBatch, chunkSize * _dim);
                        cmd.ResourceBarrierUnorderedAccessView(_dXBatch);
                    }
                    else
                    {
                        int shexpFfnDim = expertFfnDim * 2;
                        if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                            shexpFfnDim = (int)tShexp.Dimensions[1];

                        cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(_dFfnOutBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(_dNormX, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);

                        for (int t = 0; t < chunkSize; t++)
                        {
                            ulong tokenOffset = (ulong)((long)t * _dim * sizeof(float));
                            cmd.CopyBufferRegion(_dNormX, 0, _dNormXBatch, tokenOffset, (ulong)(_dim * sizeof(float)));
                            cmd.CopyBufferRegion(_dFfnOut!, 0, _dFfnOutBatch!, tokenOffset, (ulong)(_dim * sizeof(float)));
                            cmd.CopyBufferRegion(_dX, 0, _dXBatch, tokenOffset, (ulong)(_dim * sizeof(float)));

                            cmd.ResourceBarrierTransition(_dNormX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                            cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                            cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                            DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate, _dNormX, lw.FfnGateShexpWeight!, _dim, shexpFfnDim);
                            DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp, _dNormX, lw.FfnUpShexpWeight!, _dim, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchSwiglu(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut, _dShexpAct!, lw.FfnDownShexpWeight!, shexpFfnDim, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            if (hasShexpGate)
                            {
                                var dShexpGateW = GetShexpGateWeight(modelLayer);
                                _ctx.DispatchMoeSharedExpertGate(cmd, _dNormX, dShexpGateW, _dShexpGateScalar!, _dim);
                                cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar!);

                                _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar!, 0, _dim, accumulate: 1);
                            }
                            else
                            {
                                DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                            }
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchVecAdd(cmd, _dFfnOut!, _dX, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.CopyDest);
                            cmd.CopyBufferRegion(_dXBatch, tokenOffset, _dX, 0, (ulong)(_dim * sizeof(float)));
                            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopyDest, ResourceStates.CopySource);

                            cmd.ResourceBarrierTransition(_dX, ResourceStates.CopySource, ResourceStates.CopyDest);
                            cmd.ResourceBarrierTransition(_dNormX, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                            cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                        }

                        cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dFfnOutBatch!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dNormX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dFfnOut!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);
                    }
                }
                else
                {
                    int expertFfnDim = _weights.ExpertFeedForwardLength;
                    int numExperts = _weights.ExpertCount;
                    int modelLayer = StartLayer + l;
                    var layer = _weights.Layers[modelLayer];
                    bool hasShexp = (lw.FfnGateShexpWeight != null);
                    bool hasShexpGate = hasShexp && layer.FfnGateInpShexpWeight != null;

                    _dTopKIndicesBatch ??= _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * topK * sizeof(uint)));
                    _dTopKWeightsBatch ??= _ctx.CreateDeviceBuffer((ulong)(MaxBatchChunk * topK * sizeof(float)));
                    if (hasShexpGate && _dShexpGateScalar == null)
                        _dShexpGateScalar = _ctx.CreateDeviceBuffer(sizeof(float));

                    // Batched router gate projection: [chunkSize, _dim] x [_dim, numExperts] -> [chunkSize, numExperts]
                    DispatchGemmBatch(cmd, GgufType.F32, _dRouterLogitsBatch, _dNormXBatch, lw.FfnGateInpWeight!, _dim, numExperts, chunkSize, bias: lw.FfnGateInpBias);
                    cmd.ResourceBarrierUnorderedAccessView(_dRouterLogitsBatch!);

                    // Batched Softmax Top-K directly in VRAM across all chunkSize tokens (zero host sync)
                    _ctx.DispatchMoeRouterSoftmaxTopK(cmd, _dRouterLogitsBatch!, _dTopKIndicesBatch, _dTopKWeightsBatch, numExperts, topK, _weights.NormTopK, batchSize: chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKIndicesBatch);
                    cmd.ResourceBarrierUnorderedAccessView(_dTopKWeightsBatch);

                    ulong gateSliceBytes = lw.FfnGateExpsWeight != null ? GetTensorSliceBytes(lw.FfnGateExpsType, expertFfnDim, _dim) : 0;
                    ulong upSliceBytes = lw.FfnUpExpsWeight != null ? GetTensorSliceBytes(lw.FfnUpExpsType, expertFfnDim, _dim) : 0;
                    ulong downSliceBytes = lw.FfnDownExpsWeight != null ? GetTensorSliceBytes(lw.FfnDownExpsType, _dim, expertFfnDim) : 0;

                    cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dNormX, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);

                    for (int t = 0; t < chunkSize; t++)
                    {
                        ulong tokenOffset = (ulong)((long)t * _dim * sizeof(float));

                        // Copy token t's normalized hidden state into _dNormX and residual state into _dX
                        cmd.CopyBufferRegion(_dNormX, 0, _dNormXBatch, tokenOffset, (ulong)(_dim * sizeof(float)));
                        cmd.CopyBufferRegion(_dX, 0, _dXBatch, tokenOffset, (ulong)(_dim * sizeof(float)));

                        cmd.ResourceBarrierTransition(_dNormX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                        for (int k = 0; k < topK; k++)
                        {
                            int tokenKIndex = t * topK + k;
                            _ctx.DispatchExpertGemv(cmd, lw.FfnGateExpsType, _dExpertGate!, _dNormX, lw.FfnGateExpsWeight!, _dTopKIndicesBatch, tokenKIndex, gateSliceBytes, _dim, expertFfnDim);
                            _ctx.DispatchExpertGemv(cmd, lw.FfnUpExpsType, _dExpertUp!, _dNormX, lw.FfnUpExpsWeight!, _dTopKIndicesBatch, tokenKIndex, upSliceBytes, _dim, expertFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchSwiglu(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            _ctx.DispatchExpertGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut!, _dExpertAct!, lw.FfnDownExpsWeight!, _dTopKIndicesBatch, tokenKIndex, downSliceBytes, expertFfnDim, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dTopKWeightsBatch, tokenKIndex, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                            cmd.ResourceBarrierUnorderedAccessView(null!);
                        }

                        // Shared Expert if present
                        if (hasShexp)
                        {
                            int shexpFfnDim = expertFfnDim * 2;
                            if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                                shexpFfnDim = (int)tShexp.Dimensions[1];

                            DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate, _dNormX, lw.FfnGateShexpWeight!, _dim, shexpFfnDim);
                            DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp, _dNormX, lw.FfnUpShexpWeight!, _dim, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchSwiglu(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut, _dShexpAct!, lw.FfnDownShexpWeight!, shexpFfnDim, _dim);
                            cmd.ResourceBarrierUnorderedAccessView(null!);

                            if (hasShexpGate)
                            {
                                var dShexpGateW = GetShexpGateWeight(modelLayer);
                                _ctx.DispatchMoeSharedExpertGate(cmd, _dNormX, dShexpGateW, _dShexpGateScalar!, _dim);
                                cmd.ResourceBarrierUnorderedAccessView(_dShexpGateScalar!);

                                _ctx.DispatchVecAddWeightedFromBuffer(cmd, _dFfnOut!, _dExpertDownOut!, _dShexpGateScalar!, 0, _dim, accumulate: 1);
                            }
                            else
                            {
                                DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                            }
                            cmd.ResourceBarrierUnorderedAccessView(null!);
                        }

                        // Residual connection: _dX += _dFfnOut
                        DispatchVecAdd(cmd, _dFfnOut!, _dX, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        // Copy updated _dX back into _dXBatch at tokenOffset
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.CopyDest);
                        cmd.CopyBufferRegion(_dXBatch, tokenOffset, _dX, 0, (ulong)(_dim * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopyDest, ResourceStates.CopySource);

                        // Prepare _dX and _dNormX for next token copy
                        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopySource, ResourceStates.CopyDest);
                        cmd.ResourceBarrierTransition(_dNormX, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
                    }

                    // After token loop: transition back to UnorderedAccess / Common
                    cmd.ResourceBarrierTransition(_dNormXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dNormX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);
                }
            }

            if (IsLastStage)
            {
                if (computeAllLogits)
                {
                    EnsureVerifyLogitsBuffers(chunkSize);
                    DispatchRmsNormBatch(cmd, _dXBatch, _dOutNormWeight!, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(_dNormXBatch);

                    DispatchGemmBatch(cmd, NormalizeGpuType(_weights.OutType), _dLogitsBatch!, _dNormXBatch, _dOutWeight!, _dim, _weights.VocabSize, chunkSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dLogitsBatch!);

                    cmd.ResourceBarrierTransition(_dLogitsBatch!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(_readbackLogitsBatch!, 0, _dLogitsBatch!, 0, (ulong)chunkSize * (ulong)_weights.VocabSize * sizeof(float));
                    cmd.ResourceBarrierTransition(_dLogitsBatch!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.Common);
                }
                else if (isLastChunk && computeLastLogits)
                {
                    // Copy the final token's hidden state into _dX for LM Head evaluation
                    ulong lastTokenOffset = (ulong)((chunkSize - 1) * _dim * sizeof(float));
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
                    cmd.CopyBufferRegion(_dX, 0, _dXBatch, lastTokenOffset, (ulong)(_dim * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.Common);

                    // Final RMSNorm on final token (_dNormX and _dLogits remain in UnorderedAccess across calls)
                    DispatchRmsNorm(cmd, _dX, _dOutNormWeight!, _dNormX, _dim, _weights.RmsNormEps);
                    cmd.ResourceBarrierUnorderedAccessView(_dNormX);

                    // Output projection (LM Head)
                    DispatchGemv(cmd, NormalizeGpuType(_weights.OutType), _dLogits, _dNormX, _dOutWeight!, _dim, _weights.VocabSize);
                    cmd.ResourceBarrierUnorderedAccessView(_dLogits);

                    if (!logits.IsEmpty)
                    {
                        cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                        cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                        cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
                    }
                    cmd.ResourceBarrierTransition(_dX, ResourceStates.UnorderedAccess, ResourceStates.Common);
                }
                else
                {
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.Common);
                }
            }
            else
            {
                if (!outputXBatch.IsEmpty && _readbackActivationBatch != null)
                {
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(_readbackActivationBatch, 0, _dXBatch, 0, (ulong)(chunkSize * _dim * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopySource, ResourceStates.Common);
                }
                else
                {
                    cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.UnorderedAccess, ResourceStates.Common);
                }
            }

            double recMs = Stopwatch.GetElapsedTime(startRec).TotalMilliseconds;
            long startGpu = Stopwatch.GetTimestamp();
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            double gpuMs = Stopwatch.GetElapsedTime(startGpu).TotalMilliseconds;
            LastTimings = (recMs, gpuMs);

            if (IsLastStage)
            {
                if (computeAllLogits && _pReadbackLogitsBatch != null)
                {
                    int vocabSize = _weights.VocabSize;
                    for (int t = 0; t < chunkSize; t++)
                    {
                        new ReadOnlySpan<float>(_pReadbackLogitsBatch + (long)t * vocabSize, vocabSize)
                            .CopyTo(allTokensLogits.Slice((offset + t) * allTokensLogitsStride, vocabSize));
                    }
                }
                else if (isLastChunk && computeLastLogits && !logits.IsEmpty)
                {
                    fixed (float* pLogits = logits)
                    {
                        Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
                    }
                }
            }
            else
            {
                if (!outputXBatch.IsEmpty && _pReadbackActivationBatch != null)
                {
                    fixed (float* pOut = outputXBatch)
                    {
                        Buffer.MemoryCopy(
                            _pReadbackActivationBatch,
                            pOut + (long)offset * _dim,
                            (ulong)(chunkSize * _dim * sizeof(float)),
                            (ulong)(chunkSize * _dim * sizeof(float)));
                    }
                }
            }

            offset += chunkSize;
        }
    }

    private void ComputeTieredRoutedExpertsCpuBatch(
        int modelLayer,
        int expertFfnDim,
        int chunkSize,
        int topK,
        int* allIndices,
        float* allWeights)
    {
        var lwHost = _weights.Layers[modelLayer];
        var ws = _expertWorkspaces![0];
        fixed (float* pNormXSums = _normXSums, pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
        {
            for (int t = 0; t < chunkSize; t++)
            {
                float* normX = _pReadbackNormXBatch + (long)t * _dim;
                float* outVec = _pUploadExpertOutBatch + (long)t * _dim;
                new Span<float>(outVec, _dim).Clear();
                QuantKernels.ComputeBlockSums32(normX, pNormXSums, _dim);

                int* tokenIndices = allIndices + (long)t * topK;
                float* tokenWeights = allWeights + (long)t * topK;
                for (int k = 0; k < topK; k++)
                {
                    TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                        lwHost,
                        tokenIndices[k],
                        tokenWeights[k],
                        normX,
                        pNormXSums,
                        outVec,
                        _dim,
                        expertFfnDim,
                        pGate,
                        pUp,
                        pAct,
                        pActSums,
                        pDownOut);
                }
            }
        }
    }

    private void ComputeTieredRoutedExpertsCpu(
        int l,
        int modelLayer,
        int expertFfnDim,
        int topK,
        int* selectedIndices,
        float* selectedWeights)
    {
        var lwHost = _weights.Layers[modelLayer];
        fixed (float* pNormXSums = _normXSums)
        {
            QuantKernels.ComputeBlockSums32(_pReadbackNormX, pNormXSums, _dim);
            new Span<float>(_pUploadExpertOut, _dim).Clear();

            var ws = _expertWorkspaces![0];
            fixed (float* pGate = ws.Gate, pUp = ws.Up, pAct = ws.Act, pActSums = ws.ActSums, pDownOut = ws.DownOut)
            {
                for (int k = 0; k < topK; k++)
                {
                    TieredMoEMissSplitter.ExecuteCpuSingleExpert(
                        lwHost,
                        selectedIndices[k],
                        selectedWeights[k],
                        _pReadbackNormX,
                        pNormXSums,
                        _pUploadExpertOut,
                        _dim,
                        expertFfnDim,
                        pGate,
                        pUp,
                        pAct,
                        pActSums,
                        pDownOut);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddVector(float* target, float* bias, int length)
    {
        QuantKernels.VecAdd(bias, target, length);
    }
}
