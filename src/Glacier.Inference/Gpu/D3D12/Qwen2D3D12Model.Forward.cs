namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

public sealed unsafe partial class Qwen2D3D12Model
{
    /// <summary>
    /// Executes full transformer forward pass directly on Direct3D 12 GPU.
    /// Zero CPU synchronization during layer execution.
    /// </summary>
    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        // 1. Extract embedding directly into persistently mapped upload buffer
        QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, _pUploadEmbedding, _dim);

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Record full transformer graph into command list
        var swRec = Stopwatch.StartNew();
        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        // Copy embedding from upload buffer into _dX inside the same command list
        cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_dX, 0, _uploadEmbedding, 0, (ulong)(_dim * sizeof(float)));
        cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        for (int l = 0; l < _weights.BlockCount; l++)
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
            DispatchGemv(cmd, lw.AttnOutType, null, _dAttnOut, lw.AttnOutWeight, _dim, _dim, null, residual: _dX);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // FFN pre-norm
            DispatchRmsNorm(cmd, _dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (lw.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;

                // Router GEMV: compute logits for all experts into _dRouterLogits
                DispatchGemv(cmd, GgufType.F32, _dRouterLogits!, _dNormX, lw.FfnGateInpWeight!, _dim, numExperts, lw.FfnGateInpBias);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Readback router logits to CPU
                cmd.ResourceBarrierTransition(_dRouterLogits!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackRouterLogits!, 0, _dRouterLogits!, 0, (ulong)(numExperts * sizeof(float)));
                cmd.ResourceBarrierTransition(_dRouterLogits!, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

                // Execute up to router readback and wait for logits
                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();

                // Compute Softmax and Top-K on CPU
                QuantKernels.SoftmaxTopK(_pReadbackRouterLogits, numExperts, topK, selectedIndices, selectedWeights);

                // Restart command list for expert dispatches
                _ctx.BeginCommands();
                cmd = _ctx.CommandList;

                ulong gateSliceBytes = GetTensorSliceBytes(lw.FfnGateExpsType, expertFfnDim, _dim);
                ulong upSliceBytes = GetTensorSliceBytes(lw.FfnUpExpsType, expertFfnDim, _dim);
                ulong downSliceBytes = GetTensorSliceBytes(lw.FfnDownExpsType, _dim, expertFfnDim);

                for (int k = 0; k < topK; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    ulong expGateAddr = lw.FfnGateExpsWeight!.GPUVirtualAddress + (ulong)expertIdx * gateSliceBytes;
                    ulong expUpAddr = lw.FfnUpExpsWeight!.GPUVirtualAddress + (ulong)expertIdx * upSliceBytes;
                    ulong expDownAddr = lw.FfnDownExpsWeight!.GPUVirtualAddress + (ulong)expertIdx * downSliceBytes;

                    // Gate: expGateAddr * _dNormX -> _dExpertGate
                    DispatchGemv(cmd, lw.FfnGateExpsType, _dExpertGate, _dNormX, expGateAddr, _dim, expertFfnDim);

                    // Up: expUpAddr * _dNormX -> _dExpertUp
                    DispatchGemv(cmd, lw.FfnUpExpsType, _dExpertUp, _dNormX, expUpAddr, _dim, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // SwiGLU: SiLU(_dExpertGate) * _dExpertUp -> _dExpertAct
                    DispatchSwiglu(cmd, _dExpertGate!, _dExpertUp!, _dExpertAct!, expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // Down: expDownAddr * _dExpertAct -> _dExpertDownOut
                    DispatchGemv(cmd, lw.FfnDownExpsType, _dExpertDownOut, _dExpertAct!, expDownAddr, expertFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    // Weighted accumulate into _dFfnOut
                    DispatchVecAddWeighted(cmd, _dExpertDownOut!, _dFfnOut!, weight, _dim, accumulate: (uint)(k == 0 ? 0 : 1));
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // Shared Expert if present
                if (lw.FfnGateShexpWeight != null)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{l}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        shexpFfnDim = (int)tShexp.Dimensions[1];

                    DispatchGemv(cmd, lw.FfnGateShexpType, _dShexpGate, _dNormX, lw.FfnGateShexpWeight, _dim, shexpFfnDim);
                    DispatchGemv(cmd, lw.FfnUpShexpType, _dShexpUp, _dNormX, lw.FfnUpShexpWeight!, _dim, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchSwiglu(cmd, _dShexpGate!, _dShexpUp!, _dShexpAct!, shexpFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchGemv(cmd, lw.FfnDownShexpType, _dExpertDownOut, _dShexpAct!, lw.FfnDownShexpWeight!, shexpFfnDim, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    DispatchVecAdd(cmd, _dExpertDownOut!, _dFfnOut!, _dim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                // Residual connection: _dX += _dFfnOut
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

        if (computeLogits)
        {
            // Final RMSNorm
            DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            // Output projection (LM Head)
            DispatchGemv(cmd, _weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            if (!logits.IsEmpty)
            {
                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
            }
        }
        swRec.Stop();

        var swGpu = Stopwatch.StartNew();
        _ctx.EndCommandsAndExecute();
        _ctx.Synchronize();
        swGpu.Stop();

        LastTimings = (swRec.Elapsed.TotalMilliseconds, swGpu.Elapsed.TotalMilliseconds);

        // 3. Read back logits if requested
        if (computeLogits && !logits.IsEmpty)
        {
            fixed (float* pLogits = logits)
            {
                Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
            }
        }
    }

    /// <summary>
    /// Batched prompt prefill execution on Direct3D 12 GPU.
    /// Pipelined with register-tiled Batched GEMM (weights streamed ONCE per chunk).
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        if (tokens.IsEmpty) return;

        if (tokens.Length == 1)
        {
            Forward(tokens[0], startPos, logits, computeLogits);
            return;
        }

        if (_weights.IsMoe)
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                bool isLast = (t == tokens.Length - 1);
                Forward(tokens[t], startPos + t, isLast ? logits : Span<float>.Empty, isLast && computeLogits);
            }
            return;
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        for (int offset = 0; offset < tokens.Length; offset += MaxBatchChunk)
        {
            int chunkSize = Math.Min(MaxBatchChunk, tokens.Length - offset);
            bool isLastChunk = (offset + chunkSize == tokens.Length);
            int chunkStartPos = startPos + offset;

            // 1. Vectorized host embedding extraction for the chunk
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

            var swRec = Stopwatch.StartNew();
            _ctx.BeginCommands();
            var cmd = _ctx.CommandList;

            // Copy entire chunk of embeddings into _dXBatch
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dXBatch, 0, _uploadEmbeddingBatch, 0, (ulong)(chunkSize * _dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dXBatch, ResourceStates.CopyDest, ResourceStates.Common);

            // Execute all layers across all tokens in the chunk simultaneously!
            // Weights for each layer are read from VRAM ONCE per chunk!
            for (int l = 0; l < _weights.BlockCount; l++)
            {
                var lw = _layerWeights[l];

                // Attention pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.AttnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Q, K, V projections (weights read ONCE into registers)
                DispatchGemmBatch(cmd, lw.QType, _dQBatch, _dNormXBatch, lw.QWeight, _dim, qDim, chunkSize, lw.QBias);
                DispatchGemmBatch(cmd, lw.KType, _dKBatch, _dNormXBatch, lw.KWeight, _dim, kvDim, chunkSize, lw.KBias);
                DispatchGemmBatch(cmd, lw.VType, _dVBatch, _dNormXBatch, lw.VWeight, _dim, kvDim, chunkSize, lw.VBias);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched RoPE
                DispatchRopeBatch(cmd, _dQBatch, _dKBatch, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched KV Cache Store
                DispatchKvStoreBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Attention GQA
                DispatchAttentionBatch(cmd, l, chunkStartPos, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched Attn Out with fused residual addition (_dXBatch += attnOut)
                DispatchGemmBatch(cmd, lw.AttnOutType, null, _dAttnOutBatch, lw.AttnOutWeight, _dim, _dim, chunkSize, null, residual: _dXBatch);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // FFN pre-norm across all tokens in parallel
                DispatchRmsNormBatch(cmd, _dXBatch, lw.FfnNormWeight, _dNormXBatch, _dim, chunkSize, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched FFN Gate and Up projections
                DispatchGemmBatch(cmd, lw.FfnGateType, _dGateBatch, _dNormXBatch, lw.FfnGateWeight!, _dim, _ffnDim, chunkSize);
                DispatchGemmBatch(cmd, lw.FfnUpType, _dUpBatch, _dNormXBatch, lw.FfnUpWeight!, _dim, _ffnDim, chunkSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // SwiGLU activation across all tokens
                DispatchSwiglu(cmd, _dGateBatch, _dUpBatch, _dFfnActBatch, chunkSize * _ffnDim);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Batched FFN Down projection with fused residual addition (_dXBatch += ffnDown)
                DispatchGemmBatch(cmd, lw.FfnDownType, null, _dFfnActBatch, lw.FfnDownWeight!, _ffnDim, _dim, chunkSize, null, residual: _dXBatch);
            }

            if (isLastChunk && computeLogits)
            {
                // Copy the final token's hidden state into _dX for LM Head evaluation
                ulong lastTokenOffset = (ulong)((chunkSize - 1) * _dim * sizeof(float));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(_dX, 0, _dXBatch, lastTokenOffset, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dX, ResourceStates.CopyDest, ResourceStates.Common);

                // Final RMSNorm on final token
                DispatchRmsNorm(cmd, _dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                // Output projection (LM Head)
                DispatchGemv(cmd, _weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                if (!logits.IsEmpty)
                {
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                    cmd.CopyBufferRegion(_readbackLogits, 0, _dLogits, 0, (ulong)(_weights.VocabSize * sizeof(float)));
                    cmd.ResourceBarrierTransition(_dLogits, ResourceStates.CopySource, ResourceStates.Common);
                }
            }
            swRec.Stop();
            var swGpu = Stopwatch.StartNew();
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
            swGpu.Stop();
            LastTimings = (swRec.Elapsed.TotalMilliseconds, swGpu.Elapsed.TotalMilliseconds);


            if (isLastChunk && computeLogits && !logits.IsEmpty)
            {
                fixed (float* pLogits = logits)
                {
                    Buffer.MemoryCopy(_pReadbackLogits, pLogits, (ulong)(_weights.VocabSize * sizeof(float)), (ulong)(_weights.VocabSize * sizeof(float)));
                }
            }
        }
    }
}
