namespace Glacier.Inference.Gpu;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

public sealed unsafe partial class Qwen2GpuModel
{

    /// <summary>
    /// Executes transformer forward pass directly on NVIDIA GPU with zero intermediate PCIe round-trips.
    /// RoPE, GQA Attention, RMSNorm, GEMV, SwiGLU, and residual additions all execute in-VRAM.
    /// </summary>
    public void Forward(int token, int pos, KVCache? kvCache = null, Span<float> logits = default, bool computeLogits = true)
    {
        ForwardStage(token, pos, default, default, logits, computeLogits);
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
        // 1. Input activation: either extract embedding or copy incoming hidden vector
        if (StartLayer == 0)
        {
            fixed (float* pX = _hX)
            {
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, pX, _dim);
                _gpu.CopyToDevice(_dX, (IntPtr)pX, (nuint)(_dim * sizeof(float)));
            }
        }
        else
        {
            fixed (float* pX = inputX)
            {
                _gpu.CopyToDevice(_dX, (IntPtr)pX, (nuint)(_dim * sizeof(float)));
            }
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Transformer layers assigned to this stage
        for (int l = 0; l < LayerCount; l++)
        {
            var lw = _layerWeights[l];

            // Attention pre-norm in VRAM
            LaunchRmsNorm(_dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Wait on GPU for RMSNorm output to be ready before K and V read it on their streams
            CuDriver.EventRecord(_eventNormDone, IntPtr.Zero);
            CuDriver.StreamWaitEvent(_streamK, _eventNormDone, 0);
            CuDriver.StreamWaitEvent(_streamV, _eventNormDone, 0);

            // Q, K, V projections on GPU (fused bias addition; K and V launched concurrently on dedicated CUDA streams)
            LaunchGemv(lw.QType, _dQ, _dNormX, lw.QWeight, _dim, qDim, lw.QBias);
            LaunchGemv(lw.KType, _dK, _dNormX, lw.KWeight, _dim, kvDim, lw.KBias, hStream: _streamK);
            LaunchGemv(lw.VType, _dV, _dNormX, lw.VWeight, _dim, kvDim, lw.VBias, hStream: _streamV);

            // Instruct default stream to wait on GPU for K and V completion prior to RoPE and KV cache storage
            CuDriver.EventRecord(_eventKDone, _streamK);
            CuDriver.EventRecord(_eventVDone, _streamV);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventKDone, 0);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventVDone, 0);

            // Optional QK-Norm (Qwen 3 / Qwen 3.8)
            if (lw.AttnQNormWeight != IntPtr.Zero)
            {
                LaunchRmsNormHeads(_dQ, lw.AttnQNormWeight, _nHeads, _headDim, _weights.RmsNormEps);
            }
            if (lw.AttnKNormWeight != IntPtr.Zero)
            {
                LaunchRmsNormHeads(_dK, lw.AttnKNormWeight, _nHeadsKv, _headDim, _weights.RmsNormEps);
            }

            // Rotary Position Embedding (RoPE) directly in VRAM
            LaunchRope(_dQ, _dK, pos);

            // Store in GPU VRAM KV Cache
            LaunchKvStore(l, pos);

            // Multi-Head / Grouped Query Attention (GQA) in VRAM
            LaunchAttention(l, pos);

            // Attention out projection (fused residual accumulation directly into _dX: _dX += attnProj)
            LaunchGemv(lw.AttnOutType, IntPtr.Zero, _dAttnOut, lw.AttnOutWeight, qDim, _dim, IntPtr.Zero, _dX);

            // FFN pre-norm
            LaunchRmsNorm(_dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // SwiGLU FFN projections on GPU (Fused kernel computes Gate + Up in registers and applies SiLU)
            if (lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
            {
                LaunchSwigluFused(_dFfnAct, _dNormX, lw.FfnGateWeight, lw.FfnUpWeight, _dim, _ffnDim);
            }
            else
            {
                LaunchGemv(lw.FfnGateType, _dGate, _dNormX, lw.FfnGateWeight, _dim, _ffnDim);
                LaunchGemv(lw.FfnUpType, _dUp, _dNormX, lw.FfnUpWeight, _dim, _ffnDim);
                LaunchSwiglu(_dGate, _dUp, _dFfnAct, _ffnDim);
            }

            // FFN down projection (fused residual accumulation directly into _dX: _dX += ffnDown)
            LaunchGemv(lw.FfnDownType, IntPtr.Zero, _dFfnAct, lw.FfnDownWeight, _ffnDim, _dim, IntPtr.Zero, _dX);
        }

        if (IsLastStage)
        {
            if (computeLogits)
            {
                // Final RMSNorm
                LaunchRmsNorm(_dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);

                // Output projection (LM Head) on GPU
                LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);

                // Copy logits from GPU to CPU (skipped when logits buffer is empty for pure GPU sampling)
                if (!logits.IsEmpty)
                {
                    fixed (float* pLogits = logits)
                    {
                        _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
                    }
                }
            }
        }
        else
        {
            // Intermediate stage: copy _dX back to host buffer outputX
            if (!outputX.IsEmpty)
            {
                fixed (float* pOut = outputX)
                {
                    _gpu.CopyToHost((IntPtr)pOut, _dX, (nuint)(_dim * sizeof(float)));
                }
            }
        }
    }

    /// <summary>
    /// Executes batched transformer prompt prefill directly on NVIDIA GPU.
    /// Weights are loaded once from VRAM and multiplied across up to MaxBatchSize tokens simultaneously.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits);
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
        int totalTokens = !tokens.IsEmpty ? tokens.Length : (inputXBatch.Length / _dim);
        int offset = 0;
        while (offset < totalTokens)
        {
            int batchSize = Math.Min(totalTokens - offset, MaxBatchSize);
            bool isLastChunk = (offset + batchSize == totalTokens);
            var inSlice = !inputXBatch.IsEmpty ? inputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var outSlice = !outputXBatch.IsEmpty ? outputXBatch.Slice(offset * _dim, batchSize * _dim) : default;
            var tokSlice = !tokens.IsEmpty ? tokens.Slice(offset, batchSize) : default;

            ForwardBatchChunk(
                tokSlice,
                startPos + offset,
                inSlice,
                outSlice,
                isLastChunk && computeLogits ? logits : Span<float>.Empty,
                isLastChunk && computeLogits);
            offset += batchSize;
        }
    }

    /// <summary>
    /// Executes batched speculative verification directly on GPU.
    /// Evaluates all candidate tokens in a single batched transformer pass,
    /// computes LM Head and fast GPU argmax for each position,
    /// and writes the predicted next token for each candidate position.
    /// </summary>
    public void VerifyBatch(ReadOnlySpan<int> tokens, int startPos, Span<int> predictedTokens)
    {
        int batchSize = tokens.Length;
        if (batchSize == 0) return;
        if (batchSize > MaxBatchSize)
            throw new ArgumentException($"Batch size {batchSize} exceeds MaxBatchSize {MaxBatchSize}");

        // Forward through all transformer layers in batch
        ForwardBatchChunk(tokens, startPos, default, default, Span<float>.Empty, computeLogits: false);

        // For each token position in the batch, compute RMSNorm + LM Head + GPU argmax
        for (int t = 0; t < batchSize; t++)
        {
            IntPtr dXt = _dXBatch + t * _dim * sizeof(float);
            LaunchRmsNorm(dXt, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);
            LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);
            predictedTokens[t] = SampleGreedy();
        }
    }

    /// <summary>
    /// Executes fast GPU argmax reduction directly across the 152K vocab logits in ~3 microseconds.
    /// Transfers only 4 bytes (the sampled token ID) across PCIe, bypassing 608 KB host copies.
    /// </summary>
    public int SampleGreedy(ReadOnlySpan<int> recentTokens = default, float repetitionPenalty = 1.0f)
    {
        IntPtr dLogits = _dLogits;
        IntPtr dBestToken = _dBestToken;
        IntPtr dBestLogit = _dBestLogit;
        IntPtr dRecentTokens = _dRecentTokens;

        if (repetitionPenalty != 1.0f && !recentTokens.IsEmpty)
        {
            int count = Math.Min(recentTokens.Length, 1024);
            fixed (int* pRecent = recentTokens)
            {
                _gpu.CopyToDevice(dRecentTokens, (IntPtr)pRecent, (nuint)(count * sizeof(int)));
            }

            void** pPenArgs = stackalloc void*[4];
            pPenArgs[0] = &dLogits;
            pPenArgs[1] = &dRecentTokens;
            pPenArgs[2] = &count;
            pPenArgs[3] = &repetitionPenalty;

            uint blockSize = 64;
            uint gridSize = (uint)((count + (int)blockSize - 1) / (int)blockSize);

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnRepetitionPenalty,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pPenArgs,
                IntPtr.Zero), "LaunchKernel(apply_repetition_penalty)");
        }

        int vocabSize = _weights.VocabSize;
        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dLogits;
        pArgs[1] = &vocabSize;
        pArgs[2] = &dBestToken;
        pArgs[3] = &dBestLogit;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnArgmax,
            1, 1, 1,
            512, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(argmax)");

        int bestToken = 0;
        _gpu.CopyToHost((IntPtr)(&bestToken), dBestToken, (nuint)sizeof(int));
        return bestToken;
    }

    /// <summary>
    /// Samples next token with support for greedy GPU reduction and CPU temperature / top-P sampling.
    /// </summary>
    public int SampleToken(Sampling.SamplingOptions options, ReadOnlySpan<int> recentTokens = default)
    {
        if (options.Temperature <= 0.001f || options.TopK == 1)
        {
            return SampleGreedy(recentTokens, options.RepetitionPenalty);
        }

        _hostLogits ??= new float[_weights.VocabSize];
        fixed (float* pLogits = _hostLogits)
        {
            _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
        }

        return _sampler.Sample(_hostLogits.AsSpan(), options, recentTokens);
    }

    private void ForwardBatchChunk(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        int batchSize = !chunkTokens.IsEmpty ? chunkTokens.Length : (inputXBatch.Length / _dim);
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 1. Extract embeddings into host batch buffer or copy from input activations
        if (StartLayer == 0)
        {
            fixed (float* pXBatch = _hXBatch)
            {
                for (int t = 0; t < batchSize; t++)
                {
                    QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], pXBatch + t * _dim, _dim);
                }
                _gpu.CopyToDevice(_dXBatch, (IntPtr)pXBatch, (nuint)(batchSize * _dim * sizeof(float)));
            }
        }
        else
        {
            fixed (float* pInput = inputXBatch)
            {
                _gpu.CopyToDevice(_dXBatch, (IntPtr)pInput, (nuint)(batchSize * _dim * sizeof(float)));
            }
        }

        // 2. Transformer layers in batch
        for (int l = 0; l < LayerCount; l++)
        {
            var lw = _layerWeights[l];

            LaunchRmsNormBatch(_dXBatch, lw.AttnNormWeight, _dNormXBatch, _dim, _weights.RmsNormEps, batchSize);
            CuDriver.EventRecord(_eventNormDone, IntPtr.Zero);
            CuDriver.StreamWaitEvent(_streamK, _eventNormDone, 0);
            CuDriver.StreamWaitEvent(_streamV, _eventNormDone, 0);

            LaunchGemmBatch(lw.QType, _dQBatch, _dNormXBatch, lw.QWeight, _dim, qDim, batchSize, lw.QBias);
            LaunchGemmBatch(lw.KType, _dKBatch, _dNormXBatch, lw.KWeight, _dim, kvDim, batchSize, lw.KBias, hStream: _streamK);
            LaunchGemmBatch(lw.VType, _dVBatch, _dNormXBatch, lw.VWeight, _dim, kvDim, batchSize, lw.VBias, hStream: _streamV);
            CuDriver.EventRecord(_eventKDone, _streamK);
            CuDriver.EventRecord(_eventVDone, _streamV);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventKDone, 0);
            CuDriver.StreamWaitEvent(IntPtr.Zero, _eventVDone, 0);

            // Optional QK-Norm (Qwen 3 / Qwen 3.8)
            if (lw.AttnQNormWeight != IntPtr.Zero)
            {
                LaunchRmsNormBatchHeads(_dQBatch, lw.AttnQNormWeight, _nHeads, _headDim, batchSize, _weights.RmsNormEps);
            }
            if (lw.AttnKNormWeight != IntPtr.Zero)
            {
                LaunchRmsNormBatchHeads(_dKBatch, lw.AttnKNormWeight, _nHeadsKv, _headDim, batchSize, _weights.RmsNormEps);
            }

            LaunchRopeBatch(_dQBatch, _dKBatch, chunkStartPos, batchSize);
            LaunchKvCacheStoreBatch(l, chunkStartPos, batchSize);

            LaunchAttentionBatch(l, chunkStartPos, batchSize);

            LaunchGemmBatch(lw.AttnOutType, IntPtr.Zero, _dAttnOutBatch, lw.AttnOutWeight, qDim, _dim, batchSize, IntPtr.Zero, _dXBatch);

            LaunchRmsNormBatch(_dXBatch, lw.FfnNormWeight, _dNormXBatch, _dim, _weights.RmsNormEps, batchSize);

            if (lw.FfnGateType == GgufType.Q4_K && lw.FfnUpType == GgufType.Q4_K)
            {
                LaunchSwigluBatch(_dFfnActBatch, _dNormXBatch, lw.FfnGateWeight, lw.FfnUpWeight, _dim, _ffnDim, batchSize);
            }
            else
            {
                for (int t = 0; t < batchSize; t++)
                {
                    IntPtr dNorm = _dNormXBatch + t * _dim * sizeof(float);
                    IntPtr dAct = _dFfnActBatch + t * _ffnDim * sizeof(float);
                    LaunchGemv(lw.FfnGateType, _dGate, dNorm, lw.FfnGateWeight, _dim, _ffnDim);
                    LaunchGemv(lw.FfnUpType, _dUp, dNorm, lw.FfnUpWeight, _dim, _ffnDim);
                    LaunchSwiglu(_dGate, _dUp, dAct, _ffnDim);
                }
            }

            LaunchGemmBatch(lw.FfnDownType, IntPtr.Zero, _dFfnActBatch, lw.FfnDownWeight, _ffnDim, _dim, batchSize, IntPtr.Zero, _dXBatch);
        }

        if (IsLastStage)
        {
            if (computeLogits)
            {
                // We only need logits for the last token in the batch
                int lastTokenIdx = batchSize - 1;
                IntPtr dXLast = _dXBatch + lastTokenIdx * _dim * sizeof(float);

                // Final RMSNorm
                LaunchRmsNorm(dXLast, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);

                // Output projection (LM Head) on GPU
                LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);

                // Copy logits from GPU to CPU
                if (!logits.IsEmpty)
                {
                    fixed (float* pLogits = logits)
                    {
                        _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
                    }
                }
            }
        }
        else
        {
            if (!outputXBatch.IsEmpty)
            {
                fixed (float* pOut = outputXBatch)
                {
                    _gpu.CopyToHost((IntPtr)pOut, _dXBatch, (nuint)(batchSize * _dim * sizeof(float)));
                }
            }
        }
    }
}
