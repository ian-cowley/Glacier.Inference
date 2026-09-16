namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

public sealed unsafe partial class Qwen2Model
{
    /// <summary>
    /// Executes forward pass for a single token at position pos.
    /// Writes logits of size VocabSize into destination buffer if computeLogits is true.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        ForwardStage(token, pos, default, default, logits, computeLogits, kvCache);
    }

    /// <summary>
    /// Executes layers assigned to this stage for a single token forward step.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits,
        KVCache? kvCache = null)
    {
        // 1. Embedding lookup or intermediate activation copy
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

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 2. Transformer layers
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var layer = _weights.Layers[modelLayer];

            // Attention pre-norm
            QuantKernels.RMSNorm(_x, layer.AttnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            if (layer.IsMla)
            {
                ForwardMlaLayer(l, modelLayer, pos, kvCache);
            }
            else
            {
                int qDim = _nHeads * _headDim;
                int kvDim = _nHeadsKv * _headDim;

                if (layer.HasFusedQkv)
                {
                    int qkvDim = qDim + 2 * kvDim;
                    QuantKernels.MatVecMul(layer.QkvType, layer.QkvWeight, _normX, _qkvFused, _dim, qkvDim, _normXSums);
                    if (layer.QkvBias != null) AddVector(_qkvFused, layer.QkvBias, qkvDim);

                    Buffer.MemoryCopy(_qkvFused, _q, (long)qDim * sizeof(float), (long)qDim * sizeof(float));
                    Buffer.MemoryCopy(_qkvFused + qDim, _k, (long)kvDim * sizeof(float), (long)kvDim * sizeof(float));
                    Buffer.MemoryCopy(_qkvFused + qDim + kvDim, _v, (long)kvDim * sizeof(float), (long)kvDim * sizeof(float));
                }
                else
                {
                    // Q, K, V projections (reusing _normXSums across all 3)
                    QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
                    QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
                    QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

                    // Add Q, K, V biases if present
                    if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
                    if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
                    if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);
                }

                // Add LoRA deltas if present
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Q, _normX, _q, _dim, qDim);
                    LoraWeights.Apply(modelLayer, LoraProjection.K, _normX, _k, _dim, kvDim);
                    LoraWeights.Apply(modelLayer, LoraProjection.V, _normX, _v, _dim, kvDim);
                }

                // Optional QK-Norm (e.g. Qwen3)
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

                // Rotary Position Embedding (RoPE)
                QuantKernels.RoPE(_q, _k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight);

                // Store in KV cache (stage-relative layer l)
                kvCache?.Store(l, pos, _k, _v);

                // Multi-Head / Grouped Query Attention (GQA)
                if (kvCache != null)
                {
                    ComputeAttention(l, modelLayer, pos, kvCache);
                }

                // Attention output projection
                QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
                QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.AttnOut, _attnOut, _attnProj, qDim, _dim);
                }
                if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

                // Residual connection: x = x + attnProj
                AddVector(_x, _attnProj, _dim);
            }

            // FFN pre-norm
            QuantKernels.RMSNorm(_x, layer.FfnNormWeight, _normX, _dim, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

            if (layer.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;

                QuantKernels.RouterTopK(_normX, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);

                new Span<float>(_ffnOut, _dim).Clear();

                long gateSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
                long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
                long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

                for (int k = 0; k < topK; k++)
                {
                    int expertIdx = selectedIndices[k];
                    float weight = selectedWeights[k];

                    byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
                    byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;
                    byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;

                    QuantKernels.MatVecMul(layer.FfnGateExpsType, expGateWeight, _normX, _gate, _dim, expertFfnDim, _normXSums);
                    if (layer.FfnGateExpsBias != null) AddVector(_gate, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                    QuantKernels.MatVecMul(layer.FfnUpExpsType, expUpWeight, _normX, _up, _dim, expertFfnDim, _normXSums);
                    if (layer.FfnUpExpsBias != null) AddVector(_up, layer.FfnUpExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);

                    QuantKernels.MatVecMul(layer.FfnDownExpsType, expDownWeight, _ffnAct, _expertDownOut, expertFfnDim, _dim, _ffnActSums);
                    if (layer.FfnDownExpsBias != null) AddVector(_expertDownOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);

                    for (int d = 0; d < _dim; d++)
                    {
                        _ffnOut[d] += weight * _expertDownOut[d];
                    }
                }

                // Shared Expert if present
                if (layer.FfnGateShexpWeight != null)
                {
                    int shexpFfnDim = expertFfnDim * 2;
                    if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                    {
                        shexpFfnDim = (int)tShexp.Dimensions[1];
                    }

                    QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, _normX, _gate, _dim, shexpFfnDim, _normXSums);
                    QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, _normX, _up, _dim, shexpFfnDim, _normXSums);
                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                    QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                    QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                    AddVector(_ffnOut, _expertDownOut, _dim);
                }

                // Residual connection: x = x + ffnOut
                AddVector(_x, _ffnOut, _dim);
            }
            else
            {
                // Dense FFN
                if (layer.HasFusedGateUp)
                {
                    int gateUpDim = 2 * _ffnDim;
                    QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _gateUpFused, _dim, gateUpDim, _normXSums);
                    if (layer.FfnUpBias != null) AddVector(_gateUpFused, layer.FfnUpBias, gateUpDim);

                    QuantKernels.SwiGLU(_gateUpFused, _gateUpFused + _ffnDim, _ffnAct, _ffnDim);
                }
                else
                {
                    // SwiGLU FFN projections (reusing _normXSums for Gate and Up)
                    QuantKernels.MatVecMul(layer.FfnGateType, layer.FfnGateWeight, _normX, _gate, _dim, _ffnDim, _normXSums);
                    QuantKernels.MatVecMul(layer.FfnUpType, layer.FfnUpWeight, _normX, _up, _dim, _ffnDim, _normXSums);
                    if (LoraWeights != null)
                    {
                        LoraWeights.Apply(modelLayer, LoraProjection.Gate, _normX, _gate, _dim, _ffnDim);
                        LoraWeights.Apply(modelLayer, LoraProjection.Up, _normX, _up, _dim, _ffnDim);
                    }
                    QuantKernels.SwiGLU(_gate, _up, _ffnAct, _ffnDim);
                }
                QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, _ffnDim);
                QuantKernels.MatVecMul(layer.FfnDownType, layer.FfnDownWeight, _ffnAct, _ffnOut, _ffnDim, _dim, _ffnActSums);
                if (LoraWeights != null)
                {
                    LoraWeights.Apply(modelLayer, LoraProjection.Down, _ffnAct, _ffnOut, _ffnDim, _dim);
                }

                // Residual connection: x = x + ffnOut
                AddVector(_x, _ffnOut, _dim);
            }
        }

        if (IsLastStage)
        {
            if (computeLogits)
            {
                // Final RMSNorm
                QuantKernels.RMSNorm(_x, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                // LM Head output projection
                if (!logits.IsEmpty)
                {
                    fixed (float* logitsPtr = logits)
                    {
                        QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, logitsPtr, _dim, _weights.VocabSize, _normXSums);
                    }
                }
            }
        }
        else
        {
            if (!outputX.IsEmpty)
            {
                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_x, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
            }
        }
    }

    /// <summary>
    /// Evaluates a sequence of prompt tokens in batched chunks. Weights are streamed once per chunk
    /// across all tokens, reducing memory bus traffic by up to MaxBatchSize-fold.
    /// </summary>
    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, KVCache kvCache, bool computeLogits = true)
    {
        ForwardBatchStage(tokens, startPos, default, default, logits, computeLogits, kvCache);
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
        bool computeLogits,
        KVCache? kvCache = null)
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
                kvCache,
                isLastChunk && computeLogits);
            offset += batchSize;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardBatchChunk(
        ReadOnlySpan<int> chunkTokens,
        int chunkStartPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        KVCache? kvCache,
        bool computeLogits)
    {
        int batchSize = !chunkTokens.IsEmpty ? chunkTokens.Length : (inputXBatch.Length / _dim);
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int normXChunks = _dim / 32;
        int ffnChunks = _ffnDim / 32;

        int maxTopK = _weights.ExpertUsedCount > 0 ? _weights.ExpertUsedCount : 1;
        int* selectedIndices = stackalloc int[maxTopK];
        float* selectedWeights = stackalloc float[maxTopK];

        // 1. Extract embeddings or copy intermediate input activations
        if (StartLayer == 0)
        {
            for (int t = 0; t < batchSize; t++)
            {
                QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, chunkTokens[t], _xBatch + t * _dim, _dim);
            }
        }
        else
        {
            fixed (float* pIn = inputXBatch)
            {
                Buffer.MemoryCopy(pIn, _xBatch, (ulong)(batchSize * _dim * sizeof(float)), (ulong)(batchSize * _dim * sizeof(float)));
            }
        }

        // 2. Transformer layers
        for (int l = 0; l < LayerCount; l++)
        {
            int modelLayer = StartLayer + l;
            var layer = _weights.Layers[modelLayer];

            // Attention pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.AttnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

            if (layer.IsMla)
            {
                ForwardMlaBatchChunkLayer(l, modelLayer, chunkStartPos, batchSize, kvCache);
            }
            else
            {
                if (layer.HasFusedQkv)
                {
                    int qkvDim = qDim + 2 * kvDim;
                    QuantKernels.MatMulBatch(layer.QkvType, layer.QkvWeight, _normXBatch, _qkvBatch, _dim, qkvDim, batchSize, _normXSumBatch);
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* src = _qkvBatch + t * qkvDim;
                        if (layer.QkvBias != null) AddVector(src, layer.QkvBias, qkvDim);

                        Buffer.MemoryCopy(src, _qBatch + t * qDim, (long)qDim * sizeof(float), (long)qDim * sizeof(float));
                        Buffer.MemoryCopy(src + qDim, _kBatch + t * kvDim, (long)kvDim * sizeof(float), (long)kvDim * sizeof(float));
                        Buffer.MemoryCopy(src + qDim + kvDim, _vBatch + t * kvDim, (long)kvDim * sizeof(float), (long)kvDim * sizeof(float));
                    }
                }
                else
                {
                    // Batched Q, K, V projections (weights streamed once!)
                    QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
                    QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumBatch);
                    QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumBatch);

                    // Add Q, K, V biases if present
                    if (layer.QBias != null || layer.KBias != null || layer.VBias != null)
                    {
                        for (int t = 0; t < batchSize; t++)
                        {
                            if (layer.QBias != null) AddVector(_qBatch + t * qDim, layer.QBias, qDim);
                            if (layer.KBias != null) AddVector(_kBatch + t * kvDim, layer.KBias, kvDim);
                            if (layer.VBias != null) AddVector(_vBatch + t * kvDim, layer.VBias, kvDim);
                        }
                    }
                }

                // Per-token RoPE, KV cache store, and Attention
                for (int t = 0; t < batchSize; t++)
                {
                    int pos = chunkStartPos + t;
                    float* q = _qBatch + t * qDim;
                    float* k = _kBatch + t * kvDim;
                    float* v = _vBatch + t * kvDim;

                    // Optional QK-Norm
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

                    QuantKernels.RoPE(q, k, _nHeads, _nHeadsKv, _headDim, pos, _weights.RopeFreqBase, ropeFreqs: _weights.RopeFreqsWeight);
                    kvCache?.Store(l, pos, k, v);

                    if (kvCache != null)
                    {
                        ComputeAttentionToken(l, modelLayer, pos, q, _attnOutBatch + t * qDim, kvCache);
                    }
                }

                // Attention output projection (weights streamed once!)
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

                // Residual connection
                for (int t = 0; t < batchSize; t++)
                {
                    AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
                }
            }

            // FFN pre-norm for all tokens in chunk
            for (int t = 0; t < batchSize; t++)
            {
                float* xt = _xBatch + t * _dim;
                float* normXt = _normXBatch + t * _dim;
                QuantKernels.RMSNorm(xt, layer.FfnNormWeight, normXt, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(normXt, _normXSumBatch + t * normXChunks, _dim);
            }

            if (layer.IsMoe)
            {
                int expertFfnDim = _weights.ExpertFeedForwardLength;
                int numExperts = _weights.ExpertCount;
                int topK = _weights.ExpertUsedCount;

                long gateSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
                long upSliceBytes = (long)expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
                long downSliceBytes = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, expertFfnDim);

                for (int t = 0; t < batchSize; t++)
                {
                    float* normXt = _normXBatch + t * _dim;
                    float* normXSumst = _normXSumBatch + t * normXChunks;
                    float* ffnOutt = _ffnOutBatch + t * _dim;

                    QuantKernels.RouterTopK(normXt, layer.FfnGateInpWeight, layer.FfnGateInpBias, _dim, numExperts, topK, selectedIndices, selectedWeights, _weights.NormTopK);

                    for (int d = 0; d < _dim; d++) ffnOutt[d] = 0f;

                    for (int k = 0; k < topK; k++)
                    {
                        int expertIdx = selectedIndices[k];
                        float weight = selectedWeights[k];

                        byte* expGateWeight = layer.FfnGateExpsWeight + expertIdx * gateSliceBytes;
                        byte* expUpWeight = layer.FfnUpExpsWeight + expertIdx * upSliceBytes;
                        byte* expDownWeight = layer.FfnDownExpsWeight + expertIdx * downSliceBytes;

                        QuantKernels.MatVecMul(layer.FfnGateExpsType, expGateWeight, normXt, _gate, _dim, expertFfnDim, normXSumst);
                        if (layer.FfnGateExpsBias != null) AddVector(_gate, layer.FfnGateExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                        QuantKernels.MatVecMul(layer.FfnUpExpsType, expUpWeight, normXt, _up, _dim, expertFfnDim, normXSumst);
                        if (layer.FfnUpExpsBias != null) AddVector(_up, layer.FfnUpExpsBias + (long)expertIdx * expertFfnDim, expertFfnDim);

                        QuantKernels.SwiGLU(_gate, _up, _ffnAct, expertFfnDim);
                        QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, expertFfnDim);

                        QuantKernels.MatVecMul(layer.FfnDownExpsType, expDownWeight, _ffnAct, _expertDownOut, expertFfnDim, _dim, _ffnActSums);
                        if (layer.FfnDownExpsBias != null) AddVector(_expertDownOut, layer.FfnDownExpsBias + (long)expertIdx * _dim, _dim);

                        for (int d = 0; d < _dim; d++)
                        {
                            ffnOutt[d] += weight * _expertDownOut[d];
                        }
                    }

                    // Shared Expert if present
                    if (layer.FfnGateShexpWeight != null)
                    {
                        int shexpFfnDim = expertFfnDim * 2;
                        if (_weights.Gguf.TryGetTensor($"blk.{modelLayer}.ffn_gate_shexp.weight", out var tShexp) && tShexp != null)
                        {
                            shexpFfnDim = (int)tShexp.Dimensions[1];
                        }

                        QuantKernels.MatVecMul(layer.FfnGateShexpType, layer.FfnGateShexpWeight, normXt, _gate, _dim, shexpFfnDim, normXSumst);
                        QuantKernels.MatVecMul(layer.FfnUpShexpType, layer.FfnUpShexpWeight, normXt, _up, _dim, shexpFfnDim, normXSumst);
                        QuantKernels.SwiGLU(_gate, _up, _ffnAct, shexpFfnDim);
                        QuantKernels.ComputeBlockSums32(_ffnAct, _ffnActSums, shexpFfnDim);
                        QuantKernels.MatVecMul(layer.FfnDownShexpType, layer.FfnDownShexpWeight, _ffnAct, _expertDownOut, shexpFfnDim, _dim, _ffnActSums);

                        AddVector(ffnOutt, _expertDownOut, _dim);
                    }

                    AddVector(_xBatch + t * _dim, ffnOutt, _dim);
                }
            }
            else
            {
                if (layer.HasFusedGateUp)
                {
                    int gateUpDim = 2 * _ffnDim;
                    QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _gateUpBatch, _dim, gateUpDim, batchSize, _normXSumBatch);

                    // SwiGLU activation
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* src = _gateUpBatch + t * gateUpDim;
                        if (layer.FfnUpBias != null) AddVector(src, layer.FfnUpBias, gateUpDim);

                        float* act = _ffnActBatch + t * _ffnDim;
                        QuantKernels.SwiGLU(src, src + _ffnDim, act, _ffnDim);
                        QuantKernels.ComputeBlockSums32(act, _ffnActSumBatch + t * ffnChunks, _ffnDim);
                    }
                }
                else
                {
                    // Batched Gate and Up projections (weights streamed once!)
                    QuantKernels.MatMulBatch(layer.FfnGateType, layer.FfnGateWeight, _normXBatch, _gateBatch, _dim, _ffnDim, batchSize, _normXSumBatch);
                    QuantKernels.MatMulBatch(layer.FfnUpType, layer.FfnUpWeight, _normXBatch, _upBatch, _dim, _ffnDim, batchSize, _normXSumBatch);

                    // SwiGLU activation
                    for (int t = 0; t < batchSize; t++)
                    {
                        float* act = _ffnActBatch + t * _ffnDim;
                        QuantKernels.SwiGLU(_gateBatch + t * _ffnDim, _upBatch + t * _ffnDim, act, _ffnDim);
                        QuantKernels.ComputeBlockSums32(act, _ffnActSumBatch + t * ffnChunks, _ffnDim);
                    }
                }

                // Batched FFN Down projection (weights streamed once!)
                QuantKernels.MatMulBatch(layer.FfnDownType, layer.FfnDownWeight, _ffnActBatch, _ffnOutBatch, _ffnDim, _dim, batchSize, _ffnActSumBatch);

                // Residual connection
                for (int t = 0; t < batchSize; t++)
                {
                    AddVector(_xBatch + t * _dim, _ffnOutBatch + t * _dim, _dim);
                }
            }
        }

        if (IsLastStage)
        {
            if (computeLogits)
            {
                // Compute logits only for the last token in the batch
                int lastTokenIdx = batchSize - 1;
                float* xLast = _xBatch + lastTokenIdx * _dim;
                QuantKernels.RMSNorm(xLast, _weights.OutNormWeight, _normX, _dim, _weights.RmsNormEps);
                QuantKernels.ComputeBlockSums32(_normX, _normXSums, _dim);

                if (!logits.IsEmpty)
                {
                    fixed (float* logitsPtr = logits)
                    {
                        QuantKernels.MatVecMul(_weights.OutType, _weights.OutWeight, _normX, logitsPtr, _dim, _weights.VocabSize, _normXSums);
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
                    Buffer.MemoryCopy(_xBatch, pOut, (ulong)(batchSize * _dim * sizeof(float)), (ulong)(batchSize * _dim * sizeof(float)));
                }
            }
        }
    }
}
