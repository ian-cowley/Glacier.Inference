namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

public sealed unsafe partial class Qwen2Model
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardAttention(int stageLayer, int modelLayer, int pos, float* x, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];

        if (layer.IsMla)
        {
            ForwardMlaLayer(stageLayer, modelLayer, pos, kvCache);
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
                QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
                QuantKernels.MatVecMul(layer.KType, layer.KWeight, _normX, _k, _dim, kvDim, _normXSums);
                QuantKernels.MatVecMul(layer.VType, layer.VWeight, _normX, _v, _dim, kvDim, _normXSums);

                if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);
                if (layer.KBias != null) AddVector(_k, layer.KBias, kvDim);
                if (layer.VBias != null) AddVector(_v, layer.VBias, kvDim);
            }

            if (LoraWeights != null)
            {
                LoraWeights.Apply(modelLayer, LoraProjection.Q, _normX, _q, _dim, qDim);
                LoraWeights.Apply(modelLayer, LoraProjection.K, _normX, _k, _dim, kvDim);
                LoraWeights.Apply(modelLayer, LoraProjection.V, _normX, _v, _dim, kvDim);
            }

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
                ComputeAttention(stageLayer, modelLayer, pos, kvCache);
            }

            QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, qDim);
            QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, qDim, _dim, _attnOutSums);
            if (LoraWeights != null)
            {
                LoraWeights.Apply(modelLayer, LoraProjection.AttnOut, _attnOut, _attnProj, qDim, _dim);
            }
            if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

            AddVector(x, _attnProj, _dim);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    protected override void ForwardBatchChunkAttention(int stageLayer, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        if (layer.IsMla)
        {
            ForwardMlaBatchChunkLayer(stageLayer, modelLayer, chunkStartPos, batchSize, kvCache);
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
                QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
                QuantKernels.MatMulBatch(layer.KType, layer.KWeight, _normXBatch, _kBatch, _dim, kvDim, batchSize, _normXSumBatch);
                QuantKernels.MatMulBatch(layer.VType, layer.VWeight, _normXBatch, _vBatch, _dim, kvDim, batchSize, _normXSumBatch);

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
                    ComputeAttentionToken(stageLayer, modelLayer, pos, q, _attnOutBatch + t * qDim, kvCache);
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

            for (int t = 0; t < batchSize; t++)
            {
                AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
            }
        }
    }
}
