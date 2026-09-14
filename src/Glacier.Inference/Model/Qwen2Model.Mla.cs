namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

public sealed unsafe partial class Qwen2Model
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardMlaLayer(int l, int modelLayer, int pos, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvMqaDim = _kvLoraRank + _qkRopeDim;
        int decompKvDim = _nHeads * (_qkNopeDim + _vHeadDim);
        int attnOutDim = _nHeads * _vHeadDim;

        // 1. Q projection
        QuantKernels.MatVecMul(layer.QType, layer.QWeight, _normX, _q, _dim, qDim, _normXSums);
        if (layer.QBias != null) AddVector(_q, layer.QBias, qDim);

        // 2. Compressed KV projection
        QuantKernels.MatVecMul(layer.AttnKvAMqaType, layer.AttnKvAMqaWeight, _normX, _compressedKv, _dim, kvMqaDim, _normXSums);

        // 3. RMSNorm on latent c_KV
        QuantKernels.RMSNorm(_compressedKv, layer.AttnKvANormWeight, _cKvNorm, _kvLoraRank, _weights.RmsNormEps);
        QuantKernels.ComputeBlockSums32(_cKvNorm, _cKvSums, _kvLoraRank);

        // 4. Decompress c_KV into K_nope and V
        QuantKernels.MatVecMul(layer.AttnKvBType, layer.AttnKvBWeight, _cKvNorm, _decompressedKv, _kvLoraRank, decompKvDim, _cKvSums);

        // 5. Decoupled RoPE
        float* kPe = _compressedKv + _kvLoraRank;
        QuantKernels.RoPEMla(_q, kPe, _nHeads, _headDim, _qkNopeDim, _qkRopeDim, pos, _weights.RopeFreqBase, _yarnInvFreq, _yarnMscale);

        // 6. Assemble K [192] and V [128] per head
        for (int h = 0; h < _nHeadsKv; h++)
        {
            float* kDst = _k + h * _headDim;
            float* kNopeSrc = _decompressedKv + h * (_qkNopeDim + _vHeadDim);
            Buffer.MemoryCopy(kNopeSrc, kDst, (ulong)(_qkNopeDim * sizeof(float)), (ulong)(_qkNopeDim * sizeof(float)));
            Buffer.MemoryCopy(kPe, kDst + _qkNopeDim, (ulong)(_qkRopeDim * sizeof(float)), (ulong)(_qkRopeDim * sizeof(float)));

            float* vDst = _v + h * _vHeadDim;
            float* vSrc = _decompressedKv + h * (_qkNopeDim + _vHeadDim) + _qkNopeDim;
            Buffer.MemoryCopy(vSrc, vDst, (ulong)(_vHeadDim * sizeof(float)), (ulong)(_vHeadDim * sizeof(float)));
        }

        // 7. Store in KV cache
        kvCache?.Store(l, pos, _k, _v);

        // 8. Multi-Head Attention
        if (kvCache != null)
        {
            ComputeAttention(l, modelLayer, pos, kvCache);
        }

        // 9. Output projection
        QuantKernels.ComputeBlockSums32(_attnOut, _attnOutSums, attnOutDim);
        QuantKernels.MatVecMul(layer.AttnOutType, layer.AttnOutWeight, _attnOut, _attnProj, attnOutDim, _dim, _attnOutSums);
        if (layer.AttnOutBias != null) AddVector(_attnProj, layer.AttnOutBias, _dim);

        // 10. Residual connection
        AddVector(_x, _attnProj, _dim);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardMlaBatchChunkLayer(int l, int modelLayer, int chunkStartPos, int batchSize, KVCache? kvCache)
    {
        var layer = _weights.Layers[modelLayer];
        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;
        int kvMqaDim = _kvLoraRank + _qkRopeDim;
        int decompKvDim = _nHeads * (_qkNopeDim + _vHeadDim);
        int attnOutDim = _nHeads * _vHeadDim;

        // 1. Batched Q projection
        QuantKernels.MatMulBatch(layer.QType, layer.QWeight, _normXBatch, _qBatch, _dim, qDim, batchSize, _normXSumBatch);
        if (layer.QBias != null)
        {
            for (int t = 0; t < batchSize; t++) AddVector(_qBatch + t * qDim, layer.QBias, qDim);
        }

        // 2. Batched Compressed KV projection
        QuantKernels.MatMulBatch(layer.AttnKvAMqaType, layer.AttnKvAMqaWeight, _normXBatch, _compressedKvBatch, _dim, kvMqaDim, batchSize, _normXSumBatch);

        // 3. Per-token decompress, RoPE, store, and attention
        for (int t = 0; t < batchSize; t++)
        {
            int pos = chunkStartPos + t;
            float* compressedKv = _compressedKvBatch + t * kvMqaDim;
            float* kPe = compressedKv + _kvLoraRank;
            float* q = _qBatch + t * qDim;

            // RMSNorm on latent c_KV
            QuantKernels.RMSNorm(compressedKv, layer.AttnKvANormWeight, _cKvNorm, _kvLoraRank, _weights.RmsNormEps);
            QuantKernels.ComputeBlockSums32(_cKvNorm, _cKvSums, _kvLoraRank);

            // Decompress into K_nope and V
            QuantKernels.MatVecMul(layer.AttnKvBType, layer.AttnKvBWeight, _cKvNorm, _decompressedKv, _kvLoraRank, decompKvDim, _cKvSums);

            // Decoupled RoPE
            QuantKernels.RoPEMla(q, kPe, _nHeads, _headDim, _qkNopeDim, _qkRopeDim, pos, _weights.RopeFreqBase, _yarnInvFreq, _yarnMscale);

            // Assemble K and V
            float* kt = _kBatch + t * kvDim;
            float* vt = _vBatch + t * (_nHeadsKv * _vHeadDim);
            for (int h = 0; h < _nHeadsKv; h++)
            {
                float* kDst = kt + h * _headDim;
                float* kNopeSrc = _decompressedKv + h * (_qkNopeDim + _vHeadDim);
                Buffer.MemoryCopy(kNopeSrc, kDst, (ulong)(_qkNopeDim * sizeof(float)), (ulong)(_qkNopeDim * sizeof(float)));
                Buffer.MemoryCopy(kPe, kDst + _qkNopeDim, (ulong)(_qkRopeDim * sizeof(float)), (ulong)(_qkRopeDim * sizeof(float)));

                float* vDst = vt + h * _vHeadDim;
                float* vSrc = _decompressedKv + h * (_qkNopeDim + _vHeadDim) + _qkNopeDim;
                Buffer.MemoryCopy(vSrc, vDst, (ulong)(_vHeadDim * sizeof(float)), (ulong)(_vHeadDim * sizeof(float)));
            }

            // Store in KV cache
            kvCache?.Store(l, pos, kt, vt);

            // Attention
            if (kvCache != null)
            {
                ComputeAttentionToken(l, modelLayer, pos, q, _attnOutBatch + t * attnOutDim, kvCache);
            }
        }

        // 4. Batched attention output projection
        int attnOutChunks = (attnOutDim + 31) / 32;
        for (int t = 0; t < batchSize; t++)
        {
            QuantKernels.ComputeBlockSums32(_attnOutBatch + t * attnOutDim, _attnOutSumBatch + t * attnOutChunks, attnOutDim);
        }
        QuantKernels.MatMulBatch(layer.AttnOutType, layer.AttnOutWeight, _attnOutBatch, _attnProjBatch, attnOutDim, _dim, batchSize, _attnOutSumBatch);
        if (layer.AttnOutBias != null)
        {
            for (int t = 0; t < batchSize; t++) AddVector(_attnProjBatch + t * _dim, layer.AttnOutBias, _dim);
        }

        // 5. Residual connection
        for (int t = 0; t < batchSize; t++)
        {
            AddVector(_xBatch + t * _dim, _attnProjBatch + t * _dim, _dim);
        }
    }
}
