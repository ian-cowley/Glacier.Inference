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

    private void LaunchRope(IntPtr dQ, IntPtr dK, int pos)
    {
        int halfDim = _headDim / 2;
        int totalHalf = (_nHeads + _nHeadsKv) * halfDim;
        uint blockSize = 256;
        uint gridSize = (uint)((totalHalf + (int)blockSize - 1) / (int)blockSize);

        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        float freqBase = _weights.RopeFreqBase;
        float freqScale = 1.0f;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &pos;
        pArgs[6] = &freqBase;
        pArgs[7] = &freqScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRope,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope)");
    }

    private void LaunchKvStore(int layer, int pos)
    {
        int total = _nHeadsKv * _headDim;
        uint blockSize = 256;
        uint gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        IntPtr dK = _dK;
        IntPtr dV = _dV;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dKCache;
        pArgs[1] = &dVCache;
        pArgs[2] = &dK;
        pArgs[3] = &dV;
        pArgs[4] = &nHeadsKv;
        pArgs[5] = &headDim;
        pArgs[6] = &maxSeq;
        pArgs[7] = &pos;

        IntPtr fnKv = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnKvCacheStoreF16,
            KvCachePrecision.Fp8 => _fnKvCacheStoreFp8,
            _ => _fnKvCacheStore
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKv,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store)");
    }

    private void LaunchAttention(int layer, int pos) => LaunchAttention(layer, pos, _dQ, _dAttnOut);

    private void LaunchAttention(int layer, int pos, IntPtr dQ, IntPtr dOut)
    {
        uint gridSize = (uint)_nHeads;
        uint blockSize = (uint)_headDim;

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dScores = _dScoresBuf;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[11];
        pArgs[0] = &dQ;
        pArgs[1] = &dKCache;
        pArgs[2] = &dVCache;
        pArgs[3] = &dOut;
        pArgs[4] = &dScores;
        pArgs[5] = &nHeadsQ;
        pArgs[6] = &nHeadsKv;
        pArgs[7] = &headDim;
        pArgs[8] = &maxSeq;
        pArgs[9] = &pos;
        pArgs[10] = &attnScale;

        IntPtr fnAttn = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnAttentionGqaF16,
            KvCachePrecision.Fp8 => _fnAttentionGqaFp8,
            _ => _fnAttentionGqa
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa)");
    }

    private void LaunchAttentionBatch(int layer, int chunkStartPos, int batchSize)
    {
        uint gridX = (uint)_nHeads;
        uint gridY = (uint)batchSize;
        uint blockSize = (uint)_headDim;

        IntPtr dQ = _dQBatch;
        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dOut = _dAttnOutBatch;
        IntPtr dScores = _dScoresBufBatch;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[12];
        pArgs[0] = &dQ;
        pArgs[1] = &dKCache;
        pArgs[2] = &dVCache;
        pArgs[3] = &dOut;
        pArgs[4] = &dScores;
        pArgs[5] = &nHeadsQ;
        pArgs[6] = &nHeadsKv;
        pArgs[7] = &headDim;
        pArgs[8] = &maxSeq;
        pArgs[9] = &chunkStartPos;
        pArgs[10] = &batchSize;
        pArgs[11] = &attnScale;

        IntPtr fnAttnBatch = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnAttentionGqaBatchF16,
            KvCachePrecision.Fp8 => _fnAttentionGqaBatchFp8,
            _ => _fnAttentionGqaBatch
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttnBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa_batch)");
    }

    private void LaunchGemv(
        GgufType type,
        IntPtr dY,
        IntPtr dX,
        IntPtr dW,
        int kCols,
        int mRows,
        IntPtr dBias = default,
        IntPtr dResidual = default,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemvQ4K,
            GgufType.Q6_K => _fnGemvQ6K,
            GgufType.Q8_0 => _fnGemvQ8_0,
            _ => throw new NotSupportedException($"GPU GEMV does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &dBias;
        pArgs[6] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemv)");
    }

    private void LaunchSwigluFused(IntPtr dDst, IntPtr dX, IntPtr dWGate, IntPtr dWUp, int kCols, int mRows)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dDst;
        pArgs[1] = &dX;
        pArgs[2] = &dWGate;
        pArgs[3] = &dWUp;
        pArgs[4] = &kCols;
        pArgs[5] = &mRows;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwigluFused,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemv_q4_k_swiglu_fused)");
    }

    private void LaunchRmsNorm(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps)
    {
        uint blockSize = 256;
        uint gridSize = 1;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &size;
        pArgs[4] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNorm,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm)");
    }

    private void LaunchSwiglu(IntPtr dGate, IntPtr dUp, IntPtr dDst, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dGate;
        pArgs[1] = &dUp;
        pArgs[2] = &dDst;
        pArgs[3] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnSwiglu,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(swiglu)");
    }

    private void LaunchAddBias(IntPtr dY, IntPtr dBias, int size, IntPtr hStream = default)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dY;
        pArgs[1] = &dBias;
        pArgs[2] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAddBias,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(add_bias)");
    }

    private void LaunchVecAdd(IntPtr dA, IntPtr dB, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dA;
        pArgs[1] = &dB;
        pArgs[2] = &size;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAdd,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add)");
    }

    private void LaunchGemmBatch(
        GgufType type,
        IntPtr dY,
        IntPtr dX,
        IntPtr dW,
        int kCols,
        int mRows,
        int batchSize,
        IntPtr dBias = default,
        IntPtr dResidual = default,
        IntPtr hStream = default)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBatch,
            GgufType.Q6_K => _fnGemmQ6KBatch,
            GgufType.Q8_0 => _fnGemmQ8_0Batch,
            _ => throw new NotSupportedException($"GPU batch GEMM does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = 1;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &kCols;
        pArgs[4] = &mRows;
        pArgs[5] = &batchSize;
        pArgs[6] = &dBias;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_batch)");
    }

    private void LaunchSwigluBatch(IntPtr dDst, IntPtr dX, IntPtr dWGate, IntPtr dWUp, int kCols, int mRows, int batchSize)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = 1;

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dDst;
        pArgs[1] = &dX;
        pArgs[2] = &dWGate;
        pArgs[3] = &dWUp;
        pArgs[4] = &kCols;
        pArgs[5] = &mRows;
        pArgs[6] = &batchSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmSwigluBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q4_k_swiglu_batch)");
    }

    private void LaunchRmsNormBatch(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps, int batchSize)
    {
        uint blockSize = 256;
        uint gridSize = (uint)batchSize;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &size;
        pArgs[4] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_batch)");
    }

    private void LaunchAddBiasBatch(IntPtr dY, IntPtr dBias, int size, int batchSize, IntPtr hStream = default)
    {
        int totalElements = size * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dY;
        pArgs[1] = &dBias;
        pArgs[2] = &size;
        pArgs[3] = &totalElements;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAddBiasBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(add_bias_batch)");
    }

    private void LaunchVecAddBatch(IntPtr dA, IntPtr dB, int totalElements)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dA;
        pArgs[1] = &dB;
        pArgs[2] = &totalElements;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAddBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add_batch)");
    }

    private void LaunchRopeBatch(IntPtr dQ, IntPtr dK, int startPos, int batchSize)
    {
        int halfDim = _headDim / 2;
        int totalHalf = (_nHeads + _nHeadsKv) * halfDim;
        int totalAll = totalHalf * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((totalAll + (int)blockSize - 1) / (int)blockSize);

        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        float freqBase = _weights.RopeFreqBase;
        float freqScale = 1.0f;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &nHeadsQ;
        pArgs[3] = &nHeadsKv;
        pArgs[4] = &headDim;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &freqBase;
        pArgs[8] = &freqScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRopeBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_batch)");
    }

    private void LaunchKvCacheStoreBatch(int layer, int startPos, int batchSize)
    {
        int total = _nHeadsKv * _headDim * batchSize;
        uint blockSize = 256;
        uint gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dK = _dKBatch;
        IntPtr dV = _dVBatch;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dKCache;
        pArgs[1] = &dVCache;
        pArgs[2] = &dK;
        pArgs[3] = &dV;
        pArgs[4] = &nHeadsKv;
        pArgs[5] = &headDim;
        pArgs[6] = &maxSeq;
        pArgs[7] = &startPos;
        pArgs[8] = &batchSize;

        IntPtr fnKvBatch = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnKvCacheStoreBatchF16,
            KvCachePrecision.Fp8 => _fnKvCacheStoreBatchFp8,
            _ => _fnKvCacheStoreBatch
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKvBatch,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store_batch)");
    }
}
