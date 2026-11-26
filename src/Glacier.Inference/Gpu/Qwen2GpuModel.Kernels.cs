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

    private void LaunchRope(IntPtr dQ, IntPtr dK, int pos, IntPtr hStream = default)
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
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope)");
    }

    private void LaunchRopeDpos(IntPtr dQ, IntPtr dK, IntPtr dPos, IntPtr hStream = default)
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
        pArgs[5] = &dPos;
        pArgs[6] = &freqBase;
        pArgs[7] = &freqScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRopeDpos,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_dpos)");
    }

    private void LaunchKvStore(int layer, int pos, IntPtr hStream = default)
    {
        uint blockSize = 256;
        uint gridSize;
        if (KvPrecision == KvCachePrecision.Q4_0)
        {
            int blocksPerHead = (_headDim + 31) / 32;
            int totalBlocks = _nHeadsKv * blocksPerHead;
            gridSize = (uint)((totalBlocks + 7) / 8); // 8 warps per 256-thread block
        }
        else
        {
            int total = _nHeadsKv * _headDim;
            gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);
        }

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
            KvCachePrecision.Q4_0 => _fnKvCacheStoreQ4_0,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
            _ => _fnKvCacheStore
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKv,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store)");
    }

    private void LaunchKvStoreDpos(int layer, IntPtr dPos, IntPtr hStream = default)
    {
        uint blockSize = 256;
        uint gridSize;
        if (KvPrecision == KvCachePrecision.Q4_0)
        {
            int blocksPerHead = (_headDim + 31) / 32;
            int totalBlocks = _nHeadsKv * blocksPerHead;
            gridSize = (uint)((totalBlocks + 7) / 8);
        }
        else
        {
            int total = _nHeadsKv * _headDim;
            gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);
        }

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
        pArgs[7] = &dPos;

        IntPtr fnKv = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnKvCacheStoreF16Dpos,
            KvCachePrecision.Fp8 => _fnKvCacheStoreFp8Dpos,
            KvCachePrecision.Q4_0 => _fnKvCacheStoreQ4_0Dpos,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
            _ => _fnKvCacheStoreDpos
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnKv,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(kv_cache_store_dpos)");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private uint GetAttentionBlockSize() => GetAttentionBlockSize(_headDim, _headDim);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint GetAttentionBlockSize(int headDim, int vHeadDim)
    {
        int maxDim = Math.Max(headDim, vHeadDim);
        return (uint)Math.Min(256, Math.Max(32, ((maxDim + 31) / 32) * 32));
    }

    private void LaunchAttention(int layer, int pos) => LaunchAttention(layer, pos, _dQ, _dAttnOut);

    private void LaunchAttention(int layer, int pos, IntPtr dQ, IntPtr dOut, IntPtr hStream = default)
    {
        uint gridSize = (uint)_nHeads;
        uint blockSize = GetAttentionBlockSize();

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dScores = IntPtr.Zero;
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
            KvCachePrecision.Q4_0 => _fnAttentionGqaQ4_0,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
            _ => _fnAttentionGqa
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa)");
    }

    private void LaunchAttentionDpos(int layer, IntPtr dPos, IntPtr hStream = default)
    {
        uint gridSize = (uint)_nHeads;
        uint blockSize = GetAttentionBlockSize();

        IntPtr dQ = _dQ;
        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dOut = _dAttnOut;
        IntPtr dScores = IntPtr.Zero;
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
        pArgs[9] = &dPos;
        pArgs[10] = &attnScale;

        IntPtr fnAttn = KvPrecision switch
        {
            KvCachePrecision.Fp16 => _fnAttentionGqaF16Dpos,
            KvCachePrecision.Fp8 => _fnAttentionGqaFp8Dpos,
            KvCachePrecision.Q4_0 => _fnAttentionGqaQ4_0Dpos,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
            _ => _fnAttentionGqaDpos
        };

        CuDriver.Check(CuDriver.LaunchKernel(
            fnAttn,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_gqa_dpos)");
    }

    private void LaunchAttentionBatch(int layer, int chunkStartPos, int batchSize)
    {
        uint gridX = (uint)_nHeads;
        uint gridY = (uint)batchSize;
        uint blockSize = GetAttentionBlockSize();

        IntPtr dQ = _dQBatch;
        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dOut = _dAttnOutBatch;
        IntPtr dScores = IntPtr.Zero;
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
            KvCachePrecision.Q4_0 => _fnAttentionGqaBatchQ4_0,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
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

    private static void UploadModuleGlobalUlongArray(IntPtr module, string symbolName, ulong[] values)
    {
        CuDriver.Check(CuDriver.ModuleGetGlobal(out IntPtr dPtr, out nuint bytes, module, symbolName), $"ModuleGetGlobal({symbolName})");
        nuint expectedBytes = (nuint)values.Length * sizeof(ulong);
        if (bytes < expectedBytes)
            throw new InvalidOperationException($"CUDA device global '{symbolName}' size ({bytes}) is smaller than required ({expectedBytes}).");

        fixed (ulong* pValues = values)
        {
            CuDriver.Check(CuDriver.MemcpyHtoD(dPtr, (IntPtr)pValues, expectedBytes), $"MemcpyHtoD({symbolName})");
        }
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
            GgufType.Q5_K => _fnGemvQ5K,
            GgufType.Q3_K => _fnGemvQ3K,
            GgufType.Q6_K => _fnGemvQ6K,
            GgufType.Q8_0 => _fnGemvQ8_0,
            GgufType.Q2_K => _fnGemvQ2_K,
            GgufType.Q4_0 => _fnGemvQ4_0,
            GgufType.Q5_0 => _fnGemvQ5_0,
            GgufType.IQ2_XXS => _fnGemvIQ2_XXS,
            GgufType.IQ2_XS => _fnGemvIQ2_XS,
            GgufType.IQ2_S => _fnGemvIQ2_S,
            GgufType.IQ3_XXS => _fnGemvIQ3_XXS,
            GgufType.IQ3_S => _fnGemvIQ3_S,
            GgufType.IQ4_NL => _fnGemvIQ4_NL,
            GgufType.IQ4_XS => _fnGemvIQ4_XS,
            GgufType.MXFP4 => _fnGemvMXFP4,
            GgufType.F16 => _fnGemvF16,
            GgufType.BF16 => _fnGemvBf16,
            GgufType.F32 => _fnGemvFp32,
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

    private void LaunchSwigluFused(IntPtr dDst, IntPtr dX, IntPtr dWGate, IntPtr dWUp, int kCols, int mRows, IntPtr hStream = default)
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
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemv_q4_k_swiglu_fused)");
    }

    private void LaunchRmsNorm(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps, IntPtr hStream = default)
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
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm)");
    }

    private void LaunchSwiglu(IntPtr dGate, IntPtr dUp, IntPtr dDst, int size, IntPtr hStream = default)
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
            0, hStream,
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
        if (type is not (GgufType.Q4_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.F32))
        {
            for (int t = 0; t < batchSize; t++)
            {
                IntPtr dYT = dY != IntPtr.Zero ? dY + (nint)t * mRows * sizeof(float) : IntPtr.Zero;
                IntPtr dXT = dX + (nint)t * kCols * sizeof(float);
                IntPtr dResT = dResidual != IntPtr.Zero ? dResidual + (nint)t * mRows * sizeof(float) : IntPtr.Zero;
                LaunchGemv(type, dYT, dXT, dW, kCols, mRows, dBias, dResT, hStream);
            }
            return;
        }

        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemmQ4KBatch,
            GgufType.Q6_K => _fnGemmQ6KBatch,
            GgufType.Q8_0 => _fnGemmQ8_0Batch,
            GgufType.F32 => _fnGemmFp32Batch,
            _ => throw new NotSupportedException($"GPU batch GEMM does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX;
        uint gridY;
        if (type == GgufType.F32)
        {
            int tilesPerRow = (batchSize + 7) / 8;
            long totalWarps = (long)mRows * tilesPerRow;
            gridX = (uint)((totalWarps + (int)numWarps - 1) / (int)numWarps);
            gridY = 1;
        }
        else
        {
            gridX = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);
            int tileSize = type switch
            {
                GgufType.Q6_K => 8,
                GgufType.Q8_0 => 8,
                _ => 16
            };
            gridY = (uint)((batchSize + tileSize - 1) / tileSize);
        }

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
        uint gridY = (uint)((batchSize + 15) / 16);

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
        uint blockSize = 256;
        uint gridSize;
        if (KvPrecision == KvCachePrecision.Q4_0)
        {
            int blocksPerHead = (_headDim + 31) / 32;
            int totalBlocks = batchSize * _nHeadsKv * blocksPerHead;
            gridSize = (uint)((totalBlocks + 7) / 8); // 8 warps per 256-thread block
        }
        else
        {
            int total = _nHeadsKv * _headDim * batchSize;
            gridSize = (uint)((total + (int)blockSize - 1) / (int)blockSize);
        }

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
            KvCachePrecision.Q4_0 => _fnKvCacheStoreBatchQ4_0,
            KvCachePrecision.Q8_0 => throw new NotSupportedException("Contiguous GPU KV cache does not support Q8_0."),
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

    private void LaunchRmsNormHeads(IntPtr dBuf, IntPtr dWeight, int nHeads, int headDim, float eps, IntPtr hStream = default)
    {
        uint blockSize = (uint)Math.Min(256, Math.Max(32, ((headDim + 31) / 32) * 32));
        uint gridSize = (uint)nHeads;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dBuf;
        pArgs[1] = &dWeight;
        pArgs[2] = &nHeads;
        pArgs[3] = &headDim;
        pArgs[4] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormHeads,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, hStream,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_heads)");
    }

    private void LaunchRmsNormBatchHeads(IntPtr dBuf, IntPtr dWeight, int nHeads, int headDim, int batchSize, float eps)
    {
        uint blockSize = (uint)Math.Min(256, Math.Max(32, ((headDim + 31) / 32) * 32));
        uint gridX = (uint)nHeads;
        uint gridY = (uint)batchSize;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dBuf;
        pArgs[1] = &dWeight;
        pArgs[2] = &nHeads;
        pArgs[3] = &headDim;
        pArgs[4] = &batchSize;
        pArgs[5] = &eps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBatchHeads,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_batch_heads)");
    }

    private void LaunchPagedKvStoreF16(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
    {
        int total = _nHeadsKv * _headDim;
        uint threads = 256;
        uint gridSize = (uint)((total + (int)threads - 1) / (int)threads);

        IntPtr dK = _dK;
        IntPtr dV = _dV;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;

        void** pArgs = stackalloc void*[12];
        pArgs[0] = &dK;
        pArgs[1] = &dV;
        pArgs[2] = &dKPool;
        pArgs[3] = &dVPool;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &pos;
        pArgs[6] = &modelLayer;
        pArgs[7] = &numLayers;
        pArgs[8] = &nHeadsKv;
        pArgs[9] = &headDim;
        pArgs[10] = &vHeadDim;
        pArgs[11] = &kvBlockSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnPagedKvStoreF16,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(paged_kv_cache_store_f16_kernel)");
    }

    private void LaunchPagedKvStoreBatchF16(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
    {
        int total = batchSize * _nHeadsKv * _headDim;
        uint threads = 256;
        uint gridSize = (uint)((total + (int)threads - 1) / (int)threads);

        IntPtr dK = _dKBatch;
        IntPtr dV = _dVBatch;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;

        void** pArgs = stackalloc void*[13];
        pArgs[0] = &dK;
        pArgs[1] = &dV;
        pArgs[2] = &dKPool;
        pArgs[3] = &dVPool;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &modelLayer;
        pArgs[8] = &numLayers;
        pArgs[9] = &nHeadsKv;
        pArgs[10] = &headDim;
        pArgs[11] = &vHeadDim;
        pArgs[12] = &kvBlockSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnPagedKvStoreBatchF16,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(paged_kv_cache_store_batch_f16_kernel)");
    }

    private void LaunchPagedAttentionF16(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
    {
        uint gridSize = (uint)_nHeads;
        uint threads = GetAttentionBlockSize();

        IntPtr dQ = _dQ;
        IntPtr dOut = _dAttnOut;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[14];
        pArgs[0] = &dQ;
        pArgs[1] = &dKPool;
        pArgs[2] = &dVPool;
        pArgs[3] = &dOut;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &pos;
        pArgs[6] = &modelLayer;
        pArgs[7] = &numLayers;
        pArgs[8] = &nHeadsQ;
        pArgs[9] = &nHeadsKv;
        pArgs[10] = &headDim;
        pArgs[11] = &vHeadDim;
        pArgs[12] = &kvBlockSize;
        pArgs[13] = &attnScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnPagedAttnGqaF16,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(paged_attention_gqa_f16_kernel)");
    }

    private void LaunchPagedAttentionBatchF16(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
    {
        uint gridX = (uint)_nHeads;
        uint gridY = (uint)batchSize;
        uint threads = GetAttentionBlockSize();

        IntPtr dQ = _dQBatch;
        IntPtr dOut = _dAttnOutBatch;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[15];
        pArgs[0] = &dQ;
        pArgs[1] = &dKPool;
        pArgs[2] = &dVPool;
        pArgs[3] = &dOut;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &modelLayer;
        pArgs[8] = &numLayers;
        pArgs[9] = &nHeadsQ;
        pArgs[10] = &nHeadsKv;
        pArgs[11] = &headDim;
        pArgs[12] = &vHeadDim;
        pArgs[13] = &kvBlockSize;
        pArgs[14] = &attnScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnPagedAttnGqaBatchF16,
            gridX, gridY, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(paged_attention_gqa_batch_f16_kernel)");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static KvCachePrecision ResolvePagedKvPrecision(KvCachePrecision requested) => requested switch
    {
        KvCachePrecision.Auto => KvCachePrecision.Fp16,
        KvCachePrecision.Fp16 or KvCachePrecision.Q8_0 or KvCachePrecision.Q4_0 => requested,
        _ => throw new NotSupportedException($"Unsupported CUDA paged KV cache precision: {requested}. Supported: Fp16, Q8_0, Q4_0.")
    };

    private void LaunchPagedKvStore(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize, KvCachePrecision kvPrecision)
    {
        switch (ResolvePagedKvPrecision(kvPrecision))
        {
            case KvCachePrecision.Fp16:
                LaunchPagedKvStoreF16(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q8_0:
                LaunchPagedKvStoreQ8_0(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q4_0:
                LaunchPagedKvStoreQ4_0(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
        }
    }

    private void LaunchPagedKvStoreBatch(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize, KvCachePrecision kvPrecision)
    {
        switch (ResolvePagedKvPrecision(kvPrecision))
        {
            case KvCachePrecision.Fp16:
                LaunchPagedKvStoreBatchF16(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q8_0:
                LaunchPagedKvStoreBatchQ8_0(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q4_0:
                LaunchPagedKvStoreBatchQ4_0(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
        }
    }

    private void LaunchPagedAttention(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize, KvCachePrecision kvPrecision)
    {
        switch (ResolvePagedKvPrecision(kvPrecision))
        {
            case KvCachePrecision.Fp16:
                LaunchPagedAttentionF16(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q8_0:
                LaunchPagedAttentionQ8_0(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q4_0:
                LaunchPagedAttentionQ4_0(dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);
                break;
        }
    }

    private void LaunchPagedAttentionBatch(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize, KvCachePrecision kvPrecision)
    {
        switch (ResolvePagedKvPrecision(kvPrecision))
        {
            case KvCachePrecision.Fp16:
                LaunchPagedAttentionBatchF16(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q8_0:
                LaunchPagedAttentionBatchQ8_0(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
            case KvCachePrecision.Q4_0:
                LaunchPagedAttentionBatchQ4_0(dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
                break;
        }
    }

    private void LaunchPagedKvStoreQuantized(IntPtr fn, string name, IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
    {
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        int maxBlocksPerHead = Math.Max((headDim + 31) / 32, (vHeadDim + 31) / 32);
        int totalWarps = nHeadsKv * maxBlocksPerHead;
        uint threads = 256;
        uint gridSize = (uint)((totalWarps + 7) / 8);

        IntPtr dK = _dK;
        IntPtr dV = _dV;

        void** pArgs = stackalloc void*[12];
        pArgs[0] = &dK;
        pArgs[1] = &dV;
        pArgs[2] = &dKPool;
        pArgs[3] = &dVPool;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &pos;
        pArgs[6] = &modelLayer;
        pArgs[7] = &numLayers;
        pArgs[8] = &nHeadsKv;
        pArgs[9] = &headDim;
        pArgs[10] = &vHeadDim;
        pArgs[11] = &kvBlockSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), $"LaunchKernel({name})");
    }

    private void LaunchPagedKvStoreBatchQuantized(IntPtr fn, string name, IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
    {
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        int maxBlocksPerHead = Math.Max((headDim + 31) / 32, (vHeadDim + 31) / 32);
        int totalWarps = batchSize * nHeadsKv * maxBlocksPerHead;
        uint threads = 256;
        uint gridSize = (uint)((totalWarps + 7) / 8);

        IntPtr dK = _dKBatch;
        IntPtr dV = _dVBatch;

        void** pArgs = stackalloc void*[13];
        pArgs[0] = &dK;
        pArgs[1] = &dV;
        pArgs[2] = &dKPool;
        pArgs[3] = &dVPool;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &modelLayer;
        pArgs[8] = &numLayers;
        pArgs[9] = &nHeadsKv;
        pArgs[10] = &headDim;
        pArgs[11] = &vHeadDim;
        pArgs[12] = &kvBlockSize;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), $"LaunchKernel({name})");
    }

    private void LaunchPagedAttentionQuantized(IntPtr fn, string name, IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
    {
        uint gridSize = (uint)_nHeads;
        uint threads = GetAttentionBlockSize(_headDim, _headDim);

        IntPtr dQ = _dQ;
        IntPtr dOut = _dAttnOut;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[14];
        pArgs[0] = &dQ;
        pArgs[1] = &dKPool;
        pArgs[2] = &dVPool;
        pArgs[3] = &dOut;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &pos;
        pArgs[6] = &modelLayer;
        pArgs[7] = &numLayers;
        pArgs[8] = &nHeadsQ;
        pArgs[9] = &nHeadsKv;
        pArgs[10] = &headDim;
        pArgs[11] = &vHeadDim;
        pArgs[12] = &kvBlockSize;
        pArgs[13] = &attnScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridSize, 1, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), $"LaunchKernel({name})");
    }

    private void LaunchPagedAttentionBatchQuantized(IntPtr fn, string name, IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
    {
        uint gridX = (uint)_nHeads;
        uint gridY = (uint)batchSize;
        uint threads = GetAttentionBlockSize(_headDim, _headDim);

        IntPtr dQ = _dQBatch;
        IntPtr dOut = _dAttnOutBatch;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int vHeadDim = _headDim;
        float attnScale = _attnScale;

        void** pArgs = stackalloc void*[15];
        pArgs[0] = &dQ;
        pArgs[1] = &dKPool;
        pArgs[2] = &dVPool;
        pArgs[3] = &dOut;
        pArgs[4] = &dBlockTable;
        pArgs[5] = &startPos;
        pArgs[6] = &batchSize;
        pArgs[7] = &modelLayer;
        pArgs[8] = &numLayers;
        pArgs[9] = &nHeadsQ;
        pArgs[10] = &nHeadsKv;
        pArgs[11] = &headDim;
        pArgs[12] = &vHeadDim;
        pArgs[13] = &kvBlockSize;
        pArgs[14] = &attnScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            fn,
            gridX, gridY, 1,
            threads, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), $"LaunchKernel({name})");
    }

    private void LaunchPagedKvStoreQ4_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedKvStoreQuantized(_fnPagedKvStoreQ4_0, "paged_kv_cache_store_q4_0_kernel", dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedKvStoreBatchQ4_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedKvStoreBatchQuantized(_fnPagedKvStoreBatchQ4_0, "paged_kv_cache_store_batch_q4_0_kernel", dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedAttentionQ4_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedAttentionQuantized(_fnPagedAttnGqaQ4_0, "paged_attention_gqa_q4_0_kernel", dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedAttentionBatchQ4_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedAttentionBatchQuantized(_fnPagedAttnGqaBatchQ4_0, "paged_attention_gqa_batch_q4_0_kernel", dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedKvStoreQ8_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedKvStoreQuantized(_fnPagedKvStoreQ8_0, "paged_kv_cache_store_q8_0_kernel", dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedKvStoreBatchQ8_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedKvStoreBatchQuantized(_fnPagedKvStoreBatchQ8_0, "paged_kv_cache_store_batch_q8_0_kernel", dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedAttentionQ8_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int pos, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedAttentionQuantized(_fnPagedAttnGqaQ8_0, "paged_attention_gqa_q8_0_kernel", dKPool, dVPool, dBlockTable, pos, modelLayer, numLayers, kvBlockSize);

    private void LaunchPagedAttentionBatchQ8_0(IntPtr dKPool, IntPtr dVPool, IntPtr dBlockTable, int startPos, int batchSize, int modelLayer, int numLayers, int kvBlockSize)
        => LaunchPagedAttentionBatchQuantized(_fnPagedAttnGqaBatchQ8_0, "paged_attention_gqa_batch_q8_0_kernel", dKPool, dVPool, dBlockTable, startPos, batchSize, modelLayer, numLayers, kvBlockSize);
}
