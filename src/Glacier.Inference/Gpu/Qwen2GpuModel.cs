namespace Glacier.Inference.Gpu;

using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// GPU-accelerated transformer layer weights residing directly in GPU VRAM.
/// </summary>
public sealed class GpuLayerWeights
{
    public IntPtr AttnNormWeight { get; init; }
    public IntPtr QWeight { get; init; }
    public IntPtr QBias { get; init; }
    public IntPtr KWeight { get; init; }
    public IntPtr KBias { get; init; }
    public IntPtr VWeight { get; init; }
    public IntPtr VBias { get; init; }
    public IntPtr AttnOutWeight { get; init; }

    public IntPtr FfnNormWeight { get; init; }
    public IntPtr FfnGateWeight { get; init; }
    public IntPtr FfnUpWeight { get; init; }
    public IntPtr FfnDownWeight { get; init; }

    public GgufType QType { get; init; }
    public GgufType KType { get; init; }
    public GgufType VType { get; init; }
    public GgufType AttnOutType { get; init; }
    public GgufType FfnGateType { get; init; }
    public GgufType FfnUpType { get; init; }
    public GgufType FfnDownType { get; init; }
}

/// <summary>
/// High-performance bare-metal GPU transformer runtime for Qwen2 / Qwen2.5 models.
/// Executes directly on NVIDIA hardware via nvcuda.dll with zero external C++ DLL dependencies.
/// </summary>
public sealed unsafe class Qwen2GpuModel : IDisposable
{
    private readonly GpuContext _gpu;
    private readonly ModelWeights _weights;
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly int _headDim;
    private readonly int _nHeads;
    private readonly int _nHeadsKv;
    private readonly int _groupSize;
    private readonly float _attnScale;
    private readonly int _maxSeqLen;

    private IntPtr _module;
    private IntPtr _fnGemvQ4K;
    private IntPtr _fnGemvQ6K;
    private IntPtr _fnRmsNorm;
    private IntPtr _fnSwiglu;
    private IntPtr _fnAddBias;
    private IntPtr _fnVecAdd;
    private IntPtr _fnRope;
    private IntPtr _fnKvCacheStore;
    private IntPtr _fnAttentionGqa;

    // GPU VRAM Weights
    private IntPtr _dOutNormWeight;
    private IntPtr _dOutWeight;
    private readonly GpuLayerWeights[] _layerWeights;

    // GPU VRAM Intermediate Buffers
    private IntPtr _dX;
    private IntPtr _dNormX;
    private IntPtr _dQ;
    private IntPtr _dK;
    private IntPtr _dV;
    private IntPtr _dAttnOut;
    private IntPtr _dAttnProj;
    private IntPtr _dGate;
    private IntPtr _dUp;
    private IntPtr _dFfnAct;
    private IntPtr _dFfnOut;
    private IntPtr _dLogits;
    private IntPtr _dScoresBuf;

    // GPU VRAM KV Cache
    private readonly IntPtr[] _dKeyCache;
    private readonly IntPtr[] _dValCache;

    // Host buffer for input embedding
    private float[] _hX;

    private bool _disposed;

    public GpuContext Context => _gpu;
    public ModelWeights Weights => _weights;

    public Qwen2GpuModel(GpuContext gpu, ModelWeights weights, int maxSeqLen = 4096)
    {
        _gpu = gpu;
        _weights = weights;
        _dim = weights.EmbeddingLength;
        _ffnDim = weights.FeedForwardLength;
        _headDim = weights.HeadDim;
        _nHeads = weights.HeadCount;
        _nHeadsKv = weights.HeadCountKv;
        _groupSize = _nHeads / _nHeadsKv;
        _attnScale = 1.0f / MathF.Sqrt(_headDim);
        _maxSeqLen = maxSeqLen;

        // 1. Load compiled CUBIN kernel module
        byte[] cubin = KernelCompiler.GetOrCompileKernels("sm_89");
        CuDriver.Check(CuDriver.ModuleLoadData(out _module, cubin), "cuModuleLoadData(kernels.cubin)");

        // 2. Retrieve kernel function handles
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ4K, _module, "gemv_q4_k"), "ModuleGetFunction(gemv_q4_k)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnGemvQ6K, _module, "gemv_q6_k"), "ModuleGetFunction(gemv_q6_k)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRmsNorm, _module, "rms_norm_kernel"), "ModuleGetFunction(rms_norm_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnSwiglu, _module, "swiglu_kernel"), "ModuleGetFunction(swiglu_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAddBias, _module, "add_bias_kernel"), "ModuleGetFunction(add_bias_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnVecAdd, _module, "vec_add_kernel"), "ModuleGetFunction(vec_add_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnRope, _module, "rope_kernel"), "ModuleGetFunction(rope_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnKvCacheStore, _module, "kv_cache_store_kernel"), "ModuleGetFunction(kv_cache_store_kernel)");
        CuDriver.Check(CuDriver.ModuleGetFunction(out _fnAttentionGqa, _module, "attention_gqa_kernel"), "ModuleGetFunction(attention_gqa_kernel)");

        // 3. Allocate GPU VRAM scratch buffers
        _dX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dNormX = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dQ = _gpu.AllocateDevice((nuint)(_nHeads * _headDim * sizeof(float)));
        _dK = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _dV = _gpu.AllocateDevice((nuint)(_nHeadsKv * _headDim * sizeof(float)));
        _dAttnOut = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dAttnProj = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dGate = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dUp = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dFfnAct = _gpu.AllocateDevice((nuint)(_ffnDim * sizeof(float)));
        _dFfnOut = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _dLogits = _gpu.AllocateDevice((nuint)(_weights.VocabSize * sizeof(float)));
        _dScoresBuf = _gpu.AllocateDevice((nuint)((long)_nHeads * _maxSeqLen * sizeof(float)));

        // 4. Allocate GPU VRAM KV Cache per layer
        _dKeyCache = new IntPtr[_weights.BlockCount];
        _dValCache = new IntPtr[_weights.BlockCount];
        nuint kvBytes = (nuint)((long)_nHeadsKv * _maxSeqLen * _headDim * sizeof(float));
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            _dKeyCache[l] = _gpu.AllocateDevice(kvBytes);
            _dValCache[l] = _gpu.AllocateDevice(kvBytes);
        }

        _hX = new float[_dim];

        // 4. Upload model weights into GPU VRAM
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($">> Uploading 4.68 GB model weights into {gpu.DeviceName} VRAM...");
        var sw = Stopwatch.StartNew();

        _dOutNormWeight = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
        _gpu.CopyToDevice(_dOutNormWeight, (IntPtr)_weights.OutNormWeight, (nuint)(_dim * sizeof(float)));

        nuint outBytes = (nuint)GgufTypes.GetRowBytes(_weights.OutType, _dim) * (nuint)_weights.VocabSize;
        _dOutWeight = _gpu.AllocateDevice(outBytes);
        _gpu.CopyToDevice(_dOutWeight, (IntPtr)_weights.OutWeight, outBytes);

        _layerWeights = new GpuLayerWeights[_weights.BlockCount];
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _weights.Layers[l];

            nuint qDim = (nuint)(_nHeads * _headDim);
            nuint kvDim = (nuint)(_nHeadsKv * _headDim);

            nuint qBytes = (nuint)GgufTypes.GetRowBytes(lw.QType, _dim) * qDim;
            nuint kBytes = (nuint)GgufTypes.GetRowBytes(lw.KType, _dim) * kvDim;
            nuint vBytes = (nuint)GgufTypes.GetRowBytes(lw.VType, _dim) * kvDim;
            nuint attnOutBytes = (nuint)GgufTypes.GetRowBytes(lw.AttnOutType, _dim) * (nuint)_dim;

            nuint ffnGateBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnGateType, _dim) * (nuint)_ffnDim;
            nuint ffnUpBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnUpType, _dim) * (nuint)_ffnDim;
            nuint ffnDownBytes = (nuint)GgufTypes.GetRowBytes(lw.FfnDownType, _ffnDim) * (nuint)_dim;

            IntPtr dAttnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _gpu.CopyToDevice(dAttnNorm, (IntPtr)lw.AttnNormWeight, (nuint)(_dim * sizeof(float)));

            IntPtr dQ = _gpu.AllocateDevice(qBytes);
            _gpu.CopyToDevice(dQ, (IntPtr)lw.QWeight, qBytes);
            IntPtr dQBias = _gpu.AllocateDevice(qDim * sizeof(float));
            _gpu.CopyToDevice(dQBias, (IntPtr)lw.QBias, qDim * sizeof(float));

            IntPtr dK = _gpu.AllocateDevice(kBytes);
            _gpu.CopyToDevice(dK, (IntPtr)lw.KWeight, kBytes);
            IntPtr dKBias = _gpu.AllocateDevice(kvDim * sizeof(float));
            _gpu.CopyToDevice(dKBias, (IntPtr)lw.KBias, kvDim * sizeof(float));

            IntPtr dV = _gpu.AllocateDevice(vBytes);
            _gpu.CopyToDevice(dV, (IntPtr)lw.VWeight, vBytes);
            IntPtr dVBias = _gpu.AllocateDevice(kvDim * sizeof(float));
            _gpu.CopyToDevice(dVBias, (IntPtr)lw.VBias, kvDim * sizeof(float));

            IntPtr dAttnOut = _gpu.AllocateDevice(attnOutBytes);
            _gpu.CopyToDevice(dAttnOut, (IntPtr)lw.AttnOutWeight, attnOutBytes);

            IntPtr dFfnNorm = _gpu.AllocateDevice((nuint)(_dim * sizeof(float)));
            _gpu.CopyToDevice(dFfnNorm, (IntPtr)lw.FfnNormWeight, (nuint)(_dim * sizeof(float)));

            IntPtr dFfnGate = _gpu.AllocateDevice(ffnGateBytes);
            _gpu.CopyToDevice(dFfnGate, (IntPtr)lw.FfnGateWeight, ffnGateBytes);

            IntPtr dFfnUp = _gpu.AllocateDevice(ffnUpBytes);
            _gpu.CopyToDevice(dFfnUp, (IntPtr)lw.FfnUpWeight, ffnUpBytes);

            IntPtr dFfnDown = _gpu.AllocateDevice(ffnDownBytes);
            _gpu.CopyToDevice(dFfnDown, (IntPtr)lw.FfnDownWeight, ffnDownBytes);

            _layerWeights[l] = new GpuLayerWeights
            {
                AttnNormWeight = dAttnNorm,
                QWeight = dQ,
                QBias = dQBias,
                KWeight = dK,
                KBias = dKBias,
                VWeight = dV,
                VBias = dVBias,
                AttnOutWeight = dAttnOut,
                FfnNormWeight = dFfnNorm,
                FfnGateWeight = dFfnGate,
                FfnUpWeight = dFfnUp,
                FfnDownWeight = dFfnDown,
                QType = lw.QType,
                KType = lw.KType,
                VType = lw.VType,
                AttnOutType = lw.AttnOutType,
                FfnGateType = lw.FfnGateType,
                FfnUpType = lw.FfnUpType,
                FfnDownType = lw.FfnDownType
            };
        }

        sw.Stop();
        Console.WriteLine($"   Weights uploaded to GPU VRAM in {sw.ElapsedMilliseconds} ms ({sw.Elapsed.TotalSeconds:F2} s)!");
        Console.ResetColor();
    }

    /// <summary>
    /// Executes transformer forward pass directly on NVIDIA GPU with zero intermediate PCIe round-trips.
    /// RoPE, GQA Attention, RMSNorm, GEMV, SwiGLU, and residual additions all execute in-VRAM.
    /// </summary>
    public void Forward(int token, int pos, KVCache kvCache, Span<float> logits, bool computeLogits = true)
    {
        // 1. Extract embedding into host buffer and copy to GPU
        fixed (float* pX = _hX)
        {
            QuantKernels.ExtractEmbedding(_weights.EmbdType, _weights.EmbdWeight, token, pX, _dim);
            _gpu.CopyToDevice(_dX, (IntPtr)pX, (nuint)(_dim * sizeof(float)));
        }

        int qDim = _nHeads * _headDim;
        int kvDim = _nHeadsKv * _headDim;

        // 2. Transformer layers (100% inside GPU VRAM)
        for (int l = 0; l < _weights.BlockCount; l++)
        {
            var lw = _layerWeights[l];

            // Attention pre-norm in VRAM
            LaunchRmsNorm(_dX, lw.AttnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Q, K, V projections on GPU
            LaunchGemv(lw.QType, _dQ, _dNormX, lw.QWeight, _dim, qDim);
            LaunchAddBias(_dQ, lw.QBias, qDim);

            LaunchGemv(lw.KType, _dK, _dNormX, lw.KWeight, _dim, kvDim);
            LaunchAddBias(_dK, lw.KBias, kvDim);

            LaunchGemv(lw.VType, _dV, _dNormX, lw.VWeight, _dim, kvDim);
            LaunchAddBias(_dV, lw.VBias, kvDim);

            // Rotary Position Embedding (RoPE) directly in VRAM
            LaunchRope(_dQ, _dK, pos);

            // Store in GPU VRAM KV Cache
            LaunchKvStore(l, pos);

            // Multi-Head / Grouped Query Attention (GQA) in VRAM
            LaunchAttention(l, pos);

            // Attention out projection
            LaunchGemv(lw.AttnOutType, _dAttnProj, _dAttnOut, lw.AttnOutWeight, _dim, _dim);

            // Residual connection: dX += dAttnProj
            LaunchVecAdd(_dX, _dAttnProj, _dim);

            // FFN pre-norm
            LaunchRmsNorm(_dX, lw.FfnNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // SwiGLU FFN projections on GPU
            LaunchGemv(lw.FfnGateType, _dGate, _dNormX, lw.FfnGateWeight, _dim, _ffnDim);
            LaunchGemv(lw.FfnUpType, _dUp, _dNormX, lw.FfnUpWeight, _dim, _ffnDim);
            LaunchSwiglu(_dGate, _dUp, _dFfnAct, _ffnDim);
            LaunchGemv(lw.FfnDownType, _dFfnOut, _dFfnAct, lw.FfnDownWeight, _ffnDim, _dim);

            // Residual connection: dX += dFfnOut
            LaunchVecAdd(_dX, _dFfnOut, _dim);
        }

        if (computeLogits)
        {
            // Final RMSNorm
            LaunchRmsNorm(_dX, _dOutNormWeight, _dNormX, _dim, _weights.RmsNormEps);

            // Output projection (LM Head) on GPU
            LaunchGemv(_weights.OutType, _dLogits, _dNormX, _dOutWeight, _dim, _weights.VocabSize);

            // Copy logits from GPU to CPU
            fixed (float* pLogits = logits)
            {
                _gpu.CopyToHost((IntPtr)pLogits, _dLogits, (nuint)(_weights.VocabSize * sizeof(float)));
            }
        }
    }

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

        void* pQ = &dQ;
        void* pK = &dK;
        void* pHQ = &nHeadsQ;
        void* pHKv = &nHeadsKv;
        void* pHD = &headDim;
        void* pPos = &pos;
        void* pFB = &freqBase;
        void* pFS = &freqScale;
        void*[] args = [pQ, pK, pHQ, pHKv, pHD, pPos, pFB, pFS];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnRope,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(rope)");
        }
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
        void* pKC = &dKCache;
        void* pVC = &dVCache;
        void* pK = &dK;
        void* pV = &dV;
        void* pHKv = &nHeadsKv;
        void* pHD = &headDim;
        void* pMax = &maxSeq;
        void* pPos = &pos;
        void*[] args = [pKC, pVC, pK, pV, pHKv, pHD, pMax, pPos];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnKvCacheStore,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(kv_cache_store)");
        }
    }

    private void LaunchAttention(int layer, int pos)
    {
        uint gridSize = (uint)_nHeads;
        uint blockSize = (uint)_headDim;

        IntPtr dKCache = _dKeyCache[layer];
        IntPtr dVCache = _dValCache[layer];
        IntPtr dQ = _dQ;
        IntPtr dOut = _dAttnOut;
        IntPtr dScores = _dScoresBuf;
        int nHeadsQ = _nHeads;
        int nHeadsKv = _nHeadsKv;
        int headDim = _headDim;
        int maxSeq = _maxSeqLen;
        float attnScale = _attnScale;

        void* pQ = &dQ;
        void* pKC = &dKCache;
        void* pVC = &dVCache;
        void* pOut = &dOut;
        void* pScores = &dScores;
        void* pHQ = &nHeadsQ;
        void* pHKv = &nHeadsKv;
        void* pHD = &headDim;
        void* pMax = &maxSeq;
        void* pPos = &pos;
        void* pScale = &attnScale;
        void*[] args = [pQ, pKC, pVC, pOut, pScores, pHQ, pHKv, pHD, pMax, pPos, pScale];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttentionGqa,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(attention_gqa)");
        }
    }

    private void LaunchGemv(GgufType type, IntPtr dY, IntPtr dX, IntPtr dW, int kCols, int mRows)
    {
        IntPtr fn = type switch
        {
            GgufType.Q4_K => _fnGemvQ4K,
            GgufType.Q6_K => _fnGemvQ6K,
            _ => throw new NotSupportedException($"GPU GEMV does not support type {type}")
        };

        uint blockSize = 128;
        uint numWarps = 4;
        uint gridSize = (uint)((mRows + (int)numWarps - 1) / (int)numWarps);

        void* pDY = &dY;
        void* pDX = &dX;
        void* pDW = &dW;
        void* pK = &kCols;
        void* pM = &mRows;
        void*[] args = [pDY, pDX, pDW, pK, pM];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                fn,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(gemv)");
        }
    }

    private void LaunchRmsNorm(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, float eps)
    {
        uint blockSize = 256;
        uint gridSize = 1;

        void* pDX = &dX;
        void* pDW = &dWeight;
        void* pDDst = &dDst;
        void* pSize = &size;
        void* pEps = &eps;
        void*[] args = [pDX, pDW, pDDst, pSize, pEps];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnRmsNorm,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(rms_norm)");
        }
    }

    private void LaunchSwiglu(IntPtr dGate, IntPtr dUp, IntPtr dDst, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void* pDGate = &dGate;
        void* pDUp = &dUp;
        void* pDDst = &dDst;
        void* pSize = &size;
        void*[] args = [pDGate, pDUp, pDDst, pSize];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnSwiglu,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(swiglu)");
        }
    }

    private void LaunchAddBias(IntPtr dY, IntPtr dBias, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void* pDY = &dY;
        void* pDBias = &dBias;
        void* pSize = &size;
        void*[] args = [pDY, pDBias, pSize];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAddBias,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(add_bias)");
        }
    }

    private void LaunchVecAdd(IntPtr dA, IntPtr dB, int size)
    {
        uint blockSize = 256;
        uint gridSize = (uint)((size + (int)blockSize - 1) / (int)blockSize);

        void* pDA = &dA;
        void* pDB = &dB;
        void* pSize = &size;
        void*[] args = [pDA, pDB, pSize];

        fixed (void** pArgs = args)
        {
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnVecAdd,
                gridSize, 1, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(vec_add)");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gpu.FreeDevice(_dX);
            _gpu.FreeDevice(_dNormX);
            _gpu.FreeDevice(_dQ);
            _gpu.FreeDevice(_dK);
            _gpu.FreeDevice(_dV);
            _gpu.FreeDevice(_dAttnOut);
            _gpu.FreeDevice(_dAttnProj);
            _gpu.FreeDevice(_dGate);
            _gpu.FreeDevice(_dUp);
            _gpu.FreeDevice(_dFfnAct);
            _gpu.FreeDevice(_dFfnOut);
            _gpu.FreeDevice(_dLogits);
            _gpu.FreeDevice(_dScoresBuf);
            _gpu.FreeDevice(_dOutNormWeight);
            _gpu.FreeDevice(_dOutWeight);

            if (_dKeyCache != null)
            {
                for (int l = 0; l < _dKeyCache.Length; l++)
                {
                    if (_dKeyCache[l] != IntPtr.Zero) _gpu.FreeDevice(_dKeyCache[l]);
                    if (_dValCache[l] != IntPtr.Zero) _gpu.FreeDevice(_dValCache[l]);
                }
            }

            if (_layerWeights != null)
            {
                for (int l = 0; l < _layerWeights.Length; l++)
                {
                    var lw = _layerWeights[l];
                    if (lw != null)
                    {
                        _gpu.FreeDevice(lw.AttnNormWeight);
                        _gpu.FreeDevice(lw.QWeight);
                        _gpu.FreeDevice(lw.QBias);
                        _gpu.FreeDevice(lw.KWeight);
                        _gpu.FreeDevice(lw.KBias);
                        _gpu.FreeDevice(lw.VWeight);
                        _gpu.FreeDevice(lw.VBias);
                        _gpu.FreeDevice(lw.AttnOutWeight);
                        _gpu.FreeDevice(lw.FfnNormWeight);
                        _gpu.FreeDevice(lw.FfnGateWeight);
                        _gpu.FreeDevice(lw.FfnUpWeight);
                        _gpu.FreeDevice(lw.FfnDownWeight);
                    }
                }
            }

            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~Qwen2GpuModel()
    {
        Dispose();
    }
}
