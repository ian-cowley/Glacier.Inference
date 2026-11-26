namespace Glacier.Inference.Benchmarks;

using System;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

/// <summary>
/// Benchmarks multithreaded GEMV (matrix-vector multiplication) and batched GEMM
/// across all CPU cores for model hidden dimensions.
/// </summary>
[MemoryDiagnoser]
public unsafe class GemvBenchmarks : IDisposable
{
    [Params(3584, 4096)]
    public int Dim { get; set; }

    private byte[] _weightQ4K = null!;
    private byte[] _weightQ80 = null!;
    private byte[] _weightTrl2 = null!;
    private byte[] _weightTrl3 = null!;
    private Half[] _trlS16 = null!;
    private float[] _trlPre = null!;
    private float[] _trlPost = null!;
    private TrellisWeightBinding _trl2Binding = null!;
    private TrellisWeightBinding _trl3Binding = null!;
    private GCHandle _trl2Handle;
    private GCHandle _trl3Handle;
    private GCHandle _trlS16Handle;
    private GCHandle _trlPreHandle;
    private GCHandle _trlPostHandle;
    private float[] _x = null!;
    private float[] _y = null!;
    private float[] _xBatch8 = null!;
    private float[] _yBatch8 = null!;
    private float[] _xBatch32 = null!;
    private float[] _yBatch32 = null!;

    [GlobalSetup]
    public void Setup()
    {
        _x = new float[Dim];
        _y = new float[Dim];
        for (int i = 0; i < Dim; i++) _x[i] = 1.0f + (i % 7) * 0.1f;

        // Weights: [Dim, Dim]
        int rowBytesQ4K = (int)GgufTypes.GetRowBytes(GgufType.Q4_K, Dim);
        _weightQ4K = new byte[Dim * rowBytesQ4K];
        for (int i = 0; i < _weightQ4K.Length; i++) _weightQ4K[i] = (byte)(i & 0xFF);

        int rowBytesQ80 = (int)GgufTypes.GetRowBytes(GgufType.Q8_0, Dim);
        _weightQ80 = new byte[Dim * rowBytesQ80];
        for (int i = 0; i < _weightQ80.Length; i++) _weightQ80[i] = (byte)(i & 0x7F);

        int kTiles = Dim / 16;
        int nTiles = Dim / 16;
        _weightTrl2 = new byte[nTiles * kTiles * 64];
        _weightTrl3 = new byte[nTiles * kTiles * 96];
        for (int i = 0; i < _weightTrl2.Length; i++) _weightTrl2[i] = (byte)((i * 73) & 0xFF);
        for (int i = 0; i < _weightTrl3.Length; i++) _weightTrl3[i] = (byte)((i * 137) & 0xFF);

        _trlS16 = new Half[kTiles];
        Array.Fill(_trlS16, (Half)0.05f);
        _trlPre = new float[Dim];
        _trlPost = new float[Dim];
        Array.Fill(_trlPre, 1.0f);
        Array.Fill(_trlPost, 1.0f);

        _trl2Handle = GCHandle.Alloc(_weightTrl2, GCHandleType.Pinned);
        _trl3Handle = GCHandle.Alloc(_weightTrl3, GCHandleType.Pinned);
        _trlS16Handle = GCHandle.Alloc(_trlS16, GCHandleType.Pinned);
        _trlPreHandle = GCHandle.Alloc(_trlPre, GCHandleType.Pinned);
        _trlPostHandle = GCHandle.Alloc(_trlPost, GCHandleType.Pinned);

        int[] desc = [1, 128, 1, 1, 128, 1, 0];
        _trl2Binding = new TrellisWeightBinding
        {
            Type = GgufType.TRL2,
            CodeData = (byte*)_trl2Handle.AddrOfPinnedObject(),
            K = Dim,
            N = Dim,
            S16 = (ushort*)_trlS16Handle.AddrOfPinnedObject(),
            Pre = (float*)_trlPreHandle.AddrOfPinnedObject(),
            Post = (float*)_trlPostHandle.AddrOfPinnedObject(),
            Descriptor = desc
        };
        _trl3Binding = new TrellisWeightBinding
        {
            Type = GgufType.TRL3,
            CodeData = (byte*)_trl3Handle.AddrOfPinnedObject(),
            K = Dim,
            N = Dim,
            S16 = (ushort*)_trlS16Handle.AddrOfPinnedObject(),
            Pre = (float*)_trlPreHandle.AddrOfPinnedObject(),
            Post = (float*)_trlPostHandle.AddrOfPinnedObject(),
            Descriptor = desc
        };

        // Batched inputs
        _xBatch8 = new float[8 * Dim];
        _yBatch8 = new float[8 * Dim];
        for (int i = 0; i < _xBatch8.Length; i++) _xBatch8[i] = 1.0f + (i % 11) * 0.05f;

        _xBatch32 = new float[32 * Dim];
        _yBatch32 = new float[32 * Dim];
        for (int i = 0; i < _xBatch32.Length; i++) _xBatch32[i] = 1.0f + (i % 13) * 0.05f;
    }

    [Benchmark]
    public void MatVecMul_TRL2()
    {
        fixed (float* pX = _x, pY = _y)
        {
            TrellisKernels.MatVecMul(_trl2Binding, pX, pY);
        }
    }

    [Benchmark]
    public void MatVecMul_TRL3()
    {
        fixed (float* pX = _x, pY = _y)
        {
            TrellisKernels.MatVecMul(_trl3Binding, pX, pY);
        }
    }

    [Benchmark]
    public void MatVecMul_Q4_K()
    {
        fixed (byte* pW = _weightQ4K)
        fixed (float* pX = _x, pY = _y)
        {
            QuantKernels.MatVecMul(GgufType.Q4_K, pW, pX, pY, Dim, Dim);
        }
    }

    [Benchmark]
    public void MatVecMul_Q8_0()
    {
        fixed (byte* pW = _weightQ80)
        fixed (float* pX = _x, pY = _y)
        {
            QuantKernels.MatVecMul(GgufType.Q8_0, pW, pX, pY, Dim, Dim);
        }
    }

    [Benchmark]
    public void MatMulBatch_Q4_K_Batch8()
    {
        fixed (byte* pW = _weightQ4K)
        fixed (float* pX = _xBatch8, pY = _yBatch8)
        {
            QuantKernels.MatMulBatch(GgufType.Q4_K, pW, pX, pY, Dim, Dim, 8);
        }
    }

    [Benchmark]
    public void MatMulBatch_Q4_K_Batch32()
    {
        fixed (byte* pW = _weightQ4K)
        fixed (float* pX = _xBatch32, pY = _yBatch32)
        {
            QuantKernels.MatMulBatch(GgufType.Q4_K, pW, pX, pY, Dim, Dim, 32);
        }
    }

    public void Dispose()
    {
        if (_trl2Handle.IsAllocated) _trl2Handle.Free();
        if (_trl3Handle.IsAllocated) _trl3Handle.Free();
        if (_trlS16Handle.IsAllocated) _trlS16Handle.Free();
        if (_trlPreHandle.IsAllocated) _trlPreHandle.Free();
        if (_trlPostHandle.IsAllocated) _trlPostHandle.Free();
    }
}
