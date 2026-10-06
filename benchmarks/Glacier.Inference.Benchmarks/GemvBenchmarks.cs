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

        // Batched inputs
        _xBatch8 = new float[8 * Dim];
        _yBatch8 = new float[8 * Dim];
        for (int i = 0; i < _xBatch8.Length; i++) _xBatch8[i] = 1.0f + (i % 11) * 0.05f;

        _xBatch32 = new float[32 * Dim];
        _yBatch32 = new float[32 * Dim];
        for (int i = 0; i < _xBatch32.Length; i++) _xBatch32[i] = 1.0f + (i % 13) * 0.05f;
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
    }
}
