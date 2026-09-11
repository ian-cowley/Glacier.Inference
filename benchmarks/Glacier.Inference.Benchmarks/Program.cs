namespace Glacier.Inference.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Glacier.Inference.Quant;

[MemoryDiagnoser]
public unsafe class KernelBenchmarks
{
    private const int Dim = 3584;
    private float[] _x = new float[Dim];
    private float[] _weight = new float[Dim];
    private float[] _dst = new float[Dim];

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < Dim; i++)
        {
            _x[i] = (i % 7) - 3.0f;
            _weight[i] = 1.0f;
        }
    }

    [Benchmark]
    public void RMSNorm_Benchmark()
    {
        fixed (float* pX = _x, pW = _weight, pDst = _dst)
        {
            QuantKernels.RMSNorm(pX, pW, pDst, Dim, 1e-5f);
        }
    }

    [Benchmark]
    public float VecDotF32_Benchmark()
    {
        fixed (float* pX = _x, pW = _weight)
        {
            return QuantKernels.VecDotF32(pX, pW, Dim);
        }
    }
}

public static class Program
{
    public static void Main(string[] args)
    {
        BenchmarkRunner.Run<KernelBenchmarks>();
    }
}
