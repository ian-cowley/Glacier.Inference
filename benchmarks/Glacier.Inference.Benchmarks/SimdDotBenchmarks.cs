namespace Glacier.Inference.Benchmarks;

using System;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Glacier.Inference.Quant;

/// <summary>
/// Micro-benchmarks for pure C# quantized SIMD dot products (AVX-512 / AVX2 / ARM NEON)
/// evaluating bandwidth (GB/s) and execution latency across quantized formats.
/// </summary>
[MemoryDiagnoser]
public unsafe class SimdDotBenchmarks : IDisposable
{
    [Params(3584, 4096)]
    public int Dim { get; set; }

    private float[] _x = null!;
    private float[] _xSums = null!;
    private float[] _weightF32 = null!;
    private float[] _dst = null!;

    private byte[] _q4kBytes = null!;
    private byte[] _q80Bytes = null!;
    private byte[] _q6kBytes = null!;
    private byte[] _iq4xsBytes = null!;
    private byte[] _mxfp4Bytes = null!;

    [GlobalSetup]
    public void Setup()
    {
        _x = new float[Dim];
        _xSums = new float[Dim / 32];
        _weightF32 = new float[Dim];
        _dst = new float[Dim];

        for (int i = 0; i < Dim; i++)
        {
            _x[i] = (i % 7) - 3.0f;
            _weightF32[i] = 1.0f + (i % 5) * 0.1f;
        }

        fixed (float* pX = _x, pSums = _xSums)
        {
            QuantKernels.ComputeBlockSums32(pX, pSums, Dim);
        }

        // Q4_K: Dim / 256 super-blocks (144 bytes each)
        int nbQ4K = Dim / 256;
        _q4kBytes = new byte[nbQ4K * 144];
        fixed (byte* p = _q4kBytes)
        {
            var pB = (BlockQ4_K*)p;
            for (int i = 0; i < nbQ4K; i++)
            {
                pB[i].Delta = (Half)1.0f;
                pB[i].DeltaMin = (Half)0.5f;
                for (int s = 0; s < 12; s++) pB[i].Scales[s] = (byte)(10 + (s % 3));
                for (int q = 0; q < 128; q++) pB[i].Qs[q] = (byte)((q & 0x0F) | (((q + 1) & 0x0F) << 4));
            }
        }

        // Q8_0: Dim / 32 blocks (34 bytes each)
        int nbQ80 = Dim / 32;
        _q80Bytes = new byte[nbQ80 * 34];
        fixed (byte* p = _q80Bytes)
        {
            var pB = (BlockQ8_0*)p;
            for (int i = 0; i < nbQ80; i++)
            {
                pB[i].Delta = (Half)0.5f;
                for (int q = 0; q < 32; q++) pB[i].Qs[q] = (sbyte)((q % 7) - 3);
            }
        }

        // Q6_K: Dim / 256 super-blocks (210 bytes each)
        int nbQ6K = Dim / 256;
        _q6kBytes = new byte[nbQ6K * 210];
        fixed (byte* p = _q6kBytes)
        {
            var pB = (BlockQ6_K*)p;
            for (int i = 0; i < nbQ6K; i++)
            {
                pB[i].Delta = (Half)0.25f;
                for (int q = 0; q < 128; q++) pB[i].Ql[q] = (byte)(q * 3);
                for (int q = 0; q < 64; q++) pB[i].Qh[q] = (byte)(q % 4);
                for (int s = 0; s < 16; s++) pB[i].Scales[s] = (sbyte)((s % 5) - 2);
            }
        }

        // IQ4_XS: Dim / 256 super-blocks (136 bytes each)
        int nbIQ4XS = Dim / 256;
        _iq4xsBytes = new byte[nbIQ4XS * 136];
        fixed (byte* p = _iq4xsBytes)
        {
            var pB = (BlockIQ4_XS*)p;
            for (int i = 0; i < nbIQ4XS; i++)
            {
                pB[i].Delta = (Half)1.0f;
                pB[i].ScalesH = 0x5555;
                for (int s = 0; s < 4; s++) pB[i].ScalesL[s] = 0x22;
                for (int q = 0; q < 128; q++) pB[i].Qs[q] = (byte)(q & 0x0F);
            }
        }

        // MXFP4: Dim / 32 blocks (17 bytes each)
        int nbMXFP4 = Dim / 32;
        _mxfp4Bytes = new byte[nbMXFP4 * 17];
        fixed (byte* p = _mxfp4Bytes)
        {
            var pB = (BlockMXFP4*)p;
            for (int i = 0; i < nbMXFP4; i++)
            {
                pB[i].Scale = 127;
                for (int q = 0; q < 16; q++) pB[i].Qs[q] = (byte)(q | (q << 4));
            }
        }
    }

    [Benchmark(Baseline = true)]
    public float VecDotF32()
    {
        fixed (float* pX = _x, pW = _weightF32)
        {
            return QuantKernels.VecDotF32(pW, pX, Dim);
        }
    }

    [Benchmark]
    public float VecDotQ4_K()
    {
        fixed (byte* pW = _q4kBytes)
        fixed (float* pX = _x, pSums = _xSums)
        {
            return QuantKernels.VecDotQ4_K((BlockQ4_K*)pW, pX, pSums, Dim);
        }
    }

    [Benchmark]
    public float VecDotQ8_0()
    {
        fixed (byte* pW = _q80Bytes)
        fixed (float* pX = _x)
        {
            return QuantKernels.VecDotQ8_0((BlockQ8_0*)pW, pX, Dim);
        }
    }

    [Benchmark]
    public float VecDotQ6_K()
    {
        fixed (byte* pW = _q6kBytes)
        fixed (float* pX = _x)
        {
            return QuantKernels.VecDotQ6_K((BlockQ6_K*)pW, pX, Dim);
        }
    }

    [Benchmark]
    public float VecDotIQ4_XS()
    {
        fixed (byte* pW = _iq4xsBytes)
        fixed (float* pX = _x)
        {
            return QuantKernels.VecDotIQ4_XS((BlockIQ4_XS*)pW, pX, Dim);
        }
    }

    [Benchmark]
    public float VecDotMXFP4()
    {
        fixed (byte* pW = _mxfp4Bytes)
        fixed (float* pX = _x)
        {
            return QuantKernels.VecDotMXFP4((BlockMXFP4*)pW, pX, Dim);
        }
    }

    [Benchmark]
    public void RMSNorm()
    {
        fixed (float* pX = _x, pW = _weightF32, pDst = _dst)
        {
            QuantKernels.RMSNorm(pX, pW, pDst, Dim, 1e-5f);
        }
    }

    public void Dispose()
    {
    }
}
