namespace Glacier.Inference.Embedding;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Portable-SIMD weight matrix for the Gemma 2 encoder. Q8_0 weights stay quantised in memory
/// (int8 quants + FP32 scales, 4x smaller than FP32) and are multiplied against FP32 activations.
/// The kernels use <see cref="Vector128"/>/<see cref="Vector256"/>/<see cref="Vector512"/> cross-platform
/// intrinsics so the JIT emits SSE/AVX2/AVX-512 on x64 and NEON on ARM64 (Apple Silicon, Snapdragon, Cortex-X).
/// </summary>
internal sealed unsafe class Gemma2Linear : IDisposable
{
    private float* _f32;       // non-null for float-weight matrices
    private sbyte* _quants;    // [rows, in]
    private float* _scales;    // [rows, in/32]

    public int In { get; }
    public int Out { get; }
    public bool IsQuantized => _quants != null;

    internal float* F32 => _f32;
    internal sbyte* Quants => _quants;
    internal float* Scales => _scales;

    private Gemma2Linear(int inDim, int outDim) { In = inDim; Out = outDim; }

    public static Gemma2Linear Load(GgufFile gguf, string name, int inDim, int outDim, bool keepQuantized = true)
    {
        var info = Gemma2Tensors.Require(gguf, name, (ulong)inDim, (ulong)outDim);
        uint type = (uint)info.Type;
        byte* src = gguf.GetTensorPointer(info);
        var lin = new Gemma2Linear(inDim, outDim);

        if (type == Gemma2Tensors.Q8_0 && keepQuantized && inDim % 32 == 0)
        {
            int blocks = inDim / 32;
            lin._quants = (sbyte*)NativeMemory.AlignedAlloc((nuint)((long)outDim * inDim), 64);
            lin._scales = (float*)NativeMemory.AlignedAlloc((nuint)((long)outDim * blocks * sizeof(float)), 64);
            for (int r = 0; r < outDim; r++)
            {
                byte* row = src + r * (long)blocks * 34;
                for (int b = 0; b < blocks; b++)
                {
                    byte* blk = row + b * 34;
                    lin._scales[(long)r * blocks + b] = (float)Unsafe.ReadUnaligned<Half>(blk);
                    Buffer.MemoryCopy(blk + 2, lin._quants + (long)r * inDim + b * 32, 32, 32);
                }
            }
        }
        else
        {
            lin._f32 = (float*)NativeMemory.AlignedAlloc((nuint)((long)outDim * inDim * sizeof(float)), 64);
            for (int r = 0; r < outDim; r++)
                Gemma2Tensors.DequantRow(src, type, r, inDim, lin._f32 + (long)r * inDim);
        }
        return lin;
    }

    /// <summary>out[t, o - rowStart] = dot(W[o], in[t]) for o in [rowStart, rowStart+rowCount), t in [0, n).</summary>
    public void MatMul(float* input, int n, float* output, int rowStart = 0, int rowCount = -1)
    {
        if (rowCount < 0) rowCount = Out - rowStart;
        int inDim = In;
        nint pin = (nint)input, pout = (nint)output;
        Gemma2Linear self = this;

        // Cache blocking: each task owns a small band of weight rows and sweeps the tokens in tiles, so a tile of
        // activations stays hot in L1/L2 while it is reused by every row of the band (the naive order re-streams
        // the whole activation matrix from memory once per weight row).
        int chunk = Math.Clamp(rowCount / (Environment.ProcessorCount * 2), 4, 64);
        int chunks = (rowCount + chunk - 1) / chunk;
        const int TokTile = 32;
        Parallel.For(0, chunks, c =>
        {
            int start = c * chunk, end = Math.Min(rowCount, start + chunk);
            float* inp = (float*)pin;
            float* outp = (float*)pout;
            for (int tb = 0; tb < n; tb += TokTile)
            {
                int te = Math.Min(n, tb + TokTile);
                for (int o = start; o < end; o++)
                {
                    int wrow = rowStart + o;
                    if (self._quants != null)
                    {
                        sbyte* q = self._quants + (long)wrow * inDim;
                        float* s = self._scales + (long)wrow * (inDim / 32);
                        for (int t = tb; t < te; t++)
                            outp[(long)t * rowCount + o] = Simd.DotQ8(q, s, inp + (long)t * inDim, inDim);
                    }
                    else
                    {
                        float* w = self._f32 + (long)wrow * inDim;
                        for (int t = tb; t < te; t++)
                            outp[(long)t * rowCount + o] = Simd.Dot(w, inp + (long)t * inDim, inDim);
                    }
                }
            }
        });
    }
    public void Dispose()
    {
        if (_f32 != null) { NativeMemory.AlignedFree(_f32); _f32 = null; }
        if (_quants != null) { NativeMemory.AlignedFree(_quants); _quants = null; }
        if (_scales != null) { NativeMemory.AlignedFree(_scales); _scales = null; }
    }
}

/// <summary>Cross-platform SIMD primitives (x64 SSE/AVX2/AVX-512 and ARM64 NEON via the same source).</summary>
internal static unsafe class Simd
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(float* a, float* b, int n)
    {
        int i = 0;
        float sum = 0f;
        if (Vector512.IsHardwareAccelerated && n >= 16)
        {
            var acc = Vector512<float>.Zero;
            for (; i <= n - 16; i += 16)
                acc += Vector512.Load(a + i) * Vector512.Load(b + i);
            sum = Vector512.Sum(acc);
        }
        else if (Vector256.IsHardwareAccelerated && n >= 8)
        {
            var a0 = Vector256<float>.Zero;
            var a1 = Vector256<float>.Zero;
            for (; i <= n - 16; i += 16)
            {
                a0 += Vector256.Load(a + i) * Vector256.Load(b + i);
                a1 += Vector256.Load(a + i + 8) * Vector256.Load(b + i + 8);
            }
            var acc = a0 + a1;
            for (; i <= n - 8; i += 8)
                acc += Vector256.Load(a + i) * Vector256.Load(b + i);
            sum = Vector256.Sum(acc);
        }
        else if (Vector128.IsHardwareAccelerated && n >= 4)
        {
            var a0 = Vector128<float>.Zero;
            var a1 = Vector128<float>.Zero;
            var a2 = Vector128<float>.Zero;
            var a3 = Vector128<float>.Zero;
            for (; i <= n - 16; i += 16)
            {
                a0 += Vector128.Load(a + i) * Vector128.Load(b + i);
                a1 += Vector128.Load(a + i + 4) * Vector128.Load(b + i + 4);
                a2 += Vector128.Load(a + i + 8) * Vector128.Load(b + i + 8);
                a3 += Vector128.Load(a + i + 12) * Vector128.Load(b + i + 12);
            }
            var acc = (a0 + a1) + (a2 + a3);
            for (; i <= n - 4; i += 4)
                acc += Vector128.Load(a + i) * Vector128.Load(b + i);
            sum = Vector128.Sum(acc);
        }
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Block-wise Q8 dot: sum over blocks of scale times the int8/float dot of 32 elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ8(sbyte* q, float* scales, float* x, int n)
    {
        int blocks = n / 32;
        float total = 0f;
        if (Vector256.IsHardwareAccelerated)
        {
            for (int b = 0; b < blocks; b++)
            {
                sbyte* qb = q + b * 32;
                float* xb = x + b * 32;
                var q8 = Vector256.Load(qb);
                var (lo16, hi16) = Vector256.Widen(q8);
                var (i0, i1) = Vector256.Widen(lo16);
                var (i2, i3) = Vector256.Widen(hi16);
                var acc = Vector256.ConvertToSingle(i0) * Vector256.Load(xb);
                acc += Vector256.ConvertToSingle(i1) * Vector256.Load(xb + 8);
                acc += Vector256.ConvertToSingle(i2) * Vector256.Load(xb + 16);
                acc += Vector256.ConvertToSingle(i3) * Vector256.Load(xb + 24);
                total += Vector256.Sum(acc) * scales[b];
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            for (int b = 0; b < blocks; b++)
            {
                sbyte* qb = q + b * 32;
                float* xb = x + b * 32;
                var acc = Vector128<float>.Zero;
                for (int h = 0; h < 2; h++)
                {
                    var q8 = Vector128.Load(qb + h * 16);
                    var (lo16, hi16) = Vector128.Widen(q8);
                    var (i0, i1) = Vector128.Widen(lo16);
                    var (i2, i3) = Vector128.Widen(hi16);
                    float* xh = xb + h * 16;
                    acc += Vector128.ConvertToSingle(i0) * Vector128.Load(xh);
                    acc += Vector128.ConvertToSingle(i1) * Vector128.Load(xh + 4);
                    acc += Vector128.ConvertToSingle(i2) * Vector128.Load(xh + 8);
                    acc += Vector128.ConvertToSingle(i3) * Vector128.Load(xh + 12);
                }
                total += Vector128.Sum(acc) * scales[b];
            }
        }
        else
        {
            for (int b = 0; b < blocks; b++)
            {
                float s = 0f;
                for (int i = 0; i < 32; i++) s += q[b * 32 + i] * x[b * 32 + i];
                total += s * scales[b];
            }
        }
        return total;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot64(float* a, float* b)
    {
        if (Vector128.IsHardwareAccelerated)
        {
            var acc0 = Vector128.Load(a) * Vector128.Load(b);
            var acc1 = Vector128.Load(a + 4) * Vector128.Load(b + 4);
            var acc2 = Vector128.Load(a + 8) * Vector128.Load(b + 8);
            var acc3 = Vector128.Load(a + 12) * Vector128.Load(b + 12);
            var acc4 = Vector128.Load(a + 16) * Vector128.Load(b + 16);
            var acc5 = Vector128.Load(a + 20) * Vector128.Load(b + 20);
            var acc6 = Vector128.Load(a + 24) * Vector128.Load(b + 24);
            var acc7 = Vector128.Load(a + 28) * Vector128.Load(b + 28);
            var acc8 = Vector128.Load(a + 32) * Vector128.Load(b + 32);
            var acc9 = Vector128.Load(a + 36) * Vector128.Load(b + 36);
            var acc10 = Vector128.Load(a + 40) * Vector128.Load(b + 40);
            var acc11 = Vector128.Load(a + 44) * Vector128.Load(b + 44);
            var acc12 = Vector128.Load(a + 48) * Vector128.Load(b + 48);
            var acc13 = Vector128.Load(a + 52) * Vector128.Load(b + 52);
            var acc14 = Vector128.Load(a + 56) * Vector128.Load(b + 56);
            var acc15 = Vector128.Load(a + 60) * Vector128.Load(b + 60);

            var s0 = (acc0 + acc1) + (acc2 + acc3);
            var s1 = (acc4 + acc5) + (acc6 + acc7);
            var s2 = (acc8 + acc9) + (acc10 + acc11);
            var s3 = (acc12 + acc13) + (acc14 + acc15);

            return Vector128.Sum((s0 + s1) + (s2 + s3));
        }
        return Dot(a, b, 64);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void WeightedSum64(float* dst, float* scores, float* vBase, int vStride, int P, float inv)
    {
        if (Vector128.IsHardwareAccelerated)
        {
            var a0 = Vector128<float>.Zero; var a1 = Vector128<float>.Zero;
            var a2 = Vector128<float>.Zero; var a3 = Vector128<float>.Zero;
            var a4 = Vector128<float>.Zero; var a5 = Vector128<float>.Zero;
            var a6 = Vector128<float>.Zero; var a7 = Vector128<float>.Zero;
            var a8 = Vector128<float>.Zero; var a9 = Vector128<float>.Zero;
            var a10 = Vector128<float>.Zero; var a11 = Vector128<float>.Zero;
            var a12 = Vector128<float>.Zero; var a13 = Vector128<float>.Zero;
            var a14 = Vector128<float>.Zero; var a15 = Vector128<float>.Zero;

            for (int j = 0; j < P; j++)
            {
                var vw = Vector128.Create(scores[j] * inv);
                float* vh = vBase + (long)j * vStride;
                a0 += vw * Vector128.Load(vh);
                a1 += vw * Vector128.Load(vh + 4);
                a2 += vw * Vector128.Load(vh + 8);
                a3 += vw * Vector128.Load(vh + 12);
                a4 += vw * Vector128.Load(vh + 16);
                a5 += vw * Vector128.Load(vh + 20);
                a6 += vw * Vector128.Load(vh + 24);
                a7 += vw * Vector128.Load(vh + 28);
                a8 += vw * Vector128.Load(vh + 32);
                a9 += vw * Vector128.Load(vh + 36);
                a10 += vw * Vector128.Load(vh + 40);
                a11 += vw * Vector128.Load(vh + 44);
                a12 += vw * Vector128.Load(vh + 48);
                a13 += vw * Vector128.Load(vh + 52);
                a14 += vw * Vector128.Load(vh + 56);
                a15 += vw * Vector128.Load(vh + 60);
            }

            a0.Store(dst);
            a1.Store(dst + 4);
            a2.Store(dst + 8);
            a3.Store(dst + 12);
            a4.Store(dst + 16);
            a5.Store(dst + 20);
            a6.Store(dst + 24);
            a7.Store(dst + 28);
            a8.Store(dst + 32);
            a9.Store(dst + 36);
            a10.Store(dst + 40);
            a11.Store(dst + 44);
            a12.Store(dst + 48);
            a13.Store(dst + 52);
            a14.Store(dst + 56);
            a15.Store(dst + 60);
        }
        else
        {
            for (int d = 0; d < 64; d++) dst[d] = 0;
            for (int j = 0; j < P; j++)
            {
                float w = scores[j] * inv;
                float* vh = vBase + (long)j * vStride;
                for (int d = 0; d < 64; d++) dst[d] += w * vh[d];
            }
        }
    }
}

