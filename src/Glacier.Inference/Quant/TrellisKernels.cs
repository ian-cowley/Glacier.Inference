namespace Glacier.Inference.Quant;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Binding of a NariQ trellis-coded weight tensor (<c>GGML_TYPE_TRL1..TRL6</c>) to its
/// side tensors (<c>trellis_s16</c>, <c>trellis_pre</c>, <c>trellis_post</c>, optional
/// <c>trellis_pin</c>/<c>trellis_pout</c>) and descriptor metadata (<c>trellis.desc.&lt;w&gt;</c>).
/// </summary>
public sealed unsafe class TrellisWeightBinding
{
    public required GgufType Type { get; init; }
    public required byte* CodeData { get; init; }
    public required int K { get; init; }
    public required int N { get; init; }
    public required ushort* S16 { get; init; }      // F16 [K/16]
    public required float* Pre { get; init; }       // F32 [K]
    public required float* Post { get; init; }      // F32 [N]
    public int* Pin { get; init; }                  // I32 [K], optional
    public int* Pout { get; init; }                 // I32 [N], optional
    public required int[] Descriptor { get; init; } // [rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts, part_0..]
}

/// <summary>
/// Frequency-banded NariQ trellis linear layer (e.g., <c>output.band00..band15.weight</c>
/// + <c>output.vocab_gather</c>), allowing mixed bitrates across token frequency bands.
/// </summary>
public sealed unsafe class TrellisBandedWeight
{
    public required TrellisWeightBinding[] Bands { get; init; }
    public required int* VocabGather { get; init; } // I32 [TotalN]: bandRow = VocabGather[vocabId]
    public required int K { get; init; }
    public required int TotalN { get; init; }
}

/// <summary>
/// Process-wide pointer registry mapping raw GGUF weight pointers (<c>byte*</c>) to their
/// <see cref="TrellisWeightBinding"/> or <see cref="TrellisBandedWeight"/> so <c>QuantKernels.MatVecMul</c>
/// and <c>QuantKernels.MatMulBatch</c> dispatch NariQ trellis tensors seamlessly without
/// changing model runner call sites.
/// </summary>
public static unsafe class TrellisWeightRegistry
{
    private static readonly ConcurrentDictionary<nint, TrellisWeightBinding> _bindings = new();
    private static readonly ConcurrentDictionary<nint, TrellisBandedWeight> _banded = new();

    public static void Register(TrellisWeightBinding binding)
    {
        ArgumentNullException.ThrowIfNull(binding);
        _bindings[(nint)binding.CodeData] = binding;
    }

    public static void RegisterBanded(byte* keyPtr, TrellisBandedWeight banded)
    {
        ArgumentNullException.ThrowIfNull(banded);
        _banded[(nint)keyPtr] = banded;
    }

    public static bool TryGet(byte* weightData, out TrellisWeightBinding? binding) =>
        _bindings.TryGetValue((nint)weightData, out binding);

    public static bool TryGetBanded(byte* weightData, out TrellisBandedWeight? banded) =>
        _banded.TryGetValue((nint)weightData, out banded);

    public static void Unregister(byte* weightData)
    {
        _bindings.TryRemove((nint)weightData, out _);
        _banded.TryRemove((nint)weightData, out _);
    }

    public static void Clear()
    {
        _bindings.Clear();
        _banded.Clear();
    }
}

/// <summary>
/// Bit-exact NariQ Trellis-Coded Quantization kernels (<c>GGML_TYPE_TRL1..TRL6</c>,
/// <c>nariq/FORMAT.md</c> v1):
/// <list type="bullet">
///   <item>16x16 output-tile-major circular bitstream decode (<c>0x83DCD12Du</c> INT8 grid in [-126, 126]).</item>
///   <item>Normalized block Sylvester FWHT (<c>kind=1</c>) and whole-slice Paley Hadamard (<c>kind=2</c>, m in {1, 12, 20, 68}).</item>
///   <item>Coded linear forward pass: <c>y = gather_pout(post * T_rows^T(W_rotated * T_cols(pre * gather_pin(x))))</c>.</item>
///   <item>Full dense dequantization (<see cref="DequantizeMatrix"/>) and single-row embedding decode (<see cref="DecodeEmbeddingRow"/>).</item>
/// </list>
/// </summary>
public static unsafe class TrellisKernels
{
    public const int MaxParts = 8;

    /// <summary>
    /// Builds the 64 KB packed signed-int8 lookup table (16,384 uint32 words, 4 int8 values per word)
    /// mapping every 16-bit trellis state (0..65535) to its exact integer grid value in [-126, +126].
    /// </summary>
    public static uint[] BuildInt8LutPacked()
    {
        var lut = new uint[16384];
        for (uint s = 0; s < 65536; s++)
        {
            uint xv = s * 0x83DCD12Du;
            int v = (int)(((xv >> 2) & 63u) + ((xv >> 10) & 63u) + ((xv >> 18) & 63u) + ((xv >> 26) & 63u)) - 126;
            lut[s >> 2] |= (uint)(byte)(sbyte)v << (int)((s & 3u) << 3);
        }
        return lut;
    }

    private static readonly sbyte[] PaleyH12 = MakePaley(12);
    private static readonly sbyte[] PaleyH20 = MakePaley(20);
    private static readonly sbyte[] PaleyH68 = MakePaley(68);

    private static int Chi(int d, int q)
    {
        d = ((d % q) + q) % q;
        if (d == 0) return 0;
        for (int i = 1; i < q; i++)
        {
            if ((i * i) % q == d) return 1;
        }
        return -1;
    }

    private static sbyte[] MakePaley(int m)
    {
        var h = new sbyte[m * m];
        int q = m - 1;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : Chi(j - i, q));
                h[i * m + j] = (sbyte)(s + (i == j ? 1 : 0));
            }
        }
        return h;
    }

    private static sbyte[] GetPaley(int m) => m switch
    {
        12 => PaleyH12,
        20 => PaleyH20,
        68 => PaleyH68,
        _ => throw new ArgumentOutOfRangeException(nameof(m), m, "Unsupported Paley Hadamard factor.")
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPow2(int v) => v > 0 && (v & (v - 1)) == 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float InvSqrt(int n) => (float)Math.Pow(n, -0.5);

    public static int GetHadFullM(int len)
    {
        if (len >= 1 && IsPow2(len)) return 1;
        if (len >= 12 && (len % 12) == 0 && IsPow2(len / 12)) return 12;
        if (len >= 20 && (len % 20) == 0 && IsPow2(len / 20)) return 20;
        if (len >= 68 && (len % 68) == 0 && IsPow2(len / 68)) return 68;
        return 0;
    }

    private struct PartsLayout
    {
        public int Count;
        public fixed int Start[MaxParts];
        public fixed int Shard[MaxParts];
        public fixed int Cum[MaxParts + 1];
    }

    private static bool TryMakeParts(out PartsLayout p, int n, ReadOnlySpan<int> parts, int shards)
    {
        p = default;
        if (shards < 1 || (!parts.IsEmpty && (parts.Length < 1 || parts.Length > MaxParts)))
            return false;

        p.Count = parts.IsEmpty ? 1 : parts.Length;
        int start = 0, cum = 0;
        for (int i = 0; i < p.Count; i++)
        {
            int size = parts.IsEmpty ? n : parts[i];
            if (size <= 0 || (size % shards) != 0)
                return false;
            p.Start[i] = start;
            p.Shard[i] = size / shards;
            p.Cum[i] = cum;
            start += size;
            cum += size / shards;
        }
        p.Cum[p.Count] = cum;
        return start == n;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int StoredIndex(in PartsLayout p, int r, int l)
    {
        int i = 0;
        while (i + 1 < p.Count && l >= p.Cum[i + 1])
            i++;
        return p.Start[i] + r * p.Shard[i] + (l - p.Cum[i]);
    }

    /// <summary>
    /// Validates a NariQ trellis descriptor against matrix dimensions <paramref name="k"/> (in) and <paramref name="n"/> (out).
    /// </summary>
    public static bool IsDescriptorSupported(int k, int n, ReadOnlySpan<int> desc)
    {
        if (desc.Length < 7 || desc[6] < 0 || desc[6] > MaxParts || desc.Length != 7 + desc[6] || (k % 16) != 0 || (n % 16) != 0)
            return false;

        static bool SideOk(int len, int kind, int block, int shards, ReadOnlySpan<int> parts) => kind switch
        {
            0 => true,
            1 => IsPow2(block) && block >= 2 && (len % block) == 0,
            2 => TryMakeParts(out _, len, parts, shards) && GetHadFullM(len / shards) != 0,
            _ => false
        };

        ReadOnlySpan<int> rowParts = desc[6] > 0 ? desc.Slice(7, desc[6]) : ReadOnlySpan<int>.Empty;
        return SideOk(n, desc[0], desc[1], desc[2], rowParts) &&
               SideOk(k, desc[3], desc[4], desc[5], ReadOnlySpan<int>.Empty);
    }

    /// <summary>
    /// Normalized Sylvester FWHT butterflies on <paramref name="block"/> entries in FP32, then scale by <paramref name="c"/>.
    /// Bit-identical to <c>libtrellis tq_rht</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Fwht(float* u, int block, float c)
    {
        for (int h = 1; h < block; h <<= 1)
        {
            int step = h << 1;
            for (int i = 0; i < block; i += step)
            {
                for (int j = i; j < i + h; j++)
                {
                    float a = u[j];
                    float b = u[j + h];
                    u[j] = a + b;
                    u[j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < block; i++)
            u[i] *= c;
    }

    /// <summary>
    /// Normalized blockwise Sylvester Hadamard transform of <paramref name="n"/> floats in place.
    /// </summary>
    public static void Rht(float* v, int n, int block)
    {
        if (!IsPow2(block) || (n % block) != 0)
            throw new ArgumentException($"Invalid RHT parameters: n={n}, block={block}");
        float c = InvSqrt(block);
        for (int b0 = 0; b0 < n; b0 += block)
            Fwht(v + b0, block, c);
    }

    private static void CrossPaley(float* input, float* output, int m, int run, bool transpose, float c)
    {
        sbyte[] h = GetPaley(m);
        for (int a = 0; a < m; a++)
        {
            float* o = output + a * run;
            new Span<float>(o, run).Clear();
            for (int b = 0; b < m; b++)
            {
                int s = transpose ? h[b * m + a] : h[a * m + b];
                float* x = input + b * run;
                if (s > 0)
                {
                    for (int j = 0; j < run; j++) o[j] += x[j];
                }
                else
                {
                    for (int j = 0; j < run; j++) o[j] -= x[j];
                }
            }
            for (int j = 0; j < run; j++) o[j] *= c;
        }
    }

    public static bool HadFullSlice(
        float* v, int n, ReadOnlySpan<int> parts, int shards, int r, bool inverse,
        float* pre, float* post, float* tmp)
    {
        if (!TryMakeParts(out PartsLayout p, n, parts, shards) || r < 0 || r >= shards)
            return false;

        int len = n / shards;
        int m = GetHadFullM(len);
        if (m == 0) return false;

        int run = len / m;
        float cRun = InvSqrt(run);
        float cM = InvSqrt(m);

        float* a = tmp;
        float* b = tmp + len;
        for (int l = 0; l < len; l++)
        {
            int idx = StoredIndex(in p, r, l);
            a[l] = pre != null ? v[idx] * pre[idx] : v[idx];
        }

        float* res = a;
        if (m == 1)
        {
            Fwht(a, len, cRun);
        }
        else if (!inverse)
        {
            for (int i = 0; i < m; i++)
                Fwht(a + i * run, run, cRun);
            CrossPaley(a, b, m, run, transpose: false, cM);
            res = b;
        }
        else
        {
            CrossPaley(a, b, m, run, transpose: true, cM);
            for (int i = 0; i < m; i++)
                Fwht(b + i * run, run, cRun);
            res = b;
        }

        for (int l = 0; l < len; l++)
        {
            int idx = StoredIndex(in p, r, l);
            v[idx] = post != null ? res[l] * post[idx] : res[l];
        }
        return true;
    }

    private static bool ApplySide(
        float* v, int n, int kind, int block, int shards, ReadOnlySpan<int> parts, bool inverse, float* tmp)
    {
        if (kind == 0) return true;
        if (kind == 1)
        {
            Rht(v, n, block);
            return true;
        }
        if (kind == 2)
        {
            for (int r = 0; r < shards; r++)
            {
                if (!HadFullSlice(v, n, parts, shards, r, inverse, null, null, tmp))
                    return false;
            }
            return true;
        }
        return false;
    }

    /// <summary>
    /// Decodes <paramref name="nRows"/> rows (multiple of 16) starting at <paramref name="n0"/> (multiple of 16)
    /// of a <c>GGML_TYPE_TRL1..TRL6</c> weight matrix into <paramref name="output"/> (<c>[nRows, k]</c> FP32),
    /// multiplying each 16-column tile by its FP16 <paramref name="s16"/> scale.
    /// </summary>
    public static void DecodeRows(
        GgufType type, byte* data, int k, int n0, int nRows, ushort* s16, float* output)
    {
        if (!GgufTypes.IsTrellis(type))
            throw new ArgumentException($"Not a trellis type: {type}", nameof(type));
        if ((k % 16) != 0 || (n0 % 16) != 0 || (nRows % 16) != 0)
            throw new ArgumentException($"Trellis dimensions must be multiples of 16 (k={k}, n0={n0}, nRows={nRows}).");

        int bits = GgufTypes.GetTrellisBits(type);
        int nw = 8 * bits;
        int total = 256 * bits;
        int kt = k / 16;
        uint* words = (uint*)data;

        int jStart = n0 / 16;
        int jEnd = (n0 + nRows) / 16;

        for (int j = jStart; j < jEnd; j++)
        {
            for (int i = 0; i < kt; i++)
            {
                uint* t = words + ((long)j * kt + i) * nw;
                float s = s16 != null ? (float)BitConverter.UInt16BitsToHalf(s16[i]) : 1.0f;
                int baseRow = 16 * j - n0;
                int baseCol = 16 * i;

                for (int p = 0; p < 256; p++)
                {
                    int start = ((p + 1) * bits - 16 + total) % total;
                    int w = start >> 5;
                    int o = start & 31;
                    ulong pair = ((ulong)t[w] << 32) | t[(w + 1) % nw];
                    uint state = (uint)(pair >> (48 - o)) & 0xFFFFu;
                    uint x = unchecked(state * 0x83DCD12Du);
                    int v = (int)(((x >> 2) & 63u) + ((x >> 10) & 63u) + ((x >> 18) & 63u) + ((x >> 26) & 63u)) - 126;

                    int a = p >> 3;
                    int jj = p & 7;
                    int row = (a >> 2) + 8 * ((jj >> 2) & 1);
                    int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);

                    output[(baseRow + row) * k + baseCol + col] = v * s;
                }
            }
        }
    }

    /// <summary>
    /// Computes <c>y = gather_pout(post * T_rows^T(W_rotated * T_cols(pre * gather_pin(x))))</c>
    /// for a single vector <paramref name="x"/> (<c>[K]</c>) producing <paramref name="y"/> (<c>[N]</c>).
    /// Streams 16x16 tiles directly without materializing the dense <c>[N, K]</c> matrix.
    /// </summary>
    public static void MatVecMul(TrellisWeightBinding b, float* x, float* y)
    {
        int k = b.K;
        int n = b.N;
        int[] desc = b.Descriptor;
        int bits = GgufTypes.GetTrellisBits(b.Type);
        int nw = 8 * bits;
        int total = 256 * bits;
        int kt = k / 16;
        int nt = n / 16;

        float[] xRotArr = ArrayPool<float>.Shared.Rent(k);
        float[] yRotArr = ArrayPool<float>.Shared.Rent(n);
        float[] tmpArr = ArrayPool<float>.Shared.Rent(2 * Math.Max(k, n));

        try
        {
            fixed (float* xRot = xRotArr)
            fixed (float* yRot = yRotArr)
            fixed (float* tmp = tmpArr)
            {
                // 1. x_rot = T_cols(pre * gather_pin(x))
                if (b.Pin != null)
                {
                    for (int i = 0; i < k; i++)
                        xRot[i] = x[b.Pin[i]] * b.Pre[i];
                }
                else
                {
                    for (int i = 0; i < k; i++)
                        xRot[i] = x[i] * b.Pre[i];
                }

                if (!ApplySide(xRot, k, desc[3], desc[4], desc[5], ReadOnlySpan<int>.Empty, inverse: false, tmp))
                    throw new InvalidOperationException("Unsupported NariQ column transform in descriptor.");

                // 2. y_rot = W_rotated * x_rot (parallelized across 16-row output tiles)
                uint* words = (uint*)b.CodeData;
                ushort* s16 = b.S16;
                float* xRotPtr = xRot;
                float* yRotPtr = yRot;

                Action<int> computeTileRow = j =>
                {
                    float* acc = stackalloc float[16];
                    float* tileAcc = stackalloc float[16];
                    for (int r = 0; r < 16; r++) acc[r] = 0f;

                    for (int i = 0; i < kt; i++)
                    {
                        uint* t = words + ((long)j * kt + i) * nw;
                        float s = s16 != null ? (float)BitConverter.UInt16BitsToHalf(s16[i]) : 1.0f;
                        float* xTile = xRotPtr + 16 * i;

                        for (int r = 0; r < 16; r++) tileAcc[r] = 0f;

                        for (int p = 0; p < 256; p++)
                        {
                            int start = ((p + 1) * bits - 16 + total) % total;
                            int w = start >> 5;
                            int o = start & 31;
                            ulong pair = ((ulong)t[w] << 32) | t[(w + 1) % nw];
                            uint state = (uint)(pair >> (48 - o)) & 0xFFFFu;
                            uint xv = unchecked(state * 0x83DCD12Du);
                            int v = (int)(((xv >> 2) & 63u) + ((xv >> 10) & 63u) + ((xv >> 18) & 63u) + ((xv >> 26) & 63u)) - 126;

                            int a = p >> 3;
                            int jj = p & 7;
                            int row = (a >> 2) + 8 * ((jj >> 2) & 1);
                            int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);

                            tileAcc[row] += v * xTile[col];
                        }

                        for (int r = 0; r < 16; r++)
                            acc[r] += tileAcc[r] * s;
                    }

                    int rowOut = 16 * j;
                    for (int r = 0; r < 16; r++)
                        yRotPtr[rowOut + r] = acc[r];
                };

                if (nt < Environment.ProcessorCount * 2)
                {
                    for (int j = 0; j < nt; j++) computeTileRow(j);
                }
                else
                {
                    Parallel.For(0, nt, computeTileRow);
                }

                // 3. y = gather_pout(post * T_rows^T(y_rot))
                ReadOnlySpan<int> rowParts = desc[6] > 0 ? desc.AsSpan(7, desc[6]) : ReadOnlySpan<int>.Empty;
                if (!ApplySide(yRot, n, desc[0], desc[1], desc[2], rowParts, inverse: true, tmp))
                    throw new InvalidOperationException("Unsupported NariQ row transform in descriptor.");

                for (int i = 0; i < n; i++)
                    yRot[i] *= b.Post[i];

                if (b.Pout != null)
                {
                    for (int i = 0; i < n; i++)
                        y[i] = yRot[b.Pout[i]];
                }
                else
                {
                    Buffer.MemoryCopy(yRot, y, (long)n * sizeof(float), (long)n * sizeof(float));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xRotArr);
            ArrayPool<float>.Shared.Return(yRotArr);
            ArrayPool<float>.Shared.Return(tmpArr);
        }
    }

    /// <summary>
    /// Full dense dequantization into <paramref name="output"/> (<c>[N, K]</c> row-major FP32),
    /// bit-equivalent to <c>ggml_trellis_dequantize</c>. Used for parity verification and
    /// small/reference materialization.
    /// </summary>
    public static bool DequantizeMatrix(TrellisWeightBinding b, float* output)
    {
        int k = b.K;
        int n = b.N;
        int[] desc = b.Descriptor;
        if (!GgufTypes.IsTrellis(b.Type) || !IsDescriptorSupported(k, n, desc) || b.Pre == null || b.Post == null)
            return false;

        if (b.Pin != null)
        {
            for (int j = 0; j < k; j++)
                if (b.Pin[j] < 0 || b.Pin[j] >= k) return false;
        }
        if (b.Pout != null)
        {
            for (int i = 0; i < n; i++)
                if (b.Pout[i] < 0 || b.Pout[i] >= n) return false;
        }

        DecodeRows(b.Type, b.CodeData, k, 0, n, b.S16, output);

        int nMax = Math.Max(k, n);
        float[] tmpArr = ArrayPool<float>.Shared.Rent(2 * nMax);
        float[] rowArr = ArrayPool<float>.Shared.Rent(k);
        float[] colsArr = ArrayPool<float>.Shared.Rent(16 * n);
        try
        {
            fixed (float* tmp = tmpArr)
            fixed (float* row = rowArr)
            fixed (float* cols = colsArr)
            {
                for (int r = 0; r < n; r++)
                {
                    float* w = output + (long)r * k;
                    if (!ApplySide(w, k, desc[3], desc[4], desc[5], ReadOnlySpan<int>.Empty, inverse: true, tmp))
                        return false;
                    for (int j = 0; j < k; j++)
                        w[j] *= b.Pre[j];
                    if (b.Pin != null)
                    {
                        new Span<float>(row, k).Clear();
                        for (int j = 0; j < k; j++)
                            row[b.Pin[j]] += w[j];
                        Buffer.MemoryCopy(row, w, (long)k * sizeof(float), (long)k * sizeof(float));
                    }
                }

                ReadOnlySpan<int> rowParts = desc[6] > 0 ? desc.AsSpan(7, desc[6]) : ReadOnlySpan<int>.Empty;
                for (int c0 = 0; c0 < k; c0 += 16)
                {
                    for (int r = 0; r < n; r++)
                        for (int c = 0; c < 16; c++)
                            cols[c * n + r] = output[(long)r * k + c0 + c];

                    for (int c = 0; c < 16; c++)
                    {
                        if (!ApplySide(cols + c * n, n, desc[0], desc[1], desc[2], rowParts, inverse: true, tmp))
                            return false;
                    }

                    for (int r = 0; r < n; r++)
                        for (int c = 0; c < 16; c++)
                            output[(long)r * k + c0 + c] = cols[c * n + r] * b.Post[r];
                }

                if (b.Pout != null)
                {
                    float[] rowsCopy = new float[(long)n * k];
                    fixed (float* dst = rowsCopy)
                    {
                        for (int i = 0; i < n; i++)
                            Buffer.MemoryCopy(output + (long)b.Pout[i] * k, dst + (long)i * k, (long)k * sizeof(float), (long)k * sizeof(float));
                        Buffer.MemoryCopy(dst, output, (long)n * k * sizeof(float), (long)n * k * sizeof(float));
                    }
                }
            }
            return true;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpArr);
            ArrayPool<float>.Shared.Return(rowArr);
            ArrayPool<float>.Shared.Return(colsArr);
        }
    }

    /// <summary>
    /// Computes a frequency-banded NariQ linear projection (e.g., 16-band mixed-bitrate LM head)
    /// and scatters/gathers outputs back into vocabulary order via <c>VocabGather</c>.
    /// </summary>
    public static void MatVecMulBanded(TrellisBandedWeight banded, float* x, float* y)
    {
        int totalN = banded.TotalN;
        float[] rawArr = ArrayPool<float>.Shared.Rent(totalN);
        try
        {
            fixed (float* raw = rawArr)
            {
                int offset = 0;
                for (int i = 0; i < banded.Bands.Length; i++)
                {
                    var band = banded.Bands[i];
                    MatVecMul(band, x, raw + offset);
                    offset += band.N;
                }

                int* gather = banded.VocabGather;
                for (int v = 0; v < totalN; v++)
                {
                    y[v] = raw[gather[v]];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rawArr);
        }
    }

    /// <summary>
    /// Decodes a single embedding row <paramref name="rowIndex"/> from a NariQ trellis embedding
    /// (with no row transform, <c>rows_kind == 0</c>) into <paramref name="dest"/> (<c>[K]</c>).
    /// </summary>
    public static void DecodeEmbeddingRow(TrellisWeightBinding b, int rowIndex, float* dest)
    {
        int k = b.K;
        int[] desc = b.Descriptor;
        int codedRow = b.Pout != null ? b.Pout[rowIndex] : rowIndex;
        int tileRow0 = (codedRow / 16) * 16;
        int localRow = codedRow - tileRow0;

        float[] tile16Arr = ArrayPool<float>.Shared.Rent(16 * k);
        float[] tmpArr = ArrayPool<float>.Shared.Rent(2 * k);
        try
        {
            fixed (float* tile16 = tile16Arr)
            fixed (float* tmp = tmpArr)
            {
                DecodeRows(b.Type, b.CodeData, k, tileRow0, 16, b.S16, tile16);
                float* w = tile16 + localRow * k;
                ApplySide(w, k, desc[3], desc[4], desc[5], ReadOnlySpan<int>.Empty, inverse: true, tmp);
                float postScale = b.Post[codedRow];

                if (b.Pin != null)
                {
                    new Span<float>(dest, k).Clear();
                    for (int j = 0; j < k; j++)
                        dest[b.Pin[j]] += w[j] * b.Pre[j] * postScale;
                }
                else
                {
                    for (int j = 0; j < k; j++)
                        dest[j] = w[j] * b.Pre[j] * postScale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tile16Arr);
            ArrayPool<float>.Shared.Return(tmpArr);
        }
    }
}
