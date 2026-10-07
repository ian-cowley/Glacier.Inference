namespace Glacier.Inference.Embedding;

using System;
using System.Numerics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Image preprocessing for EmbeddingGemma 2 (matches the Hugging Face <c>Gemma4ImageProcessor</c>):
/// aspect-preserving "floor to a multiple of 48" target size under a soft-token budget, PIL-style bicubic resize,
/// 1/255 rescale and 16x16 patchification in (py, px, channel) order.
/// </summary>
public static class Gemma4ImagePreprocessor
{
    public const int PatchSize = 16;
    public const int PoolKernel = 3;

    /// <summary>Soft-token budget for still images (HF default 280); video frames use 140.</summary>
    public const int DefaultMaxSoftTokens = 280;

    /// <summary>Target pixel size for an input image (multiples of 48 so the 3x3 pooling grid is exact).</summary>
    public static (int Height, int Width) TargetSize(int height, int width, int maxSoftTokens = DefaultMaxSoftTokens)
    {
        int maxPatches = maxSoftTokens * PoolKernel * PoolKernel;
        double factor = Math.Sqrt((double)maxPatches * PatchSize * PatchSize / ((double)height * width));
        int side = PoolKernel * PatchSize; // 48
        int th = (int)Math.Floor(factor * height / side) * side;
        int tw = (int)Math.Floor(factor * width / side) * side;
        int maxSide = maxPatches / (PoolKernel * PoolKernel) * side;
        if (th == 0 && tw == 0) { th = side; tw = side; }
        else if (th == 0) { th = side; tw = Math.Min((int)Math.Floor((double)width / height) * side, maxSide); if (tw == 0) tw = side; }
        else if (tw == 0) { tw = side; th = Math.Min((int)Math.Floor((double)height / width) * side, maxSide); if (th == 0) th = side; }
        return (th, tw);
    }

    /// <summary>
    /// Converts packed RGB8 (row-major, 3 bytes per pixel) into patches [P, 768] in [0,1] plus the patch grid.
    /// </summary>
    public static float[] Preprocess(ReadOnlySpan<byte> rgb, int width, int height, int maxSoftTokens,
        out int gridH, out int gridW)
    {
        if (rgb.Length != width * height * 3) throw new ArgumentException("rgb must contain width*height*3 bytes.", nameof(rgb));
        var (th, tw) = TargetSize(height, width, maxSoftTokens);
        byte[] pix = (th == height && tw == width) ? rgb.ToArray() : ResizeBicubic(rgb, width, height, tw, th);

        gridH = th / PatchSize; gridW = tw / PatchSize;
        var patches = new float[(long)gridH * gridW * PatchSize * PatchSize * 3];
        int pd = PatchSize * PatchSize * 3;
        for (int gy = 0; gy < gridH; gy++)
            for (int gx = 0; gx < gridW; gx++)
            {
                int p = gy * gridW + gx;
                for (int py = 0; py < PatchSize; py++)
                    for (int px = 0; px < PatchSize; px++)
                        for (int c = 0; c < 3; c++)
                            patches[(long)p * pd + (py * PatchSize + px) * 3 + c] =
                                pix[((gy * PatchSize + py) * tw + gx * PatchSize + px) * 3 + c] / 255f;
            }
        return patches;
    }

    // ---------------------------------------------------------------- PIL-compatible bicubic (a = -0.5, antialiased)

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    }

    private static (int[] start, int[] count, double[][] w) Coefficients(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = 2.0 * filterScale;
        var starts = new int[outSize]; var counts = new int[outSize]; var ws = new double[outSize][];
        for (int i = 0; i < outSize; i++)
        {
            double center = (i + 0.5) * scale;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inSize, (int)(center + support + 0.5));
            int n = xmax - xmin;
            var w = new double[n];
            double sum = 0;
            for (int j = 0; j < n; j++) { w[j] = Cubic((j + xmin - center + 0.5) / filterScale); sum += w[j]; }
            if (sum != 0) for (int j = 0; j < n; j++) w[j] /= sum;
            starts[i] = xmin; counts[i] = n; ws[i] = w;
        }
        return (starts, counts, ws);
    }

    private static byte Clip(double v) => (byte)Math.Clamp((int)Math.Round(v, MidpointRounding.AwayFromZero), 0, 255);

    /// <summary>Separable resize: horizontal pass then vertical pass, each rounded to uint8 as Pillow does.</summary>
    public static byte[] ResizeBicubic(ReadOnlySpan<byte> src, int sw, int sh, int dw, int dh)
    {
        var (hx0, hxn, hw) = Coefficients(sw, dw);
        var (vy0, vyn, vw) = Coefficients(sh, dh);
        byte[] srcArr = src.ToArray();
        var tmp = new byte[sh * dw * 3];
        Parallel.For(0, sh, y =>
        {
            for (int x = 0; x < dw; x++)
                for (int c = 0; c < 3; c++)
                {
                    double acc = 0; var w = hw[x]; int s = hx0[x];
                    for (int j = 0; j < hxn[x]; j++) acc += w[j] * srcArr[(y * sw + s + j) * 3 + c];
                    tmp[(y * dw + x) * 3 + c] = Clip(acc);
                }
        });
        var dst = new byte[dh * dw * 3];
        Parallel.For(0, dh, y =>
        {
            var w = vw[y]; int s = vy0[y];
            for (int x = 0; x < dw; x++)
                for (int c = 0; c < 3; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < vyn[y]; j++) acc += w[j] * tmp[((s + j) * dw + x) * 3 + c];
                    dst[(y * dw + x) * 3 + c] = Clip(acc);
                }
        });
        return dst;
    }
}

/// <summary>
/// Gemma 4 vision encoder + multimodal embedder as used by EmbeddingGemma 2 (weights from the <c>mmproj</c> GGUF).
/// 16 pre/post-norm transformer layers over 16x16 patches with learned x/y position tables and axial 2-D RoPE,
/// 3x3 average pooling, then <c>RMSNorm → Linear(768→512)</c> producing soft tokens for the text tower.
/// </summary>
internal sealed unsafe class Gemma4VisionTower : IDisposable
{
    private const int Hidden = 768, Heads = 12, HeadDim = 64, Ffn = 3072, OutDim = 512, PosTable = 10240;
    private const float Eps = 1e-6f;

    private sealed class Block
    {
        public required Gemma2Linear Q, K, V, O, Gate, Up, Down;
        public required float[] Ln1, QNorm, KNorm, PostAttn, Ln2, PostFfn;
    }

    private readonly GgufFile _gguf;
    private readonly int _layers;
    private readonly float[] _patchW;       // [768 out][768 in] in (py, px, c) order
    private readonly float* _posX, _posY;   // [10240, 768] each (inside the mapped GGUF)
    private readonly Block[] _blocks;
    private readonly Gemma2Linear _proj;
    private readonly float[] _invFreq = new float[16];
    private readonly object _gate = new();

    public static bool IsVisionMmproj(GgufFile g) => g.Tensors.ContainsKey("v.patch_embd.weight") && g.Tensors.ContainsKey("mm.input_projection.weight");

    public Gemma4VisionTower(GgufFile mmproj)
    {
        _gguf = mmproj;
        _layers = (int)mmproj.GetMetadataUInt32("clip.vision.block_count", 16);

        // ggml dims [16,16,3,768] = (px fastest, py, c, out); HF flat order is (py, px, c).
        var pe = Gemma2Tensors.Require(mmproj, "v.patch_embd.weight");
        float* raw = (float*)mmproj.GetTensorPointer(pe);
        _patchW = new float[Hidden * Hidden];
        for (int o = 0; o < Hidden; o++)
            for (int c = 0; c < 3; c++)
                for (int py = 0; py < 16; py++)
                    for (int px = 0; px < 16; px++)
                        _patchW[o * Hidden + (py * 16 + px) * 3 + c] = raw[(((long)o * 3 + c) * 16 + py) * 16 + px];

        var pt = Gemma2Tensors.Require(mmproj, "v.position_embd.weight");
        if ((uint)pt.Type != Gemma2Tensors.F32) throw new NotSupportedException("Vision position table must be F32.");
        _posX = (float*)mmproj.GetTensorPointer(pt);
        _posY = _posX + (long)PosTable * Hidden;

        _blocks = new Block[_layers];
        for (int i = 0; i < _layers; i++)
        {
            string p = $"v.blk.{i}.";
            _blocks[i] = new Block
            {
                Q = Gemma2Linear.Load(mmproj, p + "attn_q.weight", Hidden, Hidden),
                K = Gemma2Linear.Load(mmproj, p + "attn_k.weight", Hidden, Hidden),
                V = Gemma2Linear.Load(mmproj, p + "attn_v.weight", Hidden, Hidden),
                O = Gemma2Linear.Load(mmproj, p + "attn_out.weight", Hidden, Hidden),
                Gate = Gemma2Linear.Load(mmproj, p + "ffn_gate.weight", Hidden, Ffn),
                Up = Gemma2Linear.Load(mmproj, p + "ffn_up.weight", Hidden, Ffn),
                Down = Gemma2Linear.Load(mmproj, p + "ffn_down.weight", Ffn, Hidden),
                Ln1 = Gemma2Tensors.LoadVector(mmproj, p + "ln1.weight", Hidden),
                QNorm = Gemma2Tensors.LoadVector(mmproj, p + "attn_q_norm.weight", HeadDim),
                KNorm = Gemma2Tensors.LoadVector(mmproj, p + "attn_k_norm.weight", HeadDim),
                PostAttn = Gemma2Tensors.LoadVector(mmproj, p + "attn_post_norm.weight", Hidden),
                Ln2 = Gemma2Tensors.LoadVector(mmproj, p + "ln2.weight", Hidden),
                PostFfn = Gemma2Tensors.LoadVector(mmproj, p + "ffn_post_norm.weight", Hidden),
            };
        }
        _proj = Gemma2Linear.Load(mmproj, "mm.input_projection.weight", Hidden, OutDim);
        for (int i = 0; i < 16; i++) _invFreq[i] = (float)Math.Pow(100.0, -i / 16.0);
    }

    /// <summary>Checkpoints for validation (null unless requested).</summary>
    public sealed class Trace
    {
        public float[]? PatchEmbed, Layer0, EncoderOut, TowerOut;
    }

    /// <summary>patches [P, 768] in [0,1] on a gridH x gridW patch grid (both multiples of 3) → soft tokens [P/9, 512].</summary>
    public float[] Encode(float[] patches, int gridH, int gridW, Trace? trace = null)
    {
        if (gridH % 3 != 0 || gridW % 3 != 0) throw new ArgumentException("Patch grid must be a multiple of 3 in both directions.");
        int P = gridH * gridW;
        if (patches.Length != (long)P * Hidden) throw new ArgumentException("patches length does not match grid.");
        if (gridH > PosTable || gridW > PosTable) throw new ArgumentException("Image too large for the position table.");

        lock (_gate)
        {
            var h = new float[(long)P * Hidden];
            var x = new float[(long)P * Hidden];
            for (long i = 0; i < x.LongLength; i++) x[i] = 2f * patches[i] - 1f;
            fixed (float* px = x, ph = h, pw = _patchW)
            {
                // h = x · Wᵀ  (reuse the FP32 kernel via a temporary Gemma2Linear-less loop)
                int pCount = P;
                nint ipx = (nint)px, iph = (nint)ph, ipw = (nint)pw;
                Parallel.For(0, pCount, t =>
                {
                    float* xi = (float*)ipx + (long)t * Hidden;
                    float* hi = (float*)iph + (long)t * Hidden;
                    float* wb = (float*)ipw;
                    for (int o = 0; o < Hidden; o++) hi[o] = Simd.Dot(wb + (long)o * Hidden, xi, Hidden);
                    int col = t % gridW, row = t / gridW;
                    float* ax = _posX + (long)col * Hidden, ay = _posY + (long)row * Hidden;
                    for (int d = 0; d < Hidden; d++) hi[d] += ax[d] + ay[d];
                });
            }
            if (trace != null) trace.PatchEmbed = (float[])h.Clone();

            var normed = new float[(long)P * Hidden];
            var tmp = new float[(long)P * Hidden];
            var q = new float[(long)P * Hidden];
            var k = new float[(long)P * Hidden];
            var v = new float[(long)P * Hidden];
            var att = new float[(long)P * Hidden];
            var gate = new float[(long)P * Ffn];
            var up = new float[(long)P * Ffn];

            for (int l = 0; l < _layers; l++)
            {
                var b = _blocks[l];
                RmsRows(h, b.Ln1, normed, P, Hidden);
                fixed (float* pn = normed, pq = q, pk = k, pv = v)
                {
                    b.Q.MatMul(pn, P, pq); b.K.MatMul(pn, P, pk); b.V.MatMul(pn, P, pv);
                }
                Parallel.For(0, P, t =>
                {
                    int col = t % gridW, row = t / gridW;
                    for (int hd = 0; hd < Heads; hd++)
                    {
                        long o = (long)t * Hidden + hd * HeadDim;
                        RmsHead(q, o, b.QNorm); Rope2D(q, o, col, row);
                        RmsHead(k, o, b.KNorm); Rope2D(k, o, col, row);
                        RmsHead(v, o, null);
                    }
                });
                Attention(q, k, v, att, P);
                fixed (float* pa = att, pt = tmp) b.O.MatMul(pa, P, pt);
                RmsRows(tmp, b.PostAttn, tmp, P, Hidden);
                AddInto(h, tmp);

                RmsRows(h, b.Ln2, normed, P, Hidden);
                fixed (float* pn = normed, pg = gate, pu = up)
                {
                    b.Gate.MatMul(pn, P, pg); b.Up.MatMul(pn, P, pu);
                }
                long fc = (long)P * Ffn;
                Parallel.For(0, (int)((fc + 65535) / 65536), blk =>
                {
                    long s = blk * 65536L, e = Math.Min(fc, s + 65536);
                    for (long i = s; i < e; i++) gate[i] = Gemma2CpuBackend.GeluTanh(gate[i]) * up[i];
                });
                fixed (float* pg = gate, pt = tmp) b.Down.MatMul(pg, P, pt);
                RmsRows(tmp, b.PostFfn, tmp, P, Hidden);
                AddInto(h, tmp);

                if (l == 0 && trace != null) trace.Layer0 = (float[])h.Clone();
            }
            if (trace != null) trace.EncoderOut = (float[])h.Clone();

            // 3x3 average pooling (row-major over the pooled grid), then * sqrt(hidden).
            int ph2 = gridH / 3, pw2 = gridW / 3, N = ph2 * pw2;
            var pooled = new float[(long)N * Hidden];
            float sc = MathF.Sqrt(Hidden) / 9f;
            for (int oy = 0; oy < ph2; oy++)
                for (int ox = 0; ox < pw2; ox++)
                {
                    long dst = ((long)oy * pw2 + ox) * Hidden;
                    for (int dy = 0; dy < 3; dy++)
                        for (int dx = 0; dx < 3; dx++)
                        {
                            long src = ((long)(oy * 3 + dy) * gridW + ox * 3 + dx) * Hidden;
                            for (int d = 0; d < Hidden; d++) pooled[dst + d] += h[src + d];
                        }
                    for (int d = 0; d < Hidden; d++) pooled[dst + d] *= sc;
                }
            if (trace != null) trace.TowerOut = (float[])pooled.Clone();

            // embedder: RMSNorm (no weight) → Linear 768 → 512
            RmsRows(pooled, null, pooled, N, Hidden);
            var soft = new float[(long)N * OutDim];
            fixed (float* pp = pooled, ps = soft) _proj.MatMul(pp, N, ps);
            return soft;
        }
    }

    // ------------------------------------------------------------------ kernels

    private static void RmsRows(float[] src, float[]? w, float[] dst, int rows, int len)
    {
        Parallel.For(0, rows, r =>
        {
            long o = (long)r * len;
            double ss = 0;
            for (int i = 0; i < len; i++) ss += (double)src[o + i] * src[o + i];
            float s = (float)(1.0 / Math.Sqrt(ss / len + Eps));
            if (w == null) for (int i = 0; i < len; i++) dst[o + i] = src[o + i] * s;
            else for (int i = 0; i < len; i++) dst[o + i] = src[o + i] * s * w[i];
        });
    }

    private static void RmsHead(float[] a, long o, float[]? w)
    {
        double ss = 0;
        for (int i = 0; i < HeadDim; i++) ss += (double)a[o + i] * a[o + i];
        float s = (float)(1.0 / Math.Sqrt(ss / HeadDim + Eps));
        if (w == null) for (int i = 0; i < HeadDim; i++) a[o + i] *= s;
        else for (int i = 0; i < HeadDim; i++) a[o + i] = a[o + i] * s * w[i];
    }

    /// <summary>Axial 2-D RoPE: dims 0..31 rotate with the x (column) position, dims 32..63 with y (row); NeoX pairing in each half.</summary>
    private void Rope2D(float[] a, long o, int x, int y)
    {
        for (int half = 0; half < 2; half++)
        {
            int pos = half == 0 ? x : y;
            long b = o + half * 32;
            for (int j = 0; j < 16; j++)
            {
                float ang = pos * _invFreq[j];
                float c = MathF.Cos(ang), s = MathF.Sin(ang);
                float u = a[b + j], w = a[b + j + 16];
                a[b + j] = u * c - w * s;
                a[b + j + 16] = w * c + u * s;
            }
        }
    }

    private static void AddInto(float[] dst, float[] src)
    {
        int vs = Vector<float>.Count; long n = dst.LongLength;
        Parallel.For(0, (int)((n + 65535) / 65536), blk =>
        {
            long s = blk * 65536L, e = Math.Min(n, s + 65536), i = s;
            for (; i <= e - vs; i += vs)
                (new Vector<float>(dst, (int)i) + new Vector<float>(src, (int)i)).CopyTo(dst, (int)i);
            for (; i < e; i++) dst[i] += src[i];
        });
    }

    /// <summary>Full bidirectional attention, scale 1.0 (Q/K are RMS-normalised), 12 MHA heads of dim 64.</summary>
    private static void Attention(float[] q, float[] k, float[] v, float[] o, int P)
    {
        Parallel.For(0, P * Heads, () => new float[P], (idx, loopState, scores) =>
        {
            int t = idx / Heads, h = idx % Heads;
            fixed (float* pq = q, pk = k, pv = v, po = o, pscores = scores)
            {
                float* qh = pq + (long)t * Hidden + h * HeadDim;
                float* kBase = pk + h * HeadDim;
                float max = float.NegativeInfinity;
                for (int j = 0; j < P; j++)
                {
                    float s = Simd.Dot64(qh, kBase + (long)j * Hidden);
                    pscores[j] = s; if (s > max) max = s;
                }
                float sum = 0;
                for (int j = 0; j < P; j++) { float e = MathF.Exp(pscores[j] - max); pscores[j] = e; sum += e; }
                float inv = 1f / sum;
                float* oh = po + (long)t * Hidden + h * HeadDim;
                Simd.WeightedSum64(oh, pscores, pv + h * HeadDim, Hidden, P, inv);
            }
            return scores;
        }, _ => {});
    }

    public void Dispose()
    {
        _proj.Dispose();
        foreach (var b in _blocks)
        { b.Q.Dispose(); b.K.Dispose(); b.V.Dispose(); b.O.Dispose(); b.Gate.Dispose(); b.Up.Dispose(); b.Down.Dispose(); }
        _gguf.Dispose();
    }
}
