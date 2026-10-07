namespace Glacier.Inference.Embedding;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>Token-embedding gather shared by all backends (rows stay quantised in the GGUF; only needed rows are dequantised).</summary>
internal static unsafe class Gemma2Embedding
{
    /// <summary>Returns [n, hidden] embeddings scaled by sqrt(hidden), as the model's embedding layer does.</summary>
    public static float[] Gather(GgufFile gguf, Gemma2Config cfg, ReadOnlySpan<int> tokens)
    {
        var info = gguf.Tensors["token_embd.weight"];
        uint type = (uint)info.Type;
        byte* data = gguf.GetTensorPointer(info);
        int H = cfg.Hidden;
        float scale = MathF.Sqrt(H);
        var x = new float[tokens.Length * H];
        fixed (float* px = x)
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                int id = tokens[t];
                if ((uint)id >= (uint)cfg.Vocab) throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {id} out of range.");
                Gemma2Tensors.DequantRow(data, type, id, H, px + (long)t * H);
                for (int d = 0; d < H; d++) px[(long)t * H + d] *= scale;
            }
        }
        return x;
    }
}

/// <summary>
/// Multi-threaded SIMD CPU backend (x64 AVX2/AVX-512, ARM64 NEON). Quantised weights stay Q8_0 in memory and
/// per-layer embeddings are produced layer-by-layer so memory stays O(tokens x hidden).
/// </summary>
internal sealed unsafe class Gemma2CpuBackend : IGemma2Backend
{
    private sealed class Layer
    {
        public required Gemma2Linear Q, K, V, O, Gate, Up, Down, PleGate, PleProj;
        public required float[] AttnNorm, QNorm, KNorm, AttnPostNorm, FfnNorm, FfnPostNorm, PleNorm;
        public float OutScale;
    }

    private readonly GgufFile _gguf;
    private readonly Gemma2Config _cfg;
    private readonly Layer[] _layers;
    private readonly Gemma2Linear _perLayerProj;
    private readonly float[] _perLayerProjNorm;
    private readonly float[] _outputNorm;
    private readonly object _gate = new();
    private bool _disposed;

    public string Name => $"CPU SIMD ({(Vector512.IsHardwareAccelerated ? "512-bit" : Vector256.IsHardwareAccelerated ? "256-bit" : "128-bit")}, {Environment.ProcessorCount} threads, Q8_0 weights)";

    public Gemma2CpuBackend(GgufFile gguf, Gemma2Config cfg)
    {
        _gguf = gguf;
        _cfg = cfg;
        int H = cfg.Hidden;
        _perLayerProj = Gemma2Linear.Load(gguf, "per_layer_model_proj.weight", H, cfg.PleDim * cfg.Layers, keepQuantized: false);
        _perLayerProjNorm = Gemma2Tensors.LoadVector(gguf, "per_layer_proj_norm.weight", cfg.PleDim);
        _outputNorm = Gemma2Tensors.LoadVector(gguf, "output_norm.weight", H);

        _layers = new Layer[cfg.Layers];
        for (int l = 0; l < cfg.Layers; l++)
        {
            int hd = cfg.HeadDim[l], kv = cfg.KvHeads[l];
            string p = $"blk.{l}.";
            _layers[l] = new Layer
            {
                Q = Gemma2Linear.Load(gguf, p + "attn_q.weight", H, cfg.Heads * hd),
                K = Gemma2Linear.Load(gguf, p + "attn_k.weight", H, kv * hd),
                V = Gemma2Linear.Load(gguf, p + "attn_v.weight", H, kv * hd),
                O = Gemma2Linear.Load(gguf, p + "attn_output.weight", cfg.Heads * hd, H),
                Gate = Gemma2Linear.Load(gguf, p + "ffn_gate.weight", H, cfg.FfnDim),
                Up = Gemma2Linear.Load(gguf, p + "ffn_up.weight", H, cfg.FfnDim),
                Down = Gemma2Linear.Load(gguf, p + "ffn_down.weight", cfg.FfnDim, H),
                PleGate = Gemma2Linear.Load(gguf, p + "inp_gate.weight", H, cfg.PleDim),
                PleProj = Gemma2Linear.Load(gguf, p + "proj.weight", cfg.PleDim, H),
                AttnNorm = Gemma2Tensors.LoadVector(gguf, p + "attn_norm.weight", H),
                QNorm = Gemma2Tensors.LoadVector(gguf, p + "attn_q_norm.weight", hd),
                KNorm = Gemma2Tensors.LoadVector(gguf, p + "attn_k_norm.weight", hd),
                AttnPostNorm = Gemma2Tensors.LoadVector(gguf, p + "post_attention_norm.weight", H),
                FfnNorm = Gemma2Tensors.LoadVector(gguf, p + "ffn_norm.weight", H),
                FfnPostNorm = Gemma2Tensors.LoadVector(gguf, p + "post_ffw_norm.weight", H),
                PleNorm = Gemma2Tensors.LoadVector(gguf, p + "post_norm.weight", H),
                OutScale = Gemma2Tensors.LoadVector(gguf, p + "layer_output_scale.weight", 1)[0],
            };
        }
    }

    public float[] ForwardPooled(ReadOnlySpan<int> tokens)
    {
        float[] h = ForwardHidden(tokens);
        int H = _cfg.Hidden, n = tokens.Length;
        var pooled = new float[H];
        for (int t = 0; t < n; t++)
            for (int d = 0; d < H; d++) pooled[d] += h[t * H + d];
        for (int d = 0; d < H; d++) pooled[d] /= n;
        return pooled;
    }

    public float[] ForwardHidden(ReadOnlySpan<int> tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_gate) // scratch is per-call, but the thread pool is shared; serialise to avoid oversubscription
        {
            return Run(tokens);
        }
    }

    public float[] ForwardPooledEmbeds(float[] x0, int n)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        float[] h;
        lock (_gate) h = Run(x0, n);
        int H = _cfg.Hidden;
        var pooled = new float[H];
        for (int t = 0; t < n; t++)
            for (int d = 0; d < H; d++) pooled[d] += h[t * H + d];
        for (int d = 0; d < H; d++) pooled[d] /= n;
        return pooled;
    }

    private float[] Run(ReadOnlySpan<int> tokens) => Run(Gemma2Embedding.Gather(_gguf, _cfg, tokens), tokens.Length);

    private float[] Run(float[] x0, int n)
    {
        int H = _cfg.Hidden, P = _cfg.PleDim, F = _cfg.FfnDim;
        float[] x = (float[])x0.Clone();

        var normed = new float[n * H];
        var tmpH = new float[n * H];
        var ffnGate = new float[(long)n * F];
        var ffnUp = new float[(long)n * F];
        var pleIn = new float[n * P];
        var pleG = new float[n * P];
        float pleScale = 1f / MathF.Sqrt(H);

        fixed (float* px0 = x0, px = x, pn = normed, po = tmpH, pg = ffnGate, pu = ffnUp, ppi = pleIn, ppg = pleG)
        {
            for (int l = 0; l < _cfg.Layers; l++)
            {
                var layer = _layers[l];

                // ---- attention ----
                for (int t = 0; t < n; t++) RmsNorm(px + (long)t * H, layer.AttnNorm, pn + (long)t * H, H);
                float[] attn = Attention(layer, l, normed, n);
                fixed (float* pa = attn) layer.O.MatMul(pa, n, po);
                for (int t = 0; t < n; t++)
                {
                    RmsNorm(po + (long)t * H, layer.AttnPostNorm, po + (long)t * H, H);
                    for (int d = 0; d < H; d++) px[(long)t * H + d] += po[(long)t * H + d];
                }

                // ---- feed-forward ----
                for (int t = 0; t < n; t++) RmsNorm(px + (long)t * H, layer.FfnNorm, pn + (long)t * H, H);
                layer.Gate.MatMul(pn, n, pg);
                layer.Up.MatMul(pn, n, pu);
                long cnt = (long)n * F;
                nint ppg2 = (nint)pg, ppu2 = (nint)pu;
                Parallel.For(0, (int)((cnt + 8191) / 8192), c =>
                {
                    float* g = (float*)ppg2; float* u = (float*)ppu2;
                    long s = (long)c * 8192, e = Math.Min(cnt, s + 8192);
                    for (long i = s; i < e; i++) g[i] = GeluTanh(g[i]) * u[i];
                });
                layer.Down.MatMul(pg, n, po);
                for (int t = 0; t < n; t++)
                {
                    RmsNorm(po + (long)t * H, layer.FfnPostNorm, po + (long)t * H, H);
                    for (int d = 0; d < H; d++) px[(long)t * H + d] += po[(long)t * H + d];
                }

                // ---- per-layer embedding (projection-only PLE, computed for this layer's slice) ----
                _perLayerProj.MatMul(px0, n, ppi, rowStart: l * P, rowCount: P);
                for (int t = 0; t < n; t++)
                {
                    float* s = ppi + (long)t * P;
                    for (int i = 0; i < P; i++) s[i] *= pleScale;
                    RmsNorm(s, _perLayerProjNorm, s, P);
                }
                layer.PleGate.MatMul(px, n, ppg);
                for (int i = 0; i < n * P; i++) ppg[i] = GeluTanh(ppg[i]) * ppi[i];
                layer.PleProj.MatMul(ppg, n, po);
                for (int t = 0; t < n; t++)
                {
                    RmsNorm(po + (long)t * H, layer.PleNorm, po + (long)t * H, H);
                    for (int d = 0; d < H; d++)
                        px[(long)t * H + d] = (px[(long)t * H + d] + po[(long)t * H + d]) * layer.OutScale;
                }
            }

            for (int t = 0; t < n; t++) RmsNorm(px + (long)t * H, _outputNorm, px + (long)t * H, H);
        }
        return x;
    }

    private float[] Attention(Layer layer, int li, float[] normed, int n)
    {
        int hd = _cfg.HeadDim[li], kvh = _cfg.KvHeads[li], heads = _cfg.Heads;
        int qDim = heads * hd, kvDim = kvh * hd, group = heads / kvh;
        bool swa = _cfg.IsSwa[li];

        var q = new float[(long)n * qDim];
        var k = new float[(long)n * kvDim];
        var v = new float[(long)n * kvDim];
        var output = new float[(long)n * qDim];

        float theta = swa ? _cfg.RopeBaseSwa : _cfg.RopeBase;
        int half = hd / 2;
        var invFreq = new float[half];
        for (int i = 0; i < half; i++) invFreq[i] = (float)(1.0 / Math.Pow(theta, (2.0 * i) / hd));
        int window = swa ? _cfg.HalfWindow : int.MaxValue;
        float eps = _cfg.Eps;

        fixed (float* pn = normed, pq = q, pk = k, pv = v, po = output)
        {
            layer.Q.MatMul(pn, n, pq);
            layer.K.MatMul(pn, n, pk);
            layer.V.MatMul(pn, n, pv);

            nint nq = (nint)pq, nk = (nint)pk, nv = (nint)pv, nout = (nint)po;
            float[] qNorm = layer.QNorm, kNorm = layer.KNorm;

            Parallel.For(0, n, t =>
            {
                float* qq = (float*)nq; float* kk = (float*)nk; float* vv = (float*)nv;
                for (int h = 0; h < heads; h++)
                {
                    float* head = qq + (long)t * qDim + h * hd;
                    RmsNormEps(head, qNorm, head, hd, eps);
                    Rope(head, t, invFreq);
                }
                for (int h = 0; h < kvh; h++)
                {
                    float* kh = kk + (long)t * kvDim + h * hd;
                    RmsNormEps(kh, kNorm, kh, hd, eps);
                    Rope(kh, t, invFreq);
                    float* vh = vv + (long)t * kvDim + h * hd;
                    RmsNormEps(vh, null, vh, hd, eps);
                }
            });

            Parallel.For(0, n * heads, idx =>
            {
                float* qq = (float*)nq; float* kk = (float*)nk; float* vv = (float*)nv; float* oo = (float*)nout;
                int t = idx / heads, h = idx % heads, kvHead = h / group;
                int lo = window == int.MaxValue ? 0 : Math.Max(0, t - window);
                int hi = window == int.MaxValue ? n - 1 : Math.Min(n - 1, t + window);
                int count = hi - lo + 1;

                float[] scores = new float[count];
                float* qh = qq + (long)t * qDim + h * hd;
                float max = float.NegativeInfinity;
                for (int j = 0; j < count; j++)
                {
                    float s = Simd.Dot(qh, kk + (long)(lo + j) * kvDim + kvHead * hd, hd); // attention scale == 1.0
                    scores[j] = s;
                    if (s > max) max = s;
                }
                float sum = 0f;
                for (int j = 0; j < count; j++) { float e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                float inv = 1f / sum;
                float* oh = oo + (long)t * qDim + h * hd;
                for (int j = 0; j < count; j++)
                {
                    float w = scores[j] * inv;
                    Axpy(oh, vv + (long)(lo + j) * kvDim + kvHead * hd, w, hd);
                }
            });
        }
        return output;
    }

    // ------------------------------------------------------------------ math helpers

    private void RmsNorm(float* x, float[]? weight, float* dst, int len) => RmsNormEps(x, weight, dst, len, _cfg.Eps);

    private static void RmsNormEps(float* x, float[]? weight, float* dst, int len, float eps)
    {
        float ss = Simd.Dot(x, x, len);
        float scale = 1f / MathF.Sqrt(ss / len + eps);
        if (weight is null)
        {
            for (int i = 0; i < len; i++) dst[i] = x[i] * scale;
        }
        else
        {
            for (int i = 0; i < len; i++) dst[i] = x[i] * scale * weight[i];
        }
    }

    private static void Rope(float* head, int pos, float[] invFreq)
    {
        int half = invFreq.Length;
        for (int i = 0; i < half; i++)
        {
            float angle = pos * invFreq[i];
            float c = MathF.Cos(angle), s = MathF.Sin(angle);
            float a = head[i], b = head[i + half];
            head[i] = a * c - b * s;
            head[i + half] = b * c + a * s;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Axpy(float* y, float* x, float a, int n)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var va = Vector256.Create(a);
            for (; i <= n - 8; i += 8) (Vector256.Load(y + i) + va * Vector256.Load(x + i)).Store(y + i);
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            var va = Vector128.Create(a);
            for (; i <= n - 4; i += 4) (Vector128.Load(y + i) + va * Vector128.Load(x + i)).Store(y + i);
        }
        for (; i < n; i++) y[i] += a * x[i];
    }

    /// <summary>GELU, tanh approximation (<c>gelu_pytorch_tanh</c>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float GeluTanh(float x)
    {
        const float c = 0.7978845608028654f;
        return 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x)));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _perLayerProj.Dispose();
        foreach (var l in _layers)
        {
            l.Q.Dispose(); l.K.Dispose(); l.V.Dispose(); l.O.Dispose();
            l.Gate.Dispose(); l.Up.Dispose(); l.Down.Dispose(); l.PleGate.Dispose(); l.PleProj.Dispose();
        }
    }
}

