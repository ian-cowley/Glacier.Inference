namespace Glacier.Inference.Embedding;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

/// <summary>
/// Audio preprocessing for EmbeddingGemma 2 (matches Hugging Face <c>Gemma4AudioFeatureExtractor</c>):
/// 16 kHz mono float32, left-pad 160 zeros, 320-sample periodic Hann window, 160 hop size, 512-pt real FFT,
/// 128-bin HTK mel filterbank (0-8000 Hz), natural log(mel + 1e-3).
/// </summary>
public static class Gemma4AudioPreprocessor
{
    public const int SampleRate = 16000;
    public const int WindowSize = 320;
    public const int HopSize = 160;
    public const int FftSize = 512;
    public const int FftBins = FftSize / 2 + 1; // 257
    public const int MelBins = 128;

    private static readonly Lazy<float[,]> HtkMelFilters = new(ComputeMelFilters);

    private static float[,] ComputeMelFilters()
    {
        static double HtkHzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
        static double HtkMelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

        double melMin = HtkHzToMel(0.0);
        double melMax = HtkHzToMel(8000.0);
        var melPoints = new double[MelBins + 2];
        for (int i = 0; i < MelBins + 2; i++)
            melPoints[i] = melMin + (melMax - melMin) * i / (MelBins + 1);

        var hzPoints = new double[MelBins + 2];
        for (int i = 0; i < MelBins + 2; i++) hzPoints[i] = HtkMelToHz(melPoints[i]);

        var fftFreqs = new double[FftBins];
        for (int k = 0; k < FftBins; k++) fftFreqs[k] = 8000.0 * k / (FftBins - 1);

        var weights = new float[FftBins, MelBins];
        for (int i = 0; i < MelBins; i++)
        {
            double lower = hzPoints[i], center = hzPoints[i + 1], upper = hzPoints[i + 2];
            for (int k = 0; k < FftBins; k++)
            {
                double f = fftFreqs[k];
                if (f >= lower && f <= center)
                    weights[k, i] = (float)((f - lower) / (center - lower));
                else if (f >= center && f <= upper)
                    weights[k, i] = (float)((upper - f) / (upper - center));
            }
        }
        return weights;
    }

    /// <summary>
    /// Computes [T, 128] log-mel spectrogram from 16kHz mono audio samples.
    /// </summary>
    public static float[] ExtractLogMel(ReadOnlySpan<float> audio)
    {
        if (audio.IsEmpty) return Array.Empty<float>();

        // Left pad 160 zeros
        const int leftPad = 160;
        int totalLen = audio.Length + leftPad;
        if (totalLen < 321) return Array.Empty<float>();

        int nFrames = (totalLen - 321) / HopSize + 1;
        var padded = new float[totalLen];
        audio.CopyTo(padded.AsSpan(leftPad));

        // Periodic Hann window: 0.5 - 0.5 * cos(2*pi*n / 320)
        var window = new float[WindowSize];
        for (int n = 0; n < WindowSize; n++)
            window[n] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / WindowSize));

        var logMel = new float[nFrames * MelBins];
        var melFilters = HtkMelFilters.Value;

        // Precompute twiddle factors for FFT
        var (cosTable, sinTable) = PrecomputeFftTables(FftSize);

        Parallel.For(0, nFrames, fr =>
        {
            int start = fr * HopSize;
            float[] re = new float[FftSize];
            float[] im = new float[FftSize];
            for (int n = 0; n < WindowSize; n++)
                re[n] = (double)padded[start + n] == 0 ? 0f : padded[start + n] * window[n];

            // In-place Radix-2 Cooley-Tukey FFT
            Fft(re, im, FftSize, cosTable, sinTable);

            // Magnitude of first 257 bins
            Span<float> mag = stackalloc float[FftBins];
            for (int k = 0; k < FftBins; k++)
                mag[k] = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);

            // Matrix multiply: mag [257] @ melFilters [257, 128] -> mel [128]
            int outOffset = fr * MelBins;
            for (int m = 0; m < MelBins; m++)
            {
                double sum = 0;
                for (int k = 0; k < FftBins; k++) sum += (double)mag[k] * melFilters[k, m];
                logMel[outOffset + m] = MathF.Log((float)sum + 1e-3f);
            }
        });

        return logMel;
    }

    private static (float[] cos, float[] sin) PrecomputeFftTables(int n)
    {
        var cos = new float[n / 2];
        var sin = new float[n / 2];
        for (int i = 0; i < n / 2; i++)
        {
            double angle = -2.0 * Math.PI * i / n;
            cos[i] = (float)Math.Cos(angle);
            sin[i] = (float)Math.Sin(angle);
        }
        return (cos, sin);
    }

    private static void Fft(float[] re, float[] im, int n, float[] cosTable, float[] sinTable)
    {
        // Bit reversal
        int j = 0;
        for (int i = 0; i < n - 1; i++)
        {
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
            int k = n >> 1;
            while (k <= j) { j -= k; k >>= 1; }
            j += k;
        }

        // Cooley-Tukey butterflies
        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            int step = n / len;
            for (int i = 0; i < n; i += len)
            {
                for (int k = 0; k < half; k++)
                {
                    int tableIdx = k * step;
                    float c = cosTable[tableIdx];
                    float s = sinTable[tableIdx];
                    float tr = re[i + half + k] * c - im[i + half + k] * s;
                    float ti = re[i + half + k] * s + im[i + half + k] * c;
                    re[i + half + k] = re[i + k] - tr;
                    im[i + half + k] = im[i + k] - ti;
                    re[i + k] += tr;
                    im[i + k] += ti;
                }
            }
        }
    }
}

/// <summary>
/// Conformer audio tower + multimodal embedder for EmbeddingGemma 2 (weights from the <c>mmproj</c> GGUF).
/// SubSampleConvProjection, 12 Conformer blocks with chunked relative attention and causal light conv,
/// 1024→1536 output projection with bias, and RMSNorm → 1536→512 soft token projection.
/// </summary>
internal sealed unsafe class Gemma4AudioTower : IDisposable
{
    private const int Hidden = 1024, Heads = 8, HeadDim = 128, Ffn = 4096;
    private const int ConvKernel = 5, LeftPad = 4, ChunkSize = 12, ContextSize = 24, RelPositions = 13;
    private const int OutProjDim = 1536, SoftDim = 512;
    private const float Eps = 1e-6f;

    private sealed class ClippedLinear : IDisposable
    {
        public Gemma2Linear Lin;
        public float InMin, InMax, OutMin, OutMax;
        public bool HasClip;

        public ClippedLinear(GgufFile gguf, string name, int inDim, int outDim, bool clip = true)
        {
            Lin = Gemma2Linear.Load(gguf, name + ".weight", inDim, outDim);
            HasClip = clip && gguf.Tensors.ContainsKey(name + ".input_min");
            if (HasClip)
            {
                InMin = Gemma2Tensors.LoadVector(gguf, name + ".input_min", 1)[0];
                InMax = Gemma2Tensors.LoadVector(gguf, name + ".input_max", 1)[0];
                OutMin = Gemma2Tensors.LoadVector(gguf, name + ".output_min", 1)[0];
                OutMax = Gemma2Tensors.LoadVector(gguf, name + ".output_max", 1)[0];
            }
        }

        public void MatMul(float* input, int n, float* output)
        {
            if (HasClip)
            {
                for (long i = 0; i < (long)n * Lin.In; i++)
                    input[i] = Math.Clamp(input[i], InMin, InMax);
            }
            Lin.MatMul(input, n, output);
            if (HasClip)
            {
                for (long i = 0; i < (long)n * Lin.Out; i++)
                    output[i] = Math.Clamp(output[i], OutMin, OutMax);
            }
        }

        public void Dispose() => Lin.Dispose();
    }

    private sealed class ConformerBlock : IDisposable
    {
        public required ClippedLinear Ff1Up, Ff1Down, Ff2Up, Ff2Down;
        public required ClippedLinear Q, K, V, Post, ConvPw1, ConvPw2;
        public required Gemma2Linear KRel;
        public required float[] Ff1PreNorm, Ff1PostNorm, Ff2PreNorm, Ff2PostNorm;
        public required float[] AttnPreNorm, AttnPostNorm;
        public required float[] ConvPreNorm, ConvNorm;
        public required float[] ConvDw; // [1024 * 5]
        public required float[] PerDimScale; // [128]
        public required float[] NormOut;

        public void Dispose()
        {
            Ff1Up.Dispose(); Ff1Down.Dispose(); Ff2Up.Dispose(); Ff2Down.Dispose();
            Q.Dispose(); K.Dispose(); V.Dispose(); Post.Dispose(); ConvPw1.Dispose(); ConvPw2.Dispose();
            KRel.Dispose();
        }
    }

    private readonly GgufFile _gguf;
    private readonly int _layers;
    // Subsample conv projection
    private readonly float[] _w0;   // [128, 1, 3, 3]
    private readonly float[] _ln0;  // [128]
    private readonly float[] _w1;   // [32, 128, 3, 3]
    private readonly float[] _ln1;  // [32]
    private readonly Gemma2Linear _subProj; // [1024, 1024]
    private readonly float[] _relPosTable;  // [13, 1024]

    private readonly ConformerBlock[] _blocks;
    private readonly Gemma2Linear _outProj; // [1024, 1536]
    private readonly float[] _outProjBias;  // [1536]
    private readonly Gemma2Linear _softProj; // [1536, 512]
    private readonly object _gate = new();

    public static bool IsAudioMmproj(GgufFile g) =>
        g.Tensors.ContainsKey("a.conv1d.0.weight") && g.Tensors.ContainsKey("mm.a.input_projection.weight");

    public Gemma4AudioTower(GgufFile mmproj)
    {
        _gguf = mmproj;
        _layers = (int)mmproj.GetMetadataUInt32("clip.audio.block_count", 12);

        _w0 = LoadRawFloats(mmproj, "a.conv1d.0.weight", 128 * 1 * 3 * 3);
        _ln0 = Gemma2Tensors.LoadVector(mmproj, "a.conv1d.0.norm.weight", 128);

        _w1 = LoadRawFloats(mmproj, "a.conv1d.1.weight", 32 * 128 * 3 * 3);
        _ln1 = Gemma2Tensors.LoadVector(mmproj, "a.conv1d.1.norm.weight", 32);

        _subProj = Gemma2Linear.Load(mmproj, "a.input_projection.weight", Hidden, Hidden);

        // Rel pos table: 13 rows of [sin(p*inv), cos(p*inv)], p = 12..0
        _relPosTable = new float[RelPositions * Hidden];
        for (int p = 0; p < RelPositions; p++)
        {
            int pos = 12 - p;
            for (int i = 0; i < 512; i++)
            {
                double inv = Math.Exp(-i * Math.Log(10000.0) / 511.0);
                double angle = pos * inv;
                _relPosTable[p * Hidden + i] = (float)Math.Sin(angle);
                _relPosTable[p * Hidden + 512 + i] = (float)Math.Cos(angle);
            }
        }

        _blocks = new ConformerBlock[_layers];
        for (int i = 0; i < _layers; i++)
        {
            string p = $"a.blk.{i}.";
            _blocks[i] = new ConformerBlock
            {
                Ff1PreNorm = Gemma2Tensors.LoadVector(mmproj, p + "ffn_norm.weight", Hidden),
                Ff1Up = new ClippedLinear(mmproj, p + "ffn_up", Hidden, Ffn),
                Ff1Down = new ClippedLinear(mmproj, p + "ffn_down", Ffn, Hidden),
                Ff1PostNorm = Gemma2Tensors.LoadVector(mmproj, p + "ffn_post_norm.weight", Hidden),

                AttnPreNorm = Gemma2Tensors.LoadVector(mmproj, p + "attn_pre_norm.weight", Hidden),
                Q = new ClippedLinear(mmproj, p + "attn_q", Hidden, Hidden),
                K = new ClippedLinear(mmproj, p + "attn_k", Hidden, Hidden),
                V = new ClippedLinear(mmproj, p + "attn_v", Hidden, Hidden),
                KRel = Gemma2Linear.Load(mmproj, p + "attn_k_rel.weight", Hidden, Hidden),
                PerDimScale = Gemma2Tensors.LoadVector(mmproj, p + "per_dim_scale.weight", HeadDim),
                Post = new ClippedLinear(mmproj, p + "attn_out", Hidden, Hidden),
                AttnPostNorm = Gemma2Tensors.LoadVector(mmproj, p + "attn_post_norm.weight", Hidden),

                ConvPreNorm = Gemma2Tensors.LoadVector(mmproj, p + "conv_norm.weight", Hidden),
                ConvPw1 = new ClippedLinear(mmproj, p + "conv_pw1", Hidden, 2048),
                ConvDw = LoadRawFloats(mmproj, p + "conv_dw.weight", Hidden * ConvKernel),
                ConvNorm = Gemma2Tensors.LoadVector(mmproj, p + "norm_conv.weight", Hidden),
                ConvPw2 = new ClippedLinear(mmproj, p + "conv_pw2", Hidden, Hidden),

                Ff2PreNorm = Gemma2Tensors.LoadVector(mmproj, p + "ffn_norm_1.weight", Hidden),
                Ff2Up = new ClippedLinear(mmproj, p + "ffn_up_1", Hidden, Ffn),
                Ff2Down = new ClippedLinear(mmproj, p + "ffn_down_1", Ffn, Hidden),
                Ff2PostNorm = Gemma2Tensors.LoadVector(mmproj, p + "ffn_post_norm_1.weight", Hidden),

                NormOut = Gemma2Tensors.LoadVector(mmproj, p + "ln2.weight", Hidden),
            };
        }

        _outProj = Gemma2Linear.Load(mmproj, "a.pre_encode.out.weight", Hidden, OutProjDim);
        _outProjBias = Gemma2Tensors.LoadVector(mmproj, "a.pre_encode.out.bias", OutProjDim);
        _softProj = Gemma2Linear.Load(mmproj, "mm.a.input_projection.weight", OutProjDim, SoftDim);
    }

    /// <summary>
    /// Encodes log-mel features [T, 128] into soft tokens [T'', 512].
    /// </summary>
    public float[] Encode(float[] logMel, int numFrames)
    {
        if (numFrames == 0) return Array.Empty<float>();

        lock (_gate)
        {
            // 1. SubSampleConvProjection
            float[] h = Subsample(logMel, numFrames, out int tSub);

            // 2. Conformer blocks
            var tmpH = new float[tSub * Hidden];
            var ffnBuf = new float[tSub * Ffn];
            var q = new float[tSub * Hidden];
            var k = new float[tSub * Hidden];
            var v = new float[tSub * Hidden];
            var attnOut = new float[tSub * Hidden];
            var pw1Out = new float[tSub * 2048];
            var gluOut = new float[tSub * Hidden];
            var dwOut = new float[tSub * Hidden];

            for (int l = 0; l < _layers; l++)
            {
                var b = _blocks[l];

                // FFN1: h + 0.5 * post_norm(ffw2(silu(ffw1(pre_norm(h)))))
                RmsRows(h, b.Ff1PreNorm, tmpH, tSub, Hidden);
                fixed (float* pt = tmpH, pf = ffnBuf)
                {
                    b.Ff1Up.MatMul(pt, tSub, pf);
                    for (int i = 0; i < tSub * Ffn; i++) pf[i] = Silu(pf[i]);
                    b.Ff1Down.MatMul(pf, tSub, pt);
                }
                RmsRows(tmpH, b.Ff1PostNorm, tmpH, tSub, Hidden);
                for (int i = 0; i < tSub * Hidden; i++) h[i] += 0.5f * tmpH[i];

                // Attention
                float[] rAttn = (float[])h.Clone();
                RmsRows(h, b.AttnPreNorm, tmpH, tSub, Hidden);
                fixed (float* pt = tmpH, pq = q, pk = k, pv = v, po = attnOut)
                {
                    b.Q.MatMul(pt, tSub, pq);
                    b.K.MatMul(pt, tSub, pk);
                    b.V.MatMul(pt, tSub, pv);

                    // Scale Q and K
                    float qScale = (float)(Math.Pow(HeadDim, -0.5) / Math.Log(2.0));
                    float kScale = (float)(Math.Log(1.0 + Math.E) / Math.Log(2.0));
                    for (int t = 0; t < tSub; t++)
                    {
                        for (int hd = 0; hd < Heads; hd++)
                        {
                            for (int d = 0; d < HeadDim; d++)
                            {
                                int idx = t * Hidden + hd * HeadDim + d;
                                pq[idx] *= qScale * b.PerDimScale[d];
                                pk[idx] *= kScale;
                            }
                        }
                    }

                    // Compute relative position attention
                    ChunkedRelativeAttention(pq, pk, pv, b.KRel, po, tSub);

                    b.Post.MatMul(po, tSub, pt);
                }
                RmsRows(tmpH, b.AttnPostNorm, tmpH, tSub, Hidden);
                for (int i = 0; i < tSub * Hidden; i++) h[i] = rAttn[i] + tmpH[i];

                // LConv1d
                float[] rConv = (float[])h.Clone();
                RmsRows(h, b.ConvPreNorm, tmpH, tSub, Hidden);
                fixed (float* pt = tmpH, ppw1 = pw1Out, pglu = gluOut, pdw = dwOut)
                {
                    b.ConvPw1.MatMul(pt, tSub, ppw1);
                    // GLU: first 1024 * sigmoid(second 1024)
                    for (int t = 0; t < tSub; t++)
                    {
                        for (int d = 0; d < Hidden; d++)
                        {
                            float a = ppw1[t * 2048 + d];
                            float sig = Sigmoid(ppw1[t * 2048 + Hidden + d]);
                            pglu[t * Hidden + d] = a * sig;
                        }
                    }

                    // Depthwise Causal Conv1d: kernel 5, left pad 4
                    CausalDepthwiseConv(pglu, b.ConvDw, pdw, tSub);
                }
                RmsRows(dwOut, b.ConvNorm, dwOut, tSub, Hidden);
                for (int i = 0; i < tSub * Hidden; i++) dwOut[i] = Silu(dwOut[i]);
                fixed (float* pdw = dwOut, pt = tmpH)
                {
                    b.ConvPw2.MatMul(pdw, tSub, pt);
                }
                for (int i = 0; i < tSub * Hidden; i++) h[i] = rConv[i] + tmpH[i];

                // FFN2: h + 0.5 * post_norm(ffw2(silu(ffw1(pre_norm(h)))))
                RmsRows(h, b.Ff2PreNorm, tmpH, tSub, Hidden);
                fixed (float* pt = tmpH, pf = ffnBuf)
                {
                    b.Ff2Up.MatMul(pt, tSub, pf);
                    for (int i = 0; i < tSub * Ffn; i++) pf[i] = Silu(pf[i]);
                    b.Ff2Down.MatMul(pf, tSub, pt);
                }
                RmsRows(tmpH, b.Ff2PostNorm, tmpH, tSub, Hidden);
                for (int i = 0; i < tSub * Hidden; i++) h[i] += 0.5f * tmpH[i];

                // Final layer norm
                RmsRows(h, b.NormOut, h, tSub, Hidden);
            }

            // Output projection: 1024 -> 1536 with bias
            var towerOut = new float[tSub * OutProjDim];
            fixed (float* ph = h, po = towerOut)
            {
                _outProj.MatMul(ph, tSub, po);
                for (int t = 0; t < tSub; t++)
                    for (int d = 0; d < OutProjDim; d++)
                        po[t * OutProjDim + d] += _outProjBias[d];
            }

            // Embedder: RMSNorm (no weight) -> Linear 1536 -> 512
            RmsRows(towerOut, null, towerOut, tSub, OutProjDim);
            var softTokens = new float[tSub * SoftDim];
            fixed (float* pt = towerOut, ps = softTokens)
            {
                _softProj.MatMul(pt, tSub, ps);
            }
            return softTokens;
        }
    }


    // ---------------------------------------------------------------- subsample conv

    internal float[] Subsample(float[] logMel, int T, out int tOut)
    {
        // Conv0: [1, T, 128] -> [128, T1, 64]
        int t1 = (T - 1) / 2 + 1;
        const int f1 = 64;
        var c0 = new float[t1 * f1 * 128];

        Parallel.For(0, t1, yt =>
        {
            int inT = yt * 2;
            for (int xf = 0; xf < f1; xf++)
            {
                int inF = xf * 2;
                for (int oc = 0; oc < 128; oc++)
                {
                    double sum = 0;
                    for (int ky = 0; ky < 3; ky++)
                    {
                        int pt = inT - 1 + ky;
                        if (pt < 0 || pt >= T) continue;
                        for (int kx = 0; kx < 3; kx++)
                        {
                            int pf = inF - 1 + kx;
                            if (pf < 0 || pf >= 128) continue;
                            sum += (double)logMel[pt * 128 + pf] * _w0[((oc * 1 + 0) * 3 + ky) * 3 + kx];
                        }
                    }
                    c0[(yt * f1 + xf) * 128 + oc] = (float)sum;
                }
            }
        });

        // LayerNorm0 over 128 channels + ReLU
        for (int i = 0; i < t1 * f1; i++)
        {
            int baseIdx = i * 128;
            double mean = 0;
            for (int c = 0; c < 128; c++) mean += c0[baseIdx + c];
            mean /= 128.0;
            double var = 0;
            for (int c = 0; c < 128; c++) { double diff = c0[baseIdx + c] - mean; var += diff * diff; }
            float invStd = (float)(1.0 / Math.Sqrt(var / 128.0 + Eps));
            for (int c = 0; c < 128; c++)
            {
                float v = (float)((c0[baseIdx + c] - mean) * invStd) * _ln0[c];
                c0[baseIdx + c] = Math.Max(0f, v);
            }
        }

        // Conv1: [128, T1, 64] -> [32, T2, 32]
        int t2 = (t1 - 1) / 2 + 1;
        const int f2 = 32;
        var c1 = new float[t2 * f2 * 32];

        Parallel.For(0, t2, yt =>
        {
            int inT = yt * 2;
            for (int xf = 0; xf < f2; xf++)
            {
                int inF = xf * 2;
                for (int oc = 0; oc < 32; oc++)
                {
                    double sum = 0;
                    for (int ic = 0; ic < 128; ic++)
                    {
                        for (int ky = 0; ky < 3; ky++)
                        {
                            int pt = inT - 1 + ky;
                            if (pt < 0 || pt >= t1) continue;
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int pf = inF - 1 + kx;
                                if (pf < 0 || pf >= f1) continue;
                                sum += (double)c0[(pt * f1 + pf) * 128 + ic] * _w1[((oc * 128 + ic) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                    c1[(yt * f2 + xf) * 32 + oc] = (float)sum;
                }
            }
        });

        // LayerNorm1 over 32 channels + ReLU
        for (int i = 0; i < t2 * f2; i++)
        {
            int baseIdx = i * 32;
            double mean = 0;
            for (int c = 0; c < 32; c++) mean += c1[baseIdx + c];
            mean /= 32.0;
            double var = 0;
            for (int c = 0; c < 32; c++) { double diff = c1[baseIdx + c] - mean; var += diff * diff; }
            float invStd = (float)(1.0 / Math.Sqrt(var / 32.0 + Eps));
            for (int c = 0; c < 32; c++)
            {
                float v = (float)((c1[baseIdx + c] - mean) * invStd) * _ln1[c];
                c1[baseIdx + c] = Math.Max(0f, v);
            }
        }

        // Reshape [t2, 32_freq, 32_ch] -> [t2, 1024] (channel fastest)
        // Then Linear 1024 -> 1024
        var subOut = new float[t2 * Hidden];
        fixed (float* pc1 = c1, pso = subOut)
        {
            _subProj.MatMul(pc1, t2, pso);
        }
        tOut = t2;
        return subOut;
    }

    // ---------------------------------------------------------------- chunked relative attention

    private void ChunkedRelativeAttention(float* q, float* k, float* v, Gemma2Linear kRel, float* output, int seqLen)
    {
        // 1. Project relative position table: [13, 1024] -> [13, 1024]
        var relKeys = new float[RelPositions * Hidden];
        fixed (float* pr = _relPosTable, prk = relKeys)
        {
            kRel.MatMul(pr, RelPositions, prk);
        }

        int numBlocks = (seqLen + ChunkSize - 1) / ChunkSize;

        Parallel.For(0, numBlocks * Heads, idx =>
        {
            int b = idx / Heads;
            int h = idx % Heads;

            int qStart = b * ChunkSize;
            int qCount = Math.Min(ChunkSize, seqLen - qStart);
            int ctxStart = b * ChunkSize - 12;

            // Compute bd_raw: [qCount, 13]
            var bdRaw = new float[ChunkSize * RelPositions];
            for (int qi = 0; qi < qCount; qi++)
            {
                int gq = qStart + qi;
                float* qHead = q + (long)gq * Hidden + h * HeadDim;
                for (int pos = 0; pos < RelPositions; pos++)
                {
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++)
                        dot += (double)qHead[d] * relKeys[pos * Hidden + h * HeadDim + d];
                    bdRaw[qi * RelPositions + pos] = (float)dot;
                }
            }

            // Rel-shift bd_raw -> bdShifted [ChunkSize, ContextSize]
            var bdShifted = new float[ChunkSize * ContextSize];
            for (int qi = 0; qi < ChunkSize; qi++)
            {
                for (int ki = 0; ki < ContextSize; ki++)
                {
                    int flat = qi * ContextSize + ki;
                    int r = flat / (ContextSize + 1);
                    int c = flat % (ContextSize + 1);
                    if (c < RelPositions && r < ChunkSize)
                        bdShifted[qi * ContextSize + ki] = bdRaw[r * RelPositions + c];
                }
            }

            // Attention scores: ac + bd
            var scores = new float[ContextSize];
            for (int qi = 0; qi < qCount; qi++)
            {
                int gq = qStart + qi;
                float* qHead = q + (long)gq * Hidden + h * HeadDim;

                for (int ki = 0; ki < ContextSize; ki++)
                {
                    int gk = ctxStart + ki;
                    bool valid = gk >= 0 && gk < seqLen && gk <= gq && (gq - gk) < 12;
                    if (!valid)
                    {
                        scores[ki] = -1e9f;
                        continue;
                    }

                    // Content-content score ac
                    float* kHead = k + (long)gk * Hidden + h * HeadDim;
                    double ac = 0;
                    for (int d = 0; d < HeadDim; d++) ac += (double)qHead[d] * kHead[d];

                    float s = (float)ac + bdShifted[qi * ContextSize + ki];
                    // Softcap 50
                    scores[ki] = 50.0f * MathF.Tanh(s / 50.0f);
                }

                // Softmax over 24
                float max = float.NegativeInfinity;
                for (int ki = 0; ki < ContextSize; ki++) if (scores[ki] > max) max = scores[ki];
                double sum = 0;
                for (int ki = 0; ki < ContextSize; ki++)
                {
                    scores[ki] = MathF.Exp(scores[ki] - max);
                    sum += scores[ki];
                }
                float invSum = (float)(1.0 / sum);
                for (int ki = 0; ki < ContextSize; ki++) scores[ki] *= invSum;

                // Weighted sum of values
                float* outHead = output + (long)gq * Hidden + h * HeadDim;
                for (int d = 0; d < HeadDim; d++) outHead[d] = 0;

                for (int ki = 0; ki < ContextSize; ki++)
                {
                    int gk = ctxStart + ki;
                    if (gk < 0 || gk >= seqLen) continue;
                    float weight = scores[ki];
                    float* vHead = v + (long)gk * Hidden + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++)
                        outHead[d] += weight * vHead[d];
                }
            }
        });
    }

    // ---------------------------------------------------------------- depthwise conv

    private static void CausalDepthwiseConv(float* input, float[] weight, float* output, int seqLen)
    {
        // k=5, left_pad=4, causal: tap 0 = t-4, tap 4 = t
        Parallel.For(0, Hidden, ch =>
        {
            float w0 = weight[ch * 5 + 0];
            float w1 = weight[ch * 5 + 1];
            float w2 = weight[ch * 5 + 2];
            float w3 = weight[ch * 5 + 3];
            float w4 = weight[ch * 5 + 4];

            for (int t = 0; t < seqLen; t++)
            {
                double sum = 0;
                if (t >= 4) sum += (double)input[(t - 4) * Hidden + ch] * w0;
                if (t >= 3) sum += (double)input[(t - 3) * Hidden + ch] * w1;
                if (t >= 2) sum += (double)input[(t - 2) * Hidden + ch] * w2;
                if (t >= 1) sum += (double)input[(t - 1) * Hidden + ch] * w3;
                sum += (double)input[t * Hidden + ch] * w4;
                output[t * Hidden + ch] = (float)sum;
            }
        });
    }

    // ---------------------------------------------------------------- helpers

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Silu(float x) => x * Sigmoid(x);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

    private static float[] LoadRawFloats(GgufFile gguf, string name, int totalFloats)
    {
        if (!gguf.Tensors.TryGetValue(name, out var info))
            throw new InvalidOperationException($"Tensor '{name}' not found.");
        var res = new float[totalFloats];
        fixed (float* p = res)
            Gemma2Tensors.DequantRow(gguf.GetTensorPointer(info), (uint)info.Type, 0, totalFloats, p);
        return res;
    }

    public void Dispose()
    {
        _subProj.Dispose();
        _outProj.Dispose();
        _softProj.Dispose();
        foreach (var b in _blocks) b.Dispose();
        _gguf.Dispose();
    }
}
