namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Collections.Generic;
using Glacier.Inference.Format;

/// <summary>
/// StyleTTS 2 / KittenTTS Decoder and HiFi-GAN Generator module.
/// Converts acoustic features, pitch (F0), noise energy (N), and voice style into 24kHz CD-quality audio PCM waveform.
/// 100% pure C# .NET 10 bare-metal execution.
/// </summary>
public sealed class KittenDecoder
{
    private readonly float[] _asrResWeight; // [64, 128, 1]
    private readonly float[] _asrResBias;   // [64]

    private readonly float[] _f0ConvWeight; // [1, 1, 3]
    private readonly float[] _f0ConvBias;   // [1]

    private readonly float[] _nConvWeight;  // [1, 1, 3]
    private readonly float[] _nConvBias;    // [1]

    private readonly EncodeBlock _encode;
    private readonly DecodeBlock[] _decodeBlocks;
    private readonly Generator _generator;

    private const int StyleHalf = 128;

    public KittenDecoder(SafetensorsFile file, KittenConfig config)
    {
        _asrResWeight = LoadArray(file, "decoder.asr_res.0.weight");
        _asrResBias = LoadArray(file, "decoder.asr_res.0.bias");

        _f0ConvWeight = LoadArray(file, "decoder.F0_conv.weight");
        _f0ConvBias = LoadArray(file, "decoder.F0_conv.bias");

        _nConvWeight = LoadArray(file, "decoder.N_conv.weight");
        _nConvBias = LoadArray(file, "decoder.N_conv.bias");

        _encode = new EncodeBlock(file, "decoder.encode");

        _decodeBlocks = new DecodeBlock[4];
        for (int i = 0; i < 4; i++)
        {
            _decodeBlocks[i] = new DecodeBlock(file, $"decoder.decode.{i}", isUpsample: i == 3);
        }

        _generator = new Generator(file, config);
    }

    private static unsafe float[] LoadArray(SafetensorsFile file, string tensorName)
    {
        var info = file.Tensors[tensorName];
        float[] arr = new float[info.ElementCount];
        fixed (float* pDst = arr)
        {
            Buffer.MemoryCopy(file.GetTensorPointer(tensorName), pDst, (long)info.ByteSize, (long)info.ByteSize);
        }
        return arr;
    }

    /// <summary>
    /// Executes Decoder and Generator pipeline.
    /// Returns 24kHz audio samples [numSamples].
    /// </summary>
    public float[] Forward(
        ReadOnlySpan<float> sharedLstmOut, // [128, T]
        ReadOnlySpan<float> asrFeatures,    // [128, T]
        ReadOnlySpan<float> f0,             // [1, 2T]
        ReadOnlySpan<float> nAmp,           // [1, 2T]
        ReadOnlySpan<float> style)          // [256]
    {
        // Decoder uses first half of style: style[0..127]
        var styleHalf = style.Slice(0, StyleHalf);

        int t2 = f0.Length; // 2T
        int t = t2 / 2;
        if (t == 0) return Array.Empty<float>();

        // 1. Project ASR features: [128, T] -> [64, T]
        float[] asr = new float[64 * t];
        KittenTensorOps.Conv1D(asrFeatures, _asrResWeight, _asrResBias, asr, 128, 64, t, 1, padding: 0);

        // 2. Downsample F0 and N from 2T -> T (stride=2, padding=1, kernel=3)
        float[] f0Down = new float[1 * t];
        KittenTensorOps.Conv1D(f0, _f0ConvWeight, _f0ConvBias, f0Down, 1, 1, t2, 3, padding: 1, stride: 2);

        float[] nDown = new float[1 * t];
        KittenTensorOps.Conv1D(nAmp, _nConvWeight, _nConvBias, nDown, 1, 1, t2, 3, padding: 1, stride: 2);

        // Align time dimension if downsample yielded slightly different length
        int tAligned = Math.Min(t, Math.Min(f0Down.Length, nDown.Length));

        // 3. EncodeBlock: input = concat[shared_lstm(128), f0_down(1), n_down(1)] = 130ch
        float[] encIn = new float[130 * tAligned];
        for (int i = 0; i < tAligned; i++)
        {
            // shared_lstm (128)
            for (int c = 0; c < 128; c++) encIn[c * tAligned + i] = sharedLstmOut[c * t + i];
            // f0_down (1)
            encIn[128 * tAligned + i] = f0Down[i];
            // n_down (1)
            encIn[129 * tAligned + i] = nDown[i];
        }

        float[] h = _encode.Forward(encIn, styleHalf, tAligned); // [256, tAligned]

        // 4. Four DecodeBlocks
        for (int b = 0; b < 4; b++)
        {
            int curT = h.Length / 256;
            // Assemble decIn: concat[h(256), asr(64), f0_down(1), n_down(1)] = 322ch
            float[] decIn = new float[322 * curT];
            for (int i = 0; i < curT; i++)
            {
                int asrIdx = Math.Min(i, tAligned - 1);
                // h (256)
                for (int c = 0; c < 256; c++) decIn[c * curT + i] = h[c * curT + i];
                // asr (64)
                for (int c = 0; c < 64; c++) decIn[(256 + c) * curT + i] = asr[c * tAligned + asrIdx];
                // f0 (1)
                decIn[320 * curT + i] = f0Down[asrIdx];
                // n (1)
                decIn[321 * curT + i] = nDown[asrIdx];
            }

            h = _decodeBlocks[b].Forward(decIn, styleHalf, curT);
        }
        // After block 3 (upsampling block): h is [256, 2 * tAligned]

        int finalAcousticT = h.Length / 256;

        // F0 upsampled to match final acoustic length
        float[] f0Gen = new float[finalAcousticT];
        KittenPredictor.NearestUpsample1D(f0, f0Gen, 1, t2, finalAcousticT);

        // 5. Generator -> 24kHz audio waveform
        return _generator.Forward(h, f0Gen, styleHalf, finalAcousticT);
    }

    private sealed class EncodeBlock
    {
        private readonly float[] _conv1x1W; // [256, 130, 1]
        private readonly KittenPredictor.AdaInLayer _norm1;
        private readonly float[] _conv1W;   // [256, 130, 3]
        private readonly float[] _conv1B;   // [256]
        private readonly KittenPredictor.AdaInLayer _norm2;
        private readonly float[] _conv2W;   // [256, 256, 3]
        private readonly float[] _conv2B;   // [256]

        public EncodeBlock(SafetensorsFile file, string prefix)
        {
            _conv1x1W = LoadArray(file, $"{prefix}.conv1x1.weight");
            _norm1 = new KittenPredictor.AdaInLayer(file, $"{prefix}.norm1", styleDim: 128, channels: 130);
            _conv1W = LoadArray(file, $"{prefix}.conv1.weight");
            _conv1B = LoadArray(file, $"{prefix}.conv1.bias");
            _norm2 = new KittenPredictor.AdaInLayer(file, $"{prefix}.norm2", styleDim: 128, channels: 256);
            _conv2W = LoadArray(file, $"{prefix}.conv2.weight");
            _conv2B = LoadArray(file, $"{prefix}.conv2.bias");
        }

        public float[] Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> styleHalf, int t)
        {
            float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);

            // Skip: conv1x1(130 -> 256, k=1, no bias)
            float[] skip = new float[256 * t];
            KittenTensorOps.Conv1D(input, _conv1x1W, ReadOnlySpan<float>.Empty, skip, 130, 256, t, 1, padding: 0);

            // Main path
            float[] n1 = new float[130 * t];
            _norm1.Forward(input, styleHalf, n1, 130, t);
            KittenTensorOps.LeakyRelu(n1, 0.2f);

            float[] c1 = new float[256 * t];
            KittenTensorOps.Conv1D(n1, _conv1W, _conv1B, c1, 130, 256, t, 3, padding: 1);

            float[] n2 = new float[256 * t];
            _norm2.Forward(c1, styleHalf, n2, 256, t);
            KittenTensorOps.LeakyRelu(n2, 0.2f);

            float[] c2 = new float[256 * t];
            KittenTensorOps.Conv1D(n2, _conv2W, _conv2B, c2, 256, 256, t, 3, padding: 1);

            float[] output = new float[256 * t];
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = (skip[i] + c2[i]) * invSqrt2;
            }
            return output;
        }
    }

    private sealed class DecodeBlock
    {
        private readonly float[] _conv1x1W; // [256, 322, 1]
        private readonly KittenPredictor.AdaInLayer _norm1;
        private readonly float[] _conv1W;   // [256, 322, 3]
        private readonly float[] _conv1B;   // [256]
        private readonly KittenPredictor.AdaInLayer _norm2;
        private readonly float[] _conv2W;   // [256, 256, 3]
        private readonly float[] _conv2B;   // [256]

        private readonly bool _isUpsample;
        private readonly float[]? _poolW;    // [322, 1, 3]
        private readonly float[]? _poolB;    // [322]

        public DecodeBlock(SafetensorsFile file, string prefix, bool isUpsample)
        {
            _isUpsample = isUpsample;
            _conv1x1W = LoadArray(file, $"{prefix}.conv1x1.weight");
            _norm1 = new KittenPredictor.AdaInLayer(file, $"{prefix}.norm1", styleDim: 128, channels: 322);
            _conv1W = LoadArray(file, $"{prefix}.conv1.weight");
            _conv1B = LoadArray(file, $"{prefix}.conv1.bias");
            _norm2 = new KittenPredictor.AdaInLayer(file, $"{prefix}.norm2", styleDim: 128, channels: 256);
            _conv2W = LoadArray(file, $"{prefix}.conv2.weight");
            _conv2B = LoadArray(file, $"{prefix}.conv2.bias");

            if (isUpsample)
            {
                _poolW = LoadArray(file, $"{prefix}.pool.weight");
                _poolB = LoadArray(file, $"{prefix}.pool.bias");
            }
        }

        public float[] Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> styleHalf, int t)
        {
            float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);

            if (_isUpsample)
            {
                int t2 = t * 2;
                // Skip path: nearest upsample T -> 2T, then conv1x1
                float[] xUp = new float[322 * t2];
                KittenPredictor.NearestUpsample1D(input, xUp, 322, t, t2);

                float[] skip = new float[256 * t2];
                KittenTensorOps.Conv1D(xUp, _conv1x1W, ReadOnlySpan<float>.Empty, skip, 322, 256, t2, 1, padding: 0);

                // Main path: norm1 on T -> leaky_relu -> pool upsample to 2T -> conv1 -> norm2 -> leaky_relu -> conv2
                float[] n1 = new float[322 * t];
                _norm1.Forward(input, styleHalf, n1, 322, t);
                KittenTensorOps.LeakyRelu(n1, 0.2f);

                float[] poolOut = new float[322 * t2];
                KittenPredictor.DepthwiseConvTranspose1D(n1, _poolW!, _poolB!, poolOut, 322, t, stride: 2, padding: 1, outputPadding: 1, kernelSize: 3);

                float[] c1 = new float[256 * t2];
                KittenTensorOps.Conv1D(poolOut, _conv1W, _conv1B, c1, 322, 256, t2, 3, padding: 1);

                float[] n2 = new float[256 * t2];
                _norm2.Forward(c1, styleHalf, n2, 256, t2);
                KittenTensorOps.LeakyRelu(n2, 0.2f);

                float[] c2 = new float[256 * t2];
                KittenTensorOps.Conv1D(n2, _conv2W, _conv2B, c2, 256, 256, t2, 3, padding: 1);

                float[] output = new float[256 * t2];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = (skip[i] + c2[i]) * invSqrt2;
                }
                return output;
            }
            else
            {
                // Standard decode block
                float[] skip = new float[256 * t];
                KittenTensorOps.Conv1D(input, _conv1x1W, ReadOnlySpan<float>.Empty, skip, 322, 256, t, 1, padding: 0);

                float[] n1 = new float[322 * t];
                _norm1.Forward(input, styleHalf, n1, 322, t);
                KittenTensorOps.LeakyRelu(n1, 0.2f);

                float[] c1 = new float[256 * t];
                KittenTensorOps.Conv1D(n1, _conv1W, _conv1B, c1, 322, 256, t, 3, padding: 1);

                float[] n2 = new float[256 * t];
                _norm2.Forward(c1, styleHalf, n2, 256, t);
                KittenTensorOps.LeakyRelu(n2, 0.2f);

                float[] c2 = new float[256 * t];
                KittenTensorOps.Conv1D(n2, _conv2W, _conv2B, c2, 256, 256, t, 3, padding: 1);

                float[] output = new float[256 * t];
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = (skip[i] + c2[i]) * invSqrt2;
                }
                return output;
            }
        }
    }

    private sealed class Generator
    {
        // Upsampling stages
        // ups.0: 256 -> 128 (stride 10, kernel 20, padding 5)
        // ups.1: 128 -> 64  (stride 6,  kernel 12, padding 3)
        private readonly float[] _ups0W; // [256, 128, 20]
        private readonly float[] _ups0B; // [128]
        private readonly float[] _ups1W; // [128, 64, 12]
        private readonly float[] _ups1B; // [64]

        // Noise projections
        // noise_convs.0: 22 -> 128, k=12, stride=6, pad=3
        private readonly float[] _noiseConv0W;
        private readonly float[] _noiseConv0B;
        // noise_convs.1: 22 -> 64,  k=1,  stride=1, pad=0
        private readonly float[] _noiseConv1W;
        private readonly float[] _noiseConv1B;

        // Noise ResBlocks: stage 0 (128ch, k=7), stage 1 (64ch, k=11)
        private readonly AdaInResBlock _noiseRes0;
        private readonly AdaInResBlock _noiseRes1;

        // Parallel AdaIN ResBlocks: 2 per stage (kernel=3)
        private readonly AdaInResBlock[] _resblocks; // 4 resblocks

        // Post conv: 64 -> 22, k=7, pad=3
        private readonly float[] _convPostW;
        private readonly float[] _convPostB;

        // Harmonic Source
        private readonly float[] _mSourceW; // [1, 9]
        private readonly float[] _mSourceB; // [1]

        // STFT filterbank
        private readonly float[] _stftFwdReal; // [11, 1, 20]
        private readonly float[] _stftFwdImag; // [11, 1, 20]
        private readonly float[] _stftBwdReal; // [11, 1, 20]
        private readonly float[] _stftBwdImag; // [11, 1, 20]

        private readonly int _sampleRate;

        public Generator(SafetensorsFile file, KittenConfig cfg)
        {
            _sampleRate = cfg.SampleRate;

            _ups0W = LoadArray(file, "decoder.generator.ups.0.weight");
            _ups0B = LoadArray(file, "decoder.generator.ups.0.bias");
            _ups1W = LoadArray(file, "decoder.generator.ups.1.weight");
            _ups1B = LoadArray(file, "decoder.generator.ups.1.bias");

            _noiseConv0W = LoadArray(file, "decoder.generator.noise_convs.0.weight");
            _noiseConv0B = LoadArray(file, "decoder.generator.noise_convs.0.bias");
            _noiseConv1W = LoadArray(file, "decoder.generator.noise_convs.1.weight");
            _noiseConv1B = LoadArray(file, "decoder.generator.noise_convs.1.bias");

            _noiseRes0 = new AdaInResBlock(file, "decoder.generator.noise_res.0", channels: 128, kernel: 7);
            _noiseRes1 = new AdaInResBlock(file, "decoder.generator.noise_res.1", channels: 64, kernel: 11);

            _resblocks = new AdaInResBlock[4];
            _resblocks[0] = new AdaInResBlock(file, "decoder.generator.resblocks.0", channels: 128, kernel: 3);
            _resblocks[1] = new AdaInResBlock(file, "decoder.generator.resblocks.1", channels: 128, kernel: 3);
            _resblocks[2] = new AdaInResBlock(file, "decoder.generator.resblocks.2", channels: 64, kernel: 3);
            _resblocks[3] = new AdaInResBlock(file, "decoder.generator.resblocks.3", channels: 64, kernel: 3);

            _convPostW = LoadArray(file, "decoder.generator.conv_post.weight");
            _convPostB = LoadArray(file, "decoder.generator.conv_post.bias");

            _mSourceW = LoadArray(file, "decoder.generator.m_source.l_linear.weight");
            _mSourceB = LoadArray(file, "decoder.generator.m_source.l_linear.bias");

            _stftFwdReal = LoadArray(file, "decoder.generator.stft.weight_forward_real");
            _stftFwdImag = LoadArray(file, "decoder.generator.stft.weight_forward_imag");
            _stftBwdReal = LoadArray(file, "decoder.generator.stft.weight_backward_real");
            _stftBwdImag = LoadArray(file, "decoder.generator.stft.weight_backward_imag");
        }

        public float[] Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> f0, ReadOnlySpan<float> styleHalf, int t)
        {
            // 1. Harmonic Source: compute multi-harmonic excitation at audio rate
            int harmonicAudioT = t * 10 * 6 * 5; // T * 300
            float[] f0Audio = new float[harmonicAudioT];
            KittenPredictor.NearestUpsample1D(f0, f0Audio, 1, t, harmonicAudioT);

            float[] harmonicSrc = ComputeHarmonicSource(f0Audio, _sampleRate); // [1, harmonicAudioT]

            // 2. Forward STFT analysis on harmonic source -> [22, tStft]
            float[] noiseStft = StftForwardAnalysis(harmonicSrc);

            // 3. Stage 0: 256 -> 128 (stride 10, kernel 20, pad 5)
            float[] h0Leaky = new float[256 * t];
            x.CopyTo(h0Leaky);
            KittenTensorOps.LeakyRelu(h0Leaky, 0.1f);

            int tStft = noiseStft.Length / 22;
            int tNoise0 = (tStft + 2 * 3 - 12) / 6 + 1;
            float[] noise0 = new float[128 * tNoise0];
            KittenTensorOps.Conv1D(noiseStft, _noiseConv0W, _noiseConv0B, noise0, 22, 128, tStft, 12, padding: 3, stride: 6);
            float[] noise0Res = _noiseRes0.Forward(noise0, styleHalf, tNoise0);

            int tUps0 = (t - 1) * 10 + 20 - 2 * 5; // (t-1)*10 + 10 = 10*t
            float[] ups0 = new float[128 * tUps0];
            KittenTensorOps.ConvTranspose1D(h0Leaky, _ups0W, _ups0B, ups0, 256, 128, t, 20, stride: 10, padding: 5);

            // Match noise to ups0 time dim and add
            AddMatchedNoise(ups0, noise0Res, 128, tUps0, tNoise0);

            // Resblocks 0 and 1 (parallel averaged)
            float[] rb0Out = _resblocks[0].Forward(ups0, styleHalf, tUps0);
            float[] rb1Out = _resblocks[1].Forward(ups0, styleHalf, tUps0);
            float[] h1 = new float[128 * tUps0];
            for (int i = 0; i < h1.Length; i++)
            {
                h1[i] = (rb0Out[i] + rb1Out[i]) * 0.5f;
            }

            // 4. Stage 1: 128 -> 64 (stride 6, kernel 12, pad 3)
            KittenTensorOps.LeakyRelu(h1, 0.1f);

            float[] noise1 = new float[64 * tStft];
            KittenTensorOps.Conv1D(noiseStft, _noiseConv1W, _noiseConv1B, noise1, 22, 64, tStft, 1, padding: 0, stride: 1);
            float[] noise1Res = _noiseRes1.Forward(noise1, styleHalf, tStft);

            int tUps1 = (tUps0 - 1) * 6 + 12 - 2 * 3; // 6 * tUps0
            float[] ups1 = new float[64 * tUps1];
            KittenTensorOps.ConvTranspose1D(h1, _ups1W, _ups1B, ups1, 128, 64, tUps0, 12, stride: 6, padding: 3);

            // Reflect-pad 1 sample at start of time dimension (pads=[1, 0])
            int tUps1Padded = tUps1 + 1;
            float[] ups1Padded = new float[64 * tUps1Padded];
            for (int c = 0; c < 64; c++)
            {
                int srcOff = c * tUps1;
                int dstOff = c * tUps1Padded;
                // Position 0 reflects index 1
                ups1Padded[dstOff] = (tUps1 >= 2) ? ups1[srcOff + 1] : ups1[srcOff];
                for (int ti = 0; ti < tUps1; ti++)
                {
                    ups1Padded[dstOff + 1 + ti] = ups1[srcOff + ti];
                }
            }

            // Match noise to ups1Padded and add
            AddMatchedNoise(ups1Padded, noise1Res, 64, tUps1Padded, tStft);

            // Resblocks 2 and 3 (parallel averaged)
            float[] rb2Out = _resblocks[2].Forward(ups1Padded, styleHalf, tUps1Padded);
            float[] rb3Out = _resblocks[3].Forward(ups1Padded, styleHalf, tUps1Padded);
            float[] h2 = new float[64 * tUps1Padded];
            for (int i = 0; i < h2.Length; i++)
            {
                h2[i] = (rb2Out[i] + rb3Out[i]) * 0.5f;
            }

            // 5. Post Conv: 64 -> 22, k=7, pad=3
            KittenTensorOps.LeakyRelu(h2, 0.01f);
            float[] stftOut = new float[22 * tUps1Padded];
            KittenTensorOps.Conv1D(h2, _convPostW, _convPostB, stftOut, 64, 22, tUps1Padded, 7, padding: 3);

            // 6. STFT Inverse Synthesis -> 24kHz Audio
            float[] rawWaveform = StftInverseSynthesis(stftOut, tUps1Padded);

            // 7. Tanh clamp and trim 5000 samples
            for (int i = 0; i < rawWaveform.Length; i++)
            {
                rawWaveform[i] = MathF.Tanh(rawWaveform[i]);
            }

            int trimLen = Math.Max(0, rawWaveform.Length - 5000);
            if (trimLen > 0)
            {
                float[] finalWave = new float[trimLen];
                Array.Copy(rawWaveform, finalWave, trimLen);
                return finalWave;
            }

            return rawWaveform;
        }

        private static void AddMatchedNoise(Span<float> dst, ReadOnlySpan<float> src, int channels, int dstT, int srcT)
        {
            int copyT = Math.Min(dstT, srcT);
            for (int c = 0; c < channels; c++)
            {
                int dstOff = c * dstT;
                int srcOff = c * srcT;
                for (int t = 0; t < copyT; t++)
                {
                    dst[dstOff + t] += src[srcOff + t];
                }
            }
        }

        private float[] ComputeHarmonicSource(ReadOnlySpan<float> f0, int sampleRate)
        {
            int t = f0.Length;
            int numHarmonics = 9;
            float twoPi = 2.0f * MathF.PI;
            float scale = twoPi / sampleRate;

            float[] harmonicsSin = new float[numHarmonics * t];
            for (int h = 0; h < numHarmonics; h++)
            {
                float k = (h + 1);
                float acc = 0.0f;
                int off = h * t;
                for (int ti = 0; ti < t; ti++)
                {
                    acc += f0[ti] * k * scale;
                    harmonicsSin[off + ti] = MathF.Sin(acc) * 0.1f;
                }
            }

            // Linear combination with _mSourceW [1, 9] + _mSourceB [1], followed by Tanh
            float[] outSignal = new float[t];
            float b = _mSourceB[0];
            for (int ti = 0; ti < t; ti++)
            {
                if (f0[ti] <= 10.0f)
                {
                    outSignal[ti] = 0.0f; // Voiced mask
                    continue;
                }

                float sum = b;
                for (int h = 0; h < numHarmonics; h++)
                {
                    sum += harmonicsSin[h * t + ti] * _mSourceW[h];
                }
                outSignal[ti] = MathF.Tanh(sum);
            }

            return outSignal;
        }

        private float[] StftForwardAnalysis(ReadOnlySpan<float> audio)
        {
            // Edge pad by 10
            int t = audio.Length;
            int padLen = 10;
            int paddedT = t + 2 * padLen;
            float[] padded = new float[paddedT];

            float firstVal = audio[0];
            float lastVal = audio[t - 1];

            for (int i = 0; i < padLen; i++) padded[i] = firstVal;
            audio.CopyTo(padded.AsSpan(padLen, t));
            for (int i = 0; i < padLen; i++) padded[padLen + t + i] = lastVal;

            // Conv1D with real and imag filterbanks: 1 -> 11, k=20, stride=5, pad=0
            int outT = (paddedT - 20) / 5 + 1;
            float[] real = new float[11 * outT];
            float[] imag = new float[11 * outT];

            KittenTensorOps.Conv1D(padded, _stftFwdReal, ReadOnlySpan<float>.Empty, real, 1, 11, paddedT, 20, padding: 0, stride: 5);
            KittenTensorOps.Conv1D(padded, _stftFwdImag, ReadOnlySpan<float>.Empty, imag, 1, 11, paddedT, 20, padding: 0, stride: 5);

            // Polar form: magnitude = sqrt(real^2 + imag^2 + 1e-7), phase = atan2(imag, real)
            float[] output = new float[22 * outT];
            for (int c = 0; c < 11; c++)
            {
                int rOff = c * outT;
                int magOff = c * outT;
                int phaseOff = (11 + c) * outT;

                for (int ti = 0; ti < outT; ti++)
                {
                    float r = real[rOff + ti];
                    float im = imag[rOff + ti];
                    output[magOff + ti] = MathF.Sqrt(r * r + im * im + 1e-7f);
                    output[phaseOff + ti] = MathF.Atan2(im, r);
                }
            }

            return output;
        }

        private float[] StftInverseSynthesis(ReadOnlySpan<float> stftIn, int tStft)
        {
            // stftIn: [22, tStft] -> first 11 is log_amp, next 11 is raw_phase
            float[] realPart = new float[11 * tStft];
            float[] imagPart = new float[11 * tStft];

            for (int c = 0; c < 11; c++)
            {
                int logAmpOff = c * tStft;
                int phaseOff = (11 + c) * tStft;

                for (int ti = 0; ti < tStft; ti++)
                {
                    float amp = MathF.Exp(stftIn[logAmpOff + ti]);
                    float rawPhase = stftIn[phaseOff + ti];
                    float phase1 = MathF.Sin(rawPhase);
                    float sinP = MathF.Sin(phase1);
                    float cosP = MathF.Cos(phase1);

                    realPart[logAmpOff + ti] = amp * cosP;
                    imagPart[logAmpOff + ti] = amp * sinP;
                }
            }

            // ConvTranspose1D: 11 -> 1, k=20, stride=5, pad=0
            int outAudioT = (tStft - 1) * 5 + 20;
            float[] wvReal = new float[outAudioT];
            float[] wvImag = new float[outAudioT];

            KittenTensorOps.ConvTranspose1D(realPart, _stftBwdReal, ReadOnlySpan<float>.Empty, wvReal, 11, 1, tStft, 20, stride: 5, padding: 0);
            KittenTensorOps.ConvTranspose1D(imagPart, _stftBwdImag, ReadOnlySpan<float>.Empty, wvImag, 11, 1, tStft, 20, stride: 5, padding: 0);

            // waveform = wv_real - wv_imag, trim 10 samples from each end
            int trimStart = Math.Min(10, outAudioT);
            int trimEnd = Math.Max(trimStart, outAudioT - 10);
            int finalLen = trimEnd - trimStart;

            float[] waveform = new float[finalLen];
            for (int i = 0; i < finalLen; i++)
            {
                waveform[i] = wvReal[trimStart + i] - wvImag[trimStart + i];
            }

            return waveform;
        }
    }

    /// <summary>
    /// HiFi-GAN AdaIN ResBlock with Snake activation.
    /// 3 dilated branches [1, 3, 5], each with adain1 -> snake -> convs1 -> adain2 -> snake -> convs2 -> residual add.
    /// </summary>
    private sealed class AdaInResBlock
    {
        private readonly KittenPredictor.AdaInLayer[] _adain1;
        private readonly KittenPredictor.AdaInLayer[] _adain2;
        private readonly float[][] _alpha1; // [1, channels, 1]
        private readonly float[][] _alpha2;
        private readonly float[][] _convs1W;
        private readonly float[][] _convs1B;
        private readonly float[][] _convs2W;
        private readonly float[][] _convs2B;
        private readonly int[] _dilations = { 1, 3, 5 };
        private readonly int _channels;
        private readonly int _kernel;

        public AdaInResBlock(SafetensorsFile file, string prefix, int channels, int kernel)
        {
            _channels = channels;
            _kernel = kernel;

            _adain1 = new KittenPredictor.AdaInLayer[3];
            _adain2 = new KittenPredictor.AdaInLayer[3];
            _alpha1 = new float[3][];
            _alpha2 = new float[3][];
            _convs1W = new float[3][];
            _convs1B = new float[3][];
            _convs2W = new float[3][];
            _convs2B = new float[3][];

            for (int j = 0; j < 3; j++)
            {
                _adain1[j] = new KittenPredictor.AdaInLayer(file, $"{prefix}.adain1.{j}", styleDim: 128, channels: channels);
                _adain2[j] = new KittenPredictor.AdaInLayer(file, $"{prefix}.adain2.{j}", styleDim: 128, channels: channels);

                _alpha1[j] = LoadArray(file, $"{prefix}.alpha1.{j}");
                _alpha2[j] = LoadArray(file, $"{prefix}.alpha2.{j}");

                _convs1W[j] = LoadArray(file, $"{prefix}.convs1.{j}.weight");
                _convs1B[j] = LoadArray(file, $"{prefix}.convs1.{j}.bias");

                _convs2W[j] = LoadArray(file, $"{prefix}.convs2.{j}.weight");
                _convs2B[j] = LoadArray(file, $"{prefix}.convs2.{j}.bias");
            }
        }

        public float[] Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> styleHalf, int t)
        {
            float[] output = new float[_channels * t];
            input.CopyTo(output);

            float[] h = new float[_channels * t];
            float[] convOut = new float[_channels * t];

            for (int j = 0; j < 3; j++)
            {
                int dil = _dilations[j];
                int pad1 = dil * (_kernel - 1) / 2;
                int pad2 = (_kernel - 1) / 2;

                // adain1 -> snake(alpha1) -> convs1 (with dilation)
                _adain1[j].Forward(output, styleHalf, h, _channels, t);
                ApplySnake(h, _alpha1[j], _channels, t);
                KittenTensorOps.Conv1D(h, _convs1W[j], _convs1B[j], convOut, _channels, _channels, t, _kernel, padding: pad1, stride: 1, dilation: dil);

                // adain2 -> snake(alpha2) -> convs2 (no dilation)
                _adain2[j].Forward(convOut, styleHalf, h, _channels, t);
                ApplySnake(h, _alpha2[j], _channels, t);
                KittenTensorOps.Conv1D(h, _convs2W[j], _convs2B[j], convOut, _channels, _channels, t, _kernel, padding: pad2, stride: 1, dilation: 1);

                // Residual sum
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] += convOut[i];
                }
            }

            return output;
        }

        private static unsafe void ApplySnake(Span<float> x, ReadOnlySpan<float> alpha, int channels, int time)
        {
            // Snake activation: x + sin^2(alpha * x) / alpha
            fixed (float* pX = x)
            fixed (float* pAlpha = alpha)
            {
                float* xPtr = pX;
                float* aPtr = pAlpha;

                Parallel.For(0, channels, c =>
                {
                    float a = MathF.Max(aPtr[c], 1e-9f);
                    float invA = 1.0f / a;
                    int off = c * time;

                    for (int t = 0; t < time; t++)
                    {
                        float v = xPtr[off + t];
                        float s = MathF.Sin(a * v);
                        xPtr[off + t] = v + (s * s) * invA;
                    }
                });
            }
        }
    }
}
