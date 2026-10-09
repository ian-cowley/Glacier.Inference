namespace Glacier.Inference.Audio.Kitten;

using System;
using Glacier.Inference.Format;

/// <summary>
/// Predictor module for StyleTTS 2 / KittenTTS.
/// Predicts phoneme durations, shared acoustic features, pitch (F0), and voicing energy (N).
/// </summary>
public sealed class KittenPredictor
{
    private readonly KittenBiLstm _durLstm;
    private readonly float[] _durationProjWeight; // [50, 128]
    private readonly float[] _durationProjBias;   // [50]

    private readonly KittenBiLstm _sharedLstm;

    private readonly PredictorBranch _f0Branch;
    private readonly PredictorBranch _nBranch;

    private const int MaxDuration = 50;
    private const int LstmHidden = 64;
    private const int LstmOut = 128;
    private const int StyleHalf = 128;

    public KittenPredictor(SafetensorsFile file)
    {
        _durLstm = new KittenBiLstm(
            file,
            "predictor.lstm.W",
            "predictor.lstm.R",
            "predictor.lstm.B",
            inputSize: 256,
            hiddenSize: LstmHidden);

        _durationProjWeight = LoadArray(file, "predictor.duration_proj.linear_layer.weight");
        _durationProjBias = LoadArray(file, "predictor.duration_proj.linear_layer.bias");

        _sharedLstm = new KittenBiLstm(
            file,
            "shared.lstm.W",
            "shared.lstm.R",
            "shared.lstm.B",
            inputSize: 256,
            hiddenSize: LstmHidden);

        _f0Branch = new PredictorBranch(file, "predictor.F0", "predictor.F0_proj");
        _nBranch = new PredictorBranch(file, "predictor.N", "predictor.N_proj");
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
    /// Executes duration prediction, length expansion, shared BiLSTM, and F0/N prediction.
    /// Returns:
    /// - durations: [seqLen]
    /// - sharedLstmOut: [128, T]
    /// - f0: [1, 2T]
    /// - nAmp: [1, 2T]
    /// </summary>
    public (int[] durations, float[] sharedLstmOut, float[] f0, float[] nAmp) Forward(
        ReadOnlySpan<float> textFeatures, // [seqLen, 256]
        ReadOnlySpan<float> style,        // [256]
        float speed = 1.0f)
    {
        int seqLen = textFeatures.Length / 256;
        if (seqLen == 0) return (Array.Empty<int>(), Array.Empty<float>(), Array.Empty<float>(), Array.Empty<float>());

        var styleHalf = style.Slice(StyleHalf, StyleHalf);

        // 1. Duration prediction
        float[] durHidden = new float[seqLen * LstmOut];
        _durLstm.Forward(textFeatures, durHidden, seqLen);

        int[] durations = new int[seqLen];
        Span<float> logits = stackalloc float[MaxDuration];
        int totalFrames = 0;

        for (int s = 0; s < seqLen; s++)
        {
            var hiddenSlice = new ReadOnlySpan<float>(durHidden, s * LstmOut, LstmOut);
            KittenTensorOps.Linear(hiddenSlice, _durationProjWeight, _durationProjBias, logits, LstmOut, MaxDuration);

            float probSum = 0f;
            for (int k = 0; k < MaxDuration; k++)
            {
                probSum += KittenTensorOps.Sigmoid(logits[k]);
            }

            float durScaled = probSum / MathF.Max(speed, 0.1f);
            int d = Math.Clamp((int)MathF.Round(durScaled), 1, MaxDuration);
            durations[s] = d;
            totalFrames += d;
        }

        if (totalFrames == 0) totalFrames = 1;

        // 2. Length Regulation: repeat textFeatures [seqLen, 256] -> [totalFrames, 256]
        float[] expanded = new float[totalFrames * 256];
        int frameIdx = 0;
        for (int s = 0; s < seqLen; s++)
        {
            var tokenFeat = textFeatures.Slice(s * 256, 256);
            int d = durations[s];
            for (int r = 0; r < d; r++)
            {
                var dst = expanded.AsSpan((frameIdx++) * 256, 256);
                tokenFeat.CopyTo(dst);
            }
        }

        // 3. Shared BiLSTM over expanded features -> [totalFrames, 128]
        float[] sharedHiddenTimeFirst = new float[totalFrames * LstmOut];
        _sharedLstm.Forward(expanded, sharedHiddenTimeFirst, totalFrames);

        // Transpose to channel-first [128, totalFrames]
        float[] sharedLstmOut = new float[LstmOut * totalFrames];
        for (int t = 0; t < totalFrames; t++)
        {
            for (int c = 0; c < LstmOut; c++)
            {
                sharedLstmOut[c * totalFrames + t] = sharedHiddenTimeFirst[t * LstmOut + c];
            }
        }

        // 4. Predict F0 and N branches (each outputs [1, 2 * totalFrames])
        float[] f0 = _f0Branch.Forward(sharedLstmOut, styleHalf, totalFrames);
        float[] nAmp = _nBranch.Forward(sharedLstmOut, styleHalf, totalFrames);

        return (durations, sharedLstmOut, f0, nAmp);
    }

    /// <summary>
    /// Predictor branch (for F0 or N):
    /// Block0 (128 -> 128) -> Block1 (128 -> 64, pool upsamples T -> 2T) -> Block2 (64 -> 64) -> proj (64 -> 1).
    /// </summary>
    private sealed class PredictorBranch
    {
        // Block 0: 128 -> 128
        private readonly AdaInLayer _b0Norm1;
        private readonly float[] _b0Conv1W; // [128, 128, 3]
        private readonly float[] _b0Conv1B; // [128]
        private readonly AdaInLayer _b0Norm2;
        private readonly float[] _b0Conv2W; // [128, 128, 3]
        private readonly float[] _b0Conv2B; // [128]

        // Block 1: 128 -> 64 (upsamples T -> 2T)
        private readonly float[] _b1Conv1x1W; // [64, 128, 1]
        private readonly AdaInLayer _b1Norm1;
        private readonly float[] _b1PoolW;    // [128, 1, 3]
        private readonly float[] _b1PoolB;    // [128]
        private readonly float[] _b1Conv1W;   // [64, 128, 3]
        private readonly float[] _b1Conv1B;   // [64]
        private readonly AdaInLayer _b1Norm2;
        private readonly float[] _b1Conv2W;   // [64, 64, 3]
        private readonly float[] _b1Conv2B;   // [64]

        // Block 2: 64 -> 64
        private readonly AdaInLayer _b2Norm1;
        private readonly float[] _b2Conv1W; // [64, 64, 3]
        private readonly float[] _b2Conv1B; // [64]
        private readonly AdaInLayer _b2Norm2;
        private readonly float[] _b2Conv2W; // [64, 64, 3]
        private readonly float[] _b2Conv2B; // [64]

        // Projection: 64 -> 1
        private readonly float[] _projW; // [1, 64, 1]
        private readonly float[] _projB; // [1]

        public PredictorBranch(SafetensorsFile file, string prefix, string projName)
        {
            // Block 0
            _b0Norm1 = new AdaInLayer(file, $"{prefix}.0.norm1", styleDim: 128, channels: 128);
            _b0Conv1W = LoadArray(file, $"{prefix}.0.conv1.weight");
            _b0Conv1B = LoadArray(file, $"{prefix}.0.conv1.bias");
            // norm2 reuses norm1 affine weights
            _b0Norm2 = new AdaInLayer(file, $"{prefix}.0.norm2", styleDim: 128, channels: 128, _b0Norm1.NormWeight, _b0Norm1.NormBias);
            _b0Conv2W = LoadArray(file, $"{prefix}.0.conv2.weight");
            _b0Conv2B = LoadArray(file, $"{prefix}.0.conv2.bias");

            // Block 1
            _b1Conv1x1W = LoadArray(file, $"{prefix}.1.conv1x1.weight");
            // norm1 reuses Block0 norm1 affine weights
            _b1Norm1 = new AdaInLayer(file, $"{prefix}.1.norm1", styleDim: 128, channels: 128, _b0Norm1.NormWeight, _b0Norm1.NormBias);
            _b1PoolW = LoadArray(file, $"{prefix}.1.pool.weight");
            _b1PoolB = LoadArray(file, $"{prefix}.1.pool.bias");
            _b1Conv1W = LoadArray(file, $"{prefix}.1.conv1.weight");
            _b1Conv1B = LoadArray(file, $"{prefix}.1.conv1.bias");
            _b1Norm2 = new AdaInLayer(file, $"{prefix}.1.norm2", styleDim: 128, channels: 64);
            _b1Conv2W = LoadArray(file, $"{prefix}.1.conv2.weight");
            _b1Conv2B = LoadArray(file, $"{prefix}.1.conv2.bias");

            // Block 2
            _b2Norm1 = new AdaInLayer(file, $"{prefix}.2.norm1", styleDim: 128, channels: 64);
            _b2Conv1W = LoadArray(file, $"{prefix}.2.conv1.weight");
            _b2Conv1B = LoadArray(file, $"{prefix}.2.conv1.bias");
            _b2Norm2 = new AdaInLayer(file, $"{prefix}.2.norm2", styleDim: 128, channels: 64);
            _b2Conv2W = LoadArray(file, $"{prefix}.2.conv2.weight");
            _b2Conv2B = LoadArray(file, $"{prefix}.2.conv2.bias");

            // Proj
            _projW = LoadArray(file, $"{projName}.weight");
            _projB = LoadArray(file, $"{projName}.bias");
        }

        public float[] Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> styleHalf, int t)
        {
            float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);

            // ── Block 0 (128ch, T frames) ──
            float[] b0Norm1 = new float[128 * t];
            _b0Norm1.Forward(input, styleHalf, b0Norm1, 128, t);
            KittenTensorOps.LeakyRelu(b0Norm1, 0.2f);

            float[] b0Conv1 = new float[128 * t];
            KittenTensorOps.Conv1D(b0Norm1, _b0Conv1W, _b0Conv1B, b0Conv1, 128, 128, t, 3, padding: 1);

            float[] b0Norm2 = new float[128 * t];
            _b0Norm2.Forward(b0Conv1, styleHalf, b0Norm2, 128, t);
            KittenTensorOps.LeakyRelu(b0Norm2, 0.2f);

            float[] b0Conv2 = new float[128 * t];
            KittenTensorOps.Conv1D(b0Norm2, _b0Conv2W, _b0Conv2B, b0Conv2, 128, 128, t, 3, padding: 1);

            float[] b0Out = new float[128 * t];
            for (int i = 0; i < b0Out.Length; i++)
            {
                b0Out[i] = (input[i] + b0Conv2[i]) * invSqrt2;
            }

            // ── Block 1 (128ch -> 64ch, T -> 2T frames) ──
            int t2 = t * 2;

            // Skip: nearest upsample T -> 2T, then conv1x1 (no bias)
            float[] b0Up = new float[128 * t2];
            NearestUpsample1D(b0Out, b0Up, 128, t, t2);

            float[] b1Skip = new float[64 * t2];
            KittenTensorOps.Conv1D(b0Up, _b1Conv1x1W, ReadOnlySpan<float>.Empty, b1Skip, 128, 64, t2, 1, padding: 0);

            // Main path: norm1 on T -> leaky_relu -> pool upsample to 2T -> conv1 -> norm2 -> leaky_relu -> conv2
            float[] b1Norm1 = new float[128 * t];
            _b1Norm1.Forward(b0Out, styleHalf, b1Norm1, 128, t);
            KittenTensorOps.LeakyRelu(b1Norm1, 0.2f);

            float[] b1Pool = new float[128 * t2];
            DepthwiseConvTranspose1D(b1Norm1, _b1PoolW, _b1PoolB, b1Pool, 128, t, stride: 2, padding: 1, outputPadding: 1, kernelSize: 3);

            float[] b1Conv1 = new float[64 * t2];
            KittenTensorOps.Conv1D(b1Pool, _b1Conv1W, _b1Conv1B, b1Conv1, 128, 64, t2, 3, padding: 1);

            float[] b1Norm2 = new float[64 * t2];
            _b1Norm2.Forward(b1Conv1, styleHalf, b1Norm2, 64, t2);
            KittenTensorOps.LeakyRelu(b1Norm2, 0.2f);

            float[] b1Conv2 = new float[64 * t2];
            KittenTensorOps.Conv1D(b1Norm2, _b1Conv2W, _b1Conv2B, b1Conv2, 64, 64, t2, 3, padding: 1);

            float[] b1Out = new float[64 * t2];
            for (int i = 0; i < b1Out.Length; i++)
            {
                b1Out[i] = (b1Skip[i] + b1Conv2[i]) * invSqrt2;
            }

            // ── Block 2 (64ch, 2T frames) ──
            float[] b2Norm1 = new float[64 * t2];
            _b2Norm1.Forward(b1Out, styleHalf, b2Norm1, 64, t2);
            KittenTensorOps.LeakyRelu(b2Norm1, 0.2f);

            float[] b2Conv1 = new float[64 * t2];
            KittenTensorOps.Conv1D(b2Norm1, _b2Conv1W, _b2Conv1B, b2Conv1, 64, 64, t2, 3, padding: 1);

            float[] b2Norm2 = new float[64 * t2];
            _b2Norm2.Forward(b2Conv1, styleHalf, b2Norm2, 64, t2);
            KittenTensorOps.LeakyRelu(b2Norm2, 0.2f);

            float[] b2Conv2 = new float[64 * t2];
            KittenTensorOps.Conv1D(b2Norm2, _b2Conv2W, _b2Conv2B, b2Conv2, 64, 64, t2, 3, padding: 1);

            float[] b2Out = new float[64 * t2];
            for (int i = 0; i < b2Out.Length; i++)
            {
                b2Out[i] = (b1Out[i] + b2Conv2[i]) * invSqrt2;
            }

            // ── Projection: 64 -> 1 (at 2T frames) ──
            float[] projOut = new float[1 * t2];
            KittenTensorOps.Conv1D(b2Out, _projW, _projB, projOut, 64, 1, t2, 1, padding: 0);

            return projOut;
        }
    }

    /// <summary>
    /// Reusable AdaIN layer with optional instance norm affine parameters (gamma, beta).
    /// </summary>
    internal sealed class AdaInLayer
    {
        private readonly float[] _fcWeight; // [2*channels, styleDim]
        private readonly float[] _fcBias;   // [2*channels]
        public float[]? NormWeight { get; }
        public float[]? NormBias { get; }

        public AdaInLayer(SafetensorsFile file, string prefix, int styleDim, int channels, float[]? sharedNormW = null, float[]? sharedNormB = null)
        {
            _fcWeight = LoadArray(file, $"{prefix}.fc.weight");
            _fcBias = LoadArray(file, $"{prefix}.fc.bias");

            if (sharedNormW != null && sharedNormB != null)
            {
                NormWeight = sharedNormW;
                NormBias = sharedNormB;
            }
            else
            {
                if (file.ContainsTensor($"{prefix}.norm.weight"))
                {
                    NormWeight = LoadArray(file, $"{prefix}.norm.weight");
                    NormBias = LoadArray(file, $"{prefix}.norm.bias");
                }
            }
        }

        public void Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> style, Span<float> output, int channels, int time)
        {
            // 1. Instance norm (per-channel over time)
            KittenTensorOps.InstanceNorm(input, output, channels, time);

            // Optional affine parameters
            if (NormWeight != null && NormBias != null)
            {
                for (int c = 0; c < channels; c++)
                {
                    float w = NormWeight[c];
                    float b = NormBias[c];
                    int offset = c * time;
                    for (int t = 0; t < time; t++)
                    {
                        output[offset + t] = output[offset + t] * w + b;
                    }
                }
            }

            // 2. Project style: style * fcW^T + fcB -> [2 * channels]
            Span<float> proj = stackalloc float[2 * channels];
            KittenTensorOps.Linear(style, _fcWeight, _fcBias, proj, style.Length, 2 * channels);

            // 3. AdaIN modulation: output * (gamma + 1.0) + beta
            for (int c = 0; c < channels; c++)
            {
                float gamma = proj[c] + 1.0f;
                float beta = proj[channels + c];
                int offset = c * time;
                for (int t = 0; t < time; t++)
                {
                    output[offset + t] = output[offset + t] * gamma + beta;
                }
            }
        }
    }

    /// <summary>
    /// Nearest-neighbor 1D temporal upsampling for channel-first tensors [channels, tIn] -> [channels, tOut].
    /// </summary>
    internal static unsafe void NearestUpsample1D(ReadOnlySpan<float> input, Span<float> output, int channels, int tIn, int tOut)
    {
        if (tIn == 0 || tOut == 0) return;
        float scale = (float)tIn / tOut;

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* outPtr = pOut;

            Parallel.For(0, channels, c =>
            {
                int inOffset = c * tIn;
                int outOffset = c * tOut;

                for (int outIdx = 0; outIdx < tOut; outIdx++)
                {
                    int inIdx = Math.Min((int)(outIdx * scale), tIn - 1);
                    outPtr[outOffset + outIdx] = inPtr[inOffset + inIdx];
                }
            });
        }
    }

    /// <summary>
    /// Depthwise ConvTranspose1D: each channel uses its own kernel [1, kernelSize].
    /// </summary>
    internal static unsafe void DepthwiseConvTranspose1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int channels,
        int tIn,
        int stride,
        int padding,
        int outputPadding,
        int kernelSize)
    {
        int tOut = (tIn - 1) * stride + kernelSize + outputPadding - 2 * padding;
        if (tOut <= 0) return;

        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pB = bias)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* wPtr = pW;
            float* bPtr = pB;
            float* outPtr = pOut;

            Parallel.For(0, channels, c =>
            {
                int inOffset = c * tIn;
                int outOffset = c * tOut;
                int wOffset = c * kernelSize;
                float b = bPtr != null ? bPtr[c] : 0.0f;

                for (int t = 0; t < tOut; t++)
                {
                    outPtr[outOffset + t] = b;
                }

                for (int i = 0; i < tIn; i++)
                {
                    float inVal = inPtr[inOffset + i];
                    if (inVal == 0.0f) continue;
                    int outStart = i * stride;

                    for (int k = 0; k < kernelSize; k++)
                    {
                        int outIdx = outStart + k;
                        if (outIdx >= padding && (outIdx - padding) < tOut)
                        {
                            outPtr[outOffset + outIdx - padding] += inVal * wPtr[wOffset + k];
                        }
                    }
                }
            });
        }
    }
}
