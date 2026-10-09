namespace Glacier.Inference.Audio.Kitten;

using System;
using Glacier.Inference.Format;

/// <summary>
/// StyleTTS 2 / KittenTTS Text Encoder.
/// Combines CNN feature extraction path (for Decoder) and AdaIN BiLSTM chain path (for Predictor).
/// </summary>
public sealed class KittenTextEncoder
{
    private readonly float[] _embedding; // [178 * 128]

    // CNN path
    private readonly float[] _cnn0Weight; // [128, 128, 5]
    private readonly float[] _cnn0Bias;   // [128]
    private readonly float[] _cnn0Gamma;  // [128]
    private readonly float[] _cnn0Beta;   // [128]

    private readonly float[] _cnn1Weight; // [128, 128, 5]
    private readonly float[] _cnn1Bias;   // [128]
    private readonly float[] _cnn1Gamma;  // [128]
    private readonly float[] _cnn1Beta;   // [128]

    private readonly KittenBiLstm _cnnLstm;

    // LSTM chain path
    private readonly KittenBiLstm _lstm0;
    private readonly float[] _adain1FcWeight; // [256, 128]
    private readonly float[] _adain1FcBias;   // [256]

    private readonly KittenBiLstm _lstm2;
    private readonly float[] _adain3FcWeight; // [256, 128]
    private readonly float[] _adain3FcBias;   // [256]

    private const int HiddenDim = 128;
    private const int LstmHidden = 64;
    private const int StyleHalf = 128;

    public KittenTextEncoder(SafetensorsFile file)
    {
        _embedding = LoadArray(file, "text_encoder.embedding.weight");

        _cnn0Weight = LoadArray(file, "text_encoder.cnn.0.0.weight");
        _cnn0Bias = LoadArray(file, "text_encoder.cnn.0.0.bias");
        _cnn0Gamma = LoadArray(file, "text_encoder.cnn.0.1.gamma");
        _cnn0Beta = LoadArray(file, "text_encoder.cnn.0.1.beta");

        _cnn1Weight = LoadArray(file, "text_encoder.cnn.1.0.weight");
        _cnn1Bias = LoadArray(file, "text_encoder.cnn.1.0.bias");
        _cnn1Gamma = LoadArray(file, "text_encoder.cnn.1.1.gamma");
        _cnn1Beta = LoadArray(file, "text_encoder.cnn.1.1.beta");

        _cnnLstm = new KittenBiLstm(
            file,
            "predictor.text_encoder.lstm.W",
            "predictor.text_encoder.lstm.R",
            "predictor.text_encoder.lstm.B",
            inputSize: HiddenDim,
            hiddenSize: LstmHidden);

        _lstm0 = new KittenBiLstm(
            file,
            "predictor.text_encoder.lstms.0.W",
            "predictor.text_encoder.lstms.0.R",
            "predictor.text_encoder.lstms.0.B",
            inputSize: HiddenDim + StyleHalf,
            hiddenSize: LstmHidden);

        _adain1FcWeight = LoadArray(file, "predictor.text_encoder.lstms.1.fc.weight");
        _adain1FcBias = LoadArray(file, "predictor.text_encoder.lstms.1.fc.bias");

        _lstm2 = new KittenBiLstm(
            file,
            "predictor.text_encoder.lstms.2.W",
            "predictor.text_encoder.lstms.2.R",
            "predictor.text_encoder.lstms.2.B",
            inputSize: HiddenDim + StyleHalf,
            hiddenSize: LstmHidden);

        _adain3FcWeight = LoadArray(file, "predictor.text_encoder.lstms.3.fc.weight");
        _adain3FcBias = LoadArray(file, "predictor.text_encoder.lstms.3.fc.bias");
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
    /// Executes Text Encoder forward pass.
    /// Returns:
    /// - lstmFeatures: [seqLen, 256] for Predictor
    /// - cnnFeatures:  [128, seqLen] for Decoder
    /// </summary>
    public (float[] lstmFeatures, float[] cnnFeatures) Forward(
        ReadOnlySpan<float> bertOutput,
        ReadOnlySpan<int> inputIds,
        ReadOnlySpan<float> style)
    {
        int seqLen = inputIds.Length;
        if (seqLen == 0) return (Array.Empty<float>(), Array.Empty<float>());

        // Style for LSTM chain is the second half: style[128..255]
        var styleForLstm = style.Slice(StyleHalf, StyleHalf);

        // ══════════════════════════════════════════════════════════════════════
        // 1. CNN PATH (for Decoder)
        // ══════════════════════════════════════════════════════════════════════
        // a) Embedding [128, seqLen] in channel-first order
        float[] cnnIn = new float[HiddenDim * seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            int token = Math.Clamp(inputIds[i], 0, 177);
            for (int c = 0; c < HiddenDim; c++)
            {
                cnnIn[c * seqLen + i] = _embedding[token * HiddenDim + c];
            }
        }

        // b) CNN 0: Conv1D(k=5, p=2) -> LayerNorm over channels -> LeakyReLU(0.2)
        // Output stays [128, seqLen]
        float[] cnn0Out = new float[HiddenDim * seqLen];
        KittenTensorOps.Conv1D(cnnIn, _cnn0Weight, _cnn0Bias, cnn0Out, HiddenDim, HiddenDim, seqLen, 5, padding: 2);
        ApplyCnnNormAndAct(cnn0Out, _cnn0Gamma, _cnn0Beta, seqLen, transposeBack: true);

        // c) CNN 1: Conv1D(k=5, p=2) -> LayerNorm over channels -> LeakyReLU(0.2)
        // Output is transposed to [seqLen, 128] for LSTM
        float[] cnn1Out = new float[HiddenDim * seqLen];
        KittenTensorOps.Conv1D(cnn0Out, _cnn1Weight, _cnn1Bias, cnn1Out, HiddenDim, HiddenDim, seqLen, 5, padding: 2);
        float[] cnnTimeFirst = new float[seqLen * HiddenDim];
        ApplyCnnNormAndActTransposed(cnn1Out, _cnn1Gamma, _cnn1Beta, cnnTimeFirst, seqLen);

        // d) CNN BiLSTM: [seqLen, 128] -> [seqLen, 128]
        float[] cnnLstmOut = new float[seqLen * HiddenDim];
        _cnnLstm.Forward(cnnTimeFirst, cnnLstmOut, seqLen);

        // e) Transpose to channel-first [128, seqLen] for Decoder
        float[] cnnFeatures = new float[HiddenDim * seqLen];
        for (int t = 0; t < seqLen; t++)
        {
            for (int c = 0; c < HiddenDim; c++)
            {
                cnnFeatures[c * seqLen + t] = cnnLstmOut[t * HiddenDim + c];
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // 2. LSTM CHAIN PATH (for Predictor)
        // ══════════════════════════════════════════════════════════════════════
        // a) lstm0 input: concat(bertOutput [seqLen, 128], broadcast styleForLstm [128]) -> [seqLen, 256]
        float[] lstm0In = new float[seqLen * (HiddenDim + StyleHalf)];
        for (int t = 0; t < seqLen; t++)
        {
            var bertSlice = bertOutput.Slice(t * HiddenDim, HiddenDim);
            var dstSlice = lstm0In.AsSpan(t * (HiddenDim + StyleHalf), HiddenDim + StyleHalf);
            bertSlice.CopyTo(dstSlice.Slice(0, HiddenDim));
            styleForLstm.CopyTo(dstSlice.Slice(HiddenDim, StyleHalf));
        }

        float[] lstm0Out = new float[seqLen * HiddenDim];
        _lstm0.Forward(lstm0In, lstm0Out, seqLen);

        // b) AdaIN 1 over lstm0Out
        float[] adain1Out = new float[seqLen * HiddenDim];
        ApplyLstmChainAdaIn(lstm0Out, styleForLstm, _adain1FcWeight, _adain1FcBias, adain1Out, seqLen, HiddenDim);

        // c) lstm2 input: concat(adain1Out [seqLen, 128], broadcast styleForLstm [128]) -> [seqLen, 256]
        float[] lstm2In = new float[seqLen * (HiddenDim + StyleHalf)];
        for (int t = 0; t < seqLen; t++)
        {
            var adainSlice = new ReadOnlySpan<float>(adain1Out, t * HiddenDim, HiddenDim);
            var dstSlice = lstm2In.AsSpan(t * (HiddenDim + StyleHalf), HiddenDim + StyleHalf);
            adainSlice.CopyTo(dstSlice.Slice(0, HiddenDim));
            styleForLstm.CopyTo(dstSlice.Slice(HiddenDim, StyleHalf));
        }

        float[] lstm2Out = new float[seqLen * HiddenDim];
        _lstm2.Forward(lstm2In, lstm2Out, seqLen);

        // d) AdaIN 3 over lstm2Out
        float[] adain3Out = new float[seqLen * HiddenDim];
        ApplyLstmChainAdaIn(lstm2Out, styleForLstm, _adain3FcWeight, _adain3FcBias, adain3Out, seqLen, HiddenDim);

        // e) Final concatenation with style: concat(adain3Out [seqLen, 128], broadcast styleForLstm [128]) -> [seqLen, 256]
        float[] lstmFeatures = new float[seqLen * (HiddenDim + StyleHalf)];
        for (int t = 0; t < seqLen; t++)
        {
            var adainSlice = new ReadOnlySpan<float>(adain3Out, t * HiddenDim, HiddenDim);
            var dstSlice = lstmFeatures.AsSpan(t * (HiddenDim + StyleHalf), HiddenDim + StyleHalf);
            adainSlice.CopyTo(dstSlice.Slice(0, HiddenDim));
            styleForLstm.CopyTo(dstSlice.Slice(HiddenDim, StyleHalf));
        }

        return (lstmFeatures, cnnFeatures);
    }

    private static void ApplyCnnNormAndAct(
        Span<float> cnnData,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        int length,
        bool transposeBack)
    {
        // cnnData is [128, length]
        // LayerNorm is applied over the 128 channels for each time step
        Span<float> stepFeatures = stackalloc float[HiddenDim];
        Span<float> normedFeatures = stackalloc float[HiddenDim];

        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < HiddenDim; c++)
            {
                stepFeatures[c] = cnnData[c * length + t];
            }

            KittenTensorOps.LayerNorm(stepFeatures, gamma, beta, normedFeatures, 1, HiddenDim);
            KittenTensorOps.LeakyRelu(normedFeatures, 0.2f);

            for (int c = 0; c < HiddenDim; c++)
            {
                cnnData[c * length + t] = normedFeatures[c];
            }
        }
    }

    private static void ApplyCnnNormAndActTransposed(
        ReadOnlySpan<float> cnnData,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        Span<float> dstTimeFirst,
        int length)
    {
        // cnnData is [128, length], dstTimeFirst is [length, 128]
        Span<float> stepFeatures = stackalloc float[HiddenDim];
        Span<float> normedFeatures = stackalloc float[HiddenDim];

        for (int t = 0; t < length; t++)
        {
            for (int c = 0; c < HiddenDim; c++)
            {
                stepFeatures[c] = cnnData[c * length + t];
            }

            KittenTensorOps.LayerNorm(stepFeatures, gamma, beta, normedFeatures, 1, HiddenDim);
            KittenTensorOps.LeakyRelu(normedFeatures, 0.2f);

            var dstSlice = dstTimeFirst.Slice(t * HiddenDim, HiddenDim);
            normedFeatures.CopyTo(dstSlice);
        }
    }

    private static void ApplyLstmChainAdaIn(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> style,
        ReadOnlySpan<float> fcW,
        ReadOnlySpan<float> fcB,
        Span<float> output,
        int seqLen,
        int featDim)
    {
        // In text_encoder.rs AdaIn:
        // 1. LayerNorm over featDim for each step t
        // 2. style projection: proj = style * fcW^T + fcB (length 2 * featDim)
        // 3. scale = proj[0..featDim] + 1, bias = proj[featDim..2*featDim]
        // 4. out[t] = normed[t] * scale + bias
        Span<float> proj = stackalloc float[2 * featDim];
        KittenTensorOps.Linear(style, fcW, fcB, proj, featDim, 2 * featDim);

        var scale = proj.Slice(0, featDim);
        var bias = proj.Slice(featDim, featDim);

        Span<float> normed = stackalloc float[featDim];

        for (int t = 0; t < seqLen; t++)
        {
            var xt = x.Slice(t * featDim, featDim);
            var outT = output.Slice(t * featDim, featDim);

            // LayerNorm with gamma=1, beta=0
            KittenTensorOps.LayerNorm(xt, ReadOnlySpan<float>.Empty, ReadOnlySpan<float>.Empty, normed, 1, featDim);

            for (int i = 0; i < featDim; i++)
            {
                outT[i] = normed[i] * (scale[i] + 1.0f) + bias[i];
            }
        }
    }
}
