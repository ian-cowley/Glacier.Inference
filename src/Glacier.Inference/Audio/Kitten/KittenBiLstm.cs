namespace Glacier.Inference.Audio.Kitten;

using System;
using Glacier.Inference.Format;

/// <summary>
/// High-performance bidirectional LSTM module for KittenTTS text encoder and predictor.
/// Weights follow ONNX layout: W [2, 4*H, I], R [2, 4*H, H], B [2, 8*H].
/// Gate ordering: i (input), o (output), f (forget), c (cell).
/// </summary>
public sealed class KittenBiLstm
{
    private readonly float[] _wFwd; // [4*H * I]
    private readonly float[] _rFwd; // [4*H * H]
    private readonly float[] _bFwd; // [4*H] combined (W_bias + R_bias)

    private readonly float[] _wBwd; // [4*H * I]
    private readonly float[] _rBwd; // [4*H * H]
    private readonly float[] _bBwd; // [4*H] combined

    public int HiddenSize { get; }
    public int InputSize { get; }

    public KittenBiLstm(
        SafetensorsFile file,
        string wName,
        string rName,
        string bName,
        int inputSize,
        int hiddenSize)
    {
        HiddenSize = hiddenSize;
        InputSize = inputSize;
        int h4 = 4 * hiddenSize;

        _wFwd = new float[h4 * inputSize];
        _wBwd = new float[h4 * inputSize];
        _rFwd = new float[h4 * hiddenSize];
        _rBwd = new float[h4 * hiddenSize];
        _bFwd = new float[h4];
        _bBwd = new float[h4];

        unsafe
        {
            float* pW = (float*)file.GetTensorPointer(wName);
            float* pR = (float*)file.GetTensorPointer(rName);
            float* pB = (float*)file.GetTensorPointer(bName);

            fixed (float* pWf = _wFwd, pWb = _wBwd)
            {
                Buffer.MemoryCopy(pW, pWf, (long)_wFwd.Length * sizeof(float), (long)_wFwd.Length * sizeof(float));
                Buffer.MemoryCopy(pW + h4 * inputSize, pWb, (long)_wBwd.Length * sizeof(float), (long)_wBwd.Length * sizeof(float));
            }

            fixed (float* pRf = _rFwd, pRb = _rBwd)
            {
                Buffer.MemoryCopy(pR, pRf, (long)_rFwd.Length * sizeof(float), (long)_rFwd.Length * sizeof(float));
                Buffer.MemoryCopy(pR + h4 * hiddenSize, pRb, (long)_rBwd.Length * sizeof(float), (long)_rBwd.Length * sizeof(float));
            }

            // Combine input bias and recurrent bias: B [2, 8*H] -> first 4*H is W_bias, second 4*H is R_bias
            for (int i = 0; i < h4; i++)
            {
                _bFwd[i] = pB[i] + pB[h4 + i];
                _bBwd[i] = pB[8 * hiddenSize + i] + pB[8 * hiddenSize + h4 + i];
            }
        }
    }

    /// <summary>
    /// Executes bidirectional LSTM forward pass.
    /// input: [seqLen, inputSize]
    /// output: [seqLen, 2 * hiddenSize]
    /// </summary>
    public void Forward(ReadOnlySpan<float> input, Span<float> output, int seqLen)
    {
        int h = HiddenSize;
        int h4 = 4 * h;
        int outDim = 2 * h;

        float[] fwdOutputs = new float[seqLen * h];
        float[] bwdOutputs = new float[seqLen * h];

        // 1. Forward direction
        RunDirection(input, _wFwd, _rFwd, _bFwd, fwdOutputs, seqLen, reverse: false);

        // 2. Backward direction
        RunDirection(input, _wBwd, _rBwd, _bBwd, bwdOutputs, seqLen, reverse: true);

        // 3. Concatenate forward and backward outputs [seqLen, 2*H]
        for (int t = 0; t < seqLen; t++)
        {
            var dst = output.Slice(t * outDim, outDim);
            var fwdSrc = new ReadOnlySpan<float>(fwdOutputs, t * h, h);
            var bwdSrc = new ReadOnlySpan<float>(bwdOutputs, t * h, h);

            fwdSrc.CopyTo(dst.Slice(0, h));
            bwdSrc.CopyTo(dst.Slice(h, h));
        }
    }

    private void RunDirection(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> w,
        ReadOnlySpan<float> r,
        ReadOnlySpan<float> b,
        Span<float> output,
        int seqLen,
        bool reverse)
    {
        int h = HiddenSize;
        int h4 = 4 * h;

        Span<float> hidden = stackalloc float[h];
        Span<float> cell = stackalloc float[h];
        Span<float> gates = stackalloc float[h4];
        hidden.Clear();
        cell.Clear();

        for (int step = 0; step < seqLen; step++)
        {
            int t = reverse ? (seqLen - 1 - step) : step;
            var xt = input.Slice(t * InputSize, InputSize);

            // gates = xt * W^T + hidden * R^T + b
            b.CopyTo(gates);
            for (int g = 0; g < h4; g++)
            {
                var wRow = w.Slice(g * InputSize, InputSize);
                var rRow = r.Slice(g * h, h);
                gates[g] += KittenTensorOps.Dot(xt, wRow) + KittenTensorOps.Dot(hidden, rRow);
            }

            // ONNX gate order: i (0..h), o (h..2h), f (2h..3h), c (3h..4h)
            for (int j = 0; j < h; j++)
            {
                float iGate = KittenTensorOps.Sigmoid(gates[j]);
                float oGate = KittenTensorOps.Sigmoid(gates[h + j]);
                float fGate = KittenTensorOps.Sigmoid(gates[2 * h + j]);
                float gGate = MathF.Tanh(gates[3 * h + j]);

                float cNew = fGate * cell[j] + iGate * gGate;
                float hNew = oGate * MathF.Tanh(cNew);

                cell[j] = cNew;
                hidden[j] = hNew;
                output[t * h + j] = hNew;
            }
        }
    }
}
