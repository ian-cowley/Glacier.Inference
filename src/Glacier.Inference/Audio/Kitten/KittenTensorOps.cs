namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

/// <summary>
/// High-performance pure C# SIMD tensor mathematical operations for StyleTTS 2 / KittenTTS.
/// Fully hardware-accelerated via .NET 10 hardware intrinsics (AVX-512 / AVX2 / NEON).
/// </summary>
public static unsafe class KittenTensorOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = a.Length;
        int i = 0;
        float sum = 0f;
        fixed (float* pA = a, pB = b)
        {
            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vSum = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    vSum = Vector512.FusedMultiplyAdd(Vector512.Load(pA + i), Vector512.Load(pB + i), vSum);
                }
                sum += Vector512.Sum(vSum);
            }
            else if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var vSum = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    vSum = Vector256.FusedMultiplyAdd(Vector256.Load(pA + i), Vector256.Load(pB + i), vSum);
                }
                sum += Vector256.Sum(vSum);
            }
            for (; i < length; i++)
            {
                sum += pA[i] * pB[i];
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> a)
    {
        int length = a.Length;
        int i = 0;
        float sum = 0f;
        fixed (float* pA = a)
        {
            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vSum = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    vSum += Vector512.Load(pA + i);
                }
                sum += Vector512.Sum(vSum);
            }
            else if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var vSum = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    vSum += Vector256.Load(pA + i);
                }
                sum += Vector256.Sum(vSum);
            }
            for (; i < length; i++)
            {
                sum += pA[i];
            }
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ReadOnlySpan<float> a)
    {
        if (a.IsEmpty) return 0f;
        float max = a[0];
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] > max) max = a[i];
        }
        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Linear(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inFeatures,
        int outFeatures)
    {
        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pOut = output)
        fixed (float* pB = bias)
        {
            for (int o = 0; o < outFeatures; o++)
            {
                float sum = pB != null ? pB[o] : 0.0f;
                var rowW = new ReadOnlySpan<float>(pW + o * inFeatures, inFeatures);
                sum += Dot(input, rowW);
                pOut[o] = sum;
            }
        }
    }

    /// <summary>
    /// Batched linear projection over sequence tokens: input [seqLen, inFeatures] * W^T [inFeatures, outFeatures] + b.
    /// </summary>
    public static void BatchLinear(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int seqLen,
        int inFeatures,
        int outFeatures)
    {
        for (int s = 0; s < seqLen; s++)
        {
            var inSlice = input.Slice(s * inFeatures, inFeatures);
            var outSlice = output.Slice(s * outFeatures, outFeatures);
            Linear(inSlice, weight, bias, outSlice, inFeatures, outFeatures);
        }
    }

    /// <summary>
    /// Standard 1D Convolution over channels: input [channels, length] -> output [outChannels, outLength].
    /// Weight layout: [outChannels, inChannels, kernelSize].
    /// </summary>
    public static void Conv1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inChannels,
        int outChannels,
        int length,
        int kernelSize,
        int padding = 0,
        int stride = 1,
        int dilation = 1)
    {
        int effKernel = (kernelSize - 1) * dilation + 1;
        int outLength = (length + 2 * padding - effKernel) / stride + 1;
        if (outLength <= 0) return;

        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pB = bias)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* wPtr = pW;
            float* bPtr = pB;
            float* outPtr = pOut;

            Parallel.For(0, outChannels, oc =>
            {
                float baseBias = bPtr != null ? bPtr[oc] : 0.0f;
                int ocOffset = oc * outLength;

                for (int t = 0; t < outLength; t++)
                {
                    float sum = baseBias;
                    int inStart = t * stride - padding;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        int icInOffset = ic * length;
                        int icWOffset = (oc * inChannels + ic) * kernelSize;

                        for (int k = 0; k < kernelSize; k++)
                        {
                            int inIdx = inStart + k * dilation;
                            if (inIdx >= 0 && inIdx < length)
                            {
                                sum += inPtr[icInOffset + inIdx] * wPtr[icWOffset + k];
                            }
                        }
                    }

                    outPtr[ocOffset + t] = sum;
                }
            });
        }
    }

    /// <summary>
    /// 1D Transposed Convolution (upsampling): input [inChannels, length] -> output [outChannels, outLength].
    /// Weight layout: [inChannels, outChannels, kernelSize].
    /// </summary>
    public static void ConvTranspose1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inChannels,
        int outChannels,
        int length,
        int kernelSize,
        int stride,
        int padding = 0,
        int outputPadding = 0)
    {
        int outLength = (length - 1) * stride + kernelSize + outputPadding - 2 * padding;
        if (outLength <= 0) return;

        output.Clear();

        fixed (float* pIn = input)
        fixed (float* pW = weight)
        fixed (float* pB = bias)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* wPtr = pW;
            float* bPtr = pB;
            float* outPtr = pOut;

            Parallel.For(0, outChannels, oc =>
            {
                float bVal = bPtr != null ? bPtr[oc] : 0.0f;
                int ocOutOffset = oc * outLength;
                if (bVal != 0.0f)
                {
                    for (int t = 0; t < outLength; t++)
                    {
                        outPtr[ocOutOffset + t] = bVal;
                    }
                }

                for (int ic = 0; ic < inChannels; ic++)
                {
                    int icInOffset = ic * length;
                    int wOffset = (ic * outChannels + oc) * kernelSize;

                    for (int i = 0; i < length; i++)
                    {
                        float inVal = inPtr[icInOffset + i];
                        if (inVal == 0.0f) continue;
                        int outStart = i * stride;

                        for (int k = 0; k < kernelSize; k++)
                        {
                            int outIdx = outStart + k - padding;
                            if (outIdx >= 0 && outIdx < outLength)
                            {
                                outPtr[ocOutOffset + outIdx] += inVal * wPtr[wOffset + k];
                            }
                        }
                    }
                }
            });
        }
    }

    /// <summary>
    /// Layer Normalization over the last dimension (hiddenSize).
    /// </summary>
    public static void LayerNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        Span<float> output,
        int count,
        int hiddenSize,
        float eps = 1e-5f)
    {
        for (int c = 0; c < count; c++)
        {
            var inSlice = input.Slice(c * hiddenSize, hiddenSize);
            var outSlice = output.Slice(c * hiddenSize, hiddenSize);

            float mean = Sum(inSlice) / hiddenSize;
            float varSum = 0.0f;
            for (int i = 0; i < hiddenSize; i++)
            {
                float diff = inSlice[i] - mean;
                varSum += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt((varSum / hiddenSize) + eps);

            for (int i = 0; i < hiddenSize; i++)
            {
                float g = gamma.Length > 0 ? gamma[i] : 1.0f;
                float b = beta.Length > 0 ? beta[i] : 0.0f;
                outSlice[i] = ((inSlice[i] - mean) * invStd) * g + b;
            }
        }
    }

    /// <summary>
    /// Instance Normalization over time dimension for channels [channels, time].
    /// </summary>
    public static void InstanceNorm(
        ReadOnlySpan<float> input,
        Span<float> output,
        int channels,
        int time,
        float eps = 1e-5f)
    {
        if (time == 0) return;

        fixed (float* pIn = input)
        fixed (float* pOut = output)
        {
            float* inPtr = pIn;
            float* outPtr = pOut;

            Parallel.For(0, channels, c =>
            {
                float* inSlice = inPtr + c * time;
                float* outSlice = outPtr + c * time;

                float sum = 0.0f;
                for (int t = 0; t < time; t++) sum += inSlice[t];
                float mean = sum / time;

                float varSum = 0.0f;
                for (int t = 0; t < time; t++)
                {
                    float diff = inSlice[t] - mean;
                    varSum += diff * diff;
                }
                float invStd = 1.0f / MathF.Sqrt((varSum / time) + eps);

                for (int t = 0; t < time; t++)
                {
                    outSlice[t] = (inSlice[t] - mean) * invStd;
                }
            });
        }
    }

    /// <summary>
    /// Adaptive Instance Normalization (AdaIN):
    /// normed = InstanceNorm(x)
    /// out = normed * (gamma + 1.0) + beta
    /// where gamma, beta come from style projection [2 * channels].
    /// </summary>
    public static void AdaIN(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> styleProjection,
        Span<float> output,
        int channels,
        int time)
    {
        InstanceNorm(input, output, channels, time);

        fixed (float* pProj = styleProjection)
        fixed (float* pOut = output)
        {
            for (int c = 0; c < channels; c++)
            {
                float gamma = pProj[c] + 1.0f;
                float beta = pProj[channels + c];
                int offset = c * time;

                for (int t = 0; t < time; t++)
                {
                    pOut[offset + t] = pOut[offset + t] * gamma + beta;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LeakyRelu(Span<float> span, float alpha = 0.2f)
    {
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            if (v < 0.0f) span[i] = v * alpha;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x)
    {
        return 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    public static void GeluInPlace(Span<float> span)
    {
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = Gelu(span[i]);
        }
    }

    public static void Softmax(Span<float> span)
    {
        float max = Max(span);
        float sum = 0.0f;
        for (int i = 0; i < span.Length; i++)
        {
            float exp = MathF.Exp(span[i] - max);
            span[i] = exp;
            sum += exp;
        }
        float invSum = 1.0f / MathF.Max(sum, 1e-9f);
        for (int i = 0; i < span.Length; i++)
        {
            span[i] *= invSum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }
}
