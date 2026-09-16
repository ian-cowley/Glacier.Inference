namespace Glacier.Inference.Quant;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;

public static unsafe partial class QuantKernels
{
    /// <summary>
    /// Evaluates router gating logits, applies softmax, and selects the top-K active experts.
    /// Renormalizes top-K weights so their sum equals 1.0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RouterTopK(
        float* x,
        float* gateInpWeight,
        float* gateInpBias,
        int dim,
        int expertCount,
        int topK,
        int* selectedIndices,
        float* selectedWeights,
        bool normTopK = true)
    {
        float* logits = stackalloc float[expertCount];

        // 1. Compute router logits for all experts
        for (int e = 0; e < expertCount; e++)
        {
            float* row = gateInpWeight + (long)e * dim;
            float z = VecDotF32(row, x, dim);
            if (gateInpBias != null)
            {
                z += gateInpBias[e];
            }
            logits[e] = z;
        }

        SoftmaxTopK(logits, expertCount, topK, selectedIndices, selectedWeights, normTopK);
    }

    /// <summary>
    /// Computes Softmax across router logits, selects the top-K experts, and optionally renormalizes weights to sum to 1.0.
    /// </summary>
    public static void SoftmaxTopK(
        float* logits,
        int expertCount,
        int topK,
        int* selectedIndices,
        float* selectedWeights,
        bool normTopK = true)
    {
        float maxLogit = float.MinValue;
        for (int e = 0; e < expertCount; e++)
        {
            if (logits[e] > maxLogit) maxLogit = logits[e];
        }

        // 2. Softmax probabilities over all experts
        float* probs = stackalloc float[expertCount];
        float sumExp = 0f;
        for (int e = 0; e < expertCount; e++)
        {
            float ep = MathF.Exp(logits[e] - maxLogit);
            probs[e] = ep;
            sumExp += ep;
        }

        float invSumExp = sumExp > 0f ? 1.0f / sumExp : 1.0f;
        for (int e = 0; e < expertCount; e++)
        {
            probs[e] *= invSumExp;
        }

        // 3. Select top-K experts with largest probabilities
        for (int k = 0; k < topK; k++)
        {
            float bestVal = -1f;
            int bestIdx = -1;
            for (int e = 0; e < expertCount; e++)
            {
                if (probs[e] > bestVal)
                {
                    bestVal = probs[e];
                    bestIdx = e;
                }
            }

            selectedIndices[k] = bestIdx;
            selectedWeights[k] = bestVal;
            if (bestIdx >= 0)
            {
                probs[bestIdx] = -2f; // Mark as visited
            }
        }

        // 4. Renormalize top-K weights if requested (e.g. Qwen2-MoE, Mixtral)
        if (normTopK)
        {
            float sumWeights = 0f;
            for (int k = 0; k < topK; k++)
            {
                sumWeights += selectedWeights[k];
            }

            float invSum = sumWeights > 0f ? 1.0f / sumWeights : 1.0f / topK;
            for (int k = 0; k < topK; k++)
            {
                selectedWeights[k] *= invSum;
            }
        }
    }


    /// <summary>
    /// Root Mean Square Normalization: dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RMSNorm(float* x, float* weight, float* dst, int size, float eps)
    {
        float sumSq = 0f;
        int i = 0;

        if (Vector256.IsHardwareAccelerated)
        {
            int vecLimit = size - 8;
            var acc = Vector256<float>.Zero;
            for (; i <= vecLimit; i += 8)
            {
                var v = Vector256.Load(x + i);
                acc += v * v;
            }
            sumSq = Vector256.Sum(acc);
        }

        for (; i < size; i++)
        {
            float v = x[i];
            sumSq += v * v;
        }

        float rms = 1.0f / MathF.Sqrt((sumSq / size) + eps);

        i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var vRms = Vector256.Create(rms);
            int vecLimit = size - 8;
            for (; i <= vecLimit; i += 8)
            {
                var vx = Vector256.Load(x + i);
                var vw = Vector256.Load(weight + i);
                var vout = vx * vRms * vw;
                vout.Store(dst + i);
            }
        }

        for (; i < size; i++)
        {
            dst[i] = x[i] * rms * weight[i];
        }
    }

    /// <summary>
    /// Applies Rotary Position Embedding (RoPE NeOX style) in-place to Q and K tensors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RoPE(
        float* q, float* k,
        int nHeadsQ, int nHeadsKv,
        int headDim, int pos,
        float freqBase, float freqScale = 1.0f,
        float* ropeFreqs = null)
    {
        int halfDim = headDim / 2;

        Span<float> cosTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];

        for (int i = 0; i < halfDim; i++)
        {
            float baseFreq = 1.0f / MathF.Pow(freqBase, (float)(2 * i) / headDim);
            float freq = ropeFreqs != null ? (baseFreq / ropeFreqs[i]) : baseFreq;
            float theta = pos * freq * freqScale;
            cosTable[i] = MathF.Cos(theta);
            sinTable[i] = MathF.Sin(theta);
        }

        fixed (float* pCos = cosTable, pSin = sinTable)
        {
            // Apply to Q heads
            for (int h = 0; h < nHeadsQ; h++)
            {
                float* head = q + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }

            // Apply to K heads
            for (int h = 0; h < nHeadsKv; h++)
            {
                float* head = k + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float v0 = head[i];
                    float v1 = head[i + halfDim];

                    head[i] = v0 * c - v1 * s;
                    head[i + halfDim] = v0 * s + v1 * c;
                }
            }
        }
    }

    /// <summary>
    /// Applies Rotary Position Embedding (RoPE NeOX style with YaRN support) to Multi-Head Latent Attention (MLA) vectors.
    /// In MLA:
    /// - Q has nHeads of dimension (qkNopeDim + qkRopeDim). RoPE is applied ONLY to the trailing qkRopeDim slice.
    /// - kRope has dimension qkRopeDim (shared positional key vector). RoPE is applied to this vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RoPEMla(
        float* q, float* kRope,
        int nHeads,
        int qHeadDim,
        int qkNopeDim,
        int qkRopeDim,
        int pos,
        float freqBase,
        float* invFreqTable = null,
        float mscale = 1.0f)
    {
        int halfDim = qkRopeDim / 2;

        Span<float> cosTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinTable = halfDim <= 128 ? stackalloc float[halfDim] : new float[halfDim];

        for (int i = 0; i < halfDim; i++)
        {
            float freq = invFreqTable != null ? invFreqTable[i] : 1.0f / MathF.Pow(freqBase, (float)(2 * i) / qkRopeDim);
            float theta = pos * freq;
            cosTable[i] = MathF.Cos(theta) * mscale;
            sinTable[i] = MathF.Sin(theta) * mscale;
        }

        fixed (float* pCos = cosTable, pSin = sinTable)
        {
            Span<float> temp = qkRopeDim <= 128 ? stackalloc float[qkRopeDim] : new float[qkRopeDim];

            // Apply to Q heads (trailing qkRopeDim slice)
            for (int h = 0; h < nHeads; h++)
            {
                float* qPe = q + h * qHeadDim + qkNopeDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float q0 = qPe[2 * i];
                    float q1 = qPe[2 * i + 1];

                    temp[i] = q0 * c - q1 * s;
                    temp[i + halfDim] = q1 * c + q0 * s;
                }
                for (int d = 0; d < qkRopeDim; d++)
                {
                    qPe[d] = temp[d];
                }
            }

            // Apply to single shared K positional vector
            if (kRope != null)
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float c = pCos[i];
                    float s = pSin[i];
                    float k0 = kRope[2 * i];
                    float k1 = kRope[2 * i + 1];

                    temp[i] = k0 * c - k1 * s;
                    temp[i + halfDim] = k1 * c + k0 * s;
                }
                for (int d = 0; d < qkRopeDim; d++)
                {
                    kRope[d] = temp[d];
                }
            }
        }
    }

    /// <summary>
    /// Precomputes YaRN (Yet another RoPE extensioN) inverse frequencies for a given dimension.
    /// </summary>
    public static void PrecomputeYarnFrequencies(
        Span<float> invFreq,
        int dim,
        float freqBase,
        float scalingFactor,
        int originalCtx,
        float betaFast = 32.0f,
        float betaSlow = 1.0f)
    {
        int halfDim = dim / 2;
        float logBase = MathF.Log(freqBase);
        float lowDim = (dim * MathF.Log(originalCtx / (betaFast * 2.0f * MathF.PI))) / (2.0f * logBase);
        float highDim = (dim * MathF.Log(originalCtx / (betaSlow * 2.0f * MathF.PI))) / (2.0f * logBase);

        float low = Math.Max(MathF.Floor(lowDim), 0.0f);
        float high = Math.Min(MathF.Ceiling(highDim), (float)(dim - 1));
        float invRange = (high > low) ? 1.0f / (high - low) : 1.0f;

        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(freqBase, (float)(2 * i) / dim);
            float ramp = Math.Clamp((i - low) * invRange, 0.0f, 1.0f);
            float yarnFreq = (freq / scalingFactor) * ramp + freq * (1.0f - ramp);
            invFreq[i] = yarnFreq;
        }
    }

    /// <summary>
    /// SwiGLU activation function: dst[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SwiGLU(float* gate, float* up, float* dst, int size)
    {
        for (int i = 0; i < size; i++)
        {
            float g = gate[i];
            float silu = g / (1.0f + MathF.Exp(-g));
            dst[i] = silu * up[i];
        }
    }

    /// <summary>
    /// In-place numerically stable Softmax over a vector of logits with optional attention sink logit.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Softmax(float* x, int size, float? sinkLogit = null)
    {
        if (size <= 0) return;

        float maxVal = x[0];
        for (int i = 1; i < size; i++)
        {
            if (x[i] > maxVal) maxVal = x[i];
        }
        if (sinkLogit.HasValue && sinkLogit.Value > maxVal)
        {
            maxVal = sinkLogit.Value;
        }

        float sumExp = sinkLogit.HasValue ? MathF.Exp(sinkLogit.Value - maxVal) : 0f;
        for (int i = 0; i < size; i++)
        {
            float expVal = MathF.Exp(x[i] - maxVal);
            x[i] = expVal;
            sumExp += expVal;
        }

        float invSum = 1.0f / sumExp;
        for (int i = 0; i < size; i++)
        {
            x[i] *= invSum;
        }
    }

}
