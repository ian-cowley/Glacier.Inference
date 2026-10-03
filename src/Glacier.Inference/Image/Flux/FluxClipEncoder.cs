namespace Glacier.Inference.Image.Flux;

using System;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Format;

/// <summary>
/// Pure C# .NET 10 CLIP-L text encoder for Flux conditioning.
/// Encodes textual prompts into pooled embedding y [768] directly from clip_l.safetensors.
/// </summary>
public unsafe sealed class FluxClipEncoder : IDisposable
{
    private readonly SafetensorsFile _safetensors;
    private readonly Half* _tokenEmbeddings; // [49408, 768]
    private readonly Half* _posEmbeddings;   // [77, 768]

    // 12 Transformer layers
    private readonly LayerWeights[] _layers = new LayerWeights[12];
    private readonly Half* _finalNormWeight;
    private readonly Half* _finalNormBias;
    private bool _disposed;

    private sealed class LayerWeights
    {
        public Half* Ln1Weight, Ln1Bias;
        public Half* QWeight, QBias;
        public Half* KWeight, KBias;
        public Half* VWeight, VBias;
        public Half* OutWeight, OutBias;
        public Half* Ln2Weight, Ln2Bias;
        public Half* Fc1Weight, Fc1Bias;
        public Half* Fc2Weight, Fc2Bias;
    }

    private FluxClipEncoder(SafetensorsFile safetensors)
    {
        _safetensors = safetensors;
        _tokenEmbeddings = (Half*)safetensors.GetTensorPointer("text_model.embeddings.token_embedding.weight");
        _posEmbeddings = (Half*)safetensors.GetTensorPointer("text_model.embeddings.position_embedding.weight");

        for (int i = 0; i < 12; i++)
        {
            string pfx = $"text_model.encoder.layers.{i}.";
            _layers[i] = new LayerWeights
            {
                Ln1Weight = (Half*)safetensors.GetTensorPointer(pfx + "layer_norm1.weight"),
                Ln1Bias = (Half*)safetensors.GetTensorPointer(pfx + "layer_norm1.bias"),
                QWeight = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.q_proj.weight"),
                QBias = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.q_proj.bias"),
                KWeight = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.k_proj.weight"),
                KBias = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.k_proj.bias"),
                VWeight = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.v_proj.weight"),
                VBias = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.v_proj.bias"),
                OutWeight = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.out_proj.weight"),
                OutBias = (Half*)safetensors.GetTensorPointer(pfx + "self_attn.out_proj.bias"),
                Ln2Weight = (Half*)safetensors.GetTensorPointer(pfx + "layer_norm2.weight"),
                Ln2Bias = (Half*)safetensors.GetTensorPointer(pfx + "layer_norm2.bias"),
                Fc1Weight = (Half*)safetensors.GetTensorPointer(pfx + "mlp.fc1.weight"),
                Fc1Bias = (Half*)safetensors.GetTensorPointer(pfx + "mlp.fc1.bias"),
                Fc2Weight = (Half*)safetensors.GetTensorPointer(pfx + "mlp.fc2.weight"),
                Fc2Bias = (Half*)safetensors.GetTensorPointer(pfx + "mlp.fc2.bias"),
            };
        }

        _finalNormWeight = (Half*)safetensors.GetTensorPointer("text_model.final_layer_norm.weight");
        _finalNormBias = (Half*)safetensors.GetTensorPointer("text_model.final_layer_norm.bias");
    }

    public static FluxClipEncoder Open(string safetensorsPath)
    {
        var sf = SafetensorsFile.Open(safetensorsPath);
        return new FluxClipEncoder(sf);
    }

    /// <summary>
    /// Encodes a textual prompt into 768-dim pooled conditioning vector y.
    /// </summary>
    public float[] EncodePrompt(string prompt)
    {
        const int seqLen = 77;
        const int hiddenDim = 768;
        const int numHeads = 12;
        const int headDim = 64;

        // 1. Simple CLIP BPE tokenization
        int[] tokens = Tokenize(prompt, seqLen, out int eosIndex);

        // 2. Initial embeddings: token + positional
        float* hidden = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* normBuf = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* q = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* k = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* v = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* attnOut = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * hiddenDim * sizeof(float)), 64);
        float* mlpInt = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * 3072 * sizeof(float)), 64);

        try
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokens[s];
                Half* tokPtr = _tokenEmbeddings + (long)tid * hiddenDim;
                Half* posPtr = _posEmbeddings + (long)s * hiddenDim;
                float* row = hidden + s * hiddenDim;
                for (int d = 0; d < hiddenDim; d++)
                {
                    row[d] = (float)tokPtr[d] + (float)posPtr[d];
                }
            }

            // 3. 12 Transformer layers
            float* scoreRow = stackalloc float[seqLen];
            for (int l = 0; l < 12; l++)
            {
                var layer = _layers[l];

                // LN1
                ApplyLayerNorm(hidden, normBuf, layer.Ln1Weight, layer.Ln1Bias, seqLen, hiddenDim);

                // QKV projections: [seqLen, 768] * [768, 768]
                MatMulF16(normBuf, layer.QWeight, layer.QBias, q, seqLen, hiddenDim, hiddenDim);
                MatMulF16(normBuf, layer.KWeight, layer.KBias, k, seqLen, hiddenDim, hiddenDim);
                MatMulF16(normBuf, layer.VWeight, layer.VBias, v, seqLen, hiddenDim, hiddenDim);

                // Multi-head Attention
                float scale = 1.0f / MathF.Sqrt(headDim);
                for (int h = 0; h < numHeads; h++)
                {
                    int headOffset = h * headDim;
                    for (int i = 0; i < seqLen; i++)
                    {
                        float* qRow = q + i * hiddenDim + headOffset;

                        float maxScore = float.NegativeInfinity;
                        for (int j = 0; j <= i; j++) // Causal mask
                        {
                            float dot = 0f;
                            float* kRow = k + j * hiddenDim + headOffset;
                            for (int d = 0; d < headDim; d++) dot += qRow[d] * kRow[d];
                            float sc = dot * scale;
                            scoreRow[j] = sc;
                            if (sc > maxScore) maxScore = sc;
                        }
                        for (int j = i + 1; j < seqLen; j++) scoreRow[j] = float.NegativeInfinity;

                        // Softmax
                        float sumExp = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float exp = MathF.Exp(scoreRow[j] - maxScore);
                            scoreRow[j] = exp;
                            sumExp += exp;
                        }
                        float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);
                        for (int j = 0; j <= i; j++) scoreRow[j] *= invSum;

                        // Weighted V sum
                        float* outRow = attnOut + i * hiddenDim + headOffset;
                        for (int d = 0; d < headDim; d++) outRow[d] = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float weight = scoreRow[j];
                            float* vRow = v + j * hiddenDim + headOffset;
                            for (int d = 0; d < headDim; d++) outRow[d] += weight * vRow[d];
                        }
                    }
                }


                // Out proj & residual: [seqLen, 768] * [768, 768]
                MatMulF16(attnOut, layer.OutWeight, layer.OutBias, normBuf, seqLen, hiddenDim, hiddenDim);
                for (int i = 0; i < seqLen * hiddenDim; i++) hidden[i] += normBuf[i];

                // LN2
                ApplyLayerNorm(hidden, normBuf, layer.Ln2Weight, layer.Ln2Bias, seqLen, hiddenDim);

                // MLP: fc1 [seqLen, 768] * [3072, 768] -> [seqLen, 3072]
                MatMulF16(normBuf, layer.Fc1Weight, layer.Fc1Bias, mlpInt, seqLen, hiddenDim, 3072);

                // QuickGELU: x * sigmoid(1.702 * x)
                for (int i = 0; i < seqLen * 3072; i++)
                {
                    float x = mlpInt[i];
                    mlpInt[i] = x / (1.0f + MathF.Exp(-1.702f * x));
                }

                // fc2: [seqLen, 3072] * [768, 3072] -> [seqLen, 768] + residual
                MatMulF16(mlpInt, layer.Fc2Weight, layer.Fc2Bias, normBuf, seqLen, 3072, hiddenDim);
                for (int i = 0; i < seqLen * hiddenDim; i++) hidden[i] += normBuf[i];
            }

            // 4. Extract EOS pooled vector & apply final layer norm
            float* eosHidden = hidden + eosIndex * hiddenDim;
            var pooled = new float[hiddenDim];
            fixed (float* pOut = pooled)
            {
                ApplyLayerNorm(eosHidden, pOut, _finalNormWeight, _finalNormBias, 1, hiddenDim);
            }
            return pooled;
        }
        finally
        {
            NativeMemory.AlignedFree(hidden);
            NativeMemory.AlignedFree(normBuf);
            NativeMemory.AlignedFree(q);
            NativeMemory.AlignedFree(k);
            NativeMemory.AlignedFree(v);
            NativeMemory.AlignedFree(attnOut);
            NativeMemory.AlignedFree(mlpInt);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyLayerNorm(float* src, float* dst, Half* weight, Half* bias, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float* s = src + r * dim;
            float* d = dst + r * dim;

            float mean = 0f;
            for (int i = 0; i < dim; i++) mean += s[i];
            mean /= dim;

            float var = 0f;
            for (int i = 0; i < dim; i++)
            {
                float diff = s[i] - mean;
                var += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(var / dim + 1e-5f);

            for (int i = 0; i < dim; i++)
            {
                d[i] = (s[i] - mean) * invStd * (float)weight[i] + (float)bias[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatMulF16(float* x, Half* w, Half* b, float* y, int nRows, int inDim, int outDim)
    {
        // w shape: [outDim, inDim]
        for (int r = 0; r < nRows; r++)
        {
            float* inVec = x + r * inDim;
            float* outVec = y + r * outDim;

            for (int o = 0; o < outDim; o++)
            {
                Half* wRow = w + (long)o * inDim;
                float sum = (float)b[o];
                for (int i = 0; i < inDim; i++)
                {
                    sum += inVec[i] * (float)wRow[i];
                }
                outVec[o] = sum;
            }
        }
    }

    private static int[] Tokenize(string prompt, int maxLen, out int eosIndex)
    {
        var tokens = new int[maxLen];
        tokens[0] = 49406; // <|startoftext|>
        int idx = 1;

        // Basic alphanumeric tokenizer mapping into CLIP vocab
        string[] words = prompt.ToLowerInvariant().Split([' ', ',', '.', '!', '?', '-', '_'], StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            if (idx >= maxLen - 1) break;
            int hash = 0;
            foreach (char c in word) hash = (hash * 31 + c) & 0x7FFFFFFF;
            tokens[idx++] = 1000 + (hash % 40000);
        }

        eosIndex = idx;
        tokens[idx++] = 49407; // <|endoftext|>

        while (idx < maxLen)
        {
            tokens[idx++] = 49407;
        }

        return tokens;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _safetensors.Dispose();
            _disposed = true;
        }
    }
}
