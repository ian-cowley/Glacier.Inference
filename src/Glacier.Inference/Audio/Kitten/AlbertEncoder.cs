namespace Glacier.Inference.Audio.Kitten;

using System;
using Glacier.Inference.Format;

/// <summary>
/// Pure C# ALBERT (A Lite BERT) Transformer encoder for KittenTTS text conditioning.
/// 12 shared transformer layers, 12 attention heads, 768 hidden dimension, 128 output dimension.
/// </summary>
public sealed class AlbertEncoder
{
    private readonly float[] _wordEmbeddings; // [178 * 128]
    private readonly float[] _tokenTypeEmbeddings; // [2 * 128]
    private readonly float[] _positionEmbeddings; // [512 * 128]
    private readonly float[] _embedNormWeight; // [128]
    private readonly float[] _embedNormBias; // [128]

    private readonly float[] _mappingWeight; // [768 * 128]
    private readonly float[] _mappingBias; // [768]

    // Shared attention
    private readonly float[] _queryWeight; // [768 * 768]
    private readonly float[] _queryBias;
    private readonly float[] _keyWeight; // [768 * 768]
    private readonly float[] _keyBias;
    private readonly float[] _valueWeight; // [768 * 768]
    private readonly float[] _valueBias;
    private readonly float[] _denseWeight; // [768 * 768]
    private readonly float[] _denseBias;
    private readonly float[] _attnNormWeight; // [768]
    private readonly float[] _attnNormBias;

    // Shared FFN
    private readonly float[] _ffnWeight; // [2048 * 768]
    private readonly float[] _ffnBias;
    private readonly float[] _ffnOutWeight; // [768 * 2048]
    private readonly float[] _ffnOutBias;
    private readonly float[] _fullNormWeight; // [768]
    private readonly float[] _fullNormBias;

    // Final output projection: 768 -> 128
    private readonly float[] _bertEncoderWeight; // [128 * 768]
    private readonly float[] _bertEncoderBias;

    private const int EmbedDim = 128;
    private const int HiddenDim = 768;
    private const int FfnDim = 2048;
    private const int NumHeads = 12;
    private const int HeadDim = 64;
    private const int NumLayers = 12;

    public AlbertEncoder(SafetensorsFile file)
    {
        _wordEmbeddings = LoadArray(file, "bert.embeddings.word_embeddings.weight");
        _tokenTypeEmbeddings = LoadArray(file, "bert.embeddings.token_type_embeddings.weight");
        _positionEmbeddings = LoadArray(file, "bert.embeddings.position_embeddings.weight");
        _embedNormWeight = LoadArray(file, "bert.embeddings.LayerNorm.weight");
        _embedNormBias = LoadArray(file, "bert.embeddings.LayerNorm.bias");

        _mappingWeight = LoadArray(file, "bert.encoder.embedding_hidden_mapping_in.weight");
        _mappingBias = LoadArray(file, "bert.encoder.embedding_hidden_mapping_in.bias");

        string prefix = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
        _queryWeight = LoadArray(file, prefix + "attention.query.weight");
        _queryBias = LoadArray(file, prefix + "attention.query.bias");
        _keyWeight = LoadArray(file, prefix + "attention.key.weight");
        _keyBias = LoadArray(file, prefix + "attention.key.bias");
        _valueWeight = LoadArray(file, prefix + "attention.value.weight");
        _valueBias = LoadArray(file, prefix + "attention.value.bias");
        _denseWeight = LoadArray(file, prefix + "attention.dense.weight");
        _denseBias = LoadArray(file, prefix + "attention.dense.bias");
        _attnNormWeight = LoadArray(file, prefix + "attention.LayerNorm.weight");
        _attnNormBias = LoadArray(file, prefix + "attention.LayerNorm.bias");

        _ffnWeight = LoadArray(file, prefix + "ffn.weight");
        _ffnBias = LoadArray(file, prefix + "ffn.bias");
        _ffnOutWeight = LoadArray(file, prefix + "ffn_output.weight");
        _ffnOutBias = LoadArray(file, prefix + "ffn_output.bias");
        _fullNormWeight = LoadArray(file, prefix + "full_layer_layer_norm.weight");
        _fullNormBias = LoadArray(file, prefix + "full_layer_layer_norm.bias");

        _bertEncoderWeight = LoadArray(file, "bert_encoder.weight");
        _bertEncoderBias = LoadArray(file, "bert_encoder.bias");
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
    /// Forward pass through ALBERT transformer encoder.
    /// Returns output tensor [seqLen, 128].
    /// </summary>
    public float[] Forward(ReadOnlySpan<int> inputIds)
    {
        int seqLen = inputIds.Length;
        if (seqLen == 0) return Array.Empty<float>();

        // 1. Embeddings: word + type (0) + position
        float[] emb = new float[seqLen * EmbedDim];
        for (int i = 0; i < seqLen; i++)
        {
            int token = Math.Clamp(inputIds[i], 0, 177);
            int pos = Math.Clamp(i, 0, 511);

            int embOffset = i * EmbedDim;
            int wordOffset = token * EmbedDim;
            int posOffset = pos * EmbedDim;

            for (int d = 0; d < EmbedDim; d++)
            {
                emb[embOffset + d] = _wordEmbeddings[wordOffset + d] +
                                     _tokenTypeEmbeddings[d] +
                                     _positionEmbeddings[posOffset + d];
            }
        }

        // LayerNorm over embeddings
        KittenTensorOps.LayerNorm(emb, _embedNormWeight, _embedNormBias, emb, seqLen, EmbedDim);

        // 2. Embedding to Hidden Mapping (128 -> 768)
        float[] hidden = new float[seqLen * HiddenDim];
        KittenTensorOps.BatchLinear(emb, _mappingWeight, _mappingBias, hidden, seqLen, EmbedDim, HiddenDim);

        // Scratch buffers for shared transformer layer
        float[] q = new float[seqLen * HiddenDim];
        float[] k = new float[seqLen * HiddenDim];
        float[] v = new float[seqLen * HiddenDim];
        float[] ctx = new float[seqLen * HiddenDim];
        float[] denseOut = new float[seqLen * HiddenDim];
        float[] attnNorm = new float[seqLen * HiddenDim];
        float[] ffnMid = new float[seqLen * FfnDim];
        float[] ffnOut = new float[seqLen * HiddenDim];
        float[] scores = new float[seqLen];

        float scale = 1.0f / MathF.Sqrt(HeadDim); // 1.0 / 8.0 = 0.125f

        // 3. 12 Shared Transformer Layers
        for (int layer = 0; layer < NumLayers; layer++)
        {
            // Project Q, K, V
            KittenTensorOps.BatchLinear(hidden, _queryWeight, _queryBias, q, seqLen, HiddenDim, HiddenDim);
            KittenTensorOps.BatchLinear(hidden, _keyWeight, _keyBias, k, seqLen, HiddenDim, HiddenDim);
            KittenTensorOps.BatchLinear(hidden, _valueWeight, _valueBias, v, seqLen, HiddenDim, HiddenDim);

            // Multi-head self-attention
            for (int h = 0; h < NumHeads; h++)
            {
                int headOffset = h * HeadDim;

                for (int qi = 0; qi < seqLen; qi++)
                {
                    int qIdx = qi * HiddenDim + headOffset;
                    var qSpan = new ReadOnlySpan<float>(q, qIdx, HeadDim);

                    for (int ki = 0; ki < seqLen; ki++)
                    {
                        int kIdx = ki * HiddenDim + headOffset;
                        var kSpan = new ReadOnlySpan<float>(k, kIdx, HeadDim);
                        scores[ki] = KittenTensorOps.Dot(qSpan, kSpan) * scale;
                    }

                    // Softmax across keys for token qi
                    KittenTensorOps.Softmax(scores);

                    // Weighted sum of V
                    int outIdx = qi * HiddenDim + headOffset;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        float sum = 0f;
                        for (int ki = 0; ki < seqLen; ki++)
                        {
                            sum += scores[ki] * v[ki * HiddenDim + headOffset + d];
                        }
                        ctx[outIdx + d] = sum;
                    }
                }
            }

            // Attention dense projection + residual add with hidden
            KittenTensorOps.BatchLinear(ctx, _denseWeight, _denseBias, denseOut, seqLen, HiddenDim, HiddenDim);
            for (int i = 0; i < hidden.Length; i++)
            {
                denseOut[i] += hidden[i];
            }

            // LayerNorm after attention
            KittenTensorOps.LayerNorm(denseOut, _attnNormWeight, _attnNormBias, attnNorm, seqLen, HiddenDim);

            // FFN: 768 -> 2048 -> GELU -> 768
            KittenTensorOps.BatchLinear(attnNorm, _ffnWeight, _ffnBias, ffnMid, seqLen, HiddenDim, FfnDim);
            KittenTensorOps.GeluInPlace(ffnMid);
            KittenTensorOps.BatchLinear(ffnMid, _ffnOutWeight, _ffnOutBias, ffnOut, seqLen, FfnDim, HiddenDim);

            // Residual add with attnNorm
            for (int i = 0; i < hidden.Length; i++)
            {
                ffnOut[i] += attnNorm[i];
            }

            // LayerNorm after FFN -> becomes hidden for next layer
            KittenTensorOps.LayerNorm(ffnOut, _fullNormWeight, _fullNormBias, hidden, seqLen, HiddenDim);
        }

        // 4. Output projection: 768 -> 128
        float[] output = new float[seqLen * EmbedDim];
        KittenTensorOps.BatchLinear(hidden, _bertEncoderWeight, _bertEncoderBias, output, seqLen, HiddenDim, EmbedDim);

        return output;
    }
}
