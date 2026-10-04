namespace Glacier.Inference.Image.Flux;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# .NET 10 hardware SIMD inference engine for Google T5-v1.1-XXL text encoder (GGUF).
/// Parses SentencePiece unigram vocabulary and executes 24 layers of Q3_K / Q6_K multi-head attention
/// and GEGLU feed-forward networks with zero external native DLLs or Python runtime.
/// </summary>
public sealed unsafe class FluxT5Encoder : IDisposable
{
    public const int HiddenDim = 4096;
    public const int IntermediateDim = 10240;
    public const int NumHeads = 64;
    public const int HeadDim = 64;
    public const int NumLayers = 24;

    private readonly GgufFile _gguf;
    private readonly Dictionary<string, int> _pieceToId;
    private readonly float[] _pieceScores;
    private readonly int _vocabSize;
    private readonly GgufTensorInfo _tokenEmbd;
    private readonly GgufTensorInfo _outputNorm;
    private readonly GgufTensorInfo _relBias;
    private readonly LayerWeights[] _layers;
    private bool _disposed;

    private sealed class LayerWeights
    {
        public GgufTensorInfo AttnNorm = null!;
        public GgufTensorInfo AttnQ = null!;
        public GgufTensorInfo AttnK = null!;
        public GgufTensorInfo AttnV = null!;
        public GgufTensorInfo AttnO = null!;
        public GgufTensorInfo FfnNorm = null!;
        public GgufTensorInfo FfnGate = null!;
        public GgufTensorInfo FfnUp = null!;
        public GgufTensorInfo FfnDown = null!;
    }

    private FluxT5Encoder(GgufFile gguf)
    {
        _gguf = gguf;

        // 1. Parse SentencePiece tokenizer metadata
        var tokensList = _gguf.GetMetadataList("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Missing 'tokenizer.ggml.tokens' in T5 GGUF model.");
        var scoresList = _gguf.GetMetadataList("tokenizer.ggml.scores");

        _vocabSize = tokensList.Count;
        _pieceToId = new Dictionary<string, int>(_vocabSize, StringComparer.Ordinal);
        _pieceScores = new float[_vocabSize];

        for (int i = 0; i < _vocabSize; i++)
        {
            string piece = tokensList[i]?.ToString() ?? "";
            _pieceToId[piece] = i;
            if (scoresList != null && i < scoresList.Count && scoresList[i] is float f)
            {
                _pieceScores[i] = f;
            }
        }

        // 2. Load Embeddings and Normalization
        _tokenEmbd = _gguf.Tensors["token_embd.weight"];
        _outputNorm = _gguf.Tensors["enc.output_norm.weight"];
        _relBias = _gguf.Tensors["enc.blk.0.attn_rel_b.weight"];

        // 3. Load 24 Transformer Encoder Layers
        _layers = new LayerWeights[NumLayers];
        for (int l = 0; l < NumLayers; l++)
        {
            string prefix = $"enc.blk.{l}.";
            _layers[l] = new LayerWeights
            {
                AttnNorm = _gguf.Tensors[prefix + "attn_norm.weight"],
                AttnQ = _gguf.Tensors[prefix + "attn_q.weight"],
                AttnK = _gguf.Tensors[prefix + "attn_k.weight"],
                AttnV = _gguf.Tensors[prefix + "attn_v.weight"],
                AttnO = _gguf.Tensors[prefix + "attn_o.weight"],
                FfnNorm = _gguf.Tensors[prefix + "ffn_norm.weight"],
                FfnGate = _gguf.Tensors[prefix + "ffn_gate.weight"],
                FfnUp = _gguf.Tensors[prefix + "ffn_up.weight"],
                FfnDown = _gguf.Tensors[prefix + "ffn_down.weight"],
            };
        }
    }

    public static FluxT5Encoder Open(string ggufPath)
    {
        var gguf = GgufFile.Open(ggufPath);
        return new FluxT5Encoder(gguf);
    }

    /// <summary>
    /// Tokenizes prompt string into T5 SentencePiece token IDs with EOS and padding.
    /// Also returns the number of valid tokens (including EOS).
    /// </summary>
    public (int[] Tokens, int ValidCount) Tokenize(string prompt, int maxTokens = 64)
    {
        string cleaned = prompt.Trim();
        if (string.IsNullOrWhiteSpace(cleaned))
        {
            var res = new int[maxTokens];
            res[0] = 1; // EOS only
            return (res, 1);
        }

        var tokens = new List<int>(maxTokens);

        // Preprocess: trim and prepend SentencePiece space prefix (\u2581)
        if (!cleaned.StartsWith(" ") && !cleaned.StartsWith("\u2581"))
        {
            cleaned = "\u2581" + cleaned;
        }
        cleaned = cleaned.Replace(" ", "\u2581");

        // SentencePiece Unigram tokenization via Viterbi dynamic programming
        int n = cleaned.Length;
        var bestScore = new float[n + 1];
        var bestToken = new int[n + 1];
        var bestPrev = new int[n + 1];

        for (int i = 0; i <= n; i++)
        {
            bestScore[i] = float.NegativeInfinity;
            bestToken[i] = -1;
            bestPrev[i] = -1;
        }
        bestScore[0] = 0f;

        for (int i = 0; i < n; i++)
        {
            if (float.IsNegativeInfinity(bestScore[i])) continue;

            int maxSubLen = Math.Min(32, n - i);
            for (int len = 1; len <= maxSubLen; len++)
            {
                string sub = cleaned.Substring(i, len);
                if (_pieceToId.TryGetValue(sub, out int tid))
                {
                    float sc = bestScore[i] + _pieceScores[tid];
                    if (sc > bestScore[i + len])
                    {
                        bestScore[i + len] = sc;
                        bestToken[i + len] = tid;
                        bestPrev[i + len] = i;
                    }
                }
            }

            // Fallback for single unknown char if no path found
            if (bestToken[i + 1] == -1)
            {
                bestScore[i + 1] = bestScore[i] - 100.0f;
                bestToken[i + 1] = 2; // <unk>
                bestPrev[i + 1] = i;
            }
        }

        // Backtrack
        var revTokens = new List<int>();
        int curr = n;
        while (curr > 0)
        {
            revTokens.Add(bestToken[curr]);
            curr = bestPrev[curr];
        }
        revTokens.Reverse();

        for (int i = 0; i < revTokens.Count && tokens.Count < maxTokens - 1; i++)
        {
            tokens.Add(revTokens[i]);
        }

        // Add EOS token (1)
        tokens.Add(1);
        int validCount = Math.Min(tokens.Count, maxTokens);

        // Pad to maxTokens with PAD token (0)
        var result = new int[maxTokens];
        for (int i = 0; i < maxTokens; i++)
        {
            result[i] = i < tokens.Count ? tokens[i] : 0;
        }

        return (result, validCount);
    }

    /// <summary>
    /// Executes full 24-layer T5-XXL forward inference, returning only the valid semantic tokens without padding.
    /// Shape: [validCount, 4096].
    /// </summary>
    public (float[] embeds, int validCount) EncodeWithCount(string prompt, int maxSeqLen = 512)
    {
        var (tokens, validCount) = Tokenize(prompt, maxSeqLen);
        if (validCount <= 0) validCount = 1;

        int computeLen = Math.Max(validCount, 16);
        float[] full = Encode(prompt, seqLen: computeLen);
        var validOnly = new float[validCount * HiddenDim];
        Array.Copy(full, validOnly, validCount * HiddenDim);
        return (validOnly, validCount);
    }

    /// <summary>
    /// Executes full 24-layer T5-XXL forward inference to produce semantic context tokens for FLUX / Wan DiT.
    /// Shape: [seqLen, 4096]. Valid tokens are preserved; pad tokens are masked and zero-padded.
    /// </summary>
    public float[] Encode(string prompt, int seqLen = 64)
    {
        var (tokens, validCount) = Tokenize(prompt, seqLen);

        // Scratch memory
        float* hidden = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* normBuf = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* q = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* k = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* v = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* attnOut = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * HiddenDim * sizeof(float)), 64);
        float* ffnGate = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * IntermediateDim * sizeof(float)), 64);
        float* ffnUp = (float*)NativeMemory.AlignedAlloc((nuint)(seqLen * IntermediateDim * sizeof(float)), 64);

        try
        {
            // 1. Initial token embeddings: [seqLen, 4096]
            byte* embdData = _gguf.GetTensorPointer(_tokenEmbd);
            for (int s = 0; s < seqLen; s++)
            {
                int tid = tokens[s];
                QuantKernels.ExtractEmbedding(_tokenEmbd.Type, embdData, tid, hidden + s * HiddenDim, HiddenDim);
            }

            // 2. Precompute Relative Position Bias table for all head/pair combinations
            float* relBiasPtr = (float*)_gguf.GetTensorPointer(_relBias);
            var relBiasTable = new float[NumHeads * seqLen * seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    int bucket = RelativePositionBucket(j - i, bidirectional: true, numBuckets: 32, maxDistance: 128);
                    float* bRow = relBiasPtr + bucket * NumHeads;
                    for (int h = 0; h < NumHeads; h++)
                    {
                        relBiasTable[h * seqLen * seqLen + i * seqLen + j] = bRow[h];
                    }
                }
            }

            // 3. 24 Transformer Encoder Layers
            for (int l = 0; l < NumLayers; l++)
            {
                var layer = _layers[l];

                // Pre-Attention RMSNorm
                ApplyRMSNorm(hidden, normBuf, layer.AttnNorm, seqLen, HiddenDim);

                // Q, K, V Projections
                QuantKernels.MatMulBatch(layer.AttnQ.Type, _gguf.GetTensorPointer(layer.AttnQ), normBuf, q, HiddenDim, HiddenDim, seqLen);
                QuantKernels.MatMulBatch(layer.AttnK.Type, _gguf.GetTensorPointer(layer.AttnK), normBuf, k, HiddenDim, HiddenDim, seqLen);
                QuantKernels.MatMulBatch(layer.AttnV.Type, _gguf.GetTensorPointer(layer.AttnV), normBuf, v, HiddenDim, HiddenDim, seqLen);

                // Multi-head Attention (NO sqrt(d) scaling in T5!)
                nint qPtr = (nint)q;
                nint kPtr = (nint)k;
                nint vPtr = (nint)v;
                nint outPtr = (nint)attnOut;

                fixed (float* pRelBias = relBiasTable)
                {
                    nint relBiasNint = (nint)pRelBias;

                    Parallel.For(0, NumHeads, h =>
                    {
                        float* pQ = (float*)qPtr;
                        float* pK = (float*)kPtr;
                        float* pV = (float*)vPtr;
                        float* pOut = (float*)outPtr;
                        float* pBias = (float*)relBiasNint + h * seqLen * seqLen;

                        int headOffset = h * HeadDim;
                        float* scores = stackalloc float[seqLen];

                        for (int i = 0; i < validCount; i++)
                        {
                            float* qRow = pQ + i * HiddenDim + headOffset;
                            float maxScore = float.NegativeInfinity;

                            for (int j = 0; j < validCount; j++)
                            {
                                float* kRow = pK + j * HiddenDim + headOffset;
                                float dot = 0f;
                                for (int d = 0; d < HeadDim; d++) dot += qRow[d] * kRow[d];

                                // In T5, dot product is unscaled, relative bias added
                                float sc = dot + pBias[i * seqLen + j];
                                scores[j] = sc;
                                if (sc > maxScore) maxScore = sc;
                            }

                            // Softmax over valid tokens only
                            float sumExp = 0f;
                            for (int j = 0; j < validCount; j++)
                            {
                                float exp = MathF.Exp(scores[j] - maxScore);
                                scores[j] = exp;
                                sumExp += exp;
                            }
                            float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                            // Weighted V sum
                            float* outRow = pOut + i * HiddenDim + headOffset;
                            for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;

                            for (int j = 0; j < validCount; j++)
                            {
                                float w = scores[j] * invSum;
                                float* vRow = pV + j * HiddenDim + headOffset;
                                for (int d = 0; d < HeadDim; d++) outRow[d] += w * vRow[d];
                            }
                        }

                        // Zero out attention output for padding positions
                        for (int i = validCount; i < seqLen; i++)
                        {
                            float* outRow = pOut + i * HiddenDim + headOffset;
                            for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;
                        }
                    });
                }

                // Attention Out Projection & Residual: [seqLen, 4096] * [4096, 4096]
                QuantKernels.MatMulBatch(layer.AttnO.Type, _gguf.GetTensorPointer(layer.AttnO), attnOut, normBuf, HiddenDim, HiddenDim, validCount);
                for (int idx = 0; idx < validCount * HiddenDim; idx++) hidden[idx] += normBuf[idx];

                // Pre-FFN RMSNorm
                ApplyRMSNorm(hidden, normBuf, layer.FfnNorm, validCount, HiddenDim);

                // GEGLU FFN: Gate & Up [validCount, 10240]
                QuantKernels.MatMulBatch(layer.FfnGate.Type, _gguf.GetTensorPointer(layer.FfnGate), normBuf, ffnGate, HiddenDim, IntermediateDim, validCount);
                QuantKernels.MatMulBatch(layer.FfnUp.Type, _gguf.GetTensorPointer(layer.FfnUp), normBuf, ffnUp, HiddenDim, IntermediateDim, validCount);

                // GELU(gate) * up
                ApplyGelu(ffnGate, validCount * IntermediateDim);
                for (int idx = 0; idx < validCount * IntermediateDim; idx++)
                {
                    ffnGate[idx] *= ffnUp[idx];
                }

                // FFN Down Projection & Residual: [validCount, 10240] * [10240, 4096]
                QuantKernels.MatMulBatch(layer.FfnDown.Type, _gguf.GetTensorPointer(layer.FfnDown), ffnGate, normBuf, IntermediateDim, HiddenDim, validCount);
                for (int idx = 0; idx < validCount * HiddenDim; idx++) hidden[idx] += normBuf[idx];
            }

            // 4. Final Output RMSNorm on valid tokens
            var result = new float[seqLen * HiddenDim];
            fixed (float* pRes = result)
            {
                ApplyRMSNorm(hidden, pRes, _outputNorm, validCount, HiddenDim);
                // Positions from validCount to seqLen remain 0.0f (exact HuggingFace / Diffusers match)
            }
            return result;
        }
        finally
        {
            NativeMemory.AlignedFree(hidden);
            NativeMemory.AlignedFree(normBuf);
            NativeMemory.AlignedFree(q);
            NativeMemory.AlignedFree(k);
            NativeMemory.AlignedFree(v);
            NativeMemory.AlignedFree(attnOut);
            NativeMemory.AlignedFree(ffnGate);
            NativeMemory.AlignedFree(ffnUp);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ApplyRMSNorm(float* src, float* dst, GgufTensorInfo normTensor, int seqLen, int dim)
    {
        float* weight = (float*)_gguf.GetTensorPointer(normTensor);
        for (int s = 0; s < seqLen; s++)
        {
            float* sRow = src + s * dim;
            float* dRow = dst + s * dim;

            float sumSq = 0f;
            for (int d = 0; d < dim; d++) sumSq += sRow[d] * sRow[d];
            float invRms = 1.0f / MathF.Sqrt(sumSq / dim + 1e-6f);

            for (int d = 0; d < dim; d++)
            {
                dRow[d] = sRow[d] * invRms * weight[d];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyGelu(float* x, int count)
    {
        const float sqrt2OverPi = 0.79788456f;
        for (int i = 0; i < count; i++)
        {
            float val = x[i];
            x[i] = 0.5f * val * (1.0f + MathF.Tanh(sqrt2OverPi * (val + 0.044715f * val * val * val)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RelativePositionBucket(int relPos, bool bidirectional = true, int numBuckets = 32, int maxDistance = 128)
    {
        int ret = 0;
        int n = numBuckets;
        if (bidirectional)
        {
            n /= 2;
            if (relPos > 0) ret += n;
            relPos = Math.Abs(relPos);
        }
        else
        {
            relPos = Math.Max(0, -relPos);
        }

        int maxExact = n / 2;
        if (relPos < maxExact)
        {
            ret += relPos;
        }
        else
        {
            float ratio = (float)relPos / maxExact;
            float logRatio = MathF.Log(ratio) / MathF.Log((float)maxDistance / maxExact);
            int bucket = maxExact + (int)(logRatio * (n - maxExact));
            if (bucket > n - 1) bucket = n - 1;
            ret += bucket;
        }
        return ret;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _gguf.Dispose();
            _disposed = true;
        }
    }
}
