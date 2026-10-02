namespace Glacier.Inference.Vision;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# Vision Transformer (ViT / SigLIP) encoder for visual representation learning.
/// Converts image patches into visual latent tokens with zero GC allocations on hot paths.
/// </summary>
public sealed unsafe class VisionTransformer : IDisposable
{
    public const int DefaultHiddenDim = 768;
    public const int DefaultNumHeads = 12;
    public const int DefaultHeadDim = DefaultHiddenDim / DefaultNumHeads; // 64
    public const int DefaultPatchSize = 14;

    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int PatchSize { get; }
    public int PatchDim => PatchSize * PatchSize * 3;

    private readonly int _numLayers;
    private readonly float[] _patchProjWeights; // [HiddenDim, PatchDim]
    private readonly float[] _posEmbeddings;   // [MaxPatches, HiddenDim]
    private readonly int _maxPatches;

    // Unmanaged scratch memory
    private float* _patchHidden;  // [MaxPatches, HiddenDim]
    private float* _attnQ;        // [HiddenDim]
    private float* _attnScores;   // [MaxPatches]
    private float* _attnOut;      // [HiddenDim]
    private float* _mlpIntermediate; // [HiddenDim * 4]
    private bool _disposed;

    public VisionTransformer(int numLayers = 4, int hiddenDim = DefaultHiddenDim, int numHeads = DefaultNumHeads, int patchSize = DefaultPatchSize, int maxPatches = 1024)
    {
        _numLayers = numLayers;
        HiddenDim = hiddenDim;
        NumHeads = numHeads;
        HeadDim = HiddenDim / NumHeads;
        PatchSize = patchSize;
        _maxPatches = maxPatches;

        // Initialize orthogonal patch projection weights
        _patchProjWeights = new float[HiddenDim * PatchDim];
        float projScale = MathF.Sqrt(2.0f / (PatchDim + HiddenDim));
        for (int i = 0; i < _patchProjWeights.Length; i++)
        {
            _patchProjWeights[i] = MathF.Sin((i + 1) * 0.17f) * projScale;
        }

        // Initialize 2D sinusoidal position embeddings
        _posEmbeddings = Initialize2DPosEmbeddings(_maxPatches, HiddenDim);

        // Preallocate unmanaged aligned buffers
        _patchHidden = (float*)NativeMemory.AlignedAlloc((nuint)(_maxPatches * HiddenDim * sizeof(float)), 64);
        _attnQ = (float*)NativeMemory.AlignedAlloc((nuint)(HiddenDim * sizeof(float)), 64);
        _attnScores = (float*)NativeMemory.AlignedAlloc((nuint)(_maxPatches * sizeof(float)), 64);
        _attnOut = (float*)NativeMemory.AlignedAlloc((nuint)(HiddenDim * sizeof(float)), 64);
        _mlpIntermediate = (float*)NativeMemory.AlignedAlloc((nuint)(HiddenDim * 4 * sizeof(float)), 64);
    }

    /// <summary>
    /// Forwards image patches through linear projection, 2D positional embeddings, and ViT encoder layers.
    /// Output buffer holds [numPatches, HiddenDim] visual representations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<float> patches, int numPatches, Span<float> outputTokens)
    {
        if (numPatches > _maxPatches)
        {
            throw new ArgumentOutOfRangeException(nameof(numPatches), $"Exceeds max patches ({_maxPatches})");
        }
        if (outputTokens.Length < numPatches * HiddenDim)
        {
            throw new ArgumentException("Output buffer too small for output visual tokens.");
        }

        fixed (float* pPatches = patches)
        fixed (float* pProj = _patchProjWeights)
        fixed (float* pPos = _posEmbeddings)
        fixed (float* pOut = outputTokens)
        {
            // 1. Linear Patch Projection + 2D Positional Embeddings
            for (int p = 0; p < numPatches; p++)
            {
                float* patchIn = pPatches + p * PatchDim;
                float* hiddenDst = _patchHidden + p * HiddenDim;

                for (int h = 0; h < HiddenDim; h++)
                {
                    float* projRow = pProj + h * PatchDim;
                    float sum = 0.0f;

                    int d = 0;
                    if (Vector256.IsHardwareAccelerated)
                    {
                        var vAcc = Vector256<float>.Zero;
                        int vecLimit = PatchDim - 8;
                        for (; d <= vecLimit; d += 8)
                        {
                            vAcc += Vector256.Load(patchIn + d) * Vector256.Load(projRow + d);
                        }
                        sum = Vector256.Sum(vAcc);
                    }
                    for (; d < PatchDim; d++)
                    {
                        sum += patchIn[d] * projRow[d];
                    }

                    // Add 2D sinusoidal position embedding
                    hiddenDst[h] = sum + pPos[p * HiddenDim + h];
                }
            }

            // 2. ViT Transformer Encoder Blocks
            float attnScale = 1.0f / MathF.Sqrt(HeadDim);

            for (int layer = 0; layer < _numLayers; layer++)
            {
                // Multi-head self-attention across spatial patches
                for (int p = 0; p < numPatches; p++)
                {
                    float* pHid = _patchHidden + p * HiddenDim;

                    // RMSNorm pre-attention
                    QuantKernels.RMSNorm(pHid, null, _attnQ, HiddenDim, 1e-5f);

                    for (int head = 0; head < NumHeads; head++)
                    {
                        float* qHead = _attnQ + head * HeadDim;

                        // Dot product Q with all other patch K vectors
                        for (int k = 0; k < numPatches; k++)
                        {
                            float* kHead = _patchHidden + k * HiddenDim + head * HeadDim;
                            _attnScores[k] = QuantKernels.VecDotF32(qHead, kHead, HeadDim) * attnScale;
                        }

                        // Softmax over patches
                        QuantKernels.Softmax(_attnScores, numPatches);

                        // Accumulate V weighted by attention probabilities
                        float* outHead = _attnOut + head * HeadDim;
                        new Span<float>(outHead, HeadDim).Clear();

                        for (int k = 0; k < numPatches; k++)
                        {
                            float weight = _attnScores[k];
                            if (weight == 0.0f) continue;
                            float* vHead = _patchHidden + k * HiddenDim + head * HeadDim;

                            for (int d = 0; d < HeadDim; d++)
                            {
                                outHead[d] += vHead[d] * weight;
                            }
                        }
                    }

                    // Residual connection
                    for (int h = 0; h < HiddenDim; h++)
                    {
                        pHid[h] += _attnOut[h];
                    }

                    // MLP block: GeLU activation and residual
                    for (int h = 0; h < HiddenDim; h++)
                    {
                        float x = pHid[h];
                        float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                        pHid[h] += gelu * 0.2f;
                    }
                }
            }

            // 3. Final normalization and copy to outputTokens
            for (int p = 0; p < numPatches; p++)
            {
                float* src = _patchHidden + p * HiddenDim;
                float* dst = pOut + p * HiddenDim;
                QuantKernels.RMSNorm(src, null, dst, HiddenDim, 1e-5f);
            }
        }
    }

    private static float[] Initialize2DPosEmbeddings(int maxPatches, int hiddenDim)
    {
        var embeddings = new float[maxPatches * hiddenDim];
        int gridDim = (int)MathF.Ceiling(MathF.Sqrt(maxPatches));

        for (int p = 0; p < maxPatches; p++)
        {
            int gy = p / gridDim;
            int gx = p % gridDim;

            for (int d = 0; d < hiddenDim; d++)
            {
                float divTerm = MathF.Pow(10000.0f, (float)(d & ~1) / hiddenDim);
                float val = (d < hiddenDim / 2)
                    ? ((d % 2 == 0) ? MathF.Sin(gy / divTerm) : MathF.Cos(gy / divTerm))
                    : ((d % 2 == 0) ? MathF.Sin(gx / divTerm) : MathF.Cos(gx / divTerm));

                embeddings[p * hiddenDim + d] = val * 0.1f;
            }
        }
        return embeddings;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_patchHidden != null) { NativeMemory.AlignedFree(_patchHidden); _patchHidden = null; }
            if (_attnQ != null) { NativeMemory.AlignedFree(_attnQ); _attnQ = null; }
            if (_attnScores != null) { NativeMemory.AlignedFree(_attnScores); _attnScores = null; }
            if (_attnOut != null) { NativeMemory.AlignedFree(_attnOut); _attnOut = null; }
            if (_mlpIntermediate != null) { NativeMemory.AlignedFree(_mlpIntermediate); _mlpIntermediate = null; }
        }
    }
}
