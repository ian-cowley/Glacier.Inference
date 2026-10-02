namespace Glacier.Inference.Vision;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Quant;

/// <summary>
/// High-performance Multimodal Projector mapping Vision Transformer patch latents
/// to Large Language Model embedding space (e.g., Qwen 2-VL 2D Spatial Merging, LLaVA 2-layer MLP).
/// </summary>
public sealed unsafe class MultimodalProjector : IDisposable
{
    public int VisionDim { get; }
    public int LlmDim { get; }
    public int SpatialMergeFactor { get; } // e.g. 2 for 2x2 patch merging (4x compression)

    private readonly float[] _w1; // [LlmDim, VisionDim * SpatialMergeFactor * SpatialMergeFactor]
    private readonly float[] _w2; // [LlmDim, LlmDim]
    private readonly int _mergedInputDim;

    // Unmanaged scratch memory
    private float* _scratchIntermediate;
    private bool _disposed;

    public MultimodalProjector(int visionDim = 768, int llmDim = 3584, int spatialMergeFactor = 2)
    {
        VisionDim = visionDim;
        LlmDim = llmDim;
        SpatialMergeFactor = spatialMergeFactor;
        _mergedInputDim = VisionDim * SpatialMergeFactor * SpatialMergeFactor;

        // Initialize orthogonal projection weights
        _w1 = new float[LlmDim * _mergedInputDim];
        _w2 = new float[LlmDim * LlmDim];

        float scale1 = MathF.Sqrt(2.0f / (_mergedInputDim + LlmDim));
        for (int i = 0; i < _w1.Length; i++)
        {
            _w1[i] = MathF.Sin((i + 1) * 0.23f) * scale1;
        }

        float scale2 = MathF.Sqrt(2.0f / (LlmDim + LlmDim));
        for (int i = 0; i < _w2.Length; i++)
        {
            _w2[i] = MathF.Cos((i + 1) * 0.19f) * scale2;
        }

        _scratchIntermediate = (float*)NativeMemory.AlignedAlloc((nuint)(LlmDim * sizeof(float)), 64);
    }

    /// <summary>
    /// Performs 2D spatial patch merging ($2\times 2 \to 1$) and MLP projection.
    /// Input: [gridH * gridW, VisionDim].
    /// Output: [(gridH / 2) * (gridW / 2), LlmDim].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int ProjectPatches(
        ReadOnlySpan<float> visionTokens,
        int gridW,
        int gridH,
        Span<float> outputLlmTokens)
    {
        if (gridW % SpatialMergeFactor != 0 || gridH % SpatialMergeFactor != 0)
        {
            throw new ArgumentException($"Grid dimensions ({gridW}x{gridH}) must be divisible by SpatialMergeFactor ({SpatialMergeFactor}).");
        }

        int mergedW = gridW / SpatialMergeFactor;
        int mergedH = gridH / SpatialMergeFactor;
        int totalMergedTokens = mergedW * mergedH;
        int requiredOutputSize = totalMergedTokens * LlmDim;

        if (outputLlmTokens.Length < requiredOutputSize)
        {
            throw new ArgumentException($"Output buffer ({outputLlmTokens.Length}) too small for {requiredOutputSize} floats.");
        }

        fixed (float* pVis = visionTokens)
        fixed (float* pW1 = _w1)
        fixed (float* pW2 = _w2)
        fixed (float* pOut = outputLlmTokens)
        {
            // Allocate temporary merged patch buffer on unmanaged stack/aligned memory
            float* pMerged = (float*)NativeMemory.AlignedAlloc((nuint)(_mergedInputDim * sizeof(float)), 64);

            try
            {
                int outTokenIdx = 0;

                for (int my = 0; my < mergedH; my++)
                {
                    for (int mx = 0; mx < mergedW; mx++)
                    {
                        // 1. Pack 2x2 adjacent spatial patches into a single contiguous vector
                        int packOffset = 0;
                        for (int dy = 0; dy < SpatialMergeFactor; dy++)
                        {
                            int gy = my * SpatialMergeFactor + dy;
                            for (int dx = 0; dx < SpatialMergeFactor; dx++)
                            {
                                int gx = mx * SpatialMergeFactor + dx;
                                int patchIdx = gy * gridW + gx;
                                float* pSrcPatch = pVis + patchIdx * VisionDim;

                                Buffer.MemoryCopy(pSrcPatch, pMerged + packOffset, (ulong)(VisionDim * sizeof(float)), (ulong)(VisionDim * sizeof(float)));
                                packOffset += VisionDim;
                            }
                        }

                        // 2. Layer 1 MLP: intermediate = GeLU(W1 * merged)
                        for (int i = 0; i < LlmDim; i++)
                        {
                            float* wRow = pW1 + i * _mergedInputDim;
                            float sum = 0.0f;

                            int d = 0;
                            if (Vector256.IsHardwareAccelerated)
                            {
                                var vAcc = Vector256<float>.Zero;
                                int vecLimit = _mergedInputDim - 8;
                                for (; d <= vecLimit; d += 8)
                                {
                                    vAcc += Vector256.Load(pMerged + d) * Vector256.Load(wRow + d);
                                }
                                sum = Vector256.Sum(vAcc);
                            }
                            for (; d < _mergedInputDim; d++)
                            {
                                sum += pMerged[d] * wRow[d];
                            }

                            // GeLU activation
                            float gelu = 0.5f * sum * (1.0f + MathF.Tanh(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
                            _scratchIntermediate[i] = gelu;
                        }

                        // 3. Layer 2 MLP: out = W2 * intermediate
                        float* dstToken = pOut + outTokenIdx * LlmDim;
                        for (int i = 0; i < LlmDim; i++)
                        {
                            float* wRow = pW2 + i * LlmDim;
                            float sum = 0.0f;

                            int d = 0;
                            if (Vector256.IsHardwareAccelerated)
                            {
                                var vAcc = Vector256<float>.Zero;
                                int vecLimit = LlmDim - 8;
                                for (; d <= vecLimit; d += 8)
                                {
                                    vAcc += Vector256.Load(_scratchIntermediate + d) * Vector256.Load(wRow + d);
                                }
                                sum = Vector256.Sum(vAcc);
                            }
                            for (; d < LlmDim; d++)
                            {
                                sum += _scratchIntermediate[d] * wRow[d];
                            }

                            dstToken[i] = sum;
                        }

                        outTokenIdx++;
                    }
                }

                return totalMergedTokens;
            }
            finally
            {
                NativeMemory.AlignedFree(pMerged);
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_scratchIntermediate != null)
            {
                NativeMemory.AlignedFree(_scratchIntermediate);
                _scratchIntermediate = null;
            }
        }
    }
}
