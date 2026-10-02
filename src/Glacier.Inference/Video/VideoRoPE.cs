namespace Glacier.Inference.Video;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// 3D Rotary Position Embeddings (3D-RoPE) for multimodal video and spatio-temporal reasoning.
/// Decomposes head dimensions across Temporal (t), Vertical (h), and Horizontal (w) axes.
/// Used by state-of-the-art video-language models (e.g., Qwen 2-VL video mode).
/// </summary>
public sealed class VideoRoPE
{
    private readonly int _headDim;
    private readonly int _temporalDim;
    private readonly int _heightDim;
    private readonly int _widthDim;
    private readonly float _ropeTheta;

    public int HeadDim => _headDim;
    public int TemporalDim => _temporalDim;
    public int HeightDim => _heightDim;
    public int WidthDim => _widthDim;

    public VideoRoPE(int headDim = 64, float ropeTheta = 10000.0f)
    {
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("Head dimension must be even.", nameof(headDim));
        }

        _headDim = headDim;
        _ropeTheta = ropeTheta;

        // Partition head dimension: 25% temporal, 37.5% height, 37.5% width (rounded to even)
        _temporalDim = (headDim / 4) & ~1;
        int remaining = headDim - _temporalDim;
        _heightDim = (remaining / 2) & ~1;
        _widthDim = headDim - _temporalDim - _heightDim;
    }

    /// <summary>
    /// Applies 3D-RoPE in-place to a single head vector given spatio-temporal coordinates (t, h, w).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe void Apply3DRoPE(Span<float> vector, int t, int h, int w)
    {
        if (vector.Length < _headDim)
        {
            throw new ArgumentException($"Vector length ({vector.Length}) must be >= HeadDim ({_headDim}).");
        }

        fixed (float* pVec = vector)
        {
            int offset = 0;

            // 1. Temporal Rotary Subspace
            RotateSubspace(pVec + offset, _temporalDim, t);
            offset += _temporalDim;

            // 2. Spatial Height Rotary Subspace
            RotateSubspace(pVec + offset, _heightDim, h);
            offset += _heightDim;

            // 3. Spatial Width Rotary Subspace
            RotateSubspace(pVec + offset, _widthDim, w);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private unsafe void RotateSubspace(float* pSubspace, int dim, int position)
    {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(_ropeTheta, (float)(2 * i) / dim);
            float angle = position * freq;
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);

            float x0 = pSubspace[i];
            float x1 = pSubspace[i + halfDim];

            pSubspace[i] = x0 * cos - x1 * sin;
            pSubspace[i + halfDim] = x0 * sin + x1 * cos;
        }
    }
}
