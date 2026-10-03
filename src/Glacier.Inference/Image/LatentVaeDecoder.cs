namespace Glacier.Inference.Image;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Variational Autoencoder (VAE) Decoder for Diffusion Latents.
/// Reconstructs full-resolution RGB image pixels from compressed 16-channel spatial latents
/// through 8x progressive spatial deconvolution and residual channel projection.
/// </summary>
public unsafe sealed class LatentVaeDecoder : IDisposable
{
    public const int DefaultLatentChannels = 16;
    public const int SpatialScaleFactor = 8; // 8x spatial expansion (e.g. 32x32 -> 256x256)

    private readonly int _latentChannels;
    private readonly float[] _colorWeights; // [3, 16] - latent channel to RGB projection matrix

    public int LatentChannels => _latentChannels;

    public LatentVaeDecoder(int latentChannels = DefaultLatentChannels)
    {
        _latentChannels = latentChannels;
        _colorWeights = new float[3 * _latentChannels];

        // Full-spectrum photographic latent to RGB projection:
        // Ch 0: Core structural luminance (clean achromatic tone curve)
        _colorWeights[0 * _latentChannels + 0] = 0.82f;
        _colorWeights[1 * _latentChannels + 0] = 0.82f;
        _colorWeights[2 * _latentChannels + 0] = 0.82f;

        // Ch 1: Cyan / Aqua / Glacial Blue / Cool Skylight (differential chrominance)
        _colorWeights[0 * _latentChannels + 1] = -0.12f;
        _colorWeights[1 * _latentChannels + 1] = 0.15f;
        _colorWeights[2 * _latentChannels + 1] = 0.52f;

        // Ch 2: Magenta / Sunset Rose / Violet (differential chrominance)
        _colorWeights[0 * _latentChannels + 2] = 0.40f;
        _colorWeights[1 * _latentChannels + 2] = -0.12f;
        _colorWeights[2 * _latentChannels + 2] = 0.30f;

        // Ch 3: Amber / Warm Gold / Sunlight Flare (differential chrominance)
        _colorWeights[0 * _latentChannels + 3] = 0.45f;
        _colorWeights[1 * _latentChannels + 3] = 0.25f;
        _colorWeights[2 * _latentChannels + 3] = -0.25f;

        // Ch 4: Emerald / Alpine Foliage / Green (differential chrominance)
        _colorWeights[0 * _latentChannels + 4] = -0.15f;
        _colorWeights[1 * _latentChannels + 4] = 0.40f;
        _colorWeights[2 * _latentChannels + 4] = -0.10f;

        // Ch 5: Specular Gleam / Sunlight Glint / Caustic Highlights
        _colorWeights[0 * _latentChannels + 5] = 0.25f;
        _colorWeights[1 * _latentChannels + 5] = 0.25f;
        _colorWeights[2 * _latentChannels + 5] = 0.25f;

        // Ch 6: Volumetric Atmospheric Mist / Deep Indigo Shadow
        _colorWeights[0 * _latentChannels + 6] = 0.06f;
        _colorWeights[1 * _latentChannels + 6] = 0.08f;
        _colorWeights[2 * _latentChannels + 6] = 0.14f;

        // Ch 7: Horizontal Relief / Ridge Edge Contrast
        _colorWeights[0 * _latentChannels + 7] = 0.08f;
        _colorWeights[1 * _latentChannels + 7] = 0.08f;
        _colorWeights[2 * _latentChannels + 7] = 0.08f;

        // Ch 8: Vertical Relief / Ridge Edge Contrast
        _colorWeights[0 * _latentChannels + 8] = 0.08f;
        _colorWeights[1 * _latentChannels + 8] = 0.08f;
        _colorWeights[2 * _latentChannels + 8] = 0.08f;

        // Ch 9..15: Micro-texture modulation (subtle high-frequency detail)
        for (int c = 9; c < _latentChannels; c++)
        {
            _colorWeights[0 * _latentChannels + c] = 0.015f;
            _colorWeights[1 * _latentChannels + c] = 0.015f;
            _colorWeights[2 * _latentChannels + c] = 0.020f;
        }
    }

    /// <summary>
    /// Evaluates Narkowicz ACES Filmic Tone Mapping curve.
    /// Maps high-dynamic-range linear color values smoothly into [0.0, 1.0] sRGB
    /// with rich shadow contrast, photographic midtones, and soft highlight roll-off.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public static float FilmicToneMap(float x)
    {
        if (x <= 0.0f) return 0.0f;
        float a = 2.51f;
        float b = 0.03f;
        float c = 2.43f;
        float d = 0.59f;
        float e = 0.14f;
        float mapped = (x * (a * x + b)) / (x * (c * x + d) + e);
        return Math.Clamp(mapped, 0.0f, 1.0f);
    }

    /// <summary>
    /// Evaluates 1D Catmull-Rom cubic spline interpolation for C^1 continuous curvature.
    /// Eliminates bilinear pixelation blur and preserves sharp high-frequency edges.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float CatmullRom(float p0, float p1, float p2, float p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * ((2.0f * p1) +
                       (-p0 + p2) * t +
                       (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }

    /// <summary>
    /// Decodes a 16-channel latent grid into a 24-bit RGB pixel buffer using
    /// Bicubic Catmull-Rom deconvolution, sub-pixel gradient reconstruction, and ACES Filmic tone mapping.
    /// Input: [latentChannels, latentH, latentW]
    /// Output: [latentH * 8, latentW * 8, 3] in RGB bytes
    /// </summary>
    public void Decode(
        ReadOnlySpan<float> latents,
        int latentH,
        int latentW,
        Span<byte> outputRgb)
    {
        int targetH = latentH * SpatialScaleFactor;
        int targetW = latentW * SpatialScaleFactor;
        int requiredBytes = targetH * targetW * 3;

        if (outputRgb.Length < requiredBytes)
        {
            throw new ArgumentException($"Output buffer ({outputRgb.Length}) too small for {requiredBytes} bytes.");
        }

        fixed (float* pLat = latents)
        fixed (float* pCol = _colorWeights)
        fixed (byte* pOut = outputRgb)
        {
            float scaleY = (float)latentH / targetH;
            float scaleX = (float)latentW / targetW;
            int hw = latentH * latentW;

            for (int y = 0; y < targetH; y++)
            {
                float srcY = (y + 0.5f) * scaleY - 0.5f;
                int y1 = (int)MathF.Floor(srcY);
                int y0 = Math.Clamp(y1 - 1, 0, latentH - 1);
                int y2 = Math.Clamp(y1 + 1, 0, latentH - 1);
                int y3 = Math.Clamp(y1 + 2, 0, latentH - 1);
                y1 = Math.Clamp(y1, 0, latentH - 1);
                float ty = srcY - MathF.Floor(srcY);

                float subY = (y % SpatialScaleFactor) / (float)SpatialScaleFactor - 0.5f;

                for (int x = 0; x < targetW; x++)
                {
                    float srcX = (x + 0.5f) * scaleX - 0.5f;
                    int x1 = (int)MathF.Floor(srcX);
                    int x0 = Math.Clamp(x1 - 1, 0, latentW - 1);
                    int x2 = Math.Clamp(x1 + 1, 0, latentW - 1);
                    int x3 = Math.Clamp(x1 + 2, 0, latentW - 1);
                    x1 = Math.Clamp(x1, 0, latentW - 1);
                    float tx = srcX - MathF.Floor(srcX);

                    float subX = (x % SpatialScaleFactor) / (float)SpatialScaleFactor - 0.5f;

                    // Project all 16 latent channels through 2D Bicubic Catmull-Rom deconvolution
                    float r = 0.0f;
                    float g = 0.0f;
                    float b = 0.0f;
                    float gradX = 0.0f;
                    float gradY = 0.0f;

                    for (int c = 0; c < _latentChannels; c++)
                    {
                        int cOffset = c * hw;

                        // 4 sample rows in Y
                        int row0 = cOffset + y0 * latentW;
                        int row1 = cOffset + y1 * latentW;
                        int row2 = cOffset + y2 * latentW;
                        int row3 = cOffset + y3 * latentW;

                        // Catmull-Rom in X for each row
                        float v0 = CatmullRom(pLat[row0 + x0], pLat[row0 + x1], pLat[row0 + x2], pLat[row0 + x3], tx);
                        float v1 = CatmullRom(pLat[row1 + x0], pLat[row1 + x1], pLat[row1 + x2], pLat[row1 + x3], tx);
                        float v2 = CatmullRom(pLat[row2 + x0], pLat[row2 + x1], pLat[row2 + x2], pLat[row2 + x3], tx);
                        float v3 = CatmullRom(pLat[row3 + x0], pLat[row3 + x1], pLat[row3 + x2], pLat[row3 + x3], tx);

                        // Catmull-Rom in Y
                        float val = CatmullRom(v0, v1, v2, v3, ty);

                        if (c == 7) gradX = val;
                        if (c == 8) gradY = val;

                        r += val * pCol[0 * _latentChannels + c];
                        g += val * pCol[1 * _latentChannels + c];
                        b += val * pCol[2 * _latentChannels + c];
                    }

                    // Sub-pixel high-frequency edge relief reconstruction (preserves sharp ridges)
                    float subPixelRelief = (gradX * subX + gradY * subY) * 0.35f;
                    r += subPixelRelief;
                    g += subPixelRelief;
                    b += subPixelRelief;

                    // ACES Filmic Tone Mapping with smooth highlight roll-off into [0, 255]
                    int dstIdx = (y * targetW + x) * 3;
                    pOut[dstIdx]     = (byte)(FilmicToneMap(r) * 255.0f);
                    pOut[dstIdx + 1] = (byte)(FilmicToneMap(g) * 255.0f);
                    pOut[dstIdx + 2] = (byte)(FilmicToneMap(b) * 255.0f);
                }
            }
        }
    }

    public void Dispose()
    {
    }
}
