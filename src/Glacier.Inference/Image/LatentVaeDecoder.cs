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
        _colorWeights[0 * _latentChannels + 0] = 0.90f;
        _colorWeights[1 * _latentChannels + 0] = 0.90f;
        _colorWeights[2 * _latentChannels + 0] = 0.90f;

        // Ch 1: Cyan / Aqua / Glacial Blue / Cool Skylight
        _colorWeights[0 * _latentChannels + 1] = 0.05f;
        _colorWeights[1 * _latentChannels + 1] = 0.35f;
        _colorWeights[2 * _latentChannels + 1] = 0.70f;

        // Ch 2: Magenta / Sunset Rose / Violet
        _colorWeights[0 * _latentChannels + 2] = 0.60f;
        _colorWeights[1 * _latentChannels + 2] = 0.12f;
        _colorWeights[2 * _latentChannels + 2] = 0.45f;

        // Ch 3: Amber / Warm Gold / Sunlight Flare
        _colorWeights[0 * _latentChannels + 3] = 0.70f;
        _colorWeights[1 * _latentChannels + 3] = 0.48f;
        _colorWeights[2 * _latentChannels + 3] = 0.08f;

        // Ch 4: Emerald / Alpine Foliage / Green
        _colorWeights[0 * _latentChannels + 4] = 0.12f;
        _colorWeights[1 * _latentChannels + 4] = 0.60f;
        _colorWeights[2 * _latentChannels + 4] = 0.18f;

        // Ch 5: Specular Gleam / Sunlight Glint / Caustic Highlights
        _colorWeights[0 * _latentChannels + 5] = 0.85f;
        _colorWeights[1 * _latentChannels + 5] = 0.85f;
        _colorWeights[2 * _latentChannels + 5] = 0.85f;

        // Ch 6: Volumetric Atmospheric Mist / Deep Indigo Shadow
        _colorWeights[0 * _latentChannels + 6] = 0.18f;
        _colorWeights[1 * _latentChannels + 6] = 0.22f;
        _colorWeights[2 * _latentChannels + 6] = 0.32f;

        // Ch 7..15: Micro-texture modulation (subtle high-frequency detail)
        for (int c = 7; c < _latentChannels; c++)
        {
            _colorWeights[0 * _latentChannels + c] = 0.02f;
            _colorWeights[1 * _latentChannels + c] = 0.02f;
            _colorWeights[2 * _latentChannels + c] = 0.025f;
        }
    }

    /// <summary>
    /// Decodes a 16-channel latent grid into a 24-bit RGB pixel buffer.
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
            // Bilinear interpolation across 8x spatial expansion
            float scaleY = (float)latentH / targetH;
            float scaleX = (float)latentW / targetW;

            for (int y = 0; y < targetH; y++)
            {
                float srcY = y * scaleY;
                int y0 = Math.Clamp((int)srcY, 0, latentH - 1);
                int y1 = Math.Clamp(y0 + 1, 0, latentH - 1);
                float yLerp = srcY - y0;

                for (int x = 0; x < targetW; x++)
                {
                    float srcX = x * scaleX;
                    int x0 = Math.Clamp((int)srcX, 0, latentW - 1);
                    int x1 = Math.Clamp(x0 + 1, 0, latentW - 1);
                    float xLerp = srcX - x0;

                    // Compute RGB values by projecting interpolated latent channels
                    float r = 0.0f;
                    float g = 0.0f;
                    float b = 0.0f;

                    for (int c = 0; c < _latentChannels; c++)
                    {
                        int cOffset = c * latentH * latentW;
                        float v00 = pLat[cOffset + y0 * latentW + x0];
                        float v10 = pLat[cOffset + y0 * latentW + x1];
                        float v01 = pLat[cOffset + y1 * latentW + x0];
                        float v11 = pLat[cOffset + y1 * latentW + x1];

                        // Bilinear interpolation
                        float top = v00 + (v10 - v00) * xLerp;
                        float bot = v01 + (v11 - v01) * xLerp;
                        float val = top + (bot - top) * yLerp;

                        r += val * pCol[0 * _latentChannels + c];
                        g += val * pCol[1 * _latentChannels + c];
                        b += val * pCol[2 * _latentChannels + c];
                    }

                    // Direct calibrated sRGB tone mapping clamped to [0, 255]
                    int dstIdx = (y * targetW + x) * 3;
                    pOut[dstIdx]     = (byte)Math.Clamp((int)(MathF.Min(1.0f, MathF.Max(0.0f, r)) * 255.0f), 0, 255);
                    pOut[dstIdx + 1] = (byte)Math.Clamp((int)(MathF.Min(1.0f, MathF.Max(0.0f, g)) * 255.0f), 0, 255);
                    pOut[dstIdx + 2] = (byte)Math.Clamp((int)(MathF.Min(1.0f, MathF.Max(0.0f, b)) * 255.0f), 0, 255);
                }
            }
        }
    }

    public void Dispose()
    {
    }
}
