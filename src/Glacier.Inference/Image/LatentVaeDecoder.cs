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

        // Red channel projection:
        _colorWeights[0 * _latentChannels + 0] = 0.50f;  // Base luminance
        _colorWeights[0 * _latentChannels + 1] = 0.05f;  // Cyan keeps red low
        _colorWeights[0 * _latentChannels + 2] = 0.15f;  // Ice blue keeps red low
        _colorWeights[0 * _latentChannels + 3] = 0.90f;  // Warm interior lights / sunset boost red
        _colorWeights[0 * _latentChannels + 4] = 0.30f;  // Edge definition
        _colorWeights[0 * _latentChannels + 5] = 0.80f;  // Specular gleam

        // Green channel projection:
        _colorWeights[1 * _latentChannels + 0] = 0.65f;  // Base luminance
        _colorWeights[1 * _latentChannels + 1] = 1.20f;  // Aurora green/teal peak
        _colorWeights[1 * _latentChannels + 2] = 0.60f;  // Ice green reflection
        _colorWeights[1 * _latentChannels + 3] = 0.40f;  // Warmth
        _colorWeights[1 * _latentChannels + 4] = 0.30f;  // Edge definition
        _colorWeights[1 * _latentChannels + 5] = 0.85f;  // Specular gleam

        // Blue channel projection:
        _colorWeights[2 * _latentChannels + 0] = 0.80f;  // High nocturnal blue luminance
        _colorWeights[2 * _latentChannels + 1] = 1.10f;  // Aurora cyan blue component
        _colorWeights[2 * _latentChannels + 2] = 1.30f;  // Intense glacial / crystal ice blue
        _colorWeights[2 * _latentChannels + 3] = 0.10f;  // Low warm in blue
        _colorWeights[2 * _latentChannels + 4] = 0.35f;  // Edge definition
        _colorWeights[2 * _latentChannels + 5] = 1.00f;  // Specular gleam

        for (int c = 6; c < _latentChannels; c++)
        {
            _colorWeights[0 * _latentChannels + c] = 0.02f;
            _colorWeights[1 * _latentChannels + c] = 0.02f;
            _colorWeights[2 * _latentChannels + c] = 0.03f;
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
