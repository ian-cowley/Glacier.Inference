namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using Glacier.Inference.Image;
using Glacier.Inference.Image.Flux;

/// <summary>
/// 3D Spatio-Temporal Variational Autoencoder (VAE) Decoder.
/// Reconstructs full-resolution RGB video frame sequences from 4D spatio-temporal latents
/// [T_lat, C, H_lat, W_lat] through Catmull-Rom cubic temporal spline upsampling and
/// 8x progressive spatial deconvolution or pure CUDA neural VAE (ae.safetensors) in pure C# .NET 10.
/// </summary>
public sealed class TemporalLatentVaeDecoder : IDisposable
{
    public const int DefaultLatentChannels = 16;
    public const int SpatialScaleFactor = 8; // 8x spatial expansion (32x32 -> 256x256)

    private readonly LatentVaeDecoder _spatialDecoder;
    private readonly FluxVaeDecoder? _neuralDecoder;
    private readonly int _latentChannels;
    private bool _disposed;

    public int LatentChannels => _latentChannels;
    public bool HasNeuralVae => _neuralDecoder != null;

    public TemporalLatentVaeDecoder(int latentChannels = DefaultLatentChannels, FluxVaeDecoder? neuralDecoder = null)
    {
        _latentChannels = latentChannels;
        _neuralDecoder = neuralDecoder;
        _spatialDecoder = new LatentVaeDecoder(latentChannels);
    }

    /// <summary>
    /// Decodes spatio-temporal latents into a sequence of 24-bit RGB video frames.
    /// Input: latents [temporalLatentFrames, latentChannels, latentH, latentW]
    /// Output: List of frame RGB byte arrays [targetH * targetW * 3] for targetFrames count.
    /// </summary>
    public List<byte[]> DecodeVideo(
        ReadOnlySpan<float> spatioTemporalLatents,
        int temporalLatentFrames,
        int targetFrames,
        int latentH,
        int latentW)
    {
        int targetH = latentH * SpatialScaleFactor;
        int targetW = latentW * SpatialScaleFactor;
        int frameRgbBytes = targetH * targetW * 3;
        int frameLatentFloats = _latentChannels * latentH * latentW;

        if (spatioTemporalLatents.Length < temporalLatentFrames * frameLatentFloats)
        {
            throw new ArgumentException("Latents buffer too small for specified dimensions.");
        }

        var frames = new List<byte[]>(targetFrames);

        // Scratch buffers for temporal spline interpolation
        float[] interpolatedLatent = new float[frameLatentFloats];

        for (int frameIdx = 0; frameIdx < targetFrames; frameIdx++)
        {
            // Normalized time progression across the video [0.0 .. 1.0]
            float u = (targetFrames <= 1) ? 0.0f : (float)frameIdx / (targetFrames - 1);

            // Map u to temporal latent index space [0 .. temporalLatentFrames - 1]
            float tLat = u * (temporalLatentFrames - 1);
            int k1 = (int)MathF.Floor(tLat);
            int k2 = Math.Min(k1 + 1, temporalLatentFrames - 1);
            int k0 = Math.Max(0, k1 - 1);
            int k3 = Math.Min(temporalLatentFrames - 1, k2 + 1);
            float frac = tLat - k1;

            // Catmull-Rom Cubic Spline weights for C^1 continuous velocity
            // p(s) = 0.5 * [ (2*p1) + (-p0 + p2)*s + (2*p0 - 5*p1 + 4*p2 - p3)*s^2 + (-p0 + 3*p1 - 3*p2 + p3)*s^3 ]
            float s = frac;
            float s2 = s * s;
            float s3 = s2 * s;

            float w0 = 0.5f * (-s + 2.0f * s2 - s3);
            float w1 = 0.5f * (2.0f - 5.0f * s2 + 3.0f * s3);
            float w2 = 0.5f * (s + 4.0f * s2 - 3.0f * s3);
            float w3 = 0.5f * (-s2 + s3);

            int off0 = k0 * frameLatentFloats;
            int off1 = k1 * frameLatentFloats;
            int off2 = k2 * frameLatentFloats;
            int off3 = k3 * frameLatentFloats;

            for (int i = 0; i < frameLatentFloats; i++)
            {
                float p0 = spatioTemporalLatents[off0 + i];
                float p1 = spatioTemporalLatents[off1 + i];
                float p2 = spatioTemporalLatents[off2 + i];
                float p3 = spatioTemporalLatents[off3 + i];

                float val = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
                interpolatedLatent[i] = Math.Clamp(val, -1.0f, 2.5f);
            }

            // Spatial progressive decode of interpolated latent frame to RGB
            byte[] frameBytes = new byte[frameRgbBytes];
            if (_neuralDecoder != null)
            {
                _neuralDecoder.Decode(interpolatedLatent, latentH, latentW, frameBytes);
            }
            else
            {
                _spatialDecoder.Decode(interpolatedLatent, latentH, latentW, frameBytes);
            }

            // Optional temporal anti-flicker smoothing against previous frame
            if (frames.Count > 0)
            {
                byte[] prev = frames[^1];
                for (int b = 0; b < frameBytes.Length; b++)
                {
                    // 15% temporal blend with previous frame for cinematic persistence of vision
                    int smoothed = (int)(frameBytes[b] * 0.85f + prev[b] * 0.15f);
                    frameBytes[b] = (byte)Math.Clamp(smoothed, 0, 255);
                }
            }

            frames.Add(frameBytes);
        }

        return frames;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _neuralDecoder?.Dispose();
            _spatialDecoder.Dispose();
        }
    }
}
