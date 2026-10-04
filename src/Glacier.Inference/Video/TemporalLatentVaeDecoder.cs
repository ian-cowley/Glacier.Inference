namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using Glacier.Inference.Image;
using Glacier.Inference.Image.Flux;
using Glacier.Inference.Video.Wan;

/// <summary>
/// 3D Spatio-Temporal Variational Autoencoder (VAE) Decoder.
/// Reconstructs full-resolution RGB video frame sequences from 4D spatio-temporal latents
/// [T_lat, C, H_lat, W_lat] through official 3D Causal Autoencoder (wan_2.1_vae.safetensors),
/// pure CUDA neural VAE (ae.safetensors), or Catmull-Rom cubic temporal spline upsampling in pure C# .NET 10.
/// </summary>
public sealed class TemporalLatentVaeDecoder : IDisposable
{
    public const int DefaultLatentChannels = 16;
    public const int SpatialScaleFactor = 8; // 8x spatial expansion (30x30 -> 240x240)

    private readonly LatentVaeDecoder _spatialDecoder;
    private readonly FluxVaeDecoder? _neuralDecoder;
    private readonly Wan3DVaeDecoder? _wanDecoder;
    private readonly int _latentChannels;
    private bool _disposed;

    public int LatentChannels => _latentChannels;
    public bool HasNeuralVae => _wanDecoder != null || _neuralDecoder != null;
    public Wan3DVaeDecoder? WanDecoder => _wanDecoder;

    public TemporalLatentVaeDecoder(
        int latentChannels = DefaultLatentChannels,
        FluxVaeDecoder? neuralDecoder = null,
        Wan3DVaeDecoder? wanDecoder = null)
    {
        _latentChannels = latentChannels;
        _neuralDecoder = neuralDecoder;
        _wanDecoder = wanDecoder;
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

        if (_wanDecoder != null)
        {
            var nativeFrames = _wanDecoder.DecodeVideo(spatioTemporalLatents, temporalLatentFrames, targetFrames, latentH, latentW);
            if (nativeFrames.Count < targetFrames && nativeFrames.Count > 1)
            {
                return InterpolateFrames(nativeFrames, targetFrames, targetH, targetW);
            }
            return nativeFrames;
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
                interpolatedLatent[i] = (_neuralDecoder != null) ? val : Math.Clamp(val, 0.0f, 1.0f);
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

            frames.Add(frameBytes);
        }

        return frames;
    }

    /// <summary>
    /// Smoothly interpolates native RGB keyframe sequence to the target frame count using C^1 Catmull-Rom cubic splines.
    /// </summary>
    public static List<byte[]> InterpolateFrames(IReadOnlyList<byte[]> keyframes, int targetFrames, int height, int width)
    {
        int nativeCount = keyframes.Count;
        if (targetFrames <= nativeCount) return new List<byte[]>(keyframes);

        int frameBytes = height * width * 3;
        var result = new List<byte[]>(targetFrames);

        for (int i = 0; i < targetFrames; i++)
        {
            if (i == 0)
            {
                byte[] exact0 = new byte[frameBytes];
                Array.Copy(keyframes[0], exact0, frameBytes);
                result.Add(exact0);
                continue;
            }
            if (i == targetFrames - 1)
            {
                byte[] exactLast = new byte[frameBytes];
                Array.Copy(keyframes[nativeCount - 1], exactLast, frameBytes);
                result.Add(exactLast);
                continue;
            }

            float u = (float)i / (targetFrames - 1);
            float s = u * (nativeCount - 1);
            int k1 = (int)MathF.Floor(s);
            int k2 = Math.Min(k1 + 1, nativeCount - 1);
            int k0 = Math.Max(0, k1 - 1);
            int k3 = Math.Min(nativeCount - 1, k2 + 1);
            float frac = s - k1;

            if (frac == 0f)
            {
                byte[] exact = new byte[frameBytes];
                Array.Copy(keyframes[k1], exact, frameBytes);
                result.Add(exact);
                continue;
            }

            float f = frac;
            float f2 = f * f;
            float f3 = f2 * f;

            float w0 = 0.5f * (-f + 2.0f * f2 - f3);
            float w1 = 0.5f * (2.0f - 5.0f * f2 + 3.0f * f3);
            float w2 = 0.5f * (f + 4.0f * f2 - 3.0f * f3);
            float w3 = 0.5f * (-f2 + f3);

            byte[] p0 = keyframes[k0];
            byte[] p1 = keyframes[k1];
            byte[] p2 = keyframes[k2];
            byte[] p3 = keyframes[k3];

            byte[] interp = new byte[frameBytes];
            Parallel.For(0, frameBytes, p =>
            {
                float val = w0 * p0[p] + w1 * p1[p] + w2 * p2[p] + w3 * p3[p];
                interp[p] = (byte)Math.Clamp((int)MathF.Round(val), 0, 255);
            });

            result.Add(interp);
        }

        return result;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _wanDecoder?.Dispose();
            _neuralDecoder?.Dispose();
            _spatialDecoder.Dispose();
        }
    }
}
