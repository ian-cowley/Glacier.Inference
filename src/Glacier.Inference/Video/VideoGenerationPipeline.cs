namespace Glacier.Inference.Video;

using System;
using System.Diagnostics;
using Glacier.Inference.Image;

/// <summary>
/// End-to-End Generative Video Production Pipeline.
/// Combines 3D Spatio-Temporal Flow-Matching DiT (Text-to-Video), 3D-RoPE spatial-temporal embeddings,
/// Catmull-Rom cubic temporal spline VAE decoding, and zero-dependency video container export (APNG, GIF, AVI).
/// </summary>
public sealed class VideoGenerationPipeline : IDisposable
{
    private readonly SpatioTemporalDiT _dit;
    private readonly TemporalLatentVaeDecoder _vae;
    private bool _disposed;

    public SpatioTemporalDiT DiT => _dit;
    public TemporalLatentVaeDecoder VAE => _vae;

    public VideoGenerationPipeline(
        int numLayers = 3,
        int hiddenDim = SpatioTemporalDiT.DefaultHiddenDim,
        int numHeads = SpatioTemporalDiT.DefaultNumHeads,
        int latentChannels = SpatioTemporalDiT.DefaultLatentChannels)
    {
        _dit = new SpatioTemporalDiT(numLayers, hiddenDim, numHeads, latentChannels);
        _vae = new TemporalLatentVaeDecoder(latentChannels);
    }

    /// <summary>
    /// Generates a cinematic video sequence from a textual prompt with camera motion steering.
    /// </summary>
    /// <param name="prompt">Semantic scene description.</param>
    /// <param name="width">Output video frame pixel width (must be divisible by 16).</param>
    /// <param name="height">Output video frame pixel height (must be divisible by 16).</param>
    /// <param name="numFrames">Total output video frame count (e.g. 16 frames).</param>
    /// <param name="fps">Video playback framerate (e.g. 8 or 12 FPS).</param>
    /// <param name="numSteps">Flow-Matching ODE integration steps (e.g. 4 or 8).</param>
    /// <param name="motion">Camera motion dynamics trajectory (e.g. PanRight, ZoomIn, Orbit).</param>
    /// <param name="seed">Random seed for noise generation.</param>
    public VideoGenerationResult Generate(
        string prompt,
        int width = 256,
        int height = 256,
        int numFrames = 16,
        int fps = 8,
        int numSteps = 4,
        CameraMotion motion = CameraMotion.PanRight,
        int? seed = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (width % 16 != 0 || height % 16 != 0)
        {
            throw new ArgumentException($"Width ({width}) and Height ({height}) must be divisible by 16.");
        }

        if (numFrames < 1)
        {
            throw new ArgumentException("NumFrames must be at least 1.", nameof(numFrames));
        }

        var sw = Stopwatch.StartNew();

        int latentH = height / TemporalLatentVaeDecoder.SpatialScaleFactor; // e.g. 32
        int latentW = width / TemporalLatentVaeDecoder.SpatialScaleFactor;  // e.g. 32
        int latentChannels = _dit.LatentChannels;                          // 16

        // Compute keyframe latent count (temporal downscale factor 4, min 2 frames)
        int temporalLatentFrames = Math.Max(2, Math.Min(numFrames, (numFrames + 3) / 4));

        int frameLatentSize = latentChannels * latentH * latentW;
        int totalLatentSize = temporalLatentFrames * frameLatentSize;

        // 1. Synthesize Semantic Prompt Target Latents (z_0)
        var targetLatents = new float[frameLatentSize];
        PromptSemanticSynthesizer.SynthesizeTargetLatents(prompt, targetLatents, latentH, latentW, latentChannels, seed ?? 42);

        // 2. Initialize Spatio-Temporal Gaussian Noise Latents (z_1 ~ N(0, I))
        var latents = new float[totalLatentSize];
        var velocity = new float[totalLatentSize];

        var rnd = new Random(seed ?? 42);
        for (int i = 0; i < totalLatentSize; i++)
        {
            // Box-Muller standard normal transform
            double u1 = Math.Max(1e-7, rnd.NextDouble());
            double u2 = rnd.NextDouble();
            latents[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // 3. Flow Matching (Euler ODE Integration Loop across continuous time t in [1.0 -> 0.0])
        float dt = 1.0f / numSteps;

        for (int step = 0; step < numSteps; step++)
        {
            float t = 1.0f - (step * dt);

            // Predict velocity field across spatio-temporal latents
            _dit.PredictVelocity(latents, t, targetLatents, motion, temporalLatentFrames, latentH, latentW, velocity);

            // Euler integration step: z_{t - dt} = z_t - dt * v_t
            for (int i = 0; i < totalLatentSize; i++)
            {
                latents[i] -= dt * velocity[i];
            }
        }

        // 4. 3D VAE Temporal Spline Upsampling + Spatial Progressive Deconvolution to RGB
        var frames = _vae.DecodeVideo(latents, temporalLatentFrames, numFrames, latentH, latentW);

        sw.Stop();

        return new VideoGenerationResult(
            frames,
            width,
            height,
            fps,
            sw.ElapsedMilliseconds,
            prompt,
            motion);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _dit.Dispose();
        }
    }
}
