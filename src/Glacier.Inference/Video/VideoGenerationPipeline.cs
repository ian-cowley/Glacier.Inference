namespace Glacier.Inference.Video;

using System;
using System.Diagnostics;
using System.IO;
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

        // 1. Synthesize Semantic Prompt Target Latents (z_0) for each keyframe with continuous camera motion
        var targetLatents = new float[totalLatentSize];
        for (int k = 0; k < temporalLatentFrames; k++)
        {
            float u = (temporalLatentFrames <= 1) ? 0.0f : (float)k / (temporalLatentFrames - 1);
            var (offX, offY, zoom) = GetCameraTransform(motion, u);
            int frameOffset = k * frameLatentSize;

            PromptSemanticSynthesizer.SynthesizeTargetLatents(
                prompt,
                targetLatents.AsSpan(frameOffset, frameLatentSize),
                latentH,
                latentW,
                latentChannels,
                seed ?? 42,
                offX,
                offY,
                zoom);
        }

        // 2. Initialize Spatio-Temporal Gaussian Noise Latents (z_1 ~ N(0, I))
        var latents = new float[totalLatentSize];
        var velocity = new float[totalLatentSize];
        var ditVelocity = new float[totalLatentSize];

        var rnd = new Random(seed ?? 42);
        for (int i = 0; i < totalLatentSize; i++)
        {
            // Box-Muller standard normal transform
            double u1 = Math.Max(1e-7, rnd.NextDouble());
            double u2 = rnd.NextDouble();
            float z = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));

            // Structured latent initial condition: 25% semantic target scaffold + 75% Gaussian noise
            latents[i] = targetLatents[i] * 0.25f + z * 0.75f;
        }

        // 3. Flow Matching (Euler ODE Integration Loop across continuous time t in [1.0 -> 0.0])
        var scheduler = new FlowMatchingScheduler(numSteps);
        var timesteps = scheduler.Timesteps;

        for (int step = 0; step < numSteps; step++)
        {
            float currentT = timesteps[step];
            float nextT = timesteps[step + 1];

            // Evaluate Spatio-Temporal DiT (Intra-frame 3D-RoPE Attention + Inter-frame Cross Attention)
            _dit.PredictVelocity(latents, currentT, targetLatents, motion, temporalLatentFrames, latentH, latentW, ditVelocity);

            float denom = MathF.Max(currentT, 0.05f);
            float ditModulation = currentT * (1.0f - currentT) * 2.0f; // Smooth bell curve peaking at t=0.5

            for (int i = 0; i < totalLatentSize; i++)
            {
                // Rectified Flow Velocity points from noisy state x_t to clean semantic target x_0
                float baseVelocity = (latents[i] - targetLatents[i]) / denom;
                velocity[i] = baseVelocity + ditVelocity[i] * ditModulation;
            }

            // Euler integration step: x_{t + dt} = x_t + dt * v (where dt = nextT - currentT < 0)
            FlowMatchingScheduler.Step(latents, velocity, currentT, nextT);
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

    /// <summary>
    /// Generates video by animating an input reference image with continuous camera motion and 3D VAE splines.
    /// </summary>
    public VideoGenerationResult GenerateFromImage(
        byte[] rgbPixels,
        int sourceWidth,
        int sourceHeight,
        string prompt = "cinematic camera animation",
        int width = 256,
        int height = 256,
        int numFrames = 16,
        int fps = 8,
        CameraMotion motion = CameraMotion.PanRight)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sw = Stopwatch.StartNew();

        int latentH = height / TemporalLatentVaeDecoder.SpatialScaleFactor; // e.g. 32
        int latentW = width / TemporalLatentVaeDecoder.SpatialScaleFactor;  // e.g. 32
        int latentChannels = _dit.LatentChannels;                          // 16

        int temporalLatentFrames = Math.Max(2, Math.Min(numFrames, (numFrames + 3) / 4));
        int frameLatentSize = latentChannels * latentH * latentW;
        int totalLatentSize = temporalLatentFrames * frameLatentSize;

        // Downsample input RGB to base latent representation
        float[] baseLatents = new float[frameLatentSize];
        int hw = latentH * latentW;

        for (int y = 0; y < latentH; y++)
        {
            int srcY = Math.Clamp((int)((float)y / latentH * sourceHeight), 0, sourceHeight - 1);
            for (int x = 0; x < latentW; x++)
            {
                int srcX = Math.Clamp((int)((float)x / latentW * sourceWidth), 0, sourceWidth - 1);
                int srcIdx = (srcY * sourceWidth + srcX) * 3;

                float r = rgbPixels[srcIdx] / 255.0f;
                float g = rgbPixels[srcIdx + 1] / 255.0f;
                float b = rgbPixels[srcIdx + 2] / 255.0f;

                float lum = 0.299f * r + 0.587f * g + 0.114f * b;
                float cyan = MathF.Max(0f, (b + g) * 0.5f - r);
                float amber = MathF.Max(0f, (r + g) * 0.5f - b);
                float emerald = MathF.Max(0f, g - (r + b) * 0.5f);
                float specular = (lum > 0.85f) ? (lum - 0.85f) * 5.0f : 0f;

                int spatialIdx = y * latentW + x;
                baseLatents[0 * hw + spatialIdx] = lum;
                baseLatents[1 * hw + spatialIdx] = cyan * 0.8f;
                baseLatents[3 * hw + spatialIdx] = amber * 0.7f;
                baseLatents[4 * hw + spatialIdx] = emerald * 0.6f;
                baseLatents[5 * hw + spatialIdx] = specular;
            }
        }

        // Synthesize camera-shifted keyframes with bilinear texture coordinate sampling
        var latents = new float[totalLatentSize];
        for (int k = 0; k < temporalLatentFrames; k++)
        {
            float u = (temporalLatentFrames <= 1) ? 0.0f : (float)k / (temporalLatentFrames - 1);
            var (offX, offY, zoom) = GetCameraTransform(motion, u);
            int frameOffset = k * frameLatentSize;

            for (int c = 0; c < latentChannels; c++)
            {
                int cOff = c * hw;
                for (int y = 0; y < latentH; y++)
                {
                    float ny = (float)y / (latentH - 1);
                    float sy = (ny - 0.5f) / zoom + 0.5f + offY;

                    for (int x = 0; x < latentW; x++)
                    {
                        float nx = (float)x / (latentW - 1);
                        float sx = (nx - 0.5f) / zoom + 0.5f + offX;

                        float sampleX = Math.Clamp(sx * (latentW - 1), 0f, latentW - 1);
                        float sampleY = Math.Clamp(sy * (latentH - 1), 0f, latentH - 1);

                        int x0 = (int)sampleX;
                        int y0 = (int)sampleY;
                        int x1 = Math.Min(x0 + 1, latentW - 1);
                        int y1 = Math.Min(y0 + 1, latentH - 1);

                        float fx = sampleX - x0;
                        float fy = sampleY - y0;

                        float v00 = baseLatents[cOff + y0 * latentW + x0];
                        float v10 = baseLatents[cOff + y0 * latentW + x1];
                        float v01 = baseLatents[cOff + y1 * latentW + x0];
                        float v11 = baseLatents[cOff + y1 * latentW + x1];

                        float val = (v00 * (1f - fx) + v10 * fx) * (1f - fy) + (v01 * (1f - fx) + v11 * fx) * fy;
                        latents[frameOffset + cOff + y * latentW + x] = val;
                    }
                }
            }
        }

        // Decode through 3D VAE with Catmull-Rom temporal spline upsampling
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

    /// <summary>
    /// Computes the camera transformation vector (offsetX, offsetY, zoom) along the timeline u in [0, 1].
    /// </summary>
    public static (float offX, float offY, float zoom) GetCameraTransform(CameraMotion motion, float u)
    {
        return motion switch
        {
            CameraMotion.PanRight => (u * 0.22f, 0.0f, 1.0f),
            CameraMotion.PanLeft => (-u * 0.22f, 0.0f, 1.0f),
            CameraMotion.TiltUp => (0.0f, -u * 0.15f, 1.0f),
            CameraMotion.TiltDown => (0.0f, u * 0.15f, 1.0f),
            CameraMotion.ZoomIn => (0.0f, 0.0f, 1.0f + u * 0.30f),
            CameraMotion.ZoomOut => (0.0f, 0.0f, 1.30f - u * 0.30f),
            CameraMotion.Orbit => (
                MathF.Sin(u * MathF.PI * 0.8f) * 0.15f,
                MathF.Cos(u * MathF.PI * 0.8f) * 0.06f,
                1.0f + MathF.Sin(u * MathF.PI) * 0.08f
            ),
            CameraMotion.DynamicFluid => (
                MathF.Sin(u * 3.0f) * 0.10f,
                MathF.Cos(u * 2.5f) * 0.06f,
                1.0f
            ),
            _ => (0.0f, 0.0f, 1.0f)
        };
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
