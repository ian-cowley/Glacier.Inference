namespace Glacier.Inference.Image;

using System;
using System.Diagnostics;
using System.Security.Cryptography;

/// <summary>
/// Generation metrics for image production.
/// </summary>
public sealed record ImageGenerationResult(
    int Width,
    int Height,
    int Steps,
    long ElapsedMilliseconds,
    float PixelsPerSecond,
    byte[] RgbPixels);

/// <summary>
/// End-to-end Text-to-Image Generation pipeline executing distilled Flow-Matching DiT
/// and Progressive Latent VAE decoding in pure C# .NET 10.
/// </summary>
public sealed class ImageGenerationPipeline : IDisposable
{
    private readonly DiffusionTransformer _dit;
    private readonly LatentVaeDecoder _vae;
    private bool _disposed;

    public DiffusionTransformer DiT => _dit;
    public LatentVaeDecoder VAE => _vae;

    public ImageGenerationPipeline(
        int numLayers = 4,
        int hiddenDim = DiffusionTransformer.DefaultHiddenDim,
        int numHeads = DiffusionTransformer.DefaultNumHeads,
        int latentChannels = DiffusionTransformer.DefaultLatentChannels)
    {
        _dit = new DiffusionTransformer(numLayers, hiddenDim, numHeads, latentChannels);
        _vae = new LatentVaeDecoder(latentChannels);
    }

    /// <summary>
    /// Generates an image from a textual prompt using Flow Matching (Rectified Flow).
    /// </summary>
    public ImageGenerationResult Generate(
        string prompt,
        int width = 256,
        int height = 256,
        int numSteps = 4,
        int? seed = null)
    {
        if (width % 16 != 0 || height % 16 != 0)
        {
            throw new ArgumentException($"Width ({width}) and Height ({height}) must be divisible by 16.");
        }

        var sw = Stopwatch.StartNew();

        int latentH = height / LatentVaeDecoder.SpatialScaleFactor; // e.g. 32
        int latentW = width / LatentVaeDecoder.SpatialScaleFactor;  // e.g. 32
        int latentChannels = _dit.LatentChannels;                  // 16
        int latentSize = latentChannels * latentH * latentW;

        // 1. Initialize Gaussian Latent Noise z_1 ~ N(0, I)
        var latents = new float[latentSize];
        var velocity = new float[latentSize];
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

        // Box-Muller transform for standard normal distribution
        for (int i = 0; i < latentSize; i += 2)
        {
            float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
            float u2 = (float)rng.NextDouble();
            float mag = MathF.Sqrt(-2.0f * MathF.Log(u1));
            float z0 = mag * MathF.Cos(2.0f * MathF.PI * u2);
            float z1 = mag * MathF.Sin(2.0f * MathF.PI * u2);

            latents[i] = z0;
            if (i + 1 < latentSize) latents[i + 1] = z1;
        }

        // 2. Synthesize Prompt Conditioning Embedding
        var promptEmbedding = new float[128];
        for (int i = 0; i < prompt.Length; i++)
        {
            int idx = i % promptEmbedding.Length;
            promptEmbedding[idx] += (prompt[i] % 32) / 32.0f;
        }

        // 3. Flow Matching Euler ODE Trajectory
        var scheduler = new FlowMatchingScheduler(numSteps);
        var timesteps = scheduler.Timesteps;

        for (int step = 0; step < numSteps; step++)
        {
            float currentT = timesteps[step];
            float nextT = timesteps[step + 1];

            // Velocity prediction v_theta(x_t, t, c)
            _dit.PredictVelocity(latents, latentH, latentW, currentT, promptEmbedding, velocity);

            // Euler integration step: x_{t + dt} = x_t + dt * v
            FlowMatchingScheduler.Step(latents, velocity, currentT, nextT);
        }

        // 4. Latent VAE Decoding: [16, 32, 32] -> [256, 256, 3] RGB
        var rgbPixels = new byte[width * height * 3];
        _vae.Decode(latents, latentH, latentW, rgbPixels);

        sw.Stop();
        float pxPerSec = (width * height) / (float)sw.Elapsed.TotalSeconds;

        return new ImageGenerationResult(
            Width: width,
            Height: height,
            Steps: numSteps,
            ElapsedMilliseconds: sw.ElapsedMilliseconds,
            PixelsPerSecond: pxPerSec,
            RgbPixels: rgbPixels);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _dit.Dispose();
            _vae.Dispose();
            _disposed = true;
        }
    }
}
