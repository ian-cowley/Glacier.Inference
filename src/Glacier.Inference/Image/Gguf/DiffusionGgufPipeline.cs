namespace Glacier.Inference.Image.Gguf;

using System;
using System.Diagnostics;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Gguf;
using Glacier.Inference.Image;
using Glacier.Inference.Model;

/// <summary>
/// Production Text-to-Image Generation pipeline driven by pre-trained Flux.1 Diffusion GGUF weights
/// with hardware SIMD and CUDA GPU acceleration.
/// </summary>
public sealed class DiffusionGgufPipeline : IDisposable
{
    private readonly DiffusionGgufModel _model;
    private readonly DiffusionGgufModel? _vaeModel;
    private readonly Glacier.Inference.Image.Flux.FluxVaeDecoder? _neuralVae;
    private readonly Glacier.Inference.Image.Flux.FluxDiT? _fluxDit;
    private readonly Glacier.Inference.Image.Flux.FluxClipEncoder? _clipEncoder;
    private readonly Glacier.Inference.Image.Flux.FluxT5Encoder? _t5Encoder;
    private bool _disposed;

    public DiffusionGgufModel Model => _model;
    public DiffusionGgufModel? VaeModel => _vaeModel;
    public Glacier.Inference.Image.Flux.FluxVaeDecoder? NeuralVae => _neuralVae;
    public Glacier.Inference.Image.Flux.FluxDiT? FluxDiT => _fluxDit;
    public Glacier.Inference.Image.Flux.FluxClipEncoder? ClipEncoder => _clipEncoder;
    public Glacier.Inference.Image.Flux.FluxT5Encoder? T5Encoder => _t5Encoder;
    public string ActiveBackend => (_fluxDit != null && _fluxDit.IsGpuAccelerated) ? "Cuda" : "Cpu";
    public Exception? FallbackException { get; private set; }

    public DiffusionGgufPipeline(
        DiffusionGgufModel model,
        DiffusionGgufModel? vaeModel = null,
        string? vaeSafetensorsPath = null,
        string? clipSafetensorsPath = null,
        string? t5GgufPath = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _vaeModel = vaeModel;

        string? vaePath = vaeSafetensorsPath;
        if (string.IsNullOrEmpty(vaePath))
        {
            if (File.Exists("models/ae.safetensors")) vaePath = "models/ae.safetensors";
            else if (File.Exists(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "ae.safetensors")))
                vaePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "ae.safetensors");
        }

        if (!string.IsNullOrEmpty(vaePath) && File.Exists(vaePath))
        {
            try
            {
                _neuralVae = Glacier.Inference.Image.Flux.FluxVaeDecoder.Open(vaePath);
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"Neural VAE load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                _neuralVae = null;
                FallbackException ??= ex;
            }
        }

        string? clipPath = clipSafetensorsPath;
        if (string.IsNullOrEmpty(clipPath))
        {
            if (File.Exists("models/clip_l.safetensors")) clipPath = "models/clip_l.safetensors";
            else if (File.Exists(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "clip_l.safetensors")))
                clipPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "clip_l.safetensors");
        }

        if (!string.IsNullOrEmpty(clipPath) && File.Exists(clipPath))
        {
            try
            {
                _clipEncoder = Glacier.Inference.Image.Flux.FluxClipEncoder.Open(clipPath);
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"CLIP encoder load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                _clipEncoder = null;
                FallbackException ??= ex;
            }
        }

        string? t5Path = t5GgufPath;
        if (string.IsNullOrEmpty(t5Path))
        {
            if (File.Exists("models/t5xxl.gguf")) t5Path = "models/t5xxl.gguf";
            else if (File.Exists(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "t5xxl.gguf")))
                t5Path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "t5xxl.gguf");
        }

        if (!string.IsNullOrEmpty(t5Path) && File.Exists(t5Path))
        {
            try
            {
                _t5Encoder = Glacier.Inference.Image.Flux.FluxT5Encoder.Open(t5Path);
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"T5 encoder load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                _t5Encoder = null;
                FallbackException ??= ex;
            }
        }

        if (_model.DiffusionArch == UniversalArchitecture.Flux)
        {
            try
            {
                _fluxDit = Glacier.Inference.Image.Flux.FluxDiT.Open(_model.Gguf.FilePath);
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"Flux DiT load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                _fluxDit = null;
                FallbackException ??= ex;
            }
        }
    }


    private void EnsureFluxComponentsLoaded()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_model.DiffusionArch != UniversalArchitecture.Flux)
        {
            throw new NotSupportedException(
                $"Diffusion architecture '{_model.DiffusionArch}' is not supported by DiffusionGgufPipeline. " +
                "Only UniversalArchitecture.Flux (Flux.1) is currently implemented; procedural fallback has been disabled.");
        }

        if (_fluxDit == null)
        {
            throw new FileNotFoundException(
                "Flux.1 DiT weights failed to load from the GGUF model.",
                FallbackException);
        }

        if (_clipEncoder == null)
        {
            throw new FileNotFoundException(
                "Flux CLIP-L text encoder weights (clip_l.safetensors) are required for Flux.1 generation. Weightless fallback has been disabled.",
                FallbackException);
        }

        if (_t5Encoder == null)
        {
            throw new FileNotFoundException(
                "Flux T5-XXL text encoder weights (t5xxl.gguf) are required for Flux.1 generation. Weightless ASCII fallback has been disabled.",
                FallbackException);
        }

        if (_neuralVae == null)
        {
            throw new FileNotFoundException(
                "Flux neural VAE decoder weights (ae.safetensors) are required for Flux.1 generation. Procedural VAE fallback has been disabled.",
                FallbackException);
        }
    }

    /// <summary>
    /// Executes end-to-end text-to-image generation from prompt using pre-trained Flux.1 GGUF weights.
    /// </summary>
    public ImageGenerationResult Generate(
        string prompt,
        int? width = null,
        int? height = null,
        int? numSteps = null,
        float cfgScale = 3.5f,
        int? seed = null)
    {
        EnsureFluxComponentsLoaded();
        ArgumentNullException.ThrowIfNull(prompt);

        var sw = Stopwatch.StartNew();

        int targetW = width ?? _model.RecommendedResolution;
        int targetH = height ?? _model.RecommendedResolution;
        int steps = numSteps ?? _model.RecommendedSteps;
        int channels = _model.InChannels;

        int latentW = targetW / 8;
        int latentH = targetH / 8;
        int latentSize = channels * latentH * latentW;

        // 1. Initialize Latent Noise Field
        var latents = new float[latentSize];
        var velocity = new float[latentSize];
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

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

        // 2. Multi-Step Flow-Matching Trajectory
        var scheduler = new FlowMatchingScheduler(steps);
        var timesteps = scheduler.Timesteps;

        float[] pooledY = _clipEncoder!.EncodePrompt(prompt);
        const int numTxtTokens = 64;
        float[] contextTxt = _t5Encoder!.Encode(prompt, seqLen: numTxtTokens);

        var swStep = Stopwatch.StartNew();
        for (int step = 0; step < steps; step++)
        {
            var swS = Stopwatch.StartNew();
            float currentT = timesteps[step];
            float nextT = timesteps[step + 1];

            _fluxDit!.PredictVelocity(latents, latentH, latentW, currentT, pooledY, contextTxt, numTxtTokens, velocity);
            FlowMatchingScheduler.Step(latents, velocity, currentT, nextT);
            GlacierDiagnostics.LogInformation($"[PIPELINE TIMING] Step {step + 1}/{steps}: {swS.ElapsedMilliseconds} ms (Flow velocity + Euler step)");
        }
        GlacierDiagnostics.LogInformation($"[PIPELINE TIMING] Total DiT Latent Trajectory ({steps} steps): {swStep.ElapsedMilliseconds} ms");

        // 3. Decode Latents to RGB Pixels via Neural Flux VAE
        var rgbPixels = new byte[targetW * targetH * 3];
        var swVae = Stopwatch.StartNew();
        _neuralVae!.Decode(latents, latentH, latentW, rgbPixels);
        GlacierDiagnostics.LogInformation($"[PIPELINE TIMING] VAE Latent Decode ({latentW}x{latentH} -> {targetW}x{targetH}): {swVae.ElapsedMilliseconds} ms");

        sw.Stop();
        float pxPerSec = (targetW * targetH) / (float)Math.Max(sw.Elapsed.TotalSeconds, 1e-6);

        return new ImageGenerationResult(
            Width: targetW,
            Height: targetH,
            Steps: steps,
            ElapsedMilliseconds: sw.ElapsedMilliseconds,
            PixelsPerSecond: pxPerSec,
            RgbPixels: rgbPixels);
    }

    /// <summary>
    /// Generates a multi-frame video sequence via keyframe Flux.1 DiT inference and Neural Flux VAE decoding.
    /// </summary>
    public Glacier.Inference.Video.VideoGenerationResult GenerateVideo(
        string prompt,
        int? width = null,
        int? height = null,
        int numFrames = 8,
        int fps = 8,
        int? numSteps = null,
        int? seed = null)
    {
        EnsureFluxComponentsLoaded();
        ArgumentNullException.ThrowIfNull(prompt);

        var sw = Stopwatch.StartNew();

        int targetW = width ?? 256;
        int targetH = height ?? 256;
        int steps = numSteps ?? 4;
        int channels = _model.InChannels;

        int latentW = targetW / 8;
        int latentH = targetH / 8;
        int frameLatentSize = channels * latentH * latentW;

        int keyframeCount = Math.Clamp((numFrames + 2) / 3, 2, 4);

        float[] pooledY = _clipEncoder!.EncodePrompt(prompt);
        const int numTxtTokens = 64;
        float[] contextTxt = _t5Encoder!.Encode(prompt, seqLen: numTxtTokens);

        var keyframeLatents = new List<float[]>(keyframeCount);
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

        var baseNoise = new float[frameLatentSize];
        for (int i = 0; i < frameLatentSize; i += 2)
        {
            float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
            float u2 = (float)rng.NextDouble();
            float mag = MathF.Sqrt(-2.0f * MathF.Log(u1));
            baseNoise[i] = mag * MathF.Cos(2.0f * MathF.PI * u2);
            if (i + 1 < frameLatentSize) baseNoise[i + 1] = mag * MathF.Sin(2.0f * MathF.PI * u2);
        }

        for (int k = 0; k < keyframeCount; k++)
        {
            float u = (keyframeCount <= 1) ? 0.0f : (float)k / (keyframeCount - 1);
            GlacierDiagnostics.LogInformation($"[NEURAL VIDEO] Denoising Keyframe {k + 1}/{keyframeCount} (progression={u:F2})...");

            var currentLatents = new float[frameLatentSize];
            float correlation = MathF.Cos(u * 0.40f);
            float innovation = MathF.Sin(u * 0.40f);

            for (int i = 0; i < frameLatentSize; i += 2)
            {
                float u1 = MathF.Max(1e-7f, (float)rng.NextDouble());
                float u2 = (float)rng.NextDouble();
                float mag = MathF.Sqrt(-2.0f * MathF.Log(u1));
                float z0 = mag * MathF.Cos(2.0f * MathF.PI * u2);
                float z1 = mag * MathF.Sin(2.0f * MathF.PI * u2);

                currentLatents[i] = baseNoise[i] * correlation + z0 * innovation;
                if (i + 1 < frameLatentSize)
                {
                    currentLatents[i + 1] = baseNoise[i + 1] * correlation + z1 * innovation;
                }
            }

            var scheduler = new FlowMatchingScheduler(steps);
            var timesteps = scheduler.Timesteps;
            var velocity = new float[frameLatentSize];

            for (int step = 0; step < steps; step++)
            {
                float currentT = timesteps[step];
                float nextT = timesteps[step + 1];

                _fluxDit!.PredictVelocity(currentLatents, latentH, latentW, currentT, pooledY, contextTxt, numTxtTokens, velocity);
                FlowMatchingScheduler.Step(currentLatents, velocity, currentT, nextT);
            }

            keyframeLatents.Add(currentLatents);
        }

        var frames = new List<byte[]>(numFrames);
        int frameRgbBytes = targetW * targetH * 3;
        var interpolatedLatent = new float[frameLatentSize];

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            float u = (numFrames <= 1) ? 0.0f : (float)frameIdx / (numFrames - 1);
            float tKey = u * (keyframeCount - 1);
            int k1 = (int)MathF.Floor(tKey);
            int k2 = Math.Min(k1 + 1, keyframeCount - 1);
            int k0 = Math.Max(0, k1 - 1);
            int k3 = Math.Min(keyframeCount - 1, k2 + 1);
            float s = tKey - k1;

            float s2 = s * s;
            float s3 = s2 * s;
            float w0 = 0.5f * (-s + 2.0f * s2 - s3);
            float w1 = 0.5f * (2.0f - 5.0f * s2 + 3.0f * s3);
            float w2 = 0.5f * (s + 4.0f * s2 - 3.0f * s3);
            float w3 = 0.5f * (-s2 + s3);

            var p0 = keyframeLatents[k0];
            var p1 = keyframeLatents[k1];
            var p2 = keyframeLatents[k2];
            var p3 = keyframeLatents[k3];

            for (int i = 0; i < frameLatentSize; i++)
            {
                interpolatedLatent[i] = w0 * p0[i] + w1 * p1[i] + w2 * p2[i] + w3 * p3[i];
            }

            byte[] frameBytes = new byte[frameRgbBytes];
            _neuralVae!.Decode(interpolatedLatent, latentH, latentW, frameBytes);
            frames.Add(frameBytes);
        }

        sw.Stop();

        return new Glacier.Inference.Video.VideoGenerationResult(
            frames,
            targetW,
            targetH,
            fps,
            sw.ElapsedMilliseconds,
            prompt,
            Glacier.Inference.Video.CameraMotion.ZoomIn);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _model.Dispose();
            _vaeModel?.Dispose();
            _neuralVae?.Dispose();
            _fluxDit?.Dispose();
            _clipEncoder?.Dispose();
            _t5Encoder?.Dispose();
            _disposed = true;
        }
    }
}

