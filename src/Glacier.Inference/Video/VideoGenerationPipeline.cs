namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Image;
using Glacier.Inference.Video.Wan;
using Glacier.Inference.Image.Flux;

/// <summary>
/// End-to-End Generative Video Production Pipeline powered by Wan2.1 Spatio-Temporal Diffusion Transformer GGUF weights,
/// T5-XXL text conditioning, 3D Causal VAE decoding, and zero-dependency video container export (APNG, GIF, AVI).
/// </summary>
public sealed class VideoGenerationPipeline : IDisposable
{
    private readonly WanDiT? _wanDit;
    private readonly FluxT5Encoder? _t5Encoder;
#pragma warning disable GLACIER_EXP001
    private readonly SpatioTemporalDiT _dit;
    private readonly TemporalLatentVaeDecoder _vae;
#pragma warning restore GLACIER_EXP001
    private bool _disposed;

    public WanDiT? WanDiT => _wanDit;
    public FluxT5Encoder? T5Encoder => _t5Encoder;
    [Experimental("GLACIER_EXP001")]
    public SpatioTemporalDiT DiT => _dit;
    [Experimental("GLACIER_EXP001")]
    public TemporalLatentVaeDecoder VAE => _vae;
    public string ActiveBackend => (_wanDit != null && _wanDit.IsGpuAccelerated) ? "Cuda" : "Cpu";
    public Exception? FallbackException { get; private set; }

#pragma warning disable GLACIER_EXP001
    public VideoGenerationPipeline(
        string? modelPath = null,
        string? t5Path = null,
        string? vaePath = null,
        int numLayers = 3,
        int hiddenDim = SpatioTemporalDiT.DefaultHiddenDim,
        int numHeads = SpatioTemporalDiT.DefaultNumHeads,
        int latentChannels = SpatioTemporalDiT.DefaultLatentChannels)
    {
        _dit = new SpatioTemporalDiT(numLayers, hiddenDim, numHeads, latentChannels);

        string? resolvedWanVae = vaePath ?? FindModelFile("wan_2.1_vae.safetensors");
        Wan3DVaeDecoder? wanVae = null;
        if (!string.IsNullOrEmpty(resolvedWanVae) && File.Exists(resolvedWanVae))
        {
            try
            {
                wanVae = Wan3DVaeDecoder.Open(resolvedWanVae, enableGpu: true);
                GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Loaded Wan 2.1 3D Causal VAE Decoder: {Path.GetFileName(resolvedWanVae)}");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER VIDEO] Wan 3D VAE load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                FallbackException ??= ex;
                wanVae = null;
            }
        }

        string? resolvedFluxVae = (wanVae == null) ? (vaePath ?? FindModelFile("ae.safetensors")) : null;
        FluxVaeDecoder? neuralVae = null;
        if (!string.IsNullOrEmpty(resolvedFluxVae) && File.Exists(resolvedFluxVae))
        {
            try
            {
                neuralVae = FluxVaeDecoder.Open(resolvedFluxVae, enableGpu: true);
                GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Loaded Neural VAE Decoder: {Path.GetFileName(resolvedFluxVae)} on GPU");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER VIDEO] Neural VAE load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                FallbackException ??= ex;
            }
        }
        _vae = new TemporalLatentVaeDecoder(latentChannels, neuralVae, wanVae);

        string? resolvedModel = modelPath ?? FindModelFile("Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        if (!string.IsNullOrEmpty(resolvedModel) && File.Exists(resolvedModel))
        {
            try
            {
                _wanDit = WanDiT.Open(resolvedModel, enableGpu: true);
                string mode = _wanDit.IsGpuAccelerated
                    ? $"{_wanDit.GpuDeviceName ?? "CUDA"} GPU"
                    : "CPU SIMD";
                GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Loaded pre-trained Video DiT: {Path.GetFileName(resolvedModel)} (30 blocks, 1.3B params) on {mode}");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER VIDEO] WanDiT load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                FallbackException ??= ex;
                _wanDit = null;
            }
        }

        string? resolvedT5 = t5Path ?? FindModelFile("t5xxl.gguf");
        if (!string.IsNullOrEmpty(resolvedT5) && File.Exists(resolvedT5))
        {
            try
            {
                _t5Encoder = FluxT5Encoder.Open(resolvedT5);
                GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Loaded T5-XXL text encoder: {Path.GetFileName(resolvedT5)}");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[GLACIER VIDEO] T5 load failed: {ex.GetType().FullName}: {ex.Message}", ex);
                FallbackException ??= ex;
                _t5Encoder = null;
            }
        }
    }
#pragma warning restore GLACIER_EXP001

    private static string? FindModelFile(string filename)
    {
        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates = [
            filename,
            Path.Combine("models", filename),
            Path.Combine("Glacier.Inference", "models", filename),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", filename),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models", filename),
            Path.Combine(userProfile, ".cache", "glacier", "models", filename)
        ];
        foreach (var c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return null;
    }

#pragma warning disable GLACIER_EXP001
    private void EnsureWanComponentsLoaded()
    {
        if (_wanDit == null)
        {
            throw new FileNotFoundException(
                "Wan2.1 Video DiT weights (e.g., Wan2.1-T2V-1.3B-Q4_K_M.gguf) were not found or failed to load. Procedural SpatioTemporalDiT fallback has been disabled.",
                FallbackException);
        }

        if (_t5Encoder == null)
        {
            throw new FileNotFoundException(
                "T5-XXL text encoder weights (t5xxl.gguf) are required for Wan2.1 video generation. ASCII prompt hashing fallback has been disabled.",
                FallbackException);
        }

        if (!_vae.HasNeuralVae)
        {
            throw new FileNotFoundException(
                "Neural 3D VAE weights (wan_2.1_vae.safetensors or ae.safetensors) are required for Wan2.1 video generation. Procedural VAE fallback has been disabled.",
                FallbackException);
        }
    }
#pragma warning restore GLACIER_EXP001

    /// <summary>
    /// Generates a cinematic video sequence from a textual prompt with Wan2.1 Flow Matching inference.
    /// </summary>
    public VideoGenerationResult Generate(
        string prompt,
        int width = 832,
        int height = 480,
        int numFrames = 17,
        int fps = 16,
        int numSteps = 25,
        CameraMotion motion = CameraMotion.PanRight,
        int? seed = null,
        float guidanceScale = 5.0f)
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

        EnsureWanComponentsLoaded();
        var wanDit = _wanDit!;
        var t5Encoder = _t5Encoder!;

        var sw = Stopwatch.StartNew();

#pragma warning disable GLACIER_EXP001
        int latentH = height / TemporalLatentVaeDecoder.SpatialScaleFactor;
        int latentW = width / TemporalLatentVaeDecoder.SpatialScaleFactor;
        int latentChannels = _dit.LatentChannels;
#pragma warning restore GLACIER_EXP001

        // Compute keyframe latent count: for Wan 3D causal VAE, N_frames = 1 + 4 * (K - 1)
        int spatialTokens = (latentH / 2) * (latentW / 2);
        int maxWanKeyframes = spatialTokens > 0 ? Math.Min(9, wanDit.MaxTokens / spatialTokens) : 8;
        int temporalLatentFrames = Math.Clamp((numFrames - 1) / 4 + 1, 2, maxWanKeyframes);

        int frameLatentSize = latentChannels * latentH * latentW;
        int totalLatentSize = temporalLatentFrames * frameLatentSize;

        const int textSeqLen = 512;
        GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Encoding prompt text context via pure C# T5-XXL (padded to {textSeqLen} tokens)...");
        var (promptEmbeds, validCount) = t5Encoder.EncodeWithCount(prompt, maxSeqLen: textSeqLen);
        float[] contextTxt = new float[textSeqLen * FluxT5Encoder.HiddenDim];
        Array.Copy(promptEmbeds, contextTxt, promptEmbeds.Length);

        float[]? uncondTxt = null;
        if (guidanceScale > 1.0f)
        {
            const string defaultNegPrompt = "blurry, distorted, deformed, low quality, cartoon, static, bad anatomy";
            var (negEmbeds, _) = t5Encoder.EncodeWithCount(defaultNegPrompt, maxSeqLen: textSeqLen);
            uncondTxt = new float[textSeqLen * FluxT5Encoder.HiddenDim];
            Array.Copy(negEmbeds, uncondTxt, negEmbeds.Length);
        }
        GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Text context encoded successfully ({validCount} valid prompt tokens padded to {textSeqLen}, uncond={textSeqLen} tokens).");

        var latents = new float[totalLatentSize];
        var velocity = new float[totalLatentSize];
        var uncondVel = (guidanceScale > 1.0f) ? new float[totalLatentSize] : null;
        var rnd = seed.HasValue ? new Random(seed.Value) : new Random();
        for (int i = 0; i < totalLatentSize; i++)
        {
            double u1 = Math.Max(1e-7, rnd.NextDouble());
            double u2 = rnd.NextDouble();
            latents[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // Flow Matching Scheduler with Wan 2.1 flow-shift 3.0
        var scheduler = new FlowMatchingScheduler(numSteps, timeShift: 3.0f);
        var timesteps = scheduler.Timesteps;

        GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Starting Flow Matching Euler Solver ({numSteps} steps, guidance={guidanceScale:F1}, tokens={temporalLatentFrames * (latentH / 2) * (latentW / 2)})...");
        for (int step = 0; step < numSteps; step++)
        {
            float currentT = timesteps[step];
            float nextT = timesteps[step + 1];
            var stepSw = Stopwatch.StartNew();

            wanDit.PredictVelocity(latents, temporalLatentFrames, latentH, latentW, currentT, contextTxt, textSeqLen, velocity);

            if (guidanceScale > 1.0f && uncondTxt != null && uncondVel != null)
            {
                wanDit.PredictVelocity(latents, temporalLatentFrames, latentH, latentW, currentT, uncondTxt, textSeqLen, uncondVel);
                for (int i = 0; i < totalLatentSize; i++)
                {
                    velocity[i] = uncondVel[i] + guidanceScale * (velocity[i] - uncondVel[i]);
                }
            }

            FlowMatchingScheduler.Step(latents, velocity, currentT, nextT);
            stepSw.Stop();

            float vMin = float.MaxValue, vMax = float.MinValue, vSum = 0f;
            float lMin = float.MaxValue, lMax = float.MinValue, lSum = 0f;
            for (int i = 0; i < latents.Length; i++)
            {
                float v = velocity[i];
                float l = latents[i];
                if (v < vMin) vMin = v;
                if (v > vMax) vMax = v;
                vSum += v;
                if (l < lMin) lMin = l;
                if (l > lMax) lMax = l;
                lSum += l;
            }
            float vMean = vSum / latents.Length;
            float lMean = lSum / latents.Length;

            GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Step {step + 1}/{numSteps} (t={currentT:F3} -> {nextT:F3}): {stepSw.ElapsedMilliseconds} ms | Vel[min={vMin:F2}, max={vMax:F2}, mean={vMean:F3}] | Lat[min={lMin:F2}, max={lMax:F2}, mean={lMean:F3}]");
        }

        GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] Decoding {temporalLatentFrames} latent keyframes into {numFrames} RGB frames via 3D Causal VAE...");
#pragma warning disable GLACIER_EXP001
        var frames = _vae.DecodeVideo(latents, temporalLatentFrames, numFrames, latentH, latentW);
#pragma warning restore GLACIER_EXP001
        GlacierDiagnostics.LogInformation($"[GLACIER VIDEO] 3D VAE decoding complete ({frames.Count} frames).");
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
    /// Generates video via neural Image-to-Video (I2V) Wan2.1 Diffusion Transformer inference.
    /// Encodes reference image into 16-channel spatial latents, evaluates Spatio-Temporal
    /// attention steered by camera motion, and decodes the neural trajectory through the 3D VAE decoder.
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

        EnsureWanComponentsLoaded();
        var wanDit = _wanDit!;
        var t5Encoder = _t5Encoder!;

        var sw = Stopwatch.StartNew();

#pragma warning disable GLACIER_EXP001
        int latentH = height / TemporalLatentVaeDecoder.SpatialScaleFactor;
        int latentW = width / TemporalLatentVaeDecoder.SpatialScaleFactor;
        int latentChannels = _dit.LatentChannels;
#pragma warning restore GLACIER_EXP001

        int spatialTokens = (latentH / 2) * (latentW / 2);
        int maxWanKeyframes = spatialTokens > 0 ? Math.Min(9, wanDit.MaxTokens / spatialTokens) : 8;
        int temporalLatentFrames = Math.Clamp((numFrames + 3) / 4, 2, maxWanKeyframes);
        int frameLatentSize = latentChannels * latentH * latentW;
        int totalLatentSize = temporalLatentFrames * frameLatentSize;

        // 1. Latent Encoding: map 24-bit RGB reference image to 16-channel spatial latents
        float[] baseImageLatent = EncodeRgbToLatents(rgbPixels, sourceWidth, sourceHeight, latentH, latentW, latentChannels);

        // 2. Synthesize multi-frame latent field with continuous optical warp
        var targetLatents = new float[totalLatentSize];
        for (int k = 0; k < temporalLatentFrames; k++)
        {
            float u = (temporalLatentFrames <= 1) ? 0.0f : (float)k / (temporalLatentFrames - 1);
            var (offX, offY, zoom, _) = GetCameraTransform(motion, u);
            int frameOffset = k * frameLatentSize;

            WarpLatents(baseImageLatent, targetLatents.AsSpan(frameOffset, frameLatentSize), latentH, latentW, latentChannels, offX, offY, zoom);
        }

        // 3. Initialize Spatio-Temporal Latents from reference trajectory
        var latents = new float[totalLatentSize];
        var velocity = new float[totalLatentSize];
        Array.Copy(targetLatents, latents, totalLatentSize);

        // 4. Flow Matching ODE Integration Loop (Wan2.1 DiT Forward Pass)
        var scheduler = new FlowMatchingScheduler(numSteps);
        var timesteps = scheduler.Timesteps;

        const int numTxtTokens = 64;
        float[] contextTxt = t5Encoder.Encode(prompt, seqLen: numTxtTokens);

        for (int step = 0; step < numSteps; step++)
        {
            float currentT = timesteps[step];
            float nextT = timesteps[step + 1];

            wanDit.PredictVelocity(latents, temporalLatentFrames, latentH, latentW, currentT, contextTxt, numTxtTokens, velocity);
            FlowMatchingScheduler.Step(latents, velocity, currentT, nextT);
        }

        // 5. 3D Spatio-Temporal Neural VAE Latent Decoding
#pragma warning disable GLACIER_EXP001
        var frames = _vae.DecodeVideo(latents, temporalLatentFrames, numFrames, latentH, latentW);
#pragma warning restore GLACIER_EXP001

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
    /// Infers camera motion dynamically from natural language prompt semantics.
    /// Enables 100% prompt-driven camera kinematics without requiring manual CLI motion flags.
    /// </summary>
    public static CameraMotion InferMotionFromPrompt(string prompt, CameraMotion defaultMotion = CameraMotion.PanRight)
    {
        if (string.IsNullOrWhiteSpace(prompt)) return defaultMotion;

        string p = prompt.ToLowerInvariant();

        if (p.Contains("toward the camera") || p.Contains("towards the camera") ||
            p.Contains("toward camera") || p.Contains("towards camera") ||
            p.Contains("walking toward") || p.Contains("walking towards") ||
            p.Contains("approaching") || p.Contains("coming closer") ||
            p.Contains("static") || p.Contains("tripod") || p.Contains("steady") || p.Contains("locked shot"))
        {
            return CameraMotion.Static;
        }

        if (p.Contains("push in") || p.Contains("push-in") || p.Contains("zoom in") ||
            p.Contains("zooming in") || p.Contains("dolly in"))
        {
            return CameraMotion.ZoomIn;
        }

        if (p.Contains("walking away") || p.Contains("receding") || p.Contains("pull back") ||
            p.Contains("pull-back") || p.Contains("zoom out") || p.Contains("zooming out") || p.Contains("dolly out"))
        {
            return CameraMotion.ZoomOut;
        }

        if (p.Contains("orbit") || p.Contains("orbiting") || p.Contains("circle") || p.Contains("circling") ||
            p.Contains("rotat") || p.Contains("around"))
        {
            return CameraMotion.Orbit;
        }

        if (p.Contains("tilt up") || p.Contains("tilting up") || p.Contains("rising") || p.Contains("looking up") ||
            p.Contains("ascending") || p.Contains("crane up"))
        {
            return CameraMotion.TiltUp;
        }

        if (p.Contains("tilt down") || p.Contains("tilting down") || p.Contains("looking down") ||
            p.Contains("descending") || p.Contains("crane down"))
        {
            return CameraMotion.TiltDown;
        }

        if (p.Contains("pan left") || p.Contains("panning left") || p.Contains("sweep left") || p.Contains("drift left"))
        {
            return CameraMotion.PanLeft;
        }

        if (p.Contains("pan right") || p.Contains("panning right") || p.Contains("sweep right") || p.Contains("drift right"))
        {
            return CameraMotion.PanRight;
        }

        if (p.Contains("drone") || p.Contains("aerial") || p.Contains("fly-through") || p.Contains("fpv") ||
            p.Contains("handheld camera"))
        {
            return CameraMotion.DynamicFluid;
        }

        return defaultMotion;
    }

    /// <summary>
    /// Computes the camera transformation vector (offsetX, offsetY, zoom, roll) along the timeline u in [0, 1].
    /// Uses Hermite SmoothStep easing and an overscanned viewport to ensure the camera sweeps
    /// authentically across the scene with zero edge clipping, border clamping, or trailing smears.
    /// </summary>
    public static (float offX, float offY, float zoom, float roll) GetCameraTransform(CameraMotion motion, float u)
    {
        // Smooth S-curve easing (Hermite SmoothStep) like a physical drone / stabilized gimbal
        float t = u * u * (3.0f - 2.0f * u);

        return motion switch
        {
            CameraMotion.PanRight => (
                -0.075f + t * 0.150f,                     // Smooth sweep from left to right (reveals new scenery on the right)
                MathF.Sin(u * MathF.PI) * 0.008f,         // Gentle natural vertical arc
                1.25f + t * 0.03f,                        // Subtle forward push
                -MathF.Sin(u * MathF.PI) * 0.010f         // Gentle drone bank into the turn
            ),
            CameraMotion.PanLeft => (
                0.075f - t * 0.150f,                      // Smooth sweep from right to left (reveals new scenery on the left)
                MathF.Sin(u * MathF.PI) * 0.008f,         // Gentle natural vertical arc
                1.25f + t * 0.03f,                        // Subtle forward push
                MathF.Sin(u * MathF.PI) * 0.010f          // Gentle drone bank into the turn
            ),
            CameraMotion.TiltUp => (
                0.0f,
                0.065f - t * 0.130f,                      // Starts low in the valley, sweeps up to majestic peaks & sky
                1.25f,
                0.0f
            ),
            CameraMotion.TiltDown => (
                0.0f,
                -0.065f + t * 0.130f,                     // Starts on peaks & sky, sweeps down into deep valley
                1.25f,
                0.0f
            ),
            CameraMotion.ZoomIn => (
                0.0f,
                0.0f,
                1.12f + t * 0.30f,                        // Smooth cinematic push-in towards focal subject
                0.0f
            ),
            CameraMotion.ZoomOut => (
                0.0f,
                0.0f,
                1.42f - t * 0.30f,                        // Smooth cinematic pull-back revealing wider scene
                0.0f
            ),
            CameraMotion.Orbit => (
                MathF.Sin((u - 0.5f) * MathF.PI * 0.70f) * 0.060f,
                (MathF.Cos((u - 0.5f) * MathF.PI * 0.70f) - 1.0f) * 0.035f,
                1.24f + MathF.Sin(u * MathF.PI) * 0.05f,
                -MathF.Sin((u - 0.5f) * MathF.PI * 0.70f) * 0.018f
            ),
            CameraMotion.DynamicFluid => (
                MathF.Sin(u * MathF.PI * 2.0f) * 0.040f,
                MathF.Cos(u * MathF.PI * 2.0f) * 0.025f,
                1.22f + MathF.Sin(u * MathF.PI) * 0.03f,
                MathF.Sin(u * MathF.PI * 2.0f) * 0.012f
            ),
            _ => (0.0f, 0.0f, 1.15f, 0.0f)
        };
    }

    /// <summary>
    /// Encodes a 24-bit RGB reference image into 16-channel spatial latents matching LatentVaeDecoder.
    /// Resamples the source image into [latentH, latentW] patches and extracts luminance, chromatic
    /// channels, and high-frequency structural contours.
    /// </summary>
    public static float[] EncodeRgbToLatents(
        byte[] rgbPixels,
        int sourceWidth,
        int sourceHeight,
        int latentH,
        int latentW,
        int channels = 16)
    {
        var latents = new float[channels * latentH * latentW];
        int hw = latentH * latentW;

        float blockW = (float)sourceWidth / latentW;
        float blockH = (float)sourceHeight / latentH;

        // Pass 1: downsample RGB to latent grid
        var avgR = new float[hw];
        var avgG = new float[hw];
        var avgB = new float[hw];
        var lum = new float[hw];

        for (int ly = 0; ly < latentH; ly++)
        {
            int startY = Math.Clamp((int)(ly * blockH), 0, sourceHeight - 1);
            int endY = Math.Clamp((int)((ly + 1) * blockH), startY + 1, sourceHeight);

            for (int lx = 0; lx < latentW; lx++)
            {
                int startX = Math.Clamp((int)(lx * blockW), 0, sourceWidth - 1);
                int endX = Math.Clamp((int)((lx + 1) * blockW), startX + 1, sourceWidth);

                float rSum = 0f, gSum = 0f, bSum = 0f;
                int count = 0;

                for (int y = startY; y < endY; y++)
                {
                    int rowOff = y * sourceWidth * 3;
                    for (int x = startX; x < endX; x++)
                    {
                        int px = rowOff + x * 3;
                        rSum += rgbPixels[px] / 255.0f;
                        gSum += rgbPixels[px + 1] / 255.0f;
                        bSum += rgbPixels[px + 2] / 255.0f;
                        count++;
                    }
                }

                int sIdx = ly * latentW + lx;
                float r = (count > 0) ? (rSum / count) : 0.5f;
                float g = (count > 0) ? (gSum / count) : 0.5f;
                float b = (count > 0) ? (bSum / count) : 0.5f;

                avgR[sIdx] = r;
                avgG[sIdx] = g;
                avgB[sIdx] = b;
                lum[sIdx] = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }

        // Pass 2: Project into 16-channel manifold aligned with LatentVaeDecoder photographic color weights
        for (int ly = 0; ly < latentH; ly++)
        {
            for (int lx = 0; lx < latentW; lx++)
            {
                int sIdx = ly * latentW + lx;
                float r = avgR[sIdx];
                float g = avgG[sIdx];
                float b = avgB[sIdx];
                float l = lum[sIdx];

                // Ch 0: Primary Red channel
                latents[0 * hw + sIdx] = r;

                // Ch 1: Primary Green channel
                latents[1 * hw + sIdx] = g;

                // Ch 2: Primary Blue channel
                latents[2 * hw + sIdx] = b;

                // Ch 3: Achromatic luminance structure
                latents[3 * hw + sIdx] = l;

                // Ch 4: Chroma warmth
                latents[4 * hw + sIdx] = Math.Clamp(r - b, -1.0f, 1.0f) * 0.5f;

                // Ch 5: Specular Gleam / Sunlight Glint
                latents[5 * hw + sIdx] = l > 0.8f ? (l - 0.8f) * 2.0f : 0.0f;

                // Ch 6: Atmospheric Shadow / Depth
                latents[6 * hw + sIdx] = (1.0f - l) * 0.30f;

                // Ch 7: Horizontal gradient (Sobel-like edge)
                int leftIdx = ly * latentW + Math.Max(0, lx - 1);
                int rightIdx = ly * latentW + Math.Min(latentW - 1, lx + 1);
                latents[7 * hw + sIdx] = (lum[rightIdx] - lum[leftIdx]) * 0.5f;

                // Ch 8: Vertical gradient
                int topIdx = Math.Max(0, ly - 1) * latentW + lx;
                int botIdx = Math.Min(latentH - 1, ly + 1) * latentW + lx;
                latents[8 * hw + sIdx] = (lum[botIdx] - lum[topIdx]) * 0.5f;

                // Ch 9..15: Remaining channels initialized cleanly
                for (int c = 9; c < channels; c++)
                {
                    latents[c * hw + sIdx] = 0.0f;
                }
            }
        }

        return latents;
    }

    /// <summary>
    /// Applies continuous sub-pixel perspective transformation to latent channels with bilinear resampling.
    /// Preserves exact underlying image structures and chromatic channels with zero procedural shader corruption.
    /// </summary>
    public static void WarpLatents(
        ReadOnlySpan<float> srcLatents,
        Span<float> dstLatents,
        int latentH,
        int latentW,
        int channels,
        float offX,
        float offY,
        float zoom)
    {
        int hw = latentH * latentW;
        for (int y = 0; y < latentH; y++)
        {
            float ny = (float)y / (latentH - 1);
            float sy = (ny - 0.5f) / zoom + 0.5f + offY;
            float srcY = sy * (latentH - 1);

            int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, latentH - 1);
            int y1 = Math.Clamp(y0 + 1, 0, latentH - 1);
            float fy = srcY - y0;

            for (int x = 0; x < latentW; x++)
            {
                float nx = (float)x / (latentW - 1);
                float sx = (nx - 0.5f) / zoom + 0.5f + offX;
                float srcX = sx * (latentW - 1);

                int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, latentW - 1);
                int x1 = Math.Clamp(x0 + 1, 0, latentW - 1);
                float fx = srcX - x0;

                int dstIdx = y * latentW + x;

                for (int c = 0; c < channels; c++)
                {
                    int chOff = c * hw;
                    float v00 = srcLatents[chOff + y0 * latentW + x0];
                    float v10 = srcLatents[chOff + y0 * latentW + x1];
                    float v01 = srcLatents[chOff + y1 * latentW + x0];
                    float v11 = srcLatents[chOff + y1 * latentW + x1];

                    float top = v00 * (1f - fx) + v10 * fx;
                    float bot = v01 * (1f - fx) + v11 * fx;
                    dstLatents[chOff + dstIdx] = top * (1f - fy) + bot * fy;
                }
            }
        }
    }

#pragma warning disable GLACIER_EXP001
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _wanDit?.Dispose();
            _t5Encoder?.Dispose();
            _dit.Dispose();
            _vae.Dispose();
        }
    }
#pragma warning restore GLACIER_EXP001
}
