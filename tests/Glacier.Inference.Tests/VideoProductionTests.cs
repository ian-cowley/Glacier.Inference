namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using Glacier.Inference.Video;
using Xunit;

public class VideoProductionTests
{
    [Fact]
    public void SpatioTemporalDiT_PredictsConsistentVelocity()
    {
        int temporalFrames = 4;
        int latentH = 16;
        int latentW = 16;
        int latentChannels = 16;
        int totalLatents = temporalFrames * latentChannels * latentH * latentW;

        using var dit = new SpatioTemporalDiT(numLayers: 2, hiddenDim: 128, numHeads: 4);

        float[] latents = new float[totalLatents];
        float[] velocity = new float[totalLatents];
        float[] promptTarget = new float[latentChannels * latentH * latentW];

        for (int i = 0; i < totalLatents; i++) latents[i] = (float)Math.Sin(i * 0.1);
        for (int i = 0; i < promptTarget.Length; i++) promptTarget[i] = (float)Math.Cos(i * 0.15);

        dit.PredictVelocity(latents, 0.5f, promptTarget, CameraMotion.PanRight, temporalFrames, latentH, latentW, velocity);

        Assert.Equal(totalLatents, velocity.Length);
        for (int i = 0; i < velocity.Length; i++)
        {
            Assert.False(float.IsNaN(velocity[i]), $"Velocity at {i} is NaN");
            Assert.False(float.IsInfinity(velocity[i]), $"Velocity at {i} is Infinity");
        }

        // Verify motion affects velocity output
        float[] velocityPanLeft = new float[totalLatents];
        dit.PredictVelocity(latents, 0.5f, promptTarget, CameraMotion.PanLeft, temporalFrames, latentH, latentW, velocityPanLeft);

        bool hasDifference = false;
        for (int i = 0; i < velocity.Length; i++)
        {
            if (Math.Abs(velocity[i] - velocityPanLeft[i]) > 1e-4f)
            {
                hasDifference = true;
                break;
            }
        }
        Assert.True(hasDifference, "Different camera motions must produce distinct velocity fields.");
    }

    [Fact]
    public void FluxT5Encoder_TestOutput()
    {
        string t5Path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "t5xxl.gguf");
        if (!File.Exists(t5Path)) return;

        using var t5 = Glacier.Inference.Image.Flux.FluxT5Encoder.Open(t5Path);
        var (embeds, validCount) = t5.EncodeWithCount("A man walking along a city street");
        Console.WriteLine($"[T5 Test] Valid count: {validCount}, Embeds length: {embeds.Length}");
        float min = float.MaxValue, max = float.MinValue, sum = 0f;
        for (int i = 0; i < embeds.Length; i++)
        {
            float v = embeds[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        float mean = sum / embeds.Length;
        float var = 0f;
        for (int i = 0; i < embeds.Length; i++) var += MathF.Pow(embeds[i] - mean, 2);
        float std = MathF.Sqrt(var / embeds.Length);
        Console.WriteLine($"[T5 Test] Min: {min:F4}, Max: {max:F4}, Mean: {mean:F4}, Std: {std:F4}");
        Console.WriteLine($"[T5 Test] First 10 floats: {string.Join(", ", embeds.Take(10).Select(x => x.ToString("F3")))}");
    }

    [Fact]
    public void TemporalLatentVaeDecoder_UpsamplesAndDecodesSmoothly()
    {
        int temporalLatents = 3;
        int targetFrames = 8;
        int latentH = 16;
        int latentW = 16;
        int latentChannels = 16;
        int totalLatents = temporalLatents * latentChannels * latentH * latentW;

        var decoder = new TemporalLatentVaeDecoder(latentChannels);
        float[] latents = new float[totalLatents];
        int frameLatentFloats = latentChannels * latentH * latentW;
        for (int t = 0; t < temporalLatents; t++)
        {
            int tOff = t * frameLatentFloats;
            for (int c = 0; c < latentChannels; c++)
            {
                int cOff = tOff + c * latentH * latentW;
                for (int y = 0; y < latentH; y++)
                {
                    for (int x = 0; x < latentW; x++)
                    {
                        latents[cOff + y * latentW + x] = MathF.Sin(x * 0.2f + t * 0.15f) * MathF.Cos(y * 0.2f);
                    }
                }
            }
        }

        var frames = decoder.DecodeVideo(latents, temporalLatents, targetFrames, latentH, latentW);

        Assert.Equal(targetFrames, frames.Count);
        int expectedFrameBytes = (latentH * 8) * (latentW * 8) * 3; // 128x128x3
        foreach (var frame in frames)
        {
            Assert.Equal(expectedFrameBytes, frame.Length);
            Assert.Contains(frame, b => b > 0);
        }

        // Verify temporal continuity (neighboring frames should have low variance)
        for (int i = 1; i < frames.Count; i++)
        {
            double diffSum = 0;
            byte[] f1 = frames[i - 1];
            byte[] f2 = frames[i];
            for (int b = 0; b < f1.Length; b++)
            {
                int diff = f1[b] - f2[b];
                diffSum += diff * diff;
            }
            double mse = diffSum / f1.Length;
            Assert.True(mse < 2000.0, $"Inter-frame MSE ({mse}) too high for smooth video transitions.");
        }
    }

    [Fact]
    public void ApngWriter_ProducesValidApngSpecification()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.apng");
        try
        {
            int w = 64;
            int h = 64;
            int numFrames = 4;
            var frames = new List<byte[]>();

            for (int f = 0; f < numFrames; f++)
            {
                byte[] frame = new byte[w * h * 3];
                for (int i = 0; i < frame.Length; i += 3)
                {
                    frame[i] = (byte)(f * 60);
                    frame[i + 1] = 100;
                    frame[i + 2] = (byte)(255 - f * 60);
                }
                frames.Add(frame);
            }

            ApngWriter.SaveApng(tmpFile, frames, w, h, fps: 10);
            Assert.True(File.Exists(tmpFile));

            byte[] bytes = File.ReadAllBytes(tmpFile);
            Assert.True(bytes.Length > 100);

            // PNG signature
            byte[] pngSig = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
            Assert.Equal(pngSig, bytes.Take(8).ToArray());

            // Check required chunks: IHDR, acTL, fcTL, IDAT, fdAT, IEND
            string binaryStr = System.Text.Encoding.ASCII.GetString(bytes);
            Assert.Contains("IHDR", binaryStr);
            Assert.Contains("acTL", binaryStr);
            Assert.Contains("fcTL", binaryStr);
            Assert.Contains("IDAT", binaryStr);
            Assert.Contains("fdAT", binaryStr);
            Assert.Contains("IEND", binaryStr);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void GifWriter_ProducesValidGif89a()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.gif");
        try
        {
            int w = 64;
            int h = 64;
            int numFrames = 4;
            var frames = new List<byte[]>();

            for (int f = 0; f < numFrames; f++)
            {
                byte[] frame = new byte[w * h * 3];
                for (int i = 0; i < frame.Length; i += 3)
                {
                    frame[i] = (byte)(f * 50);
                    frame[i + 1] = 120;
                    frame[i + 2] = 200;
                }
                frames.Add(frame);
            }

            GifWriter.SaveGif(tmpFile, frames, w, h, fps: 8);
            Assert.True(File.Exists(tmpFile));

            byte[] bytes = File.ReadAllBytes(tmpFile);
            Assert.True(bytes.Length > 100);

            // GIF89a signature
            Assert.Equal("GIF89a", System.Text.Encoding.ASCII.GetString(bytes.Take(6).ToArray()));

            // Netscape looping extension
            string content = System.Text.Encoding.ASCII.GetString(bytes);
            Assert.Contains("NETSCAPE2.0", content);

            // GIF Trailer
            Assert.Equal(0x3B, bytes[^1]);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void AviWriter_ProducesValidRiffContainer()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.avi");
        try
        {
            int w = 64;
            int h = 64;
            int numFrames = 4;
            var frames = new List<byte[]>();

            for (int f = 0; f < numFrames; f++)
            {
                byte[] frame = new byte[w * h * 3];
                for (int i = 0; i < frame.Length; i += 3)
                {
                    frame[i] = 10;
                    frame[i + 1] = (byte)(f * 60);
                    frame[i + 2] = 220;
                }
                frames.Add(frame);
            }

            AviWriter.SaveAvi(tmpFile, frames, w, h, fps: 8);
            Assert.True(File.Exists(tmpFile));

            byte[] bytes = File.ReadAllBytes(tmpFile);
            Assert.True(bytes.Length > 100);

            // RIFF header
            Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(bytes.Take(4).ToArray()));
            Assert.Equal("AVI ", System.Text.Encoding.ASCII.GetString(bytes.Skip(8).Take(4).ToArray()));

            string content = System.Text.Encoding.ASCII.GetString(bytes);
            Assert.Contains("hdrl", content);
            Assert.Contains("avih", content);
            Assert.Contains("strl", content);
            Assert.Contains("movi", content);
            Assert.Contains("idx1", content);
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void VideoGenerationPipeline_RunsEndToEndSyntheticFlow()
    {
        using var pipeline = new VideoGenerationPipeline(numLayers: 2, hiddenDim: 128, numHeads: 4);
        var result = pipeline.Generate(
            "A cinematic landscape of neon auroras over icy fjords",
            width: 128,
            height: 128,
            numFrames: 4,
            fps: 8,
            numSteps: 2,
            motion: CameraMotion.PanRight,
            seed: 99);

        Assert.NotNull(result);
        Assert.Equal(4, result.NumFrames);
        Assert.Equal(128, result.Width);
        Assert.Equal(128, result.Height);
        Assert.Equal(8, result.Fps);
        Assert.True(result.ElapsedMilliseconds >= 0);

        string tmpDir = Path.Combine(Path.GetTempPath(), $"glacier_video_run_{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tmpDir);
            string apng = Path.Combine(tmpDir, "video.apng");
            string gif = Path.Combine(tmpDir, "video.gif");
            string avi = Path.Combine(tmpDir, "video.avi");

            result.SaveApng(apng);
            result.SaveGif(gif);
            result.SaveAvi(avi);
            result.SaveFrames(tmpDir);

            Assert.True(File.Exists(apng));
            Assert.True(File.Exists(gif));
            Assert.True(File.Exists(avi));
            Assert.True(File.Exists(Path.Combine(tmpDir, "frame_0001.png")));
            Assert.True(File.Exists(Path.Combine(tmpDir, "frame_0004.png")));
        }
        finally
        {
            if (Directory.Exists(tmpDir)) Directory.Delete(tmpDir, recursive: true);
        }
    }

    [Fact]
    public void VideoGenerationPipeline_RunsEndToEndImageToVideoFlow()
    {
        using var pipeline = new VideoGenerationPipeline(numLayers: 2, hiddenDim: 128, numHeads: 4);

        // Generate synthetic reference image RGB [128x128x3]
        byte[] rgbPixels = new byte[128 * 128 * 3];
        for (int y = 0; y < 128; y++)
        {
            for (int x = 0; x < 128; x++)
            {
                int idx = (y * 128 + x) * 3;
                rgbPixels[idx] = (byte)(x * 2);
                rgbPixels[idx + 1] = (byte)(y * 2);
                rgbPixels[idx + 2] = (byte)((x + y));
            }
        }

        var result = pipeline.GenerateFromImage(
            rgbPixels,
            sourceWidth: 128,
            sourceHeight: 128,
            prompt: "Cinematic mountain panorama",
            width: 128,
            height: 128,
            numFrames: 4,
            fps: 8,
            numSteps: 2,
            motion: CameraMotion.PanRight,
            seed: 42);

        Assert.NotNull(result);
        Assert.Equal(4, result.NumFrames);
        Assert.Equal(128, result.Width);
        Assert.Equal(128, result.Height);
        Assert.Equal(8, result.Fps);
        Assert.Equal(4, result.Frames.Count);
        Assert.Equal(128 * 128 * 3, result.Frames[0].Length);
    }

    [Fact]
    public void Wan3DVaeDecoder_DecodesLatentsToRGBFrames()
    {
        string? modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models");
        string vaePath = Path.Combine(modelDir, "wan_2.1_vae.safetensors");
        if (!File.Exists(vaePath))
        {
            string altPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "wan_2.1_vae.safetensors");
            if (File.Exists(altPath)) vaePath = altPath;
        }

        if (!File.Exists(vaePath)) return; // Skip if weights not present

        using var decoder = Glacier.Inference.Video.Wan.Wan3DVaeDecoder.Open(vaePath, enableGpu: false);
        int temporalLatents = 2; // Produces 5 frames (1 on chunk 0, 4 on chunk 1)
        int latentH = 16;
        int latentW = 16;
        int frameFloats = 16 * latentH * latentW;
        float[] latents = new float[temporalLatents * frameFloats];
        var rnd = new Random(42);
        for (int i = 0; i < latents.Length; i++) latents[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);

        var frames = decoder.DecodeVideo(latents, temporalLatents, 5, latentH, latentW);
        Assert.NotNull(frames);
        Assert.Equal(5, frames.Count);
        int expectedBytes = (latentH * 8) * (latentW * 8) * 3;
        Assert.Equal(expectedBytes, frames[0].Length);
    }

    [Fact]
    public void WanDiT_PredictVelocity_MatchesPyTorchReference()
    {
        string? modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models");
        string ggufPath = Path.Combine(modelDir, "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        if (!File.Exists(ggufPath))
        {
            string altPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
            if (File.Exists(altPath)) ggufPath = altPath;
        }

        if (!File.Exists(ggufPath)) return;

        string binPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_input.bin");
        if (!File.Exists(binPath))
        {
            string altBin = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_input.bin");
            if (File.Exists(altBin)) binPath = altBin;
        }
        if (!File.Exists(binPath)) return;

        byte[] bytes = File.ReadAllBytes(binPath);
        float[] latents = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, latents, 0, bytes.Length);

        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: false);
        float[] velocity = new float[latents.Length];
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: ReadOnlySpan<float>.Empty, numTxtTokens: 0, velocity);

        float vMin = float.MaxValue, vMax = float.MinValue, vSum = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float v = velocity[i];
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
            vSum += v;
        }
        float vMean = vSum / velocity.Length;
        float vVar = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float d = velocity[i] - vMean;
            vVar += d * d;
        }
        float vStd = MathF.Sqrt(vVar / velocity.Length);

        Console.WriteLine($"[WanDiT CPU] Min: {vMin:F4}, Max: {vMax:F4}, Mean: {vMean:F6}, Std: {vStd:F4}");
        for (int c = 0; c < 3; c++)
        {
            int cOff = c * 30 * 30;
            Console.WriteLine($"Channel {c}: [{velocity[cOff]:F4}, {velocity[cOff + 1]:F4}; {velocity[cOff + 30]:F4}, {velocity[cOff + 31]:F4}]");
        }
    }

    [Fact]
    public void WanDiT_PredictVelocity_WithTextContext_MatchesPyTorchReference()
    {
        string? modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models");
        string ggufPath = Path.Combine(modelDir, "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        if (!File.Exists(ggufPath))
        {
            string altPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
            if (File.Exists(altPath)) ggufPath = altPath;
        }
        if (!File.Exists(ggufPath)) return;

        string binPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_input.bin");
        if (!File.Exists(binPath))
        {
            string altBin = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_input.bin");
            if (File.Exists(altBin)) binPath = altBin;
        }
        if (!File.Exists(binPath)) return;

        string txtBinPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_text.bin");
        if (!File.Exists(txtBinPath))
        {
            string altTxt = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_text.bin");
            if (File.Exists(altTxt)) txtBinPath = altTxt;
        }
        if (!File.Exists(txtBinPath)) return;

        byte[] bytes = File.ReadAllBytes(binPath);
        float[] latents = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, latents, 0, bytes.Length);

        byte[] txtBytes = File.ReadAllBytes(txtBinPath);
        float[] txtContext = new float[txtBytes.Length / 4];
        Buffer.BlockCopy(txtBytes, 0, txtContext, 0, txtBytes.Length);

        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: false);
        float[] velocity = new float[latents.Length];
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: txtContext, numTxtTokens: 64, velocity);

        float vMin = float.MaxValue, vMax = float.MinValue, vSum = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float v = velocity[i];
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
            vSum += v;
        }
        float vMean = vSum / velocity.Length;
        float vVar = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float d = velocity[i] - vMean;
            vVar += d * d;
        }
        float vStd = MathF.Sqrt(vVar / velocity.Length);

        Console.WriteLine($"[WanDiT CPU With Text] Min: {vMin:F4}, Max: {vMax:F4}, Mean: {vMean:F6}, Std: {vStd:F4}");
        for (int c = 0; c < 3; c++)
        {
            int cOff = c * 30 * 30;
            Console.WriteLine($"Channel {c}: [{velocity[cOff]:F4}, {velocity[cOff + 1]:F4}; {velocity[cOff + 30]:F4}, {velocity[cOff + 31]:F4}]");
        }

        // Validate parity against PyTorch reference:
        // PyTorch: min=-2.6775, max=1.8894, mean=-0.318322, std=0.6432
        // Channel 0: [0.5165, -0.1233; -0.3323, 0.0109]
        Assert.InRange(vMean, -0.40f, -0.20f);
        Assert.InRange(vStd, 0.55f, 0.75f);
    }

    [Fact]
    public void WanDiT_PredictVelocity_Gpu_WithTextContext_MatchesPyTorchReference()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;

        string? modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models");
        string ggufPath = Path.Combine(modelDir, "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        if (!File.Exists(ggufPath))
        {
            string altPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
            if (File.Exists(altPath)) ggufPath = altPath;
        }
        if (!File.Exists(ggufPath)) return;

        string binPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_input.bin");
        if (!File.Exists(binPath))
        {
            string altBin = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_input.bin");
            if (File.Exists(altBin)) binPath = altBin;
        }
        if (!File.Exists(binPath)) return;

        string txtBinPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_text.bin");
        if (!File.Exists(txtBinPath))
        {
            string altTxt = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_text.bin");
            if (File.Exists(altTxt)) txtBinPath = altTxt;
        }
        if (!File.Exists(txtBinPath)) return;

        byte[] bytes = File.ReadAllBytes(binPath);
        float[] latents = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, latents, 0, bytes.Length);

        byte[] txtBytes = File.ReadAllBytes(txtBinPath);
        float[] txtContext = new float[txtBytes.Length / 4];
        Buffer.BlockCopy(txtBytes, 0, txtContext, 0, txtBytes.Length);

        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: true);
        if (!dit.IsGpuAccelerated) return;

        float[] velocity = new float[latents.Length];
        var sw = System.Diagnostics.Stopwatch.StartNew();
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: txtContext, numTxtTokens: 64, velocity);
        sw.Stop();

        float vMin = float.MaxValue, vMax = float.MinValue, vSum = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float v = velocity[i];
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
            vSum += v;
        }
        float vMean = vSum / velocity.Length;
        float vVar = 0f;
        for (int i = 0; i < velocity.Length; i++)
        {
            float d = velocity[i] - vMean;
            vVar += d * d;
        }
        float vStd = MathF.Sqrt(vVar / velocity.Length);

        Console.WriteLine($"[WanDiT GPU With Text in {sw.ElapsedMilliseconds} ms] Min: {vMin:F4}, Max: {vMax:F4}, Mean: {vMean:F6}, Std: {vStd:F4}");
        for (int tok = 0; tok < 4; tok++)
        {
            int px = tok * 2;
            float p00 = velocity[0 * 30 + px];
            float p01 = velocity[0 * 30 + px + 1];
            float p10 = velocity[1 * 30 + px];
            float p11 = velocity[1 * 30 + px + 1];
            Console.WriteLine($"Channel 0 Token {tok}: [{p00:F4}, {p01:F4}; {p10:F4}, {p11:F4}]");
        }

        Assert.InRange(vMean, -0.40f, -0.20f);
        Assert.InRange(vStd, 0.55f, 0.75f);
    }

    [Fact]
    public void WanDiT_TestCFG_WithPaddedTokens()
    {
        if (!Glacier.Inference.Gpu.GpuContext.IsSupported) return;

        string? modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models");
        string ggufPath = Path.Combine(modelDir, "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        if (!File.Exists(ggufPath))
        {
            string altPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
            if (File.Exists(altPath)) ggufPath = altPath;
        }
        if (!File.Exists(ggufPath)) return;

        string binPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_input.bin");
        if (!File.Exists(binPath))
        {
            string altBin = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_input.bin");
            if (File.Exists(altBin)) binPath = altBin;
        }
        if (!File.Exists(binPath)) return;

        string txtBinPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "test_text.bin");
        if (!File.Exists(txtBinPath))
        {
            string altTxt = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "PolarsPlus", "Glacier.Inference", "test_text.bin");
            if (File.Exists(altTxt)) txtBinPath = altTxt;
        }
        if (!File.Exists(txtBinPath)) return;

        byte[] bytes = File.ReadAllBytes(binPath);
        float[] latents = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, latents, 0, bytes.Length);

        byte[] txtBytes = File.ReadAllBytes(txtBinPath);
        float[] condTxt = new float[txtBytes.Length / 4];
        Buffer.BlockCopy(txtBytes, 0, condTxt, 0, txtBytes.Length);

        // Create 64-token unconditional embedding (empty string padded to 64 tokens)
        float[] uncondTxtPadded = new float[64 * 4096];
        // Create 1-token unconditional embedding
        float[] uncondTxt1 = new float[1 * 4096];

        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: true);
        if (!dit.IsGpuAccelerated) return;

        float[] condVel = new float[latents.Length];
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: condTxt, numTxtTokens: 64, condVel);

        float cvMin = float.MaxValue, cvMax = float.MinValue, cvSum = 0f;
        for (int i = 0; i < condVel.Length; i++) {
            if (condVel[i] < cvMin) cvMin = condVel[i];
            if (condVel[i] > cvMax) cvMax = condVel[i];
            cvSum += condVel[i];
        }
        float cvMean = cvSum / condVel.Length;
        float cvVar = 0f;
        for (int i = 0; i < condVel.Length; i++) cvVar += MathF.Pow(condVel[i] - cvMean, 2);
        float cvStd = MathF.Sqrt(cvVar / condVel.Length);
        Console.WriteLine($"[C# condVel] Min: {cvMin:F4}, Max: {cvMax:F4}, Mean: {cvMean:F4}, Std: {cvStd:F4}");
        for (int c = 0; c < 3; c++) {
            int cOff = c * 30 * 30;
            Console.WriteLine($"C# Channel {c}: [{condVel[cOff]:F4}, {condVel[cOff+1]:F4}; {condVel[cOff+30]:F4}, {condVel[cOff+31]:F4}]");
        }

        float[] uncondVelPadded = new float[latents.Length];
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: uncondTxtPadded, numTxtTokens: 64, uncondVelPadded);

        float[] uncondVel1 = new float[latents.Length];
        dit.PredictVelocity(latents, temporalFrames: 2, latentH: 30, latentW: 30, timestep: 1000.0f, textContext: uncondTxt1, numTxtTokens: 1, uncondVel1);

        // Compare CFG with 64-token uncond vs 1-token uncond
        float[] cfg64 = new float[latents.Length];
        float[] cfg1 = new float[latents.Length];
        for (int i = 0; i < latents.Length; i++)
        {
            cfg64[i] = uncondVelPadded[i] + 4.0f * (condVel[i] - uncondVelPadded[i]);
            cfg1[i] = uncondVel1[i] + 4.0f * (condVel[i] - uncondVel1[i]);
        }

        float var64 = 0f, var1 = 0f, mean64 = 0f, mean1 = 0f;
        for (int i = 0; i < latents.Length; i++) { mean64 += cfg64[i]; mean1 += cfg1[i]; }
        mean64 /= latents.Length; mean1 /= latents.Length;
        for (int i = 0; i < latents.Length; i++) { var64 += MathF.Pow(cfg64[i] - mean64, 2); var1 += MathF.Pow(cfg1[i] - mean1, 2); }
        float std64 = MathF.Sqrt(var64 / latents.Length);
        float std1 = MathF.Sqrt(var1 / latents.Length);

        Console.WriteLine($"[CFG-64 tokens] Mean: {mean64:F4}, Std: {std64:F4}");
        Console.WriteLine($"[CFG-1  token ] Mean: {mean1:F4}, Std: {std1:F4}");
        for (int tok = 0; tok < 4; tok++)
        {
            int px = tok * 2;
            Console.WriteLine($"CFG64 Token {tok}: [{cfg64[px]:F4}, {cfg64[px + 1]:F4}; {cfg64[30 + px]:F4}, {cfg64[30 + px + 1]:F4}]");
            Console.WriteLine($"CFG1  Token {tok}: [{cfg1[px]:F4}, {cfg1[px + 1]:F4}; {cfg1[30 + px]:F4}, {cfg1[30 + px + 1]:F4}]");
        }
    }

    [Fact]
    public void Wan3DVaeDecoder_DecodesLatentsDump()
    {
        string binPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference", "diffusers_step1_latents.bin");
        if (!File.Exists(binPath)) binPath = "diffusers_step1_latents.bin";
        if (!File.Exists(binPath)) binPath = "latents_dump.bin";
        if (!File.Exists(binPath)) return;

        string vaePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "wan_2.1_vae.safetensors");
        if (!File.Exists(vaePath)) return;

        using var br = new BinaryReader(File.OpenRead(binPath));
        int frames = br.ReadInt32();
        int H = br.ReadInt32();
        int W = br.ReadInt32();
        int count = frames * 16 * H * W;
        float[] latents = new float[count];
        for (int i = 0; i < count; i++) latents[i] = br.ReadSingle();

        using var decoder = Glacier.Inference.Video.Wan.Wan3DVaeDecoder.Open(vaePath);
        int expectedFrames = frames * 4 - 3;
        var decoded = decoder.DecodeVideo(latents, frames, targetFrames: expectedFrames, H, W);
        Assert.Equal(expectedFrames, decoded.Count);

        string repoDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference");
        int finalW = W * 8;
        int finalH = H * 8;

        Glacier.Inference.Video.ApngWriter.SaveApng(Path.Combine(repoDir, "walk_hd.apng"), decoded, finalW, finalH, fps: 16);
        Glacier.Inference.Video.GifWriter.SaveGif(Path.Combine(repoDir, "walk_hd.gif"), decoded, finalW, finalH, fps: 16);

        Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_hd_frame_0.png"), decoded[0], finalW, finalH);
        if (decoded.Count > expectedFrames / 2)
            Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_hd_frame_8.png"), decoded[expectedFrames / 2], finalW, finalH);
        if (decoded.Count > 1)
            Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_hd_frame_16.png"), decoded[^1], finalW, finalH);
    }

    [Fact]
    public void WanDiT_Step0_ParityWithDiffusers()
    {
        string repoDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference");
        string initLatPath = Path.Combine(repoDir, "diffusers_init_latents.bin");
        string promptEmbPath = Path.Combine(repoDir, "diffusers_prompt_embeds.bin");
        string refPredPath = Path.Combine(repoDir, "diffusers_step0_pred.bin");

        if (!File.Exists(initLatPath) || !File.Exists(promptEmbPath) || !File.Exists(refPredPath))
        {
            Console.WriteLine("[TEST SKIP] Diffusers step 0 reference files not found.");
            return;
        }

        // 1. Load inputs
        int frames, H, W;
        float[] latents;
        using (var br = new BinaryReader(File.OpenRead(initLatPath)))
        {
            frames = br.ReadInt32();
            H = br.ReadInt32();
            W = br.ReadInt32();
            int count = frames * 16 * H * W;
            latents = new float[count];
            for (int i = 0; i < count; i++) latents[i] = br.ReadSingle();
        }

        int txtSeqLen, txtDim;
        float[] promptEmbeds;
        using (var br = new BinaryReader(File.OpenRead(promptEmbPath)))
        {
            txtSeqLen = br.ReadInt32();
            txtDim = br.ReadInt32();
            promptEmbeds = new float[txtSeqLen * txtDim];
            for (int i = 0; i < promptEmbeds.Length; i++) promptEmbeds[i] = br.ReadSingle();
        }

        float[] refPred;
        using (var br = new BinaryReader(File.OpenRead(refPredPath)))
        {
            int rF = br.ReadInt32();
            int rH = br.ReadInt32();
            int rW = br.ReadInt32();
            refPred = new float[rF * 16 * rH * rW];
            for (int i = 0; i < refPred.Length; i++) refPred[i] = br.ReadSingle();
        }

        // 2. Run C# WanDiT
        string ggufPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference", "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");
        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: true);

        float[] csPred = new float[latents.Length];
        dit.PredictVelocity(latents, frames, H, W, timestep: 1000.0f, promptEmbeds, txtSeqLen, csPred);

        // 3. Compute statistics and differences
        float csMin = float.MaxValue, csMax = float.MinValue, csSum = 0f;
        float refMin = float.MaxValue, refMax = float.MinValue, refSum = 0f;
        float maeSum = 0f, maxDiff = 0f;

        for (int i = 0; i < latents.Length; i++)
        {
            float c = csPred[i];
            float r = refPred[i];
            float diff = MathF.Abs(c - r);

            if (c < csMin) csMin = c;
            if (c > csMax) csMax = c;
            csSum += c;

            if (r < refMin) refMin = r;
            if (r > refMax) refMax = r;
            refSum += r;

            maeSum += diff;
            if (diff > maxDiff) maxDiff = diff;
        }

        float csMean = csSum / latents.Length;
        float refMean = refSum / latents.Length;
        float mae = maeSum / latents.Length;

        float csVar = 0f, refVar = 0f;
        for (int i = 0; i < latents.Length; i++)
        {
            csVar += MathF.Pow(csPred[i] - csMean, 2);
            refVar += MathF.Pow(refPred[i] - refMean, 2);
        }
        float csStd = MathF.Sqrt(csVar / latents.Length);
        float refStd = MathF.Sqrt(refVar / latents.Length);

        Console.WriteLine($"=== WanDiT Step 0 Parity Comparison ===");
        Console.WriteLine($"Diffusers PyTorch : min={refMin:F4}, max={refMax:F4}, mean={refMean:F4}, std={refStd:F4}");
        Console.WriteLine($"Glacier C# Engine : min={csMin:F4}, max={csMax:F4}, mean={csMean:F4}, std={csStd:F4}");
        Console.WriteLine($"Mean Absolute Error (MAE): {mae:F4} | Max Difference: {maxDiff:F4}");

        for (int c = 0; c < 3; c++)
        {
            int off = c * H * W;
            Console.WriteLine($"Diffusers Ch{c} Top-Left 2x2: [{refPred[off]:F4}, {refPred[off+1]:F4}; {refPred[off+W]:F4}, {refPred[off+W+1]:F4}]");
            Console.WriteLine($"Glacier C# Ch{c} Top-Left 2x2: [{csPred[off]:F4}, {csPred[off+1]:F4}; {csPred[off+W]:F4}, {csPred[off+W+1]:F4}]");
        }
    }

    [Fact]
    public void WanPipeline_5Step_ExactDiffusersEquivalence()
    {
        string repoDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "source", "repos", "PolarsPlus", "Glacier.Inference");
        string initLatPath = Path.Combine(repoDir, "diffusers_init_latents.bin");
        string promptEmbPath = Path.Combine(repoDir, "diffusers_prompt_embeds.bin");
        string negEmbPath = Path.Combine(repoDir, "diffusers_neg_embeds.bin");
        string vaePath = Path.Combine(repoDir, "models", "wan_2.1_vae.safetensors");
        string ggufPath = Path.Combine(repoDir, "models", "Wan2.1-T2V-1.3B-Q4_K_M.gguf");

        if (!File.Exists(initLatPath) || !File.Exists(promptEmbPath) || !File.Exists(negEmbPath)) return;

        // 1. Load inputs
        int frames, H, W;
        float[] latents;
        using (var br = new BinaryReader(File.OpenRead(initLatPath)))
        {
            frames = br.ReadInt32();
            H = br.ReadInt32();
            W = br.ReadInt32();
            int count = frames * 16 * H * W;
            latents = new float[count];
            for (int i = 0; i < count; i++) latents[i] = br.ReadSingle();
        }

        int txtSeqLen, txtDim;
        float[] promptEmbeds;
        using (var br = new BinaryReader(File.OpenRead(promptEmbPath)))
        {
            txtSeqLen = br.ReadInt32();
            txtDim = br.ReadInt32();
            promptEmbeds = new float[txtSeqLen * txtDim];
            for (int i = 0; i < promptEmbeds.Length; i++) promptEmbeds[i] = br.ReadSingle();
        }

        float[] negEmbeds;
        using (var br = new BinaryReader(File.OpenRead(negEmbPath)))
        {
            br.ReadInt32(); br.ReadInt32();
            negEmbeds = new float[txtSeqLen * txtDim];
            for (int i = 0; i < negEmbeds.Length; i++) negEmbeds[i] = br.ReadSingle();
        }

        using var dit = Glacier.Inference.Video.Wan.WanDiT.Open(ggufPath, enableGpu: true);
        using var vae = Glacier.Inference.Video.Wan.Wan3DVaeDecoder.Open(vaePath);

        float[] timesteps = [1000.0f, 750.25f, 500.5f, 250.75f, 1.0f];
        float[] sigmas    = [1.0000f, 0.75025f, 0.5005f, 0.25075f, 0.0010f, 0.0000f];

        float[] condVel = new float[latents.Length];
        float[] uncondVel = new float[latents.Length];
        float[] velocity = new float[latents.Length];
        float guidanceScale = 5.0f;

        for (int step = 0; step < 5; step++)
        {
            float t = timesteps[step];
            float dt = sigmas[step + 1] - sigmas[step];

            dit.PredictVelocity(latents, frames, H, W, timestep: t, promptEmbeds, txtSeqLen, condVel);
            dit.PredictVelocity(latents, frames, H, W, timestep: t, negEmbeds, txtSeqLen, uncondVel);

            for (int i = 0; i < latents.Length; i++)
            {
                velocity[i] = uncondVel[i] + guidanceScale * (condVel[i] - uncondVel[i]);
                latents[i] += dt * velocity[i];
            }

            Console.WriteLine($"[C# Step {step + 1}/5] t={t:F1}, dt={dt:F4} | Lat: min={latents.Min():F3}, max={latents.Max():F3}, avgAbs={latents.Average(x => Math.Abs(x)):F3}");
        }

        var decoded = vae.DecodeVideo(latents, frames, targetFrames: frames * 4 - 3, H, W);
        Assert.Equal(17, decoded.Count);

        int finalW = W * 8;
        int finalH = H * 8;
        Glacier.Inference.Video.ApngWriter.SaveApng(Path.Combine(repoDir, "walk_cs.apng"), decoded, finalW, finalH, fps: 16);
        Glacier.Inference.Video.GifWriter.SaveGif(Path.Combine(repoDir, "walk_cs.gif"), decoded, finalW, finalH, fps: 16);
        Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_cs_frame_0.png"), decoded[0], finalW, finalH);
        Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_cs_frame_8.png"), decoded[8], finalW, finalH);
        Glacier.Inference.Image.PngWriter.SavePng24(Path.Combine(repoDir, "walk_cs_frame_16.png"), decoded[16], finalW, finalH);
    }
}

