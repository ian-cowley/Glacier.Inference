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
}
