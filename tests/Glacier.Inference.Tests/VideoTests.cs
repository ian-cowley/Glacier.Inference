namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using Glacier.Inference.Video;
using Xunit;

public class VideoTests
{
    [Fact]
    public void VideoFrameSampler_UniformSampling_ProducesCorrectIndices()
    {
        int totalFrames = 100;
        int targetKeyframes = 5;

        int[] indices = VideoFrameSampler.SampleUniformIndices(totalFrames, targetKeyframes);

        Assert.Equal(5, indices.Length);
        Assert.Equal(0, indices[0]);
        Assert.Equal(25, indices[1]);
        Assert.Equal(50, indices[2]);
        Assert.Equal(74, indices[3]);
        Assert.Equal(99, indices[4]);
    }

    [Fact]
    public void VideoFrameSampler_MotionEnergy_DetectsDynamicChanges()
    {
        int size = 100 * 100 * 3;
        byte[] staticA = new byte[size];
        byte[] staticB = new byte[size];
        Array.Fill(staticA, (byte)128);
        Array.Fill(staticB, (byte)128);

        float staticMotion = VideoFrameSampler.ComputeMotionEnergy(staticA, staticB);
        Assert.Equal(0.0f, staticMotion);

        byte[] moving = new byte[size];
        Array.Fill(moving, (byte)255);

        float activeMotion = VideoFrameSampler.ComputeMotionEnergy(staticA, moving);
        Assert.True(activeMotion > 0.4f, $"Expected high motion energy, got {activeMotion}");
    }

    [Fact]
    public void VideoRoPE_Applies3DRotaryEmbeddings()
    {
        var rope = new VideoRoPE(headDim: 64, ropeTheta: 10000.0f);
        Assert.Equal(16, rope.TemporalDim);
        Assert.Equal(24, rope.HeightDim);
        Assert.Equal(24, rope.WidthDim);

        float[] vec = new float[64];
        for (int i = 0; i < vec.Length; i++) vec[i] = 1.0f;

        rope.Apply3DRoPE(vec, t: 2, h: 5, w: 7);

        // Verify values rotated without NaN or Infinity
        for (int i = 0; i < vec.Length; i++)
        {
            Assert.False(float.IsNaN(vec[i]));
            Assert.False(float.IsInfinity(vec[i]));
        }
    }

    [Fact]
    public void VideoPipeline_ProcessesMultiFrameSequence_AnswersQueries()
    {
        using var pipeline = new VideoPipeline(numLayers: 2, visionDim: 256, llmDim: 512, patchSize: 14);

        int w = 120;
        int h = 90;
        int frameSize = w * h * 3;

        // Generate 6 synthetic video frames simulating a moving bright object
        var frames = new List<byte[]>();
        for (int f = 0; f < 6; f++)
        {
            byte[] frame = new byte[frameSize];
            int objX = f * 15;
            for (int y = 30; y < 60; y++)
            {
                for (int x = objX; x < Math.Min(w, objX + 20); x++)
                {
                    int idx = (y * w + x) * 3;
                    frame[idx] = 255;
                    frame[idx + 1] = 200;
                    frame[idx + 2] = 50;
                }
            }
            frames.Add(frame);
        }

        var (tokens, emb, meta) = pipeline.ProcessVideo(frames, w, h, targetKeyframes: 2);

        Assert.True(tokens > 0);
        Assert.NotEmpty(emb);
        Assert.Equal(6, meta.TotalFrames);
        Assert.Equal(2, meta.KeyframeSampleCount);
        Assert.True(meta.AverageMotionEnergy > 0.0f);

        // Query motion
        string motionAnswer = pipeline.Query(meta, "Describe the motion in this video", tokens);
        Assert.Contains("motion", motionAnswer, StringComparison.OrdinalIgnoreCase);

        // Query frame count
        string frameAnswer = pipeline.Query(meta, "How many frames are in the video?", tokens);
        Assert.Contains("6", frameAnswer);
    }
}
