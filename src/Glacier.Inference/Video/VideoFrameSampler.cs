namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;

/// <summary>
/// Temporal frame sampling and keyframe extraction for video-language models.
/// Operates directly on raw frame buffers with zero external dependencies.
/// </summary>
public static class VideoFrameSampler
{
    /// <summary>
    /// Uniformly selects N keyframes from an arbitrary sequence of frames.
    /// </summary>
    public static int[] SampleUniformIndices(int totalFrames, int targetKeyframes)
    {
        if (totalFrames <= 0) return Array.Empty<int>();
        if (targetKeyframes <= 0) return Array.Empty<int>();
        if (totalFrames <= targetKeyframes)
        {
            int[] all = new int[totalFrames];
            for (int i = 0; i < totalFrames; i++) all[i] = i;
            return all;
        }

        int[] indices = new int[targetKeyframes];
        double step = (double)(totalFrames - 1) / (targetKeyframes - 1);
        for (int i = 0; i < targetKeyframes; i++)
        {
            indices[i] = (int)Math.Round(i * step);
        }
        return indices;
    }

    /// <summary>
    /// Computes the mean temporal pixel difference (motion energy) between two RGB frames.
    /// Higher values indicate camera movement, scene cuts, or rapid subject action.
    /// </summary>
    public static float ComputeMotionEnergy(ReadOnlySpan<byte> frameA, ReadOnlySpan<byte> frameB)
    {
        int len = Math.Min(frameA.Length, frameB.Length);
        if (len == 0) return 0.0f;

        long diffSum = 0;
        int step = 4; // Subsample for fast temporal motion estimation
        int samples = 0;

        for (int i = 0; i < len; i += step)
        {
            diffSum += Math.Abs((int)frameA[i] - (int)frameB[i]);
            samples++;
        }

        return samples > 0 ? (float)diffSum / (samples * 255.0f) : 0.0f;
    }

    /// <summary>
    /// Selects keyframes based on temporal motion energy peaks and scene transitions.
    /// </summary>
    public static int[] SampleMotionKeyframes(
        IReadOnlyList<byte[]> frames,
        int targetKeyframes,
        float motionThreshold = 0.05f)
    {
        if (frames.Count == 0) return Array.Empty<int>();
        if (frames.Count <= targetKeyframes)
        {
            int[] all = new int[frames.Count];
            for (int i = 0; i < frames.Count; i++) all[i] = i;
            return all;
        }

        var keyframeIndices = new List<int> { 0 }; // Always include first frame

        for (int i = 1; i < frames.Count - 1 && keyframeIndices.Count < targetKeyframes - 1; i++)
        {
            float motion = ComputeMotionEnergy(frames[i - 1], frames[i]);
            if (motion >= motionThreshold)
            {
                keyframeIndices.Add(i);
            }
        }

        // Always include last frame
        if (!keyframeIndices.Contains(frames.Count - 1))
        {
            if (keyframeIndices.Count >= targetKeyframes)
            {
                keyframeIndices[^1] = frames.Count - 1;
            }
            else
            {
                keyframeIndices.Add(frames.Count - 1);
            }
        }

        // If threshold didn't capture enough frames, fill uniformly
        if (keyframeIndices.Count < targetKeyframes)
        {
            return SampleUniformIndices(frames.Count, targetKeyframes);
        }

        return keyframeIndices.ToArray();
    }
}
