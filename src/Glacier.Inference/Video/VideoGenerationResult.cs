namespace Glacier.Inference.Video;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Image;

/// <summary>
/// Result container for generated video containing frame pixel buffers and multi-format exporters.
/// </summary>
public sealed class VideoGenerationResult
{
    public IReadOnlyList<byte[]> Frames { get; }
    public int Width { get; }
    public int Height { get; }
    public int NumFrames => Frames.Count;
    public int Fps { get; }
    public long ElapsedMilliseconds { get; }
    public string Prompt { get; }
    public CameraMotion CameraMotion { get; }

    public VideoGenerationResult(
        IReadOnlyList<byte[]> frames,
        int width,
        int height,
        int fps,
        long elapsedMilliseconds,
        string prompt,
        CameraMotion motion)
    {
        Frames = frames;
        Width = width;
        Height = height;
        Fps = fps;
        ElapsedMilliseconds = elapsedMilliseconds;
        Prompt = prompt;
        CameraMotion = motion;
    }

    /// <summary>
    /// Exports the generated video sequence to an Animated PNG (APNG) file.
    /// </summary>
    public void SaveApng(string path)
    {
        ApngWriter.SaveApng(path, Frames, Width, Height, Fps);
    }

    /// <summary>
    /// Exports the generated video sequence to an Animated GIF89a file.
    /// </summary>
    public void SaveGif(string path)
    {
        GifWriter.SaveGif(path, Frames, Width, Height, Fps);
    }

    /// <summary>
    /// Exports the generated video sequence to a standard RIFF/AVI container file.
    /// </summary>
    public void SaveAvi(string path)
    {
        AviWriter.SaveAvi(path, Frames, Width, Height, Fps);
    }

    /// <summary>
    /// Exports each video frame as a separate PNG or BMP file in the target directory.
    /// </summary>
    public void SaveFrames(string directory, string format = "png")
    {
        Directory.CreateDirectory(directory);
        for (int i = 0; i < Frames.Count; i++)
        {
            string ext = format.TrimStart('.').ToLowerInvariant();
            string fileName = $"frame_{i + 1:D4}.{ext}";
            string fullPath = Path.Combine(directory, fileName);

            if (ext == "bmp")
            {
                BmpWriter.SaveBmp24(fullPath, Frames[i], Width, Height);
            }
            else
            {
                PngWriter.SavePng24(fullPath, Frames[i], Width, Height);
            }
        }
    }

    /// <summary>
    /// Automatically saves the video based on the file extension (.apng, .png, .gif, .avi).
    /// </summary>
    public void Save(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        switch (ext)
        {
            case ".gif":
                SaveGif(path);
                break;
            case ".avi":
                SaveAvi(path);
                break;
            case ".apng":
            case ".png":
            default:
                SaveApng(path);
                break;
        }
    }
}
