namespace Glacier.Inference.Audio;

using System;
using System.IO;
using System.Text;

/// <summary>
/// Pure C# high-performance WAV audio encoder for 16-bit PCM and 32-bit IEEE float formats.
/// Zero external dependencies, Native AOT compatible.
/// </summary>
public static class WavWriter
{
    /// <summary>
    /// Writes 32-bit floating-point audio samples (-1.0f to +1.0f) as a 16-bit PCM WAV file.
    /// </summary>
    public static void WritePcm16(string filePath, ReadOnlySpan<float> samples, int sampleRate = 24000, short channels = 1)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);

        int bytesPerSample = 2; // 16-bit PCM
        int dataChunkSize = samples.Length * bytesPerSample * channels;
        int fileSize = 36 + dataChunkSize;

        // RIFF header
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(fileSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        // 'fmt ' chunk
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16); // Chunk size (16 for PCM)
        writer.Write((short)1); // Audio format: 1 = PCM
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * bytesPerSample); // Byte rate
        writer.Write((short)(channels * bytesPerSample)); // Block align
        writer.Write((short)(bytesPerSample * 8)); // Bits per sample

        // 'data' chunk
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataChunkSize);

        // Convert float (-1.0 to +1.0) to 16-bit signed PCM with clipping
        for (int i = 0; i < samples.Length; i++)
        {
            float s = Math.Clamp(samples[i], -1.0f, 1.0f);
            short pcm = (short)Math.Round(s * 32767.0f);
            writer.Write(pcm);
        }
    }

    /// <summary>
    /// Writes 32-bit floating-point audio samples as a 32-bit IEEE float WAV file.
    /// </summary>
    public static void WriteFloat32(string filePath, ReadOnlySpan<float> samples, int sampleRate = 24000, short channels = 1)
    {
        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);

        int bytesPerSample = 4; // 32-bit float
        int dataChunkSize = samples.Length * bytesPerSample * channels;
        int fileSize = 36 + dataChunkSize;

        // RIFF header
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(fileSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        // 'fmt ' chunk
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)3); // Audio format: 3 = IEEE float
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * bytesPerSample);
        writer.Write((short)(channels * bytesPerSample));
        writer.Write((short)(bytesPerSample * 8));

        // 'data' chunk
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataChunkSize);

        for (int i = 0; i < samples.Length; i++)
        {
            writer.Write(samples[i]);
        }
    }

    /// <summary>
    /// Encodes float samples into an in-memory byte buffer containing the complete 16-bit PCM WAV.
    /// </summary>
    public static byte[] ToPcm16Bytes(ReadOnlySpan<float> samples, int sampleRate = 24000, short channels = 1)
    {
        int bytesPerSample = 2;
        int dataChunkSize = samples.Length * bytesPerSample * channels;
        byte[] buffer = new byte[44 + dataChunkSize];

        using var ms = new MemoryStream(buffer);
        using var writer = new BinaryWriter(ms);

        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataChunkSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write(channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * bytesPerSample);
        writer.Write((short)(channels * bytesPerSample));
        writer.Write((short)(bytesPerSample * 8));

        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataChunkSize);

        for (int i = 0; i < samples.Length; i++)
        {
            float s = Math.Clamp(samples[i], -1.0f, 1.0f);
            short pcm = (short)Math.Round(s * 32767.0f);
            writer.Write(pcm);
        }

        return buffer;
    }
}
