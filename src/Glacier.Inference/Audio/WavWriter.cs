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

    /// <summary>
    /// Parses a standard RIFF WAVE file (supporting arbitrary intermediate chunks such as LIST/JUNK/fact,
    /// 16/24/32-bit PCM, 32-bit IEEE float, WAVE_FORMAT_EXTENSIBLE, and multi-channel downmixing) into mono float samples.
    /// </summary>
    public static float[] ReadMonoFloat(ReadOnlySpan<byte> wavBytes, out int sampleRate)
    {
        if (wavBytes.Length < 12 ||
            wavBytes[0] != (byte)'R' || wavBytes[1] != (byte)'I' || wavBytes[2] != (byte)'F' || wavBytes[3] != (byte)'F' ||
            wavBytes[8] != (byte)'W' || wavBytes[9] != (byte)'A' || wavBytes[10] != (byte)'V' || wavBytes[11] != (byte)'E')
        {
            throw new InvalidDataException("Invalid RIFF WAVE header.");
        }

        bool foundFmt = false;
        ushort audioFormat = 0;
        ushort channels = 0;
        sampleRate = 0;
        ushort bitsPerSample = 0;
        int dataOffset = -1;
        int dataLength = 0;

        int pos = 12;
        while (pos + 8 <= wavBytes.Length)
        {
            uint chunkId = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(wavBytes.Slice(pos, 4));
            int chunkSize = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(wavBytes.Slice(pos + 4, 4));
            if (chunkSize < 0)
            {
                throw new InvalidDataException("Corrupt RIFF chunk size.");
            }

            int payloadStart = pos + 8;
            int available = wavBytes.Length - payloadStart;
            int effectiveSize = Math.Min(chunkSize, available);

            // 'fmt ' in little-endian: 'f' (0x66), 'm' (0x6D), 't' (0x74), ' ' (0x20) => 0x20746D66
            if (chunkId == 0x20746D66u)
            {
                if (effectiveSize < 16)
                {
                    throw new InvalidDataException("RIFF 'fmt ' chunk is truncated.");
                }
                var fmt = wavBytes.Slice(payloadStart, effectiveSize);
                audioFormat = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(fmt.Slice(0, 2));
                channels = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(fmt.Slice(2, 2));
                sampleRate = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(fmt.Slice(4, 4));
                bitsPerSample = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(fmt.Slice(14, 2));

                // WAVE_FORMAT_EXTENSIBLE (0xFFFE): subformat GUID first 2 bytes at offset 24 store underlying format tag (1=PCM, 3=IEEE Float)
                if (audioFormat == 0xFFFE && effectiveSize >= 26)
                {
                    audioFormat = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(fmt.Slice(24, 2));
                }
                foundFmt = true;
            }
            // 'data' in little-endian: 'd' (0x64), 'a' (0x61), 't' (0x74), 'a' (0x61) => 0x61746164
            else if (chunkId == 0x61746164u)
            {
                dataOffset = payloadStart;
                dataLength = effectiveSize;
                break;
            }

            long nextPos = (long)payloadStart + chunkSize + (chunkSize & 1); // RIFF word alignment
            if (nextPos > wavBytes.Length || nextPos <= pos) break;
            pos = (int)nextPos;
        }

        if (!foundFmt || channels == 0 || sampleRate <= 0)
        {
            throw new InvalidDataException("RIFF WAVE file is missing a valid 'fmt ' chunk.");
        }
        if (dataOffset < 0)
        {
            throw new InvalidDataException("RIFF WAVE file is missing a 'data' chunk.");
        }

        int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample <= 0)
        {
            throw new NotSupportedException($"Unsupported WAV bitsPerSample: {bitsPerSample}.");
        }

        int frameStride = bytesPerSample * channels;
        int frameCount = dataLength / frameStride;
        var mono = new float[frameCount];
        var data = wavBytes.Slice(dataOffset, frameCount * frameStride);

        for (int f = 0; f < frameCount; f++)
        {
            float sum = 0f;
            int frameBase = f * frameStride;
            for (int c = 0; c < channels; c++)
            {
                int off = frameBase + c * bytesPerSample;
                float sample = (audioFormat, bitsPerSample) switch
                {
                    (1, 8) => (data[off] - 128) / 128.0f,
                    (1, 16) => System.Buffers.Binary.BinaryPrimitives.ReadInt16LittleEndian(data.Slice(off, 2)) / 32768.0f,
                    (1, 24) => (((sbyte)data[off + 2] << 16) | (data[off + 1] << 8) | data[off]) / 8388608.0f,
                    (1, 32) => System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(data.Slice(off, 4)) / 2147483648.0f,
                    (3, 32) => System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(data.Slice(off, 4)),
                    _ => throw new NotSupportedException($"Unsupported WAV encoding (format={audioFormat}, bits={bitsPerSample}).")
                };
                sum += sample;
            }
            mono[f] = sum / channels;
        }

        return mono;
    }
}
