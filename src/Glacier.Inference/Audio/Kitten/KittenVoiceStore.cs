namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Format;

/// <summary>
/// Voice style vector repository for KittenTTS / StyleTTS 2.
/// Loads pre-computed voice style matrices ([400, 256]) and custom curated voices.
/// Each row in the matrix corresponds to a sentence length bucket (0..399).
/// </summary>
public sealed class KittenVoiceStore : IDisposable
{
    private readonly Dictionary<string, float[]> _voices = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, float[]> _customVoices = new(StringComparer.OrdinalIgnoreCase);
    private SafetensorsFile? _file;

    public IReadOnlyCollection<string> AvailableVoices
    {
        get
        {
            var set = new HashSet<string>(_voices.Keys, StringComparer.OrdinalIgnoreCase);
            foreach (var custom in _customVoices.Keys) set.Add(custom);
            return set;
        }
    }

    public KittenVoiceStore()
    {
    }

    /// <summary>
    /// Loads the built-in voices from kitten-voices.safetensors.
    /// </summary>
    public static KittenVoiceStore Load(string filePath)
    {
        var store = new KittenVoiceStore();
        store._file = SafetensorsFile.Open(filePath);

        unsafe
        {
            foreach (var kvp in store._file.Tensors)
            {
                if (kvp.Value.Shape.Length == 2 && kvp.Value.Shape[0] == 400 && kvp.Value.Shape[1] == 256)
                {
                    float[] data = new float[400 * 256];
                    fixed (float* pDst = data)
                    {
                        Buffer.MemoryCopy(store._file.GetTensorPointer(kvp.Key), pDst, (long)kvp.Value.ByteSize, (long)kvp.Value.ByteSize);
                    }
                    store._voices[kvp.Key] = data;
                }
            }
        }

        return store;
    }

    /// <summary>
    /// Registers a custom voice style vector [256] or matrix [400, 256].
    /// </summary>
    public void RegisterVoice(string name, float[] styleData)
    {
        if (styleData.Length != 256 && styleData.Length != 400 * 256)
            throw new ArgumentException("Style vector must have length 256 or 400*256", nameof(styleData));

        _customVoices[name] = styleData;
    }

    /// <summary>
    /// Retrieves a 256-dimensional style vector for a given voice and text length.
    /// </summary>
    public ReadOnlySpan<float> GetStyle(string voiceName, int textLength)
    {
        if (_customVoices.TryGetValue(voiceName, out var customData))
        {
            if (customData.Length == 256)
                return customData;
            int bucket = Math.Clamp(textLength, 0, 399);
            return new ReadOnlySpan<float>(customData, bucket * 256, 256);
        }

        if (_voices.TryGetValue(voiceName, out var voiceData))
        {
            int bucket = Math.Clamp(textLength, 0, 399);
            return new ReadOnlySpan<float>(voiceData, bucket * 256, 256);
        }

        // Fallback to first available voice if name not found
        foreach (var v in _voices.Values)
        {
            int bucket = Math.Clamp(textLength, 0, 399);
            return new ReadOnlySpan<float>(v, bucket * 256, 256);
        }

        throw new InvalidOperationException($"Voice '{voiceName}' not found and no default voices available.");
    }

    public void Dispose()
    {
        _file?.Dispose();
        _file = null;
    }
}
