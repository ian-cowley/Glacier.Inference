namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Format;

/// <summary>
/// Master neural Text-to-Speech (TTS) engine based on KittenTTS / StyleTTS 2 distilled architecture.
/// 100% pure C# .NET 10 bare-metal execution.
/// Runs ALBERT transformer, dual-path Text Encoder, Duration/F0/N Predictor, and HiFi-GAN Generator
/// with AVX-512/AVX2 SIMD acceleration. Zero ONNX, zero Python, zero external native DLL dependencies.
/// </summary>
public sealed class KittenTtsEngine : IDisposable
{
    private readonly SafetensorsFile _modelFile;
    private readonly KittenVoiceStore _voiceStore;
    private readonly KittenConfig _config;

    private readonly AlbertEncoder _albert;
    private readonly KittenTextEncoder _textEncoder;
    private readonly KittenPredictor _predictor;
    private readonly KittenDecoder _decoder;

    private bool _disposed;

    public int SampleRate => _config.SampleRate;
    public IReadOnlyCollection<string> AvailableVoices => _voiceStore.AvailableVoices;

    public KittenTtsEngine(string modelPath, string voicesPath, string? configPath = null)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"KittenTTS model file not found: {modelPath}", modelPath);
        if (!File.Exists(voicesPath))
            throw new FileNotFoundException($"KittenTTS voices file not found: {voicesPath}", voicesPath);

        _config = !string.IsNullOrEmpty(configPath) && File.Exists(configPath)
            ? KittenConfig.Load(configPath)
            : KittenConfig.Default;

        _modelFile = SafetensorsFile.Open(modelPath);
        _voiceStore = KittenVoiceStore.Load(voicesPath);

        _albert = new AlbertEncoder(_modelFile);
        _textEncoder = new KittenTextEncoder(_modelFile);
        _predictor = new KittenPredictor(_modelFile);
        _decoder = new KittenDecoder(_modelFile, _config);
    }

    /// <summary>
    /// Attempts to create KittenTtsEngine by searching default locations.
    /// </summary>
    public static KittenTtsEngine CreateDefault(string? baseDir = null)
    {
        baseDir ??= AppContext.BaseDirectory;
        string[] searchPaths =
        {
            baseDir,
            Path.Combine(baseDir, "models", "kitten"),
            Path.Combine(baseDir, "..", "..", "..", "..", "models", "kitten"),
            Path.Combine(Directory.GetCurrentDirectory(), "models", "kitten"),
            Path.Combine(Directory.GetCurrentDirectory(), "..", "models", "kitten")
        };

        string? foundModel = null;
        string? foundVoices = null;
        string? foundConfig = null;

        foreach (var dir in searchPaths)
        {
            try
            {
                var full = Path.GetFullPath(dir);
                string m = Path.Combine(full, "kitten-nano.safetensors");
                string v = Path.Combine(full, "kitten-voices.safetensors");
                string c = Path.Combine(full, "config.json");

                if (File.Exists(m) && File.Exists(v))
                {
                    foundModel = m;
                    foundVoices = v;
                    if (File.Exists(c)) foundConfig = c;
                    break;
                }
            }
            catch { }
        }

        if (foundModel == null || foundVoices == null)
        {
            throw new FileNotFoundException(
                "Could not locate KittenTTS model files (kitten-nano.safetensors, kitten-voices.safetensors). " +
                "Ensure they exist in models/kitten/.");
        }

        return new KittenTtsEngine(foundModel, foundVoices, foundConfig);
    }

    /// <summary>
    /// Registers a custom or scraped voice style vector [256] or matrix [400, 256].
    /// </summary>
    public void RegisterVoice(string voiceName, float[] styleData)
    {
        _voiceStore.RegisterVoice(voiceName, styleData);
    }

    /// <summary>
    /// Synthesizes high-fidelity 24kHz audio from input English text.
    /// </summary>
    /// <param name="text">Input English text.</param>
    /// <param name="voice">Voice name (e.g. 'bella', 'bruno', 'hugo', 'jasper', 'kiki', 'leo', 'luna', 'rosie').</param>
    /// <param name="speed">Speaking speed multiplier (1.0 = default rate).</param>
    /// <returns>Array of 32-bit floating-point audio samples at 24,000 Hz.</returns>
    public float[] Synthesize(string text, string voice = "bella", float speed = 1.0f)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<float>();

        // 1. Text normalization & ensure terminal punctuation
        string normalized = KittenTextPreprocess.Normalize(EnsurePunctuation(text));

        // 2. G2P conversion (English -> IPA phonemes)
        string phonemes = KittenPhonemizer.Phonemize(normalized);

        // 3. Map phonemes to token IDs
        int[] tokenIds = KittenPhonemeMap.Map(phonemes);
        if (tokenIds.Length == 0)
            return Array.Empty<float>();

        // 4. Retrieve voice style vector [256] using text length (matching reference implementation)
        var style = _voiceStore.GetStyle(voice, normalized.Length);

        // 5. ALBERT Transformer forward pass -> [seqLen, 128]
        float[] bertOut = _albert.Forward(tokenIds);

        // 6. Text Encoder forward pass -> (lstmFeatures [seqLen, 256], cnnFeatures [128, seqLen])
        var (lstmFeatures, cnnFeatures) = _textEncoder.Forward(bertOut, tokenIds, style);

        // 7. Predictor forward pass -> (durations, sharedLstmOut, f0, nAmp) with calibrated speed prior
        float effectiveSpeed = speed * _config.GetSpeedPrior(voice);
        var (durations, sharedLstmOut, f0, nAmp) = _predictor.Forward(lstmFeatures, style, effectiveSpeed);

        // 8. Duration expansion for CNN features: [128, seqLen] -> [128, totalFrames]
        int totalFrames = 0;
        foreach (int d in durations) totalFrames += d;
        if (totalFrames == 0) totalFrames = 1;

        float[] expandedCnn = new float[128 * totalFrames];
        int frameIdx = 0;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int d = durations[s];
            for (int r = 0; r < d; r++)
            {
                int outT = frameIdx++;
                for (int c = 0; c < 128; c++)
                {
                    expandedCnn[c * totalFrames + outT] = cnnFeatures[c * tokenIds.Length + s];
                }
            }
        }

        // 9. Decoder forward pass -> 24kHz audio waveform (expandedCnn feeds both encode and asr_res)
        return _decoder.Forward(expandedCnn, expandedCnn, f0, nAmp, style);
    }

    /// <summary>
    /// Synthesizes text to a 16-bit PCM WAV file.
    /// </summary>
    public void SynthesizeToFile(string text, string outputPath, string voice = "bella", float speed = 1.0f)
    {
        var samples = Synthesize(text, voice, speed);
        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        WavWriter.WritePcm16(outputPath, samples, SampleRate);
    }

    /// <summary>
    /// Synthesizes text and returns WAV-formatted file bytes.
    /// </summary>
    public byte[] SynthesizeToWavBytes(string text, string voice = "bella", float speed = 1.0f)
    {
        var samples = Synthesize(text, voice, speed);
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            int bytesPerSample = 2;
            int dataChunkSize = samples.Length * bytesPerSample;
            int fileSize = 36 + dataChunkSize;

            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(fileSize);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)1); // mono
            writer.Write(SampleRate);
            writer.Write(SampleRate * bytesPerSample);
            writer.Write((short)bytesPerSample);
            writer.Write((short)16);

            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(dataChunkSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float s = Math.Clamp(samples[i], -1.0f, 1.0f);
                short pcm = (short)Math.Round(s * 32767.0f);
                writer.Write(pcm);
            }
        }
        return ms.ToArray();
    }

    private static string EnsurePunctuation(string text)
    {
        string trimmed = text.Trim();
        if (trimmed.Length == 0) return trimmed;
        char last = trimmed[^1];
        if (last != '.' && last != '!' && last != '?' && last != ',' && last != ';' && last != ':')
        {
            return trimmed + ".";
        }
        return trimmed;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _voiceStore.Dispose();
            _modelFile.Dispose();
            _disposed = true;
        }
    }
}
