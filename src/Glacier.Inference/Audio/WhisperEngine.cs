namespace Glacier.Inference.Audio;

using System;
using System.Diagnostics.CodeAnalysis;

/// <summary>
/// Experimental Speech-to-Text (STT) scaffold for the Whisper encoder-decoder architecture.
/// Pretrained GGUF Whisper weight loading is not yet implemented; weightless stub transcription is disabled.
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed class WhisperEngine : IDisposable
{
    public const int DefaultSampleRate = 16000;
    public const int MelBins = 80;
    public const int EncoderDim = 512;
    public const int DecoderDim = 512;
    public const int EncoderHeads = 8;
    public const int DecoderHeads = 8;
    public const int HeadDim = EncoderDim / EncoderHeads; // 64

    // Special Whisper tokens
    public const int TokenStartOfTranscript = 50258;
    public const int TokenEnglish = 50259;
    public const int TokenTranscribe = 50359;
    public const int TokenNoTimestamps = 50363;
    public const int TokenEndOfTranscript = 50257;

    private readonly MelSpectrogram _melDsp;
    private readonly int _encoderLayers;
    private readonly int _decoderLayers;
    private bool _disposed;

    public int EncoderLayers => _encoderLayers;
    public int DecoderLayers => _decoderLayers;

    public WhisperEngine(int encoderLayers = 4, int decoderLayers = 4)
    {
        _encoderLayers = encoderLayers;
        _decoderLayers = decoderLayers;
        _melDsp = new MelSpectrogram(DefaultSampleRate, MelBins);
    }

    /// <summary>
    /// Transcribes 16kHz audio samples into text once Whisper GGUF weights are loaded.
    /// Currently throws <see cref="NotSupportedException"/> because weightless stub transcription is disabled.
    /// </summary>
    public string Transcribe(ReadOnlySpan<float> audio)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _ = audio;
        throw new NotSupportedException(
            "WhisperEngine requires GGUF Whisper model weights which are not yet loaded. Weightless stub transcription has been disabled.");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _melDsp.Dispose();
        }
    }
}
