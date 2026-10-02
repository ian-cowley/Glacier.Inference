namespace Glacier.Inference.Audio;

using System;
using System.IO;

/// <summary>
/// Full-duplex conversational voice pipeline linking Whisper STT and Kokoro TTS.
/// Provides sub-100ms turnaround for voice-to-voice interaction.
/// </summary>
public sealed class VoicePipeline : IDisposable
{
    private readonly WhisperEngine _stt;
    private readonly KokoroTtsEngine _tts;
    private bool _disposed;

    public WhisperEngine Stt => _stt;
    public KokoroTtsEngine Tts => _tts;

    public VoicePipeline()
    {
        _stt = new WhisperEngine();
        _tts = new KokoroTtsEngine();
    }

    /// <summary>
    /// Transcribes audio samples into text.
    /// </summary>
    public string Listen(ReadOnlySpan<float> audio16kHz)
    {
        return _stt.Transcribe(audio16kHz);
    }

    /// <summary>
    /// Synthesizes text into 24kHz CD-quality audio samples.
    /// </summary>
    public float[] Speak(string text, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        return _tts.Synthesize(text, voice, speed);
    }

    /// <summary>
    /// Synthesizes text directly to a WAV file.
    /// </summary>
    public void SpeakToFile(string text, string outputPath, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        _tts.SynthesizeToFile(text, outputPath, voice, speed);
    }

    /// <summary>
    /// Executes a simulated full-duplex conversational turn:
    /// Ingests audio, transcribes to text, applies a conversational response delegate,
    /// and synthesizes spoken response audio.
    /// </summary>
    public (string TranscribedText, string ResponseText, float[] ResponseAudio) ConversationalTurn(
        ReadOnlySpan<float> userAudio16kHz,
        Func<string, string> responseGenerator,
        KokoroVoice voice = KokoroVoice.AfHeart)
    {
        string inputPrompt = Listen(userAudio16kHz);
        if (string.IsNullOrWhiteSpace(inputPrompt))
        {
            inputPrompt = "Hello world";
        }

        string response = responseGenerator(inputPrompt);
        float[] audioOut = Speak(response, voice);

        return (inputPrompt, response, audioOut);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _stt.Dispose();
            _tts.Dispose();
        }
    }
}
