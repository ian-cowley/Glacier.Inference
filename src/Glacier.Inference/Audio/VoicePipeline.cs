namespace Glacier.Inference.Audio;

using System;
using System.Diagnostics.CodeAnalysis;

/// <summary>
/// Experimental conversational voice pipeline linking Whisper STT (requires pretrained weights)
/// and procedural formant TTS (<see cref="KokoroTtsEngine"/>).
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed class VoicePipeline : IDisposable
{
#pragma warning disable GLACIER_EXP001
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
#pragma warning restore GLACIER_EXP001

    /// <summary>
    /// Synthesizes text into 24kHz audio samples via procedural formant synthesis.
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
    /// Executes a conversational turn: ingests audio, transcribes to text, applies a response delegate,
    /// and synthesizes spoken response audio.
    /// </summary>
    public (string TranscribedText, string ResponseText, float[] ResponseAudio) ConversationalTurn(
        ReadOnlySpan<float> userAudio16kHz,
        Func<string, string> responseGenerator,
        KokoroVoice voice = KokoroVoice.AfHeart)
    {
        string inputPrompt = Listen(userAudio16kHz);
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
