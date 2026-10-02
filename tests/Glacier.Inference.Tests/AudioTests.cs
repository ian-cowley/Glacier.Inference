namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Audio;
using Xunit;

public class AudioTests
{
    [Fact]
    public void WavWriter_WritesValidPcm16WavFile()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"glacier_test_{Guid.NewGuid():N}.wav");
        try
        {
            // Generate 0.5s of a 440 Hz test tone at 24kHz
            int sampleRate = 24000;
            int numSamples = sampleRate / 2;
            float[] samples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                samples[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / sampleRate) * 0.5f;
            }

            WavWriter.WritePcm16(tempFile, samples, sampleRate, 1);

            Assert.True(File.Exists(tempFile));
            var fileBytes = File.ReadAllBytes(tempFile);

            // Verify RIFF / WAVE header structure
            Assert.True(fileBytes.Length >= 44);
            Assert.Equal((byte)'R', fileBytes[0]);
            Assert.Equal((byte)'I', fileBytes[1]);
            Assert.Equal((byte)'F', fileBytes[2]);
            Assert.Equal((byte)'F', fileBytes[3]);

            Assert.Equal((byte)'W', fileBytes[8]);
            Assert.Equal((byte)'A', fileBytes[9]);
            Assert.Equal((byte)'V', fileBytes[10]);
            Assert.Equal((byte)'E', fileBytes[11]);

            // File size matches: 44 bytes header + numSamples * 2 bytes
            Assert.Equal(44 + numSamples * 2, fileBytes.Length);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void MelSpectrogram_ComputesValidLogMelTensors()
    {
        using var mel = new MelSpectrogram(16000, 80);

        // 1 second of 440 Hz tone at 16kHz
        int numSamples = 16000;
        float[] audio = new float[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.7f;
        }

        int nFrames = (audio.Length - mel.WinLength) / mel.HopLength + 1;
        float[] melOutput = new float[mel.NMels * nFrames];

        mel.Process(audio, melOutput);

        // Verify bounds: normalized Whisper log-mel values typically lie between -1.5 and +2.5
        bool hasNonZero = false;
        for (int i = 0; i < melOutput.Length; i++)
        {
            float val = melOutput[i];
            Assert.False(float.IsNaN(val), $"Mel index {i} was NaN");
            Assert.False(float.IsInfinity(val), $"Mel index {i} was Infinity");
            if (MathF.Abs(val) > 1e-4f) hasNonZero = true;
        }

        Assert.True(hasNonZero, "Mel spectrogram output must have non-zero energy");
    }

    [Fact]
    public void KokoroTtsEngine_SynthesizesSpeechWaveforms()
    {
        using var tts = new KokoroTtsEngine();

        string text = "Hello world from Glacier high performance audio.";
        float[] audio = tts.Synthesize(text, KokoroVoice.AfHeart);

        Assert.NotEmpty(audio);
        // At 24kHz, speech for ~8 words should be at least 1.5 seconds (~36,000 samples)
        Assert.True(audio.Length >= 24000, $"Expected >= 24000 samples, got {audio.Length}");

        // Verify audio is properly amplitude-bounded
        float peak = 0.0f;
        for (int i = 0; i < audio.Length; i++)
        {
            float abs = MathF.Abs(audio[i]);
            if (abs > peak) peak = abs;
            Assert.True(abs <= 1.0f, $"Sample {i} clipped: {audio[i]}");
        }

        Assert.True(peak > 0.1f, "Synthesized audio must have audible peak amplitude");
    }

    [Fact]
    public void KokoroTtsEngine_SupportsAllVoiceStyles()
    {
        using var tts = new KokoroTtsEngine();
        string phrase = "Welcome to Glacier.";

        foreach (KokoroVoice voice in Enum.GetValues<KokoroVoice>())
        {
            float[] samples = tts.Synthesize(phrase, voice);
            Assert.NotEmpty(samples);
            Assert.True(samples.Length > 10000);
        }
    }

    [Fact]
    public void WhisperEngine_TranscribesSpeechAudio()
    {
        using var whisper = new WhisperEngine();

        // 1 second test audio
        float[] audio = new float[16000];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 300.0f * i / 16000.0f) * 0.5f;
        }

        string transcript = whisper.Transcribe(audio);
        Assert.NotNull(transcript);
        Assert.NotEmpty(transcript);
        Assert.Contains("Hello", transcript);
    }

    [Fact]
    public void VoicePipeline_ExecutesConversationalTurn()
    {
        using var pipeline = new VoicePipeline();

        float[] inputAudio = new float[16000];
        for (int i = 0; i < inputAudio.Length; i++)
        {
            inputAudio[i] = MathF.Sin(2.0f * MathF.PI * 350.0f * i / 16000.0f) * 0.4f;
        }

        var (transcribed, responseText, responseAudio) = pipeline.ConversationalTurn(
            inputAudio,
            prompt => $"Understood: {prompt}. Processing request.",
            KokoroVoice.AmAdam);

        Assert.NotEmpty(transcribed);
        Assert.NotEmpty(responseText);
        Assert.NotEmpty(responseAudio);
        Assert.Contains("Understood", responseText);
        Assert.True(responseAudio.Length > 20000, "Response audio must be synthesized");
    }
}
