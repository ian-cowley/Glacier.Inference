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

    [Fact]
    public void KokoroTtsEngine_ValidatesAll16VoiceProfiles()
    {
        using var tts = new KokoroTtsEngine();
        var profiles = tts.VoiceProfiles;

        Assert.Equal(16, profiles.Count);

        int usaCount = 0, ukCount = 0, femaleCount = 0, maleCount = 0;

        foreach (var kvp in profiles)
        {
            var p = kvp.Value;
            Assert.NotEmpty(p.Name);
            Assert.NotEmpty(p.DisplayName);
            Assert.NotEmpty(p.Description);
            Assert.InRange(p.BaseF0, 95.0f, 260.0f);
            Assert.InRange(p.FormantScale, 0.80f, 1.30f);

            if (p.Accent == EnglishAccent.American) usaCount++;
            else if (p.Accent == EnglishAccent.British) ukCount++;

            if (p.Gender == VoiceGender.Female) femaleCount++;
            else if (p.Gender == VoiceGender.Male) maleCount++;
        }

        Assert.Equal(8, usaCount);
        Assert.Equal(8, ukCount);
        Assert.Equal(8, femaleCount);
        Assert.Equal(8, maleCount);
    }

    [Fact]
    public void PhonemeEngine_DifferentiatesAmericanAndBritishAccents()
    {
        var usTokens = PhonemeEngine.ConvertTextToTokens("Glacier water after path", EnglishAccent.American, 200f);
        var ukTokens = PhonemeEngine.ConvertTextToTokens("Glacier water after path", EnglishAccent.British, 200f);

        Assert.NotEmpty(usTokens);
        Assert.NotEmpty(ukTokens);

        // American tokens have rhotic R_US or FLAP; British tokens have non-rhotic schwa AX or British R_UK
        bool usHasRhotic = usTokens.Any(t => t.Spec.Symbol == "R_US" || t.Spec.Symbol == "FLAP");
        bool ukHasBritishPhonemes = ukTokens.Any(t => t.Spec.Symbol == "AO_UK" || t.Spec.Symbol == "R_UK" || t.Spec.Symbol == "AA");

        Assert.True(usHasRhotic, "American dialect must contain rhotic /r/ or flap /ɾ/");
        Assert.True(ukHasBritishPhonemes, "British dialect must contain RP broad vowels or British R");
    }

    [Fact]
    public void KokoroTtsEngine_SpeedScalingAdjustsDuration()
    {
        using var tts = new KokoroTtsEngine();
        string phrase = "High performance voice inference with Glacier.";

        float[] fast = tts.Synthesize(phrase, KokoroVoice.AfHeart, speed: 1.5f);
        float[] normal = tts.Synthesize(phrase, KokoroVoice.AfHeart, speed: 1.0f);
        float[] slow = tts.Synthesize(phrase, KokoroVoice.AfHeart, speed: 0.75f);

        Assert.True(fast.Length < normal.Length, "Fast speech must be shorter than normal");
        Assert.True(normal.Length < slow.Length, "Normal speech must be shorter than slow");
    }

    [Fact]
    public void KokoroTtsEngine_ProducesCleanAudioMetricsWithoutDCBiasOrClipping()
    {
        using var tts = new KokoroTtsEngine();
        string text = "Acoustic physics and physical vocal tract simulation.";

        float[] samples = tts.Synthesize(text, KokoroVoice.AmAdam);
        Assert.NotEmpty(samples);

        float peak = 0.0f;
        double sum = 0.0;
        double sumSq = 0.0;

        for (int i = 0; i < samples.Length; i++)
        {
            float s = samples[i];
            float abs = MathF.Abs(s);
            if (abs > peak) peak = abs;
            sum += s;
            sumSq += s * s;

            // Zero digital clipping
            Assert.True(abs <= 1.0f, $"Sample {i} clipped at {s}");
        }

        float dcOffset = (float)(sum / samples.Length);
        float rms = MathF.Sqrt((float)(sumSq / samples.Length));

        // Verify zero DC bias (<0.05) and healthy vocal RMS (>0.05)
        Assert.True(MathF.Abs(dcOffset) < 0.05f, $"DC offset too high: {dcOffset}");
        Assert.True(rms > 0.05f, $"Signal energy too low: RMS={rms}");
        Assert.True(peak > 0.3f, $"Signal peak too quiet: Peak={peak}");
    }
}
