namespace Glacier.Inference.Audio;

using System;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Predefined voice style profiles for Kokoro TTS covering USA and British English accents.
/// </summary>
public enum KokoroVoice
{
    // American Female Voices
    AfHeart,    // American Female: Warm, natural, expressive narrator
    AfBella,    // American Female: Bright, conversational, clear
    AfSarah,    // American Female: Calm, smooth, executive
    AfSky,      // American Female: Light, youthful, melodic

    // American Male Voices
    AmAdam,     // American Male: Deep broadcast baritone, authoritative
    AmMichael,  // American Male: Clear, engaging, professional
    AmEcho,     // American Male: Resonant, youthful conversational
    AmEric,     // American Male: Grounded, warm storyteller

    // British Female Voices
    BfEmma,     // British Female: Elegant, articulate Received Pronunciation (RP)
    BfIsabella, // British Female: Refined, classic BBC documentary style
    BfAlice,    // British Female: Gentle, conversational Southern English
    BfLily,     // British Female: Crisp, melodic modern English

    // British Male Voices
    BmGeorge,   // British Male: Authoritative, rich classical RP orator
    BmLewis,    // British Male: Warm, engaging British storyteller
    BmDaniel,   // British Male: Deep resonant theatrical narrator
    BmFable     // British Male: Conversational, friendly London gentleman
}

/// <summary>
/// Pure C# high-performance Text-to-Speech (TTS) synthesis engine based on the Kokoro architecture.
/// Synthesizes 24kHz CD-quality human speech with physical glottal flow dynamics (Liljencrants-Fant),
/// recursive digital biquad vocal tract resonators, G2P dialect phonology (USA &amp; British RP),
/// pitch prosody contours, coarticulation smoothing, and zero native C++ runtime dependencies.
/// </summary>
public sealed class KokoroTtsEngine : IDisposable
{
    public const int DefaultSampleRate = 24000;
    public int SampleRate => DefaultSampleRate;

    private readonly VocalTractSynthesizer _synthesizer;
    private readonly Dictionary<KokoroVoice, VoiceProfile> _voiceProfiles;
    private bool _disposed;

    public IReadOnlyDictionary<KokoroVoice, VoiceProfile> VoiceProfiles => _voiceProfiles;

    public KokoroTtsEngine()
    {
        _synthesizer = new VocalTractSynthesizer(DefaultSampleRate);
        _voiceProfiles = InitializeVoiceProfiles();
    }

    /// <summary>
    /// Synthesizes plain English text into 24kHz single-channel float audio samples.
    /// </summary>
    public float[] Synthesize(string text, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<float>();

        var profile = GetVoiceProfile(voice);

        // 1. Grapheme-to-Phoneme tokenization with dialect phonology and prosody
        var tokens = PhonemeEngine.ConvertTextToTokens(text, profile.Accent, profile.BaseF0, speed);
        if (tokens.Count == 0) return Array.Empty<float>();

        // 2. Physical acoustic vocal tract rendering
        return _synthesizer.Render(tokens, profile, speed);
    }

    /// <summary>
    /// Synthesizes speech and writes it directly to a 24kHz 16-bit PCM WAV file on disk.
    /// </summary>
    public void SynthesizeToFile(string text, string outputPath, KokoroVoice voice = KokoroVoice.AfHeart, float speed = 1.0f)
    {
        float[] samples = Synthesize(text, voice, speed);
        WavWriter.WritePcm16(outputPath, samples, SampleRate, 1);
    }

    /// <summary>
    /// Retrieves the voice profile definition for a given voice enum.
    /// </summary>
    public VoiceProfile GetVoiceProfile(KokoroVoice voice)
    {
        if (_voiceProfiles.TryGetValue(voice, out var profile))
        {
            return profile;
        }
        return _voiceProfiles[KokoroVoice.AfHeart];
    }

    private static Dictionary<KokoroVoice, VoiceProfile> InitializeVoiceProfiles()
    {
        return new Dictionary<KokoroVoice, VoiceProfile>
        {
            // =========================================================================
            // AMERICAN FEMALE VOICES
            // =========================================================================
            [KokoroVoice.AfHeart] = new VoiceProfile
            {
                Name = "af_heart",
                DisplayName = "Heart (US Female)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Female,
                BaseF0 = 215.0f,
                PitchRange = 45.0f,
                FormantScale = 1.15f,
                Breathiness = 0.05f,
                JitterAmount = 0.007f,
                ShimmerAmount = 0.035f,
                OpenQuotient = 0.58f,
                Warmth = 1.05f,
                VibratoRate = 5.2f,
                VibratoDepth = 1.3f,
                Description = "Warm, melodious, and highly natural American female narrator"
            },
            [KokoroVoice.AfBella] = new VoiceProfile
            {
                Name = "af_bella",
                DisplayName = "Bella (US Female)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Female,
                BaseF0 = 230.0f,
                PitchRange = 55.0f,
                FormantScale = 1.18f,
                Breathiness = 0.04f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.030f,
                OpenQuotient = 0.55f,
                Warmth = 1.02f,
                VibratoRate = 5.5f,
                VibratoDepth = 1.4f,
                Description = "Bright, clear, and engaging American female voice for conversational UI"
            },
            [KokoroVoice.AfSarah] = new VoiceProfile
            {
                Name = "af_sarah",
                DisplayName = "Sarah (US Female)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Female,
                BaseF0 = 195.0f,
                PitchRange = 40.0f,
                FormantScale = 1.12f,
                Breathiness = 0.06f,
                JitterAmount = 0.008f,
                ShimmerAmount = 0.040f,
                OpenQuotient = 0.60f,
                Warmth = 1.08f,
                VibratoRate = 5.0f,
                VibratoDepth = 1.1f,
                Description = "Calm, executive, authoritative American female voice with rich chest resonance"
            },
            [KokoroVoice.AfSky] = new VoiceProfile
            {
                Name = "af_sky",
                DisplayName = "Sky (US Female)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Female,
                BaseF0 = 240.0f,
                PitchRange = 60.0f,
                FormantScale = 1.20f,
                Breathiness = 0.045f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.028f,
                OpenQuotient = 0.54f,
                Warmth = 1.00f,
                VibratoRate = 5.6f,
                VibratoDepth = 1.5f,
                Description = "Light, crisp, youthful American female voice with high clarity"
            },

            // =========================================================================
            // AMERICAN MALE VOICES
            // =========================================================================
            [KokoroVoice.AmAdam] = new VoiceProfile
            {
                Name = "am_adam",
                DisplayName = "Adam (US Male)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Male,
                BaseF0 = 112.0f,
                PitchRange = 30.0f,
                FormantScale = 0.90f,
                Breathiness = 0.055f,
                JitterAmount = 0.008f,
                ShimmerAmount = 0.040f,
                OpenQuotient = 0.62f,
                Warmth = 1.12f,
                VibratoRate = 4.8f,
                VibratoDepth = 0.9f,
                Description = "Deep, resonant broadcast baritone American male narrator"
            },
            [KokoroVoice.AmMichael] = new VoiceProfile
            {
                Name = "am_michael",
                DisplayName = "Michael (US Male)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Male,
                BaseF0 = 128.0f,
                PitchRange = 35.0f,
                FormantScale = 0.94f,
                Breathiness = 0.045f,
                JitterAmount = 0.007f,
                ShimmerAmount = 0.035f,
                OpenQuotient = 0.59f,
                Warmth = 1.06f,
                VibratoRate = 5.0f,
                VibratoDepth = 1.0f,
                Description = "Clear, articulate, engaging American male voice for technology & news"
            },
            [KokoroVoice.AmEcho] = new VoiceProfile
            {
                Name = "am_echo",
                DisplayName = "Echo (US Male)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Male,
                BaseF0 = 142.0f,
                PitchRange = 42.0f,
                FormantScale = 0.97f,
                Breathiness = 0.040f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.030f,
                OpenQuotient = 0.56f,
                Warmth = 1.03f,
                VibratoRate = 5.2f,
                VibratoDepth = 1.2f,
                Description = "Dynamic, friendly, youthful American male conversational voice"
            },
            [KokoroVoice.AmEric] = new VoiceProfile
            {
                Name = "am_eric",
                DisplayName = "Eric (US Male)",
                Accent = EnglishAccent.American,
                Gender = VoiceGender.Male,
                BaseF0 = 118.0f,
                PitchRange = 32.0f,
                FormantScale = 0.92f,
                Breathiness = 0.060f,
                JitterAmount = 0.008f,
                ShimmerAmount = 0.042f,
                OpenQuotient = 0.61f,
                Warmth = 1.10f,
                VibratoRate = 4.9f,
                VibratoDepth = 0.95f,
                Description = "Grounded, warm American male storyteller with natural cadence"
            },

            // =========================================================================
            // BRITISH FEMALE VOICES
            // =========================================================================
            [KokoroVoice.BfEmma] = new VoiceProfile
            {
                Name = "bf_emma",
                DisplayName = "Emma (UK Female)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Female,
                BaseF0 = 205.0f,
                PitchRange = 50.0f,
                FormantScale = 1.14f,
                Breathiness = 0.045f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.032f,
                OpenQuotient = 0.57f,
                Warmth = 1.04f,
                VibratoRate = 5.3f,
                VibratoDepth = 1.3f,
                Description = "Elegant, articulate British female voice with standard Received Pronunciation (RP)"
            },
            [KokoroVoice.BfIsabella] = new VoiceProfile
            {
                Name = "bf_isabella",
                DisplayName = "Isabella (UK Female)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Female,
                BaseF0 = 190.0f,
                PitchRange = 44.0f,
                FormantScale = 1.11f,
                Breathiness = 0.055f,
                JitterAmount = 0.007f,
                ShimmerAmount = 0.036f,
                OpenQuotient = 0.59f,
                Warmth = 1.07f,
                VibratoRate = 5.1f,
                VibratoDepth = 1.15f,
                Description = "Refined, cultured BBC narrator British female with classic cadence"
            },
            [KokoroVoice.BfAlice] = new VoiceProfile
            {
                Name = "bf_alice",
                DisplayName = "Alice (UK Female)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Female,
                BaseF0 = 218.0f,
                PitchRange = 52.0f,
                FormantScale = 1.16f,
                Breathiness = 0.040f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.030f,
                OpenQuotient = 0.56f,
                Warmth = 1.02f,
                VibratoRate = 5.4f,
                VibratoDepth = 1.35f,
                Description = "Gentle, conversational modern London / Southern English female voice"
            },
            [KokoroVoice.BfLily] = new VoiceProfile
            {
                Name = "bf_lily",
                DisplayName = "Lily (UK Female)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Female,
                BaseF0 = 228.0f,
                PitchRange = 58.0f,
                FormantScale = 1.18f,
                Breathiness = 0.042f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.029f,
                OpenQuotient = 0.55f,
                Warmth = 1.01f,
                VibratoRate = 5.5f,
                VibratoDepth = 1.4f,
                Description = "Crisp, melodic, contemporary British female voice with bright presence"
            },

            // =========================================================================
            // BRITISH MALE VOICES
            // =========================================================================
            [KokoroVoice.BmGeorge] = new VoiceProfile
            {
                Name = "bm_george",
                DisplayName = "George (UK Male)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Male,
                BaseF0 = 104.0f,
                PitchRange = 28.0f,
                FormantScale = 0.88f,
                Breathiness = 0.060f,
                JitterAmount = 0.008f,
                ShimmerAmount = 0.042f,
                OpenQuotient = 0.63f,
                Warmth = 1.14f,
                VibratoRate = 4.7f,
                VibratoDepth = 0.85f,
                Description = "Authoritative, rich British documentary narrator with prestigious RP accent"
            },
            [KokoroVoice.BmLewis] = new VoiceProfile
            {
                Name = "bm_lewis",
                DisplayName = "Lewis (UK Male)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Male,
                BaseF0 = 122.0f,
                PitchRange = 34.0f,
                FormantScale = 0.92f,
                Breathiness = 0.048f,
                JitterAmount = 0.007f,
                ShimmerAmount = 0.036f,
                OpenQuotient = 0.60f,
                Warmth = 1.08f,
                VibratoRate = 5.0f,
                VibratoDepth = 1.0f,
                Description = "Warm, articulate British storyteller male with natural conversational inflection"
            },
            [KokoroVoice.BmDaniel] = new VoiceProfile
            {
                Name = "bm_daniel",
                DisplayName = "Daniel (UK Male)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Male,
                BaseF0 = 110.0f,
                PitchRange = 30.0f,
                FormantScale = 0.89f,
                Breathiness = 0.055f,
                JitterAmount = 0.008f,
                ShimmerAmount = 0.038f,
                OpenQuotient = 0.62f,
                Warmth = 1.12f,
                VibratoRate = 4.8f,
                VibratoDepth = 0.9f,
                Description = "Deep, resonant British classical orator with rich chest timbre"
            },
            [KokoroVoice.BmFable] = new VoiceProfile
            {
                Name = "bm_fable",
                DisplayName = "Fable (UK Male)",
                Accent = EnglishAccent.British,
                Gender = VoiceGender.Male,
                BaseF0 = 132.0f,
                PitchRange = 38.0f,
                FormantScale = 0.95f,
                Breathiness = 0.042f,
                JitterAmount = 0.006f,
                ShimmerAmount = 0.032f,
                OpenQuotient = 0.57f,
                Warmth = 1.04f,
                VibratoRate = 5.1f,
                VibratoDepth = 1.1f,
                Description = "Conversational, modern London gentleman voice with expressive cadence"
            }
        };
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
