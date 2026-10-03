namespace Glacier.Inference.Audio;

using System;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// English accent dialect classifications supported by the speech synthesis engine.
/// </summary>
public enum EnglishAccent
{
    American,
    British
}

/// <summary>
/// Biological vocal tract / gender resonance profile.
/// </summary>
public enum VoiceGender
{
    Female,
    Male
}

/// <summary>
/// Acoustic specifications and formant frequencies for a synthesized speech phoneme,
/// with support for diphthong dynamic offglide trajectories and nasal zero anti-resonances.
/// </summary>
public readonly struct PhonemeSpec
{
    public readonly string Symbol;
    public readonly float F1;
    public readonly float F2;
    public readonly float F3;
    public readonly float F4;
    public readonly float F5;
    public readonly float F1End;
    public readonly float F2End;
    public readonly float F3End;
    public readonly float Bandwidth1;
    public readonly float Bandwidth2;
    public readonly float Bandwidth3;
    public readonly float DurationMs;
    public readonly bool IsVoiced;
    public readonly bool IsDiphthong;
    public readonly bool IsNasal;
    public readonly bool IsFricative;
    public readonly bool IsPlosive;
    public readonly float NoiseGain;
    public readonly float VoicingGain;
    public readonly float FricativeFreq;

    public PhonemeSpec(
        string symbol,
        float f1, float f2, float f3, float f4, float f5,
        float b1, float b2, float b3,
        float durationMs,
        bool isVoiced,
        float f1End = 0.0f, float f2End = 0.0f, float f3End = 0.0f,
        bool isDiphthong = false,
        bool isNasal = false,
        bool isFricative = false,
        bool isPlosive = false,
        float noiseGain = 0.0f,
        float voicingGain = 1.0f,
        float fricativeFreq = 4500.0f)
    {
        Symbol = symbol;
        F1 = f1;
        F2 = f2;
        F3 = f3;
        F4 = f4;
        F5 = f5;
        F1End = f1End > 0 ? f1End : f1;
        F2End = f2End > 0 ? f2End : f2;
        F3End = f3End > 0 ? f3End : f3;
        Bandwidth1 = b1;
        Bandwidth2 = b2;
        Bandwidth3 = b3;
        DurationMs = durationMs;
        IsVoiced = isVoiced;
        IsDiphthong = isDiphthong;
        IsNasal = isNasal;
        IsFricative = isFricative;
        IsPlosive = isPlosive;
        NoiseGain = noiseGain;
        VoicingGain = voicingGain;
        FricativeFreq = fricativeFreq;
    }
}

/// <summary>
/// Tokenized utterance phoneme instance with contextual prosody, pitch target, and duration.
/// </summary>
public sealed class SynthesizedToken
{
    public PhonemeSpec Spec { get; set; }
    public float TargetF0 { get; set; }
    public float DurationSec { get; set; }
    public bool IsPause { get; set; }
    public bool IsStressed { get; set; }
    public float StressEmphasis { get; set; } = 1.0f;

    public SynthesizedToken(PhonemeSpec spec, float durationSec, bool isPause = false, bool isStressed = false)
    {
        Spec = spec;
        DurationSec = durationSec;
        IsPause = isPause;
        IsStressed = isStressed;
    }
}

/// <summary>
/// High-performance pure C# Grapheme-to-Phoneme (G2P) engine with English accent phonology (USA vs British RP),
/// syllable stress accentuation, prosodic boundary intonation, and dictionary-assisted phonetic transcription.
/// </summary>
public static class PhonemeEngine
{
    private static readonly Dictionary<string, PhonemeSpec> Specs = new(StringComparer.OrdinalIgnoreCase)
    {
        // -------------------------------------------------------------
        // VOWELS (Monophthongs) - Natural conversational durations (55-85ms)
        // -------------------------------------------------------------
        ["IY"] = new PhonemeSpec("IY", 280, 2300, 3050, 3600, 4500, 55, 90, 150, 75, true, voicingGain: 1.0f),
        ["IH"] = new PhonemeSpec("IH", 400, 1950, 2600, 3600, 4500, 65, 100, 160, 55, true, voicingGain: 0.95f),
        ["EH"] = new PhonemeSpec("EH", 550, 1800, 2550, 3600, 4500, 70, 110, 170, 65, true, voicingGain: 1.0f),
        ["AE"] = new PhonemeSpec("AE", 700, 1650, 2450, 3600, 4500, 75, 115, 175, 78, true, voicingGain: 1.05f),
        ["AA"] = new PhonemeSpec("AA", 780, 1150, 2400, 3500, 4400, 80, 120, 175, 82, true, voicingGain: 1.05f),
        ["AO_UK"] = new PhonemeSpec("AO_UK", 600, 950, 2400, 3500, 4400, 75, 110, 170, 70, true, voicingGain: 1.0f),
        ["AO"] = new PhonemeSpec("AO", 570, 880, 2400, 3500, 4400, 75, 100, 170, 78, true, voicingGain: 1.0f),
        ["AH"] = new PhonemeSpec("AH", 650, 1200, 2500, 3600, 4500, 75, 110, 170, 58, true, voicingGain: 0.95f),
        ["AX"] = new PhonemeSpec("AX", 500, 1400, 2400, 3500, 4400, 75, 100, 160, 45, true, voicingGain: 0.85f),
        ["UH"] = new PhonemeSpec("UH", 450, 1050, 2350, 3500, 4400, 65, 100, 160, 60, true, voicingGain: 0.95f),
        ["UW"] = new PhonemeSpec("UW", 320, 850, 2300, 3400, 4300, 60, 90, 150, 78, true, voicingGain: 1.0f),
        ["ER"] = new PhonemeSpec("ER", 480, 1350, 1700, 3400, 4400, 65, 100, 150, 82, true, voicingGain: 1.0f),

        // -------------------------------------------------------------
        // DIPHTHONGS - Dynamic formant trajectory from starting vowel to ending vowel
        // -------------------------------------------------------------
        // /eɪ/ ('face', 'say', 'glacier', 'make')
        ["EY"] = new PhonemeSpec("EY", 520, 1850, 2600, 3600, 4500, 65, 95, 160, 95, true, f1End: 350, f2End: 2200, f3End: 2750, isDiphthong: true, voicingGain: 1.0f),
        // /aɪ/ ('price', 'like', 'my', 'time')
        ["AY"] = new PhonemeSpec("AY", 750, 1250, 2550, 3600, 4500, 75, 105, 165, 105, true, f1End: 320, f2End: 2150, f3End: 2700, isDiphthong: true, voicingGain: 1.05f),
        // /ɔɪ/ ('voice', 'choice', 'boy')
        ["OY"] = new PhonemeSpec("OY", 550, 950, 2450, 3500, 4400, 70, 100, 165, 105, true, f1End: 340, f2End: 2100, f3End: 2700, isDiphthong: true, voicingGain: 1.0f),
        // /aʊ/ ('mouth', 'sound', 'now', 'how')
        ["AW"] = new PhonemeSpec("AW", 750, 1250, 2450, 3500, 4400, 75, 105, 165, 105, true, f1End: 420, f2End: 900, f3End: 2400, isDiphthong: true, voicingGain: 1.05f),
        // /oʊ/ ('goat', 'hello', 'no', 'code')
        ["OW"] = new PhonemeSpec("OW", 520, 950, 2400, 3500, 4400, 70, 95, 160, 95, true, f1End: 380, f2End: 820, f3End: 2350, isDiphthong: true, voicingGain: 1.0f),
        // /ɪə/ (British 'near', 'here', 'clear')
        ["IA"] = new PhonemeSpec("IA", 360, 2050, 2600, 3500, 4400, 65, 95, 160, 95, true, f1End: 550, f2End: 1350, f3End: 2400, isDiphthong: true, voicingGain: 0.95f),
        // /eə/ (British 'square', 'care', 'bear')
        ["EA"] = new PhonemeSpec("EA", 580, 1750, 2500, 3500, 4400, 70, 100, 160, 95, true, f1End: 500, f2End: 1400, f3End: 2400, isDiphthong: true, voicingGain: 0.95f),

        // -------------------------------------------------------------
        // LIQUIDS & GLIDES
        // -------------------------------------------------------------
        ["R_US"] = new PhonemeSpec("R_US", 350, 1250, 1650, 3100, 4200, 65, 90, 120, 58, true, voicingGain: 0.88f),
        ["R_UK"] = new PhonemeSpec("R_UK", 400, 1350, 2300, 3300, 4300, 65, 90, 140, 52, true, voicingGain: 0.88f),
        ["L"] = new PhonemeSpec("L", 380, 1150, 2800, 3500, 4400, 65, 105, 155, 55, true, voicingGain: 0.88f),
        ["W"] = new PhonemeSpec("W", 320, 750, 2350, 3400, 4300, 60, 85, 145, 52, true, voicingGain: 0.90f),
        ["Y"] = new PhonemeSpec("Y", 300, 2250, 2950, 3600, 4500, 55, 85, 145, 52, true, voicingGain: 0.90f),

        // -------------------------------------------------------------
        // NASALS
        // -------------------------------------------------------------
        ["M"] = new PhonemeSpec("M", 280, 1100, 2250, 3400, 4300, 75, 120, 190, 52, true, isNasal: true, voicingGain: 0.78f),
        ["N"] = new PhonemeSpec("N", 280, 1550, 2450, 3500, 4400, 75, 120, 190, 48, true, isNasal: true, voicingGain: 0.78f),
        ["NG"] = new PhonemeSpec("NG", 300, 1950, 2550, 3500, 4400, 80, 130, 200, 55, true, isNasal: true, voicingGain: 0.78f),

        // -------------------------------------------------------------
        // FRICATIVES
        // -------------------------------------------------------------
        ["S"] = new PhonemeSpec("S", 300, 1800, 4800, 5800, 7200, 150, 300, 400, 65, false, isFricative: true, noiseGain: 0.65f, voicingGain: 0.0f, fricativeFreq: 6200.0f),
        ["Z"] = new PhonemeSpec("Z", 250, 1750, 4600, 5600, 7000, 120, 250, 350, 60, true, isFricative: true, noiseGain: 0.35f, voicingGain: 0.65f, fricativeFreq: 5800.0f),
        ["SH"] = new PhonemeSpec("SH", 350, 1900, 3200, 4200, 5200, 140, 250, 320, 68, false, isFricative: true, noiseGain: 0.70f, voicingGain: 0.0f, fricativeFreq: 3400.0f),
        ["ZH"] = new PhonemeSpec("ZH", 280, 1850, 3100, 4100, 5100, 120, 220, 300, 60, true, isFricative: true, noiseGain: 0.40f, voicingGain: 0.65f, fricativeFreq: 3200.0f),
        ["F"] = new PhonemeSpec("F", 300, 1400, 2600, 3800, 5000, 160, 280, 380, 58, false, isFricative: true, noiseGain: 0.45f, voicingGain: 0.0f, fricativeFreq: 2600.0f),
        ["V"] = new PhonemeSpec("V", 250, 1350, 2500, 3700, 4900, 130, 240, 340, 54, true, isFricative: true, noiseGain: 0.30f, voicingGain: 0.70f, fricativeFreq: 2400.0f),
        ["TH"] = new PhonemeSpec("TH", 320, 1500, 2700, 3900, 5200, 170, 300, 420, 58, false, isFricative: true, noiseGain: 0.40f, voicingGain: 0.0f, fricativeFreq: 3800.0f),
        ["DH"] = new PhonemeSpec("DH", 260, 1450, 2600, 3800, 5000, 140, 260, 360, 48, true, isFricative: true, noiseGain: 0.25f, voicingGain: 0.75f, fricativeFreq: 3600.0f),
        ["HH"] = new PhonemeSpec("HH", 400, 1600, 2500, 3600, 4600, 120, 200, 280, 50, false, isFricative: true, noiseGain: 0.40f, voicingGain: 0.0f, fricativeFreq: 2200.0f),

        // -------------------------------------------------------------
        // PLOSIVES / STOPS
        // -------------------------------------------------------------
        ["P"] = new PhonemeSpec("P", 250, 1100, 2300, 3400, 4400, 100, 150, 200, 42, false, isPlosive: true, noiseGain: 0.55f, voicingGain: 0.0f, fricativeFreq: 800.0f),
        ["B"] = new PhonemeSpec("B", 200, 1050, 2200, 3300, 4300, 80, 120, 180, 35, true, isPlosive: true, noiseGain: 0.20f, voicingGain: 0.85f, fricativeFreq: 700.0f),
        ["T"] = new PhonemeSpec("T", 250, 1750, 3200, 4200, 5200, 100, 180, 250, 42, false, isPlosive: true, noiseGain: 0.65f, voicingGain: 0.0f, fricativeFreq: 4200.0f),
        ["FLAP"] = new PhonemeSpec("FLAP", 280, 1650, 2600, 3600, 4500, 70, 110, 160, 24, true, voicingGain: 0.85f),
        ["D"] = new PhonemeSpec("D", 220, 1650, 2700, 3600, 4500, 80, 130, 200, 35, true, isPlosive: true, noiseGain: 0.22f, voicingGain: 0.85f, fricativeFreq: 3800.0f),
        ["K"] = new PhonemeSpec("K", 300, 1600, 2400, 3500, 4500, 110, 190, 260, 45, false, isPlosive: true, noiseGain: 0.65f, voicingGain: 0.0f, fricativeFreq: 2200.0f),
        ["G"] = new PhonemeSpec("G", 240, 1500, 2300, 3400, 4400, 90, 140, 210, 38, true, isPlosive: true, noiseGain: 0.22f, voicingGain: 0.85f, fricativeFreq: 2000.0f),

        // -------------------------------------------------------------
        // AFFRICATES
        // -------------------------------------------------------------
        ["CH"] = new PhonemeSpec("CH", 320, 1850, 3200, 4200, 5200, 120, 220, 300, 70, false, isFricative: true, isPlosive: true, noiseGain: 0.70f, voicingGain: 0.0f, fricativeFreq: 3600.0f),
        ["JH"] = new PhonemeSpec("JH", 260, 1800, 3000, 4000, 5000, 100, 190, 260, 65, true, isFricative: true, isPlosive: true, noiseGain: 0.35f, voicingGain: 0.70f, fricativeFreq: 3400.0f),

        // -------------------------------------------------------------
        // SILENCES & PAUSES
        // -------------------------------------------------------------
        ["_SIL"] = new PhonemeSpec("_SIL", 0, 0, 0, 0, 0, 0, 0, 0, 45, false),
        ["_GAP"] = new PhonemeSpec("_GAP", 0, 0, 0, 0, 0, 0, 0, 0, 18, false),
        ["_COMMA"] = new PhonemeSpec("_COMMA", 0, 0, 0, 0, 0, 0, 0, 0, 110, false),
        ["_PERIOD"] = new PhonemeSpec("_PERIOD", 0, 0, 0, 0, 0, 0, 0, 0, 190, false),
        ["_QUESTION"] = new PhonemeSpec("_QUESTION", 0, 0, 0, 0, 0, 0, 0, 0, 180, false)
    };

    private static readonly Dictionary<string, (string[] Us, string[] Uk)> WordLexicon = new(StringComparer.OrdinalIgnoreCase)
    {
        ["the"] = (["DH", "AX"], ["DH", "AX"]),
        ["to"] = (["T", "UW"], ["T", "UW"]),
        ["and"] = (["AE", "N", "D"], ["AE", "N", "D"]),
        ["a"] = (["AX"], ["AX"]),
        ["in"] = (["IH", "N"], ["IH", "N"]),
        ["that"] = (["DH", "AE", "T"], ["DH", "AE", "T"]),
        ["have"] = (["HH", "AE", "V"], ["HH", "AE", "V"]),
        ["i"] = (["AY"], ["AY"]),
        ["it"] = (["IH", "T"], ["IH", "T"]),
        ["for"] = (["F", "AO", "R_US"], ["F", "AO"]),
        ["not"] = (["N", "AA", "T"], ["N", "AO_UK", "T"]),
        ["on"] = (["AA", "N"], ["AO_UK", "N"]),
        ["with"] = (["W", "IH", "DH"], ["W", "IH", "DH"]),
        ["he"] = (["HH", "IY"], ["HH", "IY"]),
        ["as"] = (["AE", "Z"], ["AE", "Z"]),
        ["you"] = (["Y", "UW"], ["Y", "UW"]),
        ["do"] = (["D", "UW"], ["D", "UW"]),
        ["at"] = (["AE", "T"], ["AE", "T"]),
        ["this"] = (["DH", "IH", "S"], ["DH", "IH", "S"]),
        ["but"] = (["B", "AH", "T"], ["B", "AH", "T"]),
        ["his"] = (["HH", "IH", "Z"], ["HH", "IH", "Z"]),
        ["by"] = (["B", "AY"], ["B", "AY"]),
        ["from"] = (["F", "R_US", "AH", "M"], ["F", "R_UK", "AO_UK", "M"]),
        ["they"] = (["DH", "EY"], ["DH", "EY"]),
        ["we"] = (["W", "IY"], ["W", "IY"]),
        ["say"] = (["S", "EY"], ["S", "EY"]),
        ["her"] = (["HH", "ER"], ["HH", "ER"]),
        ["she"] = (["SH", "IY"], ["SH", "IY"]),
        ["or"] = (["AO", "R_US"], ["AO"]),
        ["an"] = (["AE", "N"], ["AE", "N"]),
        ["will"] = (["W", "IH", "L"], ["W", "IH", "L"]),
        ["my"] = (["M", "AY"], ["M", "AY"]),
        ["one"] = (["W", "AH", "N"], ["W", "AH", "N"]),
        ["all"] = (["AO", "L"], ["AO", "L"]),
        ["would"] = (["W", "UH", "D"], ["W", "UH", "D"]),
        ["there"] = (["DH", "EH", "R_US"], ["DH", "EA"]),
        ["their"] = (["DH", "EH", "R_US"], ["DH", "EA"]),
        ["what"] = (["W", "AH", "T"], ["W", "AO_UK", "T"]),
        ["so"] = (["S", "OW"], ["S", "OW"]),
        ["up"] = (["AH", "P"], ["AH", "P"]),
        ["out"] = (["AW", "T"], ["AW", "T"]),
        ["if"] = (["IH", "F"], ["IH", "F"]),
        ["about"] = (["AX", "B", "AW", "T"], ["AX", "B", "AW", "T"]),
        ["who"] = (["HH", "UW"], ["HH", "UW"]),
        ["get"] = (["G", "EH", "T"], ["G", "EH", "T"]),
        ["which"] = (["W", "IH", "CH"], ["W", "IH", "CH"]),
        ["go"] = (["G", "OW"], ["G", "OW"]),
        ["me"] = (["M", "IY"], ["M", "IY"]),
        ["when"] = (["W", "EH", "N"], ["W", "EH", "N"]),
        ["make"] = (["M", "EY", "K"], ["M", "EY", "K"]),
        ["can"] = (["K", "AE", "N"], ["K", "AE", "N"]),
        ["like"] = (["L", "AY", "K"], ["L", "AY", "K"]),
        ["time"] = (["T", "AY", "M"], ["T", "AY", "M"]),
        ["no"] = (["N", "OW"], ["N", "OW"]),
        ["just"] = (["JH", "AH", "S", "T"], ["JH", "AH", "S", "T"]),
        ["him"] = (["HH", "IH", "M"], ["HH", "IH", "M"]),
        ["know"] = (["N", "OW"], ["N", "OW"]),
        ["take"] = (["T", "EY", "K"], ["T", "EY", "K"]),
        ["people"] = (["P", "IY", "P", "AX", "L"], ["P", "IY", "P", "AX", "L"]),
        ["into"] = (["IH", "N", "T", "UW"], ["IH", "N", "T", "UW"]),
        ["year"] = (["Y", "IH", "R_US"], ["Y", "IA"]),
        ["your"] = (["Y", "AO", "R_US"], ["Y", "AO"]),
        ["good"] = (["G", "UH", "D"], ["G", "UH", "D"]),
        ["some"] = (["S", "AH", "M"], ["S", "AH", "M"]),
        ["could"] = (["K", "UH", "D"], ["K", "UH", "D"]),
        ["them"] = (["DH", "EH", "M"], ["DH", "EH", "M"]),
        ["see"] = (["S", "IY"], ["S", "IY"]),
        ["other"] = (["AH", "DH", "ER"], ["AH", "DH", "AX"]),
        ["than"] = (["DH", "AE", "N"], ["DH", "AE", "N"]),
        ["then"] = (["DH", "EH", "N"], ["DH", "EH", "N"]),
        ["now"] = (["N", "AW"], ["N", "AW"]),
        ["look"] = (["L", "UH", "K"], ["L", "UH", "K"]),
        ["only"] = (["OW", "N", "L", "IY"], ["OW", "N", "L", "IY"]),
        ["come"] = (["K", "AH", "M"], ["K", "AH", "M"]),
        ["its"] = (["IH", "T", "S"], ["IH", "T", "S"]),
        ["it's"] = (["IH", "T", "S"], ["IH", "T", "S"]),
        ["over"] = (["OW", "V", "ER"], ["OW", "V", "AX"]),
        ["think"] = (["TH", "IH", "NG", "K"], ["TH", "IH", "NG", "K"]),
        ["also"] = (["AO", "L", "S", "OW"], ["AO", "L", "S", "OW"]),
        ["back"] = (["B", "AE", "K"], ["B", "AE", "K"]),
        ["after"] = (["AE", "F", "T", "ER"], ["AA", "F", "T", "AX"]),
        ["use"] = (["Y", "UW", "Z"], ["Y", "UW", "Z"]),
        ["two"] = (["T", "UW"], ["T", "UW"]),
        ["how"] = (["HH", "AW"], ["HH", "AW"]),
        ["our"] = (["AW", "ER"], ["AW", "AX"]),
        ["work"] = (["W", "ER", "K"], ["W", "ER", "K"]),
        ["first"] = (["F", "ER", "S", "T"], ["F", "ER", "S", "T"]),
        ["well"] = (["W", "EH", "L"], ["W", "EH", "L"]),
        ["way"] = (["W", "EY"], ["W", "EY"]),
        ["even"] = (["IY", "V", "AX", "N"], ["IY", "V", "AX", "N"]),
        ["new"] = (["N", "UW"], ["N", "Y", "UW"]),
        ["want"] = (["W", "AA", "N", "T"], ["W", "AO_UK", "N", "T"]),
        ["because"] = (["B", "IH", "K", "AH", "Z"], ["B", "IH", "K", "AO_UK", "Z"]),
        ["any"] = (["EH", "N", "IY"], ["EH", "N", "IY"]),
        ["these"] = (["DH", "IY", "Z"], ["DH", "IY", "Z"]),
        ["give"] = (["G", "IH", "V"], ["G", "IH", "V"]),
        ["day"] = (["D", "EY"], ["D", "EY"]),
        ["most"] = (["M", "OW", "S", "T"], ["M", "OW", "S", "T"]),
        ["us"] = (["AH", "S"], ["AH", "S"]),
        ["glacier"] = (["G", "L", "EY", "SH", "ER"], ["G", "L", "AE", "S", "IY", "AX"]),
        ["inference"] = (["IH", "N", "F", "ER", "AX", "N", "S"], ["IH", "N", "F", "AX", "R_UK", "AX", "N", "S"]),
        ["voice"] = (["V", "OY", "S"], ["V", "OY", "S"]),
        ["voices"] = (["V", "OY", "S", "IH", "Z"], ["V", "OY", "S", "IH", "Z"]),
        ["gen"] = (["JH", "EH", "N"], ["JH", "EH", "N"]),
        ["human"] = (["HH", "Y", "UW", "M", "AX", "N"], ["HH", "Y", "UW", "M", "AX", "N"]),
        ["quality"] = (["K", "W", "AA", "L", "IH", "T", "IY"], ["K", "W", "AO_UK", "L", "IH", "T", "IY"]),
        ["exceptional"] = (["IH", "K", "S", "EH", "P", "SH", "AX", "N", "AX", "L"], ["IH", "K", "S", "EH", "P", "SH", "AX", "N", "AX", "L"]),
        ["speech"] = (["S", "P", "IY", "CH"], ["S", "P", "IY", "CH"]),
        ["sound"] = (["S", "AW", "N", "D"], ["S", "AW", "N", "D"]),
        ["sounds"] = (["S", "AW", "N", "D", "Z"], ["S", "AW", "N", "D", "Z"]),
        ["american"] = (["AX", "M", "EH", "R_US", "IH", "K", "AX", "N"], ["AX", "M", "EH", "R_UK", "IH", "K", "AX", "N"]),
        ["british"] = (["B", "R_US", "IH", "T", "IH", "SH"], ["B", "R_UK", "IH", "T", "IH", "SH"]),
        ["english"] = (["IH", "NG", "G", "L", "IH", "SH"], ["IH", "NG", "G", "L", "IH", "SH"]),
        ["accent"] = (["AE", "K", "S", "EH", "N", "T"], ["AE", "K", "S", "AX", "N", "T"]),
        ["accents"] = (["AE", "K", "S", "EH", "N", "T", "S"], ["AE", "K", "S", "AX", "N", "T", "S"]),
        ["engine"] = (["EH", "N", "JH", "IH", "N"], ["EH", "N", "JH", "IH", "N"]),
        ["audio"] = (["AO", "D", "IY", "OW"], ["AO", "D", "IY", "OW"]),
        ["high"] = (["HH", "AY"], ["HH", "AY"]),
        ["performance"] = (["P", "ER", "F", "AO", "R_US", "M", "AX", "N", "S"], ["P", "AX", "F", "AO", "M", "AX", "N", "S"]),
        ["pure"] = (["P", "Y", "UH", "R_US"], ["P", "Y", "AO"]),
        ["world"] = (["W", "ER", "L", "D"], ["W", "ER", "L", "D"]),
        ["welcome"] = (["W", "EH", "L", "K", "AH", "M"], ["W", "EH", "L", "K", "AH", "M"]),
        ["hello"] = (["HH", "EH", "L", "OW"], ["HH", "EH", "L", "OW"]),
        ["male"] = (["M", "EY", "L"], ["M", "EY", "L"]),
        ["female"] = (["F", "IY", "M", "EY", "L"], ["F", "IY", "M", "EY", "L"]),
        ["water"] = (["W", "AO", "T", "ER"], ["W", "AO", "T", "AX"]),
        ["better"] = (["B", "EH", "T", "ER"], ["B", "EH", "T", "AX"]),
        ["butter"] = (["B", "AH", "T", "ER"], ["B", "AH", "T", "AX"]),
        ["fast"] = (["F", "AE", "S", "T"], ["F", "AA", "S", "T"]),
        ["path"] = (["P", "AE", "TH"], ["P", "AA", "TH"]),
        ["glass"] = (["G", "L", "AE", "S"], ["G", "L", "AA", "S"]),
        ["ask"] = (["AE", "S", "K"], ["AA", "S", "K"]),
        ["dance"] = (["D", "AE", "N", "S"], ["D", "AA", "N", "S"]),
        ["car"] = (["K", "AA", "R_US"], ["K", "AA"]),
        ["hard"] = (["HH", "AA", "R_US", "D"], ["HH", "AA", "D"]),
        ["star"] = (["S", "T", "AA", "R_US"], ["S", "T", "AA"]),
        ["morning"] = (["M", "AO", "R_US", "N", "IH", "NG"], ["M", "AO", "N", "IH", "NG"]),
        ["evening"] = (["IY", "V", "N", "IH", "NG"], ["IY", "V", "N", "IH", "NG"]),
        ["afternoon"] = (["AE", "F", "T", "ER", "N", "UW", "N"], ["AA", "F", "T", "AX", "N", "UW", "N"]),
        ["warm"] = (["W", "AO", "R_US", "M"], ["W", "AO", "M"]),
        ["clear"] = (["K", "L", "IH", "R_US"], ["K", "L", "IA"]),
        ["crisp"] = (["K", "R_US", "IH", "S", "P"], ["K", "R_UK", "IH", "S", "P"]),
        ["smooth"] = (["S", "M", "UW", "DH"], ["S", "M", "UW", "DH"]),
        ["natural"] = (["N", "AE", "CH", "ER", "AX", "L"], ["N", "AE", "CH", "AX", "R_UK", "AX", "L"]),
        ["neural"] = (["N", "UH", "R_US", "AX", "L"], ["N", "Y", "UH", "R_UK", "AX", "L"]),
        ["great"] = (["G", "R_US", "EY", "T"], ["G", "R_UK", "EY", "T"]),
        ["beautiful"] = (["B", "Y", "UW", "T", "IH", "F", "AX", "L"], ["B", "Y", "UW", "T", "IH", "F", "AX", "L"]),
        ["system"] = (["S", "IH", "S", "T", "AX", "M"], ["S", "IH", "S", "T", "AX", "M"]),
        ["real"] = (["R_US", "IY", "AX", "L"], ["R_UK", "IA", "L"]),
        ["today"] = (["T", "AX", "D", "EY"], ["T", "AX", "D", "EY"]),
        ["listen"] = (["L", "IH", "S", "AX", "N"], ["L", "IH", "S", "AX", "N"]),
        ["hear"] = (["HH", "IH", "R_US"], ["HH", "IA"]),
        ["test"] = (["T", "EH", "S", "T"], ["T", "EH", "S", "T"]),
        ["pipeline"] = (["P", "AY", "P", "L", "AY", "N"], ["P", "AY", "P", "L", "AY", "N"]),
        ["ready"] = (["R_US", "EH", "D", "IY"], ["R_UK", "EH", "D", "IY"]),
        ["state"] = (["S", "T", "EY", "T"], ["S", "T", "EY", "T"]),
        ["art"] = (["AA", "R_US", "T"], ["AA", "T"]),
        ["several"] = (["S", "EH", "V", "ER", "AX", "L"], ["S", "EH", "V", "R_UK", "AX", "L"]),
        ["process"] = (["P", "R_US", "AA", "S", "EH", "S"], ["P", "R_UK", "OW", "S", "EH", "S"]),
        ["until"] = (["AH", "N", "T", "IH", "L"], ["AH", "N", "T", "IH", "L"]),
        ["are"] = (["AA", "R_US"], ["AA"]),
        ["improving"] = (["IH", "M", "P", "R_US", "UW", "V", "IH", "NG"], ["IH", "M", "P", "R_UK", "UW", "V", "IH", "NG"]),
        ["iterate"] = (["IH", "T", "ER", "EY", "T"], ["IH", "T", "AX", "R_UK", "EY", "T"]),
        ["itterate"] = (["IH", "T", "ER", "EY", "T"], ["IH", "T", "AX", "R_UK", "EY", "T"]),
        ["improvement"] = (["IH", "M", "P", "R_US", "UW", "V", "M", "AX", "N", "T"], ["IH", "M", "P", "R_UK", "UW", "V", "M", "AX", "N", "T"]),
        ["intelligent"] = (["IH", "N", "T", "EH", "L", "IH", "JH", "AX", "N", "T"], ["IH", "N", "T", "EH", "L", "IH", "JH", "AX", "N", "T"]),
        ["assistant"] = (["AX", "S", "IH", "S", "T", "AX", "N", "T"], ["AX", "S", "IH", "S", "T", "AX", "N", "T"]),
        ["computer"] = (["K", "AX", "M", "P", "Y", "UW", "T", "ER"], ["K", "AX", "M", "P", "Y", "UW", "T", "AX"]),
        ["model"] = (["M", "AA", "D", "AX", "L"], ["M", "AO_UK", "D", "AX", "L"]),
        ["technology"] = (["T", "EH", "K", "N", "AA", "L", "AX", "JH", "IY"], ["T", "EH", "K", "N", "AO_UK", "L", "AX", "JH", "IY"]),
        ["conversation"] = (["K", "AA", "N", "V", "ER", "S", "EY", "SH", "AX", "N"], ["K", "AO_UK", "N", "V", "AX", "S", "EY", "SH", "AX", "N"]),
        ["please"] = (["P", "L", "IY", "Z"], ["P", "L", "IY", "Z"]),
        ["thank"] = (["TH", "AE", "NG", "K"], ["TH", "AE", "NG", "K"]),
        ["thanks"] = (["TH", "AE", "NG", "K", "S"], ["TH", "AE", "NG", "K", "S"]),
        ["yes"] = (["Y", "EH", "S"], ["Y", "EH", "S"]),
        ["sure"] = (["SH", "UH", "R_US"], ["SH", "AO"]),
        ["fine"] = (["F", "AY", "N"], ["F", "AY", "N"]),
        ["ok"] = (["OW", "K", "EY"], ["OW", "K", "EY"]),
        ["okay"] = (["OW", "K", "EY"], ["OW", "K", "EY"])
    };

    public static List<SynthesizedToken> ConvertTextToTokens(
        string text,
        EnglishAccent accent,
        float baseF0,
        float speed = 1.0f)
    {
        var tokens = new List<SynthesizedToken>();
        float speedClamped = Math.Clamp(speed, 0.5f, 2.5f);

        tokens.Add(new SynthesizedToken(Specs["_SIL"], 0.04f / speedClamped, isPause: true));

        var words = SplitIntoWordsAndPunctuation(text);
        if (words.Count == 0) return tokens;

        int totalWords = words.Count;
        float sentenceDeclinationFactor = 0.88f;

        for (int wIdx = 0; wIdx < totalWords; wIdx++)
        {
            var item = words[wIdx];
            string w = item.Text;
            bool isQuestion = item.IsQuestion;
            bool isExclamation = item.IsExclamation;
            bool isPeriod = item.IsPeriod;
            bool isComma = item.IsComma;

            float wordProgress = (float)wIdx / Math.Max(1, totalWords - 1);
            float declinationF0 = baseF0 * (1.0f - (1.0f - sentenceDeclinationFactor) * wordProgress);

            string[] phonemeKeys = LookupOrTranscribeWord(w, accent);
            int stressedIdx = DetermineStressedPhonemeIndex(phonemeKeys);

            for (int pIdx = 0; pIdx < phonemeKeys.Length; pIdx++)
            {
                string key = phonemeKeys[pIdx];

                if (accent == EnglishAccent.American && key == "T")
                {
                    bool prevIsVowel = pIdx > 0 && IsVowel(phonemeKeys[pIdx - 1]);
                    bool nextIsVowel = pIdx < phonemeKeys.Length - 1 && IsVowel(phonemeKeys[pIdx + 1]);
                    if (prevIsVowel && nextIsVowel)
                    {
                        key = "FLAP";
                    }
                }

                if (!Specs.TryGetValue(key, out var spec))
                {
                    spec = Specs["AX"];
                }

                bool isStressed = (pIdx == stressedIdx);
                float durationSec = (spec.DurationMs / 1000.0f) / speedClamped;

                float targetF0 = declinationF0;
                float stressEmphasis = 1.0f;

                if (isStressed && spec.IsVoiced)
                {
                    durationSec *= 1.25f;
                    targetF0 *= 1.15f;
                    stressEmphasis = 1.18f;
                }

                if (isQuestion && wIdx >= totalWords - 2)
                {
                    float questionRise = 1.0f + 0.30f * ((float)(pIdx + 1) / phonemeKeys.Length);
                    targetF0 *= questionRise;
                }
                else if (isExclamation)
                {
                    targetF0 *= 1.12f;
                    stressEmphasis *= 1.15f;
                }
                else if (isPeriod && wIdx == totalWords - 1 && pIdx == phonemeKeys.Length - 1)
                {
                    targetF0 *= 0.88f;
                }

                var tok = new SynthesizedToken(spec, durationSec, isPause: false, isStressed: isStressed)
                {
                    TargetF0 = targetF0,
                    StressEmphasis = stressEmphasis
                };
                tokens.Add(tok);
            }

            if (isComma)
            {
                tokens.Add(new SynthesizedToken(Specs["_COMMA"], 0.10f / speedClamped, isPause: true));
            }
            else if (isPeriod)
            {
                tokens.Add(new SynthesizedToken(Specs["_PERIOD"], 0.18f / speedClamped, isPause: true));
            }
            else if (isQuestion)
            {
                tokens.Add(new SynthesizedToken(Specs["_QUESTION"], 0.18f / speedClamped, isPause: true));
            }
            else if (isExclamation)
            {
                tokens.Add(new SynthesizedToken(Specs["_PERIOD"], 0.16f / speedClamped, isPause: true));
            }
            else
            {
                tokens.Add(new SynthesizedToken(Specs["_GAP"], 0.018f / speedClamped, isPause: true));
            }
        }

        tokens.Add(new SynthesizedToken(Specs["_SIL"], 0.06f / speedClamped, isPause: true));
        return tokens;
    }

    private static string[] LookupOrTranscribeWord(string word, EnglishAccent accent)
    {
        string clean = word.Trim().ToLowerInvariant();

        if (WordLexicon.TryGetValue(clean, out var entry))
        {
            return accent == EnglishAccent.American ? entry.Us : entry.Uk;
        }

        return TranscribeByRules(clean, accent);
    }

    private static string[] TranscribeByRules(string word, EnglishAccent accent)
    {
        var list = new List<string>();
        int i = 0;
        int len = word.Length;

        while (i < len)
        {
            if (i == len - 4 && word.EndsWith("tion"))
            {
                list.Add("SH");
                list.Add("AX");
                list.Add("N");
                break;
            }
            if (i == len - 4 && word.EndsWith("sion"))
            {
                list.Add("ZH");
                list.Add("AX");
                list.Add("N");
                break;
            }
            if (i == len - 3 && word.EndsWith("ing"))
            {
                list.Add("IH");
                list.Add("NG");
                break;
            }

            if (i + 1 < len)
            {
                string two = word.Substring(i, 2);
                switch (two)
                {
                    case "th":
                        list.Add(i == 0 && len <= 4 ? "DH" : "TH");
                        i += 2;
                        continue;
                    case "sh":
                        list.Add("SH");
                        i += 2;
                        continue;
                    case "ch":
                        list.Add("CH");
                        i += 2;
                        continue;
                    case "ph":
                        list.Add("F");
                        i += 2;
                        continue;
                    case "wh":
                        list.Add("W");
                        i += 2;
                        continue;
                    case "ee":
                        list.Add("IY");
                        i += 2;
                        continue;
                    case "oo":
                        list.Add("UW");
                        i += 2;
                        continue;
                    case "ea":
                        list.Add("IY");
                        i += 2;
                        continue;
                    case "ai":
                    case "ay":
                        list.Add("EY");
                        i += 2;
                        continue;
                    case "oa":
                        list.Add("OW");
                        i += 2;
                        continue;
                    case "oi":
                    case "oy":
                        list.Add("OY");
                        i += 2;
                        continue;
                    case "ou":
                    case "ow":
                        list.Add("AW");
                        i += 2;
                        continue;
                    case "au":
                    case "aw":
                        list.Add("AO");
                        i += 2;
                        continue;
                    case "ck":
                        list.Add("K");
                        i += 2;
                        continue;
                    case "ng":
                        list.Add("NG");
                        i += 2;
                        continue;
                    case "qu":
                        list.Add("K");
                        list.Add("W");
                        i += 2;
                        continue;
                }
            }

            char c = word[i];

            if (c == 'e' && i == len - 1 && i > 1)
            {
                i++;
                continue;
            }

            switch (c)
            {
                case 'a':
                    if (i + 2 < len && !IsVowelChar(word[i + 1]) && word[i + 2] == 'e' && i + 2 == len - 1)
                        list.Add("EY");
                    else if (accent == EnglishAccent.British && (word.EndsWith("ast") || word.EndsWith("ath") || word.EndsWith("ass")))
                        list.Add("AA");
                    else
                        list.Add("AE");
                    break;
                case 'e':
                    list.Add("EH");
                    break;
                case 'i':
                    if (i + 2 < len && !IsVowelChar(word[i + 1]) && word[i + 2] == 'e' && i + 2 == len - 1)
                        list.Add("AY");
                    else
                        list.Add("IH");
                    break;
                case 'o':
                    if (i + 2 < len && !IsVowelChar(word[i + 1]) && word[i + 2] == 'e' && i + 2 == len - 1)
                        list.Add("OW");
                    else if (accent == EnglishAccent.British)
                        list.Add("AO_UK");
                    else
                        list.Add("AA");
                    break;
                case 'u':
                    if (i + 2 < len && !IsVowelChar(word[i + 1]) && word[i + 2] == 'e' && i + 2 == len - 1)
                    {
                        if (accent == EnglishAccent.British) list.Add("Y");
                        list.Add("UW");
                    }
                    else
                    {
                        list.Add("AH");
                    }
                    break;
                case 'y':
                    list.Add(i == 0 ? "Y" : "IY");
                    break;
                case 'b': list.Add("B"); break;
                case 'c':
                    if (i + 1 < len && (word[i + 1] == 'e' || word[i + 1] == 'i' || word[i + 1] == 'y'))
                        list.Add("S");
                    else
                        list.Add("K");
                    break;
                case 'd': list.Add("D"); break;
                case 'f': list.Add("F"); break;
                case 'g':
                    if (i + 1 < len && (word[i + 1] == 'e' || word[i + 1] == 'i' || word[i + 1] == 'y'))
                        list.Add("JH");
                    else
                        list.Add("G");
                    break;
                case 'h': list.Add("HH"); break;
                case 'j': list.Add("JH"); break;
                case 'k': list.Add("K"); break;
                case 'l': list.Add("L"); break;
                case 'm': list.Add("M"); break;
                case 'n': list.Add("N"); break;
                case 'p': list.Add("P"); break;
                case 'r':
                    if (accent == EnglishAccent.British)
                    {
                        bool prevIsVowel = i > 0 && IsVowelChar(word[i - 1]);
                        bool isEndOrConsonant = (i == len - 1) || (i + 1 < len && !IsVowelChar(word[i + 1]));
                        if (prevIsVowel && isEndOrConsonant)
                        {
                            list.Add("AX");
                        }
                        else
                        {
                            list.Add("R_UK");
                        }
                    }
                    else
                    {
                        list.Add("R_US");
                    }
                    break;
                case 's':
                    if (i == len - 1 && i > 0 && (IsVowelChar(word[i - 1]) || word[i - 1] == 'd' || word[i - 1] == 'g' || word[i - 1] == 'm' || word[i - 1] == 'n'))
                        list.Add("Z");
                    else
                        list.Add("S");
                    break;
                case 't': list.Add("T"); break;
                case 'v': list.Add("V"); break;
                case 'w': list.Add("W"); break;
                case 'x':
                    list.Add("K");
                    list.Add("S");
                    break;
                case 'z': list.Add("Z"); break;
                default:
                    break;
            }
            i++;
        }

        if (list.Count == 0) list.Add("AX");
        return list.ToArray();
    }

    private static bool IsVowelChar(char c) => c is 'a' or 'e' or 'i' or 'o' or 'u' or 'y';

    private static bool IsVowel(string phoneme)
    {
        return phoneme is "IY" or "IH" or "EH" or "AE" or "AA" or "AO" or "AO_UK" or "AH" or "AX" or "UH" or "UW" or "ER" or "EY" or "AY" or "OY" or "AW" or "OW" or "IA" or "EA";
    }

    private static int DetermineStressedPhonemeIndex(string[] phonemes)
    {
        for (int i = 0; i < phonemes.Length; i++)
        {
            if (IsVowel(phonemes[i])) return i;
        }
        return 0;
    }

    private sealed class WordEntry
    {
        public string Text { get; set; } = string.Empty;
        public bool IsPeriod { get; set; }
        public bool IsQuestion { get; set; }
        public bool IsExclamation { get; set; }
        public bool IsComma { get; set; }
    }

    private static List<WordEntry> SplitIntoWordsAndPunctuation(string text)
    {
        var result = new List<WordEntry>();
        var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (var raw in tokens)
        {
            string t = raw.Trim();
            if (string.IsNullOrEmpty(t)) continue;

            bool isQ = t.EndsWith('?');
            bool isExcl = t.EndsWith('!');
            bool isPer = t.EndsWith('.');
            bool isCom = t.EndsWith(',') || t.EndsWith(';') || t.EndsWith(':');

            string clean = t.TrimEnd('.', ',', '!', '?', ';', ':', '-', '"', '(', ')');
            if (string.IsNullOrEmpty(clean)) continue;

            result.Add(new WordEntry
            {
                Text = clean,
                IsQuestion = isQ,
                IsExclamation = isExcl,
                IsPeriod = isPer,
                IsComma = isCom
            });
        }
        return result;
    }
}
