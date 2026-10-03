namespace Glacier.Inference.Audio;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

/// <summary>
/// Voice style configuration containing physiological vocal tract and acoustic parameters.
/// </summary>
public sealed class VoiceProfile
{
    public string Name { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public EnglishAccent Accent { get; set; }
    public VoiceGender Gender { get; set; }
    public float BaseF0 { get; set; }
    public float PitchRange { get; set; }
    public float FormantScale { get; set; } = 1.0f;
    public float Breathiness { get; set; } = 0.04f;
    public float JitterAmount { get; set; } = 0.006f;
    public float ShimmerAmount { get; set; } = 0.025f;
    public float OpenQuotient { get; set; } = 0.60f;
    public float Warmth { get; set; } = 1.15f;
    public float VibratoRate { get; set; } = 5.2f;
    public float VibratoDepth { get; set; } = 1.1f;
    public string Description { get; set; } = string.Empty;
}

/// <summary>
/// High-fidelity acoustic vocal tract synthesizer implementing physical glottal flow dynamics (Liljencrants-Fant / Rosenberg),
/// cascade recursive digital biquad resonators (F1-F5), nasal anti-resonance zero filters, diphthong formant trajectory glides,
/// organic micro-perturbations (jitter/shimmer/breathiness), coarticulation smoothing, consonant transient shaping, and lip radiation impedance.
/// </summary>
public sealed class VocalTractSynthesizer
{
    public const int DefaultSampleRate = 24000;
    private readonly int _sampleRate;

    public int SampleRate => _sampleRate;

    public VocalTractSynthesizer(int sampleRate = DefaultSampleRate)
    {
        _sampleRate = sampleRate;
    }

    /// <summary>
    /// Second-order recursive digital IIR biquad resonator filter.
    /// Implements unity-DC gain resonator for series cascade vocal tract modeling.
    /// </summary>
    private struct CascadeResonator
    {
        private float _b1;
        private float _b2;
        private float _gain;
        private float _y1;
        private float _y2;

        public void SetParameters(float f, float bw, int sampleRate)
        {
            f = Math.Clamp(f, 40.0f, sampleRate * 0.48f);
            bw = Math.Clamp(bw, 30.0f, 1800.0f);

            float r = MathF.Exp(-MathF.PI * bw / sampleRate);
            float theta = 2.0f * MathF.PI * f / sampleRate;

            _b1 = 2.0f * r * MathF.Cos(theta);
            _b2 = -r * r;
            _gain = 1.0f - _b1 - _b2; // Unity DC gain: H(1) = 1.0
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Process(float input)
        {
            float output = _gain * input + _b1 * _y1 + _b2 * _y2;
            _y2 = _y1;
            _y1 = output;
            return output;
        }

        public void Reset()
        {
            _y1 = 0.0f;
            _y2 = 0.0f;
        }
    }

    /// <summary>
    /// Biquad anti-resonator (zero filter) for nasal cavity anti-resonance notch.
    /// </summary>
    private struct AntiResonator
    {
        private float _a0;
        private float _a1;
        private float _a2;
        private float _x1;
        private float _x2;

        public void SetParameters(float fZero, float bw, int sampleRate)
        {
            float r = MathF.Exp(-MathF.PI * bw / sampleRate);
            float theta = 2.0f * MathF.PI * fZero / sampleRate;

            float b1 = 2.0f * r * MathF.Cos(theta);
            float b2 = -r * r;
            float scale = 1.0f / (1.0f - b1 - b2);

            _a0 = scale;
            _a1 = -scale * b1;
            _a2 = -scale * b2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Process(float input)
        {
            float output = _a0 * input + _a1 * _x1 + _a2 * _x2;
            _x2 = _x1;
            _x1 = input;
            return output;
        }
    }

    /// <summary>
    /// Bandpass resonator for fricative and aspiration noise shaping.
    /// </summary>
    private struct BandpassResonator
    {
        private float _b1;
        private float _b2;
        private float _gain;
        private float _y1;
        private float _y2;

        public void SetParameters(float f, float bw, int sampleRate)
        {
            f = Math.Clamp(f, 100.0f, sampleRate * 0.48f);
            bw = Math.Clamp(bw, 50.0f, 2500.0f);

            float r = MathF.Exp(-MathF.PI * bw / sampleRate);
            float theta = 2.0f * MathF.PI * f / sampleRate;

            _b1 = 2.0f * r * MathF.Cos(theta);
            _b2 = -r * r;
            _gain = 1.0f - r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Process(float input)
        {
            float output = _gain * input + _b1 * _y1 + _b2 * _y2;
            _y2 = _y1;
            _y1 = output;
            return output;
        }
    }

    /// <summary>
    /// Synthesizes high-fidelity 24kHz audio from a sequence of phonetic tokens and voice profile parameters.
    /// </summary>
    public float[] Render(List<SynthesizedToken> tokens, VoiceProfile profile, float speed = 1.0f)
    {
        if (tokens == null || tokens.Count == 0) return Array.Empty<float>();

        int totalSamples = 0;
        foreach (var tok in tokens)
        {
            totalSamples += (int)(tok.DurationSec * _sampleRate);
        }

        if (totalSamples <= 0) return Array.Empty<float>();

        var output = new float[totalSamples];
        int sampleCursor = 0;

        // Series Cascade Resonators: F1 -> F2 -> F3 -> F4 -> F5 (Acoustic Waveguide)
        var r1 = new CascadeResonator();
        var r2 = new CascadeResonator();
        var r3 = new CascadeResonator();
        var r4 = new CascadeResonator();
        var r5 = new CascadeResonator();
        var rNasalPole = new CascadeResonator();
        var rNasalZero = new AntiResonator();

        // Parallel Resonators for turbulent friction & plosive bursts
        var rFricative = new BandpassResonator();
        var rBurst = new BandpassResonator();

        // Initial setup
        r1.SetParameters(500, 70, _sampleRate);
        r2.SetParameters(1500, 100, _sampleRate);
        r3.SetParameters(2500, 140, _sampleRate);
        r4.SetParameters(3600 * profile.FormantScale, 180, _sampleRate);
        r5.SetParameters(4500 * profile.FormantScale, 220, _sampleRate);
        rNasalPole.SetParameters(280, 80, _sampleRate);
        rNasalZero.SetParameters(1000, 120, _sampleRate);
        rFricative.SetParameters(5500, 1200, _sampleRate);
        rBurst.SetParameters(3500, 800, _sampleRate);

        float glottalPhase = 0.0f;
        float lipPrev = 0.0f;

        float curF1 = 500f, curF2 = 1500f, curF3 = 2500f;
        float curPitch = profile.BaseF0;

        // Physiological transition time constant (~32ms = 768 samples at 24kHz)
        const float FormantAlpha = 0.0013f;
        const float PitchAlpha = 0.0016f;

        var prng = new Random(42);

        float oq = profile.OpenQuotient;
        const float K_ret = 6.0f;
        float aRet = (oq * (2.0f / MathF.PI)) / ((1.0f - oq) * (1.0f - MathF.Exp(-K_ret)) / K_ret);

        for (int tIdx = 0; tIdx < tokens.Count; tIdx++)
        {
            var tok = tokens[tIdx];
            var spec = tok.Spec;
            int tokenSamples = (int)(tok.DurationSec * _sampleRate);
            if (tokenSamples <= 0) continue;

            if (tok.IsPause)
            {
                int decay = Math.Min(tokenSamples, 100);
                for (int s = 0; s < decay && (sampleCursor + s) < totalSamples; s++)
                {
                    float fade = 1.0f - (float)s / decay;
                    output[sampleCursor + s] *= fade;
                }
                sampleCursor += tokenSamples;
                glottalPhase = 0.0f;
                continue;
            }

            // Target formants with voice profile vocal-tract scaling
            float targetStartF1 = spec.F1 * (spec.IsVoiced ? profile.FormantScale : 1.0f);
            float targetStartF2 = spec.F2 * profile.FormantScale;
            float targetStartF3 = spec.F3 * profile.FormantScale;
            float targetEndF1 = spec.F1End * (spec.IsVoiced ? profile.FormantScale : 1.0f);
            float targetEndF2 = spec.F2End * profile.FormantScale;
            float targetEndF3 = spec.F3End * profile.FormantScale;

            float targetF4 = spec.F4 * profile.FormantScale;
            float targetF5 = spec.F5 * profile.FormantScale;
            float targetPitch = tok.TargetF0 > 20.0f ? tok.TargetF0 : profile.BaseF0;

            r4.SetParameters(targetF4, 180, _sampleRate);
            r5.SetParameters(targetF5, 220, _sampleRate);

            if (spec.IsFricative)
            {
                rFricative.SetParameters(spec.FricativeFreq, 1000, _sampleRate);
            }

            bool isPlosive = spec.IsPlosive;
            int closureSamples = isPlosive ? (int)(tokenSamples * 0.60f) : 0;
            int burstSamples = isPlosive ? Math.Min(tokenSamples - closureSamples, (int)(0.006f * _sampleRate)) : 0;

            if (isPlosive)
            {
                rBurst.SetParameters(spec.FricativeFreq, 900, _sampleRate);
            }

            for (int s = 0; s < tokenSamples && (sampleCursor + s) < totalSamples; s++)
            {
                float tNorm = (float)s / tokenSamples;

                // Diphthong dynamic formant glide across token duration
                float targetF1 = spec.IsDiphthong ? (targetStartF1 + (targetEndF1 - targetStartF1) * tNorm) : targetStartF1;
                float targetF2 = spec.IsDiphthong ? (targetStartF2 + (targetEndF2 - targetStartF2) * tNorm) : targetStartF2;
                float targetF3 = spec.IsDiphthong ? (targetStartF3 + (targetEndF3 - targetStartF3) * tNorm) : targetStartF3;

                // Smooth physiological articulatory tracking
                curF1 += (targetF1 - curF1) * FormantAlpha;
                curF2 += (targetF2 - curF2) * FormantAlpha;
                curF3 += (targetF3 - curF3) * FormantAlpha;
                curPitch += (targetPitch - curPitch) * PitchAlpha;

                // Update cascade oral resonators
                r1.SetParameters(curF1, spec.Bandwidth1, _sampleRate);
                r2.SetParameters(curF2, spec.Bandwidth2, _sampleRate);
                r3.SetParameters(curF3, spec.Bandwidth3, _sampleRate);

                // --- 1. Glottal Flow Excitation (Liljencrants-Fant derivative) ---
                float excitation = 0.0f;

                if (spec.IsVoiced)
                {
                    // Organic pitch jitter & vibrato
                    float vibrato = MathF.Sin(2.0f * MathF.PI * profile.VibratoRate * (sampleCursor + s) / _sampleRate) * profile.VibratoDepth;
                    float jitter = ((float)prng.NextDouble() * 2.0f - 1.0f) * profile.JitterAmount * curPitch;
                    float instPitch = Math.Max(50.0f, curPitch + vibrato + jitter);

                    float phaseInc = 2.0f * MathF.PI * instPitch / _sampleRate;
                    glottalPhase += phaseInc;
                    if (glottalPhase >= 2.0f * MathF.PI)
                    {
                        glottalPhase -= 2.0f * MathF.PI;
                    }

                    float cyclePos = glottalPhase / (2.0f * MathF.PI);
                    float glottalPulse;

                    if (cyclePos < oq)
                    {
                        // Open phase: positive flow rise derivative
                        float tau = cyclePos / oq;
                        glottalPulse = MathF.Sin(MathF.PI * tau);
                    }
                    else
                    {
                        // Return phase: rapid closure spike exciting vocal tract
                        float retPos = (cyclePos - oq) / (1.0f - oq);
                        glottalPulse = -aRet * MathF.Exp(-retPos * K_ret);
                    }

                    // Shimmer: amplitude perturbation
                    float shimmer = 1.0f + ((float)prng.NextDouble() * 2.0f - 1.0f) * profile.ShimmerAmount;

                    // Transglottal breathiness (turbulent aspiration noise)
                    float aspiration = ((float)prng.NextDouble() * 2.0f - 1.0f) * profile.Breathiness;

                    excitation = (glottalPulse * shimmer + aspiration) * spec.VoicingGain;

                    // Plosive closure voice bar
                    if (isPlosive && s < closureSamples)
                    {
                        excitation *= 0.12f;
                    }
                }

                // --- 2. Cascade Vocal Tract Oral Filtering ---
                float voicedAcoustic = 0.0f;
                if (spec.IsVoiced)
                {
                    // Series cascade through F1 -> F2 -> F3 -> F4 -> F5
                    float sOut = r1.Process(excitation);
                    sOut = r2.Process(sOut);
                    sOut = r3.Process(sOut);
                    sOut = r4.Process(sOut);
                    sOut = r5.Process(sOut);

                    // Nasal coupling: nasal pole + anti-resonance zero notch
                    if (spec.IsNasal)
                    {
                        float nasalP = rNasalPole.Process(sOut);
                        sOut = rNasalZero.Process(nasalP) * 0.65f;
                    }

                    voicedAcoustic = sOut;
                }

                // --- 3. Parallel Unvoiced Frication & Burst Generation ---
                float unvoicedAcoustic = 0.0f;
                if (spec.NoiseGain > 0.01f)
                {
                    float rawNoise = ((float)prng.NextDouble() * 2.0f - 1.0f);

                    if (isPlosive)
                    {
                        if (s >= closureSamples && s < closureSamples + burstSamples)
                        {
                            float burstEnv = 1.0f - (float)(s - closureSamples) / burstSamples;
                            float burstNoise = rawNoise * burstEnv * spec.NoiseGain * 2.2f;
                            unvoicedAcoustic = rBurst.Process(burstNoise);
                        }
                        else if (s >= closureSamples + burstSamples)
                        {
                            float votDecay = MathF.Max(0.0f, 1.0f - (float)(s - closureSamples - burstSamples) / (0.015f * _sampleRate));
                            unvoicedAcoustic = rawNoise * votDecay * spec.NoiseGain * 0.35f;
                        }
                    }
                    else
                    {
                        float fricNoise = rawNoise * spec.NoiseGain;
                        unvoicedAcoustic = rFricative.Process(fricNoise);
                    }
                }

                // --- 4. Acoustic Sum & Lip Radiation Differentiation ---
                float acousticTotal = voicedAcoustic + unvoicedAcoustic;

                // Lip radiation: +6dB/octave high-frequency acoustic differentiation
                float radiated = acousticTotal - 0.95f * lipPrev;
                lipPrev = acousticTotal;

                // --- 5. Natural Syllable Dynamics & Envelope ---
                float envelope = 1.0f;
                int edgeSamples = Math.Min(tokenSamples / 4, 80);
                if (s < edgeSamples)
                {
                    envelope = (float)s / edgeSamples;
                }
                else if (s > tokenSamples - edgeSamples)
                {
                    envelope = (float)(tokenSamples - s) / edgeSamples;
                }

                float sampleVal = radiated * envelope * tok.StressEmphasis * 0.50f;

                // --- 6. Soft Analogue Tube Saturation ---
                float saturated = MathF.Tanh(sampleVal * profile.Warmth);

                output[sampleCursor + s] = Math.Clamp(saturated, -1.0f, 1.0f);
            }

            sampleCursor += tokenSamples;
        }

        // Gentle studio mastering de-emphasis to preserve warmth without sibilance harshness
        if (totalSamples > 4)
        {
            float prev = output[0];
            for (int i = 1; i < totalSamples; i++)
            {
                float cur = output[i];
                output[i] = 0.85f * cur + 0.15f * prev;
                prev = cur;
            }
        }

        return output;
    }
}
