namespace Glacier.Inference.Audio;

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// High-performance SIMD Log-Mel Spectrogram DSP processor for speech recognition (Whisper / SenseVoice).
/// Pure C# implementation with zero heap allocations on inner hot paths.
/// </summary>
public sealed unsafe class MelSpectrogram : IDisposable
{
    public const int DefaultSampleRate = 16000;
    public const int DefaultNfft = 512;
    public const int DefaultHopLength = 160;   // 10ms at 16kHz
    public const int DefaultWinLength = 400;   // 25ms at 16kHz
    public const int DefaultNMels = 80;        // 80 for Whisper base/small/medium/turbo, 128 for large-v3

    public int SampleRate { get; }
    public int NFft { get; }
    public int HopLength { get; }
    public int WinLength { get; }
    public int NMels { get; }
    public int FftBins => NFft / 2 + 1; // 257 bins

    private readonly float[] _window;
    private readonly float[] _melFilters; // [NMels, FftBins] flattened
    private readonly int[] _bitReverse;
    private readonly float[] _twiddleCos;
    private readonly float[] _twiddleSin;

    // Preallocated scratch memory
    private float* _fftReal;
    private float* _fftImag;
    private float* _powerSpectrum;
    private bool _disposed;

    public MelSpectrogram(
        int sampleRate = DefaultSampleRate,
        int nMels = DefaultNMels,
        int nFft = DefaultNfft,
        int hopLength = DefaultHopLength,
        int winLength = DefaultWinLength)
    {
        SampleRate = sampleRate;
        NMels = nMels;
        NFft = nFft;
        HopLength = hopLength;
        WinLength = winLength;

        _window = BuildHannWindow(WinLength);
        _melFilters = BuildMelFilterbank(NMels, FftBins, SampleRate, 0.0f, 8000.0f);
        
        // Precompute FFT structures for size 512
        _bitReverse = BuildBitReverseTable(NFft);
        _twiddleCos = new float[NFft / 2];
        _twiddleSin = new float[NFft / 2];
        for (int i = 0; i < NFft / 2; i++)
        {
            double angle = -2.0 * Math.PI * i / NFft;
            _twiddleCos[i] = (float)Math.Cos(angle);
            _twiddleSin[i] = (float)Math.Sin(angle);
        }

        // Allocate unmanaged 64-byte aligned scratch buffers
        _fftReal = (float*)NativeMemory.AlignedAlloc((nuint)(NFft * sizeof(float)), 64);
        _fftImag = (float*)NativeMemory.AlignedAlloc((nuint)(NFft * sizeof(float)), 64);
        _powerSpectrum = (float*)NativeMemory.AlignedAlloc((nuint)(FftBins * sizeof(float)), 64);
    }

    /// <summary>
    /// Processes a stream of 16kHz audio samples into a log-mel spectrogram tensor [nMels, nFrames].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Process(ReadOnlySpan<float> audio, Span<float> outputMel)
    {
        int nFrames = (audio.Length - WinLength) / HopLength + 1;
        if (nFrames <= 0) return;
        if (outputMel.Length < NMels * nFrames)
        {
            throw new ArgumentException($"Output buffer length ({outputMel.Length}) is too small for {NMels}x{nFrames} = {NMels * nFrames} elements.");
        }

        fixed (float* pAudio = audio)
        fixed (float* pWindow = _window)
        fixed (float* pMelFilters = _melFilters)
        fixed (float* pOutMel = outputMel)
        {
            float maxLogVal = -1e30f;

            for (int f = 0; f < nFrames; f++)
            {
                int audioOffset = f * HopLength;

                // 1. Apply Hann window and zero pad to NFft
                new Span<float>(_fftReal, NFft).Clear();
                new Span<float>(_fftImag, NFft).Clear();

                int copyLen = Math.Min(WinLength, audio.Length - audioOffset);
                int i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    int vecLimit = copyLen - 8;
                    for (; i <= vecLimit; i += 8)
                    {
                        var vAudio = Vector256.Load(pAudio + audioOffset + i);
                        var vWin = Vector256.Load(pWindow + i);
                        (vAudio * vWin).Store(_fftReal + i);
                    }
                }
                for (; i < copyLen; i++)
                {
                    _fftReal[i] = pAudio[audioOffset + i] * pWindow[i];
                }

                // 2. Perform in-place Radix-2 FFT
                ComputeFft(_fftReal, _fftImag, NFft);

                // 3. Compute Power Spectrum: |X|^2 = real^2 + imag^2
                int b = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    int vecLimit = FftBins - 8;
                    for (; b <= vecLimit; b += 8)
                    {
                        var vr = Vector256.Load(_fftReal + b);
                        var vi = Vector256.Load(_fftImag + b);
                        (vr * vr + vi * vi).Store(_powerSpectrum + b);
                    }
                }
                for (; b < FftBins; b++)
                {
                    float r = _fftReal[b];
                    float im = _fftImag[b];
                    _powerSpectrum[b] = r * r + im * im;
                }

                // 4. Multiply with Mel filterbank matrix: mel[m, f] = dot(filter[m], powerSpectrum)
                for (int m = 0; m < NMels; m++)
                {
                    float* filterRow = pMelFilters + m * FftBins;
                    float sum = 0.0f;

                    int k = 0;
                    if (Vector256.IsHardwareAccelerated)
                    {
                        var vAcc = Vector256<float>.Zero;
                        int vecLimit = FftBins - 8;
                        for (; k <= vecLimit; k += 8)
                        {
                            var vFilt = Vector256.Load(filterRow + k);
                            var vPow = Vector256.Load(_powerSpectrum + k);
                            vAcc += vFilt * vPow;
                        }
                        sum = Vector256.Sum(vAcc);
                    }
                    for (; k < FftBins; k++)
                    {
                        sum += filterRow[k] * _powerSpectrum[k];
                    }

                    // 5. Log compression: log10(max(sum, 1e-5))
                    float logVal = MathF.Log10(Math.Max(sum, 1e-5f));
                    pOutMel[m * nFrames + f] = logVal;

                    if (logVal > maxLogVal)
                    {
                        maxLogVal = logVal;
                    }
                }
            }

            // 6. Normalize dynamic range: clamp to [max - 8.0, max] then (x + 4.0) / 4.0
            float clampMin = maxLogVal - 8.0f;
            int totalElements = NMels * nFrames;
            int j = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var vClampMin = Vector256.Create(clampMin);
                var vFour = Vector256.Create(4.0f);
                var vInvFour = Vector256.Create(0.25f);
                int vecLimit = totalElements - 8;
                for (; j <= vecLimit; j += 8)
                {
                    var v = Vector256.Load(pOutMel + j);
                    var vClamped = Vector256.Max(v, vClampMin);
                    ((vClamped + vFour) * vInvFour).Store(pOutMel + j);
                }
            }
            for (; j < totalElements; j++)
            {
                float v = Math.Max(pOutMel[j], clampMin);
                pOutMel[j] = (v + 4.0f) * 0.25f;
            }
        }
    }

    /// <summary>
    /// Computes in-place Radix-2 Cooley-Tukey FFT.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeFft(float* real, float* imag, int n)
    {
        // 1. Bit-reversal permutation
        for (int i = 0; i < n; i++)
        {
            int j = _bitReverse[i];
            if (i < j)
            {
                float tempR = real[i]; real[i] = real[j]; real[j] = tempR;
                float tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
            }
        }

        // 2. Butterfly stages
        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            int step = n / len;

            for (int i = 0; i < n; i += len)
            {
                for (int k = 0; k < half; k++)
                {
                    int twiddleIdx = k * step;
                    float c = _twiddleCos[twiddleIdx];
                    float s = _twiddleSin[twiddleIdx];

                    int uIdx = i + k;
                    int vIdx = i + k + half;

                    float vReal = real[vIdx];
                    float vImag = imag[vIdx];

                    // Complex multiply: t = v * W
                    float tReal = vReal * c - vImag * s;
                    float tImag = vReal * s + vImag * c;

                    float uReal = real[uIdx];
                    float uImag = imag[uIdx];

                    real[uIdx] = uReal + tReal;
                    imag[uIdx] = uImag + tImag;
                    real[vIdx] = uReal - tReal;
                    imag[vIdx] = uImag - tImag;
                }
            }
        }
    }

    private static float[] BuildHannWindow(int length)
    {
        var win = new float[length];
        double factor = 2.0 * Math.PI / length;
        for (int i = 0; i < length; i++)
        {
            win[i] = (float)(0.5 * (1.0 - Math.Cos(factor * i)));
        }
        return win;
    }

    private static int[] BuildBitReverseTable(int n)
    {
        int bits = (int)Math.Log2(n);
        var table = new int[n];
        for (int i = 0; i < n; i++)
        {
            int rev = 0;
            int val = i;
            for (int b = 0; b < bits; b++)
            {
                rev = (rev << 1) | (val & 1);
                val >>= 1;
            }
            table[i] = rev;
        }
        return table;
    }

    private static float HzToMel(float hz) => 2595.0f * MathF.Log10(1.0f + hz / 700.0f);
    private static float MelToHz(float mel) => 700.0f * (MathF.Pow(10.0f, mel / 2595.0f) - 1.0f);

    private static float[] BuildMelFilterbank(int nMels, int nFftBins, int sampleRate, float fMin, float fMax)
    {
        var filters = new float[nMels * nFftBins];
        float melMin = HzToMel(fMin);
        float melMax = HzToMel(fMax);

        var melPoints = new float[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            melPoints[i] = MelToHz(melMin + (melMax - melMin) * i / (nMels + 1));
        }

        var fftFreqs = new float[nFftBins];
        for (int i = 0; i < nFftBins; i++)
        {
            fftFreqs[i] = (float)i * sampleRate / ((nFftBins - 1) * 2);
        }

        for (int m = 0; m < nMels; m++)
        {
            float fPrev = melPoints[m];
            float fCurr = melPoints[m + 1];
            float fNext = melPoints[m + 2];

            for (int i = 0; i < nFftBins; i++)
            {
                float freq = fftFreqs[i];
                float weight = 0.0f;

                if (freq >= fPrev && freq <= fCurr)
                {
                    weight = (freq - fPrev) / (fCurr - fPrev);
                }
                else if (freq > fCurr && freq <= fNext)
                {
                    weight = (fNext - freq) / (fNext - fCurr);
                }

                // Slaney normalization: 2 / (fNext - fPrev)
                float norm = 2.0f / (fNext - fPrev);
                filters[m * nFftBins + i] = weight * norm;
            }
        }

        return filters;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_fftReal != null) { NativeMemory.AlignedFree(_fftReal); _fftReal = null; }
            if (_fftImag != null) { NativeMemory.AlignedFree(_fftImag); _fftImag = null; }
            if (_powerSpectrum != null) { NativeMemory.AlignedFree(_powerSpectrum); _powerSpectrum = null; }
        }
    }
}
