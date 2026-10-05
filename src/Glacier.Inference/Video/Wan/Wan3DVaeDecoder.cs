namespace Glacier.Inference.Video.Wan;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Format;

/// <summary>
/// Production Pure C# 3D Spatio-Temporal Causal Variational Autoencoder (VAE) Decoder for Wan 2.1 Video.
/// Reconstructs photorealistic RGB video frame sequences from 16-channel spatio-temporal latents
/// using the official 194-tensor 3D Causal Autoencoder (wan_2.1_vae.safetensors).
/// Applies causal 3D convolutions with temporal history caching, RMSNorm, 2D self-attention,
/// and spatial/temporal progressive pixel resampling in 100% pure C# .NET 10.
/// </summary>
public unsafe sealed partial class Wan3DVaeDecoder : IDisposable
{
    private readonly SafetensorsFile _weights;
    private readonly Dictionary<string, float[]> _tensorCache = new(StringComparer.Ordinal);
    private bool _disposed;

    // Normalization constants from official Wan 2.1 AutoencoderKLWan specification
    public static readonly float[] LatentsMean =
    [
        -0.7571f, -0.7089f, -0.9113f,  0.1075f,
        -0.1745f,  0.9653f, -0.1517f,  1.5508f,
         0.4134f, -0.0715f,  0.5517f, -0.3632f,
        -0.1922f, -0.9497f,  0.2503f, -0.2921f
    ];

    public static readonly float[] LatentsStd =
    [
        2.8184f, 1.4541f, 2.3275f, 2.6558f,
        1.2196f, 1.7708f, 2.6052f, 2.0743f,
        3.2687f, 2.1526f, 2.8652f, 1.5579f,
        1.6382f, 1.1253f, 2.8251f, 1.9160f
    ];

    public Wan3DVaeDecoder(SafetensorsFile weights)
    {
        _weights = weights;
        PreloadWeights();
    }

    public static Wan3DVaeDecoder Open(string safetensorsPath, bool enableGpu = true)
    {
        var sf = SafetensorsFile.Open(safetensorsPath);
        return new Wan3DVaeDecoder(sf);
    }

    private void PreloadWeights()
    {
        foreach (var name in _weights.Tensors.Keys)
        {
            if (name.StartsWith("decoder.") || name.StartsWith("conv2."))
            {
                _tensorCache[name] = LoadTensorFloats(_weights, name);
            }
        }
    }

    private float[] GetW(string name)
    {
        if (_tensorCache.TryGetValue(name, out var arr)) return arr;
        var loaded = LoadTensorFloats(_weights, name);
        _tensorCache[name] = loaded;
        return loaded;
    }

    private float[]? TryGetW(string name)
    {
        if (_tensorCache.TryGetValue(name, out var arr)) return arr;
        if (!_weights.ContainsTensor(name)) return null;
        var loaded = LoadTensorFloats(_weights, name);
        _tensorCache[name] = loaded;
        return loaded;
    }

    private static float[] LoadTensorFloats(SafetensorsFile sf, string name)
    {
        if (!sf.Tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Tensor '{name}' not found in Safetensors file.");

        int count = (int)info.ElementCount;
        float[] result = new float[count];
        byte* ptr = sf.GetTensorPointer(name);

        if (info.Dtype == "BF16")
        {
            ushort* uPtr = (ushort*)ptr;
            for (int i = 0; i < count; i++)
            {
                result[i] = BitConverter.UInt32BitsToSingle((uint)uPtr[i] << 16);
            }
        }
        else if (info.Dtype == "F16")
        {
            Half* hPtr = (Half*)ptr;
            for (int i = 0; i < count; i++)
            {
                result[i] = (float)hPtr[i];
            }
        }
        else if (info.Dtype == "F32")
        {
            float* fPtr = (float*)ptr;
            for (int i = 0; i < count; i++)
            {
                result[i] = fPtr[i];
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported tensor dtype: {info.Dtype}");
        }

        return result;
    }

    /// <summary>
    /// Decodes spatio-temporal latents [T_lat, 16, H_lat, W_lat] into continuous 24-bit sRGB video frames.
    /// </summary>
    public List<byte[]> DecodeVideo(
        ReadOnlySpan<float> spatioTemporalLatents,
        int temporalLatentFrames,
        int targetFrames,
        int latentH,
        int latentW)
    {
        int spatialLatent = latentH * latentW;
        int frameLatentSize = 16 * spatialLatent;

        // 1. Denormalize latents from DiT normalized space to VAE input space:
        // z[c] = latents[c] * LatentsStd[c] + LatentsMean[c]
        float[] z = new float[temporalLatentFrames * frameLatentSize];
        for (int f = 0; f < temporalLatentFrames; f++)
        {
            int fOff = f * frameLatentSize;
            for (int c = 0; c < 16; c++)
            {
                float mean = LatentsMean[c];
                float std = LatentsStd[c];
                int cOff = fOff + c * spatialLatent;
                for (int s = 0; s < spatialLatent; s++)
                {
                    z[cOff + s] = spatioTemporalLatents[cOff + s] * std + mean;
                }
            }
        }

        // 2. Post-Quant Convolution (conv2: 16 -> 16 channel projection)
        float[] postQuant = new float[temporalLatentFrames * frameLatentSize];
        float[] conv2W = GetW("conv2.weight");
        float[] conv2B = GetW("conv2.bias");

        for (int f = 0; f < temporalLatentFrames; f++)
        {
            int fOff = f * frameLatentSize;
            fixed (float* pSrc = &z[fOff], pDst = &postQuant[fOff], pW = conv2W, pB = conv2B)
            {
                Conv2D_1x1(pSrc, pDst, pW, pB, 16, 16, spatialLatent);
            }
        }

        // 3. Causal chunked decoding loop across time frames
        CausalFeatureCache cache = new();
        List<byte[]> outputFrames = new(targetFrames);

        for (int chunkIdx = 0; chunkIdx < temporalLatentFrames; chunkIdx++)
        {
            bool firstChunk = (chunkIdx == 0);

            // Extract single frame latent slice [16, latentH, latentW]
            float[] chunkInput = new float[frameLatentSize];
            Array.Copy(postQuant, chunkIdx * frameLatentSize, chunkInput, 0, frameLatentSize);

            // Forward through 3D Causal Decoder
            List<float[]> decodedChunks = DecodeChunk(chunkInput, latentH, latentW, cache, firstChunk);

            int finalH = latentH * 8;
            int finalW = latentW * 8;

            foreach (var framePlanar in decodedChunks)
            {
                byte[] rgb = new byte[finalH * finalW * 3];
                fixed (float* pPlanar = framePlanar)
                fixed (byte* pRgb = rgb)
                {
                    ClampToRgb(pPlanar, pRgb, finalH, finalW);
                }
                outputFrames.Add(rgb);
            }

            if (outputFrames.Count >= targetFrames)
            {
                break;
            }
        }

        if (outputFrames.Count > targetFrames)
        {
            return outputFrames.GetRange(0, targetFrames);
        }

        return outputFrames;
    }

    private List<float[]> DecodeChunk(
        float[] chunkInput,
        int latentH,
        int latentW,
        CausalFeatureCache cache,
        bool firstChunk)
    {
        // Initial 3D Causal Conv: decoder.conv1 (16 -> 384, kernel 3x3x3)
        int curH = latentH;
        int curW = latentW;
        int curC = 384;
        List<float[]> frames = RunCausalConv3D([chunkInput], 16, curC, curH, curW, "decoder.conv1", cache);

        // Middle Block
        // ResBlock 0 (384 -> 384)
        frames = RunResNetBlock(frames, curC, curC, curH, curW, "decoder.middle.0", cache);
        // Self-Attention (384)
        frames = RunSpatialAttention(frames, curC, curH, curW, "decoder.middle.1");
        // ResBlock 2 (384 -> 384)
        frames = RunResNetBlock(frames, curC, curC, curH, curW, "decoder.middle.2", cache);

        // Upsamples Block 0: upsamples.0, 1, 2, then upsampler 3
        frames = RunResNetBlock(frames, 384, 384, curH, curW, "decoder.upsamples.0", cache);
        frames = RunResNetBlock(frames, 384, 384, curH, curW, "decoder.upsamples.1", cache);
        frames = RunResNetBlock(frames, 384, 384, curH, curW, "decoder.upsamples.2", cache);

        // Upsampler 3: Resample3D (time_conv: 384 -> 768; resample.1: 384 -> 192)
        frames = RunResample3D(frames, 384, 192, ref curH, ref curW, "decoder.upsamples.3", cache, firstChunk);
        curC = 192;

        // Upsamples Block 1: upsamples.4 (with shortcut 192 -> 384), 5, 6, then upsampler 7
        frames = RunResNetBlockWithShortcut(frames, 192, 384, curH, curW, "decoder.upsamples.4", cache);
        frames = RunResNetBlock(frames, 384, 384, curH, curW, "decoder.upsamples.5", cache);
        frames = RunResNetBlock(frames, 384, 384, curH, curW, "decoder.upsamples.6", cache);

        // Upsampler 7: Resample3D (time_conv: 384 -> 768; resample.1: 384 -> 192)
        frames = RunResample3D(frames, 384, 192, ref curH, ref curW, "decoder.upsamples.7", cache, firstChunk);
        curC = 192;

        // Upsamples Block 2: upsamples.8, 9, 10, then upsampler 11 (Resample2D)
        frames = RunResNetBlock(frames, 192, 192, curH, curW, "decoder.upsamples.8", cache);
        frames = RunResNetBlock(frames, 192, 192, curH, curW, "decoder.upsamples.9", cache);
        frames = RunResNetBlock(frames, 192, 192, curH, curW, "decoder.upsamples.10", cache);

        // Upsampler 11: Resample2D (spatial 2x nearest-neighbor + conv2d 192 -> 96)
        frames = RunResample2D(frames, 192, 96, ref curH, ref curW, "decoder.upsamples.11");
        curC = 96;

        // Upsamples Block 3: upsamples.12, 13, 14 (96 -> 96, spatial 240x240)
        frames = RunResNetBlock(frames, 96, 96, curH, curW, "decoder.upsamples.12", cache);
        frames = RunResNetBlock(frames, 96, 96, curH, curW, "decoder.upsamples.13", cache);
        frames = RunResNetBlock(frames, 96, 96, curH, curW, "decoder.upsamples.14", cache);

        // Head: decoder.head (norm_out 96 + silu + conv_out 96 -> 3)
        float[] normGamma = GetW("decoder.head.0.gamma");
        List<float[]> normedFrames = new(frames.Count);
        int spatial = curH * curW;

        foreach (var sf in frames)
        {
            float[] normed = new float[curC * spatial];
            fixed (float* pSrc = sf, pDst = normed, pGamma = normGamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, curC, spatial, applySilu: true);
            }
            normedFrames.Add(normed);
        }

        // decoder.head.2 (3D causal conv 96 -> 3, kernel 3x3x3)
        List<float[]> rgbFrames = RunCausalConv3D(normedFrames, 96, 3, curH, curW, "decoder.head.2", cache);
        return rgbFrames;
    }

    private List<float[]> RunCausalConv3D(
        List<float[]> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        CausalFeatureCache cache)
    {
        var (cachedT0, cachedT1) = cache.GetPastFrames(prefix);

        float[] weight = GetW($"{prefix}.weight");
        float[]? bias = TryGetW($"{prefix}.bias");
        int count = inputFrames.Count;
        List<float[]> output = new(count);

        for (int k = 0; k < count; k++)
        {
            // Identify the 3 temporal frames (t0, t1, t2) for causal window
            float[]? t0;
            float[]? t1;
            float[] t2 = inputFrames[k];

            if (k == 0)
            {
                t0 = cachedT0;
                t1 = cachedT1;
            }
            else if (k == 1)
            {
                t0 = cachedT1;
                t1 = inputFrames[0];
            }
            else
            {
                t0 = inputFrames[k - 2];
                t1 = inputFrames[k - 1];
            }

            float[] dst = new float[outC * H * W];
            fixed (float* pT0 = t0, pT1 = t1, pT2 = t2, pW = weight, pB = bias, pDst = dst)
            {
                Conv3D_Causal(pT0, pT1, pT2, pDst, pW, pB, inC, outC, H, W);
            }
            output.Add(dst);
        }

        // Update cache with the last 2 frames of the sequence
        if (count == 1)
        {
            cache.Update(prefix, cachedT1, inputFrames[0]);
        }
        else
        {
            cache.Update(prefix, inputFrames[^2], inputFrames[^1]);
        }

        return output;
    }

    private List<float[]> RunResNetBlock(
        List<float[]> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        CausalFeatureCache cache)
    {
        int spatial = H * W;

        // 1. norm1 + silu on all frames
        float[] norm1Gamma = GetW($"{prefix}.residual.0.gamma");
        List<float[]> norm1Frames = new(inputFrames.Count);
        foreach (var frame in inputFrames)
        {
            float[] norm1 = new float[inC * spatial];
            fixed (float* pSrc = frame, pDst = norm1, pGamma = norm1Gamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, inC, spatial, applySilu: true);
            }
            norm1Frames.Add(norm1);
        }

        // 2. conv1 (3D causal conv inC -> outC)
        List<float[]> conv1Frames = RunCausalConv3D(norm1Frames, inC, outC, H, W, $"{prefix}.residual.2", cache);

        // 3. norm2 + silu on all frames
        float[] norm2Gamma = GetW($"{prefix}.residual.3.gamma");
        List<float[]> norm2Frames = new(conv1Frames.Count);
        foreach (var frame in conv1Frames)
        {
            float[] norm2 = new float[outC * spatial];
            fixed (float* pSrc = frame, pDst = norm2, pGamma = norm2Gamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, outC, spatial, applySilu: true);
            }
            norm2Frames.Add(norm2);
        }

        // 4. conv2 (3D causal conv outC -> outC)
        List<float[]> conv2Frames = RunCausalConv3D(norm2Frames, outC, outC, H, W, $"{prefix}.residual.6", cache);

        // 5. Add residual identity
        for (int k = 0; k < conv2Frames.Count; k++)
        {
            float[] outFrame = conv2Frames[k];
            float[] inFrame = inputFrames[k];
            for (int i = 0; i < outC * spatial; i++)
            {
                outFrame[i] += inFrame[i];
            }
        }

        return conv2Frames;
    }

    private List<float[]> RunResNetBlockWithShortcut(
        List<float[]> inputFrames,
        int inC,
        int outC,
        int H,
        int W,
        string prefix,
        CausalFeatureCache cache)
    {
        int spatial = H * W;
        float[] shortW = GetW($"{prefix}.shortcut.weight");
        float[]? shortB = TryGetW($"{prefix}.shortcut.bias");

        // 1. Shortcut projection on all frames
        List<float[]> shortcutFrames = new(inputFrames.Count);
        foreach (var frame in inputFrames)
        {
            float[] shortcut = new float[outC * spatial];
            fixed (float* pSrc = frame, pDst = shortcut, pW = shortW, pB = shortB)
            {
                Conv2D_1x1(pSrc, pDst, pW, pB, inC, outC, spatial);
            }
            shortcutFrames.Add(shortcut);
        }

        // 2. norm1 + silu on all frames
        float[] norm1Gamma = GetW($"{prefix}.residual.0.gamma");
        List<float[]> norm1Frames = new(inputFrames.Count);
        foreach (var frame in inputFrames)
        {
            float[] norm1 = new float[inC * spatial];
            fixed (float* pSrc = frame, pDst = norm1, pGamma = norm1Gamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, inC, spatial, applySilu: true);
            }
            norm1Frames.Add(norm1);
        }

        // 3. conv1 (3D causal conv inC -> outC)
        List<float[]> conv1Frames = RunCausalConv3D(norm1Frames, inC, outC, H, W, $"{prefix}.residual.2", cache);

        // 4. norm2 + silu on all frames
        float[] norm2Gamma = GetW($"{prefix}.residual.3.gamma");
        List<float[]> norm2Frames = new(conv1Frames.Count);
        foreach (var frame in conv1Frames)
        {
            float[] norm2 = new float[outC * spatial];
            fixed (float* pSrc = frame, pDst = norm2, pGamma = norm2Gamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, outC, spatial, applySilu: true);
            }
            norm2Frames.Add(norm2);
        }

        // 5. conv2 (3D causal conv outC -> outC)
        List<float[]> conv2Frames = RunCausalConv3D(norm2Frames, outC, outC, H, W, $"{prefix}.residual.6", cache);

        // 6. Add shortcut
        for (int k = 0; k < conv2Frames.Count; k++)
        {
            float[] outFrame = conv2Frames[k];
            float[] scFrame = shortcutFrames[k];
            for (int i = 0; i < outC * spatial; i++)
            {
                outFrame[i] += scFrame[i];
            }
        }

        return conv2Frames;
    }

    private List<float[]> RunSpatialAttention(List<float[]> inputFrames, int channels, int H, int W, string prefix)
    {
        int spatial = H * W;
        float[] normGamma = GetW($"{prefix}.norm.gamma");
        float[] qkvW = GetW($"{prefix}.to_qkv.weight");
        float[]? qkvB = TryGetW($"{prefix}.to_qkv.bias");
        float[] projW = GetW($"{prefix}.proj.weight");
        float[]? projB = TryGetW($"{prefix}.proj.bias");
        float scale = 1.0f / MathF.Sqrt(channels);

        List<float[]> outputFrames = new(inputFrames.Count);

        foreach (var frame in inputFrames)
        {
            // 1. RMSNorm
            float[] normed = new float[channels * spatial];
            fixed (float* pSrc = frame, pDst = normed, pGamma = normGamma)
            {
                WanRMSNorm(pSrc, pDst, pGamma, channels, spatial, applySilu: false);
            }

            // 2. to_qkv (1x1 conv channels -> channels * 3)
            float[] qkv = new float[channels * 3 * spatial];
            fixed (float* pSrc = normed, pDst = qkv, pW = qkvW, pB = qkvB)
            {
                Conv2D_1x1(pSrc, pDst, pW, pB, channels, channels * 3, spatial);
            }

            // 3. Scaled dot-product spatial attention (single head) with contiguous SIMD
            float[] attnOut = new float[channels * spatial];
            float[] qTok = new float[spatial * channels];
            float[] kTok = new float[spatial * channels];
            float[] vTok = new float[spatial * channels];
            float[] attnTok = new float[spatial * channels];

            fixed (float* pQkv = qkv, pQtok = qTok, pKtok = kTok, pVtok = vTok, pAttnTok = attnTok, pAttn = attnOut)
            {
                nint qAddr = (nint)pQkv;
                nint kAddr = (nint)(pQkv + (long)channels * spatial);
                nint vAddr = (nint)(pQkv + (long)channels * 2 * spatial);
                nint qTokAddr = (nint)pQtok;
                nint kTokAddr = (nint)pKtok;
                nint vTokAddr = (nint)pVtok;
                nint attnTokAddr = (nint)pAttnTok;
                nint attnAddr = (nint)pAttn;

                // Transpose to token-major [spatial, channels] for contiguous cache streaming
                Parallel.For(0, spatial, i =>
                {
                    float* pQ = (float*)qAddr;
                    float* pK = (float*)kAddr;
                    float* pV = (float*)vAddr;
                    float* dstQ = (float*)qTokAddr + (long)i * channels;
                    float* dstK = (float*)kTokAddr + (long)i * channels;
                    float* dstV = (float*)vTokAddr + (long)i * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        long srcIdx = (long)c * spatial + i;
                        dstQ[c] = pQ[srcIdx];
                        dstK[c] = pK[srcIdx];
                        dstV[c] = pV[srcIdx];
                    }
                });

                Parallel.For(0, spatial, i =>
                {
                    float* pQi = (float*)qTokAddr + (long)i * channels;
                    float* pKtok = (float*)kTokAddr;
                    float* pVtok = (float*)vTokAddr;
                    float* pDstI = (float*)attnTokAddr + (long)i * channels;
                    new Span<float>(pDstI, channels).Clear();

                    float m = float.NegativeInfinity;
                    float l = 0f;

                    for (int j = 0; j < spatial; j++)
                    {
                        float* pKj = pKtok + (long)j * channels;
                        float dot = 0f;
                        int c = 0;
                        if (Vector256.IsHardwareAccelerated)
                        {
                            var vSum = Vector256<float>.Zero;
                            for (; c <= channels - 8; c += 8)
                            {
                                var qVec = Vector256.Load(pQi + c);
                                var kVec = Vector256.Load(pKj + c);
                                vSum = Vector256.FusedMultiplyAdd(qVec, kVec, vSum);
                            }
                            dot = Vector256.Sum(vSum);
                        }
                        for (; c < channels; c++)
                        {
                            dot += pQi[c] * pKj[c];
                        }

                        float s_j = dot * scale;
                        float m_new = MathF.Max(m, s_j);
                        float alpha = MathF.Exp(m - m_new);
                        float w_j = MathF.Exp(s_j - m_new);

                        float* pVj = pVtok + (long)j * channels;
                        c = 0;
                        if (Vector256.IsHardwareAccelerated)
                        {
                            var vAlpha = Vector256.Create(alpha);
                            var vW = Vector256.Create(w_j);
                            for (; c <= channels - 8; c += 8)
                            {
                                var vDst = Vector256.Load(pDstI + c);
                                var vVal = Vector256.Load(pVj + c);
                                vDst = Vector256.Multiply(vDst, vAlpha);
                                vDst = Vector256.FusedMultiplyAdd(vW, vVal, vDst);
                                vDst.Store(pDstI + c);
                            }
                        }
                        for (; c < channels; c++)
                        {
                            pDstI[c] = pDstI[c] * alpha + w_j * pVj[c];
                        }

                        l = l * alpha + w_j;
                        m = m_new;
                    }

                    if (l > 0f)
                    {
                        float invL = 1.0f / l;
                        int c = 0;
                        if (Vector256.IsHardwareAccelerated)
                        {
                            var vInvL = Vector256.Create(invL);
                            for (; c <= channels - 8; c += 8)
                            {
                                var vDst = Vector256.Load(pDstI + c);
                                vDst = Vector256.Multiply(vDst, vInvL);
                                vDst.Store(pDstI + c);
                            }
                        }
                        for (; c < channels; c++)
                        {
                            pDstI[c] *= invL;
                        }
                    }
                });

                // Transpose back to channel-first [channels, spatial]
                Parallel.For(0, spatial, i =>
                {
                    float* srcI = (float*)attnTokAddr + (long)i * channels;
                    float* pDstAttn = (float*)attnAddr;
                    for (int c = 0; c < channels; c++)
                    {
                        pDstAttn[(long)c * spatial + i] = srcI[c];
                    }
                });
            }

            // 4. proj (1x1 conv channels -> channels)
            float[] projOut = new float[channels * spatial];
            fixed (float* pSrc = attnOut, pDst = projOut, pW = projW, pB = projB)
            {
                Conv2D_1x1(pSrc, pDst, pW, pB, channels, channels, spatial);
            }

            // 5. Add residual
            for (int i = 0; i < channels * spatial; i++)
            {
                projOut[i] += frame[i];
            }

            outputFrames.Add(projOut);
        }

        return outputFrames;
    }


    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _tensorCache.Clear();
            _weights.Dispose();
        }
    }

    private sealed class CausalFeatureCache
    {
        private readonly Dictionary<string, (float[]? t0, float[]? t1)> _slots = new(StringComparer.Ordinal);

        public (float[]? t0, float[]? t1) GetPastFrames(string prefix)
        {
            if (_slots.TryGetValue(prefix, out var pair)) return pair;
            return (null, null);
        }

        public void MarkTimeConvRep(string prefix)
        {
            _slots[prefix] = (null, null);
        }

        public void Update(string prefix, float[]? t0, float[]? t1)
        {
            _slots[prefix] = (t0, t1);
        }
    }
}
