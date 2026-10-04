namespace Glacier.Inference.Video.Wan;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.Inference.Format;

/// <summary>
/// Production Pure C# 3D Spatio-Temporal Causal Variational Autoencoder (VAE) Decoder for Wan 2.1 Video.
/// Reconstructs photorealistic RGB video frame sequences from 16-channel spatio-temporal latents
/// using the official 194-tensor 3D Causal Autoencoder (wan_2.1_vae.safetensors).
/// Applies causal 3D convolutions with temporal history caching, RMSNorm, 2D self-attention,
/// and spatial/temporal progressive pixel resampling in 100% pure C# .NET 10.
/// </summary>
public unsafe sealed class Wan3DVaeDecoder : IDisposable
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

                    float[] scoresArr = new float[spatial];
                    fixed (float* scores = scoresArr)
                    {
                        float maxScore = float.NegativeInfinity;

                        for (int j = 0; j < spatial; j++)
                        {
                            float* pKj = pKtok + (long)j * channels;
                            float dot = 0f;
                            int c = 0;
                            if (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated)
                            {
                                var vSum = System.Runtime.Intrinsics.Vector256<float>.Zero;
                                for (; c <= channels - 8; c += 8)
                                {
                                    var qVec = System.Runtime.Intrinsics.Vector256.Load(pQi + c);
                                    var kVec = System.Runtime.Intrinsics.Vector256.Load(pKj + c);
                                    vSum = System.Runtime.Intrinsics.Vector256.FusedMultiplyAdd(qVec, kVec, vSum);
                                }
                                dot = System.Runtime.Intrinsics.Vector256.Sum(vSum);
                            }
                            for (; c < channels; c++)
                            {
                                dot += pQi[c] * pKj[c];
                            }

                            dot *= scale;
                            scores[j] = dot;
                            if (dot > maxScore) maxScore = dot;
                        }

                        float sumExp = 0f;
                        for (int j = 0; j < spatial; j++)
                        {
                            float exp = MathF.Exp(scores[j] - maxScore);
                            scores[j] = exp;
                            sumExp += exp;
                        }
                        float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                        for (int j = 0; j < spatial; j++)
                        {
                            float s = scores[j] * invSum;
                            if (s == 0f) continue;
                            float* pVj = pVtok + (long)j * channels;

                            int c = 0;
                            if (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated)
                            {
                                var vs = System.Runtime.Intrinsics.Vector256.Create(s);
                                for (; c <= channels - 8; c += 8)
                                {
                                    var vDst = System.Runtime.Intrinsics.Vector256.Load(pDstI + c);
                                    var vVal = System.Runtime.Intrinsics.Vector256.Load(pVj + c);
                                    vDst = System.Runtime.Intrinsics.Vector256.FusedMultiplyAdd(vs, vVal, vDst);
                                    *(System.Runtime.Intrinsics.Vector256<float>*)(pDstI + c) = vDst;
                                }
                            }
                            for (; c < channels; c++)
                            {
                                pDstI[c] += s * pVj[c];
                            }
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

    private List<float[]> RunResample3D(
        List<float[]> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix,
        CausalFeatureCache cache,
        bool firstChunk)
    {
        List<float[]> timeSlices = [];

        if (firstChunk)
        {
            // On chunk 0, time_conv is bypassed, keeping t=1 frame
            cache.MarkTimeConvRep(prefix);
            timeSlices.AddRange(inputFrames);
        }
        else
        {
            // On subsequent chunks, time_conv doubles time dimension t -> 2t
            var (cachedT0, cachedT1) = cache.GetPastFrames(prefix);
            float[] timeConvW = GetW($"{prefix}.time_conv.weight");
            float[]? timeConvB = TryGetW($"{prefix}.time_conv.bias");

            for (int k = 0; k < inputFrames.Count; k++)
            {
                float[]? t0 = (k == 0) ? cachedT0 : ((k == 1) ? cachedT1 : inputFrames[k - 2]);
                float[]? t1 = (k == 0) ? cachedT1 : inputFrames[k - 1];
                float[] t2 = inputFrames[k];

                float[] timeConvDst = new float[inC * 2 * H * W];
                fixed (float* pT0 = t0, pT1 = t1, pT2 = t2, pW = timeConvW, pB = timeConvB, pDst = timeConvDst)
                {
                    Conv1x1_3D_Causal(pT0, pT1, pT2, pDst, pW, pB, inC, inC * 2, H, W);
                }

                // Channel interleave: lower inC channels -> Frame 0, upper inC channels -> Frame 1
                int frameSize = inC * H * W;
                float[] frame0 = new float[frameSize];
                float[] frame1 = new float[frameSize];
                Array.Copy(timeConvDst, 0, frame0, 0, frameSize);
                Array.Copy(timeConvDst, frameSize, frame1, 0, frameSize);

                timeSlices.Add(frame0);
                timeSlices.Add(frame1);
            }

            if (inputFrames.Count == 1)
            {
                cache.Update(prefix, cachedT1, inputFrames[0]);
            }
            else
            {
                cache.Update(prefix, inputFrames[^2], inputFrames[^1]);
            }
        }

        // Spatial 2x upsampling on each temporal slice
        int outH = H * 2;
        int outW = W * 2;
        float[] convW = GetW($"{prefix}.resample.1.weight");
        float[]? convB = TryGetW($"{prefix}.resample.1.bias");

        List<float[]> result = new(timeSlices.Count);
        foreach (var slice in timeSlices)
        {
            float[] upsampled = new float[inC * outH * outW];
            fixed (float* pSrc = slice, pDst = upsampled)
            {
                Upsample2x_Nearest(pSrc, pDst, inC, H, W);
            }

            float[] convOut = new float[outC * outH * outW];
            fixed (float* pSrc = upsampled, pDst = convOut, pW = convW, pB = convB)
            {
                Conv2D_3x3(pSrc, pDst, pW, pB, inC, outC, outH, outW);
            }
            result.Add(convOut);
        }

        H = outH;
        W = outW;
        return result;
    }

    private List<float[]> RunResample2D(
        List<float[]> inputFrames,
        int inC,
        int outC,
        ref int H,
        ref int W,
        string prefix)
    {
        int outH = H * 2;
        int outW = W * 2;
        float[] convW = GetW($"{prefix}.resample.1.weight");
        float[]? convB = TryGetW($"{prefix}.resample.1.bias");

        List<float[]> result = new(inputFrames.Count);
        foreach (var frame in inputFrames)
        {
            float[] upsampled = new float[inC * outH * outW];
            fixed (float* pSrc = frame, pDst = upsampled)
            {
                Upsample2x_Nearest(pSrc, pDst, inC, H, W);
            }

            float[] convOut = new float[outC * outH * outW];
            fixed (float* pSrc = upsampled, pDst = convOut, pW = convW, pB = convB)
            {
                Conv2D_3x3(pSrc, pDst, pW, pB, inC, outC, outH, outW);
            }
            result.Add(convOut);
        }

        H = outH;
        W = outW;
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WanRMSNorm(float* src, float* dst, float* gamma, int channels, int spatial, bool applySilu)
    {
        float scale = MathF.Sqrt(channels);

        Parallel.For(0, spatial, s =>
        {
            float sumSq = 0f;
            for (int c = 0; c < channels; c++)
            {
                float v = src[(long)c * spatial + s];
                sumSq += v * v;
            }
            float invRms = (1.0f / MathF.Sqrt(sumSq + 1e-12f)) * scale;

            for (int c = 0; c < channels; c++)
            {
                float norm = src[(long)c * spatial + s] * invRms * gamma[c];
                if (applySilu)
                {
                    norm /= (1.0f + MathF.Exp(-norm));
                }
                dst[(long)c * spatial + s] = norm;
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv3D_Causal(
        float* pT0, float* pT1, float* pT2, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 27;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int y = 0; y < H; y++)
            {
                int y0 = y - 1, y1 = y, y2 = y + 1;

                for (int x = 0; x < W; x++)
                {
                    int x0 = x - 1, x1 = x, x2 = x + 1;
                    float sum = b;

                    for (int ic = 0; ic < inC; ic++)
                    {
                        float* pWIc = pWOc + ic * 27;
                        long icSpatial = (long)ic * spatial;
                        float* pSrcT0 = pT0 != null ? pT0 + icSpatial : null;
                        float* pSrcT1 = pT1 != null ? pT1 + icSpatial : null;
                        float* pSrcT2 = pT2 + icSpatial;

                        // dt = 0 (past 2 frames ago)
                        if (pSrcT0 != null)
                        {
                            float* wDt = pWIc + 0;
                            if (y0 >= 0)
                            {
                                long r = (long)y0 * W;
                                if (x0 >= 0) sum += pSrcT0[r + x0] * wDt[0];
                                sum += pSrcT0[r + x1] * wDt[1];
                                if (x2 < W) sum += pSrcT0[r + x2] * wDt[2];
                            }
                            {
                                long r = (long)y1 * W;
                                if (x0 >= 0) sum += pSrcT0[r + x0] * wDt[3];
                                sum += pSrcT0[r + x1] * wDt[4];
                                if (x2 < W) sum += pSrcT0[r + x2] * wDt[5];
                            }
                            if (y2 < H)
                            {
                                long r = (long)y2 * W;
                                if (x0 >= 0) sum += pSrcT0[r + x0] * wDt[6];
                                sum += pSrcT0[r + x1] * wDt[7];
                                if (x2 < W) sum += pSrcT0[r + x2] * wDt[8];
                            }
                        }

                        // dt = 1 (past 1 frame ago)
                        if (pSrcT1 != null)
                        {
                            float* wDt = pWIc + 9;
                            if (y0 >= 0)
                            {
                                long r = (long)y0 * W;
                                if (x0 >= 0) sum += pSrcT1[r + x0] * wDt[0];
                                sum += pSrcT1[r + x1] * wDt[1];
                                if (x2 < W) sum += pSrcT1[r + x2] * wDt[2];
                            }
                            {
                                long r = (long)y1 * W;
                                if (x0 >= 0) sum += pSrcT1[r + x0] * wDt[3];
                                sum += pSrcT1[r + x1] * wDt[4];
                                if (x2 < W) sum += pSrcT1[r + x2] * wDt[5];
                            }
                            if (y2 < H)
                            {
                                long r = (long)y2 * W;
                                if (x0 >= 0) sum += pSrcT1[r + x0] * wDt[6];
                                sum += pSrcT1[r + x1] * wDt[7];
                                if (x2 < W) sum += pSrcT1[r + x2] * wDt[8];
                            }
                        }

                        // dt = 2 (current frame)
                        {
                            float* wDt = pWIc + 18;
                            if (y0 >= 0)
                            {
                                long r = (long)y0 * W;
                                if (x0 >= 0) sum += pSrcT2[r + x0] * wDt[0];
                                sum += pSrcT2[r + x1] * wDt[1];
                                if (x2 < W) sum += pSrcT2[r + x2] * wDt[2];
                            }
                            {
                                long r = (long)y1 * W;
                                if (x0 >= 0) sum += pSrcT2[r + x0] * wDt[3];
                                sum += pSrcT2[r + x1] * wDt[4];
                                if (x2 < W) sum += pSrcT2[r + x2] * wDt[5];
                            }
                            if (y2 < H)
                            {
                                long r = (long)y2 * W;
                                if (x0 >= 0) sum += pSrcT2[r + x0] * wDt[6];
                                sum += pSrcT2[r + x1] * wDt[7];
                                if (x2 < W) sum += pSrcT2[r + x2] * wDt[8];
                            }
                        }
                    }

                    pDstOc[y * W + x] = sum;
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv1x1_3D_Causal(
        float* pT0, float* pT1, float* pT2, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 3;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int s = 0; s < spatial; s++)
            {
                float sum = b;
                for (int ic = 0; ic < inC; ic++)
                {
                    float* wIc = pWOc + ic * 3;
                    long icSpatial = (long)ic * spatial;
                    if (pT0 != null) sum += pT0[icSpatial + s] * wIc[0];
                    if (pT1 != null) sum += pT1[icSpatial + s] * wIc[1];
                    sum += pT2[icSpatial + s] * wIc[2];
                }
                pDstOc[s] = sum;
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2D_1x1(
        float* pSrc, float* pDst, float* pW, float* pB,
        int inC, int outC, int spatial)
    {
        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int s = 0; s < spatial; s++)
            {
                float sum = b;
                for (int ic = 0; ic < inC; ic++)
                {
                    sum += pSrc[(long)ic * spatial + s] * pWOc[ic];
                }
                pDstOc[s] = sum;
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Conv2D_3x3(
        float* pSrc, float* pDst, float* pW, float* pB,
        int inC, int outC, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, outC, oc =>
        {
            float b = pB != null ? pB[oc] : 0f;
            float* pWOc = pW + (long)oc * inC * 9;
            float* pDstOc = pDst + (long)oc * spatial;

            for (int y = 0; y < H; y++)
            {
                int y0 = y - 1, y1 = y, y2 = y + 1;

                for (int x = 0; x < W; x++)
                {
                    int x0 = x - 1, x1 = x, x2 = x + 1;
                    float sum = b;

                    for (int ic = 0; ic < inC; ic++)
                    {
                        float* wIc = pWOc + ic * 9;
                        float* pSrcIc = pSrc + (long)ic * spatial;

                        if (y0 >= 0)
                        {
                            long r = (long)y0 * W;
                            if (x0 >= 0) sum += pSrcIc[r + x0] * wIc[0];
                            sum += pSrcIc[r + x1] * wIc[1];
                            if (x2 < W) sum += pSrcIc[r + x2] * wIc[2];
                        }
                        {
                            long r = (long)y1 * W;
                            if (x0 >= 0) sum += pSrcIc[r + x0] * wIc[3];
                            sum += pSrcIc[r + x1] * wIc[4];
                            if (x2 < W) sum += pSrcIc[r + x2] * wIc[5];
                        }
                        if (y2 < H)
                        {
                            long r = (long)y2 * W;
                            if (x0 >= 0) sum += pSrcIc[r + x0] * wIc[6];
                            sum += pSrcIc[r + x1] * wIc[7];
                            if (x2 < W) sum += pSrcIc[r + x2] * wIc[8];
                        }
                    }

                    pDstOc[y * W + x] = sum;
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Upsample2x_Nearest(float* src, float* dst, int channels, int inH, int inW)
    {
        int outH = inH * 2;
        int outW = inW * 2;
        int inSpatial = inH * inW;
        int outSpatial = outH * outW;

        Parallel.For(0, channels, c =>
        {
            float* pSrcC = src + (long)c * inSpatial;
            float* pDstC = dst + (long)c * outSpatial;

            for (int y = 0; y < outH; y++)
            {
                int inY = y / 2;
                long rIn = (long)inY * inW;
                long rOut = (long)y * outW;

                for (int x = 0; x < outW; x++)
                {
                    int inX = x / 2;
                    pDstC[rOut + x] = pSrcC[rIn + inX];
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ClampToRgb(float* srcPlanar, byte* dstRgb, int H, int W)
    {
        int spatial = H * W;

        Parallel.For(0, spatial, s =>
        {
            float r = srcPlanar[0 * spatial + s];
            float g = srcPlanar[1 * spatial + s];
            float b = srcPlanar[2 * spatial + s];

            int ir = (int)((r * 0.5f + 0.5f) * 255.0f);
            int ig = (int)((g * 0.5f + 0.5f) * 255.0f);
            int ib = (int)((b * 0.5f + 0.5f) * 255.0f);

            int dstIdx = s * 3;
            dstRgb[dstIdx]     = (byte)(ir < 0 ? 0 : (ir > 255 ? 255 : ir));
            dstRgb[dstIdx + 1] = (byte)(ig < 0 ? 0 : (ig > 255 ? 255 : ig));
            dstRgb[dstIdx + 2] = (byte)(ib < 0 ? 0 : (ib > 255 ? 255 : ib));
        });
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
