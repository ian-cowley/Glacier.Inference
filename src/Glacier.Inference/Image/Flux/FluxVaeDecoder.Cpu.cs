namespace Glacier.Inference.Image.Flux;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;

public unsafe sealed partial class FluxVaeDecoder
{
    private void DecodeCpu(ReadOnlySpan<float> latents, int latentH, int latentW, Span<byte> outputRgb)
    {
        int targetH = latentH * 8;
        int targetW = latentW * 8;

        // Step 0: Preprocess and shift/scale latents
        int inC = 16;
        int spatialSize = latentH * latentW;
        float[] scaledLatents = new float[inC * spatialSize];
        fixed (float* pDst = scaledLatents)
        fixed (float* pSrc = latents)
        {
            for (int i = 0; i < inC * spatialSize; i++)
            {
                pDst[i] = (pSrc[i] / ScaleFactor) + ShiftFactor;
            }
        }

        // Feature map buffers (ping-pong allocation)
        int curH = latentH;
        int curW = latentW;

        fixed (float* pLat = scaledLatents)
        {
            // 1. decoder.conv_in (16 -> 512 channels, 3x3)
            float[] bufA = new float[512 * curH * curW];
            fixed (float* pA = bufA)
            {
                Conv3x3(pLat, pA, 16, 512, curH, curW, GetWeight("decoder.conv_in.weight"), GetWeight("decoder.conv_in.bias"));

                // 2. decoder.mid.block_1 (ResNet 512 -> 512)
                float[] bufB = new float[512 * curH * curW];
                fixed (float* pB = bufB)
                {
                    ResNetBlock(pA, pB, 512, 512, curH, curW, "decoder.mid.block_1");

                    // 3. decoder.mid.attn_1 (Self-Attention 512)
                    SelfAttention(pB, pA, 512, curH, curW, "decoder.mid.attn_1");

                    // 4. decoder.mid.block_2 (ResNet 512 -> 512)
                    ResNetBlock(pA, pB, 512, 512, curH, curW, "decoder.mid.block_2");

                    // Copy active output to pA for upsampling pipeline
                    Buffer.BlockCopy(bufB, 0, bufA, 0, 512 * curH * curW * sizeof(float));
                }
            }

            // 5. Up-blocks: up.3, up.2, up.1, up.0
            float[] currentFeat = bufA;

            // --- Level 3: 512 channels -> 512 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 512, ref curH, ref curW, "decoder.up.3", hasUpsample: true);

            // --- Level 2: 512 channels -> 512 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 512, ref curH, ref curW, "decoder.up.2", hasUpsample: true);

            // --- Level 1: 512 channels -> 256 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 256, ref curH, ref curW, "decoder.up.1", hasUpsample: true);

            // --- Level 0: 256 channels -> 128 channels (no upsample, already at 8x target resolution) ---
            currentFeat = RunUpStage(currentFeat, 256, 128, ref curH, ref curW, "decoder.up.0", hasUpsample: false);

            // 6. decoder.norm_out (GroupNorm 32, 128) + SiLU
            float[] finalNorm = new float[128 * curH * curW];
            fixed (float* pIn = currentFeat)
            fixed (float* pNorm = finalNorm)
            {
                GroupNorm(pIn, pNorm, 128, curH, curW, 32, GetWeight("decoder.norm_out.weight"), GetWeight("decoder.norm_out.bias"));
                SiLU(pNorm, 128 * curH * curW);

                // 7. decoder.conv_out (128 -> 3 channels, 3x3)
                float[] finalRgbFloats = new float[3 * curH * curW];
                fixed (float* pRgbF = finalRgbFloats)
                fixed (byte* pOut = outputRgb)
                {
                    Conv3x3(pNorm, pRgbF, 128, 3, curH, curW, GetWeight("decoder.conv_out.weight"), GetWeight("decoder.conv_out.bias"));

                    // 8. Convert [-1, 1] floats to standard 24-bit sRGB bytes
                    int planeSize = curH * curW;
                    nint rgbPtr = (nint)pRgbF;
                    nint outPtr = (nint)pOut;

                    Parallel.For(0, curH, y =>
                    {
                        float* pRgbLocal = (float*)rgbPtr;
                        byte* pOutLocal = (byte*)outPtr;

                        for (int x = 0; x < curW; x++)
                        {
                            int srcIdx = y * curW + x;
                            int dstIdx = (y * curW + x) * 3;

                            float r = pRgbLocal[0 * planeSize + srcIdx];
                            float g = pRgbLocal[1 * planeSize + srcIdx];
                            float b = pRgbLocal[2 * planeSize + srcIdx];

                            pOutLocal[dstIdx]     = (byte)Math.Clamp((int)((r * 0.5f + 0.5f) * 255.0f), 0, 255);
                            pOutLocal[dstIdx + 1] = (byte)Math.Clamp((int)((g * 0.5f + 0.5f) * 255.0f), 0, 255);
                            pOutLocal[dstIdx + 2] = (byte)Math.Clamp((int)((b * 0.5f + 0.5f) * 255.0f), 0, 255);
                        }
                    });
                }
            }
        }
    }

    private float[] RunUpStage(float[] input, int inC, int outC, ref int h, ref int w, string prefix, bool hasUpsample)
    {
        int curC = inC;
        float[] cur = input;

        // 3 ResNet blocks per up stage
        for (int b = 0; b < 3; b++)
        {
            string blockPrefix = $"{prefix}.block.{b}";
            int bOutC = (b == 0) ? outC : outC;
            int bInC = (b == 0) ? curC : outC;

            float[] next = new float[bOutC * h * w];
            fixed (float* pSrc = cur)
            fixed (float* pDst = next)
            {
                ResNetBlock(pSrc, pDst, bInC, bOutC, h, w, blockPrefix);
            }
            cur = next;
            curC = bOutC;
        }

        if (hasUpsample)
        {
            // 2x Nearest Neighbor Spatial Upsample
            int newH = h * 2;
            int newW = w * 2;
            float[] upsampled = new float[curC * newH * newW];
            fixed (float* pSrc = cur)
            fixed (float* pDst = upsampled)
            {
                Upsample2x(pSrc, pDst, curC, h, w);
            }

            // Up-stage convolution (curC -> curC, 3x3)
            float[] convOut = new float[curC * newH * newW];
            fixed (float* pUp = upsampled)
            fixed (float* pOut = convOut)
            {
                Conv3x3(pUp, pOut, curC, curC, newH, newW, GetWeight($"{prefix}.upsample.conv.weight"), GetWeight($"{prefix}.upsample.conv.bias"));
            }

            h = newH;
            w = newW;
            return convOut;
        }

        return cur;
    }

    private void ResNetBlock(float* src, float* dst, int inC, int outC, int h, int w, string prefix)
    {
        int spatial = h * w;
        float[] norm1Buf = new float[inC * spatial];
        float[] conv1Buf = new float[outC * spatial];
        float[] norm2Buf = new float[outC * spatial];

        fixed (float* pNorm1 = norm1Buf)
        fixed (float* pConv1 = conv1Buf)
        fixed (float* pNorm2 = norm2Buf)
        {
            // 1. norm1 (GroupNorm 32, inC)
            GroupNorm(src, pNorm1, inC, h, w, 32, GetWeight($"{prefix}.norm1.weight"), GetWeight($"{prefix}.norm1.bias"));

            // 2. silu
            SiLU(pNorm1, inC * spatial);

            // 3. conv1 (inC -> outC, 3x3)
            Conv3x3(pNorm1, pConv1, inC, outC, h, w, GetWeight($"{prefix}.conv1.weight"), GetWeight($"{prefix}.conv1.bias"));

            // 4. norm2 (GroupNorm 32, outC)
            GroupNorm(pConv1, pNorm2, outC, h, w, 32, GetWeight($"{prefix}.norm2.weight"), GetWeight($"{prefix}.norm2.bias"));

            // 5. silu
            SiLU(pNorm2, outC * spatial);

            // 6. conv2 (outC -> outC, 3x3) into dst
            Conv3x3(pNorm2, dst, outC, outC, h, w, GetWeight($"{prefix}.conv2.weight"), GetWeight($"{prefix}.conv2.bias"));

            // 7. Residual addition: dst += (nin_shortcut(src) or src)
            if (inC != outC && HasTensor($"{prefix}.nin_shortcut.weight"))
            {
                float[] shortcut = new float[outC * spatial];
                fixed (float* pShort = shortcut)
                {
                    Conv1x1(src, pShort, inC, outC, h, w, GetWeight($"{prefix}.nin_shortcut.weight"), GetWeight($"{prefix}.nin_shortcut.bias"));
                    for (int i = 0; i < outC * spatial; i++)
                    {
                        dst[i] += pShort[i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < outC * spatial; i++)
                {
                    dst[i] += src[i];
                }
            }
        }
    }

    private void SelfAttention(float* src, float* dst, int channels, int h, int w, string prefix)
    {
        int spatial = h * w;
        float[] normBuf = new float[channels * spatial];

        fixed (float* pNorm = normBuf)
        {
            // 1. Norm
            GroupNorm(src, pNorm, channels, h, w, 32, GetWeight($"{prefix}.norm.weight"), GetWeight($"{prefix}.norm.bias"));

            // 2. Q, K, V (1x1 Conv)
            float[] qBuf = new float[channels * spatial];
            float[] kBuf = new float[channels * spatial];
            float[] vBuf = new float[channels * spatial];

            fixed (float* pQ = qBuf)
            fixed (float* pK = kBuf)
            fixed (float* pV = vBuf)
            {
                Conv1x1(pNorm, pQ, channels, channels, h, w, GetWeight($"{prefix}.q.weight"), GetWeight($"{prefix}.q.bias"));
                Conv1x1(pNorm, pK, channels, channels, h, w, GetWeight($"{prefix}.k.weight"), GetWeight($"{prefix}.k.bias"));
                Conv1x1(pNorm, pV, channels, channels, h, w, GetWeight($"{prefix}.v.weight"), GetWeight($"{prefix}.v.bias"));

                // Scaled Dot-Product Attention: [spatial, spatial]
                float scale = 1.0f / MathF.Sqrt(channels);
                float[] attnOut = new float[channels * spatial];

                fixed (float* pAttn = attnOut)
                {
                    nint qPtr = (nint)pQ;
                    nint kPtr = (nint)pK;
                    nint vPtr = (nint)pV;
                    nint attnPtr = (nint)pAttn;

                    // Compute attention over spatial tokens
                    Parallel.For(0, spatial, i =>
                    {
                        float* pQLocal = (float*)qPtr;
                        float* pKLocal = (float*)kPtr;
                        float* pVLocal = (float*)vPtr;
                        float* pAttnLocal = (float*)attnPtr;

                        float[] scores = new float[spatial];
                        float maxScore = -float.MaxValue;

                        for (int j = 0; j < spatial; j++)
                        {
                            float dot = 0.0f;
                            for (int c = 0; c < channels; c++)
                            {
                                dot += pQLocal[c * spatial + i] * pKLocal[c * spatial + j];
                            }
                            dot *= scale;
                            scores[j] = dot;
                            if (dot > maxScore) maxScore = dot;
                        }

                        // Softmax
                        float sumExp = 0.0f;
                        for (int j = 0; j < spatial; j++)
                        {
                            float exp = MathF.Exp(scores[j] - maxScore);
                            scores[j] = exp;
                            sumExp += exp;
                        }
                        float invSum = 1.0f / sumExp;

                        // Weighted V sum
                        for (int c = 0; c < channels; c++)
                        {
                            float val = 0.0f;
                            for (int j = 0; j < spatial; j++)
                            {
                                val += scores[j] * invSum * pVLocal[c * spatial + j];
                            }
                            pAttnLocal[c * spatial + i] = val;
                        }
                    });

                    // 3. proj_out (1x1 Conv)
                    float[] projBuf = new float[channels * spatial];
                    fixed (float* pProj = projBuf)
                    {
                        Conv1x1(pAttn, pProj, channels, channels, h, w, GetWeight($"{prefix}.proj_out.weight"), GetWeight($"{prefix}.proj_out.bias"));

                        // Residual add: dst = src + proj
                        for (int i = 0; i < channels * spatial; i++)
                        {
                            dst[i] = src[i] + pProj[i];
                        }
                    }
                }
            }
        }
    }

    private static void GroupNorm(float* src, float* dst, int channels, int h, int w, int groups, float* weight, float* bias)
    {
        int channelsPerGroup = channels / groups;
        int spatial = h * w;
        int groupSpatial = channelsPerGroup * spatial;

        Parallel.For(0, groups, g =>
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            for (int c = 0; c < channelsPerGroup; c++)
            {
                int fullC = g * channelsPerGroup + c;
                float* pC = src + fullC * spatial;
                for (int i = 0; i < spatial; i++)
                {
                    float v = pC[i];
                    sum += v;
                    sumSq += v * v;
                }
            }

            float mean = sum / groupSpatial;
            float variance = MathF.Max(0.0f, (sumSq / groupSpatial) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);

            for (int c = 0; c < channelsPerGroup; c++)
            {
                int fullC = g * channelsPerGroup + c;
                float gamma = weight[fullC];
                float beta = bias[fullC];
                float* pSrcC = src + fullC * spatial;
                float* pDstC = dst + fullC * spatial;

                for (int i = 0; i < spatial; i++)
                {
                    pDstC[i] = (pSrcC[i] - mean) * invStd * gamma + beta;
                }
            }
        });
    }

    private static void SiLU(float* data, int length)
    {
        Parallel.For(0, length / 1024 + 1, chunk =>
        {
            int start = chunk * 1024;
            int end = Math.Min(start + 1024, length);
            for (int i = start; i < end; i++)
            {
                float x = data[i];
                data[i] = x / (1.0f + MathF.Exp(-x));
            }
        });
    }

    private static void Upsample2x(float* src, float* dst, int channels, int inH, int inW)
    {
        int outH = inH * 2;
        int outW = inW * 2;

        Parallel.For(0, channels, c =>
        {
            float* pSrc = src + c * (inH * inW);
            float* pDst = dst + c * (outH * outW);

            for (int y = 0; y < outH; y++)
            {
                int inY = y / 2;
                for (int x = 0; x < outW; x++)
                {
                    int inX = x / 2;
                    pDst[y * outW + x] = pSrc[inY * inW + inX];
                }
            }
        });
    }

    private static void Conv1x1(float* src, float* dst, int inC, int outC, int h, int w, float* weight, float* bias)
    {
        int spatial = h * w;

        Parallel.For(0, outC, oc =>
        {
            float b = bias[oc];
            float* pDstOc = dst + oc * spatial;

            // Initialize with bias
            if (Vector256.IsHardwareAccelerated)
            {
                var vb = Vector256.Create(b);
                int i = 0;
                for (; i <= spatial - 8; i += 8) vb.Store(pDstOc + i);
                for (; i < spatial; i++) pDstOc[i] = b;
            }
            else
            {
                for (int i = 0; i < spatial; i++) pDstOc[i] = b;
            }

            for (int ic = 0; ic < inC; ic++)
            {
                float wVal = weight[oc * inC + ic];
                float* pSrcIc = src + ic * spatial;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vw = Vector256.Create(wVal);
                    int i = 0;
                    for (; i <= spatial - 16; i += 16)
                    {
                        var d0 = Vector256.Load(pDstOc + i) + vw * Vector256.Load(pSrcIc + i);
                        var d1 = Vector256.Load(pDstOc + i + 8) + vw * Vector256.Load(pSrcIc + i + 8);
                        d0.Store(pDstOc + i);
                        d1.Store(pDstOc + i + 8);
                    }
                    for (; i < spatial; i++)
                    {
                        pDstOc[i] += pSrcIc[i] * wVal;
                    }
                }
                else
                {
                    for (int i = 0; i < spatial; i++)
                    {
                        pDstOc[i] += pSrcIc[i] * wVal;
                    }
                }
            }
        });
    }

    private static void Conv3x3(float* src, float* dst, int inC, int outC, int h, int w, float* weight, float* bias)
    {
        int spatial = h * w;

        Parallel.For(0, outC * h, idx =>
        {
            int oc = idx / h;
            int y = idx % h;

            float b = bias[oc];
            float* pDstRow = dst + oc * spatial + y * w;

            // Initialize row with bias
            if (Vector256.IsHardwareAccelerated)
            {
                var vb = Vector256.Create(b);
                int x = 0;
                for (; x <= w - 8; x += 8) vb.Store(pDstRow + x);
                for (; x < w; x++) pDstRow[x] = b;
            }
            else
            {
                for (int x = 0; x < w; x++) pDstRow[x] = b;
            }

            int y0 = y - 1;
            int y1 = y;
            int y2 = y + 1;

            float* wOc = weight + oc * (inC * 9);

            for (int ic = 0; ic < inC; ic++)
            {
                float* wIc = wOc + ic * 9;
                float* pSrcIc = src + ic * spatial;

                float* r0 = y0 >= 0 ? (pSrcIc + y0 * w) : null;
                float* r1 = pSrcIc + y1 * w;
                float* r2 = y2 < h ? (pSrcIc + y2 * w) : null;

                float w0 = wIc[0], w1 = wIc[1], w2 = wIc[2];
                float w3 = wIc[3], w4 = wIc[4], w5 = wIc[5];
                float w6 = wIc[6], w7 = wIc[7], w8 = wIc[8];

                // Left border x = 0
                {
                    float sum = 0.0f;
                    if (r0 != null) sum += r0[0] * w1 + (w > 1 ? r0[1] * w2 : 0f);
                    sum += r1[0] * w4 + (w > 1 ? r1[1] * w5 : 0f);
                    if (r2 != null) sum += r2[0] * w7 + (w > 1 ? r2[1] * w8 : 0f);
                    pDstRow[0] += sum;
                }

                // Middle vector loop x in [1, w - 8]
                int xMid = 1;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vw0 = Vector256.Create(w0);
                    var vw1 = Vector256.Create(w1);
                    var vw2 = Vector256.Create(w2);
                    var vw3 = Vector256.Create(w3);
                    var vw4 = Vector256.Create(w4);
                    var vw5 = Vector256.Create(w5);
                    var vw6 = Vector256.Create(w6);
                    var vw7 = Vector256.Create(w7);
                    var vw8 = Vector256.Create(w8);

                    int vecEnd = w - 9;
                    for (; xMid <= vecEnd; xMid += 8)
                    {
                        var acc = Vector256.Load(pDstRow + xMid);

                        if (r0 != null)
                        {
                            acc += Vector256.Load(r0 + xMid - 1) * vw0
                                 + Vector256.Load(r0 + xMid) * vw1
                                 + Vector256.Load(r0 + xMid + 1) * vw2;
                        }

                        acc += Vector256.Load(r1 + xMid - 1) * vw3
                             + Vector256.Load(r1 + xMid) * vw4
                             + Vector256.Load(r1 + xMid + 1) * vw5;

                        if (r2 != null)
                        {
                            acc += Vector256.Load(r2 + xMid - 1) * vw6
                                 + Vector256.Load(r2 + xMid) * vw7
                                 + Vector256.Load(r2 + xMid + 1) * vw8;
                        }

                        acc.Store(pDstRow + xMid);
                    }
                }

                // Remaining pixels up to right border
                for (; xMid < w; xMid++)
                {
                    int x0 = xMid - 1;
                    int x1 = xMid;
                    int x2 = xMid + 1;

                    float sum = 0.0f;
                    if (r0 != null)
                    {
                        if (x0 >= 0) sum += r0[x0] * w0;
                        sum += r0[x1] * w1;
                        if (x2 < w) sum += r0[x2] * w2;
                    }

                    if (x0 >= 0) sum += r1[x0] * w3;
                    sum += r1[x1] * w4;
                    if (x2 < w) sum += r1[x2] * w5;

                    if (r2 != null)
                    {
                        if (x0 >= 0) sum += r2[x0] * w6;
                        sum += r2[x1] * w7;
                        if (x2 < w) sum += r2[x2] * w8;
                    }

                    pDstRow[xMid] += sum;
                }
            }
        });
    }

}
