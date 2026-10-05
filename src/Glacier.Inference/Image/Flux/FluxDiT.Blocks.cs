namespace Glacier.Inference.Image.Flux;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gpu;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

public sealed unsafe partial class FluxDiT
{
    private static void Compute3DRoPE(int numTxtTokens, int tokenH, int tokenW, float* ropeCos, float* ropeSin)
    {
        // axes_dim = {16, 56, 56}, total headDim = 128, halfDim = 64
        int numImgTokens = tokenH * tokenW;
        int numTotalTokens = numTxtTokens + numImgTokens;

        // Axis 0: 16 (half 8), Axis 1: 56 (half 28), Axis 2: 56 (half 28)
        float* omega0 = stackalloc float[8];
        for (int i = 0; i < 8; i++) omega0[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 16.0f);

        float* omega1 = stackalloc float[28];
        for (int i = 0; i < 28; i++) omega1[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 56.0f);

        float* omega2 = stackalloc float[28];
        for (int i = 0; i < 28; i++) omega2[i] = 1.0f / MathF.Pow(10000.0f, (i * 2.0f) / 56.0f);

        // Text tokens: pos = 0 for all axes -> cos = 1, sin = 0
        for (int t = 0; t < numTxtTokens; t++)
        {
            float* cRow = ropeCos + t * 64;
            float* sRow = ropeSin + t * 64;
            for (int d = 0; d < 64; d++)
            {
                cRow[d] = 1.0f;
                sRow[d] = 0.0f;
            }
        }

        // Image tokens: axis 0 pos = 0, axis 1 pos = row, axis 2 pos = col
        for (int th = 0; th < tokenH; th++)
        {
            for (int tw = 0; tw < tokenW; tw++)
            {
                int imgIdx = th * tokenW + tw;
                int tokenIdx = numTxtTokens + imgIdx;
                float* cRow = ropeCos + tokenIdx * 64;
                float* sRow = ropeSin + tokenIdx * 64;

                // Axis 0 (dim 16, half 8): pos = 0
                for (int d = 0; d < 8; d++)
                {
                    cRow[d] = 1.0f;
                    sRow[d] = 0.0f;
                }
                // Axis 1 (dim 56, half 28): pos = th
                for (int d = 0; d < 28; d++)
                {
                    float angle = th * omega1[d];
                    cRow[8 + d] = MathF.Cos(angle);
                    sRow[8 + d] = MathF.Sin(angle);
                }
                // Axis 2 (dim 56, half 28): pos = tw
                for (int d = 0; d < 28; d++)
                {
                    float angle = tw * omega2[d];
                    cRow[36 + d] = MathF.Cos(angle);
                    sRow[36 + d] = MathF.Sin(angle);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyAdaLN(float* src, float* dst, float* shift, float* scale, int numTokens, int dim)
    {
        for (int t = 0; t < numTokens; t++)
        {
            float* s = src + t * dim;
            float* d = dst + t * dim;

            // Pure LayerNorm: (x - mean) / std (no affine parameters)
            float mean = 0f;
            for (int i = 0; i < dim; i++) mean += s[i];
            mean /= dim;

            float var = 0f;
            for (int i = 0; i < dim; i++)
            {
                float diff = s[i] - mean;
                var += diff * diff;
            }
            float invStd = 1.0f / MathF.Sqrt(var / dim + 1e-6f);

            // Modulate: (1 + scale) * norm(x) + shift
            for (int i = 0; i < dim; i++)
            {
                d[i] = ((s[i] - mean) * invStd) * (1.0f + scale[i]) + shift[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyResidualGated(float* target, float* update, float* gate, int numTokens, int dim)
    {
        for (int t = 0; t < numTokens; t++)
        {
            float* tgt = target + t * dim;
            float* upd = update + t * dim;
            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i] * gate[i];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyGelu(float* x, int count)
    {
        const float sqrt2OverPi = 0.79788456f;
        for (int i = 0; i < count; i++)
        {
            float val = x[i];
            x[i] = 0.5f * val * (1.0f + MathF.Tanh(sqrt2OverPi * (val + 0.044715f * val * val * val)));
        }
    }

    private void JointAttentionDouble(
        float* txtQkv, float* imgQkv,
        DoubleBlockWeights blk,
        float* ropeCos, float* ropeSin,
        int numTxtTokens, int numImgTokens,
        float* attnOut)
    {
        int numTotalTokens = numTxtTokens + numImgTokens;
        if (_gpu != null && blk.D_TxtQueryNorm != IntPtr.Zero)
        {
            EnsureGpuActivationBuffers(numTotalTokens);

            nuint txtBytes = (nuint)((long)numTxtTokens * 9216 * sizeof(float));
            nuint imgBytes = (nuint)((long)numImgTokens * 9216 * sizeof(float));
            _gpu.CopyToDevice(_dQkvMlpBuf, (IntPtr)txtQkv, txtBytes);
            _gpu.CopyToDevice(_dQkvMlpBuf + (nint)txtBytes, (IntPtr)imgQkv, imgBytes);

            int stride = 9216;
            int nHeads = NumHeads;
            int headDim = HeadDim;
            int txtOffset = 0;
            int imgOffset = numTxtTokens;
            int localNumTxt = numTxtTokens;
            int localNumImg = numImgTokens;
            int localNumTotal = numTotalTokens;

            IntPtr dQkvSrc = _dQkvMlpBuf;
            IntPtr dQ = _dQ;
            IntPtr dK = _dK;
            IntPtr dV = _dV;
            IntPtr dRopeCos = _dRopeCos;
            IntPtr dRopeSin = _dRopeSin;
            IntPtr dTxtQNorm = blk.D_TxtQueryNorm;
            IntPtr dTxtKNorm = blk.D_TxtKeyNorm;
            IntPtr dImgQNorm = blk.D_ImgQueryNorm;
            IntPtr dImgKNorm = blk.D_ImgKeyNorm;
            IntPtr dAttnOut = _dAttnOut;

            // 1. Text QKV Prep: unpack, RMSNorm, RoPE, store into _dQ, _dK, _dV
            void** pTxtArgs = stackalloc void*[13];
            pTxtArgs[0] = &dQkvSrc;
            pTxtArgs[1] = &dQ;
            pTxtArgs[2] = &dK;
            pTxtArgs[3] = &dV;
            pTxtArgs[4] = &dTxtQNorm;
            pTxtArgs[5] = &dTxtKNorm;
            pTxtArgs[6] = &dRopeCos;
            pTxtArgs[7] = &dRopeSin;
            pTxtArgs[8] = &stride;
            pTxtArgs[9] = &nHeads;
            pTxtArgs[10] = &headDim;
            pTxtArgs[11] = &localNumTxt;
            pTxtArgs[12] = &txtOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTxtTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:txt)");

            // 2. Image QKV Prep: unpack, RMSNorm, RoPE, store into _dQ, _dK, _dV
            IntPtr dImgQkvSrc = _dQkvMlpBuf + (nint)txtBytes;
            void** pImgArgs = stackalloc void*[13];
            pImgArgs[0] = &dImgQkvSrc;
            pImgArgs[1] = &dQ;
            pImgArgs[2] = &dK;
            pImgArgs[3] = &dV;
            pImgArgs[4] = &dImgQNorm;
            pImgArgs[5] = &dImgKNorm;
            pImgArgs[6] = &dRopeCos;
            pImgArgs[7] = &dRopeSin;
            pImgArgs[8] = &stride;
            pImgArgs[9] = &nHeads;
            pImgArgs[10] = &headDim;
            pImgArgs[11] = &localNumImg;
            pImgArgs[12] = &imgOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numImgTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:img)");

            // 3. Bidirectional Multi-Head Attention: _dQ, _dK, _dV -> _dAttnOut
            float attnScale = 1.0f / MathF.Sqrt(HeadDim);
            void** pAttnArgs = stackalloc void*[8];
            pAttnArgs[0] = &dQ;
            pAttnArgs[1] = &dK;
            pAttnArgs[2] = &dV;
            pAttnArgs[3] = &dAttnOut;
            pAttnArgs[4] = &nHeads;
            pAttnArgs[5] = &headDim;
            pAttnArgs[6] = &localNumTotal;
            pAttnArgs[7] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBidirectionalBatch, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttnArgs, IntPtr.Zero),
                "LaunchKernel(attention_bidirectional_batch:joint)");

            _gpu.Synchronize();
            _gpu.CopyToHost((IntPtr)attnOut, _dAttnOut, (nuint)((long)numTotalTokens * HiddenDim * sizeof(float)));
            return;
        }

        // CPU Reference Fallback
        float invSqrtHeadDim = 1.0f / MathF.Sqrt(HeadDim);

        float* txtQNormScale = (float*)_gguf.GetTensorPointer(blk.TxtQueryNorm);
        float* txtKNormScale = (float*)_gguf.GetTensorPointer(blk.TxtKeyNorm);
        float* imgQNormScale = (float*)_gguf.GetTensorPointer(blk.ImgQueryNorm);
        float* imgKNormScale = (float*)_gguf.GetTensorPointer(blk.ImgKeyNorm);

        // Pre-normalize Q and K for all tokens and apply RoPE
        float* qAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);
        float* kAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);
        float* vAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTotalTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            // 1. Unpack text stream (first numTxtTokens)
            for (int t = 0; t < numTxtTokens; t++)
            {
                float* qkv = txtQkv + t * 9216;
                float* qDst = qAll + t * HiddenDim;
                float* kDst = kAll + t * HiddenDim;
                float* vDst = vAll + t * HiddenDim;
                Buffer.MemoryCopy(qkv + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                // QKNorm per head
                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(qkv + h * HeadDim, qDst + h * HeadDim, txtQNormScale);
                    ApplyRMSNormHead(qkv + 3072 + h * HeadDim, kDst + h * HeadDim, txtKNormScale);
                }
            }

            // 2. Unpack image stream (subsequent numImgTokens)
            for (int t = 0; t < numImgTokens; t++)
            {
                int totalIdx = numTxtTokens + t;
                float* qkv = imgQkv + t * 9216;
                float* qDst = qAll + totalIdx * HiddenDim;
                float* kDst = kAll + totalIdx * HiddenDim;
                float* vDst = vAll + totalIdx * HiddenDim;
                Buffer.MemoryCopy(qkv + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                // QKNorm per head
                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(qkv + h * HeadDim, qDst + h * HeadDim, imgQNormScale);
                    ApplyRMSNormHead(qkv + 3072 + h * HeadDim, kDst + h * HeadDim, imgKNormScale);
                }
            }

            // 3. Apply RoPE to Q and K
            for (int t = 0; t < numTotalTokens; t++)
            {
                float* c = ropeCos + t * 64;
                float* s = ropeSin + t * 64;
                for (int h = 0; h < NumHeads; h++)
                {
                    RotateHead(qAll + t * HiddenDim + h * HeadDim, c, s);
                    RotateHead(kAll + t * HiddenDim + h * HeadDim, c, s);
                }
            }

            // 4. Multi-head Attention
            nint qPtr = (nint)qAll;
            nint kPtr = (nint)kAll;
            nint vPtr = (nint)vAll;
            nint outPtr = (nint)attnOut;

            Parallel.For(0, NumHeads, h =>
            {
                float* pQ = (float*)qPtr;
                float* pK = (float*)kPtr;
                float* pV = (float*)vPtr;
                float* pOut = (float*)outPtr;

                int headOffset = h * HeadDim;
                float[] scores = new float[numTotalTokens];

                for (int i = 0; i < numTotalTokens; i++)
                {
                    float* qRow = pQ + i * HiddenDim + headOffset;

                    float maxScore = float.NegativeInfinity;
                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float* kRow = pK + j * HiddenDim + headOffset;
                        float dot = DotHead(qRow, kRow);
                        float sc = dot * invSqrtHeadDim;
                        scores[j] = sc;
                        if (sc > maxScore) maxScore = sc;
                    }

                    // Softmax
                    float sumExp = 0f;
                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float exp = MathF.Exp(scores[j] - maxScore);
                        scores[j] = exp;
                        sumExp += exp;
                    }
                    float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                    // Output
                    float* outRow = pOut + i * HiddenDim + headOffset;
                    for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;

                    for (int j = 0; j < numTotalTokens; j++)
                    {
                        float w = scores[j] * invSum;
                        float* vRow = pV + j * HiddenDim + headOffset;
                        AccumulateHead(outRow, vRow, w);
                    }
                }
            });
        }
        finally
        {
            NativeMemory.AlignedFree(qAll);
            NativeMemory.AlignedFree(kAll);
            NativeMemory.AlignedFree(vAll);
        }
    }

    private void SingleAttentionAndMlp(
        float* qkvMlp,
        GgufTensorInfo qNorm, GgufTensorInfo kNorm,
        float* ropeCos, float* ropeSin,
        int numTokens,
        float* attnOutMlp)
    {
        float invSqrtHeadDim = 1.0f / MathF.Sqrt(HeadDim);
        float* qNormScale = (float*)_gguf.GetTensorPointer(qNorm);
        float* kNormScale = (float*)_gguf.GetTensorPointer(kNorm);

        float* qAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);
        float* kAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);
        float* vAll = (float*)NativeMemory.AlignedAlloc((nuint)(numTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            for (int t = 0; t < numTokens; t++)
            {
                float* row = qkvMlp + t * 21504;
                float* qDst = qAll + t * HiddenDim;
                float* kDst = kAll + t * HiddenDim;
                float* vDst = vAll + t * HiddenDim;
                Buffer.MemoryCopy(row + 6144, vDst, HiddenDim * sizeof(float), HiddenDim * sizeof(float));

                for (int h = 0; h < NumHeads; h++)
                {
                    ApplyRMSNormHead(row + h * HeadDim, qDst + h * HeadDim, qNormScale);
                    ApplyRMSNormHead(row + 3072 + h * HeadDim, kDst + h * HeadDim, kNormScale);
                }

                // RoPE
                float* c = ropeCos + t * 64;
                float* s = ropeSin + t * 64;
                for (int h = 0; h < NumHeads; h++)
                {
                    RotateHead(qDst + h * HeadDim, c, s);
                    RotateHead(kDst + h * HeadDim, c, s);
                }
            }

            // Multi-head Attention
            nint qPtr = (nint)qAll;
            nint kPtr = (nint)kAll;
            nint vPtr = (nint)vAll;
            nint outPtr = (nint)attnOutMlp;

            Parallel.For(0, NumHeads, h =>
            {
                float* pQ = (float*)qPtr;
                float* pK = (float*)kPtr;
                float* pV = (float*)vPtr;
                float* pOut = (float*)outPtr;

                int headOffset = h * HeadDim;
                float[] scores = new float[numTokens];

                for (int i = 0; i < numTokens; i++)
                {
                    float* qRow = pQ + i * HiddenDim + headOffset;

                    float maxScore = float.NegativeInfinity;
                    for (int j = 0; j < numTokens; j++)
                    {
                        float* kRow = pK + j * HiddenDim + headOffset;
                        float dot = DotHead(qRow, kRow);
                        float sc = dot * invSqrtHeadDim;
                        scores[j] = sc;
                        if (sc > maxScore) maxScore = sc;
                    }

                    float sumExp = 0f;
                    for (int j = 0; j < numTokens; j++)
                    {
                        float exp = MathF.Exp(scores[j] - maxScore);
                        scores[j] = exp;
                        sumExp += exp;
                    }
                    float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                    float* outRow = pOut + i * 15360 + headOffset;
                    for (int d = 0; d < HeadDim; d++) outRow[d] = 0f;

                    for (int j = 0; j < numTokens; j++)
                    {
                        float w = scores[j] * invSum;
                        float* vRow = pV + j * HiddenDim + headOffset;
                        AccumulateHead(outRow, vRow, w);
                    }
                }
            });

            // Fused MLP activation: copy activated MLP (12288) to position [3072..15359]
            nint qkvMlpPtr = (nint)qkvMlp;
            nint attnOutMlpPtr = (nint)attnOutMlp;

            Parallel.For(0, numTokens, t =>
            {
                float* mlpSrc = (float*)qkvMlpPtr + t * 21504 + 9216;
                float* mlpDst = (float*)attnOutMlpPtr + t * 15360 + 3072;
                ApplyGelu(mlpSrc, 12288);
                Buffer.MemoryCopy(mlpSrc, mlpDst, 12288 * sizeof(float), 12288 * sizeof(float));
            });
        }
        finally
        {
            NativeMemory.AlignedFree(qAll);
            NativeMemory.AlignedFree(kAll);
            NativeMemory.AlignedFree(vAll);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotHead(float* a, float* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vSum0 = Vector256<float>.Zero;
            var vSum1 = Vector256<float>.Zero;
            for (int d = 0; d < 128; d += 16)
            {
                vSum0 += Vector256.Load(a + d) * Vector256.Load(b + d);
                vSum1 += Vector256.Load(a + d + 8) * Vector256.Load(b + d + 8);
            }
            return Vector256.Sum(vSum0 + vSum1);
        }
        float dot = 0f;
        for (int d = 0; d < 128; d++) dot += a[d] * b[d];
        return dot;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateHead(float* dst, float* v, float w)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vw = Vector256.Create(w);
            for (int d = 0; d < 128; d += 8)
            {
                var res = Vector256.Load(dst + d) + vw * Vector256.Load(v + d);
                res.Store(dst + d);
            }
            return;
        }
        for (int d = 0; d < 128; d++) dst[d] += w * v[d];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRMSNormHead(float* src, float* dst, float* scale)
    {
        float sumSq = 0f;
        for (int i = 0; i < HeadDim; i++) sumSq += src[i] * src[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / HeadDim + 1e-6f);
        for (int i = 0; i < HeadDim; i++) dst[i] = src[i] * invRms * scale[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHead(float* vec, float* cos, float* sin)
    {
        for (int i = 0; i < 64; i++)
        {
            float v0 = vec[2 * i];
            float v1 = vec[2 * i + 1];
            float c = cos[i];
            float s = sin[i];
            vec[2 * i] = v0 * c - v1 * s;
            vec[2 * i + 1] = v0 * s + v1 * c;
        }
    }

    private void DispatchLinearVec(GgufTensorInfo w, GgufTensorInfo b, float* x, float* y)
    {
        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = (float*)_gguf.GetTensorPointer(b);
        int nCols = (int)w.Dimensions[0];
        int nRows = (int)w.Dimensions[1];

        QuantKernels.MatVecMul(w.Type, wPtr, x, y, nCols, nRows);
        for (int i = 0; i < nRows; i++) y[i] += bPtr[i];
    }

}
