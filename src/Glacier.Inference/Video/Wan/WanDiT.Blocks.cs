namespace Glacier.Inference.Video.Wan;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

public sealed unsafe partial class WanDiT
{
    private void ExecuteBlockCpu(
        float* currentTokens,
        int totalImgTokens,
        int frames,
        int tokenH,
        int tokenW,
        float* ropeCos,
        float* ropeSin,
        float* projectedTxt,
        int numTxtTokens,
        WanBlockWeights blk,
        float* timeModAll)
    {
        float* modBase = (float*)_gguf.GetTensorPointer(blk.Modulation);
        float* shift_msa   = stackalloc float[HiddenDim];
        float* scale_msa   = stackalloc float[HiddenDim];
        float* gate_msa    = stackalloc float[HiddenDim];
        float* c_shift_msa = stackalloc float[HiddenDim];
        float* c_scale_msa = stackalloc float[HiddenDim];
        float* c_gate_msa  = stackalloc float[HiddenDim];

        for (int d = 0; d < HiddenDim; d++)
        {
            shift_msa[d]   = modBase[0 * HiddenDim + d] + timeModAll[0 * HiddenDim + d];
            scale_msa[d]   = modBase[1 * HiddenDim + d] + timeModAll[1 * HiddenDim + d];
            gate_msa[d]    = modBase[2 * HiddenDim + d] + timeModAll[2 * HiddenDim + d];
            c_shift_msa[d] = modBase[3 * HiddenDim + d] + timeModAll[3 * HiddenDim + d];
            c_scale_msa[d] = modBase[4 * HiddenDim + d] + timeModAll[4 * HiddenDim + d];
            c_gate_msa[d]  = modBase[5 * HiddenDim + d] + timeModAll[5 * HiddenDim + d];
        }

        // 1. Self-Attention
        // 1a. AdaLN LayerNorm: _tokenBufferB = LayerNorm(currentTokens) * (1 + scale_msa) + shift_msa
        ApplyAdaLnCpu(currentTokens, _tokenBufferB, shift_msa, scale_msa, totalImgTokens, HiddenDim);

        // 1b. Self-Attention Q, K, V Projections from _tokenBufferB -> output into _tokenBufferC
        ExecuteSelfAttentionCpu(_tokenBufferB, totalImgTokens, blk, ropeCos, ropeSin, _tokenBufferC);

        // 1c. Gated Residual: currentTokens += _tokenBufferC * gate_msa
        ApplyGatedResidualCpu(currentTokens, _tokenBufferC, gate_msa, totalImgTokens, HiddenDim);

        // 2. Cross-Attention (if prompt text context provided)
        if (numTxtTokens > 0)
        {
            float* norm3W = (float*)_gguf.GetTensorPointer(blk.Norm3_W);
            float* norm3B = (float*)_gguf.GetTensorPointer(blk.Norm3_B);

            // 2a. LayerNorm with affine norm3: _tokenBufferB = LayerNorm(currentTokens) * norm3W + norm3B
            ApplyAffineLayerNormCpu(currentTokens, _tokenBufferB, norm3W, norm3B, totalImgTokens, HiddenDim);

            // 2b. Cross-Attention between _tokenBufferB and projectedTxt -> output into _tokenBufferC
            ExecuteCrossAttentionCpu(_tokenBufferB, totalImgTokens, projectedTxt, numTxtTokens, blk, _tokenBufferC);

            // 2c. Residual: currentTokens += _tokenBufferC
            ApplyResidualCpu(currentTokens, _tokenBufferC, totalImgTokens, HiddenDim);
        }

        // 3. Feed-Forward Network
        // 3a. AdaLN LayerNorm: _tokenBufferB = LayerNorm(currentTokens) * (1 + c_scale_msa) + c_shift_msa
        ApplyAdaLnCpu(currentTokens, _tokenBufferB, c_shift_msa, c_scale_msa, totalImgTokens, HiddenDim);

        // 3b. FFN -> output into _tokenBufferC
        ExecuteFFNCpu(_tokenBufferB, totalImgTokens, blk, _tokenBufferC);

        // 3c. Gated Residual: currentTokens += _tokenBufferC * c_gate_msa
        ApplyGatedResidualCpu(currentTokens, _tokenBufferC, c_gate_msa, totalImgTokens, HiddenDim);
    }

    private void Precompute3DRoPE(int frames, int tokenH, int tokenW, float* ropeCos, float* ropeSin)
    {
        for (int f = 0; f < frames; f++)
        {
            for (int th = 0; th < tokenH; th++)
            {
                for (int tw = 0; tw < tokenW; tw++)
                {
                    int tokIdx = f * (tokenH * tokenW) + th * tokenW + tw;
                    float* cosT = ropeCos + tokIdx * 64;
                    float* sinT = ropeSin + tokIdx * 64;

                    // Wan 2.1 3D-RoPE: HeadDim = 128 (64 frequency pairs)
                    // Split sizes: t_dim = 44 (22 pairs), h_dim = 42 (21 pairs), w_dim = 42 (21 pairs)
                    // Base theta = 10000.0f
                    // 1. Temporal: 22 pairs (t_dim = 44)
                    for (int k = 0; k < 22; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 44.0f);
                        float theta = f * freq;
                        cosT[k] = MathF.Cos(theta);
                        sinT[k] = MathF.Sin(theta);
                    }

                    // 2. Height: 21 pairs (h_dim = 42)
                    for (int k = 0; k < 21; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 42.0f);
                        float theta = th * freq;
                        cosT[22 + k] = MathF.Cos(theta);
                        sinT[22 + k] = MathF.Sin(theta);
                    }

                    // 3. Width: 21 pairs (w_dim = 42)
                    for (int k = 0; k < 21; k++)
                    {
                        float freq = 1.0f / MathF.Pow(10000.0f, (2.0f * k) / 42.0f);
                        float theta = tw * freq;
                        cosT[43 + k] = MathF.Cos(theta);
                        sinT[43 + k] = MathF.Sin(theta);
                    }
                }
            }
        }
    }

    private static void Apply3DRoPEToTokens(float* qk, int totalTokens, float* ropeCos, float* ropeSin)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* cosT = ropeCos + t * 64;
            float* sinT = ropeSin + t * 64;

            for (int h = 0; h < NumHeads; h++)
            {
                float* head = qk + (long)t * HiddenDim + h * HeadDim;

                for (int p = 0; p < 64; p++)
                {
                    float c = cosT[p];
                    float s = sinT[p];

                    float v0 = head[2 * p];
                    float v1 = head[2 * p + 1];

                    head[2 * p]     = v0 * c - v1 * s;
                    head[2 * p + 1] = v0 * s + v1 * c;
                }
            }
        });
    }

    private void ComputeTimeEmbedding(float timestep, float* timeVec, float* timeModAll)
    {
        float* rawTime = stackalloc float[256];
        float halfDim = 128f;
        float tVal = timestep <= 1.0f ? timestep * 1000.0f : timestep;
        for (int i = 0; i < 128; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / halfDim);
            float arg = tVal * freq;
            rawTime[i] = MathF.Cos(arg);
            rawTime[128 + i] = MathF.Sin(arg);
        }

        float* timeInt = stackalloc float[HiddenDim];
        DispatchLinearVec(_timeEmbed0W, _timeEmbed0B, rawTime, timeInt);
        ApplySilu(timeInt, HiddenDim);

        DispatchLinearVec(_timeEmbed2W, _timeEmbed2B, timeInt, timeVec);

        float* siluTimeVec = stackalloc float[HiddenDim];
        for (int i = 0; i < HiddenDim; i++)
        {
            siluTimeVec[i] = timeVec[i] / (1.0f + MathF.Exp(-timeVec[i]));
        }

        DispatchLinearVec(_timeProj1W, _timeProj1B, siluTimeVec, timeModAll);
    }

    private void ProjectText(ReadOnlySpan<float> textContext, int numTxtTokens, float* projectedTxt)
    {
        fixed (float* pTxt = textContext)
        {
            nint pTxtNint = (nint)pTxt;
            nint projTxtNint = (nint)projectedTxt;

            Parallel.For(0, numTxtTokens, t =>
            {
                float* intermediate = stackalloc float[HiddenDim];
                float* src = (float*)pTxtNint + t * 4096;
                float* dst = (float*)projTxtNint + t * HiddenDim;

                DispatchLinearVec(_textEmbed0W, _textEmbed0B, src, intermediate);
                ApplyGelu(intermediate, HiddenDim);
                DispatchLinearVec(_textEmbed2W, _textEmbed2B, intermediate, dst);
            });
        }
    }

    private void PatchifyAndProject(ReadOnlySpan<float> latents, int frames, int H, int W, float* outputTokens)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;
        int spatialTokens = tokenH * tokenW;
        int totalTokens = frames * spatialTokens;

        float* allPatches = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * PatchDim * sizeof(float)), 64);

        try
        {
            fixed (float* pLat = latents)
            {
                int tokenIdx = 0;
                for (int f = 0; f < frames; f++)
                {
                    int frameLatentOffset = f * (InChannels * H * W);

                    for (int th = 0; th < tokenH; th++)
                    {
                        for (int tw = 0; tw < tokenW; tw++)
                        {
                            float* dstPatch = allPatches + (long)tokenIdx * PatchDim;
                            int pIdx = 0;

                            for (int c = 0; c < InChannels; c++)
                            {
                                for (int ph = 0; ph < PatchSize; ph++)
                                {
                                    for (int pw = 0; pw < PatchSize; pw++)
                                    {
                                        int y = th * PatchSize + ph;
                                        int x = tw * PatchSize + pw;
                                        dstPatch[pIdx++] = pLat[frameLatentOffset + c * H * W + y * W + x];
                                    }
                                }
                            }
                            tokenIdx++;
                        }
                    }
                }
            }

            var (nCols, nRows) = GetMatrixDims(_patchEmbedW); // 64, 1536
            byte* wPtr = _gguf.GetTensorPointer(_patchEmbedW);
            float* bPtr = (float*)_gguf.GetTensorPointer(_patchEmbedB);

            QuantKernels.MatMulBatch(_patchEmbedW.Type, wPtr, allPatches, outputTokens, nCols, nRows, totalTokens);

            if (bPtr != null)
            {
                Parallel.For(0, totalTokens, t =>
                {
                    float* tok = outputTokens + (long)t * HiddenDim;
                    for (int d = 0; d < HiddenDim; d++) tok[d] += bPtr[d];
                });
            }
        }
        finally
        {
            NativeMemory.AlignedFree(allPatches);
        }
    }

    private static void ApplyAdaLnCpu(
        float* src, float* dst, float* shift, float* scale, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* s = src + (long)t * dim;
            float* d = dst + (long)t * dim;

            float sum = 0f;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++)
            {
                float v = s[i];
                sum += v;
                sumSq += v * v;
            }

            float mean = sum / dim;
            float var = MathF.Max(0f, (sumSq / dim) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(var + 1e-6f);

            for (int i = 0; i < dim; i++)
            {
                d[i] = ((s[i] - mean) * invStd) * (1.0f + scale[i]) + shift[i];
            }
        });
    }

    private static void ApplyAffineLayerNormCpu(
        float* src, float* dst, float* weight, float* bias, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* s = src + (long)t * dim;
            float* d = dst + (long)t * dim;

            float sum = 0f;
            float sumSq = 0f;
            for (int i = 0; i < dim; i++)
            {
                float v = s[i];
                sum += v;
                sumSq += v * v;
            }

            float mean = sum / dim;
            float var = MathF.Max(0f, (sumSq / dim) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(var + 1e-6f);

            for (int i = 0; i < dim; i++)
            {
                float w = weight != null ? weight[i] : 1.0f;
                float b = bias != null ? bias[i] : 0.0f;
                d[i] = ((s[i] - mean) * invStd) * w + b;
            }
        });
    }

    private static void ApplyGatedResidualCpu(
        float* target, float* update, float* gate, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* tgt = target + (long)t * dim;
            float* upd = update + (long)t * dim;

            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i] * gate[i];
            }
        });
    }

    private static void ApplyResidualCpu(
        float* target, float* update, int totalTokens, int dim)
    {
        Parallel.For(0, totalTokens, t =>
        {
            float* tgt = target + (long)t * dim;
            float* upd = update + (long)t * dim;

            for (int i = 0; i < dim; i++)
            {
                tgt[i] += upd[i];
            }
        });
    }

    private void ExecuteSelfAttentionCpu(
        float* normTokens, int totalTokens, WanBlockWeights blk, float* ropeCos, float* ropeSin, float* dstTokens)
    {
        var (qCols, qRows) = GetMatrixDims(blk.SelfQ_W);
        var (kCols, kRows) = GetMatrixDims(blk.SelfK_W);
        var (vCols, vRows) = GetMatrixDims(blk.SelfV_W);

        byte* qW = _gguf.GetTensorPointer(blk.SelfQ_W);
        byte* kW = _gguf.GetTensorPointer(blk.SelfK_W);
        byte* vW = _gguf.GetTensorPointer(blk.SelfV_W);

        float* qB = (float*)_gguf.GetTensorPointer(blk.SelfQ_B);
        float* kB = (float*)_gguf.GetTensorPointer(blk.SelfK_B);
        float* vB = (float*)_gguf.GetTensorPointer(blk.SelfV_B);

        QuantKernels.MatMulBatch(blk.SelfQ_W.Type, qW, normTokens, _qBuffer, qCols, qRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfK_W.Type, kW, normTokens, _kBuffer, kCols, kRows, totalTokens);
        QuantKernels.MatMulBatch(blk.SelfV_W.Type, vW, normTokens, _vBuffer, vCols, vRows, totalTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.SelfNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.SelfNormK);

        // Across-heads RMSNorm on Q and K
        Parallel.For(0, totalTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;

            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            float sumSqQ = 0f;
            float sumSqK = 0f;
            for (int d = 0; d < HiddenDim; d++)
            {
                sumSqQ += q[d] * q[d];
                sumSqK += k[d] * k[d];
            }
            float invRmsQ = 1.0f / MathF.Sqrt(sumSqQ / HiddenDim + 1e-6f);
            float invRmsK = 1.0f / MathF.Sqrt(sumSqK / HiddenDim + 1e-6f);

            for (int d = 0; d < HiddenDim; d++)
            {
                q[d] = (q[d] * invRmsQ) * normQScale[d];
                k[d] = (k[d] * invRmsK) * normKScale[d];
            }
        });

        // 3D-RoPE
        Apply3DRoPEToTokens(_qBuffer, totalTokens, ropeCos, ropeSin);
        Apply3DRoPEToTokens(_kBuffer, totalTokens, ropeCos, ropeSin);

        float scale = 1.0f / MathF.Sqrt(HeadDim);

        Parallel.For(0, NumHeads, h =>
        {
            float* scores = stackalloc float[totalTokens];

            for (int i = 0; i < totalTokens; i++)
            {
                float* qRow = _qBuffer + (long)i * HiddenDim + h * HeadDim;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < totalTokens; j++)
                {
                    float* kRow = _kBuffer + (long)j * HiddenDim + h * HeadDim;
                    float dot = DotHead(qRow, kRow) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < totalTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    sumExp += exp;
                }
                float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                float* outHead = dstTokens + (long)i * HiddenDim + h * HeadDim;
                for (int d = 0; d < HeadDim; d++) outHead[d] = 0f;

                for (int j = 0; j < totalTokens; j++)
                {
                    float w = scores[j] * invSum;
                    float* vRow = _vBuffer + (long)j * HiddenDim + h * HeadDim;
                    AccumulateHead(outHead, vRow, w);
                }
            }
        });

        var (oCols, oRows) = GetMatrixDims(blk.SelfO_W);
        byte* oW = _gguf.GetTensorPointer(blk.SelfO_W);
        float* oB = (float*)_gguf.GetTensorPointer(blk.SelfO_B);

        QuantKernels.MatMulBatch(blk.SelfO_W.Type, oW, dstTokens, _tokenBufferB, oCols, oRows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _tokenBufferB + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteCrossAttentionCpu(
        float* normTokens, int totalImgTokens, float* txtTokens, int numTxtTokens, WanBlockWeights blk, float* dstTokens)
    {
        var (qCols, qRows) = GetMatrixDims(blk.CrossQ_W);
        var (kCols, kRows) = GetMatrixDims(blk.CrossK_W);
        var (vCols, vRows) = GetMatrixDims(blk.CrossV_W);

        byte* qW = _gguf.GetTensorPointer(blk.CrossQ_W);
        byte* kW = _gguf.GetTensorPointer(blk.CrossK_W);
        byte* vW = _gguf.GetTensorPointer(blk.CrossV_W);

        float* qB = (float*)_gguf.GetTensorPointer(blk.CrossQ_B);
        float* kB = (float*)_gguf.GetTensorPointer(blk.CrossK_B);
        float* vB = (float*)_gguf.GetTensorPointer(blk.CrossV_B);

        QuantKernels.MatMulBatch(blk.CrossQ_W.Type, qW, normTokens, _qBuffer, qCols, qRows, totalImgTokens);
        QuantKernels.MatMulBatch(blk.CrossK_W.Type, kW, txtTokens, _kBuffer, kCols, kRows, numTxtTokens);
        QuantKernels.MatMulBatch(blk.CrossV_W.Type, vW, txtTokens, _vBuffer, vCols, vRows, numTxtTokens);

        float* normQScale = (float*)_gguf.GetTensorPointer(blk.CrossNormQ);
        float* normKScale = (float*)_gguf.GetTensorPointer(blk.CrossNormK);

        // Across-heads RMSNorm on Cross-Q
        Parallel.For(0, totalImgTokens, t =>
        {
            float* q = _qBuffer + (long)t * HiddenDim;
            if (qB != null) for (int d = 0; d < HiddenDim; d++) q[d] += qB[d];

            float sumSq = 0f;
            for (int d = 0; d < HiddenDim; d++) sumSq += q[d] * q[d];
            float invRms = 1.0f / MathF.Sqrt(sumSq / HiddenDim + 1e-6f);
            for (int d = 0; d < HiddenDim; d++) q[d] = (q[d] * invRms) * normQScale[d];
        });

        // Across-heads RMSNorm on Cross-K
        Parallel.For(0, numTxtTokens, t =>
        {
            float* k = _kBuffer + (long)t * HiddenDim;
            float* v = _vBuffer + (long)t * HiddenDim;
            if (kB != null) for (int d = 0; d < HiddenDim; d++) k[d] += kB[d];
            if (vB != null) for (int d = 0; d < HiddenDim; d++) v[d] += vB[d];

            float sumSq = 0f;
            for (int d = 0; d < HiddenDim; d++) sumSq += k[d] * k[d];
            float invRms = 1.0f / MathF.Sqrt(sumSq / HiddenDim + 1e-6f);
            for (int d = 0; d < HiddenDim; d++) k[d] = (k[d] * invRms) * normKScale[d];
        });

        float scale = 1.0f / MathF.Sqrt(HeadDim);
        Parallel.For(0, NumHeads, h =>
        {
            float* scores = stackalloc float[numTxtTokens];

            for (int i = 0; i < totalImgTokens; i++)
            {
                float* qRow = _qBuffer + (long)i * HiddenDim + h * HeadDim;

                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < numTxtTokens; j++)
                {
                    float* kRow = _kBuffer + (long)j * HiddenDim + h * HeadDim;
                    float dot = DotHead(qRow, kRow) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < numTxtTokens; j++)
                {
                    float exp = MathF.Exp(scores[j] - maxScore);
                    scores[j] = exp;
                    sumExp += exp;
                }
                float invSum = 1.0f / MathF.Max(sumExp, 1e-12f);

                float* outHead = dstTokens + (long)i * HiddenDim + h * HeadDim;
                for (int d = 0; d < HeadDim; d++) outHead[d] = 0f;

                for (int j = 0; j < numTxtTokens; j++)
                {
                    float w = scores[j] * invSum;
                    float* vRow = _vBuffer + (long)j * HiddenDim + h * HeadDim;
                    AccumulateHead(outHead, vRow, w);
                }
            }
        });

        var (oCols, oRows) = GetMatrixDims(blk.CrossO_W);
        byte* oW = _gguf.GetTensorPointer(blk.CrossO_W);
        float* oB = (float*)_gguf.GetTensorPointer(blk.CrossO_B);

        QuantKernels.MatMulBatch(blk.CrossO_W.Type, oW, dstTokens, _tokenBufferB, oCols, oRows, totalImgTokens);

        Parallel.For(0, totalImgTokens, t =>
        {
            float* outFinal = dstTokens + (long)t * HiddenDim;
            float* intermediate = _tokenBufferB + (long)t * HiddenDim;

            for (int d = 0; d < HiddenDim; d++)
            {
                float bVal = oB != null ? oB[d] : 0f;
                outFinal[d] = intermediate[d] + bVal;
            }
        });
    }

    private void ExecuteFFNCpu(float* normTokens, int totalTokens, WanBlockWeights blk, float* dstTokens)
    {
        var (f0Cols, f0Rows) = GetMatrixDims(blk.Ffn0_W);
        var (f2Cols, f2Rows) = GetMatrixDims(blk.Ffn2_W);

        byte* f0W = _gguf.GetTensorPointer(blk.Ffn0_W);
        byte* f2W = _gguf.GetTensorPointer(blk.Ffn2_W);

        float* f0B = (float*)_gguf.GetTensorPointer(blk.Ffn0_B);
        float* f2B = (float*)_gguf.GetTensorPointer(blk.Ffn2_B);

        QuantKernels.MatMulBatch(blk.Ffn0_W.Type, f0W, normTokens, _ffnIntermediate, f0Cols, f0Rows, totalTokens);

        Parallel.For(0, totalTokens, t =>
        {
            float* inter = _ffnIntermediate + (long)t * FfnDim;
            if (f0B != null)
            {
                for (int d = 0; d < FfnDim; d++) inter[d] += f0B[d];
            }
            ApplyGelu(inter, FfnDim);
        });

        QuantKernels.MatMulBatch(blk.Ffn2_W.Type, f2W, _ffnIntermediate, dstTokens, f2Cols, f2Rows, totalTokens);

        if (f2B != null)
        {
            Parallel.For(0, totalTokens, t =>
            {
                float* outToken = dstTokens + (long)t * HiddenDim;
                for (int d = 0; d < HiddenDim; d++) outToken[d] += f2B[d];
            });
        }
    }

    private void UnpatchifyAndOutput(float* tokens, int frames, int H, int W, Span<float> velocityOut)
    {
        int tokenH = H / PatchSize;
        int tokenW = W / PatchSize;
        int spatialTokens = tokenH * tokenW;
        int totalTokens = frames * spatialTokens;

        var (nCols, nRows) = GetMatrixDims(_headW); // 1536, 64
        byte* wPtr = _gguf.GetTensorPointer(_headW);
        float* bPtr = (float*)_gguf.GetTensorPointer(_headB);
        float* modPtr = (float*)_gguf.GetTensorPointer(_headMod);

        float* patchBufAll = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * PatchDim * sizeof(float)), 64);
        float* normTokens = (float*)NativeMemory.AlignedAlloc((nuint)(totalTokens * HiddenDim * sizeof(float)), 64);

        try
        {
            float* headShift = stackalloc float[HiddenDim];
            float* headScale = stackalloc float[HiddenDim];

            for (int d = 0; d < HiddenDim; d++)
            {
                headShift[d] = modPtr[0 * HiddenDim + d] + _timeVec[d];
                headScale[d] = modPtr[1 * HiddenDim + d] + _timeVec[d];
            }

            // Head LayerNorm + AdaLN modulation
            ApplyAdaLnCpu(tokens, normTokens, headShift, headScale, totalTokens, HiddenDim);

            // Project 1536 -> 64
            QuantKernels.MatMulBatch(_headW.Type, wPtr, normTokens, patchBufAll, nCols, nRows, totalTokens);

            if (bPtr != null)
            {
                Parallel.For(0, totalTokens, t =>
                {
                    float* p = patchBufAll + (long)t * PatchDim;
                    for (int d = 0; d < PatchDim; d++) p[d] += bPtr[d];
                });
            }

            // Unpatchify: 64 output channels are ordered (ph * PatchSize + pw) * InChannels + c
            fixed (float* pOut = velocityOut)
            {
                int tokenIdx = 0;
                for (int f = 0; f < frames; f++)
                {
                    int frameLatentOffset = f * (InChannels * H * W);

                    for (int th = 0; th < tokenH; th++)
                    {
                        for (int tw = 0; tw < tokenW; tw++)
                        {
                            float* patchBuf = patchBufAll + (long)tokenIdx * PatchDim;

                            for (int ph = 0; ph < PatchSize; ph++)
                            {
                                for (int pw = 0; pw < PatchSize; pw++)
                                {
                                    int y = th * PatchSize + ph;
                                    int x = tw * PatchSize + pw;

                                    for (int c = 0; c < InChannels; c++)
                                    {
                                        int pIdx = (ph * PatchSize + pw) * InChannels + c;
                                        pOut[frameLatentOffset + c * H * W + y * W + x] = patchBuf[pIdx];
                                    }
                                }
                            }
                            tokenIdx++;
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(patchBufAll);
            NativeMemory.AlignedFree(normTokens);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (int nCols, int nRows) GetMatrixDims(GgufTensorInfo w)
    {
        if (w.Dimensions.Length == 1)
            return (1, (int)w.Dimensions[0]);
        if (w.Dimensions.Length == 2)
            return ((int)w.Dimensions[0], (int)w.Dimensions[1]);

        int nRows = (int)w.Dimensions[^1];
        int nCols = 1;
        for (int i = 0; i < w.Dimensions.Length - 1; i++)
            nCols *= (int)w.Dimensions[i];
        return (nCols, nRows);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchLinearVec(GgufTensorInfo w, GgufTensorInfo? b, float* x, float* y)
    {
        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = b != null ? (float*)_gguf.GetTensorPointer(b) : null;
        var (nCols, nRows) = GetMatrixDims(w);

        QuantKernels.MatVecMul(w.Type, wPtr, x, y, nCols, nRows);
        if (bPtr != null)
        {
            for (int i = 0; i < nRows; i++) y[i] += bPtr[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotHead(float* a, float* b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vSum0 = Vector256<float>.Zero;
            var vSum1 = Vector256<float>.Zero;
            for (int d = 0; d < HeadDim; d += 16)
            {
                vSum0 += Vector256.Load(a + d) * Vector256.Load(b + d);
                vSum1 += Vector256.Load(a + d + 8) * Vector256.Load(b + d + 8);
            }
            return Vector256.Sum(vSum0 + vSum1);
        }
        float dot = 0f;
        for (int d = 0; d < HeadDim; d++) dot += a[d] * b[d];
        return dot;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateHead(float* dst, float* v, float w)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            var vw = Vector256.Create(w);
            for (int d = 0; d < HeadDim; d += 8)
            {
                var res = Vector256.Load(dst + d) + vw * Vector256.Load(v + d);
                res.Store(dst + d);
            }
            return;
        }
        for (int d = 0; d < HeadDim; d++) dst[d] += w * v[d];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySilu(float* x, int count)
    {
        for (int i = 0; i < count; i++)
        {
            x[i] = x[i] / (1.0f + MathF.Exp(-x[i]));
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

}
