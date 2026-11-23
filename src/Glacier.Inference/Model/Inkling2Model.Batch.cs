namespace Glacier.Inference.Model;

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
#if !ANDROID
using Glacier.Inference.Gpu.D3D12;
#endif
using Glacier.Inference.Quant;

public sealed unsafe partial class Inkling2Model
{
    private const int PrefillChunkSize = 16;

    /// <summary>
    /// True multi-token batched prefill for Inkling 2 Mini.
    /// Processes chunks of up to 16 tokens simultaneously, reusing every Attention and MoE weight
    /// across all tokens in the chunk (10x-20x less memory bandwidth than token-by-token prefill)
    /// while preserving exact causal ordering for short-convolutions and KV caches.
    /// </summary>
    private void ExecuteBatchPrefill(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits)
    {
        int offset = 0;
        while (offset < tokens.Length)
        {
            int chunkLen = Math.Min(PrefillChunkSize, tokens.Length - offset);
            bool isLastChunk = computeLogits && (offset + chunkLen == tokens.Length);
            ExecuteBatchChunk(
                tokens.Slice(offset, chunkLen),
                startPos + offset,
                isLastChunk ? logits : Span<float>.Empty,
                isLastChunk);
            offset += chunkLen;
        }
    }

    private void ExecuteBatchChunk(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits)
    {
        int B = tokens.Length;
        int embd = _embd;

        // 1. Token embeddings + RMSNorm for all B tokens
        float[] curBatch = new float[B * embd];
        float[] taus = new float[B];
        fixed (float* curBase = curBatch)
        {
            for (int t = 0; t < B; t++)
            {
                int tok = tokens[t];
                if ((uint)tok >= (uint)_w.VocabSizePadded) tok = 0;
                float* dst = curBase + t * embd;
                QuantKernels.ExtractEmbedding(_w.TokenEmbdType, _w.TokenEmbd, tok, dst, embd);
                RmsNorm(dst, _w.TokenEmbdNorm, dst, embd, _w.RmsNormEps);
                taus[t] = 1.0f + _w.LogScaleAlpha * MathF.Log(MathF.Max(1.0f, (startPos + t + 1.0f) / _w.LogScaleFloor));
            }
        }

        // Block-attention residual history: each entry is [B * embd]
        var blocks = new List<float[]>(16);

        // Prelude: physical layers 0 .. CoreStart-1
        for (int il = 0; il < _w.CoreStart; il++)
        {
            ExecuteLayerBatch(il, il, startPos, B, taus, curBatch, blocks, hasMessages: false, msgBatch: null);
        }

        // Save prelude output & blocks for 3-pass looped core
        float[] preludeBatch = (float[])curBatch.Clone();
        var preludeBlocks = new List<float[]>(blocks.Count);
        foreach (var b in blocks) preludeBlocks.Add((float[])b.Clone());

        // 3-Pass Looped Core
        int coreLen = _w.CoreEnd - _w.CoreStart;
        var bank = new List<float[]>(8);
        float[] bankOut = new float[B * embd];
        float[] msgBatch = new float[B * 4 * embd];

        for (int pass = 0; pass < _w.NumLoops; pass++)
        {
            if (pass > 0)
            {
                bank.Clear();
                for (int bi = preludeBlocks.Count + 1; bi < blocks.Count; bi++)
                    bank.Add(blocks[bi]);
                Array.Copy(curBatch, bankOut, B * embd);

                Array.Copy(preludeBatch, curBatch, B * embd);
                blocks.Clear();
                foreach (var b in preludeBlocks) blocks.Add((float[])b.Clone());
            }

            for (int ip = _w.CoreStart; ip < _w.CoreEnd; ip++)
            {
                int il = pass == 0 ? ip : _w.PhysicalLayerCount + (pass - 1) * coreLen + (ip - _w.CoreStart);
                if (pass == 0)
                {
                    ExecuteLayerBatch(il, ip, startPos, B, taus, curBatch, blocks, hasMessages: false, msgBatch: null);
                }
                else
                {
                    ComputeCrossPassMessagesBatch(ip, B, bank, bankOut, msgBatch);
                    ExecuteLayerBatch(il, ip, startPos, B, taus, curBatch, blocks, hasMessages: true, msgBatch: msgBatch);
                }
            }
        }

        // Epilogue: physical layers CoreEnd .. PhysicalLayerCount-1
        for (int il = _w.CoreEnd; il < _w.PhysicalLayerCount; il++)
        {
            ExecuteLayerBatch(il, il, startPos, B, taus, curBatch, blocks, hasMessages: false, msgBatch: null);
        }

        // Copy last token's hidden state into _cur (in case ExtractEmbedding/ForwardStage inspects it)
        Array.Copy(curBatch, (B - 1) * embd, _cur, 0, embd);

        if (!computeLogits || logits.IsEmpty)
            return;

        // Output head for ONLY the last token (t = B - 1)
        int lastT = B - 1;
        var lastBlocks = new List<float[]>(blocks.Count);
        foreach (var blk in blocks)
        {
            float[] slice = new float[embd];
            Array.Copy(blk, lastT * embd, slice, 0, embd);
            lastBlocks.Add(slice);
        }

        Mix(lastBlocks, _cur, _w.OutputResQuery, 1, _mixOutBuf);
        fixed (float* normOut = _cur)
        fixed (float* mixPtr = _mixOutBuf)
        {
            RmsNorm(mixPtr, _w.OutputNorm, normOut, embd, _w.RmsNormEps);
        }

        int vocabOut = Math.Min(_w.VocabSizeUnpadded, logits.Length);
#if !ANDROID
        if (_d3d12Offload != null && _d3d12Offload.OutputHeadPinned)
        {
            _d3d12Offload.ExecuteOutputHead(_cur, logits, vocabOut);
        }
        else
#endif
        {
            MatVecParallel(_w.OutputWeight, _w.OutputWeightType, _cur, logits, vocabOut, embd);
        }

        if (_w.FinalLogitSoftcapping > 0.0f)
        {
            float cap = _w.FinalLogitSoftcapping;
            float invCap = 1.0f / cap;
            for (int v = 0; v < vocabOut; v++)
            {
                logits[v] = cap * MathF.Tanh(logits[v] * invCap);
            }
        }
    }

    private void ComputeCrossPassMessagesBatch(int ip, int B, List<float[]> bank, float[] bankOut, float[] msgBatch)
    {
        var layer = _w.PhysicalLayers[ip];
        int embd = _embd;
        int cIdx = ip - _w.CoreStart;

        Parallel.For(0, B, t =>
        {
            float[] localCur = new float[embd];
            Array.Copy(bankOut, t * embd, localCur, 0, embd);
            float[] localOut = new float[4 * embd];

            if (layer.LoopMsgQuery != null)
            {
                MixTokenFromBatch(bank, t, localCur, layer.LoopMsgQuery, 4, localOut);
                Array.Copy(localOut, 0, msgBatch, t * 4 * embd, 4 * embd);
            }
            else if (_w.LoopResQuery != null && _w.LoopKvResQuery != null)
            {
                float[] tmp = new float[embd];
                MixTokenFromBatch(bank, t, localCur, _w.LoopResQuery + (2 * cIdx) * embd, 1, tmp);
                Array.Copy(tmp, 0, msgBatch, t * 4 * embd + 0 * embd, embd);

                MixTokenFromBatch(bank, t, localCur, _w.LoopKvResQuery + (2 * cIdx) * embd, 1, tmp);
                Array.Copy(tmp, 0, msgBatch, t * 4 * embd + 1 * embd, embd);

                MixTokenFromBatch(bank, t, localCur, _w.LoopKvResQuery + (2 * cIdx + 1) * embd, 1, tmp);
                Array.Copy(tmp, 0, msgBatch, t * 4 * embd + 2 * embd, embd);

                MixTokenFromBatch(bank, t, localCur, _w.LoopResQuery + (2 * cIdx + 1) * embd, 1, tmp);
                Array.Copy(tmp, 0, msgBatch, t * 4 * embd + 3 * embd, embd);
            }
        });
    }

    private void MixTokenFromBatch(List<float[]> blocksBatch, int tokenIdx, float[] current, float* queries, int nQueries, float[] outBuf)
    {
        int embd = _embd;
        int nStates = blocksBatch.Count + 1;
        float[] normAllArr = new float[nStates * embd];

        fixed (float* normAll = normAllArr)
        {
            for (int b = 0; b < blocksBatch.Count; b++)
            {
                fixed (float* src = blocksBatch[b])
                {
                    RmsNormNoWeight(src + tokenIdx * embd, normAll + b * embd, embd, 1e-6f);
                }
            }
            fixed (float* curPtr = current)
            {
                RmsNormNoWeight(curPtr, normAll + blocksBatch.Count * embd, embd, 1e-6f);
            }

            float invSqrtD = 1.0f / MathF.Sqrt(embd);
            Span<float> scores = stackalloc float[nStates];
            for (int q = 0; q < nQueries; q++)
            {
                float* qPtr = queries + q * embd;
                float maxScore = float.NegativeInfinity;
                for (int s = 0; s < nStates; s++)
                {
                    float dot = 0.0f;
                    float* st = normAll + s * embd;
                    for (int d = 0; d < embd; d++) dot += qPtr[d] * st[d];
                    dot *= invSqrtD;
                    scores[s] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0.0f;
                for (int s = 0; s < nStates; s++)
                {
                    float e = MathF.Exp(scores[s] - maxScore);
                    scores[s] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;

                int dstOff = q * embd;
                Array.Clear(outBuf, dstOff, embd);
                for (int s = 0; s < nStates; s++)
                {
                    float w = scores[s] * invSum;
                    float* st = normAll + s * embd;
                    for (int d = 0; d < embd; d++)
                        outBuf[dstOff + d] += w * st[d];
                }
            }
        }
    }

    private void ApplyLoopControlToken(float[] dstBatch, int dstOff, float[] msgBatch, int msgOff, Inkling2LayerWeights layer, int ctrl)
    {
        int embd = _embd;
        int rank = _w.GateRank;
        float[] cat = new float[2 * embd];
        float[] gate = new float[rank];
        float[] gain = new float[embd];
        float[] forget = new float[embd];

        fixed (float* dstPtr = dstBatch)
        fixed (float* msgPtr = msgBatch)
        fixed (float* catPtr = cat)
        {
            RmsNormNoWeight(dstPtr + dstOff, catPtr, embd, 1e-6f);
            RmsNormNoWeight(msgPtr + msgOff, catPtr + embd, embd, 1e-6f);
        }

        MatVecLocal(layer.LoopCtrlDown[ctrl], layer.LoopCtrlDownType[ctrl], cat, gate, rank, 2 * embd);
        MatVecLocal(layer.LoopCtrlGain[ctrl], layer.LoopCtrlGainType[ctrl], gate, gain, embd, rank);
        MatVecLocal(layer.LoopCtrlForget[ctrl], layer.LoopCtrlForgetType[ctrl], gate, forget, embd, rank);

        for (int d = 0; d < embd; d++)
        {
            float g = MathF.Tanh(gain[d]);
            float f = MathF.Tanh(forget[d]);
            dstBatch[dstOff + d] = (1.0f - f) * dstBatch[dstOff + d] + g * msgBatch[msgOff + d];
        }
    }

    private static void MatVecLocal(byte* weight, GgufType type, float[] x, float[] y, int rows, int cols)
    {
        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        fixed (float* xPtr = x)
        fixed (float* yPtr = y)
        {
            float* xSumsPtr = null;
            float[]? xSumsArr = null;
            if (type == GgufType.Q4_K)
            {
                int nb = cols / 32;
                xSumsArr = new float[nb];
                for (int b = 0; b < nb; b++)
                {
                    float s = 0f;
                    for (int j = 0; j < 32; j++) s += xPtr[b * 32 + j];
                    xSumsArr[b] = s;
                }
            }
            fixed (float* xsFixed = xSumsArr)
            {
                xSumsPtr = xsFixed;
                for (int r = 0; r < rows; r++)
                {
                    yPtr[r] = QuantKernels.ComputeDot(type, weight + r * rowBytes, xPtr, xSumsPtr, cols);
                }
            }
        }
    }

    private void ExecuteLayerBatch(
        int il,
        int ip,
        int startPos,
        int B,
        float[] taus,
        float[] curBatch,
        List<float[]> blocks,
        bool hasMessages,
        float[]? msgBatch)
    {
        var layer = _w.PhysicalLayers[ip];
        bool isSwa = layer.IsSwa;
        int embd = _embd;
        int nHead = layer.HeadCount;
        int nHeadKv = layer.HeadCountKv;
        int headDim = layer.HeadDim;
        int qDim = nHead * headDim;
        int kvDim = nHeadKv * headDim;
        int nRel = _w.RelativeLength;
        int rDim = nRel * nHead;

        float[] inpSaBatch = (float[])curBatch.Clone();
        float[] in0Batch = new float[B * embd];
        float[] in1Batch = new float[B * embd];
        float[] in2Batch = new float[B * embd];

        // 1. Per-token Block-Attention Residual Mix + LoopControl + RMSNorm (parallel over B)
        Parallel.For(0, B, t =>
        {
            int off = t * embd;
            if (layer.AttnResQuery != null)
            {
                float[] localCur = new float[embd];
                Array.Copy(curBatch, off, localCur, 0, embd);
                int nQ = isSwa ? 1 : 3;
                float[] mixOut = new float[nQ * embd];
                MixTokenFromBatch(blocks, t, localCur, layer.AttnResQuery, nQ, mixOut);
                Array.Copy(mixOut, 0, in0Batch, off, embd);
                Array.Copy(mixOut, isSwa ? 0 : embd, in1Batch, off, embd);
                Array.Copy(mixOut, isSwa ? 0 : 2 * embd, in2Batch, off, embd);
            }
            else
            {
                Array.Copy(curBatch, off, in0Batch, off, embd);
                Array.Copy(curBatch, off, in1Batch, off, embd);
                Array.Copy(curBatch, off, in2Batch, off, embd);
            }

            if (hasMessages && msgBatch != null)
            {
                int mBase = t * 4 * embd;
                ApplyLoopControlToken(in0Batch, off, msgBatch, mBase + 0 * embd, layer, 0);
                ApplyLoopControlToken(in1Batch, off, msgBatch, mBase + 1 * embd, layer, 1);
                ApplyLoopControlToken(in2Batch, off, msgBatch, mBase + 2 * embd, layer, 2);
            }

            fixed (float* p0Base = in0Batch, p1Base = in1Batch, p2Base = in2Batch)
            {
                float* p0 = p0Base + off;
                float* p1 = p1Base + off;
                float* p2 = p2Base + off;
                if (isSwa && hasMessages)
                {
                    RmsNorm(p0, layer.AttnNorm, p0, embd, _w.RmsNormEps);
                    RmsNorm(p1, layer.AttnNorm, p1, embd, _w.RmsNormEps);
                    RmsNorm(p2, layer.AttnNorm, p2, embd, _w.RmsNormEps);
                }
                else if (isSwa)
                {
                    RmsNorm(p0, layer.AttnNorm, p0, embd, _w.RmsNormEps);
                    Array.Copy(in0Batch, off, in1Batch, off, embd);
                    Array.Copy(in0Batch, off, in2Batch, off, embd);
                }
                else
                {
                    RmsNorm(p0, layer.AttnQInpNorm, p0, embd, _w.RmsNormEps);
                    RmsNorm(p1, layer.AttnKInpNorm, p1, embd, _w.RmsNormEps);
                    RmsNorm(p2, layer.AttnVInpNorm, p2, embd, _w.RmsNormEps);
                }
            }
        });

        // 2. TRUE BATCHED Q, K, V, R, AttnGate, Sinks projections (weights read ONCE for all B tokens!)
        float[] qBatch = new float[B * qDim];
        float[] kBatch = new float[B * kvDim];
        float[] vBatch = new float[B * kvDim];
        float[] rBatch = new float[B * rDim];
        float[] gateBatch = new float[B * qDim];
        float[] sinksBatch = new float[B * nHead];

        fixed (float* p0 = in0Batch, p1 = in1Batch, p2 = in2Batch)
        fixed (float* pq = qBatch, pk = kBatch, pv = vBatch, pr = rBatch, pg = gateBatch, ps = sinksBatch)
        {
            QuantKernels.MatMulBatch(layer.WqType, layer.Wq, p0, pq, embd, qDim, B);
            QuantKernels.MatMulBatch(layer.WkType, layer.Wk, p1, pk, embd, kvDim, B);
            QuantKernels.MatMulBatch(layer.WvType, layer.Wv, p2, pv, embd, kvDim, B);
            QuantKernels.MatMulBatch(layer.AttnRelType, layer.AttnRel, p0, pr, embd, rDim, B);
            QuantKernels.MatMulBatch(layer.AttnGateType, layer.AttnGate, p0, pg, embd, qDim, B);
            QuantKernels.MatMulBatch(layer.AttnSinkProjType, layer.AttnSinkProj, p0, ps, embd, nHead, B);
        }

        // Dynamic taps for Global layers (batched)
        float[]? kDynBatch = null;
        float[]? vDynBatch = null;
        int dynDim = _w.ShortConvLCache * nHeadKv;
        if (!isSwa && layer.AttnKConvProj != null && layer.AttnVConvProj != null)
        {
            kDynBatch = new float[B * dynDim];
            vDynBatch = new float[B * dynDim];
            fixed (float* p1 = in1Batch, p2 = in2Batch, pkd = kDynBatch, pvd = vDynBatch)
            {
                QuantKernels.MatMulBatch(layer.AttnKConvProjType, layer.AttnKConvProj, p1, pkd, embd, dynDim, B);
                QuantKernels.MatMulBatch(layer.AttnVConvProjType, layer.AttnVConvProj, p2, pvd, embd, dynDim, B);
            }
        }

        // Causal Short-Conv + Clip + V RMSNorm + tau scaling + KV cache write (strictly chronological t = 0..B-1)
        int cap = _kvCap[il];
        float[] tmpQ = new float[qDim];
        float[] tmpK = new float[kvDim];
        float[] tmpV = new float[kvDim];
        float[] tmpR = new float[rDim];
        float[]? tmpKDyn = kDynBatch != null ? new float[dynDim] : null;
        float[]? tmpVDyn = vDynBatch != null ? new float[dynDim] : null;

        for (int t = 0; t < B; t++)
        {
            Array.Copy(qBatch, t * qDim, tmpQ, 0, qDim);
            Array.Copy(kBatch, t * kvDim, tmpK, 0, kvDim);
            Array.Copy(vBatch, t * kvDim, tmpV, 0, kvDim);
            Array.Copy(rBatch, t * rDim, tmpR, 0, rDim);

            if (kDynBatch != null && vDynBatch != null && tmpKDyn != null && tmpVDyn != null)
            {
                Array.Copy(kDynBatch, t * dynDim, tmpKDyn, 0, dynDim);
                Array.Copy(vDynBatch, t * dynDim, tmpVDyn, 0, dynDim);
            }

            ApplyShortConv(tmpK, kvDim, layer.AttnKConv, tmpKDyn, nHeadKv, _kConvHist[il]);
            ApplyShortConv(tmpV, kvDim, layer.AttnVConv, tmpVDyn, nHeadKv, _vConvHist[il]);

            ClipPerHead(tmpQ, headDim, nHead, layer.AttnQClip, _w.QkClip);
            ClipPerHead(tmpK, headDim, nHeadKv, null, _w.QkClip);
            RmsNormPerHead(tmpV, headDim, nHeadKv, _w.RmsNormEps);
            ClipPerHead(tmpR, nRel, nHead, layer.AttnRelClip, _w.RelativeClip);

            if (!isSwa)
            {
                ScaleSpan(tmpQ, taus[t]);
                ScaleSpan(tmpR, taus[t]);
            }

            Array.Copy(tmpQ, 0, qBatch, t * qDim, qDim);
            Array.Copy(tmpR, 0, rBatch, t * rDim, rDim);

            int pos = startPos + t;
            int slot = pos % cap;
            Array.Copy(tmpK, 0, _kCache[il], slot * kvDim, kvDim);
            Array.Copy(tmpV, 0, _vCache[il], slot * kvDim, kvDim);
        }

        // 3. Causal Relative Positional Attention across all B * nHead items in parallel!
        float[] attnOutBatch = new float[B * qDim];
        int groupSize = nHead / nHeadKv;
        float invSqrtD = 1.0f / MathF.Sqrt(headDim);
        int relExtent = layer.RelExtent;

        Parallel.For(0, B * nHead, th =>
        {
            int t = th / nHead;
            int h = th - t * nHead;
            int pos = startPos + t;
            int startT = isSwa ? Math.Max(0, pos - _w.SlidingWindow + 1) : 0;
            int kvLen = pos - startT + 1;
            int kvHead = h / groupSize;

            int qOff = t * qDim + h * headDim;
            int rOff = t * rDim + h * nRel;
            float sinkVal = sinksBatch[t * nHead + h];
            float[] kCacheL = _kCache[il];
            float[] vCacheL = _vCache[il];

            float[] scores = new float[kvLen];
            float maxLogit = 0.0f;

            for (int ti = 0; ti < kvLen; ti++)
            {
                int kt = startT + ti;
                int dist = pos - kt;
                int kOff = (kt % cap) * kvDim + kvHead * headDim;

                float dot = DotProduct(qBatch, qOff, kCacheL, kOff, headDim) * invSqrtD;
                float relBias = 0.0f;
                if (dist < relExtent)
                {
                    float* biasRow = layer.AttnRelBias + dist * nRel;
                    for (int ri = 0; ri < nRel; ri++)
                        relBias += rBatch[rOff + ri] * biasRow[ri];
                }

                float s = dot + relBias - sinkVal;
                scores[ti] = s;
                if (s > maxLogit) maxLogit = s;
            }

            float sumExp = MathF.Exp(-maxLogit);
            for (int ti = 0; ti < kvLen; ti++)
            {
                float e = MathF.Exp(scores[ti] - maxLogit);
                scores[ti] = e;
                sumExp += e;
            }

            float invSum = 1.0f / sumExp;
            int outOff = t * qDim + h * headDim;
            for (int ti = 0; ti < kvLen; ti++)
            {
                float w = scores[ti] * invSum;
                int vOff = ((startT + ti) % cap) * kvDim + kvHead * headDim;
                for (int d = 0; d < headDim; d++)
                    attnOutBatch[outOff + d] += w * vCacheL[vOff + d];
            }

            // Apply sigmoid attention output gate inline
            for (int d = 0; d < headDim; d++)
            {
                float sig = 1.0f / (1.0f + MathF.Exp(-gateBatch[outOff + d]));
                attnOutBatch[outOff + d] *= sig;
            }
        });

        // Batched Wo projection (weights read ONCE for all B tokens!)
        float[] attnProjBatch = new float[B * embd];
        fixed (float* pAo = attnOutBatch, pAp = attnProjBatch)
        {
            QuantKernels.MatMulBatch(layer.WoType, layer.Wo, pAo, pAp, qDim, embd, B);
        }

        // Causal AttnPostConv (strictly chronological t = 0..B-1)
        float[] tmpEmb = new float[embd];
        for (int t = 0; t < B; t++)
        {
            Array.Copy(attnProjBatch, t * embd, tmpEmb, 0, embd);
            ApplyShortConv(tmpEmb, embd, layer.AttnPostConv, null, 0, _attnPostConvHist[il]);
            Array.Copy(tmpEmb, 0, attnProjBatch, t * embd, embd);
        }

        // Block residual stream boundary
        if (ip % _w.AttnResBlockSize == 0)
        {
            blocks.Add(inpSaBatch);
            Array.Copy(attnProjBatch, curBatch, B * embd);
        }
        else
        {
            for (int i = 0; i < B * embd; i++)
                curBatch[i] = inpSaBatch[i] + attnProjBatch[i];
        }

        float[] ffnInpBatch = (float[])curBatch.Clone();

        // 4. MoE FFN with SiTU + Post-Conv
        Parallel.For(0, B, t =>
        {
            int off = t * embd;
            float[] localCur = new float[embd];
            Array.Copy(curBatch, off, localCur, 0, embd);
            float[] mixOut = new float[embd];
            MixTokenFromBatch(blocks, t, localCur, layer.FfnResQuery, 1, mixOut);
            Array.Copy(mixOut, 0, curBatch, off, embd);

            if (hasMessages && msgBatch != null)
            {
                ApplyLoopControlToken(curBatch, off, msgBatch, t * 4 * embd + 3 * embd, layer, 3);
            }

            fixed (float* pCurBase = curBatch)
            {
                RmsNorm(pCurBase + off, layer.FfnNorm, pCurBase + off, embd, _w.RmsNormEps);
            }
        });

        // Batched Router
        int nExp = _w.ExpertCount;
        int topK = _w.ExpertUsedCount;
        float[] routerBatch = new float[B * nExp];
        fixed (float* pCur = curBatch, pRtr = routerBatch)
        {
            QuantKernels.MatMulBatch(layer.FfnGateInpType, layer.FfnGateInp, pCur, pRtr, embd, nExp, B);
        }

        int[] activeIdsFlat = new int[B * topK];
        float[] activeWeightsFlat = new float[B * topK];
        Span<int> topIds = stackalloc int[topK];
        Span<float> topProbs = stackalloc float[topK];
        Span<float> topScores = stackalloc float[topK];
        for (int t = 0; t < B; t++)
        {
            int rBase = t * nExp;
            int outBase = t * topK;
            for (int k = 0; k < topK; k++) { topIds[k] = -1; topScores[k] = float.NegativeInfinity; }

            for (int e = 0; e < nExp; e++)
            {
                float prob = 1.0f / (1.0f + MathF.Exp(-routerBatch[rBase + e]));
                float score = prob + (layer.ExpProbsBias != null ? layer.ExpProbsBias[e] : 0.0f);
                if (score > topScores[topK - 1])
                {
                    int posIns = topK - 1;
                    while (posIns > 0 && score > topScores[posIns - 1])
                    {
                        topScores[posIns] = topScores[posIns - 1];
                        topProbs[posIns] = topProbs[posIns - 1];
                        topIds[posIns] = topIds[posIns - 1];
                        posIns--;
                    }
                    topScores[posIns] = score;
                    topProbs[posIns] = prob;
                    topIds[posIns] = e;
                }
            }

            float probSum = 1e-20f;
            for (int k = 0; k < topK; k++) probSum += topProbs[k];
            float wMul = _w.ExpertWeightsScale / probSum;
            for (int k = 0; k < topK; k++)
            {
                activeIdsFlat[outBase + k] = topIds[k];
                activeWeightsFlat[outBase + k] = topProbs[k] * wMul;
            }
        }

        float[] moeOutBatch = new float[B * embd];
#if !ANDROID
        if (_d3d12Offload != null && _d3d12Offload.IsLayerPinned(ip))
        {
            _d3d12Offload.ExecuteMoeLayerBatch(ip, curBatch, B, activeIdsFlat, activeWeightsFlat, moeOutBatch);
        }
        else
#endif
        {
            // Grouped-by-Expert Batched CPU MoE: reads each NVFP4 expert ONCE for all routed tokens!
            int ffExp = _w.ExpertFfnDim;
            long guRowBytes = GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, embd);
            long dnRowBytes = GgufTypes.GetRowBytes(layer.FfnDownExpsType, ffExp);
            long gateUpExpBytes = (long)(2 * ffExp) * guRowBytes;
            long downExpBytes = (long)embd * dnRowBytes;
            float beta = _w.SituBeta;
            float invBeta = 1.0f / beta;
            float lb = _w.SituLinearBeta;
            float invLb = lb > 0.0f ? 1.0f / lb : 0.0f;

            var buckets = new List<(int Token, float Weight)>[nExp];
            for (int t = 0; t < B; t++)
            {
                for (int k = 0; k < topK; k++)
                {
                    int expId = activeIdsFlat[t * topK + k];
                    float w = activeWeightsFlat[t * topK + k];
                    (buckets[expId] ??= new List<(int, float)>(4)).Add((t, w));
                }
            }

            for (int expId = 0; expId < nExp; expId++)
            {
                var bucket = buckets[expId];
                if (bucket == null || bucket.Count == 0) continue;

                int m = bucket.Count;
                float[] expIn = new float[m * embd];
                for (int bi = 0; bi < m; bi++)
                    Array.Copy(curBatch, bucket[bi].Token * embd, expIn, bi * embd, embd);

                float[] expGu = new float[m * 2 * ffExp];
                byte* guPtr = layer.FfnGateUpExps + (long)expId * gateUpExpBytes;
                fixed (float* pIn = expIn, pGu = expGu)
                {
                    QuantKernels.MatMulBatch(layer.FfnGateUpExpsType, guPtr, pIn, pGu, embd, 2 * ffExp, m);
                }

                float guScale = layer.FfnGateUpExpsScale != null ? layer.FfnGateUpExpsScale[expId] : 1.0f;
                float[] expHid = new float[m * ffExp];
                for (int bi = 0; bi < m; bi++)
                {
                    int guOff = bi * 2 * ffExp;
                    int hOff = bi * ffExp;
                    for (int j = 0; j < ffExp; j++)
                    {
                        float g = expGu[guOff + j] * guScale;
                        float u = expGu[guOff + ffExp + j] * guScale;
                        float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                        if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                        expHid[hOff + j] = act * u;
                    }
                }

                float[] expDn = new float[m * embd];
                byte* dnPtr = layer.FfnDownExps + (long)expId * downExpBytes;
                fixed (float* pHid = expHid, pDn = expDn)
                {
                    QuantKernels.MatMulBatch(layer.FfnDownExpsType, dnPtr, pHid, pDn, ffExp, embd, m);
                }

                float dnScale = layer.FfnDownExpsScale != null ? layer.FfnDownExpsScale[expId] : 1.0f;
                for (int bi = 0; bi < m; bi++)
                {
                    float w = bucket[bi].Weight * dnScale;
                    int dstOff = bucket[bi].Token * embd;
                    int srcOff = bi * embd;
                    for (int d = 0; d < embd; d++)
                        moeOutBatch[dstOff + d] += w * expDn[srcOff + d];
                }
            }
        }

        // Causal FfnPostConv + residual add (strictly chronological t = 0..B-1)
        for (int t = 0; t < B; t++)
        {
            Array.Copy(moeOutBatch, t * embd, tmpEmb, 0, embd);
            ApplyShortConv(tmpEmb, embd, layer.FfnPostConv, null, 0, _ffnPostConvHist[il]);
            int off = t * embd;
            for (int d = 0; d < embd; d++)
                curBatch[off + d] = ffnInpBatch[off + d] + tmpEmb[d];
        }
    }
}
