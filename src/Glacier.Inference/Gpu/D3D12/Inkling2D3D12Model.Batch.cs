#if !ANDROID
namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;

public sealed unsafe partial class Inkling2D3D12Model
{
    // Batch-specific root signatures & PSOs (cs_5_0: NVIDIA + AMD RDNA 3.5)
    private ID3D12PipelineState _psoQ8_0Batch = null!;
    private ID3D12PipelineState _psoQ6KBatch = null!;
    private ID3D12PipelineState _psoQ5KBatch = null!;
    private ID3D12PipelineState _psoF32Batch = null!;
    private ID3D12PipelineState _psoNvfp4IndBatch = null!;

    private ID3D12RootSignature _sigRmsNormBatch = null!;    // b0(8), t0, u0, u1
    private ID3D12PipelineState _psoRmsNormBatch = null!;

    private ID3D12PipelineState _psoMixBatch = null!;        // reuses _sigMix (b0(6), t0, u0, u1, u2)
    private ID3D12PipelineState _psoLoopBlendBatch = null!;  // reuses _sigLoopBlend (b0(4), u0..u3)

    private ID3D12RootSignature _sigShortConvBatch = null!;  // b0(6), t0, u0, u1, u2
    private ID3D12PipelineState _psoShortConvBatch = null!;

    private ID3D12RootSignature _sigHeadNormBatch = null!;   // b0(12), t0, u0, u1
    private ID3D12PipelineState _psoHeadNormBatch = null!;

    private ID3D12RootSignature _sigAttnBatch = null!;       // b0(10), t0, u0..u6
    private ID3D12PipelineState _psoAttnBatch = null!;

    private ID3D12RootSignature _sigRouterBatch = null!;     // b0(8), t0, t1, t2, u0, u1, u2
    private ID3D12PipelineState _psoRouterBatch = null!;

    private void InitBatchPipelines()
    {
        _psoQ8_0Batch = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemmQ8_0Batch));
        _psoQ6KBatch = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemmQ6KBatch));
        _psoQ5KBatch = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemmQ5KBatch));
        _psoF32Batch = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.Inkling2GemmF32Batch));
        _psoNvfp4IndBatch = _ctx.CreatePipelineState(_sigNvfp4Ind, _ctx.CompileShader(D3D12Shaders.Inkling2GemvNVFP4IndirectBatch));

        _sigRmsNormBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 8), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
        }));
        _psoRmsNormBatch = _ctx.CreatePipelineState(_sigRmsNormBatch, _ctx.CompileShader(D3D12Shaders.Inkling2RmsNormBatch));

        _psoMixBatch = _ctx.CreatePipelineState(_sigMix, _ctx.CompileShader(D3D12Shaders.Inkling2MixBatch));
        _psoLoopBlendBatch = _ctx.CreatePipelineState(_sigLoopBlend, _ctx.CompileShader(D3D12Shaders.Inkling2LoopCtrlBlendBatch));

        _sigShortConvBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
        }));
        _psoShortConvBatch = _ctx.CreatePipelineState(_sigShortConvBatch, _ctx.CompileShader(D3D12Shaders.Inkling2ShortConvBatch));

        _sigHeadNormBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 12), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
        }));
        _psoHeadNormBatch = _ctx.CreatePipelineState(_sigHeadNormBatch, _ctx.CompileShader(D3D12Shaders.Inkling2HeadNormClipKvBatch));

        _sigAttnBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 10), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(4, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(5, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(6, 0), ShaderVisibility.All),
        }));
        _psoAttnBatch = _ctx.CreatePipelineState(_sigAttnBatch, _ctx.CompileShader(D3D12Shaders.Inkling2AttnBatch));

        _sigRouterBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new(new RootConstants(0, 0, 8), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
        }));
        _psoRouterBatch = _ctx.CreatePipelineState(_sigRouterBatch, _ctx.CompileShader(D3D12Shaders.Inkling2RouterTopKBatch));
    }

    private void DisposeBatchPipelines()
    {
        _psoQ8_0Batch?.Dispose();
        _psoQ6KBatch?.Dispose();
        _psoQ5KBatch?.Dispose();
        _psoF32Batch?.Dispose();
        _psoNvfp4IndBatch?.Dispose();
        _psoRmsNormBatch?.Dispose(); _sigRmsNormBatch?.Dispose();
        _psoMixBatch?.Dispose();
        _psoLoopBlendBatch?.Dispose();
        _psoShortConvBatch?.Dispose(); _sigShortConvBatch?.Dispose();
        _psoHeadNormBatch?.Dispose(); _sigHeadNormBatch?.Dispose();
        _psoAttnBatch?.Dispose(); _sigAttnBatch?.Dispose();
        _psoRouterBatch?.Dispose(); _sigRouterBatch?.Dispose();
    }

    /// <summary>
    /// Runs the full 48-layer Inkling-2 GPU forward pass over <paramref name="tokens"/> and writes each token's
    /// post-OutputNorm hidden vector ([tokens.Length * EmbeddingDim]) into <paramref name="outHiddenStates"/>.
    /// </summary>
    public void ExtractHiddenStatesBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> outHiddenStates)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) return;
        if (outHiddenStates.Length < tokens.Length * _embd)
            throw new ArgumentException($"Output buffer too small ({outHiddenStates.Length} < {tokens.Length * _embd}).", nameof(outHiddenStates));

        int offset = 0;
        while (offset < tokens.Length)
        {
            int chunkLen = Math.Min(MaxPrefillBatch, tokens.Length - offset);
            ExecuteBatchChunkGpu(
                tokens.Slice(offset, chunkLen),
                startPos + offset,
                Span<float>.Empty,
                computeLogits: false,
                outHiddenChunk: outHiddenStates.Slice(offset * _embd, chunkLen * _embd));
            offset += chunkLen;
        }
    }

    private void ExecuteBatchPrefill(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits)
    {
        int offset = 0;
        while (offset < tokens.Length)
        {
            int chunkLen = Math.Min(MaxPrefillBatch, tokens.Length - offset);
            bool isLastChunk = computeLogits && (offset + chunkLen == tokens.Length);
            if (chunkLen == 1)
            {
                Forward(tokens[offset], startPos + offset, null!, isLastChunk ? logits : Span<float>.Empty, isLastChunk);
            }
            else
            {
                ExecuteBatchChunkGpu(tokens.Slice(offset, chunkLen), startPos + offset, isLastChunk ? logits : Span<float>.Empty, isLastChunk);
            }
            offset += chunkLen;
        }
    }

    private void ExecuteBatchChunkGpu(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits, Span<float> outHiddenChunk = default)
    {
        int B = tokens.Length;
        int embd = _embd;
        ulong bEmbdBytes = (ulong)(B * embd) * 4UL;
        int blockStrideFloats = B * embd;

        // 1. Extract all B token embeddings into mapped _uploadBuf
        for (int t = 0; t < B; t++)
        {
            int tok = tokens[t];
            if ((uint)tok >= (uint)_w.VocabSizePadded) tok = 0;
            QuantKernels.ExtractEmbedding(_w.TokenEmbdType, _w.TokenEmbd, tok, _pUpload + t * embd, embd);
        }

        _ctx.BeginCommands();
        var cmd = _ctx.CommandList;

        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
        cmd.CopyBufferRegion(_curBuf, 0, _uploadBuf, 0, bEmbdBytes);
        cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

        // TokenEmbdNorm across all B tokens in parallel
        DispatchRmsNormBatch(cmd, _tokenEmbdNormGpu, _curBuf, _curBuf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
        _ctx.UavBarrier(_curBuf);

        int blocksCount = 0;

        // Prelude: physical layers 0 .. CoreStart-1
        for (int il = 0; il < _w.CoreStart; il++)
        {
            ExecuteLayerBatchGpu(ref cmd, il, il, startPos, B, ref blocksCount, hasMessages: false);
        }

        // Save prelude output & blocks on GPU
        int preludeBlocksCount = blocksCount;
        GpuCopy(cmd, _preludeBuf, 0, _curBuf, 0, bEmbdBytes, barrier: false);
        if (preludeBlocksCount > 0)
            GpuCopy(cmd, _preludeBlocksBuf, 0, _blocksBuf, 0, (ulong)preludeBlocksCount * bEmbdBytes, barrier: false);
        _ctx.UavBarrier();

        // 3-Pass Looped Core
        int coreLen = _w.CoreEnd - _w.CoreStart;
        int bankCount = 0;

        for (int pass = 0; pass < _w.NumLoops; pass++)
        {
            if (pass > 0)
            {
                int bankStart = preludeBlocksCount + 1;
                bankCount = Math.Max(0, blocksCount - bankStart);
                if (bankCount > 0)
                    GpuCopy(cmd, _bankBuf, 0, _blocksBuf, (ulong)bankStart * bEmbdBytes, (ulong)bankCount * bEmbdBytes, barrier: false);
                GpuCopy(cmd, _bankOutBuf, 0, _curBuf, 0, bEmbdBytes, barrier: false);
                _ctx.UavBarrier();

                GpuCopy(cmd, _curBuf, 0, _preludeBuf, 0, bEmbdBytes, barrier: false);
                if (preludeBlocksCount > 0)
                    GpuCopy(cmd, _blocksBuf, 0, _preludeBlocksBuf, 0, (ulong)preludeBlocksCount * bEmbdBytes, barrier: false);
                _ctx.UavBarrier();
                blocksCount = preludeBlocksCount;
            }

            for (int ip = _w.CoreStart; ip < _w.CoreEnd; ip++)
            {
                int il = pass == 0 ? ip : _nPhys + (pass - 1) * coreLen + (ip - _w.CoreStart);
                if (pass > 0)
                {
                    ComputeCrossPassMessagesBatchGpu(cmd, ip, bankCount, B);
                }
                ExecuteLayerBatchGpu(ref cmd, il, ip, startPos, B, ref blocksCount, hasMessages: pass > 0);
            }
        }

        // Epilogue: physical layers CoreEnd .. PhysicalLayerCount-1
        for (int il = _w.CoreEnd; il < _nPhys; il++)
        {
            ExecuteLayerBatchGpu(ref cmd, il, il, startPos, B, ref blocksCount, hasMessages: false);
        }

        if (!outHiddenChunk.IsEmpty)
        {
            DispatchMixBatch(cmd, _outputResQueryGpu, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, embd, 0, blockStrideFloats, B);
            _ctx.UavBarrier(_mixOutBuf);

            DispatchRmsNormBatch(cmd, _outputNormGpu, _mixOutBuf, _curBuf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            _ctx.UavBarrier(_curBuf);

            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_readbackBuf, 0, _curBuf, 0, bEmbdBytes);
            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            fixed (float* pDst = outHiddenChunk)
            {
                Buffer.MemoryCopy(_pReadback, pDst, bEmbdBytes, bEmbdBytes);
            }
        }
        else if (computeLogits && !logits.IsEmpty)
        {
            int vocabOut = Math.Min(_w.VocabSizeUnpadded, logits.Length);
            // Compute Mix across all B tokens (fast!) then run OutputNorm + Q6_K lm_head ONLY on last token (t = B - 1)!
            DispatchMixBatch(cmd, _outputResQueryGpu, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, embd, 0, blockStrideFloats, B);
            _ctx.UavBarrier(_mixOutBuf);

            int lastOff = (B - 1) * embd;
            DispatchRmsNorm(cmd, _outputNormGpu, _mixOutBuf, _curBuf, embd, _w.RmsNormEps, lastOff, 0, hasWeight: true);
            _ctx.UavBarrier(_curBuf);

            // Q6_K lm_head for ONLY the last token
            cmd.SetComputeRootSignature(_sigGemv);
            cmd.SetPipelineState(_psoQ6K);
            cmd.SetComputeRootShaderResourceView(1, _outputWeightGpu.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _curBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _logitsBuf.GPUVirtualAddress);

            uint* c = stackalloc uint[4];
            c[0] = (uint)embd;
            c[1] = (uint)vocabOut;
            c[2] = 0u;
            int rowOff = 0;
            while (rowOff < vocabOut)
            {
                int rowsThis = Math.Min(vocabOut - rowOff, D3D12TensorAlign.GemvChunkRows);
                c[3] = (uint)rowOff;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
                cmd.Dispatch((uint)((rowsThis + 15) / 16), 1, 1);
                rowOff += rowsThis;
            }
            _ctx.UavBarrier(_logitsBuf);

            bool useHeadLora = HeadLora != null;
            if (!useHeadLora && _w.FinalLogitSoftcapping > 0.0f)
            {
                cmd.SetComputeRootSignature(_sigSoftcap);
                cmd.SetPipelineState(_psoSoftcap);
                cmd.SetComputeRootUnorderedAccessView(1, _logitsBuf.GPUVirtualAddress);
                c[0] = (uint)vocabOut;
                c[1] = BitConverter.SingleToUInt32Bits(_w.FinalLogitSoftcapping);
                c[2] = BitConverter.SingleToUInt32Bits(1.0f / _w.FinalLogitSoftcapping);
                c[3] = 0u;
                cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
                cmd.Dispatch((uint)((vocabOut + 255) / 256), 1, 1);
                _ctx.UavBarrier(_logitsBuf);
            }

            ulong outBytes = (ulong)vocabOut * 4UL;
            cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_readbackBuf, 0, _logitsBuf, 0, outBytes);
            cmd.ResourceBarrierTransition(_logitsBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            if (useHeadLora)
            {
                cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_cpuMoeReadbackBuf, 0, _curBuf, 0, (ulong)embd * 4UL);
                cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            }

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            fixed (float* pDst = logits)
            {
                Buffer.MemoryCopy(_pReadback, pDst, outBytes, outBytes);
            }

            if (useHeadLora)
            {
                HeadLora!.ApplyDelta(new ReadOnlySpan<float>((float*)_pCpuMoeReadback, embd), logits.Slice(0, vocabOut));
                Inkling2HeadLoraAdapter.ApplySoftcapInPlace(logits.Slice(0, vocabOut), _w.FinalLogitSoftcapping);
            }
        }
        else
        {
            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();
        }
    }

    private void ComputeCrossPassMessagesBatchGpu(ID3D12GraphicsCommandList cmd, int ip, int bankCount, int B)
    {
        var lg = _layers[ip];
        int embd = _embd;
        int blockStride = B * embd;
        int msgStride = 4 * embd;

        if (lg.LoopMsgQuery != null)
        {
            DispatchMixBatch(cmd, lg.LoopMsgQuery, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 4, 0, msgStride, 0, blockStride, B);
            _ctx.UavBarrier(_msgBuf);
            return;
        }

        int cIdx = ip - _w.CoreStart;
        if (_loopResQueryGpu != null && _loopKvResQueryGpu != null)
        {
            DispatchMixBatch(cmd, _loopResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx) * embd, msgStride, 0 * embd, blockStride, B);
            DispatchMixBatch(cmd, _loopKvResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx) * embd, msgStride, 1 * embd, blockStride, B);
            DispatchMixBatch(cmd, _loopKvResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx + 1) * embd, msgStride, 2 * embd, blockStride, B);
            DispatchMixBatch(cmd, _loopResQueryGpu, _bankBuf, _bankOutBuf, _msgBuf, bankCount, 1, (2 * cIdx + 1) * embd, msgStride, 3 * embd, blockStride, B);
            _ctx.UavBarrier(_msgBuf);
        }
    }

    private void ApplyLoopControlBatchGpu(ID3D12GraphicsCommandList cmd, ID3D12Resource targetBuf, int msgSubOff, LayerGpu lg, int ctrl, int B)
    {
        int embd = _embd;
        int catStride = 2 * embd;
        int msgStride = 4 * embd;

        DispatchRmsNormBatch(cmd, _dummySrv, targetBuf, _ctrlCatBuf, embd, 1e-6f, embd, 0, catStride, 0, false, B);
        DispatchRmsNormBatch(cmd, _dummySrv, _msgBuf, _ctrlCatBuf, embd, 1e-6f, msgStride, msgSubOff, catStride, embd, false, B);
        _ctx.UavBarrier(_ctrlCatBuf);

        DispatchGemmBatchByType(cmd, GgufType.F32, lg.LoopCtrlDown[ctrl]!, _ctrlCatBuf, _ctrlGateBuf, 2 * embd, _w.GateRank, B);
        _ctx.UavBarrier(_ctrlGateBuf);

        DispatchGemmBatchByType(cmd, GgufType.F32, lg.LoopCtrlGain[ctrl]!, _ctrlGateBuf, _ctrlGainBuf, _w.GateRank, embd, B);
        DispatchGemmBatchByType(cmd, GgufType.F32, lg.LoopCtrlForget[ctrl]!, _ctrlGateBuf, _ctrlForgetBuf, _w.GateRank, embd, B);
        _ctx.UavBarrier();

        cmd.SetComputeRootSignature(_sigLoopBlend);
        cmd.SetPipelineState(_psoLoopBlendBatch);
        cmd.SetComputeRootUnorderedAccessView(1, targetBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _msgBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _ctrlGainBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _ctrlForgetBuf.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)embd;
        c[1] = (uint)msgSubOff;
        c[2] = (uint)msgStride;
        c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((embd + 255) / 256), (uint)B, 1);
        _ctx.UavBarrier(targetBuf);
    }

    private void ExecuteLayerBatchGpu(ref ID3D12GraphicsCommandList cmd, int il, int ip, int startPos, int B, ref int blocksCount, bool hasMessages)
    {
        var l = _w.PhysicalLayers[ip];
        var lg = _layers[ip];
        bool isSwa = l.IsSwa;
        int embd = _embd;
        int nHead = l.HeadCount;
        int nHeadKv = l.HeadCountKv;
        int headDim = l.HeadDim;
        int qDim = nHead * headDim;
        int kvDim = nHeadKv * headDim;
        int nRel = _w.RelativeLength;
        int rDim = nRel * nHead;
        ulong bEmbdBytes = (ulong)(B * embd) * 4UL;
        int blockStride = B * embd;

        // Save inpSa = cur
        GpuCopy(cmd, _inpSaBuf, 0, _curBuf, 0, bEmbdBytes);

        // 1. Attention inputs via Batched Mix + LoopControl + RMSNorm
        ID3D12Resource kInBuf = _in1Buf;
        ID3D12Resource vInBuf = _in2Buf;

        if (isSwa && !hasMessages)
        {
            if (lg.AttnResQuery != null)
            {
                DispatchMixBatch(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, embd, 0, blockStride, B);
                _ctx.UavBarrier(_mixOutBuf);
                DispatchRmsNormBatch(cmd, lg.AttnNorm!, _mixOutBuf, _in0Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            }
            else
            {
                DispatchRmsNormBatch(cmd, lg.AttnNorm!, _curBuf, _in0Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            }
            _ctx.UavBarrier(_in0Buf);
            kInBuf = _in0Buf;
            vInBuf = _in0Buf;
        }
        else if (!isSwa && !hasMessages)
        {
            if (lg.AttnResQuery != null)
            {
                DispatchMixBatch(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 3, 0, 3 * embd, 0, blockStride, B);
                _ctx.UavBarrier(_mixOutBuf);
                DispatchRmsNormBatch(cmd, lg.AttnQInpNorm!, _mixOutBuf, _in0Buf, embd, _w.RmsNormEps, 3 * embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnKInpNorm!, _mixOutBuf, _in1Buf, embd, _w.RmsNormEps, 3 * embd, embd, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnVInpNorm!, _mixOutBuf, _in2Buf, embd, _w.RmsNormEps, 3 * embd, 2 * embd, embd, 0, true, B);
            }
            else
            {
                DispatchRmsNormBatch(cmd, lg.AttnQInpNorm!, _curBuf, _in0Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnKInpNorm!, _curBuf, _in1Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnVInpNorm!, _curBuf, _in2Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            }
            _ctx.UavBarrier();
        }
        else
        {
            if (lg.AttnResQuery != null)
            {
                int nQ = isSwa ? 1 : 3;
                int outStride = nQ * embd;
                DispatchMixBatch(cmd, lg.AttnResQuery, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, nQ, 0, outStride, 0, blockStride, B);
                _ctx.UavBarrier(_mixOutBuf);
                // Extract per-token slices into contiguous [B * embd] buffers in a single barrier wave
                if (isSwa)
                {
                    GpuCopy(cmd, _in0Buf, 0, _mixOutBuf, 0, bEmbdBytes, barrier: false);
                    GpuCopy(cmd, _in1Buf, 0, _mixOutBuf, 0, bEmbdBytes, barrier: false);
                    GpuCopy(cmd, _in2Buf, 0, _mixOutBuf, 0, bEmbdBytes, barrier: false);
                }
                else
                {
                    ulong eBytes = (ulong)embd * 4UL;
                    for (int t = 0; t < B; t++)
                    {
                        ulong dstOff = (ulong)t * eBytes;
                        ulong srcBase = (ulong)(t * 3) * eBytes;
                        GpuCopy(cmd, _in0Buf, dstOff, _mixOutBuf, srcBase + 0UL, eBytes, barrier: false);
                        GpuCopy(cmd, _in1Buf, dstOff, _mixOutBuf, srcBase + eBytes, eBytes, barrier: false);
                        GpuCopy(cmd, _in2Buf, dstOff, _mixOutBuf, srcBase + 2UL * eBytes, eBytes, barrier: false);
                    }
                }
            }
            else
            {
                GpuCopy(cmd, _in0Buf, 0, _curBuf, 0, bEmbdBytes, barrier: false);
                GpuCopy(cmd, _in1Buf, 0, _curBuf, 0, bEmbdBytes, barrier: false);
                GpuCopy(cmd, _in2Buf, 0, _curBuf, 0, bEmbdBytes, barrier: false);
            }
            _ctx.UavBarrier();

            ApplyLoopControlBatchGpu(cmd, _in0Buf, 0 * embd, lg, 0, B);
            ApplyLoopControlBatchGpu(cmd, _in1Buf, 1 * embd, lg, 1, B);
            ApplyLoopControlBatchGpu(cmd, _in2Buf, 2 * embd, lg, 2, B);

            if (isSwa)
            {
                DispatchRmsNormBatch(cmd, lg.AttnNorm!, _in0Buf, _in0Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnNorm!, _in1Buf, _in1Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnNorm!, _in2Buf, _in2Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            }
            else
            {
                DispatchRmsNormBatch(cmd, lg.AttnQInpNorm!, _in0Buf, _in0Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnKInpNorm!, _in1Buf, _in1Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
                DispatchRmsNormBatch(cmd, lg.AttnVInpNorm!, _in2Buf, _in2Buf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
            }
            _ctx.UavBarrier();
        }

        // 2. TRUE BATCHED GEMM Attention Projections (reads every weight ONCE for all B tokens!)
        DispatchGemmBatchByType(cmd, l.WqType, lg.Wq, _in0Buf, _qBuf, embd, qDim, B);
        DispatchGemmBatchByType(cmd, l.WkType, lg.Wk, kInBuf, _kBuf, embd, kvDim, B);
        DispatchGemmBatchByType(cmd, l.WvType, lg.Wv, vInBuf, _vBuf, embd, kvDim, B);
        DispatchGemmBatchByType(cmd, l.AttnRelType, lg.AttnRel, _in0Buf, _rBuf, embd, rDim, B);
        DispatchGemmBatchByType(cmd, l.AttnGateType, lg.AttnGate, _in0Buf, _gateBuf, embd, qDim, B);
        DispatchGemmBatchByType(cmd, GgufType.F32, lg.AttnSinkProj, _in0Buf, _sinksBuf, embd, nHead, B);

        bool hasDyn = !isSwa && lg.AttnKConvProj != null && lg.AttnVConvProj != null;
        int dynDim = _w.ShortConvLCache * nHeadKv;
        if (hasDyn)
        {
            DispatchGemmBatchByType(cmd, GgufType.F32, lg.AttnKConvProj!, kInBuf, _kDynBuf, embd, dynDim, B);
            DispatchGemmBatchByType(cmd, GgufType.F32, lg.AttnVConvProj!, vInBuf, _vDynBuf, embd, dynDim, B);
        }
        _ctx.UavBarrier();

        // Register-resident causal Short-Conv on K & V across all B tokens in 1 dispatch!
        DispatchShortConvBatch(cmd, lg.AttnKConv, _kBuf, _kConvHistBuf, hasDyn ? _kDynBuf : _dummyUav, kvDim, _kvHistBaseFloats[il], hasDyn, headDim, B, dynDim);
        DispatchShortConvBatch(cmd, lg.AttnVConv, _vBuf, _vConvHistBuf, hasDyn ? _vDynBuf : _dummyUav, kvDim, _kvHistBaseFloats[il], hasDyn, headDim, B, dynDim);
        _ctx.UavBarrier();

        // Batched Per-Head L2 Clip + V RMSNorm + tau(pos+t) + FP16 KV Cache Write across all B tokens!
        int cap = _kvCap[il];
        DispatchHeadNormClipKvBatch(cmd, lg.AttnQClip, _qBuf, _dummyUav, headDim, nHead, mode: 0, _w.QkClip, startPos, cap, _kvLayerBaseU32[il], writeKv: false, applyTau: !isSwa, B);
        DispatchHeadNormClipKvBatch(cmd, null, _kBuf, _kCacheFp16, headDim, nHeadKv, mode: 0, _w.QkClip, startPos, cap, _kvLayerBaseU32[il], writeKv: true, applyTau: false, B);
        DispatchHeadNormClipKvBatch(cmd, null, _vBuf, _vCacheFp16, headDim, nHeadKv, mode: 1, _w.RmsNormEps, startPos, cap, _kvLayerBaseU32[il], writeKv: true, applyTau: false, B);
        DispatchHeadNormClipKvBatch(cmd, lg.AttnRelClip, _rBuf, _dummyUav, nRel, nHead, mode: 0, _w.RelativeClip, startPos, cap, _kvLayerBaseU32[il], writeKv: false, applyTau: !isSwa, B);
        _ctx.UavBarrier();

        // 3. Batched Causal FlashAttention across B * nHead heads in parallel!
        cmd.SetComputeRootSignature(_sigAttnBatch);
        cmd.SetPipelineState(_psoAttnBatch);
        cmd.SetComputeRootShaderResourceView(1, lg.AttnRelBias.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, _qBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, _rBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _sinksBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, _gateBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(6, _kCacheFp16.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(7, _vCacheFp16.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(8, _attnOutBuf.GPUVirtualAddress);

        uint* ac = stackalloc uint[10];
        ac[0] = (uint)startPos;
        ac[1] = (uint)_w.SlidingWindow;
        ac[2] = isSwa ? 1u : 0u;
        ac[3] = (uint)cap;
        ac[4] = (uint)(kvDim / 2);
        ac[5] = (uint)(nHead / nHeadKv);
        ac[6] = (uint)l.RelExtent;
        ac[7] = (uint)nRel;
        ac[8] = _kvLayerBaseU32[il];
        ac[9] = (uint)nHead;
        cmd.SetComputeRoot32BitConstants(0, 10, (IntPtr)ac, 0);
        cmd.Dispatch((uint)nHead, (uint)B, 1);
        _ctx.UavBarrier(_attnOutBuf);

        // Batched Wo GEMM (reads Wo ONCE for all B tokens!) + causal AttnPostConv
        DispatchGemmBatchByType(cmd, l.WoType, lg.Wo, _attnOutBuf, _attnProjBuf, qDim, embd, B);
        _ctx.UavBarrier(_attnProjBuf);

        DispatchShortConvBatch(cmd, lg.AttnPostConv, _attnProjBuf, _attnPostHistBuf, _dummyUav, embd, _embdHistBaseFloats[il], false, 0, B, 0);
        _ctx.UavBarrier(_attnProjBuf);

        // Block residual stream boundary
        if (ip % _w.AttnResBlockSize == 0)
        {
            GpuCopy(cmd, _blocksBuf, (ulong)blocksCount * bEmbdBytes, _inpSaBuf, 0, bEmbdBytes, barrier: false);
            blocksCount++;
            GpuCopy(cmd, _curBuf, 0, _attnProjBuf, 0, bEmbdBytes, barrier: false);
            _ctx.UavBarrier();
        }
        else
        {
            DispatchVecAdd(cmd, _inpSaBuf, _attnProjBuf, _curBuf, B * embd);
            _ctx.UavBarrier(_curBuf);
        }

        // Save ffnInp = cur
        GpuCopy(cmd, _ffnInpBuf, 0, _curBuf, 0, bEmbdBytes);

        // 4. MoE FFN: Batched Mix -> LoopCtrl -> FfnNorm -> Batched Router -> Batched TopK -> MoE -> FfnPostConv -> Residual Add
        DispatchMixBatch(cmd, lg.FfnResQuery!, _blocksBuf, _curBuf, _mixOutBuf, blocksCount, 1, 0, embd, 0, blockStride, B);
        _ctx.UavBarrier(_mixOutBuf);

        if (hasMessages)
        {
            GpuCopy(cmd, _curBuf, 0, _mixOutBuf, 0, bEmbdBytes);
            ApplyLoopControlBatchGpu(cmd, _curBuf, 3 * embd, lg, 3, B);
            DispatchRmsNormBatch(cmd, lg.FfnNorm!, _curBuf, _curBuf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
        }
        else
        {
            DispatchRmsNormBatch(cmd, lg.FfnNorm!, _mixOutBuf, _curBuf, embd, _w.RmsNormEps, embd, 0, embd, 0, true, B);
        }
        _ctx.UavBarrier(_curBuf);

        // Batched Router GEMM + Batched Top-8 Selection on GPU
        DispatchGemmBatchByType(cmd, GgufType.F32, lg.FfnGateInp, _curBuf, _routerLogitsBuf, embd, _w.ExpertCount, B);
        _ctx.UavBarrier(_routerLogitsBuf);

        cmd.SetComputeRootSignature(_sigRouterBatch);
        cmd.SetPipelineState(_psoRouterBatch);
        cmd.SetComputeRootShaderResourceView(1, (lg.ExpProbsBias ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(2, (lg.GuScales ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootShaderResourceView(3, (lg.DnScales ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _routerLogitsBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, _topkIdsBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(6, _metaBuf.GPUVirtualAddress);

        uint* rc = stackalloc uint[8];
        rc[0] = (uint)_w.ExpertCount;
        rc[1] = (uint)_topK;
        rc[2] = BitConverter.SingleToUInt32Bits(_w.ExpertWeightsScale);
        rc[3] = lg.ExpProbsBias != null ? 1u : 0u;
        rc[4] = lg.GuScales != null ? 1u : 0u;
        rc[5] = lg.DnScales != null ? 1u : 0u;
        rc[6] = (uint)B;
        rc[7] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 8, (IntPtr)rc, 0);
        cmd.Dispatch(1, (uint)B, 1);
        _ctx.UavBarrier();

        int totalPairs = B * _topK;
        if (lg.MoePinned)
        {
            // 100% GPU-RESIDENT BATCHED 2D INDIRECT NVFP4 MoE (ALL B*8 pairs in ONE GateUp + ONE Down dispatch!)
            cmd.SetComputeRootSignature(_sigNvfp4Ind);
            cmd.SetPipelineState(_psoNvfp4IndBatch);
            cmd.SetComputeRootShaderResourceView(1, lg.GateUpExps!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _curBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _guBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _topkIdsBuf.GPUVirtualAddress);

            uint guGroups = (uint)((2 * _ffExp + 15) / 16);
            uint* mc = stackalloc uint[6];
            mc[0] = (uint)embd;
            mc[1] = (uint)(2 * _ffExp);
            mc[2] = (uint)embd;
            mc[3] = (uint)(2 * _ffExp);
            mc[4] = lg.GateUpExpStrideBytes;
            mc[5] = (uint)_topK; // x_pair_div = 8
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)mc, 0);
            cmd.Dispatch(guGroups, (uint)totalPairs, 1);
            _ctx.UavBarrier(_guBuf);

            // SiTU across all B*8 pairs in 1 dispatch
            cmd.SetComputeRootSignature(_sigReduce);
            cmd.SetPipelineState(_psoSituBatch);
            cmd.SetComputeRootUnorderedAccessView(1, _guBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
            uint* uc = stackalloc uint[4];
            uc[0] = (uint)_ffExp;
            uc[1] = (uint)totalPairs;
            uc[2] = BitConverter.SingleToUInt32Bits(_w.SituBeta);
            uc[3] = BitConverter.SingleToUInt32Bits(_w.SituLinearBeta);
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)uc, 0);
            cmd.Dispatch((uint)((totalPairs * _ffExp + 255) / 256), 1, 1);
            _ctx.UavBarrier(_hidBuf);

            // Down NVFP4 across all B*8 pairs in ONE 2D dispatch
            cmd.SetComputeRootSignature(_sigNvfp4Ind);
            cmd.SetPipelineState(_psoNvfp4IndBatch);
            cmd.SetComputeRootShaderResourceView(1, lg.DownExps!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _hidBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _dnBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, _topkIdsBuf.GPUVirtualAddress);

            uint dnGroups = (uint)((embd + 15) / 16);
            mc[0] = (uint)_ffExp;
            mc[1] = (uint)embd;
            mc[2] = (uint)_ffExp;
            mc[3] = (uint)embd;
            mc[4] = lg.DownExpStrideBytes;
            mc[5] = 1u; // x_pair_div = 1
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)mc, 0);
            cmd.Dispatch(dnGroups, (uint)totalPairs, 1);
            _ctx.UavBarrier(_dnBuf);

            // WeightedSum8 across all B tokens in 1 dispatch
            cmd.SetComputeRootSignature(_sigReduce);
            cmd.SetPipelineState(_psoWeightedSum);
            cmd.SetComputeRootUnorderedAccessView(1, _dnBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _moeOutBuf.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _metaBuf.GPUVirtualAddress);
            uc[0] = (uint)embd;
            uc[1] = (uint)_topK;
            uc[2] = (uint)totalPairs; // weights_offset in meta
            uc[3] = (uint)B;
            cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)uc, 0);
            cmd.Dispatch((uint)((embd + 255) / 256), (uint)B, 1);
            _ctx.UavBarrier(_moeOutBuf);
        }
        else
        {
            // ONE sync for ALL B=16 tokens + Zero-Alloc Parallel AVX-512 4-Row Unrolled CPU MoE!
            ulong topkBytes = (ulong)totalPairs * 4UL;
            ulong metaBytes = (ulong)(2 * totalPairs) * 4UL;

            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.ResourceBarrierTransition(_topkIdsBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, 0, _curBuf, 0, bEmbdBytes);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, bEmbdBytes, _topkIdsBuf, 0, topkBytes);
            cmd.CopyBufferRegion(_cpuMoeReadbackBuf, bEmbdBytes + topkBytes, _metaBuf, 0, metaBytes);
            cmd.ResourceBarrierTransition(_curBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(_topkIdsBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);
            cmd.ResourceBarrierTransition(_metaBuf, ResourceStates.CopySource, ResourceStates.UnorderedAccess);

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            _cpuBatchCur ??= new float[MaxPrefillBatch * embd];
            _cpuBatchMoeOut ??= new float[MaxPrefillBatch * embd];
            _cpuBatchPairDn ??= new float[MaxPrefillBatch * _topK * embd];

            fixed (float* pCur = _cpuBatchCur)
                Buffer.MemoryCopy(_pCpuMoeReadback, pCur, bEmbdBytes, bEmbdBytes);

            uint* pIds = (uint*)(_pCpuMoeReadback + bEmbdBytes);
            float* pMeta = (float*)(_pCpuMoeReadback + bEmbdBytes + topkBytes);

            ExecuteCpuMoeBatchFallback(l, _cpuBatchCur, _cpuBatchMoeOut, _cpuBatchPairDn, B, pIds, pMeta, totalPairs);

            fixed (float* pOut = _cpuBatchMoeOut)
                Buffer.MemoryCopy(pOut, _pUpload, bEmbdBytes, bEmbdBytes);

            _ctx.BeginCommands();
            cmd = _ctx.CommandList;
            cmd.ResourceBarrierTransition(_moeOutBuf, ResourceStates.UnorderedAccess, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_moeOutBuf, 0, _uploadBuf, 0, bEmbdBytes);
            cmd.ResourceBarrierTransition(_moeOutBuf, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);
        }

        // Register-resident causal FfnPostConv + Residual Add
        DispatchShortConvBatch(cmd, lg.FfnPostConv, _moeOutBuf, _ffnPostHistBuf, _dummyUav, embd, _embdHistBaseFloats[il], false, 0, B, 0);
        _ctx.UavBarrier(_moeOutBuf);

        DispatchVecAdd(cmd, _ffnInpBuf, _moeOutBuf, _curBuf, B * embd);
        _ctx.UavBarrier(_curBuf);
    }

    private void ExecuteCpuMoeBatchFallback(
        Inkling2LayerWeights layer,
        float[] curBatch,
        float[] moeOutBatch,
        float[] pairDnScratch,
        int B,
        uint* pIds,
        float* pMeta,
        int totalPairs)
    {
        int embd = _embd;
        int ffExp = _ffExp;
        int topK = _topK;
        int nExp = _w.ExpertCount;
        long guRowBytes = GgufTypes.GetRowBytes(layer.FfnGateUpExpsType, embd);
        long dnRowBytes = GgufTypes.GetRowBytes(layer.FfnDownExpsType, ffExp);
        long guExpBytes = (long)(2 * ffExp) * guRowBytes;
        long dnExpBytes = (long)embd * dnRowBytes;
        float beta = _w.SituBeta;
        float invBeta = 1.0f / beta;
        float lb = _w.SituLinearBeta;
        float invLb = lb > 0.0f ? 1.0f / lb : 0.0f;

        var buckets = new List<(int Token, int PairIdx, float GuScale, float CombinedWeight)>[nExp];
        var activeExperts = new List<int>(totalPairs);
        for (int t = 0; t < B; t++)
        {
            for (int k = 0; k < topK; k++)
            {
                int pairIdx = t * topK + k;
                int expId = (int)pIds[pairIdx];
                float guScale = pMeta[pairIdx];
                float combW = pMeta[totalPairs + pairIdx]; // already weight * dnScale
                var b = buckets[expId];
                if (b == null)
                {
                    b = new List<(int, int, float, float)>(2);
                    buckets[expId] = b;
                    activeExperts.Add(expId);
                }
                b.Add((t, pairIdx, guScale, combW));
            }
        }

        var guType = layer.FfnGateUpExpsType;
        var dnType = layer.FfnDownExpsType;
        bool useVnniGu = _useVnniMoe && guType == GgufType.NVFP4;
        bool useVnniDn = _useVnniMoe && dnType == GgufType.NVFP4;
        int embdBlocks = embd / 64;
        if (useVnniGu)
        {
            _cpuBatchQ8Cur ??= new BlockQ8NVFP4[MaxPrefillBatch * embdBlocks];
        }
        nint guBase = (nint)layer.FfnGateUpExps;
        nint dnBase = (nint)layer.FfnDownExps;

        fixed (float* pCurBatch = curBatch, pPairDn = pairDnScratch)
        fixed (BlockQ8NVFP4* pQ8CurBatch = _cpuBatchQ8Cur)
        {
            if (useVnniGu && pQ8CurBatch != null)
            {
                for (int t = 0; t < B; t++)
                {
                    QuantKernels.QuantizeXForNVFP4(pCurBatch + t * embd, pQ8CurBatch + t * embdBlocks, embd);
                }
            }

            nint curAddr = (nint)pCurBatch;
            nint q8CurAddr = (nint)pQ8CurBatch;
            nint pairDnAddr = (nint)pPairDn;

            Parallel.For(0, activeExperts.Count, aeIdx =>
            {
                int expId = activeExperts[aeIdx];
                var bucket = buckets[expId]!;
                int m = bucket.Count;
                float* xBase = (float*)curAddr;
                BlockQ8NVFP4* q8Base = (BlockQ8NVFP4*)q8CurAddr;
                float* outBase = (float*)pairDnAddr;
                byte* guExpPtr = (byte*)guBase + (long)expId * guExpBytes;
                byte* dnExpPtr = (byte*)dnBase + (long)expId * dnExpBytes;

                int q8HidBlocks = ffExp / 64;
                BlockQ8NVFP4[]? rentedQ8Hid = q8HidBlocks > 64 ? ArrayPool<BlockQ8NVFP4>.Shared.Rent(q8HidBlocks) : null;
                Span<BlockQ8NVFP4> q8HidSpan = q8HidBlocks <= 64 ? stackalloc BlockQ8NVFP4[q8HidBlocks] : rentedQ8Hid.AsSpan(0, q8HidBlocks);

                float[] expHid = ArrayPool<float>.Shared.Rent(ffExp);
                try
                {
                    fixed (float* pHid = expHid)
                    fixed (BlockQ8NVFP4* pQ8Hid = q8HidSpan)
                    {
                        for (int bi = 0; bi < m; bi++)
                        {
                            var item = bucket[bi];
                            float* xPtr = xBase + item.Token * embd;
                            BlockQ8NVFP4* q8XPtr = q8Base + item.Token * embdBlocks;
                            float gs = item.GuScale;
                            byte* gRow = guExpPtr;
                            byte* uRow = guExpPtr + (long)ffExp * guRowBytes;
                            int j = 0;

                            if (guType == GgufType.NVFP4)
                            {
                                if (useVnniGu && q8Base != null)
                                {
                                    for (; j + 1 < ffExp; j += 2)
                                    {
                                        QuantKernels.VecDotNVFP4x4Q8(
                                            (BlockNVFP4*)gRow,
                                            (BlockNVFP4*)(gRow + guRowBytes),
                                            (BlockNVFP4*)uRow,
                                            (BlockNVFP4*)(uRow + guRowBytes),
                                            q8XPtr,
                                            embd,
                                            out float g0,
                                            out float g1,
                                            out float u0,
                                            out float u1);
                                        g0 *= gs; g1 *= gs;
                                        u0 *= gs; u1 *= gs;
                                        float act0 = beta * MathF.Tanh(g0 * invBeta) / (1.0f + MathF.Exp(-g0));
                                        float act1 = beta * MathF.Tanh(g1 * invBeta) / (1.0f + MathF.Exp(-g1));
                                        if (lb > 0.0f)
                                        {
                                            u0 = lb * MathF.Tanh(u0 * invLb);
                                            u1 = lb * MathF.Tanh(u1 * invLb);
                                        }
                                        pHid[j] = act0 * u0;
                                        pHid[j + 1] = act1 * u1;
                                        gRow += 2 * guRowBytes;
                                        uRow += 2 * guRowBytes;
                                    }
                                    for (; j < ffExp; j++)
                                    {
                                        float g = QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)gRow, q8XPtr, embd) * gs;
                                        float u = QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)uRow, q8XPtr, embd) * gs;
                                        float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                                        if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                                        pHid[j] = act * u;
                                        gRow += guRowBytes;
                                        uRow += guRowBytes;
                                    }
                                }
                                else
                                {
                                    for (; j + 1 < ffExp; j += 2)
                                    {
                                        QuantKernels.VecDotNVFP4x4(
                                            (BlockNVFP4*)gRow,
                                            (BlockNVFP4*)(gRow + guRowBytes),
                                            (BlockNVFP4*)uRow,
                                            (BlockNVFP4*)(uRow + guRowBytes),
                                            xPtr,
                                            embd,
                                            out float g0,
                                            out float g1,
                                            out float u0,
                                            out float u1);
                                        g0 *= gs; g1 *= gs;
                                        u0 *= gs; u1 *= gs;
                                        float act0 = beta * MathF.Tanh(g0 * invBeta) / (1.0f + MathF.Exp(-g0));
                                        float act1 = beta * MathF.Tanh(g1 * invBeta) / (1.0f + MathF.Exp(-g1));
                                        if (lb > 0.0f)
                                        {
                                            u0 = lb * MathF.Tanh(u0 * invLb);
                                            u1 = lb * MathF.Tanh(u1 * invLb);
                                        }
                                        pHid[j] = act0 * u0;
                                        pHid[j + 1] = act1 * u1;
                                        gRow += 2 * guRowBytes;
                                        uRow += 2 * guRowBytes;
                                    }
                                }
                            }

                            for (; j < ffExp; j++)
                            {
                                float g = QuantKernels.ComputeDot(guType, gRow, xPtr, null, embd) * gs;
                                float u = QuantKernels.ComputeDot(guType, uRow, xPtr, null, embd) * gs;
                                float act = beta * MathF.Tanh(g * invBeta) / (1.0f + MathF.Exp(-g));
                                if (lb > 0.0f) u = lb * MathF.Tanh(u * invLb);
                                pHid[j] = act * u;
                                gRow += guRowBytes;
                                uRow += guRowBytes;
                            }

                            float w = item.CombinedWeight;
                            float* pairOut = outBase + item.PairIdx * embd;
                            byte* dnRow = dnExpPtr;
                            int d = 0;

                            if (useVnniDn)
                            {
                                QuantKernels.QuantizeXForNVFP4(pHid, pQ8Hid, ffExp);
                                for (; d + 3 < embd; d += 4)
                                {
                                    QuantKernels.VecDotNVFP4x4Q8(
                                        (BlockNVFP4*)dnRow,
                                        (BlockNVFP4*)(dnRow + dnRowBytes),
                                        (BlockNVFP4*)(dnRow + 2 * dnRowBytes),
                                        (BlockNVFP4*)(dnRow + 3 * dnRowBytes),
                                        pQ8Hid,
                                        ffExp,
                                        out float r0,
                                        out float r1,
                                        out float r2,
                                        out float r3);
                                    pairOut[d + 0] = w * r0;
                                    pairOut[d + 1] = w * r1;
                                    pairOut[d + 2] = w * r2;
                                    pairOut[d + 3] = w * r3;
                                    dnRow += 4 * dnRowBytes;
                                }
                                for (; d < embd; d++)
                                {
                                    pairOut[d] = w * QuantKernels.VecDotNVFP4Q8((BlockNVFP4*)dnRow, pQ8Hid, ffExp);
                                    dnRow += dnRowBytes;
                                }
                            }
                            else
                            {
                                if (dnType == GgufType.NVFP4)
                                {
                                    for (; d + 3 < embd; d += 4)
                                    {
                                        QuantKernels.VecDotNVFP4x4(
                                            (BlockNVFP4*)dnRow,
                                            (BlockNVFP4*)(dnRow + dnRowBytes),
                                            (BlockNVFP4*)(dnRow + 2 * dnRowBytes),
                                            (BlockNVFP4*)(dnRow + 3 * dnRowBytes),
                                            pHid,
                                            ffExp,
                                            out float r0,
                                            out float r1,
                                            out float r2,
                                            out float r3);
                                        pairOut[d + 0] = w * r0;
                                        pairOut[d + 1] = w * r1;
                                        pairOut[d + 2] = w * r2;
                                        pairOut[d + 3] = w * r3;
                                        dnRow += 4 * dnRowBytes;
                                    }
                                }

                                for (; d < embd; d++)
                                {
                                    pairOut[d] = w * QuantKernels.ComputeDot(dnType, dnRow, pHid, null, ffExp);
                                    dnRow += dnRowBytes;
                                }
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(expHid);
                    if (rentedQ8Hid != null) ArrayPool<BlockQ8NVFP4>.Shared.Return(rentedQ8Hid);
                }
            });
        }

        Array.Clear(moeOutBatch, 0, B * embd);
        for (int t = 0; t < B; t++)
        {
            int dstOff = t * embd;
            for (int k = 0; k < topK; k++)
            {
                int srcOff = (t * topK + k) * embd;
                for (int d = 0; d < embd; d++)
                    moeOutBatch[dstOff + d] += pairDnScratch[srcOff + d];
            }
        }
    }

    // ---- Batch Dispatch Helpers ----

    private void DispatchGemmBatchByType(ID3D12GraphicsCommandList cmd, GgufType type, ID3D12Resource w, ID3D12Resource x, ID3D12Resource y, int kCols, int mRows, int B)
    {
        var (pso, rowsPerGroup) = type switch
        {
            GgufType.Q8_0 => (_psoQ8_0Batch, 8),
            GgufType.Q6_K => (_psoQ6KBatch, 4),
            GgufType.Q5_K => (_psoQ5KBatch, 4),
            GgufType.F32 => (_psoF32Batch, 8),
            _ => throw new NotSupportedException($"Inkling2D3D12Model Batch does not support quant type {type}.")
        };
        cmd.SetComputeRootSignature(_sigGemv);
        cmd.SetPipelineState(pso);
        cmd.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, y.GPUVirtualAddress);
        uint* c = stackalloc uint[4];
        c[0] = (uint)kCols; c[1] = (uint)mRows; c[2] = (uint)B; c[3] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)c, 0);
        cmd.Dispatch((uint)((mRows + rowsPerGroup - 1) / rowsPerGroup), 1, 1);
    }

    private void DispatchRmsNormBatch(ID3D12GraphicsCommandList cmd, ID3D12Resource w, ID3D12Resource src, ID3D12Resource dst, int dim, float eps, int inStride, int inSubOff, int outStride, int outSubOff, bool hasWeight, int B)
    {
        cmd.SetComputeRootSignature(_sigRmsNormBatch);
        cmd.SetPipelineState(_psoRmsNormBatch);
        cmd.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, src.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);
        uint* c = stackalloc uint[8];
        c[0] = (uint)dim;
        c[1] = BitConverter.SingleToUInt32Bits(eps);
        c[2] = (uint)inStride;
        c[3] = (uint)inSubOff;
        c[4] = (uint)outStride;
        c[5] = (uint)outSubOff;
        c[6] = hasWeight ? 1u : 0u;
        c[7] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 8, (IntPtr)c, 0);
        cmd.Dispatch(1, (uint)B, 1);
    }

    private void DispatchMixBatch(ID3D12GraphicsCommandList cmd, ID3D12Resource queries, ID3D12Resource blocksBuf, ID3D12Resource curBuf, ID3D12Resource outBuf, int nBlocks, int nQueries, int qBaseFloats, int outStrideFloats, int outSubOff, int blockStrideFloats, int B)
    {
        cmd.SetComputeRootSignature(_sigMix);
        cmd.SetPipelineState(_psoMixBatch);
        cmd.SetComputeRootShaderResourceView(1, queries.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, blocksBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, curBuf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, outBuf.GPUVirtualAddress);
        uint* c = stackalloc uint[6];
        c[0] = (uint)_embd;
        c[1] = (uint)nBlocks;
        c[2] = (uint)qBaseFloats;
        c[3] = (uint)outStrideFloats;
        c[4] = (uint)outSubOff;
        c[5] = (uint)blockStrideFloats;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
        cmd.Dispatch((uint)nQueries, (uint)B, 1);
    }

    private void DispatchShortConvBatch(ID3D12GraphicsCommandList cmd, ID3D12Resource convW, ID3D12Resource cur, ID3D12Resource hist, ID3D12Resource dyn, int channels, uint histBaseFloats, bool hasDyn, int headDim, int B, int dynStride)
    {
        cmd.SetComputeRootSignature(_sigShortConvBatch);
        cmd.SetPipelineState(_psoShortConvBatch);
        cmd.SetComputeRootShaderResourceView(1, convW.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, cur.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, hist.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dyn.GPUVirtualAddress);
        uint* c = stackalloc uint[6];
        c[0] = (uint)channels;
        c[1] = histBaseFloats;
        c[2] = hasDyn ? 1u : 0u;
        c[3] = (uint)headDim;
        c[4] = (uint)B;
        c[5] = (uint)dynStride;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
        cmd.Dispatch((uint)((channels + 255) / 256), 1, 1);
    }

    private void DispatchHeadNormClipKvBatch(ID3D12GraphicsCommandList cmd, ID3D12Resource? scale, ID3D12Resource buf, ID3D12Resource kvFp16, int dim, int nHeads, uint mode, float threshOrEps, int startPos, int kvCap, uint kvLayerBaseU32, bool writeKv, bool applyTau, int B)
    {
        cmd.SetComputeRootSignature(_sigHeadNormBatch);
        cmd.SetPipelineState(_psoHeadNormBatch);
        cmd.SetComputeRootShaderResourceView(1, (scale ?? _dummySrv).GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, buf.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, kvFp16.GPUVirtualAddress);
        uint* c = stackalloc uint[12];
        c[0] = (uint)dim;
        c[1] = mode;
        c[2] = scale != null ? 1u : 0u;
        c[3] = BitConverter.SingleToUInt32Bits(threshOrEps);
        c[4] = (uint)startPos;
        c[5] = (uint)kvCap;
        c[6] = kvLayerBaseU32;
        c[7] = writeKv ? 1u : 0u;
        c[8] = applyTau ? 1u : 0u;
        c[9] = BitConverter.SingleToUInt32Bits(_w.LogScaleAlpha);
        c[10] = BitConverter.SingleToUInt32Bits(_w.LogScaleFloor);
        c[11] = (uint)nHeads;
        cmd.SetComputeRoot32BitConstants(0, 12, (IntPtr)c, 0);
        cmd.Dispatch((uint)nHeads, (uint)B, 1);
    }
}
#endif
