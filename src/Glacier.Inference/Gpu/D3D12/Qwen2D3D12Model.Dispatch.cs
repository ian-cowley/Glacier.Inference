namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

public sealed unsafe partial class Qwen2D3D12Model
{
    private void DispatchRmsNorm(ID3D12GraphicsCommandList cmdList, ID3D12Resource x, ID3D12Resource weight, ID3D12Resource dst, int size, float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNorm);
        cmdList.SetPipelineState(_psoRmsNorm);

        uint* pConsts = stackalloc uint[2];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;
        cmdList.SetComputeRoot32BitConstants(0, 2, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(1, 1, 1);
    }

    private void DispatchGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ulong wGpuVirtualAddress,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
    {
        cmdList.SetComputeRootSignature(_sigGemv);
        var pso = type switch
        {
            GgufType.Q4_K => _psoGemvQ4K,
            GgufType.Q5_K => _psoGemvQ5K,
            GgufType.Q6_K => _psoGemvQ6K,
            GgufType.Q3_K => _psoGemvQ3K,
            GgufType.Q8_0 => _psoGemvQ8_0,
            GgufType.F32 => _psoGemvFp32,
            _ => throw new NotSupportedException($"Direct3D 12 GEMV does not support tensor quantization type {type}.")
        };
        cmdList.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)k_cols;
        pConsts[1] = (uint)m_rows;
        pConsts[2] = bias != null ? 1u : 0u;
        pConsts[3] = residual != null ? 1u : 0u;
        pConsts[4] = y != null ? 1u : 0u;
        cmdList.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, wGpuVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y?.GPUVirtualAddress ?? _ctx.DummyBuffer.GPUVirtualAddress);

        cmdList.Dispatch(((uint)m_rows + 3) / 4, 1, 1);
    }

    private void DispatchGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ID3D12Resource W,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
        => DispatchGemv(cmdList, type, y, x, W.GPUVirtualAddress, k_cols, m_rows, bias, residual);

    private void DispatchVecAddWeighted(ID3D12GraphicsCommandList cmdList, ID3D12Resource b, ID3D12Resource a, float weight, int size, uint accumulate = 1)
    {
        cmdList.SetComputeRootSignature(_sigVecAddWeighted);
        cmdList.SetPipelineState(_psoVecAddWeighted);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = weight;
        pConsts[2] = accumulate;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, b.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, a.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchVecAdd(ID3D12GraphicsCommandList cmdList, ID3D12Resource b, ID3D12Resource a, int size)
    {
        cmdList.SetComputeRootSignature(_sigVecAdd);
        cmdList.SetPipelineState(_psoVecAdd);

        uint* pConsts = stackalloc uint[1];
        pConsts[0] = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, b.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, a.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchRmsNormHeads(ID3D12GraphicsCommandList cmdList, ID3D12Resource x, ID3D12Resource weight, int nHeads, int headDim, float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNormHeads);
        cmdList.SetPipelineState(_psoRmsNormHeads);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)headDim;
        pConsts[1] = (uint)nHeads;
        *(float*)(&pConsts[2]) = eps;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);

        cmdList.Dispatch((uint)nHeads, 1, 1);
    }

    private void DispatchRope(ID3D12GraphicsCommandList cmdList, ID3D12Resource q, ID3D12Resource k, int pos)
    {
        cmdList.SetComputeRootSignature(_sigRope);
        cmdList.SetPipelineState(_psoRope);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)pos;
        *(float*)(&pConsts[4]) = _weights.RopeFreqBase;
        *(float*)(&pConsts[5]) = 1.0f;
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, q.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, k.GPUVirtualAddress);

        uint totalHalf = (uint)((_nHeads + _nHeadsKv) * (_headDim / 2));
        cmdList.Dispatch((totalHalf + 255) / 256, 1, 1);
    }

    private void DispatchKvStore(ID3D12GraphicsCommandList cmdList, int layerIdx, int pos)
    {
        cmdList.SetComputeRootSignature(_sigKvStore);
        cmdList.SetPipelineState(_psoKvStore);

        uint* pConsts = stackalloc uint[4];
        pConsts[0] = (uint)_nHeadsKv;
        pConsts[1] = (uint)_headDim;
        pConsts[2] = (uint)_maxSeqLen;
        pConsts[3] = (uint)pos;
        cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dK.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dV.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dValCache[layerIdx].GPUVirtualAddress);

        uint total = (uint)(_nHeadsKv * _headDim);
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchAttention(ID3D12GraphicsCommandList cmdList, int layerIdx, int pos)
    {
        cmdList.SetComputeRootSignature(_sigAttention);
        cmdList.SetPipelineState(_psoAttention);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)_maxSeqLen;
        pConsts[4] = (uint)pos;
        *(float*)(&pConsts[5]) = _attnScale;
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dQ.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dValCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dAttnOut.GPUVirtualAddress);

        cmdList.Dispatch((uint)_nHeads, 1, 1);
    }

    private void DispatchSwiglu(ID3D12GraphicsCommandList cmdList, ID3D12Resource gate, ID3D12Resource up, ID3D12Resource dst, int size)
    {
        cmdList.SetComputeRootSignature(_sigSwiglu);
        cmdList.SetPipelineState(_psoSwiglu);

        uint* pConsts = stackalloc uint[1];
        pConsts[0] = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, gate.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, up.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch(((uint)size + 255) / 256, 1, 1);
    }

    private void DispatchGemmBatch(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ID3D12Resource w,
        int kCols,
        int mRows,
        int batchSize,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
    {
        cmdList.SetComputeRootSignature(_sigGemmBatch);
        if (type == GgufType.Q4_K)
            cmdList.SetPipelineState(_psoGemmQ4KBatch);
        else if (type == GgufType.Q6_K)
            cmdList.SetPipelineState(_psoGemmQ6KBatch);
        else if (type == GgufType.Q8_0)
            cmdList.SetPipelineState(_psoGemmQ8_0Batch);
        else if (type == GgufType.F32)
            cmdList.SetPipelineState(_psoGemmFp32Batch);
        else
            throw new NotSupportedException($"Direct3D 12 batch GEMM does not support tensor quantization type {type}.");

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = (uint)batchSize;
        pConsts[3] = (uint)(bias != null ? 1 : 0);
        pConsts[4] = (uint)(residual != null ? 1 : 0);
        pConsts[5] = (uint)(y != null ? 1 : 0);
        cmdList.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, w.GPUVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias != null ? bias.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual != null ? residual.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y != null ? y.GPUVirtualAddress : _ctx.DummyBuffer.GPUVirtualAddress);


        cmdList.Dispatch((uint)(mRows + 3) / 4, (uint)(batchSize + 31) / 32, 1);
    }



    private void DispatchRmsNormBatch(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource x,
        ID3D12Resource weight,
        ID3D12Resource dst,
        int size,
        int batchSize,
        float eps)
    {
        cmdList.SetComputeRootSignature(_sigRmsNormBatch);
        cmdList.SetPipelineState(_psoRmsNormBatch);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)size;
        *(float*)(&pConsts[1]) = eps;
        pConsts[2] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootShaderResourceView(1, weight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dst.GPUVirtualAddress);

        cmdList.Dispatch((uint)batchSize, 1, 1);
    }

    private void DispatchRopeBatch(ID3D12GraphicsCommandList cmdList, ID3D12Resource q, ID3D12Resource k, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigRopeBatch);
        cmdList.SetPipelineState(_psoRopeBatch);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)startPos;
        *(float*)(&pConsts[4]) = _weights.RopeFreqBase;
        *(float*)(&pConsts[5]) = 1.0f;
        pConsts[6] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, q.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, k.GPUVirtualAddress);

        uint halfDim = (uint)(_headDim / 2);
        uint totalHalfPerToken = (uint)((_nHeads + _nHeadsKv) * halfDim);
        uint total = (uint)batchSize * totalHalfPerToken;
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchKvStoreBatch(ID3D12GraphicsCommandList cmdList, int layerIdx, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigKvStoreBatch);
        cmdList.SetPipelineState(_psoKvStoreBatch);

        uint* pConsts = stackalloc uint[5];
        pConsts[0] = (uint)_nHeadsKv;
        pConsts[1] = (uint)_headDim;
        pConsts[2] = (uint)_maxSeqLen;
        pConsts[3] = (uint)startPos;
        pConsts[4] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 5, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dKBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dVBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dValCache[layerIdx].GPUVirtualAddress);

        uint total = (uint)(batchSize * _nHeadsKv * _headDim);
        cmdList.Dispatch((total + 255) / 256, 1, 1);
    }

    private void DispatchAttentionBatch(ID3D12GraphicsCommandList cmdList, int layerIdx, int startPos, int batchSize)
    {
        cmdList.SetComputeRootSignature(_sigAttentionBatch);
        cmdList.SetPipelineState(_psoAttentionBatch);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)_nHeads;
        pConsts[1] = (uint)_nHeadsKv;
        pConsts[2] = (uint)_headDim;
        pConsts[3] = (uint)_maxSeqLen;
        pConsts[4] = (uint)startPos;
        *(float*)(&pConsts[5]) = _attnScale;
        pConsts[6] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, _dQBatch.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, _dKeyCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, _dValCache[layerIdx].GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, _dAttnOutBatch.GPUVirtualAddress);

        cmdList.Dispatch((uint)_nHeads, (uint)batchSize, 1);
    }
}
