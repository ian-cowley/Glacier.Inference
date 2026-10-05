namespace Glacier.Inference.Video.Wan;

using System;
using Glacier.Inference.Gpu;

public sealed unsafe partial class WanDiT
{
    private void ExecuteBlockGpu(
        WanBlockWeights blk,
        int totalImgTokens,
        int frames,
        int tokenH,
        int tokenW,
        float* ropeCos,
        float* ropeSin,
        float* projectedTxt,
        int numTxtTokens,
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

        // Upload block modulation vectors to GPU scratch
        _gpu!.CopyToDevice(_dShiftMsa, (IntPtr)shift_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dScaleMsa, (IntPtr)scale_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dGateMsa, (IntPtr)gate_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCShiftMsa, (IntPtr)c_shift_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCScaleMsa, (IntPtr)c_scale_msa, (nuint)(HiddenDim * sizeof(float)));
        _gpu.CopyToDevice(_dCGateMsa, (IntPtr)c_gate_msa, (nuint)(HiddenDim * sizeof(float)));

        // 1. Self-Attention
        // 1a. AdaLN LayerNorm on GPU: _dXDevice -> _dYDevice
        DispatchAdaLnDevice(_dXDevice, _dYDevice, _dShiftMsa, _dScaleMsa, totalImgTokens, HiddenDim);

        // 1b. QKV MatMuls
        DispatchMatMulBatchDevice(blk.D_SelfQ_W, blk.D_SelfQ_B, _dYDevice, _dQDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchDevice(blk.D_SelfK_W, blk.D_SelfK_B, _dYDevice, _dKDevice, HiddenDim, HiddenDim, totalImgTokens);
        DispatchMatMulBatchQ6KDevice(blk.D_SelfV_W, blk.D_SelfV_B, _dYDevice, _dVDevice, HiddenDim, HiddenDim, totalImgTokens);

        // 1c. Across-heads RMSNorm on Q and K
        DispatchRmsNormBatchDevice(_dQDevice, blk.D_SelfNormQ, _dQDevice, HiddenDim, totalImgTokens);
        DispatchRmsNormBatchDevice(_dKDevice, blk.D_SelfNormK, _dKDevice, HiddenDim, totalImgTokens);

        // 1d. 3D-RoPE rotation on Q and K directly in GPU VRAM (zero host PCIe round-trips)
        DispatchRoPE3DDevice(_dQDevice, _dRoPECos, _dRoPESin, totalImgTokens, NumHeads, HeadDim);
        DispatchRoPE3DDevice(_dKDevice, _dRoPECos, _dRoPESin, totalImgTokens, NumHeads, HeadDim);

        // 1e. Bidirectional Attention across spatio-temporal tokens
        float scale = 1.0f / MathF.Sqrt(HeadDim);
        DispatchAttentionDevice(_dQDevice, _dKDevice, _dVDevice, _dAttnOut, NumHeads, HeadDim, totalImgTokens, scale);

        // 1f. Self-Attention Output Projection into _dYDevice
        DispatchMatMulBatchDevice(blk.D_SelfO_W, blk.D_SelfO_B, _dAttnOut, _dYDevice, HiddenDim, HiddenDim, totalImgTokens);

        // 1g. Gated Residual: _dXDevice += _dYDevice * gate_msa
        DispatchResidualGatedDevice(_dXDevice, _dYDevice, _dGateMsa, totalImgTokens * HiddenDim, HiddenDim);

        // 2. Cross-Attention (if prompt text context provided)
        if (numTxtTokens > 0)
        {
            // 2a. Affine LayerNorm: _dXDevice -> _dYDevice using blk.D_Norm3_W and blk.D_Norm3_B
            DispatchLayerNormAffineDevice(_dXDevice, _dYDevice, blk.D_Norm3_W, blk.D_Norm3_B, totalImgTokens, HiddenDim);

            // 2b. Q projection from image tokens (_dYDevice) into _dQDevice [totalImgTokens, HiddenDim]
            DispatchMatMulBatchDevice(blk.D_CrossQ_W, blk.D_CrossQ_B, _dYDevice, _dQDevice, HiddenDim, HiddenDim, totalImgTokens);

            // 2c. K projection from text tokens (_dTxtDevice) into _dKDevice [numTxtTokens, HiddenDim]
            DispatchMatMulBatchDevice(blk.D_CrossK_W, blk.D_CrossK_B, _dTxtDevice, _dKDevice, HiddenDim, HiddenDim, numTxtTokens);

            // 2d. V projection from text tokens (_dTxtDevice) into _dVDevice [numTxtTokens, HiddenDim] (Q6_K)
            DispatchMatMulBatchQ6KDevice(blk.D_CrossV_W, blk.D_CrossV_B, _dTxtDevice, _dVDevice, HiddenDim, HiddenDim, numTxtTokens);

            // 2e. Across-heads RMSNorm on Q and K
            DispatchRmsNormBatchDevice(_dQDevice, blk.D_CrossNormQ, _dQDevice, HiddenDim, totalImgTokens);
            DispatchRmsNormBatchDevice(_dKDevice, blk.D_CrossNormK, _dKDevice, HiddenDim, numTxtTokens);

            // 2f. FlashAttention-2 Cross-Attention: Q [totalImgTokens], K [numTxtTokens], V [numTxtTokens] -> _dAttnOut [totalImgTokens]
            DispatchAttentionCrossDevice(_dQDevice, _dKDevice, _dVDevice, _dAttnOut, NumHeads, HeadDim, totalImgTokens, numTxtTokens, scale);

            // 2g. Output projection into _dYDevice
            DispatchMatMulBatchDevice(blk.D_CrossO_W, blk.D_CrossO_B, _dAttnOut, _dYDevice, HiddenDim, HiddenDim, totalImgTokens);

            // 2h. Residual addition: _dXDevice += _dYDevice
            DispatchVecAddBatchDevice(_dXDevice, _dYDevice, totalImgTokens * HiddenDim);
        }

        // 3. Feed-Forward Network: 1536 -> 8960 -> GELU -> 1536 + Gated Residual
        // 3a. AdaLN LayerNorm: _dXDevice -> _dYDevice using c_shift_msa and c_scale_msa
        DispatchAdaLnDevice(_dXDevice, _dYDevice, _dCShiftMsa, _dCScaleMsa, totalImgTokens, HiddenDim);

        // 3b. FFN0
        DispatchMatMulBatchDevice(blk.D_Ffn0_W, blk.D_Ffn0_B, _dYDevice, _dFfnInter, HiddenDim, FfnDim, totalImgTokens);
        DispatchGeluDevice(_dFfnInter, totalImgTokens * FfnDim);

        // 3c. FFN2 into _dYDevice
        DispatchMatMulBatchQ6KDevice(blk.D_Ffn2_W, blk.D_Ffn2_B, _dFfnInter, _dYDevice, FfnDim, HiddenDim, totalImgTokens);

        // 3d. Gated Residual: _dXDevice += _dYDevice * c_gate_msa
        DispatchResidualGatedDevice(_dXDevice, _dYDevice, _dCGateMsa, totalImgTokens * HiddenDim, HiddenDim);
    }

    private void DispatchAdaLnDevice(IntPtr dSrc, IntPtr dDst, IntPtr dShift, IntPtr dScale, int numTokens, int dim)
    {
        uint blockSize = 128;
        uint gridX = (uint)numTokens;

        int localTokens = numTokens;
        int localDim = dim;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dShift;
        pArgs[3] = &dScale;
        pArgs[4] = &localTokens;
        pArgs[5] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAdaLn,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_adaln_kernel)");
    }

    private void DispatchLayerNormAffineDevice(IntPtr dSrc, IntPtr dDst, IntPtr dWeight, IntPtr dBias, int numTokens, int dim)
    {
        uint blockSize = 128;
        uint gridX = (uint)numTokens;

        int localTokens = numTokens;
        int localDim = dim;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dWeight;
        pArgs[3] = &dBias;
        pArgs[4] = &localTokens;
        pArgs[5] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnLayerNormAffine,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(layer_norm_affine_kernel)");
    }

    private void DispatchRmsNormBatchDevice(IntPtr dX, IntPtr dWeight, IntPtr dDst, int size, int numTokens, float eps = 1e-6f)
    {
        uint blockSize = 256;
        uint gridX = (uint)numTokens;

        int localSize = size;
        float localEps = eps;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dX;
        pArgs[1] = &dWeight;
        pArgs[2] = &dDst;
        pArgs[3] = &localSize;
        pArgs[4] = &localEps;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRmsNormBatch,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rms_norm_batch)");
    }

    private void DispatchResidualGatedDevice(IntPtr dTarget, IntPtr dUpdate, IntPtr dGate, int totalElements, int dim)
    {
        uint blockSize = 256;
        uint gridX = (uint)((totalElements + (int)blockSize - 1) / (int)blockSize);

        int localCount = totalElements;
        int localDim = dim;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dTarget;
        pArgs[1] = &dUpdate;
        pArgs[2] = &dGate;
        pArgs[3] = &localCount;
        pArgs[4] = &localDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnResidualGated,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_residual_gated)");
    }

    private void DispatchVecAddBatchDevice(IntPtr dTarget, IntPtr dSource, int count)
    {
        uint blockSize = 256;
        uint gridX = (uint)((count + (int)blockSize - 1) / (int)blockSize);

        int localCount = count;

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dTarget;
        pArgs[1] = &dSource;
        pArgs[2] = &localCount;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnVecAdd,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(vec_add_kernel)");
    }

    private void DispatchMatMulBatchDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize, IntPtr dResidual = default)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);

        int localCols = nCols;
        int localRows = nRows;
        int localBatch = batchSize;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &localCols;
        pArgs[4] = &localRows;
        pArgs[5] = &localBatch;
        pArgs[6] = &dB;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmQ4KBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q4_k_batch)");
    }

    private void DispatchMatMulBatchQ6KDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize, IntPtr dResidual = default)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 7) / 8);

        int localCols = nCols;
        int localRows = nRows;
        int localBatch = batchSize;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dY;
        pArgs[1] = &dX;
        pArgs[2] = &dW;
        pArgs[3] = &localCols;
        pArgs[4] = &localRows;
        pArgs[5] = &localBatch;
        pArgs[6] = &dB;
        pArgs[7] = &dResidual;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGemmQ6KBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(gemm_q6_k_batch)");
    }

    private void DispatchAttentionDevice(
        IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dAttnOut,
        int nHeads, int headDim, int numTokens, float attnScale)
    {
        uint blockSize = 128;
        uint gridX = (uint)nHeads;
        uint gridY = (uint)numTokens;

        int localHeads = nHeads;
        int localHeadDim = headDim;
        int localTokens = numTokens;
        float localScale = attnScale;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dAttnOut;
        pArgs[4] = &localHeads;
        pArgs[5] = &localHeadDim;
        pArgs[6] = &localTokens;
        pArgs[7] = &localScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnBidirectionalBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_bidirectional_batch)");
    }

    private void DispatchAttentionCrossDevice(
        IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dAttnOut,
        int nHeads, int headDim, int numQTokens, int numKvTokens, float attnScale)
    {
        uint blockSize = 128;
        uint gridX = (uint)nHeads;
        uint gridY = (uint)numQTokens;

        int localHeads = nHeads;
        int localHeadDim = headDim;
        int localQTokens = numQTokens;
        int localKvTokens = numKvTokens;
        float localScale = attnScale;

        void** pArgs = stackalloc void*[9];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dAttnOut;
        pArgs[4] = &localHeads;
        pArgs[5] = &localHeadDim;
        pArgs[6] = &localQTokens;
        pArgs[7] = &localKvTokens;
        pArgs[8] = &localScale;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAttnCrossBatch,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(attention_cross_batch)");
    }

    private void DispatchRoPE3DDevice(IntPtr dQk, IntPtr dCos, IntPtr dSin, int totalTokens, int nHeads, int headDim)
    {
        uint blockSize = 64;
        uint gridX = (uint)totalTokens;
        uint gridY = (uint)nHeads;

        int localTokens = totalTokens;
        int localHeads = nHeads;
        int localHeadDim = headDim;

        void** pArgs = stackalloc void*[6];
        pArgs[0] = &dQk;
        pArgs[1] = &dCos;
        pArgs[2] = &dSin;
        pArgs[3] = &localTokens;
        pArgs[4] = &localHeads;
        pArgs[5] = &localHeadDim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnRoPE3D,
            gridX, gridY, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(rope_3d_in_vram)");
    }

    private void DispatchGeluDevice(IntPtr dX, int count)
    {
        uint blockSize = 256;
        uint gridX = (uint)((count + (int)blockSize - 1) / (int)blockSize);
        int localCount = count;

        void** pArgs = stackalloc void*[2];
        pArgs[0] = &dX;
        pArgs[1] = &localCount;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnGelu,
            gridX, 1, 1,
            blockSize, 1, 1,
            0, IntPtr.Zero,
            (IntPtr)pArgs,
            IntPtr.Zero), "LaunchKernel(flux_gelu_kernel)");
    }

}
