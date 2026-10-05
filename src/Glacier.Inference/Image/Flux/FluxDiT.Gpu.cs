namespace Glacier.Inference.Image.Flux;

using System;
using System.Diagnostics;
using Glacier.Inference.Format;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Quant;

public sealed unsafe partial class FluxDiT
{
    private void ExecuteDoubleBlocksGpu(
        float* txtTokens, float* imgTokens,
        int numTxtTokens, int numImgTokens,
        float* ropeCos, float* ropeSin)
    {
        int numTotalTokens = numTxtTokens + numImgTokens;
        EnsureGpuActivationBuffers(numTotalTokens);

        nuint singleStreamBytes = (nuint)((long)numTotalTokens * HiddenDim * sizeof(float));
        nuint txtBytes = (nuint)((long)numTxtTokens * HiddenDim * sizeof(float));
        nuint imgBytes = (nuint)((long)numImgTokens * HiddenDim * sizeof(float));

        // Upload tokens to GPU once at the start of block 0
        _gpu!.CopyToDevice(_dSingleStream, (IntPtr)txtTokens, singleStreamBytes);

        IntPtr dTxtTokens = _dSingleStream;
        IntPtr dImgTokens = _dSingleStream + (nint)txtBytes;

        IntPtr dNormBufTxt = _dNormBuf;
        IntPtr dNormBufImg = _dNormBuf + (nint)txtBytes;

        nuint txtQkvBytes = (nuint)((long)numTxtTokens * 9216 * sizeof(float));
        IntPtr dTxtQkv = _dQkvMlpBuf;
        IntPtr dImgQkv = _dQkvMlpBuf + (nint)txtQkvBytes;

        int stride = 9216;
        int nHeads = NumHeads;
        int headDim = HeadDim;
        int dim = HiddenDim;
        float attnScale = 1.0f / MathF.Sqrt(HeadDim);

        int localNumTxt = numTxtTokens;
        int localNumImg = numImgTokens;
        int localNumTotal = numTotalTokens;
        int txtOffset = 0;
        int imgOffset = numTxtTokens;

        IntPtr dQ = _dQ;
        IntPtr dK = _dK;
        IntPtr dV = _dV;
        IntPtr dRopeCos = _dRopeCos;
        IntPtr dRopeSin = _dRopeSin;
        IntPtr dAttnOut = _dAttnOut;
        IntPtr dQkvMlpBuf = _dQkvMlpBuf;

        void** pAdaTxtArgs = stackalloc void*[6];
        void** pAdaImgArgs = stackalloc void*[6];
        void** pTxtQkvArgs = stackalloc void*[13];
        void** pImgQkvArgs = stackalloc void*[13];
        void** pAttnArgs = stackalloc void*[8];
        void** pGeluArgs = stackalloc void*[2];
        void** pResTxtArgs = stackalloc void*[5];
        void** pResImgArgs = stackalloc void*[5];

        int totalTxtResElements = numTxtTokens * HiddenDim;
        uint resTxtBlocks = (uint)((totalTxtResElements + 255) / 256);

        int totalImgResElements = numImgTokens * HiddenDim;
        uint resImgBlocks = (uint)((totalImgResElements + 255) / 256);

        for (int b = 0; b < DoubleBlocksCount; b++)
        {
            var blk = _doubleBlocks[b];

            // 1. Modulations computed on CPU, copied to GPU buffers (18432 floats = 72 KB each)
            DispatchLinearVec(blk.ImgModW, blk.ImgModB, _siluVecBuffer, _imgModBuffer);
            DispatchLinearVec(blk.TxtModW, blk.TxtModB, _siluVecBuffer, _txtModBuffer);

            _gpu.CopyToDevice(_dDoubleModImg, (IntPtr)_imgModBuffer, 18432 * sizeof(float));
            _gpu.CopyToDevice(_dDoubleModTxt, (IntPtr)_txtModBuffer, 18432 * sizeof(float));

            IntPtr dTxtShift1 = _dDoubleModTxt;
            IntPtr dTxtScale1 = _dDoubleModTxt + 3072 * sizeof(float);
            IntPtr dTxtGate1  = _dDoubleModTxt + 6144 * sizeof(float);
            IntPtr dTxtShift2 = _dDoubleModTxt + 9216 * sizeof(float);
            IntPtr dTxtScale2 = _dDoubleModTxt + 12288 * sizeof(float);
            IntPtr dTxtGate2  = _dDoubleModTxt + 15360 * sizeof(float);

            IntPtr dImgShift1 = _dDoubleModImg;
            IntPtr dImgScale1 = _dDoubleModImg + 3072 * sizeof(float);
            IntPtr dImgGate1  = _dDoubleModImg + 6144 * sizeof(float);
            IntPtr dImgShift2 = _dDoubleModImg + 9216 * sizeof(float);
            IntPtr dImgScale2 = _dDoubleModImg + 12288 * sizeof(float);
            IntPtr dImgGate2  = _dDoubleModImg + 15360 * sizeof(float);

            // 2. AdaLN 1: txt & img -> normBuf
            pAdaTxtArgs[0] = &dTxtTokens;
            pAdaTxtArgs[1] = &dNormBufTxt;
            pAdaTxtArgs[2] = &dTxtShift1;
            pAdaTxtArgs[3] = &dTxtScale1;
            pAdaTxtArgs[4] = &localNumTxt;
            pAdaTxtArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTxtTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:txt1)");

            pAdaImgArgs[0] = &dImgTokens;
            pAdaImgArgs[1] = &dNormBufImg;
            pAdaImgArgs[2] = &dImgShift1;
            pAdaImgArgs[3] = &dImgScale1;
            pAdaImgArgs[4] = &localNumImg;
            pAdaImgArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:img1)");

            // 3. QKV Projections
            DispatchMatMulBatchDevice(blk.D_TxtQkvW, blk.D_TxtQkvB, dNormBufTxt, dTxtQkv, 3072, 9216, numTxtTokens);
            DispatchMatMulBatchDevice(blk.D_ImgQkvW, blk.D_ImgQkvB, dNormBufImg, dImgQkv, 3072, 9216, numImgTokens);

            // 4. QKV Prep: unpack, RMSNorm, 3D-RoPE into _dQ, _dK, _dV
            IntPtr dTxtQNorm = blk.D_TxtQueryNorm;
            IntPtr dTxtKNorm = blk.D_TxtKeyNorm;
            pTxtQkvArgs[0] = &dTxtQkv;
            pTxtQkvArgs[1] = &dQ;
            pTxtQkvArgs[2] = &dK;
            pTxtQkvArgs[3] = &dV;
            pTxtQkvArgs[4] = &dTxtQNorm;
            pTxtQkvArgs[5] = &dTxtKNorm;
            pTxtQkvArgs[6] = &dRopeCos;
            pTxtQkvArgs[7] = &dRopeSin;
            pTxtQkvArgs[8] = &stride;
            pTxtQkvArgs[9] = &nHeads;
            pTxtQkvArgs[10] = &headDim;
            pTxtQkvArgs[11] = &localNumTxt;
            pTxtQkvArgs[12] = &txtOffset;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTxtTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pTxtQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:txt)");

            IntPtr dImgQNorm = blk.D_ImgQueryNorm;
            IntPtr dImgKNorm = blk.D_ImgKeyNorm;
            pImgQkvArgs[0] = &dImgQkv;
            pImgQkvArgs[1] = &dQ;
            pImgQkvArgs[2] = &dK;
            pImgQkvArgs[3] = &dV;
            pImgQkvArgs[4] = &dImgQNorm;
            pImgQkvArgs[5] = &dImgKNorm;
            pImgQkvArgs[6] = &dRopeCos;
            pImgQkvArgs[7] = &dRopeSin;
            pImgQkvArgs[8] = &stride;
            pImgQkvArgs[9] = &nHeads;
            pImgQkvArgs[10] = &headDim;
            pImgQkvArgs[11] = &localNumImg;
            pImgQkvArgs[12] = &imgOffset;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numImgTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pImgQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:img)");

            // 5. FlashAttention-2 Joint Attention -> _dAttnOut [numTotalTokens, 3072]
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

            // 6. Attention Output Projections & Residual 1
            IntPtr dAttnOutImg = _dAttnOut + (nint)txtBytes;
            DispatchMatMulBatchDevice(blk.D_TxtProjW, blk.D_TxtProjB, dAttnOut, dNormBufTxt, 3072, 3072, numTxtTokens);
            pResTxtArgs[0] = &dTxtTokens;
            pResTxtArgs[1] = &dNormBufTxt;
            pResTxtArgs[2] = &dTxtGate1;
            pResTxtArgs[3] = &totalTxtResElements;
            pResTxtArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:txt1)");

            DispatchMatMulBatchDevice(blk.D_ImgProjW, blk.D_ImgProjB, dAttnOutImg, dNormBufImg, 3072, 3072, numImgTokens);
            pResImgArgs[0] = &dImgTokens;
            pResImgArgs[1] = &dNormBufImg;
            pResImgArgs[2] = &dImgGate1;
            pResImgArgs[3] = &totalImgResElements;
            pResImgArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:img1)");

            // 7. AdaLN 2: txt & img -> normBuf
            pAdaTxtArgs[0] = &dTxtTokens;
            pAdaTxtArgs[1] = &dNormBufTxt;
            pAdaTxtArgs[2] = &dTxtShift2;
            pAdaTxtArgs[3] = &dTxtScale2;
            pAdaTxtArgs[4] = &localNumTxt;
            pAdaTxtArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTxtTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:txt2)");

            pAdaImgArgs[0] = &dImgTokens;
            pAdaImgArgs[1] = &dNormBufImg;
            pAdaImgArgs[2] = &dImgShift2;
            pAdaImgArgs[3] = &dImgScale2;
            pAdaImgArgs[4] = &localNumImg;
            pAdaImgArgs[5] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel:img2)");

            // 8. Txt MLP: [3072] -> [12288] -> GeLU -> [3072] + Residual
            DispatchMatMulBatchDevice(blk.D_TxtMlp0W, blk.D_TxtMlp0B, dNormBufTxt, _dQkvMlpBuf, 3072, 12288, numTxtTokens);
            int txtMlpCount = numTxtTokens * 12288;
            uint geluTxtBlocks = (uint)((txtMlpCount + 255) / 256);
            pGeluArgs[0] = &dQkvMlpBuf;
            pGeluArgs[1] = &txtMlpCount;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGelu, geluTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_gelu_kernel:txt)");
            DispatchMatMulBatchDevice(blk.D_TxtMlp2W, blk.D_TxtMlp2B, _dQkvMlpBuf, dNormBufTxt, 12288, 3072, numTxtTokens);
            pResTxtArgs[0] = &dTxtTokens;
            pResTxtArgs[1] = &dNormBufTxt;
            pResTxtArgs[2] = &dTxtGate2;
            pResTxtArgs[3] = &totalTxtResElements;
            pResTxtArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resTxtBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResTxtArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:txt2)");

            // 9. Img MLP: [3072] -> [12288] -> GeLU -> [3072] + Residual
            DispatchMatMulBatchDevice(blk.D_ImgMlp0W, blk.D_ImgMlp0B, dNormBufImg, _dQkvMlpBuf, 3072, 12288, numImgTokens);
            int imgMlpCount = numImgTokens * 12288;
            uint geluImgBlocks = (uint)((imgMlpCount + 255) / 256);
            pGeluArgs[0] = &dQkvMlpBuf;
            pGeluArgs[1] = &imgMlpCount;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGelu, geluImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_gelu_kernel:img)");
            DispatchMatMulBatchDevice(blk.D_ImgMlp2W, blk.D_ImgMlp2B, _dQkvMlpBuf, dNormBufImg, 12288, 3072, numImgTokens);
            pResImgArgs[0] = &dImgTokens;
            pResImgArgs[1] = &dNormBufImg;
            pResImgArgs[2] = &dImgGate2;
            pResImgArgs[3] = &totalImgResElements;
            pResImgArgs[4] = &dim;
            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resImgBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResImgArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated:img2)");
        }

        if (!_fullGpuMode)
        {
            _gpu!.Synchronize();
            _gpu!.CopyToHost((IntPtr)txtTokens, _dSingleStream, singleStreamBytes);
        }
    }

    private void ExecuteSingleBlocksGpu(float* singleStream, int numTotalTokens, bool alreadyOnGpu = false)
    {
        EnsureGpuActivationBuffers(numTotalTokens);

        nuint singleStreamBytes = (nuint)((long)numTotalTokens * HiddenDim * sizeof(float));
        if (!alreadyOnGpu)
        {
            _gpu!.CopyToDevice(_dSingleStream, (IntPtr)singleStream, singleStreamBytes);
        }

        float attnScale = 1.0f / MathF.Sqrt(HeadDim);
        int stride1 = 21504;
        int nHeads = NumHeads;
        int headDim = HeadDim;
        int dim = HiddenDim;
        int localTotalTokens = numTotalTokens;

        IntPtr dSingleStream = _dSingleStream;
        IntPtr dNormBuf = _dNormBuf;
        IntPtr dQkvMlpBuf = _dQkvMlpBuf;
        IntPtr dAttnOut = _dAttnOut;
        IntPtr dConcatBuf = _dConcatBuf;
        IntPtr dQ = _dQ;
        IntPtr dK = _dK;
        IntPtr dV = _dV;
        IntPtr dRopeCos = _dRopeCos;
        IntPtr dRopeSin = _dRopeSin;
        IntPtr dSingleMod = _dSingleMod;

        void** pAdaArgs = stackalloc void*[6];
        void** pQkvArgs = stackalloc void*[13];
        void** pAttnArgs = stackalloc void*[8];
        void** pGeluArgs = stackalloc void*[4];
        void** pResArgs = stackalloc void*[5];

        for (int b = 0; b < SingleBlocksCount; b++)
        {
            var blk = _singleBlocks[b];

            // 1. Modulation vector on CPU -> copy 36KB to GPU
            DispatchLinearVec(blk.ModW, blk.ModB, _siluVecBuffer, _singleModBuffer);
            _gpu!.CopyToDevice(dSingleMod, (IntPtr)_singleModBuffer, 9216 * sizeof(float));

            IntPtr dShift = dSingleMod;
            IntPtr dScale = dSingleMod + 3072 * sizeof(float);
            IntPtr dGate = dSingleMod + 6144 * sizeof(float);

            // 2. AdaLN: _dSingleStream -> _dNormBuf
            pAdaArgs[0] = &dSingleStream;
            pAdaArgs[1] = &dNormBuf;
            pAdaArgs[2] = &dShift;
            pAdaArgs[3] = &dScale;
            pAdaArgs[4] = &localTotalTokens;
            pAdaArgs[5] = &dim;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAdaLn, (uint)numTotalTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaArgs, IntPtr.Zero),
                "LaunchKernel(flux_adaln_kernel)");

            // 3. Fused Linear1: _dNormBuf [N, 3072] -> _dQkvMlpBuf [N, 21504]
            DispatchMatMulBatchDevice(blk.D_Lin1W, blk.D_Lin1B, dNormBuf, dQkvMlpBuf, 3072, 21504, numTotalTokens);

            // 4. Fused QKV Unpack + RMSNorm + 3D-RoPE: _dQkvMlpBuf -> _dQ, _dK, _dV
            int tokenOffset = 0;
            IntPtr dQueryNorm = blk.D_QueryNorm;
            IntPtr dKeyNorm = blk.D_KeyNorm;
            pQkvArgs[0] = &dQkvMlpBuf;
            pQkvArgs[1] = &dQ;
            pQkvArgs[2] = &dK;
            pQkvArgs[3] = &dV;
            pQkvArgs[4] = &dQueryNorm;
            pQkvArgs[5] = &dKeyNorm;
            pQkvArgs[6] = &dRopeCos;
            pQkvArgs[7] = &dRopeSin;
            pQkvArgs[8] = &stride1;
            pQkvArgs[9] = &nHeads;
            pQkvArgs[10] = &headDim;
            pQkvArgs[11] = &localTotalTokens;
            pQkvArgs[12] = &tokenOffset;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnQkvPrep, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pQkvArgs, IntPtr.Zero),
                "LaunchKernel(flux_qkv_prep:single)");

            // 5. FlashAttention-2: _dQ, _dK, _dV -> _dAttnOut [N, 3072]
            pAttnArgs[0] = &dQ;
            pAttnArgs[1] = &dK;
            pAttnArgs[2] = &dV;
            pAttnArgs[3] = &dAttnOut;
            pAttnArgs[4] = &nHeads;
            pAttnArgs[5] = &headDim;
            pAttnArgs[6] = &localTotalTokens;
            pAttnArgs[7] = &attnScale;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnAttnBidirectionalBatch, (uint)NumHeads, (uint)numTotalTokens, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAttnArgs, IntPtr.Zero),
                "LaunchKernel(attention_bidirectional_batch:single)");

            // 6. Fused GELU + Concat: _dAttnOut [N, 3072] + GELU(_dQkvMlpBuf[:, 9216..]) -> _dConcatBuf [N, 15360]
            int totalConcatElements = numTotalTokens * 15360;
            uint concatBlocks = (uint)((totalConcatElements + 255) / 256);
            pGeluArgs[0] = &dAttnOut;
            pGeluArgs[1] = &dQkvMlpBuf;
            pGeluArgs[2] = &dConcatBuf;
            pGeluArgs[3] = &localTotalTokens;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGeluConcat, concatBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pGeluArgs, IntPtr.Zero),
                "LaunchKernel(flux_fused_gelu_concat)");

            // 7. Fused Linear2: _dConcatBuf [N, 15360] -> _dNormBuf [N, 3072]
            DispatchMatMulBatchDevice(blk.D_Lin2W, blk.D_Lin2B, dConcatBuf, dNormBuf, 15360, 3072, numTotalTokens);

            // 8. Residual Gated: _dSingleStream += _dNormBuf * gate
            int totalResElements = numTotalTokens * HiddenDim;
            uint resBlocks = (uint)((totalResElements + 255) / 256);
            pResArgs[0] = &dSingleStream;
            pResArgs[1] = &dNormBuf;
            pResArgs[2] = &dGate;
            pResArgs[3] = &totalResElements;
            pResArgs[4] = &dim;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnResidualGated, resBlocks, 1, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)pResArgs, IntPtr.Zero),
                "LaunchKernel(flux_residual_gated)");
        }

        // Copy back to host only if final layer is not going to execute on GPU
        if (!_fullGpuMode || _dFinalLinearW == IntPtr.Zero)
        {
            _gpu!.Synchronize();
            _gpu!.CopyToHost((IntPtr)singleStream, _dSingleStream, singleStreamBytes);
        }
    }

    private void ExecuteFinalLayerGpu(int numTxtTokens, int numImgTokens, float* patchifiedVel)
    {
        // 1. Final modulation vector on CPU (6144 floats = 24 KB)
        DispatchLinearVec(_finalAdaLnW, _finalAdaLnB, _siluVecBuffer, _finalModBuffer);
        _gpu!.CopyToDevice(_dSingleMod, (IntPtr)_finalModBuffer, 6144 * sizeof(float));

        IntPtr dShift = _dSingleMod;
        IntPtr dScale = _dSingleMod + 3072 * sizeof(float);

        IntPtr dImgTokens = _dSingleStream + (nint)((long)numTxtTokens * HiddenDim * sizeof(float));
        IntPtr dNormBuf = _dNormBuf;
        int dim = HiddenDim;
        int localImgTokens = numImgTokens;

        void** pAdaArgs = stackalloc void*[6];
        pAdaArgs[0] = &dImgTokens;
        pAdaArgs[1] = &dNormBuf;
        pAdaArgs[2] = &dShift;
        pAdaArgs[3] = &dScale;
        pAdaArgs[4] = &localImgTokens;
        pAdaArgs[5] = &dim;

        CuDriver.Check(CuDriver.LaunchKernel(
            _fnAdaLn, (uint)numImgTokens, 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)pAdaArgs, IntPtr.Zero),
            "LaunchKernel(flux_adaln_kernel:final)");

        // 2. Final Linear: [numImgTokens, 3072] * [3072, 64] -> [numImgTokens, 64]
        DispatchMatMulBatchDevice(_dFinalLinearW, _dFinalLinearB, dNormBuf, _dFinalOut, 3072, PatchDim, numImgTokens);

        nuint velBytes = (nuint)((long)numImgTokens * PatchDim * sizeof(float));
        _gpu.Synchronize();
        _gpu.CopyToHost((IntPtr)patchifiedVel, _dFinalOut, velBytes);
    }

    private IntPtr UploadVector(GgufTensorInfo t, int count)
    {
        if (_gpu == null) return IntPtr.Zero;
        nuint bytes = (nuint)((long)count * sizeof(float));
        IntPtr d = _gpu.AllocateDevice(bytes);
        _gpu.CopyToDevice(d, (IntPtr)_gguf.GetTensorPointer(t), bytes);
        return d;
    }

    private (IntPtr dW, IntPtr dB) UploadWeightAndBias(GgufTensorInfo w, GgufTensorInfo b, int nRows)
    {
        if (_gpu == null || w.Type != GgufType.Q4_K) return (IntPtr.Zero, IntPtr.Zero);

        nuint wBytes = (nuint)w.GetByteSize();
        nuint bBytes = (nuint)(nRows * sizeof(float));

        IntPtr dW = _gpu.AllocateDevice(wBytes);
        IntPtr dB = _gpu.AllocateDevice(bBytes);

        _gpu.CopyToDevice(dW, (IntPtr)_gguf.GetTensorPointer(w), wBytes);
        _gpu.CopyToDevice(dB, (IntPtr)_gguf.GetTensorPointer(b), bBytes);

        return (dW, dB);
    }

    private void FreeWeightAndBias(ref IntPtr dW, ref IntPtr dB)
    {
        if (_gpu != null)
        {
            if (dW != IntPtr.Zero) { _gpu.FreeDevice(dW); dW = IntPtr.Zero; }
            if (dB != IntPtr.Zero) { _gpu.FreeDevice(dB); dB = IntPtr.Zero; }
        }
    }

    private void UploadDoubleBlocks()
    {
        if (_gpu == null || _doubleBlocksLoaded) return;
        for (int i = 0; i < DoubleBlocksCount; i++)
        {
            var blk = _doubleBlocks[i];
            (blk.D_ImgQkvW, blk.D_ImgQkvB) = UploadWeightAndBias(blk.ImgQkvW, blk.ImgQkvB, 9216);
            (blk.D_TxtQkvW, blk.D_TxtQkvB) = UploadWeightAndBias(blk.TxtQkvW, blk.TxtQkvB, 9216);
            (blk.D_ImgProjW, blk.D_ImgProjB) = UploadWeightAndBias(blk.ImgProjW, blk.ImgProjB, HiddenDim);
            (blk.D_TxtProjW, blk.D_TxtProjB) = UploadWeightAndBias(blk.TxtProjW, blk.TxtProjB, HiddenDim);
            (blk.D_ImgMlp0W, blk.D_ImgMlp0B) = UploadWeightAndBias(blk.ImgMlp0W, blk.ImgMlp0B, 12288);
            (blk.D_ImgMlp2W, blk.D_ImgMlp2B) = UploadWeightAndBias(blk.ImgMlp2W, blk.ImgMlp2B, HiddenDim);
            (blk.D_TxtMlp0W, blk.D_TxtMlp0B) = UploadWeightAndBias(blk.TxtMlp0W, blk.TxtMlp0B, 12288);
            (blk.D_TxtMlp2W, blk.D_TxtMlp2B) = UploadWeightAndBias(blk.TxtMlp2W, blk.TxtMlp2B, HiddenDim);
            blk.D_ImgQueryNorm = UploadVector(blk.ImgQueryNorm, HeadDim);
            blk.D_ImgKeyNorm = UploadVector(blk.ImgKeyNorm, HeadDim);
            blk.D_TxtQueryNorm = UploadVector(blk.TxtQueryNorm, HeadDim);
            blk.D_TxtKeyNorm = UploadVector(blk.TxtKeyNorm, HeadDim);
        }
        _doubleBlocksLoaded = true;
    }

    private void FreeDoubleBlocks()
    {
        if (_gpu == null || !_doubleBlocksLoaded) return;
        for (int i = 0; i < DoubleBlocksCount; i++)
        {
            var blk = _doubleBlocks[i];
            FreeWeightAndBias(ref blk.D_ImgQkvW, ref blk.D_ImgQkvB);
            FreeWeightAndBias(ref blk.D_TxtQkvW, ref blk.D_TxtQkvB);
            FreeWeightAndBias(ref blk.D_ImgProjW, ref blk.D_ImgProjB);
            FreeWeightAndBias(ref blk.D_TxtProjW, ref blk.D_TxtProjB);
            FreeWeightAndBias(ref blk.D_ImgMlp0W, ref blk.D_ImgMlp0B);
            FreeWeightAndBias(ref blk.D_ImgMlp2W, ref blk.D_ImgMlp2B);
            FreeWeightAndBias(ref blk.D_TxtMlp0W, ref blk.D_TxtMlp0B);
            FreeWeightAndBias(ref blk.D_TxtMlp2W, ref blk.D_TxtMlp2B);
            if (blk.D_ImgQueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_ImgQueryNorm); blk.D_ImgQueryNorm = IntPtr.Zero; }
            if (blk.D_ImgKeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_ImgKeyNorm); blk.D_ImgKeyNorm = IntPtr.Zero; }
            if (blk.D_TxtQueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_TxtQueryNorm); blk.D_TxtQueryNorm = IntPtr.Zero; }
            if (blk.D_TxtKeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_TxtKeyNorm); blk.D_TxtKeyNorm = IntPtr.Zero; }
        }
        _doubleBlocksLoaded = false;
    }

    private void UploadSingleBlocks()
    {
        if (_gpu == null || _singleBlocksLoaded) return;
        for (int i = 0; i < SingleBlocksCount; i++)
        {
            var blk = _singleBlocks[i];
            (blk.D_Lin1W, blk.D_Lin1B) = UploadWeightAndBias(blk.Lin1W, blk.Lin1B, 21504);
            (blk.D_Lin2W, blk.D_Lin2B) = UploadWeightAndBias(blk.Lin2W, blk.Lin2B, HiddenDim);
            blk.D_QueryNorm = UploadVector(blk.QueryNorm, HeadDim);
            blk.D_KeyNorm = UploadVector(blk.KeyNorm, HeadDim);
        }
        _singleBlocksLoaded = true;
    }

    private void FreeSingleBlocks()
    {
        if (_gpu == null || !_singleBlocksLoaded) return;
        for (int i = 0; i < SingleBlocksCount; i++)
        {
            var blk = _singleBlocks[i];
            FreeWeightAndBias(ref blk.D_Lin1W, ref blk.D_Lin1B);
            FreeWeightAndBias(ref blk.D_Lin2W, ref blk.D_Lin2B);
            if (blk.D_QueryNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_QueryNorm); blk.D_QueryNorm = IntPtr.Zero; }
            if (blk.D_KeyNorm != IntPtr.Zero) { _gpu.FreeDevice(blk.D_KeyNorm); blk.D_KeyNorm = IntPtr.Zero; }
        }
        _singleBlocksLoaded = false;
    }

    private void EnsureGpuActivationBuffers(int numTotalTokens)
    {
        if (_gpu == null || _allocatedGpuTokens >= numTotalTokens) return;
        FreeGpuActivationBuffers();

        int n = numTotalTokens;
        _dSingleStream = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dNormBuf = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dQkvMlpBuf = _gpu.AllocateDevice((nuint)((long)n * 21504 * sizeof(float)));
        _dAttnOut = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dConcatBuf = _gpu.AllocateDevice((nuint)((long)n * 15360 * sizeof(float)));
        _dQ = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dK = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dV = _gpu.AllocateDevice((nuint)((long)n * HiddenDim * sizeof(float)));
        _dRopeCos = _gpu.AllocateDevice((nuint)((long)n * 64 * sizeof(float)));
        _dRopeSin = _gpu.AllocateDevice((nuint)((long)n * 64 * sizeof(float)));
        _dSingleMod = _gpu.AllocateDevice((nuint)(9216 * sizeof(float)));
        _dDoubleModImg = _gpu.AllocateDevice((nuint)(18432 * sizeof(float)));
        _dDoubleModTxt = _gpu.AllocateDevice((nuint)(18432 * sizeof(float)));
        _dFinalOut = _gpu.AllocateDevice((nuint)((long)n * PatchDim * sizeof(float)));

        _allocatedGpuTokens = n;
    }

    private void FreeGpuActivationBuffers()
    {
        if (_gpu != null)
        {
            if (_dSingleStream != IntPtr.Zero) { _gpu.FreeDevice(_dSingleStream); _dSingleStream = IntPtr.Zero; }
            if (_dNormBuf != IntPtr.Zero) { _gpu.FreeDevice(_dNormBuf); _dNormBuf = IntPtr.Zero; }
            if (_dQkvMlpBuf != IntPtr.Zero) { _gpu.FreeDevice(_dQkvMlpBuf); _dQkvMlpBuf = IntPtr.Zero; }
            if (_dAttnOut != IntPtr.Zero) { _gpu.FreeDevice(_dAttnOut); _dAttnOut = IntPtr.Zero; }
            if (_dConcatBuf != IntPtr.Zero) { _gpu.FreeDevice(_dConcatBuf); _dConcatBuf = IntPtr.Zero; }
            if (_dQ != IntPtr.Zero) { _gpu.FreeDevice(_dQ); _dQ = IntPtr.Zero; }
            if (_dK != IntPtr.Zero) { _gpu.FreeDevice(_dK); _dK = IntPtr.Zero; }
            if (_dV != IntPtr.Zero) { _gpu.FreeDevice(_dV); _dV = IntPtr.Zero; }
            if (_dRopeCos != IntPtr.Zero) { _gpu.FreeDevice(_dRopeCos); _dRopeCos = IntPtr.Zero; }
            if (_dRopeSin != IntPtr.Zero) { _gpu.FreeDevice(_dRopeSin); _dRopeSin = IntPtr.Zero; }
            if (_dSingleMod != IntPtr.Zero) { _gpu.FreeDevice(_dSingleMod); _dSingleMod = IntPtr.Zero; }
            if (_dDoubleModImg != IntPtr.Zero) { _gpu.FreeDevice(_dDoubleModImg); _dDoubleModImg = IntPtr.Zero; }
            if (_dDoubleModTxt != IntPtr.Zero) { _gpu.FreeDevice(_dDoubleModTxt); _dDoubleModTxt = IntPtr.Zero; }
            if (_dFinalOut != IntPtr.Zero) { _gpu.FreeDevice(_dFinalOut); _dFinalOut = IntPtr.Zero; }
        }
        _allocatedGpuTokens = 0;
    }

    private void PrepareDoubleStreamGpu()
    {
        if (_gpu == null || _fullGpuMode) return;
        FreeSingleBlocks();
        UploadDoubleBlocks();
    }

    private void PrepareSingleStreamGpu()
    {
        if (_gpu == null || _fullGpuMode) return;
        FreeDoubleBlocks();
        UploadSingleBlocks();
    }

    private void EnsureGpuScratchBuffers(int batchSize, int nCols, int nRows)
    {
        nuint neededX = (nuint)((long)batchSize * nCols * sizeof(float));
        nuint neededY = (nuint)((long)batchSize * nRows * sizeof(float));

        if (neededX > _dXDeviceCapacity)
        {
            if (_dXDevice != IntPtr.Zero) _gpu!.FreeDevice(_dXDevice);
            _dXDeviceCapacity = (nuint)(neededX * 12 / 10 + 1024 * 1024);
            _dXDevice = _gpu!.AllocateDevice(_dXDeviceCapacity);
        }

        if (neededY > _dYDeviceCapacity)
        {
            if (_dYDevice != IntPtr.Zero) _gpu!.FreeDevice(_dYDevice);
            _dYDeviceCapacity = (nuint)(neededY * 12 / 10 + 1024 * 1024);
            _dYDevice = _gpu!.AllocateDevice(_dYDeviceCapacity);
        }
    }

    private void DispatchMatMulBatch(GgufTensorInfo w, GgufTensorInfo b, float* xBatch, float* yBatch, int nCols, int nRows, int batchSize)
        => DispatchMatMulBatch(w, b, IntPtr.Zero, IntPtr.Zero, xBatch, yBatch, nCols, nRows, batchSize);

    private void DispatchMatMulBatch(
        GgufTensorInfo w, GgufTensorInfo b,
        IntPtr dW, IntPtr dB,
        float* xBatch, float* yBatch,
        int nCols, int nRows, int batchSize)
    {
        if (_gpu != null && dW != IntPtr.Zero && w.Type == GgufType.Q4_K)
        {
            EnsureGpuScratchBuffers(batchSize, nCols, nRows);

            nuint xBytes = (nuint)((long)batchSize * nCols * sizeof(float));
            nuint yBytes = (nuint)((long)batchSize * nRows * sizeof(float));

            _gpu.CopyToDevice(_dXDevice, (IntPtr)xBatch, xBytes);

            uint blockSize = 128;
            uint numWarps = 4;
            uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
            uint gridY = (uint)((batchSize + 15) / 16);
            IntPtr dResidual = IntPtr.Zero;

            IntPtr dY = _dYDevice;
            IntPtr dX = _dXDevice;
            int localCols = nCols;
            int localRows = nRows;
            int localBatch = batchSize;
            IntPtr localW = dW;
            IntPtr localB = dB;

            void** pArgs = stackalloc void*[8];
            pArgs[0] = &dY;
            pArgs[1] = &dX;
            pArgs[2] = &localW;
            pArgs[3] = &localCols;
            pArgs[4] = &localRows;
            pArgs[5] = &localBatch;
            pArgs[6] = &localB;
            pArgs[7] = &dResidual;

            CuDriver.Check(CuDriver.LaunchKernel(
                _fnGemmQ4KBatch,
                gridX, gridY, 1,
                blockSize, 1, 1,
                0, IntPtr.Zero,
                (IntPtr)pArgs,
                IntPtr.Zero), "LaunchKernel(gemm_q4_k_batch)");

            _gpu.Synchronize();
            _gpu.CopyToHost((IntPtr)yBatch, _dYDevice, yBytes);
            return;
        }

        byte* wPtr = _gguf.GetTensorPointer(w);
        float* bPtr = (float*)_gguf.GetTensorPointer(b);

        QuantKernels.MatMulBatch(w.Type, wPtr, xBatch, yBatch, nCols, nRows, batchSize);

        // Add bias across batch
        nint yPtr = (nint)yBatch;
        nint bVal = (nint)bPtr;

        Parallel.For(0, batchSize, bIdx =>
        {
            float* row = (float*)yPtr + (long)bIdx * nRows;
            float* bArr = (float*)bVal;
            for (int r = 0; r < nRows; r++) row[r] += bArr[r];
        });
    }

    private void DispatchMatMulBatchDevice(
        IntPtr dW, IntPtr dB,
        IntPtr dX, IntPtr dY,
        int nCols, int nRows, int batchSize)
    {
        uint blockSize = 128;
        uint numWarps = 4;
        uint gridX = (uint)((nRows + (int)numWarps - 1) / (int)numWarps);
        uint gridY = (uint)((batchSize + 15) / 16);
        IntPtr dResidual = IntPtr.Zero;

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

}
