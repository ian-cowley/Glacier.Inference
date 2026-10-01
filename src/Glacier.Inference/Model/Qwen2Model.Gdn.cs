namespace Glacier.Inference.Model;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;

public sealed unsafe partial class Qwen2Model
{
    private void InitializeGdnBuffers()
    {
        if (!_hasGdn) return;

        // Allocate unmanaged zero-allocation recurrent state cache for Gated DeltaNet
        _ssmCache = new SsmStateCache(
            _weights.BlockCount,
            _ssmConvKernel,
            _gdnConvChannels,
            _ssmHeads,
            _ssmStateDim);

        // Scratch buffers for linear projections & gating
        _gdnQkv = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnConvOut = (float*)NativeMemory.AllocZeroed((nuint)(_gdnConvChannels * sizeof(float)));
        _gdnZ = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        _gdnA = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnB = (float*)NativeMemory.AllocZeroed((nuint)(_ssmHeads * sizeof(float)));
        _gdnY = (float*)NativeMemory.AllocZeroed((nuint)(_ssmInnerSize * sizeof(float)));
        int yChunks = (_ssmInnerSize + 31) / 32;
        _gdnYSums = (float*)NativeMemory.AllocZeroed((nuint)(yChunks * sizeof(float)));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayer(int stageLayer, int modelLayer, float* x)
    {
        ForwardGdnLayerWithNorm(stageLayer, modelLayer, x, _normX, _normXSums);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ForwardGdnLayerWithNorm(int stageLayer, int modelLayer, float* x, float* normX, float* normXSums)
    {
        var layer = _weights.Layers[modelLayer];
        if (_ssmCache == null) return;

        // 1. Compute projections from normalized x
        QuantKernels.MatVecMul(layer.QkvType, layer.QkvWeight, normX, _gdnQkv, _dim, _gdnConvChannels, normXSums);
        QuantKernels.MatVecMul(layer.AttnGateType, layer.AttnGateWeight, normX, _gdnZ, _dim, _ssmInnerSize, normXSums);
        QuantKernels.MatVecMul(GgufType.F32, (byte*)layer.SsmAlphaWeight, normX, _gdnA, _dim, _ssmHeads, normXSums);
        QuantKernels.MatVecMul(GgufType.F32, (byte*)layer.SsmBetaWeight, normX, _gdnB, _dim, _ssmHeads, normXSums);

        // 2. Depthwise 1D Causal Convolution with SiLU activation
        float* convState = _ssmCache.GetConvState(modelLayer);
        int convHistLen = _ssmConvKernel - 1; // 3
        for (int c = 0; c < _gdnConvChannels; c++)
        {
            float* w = layer.SsmConv1dWeight + c * _ssmConvKernel;
            float* h = convState + c * convHistLen;
            float qkvVal = _gdnQkv[c];
            float val = w[0] * h[0] + w[1] * h[1] + w[2] * h[2] + w[3] * qkvVal;
            h[0] = h[1];
            h[1] = h[2];
            h[2] = qkvVal;
            float sig = 1.0f / (1.0f + MathF.Exp(-val));
            _gdnConvOut[c] = val * sig;
        }

        // 3. Split q, k, v
        // q: SsmGroupCount * SsmStateDim (16 * 128 = 2048)
        // k: SsmGroupCount * SsmStateDim (16 * 128 = 2048)
        // v: SsmHeads * SsmStateDim (48 * 128 = 6144)
        int qkDim = _ssmGroupCount * _ssmStateDim;
        float* q = _gdnConvOut;
        float* k = _gdnConvOut + qkDim;
        float* v = _gdnConvOut + 2 * qkDim;

        // 4. L2-normalize q and k per head (with 1/sqrt(headDim) scale folded into q)
        float invSqrtDk = 1.0f / MathF.Sqrt((float)_ssmStateDim);
        for (int hQk = 0; hQk < _ssmGroupCount; hQk++)
        {
            float* qHead = q + hQk * _ssmStateDim;
            float* kHead = k + hQk * _ssmStateDim;

            float sumK = 0f;
            for (int i = 0; i < _ssmStateDim; i++) sumK += kHead[i] * kHead[i];
            float invNormK = 1.0f / MathF.Sqrt(sumK + 1e-6f);
            for (int i = 0; i < _ssmStateDim; i++) kHead[i] *= invNormK;

            float sumQ = 0f;
            for (int i = 0; i < _ssmStateDim; i++) sumQ += qHead[i] * qHead[i];
            float invNormQ = invSqrtDk / MathF.Sqrt(sumQ + 1e-6f);
            for (int i = 0; i < _ssmStateDim; i++) qHead[i] *= invNormQ;
        }

        // 5. Parallel recurrent Gated DeltaNet state update across all v heads
        int groupRatio = _ssmHeads / _ssmGroupCount; // e.g. 48 / 16 = 3
        Parallel.For(0, _ssmHeads, h =>
        {
            float valA = _gdnA[h] + (layer.SsmDtBias != null ? layer.SsmDtBias[h] : 0f);
            float dt = valA > 20.0f ? valA : MathF.Log(1.0f + MathF.Exp(valA));
            float aLog = layer.SsmAWeight != null ? layer.SsmAWeight[h] : 0f;
            float g = -MathF.Exp(aLog) * dt;
            float decay = MathF.Exp(g);

            float betaVal = 1.0f / (1.0f + MathF.Exp(-_gdnB[h]));

            int hQk = h / groupRatio;
            float* qHead = q + hQk * _ssmStateDim;
            float* kHead = k + hQk * _ssmStateDim;
            float* vHead = v + h * _ssmStateDim;
            float* yHead = _gdnY + h * _ssmStateDim;

            float* state = _ssmCache.GetRecurrentState(modelLayer, h);

            for (int dv = 0; dv < _ssmStateDim; dv++)
            {
                float* stateRow = state + dv * _ssmStateDim;
                float kvMem = 0.0f;
                for (int dk = 0; dk < _ssmStateDim; dk++)
                {
                    stateRow[dk] *= decay;
                    kvMem += stateRow[dk] * kHead[dk];
                }

                float delta = (vHead[dv] - kvMem) * betaVal;

                float outVal = 0.0f;
                for (int dk = 0; dk < _ssmStateDim; dk++)
                {
                    stateRow[dk] += kHead[dk] * delta;
                    outVal += stateRow[dk] * qHead[dk];
                }
                yHead[dv] = outVal;
            }
        });

        // 6. Head RMSNorm on _gdnY
        if (layer.SsmNormWeight != null)
        {
            for (int h = 0; h < _ssmHeads; h++)
            {
                QuantKernels.RMSNorm(_gdnY + h * _ssmStateDim, layer.SsmNormWeight, _gdnY + h * _ssmStateDim, _ssmStateDim, _weights.RmsNormEps);
            }
        }

        // 7. Output gating with SiLU(_gdnZ)
        for (int i = 0; i < _ssmInnerSize; i++)
        {
            float zVal = _gdnZ[i];
            float sigZ = 1.0f / (1.0f + MathF.Exp(-zVal));
            _gdnY[i] *= (zVal * sigZ);
        }

        // 8. Projection back to residual stream
        QuantKernels.ComputeBlockSums32(_gdnY, _gdnYSums, _ssmInnerSize);
        QuantKernels.MatVecMul(layer.SsmOutType, layer.SsmOutWeight, _gdnY, _attnProj, _ssmInnerSize, _dim, _gdnYSums);
        AddVector(x, _attnProj, _dim);
    }
}
