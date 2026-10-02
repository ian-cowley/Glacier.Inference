namespace Glacier.Inference.Quant;

using System;
using System.Runtime.CompilerServices;
using Glacier.Inference.Gguf;

public static unsafe partial class QuantKernels
{
    /// <summary>
    /// Executes a single Gemma 4 MoE expert:
    /// 1. Fused Gate-Up projection [dim -> 2 * expertFfnLength] via matrix-vector product
    /// 2. GeLU-GLU activation: gelu(gate) * up [expertFfnLength]
    /// 3. Down projection [expertFfnLength -> dim] via matrix-vector product
    /// 4. Scales by expert down-scale and routing weight, accumulating directly into moeAccumulator.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ExecuteGemma4Expert(
        int expertIndex,
        float weight,
        float* x,
        float* xSums,
        byte* gateUpExpsWeight,
        GgufType gateUpExpsType,
        byte* downExpsWeight,
        GgufType downExpsType,
        float* downExpsScale,
        int dim,
        int expertFfnLength,
        float* scratchFused,
        float* scratchAct,
        float* scratchActSums,
        float* moeAccumulator)
    {
        if (weight <= 0.0f)
        {
            new Span<float>(moeAccumulator, dim).Clear();
            return;
        }

        int fusedRows = 2 * expertFfnLength;
        int rowBytesGateUp = (int)GgufTypes.GetRowBytes(gateUpExpsType, dim);
        long expertOffsetGateUp = (long)expertIndex * fusedRows * rowBytesGateUp;
        byte* expertGateUpPtr = gateUpExpsWeight + expertOffsetGateUp;

        // 1. Fused Gate-Up Matrix-Vector Multiplication
        for (int r = 0; r < fusedRows; r++)
        {
            byte* rowPtr = expertGateUpPtr + (long)r * rowBytesGateUp;
            if (r + 1 < fusedRows && System.Runtime.Intrinsics.X86.Sse.IsSupported)
            {
                System.Runtime.Intrinsics.X86.Sse.Prefetch0(rowPtr + rowBytesGateUp);
            }
            scratchFused[r] = ComputeDot(gateUpExpsType, rowPtr, x, xSums, dim);
        }

        // 2. GeLU-GLU: gelu(gate) * up
        float* gate = scratchFused;
        float* up = scratchFused + expertFfnLength;
        GeluGLU(gate, up, scratchAct, expertFfnLength);

        // 3. Down Projection
        ComputeBlockSums32(scratchAct, scratchActSums, expertFfnLength);
        int rowBytesDown = (int)GgufTypes.GetRowBytes(downExpsType, expertFfnLength);
        long expertOffsetDown = (long)expertIndex * dim * rowBytesDown;
        byte* expertDownPtr = downExpsWeight + expertOffsetDown;

        float expertScale = downExpsScale != null ? downExpsScale[expertIndex] : 1.0f;
        float finalScale = expertScale * weight;

        for (int r = 0; r < dim; r++)
        {
            byte* rowPtr = expertDownPtr + (long)r * rowBytesDown;
            if (r + 1 < dim && System.Runtime.Intrinsics.X86.Sse.IsSupported)
            {
                System.Runtime.Intrinsics.X86.Sse.Prefetch0(rowPtr + rowBytesDown);
            }
            float d = ComputeDot(downExpsType, rowPtr, scratchAct, scratchActSums, expertFfnLength);
            moeAccumulator[r] = d * finalScale;
        }
    }
}

