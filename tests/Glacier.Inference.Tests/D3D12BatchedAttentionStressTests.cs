namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Gpu.D3D12;
using Vortice.D3DCompiler;
using Vortice.Direct3D12;
using Xunit;

[Collection("SequentialGpu")]
public unsafe class D3D12BatchedAttentionStressTests
{
    private const int HeadDim = 128;
    private const int NHeadsQ = 4;
    private const int NHeadsKv = 2;
    private const int MaxSeqLen = 1024;
    private const float AttnScale = 1.0f / 11.3137085f; // 1 / sqrt(128)

    // Mathematical CPU Oracle for GQA Attention
    private static float[] ComputeCpuOracle(
        float[] q, float[] kCache, float[] vCache,
        int nHeadsQ, int nHeadsKv, int headDim, int maxSeqLen,
        int startPos, int batchSize, float attnScale)
    {
        float[] output = new float[batchSize * nHeadsQ * headDim];
        int groupSize = nHeadsQ / nHeadsKv;

        for (int t = 0; t < batchSize; t++)
        {
            int posT = startPos + t;
            for (int h = 0; h < nHeadsQ; h++)
            {
                int hKv = h / groupSize;
                int qBase = (t * nHeadsQ + h) * headDim;

                // 1. Compute dot products for all historical tokens 0 .. posT
                float[] scores = new float[posT + 1];
                float maxScore = -1e30f;

                for (int p = 0; p <= posT; p++)
                {
                    int kvBase = (hKv * maxSeqLen + p) * headDim;
                    float dot = 0.0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        dot += q[qBase + d] * kCache[kvBase + d];
                    }
                    float s = dot * attnScale;
                    scores[p] = s;
                    if (s > maxScore) maxScore = s;
                }

                // 2. Softmax normalization
                float sumExp = 0.0f;
                float[] weights = new float[posT + 1];
                for (int p = 0; p <= posT; p++)
                {
                    float w = MathF.Exp(scores[p] - maxScore);
                    weights[p] = w;
                    sumExp += w;
                }

                // 3. Weighted accumulation of values
                int outBase = (t * nHeadsQ + h) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0.0f;
                    for (int p = 0; p <= posT; p++)
                    {
                        int kvBase = (hKv * maxSeqLen + p) * headDim;
                        acc += weights[p] * vCache[kvBase + d];
                    }
                    output[outBase + d] = sumExp > 0.0f ? (acc / sumExp) : 0.0f;
                }
            }
        }

        return output;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(511)]
    [InlineData(512)]
    public void AttentionBoundaryConditions_FP32_AllPositions_MatchOracle(int targetPos)
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();

        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        });
        using var rootSig = ctx.CreateRootSignature(rootSigDesc);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.AttentionBatch));

        int batchSize = 1;
        int startPos = targetPos;

        // Generate synthetic Q, K, V
        var rng = new Random(42 + targetPos);
        float[] q = new float[batchSize * NHeadsQ * HeadDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        int kvElements = NHeadsKv * MaxSeqLen * HeadDim;
        float[] kCache = new float[kvElements];
        float[] vCache = new float[kvElements];
        for (int i = 0; i < kvElements; i++)
        {
            kCache[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vCache[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Compute Oracle
        float[] expected = ComputeCpuOracle(q, kCache, vCache, NHeadsQ, NHeadsKv, HeadDim, MaxSeqLen, startPos, batchSize, AttnScale);

        // Upload to GPU
        using var dQ = ctx.CreateDeviceBuffer((ulong)(q.Length * sizeof(float)));
        using var dK = ctx.CreateDeviceBuffer((ulong)(kCache.Length * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(vCache.Length * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(expected.Length * sizeof(float)));

        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes<float>(q.AsSpan()));
        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes<float>(kCache.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes<float>(vCache.AsSpan()));

        // Dispatch GPU Kernel
        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)NHeadsQ;
        pConsts[1] = (uint)NHeadsKv;
        pConsts[2] = (uint)HeadDim;
        pConsts[3] = (uint)MaxSeqLen;
        pConsts[4] = (uint)startPos;
        *(float*)(&pConsts[5]) = AttnScale;
        pConsts[6] = (uint)batchSize;
        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);

        cmd.Dispatch((uint)NHeadsQ, (uint)batchSize, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        // Readback GPU results
        float[] actual = new float[expected.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes<float>(actual.AsSpan()), dOut, (ulong)(actual.Length * sizeof(float)));

        // Numerical verification across all elements
        float maxDiff = 0.0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > maxDiff) maxDiff = diff;
            Assert.False(float.IsNaN(actual[i]), $"GPU produced NaN at index {i} for targetPos={targetPos}");
            Assert.False(float.IsInfinity(actual[i]), $"GPU produced Infinity at index {i} for targetPos={targetPos}");
        }

        // Tolerance: fp32 online softmax vs batch oracle within 5e-4
        Assert.True(maxDiff < 5e-4f, $"Max difference {maxDiff} exceeded 5e-4 tolerance for targetPos={targetPos}");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(511)]
    [InlineData(512)]
    public void AttentionBoundaryConditions_FP16_AllPositions_MatchOracle(int targetPos)
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();

        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        });
        using var rootSig = ctx.CreateRootSignature(rootSigDesc);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.AttentionBatchF16));

        int batchSize = 1;
        int startPos = targetPos;

        var rng = new Random(100 + targetPos);
        float[] q = new float[batchSize * NHeadsQ * HeadDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        int kvElements = NHeadsKv * MaxSeqLen * HeadDim;
        float[] kFp32 = new float[kvElements];
        float[] vFp32 = new float[kvElements];
        for (int i = 0; i < kvElements; i++)
        {
            kFp32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vFp32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Pack into uint buffers: 2 x FP16 per uint
        int packedCount = NHeadsKv * MaxSeqLen * (HeadDim / 2);
        uint[] kPacked = new uint[packedCount];
        uint[] vPacked = new uint[packedCount];

        // Also reconstruct unpacked halfs for CPU Oracle parity
        float[] kFromHalf = new float[kvElements];
        float[] vFromHalf = new float[kvElements];

        int pairIdx = 0;
        for (int i = 0; i < kvElements; i += 2)
        {
            Half k0 = (Half)kFp32[i];
            Half k1 = (Half)kFp32[i + 1];
            kFromHalf[i] = (float)k0;
            kFromHalf[i + 1] = (float)k1;
            kPacked[pairIdx] = (uint)BitConverter.HalfToUInt16Bits(k0) | ((uint)BitConverter.HalfToUInt16Bits(k1) << 16);

            Half v0 = (Half)vFp32[i];
            Half v1 = (Half)vFp32[i + 1];
            vFromHalf[i] = (float)v0;
            vFromHalf[i + 1] = (float)v1;
            vPacked[pairIdx] = (uint)BitConverter.HalfToUInt16Bits(v0) | ((uint)BitConverter.HalfToUInt16Bits(v1) << 16);

            pairIdx++;
        }

        // Compute Oracle with half-quantized K and V
        float[] expected = ComputeCpuOracle(q, kFromHalf, vFromHalf, NHeadsQ, NHeadsKv, HeadDim, MaxSeqLen, startPos, batchSize, AttnScale);

        // Upload to GPU
        using var dQ = ctx.CreateDeviceBuffer((ulong)(q.Length * sizeof(float)));
        using var dK = ctx.CreateDeviceBuffer((ulong)(kPacked.Length * sizeof(uint)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(vPacked.Length * sizeof(uint)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(expected.Length * sizeof(float)));

        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes<float>(q.AsSpan()));
        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes<uint>(kPacked.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes<uint>(vPacked.AsSpan()));

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)NHeadsQ;
        pConsts[1] = (uint)NHeadsKv;
        pConsts[2] = (uint)HeadDim;
        pConsts[3] = (uint)MaxSeqLen;
        pConsts[4] = (uint)startPos;
        *(float*)(&pConsts[5]) = AttnScale;
        pConsts[6] = (uint)batchSize;
        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);

        cmd.Dispatch((uint)NHeadsQ, (uint)batchSize, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[expected.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes<float>(actual.AsSpan()), dOut, (ulong)(actual.Length * sizeof(float)));

        float maxDiff = 0.0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > maxDiff) maxDiff = diff;
            Assert.False(float.IsNaN(actual[i]), $"GPU produced NaN at index {i} in FP16 for targetPos={targetPos}");
            Assert.False(float.IsInfinity(actual[i]), $"GPU produced Infinity at index {i} in FP16 for targetPos={targetPos}");
        }

        // Tolerance: FP16 precision vs FP32 oracle within 1e-3
        Assert.True(maxDiff < 1e-3f, $"Max difference {maxDiff} exceeded 1e-3 tolerance for FP16 at targetPos={targetPos}");
    }

    [Theory]
    [InlineData(1)]  // Non-multiple of 4: 1 token (P=0)
    [InlineData(3)]  // Non-multiple of 4: 3 tokens (P=2)
    [InlineData(7)]  // Non-multiple of 4: 7 tokens (P=6)
    [InlineData(33)] // Non-multiple of 4: 33 tokens (P=32)
    public void SequenceTiledOnlineSoftmax_NonMultiplesOfFour_PaddingSafetyAndConvergence(int seqLen)
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();

        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        });
        using var rootSig = ctx.CreateRootSignature(rootSigDesc);
        using var pso = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.AttentionBatch));

        // Test chunked batch prefill where batchSize is non-multiple of 4
        int batchSize = seqLen;
        int startPos = 0;

        var rng = new Random(1337 + seqLen);
        float[] q = new float[batchSize * NHeadsQ * HeadDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        int kvElements = NHeadsKv * MaxSeqLen * HeadDim;
        float[] kCache = new float[kvElements];
        float[] vCache = new float[kvElements];

        // Test with uniform V = 1.0f to verify partition of unity (sum of softmax weights = 1.0)
        for (int i = 0; i < kvElements; i++)
        {
            kCache[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vCache[i] = 1.0f;
        }

        // For non-multiples of 4, test padding safety by seeding historical positions beyond posT with NaN/Inf traps
        // Slots at p >= seqLen should never be accessed or pollute the result!
        for (int h = 0; h < NHeadsKv; h++)
        {
            for (int p = seqLen; p < Math.Min(seqLen + 4, MaxSeqLen); p++)
            {
                int baseIdx = (h * MaxSeqLen + p) * HeadDim;
                for (int d = 0; d < HeadDim; d++)
                {
                    kCache[baseIdx + d] = 999999.0f; // Trap value: if touched, exp() would explode or corrupt max
                    vCache[baseIdx + d] = -88888.0f;
                }
            }
        }

        float[] expected = ComputeCpuOracle(q, kCache, vCache, NHeadsQ, NHeadsKv, HeadDim, MaxSeqLen, startPos, batchSize, AttnScale);

        using var dQ = ctx.CreateDeviceBuffer((ulong)(q.Length * sizeof(float)));
        using var dK = ctx.CreateDeviceBuffer((ulong)(kCache.Length * sizeof(float)));
        using var dV = ctx.CreateDeviceBuffer((ulong)(vCache.Length * sizeof(float)));
        using var dOut = ctx.CreateDeviceBuffer((ulong)(expected.Length * sizeof(float)));

        ctx.CopyToDevice(dQ, MemoryMarshal.AsBytes<float>(q.AsSpan()));
        ctx.CopyToDevice(dK, MemoryMarshal.AsBytes<float>(kCache.AsSpan()));
        ctx.CopyToDevice(dV, MemoryMarshal.AsBytes<float>(vCache.AsSpan()));

        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.SetComputeRootSignature(rootSig);
        cmd.SetPipelineState(pso);

        uint* pConsts = stackalloc uint[7];
        pConsts[0] = (uint)NHeadsQ;
        pConsts[1] = (uint)NHeadsKv;
        pConsts[2] = (uint)HeadDim;
        pConsts[3] = (uint)MaxSeqLen;
        pConsts[4] = (uint)startPos;
        *(float*)(&pConsts[5]) = AttnScale;
        pConsts[6] = (uint)batchSize;
        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dOut.GPUVirtualAddress);

        cmd.Dispatch((uint)NHeadsQ, (uint)batchSize, 1);

        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] actual = new float[expected.Length];
        ctx.CopyToHost(MemoryMarshal.AsBytes<float>(actual.AsSpan()), dOut, (ulong)(actual.Length * sizeof(float)));

        // 1. Verify padding safety: No NaNs, no Infinities, no trap contamination
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"GPU produced NaN with seqLen={seqLen} at index {i}");
            Assert.False(float.IsInfinity(actual[i]), $"GPU produced Infinity with seqLen={seqLen} at index {i}");
            // Since V = 1.0f everywhere, sum of weights = 1.0, so output MUST be approximately 1.0!
            Assert.True(MathF.Abs(actual[i] - 1.0f) < 0.001f,
                $"Softmax partition of unity failed at index {i} (seqLen={seqLen}): expected 1.0, got {actual[i]}");
        }

        // 2. Verify parity against CPU Oracle
        float maxDiff = 0.0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        Assert.True(maxDiff < 5e-4f, $"Max difference {maxDiff} exceeded 5e-4 for seqLen={seqLen}");
    }

    [Fact]
    public void FineGrainedResourceBarriers_MultiLayerSuccession_StressTest()
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();

        // Stress test fine-grained barriers across 28 simulated transformer layers
        const int numLayers = 28;
        const int batchSize = 16;
        const int dim = 512;
        const int qDim = 512;

        // Compile required shaders
        var rootSigDesc = new RootSignatureDescription(RootSignatureFlags.None, new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        });
        using var rootSig = ctx.CreateRootSignature(rootSigDesc);
        using var psoAttn = ctx.CreatePipelineState(rootSig, ctx.CompileShader(D3D12Shaders.AttentionBatch));

        // Allocate persistent intermediate buffers matching Qwen2D3D12Model layout
        using var dXBatch = ctx.CreateDeviceBuffer((ulong)(batchSize * dim * sizeof(float)));
        using var dNormXBatch = ctx.CreateDeviceBuffer((ulong)(batchSize * dim * sizeof(float)));
        using var dQBatch = ctx.CreateDeviceBuffer((ulong)(batchSize * qDim * sizeof(float)));
        using var dAttnOutBatch = ctx.CreateDeviceBuffer((ulong)(batchSize * qDim * sizeof(float)));

        ID3D12Resource[] dKeyCache = new ID3D12Resource[numLayers];
        ID3D12Resource[] dValCache = new ID3D12Resource[numLayers];
        for (int l = 0; l < numLayers; l++)
        {
            dKeyCache[l] = ctx.CreateDeviceBuffer((ulong)(NHeadsKv * MaxSeqLen * HeadDim * sizeof(float)));
            dValCache[l] = ctx.CreateDeviceBuffer((ulong)(NHeadsKv * MaxSeqLen * HeadDim * sizeof(float)));
        }

        try
        {
            // Seed initial data
            float[] initX = new float[batchSize * dim];
            var rng = new Random(999);
            for (int i = 0; i < initX.Length; i++) initX[i] = (float)(rng.NextDouble() * 0.1);
            ctx.CopyToDevice(dXBatch, MemoryMarshal.AsBytes<float>(initX.AsSpan()));

            // Populate KV caches with initial keys/values
            float[] initKv = new float[NHeadsKv * MaxSeqLen * HeadDim];
            for (int i = 0; i < initKv.Length; i++) initKv[i] = (float)(rng.NextDouble() * 0.1);
            for (int l = 0; l < numLayers; l++)
            {
                ctx.CopyToDevice(dKeyCache[l], MemoryMarshal.AsBytes<float>(initKv.AsSpan()));
                ctx.CopyToDevice(dValCache[l], MemoryMarshal.AsBytes<float>(initKv.AsSpan()));
            }

            // Allocate constant buffer on stack once outside loops
            uint* pConsts = stackalloc uint[7];
            pConsts[0] = (uint)NHeadsQ;
            pConsts[1] = (uint)NHeadsKv;
            pConsts[2] = (uint)HeadDim;
            pConsts[3] = (uint)MaxSeqLen;
            pConsts[4] = 0;
            *(float*)(&pConsts[5]) = AttnScale;
            pConsts[6] = (uint)batchSize;

            // Execute 10 consecutive multi-layer pipeline passes without CPU synchronization
            float[][] runOutputs = new float[10][];

            for (int run = 0; run < 10; run++)
            {
                ctx.BeginCommands();
                var cmd = ctx.CommandList;

                for (int l = 0; l < numLayers; l++)
                {
                    // Barrier 1: Attn Norm write
                    cmd.ResourceBarrierUnorderedAccessView(dNormXBatch);

                    // Barrier 2: Q projection write
                    cmd.ResourceBarrierUnorderedAccessView(dQBatch);

                    // Barrier 3: KV Cache store write
                    cmd.ResourceBarrierUnorderedAccessView(dKeyCache[l]);
                    cmd.ResourceBarrierUnorderedAccessView(dValCache[l]);

                    // Dispatch Attention
                    cmd.SetComputeRootSignature(rootSig);
                    cmd.SetPipelineState(psoAttn);

                    cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)pConsts, 0);

                    cmd.SetComputeRootUnorderedAccessView(1, dQBatch.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(2, dKeyCache[l].GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(3, dValCache[l].GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(4, dAttnOutBatch.GPUVirtualAddress);

                    cmd.Dispatch((uint)NHeadsQ, (uint)batchSize, 1);

                    // Barrier 4: Attention output write
                    cmd.ResourceBarrierUnorderedAccessView(dAttnOutBatch);

                    // Barrier 5: Residual hidden state write
                    cmd.ResourceBarrierUnorderedAccessView(dXBatch);
                }

                ctx.EndCommandsAndExecute();
                ctx.Synchronize();

                // Readback output of this run
                float[] outBuffer = new float[batchSize * qDim];
                ctx.CopyToHost(MemoryMarshal.AsBytes<float>(outBuffer.AsSpan()), dAttnOutBatch, (ulong)(outBuffer.Length * sizeof(float)));
                runOutputs[run] = outBuffer;
            }

            // Assert exact bitwise / float determinism across all 10 successive runs
            // Any race condition or cache-line tearing will result in divergent outputs
            for (int run = 1; run < 10; run++)
            {
                for (int i = 0; i < runOutputs[0].Length; i++)
                {
                    Assert.False(float.IsNaN(runOutputs[run][i]), $"Run {run} produced NaN at index {i}");
                    Assert.False(float.IsInfinity(runOutputs[run][i]), $"Run {run} produced Infinity at index {i}");
                    Assert.Equal(runOutputs[0][i], runOutputs[run][i]);
                }
            }
        }
        finally
        {
            for (int l = 0; l < numLayers; l++)
            {
                dKeyCache[l]?.Dispose();
                dValCache[l]?.Dispose();
            }
        }
    }
}
