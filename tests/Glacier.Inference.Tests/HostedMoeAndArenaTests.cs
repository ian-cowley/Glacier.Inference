namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;
using Xunit;

public sealed unsafe class HostRamExpertArenaTests : IDisposable
{
    private readonly string _tempDir;

    public HostRamExpertArenaTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierHostArena_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
        }
    }

    [Fact]
    public void HostRamExpertArena_UnbufferedSplitExperts_ExactBytesAndCappedBudget()
    {
        // Build a 2-layer MoE GGUF whose per-tensor expert byte size exceeds 4 MiB so
        // Windows exercises FILE_FLAG_NO_BUFFERING alignment and EOF tail fallback.
        string path = Path.Combine(_tempDir, "unbuffered_moe.gguf");
        BuildLargeSplitMoeGguf(path, numLayers: 2, dim: 1024, expFfnDim: 256, numExperts: 8);

        using var ggufRef = GgufFile.Open(path);
        using var weightsRef = new ModelWeights(ggufRef);

        // Full residency budget
        using (var gguf = GgufFile.Open(path))
        using (var weights = new ModelWeights(gguf))
        {
            IntPtr origL0Gate = (IntPtr)weights.Layers[0].FfnGateExpsWeight;
            IntPtr origL1Down = (IntPtr)weights.Layers[1].FfnDownExpsWeight;

            ulong layer0Bytes =
                gguf.Tensors["blk.0.ffn_gate_exps.weight"].GetByteSize() +
                gguf.Tensors["blk.0.ffn_up_exps.weight"].GetByteSize() +
                gguf.Tensors["blk.0.ffn_down_exps.weight"].GetByteSize();

            using var arena = HostRamExpertArena.TryAttach(weights, budgetBytes: layer0Bytes * 3UL, preferUnbufferedIo: true);
            Assert.NotNull(arena);
            Assert.Equal(2, arena!.TotalMoeLayers);
            Assert.Equal(2, arena.ResidentMoeLayers);
            Assert.True(arena.AllocatedBytes >= layer0Bytes * 2UL);

            if (OperatingSystem.IsWindows())
            {
                Assert.True(arena.UsedUnbufferedIo);
            }

            // Pointers must be repointed into owned DRAM slabs (32-byte aligned after unbuffered page-prefix adjustment)
            Assert.NotEqual(origL0Gate, (IntPtr)weights.Layers[0].FfnGateExpsWeight);
            Assert.NotEqual(origL1Down, (IntPtr)weights.Layers[1].FfnDownExpsWeight);
            Assert.Equal(0L, ((long)(IntPtr)weights.Layers[0].FfnGateExpsWeight) % 32L);

            for (int l = 0; l < 2; l++)
            {
                int gateLen = checked((int)ggufRef.Tensors[$"blk.{l}.ffn_gate_exps.weight"].GetByteSize());
                int upLen = checked((int)ggufRef.Tensors[$"blk.{l}.ffn_up_exps.weight"].GetByteSize());
                int downLen = checked((int)ggufRef.Tensors[$"blk.{l}.ffn_down_exps.weight"].GetByteSize());

                Assert.True(new ReadOnlySpan<byte>(weightsRef.Layers[l].FfnGateExpsWeight, gateLen)
                    .SequenceEqual(new ReadOnlySpan<byte>(weights.Layers[l].FfnGateExpsWeight, gateLen)));
                Assert.True(new ReadOnlySpan<byte>(weightsRef.Layers[l].FfnUpExpsWeight, upLen)
                    .SequenceEqual(new ReadOnlySpan<byte>(weights.Layers[l].FfnUpExpsWeight, upLen)));
                Assert.True(new ReadOnlySpan<byte>(weightsRef.Layers[l].FfnDownExpsWeight, downLen)
                    .SequenceEqual(new ReadOnlySpan<byte>(weights.Layers[l].FfnDownExpsWeight, downLen)));
            }
        }

        // Capped budget fitting only 1 of 2 MoE layers
        using (var ggufCapped = GgufFile.Open(path))
        using (var weightsCapped = new ModelWeights(ggufCapped))
        {
            ulong layer0Bytes =
                ggufCapped.Tensors["blk.0.ffn_gate_exps.weight"].GetByteSize() +
                ggufCapped.Tensors["blk.0.ffn_up_exps.weight"].GetByteSize() +
                ggufCapped.Tensors["blk.0.ffn_down_exps.weight"].GetByteSize();

            IntPtr origL1Gate = (IntPtr)weightsCapped.Layers[1].FfnGateExpsWeight;
            using var arena = HostRamExpertArena.TryAttach(weightsCapped, budgetBytes: layer0Bytes + (layer0Bytes / 2UL), preferUnbufferedIo: true);
            Assert.NotNull(arena);
            Assert.Equal(2, arena!.TotalMoeLayers);
            Assert.Equal(1, arena.ResidentMoeLayers);
            Assert.True(weightsCapped.Layers[0].FfnGateExpsWeight != null);
            Assert.Equal(origL1Gate, (IntPtr)weightsCapped.Layers[1].FfnGateExpsWeight);
        }
    }

    [Fact]
    public void InferenceSession_PleIoAuto_And_HostArenaBudget_ConfigureCorrectly()
    {
        string path = Path.Combine(_tempDir, "session_auto.gguf");
        HostedTestGgufBuilder.BuildRealLayoutQwen4ExpGguf(path);

        var disabledSnapBefore = HardwareRuntimeProfiler.CaptureSnapshot();
        using (var disabled = new InferenceSession(path, maxSeqLen: 64, device: "cpu", pleIoMode: PleIoMode.Auto, hostArenaBudgetBytes: 0))
        {
            var disabledSnapAfter = HardwareRuntimeProfiler.CaptureSnapshot();
            Assert.Null(disabled.ExpertArena);
            Assert.NotNull(disabled.PleTable);
            PleIoMode actualMode = disabled.PleTable!.IoMode;
            Assert.True(actualMode is PleIoMode.MemoryMapped or PleIoMode.DirectIo);

            PleIoMode expectedBefore = HardwareRuntimeProfiler.RecommendPolicy(disabledSnapBefore, disabled.Weights).RecommendedPleIoMode;
            PleIoMode expectedAfter = HardwareRuntimeProfiler.RecommendPolicy(disabledSnapAfter, disabled.Weights).RecommendedPleIoMode;
            Assert.True(actualMode == expectedBefore || actualMode == expectedAfter,
                $"PleIoMode.Auto resolved to {actualMode}, expected {expectedBefore} (before) or {expectedAfter} (after).");
        }

        var autoSnapBefore = HardwareRuntimeProfiler.CaptureSnapshot();
        using (var autoSession = new InferenceSession(path, maxSeqLen: 64, device: "cpu", pleIoMode: PleIoMode.Auto))
        {
            var autoSnapAfter = HardwareRuntimeProfiler.CaptureSnapshot();
            Assert.NotNull(autoSession.ExpertArena);
            Assert.Equal(autoSession.ExpertArena!.TotalMoeLayers, autoSession.ExpertArena.ResidentMoeLayers);
            Assert.NotNull(autoSession.PleTable);
            PleIoMode actualMode = autoSession.PleTable!.IoMode;
            Assert.True(actualMode is PleIoMode.MemoryMapped or PleIoMode.DirectIo);

            PleIoMode expectedBefore = HardwareRuntimeProfiler.RecommendPolicy(autoSnapBefore, autoSession.Weights).RecommendedPleIoMode;
            PleIoMode expectedAfter = HardwareRuntimeProfiler.RecommendPolicy(autoSnapAfter, autoSession.Weights).RecommendedPleIoMode;
            Assert.True(actualMode == expectedBefore || actualMode == expectedAfter,
                $"PleIoMode.Auto resolved to {actualMode}, expected {expectedBefore} (before) or {expectedAfter} (after).");
        }

        using (var mmapSession = new InferenceSession(path, maxSeqLen: 64, device: "cpu", pleIoMode: PleIoMode.MemoryMapped))
        {
            Assert.NotNull(mmapSession.PleTable);
            Assert.Equal(PleIoMode.MemoryMapped, mmapSession.PleTable!.IoMode);
        }

        using (var directSession = new InferenceSession(path, maxSeqLen: 64, device: "cpu", pleIoMode: PleIoMode.DirectIo))
        {
            Assert.NotNull(directSession.PleTable);
            Assert.Equal(PleIoMode.DirectIo, directSession.PleTable!.IoMode);
        }
    }

    private static void BuildLargeSplitMoeGguf(string path, int numLayers, int dim, int expFfnDim, int numExperts)
    {
        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen3moe"),
            ("general.alignment", 4U, 32U),
            ("qwen3moe.block_count", 4U, (uint)numLayers),
            ("qwen3moe.embedding_length", 4U, (uint)dim),
            ("qwen3moe.feed_forward_length", 4U, (uint)expFfnDim),
            ("qwen3moe.context_length", 4U, 128U),
            ("qwen3moe.attention.head_count", 4U, 4U),
            ("qwen3moe.attention.head_count_kv", 4U, 2U),
            ("qwen3moe.attention.key_length", 4U, 64U),
            ("qwen3moe.attention.value_length", 4U, 64U),
            ("qwen3moe.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen3moe.rope.dimension_count", 4U, 64U),
            ("qwen3moe.expert_count", 4U, (uint)numExperts),
            ("qwen3moe.expert_used_count", 4U, 2U),
            ("qwen3moe.expert_feed_forward_length", 4U, (uint)expFfnDim),
            ("tokenizer.ggml.tokens", 9U, (8U, new object[] { "<unk>", "a", "b", "c" })),
            ("tokenizer.ggml.bos_token_id", 4U, 0U),
            ("tokenizer.ggml.eos_token_id", 4U, 3U)
        };

        var tensors = new List<HostedTestGgufBuilder.TensorDef>();
        byte[] normData = new byte[dim * 4];
        MemoryMarshal.Cast<byte, float>(normData.AsSpan()).Fill(1.0f);

        tensors.Add(new("token_embd.weight", [(ulong)dim, 4UL], GgufType.F32, new byte[dim * 4 * 4]));
        tensors.Add(new("output_norm.weight", [(ulong)dim], GgufType.F32, normData));
        tensors.Add(new("output.weight", [(ulong)dim, 4UL], GgufType.F32, new byte[dim * 4 * 4]));

        byte[] FillPattern(int bytes, byte seed)
        {
            byte[] arr = GC.AllocateUninitializedArray<byte>(bytes);
            for (int i = 0; i < arr.Length; i++)
                arr[i] = (byte)(seed + (i * 131 + (i >> 8) * 17));
            return arr;
        }

        // Q8_0 (34 bytes per 32 weights) => 1024 x 256 x 8 => 8 * 256 * 32 * 34 = 2.18 MB;
        // use 16 experts or F32 small-attn + F32 exps -> F32 1024 x 256 x 8 = 8,388,608 bytes (8 MiB > 4 MiB threshold!)
        int qDim = 4 * 64;
        int kvDim = 2 * 64;
        for (int l = 0; l < numLayers; l++)
        {
            tensors.Add(new($"blk.{l}.attn_norm.weight", [(ulong)dim], GgufType.F32, normData));
            tensors.Add(new($"blk.{l}.attn_q.weight", [(ulong)dim, (ulong)qDim], GgufType.F32, new byte[dim * qDim * 4]));
            tensors.Add(new($"blk.{l}.attn_k.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, new byte[dim * kvDim * 4]));
            tensors.Add(new($"blk.{l}.attn_v.weight", [(ulong)dim, (ulong)kvDim], GgufType.F32, new byte[dim * kvDim * 4]));
            tensors.Add(new($"blk.{l}.attn_output.weight", [(ulong)qDim, (ulong)dim], GgufType.F32, new byte[qDim * dim * 4]));
            tensors.Add(new($"blk.{l}.ffn_norm.weight", [(ulong)dim], GgufType.F32, normData));
            tensors.Add(new($"blk.{l}.ffn_gate_inp.weight", [(ulong)dim, (ulong)numExperts], GgufType.F32, new byte[dim * numExperts * 4]));

            int expsBytes = dim * expFfnDim * numExperts * 4; // 8 MiB per tensor
            tensors.Add(new($"blk.{l}.ffn_gate_exps.weight", [(ulong)dim, (ulong)expFfnDim, (ulong)numExperts], GgufType.F32, FillPattern(expsBytes, (byte)(l * 13 + 1))));
            tensors.Add(new($"blk.{l}.ffn_up_exps.weight", [(ulong)dim, (ulong)expFfnDim, (ulong)numExperts], GgufType.F32, FillPattern(expsBytes, (byte)(l * 13 + 5))));
            tensors.Add(new($"blk.{l}.ffn_down_exps.weight", [(ulong)expFfnDim, (ulong)dim, (ulong)numExperts], GgufType.F32, FillPattern(expsBytes, (byte)(l * 13 + 9))));
        }

        HostedTestGgufBuilder.Write(path, kv, tensors);
    }
}

[Trait("Hardware", "Gpu")]
[Collection("SequentialGpu")]
public sealed unsafe class D3D12HostedMoeAndNewQuantsTests : IDisposable
{
    private readonly string _tempDir;

    public D3D12HostedMoeAndNewQuantsTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierD3D12Qwen4Exp_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
        }
    }

    [Theory]
    [InlineData(GgufType.Q4_0, 256, 16)]
    [InlineData(GgufType.Q5_0, 256, 16)]
    [InlineData(GgufType.IQ4_XS, 256, 16)]
    [InlineData(GgufType.MXFP4, 256, 16)]
    [InlineData(GgufType.BF16, 256, 16)]
    public void D3D12_NewQuantShaders_Gemv_And_BatchedGemm_MatchCpu(GgufType type, int kCols, int mRows)
    {
        if (!OperatingSystem.IsWindows()) return;

        using var ctx = new D3D12Context();
        byte[] rawWeights = BuildSyntheticQuantMatrix(type, kCols, mRows, seed: 42);

        // 1. Test Single-Vector GEMV
        float[] x = new float[kCols];
        for (int i = 0; i < kCols; i++) x[i] = MathF.Sin((i + 1) * 0.13f) * 0.7f;

        float[] cpuY = new float[mRows];
        fixed (byte* pW = rawWeights)
        fixed (float* pX = x, pY = cpuY)
        {
            QuantKernels.MatVecMul(type, pW, pX, pY, kCols, mRows);
        }

        string gemvHlsl = type switch
        {
            GgufType.Q4_0 => D3D12Shaders.GemvQ4_0,
            GgufType.Q5_0 => D3D12Shaders.GemvQ5_0,
            GgufType.IQ4_XS => D3D12Shaders.GemvIQ4_XS,
            GgufType.MXFP4 => D3D12Shaders.GemvMXFP4,
            GgufType.BF16 => D3D12Shaders.GemvBf16,
            _ => throw new NotSupportedException()
        };

        var sigGemv = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None,
        [
            new(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        ]));
        var psoGemv = ctx.CreatePipelineState(sigGemv, ctx.CompileShader(gemvHlsl));

        ID3D12Resource dW;
        fixed (byte* pW = rawWeights)
        {
            dW = type == GgufType.BF16
                ? D3D12TensorAlign.UploadCompactBf16Tensor(ctx, (IntPtr)pW, mRows, kCols)
                : D3D12TensorAlign.UploadAlignedTensor(ctx, type, (IntPtr)pW, mRows, kCols);
        }

        using (dW)
        using (var dX = ctx.CreateDeviceBuffer((ulong)(kCols * sizeof(float))))
        using (var dY = ctx.CreateDeviceBuffer((ulong)(mRows * sizeof(float))))
        using (sigGemv)
        using (psoGemv)
        {
            ctx.CopyToDevice(dX, MemoryMarshal.AsBytes(x.AsSpan()));
            ctx.BeginCommands();
            var cmd = ctx.CommandList;
            cmd.SetComputeRootSignature(sigGemv);
            cmd.SetPipelineState(psoGemv);
            uint* c = stackalloc uint[6];
            c[0] = (uint)kCols;
            c[1] = (uint)mRows;
            c[2] = 0; c[3] = 0; c[4] = 1; c[5] = 0;
            cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
            cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
            cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);
            cmd.Dispatch((uint)((mRows + 3) / 4), 1, 1);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            float[] gpuY = new float[mRows];
            ctx.CopyToHost(MemoryMarshal.AsBytes(gpuY.AsSpan()), dY, (ulong)(mRows * sizeof(float)));
            for (int r = 0; r < mRows; r++)
            {
                Assert.True(MathF.Abs(cpuY[r] - gpuY[r]) < 1e-2f,
                    $"{type} GEMV mismatch at row {r}: cpu={cpuY[r]}, gpu={gpuY[r]}");
            }

            // 2. Test Batched GEMM at batchSize=5 and batchSize=37 (crossing 32-token tile boundary)
            string? batchHlsl = type switch
            {
                GgufType.Q4_0 => D3D12Shaders.GemmQ4_0Batch,
                GgufType.Q5_0 => D3D12Shaders.GemmQ5_0Batch,
                GgufType.IQ4_XS => D3D12Shaders.GemmIQ4_XSBatch,
                GgufType.BF16 => D3D12Shaders.GemmBf16Batch,
                _ => null
            };
            if (batchHlsl == null) return;

            var sigBatch = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None,
            [
                new(new RootConstants(0, 0, 6), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
            ]));
            var psoBatch = ctx.CreatePipelineState(sigBatch, ctx.CompileShader(batchHlsl));

            using (sigBatch)
            using (psoBatch)
            {
                foreach (int batchSize in new[] { 5, 37 })
                {
                    float[] xBatch = new float[batchSize * kCols];
                    for (int t = 0; t < batchSize; t++)
                        for (int i = 0; i < kCols; i++)
                            xBatch[t * kCols + i] = MathF.Cos((t * 17 + i) * 0.09f) * 0.6f;

                    float[] cpuBatch = new float[batchSize * mRows];
                    fixed (byte* pW = rawWeights)
                    fixed (float* pXb = xBatch, pYb = cpuBatch)
                    {
                        for (int t = 0; t < batchSize; t++)
                            QuantKernels.MatVecMul(type, pW, pXb + t * kCols, pYb + t * mRows, kCols, mRows);
                    }

                    using var dXb = ctx.CreateDeviceBuffer((ulong)(batchSize * kCols * sizeof(float)));
                    using var dYb = ctx.CreateDeviceBuffer((ulong)(batchSize * mRows * sizeof(float)));
                    ctx.CopyToDevice(dXb, MemoryMarshal.AsBytes(xBatch.AsSpan()));

                    ctx.BeginCommands();
                    var cmdB = ctx.CommandList;
                    cmdB.SetComputeRootSignature(sigBatch);
                    cmdB.SetPipelineState(psoBatch);
                    c[0] = (uint)kCols;
                    c[1] = (uint)mRows;
                    c[2] = (uint)batchSize;
                    c[3] = 0; c[4] = 0; c[5] = 1;
                    cmdB.SetComputeRoot32BitConstants(0, 6, (IntPtr)c, 0);
                    cmdB.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
                    cmdB.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
                    cmdB.SetComputeRootUnorderedAccessView(3, dXb.GPUVirtualAddress);
                    cmdB.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
                    cmdB.SetComputeRootUnorderedAccessView(5, dYb.GPUVirtualAddress);
                    cmdB.Dispatch((uint)((mRows + 3) / 4), (uint)((batchSize + 31) / 32), 1);
                    ctx.EndCommandsAndExecute();
                    ctx.Synchronize();

                    float[] gpuBatch = new float[batchSize * mRows];
                    ctx.CopyToHost(MemoryMarshal.AsBytes(gpuBatch.AsSpan()), dYb, (ulong)(batchSize * mRows * sizeof(float)));

                    for (int i = 0; i < gpuBatch.Length; i++)
                    {
                        Assert.True(MathF.Abs(cpuBatch[i] - gpuBatch[i]) < 2e-2f,
                            $"{type} Batch={batchSize} GEMM mismatch at idx {i}: cpu={cpuBatch[i]}, gpu={gpuBatch[i]}");
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData(false)] // VRAM MoE
    [InlineData(true)]  // Hosted MoE (slot pool + DDR4 HostRamExpertArena)
    public void D3D12_Qwen4Exp_VramAndHostedMoe_SequentialAndBatched_MatchCpu(bool forceTieredMoe)
    {
        if (!OperatingSystem.IsWindows()) return;

        string modelPath = Path.Combine(_tempDir, $"qwen4exp_gpu_{forceTieredMoe}.gguf");
        HostedTestGgufBuilder.BuildRealLayoutQwen4ExpGguf(modelPath);

        // 9 tokens exercises QSA topK=4 sparse selection for pos >= 4 (calls _psoSparseAttn)
        int[] prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        int nextToken = 10;

        using var ggufCpu = GgufFile.Open(modelPath);
        using var weightsCpu = new ModelWeights(ggufCpu);
        using var cpuModel = new Qwen3HybridModel(weightsCpu, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
        using var kvCpu = new KVCache(weightsCpu.BlockCount, weightsCpu.HeadCountKv, weightsCpu.HeadDim, 64, weightsCpu.ValueDim);

        using var ggufGpu = GgufFile.Open(modelPath);
        using var weightsGpu = new ModelWeights(ggufGpu);
        if (forceTieredMoe)
        {
            weightsGpu.AttachExpertArena(64UL * 1024 * 1024, preferUnbufferedIo: true);
        }

        using var d3dCtx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(d3dCtx, weightsGpu, maxSeqLen: 64, tieredMoe: forceTieredMoe, pleIoMode: PleIoMode.Auto);

        Assert.Equal(forceTieredMoe, gpuModel.TieredMoe);
        if (forceTieredMoe)
        {
            Assert.True(gpuModel.ActiveHostedMoeSlots > 0);
        }

        float[] cpuLogits = new float[weightsCpu.VocabSize];
        float[] gpuLogits = new float[weightsGpu.VocabSize];

        // 1. Sequential decode parity (pos 0..8 crossing QSA topK=4)
        for (int pos = 0; pos < prompt.Length; pos++)
        {
            cpuModel.Forward(prompt[pos], pos, kvCpu, cpuLogits, computeLogits: true);
            gpuModel.Forward(prompt[pos], pos, gpuLogits, computeLogits: true);

            AssertLogitsClose(cpuLogits, gpuLogits, tol: 5e-3f, $"seq pos={pos}, tiered={forceTieredMoe}");
            Assert.Equal(ArgMax(cpuLogits), ArgMax(gpuLogits));
        }

        if (forceTieredMoe)
        {
            Assert.True(gpuModel.GpuExpertHitCount + gpuModel.GpuExpertTransferCount > 0);
        }

        // 2. Batched prefill (9 tokens) + follow-up decode parity
        cpuModel.ResetState();
        kvCpu.Reset();
        gpuModel.ResetKvCache();

        cpuModel.ForwardBatch(prompt, 0, cpuLogits, kvCpu, computeLogits: true);
        gpuModel.ForwardBatch(prompt, 0, gpuLogits, computeLogits: true);
        AssertLogitsClose(cpuLogits, gpuLogits, tol: 5e-3f, $"batch prefill, tiered={forceTieredMoe}");
        Assert.Equal(ArgMax(cpuLogits), ArgMax(gpuLogits));

        cpuModel.Forward(nextToken, prompt.Length, kvCpu, cpuLogits, computeLogits: true);
        gpuModel.Forward(nextToken, prompt.Length, gpuLogits, computeLogits: true);
        AssertLogitsClose(cpuLogits, gpuLogits, tol: 5e-3f, $"post-batch decode, tiered={forceTieredMoe}");
        Assert.Equal(ArgMax(cpuLogits), ArgMax(gpuLogits));
    }

    [Fact]
    public void D3D12_Qwen4Exp_RecurrentStateSnapshot_And_VerifyBatchLogits_MatchSequential()
    {
        if (!OperatingSystem.IsWindows()) return;

        string modelPath = Path.Combine(_tempDir, "qwen4exp_snapshot.gguf");
        HostedTestGgufBuilder.BuildRealLayoutQwen4ExpGguf(modelPath);

        using var gguf = GgufFile.Open(modelPath);
        using var weights = new ModelWeights(gguf);
        using var d3dCtx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(d3dCtx, weights, maxSeqLen: 64, tieredMoe: true, pleIoMode: PleIoMode.MemoryMapped);

        Assert.True(gpuModel.StateSnapshotBytes > 0);

        float[] logitsA = new float[weights.VocabSize];
        float[] logitsB = new float[weights.VocabSize];

        // Prefill prefix tokens 1, 2
        gpuModel.ForwardBatch([1, 2], 0, logitsA, computeLogits: true);
        byte[] prefixSnap = new byte[gpuModel.StateSnapshotBytes];
        gpuModel.SaveRecurrentState(prefixSnap);

        // Exploratory speculative tokens 3, 4
        gpuModel.ForwardBatch([3, 4], 2, logitsA, computeLogits: true);

        // Restore back to pos=2 and replay [5, 6]
        gpuModel.RestoreRecurrentState(prefixSnap);
        gpuModel.Forward(5, 2, logitsA, computeLogits: true);
        gpuModel.Forward(6, 3, logitsB, computeLogits: true);

        // Verify batched VerifyBatchLogits produces identical logits and per-step snapshots rollback cleanly
        gpuModel.ResetKvCache();
        gpuModel.ForwardBatch([1, 2], 0, Span<float>.Empty, computeLogits: false);
        float[] verifyFlat = new float[2 * weights.VocabSize];
        byte[][] perStep = [new byte[gpuModel.StateSnapshotBytes], new byte[gpuModel.StateSnapshotBytes]];
        gpuModel.VerifyBatchLogits([5, 6], 2, verifyFlat, weights.VocabSize, perStep);

        AssertLogitsClose(logitsA, verifyFlat.AsSpan(0, weights.VocabSize), tol: 2e-3f, "VerifyBatchLogits step 0");
        AssertLogitsClose(logitsB, verifyFlat.AsSpan(weights.VocabSize, weights.VocabSize), tol: 2e-3f, "VerifyBatchLogits step 1");

        // Roll back to step 0 (after token 5 at pos=2) and verify token 7 branch equals clean sequential replay [1,2,5,7]
        float[] expected7 = new float[weights.VocabSize];
        float[] rolled7 = new float[weights.VocabSize];

        gpuModel.RestoreRecurrentState(perStep[0]);
        gpuModel.Forward(7, 3, rolled7, computeLogits: true);

        gpuModel.ResetKvCache();
        gpuModel.ForwardBatch([1, 2, 5], 0, Span<float>.Empty, computeLogits: false);
        gpuModel.Forward(7, 3, expected7, computeLogits: true);

        AssertLogitsClose(expected7, rolled7, tol: 2e-3f, "perStep[0] rollback replay");
    }

    [Fact]
    public void D3D12_Qwen35Moe_SplitExperts_EmbeddedNextN_Mtp_MatchesCpu()
    {
        if (!OperatingSystem.IsWindows()) return;

        string modelPath = Path.Combine(_tempDir, "qwen35moe_embedded_nextn.gguf");
        HostedTestGgufBuilder.BuildRealLayoutQwen35MoeEmbeddedNextNGguf(modelPath);

        int[] prompt = [1, 2, 3, 4, 5];

        using var ggufCpu = GgufFile.Open(modelPath);
        using var weightsCpu = new ModelWeights(ggufCpu);
        Assert.Equal(2, weightsCpu.BlockCount);
        Assert.Equal(2, weightsCpu.EmbeddedMtpBlockIndex);

        using var cpuModel = new Qwen3HybridModel(weightsCpu, maxSeqLen: 64, pleIoMode: PleIoMode.MemoryMapped);
        using var kvCpu = new KVCache(weightsCpu.BlockCount, weightsCpu.HeadCountKv, weightsCpu.HeadDim, 64, weightsCpu.ValueDim);
        using var cpuMtp = MtpDraftProvider.TryCreateFromModel(cpuModel, weightsCpu, maxSeqLen: 64);
        Assert.NotNull(cpuMtp);

        using var ggufGpu = GgufFile.Open(modelPath);
        using var weightsGpu = new ModelWeights(ggufGpu);
        Assert.Equal(2, weightsGpu.BlockCount);
        Assert.Equal(2, weightsGpu.EmbeddedMtpBlockIndex);
        weightsGpu.AttachExpertArena(64UL * 1024 * 1024, preferUnbufferedIo: true);

        using var d3dCtx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(d3dCtx, weightsGpu, maxSeqLen: 64, tieredMoe: true, pleIoMode: PleIoMode.Auto);
        Assert.True(gpuModel.TieredMoe);
        Assert.True(gpuModel.ActiveHostedMoeSlots > 0);

        using var gpuMtp = MtpDraftProvider.TryCreateFromModel(gpuModel, weightsGpu, maxSeqLen: 64);
        Assert.NotNull(gpuMtp);

        float[] cpuLogits = new float[weightsCpu.VocabSize];
        float[] gpuLogits = new float[weightsGpu.VocabSize];

        for (int pos = 0; pos < prompt.Length; pos++)
        {
            cpuModel.Forward(prompt[pos], pos, kvCpu, cpuLogits, computeLogits: true);
            gpuModel.Forward(prompt[pos], pos, gpuLogits, computeLogits: true);
            AssertLogitsClose(cpuLogits, gpuLogits, tol: 5e-3f, $"qwen35moe seq pos={pos}");
            Assert.Equal(ArgMax(cpuLogits), ArgMax(gpuLogits));
        }

        int lastTok = ArgMax(gpuLogits);
        int[] contextWithLast = [.. prompt, lastTok];
        int[] cpuDraft = new int[2];
        int[] gpuDraft = new int[2];
        int cpuDraftCount = cpuMtp!.Draft(contextWithLast, maxDraftTokens: 2, cpuDraft);
        int gpuDraftCount = gpuMtp!.Draft(contextWithLast, maxDraftTokens: 2, gpuDraft);
        Assert.Equal(2, cpuDraftCount);
        Assert.Equal(2, gpuDraftCount);
        Assert.Equal(cpuDraft[0], gpuDraft[0]);
        Assert.Equal(cpuDraft[1], gpuDraft[1]);
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    private static void AssertLogitsClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tol, string context)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxDiff = -1f;
        int maxIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]);
            if (float.IsNaN(d) || d > maxDiff) { maxDiff = d; maxIdx = i; }
        }
        if (!(maxDiff < tol))
        {
            Assert.Fail($"Max logit diff {maxDiff:G5} at idx {maxIdx} (exp={expected[maxIdx]}, act={actual[maxIdx]}) [{context}]");
        }
    }

    private static byte[] BuildSyntheticQuantMatrix(GgufType type, int kCols, int mRows, int seed)
    {
        int blocksPerRow = type switch
        {
            GgufType.Q4_0 or GgufType.Q5_0 or GgufType.MXFP4 => kCols / 32,
            GgufType.IQ4_XS => kCols / 256,
            GgufType.BF16 => 1,
            _ => throw new NotSupportedException()
        };
        int rowBytes = (int)GgufTypes.GetRowBytes(type, kCols);
        byte[] data = new byte[mRows * rowBytes];

        fixed (byte* pData = data)
        {
            for (int r = 0; r < mRows; r++)
            {
                byte* row = pData + (long)r * rowBytes;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    seed = unchecked(seed * 1664525 + 1013904223);
                    if (type == GgufType.Q4_0)
                    {
                        BlockQ4_0* blk = ((BlockQ4_0*)row) + b;
                        blk->Delta = (Half)(0.05f * (((seed & 7) + 1)));
                        for (int j = 0; j < 16; j++)
                        {
                            seed = unchecked(seed * 1664525 + 1013904223);
                            blk->Qs[j] = (byte)(seed & 0xFF);
                        }
                    }
                    else if (type == GgufType.Q5_0)
                    {
                        BlockQ5_0* blk = ((BlockQ5_0*)row) + b;
                        blk->Delta = (Half)(0.04f * (((seed & 7) + 1)));
                        blk->Qh = (uint)seed;
                        for (int j = 0; j < 16; j++)
                        {
                            seed = unchecked(seed * 1664525 + 1013904223);
                            blk->Qs[j] = (byte)(seed & 0xFF);
                        }
                    }
                    else if (type == GgufType.IQ4_XS)
                    {
                        BlockIQ4_XS* blk = ((BlockIQ4_XS*)row) + b;
                        blk->Delta = (Half)(0.004f * (((seed & 7) + 1)));
                        blk->ScalesH = (ushort)(seed & 0xFFFF);
                        for (int s = 0; s < 4; s++) blk->ScalesL[s] = (byte)((seed >> (s * 4)) & 0xFF);
                        for (int j = 0; j < 128; j++)
                        {
                            seed = unchecked(seed * 1664525 + 1013904223);
                            blk->Qs[j] = (byte)(seed & 0xFF);
                        }
                    }
                    else if (type == GgufType.MXFP4)
                    {
                        BlockMXFP4* blk = ((BlockMXFP4*)row) + b;
                        blk->Scale = (byte)(124 + (seed & 7)); // exponents near 2^-3..2^4
                        for (int j = 0; j < 16; j++)
                        {
                            seed = unchecked(seed * 1664525 + 1013904223);
                            blk->Qs[j] = (byte)(seed & 0xFF);
                        }
                    }
                    else if (type == GgufType.BF16)
                    {
                        ushort* bfRow = (ushort*)row;
                        for (int c = 0; c < kCols; c++)
                        {
                            seed = unchecked(seed * 1664525 + 1013904223);
                            float v = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.4f;
                            bfRow[c] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
                        }
                    }
                }
            }
        }

        return data;
    }
}

internal static unsafe class HostedTestGgufBuilder
{
    public sealed record TensorDef(string Name, ulong[] Dims, GgufType Type, byte[] Data)
    {
        public long Offset { get; set; }
    }

    public static void BuildRealLayoutQwen4ExpGguf(string path)
    {
        const int dim = 32;
        const int expFfnDim = 32;
        const int sharedFfnDim = 48;
        const int vocabSize = 48;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int headDim = 8;
        const int ssmConvKernel = 4;
        const int ssmStateSize = 4;
        const int ssmGroupCount = 2;
        const int ssmTimeStepRank = 4;
        const int ssmInnerSize = 16;
        const int gdnConvChannels = (ssmGroupCount * 2 + ssmTimeStepRank) * ssmStateSize; // 32
        const int hcCount = 4;
        const int hcLowRank = 4;
        const int hcDim = hcCount * dim; // 128
        const int idxHeads = 2;
        const int idxKeyDim = 16;
        const int idxTopK = 4;
        const int pleNgram = 3;
        const int pleHeadsPerNgram = 4;
        const int pleHeadDim = 32;
        const long pleTotalRows = 240;

        var tokList = new object[vocabSize];
        for (int i = 0; i < vocabSize; i++) tokList[i] = i == 47 ? "<|endoftext|>" : $"tok_{i}";

        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen4exp"),
            ("general.alignment", 4U, 32U),
            ("qwen4exp.block_count", 4U, 2U),
            ("qwen4exp.embedding_length", 4U, (uint)dim),
            ("qwen4exp.context_length", 4U, 128U),
            ("qwen4exp.attention.head_count", 9U, (4U, new object[] { 0U, (uint)nHeads })),
            ("qwen4exp.attention.head_count_kv", 9U, (4U, new object[] { 0U, (uint)nHeadsKv })),
            ("qwen4exp.attention.key_length", 4U, (uint)headDim),
            ("qwen4exp.attention.value_length", 4U, (uint)headDim),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen4exp.rope.dimension_count", 4U, (uint)headDim),
            ("qwen4exp.rope.freq_base", 6U, 10000.0f),
            ("qwen4exp.ssm.conv_kernel", 4U, (uint)ssmConvKernel),
            ("qwen4exp.ssm.state_size", 4U, (uint)ssmStateSize),
            ("qwen4exp.ssm.group_count", 4U, (uint)ssmGroupCount),
            ("qwen4exp.ssm.time_step_rank", 4U, (uint)ssmTimeStepRank),
            ("qwen4exp.ssm.inner_size", 4U, (uint)ssmInnerSize),
            ("qwen4exp.expert_count", 4U, 4U),
            ("qwen4exp.expert_used_count", 4U, 2U),
            ("qwen4exp.expert_feed_forward_length", 4U, (uint)expFfnDim),
            ("qwen4exp.expert_shared_feed_forward_length", 4U, (uint)sharedFfnDim),
            ("qwen4exp.hyper_connection.count", 4U, (uint)hcCount),
            ("qwen4exp.hyper_connection.low_rank", 4U, (uint)hcLowRank),
            ("qwen4exp.attention.indexer.head_count", 4U, (uint)idxHeads),
            ("qwen4exp.attention.indexer.key_length", 4U, (uint)idxKeyDim),
            ("qwen4exp.attention.indexer.top_k", 4U, (uint)idxTopK),
            ("qwen4exp.attention.compress_ratios", 9U, (4U, new object[] { 0U, 4U })),
            ("qwen4exp.ple.layers", 9U, (4U, new object[] { 0U })),
            ("qwen4exp.ple.ngram_size", 4U, (uint)pleNgram),
            ("qwen4exp.ple.heads_per_ngram", 4U, (uint)pleHeadsPerNgram),
            ("qwen4exp.ple.conv_kernel", 4U, 4U),
            ("qwen4exp.ple.eos_token_id", 4U, 47U),
            ("qwen4exp.embedding_length_per_layer_input", 4U, (uint)pleHeadDim),
            ("qwen4exp.ple.layer_multipliers", 9U, (10U, new object[] { 11UL, 23UL, 37UL })),
            ("qwen4exp.ple.head_offsets", 9U, (10U, new object[] { 0UL, 17UL, 36UL, 59UL, 88UL, 119UL, 156UL, 197UL })),
            ("qwen4exp.ple.head_vocab_sizes", 9U, (10U, new object[] { 17UL, 19UL, 23UL, 29UL, 31UL, 37UL, 41UL, 43UL })),
            ("tokenizer.ggml.tokens", 9U, (8U, tokList)),
            ("tokenizer.ggml.bos_token_id", 4U, 0U),
            ("tokenizer.ggml.eos_token_id", 4U, 47U)
        };

        var tensors = new List<TensorDef>();
        int seed = 31337;

        void AddF32(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 4];
            Span<float> fSpan = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                fSpan[(int)i] = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.1f;
            }
            tensors.Add(new TensorDef(name, dims, GgufType.F32, bytes));
        }

        void AddBf16(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 2];
            Span<ushort> bfSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                float v = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.1f;
                bfSpan[(int)i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
            }
            tensors.Add(new TensorDef(name, dims, GgufType.BF16, bytes));
        }

        void AddOneNorm(string name, int len, bool asBf16 = false)
        {
            if (asBf16)
            {
                byte[] b = new byte[len * 2];
                MemoryMarshal.Cast<byte, ushort>(b.AsSpan()).Fill(0x3F80);
                tensors.Add(new TensorDef(name, [(ulong)len], GgufType.BF16, b));
            }
            else
            {
                byte[] b = new byte[len * 4];
                MemoryMarshal.Cast<byte, float>(b.AsSpan()).Fill(1.0f);
                tensors.Add(new TensorDef(name, [(ulong)len], GgufType.F32, b));
            }
        }

        AddBf16("token_embd.weight", dim, vocabSize);
        AddBf16("output.weight", dim, vocabSize);
        AddOneNorm("hc_head_norm.weight", hcDim, asBf16: true);
        AddBf16("hc_head_down.weight", hcDim, hcLowRank);
        AddBf16("hc_head_up.weight", hcLowRank, hcDim);
        AddBf16("per_layer_token_embd.weight", pleHeadDim, (ulong)pleTotalRows);

        // Layer 0 (GDN + PLE + HC + Fused MoE)
        AddOneNorm("blk.0.hc_attn_norm.weight", hcDim, asBf16: true);
        AddBf16("blk.0.hc_attn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddBf16("blk.0.hc_attn_up.weight", hcLowRank, hcDim);
        AddOneNorm("blk.0.hc_ffn_norm.weight", hcDim, asBf16: true);
        AddBf16("blk.0.hc_ffn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddBf16("blk.0.hc_ffn_up.weight", hcLowRank, hcDim);

        AddBf16("blk.0.attn_qkv.weight", dim, gdnConvChannels);
        AddBf16("blk.0.attn_gate.weight", dim, ssmInnerSize);
        AddF32("blk.0.ssm_alpha.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_beta.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_conv1d.weight", ssmConvKernel, gdnConvChannels);
        AddF32("blk.0.ssm_dt.bias", ssmTimeStepRank);
        AddF32("blk.0.ssm_a", ssmTimeStepRank);
        AddOneNorm("blk.0.ssm_norm.weight", ssmStateSize);
        AddBf16("blk.0.ssm_out.weight", ssmInnerSize, dim);

        int pleEmbDim = (pleNgram - 1) * pleHeadsPerNgram * pleHeadDim;
        AddOneNorm("blk.0.ple_norm_key.weight", hcDim);
        AddOneNorm("blk.0.ple_norm_query.weight", hcDim);
        AddOneNorm("blk.0.ple_norm_conv.weight", hcDim);
        AddF32("blk.0.ple_key.weight", (ulong)pleEmbDim, hcDim);
        AddF32("blk.0.ple_value.weight", (ulong)pleEmbDim, dim);
        AddF32("blk.0.ple_conv1d.weight", 4, hcDim);

        AddBf16("blk.0.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.0.ffn_gate_up_exps.weight", dim, 2 * expFfnDim, 4);
        AddF32("blk.0.ffn_down_exps.weight", expFfnDim, dim, 4);
        AddF32("blk.0.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.0.ffn_gate_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_up_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.0.ffn_down_shexp.weight", sharedFfnDim, dim);

        // Layer 1 (QSA + HC + Fused MoE)
        AddOneNorm("blk.1.hc_attn_norm.weight", hcDim, asBf16: true);
        AddBf16("blk.1.hc_attn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddBf16("blk.1.hc_attn_up.weight", hcLowRank, hcDim);
        AddOneNorm("blk.1.hc_ffn_norm.weight", hcDim, asBf16: true);
        AddBf16("blk.1.hc_ffn_down_inject.weight", hcDim, hcLowRank + hcCount);
        AddBf16("blk.1.hc_ffn_up.weight", hcLowRank, hcDim);

        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;
        AddF32("blk.1.attn_qkv.weight", dim, (ulong)(2 * qDim + 2 * kvDim));
        AddF32("blk.1.attn_output.weight", (ulong)qDim, dim);
        AddOneNorm("blk.1.attn_q_norm.weight", headDim, asBf16: true);
        AddOneNorm("blk.1.attn_k_norm.weight", headDim, asBf16: true);

        AddF32("blk.1.indexer.q_proj.weight", dim, idxHeads * idxKeyDim);
        AddF32("blk.1.indexer.k_proj.weight", dim, idxKeyDim);
        AddOneNorm("blk.1.indexer.q_norm.weight", idxKeyDim, asBf16: true);
        AddOneNorm("blk.1.indexer.k_norm.weight", idxKeyDim, asBf16: true);

        AddBf16("blk.1.ffn_gate_inp.weight", dim, 4);
        AddF32("blk.1.ffn_gate_up_exps.weight", dim, 2 * expFfnDim, 4);
        AddF32("blk.1.ffn_down_exps.weight", expFfnDim, dim, 4);
        AddF32("blk.1.ffn_gate_inp_shexp.weight", dim);
        AddF32("blk.1.ffn_gate_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.1.ffn_up_shexp.weight", dim, sharedFfnDim);
        AddF32("blk.1.ffn_down_shexp.weight", sharedFfnDim, dim);

        Write(path, kv, tensors);
    }

    public static void BuildRealLayoutQwen35MoeEmbeddedNextNGguf(string path)
    {
        const int dim = 64;
        const int expFfnDim = 64;
        const int sharedFfnDim = 64;
        const int vocabSize = 48;
        const int nHeads = 4;
        const int nHeadsKv = 2;
        const int headDim = 16;
        const int ropeDim = 8;
        const int ssmConvKernel = 4;
        const int ssmStateSize = 8;
        const int ssmGroupCount = 2;
        const int ssmTimeStepRank = 4;
        const int ssmInnerSize = 32;
        const int gdnConvChannels = (ssmGroupCount * 2 + ssmTimeStepRank) * ssmStateSize; // 64
        const int numExperts = 8;
        const int topK = 4;

        var tokList = new object[vocabSize];
        for (int i = 0; i < vocabSize; i++) tokList[i] = i == 47 ? "<|endoftext|>" : $"tok_{i}";

        var kv = new List<(string Key, uint Type, object Value)>
        {
            ("general.architecture", 8U, "qwen35moe"),
            ("general.alignment", 4U, 32U),
            ("qwen35moe.block_count", 4U, 3U),
            ("qwen35moe.embedding_length", 4U, (uint)dim),
            ("qwen35moe.context_length", 4U, 128U),
            ("qwen35moe.attention.head_count", 4U, (uint)nHeads),
            ("qwen35moe.attention.head_count_kv", 4U, (uint)nHeadsKv),
            ("qwen35moe.attention.key_length", 4U, (uint)headDim),
            ("qwen35moe.attention.value_length", 4U, (uint)headDim),
            ("qwen35moe.attention.layer_norm_rms_epsilon", 6U, 1e-6f),
            ("qwen35moe.rope.dimension_count", 4U, (uint)ropeDim),
            ("qwen35moe.rope.freq_base", 6U, 10000.0f),
            ("qwen35moe.full_attention_interval", 4U, 2U),
            ("qwen35moe.ssm.conv_kernel", 4U, (uint)ssmConvKernel),
            ("qwen35moe.ssm.state_size", 4U, (uint)ssmStateSize),
            ("qwen35moe.ssm.group_count", 4U, (uint)ssmGroupCount),
            ("qwen35moe.ssm.time_step_rank", 4U, (uint)ssmTimeStepRank),
            ("qwen35moe.ssm.inner_size", 4U, (uint)ssmInnerSize),
            ("qwen35moe.expert_count", 4U, (uint)numExperts),
            ("qwen35moe.expert_used_count", 4U, (uint)topK),
            ("qwen35moe.expert_feed_forward_length", 4U, (uint)expFfnDim),
            ("qwen35moe.expert_shared_feed_forward_length", 4U, (uint)sharedFfnDim),
            ("qwen35moe.nextn_predict_layers", 4U, 1U),
            ("tokenizer.ggml.tokens", 9U, (8U, tokList)),
            ("tokenizer.ggml.bos_token_id", 4U, 0U),
            ("tokenizer.ggml.eos_token_id", 4U, 47U)
        };

        var tensors = new List<TensorDef>();
        int seed = 5150;

        void AddF32(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 4];
            Span<float> fSpan = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                fSpan[(int)i] = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.1f;
            }
            tensors.Add(new TensorDef(name, dims, GgufType.F32, bytes));
        }

        void AddBf16(string name, params ulong[] dims)
        {
            long count = 1;
            foreach (ulong d in dims) count *= (long)d;
            byte[] bytes = new byte[count * 2];
            Span<ushort> bfSpan = MemoryMarshal.Cast<byte, ushort>(bytes.AsSpan());
            for (long i = 0; i < count; i++)
            {
                seed = unchecked(seed * 1664525 + 1013904223);
                float v = ((seed & 0xFFFF) / 32768.0f - 1.0f) * 0.1f;
                bfSpan[(int)i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
            }
            tensors.Add(new TensorDef(name, dims, GgufType.BF16, bytes));
        }

        void AddOneNorm(string name, int len, bool asBf16 = false)
        {
            if (asBf16)
            {
                byte[] b = new byte[len * 2];
                MemoryMarshal.Cast<byte, ushort>(b.AsSpan()).Fill(0x3F80);
                tensors.Add(new TensorDef(name, [(ulong)len], GgufType.BF16, b));
            }
            else
            {
                byte[] b = new byte[len * 4];
                MemoryMarshal.Cast<byte, float>(b.AsSpan()).Fill(1.0f);
                tensors.Add(new TensorDef(name, [(ulong)len], GgufType.F32, b));
            }
        }

        void AddSplitMoe(int layerIdx, bool bf16Routers = false)
        {
            string p = $"blk.{layerIdx}.";
            if (bf16Routers)
            {
                AddBf16(p + "ffn_gate_inp.weight", dim, numExperts);
                AddBf16(p + "ffn_gate_inp_shexp.weight", dim);
            }
            else
            {
                AddF32(p + "ffn_gate_inp.weight", dim, numExperts);
                AddF32(p + "ffn_gate_inp_shexp.weight", dim);
            }
            AddF32(p + "ffn_gate_exps.weight", dim, expFfnDim, numExperts);
            AddF32(p + "ffn_up_exps.weight", dim, expFfnDim, numExperts);
            AddF32(p + "ffn_down_exps.weight", expFfnDim, dim, numExperts);
            AddF32(p + "ffn_gate_shexp.weight", dim, sharedFfnDim);
            AddF32(p + "ffn_up_shexp.weight", dim, sharedFfnDim);
            AddF32(p + "ffn_down_shexp.weight", sharedFfnDim, dim);
        }

        AddF32("token_embd.weight", dim, vocabSize);
        AddOneNorm("output_norm.weight", dim);
        AddF32("output.weight", dim, vocabSize);

        // blk.0: GDN + split MoE
        AddOneNorm("blk.0.attn_norm.weight", dim);
        AddF32("blk.0.attn_qkv.weight", dim, gdnConvChannels);
        AddF32("blk.0.attn_gate.weight", dim, ssmInnerSize);
        AddF32("blk.0.ssm_alpha.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_beta.weight", dim, ssmTimeStepRank);
        AddF32("blk.0.ssm_conv1d.weight", ssmConvKernel, gdnConvChannels);
        AddF32("blk.0.ssm_dt.bias", ssmTimeStepRank);
        AddF32("blk.0.ssm_a", ssmTimeStepRank);
        AddOneNorm("blk.0.ssm_norm.weight", ssmStateSize);
        AddF32("blk.0.ssm_out.weight", ssmInnerSize, dim);
        AddOneNorm("blk.0.post_attention_norm.weight", dim);
        AddSplitMoe(0);

        int qDim = nHeads * headDim;
        int kvDim = nHeadsKv * headDim;

        // blk.1: Q-gated Full Attention + split MoE
        AddOneNorm("blk.1.attn_norm.weight", dim);
        AddF32("blk.1.attn_q.weight", dim, (ulong)(2 * qDim));
        AddF32("blk.1.attn_k.weight", dim, (ulong)kvDim);
        AddF32("blk.1.attn_v.weight", dim, (ulong)kvDim);
        AddF32("blk.1.attn_output.weight", (ulong)qDim, dim);
        AddOneNorm("blk.1.attn_q_norm.weight", headDim);
        AddOneNorm("blk.1.attn_k_norm.weight", headDim);
        AddOneNorm("blk.1.post_attention_norm.weight", dim);
        AddSplitMoe(1);

        // blk.2: Embedded trailing NextN MTP block with BF16 routers + split MoE
        AddOneNorm("blk.2.attn_norm.weight", dim);
        AddF32("blk.2.attn_q.weight", dim, (ulong)(2 * qDim));
        AddF32("blk.2.attn_k.weight", dim, (ulong)kvDim);
        AddF32("blk.2.attn_v.weight", dim, (ulong)kvDim);
        AddF32("blk.2.attn_output.weight", (ulong)qDim, dim);
        AddOneNorm("blk.2.attn_q_norm.weight", headDim);
        AddOneNorm("blk.2.attn_k_norm.weight", headDim);
        AddOneNorm("blk.2.post_attention_norm.weight", dim);
        AddSplitMoe(2, bf16Routers: true);
        AddOneNorm("blk.2.nextn.enorm.weight", dim);
        AddOneNorm("blk.2.nextn.hnorm.weight", dim);
        AddF32("blk.2.nextn.eh_proj.weight", (ulong)(2 * dim), dim);
        AddOneNorm("blk.2.nextn.shared_head_norm.weight", dim);

        Write(path, kv, tensors);
    }

    public static void Write(string path, List<(string Key, uint Type, object Value)> kv, List<TensorDef> tensors)
    {
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offset = (offset + 31L) & ~31L;
            tensors[i].Offset = offset;
            offset += tensors[i].Data.Length;
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        bw.Write(0x46554747U);
        bw.Write(3U);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)kv.Count);

        static void WriteStr(BinaryWriter w, string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        static void WriteVal(BinaryWriter w, uint type, object val)
        {
            switch (type)
            {
                case 4U: w.Write((uint)val); break;
                case 5U: w.Write((int)val); break;
                case 6U: w.Write((float)val); break;
                case 8U: WriteStr(w, (string)val); break;
                case 10U: w.Write((ulong)val); break;
                case 9U:
                {
                    var (elemType, arr) = ((uint, object[]))val;
                    w.Write(elemType);
                    w.Write((ulong)arr.Length);
                    for (int j = 0; j < arr.Length; j++) WriteVal(w, elemType, arr[j]);
                    break;
                }
                default:
                    throw new NotSupportedException($"Unsupported GGUF metadata type {type}.");
            }
        }

        foreach (var (key, type, value) in kv)
        {
            WriteStr(bw, key);
            bw.Write(type);
            WriteVal(bw, type, value);
        }

        foreach (var t in tensors)
        {
            WriteStr(bw, t.Name);
            bw.Write((uint)t.Dims.Length);
            foreach (ulong d in t.Dims) bw.Write(d);
            bw.Write((uint)t.Type);
            bw.Write((ulong)t.Offset);
        }

        long pos = fs.Position;
        long dataStart = (pos + 31L) & ~31L;
        if (dataStart > pos) bw.Write(new byte[dataStart - pos]);

        for (int i = 0; i < tensors.Count; i++)
        {
            long targetPos = dataStart + tensors[i].Offset;
            if (fs.Position < targetPos) bw.Write(new byte[targetPos - fs.Position]);
            bw.Write(tensors[i].Data);
        }
    }
}
