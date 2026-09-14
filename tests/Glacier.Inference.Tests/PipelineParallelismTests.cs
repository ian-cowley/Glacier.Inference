namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Pipeline;
using Glacier.Inference.Sampling;
using Xunit;
using Xunit.Abstractions;

public class PipelineParallelismTests
{
    private readonly ITestOutputHelper _output;
    private static readonly string ModelPath = CudaFactAttribute.ModelPath;

    public PipelineParallelismTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void PipelineSplitConfig_ParseManualSyntax_Success()
    {
        if (!File.Exists(ModelPath)) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        int totalLayers = weights.BlockCount;

        // Test "14,14" or split into two equal halves
        int half = totalLayers / 2;
        int rem = totalLayers - half;
        var stages = PipelineSplitConfig.ResolveStages($"{half},{rem}", weights);

        Assert.NotNull(stages);
        Assert.Equal(2, stages.Count);
        Assert.Equal(0, stages[0].StartLayer);
        Assert.Equal(half, stages[0].LayerCount);
        Assert.Equal(half, stages[1].StartLayer);
        Assert.Equal(rem, stages[1].LayerCount);
        Assert.Equal(totalLayers, stages[0].LayerCount + stages[1].LayerCount);
    }

    [Fact]
    public void PipelineSplitConfig_AutoPartition_CoversAllLayers()
    {
        if (!File.Exists(ModelPath)) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);

        var stages = PipelineSplitConfig.ResolveStages("auto", weights);
        Assert.NotNull(stages);
        Assert.NotEmpty(stages);

        int totalAssigned = stages.Sum(s => s.LayerCount);
        Assert.Equal(weights.BlockCount, totalAssigned);
        Assert.Equal(0, stages[0].StartLayer);

        for (int i = 1; i < stages.Count; i++)
        {
            Assert.Equal(stages[i - 1].EndLayer, stages[i].StartLayer);
        }
    }

    [Fact]
    public void MultiStage_Cpu_NumericalParity_With_SingleStage()
    {
        if (!File.Exists(ModelPath)) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        int totalLayers = weights.BlockCount;

        int half = totalLayers / 2;
        int rem = totalLayers - half;

        // 1. Single-stage execution
        using var singleModel = new Qwen2Model(weights, maxSeqLen: 128);
        using var singleKv = new KVCache(totalLayers, weights.HeadCountKv, weights.HeadDim, 128);
        float[] singleLogits = new float[weights.VocabSize];
        singleModel.Forward(1234, 0, singleKv, singleLogits.AsSpan(), computeLogits: true);

        // 2. Multi-stage execution on CPU
        var cpuDev = DeviceManager.ResolveDevice("cpu");
        using var stage0 = new CpuPipelineStage(cpuDev, weights, 0, half, isFirstStage: true, isLastStage: false, maxSeqLen: 128);
        using var stage1 = new CpuPipelineStage(cpuDev, weights, half, rem, isFirstStage: false, isLastStage: true, maxSeqLen: 128);
        using var pipeSession = new PipelineSession(new IPipelineStage[] { stage0, stage1 }, weights.EmbeddingLength, weights.VocabSize);

        float[] multiLogits = new float[weights.VocabSize];
        pipeSession.Forward(1234, 0, multiLogits.AsSpan(), computeLogits: true);

        // 3. Compare logits
        float maxDiff = 0.0f;
        for (int i = 0; i < weights.VocabSize; i++)
        {
            float diff = MathF.Abs(singleLogits[i] - multiLogits[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        _output.WriteLine($"[CPU Multi-Stage Parity]: Max logits difference = {maxDiff:E6}");
        Assert.True(maxDiff < 1e-4f, $"Multi-stage execution diverged: max difference = {maxDiff}");
    }
}
