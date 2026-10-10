namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Engine;
using Xunit;

public class TieredMoERunnerTests
{
    [Fact]
    public void TieredRunnerOptions_HasCorrectDefaults()
    {
        var options = new TieredRunnerOptions();

        Assert.Equal(-1, options.GpuAdapterIndex);
        Assert.Equal(8192ul, options.VramBudgetMb);
        Assert.Equal(4096, options.MaxSequenceLength);
        Assert.Equal(TieringExecutionPolicy.Auto, options.Policy);
        Assert.Equal(0.7f, options.Temperature);
        Assert.Equal(0.9f, options.TopP);
        Assert.Equal(40, options.TopK);
    }

    [Fact]
    public void Constructor_ThrowsFileNotFound_ForMissingModel()
    {
        string fakePath = Path.Combine(Path.GetTempPath(), "nonexistent_model_" + Guid.NewGuid() + ".gguf");

        var ex = Assert.Throws<FileNotFoundException>(() =>
        {
            using var runner = new TieredMoERunner(fakePath);
        });

        Assert.Contains(fakePath, ex.Message);
    }

    [Fact]
    public void TieredStepTelemetry_StoresPropertiesAccurately()
    {
        var telemetry = new TieredStepTelemetry
        {
            StepIndex = 5,
            TokenId = 1337,
            TokenText = " world",
            TotalStepDurationMs = 28.5,
            AttentionVramDurationMs = 8.2,
            RoutedExpertDurationMs = 18.9,
            NetworkInterconnectDurationMs = 1.4
        };

        Assert.Equal(5, telemetry.StepIndex);
        Assert.Equal(1337, telemetry.TokenId);
        Assert.Equal(" world", telemetry.TokenText);
        Assert.Equal(28.5, telemetry.TotalStepDurationMs);
        Assert.Equal(8.2, telemetry.AttentionVramDurationMs);
        Assert.Equal(18.9, telemetry.RoutedExpertDurationMs);
        Assert.Equal(1.4, telemetry.NetworkInterconnectDurationMs);
    }
}
