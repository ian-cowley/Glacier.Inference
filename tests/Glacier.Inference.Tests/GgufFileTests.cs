namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Xunit;

public class GgufFileTests
{
    private static readonly string LocalQwenPath = CudaFactAttribute.ModelPath;

    [Fact]
    public void Open_ParsesHeaderAndMetadata_WhenFileExists()
    {
        if (!File.Exists(LocalQwenPath)) return;

        using var gguf = GgufFile.Open(LocalQwenPath);

        Assert.Equal("qwen2", gguf.Architecture);
        Assert.Equal(28, gguf.BlockCount);
        Assert.Equal(3584, gguf.EmbeddingLength);
        Assert.Equal(18944, gguf.FeedForwardLength);
        Assert.Equal(28, gguf.HeadCount);
        Assert.Equal(4, gguf.HeadCountKv);
        Assert.Equal(151645, gguf.EosTokenId);
        Assert.True(gguf.TensorCount >= 339);
    }

    [Fact]
    public void InspectUniversalArchitectures()
    {
        string[] paths = [
            @"D:\lmstudio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            @"D:\lmstudio\models\lmstudio-community\DeepSeek-Coder-V2-Lite-Instruct-GGUF\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
            @"D:\lmstudio\models\lmstudio-community\phi-4-GGUF\phi-4-Q4_K_M.gguf",
            @"D:\lmstudio\models\lmstudio-community\Devstral-Small-2505-GGUF\Devstral-Small-2505-Q4_K_M.gguf"
        ];

        foreach (var path in paths)
        {
            if (!File.Exists(path)) continue;
            using var gguf = GgufFile.Open(path);
            Console.WriteLine($"\n[MODEL INFO] Path: {Path.GetFileName(path)}");
            Console.WriteLine($"  Arch: {gguf.Architecture}");
            Console.WriteLine($"  Layers: {gguf.BlockCount}, Dim: {gguf.EmbeddingLength}, FFN: {gguf.FeedForwardLength}");
            Console.WriteLine($"  Heads: {gguf.HeadCount}, HeadsKv: {gguf.HeadCountKv}, HeadDim: {gguf.HeadDim}, ValDim: {gguf.ValueDim}");
            Console.WriteLine($"  RopeFreqBase: {gguf.RopeFreqBase}, RopeDim: {gguf.RopeDimensionCount}");
            Console.WriteLine($"  RopeScalingType: {gguf.RopeScalingType}, Factor: {gguf.RopeScalingFactor}");
            Console.WriteLine($"  RmsNormEps: {gguf.RmsNormEps}, ContextLen: {gguf.ContextLength}");
            Console.WriteLine($"  IsMla: {gguf.IsMla}, IsMoe: {gguf.IsMoe}, Experts: {gguf.ExpertCount} (used: {gguf.ExpertUsedCount})");
            Console.WriteLine("  Layer 0 detailed tensors:");
            foreach (var kv in gguf.Tensors.Where(t => t.Key.StartsWith("blk.0.")))
            {
                Console.WriteLine($"    {kv.Key}: Type={kv.Value.Type}, Dims=[{string.Join(", ", kv.Value.Dimensions)}]");
            }
        }
    }
}
