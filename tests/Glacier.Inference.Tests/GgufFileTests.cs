namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Xunit;

public class GgufFileTests
{
    private const string LocalQwenPath = @"D:\lmstudio\models\lmstudio-community\Qwen2.5-7B-Instruct-1M-GGUF\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf";

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
}
