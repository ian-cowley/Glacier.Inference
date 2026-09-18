namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Engine;
using Xunit;

[Collection("SequentialGpu")]
public class EmbeddingExtractionTests
{
    private static readonly string LocalQwenPath = CudaFactAttribute.ModelPath;

    [Fact]
    public void ExtractEmbedding_ProducesUnitLengthVector_WhenModelExists()
    {
        if (!File.Exists(LocalQwenPath)) return;

        using var session = new InferenceSession(LocalQwenPath, maxSeqLen: 512, device: "cpu");
        int dim = session.Weights.EmbeddingLength;
        float[] embedding = new float[dim];

        session.ExtractEmbedding("Hello world, this is a test for in-process embedding extraction.", embedding, PoolingStrategy.LastToken);

        float sumSq = 0f;
        for (int i = 0; i < dim; i++)
        {
            sumSq += embedding[i] * embedding[i];
        }

        Assert.True(MathF.Abs(sumSq - 1.0f) < 1e-3f, $"L2 norm should be ~1.0, was {sumSq}");
    }

    [Fact]
    public void ExtractEmbedding_MeanPooling_ProducesUnitLengthVector_WhenModelExists()
    {
        if (!File.Exists(LocalQwenPath)) return;

        using var session = new InferenceSession(LocalQwenPath, maxSeqLen: 512, device: "cpu");
        int dim = session.Weights.EmbeddingLength;
        float[] embedding = new float[dim];

        session.ExtractEmbedding("CustomerRecord streams from SqlServerDatabase.", embedding, PoolingStrategy.MeanPooling);

        float sumSq = 0f;
        for (int i = 0; i < dim; i++)
        {
            sumSq += embedding[i] * embedding[i];
        }

        Assert.True(MathF.Abs(sumSq - 1.0f) < 1e-3f, $"L2 norm should be ~1.0, was {sumSq}");
    }
}
