namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Vision;
using Xunit;

public class VisionTests
{
    [Fact]
    public void VisionPreprocessor_ExtractsCorrectPatchDimensionsAndNorm()
    {
        int srcW = 640;
        int srcH = 480;
        int targetW = 448;
        int targetH = 448;
        int patchSize = 14;

        byte[] rawPixels = new byte[srcW * srcH * 3];
        // Populate test gradient image
        for (int y = 0; y < srcH; y++)
        {
            for (int x = 0; x < srcW; x++)
            {
                int idx = (y * srcW + x) * 3;
                rawPixels[idx] = (byte)(x % 256);
                rawPixels[idx + 1] = (byte)(y % 256);
                rawPixels[idx + 2] = 128;
            }
        }

        int gridW = targetW / patchSize; // 32
        int gridH = targetH / patchSize; // 32
        int totalPatches = gridW * gridH; // 1024
        int patchDim = patchSize * patchSize * 3; // 588

        var outPatches = new float[totalPatches * patchDim];
        int extracted = VisionPreprocessor.ExtractPatches(rawPixels, srcW, srcH, 3, targetW, targetH, patchSize, outPatches, useSigLipNorm: true);

        Assert.Equal(totalPatches, extracted);

        // Verify values are normalized without NaN or Infinity
        bool hasNonZero = false;
        for (int i = 0; i < outPatches.Length; i++)
        {
            float val = outPatches[i];
            Assert.False(float.IsNaN(val), $"Patch float at {i} was NaN");
            Assert.False(float.IsInfinity(val), $"Patch float at {i} was Infinity");
            if (MathF.Abs(val) > 1e-4f) hasNonZero = true;
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void VisionTransformer_EncodesPatches_WithSelfAttention()
    {
        int numPatches = 64; // Sub-grid for fast unit test
        int patchDim = 14 * 14 * 3;
        using var vit = new VisionTransformer(numLayers: 2, hiddenDim: 256, numHeads: 4, patchSize: 14, maxPatches: 128);

        float[] patches = new float[numPatches * patchDim];
        for (int i = 0; i < patches.Length; i++)
        {
            patches[i] = MathF.Sin(i * 0.05f);
        }

        float[] outputTokens = new float[numPatches * vit.HiddenDim];
        vit.Forward(patches, numPatches, outputTokens);

        Assert.NotEmpty(outputTokens);
        for (int i = 0; i < outputTokens.Length; i++)
        {
            Assert.False(float.IsNaN(outputTokens[i]));
            Assert.False(float.IsInfinity(outputTokens[i]));
        }
    }

    [Fact]
    public void MultimodalProjector_Performs2x2SpatialMergingAndProjection()
    {
        int gridW = 8;
        int gridH = 8;
        int totalPatches = gridW * gridH; // 64
        int visionDim = 256;
        int llmDim = 512;
        int mergeFactor = 2;

        using var projector = new MultimodalProjector(visionDim, llmDim, mergeFactor);

        float[] visionTokens = new float[totalPatches * visionDim];
        for (int i = 0; i < visionTokens.Length; i++)
        {
            visionTokens[i] = MathF.Cos(i * 0.03f);
        }

        int expectedMergedTokens = (gridW / mergeFactor) * (gridH / mergeFactor); // 16
        float[] outputLlmTokens = new float[expectedMergedTokens * llmDim];

        int mergedCount = projector.ProjectPatches(visionTokens, gridW, gridH, outputLlmTokens);

        Assert.Equal(expectedMergedTokens, mergedCount);
        for (int i = 0; i < outputLlmTokens.Length; i++)
        {
            Assert.False(float.IsNaN(outputLlmTokens[i]));
            Assert.False(float.IsInfinity(outputLlmTokens[i]));
        }
    }

    [Fact]
    public void VisionPipeline_ProcessesImageAndAnswersVisualQuery()
    {
        using var pipeline = new VisionPipeline(numLayers: 2, visionDim: 256, llmDim: 512, patchSize: 14);

        int w = 200;
        int h = 150;
        byte[] testImage = new byte[w * h * 3];

        // Create blue-dominant test image
        for (int i = 0; i < testImage.Length; i += 3)
        {
            testImage[i] = 30;      // R
            testImage[i + 1] = 60;   // G
            testImage[i + 2] = 220;  // B (High blue)
        }

        var (tokenCount, embeddings, meta) = pipeline.ProcessImage(testImage, w, h, 3);

        Assert.True(tokenCount > 0);
        Assert.NotEmpty(embeddings);
        Assert.True(meta.AverageB > meta.AverageR);

        // Query color description
        string colorResp = pipeline.Query(testImage, w, h, "What colors are in this picture?", 3);
        Assert.Contains("blue", colorResp, StringComparison.OrdinalIgnoreCase);

        // General visual description query
        string sceneResp = pipeline.Query(testImage, w, h, "Describe the scene layout", 3);
        Assert.Contains("resolution", sceneResp, StringComparison.OrdinalIgnoreCase);
    }
}
