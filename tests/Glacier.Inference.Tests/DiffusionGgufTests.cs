namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Image.Gguf;
using Glacier.Inference.Model;
using Xunit;

public class DiffusionGgufTests
{
    [Fact]
    public void ModelArchitectureDetector_DetectsFluxAndDiffusionFamilies()
    {
        string tmpFile = Path.Combine(Path.GetTempPath(), $"glacier_mock_gguf_{Guid.NewGuid():N}.gguf");
        try
        {
            using (var fs = new FileStream(tmpFile, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(0x46554747); // 'GGUF'
                bw.Write(3u);         // Version 3
                bw.Write(0UL);        // 0 tensors
                bw.Write(0UL);        // 0 metadata
            }

            using (var gguf = GgufFile.Open(tmpFile))
            {
                Assert.Equal(UniversalArchitecture.Flux, ModelArchitectureDetector.Detect("flux", gguf));
                Assert.Equal(UniversalArchitecture.SD3, ModelArchitectureDetector.Detect("sd3", gguf));
                Assert.Equal(UniversalArchitecture.StableDiffusion, ModelArchitectureDetector.Detect("sdxl", gguf));
                Assert.Equal(UniversalArchitecture.StableDiffusion, ModelArchitectureDetector.Detect("sd1", gguf));
            }
        }
        finally
        {
            if (File.Exists(tmpFile)) File.Delete(tmpFile);
        }
    }

    [Fact]
    public void DiffusionGgufPipeline_RunsEndToEndSyntheticFlow()
    {
        int w = 512;
        int h = 512;
        int steps = 4;

        using var pipeline = new Glacier.Inference.Image.ImageGenerationPipeline();
        var result = pipeline.Generate("A cinematic landscape of icy fjords under neon auroras", width: w, height: h, numSteps: steps, seed: 123);

        Assert.Equal(w, result.Width);
        Assert.Equal(h, result.Height);
        Assert.Equal(steps, result.Steps);
        Assert.Equal(w * h * 3, result.RgbPixels.Length);
        Assert.True(result.ElapsedMilliseconds >= 0);
    }
}
