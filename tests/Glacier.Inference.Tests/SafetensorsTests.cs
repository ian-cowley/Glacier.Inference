namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Format;
using Xunit;

public class SafetensorsTests
{
    [Fact]
    public void SafetensorsFile_OpensAndDiscoversVaeTensors()
    {
        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "models", "ae.safetensors");
        if (!File.Exists(path))
        {
            path = "models/ae.safetensors";
        }

        if (File.Exists(path))
        {
            using var sf = SafetensorsFile.Open(path);
            Assert.True(sf.Tensors.Count > 200, "Flux VAE must contain >200 tensors");
            Assert.True(sf.ContainsTensor("decoder.conv_in.weight"));
            Assert.True(sf.ContainsTensor("decoder.conv_out.weight"));
            
            var convIn = sf.Tensors["decoder.conv_in.weight"];
            Assert.Equal(4, convIn.Shape.Length);
            Assert.Equal(512, convIn.Shape[0]);
            Assert.Equal(16, convIn.Shape[1]);
            Assert.Equal(3, convIn.Shape[2]);
            Assert.Equal(3, convIn.Shape[3]);
        }
    }

    [Fact]
    public void FluxVaeDecoder_Decodes16ChannelLatents_ToValidRgb()
    {
        string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "models", "ae.safetensors");
        if (!File.Exists(path)) path = "models/ae.safetensors";

        if (File.Exists(path))
        {
            using var vae = Glacier.Inference.Image.Flux.FluxVaeDecoder.Open(path);

            int latentH = 8;
            int latentW = 8;
            int targetH = latentH * 8; // 64
            int targetW = latentW * 8; // 64

            float[] latents = new float[16 * latentH * latentW];
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] = (float)Math.Sin(i * 0.1);
            }

            byte[] rgb = new byte[targetH * targetW * 3];
            vae.Decode(latents, latentH, latentW, rgb);

            Assert.Equal(targetH * targetW * 3, rgb.Length);
            bool anyNonZero = false;
            for (int i = 0; i < rgb.Length; i++)
            {
                if (rgb[i] > 0) anyNonZero = true;
            }
            Assert.True(anyNonZero, "VAE output must contain non-zero pixel data");
        }
    }
}
