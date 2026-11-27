namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Text;
using Glacier.Inference.Format;
using Glacier.Inference.Memory;
using Glacier.Inference.Tokenizer;
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
            using var vae = Glacier.Media.Image.Flux.FluxVaeDecoder.Open(path);

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

    [Fact]
    public void SafetensorsFile_Open_RejectsOversizedHeaderAndOutOfBoundsOffsets()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_st_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            string oversizedPath = Path.Combine(tempDir, "oversized.safetensors");
            using (var fs = File.Create(oversizedPath))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(300UL * 1024UL * 1024UL);
                bw.Write(new byte[16]);
            }
            Assert.Throws<InvalidDataException>(() => SafetensorsFile.Open(oversizedPath));

            string oobPath = Path.Combine(tempDir, "oob.safetensors");
            byte[] headerBytes = Encoding.UTF8.GetBytes("{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,9999]}}");
            using (var fs = File.Create(oobPath))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write((ulong)headerBytes.Length);
                bw.Write(headerBytes);
                bw.Write(new byte[16]);
            }
            Assert.Throws<InvalidDataException>(() => SafetensorsFile.Open(oobPath));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void BpeTokenizer_SentencePiece_RoundTripsSpaceAndByteFallback()
    {
        var vocab = new[]
        {
            "<unk>", "<s>", "</s>",
            "\u2581hello", "\u2581", "world",
            "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"
        };
        var tok = new BpeTokenizer(vocab, merges: null, isSentencePiece: true, eosTokenId: 2, bosTokenId: 1, addSpacePrefix: true);

        int[] ids = tok.Encode("hello\U0001F600");
        Assert.Contains(3, ids);
        Assert.Contains(6, ids);
        Assert.Contains(9, ids);

        string decoded = tok.Decode(ids);
        Assert.Contains("hello", decoded);
        Assert.Contains("\U0001F600", decoded);
    }

    [Fact]
    public void GpuResidentSlotManager_WhenAllocateVramTrueWithoutCuda_DoesNotFabricateFakePointers()
    {
        using var slotMgr = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 1024, allocateVram: true);
        bool allocated = slotMgr.AllocateOrEvictSlot(0, 0, out _, out _, out _, out IntPtr devPtr);
        Assert.True(allocated);
        if (!slotMgr.HasRealVram)
        {
            Assert.Equal(IntPtr.Zero, devPtr);
        }
        else
        {
            Assert.NotEqual(IntPtr.Zero, devPtr);
        }
    }
}
