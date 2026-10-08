namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using System.Text;
using Glacier.Inference.Gguf;
using Xunit;

[Collection("SequentialGpu")]
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
    public void Open_ThrowsInvalidDataException_OnEmptyOrTruncatedFile()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_trunc_{Guid.NewGuid():N}.gguf");
        try
        {
            // 0 bytes
            File.WriteAllBytes(tempPath, Array.Empty<byte>());
            Assert.Throws<InvalidDataException>(() => GgufFile.Open(tempPath));

            // Truncated header (only 12 bytes, less than minimum header 24 bytes)
            File.WriteAllBytes(tempPath, new byte[12]);
            Assert.Throws<InvalidDataException>(() => GgufFile.Open(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Open_ThrowsInvalidDataException_OnInvalidMagic()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_badmagic_{Guid.NewGuid():N}.gguf");
        try
        {
            byte[] bytes = new byte[32];
            BitConverter.GetBytes(0x12345678U).CopyTo(bytes, 0); // wrong magic
            BitConverter.GetBytes(3U).CopyTo(bytes, 4); // v3
            File.WriteAllBytes(tempPath, bytes);

            Assert.Throws<InvalidDataException>(() => GgufFile.Open(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Open_ThrowsNotSupportedException_OnUnsupportedVersion()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_badver_{Guid.NewGuid():N}.gguf");
        try
        {
            byte[] bytes = new byte[32];
            BitConverter.GetBytes(0x46554747U).CopyTo(bytes, 0); // 'GGUF'
            BitConverter.GetBytes(999U).CopyTo(bytes, 4); // unsupported version
            File.WriteAllBytes(tempPath, bytes);

            Assert.Throws<NotSupportedException>(() => GgufFile.Open(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Open_ThrowsInvalidDataException_OnCorruptedCountsExceedingFileSize()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_corrupt_count_{Guid.NewGuid():N}.gguf");
        try
        {
            byte[] bytes = new byte[40];
            BitConverter.GetBytes(0x46554747U).CopyTo(bytes, 0); // 'GGUF'
            BitConverter.GetBytes(3U).CopyTo(bytes, 4);          // version 3
            BitConverter.GetBytes(1_000_000UL).CopyTo(bytes, 8); // Huge tensor count with tiny file
            BitConverter.GetBytes(1_000_000UL).CopyTo(bytes, 16); // Huge KV count with tiny file
            File.WriteAllBytes(tempPath, bytes);

            Assert.Throws<InvalidDataException>(() => GgufFile.Open(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Open_ThrowsInvalidDataException_WhenTensorBoundsExceedFileSize()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_bad_tensor_{Guid.NewGuid():N}.gguf");
        try
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);

            // Header
            bw.Write(0x46554747U); // 'GGUF'
            bw.Write(3U);          // Version 3
            bw.Write(1UL);         // TensorCount = 1
            bw.Write(0UL);         // MetadataKvCount = 0

            // Tensor info
            string name = "test.weight";
            bw.Write((ulong)name.Length);
            bw.Write(Encoding.UTF8.GetBytes(name));
            bw.Write(1U);          // nDims = 1
            bw.Write(1024UL);      // dim0 = 1024 elements
            bw.Write(0U);          // ggmlType = F32 (4 bytes per elem -> 4096 bytes)
            bw.Write(100_000UL);   // Offset = 100,000 (well beyond small stream)

            File.WriteAllBytes(tempPath, ms.ToArray());

            Assert.Throws<InvalidDataException>(() => GgufFile.Open(tempPath));
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void Open_ValidatesAndParsesWellFormedMinimalGguf()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"gguf_valid_min_{Guid.NewGuid():N}.gguf");
        try
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);

            // Header
            bw.Write(0x46554747U); // 'GGUF'
            bw.Write(3U);          // Version 3
            bw.Write(1UL);         // TensorCount = 1
            bw.Write(1UL);         // MetadataKvCount = 1

            // KV: "general.architecture" -> "llama" (GgufValueType.String = 8)
            string key = "general.architecture";
            bw.Write((ulong)key.Length);
            bw.Write(Encoding.UTF8.GetBytes(key));
            bw.Write(8U);          // String type
            string val = "llama";
            bw.Write((ulong)val.Length);
            bw.Write(Encoding.UTF8.GetBytes(val));

            // Tensor info
            string tName = "token_embd.weight";
            bw.Write((ulong)tName.Length);
            bw.Write(Encoding.UTF8.GetBytes(tName));
            bw.Write(1U);          // nDims = 1
            bw.Write(4UL);         // 4 elements
            bw.Write(0U);          // F32 (16 bytes)
            bw.Write(0UL);         // offset = 0

            // Alignment padding
            long currentPos = ms.Position;
            long rem = currentPos % 32;
            long pad = rem == 0 ? 0 : 32 - rem;
            for (int i = 0; i < pad; i++) bw.Write((byte)0);

            // Tensor raw data: 4 floats (16 bytes)
            for (int i = 0; i < 4; i++) bw.Write((float)i);

            File.WriteAllBytes(tempPath, ms.ToArray());

            using var gguf = GgufFile.Open(tempPath);
            Assert.Equal("llama", gguf.Architecture);
            Assert.Equal(1UL, gguf.TensorCount);
            Assert.True(gguf.Tensors.ContainsKey("token_embd.weight"));
            var t = gguf.Tensors["token_embd.weight"];
            Assert.Equal(16UL, t.GetByteSize());

            unsafe
            {
                float* p = (float*)gguf.GetTensorPointer(t);
                Assert.Equal(0f, p[0]);
                Assert.Equal(1f, p[1]);
                Assert.Equal(2f, p[2]);
                Assert.Equal(3f, p[3]);
            }
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
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
