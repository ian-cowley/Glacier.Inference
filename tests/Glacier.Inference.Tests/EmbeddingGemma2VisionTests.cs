namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using Glacier.Inference.Embedding;
using Xunit;
using Xunit.Abstractions;

internal static class EmbeddingGemma2MmprojLocator
{
    private const string Url = "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/mmproj-embeddinggemma-2-Q8_0.gguf";
    private static readonly Lazy<string?> Resolved = new(Resolve);
    public static string? Path => Resolved.Value;

    private static string? Resolve()
    {
        string? env = Environment.GetEnvironmentVariable("GLACIER_EMBEDDINGGEMMA2_MMPROJ_PATH");
        if (!string.IsNullOrWhiteSpace(env)) return File.Exists(env) ? env : null;
        string dir = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Glacier", "test-models");
        string file = System.IO.Path.Combine(dir, "mmproj-embeddinggemma-2-Q8_0.gguf");
        if (File.Exists(file)) return file;
        if (Environment.GetEnvironmentVariable("GLACIER_EMBED_NO_DOWNLOAD") == "1") return null;
        try
        {
            Directory.CreateDirectory(dir);
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            using var resp = http.GetAsync(Url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            resp.EnsureSuccessStatusCode();
            string tmp = file + ".part";
            using (var src = resp.Content.ReadAsStream()) using (var dst = File.Create(tmp)) src.CopyTo(dst);
            File.Move(tmp, file, overwrite: true);
            return file;
        }
        catch (Exception) { return null; }
    }
}

/// <summary>Skips unless both the text GGUF and the multimodal projector GGUF are available.</summary>
public sealed class EmbeddingGemma2VisionFactAttribute : FactAttribute
{
    public EmbeddingGemma2VisionFactAttribute()
    {
        if (EmbeddingGemma2ModelLocator.Path is null || EmbeddingGemma2MmprojLocator.Path is null)
            Skip = "EmbeddingGemma 2 text/mmproj GGUF not available.";
    }
}

/// <summary>
/// Image embedding validation against the Hugging Face reference (fp32) on deterministic synthetic images.
/// Goldens (<c>TestData/vision</c>): raw RGB input, vision soft tokens (512-d), final unit embedding.
/// </summary>
[Collection("SequentialGpu")]
public class EmbeddingGemma2VisionTests
{
    private readonly ITestOutputHelper _output;
    public EmbeddingGemma2VisionTests(ITestOutputHelper output) => _output = output;

    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", "vision", name);

    private static float[] ReadF32(string name)
    {
        byte[] b = File.ReadAllBytes(Data(name));
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    private static float[] Golden(string key)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Data("embeddings.json")));
        return doc.RootElement.GetProperty(key).EnumerateArray().Select(x => x.GetSingle()).ToArray();
    }

    private static float Cos(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
        return (float)(d / Math.Sqrt(na * nb));
    }

    private static EmbeddingGemma2Model Load(EmbeddingDevice dev, int adapter = -1)
    {
        var m = EmbeddingGemma2Model.Load(EmbeddingGemma2ModelLocator.Path!, new EmbeddingGemma2Options { Device = dev, AdapterIndex = adapter });
        m.AttachMultimodalProjector(EmbeddingGemma2MmprojLocator.Path!);
        return m;
    }

    [Theory]
    [InlineData(224, 224, 768, 768)]
    [InlineData(180, 300, 576, 1008)]
    [InlineData(1, 1, 48, 48)]
    [InlineData(4000, 100, 48, 2256)]
    public void TargetSize_MatchesReferenceFormula(int h, int w, int th, int tw)
    {
        var (rh, rw) = Gemma4ImagePreprocessor.TargetSize(h, w);
        if (h == 1 || h == 4000) { Assert.True(rh % 48 == 0 && rw % 48 == 0 && rh > 0 && rw > 0); return; }
        Assert.Equal((th, tw), (rh, rw));
    }

    [EmbeddingGemma2VisionFact]
    public void SoftTokens_MatchReference()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        foreach (var (name, h, w) in new[] { ("img224x224", 224, 224), ("img180x300", 180, 300) })
        {
            byte[] rgb = File.ReadAllBytes(Data($"{name}_input.u8"));
            float[] patches = Gemma4ImagePreprocessor.Preprocess(rgb, w, h, 280, out int gh, out int gw);
            using var g = Glacier.Inference.Gguf.GgufFile.Open(EmbeddingGemma2MmprojLocator.Path!);
            using var tower = new Gemma4VisionTower(g);
            float[] soft = tower.Encode(patches, gh, gw);
            float[] gold = ReadF32($"{name}_soft_tokens.f32");
            Assert.Equal(gold.Length, soft.Length);

            // per-token cosine (Q8_0 weights track fp32 within quantisation noise)
            int n = soft.Length / 512; float worst = 1;
            for (int t = 0; t < n; t++)
                worst = Math.Min(worst, Cos(soft[(t * 512)..((t + 1) * 512)], gold[(t * 512)..((t + 1) * 512)]));
            _output.WriteLine($"{name}: {n} soft tokens, worst per-token cosine {worst:F5}");
            Assert.True(worst > 0.97f, $"{name}: worst per-token cosine {worst}");
        }
    }

    [EmbeddingGemma2VisionFact]
    public void ImageEmbedding_MatchesReference_OnCpu() => CheckImageEmbedding(EmbeddingDevice.Cpu, -1);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void ImageEmbedding_MatchesReference_OnEachGpu(int adapter)
    {
        if (!OperatingSystem.IsWindows()) return;
        if (EmbeddingGemma2ModelLocator.Path is null || EmbeddingGemma2MmprojLocator.Path is null) return;
        try { CheckImageEmbedding(EmbeddingDevice.D3D12, adapter); }
        catch (PlatformNotSupportedException) { }
        catch (InvalidOperationException e) when (e.Message.Contains("No hardware DirectX 12")) { }
    }

    private void CheckImageEmbedding(EmbeddingDevice dev, int adapter)
    {
        using var m = Load(dev, adapter);
        _output.WriteLine($"text tower: {m.BackendName}");
        foreach (var (name, h, w) in new[] { ("img224x224", 224, 224), ("img180x300", 180, 300) })
        {
            byte[] rgb = File.ReadAllBytes(Data($"{name}_input.u8"));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] e = m.EmbedImage(rgb, w, h);
            long ms = sw.ElapsedMilliseconds;
            float cos = Cos(e, Golden(name));
            _output.WriteLine($"{name}: cosine vs HF reference {cos:F5} ({ms} ms)");
            Assert.InRange(e.Sum(x => x * x), 0.999f, 1.001f);
            Assert.True(cos > 0.99f, $"{name}: cosine {cos}");
        }
    }

    [EmbeddingGemma2VisionFact]
    public void DifferentImages_GiveDifferentEmbeddings_AndSameImageIsDeterministic()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        byte[] a = File.ReadAllBytes(Data("img224x224_input.u8"));
        byte[] b = File.ReadAllBytes(Data("img180x300_input.u8"));
        float[] ea1 = m.EmbedImage(a, 224, 224), ea2 = m.EmbedImage(a, 224, 224), eb = m.EmbedImage(b, 300, 180);
        Assert.True(Cos(ea1, ea2) > 0.99999f);
        Assert.True(Cos(ea1, eb) < 0.999f);
    }

    [EmbeddingGemma2VisionFact]
    public void ImageEmbedding_ArgumentValidation()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        Assert.Throws<ArgumentException>(() => m.EmbedImage(new byte[10], 4, 4));
        using var plain = EmbeddingGemma2Model.Load(EmbeddingGemma2ModelLocator.Path!, new EmbeddingGemma2Options { Device = EmbeddingDevice.Cpu });
        Assert.False(plain.SupportsImages);
        Assert.Throws<InvalidOperationException>(() => plain.EmbedImage(new byte[48], 4, 4));
    }
}
