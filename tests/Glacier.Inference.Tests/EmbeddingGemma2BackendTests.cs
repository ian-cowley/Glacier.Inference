namespace Glacier.Inference.Tests;

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using Glacier.Inference.Embedding;
using Xunit;
using Xunit.Abstractions;

/// <summary>Runs the golden/parity checks on every compute backend available on this machine (CPU SIMD, each D3D12 adapter).</summary>
[Collection("SequentialGpu")]
public class EmbeddingGemma2BackendTests
{
    private readonly ITestOutputHelper _output;
    public EmbeddingGemma2BackendTests(ITestOutputHelper output) => _output = output;

    private static EmbeddingGemma2Model? TryLoad(EmbeddingDevice dev, int adapter)
    {
        try
        {
            return EmbeddingGemma2Model.Load(EmbeddingGemma2ModelLocator.Path!,
                new EmbeddingGemma2Options { Device = dev, AdapterIndex = adapter });
        }
        catch (InvalidOperationException e) when (dev == EmbeddingDevice.D3D12 && e.Message.Contains("No hardware DirectX 12"))
        {
            Console.WriteLine("D3D12 adapter " + adapter + " failed: " + e);
            return null;
        }
    }

    public static TheoryData<EmbeddingDevice, int> Backends => new()
    {
        { EmbeddingDevice.Cpu, -1 },
        { EmbeddingDevice.D3D12, 0 },
        { EmbeddingDevice.D3D12, 1 },
        { EmbeddingDevice.D3D12, 2 },
    };

    private static float Dot(float[] a, float[] b) { double s = 0; for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i]; return (float)s; }

    private static (string text, float[] emb)[] Golden()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "embeddinggemma2_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray().Select(e => (
            e.GetProperty("text").GetString()!,
            e.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray())).ToArray();
    }

    private static string Expand(string t) => t != "@LONG" ? t : string.Join(" ",
        Enumerable.Range(0, 150).Select(i => $"Sentence number {i}: the quick brown fox jumps over the lazy dog near river bank {i % 7}."));

    [Theory]
    [MemberData(nameof(Backends))]
    public void Backend_MatchesGolden_AndReportsSpeed(EmbeddingDevice dev, int adapter)
    {
        if (EmbeddingGemma2ModelLocator.Path is null) return;
        using var model = TryLoad(dev, adapter);
        if (model is null) { _output.WriteLine($"adapter {adapter} not available - skipped"); return; }
        _output.WriteLine($"Backend: {model.BackendName}");

        float min = 1;
        foreach (var (text, golden) in Golden())
        {
            var sw = Stopwatch.StartNew();
            float cos = Dot(model.Embed(Expand(text)), golden);
            _output.WriteLine($"  {(text.Length > 30 ? text[..30] : text),-32} cos={cos:F6} {sw.ElapsedMilliseconds} ms");
            min = Math.Min(min, cos);
        }
        Assert.True(min > 0.998f, $"min cosine {min}");
    }

    [Theory]
    [MemberData(nameof(Backends))]
    public void Backend_HiddenStatesMatchCpu(EmbeddingDevice dev, int adapter)
    {
        if (EmbeddingGemma2ModelLocator.Path is null || dev == EmbeddingDevice.Cpu) return;
        using var model = TryLoad(dev, adapter);
        if (model is null) return;
        using var cpu = TryLoad(EmbeddingDevice.Cpu, -1)!;

        int[] ids = cpu.Tokenizer.Encode(Expand("@LONG"))[..1300]; // beyond the 2x512 window
        float[] a = model.ForwardHidden(ids), b = cpu.ForwardHidden(ids);
        Assert.Equal(b.Length, a.Length);
        int H = model.HiddenSize;
        float worst = 1;
        for (int t = 0; t < ids.Length; t++)
        {
            double dot = 0, na = 0, nb = 0;
            for (int d = 0; d < H; d++) { double x = a[t * H + d], y = b[t * H + d]; dot += x * y; na += x * x; nb += y * y; }
            worst = Math.Min(worst, (float)(dot / Math.Sqrt(na * nb)));
        }
        _output.WriteLine($"{model.BackendName}: worst per-token cosine vs CPU = {worst:F6}");
        Assert.True(worst > 0.999f);
    }
}

