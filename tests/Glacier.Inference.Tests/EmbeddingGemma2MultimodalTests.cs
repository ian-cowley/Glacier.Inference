namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using Glacier.Inference.Embedding;
using Xunit;
using Xunit.Abstractions;

/// <summary>
/// End-to-end multimodal validation: Audio (Conformer) and Video (multi-frame) against Hugging Face golden references.
/// </summary>
[Collection("SequentialGpu")]
public class EmbeddingGemma2MultimodalTests
{
    private readonly ITestOutputHelper _output;
    public EmbeddingGemma2MultimodalTests(ITestOutputHelper output) => _output = output;

    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "TestData", "multimodal", name);

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

    private static EmbeddingGemma2Model Load(EmbeddingDevice dev = EmbeddingDevice.Cpu, int adapter = -1)
    {
        var m = EmbeddingGemma2Model.Load(EmbeddingGemma2ModelLocator.Path!, new EmbeddingGemma2Options { Device = dev, AdapterIndex = adapter });
        m.AttachMultimodalProjector(EmbeddingGemma2MmprojLocator.Path!);
        return m;
    }

    // ---------------------------------------------------------------- AUDIO TESTS

    [EmbeddingGemma2VisionFact]
    public void Audio_LogMelExtractor_MatchesReference()
    {
        float[] audio = ReadF32("audio_input.f32");
        float[] logMel = Gemma4AudioPreprocessor.ExtractLogMel(audio);
        float[] goldMel = ReadF32("audio_logmel.f32");

        Assert.Equal(goldMel.Length, logMel.Length);
        float maxDiff = 0;
        for (int i = 0; i < goldMel.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(logMel[i] - goldMel[i]));

        _output.WriteLine($"Audio log-mel: 199 frames x 128 bins, max abs diff vs reference: {maxDiff:E4}");
        Assert.True(maxDiff < 5e-3f, $"Log-mel max abs diff {maxDiff} exceeds tolerance.");
    }

    [EmbeddingGemma2VisionFact]
    public void Audio_SubSample_MatchesReference()
    {
        using var g = Glacier.Inference.Gguf.GgufFile.Open(EmbeddingGemma2MmprojLocator.Path!);
        using var tower = new Gemma4AudioTower(g);

        float[] goldMel = ReadF32("audio_logmel.f32");
        float[] sub = tower.Subsample(goldMel, goldMel.Length / 128, out int tSub);
        float[] goldSub = ReadF32("audio_subsample.f32");

        Assert.Equal(goldSub.Length, sub.Length);
        float worst = 1;
        for (int t = 0; t < tSub; t++)
            worst = Math.Min(worst, Cos(sub[(t * 1024)..((t + 1) * 1024)], goldSub[(t * 1024)..((t + 1) * 1024)]));

        _output.WriteLine($"Audio subsample: {tSub} steps x 1024, worst per-step cosine vs reference: {worst:F5}");
        Assert.True(worst > 0.99f, $"Audio subsample worst cosine {worst} < 0.99");
    }


    [EmbeddingGemma2VisionFact]
    public void Audio_SoftTokens_MatchReference()
    {
        using var g = Glacier.Inference.Gguf.GgufFile.Open(EmbeddingGemma2MmprojLocator.Path!);
        using var tower = new Gemma4AudioTower(g);

        float[] goldMel = ReadF32("audio_logmel.f32");
        float[] soft = tower.Encode(goldMel, goldMel.Length / 128);
        float[] goldSoft = ReadF32("audio_soft_tokens.f32");

        Assert.Equal(goldSoft.Length, soft.Length);
        int n = soft.Length / 512;
        float worst = 1;
        float sumCos = 0;
        var sb = new System.Text.StringBuilder();
        for (int t = 0; t < n; t++)
        {
            float c = Cos(soft[(t * 512)..((t + 1) * 512)], goldSoft[(t * 512)..((t + 1) * 512)]);
            worst = Math.Min(worst, c);
            sumCos += c;
            sb.Append($"{t}:{c:F3} ");
        }

        float meanCos = sumCos / n;
        _output.WriteLine($"Audio soft tokens: {n} tokens, mean: {meanCos:F5}, worst: {worst:F5}\n{sb}");
        Assert.True(meanCos > 0.88f, $"Audio mean per-token cosine {meanCos} < 0.88");
        Assert.True(worst > 0.40f, $"Audio worst per-token cosine {worst} < 0.40");
    }

    [EmbeddingGemma2VisionFact]
    public void Audio_Embedding_MatchesReference_OnCpu()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        Assert.True(m.SupportsAudio);

        float[] audio = ReadF32("audio_input.f32");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] emb = m.EmbedAudio(audio);
        long ms = sw.ElapsedMilliseconds;

        float[] gold = Golden("audio");
        float cos = Cos(emb, gold);
        _output.WriteLine($"Audio embedding on CPU: cosine vs HF reference {cos:F5} ({ms} ms)");

        Assert.InRange(emb.Sum(x => x * x), 0.999f, 1.001f);
        Assert.True(cos > 0.985f, $"Audio cosine {cos} < 0.985");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Audio_Embedding_MatchesReference_OnEachGpu(int adapter)
    {
        if (!OperatingSystem.IsWindows()) return;
        if (EmbeddingGemma2ModelLocator.Path is null || EmbeddingGemma2MmprojLocator.Path is null) return;
        try
        {
            using var m = Load(EmbeddingDevice.D3D12, adapter);
            float[] audio = ReadF32("audio_input.f32");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] emb = m.EmbedAudio(audio);
            long ms = sw.ElapsedMilliseconds;

            float[] gold = Golden("audio");
            float cos = Cos(emb, gold);
            _output.WriteLine($"Audio embedding on GPU {adapter} ({m.BackendName}): cosine {cos:F5} ({ms} ms)");

            Assert.InRange(emb.Sum(x => x * x), 0.999f, 1.001f);
            Assert.True(cos > 0.985f, $"Audio cosine on GPU {cos} < 0.985");
        }
        catch (PlatformNotSupportedException) { }
        catch (InvalidOperationException e) when (e.Message.Contains("No hardware DirectX 12")) { }
    }

    // ---------------------------------------------------------------- VIDEO TESTS

    [EmbeddingGemma2VisionFact]
    public void Video_Embedding_MatchesReference_OnCpu()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        Assert.True(m.SupportsImages);

        byte[] rawVid = File.ReadAllBytes(Data("video_input.u8"));
        int frames = 3, h = 96, w = 128;
        int frameBytes = h * w * 3;
        var frameList = new byte[frames][];
        for (int f = 0; f < frames; f++)
        {
            frameList[f] = new byte[frameBytes];
            Array.Copy(rawVid, f * frameBytes, frameList[f], 0, frameBytes);
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        float[] emb = m.EmbedVideo(frameList, w, h);
        long ms = sw.ElapsedMilliseconds;

        float[] gold = Golden("video");
        float cos = Cos(emb, gold);
        _output.WriteLine($"Video embedding on CPU (3 frames of 96x128): cosine vs HF reference {cos:F5} ({ms} ms)");

        Assert.InRange(emb.Sum(x => x * x), 0.999f, 1.001f);
        Assert.True(cos > 0.99f, $"Video cosine {cos} < 0.99");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Video_Embedding_MatchesReference_OnEachGpu(int adapter)
    {
        if (!OperatingSystem.IsWindows()) return;
        if (EmbeddingGemma2ModelLocator.Path is null || EmbeddingGemma2MmprojLocator.Path is null) return;
        try
        {
            using var m = Load(EmbeddingDevice.D3D12, adapter);
            byte[] rawVid = File.ReadAllBytes(Data("video_input.u8"));
            int frames = 3, h = 96, w = 128;
            int frameBytes = h * w * 3;
            var frameList = new byte[frames][];
            for (int f = 0; f < frames; f++)
            {
                frameList[f] = new byte[frameBytes];
                Array.Copy(rawVid, f * frameBytes, frameList[f], 0, frameBytes);
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] emb = m.EmbedVideo(frameList, w, h);
            long ms = sw.ElapsedMilliseconds;

            float[] gold = Golden("video");
            float cos = Cos(emb, gold);
            _output.WriteLine($"Video embedding on GPU {adapter} ({m.BackendName}): cosine {cos:F5} ({ms} ms)");

            Assert.InRange(emb.Sum(x => x * x), 0.999f, 1.001f);
            Assert.True(cos > 0.99f, $"Video cosine on GPU {cos} < 0.99");
        }
        catch (PlatformNotSupportedException) { }
        catch (InvalidOperationException e) when (e.Message.Contains("No hardware DirectX 12")) { }
    }

    // ---------------------------------------------------------------- CROSS-MODAL PARITY & VALIDATION

    [EmbeddingGemma2VisionFact]
    public void Multimodal_CrossModalEmbeddings_AreDistinctAndWellFormed()
    {
        using var m = Load(EmbeddingDevice.Cpu);
        byte[] img = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "TestData", "vision", "img224x224_input.u8"));
        float[] audio = ReadF32("audio_input.f32");

        float[] eImg = m.EmbedImage(img, 224, 224);
        float[] eAudio = m.EmbedAudio(audio);
        float[] eText = m.Embed("a photo of a red circle on a gradient");

        Assert.True(Cos(eImg, eAudio) < 0.95f, "Image and audio should not be identical embeddings.");
        Assert.True(Cos(eImg, eText) < 0.95f, "Image and text should not be identical embeddings.");
    }
}
