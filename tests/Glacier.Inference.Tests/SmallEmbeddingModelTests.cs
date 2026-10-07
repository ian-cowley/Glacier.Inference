namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Glacier.Inference.Engine;
using Xunit;

/// <summary>
/// Semantic sanity tests for very small embedding models (EmbeddingGemma-class).
/// Models come from GLACIER_EMBED_MODELS (semicolon-separated GGUF paths) or are downloaded
/// on demand from Hugging Face into a local cache. Tests silently pass (skip) when no model is
/// available, e.g. offline. Set GLACIER_EMBED_NO_DOWNLOAD=1 to disable downloading.
/// </summary>
[Collection("SequentialGpu")]
public class SmallEmbeddingModelTests
{
    // Public, ungated small GGUF embedding model (~0.6B). No credentials needed.
    private const string DefaultModelUrl =
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf";

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Glacier", "test-models");

    public static TheoryData<string> ModelPaths()
    {
        var data = new TheoryData<string>();
        string? env = Environment.GetEnvironmentVariable("GLACIER_EMBED_MODELS");
        if (!string.IsNullOrWhiteSpace(env))
        {
            foreach (var p in env.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                data.Add(p);
        }
        else
        {
            data.Add(Path.Combine(CacheDir, Path.GetFileName(new Uri(DefaultModelUrl).LocalPath)));
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(ModelPaths))]
    public async Task SimilarTextsScoreHigherThanUnrelated(string modelPath)
    {
        if (!await EnsureModelAsync(modelPath)) return;

        using var session = new InferenceSession(modelPath, maxSeqLen: 256, device: "cpu");
        int dim = session.Weights.EmbeddingLength;

        float[] Embed(string text, PoolingStrategy pooling)
        {
            var v = new float[dim];
            session.ExtractEmbedding(text, v, pooling);
            return v;
        }

        foreach (var pooling in new[] { PoolingStrategy.MeanPooling, PoolingStrategy.LastToken })
        {
            var cat = Embed("A cat is sleeping on the sofa.", pooling);
            var kitten = Embed("A kitten naps on the couch.", pooling);
            var finance = Embed("Quarterly interest rates affect bond yields.", pooling);

            Assert.InRange(Norm(cat), 0.999f, 1.001f);
            Assert.All(cat, x => Assert.True(float.IsFinite(x)));

            float related = Dot(cat, kitten);
            float unrelated = Dot(cat, finance);
            Assert.True(related > unrelated,
                $"[{Path.GetFileName(modelPath)} / {pooling}] related={related:F3} should exceed unrelated={unrelated:F3}");
        }
    }

    private static float Dot(float[] a, float[] b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static float Norm(float[] a) => MathF.Sqrt(Dot(a, a));

    private static async Task<bool> EnsureModelAsync(string path)
    {
        if (File.Exists(path)) return true;
        if (Environment.GetEnvironmentVariable("GLACIER_EMBED_NO_DOWNLOAD") == "1") return false;
        if (!path.StartsWith(CacheDir, StringComparison.OrdinalIgnoreCase)) return false;

        try
        {
            Directory.CreateDirectory(CacheDir);
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
            using var resp = await http.GetAsync(DefaultModelUrl, HttpCompletionOption.ResponseHeadersRead);
            resp.EnsureSuccessStatusCode();
            string tmp = path + ".part";
            await using (var src = await resp.Content.ReadAsStreamAsync())
            await using (var dst = File.Create(tmp))
                await src.CopyToAsync(dst);
            File.Move(tmp, path, overwrite: true);
            return true;
        }
        catch (Exception)
        {
            return false; // offline / blocked: skip
        }
    }
}
