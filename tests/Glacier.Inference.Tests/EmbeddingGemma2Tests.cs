namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using Glacier.Inference.Embedding;
using Glacier.Inference.Engine;
using Xunit;
using Xunit.Abstractions;

/// <summary>
/// Locates (or downloads once) the EmbeddingGemma 2 Q8_0 GGUF. Override with GLACIER_EMBEDDINGGEMMA2_PATH;
/// set GLACIER_EMBED_NO_DOWNLOAD=1 to forbid network access.
/// </summary>
internal static class EmbeddingGemma2ModelLocator
{
    private const string Url = "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-Q8_0.gguf";

    private static readonly Lazy<string?> Resolved = new(Resolve);
    public static string? Path => Resolved.Value;

    private static string? Resolve()
    {
        string? env = Environment.GetEnvironmentVariable("GLACIER_EMBEDDINGGEMMA2_PATH");
        if (!string.IsNullOrWhiteSpace(env)) return File.Exists(env) ? env : null;

        string dir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Glacier", "test-models");
        string file = System.IO.Path.Combine(dir, "embeddinggemma-2-Q8_0.gguf");
        if (File.Exists(file)) return file;
        if (Environment.GetEnvironmentVariable("GLACIER_EMBED_NO_DOWNLOAD") == "1") return null;

        try
        {
            Directory.CreateDirectory(dir);
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(20) };
            using var resp = http.GetAsync(Url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            resp.EnsureSuccessStatusCode();
            string tmp = file + ".part";
            using (var src = resp.Content.ReadAsStream())
            using (var dst = File.Create(tmp))
                src.CopyTo(dst);
            File.Move(tmp, file, overwrite: true);
            return file;
        }
        catch (Exception)
        {
            return null;
        }
    }
}

/// <summary>Skips when the EmbeddingGemma 2 weights cannot be found or downloaded.</summary>
public sealed class EmbeddingGemma2FactAttribute : FactAttribute
{
    public EmbeddingGemma2FactAttribute()
    {
        if (EmbeddingGemma2ModelLocator.Path is null)
            Skip = "EmbeddingGemma 2 GGUF not available (set GLACIER_EMBEDDINGGEMMA2_PATH or allow download).";
    }
}

public sealed class EmbeddingGemma2Fixture : IDisposable
{
    public EmbeddingGemma2Model Model { get; }

    public EmbeddingGemma2Fixture()
    {
        string? path = EmbeddingGemma2ModelLocator.Path;
        Model = path is null ? null! : EmbeddingGemma2Model.Load(path);
    }

    public void Dispose() => Model?.Dispose();
}

/// <summary>
/// Validation of the native EmbeddingGemma 2 encoder. Golden vectors in <c>TestData/embeddinggemma2_golden.json</c>
/// were produced by the reference Hugging Face implementation (FP32, mean pooling + L2 norm) from
/// <c>google/embeddinggemma-2</c>; the Q8_0 GGUF is expected to track them to within quantisation noise.
/// </summary>
[Collection("SequentialGpu")]
public class EmbeddingGemma2Tests : IClassFixture<EmbeddingGemma2Fixture>
{
    private const string QueryPrefix = "task: search result | query: ";
    private const string DocPrefix = "title: none | text: ";

    private readonly EmbeddingGemma2Model _model;
    private readonly ITestOutputHelper _output;

    public EmbeddingGemma2Tests(EmbeddingGemma2Fixture fixture, ITestOutputHelper output)
    {
        _model = fixture.Model;
        _output = output;
    }

    // ---------------------------------------------------------------- golden data

    private sealed record GoldenEntry(string Text, int[] Ids, float[] Embedding)
    {
        public string Expanded => ExpandText(Text);
    }

    private static string ExpandText(string t) => t != "@LONG" ? t : string.Join(" ",
        Enumerable.Range(0, 150).Select(i => $"Sentence number {i}: the quick brown fox jumps over the lazy dog near river bank {i % 7}."));

    private static GoldenEntry[] LoadGolden()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "embeddinggemma2_golden.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.EnumerateArray().Select(e => new GoldenEntry(
            e.GetProperty("text").GetString()!,
            e.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
            e.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray())).ToArray();
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
        return (float)s;
    }

    private static float Norm(ReadOnlySpan<float> a) => MathF.Sqrt(Dot(a, a));

    // ---------------------------------------------------------------- model / structure

    [EmbeddingGemma2Fact]
    public void Model_ReportsExpectedArchitecture()
    {
        Assert.Equal(768, _model.EmbeddingDimension);
        Assert.Equal(512, _model.HiddenSize);
        Assert.Equal(24, _model.LayerCount);
        Assert.Equal(262144, _model.Tokenizer.VocabSize);
        Assert.Equal(2, _model.Tokenizer.BosTokenId);
        Assert.Equal(1, _model.Tokenizer.EosTokenId);
        Assert.True(_model.Tokenizer.AddBos && _model.Tokenizer.AddEos);
    }

    [EmbeddingGemma2Fact]
    public void InferenceSession_RejectsEncoderOnlyModelWithClearMessage()
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => new InferenceSession(EmbeddingGemma2ModelLocator.Path!, maxSeqLen: 128, device: "cpu"));
        Assert.Contains("EmbeddingGemma2Model", ex.Message);
    }

    // ---------------------------------------------------------------- tokenizer parity

    [EmbeddingGemma2Fact]
    public void Tokenizer_MatchesReferenceIds_ForAllGoldenInputs()
    {
        foreach (var g in LoadGolden())
        {
            int[] ours = _model.Tokenizer.Encode(g.Expanded);
            Assert.True(g.Ids.SequenceEqual(ours),
                $"Token mismatch for '{(g.Text.Length > 40 ? g.Text[..40] : g.Text)}': " +
                $"expected {g.Ids.Length} ids, got {ours.Length}; first diff at {FirstDiff(g.Ids, ours)}");
        }
    }

    private static int FirstDiff(int[] a, int[] b)
    {
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (a[i] != b[i]) return i;
        return Math.Min(a.Length, b.Length);
    }

    [EmbeddingGemma2Fact]
    public void Tokenizer_HandlesByteFallbackAndControlTokens()
    {
        var tok = _model.Tokenizer;
        int[] ids = tok.Encode("<bos>hi<eos>", addSpecialTokens: false);
        Assert.Equal(tok.BosTokenId, ids[0]);
        Assert.Equal(tok.EosTokenId, ids[^1]);

        int[] emoji = tok.Encode("🙂", addSpecialTokens: false);
        Assert.NotEmpty(emoji);
        Assert.DoesNotContain(tok.UnkTokenId, emoji);

        int[] empty = tok.Encode("");
        Assert.Equal([tok.BosTokenId, tok.EosTokenId], empty);
    }

    // ---------------------------------------------------------------- numerical parity with reference

    [EmbeddingGemma2Fact]
    public void Embeddings_MatchReferenceImplementation()
    {
        double minCos = 1;
        foreach (var g in LoadGolden())
        {
            float[] ours = _model.Embed(g.Expanded);
            Assert.Equal(768, ours.Length);
            Assert.InRange(Norm(ours), 0.9999f, 1.0001f);

            float cos = Dot(ours, g.Embedding);
            minCos = Math.Min(minCos, cos);
            _output.WriteLine($"cos={cos:F6}  tokens={g.Ids.Length,5}  '{(g.Text.Length > 50 ? g.Text[..50] : g.Text).Replace("\n", "\\n")}'");
            Assert.True(cos > 0.998f, $"Cosine vs reference only {cos:F6} for '{(g.Text.Length > 40 ? g.Text[..40] : g.Text)}'");
        }
        _output.WriteLine($"min cosine = {minCos:F6}");
    }

    [EmbeddingGemma2Fact]
    public void LongInput_BeyondSlidingWindow_MatchesReference()
    {
        var g = LoadGolden().Single(e => e.Text == "@LONG");
        Assert.True(g.Ids.Length > 1100, "Golden long input must exceed the 2x512 sliding window.");

        float cos = Dot(_model.Embed(g.Expanded), g.Embedding);
        _output.WriteLine($"long-input cosine = {cos:F6} over {g.Ids.Length} tokens");
        Assert.True(cos > 0.998f, $"Long input diverges from reference: cos={cos:F6}");
    }

    [EmbeddingGemma2Fact]
    public void MaxTokens_TruncatesAndPreservesEos()
    {
        string longText = ExpandText("@LONG");
        float[] full = _model.Embed(longText);

        int old = _model.MaxTokens;
        try
        {
            _model.MaxTokens = 64;
            float[] truncated = _model.Embed(longText);
            Assert.InRange(Norm(truncated), 0.9999f, 1.0001f);
            Assert.All(truncated, x => Assert.True(float.IsFinite(x)));
            Assert.True(Dot(full, truncated) < 0.9999f, "Truncation to 64 tokens should change the embedding.");
        }
        finally { _model.MaxTokens = old; }
    }

    // ---------------------------------------------------------------- semantic behaviour

    [EmbeddingGemma2Fact]
    public void Retrieval_RanksRelevantDocumentFirst()
    {
        string[] docs =
        [
            "Sourdough is made from flour, water and a fermented starter culture that leavens the dough.",
            "The central bank raised interest rates to curb inflation.",
            "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
            "The goalkeeper saved a penalty in the final minute of the match.",
        ];
        (string Query, int Expected)[] cases =
        [
            ("How do I bake sourdough bread at home?", 0),
            ("Why did borrowing costs go up recently?", 1),
            ("How do plants make energy from light?", 2),
            ("Who won the football game at the last second?", 3),
        ];

        var docVecs = docs.Select(d => _model.Embed(DocPrefix + d)).ToArray();
        foreach (var (query, expected) in cases)
        {
            float[] q = _model.Embed(QueryPrefix + query);
            float[] scores = docVecs.Select(d => Dot(q, d)).ToArray();
            int best = Array.IndexOf(scores, scores.Max());
            _output.WriteLine($"{query} -> doc {best} ({string.Join(", ", scores.Select(s => s.ToString("F3")))})");
            Assert.Equal(expected, best);

            float runnerUp = scores.Where((_, i) => i != best).Max();
            Assert.True(scores[best] - runnerUp > 0.03f, "Winning document should clear the runner-up by a margin.");
        }
    }

    [EmbeddingGemma2Fact]
    public void Paraphrases_ScoreHigherThanUnrelated_WithMargin()
    {
        float[] cat = _model.Embed("A cat is sleeping on the sofa.");
        float[] kitten = _model.Embed("A kitten naps on the couch.");
        float[] finance = _model.Embed("Quarterly interest rates affect bond yields.");

        float related = Dot(cat, kitten), unrelated = Dot(cat, finance);
        _output.WriteLine($"related={related:F3} unrelated={unrelated:F3}");
        Assert.True(related - unrelated > 0.15f);
    }

    [EmbeddingGemma2Fact]
    public void CrossLingual_TranslationsAreCloserThanUnrelatedText()
    {
        float[] en = _model.Embed("The cat is sleeping on the sofa.");
        float[] fr = _model.Embed("Le chat dort sur le canapé.");
        float[] ja = _model.Embed("猫がソファで寝ています。");
        float[] other = _model.Embed("The central bank raised interest rates.");

        Assert.True(Dot(en, fr) > Dot(en, other) + 0.1f, $"en/fr={Dot(en, fr):F3} en/other={Dot(en, other):F3}");
        Assert.True(Dot(en, ja) > Dot(en, other) + 0.05f, $"en/ja={Dot(en, ja):F3} en/other={Dot(en, other):F3}");
    }

    [EmbeddingGemma2Fact]
    public void WordOrderAndNegation_ProduceDistinctEmbeddings()
    {
        float[] a = _model.Embed("The dog bit the man.");
        float[] b = _model.Embed("The man bit the dog.");
        float sim = Dot(a, b);
        Assert.InRange(sim, 0.5f, 0.9999f); // related, but not identical -> position/context is encoded
    }

    // ---------------------------------------------------------------- Matryoshka

    [EmbeddingGemma2Fact]
    public void Matryoshka_TruncationEqualsRenormalisedPrefix_AndTracksReference()
    {
        var g = LoadGolden().First();
        float[] full = _model.Embed(g.Expanded);

        foreach (int dim in new[] { 512, 256, 128 })
        {
            float[] small = _model.Embed(g.Expanded, dim);
            Assert.Equal(dim, small.Length);
            Assert.InRange(Norm(small), 0.9999f, 1.0001f);

            float prefixNorm = Norm(full.AsSpan(0, dim));
            for (int i = 0; i < dim; i++)
                Assert.Equal(full[i] / prefixNorm, small[i], 4);

            // Reference truncated + renormalised
            float[] refSmall = g.Embedding.AsSpan(0, dim).ToArray();
            float rn = Norm(refSmall);
            for (int i = 0; i < dim; i++) refSmall[i] /= rn;
            Assert.True(Dot(small, refSmall) > 0.995f, $"dim {dim} diverges from reference");
        }
    }

    [EmbeddingGemma2Fact]
    public void Matryoshka_PreservesRetrievalRanking()
    {
        string[] docs =
        [
            "Sourdough is made from flour, water and a fermented starter culture.",
            "The central bank raised interest rates to curb inflation.",
            "The goalkeeper saved a penalty in the final minute.",
        ];
        foreach (int dim in new[] { 768, 256, 128 })
        {
            var q = _model.Embed(QueryPrefix + "How do I make bread with a starter?", dim);
            var scores = docs.Select(d => Dot(q, _model.Embed(DocPrefix + d, dim))).ToArray();
            Assert.Equal(0, Array.IndexOf(scores, scores.Max()));
        }
    }

    // ---------------------------------------------------------------- robustness / invariants

    [EmbeddingGemma2Fact]
    public void Embedding_IsDeterministic_AndFreeOfCrossCallState()
    {
        float[] a1 = _model.Embed("Deterministic embedding check.");
        _ = _model.Embed("Some other, much longer text that should not leak any state into the next call at all.");
        float[] a2 = _model.Embed("Deterministic embedding check.");
        for (int i = 0; i < a1.Length; i++) Assert.Equal(a1[i], a2[i], 5);
    }

    [EmbeddingGemma2Fact]
    public void Embed_SupportsConcurrentCallers()
    {
        string[] texts = Enumerable.Range(0, 8).Select(i => $"Concurrent request number {i} about topic {i * 7}.").ToArray();
        float[][] serial = texts.Select(t => _model.Embed(t)).ToArray();
        float[][] parallel = new float[texts.Length][];
        System.Threading.Tasks.Parallel.For(0, texts.Length, i => parallel[i] = _model.Embed(texts[i]));
        for (int i = 0; i < texts.Length; i++)
            Assert.True(Dot(serial[i], parallel[i]) > 0.99999f, $"Concurrent result {i} differs from serial result.");
    }

    [EmbeddingGemma2Fact]
    public void EdgeInputs_ProduceFiniteUnitVectors()
    {
        string[] inputs = ["", " ", "\n\n\n", "x", new string('a', 5000), "🙂🙂🙂", "<unused0> weird", "\u0000\u0001 control chars"];
        foreach (var s in inputs)
        {
            float[] e = _model.Embed(s);
            Assert.All(e, x => Assert.True(float.IsFinite(x), $"non-finite value for input '{s.Replace("\n", "\\n")}'"));
            Assert.InRange(Norm(e), 0.9999f, 1.0001f);
        }
    }

    [EmbeddingGemma2Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => _model.Embed(ReadOnlySpan<int>.Empty, new float[768]));
        Assert.Throws<ArgumentOutOfRangeException>(() => _model.Embed("x", new float[769]));
        Assert.Throws<ArgumentOutOfRangeException>(() => _model.Embed("x", Span<float>.Empty));
        Assert.Throws<ArgumentOutOfRangeException>(() => _model.Embed(new[] { 0, 262144 }, new float[768]));
    }

    [EmbeddingGemma2Fact]
    public void DissimilarTexts_AreNotCollapsed_EmbeddingsAreDiverse()
    {
        string[] texts =
        [
            "Photosynthesis in plants", "Rust ownership and borrowing", "Medieval castle architecture",
            "Quantum entanglement experiments", "Recipe for chocolate cake", "Stock market volatility",
        ];
        var vecs = texts.Select(t => _model.Embed(t)).ToArray();
        for (int i = 0; i < vecs.Length; i++)
        {
            for (int j = i + 1; j < vecs.Length; j++)
                Assert.True(Dot(vecs[i], vecs[j]) < 0.9f, $"'{texts[i]}' and '{texts[j]}' are suspiciously similar (embedding collapse).");
        }
    }
}
