using System;
using System.IO;
using System.Linq;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests;

public class UniversalArchitectureTests
{
    private const string Llama3Path = @"D:\lmstudio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    private const string DeepSeekPath = @"D:\lmstudio\models\lmstudio-community\DeepSeek-Coder-V2-Lite-Instruct-GGUF\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf";
    private const string Phi4Path = @"D:\lmstudio\models\lmstudio-community\phi-4-GGUF\phi-4-Q4_K_M.gguf";
    private const string DevstralPath = @"D:\lmstudio\models\lmstudio-community\Devstral-Small-2505-GGUF\Devstral-Small-2505-Q4_K_M.gguf";

    [Fact]
    public void UniversalArchitecture_Detection_ValidatesAllFamilies()
    {
        if (File.Exists(Llama3Path))
        {
            using var gguf = GgufFile.Open(Llama3Path);
            var weights = new ModelWeights(gguf);
            Assert.Equal(UniversalArchitecture.Llama, weights.ArchitectureFamily);
            Assert.True(weights.HasRopeFreqs, "Llama 3.1 must have precomputed rope_freqs");
        }

        if (File.Exists(DeepSeekPath))
        {
            using var gguf = GgufFile.Open(DeepSeekPath);
            var weights = new ModelWeights(gguf);
            Assert.Equal(UniversalArchitecture.DeepSeek, weights.ArchitectureFamily);
            Assert.True(weights.IsMla, "DeepSeek must have MLA attention");
        }

        if (File.Exists(Phi4Path))
        {
            using var gguf = GgufFile.Open(Phi4Path);
            var weights = new ModelWeights(gguf);
            Assert.Equal(UniversalArchitecture.Phi, weights.ArchitectureFamily);
            Assert.True(weights.Layers[0].HasFusedQkv, "Phi-4 must have fused QKV");
            Assert.True(weights.Layers[0].HasFusedGateUp, "Phi-4 must have fused Gate/Up SwiGLU");
        }

        if (File.Exists(DevstralPath))
        {
            using var gguf = GgufFile.Open(DevstralPath);
            var weights = new ModelWeights(gguf);
            Assert.Equal(UniversalArchitecture.Mistral, weights.ArchitectureFamily);
            Assert.True(weights.RopeFreqBase >= 1e8f, "Devstral must have 10^9 RoPE freq base");
        }
    }

    [Fact]
    public void Llama3_GeneratesCoherentLogits()
    {
        if (!File.Exists(Llama3Path)) return;

        using var gguf = GgufFile.Open(Llama3Path);
        var weights = new ModelWeights(gguf);
        var tokenizer = new BpeTokenizer(gguf);
        using var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128, vHeadDim: weights.ValueDim);
        using var model = new Qwen2Model(weights, maxSeqLen: 128);

        string prompt = tokenizer.FormatChatML("What is the capital of France?");
        int[] tokens = tokenizer.Encode(prompt);
        float[] logits = new float[weights.VocabSize];

        model.ForwardBatch(tokens, 0, logits.AsSpan(), kvCache, computeLogits: true);

        var top5 = logits
            .Select((logit, id) => (logit, id))
            .OrderByDescending(x => x.logit)
            .Take(5)
            .ToList();

        Console.WriteLine($"\n[LLAMA 3.1 INSTRUCT PREDICTIONS] Prompt: '{prompt}'");
        foreach (var (logit, id) in top5)
        {
            Console.WriteLine($"  - '{tokenizer.DecodeToken(id)}' (id {id}, logit {logit:F2})");
        }

        Assert.Contains(top5, t => tokenizer.DecodeToken(t.id).Contains("Paris") || tokenizer.DecodeToken(t.id).Contains("The"));
    }

    [Fact]
    public void Phi4_FusedQkvGateUp_GeneratesCoherently()
    {
        if (!File.Exists(Phi4Path)) return;

        using var gguf = GgufFile.Open(Phi4Path);
        var weights = new ModelWeights(gguf);
        var tokenizer = new BpeTokenizer(gguf);
        using var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128, vHeadDim: weights.ValueDim);
        using var model = new Qwen2Model(weights, maxSeqLen: 128);

        Assert.True(weights.Layers[0].HasFusedQkv);
        Assert.True(weights.Layers[0].HasFusedGateUp);

        string prompt = tokenizer.FormatChatML("What is the capital of France?");
        int[] tokens = tokenizer.Encode(prompt);
        float[] logits = new float[weights.VocabSize];

        model.ForwardBatch(tokens, 0, logits.AsSpan(), kvCache, computeLogits: true);

        var top5 = logits
            .Select((logit, id) => (logit, id))
            .OrderByDescending(x => x.logit)
            .Take(5)
            .ToList();

        Console.WriteLine($"\n[PHI-4 PREDICTIONS] Prompt: '{prompt}'");
        foreach (var (logit, id) in top5)
        {
            Console.WriteLine($"  - '{tokenizer.DecodeToken(id)}' (id {id}, logit {logit:F2})");
        }

        Assert.NotEmpty(top5);
        Assert.Contains(top5, t => tokenizer.DecodeToken(t.id).Contains("Paris") || tokenizer.DecodeToken(t.id).Contains("The"));
    }

    [Fact]
    public void Devstral_HighRope_GeneratesCoherently()
    {
        if (!File.Exists(DevstralPath)) return;

        using var gguf = GgufFile.Open(DevstralPath);
        var weights = new ModelWeights(gguf);
        var tokenizer = new BpeTokenizer(gguf);
        using var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128, vHeadDim: weights.ValueDim);
        using var model = new Qwen2Model(weights, maxSeqLen: 128);

        string prompt = tokenizer.FormatChatML("What is the capital of France?");
        int[] tokens = tokenizer.Encode(prompt);
        float[] logits = new float[weights.VocabSize];

        model.ForwardBatch(tokens, 0, logits.AsSpan(), kvCache, computeLogits: true);

        var top5 = logits
            .Select((logit, id) => (logit, id))
            .OrderByDescending(x => x.logit)
            .Take(5)
            .ToList();

        Console.WriteLine($"\n[DEVSTRAL PREDICTIONS] Prompt: '{prompt}'");
        foreach (var (logit, id) in top5)
        {
            Console.WriteLine($"  - '{tokenizer.DecodeToken(id)}' (id {id}, logit {logit:F2})");
        }

        Assert.NotEmpty(top5);
        Assert.Contains(top5, t => tokenizer.DecodeToken(t.id).Contains("Paris") || tokenizer.DecodeToken(t.id).Contains("The"));
    }
}
