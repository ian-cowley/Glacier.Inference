using System;
using System.IO;
using System.Linq;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests;

public class Llama3DiscoveryTests
{
    private const string Llama3Path = @"D:\lmstudio\models\lmstudio-community\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

    [Fact]
    public void Llama3_Tokenizer_CanEncodeAndDecode()
    {
        if (!File.Exists(Llama3Path)) return;

        using var gguf = GgufFile.Open(Llama3Path);
        var tokenizer = new BpeTokenizer(gguf);

        Assert.True(tokenizer.VocabSize >= 128000);
        string sample = "Hello, how are you today?";
        int[] tokens = tokenizer.Encode(sample);
        Assert.NotEmpty(tokens);
        string decoded = string.Concat(tokens.Select(t => tokenizer.DecodeToken(t)));
        Assert.Equal(sample, decoded);
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

        string prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        int[] tokens = tokenizer.Encode(prompt);
        float[] logits = new float[weights.VocabSize];

        model.ForwardBatch(tokens, 0, logits.AsSpan(), kvCache, computeLogits: true);

        var top5 = logits
            .Select((logit, id) => (logit, id))
            .OrderByDescending(x => x.logit)
            .Take(5)
            .ToList();

        Console.WriteLine($"\n[LLAMA 3.1 INSTRUCT TOP 5 PREDICTIONS] Prompt: '{prompt}'");
        foreach (var (logit, id) in top5)
        {
            Console.WriteLine($"  - '{tokenizer.DecodeToken(id)}' (id {id}, logit {logit:F2})");
        }

        Assert.Contains(top5, t => tokenizer.DecodeToken(t.id).Contains("Paris"));
    }
}



