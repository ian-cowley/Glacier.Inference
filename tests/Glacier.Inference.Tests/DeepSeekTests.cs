using System;
using System.IO;
using System.Linq;
using System.Text;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests;

[Collection("SequentialGpu")]
public class DeepSeekTests
{
    private const string ModelPath = @"D:\lmstudio\models\lmstudio-community\DeepSeek-Coder-V2-Lite-Instruct-GGUF\DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf";

    [Fact]
    public void DeepSeek_MlaMoe_GeneratesCoherently()
    {
        if (!File.Exists(ModelPath)) return;

        using var gguf = GgufFile.Open(ModelPath);
        var tokenizer = new BpeTokenizer(gguf);

        using var session = new InferenceSession(ModelPath, maxSeqLen: 512, device: "cpu", engine: InferenceEngineType.Cpu);
        var weights = session.Weights;
        var kvCache = new KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 512, vHeadDim: weights.ValueDim);
        var model = new Qwen2Model(weights, maxSeqLen: 512);

        // Test Chat Prompt: "What is 1 + 1?"
        string chatMath = tokenizer.FormatChatML("What is 1 + 1?");
        int[] mathTokens = tokenizer.Encode(chatMath);
        float[] mathLogits = new float[weights.VocabSize];

        model.ForwardBatch(mathTokens, 0, mathLogits.AsSpan(), kvCache, computeLogits: true);

        var mathTop5 = mathLogits
            .Select((logit, id) => (logit, id))
            .OrderByDescending(x => x.logit)
            .Take(5)
            .ToList();

        // Verify top token prediction is ' The'
        Assert.NotEmpty(mathTop5);
        string firstPiece = tokenizer.DecodeToken(mathTop5[0].id);
        Assert.Equal(" The", firstPiece);
        Assert.True(mathTop5[0].logit > 20.0f, $"Expected high logit confidence, got {mathTop5[0].logit}");

        // Autoregressive generation
        var outputBuilder = new StringBuilder();
        int currToken = mathTop5[0].id;
        outputBuilder.Append(firstPiece);
        int currPos = mathTokens.Length;
        bool hitEos = false;

        for (int step = 0; step < 25; step++)
        {
            model.Forward(currToken, currPos, kvCache, mathLogits.AsSpan(), computeLogits: true);
            currPos++;

            int nextToken = 0;
            float maxL = float.MinValue;
            for (int v = 0; v < weights.VocabSize; v++)
            {
                if (mathLogits[v] > maxL)
                {
                    maxL = mathLogits[v];
                    nextToken = v;
                }
            }

            currToken = nextToken;
            string piece = tokenizer.DecodeToken(currToken);
            outputBuilder.Append(piece);

            if (currToken == tokenizer.EosTokenId || currToken == 100001)
            {
                hitEos = true;
                break;
            }
        }

        string generated = outputBuilder.ToString();
        Assert.True(hitEos, $"Expected EOS token to be emitted. Output so far: {generated}");
        Assert.Contains("2", generated);
    }
}
