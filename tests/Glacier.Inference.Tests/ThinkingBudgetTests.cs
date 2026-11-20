namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Inference.Cli;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

public sealed class ThinkingBudgetTests : IDisposable
{
    private readonly string _tempDir;

    public ThinkingBudgetTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "GlacierThinkBudget_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
        }
        catch
        {
        }
    }

    [Theory]
    [InlineData(4096, 2662)]
    [InlineData(2048, 1331)]
    [InlineData(1024, 665)]
    [InlineData(512, 332)]
    [InlineData(256, 166)]
    [InlineData(64, 41)]
    [InlineData(12, 7)]
    [InlineData(1, 0)]
    [InlineData(0, 0)]
    public void ComputeAutoMaxReasoningTokens_ReservesFinalAnswerHeadroom(int maxTokens, int expectedBudget)
    {
        int actual = ReasoningBudgetPolicy.ComputeAutoMaxReasoningTokens(maxTokens);
        Assert.Equal(expectedBudget, actual);
        if (maxTokens > 1)
        {
            Assert.True(actual < maxTokens, $"Reasoning budget ({actual}) must leave room for final answer within MaxTokens ({maxTokens}).");
        }
    }

    [Theory]
    [InlineData("auto", ReasoningBudgetPolicy.AutoBudgetSentinel, true)]
    [InlineData("AUTO", ReasoningBudgetPolicy.AutoBudgetSentinel, true)]
    [InlineData(null, ReasoningBudgetPolicy.AutoBudgetSentinel, true)]
    [InlineData("off", 0, true)]
    [InlineData("none", 0, true)]
    [InlineData("0", 0, true)]
    [InlineData("750", 750, true)]
    [InlineData("-5", 0, false)]
    [InlineData("invalid", 0, false)]
    public void TryParseSetting_ParsesAllSupportedModes(string? raw, int expectedSetting, bool expectedOk)
    {
        bool ok = ReasoningBudgetPolicy.TryParseSetting(raw, out int setting);
        Assert.Equal(expectedOk, ok);
        if (expectedOk)
        {
            Assert.Equal(expectedSetting, setting);
        }
    }

    [Fact]
    public void ParseSamplingOptions_SupportsRootNestedAndDefaultThinkBudget()
    {
        using var defaultDoc = JsonDocument.Parse("""{"max_tokens": 2048}""");
        var autoOpts = Program.ParseSamplingOptions(defaultDoc.RootElement, ReasoningBudgetPolicy.AutoBudgetSentinel);
        Assert.Equal(1331, autoOpts.MaxReasoningTokens);

        var offOpts = Program.ParseSamplingOptions(defaultDoc.RootElement, defaultMaxThinkTokens: 0);
        Assert.Equal(0, offOpts.MaxReasoningTokens);

        using var overrideDoc = JsonDocument.Parse("""{"max_tokens": 1000, "max_thinking_tokens": 250}""");
        var overrideOpts = Program.ParseSamplingOptions(overrideDoc.RootElement, ReasoningBudgetPolicy.AutoBudgetSentinel);
        Assert.Equal(250, overrideOpts.MaxReasoningTokens);

        using var nestedDoc = JsonDocument.Parse("""{"options": {"num_predict": 512, "reasoning_budget": "off"}}""");
        var nestedOpts = Program.ParseSamplingOptions(nestedDoc.RootElement, ReasoningBudgetPolicy.AutoBudgetSentinel);
        Assert.Equal(512, nestedOpts.MaxTokens);
        Assert.Equal(0, nestedOpts.MaxReasoningTokens);
    }

    [Fact]
    public void ReasoningBudgetTracker_TracksPromptOpenAndTriggersOnceAtLimit()
    {
        const int thinkId = 300;
        const int endThinkId = 301;
        var tracker = new ReasoningBudgetTracker(thinkId, endThinkId, maxReasoningTokens: 3, promptTokens: new[] { 10, thinkId, 20 });
        Assert.True(tracker.InThink);

        Assert.False(tracker.ObserveTokenAndShouldForceClose(21)); // 1
        Assert.False(tracker.ObserveTokenAndShouldForceClose(22)); // 2
        Assert.True(tracker.ObserveTokenAndShouldForceClose(23));  // 3 -> trigger close!
        Assert.True(tracker.ForcedThinkClose);
        Assert.False(tracker.InThink);

        // Subsequent tokens must never re-trigger forced close.
        Assert.False(tracker.ObserveTokenAndShouldForceClose(24));
        Assert.False(tracker.ObserveTokenAndShouldForceClose(thinkId));
    }

    [Fact]
    public void ReasoningBudgetTracker_ResetsWhenModelClosesNaturally()
    {
        const int thinkId = 300;
        const int endThinkId = 301;
        var tracker = new ReasoningBudgetTracker(thinkId, endThinkId, maxReasoningTokens: 2, promptTokens: ReadOnlySpan<int>.Empty);
        Assert.False(tracker.InThink);

        Assert.False(tracker.ObserveTokenAndShouldForceClose(thinkId));
        Assert.True(tracker.InThink);
        Assert.False(tracker.ObserveTokenAndShouldForceClose(42)); // 1
        Assert.False(tracker.ObserveTokenAndShouldForceClose(endThinkId)); // closed before limit!
        Assert.False(tracker.InThink);
        Assert.False(tracker.ForcedThinkClose);

        // Generating answer tokens after natural </think> must not count toward thinking budget.
        for (int i = 0; i < 10; i++)
        {
            Assert.False(tracker.ObserveTokenAndShouldForceClose(50 + i));
        }
    }

    [Fact]
    public void ReasoningBudgetTracker_MultiTokenBpeThinkTags_CapsAtLimit()
    {
        var tracker = new ReasoningBudgetTracker(-1, -1, maxReasoningTokens: 3, promptTokens: ReadOnlySpan<int>.Empty);
        var sb = new StringBuilder();

        string[] openPieces = ["<", "think", ">"];
        foreach (string piece in openPieces)
        {
            sb.Append(piece);
            Assert.False(tracker.ObserveDecodedTextAndShouldForceClose(sb));
        }
        Assert.True(tracker.InThink);

        sb.Append("first");
        Assert.False(tracker.ObserveDecodedTextAndShouldForceClose(sb)); // 1
        sb.Append(" second");
        Assert.False(tracker.ObserveDecodedTextAndShouldForceClose(sb)); // 2
        sb.Append(" third");
        Assert.True(tracker.ObserveDecodedTextAndShouldForceClose(sb));  // 3 -> force close
        Assert.True(tracker.ForcedThinkClose);
        Assert.False(tracker.InThink);

        // Must never re-trigger after forced close
        sb.Append("\n</think>\n\n<think>again");
        Assert.False(tracker.ObserveDecodedTextAndShouldForceClose(sb));
    }

    [Fact]
    public void ReasoningBudgetTracker_MultiTokenBpeNaturalCloseAndPromptPrefill()
    {
        var trackerNatural = new ReasoningBudgetTracker(-1, -1, maxReasoningTokens: 4, promptTokens: ReadOnlySpan<int>.Empty);
        var sb = new StringBuilder();
        foreach (string piece in new[] { "<th", "ink>", "reason", "</", "think", ">", "final", " answer", " tokens", " here" })
        {
            sb.Append(piece);
            Assert.False(trackerNatural.ObserveDecodedTextAndShouldForceClose(sb));
            if (piece == "</think>" || piece == ">")
            {
                // Closed naturally after '>'
            }
        }
        Assert.False(trackerNatural.InThink);
        Assert.False(trackerNatural.ForcedThinkClose);

        // Prompt prefilled inside <think>
        var trackerPrefill = new ReasoningBudgetTracker(-1, -1, maxReasoningTokens: 2, promptTokens: ReadOnlySpan<int>.Empty, promptText: "User: hi\nAssistant: <think>\nstep1");
        Assert.True(trackerPrefill.InThink);
        var sb2 = new StringBuilder(" step2");
        Assert.False(trackerPrefill.ObserveDecodedTextAndShouldForceClose(sb2)); // 1
        sb2.Append(" step3");
        Assert.True(trackerPrefill.ObserveDecodedTextAndShouldForceClose(sb2));  // 2 -> force close
    }

    [Fact]
    public async Task InferenceSession_AutoClosesRunawayThinkAndContinuesToFinalAnswer()
    {
        string ggufPath = Path.Combine(_tempDir, "think_budget_synthetic.gguf");
        WriteSyntheticThinkLoopQwen2Gguf(ggufPath, out int nlTokenId, out int thinkTokenId, out int endThinkTokenId, out int xTokenId, out int answerTokenId, out int imEndTokenId);

        using var session = new InferenceSession(ggufPath, maxSeqLen: 64, device: "cpu", engine: InferenceEngineType.Cpu);

        var streamed = new List<string>();
        var opts = new SamplingOptions
        {
            MaxTokens = 12,
            MaxReasoningTokens = 3,
            Temperature = 0f,
            TopK = 1,
            RepetitionPenalty = 1.0f,
            ThinkTokenId = thinkTokenId,
            EndThinkTokenId = endThinkTokenId
        };

        // Prompt ends inside <think>; zeroed attention/FFN keeps hidden state equal to last token embedding,
        // so model keeps emitting 'x' inside <think> until budget=3 forces "\n</think>\n\n", after which
        // the '\n' embedding routes to 'A' -> '<|im_end|>'.
        var result = await session.GenerateAsync("<think>hi", opts, formatChat: false, onToken: t => streamed.Add(t));

        Assert.Equal("xxx\n</think>\n\nA", result.Text);
        Assert.Equal("xxx\n</think>\n\nA", string.Concat(streamed));
        Assert.StartsWith("stop(", result.FinishReason);
        Assert.True(nlTokenId >= 0 && xTokenId >= 0 && answerTokenId >= 0 && imEndTokenId >= 0);
    }

    private static void WriteSyntheticThinkLoopQwen2Gguf(
        string path,
        out int nlTokenId,
        out int thinkTokenId,
        out int endThinkTokenId,
        out int xTokenId,
        out int answerTokenId,
        out int imEndTokenId)
    {
        const int dim = 64;
        const int ffnDim = 128;
        const int headCount = 2;
        const int headCountKv = 1;
        const int headDim = dim / headCount;
        const int qDim = headCount * headDim;
        const int kvDim = headCountKv * headDim;
        const int vocabSize = 288;

        string[] vocab = BuildGpt2ByteVocab(vocabSize);
        thinkTokenId = vocabSize - 4;
        endThinkTokenId = vocabSize - 3;
        imEndTokenId = vocabSize - 1;
        vocab[thinkTokenId] = "<think>";
        vocab[endThinkTokenId] = "</think>";
        vocab[vocabSize - 2] = "<|im_start|>";
        vocab[imEndTokenId] = "<|im_end|>";

        // Locate GPT-2 byte tokens for '\n', 'x', and 'A'
        nlTokenId = Array.IndexOf(vocab, ((char)266).ToString()); // byte 10 -> 256 + 10
        xTokenId = Array.IndexOf(vocab, "x");
        answerTokenId = Array.IndexOf(vocab, "A");

        float[] tokenEmbd = new float[dim * vocabSize];
        for (int tok = 0; tok < vocabSize; tok++)
        {
            int basis = tok == nlTokenId || tok == endThinkTokenId ? 1 : tok == answerTokenId ? 2 : 0;
            tokenEmbd[tok * dim + basis] = 8.0f;
        }

        // GGUF stores [inDim=dim, outDim=vocabSize], row-major by token row: outWeight[tok * dim + d]
        float[] outWeight = new float[dim * vocabSize];
        for (int tok = 0; tok < vocabSize; tok++)
        {
            outWeight[tok * dim + 0] = tok == xTokenId ? 5.0f : -5.0f;
            outWeight[tok * dim + 1] = tok == answerTokenId ? 5.0f : -5.0f;
            outWeight[tok * dim + 2] = tok == imEndTokenId ? 5.0f : -5.0f;
        }

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("qwen2.block_count", GgufValueType.Uint32, 1u),
            ("qwen2.context_length", GgufValueType.Uint32, 64u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)imEndTokenId),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, vocab)
        };

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", new[] { (ulong)dim, (ulong)vocabSize }, tokenEmbd),
            ("output_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("output.weight", new[] { (ulong)dim, (ulong)vocabSize }, outWeight),
            ("blk.0.attn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.attn_q.weight", new[] { (ulong)dim, (ulong)qDim }, new float[dim * qDim]),
            ("blk.0.attn_k.weight", new[] { (ulong)dim, (ulong)kvDim }, new float[dim * kvDim]),
            ("blk.0.attn_v.weight", new[] { (ulong)dim, (ulong)kvDim }, new float[dim * kvDim]),
            ("blk.0.attn_output.weight", new[] { (ulong)qDim, (ulong)dim }, new float[qDim * dim]),
            ("blk.0.ffn_norm.weight", new[] { (ulong)dim }, ConstantFloats(dim, 1.0f)),
            ("blk.0.ffn_gate.weight", new[] { (ulong)dim, (ulong)ffnDim }, new float[dim * ffnDim]),
            ("blk.0.ffn_up.weight", new[] { (ulong)dim, (ulong)ffnDim }, new float[dim * ffnDim]),
            ("blk.0.ffn_down.weight", new[] { (ulong)ffnDim, (ulong)dim }, new float[ffnDim * dim])
        };

        WriteF32Gguf(path, metadata, tensors);
    }

    private static string[] BuildGpt2ByteVocab(int vocabSize)
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var byteToToken = new string[256];
        for (int i = 0; i < 256; i++)
        {
            byteToToken[bs[i]] = ((char)cs[i]).ToString();
        }

        string[] vocab = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i < 256) vocab[i] = byteToToken[i];
            else vocab[i] = $"<|extra_{i}|>";
        }
        return vocab;
    }

    private static void WriteF32Gguf(
        string path,
        List<(string Key, GgufValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, float[] Data)> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x46554747U);
        bw.Write(3U);
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, val) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case GgufValueType.String:
                    WriteGgufString(bw, (string)val);
                    break;
                case GgufValueType.Uint32:
                    bw.Write((uint)val);
                    break;
                case GgufValueType.Float32:
                    bw.Write((float)val);
                    break;
                case GgufValueType.Array:
                    var arr = (string[])val;
                    bw.Write((uint)GgufValueType.String);
                    bw.Write((ulong)arr.Length);
                    for (int i = 0; i < arr.Length; i++) WriteGgufString(bw, arr[i]);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported GGUF metadata type {type}");
            }
        }

        ulong runningOffset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++) bw.Write(dims[d]);
            bw.Write((uint)GgufType.F32);
            bw.Write(runningOffset);

            ulong byteLen = (ulong)data.Length * sizeof(float);
            runningOffset = AlignUp(runningOffset + byteLen, 32UL);
        }

        long headerEnd = fs.Position;
        long alignedStart = (long)AlignUp((ulong)headerEnd, 32UL);
        while (fs.Position < alignedStart) bw.Write((byte)0);

        for (int t = 0; t < tensors.Count; t++)
        {
            float[] data = tensors[t].Data;
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
            long alignedNext = (long)AlignUp((ulong)fs.Position - (ulong)alignedStart, 32UL) + alignedStart;
            while (fs.Position < alignedNext) bw.Write((byte)0);
        }
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        ulong rem = value % alignment;
        return rem == 0 ? value : value + (alignment - rem);
    }

    private static float[] ConstantFloats(int count, float value)
    {
        float[] arr = new float[count];
        Array.Fill(arr, value);
        return arr;
    }
}
