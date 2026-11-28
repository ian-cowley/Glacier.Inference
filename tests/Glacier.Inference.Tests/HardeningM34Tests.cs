namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Glacier.Inference.Cli;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Format;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;
using Glacier.Inference.Tests.VulkanE2E.Fixtures;
using Glacier.Inference.Tokenizer;
using Xunit;

public unsafe class HardeningM34Tests
{
    private sealed class TempDirectoryScope : IDisposable
    {
        public string Dir { get; }

        public TempDirectoryScope(string prefix)
        {
            Dir = Path.Combine(Path.GetTempPath(), $"{prefix}_{Guid.NewGuid():N}");
            Directory.CreateDirectory(Dir);
        }

        public void Dispose()
        {
            if (Directory.Exists(Dir))
            {
                try { Directory.Delete(Dir, recursive: true); } catch { }
            }
        }
    }

    private static void WriteGgufString(BinaryWriter bw, string value)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(value);
        bw.Write((ulong)utf8.Length);
        bw.Write(utf8);
    }

    private static ulong AlignUp(ulong val, ulong align) => (val + align - 1) & ~(align - 1);

    private static void WriteCustomGguf(
        string path,
        List<(string Key, GgufValueType Type, object Value)> metadata,
        List<(string Name, ulong[] Dims, GgufType Type, byte[] RawBytes)> tensors,
        uint alignment = 32)
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
                case GgufValueType.Int32:
                    bw.Write((int)val);
                    break;
                case GgufValueType.Float32:
                    bw.Write((float)val);
                    break;
                case GgufValueType.Array when val is string[] sArr:
                    bw.Write((uint)GgufValueType.String);
                    bw.Write((ulong)sArr.Length);
                    foreach (string s in sArr) WriteGgufString(bw, s);
                    break;
                case GgufValueType.Array when val is uint[] uArr:
                    bw.Write((uint)GgufValueType.Uint32);
                    bw.Write((ulong)uArr.Length);
                    foreach (uint u in uArr) bw.Write(u);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported synthetic value {type}");
            }
        }

        ulong runningOffset = 0;
        foreach (var (name, dims, type, bytes) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            foreach (ulong d in dims) bw.Write(d);
            bw.Write((uint)type);
            bw.Write(runningOffset);
            runningOffset = AlignUp(runningOffset + (ulong)bytes.Length, alignment);
        }

        ulong headerEnd = (ulong)fs.Position;
        ulong alignedStart = AlignUp(headerEnd, alignment);
        while ((ulong)fs.Position < alignedStart) bw.Write((byte)0);

        foreach (var (_, _, _, bytes) in tensors)
        {
            bw.Write(bytes);
            ulong relPos = (ulong)fs.Position - alignedStart;
            ulong alignedNext = AlignUp(relPos, alignment);
            while ((ulong)fs.Position - alignedStart < alignedNext) bw.Write((byte)0);
        }
    }

    private static void WriteSafetensorsFile(string path, string headerJson, byte[] payload)
    {
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);
        bw.Write((ulong)headerBytes.Length);
        bw.Write(headerBytes);
        bw.Write(payload);
    }

    // -------------------------------------------------------------------------
    // Item 4: VulkanContext ordinal alignment & leak-free failure
    // -------------------------------------------------------------------------

    [Fact]
    public void VulkanContext_ResolveHardwareDeviceOrdinal_SkipsSoftwareIcdsAndMapsDeterministically()
    {
        var devices = new (bool IsSoftware, string Name)[]
        {
            (true, "llvmpipe (LLVM 17.0.6, 256 bits)"),
            (false, "AMD Radeon RX 7900 XTX"),
            (true, "SwiftShader Device (Subzero)"),
            (false, "NVIDIA GeForce RTX 4090")
        };

        Assert.Equal(1, VulkanContext.ResolveHardwareDeviceOrdinal(devices, 0));
        Assert.Equal(3, VulkanContext.ResolveHardwareDeviceOrdinal(devices, 1));
        Assert.Equal(-1, VulkanContext.ResolveHardwareDeviceOrdinal(devices, 2));
        Assert.Equal(-1, VulkanContext.ResolveHardwareDeviceOrdinal(devices, -1));

        var softwareOnly = new (bool IsSoftware, string Name)[]
        {
            (true, "llvmpipe"),
            (true, "lavapipe")
        };
        Assert.Equal(-1, VulkanContext.ResolveHardwareDeviceOrdinal(softwareOnly, 0));
    }

    // -------------------------------------------------------------------------
    // Item 5: GgufFile & SafetensorsFile validation, split shards, zero-copy
    // -------------------------------------------------------------------------

    [Fact]
    public void GgufFile_NonPowerOfTwoAlignment_ThrowsInvalidDataException()
    {
        using var tmp = new TempDirectoryScope("gguf_align");
        string path = Path.Combine(tmp.Dir, "bad_align.gguf");
        var meta = new List<(string, GgufValueType, object)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("general.alignment", GgufValueType.Uint32, 24u),
            ("tokenizer.ggml.tokens", GgufValueType.Array, new[] { "a", "b" })
        };
        var tensors = new List<(string, ulong[], GgufType, byte[])>
        {
            ("t0", [4UL], GgufType.F32, new byte[16])
        };
        WriteCustomGguf(path, meta, tensors, alignment: 32);

        Assert.Throws<InvalidDataException>(() =>
        {
            using var _ = new GgufFile(path);
        });
    }

    [Fact]
    public void GgufFile_SplitShards_LoadsBothShards_AndCorruptSecondShardReleasesAllHandles()
    {
        using var tmp = new TempDirectoryScope("gguf_shards");
        string shard1 = Path.Combine(tmp.Dir, "test-model-00001-of-00002.gguf");
        string shard2 = Path.Combine(tmp.Dir, "test-model-00002-of-00002.gguf");

        byte[] t1Data = new byte[16];
        BitConverter.TryWriteBytes(t1Data.AsSpan(0, 4), 1.25f);
        byte[] t2Data = new byte[16];
        BitConverter.TryWriteBytes(t2Data.AsSpan(0, 4), 2.50f);

        var meta = new List<(string, GgufValueType, object)>
        {
            ("general.architecture", GgufValueType.String, "llama"),
            ("tokenizer.ggml.tokens", GgufValueType.Array, new[] { "x", "y" })
        };
        WriteCustomGguf(shard1, meta, [("tensor_a", [4UL], GgufType.F32, t1Data)]);
        WriteCustomGguf(shard2, [], [("tensor_b", [4UL], GgufType.F32, t2Data)]);

        using (var gguf = new GgufFile(shard1))
        {
            Assert.True(gguf.Tensors.ContainsKey("tensor_a"));
            Assert.True(gguf.Tensors.ContainsKey("tensor_b"));
            float* pa = (float*)gguf.GetTensorPointer("tensor_a");
            float* pb = (float*)gguf.GetTensorPointer("tensor_b");
            Assert.Equal(1.25f, pa[0]);
            Assert.Equal(2.50f, pb[0]);
        }

        // Now corrupt shard 2 and ensure open throws AND leaves zero leaked file handles
        File.WriteAllBytes(shard2, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        Assert.Throws<InvalidDataException>(() =>
        {
            using var _ = new GgufFile(shard1);
        });

        File.Delete(shard1);
        File.Delete(shard2);
        Assert.False(File.Exists(shard1));
        Assert.False(File.Exists(shard2));
    }

    [Fact]
    public void SafetensorsFile_ZeroCopyParse_ValidatesOverflowRankOffsets_AndDisposedAccess()
    {
        using var tmp = new TempDirectoryScope("st_hardening");

        string validPath = Path.Combine(tmp.Dir, "valid.safetensors");
        byte[] payload = new byte[8];
        BitConverter.TryWriteBytes(payload.AsSpan(0, 4), 3.5f);
        BitConverter.TryWriteBytes(payload.AsSpan(4, 4), -1.25f);
        WriteSafetensorsFile(validPath, "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", payload);

        var st = SafetensorsFile.Open(validPath);
        Assert.True(st.Tensors.TryGetValue("w", out var info));
        Assert.Equal(2L, info!.ElementCount);
        float* ptr = (float*)st.GetTensorPointer("w");
        Assert.Equal(3.5f, ptr[0]);
        Assert.Equal(-1.25f, ptr[1]);
        st.Dispose();
        Assert.Throws<ObjectDisposedException>(() => st.GetTensorPointer("w"));

        // Overflowing ElementCount check (2147483647^2 * 4 exceeds long.MaxValue)
        var hugeInfo = new SafetensorsTensorInfo("big", "F32", [int.MaxValue, int.MaxValue, 4], 0, 8);
        Assert.Throws<OverflowException>(() => _ = hugeInfo.ElementCount);

        // Excess rank > 16 check
        string highRankPath = Path.Combine(tmp.Dir, "high_rank.safetensors");
        string shape17 = "[" + string.Join(',', Enumerable.Repeat("1", 17)) + "]";
        WriteSafetensorsFile(highRankPath, $"{{\"bad\":{{\"dtype\":\"F32\",\"shape\":{shape17},\"data_offsets\":[0,4]}}}}", new byte[4]);
        Assert.Throws<InvalidDataException>(() =>
        {
            using var _ = SafetensorsFile.Open(highRankPath);
        });

        // Mismatched data_offsets length vs dtype * element count
        string badLenPath = Path.Combine(tmp.Dir, "bad_len.safetensors");
        WriteSafetensorsFile(badLenPath, "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}", payload);
        Assert.Throws<InvalidDataException>(() =>
        {
            using var _ = SafetensorsFile.Open(badLenPath);
        });
    }

    // -------------------------------------------------------------------------
    // Item 7: Dynamic EOS/EOT arrays without hardcoded Qwen IDs + Session backend
    // -------------------------------------------------------------------------

    [Fact]
    public void GgufFile_And_BpeTokenizer_UseConfiguredEosArray_AndDoNotHardcodeQwenStopIds()
    {
        using var tmp = new TempDirectoryScope("dyn_eos");
        string path = Path.Combine(tmp.Dir, "custom_eos.gguf");
        const int vocabSize = 151650;
        var tokens = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++) tokens[i] = $"<t{i}>";
        tokens[7] = "<stop_a>";
        tokens[9] = "<stop_b>";
        tokens[11] = "<eot_custom>";
        tokens[151643] = "normal_vocab_151643";
        tokens[151645] = "normal_vocab_151645";

        var meta = new List<(string, GgufValueType, object)>
        {
            ("general.architecture", GgufValueType.String, "llama"),
            ("tokenizer.ggml.tokens", GgufValueType.Array, tokens),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Array, new uint[] { 7u, 9u }),
            ("tokenizer.ggml.eot_token_id", GgufValueType.Uint32, 11u),
        };
        WriteCustomGguf(path, meta, []);

        using var gguf = new GgufFile(path);
        Assert.Equal(7, gguf.EosTokenId);
        Assert.Equal(new[] { 7, 9, 11 }, gguf.GetEosTokenIds());

        var tok = new BpeTokenizer(gguf);
        Assert.Equal(7, tok.EosTokenId);
        Assert.True(tok.IsStopToken(7));
        Assert.True(tok.IsStopToken(9));
        Assert.True(tok.IsStopToken(11));
        Assert.False(tok.IsStopToken(151643));
        Assert.False(tok.IsStopToken(151645));
    }

    [Fact]
    public void InferenceSession_DirectMlEngineFlag_ReportsGpuAcceleratedAndDirect3D12Backend()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();
        using var session = new InferenceSession(handle.Path, device: "cpu");

        Assert.Equal("Cpu", session.ActiveBackend);
        Assert.False(session.IsGpuAccelerated);

        typeof(InferenceSession).GetProperty(nameof(InferenceSession.Engine), BindingFlags.Public | BindingFlags.Instance)!
            .SetValue(session, InferenceEngineType.DirectML);

        Assert.Equal("Direct3D12", session.ActiveBackend);
        Assert.True(session.IsGpuAccelerated);
    }

    // -------------------------------------------------------------------------
    // Items 8 & 11: SanitizePrompt coverage, Ctor 2 _specialTokens, O(n log n) BPE
    // -------------------------------------------------------------------------

    [Fact]
    public void BpeTokenizer_SanitizePrompt_EscapesAllModelFamilyControlTokens()
    {
        string[] dangerousTokens =
        [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "<|thought|>", "<|/thought|>",
            "[INST]", "[/INST]", "<s>", "</s>", "<unk>", "[TOOL_CALLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]",
            "<｜ref｜>", "<｜/ref｜>", "<｜det｜>", "<｜/det｜>", "<｜grounding｜>", "<｜/grounding｜>", "<｜polygon｜>", "<｜/polygon｜>"
        ];

        string combined = string.Join(" payload ", dangerousTokens);
        string sanitized = BpeTokenizer.SanitizePrompt(combined);

        foreach (string tok in dangerousTokens)
        {
            Assert.DoesNotContain(tok, sanitized, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void BpeTokenizer_DirectVocabCtor_IsolatesSpecialTokens_AndSupportsFastBpeMerges()
    {
        // Build >10k vocab to ensure ctor 2 does NOT mark ordinary vocab entries as special tokens
        var vocab = new List<string>(10050);
        for (int i = 0; i < 256; i++)
        {
            vocab.Add(BpeTokenizerBytesString((byte)i));
        }
        vocab.Add("he");
        vocab.Add("ll");
        vocab.Add("lo");
        vocab.Add("hello");
        int imEndId = vocab.Count;
        vocab.Add("<|im_end|>");
        for (int i = vocab.Count; i < 10050; i++)
        {
            vocab.Add($"tok_{i}");
        }

        var merges = new List<string>
        {
            "h e",
            "l l",
            "he ll",
            "hell o"
        };
        // Include intermediate merged tokens in vocab for standard BPE chain
        vocab[vocab.IndexOf("tok_300")] = "hell";

        var byteBpe = new BpeTokenizer(vocab, merges, isSentencePiece: false, eosTokenId: imEndId, bosTokenId: -1, addSpacePrefix: false);

        // Normal token matching substring "tok_500" must NOT behave like a special token boundary
        int[] normalTokens = byteBpe.Encode("tok_500");
        Assert.True(normalTokens.Length >= 4, "Normal non-special vocab entry should remain BPE-tokenized, not matched as a special token");

        // Explicit special token MUST be emitted intact
        int[] specialTokens = byteBpe.Encode("hello<|im_end|>");
        Assert.Equal(imEndId, specialTokens[^1]);

        // Performance check: 60,000-character input under O(n log n) priority-queue BPE
        string largeInput = string.Concat(Enumerable.Repeat("hello ", 10000));
        var sw = Stopwatch.StartNew();
        int[] encodedLarge = byteBpe.Encode(largeInput);
        sw.Stop();
        Assert.True(encodedLarge.Length > 0);
        Assert.True(sw.ElapsedMilliseconds < 2500, $"Byte-level BPE took too long: {sw.ElapsedMilliseconds}ms");
        Assert.Equal(largeInput, byteBpe.Decode(encodedLarge));

        // SentencePiece path check with large input
        var spVocab = new List<string> { "<unk>", "▁", "a", "b", "ab", "▁a", "▁ab" };
        var spMerges = new List<string> { "▁ a", "a b", "▁a b" };
        var spTok = new BpeTokenizer(spVocab, spMerges, isSentencePiece: true, eosTokenId: 0, bosTokenId: -1, addSpacePrefix: false);
        string spInput = string.Concat(Enumerable.Repeat(" ab", 15000));
        sw.Restart();
        int[] spEncoded = spTok.Encode(spInput);
        sw.Stop();
        Assert.Equal(15000, spEncoded.Length);
        Assert.True(sw.ElapsedMilliseconds < 2500, $"SentencePiece BPE took too long: {sw.ElapsedMilliseconds}ms");
    }

    private static string BpeTokenizerBytesString(byte b)
    {
        // Mirrors GPT-2 byte-to-unicode table in BpeTokenizer
        int codepoint;
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
        {
            codepoint = b;
        }
        else
        {
            int shift = 0;
            for (int i = 0; i < b; i++)
            {
                if (!((i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF)))
                    shift++;
            }
            codepoint = 256 + shift;
        }
        return char.ConvertFromUtf32(codepoint);
    }

    // -------------------------------------------------------------------------
    // Item 9: KVCache wraparound validity tracking
    // -------------------------------------------------------------------------

    [Fact]
    public void KVCache_RingBuffer_RejectsEvictedReads_AndAllowsActiveWindow()
    {
        using var kv = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 4, isRingBuffer: true);

        float[] k = new float[4];
        float[] v = new float[4];
        for (int pos = 0; pos < 6; pos++)
        {
            for (int d = 0; d < 4; d++)
            {
                k[d] = pos * 10 + d + 1;
                v[d] = pos * 100 + d + 1;
            }
            fixed (float* pk = k, pv = v)
            {
                kv.Store(0, pos, pk, pv);
            }
        }

        Assert.False(kv.IsPositionValid(0, 0));
        Assert.False(kv.IsPositionValid(0, 1));
        for (int pos = 2; pos < 6; pos++)
        {
            Assert.True(kv.IsPositionValid(0, pos));
        }

        float[] tmp = new float[4];
        fixed (float* pTmp = tmp)
        {
            nint pAddr = (nint)pTmp;
            Assert.Throws<InvalidOperationException>(() => kv.ReadKey(0, 0, 0, (float*)pAddr));
            Assert.Throws<InvalidOperationException>(() => kv.ReadValue(0, 0, 1, (float*)pAddr));
            Assert.Throws<InvalidOperationException>(() => _ = kv.DotKey(0, 0, 0, (float*)pAddr));
            Assert.Throws<InvalidOperationException>(() => kv.AccumulateValue(0, 0, 1, 1.0f, (float*)pAddr));
        }
        Assert.Throws<InvalidOperationException>(() => _ = (nint)kv.GetKeyPtr(0, 0, 0));
        Assert.Throws<InvalidOperationException>(() => _ = (nint)kv.GetValuePtr(0, 0, 1));

        fixed (float* pOut = tmp)
        {
            kv.ReadKey(0, 0, 5, pOut);
        }
        Assert.Equal(51f, tmp[0]);
        Assert.Equal(54f, tmp[3]);

        kv.Reset();
        Assert.False(kv.IsPositionValid(0, 5));
    }

    // -------------------------------------------------------------------------
    // Item 10: QuantKernels VecDot Vector512 & MatMulBatch B_TILE=4 parity
    // -------------------------------------------------------------------------

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(512)]
    public void QuantKernels_VecDotQ4_0_MatchesDoublePrecisionOracle(int k)
    {
        int nb = k / 32;
        var blocks = new BlockQ4_0[nb];
        var rng = new Random(4040 + k);
        for (int i = 0; i < nb; i++)
        {
            blocks[i].Delta = (Half)(float)(rng.NextDouble() * 1.5 + 0.1);
            for (int q = 0; q < 16; q++) blocks[i].Qs[q] = (byte)rng.Next(0, 256);
        }

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

        double oracle = 0.0;
        for (int b = 0; b < nb; b++)
        {
            float d = (float)blocks[b].Delta;
            int offset = b * 32;
            for (int i = 0; i < 16; i++)
            {
                byte qv = blocks[b].Qs[i];
                int q0 = (qv & 0x0F) - 8;
                int q1 = ((qv >> 4) & 0x0F) - 8;
                oracle += (double)(q0 * d) * x[offset + i];
                oracle += (double)(q1 * d) * x[offset + 16 + i];
            }
        }

        fixed (BlockQ4_0* pB = blocks)
        fixed (float* pX = x)
        {
            float actual = QuantKernels.VecDotQ4_0(pB, pX, k);
            float relErr = MathF.Abs((float)oracle - actual) / MathF.Max(1.0f, MathF.Abs((float)oracle));
            Assert.True(relErr < 1e-3f, $"Q4_0 VecDot mismatch at k={k}: oracle={oracle}, actual={actual}");
        }
    }

    [Theory]
    [InlineData(GgufType.F32, 64, 5, 7)]
    [InlineData(GgufType.F16, 64, 5, 7)]
    [InlineData(GgufType.Q8_0, 64, 5, 7)]
    [InlineData(GgufType.Q4_0, 64, 5, 7)]
    [InlineData(GgufType.Q4_K, 256, 5, 8)]
    [InlineData(GgufType.Q6_K, 256, 5, 3)]
    public void QuantKernels_MatMulBatch_TiledExecutionMatchesSingleRowReference(
        GgufType type, int nCols, int nRows, int batchSize)
    {
        var rng = new Random(9001 + (int)type);
        long totalRowBytes = type switch
        {
            GgufType.F32 => (long)nCols * nRows * 4L,
            GgufType.F16 => (long)nCols * nRows * 2L,
            GgufType.Q8_0 => ((long)nCols * nRows / 32L) * 34L,
            GgufType.Q4_0 => ((long)nCols * nRows / 32L) * 18L,
            GgufType.Q4_K => ((long)nCols * nRows / 256L) * 144L,
            GgufType.Q6_K => ((long)nCols * nRows / 256L) * 210L,
            _ => throw new NotSupportedException()
        };
        byte[] w = new byte[totalRowBytes];
        rng.NextBytes(w);

        // Keep scale Half fields finite/reasonable on block types
        if (type == GgufType.F16)
        {
            for (int i = 0; i < nCols * nRows; i++)
                BitConverter.TryWriteBytes(w.AsSpan(i * 2, 2), (Half)(float)(rng.NextDouble() * 2.0 - 1.0));
        }
        else if (type == GgufType.F32)
        {
            for (int i = 0; i < nCols * nRows; i++)
                BitConverter.TryWriteBytes(w.AsSpan(i * 4, 4), (float)(rng.NextDouble() * 2.0 - 1.0));
        }
        else if (type == GgufType.Q8_0 || type == GgufType.Q4_0)
        {
            int blkBytes = type == GgufType.Q8_0 ? 34 : 18;
            int numBlocks = (nCols * nRows) / 32;
            for (int b = 0; b < numBlocks; b++)
                BitConverter.TryWriteBytes(w.AsSpan(b * blkBytes, 2), (Half)0.125f);
        }
        else if (type == GgufType.Q4_K)
        {
            int blkBytes = 144;
            int numBlocks = (nCols * nRows) / 256;
            for (int b = 0; b < numBlocks; b++)
            {
                BitConverter.TryWriteBytes(w.AsSpan(b * blkBytes, 2), (Half)0.125f);
                BitConverter.TryWriteBytes(w.AsSpan(b * blkBytes + 2, 2), (Half)0.0625f);
            }
        }
        else if (type == GgufType.Q6_K)
        {
            int blkBytes = 210;
            int numBlocks = (nCols * nRows) / 256;
            for (int b = 0; b < numBlocks; b++)
                BitConverter.TryWriteBytes(w.AsSpan(b * blkBytes + 208, 2), (Half)0.125f);
        }

        float[] x = new float[batchSize * nCols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] yBatch = new float[batchSize * nRows];
        float[] ySingle = new float[nRows];

        fixed (byte* pW = w)
        fixed (float* pX = x, pY = yBatch, pSingle = ySingle)
        {
            QuantKernels.MatMulBatch(type, pW, pX, pY, nCols, nRows, batchSize);
            for (int b = 0; b < batchSize; b++)
            {
                QuantKernels.MatMulBatch(type, pW, pX + b * nCols, pSingle, nCols, nRows, 1);
                for (int r = 0; r < nRows; r++)
                {
                    float expected = ySingle[r];
                    float actual = yBatch[b * nRows + r];
                    Assert.True(MathF.Abs(expected - actual) <= 1e-4f + 1e-4f * MathF.Abs(expected),
                        $"{type} batch={batchSize} b={b} r={r}: expected={expected}, actual={actual}");
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Item 12: Program.Serve continuous-batch gate & FixedTimeTokenEquals
    // -------------------------------------------------------------------------

    [Fact]
    public void ProgramServe_ContinuousBatchGate_AllowsApiKey_AndBlocksSplitMode()
    {
        using var handle = VulkanE2ETestFixtures.SyntheticGgufHandle.CreateQwen2();

        Assert.True(Program.CanUseContinuousBatchEngine(handle.Path, "secret-token", null, "cpu", InferenceEngineType.Cpu));
        Assert.True(Program.CanUseContinuousBatchEngine(handle.Path, null, null, "auto", InferenceEngineType.Auto));
        Assert.False(Program.CanUseContinuousBatchEngine(handle.Path, "secret-token", "cuda:0", "cuda", InferenceEngineType.BareMetal));
        Assert.False(Program.CanUseContinuousBatchEngine(handle.Path, "secret-token", null, "vulkan", InferenceEngineType.Vulkan));
        Assert.False(Program.CanUseContinuousBatchEngine(Path.Combine(Path.GetTempPath(), $"missing_{Guid.NewGuid():N}.gguf"), "secret-token", null, "cpu", InferenceEngineType.Cpu));
    }

    [Fact]
    public void ProgramServe_FixedTimeTokenEquals_HandlesMatchMismatchAndDifferentLengths()
    {
        Assert.True(Program.FixedTimeTokenEquals("glacier-secret-123", "glacier-secret-123"));
        Assert.False(Program.FixedTimeTokenEquals("glacier-secret-124", "glacier-secret-123"));
        Assert.False(Program.FixedTimeTokenEquals("short", "glacier-secret-123"));
        Assert.False(Program.FixedTimeTokenEquals("glacier-secret-123-extra", "glacier-secret-123"));
        Assert.False(Program.FixedTimeTokenEquals(string.Empty, "non-empty"));
        Assert.True(Program.FixedTimeTokenEquals(string.Empty, string.Empty));
        Assert.True(Program.FixedTimeTokenEquals("tok-αβγ", "tok-αβγ"));
    }
}
