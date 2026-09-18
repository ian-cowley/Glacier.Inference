namespace Glacier.Inference.Tokenizer;

using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using Glacier.Inference.Gguf;

/// <summary>
/// Embedded Byte-Pair Encoding (BPE) Tokenizer for Qwen2 and GPT-2 style GGUF models.
/// Parses vocabulary, merges, and special tokens directly from GGUF metadata.
/// </summary>
public sealed partial class BpeTokenizer
{
    private readonly Dictionary<string, int> _tokenToId = new(160000, StringComparer.Ordinal);
    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _bpeRanks = new(160000, StringComparer.Ordinal);
    private readonly Dictionary<string, int> _specialTokens = new(StringComparer.Ordinal);
    private readonly HashSet<int> _stopTokens = new();
    private readonly string _architecture = "";
    private readonly string _chatTemplate = "";

    // Byte-level BPE character mappings
    private static readonly char[] ByteToChar = new char[256];
    private static readonly Dictionary<char, byte> CharToByte = new(256);

    private static readonly string[] CharToStringLut = new string[256];

    // Qwen2 / GPT-2 source-generated regex pattern for Native AOT pre-tokenization
    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")]
    private static partial Regex GetTokenSplitterRegex();

    public int VocabSize => _idToToken.Length;
    public int EosTokenId { get; }
    public int BosTokenId { get; }
    public int PadTokenId { get; }
    public bool AddBosToken { get; }
    public bool AddEosToken { get; }

    public bool IsStopToken(int tokenId) => _stopTokens.Contains(tokenId);

    static BpeTokenizer()
    {
        // Construct the standard GPT-2 byte <-> unicode mapping
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

        for (int i = 0; i < 256; i++)
        {
            byte b = (byte)bs[i];
            char c = (char)cs[i];
            ByteToChar[b] = c;
            CharToByte[c] = b;
            CharToStringLut[b] = c.ToString();
        }
    }

    public BpeTokenizer(GgufFile gguf)
    {
        EosTokenId = gguf.EosTokenId;
        BosTokenId = gguf.BosTokenId;
        PadTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.padding_token_id", (uint)BosTokenId);
        AddBosToken = gguf.AddBosToken;
        AddEosToken = gguf.AddEosToken;
        _architecture = gguf.Architecture;
        _chatTemplate = gguf.Metadata.TryGetValue("tokenizer.chat_template", out var ct) ? ct?.ToString() ?? "" : "";

        // Load tokens
        if (!gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokensObj) || tokensObj is not List<object> tokenList)
        {
            throw new InvalidOperationException("GGUF file missing 'tokenizer.ggml.tokens' metadata.");
        }

        _idToToken = new string[tokenList.Count];
        for (int i = 0; i < tokenList.Count; i++)
        {
            string tok = tokenList[i]?.ToString() ?? "";
            _idToToken[i] = tok;
            _tokenToId[tok] = i;
        }

        // Load merges
        if (gguf.Metadata.TryGetValue("tokenizer.ggml.merges", out var mergesObj) && mergesObj is List<object> mergeList)
        {
            for (int i = 0; i < mergeList.Count; i++)
            {
                string merge = mergeList[i]?.ToString() ?? "";
                _bpeRanks[merge] = i;
            }
        }

        // Register special tokens across architectures (Qwen, LLaMA 3, DeepSeek, Mistral)
        RegisterSpecialToken("<|im_start|>");
        RegisterSpecialToken("<|im_end|>");
        RegisterSpecialToken("<|endoftext|>");
        RegisterSpecialToken("<|begin_of_text|>");
        RegisterSpecialToken("<|end_of_text|>");
        RegisterSpecialToken("<|start_header_id|>");
        RegisterSpecialToken("<|end_header_id|>");
        RegisterSpecialToken("<|eot_id|>");
        RegisterSpecialToken("<s>");
        RegisterSpecialToken("</s>");
        RegisterSpecialToken("<unk>");
        RegisterSpecialToken("<think>");
        RegisterSpecialToken("</think>");
        RegisterSpecialToken("<｜begin of sentence｜>");
        RegisterSpecialToken("<｜end of sentence｜>");

        // Dynamically register model's configured BOS/EOS
        if (BosTokenId >= 0 && BosTokenId < _idToToken.Length)
        {
            RegisterSpecialToken(_idToToken[BosTokenId]);
        }
        if (EosTokenId >= 0 && EosTokenId < _idToToken.Length)
        {
            RegisterSpecialToken(_idToToken[EosTokenId]);
            RegisterStopToken(_idToToken[EosTokenId]);
        }

        // Initialize stop tokens
        if (EosTokenId >= 0) _stopTokens.Add(EosTokenId);
        RegisterStopToken("<|im_end|>");
        RegisterStopToken("<|endoftext|>");
        RegisterStopToken("<|end_of_text|>");
        RegisterStopToken("<|eot_id|>");
        RegisterStopToken("</s>");
        RegisterStopToken("<｜end of sentence｜>");
    }

    public BpeTokenizer(IEnumerable<string> vocab, int eosTokenId = 151643, int bosTokenId = 151644)
    {
        EosTokenId = eosTokenId;
        BosTokenId = bosTokenId;
        PadTokenId = bosTokenId;

        var tokenList = new List<string>(vocab);
        _idToToken = tokenList.ToArray();
        for (int i = 0; i < tokenList.Count; i++)
        {
            _tokenToId[tokenList[i]] = i;
            _specialTokens[tokenList[i]] = i;
        }

        if (EosTokenId >= 0) _stopTokens.Add(EosTokenId);
        RegisterStopToken("<|im_end|>");
        RegisterStopToken("<|endoftext|>");
        RegisterStopToken("<|end_of_text|>");
        RegisterStopToken("<|eot_id|>");
        RegisterStopToken("</s>");
    }

    private void RegisterSpecialToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out int id))
        {
            _specialTokens[token] = id;
        }
    }

    private void RegisterStopToken(string token)
    {
        if (_specialTokens.TryGetValue(token, out int id))
        {
            _stopTokens.Add(id);
        }
        else if (_tokenToId.TryGetValue(token, out int id2))
        {
            _stopTokens.Add(id2);
        }
    }

    /// <summary>
    /// Formats a user prompt and optional system prompt into the model's native chat template (ChatML or LLaMA-3).
    /// </summary>
    public string FormatChatML(string prompt, string systemPrompt = "You are a helpful assistant.")
    {
        // 1. Check for LLaMA 3 header format
        if (_specialTokens.ContainsKey("<|start_header_id|>"))
        {
            var sb = new StringBuilder();
            if (_specialTokens.ContainsKey("<|begin_of_text|>"))
            {
                sb.Append("<|begin_of_text|>");
            }
            if (!string.IsNullOrEmpty(systemPrompt))
            {
                sb.Append($"<|start_header_id|>system<|end_header_id|>\n\n{systemPrompt}<|eot_id|>");
            }
            sb.Append($"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>");
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            return sb.ToString();
        }

        // 2. Microsoft Phi-3 / Phi-4 format with <|im_sep|>
        if (_specialTokens.ContainsKey("<|im_sep|>"))
        {
            var sb = new StringBuilder();
            if (!string.IsNullOrEmpty(systemPrompt))
            {
                sb.Append($"<|im_start|>system<|im_sep|>{systemPrompt}<|im_end|>");
            }
            sb.Append($"<|im_start|>user<|im_sep|>{prompt}<|im_end|>");
            sb.Append("<|im_start|>assistant<|im_sep|>");
            return sb.ToString();
        }

        // 3. DeepSeek format ({{ bos_token }}User: ... \n\nAssistant:)
        if (_architecture.Equals("deepseek2", StringComparison.OrdinalIgnoreCase) || _chatTemplate.Contains("User: "))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length)
            {
                sb.Append(_idToToken[BosTokenId]);
            }
            if (!string.IsNullOrEmpty(systemPrompt) && systemPrompt != "You are a helpful assistant.")
            {
                sb.Append(systemPrompt);
                sb.Append("\n\n");
            }
            sb.Append("User: ");
            sb.Append(prompt);
            sb.Append("\n\nAssistant:");
            return sb.ToString();
        }

        // 4. Mistral / Devstral format ([INST] ... [/INST])
        if (_specialTokens.ContainsKey("[INST]"))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length)
            {
                sb.Append(_idToToken[BosTokenId]);
            }
            sb.Append("[INST] ");
            if (!string.IsNullOrEmpty(systemPrompt) && systemPrompt != "You are a helpful assistant.")
            {
                sb.Append(systemPrompt).Append("\n\n");
            }
            sb.Append(prompt);
            sb.Append(" [/INST]");
            return sb.ToString();
        }

        // 3. Standard ChatML format (Qwen2, Qwen3, MiMo)
        {
            var sb = new StringBuilder();
            if (!string.IsNullOrEmpty(systemPrompt))
            {
                sb.Append("<|im_start|>system\n");
                sb.Append(systemPrompt);
                sb.Append("<|im_end|>\n");
            }
            sb.Append("<|im_start|>user\n");
            sb.Append(prompt);
            sb.Append("<|im_end|>\n");
            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Encodes a text prompt into an array of token IDs.
    /// Handles ChatML special tokens and byte-level BPE merges.
    /// </summary>
    public int[] Encode(string text)
    {
        var result = new List<int>();
        int index = 0;
        Span<byte> utf8StackBuffer = stackalloc byte[256];
        List<string> word = new(64);

        while (index < text.Length)
        {
            // Check for special tokens at current position
            bool matchedSpecial = false;
            foreach (var kvp in _specialTokens)
            {
                string spec = kvp.Key;
                if (text.Length - index >= spec.Length &&
                    text.AsSpan(index, spec.Length).SequenceEqual(spec.AsSpan()))
                {
                    result.Add(kvp.Value);
                    index += spec.Length;
                    matchedSpecial = true;
                    break;
                }
            }

            if (matchedSpecial) continue;

            // Find next special token index
            int nextSpecialIndex = text.Length;
            foreach (var kvp in _specialTokens)
            {
                int nextIdx = text.IndexOf(kvp.Key, index, StringComparison.Ordinal);
                if (nextIdx >= 0 && nextIdx < nextSpecialIndex)
                {
                    nextSpecialIndex = nextIdx;
                }
            }

            // Slice span before the next special token
            ReadOnlySpan<char> segment = text.AsSpan(index, nextSpecialIndex - index);
            index = nextSpecialIndex;

            // Pre-tokenize segment using source-generated regex with zero-allocation span enumeration
            foreach (var match in GetTokenSplitterRegex().EnumerateMatches(segment))
            {
                ReadOnlySpan<char> matchSpan = segment.Slice(match.Index, match.Length);
                int maxBytes = Encoding.UTF8.GetMaxByteCount(matchSpan.Length);
                byte[]? rentedBytes = null;
                Span<byte> byteSpan = maxBytes <= 256 ? utf8StackBuffer : (rentedBytes = System.Buffers.ArrayPool<byte>.Shared.Rent(maxBytes));

                try
                {
                    int bytesWritten = Encoding.UTF8.GetBytes(matchSpan, byteSpan);
                    word.Clear();
                    for (int i = 0; i < bytesWritten; i++)
                    {
                        word.Add(CharToStringLut[byteSpan[i]]);
                    }

                    // Apply BPE merges
                    while (word.Count > 1)
                    {
                        int bestRank = int.MaxValue;
                        int bestIndex = -1;

                        for (int i = 0; i < word.Count - 1; i++)
                        {
                            string pair = $"{word[i]} {word[i + 1]}";
                            if (_bpeRanks.TryGetValue(pair, out int rank) && rank < bestRank)
                            {
                                bestRank = rank;
                                bestIndex = i;
                            }
                        }

                        if (bestIndex == -1) break; // No more merges possible

                        string merged = word[bestIndex] + word[bestIndex + 1];
                        word[bestIndex] = merged;
                        word.RemoveAt(bestIndex + 1);
                    }

                    // Map merged tokens to IDs
                    foreach (string piece in word)
                    {
                        if (_tokenToId.TryGetValue(piece, out int id))
                        {
                            result.Add(id);
                        }
                    }
                }
                finally
                {
                    if (rentedBytes != null)
                    {
                        System.Buffers.ArrayPool<byte>.Shared.Return(rentedBytes);
                    }
                }
            }
        }

        if (AddBosToken && BosTokenId >= 0)
        {
            if (result.Count == 0 || result[0] != BosTokenId)
            {
                result.Insert(0, BosTokenId);
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Decodes a single token ID into its string representation.
    /// </summary>
    public string DecodeToken(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _idToToken.Length)
            return "";

        string piece = _idToToken[tokenId];
        if (_specialTokens.ContainsKey(piece))
            return piece;

        var byteList = new List<byte>(piece.Length);
        foreach (char c in piece)
        {
            if (CharToByte.TryGetValue(c, out byte b))
            {
                byteList.Add(b);
            }
            else
            {
                byte[] raw = Encoding.UTF8.GetBytes(c.ToString());
                byteList.AddRange(raw);
            }
        }

        return Encoding.UTF8.GetString(byteList.ToArray());
    }

    /// <summary>
    /// Decodes a sequence of token IDs back into text.
    /// </summary>
    public string Decode(ReadOnlySpan<int> tokens)
    {
        var sb = new StringBuilder();
        var byteBuffer = new List<byte>(tokens.Length * 4);

        for (int i = 0; i < tokens.Length; i++)
        {
            int tid = tokens[i];
            if (tid < 0 || tid >= _idToToken.Length) continue;

            string piece = _idToToken[tid];
            if (_specialTokens.ContainsKey(piece))
            {
                if (byteBuffer.Count > 0)
                {
                    sb.Append(Encoding.UTF8.GetString(byteBuffer.ToArray()));
                    byteBuffer.Clear();
                }
                sb.Append(piece);
                continue;
            }

            foreach (char c in piece)
            {
                if (CharToByte.TryGetValue(c, out byte b))
                {
                    byteBuffer.Add(b);
                }
                else
                {
                    byte[] raw = Encoding.UTF8.GetBytes(c.ToString());
                    byteBuffer.AddRange(raw);
                }
            }
        }

        if (byteBuffer.Count > 0)
        {
            sb.Append(Encoding.UTF8.GetString(byteBuffer.ToArray()));
        }

        return sb.ToString();
    }
}
