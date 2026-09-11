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

    // Byte-level BPE character mappings
    private static readonly char[] ByteToChar = new char[256];
    private static readonly Dictionary<char, byte> CharToByte = new(256);

    // Qwen2 / GPT-2 regex pattern for pre-tokenization
    private static readonly Regex TokenSplitterRegex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled);

    public int VocabSize => _idToToken.Length;
    public int EosTokenId { get; }
    public int BosTokenId { get; }
    public int PadTokenId { get; }

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
        }
    }

    public BpeTokenizer(GgufFile gguf)
    {
        EosTokenId = gguf.EosTokenId;
        BosTokenId = gguf.BosTokenId;
        PadTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.padding_token_id", (uint)BosTokenId);

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

        // Register special tokens
        RegisterSpecialToken("<|im_start|>");
        RegisterSpecialToken("<|im_end|>");
        RegisterSpecialToken("<|endoftext|>");
    }

    private void RegisterSpecialToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out int id))
        {
            _specialTokens[token] = id;
        }
    }

    /// <summary>
    /// Formats a user prompt and optional system prompt into standard Qwen2 ChatML format.
    /// </summary>
    public string FormatChatML(string prompt, string systemPrompt = "You are a helpful assistant.")
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

    /// <summary>
    /// Encodes a text prompt into an array of token IDs.
    /// Handles ChatML special tokens and byte-level BPE merges.
    /// </summary>
    public int[] Encode(string text)
    {
        var result = new List<int>();
        int index = 0;

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

            // Slice substring before the next special token
            string segment = text.Substring(index, nextSpecialIndex - index);
            index = nextSpecialIndex;

            // Pre-tokenize segment using regex
            var matches = TokenSplitterRegex.Matches(segment);
            foreach (Match match in matches)
            {
                string matchText = match.Value;
                byte[] utf8Bytes = Encoding.UTF8.GetBytes(matchText);

                // Convert bytes to unicode BPE chars
                var chars = new char[utf8Bytes.Length];
                for (int i = 0; i < utf8Bytes.Length; i++)
                {
                    chars[i] = ByteToChar[utf8Bytes[i]];
                }

                // Initial word pieces
                var word = new List<string>(chars.Length);
                for (int i = 0; i < chars.Length; i++)
                {
                    word.Add(chars[i].ToString());
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
