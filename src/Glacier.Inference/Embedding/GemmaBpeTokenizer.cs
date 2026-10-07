namespace Glacier.Inference.Embedding;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Glacier.Inference.Gguf;

/// <summary>
/// Tokenizer for the Gemma 4 / EmbeddingGemma 2 vocabulary (<c>tokenizer.ggml.model = gemma4</c>).
/// </summary>
/// <remarks>
/// This is a SentencePiece-style BPE, NOT the GPT-2 byte-level BPE used by <c>BpeTokenizer</c>:
/// spaces are normalized to U+2581 ('▁'), no regex pre-split is applied (the whole text is one BPE word),
/// merges operate on Unicode code points, and code points absent from the vocabulary fall back to
/// the <c>&lt;0xNN&gt;</c> byte tokens. Control tokens (e.g. <c>&lt;bos&gt;</c>) present literally in
/// the input are emitted as single ids.
/// </remarks>
public sealed class GemmaBpeTokenizer
{
    private const string SpaceMarker = "\u2581";

    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<(int Left, int Right), (int Rank, int Merged)> _merges;
    private readonly int[] _byteTokens = new int[256];
    private readonly List<(string Text, int Id)> _controlTokens = [];

    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnkTokenId { get; }
    public bool AddBos { get; }
    public bool AddEos { get; }
    public int VocabSize => _idToToken.Length;

    public GemmaBpeTokenizer(GgufFile gguf)
    {
        if (!gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokObj) || tokObj is not List<object> tokens)
            throw new InvalidOperationException("GGUF is missing 'tokenizer.ggml.tokens'.");
        if (!gguf.Metadata.TryGetValue("tokenizer.ggml.merges", out var mergeObj) || mergeObj is not List<object> merges)
            throw new InvalidOperationException("GGUF is missing 'tokenizer.ggml.merges'.");

        BosTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.bos_token_id", 2);
        EosTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.eos_token_id", 1);
        UnkTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.unknown_token_id", 3);
        AddBos = gguf.GetMetadataBool("tokenizer.ggml.add_bos_token", true);
        AddEos = gguf.GetMetadataBool("tokenizer.ggml.add_eos_token", true);

        _idToToken = new string[tokens.Count];
        _tokenToId = new Dictionary<string, int>(tokens.Count, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Count; i++)
        {
            string t = tokens[i]?.ToString() ?? string.Empty;
            _idToToken[i] = t;
            _tokenToId.TryAdd(t, i);
        }

        Array.Fill(_byteTokens, -1);
        for (int b = 0; b < 256; b++)
        {
            if (_tokenToId.TryGetValue($"<0x{b:X2}>", out int id)) _byteTokens[b] = id;
        }

        // Control (3) / user-defined (4) tokens are matched literally in the raw text.
        if (gguf.Metadata.TryGetValue("tokenizer.ggml.token_type", out var typeObj) && typeObj is List<object> types)
        {
            for (int i = 0; i < types.Count && i < tokens.Count; i++)
            {
                int ty = Convert.ToInt32(types[i], CultureInfo.InvariantCulture);
                if ((ty == 3 || ty == 4) && _idToToken[i].Length > 1 && _idToToken[i] != "<unk>")
                    _controlTokens.Add((_idToToken[i], i));
            }
            _controlTokens.Sort((a, b) => b.Text.Length.CompareTo(a.Text.Length));
        }

        _merges = new Dictionary<(int, int), (int, int)>(merges.Count);
        for (int rank = 0; rank < merges.Count; rank++)
        {
            string m = merges[rank]?.ToString() ?? string.Empty;
            int sp = m.IndexOf(' ', 1); // tokens never contain ' ' (normalized to U+2581); index 1 tolerates a leading-space edge
            if (sp <= 0) continue;
            string l = m[..sp], r = m[(sp + 1)..];
            if (_tokenToId.TryGetValue(l, out int li) && _tokenToId.TryGetValue(r, out int ri) &&
                _tokenToId.TryGetValue(l + r, out int merged))
            {
                _merges.TryAdd((li, ri), (rank, merged));
            }
        }
    }

    public string IdToToken(int id) => _idToToken[id];

    /// <summary>Encodes text exactly like the reference tokenizer, adding BOS/EOS when configured.</summary>
    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = new List<int>(text.Length / 3 + 4);
        if (addSpecialTokens && AddBos) ids.Add(BosTokenId);

        int pos = 0;
        int segStart = 0;
        while (pos < text.Length)
        {
            int matchedId = -1, matchedLen = 0;
            if (text[pos] == '<' || text[pos] == '[')
            {
                foreach (var (ct, id) in _controlTokens)
                {
                    if (string.CompareOrdinal(text, pos, ct, 0, ct.Length) == 0)
                    {
                        matchedId = id; matchedLen = ct.Length; break;
                    }
                }
            }

            if (matchedId >= 0)
            {
                EncodeSegment(text.AsSpan(segStart, pos - segStart), ids);
                ids.Add(matchedId);
                pos += matchedLen;
                segStart = pos;
            }
            else
            {
                pos++;
            }
        }
        EncodeSegment(text.AsSpan(segStart), ids);

        if (addSpecialTokens && AddEos) ids.Add(EosTokenId);
        return ids.ToArray();
    }

    private void EncodeSegment(ReadOnlySpan<char> segment, List<int> output)
    {
        if (segment.IsEmpty) return;

        string normalized = segment.ToString().Replace(" ", SpaceMarker, StringComparison.Ordinal);

        // Initial symbols: one per Unicode code point (byte-fallback tokens for unknown ones).
        var symbols = new List<int>(normalized.Length);
        for (int i = 0; i < normalized.Length;)
        {
            int len = char.IsHighSurrogate(normalized[i]) && i + 1 < normalized.Length && char.IsLowSurrogate(normalized[i + 1]) ? 2 : 1;
            string piece = normalized.Substring(i, len);
            i += len;
            if (_tokenToId.TryGetValue(piece, out int id))
            {
                symbols.Add(id);
            }
            else
            {
                foreach (byte b in Encoding.UTF8.GetBytes(piece))
                    symbols.Add(_byteTokens[b] >= 0 ? _byteTokens[b] : UnkTokenId);
            }
        }

        BpeMerge(symbols);
        output.AddRange(symbols);
    }

    /// <summary>Rank-ordered pair merging using a linked list + priority queue (O(n log n)).</summary>
    private void BpeMerge(List<int> symbols)
    {
        int n = symbols.Count;
        if (n < 2) return;

        var sym = symbols.ToArray();
        var prev = new int[n];
        var next = new int[n];
        var alive = new bool[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; alive[i] = true; }

        var queue = new PriorityQueue<(int Left, int Right, int LeftId, int RightId), (int Rank, int Pos)>();

        void TryPush(int left)
        {
            int right = left >= 0 ? next[left] : -1;
            if (left < 0 || right < 0) return;
            if (_merges.TryGetValue((sym[left], sym[right]), out var m))
                queue.Enqueue((left, right, sym[left], sym[right]), (m.Rank, left));
        }

        for (int i = 0; i < n - 1; i++) TryPush(i);

        while (queue.TryDequeue(out var item, out _))
        {
            // Skip stale entries (a participant was merged away or changed since enqueue).
            if (!alive[item.Left] || !alive[item.Right] || next[item.Left] != item.Right ||
                sym[item.Left] != item.LeftId || sym[item.Right] != item.RightId)
                continue;

            var m = _merges[(item.LeftId, item.RightId)];
            sym[item.Left] = m.Merged;
            alive[item.Right] = false;
            int after = next[item.Right];
            next[item.Left] = after;
            if (after >= 0) prev[after] = item.Left;

            TryPush(prev[item.Left]);
            TryPush(item.Left);
        }

        symbols.Clear();
        for (int i = 0; i >= 0; i = next[i])
        {
            if (alive[i]) symbols.Add(sym[i]);
            if (next[i] < 0) break;
        }
    }
}
