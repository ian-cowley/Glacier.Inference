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
    private readonly Dictionary<(string Left, string Right), int> _bpePairRanks = new(160000);
    private readonly Dictionary<string, int> _specialTokens = new(StringComparer.Ordinal);
    private readonly HashSet<int> _stopTokens = new();
    private readonly string _architecture = "";
    private readonly string _chatTemplate = "";
    private readonly bool _isSentencePiece;
    private readonly bool _addSpacePrefix;

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

    /// <summary>
    /// Resolves a token string or special token to its integer token ID.
    /// </summary>
    public bool TryGetTokenId(string token, out int tokenId)
    {
        if (_specialTokens.TryGetValue(token, out tokenId)) return true;
        return _tokenToId.TryGetValue(token, out tokenId);
    }

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
        BosTokenId = gguf.BosTokenId;
        PadTokenId = (int)gguf.GetMetadataUInt32("tokenizer.ggml.padding_token_id", BosTokenId >= 0 ? (uint)BosTokenId : 0u);
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
                AddMergeRule(merge, i);
            }
        }

        RegisterKnownSpecialTokens();

        // Dynamically register model's configured BOS/EOS/EOT tokens (supports scalar and array GGUF metadata)
        if (BosTokenId >= 0 && BosTokenId < _idToToken.Length)
        {
            RegisterSpecialToken(_idToToken[BosTokenId]);
        }

        var eosIds = gguf.GetEosTokenIds();
        for (int i = 0; i < eosIds.Count; i++)
        {
            int id = eosIds[i];
            if (id >= 0)
            {
                if (id < _idToToken.Length)
                    RegisterSpecialToken(_idToToken[id]);
                _stopTokens.Add(id);
            }
        }

        int resolvedEos = gguf.EosTokenId;
        if (resolvedEos < 0 && eosIds.Count > 0)
            resolvedEos = eosIds[0];

        if (resolvedEos < 0)
        {
            ReadOnlySpan<string> fallbackEosNames =
            [
                "<|im_end|>",
                "<|endoftext|>",
                "</s>",
                "<eos>",
                "<|eot_id|>",
                "<|end_of_text|>",
                "<｜end▁of▁sentence｜>",
                "<｜end of sentence｜>",
                "<|content_model_end_sampling|>",
                "<turn|>"
            ];
            for (int i = 0; i < fallbackEosNames.Length; i++)
            {
                if (_tokenToId.TryGetValue(fallbackEosNames[i], out int candidateId))
                {
                    resolvedEos = candidateId;
                    break;
                }
            }
        }
        EosTokenId = resolvedEos;

        if (EosTokenId >= 0 && EosTokenId < _idToToken.Length)
        {
            RegisterSpecialToken(_idToToken[EosTokenId]);
        }

        // Initialize stop tokens (NEVER register <|end_message|>: it delimits internal blocks within one turn)
        if (EosTokenId >= 0) _stopTokens.Add(EosTokenId);
        RegisterKnownStopTokens();

        string tokModel = gguf.GetMetadataString("tokenizer.ggml.model", "");
        _addSpacePrefix = gguf.GetMetadataBool("tokenizer.ggml.add_space_prefix", fallback: true);
        bool hasSpSpace = _tokenToId.ContainsKey("\u2581the") || _tokenToId.ContainsKey("\u2581a") || _tokenToId.ContainsKey("\u2581");
        bool hasGpt2Space = _tokenToId.ContainsKey("\u0120the") || _tokenToId.ContainsKey("\u0120a");
        _isSentencePiece = string.Equals(tokModel, "llama", StringComparison.OrdinalIgnoreCase) ||
                           (hasSpSpace && !hasGpt2Space && !string.Equals(tokModel, "gpt2", StringComparison.OrdinalIgnoreCase));
    }

    public BpeTokenizer(IEnumerable<string> vocab, int eosTokenId = 151643, int bosTokenId = 151644)
        : this(vocab, merges: null, isSentencePiece: false, eosTokenId: eosTokenId, bosTokenId: bosTokenId, addSpacePrefix: true)
    {
    }

    public BpeTokenizer(
        IEnumerable<string> vocab,
        IEnumerable<string>? merges,
        bool isSentencePiece,
        int eosTokenId = 2,
        int bosTokenId = 1,
        bool addSpacePrefix = true)
    {
        EosTokenId = eosTokenId;
        BosTokenId = bosTokenId;
        PadTokenId = bosTokenId;
        _isSentencePiece = isSentencePiece;
        _addSpacePrefix = addSpacePrefix;

        var tokenList = new List<string>(vocab);
        _idToToken = tokenList.ToArray();
        for (int i = 0; i < tokenList.Count; i++)
        {
            _tokenToId[tokenList[i]] = i;
        }

        if (merges != null)
        {
            int rank = 0;
            foreach (string merge in merges)
            {
                AddMergeRule(merge, rank++);
            }
        }

        RegisterKnownSpecialTokens();
        if (BosTokenId >= 0 && BosTokenId < _idToToken.Length) RegisterSpecialToken(_idToToken[BosTokenId]);
        if (EosTokenId >= 0 && EosTokenId < _idToToken.Length) RegisterSpecialToken(_idToToken[EosTokenId]);

        if (EosTokenId >= 0) _stopTokens.Add(EosTokenId);
        RegisterKnownStopTokens();
    }

    private void AddMergeRule(string merge, int rank)
    {
        _bpeRanks[merge] = rank;
        int sp = merge.IndexOf(' ', 1);
        if (sp <= 0) sp = merge.IndexOf(' ');
        if (sp > 0 && sp + 1 < merge.Length)
        {
            string left = merge.Substring(0, sp);
            string right = merge.Substring(sp + 1);
            _bpePairRanks[(left, right)] = rank;
        }
    }

    private void RegisterKnownSpecialTokens()
    {
        // Qwen, LLaMA 3, DeepSeek, Phi, Mistral
        RegisterSpecialToken("<|im_start|>");
        RegisterSpecialToken("<|im_end|>");
        RegisterSpecialToken("<|im_sep|>");
        RegisterSpecialToken("<|endoftext|>");
        RegisterSpecialToken("<|begin_of_text|>");
        RegisterSpecialToken("<|end_of_text|>");
        RegisterSpecialToken("<|start_header_id|>");
        RegisterSpecialToken("<|end_header_id|>");
        RegisterSpecialToken("<|eot_id|>");
        RegisterSpecialToken("<s>");
        RegisterSpecialToken("</s>");
        RegisterSpecialToken("<unk>");
        RegisterSpecialToken("[INST]");
        RegisterSpecialToken("[/INST]");
        RegisterSpecialToken("[TOOL_CALLS]");
        RegisterSpecialToken("[TOOL_RESULTS]");
        RegisterSpecialToken("[/TOOL_RESULTS]");
        RegisterSpecialToken("<think>");
        RegisterSpecialToken("</think>");
        RegisterSpecialToken("<｜begin of sentence｜>");
        RegisterSpecialToken("<｜begin▁of▁sentence｜>");
        RegisterSpecialToken("<｜end of sentence｜>");
        RegisterSpecialToken("<｜end▁of▁sentence｜>");
        RegisterSpecialToken("<｜System｜>");
        RegisterSpecialToken("<｜User｜>");
        RegisterSpecialToken("<｜Assistant｜>");
        RegisterSpecialToken("<｜latest_reminder｜>");
        RegisterSpecialToken("｜DSML｜");
        RegisterSpecialToken("<｜ref｜>");
        RegisterSpecialToken("<｜/ref｜>");
        RegisterSpecialToken("<｜det｜>");
        RegisterSpecialToken("<｜/det｜>");
        RegisterSpecialToken("<｜box｜>");
        RegisterSpecialToken("<｜/box｜>");
        RegisterSpecialToken("<｜point｜>");
        RegisterSpecialToken("<｜/point｜>");
        RegisterSpecialToken("<｜polygon｜>");
        RegisterSpecialToken("<｜/polygon｜>");
        RegisterSpecialToken("<｜grounding｜>");
        RegisterSpecialToken("<｜/grounding｜>");

        // Gemma 2 / 4 special tokens
        RegisterSpecialToken("<bos>");
        RegisterSpecialToken("<eos>");
        RegisterSpecialToken("<start_of_turn>");
        RegisterSpecialToken("<end_of_turn>");
        RegisterSpecialToken("<|turn>");
        RegisterSpecialToken("<turn|>");
        RegisterSpecialToken("<|channel>");
        RegisterSpecialToken("<channel|>");
        RegisterSpecialToken("<|tool>");
        RegisterSpecialToken("<tool|>");
        RegisterSpecialToken("<|tool_call>");
        RegisterSpecialToken("<tool_call|>");
        RegisterSpecialToken("<|tool_response>");
        RegisterSpecialToken("<tool_response|>");
        RegisterSpecialToken("<|think|>");

        // Thinking Machines Inkling-2 special control tokens (IDs 200000..200057)
        RegisterSpecialToken("<|message_user|>");
        RegisterSpecialToken("<|message_model|>");
        RegisterSpecialToken("<|message_system|>");
        RegisterSpecialToken("<|message_tool|>");
        RegisterSpecialToken("<|content_text|>");
        RegisterSpecialToken("<|content_thinking|>");
        RegisterSpecialToken("<|content_image|>");
        RegisterSpecialToken("<|content_xml|>");
        RegisterSpecialToken("<|content_invoke_tool_json|>");
        RegisterSpecialToken("<|content_invoke_tool_text|>");
        RegisterSpecialToken("<|content_tool_error|>");
        RegisterSpecialToken("<|end_message|>");
        RegisterSpecialToken("<|content_model_end_sampling|>");
        RegisterSpecialToken("<|unused_200054|>");
    }

    private void RegisterKnownStopTokens()
    {
        RegisterStopToken("<|content_model_end_sampling|>");
        RegisterStopToken("<|im_end|>");
        RegisterStopToken("<|endoftext|>");
        RegisterStopToken("<|end_of_text|>");
        RegisterStopToken("<|eot_id|>");
        RegisterStopToken("</s>");
        RegisterStopToken("<｜end of sentence｜>");
        RegisterStopToken("<｜end▁of▁sentence｜>");
        RegisterStopToken("<｜User｜>");
        RegisterStopToken("<｜System｜>");
        RegisterStopToken("<turn|>");
        RegisterStopToken("<eos>");
        RegisterStopToken("<|tool_call>");
        RegisterStopToken("<|tool_response>");
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
    /// Sanitizes user prompt content to prevent injection of control tokens and role impersonation.
    /// </summary>
    public static string SanitizePrompt(string input)
    {
        if (string.IsNullOrEmpty(input)) return input;

        return input
            .Replace("<｜begin of sentence｜>", "< ｜begin of sentence｜ >")
            .Replace("<｜begin▁of▁sentence｜>", "< ｜begin▁of▁sentence｜ >")
            .Replace("<｜end of sentence｜>", "< ｜end of sentence｜ >")
            .Replace("<｜end▁of▁sentence｜>", "< ｜end▁of▁sentence｜ >")
            .Replace("<｜System｜>", "< ｜System｜ >")
            .Replace("<｜User｜>", "< ｜User｜ >")
            .Replace("<｜Assistant｜>", "< ｜Assistant｜ >")
            .Replace("<｜latest_reminder｜>", "< ｜latest_reminder｜ >")
            .Replace("｜DSML｜", "｜ DSML ｜")
            .Replace("<｜ref｜>", "< ｜ref｜ >")
            .Replace("<｜/ref｜>", "< ｜/ref｜ >")
            .Replace("<｜det｜>", "< ｜det｜ >")
            .Replace("<｜/det｜>", "< ｜/det｜ >")
            .Replace("<｜box｜>", "< ｜box｜ >")
            .Replace("<｜/box｜>", "< ｜/box｜ >")
            .Replace("<｜point｜>", "< ｜point｜ >")
            .Replace("<｜/point｜>", "< ｜/point｜ >")
            .Replace("<｜polygon｜>", "< ｜polygon｜ >")
            .Replace("<｜/polygon｜>", "< ｜/polygon｜ >")
            .Replace("<｜grounding｜>", "< ｜grounding｜ >")
            .Replace("<｜/grounding｜>", "< ｜/grounding｜ >")
            .Replace("<think>", "< think >")
            .Replace("</think>", "< /think >")
            .Replace("<|think|>", "< |think| >")
            .Replace("<|thought|>", "< |thought| >")
            .Replace("<|/thought|>", "< |/thought| >")
            .Replace("<|im_start|>", "< |im_start| >")
            .Replace("<|im_end|>", "< |im_end| >")
            .Replace("<|im_sep|>", "< |im_sep| >")
            .Replace("<|start_header_id|>", "< |start_header_id| >")
            .Replace("<|end_header_id|>", "< |end_header_id| >")
            .Replace("<|eot_id|>", "< |eot_id| >")
            .Replace("<|begin_of_text|>", "< |begin_of_text| >")
            .Replace("<|end_of_text|>", "< |end_of_text| >")
            .Replace("<|endoftext|>", "< |endoftext| >")
            .Replace("<start_of_turn>", "< start_of_turn >")
            .Replace("<end_of_turn>", "< end_of_turn >")
            .Replace("<|turn>", "< |turn >")
            .Replace("<turn|>", "< turn| >")
            .Replace("<|channel>", "< |channel >")
            .Replace("<channel|>", "< channel| >")
            .Replace("<|tool>", "< |tool >")
            .Replace("<tool|>", "< tool| >")
            .Replace("<|tool_call>", "< |tool_call >")
            .Replace("<tool_call|>", "< tool_call| >")
            .Replace("<|tool_response>", "< |tool_response >")
            .Replace("<tool_response|>", "< tool_response| >")
            .Replace("<eos>", "< eos >")
            .Replace("<bos>", "< bos >")
            .Replace("<|message_user|>", "< |message_user| >")
            .Replace("<|message_model|>", "< |message_model| >")
            .Replace("<|message_system|>", "< |message_system| >")
            .Replace("<|message_tool|>", "< |message_tool| >")
            .Replace("<|content_text|>", "< |content_text| >")
            .Replace("<|content_thinking|>", "< |content_thinking| >")
            .Replace("<|content_image|>", "< |content_image| >")
            .Replace("<|content_xml|>", "< |content_xml| >")
            .Replace("<|content_invoke_tool_json|>", "< |content_invoke_tool_json| >")
            .Replace("<|content_invoke_tool_text|>", "< |content_invoke_tool_text| >")
            .Replace("<|content_tool_error|>", "< |content_tool_error| >")
            .Replace("<|end_message|>", "< |end_message| >")
            .Replace("<|content_model_end_sampling|>", "< |content_model_end_sampling| >")
            .Replace("<|unused_200054|>", "< |unused_200054| >")
            .Replace("[INST]", "[ INST ]")
            .Replace("[/INST]", "[ /INST ]")
            .Replace("<s>", "< s >")
            .Replace("</s>", "< /s >")
            .Replace("<unk>", "< unk >")
            .Replace("[TOOL_CALLS]", "[ TOOL_CALLS ]")
            .Replace("[TOOL_RESULTS]", "[ TOOL_RESULTS ]")
            .Replace("[/TOOL_RESULTS]", "[ /TOOL_RESULTS ]");
    }

    /// <summary>
    /// Sanitizes user prompt content to prevent injection of control tokens and role impersonation.
    /// </summary>
    public string Sanitize(string input) => SanitizePrompt(input);

    /// <summary>
    /// Formats a multi-turn conversation into the model's native chat template
    /// supporting DeepSeek (deepseek-recipe), ChatML, LLaMA-3, and Gemma.
    /// </summary>
    public string FormatConversation(IEnumerable<(string Role, string Content)> messages, bool thinkingMode = false)
    {
        // 0. Thinking Machines Inkling-2 format (<|message_user|><|content_text|>...<|end_message|>)
        if (_specialTokens.ContainsKey("<|message_user|>") && _specialTokens.ContainsKey("<|end_message|>"))
        {
            var sb = new StringBuilder();
            bool effortEmitted = false;
            string effortStr = thinkingMode ? "0.7" : "0.2";
            foreach (var (role, content) in messages)
            {
                bool isSystem = role.Equals("system", StringComparison.OrdinalIgnoreCase);
                if (!effortEmitted && !isSystem)
                {
                    sb.Append("<|message_system|><|content_text|>Thinking effort level: ")
                      .Append(effortStr)
                      .Append("<|end_message|>");
                    effortEmitted = true;
                }

                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                string roleTok = role.ToLowerInvariant() switch
                {
                    "system" => "<|message_system|>",
                    "user" => "<|message_user|>",
                    "assistant" => "<|message_model|>",
                    "tool" => "<|message_tool|>",
                    _ => "<|message_user|>"
                };
                sb.Append(roleTok).Append("<|content_text|>").Append(safeContent).Append("<|end_message|>");
                if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                {
                    sb.Append("<|content_model_end_sampling|>");
                }
            }
            if (!effortEmitted)
            {
                sb.Append("<|message_system|><|content_text|>Thinking effort level: ")
                  .Append(effortStr)
                  .Append("<|end_message|>");
            }
            if (!thinkingMode)
            {
                sb.Append("<|message_model|><|content_text|>");
            }
            return sb.ToString();
        }

        // 1. DeepSeek V3 / R1 / V4 format (from deepseek-recipe)
        if ((_specialTokens.ContainsKey("<｜User｜>") || _specialTokens.ContainsKey("<｜begin of sentence｜>") || _specialTokens.ContainsKey("<｜begin▁of▁sentence｜>"))
            && !_chatTemplate.Contains("User: "))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length)
            {
                sb.Append(_idToToken[BosTokenId]);
            }
            else if (_specialTokens.TryGetValue("<｜begin of sentence｜>", out int bosId))
            {
                sb.Append(_idToToken[bosId]);
            }
            else if (_specialTokens.TryGetValue("<｜begin▁of▁sentence｜>", out int bos2Id))
            {
                sb.Append(_idToToken[bos2Id]);
            }

            string eosToken = _specialTokens.ContainsKey("<｜end▁of▁sentence｜>") && !_specialTokens.ContainsKey("<｜end of sentence｜>")
                ? "<｜end▁of▁sentence｜>"
                : "<｜end of sentence｜>";

            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase)
                    ? content
                    : SanitizePrompt(content);

                if (role.Equals("system", StringComparison.OrdinalIgnoreCase))
                {
                    sb.Append("<｜System｜>").Append(safeContent);
                }
                else if (role.Equals("user", StringComparison.OrdinalIgnoreCase))
                {
                    sb.Append("<｜User｜>").Append(safeContent);
                }
                else if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                {
                    sb.Append("<｜Assistant｜>").Append(safeContent).Append(eosToken);
                }
            }
            sb.Append("<｜Assistant｜>");
            if (thinkingMode && _specialTokens.ContainsKey("<think>"))
            {
                sb.Append("<think>\n");
            }
            return sb.ToString();
        }

        // 2. Gemma format (<start_of_turn> or <|turn>)
        if (_specialTokens.ContainsKey("<start_of_turn>"))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length) sb.Append(_idToToken[BosTokenId]);
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                string r = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? "model" : role.ToLowerInvariant();
                sb.Append($"<start_of_turn>{r}\n{safeContent}<end_of_turn>\n");
            }
            sb.Append("<start_of_turn>model\n");
            return sb.ToString();
        }

        if (_specialTokens.ContainsKey("<|turn>"))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length) sb.Append(_idToToken[BosTokenId]);
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                string r = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? "model" : role;
                sb.Append($"<|turn>{r}\n{safeContent}<turn|>\n");
            }
            sb.Append("<|turn>model\n");
            return sb.ToString();
        }

        // 3. LLaMA 3 format (<|start_header_id|>)
        if (_specialTokens.ContainsKey("<|start_header_id|>"))
        {
            var sb = new StringBuilder();
            if (_specialTokens.ContainsKey("<|begin_of_text|>")) sb.Append("<|begin_of_text|>");
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                sb.Append($"<|start_header_id|>{role.ToLowerInvariant()}<|end_header_id|>\n\n{safeContent}<|eot_id|>");
            }
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
            return sb.ToString();
        }

        // 4. Microsoft Phi-3 / Phi-4 format with <|im_sep|>
        if (_specialTokens.ContainsKey("<|im_sep|>"))
        {
            var sb = new StringBuilder();
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                sb.Append($"<|im_start|>{role}<|im_sep|>{safeContent}<|im_end|>");
            }
            sb.Append("<|im_start|>assistant<|im_sep|>");
            return sb.ToString();
        }

        // 5. Mistral format ([INST] ... [/INST])
        if (_specialTokens.ContainsKey("[INST]"))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length) sb.Append(_idToToken[BosTokenId]);
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                if (role.Equals("user", StringComparison.OrdinalIgnoreCase))
                    sb.Append("[INST] ").Append(safeContent).Append(" [/INST]");
                else if (role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                    sb.Append(' ').Append(safeContent).Append(' ');
                else
                    sb.Append(safeContent).Append("\n\n");
            }
            return sb.ToString();
        }

        // 6. Legacy DeepSeek format (User: ... \n\nAssistant:)
        if (_architecture.Equals("deepseek2", StringComparison.OrdinalIgnoreCase) || _chatTemplate.Contains("User: "))
        {
            var sb = new StringBuilder();
            if (BosTokenId >= 0 && BosTokenId < _idToToken.Length) sb.Append(_idToToken[BosTokenId]);
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                if (role.Equals("system", StringComparison.OrdinalIgnoreCase))
                    sb.Append(safeContent).Append("\n\n");
                else if (role.Equals("user", StringComparison.OrdinalIgnoreCase))
                    sb.Append("User: ").Append(safeContent).Append("\n\n");
                else
                    sb.Append("Assistant: ").Append(safeContent).Append("\n\n");
            }
            sb.Append("Assistant:");
            return sb.ToString();
        }

        // 7. Standard ChatML format (Qwen2, Qwen3, etc.)
        {
            var sb = new StringBuilder();
            foreach (var (role, content) in messages)
            {
                string safeContent = role.Equals("assistant", StringComparison.OrdinalIgnoreCase) ? content : SanitizePrompt(content);
                sb.Append($"<|im_start|>{role.ToLowerInvariant()}\n{safeContent}<|im_end|>\n");
            }
            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Formats a user prompt and optional system prompt into the model's native chat template (DeepSeek, ChatML, LLaMA-3, Gemma, Mistral, Phi).
    /// </summary>
    public string FormatChatML(string prompt, string systemPrompt = "You are a helpful assistant.")
    {
        var messages = new List<(string Role, string Content)>();
        bool isDefaultSystem = string.IsNullOrEmpty(systemPrompt) || systemPrompt == "You are a helpful assistant.";

        bool suppressDefaultSystem = _specialTokens.ContainsKey("<start_of_turn>") ||
                                     _specialTokens.ContainsKey("<|turn>") ||
                                     _specialTokens.ContainsKey("[INST]") ||
                                     _architecture.Equals("deepseek2", StringComparison.OrdinalIgnoreCase) ||
                                     _chatTemplate.Contains("User: ") ||
                                     ((_specialTokens.ContainsKey("<｜User｜>") || _specialTokens.ContainsKey("<｜begin of sentence｜>") || _specialTokens.ContainsKey("<｜begin▁of▁sentence｜>")) && !_chatTemplate.Contains("User: "));

        if (!string.IsNullOrEmpty(systemPrompt) && (!isDefaultSystem || !suppressDefaultSystem))
        {
            messages.Add(("system", systemPrompt));
        }

        messages.Add(("user", prompt));
        return FormatConversation(messages, thinkingMode: false);
    }

    /// <summary>
    /// Formats a sequence of chat messages into the model's native chat template (DeepSeek, ChatML, LLaMA-3, Gemma, Mistral, Phi).
    /// </summary>
    public string FormatChat(IEnumerable<(string Role, string Content)> messages)
    {
        return FormatConversation(messages, thinkingMode: false);
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
            bool isFirstSegment = index == 0;
            index = nextSpecialIndex;

            if (_isSentencePiece)
            {
                EncodeSentencePieceSegment(segment, isFirstSegment && _addSpacePrefix, word, result);
                continue;
            }

            // Pre-tokenize segment using source-generated regex with zero-allocation span enumeration
            foreach (var match in GetTokenSplitterRegex().EnumerateMatches(segment))
            {
                ReadOnlySpan<char> matchSpan = segment.Slice(match.Index, match.Length);
                if (_bpeRanks.Count == 0 && _tokenToId.TryGetValue(matchSpan.ToString(), out int directId))
                {
                    result.Add(directId);
                    continue;
                }
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

                    // Apply BPE merges via O(n log n) min-heap priority queue
                    ApplyBpeMerges(word);

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
    /// Gets the raw UTF-8 bytes for a single token ID.
    /// </summary>
    public byte[] GetTokenBytes(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _idToToken.Length)
            return Array.Empty<byte>();

        string piece = _idToToken[tokenId];
        if (_specialTokens.ContainsKey(piece))
            return Encoding.UTF8.GetBytes(piece);

        if (_isSentencePiece)
        {
            if (TryParseByteFallbackToken(piece, out byte spByte))
            {
                return [spByte];
            }
            return Encoding.UTF8.GetBytes(piece.Replace('\u2581', ' '));
        }

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

        return byteList.ToArray();
    }

    /// <summary>
    /// Decodes a single token ID into its string representation.
    /// </summary>
    public string DecodeToken(int tokenId)
    {
        byte[] bytes = GetTokenBytes(tokenId);
        return bytes.Length == 0 ? "" : Encoding.UTF8.GetString(bytes);
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

            if (_isSentencePiece)
            {
                if (TryParseByteFallbackToken(piece, out byte spByte))
                {
                    byteBuffer.Add(spByte);
                }
                else
                {
                    byteBuffer.AddRange(Encoding.UTF8.GetBytes(piece.Replace('\u2581', ' ')));
                }
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

    private void EncodeSentencePieceSegment(
        ReadOnlySpan<char> segment,
        bool prependSpacePrefix,
        List<string> workSymbols,
        List<int> output)
    {
        if (segment.IsEmpty) return;

        string normalized = segment.ToString().Replace(' ', '\u2581');
        if (prependSpacePrefix && !normalized.StartsWith('\u2581'))
        {
            normalized = "\u2581" + normalized;
        }

        if (_bpeRanks.Count == 0)
        {
            int pos = 0;
            while (pos < normalized.Length)
            {
                int maxLen = Math.Min(32, normalized.Length - pos);
                bool matched = false;
                for (int len = maxLen; len >= 1; len--)
                {
                    string candidate = normalized.Substring(pos, len);
                    if (_tokenToId.TryGetValue(candidate, out int id))
                    {
                        output.Add(id);
                        pos += len;
                        matched = true;
                        break;
                    }
                }
                if (!matched)
                {
                    int charStep = (pos + 1 < normalized.Length &&
                                    char.IsHighSurrogate(normalized[pos]) &&
                                    char.IsLowSurrogate(normalized[pos + 1]))
                        ? 2
                        : 1;
                    AddUtf8ByteFallback(normalized.Substring(pos, charStep), output);
                    pos += charStep;
                }
            }
            return;
        }

        workSymbols.Clear();
        for (int i = 0; i < normalized.Length;)
        {
            int charStep = (i + 1 < normalized.Length &&
                            char.IsHighSurrogate(normalized[i]) &&
                            char.IsLowSurrogate(normalized[i + 1]))
                ? 2
                : 1;
            workSymbols.Add(normalized.Substring(i, charStep));
            i += charStep;
        }

        ApplyBpeMerges(workSymbols);

        foreach (string piece in workSymbols)
        {
            if (_tokenToId.TryGetValue(piece, out int id))
            {
                output.Add(id);
            }
            else
            {
                AddUtf8ByteFallback(piece, output);
            }
        }
    }

    private void ApplyBpeMerges(List<string> symbols)
    {
        int n = symbols.Count;
        if (n <= 1 || _bpePairRanks.Count == 0)
            return;

        var sym = new string[n];
        var prev = new int[n];
        var next = new int[n];
        var alive = new bool[n];
        var gen = new int[n];

        for (int i = 0; i < n; i++)
        {
            sym[i] = symbols[i];
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
            alive[i] = true;
        }

        var pq = new PriorityQueue<(int Left, int Right, int LeftGen, int RightGen), (int Rank, int Pos)>(n);

        void TryPush(int l)
        {
            if (l < 0 || !alive[l]) return;
            int r = next[l];
            if (r < 0 || !alive[r]) return;
            if (_bpePairRanks.TryGetValue((sym[l], sym[r]), out int rank))
            {
                pq.Enqueue((l, r, gen[l], gen[r]), (rank, l));
            }
        }

        for (int i = 0; i < n - 1; i++)
        {
            TryPush(i);
        }

        while (pq.TryDequeue(out var item, out _))
        {
            int l = item.Left;
            int r = item.Right;
            if (!alive[l] || !alive[r] || next[l] != r || gen[l] != item.LeftGen || gen[r] != item.RightGen)
                continue;

            sym[l] = string.Concat(sym[l], sym[r]);
            gen[l]++;
            alive[r] = false;
            int rr = next[r];
            next[l] = rr;
            if (rr >= 0)
                prev[rr] = l;

            TryPush(prev[l]);
            TryPush(l);
        }

        symbols.Clear();
        for (int cur = 0; cur >= 0; cur = next[cur])
        {
            if (alive[cur])
                symbols.Add(sym[cur]);
        }
    }

    private void AddUtf8ByteFallback(string piece, List<int> output)
    {
        Span<byte> buf = stackalloc byte[Encoding.UTF8.GetMaxByteCount(piece.Length)];
        int count = Encoding.UTF8.GetBytes(piece.AsSpan(), buf);
        for (int i = 0; i < count; i++)
        {
            byte b = buf[i];
            if (_tokenToId.TryGetValue($"<0x{b:X2}>", out int id) ||
                _tokenToId.TryGetValue($"<0x{b:x2}>", out id))
            {
                output.Add(id);
            }
        }
    }

    private static bool TryParseByteFallbackToken(string piece, out byte value)
    {
        if (piece.Length == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>')
        {
            return byte.TryParse(piece.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
        }
        value = 0;
        return false;
    }
}
