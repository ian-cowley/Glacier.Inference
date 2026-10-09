namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// Grapheme-to-Phoneme (G2P) converter that maps English words to International Phonetic Alphabet (IPA) strings
/// compatible with StyleTTS 2 / KittenTTS.
/// </summary>
public static class KittenPhonemizer
{
    private static readonly Dictionary<string, string> EnglishIpaLexicon = new(StringComparer.OrdinalIgnoreCase)
    {
        ["hello"] = "həlˈəʊ",
        ["world"] = "wˈɜːld",
        ["from"] = "fɹəm",
        ["glacier"] = "ɡlˈeɪʃə",
        ["high"] = "hˈaɪ",
        ["performance"] = "pəfˈɔːməns",
        ["audio"] = "ˈɔːdɪˌəʊ",
        ["voice"] = "vˈɔɪs",
        ["system"] = "sˈɪstəm",
        ["welcome"] = "wˈɛlkəm",
        ["to"] = "tə",
        ["the"] = "ðə",
        ["this"] = "ðˈɪs",
        ["is"] = "ˈɪz",
        ["a"] = "ə",
        ["an"] = "ən",
        ["test"] = "tˈɛst",
        ["of"] = "əv",
        ["text"] = "tˈɛkst",
        ["speech"] = "spˈiːtʃ",
        ["synthesis"] = "sˈɪnθəsɪs",
        ["engine"] = "ˈɛndʒɪn",
        ["pure"] = "pjˈʊə",
        ["c#"] = "sˈiː ʃˈɑːp",
        ["neural"] = "njˈʊəɹəl",
        ["running"] = "ɹˈʌnɪŋ",
        ["fast"] = "fˈɑːst",
        ["and"] = "ænd",
        ["clear"] = "klˈɪə",
        ["human"] = "hjˈuːmən",
        ["quality"] = "kwˈɒlɪti",
        ["sound"] = "sˈaʊnd",
        ["sounds"] = "sˈaʊndz",
        ["great"] = "ɡɹˈeɪt",
        ["natural"] = "nˈætʃɹəl",
        ["ai"] = "ˈeɪ ˈaɪ",
        ["model"] = "mˈɒdəl",
        ["deep"] = "dˈiːp",
        ["learning"] = "lˈɜːnɪŋ",
        ["today"] = "tədˈeɪ",
        ["good"] = "ɡˈʊd",
        ["morning"] = "mˈɔːnɪŋ",
        ["afternoon"] = "ˌɑːftənˈuːn",
        ["evening"] = "ˈiːvnɪŋ",
        ["night"] = "nˈaɪt",
        ["how"] = "hˈaʊ",
        ["are"] = "ɑː",
        ["you"] = "jˈuː",
        ["doing"] = "dˈuːɪŋ",
        ["i"] = "ˈaɪ",
        ["am"] = "æm",
        ["ready"] = "ɹˈɛdi",
        ["for"] = "fɔː",
        ["action"] = "ˈækʃən",
        ["generation"] = "ˌdʒɛnəɹˈeɪʃən",
        ["intelligence"] = "ɪntˈɛlɪdʒəns",
    };

    public static string Phonemize(string text) => TextToIpa(text);

    public static string TextToIpa(string text)
    {
        string normalized = KittenTextPreprocess.Normalize(text);
        if (string.IsNullOrEmpty(normalized)) return string.Empty;

        var sb = new StringBuilder();
        var words = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i < words.Length; i++)
        {
            string w = words[i];

            // Extract trailing punctuation
            char punct = '\0';
            if (w.Length > 1 && (w.EndsWith('.') || w.EndsWith(',') || w.EndsWith('!') || w.EndsWith('?') || w.EndsWith(';') || w.EndsWith(':')))
            {
                punct = w[^1];
                w = w[..^1];
            }

            if (EnglishIpaLexicon.TryGetValue(w, out string? ipa))
            {
                sb.Append(ipa);
            }
            else
            {
                // Fallback to phonetic rule-based transliteration
                sb.Append(RuleBasedG2p(w));
            }

            if (punct != '\0')
            {
                sb.Append(punct);
            }

            if (i < words.Length - 1)
            {
                sb.Append(' ');
            }
        }

        return sb.ToString();
    }

    private static string RuleBasedG2p(string word)
    {
        var sb = new StringBuilder();
        string lower = word.ToLowerInvariant();

        for (int i = 0; i < lower.Length; i++)
        {
            char c = lower[i];
            if (i == 0) sb.Append('ˈ'); // Mark primary stress on first syllable by default

            if (i + 1 < lower.Length)
            {
                string pair = lower.Substring(i, 2);
                switch (pair)
                {
                    case "th": sb.Append('θ'); i++; continue;
                    case "sh": sb.Append('ʃ'); i++; continue;
                    case "ch": sb.Append("tʃ"); i++; continue;
                    case "ph": sb.Append('f'); i++; continue;
                    case "ee": sb.Append("iː"); i++; continue;
                    case "oo": sb.Append("uː"); i++; continue;
                    case "ou": sb.Append("aʊ"); i++; continue;
                    case "ai": case "ay": sb.Append("eɪ"); i++; continue;
                    case "ea": sb.Append("iː"); i++; continue;
                    case "oa": sb.Append("əʊ"); i++; continue;
                    case "er": sb.Append('ɜ'); i++; continue;
                    case "ng": sb.Append('ŋ'); i++; continue;
                    case "ck": sb.Append('k'); i++; continue;
                }
            }

            switch (c)
            {
                case 'a': sb.Append('æ'); break;
                case 'e': sb.Append('ɛ'); break;
                case 'i': sb.Append('ɪ'); break;
                case 'o': sb.Append('ɒ'); break;
                case 'u': sb.Append('ʌ'); break;
                case 'r': sb.Append('ɹ'); break;
                case 'j': sb.Append("dʒ"); break;
                default:
                    if (c >= 'a' && c <= 'z') sb.Append(c);
                    break;
            }
        }

        return sb.ToString();
    }
}
