namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Collections.Generic;

/// <summary>
/// Complete 179-token symbol table and ID encoder for KittenTTS.
/// Index 0: '$' padding, 1-15: punctuation, 16: space, 17-42: A-Z, 43-68: a-z, 69-178: IPA characters.
/// </summary>
public static class KittenPhonemeMap
{
    public static readonly string[] Symbols =
    [
        "$",      // 0 (padding)
        ";",      // 1
        ":",      // 2
        ",",      // 3
        ".",      // 4
        "!",      // 5
        "?",      // 6
        "¡",      // 7
        "¿",      // 8
        "—",      // 9
        "…",      // 10
        "\"",     // 11
        "«",      // 12
        "»",      // 13
        "\u201c", // 14  “
        "\u201d", // 15  ”
        " ",      // 16
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", // 17-29
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", // 30-42
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", // 43-55
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", // 56-68
        // IPA characters matching StyleTTS 2 / KittenTTS specification exactly (offset 69):
        "ɑ",      // 69
        "ɐ",      // 70
        "ɒ",      // 71
        "æ",      // 72
        "ɓ",      // 73
        "ʙ",      // 74
        "β",      // 75
        "ɔ",      // 76
        "ɕ",      // 77
        "ç",      // 78
        "ɗ",      // 79
        "ɖ",      // 80
        "ð",      // 81
        "ʤ",      // 82
        "ə",      // 83
        "ɘ",      // 84
        "ɚ",      // 85
        "ɛ",      // 86
        "ɜ",      // 87
        "ɝ",      // 88
        "ɞ",      // 89
        "ɟ",      // 90
        "ʄ",      // 91
        "ɡ",      // 92
        "ɠ",      // 93
        "ɢ",      // 94
        "ʛ",      // 95
        "ɦ",      // 96
        "ɧ",      // 97
        "ħ",      // 98
        "ɥ",      // 99
        "ʜ",      // 100
        "ɨ",      // 101
        "ɪ",      // 102
        "ʝ",      // 103
        "ɭ",      // 104
        "ɬ",      // 105
        "ɫ",      // 106
        "ɮ",      // 107
        "ʟ",      // 108
        "ɱ",      // 109
        "ɯ",      // 110
        "ɰ",      // 111
        "ŋ",      // 112
        "ɳ",      // 113
        "ɲ",      // 114
        "ɴ",      // 115
        "ø",      // 116
        "ɵ",      // 117
        "ɸ",      // 118
        "θ",      // 119
        "œ",      // 120
        "ɶ",      // 121
        "ʘ",      // 122
        "ɹ",      // 123
        "ɺ",      // 124
        "ɾ",      // 125
        "ɻ",      // 126
        "ʀ",      // 127
        "ʁ",      // 128
        "ɽ",      // 129
        "ʂ",      // 130
        "ʃ",      // 131
        "ʈ",      // 132
        "ʧ",      // 133
        "ʉ",      // 134
        "ʊ",      // 135
        "ʋ",      // 136
        "ⱱ",      // 137
        "ʌ",      // 138
        "ɣ",      // 139
        "ɤ",      // 140
        "ʍ",      // 141
        "χ",      // 142
        "ʎ",      // 143
        "ʏ",      // 144
        "ʑ",      // 145
        "ʐ",      // 146
        "ʒ",      // 147
        "ʓ",      // 148
        "ʔ",      // 149
        "ʡ",      // 150
        "ʕ",      // 151
        "ʢ",      // 152
        "ǀ",      // 153
        "ǁ",      // 154
        "ǂ",      // 155
        "ǃ",      // 156
        "ˈ",      // 157 (primary stress)
        "ˌ",      // 158 (secondary stress)
        "ː",      // 159 (long vowel)
        "ˑ",      // 160
        "ʼ",      // 161
        "ʴ",      // 162
        "ʰ",      // 163
        "ʱ",      // 164
        "ʲ",      // 165
        "ʷ",      // 166
        "ˠ",      // 167
        "ˤ",      // 168
        "˞",      // 169
        "↓",      // 170
        "↑",      // 171
        "→",      // 172
        "↗",      // 173
        "↘",      // 174
        "'",      // 175
        "\u0329", // 176 (combining vertical line below - ̩)
        "\u2019", // 177 (right single quote)
        "ᵻ"       // 178
    ];

    private static readonly Dictionary<char, int> CharToIdMap = InitializeMap();

    private static Dictionary<char, int> InitializeMap()
    {
        var map = new Dictionary<char, int>(256);
        for (int i = 0; i < Symbols.Length; i++)
        {
            string sym = Symbols[i];
            if (!string.IsNullOrEmpty(sym))
            {
                map.TryAdd(sym[0], i);
            }
        }
        return map;
    }

    public static int[] Map(string ipa) => MapPhonemesToIds(ipa);

    /// <summary>
    /// Maps a phonetic/IPA string to a sequence of KittenTTS token IDs.
    /// Prefixes start token 0 and appends end tokens [10, 0] matching reference behavior.
    /// </summary>
    public static int[] MapPhonemesToIds(string ipa)
    {
        var ids = new List<int>(ipa.Length + 4) { 0 }; // Start token '$' (0)

        for (int i = 0; i < ipa.Length; i++)
        {
            char c = ipa[i];
            if (CharToIdMap.TryGetValue(c, out int id))
            {
                ids.Add(id);
            }
        }

        ids.Add(10); // End punctuation token '…' (10)
        ids.Add(0);  // End padding '$' (0)
        return ids.ToArray();
    }
}
