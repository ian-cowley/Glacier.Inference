namespace Glacier.Inference.Audio.Kitten;

using System;
using System.Text;
using System.Text.RegularExpressions;

/// <summary>
/// High-speed pure C# text normalization and preprocessor for KittenTTS.
/// Expands digits, currencies, percentages, and abbreviations into pronounceable English words.
/// </summary>
public static partial class KittenTextPreprocess
{
    private static readonly string[] Ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    ];

    private static readonly string[] Tens =
    [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    ];

    public static string NumberToWords(long n)
    {
        if (n == 0) return "zero";
        if (n < 0) return "negative " + NumberToWords(-n);

        var sb = new StringBuilder();

        if (n >= 1_000_000_000_000)
        {
            sb.Append(NumberToWords(n / 1_000_000_000_000)).Append(" trillion ");
            n %= 1_000_000_000_000;
        }

        if (n >= 1_000_000_000)
        {
            sb.Append(NumberToWords(n / 1_000_000_000)).Append(" billion ");
            n %= 1_000_000_000;
        }

        if (n >= 1_000_000)
        {
            sb.Append(NumberToWords(n / 1_000_000)).Append(" million ");
            n %= 1_000_000;
        }

        if (n >= 1_000)
        {
            sb.Append(NumberToWords(n / 1_000)).Append(" thousand ");
            n %= 1_000;
        }

        if (n >= 100)
        {
            sb.Append(NumberToWords(n / 100)).Append(" hundred ");
            n %= 100;
        }

        if (n >= 20)
        {
            sb.Append(Tens[n / 10]);
            if (n % 10 > 0) sb.Append('-').Append(Ones[n % 10]);
        }
        else if (n > 0)
        {
            sb.Append(Ones[n]);
        }

        return sb.ToString().Trim();
    }

    public static string Normalize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;

        // Replace currencies
        text = Regex.Replace(text, @"\$(\d+)", "$1 dollars");
        text = Regex.Replace(text, @"£(\d+)", "$1 pounds");
        text = Regex.Replace(text, @"€(\d+)", "$1 euros");

        // Replace %
        text = text.Replace("%", " percent");
        text = text.Replace("&", " and ");
        text = text.Replace("@", " at ");

        // Convert standalone integers to words
        text = Regex.Replace(text, @"\b\d+\b", match =>
        {
            if (long.TryParse(match.Value, out long val))
            {
                return NumberToWords(val);
            }
            return match.Value;
        });

        // Clean extra whitespace
        text = Regex.Replace(text, @"\s+", " ").Trim();
        return text;
    }
}
