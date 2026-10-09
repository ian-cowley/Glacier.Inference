namespace Glacier.Inference.Audio.Kitten;

using System;
using System.IO;
using System.Text.Json;

/// <summary>
/// Configuration parameters for the KittenTTS neural text-to-speech architecture.
/// </summary>
public sealed class KittenConfig
{
    public int NToken { get; set; } = 179;
    public int BertEmbedDim { get; set; } = 128;
    public int BertHiddenDim { get; set; } = 768;
    public int BertFfnDim { get; set; } = 2048;
    public int BertNHeads { get; set; } = 12;
    public int BertMaxPos { get; set; } = 512;
    public int HiddenDim { get; set; } = 128;
    public int StyleDim { get; set; } = 256;
    public int LstmHidden { get; set; } = 64;
    public int PredictorConvDim { get; set; } = 128;
    public int DecoderDim { get; set; } = 256;
    public int[] GeneratorChannels { get; set; } = [256, 128, 64];
    public int[] GeneratorUpsampleRates { get; set; } = [10, 6];
    public int[] GeneratorUpsampleKernels { get; set; } = [20, 12];
    public int NHarmonics { get; set; } = 11;
    public int PostConvChannels { get; set; } = 22;
    public int SampleRate { get; set; } = 24000;
    public int MaxDuration { get; set; } = 50;

    public static KittenConfig Default => CreateNano();

    public static KittenConfig CreateNano()
    {
        return new KittenConfig();
    }

    public static KittenConfig Load(string jsonPath)
    {
        if (!File.Exists(jsonPath))
        {
            return CreateNano();
        }

        try
        {
            string json = File.ReadAllText(jsonPath);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var cfg = CreateNano();
            if (root.TryGetProperty("sample_rate", out var sr)) cfg.SampleRate = sr.GetInt32();
            if (root.TryGetProperty("n_token", out var nt)) cfg.NToken = nt.GetInt32();
            if (root.TryGetProperty("hidden_dim", out var hd)) cfg.HiddenDim = hd.GetInt32();
            if (root.TryGetProperty("style_dim", out var sd)) cfg.StyleDim = sd.GetInt32();
            if (root.TryGetProperty("decoder_dim", out var dd)) cfg.DecoderDim = dd.GetInt32();
            if (root.TryGetProperty("lstm_hidden", out var lh)) cfg.LstmHidden = lh.GetInt32();
            if (root.TryGetProperty("n_harmonics", out var nh)) cfg.NHarmonics = nh.GetInt32();
            if (root.TryGetProperty("bert_hidden", out var bh)) cfg.BertHiddenDim = bh.GetInt32();
            if (root.TryGetProperty("bert_heads", out var bheads)) cfg.BertNHeads = bheads.GetInt32();
            if (root.TryGetProperty("max_duration", out var md)) cfg.MaxDuration = md.GetInt32();

            return cfg;
        }
        catch
        {
            return CreateNano();
        }
    }
}
