namespace Glacier.Inference.Embedding;

using System;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;

/// <summary>Compute device selection for <see cref="EmbeddingGemma2Model"/>.</summary>
public enum EmbeddingDevice
{
    /// <summary>Try Direct3D 12 (Windows) or Vulkan (Android/Linux) and fall back to CPU SIMD.</summary>
    Auto,
    /// <summary>Portable SIMD (AVX-512/AVX2/NEON via Vector128/256/512) on all cores.</summary>
    Cpu,
    /// <summary>Direct3D 12 compute shaders (any D3D12 GPU, incl. Windows-on-ARM Adreno).</summary>
    D3D12,
    /// <summary>Vulkan 1.1+ compute shaders (Qualcomm Adreno on Android, Linux, Windows cross-vendor).</summary>
    Vulkan,
}

public sealed class EmbeddingGemma2Options
{
    public EmbeddingDevice Device { get; init; } = EmbeddingDevice.Auto;
    /// <summary>D3D12 adapter index; -1 = default (prefers a discrete/integrated AMD adapter as in D3D12Context).</summary>
    public int AdapterIndex { get; init; } = -1;
}

/// <summary>
/// Native CPU runtime for Google <b>EmbeddingGemma 2</b> (GGUF architecture <c>gemma-embedding2</c>):
/// a bidirectional (non-causal) Gemma 4 style encoder with interleaved sliding-window / global attention,
/// "projection-only" per-layer embeddings (PLE), mean pooling, a 512→768 output projection and L2 normalisation.
/// </summary>
/// <remarks>
/// Graph (per layer, all norms are RMSNorm, attention scale = 1.0 because Q/K are normalised):
/// <code>
///   h = attn_norm(x); q,k,v = Wq h, Wk h, Wv h
///   q = rope(q_norm(q)); k = rope(k_norm(k)); v = rms(v) (no weight)
///   x += attn_post_norm(Wo · bidirectional_attention(q,k,v))
///   x += ffn_post_norm(down(gelu(gate(ffn_norm(x))) * up(ffn_norm(x))))
///   x += per_layer_post_norm(proj(gelu(inp_gate(x)) * ple[layer]))
///   x *= layer_output_scale
/// </code>
/// where <c>ple = rms_norm(per_layer_model_proj(embed * sqrt(d)) / sqrt(d))</c>, sliced per layer.
/// Output: <c>normalize(output · mean_t(output_norm(x_t)))</c>.
/// Supports Q8_0, BF16, F16 and F32 tensors. Weights are dequantised to FP32 at load time.
/// </remarks>
public sealed unsafe class EmbeddingGemma2Model : IDisposable
{
    public const string ArchitectureName = "gemma-embedding2";

    private readonly GgufFile _gguf;
    private readonly float[] _outputProj; // [outDim, hidden] FP32 (tiny)
    private bool _disposed;

    public Gemma2Config Config { get; }
    public IGemma2Backend Backend { get; }
    public string BackendName => Backend.Name;
    public GemmaBpeTokenizer Tokenizer { get; }

    /// <summary>Native (un-truncated) embedding dimension, 768 for EmbeddingGemma 2.</summary>
    public int EmbeddingDimension => Config.OutDim;

    /// <summary>Hidden size of the transformer backbone (512).</summary>
    public int HiddenSize => Config.Hidden;

    public int LayerCount => Config.Layers;

    /// <summary>Maximum tokens processed per input; longer inputs are truncated (EOS preserved).</summary>
    public int MaxTokens { get; set; } = 8192;

    /// <summary>Matryoshka sizes supported by the model.</summary>
    public static ReadOnlySpan<int> MatryoshkaDimensions => [768, 512, 256, 128];

    public static bool IsSupported(GgufFile gguf) =>
        string.Equals(gguf.Architecture, ArchitectureName, StringComparison.Ordinal);

    public static EmbeddingGemma2Model Load(string path, EmbeddingGemma2Options? options = null) =>
        new(GgufFile.Open(path), options);

    /// <summary>Loads on the CPU backend (back-compat) unless options say otherwise.</summary>
    public EmbeddingGemma2Model(GgufFile gguf) : this(gguf, new EmbeddingGemma2Options { Device = EmbeddingDevice.Cpu }) { }

    public EmbeddingGemma2Model(GgufFile gguf, EmbeddingGemma2Options? options)
    {
        if (!IsSupported(gguf))
            throw new NotSupportedException($"Expected architecture '{ArchitectureName}', got '{gguf.Architecture}'.");
        options ??= new EmbeddingGemma2Options();
        _gguf = gguf;
        Config = Gemma2Config.FromGguf(gguf);
        Tokenizer = new GemmaBpeTokenizer(gguf);

        var info = Gemma2Tensors.Require(gguf, "output.weight", (ulong)Config.Hidden, (ulong)Config.OutDim);
        _outputProj = new float[(long)Config.Hidden * Config.OutDim];
        fixed (float* p = _outputProj)
            for (int r = 0; r < Config.OutDim; r++)
                Gemma2Tensors.DequantRow(gguf.GetTensorPointer(info), (uint)info.Type, r, Config.Hidden, p + (long)r * Config.Hidden);

        Backend = CreateBackend(gguf, Config, options);
    }

    private static IGemma2Backend CreateBackend(GgufFile gguf, Gemma2Config cfg, EmbeddingGemma2Options o)
    {
        switch (o.Device)
        {
            case EmbeddingDevice.Cpu:
                return new Gemma2CpuBackend(gguf, cfg);
            case EmbeddingDevice.Vulkan:
                return new Gemma2VulkanBackend(gguf, cfg, o.AdapterIndex >= 0 ? o.AdapterIndex : 0);
            case EmbeddingDevice.D3D12:
#if !ANDROID
                return new Gemma2D3D12Backend(gguf, cfg, o.AdapterIndex);
#else
                throw new PlatformNotSupportedException("Direct3D 12 is only supported on Windows.");
#endif
            default:
#if ANDROID
                if (VulkanContext.IsSupported)
                {
                    try { return new Gemma2VulkanBackend(gguf, cfg, o.AdapterIndex >= 0 ? o.AdapterIndex : 0); }
                    catch (Exception) { /* fall through to CPU */ }
                }
#else
                if (OperatingSystem.IsWindows())
                {
                    try { return new Gemma2D3D12Backend(gguf, cfg, o.AdapterIndex); }
                    catch (Exception) { /* fall through to Vulkan or CPU */ }
                }
                if (VulkanContext.IsSupported)
                {
                    try { return new Gemma2VulkanBackend(gguf, cfg, o.AdapterIndex >= 0 ? o.AdapterIndex : 0); }
                    catch (Exception) { /* fall through to CPU */ }
                }
#endif
                return new Gemma2CpuBackend(gguf, cfg);
        }
    }

    /// <summary>Tokenizes (with BOS/EOS) and embeds <paramref name="text"/>. Output is L2-normalised.</summary>
    /// <param name="text">Input text (include any task prefix, e.g. <c>task: search result | query: </c>).</param>
    /// <param name="dimensions">0 = full 768; otherwise a Matryoshka size (truncate then re-normalise).</param>
    public float[] Embed(string text, int dimensions = 0)
    {
        var result = new float[dimensions <= 0 ? Config.OutDim : dimensions];
        Embed(text, result);
        return result;
    }

    /// <summary>Embeds into <paramref name="destination"/>; its length selects the Matryoshka dimension.</summary>
    public void Embed(string text, Span<float> destination)
    {
        int[] ids = Tokenizer.Encode(text);
        if (ids.Length > MaxTokens)
        {
            ids = ids[..MaxTokens];
            if (Tokenizer.AddEos) ids[^1] = Tokenizer.EosTokenId;
        }
        Embed(ids, destination);
    }

    /// <summary>Embeds pre-tokenised ids (caller supplies BOS/EOS).</summary>
    public void Embed(ReadOnlySpan<int> tokens, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (destination.Length < 1 || destination.Length > Config.OutDim)
            throw new ArgumentOutOfRangeException(nameof(destination), $"Destination length must be in [1, {Config.OutDim}].");
        foreach (int id in tokens)
            if ((uint)id >= (uint)Config.Vocab) throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {id} out of range.");

        float[] pooled = Backend.ForwardPooled(tokens);
        Project(pooled, destination);
    }

    /// <summary>Projects a pooled [hidden] vector through <c>output.weight</c>, truncates and L2-normalises.</summary>
    internal void Project(ReadOnlySpan<float> pooled, Span<float> destination)
    {
        int H = Config.Hidden;
        fixed (float* w = _outputProj, x = pooled)
            for (int o = 0; o < destination.Length; o++)
                destination[o] = Simd.Dot(w + (long)o * H, x, H);
        double norm = 0;
        for (int i = 0; i < destination.Length; i++) norm += (double)destination[i] * destination[i];
        float inv = norm > 0 ? (float)(1.0 / Math.Sqrt(norm)) : 0f;
        for (int i = 0; i < destination.Length; i++) destination[i] *= inv;
    }

    /// <summary>Per-token final hidden states after <c>output_norm</c>: shape [tokens, hidden]. Exposed for validation.</summary>
    public float[] ForwardHidden(ReadOnlySpan<int> tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Backend.ForwardHidden(tokens);
    }

    // ------------------------------------------------------------------------------------------
    // Multimodal (image, video, audio) support
    // ------------------------------------------------------------------------------------------

    /// <summary>Special token ids of the EmbeddingGemma 2 vocabulary.</summary>
    internal const int BoiId = 255999, ImagePadId = 258880, EoiId = 258882;
    internal const int BoaId = 256000, AudioPadId = 258881, EoaId = 258883;
    internal const int VideoPadId = 258884;

    private Gemma4VisionTower? _vision;
    private Gemma4AudioTower? _audio;

    /// <summary>True once <see cref="AttachMultimodalProjector"/> loaded a vision tower.</summary>
    public bool SupportsImages => _vision != null;

    /// <summary>True once <see cref="AttachMultimodalProjector"/> loaded an audio tower.</summary>
    public bool SupportsAudio => _audio != null;

    /// <summary>
    /// Loads the multimodal projector GGUF (<c>mmproj-embeddinggemma-2-*.gguf</c>) and enables <see cref="EmbedImage"/>,
    /// <see cref="EmbedVideo"/>, and <see cref="EmbedAudio"/>.
    /// </summary>
    public void AttachMultimodalProjector(string mmprojPath)
    {
        var gVision = GgufFile.Open(mmprojPath);
        if (Gemma4VisionTower.IsVisionMmproj(gVision))
        {
            var old = _vision;
            _vision = new Gemma4VisionTower(gVision);
            old?.Dispose();
        }
        else
        {
            gVision.Dispose();
        }

        var gAudio = GgufFile.Open(mmprojPath);
        if (Gemma4AudioTower.IsAudioMmproj(gAudio))
        {
            var old = _audio;
            _audio = new Gemma4AudioTower(gAudio);
            old?.Dispose();
        }
        else
        {
            gAudio.Dispose();
        }

        if (_vision is null && _audio is null)
            throw new NotSupportedException("The GGUF does not contain a supported vision or audio tower.");
    }

    /// <summary>
    /// Embeds an RGB8 image (row-major, 3 bytes/pixel). The image is resized (aspect preserved, multiples of 48 px,
    /// up to 280 soft tokens), encoded by the vision tower, and the soft tokens run through the text tower:
    /// <c>[BOS, boi, soft x N, eoi, EOS]</c> → mean pool → projection → L2.
    /// </summary>
    public float[] EmbedImage(ReadOnlySpan<byte> rgb, int width, int height, int dimensions = 0)
    {
        if (_vision is null) throw new InvalidOperationException("Call AttachMultimodalProjector first to load vision weights.");
        float[] patches = Gemma4ImagePreprocessor.Preprocess(rgb, width, height, Gemma4ImagePreprocessor.DefaultMaxSoftTokens, out int gh, out int gw);
        float[] soft = _vision.Encode(patches, gh, gw);
        int n = soft.Length / 512;

        var ids = new int[n + 4];
        ids[0] = Tokenizer.BosTokenId; ids[1] = BoiId;
        for (int i = 0; i < n; i++) ids[2 + i] = ImagePadId;
        ids[n + 2] = EoiId; ids[n + 3] = Tokenizer.EosTokenId;
        return EmbedWithSoftTokens(ids, ImagePadId, soft, dimensions);
    }

    /// <summary>
    /// Embeds video frames (RGB8, each frame row-major 3 bytes/pixel). Each frame is independently encoded
    /// by the vision tower (budget 140 soft tokens/frame) and structured as:
    /// <c>[BOS, frame_0(boi, soft x K, eoi), ..., frame_{F-1}(boi, soft x K, eoi), EOS]</c> → mean pool → L2.
    /// </summary>
    public float[] EmbedVideo(IReadOnlyList<byte[]> framesRgb, int width, int height, int dimensions = 0)
    {
        if (_vision is null) throw new InvalidOperationException("Call AttachMultimodalProjector first to load vision weights.");
        if (framesRgb.Count == 0) throw new ArgumentException("Video must contain at least one frame.", nameof(framesRgb));

        const int maxSoftTokensPerFrame = 140;
        var frameSoftList = new System.Collections.Generic.List<float[]>(framesRgb.Count);
        int totalSoftTokens = 0;

        foreach (var frame in framesRgb)
        {
            float[] patches = Gemma4ImagePreprocessor.Preprocess(frame, width, height, maxSoftTokensPerFrame, out int gh, out int gw);
            float[] soft = _vision.Encode(patches, gh, gw);
            frameSoftList.Add(soft);
            totalSoftTokens += soft.Length / 512;
        }

        // Construct token IDs: BOS + frames * (boi + N*video_pad + eoi) + EOS
        int totalIds = 2 + framesRgb.Count * 2 + totalSoftTokens;
        var ids = new int[totalIds];
        int idIdx = 0;
        ids[idIdx++] = Tokenizer.BosTokenId;

        for (int f = 0; f < framesRgb.Count; f++)
        {
            ids[idIdx++] = BoiId;
            int nTokens = frameSoftList[f].Length / 512;
            for (int i = 0; i < nTokens; i++) ids[idIdx++] = VideoPadId;
            ids[idIdx++] = EoiId;
        }
        ids[idIdx++] = Tokenizer.EosTokenId;

        // Concatenate soft tokens across all frames
        var allSoft = new float[(long)totalSoftTokens * 512];
        int softOffset = 0;
        foreach (var soft in frameSoftList)
        {
            Array.Copy(soft, 0, allSoft, softOffset, soft.Length);
            softOffset += soft.Length;
        }

        return EmbedWithSoftTokens(ids, VideoPadId, allSoft, dimensions);
    }

    /// <summary>
    /// Embeds a 16kHz mono audio waveform (float32 samples in [-1, 1]). Log-mel features are extracted,
    /// encoded by the Conformer audio tower into 512-d soft tokens, and run through the text tower:
    /// <c>[BOS, boa, soft x M, eoa, EOS]</c> → mean pool → projection → L2.
    /// </summary>
    public float[] EmbedAudio(ReadOnlySpan<float> audioSamples, int dimensions = 0)
    {
        if (_audio is null) throw new InvalidOperationException("Call AttachMultimodalProjector first to load audio weights.");
        if (audioSamples.IsEmpty) throw new ArgumentException("Audio cannot be empty.", nameof(audioSamples));

        float[] logMel = Gemma4AudioPreprocessor.ExtractLogMel(audioSamples);
        int numFrames = logMel.Length / Gemma4AudioPreprocessor.MelBins;
        float[] soft = _audio.Encode(logMel, numFrames);
        int m = soft.Length / 512;

        var ids = new int[m + 4];
        ids[0] = Tokenizer.BosTokenId; ids[1] = BoaId;
        for (int i = 0; i < m; i++) ids[2 + i] = AudioPadId;
        ids[m + 2] = EoaId; ids[m + 3] = Tokenizer.EosTokenId;
        return EmbedWithSoftTokens(ids, AudioPadId, soft, dimensions);
    }

    /// <summary>Embeds a sequence where every <paramref name="placeholderId"/> is replaced (in order) by a row of <paramref name="soft"/> ([k, hidden], not scaled).</summary>
    internal float[] EmbedWithSoftTokens(int[] ids, int placeholderId, float[] soft, int dimensions = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int H = Config.Hidden, n = ids.Length;
        var lookup = (int[])ids.Clone();
        for (int i = 0; i < n; i++) if (lookup[i] == placeholderId) lookup[i] = 0;
        float[] x0 = Gemma2Embedding.Gather(_gguf, Config, lookup);
        int row = 0;
        for (int i = 0; i < n; i++)
        {
            if (ids[i] != placeholderId) continue;
            Array.Copy(soft, (long)row * H, x0, (long)i * H, H);
            row++;
        }
        if (row * H != soft.Length) throw new ArgumentException("Placeholder count does not match the number of soft tokens.");

        float[] pooled = Backend.ForwardPooledEmbeds(x0, n);
        var result = new float[dimensions <= 0 ? Config.OutDim : dimensions];
        Project(pooled, result);
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vision?.Dispose();
        _audio?.Dispose();
        Backend.Dispose();
        _gguf.Dispose();
    }
}
