namespace Glacier.Inference.Image.Gguf;

using System;
using System.Collections.Generic;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;

/// <summary>
/// Pre-trained Diffusion Model loaded directly from GGUF format (Flux.1, SDXL, SD3.5, SD-Turbo).
/// Maps model tensor topology, multi-stream transformer blocks, and latent VAE decoders in pure C# .NET 10.
/// </summary>
public sealed class DiffusionGgufModel : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly UniversalArchitecture _diffusionArch;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _doubleBlocks;
    private readonly int _singleBlocks;
    private readonly int _inChannels;
    private readonly int _patchSize;
    private readonly bool _hasVae;
    private readonly bool _hasTextEncoder;
    private readonly string _modelName;
    private bool _disposed;

    public GgufFile Gguf => _gguf;
    public UniversalArchitecture DiffusionArch => _diffusionArch;
    public int HiddenDim => _hiddenDim;
    public int NumLayers => _numLayers;
    public int DoubleBlocks => _doubleBlocks;
    public int SingleBlocks => _singleBlocks;
    public int InChannels => _inChannels;
    public int PatchSize => _patchSize;
    public bool HasVae => _hasVae;
    public bool HasTextEncoder => _hasTextEncoder;
    public string ModelName => _modelName;

    public string RecommendedSchedule => _diffusionArch switch
    {
        UniversalArchitecture.Flux => "FlowMatching",
        UniversalArchitecture.SD3 => "FlowMatching",
        UniversalArchitecture.Wan => "FlowMatching",
        _ => "EulerA"
    };

    public int RecommendedSteps => _diffusionArch switch
    {
        UniversalArchitecture.Flux when _modelName.Contains("schnell", StringComparison.OrdinalIgnoreCase) => 4,
        UniversalArchitecture.Flux => 20,
        UniversalArchitecture.Wan => 20,
        UniversalArchitecture.StableDiffusion when _modelName.Contains("turbo", StringComparison.OrdinalIgnoreCase) => 4,
        UniversalArchitecture.StableDiffusion when _modelName.Contains("lightning", StringComparison.OrdinalIgnoreCase) => 4,
        UniversalArchitecture.SD3 => 28,
        _ => 20
    };

    public int RecommendedResolution => _diffusionArch switch
    {
        UniversalArchitecture.Flux => 1024,
        UniversalArchitecture.SD3 => 1024,
        UniversalArchitecture.Wan => 512,
        UniversalArchitecture.StableDiffusion when _modelName.Contains("xl", StringComparison.OrdinalIgnoreCase) => 1024,
        _ => 512
    };

    private DiffusionGgufModel(GgufFile gguf, string filePath)
    {
        _gguf = gguf;
        _modelName = System.IO.Path.GetFileNameWithoutExtension(filePath);
        _diffusionArch = ModelArchitectureDetector.Detect(gguf.Architecture, gguf);

        // Analyze topology
        int dBlocks = 0;
        int sBlocks = 0;
        int layers = 0;

        foreach (var key in _gguf.Tensors.Keys)
        {
            if (key.Contains("double_blocks.") && key.EndsWith(".img_attn.qkv.weight")) dBlocks++;
            if (key.Contains("single_blocks.") && key.EndsWith(".linear1.weight")) sBlocks++;
            if (key.Contains("input_blocks.") && key.EndsWith(".0.weight")) layers++;
        }

        _doubleBlocks = dBlocks;
        _singleBlocks = sBlocks;
        _numLayers = dBlocks > 0 ? (dBlocks + sBlocks) : Math.Max(layers, (int)_gguf.BlockCount);

        // Dimension inference
        if (_gguf.EmbeddingLength > 0)
        {
            _hiddenDim = (int)_gguf.EmbeddingLength;
        }
        else if (_diffusionArch == UniversalArchitecture.Flux)
        {
            _hiddenDim = 3072; // Flux.1 hidden dim
        }
        else if (_diffusionArch == UniversalArchitecture.SD3)
        {
            _hiddenDim = 1536; // SD3 hidden dim
        }
        else if (_diffusionArch == UniversalArchitecture.Wan)
        {
            _hiddenDim = 1536; // Wan 1.3B hidden dim
        }
        else
        {
            _hiddenDim = 1280; // SDXL hidden dim
        }

        // Channels & Patching
        _inChannels = _diffusionArch switch
        {
            UniversalArchitecture.Flux => 16,
            UniversalArchitecture.SD3 => 16,
            UniversalArchitecture.Wan => 16,
            _ => 4
        };

        _patchSize = _diffusionArch switch
        {
            UniversalArchitecture.Flux => 2,
            UniversalArchitecture.SD3 => 2,
            UniversalArchitecture.Wan => 2,
            _ => 1
        };

        // Subsystem discovery
        foreach (var key in _gguf.Tensors.Keys)
        {
            if (key.StartsWith("vae.") || key.StartsWith("first_stage_model.") || key.StartsWith("ae."))
                _hasVae = true;
            if (key.StartsWith("text_encoders.") || key.StartsWith("cond_stage_model."))
                _hasTextEncoder = true;
        }
    }

    /// <summary>
    /// Loads and memory-maps a Diffusion GGUF model file.
    /// </summary>
    public static DiffusionGgufModel Open(string filePath)
    {
        var gguf = GgufFile.Open(filePath);
        return new DiffusionGgufModel(gguf, filePath);
    }

    public GgufTensorInfo? FindTensor(string pattern)
    {
        foreach (var kvp in _gguf.Tensors)
        {
            if (kvp.Key.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                return kvp.Value;
        }
        return null;
    }


    public void Dispose()
    {
        if (!_disposed)
        {
            _gguf.Dispose();
            _disposed = true;
        }
    }
}
