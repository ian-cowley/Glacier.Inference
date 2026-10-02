namespace Glacier.Inference.Model;

using System;
using Glacier.Inference.Gguf;

/// <summary>
/// Universal LLM foundation model architecture families supported natively by Glacier.Inference.
/// </summary>
public enum UniversalArchitecture
{
    /// <summary>
    /// Meta LLaMA 3, 3.1, 3.2, 3.3 family with precomputed RoPE frequency tables and zero attention bias.
    /// </summary>
    Llama,

    /// <summary>
    /// Alibaba Qwen 2 and Qwen 2.5 family with standard GQA and fused/unfused QKV biases.
    /// </summary>
    Qwen,

    /// <summary>
    /// Microsoft Phi-3 and Phi-4 family with fused QKV and fused Gate/Up SwiGLU MLP.
    /// </summary>
    Phi,

    /// <summary>
    /// DeepSeek-V2, V3, and R1 family with Multi-Head Latent Attention (MLA), Decoupled RoPE, and Shared Expert MoE.
    /// </summary>
    DeepSeek,

    /// <summary>
    /// Mistral and Devstral family with extreme RoPE base (up to 10^9) and 128k context GQA.
    /// </summary>
    Mistral,

    /// <summary>
    /// Hybrid State Space Model (SSM) / Gated DeltaNet (GDN) linear attention architecture (e.g. Qwen 3.5 / 3.6 GDN).
    /// </summary>
    HybridSsm,

    /// <summary>
    /// Google Gemma 2 &amp; 4 family with interleaved Sliding Window Attention (ISWA), dual shared+MoE MLP, logit soft-capping, and GeLU-GLU.
    /// </summary>
    Gemma4,

    /// <summary>
    /// Alias for Google Gemma family.
    /// </summary>
    Gemma = Gemma4,

    /// <summary>
    /// Flux.1 flow-matching Diffusion Transformer (DiT) architecture with double and single stream blocks.
    /// </summary>
    Flux,

    /// <summary>
    /// Stable Diffusion 1.5, 2.1, SDXL, and SD-Turbo UNet latent diffusion models.
    /// </summary>
    StableDiffusion,

    /// <summary>
    /// Stable Diffusion 3 / 3.5 MMDiT architecture.
    /// </summary>
    SD3,

    /// <summary>
    /// Generic or custom Transformer architecture.
    /// </summary>
    Generic
}

/// <summary>
/// Detects universal model architecture family from GGUF metadata and tensor topology.
/// </summary>
public static class ModelArchitectureDetector
{
    public static bool HasSsmTensors(GgufFile? gguf)
    {
        if (gguf?.Tensors == null) return false;
        foreach (var key in gguf.Tensors.Keys)
        {
            if (key.Contains(".ssm_") || key.Contains(".linear_attn"))
                return true;
        }
        return false;
    }

    public static UniversalArchitecture Detect(string arch, GgufFile? gguf)
    {
        // 1. Diffusion Model Architectures
        if (string.Equals(arch, "flux", StringComparison.OrdinalIgnoreCase) ||
            (gguf?.Tensors != null && (gguf.Tensors.ContainsKey("double_blocks.0.img_attn.qkv.weight") ||
                                      gguf.Tensors.ContainsKey("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"))))
        {
            return UniversalArchitecture.Flux;
        }

        if (string.Equals(arch, "sd3", StringComparison.OrdinalIgnoreCase) ||
            (gguf?.Tensors != null && (gguf.Tensors.ContainsKey("joint_blocks.0.x_block.attn.qkv.weight") ||
                                      gguf.Tensors.ContainsKey("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight"))))
        {
            return UniversalArchitecture.SD3;
        }

        if (string.Equals(arch, "sd1", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(arch, "sdxl", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(arch, "sd2", StringComparison.OrdinalIgnoreCase) ||
            (gguf?.Tensors != null && (gguf.Tensors.ContainsKey("model.diffusion_model.input_blocks.0.0.weight") ||
                                      gguf.Tensors.ContainsKey("diffusion_model.input_blocks.0.0.weight"))))
        {
            return UniversalArchitecture.StableDiffusion;
        }

        if (string.Equals(arch, "gemma4", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(arch, "gemma2", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(arch, "gemma", StringComparison.OrdinalIgnoreCase) ||
            arch.StartsWith("gemma", StringComparison.OrdinalIgnoreCase))
        {
            return UniversalArchitecture.Gemma4;
        }

        if (HasSsmTensors(gguf))
        {
            return UniversalArchitecture.HybridSsm;
        }

        if (string.Equals(arch, "deepseek2", StringComparison.OrdinalIgnoreCase) ||
            arch.StartsWith("deepseek", StringComparison.OrdinalIgnoreCase) ||
            (gguf?.IsMla == true))
        {
            return UniversalArchitecture.DeepSeek;
        }

        if (string.Equals(arch, "phi3", StringComparison.OrdinalIgnoreCase) ||
            arch.StartsWith("phi", StringComparison.OrdinalIgnoreCase) ||
            (gguf?.Tensors != null && gguf.Tensors.ContainsKey("blk.0.attn_qkv.weight")))
        {
            return UniversalArchitecture.Phi;
        }

        if (string.Equals(arch, "qwen2", StringComparison.OrdinalIgnoreCase) ||
            arch.StartsWith("qwen", StringComparison.OrdinalIgnoreCase))
        {
            return UniversalArchitecture.Qwen;
        }

        if (string.Equals(arch, "mistral", StringComparison.OrdinalIgnoreCase))
        {
            return UniversalArchitecture.Mistral;
        }

        if (string.Equals(arch, "llama", StringComparison.OrdinalIgnoreCase))
        {
            // Devstral / Mistral-derived models often identify as llama with 1e9 rope freq base or mistral vocab
            if (gguf != null && gguf.RopeFreqBase >= 1e8f)
            {
                return UniversalArchitecture.Mistral;
            }
            return UniversalArchitecture.Llama;
        }

        return UniversalArchitecture.Generic;
    }
}
