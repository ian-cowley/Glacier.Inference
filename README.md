![Glacier.Inference Banner](docs/images/glacier_inference_universal_banner.jpg)

# Glacier.Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Inference.svg)](https://www.nuget.org/packages/Glacier.Inference/)

`Glacier.Inference` is a pure C# .NET 10 runtime, Native AOT CLI (`glacier`), and OpenAI/Ollama-compatible HTTP server for running GGUF and Safetensors models without Python or C++ runtime wrappers. It binds directly to OS graphics and compute drivers (`nvcuda.dll`, `d3d12.dll`, Vulkan, HIP) and falls back to multi-threaded AVX-512, AVX2, or ARM64 NEON SIMD kernels when a GPU backend does not cover a model or weight format.

*(Backend scope note: on Linux and non-Windows hosts, the Vulkan runner currently accelerates hybrid Qwen 3.5/3.6 models using `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` weights; models with unsupported quantizations such as `IQ4_XS` log a notice and run on CPU SIMD.)*

## Install

```bash
# Library
dotnet add package Glacier.Inference

# Standalone CLI tool
dotnet tool install --global Glacier.Inference.Cli
```

## Quick Start

### C# Streaming Inference

```csharp
using Glacier.Inference.Engine;
using Glacier.Inference.Sampling;

using var session = new InferenceSession("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");

var options = new SamplingOptions
{
    Temperature = 0.7f,
    TopP = 0.9f,
    MaxTokens = 128
};

await session.GenerateAsync("Explain quicksort in C#.", options, onToken: token =>
{
    Console.Write(token);
});
```

### CLI & HTTP Server

```bash
glacier devices
glacier inspect "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
glacier run "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Explain quicksort in C#."
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --port 11434 --host 127.0.0.1
```

## Supported Models & Backends

| Model Family | Core Components Implemented | Tested GGUF / Safetensors Weights |
| :--- | :--- | :--- |
| **Llama 3 / 3.1 / 3.2 / 3.3** | GQA, precomputed RoPE tables (128k context), SwiGLU | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` |
| **DeepSeek-V2 / V3 / R1** | Multi-Head Latent Attention (MLA), decoupled RoPE + YaRN, MoE | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf`, `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` |
| **Qwen 2.5 / 3 / 3.5 / 3.6 / 3.8** | Dense/MoE GQA, Gated DeltaNet SSM, partial MRoPE, NariQ `TRL1..TRL6` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, `Qwen3-4B`, `Qwen3.6-35B-A3B`, `Qwen3.8-27B-NariQ` |
| **Gemma 4 & EmbeddingGemma 2** | Dual MoE, interleaved sliding-window attention, 768D multimodal embeddings | `gemma-4-26B-A4B-it`, `google_gemma-4-E4B-it`, `embeddinggemma-2-Q8_0` |
| **Phi-3 / Phi-4 & Mistral / Devstral** | Fused QKV / FFN projections, NeOX & `10^9`-base RoPE | `phi-4-Q4_K_M.gguf`, `Devstral-Small-2505` |
| **FLUX.1 & Wan 2.1 Media** | 57-block DiT + tiled VAE (image); 30-block 3D DiT + 3D VAE (video) | `flux1-schnell-Q4_K_S.gguf` + `ae.safetensors`, `Wan2.1-T2V-1.3B-Q4_K_M.gguf` |
| **Speech & Audio** | KittenTTS / Kokoro glottal formant TTS, Whisper STT client | Built-in formant voices (`af_heart`, `bm_george`) |

Hardware execution targets NVIDIA SASS (`nvcuda.dll` / `libcuda.so.1`), Direct3D 12 Wave32 (`d3d12.dll`), Vulkan 1.1/1.3 SPIR-V, AMD HIP (`amdhip64.dll`), multi-GPU layer splitting (`--split`), and AVX-512 / AVX2 / ARM64 NEON CPU kernels.

## Measured Benchmarks

All values below were measured directly on physical machines without simulated steps or estimates:

| Workload | Hardware & Backend | Measured Throughput / Latency | Source |
| :--- | :--- | ---: | :--- |
| `Qwen2.5-7B-Instruct` (`Q4_K_M`) | RTX 4060 8 GB (CUDA SASS) | **106.71 tok/s** pp / **42.90–64.05 tok/s** tg | [benchmarks.md](docs/benchmarks.md) |
| `Qwen3-4B-Instruct-2507` (`Q4_K_M`) | RTX 3060 12 GB (CUDA Graph ON) | **114.18 tok/s** pp / **71.03 tok/s** tg | [benchmarks.md](docs/benchmarks.md) |
| `Qwen3.8-27B-NariQ` (`2.5bpw`) | RTX 3060 12 GB (CUDA / Vulkan) | **19.1–19.2 tok/s** live (CUDA) / **8.8 tok/s** (Vulkan) | [benchmarks.md](docs/benchmarks.md) |
| `Gemma-4-26B-A4B-it` (`Q4_K_M`, MoE) | Radeon 890M 64 GB Linux (Vulkan) | **285.41 tok/s** pp / **29.00 tok/s** tg | [benchmarks.md](docs/benchmarks.md) |
| `Qwen3.6-35B-A3B-UD` (`Q4_K_M`) | Radeon 890M 64 GB Linux (Vulkan vs CPU) | **9.4 tok/s** (Vulkan) vs **2.7 tok/s** (24T CPU) | [VULKAN_BACKEND.md](docs/VULKAN_BACKEND.md) |
| `VecDotF32` / `VecDotQ4_K` / `VecDotQ8_0` (`Dim=3584`) | Ryzen AI 9 HX 370 (`.NET 10.0.15`) | **137.6 ns** / **425.2 ns** / **353.6 ns** (`0 B` alloc) | `SimdDotBenchmarks` |
| `MatVecMul_Q4_K` / `MatVecMul_Q8_0` (`3584x3584`) | Ryzen AI 9 HX 370 (`.NET 10.0.15`) | **284.3 us** / **245.5 us** | `GemvBenchmarks` |

Reproduce CPU kernel microbenchmarks locally:

```bash
dotnet run --project benchmarks/Glacier.Inference.Benchmarks/Glacier.Inference.Benchmarks.csproj -c Release -- --job short --filter *
```

## Documentation

- [Usage Guide](docs/usage.md) — CLI commands (`devices`, `inspect`, `run`, `bench`, `serve`, `image`, `video`, `voice`), API auth, and troubleshooting
- [Architecture](docs/architecture.md) — memory-mapped weights, ring-buffer KV cache, quantization blocks, HAL backends, and ADRs
- [Benchmarks](docs/benchmarks.md) — cross-machine physical runs, multimodal VAE timings, and BenchmarkDotNet tables
- [Vulkan Backend Notes](docs/VULKAN_BACKEND.md) — Linux RADV kernel pipeline, memory placement constraints, and re-quantization measurements
- [MLA Decoupled RoPE](docs/explanation/mla_decoupled_rope.md) & [YaRN RoPE Scaling](docs/explanation/yarn_rope_scaling.md) — positional encoding specs
- [Chat Formatting Protocols](docs/reference/chat_formatting_protocols.md) — ChatML and model template rules
- [Changelog](CHANGELOG.md) — release history

## License & Credits

Licensed under the [MIT License](LICENSE).

Developed by Ian Cowley, Antigravity (Google DeepMind) and Claude.
