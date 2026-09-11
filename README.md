# Glacier.Inference

High-Performance C# .NET 10 LLM Inference Engine & Command-Line Server Runtime.

[![CI](https://github.com/ian-cowley/Glacier.Inference/actions/workflows/publish-nuget.yml/badge.svg)](https://github.com/ian-cowley/Glacier.Inference/actions)
[![NuGet](https://img.shields.io/nuget/v/Glacier.Inference.svg)](https://www.nuget.org/packages/Glacier.Inference)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pure C# .NET 10 alternative to Ollama, vLLM, and llama.cpp. Direct memory-mapped GGUF model execution, SIMD AVX-512 / AVX2 quantized GEMV kernels (Q4_K, Q6_K, Q8_0, Q4_0, FP16), unmanaged KV-cache ring buffers, streaming terminal chat REPL, and an Ollama/OpenAI-compatible HTTP API server.

---

## Features

- **Zero-Copy GGUF Weight Mapping**: Uses `MemoryMappedFile` to instantly map multi-gigabyte models into address space in sub-100ms cold time without heap allocations.
- **Hardware SIMD Quantization Kernels**: Vectorized AVX-512 and AVX2 hardware FMA dot-products for `Q4_K`, `Q6_K`, `Q8_0`, `Q4_0`, `F16`, and `F32`.
- **Full Architecture Support**:
  - Qwen2 / Qwen2.5 (tested up to `qwen2.5:7b-instruct-32k` / 1M context)
  - Llama 3 / 3.1 / 3.2
  - Grouped Query Attention (GQA), Rotary Positional Embeddings (RoPE), QKV Bias, SwiGLU FFN, and RMSNorm.
- **Embedded BPE Tokenizer**: Reads vocabularies and BPE merge tables directly from GGUF metadata with ChatML template support.
- **Dual-Protocol HTTP Server**: Drop-in compatible with Ollama (`/api/generate`, `/api/chat`, `/api/tags`) and OpenAI (`/v1/chat/completions`, `/v1/models`).
- **Native AOT Compatible**: Sub-15ms cold startup, zero external C++ DLL dependencies.

---

## Quick Start (CLI)

```bash
# 1. Inspect any GGUF model
glacier inspect "path/to/model.gguf"

# 2. Run interactive streaming chat
glacier run "path/to/model.gguf"

# 3. Benchmark generation throughput
glacier bench "path/to/model.gguf" --tokens 32

# 4. Comparative benchmark against remote/local Ollama (e.g. NVIDIA RTX 3060)
glacier bench "path/to/model.gguf" --compare-ollama "http://192.168.1.108:11434"

# 5. Launch Ollama & OpenAI compatible HTTP server
glacier serve "path/to/model.gguf" --port 11434
```

---

## Comparative Benchmark (`qwen2.5:7b-instruct-32k`)

Measured directly against a remote Ollama daemon running on an NVIDIA GeForce RTX 3060 GPU (`192.168.1.108:11434`):

| Metric | Glacier.Inference (Pure C# .NET 10) | Ollama (NVIDIA RTX 3060 CUDA) |
| :--- | :--- | :--- |
| **Runtime** | Pure C# .NET 10 Native AOT | Go + C++ CUDA / llama.cpp |
| **Dependencies** | **Zero Native DLLs** | CUDA, cuBLAS, LibLLAMA |
| **Cold Start / Load** | **60 ms** (Memory-mapped zero-copy) | Daemon / Warm |
| **Prompt Eval Rate** | 2.8 tokens/sec | 183.1 tokens/sec |
| **Generation Rate** | 2.4 tokens/sec | 65.9 tokens/sec |
| **Semantic Fidelity** | **100% Identical predictions** | Ground Truth |

---

## C# Library API

```csharp
using Glacier.Inference.Engine;
using Glacier.Inference.Sampling;

// 1. Initialize session with sub-100ms cold start
using var session = new InferenceSession("models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf");

// 2. Stream generation with live tokens
var options = new SamplingOptions { Temperature = 0.7f, TopP = 0.9f, MaxTokens = 512 };

var result = await session.GenerateAsync(
    prompt: "Explain in two sentences what a CPU cache is.",
    options: options,
    formatChat: true,
    onToken: piece => Console.Write(piece));

Console.WriteLine($"\nThroughput: {result.Metrics.GenerationTokensPerSecond:F1} tokens/sec");
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Glacier.Inference.Cli (Host)               │
│      run  |  serve  |  inspect  |  bench                │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                 Glacier.Inference Core                  │
│  ├─ GgufFile (MemoryMappedFile zero-copy reader)        │
│  ├─ QuantKernels (AVX-512 / AVX2 Q4_K, Q6_K, Q8_0, F16) │
│  ├─ KVCache (Unmanaged contiguous ring buffer)          │
│  ├─ Qwen2Model / TransformerModel (Attention, SwiGLU)   │
│  ├─ BpeTokenizer (Direct GGUF token & merge tables)     │
│  └─ Sampler (Greedy, Temperature, Top-K, Top-P)         │
└─────────────────────────────────────────────────────────┘
```

---

## License

MIT License. (c) 2026 Ian Cowley.
