# Glacier.Inference User Manual

## Contents

1. [Requirements & Installation](#1-requirements--installation)
2. [C# Library API](#2-c-library-api)
3. [CLI Command Reference](#3-cli-command-reference)
4. [HTTP Microservice (`glacier serve`) & Authentication](#4-http-microservice-glacier-serve--authentication)
5. [Quantization Formats & VRAM Sizing](#5-quantization-formats--vram-sizing)
6. [Multi-GPU Splitting & Linux Vulkan Execution](#6-multi-gpu-splitting--linux-vulkan-execution)
7. [Video, Image, Vision & Voice Commands](#7-video-image-vision--voice-commands)
8. [Production Deployment (`systemd`) & Troubleshooting](#8-production-deployment-systemd--troubleshooting)
9. [Building & Testing](#9-building--testing)

### Topic Manuals

- [Speculative Decoding (`lookup`, `mtp`, `dflash`, `adaptive`)](manual/speculative-decoding.md) — draft providers, cost-gated routing, MoE tail skip, and CLI/C# usage
- [Multimodal Embeddings, Image, Video, Vision & Voice](manual/embeddings-and-media.md) — `EmbeddingGemma2Model`, FLUX.1 image flags, Wan 2.1 video synthesis, Vision, KittenTTS/Kokoro TTS, and Whisper STT
- [Memory Tiers, Pipelines, NPU Sidecar & Android](manual/memory-pipelines-and-npu.md) — `TieredVirtualMemoryManager`, `PipelineSession`, `NpuSidecarDiscovery`, diagnostics, settings, and Android support
- [Continuum Memory, KL Evaluation, Bounce Proxy & Formats](manual/continuum-evaluation-and-formats.md) — HOPE/Titans sleep memory, `eval-kl`, `bounce` proxy, `GgufFile`, `SafetensorsFile`, tokenizers, and `Sampler`

---

## 1. Requirements & Installation

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 (x64/ARM64), Linux (glibc >= 2.31), Android 12+ | Windows 11 or Ubuntu 24.04 LTS |
| **CPU** | x86-64 with AVX2 or ARM64-v8a | x86-64 with AVX-512 and FMA |
| **GPU (optional)** | Direct3D 12 Feature Level 12.0 or Vulkan 1.1 | NVIDIA RTX 30/40+ (driver >= 535) or AMD RDNA 2/3.5 APU |
| **RAM** | 8 GB | 32 GB+ (or unified LPDDR5X for large MoE models) |

### Three Ways to Install

1. **Standalone Release Archive:** Download the prebuilt executable archive for your platform from GitHub Releases and add it to `PATH`, then verify with `glacier devices`.
2. **.NET Global CLI Tool:**
   ```bash
   dotnet tool install --global Glacier.Inference.Cli
   glacier --help
   ```
3. **C# Library via NuGet:**
   ```bash
   dotnet add package Glacier.Inference
   ```

---

## 2. C# Library API

### `InferenceSession` & `SamplingOptions`

`InferenceSession` memory-maps a GGUF/Safetensors file, initializes the tokenizer and target backend, and streams tokens via `GenerateAsync`:

```csharp
using Glacier.Inference.Engine;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;

using var session = new InferenceSession(
    modelPath: "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    maxSeqLen: 4096,
    device: "auto",
    engine: InferenceEngineType.Auto,
    kvPrecision: KvCachePrecision.Fp8,
    split: null);

var options = new SamplingOptions
{
    Temperature = 0.7f,
    TopP = 0.9f,
    TopK = 40,
    RepetitionPenalty = 1.1f,
    MaxTokens = 256
};

// Callback overload
await session.GenerateAsync("Summarize quicksort.", options, formatChat: true, onToken: Console.Write);

// Async backpressure overload
await session.GenerateAsync("Summarize quicksort.", options, formatChat: true, async token =>
{
    await Console.Out.WriteAsync(token);
});
```

- Encoder-only `gemma-embedding2` models throw `NotSupportedException` in `InferenceSession` and should be loaded directly via `Glacier.Inference.Embedding.EmbeddingGemma2Model`.
- Inspect detected hardware programmatically via `DeviceManager.GetDevices()`, `DeviceManager.GetNpuDevices()`, `DeviceManager.GetOptimalDevice()`, and `DeviceManager.ResolveDevice(query)`.

---

## 3. CLI Command Reference

```text
Usage: glacier <command> [options]
```

| Command | Function | Example |
| :--- | :--- | :--- |
| `devices` | Lists detected GPUs, NPUs, memory pools, and supported engines | `glacier devices` |
| `config` | Displays or updates defaults in `~/.glacier/settings.json` | `glacier config --device nvidia-rtx-4060 --engine baremetal` |
| `inspect` | Prints GGUF/Safetensors header metadata, context limit, and tensor inventory | `glacier inspect "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"` |
| `run` | Runs interactive chat REPL or single prompt | `glacier run "models/model.gguf" "Write a C# dot product." --temp 0.2 --ctx 4096` |
| `bench` | Measures prefill/decode tok/s (supports `--compare-ollama`) | `glacier bench "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --tokens 64` |
| `serve` | Launches HTTP service compatible with Ollama and OpenAI | `glacier serve "models/model.gguf" --port 11434 --host 127.0.0.1` |
| `eval-kl` | Evaluates KL divergence and top-1 agreement across quantized models | `glacier eval-kl --help` |
| `bounce` | Starts a multi-node peer-critic + compiler-verified proxy | `glacier bounce --primary http://127.0.0.1:18080` |
| `video` | Generates APNG, GIF, or AVI video via Wan 2.1 | `glacier video generate "Prompt" --format all -o out.apng` |
| `image` | Generates images via FLUX.1 DiT (`generate`, `info`, `demo`) | `glacier image generate "Prompt" -w 1024 -h 1024 --steps 4 -o out.png` |
| `vision` | Experimental `VisionPipeline` image patch analysis | `glacier vision query scene.png "Describe scene" --synthetic-demo` |
| `voice` | Synthesizes KittenTTS/Kokoro speech or runs Whisper STT | `glacier voice tts "Hello world" --voice af_heart` |

### Common Options (`run`, `bench`, `serve`)

| Option | Accepted Values | Default | Effect |
| :--- | :--- | :--- | :--- |
| `--device, -d` | `nvidia-rtx-4060`, `amd-890m`, `cpu`, `0`, `auto` | `auto` | Target compute device |
| `--engine` | `baremetal`, `directml`, `vulkan`, `cpu`, `auto` | `auto` | Compute backend selection |
| `--kv-precision` | `auto`, `fp16`, `fp8`, `q8_0`, `q4_0`, `fp32` | `auto` | KV cache storage precision (`fp8` halves KV memory vs `fp16`) |
| `--ctx, -c` | Integer (`2048`, `4096`, `8192`, ...) | `2048` | Maximum sequence length |
| `--tokens, -n` | Integer | `512` (`run`) | Maximum generated tokens per turn |
| `--split` | `<dev1>:<layers>,<dev2>:<layers>` or `auto` | None | Splits layers across accelerators |
| `--lora` | `<[name=]path>` (repeatable on `serve`) | None | Attaches a LoRA adapter binary or adapter directory |
| `--ple-io` | `auto`, `direct`, `mmap` | `auto` | Per-layer embedding transport mode |
| `--host-arena-gib` | Float (`0` disables) | `auto` | Host-RAM expert arena cap in GiB |
| `--tiered-moe` | Flag | Off | Forces D3D12 GPU + CPU host-RAM expert slot pooling |
| `--speculative` | `off`, `lookup`, `mtp`, `dflash`, `adaptive` | `off` | Speculative draft verification mode ([topic manual](manual/speculative-decoding.md)) |
| `--blocks` / `--batch-size, -b` | Integers (`serve`) | `-` / `16` | PagedAttention KV block count and continuous batch limit |
| `--max-think-tokens` | `auto`, `off`, `0`, `N` (`serve`) | `auto` | Caps `<think>` tokens and auto-closes reasoning tags |
| `--sleep-memory` | Flag (`serve`) | Off | Enables HOPE + MemEx offline sleep consolidation ([topic manual](manual/continuum-evaluation-and-formats.md)) |
| `--temp` | Float | `0.7` | Sampling temperature (`0` for greedy) |
| `--top-p` | Float | `0.9` | Nucleus probability cutoff |
| `--top-k` | Integer | `40` | Top-K candidate cutoff |
| `--repeat-penalty` | Float | `1.1` | Penalty applied to recent tokens |

Compare against a local Ollama daemon side-by-side:

```bash
glacier bench "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --tokens 64 --compare-ollama "http://127.0.0.1:11434" --compare-model "qwen2.5:7b"
```

---

## 4. HTTP Microservice (`glacier serve`) & Authentication

`glacier serve` implements Ollama and OpenAI endpoints on the same listener (default `http://127.0.0.1:11434`):

- **Ollama API:** `GET /api/tags`, `POST /api/generate` (`application/x-ndjson`), `POST /api/chat`
- **OpenAI API:** `GET /v1/models`, `POST /v1/chat/completions` (`text/event-stream` SSE), `POST /v1/completions`

> [!WARNING]
> If you bind to `0.0.0.0` or `::` without setting `GLACIER_API_KEY`, any reachable network client can submit inference requests. Always set `GLACIER_API_KEY` before exposing the socket outside localhost. Clients must then pass `Authorization: Bearer <key>` or `x-api-key: <key>`, validated via `CryptographicOperations.FixedTimeEquals`.

```powershell
# Windows PowerShell
$env:GLACIER_API_KEY = "<your_api_key>"
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --host 0.0.0.0 --port 8080 --kv-precision fp8
```

```bash
# Linux / macOS
export GLACIER_API_KEY="<your_api_key>"
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --host 0.0.0.0 --port 8080 --kv-precision fp8
```

### Calling With `curl`, OpenAI Python SDK & LangChain

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_api_key>" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "What is an AVX-512 vector register?"}
    ],
    "temperature": 0.5,
    "stream": true
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="<your_api_key>")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain zero-copy memory mapping."}],
    stream=True,
)
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="<your_api_key>", model="default")
print(llm.invoke("Summarize Amdahl's Law.").content)
```

---

## 5. Quantization Formats & VRAM Sizing

### Supported Super-Block Types

| Block Format | Bits / Weight | Block Size | Bytes / Block | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `BlockQ8_0` | 8.50 | 32 | 34 | High-precision reference quant |
| `BlockQ6_K` | 6.56 | 256 | 210 | High quality with smaller memory than Q8_0 |
| `BlockQ4_K` | 4.50 | 256 | 144 | Default speed/accuracy balance |
| `BlockIQ3_S` | 3.44 | 256 | 110 | Non-linear codebook compression for larger models on 8 GB cards |
| `BlockQ2_0` | 2.50 | 64 | 20 | 4-byte aligned for Direct3D 12 coalesced loads |
| `BlockTernary256` | 2.125 | 256 | 68 | 1.58-bit ternary weights (`-1, 0, +1`) |
| `TRL1..TRL6` (`NariQ`) | 2.5–3.0 | Tile | — | Supported on `Qwen3.8` / `Qwen3.5` CUDA & Vulkan paths |

### Estimating Required VRAM

Total device memory combines the weight footprint and the KV cache:

$$\text{VRAM}_{\text{total}} = \text{FileSize}_{\text{weights}} + \left( 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C \times B_{\text{prec}} \right)$$

where $L$ is transformer layer count, $H_{\text{kv}}$ is KV head count, $D_{\text{head}}$ is head dimension, $C$ is context length (`--ctx`), and $B_{\text{prec}}$ is `2.0` bytes for `fp16` or `1.0` byte for `fp8`.

---

## 6. Multi-GPU Splitting & Linux Vulkan Execution

### Layer Splitting (`--split`)

Distribute layers across a discrete GPU and an integrated APU (for example a 28-layer 14B model split evenly):

```bash
glacier run "models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf" --split "nvidia-rtx-4060:14,amd-890m:14"

# Or let Glacier allocate layers automatically across detected devices
glacier serve "models/Large-30B-Model-Q4_K_M.gguf" --split auto
```

### Linux Vulkan (`--engine vulkan`)

On Linux with Mesa RADV or proprietary Vulkan drivers (verify with `vulkaninfo --summary`):

```bash
glacier run "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "Hello" --device amd-890m --engine vulkan --ctx 2048
```

If a model contains tensor formats the Vulkan runner does not implement (such as `IQ4_XS`), Glacier logs the reason and runs on CPU SIMD. See [Vulkan Backend Notes](VULKAN_BACKEND.md) for weight conversion and BIOS UMA carve-out details.

---

## 7. Video, Image, Vision & Voice Commands

### Video Generation (`glacier video generate`)

Uses Wan 2.1 3D-RoPE flow matching + 16-channel 3D Causal VAE:

```bash
# 15-second video (240 frames @ 16 FPS), exporting APNG, GIF, and AVI
glacier video generate "Icy glacier fjords under shimmering aurora borealis" --width 832 --height 480 --steps 25 --duration 15 --fps 16 --motion pan-right --format all --out glacier_pan.apng

# Animate an input still image
glacier video generate "Slow camera zoom into the scene" --image "input_photo.png" --motion zoom-in --format all --out animated.apng
```

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--width, -w` / `--height, -h` | `832` / `480` | Frame dimensions (multiples of 16) |
| `--frames` / `--duration` | `17` / `1.06` | Frame count (`duration` overrides frames using `round(duration * fps)`) |
| `--fps` / `--steps` / `--guidance` | `16` / `25` / `5.0` | Frame rate, Euler ODE solver steps, classifier-free guidance scale |
| `--motion` | `pan-right` | Camera motion (`pan-left`, `pan-right`, `tilt-up`, `tilt-down`, `zoom-in`, `zoom-out`, `static`, `fluid`) |
| `--format` / `--out, -o` | `apng` / `glacier_video.apng` | Output format (`apng`, `gif`, `avi`, or `all`) |

### Image Generation (`glacier image`)

```bash
glacier image generate "A dragon perched on a crystal peak at twilight" -w 1024 -h 1024 --steps 4 -o dragon.png
glacier image info "models/flux1-schnell-Q4_K_S.gguf"
glacier image demo --out "crystal_fortress.png"
```

---

## 8. Production Deployment (`systemd`) & Troubleshooting

### Linux `systemd` Unit (`/etc/systemd/system/glacier-inference.service`)

```ini
[Unit]
Description=Glacier Inference HTTP Service
After=network.target

[Service]
Type=simple
User=glacier
WorkingDirectory=/opt/glacier
ExecStart=/opt/glacier/glacier serve /opt/glacier/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 11434 --host 0.0.0.0
Environment=GLACIER_API_KEY=<your_api_key>
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now glacier-inference
```

### Troubleshooting

- **`OutOfMemoryException` on startup:** Pass `--kv-precision fp8`, reduce `--ctx 2048`, or pass `--split auto` to distribute layers across devices.
- **Windows `DXGI_ERROR_DEVICE_RESET`:** Windows display adapters abort compute dispatches lasting over ~2 seconds. Pass `--engine directml` on display-attached GPUs and reserve `--engine baremetal` for compute-only cards.
- **`address already in use` on `:11434`:** Stop any running Ollama service or bind Glacier to `--port 8080`.

---

## 9. Building & Testing

```bash
dotnet build src/Glacier.Inference.Cli/Glacier.Inference.Cli.csproj -c Release
dotnet test tests/Glacier.Inference.Tests/Glacier.Inference.Tests.csproj -c Release -f net10.0
```
