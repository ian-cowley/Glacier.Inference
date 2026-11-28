# Glacier.Inference Usage Guide

This guide covers CLI commands, server deployment and authentication, multi-GPU pipeline splitting, image/video/speech workflows, and common troubleshooting steps for `Glacier.Inference`.

## Requirements & Installation

- **OS:** Windows 10/11 (x64 or ARM64), Linux (glibc >= 2.31), or Android 12+
- **Runtime:** `.NET 10` SDK (or prebuilt native executable from GitHub Releases)
- **Drivers (optional for GPU):** NVIDIA driver >= 535 (`nvcuda.dll`), Windows D3D12 Display Driver, or Mesa RADV / Vulkan >= 1.1

```bash
# Install C# library
dotnet add package Glacier.Inference

# Or install the global CLI
dotnet tool install --global Glacier.Inference.Cli
```

## CLI Command Reference

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `devices` | Lists detected GPUs, NPUs, memory capacity, and supported engines | `glacier devices` |
| `config` | Reads or sets default device/engine preferences in `~/.glacier/settings.json` | `glacier config --device nvidia-rtx-4060 --engine baremetal` |
| `inspect` | Prints GGUF/Safetensors header metadata, context length, and tensor formats | `glacier inspect "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"` |
| `run` | Runs an interactive REPL or single prompt (`--temp`, `--top-p`, `--top-k`, `--ctx`, `--repeat-penalty`) | `glacier run "models/model.gguf" "Prompt" --ctx 4096` |
| `bench` | Measures local prefill/decode throughput (`--compare-ollama` optional) | `glacier bench "models/model.gguf" --tokens 64` |
| `serve` | Starts Ollama + OpenAI compatible HTTP server (`--port`, `--host`, `--kv-precision`) | `glacier serve "models/model.gguf" --port 11434` |
| `video` | Generates APNG / GIF / AVI clips using Wan 2.1 3D-RoPE DiT | `glacier video generate "Prompt" --format all -o out.apng` |
| `image` | Generates images using FLUX.1 DiT (`info` / `demo` / `generate`) | `glacier image generate "Prompt" -w 1024 -h 1024 --steps 4 -o out.png` |
| `vision` | Experimental multimodal patch analysis (`VisionPipeline`) | `glacier vision query scene.png "Describe scene" --synthetic-demo` |
| `voice` | Synthesizes speech (KittenTTS / Kokoro formant TTS) or runs Whisper STT | `glacier voice tts "Hello" --voice af_heart` |

### Backend Flags (`run`, `bench`, `serve`)

| Flag | Values | Default | Notes |
| :--- | :--- | :--- | :--- |
| `--device, -d` | Device slug (`nvidia-rtx-4060`, `amd-890m`, `cpu`) or index | `auto` | Overrides hardware selection |
| `--engine` | `baremetal`, `directml`, `vulkan`, `cpu`, `auto` | `auto` | On Linux, hybrid Qwen 3.5/3.6 models (`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`) use Vulkan; others fall back to CPU |
| `--kv-precision` | `auto`, `fp16`, `fp8`, `fp32` | `auto` | `fp8` halves unmanaged KV cache memory |
| `--ctx, -c` | Integer context window (`2048`, `4096`, `8192`, ...) | `2048` | Caps KV allocation size |
| `--split` | `<device-id>:<layers>,<device-id>:<layers>` or `auto` | None | Splits layers across GPUs |

## HTTP Server (`glacier serve`)

`glacier serve` implements both Ollama (`/api/tags`, `/api/generate`, `/api/chat`) and OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) JSON and streaming endpoints.

> [!WARNING]
> Binding to `0.0.0.0` or `::` exposes inference to the network. Set `GLACIER_API_KEY` before binding publicly; requests must then supply `Authorization: Bearer <key>` or `x-api-key: <key>` (validated with `CryptographicOperations.FixedTimeEquals`).

```powershell
# PowerShell
$env:GLACIER_API_KEY = "<your_api_key>"
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --host 0.0.0.0 --port 8080 --kv-precision fp8
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
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Multi-GPU & Linux Vulkan Execution

Split a 28-layer 14B model evenly between an 8 GB dGPU and an integrated APU:

```bash
glacier run "models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf" --split "nvidia-rtx-4060:14,amd-890m:14"
```

Run hybrid Qwen 3.5/3.6 on Linux Vulkan (Mesa RADV):

```bash
glacier run "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "Hello" --device amd-890m --engine vulkan --ctx 2048
```

## Video & Image CLI

```bash
# Wan 2.1 text-to-video (camera motion presets: pan-left, pan-right, tilt-up, tilt-down, zoom-in, zoom-out, static, fluid)
glacier video generate "Glacier fjords under aurora borealis" --width 832 --height 480 --steps 25 --fps 16 --motion pan-right --format all -o glacier_pan.apng

# FLUX.1 text-to-image
glacier image generate "Crystal mountain peak at twilight" -w 1024 -h 1024 --steps 4 -o mountain.png
```

## Build, Test & Troubleshooting

```bash
dotnet build src/Glacier.Inference.Cli/Glacier.Inference.Cli.csproj -c Release
dotnet test tests/Glacier.Inference.Tests/Glacier.Inference.Tests.csproj -c Release -f net10.0
```

- **Out of Memory (`OutOfMemoryException`):** Reduce `--ctx 2048`, enable `--kv-precision fp8`, or distribute layers with `--split auto`.
- **Windows Driver Reset (`DXGI_ERROR_DEVICE_RESET`):** On display-attached cards, switch to `--engine directml` so dispatches stay below the Windows ~2s TDR timeout.
- **Address Already in Use on `:11434`:** Stop any local Ollama daemon or pass `--port 8080`.
