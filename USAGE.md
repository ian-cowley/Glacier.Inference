# Glacier.Inference — Operational & User Guide

**Document Type:** Diátaxis How-To Guide & Technical Reference  
**Language Standard:** ASD-STE100 Compliant  
**Target Platform:** .NET 10 (C# 13) / Native AOT  
**Applies to:** Standalone CLI, Server Microservice, C# Library  

---

## 1. System Requirements & Installation

### 1.1 Hardware & Operating System Requirements
Glacier.Inference runs on Windows, Linux, and Android systems:

| Hardware Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit), Linux (glibc $\ge 2.31$), Android 12+ | Windows 11 (build 22621+) or Ubuntu 24.04 LTS |
| **Processor (CPU)** | x86-64 with AVX2 or ARM64-v8a | x86-64 with AVX-512 and FMA (e.g. AMD Ryzen 9 / Intel Core i9) |
| **Graphics Card (GPU)** | Direct3D 12 Feature Level 12.0 or Vulkan 1.1 compatible | NVIDIA RTX 30/40 Series (Ada/Ampere) or AMD RDNA 2/3.5 APU |
| **System Memory (RAM)** | 8 GB DDR4 | 32 GB DDR5 or LPDDR5X (Unified Memory) |
| **GPU Memory (VRAM)** | 4 GB dedicated VRAM (for 3B models) | 8 GB to 16 GB dedicated or unified VRAM |

---

### 1.2 Installation Options

#### Option A: Download Standalone Executable (Zero Dependencies)
Download the standalone executable from GitHub Releases. Extract the archive and add the directory to your system path:
* **Windows (x64):** `glacier-v1.3.2-win-x64.zip`
* **Linux (x64):** `glacier-v1.3.2-linux-x64.tar.gz`
* **Windows (ARM64):** `glacier-v1.3.2-win-arm64.zip`
* **macOS (Apple Silicon ARM64):** `glacier-v1.3.2-osx-arm64.tar.gz`

Test the installation:
```bash
glacier devices
```

#### Option B: Install as a .NET Global Tool
If you have the .NET 10 SDK installed, run:
```bash
dotnet tool install --global Glacier.Inference.Cli
```
Verify the global tool:
```bash
glacier --help
```

#### Option C: Add NuGet Package to a C# Application
Run this command inside your .NET project folder:
```bash
dotnet add package Glacier.Inference
```

---

## 2. Command Line Interface (CLI) Complete Reference

```text
Usage: glacier <command> [options]
```

### 2.1 Hardware Discovery: `glacier devices`
Discovers all physical compute devices, dedicated memory, unified memory, and driver engines:

```bash
glacier devices
```

**Sample Output:**
```text
============================================================================================================
                                  GLACIER DETECTED ACCELERATORS & ENGINES                                   
============================================================================================================
[ID]               | Device Name                        | Memory          | Safe Driver Engines      
------------------------------------------------------------------------------------------------------------
nvidia-rtx-4060    | NVIDIA GeForce RTX 4060 Laptop GPU | 8.0 GB VRAM     | SASS, Vulkan, Direct3D 12 [ACTIVE: SASS]
  └─ Driver Safety: Direct SASS driver interop (nvcuda.dll); bypasses CUDA Toolkit runtime.
amd-890m           | AMD Radeon 890M Graphics           | 15.5 GB Unified | Direct3D 12, Vulkan, HIP 
  └─ Driver Safety: Display adapter with DWM sync; safe for concurrent UI desktop rendering.
cpu                | AMD Ryzen AI 9 HX 370 24-Thread    | 31.2 GB RAM     | CPU SIMD (AVX-512)       
  └─ Driver Safety: Host CPU execution; zero thermal hazard.
------------------------------------------------------------------------------------------------------------
```

---

### 2.2 System Configuration: `glacier config`
Manages persistent user preferences stored in `~/.glacier/settings.json`.

```bash
# View current persistent configuration
glacier config

# Set default hardware device and compute engine
glacier config --device nvidia-rtx-4060 --engine baremetal

# Configure AMD integrated graphics with Direct3D 12
glacier config --device amd-890m --engine directml

# Reset all settings to automatic detection
glacier config --reset
```

---

### 2.3 Model Inspection: `glacier inspect`
Reads GGUF and Safetensors headers with zero heap allocations. It prints tensor counts, parameter sizes, and quantization formats:

```bash
glacier inspect "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

**Key Information Reported:**
* Architecture family (`llama`, `deepseek2`, `qwen2`, `wan`, `flux`)
* Sequence context length limit (`context_length`)
* Embedding dimensions, attention head counts, and layer count
* Complete tensor registry with memory alignment and quantization types

---

### 2.4 Interactive Chat & Prompt Execution: `glacier run`
Starts an interactive streaming chat session or processes a single prompt.

```bash
# Interactive REPL session
glacier run "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Single-shot prompt execution
glacier run "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Explain the Byzantine Generals Problem in two paragraphs."

# Execution with custom sampling parameters
glacier run "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" "Write a C# vector dot product method." --temp 0.2 --top-p 0.95 --ctx 4096
```

**Available Options:**
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--temp <float>` | Randomness temperature (e.g. `0.2` for precise, `0.8` for creative) | `0.7` |
| `--top-p <float>` | Nucleus sampling probability mass cutoff | `0.9` |
| `--top-k <int>` | Top-$K$ candidate tokens considered per step | `40` |
| `--repeat-penalty <float>`| Penalty factor for repetitive token generation | `1.1` |
| `--ctx, -c <int>` | Maximum context sequence length | `2048` |
| `--device <name>` | Target compute hardware accelerator | Auto |
| `--engine <name>` | Low-level execution HAL (`baremetal`, `directml`, `cpu`) | Auto |

---

### 2.5 Throughput Benchmarking: `glacier bench`
Measures prefill latency, generation throughput, and memory bandwidth on physical hardware.

```bash
# Run 64-token speed benchmark
glacier bench "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --tokens 64

# Automated side-by-side comparison against a local Ollama daemon
glacier bench "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" --tokens 64 --compare-ollama "http://127.0.0.1:11434" --compare-model "qwen2.5:7b"
```

---

## 3. High-Throughput HTTP Microservice: `glacier serve`

The `glacier serve` command starts an enterprise-grade inference microservice with drop-in compatibility for Ollama and OpenAI clients.

```bash
# Start server on default port (11434)
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Bind to custom port and set unmanaged KV Cache precision
glacier serve "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" --port 8080 --host 0.0.0.0 --kv-precision fp8 --ctx 4096
```

### 3.1 Network Security & Authentication
> **WARNING: Unauthenticated Public Exposure**  
> If you bind to `0.0.0.0` or `::` without setting an API key, any network client can execute inference. Protect public endpoints by setting the `GLACIER_API_KEY` environment variable.

```bash
# Windows PowerShell
$env:GLACIER_API_KEY = "<your_api_key>"
glacier serve "models/model.gguf" --host 0.0.0.0 --port 8080

# Linux / macOS
export GLACIER_API_KEY="<your_api_key>"
glacier serve "models/model.gguf" --host 0.0.0.0 --port 8080
```

Clients must supply the key via the `Authorization: Bearer <key>` header or `x-api-key: <key>` header. The server performs constant-time validation (`CryptographicOperations.FixedTimeEquals`) to eliminate timing attacks.

---

### 3.2 Supported Server Endpoints

#### 1. Ollama Compatible API
* **`GET /api/tags`**: Returns model inventory, parameter size, and quantization type.
* **`POST /api/generate`**: Executes text generation with optional newline-delimited JSON (`application/x-ndjson`) streaming.
* **`POST /api/chat`**: Processes conversational message histories.

**Example `curl` Request (Ollama format):**
```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "Meta-Llama-3.1-8B-Instruct",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "temperature": 0.3
  }
}'
```

#### 2. OpenAI Compatible API
* **`GET /v1/models`**: Lists available models in standard OpenAI JSON format.
* **`POST /v1/chat/completions`**: Executes chat completion with standard Server-Sent Events (`text/event-stream`) token streaming.
* **`POST /v1/completions`**: Executes standard single-prompt completions.

**Example `curl` Request (OpenAI format):**
```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_api_key>" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a concise engineering assistant."},
      {"role": "user", "content": "What is an AVX-512 vector register?"}
    ],
    "temperature": 0.5,
    "stream": true
  }'
```

---

### 3.3 Python SDK & Framework Integration

#### Python OpenAI SDK Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="<your_api_key>"  # Matches GLACIER_API_KEY
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Explain zero-copy memory mapping."}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

#### LangChain Integration
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="<your_api_key>",
    model="default"
)

response = llm.invoke("Summarize Amdahl's Law.")
print(response.content)
```

---

## 4. Model Loading & Storage Formats

### 4.1 Supported Quantization Super-Blocks
Glacier.Inference supports all official GGUF quantization types without third-party converters:

| Format Name | Bits Per Weight (bpw) | Block Size | Alignment | Primary Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **`BlockQ8_0`** | 8.50 | 32 | 34 Bytes | Reference high-precision weights; enterprise code generation. |
| **`BlockQ4_K`** | 4.50 | 256 | 144 Bytes | Recommended industry standard; optimal speed and perplexity balance. |
| **`BlockQ2_0`** | 2.50 | 64 | 20 Bytes | 4-byte aligned; maximum throughput on Direct3D 12 coalesced loads. |
| **`BlockIQ3_S`** | 3.44 | 256 | 110 Bytes | Non-linear codebook format; runs large 14B models in 8GB VRAM. |
| **`BlockTernary256`** | 2.125 | 256 | 68 Bytes | 1.58-bit BitNet models ($W \in \{-1, 0, +1\}$); multiplication-free compute. |

### 4.2 Calculating VRAM Sizing
Calculate required GPU VRAM before loading:
$$\text{VRAM}_{\text{total}} = \text{FileSize}_{\text{weights}} + \left( 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C \times B_{\text{prec}} \right)$$

Where:
* $L$ = Number of Transformer layers
* $H_{\text{kv}}$ = Number of Key-Value heads
* $D_{\text{head}}$ = Head dimension (e.g. 128)
* $C$ = Maximum context length (e.g. 2048)
* $B_{\text{prec}}$ = Precision bytes per element ($2.0\text{ bytes}$ for FP16, $1.0\text{ byte}$ for FP8)

---

## 5. Multi-GPU & Heterogeneous Pipeline Configuration

When a model exceeds the memory of a single discrete graphics card, configure heterogeneous layer splitting using `--split`.

### 5.1 Split Syntax Specification
The `--split` argument accepts comma-separated device layer assignments:
```text
--split "<device-id-1>:<layer-count-1>,<device-id-2>:<layer-count-2>"
```

### 5.2 Common Multi-GPU Patterns

#### Pattern A: Discrete NVIDIA GPU + Integrated AMD APU
On laptops equipped with an NVIDIA RTX 4060 (8GB) and an AMD Ryzen AI 9 APU (16GB Unified RAM):
```bash
# Execute 28-layer Qwen2.5-Coder-14B: 14 layers on NVIDIA dGPU + 14 layers on AMD APU
glacier run "models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf" --split "nvidia-rtx-4060:14,amd-890m:14"
```
The discrete GPU handles initial layer GEMMs. Pinned memory transfers hidden states to the APU for the remaining layers with zero host PCIe stalls.

#### Pattern B: Automatic Device Partitioning
```bash
glacier serve "models/Large-30B-Model-Q4_K_M.gguf" --split auto
```
The engine inspects available VRAM across all detected accelerators and partitions layers proportionally.

---

## 6. Generative Video & Image Production

Glacier.Inference provides pure C# text-to-video and text-to-image pipelines with zero Python dependencies.

### 6.1 Generative Video Synthesis: `glacier video`
The video subsystem implements the Alibaba Wan 2.1 architecture using 3D Rotary Position Embeddings (3D-RoPE), 16-channel 3D Causal Autoencoding, and Flow-Matching Rectified Flow ODE integration.

```bash
# Generate high-resolution 15-second cinematic video (240 frames @ 16 FPS)
glacier video generate "A stylish woman in a dark coat walking confidently along a bustling modern city street at night, glowing neon signs, wet pavement reflections" --width 832 --height 480 --steps 25 --duration 15 --fps 16 --guidance 5.0 --format all --out woman_walk.apng

# Steer video trajectory with camera motion controls
glacier video generate "Icy glacier fjords under shimmering aurora borealis" --motion pan-right --format all --out glacier_pan.apng

# Animate an existing still image
glacier video generate "Cinematic camera zoom into the scene" --image "input_photo.png" --motion zoom-in --format all --out animated_scene.apng
```

**Video CLI Options:**
| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--width, -w` | `832` | Video frame width in pixels (must be multiple of 16). |
| `--height, -h` | `480` | Video frame height in pixels (must be multiple of 16). |
| `--frames` | `17` | Number of output frames (overridden when `--duration` is set). |
| `--duration` | `1.06` | Optional video length in seconds (computes `frames = round(duration * fps)`). |
| `--fps` | `16` | Frame rate in frames per second. |
| `--steps` | `25` | Number of Flow-Matching Euler ODE numerical solver steps. |
| `--guidance` | `5.0` | Classifier-Free Guidance (CFG) scale factor. |
| `--motion` | `pan-right` | Camera motion trajectory (`pan-left`, `pan-right`, `tilt-up`, `tilt-down`, `zoom-in`, `zoom-out`, `static`, `fluid`). |
| `--format` | `apng` | Output video container (`apng`, `gif`, `avi`, or `all`). |
| `--out, -o` | `glacier_video.apng` | Output file path. |

---

### 6.2 Text-to-Image Generation: `glacier image`
The image generation subsystem executes Black Forest Labs FLUX.1 Schnell 57-block Diffusion Transformers (DiTs) and Neural VAE decoders.

```bash
# Generate high-resolution photorealistic image
glacier image generate "A cybernetic dragon perched on a crystal mountain peak at twilight" -w 1024 -h 1024 --steps 4 -o dragon.png

# Inspect a Diffusion GGUF model file
glacier image info "models/flux1-schnell-Q4_K_S.gguf"

# Run the self-contained image generation demo
glacier image demo --out "crystal_fortress.png"
```

---

## 7. Production Deployment & Troubleshooting

### 7.1 Running as a Linux Systemd Service
Create the service unit file at `/etc/systemd/system/glacier-inference.service`:

```ini
[Unit]
Description=Glacier Inference High-Performance HTTP Service
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

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now glacier-inference
sudo systemctl status glacier-inference
```

---

### 7.2 Diagnostic Procedures & Common Solutions

#### 1. Out of Memory (OOM) Errors
* **Symptom:** Process terminates with `OutOfMemoryException` during model initialization.
* **Remedy 1:** Enable FP8 KV Cache compression: `--kv-precision fp8`. This reduces cache memory consumption by 50%.
* **Remedy 2:** Lower context length: `--ctx 2048`.
* **Remedy 3:** Split layers across multiple devices: `--split auto`.

#### 2. Driver TDR Timeouts on Windows
* **Symptom:** Display flickers and generation aborts with `DXGI_ERROR_DEVICE_RESET`.
* **Remedy:** The Windows display driver resets operations exceeding 2 seconds. Use Direct3D 12 on display cards (`--engine directml`) and reserve bare-metal SASS (`--engine baremetal`) for non-display GPUs.

#### 3. Port Conflicts
* **Symptom:** `Failed to bind to address http://127.0.0.1:11434: address already in use.`
* **Remedy:** Stop any running Ollama instance (`sudo systemctl stop ollama` or terminate `ollama.exe`), or specify an alternative port: `--port 8080`.
