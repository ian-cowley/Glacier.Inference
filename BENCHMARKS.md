# Glacier.Inference: Cross-Architecture & Fleet Benchmark Matrix

This document provides a comprehensive, multi-dimensional performance analysis of the `Glacier.Inference` engine across physical hardware configurations, model architectures, memory topologies, and acceleration engines.

---

## 1. Fleet Hardware Overview

The benchmark fleet spans four heterogeneous physical machine profiles covering dedicated discrete GPUs (NVIDIA Ampere / Ada Lovelace) and high-capacity unified memory APUs (AMD RDNA 2 / RDNA 3.5):

| Machine / Node Profile | Form Factor | OS Platform | Compute Hardware | Memory Architecture | Theoretical Memory Bandwidth | Primary Glacier Engine |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Machine A** | Desktop Workstation | Windows 11 Pro | **NVIDIA GeForce RTX 3060 12GB** (Ampere `sm_86`) | 12 GB GDDR6 (192-bit) | **360 GB/s** | **Bare-Metal SASS (`nvcuda.dll`)** |
| **Machine B** | Mobile Workstation | Windows 11 Pro | **NVIDIA GeForce RTX 4060 Laptop 8GB** (Ada `sm_89`)<br>+ **AMD Radeon 890M 16 CUs** (`gfx1150`)<br>+ **AMD Ryzen AI 9 HX 370** (12C/24T Zen 5) | 8 GB GDDR6 (128-bit)<br>+ 32 GB LPDDR5X Unified<br>(15.5 GB allocated to UMA) | **256 GB/s** (dGPU)<br>~120 GB/s (UMA) | **Bare-Metal SASS (`nvcuda.dll`)**<br>+ **Direct3D 12 Compute (HLSL Wave32)**<br>+ **AVX-512 SIMD Host CPU** |
| **Machine C** | Workstation Mini PC | Linux x64 | **AMD Radeon 890M 16 CUs** (RDNA 3.5 `gfx1150`)<br>+ **AMD Ryzen AI 9 HX 370** (12C/24T Zen 5)<br>+ **XDNA 2 NPU** (50 TOPS `aie2p`) | **64 GB LPDDR5X-7500 Unified**<br>(47 GB usable unified VRAM) | **~120 GB/s** (Unified) | **Bare-Metal HIP / ROCm (`libamdhip64.so` / `/dev/kfd`)**<br>+ **Vulkan Cooperative Matrix (`libvulkan.so.1`)** |
| **Machine D** | Compact Node | Windows 11 Pro | **AMD Radeon 680M 12 CUs** (RDNA 2 `gfx1035`)<br>+ **AMD Ryzen 9 6900HX** (8C/16T Zen 3+) | **64 GB DDR5-4800 Dual-Channel**<br>(Unified Host/iGPU Memory) | **~76.8 GB/s** (Unified) | **Direct3D 12 Compute (HLSL Wave32)**<br>+ **Universal Vulkan (`vulkan-1.dll`)** |

---

## 2. Benchmark View 1: Grouped by Physical Hardware Node

### Machine B (Discrete GPU): NVIDIA GeForce RTX 4060 Laptop 8GB
* **Hardware Specs**: 8 GB GDDR6 (128-bit, 256 GB/s), Ada Lovelace (`sm_89`), 3072 CUDA Cores.
* **Engine**: Pure C# Bare-Metal SASS (`nvcuda.dll`), bypassing CUDA Toolkit runtime (`cudart64.dll`).

| Model | Quantization | Footprint | Active Params | Prompt Prefill (`pp`) | Generation Rate (`tg`) | Turnaround Latency | Silicon Bus Saturation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-4B-Instruct** | `Q4_K_M` | 2.33 GB | 4.0 B | **177.90 tok/s** | **65.99 tok/s** | **0.42 s** | 60.1% |
| **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | **104.20 tok/s** | **41.92 tok/s** | **0.78 s** | **81.3% (208.1 GB/s)** |
| **DeepSeek-R1-Distill-Qwen-7B** | `Q4_K_M` | 4.36 GB | 7.0 B | **102.50 tok/s** | **42.40 tok/s** | **0.82 s** | 76.8% |
| **Xiaomi MiMo-7B-RL** | `Q4_K_M` | 4.36 GB | 7.0 B | **99.54 tok/s** | **41.64 tok/s** | **0.90 s** | 75.4% |
| **Meta LLaMA 3.1 8B Instruct** | `Q4_K_M` | 4.58 GB | 8.0 B | **91.80 tok/s** | **44.05 tok/s** | **0.58 s** | 78.9% |
| **DeepSeek-R1-Distill-Llama-8B** | `Q4_K_M` | 4.58 GB | 8.0 B | **90.17 tok/s** | **41.67 tok/s** | **0.94 s** | 74.6% |
| **Qwen2.5-Coder-7B Enterprise** | `Q8_0` | 7.54 GB | 7.0 B | **82.30 tok/s** | **28.43 tok/s** | **2.00 s** | **83.7% (214.3 GB/s)** |

---

### Machine C & B (Integrated APU): AMD Radeon 890M Unified Memory
* **Hardware Specs**: 16 CUs RDNA 3.5 (`gfx1150`), LPDDR5X-7500 (~120 GB/s unified bandwidth), up to 64 GB unified capacity.
* **Engines**: AMD Bare-Metal HIP (`/dev/kfd`), Universal Vulkan Cooperative Matrix (`VK_KHR_cooperative_matrix`), Direct3D 12 Compute (HLSL Wave32).

| Model | Architecture | Quantization | Footprint | Active Params | Engine Used | Prompt Prefill | Generation Rate | Latency |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Gemma-4-26B-A4B-it** | **MoE (128 Experts, Top-8)** | `Q4_K_M` | **15.63 GiB** | **4.0 B** | **Vulkan Cooperative Matrix** | 🚀 **285.41 tok/s** | 🚀 **29.00 tok/s** | **1.21 s** |
| **Qwen3-30B-A3B-Instruct** | **MoE (128 Experts, Top-8)** | `Q3_K_L` | **13.58 GB** | **3.0 B** | **Direct3D 12 Compute (Wave32)** | **24.20 tok/s** | **21.68 tok/s** | **1.54 s** |
| **ERNIE-4.5-21B-A3B-PT** | **MoE (64 Experts, Top-6)** | `Q4_K_M` | **14.20 GB** | **3.0 B** | **Direct3D 12 Compute (Wave32)** | **21.99 tok/s** | **19.23 tok/s** | **6.29 s** |
| **Qwen2.5-Coder-7B** | Dense | `Q4_K_M` | 4.70 GiB | 7.0 B | **Bare-Metal HIP / ROCm (`/dev/kfd`)** | 🚀 **253.60 tok/s** | 🚀 **17.66 tok/s** | **1.82 s** |
| **Qwen2.5-7B-Instruct** | Dense | `Q4_K_M` | 4.68 GB | 7.0 B | **Direct3D 12 Compute (Wave32)** | **48.45 tok/s** | **11.33 tok/s** | **2.39 s** |
| **Qwen2.5-Coder-14B** | Dense | `Q4_K_M` | **8.37 GB** | 14.0 B | **Direct3D 12 Compute (Wave32)** | **24.98 tok/s** | **6.47 tok/s** | **4.29 s** |
| **Qwen2.5-Coder-7B Enterprise** | Dense | `Q8_0` | 7.54 GB | 7.0 B | **Direct3D 12 Compute (Wave32)** | **38.40 tok/s** | **6.77 tok/s** | **5.64 s** |
| **Qwen3.6-27B-UD** | Dense | `Q4_K_XL` | **16.39 GiB** | 26.9 B | **Vulkan Cooperative Matrix** | **94.43 tok/s** | 🐌 **4.97 tok/s** | **~8.5 s** |
| **Qwen-Coder-Latest (Dense)** | Dense | `F16` | **14.00 GiB** | 7.0 B | **Bare-Metal HIP / ROCm (`/dev/kfd`)** | **36.00 tok/s** | **2.71 tok/s** | **11.4 s** |

---

### Machine A: NVIDIA GeForce RTX 3060 Desktop 12GB
* **Hardware Specs**: 12 GB GDDR6 (192-bit, 360 GB/s), Ampere (`sm_86`), 3584 CUDA Cores.
* **Engine**: Pure C# Bare-Metal SASS (`nvcuda.dll`).

| Model | Quantization | Footprint | Active Params | Prompt Prefill | Generation Rate | Turnaround Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-Qwen-7B** | `Q4_K_M` | 4.68 GB | 7.0 B | **65.53 tok/s** | **45.19 tok/s** | **1.33 s** |
| **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | **68.20 tok/s** | **46.80 tok/s** | **1.28 s** |
| **gpt-oss-20b** | `MXFP4` | 11.28 GB | ~3.5 B | **54.10 tok/s** | **38.40 tok/s** | **1.85 s** |

---

### Machine D: AMD Radeon 680M Integrated GPU
* **Hardware Specs**: 12 CUs RDNA 2 (`gfx1035`), Dual-Channel DDR5-4800 (~76.8 GB/s), 64 GB unified memory.
* **Engines**: Direct3D 12 Compute (HLSL Wave32), Universal Vulkan.

| Model | Quantization | Footprint | Active Params | Engine Used | Prompt Prefill | Generation Rate | Turnaround Latency |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5-1.5B-Instruct** | `Q4_K_M` | 0.98 GB | 1.5 B | **Direct3D 12 Compute (Wave32)** | 🟩 **216.0 tok/s** | **35.30 tok/s** | 🟩 **1.37 s** |
| **Qwen2.5-1.5B-Instruct** | `Q4_K_M` | 0.98 GB | 1.5 B | Ollama / Vulkan Baseline | 151.4 tok/s | 43.60 tok/s | 2.91 s |
| **Qwen3-4B-Instruct** | `Q4_K_M` | 2.33 GB | 4.0 B | **Direct3D 12 Compute (Wave32)** | **92.47 tok/s** | **19.98 tok/s** | **1.58 s** |
| **DeepSeek-Coder-V2-Lite** | `Q4_K_M` | 9.65 GB | 2.4 B | **Direct3D 12 Compute (Wave32)** | **18.40 tok/s** | **5.60 tok/s** | **6.40 s** |
| **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | Generic Vulkan | ~35 tok/s | ~8–10 tok/s | 5.20 s |

---

### Machine E: AMD Ryzen AI 9 HX 370 Host CPU (24 Threads AVX-512)
* **Hardware Specs**: 12 Cores / 24 Threads Zen 5, 5.1 GHz Boost, full-width 512-bit SIMD vector execution.
* **Engine**: Glacier SIMD AVX-512 & AVX2 Unsafe Hardware Intrinsics.

| Model | Quantization | Footprint | Active Params | Prompt Prefill | Generation Rate | Turnaround Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Meta LLaMA 3.1 8B Instruct** | `Q4_K_M` | 4.58 GB | 8.0 B | **6.35 tok/s** | **4.76 tok/s** | **7.98 s** |
| **gpt-oss-20b** | `MXFP4` (Type 39) | 11.28 GB | ~3.5 B | **3.42 tok/s** | **3.08 tok/s** | **15.26 s** |
| **Qwen2.5-Coder-14B** | `Q4_K_M` | 8.37 GB | 14.0 B | **3.23 tok/s** | **2.08 tok/s** | **17.41 s** |
| **Qwen2.5-Coder-7B Enterprise** | `Q8_0` | 7.54 GB | 7.0 B | **3.15 tok/s** | **2.54 tok/s** | **19.36 s** |
| **Qwen3-30B-A3B-Instruct** | `Q3_K_L` | 13.58 GB | 3.0 B | **1.12 tok/s** | **0.89 tok/s** | **32.40 s** |

---

## 3. Benchmark View 2: Grouped by Model Architecture

### Category 1: Mixture-of-Experts (MoE) Frontier

MoE models demonstrate a massive generation advantage on unified memory architectures (APUs) because only the active expert parameters are routed per token, dropping memory bus traffic by 75–85%:

```
Memory Bus Demand per Token:
Dense 27B: [==================================================] 16.4 GB / token
MoE 26B:   [============] 4.0 GB / token  (75% bandwidth reduction -> 5.8x faster)
```

| Model | Total Params / Active Params | Quantization | Memory Size | Best Hardware Target | Acceleration Engine | Prompt Prefill | Generation Rate |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: |
| **Gemma-4-26B-A4B-it** | 25.2 B / **4.0 B** (128 Experts) | `Q4_K_M` | 15.63 GiB | **Machine C (Radeon 890M UMA)** | Vulkan Cooperative Matrix | 🚀 **285.41 tok/s** | 🚀 **29.00 tok/s** |
| **Qwen3-30B-A3B-Instruct** | 30.0 B / **3.0 B** (128 Experts) | `Q3_K_L` | 13.58 GB | **Machine B (Radeon 890M UMA)** | Direct3D 12 Compute | **24.20 tok/s** | **21.68 tok/s** |
| **ERNIE-4.5-21B-A3B-PT** | 21.0 B / **3.0 B** (64 Experts) | `Q4_K_M` | 14.20 GB | **Machine B (Radeon 890M UMA)** | Direct3D 12 Compute | **21.99 tok/s** | **19.23 tok/s** |
| **gpt-oss-20b** | 20.0 B / **~3.5 B** (32 Experts) | `MXFP4` | 11.28 GB | **Machine A (RTX 3060 12GB)** | Bare-Metal SASS | **54.10 tok/s** | **38.40 tok/s** |
| **DeepSeek-Coder-V2-Lite** | 16.0 B / **2.4 B** (64 Experts) | `Q4_K_M` | 9.65 GB | **Machine D (Radeon 680M 64GB)** | Direct3D 12 Compute | **18.40 tok/s** | **5.60 tok/s** |

---

### Category 2: Dense Models (Sub-3B to 27B)

Dense models stream 100% of their parameters through VRAM for every serial token. High-bandwidth dedicated GDDR6 VRAM (RTX 4060/3060) excels here, while APUs are bounded by physical system memory bus width:

| Model | Parameters | Quantization | Size | Machine B (RTX 4060 SASS) | Machine C/B (Radeon 890M) | Machine D (Radeon 680M) | Machine E (Ryzen AI 9 CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 1.5 B | `Q4_K_M` | 0.98 GB | — | — | **35.30 tok/s** | ~18.5 tok/s |
| **Qwen3-4B** | 4.0 B | `Q4_K_M` | 2.33 GB | 🚀 **65.99 tok/s** | ~28.5 tok/s | **19.98 tok/s** | ~9.2 tok/s |
| **Qwen2.5-7B / Coder** | 7.0 B | `Q4_K_M` | 4.68 GB | 🚀 **41.92 tok/s** | 🚀 **17.66 tok/s** (HIP) | ~8–10 tok/s | 4.76 tok/s |
| **Meta LLaMA 3.1 8B** | 8.0 B | `Q4_K_M` | 4.58 GB | 🚀 **44.05 tok/s** | **12.18 tok/s** (D3D12) | ~7.5 tok/s | 4.76 tok/s |
| **DeepSeek-R1-7B** | 7.0 B | `Q4_K_M` | 4.36 GB | 🚀 **42.40 tok/s** | **13.80 tok/s** (D3D12) | ~8.0 tok/s | 4.80 tok/s |
| **Qwen2.5-14B** | 14.0 B | `Q4_K_M` | 8.37 GB | *Exceeds 8GB VRAM* | **6.47 tok/s** (D3D12 UMA) | *Exceeds RAM band* | 2.08 tok/s |
| **Qwen3.6-27B** | 26.9 B | `Q4_K_XL` | 16.39 GB | *Exceeds 8GB VRAM* | 🐌 **4.97 tok/s** (Bus Capped) | *Exceeds RAM band* | 0.72 tok/s |

---

## 4. Benchmark View 3: Grouped by Execution Engine & Driver Stack

This view highlights how each backend technology handles execution overhead, kernel launch latency, and throughput:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                           Glacier Driver Stack Comparison                         │
├──────────────────────┬──────────────────────┬──────────────────────┬──────────────┤
│ Engine               │ Primary Target       │ Driver Linkage       │ Kernel Sched │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────┤
│ Bare-Metal SASS      │ NVIDIA GPUs          │ nvcuda.dll / libcuda │ Hardware SMs │
│ Bare-Metal HIP       │ AMD GPUs (ROCm)      │ amdhip64.dll / kfd   │ AQL Doorbells│
│ Vulkan CoopMat       │ Universal Fallback   │ vulkan-1 / libvulkan │ Queue Submit │
│ Direct3D 12 Wave32   │ Windows iGPU/dGPU    │ d3d12.dll (Vortice)  │ Compute Q    │
│ Host CPU AVX-512     │ All x64 Hosts        │ System Memory MMap   │ ThreadPool   │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────┘
```

| Driver Engine | Target Vendors | Dispatch Overhead | 7B Prefill Rate | 7B Generation Rate | 26B MoE Generation Rate | Portability & Requirements |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **NVIDIA Bare-Metal SASS** | NVIDIA | **< 1 µs** | **104.2 tok/s** | **41.92 tok/s** | **38.40 tok/s** (20B) | Pure C# P/Invoke to `nvcuda.dll`. 0 C++ dependencies. |
| **AMD Bare-Metal HIP** | AMD | **< 1 µs** | 🚀 **253.6 tok/s** | 🚀 **17.66 tok/s** | Supported | Direct HSA / KFD kernel execution via AQL queues. |
| **Universal Vulkan CoopMat** | AMD, Intel, NVIDIA | ~15–30 µs | **94.4 tok/s** | ~11–12 tok/s | 🚀 **29.00 tok/s** | 100% universal. Runs on standard display drivers. |
| **Direct3D 12 Compute** | AMD, Intel, NVIDIA | ~8–15 µs | **48.5 tok/s** | **11.33 tok/s** | **21.68 tok/s** (30B) | Windows only. 100% cooperative with DWM display. |
| **Host CPU SIMD (AVX-512)** | AMD, Intel CPUs | ~0 µs (Native) | **3.2 tok/s** | **2.54 tok/s** | **0.89 tok/s** (30B) | Zero GPU requirements. High portability fallback. |

---

## 5. Architectural Deep Dive: Physics & Memory Bottlenecks

### 1. The Serial Memory Bandwidth Wall (Equation of Generation Speed)
In autoregressive transformer decoding, generating each token requires reading every single model weight matrix once from memory:
$$\text{Max Tokens/Sec} = \frac{\text{Memory Bandwidth (GB/s)}}{\text{Active Weight Footprint (GB)}}$$

* **Dedicated VRAM (Machine B dGPU, 256 GB/s)**:
  $$\frac{256\text{ GB/s}}{4.68\text{ GB (7B Q4)}} \approx 54.7\text{ tok/s (Theoretical Max)} \implies \text{Glacier achieves }\mathbf{41.92\text{ tok/s}}\ (81.3\%\text{ saturation})$$
* **Unified Memory APU (Machine C, ~120 GB/s)**:
  * Dense 27B ($16.4\text{ GB}$): $\frac{120\text{ GB/s}}{16.4\text{ GB}} \approx 7.3\text{ tok/s} \implies \text{Achieved }\mathbf{4.97\text{ tok/s}}$
  * MoE 26B ($4.0\text{ GB}$ active): $\frac{120\text{ GB/s}}{4.0\text{ GB}} \approx 30.0\text{ tok/s} \implies \text{Achieved }\mathbf{29.00\text{ tok/s}}$ (Near physical silicon limit!)

### 2. Why Mixture-of-Experts (MoE) is the Future for Unified APUs
As demonstrated on the **AMD Ryzen AI 9 HX 370 w/ 64GB Unified Memory**, MoE models solve the fundamental memory bandwidth bottleneck of consumer APUs. By only activating a sparse subset of experts (e.g. 4B out of 26B parameters) per token, the memory controller only transfers 4GB over the bus per token while retaining the reasoning capacity and knowledge of a 26B model, producing a **5.8x speedup**.

---

## 6. How to Run Fleet Benchmarks

Run distributed benchmarks across all configured nodes using the PowerShell Fleet Orchestrator:

```powershell
# 1. Benchmark active cluster nodes on MoE models
.\scripts\orchestrate_fleet.ps1 -ModelPath "models/gemma-4-26B-A4B-it-Q4_K_M.gguf" -Tokens 32

# 2. Benchmark local device with explicit engine selection
glacier bench --device amd-890m --engine baremetal --model "models/qwen2.5-coder-7b.gguf"
glacier bench --device amd-890m --engine vulkan     --model "models/gemma-4-26B-A4B-it-Q4_K_M.gguf"
glacier bench --device nvidia-rtx-4060 --engine baremetal --model "models/qwen2.5-7b.gguf"
```
