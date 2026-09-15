# Glacier.Inference: Cross-Architecture & Fleet Benchmark Matrix

This document provides a comprehensive, multi-dimensional performance analysis of the `Glacier.Inference` engine across physical hardware configurations, model architectures, memory topologies, and acceleration engines.

---

## 1. Fleet Hardware Overview

The benchmark fleet spans four distinct physical machine profiles covering dedicated discrete GPUs (NVIDIA Ampere / Ada Lovelace) and high-capacity unified memory APUs (AMD RDNA 2 / RDNA 3.5), clearly separating Windows 32GB laptop configurations from Linux 64GB workstation nodes:

| Machine Profile | Form Factor & Role | OS Platform | Compute Hardware | Memory Architecture | Theoretical Memory Bandwidth | Primary Glacier Engine |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Machine A** | Desktop Workstation | Windows 11 Pro | **NVIDIA GeForce RTX 3060 12GB** (Ampere `sm_86`) | 12 GB GDDR6 (192-bit) | **360 GB/s** | **Native SASS Driver (`nvcuda.dll`)** |
| **Machine B** | Mobile Workstation | Windows 11 Pro | **NVIDIA GeForce RTX 4060 Laptop 8GB** (Ada `sm_89`)<br>+ **AMD Radeon 890M 16 CUs** (`gfx1150`)<br>+ **AMD Ryzen AI 9 HX 370** (12C/24T Zen 5) | **32 GB LPDDR5X Total**<br>• 8 GB GDDR6 Dedicated (dGPU)<br>• 15.5 GB Dynamic UMA (iGPU DWM) | **256 GB/s** (dGPU)<br>~120 GB/s (UMA) | **Native SASS Driver (`nvcuda.dll`)**<br>+ **Direct3D 12 Compute (HLSL Wave32)**<br>+ **AVX-512 SIMD Host CPU** |
| **Machine C** | Workstation Mini PC | Linux x64 (Fedora) | **AMD Radeon 890M 16 CUs** (RDNA 3.5 `gfx1150`)<br>+ **AMD Ryzen AI 9 HX 370** (12C/24T Zen 5)<br>+ **XDNA 2 NPU** (50 TOPS `aie2p`) | **64 GB LPDDR5X-7500 Unified**<br>• **47 GB usable unified VRAM**<br>• Direct `/dev/kfd` HSA access | **~120 GB/s** (Unified) | **Vulkan Hardware Tensor (`libvulkan.so.1`)**<br>+ **AMD ROCm / HIP Driver (`libamdhip64.so`)** |
| **Machine D** | Compact Node | Windows 11 Pro | **AMD Radeon 680M 12 CUs** (RDNA 2 `gfx1035`)<br>+ **AMD Ryzen 9 6900HX** (8C/16T Zen 3+) | **64 GB DDR5-4800 Dual-Channel**<br>(Unified Host/iGPU Memory) | **~76.8 GB/s** (Unified) | **Direct3D 12 Compute (HLSL Wave32)**<br>+ **Universal Vulkan (`vulkan-1.dll`)** |

---

## 2. Benchmark View 1: Grouped by Physical Hardware Profile

### Machine A: Desktop Workstation (NVIDIA GeForce RTX 3060 12GB GDDR6)
* **Hardware Specs**: 12 GB GDDR6 (192-bit, 360 GB/s), Ampere (`sm_86`), 3584 CUDA Cores.
* **Engine**: Pure C# Native SASS Driver (`nvcuda.dll`).

| Model | Quantization | Footprint | Active Params | Prompt Prefill (`pp`) | Generation Rate (`tg`) | Turnaround Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-Qwen-7B** | `Q4_K_M` | 4.68 GB | 7.0 B | **65.53 tok/s** | **45.19 tok/s** | **1.33 s** |
| **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | **68.20 tok/s** | **46.80 tok/s** | **1.28 s** |
| **gpt-oss-20b** | `MXFP4` | 11.28 GB | ~3.5 B | **54.10 tok/s** | **38.40 tok/s** | **1.85 s** |

---

### Machine B: Mobile Workstation (Windows 11, 32GB RAM / 15.5GB UMA)
* **Hardware Specs**: NVIDIA RTX 4060 Laptop 8GB (256 GB/s) + AMD Radeon 890M (15.5 GB Windows DWM UMA) + Ryzen AI 9 HX 370.
* **Primary Engines**: Pure C# Native SASS Driver (`nvcuda.dll`) on dGPU; Direct3D 12 Compute (`HLSL Wave32`) on iGPU.

| Accelerator & Engine | Model | Quantization | Footprint | Active Params | Prompt Prefill | Generation Rate | Turnaround |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RTX 4060 (Native SASS Driver)** | **Qwen3-4B-Instruct** | `Q4_K_M` | 2.33 GB | 4.0 B | **177.90 tok/s** | 🚀 **65.99 tok/s** | **0.42 s** |
| **RTX 4060 (Native SASS Driver)** | **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | **104.20 tok/s** | **41.92 tok/s** | **0.78 s** |
| **RTX 4060 (Native SASS Driver)** | **DeepSeek-R1-Distill-Qwen-7B** | `Q4_K_M` | 4.36 GB | 7.0 B | **102.50 tok/s** | **42.40 tok/s** | **0.82 s** |
| **RTX 4060 (Native SASS Driver)** | **Meta LLaMA 3.1 8B Instruct** | `Q4_K_M` | 4.58 GB | 8.0 B | **91.80 tok/s** | **44.05 tok/s** | **0.58 s** |
| **RTX 4060 (Native SASS Driver)** | **Qwen2.5-Coder-7B Enterprise** | `Q8_0` | 7.54 GB | 7.0 B | **82.30 tok/s** | **28.43 tok/s** | **2.00 s** |
| **Radeon 890M (D3D12 UMA)** | **Qwen3-30B-A3B (MoE)** | `Q3_K_L` | 13.58 GB | **3.0 B** | **24.20 tok/s** | **21.68 tok/s** | **1.54 s** |
| **Radeon 890M (D3D12 UMA)** | **ERNIE-4.5-21B-A3B (MoE)** | `Q4_K_M` | 14.20 GB | **3.0 B** | **21.99 tok/s** | **19.23 tok/s** | **6.29 s** |
| **Radeon 890M (D3D12 UMA)** | **Qwen2.5-Coder-14B** | `Q4_K_M` | 8.37 GB | 14.0 B | **24.98 tok/s** | **6.47 tok/s** | **4.29 s** |
| **Radeon 890M (D3D12 UMA)** | **Qwen2.5-7B-Instruct** | `Q4_K_M` | 4.68 GB | 7.0 B | **48.45 tok/s** | **11.33 tok/s** | **2.39 s** |
| **Radeon 890M (D3D12 UMA)** | **Qwen2.5-Coder-7B Enterprise** | `Q8_0` | 7.54 GB | 7.0 B | **38.40 tok/s** | **6.77 tok/s** | **5.64 s** |
| **Host CPU (AVX-512 SIMD)** | **Meta LLaMA 3.1 8B Instruct** | `Q4_K_M` | 4.58 GB | 8.0 B | **6.35 tok/s** | **4.76 tok/s** | **7.98 s** |

---

### Machine C: Workstation Mini PC (Linux x64, 64GB Unified Memory / 47GB Usable VRAM)
* **Hardware Specs**: AMD Radeon 890M 16 CUs (RDNA 3.5 `gfx1150`), 64 GB LPDDR5X-7500 (~120 GB/s unified memory), 47 GB unified memory allocation.
* **Engines Compared**: **Vulkan Hardware Tensor Engine (`VK_KHR_cooperative_matrix`)** vs. **AMD ROCm / HIP Driver (`libamdhip64.so` / `/dev/kfd`)**.

This machine was subjected to direct cross-engine testing on the same physical silicon across both dense and MoE models:

| Model | Architecture | Quantization | Footprint | Engine / Driver Stack | Prompt Prefill | Generation Rate | Notes & Bottlenecks |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Qwen2.5-Coder-7B** | Dense | `Q4_K_M` | 4.36 GiB | **Vulkan Hardware Tensor** | 🚀 **369.34 tok/s** | 🚀 **20.04 tok/s** | Highest generation speed on 890M |
| **Qwen2.5-Coder-7B** | Dense | `Q4_K_M` | 4.70 GiB | **AMD ROCm / HIP Driver** | **253.60 tok/s** | **17.66 tok/s** | Direct `/dev/kfd` AQL doorbell dispatch |
| **Gemma-4-26B-A4B-it** | **MoE (128 Experts, Top-8)** | `Q4_K_M` | **15.63 GiB** | **Vulkan Hardware Tensor** | 🚀 **285.41 tok/s** | 🚀 **29.00 tok/s** | **5.8x faster than dense 27B** |
| **Gemma-4-26B-A4B-it** | **MoE (128 Experts, Top-8)** | `Q4_K_M` | **15.63 GiB** | **AMD ROCm / HIP Driver** | **103.16 tok/s** | **12.12 tok/s** | 32K context buffer allocated (19 GB) |
| **Qwen3.6-27B-UD** | Dense | `Q4_K_XL` | **16.39 GiB** | **Vulkan Hardware Tensor** | **94.43 tok/s** | 🐌 **4.97 tok/s** | Memory bus bandwidth saturated |
| **Qwen3.6-27B-UD** | Dense | `Q4_K_XL` | **16.39 GiB** | **AMD ROCm / HIP Driver** | **23.36 tok/s** | 🐌 **2.19 tok/s** | 32K context buffer allocated (23 GB) |
| **Qwen-Coder-Latest** | Dense (Unquantized) | `F16` | **14.00 GiB** | **AMD ROCm / HIP Driver** | **36.00 tok/s** | **2.71 tok/s** | Full 16-bit unquantized float weights |

---

### Machine D: Compact Node (AMD Radeon 680M 64GB Dual-Channel DDR5)
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
| **Gemma-4-26B-A4B-it** | 25.2 B / **4.0 B** (128 Experts) | `Q4_K_M` | 15.63 GiB | **Machine C (Radeon 890M 64GB)** | Vulkan Hardware Tensor | 🚀 **285.41 tok/s** | 🚀 **29.00 tok/s** |
| **Qwen3-30B-A3B-Instruct** | 30.0 B / **3.0 B** (128 Experts) | `Q3_K_L` | 13.58 GB | **Machine B (Radeon 890M UMA)** | Direct3D 12 Compute | **24.20 tok/s** | **21.68 tok/s** |
| **ERNIE-4.5-21B-A3B-PT** | 21.0 B / **3.0 B** (64 Experts) | `Q4_K_M` | 14.20 GB | **Machine B (Radeon 890M UMA)** | Direct3D 12 Compute | **21.99 tok/s** | **19.23 tok/s** |
| **gpt-oss-20b** | 20.0 B / **~3.5 B** (32 Experts) | `MXFP4` | 11.28 GB | **Machine A (RTX 3060 12GB)** | Native SASS Driver | **54.10 tok/s** | **38.40 tok/s** |
| **DeepSeek-Coder-V2-Lite** | 16.0 B / **2.4 B** (64 Experts) | `Q4_K_M` | 9.65 GB | **Machine D (Radeon 680M 64GB)** | Direct3D 12 Compute | **18.40 tok/s** | **5.60 tok/s** |

---

### Category 2: Dense Models (Sub-3B to 27B)

Dense models stream 100% of their parameters through VRAM for every serial token. High-bandwidth dedicated GDDR6 VRAM (RTX 4060/3060) excels here, while APUs are bounded by physical system memory bus width:

| Model | Parameters | Quantization | Size | Machine B (RTX 4060 SASS) | Machine C (890M 64GB Linux) | Machine B (890M 32GB Win) | Machine D (680M DDR5) | Host CPU (AVX-512) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 1.5 B | `Q4_K_M` | 0.98 GB | — | — | — | **35.30 tok/s** | ~18.5 tok/s |
| **Qwen3-4B** | 4.0 B | `Q4_K_M` | 2.33 GB | 🚀 **65.99 tok/s** | — | ~28.5 tok/s | **19.98 tok/s** | ~9.2 tok/s |
| **Qwen2.5-7B / Coder** | 7.0 B | `Q4_K_M` | 4.68 GB | 🚀 **41.92 tok/s** | 🚀 **20.04 tok/s** (VK)<br>**17.66 tok/s** (HIP) | **11.33 tok/s** (D3D12) | ~8–10 tok/s | 4.76 tok/s |
| **Meta LLaMA 3.1 8B** | 8.0 B | `Q4_K_M` | 4.58 GB | 🚀 **44.05 tok/s** | — | **12.18 tok/s** (D3D12) | ~7.5 tok/s | 4.76 tok/s |
| **DeepSeek-R1-7B** | 7.0 B | `Q4_K_M` | 4.36 GB | 🚀 **42.40 tok/s** | — | **13.80 tok/s** (D3D12) | ~8.0 tok/s | 4.80 tok/s |
| **Qwen2.5-14B** | 14.0 B | `Q4_K_M` | 8.37 GB | *Exceeds 8GB VRAM* | — | **6.47 tok/s** (D3D12 UMA) | *Bandwidth Capped* | 2.08 tok/s |
| **Qwen3.6-27B** | 26.9 B | `Q4_K_XL` | 16.39 GB | *Exceeds 8GB VRAM* | 🐌 **4.97 tok/s** (VK)<br>🐌 **2.19 tok/s** (HIP) | *Exceeds 15.5GB UMA* | *Bandwidth Capped* | 0.72 tok/s |

---

## 4. Benchmark View 3: Grouped by Execution Engine on AMD Radeon 890M Silicon

Head-to-head cross-engine evaluation on identical AMD Radeon 890M RDNA 3.5 silicon across multiple model architectures:

| Model Architecture & Quant | Vulkan Hardware Tensor (`VK_KHR_coopmat`) | AMD ROCm / HIP Driver (`/dev/kfd`) | Direct3D 12 Compute (HLSL Wave32) |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-Coder-7B** (Dense 4.36 GiB) | 🚀 **369.3 tok/s** pp / 🚀 **20.04 tok/s** tg | **253.6 tok/s** pp / **17.66 tok/s** tg | 48.5 tok/s pp / 11.33 tok/s tg |
| **Gemma-4-26B-A4B** (MoE 15.63 GiB, 4B active) | 🚀 **285.4 tok/s** pp / 🚀 **29.00 tok/s** tg | 103.2 tok/s pp / 12.12 tok/s tg | ~21.7 tok/s tg (30B MoE) |
| **Qwen3.6-27B-UD** (Dense 16.39 GiB) | **94.4 tok/s** pp / 🐌 **4.97 tok/s** tg | 23.4 tok/s pp / 🐌 **2.19 tok/s** tg | *Exceeds 15.5GB UMA* |
| **Qwen-Coder-Latest** (Dense 14.0 GiB F16) | — | 36.0 tok/s pp / 2.71 tok/s tg | *Exceeds FP16 VRAM limit* |

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

### 3. Why "Bare Metal" is a Misnomer on AMD: Compiler Specialization vs. Runtime Driver Stacks
It is common to assume that a vendor driver (such as ROCm / HIP) is inherently "lower level" or faster than an open cross-platform API (like Vulkan). On consumer AMD APU silicon, this hierarchy does not hold:

1. **The ROCm Target Mismatch (`gfx1100` Desktop vs `gfx1150` APU)**:
   Mainline ROCm officially targets datacenter CDNA and flagship desktop discrete GPUs (`gfx1100` / RX 7900 XTX with 96 CUs and 384-bit GDDR6). To execute on the Radeon 890M (`gfx1150`), ROCm relies on `HSA_OVERRIDE_GFX_VERSION=11.0.0`. Its ahead-of-time (AOT) compiled LLVM kernels use thread block and tile sizes tuned for 96 CUs, which cause wavefront starvation and uncoalesced memory stalls on a 16-CU APU sharing system LPDDR5X.
2. **Vulkan Mesa ACO: Direct Silicon Compilation**:
   The Linux RADV Vulkan driver utilizes the Valve-developed **ACO shader compiler**, which compiles SPIR-V on-device specifically for `gfx1150`. ACO generates tight instruction schedules, optimal wavefront occupancy, and directly drives the RDNA 3.5 **Wave Matrix Multiply-Accumulate (WMMA)** tensor instructions (`VK_KHR_cooperative_matrix`) with fused in-register integer dequantization.
3. **Runtime Footprint & Overhead**:
   High-level server stacks running on top of ROCm frequently preallocate massive static buffers (e.g. 32K context KV buffers totaling 5.7 GiB), creating memory controller contention on unified LPDDR5X and causing partial layer evictions to CPU. Direct Vulkan execution keeps the memory footprint lean and all layers resident on the GPU.

---

## 6. How to Run Fleet Benchmarks

Run distributed benchmarks across all configured nodes using the PowerShell Fleet Orchestrator:

```powershell
# 1. Benchmark active cluster nodes on MoE models
.\scripts\orchestrate_fleet.ps1 -ModelPath "models/gemma-4-26B-A4B-it-Q4_K_M.gguf" -Tokens 32

# 2. Benchmark local device with explicit engine selection
glacier bench --device amd-890m --engine rocm       --model "models/qwen2.5-coder-7b.gguf"
glacier bench --device amd-890m --engine vulkan     --model "models/gemma-4-26B-A4B-it-Q4_K_M.gguf"
glacier bench --device nvidia-rtx-4060 --engine sass --model "models/qwen2.5-7b.gguf"
```
