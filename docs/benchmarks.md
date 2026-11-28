# Glacier.Inference Benchmarks

Every figure below comes directly from executed runs on physical hardware or BenchmarkDotNet (`ShortRun`) harnesses. Unexecuted targets and approximated (`~`) baseline rows have been excluded.

## 1. BenchmarkDotNet CPU Microbenchmarks (`net10.0`, `.NET 10.0.15`)

Measured on **AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12 physical cores / 24 logical threads, AVX-512), Windows 11 (`10.0.26300.9974`), .NET SDK `11.0.100`, BenchmarkDotNet `v0.15.8` (`ShortRun`)**:

```bash
dotnet run --project benchmarks/Glacier.Inference.Benchmarks/Glacier.Inference.Benchmarks.csproj -c Release -- --job short --filter *
```

### SIMD Dot Products & RMSNorm (`SimdDotBenchmarks`)

| Method | Dim | Mean | StdDev | Ratio vs F32 | Allocated |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `VecDotF32` | 3584 | 137.6 ns | 0.18 ns | 1.00 | 0 B |
| `VecDotQ8_0` | 3584 | 353.6 ns | 0.38 ns | 2.57 | 0 B |
| `VecDotQ4_0` | 3584 | 406.1 ns | 2.81 ns | 2.95 | 0 B |
| `VecDotQ4_K` | 3584 | 425.2 ns | 3.34 ns | 3.09 | 0 B |
| `VecDotQ6_K` | 3584 | 612.4 ns | 1.99 ns | 4.45 | 0 B |
| `VecDotMXFP4` | 3584 | 1,077.2 ns | 7.23 ns | 7.83 | 0 B |
| `VecDotIQ4_XS` | 3584 | 1,508.9 ns | 17.37 ns | 10.96 | 0 B |
| `VecDotF16` | 3584 | 4,713.3 ns | 58.23 ns | 34.25 | 0 B |
| `RMSNorm` | 3584 | 432.0 ns | 1.79 ns | 3.14 | 0 B |
| `VecDotF32` | 4096 | 155.3 ns | 0.82 ns | 1.00 | 0 B |
| `VecDotQ8_0` | 4096 | 400.4 ns | 1.38 ns | 2.58 | 0 B |
| `VecDotQ4_0` | 4096 | 460.6 ns | 3.54 ns | 2.97 | 0 B |
| `VecDotQ4_K` | 4096 | 485.8 ns | 4.18 ns | 3.13 | 0 B |
| `VecDotQ6_K` | 4096 | 700.0 ns | 1.24 ns | 4.51 | 0 B |
| `VecDotMXFP4` | 4096 | 1,216.7 ns | 1.84 ns | 7.83 | 0 B |
| `VecDotIQ4_XS` | 4096 | 1,736.1 ns | 16.12 ns | 11.18 | 0 B |
| `VecDotF16` | 4096 | 5,339.7 ns | 34.44 ns | 34.39 | 0 B |
| `RMSNorm` | 4096 | 520.5 ns | 2.23 ns | 3.35 | 0 B |

### Multithreaded GEMV & Batched GEMM (`GemvBenchmarks`)

| Method | Dim | Mean | StdDev | Allocated |
| :--- | ---: | ---: | ---: | ---: |
| `MatVecMul_Q8_0` | 3584 | 245.5 us | 2.70 us | 5.32 KB |
| `MatVecMul_Q4_K` | 3584 | 284.3 us | 3.15 us | 5.54 KB |
| `MatMulBatch_Q4_K_Batch8` | 3584 | 1,164.4 us | 11.91 us | 6.59 KB |
| `MatMulBatch_Q4_K_Batch32` | 3584 | 4,498.4 us | 165.77 us | 6.55 KB |
| `MatVecMul_TRL2` | 3584 | 7,088.6 us | 274.38 us | 6.61 KB |
| `MatVecMul_TRL3` | 3584 | 7,024.7 us | 154.92 us | 6.50 KB |
| `MatVecMul_Q8_0` | 4096 | 324.3 us | 5.25 us | 5.70 KB |
| `MatVecMul_Q4_K` | 4096 | 358.5 us | 4.10 us | 5.88 KB |
| `MatMulBatch_Q4_K_Batch8` | 4096 | 1,528.0 us | 12.73 us | 6.41 KB |
| `MatMulBatch_Q4_K_Batch32` | 4096 | 5,781.9 us | 161.49 us | 6.74 KB |
| `MatVecMul_TRL2` | 4096 | 8,897.7 us | 206.16 us | 6.83 KB |
| `MatVecMul_TRL3` | 4096 | 8,540.9 us | 23.23 us | 6.84 KB |

### Token Sampler (`SamplerBenchmarks`)

| Method | VocabSize | Mean | Median | StdDev | Allocated |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Sample_Greedy` | 32,000 | 32.33 us | 32.50 us | 1.756 us | 0 B |
| `Sample_RepetitionPenalty_TopK40` | 32,000 | 62.75 us | 62.55 us | 0.917 us | 88 B |
| `Sample_TopK40_TopP` | 32,000 | 122.17 us | 80.50 us | 76.275 us | 88 B |
| `Sample_TopK100_TopP` | 32,000 | 127.70 us | 113.40 us | 28.663 us | 88 B |
| `Sample_Greedy` | 151,936 | 89.90 us | 89.70 us | 2.007 us | 0 B |
| `Sample_TopK40_TopP` | 151,936 | 135.08 us | 134.65 us | 1.021 us | 88 B |
| `Sample_RepetitionPenalty_TopK40` | 151,936 | 190.90 us | 191.20 us | 7.554 us | 88 B |
| `Sample_TopK100_TopP` | 151,936 | 226.07 us | 247.50 us | 51.417 us | 88 B |

---

## 2. Physical Hardware Runs Across Test Machines

- **Machine A (Windows 11 desktop):** NVIDIA GeForce RTX 3060 12 GB GDDR6 (`360 GB/s`), CUDA `nvcuda.dll` + Vulkan 1.3.
- **Machine B (Windows 11 laptop):** AMD Ryzen AI 9 HX 370 + NVIDIA RTX 4060 Laptop 8 GB (`256 GB/s`) + AMD Radeon 890M (`15.5 GB` DWM UMA), `32 GB` RAM.
- **Machine C (Fedora 43 workstation):** AMD Ryzen AI 9 HX 370 + AMD Radeon 890M (`64 GB` LPDDR5X, `47 GB` usable VRAM, Mesa RADV / HIP).
- **Machine D (Windows 11 mini PC):** AMD Ryzen 9 6900HX + AMD Radeon 680M, `64 GB` DDR5-4800.
- **Machine E (Android 14 phone):** Snapdragon 888+ / Adreno 660 GPU, `8 GB` LPDDR5.

### Discrete NVIDIA GPUs (Machine A & Machine B)

| Machine & Backend | Model & Quant | Size | Prefill | Decode | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RTX 3060 (CUDA)** | `Qwen3-4B-Instruct-2507` (`Q4_K_M`) | 2.33 GB | 114.18 / 122.03 tok/s | **71.03** / 62.19 tok/s | CUDA Graph ON / OFF (`glacier bench`, 128 tok, +14.2%) |
| **RTX 3060 (CUDA)** | `Qwen2.5-7B-Instruct-1M` (`Q4_K_M`) | 4.36 GB | 57.41 / 62.55 tok/s | 39.45 / 44.77 tok/s | CUDA Graph ON / OFF (graph capture amortized after ~46 tok) |
| **RTX 3060 (CUDA)** | `DeepSeek-R1-Distill-Qwen-7B` (`Q4_K_M`) | 4.68 GB | 65.53 tok/s | 45.19 tok/s | Turnaround: 1.33 s |
| **RTX 3060 (CUDA)** | `Qwen3.8-27B-NariQ` (`TRL2/3 2.5bpw`) | 7.86 GB | — | **19.7 tok/s** FullGraph / **19.1–19.2** live | `nari-llama.cpp`: 24.0 tok/s; FullGraph step: 50.78 ms |
| **RTX 3060 (CUDA)** | `Qwen3.8-27B-NariQ` (`TRL2/3/4 3.0bpw`) | 9.43 GB | — | **17.9 tok/s** live (128-tok) | `nari-llama.cpp`: 22.7 tok/s |
| **RTX 3060 (Vulkan 1.3)** | `Qwen3.8-27B-NariQ` (`TRL2/3 2.5bpw`) | 7.86 GB | — | **8.8 tok/s** live (64 & 128 tok) | `staging=True, dp4a=True, sgShfl=True` |
| **RTX 4060 (CUDA)** | `Qwen2.5-7B-Instruct` (`Q4_K_M`) | 4.36 GB | 93.50–106.71 tok/s | **42.90–64.05 tok/s** | 3.34 s 7B cold load; 1,447 ms VRAM upload (`3.01 GB/s`) |
| **RTX 4060 (CUDA)** | `Qwen3-4B-Instruct` (`Q4_K_M`) | 2.33 GB | 156.25 tok/s | **64.08 tok/s** | 0.44 s turnaround |
| **RTX 4060 (CUDA)** | `Meta-Llama-3.1-8B-Instruct` (`Q4_K_M`) | 4.58 GB | 91.80 tok/s | **44.05 tok/s** | 0.58 s turnaround |
| **RTX 4060 (CUDA)** | `DeepSeek-R1-Distill-Qwen-7B` (`Q4_K_M`) | 4.68 GB | — | **41.92–42.40 tok/s** | Ollama on same RTX 4060: 43.20 tok/s gen (50-tok wall: 1.41s vs 3.64s) |
| **RTX 4060 (D3D12)** | `Qwen3.5-9B` (`Q4_K_M`, GDN SSM) | 5.63 GB | 78.40 tok/s | **28.80 tok/s** | 0.73 s turnaround |

### Integrated AMD APUs & Host CPU (Machine B, C, D)

| Machine & Backend | Model & Quant | Size | Prefill | Decode | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Machine B (890M D3D12 UMA)** | `Qwen3-4B-Instruct` (`Q4_K_M`) | 2.33 GB | 94.20 tok/s | **19.67 tok/s** | 1.13 s turnaround |
| **Machine B (890M D3D12 UMA)** | `google_gemma-4-E4B-it` (`Q4_K_S`) | 4.54 GB | 42.10 tok/s | **19.60 tok/s** | 1.27 s turnaround |
| **Machine B (890M D3D12 UMA)** | `Qwen3-30B-A3B` (`Q3_K_L`, MoE) | 13.58 GB | 22.14 tok/s | **19.12 tok/s** | 8/128 experts active; 34.7 s cold load |
| **Machine B (890M D3D12 UMA)** | `ERNIE-4.5-21B-A3B` (`Q4_K_M`, MoE) | 14.20 GB | 21.33 tok/s | **18.81 tok/s** | 6/64 experts active; 38.8 s cold load |
| **Machine B (890M D3D12 UMA)** | `Qwen3.5-9B` (`Q4_K_M`, GDN SSM) | 5.63 GB | 36.20 tok/s | **11.50–12.00 tok/s** | Bit-exact parity with CUDA across 32 layers |
| **Machine B (890M D3D12 UMA)** | `Qwen2.5-7B` (`Q4_K_M`) / `14B` (`Q4_K_M`) | 4.68 / 8.37 GB | 48.45 / 24.98 tok/s | 11.33 / 6.47 tok/s | 4.46 s upload on 7B |
| **Machine C (890M Linux Vulkan)** | `Qwen2.5-Coder-7B` (`Q4_K_M`) | 4.36 GiB | 369.34 tok/s | **20.04 tok/s** | HIP on same host: 253.60 pp / 17.66 tg |
| **Machine C (890M Linux Vulkan)** | `Gemma-4-26B-A4B-it` (`Q4_K_M`, MoE) | 15.63 GiB | 285.41 tok/s | **29.00 tok/s** | HIP on same host: 103.16 pp / 12.12 tg |
| **Machine C (890M Linux Vulkan)** | `Qwen3.6-35B-A3B-UD` (`Q4_K_M`) | 20.6 GiB | — | **9.4 tok/s** (9.2–9.6) | 24T CPU: 2.7 tok/s; `llama.cpp` Vulkan: 10.4 tok/s; see [VULKAN_BACKEND.md](VULKAN_BACKEND.md) |
| **Machine C (890M Linux Vulkan)** | `Qwen3.8-27B-NariQ` (`2.5bpw`) | 7.86 GB | — | **2.1 tok/s** live | `dp4a=True, sgShfl=True, staging=False` |
| **Machine B (Ryzen AI 9 CPU)** | `Meta-Llama-3.1-8B` / `Qwen3-4B` / `gpt-oss-20b` / `Gemma-4-26B` | — | 6.35 / 6.01 / 3.36 / 1.85 tok/s | 4.76 / 4.38 / 4.08 / 0.80 tok/s | AVX-512 host CPU |
| **Machine D (680M D3D12)** | `Qwen2.5-1.5B` / `Qwen3-4B` (`Q4_K_M`) | 0.98 / 2.33 GB | 216.0 / 92.47 tok/s | **35.30** / **19.98 tok/s** | Ollama Vulkan on 1.5B: 151.4 pp / 43.60 tg |

### Isolated NariQ FFN Microbenchmarks (`5120 x 17408`, `NariQGpuParityTests` on Machine B)

| Accelerator & Backend | `TRL2` | `TRL3` |
| :--- | :---: | :---: |
| NVIDIA RTX 4060 Laptop (CUDA) | 169.2 us (`125.6 GB/s`) | 240.6 us (`132.5 GB/s`) |
| NVIDIA RTX 4060 Laptop (Vulkan 1.3) | 1,011.8 us (`21.0 GB/s`) | 1,020.4 us (`31.2 GB/s`) |
| AMD Radeon 890M Laptop (Vulkan 1.3) | 783.5 us (`27.1 GB/s`) / Split-K `17408x5120` 689.3 us | 857.9 us (`37.2 GB/s`) |

## 3. Multimodal & Generative Media Runs

### FLUX.1 Schnell (`flux1-schnell-Q4_K_S.gguf` + `ae.safetensors`, RTX 4060)

| Mode & Resolution | DiT 4-Step Trajectory | VAE Decode | Total Runtime |
| :--- | ---: | ---: | ---: |
| `512x512` (Initial CPU VAE baseline) | 94.6 s | 121.2 s | 227.8 s |
| `512x512` (In-VRAM GPU Resident) | 69.8 s | **4.8 s** | **111.1 s** |
| `1024x1024` (Monolithic VAE) | 412.3 s | 566.8 s | 1,014.3 s |
| `1024x1024` (`2x2` Tiled `544x544` VAE) | 412.3 s | **22.6 s** | **435.0 s** |

### EmbeddingGemma 2 on Mobile Edge (Machine E: Snapdragon 888+ / Adreno 660)

| Workload | Kryo 680 CPU (AdvSIMD) | Adreno 660 GPU (Vulkan) | Cosine Similarity vs CPU |
| :--- | ---: | ---: | :---: |
| Cold Startup (mmap + SPIR-V compile) | 2,153 ms | 4,050 ms | — |
| Text Embedding (11 tokens) | 1,379 ms | 1,364 ms | `1.000000` |
| Audio Embedding (1.0 s @ 16 kHz) | 8,017 ms | 8,198 ms | `1.000000` |
| Vision Embedding (`224x224`) | 317.2 s | 314.2 s | `1.000000` |
| Video Embedding (2 frames `224x224`) | 277.1 s | 268.6 s | `1.000000` |

### Voice Synthesis (`KokoroTtsEngine` / Formant Acoustics, Ryzen AI 9 HX 370)

| Profile | Audio Duration | Wall Time | Real-Time Factor | Estimated F0 | Clipping |
| :--- | ---: | ---: | ---: | ---: | :---: |
| `af_heart` (US Female) | 5.38 s | 32 ms | 163.5x | 203.4 Hz | 0 samples |
| `am_adam` (US Male) | 6.23 s | 35 ms | 175.1x | 105.3 Hz | 0 samples |
| `bf_emma` (UK Female) | 5.40 s | 31 ms | 172.1x | 210.5 Hz | 0 samples |
| `bm_george` (UK Male) | 6.43 s | 34 ms | 185.9x | 110.6 Hz | 0 samples |
