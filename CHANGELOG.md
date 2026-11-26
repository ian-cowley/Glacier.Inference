# Changelog

All notable changes to Glacier.Inference are recorded here. Versions follow the package version in
`src/Glacier.Inference/Glacier.Inference.csproj`.

## [1.10.0] - 2026-11-26

### Added
- **Full NariQ (`TRL1..TRL6`) support across CPU, CUDA, and Vulkan (`Qwen3.8-27B-NariQ`):**
  - Added `GGML_TYPE_TRL1..TRL6` format metadata, `TrellisKernels` (Sylvester & Paley/Kronecker Hadamard RHT, permutation indices, 16-band mixed-bitrate LM head + `vocab_gather`, CPU row-by-row embedding decode), `Qwen35CudaModel` + `TrellisCudaRunner` (`trellis.cuh` + `qwen35.cuh`), and Vulkan NariQ runner (`TrellisVulkanRunner`).
  - CUDA SASS & fusion optimizations (`kernels.cubin`): zero-`>>2` unsigned `IDP.4A.U8.U8` decode, `PRMT` shift-sharing bit unpacking (`40` regs), 8-warp Split-K (`trellis_gemv_splitk8`), dynamic SMEM + warp-shuffle FWHT, fused `trellis_pre_rht_swiglu` & `q35_add_rmsnorm`, GDN state write elimination, and 3-stream CUDA Graph capture (`19.7 tok/s` FullGraph / `19.2 tok/s` live on RTX 3060 `2.5bpw`; `17.9 tok/s` live on `3.0bpw`).
- **Vulkan 1.3 upgrade (`dotPacked4x8EXT`, subgroup shuffles, Split-K, and discrete `DEVICE_LOCAL` placement):**
  - Negotiates Vulkan 1.3 (with Vulkan 1.1 SWAR fallback), adds `trellis_gemv_dp4a.spv`, `trellis_gemv_splitk_dp4a.spv`, `trellis_pre_rht_128.spv`, and `trellis_post_rht_128.spv` (using `vec2` loads, zero-`>>2` / zero-`I2F` IEEE-754 `dotPacked4x8EXT`, branchless `bitfieldExtract` `TRL2/TRL3/TRL4`, and subgroup-shuffle butterflies), and places static weights in `DEVICE_LOCAL` VRAM across `VulkanRunnerBase`, `Qwen2VulkanModel`, `Gemma4VulkanModel`, `Mamba2VulkanModel`, and `Qwen35VulkanModel` (`8.8 tok/s` RTX 3060 `--engine vulkan`, `2.1 tok/s` live / `2.8 tok/s` est. profile Fedora Radeon 890M on `2.5bpw`).
- **CUDA Graph capture for standard transformers (`Qwen2GpuModel`):**
  - Added 9 device-position (`_dPos`) RoPE/KV-cache/GQA kernels in `ops.cuh` / `attention.cuh` and single/multi-stream graph capture (`GLACIER_CUDA_NO_GRAPH=1` disables), delivering `71.03 tok/s` vs `62.19 tok/s` (`+14.2%`) on RTX 3060 `Qwen3-4B`.
- **Vulkan compute runner for Qwen 3.5 / 3.6 hybrid models** (`Qwen35VulkanModel`): Gated DeltaNet, full attention (QK-norm, partial RoPE,
  KV cache, Q gate) and dense or MoE FFN as Vulkan compute kernels with embedded SPIR-V. It is the GPU path on Linux, works on any
  hardware Vulkan device, and needs no ROCm or CUDA install. Supports `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and `TRL1..TRL6` weights. On a Radeon 890M,
  Qwen3.6-35B-A3B decodes at about 9.4 tok/s (individual runs 9.4 to 15.2), against 2.7 tok/s on the CPU engine, and Qwen3.5-4B at
  about 21.8 tok/s against 1.9. Greedy output is token-identical to the CPU engine on both. See `docs/VULKAN_BACKEND.md`.
- `InferenceSession` selects the Vulkan runner automatically on non-Windows platforms and logs why it was not used when a model is
  unsupported (the check, including weight types, runs before any GPU memory is allocated).
- `Qwen35VulkanModel.ProfileReport()` for per-kernel timing.
- `VulkanKernelTests`: every kernel validated against the CPU or a managed reference (skipped without a hardware Vulkan device);
  `GgufFileTests.Open_FollowsSymlinks_AndUsesTargetFileLength`.
- `docs/VULKAN_BACKEND.md`, ADR-007 and risk R-4 in `ARCHITECTURE.md`, new benchmark sections in `README.md` and `BENCHMARKS.md`,
  `.gitattributes` for shader sources and SPIR-V.

### Fixed
- **Qwen 3.5 / 3.6 garbage output on the CPU engine and the Direct3D 12 Gated DeltaNet shader.** V-heads were mapped to Q/K-heads by
  grouping (`h / ratio`); the model layout (and llama.cpp) tiles them (`h % nK`). It affects every model whose `ssm_time_step_rank`
  exceeds `ssm_group_count`, such as Qwen3.6-27B and Qwen3.6-35B-A3B. The Direct3D 12 fix has not been run on those two models.
- **GGUF files opened through a symlink were rejected on Linux/macOS** ("Header metadata counts exceed available bytes"): the length of
  the link, not of its target, was used for bounds checking. This broke HuggingFace cache layouts (`snapshots/<rev>/x.gguf` pointing
  into `blobs/`).

### Changed
- The "GPU hardware acceleration requires Direct3D 12 on Windows" error for AMD GPUs on Linux now describes the Vulkan runner.

### Known limitations
- The Vulkan runner processes prompt tokens one at a time (no batched prefill) and does not support `IQ*`, `Q2_K`, `Q3_K` or `F16` weight
  matrices (for example, Qwen3.6-27B `UD-Q4_K_XL` contains `IQ4_XS` and runs on the CPU engine).
- On unified-memory APUs, decode speed depends on memory placement (BIOS frame-buffer carve-out versus system-memory GTT); see
  `docs/VULKAN_BACKEND.md` section 5.
