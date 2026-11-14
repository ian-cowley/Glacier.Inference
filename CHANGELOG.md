# Changelog

All notable changes to Glacier.Inference are recorded here. Versions follow the package version in
`src/Glacier.Inference/Glacier.Inference.csproj`.

## Unreleased

### Added
- **Vulkan compute runner for Qwen 3.5 / 3.6 hybrid models** (`Qwen35VulkanModel`): Gated DeltaNet, full attention (QK-norm, partial RoPE,
  KV cache, Q gate) and dense or MoE FFN as Vulkan 1.1 compute kernels with embedded SPIR-V. It is the GPU path on Linux, works on any
  hardware Vulkan device, and needs no ROCm or CUDA install. Supports `Q4_K`, `Q5_K`, `Q6_K` and `Q8_0` weights. On a Radeon 890M,
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
