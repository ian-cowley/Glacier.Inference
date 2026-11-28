# Memory Tiers, Pipelines, NPU Sidecar & Android

This topic manual covers advanced memory management, programmatic multi-GPU pipelines, AMD/Intel NPU sidecar probing, diagnostics logging, settings persistence, and the Android host project in `Glacier.Inference`.

## 1. Large-Model Memory & Tiered MoE (`TieredVirtualMemoryManager`)

Control host RAM vs. VRAM placement directly on `InferenceSession` (or via `glacier-ai run` flags `--ple-io`, `--host-arena-gib`, `--tiered-moe`):

- **`pleIoMode` (`PleIoMode.Auto` / `Direct` / `MemoryMapped`):** Controls how large per-layer embedding tables (`per_layer_token_embd.weight`) are mapped vs. counted against host RAM arena budgets.
- **`hostArenaBudgetBytes` (`ulong?`):** Caps host RAM cached expert storage.
- **`tieredMoe` (`bool?`):** Enables GPU-resident hot expert slots backed by host-RAM LRU arenas (`TieredVirtualMemoryManager`, `TieredMemoryOptions.FromHostBudgetGiB`).
- **`vramReserveMb` (`int?`):** Leaves headroom for Windows DWM or concurrent GPU workloads (`ApplyVramHeadroomGuard`, `ResizeElasticVramSlots`, `WarmFromProfile`, `RunAdaptivePass`).

## 2. Bandwidth Co-Scheduling & Runtime Profiling

- **`HardwareRuntimeProfiler`:** Captures physical CPU/GPU topology (`CaptureSnapshot`), estimates total weight + PLE memory (`EstimateTotalModelAndPleBytes`), and computes default policies (`RecommendPolicy`, `CreateCalibratedCoScheduler`).
- **`BandwidthCoScheduler`:** Tracks measured PCIe transfer, CPU execution, and GPU execution speeds (`RecordPcieTransfer`, `RecordCpuExecution`, `RecordGpuExecution`) to decide how many missing MoE experts belong on GPU vs. CPU (`CalculateOptimalGpuExpertCount`, `Schedule`).

## 3. Programmatic Multi-GPU Pipelines (`PipelineSession`)

For custom stage splitting in C# code (backing CLI `--split`):

```csharp
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Pipeline;

using var gguf = GgufFile.Open("models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf");
var weights = new ModelWeights(gguf);
List<PipelineStageSpec>? stages = PipelineSplitConfig.ResolveStages("nvidia-rtx-4060:14,amd-890m:14", weights);
if (stages != null)
{
    using var pipeline = PipelineSession.Create(weights, stages, maxSeqLen: 4096);
    // Forward / ForwardBatch / SampleToken / ResetKvCache
}
```

## 4. NPU Sidecar Discovery (`NpuSidecarDiscovery`)

Detects local NPU inference sidecar endpoints (`NpuSidecarRuntime`, e.g., Ryzen AI FastFlowLM / Lemonade endpoints) with concurrency-safe cached probing:

- `await NpuSidecarDiscovery.ProbeCachedAsync(...)`
- `await NpuSidecarDiscovery.ProbeAsync(...)`
- `await NpuSidecarDiscovery.ProbeSingleAsync(httpClient, baseUrl, runtimeHint)`
- `NpuSidecarDiscovery.InvalidateCache()`

## 5. Logging & Persistent Settings

- **`GlacierDiagnostics`:** Global logger facade (`LogLevel`: Trace through Critical) supporting `NullGlacierLogger`, `ConsoleGlacierLogger`, and `DelegateGlacierLogger`, plus `GlacierDiagnostics.Reset()`.
- **`GlacierSettings`:** Stores persistent default device/engine configuration in `~/.glacier/settings.json` (`Load`, `Save`, `ValidateSafety`, source-generated `GlacierSettingsJsonContext`).

## 6. Android Build (`Glacier.Inference.Android`)

The solution includes an Android UI host (`src/Glacier.Inference.Android/Glacier.Inference.Android.csproj`) targeting `net10.0-android` with `MainActivity` executing quantized inference over Android Vulkan 1.1 and ARM64 NEON SIMD kernels.
