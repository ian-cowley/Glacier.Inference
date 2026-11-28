# Continuum Memory, KL Evaluation, Bounce Proxy & Low-Level Formats

This topic manual covers HOPE/Titans associative memory, quantization quality evaluation (`eval-kl`), the compiler-verified bounce proxy (`bounce`), and the binary format, tokenizer, and sampling primitives.

## 1. HOPE & Titans Continuum Memory (`glacier serve --sleep-memory`)

`Glacier.Inference.Continuum` implements multi-timescale associative memory blocks (`TitansContinuumMemoryBlock` and `HopeContinuumHierarchy`):

- **`glacier serve` flags:** Pass `--sleep-memory`, `--sleep-budget <tokens>` (default `400`), and `--sleep-checkpoint <path>` to consolidate conversation traces during offline cycles.
- **Sleep HTTP Endpoints:**
  - `GET /api/sleep/status` & `GET /v1/sleep/status` — returns consolidation telemetry.
  - `POST /api/sleep/trigger` & `POST /v1/sleep/trigger` — runs an immediate sleep consolidation pass.
- **C# API:**
  - `HopeContinuumHierarchy.CreateDefault4Tier(int dimension, float surpriseThreshold = 0.65f)` manages four clock-strided `HopeTierConfig` tiers (`Forward`, `Step`, `PruneVolatileTiers`, `ResetAll`, `GetTier`).
  - `TitansContinuumMemoryBlock` exposes coupled momentum matrix updates (`UpdateAssociativeState`), `BackwardStep`, `ResetAdjointState`, `ResetVolatileState`, and raw span access (`GetMemoryMatrixSpan`, `GetMomentumMatrixSpan`, `SaveState`).

## 2. Quantization Quality Verification (`glacier eval-kl` & `KlDivergenceVerifier`)

Compare logits between a reference precision model and a quantized candidate via Kullback-Leibler divergence and top-1 token agreement:

```bash
glacier eval-kl --help
```

In C#:

- `KlDivergenceVerifier.ComputeTokenKlDivergence(referenceLogits, quantizedLogits)`
- `KlDivergenceVerifier.EvaluateSingleTokenLogits(...)` / `EvaluateLogits(..., KlDivergenceOptions)` -> returns `KlDivergenceResult` (mean/max KL divergence, top-1 agreement, pass/fail thresholds).

## 3. Peer-Critic & Compiler Bounce Proxy (`glacier bounce`)

Starts a local proxy (`http://127.0.0.1:18090` by default) in front of a primary OpenAI-compatible server that repairs generated code using an optional peer critic and compiler syntax check:

```bash
glacier bounce --primary http://127.0.0.1:18080 --critic http://192.168.1.109:18080 --max-repairs 2 --port 18090
```

Options: `--primary`, `--critic` / `--peer-critic`, `--host`, `--port`, `--harness`, `--max-repairs <0-3>`, `--compiler-bounce` / `--no-compiler-bounce`, `--model-alias` (`Glacier-3Machine-Bounce`).

## 4. Low-Level File Loaders, Tokenizers & Sampler

- **`GgufFile`:** `GgufFile.Open(path)`, `TryGetTensor`, `TryGetTensorFileLocation`, `GetMetadataString`, `TryGetFirstMetadataInt32`, `GetMetadataUInt32`, `GetEosTokenIds`.
- **`GgufStageSlicer`:** Partitions GGUF files cleanly by layer ranges for pipeline-parallel stages (`SelectStageTensors`, `CalculateStageByteSize`, `SliceStage`, `SliceStageToFile`, `GetStageFileName`).
- **`SafetensorsFile`:** `SafetensorsFile.Open(path)` and `ContainsTensor(name)` returning zero-copy `SafetensorsTensorInfo` offsets.
- **Tokenizers:** `BpeTokenizer` (`Encode`, `FormatConversation`, `FormatChatML`, `FormatChat`, `SanitizePrompt`, `TryGetTokenId`, `IsStopToken`) and `SimdRadixTrieTokenizer` (`CreateDefault`, span-based zero-allocation `Encode`, `CountTokens`, `Decode`, `DecodeToString`).
- **`Sampler`:** Standalone autoregressive sampling helper (`Sample`, `SamplingOptions.Greedy`, `Sampler.ApplyRepetitionPenalty`).
