# Speculative Decoding (`lookup`, `mtp`, `dflash`, `adaptive`)

`Glacier.Inference` supports speculative draft-and-verify decoding via the `glacier run` CLI and programmatically through `SpeculativeEngine`. Draft providers propose candidate tokens that the target model verifies in a batched forward pass.

## 1. CLI Flags (`glacier run`)

| Flag | Values | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `--speculative` | `off`, `lookup`, `mtp`, `dflash`, `adaptive` | `off` | Selects the speculative drafting strategy |
| `--draft-model` | Path to GGUF | None | Detached MTP / NextN sidecar draft GGUF |
| `--draft-tokens` | Integer | Auto (`2` MoE / `4` dense) | Max draft tokens proposed per round |
| `--dflash-steps` | Integer | `3` | DFlash block-diffusion refinement steps |
| `--draft-vocab` | `full`, `lang`, or `path.gdvb` | `full` | Draft vocabulary subset filter (`DraftVocabularySubset`) |
| `--moe-tail-skip` | Integer | `0` (`exact`) | Minimum MoE expert rank verified during speculative tail skipping |

Example:

```bash
glacier run "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" "Implement binary search in C#." --speculative adaptive --draft-tokens 4
```

## 2. Draft Providers & Cost-Gated Routing (`Glacier.Inference.Engine`)

### Modes & Classes
- **`PromptLookupDraftProvider`:** Zero-weight n-gram matcher proposing continuations repeated from prompt and conversation history (`lookup`).
- **`DFlashDraftProvider`:** Context-conditioned block-diffusion drafter (`CreateContextConditioned`, `TryCreateFromModel`, `LoadFromSidecar`), configured by `DFlashOptions` and backed by `DFlashWeights` over an `IHiddenStateSource`.
- **`CostGatedDraftPolicy` & `AdaptiveCompositeDraftProvider`:** Dynamically weighs candidate routes (`DraftRoute` -> `DraftDecision`) using rolling acceptance rates (`RecordOutcome`, `ComputeExpectedCommittedTokens`, `GetAcceptanceEstimate`) so low-acceptance turns skip wasted draft passes.
- **`SpeculativeEngine`:** Orchestrates draft generation, batched target verification, KV-cache rollback on mismatch, and streaming via `GenerateAsync(prompt, SpeculativeOptions?, formatChat, onToken)`.
