# Glacier.Inference — Engine Improvement & Stabilization Plan ("Engine TLC")

**Document Type:** Architectural Stabilization & Kernel Enhancements  
**Target Platform:** .NET 10 (C# 13) / Native AOT  
**Applies to:** `Glacier.Inference.Engine`, `Glacier.Inference.Cli serve`, RoPE & Attention Kernels  

---

## 1. Problem Statement

During live integration testing against remote client bots with various models, three key runtime behaviors were observed that require stabilization:

1. **YaRN (Yet another RoPE extensioN) Frequency Scaling:**
   - 1-Million context models (such as `Qwen2.5-7B-Instruct-1M`) rely on YaRN positional frequency scaling.
   - When loaded into a backend that computes RoPE assuming standard linear frequency progression without YaRN scale factors, the positional embeddings disintegrate at sequence lengths > 256. Attention collapses, leaking raw template tokens or generating repetitive token loops (`aeaeaeAE...`).

2. **Template Delimiters & Stop Token Interception:**
   - Turn control tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) must be intercepted during the autoregressive sampling loop.
   - If the sampler doesn't match multi-token stop sequences or raw ChatML headers in the token stream, they leak into the client text output.

3. **Repetition Penalty & Sampling Degeneration:**
   - Under low entropy or when encountering repetitive subwords, the absence of an active repetition penalty window allows attention to fixate on the most recently generated tokens, producing infinite loops.

---

## 2. Technical Blueprint for Improvements

### 2.1 YaRN RoPE Metadata Parsing & Kernel Support
In `Glacier.Inference.Gguf.GgufFile` and `InferenceSession`:
1. Parse the following GGUF metadata keys when present:
   - `[arch].rope.freq_base` (e.g. 5,000,000 or 1,000,000)
   - `[arch].rope.scale`
   - `[arch].rope.scaling.type` (`yarn`, `linear`, `default`)
   - `[arch].rope.scaling.factor`
   - `[arch].rope.scaling.original_context_length`
   - `[arch].rope.scaling.finetuned`
2. Update the RoPE frequency calculation in `RoPEKernels.cs`:
   - Compute base frequencies modulated by YaRN ramp factor $m$:
     $$f_i = \text{freq\_base}^{-2i/d} \cdot \left((1 - \gamma_i) + \gamma_i \cdot \frac{1}{\text{factor}}\right)$$
   - This ensures 1M context models function smoothly without positional distortion.

### 2.2 Robust Stop Sequence & Delimiter Filtering
In `Program.Serve.cs` and `InferenceSession.GenerateAsync`:
1. Maintain a list of active stop sequences:
   - `<|im_end|>`
   - `<|im_start|>`
   - `<|endoftext|>`
   - `stop(eos_token_id)`
2. Buffer output tokens by $K$ tokens (where $K$ is the length of the longest stop sequence) before writing to the SSE / NDJSON stream to prevent partial token leakage.

### 2.3 Repetition Penalty Window
In `Glacier.Inference.Sampling`:
1. Maintain a circular sliding window of the last $N$ tokens (default $N = 64$).
2. Prior to softmax, apply logit penalty:
   $$\text{logit}_i = \begin{cases} \text{logit}_i / \text{penalty} & \text{if } \text{logit}_i > 0 \\ \text{logit}_i \cdot \text{penalty} & \text{if } \text{logit}_i \le 0 \end{cases}$$
   where default `penalty = 1.1`.

---

## 3. Recommended Models in the Interim

Until the YaRN scaling and LoRA fusion remediations are applied:
* **`llama3.2:latest`** (3.2B parameters, standard RoPE, rock-solid output).
* **`qwen-7b:latest`** (standard 32k context, standard RoPE).
* **`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`** (base unmerged model).
