# Multimodal Embeddings, Image, Video, Vision & Voice

This manual covers multimodal embedding models and the `image`, `video`, `vision`, and `voice` media pipelines in `Glacier.Inference`.

## 1. Multimodal Embeddings (`EmbeddingGemma2Model`)

Encoder-only embedding models (`gemma-embedding2`) run through `Glacier.Inference.Embedding.EmbeddingGemma2Model` rather than `InferenceSession`:

```csharp
using Glacier.Inference.Embedding;
using Glacier.Inference.Gguf;

using var model = EmbeddingGemma2Model.Load("models/embeddinggemma-2-Q8_0.gguf", new EmbeddingGemma2Options());
float[] textEmbedding = model.Embed("Semantic search query");

// Attach multimodal projector weights for image or audio inputs
model.AttachMultimodalProjector("models/mmproj-embeddinggemma-2-Q8_0.gguf");
// model.EmbedImage(rgbBytes, width, height);
```

- Check GGUF compatibility with `EmbeddingGemma2Model.IsSupported(gguf)`.
- Choose CPU/Vulkan target via `EmbeddingDevice` and preprocessing helpers `Gemma4AudioPreprocessor` (16 kHz audio) and `Gemma4ImagePreprocessor` (RGB frames) along with `ForwardHidden` and span-writing `Embed(text, Span<float>)` overloads.

## 2. Image Generation (`glacier image`, FLUX.1 DiT + VAE)

```bash
glacier image demo --out "crystal_fortress.png"
glacier image info "models/flux1-schnell-Q4_K_S.gguf"
glacier image generate "A cybernetic dragon on a mountain peak" -w 1024 -h 1024 --steps 4 --seed 42 -o dragon.png
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m, --model` | Auto-detected in `models/` | FLUX/SD DiT GGUF file |
| `--vae` | `models/ae.safetensors` | VAE decoder (`.safetensors` or `.gguf`) |
| `--clip` | `models/clip_l.safetensors` | CLIP-L text encoder |
| `--t5` | `models/t5xxl.gguf` | T5-XXL text encoder |
| `-w, --width` / `-h, --height` | `512` / `512` | Output pixels (multiples of 16; `1024x1024` uses tiled VAE decoding) |
| `-s, --steps` | `4` | Flow-matching Euler ODE steps |
| `--seed` | Random | PRNG seed for reproducible noise latents |
| `-o, --out` | `generated.png` | Output PNG or BMP path |

## 3. Video Generation (`glacier video`, Wan 2.1 3D DiT)

```bash
glacier video generate "Glacier fjords under aurora borealis" --width 832 --height 480 --steps 25 --duration 15 --fps 16 --motion pan-right --format all --out glacier_pan.apng
glacier video generate "Camera zoom into the mountain" --image "photo.png" --motion zoom-in --format all --out zoom.apng
glacier video demo
glacier video query frames_dir/ "Describe the motion across frames."
```

Key flags: `--image`, `--frames` (`17`), `--duration`, `--fps` (`16`), `--width` (`832`), `--height` (`480`), `--steps` (`25`), `--guidance` (`5.0`), `--motion` (`pan-left`, `pan-right`, `tilt-up`, `tilt-down`, `zoom-in`, `zoom-out`, `static`, `fluid`), `--format` (`apng`, `gif`, `avi`, `all`), `--seed`, `--out`.

## 4. Vision (`glacier vision`)

Experimental synthetic `VisionPipeline` patchification and analysis utility:

```bash
glacier vision demo
glacier vision query scene.png "Describe the scene" --synthetic-demo
```

## 5. Voice & Speech (`glacier voice`)

Provides neural KittenTTS synthesis, procedural Kokoro glottal formant synthesis, and OpenAI-compatible Whisper speech-to-text:

```bash
glacier voice voices
glacier voice tts "All systems nominal." --out speech.wav --voice af_heart --speed 1.0 --play
glacier voice stt speech.wav
glacier voice demo --play
```

Built-in 16 voice IDs:
- **US Female:** `af_heart`, `af_bella`, `af_sarah`, `af_sky`
- **US Male:** `am_adam`, `am_michael`, `am_echo`, `am_eric`
- **UK Female:** `bf_emma`, `bf_isabella`, `bf_alice`, `bf_lily`
- **UK Male:** `bm_george`, `bm_lewis`, `bm_daniel`, `bm_fable`
