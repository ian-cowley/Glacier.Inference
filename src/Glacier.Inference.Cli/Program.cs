namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Glacier.Inference.Diagnostics;
using Glacier.Media.Audio;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;
using Glacier.Media.Vision;
using Glacier.Media.Video;
using Glacier.Media.Image;
using Glacier.Media.Image.Gguf;

public static partial class Program

{
    public static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        GlacierDiagnostics.Logger = new ConsoleGlacierLogger();

        if (args.Length == 0 || args[0] is "-h" or "--help" or "help" or "/?" or "-?")
        {
            if (args.Length > 1 && args[0].Equals("help", StringComparison.OrdinalIgnoreCase))
            {
                PrintCommandHelp(args[1]);
                return 0;
            }
            PrintHelp();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] cmdArgs = args.Length > 1 ? args[1..] : [];

        if (HasHelpFlag(cmdArgs))
        {
            PrintCommandHelp(command);
            return 0;
        }

        try
        {
            return command switch
            {
                "devices" => RunDevices(cmdArgs),
                "config" => RunConfig(cmdArgs),
                "inspect" => RunInspect(cmdArgs),
                "bench" => await RunBenchAsync(cmdArgs),
                "run" => await RunChatAsync(cmdArgs),
                "serve" => await RunServeAsync(cmdArgs),
                "eval-kl" => RunEvalKl(cmdArgs),
                "bounce" => await RunBounceAsync(cmdArgs),
                "voice" => RunVoice(cmdArgs),
                "vision" => RunVision(cmdArgs),
                "video" => RunVideo(cmdArgs),
                "image" => RunImage(cmdArgs),
                _ => HandleUnknownCommand(command)
            };
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine($"\n[Error] {ex.Message}");
            Console.ResetColor();
            return 1;
        }
    }

    private static bool HasHelpFlag(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a is "--help" or "help" or "/?" or "-?") return true;
            if (a == "-h")
            {
                if (i + 1 < args.Length && int.TryParse(args[i + 1], out _))
                    continue;
                return true;
            }
        }
        return false;
    }

    private static void PrintHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║               GLACIER.INFERENCE - Pure C# .NET 10 LLM                 ║");
        Console.WriteLine("║        Direct OS & Driver Interop | Sub-100ms Cold Start | Hardware SIMD    ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage: glacier-ai <command> [options]");
        Console.WriteLine();
        Console.WriteLine("Hardware & Configuration Commands:");
        Console.WriteLine("  devices                              Enumerate detected GPUs, VRAM, and safe driver engines");
        Console.WriteLine("  config  [options]                    View or set persistent hardware & engine preferences");
        Console.WriteLine();
        Console.WriteLine("Model Execution Commands:");
        Console.WriteLine("  inspect <model.gguf>                 Inspect model architecture, metadata & tensors");
        Console.WriteLine("  bench   <model.gguf> [options]       Run speed & comparative benchmark");
        Console.WriteLine("  run     <model.gguf> [prompt]        Interactive streaming chat or single prompt");
        Console.WriteLine("  serve   <model.gguf> [options]       Start Ollama & OpenAI compatible HTTP server");
        Console.WriteLine("  eval-kl <cand.gguf> --ref <ref.gguf> Evaluate KL divergence D_KL(P_ref || Q_cand) vs reference");
        Console.WriteLine("  bounce  --primary <url> [options]    3-Machine Peer Critic + Compiler Bounce Proxy");
        Console.WriteLine("  voice   <voices|tts|stt|demo>        KittenTTS Neural Speech Synthesis & Formant TTS + Whisper STT");
        Console.WriteLine("  vision  <demo|query> [options]       [Experimental] Synthetic VisionPipeline patchification & analysis");
        Console.WriteLine("  video   <generate|demo|query>        Generative video production (DiT+VAE) & temporal reasoning");
        Console.WriteLine("  image   <demo|generate> [options]    Text-to-Image production (Flow Matching DiT + VAE)");
        Console.WriteLine();
        Console.WriteLine("Global Hardware & Engine Options (bench, run, serve):");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu, 0)");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Execution engine (default: auto safe selection)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q8_0|q4_0|fp32> KV-cache precision (default: auto adaptive)");
        Console.WriteLine("  --ctx, -c <len>                      Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU pipeline split (e.g. 'auto', 'nvidia-rtx-4060:14,amd-890m:14')");
        Console.WriteLine();
        Console.WriteLine("Subcommand Help:");
        Console.WriteLine("  glacier-ai <command> --help          Detailed help, options, and examples for any command");
        Console.WriteLine("  glacier-ai help <command>            Alternative syntax for subcommand help");
        Console.WriteLine("  Example: glacier-ai run --help");
        Console.WriteLine("  Example: glacier-ai bench --help");
        Console.WriteLine("  Example: glacier-ai serve --help");
    }

    private static void PrintCommandHelp(string command)
    {
        switch (command.ToLowerInvariant())
        {
            case "devices":
                PrintDevicesHelp();
                break;
            case "config":
                PrintConfigHelp();
                break;
            case "inspect":
                PrintInspectHelp();
                break;
            case "bench":
                PrintBenchHelp();
                break;
            case "run":
                PrintRunHelp();
                break;
            case "serve":
                PrintServeHelp();
                break;
            case "eval-kl":
                PrintEvalKlHelp();
                break;
            case "bounce":
                PrintBounceHelp();
                break;
            case "voice":
                PrintVoiceHelp();
                break;
            case "vision":
                PrintVisionHelp();
                break;
            case "video":
                PrintVideoHelp();
                break;
            case "image":
                PrintImageHelp();
                break;
            default:
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"Unknown command: '{command}'\n");
                Console.ResetColor();
                PrintHelp();
                break;
        }
    }

    private static void PrintImageHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai image - Pure C# .NET 10 Generative Image Production (Flux.1 / SDXL / DiT GGUF + VAE)");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai image demo [--out <file.png>]");
        Console.WriteLine("  glacier-ai image info <model.gguf>");
        Console.WriteLine("  glacier-ai image generate \"<prompt>\" [options]");
        Console.WriteLine();
        Console.WriteLine("Options (for 'generate'):");
        Console.WriteLine("  -m, --model <file.gguf>              Path to Flux/SD DiT GGUF (auto-detected if in models/)");
        Console.WriteLine("  --vae <file.safetensors|gguf>        Path to VAE decoder (default: models/ae.safetensors)");
        Console.WriteLine("  --clip <file.safetensors>            Path to CLIP-L text encoder (default: models/clip_l.safetensors)");
        Console.WriteLine("  --t5 <file.gguf>                     Path to T5-XXL text encoder (default: models/t5xxl.gguf)");
        Console.WriteLine("  -w, --width <px>                     Output width in pixels (default: 512, multiple of 16)");
        Console.WriteLine("  -h, --height <px>                    Output height in pixels (default: 512, multiple of 16)");
        Console.WriteLine("  -s, --steps <count>                  Euler ODE flow matching steps (default: 4 for schnell)");
        Console.WriteLine("  --seed <int>                         PRNG seed for reproducible latents (default: random)");
        Console.WriteLine("  -o, --out <file.png>                 Destination PNG or BMP image file (default: generated.png)");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  dotnet run --project Glacier.Inference/src/Glacier.Inference.Cli -- image generate \"a futuristic neon metropolis\" -o city.png");
        Console.WriteLine("  dotnet run --project Glacier.Inference/src/Glacier.Inference.Cli -- image generate \"a close up portrait of an astronaut\" -w 512 -h 512 -s 4 --seed 42 -o astronaut.png");
    }


    private static void PrintVoiceHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai voice - Pure C# .NET 10 Voice Subsystem (KittenTTS Neural Speech Synthesis & Formant TTS + Audio DSP)");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai voice tts <text> [--out <file.wav>] [--voice <id>] [--speed <float>] [--play]");
        Console.WriteLine("  glacier-ai voice voices                              # List all 16 USA & UK voices");
        Console.WriteLine("  glacier-ai voice stt <audio.wav>                     # Transcribe speech with Whisper");
        Console.WriteLine("  glacier-ai voice demo [--play]                       # Run full-duplex multi-accent demonstration");
        Console.WriteLine();
        Console.WriteLine("Supported Voices (USA & British Accents):");
        Console.WriteLine("  USA Female:     af_heart, af_bella, af_sarah, af_sky");
        Console.WriteLine("  USA Male:       am_adam, am_michael, am_echo, am_eric");
        Console.WriteLine("  British Female: bf_emma, bf_isabella, bf_alice, bf_lily");
        Console.WriteLine("  British Male:   bm_george, bm_lewis, bm_daniel, bm_fable");
    }

    private static void PrintVisionHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai vision - [Experimental] Synthetic VisionPipeline patchification & analysis");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai vision demo");
        Console.WriteLine("  glacier-ai vision query <image> \"<prompt>\" --synthetic-demo");
    }

    private static void PrintVideoHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai video - Generative Video Production & Video-Language Multimodal Reasoning");
        Console.WriteLine("                   Pure C# .NET 10 | 3D-RoPE + Spatio-Temporal DiT + 3D VAE + APNG/GIF/AVI");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai video generate \"<prompt>\" [options]");
        Console.WriteLine("  glacier-ai video demo");
        Console.WriteLine("  glacier-ai video query <dir> \"<prompt>\"");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  generate \"<prompt>\" [options]        Synthesize video from prompt (Spatio-Temporal DiT & 3D VAE)");
        Console.WriteLine("  demo                                 Execute end-to-end video synthesis & temporal reasoning");
        Console.WriteLine("  query <dir> \"<prompt>\"               Analyze directory of raw video frames and answer queries");
        Console.WriteLine();
        Console.WriteLine("Options (for generate):");
        Console.WriteLine("  --image <path>                       Optional reference image to animate with camera motion");
        Console.WriteLine("  --frames <int>                       Number of output frames (default: 17)");
        Console.WriteLine("  --duration <float>                   Optional duration in seconds (overrides --frames via duration * fps)");
        Console.WriteLine("  --fps <int>                          Framerate in FPS (default: 16)");
        Console.WriteLine("  --width <int>                        Video width in pixels (default: 832)");
        Console.WriteLine("  --height <int>                       Video height in pixels (default: 480)");
        Console.WriteLine("  --steps <int>                        Flow matching ODE steps (default: 25)");
        Console.WriteLine("  --guidance <float>                   Classifier-Free Guidance scale (default: 5.0)");
        Console.WriteLine("  --motion <name>                      pan-left|pan-right|tilt-up|tilt-down|zoom-in|zoom-out|static|fluid (default: pan-right)");
        Console.WriteLine("  --format <ext>                       apng, gif, avi, or all (default: apng)");
        Console.WriteLine("  --out <path>                         Output file path (default: glacier_video.apng)");
        Console.WriteLine("  --seed <int>                         Random seed for reproducible video synthesis");
    }

    private static void PrintDevicesHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai devices - Enumerate compute hardware and safe driver engines");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai devices");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Scans the system for all physical GPU accelerators and host CPUs using pure DXGI");
        Console.WriteLine("  and native driver queries. Displays total dedicated VRAM, unified system RAM,");
        Console.WriteLine("  and lists which execution engines (BareMetal SASS, Direct3D 12 Compute, DirectML, CPU)");
        Console.WriteLine("  are safe for each device (preventing Windows DWM display driver TDR timeouts).");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai devices");
    }

    private static void PrintConfigHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai config - View or set persistent hardware & engine preferences");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai config [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --device <id|name>                   Set default target device (e.g. nvidia-rtx-4060, amd-890m, cpu, auto)");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Set default execution engine");
        Console.WriteLine("  --reset                              Reset configuration to auto-detected defaults");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Persists configuration preferences across sessions in ~/.glacier/settings.json.");
        Console.WriteLine("  When set, commands (run, bench, serve) automatically use these preferences unless overridden.");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai config                                  # View current persistent configuration");
        Console.WriteLine("  glacier-ai config --device nvidia-rtx-4060         # Default to NVIDIA discrete GPU");
        Console.WriteLine("  glacier-ai config --device amd-890m --engine directml # Default to AMD iGPU with D3D12/DirectML");
        Console.WriteLine("  glacier-ai config --reset                          # Reset to factory auto-selection");
    }

    private static void PrintInspectHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai inspect - Inspect GGUF model architecture, metadata & tensors");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai inspect <model.gguf> [options]");
        Console.WriteLine("  glacier-ai inspect -m <model.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  -f, --filter <pattern>               Filter displayed tensors by name pattern (e.g. 'exps', 'attn')");
        Console.WriteLine("  -a, --all                            Display all tensors instead of capping at first 25");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Zero-copy memory-maps the GGUF header and metadata to display architecture family,");
        Console.WriteLine("  transformer layer count, attention heads, GQA ratio, RoPE frequencies, context window,");
        Console.WriteLine("  and total parameter storage size in gigabytes.");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai inspect models/Qwen2.5-Coder-7B-Enterprise-Q8_0.gguf");
        Console.WriteLine("  glacier-ai inspect models/Qwen3-30B-A3B-Q4_K_M.gguf --filter exps");
        Console.WriteLine("  glacier-ai inspect -m models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -a");
    }

    private static void PrintBenchHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai bench - Run speed & comparative performance benchmark");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai bench <model.gguf> [options]");
        Console.WriteLine("  glacier-ai bench -m <model.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  -n, --tokens <count>                 Number of tokens to generate (default: 25)");
        Console.WriteLine("  -p, --prompt \"<text>\"                Custom benchmark prompt (default: CPU cache question)");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q8_0|q4_0|fp32> KV-cache precision (default: auto)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
        Console.WriteLine("  --temp, --temperature <float>        Sampling temperature (default: 0.7; 0 for greedy)");
        Console.WriteLine("  --raw                                Skip chat template wrapping for the benchmark prompt");
        Console.WriteLine("  --ple-io <auto|direct|mmap>          PLE table transport mode (default: auto)");
        Console.WriteLine("  --host-arena-gib <float>             Host-RAM Expert Arena cap in GiB (0 = disable, default: auto)");
        Console.WriteLine("  --tiered-moe                         Force GPU+CPU Hosted MoE expert slot pooling on D3D12");
        Console.WriteLine("  --speculative <off|lookup|mtp|dflash|adaptive> Speculative decoding mode (default: off)");
        Console.WriteLine("  --draft-model <path>                 Detached MTP/NextN sidecar GGUF path");
        Console.WriteLine("  --draft-tokens <int>                 Max speculative draft tokens per round (default: auto 2 MoE / 4 dense)");
        Console.WriteLine("  --dflash-steps <int>                 DFlash block-diffusion refinement steps (default: 3)");
        Console.WriteLine("  --draft-vocab <full|lang|path.gdvb>  Draft vocabulary subset mode or binary file (default: full)");
        Console.WriteLine("  --moe-tail-skip <int>                Minimum MoE rank for speculative verify tail skipping (default: 0 = exact)");
        Console.WriteLine("  --compare-ollama <url>               Ollama base URL for side-by-side comparison");
        Console.WriteLine("                                       (e.g. http://127.0.0.1:11434 or http://remote-host:11434)");
        Console.WriteLine("  --compare-model <name>               Ollama model tag to query (default: qwen2.5:7b-instruct-32k)");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Measures cold load time, prompt prefill rate (tokens/sec), autoregressive generation rate");
        Console.WriteLine("  (tokens/sec), and total turnaround latency. When --compare-ollama is provided, executes an");
        Console.WriteLine("  identical prompt on the Ollama endpoint and generates a head-to-head performance table.");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai bench model.gguf -n 50");
        Console.WriteLine("  glacier-ai bench model.gguf -c 4096 --device nvidia-rtx-4060 --engine baremetal");
        Console.WriteLine("  glacier-ai bench model.gguf --split auto");
        Console.WriteLine("  glacier-ai bench model.gguf --compare-ollama http://127.0.0.1:11434 --compare-model qwen2.5:7b");
    }

    private static void PrintRunHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai run - Interactive streaming chat REPL or single-shot prompt");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai run <model.gguf> [prompt] [options]");
        Console.WriteLine("  glacier-ai run -m <model.gguf> [prompt] [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  [prompt]                             Single-shot prompt. If omitted, launches interactive REPL.");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  -n, --tokens, --max-tokens <count>   Maximum tokens to generate per response (default: 512)");
        Console.WriteLine("  --temp, --temperature <float>        Sampling temperature (default: 0.7; 0 for greedy)");
        Console.WriteLine("  --top-p <float>                      Top-p nucleus sampling cutoff (default: 0.9)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q8_0|q4_0|fp32> KV-cache precision (default: auto)");
        Console.WriteLine("  --lora <[name=]path>                 Attach 0x41524F4C LoRA adapter binary/directory (CPU)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
        Console.WriteLine("  --ple-io <auto|direct|mmap>          PLE table transport mode (default: auto)");
        Console.WriteLine("  --host-arena-gib <float>             Host-RAM Expert Arena cap in GiB (0 = disable, default: auto)");
        Console.WriteLine("  --tiered-moe                         Force GPU+CPU Hosted MoE expert slot pooling on D3D12");
        Console.WriteLine("  --speculative <off|lookup|mtp|dflash|adaptive> Speculative decoding mode (default: off)");
        Console.WriteLine("  --draft-model <path>                 Detached MTP/NextN sidecar GGUF path");
        Console.WriteLine("  --draft-tokens <int>                 Max speculative draft tokens per round (default: auto 2 MoE / 4 dense)");
        Console.WriteLine("  --dflash-steps <int>                 DFlash block-diffusion refinement steps (default: 3)");
        Console.WriteLine("  --draft-vocab <full|lang|path.gdvb>  Draft vocabulary subset mode or binary file (default: full)");
        Console.WriteLine("  --moe-tail-skip <int>                Minimum MoE rank for speculative verify tail skipping (default: 0 = exact)");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Interactive REPL Commands:");
        Console.WriteLine("  exit, quit                           Terminate the interactive chat session");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai run model.gguf                             # Launch interactive chat REPL");
        Console.WriteLine("  glacier-ai run model.gguf \"Explain quicksort in C#\"    # Single-shot streaming answer");
        Console.WriteLine("  glacier-ai run model.gguf -c 4096 --temp 0.2           # Low-temperature coding mode");
        Console.WriteLine("  glacier-ai run model.gguf --split auto                # Dual-GPU pipeline parallelism");
        Console.WriteLine("  glacier-ai run model.gguf --device amd-890m --engine directml");
    }

    private static void PrintServeHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai serve - Start Ollama & OpenAI compatible HTTP inference server");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai serve <model.gguf> [options]");
        Console.WriteLine("  glacier-ai serve -m <model.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  --port <port>                        Listening port (default: 11434)");
        Console.WriteLine("  --host <host>                        Listening host IP (default: 127.0.0.1)");
        Console.WriteLine("  --api-key <key>                      Optional API Key / Bearer token (or GLACIER_API_KEY env)");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q8_0|q4_0|fp32> KV-cache precision (default: auto)");
        Console.WriteLine("  --lora <[name=]path>                 Hot-load 0x41524F4C LoRA adapter (repeatable; dense CPU/CUDA)");
        Console.WriteLine("  --blocks <count>                     PagedAttention KV blocks (continuous batching mode)");
        Console.WriteLine("  -b, --batch-size <count>             Max parallel sequences (continuous batching mode, default: 16)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
        Console.WriteLine("  --max-think-tokens <auto|off|0|N>    Cap <think> tokens & auto-close (default: auto)");
        Console.WriteLine("  --sleep-memory                       Enable pillar-16 HOPE+MemEx sleep consolidation");
        Console.WriteLine("  --sleep-budget <tokens>              Sleep memory context budget in tokens (default: 400)");
        Console.WriteLine("  --sleep-checkpoint <path>            Persist Sleep MoE expansion weights on consolidation");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("HTTP API Endpoints:");
        Console.WriteLine("  POST /api/generate                   Ollama generation (supports streaming ndjson & non-streaming)");
        Console.WriteLine("  POST /api/chat                       Ollama multi-turn chat format");
        Console.WriteLine("  GET  /api/tags                       Ollama model tags listing");
        Console.WriteLine("  GET  /api/sleep/status               Sleep consolidation telemetry & trace stats");
        Console.WriteLine("  POST /api/sleep/trigger              Trigger immediate offline sleep consolidation cycle");
        Console.WriteLine("  GET  /v1/sleep/status                Glacier.Studio sleep status telemetry");
        Console.WriteLine("  POST /v1/sleep/trigger               Glacier.Studio sleep consolidation trigger");
        Console.WriteLine("  POST /v1/chat/completions            OpenAI-compatible chat completions");
        Console.WriteLine("  POST /v1/completions                 OpenAI-compatible text completions (continuous batch mode)");
        Console.WriteLine("  GET  /v1/models                      OpenAI-compatible model & LoRA adapter list");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier-ai serve model.gguf                           # Listen on port 11434 (Ollama default)");
        Console.WriteLine("  glacier-ai serve model.gguf --port 8080 --host 127.0.0.1 # Bind to localhost:8080");
        Console.WriteLine("  glacier-ai serve model.gguf --device nvidia-rtx-4060 --engine baremetal");
    }

    private static void PrintBounceHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier-ai bounce - 3-Machine Peer Critic + OS-Sandboxed Compiler Bounce Proxy");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier-ai bounce [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --primary <url>                      Primary Glacier/OpenAI server endpoint (default: http://192.168.1.108:18080)");
        Console.WriteLine("  --critic, --peer-critic <url>        Optional peer critic endpoint");
        Console.WriteLine("  --host <string>                      Listen host (default: 127.0.0.1)");
        Console.WriteLine("  --port <int>                         Listen port (default: 18090)");
        Console.WriteLine("  --harness <path>                     Optional EvaluationHarness executable path");
        Console.WriteLine("  --max-repairs <int>                  Max compiler/critic repair passes (0-3, default: 2)");
        Console.WriteLine("  --compiler-bounce                    Enable OS-sandboxed compiler/syntax verifier (default)");
        Console.WriteLine("  --no-compiler-bounce                 Disable compiler/syntax verifier");
        Console.WriteLine("  --model-alias <name>                 Model ID exposed on /v1/models (default: Glacier-3Machine-Bounce)");
        Console.WriteLine("  -h, --help                           Show this help message");
    }

    public static bool TryParseLoraSpec(string? spec, out string name, out string resolvedPath, out string? error)
    {
        name = string.Empty;
        resolvedPath = string.Empty;
        error = null;

        if (string.IsNullOrWhiteSpace(spec))
        {
            error = "Error: --lora requires a <[name=]path> value.";
            return false;
        }

        string trimmedSpec = spec.Trim();
        string rawPath;
        if (!File.Exists(trimmedSpec) && !Directory.Exists(trimmedSpec))
        {
            int eqIdx = trimmedSpec.IndexOf('=');
            if (eqIdx == 0)
            {
                error = "Error: LoRA adapter name before '=' cannot be empty.";
                return false;
            }
            if (eqIdx > 0)
            {
                name = trimmedSpec[..eqIdx].Trim();
                rawPath = trimmedSpec[(eqIdx + 1)..].Trim();
            }
            else
            {
                rawPath = trimmedSpec;
                name = DeriveLoraName(rawPath);
            }
        }
        else
        {
            rawPath = trimmedSpec;
            name = DeriveLoraName(rawPath);
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            error = $"Error: Could not derive a valid adapter name from '{spec}'.";
            return false;
        }

        if (name.Contains(':', StringComparison.Ordinal))
        {
            error = $"Error: LoRA adapter name '{name}' cannot contain ':'.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(rawPath))
        {
            error = "Error: LoRA adapter file path cannot be empty.";
            return false;
        }

        resolvedPath = Directory.Exists(rawPath)
            ? Path.Combine(rawPath, "adapter_model.bin")
            : rawPath;

        if (!File.Exists(resolvedPath))
        {
            error = $"Error: LoRA adapter file not found: '{resolvedPath}'.";
            return false;
        }

        return true;
    }

    private static string DeriveLoraName(string rawPath)
    {
        string trimmed = rawPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (Directory.Exists(rawPath))
            return Path.GetFileName(trimmed);
        return Path.GetFileNameWithoutExtension(trimmed);
    }

    public static bool TryParseKvPrecisionArg(string? rawValue, out KvCachePrecision precision, out string? error)
    {
        error = null;
        if (string.IsNullOrWhiteSpace(rawValue))
        {
            precision = KvCachePrecision.Auto;
            return true;
        }

        if (!KvCachePrecisionExtensions.TryParse(rawValue, out precision))
        {
            error = $"Error: Unknown KV precision '{rawValue}'. Valid values: Auto, Fp32, Fp16, Fp8, Q8_0, Q4_0 (Int4).";
            return false;
        }

        return true;
    }

    internal static bool TryRequireFlagValue(string[] args, ref int index, string flag, out string value)
    {
        if (index + 1 >= args.Length)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine($"Error: Option '{flag}' requires a value.");
            Console.ResetColor();
            value = string.Empty;
            return false;
        }

        value = args[++index];
        return true;
    }

    public static (string? Device, InferenceEngineType Engine) ResolveSingleLoraTarget(
        bool hasLora,
        string? device,
        InferenceEngineType engine)
    {
        if (hasLora &&
            (string.IsNullOrWhiteSpace(device) || string.Equals(device.Trim(), "auto", StringComparison.OrdinalIgnoreCase)) &&
            engine == InferenceEngineType.Auto)
        {
            return ("cpu", InferenceEngineType.Cpu);
        }

        return (device, engine);
    }

    private static int HandleUnknownCommand(string command)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"Unknown command: '{command}'");
        Console.ResetColor();
        Console.WriteLine();
        PrintHelp();
        return 1;
    }

    // =========================================================================
    // 1. INSPECT COMMAND
    // =========================================================================
}
