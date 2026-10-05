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
using Glacier.Inference.Audio;
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;
using Glacier.Inference.Vision;
using Glacier.Inference.Video;
using Glacier.Inference.Image;
using Glacier.Inference.Image.Gguf;

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
        Console.WriteLine("Usage: glacier <command> [options]");
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
        Console.WriteLine("  voice   <voices|tts|stt|demo>        Human-quality speech synthesis (16 voices), Whisper STT & voice");
        Console.WriteLine("  vision  <demo|query> [options]       Vision-Language Model (VLM) patchification & analysis");
        Console.WriteLine("  video   <generate|demo|query>        Generative video production (DiT+VAE) & temporal reasoning");
        Console.WriteLine("  image   <demo|generate> [options]    Text-to-Image production (Flow Matching DiT + VAE)");
        Console.WriteLine();
        Console.WriteLine("Global Hardware & Engine Options (bench, run, serve):");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu, 0)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Execution engine (default: auto safe selection)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|fp32>  KV-cache precision (default: auto adaptive)");
        Console.WriteLine("  --ctx, -c <len>                      Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU pipeline split (e.g. 'auto', 'nvidia-rtx-4060:14,amd-890m:14')");
        Console.WriteLine();
        Console.WriteLine("Subcommand Help:");
        Console.WriteLine("  glacier <command> --help             Detailed help, options, and examples for any command");
        Console.WriteLine("  glacier help <command>               Alternative syntax for subcommand help");
        Console.WriteLine("  Example: glacier run --help");
        Console.WriteLine("  Example: glacier bench --help");
        Console.WriteLine("  Example: glacier serve --help");
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
        Console.WriteLine("glacier image - Pure C# .NET 10 Generative Image Production (Flux.1 / SDXL / DiT GGUF + VAE)");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier image demo [--out <file.png>]");
        Console.WriteLine("  glacier image info <model.gguf>");
        Console.WriteLine("  glacier image generate \"<prompt>\" [options]");
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
        Console.WriteLine("glacier voice - Pure C# .NET 10 Voice Subsystem (Kokoro TTS + Whisper STT)");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier voice tts <text> [--out <file.wav>] [--voice <id>] [--speed <float>] [--play]");
        Console.WriteLine("  glacier voice voices                                 # List all 16 USA & UK voices");
        Console.WriteLine("  glacier voice stt <audio.wav>                        # Transcribe speech with Whisper");
        Console.WriteLine("  glacier voice demo [--play]                          # Run full-duplex multi-accent demonstration");
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
        Console.WriteLine("glacier vision - Vision-Language Model (VLM) patchification & analysis");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier vision demo");
        Console.WriteLine("  glacier vision query <image> \"<prompt>\"");
    }

    private static void PrintVideoHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier video - Generative Video Production & Video-Language Multimodal Reasoning");
        Console.WriteLine("                Pure C# .NET 10 | 3D-RoPE + Spatio-Temporal DiT + 3D VAE + APNG/GIF/AVI");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier video generate \"<prompt>\" [options]");
        Console.WriteLine("  glacier video demo");
        Console.WriteLine("  glacier video query <dir> \"<prompt>\"");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  generate \"<prompt>\" [options]        Synthesize video from prompt (Spatio-Temporal DiT & 3D VAE)");
        Console.WriteLine("  demo                                 Execute end-to-end video synthesis & temporal reasoning");
        Console.WriteLine("  query <dir> \"<prompt>\"               Analyze directory of raw video frames and answer queries");
        Console.WriteLine();
        Console.WriteLine("Options (for generate):");
        Console.WriteLine("  --image <path>                       Optional reference image to animate with camera motion");
        Console.WriteLine("  --frames <int>                       Number of output frames (default: 16)");
        Console.WriteLine("  --fps <int>                          Framerate in FPS (default: 8)");
        Console.WriteLine("  --width <int>                        Video width in pixels (default: 256)");
        Console.WriteLine("  --height <int>                       Video height in pixels (default: 256)");
        Console.WriteLine("  --steps <int>                        Flow matching ODE steps (default: 4)");
        Console.WriteLine("  --motion <name>                      pan-right, pan-left, zoom-in, zoom-out, tilt-up, orbit, fluid, static (default: pan-right)");
        Console.WriteLine("  --format <ext>                       apng, gif, avi, or all (default: apng)");
        Console.WriteLine("  --out <path>                         Output file path (default: glacier_video.apng)");
        Console.WriteLine("  --seed <int>                         Random seed for reproducible video synthesis");
    }

    private static void PrintDevicesHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier devices - Enumerate compute hardware and safe driver engines");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier devices");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Scans the system for all physical GPU accelerators and host CPUs using pure DXGI");
        Console.WriteLine("  and native driver queries. Displays total dedicated VRAM, unified system RAM,");
        Console.WriteLine("  and lists which execution engines (BareMetal SASS, Direct3D 12 Compute, DirectML, CPU)");
        Console.WriteLine("  are safe for each device (preventing Windows DWM display driver TDR timeouts).");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier devices");
    }

    private static void PrintConfigHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier config - View or set persistent hardware & engine preferences");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier config [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --device <id|name>                   Set default target device (e.g. nvidia-rtx-4060, amd-890m, cpu, auto)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Set default execution engine");
        Console.WriteLine("  --reset                              Reset configuration to auto-detected defaults");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Description:");
        Console.WriteLine("  Persists configuration preferences across sessions in ~/.glacier/settings.json.");
        Console.WriteLine("  When set, commands (run, bench, serve) automatically use these preferences unless overridden.");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier config                                     # View current persistent configuration");
        Console.WriteLine("  glacier config --device nvidia-rtx-4060            # Default to NVIDIA discrete GPU");
        Console.WriteLine("  glacier config --device amd-890m --engine directml  # Default to AMD iGPU with D3D12/DirectML");
        Console.WriteLine("  glacier config --reset                             # Reset to factory auto-selection");
    }

    private static void PrintInspectHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier inspect - Inspect GGUF model architecture, metadata & tensors");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier inspect <model.gguf> [options]");
        Console.WriteLine("  glacier inspect -m <model.gguf> [options]");
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
        Console.WriteLine("  glacier inspect models/Qwen2.5-Coder-7B-Enterprise-Q8_0.gguf");
        Console.WriteLine("  glacier inspect models/Qwen3-30B-A3B-Q4_K_M.gguf --filter exps");
        Console.WriteLine("  glacier inspect -m models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf -a");
    }

    private static void PrintBenchHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier bench - Run speed & comparative performance benchmark");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier bench <model.gguf> [options]");
        Console.WriteLine("  glacier bench -m <model.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  -n, --tokens <count>                 Number of tokens to generate (default: 25)");
        Console.WriteLine("  -p, --prompt \"<text>\"                Custom benchmark prompt (default: CPU cache question)");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|fp32>  KV-cache precision (default: auto)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
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
        Console.WriteLine("  glacier bench model.gguf -n 50");
        Console.WriteLine("  glacier bench model.gguf -c 4096 --device nvidia-rtx-4060 --engine baremetal");
        Console.WriteLine("  glacier bench model.gguf --split auto");
        Console.WriteLine("  glacier bench model.gguf --compare-ollama http://127.0.0.1:11434 --compare-model qwen2.5:7b");
    }

    private static void PrintRunHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier run - Interactive streaming chat REPL or single-shot prompt");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier run <model.gguf> [prompt] [options]");
        Console.WriteLine("  glacier run -m <model.gguf> [prompt] [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  [prompt]                             Single-shot prompt. If omitted, launches interactive REPL.");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  -n, --tokens, --max-tokens <count>   Maximum tokens to generate per response (default: 512)");
        Console.WriteLine("  --temp, --temperature <float>        Sampling temperature (default: 0.7)");
        Console.WriteLine("  --top-p <float>                      Top-p nucleus sampling cutoff (default: 0.9)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|fp32>  KV-cache precision (default: auto)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("Interactive REPL Commands:");
        Console.WriteLine("  exit, quit                           Terminate the interactive chat session");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier run model.gguf                                # Launch interactive chat REPL");
        Console.WriteLine("  glacier run model.gguf \"Explain quicksort in C#\"       # Single-shot streaming answer");
        Console.WriteLine("  glacier run model.gguf -c 4096 --temp 0.2              # Low-temperature coding mode");
        Console.WriteLine("  glacier run model.gguf --split auto                   # Dual-GPU pipeline parallelism");
        Console.WriteLine("  glacier run model.gguf --device amd-890m --engine directml");
    }

    private static void PrintServeHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier serve - Start Ollama & OpenAI compatible HTTP inference server");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier serve <model.gguf> [options]");
        Console.WriteLine("  glacier serve -m <model.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Arguments & Options:");
        Console.WriteLine("  <model.gguf>, -m <model.gguf>        Path to GGUF model file (required)");
        Console.WriteLine("  --port <port>                        Listening port (default: 11434)");
        Console.WriteLine("  --host <host>                        Listening host IP (default: 0.0.0.0)");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|fp32>  KV-cache precision (default: auto)");
        Console.WriteLine("  --split <spec>                       Heterogeneous multi-GPU split (e.g. 'auto', 'nvidia:14,amd:14')");
        Console.WriteLine("  -h, --help                           Show this help message");
        Console.WriteLine();
        Console.WriteLine("HTTP API Endpoints:");
        Console.WriteLine("  POST /api/generate                   Ollama generation (supports streaming ndjson & non-streaming)");
        Console.WriteLine("  POST /api/chat                       Ollama multi-turn chat format");
        Console.WriteLine("  GET  /api/tags                       Ollama model tags listing");
        Console.WriteLine("  POST /v1/chat/completions            OpenAI-compatible chat completions");
        Console.WriteLine("  GET  /v1/models                      OpenAI-compatible model list");
        Console.WriteLine();
        Console.WriteLine("Examples:");
        Console.WriteLine("  glacier serve model.gguf                              # Listen on port 11434 (Ollama default)");
        Console.WriteLine("  glacier serve model.gguf --port 8080 --host 127.0.0.1  # Bind to localhost:8080");
        Console.WriteLine("  glacier serve model.gguf --device nvidia-rtx-4060 --engine baremetal");
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
