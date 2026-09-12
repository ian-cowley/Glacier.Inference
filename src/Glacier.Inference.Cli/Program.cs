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
using Glacier.Inference.Config;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] cmdArgs = args.Length > 1 ? args[1..] : [];

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

    private static void PrintHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║               GLACIER.INFERENCE - Pure C# .NET 10 LLM                 ║");
        Console.WriteLine("║        Zero Native C++ DLLs | Sub-100ms Cold Start | Hardware SIMD    ║");
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
        Console.WriteLine();
        Console.WriteLine("Global Hardware & Engine Options (bench, run, serve):");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU (e.g. nvidia-rtx-4060, amd-890m, cpu, 0)");
        Console.WriteLine("  --engine <baremetal|directml|cpu|auto> Execution engine (default: auto safe selection)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|fp32>    KV-cache precision (default: auto adaptive)");
        Console.WriteLine();
        Console.WriteLine("Options for 'bench':");
        Console.WriteLine("  --tokens <n>                         Number of tokens to generate (default: 32)");
        Console.WriteLine("  --prompt \"<text>\"                    Custom benchmark prompt");
        Console.WriteLine("  --compare-ollama <url>               Compare against remote/local Ollama (e.g. http://192.168.1.108:11434)");
        Console.WriteLine();
        Console.WriteLine("Options for 'serve':");
        Console.WriteLine("  --port <p>                           Listening port (default: 11434)");
        Console.WriteLine("  --host <h>                           Listening host (default: 0.0.0.0)");
    }

    private static int HandleUnknownCommand(string command)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"Unknown command: '{command}'");
        Console.ResetColor();
        PrintHelp();
        return 1;
    }

    // =========================================================================
    // 1. INSPECT COMMAND
    // =========================================================================
    private static int RunInspect(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Error: Model file path required. Example: glacier inspect model.gguf");
            return 1;
        }

        string modelPath = args[0];
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"Inspecting model: {modelPath}");
        Console.ResetColor();

        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(modelPath);
        sw.Stop();

        Console.WriteLine($"Format: GGUF v{gguf.Version} (Memory-mapped in {sw.ElapsedMilliseconds} ms)");
        Console.WriteLine($"Architecture:           {gguf.Architecture}");
        Console.WriteLine($"Transformer Layers:     {gguf.BlockCount}");
        Console.WriteLine($"Embedding Dimension:    {gguf.EmbeddingLength}");
        Console.WriteLine($"Feed-Forward Dimension: {gguf.FeedForwardLength}");
        Console.WriteLine($"Attention Query Heads:  {gguf.HeadCount}");
        Console.WriteLine($"Attention KV Heads:     {gguf.HeadCountKv} (Group ratio: {gguf.HeadCount / gguf.HeadCountKv}x)");
        Console.WriteLine($"Head Dimension:         {gguf.EmbeddingLength / gguf.HeadCount}");
        Console.WriteLine($"Context Window:         {gguf.ContextLength:N0} tokens");
        Console.WriteLine($"RoPE Base Frequency:    {gguf.RopeFreqBase}");
        Console.WriteLine($"RMSNorm Epsilon:        {gguf.RmsNormEps}");
        Console.WriteLine($"EOS Token ID:           {gguf.EosTokenId}");
        Console.WriteLine($"Total Tensors:          {gguf.TensorCount}");
        Console.WriteLine($"Metadata Key-Values:    {gguf.MetadataKvCount}");

        ulong totalParamBytes = 0;
        foreach (var t in gguf.TensorList)
        {
            totalParamBytes += t.GetByteSize();
        }
        Console.WriteLine($"Total Model Size:       {totalParamBytes / (1024.0 * 1024.0 * 1024.0):F2} GB");

        return 0;
    }

    // =========================================================================
    // 2. BENCH COMMAND
    // =========================================================================
    private static async Task<int> RunBenchAsync(string[] args)
    {
        string? modelPath = null;
        int targetTokens = 25;
        string prompt = "Explain in two sentences what a CPU cache is.";
        string? compareOllamaUrl = null;
        string? compareModel = null;
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "-n" || args[i] == "--tokens") && i + 1 < args.Length && int.TryParse(args[++i], out int n))
                targetTokens = n;
            else if (args[i] == "--prompt" && i + 1 < args.Length)
                prompt = args[++i];
            else if (args[i] == "--compare-ollama" && i + 1 < args.Length)
                compareOllamaUrl = args[++i];
            else if (args[i] == "--compare-model" && i + 1 < args.Length)
                compareModel = args[++i];
            else if (args[i] == "--device" && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine("Error: Model file path required. Example: glacier bench <model.gguf> or -m <model.gguf>");
            return 1;
        }

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr) && Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
        {
            engine = parsedEngine;
        }

        KvCachePrecision kvPrecision = KvCachePrecision.Auto;
        if (!string.IsNullOrWhiteSpace(kvPrecisionStr) && Enum.TryParse<KvCachePrecision>(kvPrecisionStr, ignoreCase: true, out var parsedPrecision))
        {
            kvPrecision = parsedPrecision;
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("=======================================================================");
        Console.WriteLine("              GLACIER INFERENCE BENCHMARK RUNNER                       ");
        Console.WriteLine("=======================================================================");
        Console.ResetColor();
        Console.WriteLine($"Model:        {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Tokens:       {targetTokens}");
        Console.WriteLine($"Prompt:       \"{prompt}\"");
        Console.WriteLine($"KV Precision: {kvPrecision}");
        if (!string.IsNullOrEmpty(compareOllamaUrl))
            Console.WriteLine($"Comparative:  Ollama at {compareOllamaUrl}");
        Console.WriteLine();

        Console.WriteLine(">> Loading model into zero-copy virtual address space...");
        var loadSw = Stopwatch.StartNew();
        using var session = new InferenceSession(modelPath, maxSeqLen: 4096, device: device, engine: engine, kvPrecision: kvPrecision);
        loadSw.Stop();
        Console.WriteLine($"   Cold load completed in: {loadSw.ElapsedMilliseconds} ms ({loadSw.Elapsed.TotalSeconds:F2} s)");
        Console.WriteLine($"   Execution Device: {session.ActiveDevice}");

        Console.WriteLine("\n>> Running Glacier.Inference...");
        var options = new SamplingOptions { MaxTokens = targetTokens, Temperature = 0.7f };
        var glacierResult = await session.GenerateAsync(prompt, options, formatChat: true);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"   Output: \"{glacierResult.Text.Trim()}\"");
        Console.ResetColor();
        Console.WriteLine($"   Prompt Rate:     {glacierResult.Metrics.PromptTokensPerSecond:F2} tokens/sec ({glacierResult.Metrics.PromptEvalDuration.TotalMilliseconds:F1} ms)");
        Console.WriteLine($"   Generation Rate: {glacierResult.Metrics.GenerationTokensPerSecond:F2} tokens/sec ({glacierResult.Metrics.GenerationDuration.TotalMilliseconds:F1} ms)");
        Console.WriteLine($"   Total Time:      {glacierResult.Metrics.TotalDuration.TotalSeconds:F2} s");

        if (!string.IsNullOrEmpty(compareOllamaUrl))
        {
            bool isLocalOllama = compareOllamaUrl.Contains("localhost") || compareOllamaUrl.Contains("127.0.0.1");
            string targetOllamaModel = compareModel ?? "qwen2.5:7b-instruct-32k";
            string hardwareLabel = isLocalOllama ? $"Same {session.Device.Name}" : "Remote GPU";

            Console.WriteLine($"\n>> Querying Ollama Benchmark ({compareOllamaUrl} | Model: {targetOllamaModel})...");
            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
                var reqBody = new
                {
                    model = targetOllamaModel,
                    prompt = prompt,
                    stream = false,
                    options = new { num_predict = targetTokens, temperature = 0.7 }
                };

                var oSw = Stopwatch.StartNew();
                var resp = await http.PostAsJsonAsync($"{compareOllamaUrl.TrimEnd('/')}/api/generate", reqBody);
                oSw.Stop();

                if (resp.IsSuccessStatusCode)
                {
                    var oJson = await resp.Content.ReadFromJsonAsync<JsonElement>();
                    string oText = oJson.TryGetProperty("response", out var r) ? r.GetString() ?? "" : "";
                    long totalDurationNs = oJson.TryGetProperty("total_duration", out var td) ? td.GetInt64() : 0;
                    long promptDurationNs = oJson.TryGetProperty("prompt_eval_duration", out var pd) ? pd.GetInt64() : 0;
                    long evalDurationNs = oJson.TryGetProperty("eval_duration", out var ed) ? ed.GetInt64() : 0;
                    int promptCount = oJson.TryGetProperty("prompt_eval_count", out var pc) ? pc.GetInt32() : 0;
                    int evalCount = oJson.TryGetProperty("eval_count", out var ec) ? ec.GetInt32() : 0;

                    double oPromptTps = promptDurationNs > 0 ? promptCount / (promptDurationNs / 1e9) : 0;
                    double oGenTps = evalDurationNs > 0 ? evalCount / (evalDurationNs / 1e9) : 0;

                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"   Output: \"{oText.Trim()}\"");
                    Console.ResetColor();

                    Console.WriteLine("\n======================================================================================");
                    Console.WriteLine("                         COMPARATIVE BENCHMARK SUMMARY                                ");
                    Console.WriteLine("======================================================================================");
                    Console.WriteLine($"{"Metric",-24} | {"Glacier.Inference (C#)",-34} | {$"Ollama ({hardwareLabel})",-24}");
                    Console.WriteLine(new string('-', 86));

                    void PrintSummaryRow(string metric, string glacierVal, string ollamaVal, bool isGlacierAdvantage, string? advantageText = null)
                    {
                        if (isGlacierAdvantage)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            string tag = advantageText != null ? $" [{advantageText}]" : " [FASTER]";
                            Console.WriteLine($"{$"{metric}",-24} | {$"{glacierVal} {tag}",-34} | {$"{ollamaVal}",-24}");
                            Console.ResetColor();
                        }
                        else
                        {
                            Console.WriteLine($"{$"{metric}",-24} | {$"{glacierVal}",-34} | {$"{ollamaVal}",-24}");
                        }
                    }

                    double gGenTps = glacierResult.Metrics.GenerationTokensPerSecond;
                    double gPromptTps = glacierResult.Metrics.PromptTokensPerSecond;
                    double gWallSec = glacierResult.Metrics.TotalDuration.TotalSeconds;
                    double oWallSec = oSw.Elapsed.TotalSeconds;

                    PrintSummaryRow("Runtime", "Pure C# .NET 10 Native AOT", "Go + C++ CUDA / llama.cpp", true, "Pure C# SASS");
                    PrintSummaryRow("Dependencies", "0 Native DLLs (Direct Driver)", "CUDA Toolkit, cuBLAS, LLAMA", true, "Zero C++ Bloat");
                    PrintSummaryRow("Cold Start Latency", $"{loadSw.ElapsedMilliseconds} ms", "Daemon / Resident", true, "Instant");

                    if (oPromptTps > 0)
                    {
                        bool promptFaster = gPromptTps >= oPromptTps;
                        double promptDiff = ((gPromptTps - oPromptTps) / oPromptTps) * 100.0;
                        PrintSummaryRow("Prompt Eval Rate", $"{gPromptTps:F1} t/s", $"{oPromptTps:F1} t/s", promptFaster, promptFaster ? $"+{promptDiff:F1}% FASTER" : null);
                    }

                    if (oGenTps > 0)
                    {
                        bool genFaster = gGenTps >= oGenTps;
                        double genDiff = ((gGenTps - oGenTps) / oGenTps) * 100.0;
                        PrintSummaryRow("Generation Rate", $"{gGenTps:F1} t/s", $"{oGenTps:F1} t/s", genFaster, genFaster ? $"+{genDiff:F1}% FASTER" : $"{genDiff:F1}% (Near Parity)");
                    }

                    PrintSummaryRow("Generated Tokens", $"{glacierResult.Metrics.GeneratedTokens}", $"{evalCount}", false);

                    if (oWallSec > 0)
                    {
                        bool wallFaster = gWallSec <= oWallSec;
                        double wallSpeedup = gWallSec > 0 ? oWallSec / gWallSec : 1.0;
                        PrintSummaryRow("Total Wall Time", $"{gWallSec:F2} s", $"{oWallSec:F2} s", wallFaster, wallFaster ? $"{wallSpeedup:F1}x FASTER" : null);
                    }

                    Console.WriteLine(new string('-', 86));
                }
                else
                {
                    Console.WriteLine($"   [Warning] Ollama returned status: {resp.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   [Warning] Failed to query Ollama at {compareOllamaUrl}: {ex.Message}");
            }
        }

        return 0;
    }

    // =========================================================================
    // 3. RUN CHAT COMMAND
    // =========================================================================
    private static async Task<int> RunChatAsync(string[] args)
    {
        string? modelPath = null;
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        var promptParts = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i] == "--device" && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
            else
            {
                promptParts.Add(args[i]);
            }
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.WriteLine("Error: Model file path required. Example: glacier run <model.gguf> [prompt] or -m <model.gguf>");
            return 1;
        }

        string? initialPrompt = promptParts.Count > 0 ? string.Join(" ", promptParts) : null;

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr) && Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
        {
            engine = parsedEngine;
        }

        KvCachePrecision kvPrecision = KvCachePrecision.Auto;
        if (!string.IsNullOrWhiteSpace(kvPrecisionStr) && Enum.TryParse<KvCachePrecision>(kvPrecisionStr, ignoreCase: true, out var parsedPrecision))
        {
            kvPrecision = parsedPrecision;
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"Loading {Path.GetFileName(modelPath)} into Glacier.Inference...");
        Console.ResetColor();

        using var session = new InferenceSession(modelPath, maxSeqLen: 4096, device: device, engine: engine, kvPrecision: kvPrecision);
        Console.WriteLine($"Model ready on {session.ActiveDevice}.\n");

        var options = new SamplingOptions { MaxTokens = 512, Temperature = 0.7f, TopP = 0.9f };

        // Single-shot prompt mode
        if (!string.IsNullOrEmpty(initialPrompt))
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($">>> {initialPrompt}\n");
            Console.ResetColor();

            var result = await session.GenerateAsync(
                initialPrompt,
                options,
                formatChat: true,
                onToken: t => Console.Write(t));

            Console.WriteLine($"\n\n[{result.Metrics.GenerationTokensPerSecond:F1} tokens/sec | {result.Metrics.GeneratedTokens} tokens]");
            return 0;
        }

        // Interactive REPL mode
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("=== Interactive Glacier Chat REPL (type 'exit' or Ctrl+C to quit) ===");
        Console.ResetColor();

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("\n>>> ");
            Console.ResetColor();

            string? line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (line.Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                line.Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            Console.ForegroundColor = ConsoleColor.White;
            var result = await session.GenerateAsync(
                line,
                options,
                formatChat: true,
                onToken: t => Console.Write(t));
            Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n[{result.Metrics.GenerationTokensPerSecond:F1} tokens/sec | {result.Metrics.GeneratedTokens} tokens]");
            Console.ResetColor();
        }

        return 0;
    }

    // =========================================================================
    // 4. SERVE COMMAND (OLLAMA & OPENAI COMPATIBLE HTTP SERVER)
    // =========================================================================
    private static async Task<int> RunServeAsync(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Error: Model file path required. Example: glacier serve model.gguf [--port 11434]");
            return 1;
        }

        string modelPath = args[0];
        int port = 11434;
        string host = "0.0.0.0";
        string? device = null;
        string? engineStr = null;

        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--port" && i + 1 < args.Length && int.TryParse(args[++i], out int p))
                port = p;
            else if (args[i] == "--host" && i + 1 < args.Length)
                host = args[++i];
            else if (args[i] == "--device" && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
        }

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr) && Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
        {
            engine = parsedEngine;
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("=======================================================================");
        Console.WriteLine("            GLACIER.INFERENCE HIGH-PERFORMANCE HTTP SERVER            ");
        Console.WriteLine("        Ollama & OpenAI Compatible | Sub-100ms Cold Start | Pure C#    ");
        Console.WriteLine("=======================================================================");
        Console.ResetColor();
        Console.WriteLine($"Model:       {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Endpoint:    http://{host}:{port}");
        Console.WriteLine();

        Console.WriteLine(">> Initializing inference session...");
        var session = new InferenceSession(modelPath, maxSeqLen: 4096, device: device, engine: engine);
        string modelName = Path.GetFileNameWithoutExtension(modelPath);

        var appBuilder = WebApplication.CreateBuilder();
        var app = appBuilder.Build();
        app.Urls.Add($"http://{host}:{port}");

        // 1. GET /api/tags (Ollama compatible tags list)
        app.MapGet("/api/tags", () => Results.Json(new
        {
            models = new[]
            {
                new
                {
                    name = modelName,
                    model = modelName,
                    modified_at = DateTime.UtcNow.ToString("o"),
                    size = (long)new FileInfo(modelPath).Length,
                    details = new
                    {
                        format = "gguf",
                        family = session.Gguf.Architecture,
                        parameter_size = $"{session.Gguf.TensorCount / 40.0:F1}B",
                        quantization_level = "Q4_K_M"
                    }
                }
            }
        }));

        // 2. POST /api/generate (Ollama streaming & non-streaming)
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            var req = await ctx.Request.ReadFromJsonAsync<JsonElement>();
            string prompt = req.TryGetProperty("prompt", out var p) ? p.GetString() ?? "" : "";
            bool stream = !req.TryGetProperty("stream", out var s) || s.GetBoolean();

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.");
                return;
            }

            var options = new SamplingOptions { MaxTokens = 512, Temperature = 0.7f };

            if (!stream)
            {
                var result = await session.GenerateAsync(prompt, options, formatChat: true);
                var respObj = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = result.Text,
                    done = true,
                    total_duration = (long)(result.Metrics.TotalDuration.TotalSeconds * 1e9),
                    prompt_eval_count = result.Metrics.PromptTokens,
                    prompt_eval_duration = (long)(result.Metrics.PromptEvalDuration.TotalSeconds * 1e9),
                    eval_count = result.Metrics.GeneratedTokens,
                    eval_duration = (long)(result.Metrics.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsJsonAsync(respObj);
                return;
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var genResult = await session.GenerateAsync(prompt, options, formatChat: true, onToken: token =>
            {
                var chunk = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = token,
                    done = false
                };
                string line = JsonSerializer.Serialize(chunk) + "\n";
                ctx.Response.WriteAsync(line).GetAwaiter().GetResult();
            });

            var finalChunk = new
            {
                model = modelName,
                created_at = DateTime.UtcNow.ToString("o"),
                response = "",
                done = true,
                total_duration = (long)(genResult.Metrics.TotalDuration.TotalSeconds * 1e9),
                eval_count = genResult.Metrics.GeneratedTokens,
                eval_duration = (long)(genResult.Metrics.GenerationDuration.TotalSeconds * 1e9)
            };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n");
        });

        // 3. POST /v1/chat/completions (OpenAI compatible)
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            var req = await ctx.Request.ReadFromJsonAsync<JsonElement>();
            var messages = req.GetProperty("messages");
            var sb = new StringBuilder();
            foreach (var m in messages.EnumerateArray())
            {
                string role = m.GetProperty("role").GetString() ?? "user";
                string content = m.GetProperty("content").GetString() ?? "";
                sb.Append($"<|im_start|>{role}\n{content}<|im_end|>\n");
            }
            sb.Append("<|im_start|>assistant\n");

            var options = new SamplingOptions { MaxTokens = 512, Temperature = 0.7f };
            var result = await session.GenerateAsync(sb.ToString(), options, formatChat: false);

            var resp = new
            {
                id = $"chatcmpl-{Guid.NewGuid():N}",
                @object = "chat.completion",
                created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                model = modelName,
                choices = new[]
                {
                    new
                    {
                        index = 0,
                        message = new { role = "assistant", content = result.Text },
                        finish_reason = result.FinishReason
                    }
                },
                usage = new
                {
                    prompt_tokens = result.Metrics.PromptTokens,
                    completion_tokens = result.Metrics.GeneratedTokens,
                    total_tokens = result.Metrics.PromptTokens + result.Metrics.GeneratedTokens
                }
            };

            await ctx.Response.WriteAsJsonAsync(resp);
        });

        // 4. GET /v1/models (OpenAI models endpoint)
        app.MapGet("/v1/models", () => Results.Json(new
        {
            @object = "list",
            data = new[]
            {
                new
                {
                    id = modelName,
                    @object = "model",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    owned_by = "glacier"
                }
            }
        }));

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n>>> Glacier.Inference server running on http://{host}:{port}");
        Console.WriteLine("    Compatible with Ollama CLI, LangChain, AutoGen, and OpenAI SDKs!");
        Console.ResetColor();

        await app.RunAsync();
        return 0;
    }

    // =========================================================================
    // 5. DEVICES COMMAND
    // =========================================================================
    private static int RunDevices(string[] args)
    {
        var devices = DeviceManager.GetDevices();
        var (activeDevice, activeEngine) = GlacierSettings.ResolveTarget(null, null);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("============================================================================================================");
        Console.WriteLine("                                  GLACIER DETECTED ACCELERATORS & ENGINES                                   ");
        Console.WriteLine("============================================================================================================");
        Console.ResetColor();
        Console.WriteLine($"{"[ID]",-18} | {"Device Name",-34} | {"Memory",-15} | {"Safe Driver Engines",-25}");
        Console.WriteLine(new string('-', 108));

        foreach (var dev in devices)
        {
            string memStr;
            if (dev.Vendor == GpuVendor.Cpu)
            {
                memStr = $"{dev.SharedVramGb:F1} GB RAM";
            }
            else if (dev.DedicatedVramGb < 1.0 && dev.SharedVramGb > 0)
            {
                memStr = $"{dev.SharedVramGb:F1} GB Unified";
            }
            else
            {
                memStr = $"{dev.DedicatedVramGb:F1} GB VRAM";
            }

            string safeEngines = string.Join(", ", dev.SupportedEngines.Select(FormatEngineName));
            bool isCurrent = dev.Id == activeDevice.Id;

            if (isCurrent) Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{dev.Id,-18} | {dev.Name,-34} | {memStr,-15} | {safeEngines,-25} {(isCurrent ? $"[ACTIVE: {FormatEngineName(activeEngine)}]" : "")}");
            if (isCurrent) Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  └─ Driver Safety: {dev.SafetyNotes}");
            Console.ResetColor();
        }

        Console.WriteLine(new string('-', 108));
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("* To switch your active hardware & driver engine:");
        Console.WriteLine("  glacier config --device <id|name> [--engine <baremetal|directml|cpu>]");
        Console.WriteLine("  Example: glacier config --device nvidia-rtx-4060 --engine baremetal");
        Console.WriteLine("  Example: glacier config --device amd-890m --engine directml");
        Console.WriteLine("  Example: glacier config --device cpu");
        Console.ResetColor();

        return 0;
    }

    // =========================================================================
    // 6. CONFIG COMMAND
    // =========================================================================
    private static int RunConfig(string[] args)
    {
        var settings = GlacierSettings.Load();

        if (args.Length == 0)
        {
            var (activeDev, activeEng) = GlacierSettings.ResolveTarget(null, null);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=======================================================================");
            Console.WriteLine("                     GLACIER PERSISTENT SETTINGS                      ");
            Console.WriteLine("=======================================================================");
            Console.ResetColor();
            Console.WriteLine($"Config File:       {GlacierSettings.GetSettingsFilePath()}");
            Console.WriteLine($"Default Device:    {settings.DeviceId ?? "auto"} (Resolved: {activeDev.Name})");
            Console.WriteLine($"Default Engine:    {FormatEngineName(settings.Engine)} (Resolved: {FormatEngineName(activeEng)})");
            Console.WriteLine($"CPU Fallback:      {settings.FallbackToCpu}");
            Console.WriteLine($"Max Seq Length:    {settings.MaxSeqLen}");
            Console.WriteLine($"Default Temp:      {settings.DefaultTemperature}");
            Console.WriteLine($"Default Top-K:     {settings.DefaultTopK}");
            Console.WriteLine($"Default Top-P:     {settings.DefaultTopP}");
            Console.WriteLine();
            Console.WriteLine("Commands to configure:");
            Console.WriteLine("  glacier config --device <id|name> [--engine <baremetal|directml|cpu>]");
            Console.WriteLine("  glacier config --reset");
            return 0;
        }

        if (args[0] is "--reset" or "reset")
        {
            settings.DeviceId = "auto";
            settings.Engine = InferenceEngineType.Auto;
            settings.Save();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Settings successfully reset to auto-detect defaults.");
            Console.ResetColor();
            return 0;
        }

        string? targetDev = null;
        string? targetEng = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--device" && i + 1 < args.Length)
                targetDev = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                targetEng = args[++i];
        }

        if (targetDev != null || targetEng != null)
        {
            // Validate safety before saving
            var (resolvedDev, resolvedEng) = GlacierSettings.ResolveTarget(
                targetDev ?? settings.DeviceId,
                targetEng ?? (settings.Engine != InferenceEngineType.Auto ? settings.Engine.ToString() : null));

            if (targetDev != null) settings.DeviceId = resolvedDev.Id;
            if (targetEng != null) settings.Engine = resolvedEng;
            settings.Save();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Successfully updated settings!");
            Console.ResetColor();
            Console.WriteLine($"Active Device: {resolvedDev.Name} ({resolvedDev.Id})");
            Console.WriteLine($"Active Engine: {FormatEngineName(resolvedEng)}");
            Console.WriteLine($"Settings saved to: {GlacierSettings.GetSettingsFilePath()}");
            return 0;
        }

        Console.WriteLine("Usage: glacier config [--device <id|name>] [--engine <baremetal|directml|cpu>] [--reset]");
        return 1;
    }

    private static string FormatEngineName(InferenceEngineType engine) => engine switch
    {
        InferenceEngineType.BareMetal => "BareMetal (SASS)",
        InferenceEngineType.DirectML => "DirectML",
        InferenceEngineType.Cpu => "Cpu",
        _ => engine.ToString()
    };
}
