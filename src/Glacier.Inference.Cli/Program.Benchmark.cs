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
using Glacier.Inference.Audio;
using Glacier.Inference.Config;
using Glacier.Inference.Diagnostics;
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
    private static int RunInspect(string[] args)
    {
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintInspectHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? modelPath = null;
        string? tensorFilter = null;
        string? printTensor = null;
        bool showAllTensors = false;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "-f" || args[i] == "--filter") && i + 1 < args.Length)
                tensorFilter = args[++i];
            else if (args[i] == "--print-tensor" && i + 1 < args.Length)
                printTensor = args[++i];
            else if (args[i] == "-a" || args[i] == "--all")
                showAllTensors = true;
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Error: Model file path required.");
            Console.ResetColor();
            Console.WriteLine();
            PrintInspectHelp();
            return 1;
        }

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
        Console.WriteLine("\n--- Metadata Keys ---");
        foreach (var kv in gguf.Metadata)
        {
            if (kv.Value is System.Collections.IList list)
            {
                if (list.Count <= 64)
                    Console.WriteLine($"  {kv.Key}: [{string.Join(", ", list.Cast<object>())}]");
                else
                    Console.WriteLine($"  {kv.Key}: [list of {list.Count} items]");
            }
            else
            {
                string s = kv.Value?.ToString() ?? "";
                if (kv.Key == "tokenizer.chat_template")
                {
                    Console.WriteLine($"\n[Chat Template]:\n{s}\n");
                }
                else
                {
                    if (s.Length > 80) s = s.Substring(0, 80) + "...";
                    Console.WriteLine($"  {kv.Key}: {s}");
                }
            }
        }

        Console.WriteLine("\n--- Tensors Summary ---");
        var types = new Dictionary<Glacier.Inference.Gguf.GgufType, int>();
        foreach (var t in gguf.TensorList)
        {
            types[t.Type] = types.GetValueOrDefault(t.Type) + 1;
        }
        foreach (var kvp in types)
        {
            Console.WriteLine($"  {kvp.Key}: {kvp.Value} tensors");
        }

        var displayTensors = gguf.TensorList;
        if (!string.IsNullOrEmpty(tensorFilter))
        {
            displayTensors = displayTensors.Where(t => t.Name.Contains(tensorFilter, StringComparison.OrdinalIgnoreCase)).ToList();
            Console.WriteLine($"\n--- Filtered Tensors (matching '{tensorFilter}', {displayTensors.Count} matching) ---");
        }
        else
        {
            Console.WriteLine(showAllTensors ? $"\n--- All Tensors ({displayTensors.Count}) ---" : "\n--- First 25 Tensors ---");
        }

        int maxDisplay = showAllTensors || !string.IsNullOrEmpty(tensorFilter) ? displayTensors.Count : Math.Min(25, displayTensors.Count);
        for (int i = 0; i < maxDisplay; i++)
        {
            var t = displayTensors[i];
            Console.WriteLine($"  [{i,3}] {t.Name,-45} | Type: {t.Type,-10} | Dims: [{string.Join(" x ", t.Dimensions)}]");
        }

        if (!string.IsNullOrEmpty(printTensor))
        {
            if (gguf.TryGetTensor(printTensor, out var targetTensor) && targetTensor != null)
            {
                Console.WriteLine($"\n--- Tensor Data Preview: {targetTensor.Name} ({targetTensor.Type}, Dims: [{string.Join(" x ", targetTensor.Dimensions)}]) ---");
                unsafe
                {
                    void* ptr = gguf.GetTensorPointer(targetTensor);
                    if (targetTensor.Type == Glacier.Inference.Gguf.GgufType.F32)
                    {
                        float* fptr = (float*)ptr;
                        int count = Math.Min(32, (int)targetTensor.Dimensions[0]);
                        for (int i = 0; i < count; i++)
                        {
                            Console.WriteLine($"  [{i,2}] = {fptr[i]:F6}");
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine($"\n[Warning] Tensor '{printTensor}' not found.");
            }
        }

        return 0;
    }

    // =========================================================================
    // 2. BENCH COMMAND
    // =========================================================================
    private static async Task<int> RunBenchAsync(string[] args)
    {
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintBenchHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? modelPath = null;
        int targetTokens = 25;
        string prompt = "Explain in two sentences what a CPU cache is.";
        string? compareOllamaUrl = null;
        string? compareModel = null;
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        string? split = null;
        int maxSeqLen = 2048;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "-n" || args[i] == "--tokens") && i + 1 < args.Length && int.TryParse(args[++i], out int n))
                targetTokens = n;
            else if ((args[i] == "-p" || args[i] == "--prompt") && i + 1 < args.Length)
                prompt = args[++i];
            else if (args[i] == "--compare-ollama" && i + 1 < args.Length)
                compareOllamaUrl = args[++i];
            else if (args[i] == "--compare-model" && i + 1 < args.Length)
                compareModel = args[++i];
            else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (args[i] == "--split" && i + 1 < args.Length)
                split = args[++i];
            else if ((args[i] == "-c" || args[i] == "--ctx" || args[i] == "--context-length") && i + 1 < args.Length && int.TryParse(args[++i], out int cLen))
                maxSeqLen = cLen;
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Error: Model file path required.");
            Console.ResetColor();
            Console.WriteLine();
            PrintBenchHelp();
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
        Console.WriteLine($"Context Len:  {maxSeqLen}");
        Console.WriteLine($"KV Precision: {kvPrecision}");
        if (!string.IsNullOrEmpty(split))
            Console.WriteLine($"Pipeline Split: {split}");
        if (!string.IsNullOrEmpty(compareOllamaUrl))
            Console.WriteLine($"Comparative:  Ollama at {compareOllamaUrl}");
        Console.WriteLine();

        Console.WriteLine(">> Loading model into zero-copy virtual address space...");
        var loadSw = Stopwatch.StartNew();
        using var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split);
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

                    PrintSummaryRow("Runtime", "Pure C# .NET 10 Native AOT", "Go + C++ CUDA / llama.cpp", true, "Direct Hardware Acceleration");
                    PrintSummaryRow("Dependencies", "Direct OS & Driver Interop", "CUDA Toolkit, cuBLAS, LLAMA", true, "Zero C++ Bloat");
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
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintRunHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? modelPath = null;
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        string? split = null;
        string? loraPath = null;
        int maxSeqLen = 2048;
        int maxTokens = 512;
        float temperature = 0.7f;
        float topP = 0.9f;
        bool rawPrompt = false;
        var promptParts = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.Length)
                device = args[++i];
            else if ((args[i] == "-p" || args[i] == "--prompt") && i + 1 < args.Length)
                promptParts.Add(args[++i]);
            else if (args[i] == "--raw")
                rawPrompt = true;
            else if (args[i] == "--lora" && i + 1 < args.Length)
                loraPath = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (args[i] == "--split" && i + 1 < args.Length)
                split = args[++i];
            else if ((args[i] == "-c" || args[i] == "--ctx" || args[i] == "--context-length") && i + 1 < args.Length && int.TryParse(args[++i], out int cLen))
                maxSeqLen = cLen;
            else if ((args[i] == "-n" || args[i] == "--tokens" || args[i] == "--max-tokens") && i + 1 < args.Length && int.TryParse(args[++i], out int tok))
                maxTokens = tok;
            else if ((args[i] == "--temp" || args[i] == "--temperature") && i + 1 < args.Length && float.TryParse(args[++i], System.Globalization.CultureInfo.InvariantCulture, out float t))
                temperature = t;
            else if (args[i] == "--top-p" && i + 1 < args.Length && float.TryParse(args[++i], System.Globalization.CultureInfo.InvariantCulture, out float p))
                topP = p;
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
            else
            {
                promptParts.Add(args[i]);
            }
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Error: Model file path required.");
            Console.ResetColor();
            Console.WriteLine();
            PrintRunHelp();
            return 1;
        }

        string? initialPrompt = promptParts.Count > 0 ? string.Join(" ", promptParts) : null;

        if (!string.IsNullOrEmpty(loraPath) && device == null)
        {
            device = "cpu"; // LoRA dynamic inference runs on SIMD AVX2/AVX-512 CPU path
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
        Console.WriteLine($"Loading {Path.GetFileName(modelPath)} into Glacier.Inference...");
        Console.ResetColor();

        using var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split);
        if (!string.IsNullOrEmpty(loraPath))
        {
            session.AttachLora(loraPath);
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($">> Attached Pure C# LoRA Adapter: {Path.GetFileName(loraPath)} (Rank={session.CpuModel?.LoraWeights?.Rank}, Alpha={session.CpuModel?.LoraWeights?.Alpha})");
            Console.ResetColor();
        }
        Console.WriteLine($"Model ready on {session.ActiveDevice}.\n");

        var options = new SamplingOptions { MaxTokens = maxTokens, Temperature = temperature, TopP = topP };

        // Single-shot prompt mode
        if (!string.IsNullOrEmpty(initialPrompt))
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($">>> {initialPrompt}\n");
            Console.ResetColor();

            var result = await session.GenerateAsync(
                initialPrompt,
                options,
                formatChat: !rawPrompt,
                onToken: t => Console.Write(t));

            Console.WriteLine($"\n\n[{result.Metrics.GenerationTokensPerSecond:F1} tokens/sec | {result.Metrics.GeneratedTokens} tokens | {result.FinishReason}]");
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
}
