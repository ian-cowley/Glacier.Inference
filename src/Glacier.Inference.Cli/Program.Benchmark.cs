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
using Glacier.Media.Audio;
using Glacier.Inference.Config;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Glacier.Media.Vision;
using Glacier.Media.Video;
using Glacier.Media.Image;
using Glacier.Media.Image.Gguf;

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
            string flag = args[i];
            if (flag is "-m" or "--model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                modelPath = val;
            }
            else if (flag is "-f" or "--filter")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                tensorFilter = val;
            }
            else if (flag == "--print-tensor")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                printTensor = val;
            }
            else if (flag is "-a" or "--all")
                showAllTensors = true;
            else if (!flag.StartsWith("-") && modelPath == null)
                modelPath = flag;
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
        string kvGroupRatio = gguf.HeadCountKv > 0 ? $"{gguf.HeadCount / gguf.HeadCountKv}x" : "N/A";
        Console.WriteLine($"Attention KV Heads:     {gguf.HeadCountKv} (Group ratio: {kvGroupRatio})");
        Console.WriteLine($"Head Dimension:         {gguf.HeadDim}");
        Console.WriteLine($"Context Window:         {gguf.ContextLength:N0} tokens");
        Console.WriteLine($"RoPE Base Frequency:    {gguf.RopeFreqBase}");
        Console.WriteLine($"RMSNorm Epsilon:        {gguf.RmsNormEps}");
        Console.WriteLine($"EOS Token ID:           {gguf.EosTokenId}");
        Console.WriteLine($"Total Tensors:          {gguf.TensorCount}");
        Console.WriteLine($"Metadata Key-Values:    {gguf.MetadataKvCount}");
        PrintInspectExtensions(gguf);

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
        string? pleIoStr = null;
        double? hostArenaGib = null;
        bool? tieredMoeOverride = null;
        string? speculativeStr = null;
        string? draftModelPath = null;
        int? draftTokens = null;
        int dflashSteps = new SpeculativeOptions().DFlashDiffusionSteps;
        string? draftVocabStr = "full";
        int moeTailSkip = 0;
        float temperature = 0.7f;
        bool rawPrompt = false;
        int maxSeqLen = 2048;

        for (int i = 0; i < args.Length; i++)
        {
            string flag = args[i];
            if (flag is "-m" or "--model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                modelPath = val;
            }
            else if (flag is "-n" or "--tokens")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int n) || n <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'. Expected a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                targetTokens = n;
            }
            else if (flag is "-p" or "--prompt")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                prompt = val;
            }
            else if (flag == "--compare-ollama")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                compareOllamaUrl = val;
            }
            else if (flag == "--compare-model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                compareModel = val;
            }
            else if (flag is "-d" or "--device")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                device = val;
            }
            else if (flag == "--engine")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                engineStr = val;
            }
            else if (flag == "--kv-precision")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                kvPrecisionStr = val;
            }
            else if (flag.StartsWith("--kv-precision=", StringComparison.Ordinal))
                kvPrecisionStr = flag["--kv-precision=".Length..];
            else if (flag == "--split")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                split = val;
            }
            else if (flag == "--ple-io")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                pleIoStr = val;
            }
            else if (flag == "--host-arena-gib")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!double.TryParse(val, System.Globalization.CultureInfo.InvariantCulture, out double arenaGib))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                hostArenaGib = Math.Max(0.0, arenaGib);
            }
            else if (flag == "--tiered-moe")
                tieredMoeOverride = true;
            else if (flag == "--speculative")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                speculativeStr = val;
            }
            else if (flag == "--draft-model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                draftModelPath = val;
            }
            else if (flag == "--draft-tokens")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int dt))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                draftTokens = Math.Max(0, dt);
            }
            else if (flag == "--dflash-steps")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int dfs))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                dflashSteps = Math.Max(1, dfs);
            }
            else if (flag == "--draft-vocab")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                draftVocabStr = val;
            }
            else if (flag == "--moe-tail-skip")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int mts))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                moeTailSkip = Math.Max(0, mts);
            }
            else if (flag is "--temp" or "--temperature")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!float.TryParse(val, System.Globalization.CultureInfo.InvariantCulture, out float tempVal))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                temperature = tempVal;
            }
            else if (flag == "--raw")
                rawPrompt = true;
            else if (flag is "-c" or "--ctx" or "--context-length")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int cLen) || cLen <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'. Expected a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxSeqLen = cLen;
            }
            else if (!flag.StartsWith("-") && modelPath == null)
                modelPath = flag;
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

        if (!TryParseKvPrecisionArg(kvPrecisionStr, out var kvPrecision, out var kvError))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine(kvError);
            Console.ResetColor();
            return 1;
        }

        PleIoMode pleIoMode = ParsePleIoMode(pleIoStr);

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
        Console.WriteLine($"PLE I/O:      {pleIoMode}");
        Console.WriteLine($"Temperature:  {temperature:F2} (Raw={rawPrompt})");
        if (!string.IsNullOrWhiteSpace(speculativeStr) || !string.IsNullOrWhiteSpace(draftModelPath))
            Console.WriteLine($"Speculative:  mode={speculativeStr ?? "adaptive"}, draftTokens={draftTokens?.ToString() ?? "auto"}, vocab={draftVocabStr ?? "full"}{(moeTailSkip > 0 ? $", tailSkip={moeTailSkip}" : "")}{(!string.IsNullOrWhiteSpace(draftModelPath) ? $", sidecar={Path.GetFileName(draftModelPath)}" : "")}");
        if (!string.IsNullOrEmpty(split))
            Console.WriteLine($"Pipeline Split: {split}");
        if (!string.IsNullOrEmpty(compareOllamaUrl))
            Console.WriteLine($"Comparative:  Ollama at {compareOllamaUrl}");
        Console.WriteLine();

        var hwSnapshot = HardwareRuntimeProfiler.CaptureSnapshot();
        ulong? hostArenaBudgetBytes = hostArenaGib.HasValue ? (ulong)(hostArenaGib.Value * 1024.0 * 1024 * 1024) : null;
        Console.WriteLine($">> Hardware Profile: {HardwareRuntimeProfiler.FormatSummaryLine(hwSnapshot)}");
        Console.WriteLine(">> Loading model into zero-copy virtual address space...");
        var loadSw = Stopwatch.StartNew();
        using var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split, pleIoMode: pleIoMode, hostArenaBudgetBytes: hostArenaBudgetBytes, tieredMoe: tieredMoeOverride);
        loadSw.Stop();
        var hwPolicy = HardwareRuntimeProfiler.RecommendPolicy(hwSnapshot, session.Weights);
        int resolvedDraftTokens = draftTokens ?? hwPolicy.RecommendedMtpDraftTokens;
        session.MoeTailSkipMinRank = moeTailSkip;
        Console.WriteLine($"   Cold load completed in: {loadSw.ElapsedMilliseconds} ms ({loadSw.Elapsed.TotalSeconds:F2} s)");
        Console.WriteLine($"   Execution Device: {session.ActiveDevice}");

        using var draftBundle = CreateDraftBundle(session, speculativeStr, draftModelPath, draftVocabStr, resolvedDraftTokens, dflashSteps);
        Console.WriteLine("\n>> Running Glacier.Inference...");
        var options = CreateSamplingOptions(targetTokens, temperature);
        var (glacierText, _, glacierMetrics, specMetrics) = await ExecuteGenerationAsync(
            session,
            draftBundle.Provider,
            prompt,
            options,
            resolvedDraftTokens,
            formatChat: !rawPrompt,
            onToken: null,
            dflashSteps: dflashSteps).ConfigureAwait(false);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"   Output: \"{glacierText.Trim()}\"");
        Console.ResetColor();
        Console.WriteLine($"   Prompt Rate:     {glacierMetrics.PromptTokensPerSecond:F2} tokens/sec ({glacierMetrics.PromptEvalDuration.TotalMilliseconds:F1} ms)");
        Console.WriteLine($"   Generation Rate: {glacierMetrics.GenerationTokensPerSecond:F2} tokens/sec ({glacierMetrics.GenerationDuration.TotalMilliseconds:F1} ms)");
        Console.WriteLine($"   Total Time:      {glacierMetrics.TotalDuration.TotalSeconds:F2} s");
        PrintSessionTelemetry(session, specMetrics, draftBundle.Adaptive, hwSnapshot, hwPolicy, resolvedDraftTokens);

        if (!string.IsNullOrEmpty(compareOllamaUrl))
        {
            bool isLocalOllama = compareOllamaUrl.Contains("localhost") || compareOllamaUrl.Contains("127.0.0.1");
            string targetOllamaModel = compareModel ?? "qwen2.5:7b-instruct-32k";
            string hardwareLabel = isLocalOllama ? $"Same {session.Device.Name}" : "Remote GPU";

            Console.WriteLine($"\n>> Querying Ollama Benchmark ({compareOllamaUrl} | Model: {targetOllamaModel})...");
            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
                var reqBody = new OllamaBenchRequest
                {
                    Model = targetOllamaModel,
                    Prompt = prompt,
                    Stream = false,
                    Options = new OllamaBenchOptions { NumPredict = targetTokens, Temperature = temperature }
                };

                var oSw = Stopwatch.StartNew();
                var resp = await http.PostAsJsonAsync(
                    $"{compareOllamaUrl.TrimEnd('/')}/api/generate",
                    reqBody,
                    InferenceCliJsonContext.Default.OllamaBenchRequest);
                oSw.Stop();

                if (resp.IsSuccessStatusCode)
                {
                    var oJson = await resp.Content.ReadFromJsonAsync(InferenceCliJsonContext.Default.JsonElement);
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

                    double gGenTps = glacierMetrics.GenerationTokensPerSecond;
                    double gPromptTps = glacierMetrics.PromptTokensPerSecond;
                    double gWallSec = glacierMetrics.TotalDuration.TotalSeconds;
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

                    PrintSummaryRow("Generated Tokens", $"{glacierMetrics.GeneratedTokens}", $"{evalCount}", false);

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
        string? pleIoStr = null;
        double? hostArenaGib = null;
        bool? tieredMoeOverride = null;
        string? speculativeStr = null;
        string? draftModelPath = null;
        int? draftTokens = null;
        int dflashSteps = new SpeculativeOptions().DFlashDiffusionSteps;
        string? draftVocabStr = "full";
        int moeTailSkip = 0;
        string? loraPath = null;
        string? loraName = null;
        int loraCount = 0;
        int maxSeqLen = 2048;
        int maxTokens = 512;
        float temperature = 0.7f;
        float topP = 0.9f;
        bool rawPrompt = false;
        var promptParts = new List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            string flag = args[i];
            if (flag is "-m" or "--model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                modelPath = val;
            }
            else if (flag is "-d" or "--device")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                device = val;
            }
            else if (flag is "-p" or "--prompt")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                promptParts.Add(val);
            }
            else if (flag == "--raw")
                rawPrompt = true;
            else if (flag == "--lora" || flag.StartsWith("--lora=", StringComparison.Ordinal))
            {
                string loraSpec;
                if (flag == "--lora")
                {
                    if (!TryRequireFlagValue(args, ref i, flag, out loraSpec)) return 1;
                }
                else
                {
                    loraSpec = flag["--lora=".Length..];
                }

                if (!TryParseLoraSpec(loraSpec, out var parsedLoraName, out var parsedLoraPath, out var loraError))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine(loraError);
                    Console.ResetColor();
                    return 1;
                }
                loraCount++;
                if (loraCount > 1)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine("Error: 'glacier-ai run' supports attaching a single --lora adapter at a time; use 'glacier-ai serve' to host multiple named adapters.");
                    Console.ResetColor();
                    return 1;
                }
                loraName = parsedLoraName;
                loraPath = parsedLoraPath;
            }
            else if (flag == "--engine")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                engineStr = val;
            }
            else if (flag == "--kv-precision")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                kvPrecisionStr = val;
            }
            else if (flag.StartsWith("--kv-precision=", StringComparison.Ordinal))
                kvPrecisionStr = flag["--kv-precision=".Length..];
            else if (flag == "--split")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                split = val;
            }
            else if (flag == "--ple-io")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                pleIoStr = val;
            }
            else if (flag == "--host-arena-gib")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!double.TryParse(val, System.Globalization.CultureInfo.InvariantCulture, out double arenaGib))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                hostArenaGib = Math.Max(0.0, arenaGib);
            }
            else if (flag == "--tiered-moe")
                tieredMoeOverride = true;
            else if (flag == "--speculative")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                speculativeStr = val;
            }
            else if (flag == "--draft-model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                draftModelPath = val;
            }
            else if (flag == "--draft-tokens")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int dt))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                draftTokens = Math.Max(0, dt);
            }
            else if (flag == "--dflash-steps")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int dfs))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                dflashSteps = Math.Max(1, dfs);
            }
            else if (flag == "--draft-vocab")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                draftVocabStr = val;
            }
            else if (flag == "--moe-tail-skip")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int mts))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                moeTailSkip = Math.Max(0, mts);
            }
            else if (flag is "-c" or "--ctx" or "--context-length")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int cLen) || cLen <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'. Expected a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxSeqLen = cLen;
            }
            else if (flag is "-n" or "--tokens" or "--max-tokens")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int tok) || tok <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'. Expected a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxTokens = tok;
            }
            else if (flag is "--temp" or "--temperature")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!float.TryParse(val, System.Globalization.CultureInfo.InvariantCulture, out float t))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                temperature = t;
            }
            else if (flag == "--top-p")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!float.TryParse(val, System.Globalization.CultureInfo.InvariantCulture, out float p))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    Console.ResetColor();
                    return 1;
                }
                topP = p;
            }
            else if (!flag.StartsWith("-") && modelPath == null)
                modelPath = flag;
            else
            {
                promptParts.Add(flag);
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

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr) && Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
        {
            engine = parsedEngine;
        }

        (device, engine) = ResolveSingleLoraTarget(!string.IsNullOrEmpty(loraPath), device, engine);

        if (!TryParseKvPrecisionArg(kvPrecisionStr, out var kvPrecision, out var kvError))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine(kvError);
            Console.ResetColor();
            return 1;
        }

        PleIoMode pleIoMode = ParsePleIoMode(pleIoStr);

        var hwSnapshot = HardwareRuntimeProfiler.CaptureSnapshot();
        ulong? hostArenaBudgetBytes = hostArenaGib.HasValue ? (ulong)(hostArenaGib.Value * 1024.0 * 1024 * 1024) : null;
        Console.WriteLine($"Hardware Profile: {HardwareRuntimeProfiler.FormatSummaryLine(hwSnapshot)}");
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"Loading {Path.GetFileName(modelPath)} into Glacier.Inference (PLE I/O: {pleIoMode})...");
        Console.ResetColor();

        using var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split, pleIoMode: pleIoMode, hostArenaBudgetBytes: hostArenaBudgetBytes, tieredMoe: tieredMoeOverride);
        if (!string.IsNullOrEmpty(loraPath))
        {
            session.AttachLora(loraPath);
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($">> Attached Pure C# LoRA Adapter: {loraName ?? Path.GetFileName(loraPath)} ({Path.GetFileName(loraPath)}, Rank={session.CpuModel?.LoraWeights?.Rank}, Alpha={session.CpuModel?.LoraWeights?.Alpha})");
            Console.ResetColor();
        }

        var hwPolicy = HardwareRuntimeProfiler.RecommendPolicy(hwSnapshot, session.Weights);
        int resolvedDraftTokens = draftTokens ?? hwPolicy.RecommendedMtpDraftTokens;
        session.MoeTailSkipMinRank = moeTailSkip;

        using var draftBundle = CreateDraftBundle(session, speculativeStr, draftModelPath, draftVocabStr, resolvedDraftTokens, dflashSteps);
        Console.WriteLine($"Model ready on {session.ActiveDevice}.\n");

        var options = CreateSamplingOptions(maxTokens, temperature, topP);

        // Single-shot prompt mode
        if (!string.IsNullOrEmpty(initialPrompt))
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($">>> {initialPrompt}\n");
            Console.ResetColor();

            var (_, finishReason, metrics, specMetrics) = await ExecuteGenerationAsync(
                session,
                draftBundle.Provider,
                initialPrompt,
                options,
                resolvedDraftTokens,
                formatChat: !rawPrompt,
                onToken: t => Console.Write(t),
                dflashSteps: dflashSteps).ConfigureAwait(false);

            Console.WriteLine($"\n\n[{metrics.GenerationTokensPerSecond:F1} tokens/sec | {metrics.GeneratedTokens} tokens | {finishReason}]");
            if (session.PleTable != null || session.ExpertArena != null || specMetrics != null || session.D3D12Model is Glacier.Inference.Gpu.D3D12.Qwen3HybridD3D12Model)
            {
                PrintSessionTelemetry(session, specMetrics, draftBundle.Adaptive, hwSnapshot, hwPolicy, resolvedDraftTokens);
            }
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
            var (_, _, metrics, _) = await ExecuteGenerationAsync(
                session,
                draftBundle.Provider,
                line,
                options,
                resolvedDraftTokens,
                formatChat: !rawPrompt,
                onToken: t => Console.Write(t),
                dflashSteps: dflashSteps).ConfigureAwait(false);
            Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n[{metrics.GenerationTokensPerSecond:F1} tokens/sec | {metrics.GeneratedTokens} tokens]");
            Console.ResetColor();
        }

        return 0;
    }

    private static void PrintInspectExtensions(GgufFile gguf)
    {
        if (gguf.IsMoe)
        {
            Console.WriteLine($"MoE Experts:            {gguf.ExpertCount} (Active: {gguf.ExpertUsedCount}, Expert FFN: {gguf.ExpertFeedForwardLength})");
            int sharedFfn = (int)gguf.GetMetadataUInt32($"{gguf.Architecture}.expert_shared_feed_forward_length",
                gguf.GetMetadataUInt32("expert_shared_feed_forward_length", 0));
            if (sharedFfn > 0)
                Console.WriteLine($"Shared Expert FFN:      {sharedFfn}");
        }

        int hcCount = gguf.HyperConnectionCount;
        if (hcCount == 0 && gguf.TryGetTensor("blk.0.hc_attn_inject.weight", out var hcInj) && hcInj != null && hcInj.Dimensions.Length >= 2)
            hcCount = (int)hcInj.Dimensions[1];
        if (hcCount == 0 && gguf.TryGetTensor("blk.0.hc_attn_norm.weight", out var hcNorm) && hcNorm != null)
        {
            if (hcNorm.Dimensions.Length >= 2)
                hcCount = (int)hcNorm.Dimensions[1];
            else if (gguf.EmbeddingLength > 0 && hcNorm.ElementCount > (ulong)gguf.EmbeddingLength && hcNorm.ElementCount % (ulong)gguf.EmbeddingLength == 0)
                hcCount = (int)(hcNorm.ElementCount / (ulong)gguf.EmbeddingLength);
        }

        int hcLowRank = gguf.HyperConnectionLowRank;
        if (hcLowRank == 0 && gguf.TryGetTensor("blk.0.hc_attn_down.weight", out var hcDown) && hcDown != null && hcDown.Dimensions.Length >= 2)
            hcLowRank = (int)hcDown.Dimensions[1];
        if (hcLowRank == 0 && hcCount > 0 && gguf.TryGetTensor("blk.0.hc_attn_down_inject.weight", out var hcDownInj) && hcDownInj != null && hcDownInj.Dimensions.Length >= 2)
        {
            int nLo = (int)hcDownInj.Dimensions[1] - hcCount;
            if (nLo > 0) hcLowRank = nLo;
        }

        if (hcCount > 0 || gguf.Tensors.ContainsKey("blk.0.hc_attn_norm.weight"))
            Console.WriteLine($"Hyper-Connections:      {hcCount} streams (low-rank: {hcLowRank})");

        if (gguf.AttentionCompressRatios != null || gguf.TensorList.Any(t => t.Name.EndsWith(".indexer.q_proj.weight", StringComparison.Ordinal)))
            Console.WriteLine($"QSA Sparse Indexer:     heads={gguf.IndexerHeadCount}, key_dim={gguf.IndexerKeyLength}, top_k={gguf.IndexerTopK}");

        if (gguf.PleLayers is { Length: > 0 } || gguf.Tensors.ContainsKey("per_layer_token_embd.weight"))
            Console.WriteLine($"PLE N-gram Table:       ngram={gguf.PleNgramSize}, heads/ngram={gguf.PleHeadsPerNgram}, dim={gguf.PleRowDim}, source={gguf.PleTableFilePath ?? "(inline)"}");

        var nextnTensors = gguf.TensorList
            .Where(t => t.Name.EndsWith(".nextn.eh_proj.weight", StringComparison.Ordinal) ||
                        t.Name.EndsWith(".nextn.fc_emb.weight", StringComparison.Ordinal) ||
                        t.Name == "mtp.eh_proj.weight" ||
                        t.Name == "mtp.fc_emb.weight")
            .Select(t => t.Name)
            .ToList();
        if (nextnTensors.Count > 0)
            Console.WriteLine($"NextN / MTP Block:      {string.Join(", ", nextnTensors)}");
    }

    private static PleIoMode ParsePleIoMode(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return PleIoMode.Auto;
        return value.Trim().ToLowerInvariant() switch
        {
            "auto" => PleIoMode.Auto,
            "direct" or "directio" or "direct-io" => PleIoMode.DirectIo,
            "mmap" or "memorymapped" or "memory-mapped" => PleIoMode.MemoryMapped,
            _ => throw new ArgumentException($"Unsupported --ple-io mode '{value}'. Expected 'auto', 'direct', or 'mmap'.", nameof(value))
        };
    }

    private static SamplingOptions CreateSamplingOptions(int maxTokens, float temperature, float topP = 0.9f)
    {
        if (temperature <= Sampler.GreedyTemperatureThreshold)
        {
            return new SamplingOptions
            {
                MaxTokens = maxTokens,
                Temperature = 0f,
                TopP = 1.0f,
                TopK = 1,
                RepetitionPenalty = 1.0f,
                EnableReasoningLoopGuard = false
            };
        }

        return new SamplingOptions
        {
            MaxTokens = maxTokens,
            Temperature = temperature,
            TopP = topP
        };
    }

    private static DraftVocabularySubset? ResolveDraftVocabularySubset(BpeTokenizer tokenizer, string? draftVocabStr)
    {
        if (string.IsNullOrWhiteSpace(draftVocabStr) || draftVocabStr.Equals("full", StringComparison.OrdinalIgnoreCase))
            return null;
        if (draftVocabStr.Equals("lang", StringComparison.OrdinalIgnoreCase))
            return DraftVocabularySubset.FromLanguageFilter(tokenizer);
        return DraftVocabularySubset.LoadBinary(draftVocabStr);
    }

    private sealed class CliDraftBundle : IDisposable
    {
        public IDraftProvider? Provider { get; init; }
        public AdaptiveCompositeDraftProvider? Adaptive { get; init; }
        public MtpDraftProvider? Mtp { get; init; }
        public DFlashDraftProvider? DFlash { get; init; }
        public PromptLookupDraftProvider? Lookup { get; init; }

        public void Dispose()
        {
            Adaptive?.Dispose();
            Mtp?.Dispose();
            DFlash?.Dispose();
            Lookup?.Dispose();
        }
    }

    private static CliDraftBundle CreateDraftBundle(
        InferenceSession session,
        string? speculativeStr,
        string? draftModelPath,
        string? draftVocabStr,
        int maxDraftTokens,
        int dflashSteps = 3)
    {
        string mode = string.IsNullOrWhiteSpace(speculativeStr)
            ? (!string.IsNullOrWhiteSpace(draftModelPath) ? "adaptive" : "off")
            : speculativeStr.Trim().ToLowerInvariant();

        if (mode is "off" or "none")
            return new CliDraftBundle();

        DraftVocabularySubset? subset = ResolveDraftVocabularySubset(session.Tokenizer, draftVocabStr);

        MtpDraftProvider? CreateMtp()
        {
            if (session.HiddenStateSource is not IHiddenStateSource trunk)
            {
                if (!string.IsNullOrWhiteSpace(draftModelPath) || mode == "mtp")
                    throw new InvalidOperationException("MTP speculative decoding requires an active hidden-state runtime (CPU, Direct3D 12, or Vulkan).");
                return null;
            }

            if (!string.IsNullOrWhiteSpace(draftModelPath))
                return MtpDraftProvider.LoadFromSidecar(trunk, session.Weights, session.MaxSeqLen, draftModelPath, subset);

            return MtpDraftProvider.TryCreateFromModel(trunk, session.Weights, session.MaxSeqLen, subset);
        }

        DFlashDraftProvider? CreateDFlash(bool allowWeightFreeFallback)
        {
            var dflashOpts = new DFlashOptions
            {
                BlockSize = Math.Max(1, maxDraftTokens),
                DiffusionSteps = Math.Max(1, dflashSteps)
            };

            if (session.HiddenStateSource is IHiddenStateSource trunk)
            {
                if (!string.IsNullOrWhiteSpace(draftModelPath) && mode == "dflash")
                {
                    return DFlashDraftProvider.LoadFromSidecar(trunk, session.Weights, draftModelPath, dflashOpts, subset);
                }

                DFlashDraftProvider? embedded = DFlashDraftProvider.TryCreateFromModel(trunk, session.Weights, dflashOpts, subset);
                if (embedded != null)
                    return embedded;

                return allowWeightFreeFallback ? DFlashDraftProvider.CreateContextConditioned(trunk, dflashOpts) : null;
            }

            return allowWeightFreeFallback ? new DFlashDraftProvider(dflashOpts) : null;
        }

        switch (mode)
        {
            case "lookup":
            {
                var lookup = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);
                return new CliDraftBundle { Provider = lookup, Lookup = lookup };
            }
            case "mtp":
            {
                var mtp = CreateMtp() ?? throw new InvalidOperationException("No embedded NextN/MTP block or --draft-model sidecar available for --speculative mtp.");
                return new CliDraftBundle { Provider = mtp, Mtp = mtp };
            }
            case "dflash":
            {
                var dflash = CreateDFlash(allowWeightFreeFallback: true)!;
                return new CliDraftBundle { Provider = dflash, DFlash = dflash };
            }
            case "adaptive":
            {
                var lookup = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);
                var mtp = CreateMtp();
                var dflash = CreateDFlash(allowWeightFreeFallback: true);
                var policy = new CostGatedDraftPolicy(
                    maxMtpDraftTokens: Math.Max(1, maxDraftTokens),
                    maxLookupDraftTokens: Math.Max(2, maxDraftTokens))
                {
                    MaxDFlashDraftTokens = Math.Max(1, maxDraftTokens)
                };
                var adaptive = new AdaptiveCompositeDraftProvider(lookup, mtp, policy, ownsProviders: false, dflashProvider: dflash);
                return new CliDraftBundle { Provider = adaptive, Adaptive = adaptive, Mtp = mtp, DFlash = dflash, Lookup = lookup };
            }
            default:
                throw new ArgumentException($"Unsupported --speculative mode '{speculativeStr}'. Expected off, lookup, mtp, dflash, or adaptive.", nameof(speculativeStr));
        }
    }

    private static async Task<(string Text, string FinishReason, GenerationMetrics Metrics, SpeculativeMetrics? SpecMetrics)> ExecuteGenerationAsync(
        InferenceSession session,
        IDraftProvider? draftProvider,
        string prompt,
        SamplingOptions samplingOptions,
        int maxDraftTokens,
        bool formatChat,
        Action<string>? onToken,
        int dflashSteps = 3)
    {
        if (draftProvider != null)
        {
            using var specEngine = new SpeculativeEngine(session, draftProvider);
            var specResult = await specEngine.GenerateAsync(
                prompt,
                new SpeculativeOptions
                {
                    MaxTokens = samplingOptions.MaxTokens,
                    MaxDraftTokens = maxDraftTokens,
                    MinNgramMatch = 2,
                    MaxNgramMatch = 4,
                    Sampling = samplingOptions,
                    DFlashDiffusionSteps = Math.Max(1, dflashSteps)
                },
                formatChat: formatChat,
                onToken: onToken).ConfigureAwait(false);
            return (specResult.Text, specResult.FinishReason, specResult.Metrics, specResult.SpeculativeMetrics);
        }

        var result = await session.GenerateAsync(prompt, samplingOptions, formatChat: formatChat, onToken: onToken).ConfigureAwait(false);
        return (result.Text, result.FinishReason, result.Metrics, null);
    }

    private static void PrintSessionTelemetry(
        InferenceSession session,
        SpeculativeMetrics? specMetrics,
        AdaptiveCompositeDraftProvider? adaptive,
        HardwareRuntimeSnapshot? hwSnapshot = null,
        HardwareExecutionPolicy? hwPolicy = null,
        int? resolvedDraftTokens = null)
    {
        if (hwSnapshot != null)
        {
            string tailTag = session.MoeTailSkipMinRank > 0 ? $" | MoeTailSkipRank={session.MoeTailSkipMinRank}" : "";
            string draftTag = resolvedDraftTokens.HasValue ? $" | MtpDraft={resolvedDraftTokens.Value}" : "";
            string recTag = hwPolicy != null ? $" (Rec={hwPolicy.RecommendedMtpDraftTokens})" : "";
            Console.WriteLine($"   Hardware Profile: {HardwareRuntimeProfiler.FormatSummaryLine(hwSnapshot)}{draftTag}{recTag}{tailTag}");
        }

        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        double workingSetMb = proc.WorkingSet64 / (1024.0 * 1024.0);
        double privateMb = proc.PrivateMemorySize64 / (1024.0 * 1024.0);
        double virtualGb = proc.VirtualMemorySize64 / (1024.0 * 1024.0 * 1024.0);
        Console.WriteLine($"   Process Memory:  WorkingSet={workingSetMb:F1} MB | PrivateBytes={privateMb:F1} MB | Virtual={virtualGb:F2} GB");

        if (session.ExpertArena != null)
        {
            var arena = session.ExpertArena;
            double arenaGiB = arena.AllocatedBytes / (1024.0 * 1024.0 * 1024.0);
            Console.WriteLine($"   Host Expert Arena: {arena.ResidentMoeLayers}/{arena.TotalMoeLayers} MoE layers in RAM ({arenaGiB:F2} GiB, {arena.LoadBandwidthGBps:F1} GB/s, Unbuffered={arena.UsedUnbufferedIo})");
        }

        if (session.D3D12Model is Glacier.Inference.Gpu.D3D12.Qwen3HybridD3D12Model qhD3d && qhD3d.TieredMoe)
        {
            Console.WriteLine($"   D3D12 Hosted MoE:  Slots={qhD3d.ActiveHostedMoeSlots}/{qhD3d.HostedMoeSlotCapacity} | GPU Hits={qhD3d.GpuExpertHitCount:N0} | PCIe Xfers={qhD3d.GpuExpertTransferCount:N0} | CPU Overflow={qhD3d.CpuExpertOverflowCount:N0} | GPU Ratio={qhD3d.GpuExpertExecutionRatio:P1}");
        }

        if (session.PleTable != null)
        {
            var ple = session.PleTable;
            Console.WriteLine($"   PLE Table I/O:   Mode={ple.IoMode} | Unbuffered={ple.IsUnbufferedHandleActive} | CacheHits={ple.CacheHits:N0} | CacheMisses={ple.CacheMisses:N0} | PhysicalPages={ple.PhysicalPagesIssued:N0}");
        }

        if (specMetrics != null)
        {
            Console.WriteLine($"   Speculative:     Steps={specMetrics.SpeculativeSteps} | Proposed={specMetrics.DraftTokensProposed} | Accepted={specMetrics.DraftTokensAccepted} ({specMetrics.AcceptanceRate:P1}) | AvgTok/Step={specMetrics.AverageTokensPerStep:F2} | Speedup={specMetrics.EstimatedSpeedup:F2}x | SerialFallback={specMetrics.SerialFallbackSteps}");
        }

        if (adaptive != null)
        {
            Console.WriteLine($"   Adaptive Routes: PromptLookupRounds={adaptive.PromptLookupRounds} | MtpRounds={adaptive.MtpRounds} | DFlashRounds={adaptive.DFlashRounds} | SerialBypassRounds={adaptive.SerialBypassRounds}");
        }
    }

    // =========================================================================
    // 4. SERVE COMMAND (OLLAMA & OPENAI COMPATIBLE HTTP SERVER)
    // =========================================================================
}
