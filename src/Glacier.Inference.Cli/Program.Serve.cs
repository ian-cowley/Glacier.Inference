namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Glacier.Inference.Config;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Serve.Inference.Batching;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Server;

public static partial class Program
{
    private static readonly ConditionalWeakTable<InferenceSession, SemaphoreSlim> s_sessionLocks = new();

    private static async Task<int> RunServeAsync(string[] args)
    {
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintServeHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? modelPath = null;
        int port = 11434;
        string host = "127.0.0.1";
        string? apiKey = Environment.GetEnvironmentVariable("GLACIER_API_KEY");
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        string? split = null;
        int maxSeqLen = 2048;
        int defaultMaxThinkTokens = ReasoningBudgetPolicy.AutoBudgetSentinel;
        bool enableSleepMemory = false;
        int sleepBudget = 400;
        string? sleepCheckpointPath = null;
        int? totalBlocks = null;
        int? maxBatch = null;
        var loraSpecs = new List<(string Name, string Path)>();
        var seenLoraNames = new HashSet<string>(StringComparer.Ordinal);

        for (int i = 0; i < args.Length; i++)
        {
            string flag = args[i];
            if (flag is "-m" or "--model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                modelPath = val;
            }
            else if (flag == "--port")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int p) || p < 1 || p > 65535)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error: --port must be a valid port number between 1 and 65535.");
                    Console.ResetColor();
                    return 1;
                }
                port = p;
            }
            else if (flag == "--host")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                host = val;
            }
            else if (flag == "--api-key")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("WARNING: Specifying --api-key on the command line is deprecated for security reasons.");
                Console.WriteLine("         Set the GLACIER_API_KEY environment variable instead.");
                Console.ResetColor();
                apiKey = val;
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

                if (!TryParseLoraSpec(loraSpec, out var loraName, out var loraPath, out var loraError))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine(loraError);
                    Console.ResetColor();
                    return 1;
                }
                if (!seenLoraNames.Add(loraName))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Duplicate --lora adapter name '{loraName}'. Use '<name>=<path>' to assign distinct names.");
                    Console.ResetColor();
                    return 1;
                }
                loraSpecs.Add((loraName, loraPath));
            }
            else if (flag is "--blocks" or "--kv-blocks")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int blkVal) || blkVal <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine("Error: --blocks must be a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                totalBlocks = blkVal;
            }
            else if (flag is "-b" or "--batch-size")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int bsVal) || bsVal <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine("Error: --batch-size must be a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxBatch = bsVal;
            }
            else if (flag == "--split")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                split = val;
            }
            else if (flag == "--max-think-tokens" || flag.StartsWith("--max-think-tokens=", StringComparison.Ordinal))
            {
                string val;
                if (flag == "--max-think-tokens")
                {
                    if (!TryRequireFlagValue(args, ref i, flag, out val)) return 1;
                }
                else
                {
                    val = flag["--max-think-tokens=".Length..];
                }

                if (!TryParseCliThinkBudget(val, out int parsedThink))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine("Error: --max-think-tokens must be 'auto', 'off', '-1', '0', or a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                defaultMaxThinkTokens = parsedThink;
            }
            else if (flag == "--sleep-memory")
                enableSleepMemory = true;
            else if (flag == "--sleep-budget")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int sbVal) || sbVal <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error: --sleep-budget must be a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                sleepBudget = sbVal;
                enableSleepMemory = true;
            }
            else if (flag == "--sleep-checkpoint" || flag.StartsWith("--sleep-checkpoint=", StringComparison.Ordinal))
            {
                if (flag == "--sleep-checkpoint")
                {
                    if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                    sleepCheckpointPath = val;
                }
                else
                {
                    sleepCheckpointPath = flag["--sleep-checkpoint=".Length..];
                }
            }
            else if (flag is "-c" or "--ctx" or "--context-length")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int cLen) || cLen <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error: Context length must be a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxSeqLen = cLen;
            }
            else if (!flag.StartsWith("-", StringComparison.Ordinal) && modelPath == null)
            {
                modelPath = flag;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine($"Error: Unknown option or unexpected argument '{flag}'.");
                Console.ResetColor();
                return 1;
            }
        }

        if (string.IsNullOrEmpty(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Error: Model file path required.");
            Console.ResetColor();
            Console.WriteLine();
            PrintServeHelp();
            return 1;
        }

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr))
        {
            if (!Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Error: Unknown engine '{engineStr}'. Valid values: Auto, Cpu, BareMetal, Vulkan, DirectML.");
                Console.ResetColor();
                return 1;
            }
            engine = parsedEngine;
        }

        if (!TryParseKvPrecisionArg(kvPrecisionStr, out var kvPrecision, out var kvError))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine(kvError);
            Console.ResetColor();
            return 1;
        }

        if (!File.Exists(modelPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine($"Error: Model file not found: '{modelPath}'.");
            Console.ResetColor();
            return 1;
        }

        if ((host == "0.0.0.0" || host == "::") && string.IsNullOrEmpty(apiKey))
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("╔═════════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║ ⚠️  SECURITY NOTICE: SERVER IS BOUND TO ALL NETWORK INTERFACES WITHOUT  ║");
            Console.WriteLine("║     AUTHENTICATION! ANYONE WITH NETWORK ACCESS CAN RUN INFERENCE.       ║");
            Console.WriteLine("║     TO PROTECT THIS ENDPOINT: Provide --api-key or set GLACIER_API_KEY.  ║");
            Console.WriteLine("╚═════════════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("=======================================================================");
        Console.WriteLine("            GLACIER.INFERENCE HIGH-PERFORMANCE HTTP SERVER            ");
        Console.WriteLine("        Ollama & OpenAI Compatible | Sub-100ms Cold Start | Pure C#    ");
        Console.WriteLine("=======================================================================");
        Console.ResetColor();
        Console.WriteLine($"Model:        {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Endpoint:     http://{host}:{port}");
        Console.WriteLine($"Context Len:  {maxSeqLen}");
        Console.WriteLine($"KV Precision: {kvPrecision}");
        if (!string.IsNullOrEmpty(split))
            Console.WriteLine($"Pipeline Split: {split}");
        Console.WriteLine($"Think Budget: {(defaultMaxThinkTokens < 0 ? "auto" : defaultMaxThinkTokens == 0 ? "off" : $"{defaultMaxThinkTokens} tokens")}");
        if (enableSleepMemory)
            Console.WriteLine($"Sleep Memory: Enabled (budget={sleepBudget} tokens{(string.IsNullOrWhiteSpace(sleepCheckpointPath) ? "" : $", checkpoint={sleepCheckpointPath}")})");
        if (loraSpecs.Count > 0)
            Console.WriteLine($"LoRA Adapters:{string.Join(", ", loraSpecs.Select(l => l.Name))}");
        Console.WriteLine();

        if (CanUseContinuousBatchEngine(modelPath, apiKey, split, device, engine))
        {
            Console.WriteLine(">> Initializing Glacier.Serve ContinuousBatchEngine...");
            string modelName = Path.GetFileNameWithoutExtension(modelPath);
            int resolvedBlocks = totalBlocks ?? Math.Max(1024, ((maxSeqLen + 15) / 16) * 2);
            int resolvedBatch = maxBatch ?? 16;
            bool useGpu = !string.Equals(device, "cpu", StringComparison.OrdinalIgnoreCase) && engine != InferenceEngineType.Cpu;

            await using Glacier.Serve.Daemons.SleepMemoryService? batchSleepMemory = enableSleepMemory
                ? new Glacier.Serve.Daemons.SleepMemoryService(sleepBudget)
                : null;
            if (batchSleepMemory != null && !string.IsNullOrWhiteSpace(sleepCheckpointPath))
            {
                string ckptPath = sleepCheckpointPath;
                batchSleepMemory.Daemon.ConsolidationCompleted += _ => batchSleepMemory.ExpansionBank.Save(ckptPath);
            }

            using var batchEngine = new ContinuousBatchEngine(
                modelPath,
                totalKvBlocks: resolvedBlocks,
                maxBatchSize: resolvedBatch,
                maxSeqLen: maxSeqLen,
                useGpu: useGpu,
                defaultMaxReasoningTokens: defaultMaxThinkTokens,
                kvCachePrecision: kvPrecision);

            foreach (var (loraName, loraPath) in loraSpecs)
            {
                batchEngine.LoadLoraAdapter(loraName, loraPath);
            }

            if (batchSleepMemory != null)
            {
                batchEngine.AttachSleepMemory(batchSleepMemory);
            }

            Console.WriteLine($"Backend:      {batchEngine.ActiveBackend}");
            Console.WriteLine($"Paged KV:     {batchEngine.KvPrecision.ToDisplayName()} ({resolvedBlocks} blocks, maxBatch={resolvedBatch})");
            var loadedAdapters = batchEngine.GetLoadedLoraAdapters();
            if (loadedAdapters.Count > 0)
            {
                Console.WriteLine($"Loaded LoRA:  {string.Join(", ", loadedAdapters)}");
            }

            var serveBuilder = GlacierServeApp.CreateBuilder()
                .UseHost(host)
                .UsePort(port);
            if (!string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("API Auth:     Enabled (Bearer)");
                serveBuilder.UseBearerAuth(apiKey);
            }
            using var serveApp = serveBuilder.Build();
            serveApp.MapInference(batchEngine, modelName, batchSleepMemory);

            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            await serveApp.StartAsync(cts.Token);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n>>> Glacier.Serve continuous-batching server running on http://{host}:{port}");
            Console.WriteLine("    Compatible with Ollama CLI, LangChain, AutoGen, and OpenAI SDKs!");
            Console.ResetColor();

            try
            {
                await Task.Delay(Timeout.Infinite, cts.Token);
            }
            catch (TaskCanceledException)
            {
            }

            return 0;
        }

        if (loraSpecs.Count > 1)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Error.WriteLine("Error: Multiple --lora adapters require dense CPU/CUDA ContinuousBatchEngine (omit --api-key, --split, and DirectML/Vulkan).");
            Console.ResetColor();
            return 1;
        }

        string? attachedLoraName = null;
        if (loraSpecs.Count == 1)
        {
            attachedLoraName = loraSpecs[0].Name;
            (device, engine) = ResolveSingleLoraTarget(hasLora: true, device, engine);
        }

        Console.WriteLine(">> Initializing inference session...");
        await using Glacier.Serve.Daemons.SleepMemoryService? sleepMemory = enableSleepMemory
            ? new Glacier.Serve.Daemons.SleepMemoryService(sleepBudget)
            : null;
        if (sleepMemory != null && !string.IsNullOrWhiteSpace(sleepCheckpointPath))
        {
            string ckptPath = sleepCheckpointPath;
            sleepMemory.Daemon.ConsolidationCompleted += _ => sleepMemory.ExpansionBank.Save(ckptPath);
        }

        using var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split);
        if (loraSpecs.Count == 1)
        {
            session.AttachLora(loraSpecs[0].Path);
        }

        await using var app = CreateServeApp(session, modelPath, apiKey, host, port, defaultMaxThinkTokens, sleepMemory, attachedLoraName);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n>>> Glacier.Inference server running on http://{host}:{port}");
        Console.WriteLine("    Compatible with Ollama CLI, LangChain, AutoGen, and OpenAI SDKs!");
        Console.ResetColor();

        await app.RunAsync();
        return 0;
    }

    internal static bool FixedTimeTokenEquals(string provided, string expected)
    {
        byte[] a = SHA256.HashData(Encoding.UTF8.GetBytes(provided));
        byte[] b = SHA256.HashData(Encoding.UTF8.GetBytes(expected));
        return CryptographicOperations.FixedTimeEquals(a, b);
    }

    internal static bool CanUseContinuousBatchEngine(
        string modelPath,
        string? apiKey,
        string? split,
        string? device,
        InferenceEngineType engine)
    {
        _ = apiKey;
        if (!string.IsNullOrEmpty(split))
            return false;

        if (engine is not (InferenceEngineType.Auto or InferenceEngineType.BareMetal or InferenceEngineType.Cpu))
            return false;

        bool deviceSupported = string.IsNullOrWhiteSpace(device)
            || device.Equals("auto", StringComparison.OrdinalIgnoreCase)
            || device.Equals("cpu", StringComparison.OrdinalIgnoreCase)
            || device.Equals("gpu", StringComparison.OrdinalIgnoreCase)
            || device.Contains("nvidia", StringComparison.OrdinalIgnoreCase);
        if (!deviceSupported)
            return false;

        if (!File.Exists(modelPath))
            return false;

        using var gguf = GgufFile.Open(modelPath);
        if (gguf.IsMoe || gguf.IsMla || gguf.IsHybridSsm)
            return false;

        if (gguf.Architecture is not ("llama" or "qwen2" or "qwen3" or "mistral" or "codellama"))
            return false;

        if (gguf.Tensors.ContainsKey("blk.0.ssm_in.weight")
            || gguf.Tensors.ContainsKey("blk.0.attn_qkv.weight")
            || gguf.Tensors.ContainsKey("blk.0.ffn_gate_up.weight")
            || gguf.Tensors.ContainsKey("blk.0.attn_sinks")
            || gguf.Tensors.ContainsKey("blk.0.hc_attn_norm.weight"))
            return false;

        return true;
    }

    public static WebApplication CreateServeApp(
        InferenceSession session,
        string modelPath,
        string? apiKey = null,
        string host = "127.0.0.1",
        int port = 11434,
        int defaultMaxThinkTokens = ReasoningBudgetPolicy.AutoBudgetSentinel,
        Glacier.Serve.Daemons.SleepMemoryService? sleepMemory = null,
        string? attachedLoraName = null)
    {
        var appBuilder = WebApplication.CreateBuilder();
        appBuilder.Services.Configure<Microsoft.AspNetCore.Server.Kestrel.Core.KestrelServerOptions>(options =>
        {
            options.Limits.MaxRequestBodySize = 32 * 1024 * 1024; // 32 MB request body limit
        });
        var app = appBuilder.Build();
        if (port >= 0)
        {
            app.Urls.Add($"http://{host}:{port}");
        }

        ConfigureServeEndpoints(app, session, modelPath, apiKey, defaultMaxThinkTokens, sleepMemory, attachedLoraName);
        return app;
    }

    public static void ConfigureServeEndpoints(
        WebApplication app,
        InferenceSession session,
        string modelPath,
        string? apiKey = null,
        int defaultMaxThinkTokens = ReasoningBudgetPolicy.AutoBudgetSentinel,
        Glacier.Serve.Daemons.SleepMemoryService? sleepMemory = null,
        string? attachedLoraName = null)
    {
        var inferenceLock = s_sessionLocks.GetValue(session, static _ => new SemaphoreSlim(1, 1));
        string modelName = Path.GetFileNameWithoutExtension(modelPath);
        LoraAdapterWeights? configuredLoraWeights = null;
        if (!string.IsNullOrWhiteSpace(attachedLoraName) && session.CpuModel != null)
        {
            configuredLoraWeights = session.CpuModel.LoraWeights;
            session.CpuModel.LoraWeights = null;
        }

        // Calculate actual parameter count and quantization level from tensors
        ulong totalParams = 0;
        foreach (var t in session.Gguf.TensorList)
        {
            totalParams += t.ElementCount;
        }
        string paramSize = totalParams >= 1_000_000_000
            ? $"{totalParams / 1_000_000_000.0:F1}B"
            : totalParams >= 1_000_000
                ? $"{totalParams / 1_000_000.0:F1}M"
                : $"{totalParams}";

        string quantLevel = "F16";
        if (session.Gguf.TensorList.Count > 0)
        {
            var dominant = session.Gguf.TensorList
                .GroupBy(t => t.Type)
                .OrderByDescending(g => g.Count())
                .FirstOrDefault()?.Key;
            if (dominant.HasValue)
                quantLevel = dominant.Value.ToString();
        }

        if (!string.IsNullOrEmpty(apiKey))
        {
            Console.WriteLine("API Auth:     Enabled (Bearer / x-api-key)");
            app.Use(async (context, next) =>
            {
                string? authHeader = context.Request.Headers.Authorization.FirstOrDefault();
                string? xApiKey = context.Request.Headers["x-api-key"].FirstOrDefault();

                bool authorized = false;
                if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                {
                    string token = authHeader.Substring(7).Trim();
                    authorized = FixedTimeTokenEquals(token, apiKey);
                }
                else if (!string.IsNullOrEmpty(xApiKey))
                {
                    authorized = FixedTimeTokenEquals(xApiKey, apiKey);
                }

                if (!authorized)
                {
                    context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                    await context.Response.WriteAsync("Unauthorized: Invalid or missing API key.", context.RequestAborted);
                    return;
                }

                await next();
            });
        }

        // 1. GET /api/tags (Ollama compatible tags list)
        app.MapGet("/api/tags", async (HttpContext ctx) =>
        {
            long size = new FileInfo(modelPath).Length;
            string modifiedAt = DateTime.UtcNow.ToString("o");
            var details = new OllamaModelDetails
            {
                Format = "gguf",
                Family = session.Gguf.Architecture,
                ParameterSize = paramSize,
                QuantizationLevel = quantLevel
            };

            var models = new List<OllamaModelEntry>(2)
            {
                new OllamaModelEntry
                {
                    Name = modelName,
                    Model = modelName,
                    ModifiedAt = modifiedAt,
                    Size = size,
                    Details = details
                }
            };
            if (!string.IsNullOrWhiteSpace(attachedLoraName))
            {
                string loraModel = $"{modelName}:{attachedLoraName}";
                models.Add(new OllamaModelEntry
                {
                    Name = loraModel,
                    Model = loraModel,
                    ModifiedAt = modifiedAt,
                    Size = size,
                    Details = details
                });
            }

            await ctx.Response.WriteAsJsonAsync(
                new OllamaTagsResponse { Models = models.ToArray() },
                InferenceCliJsonContext.Default.OllamaTagsResponse,
                cancellationToken: ctx.RequestAborted);
        });

        // 2. POST /api/generate (Ollama streaming & non-streaming)
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync(InferenceCliJsonContext.Default.JsonElement, ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }
            catch (BadHttpRequestException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Bad request or body size limit exceeded.", ctx.RequestAborted);
                return;
            }

            string prompt = req.TryGetProperty("prompt", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString() ?? "" : "";
            string rawPromptForSleep = prompt;
            if (sleepMemory != null)
            {
                prompt = await sleepMemory.ConsolidatePromptAsync(prompt, ctx.RequestAborted);
            }
            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            string? reqModelGen = req.TryGetProperty("model", out var mgProp) && mgProp.ValueKind == JsonValueKind.String
                ? mgProp.GetString()?.Trim()
                : null;
            string? rawReqLoraGen = req.TryGetProperty("lora", out var lgProp) && lgProp.ValueKind == JsonValueKind.String
                ? lgProp.GetString()
                : null;
            var (loraErrStatusGen, loraErrTypeGen, loraErrMsgGen, selectedLoraGen) =
                ResolvePathBLoraRequest(reqModelGen, rawReqLoraGen, modelName, attachedLoraName);
            if (loraErrStatusGen.HasValue)
            {
                await WritePathBApiErrorAsync(ctx, loraErrStatusGen.Value, loraErrMsgGen!, loraErrTypeGen!);
                return;
            }

            if (!TryParseSamplingOptionsInternal(req, defaultMaxThinkTokens, out var options, out _, out string? budgetErr))
            {
                await WritePathBApiErrorAsync(ctx, StatusCodes.Status400BadRequest, budgetErr!, "invalid_request_error");
                return;
            }

            await inferenceLock.WaitAsync(ctx.RequestAborted);
            CancellationTokenSource? genCts = null;
            Task<GenerationResult>? genTask = null;
            try
            {
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = selectedLoraGen != null ? configuredLoraWeights : null;
                }
                if (!stream)
                {
                    var result = await session.GenerateAsync(prompt, options, formatChat: true, ct: ctx.RequestAborted);
                    sleepMemory?.RecordTurnCompletion(rawPromptForSleep, result.Text);
                    var respObj = new OllamaGenerateResponse
                    {
                        Model = modelName,
                        CreatedAt = DateTime.UtcNow.ToString("o"),
                        Response = result.Text,
                        Done = true,
                        TotalDuration = (long)(result.Metrics.TotalDuration.TotalSeconds * 1e9),
                        PromptEvalCount = result.Metrics.PromptTokens,
                        PromptEvalDuration = (long)(result.Metrics.PromptEvalDuration.TotalSeconds * 1e9),
                        EvalCount = result.Metrics.GeneratedTokens,
                        EvalDuration = (long)(result.Metrics.GenerationDuration.TotalSeconds * 1e9)
                    };
                    await ctx.Response.WriteAsJsonAsync(respObj, InferenceCliJsonContext.Default.OllamaGenerateResponse, cancellationToken: ctx.RequestAborted);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson";
                var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(128)
                {
                    SingleWriter = true,
                    SingleReader = true,
                    FullMode = BoundedChannelFullMode.Wait
                });
                genCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);
                CancellationToken genToken = genCts.Token;
                genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            prompt,
                            options,
                            formatChat: true,
                            onTokenAsync: token => channel.Writer.WriteAsync(token, genToken),
                            ct: genToken);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                await foreach (var token in channel.Reader.ReadAllAsync(genToken))
                {
                    var chunk = new OllamaGenerateResponse
                    {
                        Model = modelName,
                        CreatedAt = DateTime.UtcNow.ToString("o"),
                        Response = token,
                        Done = false
                    };
                    await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk, InferenceCliJsonContext.Default.OllamaGenerateResponse) + "\n", genToken);
                    await ctx.Response.Body.FlushAsync(genToken);
                }

                var genResult = await genTask;
                sleepMemory?.RecordTurnCompletion(rawPromptForSleep, genResult.Text);
                var finalChunk = new OllamaGenerateResponse
                {
                    Model = modelName,
                    CreatedAt = DateTime.UtcNow.ToString("o"),
                    Response = "",
                    Done = true,
                    TotalDuration = (long)(genResult.Metrics.TotalDuration.TotalSeconds * 1e9),
                    EvalCount = genResult.Metrics.GeneratedTokens,
                    EvalDuration = (long)(genResult.Metrics.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk, InferenceCliJsonContext.Default.OllamaGenerateResponse) + "\n", genToken);
                await ctx.Response.Body.FlushAsync(genToken);
            }
            finally
            {
                if (genCts != null)
                {
                    try { genCts.Cancel(); }
                    catch (ObjectDisposedException) { }
                }
                if (genTask != null)
                {
                    try
                    {
                        await genTask.ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                    }
                    catch (ChannelClosedException)
                    {
                    }
                    catch (Exception) when (genCts?.IsCancellationRequested == true)
                    {
                    }
                }
                genCts?.Dispose();
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = null;
                }
                inferenceLock.Release();
            }
        });

        // 3. POST /v1/chat/completions (OpenAI compatible)
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync(InferenceCliJsonContext.Default.JsonElement, ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }
            catch (BadHttpRequestException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Bad request or body size limit exceeded.", ctx.RequestAborted);
                return;
            }

            var messages = ParseMessages(req);
            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            string? reqModel = req.TryGetProperty("model", out var mProp) && mProp.ValueKind == JsonValueKind.String
                ? mProp.GetString()?.Trim()
                : null;
            string? rawReqLora = req.TryGetProperty("lora", out var lProp) && lProp.ValueKind == JsonValueKind.String
                ? lProp.GetString()
                : null;

            var (loraErrStatus, loraErrType, loraErrMsg, selectedLora) =
                ResolvePathBLoraRequest(reqModel, rawReqLora, modelName, attachedLoraName);
            if (loraErrStatus.HasValue)
            {
                await WritePathBApiErrorAsync(ctx, loraErrStatus.Value, loraErrMsg!, loraErrType!);
                return;
            }

            string responseModel = !string.IsNullOrWhiteSpace(reqModel)
                ? reqModel
                : (!string.IsNullOrWhiteSpace(selectedLora) ? $"{modelName}:{selectedLora}" : modelName);

            bool stream = req.TryGetProperty("stream", out var s) && (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));
            bool includeUsage = req.TryGetProperty("stream_options", out var soProp)
                && soProp.ValueKind == JsonValueKind.Object
                && soProp.TryGetProperty("include_usage", out var iuProp)
                && iuProp.ValueKind == JsonValueKind.True;
            if (!TryParseSamplingOptionsInternal(req, defaultMaxThinkTokens, out var options, out bool? explicitEnableThinking, out string? budgetErr))
            {
                await WritePathBApiErrorAsync(ctx, StatusCodes.Status400BadRequest, budgetErr!, "invalid_request_error");
                return;
            }
            string rawLastUser = ExtractLastUserMessage(messages);
            if (sleepMemory != null)
            {
                messages = await sleepMemory.ConsolidateMessagesAsync(messages, ctx.RequestAborted);
            }
            bool hasThinkToken = session.Tokenizer.TryGetTokenId("<think>", out _);
            bool thinkingMode = explicitEnableThinking ?? hasThinkToken;
            string formattedPrompt = session.Tokenizer.FormatConversation(messages, thinkingMode: thinkingMode);
            string completionId = $"chatcmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            await inferenceLock.WaitAsync(ctx.RequestAborted);
            CancellationTokenSource? genCts = null;
            Task<GenerationResult>? genTask = null;
            try
            {
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = selectedLora != null ? configuredLoraWeights : null;
                }
                if (!stream)
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, formatChat: false, ct: ctx.RequestAborted);
                    sleepMemory?.RecordTurnCompletion(rawLastUser, result.Text);
                    var resp = new OpenAiChatCompletionResponse
                    {
                        Id = completionId,
                        Created = createdTimestamp,
                        Model = responseModel,
                        Choices =
                        [
                            new OpenAiChatChoice
                            {
                                Index = 0,
                                Message = new OllamaChatMessageDto { Role = "assistant", Content = result.Text },
                                FinishReason = result.FinishReason
                            }
                        ],
                        Usage = new OpenAiUsage
                        {
                            PromptTokens = result.Metrics.PromptTokens,
                            CompletionTokens = result.Metrics.GeneratedTokens,
                            TotalTokens = result.Metrics.PromptTokens + result.Metrics.GeneratedTokens
                        }
                    };
                    await ctx.Response.WriteAsJsonAsync(resp, InferenceCliJsonContext.Default.OpenAiChatCompletionResponse, cancellationToken: ctx.RequestAborted);
                    return;
                }

                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers.CacheControl = "no-cache";

                var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(128)
                {
                    SingleWriter = true,
                    SingleReader = true,
                    FullMode = BoundedChannelFullMode.Wait
                });
                genCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);
                CancellationToken genToken = genCts.Token;
                genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            formattedPrompt,
                            options,
                            formatChat: false,
                            onTokenAsync: token => channel.Writer.WriteAsync(token, genToken),
                            ct: genToken);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                // Initial chunk with assistant role
                var initialChunk = new OpenAiChatChunkResponse
                {
                    Id = completionId,
                    Created = createdTimestamp,
                    Model = responseModel,
                    Choices =
                    [
                        new OpenAiChatChunkChoice
                        {
                            Index = 0,
                            Delta = new OpenAiChatDelta { Role = "assistant", Content = "" },
                            FinishReason = null
                        }
                    ]
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(initialChunk, InferenceCliJsonContext.Default.OpenAiChatChunkResponse)}\n\n", genToken);
                await ctx.Response.Body.FlushAsync(genToken);

                await foreach (var token in channel.Reader.ReadAllAsync(genToken))
                {
                    var chunk = new OpenAiChatChunkResponse
                    {
                        Id = completionId,
                        Created = createdTimestamp,
                        Model = responseModel,
                        Choices =
                        [
                            new OpenAiChatChunkChoice
                            {
                                Index = 0,
                                Delta = new OpenAiChatDelta { Content = token },
                                FinishReason = null
                            }
                        ]
                    };
                    await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk, InferenceCliJsonContext.Default.OpenAiChatChunkResponse)}\n\n", genToken);
                    await ctx.Response.Body.FlushAsync(genToken);
                }

                var genResult = await genTask;
                sleepMemory?.RecordTurnCompletion(rawLastUser, genResult.Text);
                var finalChunk = new OpenAiChatChunkResponse
                {
                    Id = completionId,
                    Created = createdTimestamp,
                    Model = responseModel,
                    Choices =
                    [
                        new OpenAiChatChunkChoice
                        {
                            Index = 0,
                            Delta = new OpenAiChatDelta(),
                            FinishReason = genResult.FinishReason
                        }
                    ]
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(finalChunk, InferenceCliJsonContext.Default.OpenAiChatChunkResponse)}\n\n", genToken);
                if (includeUsage)
                {
                    var usageChunk = new OpenAiChatChunkResponse
                    {
                        Id = completionId,
                        Created = createdTimestamp,
                        Model = responseModel,
                        Choices = [],
                        Usage = new OpenAiUsage
                        {
                            PromptTokens = genResult.Metrics.PromptTokens,
                            CompletionTokens = genResult.Metrics.GeneratedTokens,
                            TotalTokens = genResult.Metrics.PromptTokens + genResult.Metrics.GeneratedTokens
                        }
                    };
                    await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(usageChunk, InferenceCliJsonContext.Default.OpenAiChatChunkResponse)}\n\n", genToken);
                }
                await ctx.Response.WriteAsync("data: [DONE]\n\n", genToken);
                await ctx.Response.Body.FlushAsync(genToken);
            }
            finally
            {
                if (genCts != null)
                {
                    try { genCts.Cancel(); }
                    catch (ObjectDisposedException) { }
                }
                if (genTask != null)
                {
                    try
                    {
                        await genTask.ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                    }
                    catch (ChannelClosedException)
                    {
                    }
                    catch (Exception) when (genCts?.IsCancellationRequested == true)
                    {
                    }
                }
                genCts?.Dispose();
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = null;
                }
                inferenceLock.Release();
            }
        });

        // 4. GET /v1/models (OpenAI models endpoint; base model ID first per Glacier.Studio contract)
        app.MapGet("/v1/models", async (HttpContext ctx) =>
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var data = new List<OpenAiModelEntry>(2)
            {
                new OpenAiModelEntry
                {
                    Id = modelName,
                    Created = now
                }
            };
            if (!string.IsNullOrWhiteSpace(attachedLoraName))
            {
                data.Add(new OpenAiModelEntry
                {
                    Id = $"{modelName}:{attachedLoraName}",
                    Created = now,
                    OwnedBy = "glacier-lora"
                });
            }
            await ctx.Response.WriteAsJsonAsync(
                new OpenAiModelsResponse { Data = data.ToArray() },
                InferenceCliJsonContext.Default.OpenAiModelsResponse,
                cancellationToken: ctx.RequestAborted);
        });

        // 4b. GET /api/sleep/status & POST /api/sleep/trigger (Promoted Glacier.Serve SleepMemoryService)
        app.MapGet("/api/sleep/status", async (HttpContext ctx) =>
        {
            var payload = sleepMemory != null
                ? sleepMemory.GetStatus()
                : new Glacier.Serve.Serialization.SleepMemoryStatusResponse { Enabled = false, TokenBudget = 0, State = "Disabled" };
            string json = JsonSerializer.Serialize(payload, Glacier.Serve.Serialization.ServeJsonContext.Default.SleepMemoryStatusResponse);
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync(json, ctx.RequestAborted);
        });

        app.MapPost("/api/sleep/trigger", async (HttpContext ctx) =>
        {
            if (sleepMemory == null)
            {
                string disabledJson = JsonSerializer.Serialize(
                    new Glacier.Serve.Serialization.SleepMemoryStatusResponse { Enabled = false, TokenBudget = 0, State = "Disabled" },
                    Glacier.Serve.Serialization.ServeJsonContext.Default.SleepMemoryStatusResponse);
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json; charset=utf-8";
                await ctx.Response.WriteAsync(disabledJson, ctx.RequestAborted);
                return;
            }

            await sleepMemory.TriggerSleepAsync(ctx.RequestAborted);
            string json = JsonSerializer.Serialize(
                sleepMemory.GetStatus(),
                Glacier.Serve.Serialization.ServeJsonContext.Default.SleepMemoryStatusResponse);
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync(json, ctx.RequestAborted);
        });

        // 4c. GET /v1/sleep/status & POST /v1/sleep/trigger (Glacier.Studio contract)
        app.MapGet("/v1/sleep/status", async (HttpContext ctx) =>
        {
            var daemon = sleepMemory?.Daemon;
            var status = new Glacier.Serve.Serialization.SleepStatusResponse
            {
                Enabled = daemon != null,
                State = daemon != null ? daemon.CurrentState.ToString() : "Disabled",
                CompletedCycles = daemon?.CompletedSleepCycles ?? 0,
                TotalDreamsConsolidated = daemon?.TotalDreamsConsolidated ?? 0,
                PendingDreams = daemon?.PendingDreamCount ?? 0
            };
            string json = JsonSerializer.Serialize(status, Glacier.Serve.Serialization.ServeJsonContext.Default.SleepStatusResponse);
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync(json, ctx.RequestAborted);
        });

        app.MapPost("/v1/sleep/trigger", async (HttpContext ctx) =>
        {
            if (sleepMemory == null)
            {
                string disabledJson = JsonSerializer.Serialize(
                    new Glacier.Serve.Serialization.SleepTriggerResponse { Triggered = false, State = "Disabled" },
                    Glacier.Serve.Serialization.ServeJsonContext.Default.SleepTriggerResponse);
                ctx.Response.ContentType = "application/json; charset=utf-8";
                await ctx.Response.WriteAsync(disabledJson, ctx.RequestAborted);
                return;
            }

            var result = await sleepMemory.TriggerSleepAsync(ctx.RequestAborted);
            var daemon = sleepMemory.Daemon;
            var resp = result == null
                ? new Glacier.Serve.Serialization.SleepTriggerResponse
                {
                    Triggered = false,
                    State = daemon.CurrentState.ToString()
                }
                : new Glacier.Serve.Serialization.SleepTriggerResponse
                {
                    Triggered = true,
                    State = daemon.CurrentState.ToString(),
                    ExpertSlotIndex = result.ExpertSlotIndex,
                    TotalDreams = result.TotalDreamsGenerated,
                    AcceptedDreams = result.TotalDreamsAccepted,
                    AcceptanceRate = result.AcceptanceRate,
                    InitialLoss = result.InitialGkdLoss,
                    FinalLoss = result.FinalGkdLoss,
                    SynapticPruningSucceeded = result.SynapticPruningSucceeded
                };
            string json = JsonSerializer.Serialize(resp, Glacier.Serve.Serialization.ServeJsonContext.Default.SleepTriggerResponse);
            ctx.Response.ContentType = "application/json; charset=utf-8";
            await ctx.Response.WriteAsync(json, ctx.RequestAborted);
        });

        // 5. POST /api/chat (Ollama chat endpoint)
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync(InferenceCliJsonContext.Default.JsonElement, ctx.RequestAborted);
            }
            catch (JsonException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }
            catch (BadHttpRequestException)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Bad request or body size limit exceeded.", ctx.RequestAborted);
                return;
            }

            var messages = ParseMessages(req);
            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            string? reqModelChat = req.TryGetProperty("model", out var mcProp) && mcProp.ValueKind == JsonValueKind.String
                ? mcProp.GetString()?.Trim()
                : null;
            string? rawReqLoraChat = req.TryGetProperty("lora", out var lcProp) && lcProp.ValueKind == JsonValueKind.String
                ? lcProp.GetString()
                : null;
            var (loraErrStatusChat, loraErrTypeChat, loraErrMsgChat, selectedLoraChat) =
                ResolvePathBLoraRequest(reqModelChat, rawReqLoraChat, modelName, attachedLoraName);
            if (loraErrStatusChat.HasValue)
            {
                await WritePathBApiErrorAsync(ctx, loraErrStatusChat.Value, loraErrMsgChat!, loraErrTypeChat!);
                return;
            }

            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
            if (!TryParseSamplingOptionsInternal(req, defaultMaxThinkTokens, out var options, out bool? explicitEnableThinking, out string? budgetErr))
            {
                await WritePathBApiErrorAsync(ctx, StatusCodes.Status400BadRequest, budgetErr!, "invalid_request_error");
                return;
            }
            string rawLastUser = ExtractLastUserMessage(messages);
            if (sleepMemory != null)
            {
                messages = await sleepMemory.ConsolidateMessagesAsync(messages, ctx.RequestAborted);
            }
            bool hasThinkToken = session.Tokenizer.TryGetTokenId("<think>", out _);
            bool thinkingMode = explicitEnableThinking ?? hasThinkToken;
            string formattedPrompt = session.Tokenizer.FormatConversation(messages, thinkingMode: thinkingMode);

            await inferenceLock.WaitAsync(ctx.RequestAborted);
            CancellationTokenSource? genCts = null;
            Task<GenerationResult>? genTask = null;
            try
            {
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = selectedLoraChat != null ? configuredLoraWeights : null;
                }
                if (!stream)
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, formatChat: false, ct: ctx.RequestAborted);
                    sleepMemory?.RecordTurnCompletion(rawLastUser, result.Text);
                    var respObj = new OllamaChatResponse
                    {
                        Model = modelName,
                        CreatedAt = DateTime.UtcNow.ToString("o"),
                        Message = new OllamaChatMessageDto { Role = "assistant", Content = result.Text },
                        Done = true,
                        TotalDuration = (long)(result.Metrics.TotalDuration.TotalSeconds * 1e9),
                        PromptEvalCount = result.Metrics.PromptTokens,
                        PromptEvalDuration = (long)(result.Metrics.PromptEvalDuration.TotalSeconds * 1e9),
                        EvalCount = result.Metrics.GeneratedTokens,
                        EvalDuration = (long)(result.Metrics.GenerationDuration.TotalSeconds * 1e9)
                    };
                    await ctx.Response.WriteAsJsonAsync(respObj, InferenceCliJsonContext.Default.OllamaChatResponse, cancellationToken: ctx.RequestAborted);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson";
                var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(128)
                {
                    SingleWriter = true,
                    SingleReader = true,
                    FullMode = BoundedChannelFullMode.Wait
                });
                genCts = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted);
                CancellationToken genToken = genCts.Token;
                genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            formattedPrompt,
                            options,
                            formatChat: false,
                            onTokenAsync: token => channel.Writer.WriteAsync(token, genToken),
                            ct: genToken);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                await foreach (var token in channel.Reader.ReadAllAsync(genToken))
                {
                    var chunk = new OllamaChatResponse
                    {
                        Model = modelName,
                        CreatedAt = DateTime.UtcNow.ToString("o"),
                        Message = new OllamaChatMessageDto { Role = "assistant", Content = token },
                        Done = false
                    };
                    await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk, InferenceCliJsonContext.Default.OllamaChatResponse) + "\n", genToken);
                    await ctx.Response.Body.FlushAsync(genToken);
                }

                var genResult = await genTask;
                sleepMemory?.RecordTurnCompletion(rawLastUser, genResult.Text);
                var finalChunk = new OllamaChatResponse
                {
                    Model = modelName,
                    CreatedAt = DateTime.UtcNow.ToString("o"),
                    Message = new OllamaChatMessageDto { Role = "assistant", Content = "" },
                    Done = true,
                    TotalDuration = (long)(genResult.Metrics.TotalDuration.TotalSeconds * 1e9),
                    EvalCount = genResult.Metrics.GeneratedTokens,
                    EvalDuration = (long)(genResult.Metrics.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk, InferenceCliJsonContext.Default.OllamaChatResponse) + "\n", genToken);
                await ctx.Response.Body.FlushAsync(genToken);
            }
            finally
            {
                if (genCts != null)
                {
                    try { genCts.Cancel(); }
                    catch (ObjectDisposedException) { }
                }
                if (genTask != null)
                {
                    try
                    {
                        await genTask.ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                    }
                    catch (ChannelClosedException)
                    {
                    }
                    catch (Exception) when (genCts?.IsCancellationRequested == true)
                    {
                    }
                }
                genCts?.Dispose();
                if (session.CpuModel != null && !string.IsNullOrWhiteSpace(attachedLoraName))
                {
                    session.CpuModel.LoraWeights = null;
                }
                inferenceLock.Release();
            }
        });
    }

    public static (int? ErrorStatusCode, string? ErrorType, string? ErrorMessage, string? SelectedLora) ResolvePathBLoraRequest(
        string? reqModel,
        string? rawReqLora,
        string modelName,
        string? attachedLoraName)
    {
        string? explicitLora = null;
        if (rawReqLora != null)
        {
            if (string.IsNullOrWhiteSpace(rawReqLora))
            {
                return (
                    StatusCodes.Status400BadRequest,
                    "invalid_request_error",
                    "LoRA adapter name cannot be empty or whitespace.",
                    null);
            }
            explicitLora = rawReqLora.Trim();
        }

        string? modelLora = null;
        string trimmedModel = reqModel?.Trim() ?? string.Empty;
        if (!string.IsNullOrEmpty(trimmedModel) &&
            !string.Equals(trimmedModel, modelName, StringComparison.OrdinalIgnoreCase))
        {
            int colonIdx = trimmedModel.LastIndexOf(':');
            if (colonIdx >= 0)
            {
                string basePart = trimmedModel[..colonIdx].Trim();
                string adapterPart = trimmedModel[(colonIdx + 1)..].Trim();
                if (string.IsNullOrEmpty(basePart) || string.IsNullOrEmpty(adapterPart))
                {
                    return (
                        StatusCodes.Status400BadRequest,
                        "invalid_request_error",
                        $"Invalid model:adapter specifier '{trimmedModel}'.",
                        null);
                }
                modelLora = adapterPart;
            }
        }

        if (modelLora != null && explicitLora != null &&
            !string.Equals(modelLora, explicitLora, StringComparison.Ordinal))
        {
            return (
                StatusCodes.Status400BadRequest,
                "invalid_request_error",
                $"Conflicting LoRA adapter selection between model '{trimmedModel}' ('{modelLora}') and lora '{explicitLora}'.",
                null);
        }

        string? selectedLora = explicitLora ?? modelLora;
        if (selectedLora != null &&
            !string.Equals(selectedLora, attachedLoraName, StringComparison.Ordinal))
        {
            return (
                StatusCodes.Status404NotFound,
                "model_not_found",
                $"LoRA adapter '{selectedLora}' is not loaded on this server.",
                null);
        }

        return (null, null, null, selectedLora);
    }

    private static async Task WritePathBApiErrorAsync(HttpContext ctx, int statusCode, string message, string errorType)
    {
        var err = new Glacier.Serve.Serialization.ApiErrorResponse
        {
            Error = new Glacier.Serve.Serialization.ApiErrorDetail
            {
                Message = message,
                Type = errorType
            }
        };
        string errJson = JsonSerializer.Serialize(err, Glacier.Serve.Serialization.ServeJsonContext.Default.ApiErrorResponse);
        ctx.Response.StatusCode = statusCode;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(errJson, ctx.RequestAborted);
    }

    private static bool TryParseCliThinkBudget(string? raw, out int parsed)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            parsed = 0;
            return false;
        }
        if (int.TryParse(raw.Trim(), out int intVal) && intVal == ReasoningBudgetPolicy.AutoBudgetSentinel)
        {
            parsed = ReasoningBudgetPolicy.AutoBudgetSentinel;
            return true;
        }
        return ReasoningBudgetPolicy.TryParseSetting(raw, out parsed);
    }

    public static SamplingOptions ParseSamplingOptions(JsonElement req, int defaultMaxThinkTokens = ReasoningBudgetPolicy.AutoBudgetSentinel)
    {
        if (!TryParseSamplingOptionsInternal(req, defaultMaxThinkTokens, out var options, out _, out string? error))
        {
            throw new ArgumentOutOfRangeException(nameof(req), error);
        }
        return options;
    }

    private static bool TryParseSamplingOptionsInternal(
        JsonElement req,
        int defaultMaxThinkTokens,
        out SamplingOptions options,
        out bool? explicitEnableThinking,
        out string? errorMessage)
    {
        options = new SamplingOptions();
        explicitEnableThinking = null;
        errorMessage = null;

        if (req.TryGetProperty("temperature", out var tempProp) && tempProp.TryGetSingle(out float temp))
            options.Temperature = temp;
        if (req.TryGetProperty("top_p", out var topPProp) && topPProp.TryGetSingle(out float topP))
            options.TopP = topP;
        if (req.TryGetProperty("top_k", out var topKProp) && topKProp.TryGetInt32(out int topK))
            options.TopK = topK;
        if (req.TryGetProperty("max_tokens", out var maxTokProp) && maxTokProp.TryGetInt32(out int maxTok))
            options.MaxTokens = maxTok;
        else if (req.TryGetProperty("max_completion_tokens", out var maxCompProp) && maxCompProp.TryGetInt32(out int maxComp))
            options.MaxTokens = maxComp;
        if (req.TryGetProperty("repetition_penalty", out var repProp) && repProp.TryGetSingle(out float rep))
            options.RepetitionPenalty = rep;

        // Check nested Ollama "options" object
        if (req.TryGetProperty("options", out var subOpt) && subOpt.ValueKind == JsonValueKind.Object)
        {
            if (subOpt.TryGetProperty("temperature", out var subTemp) && subTemp.TryGetSingle(out float st))
                options.Temperature = st;
            if (subOpt.TryGetProperty("top_p", out var subTopP) && subTopP.TryGetSingle(out float sp))
                options.TopP = sp;
            if (subOpt.TryGetProperty("top_k", out var subTopK) && subTopK.TryGetInt32(out int sk))
                options.TopK = sk;
            if (subOpt.TryGetProperty("num_predict", out var subNum) && subNum.TryGetInt32(out int np))
                options.MaxTokens = np;
            if (subOpt.TryGetProperty("repeat_penalty", out var subRep) && subRep.TryGetSingle(out float sr))
                options.RepetitionPenalty = sr;
        }

        explicitEnableThinking = TryExtractEnableThinking(req);
        if (!explicitEnableThinking.HasValue &&
            req.TryGetProperty("options", out var optsForEnable) &&
            optsForEnable.ValueKind == JsonValueKind.Object)
        {
            explicitEnableThinking = TryExtractEnableThinking(optsForEnable);
        }

        int configuredThinkBudget = defaultMaxThinkTokens;
        bool hasExplicitThinkBudget = false;
        if (!TryExtractThinkBudget(req, out hasExplicitThinkBudget, out int rootThinkBudget, out errorMessage))
        {
            return false;
        }
        if (hasExplicitThinkBudget)
        {
            configuredThinkBudget = rootThinkBudget;
        }
        if (req.TryGetProperty("options", out var optsForThink) && optsForThink.ValueKind == JsonValueKind.Object)
        {
            if (!TryExtractThinkBudget(optsForThink, out bool hasNestedThinkBudget, out int nestedThinkBudget, out errorMessage))
            {
                return false;
            }
            if (!hasExplicitThinkBudget && hasNestedThinkBudget)
            {
                hasExplicitThinkBudget = true;
                configuredThinkBudget = nestedThinkBudget;
            }
        }

        if (!hasExplicitThinkBudget && explicitEnableThinking.HasValue)
        {
            configuredThinkBudget = explicitEnableThinking.Value ? ReasoningBudgetPolicy.AutoBudgetSentinel : 0;
        }

        options.MaxReasoningTokens = ReasoningBudgetPolicy.ResolveBudget(configuredThinkBudget, options.MaxTokens);
        return true;
    }

    private static string ExtractLastUserMessage(List<(string Role, string Content)> messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role.Equals("user", StringComparison.OrdinalIgnoreCase))
            {
                return messages[i].Content;
            }
        }
        return messages.Count > 0 ? messages[^1].Content : "";
    }

    private static bool? TryExtractEnableThinking(JsonElement obj)
    {
        if (obj.TryGetProperty("chat_template_kwargs", out var kwargsProp)
            && kwargsProp.ValueKind == JsonValueKind.Object
            && kwargsProp.TryGetProperty("enable_thinking", out var kwargsEnableThinking))
        {
            if (kwargsEnableThinking.ValueKind == JsonValueKind.False) return false;
            if (kwargsEnableThinking.ValueKind == JsonValueKind.True) return true;
        }

        if (obj.TryGetProperty("enable_thinking", out var enableThinkingProp))
        {
            if (enableThinkingProp.ValueKind == JsonValueKind.False) return false;
            if (enableThinkingProp.ValueKind == JsonValueKind.True) return true;
        }

        return null;
    }

    private static bool TryExtractThinkBudget(JsonElement obj, out bool found, out int budget, out string? errorMessage)
    {
        found = false;
        budget = 0;
        errorMessage = null;

        foreach (string propName in new[] { "max_think_tokens", "max_thinking_tokens", "reasoning_budget" })
        {
            if (!obj.TryGetProperty(propName, out var prop) || prop.ValueKind == JsonValueKind.Null)
                continue;

            int parsedValue;
            if (prop.ValueKind == JsonValueKind.Number)
            {
                if (!prop.TryGetInt32(out int numVal))
                {
                    errorMessage = $"Invalid '{propName}': must be a valid 32-bit integer >= {ReasoningBudgetPolicy.AutoBudgetSentinel}.";
                    return false;
                }
                if (numVal < ReasoningBudgetPolicy.AutoBudgetSentinel)
                {
                    errorMessage = $"Reasoning budget '{propName}' ({numVal}) must be >= {ReasoningBudgetPolicy.AutoBudgetSentinel}.";
                    return false;
                }
                parsedValue = numVal;
            }
            else if (prop.ValueKind == JsonValueKind.False)
            {
                parsedValue = 0;
            }
            else if (prop.ValueKind == JsonValueKind.True)
            {
                parsedValue = ReasoningBudgetPolicy.AutoBudgetSentinel;
            }
            else if (prop.ValueKind == JsonValueKind.String)
            {
                string? raw = prop.GetString();
                if (!TryParseCliThinkBudget(raw, out parsedValue))
                {
                    errorMessage = $"Invalid reasoning budget '{raw}' for '{propName}'.";
                    return false;
                }
            }
            else
            {
                errorMessage = $"Unsupported JSON token '{prop.ValueKind}' for '{propName}'.";
                return false;
            }

            if (!found)
            {
                found = true;
                budget = parsedValue;
            }
        }

        return true;
    }

    private static List<(string Role, string Content)> ParseMessages(JsonElement req)
    {
        var list = new List<(string Role, string Content)>();
        if (!req.TryGetProperty("messages", out var messages) || messages.ValueKind != JsonValueKind.Array)
            return list;

        foreach (var m in messages.EnumerateArray())
        {
            string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "user" : "user";
            string content = "";
            if (m.TryGetProperty("content", out var c))
            {
                if (c.ValueKind == JsonValueKind.String)
                {
                    content = c.GetString() ?? "";
                }
                else if (c.ValueKind == JsonValueKind.Array)
                {
                    var sb = new StringBuilder();
                    foreach (var part in c.EnumerateArray())
                    {
                        if (part.TryGetProperty("type", out var typeProp) &&
                            typeProp.GetString() == "text" &&
                            part.TryGetProperty("text", out var textProp))
                        {
                            sb.Append(textProp.GetString());
                        }
                    }
                    content = sb.ToString();
                }
            }
            list.Add((role, content));
        }
        return list;
    }
}
