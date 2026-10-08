namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
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
using Glacier.Inference.Sampling;

public static partial class Program
{
    private static readonly SemaphoreSlim s_inferenceLock = new(1, 1);

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

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i] == "--port" && i + 1 < args.Length)
            {
                if (!int.TryParse(args[++i], out int p) || p < 1 || p > 65535)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error: --port must be a valid port number between 1 and 65535.");
                    Console.ResetColor();
                    return 1;
                }
                port = p;
            }
            else if (args[i] == "--host" && i + 1 < args.Length)
                host = args[++i];
            else if (args[i] == "--api-key" && i + 1 < args.Length)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("WARNING: Specifying --api-key on the command line is deprecated for security reasons.");
                Console.WriteLine("         Set the GLACIER_API_KEY environment variable instead.");
                Console.ResetColor();
                apiKey = args[++i];
            }
            else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (args[i] == "--split" && i + 1 < args.Length)
                split = args[++i];
            else if ((args[i] == "-c" || args[i] == "--ctx" || args[i] == "--context-length") && i + 1 < args.Length)
            {
                if (!int.TryParse(args[++i], out int cLen) || cLen <= 0)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Error: Context length must be a positive integer.");
                    Console.ResetColor();
                    return 1;
                }
                maxSeqLen = cLen;
            }
            else if (!args[i].StartsWith("-") && modelPath == null)
                modelPath = args[i];
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
                Console.WriteLine($"Error: Unknown engine '{engineStr}'. Valid values: Auto, Cpu, BareMetal, DirectML.");
                Console.ResetColor();
                return 1;
            }
            engine = parsedEngine;
        }

        KvCachePrecision kvPrecision = KvCachePrecision.Auto;
        if (!string.IsNullOrWhiteSpace(kvPrecisionStr))
        {
            if (!Enum.TryParse<KvCachePrecision>(kvPrecisionStr, ignoreCase: true, out var parsedPrecision))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Error: Unknown KV precision '{kvPrecisionStr}'. Valid values: Auto, Fp32, Fp16, Q8_0, Q4_0.");
                Console.ResetColor();
                return 1;
            }
            kvPrecision = parsedPrecision;
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
        Console.WriteLine();

        Console.WriteLine(">> Initializing inference session...");
        var session = new InferenceSession(modelPath, maxSeqLen: maxSeqLen, device: device, engine: engine, kvPrecision: kvPrecision, split: split);
        var app = CreateServeApp(session, modelPath, apiKey, host, port);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n>>> Glacier.Inference server running on http://{host}:{port}");
        Console.WriteLine("    Compatible with Ollama CLI, LangChain, AutoGen, and OpenAI SDKs!");
        Console.ResetColor();

        await app.RunAsync();
        return 0;
    }

    public static WebApplication CreateServeApp(
        InferenceSession session,
        string modelPath,
        string? apiKey = null,
        string host = "127.0.0.1",
        int port = 11434)
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

        ConfigureServeEndpoints(app, session, modelPath, apiKey);
        return app;
    }

    public static void ConfigureServeEndpoints(
        WebApplication app,
        InferenceSession session,
        string modelPath,
        string? apiKey = null)
    {
        string modelName = Path.GetFileNameWithoutExtension(modelPath);

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
                    authorized = CryptographicOperations.FixedTimeEquals(
                        Encoding.UTF8.GetBytes(token),
                        Encoding.UTF8.GetBytes(apiKey));
                }
                else if (!string.IsNullOrEmpty(xApiKey))
                {
                    authorized = CryptographicOperations.FixedTimeEquals(
                        Encoding.UTF8.GetBytes(xApiKey),
                        Encoding.UTF8.GetBytes(apiKey));
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
                        parameter_size = paramSize,
                        quantization_level = quantLevel
                    }
                }
            }
        }));

        // 2. POST /api/generate (Ollama streaming & non-streaming)
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
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
            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            var options = ParseSamplingOptions(req);

            await s_inferenceLock.WaitAsync(ctx.RequestAborted);
            try
            {
                if (!stream)
                {
                    var result = await session.GenerateAsync(prompt, options, formatChat: true, ct: ctx.RequestAborted);
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
                    await ctx.Response.WriteAsJsonAsync(respObj, ctx.RequestAborted);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson";
                var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(128)
                {
                    SingleWriter = true,
                    SingleReader = true,
                    FullMode = BoundedChannelFullMode.Wait
                });
                var genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            prompt,
                            options,
                            formatChat: true,
                            onToken: token => channel.Writer.TryWrite(token),
                            ct: ctx.RequestAborted);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new
                    {
                        model = modelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        response = token,
                        done = false
                    };
                    await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk) + "\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
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
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            finally
            {
                s_inferenceLock.Release();
            }
        });

        // 3. POST /v1/chat/completions (OpenAI compatible)
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
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

            bool stream = req.TryGetProperty("stream", out var s) && (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));
            var options = ParseSamplingOptions(req);
            string formattedPrompt = session.Tokenizer.FormatChat(messages);
            string completionId = $"chatcmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            await s_inferenceLock.WaitAsync(ctx.RequestAborted);
            try
            {
                if (!stream)
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, formatChat: false, ct: ctx.RequestAborted);
                    var resp = new
                    {
                        id = completionId,
                        @object = "chat.completion",
                        created = createdTimestamp,
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
                    await ctx.Response.WriteAsJsonAsync(resp, ctx.RequestAborted);
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
                var genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            formattedPrompt,
                            options,
                            formatChat: false,
                            onToken: token => channel.Writer.TryWrite(token),
                            ct: ctx.RequestAborted);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                // Initial chunk with assistant role
                var initialChunk = new
                {
                    id = completionId,
                    @object = "chat.completion.chunk",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = "assistant", content = "" },
                            finish_reason = (string?)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(initialChunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new
                    {
                        id = completionId,
                        @object = "chat.completion.chunk",
                        created = createdTimestamp,
                        model = modelName,
                        choices = new[]
                        {
                            new
                            {
                                index = 0,
                                delta = new { content = token },
                                finish_reason = (string?)null
                            }
                        }
                    };
                    await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
                var finalChunk = new
                {
                    id = completionId,
                    @object = "chat.completion.chunk",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { },
                            finish_reason = genResult.FinishReason
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(finalChunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            finally
            {
                s_inferenceLock.Release();
            }
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

        // 5. POST /api/chat (Ollama chat endpoint)
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
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

            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
            var options = ParseSamplingOptions(req);
            string formattedPrompt = session.Tokenizer.FormatChat(messages);

            await s_inferenceLock.WaitAsync(ctx.RequestAborted);
            try
            {
                if (!stream)
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, formatChat: false, ct: ctx.RequestAborted);
                    var respObj = new
                    {
                        model = modelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new { role = "assistant", content = result.Text },
                        done = true,
                        total_duration = (long)(result.Metrics.TotalDuration.TotalSeconds * 1e9),
                        prompt_eval_count = result.Metrics.PromptTokens,
                        prompt_eval_duration = (long)(result.Metrics.PromptEvalDuration.TotalSeconds * 1e9),
                        eval_count = result.Metrics.GeneratedTokens,
                        eval_duration = (long)(result.Metrics.GenerationDuration.TotalSeconds * 1e9)
                    };
                    await ctx.Response.WriteAsJsonAsync(respObj, ctx.RequestAborted);
                    return;
                }

                ctx.Response.ContentType = "application/x-ndjson";
                var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(128)
                {
                    SingleWriter = true,
                    SingleReader = true,
                    FullMode = BoundedChannelFullMode.Wait
                });
                var genTask = Task.Run(async () =>
                {
                    try
                    {
                        return await session.GenerateAsync(
                            formattedPrompt,
                            options,
                            formatChat: false,
                            onToken: token => channel.Writer.TryWrite(token),
                            ct: ctx.RequestAborted);
                    }
                    finally
                    {
                        channel.Writer.Complete();
                    }
                });

                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new
                    {
                        model = modelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new { role = "assistant", content = token },
                        done = false
                    };
                    await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk) + "\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
                var finalChunk = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    message = new { role = "assistant", content = "" },
                    done = true,
                    total_duration = (long)(genResult.Metrics.TotalDuration.TotalSeconds * 1e9),
                    eval_count = genResult.Metrics.GeneratedTokens,
                    eval_duration = (long)(genResult.Metrics.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            finally
            {
                s_inferenceLock.Release();
            }
        });
    }

    private static SamplingOptions ParseSamplingOptions(JsonElement req)
    {
        var options = new SamplingOptions();
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
        return options;
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
