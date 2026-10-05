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
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;

public static partial class Program
{
    private static async Task<int> RunServeAsync(string[] args)
    {
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintServeHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? modelPath = null;
        int port = 11434;
        string host = "0.0.0.0";
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        string? split = null;
        int maxSeqLen = 2048;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                modelPath = args[++i];
            else if (args[i] == "--port" && i + 1 < args.Length && int.TryParse(args[++i], out int p))
                port = p;
            else if (args[i] == "--host" && i + 1 < args.Length)
                host = args[++i];
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
            PrintServeHelp();
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
        Console.WriteLine($">> Model ready on {session.ActiveDevice}\n");
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

        // 5. POST /api/chat (Ollama chat endpoint)
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            var req = await ctx.Request.ReadFromJsonAsync<JsonElement>();
            if (!req.TryGetProperty("messages", out var messages))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.");
                return;
            }

            bool stream = !req.TryGetProperty("stream", out var s) || s.GetBoolean();
            var sb = new StringBuilder();
            foreach (var m in messages.EnumerateArray())
            {
                string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "user" : "user";
                string content = m.TryGetProperty("content", out var c) ? c.GetString() ?? "" : "";
                sb.Append($"<|im_start|>{role}\n{content}<|im_end|>\n");
            }
            sb.Append("<|im_start|>assistant\n");

            var options = new SamplingOptions { MaxTokens = 512, Temperature = 0.7f };

            if (!stream)
            {
                var result = await session.GenerateAsync(sb.ToString(), options, formatChat: false);
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
                await ctx.Response.WriteAsJsonAsync(respObj);
                return;
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var genResult = await session.GenerateAsync(sb.ToString(), options, formatChat: false, onToken: token =>
            {
                var chunk = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    message = new { role = "assistant", content = token },
                    done = false
                };
                string line = JsonSerializer.Serialize(chunk) + "\n";
                ctx.Response.WriteAsync(line).GetAwaiter().GetResult();
            });

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
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n");
        });

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
}
