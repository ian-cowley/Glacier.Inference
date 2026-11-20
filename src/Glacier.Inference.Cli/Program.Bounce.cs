namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

public static partial class Program
{
    private static async Task<int> RunBounceAsync(string[] args)
    {
        string primaryEndpoint = "http://192.168.1.108:18080";
        string? criticEndpoint = null;
        string host = "127.0.0.1";
        int port = 18090;
        string? harnessPath = null;
        int maxRepairs = 2;
        bool compilerBounce = true;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--primary" when i + 1 < args.Length:
                    primaryEndpoint = args[++i];
                    break;
                case "--critic" when i + 1 < args.Length:
                case "--peer-critic" when i + 1 < args.Length:
                    criticEndpoint = args[++i];
                    break;
                case "--host" when i + 1 < args.Length:
                    host = args[++i];
                    break;
                case "--port" when i + 1 < args.Length:
                    if (int.TryParse(args[++i], out int p)) port = p;
                    break;
                case "--harness" when i + 1 < args.Length:
                    harnessPath = args[++i];
                    break;
                case "--max-repairs" when i + 1 < args.Length:
                    if (int.TryParse(args[++i], out int mr)) maxRepairs = mr;
                    break;
                case "--no-compiler-bounce":
                    compilerBounce = false;
                    break;
                case "--compiler-bounce":
                    compilerBounce = true;
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(harnessPath))
        {
            string candidate = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "tests", "Glacier.EvaluationHarness", "bin", "Release", "net10.0", "Glacier.EvaluationHarness.exe"));
            if (File.Exists(candidate))
                harnessPath = candidate;
        }

        using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(600) };
        var verifier = new LocalProcessBounceVerifier(harnessPath);
        var bounceService = new PeerBounceService(
            httpClient,
            verifier,
            primaryEndpoint,
            criticEndpoint,
            compilerBounceEnabled: compilerBounce,
            maxRepairs: maxRepairs);

        string modelAlias = "Glacier-3Machine-Bounce";
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║        GLACIER.INFERENCE - 3-Machine Critic + Compiler Bounce        ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine($"  Listen:   http://{host}:{port}");
        Console.WriteLine($"  Primary:  {primaryEndpoint}");
        Console.WriteLine($"  Critic:   {criticEndpoint ?? "(disabled)"}");
        Console.WriteLine($"  Verifier: {(compilerBounce ? $"ENABLED (maxRepairs={maxRepairs}, harness={harnessPath ?? "syntax-only"})" : "DISABLED")}");
        Console.WriteLine("  Routes:   GET /v1/models, POST /v1/chat/completions, GET /api/bounce/status, GET /health");
        Console.WriteLine();

        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls($"http://{host}:{port}");
        var app = builder.Build();

        app.MapGet("/health", () => Results.Text("{\"status\":\"ok\"}", "application/json"));

        app.MapGet("/api/bounce/status", () =>
        {
            string json = JsonSerializer.Serialize(bounceService.GetStatus(), InferenceCliJsonContext.Default.BounceStatusResponse);
            return Results.Text(json, "application/json");
        });

        app.MapGet("/v1/models", () =>
        {
            var resp = new OpenAiModelsResponse
            {
                Data =
                [
                    new OpenAiModelEntry
                    {
                        Id = modelAlias,
                        Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
                    }
                ]
            };
            string json = JsonSerializer.Serialize(resp, InferenceCliJsonContext.Default.OpenAiModelsResponse);
            return Results.Text(json, "application/json");
        });

        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            using var reader = new StreamReader(ctx.Request.Body);
            string rawBody = await reader.ReadToEndAsync(ctx.RequestAborted);
            using var doc = JsonDocument.Parse(rawBody);
            var root = doc.RootElement;

            string? reqModel = root.TryGetProperty("model", out var mProp) ? mProp.GetString() : modelAlias;
            double temp = root.TryGetProperty("temperature", out var tProp) && tProp.ValueKind == JsonValueKind.Number ? tProp.GetDouble() : 0.0;
            int maxTok = root.TryGetProperty("max_tokens", out var mtProp) && mtProp.ValueKind == JsonValueKind.Number ? mtProp.GetInt32() : 1024;
            bool stream = root.TryGetProperty("stream", out var sProp) && sProp.ValueKind == JsonValueKind.True;

            var msgs = new List<OllamaChatMessageDto>();
            if (root.TryGetProperty("messages", out var msgArr) && msgArr.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in msgArr.EnumerateArray())
                {
                    string role = item.TryGetProperty("role", out var rP) ? (rP.GetString() ?? "user") : "user";
                    string content = item.TryGetProperty("content", out var cP) ? (cP.GetString() ?? "") : "";
                    msgs.Add(new OllamaChatMessageDto { Role = role, Content = content });
                }
            }

            var result = await bounceService.CompleteAsync(msgs.ToArray(), reqModel, temp, maxTok, ctx.RequestAborted);
            long now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            string id = $"chatcmpl-bounce-{Guid.NewGuid():N}";

            if (stream)
            {
                ctx.Response.ContentType = "text/event-stream";
                var chunk = new OpenAiChatChunkResponse
                {
                    Id = id,
                    Created = now,
                    Model = modelAlias,
                    Choices =
                    [
                        new OpenAiChatChunkChoice
                        {
                            Index = 0,
                            Delta = new OpenAiChatDelta { Role = "assistant", Content = result.Content },
                            FinishReason = result.FinishReason
                        }
                    ]
                };
                string cJson = JsonSerializer.Serialize(chunk, InferenceCliJsonContext.Default.OpenAiChatChunkResponse);
                await ctx.Response.WriteAsync($"data: {cJson}\n\ndata: [DONE]\n\n", ctx.RequestAborted);
                return Results.Empty;
            }

            var completion = new OpenAiChatCompletionResponse
            {
                Id = id,
                Created = now,
                Model = modelAlias,
                Choices =
                [
                    new OpenAiChatChoice
                    {
                        Index = 0,
                        Message = new OllamaChatMessageDto { Role = "assistant", Content = result.Content },
                        FinishReason = result.FinishReason
                    }
                ],
                Usage = new OpenAiUsage
                {
                    PromptTokens = result.PromptTokens,
                    CompletionTokens = result.CompletionTokens,
                    TotalTokens = result.PromptTokens + result.CompletionTokens
                }
            };
            string outJson = JsonSerializer.Serialize(completion, InferenceCliJsonContext.Default.OpenAiChatCompletionResponse);
            return Results.Text(outJson, "application/json");
        });

        await app.RunAsync();
        return 0;
    }
}
