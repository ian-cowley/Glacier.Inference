namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Glacier.Inference.Cli;
using Glacier.Inference.Engine;
using Xunit;

public sealed class LocalModelFactAttribute : FactAttribute
{
    public LocalModelFactAttribute()
    {
        if (!File.Exists(CudaFactAttribute.ModelPath))
        {
            Skip = $"Integration test model file not found at '{CudaFactAttribute.ModelPath}'.";
        }
    }
}

public class HttpServerIntegrationTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [LocalModelFact]
    public async Task ApiTags_And_V1Models_ReturnExpectedMetadata()
    {
        int port = GetAvailablePort();
        using var session = new InferenceSession(CudaFactAttribute.ModelPath, maxSeqLen: 512, device: "cpu");
        var app = Program.CreateServeApp(session, CudaFactAttribute.ModelPath, apiKey: null, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. Test GET /api/tags
            var tagsResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.OK, tagsResp.StatusCode);
            var tagsJson = await tagsResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.True(tagsJson.TryGetProperty("models", out var models));
            Assert.True(models.GetArrayLength() > 0);
            var firstModel = models[0];
            Assert.True(firstModel.TryGetProperty("details", out var details));
            Assert.Equal("gguf", details.GetProperty("format").GetString());
            Assert.False(string.IsNullOrEmpty(details.GetProperty("parameter_size").GetString()));

            // 2. Test GET /v1/models
            var modelsResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
            var modelsJson = await modelsResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("list", modelsJson.GetProperty("object").GetString());
            Assert.True(modelsJson.GetProperty("data").GetArrayLength() > 0);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [LocalModelFact]
    public async Task ApiKeyAuth_EnforcesChallenge_WhenConfigured()
    {
        const string testKey = "test-glacier-secret-key-12345";
        int port = GetAvailablePort();
        using var session = new InferenceSession(CudaFactAttribute.ModelPath, maxSeqLen: 512, device: "cpu");
        var app = Program.CreateServeApp(session, CudaFactAttribute.ModelPath, apiKey: testKey, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. Without credentials -> 401
            var unauthResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.Unauthorized, unauthResp.StatusCode);

            // 2. With wrong Bearer key -> 401
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "wrong-key");
            var wrongResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.Unauthorized, wrongResp.StatusCode);

            // 3. With correct Bearer key -> 200
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", testKey);
            var authResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.OK, authResp.StatusCode);

            // 4. With x-api-key header -> 200
            client.DefaultRequestHeaders.Authorization = null;
            client.DefaultRequestHeaders.Add("x-api-key", testKey);
            var xApiKeyResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.OK, xApiKeyResp.StatusCode);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [LocalModelFact]
    public async Task Validation_RejectsMalformedPayloads()
    {
        int port = GetAvailablePort();
        using var session = new InferenceSession(CudaFactAttribute.ModelPath, maxSeqLen: 512, device: "cpu");
        var app = Program.CreateServeApp(session, CudaFactAttribute.ModelPath, apiKey: null, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. /api/generate without prompt -> 400
            var emptyGen = await client.PostAsJsonAsync("/api/generate", new { prompt = "" });
            Assert.Equal(HttpStatusCode.BadRequest, emptyGen.StatusCode);

            // 2. /api/chat without messages -> 400
            var emptyChat = await client.PostAsJsonAsync("/api/chat", new { });
            Assert.Equal(HttpStatusCode.BadRequest, emptyChat.StatusCode);

            // 3. /v1/chat/completions without messages -> 400
            var emptyComp = await client.PostAsJsonAsync("/v1/chat/completions", new { });
            Assert.Equal(HttpStatusCode.BadRequest, emptyComp.StatusCode);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [LocalModelFact]
    public async Task V1ChatCompletions_Streaming_EmitsSseEvents()
    {
        int port = GetAvailablePort();
        using var session = new InferenceSession(CudaFactAttribute.ModelPath, maxSeqLen: 512, device: "cpu");
        var app = Program.CreateServeApp(session, CudaFactAttribute.ModelPath, apiKey: null, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var payload = new
            {
                stream = true,
                max_tokens = 3,
                messages = new[]
                {
                    new { role = "user", content = "Hi" }
                }
            };

            using var req = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
            {
                Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
            };

            using var resp = await client.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            Assert.Equal("text/event-stream", resp.Content.Headers.ContentType?.MediaType);

            using var stream = await resp.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);

            string? line;
            bool sawChunk = false;
            bool sawDone = false;

            while ((line = await reader.ReadLineAsync()) != null)
            {
                if (line.StartsWith("data: "))
                {
                    string data = line.Substring(6).Trim();
                    if (data == "[DONE]")
                    {
                        sawDone = true;
                        break;
                    }
                    if (data.Contains("chat.completion.chunk"))
                    {
                        sawChunk = true;
                    }
                }
            }

            Assert.True(sawChunk, "Expected at least one chat.completion.chunk SSE payload.");
            Assert.True(sawDone, "Expected final data: [DONE] event.");
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }
}
