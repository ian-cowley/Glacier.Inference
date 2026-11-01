namespace Glacier.Inference.Cli;

using System.Text.Json;
using System.Text.Json.Serialization;

// ============================================================================
// Strongly-Typed Source-Generated JSON DTOs for CLI Serve & Benchmark
// 100% Native AOT compatible (zero reflection serialization)
// ============================================================================

public sealed class OllamaTagsResponse
{
    [JsonPropertyName("models")]
    public required OllamaModelEntry[] Models { get; init; }
}

public sealed class OllamaModelEntry
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("modified_at")]
    public required string ModifiedAt { get; init; }

    [JsonPropertyName("size")]
    public long Size { get; init; }

    [JsonPropertyName("details")]
    public required OllamaModelDetails Details { get; init; }
}

public sealed class OllamaModelDetails
{
    [JsonPropertyName("format")]
    public required string Format { get; init; }

    [JsonPropertyName("family")]
    public required string Family { get; init; }

    [JsonPropertyName("parameter_size")]
    public required string ParameterSize { get; init; }

    [JsonPropertyName("quantization_level")]
    public required string QuantizationLevel { get; init; }
}

public sealed class OllamaGenerateResponse
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("created_at")]
    public required string CreatedAt { get; init; }

    [JsonPropertyName("response")]
    public required string Response { get; init; }

    [JsonPropertyName("done")]
    public bool Done { get; init; }

    [JsonPropertyName("total_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? TotalDuration { get; init; }

    [JsonPropertyName("prompt_eval_count")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? PromptEvalCount { get; init; }

    [JsonPropertyName("prompt_eval_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? PromptEvalDuration { get; init; }

    [JsonPropertyName("eval_count")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? EvalCount { get; init; }

    [JsonPropertyName("eval_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? EvalDuration { get; init; }
}

public sealed class OllamaChatMessageDto
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public required string Content { get; init; }
}

public sealed class OllamaChatResponse
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("created_at")]
    public required string CreatedAt { get; init; }

    [JsonPropertyName("message")]
    public required OllamaChatMessageDto Message { get; init; }

    [JsonPropertyName("done")]
    public bool Done { get; init; }

    [JsonPropertyName("total_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? TotalDuration { get; init; }

    [JsonPropertyName("prompt_eval_count")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? PromptEvalCount { get; init; }

    [JsonPropertyName("prompt_eval_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? PromptEvalDuration { get; init; }

    [JsonPropertyName("eval_count")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? EvalCount { get; init; }

    [JsonPropertyName("eval_duration")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public long? EvalDuration { get; init; }
}

public sealed class OpenAiModelsResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("data")]
    public required OpenAiModelEntry[] Data { get; init; }
}

public sealed class OpenAiModelEntry
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = "glacier";
}

public sealed class OpenAiChatCompletionResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required OpenAiChatChoice[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public required OpenAiUsage Usage { get; init; }
}

public sealed class OpenAiChatChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public required OllamaChatMessageDto Message { get; init; }

    [JsonPropertyName("finish_reason")]
    public required string FinishReason { get; init; }
}

public sealed class OpenAiUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

public sealed class OpenAiChatChunkResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required OpenAiChatChunkChoice[] Choices { get; init; }
}

public sealed class OpenAiChatChunkChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("delta")]
    public required OpenAiChatDelta Delta { get; init; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

public sealed class OpenAiChatDelta
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; init; }
}

public sealed class OllamaBenchRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    [JsonPropertyName("options")]
    public required OllamaBenchOptions Options { get; init; }
}

public sealed class OllamaBenchOptions
{
    [JsonPropertyName("num_predict")]
    public int NumPredict { get; init; }

    [JsonPropertyName("temperature")]
    public double Temperature { get; init; }
}

[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(OllamaTagsResponse))]
[JsonSerializable(typeof(OllamaGenerateResponse))]
[JsonSerializable(typeof(OllamaChatResponse))]
[JsonSerializable(typeof(OpenAiModelsResponse))]
[JsonSerializable(typeof(OpenAiChatCompletionResponse))]
[JsonSerializable(typeof(OpenAiChatChunkResponse))]
[JsonSerializable(typeof(OllamaBenchRequest))]
public partial class InferenceCliJsonContext : JsonSerializerContext
{
}
