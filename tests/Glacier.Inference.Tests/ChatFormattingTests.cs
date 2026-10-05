using System.Collections.Generic;
using Glacier.Inference.Tokenizer;
using Xunit;

namespace Glacier.Inference.Tests;

public class ChatFormattingTests
{
    [Fact]
    public void FormatChat_ChatML_GeneratesExpectedTemplate()
    {
        var vocab = new[] { "<|im_start|>", "<|im_end|>", "system", "user", "assistant" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 1, bosTokenId: 0);

        var messages = new List<(string Role, string Content)>
        {
            ("system", "You are an assistant."),
            ("user", "Hello!"),
            ("assistant", "Hi there!"),
            ("user", "How are you?")
        };

        string formatted = tokenizer.FormatChat(messages);

        Assert.Contains("<|im_start|>system\nYou are an assistant.<|im_end|>", formatted);
        Assert.Contains("<|im_start|>user\nHello!<|im_end|>", formatted);
        Assert.Contains("<|im_start|>assistant\nHi there!<|im_end|>", formatted);
        Assert.EndsWith("<|im_start|>assistant\n", formatted);
    }

    [Fact]
    public void FormatChat_Llama3_GeneratesExpectedTemplate()
    {
        var vocab = new[] { "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 3, bosTokenId: 0);

        var messages = new List<(string Role, string Content)>
        {
            ("system", "System note"),
            ("user", "User question")
        };

        string formatted = tokenizer.FormatChat(messages);

        Assert.StartsWith("<|begin_of_text|>", formatted);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>\n\nSystem note<|eot_id|>", formatted);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nUser question<|eot_id|>", formatted);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", formatted);
    }

    [Fact]
    public void FormatChat_Gemma_GeneratesExpectedTemplate()
    {
        var vocab = new[] { "<start_of_turn>", "<end_of_turn>", "<bos>" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 1, bosTokenId: 2);

        var messages = new List<(string Role, string Content)>
        {
            ("user", "Tell me a joke"),
            ("assistant", "Why did the chicken cross the road?")
        };

        string formatted = tokenizer.FormatChat(messages);

        Assert.Contains("<start_of_turn>user\nTell me a joke<end_of_turn>", formatted);
        Assert.Contains("<start_of_turn>model\nWhy did the chicken cross the road?<end_of_turn>", formatted);
        Assert.EndsWith("<start_of_turn>model\n", formatted);
    }

    [Fact]
    public void FormatChat_Mistral_GeneratesExpectedTemplate()
    {
        var vocab = new[] { "[INST]", "[/INST]", "<s>" };
        var tokenizer = new BpeTokenizer(vocab, eosTokenId: 1, bosTokenId: 2);

        var messages = new List<(string Role, string Content)>
        {
            ("user", "What is the capital of France?")
        };

        string formatted = tokenizer.FormatChat(messages);

        Assert.Contains("[INST] What is the capital of France? [/INST]", formatted);
    }
}
