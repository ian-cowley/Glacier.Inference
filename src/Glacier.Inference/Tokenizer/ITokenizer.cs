namespace Glacier.Inference.Tokenizer;

using System;

/// <summary>
/// High-throughput zero-allocation tokenizer contract.
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Encodes UTF-8 text into token IDs without managed allocations.
    /// Returns the number of tokens written to <paramref name="destinationTokens"/>.
    /// </summary>
    int Encode(ReadOnlySpan<byte> utf8Text, Span<int> destinationTokens);

    /// <summary>
    /// Decodes token IDs back into UTF-8 text without managed allocations.
    /// Returns the number of bytes written to <paramref name="destinationUtf8"/>.
    /// </summary>
    int Decode(ReadOnlySpan<int> tokens, Span<byte> destinationUtf8);

    /// <summary>
    /// Counts the number of tokens in the UTF-8 text without materializing token IDs.
    /// </summary>
    int CountTokens(ReadOnlySpan<byte> utf8Text);
}
