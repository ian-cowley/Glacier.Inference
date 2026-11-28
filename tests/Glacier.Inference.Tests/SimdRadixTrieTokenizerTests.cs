namespace Glacier.Inference.Tests;

using System;
using System.Text;
using Glacier.Inference.Tokenizer;
using Xunit;

public class SimdRadixTrieTokenizerTests
{
    [Fact]
    public void CreateDefault_EncodesAndDecodesRoundTrip()
    {
        var tokenizer = SimdRadixTrieTokenizer.CreateDefault();
        Assert.True(tokenizer.VocabularySize >= 256);

        const string prompt = "System: Write C# code using JSON and HTTP.\n  return \"hello\";";
        byte[] utf8 = Encoding.UTF8.GetBytes(prompt);
        int[] tokens = new int[utf8.Length];

        int tokenCount = tokenizer.Encode(utf8, tokens);
        Assert.True(tokenCount > 0 && tokenCount < utf8.Length,
            $"Expected multi-byte subword compression: {tokenCount} tokens vs {utf8.Length} bytes");
        Assert.Equal(tokenCount, tokenizer.CountTokens(utf8));

        byte[] decoded = new byte[utf8.Length * 2];
        int decodedLen = tokenizer.Decode(tokens.AsSpan(0, tokenCount), decoded);
        Assert.Equal(prompt, Encoding.UTF8.GetString(decoded, 0, decodedLen));
        Assert.Equal(prompt, tokenizer.DecodeToString(tokens.AsSpan(0, tokenCount)));
    }

    [Fact]
    public void Encode_LongestMatchWins_OverShorterPrefixes()
    {
        var builder = new SimdRadixTrieBuilder();
        int idHe = builder.AddToken("he");
        int idHell = builder.AddToken("hell");
        int idHello = builder.AddToken("hello");
        var tokenizer = builder.Build();

        byte[] utf8 = Encoding.UTF8.GetBytes("hello hell he");
        int[] tokens = new int[32];
        int count = tokenizer.Encode(utf8, tokens);

        Assert.Equal(5, count);
        Assert.Equal(idHello, tokens[0]);
        Assert.Equal((byte)' ', tokens[1]);
        Assert.Equal(idHell, tokens[2]);
        Assert.Equal((byte)' ', tokens[3]);
        Assert.Equal(idHe, tokens[4]);
    }

    [Fact]
    public void Encode_OverflowFanoutAbove32Children_NeverDropsEdges()
    {
        var builder = new SimdRadixTrieBuilder();
        int[] expectedIds = new int[60];
        for (int i = 0; i < 60; i++)
        {
            byte second = (byte)(i + 1); // 1..60 (includes ASCII & control bytes, avoids byte 0)
            expectedIds[i] = builder.AddToken([(byte)'x', second]);
        }

        var tokenizer = builder.Build();
        int rootX = tokenizer.GetRootNodeIndex((byte)'x');
        Assert.True(rootX >= 0);
        Assert.Equal(60, tokenizer.GetNodeEdgeCount(rootX));

        int[] tokens = new int[4];
        for (int i = 0; i < 60; i++)
        {
            byte second = (byte)(i + 1);
            ReadOnlySpan<byte> pair = [(byte)'x', second];
            int count = tokenizer.Encode(pair, tokens);
            Assert.Equal(1, count);
            Assert.Equal(expectedIds[i], tokens[0]);
            Assert.Equal(1, tokenizer.CountTokens(pair));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(48)]
    public void FindChildVector128_MatchesFindChildScalar_AcrossEdgeCountBoundaries(int edgeCount)
    {
        var builder = new SimdRadixTrieBuilder();
        for (int i = 0; i < edgeCount; i++)
        {
            builder.AddToken([(byte)'p', (byte)(i + 10)]);
        }

        var tokenizer = builder.Build();
        int nodeIdx = tokenizer.GetRootNodeIndex((byte)'p');
        Assert.Equal(edgeCount, tokenizer.GetNodeEdgeCount(nodeIdx));

        int simdCount = Math.Min(32, edgeCount);
        for (int b = 0; b < 256; b++)
        {
            int expected = tokenizer.FindChildScalar(nodeIdx, (byte)b, simdCount);
            int actual = tokenizer.FindChildVector128(nodeIdx, (byte)b, simdCount);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Encode_MultiByteUtf8Fallback_RoundTripsCleanly()
    {
        var tokenizer = SimdRadixTrieTokenizer.CreateDefault();
        const string multiByte = "Café 🚀 αβγ 日本語";
        byte[] utf8 = Encoding.UTF8.GetBytes(multiByte);
        int[] tokens = new int[utf8.Length];

        int count = tokenizer.Encode(utf8, tokens);
        Assert.True(count > 0);
        Assert.Equal(count, tokenizer.CountTokens(utf8));
        Assert.Equal(multiByte, tokenizer.DecodeToString(tokens.AsSpan(0, count)));
    }

    [Fact]
    public void Encode_And_Decode_DestinationTooSmall_ThrowsArgumentException()
    {
        var tokenizer = SimdRadixTrieTokenizer.CreateDefault();
        byte[] utf8 = Encoding.UTF8.GetBytes("abcdef");
        int[] tinyTokens = new int[1];

        Assert.Throws<ArgumentException>(() => tokenizer.Encode(utf8, tinyTokens));

        int[] fullTokens = new int[utf8.Length];
        int count = tokenizer.Encode(utf8, fullTokens);
        byte[] tinyBytes = new byte[1];

        Assert.Throws<ArgumentException>(() => tokenizer.Decode(fullTokens.AsSpan(0, count), tinyBytes));
    }

    [Fact]
    public void Encode_EmptyInput_ReturnsZeroTokens()
    {
        var tokenizer = SimdRadixTrieTokenizer.CreateDefault();
        Assert.Equal(0, tokenizer.Encode(ReadOnlySpan<byte>.Empty, Span<int>.Empty));
        Assert.Equal(0, tokenizer.CountTokens(ReadOnlySpan<byte>.Empty));
        Assert.Equal(0, tokenizer.Decode(ReadOnlySpan<int>.Empty, Span<byte>.Empty));
        Assert.Equal(string.Empty, tokenizer.DecodeToString(ReadOnlySpan<int>.Empty));
    }
}
