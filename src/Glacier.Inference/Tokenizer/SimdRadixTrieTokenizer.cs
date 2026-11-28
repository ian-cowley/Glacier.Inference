namespace Glacier.Inference.Tokenizer;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text;

/// <summary>
/// High-throughput SIMD-accelerated compressed Radix Trie Tokenizer.
/// Supports full 256-way byte branching per node with zero silent edge drops and zero-allocation Span encoding/decoding.
/// </summary>
public sealed class SimdRadixTrieTokenizer : ITokenizer
{
    private readonly int[] _rootTable = new int[256];
    private readonly int[] _byteFallbackTokens = new int[256];
    private readonly int[] _nodeTokenIds;
    private readonly Vector256<byte>[] _nodeBranchVectors;
    private readonly ushort[] _nodeEdgeCounts;
    private readonly byte[][] _nodeOverflowKeys;
    private readonly int[][] _nodeChildIndices;
    private readonly byte[][] _nodeEdgePrefixes;
    private readonly byte[][] _tokenToBytes;

    public int VocabularySize { get; }

    internal SimdRadixTrieTokenizer(
        int[] rootTable,
        int[] byteFallbackTokens,
        int[] nodeTokenIds,
        Vector256<byte>[] nodeBranchVectors,
        ushort[] nodeEdgeCounts,
        byte[][] nodeOverflowKeys,
        int[][] nodeChildIndices,
        byte[][] nodeEdgePrefixes,
        byte[][] tokenToBytes,
        int vocabSize)
    {
        Array.Copy(rootTable, _rootTable, 256);
        Array.Copy(byteFallbackTokens, _byteFallbackTokens, 256);
        _nodeTokenIds = nodeTokenIds;
        _nodeBranchVectors = nodeBranchVectors;
        _nodeEdgeCounts = nodeEdgeCounts;
        _nodeOverflowKeys = nodeOverflowKeys;
        _nodeChildIndices = nodeChildIndices;
        _nodeEdgePrefixes = nodeEdgePrefixes;
        _tokenToBytes = tokenToBytes;
        VocabularySize = vocabSize;
    }

    public static SimdRadixTrieTokenizer CreateDefault()
    {
        var builder = new SimdRadixTrieBuilder();
        builder.AddDefaultVocabulary();
        return builder.Build();
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int Encode(ReadOnlySpan<byte> utf8Text, Span<int> destinationTokens)
    {
        int inputLen = utf8Text.Length;
        int tokenCount = 0;
        int pos = 0;

        while (pos < inputLen)
        {
            byte b = utf8Text[pos];
            int nodeIdx = _rootTable[b];

            if (nodeIdx < 0)
            {
                if (tokenCount >= destinationTokens.Length)
                {
                    ThrowDestinationTooSmall();
                }
                destinationTokens[tokenCount++] = _byteFallbackTokens[b];
                pos++;
                continue;
            }

            int bestTokenId = _nodeTokenIds[nodeIdx];
            int bestLength = 1;
            int currentLength = 1;
            int currentNode = nodeIdx;

            while (pos + currentLength < inputLen)
            {
                byte nextByte = utf8Text[pos + currentLength];
                int nextNode = FindChild(currentNode, nextByte);
                if (nextNode < 0)
                {
                    break;
                }

                byte[] prefix = _nodeEdgePrefixes[nextNode];
                if (prefix.Length > 0)
                {
                    int remaining = inputLen - (pos + currentLength + 1);
                    if (remaining < prefix.Length)
                    {
                        break;
                    }

                    ReadOnlySpan<byte> inputSlice = utf8Text.Slice(pos + currentLength + 1, prefix.Length);
                    if (!inputSlice.SequenceEqual(prefix))
                    {
                        break;
                    }
                    currentLength += 1 + prefix.Length;
                }
                else
                {
                    currentLength++;
                }

                currentNode = nextNode;
                int candidateId = _nodeTokenIds[currentNode];
                if (candidateId >= 0)
                {
                    bestTokenId = candidateId;
                    bestLength = currentLength;
                }
            }

            if (tokenCount >= destinationTokens.Length)
            {
                ThrowDestinationTooSmall();
            }

            destinationTokens[tokenCount++] = bestTokenId >= 0 ? bestTokenId : _byteFallbackTokens[b];
            pos += bestLength;
        }

        return tokenCount;
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int CountTokens(ReadOnlySpan<byte> utf8Text)
    {
        int inputLen = utf8Text.Length;
        int tokenCount = 0;
        int pos = 0;

        while (pos < inputLen)
        {
            byte b = utf8Text[pos];
            int nodeIdx = _rootTable[b];

            if (nodeIdx < 0)
            {
                tokenCount++;
                pos++;
                continue;
            }

            int bestLength = 1;
            int currentLength = 1;
            int currentNode = nodeIdx;

            while (pos + currentLength < inputLen)
            {
                byte nextByte = utf8Text[pos + currentLength];
                int nextNode = FindChild(currentNode, nextByte);
                if (nextNode < 0)
                {
                    break;
                }

                byte[] prefix = _nodeEdgePrefixes[nextNode];
                if (prefix.Length > 0)
                {
                    int remaining = inputLen - (pos + currentLength + 1);
                    if (remaining < prefix.Length)
                    {
                        break;
                    }

                    ReadOnlySpan<byte> inputSlice = utf8Text.Slice(pos + currentLength + 1, prefix.Length);
                    if (!inputSlice.SequenceEqual(prefix))
                    {
                        break;
                    }
                    currentLength += 1 + prefix.Length;
                }
                else
                {
                    currentLength++;
                }

                currentNode = nextNode;
                if (_nodeTokenIds[currentNode] >= 0)
                {
                    bestLength = currentLength;
                }
            }

            tokenCount++;
            pos += bestLength;
        }

        return tokenCount;
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int Decode(ReadOnlySpan<int> tokens, Span<byte> destinationUtf8)
    {
        int written = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int tokenId = tokens[i];
            if ((uint)tokenId >= (uint)_tokenToBytes.Length)
            {
                continue;
            }

            byte[] bytes = _tokenToBytes[tokenId];
            if (written + bytes.Length > destinationUtf8.Length)
            {
                ThrowDestinationTooSmall();
            }

            bytes.CopyTo(destinationUtf8.Slice(written));
            written += bytes.Length;
        }

        return written;
    }

    public string DecodeToString(ReadOnlySpan<int> tokens)
    {
        int totalBytes = 0;
        for (int i = 0; i < tokens.Length; i++)
        {
            int id = tokens[i];
            if ((uint)id < (uint)_tokenToBytes.Length)
            {
                totalBytes += _tokenToBytes[id].Length;
            }
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(Math.Max(1, totalBytes));
        try
        {
            int written = Decode(tokens, rented);
            return Encoding.UTF8.GetString(rented, 0, written);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int FindChild(int nodeIdx, byte b)
    {
        int edgeCount = _nodeEdgeCounts[nodeIdx];
        if (edgeCount == 0)
        {
            return -1;
        }

        int simdCount = edgeCount > 32 ? 32 : edgeCount;
        if (Vector256.IsHardwareAccelerated)
        {
            Vector256<byte> branchKeys = _nodeBranchVectors[nodeIdx];
            Vector256<byte> target = Vector256.Create(b);
            Vector256<byte> matches = Vector256.Equals(branchKeys, target);
            uint mask = matches.ExtractMostSignificantBits();
            // Avoid C# 32-bit shift count wraparound ((1u << 32) == 1u)!
            uint validMask = simdCount == 32 ? mask : (mask & ((1u << simdCount) - 1u));
            if (validMask != 0)
            {
                int matchIndex = BitOperations.TrailingZeroCount(validMask);
                return _nodeChildIndices[nodeIdx][matchIndex];
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            int idx = FindChildVector128(nodeIdx, b, simdCount);
            if (idx >= 0)
            {
                return idx;
            }
        }
        else
        {
            int idx = FindChildScalar(nodeIdx, b, simdCount);
            if (idx >= 0)
            {
                return idx;
            }
        }

        if (edgeCount > 32)
        {
            int overflowIdx = _nodeOverflowKeys[nodeIdx].AsSpan().IndexOf(b);
            if (overflowIdx >= 0)
            {
                return _nodeChildIndices[nodeIdx][32 + overflowIdx];
            }
        }

        return -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal int FindChildVector128(int nodeIdx, byte b, int simdCount)
    {
        Vector256<byte> branchKeys = _nodeBranchVectors[nodeIdx];
        Vector128<byte> target = Vector128.Create(b);
        uint loMask = Vector128.Equals(branchKeys.GetLower(), target).ExtractMostSignificantBits();
        uint hiMask = simdCount > 16
            ? Vector128.Equals(branchKeys.GetUpper(), target).ExtractMostSignificantBits()
            : 0u;
        uint mask = loMask | (hiMask << 16);
        uint validMask = simdCount == 32 ? mask : (mask & ((1u << simdCount) - 1u));
        if (validMask != 0)
        {
            int matchIndex = BitOperations.TrailingZeroCount(validMask);
            return _nodeChildIndices[nodeIdx][matchIndex];
        }
        return -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal int FindChildScalar(int nodeIdx, byte b, int simdCount)
    {
        Vector256<byte> branchKeys = _nodeBranchVectors[nodeIdx];
        int[] children = _nodeChildIndices[nodeIdx];
        for (int i = 0; i < simdCount; i++)
        {
            if (branchKeys[i] == b)
            {
                return children[i];
            }
        }
        return -1;
    }

    internal int GetRootNodeIndex(byte b) => _rootTable[b];
    internal int GetNodeEdgeCount(int nodeIdx) => _nodeEdgeCounts[nodeIdx];

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowDestinationTooSmall()
    {
        throw new ArgumentException("Destination buffer is too small to receive the output.", "destination");
    }
}
