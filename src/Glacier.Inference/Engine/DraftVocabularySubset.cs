namespace Glacier.Inference.Engine;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Represents a sorted subset of full-vocabulary token IDs used by a filtered draft output head
/// (e.g., ASCII/Latin/punctuation-only drafting for English code/chat prompts).
/// </summary>
public sealed class DraftVocabularySubset
{
    private static readonly byte[] Magic = "GDVB"u8.ToArray();
    private const int BinaryVersion = 1;

    private readonly int[] _subsetTokenIds;

    public int FullVocabSize { get; }
    public int SubsetSize => _subsetTokenIds.Length;
    public ReadOnlyMemory<int> SubsetTokenIds => _subsetTokenIds;

    private DraftVocabularySubset(int fullVocabSize, int[] sortedUniqueTokenIds)
    {
        FullVocabSize = fullVocabSize;
        _subsetTokenIds = sortedUniqueTokenIds;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int MapSubsetIndexToTokenId(int subsetIndex) => _subsetTokenIds[subsetIndex];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int FindSubsetIndex(int tokenId) => Array.BinarySearch(_subsetTokenIds, tokenId);

    public bool ContainsTokenId(int tokenId) => FindSubsetIndex(tokenId) >= 0;

    /// <summary>
    /// Builds a normalized, deduplicated ascending <see cref="DraftVocabularySubset"/> from arbitrary token IDs.
    /// </summary>
    public static DraftVocabularySubset FromTokenIds(int fullVocabSize, ReadOnlySpan<int> tokenIds)
    {
        if (fullVocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(fullVocabSize), "Full vocabulary size must be positive.");

        var unique = new HashSet<int>(tokenIds.Length);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            if ((uint)id >= (uint)fullVocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), id, $"Token ID must be in [0, {fullVocabSize}).");
            unique.Add(id);
        }

        if (unique.Count == 0)
            throw new ArgumentException("Draft vocabulary subset must contain at least one token ID.", nameof(tokenIds));

        int[] sorted = new int[unique.Count];
        unique.CopyTo(sorted);
        Array.Sort(sorted);
        return new DraftVocabularySubset(fullVocabSize, sorted);
    }

    /// <summary>
    /// Filters a <see cref="BpeTokenizer"/> vocabulary for lightweight draft-head projection.
    /// Default filtering keeps ASCII, whitespace, Latin-1/Latin Extended, General Punctuation,
    /// special/control/stop tokens, and single-byte fallback tokens while excluding CJK/Kana/Hangul blocks
    /// unless listed in <paramref name="mandatoryTokenIds"/>.
    /// </summary>
    public static DraftVocabularySubset FromLanguageFilter(
        BpeTokenizer tokenizer,
        Func<string, int, bool>? includePredicate = null,
        ReadOnlySpan<int> mandatoryTokenIds = default)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        int vocabSize = tokenizer.VocabSize;
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenizer), "Tokenizer vocabulary must be non-empty.");

        var unique = new HashSet<int>(Math.Min(vocabSize, 65536));

        for (int i = 0; i < mandatoryTokenIds.Length; i++)
        {
            int id = mandatoryTokenIds[i];
            if ((uint)id < (uint)vocabSize)
            {
                unique.Add(id);
            }
        }

        if ((uint)tokenizer.BosTokenId < (uint)vocabSize) unique.Add(tokenizer.BosTokenId);
        if ((uint)tokenizer.EosTokenId < (uint)vocabSize) unique.Add(tokenizer.EosTokenId);
        if ((uint)tokenizer.PadTokenId < (uint)vocabSize) unique.Add(tokenizer.PadTokenId);

        for (int id = 0; id < vocabSize; id++)
        {
            if (unique.Contains(id)) continue;

            if (tokenizer.IsStopToken(id))
            {
                unique.Add(id);
                continue;
            }

            byte[] rawBytes = tokenizer.GetTokenBytes(id);
            string decoded = rawBytes.Length == 0 ? string.Empty : Encoding.UTF8.GetString(rawBytes);

            bool keep = includePredicate != null
                ? includePredicate(decoded, id)
                : IsDefaultLanguageToken(decoded, rawBytes);

            if (keep)
            {
                unique.Add(id);
            }
        }

        if (unique.Count == 0)
        {
            unique.Add(0);
        }

        int[] sorted = new int[unique.Count];
        unique.CopyTo(sorted);
        Array.Sort(sorted);
        return new DraftVocabularySubset(vocabSize, sorted);
    }

    private static bool IsDefaultLanguageToken(string decoded, byte[] rawBytes)
    {
        if (rawBytes.Length == 1)
        {
            // Keep ASCII and single-byte fallback pieces; multi-byte CJK codepoints take 3+ UTF-8 bytes
            return true;
        }

        if (decoded.Length == 0)
        {
            return false;
        }

        for (int i = 0; i < decoded.Length; i++)
        {
            char c = decoded[i];
            if (IsCjkOrHangulOrKana(c))
            {
                return false;
            }

            bool allowed = c <= '\u024F' || (c >= '\u2000' && c <= '\u206F');
            if (!allowed)
            {
                return false;
            }
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsCjkOrHangulOrKana(char c) =>
        (c >= '\u3040' && c <= '\u30FF') || // Hiragana & Katakana
        (c >= '\u3400' && c <= '\u4DBF') || // CJK Unified Ideographs Extension A
        (c >= '\u4E00' && c <= '\u9FFF') || // CJK Unified Ideographs
        (c >= '\uAC00' && c <= '\uD7AF') || // Hangul Syllables
        (c >= '\uF900' && c <= '\uFAFF');   // CJK Compatibility Ideographs

    public void SaveBinary(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
        bw.Write(Magic);
        bw.Write(BinaryVersion);
        bw.Write(FullVocabSize);
        bw.Write(_subsetTokenIds.Length);
        for (int i = 0; i < _subsetTokenIds.Length; i++)
        {
            bw.Write(_subsetTokenIds[i]);
        }
    }

    public static DraftVocabularySubset LoadBinary(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

        Span<byte> header = stackalloc byte[4];
        if (br.Read(header) != 4 || !header.SequenceEqual(Magic))
            throw new InvalidDataException("Invalid DraftVocabularySubset header magic.");

        int version = br.ReadInt32();
        if (version != BinaryVersion)
            throw new InvalidDataException($"Unsupported DraftVocabularySubset binary version {version}.");

        int fullVocabSize = br.ReadInt32();
        int subsetSize = br.ReadInt32();
        if (fullVocabSize <= 0 || subsetSize <= 0 || subsetSize > fullVocabSize)
            throw new InvalidDataException($"Corrupt DraftVocabularySubset header: full={fullVocabSize}, subset={subsetSize}.");

        int[] tokenIds = new int[subsetSize];
        for (int i = 0; i < subsetSize; i++)
        {
            tokenIds[i] = br.ReadInt32();
        }

        return FromTokenIds(fullVocabSize, tokenIds);
    }
}

/// <summary>
/// Contiguous 64-byte aligned packed projection head covering only rows in a <see cref="DraftVocabularySubset"/>.
/// Cuts draft LM-head matrix-vector bandwidth dramatically without losing full-vocabulary ID mapping.
/// </summary>
public sealed unsafe class PackedDraftOutputHead : IDisposable
{
    private readonly GgufType _weightType;
    private readonly int _dim;
    private readonly int _fullVocabSize;
    private readonly DraftVocabularySubset _subset;
    private readonly int _subsetSize;
    private readonly long _rowBytes;

    private byte* _packedWeights;
    private float* _subsetLogits;
    private float* _xSums;
    private readonly float[] _subsetProbsScratch;
    private bool _disposed;

    public DraftVocabularySubset Subset => _subset;
    public int Dim => _dim;
    public int FullVocabSize => _fullVocabSize;
    public int SubsetSize => _subsetSize;

    public PackedDraftOutputHead(
        GgufType weightType,
        byte* fullOutWeight,
        int dim,
        int fullVocabSize,
        DraftVocabularySubset subset)
    {
        if (fullOutWeight == null) throw new ArgumentNullException(nameof(fullOutWeight));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (fullVocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(fullVocabSize));
        ArgumentNullException.ThrowIfNull(subset);
        if (subset.FullVocabSize != fullVocabSize)
            throw new ArgumentException($"Subset FullVocabSize ({subset.FullVocabSize}) does not match head vocab size ({fullVocabSize}).", nameof(subset));

        _weightType = weightType;
        _dim = dim;
        _fullVocabSize = fullVocabSize;
        _subset = subset;
        _subsetSize = subset.SubsetSize;
        _rowBytes = GgufTypes.GetRowBytes(weightType, dim);

        long totalBytes = Math.Max(64L, (long)_subsetSize * _rowBytes);
        _packedWeights = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);

        ReadOnlySpan<int> ids = subset.SubsetTokenIds.Span;
        ulong rowU = (ulong)_rowBytes;
        for (int i = 0; i < _subsetSize; i++)
        {
            int tokenId = ids[i];
            byte* srcRow = fullOutWeight + (long)tokenId * _rowBytes;
            byte* dstRow = _packedWeights + (long)i * _rowBytes;
            Buffer.MemoryCopy(srcRow, dstRow, rowU, rowU);
        }

        _subsetLogits = (float*)NativeMemory.AlignedAlloc((nuint)((long)_subsetSize * sizeof(float)), 64);
        int chunks = Math.Max(1, (dim + 31) / 32);
        _xSums = (float*)NativeMemory.AlignedAlloc((nuint)((long)chunks * sizeof(float)), 64);
        _subsetProbsScratch = new float[_subsetSize];
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ComputeSubsetLogits(float* normedHidden, Span<float> subsetLogitsOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (subsetLogitsOut.Length < _subsetSize)
            throw new ArgumentException($"Destination span too small ({subsetLogitsOut.Length} < {_subsetSize}).", nameof(subsetLogitsOut));

        ComputeSubsetLogitsInternal(normedHidden);
        new ReadOnlySpan<float>(_subsetLogits, _subsetSize).CopyTo(subsetLogitsOut);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeSubsetLogitsInternal(float* normedHidden)
    {
        float* xSums = null;
        if (_dim >= 32)
        {
            QuantKernels.ComputeBlockSums32(normedHidden, _xSums, _dim);
            xSums = _xSums;
        }

        QuantKernels.MatVecMul(_weightType, _packedWeights, normedHidden, _subsetLogits, _dim, _subsetSize, xSums);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetAllowedSubsetCount(int maxTokenIdExclusive)
    {
        if (maxTokenIdExclusive <= 0 || maxTokenIdExclusive >= _fullVocabSize)
            return _subsetSize;

        ReadOnlySpan<int> ids = _subset.SubsetTokenIds.Span;
        int lo = 0;
        int hi = _subsetSize;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (ids[mid] < maxTokenIdExclusive)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int SampleGreedy(float* normedHidden, int maxTokenIdExclusive = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int allowedCount = GetAllowedSubsetCount(maxTokenIdExclusive);
        if (allowedCount == 0)
            return 0;

        ComputeSubsetLogitsInternal(normedHidden);

        int bestIdx = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < allowedCount; i++)
        {
            float v = _subsetLogits[i];
            if (v > bestVal)
            {
                bestVal = v;
                bestIdx = i;
            }
        }
        return _subset.MapSubsetIndexToTokenId(bestIdx);
    }

    /// <summary>
    /// Computes normalized filtered draft probabilities over subset rows, scatters them into
    /// <paramref name="fullVocabProbsOut"/> (zeros outside subset), and samples a mapped full vocabulary token ID.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int SampleWithProbabilities(
        float* normedHidden,
        SamplingOptions sampling,
        Sampler sampler,
        ReadOnlySpan<int> recentTokens,
        Span<float> fullVocabProbsOut,
        int maxTokenIdExclusive = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentNullException.ThrowIfNull(sampler);
        if (fullVocabProbsOut.Length < _fullVocabSize)
            throw new ArgumentException($"fullVocabProbsOut too small ({fullVocabProbsOut.Length} < {_fullVocabSize}).", nameof(fullVocabProbsOut));

        int allowedCount = GetAllowedSubsetCount(maxTokenIdExclusive);
        fullVocabProbsOut.Slice(0, _fullVocabSize).Clear();
        if (allowedCount == 0)
        {
            fullVocabProbsOut[0] = 1.0f;
            return 0;
        }

        ComputeSubsetLogitsInternal(normedHidden);

        Span<float> subLogits = new Span<float>(_subsetLogits, allowedCount);
        Sampler.SanitizeLogits(subLogits);

        float effectivePenalty = sampling.RepetitionPenalty;
        ReadOnlySpan<int> penaltyWindow = recentTokens;
        if (sampling.EnableReasoningLoopGuard && !recentTokens.IsEmpty)
        {
            int loopPeriod = Sampler.DetectPeriodicReasoningLoop(recentTokens, minPeriod: sampling.ReasoningLoopMinPeriod);
            if (loopPeriod > 0)
            {
                effectivePenalty = Math.Max(effectivePenalty, sampling.LoopRecoveryRepetitionPenalty);
                int windowLen = Math.Min(recentTokens.Length, loopPeriod * 3);
                penaltyWindow = recentTokens.Slice(recentTokens.Length - windowLen, windowLen);
            }
        }

        if (effectivePenalty != 1.0f && !penaltyWindow.IsEmpty)
        {
            int[]? rented = null;
            Span<int> mappedRecent = penaltyWindow.Length <= 256
                ? stackalloc int[penaltyWindow.Length]
                : (rented = ArrayPool<int>.Shared.Rent(penaltyWindow.Length)).AsSpan(0, penaltyWindow.Length);

            try
            {
                int mappedCount = 0;
                for (int i = 0; i < penaltyWindow.Length; i++)
                {
                    int subIdx = _subset.FindSubsetIndex(penaltyWindow[i]);
                    if ((uint)subIdx < (uint)allowedCount)
                    {
                        mappedRecent[mappedCount++] = subIdx;
                    }
                }

                if (mappedCount > 0)
                {
                    Sampler.ApplyRepetitionPenalty(subLogits, mappedRecent.Slice(0, mappedCount), effectivePenalty);
                }
            }
            finally
            {
                if (rented != null) ArrayPool<int>.Shared.Return(rented);
            }
        }

        var subsetSampling = new SamplingOptions
        {
            Temperature = sampling.Temperature,
            TopP = sampling.TopP,
            TopK = sampling.TopK,
            RepetitionPenalty = 1.0f,
            EnableReasoningLoopGuard = false,
            MaxTokens = sampling.MaxTokens
        };

        Span<float> allowedProbs = _subsetProbsScratch.AsSpan(0, allowedCount);
        Sampler.ComputeFilteredDistribution(subLogits, subsetSampling, ReadOnlySpan<int>.Empty, allowedProbs);

        ReadOnlySpan<int> subsetIds = _subset.SubsetTokenIds.Span;
        for (int i = 0; i < allowedCount; i++)
        {
            float p = allowedProbs[i];
            if (p > 0f)
            {
                fullVocabProbsOut[subsetIds[i]] = p;
            }
        }

        int chosenSub = Sampler.SampleFromDistribution(allowedProbs, sampler.Rng);
        return _subset.MapSubsetIndexToTokenId(chosenSub);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_packedWeights != null) { NativeMemory.AlignedFree(_packedWeights); _packedWeights = null; }
        if (_subsetLogits != null) { NativeMemory.AlignedFree(_subsetLogits); _subsetLogits = null; }
        if (_xSums != null) { NativeMemory.AlignedFree(_xSums); _xSums = null; }
    }
}
