namespace Glacier.Inference.Engine;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Glacier.Inference.Sampling;

/// <summary>
/// Speculative draft provider that uses a secondary smaller model (or CPU draft model)
/// to autoregressively propose candidate tokens.
/// </summary>
public sealed class ModelDraftProvider : IDraftProvider
{
    private readonly ISpeculativeTarget _session;
    private readonly bool _ownsSession;
    private readonly List<int> _cachedTokens = new();
    private float[]? _logitScratch;
    private float[]? _draftVocabProbsScratch;

    public ISpeculativeTarget Session => _session;

    public ModelDraftProvider(ISpeculativeTarget session, bool ownsSession = false)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _ownsSession = ownsSession;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _cachedTokens.Clear();
        _session.ResetKvCache();
    }

    /// <summary>
    /// Synchronizes <see cref="_session"/>'s KV cache and current logits to match <paramref name="tokens"/>
    /// using exact token-by-token prefix comparison, avoiding cross-prompt collisions and unnecessary re-prefills.
    /// </summary>
    private void SyncSessionToContext(ReadOnlySpan<int> tokens)
    {
        ReadOnlySpan<int> cached = CollectionsMarshal.AsSpan(_cachedTokens);
        int maxCommon = Math.Min(cached.Length, tokens.Length);
        int lcp = 0;
        while (lcp < maxCommon && cached[lcp] == tokens[lcp])
        {
            lcp++;
        }

        // 1. Exact sequence match: KV cache and current logits already belong to tokens[^1]
        if (lcp > 0 && lcp == cached.Length && lcp == tokens.Length)
        {
            return;
        }

        // 2. Strict append onto cached sequence (e.g., 100% draft acceptance + bonus token)
        if (lcp > 0 && lcp == cached.Length && lcp < tokens.Length &&
            (_session.IsRingBufferEnabled || tokens.Length <= _session.MaxSeqLen))
        {
            for (int idx = lcp; idx < tokens.Length; idx++)
            {
                bool isLast = idx == tokens.Length - 1;
                _session.ForwardToken(tokens[idx], idx, computeLogits: isLast);
                _cachedTokens.Add(tokens[idx]);
            }
            return;
        }

        // 3. Partial draft acceptance / rewind on a pure causal KV-cache session
        if (lcp > 0 && lcp < cached.Length && _session.CanOverwriteKvFromPrefix &&
            tokens.Length <= _session.MaxSeqLen)
        {
            _cachedTokens.RemoveRange(lcp, _cachedTokens.Count - lcp);
            if (lcp < tokens.Length)
            {
                for (int idx = lcp; idx < tokens.Length; idx++)
                {
                    bool isLast = idx == tokens.Length - 1;
                    _session.ForwardToken(tokens[idx], idx, computeLogits: isLast);
                    _cachedTokens.Add(tokens[idx]);
                }
            }
            else
            {
                int lastIdx = tokens.Length - 1;
                _session.ForwardToken(tokens[lastIdx], lastIdx, computeLogits: true);
            }
            return;
        }

        // 4. Empty cache, cross-prompt divergence (lcp == 0), or stateful/recurrent session
        _session.ResetKvCache();
        _session.Prefill(tokens);
        _cachedTokens.Clear();
        for (int i = 0; i < tokens.Length; i++)
        {
            _cachedTokens.Add(tokens[i]);
        }
    }

    /// <inheritdoc />
    public int Draft(ReadOnlySpan<int> tokens, int maxDraftTokens, Span<int> draftTokens)
    {
        if (tokens.IsEmpty || maxDraftTokens <= 0 || draftTokens.IsEmpty)
        {
            return 0;
        }

        int maxAvailable = _session.IsRingBufferEnabled
            ? maxDraftTokens
            : (_session.MaxSeqLen - tokens.Length);
        int count = Math.Min(Math.Min(maxDraftTokens, draftTokens.Length), maxAvailable);
        if (count <= 0)
        {
            return 0;
        }

        SyncSessionToContext(tokens);

        int pos = tokens.Length;

        // The first draft token is sampled from the logits already computed for tokens[^1] during SyncSessionToContext
        int nextToken = _session.SampleNextToken(SamplingOptions.Greedy);
        draftTokens[0] = nextToken;

        for (int i = 1; i < count; i++)
        {
            _session.ForwardToken(nextToken, pos, computeLogits: true);
            _cachedTokens.Add(nextToken);
            pos++;
            nextToken = _session.SampleNextToken(SamplingOptions.Greedy);
            draftTokens[i] = nextToken;
        }

        // Forward the final drafted token if within MaxSeqLen so the draft model's KV cache and logits are ready
        if (_session.IsRingBufferEnabled || pos < _session.MaxSeqLen)
        {
            _session.ForwardToken(nextToken, pos, computeLogits: true);
            _cachedTokens.Add(nextToken);
        }
        else
        {
            _cachedTokens.Clear();
        }
        return count;
    }

    /// <inheritdoc />
    public int DraftWithProbabilities(
        ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> recentTokens,
        int maxDraftTokens,
        SamplingOptions sampling,
        Sampler sampler,
        int targetVocabSize,
        Span<int> draftTokens,
        Span<float> draftProbs)
    {
        if (!_session.SupportsLogitVerification)
        {
            int fallbackDrafted = Draft(tokens, maxDraftTokens, draftTokens);
            if (fallbackDrafted <= 0) return 0;
            draftProbs.Slice(0, fallbackDrafted * targetVocabSize).Clear();
            for (int i = 0; i < fallbackDrafted; i++)
            {
                int tok = draftTokens[i];
                if ((uint)tok < (uint)targetVocabSize)
                    draftProbs[i * targetVocabSize + tok] = 1.0f;
            }
            return fallbackDrafted;
        }

        if (tokens.IsEmpty || maxDraftTokens <= 0 || draftTokens.IsEmpty)
        {
            return 0;
        }

        int maxAvailable = _session.IsRingBufferEnabled
            ? maxDraftTokens
            : (_session.MaxSeqLen - tokens.Length);
        int count = Math.Min(Math.Min(maxDraftTokens, draftTokens.Length), maxAvailable);
        if (count <= 0)
        {
            return 0;
        }

        SyncSessionToContext(tokens);

        int draftVocabSize = _session.VocabSize;
        if (_logitScratch == null || _logitScratch.Length < draftVocabSize)
        {
            _logitScratch = new float[draftVocabSize];
        }

        int recentCapacity = recentTokens.Length + count;
        int[]? rentedRecent = null;
        Span<int> recentBuf = recentCapacity <= 256
            ? stackalloc int[recentCapacity]
            : (rentedRecent = ArrayPool<int>.Shared.Rent(recentCapacity)).AsSpan(0, recentCapacity);

        try
        {
            recentTokens.CopyTo(recentBuf);
            int recentLen = recentTokens.Length;
            int pos = tokens.Length;
            int actualDrafted = 0;
            bool finalAlreadyForwarded = false;

            for (int i = 0; i < count; i++)
            {
                if (i > 0)
                {
                    _session.ForwardToken(draftTokens[i - 1], pos, computeLogits: true);
                    _cachedTokens.Add(draftTokens[i - 1]);
                    pos++;
                }

                _session.CopyCurrentLogits(_logitScratch.AsSpan(0, draftVocabSize));
                Span<float> qTargetRow = draftProbs.Slice(i * targetVocabSize, targetVocabSize);

                int nextToken;
                if (draftVocabSize == targetVocabSize)
                {
                    Sampler.ComputeFilteredDistribution(
                        _logitScratch.AsSpan(0, draftVocabSize),
                        sampling,
                        recentBuf.Slice(0, recentLen),
                        qTargetRow);
                    nextToken = Sampler.SampleFromDistribution(qTargetRow, sampler.Rng);
                }
                else
                {
                    if (_draftVocabProbsScratch == null || _draftVocabProbsScratch.Length < draftVocabSize)
                    {
                        _draftVocabProbsScratch = new float[draftVocabSize];
                    }
                    Span<float> qDraft = _draftVocabProbsScratch.AsSpan(0, draftVocabSize);
                    Sampler.ComputeFilteredDistribution(
                        _logitScratch.AsSpan(0, draftVocabSize),
                        sampling,
                        recentBuf.Slice(0, recentLen),
                        qDraft);
                    nextToken = Sampler.SampleFromDistribution(qDraft, sampler.Rng);
                    if ((uint)nextToken >= (uint)targetVocabSize)
                    {
                        if (i > 0) finalAlreadyForwarded = true;
                        break;
                    }
                    qTargetRow.Clear();
                    int sharedVocab = Math.Min(draftVocabSize, targetVocabSize);
                    qDraft.Slice(0, sharedVocab).CopyTo(qTargetRow);
                }

                draftTokens[i] = nextToken;
                recentBuf[recentLen++] = nextToken;
                actualDrafted++;
            }

            if (actualDrafted > 0 && !finalAlreadyForwarded)
            {
                int finalDraft = draftTokens[actualDrafted - 1];
                if (_session.IsRingBufferEnabled || pos < _session.MaxSeqLen)
                {
                    _session.ForwardToken(finalDraft, pos, computeLogits: true);
                    _cachedTokens.Add(finalDraft);
                }
                else
                {
                    _cachedTokens.Clear();
                }
            }

            return actualDrafted;
        }
        finally
        {
            if (rentedRecent != null) ArrayPool<int>.Shared.Return(rentedRecent);
        }
    }

    public void Dispose()
    {
        _cachedTokens.Clear();
        if (_ownsSession)
        {
            _session.Dispose();
        }
    }
}
