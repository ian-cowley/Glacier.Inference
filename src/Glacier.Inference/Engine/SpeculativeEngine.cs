namespace Glacier.Inference.Engine;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

/// <summary>
/// Speculative decoding engine delivering 1.5x - 3x autoregressive throughput acceleration.
/// Coordinates draft proposal (via fast N-gram lookup or secondary draft model) with batched
/// target model verification using exact Leviathan et al. (2023) speculative rejection sampling.
/// </summary>
public sealed class SpeculativeEngine : IDisposable
{
    private readonly ISpeculativeTarget _targetSession;
    private readonly IDraftProvider _draftProvider;
    private readonly bool _ownsDraftProvider;
    private readonly Sampler _sampler;
    private bool _disposed;

    public ISpeculativeTarget TargetSession => _targetSession;
    public IDraftProvider DraftProvider => _draftProvider;

    public SpeculativeEngine(ISpeculativeTarget targetSession, IDraftProvider? draftProvider = null, int? seed = null)
    {
        _targetSession = targetSession ?? throw new ArgumentNullException(nameof(targetSession));
        _sampler = new Sampler(seed);
        if (draftProvider != null)
        {
            _draftProvider = draftProvider;
            _ownsDraftProvider = false;
        }
        else
        {
            _draftProvider = new PromptLookupDraftProvider();
            _ownsDraftProvider = true;
        }
    }

    /// <summary>
    /// Generates text using speculative decoding with real-time stateful UTF-8 token streaming.
    /// </summary>
    public async Task<SpeculativeGenerationResult> GenerateAsync(
        string prompt,
        SpeculativeOptions? options = null,
        bool formatChat = true,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        options ??= new SpeculativeOptions();
        var sampling = options.Sampling;

        _draftProvider.Reset();

        // 1. Format and tokenize prompt
        string formattedPrompt = formatChat ? _targetSession.Tokenizer.FormatChatML(prompt) : prompt;
        int[] promptTokens = _targetSession.Tokenizer.Encode(formattedPrompt);

        if (promptTokens.Length == 0)
        {
            return new SpeculativeGenerationResult
            {
                Text = "",
                FinishReason = "empty_prompt",
                Metrics = new GenerationMetrics(),
                SpeculativeMetrics = new SpeculativeMetrics()
            };
        }

        if (options.MaxTokens <= 0)
        {
            return new SpeculativeGenerationResult
            {
                Text = "",
                FinishReason = "length",
                Metrics = new GenerationMetrics
                {
                    PromptTokens = promptTokens.Length,
                    GeneratedTokens = 0
                },
                SpeculativeMetrics = new SpeculativeMetrics()
            };
        }

        var totalStopwatch = Stopwatch.StartNew();
        var promptStopwatch = Stopwatch.StartNew();

        // 2. Prefill prompt in target session
        _targetSession.Prefill(promptTokens);
        promptStopwatch.Stop();

        // 3. Generate first token from prompt prefill
        int firstToken = _targetSession.SampleNextToken(sampling);

        var genStopwatch = Stopwatch.StartNew();
        var allTokens = new List<int>(promptTokens.Length + options.MaxTokens + 16);
        allTokens.AddRange(promptTokens);

        var generatedTokens = new List<int>(options.MaxTokens + 16);
        var responseSb = new StringBuilder();
        string finishReason = "length";

        // Check if first token is EOS
        if (IsEos(firstToken))
        {
            totalStopwatch.Stop();
            genStopwatch.Stop();
            return new SpeculativeGenerationResult
            {
                Text = "",
                FinishReason = "stop",
                Metrics = new GenerationMetrics
                {
                    PromptTokens = promptTokens.Length,
                    GeneratedTokens = 0,
                    PromptEvalDuration = promptStopwatch.Elapsed,
                    GenerationDuration = genStopwatch.Elapsed,
                    TotalDuration = totalStopwatch.Elapsed
                },
                SpeculativeMetrics = new SpeculativeMetrics()
            };
        }

        var utf8Decoder = Encoding.UTF8.GetDecoder();
        char[] charBuf = new char[128];

        // Accept first token
        allTokens.Add(firstToken);
        generatedTokens.Add(firstToken);
        EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, firstToken, responseSb, onToken);

        int maxSeq = _targetSession.MaxSeqLen;
        int currentPos = promptTokens.Length;
        int pendingToken = firstToken;

        int draftTokensProposed = 0;
        int draftTokensAccepted = 0;
        int speculativeSteps = 0;
        int serialFallbackSteps = 0;

        bool isGreedy = sampling.Temperature <= Sampler.GreedyTemperatureThreshold || sampling.TopK == 1;
        bool useLeviathanRejection = !isGreedy && _targetSession.SupportsLogitVerification;
        int vocabSize = useLeviathanRejection ? _targetSession.VocabSize : 0;

        int maxDraftCfg = Math.Max(0, options.MaxDraftTokens);
        int[] draftBuf = new int[Math.Max(1, maxDraftCfg)];
        int[] verifyBatch = new int[maxDraftCfg + 1];
        int[] predictions = new int[maxDraftCfg + 1];

        float[]? draftProbs = useLeviathanRejection && maxDraftCfg > 0 ? new float[maxDraftCfg * vocabSize] : null;
        float[]? targetBatchLogits = useLeviathanRejection ? new float[(maxDraftCfg + 1) * vocabSize] : null;
        float[]? targetProbs = useLeviathanRejection ? new float[vocabSize] : null;
        float[]? residualProbs = useLeviathanRejection ? new float[vocabSize] : null;

        long ssmStateBytes = _targetSession.StateSnapshotBytes;
        byte[]? ssmSnapshot = ssmStateBytes > 0 ? new byte[ssmStateBytes] : null;

        // 4. Main Speculative Decoding Loop
        while (generatedTokens.Count < options.MaxTokens && (_targetSession.IsRingBufferEnabled || currentPos < maxSeq))
        {
            ct.ThrowIfCancellationRequested();

            int maxDraftThisStep = (_targetSession.IsRingBufferEnabled || _targetSession.HasUnsnapshottableRecurrentState)
                ? 0
                : Math.Min(maxDraftCfg, maxSeq - currentPos - 1);

            int draftedCount = 0;
            if (maxDraftThisStep > 0)
            {
                if (useLeviathanRejection)
                {
                    draftedCount = _draftProvider.DraftWithProbabilities(
                        CollectionsMarshal.AsSpan(allTokens),
                        CollectionsMarshal.AsSpan(generatedTokens),
                        maxDraftThisStep,
                        sampling,
                        _sampler,
                        vocabSize,
                        draftBuf,
                        draftProbs.AsSpan());
                }
                else
                {
                    draftedCount = _draftProvider.Draft(
                        CollectionsMarshal.AsSpan(allTokens),
                        maxDraftThisStep,
                        draftBuf);
                }
            }

            if (draftedCount <= 0)
            {
                // Fallback to single serial step
                serialFallbackSteps++;
                _targetSession.ForwardToken(pendingToken, currentPos, computeLogits: true);
                currentPos++;

                int nextToken = _targetSession.SampleNextToken(sampling, CollectionsMarshal.AsSpan(generatedTokens));
                allTokens.Add(nextToken);
                generatedTokens.Add(nextToken);
                pendingToken = nextToken;

                if (IsEos(nextToken))
                {
                    finishReason = "stop";
                    break;
                }

                EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, nextToken, responseSb, onToken);
            }
            else
            {
                speculativeSteps++;
                draftTokensProposed += draftedCount;

                verifyBatch[0] = pendingToken;
                for (int i = 0; i < draftedCount; i++)
                {
                    verifyBatch[i + 1] = draftBuf[i];
                }

                if (ssmSnapshot != null)
                {
                    _targetSession.SaveRecurrentState(ssmSnapshot);
                }

                int batchLen = draftedCount + 1;
                int acceptedInStep = 0;
                bool stopped = false;

                if (useLeviathanRejection)
                {
                    _targetSession.VerifyBatchLogits(
                        verifyBatch.AsSpan(0, batchLen),
                        currentPos,
                        targetBatchLogits.AsSpan(0, batchLen * vocabSize));

                    for (int i = 0; i < draftedCount; i++)
                    {
                        int candidate = draftBuf[i];
                        ReadOnlySpan<float> rawTargetLogits = targetBatchLogits.AsSpan(i * vocabSize, vocabSize);
                        Sampler.ComputeFilteredDistribution(
                            rawTargetLogits,
                            sampling,
                            CollectionsMarshal.AsSpan(generatedTokens),
                            targetProbs.AsSpan());

                        ReadOnlySpan<float> qRow = draftProbs.AsSpan(i * vocabSize, vocabSize);
                        float pCandidate = (uint)candidate < (uint)vocabSize ? targetProbs![candidate] : 0f;
                        float qCandidate = (uint)candidate < (uint)vocabSize ? qRow[candidate] : 0f;

                        bool accept;
                        if (qCandidate <= 0f)
                        {
                            accept = false;
                        }
                        else if (pCandidate >= qCandidate)
                        {
                            accept = true;
                        }
                        else
                        {
                            accept = _sampler.Rng.NextDouble() < (pCandidate / qCandidate);
                        }

                        if (accept)
                        {
                            acceptedInStep++;
                            allTokens.Add(candidate);
                            generatedTokens.Add(candidate);

                            if (IsEos(candidate))
                            {
                                stopped = true;
                                finishReason = "stop";
                                break;
                            }

                            EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, candidate, responseSb, onToken);

                            if (generatedTokens.Count >= options.MaxTokens)
                            {
                                stopped = true;
                                break;
                            }
                        }
                        else
                        {
                            int correctionToken = SampleResidualDistribution(targetProbs!, qRow, residualProbs!, _sampler.Rng);
                            allTokens.Add(correctionToken);
                            generatedTokens.Add(correctionToken);

                            if (ssmSnapshot != null)
                            {
                                RollbackAndReplayRecurrentState(ssmSnapshot, verifyBatch, currentPos, acceptedInStep);
                            }

                            pendingToken = correctionToken;
                            currentPos += (i + 1);

                            if (IsEos(correctionToken))
                            {
                                finishReason = "stop";
                            }
                            else
                            {
                                EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, correctionToken, responseSb, onToken);
                            }

                            stopped = true;
                            break;
                        }
                    }

                    draftTokensAccepted += acceptedInStep;

                    if (!stopped && acceptedInStep == draftedCount)
                    {
                        ReadOnlySpan<float> bonusLogits = targetBatchLogits.AsSpan(draftedCount * vocabSize, vocabSize);
                        Sampler.ComputeFilteredDistribution(
                            bonusLogits,
                            sampling,
                            CollectionsMarshal.AsSpan(generatedTokens),
                            targetProbs.AsSpan());
                        int bonusToken = Sampler.SampleFromDistribution(targetProbs!, _sampler.Rng);

                        allTokens.Add(bonusToken);
                        generatedTokens.Add(bonusToken);
                        pendingToken = bonusToken;
                        currentPos += (draftedCount + 1);

                        if (IsEos(bonusToken))
                        {
                            finishReason = "stop";
                            break;
                        }

                        EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, bonusToken, responseSb, onToken);
                    }
                }
                else
                {
                    _targetSession.VerifyBatch(
                        verifyBatch.AsSpan(0, batchLen),
                        currentPos,
                        predictions.AsSpan(0, batchLen));

                    for (int i = 0; i < draftedCount; i++)
                    {
                        int candidate = draftBuf[i];
                        int targetPrediction = predictions[i];

                        if (candidate == targetPrediction)
                        {
                            acceptedInStep++;
                            allTokens.Add(candidate);
                            generatedTokens.Add(candidate);

                            if (IsEos(candidate))
                            {
                                stopped = true;
                                finishReason = "stop";
                                break;
                            }

                            EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, candidate, responseSb, onToken);

                            if (generatedTokens.Count >= options.MaxTokens)
                            {
                                stopped = true;
                                break;
                            }
                        }
                        else
                        {
                            allTokens.Add(targetPrediction);
                            generatedTokens.Add(targetPrediction);

                            if (ssmSnapshot != null)
                            {
                                RollbackAndReplayRecurrentState(ssmSnapshot, verifyBatch, currentPos, acceptedInStep);
                            }

                            pendingToken = targetPrediction;
                            currentPos += (i + 1);

                            if (IsEos(targetPrediction))
                            {
                                finishReason = "stop";
                            }
                            else
                            {
                                EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, targetPrediction, responseSb, onToken);
                            }

                            stopped = true;
                            break;
                        }
                    }

                    draftTokensAccepted += acceptedInStep;

                    if (!stopped && acceptedInStep == draftedCount)
                    {
                        int bonusToken = predictions[draftedCount];
                        allTokens.Add(bonusToken);
                        generatedTokens.Add(bonusToken);
                        pendingToken = bonusToken;
                        currentPos += (draftedCount + 1);

                        if (IsEos(bonusToken))
                        {
                            finishReason = "stop";
                            break;
                        }

                        EmitTokenUtf8(_targetSession.Tokenizer, utf8Decoder, ref charBuf, bonusToken, responseSb, onToken);
                    }
                }

                if (finishReason == "stop")
                {
                    break;
                }
            }

            if ((generatedTokens.Count & 15) == 0)
            {
                await Task.Yield();
            }
        }

        int flushChars = utf8Decoder.GetChars(Array.Empty<byte>(), 0, 0, charBuf, 0, flush: true);
        if (flushChars > 0)
        {
            string piece = new string(charBuf, 0, flushChars);
            responseSb.Append(piece);
            onToken?.Invoke(piece);
        }

        genStopwatch.Stop();
        totalStopwatch.Stop();

        return new SpeculativeGenerationResult
        {
            Text = responseSb.ToString(),
            FinishReason = finishReason,
            Metrics = new GenerationMetrics
            {
                PromptTokens = promptTokens.Length,
                GeneratedTokens = generatedTokens.Count,
                PromptEvalDuration = promptStopwatch.Elapsed,
                GenerationDuration = genStopwatch.Elapsed,
                TotalDuration = totalStopwatch.Elapsed
            },
            SpeculativeMetrics = new SpeculativeMetrics
            {
                TotalTokensGenerated = generatedTokens.Count,
                DraftTokensProposed = draftTokensProposed,
                DraftTokensAccepted = draftTokensAccepted,
                SpeculativeSteps = speculativeSteps,
                SerialFallbackSteps = serialFallbackSteps
            }
        };
    }

    private void RollbackAndReplayRecurrentState(byte[] ssmSnapshot, int[] verifyBatch, int basePos, int acceptedDrafts)
    {
        _targetSession.RestoreRecurrentState(ssmSnapshot);
        for (int r = 0; r <= acceptedDrafts; r++)
        {
            _targetSession.ForwardToken(verifyBatch[r], basePos + r, computeLogits: false);
        }
    }

    private static int SampleResidualDistribution(
        ReadOnlySpan<float> p,
        ReadOnlySpan<float> q,
        Span<float> residualScratch,
        Random rng)
    {
        int n = p.Length;
        float sumResidual = 0f;
        for (int v = 0; v < n; v++)
        {
            float diff = p[v] - q[v];
            if (diff > 0f)
            {
                residualScratch[v] = diff;
                sumResidual += diff;
            }
            else
            {
                residualScratch[v] = 0f;
            }
        }

        if (!(sumResidual > 1e-12f) || !float.IsFinite(sumResidual))
        {
            return Sampler.SampleFromDistribution(p, rng);
        }

        float r = (float)rng.NextDouble() * sumResidual;
        float acc = 0f;
        int lastPositive = 0;
        for (int v = 0; v < n; v++)
        {
            float w = residualScratch[v];
            if (w <= 0f) continue;
            lastPositive = v;
            acc += w;
            if (r <= acc)
            {
                return v;
            }
        }
        return lastPositive;
    }

    private static void EmitTokenUtf8(
        BpeTokenizer tokenizer,
        Decoder utf8Decoder,
        ref char[] charBuf,
        int token,
        StringBuilder responseSb,
        Action<string>? onToken)
    {
        byte[] tokenBytes = tokenizer.GetTokenBytes(token);
        int maxChars = Encoding.UTF8.GetMaxCharCount(tokenBytes.Length);
        if (charBuf.Length < maxChars)
        {
            charBuf = new char[maxChars];
        }

        int charsDecoded = utf8Decoder.GetChars(tokenBytes, 0, tokenBytes.Length, charBuf, 0, flush: false);
        if (charsDecoded > 0)
        {
            string piece = new string(charBuf, 0, charsDecoded);
            responseSb.Append(piece);
            onToken?.Invoke(piece);
        }
    }

    private bool IsEos(int token)
    {
        return _targetSession.Tokenizer.IsStopToken(token) || token == _targetSession.Tokenizer.EosTokenId || token == 151645 || token == 151643;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_ownsDraftProvider)
            {
                _draftProvider.Dispose();
            }
            _disposed = true;
        }
    }
}
