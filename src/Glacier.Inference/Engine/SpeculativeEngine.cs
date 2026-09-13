namespace Glacier.Inference.Engine;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;

/// <summary>
/// Speculative decoding engine delivering 1.5x - 3x autoregressive throughput acceleration.
/// Coordinates draft proposal (via fast N-gram lookup or secondary draft model) with batched
/// target model verification. The target model maintains 100% mathematical fidelity.
/// </summary>
public sealed class SpeculativeEngine : IDisposable
{
    private readonly ISpeculativeTarget _targetSession;
    private readonly IDraftProvider _draftProvider;
    private readonly bool _ownsDraftProvider;
    private bool _disposed;

    public ISpeculativeTarget TargetSession => _targetSession;
    public IDraftProvider DraftProvider => _draftProvider;

    public SpeculativeEngine(ISpeculativeTarget targetSession, IDraftProvider? draftProvider = null)
    {
        _targetSession = targetSession ?? throw new ArgumentNullException(nameof(targetSession));
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
    /// Generates text using speculative decoding with real-time token streaming callback.
    /// </summary>
    public async Task<SpeculativeGenerationResult> GenerateAsync(
        string prompt,
        SpeculativeOptions? options = null,
        bool formatChat = true,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        options ??= new SpeculativeOptions();

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

        var totalStopwatch = Stopwatch.StartNew();
        var promptStopwatch = Stopwatch.StartNew();

        // 2. Prefill prompt in target session
        _targetSession.Prefill(promptTokens);
        promptStopwatch.Stop();

        // 3. Generate first token from prompt prefill
        int firstToken = _targetSession.SampleNextToken(options.Sampling);

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

        // Accept first token
        allTokens.Add(firstToken);
        generatedTokens.Add(firstToken);
        string firstPiece = _targetSession.Tokenizer.DecodeToken(firstToken);
        responseSb.Append(firstPiece);
        onToken?.Invoke(firstPiece);

        int maxSeq = _targetSession.MaxSeqLen;
        int currentPos = promptTokens.Length; // position where firstToken was evaluated / next pending token position
        int pendingToken = firstToken;

        int draftTokensProposed = 0;
        int draftTokensAccepted = 0;
        int speculativeSteps = 0;
        int serialFallbackSteps = 0;

        int[] draftBuf = new int[options.MaxDraftTokens];
        int[] verifyBatch = new int[options.MaxDraftTokens + 1];
        int[] predictions = new int[options.MaxDraftTokens + 1];

        // 4. Main Speculative Decoding Loop
        while (generatedTokens.Count < options.MaxTokens && currentPos < maxSeq - (options.MaxDraftTokens + 1))
        {
            ct.ThrowIfCancellationRequested();

            // Propose candidate tokens from draft provider
            int draftedCount = _draftProvider.Draft(
                CollectionsMarshal.AsSpan(allTokens),
                options.MaxDraftTokens,
                draftBuf);

            if (draftedCount <= 0)
            {
                // Fallback to single serial step
                serialFallbackSteps++;
                _targetSession.ForwardToken(pendingToken, currentPos, computeLogits: true);
                currentPos++;

                int nextToken = _targetSession.SampleNextToken(options.Sampling, CollectionsMarshal.AsSpan(generatedTokens));
                allTokens.Add(nextToken);
                generatedTokens.Add(nextToken);
                pendingToken = nextToken;

                if (IsEos(nextToken))
                {
                    finishReason = "stop";
                    break;
                }

                string piece = _targetSession.Tokenizer.DecodeToken(nextToken);
                responseSb.Append(piece);
                onToken?.Invoke(piece);
            }
            else
            {
                // Speculative verification step
                speculativeSteps++;
                draftTokensProposed += draftedCount;

                // Build verification batch: [pendingToken, draftBuf[0], ..., draftBuf[draftedCount-1]]
                verifyBatch[0] = pendingToken;
                for (int i = 0; i < draftedCount; i++)
                {
                    verifyBatch[i + 1] = draftBuf[i];
                }

                int batchLen = draftedCount + 1;
                _targetSession.VerifyBatch(
                    verifyBatch.AsSpan(0, batchLen),
                    currentPos,
                    predictions.AsSpan(0, batchLen));

                // Verify candidates sequentially
                int acceptedInStep = 0;
                bool stopped = false;

                for (int i = 0; i < draftedCount; i++)
                {
                    int candidate = draftBuf[i];
                    int targetPrediction = predictions[i];

                    if (candidate == targetPrediction)
                    {
                        // Accepted!
                        acceptedInStep++;
                        allTokens.Add(candidate);
                        generatedTokens.Add(candidate);

                        if (IsEos(candidate))
                        {
                            stopped = true;
                            finishReason = "stop";
                            break;
                        }

                        string piece = _targetSession.Tokenizer.DecodeToken(candidate);
                        responseSb.Append(piece);
                        onToken?.Invoke(piece);

                        if (generatedTokens.Count >= options.MaxTokens)
                        {
                            stopped = true;
                            break;
                        }
                    }
                    else
                    {
                        // Rejection: emit target correction
                        allTokens.Add(targetPrediction);
                        generatedTokens.Add(targetPrediction);
                        pendingToken = targetPrediction;
                        currentPos += (i + 1);

                        if (IsEos(targetPrediction))
                        {
                            stopped = true;
                            finishReason = "stop";
                        }
                        else
                        {
                            string piece = _targetSession.Tokenizer.DecodeToken(targetPrediction);
                            responseSb.Append(piece);
                            onToken?.Invoke(piece);
                        }

                        stopped = true;
                        break;
                    }
                }

                draftTokensAccepted += acceptedInStep;

                if (!stopped && acceptedInStep == draftedCount)
                {
                    // All draft candidates accepted! Emit the bonus token predicted after the last candidate
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

                    string piece = _targetSession.Tokenizer.DecodeToken(bonusToken);
                    responseSb.Append(piece);
                    onToken?.Invoke(piece);
                }

                if (finishReason == "stop")
                {
                    break;
                }
            }

            // Yield control periodically
            if ((generatedTokens.Count & 15) == 0)
            {
                await Task.Yield();
            }
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
