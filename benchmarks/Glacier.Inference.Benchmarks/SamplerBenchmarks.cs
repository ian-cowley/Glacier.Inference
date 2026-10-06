namespace Glacier.Inference.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using Glacier.Inference.Sampling;

/// <summary>
/// Benchmarks the Sampler decoding pipeline (Greedy, Top-K, Top-P, Repetition Penalty)
/// verifying zero memory allocations during token generation across standard LLM vocabulary sizes.
/// </summary>
[MemoryDiagnoser]
public class SamplerBenchmarks
{
    [Params(32000, 151936)]
    public int VocabSize { get; set; }

    private float[] _logits = null!;
    private float[] _logitsCopy = null!;
    private int[] _recentTokens = null!;
    private Sampler _sampler = null!;

    private SamplingOptions _greedyOptions = null!;
    private SamplingOptions _topK40Options = null!;
    private SamplingOptions _topK100Options = null!;
    private SamplingOptions _repPenaltyOptions = null!;

    [GlobalSetup]
    public void Setup()
    {
        _sampler = new Sampler(42);
        _logits = new float[VocabSize];
        _logitsCopy = new float[VocabSize];

        var rng = new Random(12345);
        for (int i = 0; i < VocabSize; i++)
        {
            _logits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
        }

        _recentTokens = new int[32];
        for (int i = 0; i < 32; i++)
        {
            _recentTokens[i] = rng.Next(VocabSize);
        }

        _greedyOptions = SamplingOptions.Greedy;

        _topK40Options = new SamplingOptions
        {
            Temperature = 0.7f,
            TopK = 40,
            TopP = 0.9f,
            RepetitionPenalty = 1.0f
        };

        _topK100Options = new SamplingOptions
        {
            Temperature = 0.7f,
            TopK = 100,
            TopP = 0.95f,
            RepetitionPenalty = 1.0f
        };

        _repPenaltyOptions = new SamplingOptions
        {
            Temperature = 0.7f,
            TopK = 40,
            TopP = 0.9f,
            RepetitionPenalty = 1.15f
        };
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Restore logits because sampling with repetition penalty modifies them in-place
        Array.Copy(_logits, _logitsCopy, VocabSize);
    }

    [Benchmark(Baseline = true)]
    public int Sample_Greedy()
    {
        return _sampler.Sample(_logitsCopy.AsSpan(), _greedyOptions);
    }

    [Benchmark]
    public int Sample_TopK40_TopP()
    {
        return _sampler.Sample(_logitsCopy.AsSpan(), _topK40Options);
    }

    [Benchmark]
    public int Sample_TopK100_TopP()
    {
        return _sampler.Sample(_logitsCopy.AsSpan(), _topK100Options);
    }

    [Benchmark]
    public int Sample_RepetitionPenalty_TopK40()
    {
        return _sampler.Sample(_logitsCopy.AsSpan(), _repPenaltyOptions, _recentTokens.AsSpan());
    }
}
