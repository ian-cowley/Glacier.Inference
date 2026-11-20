namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Glacier.Inference.Engine;
using Glacier.Inference.Evaluation;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;

public static partial class Program
{
    private static void PrintEvalKlHelp()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("glacier eval-kl - Evaluate KL-divergence D_KL(P_ref || Q_cand) against a reference GGUF");
        Console.ResetColor();
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  glacier eval-kl <candidate.gguf> --ref <reference.gguf> [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  <candidate.gguf>, -m, --model <path> Candidate quantized GGUF model (required)");
        Console.WriteLine("  --ref, --reference <path>            Reference GGUF model (e.g. Q8_0 or FP32) (required)");
        Console.WriteLine("  -p, --prompt \"<text>\"                Prompt to evaluate (repeatable)");
        Console.WriteLine("  --threshold <double>                 Mean KL divergence threshold in nats (default: 0.05)");
        Console.WriteLine("  --max-kl <double>                    Optional worst-case per-token KL threshold");
        Console.WriteLine("  --min-top1 <double>                  Minimum top-1 argmax agreement rate in [0, 1] (default: 0)");
        Console.WriteLine("  -c, --ctx, --context-length <len>    Maximum context sequence length (default: 2048)");
        Console.WriteLine("  --device <id|name>                   Target GPU/CPU");
        Console.WriteLine("  --engine <baremetal|vulkan|directml|cpu|auto> Execution engine (default: auto)");
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q4_0|fp32> KV-cache precision (default: auto)");
        Console.WriteLine("  --raw                                Skip chat template wrapping");
        Console.WriteLine("  --json                               Emit machine-readable AOT JSON output");
        Console.WriteLine("  -h, --help                           Show this help message");
    }

    private static int RunEvalKl(string[] args)
    {
        if (args.Length == 0 || HasHelpFlag(args))
        {
            PrintEvalKlHelp();
            return args.Length == 0 ? 1 : 0;
        }

        string? candidatePath = null;
        string? referencePath = null;
        var prompts = new List<string>();
        double threshold = KlDivergenceVerifier.DefaultThreshold;
        double? maxKlThreshold = null;
        double minTop1 = 0.0;
        int maxSeqLen = 2048;
        string? device = null;
        string? engineStr = null;
        string? kvPrecisionStr = null;
        bool rawPrompt = false;
        bool jsonOutput = false;

        for (int i = 0; i < args.Length; i++)
        {
            if ((args[i] == "-m" || args[i] == "--model") && i + 1 < args.Length)
                candidatePath = args[++i];
            else if ((args[i] == "--ref" || args[i] == "--reference") && i + 1 < args.Length)
                referencePath = args[++i];
            else if ((args[i] == "-p" || args[i] == "--prompt") && i + 1 < args.Length)
                prompts.Add(args[++i]);
            else if (args[i] == "--threshold" && i + 1 < args.Length && double.TryParse(args[++i], CultureInfo.InvariantCulture, out double th))
                threshold = th;
            else if (args[i] == "--max-kl" && i + 1 < args.Length && double.TryParse(args[++i], CultureInfo.InvariantCulture, out double mk))
                maxKlThreshold = mk;
            else if (args[i] == "--min-top1" && i + 1 < args.Length && double.TryParse(args[++i], CultureInfo.InvariantCulture, out double mt))
                minTop1 = mt;
            else if ((args[i] == "-c" || args[i] == "--ctx" || args[i] == "--context-length") && i + 1 < args.Length && int.TryParse(args[++i], out int ctx))
                maxSeqLen = ctx;
            else if ((args[i] == "-d" || args[i] == "--device") && i + 1 < args.Length)
                device = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                engineStr = args[++i];
            else if (args[i] == "--kv-precision" && i + 1 < args.Length)
                kvPrecisionStr = args[++i];
            else if (args[i] == "--raw")
                rawPrompt = true;
            else if (args[i] == "--json")
                jsonOutput = true;
            else if (!args[i].StartsWith("-") && candidatePath == null)
                candidatePath = args[i];
        }

        if (string.IsNullOrWhiteSpace(candidatePath) || string.IsNullOrWhiteSpace(referencePath))
        {
            Console.Error.WriteLine("Error: Both candidate GGUF and --ref <reference.gguf> are required.");
            PrintEvalKlHelp();
            return 1;
        }

        if (prompts.Count == 0)
        {
            prompts.Add("Explain in two sentences how virtual memory paging works.");
        }

        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(engineStr) && Enum.TryParse<InferenceEngineType>(engineStr, ignoreCase: true, out var parsedEngine))
            engine = parsedEngine;

        KvCachePrecision kvPrecision = KvCachePrecision.Auto;
        if (!string.IsNullOrWhiteSpace(kvPrecisionStr) && KvCachePrecisionExtensions.TryParse(kvPrecisionStr, out var parsedPrecision))
            kvPrecision = parsedPrecision;

        var opts = new KlDivergenceOptions
        {
            Threshold = threshold,
            MaxKlThreshold = maxKlThreshold,
            MinTop1Agreement = minTop1
        };

        KlDivergenceResult result = KlDivergenceVerifier.VerifyModels(
            referencePath,
            candidatePath,
            prompts,
            maxSeqLen: maxSeqLen,
            device: device,
            engine: engine,
            kvPrecision: kvPrecision,
            formatChat: !rawPrompt,
            evaluateAllPromptTokens: true,
            options: opts);

        if (jsonOutput)
        {
            var report = new KlDivergenceCliReport
            {
                Passed = result.Passed,
                MeanKlDivergence = result.MeanKlDivergence,
                MaxKlDivergence = result.MaxKlDivergence,
                MinKlDivergence = result.MinKlDivergence,
                Top1AgreementRate = result.Top1AgreementRate,
                Top1Matches = result.Top1Matches,
                TokenCount = result.TokenCount,
                VocabSize = result.VocabSize,
                Threshold = result.Threshold,
                MaxKlThreshold = result.MaxKlThreshold,
                MinTop1Agreement = result.MinTop1Agreement,
                HasNonFiniteQuantizedValues = result.HasNonFiniteQuantizedValues,
                FailureReason = result.FailureReason,
                ReferenceDevice = result.ReferenceDevice,
                CandidateDevice = result.CandidateDevice,
                DurationMs = result.Duration.TotalMilliseconds
            };
            Console.WriteLine(JsonSerializer.Serialize(report, InferenceCliJsonContext.Default.KlDivergenceCliReport));
        }
        else
        {
            Console.WriteLine($"Reference: {Path.GetFileName(referencePath)} ({result.ReferenceDevice ?? "N/A"})");
            Console.WriteLine($"Candidate: {Path.GetFileName(candidatePath)} ({result.CandidateDevice ?? "N/A"})");
            Console.WriteLine($"Tokens:    {result.TokenCount} | Vocab: {result.VocabSize} | Duration: {result.Duration.TotalMilliseconds:F1} ms");
            Console.WriteLine($"Mean KL:   {result.MeanKlDivergence:G6} (threshold <= {result.Threshold:G6})");
            Console.WriteLine($"Max KL:    {result.MaxKlDivergence:G6} | Min KL: {result.MinKlDivergence:G6}");
            Console.WriteLine($"Top-1 Agr: {result.Top1AgreementRate:P2} ({result.Top1Matches}/{result.TokenCount})");
            Console.WriteLine($"Status:    {(result.Passed ? "PASSED" : $"FAILED - {result.FailureReason}")}");
        }

        return result.Passed ? 0 : 2;
    }
}
