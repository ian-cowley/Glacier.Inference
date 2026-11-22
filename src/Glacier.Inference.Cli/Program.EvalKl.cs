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
        Console.WriteLine("  --kv-precision <auto|fp16|fp8|q8_0|q4_0|fp32> KV-cache precision (default: auto)");
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
            string flag = args[i];
            if (flag is "-m" or "--model")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                candidatePath = val;
            }
            else if (flag is "--ref" or "--reference")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                referencePath = val;
            }
            else if (flag is "-p" or "--prompt")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                prompts.Add(val);
            }
            else if (flag == "--threshold")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!double.TryParse(val, CultureInfo.InvariantCulture, out double th))
                {
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    return 1;
                }
                threshold = th;
            }
            else if (flag == "--max-kl")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!double.TryParse(val, CultureInfo.InvariantCulture, out double mk))
                {
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    return 1;
                }
                maxKlThreshold = mk;
            }
            else if (flag == "--min-top1")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!double.TryParse(val, CultureInfo.InvariantCulture, out double mt))
                {
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'.");
                    return 1;
                }
                minTop1 = mt;
            }
            else if (flag is "-c" or "--ctx" or "--context-length")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                if (!int.TryParse(val, out int ctx) || ctx <= 0)
                {
                    Console.Error.WriteLine($"Error: Invalid {flag} '{val}'. Expected a positive integer.");
                    return 1;
                }
                maxSeqLen = ctx;
            }
            else if (flag is "-d" or "--device")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                device = val;
            }
            else if (flag == "--engine")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                engineStr = val;
            }
            else if (flag == "--kv-precision")
            {
                if (!TryRequireFlagValue(args, ref i, flag, out string val)) return 1;
                kvPrecisionStr = val;
            }
            else if (flag.StartsWith("--kv-precision=", StringComparison.Ordinal))
                kvPrecisionStr = flag["--kv-precision=".Length..];
            else if (flag == "--raw")
                rawPrompt = true;
            else if (flag == "--json")
                jsonOutput = true;
            else if (!flag.StartsWith("-") && candidatePath == null)
                candidatePath = flag;
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

        if (!TryParseKvPrecisionArg(kvPrecisionStr, out var kvPrecision, out var kvError))
        {
            Console.Error.WriteLine(kvError);
            return 1;
        }

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
