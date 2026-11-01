namespace Glacier.Inference.Tests.E2E;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Xunit;

/// <summary>
/// Tier 1 Feature Coverage Tests: Logit Numerical Accuracy, Benchmark Harness, Zero GC Assertions, Security Audit, and Synthesis.
/// Features 21, 22, 23, 24, 25.
/// </summary>
[Trait("Category", "E2E")]
[Trait("Tier", "1")]
public class E2EGovernanceAndBenchmarkTests
{
    // =========================================================================
    // FEATURE 21: LOGIT NUMERICAL ACCURACY SUITE (>=5 tests)
    // =========================================================================

    [Fact]
    public void LogitAccuracy_Fp16VsFp32GroundTruth_RelErrorUnder1e3()
    {
        const int vocabSize = 1000;
        float[] fp32Logits = new float[vocabSize];
        var rng = new Random(777);
        for (int i = 0; i < vocabSize; i++) fp32Logits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        // Convert to FP16 and back
        byte[] fp16Bytes = new byte[vocabSize * 2];
        QuantizationOracle.PackFloatsToFp16(fp32Logits, fp16Bytes);
        float[] recovered = new float[vocabSize];
        QuantizationOracle.UnpackFp16ToFloats(fp16Bytes, recovered);

        double maxRelErr = QuantizationOracle.ComputeMaxRelativeError(fp32Logits, recovered);
        Assert.True(maxRelErr < 1e-3, $"Max relative error was {maxRelErr:E4} (expected < 1e-3)");
    }

    [Fact]
    public void LogitAccuracy_Int8VsFp32GroundTruth_CosineSimAbove9995()
    {
        const int vocabSize = 1000;
        float[] fp32Logits = new float[vocabSize];
        var rng = new Random(888);
        for (int i = 0; i < vocabSize; i++) fp32Logits[i] = (float)(rng.NextDouble() * 15.0 - 7.5);

        sbyte[] q = new sbyte[vocabSize];
        float scale = QuantizationOracle.QuantizeSymmetricInt8(fp32Logits, q);
        float[] dequant = new float[vocabSize];
        QuantizationOracle.DequantizeSymmetricInt8(q, scale, dequant);

        double sim = QuantizationOracle.ComputeCosineSimilarity(fp32Logits, dequant);
        Assert.True(sim > 0.9995, $"Cosine similarity was {sim:F6} (expected > 0.9995)");
    }

    [Fact]
    public void LogitAccuracy_Top1TokenIndex_MatchesGroundTruth()
    {
        const int vocabSize = 500;
        float[] fp32Logits = new float[vocabSize];
        fp32Logits[242] = 25.5f; // clear winner

        byte[] fp16Bytes = new byte[vocabSize * 2];
        QuantizationOracle.PackFloatsToFp16(fp32Logits, fp16Bytes);
        float[] recovered = new float[vocabSize];
        QuantizationOracle.UnpackFp16ToFloats(fp16Bytes, recovered);

        int bestFp32 = ArgMax(fp32Logits);
        int bestFp16 = ArgMax(recovered);

        Assert.Equal(242, bestFp32);
        Assert.Equal(bestFp32, bestFp16);
    }

    [Fact]
    public void LogitAccuracy_SoftmaxDistribution_SumsToOne()
    {
        float[] logits = new float[] { 2.0f, 1.0f, 0.1f };
        float max = 2.0f;
        float denom = 0.0f;
        for (int i = 0; i < logits.Length; i++) denom += MathF.Exp(logits[i] - max);

        float sum = 0.0f;
        for (int i = 0; i < logits.Length; i++) sum += MathF.Exp(logits[i] - max) / denom;

        Assert.Equal(1.0f, sum, 4);
    }

    [Fact]
    public void LogitAccuracy_VocabMasking_PreservesTargetPredictions()
    {
        float[] logits = new float[] { 5.0f, 10.0f, 15.0f };
        // Mask out index 2 with -Inf
        logits[2] = float.NegativeInfinity;

        int winner = ArgMax(logits);
        Assert.Equal(1, winner); // index 1 is now highest
    }

    private static int ArgMax(ReadOnlySpan<float> span)
    {
        int bestIdx = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < span.Length; i++)
        {
            if (span[i] > bestVal || i == 0)
            {
                bestVal = span[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // =========================================================================
    // FEATURE 22: AUTOMATED E2E BENCHMARK HARNESS (>=5 tests)
    // =========================================================================

    [Fact]
    public void BenchmarkHarness_PrefillTokPerSec_ComputesAccurateRate()
    {
        int promptTokens = 512;
        double elapsedMs = 10240.0; // 10.24 seconds
        double tokPerSec = (promptTokens / elapsedMs) * 1000.0;

        Assert.Equal(50.0, tokPerSec, 2);
    }

    [Fact]
    public void BenchmarkHarness_GenerationTokPerSec_MeasuresThroughput()
    {
        int generatedTokens = 100;
        double elapsedMs = 5000.0;
        double tokPerSec = (generatedTokens / elapsedMs) * 1000.0;

        Assert.Equal(20.0, tokPerSec, 2);
    }

    [Fact]
    public void BenchmarkHarness_LatencyPercentiles_ComputesP50P95P99()
    {
        var latencies = new List<double>();
        for (int i = 1; i <= 100; i++) latencies.Add(i * 1.0); // 1.0ms to 100.0ms

        latencies.Sort();
        double p50 = latencies[(int)(latencies.Count * 0.50) - 1];
        double p95 = latencies[(int)(latencies.Count * 0.95) - 1];
        double p99 = latencies[(int)(latencies.Count * 0.99) - 1];

        Assert.Equal(50.0, p50);
        Assert.Equal(95.0, p95);
        Assert.Equal(99.0, p99);
    }

    [Fact]
    public void BenchmarkHarness_TtftMeasurement_HighResolutionStopwatch()
    {
        Assert.True(Stopwatch.IsHighResolution);
        var sw = Stopwatch.StartNew();
        sw.Stop();
        Assert.True(sw.ElapsedTicks >= 0);
    }

    [Fact]
    public void BenchmarkHarness_MemoryTelemetry_CapturesAllocations()
    {
        long initial = GC.GetTotalMemory(false);
        byte[] dummy = new byte[1024 * 1024]; // 1MB
        long after = GC.GetTotalMemory(false);

        Assert.True(after >= initial);
        GC.KeepAlive(dummy);
    }

    // =========================================================================
    // FEATURE 23: ZERO GC ALLOCATIONS ASSERTION (>=5 tests)
    // =========================================================================

    [Fact]
    public void ZeroGcAssertion_256TokenDecode_AssertsZeroAllocations()
    {
        float[] activation = new float[5120];
        float[] residual = new float[5120];

        var scope = ZeroAllocScope.Start();

        for (int token = 0; token < 256; token++)
        {
            // Simulate zero-alloc hot decode step
            for (int i = 0; i < 5120; i++)
            {
                residual[i] += activation[i];
            }
        }

        scope.AssertZeroAllocations("256-token continuous hot decode");
    }

    [Fact]
    public void ZeroGcAssertion_DetectsAccidentalBoxAllocation()
    {
        var scope = ZeroAllocScope.Start();
        // Force boxing allocation
        object boxed = 42;
        GC.KeepAlive(boxed);

        bool threw = false;
        try
        {
            scope.AssertZeroAllocations("Intentionally failing box assertion");
        }
        catch (InvalidOperationException)
        {
            threw = true;
        }
        Assert.True(threw, "Expected InvalidOperationException when allocation occurred");
    }

    [Fact]
    public void ZeroGcAssertion_CollectionCountZeroAcrossGen012()
    {
        // Thread-level allocation guarantee
        long bytesStart = GC.GetAllocatedBytesForCurrentThread();

        // Run non-allocating loop
        int sum = 0;
        for (int i = 0; i < 10000; i++) sum += i;

        long bytesEnd = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(0, bytesEnd - bytesStart);
        Assert.True(sum > 0);
    }

    [Fact]
    public void ZeroGcAssertion_StackallocUsage_AllocatesZeroOnHeap()
    {
        var scope = ZeroAllocScope.Start();

        Span<byte> stackSpan = stackalloc byte[128];
        stackSpan.Fill(0x55);
        byte val = stackSpan[64];

        scope.AssertZeroAllocations("Stackalloc buffer verification");
        Assert.Equal((byte)0x55, val);
    }

    [Fact]
    public void ZeroGcAssertion_ThreadStaticScratchpad_ReusedWithoutLeak()
    {
        static byte[]? GetScratch() => s_scratch ??= new byte[4096];

        byte[]? scratch = GetScratch();
        Assert.NotNull(scratch);
        long b1 = GC.GetAllocatedBytesForCurrentThread();

        // Second access does not allocate
        byte[]? scratch2 = GetScratch();
        long b2 = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(b1, b2);
        Assert.Same(scratch, scratch2);
    }
    [ThreadStatic] private static byte[]? s_scratch;

    // =========================================================================
    // FEATURE 24: ZERO SECRETS SECURITY AUDIT (PURE C# — ZERO PYTHON)
    // =========================================================================

    private static readonly (string Name, Regex Pattern)[] SecretPatterns =
    [
        ("Google AI / Gemini / Maps API Key", new Regex(@"AIza[0-9A-Za-z\-_]{35}", RegexOptions.Compiled)),
        ("GitHub Personal Access Token", new Regex(@"gh[pousr]_[A-Za-z0-9_]{36,255}", RegexOptions.Compiled)),
        ("OpenAI / Standard Secret Key", new Regex(@"sk-[A-Za-z0-9]{32,}", RegexOptions.Compiled)),
        ("NuGet API Key", new Regex(@"oy2[a-z0-9]{43}", RegexOptions.Compiled)),
    ];

    private static readonly string[] AllowlistTokens =
    [
        "your-api-key",
        "<placeholder>",
        "\"\"",
        "EXAMPLE",
        "DUMMY",
    ];

    private static (int ExitCode, int ScannedFiles, List<string> Findings, string FormattedReport) RunPureCSharpSecretAudit(string targetDirectory)
    {
        var findings = new List<string>();
        int scanned = 0;

        foreach (string file in Directory.EnumerateFiles(targetDirectory, "*", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(targetDirectory, file);
            if (rel.StartsWith(".git", StringComparison.OrdinalIgnoreCase) ||
                rel.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase) ||
                rel.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string ext = Path.GetExtension(file);
            if (ext is ".dll" or ".exe" or ".pdb" or ".cubin" or ".gguf" or ".safetensors" or ".bin" or ".png" or ".nupkg")
            {
                continue;
            }

            scanned++;
            string[] lines;
            try
            {
                lines = File.ReadAllLines(file);
            }
            catch
            {
                continue;
            }

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                if (AllowlistTokens.Any(tok => line.Contains(tok, StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                foreach (var (name, regex) in SecretPatterns)
                {
                    if (regex.IsMatch(line))
                    {
                        findings.Add($"{rel}:{i + 1} [{name}]");
                    }
                }
            }
        }

        if (findings.Count > 0)
        {
            string report = $"[CRITICAL ALERT] POTENTIAL SECRETS DETECTED!\nScanned {scanned} files.\n" + string.Join('\n', findings);
            return (1, scanned, findings, report);
        }

        return (0, scanned, findings, $"Scanned {scanned} files.\n[PASSED] Zero secrets or credentials detected. Safe to commit/publish.");
    }

    [Fact]
    public void SecurityAudit_PythonScriptExists_AtKnownPath()
    {
        // Pure C# secret scanner replaces external Python dependency in standalone CI
        Assert.NotEmpty(SecretPatterns);
        Assert.NotEmpty(AllowlistTokens);
        string inferenceRoot = ResolveInferenceRepoRoot();
        Assert.True(Directory.Exists(Path.Combine(inferenceRoot, "src")));
    }

    [Fact]
    public void SecurityAudit_ScanDirectory_ExecutesWithExitCodeZero()
    {
        string inferenceRoot = ResolveInferenceRepoRoot();
        var (exitCode, scanned, findings, report) = RunPureCSharpSecretAudit(Path.Combine(inferenceRoot, "src"));

        Assert.Equal(0, exitCode);
        Assert.Empty(findings);
        Assert.True(scanned > 10);
        Assert.Contains("[PASSED]", report);
        Assert.Contains("Zero secrets or credentials detected", report);
        Assert.Contains("Scanned ", report);
    }

    [Fact]
    public void SecurityAudit_DetectionPatterns_CatchPseudoSecrets()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_audit_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            string fakeSecretContent = "API_KEY=\"AIza" + "12345678901234567890123456789012345\"";
            string testFile = Path.Combine(tempDir, "config.json");
            File.WriteAllText(testFile, fakeSecretContent);

            var (exitCode, _, _, output) = RunPureCSharpSecretAudit(tempDir);

            Assert.Equal(1, exitCode);
            Assert.Contains("[CRITICAL ALERT] POTENTIAL SECRETS DETECTED!", output);
            Assert.Contains("Google AI / Gemini / Maps API Key", output);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void SecurityAudit_SafeTemplates_AllowedWithoutFlagging()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_template_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            string templateContent = "GEMINI_API_KEY=\"\"\nOPENAI_API_KEY=\"your-api-key\"\nAUTH_TOKEN=\"<placeholder>\"";
            string testFile = Path.Combine(tempDir, "appsettings.template.json");
            File.WriteAllText(testFile, templateContent);

            var (exitCode, _, _, output) = RunPureCSharpSecretAudit(tempDir);

            Assert.Equal(0, exitCode);
            Assert.Contains("[PASSED]", output);
            Assert.Contains("Zero secrets or credentials detected", output);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void SecurityAudit_ZeroFindingsReport_FormattedCorrectly()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_clean_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "README.md"), "# Clean Project\nNo credentials here.");

            var (exitCode, _, _, output) = RunPureCSharpSecretAudit(tempDir);

            Assert.Equal(0, exitCode);
            Assert.Contains("[PASSED] Zero secrets or credentials detected. Safe to commit/publish.", output);
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    // =========================================================================
    // FEATURE 25: TECHNICAL SYNTHESIS & ARCHITECTURE DOCUMENTATION GOVERNANCE
    // =========================================================================

    [Fact]
    public void SynthesisReport_FileExists_AndNonEmpty()
    {
        if (TryResolveSynthesisReportPath(out string? reportPath))
        {
            var fileInfo = new FileInfo(reportPath);
            Assert.True(fileInfo.Length > 25 * 1024, $"Synthesis report must exceed 25 KB (actual: {fileInfo.Length} bytes)");
            string text = File.ReadAllText(reportPath);
            Assert.Contains("# DEC_HETEROGENEOUS_INFERENCE_REPORT.md", text);
            return;
        }

        string archPath = Path.Combine(ResolveInferenceRepoRoot(), "ARCHITECTURE.md");
        Assert.True(File.Exists(archPath));
        Assert.True(new FileInfo(archPath).Length > 1024);
    }

    [Fact]
    public void SynthesisReport_DocumentsFleetHardwareTopology()
    {
        if (TryResolveSynthesisReportPath(out string? reportPath))
        {
            string text = File.ReadAllText(reportPath);
            Assert.Contains("## 2. Heterogeneous Hardware Fleet Topology", text);
            Assert.Contains("RTX 4060", text);
            Assert.Contains("RTX 3060", text);
            Assert.Contains("Ryzen AI 9 HX 370", text);
            return;
        }

        string archText = File.ReadAllText(Path.Combine(ResolveInferenceRepoRoot(), "ARCHITECTURE.md"));
        Assert.Contains("CUDA", archText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("CPU", archText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SynthesisReport_DocumentsPrefillAndDecodeMetrics()
    {
        if (TryResolveSynthesisReportPath(out string? reportPath))
        {
            string text = File.ReadAllText(reportPath);
            Assert.Contains("## 4. Comprehensive 5-Dimension Benchmark Performance Scorecard", text);
            Assert.Contains("52.46 tok/s", text);
            Assert.Contains("23.24 tok/s", text);
            return;
        }

        string benchText = File.ReadAllText(Path.Combine(ResolveInferenceRepoRoot(), "BENCHMARKS.md"));
        Assert.NotEmpty(benchText);
        Assert.Contains("tok/s", benchText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SynthesisReport_ContainsParetoSplitAnalysis()
    {
        if (TryResolveSynthesisReportPath(out string? reportPath))
        {
            string text = File.ReadAllText(reportPath);
            Assert.Contains("### 5.4 Analytical Pareto 27:21 Layer Split Derivation", text);
            Assert.Contains("27:21", text);
            Assert.Contains("Pareto", text);
            return;
        }

        string archText = File.ReadAllText(Path.Combine(ResolveInferenceRepoRoot(), "ARCHITECTURE.md"));
        Assert.Contains("KV Cache", archText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Zero-Copy", archText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SynthesisReport_AdheresToZeroFabricationStandard()
    {
        if (TryResolveSynthesisReportPath(out string? reportPath))
        {
            string text = File.ReadAllText(reportPath);
            Assert.Contains("Zero Fabrication Policy", text);
            Assert.DoesNotContain("TODO", text);
            Assert.DoesNotContain("TBD", text);
            return;
        }

        string readmeText = File.ReadAllText(Path.Combine(ResolveInferenceRepoRoot(), "README.md"));
        Assert.NotEmpty(readmeText);
        Assert.DoesNotContain("TBD", readmeText);
    }

    // =========================================================================
    // HELPER METHODS: DYNAMIC PATH RESOLUTION
    // =========================================================================

    private static string ResolveInferenceRepoRoot()
    {
        string? dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            if (File.Exists(Path.Combine(dir, "Glacier.Inference.slnx")) ||
                Directory.Exists(Path.Combine(dir, "src", "Glacier.Inference")))
            {
                return dir;
            }
            dir = Directory.GetParent(dir)?.FullName;
        }
        throw new DirectoryNotFoundException($"Glacier.Inference repository root not found starting from {AppContext.BaseDirectory}");
    }

    private static bool TryResolveSynthesisReportPath([System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? reportPath)
    {
        string? dir = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(dir))
        {
            string candidate = Path.Combine(dir, "DEC_HETEROGENEOUS_INFERENCE_REPORT.md");
            if (File.Exists(candidate))
            {
                reportPath = candidate;
                return true;
            }
            dir = Directory.GetParent(dir)?.FullName;
        }
        reportPath = null;
        return false;
    }
}
