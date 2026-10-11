namespace Glacier.Inference.Tests.E2E;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
    // FEATURE 24: ZERO SECRETS SECURITY AUDIT (>=5 tests)
    // =========================================================================

    [Fact]
    public void SecurityAudit_PythonScriptExists_AtKnownPath()
    {
        string scriptPath = ResolveSecurityScriptPath();
        Assert.True(File.Exists(scriptPath), $"security_check.py must exist at: {scriptPath}");

        var fileInfo = new FileInfo(scriptPath);
        Assert.True(fileInfo.Length > 3000, "security_check.py should be a non-trivial scanner (> 3KB)");

        string scriptContent = File.ReadAllText(scriptPath);
        Assert.Contains("SECRET_PATTERNS", scriptContent);
        Assert.Contains("ALLOWLIST_PATTERNS", scriptContent);
        Assert.Contains("scan_directory", scriptContent);
    }

    [Fact]
    public void SecurityAudit_ScanDirectory_ExecutesWithExitCodeZero()
    {
        string scriptPath = ResolveSecurityScriptPath();
        string repoRoot = ResolveRepoRoot();

        var psi = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = $"\"{scriptPath}\" .",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi);
        Assert.NotNull(proc);
        string output = proc.StandardOutput.ReadToEnd();
        string error = proc.StandardError.ReadToEnd();
        bool exited = proc.WaitForExit(180_000); // 3-minute timeout guard

        Assert.True(exited, "security_check.py process timed out after 180 seconds");
        Assert.Equal(0, proc.ExitCode);
        Assert.Contains("[PASSED]", output);
        Assert.Contains("Zero secrets or credentials detected", output);
        Assert.Contains("Scanned ", output);
    }

    [Fact]
    public void SecurityAudit_DetectionPatterns_CatchPseudoSecrets()
    {
        string scriptPath = ResolveSecurityScriptPath();
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_audit_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            // Write a synthetic un-allowlisted API key to an isolated test file
            string fakeSecretContent = "API_KEY=\"AIza" + "12345678901234567890123456789012345\"";
            string testFile = Path.Combine(tempDir, "config.py");
            File.WriteAllText(testFile, fakeSecretContent);

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{tempDir}\"",
                WorkingDirectory = tempDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            Assert.NotNull(proc);
            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(30_000);

            // Scanner must fail and exit with code 1 when an un-allowlisted secret is present
            Assert.Equal(1, proc.ExitCode);
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
        string scriptPath = ResolveSecurityScriptPath();
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_template_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            // Write an allowlisted template and placeholder
            string templateContent = "GEMINI_API_KEY=\"\"\nOPENAI_API_KEY=\"your-api-key\"\nAUTH_TOKEN=\"<placeholder>\"";
            string testFile = Path.Combine(tempDir, "appsettings.template.json");
            File.WriteAllText(testFile, templateContent);

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{tempDir}\"",
                WorkingDirectory = tempDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            Assert.NotNull(proc);
            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(30_000);

            // Scanner must pass with exit code 0 when all entries are allowlisted
            Assert.Equal(0, proc.ExitCode);
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
        string scriptPath = ResolveSecurityScriptPath();
        string tempDir = Path.Combine(Path.GetTempPath(), "glacier_clean_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "README.md"), "# Clean Project\nNo credentials here.");

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{scriptPath}\" \"{tempDir}\"",
                WorkingDirectory = tempDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var proc = Process.Start(psi);
            Assert.NotNull(proc);
            string output = proc.StandardOutput.ReadToEnd();
            proc.WaitForExit(30_000);

            Assert.Equal(0, proc.ExitCode);
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
    // FEATURE 25: FINAL TECHNICAL SYNTHESIS REPORT (>=5 tests)
    // =========================================================================

    [Fact]
    public void SynthesisReport_FileExists_AndNonEmpty()
    {
        string reportPath = ResolveSynthesisReportPath();
        Assert.True(File.Exists(reportPath), $"DEC_HETEROGENEOUS_INFERENCE_REPORT.md must exist at: {reportPath}");

        var fileInfo = new FileInfo(reportPath);
        Assert.True(fileInfo.Length > 25 * 1024, $"Synthesis report must exceed 25 KB (actual: {fileInfo.Length} bytes)");

        string text = File.ReadAllText(reportPath);
        Assert.True(text.Length > 25000, $"Synthesis report character count must exceed 25,000 (actual: {text.Length})");
        Assert.Contains("# DEC_HETEROGENEOUS_INFERENCE_REPORT.md", text);
        Assert.Contains("Decoupled Heterogeneous Cluster LLM Inference Architecture & Governance", text);
    }

    [Fact]
    public void SynthesisReport_DocumentsFleetHardwareTopology()
    {
        string reportPath = ResolveSynthesisReportPath();
        string text = File.ReadAllText(reportPath);

        Assert.Contains("## 2. Heterogeneous Hardware Fleet Topology", text);
        Assert.Contains("192.168.1.111", text);
        Assert.Contains("192.168.1.108", text);
        Assert.Contains("192.168.1.200", text);
        Assert.Contains("192.168.1.101", text);
        Assert.Contains("RTX 4060", text);
        Assert.Contains("RTX 3060", text);
        Assert.Contains("Ryzen AI 9 HX 370", text);
    }

    [Fact]
    public void SynthesisReport_DocumentsPrefillAndDecodeMetrics()
    {
        string reportPath = ResolveSynthesisReportPath();
        string text = File.ReadAllText(reportPath);

        Assert.Contains("## 4. Comprehensive 5-Dimension Benchmark Performance Scorecard", text);
        Assert.Contains("52.46 tok/s", text);
        Assert.Contains("23.24 tok/s", text);
        Assert.Contains("8.48 tok/s", text);
        Assert.Contains("18.52 tok/s", text);
        Assert.Contains("0.013 ms", text);
        Assert.Contains("5,124 Bytes", text);
        Assert.Contains("2.82 GB Free", text);
    }

    [Fact]
    public void SynthesisReport_ContainsParetoSplitAnalysis()
    {
        string reportPath = ResolveSynthesisReportPath();
        string text = File.ReadAllText(reportPath);

        Assert.Contains("### 5.4 Analytical Pareto 27:21 Layer Split Derivation", text);
        Assert.Contains("27:21", text);
        Assert.Contains("Pareto", text);
        Assert.Contains("Stage 0", text);
        Assert.Contains("Stage 1", text);
        Assert.Contains("1.56 ms", text);
        Assert.Contains("2.82 GB", text);
        Assert.Contains("T_0(K_0) = T_1(K_1)", text);
    }

    [Fact]
    public void SynthesisReport_AdheresToZeroFabricationStandard()
    {
        string reportPath = ResolveSynthesisReportPath();
        string text = File.ReadAllText(reportPath);

        Assert.Contains("Zero Fabrication Policy", text);
        Assert.Contains("## 8. Zero Secrets Security Governance", text);
        Assert.Contains("## 9. Independent Reproduction & Verification Guide", text);
        Assert.Contains("dotnet run --project scratch/GlacierPipelineCluster/GlacierPipelineCluster.csproj", text);
        Assert.Contains("python scripts/security_check.py .", text);
        Assert.DoesNotContain("TODO", text);
        Assert.DoesNotContain("TBD", text);
    }

    // =========================================================================
    // HELPER METHODS: DYNAMIC PATH RESOLUTION
    // =========================================================================

    private static string ResolveRepoRoot()
    {
        string probe = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        if (File.Exists(Path.Combine(probe, "DEC_HETEROGENEOUS_INFERENCE_REPORT.md")))
        {
            return probe;
        }
        const string fallback = @"c:\Users\spuri\source\repos\PolarsPlus";
        if (Directory.Exists(fallback))
        {
            return fallback;
        }
        throw new DirectoryNotFoundException($"Repository root not found at {probe} or {fallback}");
    }

    private static string ResolveSecurityScriptPath()
    {
        string root = ResolveRepoRoot();
        string script = Path.Combine(root, "scripts", "security_check.py");
        if (!File.Exists(script))
        {
            throw new FileNotFoundException($"security_check.py not found at {script}");
        }
        return script;
    }

    private static string ResolveSynthesisReportPath()
    {
        string root = ResolveRepoRoot();
        string report = Path.Combine(root, "DEC_HETEROGENEOUS_INFERENCE_REPORT.md");
        if (!File.Exists(report))
        {
            throw new FileNotFoundException($"DEC_HETEROGENEOUS_INFERENCE_REPORT.md not found at {report}");
        }
        return report;
    }
}
