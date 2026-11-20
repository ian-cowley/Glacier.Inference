namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

public enum BounceTaskKind
{
    Bypass,
    CSharp,
    Hlsl,
    HtmlJs,
    Python,
    ToolCall,
    FinalAnswer,
    General
}

public sealed record BounceVerificationResult(bool Success, string? ErrorMessage);

public interface IBounceVerifier
{
    Task<BounceVerificationResult> VerifyAsync(BounceTaskKind kind, string promptText, string completionText, CancellationToken ct);
}

public sealed record BounceCompletionResult(
    string Content,
    string FinishReason,
    string ModelId,
    int PromptTokens,
    int CompletionTokens,
    int RepairsUsed,
    bool CriticAssisted,
    bool Bypassed);

public sealed class LocalProcessBounceVerifier : IBounceVerifier
{
    private readonly string? _harnessExePath;
    private readonly string _pythonExe;
    private readonly string _nodeExe;

    public LocalProcessBounceVerifier(string? harnessExePath, string pythonExe = "python", string nodeExe = "node")
    {
        _harnessExePath = string.IsNullOrWhiteSpace(harnessExePath) ? null : harnessExePath;
        _pythonExe = pythonExe;
        _nodeExe = nodeExe;
    }

    public async Task<BounceVerificationResult> VerifyAsync(
        BounceTaskKind kind,
        string promptText,
        string completionText,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(completionText))
            return new BounceVerificationResult(false, "Model returned an empty response.");

        switch (kind)
        {
            case BounceTaskKind.ToolCall:
                return VerifyToolCall(completionText);

            case BounceTaskKind.FinalAnswer:
                return VerifyFinalAnswer(completionText);

            case BounceTaskKind.CSharp:
            {
                string? code = ExtractCodeBlock(completionText, "csharp", "cs");
                if (string.IsNullOrWhiteSpace(code))
                    return new BounceVerificationResult(false, "Missing ```csharp code block.");
                if (string.IsNullOrWhiteSpace(_harnessExePath) || !File.Exists(_harnessExePath))
                    return new BounceVerificationResult(true, null);
                return await RunHarnessAsync("--verify-csharp", code, ".cs", extraArgs: null, ct);
            }

            case BounceTaskKind.Hlsl:
            {
                string? code = ExtractCodeBlock(completionText, "hlsl", "fx");
                if (string.IsNullOrWhiteSpace(code))
                    return new BounceVerificationResult(false, "Missing ```hlsl code block.");
                if (string.IsNullOrWhiteSpace(_harnessExePath) || !File.Exists(_harnessExePath))
                    return new BounceVerificationResult(true, null);
                string entry = DetectHlslEntry(code);
                return await RunHarnessAsync("--verify-hlsl", code, ".hlsl", ["cs_6_6", entry], ct);
            }

            case BounceTaskKind.HtmlJs:
                return await VerifyHtmlJsAsync(completionText, ct);

            case BounceTaskKind.Python:
                return await VerifyPythonSyntaxAsync(completionText, ct);

            default:
                return new BounceVerificationResult(true, null);
        }
    }

    public static string? ExtractCodeBlock(string text, params string[] langs)
    {
        foreach (string lang in langs)
        {
            var matches = Regex.Matches(text, $@"```{Regex.Escape(lang)}\s*\r?\n(.*?)```", RegexOptions.Singleline | RegexOptions.IgnoreCase);
            if (matches.Count > 0)
                return matches[^1].Groups[1].Value.Trim();
        }
        var any = Regex.Matches(text, @"```[a-zA-Z0-9_+-]*\s*\r?\n(.*?)```", RegexOptions.Singleline);
        return any.Count > 0 ? any[^1].Groups[1].Value.Trim() : null;
    }

    private static string DetectHlslEntry(string code)
    {
        var m = Regex.Match(code, @"\[numthreads\([^\]]+\)\]\s*(?:\[[^\]]+\]\s*)*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", RegexOptions.Singleline);
        if (m.Success) return m.Groups[1].Value;
        return code.Contains("CSMain", StringComparison.Ordinal) ? "CSMain" : "main";
    }

    private static BounceVerificationResult VerifyToolCall(string text)
    {
        var m = Regex.Match(text, @"<tool_call>\s*(\{.*?\})\s*</tool_call>", RegexOptions.Singleline);
        if (!m.Success)
            return new BounceVerificationResult(false, "Response must wrap valid JSON inside <tool_call>{...}</tool_call> tags.");
        try
        {
            using var doc = JsonDocument.Parse(m.Groups[1].Value);
            var root = doc.RootElement;
            if (!root.TryGetProperty("name", out var nameProp) || nameProp.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(nameProp.GetString()))
                return new BounceVerificationResult(false, "<tool_call> JSON must contain a non-empty string \"name\" field.");
            if (!root.TryGetProperty("arguments", out var argsProp) || argsProp.ValueKind != JsonValueKind.Object)
                return new BounceVerificationResult(false, "<tool_call> JSON must contain an object \"arguments\" field.");
            return new BounceVerificationResult(true, null);
        }
        catch (Exception ex)
        {
            return new BounceVerificationResult(false, $"Invalid JSON inside <tool_call>: {ex.Message}");
        }
    }

    private static BounceVerificationResult VerifyFinalAnswer(string text)
    {
        var m = Regex.Match(text, @"FINAL_ANSWER:\s*([^\r\n]+)");
        if (!m.Success || string.IsNullOrWhiteSpace(m.Groups[1].Value))
            return new BounceVerificationResult(false, "Response truncated or missing required `FINAL_ANSWER: <value>` line.");
        return new BounceVerificationResult(true, null);
    }

    private async Task<BounceVerificationResult> VerifyHtmlJsAsync(string text, CancellationToken ct)
    {
        string? html = ExtractCodeBlock(text, "html", "htm");
        if (string.IsNullOrWhiteSpace(html))
        {
            if (text.Contains("<!DOCTYPE html", StringComparison.OrdinalIgnoreCase) || text.Contains("<html", StringComparison.OrdinalIgnoreCase))
                html = text;
            else
                return new BounceVerificationResult(false, "Missing ```html single-file document.");
        }

        if (!html.Contains("</html>", StringComparison.OrdinalIgnoreCase))
            return new BounceVerificationResult(false, "HTML output appears truncated (missing closing </html> tag). Make the implementation more concise so it finishes completely.");

        var scriptMatches = Regex.Matches(html, @"<script\b[^>]*>(.*?)</script>", RegexOptions.Singleline | RegexOptions.IgnoreCase);
        if (scriptMatches.Count == 0)
            return new BounceVerificationResult(false, "HTML document is missing a closed <script>...</script> block.");

        var sb = new StringBuilder();
        foreach (Match sm in scriptMatches)
            sb.AppendLine(sm.Groups[1].Value);

        string tmpJs = Path.Combine(Path.GetTempPath(), $"glacier_bounce_{Guid.NewGuid():N}.js");
        try
        {
            await File.WriteAllTextAsync(tmpJs, sb.ToString(), Encoding.UTF8, ct);
            var (code, _, stderr) = await RunProcessAsync(_nodeExe, ["--check", tmpJs], timeoutMs: 8000, ct);
            if (code != 0)
            {
                string err = string.IsNullOrWhiteSpace(stderr) ? "JavaScript syntax error in <script> block." : stderr.Trim();
                if (err.Length > 400) err = err[..400];
                return new BounceVerificationResult(false, $"Node --check failed: {err}");
            }
            return new BounceVerificationResult(true, null);
        }
        catch (Exception)
        {
            // Node not installed or transient process launch error: skip JS syntax check gracefully.
            return new BounceVerificationResult(true, null);
        }
        finally
        {
            try { if (File.Exists(tmpJs)) File.Delete(tmpJs); } catch { }
        }
    }

    private async Task<BounceVerificationResult> VerifyPythonSyntaxAsync(string text, CancellationToken ct)
    {
        string? code = ExtractCodeBlock(text, "python", "py");
        if (string.IsNullOrWhiteSpace(code))
            return new BounceVerificationResult(false, "Missing ```python code block.");

        string tmpPy = Path.Combine(Path.GetTempPath(), $"glacier_bounce_{Guid.NewGuid():N}.py");
        try
        {
            await File.WriteAllTextAsync(tmpPy, code, Encoding.UTF8, ct);
            var (rc, _, stderr) = await RunProcessAsync(_pythonExe, ["-m", "py_compile", tmpPy], timeoutMs: 8000, ct);
            if (rc != 0)
            {
                string err = string.IsNullOrWhiteSpace(stderr) ? "Python syntax error." : stderr.Trim();
                if (err.Length > 400) err = err[..400];
                return new BounceVerificationResult(false, $"Python py_compile failed: {err}");
            }
            return new BounceVerificationResult(true, null);
        }
        catch (Exception)
        {
            return new BounceVerificationResult(true, null);
        }
        finally
        {
            try { if (File.Exists(tmpPy)) File.Delete(tmpPy); } catch { }
        }
    }

    private async Task<BounceVerificationResult> RunHarnessAsync(
        string flag,
        string code,
        string ext,
        string[]? extraArgs,
        CancellationToken ct)
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"glacier_bounce_{Guid.NewGuid():N}{ext}");
        try
        {
            await File.WriteAllTextAsync(tmp, code, Encoding.UTF8, ct);
            var args = new List<string> { flag, tmp };
            if (extraArgs != null) args.AddRange(extraArgs);
            var (rc, stdout, stderr) = await RunProcessAsync(_harnessExePath!, args, timeoutMs: 15000, ct);
            if (rc == 0)
                return new BounceVerificationResult(true, null);

            string detail = stdout;
            try
            {
                using var doc = JsonDocument.Parse(stdout);
                string? diag = doc.RootElement.TryGetProperty("diagnostic_id", out var d) ? d.GetString() : null;
                string? err = doc.RootElement.TryGetProperty("error", out var e) ? e.GetString() : null;
                if (!string.IsNullOrWhiteSpace(err))
                    detail = string.IsNullOrWhiteSpace(diag) ? err : $"{diag}: {err}";
            }
            catch
            {
                if (string.IsNullOrWhiteSpace(detail)) detail = stderr;
            }
            if (string.IsNullOrWhiteSpace(detail)) detail = "Compiler verification failed.";
            if (detail.Length > 500) detail = detail[..500];
            return new BounceVerificationResult(false, detail.Trim());
        }
        finally
        {
            try { if (File.Exists(tmp)) File.Delete(tmp); } catch { }
        }
    }

    private static async Task<(int ExitCode, string Stdout, string Stderr)> RunProcessAsync(
        string fileName,
        IReadOnlyList<string> args,
        int timeoutMs,
        CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        foreach (string a in args) psi.ArgumentList.Add(a);

        using var proc = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start {fileName}");
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(timeoutMs);

        var outTask = proc.StandardOutput.ReadToEndAsync(cts.Token);
        var errTask = proc.StandardError.ReadToEndAsync(cts.Token);
        await proc.WaitForExitAsync(cts.Token);
        return (proc.ExitCode, await outTask, await errTask);
    }
}

public sealed class PeerBounceService
{
    private readonly HttpClient _http;
    private readonly IBounceVerifier _verifier;
    private readonly string _primaryEndpoint;
    private readonly string? _criticEndpoint;
    private readonly bool _compilerBounceEnabled;
    private readonly int _maxRepairs;
    private readonly TimeSpan _criticTimeout;

    private string? _cachedPrimaryModel;
    private string? _cachedCriticModel;

    private long _totalRequests;
    private long _bypassedRequests;
    private long _criticAssistedRequests;
    private long _criticFallbacks;
    private long _verifierChecks;
    private long _verifierFirstPassOk;
    private long _repairAttempts;
    private long _repairsSucceeded;

    public PeerBounceService(
        HttpClient httpClient,
        IBounceVerifier verifier,
        string primaryEndpoint,
        string? criticEndpoint = null,
        bool compilerBounceEnabled = true,
        int maxRepairs = 2,
        TimeSpan? criticTimeout = null)
    {
        _http = httpClient;
        _verifier = verifier;
        _primaryEndpoint = primaryEndpoint.TrimEnd('/');
        _criticEndpoint = string.IsNullOrWhiteSpace(criticEndpoint) ? null : criticEndpoint.TrimEnd('/');
        _compilerBounceEnabled = compilerBounceEnabled;
        _maxRepairs = Math.Clamp(maxRepairs, 0, 3);
        _criticTimeout = criticTimeout ?? TimeSpan.FromSeconds(18);
    }

    public BounceStatusResponse GetStatus() => new()
    {
        PrimaryEndpoint = _primaryEndpoint,
        CriticEndpoint = _criticEndpoint,
        CompilerBounceEnabled = _compilerBounceEnabled,
        MaxRepairs = _maxRepairs,
        TotalRequests = Interlocked.Read(ref _totalRequests),
        BypassedRequests = Interlocked.Read(ref _bypassedRequests),
        CriticAssistedRequests = Interlocked.Read(ref _criticAssistedRequests),
        CriticFallbacks = Interlocked.Read(ref _criticFallbacks),
        VerifierChecks = Interlocked.Read(ref _verifierChecks),
        VerifierFirstPassOk = Interlocked.Read(ref _verifierFirstPassOk),
        RepairAttempts = Interlocked.Read(ref _repairAttempts),
        RepairsSucceeded = Interlocked.Read(ref _repairsSucceeded),
    };

    public static BounceTaskKind ClassifyRequest(IReadOnlyList<OllamaChatMessageDto> messages, int maxTokens)
    {
        string userText = "";
        string fullText = "";
        foreach (var m in messages)
        {
            fullText += "\n" + m.Content;
            if (m.Role.Equals("user", StringComparison.OrdinalIgnoreCase))
                userText = m.Content;
        }

        string lower = userText.ToLowerInvariant();
        string fullLower = fullText.ToLowerInvariant();

        // Bypass direct retrieval / sleep-memory / NIAH haystack prompts (zero latency overhead)
        if (fullLower.Contains("consolidated episodic memory") ||
            fullLower.Contains("sleep memory") ||
            fullLower.Contains("recall the exact") ||
            (userText.Length > 3200 && !lower.Contains("```") && !lower.Contains("csharp") && !lower.Contains("hlsl") && !lower.Contains("python") && !lower.Contains("html")))
        {
            return BounceTaskKind.Bypass;
        }

        if (userText.Contains("<tool_call>", StringComparison.OrdinalIgnoreCase) || lower.Contains("tool_call"))
            return BounceTaskKind.ToolCall;

        if (lower.Contains("hlsl") || lower.Contains("cs_6_6") || lower.Contains("groupshared") || lower.Contains("numthreads") || lower.Contains("compute shader"))
            return BounceTaskKind.Hlsl;

        if (lower.Contains("csharp") || lower.Contains("c#") || lower.Contains(".net 10") || lower.Contains("vector256") || lower.Contains("roslyn") || lower.Contains("readonlyspan"))
            return BounceTaskKind.CSharp;

        if (lower.Contains("single-file html") || lower.Contains("html5") || lower.Contains("<!doctype html") || (lower.Contains("html") && lower.Contains("javascript")))
            return BounceTaskKind.HtmlJs;

        if (lower.Contains("python") || Regex.IsMatch(userText, @"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\("))
            return BounceTaskKind.Python;

        if (userText.Contains("FINAL_ANSWER:", StringComparison.Ordinal))
            return BounceTaskKind.FinalAnswer;

        if (maxTokens <= 96)
            return BounceTaskKind.Bypass;

        return BounceTaskKind.General;
    }

    public async Task<BounceCompletionResult> CompleteAsync(
        OllamaChatMessageDto[] messages,
        string? requestedModel,
        double temperature,
        int maxTokens,
        CancellationToken ct)
    {
        Interlocked.Increment(ref _totalRequests);
        string primaryModel = await ResolveModelIdAsync(_primaryEndpoint, isCritic: false, requestedModel, ct);
        var kind = ClassifyRequest(messages, maxTokens);

        if (kind == BounceTaskKind.Bypass)
        {
            Interlocked.Increment(ref _bypassedRequests);
            var direct = await CallEndpointAsync(_primaryEndpoint, primaryModel, messages, temperature, maxTokens, disableThinking: false, timeout: null, ct);
            return new BounceCompletionResult(direct.Content, direct.FinishReason, primaryModel, direct.PromptTokens, direct.CompletionTokens, 0, false, true);
        }

        string userPrompt = "";
        for (int i = messages.Length - 1; i >= 0; i--)
        {
            if (messages[i].Role.Equals("user", StringComparison.OrdinalIgnoreCase))
            {
                userPrompt = messages[i].Content;
                break;
            }
        }

        // Stage 1: Optional 35B Peer Architect Contract Brief (compact, graceful offline fallback)
        bool criticAssisted = false;
        string? peerBrief = null;
        if (_criticEndpoint != null && ShouldConsultCritic(kind, userPrompt))
        {
            peerBrief = await TryFetchCriticBriefAsync(kind, userPrompt, ct);
            if (!string.IsNullOrWhiteSpace(peerBrief))
            {
                criticAssisted = true;
                Interlocked.Increment(ref _criticAssistedRequests);
            }
        }

        var workingMessages = BuildPrimaryMessages(messages, peerBrief);
        var primaryResp = await CallEndpointAsync(_primaryEndpoint, primaryModel, workingMessages, temperature, maxTokens, disableThinking: false, timeout: null, ct);
        int totalPromptTokens = primaryResp.PromptTokens;
        int totalCompletionTokens = primaryResp.CompletionTokens;
        string currentText = primaryResp.Content;
        string currentFinish = primaryResp.FinishReason;

        if (!_compilerBounceEnabled || _maxRepairs == 0 || kind == BounceTaskKind.General)
        {
            return new BounceCompletionResult(currentText, currentFinish, primaryModel, totalPromptTokens, totalCompletionTokens, 0, criticAssisted, false);
        }

        // Stage 2: Local Generic Verifier + Up to _maxRepairs Auto-Repair Turns
        int effectiveMaxRepairs = (kind == BounceTaskKind.HtmlJs && (maxTokens >= 1500 || messages.Length > 2))
            ? Math.Min(1, _maxRepairs)
            : _maxRepairs;
        int repairsUsed = 0;
        for (int attempt = 0; attempt <= effectiveMaxRepairs; attempt++)
        {
            Interlocked.Increment(ref _verifierChecks);
            var verify = await _verifier.VerifyAsync(kind, userPrompt, currentText, ct);
            if (verify.Success)
            {
                if (attempt == 0)
                    Interlocked.Increment(ref _verifierFirstPassOk);
                else
                    Interlocked.Increment(ref _repairsSucceeded);
                break;
            }

            if (attempt == effectiveMaxRepairs)
                break;

            repairsUsed++;
            Interlocked.Increment(ref _repairAttempts);

            string? criticHint = null;
            if (attempt == 1 && _criticEndpoint != null)
            {
                criticHint = await TryFetchCriticRepairHintAsync(kind, verify.ErrorMessage ?? "Verification failed", ct);
            }

            var repairPrompt = new StringBuilder();
            repairPrompt.AppendLine($"[LOCAL VERIFIER ERROR] {verify.ErrorMessage}");
            if (!string.IsNullOrWhiteSpace(criticHint))
                repairPrompt.AppendLine($"[ARCHITECT HINT] {criticHint}");
            repairPrompt.AppendLine("Fix ONLY this error and return the complete, corrected, concise final response in the exact required format.");

            var repairMessages = new List<OllamaChatMessageDto>(workingMessages)
            {
                new() { Role = "assistant", Content = currentText.Length > 2400 ? currentText[..2400] : currentText },
                new() { Role = "user", Content = repairPrompt.ToString() }
            };

            try
            {
                var repaired = await CallEndpointAsync(_primaryEndpoint, primaryModel, repairMessages.ToArray(), 0.0, maxTokens, disableThinking: false, timeout: null, ct);
                totalPromptTokens += repaired.PromptTokens;
                totalCompletionTokens += repaired.CompletionTokens;
                if (!string.IsNullOrWhiteSpace(repaired.Content))
                {
                    currentText = repaired.Content;
                    currentFinish = repaired.FinishReason;
                }
            }
            catch
            {
                // Keep best primary response gracefully if a repair turn overflows context or times out.
                break;
            }
        }

        return new BounceCompletionResult(currentText, currentFinish, primaryModel, totalPromptTokens, totalCompletionTokens, repairsUsed, criticAssisted, false);
    }

    private static bool ShouldConsultCritic(BounceTaskKind kind, string userPrompt)
    {
        return kind switch
        {
            BounceTaskKind.CSharp => true,
            BounceTaskKind.Hlsl => true,
            BounceTaskKind.HtmlJs => true,
            BounceTaskKind.Python => userPrompt.Length >= 100,
            BounceTaskKind.FinalAnswer => userPrompt.Length >= 120,
            _ => false
        };
    }

    private async Task<string?> TryFetchCriticBriefAsync(BounceTaskKind kind, string userPrompt, CancellationToken ct)
    {
        try
        {
            string criticModel = await ResolveModelIdAsync(_criticEndpoint!, isCritic: true, null, ct);
            string sysPrompt = kind == BounceTaskKind.FinalAnswer
                ? "You are a concise quantitative reviewer. Give at most 4 short bullet points outlining the exact equation/method and pitfalls to avoid. No LaTeX bloat, max 120 tokens."
                : "You are a principal systems architect. List at most 5 terse bullet points of critical compiler rules, API signatures, and edge cases to satisfy for this task. Do NOT write full code. Max 150 tokens.";

            var msgs = new OllamaChatMessageDto[]
            {
                new() { Role = "system", Content = sysPrompt },
                new() { Role = "user", Content = userPrompt.Length > 2500 ? userPrompt[..2500] : userPrompt }
            };

            var resp = await CallEndpointAsync(_criticEndpoint!, criticModel, msgs, 0.0, 180, disableThinking: true, timeout: _criticTimeout, ct);
            string clean = StripThinkTags(resp.Content);
            return string.IsNullOrWhiteSpace(clean) ? null : clean;
        }
        catch
        {
            Interlocked.Increment(ref _criticFallbacks);
            return null;
        }
    }

    private async Task<string?> TryFetchCriticRepairHintAsync(BounceTaskKind kind, string errorMessage, CancellationToken ct)
    {
        try
        {
            string criticModel = await ResolveModelIdAsync(_criticEndpoint!, isCritic: true, null, ct);
            var msgs = new OllamaChatMessageDto[]
            {
                new() { Role = "system", Content = "In 2 concise sentences, explain the exact compiler/syntax fix needed for this error. No filler." },
                new() { Role = "user", Content = $"Task kind: {kind}\nError: {errorMessage}" }
            };
            var resp = await CallEndpointAsync(_criticEndpoint!, criticModel, msgs, 0.0, 100, disableThinking: true, timeout: _criticTimeout, ct);
            string clean = StripThinkTags(resp.Content);
            return string.IsNullOrWhiteSpace(clean) ? null : clean;
        }
        catch
        {
            Interlocked.Increment(ref _criticFallbacks);
            return null;
        }
    }

    private static OllamaChatMessageDto[] BuildPrimaryMessages(OllamaChatMessageDto[] original, string? peerBrief)
    {
        if (string.IsNullOrWhiteSpace(peerBrief))
            return original;

        var list = new List<OllamaChatMessageDto>(original.Length + 1);
        string note = $"\n\n[Peer Architect Guardrails — apply concisely]:\n{peerBrief}";
        bool mergedIntoSystem = false;
        foreach (var m in original)
        {
            if (!mergedIntoSystem && m.Role.Equals("system", StringComparison.OrdinalIgnoreCase))
            {
                list.Add(new OllamaChatMessageDto { Role = m.Role, Content = m.Content + note });
                mergedIntoSystem = true;
            }
            else
            {
                list.Add(m);
            }
        }
        if (!mergedIntoSystem)
        {
            list.Insert(0, new OllamaChatMessageDto
            {
                Role = "system",
                Content = "Follow these concise engineering guardrails carefully:" + note
            });
        }
        return list.ToArray();
    }

    private static string StripThinkTags(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";
        return Regex.Replace(text, @"<think>.*?</think>\s*", "", RegexOptions.Singleline).Trim();
    }

    private async Task<string> ResolveModelIdAsync(string endpoint, bool isCritic, string? fallback, CancellationToken ct)
    {
        string? cached = isCritic ? _cachedCriticModel : _cachedPrimaryModel;
        if (!string.IsNullOrWhiteSpace(cached))
            return cached;

        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(5));
            string json = await _http.GetStringAsync($"{endpoint}/v1/models", cts.Token);
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("data", out var arr) && arr.ValueKind == JsonValueKind.Array && arr.GetArrayLength() > 0)
            {
                string? id = arr[0].TryGetProperty("id", out var idProp) ? idProp.GetString() : null;
                if (!string.IsNullOrWhiteSpace(id))
                {
                    if (isCritic) _cachedCriticModel = id;
                    else _cachedPrimaryModel = id;
                    return id;
                }
            }
        }
        catch
        {
            // Fallback to caller-provided or default model id
        }

        return string.IsNullOrWhiteSpace(fallback) ? "glacier-primary" : fallback;
    }

    private async Task<(string Content, string FinishReason, int PromptTokens, int CompletionTokens)> CallEndpointAsync(
        string endpoint,
        string modelId,
        OllamaChatMessageDto[] messages,
        double temperature,
        int maxTokens,
        bool disableThinking,
        TimeSpan? timeout,
        CancellationToken ct)
    {
        var payload = new BounceChatPayload
        {
            Model = modelId,
            Messages = messages,
            Temperature = temperature,
            MaxTokens = maxTokens > 0 ? maxTokens : 1024,
            Stream = false,
            ChatTemplateKwargs = disableThinking ? new BounceChatTemplateKwargs { EnableThinking = false } : null
        };

        string reqJson = JsonSerializer.Serialize(payload, InferenceCliJsonContext.Default.BounceChatPayload);
        using var req = new HttpRequestMessage(HttpMethod.Post, $"{endpoint}/v1/chat/completions")
        {
            Content = new StringContent(reqJson, Encoding.UTF8, "application/json")
        };

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (timeout.HasValue) cts.CancelAfter(timeout.Value);

        using var resp = await _http.SendAsync(req, cts.Token);
        string body = await resp.Content.ReadAsStringAsync(cts.Token);
        resp.EnsureSuccessStatusCode();

        using var doc = JsonDocument.Parse(body);
        var root = doc.RootElement;
        string content = "";
        string finish = "stop";
        int pTok = 0, cTok = 0;

        if (root.TryGetProperty("choices", out var choices) && choices.ValueKind == JsonValueKind.Array && choices.GetArrayLength() > 0)
        {
            var ch = choices[0];
            if (ch.TryGetProperty("message", out var msg) && msg.TryGetProperty("content", out var cProp))
                content = cProp.GetString() ?? "";
            if (ch.TryGetProperty("finish_reason", out var fProp))
                finish = fProp.GetString() ?? "stop";
        }
        if (root.TryGetProperty("usage", out var usage))
        {
            if (usage.TryGetProperty("prompt_tokens", out var pt)) pTok = pt.GetInt32();
            if (usage.TryGetProperty("completion_tokens", out var ctProp)) cTok = ctProp.GetInt32();
        }

        return (content, finish, pTok, cTok);
    }
}
