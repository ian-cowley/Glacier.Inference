namespace Glacier.Inference.Cli;

using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Serve.Inference.Bounce;
using Glacier.Serve.Inference.Http;
using Glacier.Serve.Server;

public static partial class Program
{
    private static async Task<int> RunBounceAsync(string[] args)
    {
        if (HasHelpFlag(args))
        {
            PrintBounceHelp();
            return 0;
        }

        string primaryEndpoint = "http://192.168.1.108:18080";
        string? criticEndpoint = null;
        string host = "127.0.0.1";
        int port = 18090;
        string? harnessPath = null;
        int maxRepairs = 2;
        bool compilerBounce = true;
        string modelAlias = "Glacier-3Machine-Bounce";

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (arg is "--primary" or "--critic" or "--peer-critic" or "--host" or "--port" or "--harness" or "--max-repairs" or "--model-alias")
            {
                if (i + 1 >= args.Length)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: {arg} requires a value.");
                    Console.ResetColor();
                    return 1;
                }
            }

            switch (arg)
            {
                case "--primary":
                    primaryEndpoint = args[++i];
                    break;
                case "--critic":
                case "--peer-critic":
                    criticEndpoint = args[++i];
                    break;
                case "--host":
                    host = args[++i];
                    break;
                case "--port":
                    if (!int.TryParse(args[++i], out int p) || p < 1 || p > 65535)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Error.WriteLine("Error: --port must be a valid port number between 1 and 65535.");
                        Console.ResetColor();
                        return 1;
                    }
                    port = p;
                    break;
                case "--harness":
                    harnessPath = args[++i];
                    break;
                case "--max-repairs":
                    if (!int.TryParse(args[++i], out int mr) || mr < 0)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Error.WriteLine("Error: --max-repairs must be a non-negative integer.");
                        Console.ResetColor();
                        return 1;
                    }
                    maxRepairs = mr;
                    break;
                case "--model-alias":
                    modelAlias = args[++i];
                    break;
                case "--no-compiler-bounce":
                    compilerBounce = false;
                    break;
                case "--compiler-bounce":
                    compilerBounce = true;
                    break;
                default:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Error: Unknown option '{arg}'.");
                    Console.ResetColor();
                    PrintBounceHelp();
                    return 1;
            }
        }

        if (string.IsNullOrWhiteSpace(harnessPath))
        {
            harnessPath = ResolveDefaultHarnessPath();
        }

        using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(600) };
        var verifier = new SandboxedProcessBounceVerifier(harnessPath);
        var bounceService = new PeerBounceService(
            httpClient,
            verifier,
            primaryEndpoint,
            criticEndpoint,
            compilerBounceEnabled: compilerBounce,
            maxRepairs: maxRepairs);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔═══════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║        GLACIER.INFERENCE - 3-Machine Critic + Compiler Bounce        ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine($"  Listen:   http://{host}:{port}");
        Console.WriteLine($"  Primary:  {primaryEndpoint}");
        Console.WriteLine($"  Critic:   {criticEndpoint ?? "(disabled)"}");
        Console.WriteLine($"  Verifier: {(compilerBounce ? $"ENABLED (maxRepairs={maxRepairs}, harness={harnessPath ?? "syntax-only"}, sandbox=OsProcessSandbox)" : "DISABLED")}");
        Console.WriteLine("  Routes:   GET /v1/models, POST /v1/chat/completions, POST /api/bounce/chat/completions, GET /api/bounce/status, GET /health");
        Console.WriteLine();

        using var app = GlacierServeApp.CreateBuilder()
            .UseHost(host)
            .UsePort(port)
            .Build();
        app.MapPeerBounce(bounceService, mapOpenAiAndHealthRoutes: true, modelAlias: modelAlias);

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        await app.StartAsync(cts.Token);

        try
        {
            await Task.Delay(Timeout.Infinite, cts.Token);
        }
        catch (TaskCanceledException)
        {
        }

        return 0;
    }

    public static string? ResolveDefaultHarnessPath(string? currentDirectory = null, string? baseDirectory = null)
    {
        string exeName = OperatingSystem.IsWindows() ? "Glacier.EvaluationHarness.exe" : "Glacier.EvaluationHarness";
        string baseDir = string.IsNullOrWhiteSpace(baseDirectory) ? AppContext.BaseDirectory : baseDirectory;
        string cwd = string.IsNullOrWhiteSpace(currentDirectory) ? Directory.GetCurrentDirectory() : currentDirectory;

        // 1. Direct sibling in AppContext.BaseDirectory or CWD
        string directBase = Path.GetFullPath(Path.Combine(baseDir, exeName));
        if (File.Exists(directBase))
            return directBase;

        string directCwd = Path.GetFullPath(Path.Combine(cwd, exeName));
        if (File.Exists(directCwd))
            return directCwd;

        // 2. Walk up from CWD and BaseDirectory looking for repo-layout bin output
        foreach (string startRoot in new[] { cwd, baseDir })
        {
            string? dir = startRoot;
            for (int depth = 0; depth < 6 && !string.IsNullOrEmpty(dir); depth++)
            {
                string releaseCandidate = Path.GetFullPath(Path.Combine(dir, "tests", "Glacier.EvaluationHarness", "bin", "Release", "net10.0", exeName));
                if (File.Exists(releaseCandidate))
                    return releaseCandidate;

                string debugCandidate = Path.GetFullPath(Path.Combine(dir, "tests", "Glacier.EvaluationHarness", "bin", "Debug", "net10.0", exeName));
                if (File.Exists(debugCandidate))
                    return debugCandidate;

                dir = Path.GetDirectoryName(dir);
            }
        }

        return null;
    }
}
