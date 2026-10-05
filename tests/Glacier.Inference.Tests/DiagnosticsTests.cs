namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Glacier.Inference.Config;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Hardware;
using Glacier.Inference.Image.Gguf;
using Glacier.Inference.Video;
using Xunit;

public class DiagnosticsTests : IDisposable
{
    public DiagnosticsTests()
    {
        GlacierDiagnostics.Reset();
    }

    public void Dispose()
    {
        GlacierDiagnostics.Reset();
    }

    [Fact]
    public void GlacierDiagnostics_Default_UsesNullGlacierLogger()
    {
        GlacierDiagnostics.Reset();
        Assert.NotNull(GlacierDiagnostics.Logger);
        Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);
        Assert.False(GlacierDiagnostics.IsEnabled(LogLevel.Information));

        // Must not throw when logging through default null logger
        GlacierDiagnostics.LogTrace("Trace message");
        GlacierDiagnostics.LogDebug("Debug message");
        GlacierDiagnostics.LogInformation("Info message");
        GlacierDiagnostics.LogWarning("Warning message", new InvalidOperationException("Test"));
        GlacierDiagnostics.LogError("Error message", new Exception("Fail"));
        GlacierDiagnostics.LogCritical("Critical message");
    }

    [Fact]
    public void GlacierDiagnostics_AssignNull_RevertsToNullGlacierLogger()
    {
        GlacierDiagnostics.Logger = null!;
        Assert.NotNull(GlacierDiagnostics.Logger);
        Assert.Same(NullGlacierLogger.Instance, GlacierDiagnostics.Logger);
    }

    [Fact]
    public void DelegateGlacierLogger_RedirectsAndCapturesAllLogLevels()
    {
        var logs = new List<(LogLevel Level, string Message, Exception? Exception)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            logs.Add((level, msg, ex));
        }, LogLevel.Trace);

        var testEx = new InvalidOperationException("Test exception");

        GlacierDiagnostics.LogTrace("Trace entry");
        GlacierDiagnostics.LogDebug("Debug entry");
        GlacierDiagnostics.LogInformation("Info entry");
        GlacierDiagnostics.LogWarning("Warning entry", testEx);
        GlacierDiagnostics.LogError("Error entry", testEx);
        GlacierDiagnostics.LogCritical("Critical entry");

        Assert.Equal(6, logs.Count);

        Assert.Equal(LogLevel.Trace, logs[0].Level);
        Assert.Equal("Trace entry", logs[0].Message);
        Assert.Null(logs[0].Exception);

        Assert.Equal(LogLevel.Debug, logs[1].Level);
        Assert.Equal("Debug entry", logs[1].Message);
        Assert.Null(logs[1].Exception);

        Assert.Equal(LogLevel.Information, logs[2].Level);
        Assert.Equal("Info entry", logs[2].Message);
        Assert.Null(logs[2].Exception);

        Assert.Equal(LogLevel.Warning, logs[3].Level);
        Assert.Equal("Warning entry", logs[3].Message);
        Assert.Same(testEx, logs[3].Exception);

        Assert.Equal(LogLevel.Error, logs[4].Level);
        Assert.Equal("Error entry", logs[4].Message);
        Assert.Same(testEx, logs[4].Exception);

        Assert.Equal(LogLevel.Critical, logs[5].Level);
        Assert.Equal("Critical entry", logs[5].Message);
        Assert.Null(logs[5].Exception);
    }

    [Fact]
    public void DelegateGlacierLogger_SingleStringDelegate_CapturesFormattedMessages()
    {
        var lines = new List<string>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger(lines.Add, LogLevel.Information);

        GlacierDiagnostics.LogDebug("Should be filtered out");
        GlacierDiagnostics.LogInformation("Important info");
        GlacierDiagnostics.LogWarning("Watch out", new Exception("WarnEx"));

        Assert.Equal(2, lines.Count);
        Assert.Contains("[Information] Important info", lines[0]);
        Assert.Contains("[Warning] Watch out: System.Exception: WarnEx", lines[1]);
    }

    [Fact]
    public void DelegateGlacierLogger_RespectsMinimumLevelFilter()
    {
        var logs = new List<(LogLevel Level, string Message, Exception? Exception)>();
        GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
        {
            logs.Add((level, msg, ex));
        }, LogLevel.Warning);

        GlacierDiagnostics.LogTrace("Trace filtered");
        GlacierDiagnostics.LogDebug("Debug filtered");
        GlacierDiagnostics.LogInformation("Info filtered");
        GlacierDiagnostics.LogWarning("Warning kept");
        GlacierDiagnostics.LogError("Error kept");
        GlacierDiagnostics.LogCritical("Critical kept");

        Assert.Equal(3, logs.Count);
        Assert.Equal(LogLevel.Warning, logs[0].Level);
        Assert.Equal(LogLevel.Error, logs[1].Level);
        Assert.Equal(LogLevel.Critical, logs[2].Level);
    }

    [Fact]
    public void ConsoleGlacierLogger_HonorsLevelsAndDoesNotThrow()
    {
        var loggerWithoutColors = new ConsoleGlacierLogger(LogLevel.Information, useColors: false);
        Assert.True(loggerWithoutColors.IsEnabled(LogLevel.Information));
        Assert.True(loggerWithoutColors.IsEnabled(LogLevel.Warning));
        Assert.False(loggerWithoutColors.IsEnabled(LogLevel.Debug));
        Assert.False(loggerWithoutColors.IsEnabled(LogLevel.None));

        // Ensure logging methods execute cleanly
        loggerWithoutColors.Log(LogLevel.Information, "Console test info");
        loggerWithoutColors.Log(LogLevel.Warning, "Console test warning", new Exception("Test warn"));

        var loggerWithColors = new ConsoleGlacierLogger(LogLevel.Debug, useColors: true);
        Assert.True(loggerWithColors.IsEnabled(LogLevel.Debug));
        loggerWithColors.Log(LogLevel.Debug, "Console test debug with color");
        loggerWithColors.Log(LogLevel.Error, "Console test error with color", new Exception("Test err"));
    }

    [Fact]
    public void DiffusionGgufPipeline_ComponentFailure_LogsWarningAndExposesFallbackException()
    {
        string tmpGguf = Path.Combine(Path.GetTempPath(), $"glacier_test_diff_{Guid.NewGuid():N}.gguf");
        string tmpVae = Path.Combine(Path.GetTempPath(), $"glacier_corrupt_vae_{Guid.NewGuid():N}.safetensors");

        try
        {
            // Write minimal valid GGUF header with general.architecture="sd3" (16 inChannels)
            using (var fs = new FileStream(tmpGguf, FileMode.Create))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(0x46554747u); // 'GGUF'
                bw.Write(3u);          // Version 3
                bw.Write(0UL);         // 0 tensors
                bw.Write(1UL);         // 1 metadata KV

                // Key: "general.architecture"
                byte[] keyBytes = Encoding.UTF8.GetBytes("general.architecture");
                bw.Write((ulong)keyBytes.Length);
                bw.Write(keyBytes);

                // Type: String = 8
                bw.Write(8u);

                // Value: "sd3"
                byte[] valBytes = Encoding.UTF8.GetBytes("sd3");
                bw.Write((ulong)valBytes.Length);
                bw.Write(valBytes);
            }

            // Write corrupt safetensors file
            File.WriteAllText(tmpVae, "THIS IS CORRUPT NON-JSON NON-SAFETENSORS DATA");

            var capturedLogs = new List<(LogLevel Level, string Message, Exception? Ex)>();
            GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
            {
                capturedLogs.Add((level, msg, ex));
            }, LogLevel.Trace);

            using (var model = DiffusionGgufModel.Open(tmpGguf))
            using (var pipeline = new DiffusionGgufPipeline(model, vaeSafetensorsPath: tmpVae))
            {
                // Verify ActiveBackend reports "Cpu"
                Assert.Equal("Cpu", pipeline.ActiveBackend);

                // Verify FallbackException captured the failure
                Assert.NotNull(pipeline.FallbackException);

                // Verify diagnostic logging captured the warning with full details
                Assert.Contains(capturedLogs, l => l.Level == LogLevel.Warning &&
                                                   l.Message.Contains("Neural VAE load failed") &&
                                                   l.Ex != null);
            }
        }
        finally
        {
            try { if (File.Exists(tmpGguf)) File.Delete(tmpGguf); } catch { }
            try { if (File.Exists(tmpVae)) File.Delete(tmpVae); } catch { }
        }
    }

    [Fact]
    public void VideoGenerationPipeline_ComponentFailure_LogsWarningAndExposesFallbackException()
    {
        string tmpVideoVae = Path.Combine(Path.GetTempPath(), $"glacier_corrupt_video_vae_{Guid.NewGuid():N}.safetensors");

        try
        {
            File.WriteAllText(tmpVideoVae, "CORRUPT VIDEO VAE CONTENT FOR TESTING FALLBACK");

            var capturedLogs = new List<(LogLevel Level, string Message, Exception? Ex)>();
            GlacierDiagnostics.Logger = new DelegateGlacierLogger((level, msg, ex) =>
            {
                capturedLogs.Add((level, msg, ex));
            }, LogLevel.Trace);

            using var pipeline = new VideoGenerationPipeline(
                modelPath: "nonexistent_video_dit_model.gguf",
                vaePath: tmpVideoVae);

            // Verify ActiveBackend reports "Cpu"
            Assert.Equal("Cpu", pipeline.ActiveBackend);

            // Verify FallbackException captured
            Assert.NotNull(pipeline.FallbackException);

            // Verify diagnostic logging captured the warning
            Assert.Contains(capturedLogs, l => l.Level == LogLevel.Warning &&
                                               l.Message.Contains("Wan 3D VAE load failed") &&
                                               l.Ex != null);
        }
        finally
        {
            try { if (File.Exists(tmpVideoVae)) File.Delete(tmpVideoVae); } catch { }
        }
    }

    [Fact]
    public void InferenceSession_ActiveBackend_ReportsCpuOnCpuSession()
    {
        if (!File.Exists(CudaFactAttribute.ModelPath))
            return;

        using var session = new InferenceSession(
            CudaFactAttribute.ModelPath,
            maxSeqLen: 128,
            device: "cpu",
            engine: InferenceEngineType.Cpu);

        Assert.Equal("Cpu", session.ActiveBackend);
        Assert.Equal(InferenceEngineType.Cpu, session.Engine);
        Assert.Equal(GpuVendor.Cpu, session.Device.Vendor);
        Assert.False(session.IsGpuAccelerated);
        Assert.Null(session.FallbackException);
    }

    [Fact]
    public void InferenceSession_EngineAndDevice_ReflectActiveBackend()
    {
        if (!File.Exists(CudaFactAttribute.ModelPath))
            return;

        using var session = new InferenceSession(
            CudaFactAttribute.ModelPath,
            maxSeqLen: 128,
            device: "cpu");

        Assert.Equal("Cpu", session.ActiveBackend);
        Assert.Equal(InferenceEngineType.Cpu, session.Engine);
        Assert.Equal("cpu", session.Device.Id);
    }
}
