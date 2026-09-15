namespace Glacier.Inference.Config;

using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Glacier.Inference.Hardware;

/// <summary>
/// Persistent user configuration for hardware acceleration and inference execution.
/// Stored in %LOCALAPPDATA%\Glacier\Inference\settings.json.
/// </summary>
public sealed class GlacierSettings
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Converters = { new JsonStringEnumConverter() }
    };

    public string? DeviceId { get; set; } = "auto";
    public InferenceEngineType Engine { get; set; } = InferenceEngineType.Auto;
    public bool FallbackToCpu { get; set; } = true;
    public int MaxSeqLen { get; set; } = 4096;
    public float DefaultTemperature { get; set; } = 0.7f;
    public int DefaultTopK { get; set; } = 40;
    public float DefaultTopP { get; set; } = 0.95f;

    public static string GetSettingsDirectory()
    {
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppData))
            localAppData = AppContext.BaseDirectory;

        return Path.Combine(localAppData, "Glacier", "Inference");
    }

    public static string GetSettingsFilePath() => Path.Combine(GetSettingsDirectory(), "settings.json");

    public static GlacierSettings Load()
    {
        string path = GetSettingsFilePath();
        try
        {
            if (File.Exists(path))
            {
                string json = File.ReadAllText(path);
                var settings = JsonSerializer.Deserialize<GlacierSettings>(json, JsonOpts);
                if (settings != null) return settings;
            }
        }
        catch
        {
            // Fall back to defaults on read error
        }

        return new GlacierSettings();
    }

    public void Save()
    {
        string dir = GetSettingsDirectory();
        Directory.CreateDirectory(dir);
        string path = GetSettingsFilePath();
        string json = JsonSerializer.Serialize(this, JsonOpts);
        File.WriteAllText(path, json);
    }

    public static bool ValidateSafety(DeviceInfo device, InferenceEngineType engine, out string? errorMessage)
    {
        if (engine == InferenceEngineType.Auto || device.IsSafeEngine(engine))
        {
            errorMessage = null;
            return true;
        }

        string safeList = string.Join(", ", device.SupportedEngines);
        errorMessage = $"Engine '{engine}' is not a verified safe option for device '{device.Name}'. Safe engines for this device: [{safeList}].";
        return false;
    }

    /// <summary>
    /// Harmonizes CLI parameters, persisted settings, and hardware discovery into a validated execution target.
    /// </summary>
    public static (DeviceInfo Device, InferenceEngineType Engine) ResolveTarget(
        string? cliDevice,
        string? cliEngine)
    {
        var settings = Load();

        // 1. Resolve Device
        string deviceQuery = !string.IsNullOrWhiteSpace(cliDevice) ? cliDevice : (settings.DeviceId ?? "auto");
        var device = DeviceManager.ResolveDevice(deviceQuery);

        // 2. Resolve Engine
        InferenceEngineType engine = InferenceEngineType.Auto;
        if (!string.IsNullOrWhiteSpace(cliEngine))
        {
            string clean = cliEngine.Trim().ToLowerInvariant();
            if (clean is "baremetal" or "bare-metal" or "sass" or "cuda" or "hip" or "rocm")
            {
                engine = InferenceEngineType.BareMetal;
            }
            else if (clean is "directml" or "dml" or "dx12" or "d3d12" or "direct3d12")
            {
                engine = InferenceEngineType.DirectML;
            }
            else if (clean is "vulkan" or "vk" or "coopmat" or "cooperative-matrix")
            {
                engine = InferenceEngineType.Vulkan;
            }
            else if (clean is "cpu" or "simd")
            {
                engine = InferenceEngineType.Cpu;
            }
            else if (clean is "auto")
            {
                engine = InferenceEngineType.Auto;
            }
            else if (Enum.TryParse<InferenceEngineType>(cliEngine, ignoreCase: true, out var parsed))
            {
                engine = parsed;
            }
            else
            {
                throw new ArgumentException($"Unknown engine '{cliEngine}'. Valid options: auto, baremetal, vulkan, directml, cpu");
            }
        }
        else if (string.IsNullOrWhiteSpace(cliDevice) && settings.Engine != InferenceEngineType.Auto)
        {
            engine = settings.Engine;
        }
        else if (!string.IsNullOrWhiteSpace(cliDevice) && settings.Engine != InferenceEngineType.Auto && device.IsSafeEngine(settings.Engine))
        {
            engine = settings.Engine;
        }

        // If engine is Auto or incompatible with device, resolve to device's recommended safe engine
        if (engine == InferenceEngineType.Auto || !device.IsSafeEngine(engine))
        {
            if (string.IsNullOrWhiteSpace(cliEngine))
            {
                engine = device.RecommendedEngine;
            }
        }

        // Validate safety
        if (!ValidateSafety(device, engine, out string? error))
        {
            throw new InvalidOperationException(error);
        }

        return (device, engine);
    }
}
