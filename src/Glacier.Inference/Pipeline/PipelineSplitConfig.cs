namespace Glacier.Inference.Pipeline;

using System;
using System.Collections.Generic;
using Glacier.Inference.Hardware;
using Glacier.Inference.Model;

/// <summary>
/// Specification for a single pipeline stage in a partitioned LLM execution.
/// </summary>
public sealed record PipelineStageSpec(
    DeviceInfo Device,
    InferenceEngineType Engine,
    int StartLayer,
    int LayerCount
)
{
    public int EndLayer => StartLayer + LayerCount;
}

/// <summary>
/// Parses and calculates layer allocations across heterogeneous GPUs and host CPU.
/// </summary>
public static class PipelineSplitConfig
{
    /// <summary>
    /// Parses a user-supplied split string or automatically partitions layers based on hardware VRAM.
    /// Returns null if pipeline parallelism is not requested.
    /// </summary>
    public static List<PipelineStageSpec>? ResolveStages(string? spec, ModelWeights weights)
    {
        if (string.IsNullOrWhiteSpace(spec))
            return null;

        var allDevices = DeviceManager.GetDevices();
        var gpus = new List<DeviceInfo>();
        DeviceInfo? cpuDevice = null;

        foreach (var dev in allDevices)
        {
            if (dev.Vendor != GpuVendor.Cpu)
                gpus.Add(dev);
            else
                cpuDevice = dev;
        }
        cpuDevice ??= DeviceManager.ResolveDevice("cpu");

        int totalLayers = weights.BlockCount;
        var result = new List<PipelineStageSpec>();

        if (spec.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            // Auto partition based on available hardware
            if (gpus.Count >= 2)
            {
                // Prioritize discrete NVIDIA GPU first (GDDR6), then AMD/Intel secondary (UMA)
                DeviceInfo firstGpu = gpus[0];
                DeviceInfo secondGpu = gpus[1];
                for (int i = 0; i < gpus.Count; i++)
                {
                    if (gpus[i].Vendor == GpuVendor.Nvidia && gpus[i].SupportedEngines.Contains(InferenceEngineType.BareMetal))
                    {
                        firstGpu = gpus[i];
                        secondGpu = (i == 0 && gpus.Count > 1) ? gpus[1] : gpus[0];
                        break;
                    }
                }

                // Calculate split based on VRAM capacity
                // Stage 0 gets layers up to ~70% of dedicated VRAM
                long modelBytes = weights.FileSizeBytes;
                long bytesPerLayer = totalLayers > 0 ? (modelBytes / totalLayers) : 100_000_000;
                long safeVram = (long)(firstGpu.DedicatedVramBytes * 0.70);
                int stage0Layers = (int)Math.Clamp(safeVram / bytesPerLayer, 1, totalLayers - 1);
                int stage1Layers = totalLayers - stage0Layers;

                var engine0 = firstGpu.Vendor == GpuVendor.Nvidia ? InferenceEngineType.BareMetal : InferenceEngineType.DirectML;
                var engine1 = secondGpu.Vendor == GpuVendor.Nvidia ? InferenceEngineType.BareMetal : InferenceEngineType.DirectML;

                result.Add(new PipelineStageSpec(firstGpu, engine0, 0, stage0Layers));
                result.Add(new PipelineStageSpec(secondGpu, engine1, stage0Layers, stage1Layers));
                return result;
            }
            else if (gpus.Count == 1)
            {
                // Single GPU + CPU split
                long modelBytes = weights.FileSizeBytes;
                long bytesPerLayer = totalLayers > 0 ? (modelBytes / totalLayers) : 100_000_000;
                long safeVram = (long)(gpus[0].DedicatedVramBytes * 0.70);
                int gpuLayers = (int)Math.Clamp(safeVram / bytesPerLayer, 1, totalLayers);
                var engine0 = gpus[0].Vendor == GpuVendor.Nvidia ? InferenceEngineType.BareMetal : InferenceEngineType.DirectML;

                result.Add(new PipelineStageSpec(gpus[0], engine0, 0, gpuLayers));
                if (gpuLayers < totalLayers)
                {
                    result.Add(new PipelineStageSpec(cpuDevice, InferenceEngineType.Cpu, gpuLayers, totalLayers - gpuLayers));
                }
                return result;
            }
            else
            {
                // Only CPU available
                result.Add(new PipelineStageSpec(cpuDevice, InferenceEngineType.Cpu, 0, totalLayers));
                return result;
            }
        }

        // Manual partition syntax:
        // Examples:
        //   "14,14"
        //   "nvidia-rtx-4060:14,amd-890m:14"
        //   "gpu0:14,gpu1:14"
        //   "nvidia:16,cpu:12"
        var parts = spec.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        int currentLayer = 0;

        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i];
            string devQuery;
            int count;

            if (part.Contains(':'))
            {
                var segs = part.Split(':', 2, StringSplitOptions.TrimEntries);
                devQuery = segs[0];
                if (segs[1] == "*" || segs[1] == "-1")
                    count = totalLayers - currentLayer;
                else
                    count = int.Parse(segs[1]);
            }
            else
            {
                // Plain number: map sequentially to detected devices
                count = int.Parse(part);
                if (i < gpus.Count)
                    devQuery = gpus[i].Id;
                else
                    devQuery = "cpu";
            }

            var dev = DeviceManager.ResolveDevice(devQuery);
            var eng = dev.Vendor switch
            {
                GpuVendor.Nvidia => InferenceEngineType.BareMetal,
                GpuVendor.Amd or GpuVendor.Intel => InferenceEngineType.DirectML,
                _ => InferenceEngineType.Cpu
            };

            count = Math.Min(count, totalLayers - currentLayer);
            result.Add(new PipelineStageSpec(dev, eng, currentLayer, count));
            currentLayer += count;

            if (currentLayer >= totalLayers)
                break;
        }

        // If any layers remain unassigned, assign to CPU
        if (currentLayer < totalLayers)
        {
            result.Add(new PipelineStageSpec(cpuDevice, InferenceEngineType.Cpu, currentLayer, totalLayers - currentLayer));
        }

        return result;
    }
}
