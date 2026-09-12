namespace Glacier.Inference.Gpu;

using System;
using System.IO;
using System.Reflection;

/// <summary>
/// Manages loading and on-demand JIT compilation of native GPU CUBIN binaries.
/// </summary>
public static class KernelCompiler
{
    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Glacier", "Inference", "Kernels");

    public static byte[] GetOrCompileKernels(string targetArch = "auto")
    {
        Directory.CreateDirectory(CacheDir);
        string cachedPath = Path.Combine(CacheDir, $"kernels_{targetArch}.cubin");

        if (File.Exists(cachedPath))
        {
            return File.ReadAllBytes(cachedPath);
        }

        // 1. Try to load embedded universal fatbinary resource (supports sm_75, sm_80, sm_86, sm_89, sm_90, and PTX compute_75)
        var asm = Assembly.GetExecutingAssembly();
        using var stream = asm.GetManifestResourceStream("Glacier.Inference.Gpu.Kernels.kernels.cubin");
        if (stream != null)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            byte[] bytes = ms.ToArray();
            File.WriteAllBytes(cachedPath, bytes);
            return bytes;
        }

        // 2. Check for local file next to source if running from development
        string localSourceCubin = Path.Combine(AppContext.BaseDirectory, "Gpu", "Kernels", "kernels.cubin");
        if (File.Exists(localSourceCubin))
        {
            byte[] bytes = File.ReadAllBytes(localSourceCubin);
            File.WriteAllBytes(cachedPath, bytes);
            return bytes;
        }

        throw new FileNotFoundException(
            $"Compiled CUDA kernels for architecture {targetArch} not found. Ensure kernels.cubin is present.");
    }
}
