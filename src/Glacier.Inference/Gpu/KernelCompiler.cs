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

    public static byte[] GetOrCompileKernels(string targetArch = "sm_89")
    {
        Directory.CreateDirectory(CacheDir);
        string cachedPath = Path.Combine(CacheDir, $"kernels_{targetArch}.cubin");

        if (File.Exists(cachedPath))
        {
            return File.ReadAllBytes(cachedPath);
        }

        // Try to load embedded resource if matches sm_89
        var asm = Assembly.GetExecutingAssembly();
        using var stream = asm.GetManifestResourceStream("Glacier.Inference.Gpu.Kernels.kernels.cubin");
        if (stream != null && targetArch == "sm_89")
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            byte[] bytes = ms.ToArray();
            File.WriteAllBytes(cachedPath, bytes);
            return bytes;
        }

        // Check for local file next to source if running from development
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
