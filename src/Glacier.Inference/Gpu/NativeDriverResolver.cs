namespace Glacier.Inference.Gpu;

using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Centralized DllImport resolver for bare-metal GPU native drivers (NVIDIA CUDA, AMD HIP, Vulkan).
/// Ensures seamless dynamic library resolution across Windows and Linux.
/// </summary>
public static class NativeDriverResolver
{
    private static int _registered;

    public static void EnsureRegistered()
    {
        if (Interlocked.Exchange(ref _registered, 1) == 0)
        {
            try
            {
                NativeLibrary.SetDllImportResolver(typeof(NativeDriverResolver).Assembly, Resolve);
            }
            catch (InvalidOperationException)
            {
                // Ignore if already set
            }
        }
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "nvcuda.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("nvcuda.dll", assembly, searchPath, out IntPtr handle))
                    return handle;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcuda.so.1", assembly, searchPath, out IntPtr handle) ||
                    NativeLibrary.TryLoad("libcuda.so", assembly, searchPath, out handle))
                    return handle;
            }
        }
        else if (libraryName == "amdhip64.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("amdhip64.dll", assembly, searchPath, out IntPtr handle) ||
                    NativeLibrary.TryLoad("amdhip64_6.dll", assembly, searchPath, out handle) ||
                    NativeLibrary.TryLoad("amdhip64_7.dll", assembly, searchPath, out handle))
                    return handle;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libamdhip64.so", assembly, searchPath, out IntPtr handle) ||
                    NativeLibrary.TryLoad("libamdhip64.so.6", assembly, searchPath, out handle) ||
                    NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", assembly, searchPath, out handle))
                    return handle;
            }
        }
        else if (libraryName == "vulkan-1.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("vulkan-1.dll", assembly, searchPath, out IntPtr handle))
                    return handle;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libvulkan.so.1", assembly, searchPath, out IntPtr handle) ||
                    NativeLibrary.TryLoad("libvulkan.so", assembly, searchPath, out handle))
                    return handle;
            }
        }
        return IntPtr.Zero;
    }
}
