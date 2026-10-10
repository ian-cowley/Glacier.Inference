namespace Glacier.Inference.Gpu;

using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Centralized DllImport resolver for bare-metal GPU native drivers (NVIDIA CUDA, AMD HIP, Vulkan, OpenCL).
/// Ensures secure, seamless dynamic library resolution across Windows, Linux, and Android.
/// Restricts Windows library searches to System32 to prevent DLL hijacking / planting.
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
                // Ignore if already set by host or test runner
            }
        }
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        // On Windows, enforce System32 search path to prevent DLL hijacking from current working directory
        DllImportSearchPath securePath = searchPath ?? (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? DllImportSearchPath.System32
            : DllImportSearchPath.ApplicationDirectory);

        if (libraryName == "nvcuda.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("nvcuda.dll", assembly, securePath, out IntPtr handle))
                    return handle;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcuda.so.1", out IntPtr handle) ||
                    NativeLibrary.TryLoad("libcuda.so", out handle))
                    return handle;
            }
        }
        else if (libraryName == "amdhip64.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("amdhip64.dll", assembly, securePath, out IntPtr handle) ||
                    NativeLibrary.TryLoad("amdhip64_6.dll", assembly, securePath, out handle) ||
                    NativeLibrary.TryLoad("amdhip64_7.dll", assembly, securePath, out handle))
                    return handle;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libamdhip64.so", out IntPtr handle) ||
                    NativeLibrary.TryLoad("libamdhip64.so.6", out handle) ||
                    NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", out handle))
                    return handle;
            }
        }
        else if (libraryName == "vulkan-1.dll" || libraryName == "libvulkan.so")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("vulkan-1.dll", assembly, securePath, out IntPtr handle))
                    return handle;
            }
            else
            {
                if (NativeLibrary.TryLoad("libvulkan.so", out IntPtr handle) ||
                    NativeLibrary.TryLoad("libvulkan.so.1", out handle))
                    return handle;
            }
        }
        else if (libraryName == "libOpenCL.so" || libraryName == "OpenCL.dll")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("OpenCL.dll", assembly, securePath, out IntPtr handle))
                    return handle;
            }
            else
            {
                if (NativeLibrary.TryLoad("libOpenCL.so", out IntPtr handle) ||
                    NativeLibrary.TryLoad("libOpenCL.so.1", out handle) ||
                    NativeLibrary.TryLoad("/vendor/lib64/libOpenCL.so", out handle) ||
                    NativeLibrary.TryLoad("/system/vendor/lib64/libOpenCL.so", assembly, searchPath, out handle))
                    return handle;
            }
        }
        return IntPtr.Zero;
    }
}
