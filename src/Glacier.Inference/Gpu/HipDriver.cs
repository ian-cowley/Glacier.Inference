namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native AMD ROCm / HIP driver (amdhip64.dll on Windows, libamdhip64.so on Linux).
/// Bypasses high-level runtimes to achieve sub-microsecond bare-metal kernel dispatch directly to AMD RDNA/CDNA GPUs.
/// </summary>
public static partial class HipDriver
{
    private const string HipLib = "amdhip64.dll";

    public const uint HIP_MEMHOSTALLOC_PORTABLE = 0x01;
    public const uint HIP_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public const uint HIP_MEMHOSTALLOC_WRITECOMBINED = 0x04;

    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    public const int HIP_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
    public const int HIP_DEVICE_ATTRIBUTE_WAVEFRONT_SIZE = 84;

    static HipDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("amdhip64.dll", out IntPtr handle) && handle != IntPtr.Zero)
                    return true;
                if (NativeLibrary.TryLoad("amdhip64_6.dll", out handle) && handle != IntPtr.Zero)
                    return true;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if ((NativeLibrary.TryLoad("libamdhip64.so", out IntPtr handle) ||
                     NativeLibrary.TryLoad("libamdhip64.so.6", out handle) ||
                     NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", out handle)) && handle != IntPtr.Zero)
                {
                    return true;
                }
            }
            return false;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [LibraryImport(HipLib, EntryPoint = "hipInit")]
    public static partial int Init(uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipDriverGetVersion")]
    public static partial int DriverGetVersion(out int driverVersion);

    [LibraryImport(HipLib, EntryPoint = "hipGetDeviceCount")]
    public static partial int DeviceGetCount(out int count);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceGet")]
    public static partial int DeviceGet(out int device, int ordinal);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceGetName")]
    public static partial int DeviceGetName([Out] byte[] name, int len, int dev);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceGetAttribute")]
    public static partial int DeviceGetAttribute(out int pi, int attrib, int dev);

    [LibraryImport(HipLib, EntryPoint = "hipDeviceTotalMem")]
    public static partial int DeviceTotalMem(out nuint bytes, int dev);

    [LibraryImport(HipLib, EntryPoint = "hipMemGetInfo")]
    public static partial int MemGetInfo(out nuint free, out nuint total);

    [LibraryImport(HipLib, EntryPoint = "hipCtxCreate")]
    public static partial int CtxCreate(out IntPtr pctx, uint flags, int dev);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipCtxSetCurrent")]
    public static partial int CtxSetCurrent(IntPtr ctx);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipCtxGetCurrent")]
    public static partial int CtxGetCurrent(out IntPtr pctx);

    [LibraryImport(HipLib, EntryPoint = "hipCtxDestroy")]
    public static partial int CtxDestroy(IntPtr ctx);

    [LibraryImport(HipLib, EntryPoint = "hipCtxSynchronize")]
    public static partial int CtxSynchronize();

    [LibraryImport(HipLib, EntryPoint = "hipModuleLoadData")]
    public static partial int ModuleLoadData(out IntPtr module, byte[] image);

    [LibraryImport(HipLib, EntryPoint = "hipModuleUnload")]
    public static partial int ModuleUnload(IntPtr hmod);

    [LibraryImport(HipLib, EntryPoint = "hipModuleGetFunction", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

    [LibraryImport(HipLib, EntryPoint = "hipMalloc")]
    public static partial int MemAlloc(out IntPtr dptr, nuint bytesize);

    [LibraryImport(HipLib, EntryPoint = "hipFree")]
    public static partial int MemFree(IntPtr dptr);

    [LibraryImport(HipLib, EntryPoint = "hipHostMalloc")]
    public static partial int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipHostFree")]
    public static partial int MemFreeHost(IntPtr p);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipHostGetDevicePointer")]
    public static partial int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags);

    [LibraryImport(HipLib, EntryPoint = "hipMemcpyHtoD")]
    public static partial int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount);

    [LibraryImport(HipLib, EntryPoint = "hipMemcpyDtoH")]
    public static partial int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount);

    [LibraryImport(HipLib, EntryPoint = "hipMemcpyDtoD")]
    public static partial int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipMemcpyDtoDAsync")]
    public static partial int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipMemcpyHtoDAsync")]
    public static partial int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipMemsetD8")]
    public static partial int MemsetD8(IntPtr dstDevice, byte uc, nuint count);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipModuleLaunchKernel")]
    public static partial int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra);

    [LibraryImport(HipLib, EntryPoint = "hipStreamCreate")]
    public static partial int StreamCreate(out IntPtr phStream);

    [LibraryImport(HipLib, EntryPoint = "hipStreamSynchronize")]
    public static partial int StreamSynchronize(IntPtr hStream);

    [LibraryImport(HipLib, EntryPoint = "hipStreamDestroy")]
    public static partial int StreamDestroy(IntPtr hStream);

    [LibraryImport(HipLib, EntryPoint = "hipEventCreate")]
    public static partial int EventCreate(out IntPtr phEvent);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipEventRecord")]
    public static partial int EventRecord(IntPtr hEvent, IntPtr hStream);

    [LibraryImport(HipLib, EntryPoint = "hipEventSynchronize")]
    public static partial int EventSynchronize(IntPtr hEvent);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipEventElapsedTime")]
    public static partial int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

    [LibraryImport(HipLib, EntryPoint = "hipEventDestroy")]
    public static partial int EventDestroy(IntPtr hEvent);

    [SuppressGCTransition]
    [LibraryImport(HipLib, EntryPoint = "hipStreamWaitEvent")]
    public static partial int StreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

    public static string GetDeviceName(int device)
    {
        var buf = new byte[256];
        DeviceGetName(buf, buf.Length, device);
        return Encoding.ASCII.GetString(buf).TrimEnd('\0');
    }

    public static unsafe string? TryGetGcnArchName(int device)
    {
        try
        {
            IntPtr lib = IntPtr.Zero;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (!NativeLibrary.TryLoad("amdhip64.dll", out lib) || lib == IntPtr.Zero)
                    NativeLibrary.TryLoad("amdhip64_6.dll", out lib);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (!NativeLibrary.TryLoad("libamdhip64.so", out lib) || lib == IntPtr.Zero)
                {
                    if (!NativeLibrary.TryLoad("libamdhip64.so.6", out lib) || lib == IntPtr.Zero)
                        NativeLibrary.TryLoad("/opt/rocm/lib/libamdhip64.so", out lib);
                }
            }

            if (lib == IntPtr.Zero)
                return null;

            if (!NativeLibrary.TryGetExport(lib, "hipGetDevicePropertiesR0600", out IntPtr fnPtr) &&
                !NativeLibrary.TryGetExport(lib, "hipGetDeviceProperties", out fnPtr))
            {
                return null;
            }

            const nuint propBufSize = 4096;
            byte* pProp = (byte*)NativeMemory.AlignedAlloc(propBufSize, 64);
            try
            {
                NativeMemory.Clear(pProp, propBufSize);
                var getProps = (delegate* unmanaged[Cdecl]<byte*, int, int>)fnPtr;
                if (getProps(pProp, device) != 0)
                    return null;

                // Skip char name[256] at offset 0 and scan for gcnArchName ("gfx...")
                for (int i = 256; i < (int)propBufSize - 6; i++)
                {
                    if (pProp[i] == (byte)'g' &&
                        pProp[i + 1] == (byte)'f' &&
                        pProp[i + 2] == (byte)'x' &&
                        Uri.IsHexDigit((char)pProp[i + 3]))
                    {
                        int end = i + 3;
                        while (end < (int)propBufSize && pProp[end] != 0 && pProp[end] != (byte)':')
                        {
                            char ch = (char)pProp[end];
                            if (!char.IsLetterOrDigit(ch) && ch != '_')
                                break;
                            end++;
                        }
                        if (end > i + 3 && (end == (int)propBufSize || pProp[end] == 0 || pProp[end] == (byte)':'))
                        {
                            return Encoding.ASCII.GetString(pProp + i, end - i);
                        }
                    }
                }
            }
            finally
            {
                NativeMemory.AlignedFree(pProp);
            }
        }
        catch
        {
            // Fall back to major/minor formatting if property struct probe is unavailable
        }
        return null;
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"AMD HIP Driver Error during '{op}': code {res}");
        }
    }
}
