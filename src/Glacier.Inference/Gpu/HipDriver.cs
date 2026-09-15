namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native AMD ROCm / HIP driver (amdhip64.dll on Windows, libamdhip64.so on Linux).
/// Bypasses high-level runtimes to achieve sub-microsecond bare-metal kernel dispatch directly to AMD RDNA/CDNA GPUs.
/// </summary>
public static class HipDriver
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

    [DllImport(HipLib, EntryPoint = "hipInit")]
    public static extern int Init(uint flags);

    [DllImport(HipLib, EntryPoint = "hipDriverGetVersion")]
    public static extern int DriverGetVersion(out int driverVersion);

    [DllImport(HipLib, EntryPoint = "hipGetDeviceCount")]
    public static extern int DeviceGetCount(out int count);

    [DllImport(HipLib, EntryPoint = "hipDeviceGet")]
    public static extern int DeviceGet(out int device, int ordinal);

    [DllImport(HipLib, EntryPoint = "hipDeviceGetName")]
    public static extern int DeviceGetName(byte[] name, int len, int dev);

    [DllImport(HipLib, EntryPoint = "hipDeviceGetAttribute")]
    public static extern int DeviceGetAttribute(out int pi, int attrib, int dev);

    [DllImport(HipLib, EntryPoint = "hipDeviceTotalMem")]
    public static extern int DeviceTotalMem(out nuint bytes, int dev);

    [DllImport(HipLib, EntryPoint = "hipCtxCreate")]
    public static extern int CtxCreate(out IntPtr pctx, uint flags, int dev);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipCtxSetCurrent")]
    public static extern int CtxSetCurrent(IntPtr ctx);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipCtxGetCurrent")]
    public static extern int CtxGetCurrent(out IntPtr pctx);

    [DllImport(HipLib, EntryPoint = "hipCtxDestroy")]
    public static extern int CtxDestroy(IntPtr ctx);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipCtxSynchronize")]
    public static extern int CtxSynchronize();

    [DllImport(HipLib, EntryPoint = "hipModuleLoadData")]
    public static extern int ModuleLoadData(out IntPtr module, byte[] image);

    [DllImport(HipLib, EntryPoint = "hipModuleGetFunction")]
    public static extern int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipMalloc")]
    public static extern int MemAlloc(out IntPtr dptr, nuint bytesize);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipFree")]
    public static extern int MemFree(IntPtr dptr);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipHostMalloc")]
    public static extern int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipHostFree")]
    public static extern int MemFreeHost(IntPtr p);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipHostGetDevicePointer")]
    public static extern int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipMemcpyHtoD")]
    public static extern int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipMemcpyDtoH")]
    public static extern int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipMemcpyDtoD")]
    public static extern int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipMemsetD8")]
    public static extern int MemsetD8(IntPtr dstDevice, byte uc, nuint count);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipModuleLaunchKernel")]
    public static extern int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra);

    [DllImport(HipLib, EntryPoint = "hipStreamCreate")]
    public static extern int StreamCreate(out IntPtr phStream);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipStreamSynchronize")]
    public static extern int StreamSynchronize(IntPtr hStream);

    [DllImport(HipLib, EntryPoint = "hipStreamDestroy")]
    public static extern int StreamDestroy(IntPtr hStream);

    [DllImport(HipLib, EntryPoint = "hipEventCreate")]
    public static extern int EventCreate(out IntPtr phEvent);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipEventRecord")]
    public static extern int EventRecord(IntPtr hEvent, IntPtr hStream);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipEventSynchronize")]
    public static extern int EventSynchronize(IntPtr hEvent);

    [SuppressGCTransition]
    [DllImport(HipLib, EntryPoint = "hipEventElapsedTime")]
    public static extern int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

    [DllImport(HipLib, EntryPoint = "hipEventDestroy")]
    public static extern int EventDestroy(IntPtr hEvent);

    public static string GetDeviceName(int device)
    {
        var buf = new byte[256];
        DeviceGetName(buf, buf.Length, device);
        return Encoding.ASCII.GetString(buf).TrimEnd('\0');
    }

    public static void Check(int res, string op)
    {
        if (res != 0)
        {
            throw new InvalidOperationException($"AMD HIP Driver Error during '{op}': code {res}");
        }
    }
}
