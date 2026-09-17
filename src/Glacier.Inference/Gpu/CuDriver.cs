namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native NVIDIA CUDA driver (nvcuda.dll).
/// Bypasses cudart64.dll and high-level wrappers to achieve sub-microsecond bare-metal dispatch.
/// </summary>
public static class CuDriver
{
    private const string CudaLib = "nvcuda.dll";

    public const uint CU_MEMHOSTALLOC_PORTABLE = 0x01;
    public const uint CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public const uint CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

    static CuDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    private static readonly Lazy<bool> _isAvailable = new(() =>
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return NativeLibrary.TryLoad("nvcuda.dll", out IntPtr handle) && handle != IntPtr.Zero;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return (NativeLibrary.TryLoad("libcuda.so.1", out IntPtr handle) || NativeLibrary.TryLoad("libcuda.so", out handle)) && handle != IntPtr.Zero;
            return false;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [DllImport(CudaLib, EntryPoint = "cuInit")]
    public static extern int Init(uint flags);

    [DllImport(CudaLib, EntryPoint = "cuDriverGetVersion")]
    public static extern int DriverGetVersion(out int driverVersion);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetCount")]
    public static extern int DeviceGetCount(out int count);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGet")]
    public static extern int DeviceGet(out int device, int ordinal);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetName")]
    public static extern int DeviceGetName(byte[] name, int len, int dev);

    [DllImport(CudaLib, EntryPoint = "cuDeviceGetAttribute")]
    public static extern int DeviceGetAttribute(out int pi, int attrib, int dev);

    [DllImport(CudaLib, EntryPoint = "cuDeviceTotalMem_v2")]
    public static extern int DeviceTotalMem(out nuint bytes, int dev);

    [DllImport(CudaLib, EntryPoint = "cuCtxCreate_v2")]
    public static extern int CtxCreate(out IntPtr pctx, uint flags, int dev);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuCtxSetCurrent")]
    public static extern int CtxSetCurrent(IntPtr ctx);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuCtxGetCurrent")]
    public static extern int CtxGetCurrent(out IntPtr pctx);

    [DllImport(CudaLib, EntryPoint = "cuCtxDestroy_v2")]
    public static extern int CtxDestroy(IntPtr ctx);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuCtxSynchronize")]
    public static extern int CtxSynchronize();

    [DllImport(CudaLib, EntryPoint = "cuModuleLoadData")]
    public static extern int ModuleLoadData(out IntPtr module, byte[] image);

    [DllImport(CudaLib, EntryPoint = "cuModuleGetFunction")]
    public static extern int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemAlloc_v2")]
    public static extern int MemAlloc(out IntPtr dptr, nuint bytesize);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemFree_v2")]
    public static extern int MemFree(IntPtr dptr);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemHostAlloc")]
    public static extern int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemFreeHost")]
    public static extern int MemFreeHost(IntPtr p);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static extern int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemcpyHtoD_v2")]
    public static extern int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoH_v2")]
    public static extern int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoD_v2")]
    public static extern int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static extern int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static extern int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuMemsetD8_v2")]
    public static extern int MemsetD8(IntPtr dstDevice, byte uc, nuint count);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuLaunchKernel")]
    public static extern int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra);

    [DllImport(CudaLib, EntryPoint = "cuStreamCreate")]
    public static extern int StreamCreate(out IntPtr phStream, uint flags);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuStreamSynchronize")]
    public static extern int StreamSynchronize(IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuStreamDestroy_v2")]
    public static extern int StreamDestroy(IntPtr hStream);

    [DllImport(CudaLib, EntryPoint = "cuEventCreate")]
    public static extern int EventCreate(out IntPtr phEvent, uint flags);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuEventRecord")]
    public static extern int EventRecord(IntPtr hEvent, IntPtr hStream);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuEventSynchronize")]
    public static extern int EventSynchronize(IntPtr hEvent);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuEventElapsedTime")]
    public static extern int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

    [DllImport(CudaLib, EntryPoint = "cuEventDestroy_v2")]
    public static extern int EventDestroy(IntPtr hEvent);

    [SuppressGCTransition]
    [DllImport(CudaLib, EntryPoint = "cuStreamWaitEvent")]
    public static extern int StreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

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
            throw new InvalidOperationException($"CUDA Driver Error during '{op}': code {res}");
        }
    }
}
