namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the native NVIDIA CUDA driver (nvcuda.dll).
/// Bypasses cudart64.dll and high-level wrappers to achieve sub-microsecond bare-metal dispatch.
/// </summary>
public static partial class CuDriver
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
            {
                if (!NativeLibrary.TryLoad("nvcuda.dll", out IntPtr handle) || handle == IntPtr.Zero)
                    return false;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (!NativeLibrary.TryLoad("libcuda.so.1", out IntPtr handle) && !NativeLibrary.TryLoad("libcuda.so", out handle))
                    return false;
                if (handle == IntPtr.Zero) return false;
            }
            else
            {
                return false;
            }

            return Init(0) == 0;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable() => _isAvailable.Value;

    [LibraryImport(CudaLib, EntryPoint = "cuInit")]
    public static partial int Init(uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuDriverGetVersion")]
    public static partial int DriverGetVersion(out int driverVersion);

    [LibraryImport(CudaLib, EntryPoint = "cuDeviceGetCount")]
    public static partial int DeviceGetCount(out int count);

    [LibraryImport(CudaLib, EntryPoint = "cuDeviceGet")]
    public static partial int DeviceGet(out int device, int ordinal);

    [LibraryImport(CudaLib, EntryPoint = "cuDeviceGetName")]
    public static partial int DeviceGetName([Out] byte[] name, int len, int dev);

    [LibraryImport(CudaLib, EntryPoint = "cuDeviceGetAttribute")]
    public static partial int DeviceGetAttribute(out int pi, int attrib, int dev);

    [LibraryImport(CudaLib, EntryPoint = "cuDeviceTotalMem_v2")]
    public static partial int DeviceTotalMem(out nuint bytes, int dev);

    [LibraryImport(CudaLib, EntryPoint = "cuMemGetInfo_v2")]
    public static partial int MemGetInfo(out nuint free, out nuint total);

    [LibraryImport(CudaLib, EntryPoint = "cuCtxCreate_v2")]
    public static partial int CtxCreate(out IntPtr pctx, uint flags, int dev);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuCtxSetCurrent")]
    public static partial int CtxSetCurrent(IntPtr ctx);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuCtxGetCurrent")]
    public static partial int CtxGetCurrent(out IntPtr pctx);

    [LibraryImport(CudaLib, EntryPoint = "cuCtxDestroy_v2")]
    public static partial int CtxDestroy(IntPtr ctx);

    [LibraryImport(CudaLib, EntryPoint = "cuCtxSynchronize")]
    public static partial int CtxSynchronize();

    [LibraryImport(CudaLib, EntryPoint = "cuModuleLoadData")]
    public static partial int ModuleLoadData(out IntPtr module, byte[] image);

    [LibraryImport(CudaLib, EntryPoint = "cuModuleUnload")]
    public static partial int ModuleUnload(IntPtr hmod);

    [LibraryImport(CudaLib, EntryPoint = "cuModuleGetFunction", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int ModuleGetFunction(out IntPtr hfunc, IntPtr hmod, string name);

    [LibraryImport(CudaLib, EntryPoint = "cuModuleGetGlobal_v2", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int ModuleGetGlobal(out IntPtr dptr, out nuint bytes, IntPtr hmod, string name);

    [LibraryImport(CudaLib, EntryPoint = "cuMemAlloc_v2")]
    public static partial int MemAlloc(out IntPtr dptr, nuint bytesize);

    [LibraryImport(CudaLib, EntryPoint = "cuMemFree_v2")]
    public static partial int MemFree(IntPtr dptr);

    [LibraryImport(CudaLib, EntryPoint = "cuMemHostAlloc")]
    public static partial int MemHostAlloc(out IntPtr pp, nuint bytesize, uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuMemHostRegister_v2")]
    public static partial int MemHostRegister(IntPtr p, nuint bytesize, uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuMemHostUnregister")]
    public static partial int MemHostUnregister(IntPtr p);

    [LibraryImport(CudaLib, EntryPoint = "cuMemFreeHost")]
    public static partial int MemFreeHost(IntPtr p);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuMemHostGetDevicePointer_v2")]
    public static partial int MemHostGetDevicePointer(out IntPtr pdptr, IntPtr p, uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuMemcpyHtoD_v2")]
    public static partial int MemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, nuint byteCount);

    [LibraryImport(CudaLib, EntryPoint = "cuMemcpyDtoH_v2")]
    public static partial int MemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, nuint byteCount);

    [LibraryImport(CudaLib, EntryPoint = "cuMemcpyDtoD_v2")]
    public static partial int MemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static partial int MemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static partial int MemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, nuint byteCount, IntPtr hStream);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuMemsetD8_v2")]
    public static partial int MemsetD8(IntPtr dstDevice, byte uc, nuint count);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuLaunchKernel")]
    public static partial int LaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr hStream,
        IntPtr kernelParams, IntPtr extra);

    [LibraryImport(CudaLib, EntryPoint = "cuStreamCreate")]
    public static partial int StreamCreate(out IntPtr phStream, uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuStreamSynchronize")]
    public static partial int StreamSynchronize(IntPtr hStream);

    [LibraryImport(CudaLib, EntryPoint = "cuStreamDestroy_v2")]
    public static partial int StreamDestroy(IntPtr hStream);

    [LibraryImport(CudaLib, EntryPoint = "cuEventCreate")]
    public static partial int EventCreate(out IntPtr phEvent, uint flags);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuEventRecord")]
    public static partial int EventRecord(IntPtr hEvent, IntPtr hStream);

    [LibraryImport(CudaLib, EntryPoint = "cuEventSynchronize")]
    public static partial int EventSynchronize(IntPtr hEvent);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuEventElapsedTime")]
    public static partial int EventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

    [LibraryImport(CudaLib, EntryPoint = "cuEventDestroy_v2")]
    public static partial int EventDestroy(IntPtr hEvent);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuStreamWaitEvent")]
    public static partial int StreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

    [LibraryImport(CudaLib, EntryPoint = "cuStreamBeginCapture_v2")]
    public static partial int StreamBeginCapture(IntPtr hStream, uint mode);

    [LibraryImport(CudaLib, EntryPoint = "cuStreamEndCapture")]
    public static partial int StreamEndCapture(IntPtr hStream, out IntPtr phGraph);

    [LibraryImport(CudaLib, EntryPoint = "cuGraphInstantiateWithFlags")]
    public static partial int GraphInstantiateWithFlags(out IntPtr phGraphExec, IntPtr hGraph, ulong flags);

    [SuppressGCTransition]
    [LibraryImport(CudaLib, EntryPoint = "cuGraphLaunch")]
    public static partial int GraphLaunch(IntPtr hGraphExec, IntPtr hStream);

    [LibraryImport(CudaLib, EntryPoint = "cuGraphExecDestroy")]
    public static partial int GraphExecDestroy(IntPtr hGraphExec);

    [LibraryImport(CudaLib, EntryPoint = "cuGraphDestroy")]
    public static partial int GraphDestroy(IntPtr hGraph);

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
