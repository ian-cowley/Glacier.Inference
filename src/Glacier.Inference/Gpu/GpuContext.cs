namespace Glacier.Inference.Gpu;

using System;
using System.IO;

/// <summary>
/// Manages the lifetime of a native CUDA driver context on the target NVIDIA GPU.
/// </summary>
public sealed class GpuContext : IDisposable
{
    private IntPtr _ctx;
    private bool _disposed;

    public string DeviceName { get; }
    public ulong TotalVramBytes { get; }
    public int DeviceOrdinal { get; }
    public IntPtr Handle => _ctx;

    public static bool IsSupported
    {
        get
        {
            try
            {
                int init = CuDriver.Init(0);
                if (init != 0) return false;
                int countRes = CuDriver.DeviceGetCount(out int count);
                return countRes == 0 && count > 0;
            }
            catch
            {
                return false;
            }
        }
    }

    public GpuContext(int deviceOrdinal = 0)
    {
        CuDriver.Check(CuDriver.Init(0), "cuInit");
        CuDriver.Check(CuDriver.DeviceGetCount(out int count), "cuDeviceGetCount");
        if (count == 0)
            throw new InvalidOperationException("No CUDA-capable GPU detected on this system.");

        DeviceOrdinal = Math.Clamp(deviceOrdinal, 0, count - 1);
        CuDriver.Check(CuDriver.DeviceGet(out int dev, DeviceOrdinal), "cuDeviceGet");
        DeviceName = CuDriver.GetDeviceName(dev);

        CuDriver.Check(CuDriver.DeviceTotalMem(out nuint totalBytes, dev), "cuDeviceTotalMem");
        TotalVramBytes = (ulong)totalBytes;

        CuDriver.Check(CuDriver.CtxCreate(out _ctx, 0, dev), "cuCtxCreate");
        CuDriver.CtxSetCurrent(_ctx);
    }

    public IntPtr AllocateDevice(nuint bytes)
    {
        CuDriver.CtxSetCurrent(_ctx);
        CuDriver.Check(CuDriver.MemAlloc(out IntPtr ptr, bytes), $"cuMemAlloc({bytes} bytes)");
        return ptr;
    }

    public void FreeDevice(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
        {
            CuDriver.CtxSetCurrent(_ctx);
            CuDriver.MemFree(ptr);
        }
    }

    public void CopyToDevice(IntPtr dstDevice, IntPtr srcHost, nuint bytes)
    {
        CuDriver.CtxSetCurrent(_ctx);
        CuDriver.Check(CuDriver.MemcpyHtoD(dstDevice, srcHost, bytes), "cuMemcpyHtoD");
    }

    public void CopyToHost(IntPtr dstHost, IntPtr srcDevice, nuint bytes)
    {
        CuDriver.CtxSetCurrent(_ctx);
        CuDriver.Check(CuDriver.MemcpyDtoH(dstHost, srcDevice, bytes), "cuMemcpyDtoH");
    }

    public void Synchronize()
    {
        CuDriver.CtxSetCurrent(_ctx);
        CuDriver.Check(CuDriver.CtxSynchronize(), "cuCtxSynchronize");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_ctx != IntPtr.Zero)
            {
                CuDriver.CtxDestroy(_ctx);
                _ctx = IntPtr.Zero;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~GpuContext()
    {
        Dispose();
    }
}
