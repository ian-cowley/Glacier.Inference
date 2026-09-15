namespace Glacier.Inference.Gpu;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Manages the lifetime of a native AMD ROCm / HIP driver context on the target AMD Radeon / Instinct GPU.
/// Uses zero external dependencies, communicating directly with amdhip64.dll on Windows and libamdhip64.so on Linux.
/// </summary>
public sealed class HipContext : IDisposable
{
    private IntPtr _ctx;
    private bool _disposed;

    public string DeviceName { get; }
    public ulong TotalVramBytes { get; }
    public int DeviceOrdinal { get; }
    public (int Major, int Minor) ComputeCapability { get; }
    public string ArchString { get; }
    public IntPtr Handle => _ctx;

    public static bool IsSupported
    {
        get
        {
            try
            {
                if (!HipDriver.IsAvailable()) return false;
                int init = HipDriver.Init(0);
                if (init != 0) return false;
                int countRes = HipDriver.DeviceGetCount(out int count);
                return countRes == 0 && count > 0;
            }
            catch
            {
                return false;
            }
        }
    }

    public HipContext(int deviceOrdinal = 0)
    {
        int init = HipDriver.Init(0);
        if (init != 0)
            throw new InvalidOperationException($"hipInit failed with code {init}");

        int countRes = HipDriver.DeviceGetCount(out int count);
        if (countRes != 0 || count == 0)
            throw new InvalidOperationException("No HIP-capable AMD GPU detected on this system.");

        DeviceOrdinal = Math.Clamp(deviceOrdinal, 0, count - 1);
        int devRes = HipDriver.DeviceGet(out int dev, DeviceOrdinal);
        if (devRes != 0)
            throw new InvalidOperationException($"hipDeviceGet failed for ordinal {DeviceOrdinal}: code {devRes}");

        byte[] nameBuf = new byte[256];
        if (HipDriver.DeviceGetName(nameBuf, nameBuf.Length, dev) == 0)
        {
            DeviceName = Encoding.UTF8.GetString(nameBuf).TrimEnd('\0').Trim();
        }
        else
        {
            DeviceName = $"AMD GPU #{DeviceOrdinal}";
        }

        int major = 11, minor = 5; // Default gfx1150
        if (HipDriver.DeviceGetAttribute(out int maj, HipDriver.HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == 0 &&
            HipDriver.DeviceGetAttribute(out int min, HipDriver.HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == 0)
        {
            major = maj;
            minor = min;
        }
        ComputeCapability = (major, minor);
        ArchString = $"gfx{major}{minor}0";

        if (HipDriver.DeviceTotalMem(out nuint totalBytes, dev) == 0)
        {
            TotalVramBytes = (ulong)totalBytes;
        }
        else
        {
            TotalVramBytes = 0;
        }

        int ctxRes = HipDriver.CtxCreate(out _ctx, 0, dev);
        if (ctxRes != 0 || _ctx == IntPtr.Zero)
            throw new InvalidOperationException($"hipCtxCreate failed: code {ctxRes}");

        HipDriver.CtxSetCurrent(_ctx);
    }

    public IntPtr AllocateDevice(nuint bytes)
    {
        HipDriver.CtxSetCurrent(_ctx);
        int res = HipDriver.MemAlloc(out IntPtr ptr, bytes);
        if (res != 0)
            throw new InvalidOperationException($"hipMalloc failed for {bytes} bytes: code {res}");
        return ptr;
    }

    public void FreeDevice(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
        {
            HipDriver.CtxSetCurrent(_ctx);
            HipDriver.MemFree(ptr);
        }
    }

    public void CopyToDevice(IntPtr dstDevice, IntPtr srcHost, nuint bytes)
    {
        HipDriver.CtxSetCurrent(_ctx);
        int res = HipDriver.MemcpyHtoD(dstDevice, srcHost, bytes);
        if (res != 0)
            throw new InvalidOperationException($"hipMemcpyHtoD failed: code {res}");
    }

    public void CopyToHost(IntPtr dstHost, IntPtr srcDevice, nuint bytes)
    {
        HipDriver.CtxSetCurrent(_ctx);
        int res = HipDriver.MemcpyDtoH(dstHost, srcDevice, bytes);
        if (res != 0)
            throw new InvalidOperationException($"hipMemcpyDtoH failed: code {res}");
    }

    public void Synchronize()
    {
        HipDriver.CtxSetCurrent(_ctx);
        int res = HipDriver.CtxSynchronize();
        if (res != 0)
            throw new InvalidOperationException($"hipCtxSynchronize failed: code {res}");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_ctx != IntPtr.Zero)
            {
                HipDriver.CtxDestroy(_ctx);
                _ctx = IntPtr.Zero;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~HipContext()
    {
        Dispose();
    }
}
