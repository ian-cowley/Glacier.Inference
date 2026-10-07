namespace Glacier.Inference.Gpu.OpenCL;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// High-level OpenCL context for mobile and desktop GPU acceleration (Qualcomm Adreno, ARM Mali, Intel, AMD, NVIDIA).
/// </summary>
public sealed unsafe class OpenCLContext : IDisposable
{
    private IntPtr _platform;
    private IntPtr _device;
    private IntPtr _context;
    private IntPtr _queue;
    private bool _disposed;

    public string PlatformName { get; }
    public string DeviceName { get; }
    public string DriverVersion { get; }
    public uint MaxComputeUnits { get; }
    public ulong GlobalMemSizeBytes { get; }
    public IntPtr DeviceHandle => _device;
    public IntPtr ContextHandle => _context;
    public IntPtr QueueHandle => _queue;

    public static bool IsSupported => OpenCLDriver.IsAvailable();

    public OpenCLContext()
    {
        uint numPlatforms = 0;
        int err = OpenCLDriver.clGetPlatformIDs(0, null, &numPlatforms);
        if (err != OpenCLDriver.CL_SUCCESS || numPlatforms == 0)
            throw new PlatformNotSupportedException($"OpenCL platform discovery failed: error {err}");

        var platforms = stackalloc IntPtr[(int)numPlatforms];
        OpenCLDriver.clGetPlatformIDs(numPlatforms, platforms, null);
        _platform = platforms[0];

        // Read platform name
        byte[] pNameBuf = new byte[256];
        fixed (byte* pBuf = pNameBuf)
        {
            OpenCLDriver.clGetPlatformInfo(_platform, OpenCLDriver.CL_PLATFORM_NAME, (nuint)pNameBuf.Length, pBuf, null);
            PlatformName = Encoding.UTF8.GetString(pNameBuf).TrimEnd('\0');
        }

        // Query GPU device
        uint numDevices = 0;
        err = OpenCLDriver.clGetDeviceIDs(_platform, OpenCLDriver.CL_DEVICE_TYPE_GPU, 0, null, &numDevices);
        if (err != OpenCLDriver.CL_SUCCESS || numDevices == 0)
        {
            // Fallback to any device
            err = OpenCLDriver.clGetDeviceIDs(_platform, 0xFFFFFFFF, 0, null, &numDevices);
            if (err != OpenCLDriver.CL_SUCCESS || numDevices == 0)
                throw new PlatformNotSupportedException("No OpenCL devices found.");
        }

        var devices = stackalloc IntPtr[(int)numDevices];
        OpenCLDriver.clGetDeviceIDs(_platform, OpenCLDriver.CL_DEVICE_TYPE_GPU, numDevices, devices, null);
        _device = devices[0];

        // Device Name
        byte[] dNameBuf = new byte[256];
        fixed (byte* pBuf = dNameBuf)
        {
            OpenCLDriver.clGetDeviceInfo(_device, OpenCLDriver.CL_DEVICE_NAME, (nuint)dNameBuf.Length, pBuf, null);
            DeviceName = Encoding.UTF8.GetString(dNameBuf).TrimEnd('\0');
        }

        // Driver Version
        byte[] drvBuf = new byte[256];
        fixed (byte* pBuf = drvBuf)
        {
            OpenCLDriver.clGetDeviceInfo(_device, OpenCLDriver.CL_DRIVER_VERSION, (nuint)drvBuf.Length, pBuf, null);
            DriverVersion = Encoding.UTF8.GetString(drvBuf).TrimEnd('\0');
        }

        // Compute Units
        uint cu = 0;
        OpenCLDriver.clGetDeviceInfo(_device, OpenCLDriver.CL_DEVICE_MAX_COMPUTE_UNITS, (nuint)sizeof(uint), &cu, null);
        MaxComputeUnits = cu;

        // Global Mem
        ulong mem = 0;
        OpenCLDriver.clGetDeviceInfo(_device, OpenCLDriver.CL_DEVICE_GLOBAL_MEM_SIZE, (nuint)sizeof(ulong), &mem, null);
        GlobalMemSizeBytes = mem;

        // Create Context
        int ctxErr = 0;
        IntPtr dev = _device;
        _context = OpenCLDriver.clCreateContext(null, 1, &dev, null, null, &ctxErr);
        if (ctxErr != OpenCLDriver.CL_SUCCESS || _context == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create OpenCL context: error {ctxErr}");

        // Create Command Queue
        int qErr = 0;
        _queue = OpenCLDriver.clCreateCommandQueue(_context, _device, 0, &qErr);
        if (qErr != OpenCLDriver.CL_SUCCESS || _queue == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to create OpenCL command queue: error {qErr}");
    }

    public IntPtr CreateBuffer(nuint byteSize, ulong flags = OpenCLDriver.CL_MEM_READ_WRITE)
    {
        int err = 0;
        IntPtr buf = OpenCLDriver.clCreateBuffer(_context, flags, byteSize, null, &err);
        if (err != OpenCLDriver.CL_SUCCESS || buf == IntPtr.Zero)
            throw new InvalidOperationException($"Failed to allocate {byteSize} bytes on OpenCL GPU: error {err}");
        return buf;
    }

    public void WriteBuffer(IntPtr buffer, ReadOnlySpan<byte> hostBytes)
    {
        fixed (byte* p = hostBytes)
        {
            int err = OpenCLDriver.clEnqueueWriteBuffer(_queue, buffer, OpenCLDriver.CL_TRUE, 0, (nuint)hostBytes.Length, p, 0, null, null);
            if (err != OpenCLDriver.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to write to OpenCL buffer: error {err}");
        }
    }

    public void ReadBuffer(IntPtr buffer, Span<byte> hostBytes)
    {
        fixed (byte* p = hostBytes)
        {
            int err = OpenCLDriver.clEnqueueReadBuffer(_queue, buffer, OpenCLDriver.CL_TRUE, 0, (nuint)hostBytes.Length, p, 0, null, null);
            if (err != OpenCLDriver.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to read from OpenCL buffer: error {err}");
        }
    }

    public IntPtr CompileKernel(string source, string kernelName)
    {
        byte[] srcBytes = Encoding.UTF8.GetBytes(source + "\0");
        fixed (byte* pSrc = srcBytes)
        {
            byte*[] ptrs = [pSrc];
            nuint len = (nuint)(srcBytes.Length - 1);
            fixed (byte** ppSrc = ptrs)
            {
                int pErr = 0;
                IntPtr prog = OpenCLDriver.clCreateProgramWithSource(_context, 1, ppSrc, &len, &pErr);
                if (pErr != OpenCLDriver.CL_SUCCESS)
                    throw new InvalidOperationException($"clCreateProgramWithSource failed: error {pErr}");

                IntPtr dev = _device;
                int bErr = OpenCLDriver.clBuildProgram(prog, 1, &dev, null, null, null);
                if (bErr != OpenCLDriver.CL_SUCCESS)
                {
                    nuint logLen = 0;
                    OpenCLDriver.clGetProgramBuildInfo(prog, _device, 0x1143 /* CL_PROGRAM_BUILD_LOG */, 0, null, &logLen);
                    byte[] logBytes = new byte[(int)logLen];
                    fixed (byte* pLog = logBytes)
                    {
                        OpenCLDriver.clGetProgramBuildInfo(prog, _device, 0x1143, logLen, pLog, null);
                    }
                    string log = Encoding.UTF8.GetString(logBytes);
                    OpenCLDriver.clReleaseProgram(prog);
                    throw new InvalidOperationException($"OpenCL build failed for '{kernelName}':\n{log}");
                }

                byte[] nameBytes = Encoding.UTF8.GetBytes(kernelName + "\0");
                fixed (byte* pName = nameBytes)
                {
                    int kErr = 0;
                    IntPtr kernel = OpenCLDriver.clCreateKernel(prog, pName, &kErr);
                    OpenCLDriver.clReleaseProgram(prog); // refcount held by kernel
                    if (kErr != OpenCLDriver.CL_SUCCESS)
                        throw new InvalidOperationException($"clCreateKernel '{kernelName}' failed: error {kErr}");
                    return kernel;
                }
            }
        }
    }

    public void Finish() => OpenCLDriver.clFinish(_queue);

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_queue != IntPtr.Zero) OpenCLDriver.clReleaseCommandQueue(_queue);
            if (_context != IntPtr.Zero) OpenCLDriver.clReleaseContext(_context);
        }
    }
}
