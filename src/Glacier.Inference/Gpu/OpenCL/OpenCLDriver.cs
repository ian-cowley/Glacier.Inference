namespace Glacier.Inference.Gpu.OpenCL;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the OpenCL runtime.
/// On Android, binds to Qualcomm's Adreno OpenCL driver (/vendor/lib64/libOpenCL.so).
/// On Windows, binds to OpenCL.dll.
/// </summary>
public static unsafe class OpenCLDriver
{
    private const string OpenCLLib = "libOpenCL.so";

    public const int CL_SUCCESS = 0;
    public const int CL_DEVICE_TYPE_GPU = 1 << 2;
    public const int CL_DEVICE_NAME = 0x102B;
    public const int CL_DEVICE_VENDOR = 0x102C;
    public const int CL_DRIVER_VERSION = 0x102D;
    public const int CL_DEVICE_VERSION = 0x102F;
    public const int CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
    public const int CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
    public const int CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
    public const int CL_DEVICE_EXTENSIONS = 0x1030;
    public const int CL_PLATFORM_NAME = 0x0902;
    public const int CL_PLATFORM_VERSION = 0x0901;

    public const ulong CL_MEM_READ_WRITE = 1 << 0;
    public const ulong CL_MEM_WRITE_ONLY = 1 << 1;
    public const ulong CL_MEM_READ_ONLY = 1 << 2;
    public const ulong CL_MEM_USE_HOST_PTR = 1 << 3;
    public const ulong CL_MEM_COPY_HOST_PTR = 1 << 5;

    public const uint CL_TRUE = 1;
    public const uint CL_FALSE = 0;

    static OpenCLDriver()
    {
        NativeDriverResolver.EnsureRegistered();
    }

    public static bool IsAvailable()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.TryLoad("OpenCL.dll", out IntPtr handle) && handle != IntPtr.Zero;
            }
            return (NativeLibrary.TryLoad("libOpenCL.so", out IntPtr h) ||
                    NativeLibrary.TryLoad("/vendor/lib64/libOpenCL.so", out h) ||
                    NativeLibrary.TryLoad("/system/vendor/lib64/libOpenCL.so", out h)) && h != IntPtr.Zero;
        }
        catch
        {
            return false;
        }
    }

    [DllImport(OpenCLLib, EntryPoint = "clGetPlatformIDs")]
    public static extern int clGetPlatformIDs(uint num_entries, IntPtr* platforms, uint* num_platforms);

    [DllImport(OpenCLLib, EntryPoint = "clGetPlatformInfo")]
    public static extern int clGetPlatformInfo(IntPtr platform, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [DllImport(OpenCLLib, EntryPoint = "clGetDeviceIDs")]
    public static extern int clGetDeviceIDs(IntPtr platform, ulong device_type, uint num_entries, IntPtr* devices, uint* num_devices);

    [DllImport(OpenCLLib, EntryPoint = "clGetDeviceInfo")]
    public static extern int clGetDeviceInfo(IntPtr device, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [DllImport(OpenCLLib, EntryPoint = "clCreateContext")]
    public static extern IntPtr clCreateContext(IntPtr* properties, uint num_devices, IntPtr* devices, void* pfn_notify, void* user_data, int* errcode_ret);

    [DllImport(OpenCLLib, EntryPoint = "clReleaseContext")]
    public static extern int clReleaseContext(IntPtr context);

    [DllImport(OpenCLLib, EntryPoint = "clCreateCommandQueue")]
    public static extern IntPtr clCreateCommandQueue(IntPtr context, IntPtr device, ulong properties, int* errcode_ret);

    [DllImport(OpenCLLib, EntryPoint = "clReleaseCommandQueue")]
    public static extern int clReleaseCommandQueue(IntPtr command_queue);

    [DllImport(OpenCLLib, EntryPoint = "clCreateBuffer")]
    public static extern IntPtr clCreateBuffer(IntPtr context, ulong flags, nuint size, void* host_ptr, int* errcode_ret);

    [DllImport(OpenCLLib, EntryPoint = "clReleaseMemObject")]
    public static extern int clReleaseMemObject(IntPtr memobj);

    [DllImport(OpenCLLib, EntryPoint = "clCreateProgramWithSource")]
    public static extern IntPtr clCreateProgramWithSource(IntPtr context, uint count, byte** strings, nuint* lengths, int* errcode_ret);

    [DllImport(OpenCLLib, EntryPoint = "clBuildProgram")]
    public static extern int clBuildProgram(IntPtr program, uint num_devices, IntPtr* device_list, byte* options, void* pfn_notify, void* user_data);

    [DllImport(OpenCLLib, EntryPoint = "clGetProgramBuildInfo")]
    public static extern int clGetProgramBuildInfo(IntPtr program, IntPtr device, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [DllImport(OpenCLLib, EntryPoint = "clReleaseProgram")]
    public static extern int clReleaseProgram(IntPtr program);

    [DllImport(OpenCLLib, EntryPoint = "clCreateKernel")]
    public static extern IntPtr clCreateKernel(IntPtr program, byte* kernel_name, int* errcode_ret);

    [DllImport(OpenCLLib, EntryPoint = "clReleaseKernel")]
    public static extern int clReleaseKernel(IntPtr kernel);

    [DllImport(OpenCLLib, EntryPoint = "clSetKernelArg")]
    public static extern int clSetKernelArg(IntPtr kernel, uint arg_index, nuint arg_size, void* arg_value);

    [DllImport(OpenCLLib, EntryPoint = "clEnqueueNDRangeKernel")]
    public static extern int clEnqueueNDRangeKernel(IntPtr command_queue, IntPtr kernel, uint work_dim, nuint* global_work_offset, nuint* global_work_size, nuint* local_work_size, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [DllImport(OpenCLLib, EntryPoint = "clEnqueueWriteBuffer")]
    public static extern int clEnqueueWriteBuffer(IntPtr command_queue, IntPtr buffer, uint blocking_write, nuint offset, nuint size, void* ptr, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [DllImport(OpenCLLib, EntryPoint = "clEnqueueReadBuffer")]
    public static extern int clEnqueueReadBuffer(IntPtr command_queue, IntPtr buffer, uint blocking_read, nuint offset, nuint size, void* ptr, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [DllImport(OpenCLLib, EntryPoint = "clFinish")]
    public static extern int clFinish(IntPtr command_queue);
}
