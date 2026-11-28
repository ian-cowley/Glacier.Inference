namespace Glacier.Inference.Gpu.OpenCL;

using System;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Direct, zero-dependency P/Invoke bindings to the OpenCL runtime.
/// On Android, binds to Qualcomm's Adreno OpenCL driver (/vendor/lib64/libOpenCL.so).
/// On Windows, binds to OpenCL.dll.
/// </summary>
public static unsafe partial class OpenCLDriver
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
                if (NativeLibrary.TryLoad("OpenCL.dll", out IntPtr handle) && handle != IntPtr.Zero)
                {
                    NativeLibrary.Free(handle);
                    return true;
                }
                return false;
            }

            if (NativeLibrary.TryLoad("libOpenCL.so", out IntPtr h) && h != IntPtr.Zero)
            {
                NativeLibrary.Free(h);
                return true;
            }

            if (NativeLibrary.TryLoad("/vendor/lib64/libOpenCL.so", out h) && h != IntPtr.Zero)
            {
                NativeLibrary.Free(h);
                return true;
            }

            if (NativeLibrary.TryLoad("/system/vendor/lib64/libOpenCL.so", out h) && h != IntPtr.Zero)
            {
                NativeLibrary.Free(h);
                return true;
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    [LibraryImport(OpenCLLib, EntryPoint = "clGetPlatformIDs")]
    public static partial int clGetPlatformIDs(uint num_entries, IntPtr* platforms, uint* num_platforms);

    [LibraryImport(OpenCLLib, EntryPoint = "clGetPlatformInfo")]
    public static partial int clGetPlatformInfo(IntPtr platform, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clGetDeviceIDs")]
    public static partial int clGetDeviceIDs(IntPtr platform, ulong device_type, uint num_entries, IntPtr* devices, uint* num_devices);

    [LibraryImport(OpenCLLib, EntryPoint = "clGetDeviceInfo")]
    public static partial int clGetDeviceInfo(IntPtr device, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clCreateContext")]
    public static partial IntPtr clCreateContext(IntPtr* properties, uint num_devices, IntPtr* devices, void* pfn_notify, void* user_data, int* errcode_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clReleaseContext")]
    public static partial int clReleaseContext(IntPtr context);

    [LibraryImport(OpenCLLib, EntryPoint = "clCreateCommandQueue")]
    public static partial IntPtr clCreateCommandQueue(IntPtr context, IntPtr device, ulong properties, int* errcode_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clReleaseCommandQueue")]
    public static partial int clReleaseCommandQueue(IntPtr command_queue);

    [LibraryImport(OpenCLLib, EntryPoint = "clCreateBuffer")]
    public static partial IntPtr clCreateBuffer(IntPtr context, ulong flags, nuint size, void* host_ptr, int* errcode_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clReleaseMemObject")]
    public static partial int clReleaseMemObject(IntPtr memobj);

    [LibraryImport(OpenCLLib, EntryPoint = "clCreateProgramWithSource")]
    public static partial IntPtr clCreateProgramWithSource(IntPtr context, uint count, byte** strings, nuint* lengths, int* errcode_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clBuildProgram")]
    public static partial int clBuildProgram(IntPtr program, uint num_devices, IntPtr* device_list, byte* options, void* pfn_notify, void* user_data);

    [LibraryImport(OpenCLLib, EntryPoint = "clGetProgramBuildInfo")]
    public static partial int clGetProgramBuildInfo(IntPtr program, IntPtr device, uint param_name, nuint param_value_size, void* param_value, nuint* param_value_size_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clReleaseProgram")]
    public static partial int clReleaseProgram(IntPtr program);

    [LibraryImport(OpenCLLib, EntryPoint = "clCreateKernel")]
    public static partial IntPtr clCreateKernel(IntPtr program, byte* kernel_name, int* errcode_ret);

    [LibraryImport(OpenCLLib, EntryPoint = "clReleaseKernel")]
    public static partial int clReleaseKernel(IntPtr kernel);

    [LibraryImport(OpenCLLib, EntryPoint = "clSetKernelArg")]
    public static partial int clSetKernelArg(IntPtr kernel, uint arg_index, nuint arg_size, void* arg_value);

    [LibraryImport(OpenCLLib, EntryPoint = "clEnqueueNDRangeKernel")]
    public static partial int clEnqueueNDRangeKernel(IntPtr command_queue, IntPtr kernel, uint work_dim, nuint* global_work_offset, nuint* global_work_size, nuint* local_work_size, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [LibraryImport(OpenCLLib, EntryPoint = "clEnqueueWriteBuffer")]
    public static partial int clEnqueueWriteBuffer(IntPtr command_queue, IntPtr buffer, uint blocking_write, nuint offset, nuint size, void* ptr, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [LibraryImport(OpenCLLib, EntryPoint = "clEnqueueReadBuffer")]
    public static partial int clEnqueueReadBuffer(IntPtr command_queue, IntPtr buffer, uint blocking_read, nuint offset, nuint size, void* ptr, uint num_events_in_wait_list, IntPtr* event_wait_list, IntPtr* @event);

    [LibraryImport(OpenCLLib, EntryPoint = "clFinish")]
    public static partial int clFinish(IntPtr command_queue);
}
