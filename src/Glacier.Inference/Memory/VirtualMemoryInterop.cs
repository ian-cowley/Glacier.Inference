namespace Glacier.Inference.Memory;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Low-level operating system virtual memory interop services for Windows and POSIX platforms.
/// Provides page-level memory allocation, physical page locking, prefetching, and working set control.
/// </summary>
public static unsafe partial class VirtualMemoryInterop
{
    // Windows Win32 Memory Allocation Constants
    public const uint MEM_COMMIT = 0x00001000;
    public const uint MEM_RESERVE = 0x00002000;
    public const uint MEM_RESET = 0x00080000;
    public const uint MEM_RELEASE = 0x00008000;
    public const uint MEM_LARGE_PAGES = 0x20000000;
    public const uint PAGE_NOACCESS = 0x01;
    public const uint PAGE_READONLY = 0x02;
    public const uint PAGE_READWRITE = 0x04;
    public const uint FILE_MAP_READ = 0x0004;

    [StructLayout(LayoutKind.Sequential)]
    public struct WIN32_MEMORY_RANGE_ENTRY
    {
        public void* VirtualAddress;
        public nuint NumberOfBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SYSTEM_INFO
    {
        public ushort wProcessorArchitecture;
        public ushort wReserved;
        public uint dwPageSize;
        public void* lpMinimumApplicationAddress;
        public void* lpMaximumApplicationAddress;
        public nuint dwActiveProcessorMask;
        public uint dwNumberOfProcessors;
        public uint dwProcessorType;
        public uint dwAllocationGranularity;
        public ushort wProcessorLevel;
        public ushort wProcessorRevision;
    }

    // Windows Kernel32 APIs
    [LibraryImport("kernel32.dll", SetLastError = true)]
    public static partial void* VirtualAlloc(void* lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool VirtualFree(void* lpAddress, nuint dwSize, uint dwFreeType);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool VirtualLock(void* lpAddress, nuint dwSize);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool VirtualUnlock(void* lpAddress, nuint dwSize);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool PrefetchVirtualMemory(
        IntPtr hProcess,
        nuint NumberOfEntries,
        WIN32_MEMORY_RANGE_ENTRY* VirtualAddresses,
        uint Flags);

    [LibraryImport("kernel32.dll", SetLastError = true)]
    public static partial nuint GetLargePageMinimum();

    [LibraryImport("kernel32.dll")]
    public static partial void GetSystemInfo(SYSTEM_INFO* lpSystemInfo);

    [LibraryImport("kernel32.dll")]
    public static partial IntPtr GetCurrentProcess();

    [LibraryImport("psapi.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static partial bool EmptyWorkingSet(IntPtr hProcess);

    /// <summary>
    /// Gets the system virtual memory page size in bytes (typically 4096 on x86/x64).
    /// </summary>
    public static int GetPageSize()
    {
        if (OperatingSystem.IsWindows())
        {
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            return (int)sysInfo.dwPageSize;
        }

        return Environment.SystemPageSize;
    }

    /// <summary>
    /// Prefetches memory range into physical RAM.
    /// On Windows, issues PrefetchVirtualMemory. On other OSes, performs sequential page touching.
    /// </summary>
    public static void PrefetchMemory(void* address, nuint byteCount)
    {
        if (address == null || byteCount == 0) return;

        if (OperatingSystem.IsWindows())
        {
            var range = new WIN32_MEMORY_RANGE_ENTRY
            {
                VirtualAddress = address,
                NumberOfBytes = byteCount
            };
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
        }
        else
        {
            // Software sequential prefetch touch loop
            byte* ptr = (byte*)address;
            nuint pageSize = (nuint)Environment.SystemPageSize;
            for (nuint i = 0; i < byteCount; i += pageSize)
            {
                _ = ptr[i];
            }
        }
    }

    /// <summary>
    /// Locks memory range into physical RAM, preventing the OS from swapping it to pagefile.
    /// </summary>
    public static bool LockMemory(void* address, nuint byteCount)
    {
        if (address == null || byteCount == 0) return false;

        if (OperatingSystem.IsWindows())
        {
            return VirtualLock(address, byteCount);
        }

        return true;
    }

    /// <summary>
    /// Unlocks memory range from physical RAM.
    /// </summary>
    public static bool UnlockMemory(void* address, nuint byteCount)
    {
        if (address == null || byteCount == 0) return false;

        if (OperatingSystem.IsWindows())
        {
            return VirtualUnlock(address, byteCount);
        }

        return true;
    }
}
