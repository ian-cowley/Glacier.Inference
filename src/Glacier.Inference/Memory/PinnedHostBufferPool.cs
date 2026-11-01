namespace Glacier.Inference.Memory;

using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Inference.Gpu;

/// <summary>
/// Thread-safe unmanaged pinned host buffer pool.
/// Pinned memory pages are locked in physical RAM, preventing paging and enabling direct hardware DMA transfers.
/// </summary>
public sealed unsafe class PinnedHostBufferPool : IDisposable
{
    public sealed class PinnedBuffer : IDisposable
    {
        private readonly PinnedHostBufferPool _pool;
        private byte* _pointer;
        private readonly nuint _sizeBytes;
        private readonly bool _isCudaPinned;
        private int _isDisposed;

        public byte* Pointer => _pointer;
        public nuint SizeBytes => _sizeBytes;
        public bool IsCudaPinned => _isCudaPinned;

        internal PinnedBuffer(PinnedHostBufferPool pool, byte* ptr, nuint size, bool isCuda)
        {
            _pool = pool;
            _pointer = ptr;
            _sizeBytes = size;
            _isCudaPinned = isCuda;
        }

        public Span<byte> AsSpan() => new(_pointer, checked((int)_sizeBytes));

        public Span<byte> AsSpan(int start, int length) => new(_pointer + start, length);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _isDisposed, 1) == 0)
            {
                _pool.Return(this);
            }
        }

        internal void Reset()
        {
            _isDisposed = 0;
        }

        internal void FreeInternal()
        {
            if (_pointer != null)
            {
                if (_isCudaPinned)
                {
                    FreeCudaPinned((IntPtr)_pointer);
                }
                else if (OperatingSystem.IsWindows())
                {
                    FreeWindowsLocked(_pointer, _sizeBytes);
                }
                else
                {
                    NativeMemory.AlignedFree(_pointer);
                }
                _pointer = null;
            }
        }
    }

    private readonly nuint _bufferSizeBytes;
    private readonly int _maxBuffers;
    private readonly bool _useCudaPinning;
    private readonly ConcurrentBag<PinnedBuffer> _freeBuffers = new();
    private int _allocatedBuffers;
    private bool _disposed;

    public nuint BufferSizeBytes => _bufferSizeBytes;
    public int MaxBuffers => _maxBuffers;
    public int FreeBufferCount => _freeBuffers.Count;
    public int TotalAllocatedCount => _allocatedBuffers;
    public bool UseCudaPinning => _useCudaPinning;

    public PinnedHostBufferPool(nuint bufferSizeBytes, int maxBuffers = 16, bool preferCuda = true)
    {
        if (bufferSizeBytes == 0) throw new ArgumentOutOfRangeException(nameof(bufferSizeBytes));
        if (maxBuffers <= 0) throw new ArgumentOutOfRangeException(nameof(maxBuffers));

        _bufferSizeBytes = bufferSizeBytes;
        _maxBuffers = maxBuffers;

        bool cudaAvailable = false;
        if (preferCuda)
        {
            try
            {
                cudaAvailable = CuDriver.IsAvailable();
            }
            catch
            {
                cudaAvailable = false;
            }
        }

        _useCudaPinning = cudaAvailable;
    }

    /// <summary>
    /// Rents a pinned memory buffer from the pool, or allocates a new pinned buffer if capacity allows.
    /// </summary>
    public PinnedBuffer Rent()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_freeBuffers.TryTake(out var buffer))
        {
            buffer.Reset();
            return buffer;
        }

        if (Interlocked.Increment(ref _allocatedBuffers) <= _maxBuffers)
        {
            try
            {
                return AllocateNewBuffer();
            }
            catch
            {
                Interlocked.Decrement(ref _allocatedBuffers);
                throw;
            }
        }

        // Capacity reached: spin or wait for returned buffer
        Interlocked.Decrement(ref _allocatedBuffers);
        SpinWait spinner = new();
        while (!_disposed)
        {
            if (_freeBuffers.TryTake(out buffer))
            {
                buffer.Reset();
                return buffer;
            }
            spinner.SpinOnce();
        }

        throw new ObjectDisposedException(nameof(PinnedHostBufferPool));
    }

    private PinnedBuffer AllocateNewBuffer()
    {
        byte* ptr = null;
        bool isCuda = false;

        if (_useCudaPinning && TryAllocCudaPinned(_bufferSizeBytes, out IntPtr pp))
        {
            ptr = (byte*)pp;
            isCuda = true;
        }

        if (ptr == null)
        {
            if (OperatingSystem.IsWindows())
            {
                ptr = TryAllocWindowsLocked(_bufferSizeBytes);
            }

            if (ptr == null)
            {
                // Fallback to 64-byte aligned unmanaged memory
                ptr = (byte*)NativeMemory.AlignedAlloc(_bufferSizeBytes, 64);
            }
        }

        return new PinnedBuffer(this, ptr, _bufferSizeBytes, isCuda);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool TryAllocCudaPinned(nuint sizeBytes, out IntPtr pp)
    {
        int err = CuDriver.MemHostAlloc(out pp, sizeBytes, 0);
        return err == 0 && pp != IntPtr.Zero;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void FreeCudaPinned(IntPtr ptr)
    {
        CuDriver.MemFreeHost(ptr);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static byte* TryAllocWindowsLocked(nuint sizeBytes)
    {
        void* vptr = VirtualMemoryInterop.VirtualAlloc(
            null,
            sizeBytes,
            VirtualMemoryInterop.MEM_COMMIT | VirtualMemoryInterop.MEM_RESERVE,
            VirtualMemoryInterop.PAGE_READWRITE);

        if (vptr != null)
        {
            VirtualMemoryInterop.VirtualLock(vptr, sizeBytes);
            return (byte*)vptr;
        }

        return null;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void FreeWindowsLocked(byte* ptr, nuint sizeBytes)
    {
        VirtualMemoryInterop.VirtualUnlock(ptr, sizeBytes);
        VirtualMemoryInterop.VirtualFree(ptr, 0, VirtualMemoryInterop.MEM_RELEASE);
    }

    internal void Return(PinnedBuffer buffer)
    {
        if (_disposed)
        {
            buffer.FreeInternal();
            return;
        }

        _freeBuffers.Add(buffer);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            while (_freeBuffers.TryTake(out var buffer))
            {
                buffer.FreeInternal();
            }
        }
    }
}
