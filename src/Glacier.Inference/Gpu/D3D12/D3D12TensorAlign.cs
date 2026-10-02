namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Inference.Gguf;
using Vortice.Direct3D12;

/// <summary>
/// Helper utilities for aligning quantized tensor memory buffers to 4-byte boundaries
/// required by Direct3D 12 ByteAddressBuffer loads, and uploading them cleanly to GPU device buffers.
/// </summary>
public static unsafe class D3D12TensorAlign
{
    public const int GemvChunkRows = 32768;

    public static byte[] AlignQ3K(ReadOnlySpan<byte> rawQ3K, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 112);
        fixed (byte* pSrc = rawQ3K, pDst = aligned)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;
            int numThreads = Math.Max(1, Environment.ProcessorCount);
            int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
            Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, totalBlocks);
                byte* pS = (byte*)srcAddr;
                byte* pD = (byte*)dstAddr;
                for (int b = start; b < end; b++)
                {
                    byte* srcBlk = pS + (long)b * 110;
                    byte* dstBlk = pD + (long)b * 112;
                    Buffer.MemoryCopy(srcBlk, dstBlk, 110, 110);
                    dstBlk[110] = 0;
                    dstBlk[111] = 0;
                }
            });
        }
        return aligned;
    }

    public static byte[] AlignQ6K(ReadOnlySpan<byte> rawQ6K, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 212);
        fixed (byte* pSrc = rawQ6K, pDst = aligned)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;
            int numThreads = Math.Max(1, Environment.ProcessorCount);
            int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
            Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, totalBlocks);
                byte* pS = (byte*)srcAddr;
                byte* pD = (byte*)dstAddr;
                for (int b = start; b < end; b++)
                {
                    byte* srcBlk = pS + (long)b * 210;
                    byte* dstBlk = pD + (long)b * 212;
                    Buffer.MemoryCopy(srcBlk, dstBlk, 210, 210);
                    dstBlk[210] = 0;
                    dstBlk[211] = 0;
                }
            });
        }
        return aligned;
    }

    public static byte[] AlignQ8_0(ReadOnlySpan<byte> rawQ8_0, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 36);
        fixed (byte* pSrc = rawQ8_0, pDst = aligned)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;
            int numThreads = Math.Max(1, Environment.ProcessorCount);
            int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
            Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(start + chunkSize, totalBlocks);
                byte* pS = (byte*)srcAddr;
                byte* pD = (byte*)dstAddr;
                for (int b = start; b < end; b++)
                {
                    byte* srcBlk = pS + (long)b * 34;
                    byte* dstBlk = pD + (long)b * 36;
                    *(ushort*)dstBlk = *(ushort*)srcBlk;
                    dstBlk[2] = 0;
                    dstBlk[3] = 0;
                    Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, 32, 32);
                }
            });
        }
        return aligned;
    }

    public static ulong GetAlignedTensorBytes(GgufType type, int rows, int cols)
    {
        if (type == GgufType.Q3_K)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 112UL;
        }
        if (type == GgufType.Q6_K)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 212UL;
        }
        if (type == GgufType.Q8_0)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 36UL;
        }
        return (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
    }

    public static ID3D12Resource UploadAlignedTensor(D3D12Context ctx, GgufType type, IntPtr hostPtr, int rows, int cols)
    {
        if (type == GgufType.Q3_K)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 110);
            byte[] aligned = AlignQ3K(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.Q6_K)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 210);
            byte[] aligned = AlignQ6K(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.Q8_0)
        {
            int nb = cols / 32;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 34);
            byte[] aligned = AlignQ8_0(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }

        ulong totalBytes = (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
        var buffer = ctx.CreateDeviceBuffer(totalBytes);
        ctx.CopyToDevice(buffer, hostPtr, totalBytes);
        return buffer;
    }
}
