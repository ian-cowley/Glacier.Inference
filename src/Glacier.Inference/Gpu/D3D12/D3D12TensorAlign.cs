#if !ANDROID
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

    public static byte[] AlignIQ2_XXS(ReadOnlySpan<byte> rawIq2, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 68);
        fixed (byte* pSrc = rawIq2, pDst = aligned)
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
                    byte* srcBlk = pS + (long)b * 66;
                    byte* dstBlk = pD + (long)b * 68;
                    dstBlk[0] = srcBlk[0];
                    dstBlk[1] = srcBlk[1];
                    dstBlk[2] = 0;
                    dstBlk[3] = 0;
                    Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, 64, 64);
                }
            });
        }
        return aligned;
    }

    public static byte[] AlignIQ4_NL(ReadOnlySpan<byte> rawIq4Nl, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 20);
        fixed (byte* pSrc = rawIq4Nl, pDst = aligned)
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
                    byte* srcBlk = pS + (long)b * 18;
                    byte* dstBlk = pD + (long)b * 20;
                    dstBlk[0] = srcBlk[0];
                    dstBlk[1] = srcBlk[1];
                    dstBlk[2] = 0;
                    dstBlk[3] = 0;
                    Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, 16, 16);
                }
            });
        }
        return aligned;
    }

    public static byte[] AlignIQ3_S(ReadOnlySpan<byte> rawIq3S, int totalBlocks)
    {
        byte[] aligned = GC.AllocateUninitializedArray<byte>(totalBlocks * 112);
        fixed (byte* pSrc = rawIq3S, pDst = aligned)
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
                    dstBlk[0] = srcBlk[0];
                    dstBlk[1] = srcBlk[1];
                    dstBlk[2] = 0;
                    dstBlk[3] = 0;
                    Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, 108, 108);
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
        if (type == GgufType.IQ2_XXS)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 68UL;
        }
        if (type == GgufType.IQ2_XS)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 76UL;
        }
        if (type == GgufType.IQ2_S)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 84UL;
        }
        if (type == GgufType.IQ3_XXS)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 100UL;
        }
        if (type == GgufType.IQ1_S)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 52UL;
        }
        if (type == GgufType.IQ1_M)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 56UL;
        }
        if (type == GgufType.IQ4_NL)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 20UL;
        }
        if (type == GgufType.GgmlQ2_0)
        {
            int nb = cols / 64;
            return (ulong)rows * (ulong)nb * 20UL;
        }
        if (type == GgufType.IQ3_S)
        {
            int nb = cols / 256;
            return (ulong)rows * (ulong)nb * 112UL;
        }
        if (type == GgufType.Q4_0)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 20UL;
        }
        if (type == GgufType.Q5_0)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 24UL;
        }
        if (type == GgufType.MXFP4)
        {
            int nb = cols / 32;
            return (ulong)rows * (ulong)nb * 20UL;
        }
        if (type == GgufType.BF16 || type == GgufType.F16)
        {
            return (ulong)rows * (ulong)cols * sizeof(float);
        }
        return (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
    }

    /// <summary>
    /// Formats and aligns raw quantized weights directly into a pre-mapped destination buffer without managed heap allocations.
    /// Small tensors (&lt; 2 MiB, e.g. individual MoE experts) execute on a tight zero-overhead pointer loop.
    /// </summary>
    public static ulong AlignTensorToBuffer(GgufType type, byte* srcRaw, byte* dstAligned, int rows, int cols)
    {
        ArgumentNullException.ThrowIfNull(srcRaw);
        ArgumentNullException.ThrowIfNull(dstAligned);

        ulong alignedBytes = GetAlignedTensorBytes(type, rows, cols);
        if (alignedBytes == 0) return 0;

        bool useParallel = alignedBytes >= 2UL * 1024 * 1024;

        switch (type)
        {
            case GgufType.Q3_K:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 110, dstBlockBytes: 112, scaleBytes: 0, useParallel);
                break;
            case GgufType.Q6_K:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 210, dstBlockBytes: 212, scaleBytes: 0, useParallel);
                break;
            case GgufType.Q8_0:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 32), srcBlockBytes: 34, dstBlockBytes: 36, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ2_XXS:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 66, dstBlockBytes: 68, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ2_XS:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 74, dstBlockBytes: 76, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ2_S:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 82, dstBlockBytes: 84, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ3_XXS:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 98, dstBlockBytes: 100, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ1_S:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 50, dstBlockBytes: 52, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ4_NL:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 32), srcBlockBytes: 18, dstBlockBytes: 20, scaleBytes: 2, useParallel);
                break;
            case GgufType.GgmlQ2_0:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 64), srcBlockBytes: 18, dstBlockBytes: 20, scaleBytes: 2, useParallel);
                break;
            case GgufType.IQ3_S:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 256), srcBlockBytes: 110, dstBlockBytes: 112, scaleBytes: 2, useParallel);
                break;
            case GgufType.Q4_0:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 32), srcBlockBytes: 18, dstBlockBytes: 20, scaleBytes: 2, useParallel);
                break;
            case GgufType.Q5_0:
                CopyPaddedBlocks(srcRaw, dstAligned, rows * (cols / 32), srcBlockBytes: 22, dstBlockBytes: 24, scaleBytes: 2, useParallel);
                break;
            case GgufType.MXFP4:
                CopyPaddedMxfp4Blocks(srcRaw, dstAligned, rows * (cols / 32), useParallel);
                break;
            case GgufType.F16:
            {
                int count = rows * cols;
                Half* pSrc = (Half*)srcRaw;
                float* pDst = (float*)dstAligned;
                if (!useParallel)
                {
                    for (int i = 0; i < count; i++)
                    {
                        pDst[i] = (float)pSrc[i];
                    }
                }
                else
                {
                    nint sAddr = (nint)pSrc;
                    nint dAddr = (nint)pDst;
                    int numThreads = Math.Max(1, Environment.ProcessorCount);
                    int chunkSize = (count + numThreads - 1) / numThreads;
                    Parallel.For(0, numThreads, t =>
                    {
                        int start = t * chunkSize;
                        int end = Math.Min(start + chunkSize, count);
                        Half* ps = (Half*)sAddr;
                        float* pd = (float*)dAddr;
                        for (int i = start; i < end; i++)
                        {
                            pd[i] = (float)ps[i];
                        }
                    });
                }
                break;
            }
            case GgufType.BF16:
            {
                int count = rows * cols;
                ushort* pSrc = (ushort*)srcRaw;
                uint* pDstU = (uint*)dstAligned;
                if (!useParallel)
                {
                    for (int i = 0; i < count; i++)
                    {
                        pDstU[i] = ((uint)pSrc[i]) << 16;
                    }
                }
                else
                {
                    nint sAddr = (nint)pSrc;
                    nint dAddr = (nint)pDstU;
                    int numThreads = Math.Max(1, Environment.ProcessorCount);
                    int chunkSize = (count + numThreads - 1) / numThreads;
                    Parallel.For(0, numThreads, t =>
                    {
                        int start = t * chunkSize;
                        int end = Math.Min(start + chunkSize, count);
                        ushort* ps = (ushort*)sAddr;
                        uint* pd = (uint*)dAddr;
                        for (int i = start; i < end; i++)
                        {
                            pd[i] = ((uint)ps[i]) << 16;
                        }
                    });
                }
                break;
            }
            default:
                Buffer.MemoryCopy(srcRaw, dstAligned, alignedBytes, alignedBytes);
                break;
        }

        return alignedBytes;
    }

    private static void CopyPaddedBlocks(
        byte* srcRaw,
        byte* dstAligned,
        int totalBlocks,
        int srcBlockBytes,
        int dstBlockBytes,
        int scaleBytes,
        bool useParallel)
    {
        if (!useParallel || totalBlocks < 4096)
        {
            CopyBlockRange(srcRaw, dstAligned, 0, totalBlocks, srcBlockBytes, dstBlockBytes, scaleBytes);
            return;
        }

        nint srcAddr = (nint)srcRaw;
        nint dstAddr = (nint)dstAligned;
        int numThreads = Math.Max(1, Environment.ProcessorCount);
        int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
        Parallel.For(0, numThreads, t =>
        {
            int start = t * chunkSize;
            int end = Math.Min(start + chunkSize, totalBlocks);
            CopyBlockRange((byte*)srcAddr, (byte*)dstAddr, start, end, srcBlockBytes, dstBlockBytes, scaleBytes);
        });
    }

    private static void CopyBlockRange(
        byte* pS,
        byte* pD,
        int start,
        int end,
        int srcBlockBytes,
        int dstBlockBytes,
        int scaleBytes)
    {
        if (scaleBytes == 2)
        {
            int tailBytes = srcBlockBytes - 2;
            for (int b = start; b < end; b++)
            {
                byte* srcBlk = pS + (long)b * srcBlockBytes;
                byte* dstBlk = pD + (long)b * dstBlockBytes;
                *(ushort*)dstBlk = *(ushort*)srcBlk;
                dstBlk[2] = 0;
                dstBlk[3] = 0;
                Buffer.MemoryCopy(srcBlk + 2, dstBlk + 4, tailBytes, tailBytes);
            }
        }
        else
        {
            for (int b = start; b < end; b++)
            {
                byte* srcBlk = pS + (long)b * srcBlockBytes;
                byte* dstBlk = pD + (long)b * dstBlockBytes;
                Buffer.MemoryCopy(srcBlk, dstBlk, srcBlockBytes, srcBlockBytes);
                dstBlk[srcBlockBytes] = 0;
                dstBlk[srcBlockBytes + 1] = 0;
            }
        }
    }

    private static void CopyPaddedMxfp4Blocks(byte* srcRaw, byte* dstAligned, int totalBlocks, bool useParallel)
    {
        static void CopyRange(byte* pS, byte* pD, int start, int end)
        {
            for (int b = start; b < end; b++)
            {
                byte* srcBlk = pS + (long)b * 17;
                byte* dstBlk = pD + (long)b * 20;
                dstBlk[0] = srcBlk[0];
                dstBlk[1] = 0;
                dstBlk[2] = 0;
                dstBlk[3] = 0;
                Buffer.MemoryCopy(srcBlk + 1, dstBlk + 4, 16, 16);
            }
        }

        if (!useParallel || totalBlocks < 4096)
        {
            CopyRange(srcRaw, dstAligned, 0, totalBlocks);
            return;
        }

        nint srcAddr = (nint)srcRaw;
        nint dstAddr = (nint)dstAligned;
        int numThreads = Math.Max(1, Environment.ProcessorCount);
        int chunkSize = (totalBlocks + numThreads - 1) / numThreads;
        Parallel.For(0, numThreads, t =>
        {
            int start = t * chunkSize;
            int end = Math.Min(start + chunkSize, totalBlocks);
            CopyRange((byte*)srcAddr, (byte*)dstAddr, start, end);
        });
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
        if (type == GgufType.IQ2_XXS)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 66);
            byte[] aligned = AlignIQ2_XXS(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ2_XS)
        {
            int totalBlocks = rows * (cols / 256);
            ulong alignedBytes = (ulong)totalBlocks * 76UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 74, dstBlockBytes: 76, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ2_S)
        {
            int totalBlocks = rows * (cols / 256);
            ulong alignedBytes = (ulong)totalBlocks * 84UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 82, dstBlockBytes: 84, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ3_XXS)
        {
            int totalBlocks = rows * (cols / 256);
            ulong alignedBytes = (ulong)totalBlocks * 100UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 98, dstBlockBytes: 100, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ1_S)
        {
            int totalBlocks = rows * (cols / 256);
            ulong alignedBytes = (ulong)totalBlocks * 52UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 50, dstBlockBytes: 52, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ4_NL)
        {
            int nb = cols / 32;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 18);
            byte[] aligned = AlignIQ4_NL(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.GgmlQ2_0)
        {
            int nb = cols / 64;
            int totalBlocks = rows * nb;
            ulong alignedBytes = (ulong)totalBlocks * 20UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 18, dstBlockBytes: 20, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.IQ3_S)
        {
            int nb = cols / 256;
            int totalBlocks = rows * nb;
            var raw = new ReadOnlySpan<byte>((void*)hostPtr, totalBlocks * 110);
            byte[] aligned = AlignIQ3_S(raw, totalBlocks);
            var buf = ctx.CreateDeviceBuffer((ulong)aligned.Length);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.Q4_0)
        {
            int totalBlocks = rows * (cols / 32);
            ulong alignedBytes = (ulong)totalBlocks * 20UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 18, dstBlockBytes: 20, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.Q5_0)
        {
            int totalBlocks = rows * (cols / 32);
            ulong alignedBytes = (ulong)totalBlocks * 24UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedBlocks((byte*)hostPtr, pDst, totalBlocks, srcBlockBytes: 22, dstBlockBytes: 24, scaleBytes: 2, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.MXFP4)
        {
            int totalBlocks = rows * (cols / 32);
            ulong alignedBytes = (ulong)totalBlocks * 20UL;
            byte[] aligned = GC.AllocateUninitializedArray<byte>(checked((int)alignedBytes));
            fixed (byte* pDst = aligned)
            {
                CopyPaddedMxfp4Blocks((byte*)hostPtr, pDst, totalBlocks, useParallel: alignedBytes >= 2UL * 1024 * 1024);
            }
            var buf = ctx.CreateDeviceBuffer(alignedBytes);
            ctx.CopyToDevice(buf, aligned);
            return buf;
        }
        if (type == GgufType.F16)
        {
            int count = rows * cols;
            float[] f32 = GC.AllocateUninitializedArray<float>(count);
            Half* pSrc = (Half*)hostPtr;
            fixed (float* pDst = f32)
            {
                for (int i = 0; i < count; i++)
                {
                    pDst[i] = (float)pSrc[i];
                }
            }
            var buf = ctx.CreateDeviceBuffer((ulong)count * sizeof(float));
            ctx.CopyToDevice(buf, MemoryMarshal.AsBytes(f32.AsSpan()));
            return buf;
        }
        if (type == GgufType.BF16)
        {
            int count = rows * cols;
            float[] f32 = GC.AllocateUninitializedArray<float>(count);
            ushort* pSrc = (ushort*)hostPtr;
            fixed (float* pDst = f32)
            {
                uint* pDstU = (uint*)pDst;
                int numThreads = Math.Max(1, Environment.ProcessorCount);
                int chunkSize = (count + numThreads - 1) / numThreads;
                Parallel.For(0, numThreads, t =>
                {
                    int start = t * chunkSize;
                    int end = Math.Min(start + chunkSize, count);
                    for (int i = start; i < end; i++)
                    {
                        pDstU[i] = ((uint)pSrc[i]) << 16;
                    }
                });
            }
            var buf = ctx.CreateDeviceBuffer((ulong)count * sizeof(float));
            ctx.CopyToDevice(buf, MemoryMarshal.AsBytes(f32.AsSpan()));
            return buf;
        }

        ulong totalBytes = (ulong)rows * (ulong)GgufTypes.GetRowBytes(type, cols);
        var buffer = ctx.CreateDeviceBuffer(totalBytes);
        ctx.CopyToDevice(buffer, hostPtr, totalBytes);
        return buffer;
    }

    /// <summary>
    /// Uploads an even-column BF16 matrix directly in compact 2-byte form (50% VRAM savings vs F32 promotion).
    /// </summary>
    public static ID3D12Resource UploadCompactBf16Tensor(D3D12Context ctx, IntPtr hostPtr, int rows, int cols)
    {
        if ((cols & 1) != 0)
            return UploadAlignedTensor(ctx, GgufType.BF16, hostPtr, rows, cols);

        ulong totalBytes = (ulong)rows * (ulong)cols * 2UL;
        var buffer = ctx.CreateDeviceBuffer(totalBytes);
        ctx.CopyToDevice(buffer, hostPtr, totalBytes);
        return buffer;
    }
}
#endif
