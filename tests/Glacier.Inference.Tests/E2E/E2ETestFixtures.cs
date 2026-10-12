namespace Glacier.Inference.Tests.E2E;

using System;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Wire precision modes supported by the dynamic cluster transport protocol.
/// </summary>
public enum WirePrecisionMode : byte
{
    FP32 = 0,
    FP16 = 1,
    INT8_Symmetric = 2,
    FP8_E4M3 = 3
}

/// <summary>
/// Flags for fused activation transport frame.
/// </summary>
[Flags]
public enum FusedActivationFlags : byte
{
    None = 0,
    IsSpeculative = 1 << 0,
    IsPrefillChunk = 1 << 1,
    HasScale = 1 << 2,
    FinalLayer = 1 << 3
}

/// <summary>
/// Contract 1: Exactly 32-byte sequentially packed wire header for fused activations.
/// When combined with ClusterFrameHeader (32B), produces a 64-byte naturally aligned header.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 32)]
public readonly struct FusedActivationHeader : IEquatable<FusedActivationHeader>
{
    public const uint FusedMagic = 0x47414354; // "GACT"
    public const int HeaderSize = 32;

    public readonly uint Magic;          // 0..3 (4B)
    public readonly ulong SequenceId;    // 4..11 (8B)
    public readonly int TokenIndex;      // 12..15 (4B)
    public readonly int LayerIndex;      // 16..19 (4B)
    public readonly int HiddenDim;       // 20..23 (4B)
    public readonly byte PrecisionMode;  // 24 (1B)
    public readonly byte Flags;          // 25 (1B)
    public readonly ushort Reserved;     // 26..27 (2B)
    public readonly float Scale;         // 28..31 (4B)

    public FusedActivationHeader(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        int hiddenDim,
        WirePrecisionMode precisionMode,
        FusedActivationFlags flags = FusedActivationFlags.None,
        float scale = 1.0f)
    {
        Magic = FusedMagic;
        SequenceId = sequenceId;
        TokenIndex = tokenIndex;
        LayerIndex = layerIndex;
        HiddenDim = hiddenDim;
        PrecisionMode = (byte)precisionMode;
        Flags = (byte)flags;
        Reserved = 0;
        Scale = scale;
    }

    public WirePrecisionMode TypedPrecision => (WirePrecisionMode)PrecisionMode;
    public FusedActivationFlags TypedFlags => (FusedActivationFlags)Flags;

    public void WriteTo(Span<byte> destination)
    {
        if (destination.Length < HeaderSize)
            throw new ArgumentException("Destination too small for FusedActivationHeader", nameof(destination));

        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(0, 4), Magic);
        BinaryPrimitives.WriteUInt64LittleEndian(destination.Slice(4, 8), SequenceId);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(12, 4), TokenIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(16, 4), LayerIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(20, 4), HiddenDim);
        destination[24] = PrecisionMode;
        destination[25] = Flags;
        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(26, 2), Reserved);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(28, 4), Scale);
    }

    public static FusedActivationHeader ReadFrom(ReadOnlySpan<byte> source)
    {
        if (source.Length < HeaderSize)
            throw new ArgumentException("Source too small for FusedActivationHeader", nameof(source));

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != FusedMagic)
            throw new InvalidOperationException($"Invalid FusedActivationHeader magic: 0x{magic:X8}, expected 0x{FusedMagic:X8}");

        ulong seq = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layer = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int dim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        byte prec = source[24];
        byte flags = source[25];
        float scale = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(28, 4));

        return new FusedActivationHeader(seq, tok, layer, dim, (WirePrecisionMode)prec, (FusedActivationFlags)flags, scale);
    }

    public static bool TryRead(ReadOnlySpan<byte> source, out FusedActivationHeader header)
    {
        if (source.Length < HeaderSize)
        {
            header = default;
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != FusedMagic)
        {
            header = default;
            return false;
        }

        ulong seq = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layer = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int dim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        byte prec = source[24];
        byte flags = source[25];
        float scale = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(28, 4));

        header = new FusedActivationHeader(seq, tok, layer, dim, (WirePrecisionMode)prec, (FusedActivationFlags)flags, scale);
        return true;
    }

    public bool Equals(FusedActivationHeader other) =>
        Magic == other.Magic &&
        SequenceId == other.SequenceId &&
        TokenIndex == other.TokenIndex &&
        LayerIndex == other.LayerIndex &&
        HiddenDim == other.HiddenDim &&
        PrecisionMode == other.PrecisionMode &&
        Flags == other.Flags &&
        Scale.Equals(other.Scale);

    public override bool Equals(object? obj) => obj is FusedActivationHeader other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Magic, SequenceId, TokenIndex, LayerIndex, HiddenDim, PrecisionMode, Scale);
}

/// <summary>
/// Pre-allocated unmanaged pinned buffer pool for zero-allocation socket transport.
/// </summary>
public sealed class PinnedSocketBufferPool : IDisposable
{
    private readonly int _bufferSize;
    private readonly int _maxBuffers;
    private readonly byte[][] _pool;
    private readonly GCHandle[] _handles;
    private readonly bool[] _inUse;
    private readonly object _lock = new();
    private bool _disposed;

    public PinnedSocketBufferPool(int bufferSize = 64 * 1024, int maxBuffers = 16)
    {
        _bufferSize = bufferSize;
        _maxBuffers = maxBuffers;
        _pool = new byte[maxBuffers][];
        _handles = new GCHandle[maxBuffers];
        _inUse = new bool[maxBuffers];

        for (int i = 0; i < maxBuffers; i++)
        {
            _pool[i] = GC.AllocateArray<byte>(bufferSize, pinned: true);
            _handles[i] = GCHandle.Alloc(_pool[i], GCHandleType.Pinned);
            _inUse[i] = false;
        }
    }

    public int BufferSize => _bufferSize;
    public int MaxBuffers => _maxBuffers;

    public Memory<byte> Rent()
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (!_inUse[i])
                {
                    _inUse[i] = true;
                    return _pool[i].AsMemory();
                }
            }
            throw new InvalidOperationException("Pinned buffer pool exhausted!");
        }
    }

    public void Return(Memory<byte> buffer)
    {
        lock (_lock)
        {
            if (_disposed) return;
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (MemoryMarshal.TryGetArray(buffer, out ArraySegment<byte> seg) && ReferenceEquals(seg.Array, _pool[i]))
                {
                    _inUse[i] = false;
                    return;
                }
            }
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (_handles[i].IsAllocated)
                    _handles[i].Free();
            }
        }
    }
}

/// <summary>
/// High-performance reference quantizer for symmetric per-token INT8 and FP16 formats.
/// </summary>
public static class QuantizationOracle
{
    public static float QuantizeSymmetricInt8(ReadOnlySpan<float> input, Span<sbyte> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output length mismatch");

        float maxAbs = 0.0f;
        for (int i = 0; i < input.Length; i++)
        {
            float a = MathF.Abs(input[i]);
            if (a > maxAbs) maxAbs = a;
        }

        if (maxAbs < 1e-8f || float.IsNaN(maxAbs) || float.IsInfinity(maxAbs))
        {
            output.Clear();
            return 1.0f;
        }

        float scale = maxAbs / 127.0f;
        float invScale = 1.0f / scale;

        for (int i = 0; i < input.Length; i++)
        {
            float val = input[i] * invScale;
            int q = (int)MathF.Round(val, MidpointRounding.AwayFromZero);
            output[i] = (sbyte)Math.Clamp(q, -127, 127);
        }

        return scale;
    }

    public static void DequantizeSymmetricInt8(ReadOnlySpan<sbyte> input, float scale, Span<float> output)
    {
        if (input.Length != output.Length)
            throw new ArgumentException("Input and output length mismatch");

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = input[i] * scale;
        }
    }

    public static void PackFloatsToFp16(ReadOnlySpan<float> input, Span<byte> destination)
    {
        if (destination.Length < input.Length * 2)
            throw new ArgumentException("Destination span too small for FP16 pack");

        for (int i = 0; i < input.Length; i++)
        {
            Half h = (Half)input[i];
            BinaryPrimitives.WriteHalfLittleEndian(destination.Slice(i * 2, 2), h);
        }
    }

    public static void UnpackFp16ToFloats(ReadOnlySpan<byte> source, Span<float> destination)
    {
        int count = source.Length / 2;
        if (destination.Length < count)
            throw new ArgumentException("Destination span too small for FP16 unpack");

        for (int i = 0; i < count; i++)
        {
            Half h = BinaryPrimitives.ReadHalfLittleEndian(source.Slice(i * 2, 2));
            destination[i] = (float)h;
        }
    }

    public static double ComputeCosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Lengths must match for cosine similarity");

        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }

        if (normA <= 0.0 || normB <= 0.0) return 0.0;
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public static double ComputeMaxRelativeError(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        if (reference.Length != actual.Length)
            throw new ArgumentException("Lengths must match for relative error");

        double maxRelErr = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            double r = reference[i];
            double a = actual[i];
            double absErr = Math.Abs(r - a);
            double denom = Math.Max(1e-5, Math.Abs(r));
            double relErr = absErr / denom;
            if (relErr > maxRelErr) maxRelErr = relErr;
        }
        return maxRelErr;
    }
}

/// <summary>
/// Reference implementation of Prompt Lookup Draft Provider for speculative N-gram matching.
/// </summary>
public sealed class PromptLookupDraftOracle
{
    private readonly int _maxNgram;
    private readonly int[] _context;
    private int _contextLength;

    public PromptLookupDraftOracle(int maxContext = 4096, int maxNgram = 3)
    {
        _context = new int[maxContext];
        _maxNgram = maxNgram;
        _contextLength = 0;
    }

    public int ContextLength => _contextLength;

    public void AppendToken(int token)
    {
        if (_contextLength < _context.Length)
        {
            _context[_contextLength++] = token;
        }
    }

    public void AppendTokens(ReadOnlySpan<int> tokens)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            AppendToken(tokens[i]);
        }
    }

    public bool TryPredictNextToken(out int predictedToken)
    {
        predictedToken = 0;
        if (_contextLength < 2) return false;

        // Try from longest n-gram down to 2-gram
        for (int n = Math.Min(_maxNgram, _contextLength - 1); n >= 1; n--)
        {
            ReadOnlySpan<int> target = _context.AsSpan(_contextLength - n, n);

            // Search history backwards
            for (int i = _contextLength - n - 1; i >= 0; i--)
            {
                if (i + n < _contextLength && _context.AsSpan(i, n).SequenceEqual(target))
                {
                    predictedToken = _context[i + n];
                    return true;
                }
            }
        }

        return false;
    }

    public void Rollback(int steps = 1)
    {
        _contextLength = Math.Max(0, _contextLength - steps);
    }

    public void Reset()
    {
        _contextLength = 0;
    }
}

/// <summary>
/// Loopback socket test pair for testing opaque socket transport.
/// </summary>
public sealed class TestSocketPair : IDisposable
{
    public Socket Client { get; }
    public Socket Server { get; }
    private readonly TcpListener _listener;

    private TestSocketPair(Socket client, Socket server, TcpListener listener)
    {
        Client = client;
        Server = server;
        _listener = listener;
    }

    public static async Task<TestSocketPair> CreateConnectedAsync()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.NoDelay, true);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        client.NoDelay = true;
        var connectTask = client.ConnectAsync(IPAddress.Loopback, port);
        var server = await listener.AcceptSocketAsync();
        server.NoDelay = true;
        await connectTask;

        return new TestSocketPair(client, server, listener);
    }

    public void Dispose()
    {
        try { Client.Dispose(); } catch { }
        try { Server.Dispose(); } catch { }
        try { _listener.Stop(); } catch { }
    }
}

/// <summary>
/// Helper scope asserting zero GC allocations on hot path.
/// </summary>
public ref struct ZeroAllocScope
{
    private readonly long _initialAllocatedBytes;
    private readonly int _gen0;
    private readonly int _gen1;
    private readonly int _gen2;

    public static ZeroAllocScope Start()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        return new ZeroAllocScope(
            GC.GetAllocatedBytesForCurrentThread(),
            GC.CollectionCount(0),
            GC.CollectionCount(1),
            GC.CollectionCount(2));
    }

    private ZeroAllocScope(long initialBytes, int gen0, int gen1, int gen2)
    {
        _initialAllocatedBytes = initialBytes;
        _gen0 = gen0;
        _gen1 = gen1;
        _gen2 = gen2;
    }

    public void AssertZeroAllocations(string context = "Hot path")
    {
        long finalBytes = GC.GetAllocatedBytesForCurrentThread();
        long diff = finalBytes - _initialAllocatedBytes;
        if (diff > 0)
        {
            throw new InvalidOperationException(
                $"Zero allocation violation in {context}! Allocated bytes: {diff}");
        }
    }
}
