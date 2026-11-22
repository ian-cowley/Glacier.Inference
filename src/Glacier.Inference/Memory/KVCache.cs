namespace Glacier.Inference.Memory;

using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.Inference.Quant;

/// <summary>
/// Contiguous unmanaged Key-Value Cache for autoregressive Transformer inference.
/// Zero-allocation ring/linear buffer supporting Grouped Query Attention (GQA)
/// across <see cref="KvCachePrecision.Fp32"/>, <see cref="KvCachePrecision.Fp16"/>,
/// <see cref="KvCachePrecision.Fp8"/>, symmetric 8-bit <see cref="KvCachePrecision.Q8_0"/>,
/// and packed 4-bit <see cref="KvCachePrecision.Q4_0"/> (<see cref="KvCachePrecision.Int4"/>).
/// Instances are NOT thread-safe; single-thread access per generation stream is required.
/// </summary>
public sealed unsafe class KVCache : IDisposable
{
    private readonly int _layers;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly int _vHeadDim;
    private readonly int _maxSeqLen;
    private readonly KvCachePrecision _precision;
    private readonly long _slotBytesK;
    private readonly long _slotBytesV;
    private readonly long _layerByteStrideK;
    private readonly long _layerByteStrideV;
    private readonly long _headByteStrideK;
    private readonly long _headByteStrideV;
    private readonly long _totalBytesK;
    private readonly long _totalBytesV;

    private int _slidingWindow;
    private bool _isRingBuffer;
    private byte* _kBuffer;
    private byte* _vBuffer;
    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int Layers => _layers;
    public int HeadsKv => _nHeadsKv;
    public int HeadDim => _headDim;
    public int ValueHeadDim => _vHeadDim;
    public KvCachePrecision Precision => _precision;
    public long KeyBufferBytes => _totalBytesK;
    public long ValueBufferBytes => _totalBytesV;
    public long TotalBytes => _totalBytesK + _totalBytesV;

    public long GetByteSize() => TotalBytes;

    public bool IsRingBuffer
    {
        get => _isRingBuffer;
        set => _isRingBuffer = value;
    }

    public int SlidingWindow
    {
        get => _slidingWindow;
        set
        {
            if (value < 0 || value > _maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(value), value, $"Sliding window must be between 0 and {_maxSeqLen}.");
            _slidingWindow = value;
        }
    }

    public int EffectiveWindowSize => _slidingWindow > 0 ? _slidingWindow : _maxSeqLen;

    public static long GetSlotByteSize(int dim, KvCachePrecision precision)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        return precision switch
        {
            KvCachePrecision.Fp32 => (long)dim * sizeof(float),
            KvCachePrecision.Fp16 => (long)dim * sizeof(ushort),
            KvCachePrecision.Fp8 => dim,
            KvCachePrecision.Q8_0 => (long)((dim + 31) / 32) * sizeof(BlockQ8_0),
            KvCachePrecision.Q4_0 => (long)((dim + 31) / 32) * sizeof(BlockQ4_0),
            _ => throw new ArgumentOutOfRangeException(nameof(precision), precision, $"Unsupported KV cache precision: {precision}.")
        };
    }

    public KVCache(
        int layers,
        int nHeadsKv,
        int headDim,
        int maxSeqLen = 4096,
        int vHeadDim = -1,
        int slidingWindow = 0,
        bool isRingBuffer = false,
        KvCachePrecision precision = KvCachePrecision.Fp32)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeadsKv);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        if (slidingWindow < 0 || slidingWindow > maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(slidingWindow), slidingWindow, $"Sliding window must be between 0 and {maxSeqLen}.");

        _layers = layers;
        _nHeadsKv = nHeadsKv;
        _headDim = headDim;
        _vHeadDim = vHeadDim > 0 ? vHeadDim : headDim;
        _maxSeqLen = maxSeqLen;
        _slidingWindow = slidingWindow;
        _isRingBuffer = isRingBuffer;
        _precision = precision == KvCachePrecision.Auto ? KvCachePrecision.Fp32 : precision;

        checked
        {
            _slotBytesK = GetSlotByteSize(_headDim, _precision);
            _headByteStrideK = (long)_maxSeqLen * _slotBytesK;
            _layerByteStrideK = (long)_nHeadsKv * _headByteStrideK;
            _totalBytesK = (long)_layers * _layerByteStrideK;

            _slotBytesV = GetSlotByteSize(_vHeadDim, _precision);
            _headByteStrideV = (long)_maxSeqLen * _slotBytesV;
            _layerByteStrideV = (long)_nHeadsKv * _headByteStrideV;
            _totalBytesV = (long)_layers * _layerByteStrideV;

            if (sizeof(nuint) == 4 && (_totalBytesK > (long)uint.MaxValue || _totalBytesV > (long)uint.MaxValue))
            {
                throw new OutOfMemoryException("Requested KV cache size exceeds addressable memory limit.");
            }

            _kBuffer = (byte*)NativeMemory.AllocZeroed((nuint)_totalBytesK);
            _vBuffer = (byte*)NativeMemory.AllocZeroed((nuint)_totalBytesV);
        }
    }

    /// <summary>
    /// Translates a logical token sequence position to its physical buffer slot.
    /// In linear mode, positions beyond capacity throw. In ring/sliding-window mode, positions wrap modulo window size.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetPhysicalSlot(int pos)
    {
        if (pos < 0)
            throw new ArgumentOutOfRangeException(nameof(pos), pos, "Position must be non-negative.");

        if (_isRingBuffer || _slidingWindow > 0)
        {
            return pos % EffectiveWindowSize;
        }

        if ((uint)pos >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position must be between 0 and {_maxSeqLen - 1}.");

        return pos;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int layer, int headKv, int pos)
    {
        if (_disposed)
            ThrowDisposed();
        if (_precision != KvCachePrecision.Fp32)
            ThrowNonFp32PointerAccess();
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerByteStrideK + (long)headKv * _headByteStrideK + (long)slot * _slotBytesK;
        return (float*)(_kBuffer + offset);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int layer, int headKv, int pos)
    {
        if (_disposed)
            ThrowDisposed();
        if (_precision != KvCachePrecision.Fp32)
            ThrowNonFp32PointerAccess();
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerByteStrideV + (long)headKv * _headByteStrideV + (long)slot * _slotBytesV;
        return (float*)(_vBuffer + offset);
    }

    [DoesNotReturn]
    private static void ThrowDisposed() => throw new ObjectDisposedException(nameof(KVCache));

    [DoesNotReturn]
    private static void ThrowNonFp32PointerAccess() => throw new InvalidOperationException(
        "Direct float* pointer access requires KvCachePrecision.Fp32. Use ReadKey/ReadValue/DotKey/AccumulateValue for quantized KV caches.");

    [DoesNotReturn]
    private void ThrowOutOfRange(int layer, int headKv, int pos)
    {
        if ((uint)layer >= (uint)_layers)
            throw new ArgumentOutOfRangeException(nameof(layer), layer, $"Layer must be between 0 and {_layers - 1}.");
        if ((uint)headKv >= (uint)_nHeadsKv)
            throw new ArgumentOutOfRangeException(nameof(headKv), headKv, $"HeadKv must be between 0 and {_nHeadsKv - 1}.");
        throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position must be between 0 and {_maxSeqLen - 1}.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private byte* GetKeySlotPtr(int layer, int headKv, int pos)
    {
        if (_disposed)
            ThrowDisposed();
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerByteStrideK + (long)headKv * _headByteStrideK + (long)slot * _slotBytesK;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private byte* GetValueSlotPtr(int layer, int headKv, int pos)
    {
        if (_disposed)
            ThrowDisposed();
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerByteStrideV + (long)headKv * _headByteStrideV + (long)slot * _slotBytesV;
        return _vBuffer + offset;
    }

    /// <summary>
    /// Stores the projected Key and Value for a specific layer and token position.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Store(int layer, int pos, float* kSrc, float* vSrc)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)layer >= (uint)_layers)
            throw new ArgumentOutOfRangeException(nameof(layer), layer, $"Layer must be between 0 and {_layers - 1}.");

        int slot = GetPhysicalSlot(pos);
        long layerOffsetK = (long)layer * _layerByteStrideK + (long)slot * _slotBytesK;
        long layerOffsetV = (long)layer * _layerByteStrideV + (long)slot * _slotBytesV;

        for (int h = 0; h < _nHeadsKv; h++)
        {
            byte* kDst = _kBuffer + layerOffsetK + (long)h * _headByteStrideK;
            byte* vDst = _vBuffer + layerOffsetV + (long)h * _headByteStrideV;
            float* kHead = kSrc + h * _headDim;
            float* vHead = vSrc + h * _vHeadDim;

            switch (_precision)
            {
                case KvCachePrecision.Fp32:
                    Buffer.MemoryCopy(kHead, kDst, _slotBytesK, _slotBytesK);
                    Buffer.MemoryCopy(vHead, vDst, _slotBytesV, _slotBytesV);
                    break;
                case KvCachePrecision.Fp16:
                    QuantKernels.QuantizeF16(kHead, (Half*)kDst, _headDim);
                    QuantKernels.QuantizeF16(vHead, (Half*)vDst, _vHeadDim);
                    break;
                case KvCachePrecision.Fp8:
                    QuantKernels.QuantizeFp8E4M3(kHead, kDst, _headDim);
                    QuantKernels.QuantizeFp8E4M3(vHead, vDst, _vHeadDim);
                    break;
                case KvCachePrecision.Q4_0:
                    QuantKernels.QuantizeQ4_0(kHead, (BlockQ4_0*)kDst, _headDim);
                    QuantKernels.QuantizeQ4_0(vHead, (BlockQ4_0*)vDst, _vHeadDim);
                    break;
                case KvCachePrecision.Q8_0:
                    QuantKernels.QuantizeQ8_0(kHead, (BlockQ8_0*)kDst, _headDim);
                    QuantKernels.QuantizeQ8_0(vHead, (BlockQ8_0*)vDst, _vHeadDim);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
            }
        }
    }

    public void Store(int layer, int pos, ReadOnlySpan<float> kSrc, ReadOnlySpan<float> vSrc)
    {
        if (kSrc.Length < _nHeadsKv * _headDim)
            throw new ArgumentException($"Key source length {kSrc.Length} is smaller than expected {_nHeadsKv * _headDim}.", nameof(kSrc));
        if (vSrc.Length < _nHeadsKv * _vHeadDim)
            throw new ArgumentException($"Value source length {vSrc.Length} is smaller than expected {_nHeadsKv * _vHeadDim}.", nameof(vSrc));

        fixed (float* pK = kSrc)
        fixed (float* pV = vSrc)
        {
            Store(layer, pos, pK, pV);
        }
    }

    /// <summary>
    /// Dequantizes and copies a cached Key head vector into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadKey(int layer, int headKv, int pos, float* dst)
    {
        byte* src = GetKeySlotPtr(layer, headKv, pos);
        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                Buffer.MemoryCopy(src, dst, _slotBytesK, _slotBytesK);
                break;
            case KvCachePrecision.Fp16:
                QuantKernels.DequantizeF16((Half*)src, dst, _headDim);
                break;
            case KvCachePrecision.Fp8:
                QuantKernels.DequantizeFp8E4M3(src, dst, _headDim);
                break;
            case KvCachePrecision.Q4_0:
                QuantKernels.DequantizeQ4_0((BlockQ4_0*)src, dst, _headDim);
                break;
            case KvCachePrecision.Q8_0:
                QuantKernels.DequantizeQ8_0((BlockQ8_0*)src, dst, _headDim);
                break;
            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    public void ReadKey(int layer, int headKv, int pos, Span<float> dst)
    {
        if (dst.Length < _headDim)
            throw new ArgumentException($"Destination length {dst.Length} is smaller than HeadDim {_headDim}.", nameof(dst));

        fixed (float* pDst = dst)
        {
            ReadKey(layer, headKv, pos, pDst);
        }
    }

    /// <summary>
    /// Dequantizes and copies a cached Value head vector into <paramref name="dst"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void ReadValue(int layer, int headKv, int pos, float* dst)
    {
        byte* src = GetValueSlotPtr(layer, headKv, pos);
        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                Buffer.MemoryCopy(src, dst, _slotBytesV, _slotBytesV);
                break;
            case KvCachePrecision.Fp16:
                QuantKernels.DequantizeF16((Half*)src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Fp8:
                QuantKernels.DequantizeFp8E4M3(src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Q4_0:
                QuantKernels.DequantizeQ4_0((BlockQ4_0*)src, dst, _vHeadDim);
                break;
            case KvCachePrecision.Q8_0:
                QuantKernels.DequantizeQ8_0((BlockQ8_0*)src, dst, _vHeadDim);
                break;
            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    public void ReadValue(int layer, int headKv, int pos, Span<float> dst)
    {
        if (dst.Length < _vHeadDim)
            throw new ArgumentException($"Destination length {dst.Length} is smaller than ValueHeadDim {_vHeadDim}.", nameof(dst));

        fixed (float* pDst = dst)
        {
            ReadValue(layer, headKv, pos, pDst);
        }
    }

    /// <summary>
    /// Computes the dot product between <paramref name="query"/> and the cached Key vector at (<paramref name="layer"/>, <paramref name="headKv"/>, <paramref name="pos"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public float DotKey(int layer, int headKv, int pos, float* query)
    {
        byte* src = GetKeySlotPtr(layer, headKv, pos);
        return _precision switch
        {
            KvCachePrecision.Fp32 => QuantKernels.VecDotF32(query, (float*)src, _headDim),
            KvCachePrecision.Fp16 => QuantKernels.VecDotF16((Half*)src, query, _headDim),
            KvCachePrecision.Fp8 => QuantKernels.VecDotFp8E4M3(src, query, _headDim),
            KvCachePrecision.Q4_0 => QuantKernels.VecDotQ4_0Any((BlockQ4_0*)src, query, _headDim),
            KvCachePrecision.Q8_0 => QuantKernels.VecDotQ8_0Any((BlockQ8_0*)src, query, _headDim),
            _ => throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.")
        };
    }

    /// <summary>
    /// Accumulates <c>outHead[d] += weight * vPast[d]</c> directly from the cached Value vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public void AccumulateValue(int layer, int headKv, int pos, float weight, float* outHead)
    {
        if (weight == 0f) return;

        byte* src = GetValueSlotPtr(layer, headKv, pos);
        switch (_precision)
        {
            case KvCachePrecision.Fp32:
                float* vPast = (float*)src;
                if (Vector512.IsHardwareAccelerated && _vHeadDim >= 16)
                {
                    var vw = Vector512.Create(weight);
                    int d = 0;
                    for (; d <= _vHeadDim - 16; d += 16)
                    {
                        var vo = Vector512.Load(outHead + d);
                        var vv = Vector512.Load(vPast + d);
                        Vector512.FusedMultiplyAdd(vw, vv, vo).Store(outHead + d);
                    }
                    for (; d < _vHeadDim; d++) outHead[d] += weight * vPast[d];
                }
                else if (Vector256.IsHardwareAccelerated && _vHeadDim >= 8)
                {
                    var vw = Vector256.Create(weight);
                    int d = 0;
                    for (; d <= _vHeadDim - 8; d += 8)
                    {
                        var vo = Vector256.Load(outHead + d);
                        var vv = Vector256.Load(vPast + d);
                        (vo + vw * vv).Store(outHead + d);
                    }
                    for (; d < _vHeadDim; d++) outHead[d] += weight * vPast[d];
                }
                else
                {
                    for (int d = 0; d < _vHeadDim; d++) outHead[d] += weight * vPast[d];
                }
                break;
            case KvCachePrecision.Fp16:
                QuantKernels.AccumulateScaledF16((Half*)src, weight, outHead, _vHeadDim);
                break;
            case KvCachePrecision.Fp8:
                QuantKernels.AccumulateScaledFp8E4M3(src, weight, outHead, _vHeadDim);
                break;
            case KvCachePrecision.Q4_0:
                QuantKernels.AccumulateScaledQ4_0((BlockQ4_0*)src, weight, outHead, _vHeadDim);
                break;
            case KvCachePrecision.Q8_0:
                QuantKernels.AccumulateScaledQ8_0((BlockQ8_0*)src, weight, outHead, _vHeadDim);
                break;
            default:
                throw new NotSupportedException($"Unsupported KV cache precision: {_precision}.");
        }
    }

    /// <summary>
    /// Clears the KV cache by resetting buffer memories to zero.
    /// </summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMemory.Clear(_kBuffer, (nuint)_totalBytesK);
        NativeMemory.Clear(_vBuffer, (nuint)_totalBytesV);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_kBuffer != null)
            {
                NativeMemory.Free(_kBuffer);
                _kBuffer = null;
            }
            if (_vBuffer != null)
            {
                NativeMemory.Free(_vBuffer);
                _vBuffer = null;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~KVCache()
    {
        Dispose();
    }
}
