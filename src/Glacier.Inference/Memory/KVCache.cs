namespace Glacier.Inference.Memory;

using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Contiguous unmanaged Key-Value Cache for autoregressive Transformer inference.
/// Zero-allocation ring/linear buffer supporting Grouped Query Attention (GQA).
/// Instances are NOT thread-safe; single-thread access per generation stream is required.
/// </summary>
public sealed unsafe class KVCache : IDisposable
{
    private readonly int _layers;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly int _vHeadDim;
    private readonly int _maxSeqLen;
    private readonly long _layerStrideK;
    private readonly long _layerStrideV;
    private readonly long _headStrideK;
    private readonly long _headStrideV;

    private int _slidingWindow;
    private bool _isRingBuffer;
    private float* _kBuffer;
    private float* _vBuffer;
    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int Layers => _layers;
    public int HeadsKv => _nHeadsKv;
    public int HeadDim => _headDim;
    public int ValueHeadDim => _vHeadDim;

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

    public KVCache(
        int layers,
        int nHeadsKv,
        int headDim,
        int maxSeqLen = 4096,
        int vHeadDim = -1,
        int slidingWindow = 0,
        bool isRingBuffer = false)
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

        checked
        {
            _headStrideK = (long)_maxSeqLen * _headDim;
            _layerStrideK = (long)_nHeadsKv * _headStrideK;
            long totalElementsK = (long)_layers * _layerStrideK;
            long totalBytesK = totalElementsK * sizeof(float);

            _headStrideV = (long)_maxSeqLen * _vHeadDim;
            _layerStrideV = (long)_nHeadsKv * _headStrideV;
            long totalElementsV = (long)_layers * _layerStrideV;
            long totalBytesV = totalElementsV * sizeof(float);

            if (sizeof(nuint) == 4 && (totalBytesK > (long)uint.MaxValue || totalBytesV > (long)uint.MaxValue))
            {
                throw new OutOfMemoryException("Requested KV cache size exceeds addressable memory limit.");
            }

            _kBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesK);
            _vBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesV);
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
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerStrideK + (long)headKv * _headStrideK + (long)slot * _headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int layer, int headKv, int pos)
    {
        if (_disposed)
            ThrowDisposed();
        if ((uint)layer >= (uint)_layers || (uint)headKv >= (uint)_nHeadsKv)
            ThrowOutOfRange(layer, headKv, pos);

        int slot = GetPhysicalSlot(pos);
        long offset = (long)layer * _layerStrideV + (long)headKv * _headStrideV + (long)slot * _vHeadDim;
        return _vBuffer + offset;
    }

    [DoesNotReturn]
    private static void ThrowDisposed() => throw new ObjectDisposedException(nameof(KVCache));

    [DoesNotReturn]
    private void ThrowOutOfRange(int layer, int headKv, int pos)
    {
        if ((uint)layer >= (uint)_layers)
            throw new ArgumentOutOfRangeException(nameof(layer), layer, $"Layer must be between 0 and {_layers - 1}.");
        if ((uint)headKv >= (uint)_nHeadsKv)
            throw new ArgumentOutOfRangeException(nameof(headKv), headKv, $"HeadKv must be between 0 and {_nHeadsKv - 1}.");
        throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position must be between 0 and {_maxSeqLen - 1}.");
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

        for (int h = 0; h < _nHeadsKv; h++)
        {
            float* kDst = _kBuffer + (long)layer * _layerStrideK + (long)h * _headStrideK + (long)slot * _headDim;
            float* vDst = _vBuffer + (long)layer * _layerStrideV + (long)h * _headStrideV + (long)slot * _vHeadDim;

            Buffer.MemoryCopy(kSrc + h * _headDim, kDst, _headDim * sizeof(float), _headDim * sizeof(float));
            Buffer.MemoryCopy(vSrc + h * _vHeadDim, vDst, _vHeadDim * sizeof(float), _vHeadDim * sizeof(float));
        }
    }

    /// <summary>
    /// Clears the KV cache by resetting buffer memories to zero.
    /// </summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long totalBytesK = (long)_layers * _layerStrideK * sizeof(float);
        long totalBytesV = (long)_layers * _layerStrideV * sizeof(float);
        NativeMemory.Clear(_kBuffer, (nuint)totalBytesK);
        NativeMemory.Clear(_vBuffer, (nuint)totalBytesV);
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
