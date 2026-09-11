namespace Glacier.Inference.Memory;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Contiguous unmanaged Key-Value Cache for autoregressive Transformer inference.
/// Zero-allocation ring/linear buffer supporting Grouped Query Attention (GQA).
/// </summary>
public sealed unsafe class KVCache : IDisposable
{
    private readonly int _layers;
    private readonly int _nHeadsKv;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly long _layerStride;
    private readonly long _headStride;

    private float* _kBuffer;
    private float* _vBuffer;
    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int Layers => _layers;
    public int HeadsKv => _nHeadsKv;
    public int HeadDim => _headDim;

    public KVCache(int layers, int nHeadsKv, int headDim, int maxSeqLen = 4096)
    {
        _layers = layers;
        _nHeadsKv = nHeadsKv;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;

        _headStride = (long)_maxSeqLen * _headDim;
        _layerStride = (long)_nHeadsKv * _headStride;
        long totalElements = (long)_layers * _layerStride;
        long totalBytes = totalElements * sizeof(float);

        _kBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytes);
        _vBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytes);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int layer, int headKv, int pos)
    {
        long offset = (long)layer * _layerStride + (long)headKv * _headStride + (long)pos * _headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int layer, int headKv, int pos)
    {
        long offset = (long)layer * _layerStride + (long)headKv * _headStride + (long)pos * _headDim;
        return _vBuffer + offset;
    }

    /// <summary>
    /// Stores the projected Key and Value for a specific layer and token position.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Store(int layer, int pos, float* kSrc, float* vSrc)
    {
        for (int h = 0; h < _nHeadsKv; h++)
        {
            float* kDst = GetKeyPtr(layer, h, pos);
            float* vDst = GetValuePtr(layer, h, pos);

            Buffer.MemoryCopy(kSrc + h * _headDim, kDst, _headDim * sizeof(float), _headDim * sizeof(float));
            Buffer.MemoryCopy(vSrc + h * _headDim, vDst, _headDim * sizeof(float), _headDim * sizeof(float));
        }
    }

    /// <summary>
    /// Clears the KV cache by resetting buffer memories to zero.
    /// </summary>
    public void Reset()
    {
        long totalElements = (long)_layers * _layerStride;
        long totalBytes = totalElements * sizeof(float);
        NativeMemory.Clear(_kBuffer, (nuint)totalBytes);
        NativeMemory.Clear(_vBuffer, (nuint)totalBytes);
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
