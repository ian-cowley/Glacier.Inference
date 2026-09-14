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
    private readonly int _vHeadDim;
    private readonly int _maxSeqLen;
    private readonly long _layerStrideK;
    private readonly long _layerStrideV;
    private readonly long _headStrideK;
    private readonly long _headStrideV;

    private float* _kBuffer;
    private float* _vBuffer;
    private bool _disposed;

    public int MaxSeqLen => _maxSeqLen;
    public int Layers => _layers;
    public int HeadsKv => _nHeadsKv;
    public int HeadDim => _headDim;
    public int ValueHeadDim => _vHeadDim;

    public KVCache(int layers, int nHeadsKv, int headDim, int maxSeqLen = 4096, int vHeadDim = -1)
    {
        _layers = layers;
        _nHeadsKv = nHeadsKv;
        _headDim = headDim;
        _vHeadDim = vHeadDim > 0 ? vHeadDim : headDim;
        _maxSeqLen = maxSeqLen;

        _headStrideK = (long)_maxSeqLen * _headDim;
        _layerStrideK = (long)_nHeadsKv * _headStrideK;
        long totalElementsK = (long)_layers * _layerStrideK;
        long totalBytesK = totalElementsK * sizeof(float);

        _headStrideV = (long)_maxSeqLen * _vHeadDim;
        _layerStrideV = (long)_nHeadsKv * _headStrideV;
        long totalElementsV = (long)_layers * _layerStrideV;
        long totalBytesV = totalElementsV * sizeof(float);

        _kBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesK);
        _vBuffer = (float*)NativeMemory.AllocZeroed((nuint)totalBytesV);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int layer, int headKv, int pos)
    {
        long offset = (long)layer * _layerStrideK + (long)headKv * _headStrideK + (long)pos * _headDim;
        return _kBuffer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int layer, int headKv, int pos)
    {
        long offset = (long)layer * _layerStrideV + (long)headKv * _headStrideV + (long)pos * _vHeadDim;
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
            Buffer.MemoryCopy(vSrc + h * _vHeadDim, vDst, _vHeadDim * sizeof(float), _vHeadDim * sizeof(float));
        }
    }

    /// <summary>
    /// Clears the KV cache by resetting buffer memories to zero.
    /// </summary>
    public void Reset()
    {
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
