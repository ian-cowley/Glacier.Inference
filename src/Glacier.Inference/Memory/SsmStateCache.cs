namespace Glacier.Inference.Memory;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// High-performance zero-allocation unmanaged recurrent state cache for Gated DeltaNet / State Space Models.
/// Maintains rolling 1D-convolution buffers and recurrent state matrices (S_t) across autoregressive steps.
/// </summary>
public sealed unsafe class SsmStateCache : IDisposable
{
    private readonly int _layerCount;
    private readonly int _convKernel;
    private readonly int _convChannels;
    private readonly int _heads;
    private readonly int _stateDim; // 128
    private readonly long _layerConvBytes;
    private readonly long _layerRecurrentBytes;
    private readonly long _totalBytes;

    private byte* _buffer;
    private bool _disposed;

    public int LayerCount => _layerCount;
    public int ConvKernel => _convKernel;
    public int ConvChannels => _convChannels;
    public int Heads => _heads;
    public int StateDim => _stateDim;
    public long TotalBytes => _totalBytes;

    public SsmStateCache(int layerCount, int convKernel, int convChannels, int heads, int stateDim)
    {
        _layerCount = layerCount;
        _convKernel = convKernel;
        _convChannels = convChannels;
        _heads = heads;
        _stateDim = stateDim;

        // Conv state: (convKernel - 1) history tokens per channel
        int convHistory = Math.Max(1, convKernel - 1);
        _layerConvBytes = (long)convHistory * convChannels * sizeof(float);

        // Recurrent state: heads x stateDim x stateDim (128x128 matrix per head)
        _layerRecurrentBytes = (long)heads * stateDim * stateDim * sizeof(float);

        long perLayerTotal = _layerConvBytes + _layerRecurrentBytes;
        _totalBytes = perLayerTotal * layerCount;

        // Allocate aligned unmanaged memory
        _buffer = (byte*)NativeMemory.AllocZeroed((nuint)_totalBytes);
    }

    /// <summary>
    /// Gets a pointer to the 1D convolution rolling history buffer for the given layer.
    /// Shape: [(convKernel - 1) x convChannels]
    /// </summary>
    public float* GetConvState(int layerIndex)
    {
        long perLayerTotal = _layerConvBytes + _layerRecurrentBytes;
        byte* layerPtr = _buffer + (layerIndex * perLayerTotal);
        return (float*)layerPtr;
    }

    /// <summary>
    /// Gets a pointer to the recurrent state matrix S for the given layer and head.
    /// Shape: [stateDim x stateDim] (e.g. 128x128 floats)
    /// </summary>
    public float* GetRecurrentState(int layerIndex, int headIndex)
    {
        long perLayerTotal = _layerConvBytes + _layerRecurrentBytes;
        byte* layerPtr = _buffer + (layerIndex * perLayerTotal) + _layerConvBytes;
        long headOffset = (long)headIndex * _stateDim * _stateDim * sizeof(float);
        return (float*)(layerPtr + headOffset);
    }

    /// <summary>
    /// Clears all historical states for a fresh prompt/conversation.
    /// </summary>
    public void Reset()
    {
        if (_buffer != null && !_disposed)
        {
            NativeMemory.Clear(_buffer, (nuint)_totalBytes);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_buffer != null)
            {
                NativeMemory.Free(_buffer);
                _buffer = null;
            }
            _disposed = true;
        }
    }
}
