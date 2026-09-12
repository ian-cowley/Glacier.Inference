namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Runtime.InteropServices;
using System.Threading;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D12;
using Vortice.DXGI;

/// <summary>
/// Low-overhead Direct3D 12 Compute Context.
/// Manages device creation, compute command queue, command list execution, and memory synchronization.
/// Optimized for AMD Radeon (Wave32 RDNA) and DirectX 12 hardware.
/// </summary>
public sealed unsafe class D3D12Context : IDisposable
{
    private readonly int _adapterIndex;
    private IDXGIFactory4 _factory = null!;
    private IDXGIAdapter1 _adapter = null!;
    private ID3D12Device _device = null!;
    private ID3D12CommandQueue _queue = null!;
    private ID3D12CommandAllocator _cmdAlloc = null!;
    private ID3D12GraphicsCommandList _cmdList = null!;
    private ID3D12Fence _fence = null!;
    private ulong _fenceValue;
    private AutoResetEvent _fenceEvent = null!;

    private ID3D12Resource _dummyBuffer = null!;
    private string _deviceName = string.Empty;
    private bool _disposed;

    public ID3D12Device Device => _device;
    public ID3D12CommandQueue Queue => _queue;
    public ID3D12GraphicsCommandList CommandList => _cmdList;
    public string DeviceName => _deviceName;
    public ID3D12Resource DummyBuffer => _dummyBuffer;

    public D3D12Context(int adapterIndex = -1)
    {
        _adapterIndex = adapterIndex;
        Initialize();
    }

    private void Initialize()
    {
        if (!OperatingSystem.IsWindows())
            throw new PlatformNotSupportedException("Direct3D 12 Compute is only supported on Windows.");

        _factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();

        IDXGIAdapter1? chosenAdapter = null;
        for (uint i = 0; _factory.EnumAdapters1(i, out IDXGIAdapter1 a).Success; i++)
        {
            var desc = a.Description1;
            if ((desc.Flags & AdapterFlags.Software) != 0)
            {
                a.Dispose();
                continue;
            }

            if (_adapterIndex >= 0)
            {
                if ((int)i == _adapterIndex)
                {
                    chosenAdapter = a;
                    break;
                }
                a.Dispose();
                continue;
            }

            // Prefer AMD Radeon adapter, otherwise first hardware adapter
            if (desc.Description.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
            {
                chosenAdapter = a;
                break;
            }

            if (chosenAdapter == null)
            {
                chosenAdapter = a;
            }
            else
            {
                a.Dispose();
            }
        }

        if (chosenAdapter == null)
            throw new InvalidOperationException("No hardware DirectX 12 compute adapters found.");

        _adapter = chosenAdapter;
        _deviceName = _adapter.Description1.Description;

        var hr = D3D12.D3D12CreateDevice(_adapter, FeatureLevel.Level_11_0, out _device!);
        if (!hr.Success || _device == null)
            throw new InvalidOperationException($"Failed to create Direct3D 12 device on {_deviceName}: {hr}");

        var queueDesc = new CommandQueueDescription(CommandListType.Compute);
        _queue = _device.CreateCommandQueue(queueDesc);

        _cmdAlloc = _device.CreateCommandAllocator(CommandListType.Compute);
        _cmdList = _device.CreateCommandList<ID3D12GraphicsCommandList>(0, CommandListType.Compute, _cmdAlloc);
        _cmdList.Close();

        _fence = _device.CreateFence(0);
        _fenceValue = 0;
        _fenceEvent = new AutoResetEvent(false);

        _dummyBuffer = CreateDeviceBuffer(256);
    }

    public ID3D12Resource CreateDeviceBuffer(ulong sizeInBytes, ResourceFlags flags = ResourceFlags.AllowUnorderedAccess)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var desc = ResourceDescription.Buffer(sizeInBytes, flags);
        var heapProps = new HeapProperties(HeapType.Default);
        return _device.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.Common);
    }

    public ID3D12Resource CreateUploadBuffer(ulong sizeInBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var desc = ResourceDescription.Buffer(sizeInBytes);
        var heapProps = new HeapProperties(HeapType.Upload);
        return _device.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.GenericRead);
    }

    public ID3D12Resource CreateReadbackBuffer(ulong sizeInBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var desc = ResourceDescription.Buffer(sizeInBytes);
        var heapProps = new HeapProperties(HeapType.Readback);
        return _device.CreateCommittedResource(heapProps, HeapFlags.None, desc, ResourceStates.CopyDest);
    }

    public void CopyToDevice(ID3D12Resource deviceBuffer, ReadOnlySpan<byte> hostData)
    {
        fixed (byte* pHost = hostData)
        {
            CopyToDevice(deviceBuffer, (IntPtr)pHost, (ulong)hostData.Length);
        }
    }

    public void CopyToDevice(ID3D12Resource deviceBuffer, IntPtr pHostData, ulong sizeInBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (sizeInBytes == 0) return;

        // Use upload staging buffer
        using var uploadBuffer = CreateUploadBuffer(sizeInBytes);
        void* pUpload = null;
        uploadBuffer.Map(0, null, &pUpload);
        Buffer.MemoryCopy((void*)pHostData, pUpload, sizeInBytes, sizeInBytes);
        uploadBuffer.Unmap(0);

        _cmdAlloc.Reset();
        _cmdList.Reset(_cmdAlloc, null);

        _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.Common, ResourceStates.CopyDest);
        _cmdList.CopyBufferRegion(deviceBuffer, 0, uploadBuffer, 0, sizeInBytes);
        _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.CopyDest, ResourceStates.Common);

        _cmdList.Close();
        _queue.ExecuteCommandList(_cmdList);
        Synchronize();
    }

    public void CopyToHost(Span<byte> hostData, ID3D12Resource deviceBuffer, ulong sizeInBytes)
    {
        fixed (byte* pHost = hostData)
        {
            CopyToHost((IntPtr)pHost, deviceBuffer, sizeInBytes);
        }
    }

    public void CopyToHost(IntPtr pHostData, ID3D12Resource deviceBuffer, ulong sizeInBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (sizeInBytes == 0) return;

        using var readbackBuffer = CreateReadbackBuffer(sizeInBytes);

        _cmdAlloc.Reset();
        _cmdList.Reset(_cmdAlloc, null);

        _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.Common, ResourceStates.CopySource);
        _cmdList.CopyBufferRegion(readbackBuffer, 0, deviceBuffer, 0, sizeInBytes);
        _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.CopySource, ResourceStates.Common);

        _cmdList.Close();
        _queue.ExecuteCommandList(_cmdList);
        Synchronize();

        void* pRead = null;
        var mapRes = readbackBuffer.Map(0, null, &pRead);
        if (pRead == null || !mapRes.Success)
        {
            var reason = _device.DeviceRemovedReason;
            throw new InvalidOperationException($"readbackBuffer.Map failed: {mapRes}, DeviceRemovedReason: {reason}");
        }
        Buffer.MemoryCopy(pRead, (void*)pHostData, sizeInBytes, sizeInBytes);
        readbackBuffer.Unmap(0);
    }

    public void Synchronize()
    {
        if (_disposed) return;
        _fenceValue++;
        _queue.Signal(_fence, _fenceValue);

        if (_fence.CompletedValue < _fenceValue)
        {
            _fence.SetEventOnCompletion(_fenceValue, _fenceEvent);
            _fenceEvent.WaitOne();
        }
    }

    public void UavBarrier(ID3D12Resource? resource = null)
    {
        _cmdList.ResourceBarrierUnorderedAccessView(resource!);
    }

    public ReadOnlyMemory<byte> CompileShader(string hlsl, string entryPoint = "main", string profile = "cs_5_0")
    {
        var bytecode = Compiler.Compile(hlsl, entryPoint, "source.hlsl", profile);
        if (bytecode.IsEmpty)
            throw new InvalidOperationException($"Failed to compile HLSL compute shader for entry point '{entryPoint}'.");
        return bytecode;
    }

    public ID3D12RootSignature CreateRootSignature(RootSignatureDescription desc)
    {
        return _device.CreateRootSignature(desc, RootSignatureVersion.Version1);
    }

    public ID3D12PipelineState CreatePipelineState(ID3D12RootSignature rootSig, ReadOnlyMemory<byte> shaderBytecode)
    {
        var psoDesc = new ComputePipelineStateDescription
        {
            RootSignature = rootSig,
            ComputeShader = shaderBytecode
        };
        return _device.CreateComputePipelineState(psoDesc);
    }

    public void BeginCommands()
    {
        _cmdAlloc.Reset();
        _cmdList.Reset(_cmdAlloc, null);
    }

    public void EndCommandsAndExecute()
    {
        _cmdList.Close();
        _queue.ExecuteCommandList(_cmdList);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            Synchronize();

            _dummyBuffer?.Dispose();
            _fenceEvent?.Dispose();
            _fence?.Dispose();
            _cmdList?.Dispose();
            _cmdAlloc?.Dispose();
            _queue?.Dispose();
            _device?.Dispose();
            _adapter?.Dispose();
            _factory?.Dispose();
        }
    }
}
