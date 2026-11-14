namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Inference.Gguf;
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
    private ID3D12Resource _stagingUploadBuffer = null!;
    private const ulong StagingUploadBufferSize = 32 * 1024 * 1024; // 32 MB reusable staging buffer
    private string _deviceName = string.Empty;
    private bool _disposed;

    public ID3D12Device Device => _device;
    public ID3D12CommandQueue Queue => _queue;
    public ID3D12GraphicsCommandList CommandList => _cmdList;
    public string DeviceName => _deviceName;
    public ID3D12Resource DummyBuffer => _dummyBuffer;
    private ulong _dedicatedVideoMemory;
    private ulong _sharedSystemMemory;
    public ulong DedicatedVideoMemory => _dedicatedVideoMemory;
    public ulong SharedSystemMemory => _sharedSystemMemory;
    public ulong TotalAccessibleMemory => (_dedicatedVideoMemory < 4_000_000_000UL || _deviceName.Contains("Radeon", StringComparison.OrdinalIgnoreCase) || _deviceName.Contains("Intel", StringComparison.OrdinalIgnoreCase))
        ? (_dedicatedVideoMemory + _sharedSystemMemory)
        : _dedicatedVideoMemory;
    public IDXGIAdapter1? Adapter => _adapter;

    /// <summary>
    /// Returns true if Direct3D 12 compute is supported and at least one hardware adapter is present.
    /// </summary>
    public static bool IsSupported
    {
        get
        {
            try
            {
                return OperatingSystem.IsWindows() && EnumerateAdapters().Count > 0;
            }
            catch
            {
                return false;
            }
        }
    }

    public D3D12Context(int adapterIndex = -1)
    {
        _adapterIndex = adapterIndex;
        Initialize();
    }

    public static List<(int Index, string Name, ulong DedicatedVideoMemory, ulong SharedSystemMemory)> EnumerateAdapters()
    {
        var list = new List<(int Index, string Name, ulong DedicatedVideoMemory, ulong SharedSystemMemory)>();
        if (!OperatingSystem.IsWindows()) return list;

        using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory4>();
        for (uint i = 0; factory.EnumAdapters1(i, out IDXGIAdapter1 a).Success; i++)
        {
            var desc = a.Description1;
            if ((desc.Flags & AdapterFlags.Software) == 0)
            {
                list.Add(((int)i, desc.Description, (ulong)desc.DedicatedVideoMemory, (ulong)desc.SharedSystemMemory));
            }
            a.Dispose();
        }
        return list;
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

            // If adapter not explicitly specified, pick hardware adapter with highest dedicated video memory
            if (chosenAdapter == null)
            {
                chosenAdapter = a;
            }
            else
            {
                if ((ulong)desc.DedicatedVideoMemory > (ulong)chosenAdapter.Description1.DedicatedVideoMemory)
                {
                    chosenAdapter.Dispose();
                    chosenAdapter = a;
                }
                else
                {
                    a.Dispose();
                }
            }
        }

        if (chosenAdapter == null)
            throw new InvalidOperationException("No hardware DirectX 12 compute adapters found.");

        _adapter = chosenAdapter;
        var desc1 = _adapter.Description1;
        _deviceName = desc1.Description;
        _dedicatedVideoMemory = (ulong)desc1.DedicatedVideoMemory;
        _sharedSystemMemory = (ulong)desc1.SharedSystemMemory;

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

        _dummyBuffer = CreateDeviceBuffer(4 * 1024 * 1024);
        _stagingUploadBuffer = CreateUploadBuffer(StagingUploadBufferSize);

        BeginCommands();
        _cmdList.ResourceBarrierTransition(_dummyBuffer, ResourceStates.Common, ResourceStates.UnorderedAccess);
        EndCommandsAndExecute();
        Synchronize();
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

        ulong offset = 0;
        byte* pSrc = (byte*)pHostData;

        while (offset < sizeInBytes)
        {
            ulong chunkSize = Math.Min(sizeInBytes - offset, StagingUploadBufferSize);

            void* pUpload = null;
            _stagingUploadBuffer.Map(0, null, &pUpload);
            Buffer.MemoryCopy(pSrc + offset, pUpload, chunkSize, chunkSize);
            _stagingUploadBuffer.Unmap(0);

            _cmdAlloc.Reset();
            _cmdList.Reset(_cmdAlloc, null);

            _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.Common, ResourceStates.CopyDest);
            _cmdList.CopyBufferRegion(deviceBuffer, offset, _stagingUploadBuffer, 0, chunkSize);
            _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.CopyDest, ResourceStates.Common);

            _cmdList.Close();
            _queue.ExecuteCommandList(_cmdList);
            Synchronize();

            offset += chunkSize;
        }
    }

    public void ZeroDeviceBuffer(ID3D12Resource deviceBuffer, ResourceStates currentState = ResourceStates.Common)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (deviceBuffer == null) return;

        ulong sizeInBytes = deviceBuffer.Description.Width;
        if (sizeInBytes == 0) return;

        ulong clearBytes = Math.Min(sizeInBytes, StagingUploadBufferSize);
        void* pUpload = null;
        _stagingUploadBuffer.Map(0, null, &pUpload);
        NativeMemory.Clear(pUpload, (nuint)clearBytes);
        _stagingUploadBuffer.Unmap(0);

        ulong offset = 0;
        while (offset < sizeInBytes)
        {
            ulong chunkSize = Math.Min(sizeInBytes - offset, StagingUploadBufferSize);

            _cmdAlloc.Reset();
            _cmdList.Reset(_cmdAlloc, null);

            if (currentState != ResourceStates.CopyDest)
            {
                _cmdList.ResourceBarrierTransition(deviceBuffer, currentState, ResourceStates.CopyDest);
            }
            _cmdList.CopyBufferRegion(deviceBuffer, offset, _stagingUploadBuffer, 0, chunkSize);
            if (currentState != ResourceStates.CopyDest)
            {
                _cmdList.ResourceBarrierTransition(deviceBuffer, ResourceStates.CopyDest, currentState);
            }

            _cmdList.Close();
            _queue.ExecuteCommandList(_cmdList);
            Synchronize();

            offset += chunkSize;
        }
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

    public void Synchronize(int timeoutMs = 300000)
    {
        if (_disposed) return;
        _fenceValue++;
        _queue.Signal(_fence, _fenceValue);

        if (_fence.CompletedValue < _fenceValue)
        {
            _fence.SetEventOnCompletion(_fenceValue, _fenceEvent);
            if (!_fenceEvent.WaitOne(timeoutMs))
            {
                var reason = _device.DeviceRemovedReason;
                throw new TimeoutException($"D3D12Context.Synchronize timed out after {timeoutMs}ms waiting for fence {_fenceValue} (Completed: {_fence.CompletedValue}). DeviceRemovedReason: 0x{reason.Code:X8} ({reason})");
            }
        }
    }

    public void UavBarrier(ID3D12Resource? resource = null)
    {
        _cmdList.ResourceBarrierUnorderedAccessView(resource!);
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, ReadOnlyMemory<byte>> _memoryShaderCache = new();
    private static readonly string _diskShaderCacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Glacier", "ShaderCache");

    public ReadOnlyMemory<byte> CompileShader(string hlsl, string entryPoint = "main", string profile = "cs_5_0")
    {
        string key = $"{profile}_{entryPoint}_{hlsl}";
        if (_memoryShaderCache.TryGetValue(key, out var cachedMem))
        {
            return cachedMem;
        }

        byte[] hashBytes = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(key));
        string hashStr = Convert.ToHexString(hashBytes).ToLowerInvariant();
        string cacheFile = Path.Combine(_diskShaderCacheDir, $"{hashStr}.cso");

        try
        {
            if (File.Exists(cacheFile))
            {
                byte[] diskBytes = File.ReadAllBytes(cacheFile);
                if (diskBytes.Length > 0)
                {
                    _memoryShaderCache[key] = diskBytes;
                    return diskBytes;
                }
            }
        }
        catch { }

        var bytecode = Compiler.Compile(hlsl, entryPoint, "source.hlsl", profile);
        if (bytecode.IsEmpty)
            throw new InvalidOperationException($"Failed to compile HLSL compute shader for entry point '{entryPoint}'.");

        try
        {
            Directory.CreateDirectory(_diskShaderCacheDir);
            File.WriteAllBytes(cacheFile, bytecode.ToArray());
        }
        catch { }

        _memoryShaderCache[key] = bytecode;
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

    private ID3D12RootSignature? _sigMoeRouterSoftmaxTopK;
    private ID3D12PipelineState? _psoMoeRouterSoftmaxTopK;

    private ID3D12RootSignature? _sigRepetitionPenalty;
    private ID3D12PipelineState? _psoRepetitionPenalty;
    private ID3D12Resource? _recentTokensUploadBuffer;

    private ID3D12RootSignature? _sigRmsNormHeads;
    private ID3D12PipelineState? _psoRmsNormHeads;

    private ID3D12RootSignature? _sigMoeSharedExpertGate;
    private ID3D12PipelineState? _psoMoeSharedExpertGate;

    private ID3D12RootSignature? _sigVecAddWeightedFromBuffer;
    private ID3D12PipelineState? _psoVecAddWeightedFromBuffer;

    private ID3D12RootSignature? _sigExpertGemv;
    private readonly Dictionary<GgufType, ID3D12PipelineState> _psoExpertGemvByType = new();

    private void EnsureMoePipelines()
    {
        if (_psoMoeRouterSoftmaxTopK == null)
        {
            var routerParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 4), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
            };
            _sigMoeRouterSoftmaxTopK = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, routerParams));
            _psoMoeRouterSoftmaxTopK = CreatePipelineState(_sigMoeRouterSoftmaxTopK, CompileShader(D3D12Shaders.MoeRouterSoftmaxTopK));
        }

        if (_psoMoeSharedExpertGate == null)
        {
            var gateParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 1), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
            };
            _sigMoeSharedExpertGate = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gateParams));
            _psoMoeSharedExpertGate = CreatePipelineState(_sigMoeSharedExpertGate, CompileShader(D3D12Shaders.MoeSharedExpertGate));
        }

        if (_psoVecAddWeightedFromBuffer == null)
        {
            var vecParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 3), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
            };
            _sigVecAddWeightedFromBuffer = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecParams));
            _psoVecAddWeightedFromBuffer = CreatePipelineState(_sigVecAddWeightedFromBuffer, CompileShader(D3D12Shaders.VecAddWeightedFromBuffer));
        }
    }

    public void DispatchMoeRouterSoftmaxTopK(
        ID3D12Resource dLogits,
        ID3D12Resource dTopKIndices,
        ID3D12Resource dTopKWeights,
        int numExperts,
        int topK,
        bool normTopK = true,
        int batchSize = 1)
    {
        EnsureMoePipelines();
        _cmdList.SetComputeRootSignature(_sigMoeRouterSoftmaxTopK!);
        _cmdList.SetPipelineState(_psoMoeRouterSoftmaxTopK!);

        uint* pConsts = stackalloc uint[4];
        pConsts[0] = (uint)numExperts;
        pConsts[1] = (uint)topK;
        pConsts[2] = normTopK ? 1u : 0u;
        pConsts[3] = (uint)batchSize;
        _cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

        _cmdList.SetComputeRootUnorderedAccessView(1, dLogits.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, dTopKIndices.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(3, dTopKWeights.GPUVirtualAddress);

        _cmdList.Dispatch((uint)batchSize, 1, 1);
    }

    public void DispatchMoeRouterSoftmaxTopK(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource dLogits,
        ID3D12Resource dTopKIndices,
        ID3D12Resource dTopKWeights,
        int numExperts,
        int topK,
        bool normTopK = true,
        int batchSize = 1)
    {
        EnsureMoePipelines();
        cmdList.SetComputeRootSignature(_sigMoeRouterSoftmaxTopK!);
        cmdList.SetPipelineState(_psoMoeRouterSoftmaxTopK!);

        uint* pConsts = stackalloc uint[4];
        pConsts[0] = (uint)numExperts;
        pConsts[1] = (uint)topK;
        pConsts[2] = normTopK ? 1u : 0u;
        pConsts[3] = (uint)batchSize;
        cmdList.SetComputeRoot32BitConstants(0, 4, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, dLogits.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, dTopKIndices.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dTopKWeights.GPUVirtualAddress);

        cmdList.Dispatch((uint)batchSize, 1, 1);
    }

    public void DispatchRmsNormHeads(
        ID3D12Resource dX,
        ID3D12Resource dWeight,
        int nHeads,
        int headDim,
        float eps)
    {
        if (nHeads <= 0 || headDim <= 0) return;

        if (_psoRmsNormHeads == null)
        {
            var rmsHeadsParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 3), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
            };
            _sigRmsNormHeads = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsHeadsParams));
            _psoRmsNormHeads = CreatePipelineState(_sigRmsNormHeads, CompileShader(D3D12Shaders.RmsNormHeads));
        }

        _cmdList.SetComputeRootSignature(_sigRmsNormHeads!);
        _cmdList.SetPipelineState(_psoRmsNormHeads!);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)headDim;
        pConsts[1] = (uint)nHeads;
        pConsts[2] = BitConverter.SingleToUInt32Bits(eps);
        _cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        _cmdList.SetComputeRootShaderResourceView(1, dWeight.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, dX.GPUVirtualAddress);

        _cmdList.Dispatch((uint)nHeads, 1, 1);
    }

    /// <summary>
    /// Deduplicates <paramref name="recentTokens"/> on the host and dispatches <see cref="D3D12Shaders.RepetitionPenalty"/>
    /// with vocabulary bounds validation on the GPU.
    /// </summary>
    public void DispatchRepetitionPenalty(
        ID3D12Resource dLogits,
        int vocabSize,
        ReadOnlySpan<int> recentTokens,
        float penalty)
    {
        if (penalty == 1.0f || recentTokens.IsEmpty || vocabSize <= 0) return;

        if (_psoRepetitionPenalty == null)
        {
            var repParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 3), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
            };
            _sigRepetitionPenalty = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, repParams));
            _psoRepetitionPenalty = CreatePipelineState(_sigRepetitionPenalty, CompileShader(D3D12Shaders.RepetitionPenalty));
        }

        const int maxPenaltyTokens = 1024;
        _recentTokensUploadBuffer ??= CreateUploadBuffer((ulong)(maxPenaltyTokens * sizeof(int)));

        int sliceStart = Math.Max(0, recentTokens.Length - maxPenaltyTokens);
        ReadOnlySpan<int> window = recentTokens.Slice(sliceStart);

        int* uniqueBuf = stackalloc int[maxPenaltyTokens];
        int count = 0;
        for (int i = 0; i < window.Length && count < maxPenaltyTokens; i++)
        {
            int tok = window[i];
            if ((uint)tok >= (uint)vocabSize) continue;
            bool seen = false;
            for (int j = 0; j < count; j++)
            {
                if (uniqueBuf[j] == tok)
                {
                    seen = true;
                    break;
                }
            }
            if (!seen)
            {
                uniqueBuf[count++] = tok;
            }
        }

        if (count == 0) return;

        void* pUpload = null;
        _recentTokensUploadBuffer.Map(0, null, &pUpload);
        Buffer.MemoryCopy(uniqueBuf, pUpload, (ulong)(maxPenaltyTokens * sizeof(int)), (ulong)(count * sizeof(int)));
        _recentTokensUploadBuffer.Unmap(0);

        _cmdList.SetComputeRootSignature(_sigRepetitionPenalty!);
        _cmdList.SetPipelineState(_psoRepetitionPenalty!);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)count;
        pConsts[1] = (uint)vocabSize;
        pConsts[2] = BitConverter.SingleToUInt32Bits(penalty);
        _cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        _cmdList.SetComputeRootShaderResourceView(1, _recentTokensUploadBuffer.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, dLogits.GPUVirtualAddress);

        uint groups = (uint)((count + 63) / 64);
        _cmdList.Dispatch(groups, 1, 1);
    }

    public void DispatchMoeSharedExpertGate(
        ID3D12Resource dX,
        ID3D12Resource dWeight,
        ID3D12Resource dShexpGateScalar,
        int size)
    {
        EnsureMoePipelines();
        _cmdList.SetComputeRootSignature(_sigMoeSharedExpertGate!);
        _cmdList.SetPipelineState(_psoMoeSharedExpertGate!);

        uint uSize = (uint)size;
        _cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&uSize), 0);

        _cmdList.SetComputeRootUnorderedAccessView(1, dWeight.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, dX.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(3, dShexpGateScalar.GPUVirtualAddress);

        _cmdList.Dispatch(1, 1, 1);
    }

    public void DispatchMoeSharedExpertGate(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource dX,
        ID3D12Resource dWeight,
        ID3D12Resource dShexpGateScalar,
        int size)
    {
        EnsureMoePipelines();
        cmdList.SetComputeRootSignature(_sigMoeSharedExpertGate!);
        cmdList.SetPipelineState(_psoMoeSharedExpertGate!);

        uint uSize = (uint)size;
        cmdList.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&uSize), 0);

        cmdList.SetComputeRootUnorderedAccessView(1, dWeight.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, dX.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dShexpGateScalar.GPUVirtualAddress);

        cmdList.Dispatch(1, 1, 1);
    }

    public void DispatchVecAddWeightedFromBuffer(
        ID3D12Resource dDst,
        ID3D12Resource dSrc,
        ID3D12Resource dWeightsBuffer,
        int weightIndex,
        int length,
        uint accumulate = 1)
    {
        EnsureMoePipelines();
        _cmdList.SetComputeRootSignature(_sigVecAddWeightedFromBuffer!);
        _cmdList.SetPipelineState(_psoVecAddWeightedFromBuffer!);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)length;
        pConsts[1] = accumulate;
        pConsts[2] = (uint)weightIndex;
        _cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        _cmdList.SetComputeRootUnorderedAccessView(1, dWeightsBuffer.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(2, dSrc.GPUVirtualAddress);
        _cmdList.SetComputeRootUnorderedAccessView(3, dDst.GPUVirtualAddress);

        _cmdList.Dispatch(((uint)length + 255) / 256, 1, 1);
    }

    public void DispatchVecAddWeightedFromBuffer(
        ID3D12GraphicsCommandList cmdList,
        ID3D12Resource dDst,
        ID3D12Resource dSrc,
        ID3D12Resource dWeightsBuffer,
        int weightIndex,
        int length,
        uint accumulate = 1)
    {
        EnsureMoePipelines();
        cmdList.SetComputeRootSignature(_sigVecAddWeightedFromBuffer!);
        cmdList.SetPipelineState(_psoVecAddWeightedFromBuffer!);

        uint* pConsts = stackalloc uint[3];
        pConsts[0] = (uint)length;
        pConsts[1] = accumulate;
        pConsts[2] = (uint)weightIndex;
        cmdList.SetComputeRoot32BitConstants(0, 3, (IntPtr)pConsts, 0);

        cmdList.SetComputeRootUnorderedAccessView(1, dWeightsBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(2, dSrc.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, dDst.GPUVirtualAddress);

        cmdList.Dispatch(((uint)length + 255) / 256, 1, 1);
    }

    public void EnsureExpertGemvPipelines(GgufType type)
    {
        if (_sigExpertGemv == null)
        {
            var gemvParams = new RootParameter[]
            {
                new(new RootConstants(0, 0, 8), ShaderVisibility.All),
                new(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
                new(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: x
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All), // u1: residual
                new(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All), // u2: y
                new(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All), // t2: topk_indices
            };
            _sigExpertGemv = CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
        }

        if (!_psoExpertGemvByType.ContainsKey(type))
        {
            string baseShader = type switch
            {
                GgufType.Q4_K => D3D12Shaders.GemvQ4K,
                GgufType.Q5_K => D3D12Shaders.GemvQ5K,
                GgufType.Q6_K => D3D12Shaders.GemvQ6K,
                GgufType.Q3_K => D3D12Shaders.GemvQ3K,
                GgufType.Q8_0 => D3D12Shaders.GemvQ8_0,
                GgufType.Q2_K => D3D12Shaders.GemvQ2_K,
                GgufType.Q2_0 => D3D12Shaders.GemvQ2_0,
                GgufType.IQ2_XXS => D3D12Shaders.GemvIQ2_XXS,
                GgufType.IQ4_NL => D3D12Shaders.GemvIQ4_NL,
                GgufType.IQ4_XS => D3D12Shaders.GemvIQ4_XS,
                GgufType.MXFP4 => D3D12Shaders.GemvMXFP4,
                GgufType.IQ3_S => D3D12Shaders.GemvIQ3_S,
                GgufType.Ternary_1_58 or GgufType.PQ2_0 => D3D12Shaders.GemvTernary,
                GgufType.F32 => D3D12Shaders.GemvFp32,
                _ => throw new NotSupportedException($"Expert GEMV does not support tensor quantization type {type}.")
            };
            string expertShader = D3D12Shaders.MakeExpertGemvShader(baseShader);
            _psoExpertGemvByType[type] = CreatePipelineState(_sigExpertGemv, CompileShader(expertShader));
        }
    }

    public void DispatchExpertGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ulong wGpuVirtualAddress,
        ID3D12Resource topkIndices,
        int kIndex,
        ulong sliceBytes,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
    {
        EnsureExpertGemvPipelines(type);
        var pso = _psoExpertGemvByType[type];

        cmdList.SetComputeRootSignature(_sigExpertGemv!);
        cmdList.SetPipelineState(pso);

        cmdList.SetComputeRootShaderResourceView(1, wGpuVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(2, bias?.GPUVirtualAddress ?? DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(4, residual?.GPUVirtualAddress ?? DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootUnorderedAccessView(5, y?.GPUVirtualAddress ?? DummyBuffer.GPUVirtualAddress);
        cmdList.SetComputeRootShaderResourceView(6, topkIndices.GPUVirtualAddress);

        const int chunkSize = D3D12TensorAlign.GemvChunkRows;
        uint* pConsts = stackalloc uint[8];
        pConsts[0] = (uint)k_cols;
        pConsts[1] = (uint)m_rows;
        pConsts[2] = bias != null ? 1u : 0u;
        pConsts[3] = residual != null ? 1u : 0u;
        pConsts[4] = y != null ? 1u : 0u;
        pConsts[6] = (uint)kIndex;
        pConsts[7] = (uint)sliceBytes;

        for (int r = 0; r < m_rows; r += chunkSize)
        {
            int currentChunk = Math.Min(m_rows - r, chunkSize);
            pConsts[5] = (uint)r;
            cmdList.SetComputeRoot32BitConstants(0, 8, (IntPtr)pConsts, 0);

            cmdList.Dispatch(((uint)currentChunk + 3) / 4, 1, 1);
        }
    }

    public void DispatchExpertGemv(
        ID3D12GraphicsCommandList cmdList,
        GgufType type,
        ID3D12Resource? y,
        ID3D12Resource x,
        ID3D12Resource W,
        ID3D12Resource topkIndices,
        int kIndex,
        ulong sliceBytes,
        int k_cols,
        int m_rows,
        ID3D12Resource? bias = null,
        ID3D12Resource? residual = null)
        => DispatchExpertGemv(cmdList, type, y, x, W.GPUVirtualAddress, topkIndices, kIndex, sliceBytes, k_cols, m_rows, bias, residual);

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            Synchronize();

            _sigExpertGemv?.Dispose();
            foreach (var pso in _psoExpertGemvByType.Values)
            {
                pso.Dispose();
            }
            _psoExpertGemvByType.Clear();

            _psoMoeRouterSoftmaxTopK?.Dispose();
            _sigMoeRouterSoftmaxTopK?.Dispose();
            _psoRepetitionPenalty?.Dispose();
            _sigRepetitionPenalty?.Dispose();
            _recentTokensUploadBuffer?.Dispose();
            _psoRmsNormHeads?.Dispose();
            _sigRmsNormHeads?.Dispose();
            _psoMoeSharedExpertGate?.Dispose();
            _sigMoeSharedExpertGate?.Dispose();
            _psoVecAddWeightedFromBuffer?.Dispose();
            _sigVecAddWeightedFromBuffer?.Dispose();

            _stagingUploadBuffer?.Dispose();
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
