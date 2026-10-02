namespace Glacier.Inference.Tests;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D12;
using Xunit;
using Xunit.Abstractions;

public class Qwen3Diagnostics
{
    private readonly ITestOutputHelper _output;
    private const string ModelPath = @"D:\lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf";

    public Qwen3Diagnostics(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public unsafe void TestGpuStepByStep()
    {
        if (!File.Exists(ModelPath) || !OperatingSystem.IsWindows()) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);

        int token = 151644; // <|im_start|>
        int vocabSize = weights.VocabSize;

        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen: 128);

        float[] gpuLogits = new float[vocabSize];
        gpuModel.Forward(token, 0, gpuLogits, true);

        _output.WriteLine($"GPU Logits[0..3]: {gpuLogits[0]:F4}, {gpuLogits[1]:F4}, {gpuLogits[2]:F4}, {gpuLogits[3]:F4}");
        float maxVal = 0;
        int maxIdx = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            if (Math.Abs(gpuLogits[i]) > maxVal)
            {
                maxVal = Math.Abs(gpuLogits[i]);
                maxIdx = i;
            }
        }
        _output.WriteLine($"GPU Max Logit: {gpuLogits[maxIdx]:F4} at token {maxIdx}");
        Assert.True(maxVal > 0, "GPU Logits should be non-zero!");
    }

    [Fact]
    public unsafe void CompareCpuAndGpuFirstTokenLogits()
    {
        if (!File.Exists(ModelPath) || !OperatingSystem.IsWindows()) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);

        int token = 151644; // <|im_start|>
        int vocabSize = weights.VocabSize;

        _output.WriteLine("[Test] Running CPU Forward Pass...");
        using var cpuModel = new Qwen3HybridModel(weights, maxSeqLen: 128);
        using var kvCache = new Glacier.Inference.Memory.KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128);
        float[] cpuLogits = new float[vocabSize];
        cpuModel.Forward(token, 0, kvCache, cpuLogits, true);

        // Top 5 for CPU
        var cpuTopIndices = new int[5];
        var cpuSorted = System.Linq.Enumerable.Range(0, vocabSize).OrderByDescending(i => cpuLogits[i]).Take(5).ToArray();
        _output.WriteLine("[CPU] Top 5 Tokens:");
        Console.WriteLine("[CPU] Top 5 Tokens:");
        for (int k = 0; k < 5; k++)
        {
            int t = cpuSorted[k];
            _output.WriteLine($"  #{k+1}: token {t} logit {cpuLogits[t]:F4}");
            Console.WriteLine($"  #{k+1}: token {t} logit {cpuLogits[t]:F4}");
        }

        _output.WriteLine("[Test] Initializing D3D12 GPU Model...");
        Console.WriteLine("[Test] Initializing D3D12 GPU Model...");
        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen: 128);

        _output.WriteLine("[Test] Running GPU Forward Pass...");
        Console.WriteLine("[Test] Running GPU Forward Pass...");
        float[] gpuLogits = new float[vocabSize];
        gpuModel.Forward(token, 0, gpuLogits, true);

        var gpuSorted = System.Linq.Enumerable.Range(0, vocabSize).OrderByDescending(i => gpuLogits[i]).Take(5).ToArray();
        _output.WriteLine("[GPU] Top 5 Tokens:");
        Console.WriteLine("[GPU] Top 5 Tokens:");
        for (int k = 0; k < 5; k++)
        {
            int t = gpuSorted[k];
            _output.WriteLine($"  #{k+1}: token {t} logit {gpuLogits[t]:F4}");
            Console.WriteLine($"  #{k+1}: token {t} logit {gpuLogits[t]:F4}");
        }

        // Compare top token
        Assert.Equal(cpuSorted[0], gpuSorted[0]);
    }

    [Fact]
    public unsafe void CompareLayer0()
    {
        if (!File.Exists(ModelPath) || !OperatingSystem.IsWindows()) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);

        int token = 151644;
        int dim = weights.EmbeddingLength;

        // 1. CPU Layer 0
        float[] cpuX = new float[dim];
        fixed (float* pX = cpuX)
        {
            QuantKernels.ExtractEmbedding(weights.EmbdType, weights.EmbdWeight, token, pX, dim);
        }
        _output.WriteLine($"[CPU Layer 0 Input] x[0..3]: {cpuX[0]:F4}, {cpuX[1]:F4}, {cpuX[2]:F4}, {cpuX[3]:F4}");

        using var cpuModel = new Qwen3HybridModel(weights, maxSeqLen: 128, startLayer: 0, layerCount: 8);
        using var kvCache = new Glacier.Inference.Memory.KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128);
        float[] dummyLogits = new float[weights.VocabSize];
        // Run forward stage with LayerCount = 8
        cpuModel.ForwardStage(token, 0, default, default, dummyLogits, false, kvCache);

        // Read out cpuModel._x
        var fi = typeof(CpuModelBase).GetField("_x", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        IntPtr pCpuX = (IntPtr)System.Reflection.Pointer.Unbox(fi!.GetValue(cpuModel)!);
        float[] cpuOutX = new float[dim];
        fixed (float* pDst = cpuOutX)
        {
            Buffer.MemoryCopy((void*)pCpuX, pDst, (ulong)(dim * sizeof(float)), (ulong)(dim * sizeof(float)));
        }
        _output.WriteLine($"[CPU Layer 7 Output] x[0..3]: {cpuOutX[0]:F4}, {cpuOutX[1]:F4}, {cpuOutX[2]:F4}, {cpuOutX[3]:F4}");

        // 2. GPU Layer 0..7
        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen: 128);

        // Run GPU forward for token with maxLayers: 8
        gpuModel.Forward(token, 0, dummyLogits, false, maxLayers: 8);

        var fiGpuX = typeof(Qwen3HybridD3D12Model).GetField("_dX", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var dX = (ID3D12Resource)fiGpuX!.GetValue(gpuModel)!;

        // Readback _dX
        using var readback = ctx.CreateReadbackBuffer((ulong)(dim * sizeof(float)));
        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        cmd.ResourceBarrierTransition(dX, ResourceStates.Common, ResourceStates.CopySource);
        cmd.CopyBufferRegion(readback, 0, dX, 0, (ulong)(dim * sizeof(float)));
        cmd.ResourceBarrierTransition(dX, ResourceStates.CopySource, ResourceStates.Common);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] gpuOutX = new float[dim];
        void* pGpu = null;
        readback.Map(0, null, &pGpu);
        fixed (float* pDst = gpuOutX)
        {
            Buffer.MemoryCopy(pGpu, pDst, (ulong)(dim * sizeof(float)), (ulong)(dim * sizeof(float)));
        }
        readback.Unmap(0);
        _output.WriteLine($"[GPU Layer 7 Output] x[0..3]: {gpuOutX[0]:F4}, {gpuOutX[1]:F4}, {gpuOutX[2]:F4}, {gpuOutX[3]:F4}");

        float maxDiff = 0;
        for (int i = 0; i < dim; i++)
        {
            float diff = Math.Abs(cpuOutX[i] - gpuOutX[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        _output.WriteLine($"[Comparison Layer 7] Max Absolute Difference across {dim} dims: {maxDiff:F4}");
        Assert.True(maxDiff < 0.2f, $"Layer 7 output differs by {maxDiff:F4}!");
    }

    [Fact]
    public unsafe void InspectLayer3Weights()
    {
        if (!File.Exists(ModelPath) || !OperatingSystem.IsWindows()) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);
        var l3 = weights.Layers[3];

        _output.WriteLine($"Layer 3: IsGdn={l3.IsGdn}, HasQGate={l3.HasQGate}");
        _output.WriteLine($"  QBias={l3.QBias != null}, KBias={l3.KBias != null}, VBias={l3.VBias != null}, AttnOutBias={l3.AttnOutBias != null}");
        _output.WriteLine($"  AttnQNorm={l3.AttnQNormWeight != null}, AttnKNorm={l3.AttnKNormWeight != null}");
        _output.WriteLine($"  AttnSinks={l3.AttnSinksWeight != null}");
        _output.WriteLine($"  QType={l3.QType}, KType={l3.KType}, VType={l3.VType}, AttnOutType={l3.AttnOutType}");
        _output.WriteLine($"  RopeFreqBase={weights.RopeFreqBase}, RopeFreqsWeight={weights.RopeFreqsWeight != null}");
    }

    [Fact]
    public unsafe void DiagnoseLayer3()
    {
        if (!File.Exists(ModelPath) || !OperatingSystem.IsWindows()) return;

        using var gguf = GgufFile.Open(ModelPath);
        var weights = new ModelWeights(gguf);

        int token = 151644;
        int dim = weights.EmbeddingLength;
        int nHeads = weights.HeadCount;
        int headsKv = weights.HeadCountKv;
        int headDim = weights.HeadDim;
        int qDim = nHeads * headDim;
        int kvDim = headsKv * headDim;

        // 1. Run CPU up to Layer 2 and Layer 3
        using var cpuModel = new Qwen3HybridModel(weights, maxSeqLen: 128, startLayer: 0, layerCount: 4);
        using var kvCache = new Glacier.Inference.Memory.KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128);
        float[] dummyLogits = new float[weights.VocabSize];

        // 2. Run GPU up to Layer 3 (maxLayers: 3 runs 0, 1, 2)
        using var ctx = new D3D12Context();
        using var gpuModel = new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen: 128);
        gpuModel.Forward(token, 0, dummyLogits, false, maxLayers: 3);

        T GetF<T>(object o, string name) =>
            (T)o.GetType().GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(o)!;

        float* GetCpuPtr(object o, string name) =>
            (float*)System.Reflection.Pointer.Unbox(o.GetType().GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(o)!);

        void CompareFloats(string name, float[] gpu, float* cpu, int count)
        {
            float maxDiff = 0;
            for (int i = 0; i < count; i++)
            {
                float d = Math.Abs(gpu[i] - cpu[i]);
                if (d > maxDiff) maxDiff = d;
            }
            _output.WriteLine($"[{name}] MaxDiff across {count} elements: {maxDiff:F6} (gpu[0]={gpu[0]:F4}, cpu[0]={cpu[0]:F4})");
            Console.WriteLine($"[{name}] MaxDiff across {count} elements: {maxDiff:F6} (gpu[0]={gpu[0]:F4}, cpu[0]={cpu[0]:F4})");
        }

        float[] Readback(ID3D12Resource res, int count)
        {
            using var rb = ctx.CreateReadbackBuffer((ulong)(count * sizeof(float)));
            ctx.BeginCommands();
            var cmd = ctx.CommandList;
            cmd.ResourceBarrierTransition(res, ResourceStates.Common, ResourceStates.CopySource);
            cmd.CopyBufferRegion(rb, 0, res, 0, (ulong)(count * sizeof(float)));
            cmd.ResourceBarrierTransition(res, ResourceStates.CopySource, ResourceStates.Common);
            ctx.EndCommandsAndExecute();
            ctx.Synchronize();

            float[] r = new float[count];
            void* p = null;
            rb.Map(0, null, &p);
            fixed (float* pDst = r)
            {
                Buffer.MemoryCopy(p, pDst, (ulong)(count * sizeof(float)), (ulong)(count * sizeof(float)));
            }
            rb.Unmap(0);
            return r;
        }

        // Now run CPU Layer 3
        // To do this cleanly, call cpuModel.ForwardStage with layerCount: 4
        // First reset cpuModel and run layerCount: 3 to get exact matching state
        using var cpuModel3 = new Qwen3HybridModel(weights, maxSeqLen: 128, startLayer: 0, layerCount: 3);
        cpuModel3.ForwardStage(token, 0, default, default, dummyLogits, false, kvCache);

        // Run full 4 layers on cpuModel to get Layer 3 internals
        using var kvCache4 = new Glacier.Inference.Memory.KVCache(weights.BlockCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen: 128);
        cpuModel.ForwardStage(token, 0, default, default, dummyLogits, false, kvCache4);

        // Now step GPU through Layer 3 manually
        var dX = GetF<ID3D12Resource>(gpuModel, "_dX");
        var dNormX = GetF<ID3D12Resource>(gpuModel, "_dNormX");
        var dQFull = GetF<ID3D12Resource>(gpuModel, "_dQFull");
        var dQ = GetF<ID3D12Resource>(gpuModel, "_dQ");
        var dQGate = GetF<ID3D12Resource>(gpuModel, "_dQGate");
        var dK = GetF<ID3D12Resource>(gpuModel, "_dK");
        var dV = GetF<ID3D12Resource>(gpuModel, "_dV");
        var dAttnOut = GetF<ID3D12Resource>(gpuModel, "_dAttnOut");
        var dAttnProjOut = GetF<ID3D12Resource>(gpuModel, "_dAttnProjOut");

        var lwList = GetF<System.Collections.IList>(gpuModel, "_layerWeights");
        var lw3 = lwList[3]!;

        // Compare input X to layer 3
        float* pCpuNormX = GetCpuPtr(cpuModel, "_normX");
        float* pCpuQFull = GetCpuPtr(cpuModel, "_qFull");
        float* pCpuQ = GetCpuPtr(cpuModel, "_q");
        float* pCpuQGate = GetCpuPtr(cpuModel, "_qGate");
        float* pCpuK = GetCpuPtr(cpuModel, "_k");
        float* pCpuV = GetCpuPtr(cpuModel, "_v");
        float* pCpuAttnOut = GetCpuPtr(cpuModel, "_attnOut");
        float* pCpuAttnProj = GetCpuPtr(cpuModel, "_attnProj");
        float* pCpuX = GetCpuPtr(cpuModel, "_x");

        // Execute Layer 3 on GPU step-by-step
        var miRmsNorm = gpuModel.GetType().GetMethod("DispatchRmsNorm", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var miGemv = gpuModel.GetType().GetMethod("DispatchGemv", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var miRmsNormHeads = gpuModel.GetType().GetMethod("DispatchRmsNormHeads", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var miVecAdd = gpuModel.GetType().GetMethod("DispatchVecAdd", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var miSwiglu = gpuModel.GetType().GetMethod("DispatchSwiGLU", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;

        // Step 1: Pre-Norm
        ctx.BeginCommands();
        var cmd = ctx.CommandList;
        var attnNormWeight = (ID3D12Resource)lw3.GetType().GetProperty("AttnNormWeight")!.GetValue(lw3)!;
        miRmsNorm.Invoke(gpuModel, new object[] { cmd, dX, attnNormWeight, dNormX, dim, weights.RmsNormEps });
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 1 NormX", Readback(dNormX, dim), pCpuNormX, dim);

        // Step 2: Q Gemv
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var qWeight = (ID3D12Resource)lw3.GetType().GetProperty("QWeight")!.GetValue(lw3)!;
        var qType = (GgufType)lw3.GetType().GetProperty("QType")!.GetValue(lw3)!;
        miGemv.Invoke(gpuModel, new object[] { cmd, qType, dQFull, dNormX, qWeight.GPUVirtualAddress, dim, 2 * qDim });
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 2 QFull", Readback(dQFull, 2 * qDim), pCpuQFull, 2 * qDim);

        // Step 3: QGateSplit
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var sigQGate = GetF<ID3D12RootSignature>(gpuModel, "_sigQGateSplit");
        var psoQGate = GetF<ID3D12PipelineState>(gpuModel, "_psoQGateSplit");
        cmd.SetComputeRootSignature(sigQGate);
        cmd.SetPipelineState(psoQGate);
        uint* qgConsts = stackalloc uint[2];
        qgConsts[0] = (uint)nHeads;
        qgConsts[1] = (uint)headDim;
        cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)qgConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQFull.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, dQGate.GPUVirtualAddress);
        cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 3 QGateSplit _dQ", Readback(dQ, qDim), pCpuQ, qDim);
        CompareFloats("Step 3 QGateSplit _dQGate", Readback(dQGate, qDim), pCpuQGate, qDim);

        // Step 4: K and V Gemv
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var kWeight = (ID3D12Resource)lw3.GetType().GetProperty("KWeight")!.GetValue(lw3)!;
        var kType = (GgufType)lw3.GetType().GetProperty("KType")!.GetValue(lw3)!;
        var vWeight = (ID3D12Resource)lw3.GetType().GetProperty("VWeight")!.GetValue(lw3)!;
        var vType = (GgufType)lw3.GetType().GetProperty("VType")!.GetValue(lw3)!;
        miGemv.Invoke(gpuModel, new object[] { cmd, kType, dK, dNormX, kWeight.GPUVirtualAddress, dim, kvDim });
        miGemv.Invoke(gpuModel, new object[] { cmd, vType, dV, dNormX, vWeight.GPUVirtualAddress, dim, kvDim });
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 4 K Gemv", Readback(dK, kvDim), pCpuK, kvDim);
        CompareFloats("Step 4 V Gemv", Readback(dV, kvDim), pCpuV, kvDim);

        // Step 5: QNorm and KNorm
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var attnQNorm = (ID3D12Resource?)lw3.GetType().GetProperty("AttnQNormWeight")!.GetValue(lw3);
        var attnKNorm = (ID3D12Resource?)lw3.GetType().GetProperty("AttnKNormWeight")!.GetValue(lw3);
        if (attnQNorm != null)
            miRmsNormHeads.Invoke(gpuModel, new object[] { cmd, dQ, attnQNorm, nHeads, headDim, weights.RmsNormEps });
        if (attnKNorm != null)
            miRmsNormHeads.Invoke(gpuModel, new object[] { cmd, dK, attnKNorm, headsKv, headDim, weights.RmsNormEps });
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 5 QNorm", Readback(dQ, qDim), pCpuQ, qDim);
        CompareFloats("Step 5 KNorm", Readback(dK, kvDim), pCpuK, kvDim);

        // Step 6: RoPE (pos = 0)
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var sigRope = GetF<ID3D12RootSignature>(gpuModel, "_sigRope");
        var psoRope = GetF<ID3D12PipelineState>(gpuModel, "_psoRope");
        cmd.SetComputeRootSignature(sigRope);
        cmd.SetPipelineState(psoRope);
        uint* ropeConsts = stackalloc uint[7];
        ropeConsts[0] = (uint)nHeads;
        ropeConsts[1] = (uint)headsKv;
        ropeConsts[2] = (uint)headDim;
        ropeConsts[3] = (uint)weights.RopeDimensionCount;
        ropeConsts[4] = 0; // pos = 0
        *(float*)(&ropeConsts[5]) = weights.RopeFreqBase;
        *(float*)(&ropeConsts[6]) = 1.0f;
        cmd.SetComputeRoot32BitConstants(0, 7, (IntPtr)ropeConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dK.GPUVirtualAddress);
        uint totalPairs = (uint)((nHeads + headsKv) * (weights.RopeDimensionCount / 2));
        cmd.Dispatch((totalPairs + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 6 RoPE Q", Readback(dQ, qDim), pCpuQ, qDim);
        CompareFloats("Step 6 RoPE K", Readback(dK, kvDim), pCpuK, kvDim);

        // Step 7: KvStore
        var keyCaches = GetF<ID3D12Resource[]>(gpuModel, "_dKeyCache");
        var valCaches = GetF<ID3D12Resource[]>(gpuModel, "_dValCache");
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var sigKvStore = GetF<ID3D12RootSignature>(gpuModel, "_sigKvStore");
        var psoKvStore = GetF<ID3D12PipelineState>(gpuModel, "_psoKvStore");
        cmd.SetComputeRootSignature(sigKvStore);
        cmd.SetPipelineState(psoKvStore);
        uint* kvConsts = stackalloc uint[4];
        kvConsts[0] = (uint)headsKv;
        kvConsts[1] = (uint)headDim;
        kvConsts[2] = 128; // maxSeqLen
        kvConsts[3] = 0; // pos = 0
        cmd.SetComputeRoot32BitConstants(0, 4, (IntPtr)kvConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dK.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dV.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, keyCaches[3].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, valCaches[3].GPUVirtualAddress);
        cmd.Dispatch(((uint)kvDim + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 7 KvStore K", Readback(keyCaches[3], kvDim), pCpuK, kvDim);
        CompareFloats("Step 7 KvStore V", Readback(valCaches[3], kvDim), pCpuV, kvDim);

        // Step 8: Attention
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var sigFlashAttn = GetF<ID3D12RootSignature>(gpuModel, "_sigFlashAttn");
        var psoFlashAttn = GetF<ID3D12PipelineState>(gpuModel, "_psoFlashAttn");
        cmd.SetComputeRootSignature(sigFlashAttn);
        cmd.SetPipelineState(psoFlashAttn);
        uint* faConsts = stackalloc uint[6];
        faConsts[0] = (uint)nHeads;
        faConsts[1] = (uint)headsKv;
        faConsts[2] = (uint)headDim;
        faConsts[3] = 128; // maxSeqLen
        faConsts[4] = 0; // pos = 0
        *(float*)(&faConsts[5]) = 1.0f / MathF.Sqrt(headDim);
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)faConsts, 0);
        cmd.SetComputeRootUnorderedAccessView(1, dQ.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, keyCaches[3].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, valCaches[3].GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, dAttnOut.GPUVirtualAddress);
        cmd.Dispatch((uint)nHeads, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        // Step 9: AttnOutGate
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var sigAttnGate = GetF<ID3D12RootSignature>(gpuModel, "_sigAttnOutGate");
        var psoAttnGate = GetF<ID3D12PipelineState>(gpuModel, "_psoAttnOutGate");
        cmd.SetComputeRootSignature(sigAttnGate);
        cmd.SetPipelineState(psoAttnGate);
        uint totalQ = (uint)qDim;
        cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&totalQ), 0);
        cmd.SetComputeRootUnorderedAccessView(1, dAttnOut.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(2, dQGate.GPUVirtualAddress);
        cmd.Dispatch(((uint)qDim + 255) / 256, 1, 1);
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();

        float[] postGateGpu = Readback(dAttnOut, qDim);
        for (int h = 0; h < nHeads; h++)
        {
            float headDiff = 0;
            for (int d = 0; d < headDim; d++)
            {
                float diff = Math.Abs(postGateGpu[h * headDim + d] - pCpuAttnOut[h * headDim + d]);
                if (diff > headDiff) headDiff = diff;
            }
            Console.WriteLine($"[Head {h:D2}] PostGate MaxDiff: {headDiff:F6} (gpu[0]={postGateGpu[h * headDim]:F4}, cpu[0]={pCpuAttnOut[h * headDim]:F4})");
        }


        // Step 10: OutProj GEMV
        ctx.BeginCommands();
        cmd = ctx.CommandList;
        var attnOutWeight = (ID3D12Resource)lw3.GetType().GetProperty("AttnOutWeight")!.GetValue(lw3)!;
        var attnOutType = (GgufType)lw3.GetType().GetProperty("AttnOutType")!.GetValue(lw3)!;
        miGemv.Invoke(gpuModel, new object[] { cmd, attnOutType, dAttnProjOut, dAttnOut, attnOutWeight.GPUVirtualAddress, qDim, dim });
        cmd.ResourceBarrierUnorderedAccessView(null!);
        ctx.EndCommandsAndExecute();
        ctx.Synchronize();
        CompareFloats("Step 10 AttnProjOut", Readback(dAttnProjOut, dim), pCpuAttnProj, dim);
    }
}


