namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Xunit;

[Collection("M1GpuSerial")]
public class VramReserveOverrideTests
{
    [Fact]
    public void WddmVramGuard_ResolveExplicitReserveBytes_HonorsZeroAndRejectsNegative()
    {
        Assert.Null(WddmVramGuard.ResolveExplicitReserveBytes(null));
        Assert.Equal(0UL, WddmVramGuard.ResolveExplicitReserveBytes(0));
        Assert.Equal(256UL * 1024 * 1024, WddmVramGuard.ResolveExplicitReserveBytes(256));
        Assert.Throws<ArgumentOutOfRangeException>(() => WddmVramGuard.ResolveExplicitReserveBytes(-1));
    }

    [Fact]
    public void WddmVramGuard_ResolveModelSafetyReserveBytes_DoesNotClampExplicitZero()
    {
        ulong zeroReserve = WddmVramGuard.ResolveModelSafetyReserveBytes(
            dedicatedVramBytes: 8UL * 1024 * 1024 * 1024,
            isWindowsWddm: true,
            isUnifiedMemory: false,
            explicitReserveBytes: 0UL);
        Assert.Equal(0UL, zeroReserve);

        ulong autoReserve = WddmVramGuard.ResolveModelSafetyReserveBytes(
            dedicatedVramBytes: 8UL * 1024 * 1024 * 1024,
            isWindowsWddm: true,
            isUnifiedMemory: false,
            explicitReserveBytes: null);
        Assert.True(autoReserve >= 350_000_000UL);

        ulong budget = 4UL * 1024 * 1024 * 1024;
        ulong usage = 1UL * 1024 * 1024 * 1024;
        var snap = WddmVramGuard.EvaluateHeadroom(
            dedicatedVramBytes: budget,
            osBudgetBytes: budget,
            currentUsageBytes: usage,
            isWindowsWddm: true,
            isUnifiedMemory: false,
            explicitReserveBytes: 0UL);

        Assert.Equal(0UL, snap.SafetyReserveBytes);
        Assert.Equal(budget - usage, snap.SafeAvailableBytes);
    }

    [Fact]
    public void D3D12Context_And_Models_HonorExplicitZeroVramReserve()
    {
        if (!D3D12Context.IsSupported)
            return;

        string tempGguf = KvCacheQ4Tests.CreateSyntheticQwen2Gguf(
            dim: 64,
            ffnDim: 64,
            nHeads: 2,
            nHeadsKv: 1,
            layers: 2,
            vocab: 64,
            expertCount: 4,
            expertUsed: 2,
            expertFfnDim: 64);

        try
        {
            using (var ctx = new D3D12Context(-1))
            {
                var snapZero = ctx.QueryVramHeadroom(0UL);
                Assert.Equal(0UL, snapZero.SafetyReserveBytes);

                var snapAuto = ctx.QueryVramHeadroom(null);
                Assert.True(snapAuto.SafetyReserveBytes > 0UL || ctx.DedicatedVideoMemory == 0UL);
            }

            using (var gguf = GgufFile.Open(tempGguf))
            using (var weights = new ModelWeights(gguf))
            {
                using (var ctxZero = new D3D12Context(-1))
                using (var modelZero = new Qwen2D3D12Model(
                    ctxZero,
                    weights,
                    maxSeqLen: 64,
                    kvPrecision: KvCachePrecision.Q4_0,
                    explicitVramReserveBytes: 0UL))
                {
                    Assert.Equal(0UL, modelZero.EffectiveVramReserveBytes);
                    Assert.Equal(KvCachePrecision.Q4_0, modelZero.KvPrecision);
                }

                using (var ctxAuto = new D3D12Context(-1))
                using (var modelAuto = new Qwen2D3D12Model(
                    ctxAuto,
                    weights,
                    maxSeqLen: 64,
                    kvPrecision: KvCachePrecision.Auto,
                    explicitVramReserveBytes: null))
                {
                    Assert.True(modelAuto.EffectiveVramReserveBytes >= 350_000_000UL);
                }
            }

            var opts = new TieredRunnerOptions
            {
                MaxSequenceLength = 64,
                VramReserveMb = 0,
                KvPrecision = KvCachePrecision.Q4_0
            };
            using var runner = new TieredMoERunner(tempGguf, opts);
            Assert.Equal(0UL, runner.EffectiveVramReserveBytes);
            Assert.Equal(KvCachePrecision.Q4_0, runner.KvPrecision);
            Assert.Equal(0UL, runner.QueryVramHeadroom().SafetyReserveBytes);
        }
        finally
        {
            KvCacheQ4Tests.TryDeleteFile(tempGguf);
        }
    }

    [Fact]
    public void InferenceSession_Propagates_KvPrecision_And_VramReserveMb()
    {
        string tempGguf = KvCacheQ4Tests.CreateSyntheticQwen2Gguf(dim: 64, ffnDim: 128, nHeads: 2, nHeadsKv: 1, layers: 2, vocab: 64);
        try
        {
            using var cpuSession = new InferenceSession(
                tempGguf,
                maxSeqLen: 64,
                device: "cpu",
                engine: InferenceEngineType.Cpu,
                kvPrecision: KvCachePrecision.Q4_0,
                vramReserveMb: 0);
            Assert.Equal(KvCachePrecision.Q4_0, cpuSession.KvPrecision);
            Assert.NotNull(cpuSession.KVCache);
            Assert.Equal(KvCachePrecision.Q4_0, cpuSession.KVCache!.Precision);

            if (D3D12Context.IsSupported)
            {
                using var d3dSession = new InferenceSession(
                    tempGguf,
                    maxSeqLen: 64,
                    device: "gpu",
                    engine: InferenceEngineType.DirectML,
                    kvPrecision: KvCachePrecision.Q4_0,
                    vramReserveMb: 0);
                Assert.Equal(KvCachePrecision.Q4_0, d3dSession.KvPrecision);
                Assert.Contains("KV: Q4_0", d3dSession.ActiveDevice);
            }
        }
        finally
        {
            KvCacheQ4Tests.TryDeleteFile(tempGguf);
        }
    }
}
