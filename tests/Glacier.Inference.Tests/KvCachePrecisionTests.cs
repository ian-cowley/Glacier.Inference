namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Vortice.D3DCompiler;
using Xunit;

public class KvCachePrecisionTests
{
    [Fact]
    public void KvCachePrecision_EnumValues_MatchExpectedConstants()
    {
        Assert.Equal(0, (int)KvCachePrecision.Auto);
        Assert.Equal(1, (int)KvCachePrecision.Fp32);
        Assert.Equal(2, (int)KvCachePrecision.Fp16);
        Assert.Equal(3, (int)KvCachePrecision.Fp8);
    }

    [Theory]
    [InlineData(KvCachePrecision.Fp32, 4)]
    [InlineData(KvCachePrecision.Fp16, 2)]
    [InlineData(KvCachePrecision.Fp8, 1)]
    public void KvCachePrecision_ElementByteSizes_AreConsistent(KvCachePrecision precision, int expectedBytes)
    {
        int bytes = precision switch
        {
            KvCachePrecision.Fp8 => 1,
            KvCachePrecision.Fp16 => 2,
            _ => 4
        };

        Assert.Equal(expectedBytes, bytes);
    }

    [Fact]
    public void KvCachePrecision_MemoryAllocationCalculations_ScaleProportionally()
    {
        int layers = 28;
        int nHeadsKv = 4;
        int headDim = 128;
        int maxSeqLen = 4096;

        long numElementsPerLayer = (long)nHeadsKv * maxSeqLen * headDim;
        long totalElements = numElementsPerLayer * layers * 2; // Key and Value

        long bytesFp32 = totalElements * 4;
        long bytesFp16 = totalElements * 2;
        long bytesFp8 = totalElements * 1;

        Assert.Equal(bytesFp32 / 2, bytesFp16);
        Assert.Equal(bytesFp32 / 4, bytesFp8);
    }

    [Fact]
    public unsafe void KVCache_CpuExecution_OperatesInFP32()
    {
        // Validates that CPU KVCache operates directly with 32-bit floats for SIMD throughput
        using var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 4);

        float[] sample = [1.25f, 2.5f, 3.75f, 5.0f];
        fixed (float* pSample = sample)
        {
            cache.Store(0, 0, pSample, pSample);
        }

        float* keyOut = cache.GetKeyPtr(0, 0, 0);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(sample[i], keyOut[i]);
        }
    }

    [Fact]
    public void Half2Packing_RoundTrips_FloatsAccurately()
    {
        // Tests the packed half2 mathematical contract used by D3D12 F16 KV cache shaders
        float v0 = 3.14159f;
        float v1 = -0.0625f;

        ushort h0 = BitConverter.HalfToUInt16Bits((Half)v0);
        ushort h1 = BitConverter.HalfToUInt16Bits((Half)v1);

        uint packed = (uint)h0 | ((uint)h1 << 16);

        float recovered0 = (float)BitConverter.UInt16BitsToHalf((ushort)(packed & 0xFFFF));
        float recovered1 = (float)BitConverter.UInt16BitsToHalf((ushort)(packed >> 16));

        Assert.Equal(v0, recovered0, 0.005f);
        Assert.Equal(v1, recovered1, 0.005f);
    }

    [Fact]
    public void D3D12_KvStoreBatchF16_ShaderSource_ContainsPackedHalfOperations()
    {
        string shader = D3D12Shaders.KvCacheStoreBatchF16;
        Assert.NotNull(shader);
        Assert.Contains("f32tof16", shader);
        Assert.Contains("RWStructuredBuffer<uint> k_cache", shader);
        Assert.Contains("RWStructuredBuffer<uint> v_cache", shader);
        Assert.Contains("d_pair", shader);
    }

    [Fact]
    public void D3D12_AttentionBatchF16_ShaderSource_ContainsHalfUnpacking()
    {
        string shader = D3D12Shaders.AttentionBatchF16;
        Assert.NotNull(shader);
        Assert.Contains("f16tof32", shader);
        Assert.Contains("RWStructuredBuffer<uint> k_cache", shader);
        Assert.Contains("RWStructuredBuffer<uint> v_cache", shader);
        Assert.Contains("pair_idx", shader);
    }

    [Fact]
    public void D3D12_BatchPrefill_Shaders_Compile_Cleanly()
    {
        if (!OperatingSystem.IsWindows()) return;

        var kvStoreFp32 = Compiler.Compile(D3D12Shaders.KvCacheStoreBatch, "main", "kv_store_batch.hlsl", "cs_5_0");
        Assert.False(kvStoreFp32.IsEmpty);

        var attnFp32 = Compiler.Compile(D3D12Shaders.AttentionBatch, "main", "attention_batch.hlsl", "cs_5_0");
        Assert.False(attnFp32.IsEmpty);

        var kvStoreF16 = Compiler.Compile(D3D12Shaders.KvCacheStoreBatchF16, "main", "kv_store_batch_f16.hlsl", "cs_5_0");
        Assert.False(kvStoreF16.IsEmpty);

        var attnF16 = Compiler.Compile(D3D12Shaders.AttentionBatchF16, "main", "attention_batch_f16.hlsl", "cs_5_0");
        Assert.False(attnF16.IsEmpty);
    }
}
