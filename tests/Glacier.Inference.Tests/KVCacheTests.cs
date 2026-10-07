namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Memory;
using Xunit;

public unsafe class KVCacheTests
{
    [Fact]
    public void KVCache_StoresAndRetrievesExactVectors()
    {
        using var cache = new KVCache(layers: 4, nHeadsKv: 2, headDim: 8, maxSeqLen: 16);

        float[] kIn = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f,  9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f];
        float[] vIn = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f, 90f, 100f, 110f, 120f, 130f, 140f, 150f, 160f];

        fixed (float* pK = kIn, pV = vIn)
        {
            cache.Store(layer: 2, pos: 5, pK, pV);
        }

        // Verify head 0
        float* kOut0 = cache.GetKeyPtr(layer: 2, headKv: 0, pos: 5);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(kIn[i], kOut0[i]);
        }

        // Verify head 1
        float* kOut1 = cache.GetKeyPtr(layer: 2, headKv: 1, pos: 5);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(kIn[8 + i], kOut1[i]);
        }

        // Verify V head 1
        float* vOut1 = cache.GetValuePtr(layer: 2, headKv: 1, pos: 5);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(vIn[8 + i], vOut1[i]);
        }
    }

    [Fact]
    public void KVCache_Reset_ZerosAllMemory()
    {
        using var cache = new KVCache(layers: 2, nHeadsKv: 2, headDim: 4, maxSeqLen: 8);
        float[] dummy = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];

        fixed (float* p = dummy)
        {
            cache.Store(0, 0, p, p);
        }

        Assert.Equal(1f, *cache.GetKeyPtr(0, 0, 0));

        cache.Reset();

        Assert.Equal(0f, *cache.GetKeyPtr(0, 0, 0));
        Assert.Equal(0f, *cache.GetValuePtr(0, 0, 0));
    }

    [Fact]
    public void KVCache_AsymmetricKeyAndValueDimensions_StoresAndRetrievesExactVectors()
    {
        // DeepSeek MLA-style: kHeadDim = 192, vHeadDim = 128
        using var cache = new KVCache(layers: 3, nHeadsKv: 2, headDim: 192, maxSeqLen: 32, vHeadDim: 128);

        float[] kIn = new float[2 * 192];
        float[] vIn = new float[2 * 128];

        for (int i = 0; i < kIn.Length; i++) kIn[i] = (i + 1) * 0.5f;
        for (int i = 0; i < vIn.Length; i++) vIn[i] = (i + 1) * 1.5f;

        fixed (float* pK = kIn, pV = vIn)
        {
            cache.Store(layer: 1, pos: 10, pK, pV);
        }

        // Verify head 0 Key (192 floats)
        float* kOut0 = cache.GetKeyPtr(layer: 1, headKv: 0, pos: 10);
        for (int i = 0; i < 192; i++)
        {
            Assert.Equal(kIn[i], kOut0[i]);
        }

        // Verify head 1 Key (192 floats)
        float* kOut1 = cache.GetKeyPtr(layer: 1, headKv: 1, pos: 10);
        for (int i = 0; i < 192; i++)
        {
            Assert.Equal(kIn[192 + i], kOut1[i]);
        }

        // Verify head 0 Value (128 floats)
        float* vOut0 = cache.GetValuePtr(layer: 1, headKv: 0, pos: 10);
        for (int i = 0; i < 128; i++)
        {
            Assert.Equal(vIn[i], vOut0[i]);
        }

        // Verify head 1 Value (128 floats)
        float* vOut1 = cache.GetValuePtr(layer: 1, headKv: 1, pos: 10);
        for (int i = 0; i < 128; i++)
        {
            Assert.Equal(vIn[128 + i], vOut1[i]);
        }
    }

    private static void StoreOnce(KVCache cache, int layer, int pos)
    {
        float[] data = [1f, 2f, 3f, 4f];
        fixed (float* p = data)
        {
            cache.Store(layer, pos, p, p);
        }
    }

    [Fact]
    public void KVCache_Store_OutOfRangePosition_Throws()
    {
        using var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => StoreOnce(cache, 0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => StoreOnce(cache, 0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => StoreOnce(cache, 1, 0));
    }

    [Fact]
    public void KVCache_AfterDispose_StoreAndResetThrow()
    {
        var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 4);
        cache.Dispose();
        cache.Dispose(); // idempotent
        Assert.Throws<ObjectDisposedException>(() => StoreOnce(cache, 0, 0));
        Assert.Throws<ObjectDisposedException>(() => cache.Reset());
    }

    [Theory]
    [InlineData(0, 1, 1, 1)]
    [InlineData(1, 0, 1, 1)]
    [InlineData(1, 1, 0, 1)]
    [InlineData(1, 1, 1, 0)]
    public void KVCache_InvalidConstructorArgs_Throw(int layers, int heads, int dim, int seq)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KVCache(layers, heads, dim, seq));
    }

    [Fact]
    public void KVCache_GetPtr_OutOfRange_Throws()
    {
        using var cache = new KVCache(layers: 2, nHeadsKv: 2, headDim: 4, maxSeqLen: 8);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetKeyPtr(2, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetKeyPtr(0, 2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetKeyPtr(0, 0, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetValuePtr(2, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetValuePtr(0, 2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetValuePtr(0, 0, 8));
    }

    [Fact]
    public void KVCache_GetPtr_AfterDispose_Throws()
    {
        var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 4);
        cache.Dispose();
        Assert.Throws<ObjectDisposedException>(() => cache.GetKeyPtr(0, 0, 0));
        Assert.Throws<ObjectDisposedException>(() => cache.GetValuePtr(0, 0, 0));
    }
}
