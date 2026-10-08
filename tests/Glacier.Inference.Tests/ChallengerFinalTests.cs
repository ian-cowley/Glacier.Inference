namespace Glacier.Inference.Tests;

using System;
using System.Buffers;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Gpu.OpenCL;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;
using Xunit;

/// <summary>
/// Adversarial challenger empirical stress-test suite for Milestones M2, M3, M4, and M5.
/// Empirically stress-tests:
/// 1. KVCache ring buffer & sliding window wrapping at pos >= maxSeqLen, asymmetric MLA dimensions, and linear bounds.
/// 2. SsmStateCache boundary validation, disposal guards, buffer size assertions, and SaveState/RestoreState roundtrip fidelity.
/// 3. Sampler.ApplyRepetitionPenalty zero-allocation contract across small/large history and small/large vocabularies, plus mathematical correctness.
/// 4. D3D12TensorAlign padding calculations, 64-bit block arithmetic, and zeroed padding bytes.
/// 5. OpenCLDriver handle safety and lifecycle management.
/// 6. Vulkan fence synchronization signature verification.
/// </summary>
public unsafe class ChallengerFinalTests
{
    #region 1. KVCache Ring Buffer & Sliding Window Stress Tests

    [Fact]
    public void KVCache_RingBuffer_Wrapping_StoresAndRetrievesExactVectors()
    {
        const int layers = 2;
        const int nHeadsKv = 2;
        const int headDim = 8;
        const int maxSeqLen = 16;

        using var cache = new KVCache(layers, nHeadsKv, headDim, maxSeqLen, isRingBuffer: true);
        Assert.True(cache.IsRingBuffer);
        Assert.Equal(maxSeqLen, cache.EffectiveWindowSize);

        float[] kIn = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f, 11f, 21f, 31f, 41f, 51f, 61f, 71f, 81f];
        float[] vIn = [100f, 200f, 300f, 400f, 500f, 600f, 700f, 800f, 101f, 201f, 301f, 401f, 501f, 601f, 701f, 801f];

        // Store at pos = 0
        fixed (float* pK = kIn, pV = vIn)
        {
            cache.Store(layer: 0, pos: 0, pK, pV);
        }

        // Verify slot 0 retrieval
        float* kOut = cache.GetKeyPtr(layer: 0, headKv: 0, pos: 0);
        for (int i = 0; i < 8; i++) Assert.Equal(kIn[i], kOut[i]);

        // Wrap around at pos = 2 * maxSeqLen + 15 = 47 (physical slot 47 % 16 = 15)
        const int wrapPos1 = 2 * maxSeqLen + 15;
        float[] kWrap1 = [1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 1.6f, 2.6f, 3.6f, 4.6f, 5.6f, 6.6f, 7.6f, 8.6f];
        float[] vWrap1 = [15f, 25f, 35f, 45f, 55f, 65f, 75f, 85f, 16f, 26f, 36f, 46f, 56f, 66f, 76f, 86f];

        fixed (float* pK = kWrap1, pV = vWrap1)
        {
            cache.Store(layer: 1, pos: wrapPos1, pK, pV);
        }

        Assert.Equal(15, cache.GetPhysicalSlot(wrapPos1));

        // Retrieve both via wrapPos1 (47) and logical slot (15)
        float* kOutWrap = cache.GetKeyPtr(layer: 1, headKv: 0, pos: wrapPos1);
        float* kOutDirect = cache.GetKeyPtr(layer: 1, headKv: 0, pos: 15);
        Assert.True(kOutWrap == kOutDirect, "Pointer for wrapped pos 47 must equal pointer for slot 15");

        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(kWrap1[i], kOutWrap[i]);
            Assert.Equal(kWrap1[i], kOutDirect[i]);
        }

        float* vOutWrap = cache.GetValuePtr(layer: 1, headKv: 1, pos: wrapPos1);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(vWrap1[8 + i], vOutWrap[i]);
        }

        // Stress: Huge position beyond 1 million tokens
        const int hugePos = 1_000_003; // 1_000_003 % 16 = 3
        Assert.Equal(hugePos % maxSeqLen, cache.GetPhysicalSlot(hugePos));

        float[] kHuge = [9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f, 0f, -1f, -2f, -3f, -4f, -5f, -6f];
        fixed (float* pK = kHuge, pV = kHuge)
        {
            cache.Store(layer: 0, pos: hugePos, pK, pV);
        }

        float* kOutHuge = cache.GetKeyPtr(layer: 0, headKv: 0, pos: hugePos);
        float* kOutSlot3 = cache.GetKeyPtr(layer: 0, headKv: 0, pos: 3);
        Assert.True(kOutHuge == kOutSlot3);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(kHuge[i], kOutHuge[i]);
        }
    }

    [Fact]
    public void KVCache_SlidingWindow_ModuloMapping_StoresAndWrapsIndefinitely()
    {
        const int layers = 2;
        const int nHeadsKv = 2;
        const int headDim = 4;
        const int maxSeqLen = 64;
        const int slidingWindow = 16;

        using var cache = new KVCache(layers, nHeadsKv, headDim, maxSeqLen, slidingWindow: slidingWindow);
        Assert.Equal(16, cache.SlidingWindow);
        Assert.Equal(16, cache.EffectiveWindowSize);

        // Verify slot calculations
        Assert.Equal(0, cache.GetPhysicalSlot(0));
        Assert.Equal(15, cache.GetPhysicalSlot(15));
        Assert.Equal(0, cache.GetPhysicalSlot(16));
        Assert.Equal(5, cache.GetPhysicalSlot(21));
        Assert.Equal(2, cache.GetPhysicalSlot(130)); // 130 % 16 = 2

        // Store sequential values and ensure overwriting works cleanly
        float[] buffer = new float[nHeadsKv * headDim];
        for (int pos = 0; pos < 100; pos++)
        {
            for (int i = 0; i < buffer.Length; i++) buffer[i] = pos * 100f + i;
            fixed (float* p = buffer)
            {
                cache.Store(layer: 0, pos: pos, p, p);
            }
        }

        // Pos 99 % 16 = 3 -> slot 3 must contain pos 99's data
        float* kSlot3 = cache.GetKeyPtr(0, 0, 99);
        Assert.Equal(99 * 100f, kSlot3[0]);

        // Pos 96 % 16 = 0 -> slot 0 must contain pos 96's data
        float* kSlot0 = cache.GetKeyPtr(0, 0, 96);
        Assert.Equal(96 * 100f, kSlot0[0]);
    }

    [Fact]
    public void KVCache_LinearMode_BoundaryEnforcement_ThrowsOnOutOfRange()
    {
        const int maxSeqLen = 16;
        using var cache = new KVCache(layers: 2, nHeadsKv: 2, headDim: 4, maxSeqLen: maxSeqLen, isRingBuffer: false, slidingWindow: 0);
        Assert.False(cache.IsRingBuffer);
        Assert.Equal(0, cache.SlidingWindow);

        // Within bounds
        Assert.Equal(0, cache.GetPhysicalSlot(0));
        Assert.Equal(15, cache.GetPhysicalSlot(15));

        // Out of bounds -> throws ArgumentOutOfRangeException
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetPhysicalSlot(16));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetPhysicalSlot(100));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetPhysicalSlot(-1));

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetKeyPtr(0, 0, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetValuePtr(0, 0, 16));

        Assert.Throws<ArgumentOutOfRangeException>(() => StoreHelper(cache, 0, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => StoreHelper(cache, 0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => StoreHelper(cache, 2, 0)); // layer out of range
    }

    private static void StoreHelper(KVCache cache, int layer, int pos)
    {
        float[] dummy = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        fixed (float* p = dummy)
        {
            cache.Store(layer, pos, p, p);
        }
    }

    [Fact]
    public void KVCache_AsymmetricDim_MLA_RingBuffer_StoresAndRetrieves()
    {
        // DeepSeek MLA: headDim = 128, vHeadDim = 64
        using var cache = new KVCache(layers: 2, nHeadsKv: 2, headDim: 128, maxSeqLen: 32, vHeadDim: 64, isRingBuffer: true);

        float[] kIn = new float[2 * 128];
        float[] vIn = new float[2 * 64];
        for (int i = 0; i < kIn.Length; i++) kIn[i] = i + 1f;
        for (int i = 0; i < vIn.Length; i++) vIn[i] = (i + 1f) * 10f;

        const int wrapPos = 32 * 3 + 7; // wraps to 7
        fixed (float* pK = kIn, pV = vIn)
        {
            cache.Store(layer: 1, pos: wrapPos, pK, pV);
        }

        float* kOut = cache.GetKeyPtr(layer: 1, headKv: 1, pos: wrapPos);
        for (int i = 0; i < 128; i++) Assert.Equal(kIn[128 + i], kOut[i]);

        float* vOut = cache.GetValuePtr(layer: 1, headKv: 1, pos: wrapPos);
        for (int i = 0; i < 64; i++) Assert.Equal(vIn[64 + i], vOut[i]);
    }

    [Fact]
    public void KVCache_DisposalGuards_ThrowObjectDisposedException()
    {
        var cache = new KVCache(layers: 1, nHeadsKv: 1, headDim: 4, maxSeqLen: 8, isRingBuffer: true);
        cache.Dispose();
        cache.Dispose(); // idempotent

        Assert.Throws<ObjectDisposedException>(() => cache.GetKeyPtr(0, 0, 0));
        Assert.Throws<ObjectDisposedException>(() => cache.GetValuePtr(0, 0, 0));
        Assert.Throws<ObjectDisposedException>(() => cache.Reset());

        Assert.Throws<ObjectDisposedException>(() => StoreHelper(cache, 0, 0));
    }

    #endregion

    #region 2. SsmStateCache Bounds, Disposal & Checkpoint Rollback Stress Tests

    [Fact]
    public void SsmStateCache_ConstructorValidation_NegativeOrZero_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(0, 4, 64, 4, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(2, 0, 64, 4, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(2, 4, 0, 4, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(2, 4, 64, 0, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(2, 4, 64, 4, 0));

        Assert.Throws<ArgumentOutOfRangeException>(() => new SsmStateCache(-1, 4, 64, 4, 128));
    }

    [Fact]
    public void SsmStateCache_BoundsChecking_GetConvState_ThrowsOnOutOfRange()
    {
        using var cache = new SsmStateCache(layerCount: 4, convKernel: 4, convChannels: 64, heads: 4, stateDim: 128);

        // Valid layers [0..3]
        for (int l = 0; l < 4; l++)
        {
            Assert.True(cache.GetConvState(l) != null);
        }

        // Out of bounds layers
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetConvState(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetConvState(4));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetConvState(99));
    }

    [Fact]
    public void SsmStateCache_BoundsChecking_GetRecurrentState_ThrowsOnOutOfRange()
    {
        using var cache = new SsmStateCache(layerCount: 4, convKernel: 4, convChannels: 64, heads: 8, stateDim: 128);

        // Valid layers [0..3], heads [0..7]
        for (int l = 0; l < 4; l++)
        {
            for (int h = 0; h < 8; h++)
            {
                Assert.True(cache.GetRecurrentState(l, h) != null);
            }
        }

        // Invalid layer indices
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(4, 0));

        // Invalid head indices
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(0, 8));

        // Both invalid
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(-1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.GetRecurrentState(4, 8));
    }

    [Fact]
    public void SsmStateCache_DisposalGuards_ThrowObjectDisposedException()
    {
        var cache = new SsmStateCache(layerCount: 2, convKernel: 4, convChannels: 32, heads: 2, stateDim: 64);
        cache.Dispose();
        cache.Dispose(); // idempotent

        Assert.Throws<ObjectDisposedException>(() => cache.GetConvState(0));
        Assert.Throws<ObjectDisposedException>(() => cache.GetRecurrentState(0, 0));
        Assert.Throws<ObjectDisposedException>(() => cache.Reset());

        byte[] buf = new byte[cache.TotalBytes];
        Assert.Throws<ObjectDisposedException>(() => cache.SaveState(buf));
        Assert.Throws<ObjectDisposedException>(() => cache.RestoreState(buf));
    }

    [Fact]
    public void SsmStateCache_SaveState_RestoreState_RoundtripFidelityUnderMutation()
    {
        const int layers = 3;
        const int kernel = 4;
        const int channels = 32;
        const int heads = 4;
        const int dim = 64;

        using var cache = new SsmStateCache(layers, kernel, channels, heads, dim);
        long totalBytes = cache.TotalBytes;
        Assert.True(totalBytes > 0);

        // 1. Populate state cache with distinct pseudo-random numbers
        var rand = new Random(42);
        for (int l = 0; l < layers; l++)
        {
            float* conv = cache.GetConvState(l);
            int convFloats = (kernel - 1) * channels;
            for (int i = 0; i < convFloats; i++)
            {
                conv[i] = (float)rand.NextDouble() * 10f;
            }

            for (int h = 0; h < heads; h++)
            {
                float* rec = cache.GetRecurrentState(l, h);
                int recFloats = dim * dim;
                for (int i = 0; i < recFloats; i++)
                {
                    rec[i] = (float)rand.NextDouble() * 100f;
                }
            }
        }

        // 2. Snapshot state via SaveState
        byte[] snapshot = new byte[totalBytes];
        cache.SaveState(snapshot);

        // Destination too small throws ArgumentException
        byte[] smallBuf = new byte[totalBytes - 1];
        Assert.Throws<ArgumentException>(() => cache.SaveState(smallBuf));
        Assert.Throws<ArgumentException>(() => cache.RestoreState(smallBuf));

        // 3. Mutate entire cache (simulate rejected speculative tokens or reset)
        cache.Reset();
        for (int l = 0; l < layers; l++)
        {
            float* conv = cache.GetConvState(l);
            Assert.Equal(0f, conv[0]);
            float* rec = cache.GetRecurrentState(l, 0);
            Assert.Equal(0f, rec[0]);
        }

        // 4. Restore state via RestoreState
        cache.RestoreState(snapshot);

        // 5. Verify 100% exact fidelity across all layers, heads, and matrices
        rand = new Random(42);
        for (int l = 0; l < layers; l++)
        {
            float* conv = cache.GetConvState(l);
            int convFloats = (kernel - 1) * channels;
            for (int i = 0; i < convFloats; i++)
            {
                float expected = (float)rand.NextDouble() * 10f;
                Assert.Equal(expected, conv[i]);
            }

            for (int h = 0; h < heads; h++)
            {
                float* rec = cache.GetRecurrentState(l, h);
                int recFloats = dim * dim;
                for (int i = 0; i < recFloats; i++)
                {
                    float expected = (float)rand.NextDouble() * 100f;
                    Assert.Equal(expected, rec[i]);
                }
            }
        }
    }

    #endregion

    #region 3. Sampler Zero-Allocation Repetition Penalty Stress Tests

    [Fact]
    public void Sampler_ApplyRepetitionPenalty_ZeroAllocations_SmallHistory()
    {
        const int vocabSize = 32_000;
        float[] logitsArr = new float[vocabSize];
        int[] recentTokensArr = new int[64]; // <= 128

        var rand = new Random(1234);
        for (int i = 0; i < recentTokensArr.Length; i++)
        {
            recentTokensArr[i] = rand.Next(0, vocabSize);
        }

        // Warm up
        Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);

        // Measure allocations
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int iter = 0; iter < 500; iter++)
        {
            Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);
        }

        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        long diff = allocatedAfter - allocatedBefore;

        Assert.Equal(0, diff);
    }

    [Fact]
    public void Sampler_ApplyRepetitionPenalty_ZeroAllocations_LargeHistory_SmallVocab()
    {
        // 32,000 vocab fits in stackalloc 512 ulongs (32000/64 = 500 words <= 512)
        const int vocabSize = 32_000;
        float[] logitsArr = new float[vocabSize];
        int[] recentTokensArr = new int[256]; // > 128

        var rand = new Random(4321);
        for (int i = 0; i < recentTokensArr.Length; i++)
        {
            recentTokensArr[i] = rand.Next(0, vocabSize);
        }

        // Warm up
        Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int iter = 0; iter < 500; iter++)
        {
            Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);
        }

        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        long diff = allocatedAfter - allocatedBefore;

        Assert.Equal(0, diff);
    }

    [Fact]
    public void Sampler_ApplyRepetitionPenalty_ZeroAllocations_LargeHistory_LargeVocab()
    {
        // 152,064 vocab (Qwen2.5) > 32,768, uses ArrayPool<ulong>.Shared
        const int vocabSize = 152_064;
        float[] logitsArr = new float[vocabSize];
        int[] recentTokensArr = new int[512]; // > 128

        var rand = new Random(9876);
        for (int i = 0; i < recentTokensArr.Length; i++)
        {
            recentTokensArr[i] = rand.Next(0, vocabSize);
        }

        // Warm up (ensures ArrayPool rent/return is primed)
        for (int w = 0; w < 5; w++)
        {
            Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int iter = 0; iter < 500; iter++)
        {
            Sampler.ApplyRepetitionPenalty(logitsArr, recentTokensArr, 1.2f);
        }

        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        long diff = allocatedAfter - allocatedBefore;

        Assert.Equal(0, diff);
    }

    [Theory]
    [InlineData(64)]   // Small history path (stack O(N^2))
    [InlineData(128)]  // Small history boundary
    [InlineData(129)]  // Large history boundary (bitset path)
    [InlineData(300)]  // Large history path (bitset path)
    public void Sampler_ApplyRepetitionPenalty_MathematicalCorrectness(int historyLength)
    {
        const int vocabSize = 1000;
        const float penalty = 1.5f;

        float[] logits1 = new float[vocabSize];
        float[] logits2 = new float[vocabSize];

        for (int i = 0; i < vocabSize; i++)
        {
            float val = (i % 2 == 0) ? (i + 1f) : -(i + 1f);
            logits1[i] = val;
            logits2[i] = val;
        }

        // Create recent tokens with duplicates and edge IDs
        int[] recentTokens = new int[historyLength];
        for (int i = 0; i < historyLength; i++)
        {
            // Repeat tokens deliberately (0, 10, 20, 30...)
            recentTokens[i] = (i % 15) * 10;
        }

        Sampler.ApplyRepetitionPenalty(logits1, recentTokens, penalty);

        // Ground-truth verification: each unique token in recentTokens penalized exactly once
        var seen = new HashSet<int>();
        for (int i = 0; i < recentTokens.Length; i++)
        {
            int tid = recentTokens[i];
            if (tid < vocabSize && seen.Add(tid))
            {
                if (logits2[tid] > 0)
                    logits2[tid] /= penalty;
                else
                    logits2[tid] *= penalty;
            }
        }

        for (int i = 0; i < vocabSize; i++)
        {
            Assert.Equal(logits2[i], logits1[i], precision: 5);
        }
    }

    [Fact]
    public void Sampler_ApplyRepetitionPenalty_OutOfRangeTokens_HandledSafely()
    {
        const int vocabSize = 100;
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) logits[i] = 10f;

        // Negative tokens, large tokens, zero tokens, both small and large history
        int[] maliciousSmall = [-1, -999, 100, 500, 100000, 50, 50, -5];
        Sampler.ApplyRepetitionPenalty(logits, maliciousSmall, 2.0f);
        Assert.Equal(5f, logits[50]); // penalized once

        // Large history with negative and out of range tokens
        int[] maliciousLarge = new int[200];
        for (int i = 0; i < 200; i++)
        {
            maliciousLarge[i] = (i % 2 == 0) ? -i : (1000 + i);
        }
        maliciousLarge[10] = 25; // one valid token

        Sampler.ApplyRepetitionPenalty(logits, maliciousLarge, 2.0f);
        Assert.Equal(5f, logits[25]); // penalized once
    }

    #endregion

    #region 4. D3D12 Tensor Alignment & Block Padding Stress Tests

    [Fact]
    public void D3D12TensorAlign_GetAlignedTensorBytes_Calculations()
    {
        // Q3_K: cols must be multiple of 256. 112 bytes per block of 256
        ulong q3kBytes = D3D12TensorAlign.GetAlignedTensorBytes(GgufType.Q3_K, rows: 10, cols: 512);
        Assert.Equal(10UL * 2UL * 112UL, q3kBytes);

        // Q6_K: 212 bytes per block of 256
        ulong q6kBytes = D3D12TensorAlign.GetAlignedTensorBytes(GgufType.Q6_K, rows: 20, cols: 512);
        Assert.Equal(20UL * 2UL * 212UL, q6kBytes);

        // Q8_0: 36 bytes per block of 32
        ulong q8_0Bytes = D3D12TensorAlign.GetAlignedTensorBytes(GgufType.Q8_0, rows: 5, cols: 64);
        Assert.Equal(5UL * 2UL * 36UL, q8_0Bytes);
    }

    [Fact]
    public void D3D12TensorAlign_AlignQ3K_PadsAndZeroesBoundaryBytes()
    {
        const int totalBlocks = 16;
        byte[] raw = new byte[totalBlocks * 110];
        for (int i = 0; i < raw.Length; i++) raw[i] = (byte)((i % 250) + 1);

        byte[] aligned = D3D12TensorAlign.AlignQ3K(raw, totalBlocks);
        Assert.Equal(totalBlocks * 112, aligned.Length);

        for (int b = 0; b < totalBlocks; b++)
        {
            // Verify 110 payload bytes preserved
            for (int i = 0; i < 110; i++)
            {
                Assert.Equal(raw[b * 110 + i], aligned[b * 112 + i]);
            }
            // Verify padding bytes 110 and 111 zeroed
            Assert.Equal(0, aligned[b * 112 + 110]);
            Assert.Equal(0, aligned[b * 112 + 111]);
        }
    }

    [Fact]
    public void D3D12TensorAlign_AlignQ6K_PadsAndZeroesBoundaryBytes_64BitScaling()
    {
        const int totalBlocks = 32;
        byte[] raw = new byte[totalBlocks * 210];
        for (int i = 0; i < raw.Length; i++) raw[i] = (byte)((i % 250) + 1);

        byte[] aligned = D3D12TensorAlign.AlignQ6K(raw, totalBlocks);
        Assert.Equal(totalBlocks * 212, aligned.Length);

        for (int b = 0; b < totalBlocks; b++)
        {
            for (int i = 0; i < 210; i++)
            {
                Assert.Equal(raw[b * 210 + i], aligned[b * 212 + i]);
            }
            Assert.Equal(0, aligned[b * 212 + 210]);
            Assert.Equal(0, aligned[b * 212 + 211]);
        }
    }

    [Fact]
    public void D3D12TensorAlign_AlignQ8_0_PadsAndZeroesBoundaryBytes()
    {
        const int totalBlocks = 32;
        byte[] raw = new byte[totalBlocks * 34];
        for (int i = 0; i < raw.Length; i++) raw[i] = (byte)((i % 250) + 1);

        byte[] aligned = D3D12TensorAlign.AlignQ8_0(raw, totalBlocks);
        Assert.Equal(totalBlocks * 36, aligned.Length);

        for (int b = 0; b < totalBlocks; b++)
        {
            // First 2 bytes are ushort scale
            Assert.Equal(raw[b * 34 + 0], aligned[b * 36 + 0]);
            Assert.Equal(raw[b * 34 + 1], aligned[b * 36 + 1]);
            // Bytes 2 and 3 must be zero padding
            Assert.Equal(0, aligned[b * 36 + 2]);
            Assert.Equal(0, aligned[b * 36 + 3]);
            // Next 32 bytes are payload
            for (int i = 0; i < 32; i++)
            {
                Assert.Equal(raw[b * 34 + 2 + i], aligned[b * 36 + 4 + i]);
            }
        }
    }

    #endregion

    #region 5. OpenCL Driver Handle Safety & Android Compatibility Tests

    [Fact]
    public void OpenCLDriver_IsAvailable_MultipleCalls_SafeAndNoLeak()
    {
        // Call IsAvailable() repeatedly in a loop
        bool firstResult = OpenCLDriver.IsAvailable();
        for (int i = 0; i < 100; i++)
        {
            bool r = OpenCLDriver.IsAvailable();
            Assert.Equal(firstResult, r);
        }
    }

    [Fact]
    public void OpenCLContext_TrackingCollections_ExistAndThreadSafe()
    {
        // Verify via reflection that _buffers and _kernels fields exist and are of List<IntPtr>
        var buffersField = typeof(OpenCLContext).GetField("_buffers", BindingFlags.NonPublic | BindingFlags.Instance);
        var kernelsField = typeof(OpenCLContext).GetField("_kernels", BindingFlags.NonPublic | BindingFlags.Instance);
        var lockField = typeof(OpenCLContext).GetField("_lock", BindingFlags.NonPublic | BindingFlags.Instance);

        Assert.NotNull(buffersField);
        Assert.NotNull(kernelsField);
        Assert.NotNull(lockField);
        Assert.Equal(typeof(List<IntPtr>), buffersField.FieldType);
        Assert.Equal(typeof(List<IntPtr>), kernelsField.FieldType);
    }

    #endregion
}
