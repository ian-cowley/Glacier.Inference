namespace Glacier.Inference.Tests;

using System;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;
using Xunit;

public sealed unsafe class NariQTrellisTests
{
    [Theory]
    [InlineData(GgufType.TRL1, 44u, 1, 16, 2)]
    [InlineData(GgufType.TRL2, 45u, 2, 16, 4)]
    [InlineData(GgufType.TRL3, 46u, 3, 16, 6)]
    [InlineData(GgufType.TRL4, 47u, 4, 16, 8)]
    [InlineData(GgufType.TRL5, 48u, 5, 16, 10)]
    [InlineData(GgufType.TRL6, 49u, 6, 16, 12)]
    public void GgufTypes_TrellisTypes_HaveExactSpecIdsAndBlockSizes(
        GgufType type, uint expectedId, int expectedBits, int expectedBlockSize, int expectedBlockBytes)
    {
        Assert.Equal(expectedId, (uint)type);
        Assert.True(GgufTypes.IsTrellis(type));
        Assert.Equal(expectedBits, GgufTypes.GetTrellisBits(type));
        Assert.True(GgufTypes.TryGetBlockInfo(type, out int bs, out int bb));
        Assert.Equal(expectedBlockSize, bs);
        Assert.Equal(expectedBlockBytes, bb);

        var info = new GgufTensorInfo
        {
            Name = "blk.0.ffn_gate.weight",
            DimensionsCount = 2,
            Dimensions = [64, 32],
            Type = type,
            Offset = 0,
        };
        Assert.Equal((ulong)(64 * 32 * expectedBits / 8), info.GetByteSize());
    }

    [Fact]
    public void SylvesterRht_IsSelfInverseOrthogonal()
    {
        const int n = 64;
        var orig = new float[n];
        var buf = new float[n];
        var rng = new Random(123);
        for (int i = 0; i < n; i++) orig[i] = buf[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        fixed (float* p = buf)
        {
            TrellisKernels.Rht(p, n, 16);
            TrellisKernels.Rht(p, n, 16);
        }

        for (int i = 0; i < n; i++)
            Assert.InRange(Math.Abs(buf[i] - orig[i]), 0f, 1e-5f);
    }

    [Theory]
    [InlineData(12)]
    [InlineData(20)]
    [InlineData(68)]
    [InlineData(48)] // 12 * 4
    public void PaleyHadFullSlice_ForwardThenInverse_RecoversIdentity(int len)
    {
        var orig = new float[len];
        var buf = new float[len];
        var tmp = new float[2 * len];
        var rng = new Random(len + 7);
        for (int i = 0; i < len; i++) orig[i] = buf[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        fixed (float* p = buf)
        fixed (float* t = tmp)
        {
            Assert.True(TrellisKernels.HadFullSlice(p, len, ReadOnlySpan<int>.Empty, shards: 1, r: 0, inverse: false, null, null, t));
            Assert.True(TrellisKernels.HadFullSlice(p, len, ReadOnlySpan<int>.Empty, shards: 1, r: 0, inverse: true, null, null, t));
        }

        for (int i = 0; i < len; i++)
            Assert.InRange(Math.Abs(buf[i] - orig[i]), 0f, 1e-4f);
    }

    [Theory]
    [InlineData(GgufType.TRL2, 32, 32, 0, 0, 1, 0, 0, 1, false)]          // Identity both sides
    [InlineData(GgufType.TRL3, 64, 48, 1, 16, 1, 1, 32, 1, true)]        // Block RHT + pin/pout gathers
    [InlineData(GgufType.TRL3, 48, 48, 2, 0, 1, 2, 0, 1, true)]          // Paley m=12 whole-slice Hadamard both sides + gathers
    [InlineData(GgufType.TRL4, 32, 64, 2, 0, 2, 1, 16, 1, false)]        // Sharded Paley rows + block RHT cols
    public void TrellisMatVecMul_MatchesDenseDequantizedMatrix_BitAccurately(
        GgufType type, int k, int n,
        int rowsKind, int rowsBlock, int rowsShards,
        int colsKind, int colsBlock, int colsShards,
        bool usePermutations)
    {
        int bits = GgufTypes.GetTrellisBits(type);
        var code = new byte[k * n * bits / 8];
        var s16 = new ushort[k / 16];
        var pre = new float[k];
        var post = new float[n];
        var pin = usePermutations ? new int[k] : null;
        var pout = usePermutations ? new int[n] : null;
        var x = new float[k];

        var rng = new Random(42 + bits + k + n);
        rng.NextBytes(code);
        for (int i = 0; i < s16.Length; i++) s16[i] = BitConverter.HalfToUInt16Bits((Half)(0.01f + 0.02f * rng.NextSingle()));
        for (int i = 0; i < k; i++)
        {
            pre[i] = (rng.Next(2) == 0 ? 1f : -1f) * (0.5f + rng.NextSingle());
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        for (int i = 0; i < n; i++) post[i] = (rng.Next(2) == 0 ? 1f : -1f) * (0.5f + rng.NextSingle());

        if (pin != null)
        {
            for (int i = 0; i < k; i++) pin[i] = (i * 7 + 3) % k;
            // Ensure bijection by simple Fisher-Yates shuffle
            for (int i = 0; i < k; i++) pin[i] = i;
            for (int i = k - 1; i > 0; i--) { int j = rng.Next(i + 1); (pin[i], pin[j]) = (pin[j], pin[i]); }
        }
        if (pout != null)
        {
            for (int i = 0; i < n; i++) pout[i] = i;
            for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (pout[i], pout[j]) = (pout[j], pout[i]); }
        }

        int[] desc = [rowsKind, rowsBlock, rowsShards, colsKind, colsBlock, colsShards, 0];
        Assert.True(TrellisKernels.IsDescriptorSupported(k, n, desc));

        var denseW = new float[n * k];
        var yTrellis = new float[n];
        var yRegistry = new float[n];
        var yDense = new float[n];

        fixed (byte* pCode = code)
        fixed (ushort* pS16 = s16)
        fixed (float* pPre = pre)
        fixed (float* pPost = post)
        fixed (int* pPin = pin)
        fixed (int* pPout = pout)
        fixed (float* pX = x)
        fixed (float* pYT = yTrellis)
        fixed (float* pYR = yRegistry)
        fixed (float* pW = denseW)
        {
            var binding = new TrellisWeightBinding
            {
                Type = type,
                CodeData = pCode,
                K = k,
                N = n,
                S16 = pS16,
                Pre = pPre,
                Post = pPost,
                Pin = pPin,
                Pout = pPout,
                Descriptor = desc,
            };

            Assert.True(TrellisKernels.DequantizeMatrix(binding, pW));
            TrellisKernels.MatVecMul(binding, pX, pYT);

            // Also verify seamless QuantKernels.MatVecMul dispatch via TrellisWeightRegistry
            TrellisWeightRegistry.Register(binding);
            try
            {
                QuantKernels.MatVecMul(type, pCode, pX, pYR, k, n);
            }
            finally
            {
                TrellisWeightRegistry.Unregister(pCode);
            }

            for (int r = 0; r < n; r++)
            {
                double sum = 0.0;
                for (int c = 0; c < k; c++) sum += denseW[r * k + c] * x[c];
                yDense[r] = (float)sum;
            }
        }

        for (int r = 0; r < n; r++)
        {
            Assert.InRange(Math.Abs(yTrellis[r] - yDense[r]), 0f, 1e-3f);
            Assert.Equal(yTrellis[r], yRegistry[r]);
        }
    }

    [Fact]
    public void DecodeEmbeddingRow_MatchesDequantizedMatrixRow()
    {
        const int k = 32;
        const int n = 32;
        var code = new byte[k * n * 3 / 8];
        var s16 = new ushort[k / 16];
        var pre = new float[k];
        var post = new float[n];
        var rng = new Random(99);
        rng.NextBytes(code);
        for (int i = 0; i < s16.Length; i++) s16[i] = BitConverter.HalfToUInt16Bits((Half)0.05f);
        for (int i = 0; i < k; i++) pre[i] = 0.5f + rng.NextSingle();
        for (int i = 0; i < n; i++) post[i] = 0.5f + rng.NextSingle();

        int[] desc = [0, 0, 1, 1, 16, 1, 0]; // rows_kind=0 (embedding), cols_kind=1 (block RHT)
        var denseW = new float[n * k];
        var rowOut = new float[k];

        fixed (byte* pCode = code)
        fixed (ushort* pS16 = s16)
        fixed (float* pPre = pre)
        fixed (float* pPost = post)
        fixed (float* pW = denseW)
        fixed (float* pRow = rowOut)
        {
            var binding = new TrellisWeightBinding
            {
                Type = GgufType.TRL3,
                CodeData = pCode,
                K = k,
                N = n,
                S16 = pS16,
                Pre = pPre,
                Post = pPost,
                Descriptor = desc,
            };
            Assert.True(TrellisKernels.DequantizeMatrix(binding, pW));
            TrellisKernels.DecodeEmbeddingRow(binding, rowIndex: 19, pRow);
        }

        for (int c = 0; c < k; c++)
            Assert.InRange(Math.Abs(rowOut[c] - denseW[19 * k + c]), 0f, 1e-5f);
    }

    [Fact]
    public void BandedLmHead_MixedBitrateTRL6_TRL3_TRL2_MatchesIndividualBandsAndVocabGather()
    {
        const int k = 32;
        const int bandN = 16;
        const int totalN = bandN * 3;
        GgufType[] bandTypes = [GgufType.TRL6, GgufType.TRL3, GgufType.TRL2];
        var rng = new Random(777);

        var codes = new byte[3][];
        var s16s = new ushort[3][];
        var pres = new float[3][];
        var posts = new float[3][];
        for (int b = 0; b < 3; b++)
        {
            int bits = GgufTypes.GetTrellisBits(bandTypes[b]);
            codes[b] = new byte[k * bandN * bits / 8];
            rng.NextBytes(codes[b]);
            s16s[b] = new ushort[k / 16];
            for (int i = 0; i < s16s[b].Length; i++) s16s[b][i] = BitConverter.HalfToUInt16Bits((Half)0.02f);
            pres[b] = new float[k];
            posts[b] = new float[bandN];
            for (int i = 0; i < k; i++) pres[b][i] = 0.5f + rng.NextSingle();
            for (int i = 0; i < bandN; i++) posts[b][i] = 0.5f + rng.NextSingle();
        }

        var gather = new int[totalN];
        for (int i = 0; i < totalN; i++) gather[i] = i;
        for (int i = totalN - 1; i > 0; i--) { int j = rng.Next(i + 1); (gather[i], gather[j]) = (gather[j], gather[i]); }

        var x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        int[] desc = [1, 16, 1, 1, 16, 1, 0];
        var yBanded = new float[totalN];
        var yExpected = new float[totalN];

        fixed (byte* c0 = codes[0], c1 = codes[1], c2 = codes[2])
        fixed (ushort* s0 = s16s[0], s1 = s16s[1], s2 = s16s[2])
        fixed (float* pr0 = pres[0], pr1 = pres[1], pr2 = pres[2])
        fixed (float* po0 = posts[0], po1 = posts[1], po2 = posts[2])
        fixed (int* pGather = gather)
        fixed (float* pX = x)
        fixed (float* pYB = yBanded)
        {
            var b0 = new TrellisWeightBinding { Type = bandTypes[0], CodeData = c0, K = k, N = bandN, S16 = s0, Pre = pr0, Post = po0, Descriptor = desc };
            var b1 = new TrellisWeightBinding { Type = bandTypes[1], CodeData = c1, K = k, N = bandN, S16 = s1, Pre = pr1, Post = po1, Descriptor = desc };
            var b2 = new TrellisWeightBinding { Type = bandTypes[2], CodeData = c2, K = k, N = bandN, S16 = s2, Pre = pr2, Post = po2, Descriptor = desc };

            var rawConcat = new float[totalN];
            fixed (float* pRaw = rawConcat)
            {
                TrellisKernels.MatVecMul(b0, pX, pRaw);
                TrellisKernels.MatVecMul(b1, pX, pRaw + bandN);
                TrellisKernels.MatVecMul(b2, pX, pRaw + 2 * bandN);
            }
            for (int v = 0; v < totalN; v++) yExpected[v] = rawConcat[gather[v]];

            var banded = new TrellisBandedWeight { Bands = [b0, b1, b2], VocabGather = pGather, K = k, TotalN = totalN };
            TrellisWeightRegistry.RegisterBanded(c0, banded);
            try
            {
                QuantKernels.MatVecMul(bandTypes[0], c0, pX, pYB, k, totalN);
            }
            finally
            {
                TrellisWeightRegistry.Unregister(c0);
            }
        }

        for (int v = 0; v < totalN; v++)
            Assert.Equal(yExpected[v], yBanded[v]);
    }
}
