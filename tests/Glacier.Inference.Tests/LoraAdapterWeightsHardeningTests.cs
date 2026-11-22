using System;
using System.IO;
using System.Runtime.InteropServices;
using Glacier.Inference.Model;
using Xunit;

namespace Glacier.Inference.Tests;

public unsafe class LoraAdapterWeightsHardeningTests
{
    private static string WriteLoraBinary(
        string dir,
        string fileName,
        uint magic,
        uint version,
        int layerCount,
        int rank,
        float alpha,
        int declaredTensorCount,
        params (string Name, int Dim0, int Dim1, float[] Data)[] tensors)
    {
        string path = Path.Combine(dir, fileName);
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);
        bw.Write(magic);
        bw.Write(version);
        bw.Write(layerCount);
        bw.Write(rank);
        bw.Write(alpha);
        bw.Write(declaredTensorCount);

        foreach (var (name, dim0, dim1, data) in tensors)
        {
            bw.Write(name);
            bw.Write(dim0);
            bw.Write(dim1);
            bw.Write(MemoryMarshal.AsBytes(data.AsSpan()));
        }

        return path;
    }

    private static float[] Transpose(float[] src, int rows, int cols)
    {
        var dst = new float[src.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        return dst;
    }

    [Fact]
    public void Constructor_InvalidBounds_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraAdapterWeights(-1, 4, 8.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraAdapterWeights(1, -1, 8.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraAdapterWeights(1, 4, float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoraAdapterWeights(1, 4, float.PositiveInfinity));
    }

    [Fact]
    public void Load_NativeAndPeftTransposed_ProduceIdenticalOutputs()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_peft_").FullName;
        try
        {
            const int inDim = 6;
            const int outDim = 5;
            const int rank = 3;
            const float alpha = 6.0f;

            var aNative = new float[inDim * rank];
            var bNative = new float[rank * outDim];
            for (int i = 0; i < aNative.Length; i++) aNative[i] = 0.1f * ((i % 7) - 3);
            for (int i = 0; i < bNative.Length; i++) bNative[i] = 0.2f * ((i % 5) - 2);

            var aPeft = Transpose(aNative, inDim, rank);   // [rank, inDim]
            var bPeft = Transpose(bNative, rank, outDim);  // [outDim, rank]

            string nativePath = WriteLoraBinary(
                tempDir, "native.lora", 0x41524F4C, 1, 1, rank, alpha, 2,
                ("blk.0.attn_q.lora_a.weight", inDim, rank, aNative),
                ("blk.0.attn_q.lora_b.weight", rank, outDim, bNative));

            string peftPath = WriteLoraBinary(
                tempDir, "peft.lora", 0x41524F4C, 1, 1, rank, alpha, 2,
                ("blk.0.attn_q.lora_a.weight", rank, inDim, aPeft),
                ("blk.0.attn_q.lora_b.weight", outDim, rank, bPeft));

            var nativeWeights = LoraAdapterWeights.Load(nativePath);
            var peftWeights = LoraAdapterWeights.Load(peftPath);

            var projNative = nativeWeights.GetProjections(0).Get(LoraProjection.Q);
            var projPeft = peftWeights.GetProjections(0).Get(LoraProjection.Q);
            Assert.Equal(inDim, projNative.inDim);
            Assert.Equal(outDim, projNative.outDim);
            Assert.Equal(inDim, projPeft.inDim);
            Assert.Equal(outDim, projPeft.outDim);

            var x = new float[] { 1.0f, -2.0f, 0.5f, 1.25f, -0.75f, 2.0f };
            var yNative = new float[outDim];
            var yPeft = new float[outDim];

            fixed (float* px = x, pyn = yNative, pyp = yPeft)
            {
                nativeWeights.Apply(0, LoraProjection.Q, px, pyn, inDim, outDim);
                peftWeights.Apply(0, LoraProjection.Q, px, pyp, inDim, outDim);
            }

            // Compute exact reference: y = (alpha / rank) * (x * A) * B
            float scaling = alpha / rank;
            var lowRank = new float[rank];
            for (int i = 0; i < inDim; i++)
                for (int r = 0; r < rank; r++)
                    lowRank[r] += x[i] * aNative[i * rank + r];

            var yRef = new float[outDim];
            for (int r = 0; r < rank; r++)
                for (int d = 0; d < outDim; d++)
                    yRef[d] += scaling * lowRank[r] * bNative[r * outDim + d];

            for (int d = 0; d < outDim; d++)
            {
                Assert.Equal(yRef[d], yNative[d], 1e-5f);
                Assert.Equal(yRef[d], yPeft[d], 1e-5f);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Load_SquareRankTensorWithPeftSiblings_TransposesSquareTensorConsistently()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_sq_").FullName;
        try
        {
            const int rank = 4;
            const int kInDim = 8;
            const int kOutDim = 4;
            const float alpha = 4.0f;

            // Q is square [4, 4] for both A and B; K is non-square [8, 4] -> [4, 8] in PEFT.
            var qANative = new float[rank * rank];
            var qBNative = new float[rank * rank];
            for (int r = 0; r < rank; r++)
            {
                for (int c = 0; c < rank; c++)
                {
                    qANative[r * rank + c] = (r + 1) * 0.25f + c * 0.1f;
                    qBNative[r * rank + c] = (r + 1) * 0.15f - c * 0.05f;
                }
            }

            var kANative = new float[kInDim * rank];
            var kBNative = new float[rank * kOutDim];
            for (int i = 0; i < kANative.Length; i++) kANative[i] = 0.05f * (i + 1);
            for (int i = 0; i < kBNative.Length; i++) kBNative[i] = 0.1f * (i + 1);

            string peftPath = WriteLoraBinary(
                tempDir, "mixed_sq_peft.lora", 0x41524F4C, 1, 1, rank, alpha, 4,
                ("blk.0.attn_q.lora_a.weight", rank, rank, Transpose(qANative, rank, rank)),
                ("blk.0.attn_q.lora_b.weight", rank, rank, Transpose(qBNative, rank, rank)),
                ("blk.0.attn_k.lora_a.weight", rank, kInDim, Transpose(kANative, kInDim, rank)),
                ("blk.0.attn_k.lora_b.weight", kOutDim, rank, Transpose(kBNative, rank, kOutDim)));

            var loaded = LoraAdapterWeights.Load(peftPath);

            var x = new float[] { 1.0f, 0.5f, -0.5f, 2.0f };
            var yActual = new float[rank];
            fixed (float* px = x, py = yActual)
            {
                loaded.Apply(0, LoraProjection.Q, px, py, rank, rank);
            }

            var lowRank = new float[rank];
            for (int i = 0; i < rank; i++)
                for (int r = 0; r < rank; r++)
                    lowRank[r] += x[i] * qANative[i * rank + r];

            var yExpected = new float[rank];
            for (int r = 0; r < rank; r++)
                for (int d = 0; d < rank; d++)
                    yExpected[d] += lowRank[r] * qBNative[r * rank + d];

            for (int d = 0; d < rank; d++)
            {
                Assert.Equal(yExpected[d], yActual[d], 1e-5f);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Load_AllSquareRankByRank_PreservesNativeOrientation()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_allsq_").FullName;
        try
        {
            const int rank = 3;
            var aNative = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var bNative = new float[] { 1, 0, 0, 0, 2, 0, 0, 0, 3 };

            string path = WriteLoraBinary(
                tempDir, "all_sq.lora", 0x41524F4C, 1, 1, rank, rank, 2,
                ("blk.0.attn_q.lora_a.weight", rank, rank, aNative),
                ("blk.0.attn_q.lora_b.weight", rank, rank, bNative));

            var loaded = LoraAdapterWeights.Load(path);
            var q = loaded.GetProjections(0).Q;
            Assert.Equal(aNative, q.A);
            Assert.Equal(bNative, q.B);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Load_MalformedFiles_ThrowsInvalidDataException_AndSkipsNegativeLayerSafely()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_bad_").FullName;
        try
        {
            // Bad magic
            string badMagic = WriteLoraBinary(tempDir, "bad_magic.lora", 0x12345678, 1, 1, 2, 4.0f, 0);
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(badMagic));

            // Bad version
            string badVer = WriteLoraBinary(tempDir, "bad_ver.lora", 0x41524F4C, 2, 1, 2, 4.0f, 0);
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(badVer));

            // Negative header values / NaN alpha
            string negLayer = WriteLoraBinary(tempDir, "neg_layer.lora", 0x41524F4C, 1, -1, 2, 4.0f, 0);
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(negLayer));

            string zeroRank = WriteLoraBinary(tempDir, "zero_rank.lora", 0x41524F4C, 1, 1, 0, 4.0f, 1,
                ("blk.0.attn_q.lora_a.weight", 4, 2, new float[8]));
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(zeroRank));

            string nanAlpha = WriteLoraBinary(tempDir, "nan_alpha.lora", 0x41524F4C, 1, 1, 2, float.NaN, 0);
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(nanAlpha));

            // Incompatible shape for rank=3
            string badShape = WriteLoraBinary(tempDir, "bad_shape.lora", 0x41524F4C, 1, 1, 3, 6.0f, 1,
                ("blk.0.attn_q.lora_a.weight", 5, 4, new float[20]));
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(badShape));

            // Truncated stream (tensor declares 4x2 = 32 bytes, only write 16 bytes)
            string truncPath = Path.Combine(tempDir, "trunc.lora");
            using (var fs = File.Create(truncPath))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(0x41524F4Cu);
                bw.Write(1u);
                bw.Write(1);
                bw.Write(2);
                bw.Write(4.0f);
                bw.Write(1);
                bw.Write("blk.0.attn_q.lora_a.weight");
                bw.Write(4);
                bw.Write(2);
                bw.Write(new byte[16]); // half of required 32 bytes
            }
            Assert.Throws<InvalidDataException>(() => LoraAdapterWeights.Load(truncPath));

            // Negative layer index in tensor name should be skipped cleanly without IndexOutOfRangeException
            string negIdxPath = WriteLoraBinary(
                tempDir, "neg_idx.lora", 0x41524F4C, 1, 1, 2, 4.0f, 2,
                ("blk.-1.attn_q.lora_a.weight", 4, 2, new float[8]),
                ("blk.0.attn_q.lora_a.weight", 4, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }));
            var skipped = LoraAdapterWeights.Load(negIdxPath);
            Assert.NotNull(skipped.GetProjections(0).Q.A);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Apply_DimensionMismatch_ThrowsInvalidOperationException()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_mismatch_").FullName;
        try
        {
            string path = WriteLoraBinary(
                tempDir, "valid.lora", 0x41524F4C, 1, 1, 2, 4.0f, 2,
                ("blk.0.attn_q.lora_a.weight", 4, 2, new float[8]),
                ("blk.0.attn_q.lora_b.weight", 2, 6, new float[12]));

            var weights = LoraAdapterWeights.Load(path);
            var x = new float[8];
            var y = new float[8];

            fixed (float* px = x, py = y)
            {
                float* pX = px;
                float* pY = py;
                Assert.Throws<InvalidOperationException>(() => weights.Apply(0, LoraProjection.Q, pX, pY, 5, 6));
                Assert.Throws<InvalidOperationException>(() => weights.Apply(0, LoraProjection.Q, pX, pY, 4, 7));
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Apply_LargeRankAboveStackAllocThreshold_UsesArrayPoolAndComputesExactDelta()
    {
        string tempDir = Directory.CreateTempSubdirectory("glacier_lora_largerank_").FullName;
        try
        {
            const int inDim = 12;
            const int outDim = 10;
            const int rank = 300; // > 256 StackAllocRankThreshold
            const float alpha = 150.0f;

            var a = new float[inDim * rank];
            var b = new float[rank * outDim];
            for (int i = 0; i < a.Length; i++) a[i] = ((i % 11) - 5) * 0.01f;
            for (int i = 0; i < b.Length; i++) b[i] = ((i % 9) - 4) * 0.02f;

            string path = WriteLoraBinary(
                tempDir, "large_rank.lora", 0x41524F4C, 1, 1, rank, alpha, 2,
                ("blk.0.ffn_down.lora_a.weight", inDim, rank, a),
                ("blk.0.ffn_down.lora_b.weight", rank, outDim, b));

            var weights = LoraAdapterWeights.Load(path);
            var x = new float[inDim];
            for (int i = 0; i < inDim; i++) x[i] = (i + 1) * 0.25f;

            var yActual = new float[outDim];
            for (int d = 0; d < outDim; d++) yActual[d] = 1.0f;

            fixed (float* px = x, py = yActual)
            {
                weights.Apply(0, LoraProjection.Down, px, py, inDim, outDim);
            }

            float scaling = alpha / rank;
            var lowRank = new float[rank];
            for (int i = 0; i < inDim; i++)
                for (int r = 0; r < rank; r++)
                    lowRank[r] += x[i] * a[i * rank + r];

            var yExpected = new float[outDim];
            for (int d = 0; d < outDim; d++) yExpected[d] = 1.0f;
            for (int r = 0; r < rank; r++)
                for (int d = 0; d < outDim; d++)
                    yExpected[d] += scaling * lowRank[r] * b[r * outDim + d];

            for (int d = 0; d < outDim; d++)
            {
                Assert.Equal(yExpected[d], yActual[d], 1e-4f);
            }
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}
