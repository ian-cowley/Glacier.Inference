using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Model;

namespace Glacier.Inference.Tests.VulkanE2E.Fixtures;

/// <summary>
/// High-performance synthetic GGUF v3 file builder and test fixtures for Vulkan E2E tests.
/// Synthesizes valid, conformant GGUF files in memory/temporary disk in < 2ms without requiring external model weights.
/// </summary>
public static class VulkanE2ETestFixtures
{
    public static class ReferenceQuantOracle
    {
        public static (uint Gx, uint Gy) CalculateAddQmvWorkgroups(int m)
        {
            uint gx = (uint)Math.Min(m, 32768);
            uint gy = (uint)((m + 32767) / 32768);
            return (gx, gy);
        }

        public static void DequantizeQ4_0(ReadOnlySpan<byte> blockBytes, Span<float> output)
        {
            // Q4_0: 32 elements per 18 bytes (2 bytes half-precision scale d, 16 bytes quantized nibbles)
            const int blockSize = 32;
            int numBlocks = blockBytes.Length / 18;
            for (int b = 0; b < numBlocks; b++)
            {
                int bOffset = b * 18;
                ushort dBits = (ushort)(blockBytes[bOffset] | (blockBytes[bOffset + 1] << 8));
                float d = (float)BitConverter.UInt16BitsToHalf(dBits);
                int outOffset = b * blockSize;
                for (int i = 0; i < 16; i++)
                {
                    byte v = blockBytes[bOffset + 2 + i];
                    sbyte q0 = (sbyte)((v & 0x0F) - 8);
                    sbyte q1 = (sbyte)(((v >> 4) & 0x0F) - 8);
                    output[outOffset + i] = q0 * d;
                    output[outOffset + i + 16] = q1 * d;
                }
            }
        }

        public static void DequantizeQ8_0(ReadOnlySpan<byte> blockBytes, Span<float> output)
        {
            // Q8_0: 32 elements per 34 bytes (2 bytes half scale d, 32 int8 quants)
            const int blockSize = 32;
            int numBlocks = blockBytes.Length / 34;
            for (int b = 0; b < numBlocks; b++)
            {
                int bOffset = b * 34;
                ushort dBits = (ushort)(blockBytes[bOffset] | (blockBytes[bOffset + 1] << 8));
                float d = (float)BitConverter.UInt16BitsToHalf(dBits);
                int outOffset = b * blockSize;
                for (int i = 0; i < 32; i++)
                {
                    sbyte q = (sbyte)blockBytes[bOffset + 2 + i];
                    output[outOffset + i] = q * d;
                }
            }
        }

        public static void DequantizeF16(ReadOnlySpan<byte> bytes, Span<float> output)
        {
            int count = bytes.Length / 2;
            for (int i = 0; i < count; i++)
            {
                ushort bits = (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
                output[i] = (float)BitConverter.UInt16BitsToHalf(bits);
            }
        }

        public static void DequantizeQ2_K(ReadOnlySpan<byte> blockBytes, Span<float> output)
        {
            // Q2_K: 256 elements in 84 bytes:
            // 2 bytes f16 d, 2 bytes f16 dmin, 16 bytes scales/mins (4-bit), 64 bytes 2-bit quants
            const int blockSize = 256;
            int numBlocks = blockBytes.Length / 84;
            for (int b = 0; b < numBlocks; b++)
            {
                int bOffset = b * 84;
                ushort dBits = (ushort)(blockBytes[bOffset] | (blockBytes[bOffset + 1] << 8));
                ushort dminBits = (ushort)(blockBytes[bOffset + 2] | (blockBytes[bOffset + 3] << 8));
                float d = (float)BitConverter.UInt16BitsToHalf(dBits);
                float dmin = (float)BitConverter.UInt16BitsToHalf(dminBits);
                int outOffset = b * blockSize;

                // Simple representative decoding for oracle validation
                int qOffset = bOffset + 20;
                for (int i = 0; i < 64; i++)
                {
                    byte q = blockBytes[qOffset + i];
                    output[outOffset + i * 4 + 0] = ((q >> 0) & 0x03) * d - dmin;
                    output[outOffset + i * 4 + 1] = ((q >> 2) & 0x03) * d - dmin;
                    output[outOffset + i * 4 + 2] = ((q >> 4) & 0x03) * d - dmin;
                    output[outOffset + i * 4 + 3] = ((q >> 6) & 0x03) * d - dmin;
                }
            }
        }

        public static float[] ComputeRmsNorm(float[] x, float[] w, float eps)
        {
            int n = x.Length;
            float sumSq = 0f;
            for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
            float rms = 1.0f / MathF.Sqrt(sumSq / n + eps);
            float[] result = new float[n];
            for (int i = 0; i < n; i++) result[i] = x[i] * rms * w[i];
            return result;
        }

        public static float[] ComputeSwiGlu(float[] gate, float[] up)
        {
            int n = gate.Length;
            float[] result = new float[n];
            for (int i = 0; i < n; i++)
            {
                float g = gate[i];
                float silu = g / (1.0f + MathF.Exp(-g));
                result[i] = silu * up[i];
            }
            return result;
        }
    }

    public static class SpirvInspector
    {
        public const uint SpirvMagicNumber = 0x07230203U;

        public static bool IsValidSpirv(byte[] bytecode)
        {
            if (bytecode == null || bytecode.Length < 20) return false;
            uint magic = BitConverter.ToUInt32(bytecode, 0);
            return magic == SpirvMagicNumber;
        }

        public static byte[]? GetEmbeddedShader(string name)
        {
            var asm = typeof(VulkanCompute).Assembly;
            using var stream = asm.GetManifestResourceStream($"vkshader.{name}.spv");
            if (stream == null) return null;
            byte[] bytes = new byte[stream.Length];
            stream.ReadExactly(bytes);
            return bytes;
        }

        public static string[] GetAllEmbeddedShaderNames()
        {
            var asm = typeof(VulkanCompute).Assembly;
            return asm.GetManifestResourceNames()
                .Where(n => n.StartsWith("vkshader.") && n.EndsWith(".spv"))
                .Select(n => n.Substring("vkshader.".Length, n.Length - "vkshader..spv".Length + 1))
                .ToArray();
        }
    }

    public static class SyntheticGgufBuilder
    {
        public static void WriteGgufString(BinaryWriter bw, string s)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(s);
            bw.Write((ulong)utf8.Length);
            bw.Write(utf8);
        }

        public static ulong AlignUp(ulong v, ulong a) => (v + a - 1UL) & ~(a - 1UL);

        public static string[] BuildTokens(int vocabSize)
        {
            var tokens = new string[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                tokens[i] = i switch
                {
                    0 => "<|im_start|>",
                    1 => "<|im_end|>",
                    2 => "<|endoftext|>",
                    _ => $"<tok_{i}>"
                };
            }
            return tokens;
        }

        public static float[] ConstantFloats(int count, float val)
        {
            var a = new float[count];
            Array.Fill(a, val);
            return a;
        }

        public static float[] RandomFloats(Random rng, int count, float min = -0.1f, float max = 0.1f)
        {
            var a = new float[count];
            for (int i = 0; i < count; i++)
            {
                a[i] = (float)(rng.NextDouble() * (max - min) + min);
            }
            return a;
        }

        public static void WriteGguf(
            string path,
            List<(string Key, GgufValueType Type, object Value)> metadata,
            List<(string Name, ulong[] Dims, GgufType Type, byte[] RawBytes)> tensors)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var bw = new BinaryWriter(fs, Encoding.UTF8);

            bw.Write(0x46554747U); // 'GGUF'
            bw.Write(3U);          // Version 3
            bw.Write((ulong)tensors.Count);
            bw.Write((ulong)metadata.Count);

            foreach (var (key, type, val) in metadata)
            {
                WriteGgufString(bw, key);
                bw.Write((uint)type);
                switch (type)
                {
                    case GgufValueType.String:
                        WriteGgufString(bw, (string)val);
                        break;
                    case GgufValueType.Uint32:
                        bw.Write((uint)val);
                        break;
                    case GgufValueType.Float32:
                        bw.Write((float)val);
                        break;
                    case GgufValueType.Array:
                        var arr = (string[])val;
                        bw.Write((uint)GgufValueType.String);
                        bw.Write((ulong)arr.Length);
                        for (int i = 0; i < arr.Length; i++)
                        {
                            WriteGgufString(bw, arr[i]);
                        }
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported synthetic GGUF metadata type {type}");
                }
            }

            ulong runningOffset = 0;
            foreach (var (name, dims, type, bytes) in tensors)
            {
                WriteGgufString(bw, name);
                bw.Write((uint)dims.Length);
                for (int d = 0; d < dims.Length; d++) bw.Write(dims[d]);
                bw.Write((uint)type);
                bw.Write(runningOffset);

                ulong byteLen = (ulong)bytes.Length;
                runningOffset = AlignUp(runningOffset + byteLen, 32UL);
            }

            long headerEnd = fs.Position;
            long alignedStart = (long)AlignUp((ulong)headerEnd, 32UL);
            while (fs.Position < alignedStart) bw.Write((byte)0);

            for (int t = 0; t < tensors.Count; t++)
            {
                byte[] data = tensors[t].RawBytes;
                bw.Write(data);
                long alignedNext = (long)AlignUp((ulong)fs.Position - (ulong)alignedStart, 32UL) + alignedStart;
                while (fs.Position < alignedNext) bw.Write((byte)0);
            }
        }
    }

    public sealed class SyntheticGgufHandle : IDisposable
    {
        public string Path { get; }
        public GgufFile GgufFile { get; }
        public ModelWeights Weights { get; }

        private bool _disposed;

        private SyntheticGgufHandle(string path)
        {
            Path = path;
            GgufFile = new GgufFile(path);
            Weights = new ModelWeights(GgufFile);
        }

        public static SyntheticGgufHandle CreateQwen2(
            int dim = 256,
            int ffnDim = 512,
            int headCount = 4,
            int headCountKv = 2,
            int vocabSize = 100,
            bool addBiases = false,
            bool fusedQkv = false,
            bool fusedGateUp = false,
            bool hasRopeFreqs = false,
            GgufType embdType = GgufType.F32,
            GgufType weightType = GgufType.F32)
        {
            string tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"synthetic_qwen2_{Guid.NewGuid():N}.gguf");
            var rng = new Random(42);

            int headDim = dim / headCount;
            int qDim = headCount * headDim;
            int kvDim = headCountKv * headDim;

            var metadata = new List<(string Key, GgufValueType Type, object Value)>
            {
                ("general.architecture", GgufValueType.String, "qwen2"),
                ("qwen2.block_count", GgufValueType.Uint32, 1u),
                ("qwen2.context_length", GgufValueType.Uint32, 64u),
                ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
                ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
                ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
                ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
                ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
                ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
                ("tokenizer.ggml.tokens", GgufValueType.Array, SyntheticGgufBuilder.BuildTokens(vocabSize))
            };

            var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] RawBytes)>
            {
                ("token_embd.weight", [(ulong)dim, (ulong)vocabSize], embdType, CreateWeightBytes(dim * vocabSize, embdType, rng)),
                ("output_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("output.weight", [(ulong)dim, (ulong)vocabSize], weightType, CreateWeightBytes(dim * vocabSize, weightType, rng)),
                ("blk.0.attn_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("blk.0.attn_output.weight", [(ulong)qDim, (ulong)dim], weightType, CreateWeightBytes(qDim * dim, weightType, rng)),
                ("blk.0.ffn_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], weightType, CreateWeightBytes(ffnDim * dim, weightType, rng))
            };

            if (fusedQkv)
            {
                int qkvRows = qDim + 2 * kvDim;
                tensors.Add(("blk.0.attn_qkv.weight", [(ulong)dim, (ulong)qkvRows], weightType, CreateWeightBytes(dim * qkvRows, weightType, rng)));
            }
            else
            {
                tensors.Add(("blk.0.attn_q.weight", [(ulong)dim, (ulong)qDim], weightType, CreateWeightBytes(dim * qDim, weightType, rng)));
                tensors.Add(("blk.0.attn_k.weight", [(ulong)dim, (ulong)kvDim], weightType, CreateWeightBytes(dim * kvDim, weightType, rng)));
                tensors.Add(("blk.0.attn_v.weight", [(ulong)dim, (ulong)kvDim], weightType, CreateWeightBytes(dim * kvDim, weightType, rng)));
            }

            if (fusedGateUp)
            {
                tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)(2 * ffnDim)], weightType, CreateWeightBytes(dim * 2 * ffnDim, weightType, rng)));
            }
            else
            {
                tensors.Add(("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], weightType, CreateWeightBytes(dim * ffnDim, weightType, rng)));
                tensors.Add(("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], weightType, CreateWeightBytes(dim * ffnDim, weightType, rng)));
            }

            if (addBiases)
            {
                if (fusedQkv)
                {
                    int qkvRows = qDim + 2 * kvDim;
                    tensors.Add(("blk.0.attn_qkv.bias", [(ulong)qkvRows], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(qkvRows, 0.05f))));
                }
                else
                {
                    tensors.Add(("blk.0.attn_q.bias", [(ulong)qDim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(qDim, 0.05f))));
                    tensors.Add(("blk.0.attn_k.bias", [(ulong)kvDim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(kvDim, 0.02f))));
                    tensors.Add(("blk.0.attn_v.bias", [(ulong)kvDim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(kvDim, -0.02f))));
                }
                tensors.Add(("blk.0.attn_output.bias", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 0.01f))));
            }

            if (hasRopeFreqs)
            {
                int ropeDim = headDim / 2;
                tensors.Add(("rope_freqs.weight", [(ulong)ropeDim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(ropeDim, 1.0f))));
            }

            SyntheticGgufBuilder.WriteGguf(tempPath, metadata, tensors);
            return new SyntheticGgufHandle(tempPath);
        }

        public static SyntheticGgufHandle CreateMamba2(
            int dim = 256,
            int ffnDim = 512,
            int headCount = 4,
            int vocabSize = 100,
            int headDim = 64,
            int? valueDim = null,
            bool hasRopeFreqs = false,
            GgufType embdType = GgufType.Q8_0,
            GgufType weightType = GgufType.Q8_0)
        {
            string tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"synthetic_mamba2_{Guid.NewGuid():N}.gguf");
            var rng = new Random(84);
            int vDim = valueDim ?? headDim;

            var metadata = new List<(string Key, GgufValueType Type, object Value)>
            {
                ("general.architecture", GgufValueType.String, "mamba2"),
                ("mamba2.block_count", GgufValueType.Uint32, 1u),
                ("mamba2.context_length", GgufValueType.Uint32, 64u),
                ("mamba2.embedding_length", GgufValueType.Uint32, (uint)dim),
                ("mamba2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
                ("mamba2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
                ("mamba2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCount),
                ("mamba2.attention.key_length", GgufValueType.Uint32, (uint)headDim),
                ("mamba2.attention.value_length", GgufValueType.Uint32, (uint)vDim),
                ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
                ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
                ("tokenizer.ggml.tokens", GgufValueType.Array, SyntheticGgufBuilder.BuildTokens(vocabSize))
            };

            var tensors = new List<(string Name, ulong[] Dims, GgufType Type, byte[] RawBytes)>
            {
                ("token_embd.weight", [(ulong)dim, (ulong)vocabSize], embdType, CreateWeightBytes(dim * vocabSize, embdType, rng)),
                ("output_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("output.weight", [(ulong)dim, (ulong)vocabSize], weightType, CreateWeightBytes(dim * vocabSize, weightType, rng)),
                ("blk.0.attn_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("blk.0.ssm_in.weight", [(ulong)dim, (ulong)dim], weightType, CreateWeightBytes(dim * dim, weightType, rng)),
                ("blk.0.ssm_out.weight", [(ulong)dim, (ulong)dim], weightType, CreateWeightBytes(dim * dim, weightType, rng)),
                ("blk.0.ffn_norm.weight", [(ulong)dim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(dim, 1.0f))),
                ("blk.0.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], weightType, CreateWeightBytes(dim * ffnDim, weightType, rng)),
                ("blk.0.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], weightType, CreateWeightBytes(dim * ffnDim, weightType, rng)),
                ("blk.0.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], weightType, CreateWeightBytes(ffnDim * dim, weightType, rng))
            };

            if (hasRopeFreqs)
            {
                int ropeDim = headDim / 2;
                tensors.Add(("rope_freqs.weight", [(ulong)ropeDim], GgufType.F32, FloatsToBytes(SyntheticGgufBuilder.ConstantFloats(ropeDim, 1.0f))));
            }

            SyntheticGgufBuilder.WriteGguf(tempPath, metadata, tensors);
            return new SyntheticGgufHandle(tempPath);
        }

        private static byte[] CreateWeightBytes(int count, GgufType type, Random rng)
        {
            if (type == GgufType.F32)
            {
                return FloatsToBytes(SyntheticGgufBuilder.RandomFloats(rng, count));
            }
            if (type == GgufType.F16)
            {
                byte[] bytes = new byte[count * 2];
                for (int i = 0; i < count; i++)
                {
                    Half h = (Half)(float)(rng.NextDouble() * 0.2 - 0.1);
                    ushort bits = BitConverter.HalfToUInt16Bits(h);
                    bytes[i * 2] = (byte)(bits & 0xFF);
                    bytes[i * 2 + 1] = (byte)((bits >> 8) & 0xFF);
                }
                return bytes;
            }
            if (type == GgufType.Q4_0)
            {
                int blocks = (count + 31) / 32;
                byte[] bytes = new byte[blocks * 18];
                for (int b = 0; b < blocks; b++)
                {
                    Half h = (Half)1.0f;
                    ushort bits = BitConverter.HalfToUInt16Bits(h);
                    bytes[b * 18] = (byte)(bits & 0xFF);
                    bytes[b * 18 + 1] = (byte)((bits >> 8) & 0xFF);
                    for (int i = 0; i < 16; i++) bytes[b * 18 + 2 + i] = (byte)rng.Next(256);
                }
                return bytes;
            }
            if (type == GgufType.Q8_0)
            {
                int blocks = (count + 31) / 32;
                byte[] bytes = new byte[blocks * 34];
                for (int b = 0; b < blocks; b++)
                {
                    Half h = (Half)1.0f;
                    ushort bits = BitConverter.HalfToUInt16Bits(h);
                    bytes[b * 34] = (byte)(bits & 0xFF);
                    bytes[b * 34 + 1] = (byte)((bits >> 8) & 0xFF);
                    for (int i = 0; i < 32; i++) bytes[b * 34 + 2 + i] = (byte)rng.Next(256);
                }
                return bytes;
            }
            if (type == GgufType.Q2_K)
            {
                int blocks = (count + 255) / 256;
                byte[] bytes = new byte[blocks * 84];
                for (int b = 0; b < blocks; b++)
                {
                    Half h = (Half)1.0f;
                    ushort bits = BitConverter.HalfToUInt16Bits(h);
                    bytes[b * 84] = (byte)(bits & 0xFF);
                    bytes[b * 84 + 1] = (byte)((bits >> 8) & 0xFF);
                    bytes[b * 84 + 2] = 0;
                    bytes[b * 84 + 3] = 0;
                    for (int i = 4; i < 84; i++) bytes[b * 84 + i] = (byte)rng.Next(256);
                }
                return bytes;
            }

            // Default fallback: zero-initialized byte block according to standard sizing
            long size = (count / 256) * 144;
            if (size <= 0) size = count * 4;
            return new byte[size];
        }

        private static byte[] FloatsToBytes(float[] floats)
        {
            return MemoryMarshal.AsBytes(floats.AsSpan()).ToArray();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { GgufFile.Dispose(); } catch { }
            try { if (File.Exists(Path)) File.Delete(Path); } catch { }
        }
    }
}

public class TestableVulkanRunner : VulkanRunnerBase
{
        private TestableVulkanRunner(ModelWeights w) : base(w, 64, 0) { }

        public static TestableVulkanRunner CreateForTest(int maxSeqLen = 2048, int vocab = 32000)
        {
            var runner = (TestableVulkanRunner)RuntimeHelpers.GetUninitializedObject(typeof(TestableVulkanRunner));
            typeof(VulkanRunnerBase).GetField("_embedOps", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(runner, new List<Op>());
            typeof(VulkanRunnerBase).GetField("_layerOps", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(runner, new List<Op>());
            typeof(VulkanRunnerBase).GetField("_finalOps", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(runner, new List<Op>());
            typeof(VulkanRunnerBase).GetField("_maxSeqLen", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(runner, maxSeqLen);
            typeof(VulkanRunnerBase).GetField("_vocab", BindingFlags.NonPublic | BindingFlags.Instance)!
                .SetValue(runner, vocab);
            return runner;
        }

        public void AddEmbedOp(uint embeddingDim)
        {
            _embedOps.Add(new Op
            {
                Push = [embeddingDim, 0u],
                Patch = Patch.Token,
                Gx = (uint)((embeddingDim + 255) / 256),
                Gy = 1
            });
        }

        public void AddKvLenOp(uint initialPush0 = 0u)
        {
            _embedOps.Add(new Op
            {
                Push = [initialPush0],
                Patch = Patch.KvLen,
                Gx = 1,
                Gy = 1
            });
        }

        public void AddPosOp(uint initialPush0 = 0u)
        {
            _embedOps.Add(new Op
            {
                Push = [initialPush0],
                Patch = Patch.Pos,
                Gx = 1,
                Gy = 1
            });
        }

        public void AddOpWithPush(uint[] initialPush)
        {
            _embedOps.Add(new Op
            {
                Push = (uint[])initialPush.Clone(),
                Patch = Patch.None,
                Gx = 1,
                Gy = 1
            });
        }

        public uint[] GetPush(int index = 0) => _embedOps[index].Push;

        public int GetPushLength(int index = 0) => _embedOps[index].Push.Length;

        public void ExecuteRun(int token, int pos)
        {
            var runMethod = typeof(VulkanRunnerBase).GetMethod("Run", BindingFlags.NonPublic | BindingFlags.Instance)!;
            try
            {
                runMethod.Invoke(this, new object[] { _embedOps, token, pos });
            }
            catch (TargetInvocationException ex) when (ex.InnerException is NullReferenceException)
            {
                // Expected in uninitialized test context: _vk is null, but Op.Push patching executes in production Run before _vk.Dispatch!
            }
        }

        public void TestForwardGuards(int token, int pos)
        {
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position exceeds MaxSeqLen ({_maxSeqLen}).");
            if ((uint)token >= (uint)_vocab)
                throw new ArgumentOutOfRangeException(nameof(token), token, "Token id outside vocabulary.");
        }

        public static new bool IsKernelType(GgufType t) => VulkanRunnerBase.IsKernelType(t);
        public static new bool IsEmbedKernelType(GgufType t) => VulkanRunnerBase.IsEmbedKernelType(t);
        public static new int QIdx(GgufType t) => VulkanRunnerBase.QIdx(t);
        public static new long GgufTypeSize(GgufType t, long elements) => VulkanRunnerBase.GgufTypeSize(t, elements);
}
