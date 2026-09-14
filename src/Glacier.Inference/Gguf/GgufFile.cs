namespace Glacier.Inference.Gguf;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;

/// <summary>
/// High-performance zero-copy memory-mapped GGUF model reader.
/// Maps multi-gigabyte models directly into virtual address space in sub-100ms cold time.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _basePointer;
    private bool _disposed;

    public string FilePath { get; }
    public uint Version { get; }
    public ulong TensorCount { get; }
    public ulong MetadataKvCount { get; }
    public ulong TensorDataOffset { get; }
    public uint Alignment { get; } = 32;

    public Dictionary<string, object> Metadata { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, GgufTensorInfo> Tensors { get; } = new(StringComparer.Ordinal);
    public List<GgufTensorInfo> TensorList { get; } = [];

    public string Architecture => GetMetadataString("general.architecture", "llama");
    public int BlockCount => (int)GetMetadataUInt32($"{Architecture}.block_count", 28);
    public int ContextLength => (int)GetMetadataUInt32($"{Architecture}.context_length", 32768);
    public int EmbeddingLength => (int)GetMetadataUInt32($"{Architecture}.embedding_length", 3584);
    public int FeedForwardLength => (int)GetMetadataUInt32($"{Architecture}.feed_forward_length", 18944);
    public int HeadCount => (int)GetMetadataUInt32($"{Architecture}.attention.head_count", 28);
    public int HeadCountKv => (int)GetMetadataUInt32($"{Architecture}.attention.head_count_kv", 4);
    public int HeadDim
    {
        get
        {
            uint keyLen = GetMetadataUInt32($"{Architecture}.attention.key_length", 0);
            if (keyLen > 0) return (int)keyLen;
            return HeadCount > 0 ? EmbeddingLength / HeadCount : 128;
        }
    }
    public float RopeFreqBase => GetMetadataSingle($"{Architecture}.rope.freq_base", 10000000.0f);
    public float RmsNormEps => GetMetadataSingle($"{Architecture}.attention.layer_norm_rms_epsilon", 1e-5f);
    public int EosTokenId => (int)GetMetadataUInt32("tokenizer.ggml.eos_token_id", 151645);
    public int BosTokenId => (int)GetMetadataUInt32("tokenizer.ggml.bos_token_id", 151643);
    public bool AddBosToken => GetMetadataBool("tokenizer.ggml.add_bos_token", false);
    public bool AddEosToken => GetMetadataBool("tokenizer.ggml.add_eos_token", false);

    public int ValueDim
    {
        get
        {
            uint valLen = GetMetadataUInt32($"{Architecture}.attention.value_length", 0);
            if (valLen > 0) return (int)valLen;
            return HeadDim;
        }
    }
    public int KvLoraRank => (int)GetMetadataUInt32($"{Architecture}.attention.kv_lora_rank", 512);
    public int RopeDimensionCount => (int)GetMetadataUInt32($"{Architecture}.rope.dimension_count", (uint)HeadDim);
    public int QkNopeHeadDim => HeadDim > RopeDimensionCount ? HeadDim - RopeDimensionCount : 128;
    public string RopeScalingType => GetMetadataString($"{Architecture}.rope.scaling.type", "");
    public float RopeScalingFactor => GetMetadataSingle($"{Architecture}.rope.scaling.factor", 1.0f);
    public int RopeScalingOriginalContextLength => (int)GetMetadataUInt32($"{Architecture}.rope.scaling.original_context_length", 4096);
    public float RopeScalingYarnLogMultiplier => GetMetadataSingle($"{Architecture}.rope.scaling.yarn_log_multiplier", 0.0707f);
    public bool IsMla => Architecture == "deepseek2" || Tensors.ContainsKey("blk.0.attn_kv_a_mqa.weight") || Tensors.ContainsKey("blk.1.attn_kv_a_mqa.weight");

    public int ExpertCount => (int)GetMetadataUInt32($"{Architecture}.expert_count", 0);
    public int ExpertUsedCount => (int)GetMetadataUInt32($"{Architecture}.expert_used_count", 0);
    public int ExpertFeedForwardLength => (int)GetMetadataUInt32($"{Architecture}.expert_feed_forward_length", 0);
    public int ExpertSharedCount => (int)GetMetadataUInt32($"{Architecture}.expert_shared_count", 0);
    public int LeadingDenseBlockCount => (int)GetMetadataUInt32($"{Architecture}.leading_dense_block_count", 0);
    public bool NormTopK => Architecture != "deepseek2" && GetMetadataBool($"{Architecture}.expert_weights_norm", Architecture != "deepseek2");
    public bool IsMoe => ExpertCount > 0 || Tensors.ContainsKey("blk.0.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.1.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.2.ffn_gate_exps.weight");

    public static GgufFile Open(string filePath) => new(filePath);

    public GgufFile(string filePath)
    {
        FilePath = filePath;
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);

        var fileInfo = new FileInfo(filePath);
        _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);

        byte* ptr = _basePointer;
        byte* endPtr = _basePointer + fileInfo.Length;

        // 1. Validate Header
        uint magic = *(uint*)ptr;
        ptr += 4;
        if (magic != 0x46554747) // 'GGUF'
            throw new InvalidDataException($"Invalid GGUF magic 0x{magic:X8}. Expected 0x46554747 ('GGUF').");

        Version = *(uint*)ptr;
        ptr += 4;
        if (Version < 2 || Version > 3)
            throw new NotSupportedException($"Unsupported GGUF version {Version}. Only v2 and v3 are supported.");

        TensorCount = *(ulong*)ptr;
        ptr += 8;
        MetadataKvCount = *(ulong*)ptr;
        ptr += 8;

        // 2. Parse Key-Value Metadata
        for (ulong i = 0; i < MetadataKvCount; i++)
        {
            string key = ReadString(ref ptr);
            uint vType = *(uint*)ptr;
            ptr += 4;
            object val = ReadValue(ref ptr, (GgufValueType)vType);
            Metadata[key] = val;

            if (key == "general.alignment" && val is uint alignVal)
            {
                Alignment = alignVal;
            }
        }

        // 3. Parse Tensor Directory
        TensorList.Capacity = (int)Math.Min((ulong)int.MaxValue, TensorCount);
        for (ulong i = 0; i < TensorCount; i++)
        {
            string tName = ReadString(ref ptr);
            uint nDims = *(uint*)ptr;
            ptr += 4;

            var dims = new ulong[nDims];
            for (uint d = 0; d < nDims; d++)
            {
                dims[d] = *(ulong*)ptr;
                ptr += 8;
            }

            uint ggmlType = *(uint*)ptr;
            ptr += 4;
            ulong offset = *(ulong*)ptr;
            ptr += 8;

            var tensor = new GgufTensorInfo
            {
                Name = tName,
                DimensionsCount = nDims,
                Dimensions = dims,
                Type = (GgufType)ggmlType,
                Offset = offset
            };

            Tensors[tName] = tensor;
            TensorList.Add(tensor);
        }

        // 4. Calculate Aligned Tensor Data Offset
        ulong headerBytes = (ulong)(ptr - _basePointer);
        ulong rem = headerBytes % Alignment;
        TensorDataOffset = rem == 0 ? headerBytes : headerBytes + (Alignment - rem);
    }

    public byte* GetTensorPointer(GgufTensorInfo tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _basePointer + TensorDataOffset + tensor.Offset;
    }

    public byte* GetTensorPointer(string tensorName)
    {
        if (!Tensors.TryGetValue(tensorName, out var tensor))
            throw new KeyNotFoundException($"Tensor '{tensorName}' not found in model.");
        return GetTensorPointer(tensor);
    }

    public bool TryGetTensor(string tensorName, out GgufTensorInfo? tensor) =>
        Tensors.TryGetValue(tensorName, out tensor);

    public string GetMetadataString(string key, string fallback = "") =>
        Metadata.TryGetValue(key, out var val) && val is string s ? s : fallback;

    public uint GetMetadataUInt32(string key, uint fallback = 0)
    {
        if (Metadata.TryGetValue(key, out var val))
        {
            if (val is uint u32) return u32;
            if (val is int i32) return (uint)i32;
            if (val is ulong u64) return (uint)u64;
        }
        return fallback;
    }

    public float GetMetadataSingle(string key, float fallback = 0f)
    {
        if (Metadata.TryGetValue(key, out var val))
        {
            if (val is float f) return f;
            if (val is double d) return (float)d;
        }
        return fallback;
    }

    public bool GetMetadataBool(string key, bool fallback = false)
    {
        if (Metadata.TryGetValue(key, out var val))
        {
            if (val is bool b) return b;
            if (val is uint u) return u != 0;
            if (val is int i) return i != 0;
        }
        return fallback;
    }

    private static string ReadString(ref byte* ptr)
    {
        ulong len = *(ulong*)ptr;
        ptr += 8;
        string s = Encoding.UTF8.GetString(ptr, (int)len);
        ptr += len;
        return s;
    }

    private static object ReadValue(ref byte* ptr, GgufValueType type)
    {
        switch (type)
        {
            case GgufValueType.Uint8:
                byte u8 = *ptr; ptr += 1; return u8;
            case GgufValueType.Int8:
                sbyte i8 = *(sbyte*)ptr; ptr += 1; return i8;
            case GgufValueType.Uint16:
                ushort u16 = *(ushort*)ptr; ptr += 2; return u16;
            case GgufValueType.Int16:
                short i16 = *(short*)ptr; ptr += 2; return i16;
            case GgufValueType.Uint32:
                uint u32 = *(uint*)ptr; ptr += 4; return u32;
            case GgufValueType.Int32:
                int i32 = *(int*)ptr; ptr += 4; return i32;
            case GgufValueType.Float32:
                float f32 = *(float*)ptr; ptr += 4; return f32;
            case GgufValueType.Bool:
                bool b = *ptr != 0; ptr += 1; return b;
            case GgufValueType.String:
                return ReadString(ref ptr);
            case GgufValueType.Array:
                uint elemType = *(uint*)ptr; ptr += 4;
                ulong count = *(ulong*)ptr; ptr += 8;
                var list = new List<object>((int)Math.Min(count, 1000000UL));
                for (ulong i = 0; i < count; i++)
                {
                    list.Add(ReadValue(ref ptr, (GgufValueType)elemType));
                }
                return list;
            case GgufValueType.Uint64:
                ulong u64 = *(ulong*)ptr; ptr += 8; return u64;
            case GgufValueType.Int64:
                long i64 = *(long*)ptr; ptr += 8; return i64;
            case GgufValueType.Float64:
                double f64 = *(double*)ptr; ptr += 8; return f64;
            default:
                throw new NotSupportedException($"Unknown GGUF value type {type}");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_basePointer != null)
            {
                _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _basePointer = null;
            }
            _accessor.Dispose();
            _mmf.Dispose();
            _disposed = true;
        }
    }
}
