namespace Glacier.Inference.Gguf;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;

/// <summary>
/// High-performance zero-copy memory-mapped GGUF model reader.
/// Maps multi-gigabyte models directly into virtual address space in sub-100ms cold time.
/// Hardened with bounds checking and defense-in-depth against malformed inputs.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private const int MaxNestingDepth = 4;
    private const int MaxDimensionsCount = 4;
    private const ulong MaxStringLength = 1024 * 1024; // 1 MB limit for keys / strings
    private const ulong MaxArrayElements = 1_000_000;

    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _basePointer;
    private readonly long _fileLength;
    private bool _disposed;

    public string FilePath { get; }
    public uint Version { get; }
    public ulong TensorCount { get; }
    public ulong MetadataKvCount { get; }
    public ulong TensorDataOffset { get; }
    public ulong MetadataEndOffset { get; }
    public byte* BasePointer => _basePointer;
    public uint Alignment { get; } = 32;

    public Dictionary<string, object> Metadata { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, GgufTensorInfo> Tensors { get; } = new(StringComparer.Ordinal);
    public List<GgufTensorInfo> TensorList { get; } = [];

    public string Architecture => GetMetadataString("general.architecture", "llama");
    public int BlockCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.block_count", 28), 0, 1024);
    public int ContextLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.context_length", 32768), 1, 1_048_576);
    public int EmbeddingLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.embedding_length", 3584), 1, 131_072);
    public int FeedForwardLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.feed_forward_length", 18944), 1, 524_288);
    public int HeadCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.head_count", 28), 1, 1024);
    public int HeadCountKv => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.head_count_kv", 4), 1, 1024);
    public int HeadDim
    {
        get
        {
            uint keyLen = GetMetadataUInt32($"{Architecture}.attention.key_length", 0);
            if (keyLen > 0) return Math.Clamp((int)keyLen, 1, 4096);
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
            if (valLen > 0) return Math.Clamp((int)valLen, 1, 4096);
            return HeadDim;
        }
    }
    public int KvLoraRank => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.kv_lora_rank", 512), 0, 8192);
    public int RopeDimensionCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.rope.dimension_count", (uint)HeadDim), 0, 4096);
    public int QkNopeHeadDim => HeadDim > RopeDimensionCount ? HeadDim - RopeDimensionCount : 128;
    public string RopeScalingType => GetMetadataString($"{Architecture}.rope.scaling.type", "");
    public float RopeScalingFactor => GetMetadataSingle($"{Architecture}.rope.scaling.factor", 1.0f);
    public int RopeScalingOriginalContextLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.rope.scaling.original_context_length", 4096), 1, 1_048_576);
    public float RopeScalingYarnLogMultiplier => GetMetadataSingle($"{Architecture}.rope.scaling.yarn_log_multiplier", 0.0707f);
    public bool IsMla => Architecture == "deepseek2" || Tensors.ContainsKey("blk.0.attn_kv_a_mqa.weight") || Tensors.ContainsKey("blk.1.attn_kv_a_mqa.weight");

    public int ExpertCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.expert_count", 0), 0, 1024);
    public int ExpertUsedCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.expert_used_count", 0), 0, 1024);
    public int ExpertFeedForwardLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.expert_feed_forward_length", 0), 0, 1_048_576);
    public int ExpertSharedCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.expert_shared_count", 0), 0, 1024);
    public int LeadingDenseBlockCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.leading_dense_block_count", 0), 0, 1024);
    public bool NormTopK => Architecture != "deepseek2" && GetMetadataBool($"{Architecture}.expert_weights_norm", Architecture != "deepseek2");
    public bool IsMoe => ExpertCount > 0 || Tensors.ContainsKey("blk.0.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.1.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.2.ffn_gate_exps.weight");

    public bool IsHybridSsm => Architecture == "qwen35" || Tensors.ContainsKey("blk.0.ssm_out.weight") || Tensors.ContainsKey("blk.0.ssm_conv1d.weight");
    public int FullAttentionInterval => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.full_attention_interval", 4), 1, 256);
    public int SsmConvKernel => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.conv_kernel", 4), 1, 256);
    public int SsmStateSize => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.state_size", 128), 1, 4096);
    public int SsmGroupCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.group_count", 16), 1, 1024);
    public int SsmTimeStepRank => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.time_step_rank", 48), 1, 1024);
    public int SsmInnerSize => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.inner_size", 6144), 1, 131_072);

    public float FinalLogitSoftcapping => GetMetadataSingle($"{Architecture}.final_logit_softcapping", 0f);
    public float AttnLogitSoftcapping => GetMetadataSingle($"{Architecture}.attention.logit_softcapping",
        GetMetadataSingle($"{Architecture}.attn_logit_softcapping", 0f));
    public int SlidingWindow => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.sliding_window", 0), 0, 1_048_576);
    public float RopeFreqBaseSwa => GetMetadataSingle($"{Architecture}.rope.freq_base_swa", 10000.0f);
    public int KeyLengthSwa => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.key_length_swa", 256), 1, 4096);
    public int ValueLengthSwa => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.value_length_swa", 256), 1, 4096);
    public int RopeDimensionCountSwa => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.rope.dimension_count_swa", 256), 0, 4096);

    public bool[]? SlidingWindowPattern
    {
        get
        {
            var list = GetMetadataList($"{Architecture}.attention.sliding_window_pattern");
            if (list == null) return null;
            var pattern = new bool[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i] is bool b) pattern[i] = b;
                else if (list[i] is uint u) pattern[i] = u != 0;
                else if (list[i] is int iv) pattern[i] = iv != 0;
            }
            return pattern;
        }
    }

    public int[]? HeadCountKvPattern
    {
        get
        {
            var list = GetMetadataList($"{Architecture}.attention.head_count_kv");
            if (list == null) return null;
            var arr = new int[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i] is uint u) arr[i] = (int)u;
                else if (list[i] is int iv) arr[i] = iv;
                else if (list[i] is ulong ul) arr[i] = (int)ul;
            }
            return arr;
        }
    }

    public static GgufFile Open(string filePath) => new(filePath);

    public GgufFile(string filePath)
    {
        FilePath = filePath;
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);

        var fileInfo = new FileInfo(filePath);
        _fileLength = fileInfo.Length;
        const long MinGgufHeaderSize = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_kv_count
        if (_fileLength < MinGgufHeaderSize)
            throw new InvalidDataException($"File is too short to be a valid GGUF file ({_fileLength} bytes).");

        _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _accessor = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);

            byte* ptr = _basePointer;
            byte* endPtr = _basePointer + _fileLength;

            // 1. Validate Header
            uint magic = ReadUInt32(ref ptr, endPtr);
            if (magic != 0x46554747) // 'GGUF'
                throw new InvalidDataException($"Invalid GGUF magic 0x{magic:X8}. Expected 0x46554747 ('GGUF').");

            Version = ReadUInt32(ref ptr, endPtr);
            if (Version < 2 || Version > 3)
                throw new NotSupportedException($"Unsupported GGUF version {Version}. Only v2 and v3 are supported.");

            TensorCount = ReadUInt64(ref ptr, endPtr);
            MetadataKvCount = ReadUInt64(ref ptr, endPtr);

            // Upper bound sanity checks against total file size
            ulong remainingBytes = (ulong)(endPtr - ptr);
            if (MetadataKvCount > remainingBytes / 8 || TensorCount > remainingBytes / 16)
            {
                throw new InvalidDataException($"Header metadata counts exceed available bytes in file (KV={MetadataKvCount}, Tensors={TensorCount}).");
            }

            // 2. Parse Key-Value Metadata
            for (ulong i = 0; i < MetadataKvCount; i++)
            {
                string key = ReadString(ref ptr, endPtr);
                uint vType = ReadUInt32(ref ptr, endPtr);
                object val = ReadValue(ref ptr, endPtr, (GgufValueType)vType, depth: 0);
                Metadata[key] = val;

                if (key == "general.alignment")
                {
                    uint parsedAlign = 0;
                    if (val is uint u32) parsedAlign = u32;
                    else if (val is int i32 && i32 > 0) parsedAlign = (uint)i32;
                    else if (val is ulong u64 && u64 <= uint.MaxValue) parsedAlign = (uint)u64;

                    if (parsedAlign > 0 && (parsedAlign & (parsedAlign - 1)) == 0)
                    {
                        Alignment = parsedAlign;
                    }
                    else if (parsedAlign > 0)
                    {
                        Alignment = parsedAlign;
                    }
                }
            }

            MetadataEndOffset = (ulong)(ptr - _basePointer);

            // 3. Parse Tensor Directory
            TensorList.Capacity = (int)Math.Min((ulong)int.MaxValue, TensorCount);
            for (ulong i = 0; i < TensorCount; i++)
            {
                string tName = ReadString(ref ptr, endPtr);
                uint nDims = ReadUInt32(ref ptr, endPtr);

                if (nDims > MaxDimensionsCount)
                    throw new InvalidDataException($"Tensor '{tName}' exceeds maximum supported dimensions ({nDims} > {MaxDimensionsCount}).");

                var dims = new ulong[nDims];
                for (uint d = 0; d < nDims; d++)
                {
                    dims[d] = ReadUInt64(ref ptr, endPtr);
                }

                uint ggmlType = ReadUInt32(ref ptr, endPtr);
                ulong offset = ReadUInt64(ref ptr, endPtr);

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
            if (Alignment == 0) Alignment = 32;
            ulong rem = headerBytes % Alignment;
            TensorDataOffset = rem == 0 ? headerBytes : headerBytes + (Alignment - rem);

            if (TensorCount > 0 && TensorDataOffset > (ulong)_fileLength)
                throw new InvalidDataException($"Tensor data offset ({TensorDataOffset}) exceeds file size ({_fileLength}).");

            // 5. Validate All Tensor Offsets and Bounds
            foreach (var tensor in TensorList)
            {
                ulong byteSize = tensor.GetByteSize();
                ulong endOffset = TensorDataOffset + tensor.Offset + byteSize;

                if (tensor.Offset > (ulong)_fileLength || endOffset < TensorDataOffset || endOffset > (ulong)_fileLength)
                {
                    throw new InvalidDataException($"Tensor '{tensor.Name}' bounds [{TensorDataOffset + tensor.Offset}..{endOffset}] exceed file size ({_fileLength}).");
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public byte* GetTensorPointer(GgufTensorInfo tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ObjectDisposedException.ThrowIf(_disposed, this);

        ulong byteSize = tensor.GetByteSize();
        ulong endOffset = TensorDataOffset + tensor.Offset + byteSize;
        if (endOffset > (ulong)_fileLength)
            throw new InvalidDataException($"Tensor '{tensor.Name}' bounds exceed file bounds.");

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
            if (val is int i32 && i32 >= 0) return (uint)i32;
            if (val is ulong u64 && u64 <= uint.MaxValue) return (uint)u64;
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

    public List<object>? GetMetadataList(string key)
    {
        if (Metadata.TryGetValue(key, out var val) && val is List<object> list)
            return list;
        return null;
    }

    private static void EnsureReadable(byte* ptr, byte* endPtr, ulong bytes)
    {
        if ((ulong)(endPtr - ptr) < bytes)
            throw new InvalidDataException("Unexpected end of file while parsing GGUF data.");
    }

    private static uint ReadUInt32(ref byte* ptr, byte* endPtr)
    {
        EnsureReadable(ptr, endPtr, 4);
        uint v = *(uint*)ptr;
        ptr += 4;
        return v;
    }

    private static ulong ReadUInt64(ref byte* ptr, byte* endPtr)
    {
        EnsureReadable(ptr, endPtr, 8);
        ulong v = *(ulong*)ptr;
        ptr += 8;
        return v;
    }

    private static string ReadString(ref byte* ptr, byte* endPtr)
    {
        ulong len = ReadUInt64(ref ptr, endPtr);
        if (len > MaxStringLength)
            throw new InvalidDataException($"String length ({len}) exceeds maximum limit ({MaxStringLength}).");

        EnsureReadable(ptr, endPtr, len);
        string s = Encoding.UTF8.GetString(ptr, (int)len);
        ptr += len;
        return s;
    }

    private static object ReadValue(ref byte* ptr, byte* endPtr, GgufValueType type, int depth)
    {
        if (depth > MaxNestingDepth)
            throw new InvalidDataException($"Maximum nesting depth ({MaxNestingDepth}) exceeded in GGUF metadata.");

        switch (type)
        {
            case GgufValueType.Uint8:
                EnsureReadable(ptr, endPtr, 1);
                byte u8 = *ptr; ptr += 1; return u8;
            case GgufValueType.Int8:
                EnsureReadable(ptr, endPtr, 1);
                sbyte i8 = *(sbyte*)ptr; ptr += 1; return i8;
            case GgufValueType.Uint16:
                EnsureReadable(ptr, endPtr, 2);
                ushort u16 = *(ushort*)ptr; ptr += 2; return u16;
            case GgufValueType.Int16:
                EnsureReadable(ptr, endPtr, 2);
                short i16 = *(short*)ptr; ptr += 2; return i16;
            case GgufValueType.Uint32:
                EnsureReadable(ptr, endPtr, 4);
                uint u32 = *(uint*)ptr; ptr += 4; return u32;
            case GgufValueType.Int32:
                EnsureReadable(ptr, endPtr, 4);
                int i32 = *(int*)ptr; ptr += 4; return i32;
            case GgufValueType.Float32:
                EnsureReadable(ptr, endPtr, 4);
                float f32 = *(float*)ptr; ptr += 4; return f32;
            case GgufValueType.Bool:
                EnsureReadable(ptr, endPtr, 1);
                bool b = *ptr != 0; ptr += 1; return b;
            case GgufValueType.String:
                return ReadString(ref ptr, endPtr);
            case GgufValueType.Array:
                uint elemType = ReadUInt32(ref ptr, endPtr);
                ulong count = ReadUInt64(ref ptr, endPtr);
                if (count > MaxArrayElements)
                    throw new InvalidDataException($"Array count ({count}) exceeds maximum allowed ({MaxArrayElements}).");

                ulong remainingBytes = (ulong)(endPtr - ptr);
                if (count > remainingBytes)
                    throw new InvalidDataException($"Array element count ({count}) exceeds remaining file bytes ({remainingBytes}).");

                var list = new List<object>((int)count);
                for (ulong i = 0; i < count; i++)
                {
                    list.Add(ReadValue(ref ptr, endPtr, (GgufValueType)elemType, depth + 1));
                }
                return list;
            case GgufValueType.Uint64:
                EnsureReadable(ptr, endPtr, 8);
                ulong u64 = *(ulong*)ptr; ptr += 8; return u64;
            case GgufValueType.Int64:
                EnsureReadable(ptr, endPtr, 8);
                long i64 = *(long*)ptr; ptr += 8; return i64;
            case GgufValueType.Float64:
                EnsureReadable(ptr, endPtr, 8);
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
                try
                {
                    _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                }
                catch { }
                _basePointer = null;
            }
            _accessor?.Dispose();
            _mmf?.Dispose();
            _disposed = true;
        }
    }
}
