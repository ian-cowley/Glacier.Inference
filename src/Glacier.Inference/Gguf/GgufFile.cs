namespace Glacier.Inference.Gguf;

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using Glacier.Inference.Memory;

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
    private readonly List<ExtraShardMapping> _extraShards = new();
    private readonly Dictionary<string, IntPtr> _shardTensorPtrs = new();
    private readonly Dictionary<string, (string Path, long FileOffset)> _shardTensorLocations = new(StringComparer.Ordinal);
    private readonly MemoryMappedViewAccessor _accessor;
    private byte* _basePointer;
    private readonly long _fileLength;
    private bool _disposed;

    private sealed class ExtraShardMapping
    {
        public required string Path { get; init; }
        public required ulong TensorDataOffset { get; init; }
        public required int TensorCount { get; init; }
        public required string? SingleTensorName { get; init; }
        public required MemoryMappedFile Mmf { get; init; }
        public required MemoryMappedViewAccessor Acc { get; init; }
        public required IntPtr BasePtr { get; set; }
    }

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
    public int HeadCount => Math.Clamp(GetMaxMetadataInt($"{Architecture}.attention.head_count", 28), 1, 1024);
    public int HeadCountKv => Math.Clamp(GetMaxMetadataInt($"{Architecture}.attention.head_count_kv", 4), 1, 1024);
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
    public int EosTokenId => TryGetFirstMetadataInt32("tokenizer.ggml.eos_token_id", out int eos) ? eos : -1;
    public int BosTokenId => TryGetFirstMetadataInt32("tokenizer.ggml.bos_token_id", out int bos) ? bos : -1;
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
    public bool NormTopK => Architecture != "deepseek2" && Architecture != "qwen2moe" && GetMetadataBool($"{Architecture}.expert_weights_norm", Architecture != "deepseek2" && Architecture != "qwen2moe");
    public bool IsMoe => ExpertCount > 0 || Tensors.ContainsKey("blk.0.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.1.ffn_gate_exps.weight") || Tensors.ContainsKey("blk.2.ffn_gate_exps.weight");

    public bool IsHybridSsm => Architecture == "qwen35" || Architecture == "qwen4exp" || Architecture == "qwen38" || Tensors.ContainsKey("blk.0.ssm_out.weight") || Tensors.ContainsKey("blk.0.ssm_conv1d.weight");

    // NariQ Trellis-Coded Quantization (GGML_TYPE_TRL1..TRL6, nariq/FORMAT.md v1)
    public uint TrellisVersion => GetMetadataUInt32("trellis.version", 0);
    public int TrellisEmbedBands => (int)GetMetadataUInt32("trellis.embed_bands", 0);
    public int TrellisLmHeadBands => (int)GetMetadataUInt32("trellis.lm_head_bands", 0);
    public bool HasTrellisTensors => TrellisVersion > 0 || TrellisEmbedBands > 0 || TrellisLmHeadBands > 0;
    public int[]? GetTrellisDescriptor(string baseTensorName) => GetMetadataInt32Array($"trellis.desc.{baseTensorName}");
    public int FullAttentionInterval => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.full_attention_interval", 4), 1, 256);
    public int SsmConvKernel => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.conv_kernel", 4), 1, 256);
    public int SsmStateSize => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.state_size", 128), 1, 4096);
    public int SsmGroupCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.group_count", 16), 1, 1024);
    public int SsmTimeStepRank => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.time_step_rank", 48), 1, 1024);
    public int SsmInnerSize => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ssm.inner_size", 6144), 1, 131_072);

    // Qwen3.8-Flash-Next (qwen4exp): Hyper-Connections, QSA Indexer, and PLE
    public int HyperConnectionCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.hyper_connection.count", 0), 0, 64);
    public int HyperConnectionLowRank => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.hyper_connection.low_rank", 0), 0, 8192);
    public int IndexerHeadCount => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.indexer.head_count", 4), 1, 256);
    public int IndexerKeyLength => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.indexer.key_length", 128), 1, 4096);
    public int IndexerTopK => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.attention.indexer.top_k", 2048), 1, 1_048_576);
    public int[]? AttentionCompressRatios => GetMetadataInt32Array($"{Architecture}.attention.compress_ratios");
    public int[]? PleLayers => GetMetadataInt32Array($"{Architecture}.ple.layers");
    public int PleNgramSize => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ple.ngram_size", 3), 1, 32);
    public int PleHeadsPerNgram => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ple.heads_per_ngram", 8), 1, 64);
    public int PleConvKernel => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.ple.conv_kernel", 4), 1, 64);
    public int PleEosTokenId => (int)GetMetadataUInt32($"{Architecture}.ple.eos_token_id", (uint)GetMetadataUInt32("tokenizer.ggml.eos_token_id", 248044));
    public int PleImageTokenId => Metadata.ContainsKey($"{Architecture}.ple.image_token_id") ? (int)GetMetadataUInt32($"{Architecture}.ple.image_token_id", 0) : -1;
    public int PleRowDim => Math.Clamp((int)GetMetadataUInt32($"{Architecture}.embedding_length_per_layer_input", 160), 1, 8192);
    public ulong[]? PleLayerMultipliers => GetMetadataUInt64Array($"{Architecture}.ple.layer_multipliers");
    public long[]? PleHeadOffsets => GetMetadataInt64Array($"{Architecture}.ple.head_offsets");
    public long[]? PleHeadVocabSizes => GetMetadataInt64Array($"{Architecture}.ple.head_vocab_sizes");

    public string? PleTableFilePath { get; private set; }
    public long PleTableByteOffset { get; private set; }

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

        _fileLength = ResolveFileLength(filePath);
        const long MinGgufHeaderSize = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_kv_count
        if (_fileLength < MinGgufHeaderSize)
            throw new InvalidDataException($"File is too short to be a valid GGUF file ({_fileLength} bytes).");

        var primaryStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        try
        {
            _mmf = MemoryMappedFile.CreateFromFile(primaryStream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        }
        catch
        {
            primaryStream.Dispose();
            throw;
        }
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
                        throw new InvalidDataException($"Invalid GGUF general.alignment ({parsedAlign}): must be a positive power of 2.");
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

            // 5. Validate All Tensor Offsets, Bounds, and Quantized Block Alignment
            foreach (var tensor in TensorList)
            {
                ValidateTensorBoundsAndAlignment(tensor, TensorDataOffset, (ulong)_fileLength, "primary file");
            }

            // 6. Record PLE table location in primary file if present, and load additional split shards
            if (Tensors.TryGetValue("per_layer_token_embd.weight", out var primaryPle))
            {
                PleTableFilePath = filePath;
                PleTableByteOffset = (long)(TensorDataOffset + primaryPle.Offset);
            }

            LoadExtraShards(filePath);
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
        if (_shardTensorPtrs.TryGetValue(tensor.Name, out IntPtr shardPtr))
            return (byte*)shardPtr;

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

    public bool TryGetTensorFileLocation(string tensorName, out string filePath, out long fileByteOffset, out ulong byteSize)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!Tensors.TryGetValue(tensorName, out var info) || info == null)
        {
            filePath = string.Empty;
            fileByteOffset = 0;
            byteSize = 0;
            return false;
        }

        byteSize = info.GetByteSize();
        if (_shardTensorLocations.TryGetValue(tensorName, out var shardLoc))
        {
            filePath = shardLoc.Path;
            fileByteOffset = shardLoc.FileOffset;
            return true;
        }

        filePath = FilePath;
        fileByteOffset = checked((long)(TensorDataOffset + info.Offset));
        return true;
    }

    public string GetMetadataString(string key, string fallback = "") =>
        Metadata.TryGetValue(key, out var val) && val is string s ? s : fallback;

    public bool TryGetFirstMetadataInt32(string key, out int value)
    {
        value = -1;
        if (!Metadata.TryGetValue(key, out var val) || val == null)
            return false;

        if (TryConvertToInt32(val, out value))
            return true;

        if (val is List<object> list)
        {
            for (int i = 0; i < list.Count; i++)
            {
                if (TryConvertToInt32(list[i], out value))
                    return true;
            }
        }
        return false;
    }

    public IReadOnlyList<int> GetEosTokenIds()
    {
        var ids = new List<int>();
        CollectMetadataInt32s("tokenizer.ggml.eos_token_id", ids);
        CollectMetadataInt32s("tokenizer.ggml.eot_token_id", ids);
        return ids;
    }

    private void CollectMetadataInt32s(string key, List<int> destination)
    {
        if (!Metadata.TryGetValue(key, out var val) || val == null)
            return;

        if (TryConvertToInt32(val, out int scalar) && scalar >= 0 && !destination.Contains(scalar))
        {
            destination.Add(scalar);
        }
        else if (val is List<object> list)
        {
            for (int i = 0; i < list.Count; i++)
            {
                if (TryConvertToInt32(list[i], out int item) && item >= 0 && !destination.Contains(item))
                {
                    destination.Add(item);
                }
            }
        }
    }

    private static bool TryConvertToInt32(object? val, out int result)
    {
        switch (val)
        {
            case uint u32 when u32 <= int.MaxValue:
                result = (int)u32;
                return true;
            case int i32 when i32 >= 0:
                result = i32;
                return true;
            case ulong u64 when u64 <= int.MaxValue:
                result = (int)u64;
                return true;
            case long i64 when i64 >= 0 && i64 <= int.MaxValue:
                result = (int)i64;
                return true;
            default:
                result = -1;
                return false;
        }
    }

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

    private int GetMaxMetadataInt(string key, int fallback)
    {
        if (!Metadata.TryGetValue(key, out var val) || val == null)
            return fallback;

        if (val is uint u32) return u32 <= int.MaxValue ? (int)u32 : int.MaxValue;
        if (val is int i32 && i32 >= 0) return i32;
        if (val is ulong u64 && u64 <= int.MaxValue) return (int)u64;
        if (val is List<object> list)
        {
            int max = 0;
            for (int i = 0; i < list.Count; i++)
            {
                int item = Convert.ToInt32(list[i]);
                if (item > max) max = item;
            }
            if (max > 0) return max;
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

    public int[]? GetMetadataInt32Array(string key)
    {
        var list = GetMetadataList(key);
        if (list == null) return null;
        var arr = new int[list.Count];
        for (int i = 0; i < list.Count; i++) arr[i] = Convert.ToInt32(list[i]);
        return arr;
    }

    public long[]? GetMetadataInt64Array(string key)
    {
        var list = GetMetadataList(key);
        if (list == null) return null;
        var arr = new long[list.Count];
        for (int i = 0; i < list.Count; i++) arr[i] = Convert.ToInt64(list[i]);
        return arr;
    }

    public ulong[]? GetMetadataUInt64Array(string key)
    {
        var list = GetMetadataList(key);
        if (list == null) return null;
        var arr = new ulong[list.Count];
        for (int i = 0; i < list.Count; i++) arr[i] = Convert.ToUInt64(list[i]);
        return arr;
    }

    /// <summary>
    /// If <c>per_layer_token_embd.weight</c> resides in its own exclusive split-GGUF shard, closes that shard's
    /// memory mapping so Windows/Linux unbuffered Direct I/O does not contend with an active mmap section.
    /// </summary>
    public bool DetachExclusivePleShardMapping()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int i = _extraShards.Count - 1; i >= 0; i--)
        {
            var shard = _extraShards[i];
            if (shard.TensorCount == 1 && shard.SingleTensorName == "per_layer_token_embd.weight")
            {
                if (shard.BasePtr != IntPtr.Zero)
                {
                    try { shard.Acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
                    catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[GgufFile] Shard ReleasePointer failed: {ex.Message}");
                    }
                    shard.BasePtr = IntPtr.Zero;
                }
                shard.Acc.Dispose();
                shard.Mmf.Dispose();
                _shardTensorPtrs.Remove("per_layer_token_embd.weight");
                _shardTensorLocations.Remove("per_layer_token_embd.weight");
                _extraShards.RemoveAt(i);
                return true;
            }
        }
        return false;
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
                catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
                {
                    System.Diagnostics.Trace.TraceWarning($"[GgufFile] Base ReleasePointer failed: {ex.Message}");
                }
                _basePointer = null;
            }
            _accessor?.Dispose();
            _mmf?.Dispose();
            foreach (var shard in _extraShards)
            {
                if (shard.BasePtr != IntPtr.Zero)
                {
                    try { shard.Acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
                    catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"[GgufFile] Extra shard ReleasePointer failed: {ex.Message}");
                    }
                }
                shard.Acc.Dispose();
                shard.Mmf.Dispose();
            }
            _extraShards.Clear();
            _disposed = true;
        }
    }

    private static long ResolveFileLength(string path)
    {
        var fileInfo = new FileInfo(path);
        long length = fileInfo.Length;
        // On Linux/macOS FileInfo.Length of a symlink (e.g. HuggingFace snapshots -> blobs) is the link size, not the target size.
        // FileStream.Length follows the link, so always prefer it.
        if (length == 0 || fileInfo.LinkTarget != null)
        {
            try
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
                length = fs.Length;
            }
            catch
            {
                if (fileInfo.LinkTarget != null && File.Exists(fileInfo.LinkTarget))
                {
                    length = new FileInfo(fileInfo.LinkTarget).Length;
                }
            }
        }
        return length;
    }

    private void LoadExtraShards(string firstShardPath)
    {
        var m = System.Text.RegularExpressions.Regex.Match(firstShardPath, @"^(.*-)(\d{5})(-of-)(\d{5})(\.gguf)$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (!m.Success) return;

        int splitCount = Metadata.TryGetValue("split.count", out var sc) ? Convert.ToInt32(sc) : int.Parse(m.Groups[4].Value);
        if (splitCount <= 1) return;

        const long MinGgufHeaderSize = 4 + 4 + 8 + 8;
        string prefix = m.Groups[1].Value, mid = m.Groups[3].Value, totalStr = m.Groups[4].Value, ext = m.Groups[5].Value;
        for (int s = 2; s <= splitCount; s++)
        {
            string shardPath = $"{prefix}{s:D5}{mid}{totalStr}{ext}";
            if (!File.Exists(shardPath))
                throw new FileNotFoundException($"Missing split-GGUF shard {s}/{splitCount}: {shardPath}", shardPath);

            long shardLen = ResolveFileLength(shardPath);
            if (shardLen < MinGgufHeaderSize)
                throw new InvalidDataException($"Split-GGUF shard '{shardPath}' is too short ({shardLen} bytes).");

            MemoryMappedFile? mmf = null;
            MemoryMappedViewAccessor? acc = null;
            byte* basePtr = null;
            bool ownershipTransferred = false;

            var shardStream = new FileStream(shardPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            try
            {
                mmf = MemoryMappedFile.CreateFromFile(shardStream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
            }
            catch
            {
                shardStream.Dispose();
                throw;
            }

            try
            {
                acc = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
                acc.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

                byte* ptr = basePtr;
                byte* endPtr = basePtr + shardLen;
                uint magic = ReadUInt32(ref ptr, endPtr);
                if (magic != 0x46554747) throw new InvalidDataException($"Invalid GGUF magic in shard {shardPath}.");
                uint ver = ReadUInt32(ref ptr, endPtr);
                if (ver < 2 || ver > 3) throw new NotSupportedException($"Unsupported GGUF version {ver} in shard {shardPath}.");

                ulong tCount = ReadUInt64(ref ptr, endPtr);
                ulong kvCount = ReadUInt64(ref ptr, endPtr);
                ulong remainingBytes = (ulong)(endPtr - ptr);
                if (kvCount > remainingBytes / 8 || tCount > remainingBytes / 16)
                    throw new InvalidDataException($"Shard '{shardPath}' header metadata counts exceed available bytes (KV={kvCount}, Tensors={tCount}).");

                uint shardAlign = Alignment;
                for (ulong i = 0; i < kvCount; i++)
                {
                    string key = ReadString(ref ptr, endPtr);
                    uint vType = ReadUInt32(ref ptr, endPtr);
                    object val = ReadValue(ref ptr, endPtr, (GgufValueType)vType, 0);
                    if (key == "general.alignment")
                    {
                        uint parsedAlign = 0;
                        if (val is uint u32) parsedAlign = u32;
                        else if (val is int i32 && i32 > 0) parsedAlign = (uint)i32;
                        else if (val is ulong u64 && u64 <= uint.MaxValue) parsedAlign = (uint)u64;

                        if (parsedAlign > 0 && (parsedAlign & (parsedAlign - 1)) == 0)
                            shardAlign = parsedAlign;
                        else if (parsedAlign > 0)
                            throw new InvalidDataException($"Invalid GGUF general.alignment ({parsedAlign}) in shard '{shardPath}': must be a positive power of 2.");
                    }
                }

                int tCountInt = checked((int)tCount);
                var shardTensors = new List<GgufTensorInfo>(tCountInt);
                for (ulong i = 0; i < tCount; i++)
                {
                    string name = ReadString(ref ptr, endPtr);
                    uint nDims = ReadUInt32(ref ptr, endPtr);
                    if (nDims > MaxDimensionsCount)
                        throw new InvalidDataException($"Tensor '{name}' in shard '{shardPath}' exceeds maximum supported dimensions ({nDims} > {MaxDimensionsCount}).");

                    ulong[] dims = new ulong[nDims];
                    for (uint d = 0; d < nDims; d++) dims[d] = ReadUInt64(ref ptr, endPtr);
                    var tType = (GgufType)ReadUInt32(ref ptr, endPtr);
                    ulong offset = ReadUInt64(ref ptr, endPtr);
                    var info = new GgufTensorInfo { Name = name, DimensionsCount = nDims, Dimensions = dims, Type = tType, Offset = offset };
                    shardTensors.Add(info);
                    Tensors[name] = info;
                    TensorList.Add(info);
                }

                ulong headerBytes = (ulong)(ptr - basePtr);
                if (shardAlign == 0) shardAlign = 32;
                ulong rem = headerBytes % shardAlign;
                ulong dataOff = rem == 0 ? headerBytes : headerBytes + (shardAlign - rem);

                _extraShards.Add(new ExtraShardMapping
                {
                    Path = shardPath,
                    TensorDataOffset = dataOff,
                    TensorCount = shardTensors.Count,
                    SingleTensorName = shardTensors.Count == 1 ? shardTensors[0].Name : null,
                    Mmf = mmf,
                    Acc = acc,
                    BasePtr = (IntPtr)basePtr
                });
                ownershipTransferred = true;

                foreach (var t in shardTensors)
                {
                    ValidateTensorBoundsAndAlignment(t, dataOff, (ulong)shardLen, $"shard '{Path.GetFileName(shardPath)}'");
                    _shardTensorPtrs[t.Name] = (IntPtr)(basePtr + dataOff + t.Offset);
                    _shardTensorLocations[t.Name] = (shardPath, checked((long)(dataOff + t.Offset)));
                    if (t.Name == "per_layer_token_embd.weight")
                    {
                        PleTableFilePath = shardPath;
                        PleTableByteOffset = (long)(dataOff + t.Offset);
                    }
                }
            }
            catch
            {
                if (!ownershipTransferred)
                {
                    if (basePtr != null && acc != null)
                    {
                        try { acc.SafeMemoryMappedViewHandle.ReleasePointer(); }
                        catch (Exception ex) when (ex is ObjectDisposedException or InvalidOperationException)
                        {
                            System.Diagnostics.Trace.TraceWarning($"[GgufFile] Shard open rollback ReleasePointer failed: {ex.Message}");
                        }
                    }
                    acc?.Dispose();
                    mmf?.Dispose();
                }
                throw;
            }
        }
    }

    private static void ValidateTensorBoundsAndAlignment(
        GgufTensorInfo tensor,
        ulong dataOffset,
        ulong fileLength,
        string contextLabel)
    {
        ulong elemCount = tensor.Dimensions.Length == 0 ? 0UL : 1UL;
        try
        {
            checked
            {
                for (int d = 0; d < tensor.Dimensions.Length; d++)
                    elemCount *= tensor.Dimensions[d];
            }
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' dimensions overflow element count in {contextLabel}.", ex);
        }

        if (GgufTypes.TryGetBlockInfo(tensor.Type, out int blockSize, out _) && blockSize > 1)
        {
            if (tensor.Dimensions.Length == 0 || tensor.Dimensions[0] == 0 || (tensor.Dimensions[0] % (ulong)blockSize) != 0)
            {
                ulong innerDim = tensor.Dimensions.Length > 0 ? tensor.Dimensions[0] : 0UL;
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' ({tensor.Type}) innermost dimension ({innerDim}) must be a positive multiple of block size {blockSize} in {contextLabel}.");
            }
        }

        ulong byteSize;
        try
        {
            checked
            {
                if (GgufTypes.TryGetBlockInfo(tensor.Type, out int bs, out int bb))
                {
                    byteSize = (elemCount / (ulong)bs) * (ulong)bb;
                }
                else
                {
                    byteSize = elemCount * 2UL;
                }
            }
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' byte size overflows in {contextLabel}.", ex);
        }

        if (tensor.Offset > fileLength ||
            dataOffset > fileLength ||
            tensor.Offset > fileLength - dataOffset ||
            byteSize > fileLength - dataOffset - tensor.Offset)
        {
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' bounds exceed {contextLabel} size ({fileLength}).");
        }
    }
}
