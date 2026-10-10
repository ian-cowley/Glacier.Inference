namespace Glacier.Inference.Gguf;

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

/// <summary>
/// Extracts and serializes self-contained GGUF stage slices for distributed pipeline inference.
/// Produces 100% compliant GGUF v3 shards containing the exact layer weights, embeddings,
/// and metadata required for an individual cluster pipeline stage.
/// </summary>
public static unsafe class GgufStageSlicer
{
    /// <summary>
    /// Generates standard deterministic stage slice filename:
    /// {model_stem}.stage{stageIndex}-L{startLayer:D2}-L{endLayer:D2}.gguf
    /// </summary>
    public static string GetStageFileName(string modelPathOrName, int stageIndex, int startLayer, int endLayer)
    {
        string baseName = Path.GetFileNameWithoutExtension(modelPathOrName);
        return $"{baseName}.stage{stageIndex}-L{startLayer:D2}-L{endLayer:D2}.gguf";
    }

    /// <summary>
    /// Selects tensors required for a pipeline stage spanning startLayer..endLayer.
    /// </summary>
    public static List<GgufTensorInfo> SelectStageTensors(
        GgufFile source,
        int startLayer,
        int endLayer,
        bool isFirstStage,
        bool isLastStage)
    {
        ArgumentNullException.ThrowIfNull(source);

        var selected = new List<GgufTensorInfo>();
        bool hasDedicatedOutputWeight = source.Tensors.ContainsKey("output.weight");

        foreach (var t in source.TensorList)
        {
            if (t.Name.StartsWith("blk."))
            {
                int dot1 = 4;
                int dot2 = t.Name.IndexOf('.', dot1);
                if (dot2 > dot1 && int.TryParse(t.Name.AsSpan(dot1, dot2 - dot1), out int layerIdx))
                {
                    if (layerIdx >= startLayer && layerIdx <= endLayer)
                    {
                        selected.Add(t);
                    }
                }
                continue;
            }

            // Global embedding & RoPE frequencies for root stage
            if (isFirstStage)
            {
                if (t.Name == "token_embd.weight" || t.Name == "rope_freqs.weight")
                {
                    selected.Add(t);
                    continue;
                }
            }

            // Global final normalization and LM head for final stage
            if (isLastStage)
            {
                if (t.Name == "output_norm.weight" || t.Name == "norm.weight" || t.Name == "output.weight")
                {
                    selected.Add(t);
                    continue;
                }

                // If tied embedding model (no output.weight), the final stage requires token_embd.weight
                if (!hasDedicatedOutputWeight && t.Name == "token_embd.weight" && !isFirstStage)
                {
                    selected.Add(t);
                    continue;
                }
            }
        }

        return selected;
    }

    /// <summary>
    /// Calculates the exact byte size of a stage slice file before generation.
    /// </summary>
    public static ulong CalculateStageByteSize(
        GgufFile source,
        int startLayer,
        int endLayer,
        bool isFirstStage,
        bool isLastStage)
    {
        var tensors = SelectStageTensors(source, startLayer, endLayer, isFirstStage, isLastStage);
        uint alignment = source.Alignment > 0 ? source.Alignment : 32;

        ulong metadataSize = source.MetadataEndOffset;
        ulong dirSize = 0;
        foreach (var t in tensors)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(t.Name);
            dirSize += 8 + (ulong)nameBytes.Length + 4 + (8 * (ulong)t.DimensionsCount) + 4 + 8;
        }

        ulong headerTotal = metadataSize + dirSize;
        ulong dataPad = (alignment - (headerTotal % alignment)) % alignment;
        ulong totalSize = headerTotal + dataPad;

        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            ulong bSize = t.GetByteSize();
            totalSize += bSize;
            if (i + 1 < tensors.Count)
            {
                ulong alignPad = (alignment - (totalSize % alignment)) % alignment;
                totalSize += alignPad;
            }
        }

        return totalSize;
    }

    /// <summary>
    /// Slices and streams stage tensors into any destination Stream.
    /// </summary>
    public static void SliceStage(
        GgufFile source,
        Stream destStream,
        int startLayer,
        int endLayer,
        bool isFirstStage,
        bool isLastStage,
        IProgress<double>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destStream);

        var tensors = SelectStageTensors(source, startLayer, endLayer, isFirstStage, isLastStage);
        uint alignment = source.Alignment > 0 ? source.Alignment : 32;

        // 1. Copy Header + Metadata KV pairs with updated tensor count
        byte[] metadataBytes = new byte[source.MetadataEndOffset];
        fixed (byte* pMeta = metadataBytes)
        {
            Buffer.MemoryCopy(source.BasePointer, pMeta, (ulong)metadataBytes.Length, (ulong)metadataBytes.Length);
        }
        BitConverter.TryWriteBytes(metadataBytes.AsSpan(8, 8), (ulong)tensors.Count);
        destStream.Write(metadataBytes, 0, metadataBytes.Length);

        // 2. Build Tensor Directory Entries
        using var dirMs = new MemoryStream();
        using var dirWriter = new BinaryWriter(dirMs, Encoding.UTF8, leaveOpen: true);

        ulong currentDataOffset = 0;
        var newOffsets = new List<ulong>(tensors.Count);

        foreach (var t in tensors)
        {
            ulong alignPad = (alignment - (currentDataOffset % alignment)) % alignment;
            currentDataOffset += alignPad;
            newOffsets.Add(currentDataOffset);
            currentDataOffset += t.GetByteSize();
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            byte[] nameUtf8 = Encoding.UTF8.GetBytes(t.Name);
            dirWriter.Write((ulong)nameUtf8.Length);
            dirWriter.Write(nameUtf8);
            dirWriter.Write(t.DimensionsCount);
            for (int d = 0; d < (int)t.DimensionsCount; d++)
            {
                dirWriter.Write(t.Dimensions[d]);
            }
            dirWriter.Write((uint)t.Type);
            dirWriter.Write(newOffsets[i]);
        }
        dirWriter.Flush();

        byte[] dirBytes = dirMs.ToArray();
        destStream.Write(dirBytes, 0, dirBytes.Length);

        // Align to TensorDataOffset
        ulong totalHeaderSize = (ulong)metadataBytes.Length + (ulong)dirBytes.Length;
        ulong dataOffsetPad = (alignment - (totalHeaderSize % alignment)) % alignment;
        if (dataOffsetPad > 0)
        {
            Span<byte> pad = stackalloc byte[(int)dataOffsetPad];
            pad.Clear();
            destStream.Write(pad);
        }

        // 3. Stream Tensor Payloads
        ulong totalPayloadBytes = currentDataOffset;
        ulong totalWritten = 0;
        byte[] copyBuf = new byte[4 * 1024 * 1024];

        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            byte* srcPtr = source.GetTensorPointer(t);
            ulong byteSize = t.GetByteSize();

            ulong written = 0;
            while (written < byteSize)
            {
                int chunk = (int)Math.Min((ulong)copyBuf.Length, byteSize - written);
                fixed (byte* pBuf = copyBuf)
                {
                    Buffer.MemoryCopy(srcPtr + written, pBuf, (ulong)copyBuf.Length, (ulong)chunk);
                }
                destStream.Write(copyBuf, 0, chunk);
                written += (ulong)chunk;
                totalWritten += (ulong)chunk;

                if (totalPayloadBytes > 0 && progress != null)
                {
                    progress.Report((double)totalWritten / totalPayloadBytes);
                }
            }

            // Gap padding to next tensor offset
            if (i + 1 < tensors.Count)
            {
                ulong gap = newOffsets[i + 1] - (newOffsets[i] + t.GetByteSize());
                if (gap > 0)
                {
                    byte[] gapBytes = new byte[gap];
                    destStream.Write(gapBytes, 0, gapBytes.Length);
                }
            }
        }

        destStream.Flush();
    }

    /// <summary>
    /// Slices stage tensors directly to a file on local storage.
    /// Uses temporary file pattern to guarantee atomic writes.
    /// </summary>
    public static void SliceStageToFile(
        GgufFile source,
        string destinationFilePath,
        int startLayer,
        int endLayer,
        bool isFirstStage,
        bool isLastStage,
        IProgress<double>? progress = null)
    {
        string dir = Path.GetDirectoryName(destinationFilePath) ?? "";
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        string tempPath = destinationFilePath + ".tmp." + Guid.NewGuid().ToString("N");
        try
        {
            using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 4 * 1024 * 1024))
            {
                SliceStage(source, fs, startLayer, endLayer, isFirstStage, isLastStage, progress);
            }

            if (File.Exists(destinationFilePath))
            {
                File.Delete(destinationFilePath);
            }
            File.Move(tempPath, destinationFilePath);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); } catch { }
            }
        }
    }
}
