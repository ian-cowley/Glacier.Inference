namespace Glacier.Inference.Model;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

public enum LoraProjection
{
    Q,
    K,
    V,
    AttnOut,
    Gate,
    Up,
    Down
}

public sealed class LoraLayerProjections
{
    public (float[]? A, float[]? B, int inDim, int outDim) Q;
    public (float[]? A, float[]? B, int inDim, int outDim) K;
    public (float[]? A, float[]? B, int inDim, int outDim) V;
    public (float[]? A, float[]? B, int inDim, int outDim) AttnOut;
    public (float[]? A, float[]? B, int inDim, int outDim) Gate;
    public (float[]? A, float[]? B, int inDim, int outDim) Up;
    public (float[]? A, float[]? B, int inDim, int outDim) Down;

    public ref (float[]? A, float[]? B, int inDim, int outDim) Get(LoraProjection proj)
    {
        switch (proj)
        {
            case LoraProjection.Q: return ref Q;
            case LoraProjection.K: return ref K;
            case LoraProjection.V: return ref V;
            case LoraProjection.AttnOut: return ref AttnOut;
            case LoraProjection.Gate: return ref Gate;
            case LoraProjection.Up: return ref Up;
            case LoraProjection.Down: return ref Down;
            default: throw new ArgumentOutOfRangeException(nameof(proj));
        }
    }
}

/// <summary>
/// Runtime LoRA adapter weights for zero-allocation inference integration.
/// Loads binary adapter matrices and applies low-rank updates during model forward pass.
/// </summary>
public sealed unsafe class LoraAdapterWeights
{
    private const int StackAllocRankThreshold = 256;

    private readonly int _layerCount;
    private readonly int _rank;
    private readonly float _alpha;
    private readonly float _scaling;
    private readonly LoraLayerProjections[] _layers;

    public int LayerCount => _layerCount;
    public int Rank => _rank;
    public float Alpha => _alpha;
    public float Scaling => _scaling;
    public LoraLayerProjections GetProjections(int layer) => _layers[layer];

    public LoraAdapterWeights(int layerCount, int rank, float alpha)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        if (!float.IsFinite(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "LoRA alpha must be finite.");

        _layerCount = layerCount;
        _rank = rank;
        _alpha = alpha;
        _scaling = rank > 0 ? alpha / rank : 1.0f;
        _layers = new LoraLayerProjections[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            _layers[i] = new LoraLayerProjections();
        }
    }

    public static LoraAdapterWeights Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"LoRA adapter file not found: {path}");

        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        uint magic = br.ReadUInt32();
        if (magic != 0x41524F4C) // 'LORA'
            throw new InvalidDataException($"Invalid LoRA magic: 0x{magic:X8}");

        uint version = br.ReadUInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported LoRA version: {version}");

        int layerCount = br.ReadInt32();
        int rank = br.ReadInt32();
        float alpha = br.ReadSingle();
        int tensorCount = br.ReadInt32();

        if (layerCount < 0)
            throw new InvalidDataException($"Invalid LoRA layer count: {layerCount}");
        if (rank < 0 || (tensorCount > 0 && rank <= 0))
            throw new InvalidDataException($"Invalid LoRA rank: {rank}");
        if (!float.IsFinite(alpha))
            throw new InvalidDataException($"Invalid non-finite LoRA alpha: {alpha}");
        if (tensorCount < 0)
            throw new InvalidDataException($"Invalid LoRA tensor count: {tensorCount}");

        var weights = new LoraAdapterWeights(layerCount, rank, alpha);
        bool sawGlacier = false;
        bool sawPeft = false;
        List<(int layer, LoraProjection proj, bool isA)>? squareTensors = null;

        for (int t = 0; t < tensorCount; t++)
        {
            string name = br.ReadString();
            int dim0 = br.ReadInt32();
            int dim1 = br.ReadInt32();
            if (dim0 <= 0 || dim1 <= 0)
                throw new InvalidDataException($"Invalid LoRA tensor '{name}' dimensions [{dim0}, {dim1}].");

            long elemCount = (long)dim0 * dim1;
            long byteCount = elemCount * sizeof(float);
            if (elemCount > int.MaxValue || byteCount > fs.Length - fs.Position)
                throw new InvalidDataException($"LoRA tensor '{name}' byte length {byteCount} exceeds remaining stream bytes ({fs.Length - fs.Position}).");

            // Parse tensor name: e.g. blk.0.attn_q.lora_a.weight
            var parts = name.Split('.');
            if (parts.Length >= 5 &&
                parts[0] == "blk" &&
                int.TryParse(parts[1], out int layerIdx) &&
                layerIdx >= 0 &&
                layerIdx < layerCount)
            {
                string projName = parts[2];
                string matrixType = parts[3]; // lora_a or lora_b

                LoraProjection proj = projName switch
                {
                    "attn_q" => LoraProjection.Q,
                    "attn_k" => LoraProjection.K,
                    "attn_v" => LoraProjection.V,
                    "attn_output" => LoraProjection.AttnOut,
                    "ffn_gate" => LoraProjection.Gate,
                    "ffn_up" => LoraProjection.Up,
                    "ffn_down" => LoraProjection.Down,
                    _ => (LoraProjection)(-1)
                };

                if ((int)proj >= 0 && (matrixType == "lora_a" || matrixType == "lora_b"))
                {
                    float[] floatData = new float[(int)elemCount];
                    fs.ReadExactly(MemoryMarshal.AsBytes(floatData.AsSpan()));

                    ref var slot = ref weights._layers[layerIdx].Get(proj);
                    if (matrixType == "lora_a")
                    {
                        if (dim0 == rank && dim1 == rank)
                        {
                            slot.A = floatData;
                            slot.inDim = rank;
                            (squareTensors ??= new List<(int, LoraProjection, bool)>()).Add((layerIdx, proj, true));
                        }
                        else if (dim1 == rank)
                        {
                            sawGlacier = true;
                            slot.A = floatData;
                            slot.inDim = dim0;
                        }
                        else if (dim0 == rank)
                        {
                            sawPeft = true;
                            slot.A = Transpose2D(floatData, dim0, dim1);
                            slot.inDim = dim1;
                        }
                        else
                        {
                            throw new InvalidDataException($"LoRA tensor '{name}' shape [{dim0}, {dim1}] is incompatible with rank {rank}.");
                        }
                    }
                    else // lora_b
                    {
                        if (dim0 == rank && dim1 == rank)
                        {
                            slot.B = floatData;
                            slot.outDim = rank;
                            (squareTensors ??= new List<(int, LoraProjection, bool)>()).Add((layerIdx, proj, false));
                        }
                        else if (dim0 == rank)
                        {
                            sawGlacier = true;
                            slot.B = floatData;
                            slot.outDim = dim1;
                        }
                        else if (dim1 == rank)
                        {
                            sawPeft = true;
                            slot.B = Transpose2D(floatData, dim0, dim1);
                            slot.outDim = dim0;
                        }
                        else
                        {
                            throw new InvalidDataException($"LoRA tensor '{name}' shape [{dim0}, {dim1}] is incompatible with rank {rank}.");
                        }
                    }

                    continue;
                }
            }

            fs.Seek(byteCount, SeekOrigin.Current);
        }

        if (sawPeft && !sawGlacier && squareTensors != null)
        {
            foreach (var (layerIdx, proj, isA) in squareTensors)
            {
                ref var slot = ref weights._layers[layerIdx].Get(proj);
                if (isA && slot.A != null)
                {
                    TransposeSquareInPlace(slot.A, rank);
                }
                else if (!isA && slot.B != null)
                {
                    TransposeSquareInPlace(slot.B, rank);
                }
            }
        }

        return weights;
    }

    private static float[] Transpose2D(float[] src, int rows, int cols)
    {
        float[] dst = new float[src.Length];
        for (int r = 0; r < rows; r++)
        {
            int srcRow = r * cols;
            for (int c = 0; c < cols; c++)
            {
                dst[c * rows + r] = src[srcRow + c];
            }
        }
        return dst;
    }

    private static void TransposeSquareInPlace(float[] mat, int n)
    {
        for (int r = 0; r < n; r++)
        {
            for (int c = r + 1; c < n; c++)
            {
                int i0 = r * n + c;
                int i1 = c * n + r;
                (mat[i0], mat[i1]) = (mat[i1], mat[i0]);
            }
        }
    }

    /// <summary>
    /// Applies LoRA low-rank delta directly to projection output:
    /// y += (alpha / r) * (x * A) * B
    /// Uses stackalloc for rank &lt;= 256 (0 heap allocations) and pooled memory for larger ranks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public void Apply(int layer, LoraProjection proj, float* x, float* y, int inDim, int outDim)
    {
        if (_rank <= 0 || x == null || y == null || inDim <= 0 || outDim <= 0 || (uint)layer >= (uint)_layers.Length) return;

        ref readonly var p = ref _layers[layer].Get(proj);
        if (p.A == null || p.B == null) return;

        int rank = _rank;
        long expectedA = (long)inDim * rank;
        long expectedB = (long)rank * outDim;
        if ((p.inDim > 0 && p.inDim != inDim) ||
            (p.outDim > 0 && p.outDim != outDim) ||
            p.A.LongLength < expectedA ||
            p.B.LongLength < expectedB)
        {
            throw new InvalidOperationException(
                $"LoRA projection {proj} at layer {layer} dimension mismatch: adapter has inDim={p.inDim}, outDim={p.outDim}, rank={rank} (A={p.A.LongLength}, B={p.B.LongLength}), caller expected inDim={inDim}, outDim={outDim}.");
        }

        Span<float> stackBuf = rank <= StackAllocRankThreshold ? stackalloc float[rank] : default;
        float[]? rented = rank > StackAllocRankThreshold ? ArrayPool<float>.Shared.Rent(rank) : null;
        try
        {
            Span<float> lowRankSpan = rented is null ? stackBuf : rented.AsSpan(0, rank);
            lowRankSpan.Clear();

            fixed (float* lowRank = lowRankSpan)
            fixed (float* pA = p.A, pB = p.B)
            {
                // 1. lowRank = x * A:  x is [inDim], A is [inDim, rank]
                // For row-major A [inDim, rank]: A[i, r] = pA[i * rank + r]
                for (int i = 0; i < inDim; i++)
                {
                    float xi = x[i];
                    float* aRow = pA + (long)i * rank;
                    for (int r = 0; r < rank; r++)
                    {
                        lowRank[r] += xi * aRow[r];
                    }
                }

                // 2. delta = lowRank * B: lowRank is [rank], B is [rank, outDim]
                // y[d] += scaling * sum_{r=0..rank-1} lowRank[r] * B[r * outDim + d]
                float scaling = _scaling;
                for (int r = 0; r < rank; r++)
                {
                    float lr = lowRank[r] * scaling;
                    if (lr == 0.0f) continue;

                    float* bRow = pB + (long)r * outDim;
                    for (int d = 0; d < outDim; d++)
                    {
                        y[d] += lr * bRow[d];
                    }
                }
            }
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }
}
