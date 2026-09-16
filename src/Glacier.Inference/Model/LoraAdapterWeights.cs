namespace Glacier.Inference.Model;

using System;
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
    private readonly int _layerCount;
    private readonly int _rank;
    private readonly float _alpha;
    private readonly float _scaling;
    private readonly LoraLayerProjections[] _layers;

    public int LayerCount => _layerCount;
    public int Rank => _rank;
    public float Alpha => _alpha;
    public float Scaling => _scaling;

    public LoraAdapterWeights(int layerCount, int rank, float alpha)
    {
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

        var weights = new LoraAdapterWeights(layerCount, rank, alpha);

        for (int t = 0; t < tensorCount; t++)
        {
            string name = br.ReadString();
            int dim0 = br.ReadInt32();
            int dim1 = br.ReadInt32();
            int elemCount = dim0 * dim1;

            byte[] byteData = br.ReadBytes(elemCount * sizeof(float));
            float[] floatData = new float[elemCount];
            Buffer.BlockCopy(byteData, 0, floatData, 0, byteData.Length);

            // Parse tensor name: e.g. blk.0.attn_q.lora_a.weight
            var parts = name.Split('.');
            if (parts.Length >= 5 && parts[0] == "blk" && int.TryParse(parts[1], out int layerIdx) && layerIdx < layerCount)
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

                if ((int)proj >= 0)
                {
                    ref var slot = ref weights._layers[layerIdx].Get(proj);
                    if (matrixType == "lora_a")
                    {
                        slot.A = floatData;
                        slot.inDim = dim0;
                    }
                    else if (matrixType == "lora_b")
                    {
                        slot.B = floatData;
                        slot.outDim = dim1;
                    }
                }
            }
        }

        return weights;
    }

    /// <summary>
    /// Applies LoRA low-rank delta directly to projection output:
    /// y += (alpha / r) * (x * A) * B
    /// Uses stackalloc for rank-16 intermediate vector, guaranteeing 0 heap allocations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public void Apply(int layer, LoraProjection proj, float* x, float* y, int inDim, int outDim)
    {
        if (layer < 0 || layer >= _layers.Length) return;

        ref readonly var p = ref _layers[layer].Get(proj);
        if (p.A == null || p.B == null) return;

        int rank = _rank;
        float* lowRank = stackalloc float[rank];
        new Span<float>(lowRank, rank).Clear();

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
}
