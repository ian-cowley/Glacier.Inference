using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Model;

namespace Glacier.Inference.Gpu;

/// <summary>
/// Shared base for Vulkan 1.1 compute inference runners (op graph recording, buffer upload,
/// quantized matvec / MoE helpers, and recurrent-state reset).
/// </summary>
public abstract unsafe class VulkanRunnerBase : ID3D12Model
{
    private protected enum Patch { None, Pos, KvLen, Token, DsaSel }

    private protected sealed class Op
    {
        public VulkanCompute.Kernel K = null!;
        public IntPtr Set;
        public uint[] Push = Array.Empty<uint>();
        public uint Gx = 1, Gy = 1;
        public Patch Patch;
    }

    protected readonly ModelWeights _w;
    private protected readonly VulkanContext _ctx;
    private protected readonly VulkanCompute _vk;
    protected readonly int _maxSeqLen, _vocab;

    private protected readonly List<Op> _embedOps = new();
    private protected readonly List<Op> _layerOps = new();
    private protected readonly List<Op> _finalOps = new();
    private protected readonly List<VulkanCompute.GBuf> _recurrentStates = new();

    private protected VulkanCompute.GBuf _dLogits = null!;
    private protected VulkanCompute.Kernel _pRmsNorm = null!, _pAdd = null!;
    private protected readonly VulkanCompute.Kernel[] _pEmbed = new VulkanCompute.Kernel[4];
    private protected readonly VulkanCompute.Kernel[] _pQmv = new VulkanCompute.Kernel[8];
    private protected readonly VulkanCompute.Kernel[] _pMoe = new VulkanCompute.Kernel[8];

    public string DeviceName => _ctx.DeviceName;
    public int MaxSeqLen => _maxSeqLen;

    protected VulkanRunnerBase(ModelWeights weights, int maxSeqLen, int deviceOrdinal)
    {
        _w = weights;
        _maxSeqLen = maxSeqLen;
        _vocab = weights.VocabSize;
        _ctx = new VulkanContext(deviceOrdinal);
        _vk = new VulkanCompute(_ctx);
    }

    protected void LoadBaseKernels(bool loadMoe)
    {
        _pRmsNorm = _vk.LoadEmbedded("rmsnorm", 3, 12);
        _pAdd = _vk.LoadEmbedded("elementwise_add", 3, 4);
        for (int q = 0; q < 8; q++)
        {
            if (q < 4) _pEmbed[q] = _vk.LoadEmbedded($"embed_q{q}", 2, 8);
            _pQmv[q] = _vk.LoadEmbedded($"qmatvec_q{q}", 3, 8);
            if (loadMoe) _pMoe[q] = _vk.LoadEmbedded($"qmatvec_moe_q{q}", 4, 16);
        }
    }

    protected static bool IsEmbedKernelType(GgufType t) => t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0;
    protected static bool IsKernelType(GgufType t) => t is GgufType.Q4_K or GgufType.Q5_K or GgufType.Q6_K or GgufType.Q8_0 or GgufType.IQ4_XS or GgufType.MXFP4 or GgufType.Q3_K or GgufType.IQ4_NL;

    protected static int QIdx(GgufType t) => t switch
    {
        GgufType.Q4_K => 0, GgufType.Q5_K => 1, GgufType.Q6_K => 2, GgufType.Q8_0 => 3,
        GgufType.IQ4_XS => 4, GgufType.MXFP4 => 5, GgufType.Q3_K => 6, GgufType.IQ4_NL => 7,
        _ => throw new NotSupportedException($"Unsupported Vulkan quant type {t}"),
    };

    protected static uint FBits(float f) => BitConverter.SingleToUInt32Bits(f);

    protected static long GgufTypeSize(GgufType t, long elements) => t switch
    {
        GgufType.Q4_K => (elements / 256) * 144,
        GgufType.Q5_K => (elements / 256) * 176,
        GgufType.Q6_K => (elements / 256) * 210,
        GgufType.Q8_0 => (elements / 32) * 34,
        GgufType.IQ4_XS => (elements / 256) * 136,
        GgufType.MXFP4 => (elements / 32) * 17,
        GgufType.Q3_K => (elements / 256) * 110,
        GgufType.IQ4_NL => (elements / 32) * 18,
        _ => throw new NotSupportedException($"Unsupported Vulkan quant type {t}"),
    };

    private protected VulkanCompute.GBuf UploadBytes(byte* ptr, long bytes) { var b = _vk.Alloc(bytes); b.UploadFrom(ptr, bytes); return b; }
    private protected VulkanCompute.GBuf UploadFloats(float* ptr, int count) { long bytes = (long)count * 4; var b = _vk.Alloc(bytes); b.UploadFrom(ptr, bytes); return b; }
    private protected VulkanCompute.GBuf AllocStateZero(long bytes) { var b = _vk.Alloc(bytes); b.Zero(); _recurrentStates.Add(b); return b; }

    private protected void AddRmsNorm(List<Op> list, VulkanCompute.GBuf x, VulkanCompute.GBuf w, VulkanCompute.GBuf o, int n, float eps) =>
        list.Add(new Op { K = _pRmsNorm, Set = _vk.MakeSet(_pRmsNorm, x, w, o), Push = [(uint)n, 1u, FBits(eps)], Gx = 1 });

    private protected void AddQmv(List<Op> list, GgufType t, VulkanCompute.GBuf w, VulkanCompute.GBuf x, VulkanCompute.GBuf y, int k, int m)
    {
        var kern = _pQmv[QIdx(t)];
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, w, x, y), Push = [(uint)k, (uint)m], Gx = (uint)((m + 3) / 4) });
    }

    private protected void AddVec(List<Op> list, VulkanCompute.Kernel kern, VulkanCompute.GBuf a, VulkanCompute.GBuf b, VulkanCompute.GBuf o, int n) =>
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, a, b, o), Push = [(uint)n], Gx = (uint)((n + 255) / 256) });

    private protected void AddMoeMv(List<Op> list, GgufType t, VulkanCompute.GBuf w, VulkanCompute.GBuf idx, VulkanCompute.GBuf x, VulkanCompute.GBuf y, int k, int m, long sliceBytes, int xStride, int topK)
    {
        var kern = _pMoe[QIdx(t)];
        list.Add(new Op { K = kern, Set = _vk.MakeSet(kern, w, idx, x, y), Push = [(uint)k, (uint)m, (uint)(sliceBytes / 4), (uint)xStride], Gx = (uint)((m + 3) / 4), Gy = (uint)topK });
    }

    public void ResetKvCache()
    {
        foreach (var buf in _recurrentStates) buf.Zero();
    }

    public void ForwardBatch(ReadOnlySpan<int> tokens, int startPos, Span<float> logits, bool computeLogits = true)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            bool last = i == tokens.Length - 1;
            Forward(tokens[i], startPos + i, last && computeLogits ? logits : Span<float>.Empty, last && computeLogits);
        }
    }

    public void Forward(int token, int pos, Span<float> logits, bool computeLogits = true)
    {
        if ((uint)pos >= (uint)_maxSeqLen) throw new ArgumentOutOfRangeException(nameof(pos), pos, $"Position exceeds MaxSeqLen ({_maxSeqLen}).");
        if ((uint)token >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(token), token, "Token id outside vocabulary.");
        bool wantLogits = computeLogits && logits.Length >= _vocab;

        _vk.Begin();
        Run(_embedOps, token, pos);
        Run(_layerOps, token, pos);
        if (wantLogits) Run(_finalOps, token, pos);
        _vk.SubmitAndWait();

        if (wantLogits)
        {
            fixed (float* p = logits) _dLogits.DownloadTo(p, (long)_vocab * 4);
        }
    }

    private void Run(List<Op> ops, int token, int pos)
    {
        foreach (var op in ops)
        {
            switch (op.Patch)
            {
                case Patch.Pos: op.Push[0] = (uint)pos; break;
                case Patch.KvLen: op.Push[0] = (uint)(pos + 1); break;
                case Patch.Token: op.Push[0] = (uint)token; break;
                case Patch.DsaSel: op.Push[0] = (uint)Math.Min(pos + 1, _w.DsaTopK); break;
            }
            _vk.Dispatch(op.K, op.Set, op.Push, op.Gx, op.Gy);
        }
    }

    public void Dispose() { _vk.Dispose(); _ctx.Dispose(); }
}
