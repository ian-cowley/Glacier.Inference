namespace Glacier.Inference.Gpu;

using System;
using System.IO;
using Glacier.Inference.Gguf;
using Glacier.Inference.Quant;

/// <summary>
/// Pure C# Vulkan 1.1+ / 1.3 compute runner for NariQ Trellis-Coded Quantization
/// (<c>GGML_TYPE_TRL1..TRL6</c>, <c>nariq/FORMAT.md</c> v1).
/// Loads embedded SPIR-V shaders (<c>trellis_pre_rht[_128].spv</c>, <c>trellis_gemv[_dp4a].spv</c>,
/// <c>trellis_gemv_splitk[_dp4a].spv</c>, <c>trellis_post_rht.spv</c>, <c>trellis_gather_f32.spv</c>)
/// and runs on AMD Radeon 890M (<c>RADV STRIX1</c>), NVIDIA Vulkan, or any Vulkan 1.1+ GPU.
/// </summary>
public sealed unsafe class TrellisVulkanRunner : IDisposable
{
    private readonly VulkanContext _ctx;
    private readonly VulkanCompute _vk;
    private readonly VulkanCompute.Kernel _kPreRht;
    private readonly VulkanCompute.Kernel? _kPreRht128;
    private readonly VulkanCompute.Kernel _kGemv;
    private readonly VulkanCompute.Kernel? _kGemvSplitK;
    private readonly VulkanCompute.Kernel _kPostRht;
    private readonly VulkanCompute.Kernel? _kPostRht128;
    private readonly VulkanCompute.Kernel _kGather;
    private readonly VulkanCompute.GBuf _dummyIdx;
    private bool _disposed;

    public string DeviceName => _ctx.DeviceName;
    public bool SupportsIntegerDotProduct => _ctx.SupportsIntegerDotProduct;

    public sealed class DeviceMatrix
    {
        internal VulkanCompute.GBuf CodeWords = null!, S16 = null!, Pre = null!, Post = null!;
        internal VulkanCompute.GBuf? Pin, Pout;
        public int K, N, Bits;
        public int RowsKind, RowsBlock, ColsKind, ColsBlock;
        public bool HasPin, HasPout;
    }

    private static bool HasEmbeddedShader(string name)
    {
        using Stream? s = typeof(VulkanCompute).Assembly.GetManifestResourceStream($"vkshader.{name}.spv");
        return s != null;
    }

    public TrellisVulkanRunner(int deviceIndex = 0)
    {
        _ctx = new VulkanContext(deviceIndex);
        _vk = new VulkanCompute(_ctx);
        _kPreRht = _vk.LoadEmbedded("trellis_pre_rht", 5, 16, 1u << 3);
        if (_ctx.SupportsSubgroupShuffle && HasEmbeddedShader("trellis_pre_rht_128"))
            _kPreRht128 = _vk.LoadEmbedded("trellis_pre_rht_128", 5, 16, 1u << 3);
        string gemvName = _ctx.SupportsIntegerDotProduct && HasEmbeddedShader("trellis_gemv_dp4a")
            ? "trellis_gemv_dp4a"
            : "trellis_gemv";
        _kGemv = _vk.LoadEmbedded(gemvName, 3, 12, 1u << 2);
        string splitKName = _ctx.SupportsIntegerDotProduct && HasEmbeddedShader("trellis_gemv_splitk_dp4a")
            ? "trellis_gemv_splitk_dp4a"
            : "trellis_gemv_splitk";
        if (HasEmbeddedShader(splitKName))
            _kGemvSplitK = _vk.LoadEmbedded(splitKName, 3, 12, 1u << 2);
        _kPostRht = _vk.LoadEmbedded("trellis_post_rht", 3, 16, 1u << 2);
        if (_ctx.SupportsSubgroupShuffle && HasEmbeddedShader("trellis_post_rht_128"))
            _kPostRht128 = _vk.LoadEmbedded("trellis_post_rht_128", 3, 16, 1u << 2);
        _kGather = _vk.LoadEmbedded("trellis_gather_f32", 3, 4, 1u << 2);
        _dummyIdx = _vk.Alloc(16);
        _dummyIdx.Zero();
    }

    public DeviceMatrix Upload(TrellisWeightBinding b)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int bits = GgufTypes.GetTrellisBits(b.Type);
        long codeBytes = (long)b.K * b.N * bits / 8;
        int s16Count = b.K / 16;
        long s16Bytes = Math.Max(4L, ((s16Count + 1) / 2) * 4L); // pad to uint32 pairs for unpackHalf2x16
        long preBytes = (long)b.K * sizeof(float);
        long postBytes = (long)b.N * sizeof(float);

        VulkanCompute.GBuf s16Buf;
        if ((long)s16Count * sizeof(ushort) == s16Bytes)
        {
            s16Buf = _vk.AllocDeviceFrom(b.S16, s16Bytes);
        }
        else
        {
            ushort[] padded = new ushort[s16Bytes / 2];
            new ReadOnlySpan<ushort>(b.S16, s16Count).CopyTo(padded);
            fixed (ushort* ps = padded) s16Buf = _vk.AllocDeviceFrom(ps, s16Bytes);
        }

        return new DeviceMatrix
        {
            K = b.K,
            N = b.N,
            Bits = bits,
            RowsKind = b.Descriptor[0],
            RowsBlock = b.Descriptor[1],
            ColsKind = b.Descriptor[3],
            ColsBlock = b.Descriptor[4],
            HasPin = b.Pin != null,
            HasPout = b.Pout != null,
            CodeWords = _vk.AllocDeviceFrom(b.CodeData, codeBytes),
            S16 = s16Buf,
            Pre = _vk.AllocDeviceFrom(b.Pre, preBytes),
            Post = _vk.AllocDeviceFrom(b.Post, postBytes),
            Pin = b.Pin != null ? _vk.AllocDeviceFrom(b.Pin, (long)b.K * sizeof(int)) : null,
            Pout = b.Pout != null ? _vk.AllocDeviceFrom(b.Pout, (long)b.N * sizeof(int)) : null,
        };
    }

    /// <summary>
    /// Executes <c>y = gather_pout(post * T_rows^T(W_rotated * T_cols(pre * gather_pin(x))))</c>
    /// in a single submitted Vulkan command buffer and returns GPU execution time in milliseconds.
    /// </summary>
    public double MatVecMul(DeviceMatrix m, ReadOnlySpan<float> xHost, Span<float> yHost)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (m.ColsKind > 1 || m.RowsKind > 1)
            throw new NotSupportedException($"Vulkan shader currently supports kind=0 (identity) and kind=1 (block RHT <= 2048); got cols={m.ColsKind}, rows={m.RowsKind}.");

        var gX = _vk.Alloc((long)m.K * sizeof(float));
        var gXRot = _vk.AllocDeviceOnly((long)m.K * sizeof(float));
        var gYRot = _vk.AllocDeviceOnly((long)m.N * sizeof(float));
        var gYScaled = m.HasPout ? _vk.AllocDeviceOnly((long)m.N * sizeof(float)) : _vk.Alloc((long)m.N * sizeof(float));
        var gYFinal = m.HasPout ? _vk.Alloc((long)m.N * sizeof(float)) : gYScaled;

        fixed (float* px = xHost)
            gX.UploadFrom(px, (long)m.K * sizeof(float));

        bool usePre128 = m.ColsKind == 1 && m.ColsBlock == 128 && _kPreRht128 != null;
        var kPre = usePre128 ? _kPreRht128! : _kPreRht;
        bool usePost128 = m.RowsKind == 1 && m.RowsBlock == 128 && _kPostRht128 != null;
        var kPost = usePost128 ? _kPostRht128! : _kPostRht;
        int kt = m.K >> 4;
        bool useSplitK = m.N <= 6144 && kt >= 64 && _kGemvSplitK != null;
        var kGemv = useSplitK ? _kGemvSplitK! : _kGemv;

        IntPtr setPre = _vk.MakeSet(kPre, gX, m.Pre, m.Pin ?? _dummyIdx, gXRot, m.S16);
        IntPtr setGemv = _vk.MakeSet(kGemv, m.CodeWords, gXRot, gYRot);
        IntPtr setPost = _vk.MakeSet(kPost, gYRot, m.Post, gYScaled);
        IntPtr setGather = m.HasPout ? _vk.MakeSet(_kGather, gYScaled, m.Pout!, gYFinal) : IntPtr.Zero;

        uint preGrid = m.ColsKind == 1 ? (uint)(m.K / m.ColsBlock) : (uint)((m.K + 255) / 256);
        uint gemvGrid = useSplitK ? (uint)((m.N + 15) / 16) : (uint)((m.N + 63) / 64);
        uint postGrid = m.RowsKind == 1 ? (uint)(m.N / m.RowsBlock) : (uint)((m.N + 255) / 256);

        _vk.Begin();
        _vk.Dispatch(kPre, setPre, [(uint)m.K, (uint)m.ColsKind, (uint)m.ColsBlock, m.HasPin ? 1u : 0u], preGrid);
        _vk.Dispatch(kGemv, setGemv, [(uint)m.K, (uint)m.N, (uint)m.Bits], gemvGrid);
        _vk.Dispatch(kPost, setPost, [(uint)m.N, (uint)m.RowsKind, (uint)m.RowsBlock, 0u], postGrid);
        if (m.HasPout)
        {
            _vk.Dispatch(_kGather, setGather, [(uint)m.N], (uint)((m.N + 255) / 256));
        }

        double ms = _vk.SubmitAndWait();
        fixed (float* py = yHost)
            gYFinal.DownloadTo(py, (long)m.N * sizeof(float));

        return ms;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vk.Dispose();
        _ctx.Dispose();
    }
}
