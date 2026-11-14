namespace Glacier.Inference.Tests;

using System;
using System.IO;
using System.Linq;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Quant;
using Xunit;

/// <summary>Skips Vulkan tests when no hardware Vulkan device is available (software rasterisers such as llvmpipe are ignored).</summary>
public sealed class VulkanFactAttribute : FactAttribute
{
    public static int DeviceOrdinal => Qwen35VulkanModel.FindVulkanDevice("");

    public VulkanFactAttribute()
    {
        if (!VulkanContext.IsSupported) Skip = "Vulkan loader is not available in this environment.";
        else if (DeviceOrdinal < 0) Skip = "No hardware Vulkan device found (software rasterisers are ignored).";
    }
}

/// <summary>
/// Validates each building block of the Vulkan Qwen 3.5/3.6 runner against the managed / CPU reference implementations,
/// using the real embedded SPIR-V kernels at small sizes: quantised mat-vecs, the Gated DeltaNet token step,
/// full attention (QK-norm, partial RoPE, KV cache, Q gate) and the MoE FFN chain.
/// </summary>
[Collection("SequentialGpu")]
public unsafe class VulkanKernelTests
{
    // ---------- helpers ----------
    private static float Gauss(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
    }

    private static float[] Rand(Random rng, int n, float scale = 1f, float offset = 0f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = offset + scale * Gauss(rng);
        return a;
    }

    /// <summary>Random quantised blocks with valid (small) fp16 scales written at <paramref name="halfOffsets"/> of every block.</summary>
    private static byte[] RandQuant(Random rng, long blocks, int blockBytes, int[] halfOffsets, float lo, float hi)
    {
        var w = new byte[blocks * blockBytes];
        rng.NextBytes(w);
        for (long b = 0; b < blocks; b++)
            foreach (int ho in halfOffsets)
                BitConverter.TryWriteBytes(w.AsSpan((int)(b * blockBytes + ho), 2), (Half)(lo + (float)rng.NextDouble() * (hi - lo)));
        return w;
    }

    private static VulkanCompute.GBuf Up(VulkanCompute vk, byte[] data)
    {
        var g = vk.Alloc(data.Length);
        fixed (byte* p = data) g.UploadFrom(p, data.Length);
        return g;
    }

    private static VulkanCompute.GBuf Up(VulkanCompute vk, float[] data)
    {
        var g = vk.Alloc(data.Length * 4L);
        g.Upload(data);
        return g;
    }

    private static float[] Down(VulkanCompute.GBuf g, int count)
    {
        var r = new float[count];
        fixed (float* p = r) g.DownloadTo(p, count * 4L);
        return r;
    }

    private static void AssertClose(float[] expected, float[] actual, double absTol, double relTol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int bad = 0; double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double e = Math.Abs(actual[i] - expected[i]);
            worst = Math.Max(worst, e);
            if (!(e <= absTol + relTol * Math.Abs(expected[i]))) bad++;   // also catches NaN
        }
        Assert.True(bad == 0, $"{what}: {bad}/{expected.Length} values outside tolerance (max abs error {worst:E2}).");
    }

    // ---------- device selection ----------
    [Fact]
    public void FindVulkanDevice_NeverSelectsSoftwareRasteriser()
    {
        int idx = Qwen35VulkanModel.FindVulkanDevice("");
        if (!VulkanContext.IsSupported) { Assert.Equal(-1, idx); return; }
        if (idx < 0) return;
        var dev = VulkanContext.EnumerateDevices().First(d => d.Index == idx);
        Assert.DoesNotContain("llvmpipe", dev.Name, StringComparison.OrdinalIgnoreCase);
    }

    // ---------- quantised mat-vec ----------
    [VulkanFact]
    public void QuantizedMatVec_MatchesCpuReference_ForAllSupportedTypes()
    {
        const int K = 1024, N = 300;     // N deliberately not a multiple of the workgroup size
        var cases = new (string Kernel, GgufType Type, int BlockElems, int BlockBytes, int[] Halves)[]
        {
            ("qmatvec_q0", GgufType.Q4_K, 256, 144, new[] { 0, 2 }),
            ("qmatvec_q1", GgufType.Q5_K, 256, 176, new[] { 0, 2 }),
            ("qmatvec_q2", GgufType.Q6_K, 256, 210, new[] { 208 }),
            ("qmatvec_q3", GgufType.Q8_0, 32, 34, new[] { 0 }),
        };
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(11);
        float[] x = Rand(rng, K);

        foreach (var c in cases)
        {
            byte[] w = RandQuant(rng, (long)N * (K / c.BlockElems), c.BlockBytes, c.Halves, 0.002f, 0.006f);
            var yCpu = new float[N];
            fixed (byte* pw = w) fixed (float* px = x) fixed (float* py = yCpu)
                QuantKernels.MatVecMul(c.Type, pw, px, py, K, N);

            var kern = vk.LoadEmbedded(c.Kernel, 3, 8);
            var gW = Up(vk, w); var gX = Up(vk, x); var gY = vk.Alloc(N * 4L);
            var set = vk.MakeSet(kern, gW, gX, gY);
            vk.Begin();
            vk.Dispatch(kern, set, new uint[] { K, N }, N);
            vk.SubmitAndWait();
            AssertClose(yCpu, Down(gY, N), 1e-3, 1e-3, $"mat-vec {c.Type}");
        }
    }

    // ---------- Mixture-of-Experts FFN ----------
    [VulkanFact]
    public void MoeFfn_RouterExpertsAndSharedExpert_MatchCpuReference()
    {
        const int DIM = 512, E = 16, TOPK = 4, F = 256;
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(21);

        long gateSlice = (long)F * (DIM / 256) * 144;   // Q4_K  rows=F   K=DIM
        long downSlice = (long)DIM * (F / 256) * 176;   // Q5_K  rows=DIM K=F
        byte[] gateW = RandQuant(rng, (long)E * F * (DIM / 256), 144, new[] { 0, 2 }, 0.0002f, 0.0006f);
        byte[] upW = RandQuant(rng, (long)E * F * (DIM / 256), 144, new[] { 0, 2 }, 0.0002f, 0.0006f);
        byte[] downW = RandQuant(rng, (long)E * DIM * (F / 256), 176, new[] { 0, 2 }, 0.0002f, 0.0006f);
        byte[] sgW = RandQuant(rng, (long)F * (DIM / 32), 34, new[] { 0 }, 0.0005f, 0.002f);
        byte[] suW = RandQuant(rng, (long)F * (DIM / 32), 34, new[] { 0 }, 0.0005f, 0.002f);
        byte[] sdW = RandQuant(rng, (long)DIM * (F / 32), 34, new[] { 0 }, 0.0005f, 0.002f);
        float[] routerW = Rand(rng, E * DIM, 0.1f), shexpGate = Rand(rng, DIM, 0.05f);
        float[] normX = Rand(rng, DIM), xRes = Rand(rng, DIM);

        // CPU reference (mirrors CpuModelBase.ForwardFfn, MoE branch with shared expert + sigmoid gate)
        var expect = (float[])xRes.Clone();
        int* idx = stackalloc int[TOPK]; float* wts = stackalloc float[TOPK];
        var gate = new float[F]; var up = new float[F]; var act = new float[F]; var down = new float[DIM]; var ffnOut = new float[DIM];
        fixed (float* pn = normX) fixed (float* pr = routerW) fixed (byte* pg = gateW) fixed (byte* pu = upW) fixed (byte* pd = downW)
        fixed (float* pgate = gate) fixed (float* pup = up) fixed (float* pact = act) fixed (float* pdown = down)
        {
            QuantKernels.RouterTopK(pn, pr, null, DIM, E, TOPK, idx, wts, true);
            for (int k = 0; k < TOPK; k++)
            {
                int e = idx[k];
                QuantKernels.MatVecMul(GgufType.Q4_K, pg + e * gateSlice, pn, pgate, DIM, F);
                QuantKernels.MatVecMul(GgufType.Q4_K, pu + e * gateSlice, pn, pup, DIM, F);
                QuantKernels.SwiGLU(pgate, pup, pact, F);
                QuantKernels.MatVecMul(GgufType.Q5_K, pd + e * downSlice, pact, pdown, F, DIM);
                for (int d = 0; d < DIM; d++) ffnOut[d] += wts[k] * down[d];
            }
        }
        var sg = new float[F]; var su = new float[F]; var sact = new float[F]; var sd = new float[DIM];
        fixed (byte* a = sgW) fixed (byte* b = suW) fixed (byte* c = sdW) fixed (float* pn = normX)
        fixed (float* psg = sg) fixed (float* psu = su) fixed (float* psa = sact) fixed (float* psd = sd)
        {
            QuantKernels.MatVecMul(GgufType.Q8_0, a, pn, psg, DIM, F);
            QuantKernels.MatVecMul(GgufType.Q8_0, b, pn, psu, DIM, F);
            QuantKernels.SwiGLU(psg, psu, psa, F);
            QuantKernels.MatVecMul(GgufType.Q8_0, c, psa, psd, F, DIM);
        }
        float dot = 0; for (int i = 0; i < DIM; i++) dot += normX[i] * shexpGate[i];
        float sgate = 1f / (1f + MathF.Exp(-dot));
        for (int d = 0; d < DIM; d++) expect[d] += ffnOut[d] + sd[d] * sgate;
        int[] expectIdx = new int[TOPK]; for (int k = 0; k < TOPK; k++) expectIdx[k] = idx[k];

        // GPU
        var kRouter = vk.LoadEmbedded("f32_matvec", 3, 8);
        var kTopk = vk.LoadEmbedded("moe_topk", 3, 12);
        var kMoe4 = vk.LoadEmbedded("qmatvec_moe_q0", 4, 16);
        var kMoe5 = vk.LoadEmbedded("qmatvec_moe_q1", 4, 16);
        var kQ8 = vk.LoadEmbedded("qmatvec_q3", 3, 8);
        var kSwi = vk.LoadEmbedded("elementwise_swiglu", 3, 4);
        var kComb = vk.LoadEmbedded("moe_combine", 6, 12);

        var gNorm = Up(vk, normX); var gX = Up(vk, xRes); var gRouter = Up(vk, routerW); var gShGate = Up(vk, shexpGate);
        var gGateW = Up(vk, gateW); var gUpW = Up(vk, upW); var gDownW = Up(vk, downW);
        var gSgW = Up(vk, sgW); var gSuW = Up(vk, suW); var gSdW = Up(vk, sdW);
        var gLogits = vk.Alloc(E * 4L); var gIdx = vk.Alloc(TOPK * 4L); var gTw = vk.Alloc(TOPK * 4L);
        var gGateOut = vk.Alloc(TOPK * F * 4L); var gUpOut = vk.Alloc(TOPK * F * 4L); var gAct = vk.Alloc(TOPK * F * 4L);
        var gDownOut = vk.Alloc(TOPK * DIM * 4L);
        var gSg = vk.Alloc(F * 4L); var gSu = vk.Alloc(F * 4L); var gSAct = vk.Alloc(F * 4L); var gSd = vk.Alloc(DIM * 4L);

        vk.Begin();
        vk.Dispatch(kRouter, vk.MakeSet(kRouter, gRouter, gNorm, gLogits), new uint[] { DIM, E }, E);
        vk.Dispatch(kTopk, vk.MakeSet(kTopk, gLogits, gIdx, gTw), new uint[] { E, TOPK, 1 }, 1);
        vk.Dispatch(kMoe4, vk.MakeSet(kMoe4, gGateW, gNorm, gGateOut, gIdx), new uint[] { DIM, F, (uint)gateSlice, 0 }, F, TOPK);
        vk.Dispatch(kMoe4, vk.MakeSet(kMoe4, gUpW, gNorm, gUpOut, gIdx), new uint[] { DIM, F, (uint)gateSlice, 0 }, F, TOPK);
        vk.Dispatch(kSwi, vk.MakeSet(kSwi, gGateOut, gUpOut, gAct), new uint[] { TOPK * F }, (TOPK * F + 255) / 256);
        vk.Dispatch(kMoe5, vk.MakeSet(kMoe5, gDownW, gAct, gDownOut, gIdx), new uint[] { F, DIM, (uint)downSlice, F }, DIM, TOPK);
        vk.Dispatch(kQ8, vk.MakeSet(kQ8, gSgW, gNorm, gSg), new uint[] { DIM, F }, F);
        vk.Dispatch(kQ8, vk.MakeSet(kQ8, gSuW, gNorm, gSu), new uint[] { DIM, F }, F);
        vk.Dispatch(kSwi, vk.MakeSet(kSwi, gSg, gSu, gSAct), new uint[] { F }, (F + 255) / 256);
        vk.Dispatch(kQ8, vk.MakeSet(kQ8, gSdW, gSAct, gSd), new uint[] { F, DIM }, DIM);
        vk.Dispatch(kComb, vk.MakeSet(kComb, gX, gNorm, gShGate, gDownOut, gTw, gSd), new uint[] { DIM, TOPK, 1 }, 1);
        vk.SubmitAndWait();

        var gpuIdxRaw = new int[TOPK];
        fixed (int* p = gpuIdxRaw) gIdx.DownloadTo(p, TOPK * 4L);
        Assert.Equal(expectIdx, gpuIdxRaw);
        AssertClose(expect, Down(gX, DIM), 2e-2, 2e-3, "MoE FFN output");
    }

    // ---------- Gated DeltaNet ----------
    private const int GdnD = 128, GdnConvKernel = 4;

    [VulkanFact]
    public void GatedDeltaNet_TokenStep_MatchesReference_AndUsesTiledHeadMapping()
    {
        // nV / nK = 2 so grouped (h / 2) and tiled (h % nK) V-head -> Q/K-head mappings differ; llama.cpp tiles.
        const int NK = 2, NV = 4, CH = (NK * 2 + NV) * GdnD, INNER = NV * GdnD, TOKENS = 5;
        const float eps = 1e-6f;
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(31);

        float[] convW = Rand(rng, CH * GdnConvKernel, 0.4f), dtBias = Rand(rng, NV, 0.3f, 0.5f), normW = Rand(rng, GdnD, 0.1f, 1f);
        float[] aW = new float[NV]; for (int h = 0; h < NV; h++) aW[h] = -(0.02f + (float)rng.NextDouble() * 1.5f);

        var kConv = vk.LoadEmbedded("gdn_conv", 4, 4); var kL2 = vk.LoadEmbedded("gdn_l2", 1, 4);
        var kRec = vk.LoadEmbedded("gdn_rec", 7, 8); var kNorm = vk.LoadEmbedded("rmsnorm", 3, 12);
        var kSwi = vk.LoadEmbedded("elementwise_swiglu", 3, 4);

        var gQkv = vk.Alloc(CH * 4L); var gZ = vk.Alloc(INNER * 4L); var gA = vk.Alloc(NV * 4L); var gB = vk.Alloc(NV * 4L);
        var gConvW = Up(vk, convW); var gConvS = vk.Alloc(CH * 3L * 4); gConvS.Zero(); var gConvOut = vk.Alloc(CH * 4L);
        var gDt = Up(vk, dtBias); var gAW = Up(vk, aW); var gState = vk.Alloc((long)NV * GdnD * GdnD * 4); gState.Zero();
        var gY = vk.Alloc(INNER * 4L); var gNormW = Up(vk, normW); var gOut = vk.Alloc(INNER * 4L);
        var sConv = vk.MakeSet(kConv, gQkv, gConvW, gConvS, gConvOut);
        var sL2 = vk.MakeSet(kL2, gConvOut);
        var sRec = vk.MakeSet(kRec, gConvOut, gA, gB, gDt, gAW, gState, gY);
        var sNorm = vk.MakeSet(kNorm, gY, gNormW, gY);
        var sSwi = vk.MakeSet(kSwi, gZ, gY, gOut);

        var refConv = new float[CH * 3];
        var refS = new float[NV][]; for (int h = 0; h < NV; h++) refS[h] = new float[GdnD * GdnD];

        for (int t = 0; t < TOKENS; t++)
        {
            float[] qkv = Rand(rng, CH), z = Rand(rng, INNER), alpha = Rand(rng, NV), beta = Rand(rng, NV);
            float[] expect = GdnReference(NK, NV, qkv, z, alpha, beta, convW, dtBias, aW, normW, eps, refConv, refS);

            gQkv.Upload(qkv); gZ.Upload(z); gA.Upload(alpha); gB.Upload(beta);
            vk.Begin();
            vk.Dispatch(kConv, sConv, new uint[] { CH }, (CH + 255) / 256);
            vk.Dispatch(kL2, sL2, new uint[] { NK }, NK);
            vk.Dispatch(kRec, sRec, new uint[] { NK, NV }, NV);
            vk.Dispatch(kNorm, sNorm, new uint[] { GdnD, NV, VulkanCompute.F(eps) }, NV);
            vk.Dispatch(kSwi, sSwi, new uint[] { INNER }, (INNER + 255) / 256);
            vk.SubmitAndWait();
            AssertClose(expect, Down(gOut, INNER), 1e-3, 1e-3, $"GDN token {t}");
        }
    }

    // Managed mirror of Qwen3HybridModel.ForwardGdnLayerWithNorm (conv+SiLU, L2 norm, delta-rule update, head RMSNorm, SiLU(z) gate).
    private static float[] GdnReference(int nK, int nV, float[] qkv, float[] z, float[] alpha, float[] betaIn, float[] convW, float[] dtBias,
                                        float[] aW, float[] normW, float eps, float[] convState, float[][] S)
    {
        int ch = (nK * 2 + nV) * GdnD, inner = nV * GdnD, qLen = nK * GdnD;
        var conv = new float[ch];
        for (int c = 0; c < ch; c++)
        {
            float acc = convState[c * 3] * convW[c * 4] + convState[c * 3 + 1] * convW[c * 4 + 1] +
                        convState[c * 3 + 2] * convW[c * 4 + 2] + qkv[c] * convW[c * 4 + 3];
            convState[c * 3] = convState[c * 3 + 1]; convState[c * 3 + 1] = convState[c * 3 + 2]; convState[c * 3 + 2] = qkv[c];
            conv[c] = acc / (1f + MathF.Exp(-acc));
        }
        float qkScale = 1f / MathF.Sqrt(GdnD);
        for (int g = 0; g < nK; g++)
        {
            float sq = 0, sk = 0;
            for (int d = 0; d < GdnD; d++) { sq += conv[g * GdnD + d] * conv[g * GdnD + d]; sk += conv[qLen + g * GdnD + d] * conv[qLen + g * GdnD + d]; }
            float iq = sq > 1e-12f ? (1f / MathF.Sqrt(sq + 1e-6f)) * qkScale : 0f;
            float ik = sk > 1e-12f ? 1f / MathF.Sqrt(sk + 1e-6f) : 0f;
            for (int d = 0; d < GdnD; d++) { conv[g * GdnD + d] *= iq; conv[qLen + g * GdnD + d] *= ik; }
        }
        var y = new float[inner]; var delta = new float[GdnD];
        for (int h = 0; h < nV; h++)
        {
            int g = h % nK;                                  // tiled mapping (llama.cpp: iq1 = iv1 % nK)
            int qo = g * GdnD, ko = qLen + g * GdnD, vo = 2 * qLen + h * GdnD;
            float a = alpha[h] + dtBias[h];
            float dt = a > 20f ? a : MathF.Log(1f + MathF.Exp(a));
            float decay = MathF.Exp(aW[h] * dt);
            float beta = 1f / (1f + MathF.Exp(-betaIn[h]));
            var s = S[h];
            for (int i = 0; i < GdnD; i++)
            {
                float kv = 0;
                for (int j = 0; j < GdnD; j++) { float dc = s[i * GdnD + j] * decay; s[i * GdnD + j] = dc; kv += dc * conv[ko + j]; }
                delta[i] = (conv[vo + i] - kv) * beta;
            }
            for (int i = 0; i < GdnD; i++)
            {
                float dot = 0;
                for (int j = 0; j < GdnD; j++) { float u = s[i * GdnD + j] + delta[i] * conv[ko + j]; s[i * GdnD + j] = u; dot += u * conv[qo + j]; }
                y[h * GdnD + i] = dot;
            }
            float ss = 0; for (int i = 0; i < GdnD; i++) ss += y[h * GdnD + i] * y[h * GdnD + i];
            float r = 1f / MathF.Sqrt(ss / GdnD + eps);
            for (int i = 0; i < GdnD; i++) y[h * GdnD + i] = y[h * GdnD + i] * r * normW[i];
        }
        var o = new float[inner];
        for (int i = 0; i < inner; i++) o[i] = y[i] * (z[i] / (1f + MathF.Exp(-z[i])));
        return o;
    }

    // ---------- Full attention ----------
    private const int AttD = 256, AttRope = 64;

    [VulkanFact]
    public void FullAttention_QkNormRopeKvCacheAndGate_MatchReference()
    {
        const int NQ = 4, NKV = 2, MAXSEQ = 16, TOKENS = 6;
        const float eps = 1e-6f, freqBase = 10000000f;
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(41);

        float[] qNorm = Rand(rng, AttD, 0.1f, 1f), kNorm = Rand(rng, AttD, 0.1f, 1f);
        var kPrep = vk.LoadEmbedded("attn_prep", 9, 28); var kDec = vk.LoadEmbedded("attn_decode", 6, 20);
        var gQF = vk.Alloc(NQ * 512L * 4); var gKR = vk.Alloc(NKV * AttD * 4L); var gVR = vk.Alloc(NKV * AttD * 4L);
        var gQN = Up(vk, qNorm); var gKN = Up(vk, kNorm);
        var gQ = vk.Alloc(NQ * AttD * 4L); var gGate = vk.Alloc(NQ * AttD * 4L);
        var gKC = vk.Alloc((long)NKV * MAXSEQ * AttD * 4); var gVC = vk.Alloc((long)NKV * MAXSEQ * AttD * 4); gKC.Zero(); gVC.Zero();
        var gSc = vk.Alloc((long)NQ * MAXSEQ * 4); var gOut = vk.Alloc(NQ * AttD * 4L);
        var sPrep = vk.MakeSet(kPrep, gQF, gKR, gVR, gQN, gKN, gQ, gGate, gKC, gVC);
        var sDec = vk.MakeSet(kDec, gQ, gGate, gKC, gVC, gSc, gOut);

        var refK = new float[NKV][]; var refV = new float[NKV][];
        for (int i = 0; i < NKV; i++) { refK[i] = new float[MAXSEQ * AttD]; refV[i] = new float[MAXSEQ * AttD]; }

        for (int pos = 0; pos < TOKENS; pos++)
        {
            float[] qf = Rand(rng, NQ * 512), kr = Rand(rng, NKV * AttD), vr = Rand(rng, NKV * AttD);
            float[] expect = AttentionReference(NQ, NKV, qf, kr, vr, qNorm, kNorm, pos, eps, freqBase, refK, refV);
            gQF.Upload(qf); gKR.Upload(kr); gVR.Upload(vr);
            vk.Begin();
            vk.Dispatch(kPrep, sPrep, new uint[] { (uint)pos, NQ, NKV, MAXSEQ, AttRope, VulkanCompute.F(eps), VulkanCompute.F(freqBase) }, NQ + 2 * NKV);
            vk.Dispatch(kDec, sDec, new uint[] { (uint)(pos + 1), NQ, NKV, MAXSEQ, VulkanCompute.F(1f / MathF.Sqrt(AttD)) }, NQ);
            vk.SubmitAndWait();
            AssertClose(expect, Down(gOut, NQ * AttD), 2e-3, 2e-3, $"attention pos {pos}");
        }
    }

    private static void RmsInPlace(float[] x, int off, float[] w, float eps)
    {
        float ss = 0; for (int i = 0; i < AttD; i++) ss += x[off + i] * x[off + i];
        float r = 1f / MathF.Sqrt(ss / AttD + eps);
        for (int i = 0; i < AttD; i++) x[off + i] = x[off + i] * r * w[i];
    }

    private static void RopeNeox(float[] x, int off, int pos, float freqBase)
    {
        int half = AttRope / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = 1f / MathF.Pow(freqBase, (float)(2 * i) / AttRope);
            float theta = pos * freq, c = MathF.Cos(theta), s = MathF.Sin(theta);
            float a = x[off + i], b = x[off + i + half];
            x[off + i] = a * c - b * s; x[off + i + half] = a * s + b * c;
        }
    }

    // Managed mirror of Qwen3HybridModel.ForwardStandardAttentionLayer after the Q/K/V projections.
    private static float[] AttentionReference(int nQ, int nKv, float[] qFull, float[] kRaw, float[] vRaw, float[] qNorm, float[] kNorm,
                                              int pos, float eps, float freqBase, float[][] kc, float[][] vc)
    {
        var q = new float[nQ * AttD]; var gate = new float[nQ * AttD];
        for (int h = 0; h < nQ; h++)
        {
            Array.Copy(qFull, h * 512, q, h * AttD, AttD);
            Array.Copy(qFull, h * 512 + AttD, gate, h * AttD, AttD);
            RmsInPlace(q, h * AttD, qNorm, eps); RopeNeox(q, h * AttD, pos, freqBase);
        }
        var k = (float[])kRaw.Clone();
        for (int h = 0; h < nKv; h++)
        {
            RmsInPlace(k, h * AttD, kNorm, eps); RopeNeox(k, h * AttD, pos, freqBase);
            Array.Copy(k, h * AttD, kc[h], pos * AttD, AttD);
            Array.Copy(vRaw, h * AttD, vc[h], pos * AttD, AttD);
        }
        var o = new float[nQ * AttD];
        int groupSize = nQ / nKv; float scale = 1f / MathF.Sqrt(AttD);
        for (int h = 0; h < nQ; h++)
        {
            int kvh = h / groupSize;
            var sc = new float[pos + 1]; float mx = float.MinValue;
            for (int t = 0; t <= pos; t++)
            {
                float dot = 0; for (int d = 0; d < AttD; d++) dot += q[h * AttD + d] * kc[kvh][t * AttD + d];
                sc[t] = dot * scale; mx = MathF.Max(mx, sc[t]);
            }
            float sum = 0; for (int t = 0; t <= pos; t++) { sc[t] = MathF.Exp(sc[t] - mx); sum += sc[t]; }
            for (int d = 0; d < AttD; d++)
            {
                float acc = 0; for (int t = 0; t <= pos; t++) acc += sc[t] / sum * vc[kvh][t * AttD + d];
                float g = gate[h * AttD + d];
                o[h * AttD + d] = acc * (1f / (1f + MathF.Exp(-g)));
            }
        }
        return o;
    }

    private static float[] RandF(Random rng, int n, float scale)
    {
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return r;
    }

    [VulkanFact]
    public void Iq4Xs_And_Mxfp4_MatVec_MatchCpuReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(42);

        // 1. IQ4_XS: K=256, M=4 -> 136 bytes/row
        int[] iq4nl = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
        int kIq = 256, mIq = 4;
        byte[] wIq = new byte[mIq * 136];
        rng.NextBytes(wIq);
        for (int r = 0; r < mIq; r++)
        {
            Half h = (Half)(0.05f * (r + 1));
            BitConverter.TryWriteBytes(wIq.AsSpan(r * 136, 2), h);
        }
        var xIq = RandF(rng, kIq, 0.5f);
        var refIq = new float[mIq];
        for (int r = 0; r < mIq; r++)
        {
            int bo = r * 136;
            float d = (float)BitConverter.ToHalf(wIq, bo);
            ushort sh = BitConverter.ToUInt16(wIq, bo + 2);
            uint sl = BitConverter.ToUInt32(wIq, bo + 4);
            float sum = 0;
            for (int sub = 0; sub < 8; sub++)
            {
                int ls = (int)(((sl >> (4 * sub)) & 0xFu) | (((sh >> (2 * sub)) & 3u) << 4));
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++)
                {
                    byte q = wIq[bo + 8 + sub * 16 + j];
                    sum += dl * (iq4nl[q & 0xF] * xIq[sub * 32 + j] + iq4nl[q >> 4] * xIq[sub * 32 + 16 + j]);
                }
            }
            refIq[r] = sum;
        }
        var kQ4 = vk.LoadEmbedded("qmatvec_q4", 3, 8);
        var gWIq = Up(vk, wIq); var gXIq = Up(vk, xIq); var gYIq = vk.Alloc(mIq * 4L);
        vk.Begin();
        vk.Dispatch(kQ4, vk.MakeSet(kQ4, gWIq, gXIq, gYIq), new uint[] { (uint)kIq, (uint)mIq }, (uint)mIq);
        vk.SubmitAndWait();
        AssertClose(refIq, Down(gYIq, mIq), 1e-3, 1e-4, "IQ4_XS matvec");

        // 2. MXFP4: K=64 (2 blocks), M=4 -> 34 bytes/row
        float[] mxfp4 = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0f, -0.5f, -1, -1.5f, -2, -3, -4, -6 };
        int kMx = 64, mMx = 4;
        byte[] wMx = new byte[mMx * 2 * 17];
        rng.NextBytes(wMx);
        for (int r = 0; r < mMx; r++)
            for (int b = 0; b < 2; b++)
                wMx[(r * 2 + b) * 17] = (byte)(126 + (r + b) % 4);
        var xMx = RandF(rng, kMx, 0.5f);
        var refMx = new float[mMx];
        for (int r = 0; r < mMx; r++)
        {
            float sum = 0;
            for (int b = 0; b < 2; b++)
            {
                int bo = (r * 2 + b) * 17;
                byte e = wMx[bo];
                float d = e == 0 ? 0f : BitConverter.UInt32BitsToSingle((uint)e << 23) * 0.5f;
                for (int j = 0; j < 16; j++)
                {
                    byte q = wMx[bo + 1 + j];
                    sum += d * (mxfp4[q & 0xF] * xMx[b * 32 + j] + mxfp4[q >> 4] * xMx[b * 32 + 16 + j]);
                }
            }
            refMx[r] = sum;
        }
        var kQ5 = vk.LoadEmbedded("qmatvec_q5", 3, 8);
        var gWMx = Up(vk, wMx); var gXMx = Up(vk, xMx); var gYMx = vk.Alloc(mMx * 4L);
        vk.Begin();
        vk.Dispatch(kQ5, vk.MakeSet(kQ5, gWMx, gXMx, gYMx), new uint[] { (uint)kMx, (uint)mMx }, (uint)mMx);
        vk.SubmitAndWait();
        AssertClose(refMx, Down(gYMx, mMx), 1e-3, 1e-4, "MXFP4 matvec");
    }

    [VulkanFact]
    public void StandardGqa_And_GeGlu_MatchReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(7);

        int nQ = 4, nKv = 2, hd = 64, maxSeq = 8;
        var qRaw = RandF(rng, nQ * hd, 0.5f);
        var kRaw = RandF(rng, nKv * hd, 0.5f);
        var vRaw = RandF(rng, nKv * hd, 0.5f);
        var dummy = new float[hd];

        var kPrep = vk.LoadEmbedded("gqa_prep", 8, 40);
        var kDec = vk.LoadEmbedded("gqa_decode", 4, 24);
        var kGe = vk.LoadEmbedded("elementwise_geglu", 3, 4);
        var kSc = vk.LoadEmbedded("gemma4_scale", 2, 12);

        var gQ = Up(vk, qRaw); var gK = Up(vk, kRaw); var gV = Up(vk, vRaw); var gDum = Up(vk, dummy);
        var gQO = vk.Alloc(nQ * hd * 4L);
        var gKC = vk.Alloc(nKv * maxSeq * hd * 4L);
        var gVC = vk.Alloc(nKv * maxSeq * hd * 4L);
        var gAO = vk.Alloc(nQ * hd * 4L);

        vk.Begin();
        vk.Dispatch(kPrep, vk.MakeSet(kPrep, gQ, gK, gV, gDum, gDum, gQO, gKC, gVC),
            new uint[] { 0u, (uint)nQ, (uint)nKv, (uint)maxSeq, (uint)hd, (uint)hd, 0u, 1u, VulkanCompute.F(1e-6f), VulkanCompute.F(10000f) },
            (uint)(nQ + 2 * nKv));
        vk.Dispatch(kDec, vk.MakeSet(kDec, gQO, gKC, gVC, gAO),
            new uint[] { 1u, (uint)nQ, (uint)nKv, (uint)maxSeq, (uint)hd, VulkanCompute.F(1f / MathF.Sqrt(hd)) },
            (uint)nQ);
        vk.SubmitAndWait();

        // At pos=0 (seqLen=1), softmax weight is 1.0 so output == vRaw for each KV group
        var outCpu = Down(gAO, nQ * hd);
        for (int h = 0; h < nQ; h++)
        {
            int kh = h / (nQ / nKv);
            for (int d = 0; d < hd; d++)
                Assert.True(MathF.Abs(outCpu[h * hd + d] - vRaw[kh * hd + d]) < 1e-4f);
        }

        // GeGLU + gemma4_scale softcap
        var ga = RandF(rng, 64, 1.0f); var gb = RandF(rng, 64, 1.0f);
        var gGa = Up(vk, ga); var gGb = Up(vk, gb); var gGo = vk.Alloc(64 * 4L);
        vk.Begin();
        vk.Dispatch(kGe, vk.MakeSet(kGe, gGa, gGb, gGo), new uint[] { 64u }, 1);
        vk.Dispatch(kSc, vk.MakeSet(kSc, gGo, gGo), new uint[] { 64u, 1u, VulkanCompute.F(30f) }, 1);
        vk.SubmitAndWait();
        var refGe = new float[64];
        for (int i = 0; i < 64; i++)
        {
            float x = ga[i];
            float g = 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x))) * gb[i];
            refGe[i] = 30f * MathF.Tanh(g / 30f);
        }
        AssertClose(refGe, Down(gGo, 64), 1e-4, 1e-5, "GeGLU + softcap");
    }

    [VulkanFact]
    public void Mamba2_Conv_And_Recurrence_MatchReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(19);

        int nH = 4, hd = 16, sd = 8, nG = 2, dIn = nH * hd, kern = 4;
        int convDim = dIn + 2 * nG * sd;
        var xbc = RandF(rng, convDim, 0.5f);
        var convW = RandF(rng, convDim * kern, 0.3f);
        var convB = RandF(rng, convDim, 0.1f);
        var convSt = new float[convDim * (kern - 1)];
        var z = RandF(rng, dIn, 0.5f);
        var dtIn = RandF(rng, nH, 0.5f);
        var A = new float[nH]; for (int i = 0; i < nH; i++) A[i] = -0.5f - 0.1f * i;
        var D = RandF(rng, nH, 0.5f);
        var dtB = RandF(rng, nH, 0.1f);
        var st = new float[nH * hd * sd];

        var kConv = vk.LoadEmbedded("mamba2_conv", 5, 12);
        var kRec = vk.LoadEmbedded("mamba2_rec", 8, 28);
        var gXbc = Up(vk, xbc); var gCW = Up(vk, convW); var gCB = Up(vk, convB);
        var gCSt = Up(vk, convSt); var gXbcO = vk.Alloc(convDim * 4L);
        var gZ = Up(vk, z); var gDt = Up(vk, dtIn); var gA = Up(vk, A); var gD = Up(vk, D);
        var gDtB = Up(vk, dtB); var gSt = Up(vk, st); var gY = vk.Alloc(dIn * 4L);

        vk.Begin();
        vk.Dispatch(kConv, vk.MakeSet(kConv, gXbc, gCW, gCB, gCSt, gXbcO), new uint[] { (uint)convDim, (uint)kern, 0u }, (uint)((convDim + 255) / 256));
        vk.Dispatch(kRec, vk.MakeSet(kRec, gXbcO, gZ, gDt, gA, gD, gDtB, gSt, gY), new uint[] { (uint)nH, (uint)hd, (uint)sd, (uint)nG, (uint)dIn, 0u, 0u }, (uint)nH);
        vk.SubmitAndWait();

        var xbcRef = new float[convDim];
        for (int c = 0; c < convDim; c++)
        {
            float s = convB[c] + xbc[c] * convW[c * kern + kern - 1];
            xbcRef[c] = s / (1f + MathF.Exp(-s));
        }
        var yRef = new float[dIn];
        for (int h = 0; h < nH; h++)
        {
            int g = h / (nH / nG);
            float dtRaw = dtIn[h] + dtB[h];
            float dt = dtRaw > 20f ? dtRaw : MathF.Log(1f + MathF.Exp(dtRaw));
            for (int d = 0; d < hd; d++)
            {
                float xv = xbcRef[h * hd + d];
                float acc = D[h] * xv;
                for (int s = 0; s < sd; s++)
                {
                    float sv = dt * xbcRef[dIn + g * sd + s] * xv;
                    acc += sv * xbcRef[dIn + (nG + g) * sd + s];
                }
                float zv = z[h * hd + d];
                yRef[h * hd + d] = acc * (zv / (1f + MathF.Exp(-zv)));
            }
        }
        AssertClose(yRef, Down(gY, dIn), 1e-4, 1e-5, "Mamba-2 conv+rec");
    }

    [VulkanFact]
    public void Mla_And_DsaTopk_MatchReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(99);

        int nQ = 2, maxSeq = 4, qkNope = 32, qkRope = 16, vDim = 32, kvRank = 16;
        int kDim = qkNope + qkRope;
        var qFull = RandF(rng, nQ * kDim, 0.5f);
        var kvB = RandF(rng, nQ * (qkNope + vDim), 0.5f);
        var kvA = RandF(rng, kvRank + qkRope, 0.5f);

        var kPrep = vk.LoadEmbedded("mla_prep", 6, 36);
        var kDec = vk.LoadEmbedded("mla_decode", 4, 24);
        var kTop = vk.LoadEmbedded("dsa_topk", 5, 16);
        var kSp = vk.LoadEmbedded("mla_sparse_decode", 5, 24);

        var gQF = Up(vk, qFull); var gKvB = Up(vk, kvB); var gKvA = Up(vk, kvA);
        var gQO = vk.Alloc(nQ * kDim * 4L);
        var gKC = vk.Alloc(nQ * maxSeq * kDim * 4L);
        var gVC = vk.Alloc(nQ * maxSeq * vDim * 4L);
        var gAO = vk.Alloc(nQ * vDim * 4L);

        vk.Begin();
        vk.Dispatch(kPrep, vk.MakeSet(kPrep, gQF, gKvB, gKvA, gQO, gKC, gVC),
            new uint[] { 0u, (uint)nQ, (uint)maxSeq, (uint)qkNope, (uint)qkRope, (uint)vDim, (uint)kvRank, 1u, VulkanCompute.F(10000f) }, (uint)nQ);
        vk.Dispatch(kDec, vk.MakeSet(kDec, gQO, gKC, gVC, gAO),
            new uint[] { 1u, (uint)nQ, (uint)maxSeq, (uint)kDim, (uint)vDim, VulkanCompute.F(1f / MathF.Sqrt(kDim)) }, (uint)nQ);
        vk.SubmitAndWait();
        var denseOut = Down(gAO, nQ * vDim);

        // Verify DSA topk + mla_sparse_decode at seqLen=1 matches denseOut
        var gIQ = Up(vk, RandF(rng, 2 * 8, 0.5f)); var gIK = vk.Alloc(maxSeq * 8 * 4L); var gIKCur = Up(vk, RandF(rng, 8, 0.5f));
        var gIW = Up(vk, RandF(rng, 2, 0.5f)); var gTopIdx = vk.Alloc(4L); var gSpO = vk.Alloc(nQ * vDim * 4L);
        vk.Begin();
        vk.Dispatch(kTop, vk.MakeSet(kTop, gIQ, gIK, gIW, gTopIdx, gIKCur), new uint[] { 1u, 2u, 8u, 1u }, 1);
        vk.Dispatch(kSp, vk.MakeSet(kSp, gQO, gKC, gVC, gTopIdx, gSpO),
            new uint[] { 1u, (uint)nQ, (uint)maxSeq, (uint)kDim, (uint)vDim, VulkanCompute.F(1f / MathF.Sqrt(kDim)) }, (uint)nQ);
        vk.SubmitAndWait();
        AssertClose(denseOut, Down(gSpO, nQ * vDim), 1e-5, 1e-6, "MLA sparse vs dense");
    }

    [VulkanFact]
    public void Mamba2_PackedOffsets_And_MultiStep_MatchReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(123);

        int nH = 4, hd = 8, sd = 6, nG = 2, dIn = nH * hd, kern = 4;
        int convDim = dIn + 2 * nG * sd;
        int projDim = dIn + convDim + nH;
        uint zOff = 0u, inOff = (uint)dIn, dtOff = (uint)(dIn + convDim);

        var convW = RandF(rng, convDim * kern, 0.3f);
        var convB = RandF(rng, convDim, 0.1f);
        var convSt = new float[convDim * (kern - 1)];
        var A = new float[nH]; for (int i = 0; i < nH; i++) A[i] = -0.4f - 0.15f * i;
        var D = RandF(rng, nH, 0.5f);
        var dtB = RandF(rng, nH, 0.1f);
        var st = new float[nH * hd * sd];

        var kConv = vk.LoadEmbedded("mamba2_conv", 5, 12);
        var kRec = vk.LoadEmbedded("mamba2_rec", 8, 28);
        var gProj = vk.Alloc(projDim * 4L);
        var gCW = Up(vk, convW); var gCB = Up(vk, convB);
        var gCSt = vk.Alloc(convSt.Length * 4L); gCSt.Zero();
        var gXbcO = vk.Alloc(convDim * 4L);
        var gA = Up(vk, A); var gD = Up(vk, D); var gDtB = Up(vk, dtB);
        var gSt = vk.Alloc(st.Length * 4L); gSt.Zero();
        var gY = vk.Alloc(dIn * 4L);

        var convSet = vk.MakeSet(kConv, gProj, gCW, gCB, gCSt, gXbcO);
        var recSet = vk.MakeSet(kRec, gXbcO, gProj, gProj, gA, gD, gDtB, gSt, gY);

        for (int step = 0; step < 3; step++)
        {
            var proj = RandF(rng, projDim, 0.5f);
            fixed (float* pProj = proj) gProj.UploadFrom(pProj, proj.Length * 4L);

            vk.Begin();
            vk.Dispatch(kConv, convSet, new uint[] { (uint)convDim, (uint)kern, inOff }, (uint)((convDim + 255) / 256));
            vk.Dispatch(kRec, recSet, new uint[] { (uint)nH, (uint)hd, (uint)sd, (uint)nG, (uint)dIn, zOff, dtOff }, (uint)nH);
            vk.SubmitAndWait();

            int k1 = kern - 1;
            var xbcRef = new float[convDim];
            for (int c = 0; c < convDim; c++)
            {
                float sum = convB[c];
                for (int j = 0; j < k1; j++) sum += convSt[c * k1 + j] * convW[c * kern + j];
                float cur = proj[inOff + c];
                sum += cur * convW[c * kern + k1];
                for (int j = 0; j + 1 < k1; j++) convSt[c * k1 + j] = convSt[c * k1 + j + 1];
                convSt[c * k1 + k1 - 1] = cur;
                xbcRef[c] = sum / (1f + MathF.Exp(-sum));
            }

            var yRef = new float[dIn];
            for (int h = 0; h < nH; h++)
            {
                int g = h / (nH / nG);
                float dtRaw = proj[dtOff + h] + dtB[h];
                float dt = dtRaw > 20f ? dtRaw : MathF.Log(1f + MathF.Exp(dtRaw));
                float dA = MathF.Exp(dt * A[h]);
                for (int d = 0; d < hd; d++)
                {
                    float xv = xbcRef[h * hd + d];
                    float acc = D[h] * xv;
                    int stBase = (h * hd + d) * sd;
                    for (int s = 0; s < sd; s++)
                    {
                        float sv = st[stBase + s] * dA + dt * xbcRef[dIn + g * sd + s] * xv;
                        st[stBase + s] = sv;
                        acc += sv * xbcRef[dIn + (nG + g) * sd + s];
                    }
                    float zv = proj[zOff + h * hd + d];
                    yRef[h * hd + d] = acc * (zv / (1f + MathF.Exp(-zv)));
                }
            }

            AssertClose(yRef, Down(gY, dIn), 1e-4, 1e-5, $"Mamba-2 packed multi-step {step}");
        }
    }

    [VulkanFact]
    public void DsaTopk_MultiStep_CacheAppend_MatchReference()
    {
        using var ctx = new VulkanContext(VulkanFactAttribute.DeviceOrdinal);
        using var vk = new VulkanCompute(ctx);
        var rng = new Random(456);

        int maxSeq = 6, nIdxH = 2, idxD = 8, topK = 2;
        var kTop = vk.LoadEmbedded("dsa_topk", 5, 16);
        var gIQ = vk.Alloc(nIdxH * idxD * 4L);
        var gIKCache = vk.Alloc(maxSeq * idxD * 4L); gIKCache.Zero();
        var gIW = vk.Alloc(nIdxH * 4L);
        var gTopIdx = vk.Alloc(topK * 4L);
        var gIKCur = vk.Alloc(idxD * 4L);
        var topSet = vk.MakeSet(kTop, gIQ, gIKCache, gIW, gTopIdx, gIKCur);

        var kCacheRef = new float[maxSeq * idxD];
        for (int seqLen = 1; seqLen <= 4; seqLen++)
        {
            var q = RandF(rng, nIdxH * idxD, 0.5f);
            var w = RandF(rng, nIdxH, 0.5f);
            var kCur = RandF(rng, idxD, 0.5f);
            Array.Copy(kCur, 0, kCacheRef, (seqLen - 1) * idxD, idxD);

            fixed (float* pQ = q) gIQ.UploadFrom(pQ, q.Length * 4L);
            fixed (float* pW = w) gIW.UploadFrom(pW, w.Length * 4L);
            fixed (float* pK = kCur) gIKCur.UploadFrom(pK, kCur.Length * 4L);

            vk.Begin();
            vk.Dispatch(kTop, topSet, new uint[] { (uint)seqLen, (uint)nIdxH, (uint)idxD, (uint)topK }, 1);
            vk.SubmitAndWait();

            int k = Math.Min(topK, seqLen);
            var gpuTop = new uint[k];
            fixed (uint* pTop = gpuTop) gTopIdx.DownloadTo(pTop, gpuTop.Length * 4L);

            var scores = new float[seqLen];
            for (int pos = 0; pos < seqLen; pos++)
            {
                float s = 0f;
                for (int h = 0; h < nIdxH; h++)
                {
                    float dot = 0f;
                    for (int d = 0; d < idxD; d++) dot += q[h * idxD + d] * kCacheRef[pos * idxD + d];
                    s += MathF.Max(dot, 0f) * w[h];
                }
                scores[pos] = s;
            }

            for (int i = 0; i < k; i++)
            {
                float best = -1e30f;
                uint bestIdx = 0;
                for (uint pos = 0; pos < (uint)seqLen; pos++)
                {
                    if (scores[pos] > best) { best = scores[pos]; bestIdx = pos; }
                }
                Assert.Equal(bestIdx, gpuTop[i]);
                scores[bestIdx] = -1e30f;
            }
        }
    }
}
