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
}
