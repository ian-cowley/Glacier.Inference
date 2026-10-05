namespace Glacier.Inference.Image.Flux;

using Glacier.Inference.Diagnostics;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.Inference.Format;

/// <summary>
/// Production Pure C# Neural Variational Autoencoder (VAE) Decoder for FLUX.1.
/// Reconstructs photorealistic RGB images from 16-channel latent fields using
/// the official 244-tensor convolutional ResNet + Self-Attention network (ae.safetensors).
/// </summary>
public unsafe sealed partial class FluxVaeDecoder : IDisposable
{
    private readonly SafetensorsFile _weights;
    private bool _disposed;

    private readonly Glacier.Inference.Gpu.GpuContext? _gpu;
    private readonly IntPtr _gpuModule;
    private readonly IntPtr _fnVaeScaleLatents;
    private readonly IntPtr _fnConv2d3x3;
    private readonly IntPtr _fnConv2d1x1;
    private readonly IntPtr _fnGroupNormSilu;
    private readonly IntPtr _fnUpsample2x;
    private readonly IntPtr _fnTensorAdd;
    private readonly IntPtr _fnVaeSpatialAttn;
    private readonly IntPtr _fnVaeClampRgb;
    private readonly Dictionary<string, IntPtr> _gpuWeights = new(StringComparer.Ordinal);
    private IntPtr _dBufA;
    private IntPtr _dBufB;
    private IntPtr _dNorm1Buf;
    private IntPtr _dConv1Buf;
    private IntPtr _dNorm2Buf;
    private IntPtr _dLatents;
    private IntPtr _dRgbDevice;
    private nuint _allocatedActivationBytes;

    // Latent scaling constants from Black Forest Labs FLUX.1 specification:
    // z = (latent / 0.3611) + 0.1159
    public const float ScaleFactor = 0.3611f;
    public const float ShiftFactor = 0.1159f;
    public string ActiveBackend => _gpu != null ? "Cuda" : "Cpu";

    public FluxVaeDecoder(SafetensorsFile weights, bool enableGpu = true)
    {
        _weights = weights;
        if (enableGpu && Glacier.Inference.Gpu.GpuContext.IsSupported)
        {
            try
            {
                _gpu = new Glacier.Inference.Gpu.GpuContext(0);
                byte[] cubin = Glacier.Inference.Gpu.KernelCompiler.GetOrCompileKernels(_gpu.ArchString);
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleLoadData(out _gpuModule, cubin), "ModuleLoadData");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnVaeScaleLatents, _gpuModule, "vae_scale_latents"), "vae_scale_latents");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnConv2d3x3, _gpuModule, "conv2d_3x3"), "conv2d_3x3");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnConv2d1x1, _gpuModule, "conv2d_1x1"), "conv2d_1x1");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnGroupNormSilu, _gpuModule, "group_norm_silu"), "group_norm_silu");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnUpsample2x, _gpuModule, "upsample2x_nearest"), "upsample2x_nearest");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnTensorAdd, _gpuModule, "tensor_add"), "tensor_add");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnVaeSpatialAttn, _gpuModule, "vae_spatial_attention"), "vae_spatial_attention");
                Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.ModuleGetFunction(out _fnVaeClampRgb, _gpuModule, "vae_clamp_rgb"), "vae_clamp_rgb");
                GlacierDiagnostics.LogInformation($"[FLUX VAE] Initialized Neural VAE GPU backend on {_gpu.DeviceName} ({_gpu.ArchString}).");
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[FLUX VAE] GPU init skipped [{ex.GetType().Name}]: {ex.Message}, falling back to CPU SIMD.", ex);
                _gpu?.Dispose();
                _gpu = null;
            }
        }
    }

    public static FluxVaeDecoder Open(string safetensorsPath, bool enableGpu = true)
    {
        var sf = SafetensorsFile.Open(safetensorsPath);
        return new FluxVaeDecoder(sf, enableGpu);
    }

    private float* GetWeight(string name) => (float*)_weights.GetTensorPointer(name);
    private bool HasTensor(string name) => _weights.ContainsTensor(name);

    private IntPtr GetGpuWeight(string name, int floatCount)
    {
        if (_gpu == null) return IntPtr.Zero;
        if (_gpuWeights.TryGetValue(name, out var d)) return d;
        nuint bytes = (nuint)((long)floatCount * sizeof(float));
        IntPtr dMem = _gpu.AllocateDevice(bytes);
        _gpu.CopyToDevice(dMem, (IntPtr)GetWeight(name), bytes);
        _gpuWeights[name] = dMem;
        return dMem;
    }

    private void EnsureGpuBuffers(int maxFloats, int targetH, int targetW)
    {
        nuint featBytes = (nuint)((long)maxFloats * sizeof(float));
        if (_allocatedActivationBytes >= featBytes && _dBufA != IntPtr.Zero) return;

        FreeGpuBuffers();

        _dBufA = _gpu!.AllocateDevice(featBytes);
        _dBufB = _gpu!.AllocateDevice(featBytes);
        _dNorm1Buf = _gpu!.AllocateDevice(featBytes);
        _dConv1Buf = _gpu!.AllocateDevice(featBytes);
        _dNorm2Buf = _gpu!.AllocateDevice(featBytes);
        _dLatents = _gpu!.AllocateDevice((nuint)(16 * (targetH / 8) * (targetW / 8) * sizeof(float)));
        _dRgbDevice = _gpu!.AllocateDevice((nuint)(targetH * targetW * 3));
        _allocatedActivationBytes = featBytes;
    }

    private void FreeGpuBuffers()
    {
        if (_gpu != null)
        {
            if (_dBufA != IntPtr.Zero) { _gpu.FreeDevice(_dBufA); _dBufA = IntPtr.Zero; }
            if (_dBufB != IntPtr.Zero) { _gpu.FreeDevice(_dBufB); _dBufB = IntPtr.Zero; }
            if (_dNorm1Buf != IntPtr.Zero) { _gpu.FreeDevice(_dNorm1Buf); _dNorm1Buf = IntPtr.Zero; }
            if (_dConv1Buf != IntPtr.Zero) { _gpu.FreeDevice(_dConv1Buf); _dConv1Buf = IntPtr.Zero; }
            if (_dNorm2Buf != IntPtr.Zero) { _gpu.FreeDevice(_dNorm2Buf); _dNorm2Buf = IntPtr.Zero; }
            if (_dLatents != IntPtr.Zero) { _gpu.FreeDevice(_dLatents); _dLatents = IntPtr.Zero; }
            if (_dRgbDevice != IntPtr.Zero) { _gpu.FreeDevice(_dRgbDevice); _dRgbDevice = IntPtr.Zero; }
        }
        _allocatedActivationBytes = 0;
    }

    private void LaunchConv3x3(IntPtr dSrc, IntPtr dDst, int inC, int outC, int h, int w, IntPtr dW, IntPtr dB)
    {
        uint gridX = (uint)((w + 15) / 16);
        uint gridY = (uint)((h + 15) / 16);
        uint gridZ = (uint)outC;
        uint blockDimX = 16;
        uint blockDimY = 16;
        uint blockDimZ = 1;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dW;
        pArgs[3] = &dB;
        pArgs[4] = &inC;
        pArgs[5] = &outC;
        pArgs[6] = &h;
        pArgs[7] = &w;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnConv2d3x3, gridX, gridY, gridZ, blockDimX, blockDimY, blockDimZ,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(conv2d_3x3)");
    }

    private void LaunchConv1x1(IntPtr dSrc, IntPtr dDst, int inC, int outC, int spatial, IntPtr dW, IntPtr dB)
    {
        uint gridX = (uint)((spatial + 15) / 16);
        uint gridY = (uint)((outC + 15) / 16);
        uint gridZ = 1;
        uint blockDimX = 16;
        uint blockDimY = 16;
        uint blockDimZ = 1;

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dW;
        pArgs[3] = &dB;
        pArgs[4] = &inC;
        pArgs[5] = &outC;
        pArgs[6] = &spatial;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnConv2d1x1, gridX, gridY, gridZ, blockDimX, blockDimY, blockDimZ,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(conv2d_1x1)");
    }

    private void LaunchGroupNormSilu(IntPtr dSrc, IntPtr dDst, int channels, int spatial, int groups, IntPtr dW, IntPtr dB, bool applySilu)
    {
        uint gridX = (uint)groups;
        uint gridY = 1;
        uint gridZ = 1;
        uint blockSize = 256;
        int siluFlag = applySilu ? 1 : 0;

        void** pArgs = stackalloc void*[8];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &dW;
        pArgs[3] = &dB;
        pArgs[4] = &channels;
        pArgs[5] = &spatial;
        pArgs[6] = &groups;
        pArgs[7] = &siluFlag;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnGroupNormSilu, gridX, gridY, gridZ, blockSize, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(group_norm_silu)");
    }

    private void LaunchUpsample2x(IntPtr dSrc, IntPtr dDst, int channels, int inH, int inW)
    {
        int outW = inW * 2;
        int outH = inH * 2;
        uint gridX = (uint)((outW + 15) / 16);
        uint gridY = (uint)((outH + 15) / 16);
        uint gridZ = (uint)channels;

        void** pArgs = stackalloc void*[5];
        pArgs[0] = &dSrc;
        pArgs[1] = &dDst;
        pArgs[2] = &channels;
        pArgs[3] = &inH;
        pArgs[4] = &inW;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnUpsample2x, gridX, gridY, gridZ, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(upsample2x_nearest)");
    }

    private void LaunchTensorAdd(IntPtr dTarget, IntPtr dSource, int count)
    {
        uint blockSize = 256;
        uint gridX = (uint)((count + 255) / 256);

        void** pArgs = stackalloc void*[3];
        pArgs[0] = &dTarget;
        pArgs[1] = &dSource;
        pArgs[2] = &count;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnTensorAdd, gridX, 1, 1, blockSize, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(tensor_add)");
    }

    private void LaunchVaeSpatialAttention(IntPtr dQ, IntPtr dK, IntPtr dV, IntPtr dOut, int spatial, int channels)
    {
        uint gridX = (uint)spatial;
        uint blockSize = 128;
        float scale = 1.0f / MathF.Sqrt(channels);

        void** pArgs = stackalloc void*[7];
        pArgs[0] = &dQ;
        pArgs[1] = &dK;
        pArgs[2] = &dV;
        pArgs[3] = &dOut;
        pArgs[4] = &spatial;
        pArgs[5] = &channels;
        pArgs[6] = &scale;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnVaeSpatialAttn, gridX, 1, 1, blockSize, 1, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(vae_spatial_attention)");
    }

    private void LaunchVaeClampRgb(IntPtr dPlanarFloat, IntPtr dPackedRgb, int h, int w)
    {
        uint gridX = (uint)((w + 15) / 16);
        uint gridY = (uint)((h + 15) / 16);

        void** pArgs = stackalloc void*[4];
        pArgs[0] = &dPlanarFloat;
        pArgs[1] = &dPackedRgb;
        pArgs[2] = &h;
        pArgs[3] = &w;

        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnVaeClampRgb, gridX, gridY, 1, 16, 16, 1,
            0, IntPtr.Zero, (IntPtr)pArgs, IntPtr.Zero), "LaunchKernel(vae_clamp_rgb)");
    }

    /// <summary>
    /// Decodes a 16-channel latent field into full-resolution 24-bit RGB pixel buffer.
    /// Input: [16, latentH, latentW]
    /// Output: [latentH * 8, latentW * 8, 3] in RGB bytes
    /// </summary>
    public void Decode(ReadOnlySpan<float> latents, int latentH, int latentW, Span<byte> outputRgb)
    {
        int targetH = latentH * 8;
        int targetW = latentW * 8;
        int requiredBytes = targetH * targetW * 3;
        if (outputRgb.Length < requiredBytes)
            throw new ArgumentException($"Output buffer too small: {outputRgb.Length} < {requiredBytes}");

        if (_gpu != null)
        {
            // For resolutions larger than 64x64 latent (> 512x512 image), use high-speed Tiled VAE Decoding!
            if (latentH > 64 || latentW > 64)
            {
                try
                {
                    DecodeTiledGpu(latents, latentH, latentW, outputRgb);
                    return;
                }
                catch (Exception ex)
                {
                    GlacierDiagnostics.LogWarning($"[FLUX VAE] Tiled GPU decode encountered [{ex.GetType().Name}]: {ex.Message}. Falling back to single-shot GPU / CPU.", ex);
                }
            }

            try
            {
                DecodeGpu(latents, latentH, latentW, outputRgb);
                return;
            }
            catch (Exception ex)
            {
                GlacierDiagnostics.LogWarning($"[FLUX VAE] GPU decode encountered [{ex.GetType().Name}]: {ex.Message}. Falling back to CPU SIMD.", ex);
            }
        }

        DecodeCpu(latents, latentH, latentW, outputRgb);
    }

    private void DecodeTiledGpu(ReadOnlySpan<float> latents, int latentH, int latentW, Span<byte> outputRgb)
    {
        int targetH = latentH * 8;
        int targetW = latentW * 8;

        var ySlices = ComputeTileSlices(latentH, maxTileLen: 68, minOverlap: 8);
        var xSlices = ComputeTileSlices(latentW, maxTileLen: 68, minOverlap: 8);
        int totalTiles = ySlices.Count * xSlices.Count;

        GlacierDiagnostics.LogInformation($"[FLUX VAE] Tiled VAE Decoding active: {xSlices.Count}x{ySlices.Count} ({totalTiles} tiles) for {targetW}x{targetH} canvas...");

        float[] imageAccum = new float[targetH * targetW * 3];
        float[] weightAccum = new float[targetH * targetW];

        int tileIdx = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        foreach (var (startY, sliceH) in ySlices)
        {
            foreach (var (startX, sliceW) in xSlices)
            {
                tileIdx++;
                var tileSw = System.Diagnostics.Stopwatch.StartNew();

                // Extract [16, sliceH, sliceW] sub-latent
                int subLatentSize = 16 * sliceH * sliceW;
                float[] subLatent = new float[subLatentSize];

                for (int c = 0; c < 16; c++)
                {
                    int srcChan = c * latentH * latentW;
                    int dstChan = c * sliceH * sliceW;
                    for (int y = 0; y < sliceH; y++)
                    {
                        int srcOffset = srcChan + (startY + y) * latentW + startX;
                        int dstOffset = dstChan + y * sliceW;
                        latents.Slice(srcOffset, sliceW).CopyTo(subLatent.AsSpan(dstOffset, sliceW));
                    }
                }

                int tilePixelH = sliceH * 8;
                int tilePixelW = sliceW * 8;
                byte[] tileRgb = new byte[tilePixelH * tilePixelW * 3];

                DecodeGpu(subLatent, sliceH, sliceW, tileRgb);
                tileSw.Stop();

                // Blend into image accumulator with linear feathering
                int py0 = startY * 8;
                int px0 = startX * 8;
                int blendRadius = 64; // 64 pixels (8 latent cells)

                for (int ty = 0; ty < tilePixelH; ty++)
                {
                    float wy = 1.0f;
                    if (py0 > 0 && ty < blendRadius)
                        wy = (float)(ty + 1) / blendRadius;
                    else if (py0 + tilePixelH < targetH && ty >= tilePixelH - blendRadius)
                        wy = (float)(tilePixelH - ty) / blendRadius;

                    int gy = py0 + ty;
                    int rowAccumOffset = gy * targetW;

                    for (int tx = 0; tx < tilePixelW; tx++)
                    {
                        float wx = 1.0f;
                        if (px0 > 0 && tx < blendRadius)
                            wx = (float)(tx + 1) / blendRadius;
                        else if (px0 + tilePixelW < targetW && tx >= tilePixelW - blendRadius)
                            wx = (float)(tilePixelW - tx) / blendRadius;

                        float w = wy * wx;
                        int gx = px0 + tx;
                        int pixelIdx = rowAccumOffset + gx;
                        int rgbAccumIdx = pixelIdx * 3;
                        int tileRgbIdx = (ty * tilePixelW + tx) * 3;

                        imageAccum[rgbAccumIdx + 0] += tileRgb[tileRgbIdx + 0] * w;
                        imageAccum[rgbAccumIdx + 1] += tileRgb[tileRgbIdx + 1] * w;
                        imageAccum[rgbAccumIdx + 2] += tileRgb[tileRgbIdx + 2] * w;
                        weightAccum[pixelIdx] += w;
                    }
                }

                GlacierDiagnostics.LogInformation($"   -> VAE Tile {tileIdx}/{totalTiles} ({tilePixelW}x{tilePixelH}) decoded in {tileSw.ElapsedMilliseconds} ms");
            }
        }
        sw.Stop();

        // Normalize accumulated RGB values
        for (int i = 0; i < targetH * targetW; i++)
        {
            float invW = 1.0f / MathF.Max(weightAccum[i], 1e-6f);
            int rgbIdx = i * 3;
            outputRgb[rgbIdx + 0] = (byte)Math.Clamp((int)(imageAccum[rgbIdx + 0] * invW + 0.5f), 0, 255);
            outputRgb[rgbIdx + 1] = (byte)Math.Clamp((int)(imageAccum[rgbIdx + 1] * invW + 0.5f), 0, 255);
            outputRgb[rgbIdx + 2] = (byte)Math.Clamp((int)(imageAccum[rgbIdx + 2] * invW + 0.5f), 0, 255);
        }

        GlacierDiagnostics.LogInformation($"[FLUX VAE] Tiled VAE decode complete: {totalTiles} tiles in {sw.ElapsedMilliseconds} ms");
    }

    private static List<(int Start, int Len)> ComputeTileSlices(int totalLen, int maxTileLen = 68, int minOverlap = 8)
    {
        var slices = new List<(int Start, int Len)>();
        if (totalLen <= maxTileLen)
        {
            slices.Add((0, totalLen));
            return slices;
        }

        int numTiles = (int)MathF.Ceiling((float)(totalLen - minOverlap) / (maxTileLen - minOverlap));
        numTiles = Math.Max(2, numTiles);

        float stride = (float)(totalLen - maxTileLen) / (numTiles - 1);
        for (int i = 0; i < numTiles; i++)
        {
            int start = (int)MathF.Round(i * stride);
            int len = Math.Min(maxTileLen, totalLen - start);
            slices.Add((start, len));
        }
        return slices;
    }

    private void DecodeGpu(ReadOnlySpan<float> latents, int latentH, int latentW, Span<byte> outputRgb)
    {
        int targetH = latentH * 8;
        int targetW = latentW * 8;
        int maxFloats = Math.Max(512 * 256 * 256, 128 * targetH * targetW);
        EnsureGpuBuffers(maxFloats, targetH, targetW);

        // 1. Copy latents to GPU & scale/shift
        int latentCount = 16 * latentH * latentW;
        fixed (float* pLat = latents)
        {
            _gpu!.CopyToDevice(_dLatents, (IntPtr)pLat, (nuint)(latentCount * sizeof(float)));
        }

        uint scaleBlockSize = 256;
        uint scaleGrid = (uint)((latentCount + 255) / 256);
        IntPtr dLat = _dLatents;
        void** pScaleArgs = stackalloc void*[3];
        pScaleArgs[0] = &dLat;
        pScaleArgs[1] = &dLat;
        pScaleArgs[2] = &latentCount;
        Glacier.Inference.Gpu.CuDriver.Check(Glacier.Inference.Gpu.CuDriver.LaunchKernel(
            _fnVaeScaleLatents, scaleGrid, 1, 1, scaleBlockSize, 1, 1, 0, IntPtr.Zero, (IntPtr)pScaleArgs, IntPtr.Zero),
            "LaunchKernel(vae_scale_latents)");

        int curH = latentH;
        int curW = latentW;

        // 2. decoder.conv_in (16 -> 512, 3x3)
        IntPtr dConvInW = GetGpuWeight("decoder.conv_in.weight", 512 * 16 * 9);
        IntPtr dConvInB = GetGpuWeight("decoder.conv_in.bias", 512);
        LaunchConv3x3(_dLatents, _dBufA, 16, 512, curH, curW, dConvInW, dConvInB);

        // 3. decoder.mid.block_1 (ResNet 512 -> 512)
        ResNetBlockGpu(_dBufA, _dBufB, 512, 512, curH, curW, "decoder.mid.block_1");

        // 4. decoder.mid.attn_1 (Self-Attention 512)
        SelfAttentionGpu(_dBufB, _dBufA, 512, curH, curW, "decoder.mid.attn_1");

        // 5. decoder.mid.block_2 (ResNet 512 -> 512)
        ResNetBlockGpu(_dBufA, _dBufB, 512, 512, curH, curW, "decoder.mid.block_2");

        // 6. Up-blocks: up.3, up.2, up.1, up.0
        IntPtr currentD = _dBufB;
        currentD = RunUpStageGpu(currentD, 512, 512, ref curH, ref curW, "decoder.up.3", hasUpsample: true);
        currentD = RunUpStageGpu(currentD, 512, 512, ref curH, ref curW, "decoder.up.2", hasUpsample: true);
        currentD = RunUpStageGpu(currentD, 512, 256, ref curH, ref curW, "decoder.up.1", hasUpsample: true);
        currentD = RunUpStageGpu(currentD, 256, 128, ref curH, ref curW, "decoder.up.0", hasUpsample: false);

        // 7. decoder.norm_out (128) + SiLU
        IntPtr nextD = (currentD == _dBufA) ? _dBufB : _dBufA;
        IntPtr dNormOutW = GetGpuWeight("decoder.norm_out.weight", 128);
        IntPtr dNormOutB = GetGpuWeight("decoder.norm_out.bias", 128);
        LaunchGroupNormSilu(currentD, nextD, 128, curH * curW, 32, dNormOutW, dNormOutB, applySilu: true);

        // 8. decoder.conv_out (128 -> 3, 3x3)
        IntPtr dConvOutW = GetGpuWeight("decoder.conv_out.weight", 3 * 128 * 9);
        IntPtr dConvOutB = GetGpuWeight("decoder.conv_out.bias", 3);
        IntPtr dFinalRgb = (nextD == _dBufA) ? _dBufB : _dBufA;
        LaunchConv3x3(nextD, dFinalRgb, 128, 3, curH, curW, dConvOutW, dConvOutB);

        // 9. Clamp to 24-bit sRGB bytes
        LaunchVaeClampRgb(dFinalRgb, _dRgbDevice, curH, curW);

        // 10. Copy 786 KB back to host
        _gpu!.Synchronize();
        fixed (byte* pOut = outputRgb)
        {
            _gpu.CopyToHost((IntPtr)pOut, _dRgbDevice, (nuint)(targetH * targetW * 3));
        }
    }

    private void ResNetBlockGpu(IntPtr dSrc, IntPtr dDst, int inC, int outC, int h, int w, string prefix)
    {
        int spatial = h * w;

        // 1. norm1 (GroupNorm 32, inC) + silu
        IntPtr dNorm1W = GetGpuWeight($"{prefix}.norm1.weight", inC);
        IntPtr dNorm1B = GetGpuWeight($"{prefix}.norm1.bias", inC);
        LaunchGroupNormSilu(dSrc, _dNorm1Buf, inC, spatial, 32, dNorm1W, dNorm1B, applySilu: true);

        // 2. conv1 (inC -> outC, 3x3)
        IntPtr dConv1W = GetGpuWeight($"{prefix}.conv1.weight", inC * outC * 9);
        IntPtr dConv1B = GetGpuWeight($"{prefix}.conv1.bias", outC);
        LaunchConv3x3(_dNorm1Buf, _dConv1Buf, inC, outC, h, w, dConv1W, dConv1B);

        // 3. norm2 (GroupNorm 32, outC) + silu
        IntPtr dNorm2W = GetGpuWeight($"{prefix}.norm2.weight", outC);
        IntPtr dNorm2B = GetGpuWeight($"{prefix}.norm2.bias", outC);
        LaunchGroupNormSilu(_dConv1Buf, _dNorm2Buf, outC, spatial, 32, dNorm2W, dNorm2B, applySilu: true);

        // 4. conv2 (outC -> outC, 3x3) into dDst
        IntPtr dConv2W = GetGpuWeight($"{prefix}.conv2.weight", outC * outC * 9);
        IntPtr dConv2B = GetGpuWeight($"{prefix}.conv2.bias", outC);
        LaunchConv3x3(_dNorm2Buf, dDst, outC, outC, h, w, dConv2W, dConv2B);

        // 5. Residual addition
        if (inC != outC && HasTensor($"{prefix}.nin_shortcut.weight"))
        {
            IntPtr dShortW = GetGpuWeight($"{prefix}.nin_shortcut.weight", inC * outC);
            IntPtr dShortB = GetGpuWeight($"{prefix}.nin_shortcut.bias", outC);
            LaunchConv1x1(dSrc, _dNorm1Buf, inC, outC, spatial, dShortW, dShortB);
            LaunchTensorAdd(dDst, _dNorm1Buf, outC * spatial);
        }
        else
        {
            LaunchTensorAdd(dDst, dSrc, outC * spatial);
        }
    }

    private void SelfAttentionGpu(IntPtr dSrc, IntPtr dDst, int channels, int h, int w, string prefix)
    {
        int spatial = h * w;

        // 1. Norm
        IntPtr dNormW = GetGpuWeight($"{prefix}.norm.weight", channels);
        IntPtr dNormB = GetGpuWeight($"{prefix}.norm.bias", channels);
        LaunchGroupNormSilu(dSrc, _dNorm1Buf, channels, spatial, 32, dNormW, dNormB, applySilu: false);

        // 2. Q, K, V (1x1 Conv)
        IntPtr dQW = GetGpuWeight($"{prefix}.q.weight", channels * channels);
        IntPtr dQB = GetGpuWeight($"{prefix}.q.bias", channels);
        IntPtr dKW = GetGpuWeight($"{prefix}.k.weight", channels * channels);
        IntPtr dKB = GetGpuWeight($"{prefix}.k.bias", channels);
        IntPtr dVW = GetGpuWeight($"{prefix}.v.weight", channels * channels);
        IntPtr dVB = GetGpuWeight($"{prefix}.v.bias", channels);

        IntPtr dQ = _dConv1Buf;
        IntPtr dK = _dNorm2Buf;
        IntPtr dV = _dBufA == dSrc ? _dBufB : _dBufA;

        LaunchConv1x1(_dNorm1Buf, dQ, channels, channels, spatial, dQW, dQB);
        LaunchConv1x1(_dNorm1Buf, dK, channels, channels, spatial, dKW, dKB);
        LaunchConv1x1(_dNorm1Buf, dV, channels, channels, spatial, dVW, dVB);

        // 3. Spatial Attention
        LaunchVaeSpatialAttention(dQ, dK, dV, _dNorm1Buf, spatial, channels);

        // 4. proj_out (1x1 Conv)
        IntPtr dProjW = GetGpuWeight($"{prefix}.proj_out.weight", channels * channels);
        IntPtr dProjB = GetGpuWeight($"{prefix}.proj_out.bias", channels);
        LaunchConv1x1(_dNorm1Buf, dDst, channels, channels, spatial, dProjW, dProjB);

        // 5. Residual addition
        LaunchTensorAdd(dDst, dSrc, channels * spatial);
    }

    private IntPtr RunUpStageGpu(IntPtr curD, int inC, int outC, ref int h, ref int w, string prefix, bool hasUpsample)
    {
        int curC = inC;
        IntPtr currentFeat = curD;

        for (int b = 0; b < 3; b++)
        {
            string blockPrefix = $"{prefix}.block.{b}";
            int bOutC = (b == 0) ? outC : outC;
            int bInC = (b == 0) ? curC : outC;

            IntPtr nextD = (currentFeat == _dBufA) ? _dBufB : _dBufA;
            ResNetBlockGpu(currentFeat, nextD, bInC, bOutC, h, w, blockPrefix);

            currentFeat = nextD;
            curC = bOutC;
        }

        if (hasUpsample)
        {
            int newH = h * 2;
            int newW = w * 2;
            IntPtr upsampledD = (currentFeat == _dBufA) ? _dBufB : _dBufA;

            LaunchUpsample2x(currentFeat, upsampledD, curC, h, w);

            IntPtr convOutD = (upsampledD == _dBufA) ? _dBufB : _dBufA;
            IntPtr dUpW = GetGpuWeight($"{prefix}.upsample.conv.weight", curC * curC * 9);
            IntPtr dUpB = GetGpuWeight($"{prefix}.upsample.conv.bias", curC);

            LaunchConv3x3(upsampledD, convOutD, curC, curC, newH, newW, dUpW, dUpB);

            h = newH;
            w = newW;
            return convOutD;
        }

        return currentFeat;
    }


    public void Dispose()
    {
        if (!_disposed)
        {
            if (_gpu != null)
            {
                FreeGpuBuffers();
                foreach (var kvp in _gpuWeights)
                {
                    if (kvp.Value != IntPtr.Zero) _gpu.FreeDevice(kvp.Value);
                }
                _gpuWeights.Clear();
                if (_gpuModule != IntPtr.Zero)
                {
                    Glacier.Inference.Gpu.CuDriver.ModuleUnload(_gpuModule);
                }
                _gpu.Dispose();
            }
            _weights.Dispose();
            _disposed = true;
        }
    }
}
