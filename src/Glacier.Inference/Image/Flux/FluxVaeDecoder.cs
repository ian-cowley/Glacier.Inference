namespace Glacier.Inference.Image.Flux;

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
public unsafe sealed class FluxVaeDecoder : IDisposable
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
                Console.WriteLine($"[FLUX VAE] Initialized Pure CUDA Neural VAE on {_gpu.DeviceName} ({_gpu.ArchString}).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FLUX VAE] GPU init skipped ({ex.Message}), falling back to CPU SIMD.");
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
                    Console.WriteLine($"[FLUX VAE] Tiled GPU decode encountered: {ex.Message}. Falling back to single-shot GPU / CPU.");
                }
            }

            try
            {
                DecodeGpu(latents, latentH, latentW, outputRgb);
                return;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FLUX VAE] GPU decode encountered: {ex.Message}. Falling back to CPU SIMD.");
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

        Console.WriteLine($"[FLUX VAE] Tiled VAE Decoding active: {xSlices.Count}x{ySlices.Count} ({totalTiles} tiles) for {targetW}x{targetH} canvas...");

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

                Console.WriteLine($"   -> VAE Tile {tileIdx}/{totalTiles} ({tilePixelW}x{tilePixelH}) decoded in {tileSw.ElapsedMilliseconds} ms");
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

        Console.WriteLine($"[FLUX VAE] Tiled VAE decode complete: {totalTiles} tiles in {sw.ElapsedMilliseconds} ms");
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

    private void DecodeCpu(ReadOnlySpan<float> latents, int latentH, int latentW, Span<byte> outputRgb)
    {
        int targetH = latentH * 8;
        int targetW = latentW * 8;

        // Step 0: Preprocess and shift/scale latents
        int inC = 16;
        int spatialSize = latentH * latentW;
        float[] scaledLatents = new float[inC * spatialSize];
        fixed (float* pDst = scaledLatents)
        fixed (float* pSrc = latents)
        {
            for (int i = 0; i < inC * spatialSize; i++)
            {
                pDst[i] = (pSrc[i] / ScaleFactor) + ShiftFactor;
            }
        }

        // Feature map buffers (ping-pong allocation)
        int curH = latentH;
        int curW = latentW;

        fixed (float* pLat = scaledLatents)
        {
            // 1. decoder.conv_in (16 -> 512 channels, 3x3)
            float[] bufA = new float[512 * curH * curW];
            fixed (float* pA = bufA)
            {
                Conv3x3(pLat, pA, 16, 512, curH, curW, GetWeight("decoder.conv_in.weight"), GetWeight("decoder.conv_in.bias"));

                // 2. decoder.mid.block_1 (ResNet 512 -> 512)
                float[] bufB = new float[512 * curH * curW];
                fixed (float* pB = bufB)
                {
                    ResNetBlock(pA, pB, 512, 512, curH, curW, "decoder.mid.block_1");

                    // 3. decoder.mid.attn_1 (Self-Attention 512)
                    SelfAttention(pB, pA, 512, curH, curW, "decoder.mid.attn_1");

                    // 4. decoder.mid.block_2 (ResNet 512 -> 512)
                    ResNetBlock(pA, pB, 512, 512, curH, curW, "decoder.mid.block_2");

                    // Copy active output to pA for upsampling pipeline
                    Buffer.BlockCopy(bufB, 0, bufA, 0, 512 * curH * curW * sizeof(float));
                }
            }

            // 5. Up-blocks: up.3, up.2, up.1, up.0
            float[] currentFeat = bufA;

            // --- Level 3: 512 channels -> 512 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 512, ref curH, ref curW, "decoder.up.3", hasUpsample: true);

            // --- Level 2: 512 channels -> 512 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 512, ref curH, ref curW, "decoder.up.2", hasUpsample: true);

            // --- Level 1: 512 channels -> 256 channels + 2x Upsample ---
            currentFeat = RunUpStage(currentFeat, 512, 256, ref curH, ref curW, "decoder.up.1", hasUpsample: true);

            // --- Level 0: 256 channels -> 128 channels (no upsample, already at 8x target resolution) ---
            currentFeat = RunUpStage(currentFeat, 256, 128, ref curH, ref curW, "decoder.up.0", hasUpsample: false);

            // 6. decoder.norm_out (GroupNorm 32, 128) + SiLU
            float[] finalNorm = new float[128 * curH * curW];
            fixed (float* pIn = currentFeat)
            fixed (float* pNorm = finalNorm)
            {
                GroupNorm(pIn, pNorm, 128, curH, curW, 32, GetWeight("decoder.norm_out.weight"), GetWeight("decoder.norm_out.bias"));
                SiLU(pNorm, 128 * curH * curW);

                // 7. decoder.conv_out (128 -> 3 channels, 3x3)
                float[] finalRgbFloats = new float[3 * curH * curW];
                fixed (float* pRgbF = finalRgbFloats)
                fixed (byte* pOut = outputRgb)
                {
                    Conv3x3(pNorm, pRgbF, 128, 3, curH, curW, GetWeight("decoder.conv_out.weight"), GetWeight("decoder.conv_out.bias"));

                    // 8. Convert [-1, 1] floats to standard 24-bit sRGB bytes
                    int planeSize = curH * curW;
                    nint rgbPtr = (nint)pRgbF;
                    nint outPtr = (nint)pOut;

                    Parallel.For(0, curH, y =>
                    {
                        float* pRgbLocal = (float*)rgbPtr;
                        byte* pOutLocal = (byte*)outPtr;

                        for (int x = 0; x < curW; x++)
                        {
                            int srcIdx = y * curW + x;
                            int dstIdx = (y * curW + x) * 3;

                            float r = pRgbLocal[0 * planeSize + srcIdx];
                            float g = pRgbLocal[1 * planeSize + srcIdx];
                            float b = pRgbLocal[2 * planeSize + srcIdx];

                            pOutLocal[dstIdx]     = (byte)Math.Clamp((int)((r * 0.5f + 0.5f) * 255.0f), 0, 255);
                            pOutLocal[dstIdx + 1] = (byte)Math.Clamp((int)((g * 0.5f + 0.5f) * 255.0f), 0, 255);
                            pOutLocal[dstIdx + 2] = (byte)Math.Clamp((int)((b * 0.5f + 0.5f) * 255.0f), 0, 255);
                        }
                    });
                }
            }
        }
    }

    private float[] RunUpStage(float[] input, int inC, int outC, ref int h, ref int w, string prefix, bool hasUpsample)
    {
        int curC = inC;
        float[] cur = input;

        // 3 ResNet blocks per up stage
        for (int b = 0; b < 3; b++)
        {
            string blockPrefix = $"{prefix}.block.{b}";
            int bOutC = (b == 0) ? outC : outC;
            int bInC = (b == 0) ? curC : outC;

            float[] next = new float[bOutC * h * w];
            fixed (float* pSrc = cur)
            fixed (float* pDst = next)
            {
                ResNetBlock(pSrc, pDst, bInC, bOutC, h, w, blockPrefix);
            }
            cur = next;
            curC = bOutC;
        }

        if (hasUpsample)
        {
            // 2x Nearest Neighbor Spatial Upsample
            int newH = h * 2;
            int newW = w * 2;
            float[] upsampled = new float[curC * newH * newW];
            fixed (float* pSrc = cur)
            fixed (float* pDst = upsampled)
            {
                Upsample2x(pSrc, pDst, curC, h, w);
            }

            // Up-stage convolution (curC -> curC, 3x3)
            float[] convOut = new float[curC * newH * newW];
            fixed (float* pUp = upsampled)
            fixed (float* pOut = convOut)
            {
                Conv3x3(pUp, pOut, curC, curC, newH, newW, GetWeight($"{prefix}.upsample.conv.weight"), GetWeight($"{prefix}.upsample.conv.bias"));
            }

            h = newH;
            w = newW;
            return convOut;
        }

        return cur;
    }

    private void ResNetBlock(float* src, float* dst, int inC, int outC, int h, int w, string prefix)
    {
        int spatial = h * w;
        float[] norm1Buf = new float[inC * spatial];
        float[] conv1Buf = new float[outC * spatial];
        float[] norm2Buf = new float[outC * spatial];

        fixed (float* pNorm1 = norm1Buf)
        fixed (float* pConv1 = conv1Buf)
        fixed (float* pNorm2 = norm2Buf)
        {
            // 1. norm1 (GroupNorm 32, inC)
            GroupNorm(src, pNorm1, inC, h, w, 32, GetWeight($"{prefix}.norm1.weight"), GetWeight($"{prefix}.norm1.bias"));

            // 2. silu
            SiLU(pNorm1, inC * spatial);

            // 3. conv1 (inC -> outC, 3x3)
            Conv3x3(pNorm1, pConv1, inC, outC, h, w, GetWeight($"{prefix}.conv1.weight"), GetWeight($"{prefix}.conv1.bias"));

            // 4. norm2 (GroupNorm 32, outC)
            GroupNorm(pConv1, pNorm2, outC, h, w, 32, GetWeight($"{prefix}.norm2.weight"), GetWeight($"{prefix}.norm2.bias"));

            // 5. silu
            SiLU(pNorm2, outC * spatial);

            // 6. conv2 (outC -> outC, 3x3) into dst
            Conv3x3(pNorm2, dst, outC, outC, h, w, GetWeight($"{prefix}.conv2.weight"), GetWeight($"{prefix}.conv2.bias"));

            // 7. Residual addition: dst += (nin_shortcut(src) or src)
            if (inC != outC && HasTensor($"{prefix}.nin_shortcut.weight"))
            {
                float[] shortcut = new float[outC * spatial];
                fixed (float* pShort = shortcut)
                {
                    Conv1x1(src, pShort, inC, outC, h, w, GetWeight($"{prefix}.nin_shortcut.weight"), GetWeight($"{prefix}.nin_shortcut.bias"));
                    for (int i = 0; i < outC * spatial; i++)
                    {
                        dst[i] += pShort[i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < outC * spatial; i++)
                {
                    dst[i] += src[i];
                }
            }
        }
    }

    private void SelfAttention(float* src, float* dst, int channels, int h, int w, string prefix)
    {
        int spatial = h * w;
        float[] normBuf = new float[channels * spatial];

        fixed (float* pNorm = normBuf)
        {
            // 1. Norm
            GroupNorm(src, pNorm, channels, h, w, 32, GetWeight($"{prefix}.norm.weight"), GetWeight($"{prefix}.norm.bias"));

            // 2. Q, K, V (1x1 Conv)
            float[] qBuf = new float[channels * spatial];
            float[] kBuf = new float[channels * spatial];
            float[] vBuf = new float[channels * spatial];

            fixed (float* pQ = qBuf)
            fixed (float* pK = kBuf)
            fixed (float* pV = vBuf)
            {
                Conv1x1(pNorm, pQ, channels, channels, h, w, GetWeight($"{prefix}.q.weight"), GetWeight($"{prefix}.q.bias"));
                Conv1x1(pNorm, pK, channels, channels, h, w, GetWeight($"{prefix}.k.weight"), GetWeight($"{prefix}.k.bias"));
                Conv1x1(pNorm, pV, channels, channels, h, w, GetWeight($"{prefix}.v.weight"), GetWeight($"{prefix}.v.bias"));

                // Scaled Dot-Product Attention: [spatial, spatial]
                float scale = 1.0f / MathF.Sqrt(channels);
                float[] attnOut = new float[channels * spatial];

                fixed (float* pAttn = attnOut)
                {
                    nint qPtr = (nint)pQ;
                    nint kPtr = (nint)pK;
                    nint vPtr = (nint)pV;
                    nint attnPtr = (nint)pAttn;

                    // Compute attention over spatial tokens
                    Parallel.For(0, spatial, i =>
                    {
                        float* pQLocal = (float*)qPtr;
                        float* pKLocal = (float*)kPtr;
                        float* pVLocal = (float*)vPtr;
                        float* pAttnLocal = (float*)attnPtr;

                        float[] scores = new float[spatial];
                        float maxScore = -float.MaxValue;

                        for (int j = 0; j < spatial; j++)
                        {
                            float dot = 0.0f;
                            for (int c = 0; c < channels; c++)
                            {
                                dot += pQLocal[c * spatial + i] * pKLocal[c * spatial + j];
                            }
                            dot *= scale;
                            scores[j] = dot;
                            if (dot > maxScore) maxScore = dot;
                        }

                        // Softmax
                        float sumExp = 0.0f;
                        for (int j = 0; j < spatial; j++)
                        {
                            float exp = MathF.Exp(scores[j] - maxScore);
                            scores[j] = exp;
                            sumExp += exp;
                        }
                        float invSum = 1.0f / sumExp;

                        // Weighted V sum
                        for (int c = 0; c < channels; c++)
                        {
                            float val = 0.0f;
                            for (int j = 0; j < spatial; j++)
                            {
                                val += scores[j] * invSum * pVLocal[c * spatial + j];
                            }
                            pAttnLocal[c * spatial + i] = val;
                        }
                    });

                    // 3. proj_out (1x1 Conv)
                    float[] projBuf = new float[channels * spatial];
                    fixed (float* pProj = projBuf)
                    {
                        Conv1x1(pAttn, pProj, channels, channels, h, w, GetWeight($"{prefix}.proj_out.weight"), GetWeight($"{prefix}.proj_out.bias"));

                        // Residual add: dst = src + proj
                        for (int i = 0; i < channels * spatial; i++)
                        {
                            dst[i] = src[i] + pProj[i];
                        }
                    }
                }
            }
        }
    }

    private static void GroupNorm(float* src, float* dst, int channels, int h, int w, int groups, float* weight, float* bias)
    {
        int channelsPerGroup = channels / groups;
        int spatial = h * w;
        int groupSpatial = channelsPerGroup * spatial;

        Parallel.For(0, groups, g =>
        {
            float sum = 0.0f;
            float sumSq = 0.0f;

            for (int c = 0; c < channelsPerGroup; c++)
            {
                int fullC = g * channelsPerGroup + c;
                float* pC = src + fullC * spatial;
                for (int i = 0; i < spatial; i++)
                {
                    float v = pC[i];
                    sum += v;
                    sumSq += v * v;
                }
            }

            float mean = sum / groupSpatial;
            float variance = MathF.Max(0.0f, (sumSq / groupSpatial) - (mean * mean));
            float invStd = 1.0f / MathF.Sqrt(variance + 1e-6f);

            for (int c = 0; c < channelsPerGroup; c++)
            {
                int fullC = g * channelsPerGroup + c;
                float gamma = weight[fullC];
                float beta = bias[fullC];
                float* pSrcC = src + fullC * spatial;
                float* pDstC = dst + fullC * spatial;

                for (int i = 0; i < spatial; i++)
                {
                    pDstC[i] = (pSrcC[i] - mean) * invStd * gamma + beta;
                }
            }
        });
    }

    private static void SiLU(float* data, int length)
    {
        Parallel.For(0, length / 1024 + 1, chunk =>
        {
            int start = chunk * 1024;
            int end = Math.Min(start + 1024, length);
            for (int i = start; i < end; i++)
            {
                float x = data[i];
                data[i] = x / (1.0f + MathF.Exp(-x));
            }
        });
    }

    private static void Upsample2x(float* src, float* dst, int channels, int inH, int inW)
    {
        int outH = inH * 2;
        int outW = inW * 2;

        Parallel.For(0, channels, c =>
        {
            float* pSrc = src + c * (inH * inW);
            float* pDst = dst + c * (outH * outW);

            for (int y = 0; y < outH; y++)
            {
                int inY = y / 2;
                for (int x = 0; x < outW; x++)
                {
                    int inX = x / 2;
                    pDst[y * outW + x] = pSrc[inY * inW + inX];
                }
            }
        });
    }

    private static void Conv1x1(float* src, float* dst, int inC, int outC, int h, int w, float* weight, float* bias)
    {
        int spatial = h * w;

        Parallel.For(0, outC, oc =>
        {
            float b = bias[oc];
            float* pDstOc = dst + oc * spatial;

            // Initialize with bias
            if (Vector256.IsHardwareAccelerated)
            {
                var vb = Vector256.Create(b);
                int i = 0;
                for (; i <= spatial - 8; i += 8) vb.Store(pDstOc + i);
                for (; i < spatial; i++) pDstOc[i] = b;
            }
            else
            {
                for (int i = 0; i < spatial; i++) pDstOc[i] = b;
            }

            for (int ic = 0; ic < inC; ic++)
            {
                float wVal = weight[oc * inC + ic];
                float* pSrcIc = src + ic * spatial;

                if (Vector256.IsHardwareAccelerated)
                {
                    var vw = Vector256.Create(wVal);
                    int i = 0;
                    for (; i <= spatial - 16; i += 16)
                    {
                        var d0 = Vector256.Load(pDstOc + i) + vw * Vector256.Load(pSrcIc + i);
                        var d1 = Vector256.Load(pDstOc + i + 8) + vw * Vector256.Load(pSrcIc + i + 8);
                        d0.Store(pDstOc + i);
                        d1.Store(pDstOc + i + 8);
                    }
                    for (; i < spatial; i++)
                    {
                        pDstOc[i] += pSrcIc[i] * wVal;
                    }
                }
                else
                {
                    for (int i = 0; i < spatial; i++)
                    {
                        pDstOc[i] += pSrcIc[i] * wVal;
                    }
                }
            }
        });
    }

    private static void Conv3x3(float* src, float* dst, int inC, int outC, int h, int w, float* weight, float* bias)
    {
        int spatial = h * w;

        Parallel.For(0, outC * h, idx =>
        {
            int oc = idx / h;
            int y = idx % h;

            float b = bias[oc];
            float* pDstRow = dst + oc * spatial + y * w;

            // Initialize row with bias
            if (Vector256.IsHardwareAccelerated)
            {
                var vb = Vector256.Create(b);
                int x = 0;
                for (; x <= w - 8; x += 8) vb.Store(pDstRow + x);
                for (; x < w; x++) pDstRow[x] = b;
            }
            else
            {
                for (int x = 0; x < w; x++) pDstRow[x] = b;
            }

            int y0 = y - 1;
            int y1 = y;
            int y2 = y + 1;

            float* wOc = weight + oc * (inC * 9);

            for (int ic = 0; ic < inC; ic++)
            {
                float* wIc = wOc + ic * 9;
                float* pSrcIc = src + ic * spatial;

                float* r0 = y0 >= 0 ? (pSrcIc + y0 * w) : null;
                float* r1 = pSrcIc + y1 * w;
                float* r2 = y2 < h ? (pSrcIc + y2 * w) : null;

                float w0 = wIc[0], w1 = wIc[1], w2 = wIc[2];
                float w3 = wIc[3], w4 = wIc[4], w5 = wIc[5];
                float w6 = wIc[6], w7 = wIc[7], w8 = wIc[8];

                // Left border x = 0
                {
                    float sum = 0.0f;
                    if (r0 != null) sum += r0[0] * w1 + (w > 1 ? r0[1] * w2 : 0f);
                    sum += r1[0] * w4 + (w > 1 ? r1[1] * w5 : 0f);
                    if (r2 != null) sum += r2[0] * w7 + (w > 1 ? r2[1] * w8 : 0f);
                    pDstRow[0] += sum;
                }

                // Middle vector loop x in [1, w - 8]
                int xMid = 1;
                if (Vector256.IsHardwareAccelerated)
                {
                    var vw0 = Vector256.Create(w0);
                    var vw1 = Vector256.Create(w1);
                    var vw2 = Vector256.Create(w2);
                    var vw3 = Vector256.Create(w3);
                    var vw4 = Vector256.Create(w4);
                    var vw5 = Vector256.Create(w5);
                    var vw6 = Vector256.Create(w6);
                    var vw7 = Vector256.Create(w7);
                    var vw8 = Vector256.Create(w8);

                    int vecEnd = w - 9;
                    for (; xMid <= vecEnd; xMid += 8)
                    {
                        var acc = Vector256.Load(pDstRow + xMid);

                        if (r0 != null)
                        {
                            acc += Vector256.Load(r0 + xMid - 1) * vw0
                                 + Vector256.Load(r0 + xMid) * vw1
                                 + Vector256.Load(r0 + xMid + 1) * vw2;
                        }

                        acc += Vector256.Load(r1 + xMid - 1) * vw3
                             + Vector256.Load(r1 + xMid) * vw4
                             + Vector256.Load(r1 + xMid + 1) * vw5;

                        if (r2 != null)
                        {
                            acc += Vector256.Load(r2 + xMid - 1) * vw6
                                 + Vector256.Load(r2 + xMid) * vw7
                                 + Vector256.Load(r2 + xMid + 1) * vw8;
                        }

                        acc.Store(pDstRow + xMid);
                    }
                }

                // Remaining pixels up to right border
                for (; xMid < w; xMid++)
                {
                    int x0 = xMid - 1;
                    int x1 = xMid;
                    int x2 = xMid + 1;

                    float sum = 0.0f;
                    if (r0 != null)
                    {
                        if (x0 >= 0) sum += r0[x0] * w0;
                        sum += r0[x1] * w1;
                        if (x2 < w) sum += r0[x2] * w2;
                    }

                    if (x0 >= 0) sum += r1[x0] * w3;
                    sum += r1[x1] * w4;
                    if (x2 < w) sum += r1[x2] * w5;

                    if (r2 != null)
                    {
                        if (x0 >= 0) sum += r2[x0] * w6;
                        sum += r2[x1] * w7;
                        if (x2 < w) sum += r2[x2] * w8;
                    }

                    pDstRow[xMid] += sum;
                }
            }
        });
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
