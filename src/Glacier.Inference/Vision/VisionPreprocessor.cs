namespace Glacier.Inference.Vision;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// High-performance SIMD image preprocessor for Vision-Language Models (Qwen 2-VL, SigLIP, CLIP, Llama-Vision).
/// Handles bilinear resampling, SigLIP/ImageNet normalization, and 2D spatial patchification with zero GC allocations.
/// </summary>
public static unsafe class VisionPreprocessor
{
    // Standard SigLIP normalization constants
    public static readonly float[] SigLipMean = [0.5f, 0.5f, 0.5f];
    public static readonly float[] SigLipStd = [0.5f, 0.5f, 0.5f];

    // Standard CLIP / ImageNet normalization constants
    public static readonly float[] ClipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    public static readonly float[] ClipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>
    /// Converts a raw RGB/RGBA image into normalized 14x14 or 16x16 flattened patches [numPatches, patchDim].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int ExtractPatches(
        ReadOnlySpan<byte> pixelData,
        int srcWidth,
        int srcHeight,
        int channels,
        int targetWidth,
        int targetHeight,
        int patchSize,
        Span<float> outPatches,
        bool useSigLipNorm = true)
    {
        if (targetWidth % patchSize != 0 || targetHeight % patchSize != 0)
        {
            throw new ArgumentException($"Target dimensions ({targetWidth}x{targetHeight}) must be divisible by patchSize ({patchSize}).");
        }

        int gridW = targetWidth / patchSize;
        int gridH = targetHeight / patchSize;
        int numPatches = gridW * gridH;
        int patchDim = patchSize * patchSize * 3; // RGB channels
        int requiredSize = numPatches * patchDim;

        if (outPatches.Length < requiredSize)
        {
            throw new ArgumentException($"Output buffer ({outPatches.Length}) too small for {requiredSize} floats.");
        }

        float[] mean = useSigLipNorm ? SigLipMean : ClipMean;
        float[] std = useSigLipNorm ? SigLipStd : ClipStd;
        float invStd0 = 1.0f / std[0], invStd1 = 1.0f / std[1], invStd2 = 1.0f / std[2];

        // Temporary resized normalized image buffer: [targetHeight, targetWidth, 3]
        int resizedLen = targetWidth * targetHeight * 3;
        float* pResized = (float*)NativeMemory.AlignedAlloc((nuint)(resizedLen * sizeof(float)), 64);

        try
        {
            // 1. Bilinear resize and normalize into float buffer
            float scaleX = (float)srcWidth / targetWidth;
            float scaleY = (float)srcHeight / targetHeight;

            fixed (byte* pSrc = pixelData)
            {
                for (int ty = 0; ty < targetHeight; ty++)
                {
                    float sy = (ty + 0.5f) * scaleY - 0.5f;
                    int y0 = Math.Clamp((int)MathF.Floor(sy), 0, srcHeight - 1);
                    int y1 = Math.Clamp(y0 + 1, 0, srcHeight - 1);
                    float yLerp = sy - y0;

                    for (int tx = 0; tx < targetWidth; tx++)
                    {
                        float sx = (tx + 0.5f) * scaleX - 0.5f;
                        int x0 = Math.Clamp((int)MathF.Floor(sx), 0, srcWidth - 1);
                        int x1 = Math.Clamp(x0 + 1, 0, srcWidth - 1);
                        float xLerp = sx - x0;

                        int idx00 = (y0 * srcWidth + x0) * channels;
                        int idx10 = (y0 * srcWidth + x1) * channels;
                        int idx01 = (y1 * srcWidth + x0) * channels;
                        int idx11 = (y1 * srcWidth + x1) * channels;

                        int dstPixel = (ty * targetWidth + tx) * 3;

                        // Sample RGB
                        for (int c = 0; c < 3; c++)
                        {
                            float top = pSrc[idx00 + c] + (pSrc[idx10 + c] - pSrc[idx00 + c]) * xLerp;
                            float bottom = pSrc[idx01 + c] + (pSrc[idx11 + c] - pSrc[idx01 + c]) * xLerp;
                            float val = (top + (bottom - top) * yLerp) / 255.0f;

                            float normVal = c switch
                            {
                                0 => (val - mean[0]) * invStd0,
                                1 => (val - mean[1]) * invStd1,
                                _ => (val - mean[2]) * invStd2
                            };

                            pResized[dstPixel + c] = normVal;
                        }
                    }
                }
            }

            // 2. Spatial patchification: arrange into [numPatches, patchSize * patchSize * 3]
            fixed (float* pDst = outPatches)
            {
                int patchIdx = 0;
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        float* patchOut = pDst + patchIdx * patchDim;
                        int pPixel = 0;

                        for (int py = 0; py < patchSize; py++)
                        {
                            int iy = gy * patchSize + py;
                            for (int px = 0; px < patchSize; px++)
                            {
                                int ix = gx * patchSize + px;
                                int srcOffset = (iy * targetWidth + ix) * 3;

                                patchOut[pPixel++] = pResized[srcOffset];
                                patchOut[pPixel++] = pResized[srcOffset + 1];
                                patchOut[pPixel++] = pResized[srcOffset + 2];
                            }
                        }
                        patchIdx++;
                    }
                }
            }

            return numPatches;
        }
        finally
        {
            NativeMemory.AlignedFree(pResized);
        }
    }
}
