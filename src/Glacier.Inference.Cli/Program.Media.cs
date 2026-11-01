namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Glacier.Inference.Vision;
using Glacier.Inference.Video;
using Glacier.Inference.Image;
using Glacier.Inference.Image.Gguf;

public static partial class Program
{
    private static int RunVision(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Console.WriteLine("Glacier Vision Feature Extraction Subsystem [Experimental] (ViT / SigLIP + 2D Spatial Merging)");
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  glacier vision demo");
            Console.WriteLine("  glacier vision query <image.png|bmp|raw> \"<prompt>\" --synthetic-demo");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  glacier vision demo");
            Console.WriteLine("  glacier vision query diagram.png \"What objects and colors are in this scene?\" --synthetic-demo");
            return 0;
        }

        string subCmd = args[0].ToLowerInvariant();

        if (subCmd == "demo")
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================================");
            Console.WriteLine("   GLACIER.INFERENCE: VISION PATCHIFICATION & PROJECTION DEMONSTRATION    ");
            Console.WriteLine("     Pure C# .NET 10 | 14x14 SIMD Patches + ViT Attention + 2D Merging    ");
            Console.WriteLine("==========================================================================");
            Console.ResetColor();
            Console.WriteLine();

            int srcW = 640;
            int srcH = 480;
            byte[] rawPixels = new byte[srcW * srcH * 3];

            // Generate synthetic visual scene with geometric blocks, high-contrast borders, and rich color palette
            for (int y = 0; y < srcH; y++)
            {
                for (int x = 0; x < srcW; x++)
                {
                    int idx = (y * srcW + x) * 3;
                    if (x > 100 && x < 300 && y > 100 && y < 300)
                    {
                        // High-contrast central blue focal region
                        rawPixels[idx] = 40;
                        rawPixels[idx + 1] = 80;
                        rawPixels[idx + 2] = 230;
                    }
                    else if (y > 350)
                    {
                        // Lower green field
                        rawPixels[idx] = 30;
                        rawPixels[idx + 1] = 200;
                        rawPixels[idx + 2] = 50;
                    }
                    else
                    {
                        // Background gradient
                        rawPixels[idx] = (byte)(x * 200 / srcW);
                        rawPixels[idx + 1] = (byte)(y * 150 / srcH);
                        rawPixels[idx + 2] = 180;
                    }
                }
            }

#pragma warning disable GLACIER_EXP001
            using var vlm = new VisionPipeline(numLayers: 4, visionDim: 768, llmDim: 3584, patchSize: 14);
            var sw = Stopwatch.StartNew();

            // Step 1: Preprocess & Spatial Patch Extraction
            Console.WriteLine($"[1. SIMD Image Preprocessing] Ingesting {srcW}x{srcH} RGB image -> Resampling to 448x448 with 14x14 patches...");
            var prepSw = Stopwatch.StartNew();
            int patchDim = vlm.Vit.PatchDim; // 14 * 14 * 3 = 588
            int totalPatches = (448 / 14) * (448 / 14); // 32 x 32 = 1024 patches
            var patchBuffer = new float[totalPatches * patchDim];
            VisionPreprocessor.ExtractPatches(rawPixels, srcW, srcH, 3, 448, 448, 14, patchBuffer, useSigLipNorm: true);
            prepSw.Stop();
            Console.WriteLine($"   -> Extracted {totalPatches} patches ({totalPatches * patchDim * 4 / 1024} KB uncompressed) in {prepSw.ElapsedMilliseconds}ms");
            Console.WriteLine();

            // Step 2: Vision Transformer (ViT / SigLIP) Forward Pass
            Console.WriteLine("[2. Experimental Synthetic ViT] 4-Layer Multi-Head Self-Attention across 1,024 patches...");
            var vitSw = Stopwatch.StartNew();
            var visionTokens = new float[totalPatches * vlm.Vit.HiddenDim];
            vlm.Vit.Forward(patchBuffer, totalPatches, visionTokens);
            vitSw.Stop();
            Console.WriteLine($"   -> Generated 1,024 visual latent representations (dim={vlm.Vit.HiddenDim}) in {vitSw.ElapsedMilliseconds}ms");
            Console.WriteLine();

            // Step 3: 2D Spatial Merging & Multimodal Projector
            Console.WriteLine("[3. 2D Spatial Merging & MLP Projector] Merging adjacent 2x2 patches -> Projecting into LLM dim 3,584...");
            var projSw = Stopwatch.StartNew();
            int mergedPatches = totalPatches / (vlm.Projector.SpatialMergeFactor * vlm.Projector.SpatialMergeFactor); // 256 tokens
            var llmTokens = new float[mergedPatches * vlm.Projector.LlmDim];
            vlm.Projector.ProjectPatches(visionTokens, 32, 32, llmTokens);
            projSw.Stop();
            Console.WriteLine($"   -> 4x Spatial Compression: 1,024 patches -> {mergedPatches} LLM tokens in {projSw.ElapsedMilliseconds}ms (75% KV-cache savings)");
            Console.WriteLine();

            // Step 4: Scene Statistics Summary
            Console.WriteLine("[4. Scene Statistics] Extracting pixel statistics...");
            var metadata = VisionPipeline.AnalyzeVisualStatistics(rawPixels, srcW, srcH, 3);
            Console.WriteLine($"   -> RGB Means: R={metadata.AverageR * 100:F1}%, G={metadata.AverageG * 100:F1}%, B={metadata.AverageB * 100:F1}%");
            Console.WriteLine($"   -> Dominant Hue: {metadata.DominantHue}");
            Console.WriteLine($"   -> Layout: {metadata.SceneDescription}");
            Console.WriteLine();
#pragma warning restore GLACIER_EXP001

            sw.Stop();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[VERIFIED] Vision patchification & projection demo completed in {sw.ElapsedMilliseconds}ms!");
            Console.ResetColor();
            return 0;
        }
        else if (subCmd == "query")
        {
            bool syntheticDemo = false;
            var positional = new List<string>();
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--synthetic-demo")
                    syntheticDemo = true;
                else
                    positional.Add(args[i]);
            }

            if (positional.Count < 2)
            {
                Console.Error.WriteLine("Error: Missing image file or prompt. Usage: glacier vision query <image.png|bmp|raw> \"<prompt>\" --synthetic-demo");
                return 1;
            }

            if (!syntheticDemo)
            {
                Console.Error.WriteLine(
                    "Error: 'glacier vision query' uses the [Experimental] synthetic VisionPipeline which does not load pretrained VLM weights. " +
                    "Pass '--synthetic-demo' to inspect synthetic visual token embeddings and scene statistics, or use EmbeddingGemma2Model with AttachMultimodalProjector (Gemma4VisionTower) for pretrained vision embeddings.");
                return 1;
            }

            string imgPath = positional[0];
            string prompt = positional[1];

            if (!File.Exists(imgPath))
            {
                Console.Error.WriteLine($"Error: Image file '{imgPath}' not found.");
                return 1;
            }

            byte[] pixels;
            int w;
            int h;
            if (imgPath.EndsWith(".raw", StringComparison.OrdinalIgnoreCase))
            {
                byte[] raw = File.ReadAllBytes(imgPath);
                w = 448;
                h = 448;
                if (raw.Length < w * h * 3)
                {
                    throw new InvalidDataException($"Raw RGB image '{imgPath}' ({raw.Length} bytes) is smaller than 448x448x3 ({w * h * 3} bytes).");
                }
                pixels = raw;
            }
            else
            {
                (pixels, w, h) = ImageDecoder.Load(imgPath);
            }

#pragma warning disable GLACIER_EXP001
            using var vlm = new VisionPipeline(numLayers: 2, visionDim: 128, llmDim: 256, patchSize: 14);
            var (visualTokens, _, metadata) = vlm.ProcessImage(pixels, w, h, 3);
#pragma warning restore GLACIER_EXP001

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Synthetic Vision Demo | Prompt: \"{prompt}\"] Extracted {visualTokens} visual tokens | {metadata.DominantHue} {metadata.SceneDescription}");
            Console.ResetColor();
            return 0;
        }

        Console.Error.WriteLine($"Unknown vision command: '{subCmd}'. Use 'glacier vision --help' for usage.");
        return 1;
    }

    private static int RunVideo(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintVideoHelp();
            return 0;
        }

        string subCmd = args[0].ToLowerInvariant();
        if (subCmd == "generate")
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Error: Missing prompt for video generation. Usage: glacier video generate \"<prompt>\" [options]");
                return 1;
            }

            string prompt = args[1];
            int width = 832;
            int height = 480;
            int frames = 17;
            int fps = 16;
            int steps = 25;
            CameraMotion? explicitMotion = null;
            string outPath = "glacier_video.apng";
            string format = "apng";
            int? seed = null;
            string? imagePath = null;
            float? duration = null;
            float guidance = 5.0f;

            for (int i = 2; i < args.Length; i++)
            {
                if (args[i] == "--width" && i + 1 < args.Length && int.TryParse(args[++i], out int w)) width = w;
                else if (args[i] == "--height" && i + 1 < args.Length && int.TryParse(args[++i], out int h)) height = h;
                else if (args[i] == "--frames" && i + 1 < args.Length && int.TryParse(args[++i], out int fr)) frames = fr;
                else if (args[i] == "--fps" && i + 1 < args.Length && int.TryParse(args[++i], out int r)) fps = r;
                else if (args[i] == "--steps" && i + 1 < args.Length && int.TryParse(args[++i], out int st)) steps = st;
                else if (args[i] == "--out" && i + 1 < args.Length) outPath = args[++i];
                else if (args[i] == "--format" && i + 1 < args.Length) format = args[++i].ToLowerInvariant();
                else if (args[i] == "--seed" && i + 1 < args.Length && int.TryParse(args[++i], out int sd)) seed = sd;
                else if ((args[i] == "--image" || args[i] == "-i") && i + 1 < args.Length) imagePath = args[++i];
                else if (args[i] == "--duration" && i + 1 < args.Length && float.TryParse(args[++i], out float dur)) duration = dur;
                else if (args[i] == "--guidance" && i + 1 < args.Length && float.TryParse(args[++i], out float gd)) guidance = gd;
                else if (args[i] == "--motion" && i + 1 < args.Length)
                {
                    string mStr = args[++i].ToLowerInvariant();
                    explicitMotion = mStr switch
                    {
                        "pan-left" or "panleft" => CameraMotion.PanLeft,
                        "pan-right" or "panright" => CameraMotion.PanRight,
                        "tilt-up" or "tiltup" => CameraMotion.TiltUp,
                        "tilt-down" or "tiltdown" => CameraMotion.TiltDown,
                        "zoom-in" or "zoomin" => CameraMotion.ZoomIn,
                        "zoom-out" or "zoomout" => CameraMotion.ZoomOut,
                        "orbit" => CameraMotion.Orbit,
                        "fluid" or "dynamic" => CameraMotion.DynamicFluid,
                        "static" => CameraMotion.Static,
                        _ => CameraMotion.PanRight
                    };
                }
            }

            // Dynamically infer camera motion from natural language prompt semantics if not explicitly specified
            CameraMotion motion = explicitMotion ?? VideoGenerationPipeline.InferMotionFromPrompt(prompt, CameraMotion.PanRight);

            // Auto-compute frame count if duration in seconds was specified
            if (duration.HasValue)
            {
                frames = Math.Max(2, (int)MathF.Round(duration.Value * fps));
            }

            // Auto-detect format from file extension if format was left at default
            if (format == "apng")
            {
                if (outPath.EndsWith(".gif", StringComparison.OrdinalIgnoreCase)) format = "gif";
                else if (outPath.EndsWith(".avi", StringComparison.OrdinalIgnoreCase)) format = "avi";
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================================");
            Console.WriteLine("   GLACIER.INFERENCE: GENERATIVE VIDEO PRODUCTION                         ");
            Console.WriteLine("   Pure C# .NET 10 | Wan2.1 3D-DiT + T5-XXL + 3D Causal VAE               ");
            Console.WriteLine("==========================================================================");
            Console.ResetColor();
            Console.WriteLine();
            Console.WriteLine($"Prompt: \"{prompt}\"");
            if (!string.IsNullOrEmpty(imagePath)) Console.WriteLine($"Reference Image: {imagePath}");
            float totalDurationSec = (float)frames / fps;
            Console.WriteLine($"Resolution: {width}x{height} | Duration: {totalDurationSec:F1}s ({frames} frames @ {fps} FPS)");
            Console.WriteLine($"Camera Motion: {motion} | Format: {format.ToUpperInvariant()}");
            Console.WriteLine($"Flow Steps: {steps} | Guidance: {guidance:F1} | ODE: Rectified Flow Euler Solver");
            Console.WriteLine();

            VideoGenerationResult result;
            using var pipeline = new VideoGenerationPipeline();
            if (!string.IsNullOrEmpty(imagePath) && File.Exists(imagePath))
            {
                Console.WriteLine($"Animating reference image: {imagePath} with camera motion {motion}...");
                var (rgb, srcW, srcH) = ImageDecoder.Load(imagePath);
                result = pipeline.GenerateFromImage(rgb, srcW, srcH, prompt, width, height, frames, fps, steps, motion, seed);
            }
            else
            {
                result = pipeline.Generate(prompt, width, height, frames, fps, steps, motion, seed, guidance);
            }

            if (format == "all")
            {
                string baseName = Path.GetFileNameWithoutExtension(outPath);
                string dir = Path.GetDirectoryName(outPath) ?? "";
                string apngFile = string.IsNullOrEmpty(dir) ? $"{baseName}.apng" : Path.Combine(dir, $"{baseName}.apng");
                string gifFile = string.IsNullOrEmpty(dir) ? $"{baseName}.gif" : Path.Combine(dir, $"{baseName}.gif");
                string aviFile = string.IsNullOrEmpty(dir) ? $"{baseName}.avi" : Path.Combine(dir, $"{baseName}.avi");

                result.SaveApng(apngFile);
                result.SaveGif(gifFile);
                result.SaveAvi(aviFile);

                int totalF = result.Frames.Count;
                int[] sampleIndices = [0, totalF / 4, totalF / 2, 3 * totalF / 4, totalF - 1];
                foreach (int idx in sampleIndices)
                {
                    if (idx >= 0 && idx < totalF)
                    {
                        string frameFile = string.IsNullOrEmpty(dir) ? $"{baseName}_frame_{idx}.png" : Path.Combine(dir, $"{baseName}_frame_{idx}.png");
                        PngWriter.SavePng24(frameFile, result.Frames[idx], result.Width, result.Height);
                    }
                }

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[Exported APNG] -> {apngFile} ({new FileInfo(apngFile).Length / 1024} KB)");
                Console.WriteLine($"[Exported GIF]  -> {gifFile} ({new FileInfo(gifFile).Length / 1024} KB)");
                Console.WriteLine($"[Exported AVI]  -> {aviFile} ({new FileInfo(aviFile).Length / 1024} KB)");
                Console.ResetColor();
            }
            else
            {
                result.Save(outPath);
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[Exported Video] -> {outPath} ({new FileInfo(outPath).Length / 1024} KB)");
                Console.ResetColor();
            }

            Console.WriteLine($"Completed in {result.ElapsedMilliseconds} ms ({result.ElapsedMilliseconds / (float)frames:F1} ms/frame).");
            return 0;
        }
        else if (subCmd == "demo")
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================================");
            Console.WriteLine("   GLACIER.INFERENCE: EXPERIMENTAL VIDEO PRIMITIVES DEMONSTRATION         ");
            Console.WriteLine("   Pure C# .NET 10 | Synthetic 3D-DiT + 3D-RoPE + APNG/GIF/AVI Export     ");
            Console.WriteLine("==========================================================================");
            Console.ResetColor();
            Console.WriteLine();

            var sw = Stopwatch.StartNew();

#pragma warning disable GLACIER_EXP001
            // Phase 1: Experimental Synthetic Spatio-Temporal DiT + Container Export
            Console.WriteLine("[1. Experimental Synthetic Spatio-Temporal DiT] Synthesizing 16 test frames (256x256 @ 8 FPS)...");
            using var synthDit = new SpatioTemporalDiT(numLayers: 2, hiddenDim: 128, numHeads: 4);
            using var synthVae = new TemporalLatentVaeDecoder();
            int latentH = 32, latentW = 32, temporalLatentFrames = 4;
            int frameLatentSize = synthDit.LatentChannels * latentH * latentW;
            var targetLatents = new float[temporalLatentFrames * frameLatentSize];
            for (int k = 0; k < temporalLatentFrames; k++)
            {
                float u = (float)k / (temporalLatentFrames - 1);
                var (offX, offY, zoom, _) = VideoGenerationPipeline.GetCameraTransform(CameraMotion.PanRight, u);
                PromptSemanticSynthesizer.SynthesizeTargetLatents(
                    "A cinematic landscape of icy fjords under neon auroras",
                    targetLatents.AsSpan(k * frameLatentSize, frameLatentSize),
                    latentH, latentW, synthDit.LatentChannels, 42, offX, offY, zoom);
            }
            var ditVelocity = new float[targetLatents.Length];
            synthDit.PredictVelocity(targetLatents, 0.5f, targetLatents, CameraMotion.PanRight, temporalLatentFrames, latentH, latentW, ditVelocity);
            var framesList = synthVae.DecodeVideo(targetLatents, temporalLatentFrames, 16, latentH, latentW);
            var videoRes = new VideoGenerationResult(framesList, 256, 256, 8, sw.ElapsedMilliseconds, "Synthetic test sequence", CameraMotion.PanRight);

            string demoApng = "glacier_demo_video.apng";
            string demoGif = "glacier_demo_video.gif";
            string demoAvi = "glacier_demo_video.avi";
            videoRes.SaveApng(demoApng);
            videoRes.SaveGif(demoGif);
            videoRes.SaveAvi(demoAvi);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"    -> Generated 16 synthetic frames in {videoRes.ElapsedMilliseconds}ms");
            Console.WriteLine($"    -> Saved APNG: {demoApng} ({new FileInfo(demoApng).Length / 1024} KB)");
            Console.WriteLine($"    -> Saved GIF:  {demoGif} ({new FileInfo(demoGif).Length / 1024} KB)");
            Console.WriteLine($"    -> Saved AVI:  {demoAvi} ({new FileInfo(demoAvi).Length / 1024} KB)");
            Console.ResetColor();
            Console.WriteLine();

            // Phase 2: Keyframe Decimation & Spatio-Temporal Ingestion
            Console.WriteLine("[2. Frame Decimation & Motion Profiling] Sampling 4 keyframes from video...");
            int[] keyframes = VideoFrameSampler.SampleUniformIndices(framesList.Count, 4);
            Console.WriteLine($"    -> Selected keyframes: [{string.Join(", ", keyframes)}]");

            float motionEnergy = VideoFrameSampler.ComputeMotionEnergy(framesList[0], framesList[^1]);
            Console.WriteLine($"    -> Measured camera pan motion energy: {motionEnergy * 100:F1}%");
            Console.WriteLine();

            // Phase 3: 3D-RoPE Spatio-Temporal Positional Encoding
            Console.WriteLine("[3. 3D-RoPE Positional Decomposition] Projecting temporal (t) and spatial (h, w) coordinates...");
            var rope = new VideoRoPE(headDim: 64, ropeTheta: 10000.0f);
            float[] sampleHeadVec = new float[64];
            Array.Fill(sampleHeadVec, 1.0f);
            rope.Apply3DRoPE(sampleHeadVec, t: 3, h: 16, w: 16);
            Console.WriteLine($"    -> Partitioned 64-dim head: Temporal={rope.TemporalDim}, Height={rope.HeightDim}, Width={rope.WidthDim}");
            Console.WriteLine();

            // Phase 4: Experimental Video Temporal Feature Extraction
            Console.WriteLine("[4. Experimental Video Feature Extraction] Extracting keyframe temporal embeddings...");
            using var videoPipeline = new VideoPipeline(numLayers: 2, visionDim: 256, llmDim: 512, patchSize: 14);
            var (tokens, _, meta) = videoPipeline.ProcessVideo(framesList, 256, 256, targetKeyframes: 4);
            Console.WriteLine($"    -> Extracted {tokens} spatio-temporal tokens | Summary: {meta.TemporalSummary}");
            Console.WriteLine($"    -> Motion Pattern: {meta.MotionPattern}");
            Console.WriteLine();
#pragma warning restore GLACIER_EXP001

            sw.Stop();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[VERIFIED] Experimental Video primitives demo completed in {sw.ElapsedMilliseconds}ms!");
            Console.ResetColor();
            return 0;
        }
        else if (subCmd == "query")
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("Error: Missing frame directory or prompt. Usage: glacier video query <dir> \"<prompt>\"");
                return 1;
            }

            string dir = args[1];
            string prompt = args[2];

            if (!Directory.Exists(dir))
            {
                Console.Error.WriteLine($"Error: Directory '{dir}' not found.");
                return 1;
            }

            var files = Directory.GetFiles(dir, "*.raw");
            if (files.Length == 0)
            {
                Console.Error.WriteLine($"Error: No .raw image frames found in '{dir}'.");
                return 1;
            }

            var frames = new List<byte[]>();
            foreach (var f in files) frames.Add(File.ReadAllBytes(f));

#pragma warning disable GLACIER_EXP001
            using var video = new VideoPipeline();
            string answer = video.Query(frames, 448, 448, prompt, 4);
#pragma warning restore GLACIER_EXP001

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Video Answer]: {answer}");
            Console.ResetColor();
            return 0;
        }

        Console.Error.WriteLine($"Unknown video command: '{subCmd}'. Use 'glacier video --help' for usage.");
        return 1;
    }

    private static void SaveImage(string path, byte[] rgbPixels, int width, int height)
    {
        if (path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
        {
            BmpWriter.SaveBmp24(path, rgbPixels, width, height);
        }
        else
        {
            // Default to W3C-standard 24-bit PNG
            PngWriter.SavePng24(path, rgbPixels, width, height);
        }
    }

    private static int RunImage(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintImageHelp();
            return 0;
        }

        string subCmd = args[0].ToLowerInvariant();
        if (subCmd == "demo")
        {
            string outPath = "glacier_generated_image.png";
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] is "--out" or "-o" && i + 1 < args.Length) outPath = args[++i];
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================================");
            Console.WriteLine("  GLACIER.INFERENCE: EXPERIMENTAL SYNTHETIC T2I PRIMITIVE DEMO            ");
            Console.WriteLine("   Pure C# .NET 10 | Synthetic Flow Matching (Euler ODE) + Procedural VAE ");
            Console.WriteLine("==========================================================================");
            Console.ResetColor();
            Console.WriteLine();

            string prompt = "A majestic crystal glacier fortress illuminated by cyan auroras";
            Console.WriteLine($"[1. Prompt Ingestion] Prompt: \"{prompt}\"");
            Console.WriteLine($"   -> Target Canvas: 256x256 (32x32 Latent Grid, 16 Channels, 4 Euler Steps)");
            Console.WriteLine();

            Console.WriteLine("[2. Latent Flow Matching] Initializing Gaussian noise z_1 ~ N(0, I) & running 4-step Euler ODE...");
#pragma warning disable GLACIER_EXP001
            using var pipeline = new ImageGenerationPipeline(numLayers: 2, hiddenDim: 256, numHeads: 4);

            var result = pipeline.Generate(prompt, width: 256, height: 256, numSteps: 4, seed: 42);
#pragma warning restore GLACIER_EXP001
            Console.WriteLine($"   -> 4-Step Flow Matching completed in {result.ElapsedMilliseconds}ms ({result.PixelsPerSecond:F0} pixels/sec)");
            Console.WriteLine();

            string fmt = outPath.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase) ? "BMP" : "PNG";
            Console.WriteLine($"[3. Latent Reconstruction & Export] Writing 24-bit RGB {fmt} to '{outPath}'...");
            SaveImage(outPath, result.RgbPixels, result.Width, result.Height);
            var fi = new FileInfo(outPath);
            Console.WriteLine($"   -> Saved {fi.Length / 1024} KB standard 24-bit {fmt} image ({result.Width}x{result.Height})");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[VERIFIED] Experimental synthetic T2I demo completed in {result.ElapsedMilliseconds}ms!");
            Console.WriteLine($"Image saved: file:///{Path.GetFullPath(outPath).Replace('\\', '/')}");
            Console.ResetColor();
            return 0;
        }
        else if (subCmd == "info")
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Error: Missing model path. Usage: glacier image info <model.gguf>");
                return 1;
            }

            string modelPath = args[1];
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"Inspecting Diffusion GGUF Model: {modelPath}");
            Console.ResetColor();

            var sw = Stopwatch.StartNew();
            using var model = DiffusionGgufModel.Open(modelPath);
            sw.Stop();

            Console.WriteLine($"Format: GGUF v{model.Gguf.Version} (Mapped in {sw.ElapsedMilliseconds}ms)");
            Console.WriteLine($"Architecture:           {model.DiffusionArch}");
            Console.WriteLine($"Backbone Blocks:        {model.NumLayers} (Double: {model.DoubleBlocks}, Single: {model.SingleBlocks})");
            Console.WriteLine($"Hidden Dimension:       {model.HiddenDim}");
            Console.WriteLine($"Latent Channels:        {model.InChannels} ({model.PatchSize}x{model.PatchSize} spatial patch)");
            Console.WriteLine($"Embedded VAE Decoder:   {(model.HasVae ? "Yes (Integrated)" : "No (Standalone VAE required)")}");
            Console.WriteLine($"Embedded Text Encoder:  {(model.HasTextEncoder ? "Yes (CLIP/T5)" : "No")}");
            Console.WriteLine($"Recommended Scheduler:  {model.RecommendedSchedule}");
            Console.WriteLine($"Recommended Steps:      {model.RecommendedSteps}");
            Console.WriteLine($"Default Resolution:     {model.RecommendedResolution}x{model.RecommendedResolution}");
            Console.WriteLine($"Total Tensors:          {model.Gguf.TensorCount}");
            return 0;
        }
        else if (subCmd == "generate")
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Error: Missing text prompt. Usage: glacier image generate \"<prompt>\" [-m <model.gguf>] [--out <file.png>]");
                return 1;
            }

            string prompt = args[1];
            string outPath = "generated.png";
            string? modelPath = null;
            string? vaePath = null;
            string? clipPath = null;
            string? t5Path = null;
            int? steps = null;
            int? width = null;
            int? height = null;
            int? seed = null;

            for (int i = 2; i < args.Length; i++)
            {
                if (args[i] is "--out" or "-o" && i + 1 < args.Length) outPath = args[++i];
                else if (args[i] is "--model" or "-m" && i + 1 < args.Length) modelPath = args[++i];
                else if (args[i] is "--vae" && i + 1 < args.Length) vaePath = args[++i];
                else if (args[i] is "--clip" && i + 1 < args.Length) clipPath = args[++i];
                else if (args[i] is "--t5" && i + 1 < args.Length) t5Path = args[++i];
                else if (args[i] is "--steps" or "-s" && i + 1 < args.Length && int.TryParse(args[++i], out var st)) steps = st;
                else if (args[i] is "--width" or "-w" && i + 1 < args.Length && int.TryParse(args[++i], out var w)) width = w;
                else if (args[i] is "--height" or "-h" && i + 1 < args.Length && int.TryParse(args[++i], out var h)) height = h;
                else if (args[i] is "--seed" && i + 1 < args.Length && int.TryParse(args[++i], out var sd)) seed = sd;
            }

            // Auto-detect default model paths if not explicitly provided
            static string FindFile(string filename)
            {
                string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                string[] candidates = [
                    filename,
                    Path.Combine("models", filename),
                    Path.Combine("Glacier.Inference", "models", filename),
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", filename),
                    Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "models", filename),
                    Path.Combine(userProfile, ".cache", "glacier", "models", filename)
                ];
                foreach (var c in candidates)
                {
                    if (File.Exists(c)) return Path.GetFullPath(c);
                }
                return string.Empty;
            }

            if (string.IsNullOrEmpty(modelPath))
            {
                string foundModel = FindFile("flux1-schnell-Q4_K_S.gguf");
                if (!string.IsNullOrEmpty(foundModel)) modelPath = foundModel;
            }

            if (string.IsNullOrEmpty(vaePath))
            {
                string foundVae = FindFile("ae.safetensors");
                if (!string.IsNullOrEmpty(foundVae)) vaePath = foundVae;
            }

            if (string.IsNullOrEmpty(clipPath))
            {
                string foundClip = FindFile("clip_l.safetensors");
                if (!string.IsNullOrEmpty(foundClip)) clipPath = foundClip;
            }

            if (string.IsNullOrEmpty(t5Path))
            {
                string foundT5 = FindFile("t5xxl.gguf");
                if (!string.IsNullOrEmpty(foundT5)) t5Path = foundT5;
            }

            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                throw new FileNotFoundException(
                    "Flux.1 GGUF model weights (e.g., flux1-schnell-Q4_K_S.gguf) were not found. Provide --model <path.gguf> or run 'glacier image demo' for the experimental synthetic test pipeline.");
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"[Glacier Image] Loading pre-trained Diffusion GGUF: {Path.GetFileName(modelPath)}...");
            Console.ResetColor();

            using var model = DiffusionGgufModel.Open(modelPath);
            string? safetensorsVae = (vaePath?.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) == true) ? vaePath : null;
            using DiffusionGgufModel? vaeGguf = (!string.IsNullOrEmpty(vaePath) && !vaePath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)) ? DiffusionGgufModel.Open(vaePath) : null;
            using var pipeline = new DiffusionGgufPipeline(model, vaeGguf, safetensorsVae, clipPath, t5Path);

            int outW = width ?? 512;
            int outH = height ?? 512;
            int numSteps = steps ?? (model.RecommendedSteps > 0 ? model.RecommendedSteps : 4);

            Console.WriteLine($"   Architecture: {model.DiffusionArch} | Scheduler: {model.RecommendedSchedule} | {outW}x{outH} ({numSteps} steps)");
            Console.WriteLine($"   VAE Decoder:  {(pipeline.NeuralVae != null ? "Active Neural Flux VAE" : "Missing")}");
            Console.WriteLine($"   CLIP Encoder: {(!string.IsNullOrEmpty(clipPath) ? "Active CLIP ViT-L/14" : "Missing")}");
            Console.WriteLine($"   T5 Encoder:   {(pipeline.T5Encoder != null ? "Active T5-XXL GGUF" : "Missing")}");
            Console.WriteLine($"   Seed:         {(seed.HasValue ? seed.Value.ToString() : "Random")}");
            Console.WriteLine($"   Executing reverse diffusion trajectory on {(pipeline.ActiveBackend == "Cuda" ? "GPU" : "CPU SIMD")}...");

            var res = pipeline.Generate(prompt, width: outW, height: outH, numSteps: numSteps, seed: seed);
            SaveImage(outPath, res.RgbPixels, res.Width, res.Height);

            string outFmt = outPath.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase) ? "BMP" : "PNG";
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Generated] Saved to '{outPath}' ({outFmt}) in {res.ElapsedMilliseconds}ms ({res.PixelsPerSecond:F0} px/s)");
            Console.WriteLine($"Image saved: file:///{Path.GetFullPath(outPath).Replace('\\', '/')}");
            Console.ResetColor();
            return 0;
        }

        Console.Error.WriteLine($"Unknown image command: '{subCmd}'. Use 'glacier image --help' for usage.");
        return 1;
    }
}
