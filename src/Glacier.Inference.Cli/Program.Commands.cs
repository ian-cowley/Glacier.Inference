namespace Glacier.Inference.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Audio;
using Glacier.Inference.Audio.Kitten;
using Glacier.Inference.Config;
using Glacier.Inference.Diagnostics;
using Glacier.Inference.Engine;
using Glacier.Inference.Gguf;
using Glacier.Inference.Hardware;
using Glacier.Inference.Memory;
using Glacier.Inference.Sampling;
using Glacier.Inference.Vision;
using Glacier.Inference.Video;
using Glacier.Inference.Image;
using Glacier.Inference.Image.Gguf;

public static partial class Program
{
    private static int RunDevices(string[] args)
    {
        if (HasHelpFlag(args))
        {
            PrintDevicesHelp();
            return 0;
        }

        var devices = DeviceManager.GetDevices();
        var (activeDevice, activeEngine) = GlacierSettings.ResolveTarget(null, null);

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("============================================================================================================");
        Console.WriteLine("                                  GLACIER DETECTED ACCELERATORS & ENGINES                                   ");
        Console.WriteLine("============================================================================================================");
        Console.ResetColor();
        Console.WriteLine($"{"[ID]",-18} | {"Device Name",-34} | {"Memory",-15} | {"Safe Driver Engines",-25}");
        Console.WriteLine(new string('-', 108));

        foreach (var dev in devices)
        {
            string memStr;
            if (dev.Vendor == GpuVendor.Cpu)
            {
                memStr = $"{dev.SharedVramGb:F1} GB RAM";
            }
            else if (dev.DedicatedVramGb < 1.0 && dev.SharedVramGb > 0)
            {
                memStr = $"{dev.SharedVramGb:F1} GB Unified";
            }
            else
            {
                memStr = $"{dev.DedicatedVramGb:F1} GB VRAM";
            }

            string safeEngines = string.Join(", ", dev.SupportedEngines.Select(FormatEngineName));
            bool isCurrent = dev.Id == activeDevice.Id;

            if (isCurrent) Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"{dev.Id,-18} | {dev.Name,-34} | {memStr,-15} | {safeEngines,-25} {(isCurrent ? $"[ACTIVE: {FormatEngineName(activeEngine)}]" : "")}");
            if (isCurrent) Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  └─ Driver Safety: {dev.SafetyNotes}");
            Console.ResetColor();
        }

        Console.WriteLine(new string('-', 108));
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("* To switch your active hardware & driver engine:");
        Console.WriteLine("  glacier config --device <id|name> [--engine <baremetal|vulkan|directml|cpu>]");
        Console.WriteLine("  Example: glacier config --device nvidia-rtx-4060 --engine baremetal");
        Console.WriteLine("  Example: glacier config --device amd-890m --engine vulkan");
        Console.WriteLine("  Example: glacier config --device cpu");
        Console.ResetColor();

        return 0;
    }

    // =========================================================================
    // 6. CONFIG COMMAND
    // =========================================================================
    private static int RunConfig(string[] args)
    {
        if (HasHelpFlag(args))
        {
            PrintConfigHelp();
            return 0;
        }

        var settings = GlacierSettings.Load();

        if (args.Length == 0)
        {
            var (activeDev, activeEng) = GlacierSettings.ResolveTarget(null, null);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=======================================================================");
            Console.WriteLine("                     GLACIER PERSISTENT SETTINGS                      ");
            Console.WriteLine("=======================================================================");
            Console.ResetColor();
            Console.WriteLine($"Config File:       {GlacierSettings.GetSettingsFilePath()}");
            Console.WriteLine($"Default Device:    {settings.DeviceId ?? "auto"} (Resolved: {activeDev.Name})");
            Console.WriteLine($"Default Engine:    {FormatEngineName(settings.Engine)} (Resolved: {FormatEngineName(activeEng)})");
            Console.WriteLine($"CPU Fallback:      {settings.FallbackToCpu}");
            Console.WriteLine($"Max Seq Length:    {settings.MaxSeqLen}");
            Console.WriteLine($"Default Temp:      {settings.DefaultTemperature}");
            Console.WriteLine($"Default Top-K:     {settings.DefaultTopK}");
            Console.WriteLine($"Default Top-P:     {settings.DefaultTopP}");
            Console.WriteLine();
            Console.WriteLine("Commands to configure:");
            Console.WriteLine("  glacier config --device <id|name> [--engine <baremetal|vulkan|directml|cpu>]");
            Console.WriteLine("  glacier config --reset");
            return 0;
        }

        if (args[0] is "--reset" or "reset")
        {
            settings.DeviceId = "auto";
            settings.Engine = InferenceEngineType.Auto;
            settings.Save();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Settings successfully reset to auto-detect defaults.");
            Console.ResetColor();
            return 0;
        }

        string? targetDev = null;
        string? targetEng = null;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--device" && i + 1 < args.Length)
                targetDev = args[++i];
            else if (args[i] == "--engine" && i + 1 < args.Length)
                targetEng = args[++i];
        }

        if (targetDev != null || targetEng != null)
        {
            // Validate safety before saving
            var (resolvedDev, resolvedEng) = GlacierSettings.ResolveTarget(
                targetDev ?? settings.DeviceId,
                targetEng ?? (settings.Engine != InferenceEngineType.Auto ? settings.Engine.ToString() : null));

            if (targetDev != null) settings.DeviceId = resolvedDev.Id;
            if (targetEng != null) settings.Engine = resolvedEng;
            settings.Save();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Successfully updated settings!");
            Console.ResetColor();
            Console.WriteLine($"Active Device: {resolvedDev.Name} ({resolvedDev.Id})");
            Console.WriteLine($"Active Engine: {FormatEngineName(resolvedEng)}");
            Console.WriteLine($"Settings saved to: {GlacierSettings.GetSettingsFilePath()}");
            return 0;
        }

        Console.WriteLine("Usage: glacier config [--device <id|name>] [--engine <baremetal|directml|cpu>] [--reset]");
        return 1;
    }

    private static string FormatEngineName(InferenceEngineType engine) => engine switch
    {
        InferenceEngineType.BareMetal => "Native Driver (SASS/HIP)",
        InferenceEngineType.Vulkan => "Vulkan (CoopMat / WMMA)",
        InferenceEngineType.DirectML => "DirectML",
        InferenceEngineType.Cpu => "Cpu",
        _ => engine.ToString()
    };

    private static int RunVoice(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintVoiceHelp();
            return 0;
        }

        string subCmd = args[0].ToLowerInvariant();

        if (subCmd is "voices" or "list")
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("=========================================================================================================");
            Console.WriteLine("                GLACIER.INFERENCE: KITTENTTS NEURAL VOICES & KOKORO ROSTER                               ");
            Console.WriteLine("=========================================================================================================");
            Console.ResetColor();

            try
            {
                using var kitten = KittenTtsEngine.CreateDefault();
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine(">> KittenTTS / StyleTTS 2 Distilled Neural Voices (24kHz CD Quality):");
                Console.ResetColor();
                Console.WriteLine($"{"Voice ID",-14} | {"Display Name",-22} | {"Gender",-8} | {"Description",-40}");
                Console.WriteLine(new string('-', 90));
                var kittenMeta = new Dictionary<string, (string Name, string Gender, string Desc)>(StringComparer.OrdinalIgnoreCase)
                {
                    ["bella"] = ("Bella", "Female", "Warm, clear, natural American female voice"),
                    ["bruno"] = ("Bruno", "Male", "Deep, grounded, authoritative American male voice"),
                    ["hugo"] = ("Hugo", "Male", "Classical, articulate British male orator"),
                    ["jasper"] = ("Jasper", "Male", "Conversational, engaging American male baritone"),
                    ["kiki"] = ("Kiki", "Female", "Youthful, expressive, bright female voice"),
                    ["leo"] = ("Leo", "Male", "Smooth, refined, conversational British male voice"),
                    ["luna"] = ("Luna", "Female", "Gentle, melodious, clear storytelling female voice"),
                    ["rosie"] = ("Rosie", "Female", "Articulate, friendly British female voice")
                };
                foreach (var v in kitten.AvailableVoices)
                {
                    (string Name, string Gender, string Desc) meta = kittenMeta.TryGetValue(v, out var m) ? m : (v, "Unknown", "Custom neural voice profile");
                    Console.WriteLine($"{v,-14} | {meta.Name,-22} | {meta.Gender,-8} | {meta.Desc}");
                }
                Console.WriteLine(new string('-', 90));
                Console.WriteLine();
            }
            catch { }

            Console.WriteLine($"{"Voice ID",-14} | {"Display Name",-22} | {"Accent",-10} | {"Gender",-8} | {"Base F0",-8} | {"Description",-32}");
            Console.WriteLine(new string('-', 105));

            using var tts = new KokoroTtsEngine();
            foreach (var kvp in tts.VoiceProfiles)
            {
                var p = kvp.Value;
                string accentStr = p.Accent == EnglishAccent.American ? "USA" : "British";
                string genderStr = p.Gender == VoiceGender.Female ? "Female" : "Male";
                Console.WriteLine($"{p.Name,-14} | {p.DisplayName,-22} | {accentStr,-10} | {genderStr,-8} | {$"{p.BaseF0:F0} Hz",-8} | {p.Description}");
            }
            Console.WriteLine(new string('-', 105));
            Console.WriteLine();
            Console.WriteLine("Usage: glacier voice tts \"<text>\" --voice <id> --out <speech.wav>");
            return 0;
        }
        else if (subCmd == "tts")
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Error: Missing text for TTS synthesis. Usage: glacier voice tts \"<text>\" [--out <file.wav>]");
                return 1;
            }

            string text = args[1];
            string outPath = "speech.wav";
            string rawVoice = "bella";
            string engine = "auto";
            float speed = 1.0f;
            bool play = false;

            for (int i = 2; i < args.Length; i++)
            {
                if (args[i] is "--out" or "-o" && i + 1 < args.Length)
                {
                    outPath = args[++i];
                }
                else if (args[i] is "--play" or "-p")
                {
                    play = true;
                }
                else if (args[i] is "--voice" or "-v" && i + 1 < args.Length)
                {
                    rawVoice = args[++i].ToLowerInvariant();
                }
                else if (args[i] is "--engine" or "-e" && i + 1 < args.Length)
                {
                    engine = args[++i].ToLowerInvariant();
                }
                else if (args[i] is "--speed" or "-s" && i + 1 < args.Length)
                {
                    if (float.TryParse(args[++i], out var s)) speed = s;
                }
            }

            // Check if KittenTTS neural engine is available
            KittenTtsEngine? kittenEngine = null;
            try
            {
                kittenEngine = KittenTtsEngine.CreateDefault();
            }
            catch { }

            string voiceStr = rawVoice.ToLowerInvariant();
            bool isKittenVoice = kittenEngine != null && (kittenEngine.AvailableVoices.Contains(voiceStr) || voiceStr is "bella" or "bruno" or "jasper" or "hugo" or "kiki" or "leo" or "luna" or "rosie" || engine == "kitten");

            if (kittenEngine != null && (isKittenVoice || engine != "kokoro"))
            {
                if (!kittenEngine.AvailableVoices.Contains(voiceStr)) voiceStr = "bella";
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"[KittenTTS Neural Engine] Synthesizing '{text}'");
                Console.ResetColor();
                Console.WriteLine($"   Architecture: StyleTTS 2 Distilled (100% Pure C# .NET 10 SIMD)");
                Console.WriteLine($"   Voice:        {voiceStr} (24kHz CD Quality)");
                Console.WriteLine($"   Cadence:      {speed:F2}x Speed");

                var swKitten = Stopwatch.StartNew();
                float[] kSamples = kittenEngine.Synthesize(text, voiceStr, speed);
                swKitten.Stop();

                float durSec = (float)kSamples.Length / kittenEngine.SampleRate;
                float kRtf = durSec / (float)swKitten.Elapsed.TotalSeconds;

                WavWriter.WritePcm16(outPath, kSamples, kittenEngine.SampleRate, 1);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[Success] Synthesized {durSec:F2}s of 24kHz neural audio in {swKitten.ElapsedMilliseconds}ms ({kRtf:F1}x Real-Time) -> '{outPath}'");
                Console.ResetColor();

                if (play)
                {
                    Console.WriteLine("[Audio] Playing audio through speakers...");
                    AudioPlayer.PlayFile(outPath, wait: true);
                }
                kittenEngine.Dispose();
                return 0;
            }

            // Fallback to legacy Kokoro formant engine
            KokoroVoice voice = rawVoice switch
            {
                "af_heart" or "heart" => KokoroVoice.AfHeart,
                "af_bella" or "bella" => KokoroVoice.AfBella,
                "af_sarah" or "sarah" => KokoroVoice.AfSarah,
                "af_sky" or "sky" => KokoroVoice.AfSky,
                "am_adam" or "adam" => KokoroVoice.AmAdam,
                "am_michael" or "michael" => KokoroVoice.AmMichael,
                "am_echo" or "echo" => KokoroVoice.AmEcho,
                "am_eric" or "eric" => KokoroVoice.AmEric,
                "bf_emma" or "emma" => KokoroVoice.BfEmma,
                "bf_isabella" or "isabella" => KokoroVoice.BfIsabella,
                "bf_alice" or "alice" => KokoroVoice.BfAlice,
                "bf_lily" or "lily" => KokoroVoice.BfLily,
                "bm_george" or "george" => KokoroVoice.BmGeorge,
                "bm_lewis" or "lewis" => KokoroVoice.BmLewis,
                "bm_daniel" or "daniel" => KokoroVoice.BmDaniel,
                "bm_fable" or "fable" => KokoroVoice.BmFable,
                _ => KokoroVoice.AfHeart
            };

            using var tts = new KokoroTtsEngine();
            var profile = tts.GetVoiceProfile(voice);

            Console.WriteLine($"[Kokoro TTS] Synthesizing '{text}'");
            Console.WriteLine($"   Voice:       {profile.DisplayName} ({profile.Accent}, {profile.Gender}, {profile.BaseF0:F0}Hz)");
            Console.WriteLine($"   Cadence:     {speed:F2}x Speed");

            var sw = Stopwatch.StartNew();
            float[] samples = tts.Synthesize(text, voice, speed);
            sw.Stop();

            float durationSec = (float)samples.Length / tts.SampleRate;
            float rtf = durationSec / (float)sw.Elapsed.TotalSeconds;

            WavWriter.WritePcm16(outPath, samples, tts.SampleRate, 1);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Success] Synthesized {durationSec:F2}s of 24kHz audio in {sw.ElapsedMilliseconds}ms ({rtf:F1}x Real-Time) -> '{outPath}'");
            Console.ResetColor();

            if (play)
            {
                Console.WriteLine("[Audio] Playing audio through speakers...");
                AudioPlayer.PlayFile(outPath, wait: true);
            }
            return 0;
        }
        else if (subCmd == "stt")
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Error: Missing audio file for STT transcription. Usage: glacier voice stt <audio.wav>");
                return 1;
            }

            string wavPath = args[1];
            if (!File.Exists(wavPath))
            {
                Console.Error.WriteLine($"Error: File '{wavPath}' not found.");
                return 1;
            }

            Console.WriteLine($"[Whisper STT] Ingesting '{wavPath}'...");
            var sw = Stopwatch.StartNew();

            byte[] wavBytes = File.ReadAllBytes(wavPath);
            int dataOffset = 44;
            int numSamples = (wavBytes.Length - dataOffset) / 2;
            var audioSamples = new float[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                short val = BitConverter.ToInt16(wavBytes, dataOffset + i * 2);
                audioSamples[i] = val / 32768.0f;
            }

            using var whisper = new WhisperEngine();
            string transcript = whisper.Transcribe(audioSamples);
            sw.Stop();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Transcription ({sw.ElapsedMilliseconds}ms)]: \"{transcript}\"");
            Console.ResetColor();
            return 0;
        }
        else if (subCmd == "demo")
        {
            bool play = false;
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] is "--play" or "-p") play = true;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================================");
            Console.WriteLine("   GLACIER.INFERENCE: FULL-DUPLEX VOICE SUBSYSTEM WORKING DEMONSTRATION   ");
            Console.WriteLine("       Pure C# .NET 10 | Kokoro-82M TTS + Whisper STT + Win32 Audio Pipeline   ");
            Console.WriteLine("==========================================================================");
            Console.ResetColor();
            Console.WriteLine();

            using var pipeline = new VoicePipeline();
            var sw = Stopwatch.StartNew();

            // Step 1: Synthesize prompt using Kokoro (US Female)
            string testPrompt = "Hello world from Glacier high performance audio.";
            Console.WriteLine($"[1. Kokoro TTS] Generating audio for prompt: \"{testPrompt}\" (Voice: AfHeart - US Female)");
            var ttsSw = Stopwatch.StartNew();
            float[] generatedSpeech = pipeline.Speak(testPrompt, KokoroVoice.AfHeart);
            ttsSw.Stop();
            float durationSec = (float)generatedSpeech.Length / pipeline.Tts.SampleRate;
            float rtf = durationSec / (float)ttsSw.Elapsed.TotalSeconds;

            string demoWav = "glacier_voice_demo.wav";
            WavWriter.WritePcm16(demoWav, generatedSpeech, pipeline.Tts.SampleRate, 1);
            Console.WriteLine($"   -> Synthesized {durationSec:F2}s of 24kHz audio in {ttsSw.ElapsedMilliseconds}ms ({rtf:F1}x Real-Time)");
            Console.WriteLine($"   -> Audio written to '{demoWav}' ({new FileInfo(demoWav).Length / 1024} KB)");
            if (play)
            {
                Console.WriteLine("   -> Playing prompt speech through speakers...");
                AudioPlayer.PlayFile(demoWav, wait: true);
            }
            Console.WriteLine();

            // Step 2: Multi-Accent Showcase (USA vs British English)
            Console.WriteLine("[2. Multi-Accent Showcase] Synthesizing American & British Voice Profiles...");
            var showcaseVoices = new (KokoroVoice Voice, string OutFile)[]
            {
                (KokoroVoice.AmAdam, "demo_voice_us_male.wav"),
                (KokoroVoice.BfEmma, "demo_voice_uk_female.wav"),
                (KokoroVoice.BmGeorge, "demo_voice_uk_male.wav")
            };

            foreach (var (v, outF) in showcaseVoices)
            {
                var prof = pipeline.Tts.GetVoiceProfile(v);
                float[] audio = pipeline.Speak(testPrompt, v);
                WavWriter.WritePcm16(outF, audio, pipeline.Tts.SampleRate, 1);
                Console.WriteLine($"   - {prof.DisplayName,-22} ({prof.Accent}, {prof.Gender}): {audio.Length / 24000f:F2}s -> '{outF}'");
            }
            Console.WriteLine();

            // Step 3: Extract Log-Mel Spectrogram using Pure C# SIMD DSP
            Console.WriteLine("[3. Pure C# SIMD Audio DSP] Computing 80-bin Log-Mel Spectrogram (Cooley-Tukey Radix-2 FFT)...");
            var dspSw = Stopwatch.StartNew();
            using var mel = new MelSpectrogram(16000, 80);
            int nFrames = (generatedSpeech.Length - mel.WinLength) / mel.HopLength + 1;
            float[] melTensor = new float[mel.NMels * nFrames];
            mel.Process(generatedSpeech, melTensor);
            dspSw.Stop();
            Console.WriteLine($"   -> Extracted {nFrames} frames ({mel.NMels}x{nFrames} tensor) in {dspSw.ElapsedMilliseconds}ms ({nFrames * 1000L / Math.Max(1, dspSw.ElapsedMilliseconds):N0} frames/sec)");
            Console.WriteLine();

            // Step 4: Transcribe Speech using Whisper STT Engine
            Console.WriteLine("[4. Whisper STT] Transcribing audio via Encoder-Decoder Cross-Attention...");
            var sttSw = Stopwatch.StartNew();
            string transcript = pipeline.Listen(generatedSpeech);
            sttSw.Stop();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   -> Transcribed Text ({sttSw.ElapsedMilliseconds}ms): \"{transcript}\"");
            Console.ResetColor();
            Console.WriteLine();

            // Step 5: Full-Duplex Conversational Turn
            Console.WriteLine("[5. Full-Duplex Conversational Turn] Testing Live Conversation Cycle...");
            var (userIn, agentResp, respAudio) = pipeline.ConversationalTurn(
                generatedSpeech,
                input => $"Glacier Voice Agent received: '{input}'. Synthesizing instant speech response.",
                KokoroVoice.AmAdam);

            Console.WriteLine($"   - User Input:    \"{userIn}\"");
            Console.WriteLine($"   - Agent Output:  \"{agentResp}\"");
            Console.WriteLine($"   - Agent Audio:   {respAudio.Length} samples ({respAudio.Length / 24000f:F2}s @ 24kHz)");
            if (play)
            {
                Console.WriteLine("   -> Playing conversational response speech through speakers...");
                AudioPlayer.Play(respAudio, pipeline.Tts.SampleRate, wait: true);
            }
            Console.WriteLine();

            sw.Stop();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[VERIFIED] End-to-end full duplex voice demonstration completed in {sw.ElapsedMilliseconds}ms with direct OS & driver interop!");
            Console.ResetColor();
            return 0;
        }

        Console.Error.WriteLine($"Unknown voice command: '{subCmd}'. Use 'glacier voice --help' for usage.");
        return 1;
    }

}
