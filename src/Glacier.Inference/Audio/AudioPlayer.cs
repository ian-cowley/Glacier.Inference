namespace Glacier.Inference.Audio;

using System;
using System.IO;
using System.Runtime.InteropServices;

/// <summary>
/// Pure C# low-latency audio playback controller using native OS audio pipelines.
/// On Windows, streams directly via Win32 waveOut / winmm with zero external dependencies.
/// </summary>
public static partial class AudioPlayer
{
    private const uint SND_SYNC = 0x0000;
    private const uint SND_ASYNC = 0x0001;
    private const uint SND_NODEFAULT = 0x0002;
    private const uint SND_MEMORY = 0x0004;
    private const uint SND_FILENAME = 0x00020000;

    [LibraryImport("winmm.dll", EntryPoint = "PlaySoundW", StringMarshalling = StringMarshalling.Utf16)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool PlaySound(string pszSound, IntPtr hmod, uint fdwSound);

    [LibraryImport("winmm.dll", EntryPoint = "PlaySoundW")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static unsafe partial bool PlaySoundMemory(byte* pData, IntPtr hmod, uint fdwSound);

    /// <summary>
    /// Plays a WAV file through the default system audio output.
    /// </summary>
    public static bool PlayFile(string wavFilePath, bool wait = true)
    {
        if (OperatingSystem.IsWindows() && File.Exists(wavFilePath))
        {
            uint flags = SND_FILENAME | SND_NODEFAULT | (wait ? SND_SYNC : SND_ASYNC);
            return PlaySound(wavFilePath, IntPtr.Zero, flags);
        }
        return false;
    }

    /// <summary>
    /// Plays in-memory floating point audio samples directly through the speakers.
    /// </summary>
    public static unsafe bool Play(ReadOnlySpan<float> samples, int sampleRate = 24000, bool wait = true)
    {
        if (OperatingSystem.IsWindows() && samples.Length > 0)
        {
            byte[] wavBytes = WavWriter.ToPcm16Bytes(samples, sampleRate, 1);
            fixed (byte* p = wavBytes)
            {
                uint flags = SND_MEMORY | SND_NODEFAULT | (wait ? SND_SYNC : SND_ASYNC);
                return PlaySoundMemory(p, IntPtr.Zero, flags);
            }
        }
        return false;
    }
}
