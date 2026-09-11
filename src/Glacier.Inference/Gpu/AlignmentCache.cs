namespace Glacier.Inference.Gpu;

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Glacier.Inference.Gguf;

/// <summary>
/// GPU Alignment Cache Engine: Repacks GGUF 144-byte super-blocks into 128-byte warp-coalesced
/// VRAM layouts with decoupled pre-unpacked scale tables, eliminating cache-line straddling.
/// </summary>
public static class AlignmentCache
{
    private const uint Magic = 0x43414C47; // 'GLAC'
    private const uint Version = 1;

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Glacier", "Inference", "Cache");

    public static string GetCachePath(string modelPath)
    {
        Directory.CreateDirectory(CacheDir);
        using var md5 = MD5.Create();
        byte[] hash = md5.ComputeHash(Encoding.UTF8.GetBytes(Path.GetFullPath(modelPath)));
        string hashStr = Convert.ToHexString(hash);
        return Path.Combine(CacheDir, $"{Path.GetFileNameWithoutExtension(modelPath)}_{hashStr}.glacier");
    }

    public static bool TryGetCachedPath(string modelPath, out string cachedPath)
    {
        cachedPath = GetCachePath(modelPath);
        if (File.Exists(cachedPath))
        {
            try
            {
                using var fs = File.OpenRead(cachedPath);
                using var br = new BinaryReader(fs);
                uint magic = br.ReadUInt32();
                uint ver = br.ReadUInt32();
                long srcLen = br.ReadInt64();
                var fi = new FileInfo(modelPath);
                if (magic == Magic && ver == Version && srcLen == fi.Length)
                {
                    return true;
                }
            }
            catch
            {
                // corrupted cache, invalidate
            }
        }
        return false;
    }
}
