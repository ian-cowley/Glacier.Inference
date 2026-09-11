namespace Glacier.Inference.Quant;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Q8_0 quantization block: 32 elements in 34 bytes (2-byte FP16 scale + 32 signed int8s).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8_0
{
    public Half Delta;
    public fixed sbyte Qs[32];
}

/// <summary>
/// Q4_0 quantization block: 32 elements in 18 bytes (2-byte FP16 scale + 16 bytes 4-bit nibbles).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ4_0
{
    public Half Delta;
    public fixed byte Qs[16];
}

/// <summary>
/// Q4_K quantization super-block: 256 elements in 144 bytes.
/// Contains scales, offsets, and 4-bit weights for 8 sub-blocks of 32.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ4_K
{
    public Half Delta;
    public Half DeltaMin;
    public fixed byte Scales[12];
    public fixed byte Qs[128];
}

/// <summary>
/// Q6_K quantization super-block: 256 elements in 210 bytes.
/// Combines 4-bit lower nibbles with 2-bit upper nibbles and 16 8-bit scales.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ6_K
{
    public fixed byte Ql[128];
    public fixed byte Qh[64];
    public fixed sbyte Scales[16];
    public Half Delta;
}
