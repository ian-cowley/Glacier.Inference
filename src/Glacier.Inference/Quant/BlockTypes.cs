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
/// Q5_0 quantization block: 32 elements in 22 bytes (2-byte FP16 scale + 4-byte high bits + 16 bytes 4-bit nibbles).
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ5_0
{
    public Half Delta;
    public uint Qh;
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

/// <summary>
/// Q5_K quantization super-block: 256 elements in 176 bytes.
/// Combines 4-bit lower nibbles with 1-bit upper quants from Qh and 12-byte scales.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ5_K
{
    public Half Delta;
    public Half DeltaMin;
    public fixed byte Scales[12];
    public fixed byte Qh[32];
    public fixed byte Qs[128];
}

/// <summary>
/// Q3_K quantization super-block: 256 elements in 110 bytes.
/// 3-bit quantization using 1-bit hmask, 2-bit qs, 12-byte scales, and FP16 delta.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ3_K
{
    public fixed byte Hmask[32];
    public fixed byte Qs[64];
    public fixed byte Scales[12];
    public Half Delta;
}

/// <summary>
/// MXFP4 (Type 39) quantization block: 32 elements in 17 bytes.
/// 1 byte E8M0 scale byte + 16 bytes of 4-bit FP4 (E2M1) values.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockMXFP4
{
    public byte Scale;
    public fixed byte Qs[16];
}

/// <summary>
/// IQ4_XS (Type 23) quantization super-block: 256 elements in 136 bytes.
/// Non-linear 4-bit quantization with 8 sub-blocks of 32 using kvalues_iq4nl lookup table.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockIQ4_XS
{
    public Half Delta;            // 2 bytes
    public ushort ScalesH;        // 2 bytes (high 2 bits of the 8 6-bit scales)
    public fixed byte ScalesL[4]; // 4 bytes (low 4 bits of the 8 6-bit scales, 2 per byte)
    public fixed byte Qs[128];    // 128 bytes (4-bit table indices for 256 elements)
}

/// <summary>
/// Q2_K quantization super-block: 256 elements in 84 bytes (~2.625 bits per weight).
/// 16 scales/mins (4-bit each, packed in 16 bytes), 64 bytes of 2-bit quants, and FP16 delta/dmin.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ2_K
{
    public fixed byte Scales[16]; // 16 bytes: 16 4-bit scales (low nibbles) and 16 4-bit mins (high nibbles)
    public fixed byte Qs[64];     // 64 bytes: 256 2-bit quants (2 bits per value, 4 values per byte)
    public Half Delta;            // 2 bytes: super-block scale
    public Half DeltaMin;         // 2 bytes: super-block min scale
}

