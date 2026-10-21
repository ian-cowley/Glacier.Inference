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

/// <summary>
/// IQ2_XXS (Type 16) quantization super-block: 256 elements in 66 bytes (2.0625 b/w).
/// 2-byte FP16 super-scale + 32 ushort (64 bytes) quantized grid/sign indices.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockIQ2_XXS
{
    public Half Delta;          // 2 bytes: super-block scale
    public fixed ushort Qs[32]; // 64 bytes (32 * 2 = 64 bytes)
}

/// <summary>
/// IQ4_NL (Type 20) non-linear 4-bit quantization block: 32 elements in 18 bytes (4.50 b/w).
/// Compatible with non-256-multiple dimensions (such as 640-col ffn_down_exps in Qwen3.8-Flash-Next).
/// Uses the 16-entry non-linear KValuesIq4Nl codebook.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockIQ4_NL
{
    public Half Delta;        // 2 bytes: FP16 block scale
    public fixed byte Qs[16]; // 16 bytes: low nibbles = elements 0..15, high nibbles = elements 16..31
}

/// <summary>
/// Q2_0 (Type 36) fastlookup-free 2-bit quantization block: 64 elements in 20 bytes (2.50 b/w).
/// Compatible with 64-element alignments (such as 640-col ffn_down_exps in Qwen3.8-Flash-Next)
/// and naturally 4-byte aligned (20 bytes) for Direct3D 12 / GPU coalesced loads with zero LUT gather stalls.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ2_0
{
    public Half Delta;        // 2 bytes: FP16 scale
    public Half Min;          // 2 bytes: FP16 min / zero-point offset
    public fixed byte Qs[16]; // 16 bytes: 64 2-bit quants (4 per byte: bits 0..1, 2..3, 4..5, 6..7)
}

/// <summary>
/// IQ3_S (Type 21) non-linear 3-bit quantization super-block: 256 elements in 110 bytes (3.4375 b/w).
/// Used for retained GSQ-RCO experts in Qwen3.8-Flash-Next and Qwen3.8-Flash-Next-Coder.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockIQ3_S
{
    public Half Delta;           // 2 bytes: super-block FP16 scale
    public fixed byte Qs[64];    // 64 bytes: low 8 bits of 64 9-bit codebook indices
    public fixed byte Qh[8];     // 8 bytes: high 1 bit of 64 9-bit codebook indices
    public fixed byte Signs[32]; // 32 bytes: 256 sign bits (4 signs per nibble)
    public fixed byte Scales[4]; // 4 bytes: 8 4-bit sub-block scales
}

/// <summary>
/// Ternary 1.58-bit (Type 100) quantization super-block: 256 elements in 68 bytes (2.125 b/w).
/// Weights W in {-1, 0, +1} encoded as 2-bit values (00=0, 01=+1, 10=-1, 11=0),
/// 4 weights per byte, 64 bytes of weights + 2-byte FP16 super-scale Delta + 2-byte alignment pad
/// for Direct3D 12 4-byte coalesced DWORD loads.
/// Multiplication-free compute: inner products require only addition and subtraction.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockTernary256
{
    public Half Delta;         // 2 bytes: super-block scale (d = mean(|W|))
    public ushort Pad;         // 2 bytes: alignment padding for 4-byte GPU coalescing
    public fixed byte Qs[64];  // 64 bytes: 256 2-bit ternary quants (4 per byte)
}



