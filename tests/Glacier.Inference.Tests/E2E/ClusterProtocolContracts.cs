namespace Glacier.Cluster.Protocol;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Message types for cluster wire protocol testing.
/// </summary>
public enum ClusterMessageType : byte
{
    HeartbeatPing = 0x01,
    HeartbeatPong = 0x02,
    NodeJoinRequest = 0x03,
    NodeJoinResponse = 0x04,
    NodeLeave = 0x05,
    MsgInitStage = 0x60,
    MsgStageReady = 0x61,
    MsgActivation = 0x70,
    MsgResult = 0x71,
    MsgShutdown = 0x73
}

/// <summary>
/// Flags for cluster frame header.
/// </summary>
[Flags]
public enum ClusterFrameFlags : byte
{
    None = 0,
    Compressed = 1 << 0,
    Encrypted = 1 << 1,
    Urgent = 1 << 2
}

/// <summary>
/// Immutable 32-byte wire protocol header for inter-node communications contract testing.
/// Stored in strict Little-Endian format with 64-bit natural machine alignment.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 32)]
public readonly struct ClusterFrameHeader : IEquatable<ClusterFrameHeader>
{
    public const uint MagicConstant = 0x47434C55; // "GCLU"
    public const ushort CurrentVersion = 1;
    public const int HeaderSize = 32;

    public readonly uint Magic;
    public readonly ushort Version;
    public readonly ClusterMessageType MessageType;
    public readonly byte Flags;
    public readonly ulong CorrelationId;
    public readonly ulong PayloadLength;
    public readonly uint PayloadCrc32;
    public readonly uint Reserved;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ClusterFrameHeader(
        ClusterMessageType messageType,
        ulong correlationId,
        ulong payloadLength,
        uint crc32 = 0,
        byte flags = 0,
        ushort version = CurrentVersion,
        uint reserved = 0)
    {
        Magic = MagicConstant;
        Version = version;
        MessageType = messageType;
        Flags = flags;
        CorrelationId = correlationId;
        PayloadLength = payloadLength;
        PayloadCrc32 = crc32;
        Reserved = reserved;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ClusterFrameHeader(
        ClusterMessageType messageType,
        ulong correlationId,
        ulong payloadLength,
        ClusterFrameFlags flags)
        : this(messageType, correlationId, payloadLength, 0, (byte)flags, CurrentVersion, 0)
    {
    }

    public ClusterFrameFlags FrameFlags => (ClusterFrameFlags)Flags;

    public bool Equals(ClusterFrameHeader other) =>
        Magic == other.Magic &&
        Version == other.Version &&
        MessageType == other.MessageType &&
        Flags == other.Flags &&
        CorrelationId == other.CorrelationId &&
        PayloadLength == other.PayloadLength &&
        PayloadCrc32 == other.PayloadCrc32 &&
        Reserved == other.Reserved;

    public override bool Equals(object? obj) => obj is ClusterFrameHeader other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Magic, CorrelationId, PayloadLength);
}
