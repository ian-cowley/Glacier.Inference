namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string GemvIQ4_XS = @"
cbuffer Params : register(b0) { uint K; uint M; uint Dummy0; uint Dummy1; };
ByteAddressBuffer W : register(t0);
StructuredBuffer<float> X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

static const int kvalues_iq4nl[16] = { -127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113 };
groupshared float smem[64];

float LoadF16(uint byteOff) {
    uint aligned = byteOff & ~3u;
    uint word = W.Load(aligned);
    uint shift = (byteOff & 3u) * 8u;
    uint h = (word >> shift) & 0xFFFFu;
    if ((byteOff & 3u) == 3u) h = ((word >> 24) | (W.Load(aligned + 4u) << 8)) & 0xFFFFu;
    return f16tof32(h);
}

uint LoadU32Unaligned(uint byteOff) {
    uint aligned = byteOff & ~3u;
    uint shift = (byteOff & 3u) * 8u;
    uint a = W.Load(aligned);
    return (shift == 0u) ? a : ((a >> shift) | (W.Load(aligned + 4u) << (32u - shift)));
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= M) return;
    uint t = tid.x;
    uint nBlk = K >> 8;
    uint rowOff = row * nBlk * 136u;
    uint sub = t >> 3;
    uint lane = t & 7u;
    float acc = 0.0f;

    for (uint blk = 0u; blk < nBlk; blk++) {
        uint bo = rowOff + blk * 136u;
        float d = LoadF16(bo);
        uint sh = LoadU32Unaligned(bo + 2u) & 0xFFFFu;
        uint sl = LoadU32Unaligned(bo + 4u);
        uint ls = (((sl >> (4u * sub)) & 0xFu) | (((sh >> (2u * sub)) & 3u) << 4));
        float dl = d * (float)((int)ls - 32);
        uint qword = LoadU32Unaligned(bo + 8u + sub * 16u + (lane & 3u) * 4u);
        uint xBase = blk * 256u + sub * 32u + lane * 4u;
        if (lane < 4u) {
            acc += dl * (
                kvalues_iq4nl[qword & 0xFu] * X[xBase + 0u] +
                kvalues_iq4nl[(qword >> 8) & 0xFu] * X[xBase + 1u] +
                kvalues_iq4nl[(qword >> 16) & 0xFu] * X[xBase + 2u] +
                kvalues_iq4nl[(qword >> 24) & 0xFu] * X[xBase + 3u]);
        } else {
            acc += dl * (
                kvalues_iq4nl[(qword >> 4) & 0xFu] * X[xBase + 0u] +
                kvalues_iq4nl[(qword >> 12) & 0xFu] * X[xBase + 1u] +
                kvalues_iq4nl[(qword >> 20) & 0xFu] * X[xBase + 2u] +
                kvalues_iq4nl[(qword >> 28) & 0xFu] * X[xBase + 3u]);
        }
    }

    smem[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32u; s > 0u; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) Y[row] = smem[0];
}
";

    public const string GemvMXFP4 = @"
cbuffer Params : register(b0) { uint K; uint M; uint Dummy0; uint Dummy1; };
ByteAddressBuffer W : register(t0);
StructuredBuffer<float> X : register(t1);
RWStructuredBuffer<float> Y : register(u0);

static const float kvalues_mxfp4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
groupshared float smem[64];

uint LoadU32Unaligned(uint byteOff) {
    uint aligned = byteOff & ~3u;
    uint shift = (byteOff & 3u) * 8u;
    uint a = W.Load(aligned);
    return (shift == 0u) ? a : ((a >> shift) | (W.Load(aligned + 4u) << (32u - shift)));
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= M) return;
    uint t = tid.x;
    uint nBlk = K >> 5;
    uint rowOff = row * nBlk * 17u;
    uint j = t & 7u;
    float acc = 0.0f;

    for (uint blk = t >> 3; blk < nBlk; blk += 8u) {
        uint bo = rowOff + blk * 17u;
        uint e = LoadU32Unaligned(bo) & 0xFFu;
        float d = (e == 0u) ? 0.0f : asfloat(e << 23) * 0.5f;
        uint qword = LoadU32Unaligned(bo + 1u + (j & 3u) * 4u);
        uint xBase = blk * 32u + j * 4u;
        if (j < 4u) {
            acc += d * (
                kvalues_mxfp4[qword & 0xFu] * X[xBase + 0u] +
                kvalues_mxfp4[(qword >> 8) & 0xFu] * X[xBase + 1u] +
                kvalues_mxfp4[(qword >> 16) & 0xFu] * X[xBase + 2u] +
                kvalues_mxfp4[(qword >> 24) & 0xFu] * X[xBase + 3u]);
        } else {
            acc += d * (
                kvalues_mxfp4[(qword >> 4) & 0xFu] * X[xBase + 0u] +
                kvalues_mxfp4[(qword >> 12) & 0xFu] * X[xBase + 1u] +
                kvalues_mxfp4[(qword >> 20) & 0xFu] * X[xBase + 2u] +
                kvalues_mxfp4[(qword >> 28) & 0xFu] * X[xBase + 3u]);
        }
    }

    smem[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32u; s > 0u; s >>= 1) {
        if (t < s) smem[t] += smem[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) Y[row] = smem[0];
}
";
}
