namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    /// <summary>
    /// Direct3D 12 Compute Shader for 1.58-bit Ternary Quantization GEMV (BlockTernary256).
    /// Super-block size: 256 weights in 68 bytes (2.125 b/w, 4-byte coalesced).
    /// Pure branchless sign extraction: s = (c &amp; 1) - ((c &gt;&gt; 1) &amp; 1), mapping natively
    /// to NVIDIA Ada Lovelace single-cycle hardware FMA without conditional pipeline bubbles.
    /// Perfectly coalesced: 32 threads in warp process 2 blocks (512 weights) per step,
    /// each thread reading 1 unique 32-bit DWORD (16 weights) with 0 memory bank conflicts.
    /// </summary>
    public const string GemvTernary = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint has_bias;
    uint has_residual;
    uint has_y;
    uint row_offset;
};

ByteAddressBuffer W : register(t0);
StructuredBuffer<float> bias : register(t1);
RWByteAddressBuffer x : register(u0);
RWStructuredBuffer<float> residual : register(u1);
RWStructuredBuffer<float> y : register(u2);

groupshared float s_mem[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = row_offset + gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 68;
        float row_sum = 0.0f;

        // Warp of 32 threads processes 2 blocks (512 weights) per iteration.
        // Threads 0..15 handle block b + 0; Threads 16..31 handle block b + 1.
        uint blk_in_pair = lane_id / 16;
        uint lane_in_blk = lane_id % 16;

        for (uint b = 0; b < nb; b += 2)
        {
            uint b_curr = b + blk_in_pair;
            uint blk_addr = row_byte_offset + b_curr * 68;

            uint d_raw = W.Load(blk_addr) & 0xFFFF;
            float d = f16tof32(d_raw);

            // Each thread loads 1 unique DWORD containing 16 packed 2-bit ternary weights
            uint w = W.Load(blk_addr + 4 + lane_in_blk * 4);

            uint x_addr = (b_curr * 256 + lane_in_blk * 16) * 4;
            float4 x0 = asfloat(x.Load4(x_addr));
            float4 x1 = asfloat(x.Load4(x_addr + 16));
            float4 x2 = asfloat(x.Load4(x_addr + 32));
            float4 x3 = asfloat(x.Load4(x_addr + 48));

            // Pure branchless sign extraction: (c & 1) - ((c >> 1) & 1)
            // Maps directly to native single-cycle GPU hardware FMA
            float s0 = (float)((int)(w & 1) - (int)((w >> 1) & 1));
            float s1 = (float)((int)((w >> 2) & 1) - (int)((w >> 3) & 1));
            float s2 = (float)((int)((w >> 4) & 1) - (int)((w >> 5) & 1));
            float s3 = (float)((int)((w >> 6) & 1) - (int)((w >> 7) & 1));

            float s4 = (float)((int)((w >> 8) & 1) - (int)((w >> 9) & 1));
            float s5 = (float)((int)((w >> 10) & 1) - (int)((w >> 11) & 1));
            float s6 = (float)((int)((w >> 12) & 1) - (int)((w >> 13) & 1));
            float s7 = (float)((int)((w >> 14) & 1) - (int)((w >> 15) & 1));

            float s8  = (float)((int)((w >> 16) & 1) - (int)((w >> 17) & 1));
            float s9  = (float)((int)((w >> 18) & 1) - (int)((w >> 19) & 1));
            float s10 = (float)((int)((w >> 20) & 1) - (int)((w >> 21) & 1));
            float s11 = (float)((int)((w >> 22) & 1) - (int)((w >> 23) & 1));

            float s12 = (float)((int)((w >> 24) & 1) - (int)((w >> 25) & 1));
            float s13 = (float)((int)((w >> 26) & 1) - (int)((w >> 27) & 1));
            float s14 = (float)((int)((w >> 28) & 1) - (int)((w >> 29) & 1));
            float s15 = (float)((int)((w >> 30) & 1) - (int)((w >> 31) & 1));

            float dot = (s0 * x0.x + s1 * x0.y + s2 * x0.z + s3 * x0.w)
                      + (s4 * x1.x + s5 * x1.y + s6 * x1.z + s7 * x1.w)
                      + (s8 * x2.x + s9 * x2.y + s10 * x2.z + s11 * x2.w)
                      + (s12 * x3.x + s13 * x3.y + s14 * x3.z + s15 * x3.w);

            row_sum += d * dot;
        }

        s_mem[s_idx] = row_sum;
    }
    else
    {
        s_mem[s_idx] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) s_mem[s_idx] += s_mem[s_idx + 16];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  s_mem[s_idx] += s_mem[s_idx + 8];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  s_mem[s_idx] += s_mem[s_idx + 4];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  s_mem[s_idx] += s_mem[s_idx + 2];
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  s_mem[s_idx] += s_mem[s_idx + 1];

    if (lane_id == 0 && warp_id < m_rows)
    {
        float final_sum = s_mem[row_in_grp * 32];
        if (has_bias != 0) final_sum += bias[warp_id];
        if (has_residual != 0) residual[warp_id] += final_sum;
        if (has_y != 0) y[warp_id] = final_sum;
    }
}
";

    public const string GemmTernaryBatch = @"
cbuffer Params : register(b0)
{
    uint k_cols;
    uint m_rows;
    uint batch_size;
    uint has_bias;
    uint has_residual;
    uint has_y;
};

ByteAddressBuffer W : register(t0);
ByteAddressBuffer bias : register(t1);
RWByteAddressBuffer x : register(u2);
RWByteAddressBuffer residual : register(u0);
RWByteAddressBuffer y : register(u1);

groupshared float4 s_mem0[128];
groupshared float4 s_mem1[128];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 8;
    uint valid_t = (t_base < batch_size) ? min(8u, batch_size - t_base) : 0u;

    float4 sum0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (warp_id < m_rows && valid_t > 0)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 68;

        uint blk_in_pair = lane_id / 16;
        uint lane_in_blk = lane_id % 16;

        if (valid_t >= 8)
        {
            uint base0 = (t_base + 0) * k_cols;
            uint base1 = (t_base + 1) * k_cols;
            uint base2 = (t_base + 2) * k_cols;
            uint base3 = (t_base + 3) * k_cols;
            uint base4 = (t_base + 4) * k_cols;
            uint base5 = (t_base + 5) * k_cols;
            uint base6 = (t_base + 6) * k_cols;
            uint base7 = (t_base + 7) * k_cols;

            for (uint b = 0; b < nb; b += 2)
            {
                uint b_curr = b + blk_in_pair;
                uint blk_addr = row_byte_offset + b_curr * 68;

                uint d_raw = W.Load(blk_addr) & 0xFFFF;
                float d = f16tof32(d_raw);

                uint w = W.Load(blk_addr + 4 + lane_in_blk * 4);

                float s0 = (float)((int)(w & 1) - (int)((w >> 1) & 1));
                float s1 = (float)((int)((w >> 2) & 1) - (int)((w >> 3) & 1));
                float s2 = (float)((int)((w >> 4) & 1) - (int)((w >> 5) & 1));
                float s3 = (float)((int)((w >> 6) & 1) - (int)((w >> 7) & 1));
                float4 sw0 = float4(s0, s1, s2, s3) * d;

                float s4 = (float)((int)((w >> 8) & 1) - (int)((w >> 9) & 1));
                float s5 = (float)((int)((w >> 10) & 1) - (int)((w >> 11) & 1));
                float s6 = (float)((int)((w >> 12) & 1) - (int)((w >> 13) & 1));
                float s7 = (float)((int)((w >> 14) & 1) - (int)((w >> 15) & 1));
                float4 sw1 = float4(s4, s5, s6, s7) * d;

                float s8  = (float)((int)((w >> 16) & 1) - (int)((w >> 17) & 1));
                float s9  = (float)((int)((w >> 18) & 1) - (int)((w >> 19) & 1));
                float s10 = (float)((int)((w >> 20) & 1) - (int)((w >> 21) & 1));
                float s11 = (float)((int)((w >> 22) & 1) - (int)((w >> 23) & 1));
                float4 sw2 = float4(s8, s9, s10, s11) * d;

                float s12 = (float)((int)((w >> 24) & 1) - (int)((w >> 25) & 1));
                float s13 = (float)((int)((w >> 26) & 1) - (int)((w >> 27) & 1));
                float s14 = (float)((int)((w >> 28) & 1) - (int)((w >> 29) & 1));
                float s15 = (float)((int)((w >> 30) & 1) - (int)((w >> 31) & 1));
                float4 sw3 = float4(s12, s13, s14, s15) * d;

                uint x_k = b_curr * 256 + lane_in_blk * 16;

                sum0.x += dot(sw0, asfloat(x.Load4((base0 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base0 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base0 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base0 + x_k + 12) * 4)));

                sum0.y += dot(sw0, asfloat(x.Load4((base1 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base1 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base1 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base1 + x_k + 12) * 4)));

                sum0.z += dot(sw0, asfloat(x.Load4((base2 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base2 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base2 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base2 + x_k + 12) * 4)));

                sum0.w += dot(sw0, asfloat(x.Load4((base3 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base3 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base3 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base3 + x_k + 12) * 4)));

                sum1.x += dot(sw0, asfloat(x.Load4((base4 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base4 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base4 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base4 + x_k + 12) * 4)));

                sum1.y += dot(sw0, asfloat(x.Load4((base5 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base5 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base5 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base5 + x_k + 12) * 4)));

                sum1.z += dot(sw0, asfloat(x.Load4((base6 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base6 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base6 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base6 + x_k + 12) * 4)));

                sum1.w += dot(sw0, asfloat(x.Load4((base7 + x_k + 0) * 4)))
                        + dot(sw1, asfloat(x.Load4((base7 + x_k + 4) * 4)))
                        + dot(sw2, asfloat(x.Load4((base7 + x_k + 8) * 4)))
                        + dot(sw3, asfloat(x.Load4((base7 + x_k + 12) * 4)));
            }
        }
        else
        {
            for (uint b = 0; b < nb; b += 2)
            {
                uint b_curr = b + blk_in_pair;
                uint blk_addr = row_byte_offset + b_curr * 68;

                uint d_raw = W.Load(blk_addr) & 0xFFFF;
                float d = f16tof32(d_raw);

                uint w = W.Load(blk_addr + 4 + lane_in_blk * 4);

                float s0 = (float)((int)(w & 1) - (int)((w >> 1) & 1));
                float s1 = (float)((int)((w >> 2) & 1) - (int)((w >> 3) & 1));
                float s2 = (float)((int)((w >> 4) & 1) - (int)((w >> 5) & 1));
                float s3 = (float)((int)((w >> 6) & 1) - (int)((w >> 7) & 1));
                float4 sw0 = float4(s0, s1, s2, s3) * d;

                float s4 = (float)((int)((w >> 8) & 1) - (int)((w >> 9) & 1));
                float s5 = (float)((int)((w >> 10) & 1) - (int)((w >> 11) & 1));
                float s6 = (float)((int)((w >> 12) & 1) - (int)((w >> 13) & 1));
                float s7 = (float)((int)((w >> 14) & 1) - (int)((w >> 15) & 1));
                float4 sw1 = float4(s4, s5, s6, s7) * d;

                float s8  = (float)((int)((w >> 16) & 1) - (int)((w >> 17) & 1));
                float s9  = (float)((int)((w >> 18) & 1) - (int)((w >> 19) & 1));
                float s10 = (float)((int)((w >> 20) & 1) - (int)((w >> 21) & 1));
                float s11 = (float)((int)((w >> 22) & 1) - (int)((w >> 23) & 1));
                float4 sw2 = float4(s8, s9, s10, s11) * d;

                float s12 = (float)((int)((w >> 24) & 1) - (int)((w >> 25) & 1));
                float s13 = (float)((int)((w >> 26) & 1) - (int)((w >> 27) & 1));
                float s14 = (float)((int)((w >> 28) & 1) - (int)((w >> 29) & 1));
                float s15 = (float)((int)((w >> 30) & 1) - (int)((w >> 31) & 1));
                float4 sw3 = float4(s12, s13, s14, s15) * d;

                uint x_k = b_curr * 256 + lane_in_blk * 16;

                [unroll]
                for (uint tt = 0; tt < 4; tt++)
                {
                    if (tt < valid_t)
                    {
                        uint base_t = (t_base + tt) * k_cols;
                        sum0[tt] += dot(sw0, asfloat(x.Load4((base_t + x_k + 0) * 4)))
                                  + dot(sw1, asfloat(x.Load4((base_t + x_k + 4) * 4)))
                                  + dot(sw2, asfloat(x.Load4((base_t + x_k + 8) * 4)))
                                  + dot(sw3, asfloat(x.Load4((base_t + x_k + 12) * 4)));
                    }
                    if ((tt + 4) < valid_t)
                    {
                        uint base_t = (t_base + tt + 4) * k_cols;
                        sum1[tt] += dot(sw0, asfloat(x.Load4((base_t + x_k + 0) * 4)))
                                  + dot(sw1, asfloat(x.Load4((base_t + x_k + 4) * 4)))
                                  + dot(sw2, asfloat(x.Load4((base_t + x_k + 8) * 4)))
                                  + dot(sw3, asfloat(x.Load4((base_t + x_k + 12) * 4)));
                    }
                }
            }
        }
    }

    s_mem0[s_idx] = sum0;
    s_mem1[s_idx] = sum1;
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16) { s_mem0[s_idx] += s_mem0[s_idx + 16]; s_mem1[s_idx] += s_mem1[s_idx + 16]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)  { s_mem0[s_idx] += s_mem0[s_idx + 8];  s_mem1[s_idx] += s_mem1[s_idx + 8]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)  { s_mem0[s_idx] += s_mem0[s_idx + 4];  s_mem1[s_idx] += s_mem1[s_idx + 4]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)  { s_mem0[s_idx] += s_mem0[s_idx + 2];  s_mem1[s_idx] += s_mem1[s_idx + 2]; }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)  { s_mem0[s_idx] += s_mem0[s_idx + 1];  s_mem1[s_idx] += s_mem1[s_idx + 1]; }
    GroupMemoryBarrierWithGroupSync();

    if (lane_id < valid_t && warp_id < m_rows)
    {
        uint base_s = row_in_grp * 32;
        float final_sum = (lane_id < 4) ? s_mem0[base_s][lane_id] : s_mem1[base_s][lane_id - 4];
        if (has_bias != 0) final_sum += asfloat(bias.Load(warp_id * 4));
        uint out_byte_addr = ((t_base + lane_id) * m_rows + warp_id) * 4;

        if (has_residual != 0)
        {
            float prev = asfloat(residual.Load(out_byte_addr));
            residual.Store(out_byte_addr, asuint(prev + final_sum));
        }
        if (has_y != 0)
        {
            y.Store(out_byte_addr, asuint(final_sum));
        }
    }
}
";
}
