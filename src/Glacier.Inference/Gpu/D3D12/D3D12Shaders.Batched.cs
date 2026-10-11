namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string GemmQ4KBatch = @"
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

// 32 columns x 32 tokens = 1024 floats = exactly 4,096 bytes (4 KB LDS)
groupshared float s_act[1024];

void get_scale_min(uint j, uint s0, uint s1, uint s2, out float d, out float m, float d_val, float min_val)
{
    uint sc, m_code;
    if (j < 4)
    {
        sc = (s0 >> (j * 8)) & 63;
        m_code = (s1 >> (j * 8)) & 63;
    }
    else
    {
        sc = ((s2 >> ((j - 4) * 8)) & 0xF) | (((s0 >> ((j - 4) * 8 + 6)) & 3) << 4);
        m_code = ((s2 >> ((j - 4) * 8 + 4)) & 0xF) | (((s1 >> ((j - 4) * 8 + 6)) & 3) << 4);
    }
    d = d_val * (float)sc;
    m = min_val * (float)m_code;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;          // 0 .. 31 (token offset in group)
    uint row_in_grp = gtid.y;       // 0 .. 3  (row offset in group)
    uint tid = row_in_grp * 32 + lane_id; // 0 .. 127 (flat thread ID for collaborative load)
    uint warp_id = gid.x * 4 + row_in_grp; // Matrix row m
    uint t_base = gid.y * 32;              // Base token index
    uint t = t_base + lane_id;             // Exact token index for this thread

    float acc = 0.0f;

    uint total_sub_blocks = k_cols / 32;
    uint nb = k_cols / 256;
    uint row_byte_offset = warp_id * nb * 144;

    for (uint sb = 0; sb < total_sub_blocks; sb++)
    {
        uint k_base = sb * 32;

        // 1. Collaborative Coalesced Load of Activations into 4 KB LDS
        uint load_t = tid / 4;          // 0 .. 31 (token in tile)
        uint load_k = (tid % 4) * 8;    // 0, 8, 16, 24 (column in tile)
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        // Transposed layout [column][token] -> zero bank conflicts
        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Decode Weights and Compute Dot Product
        if (warp_id < m_rows && t < batch_size)
        {
            uint b = sb / 8;
            uint j = sb % 8;
            uint blk_addr = row_byte_offset + b * 144;

            uint4 header = W.Load4(blk_addr);
            float d = f16tof32(header.x & 0xFFFF);
            float min_val = f16tof32(header.x >> 16);

            float d_j, min_j;
            get_scale_min(j, header.y, header.z, header.w, d_j, min_j, d, min_val);

            uint chunk = j / 2;
            uint is_high = j % 2;
            uint q_base = blk_addr + 16 + chunk * 32;

            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                uint q_val = W.Load(q_base + w_word * 4);
                uint q0 = q_val & 0xFF;
                uint q1 = (q_val >> 8) & 0xFF;
                uint q2 = (q_val >> 16) & 0xFF;
                uint q3 = q_val >> 24;

                float4 w_vec;
                if (is_high == 0)
                {
                    w_vec = d_j * float4((float)(q0 & 0x0F), (float)(q1 & 0x0F), (float)(q2 & 0x0F), (float)(q3 & 0x0F)) - min_j;
                }
                else
                {
                    w_vec = d_j * float4((float)(q0 >> 4), (float)(q1 >> 4), (float)(q2 >> 4), (float)(q3 >> 4)) - min_j;
                }

                uint k_idx = w_word * 4;
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store across all 32 lanes (Zero reduction barriers!)
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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

    public const string GemmQ6KBatch = @"
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

// 32 columns x 32 tokens = 1024 floats = 4 KB LDS
groupshared float s_act[1024];

int decode_signed_byte_batch(uint b)
{
    return (b >= 128) ? (int)b - 256 : (int)b;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;
    uint row_in_grp = gtid.y;
    uint tid = row_in_grp * 32 + lane_id;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint t_base = gid.y * 32;
    uint t = t_base + lane_id;

    float acc = 0.0f;

    uint total_sub_blocks = k_cols / 32;
    uint nb = k_cols / 256;
    uint row_byte_offset = warp_id * nb * 212;

    for (uint sb = 0; sb < total_sub_blocks; sb++)
    {
        uint k_base = sb * 32;

        // 1. Collaborative Coalesced Load of Activations
        uint load_t = tid / 4;
        uint load_k = (tid % 4) * 8;
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Decode Q6_K Weights for Sub-Block
        if (warp_id < m_rows && t < batch_size)
        {
            uint b = sb / 8;
            uint sub_in_b = sb % 8; // 0 .. 7
            uint step = sub_in_b / 4; // 0 or 1
            uint chunk = sub_in_b % 4; // 0 .. 3
            uint blk_addr = row_byte_offset + b * 212;

            float d = f16tof32(W.Load(blk_addr + 208) & 0xFFFF);
            uint ql_base = blk_addr + step * 64;
            uint qh_base = blk_addr + 128 + step * 32;
            uint sc_addr = blk_addr + 192 + step * 8;

            uint sc_low = W.Load(sc_addr);
            uint sc_high = W.Load(sc_addr + 4);

            int s0, s1;
            if (chunk == 0)
            {
                s0 = decode_signed_byte_batch(sc_low & 0xFF);
                s1 = decode_signed_byte_batch((sc_low >> 8) & 0xFF);
            }
            else if (chunk == 1)
            {
                s0 = decode_signed_byte_batch((sc_low >> 16) & 0xFF);
                s1 = decode_signed_byte_batch((sc_low >> 24) & 0xFF);
            }
            else if (chunk == 2)
            {
                s0 = decode_signed_byte_batch(sc_high & 0xFF);
                s1 = decode_signed_byte_batch((sc_high >> 8) & 0xFF);
            }
            else
            {
                s0 = decode_signed_byte_batch((sc_high >> 16) & 0xFF);
                s1 = decode_signed_byte_batch((sc_high >> 24) & 0xFF);
            }

            float d_scale0 = d * (float)s0;
            float d_scale1 = d * (float)s1;

            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                float d_scale = (w_word < 4) ? d_scale0 : d_scale1;
                uint l = w_word * 4;
                uint qh_dword = W.Load(qh_base + l);
                uint ql_offset = (chunk == 1 || chunk == 3) ? 32 : 0;
                uint ql_dword = W.Load(ql_base + ql_offset + l);

                float4 w_vec;
                [unroll]
                for (uint i = 0; i < 4; i++)
                {
                    uint ql_b = (ql_dword >> (i * 8)) & 0xFF;
                    uint qh_b = (qh_dword >> (i * 8)) & 0xFF;
                    int q;
                    if (chunk == 0)
                    {
                        q = (int)((ql_b & 0x0F) | (((qh_b >> 0) & 3) << 4)) - 32;
                    }
                    else if (chunk == 1)
                    {
                        q = (int)((ql_b & 0x0F) | (((qh_b >> 2) & 3) << 4)) - 32;
                    }
                    else if (chunk == 2)
                    {
                        q = (int)((ql_b >> 4) | (((qh_b >> 4) & 3) << 4)) - 32;
                    }
                    else
                    {
                        q = (int)((ql_b >> 4) | (((qh_b >> 6) & 3) << 4)) - 32;
                    }
                    w_vec[i] = d_scale * (float)q;
                }

                uint k_idx = w_word * 4;
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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

    public const string GemmQ8_0Batch = @"
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

// 32 columns x 32 tokens = 1024 floats = 4 KB LDS
groupshared float s_act[1024];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;
    uint row_in_grp = gtid.y;
    uint tid = row_in_grp * 32 + lane_id;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint t_base = gid.y * 32;
    uint t = t_base + lane_id;

    float acc = 0.0f;

    uint nb = k_cols / 32;
    uint row_byte_offset = warp_id * nb * 36;

    for (uint b = 0; b < nb; b++)
    {
        uint k_base = b * 32;

        // 1. Collaborative Coalesced Load of Activations
        uint load_t = tid / 4;
        uint load_k = (tid % 4) * 8;
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Decode Q8_0 Block and Accumulate
        if (warp_id < m_rows && t < batch_size)
        {
            uint blk_addr = row_byte_offset + b * 36;
            float d = f16tof32(W.Load(blk_addr) & 0xFFFF);

            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                uint q_raw = W.Load(blk_addr + 4 + w_word * 4);
                int q0 = (int)(q_raw << 24) >> 24;
                int q1 = (int)(q_raw << 16) >> 24;
                int q2 = (int)(q_raw << 8) >> 24;
                int q3 = (int)q_raw >> 24;

                float4 w_vec = d * float4((float)q0, (float)q1, (float)q2, (float)q3);
                uint k_idx = w_word * 4;
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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

    public const string GemmFp32Batch = @"
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

// 32 columns x 32 tokens = 1024 floats = 4 KB LDS
groupshared float s_act[1024];

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;
    uint row_in_grp = gtid.y;
    uint tid = row_in_grp * 32 + lane_id;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint t_base = gid.y * 32;
    uint t = t_base + lane_id;

    float acc = 0.0f;

    uint total_tiles = k_cols / 32;
    uint row_offset = warp_id * k_cols;

    for (uint tile = 0; tile < total_tiles; tile++)
    {
        uint k_base = tile * 32;

        // 1. Collaborative Coalesced Load of Activations
        uint load_t = tid / 4;
        uint load_k = (tid % 4) * 8;
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Load FP32 Weights and Accumulate
        if (warp_id < m_rows && t < batch_size)
        {
            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                uint k_idx = w_word * 4;
                float4 w_vec = asfloat(W.Load4((row_offset + k_base + k_idx) * 4));
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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

    public const string GemmQ3KBatch = @"
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

// 32 columns x 32 tokens = 1024 floats = 4 KB LDS
groupshared float s_act[1024];

int get_scale_q3_batch(uint idx, uint sc0, uint sc1, uint sc2, uint sc3)
{
    uint b;
    if (idx < 4) b = (sc0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) b = (sc1 >> ((idx - 4) * 8)) & 0xFF;
    else if (idx < 12) b = (sc2 >> ((idx - 8) * 8)) & 0xFF;
    else b = (sc3 >> ((idx - 12) * 8)) & 0xFF;
    return (int)b - 32;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;
    uint row_in_grp = gtid.y;
    uint tid = row_in_grp * 32 + lane_id;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint t_base = gid.y * 32;
    uint t = t_base + lane_id;

    float acc = 0.0f;

    uint total_sub_blocks = k_cols / 32;
    uint nb = k_cols / 256;
    uint row_byte_offset = warp_id * nb * 112;

    for (uint sb = 0; sb < total_sub_blocks; sb++)
    {
        uint k_base = sb * 32;

        // 1. Collaborative Coalesced Load of Activations
        uint load_t = tid / 4;
        uint load_k = (tid % 4) * 8;
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Decode Q3_K Weights and Compute Dot Product
        if (warp_id < m_rows && t < batch_size)
        {
            uint b = sb / 8;
            uint sub_in_b = sb % 8; // 0 .. 7
            uint step = sub_in_b / 4; // 0 or 1
            uint j = sub_in_b % 4;    // 0 .. 3
            uint blk_addr = row_byte_offset + b * 112;

            uint raw_aux0 = W.Load(blk_addr + 96);
            uint raw_aux1 = W.Load(blk_addr + 100);
            uint raw_aux2 = W.Load(blk_addr + 104);

            const uint kmask1 = 0x03030303;
            const uint kmask2 = 0x0F0F0F0F;
            uint sc0 = (raw_aux0 & kmask2) | (((raw_aux2 >> 0) & kmask1) << 4);
            uint sc1 = (raw_aux1 & kmask2) | (((raw_aux2 >> 2) & kmask1) << 4);
            uint sc2 = ((raw_aux0 >> 4) & kmask2) | (((raw_aux2 >> 4) & kmask1) << 4);
            uint sc3 = ((raw_aux1 >> 4) & kmask2) | (((raw_aux2 >> 6) & kmask1) << 4);

            float d_all = f16tof32(W.Load(blk_addr + 108) & 0xFFFF);
            uint is_idx0 = step * 8 + j * 2 + 0;
            uint is_idx1 = step * 8 + j * 2 + 1;
            float dl0 = d_all * (float)get_scale_q3_batch(is_idx0, sc0, sc1, sc2, sc3);
            float dl1 = d_all * (float)get_scale_q3_batch(is_idx1, sc0, sc1, sc2, sc3);

            uint q_base = blk_addr + 32 + step * 32;
            uint m = 1 << (step * 4 + j);

            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                float dl = (w_word < 4) ? dl0 : dl1;
                uint word_offset = w_word * 4;
                uint q_dword = W.Load(q_base + word_offset);
                uint hm_dword = W.Load(blk_addr + word_offset);

                float4 w_vec;
                [unroll]
                for (uint i = 0; i < 4; i++)
                {
                    uint q_byte = (q_dword >> (i * 8)) & 0xFF;
                    uint hm_byte = (hm_dword >> (i * 8)) & 0xFF;
                    int q = (int)((q_byte >> (j * 2)) & 3) - (((hm_byte & m) != 0) ? 0 : 4);
                    w_vec[i] = dl * (float)q;
                }

                uint k_idx = w_word * 4;
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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

    public const string GemmQ5KBatch = @"
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

// 32 columns x 32 tokens = 1024 floats = 4 KB LDS
groupshared float s_act[1024];

uint get_scale_byte_q5_batch(uint idx, uint s0, uint s1, uint s2)
{
    if (idx < 4) return (s0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) return (s1 >> ((idx - 4) * 8)) & 0xFF;
    else return (s2 >> ((idx - 8) * 8)) & 0xFF;
}

void get_scale_min_q5_batch(uint j, uint s0, uint s1, uint s2, out float d_out, out float min_out, float d, float min_val)
{
    uint sc, m;
    if (j < 4) {
        sc = get_scale_byte_q5_batch(j, s0, s1, s2) & 63;
        m  = get_scale_byte_q5_batch(j + 4, s0, s1, s2) & 63;
    } else {
        sc = (get_scale_byte_q5_batch(j + 4, s0, s1, s2) & 0x0F) | ((get_scale_byte_q5_batch(j - 4, s0, s1, s2) >> 6) << 4);
        m  = (get_scale_byte_q5_batch(j + 4, s0, s1, s2) >> 4)   | ((get_scale_byte_q5_batch(j, s0, s1, s2) >> 6) << 4);
    }
    d_out = d * (float)sc;
    min_out = min_val * (float)m;
}

[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint lane_id = gtid.x;
    uint row_in_grp = gtid.y;
    uint tid = row_in_grp * 32 + lane_id;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint t_base = gid.y * 32;
    uint t = t_base + lane_id;

    float acc = 0.0f;

    uint total_sub_blocks = k_cols / 32;
    uint nb = k_cols / 256;
    uint row_byte_offset = warp_id * nb * 176;

    for (uint sb = 0; sb < total_sub_blocks; sb++)
    {
        uint k_base = sb * 32;

        // 1. Collaborative Coalesced Load of Activations
        uint load_t = tid / 4;
        uint load_k = (tid % 4) * 8;
        uint abs_t = t_base + load_t;
        uint x_read_addr = (abs_t * k_cols + k_base + load_k) * 4;

        float4 ax0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 ax1 = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (abs_t < batch_size && (k_base + load_k) < k_cols)
        {
            ax0 = asfloat(x.Load4(x_read_addr));
            ax1 = asfloat(x.Load4(x_read_addr + 16));
        }

        s_act[(load_k + 0) * 32 + load_t] = ax0.x;
        s_act[(load_k + 1) * 32 + load_t] = ax0.y;
        s_act[(load_k + 2) * 32 + load_t] = ax0.z;
        s_act[(load_k + 3) * 32 + load_t] = ax0.w;
        s_act[(load_k + 4) * 32 + load_t] = ax1.x;
        s_act[(load_k + 5) * 32 + load_t] = ax1.y;
        s_act[(load_k + 6) * 32 + load_t] = ax1.z;
        s_act[(load_k + 7) * 32 + load_t] = ax1.w;

        GroupMemoryBarrierWithGroupSync();

        // 2. Decode Q5_K Weights and Compute Dot Product
        if (warp_id < m_rows && t < batch_size)
        {
            uint b = sb / 8;
            uint sub_in_b = sb % 8; // 0 .. 7
            uint c = sub_in_b / 2;  // 0 .. 3
            uint is_second = sub_in_b % 2; // 0 or 1
            uint blk_addr = row_byte_offset + b * 176;

            uint2 d_hdr = W.Load2(blk_addr);
            float d = f16tof32(d_hdr.x & 0xFFFF);
            float min_val = f16tof32(d_hdr.x >> 16);
            uint s0 = d_hdr.y;
            uint s1 = W.Load(blk_addr + 8);
            uint s2 = W.Load(blk_addr + 12);

            float d_c, min_c;
            get_scale_min_q5_batch(c * 2 + is_second, s0, s1, s2, d_c, min_c, d, min_val);

            uint qh_base = blk_addr + 16;
            uint ql_base = blk_addr + 48 + c * 32;

            [unroll]
            for (uint w_word = 0; w_word < 8; w_word++)
            {
                uint word_offset = w_word * 4;
                uint ql_dword = W.Load(ql_base + word_offset);
                uint qh_dword = W.Load(qh_base + word_offset);

                float4 w_vec;
                [unroll]
                for (uint i = 0; i < 4; i++)
                {
                    uint ql_b = (ql_dword >> (i * 8)) & 0xFF;
                    uint qh_b = (qh_dword >> (i * 8)) & 0xFF;
                    uint bit = (qh_b >> (c * 2 + is_second)) & 1;
                    int q = (is_second == 0) ? (int)((ql_b & 0x0F) | (bit << 4)) : (int)((ql_b >> 4) | (bit << 4));
                    w_vec[i] = d_c * (float)q - min_c;
                }

                uint k_idx = w_word * 4;
                float4 a_vec = float4(
                    s_act[(k_idx + 0) * 32 + lane_id],
                    s_act[(k_idx + 1) * 32 + lane_id],
                    s_act[(k_idx + 2) * 32 + lane_id],
                    s_act[(k_idx + 3) * 32 + lane_id]);

                acc += dot(w_vec, a_vec);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Coalesced Parallel Output Store
    if (t < batch_size && warp_id < m_rows)
    {
        float b_val = (has_bias != 0) ? asfloat(bias.Load(warp_id * 4)) : 0.0f;
        float final_sum = acc + b_val;
        uint out_byte_addr = (t * m_rows + warp_id) * 4;

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


