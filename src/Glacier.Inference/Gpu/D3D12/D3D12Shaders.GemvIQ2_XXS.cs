namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    private const string IQ2_XXS_Tables = @"
static const uint2 iq2xxs_grid[256] = {
    uint2(0x08080808, 0x08080808), uint2(0x0808082b, 0x08080808), uint2(0x08081919, 0x08080808), uint2(0x08082b08, 0x08080808),
    uint2(0x08082b2b, 0x08080808), uint2(0x08190819, 0x08080808), uint2(0x08191908, 0x08080808), uint2(0x082b0808, 0x08080808),
    uint2(0x082b082b, 0x08080808), uint2(0x082b2b08, 0x08080808), uint2(0x082b2b2b, 0x08080808), uint2(0x19080819, 0x08080808),
    uint2(0x19081908, 0x08080808), uint2(0x19190808, 0x08080808), uint2(0x19192b08, 0x08080808), uint2(0x192b0819, 0x08080808),
    uint2(0x192b1908, 0x08080808), uint2(0x2b080808, 0x08080808), uint2(0x2b08082b, 0x08080808), uint2(0x2b082b2b, 0x08080808),
    uint2(0x2b2b082b, 0x08080808), uint2(0x08080819, 0x08080819), uint2(0x08081908, 0x08080819), uint2(0x08190808, 0x08080819),
    uint2(0x08191919, 0x08080819), uint2(0x19080808, 0x08080819), uint2(0x2b081908, 0x08080819), uint2(0x2b192b08, 0x08080819),
    uint2(0x08080808, 0x0808082b), uint2(0x0808082b, 0x0808082b), uint2(0x082b082b, 0x0808082b), uint2(0x2b08082b, 0x0808082b),
    uint2(0x08080819, 0x08081908), uint2(0x08081908, 0x08081908), uint2(0x08190808, 0x08081908), uint2(0x082b0819, 0x08081908),
    uint2(0x082b1908, 0x08081908), uint2(0x19080808, 0x08081908), uint2(0x1908082b, 0x08081908), uint2(0x19082b08, 0x08081908),
    uint2(0x192b0808, 0x08081908), uint2(0x2b080819, 0x08081908), uint2(0x2b081908, 0x08081908), uint2(0x2b190808, 0x08081908),
    uint2(0x2b2b1908, 0x08081908), uint2(0x08080808, 0x08081919), uint2(0x0808082b, 0x08081919), uint2(0x08082b08, 0x08081919),
    uint2(0x082b0808, 0x08081919), uint2(0x1908192b, 0x08081919), uint2(0x192b2b19, 0x08081919), uint2(0x2b080808, 0x08081919),
    uint2(0x2b190819, 0x08081919), uint2(0x08082b19, 0x0808192b), uint2(0x08190808, 0x0808192b), uint2(0x19080808, 0x0808192b),
    uint2(0x2b081908, 0x0808192b), uint2(0x2b2b1908, 0x0808192b), uint2(0x08080808, 0x08082b08), uint2(0x08081919, 0x08082b08),
    uint2(0x08082b08, 0x08082b08), uint2(0x08191908, 0x08082b08), uint2(0x082b2b08, 0x08082b08), uint2(0x19080819, 0x08082b08),
    uint2(0x19081908, 0x08082b08), uint2(0x19190808, 0x08082b08), uint2(0x1919082b, 0x08082b08), uint2(0x2b082b08, 0x08082b08),
    uint2(0x08081908, 0x08082b19), uint2(0x19080808, 0x08082b19), uint2(0x0808082b, 0x08082b2b), uint2(0x08191908, 0x08082b2b),
    uint2(0x08080819, 0x08190808), uint2(0x08081908, 0x08190808), uint2(0x08190808, 0x08190808), uint2(0x082b0819, 0x08190808),
    uint2(0x19080808, 0x08190808), uint2(0x192b0808, 0x08190808), uint2(0x2b081908, 0x08190808), uint2(0x2b190808, 0x08190808),
    uint2(0x2b191919, 0x08190808), uint2(0x08080808, 0x08190819), uint2(0x08082b08, 0x08190819), uint2(0x082b0808, 0x08190819),
    uint2(0x19190808, 0x08190819), uint2(0x19192b2b, 0x08190819), uint2(0x2b080808, 0x08190819), uint2(0x082b1908, 0x0819082b),
    uint2(0x19081919, 0x0819082b), uint2(0x08080808, 0x08191908), uint2(0x08082b08, 0x08191908), uint2(0x082b0808, 0x08191908),
    uint2(0x082b1919, 0x08191908), uint2(0x19082b19, 0x08191908), uint2(0x2b080808, 0x08191908), uint2(0x08192b08, 0x08191919),
    uint2(0x192b082b, 0x08191919), uint2(0x08080808, 0x0819192b), uint2(0x0819192b, 0x0819192b), uint2(0x08080819, 0x08192b08),
    uint2(0x08081908, 0x08192b08), uint2(0x08190808, 0x08192b08), uint2(0x19080808, 0x08192b08), uint2(0x2b080819, 0x08192b08),
    uint2(0x08080808, 0x08192b19), uint2(0x08081919, 0x08192b19), uint2(0x2b2b0808, 0x08192b19), uint2(0x19190819, 0x08192b2b),
    uint2(0x08080808, 0x082b0808), uint2(0x0808082b, 0x082b0808), uint2(0x08082b2b, 0x082b0808), uint2(0x19081908, 0x082b0808),
    uint2(0x192b0819, 0x082b0808), uint2(0x2b080808, 0x082b0808), uint2(0x2b08082b, 0x082b0808), uint2(0x082b2b19, 0x082b0819),
    uint2(0x19082b08, 0x082b0819), uint2(0x08080808, 0x082b082b), uint2(0x0808082b, 0x082b082b), uint2(0x08080819, 0x082b1908),
    uint2(0x08081908, 0x082b1908), uint2(0x08190808, 0x082b1908), uint2(0x19080808, 0x082b1908), uint2(0x1919192b, 0x082b1908),
    uint2(0x08080808, 0x082b1919), uint2(0x19080819, 0x082b1919), uint2(0x192b1908, 0x082b1919), uint2(0x2b190808, 0x082b192b),
    uint2(0x08082b08, 0x082b2b08), uint2(0x082b0808, 0x082b2b08), uint2(0x2b191908, 0x082b2b08), uint2(0x19081908, 0x082b2b2b),
    uint2(0x08080819, 0x19080808), uint2(0x08081908, 0x19080808), uint2(0x08190808, 0x19080808), uint2(0x08192b08, 0x19080808),
    uint2(0x082b0819, 0x19080808), uint2(0x082b1908, 0x19080808), uint2(0x19080808, 0x19080808), uint2(0x19082b08, 0x19080808),
    uint2(0x1919192b, 0x19080808), uint2(0x192b0808, 0x19080808), uint2(0x2b080819, 0x19080808), uint2(0x2b081908, 0x19080808),
    uint2(0x2b190808, 0x19080808), uint2(0x08080808, 0x19080819), uint2(0x082b0808, 0x19080819), uint2(0x192b0819, 0x19080819),
    uint2(0x2b080808, 0x19080819), uint2(0x2b081919, 0x19080819), uint2(0x08080819, 0x1908082b), uint2(0x08190808, 0x1908082b),
    uint2(0x19082b08, 0x1908082b), uint2(0x1919192b, 0x1908082b), uint2(0x192b2b08, 0x1908082b), uint2(0x08080808, 0x19081908),
    uint2(0x08082b08, 0x19081908), uint2(0x082b0808, 0x19081908), uint2(0x2b080808, 0x19081908), uint2(0x2b192b19, 0x19081908),
    uint2(0x0819082b, 0x19081919), uint2(0x082b1908, 0x19081919), uint2(0x08080808, 0x1908192b), uint2(0x08080819, 0x19082b08),
    uint2(0x08081908, 0x19082b08), uint2(0x08190808, 0x19082b08), uint2(0x19080808, 0x19082b08), uint2(0x19081919, 0x19082b08),
    uint2(0x08080808, 0x19082b19), uint2(0x19192b08, 0x19082b19), uint2(0x192b0819, 0x19082b19), uint2(0x2b08082b, 0x19082b19),
    uint2(0x19081919, 0x19082b2b), uint2(0x2b190808, 0x19082b2b), uint2(0x08080808, 0x19190808), uint2(0x08082b08, 0x19190808),
    uint2(0x08190819, 0x19190808), uint2(0x08192b19, 0x19190808), uint2(0x082b0808, 0x19190808), uint2(0x2b080808, 0x19190808),
    uint2(0x2b082b08, 0x19190808), uint2(0x08081908, 0x19190819), uint2(0x1908082b, 0x19190819), uint2(0x2b2b1908, 0x19190819),
    uint2(0x2b190819, 0x1919082b), uint2(0x2b190808, 0x19191908), uint2(0x2b19082b, 0x19191908), uint2(0x08082b2b, 0x19191919),
    uint2(0x08080819, 0x1919192b), uint2(0x19191908, 0x1919192b), uint2(0x08080808, 0x19192b08), uint2(0x08190819, 0x19192b08),
    uint2(0x08192b19, 0x19192b08), uint2(0x192b1908, 0x19192b08), uint2(0x19080808, 0x19192b19), uint2(0x08082b08, 0x19192b2b),
    uint2(0x08081908, 0x192b0808), uint2(0x08190808, 0x192b0808), uint2(0x19080808, 0x192b0808), uint2(0x192b2b08, 0x192b0808),
    uint2(0x08080808, 0x192b0819), uint2(0x19191919, 0x192b0819), uint2(0x08192b08, 0x192b082b), uint2(0x192b0808, 0x192b082b),
    uint2(0x08080808, 0x192b1908), uint2(0x08081919, 0x192b1908), uint2(0x08190808, 0x192b1919), uint2(0x0819082b, 0x192b1919),
    uint2(0x2b081908, 0x192b1919), uint2(0x1908082b, 0x192b2b08), uint2(0x08080808, 0x2b080808), uint2(0x0808082b, 0x2b080808),
    uint2(0x08082b2b, 0x2b080808), uint2(0x19080819, 0x2b080808), uint2(0x2b08082b, 0x2b080808), uint2(0x08081908, 0x2b080819),
    uint2(0x08192b08, 0x2b080819), uint2(0x19080808, 0x2b080819), uint2(0x08190819, 0x2b08082b), uint2(0x08080819, 0x2b081908),
    uint2(0x08081908, 0x2b081908), uint2(0x08190808, 0x2b081908), uint2(0x08191919, 0x2b081908), uint2(0x19080808, 0x2b081908),
    uint2(0x192b0808, 0x2b081908), uint2(0x08080808, 0x2b081919), uint2(0x1908192b, 0x2b081919), uint2(0x2b191908, 0x2b081919),
    uint2(0x08082b19, 0x2b08192b), uint2(0x19080808, 0x2b08192b), uint2(0x192b0808, 0x2b08192b), uint2(0x0808082b, 0x2b082b08),
    uint2(0x08081908, 0x2b082b19), uint2(0x08190819, 0x2b082b2b), uint2(0x08081908, 0x2b190808), uint2(0x08190808, 0x2b190808),
    uint2(0x082b1908, 0x2b190808), uint2(0x19080808, 0x2b190808), uint2(0x2b2b0819, 0x2b190808), uint2(0x0819192b, 0x2b190819),
    uint2(0x2b080808, 0x2b190819), uint2(0x19081919, 0x2b19082b), uint2(0x08080808, 0x2b191908), uint2(0x082b082b, 0x2b191908),
    uint2(0x19081908, 0x2b191908), uint2(0x19190819, 0x2b191919), uint2(0x2b080819, 0x2b192b08), uint2(0x082b0808, 0x2b192b19),
    uint2(0x0808082b, 0x2b2b0808), uint2(0x19190808, 0x2b2b0808), uint2(0x2b081919, 0x2b2b0808), uint2(0x08082b19, 0x2b2b0819),
    uint2(0x08080808, 0x2b2b082b), uint2(0x08192b08, 0x2b2b1908), uint2(0x19190808, 0x2b2b2b08), uint2(0x08081908, 0x2b2b2b19),
};

static const uint ksigns_iq2xs[32] = {
    0x03828100, 0x87060584, 0x8b0a0988, 0x0f8e8d0c,
    0x93121190, 0x17969514, 0x1b9a9918, 0x9f1e1d9c,
    0xa32221a0, 0x27a6a524, 0x2baaa928, 0xaf2e2dac,
    0x33b2b130, 0xb73635b4, 0xbb3a39b8, 0x3fbebd3c,
    0xc34241c0, 0x47c6c544, 0x4bcac948, 0xcf4e4dcc,
    0x53d2d150, 0xd75655d4, 0xdb5a59d8, 0x5fdedd5c,
    0x63e2e160, 0xe76665e4, 0xeb6a69e8, 0x6feeed6c,
    0xf37271f0, 0x77f6f574, 0x7bfaf978, 0xff7e7dfc,
};
";

    public const string GemvIQ2_XXS = @"
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
" + IQ2_XXS_Tables + @"
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

        uint ib32 = lane_id / 4;
        uint l = lane_id % 4;
        uint qs_offset = 4 + ib32 * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 68;
            uint d_raw = W.Load(blk_addr) & 0xFFFF;
            float d = f16tof32(d_raw);

            uint aux0 = W.Load(blk_addr + qs_offset);
            uint aux1 = W.Load(blk_addr + qs_offset + 4);

            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            uint grid_idx = (aux0 >> (l * 8)) & 0xFF;
            uint signs_idx = (aux1 >> (7 * l)) & 127;

            uint s_word = ksigns_iq2xs[signs_idx / 4];
            uint signs = (s_word >> ((signs_idx % 4) * 8)) & 0xFF;

            uint2 g = iq2xxs_grid[grid_idx];
            float g0 = (float)(g.x & 0xFF);
            float g1 = (float)((g.x >> 8) & 0xFF);
            float g2 = (float)((g.x >> 16) & 0xFF);
            float g3 = (float)(g.x >> 24);

            float g4 = (float)(g.y & 0xFF);
            float g5 = (float)((g.y >> 8) & 0xFF);
            float g6 = (float)((g.y >> 16) & 0xFF);
            float g7 = (float)(g.y >> 24);

            uint x_addr = (b * 256 + x_offset) * 4;
            float4 x0 = asfloat(x.Load4(x_addr));
            float4 x1 = asfloat(x.Load4(x_addr + 16));

            float dot = 0.0f;
            dot += x0.x * (g0 * ((signs & 1) ? -1.0f : 1.0f));
            dot += x0.y * (g1 * ((signs & 2) ? -1.0f : 1.0f));
            dot += x0.z * (g2 * ((signs & 4) ? -1.0f : 1.0f));
            dot += x0.w * (g3 * ((signs & 8) ? -1.0f : 1.0f));

            dot += x1.x * (g4 * ((signs & 16) ? -1.0f : 1.0f));
            dot += x1.y * (g5 * ((signs & 32) ? -1.0f : 1.0f));
            dot += x1.z * (g6 * ((signs & 64) ? -1.0f : 1.0f));
            dot += x1.w * (g7 * ((signs & 128) ? -1.0f : 1.0f));

            row_sum += db * dot;
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

    public const string GemmIQ2_XXSBatch = @"
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

groupshared float s_mem[4096];
" + IQ2_XXS_Tables + @"
[numthreads(32, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row_in_grp = gtid.y;
    uint warp_id = gid.x * 4 + row_in_grp;
    uint lane_id = gtid.x;
    uint s_idx = row_in_grp * 32 + lane_id;
    uint t_base = gid.y * 32;

    float acc[32];
    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        acc[i] = 0.0f;
    }

    if (warp_id < m_rows)
    {
        uint nb = k_cols / 256;
        uint row_byte_offset = warp_id * nb * 68;

        uint ib32 = lane_id / 4;
        uint l = lane_id % 4;
        uint qs_offset = 4 + ib32 * 8;
        uint x_offset = lane_id * 8;

        for (uint b = 0; b < nb; b++)
        {
            uint blk_addr = row_byte_offset + b * 68;
            uint d_raw = W.Load(blk_addr) & 0xFFFF;
            float d = f16tof32(d_raw);

            uint aux0 = W.Load(blk_addr + qs_offset);
            uint aux1 = W.Load(blk_addr + qs_offset + 4);

            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            uint grid_idx = (aux0 >> (l * 8)) & 0xFF;
            uint signs_idx = (aux1 >> (7 * l)) & 127;

            uint s_word = ksigns_iq2xs[signs_idx / 4];
            uint signs = (s_word >> ((signs_idx % 4) * 8)) & 0xFF;

            uint2 g = iq2xxs_grid[grid_idx];
            float g0 = (float)(g.x & 0xFF);
            float g1 = (float)((g.x >> 8) & 0xFF);
            float g2 = (float)((g.x >> 16) & 0xFF);
            float g3 = (float)(g.x >> 24);

            float g4 = (float)(g.y & 0xFF);
            float g5 = (float)((g.y >> 8) & 0xFF);
            float g6 = (float)((g.y >> 16) & 0xFF);
            float g7 = (float)(g.y >> 24);

            float w0 = db * (g0 * ((signs & 1) ? -1.0f : 1.0f));
            float w1 = db * (g1 * ((signs & 2) ? -1.0f : 1.0f));
            float w2 = db * (g2 * ((signs & 4) ? -1.0f : 1.0f));
            float w3 = db * (g3 * ((signs & 8) ? -1.0f : 1.0f));
            float w4 = db * (g4 * ((signs & 16) ? -1.0f : 1.0f));
            float w5 = db * (g5 * ((signs & 32) ? -1.0f : 1.0f));
            float w6 = db * (g6 * ((signs & 64) ? -1.0f : 1.0f));
            float w7 = db * (g7 * ((signs & 128) ? -1.0f : 1.0f));

            uint col_base = b * 256 + x_offset;

            [unroll]
            for (uint i = 0; i < 32; i++)
            {
                uint t = t_base + i;
                if (t < batch_size)
                {
                    uint t_x_byte = (t * k_cols + col_base) * 4;
                    acc[i] += w0 * asfloat(x.Load(t_x_byte + 0))
                            + w1 * asfloat(x.Load(t_x_byte + 4))
                            + w2 * asfloat(x.Load(t_x_byte + 8))
                            + w3 * asfloat(x.Load(t_x_byte + 12))
                            + w4 * asfloat(x.Load(t_x_byte + 16))
                            + w5 * asfloat(x.Load(t_x_byte + 20))
                            + w6 * asfloat(x.Load(t_x_byte + 24))
                            + w7 * asfloat(x.Load(t_x_byte + 28));
                }
            }
        }
    }

    [unroll]
    for (uint i = 0; i < 32; i++)
    {
        uint t = t_base + i;
        s_mem[i * 128 + s_idx] = (t < batch_size && warp_id < m_rows) ? acc[i] : 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 16)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 16];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 8)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 8];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 4)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 4];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 2)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 2];
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane_id < 1)
    {
        [unroll]
        for (uint i = 0; i < 32; i++) s_mem[i * 128 + s_idx] += s_mem[i * 128 + s_idx + 1];
    }

    if (lane_id == 0 && warp_id < m_rows)
    {
        float b_val = 0.0f;
        if (has_bias != 0) b_val = asfloat(bias.Load(warp_id * 4));
        [unroll]
        for (uint i = 0; i < 32; i++)
        {
            uint t = t_base + i;
            if (t < batch_size)
            {
                float final_sum = s_mem[i * 128 + row_in_grp * 32] + b_val;
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
    }
}
";
}

