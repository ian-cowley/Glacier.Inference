namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    public const string RmsNorm = @"
cbuffer Params : register(b0)
{
    uint size;
    float eps;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    float local_sum = 0.0f;
    for (uint i = tid.x; i < size; i += 256)
    {
        float val = x[i];
        local_sum += val * val;
    }
    s_sum[tid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid.x < s)
        {
            s_sum[tid.x] += s_sum[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = rsqrt((s_sum[0] / (float)size) + eps);

    for (uint j = tid.x; j < size; j += 256)
    {
        dst[j] = x[j] * rms * weight[j];
    }
}
";

    public const string SwiGLU = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> gate : register(u0);
RWStructuredBuffer<float> up : register(u1);
RWStructuredBuffer<float> dst : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        float g = gate[id.x];
        float silu = g / (1.0f + exp(-g));
        dst[id.x] = silu * up[id.x];
    }
}
";

    public const string VecAdd = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> b : register(u0);
RWStructuredBuffer<float> a : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        a[id.x] += b[id.x];
    }
}
";

    public const string VecAddWeighted = @"
cbuffer Params : register(b0)
{
    uint size;
    float weight;
    uint accumulate;
};

RWStructuredBuffer<float> b : register(u0);
RWStructuredBuffer<float> a : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        if (accumulate != 0)
        {
            a[id.x] += weight * b[id.x];
        }
        else
        {
            a[id.x] = weight * b[id.x];
        }
    }
}
";

    public const string RmsNormHeads = @"
cbuffer Params : register(b0)
{
    uint head_dim;
    uint n_heads;
    float eps;
};

ByteAddressBuffer weight : register(t0);
RWStructuredBuffer<float> x : register(u0);

groupshared float s_sum[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint head_idx = gid.x;
    uint tid = gtid.x;
    if (head_idx >= n_heads) return;

    uint head_offset = head_idx * head_dim;
    float sum_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += 128)
    {
        float v = x[head_offset + i];
        sum_sq += v * v;
    }
    s_sum[tid] = sum_sq;

    GroupMemoryBarrierWithGroupSync();
    if (tid < 64) s_sum[tid] += s_sum[tid + 64];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 32) s_sum[tid] += s_sum[tid + 32];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 16) s_sum[tid] += s_sum[tid + 16];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 8)  s_sum[tid] += s_sum[tid + 8];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 4)  s_sum[tid] += s_sum[tid + 4];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 2)  s_sum[tid] += s_sum[tid + 2];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 1)  s_sum[tid] += s_sum[tid + 1];
    GroupMemoryBarrierWithGroupSync();

    float rrms = rsqrt(s_sum[0] / (float)head_dim + eps);

    for (uint j = tid; j < head_dim; j += 128)
    {
        float w = asfloat(weight.Load(j * 4));
        x[head_offset + j] = x[head_offset + j] * rrms * w;
    }
}
";

    public const string RoPE = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint pos;
    float freq_base;
    float freq_scale;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = head_dim / 2;
    uint q_half = n_heads_q * half_dim;
    uint total_half = (n_heads_q + n_heads_kv) * half_dim;

    if (id.x >= total_half) return;

    bool is_k = (id.x >= q_half);
    uint head_idx = is_k ? (id.x - q_half) / half_dim : id.x / half_dim;
    uint i = is_k ? (id.x - q_half) % half_dim : id.x % half_dim;

    float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    uint offset = head_idx * head_dim;
    if (is_k)
    {
        float v0 = k[offset + i];
        float v1 = k[offset + i + half_dim];
        k[offset + i] = v0 * cos_theta - v1 * sin_theta;
        k[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
    else
    {
        float v0 = q[offset + i];
        float v1 = q[offset + i + half_dim];
        q[offset + i] = v0 * cos_theta - v1 * sin_theta;
        q[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
}
";

    public const string KvCacheStore = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<float> k_cache : register(u2);
RWStructuredBuffer<float> v_cache : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (pos >= max_seq_len) return;
    uint total = n_heads_kv * head_dim;
    if (id.x >= total) return;

    uint h = id.x / head_dim;
    uint d = id.x % head_dim;
    uint cache_idx = (h * max_seq_len + pos) * head_dim + d;

    k_cache[cache_idx] = k[id.x];
    v_cache[cache_idx] = v[id.x];
}
";

    public const string KvCacheStoreF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<uint> k_cache : register(u2);
RWStructuredBuffer<uint> v_cache : register(u3);

[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (pos >= max_seq_len) return;
    uint half_elements = n_heads_kv * (head_dim / 2);
    uint idx = id.x;
    if (idx >= half_elements) return;

    uint h = idx / (head_dim / 2);
    uint d_pair = idx % (head_dim / 2);

    uint in_idx0 = (h * head_dim) + (d_pair * 2);
    uint in_idx1 = in_idx0 + 1;

    uint k_packed = f32tof16(k[in_idx0]) | (f32tof16(k[in_idx1]) << 16);
    uint v_packed = f32tof16(v[in_idx0]) | (f32tof16(v[in_idx1]) << 16);

    uint cache_idx = (h * max_seq_len + pos) * (head_dim / 2) + d_pair;

    k_cache[cache_idx] = k_packed;
    v_cache[cache_idx] = v_packed;
}
";

    public const string AttentionGqa = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
};

RWByteAddressBuffer q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWByteAddressBuffer attn_out : register(u3);

groupshared float s_red[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q || pos >= max_seq_len) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint q_base = h * head_dim;

    float q_reg[4];
    float acc[4];
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        q_reg[c] = (d < head_dim) ? asfloat(q.Load((q_base + d) * 4)) : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (uint t = 0; t <= pos; t++)
    {
        uint kv_base = (h_kv * max_seq_len + t) * head_dim;
        float dot_part = 0.0f;
        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint d = tid + c * 128;
            if (d < head_dim)
            {
                dot_part += q_reg[c] * k_cache[kv_base + d];
            }
        }

        s_red[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 64) s_red[tid] += s_red[tid + 64];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32) s_red[tid] += s_red[tid + 32];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16) s_red[tid] += s_red[tid + 16];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8)  s_red[tid] += s_red[tid + 8];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4)  s_red[tid] += s_red[tid + 4];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2)  s_red[tid] += s_red[tid + 2];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1)  s_red[tid] += s_red[tid + 1];
        GroupMemoryBarrierWithGroupSync();

        float total_dot = s_red[0];
        float s_t = total_dot * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint d = tid + c * 128;
            if (d < head_dim)
            {
                acc[c] = acc[c] * alpha + w_t * v_cache[kv_base + d];
            }
        }
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        if (d < head_dim)
        {
            attn_out.Store((q_base + d) * 4, asuint(acc[c] * inv_l));
        }
    }
}
";

    public const string AttentionGqaF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
};

RWByteAddressBuffer q : register(u0);
RWStructuredBuffer<uint> k_cache : register(u1);
RWStructuredBuffer<uint> v_cache : register(u2);
RWByteAddressBuffer attn_out : register(u3);

groupshared float s_red[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q || pos >= max_seq_len) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint q_base = h * head_dim;
    uint half_dim = head_dim / 2;

    float q_reg[4];
    float acc[4];
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        q_reg[c] = (d < head_dim) ? asfloat(q.Load((q_base + d) * 4)) : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    for (uint t = 0; t <= pos; t++)
    {
        uint kv_base = (h_kv * max_seq_len + t) * half_dim;
        float dot_part = 0.0f;
        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint d = tid + c * 128;
            if (d < head_dim)
            {
                uint k_packed = k_cache[kv_base + (d >> 1)];
                float k_d = ((d & 1) == 0) ? f16tof32(k_packed & 0xFFFF) : f16tof32(k_packed >> 16);
                dot_part += q_reg[c] * k_d;
            }
        }

        s_red[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 64) s_red[tid] += s_red[tid + 64];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 32) s_red[tid] += s_red[tid + 32];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 16) s_red[tid] += s_red[tid + 16];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 8)  s_red[tid] += s_red[tid + 8];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 4)  s_red[tid] += s_red[tid + 4];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 2)  s_red[tid] += s_red[tid + 2];
        GroupMemoryBarrierWithGroupSync();
        if (tid < 1)  s_red[tid] += s_red[tid + 1];
        GroupMemoryBarrierWithGroupSync();

        float total_dot = s_red[0];
        float s_t = total_dot * attn_scale;

        float m_new = max(m, s_t);
        float alpha = exp(m - m_new);
        float w_t = exp(s_t - m_new);

        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint d = tid + c * 128;
            if (d < head_dim)
            {
                uint v_packed = v_cache[kv_base + (d >> 1)];
                float v_d = ((d & 1) == 0) ? f16tof32(v_packed & 0xFFFF) : f16tof32(v_packed >> 16);
                acc[c] = acc[c] * alpha + w_t * v_d;
            }
        }
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        if (d < head_dim)
        {
            attn_out.Store((q_base + d) * 4, asuint(acc[c] * inv_l));
        }
    }
}
";

    public const string RepetitionPenalty = @"
cbuffer Params : register(b0)
{
    uint count;
    uint vocab_size;
    float penalty;
};

StructuredBuffer<int> recent_tokens : register(t0);
RWStructuredBuffer<float> logits : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= count) return;
    int tokenId = recent_tokens[id.x];
    if (tokenId >= 0 && (uint)tokenId < vocab_size)
    {
        float val = logits[tokenId];
        logits[tokenId] = (val > 0.0f) ? (val / penalty) : (val * penalty);
    }
}
";

    public const string Argmax = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> logits : register(u0);
RWStructuredBuffer<int> best_token : register(u1);
RWStructuredBuffer<float> best_logit : register(u2);

groupshared float s_val[256];
groupshared int s_idx[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    float max_val = -1e30f;
    int max_idx = 0;

    for (uint i = tid.x; i < size; i += 256)
    {
        float val = logits[i];
        if (val > max_val)
        {
            max_val = val;
            max_idx = (int)i;
        }
    }

    s_val[tid.x] = max_val;
    s_idx[tid.x] = max_idx;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid.x < s)
        {
            if (s_val[tid.x + s] > s_val[tid.x])
            {
                s_val[tid.x] = s_val[tid.x + s];
                s_idx[tid.x] = s_idx[tid.x + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0)
    {
        best_token[0] = s_idx[0];
        best_logit[0] = s_val[0];
    }
}
";

    public const string RmsNormBatch = @"
cbuffer Params : register(b0)
{
    uint size;
    float eps;
    uint batch_size;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> dst : register(u1);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint t = gid.x;
    if (t >= batch_size) return;
    uint base = t * size;

    float local_sum = 0.0f;
    for (uint i = gtid.x; i < size; i += 256)
    {
        float val = x[base + i];
        local_sum += val * val;
    }
    s_sum[gtid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (gtid.x < s)
        {
            s_sum[gtid.x] += s_sum[gtid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float rms = rsqrt((s_sum[0] / (float)size) + eps);

    for (uint j = gtid.x; j < size; j += 256)
    {
        dst[base + j] = x[base + j] * rms * weight[j];
    }
}
";

    public const string RoPEBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint start_pos;
    float freq_base;
    float freq_scale;
    uint batch_size;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = head_dim / 2;
    uint q_half = n_heads_q * half_dim;
    uint total_half_per_token = (n_heads_q + n_heads_kv) * half_dim;
    uint total_half = batch_size * total_half_per_token;

    if (id.x >= total_half) return;

    uint t = id.x / total_half_per_token;
    uint local_id = id.x % total_half_per_token;
    uint pos = start_pos + t;

    bool is_k = (local_id >= q_half);
    uint head_idx = is_k ? (local_id - q_half) / half_dim : local_id / half_dim;
    uint i = is_k ? (local_id - q_half) % half_dim : local_id % half_dim;

    float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)head_dim);
    float theta = (float)pos * freq * freq_scale;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    if (is_k)
    {
        uint offset = t * (n_heads_kv * head_dim) + head_idx * head_dim;
        float v0 = k[offset + i];
        float v1 = k[offset + i + half_dim];
        k[offset + i] = v0 * cos_theta - v1 * sin_theta;
        k[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
    else
    {
        uint offset = t * (n_heads_q * head_dim) + head_idx * head_dim;
        float v0 = q[offset + i];
        float v1 = q[offset + i + half_dim];
        q[offset + i] = v0 * cos_theta - v1 * sin_theta;
        q[offset + i + half_dim] = v0 * sin_theta + v1 * cos_theta;
    }
}
";

    /// <summary>
    /// Batched KV Cache Store kernel for 32-bit floating point (FP32) buffers.
    /// Operates with a 4-byte buffer stride (<c>RWStructuredBuffer&lt;float&gt;</c>).
    /// When executing batch prefill with this shader, the backing KV cache buffers
    /// must be allocated with 4 bytes per element to avoid buffer stride mismatch and memory overrun.
    /// For 2-byte FP16 KV cache allocations, use <see cref="KvCacheStoreBatchF16"/>.
    /// </summary>
    public const string KvCacheStoreBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    uint batch_size;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<float> k_cache : register(u2);
RWStructuredBuffer<float> v_cache : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total_per_token = n_heads_kv * head_dim;
    uint total = batch_size * total_per_token;
    if (id.x >= total) return;

    uint t = id.x / total_per_token;
    uint local_id = id.x % total_per_token;
    uint pos = start_pos + t;
    if (pos >= max_seq_len) return;

    uint h = local_id / head_dim;
    uint d = local_id % head_dim;
    uint cache_idx = (h * max_seq_len + pos) * head_dim + d;

    k_cache[cache_idx] = k[id.x];
    v_cache[cache_idx] = v[id.x];
}
";

    /// <summary>
    /// Batched KV Cache Store kernel for 16-bit half precision (FP16) buffers.
    /// Operates with a 2-byte effective element stride by packing pairs of 16-bit floats into <c>RWStructuredBuffer&lt;uint&gt;</c>.
    /// Resolves buffer stride mismatch hazard when batch prefill runs against FP16-allocated KV cache buffers.
    /// </summary>
    public const string KvCacheStoreBatchF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    uint batch_size;
};

RWStructuredBuffer<float> k : register(u0);
RWStructuredBuffer<float> v : register(u1);
RWStructuredBuffer<uint> k_cache : register(u2);
RWStructuredBuffer<uint> v_cache : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = head_dim / 2;
    uint total_per_token = n_heads_kv * half_dim;
    uint total = batch_size * total_per_token;
    if (id.x >= total) return;

    uint t = id.x / total_per_token;
    uint local_id = id.x % total_per_token;
    uint pos = start_pos + t;
    if (pos >= max_seq_len) return;

    uint h = local_id / half_dim;
    uint d_pair = local_id % half_dim;

    uint in_idx0 = t * (n_heads_kv * head_dim) + (h * head_dim) + (d_pair * 2);
    uint in_idx1 = in_idx0 + 1;

    uint k_packed = f32tof16(k[in_idx0]) | (f32tof16(k[in_idx1]) << 16);
    uint v_packed = f32tof16(v[in_idx0]) | (f32tof16(v[in_idx1]) << 16);

    uint cache_idx = (h * max_seq_len + pos) * half_dim + d_pair;

    k_cache[cache_idx] = k_packed;
    v_cache[cache_idx] = v_packed;
}
";

    /// <summary>
    /// Batched multi-head attention GQA kernel for 32-bit floating point (FP32) KV cache buffers.
    /// Operates with a 4-byte buffer stride (<c>RWStructuredBuffer&lt;float&gt;</c>).
    /// For 2-byte FP16 KV cache buffers, use <see cref="AttentionBatchF16"/>.
    /// </summary>
    public const string AttentionBatch = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    float attn_scale;
    uint batch_size;
};

RWByteAddressBuffer q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWByteAddressBuffer attn_out : register(u3);

// Tiled LDS reduction scratchpad: 4 tokens * 128 elements = 512 floats (2 KB LDS)
groupshared float s_red[4][128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint t = gid.y;
    if (h >= n_heads_q || t >= batch_size) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint pos_t = start_pos + t;
    if (pos_t >= max_seq_len) return;

    uint q_base = (t * n_heads_q + h) * head_dim;
    float q_reg[4];
    float acc[4];
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        q_reg[c] = (d < head_dim) ? asfloat(q.Load((q_base + d) * 4)) : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    uint warp_id = tid / 32;
    uint lane_id = tid % 32;

    // Process historical tokens in sequence tiles of 4
    for (uint p = 0; p <= pos_t; p += 4)
    {
        [unroll]
        for (uint i = 0; i < 4; i++)
        {
            uint curr_p = p + i;
            float dot_part = 0.0f;
            if (curr_p <= pos_t)
            {
                uint kv_base = (h_kv * max_seq_len + curr_p) * head_dim;
                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    uint d = tid + c * 128;
                    if (d < head_dim)
                    {
                        dot_part += q_reg[c] * k_cache[kv_base + d];
                    }
                }
            }
            s_red[i][tid] = dot_part;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane_id < 8)
        {
            float chunk_sum = 0.0f;
            [unroll]
            for (uint j = 0; j < 16; j++)
            {
                chunk_sum += s_red[warp_id][lane_id * 16 + j];
            }
            s_red[warp_id][lane_id] = chunk_sum;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane_id == 0)
        {
            s_red[warp_id][0] = s_red[warp_id][0] + s_red[warp_id][1] +
                                s_red[warp_id][2] + s_red[warp_id][3] +
                                s_red[warp_id][4] + s_red[warp_id][5] +
                                s_red[warp_id][6] + s_red[warp_id][7];
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 4; k++)
        {
            uint curr_p = p + k;
            if (curr_p <= pos_t)
            {
                float total_dot = s_red[k][0];
                float s_k = total_dot * attn_scale;

                float m_new = max(m, s_k);
                float alpha = exp(m - m_new);
                float w_k = exp(s_k - m_new);
                uint kv_base = (h_kv * max_seq_len + curr_p) * head_dim;

                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    uint d = tid + c * 128;
                    if (d < head_dim)
                    {
                        acc[c] = acc[c] * alpha + w_k * v_cache[kv_base + d];
                    }
                }
                l = l * alpha + w_k;
                m = m_new;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        if (d < head_dim)
        {
            attn_out.Store((q_base + d) * 4, asuint(acc[c] * inv_l));
        }
    }
}
";

    /// <summary>
    /// Batched multi-head attention GQA kernel for 16-bit half precision (FP16) KV cache buffers.
    /// Operates with a 2-byte effective element stride by unpacking pairs of half floats from <c>RWStructuredBuffer&lt;uint&gt;</c>.
    /// Resolves buffer stride mismatch hazard when batch prefill runs against FP16-allocated KV cache buffers.
    /// </summary>
    public const string AttentionBatchF16 = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint start_pos;
    float attn_scale;
    uint batch_size;
};

RWByteAddressBuffer q : register(u0);
RWStructuredBuffer<uint> k_cache : register(u1);
RWStructuredBuffer<uint> v_cache : register(u2);
RWByteAddressBuffer attn_out : register(u3);

// Tiled LDS reduction scratchpad: 4 tokens * 128 elements = 512 floats (2 KB LDS)
groupshared float s_red[4][128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    uint t = gid.y;
    if (h >= n_heads_q || t >= batch_size) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;
    uint pos_t = start_pos + t;
    if (pos_t >= max_seq_len) return;

    uint q_base = (t * n_heads_q + h) * head_dim;
    uint half_dim = head_dim / 2;
    float q_reg[4];
    float acc[4];
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        q_reg[c] = (d < head_dim) ? asfloat(q.Load((q_base + d) * 4)) : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -1e30f;
    float l = 0.0f;

    uint warp_id = tid / 32;
    uint lane_id = tid % 32;

    // Process historical tokens in sequence tiles of 4
    for (uint p = 0; p <= pos_t; p += 4)
    {
        [unroll]
        for (uint i = 0; i < 4; i++)
        {
            uint curr_p = p + i;
            float dot_part = 0.0f;
            if (curr_p <= pos_t)
            {
                uint kv_base = (h_kv * max_seq_len + curr_p) * half_dim;
                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    uint d = tid + c * 128;
                    if (d < head_dim)
                    {
                        uint pair_idx = d >> 1; uint k_packed = k_cache[kv_base + pair_idx];
                        float k_d = ((d & 1) == 0) ? f16tof32(k_packed & 0xFFFF) : f16tof32(k_packed >> 16);
                        dot_part += q_reg[c] * k_d;
                    }
                }
            }
            s_red[i][tid] = dot_part;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane_id < 8)
        {
            float chunk_sum = 0.0f;
            [unroll]
            for (uint j = 0; j < 16; j++)
            {
                chunk_sum += s_red[warp_id][lane_id * 16 + j];
            }
            s_red[warp_id][lane_id] = chunk_sum;
        }

        GroupMemoryBarrierWithGroupSync();

        if (lane_id == 0)
        {
            s_red[warp_id][0] = s_red[warp_id][0] + s_red[warp_id][1] +
                                s_red[warp_id][2] + s_red[warp_id][3] +
                                s_red[warp_id][4] + s_red[warp_id][5] +
                                s_red[warp_id][6] + s_red[warp_id][7];
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 4; k++)
        {
            uint curr_p = p + k;
            if (curr_p <= pos_t)
            {
                float total_dot = s_red[k][0];
                float s_k = total_dot * attn_scale;

                float m_new = max(m, s_k);
                float alpha = exp(m - m_new);
                float w_k = exp(s_k - m_new);
                uint kv_base = (h_kv * max_seq_len + curr_p) * half_dim;

                [unroll]
                for (uint c = 0; c < 4; c++)
                {
                    uint d = tid + c * 128;
                    if (d < head_dim)
                    {
                        uint pair_idx = d >> 1; uint v_packed = v_cache[kv_base + pair_idx];
                        float v_d = ((d & 1) == 0) ? f16tof32(v_packed & 0xFFFF) : f16tof32(v_packed >> 16);
                        acc[c] = acc[c] * alpha + w_k * v_d;
                    }
                }
                l = l * alpha + w_k;
                m = m_new;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [unroll]
    for (uint c = 0; c < 4; c++)
    {
        uint d = tid + c * 128;
        if (d < head_dim)
        {
            attn_out.Store((q_base + d) * 4, asuint(acc[c] * inv_l));
        }
    }
}
";

    public const string MoeRouterSoftmaxTopK = @"
cbuffer Params : register(b0)
{
    uint num_experts;
    uint top_k;
    uint norm_top_k;
    uint batch_size;
};

RWStructuredBuffer<float> logits : register(u0);
RWStructuredBuffer<uint> topk_indices : register(u1);
RWStructuredBuffer<float> topk_weights : register(u2);

groupshared float s_logits[256];
groupshared float s_probs[512];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint token_idx = gid.x;
    if (token_idx >= batch_size) return;
    uint tid = gtid.x;

    uint safe_experts = min(num_experts, 512u);
    uint safe_topk = min(top_k, safe_experts);
    uint logit_base = token_idx * num_experts;

    float thread_max = -1e30f;
    for (uint e = tid; e < safe_experts; e += 256)
    {
        thread_max = max(thread_max, logits[logit_base + e]);
    }
    s_logits[tid] = thread_max;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_logits[tid] = max(s_logits[tid], s_logits[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float max_logit = s_logits[0];
    GroupMemoryBarrierWithGroupSync();

    float thread_sum = 0.0f;
    for (uint e2 = tid; e2 < safe_experts; e2 += 256)
    {
        float ev = exp(logits[logit_base + e2] - max_logit);
        s_probs[e2] = ev;
        thread_sum += ev;
    }
    s_logits[tid] = thread_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s2 = 128; s2 > 0; s2 >>= 1)
    {
        if (tid < s2)
        {
            s_logits[tid] += s_logits[tid + s2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float sum_exp = s_logits[0];
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (uint e3 = tid; e3 < safe_experts; e3 += 256)
    {
        s_probs[e3] *= inv_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0)
    {
        uint out_base = token_idx * top_k;
        float sum_topk = 0.0f;

        for (uint k = 0; k < top_k; k++)
        {
            if (k < safe_topk)
            {
                float best_val = -1.0f;
                uint best_idx = 0;
                for (uint ex = 0; ex < safe_experts; ex++)
                {
                    if (s_probs[ex] > best_val)
                    {
                        best_val = s_probs[ex];
                        best_idx = ex;
                    }
                }
                topk_indices[out_base + k] = best_idx;
                topk_weights[out_base + k] = best_val;
                sum_topk += best_val;
                s_probs[best_idx] = -2.0f;
            }
            else
            {
                topk_indices[out_base + k] = 0;
                topk_weights[out_base + k] = 0.0f;
            }
        }

        if (norm_top_k != 0 && sum_topk > 0.0f)
        {
            float inv_norm = 1.0f / sum_topk;
            for (uint k2 = 0; k2 < safe_topk; k2++)
            {
                topk_weights[out_base + k2] *= inv_norm;
            }
        }
    }
}
";

    public const string MoeSharedExpertGate = @"
cbuffer Params : register(b0)
{
    uint size;
};

RWStructuredBuffer<float> weight : register(u0);
RWStructuredBuffer<float> x : register(u1);
RWStructuredBuffer<float> shexp_gate : register(u2);

groupshared float s_dot[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    float local_sum = 0.0f;
    for (uint i = tid.x; i < size; i += 256)
    {
        local_sum += x[i] * weight[i];
    }
    s_dot[tid.x] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid.x < s)
        {
            s_dot[tid.x] += s_dot[tid.x + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0)
    {
        float dot_val = s_dot[0];
        shexp_gate[0] = 1.0f / (1.0f + exp(-dot_val));
    }
}
";

    public const string VecAddWeightedFromBuffer = @"
cbuffer Params : register(b0)
{
    uint size;
    uint accumulate;
    uint weight_offset;
};

RWStructuredBuffer<float> weight_buf : register(u0);
RWStructuredBuffer<float> b : register(u1);
RWStructuredBuffer<float> a : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x < size)
    {
        float w = weight_buf[weight_offset];
        if (accumulate != 0)
        {
            a[id.x] += w * b[id.x];
        }
        else
        {
            a[id.x] = w * b[id.x];
        }
    }
}
";

    public static string MakeExpertGemvShader(string baseShader)
    {
        string shader = baseShader.Replace(
            "    uint row_offset;\r\n};",
            "    uint row_offset;\r\n    uint k_index;\r\n    uint slice_bytes;\r\n};\r\n\r\nStructuredBuffer<uint> topk_indices : register(t2);"
        ).Replace(
            "    uint row_offset;\n};",
            "    uint row_offset;\n    uint k_index;\n    uint slice_bytes;\n};\n\nStructuredBuffer<uint> topk_indices : register(t2);"
        );

        shader = shader.Replace(
            "uint row_byte_offset = warp_id *",
            "uint expert_idx = topk_indices[k_index];\r\n        uint row_byte_offset = expert_idx * slice_bytes + warp_id *"
        );
        shader = shader.Replace(
            "uint row_data_offset = warp_id * k_cols;",
            "uint expert_idx = topk_indices[k_index];\r\n        uint row_data_offset = expert_idx * (m_rows * k_cols) + warp_id * k_cols;"
        );

        return shader;
    }
}


