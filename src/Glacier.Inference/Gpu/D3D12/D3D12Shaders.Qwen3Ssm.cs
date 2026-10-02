namespace Glacier.Inference.Gpu.D3D12;

public static partial class D3D12Shaders
{
    /// <summary>
    /// Depthwise 1D causal convolution shader for Qwen 3.5 / 3.6 Gated DeltaNet.
    /// Operates over 4-step rolling history buffer per channel with SiLU activation.
    /// </summary>
    public const string Qwen3SsmConv1d = @"
cbuffer Params : register(b0)
{
    uint num_channels;
};

RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> conv_state : register(u1);
StructuredBuffer<float> conv_weight : register(t0);
RWStructuredBuffer<float> conv_out : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= num_channels) return;

    // Load state history: 3 past samples per channel
    uint state_offset = c * 3;
    float s0 = conv_state[state_offset + 0];
    float s1 = conv_state[state_offset + 1];
    float s2 = conv_state[state_offset + 2];
    float s3 = x[c]; // current sample

    // Weights: [num_channels, 4]
    uint w_offset = c * 4;
    float w0 = conv_weight[w_offset + 0];
    float w1 = conv_weight[w_offset + 1];
    float w2 = conv_weight[w_offset + 2];
    float w3 = conv_weight[w_offset + 3];

    float conv_val = s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3;

    // SiLU activation: x / (1 + exp(-x))
    float sig = 1.0f / (1.0f + exp(-conv_val));
    conv_out[c] = conv_val * sig;

    // Shift state history
    conv_state[state_offset + 0] = s1;
    conv_state[state_offset + 1] = s2;
    conv_state[state_offset + 2] = s3;
}
";

    /// <summary>
    /// L2 Normalization shader for Q and K vectors per group (16 groups, 128 elements per group).
    /// Dispatches 1 threadgroup per group (16 groups total, 128 threads per group).
    /// </summary>
    public const string Qwen3L2NormQK = @"
cbuffer Params : register(b0)
{
    uint num_groups;
    uint state_dim;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

groupshared float s_sq_q[128];
groupshared float s_sq_k[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint g = gid.x;
    if (g >= num_groups) return;

    uint tid = gtid.x;
    uint idx = g * state_dim + tid;

    float q_val = q[idx];
    float k_val = k[idx];

    s_sq_q[tid] = q_val * q_val;
    s_sq_k[tid] = k_val * k_val;
    GroupMemoryBarrierWithGroupSync();

    // Reduction in shared memory
    [unroll]
    for (uint s = 64; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sq_q[tid] += s_sq_q[tid + s];
            s_sq_k[tid] += s_sq_k[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float sum_q = s_sq_q[0];
    float sum_k = s_sq_k[0];

    float qk_scale = 1.0f / sqrt((float)state_dim);
    float inv_norm_q = sum_q > 1e-12f ? (1.0f / sqrt(sum_q + 1e-6f)) * qk_scale : 0.0f;
    float inv_norm_k = sum_k > 1e-12f ? (1.0f / sqrt(sum_k + 1e-6f)) : 0.0f;

    q[idx] = q_val * inv_norm_q;
    k[idx] = k_val * inv_norm_k;
}
";

    /// <summary>
    /// Recurrent Associative Memory (DeltaNet) state update shader for Qwen 3.5 / 3.6.
    /// Updates [128 x 128] recurrent state matrix per head on GPU and computes output vector y.
    /// Dispatches 1 threadgroup per head (32 or 48 threadgroups total, 128 threads per group).
    /// </summary>
    public const string Qwen3DeltaNetUpdate = @"
cbuffer Params : register(b0)
{
    uint num_heads;
    uint state_dim;
    uint num_groups;
    uint has_dt_bias;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);
RWStructuredBuffer<float> v : register(u2);
RWStructuredBuffer<float> dt_a : register(u3);
RWStructuredBuffer<float> beta_b : register(u4);
StructuredBuffer<float> dt_bias : register(t0);
StructuredBuffer<float> a_weight : register(t1);
RWStructuredBuffer<float> ssm_state : register(u5);
RWStructuredBuffer<float> y_out : register(u6);

groupshared float s_k[128];
groupshared float s_q[128];
groupshared float s_delta[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= num_heads) return;

    uint tid = gtid.x;
    uint heads_per_group = num_heads / num_groups;
    uint g = h / heads_per_group;

    // Load Q and K vectors for this group into shared memory
    s_q[tid] = q[g * state_dim + tid];
    s_k[tid] = k[g * state_dim + tid];
    GroupMemoryBarrierWithGroupSync();

    // Compute decay rate and learning rate beta
    float a_val = dt_a[h] + (has_dt_bias != 0 ? dt_bias[h] : 0.0f);
    float dt = a_val > 20.0f ? a_val : log(1.0f + exp(a_val));
    float a_log = a_weight[h];
    float decay = exp(a_log * dt);

    float b_val = beta_b[h];
    float beta = 1.0f / (1.0f + exp(-b_val));

    // Recurrent state matrix offset for head h: [num_heads, 128, 128]
    uint head_state_base = (h * state_dim + tid) * state_dim;

    // 1. Decay row tid and retrieve memory: kv_mem = (decay * S_{t-1}) * k
    float kv_mem = 0.0f;
    [unroll]
    for (uint j = 0; j < 128; j++)
    {
        uint idx = head_state_base + j;
        float s_val = ssm_state[idx] * decay;
        ssm_state[idx] = s_val;
        kv_mem += s_val * s_k[j];
    }

    // 2. Compute delta for element tid
    float v_val = v[h * state_dim + tid];
    s_delta[tid] = (v_val - kv_mem) * beta;
    GroupMemoryBarrierWithGroupSync();

    // 3. Update state with rank-1 update: S_t = S' + delta * k^T, then project output: y = S_t * q
    float y_dot = 0.0f;
    float d_val = s_delta[tid];
    [unroll]
    for (uint col = 0; col < 128; col++)
    {
        uint idx = head_state_base + col;
        float updated = ssm_state[idx] + d_val * s_k[col];
        ssm_state[idx] = updated;
        y_dot += updated * s_q[col];
    }

    y_out[h * state_dim + tid] = y_dot;
}
";

    /// <summary>
    /// Fused Head RMSNorm + SiLU gating on Z for Qwen 3.5 / 3.6 GDN:
    /// y = RMSNorm(y, norm_weight) * (z * sigmoid(z))
    /// Dispatches 1 threadgroup per head (128 threads per group).
    /// </summary>
    public const string Qwen3SsmGateSilu = @"
cbuffer Params : register(b0)
{
    uint num_heads;
    uint state_dim;
    float rms_eps;
    uint has_norm;
};

RWStructuredBuffer<float> y : register(u0);
RWStructuredBuffer<float> z : register(u1);
StructuredBuffer<float> norm_weight : register(t0);

groupshared float s_sq[128];

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= num_heads) return;

    uint tid = gtid.x;
    uint idx = h * state_dim + tid;

    float y_val = y[idx];
    s_sq[tid] = y_val * y_val;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 64; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sq[tid] += s_sq[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float normed_y = y_val;
    if (has_norm != 0)
    {
        float mean_sq = s_sq[0] / (float)state_dim;
        float inv_rms = rsqrt(mean_sq + rms_eps);
        normed_y = y_val * inv_rms * norm_weight[tid];
    }

    // Gating with SiLU on z: z * sigmoid(z)
    float z_val = z[idx];
    float sig_z = 1.0f / (1.0f + exp(-z_val));
    float gated_z = z_val * sig_z;

    y[idx] = normed_y * gated_z;
}
";

    /// <summary>
    /// Q-Gate split shader for Qwen 3.5 interleaved full attention layers:
    /// Splits interleaved [2 * head_dim] per head into query Q and gate QGate.
    /// </summary>
    public const string Qwen3QGateSplit = @"
cbuffer Params : register(b0)
{
    uint n_heads;
    uint head_dim;
};

RWStructuredBuffer<float> q_full : register(u0);
RWStructuredBuffer<float> q_out : register(u1);
RWStructuredBuffer<float> q_gate_out : register(u2);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint total_q = n_heads * head_dim;
    if (id.x >= total_q) return;

    uint h = id.x / head_dim;
    uint d = id.x % head_dim;

    uint q_src_idx = h * (2 * head_dim) + d;
    uint gate_src_idx = q_src_idx + head_dim;

    q_out[id.x] = q_full[q_src_idx];
    q_gate_out[id.x] = q_full[gate_src_idx];
}
";

    /// <summary>
    /// Per-head RMSNorm shader for Qwen 3.5 QK-Norm supporting head_dim up to 256.
    /// Dispatches 1 threadgroup per head (256 threads per group).
    /// </summary>
    public const string Qwen3RmsNormHeads = @"
cbuffer Params : register(b0)
{
    uint head_dim;
    uint n_heads;
    float eps;
};

StructuredBuffer<float> weight : register(t0);
RWStructuredBuffer<float> x : register(u0);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint head_idx = gid.x;
    uint tid = gtid.x;
    if (head_idx >= n_heads) return;

    uint head_offset = head_idx * head_dim;
    float val = (tid < head_dim) ? x[head_offset + tid] : 0.0f;
    s_sum[tid] = val * val;

    GroupMemoryBarrierWithGroupSync();
    if (tid < 128) s_sum[tid] += s_sum[tid + 128];
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

    if (tid < head_dim)
    {
        float w = weight[tid];
        x[head_offset + tid] = val * rrms * w;
    }
}
";

    /// <summary>
    /// Partial Rotary Position Embedding (RoPE) shader for Qwen 3.5 / 3.6.
    /// Rotates only the first rope_dim dimensions per head (e.g. 64 out of 256),
    /// leaving higher dimensions untouched.
    /// </summary>
    public const string Qwen3RoPE = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint rope_dim;
    uint pos;
    float freq_base;
    float freq_scale;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint half_dim = rope_dim / 2;
    uint q_pairs = n_heads_q * half_dim;
    uint total_pairs = (n_heads_q + n_heads_kv) * half_dim;

    if (id.x >= total_pairs) return;

    bool is_k = (id.x >= q_pairs);
    uint head_idx = is_k ? (id.x - q_pairs) / half_dim : id.x / half_dim;
    uint i = is_k ? (id.x - q_pairs) % half_dim : id.x % half_dim;

    float freq = 1.0f / pow(freq_base, (float)(2 * i) / (float)rope_dim);
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

    /// <summary>
    /// FlashAttention / Grouped Query Attention (GQA) shader supporting head_dim up to 256.
    /// Dispatches 1 threadgroup per query head (256 threads per group).
    /// </summary>
    public const string Qwen3Attention = @"
cbuffer Params : register(b0)
{
    uint n_heads_q;
    uint n_heads_kv;
    uint head_dim;
    uint max_seq_len;
    uint pos;
    float attn_scale;
};

RWStructuredBuffer<float> q : register(u0);
RWStructuredBuffer<float> k_cache : register(u1);
RWStructuredBuffer<float> v_cache : register(u2);
RWStructuredBuffer<float> attn_out : register(u3);

groupshared float s_red[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint h = gid.x;
    if (h >= n_heads_q) return;

    uint tid = gtid.x;
    uint group_size = n_heads_q / n_heads_kv;
    uint h_kv = h / group_size;

    float q_d = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;

    float m = -1e30f;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint t = 0; t <= pos; t++)
    {
        uint kv_offset = (h_kv * max_seq_len + t) * head_dim + tid;
        float k_d = (tid < head_dim) ? k_cache[kv_offset] : 0.0f;
        float v_d = (tid < head_dim) ? v_cache[kv_offset] : 0.0f;

        s_red[tid] = q_d * k_d;
        GroupMemoryBarrierWithGroupSync();

        if (tid < 128) s_red[tid] += s_red[tid + 128];
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

        acc = acc * alpha + w_t * v_d;
        l = l * alpha + w_t;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < head_dim)
    {
        attn_out[h * head_dim + tid] = (l > 0.0f) ? (acc / l) : 0.0f;
    }
}
";

    /// <summary>
    /// Post-attention Q-Gate shader: multiplies attn_out by sigmoid(q_gate).
    /// </summary>
    public const string Qwen3AttnOutGate = @"
cbuffer Params : register(b0)
{
    uint total_q;
};

RWStructuredBuffer<float> attn_out : register(u0);
RWStructuredBuffer<float> q_gate : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= total_q) return;

    float g_val = q_gate[id.x];
    float sig = 1.0f / (1.0f + exp(-g_val));
    attn_out[id.x] *= sig;
}
";
}
