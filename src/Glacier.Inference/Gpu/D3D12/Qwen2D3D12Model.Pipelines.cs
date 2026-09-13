namespace Glacier.Inference.Gpu.D3D12;

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;
using Vortice.Direct3D;
using Vortice.Direct3D12;

public sealed unsafe partial class Qwen2D3D12Model
{
    private void InitPipelines()
    {
        // 1. GEMV Root Signature: (Params b0, W t0, bias t1, x u0, residual u1, y u2)
        var gemvParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
        _psoGemvQ4K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ4K));
        _psoGemvQ5K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ5K));
        _psoGemvQ6K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ6K));
        _psoGemvQ3K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ3K));
        _psoGemvQ8_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ8_0));
        _psoGemvFp32 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvFp32));

        // 2. RMSNorm Root Signature: (Params b0, weight t0, x u0, dst u1)
        var rmsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNorm = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsParams));
        _psoRmsNorm = _ctx.CreatePipelineState(_sigRmsNorm, _ctx.CompileShader(D3D12Shaders.RmsNorm));

        // 2b. RMSNorm per-head Root Signature: (Params b0, weight t0, x u0)
        var rmsHeadsParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All)
        };
        _sigRmsNormHeads = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsHeadsParams));
        _psoRmsNormHeads = _ctx.CreatePipelineState(_sigRmsNormHeads, _ctx.CompileShader(D3D12Shaders.RmsNormHeads));

        // 3. SwiGLU Root Signature: (Params b0, gate u0, up u1, dst u2)
        var swigluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigSwiglu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, swigluParams));
        _psoSwiglu = _ctx.CreatePipelineState(_sigSwiglu, _ctx.CompileShader(D3D12Shaders.SwiGLU));

        // 4. VecAdd Root Signature: (Params b0, b u0, a u1)
        var vecAddParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAdd = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddParams));
        _psoVecAdd = _ctx.CreatePipelineState(_sigVecAdd, _ctx.CompileShader(D3D12Shaders.VecAdd));

        // 4b. VecAddWeighted Root Signature: (Params b0 (size, weight, accumulate), b u0, a u1)
        var vecAddWeightedParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAddWeighted = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddWeightedParams));
        _psoVecAddWeighted = _ctx.CreatePipelineState(_sigVecAddWeighted, _ctx.CompileShader(D3D12Shaders.VecAddWeighted));

        // 5. RoPE Root Signature: (Params b0, q u0, k u1)
        var ropeParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRope = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeParams));
        _psoRope = _ctx.CreatePipelineState(_sigRope, _ctx.CompileShader(D3D12Shaders.RoPE));

        // 6. KvStore Root Signature: (Params b0, k u0, v u1, k_cache u2, v_cache u3)
        var kvStoreParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 4), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigKvStore = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, kvStoreParams));
        _psoKvStore = _ctx.CreatePipelineState(_sigKvStore, _ctx.CompileShader(D3D12Shaders.KvCacheStore));

        // 7. Attention GQA Root Signature: (Params b0, q u0, k_cache u1, v_cache u2, attn_out u3)
        var attnParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigAttention = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnParams));
        _psoAttention = _ctx.CreatePipelineState(_sigAttention, _ctx.CompileShader(D3D12Shaders.AttentionGqa));

        // 8. Argmax Root Signature: (Params b0, logits u0, best_token u1, best_logit u2)
        var argmaxParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigArgmax = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, argmaxParams));
        _psoArgmax = _ctx.CreatePipelineState(_sigArgmax, _ctx.CompileShader(D3D12Shaders.Argmax));

        // 9. Batched GEMM Root Signature: (Params b0, W t0, bias t1, x t2, residual u0, y u1)
        var gemmBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All), // t0: W
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All), // t1: bias
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(2, 0), ShaderVisibility.All), // t2: x (SRV for L1 cache)
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All), // u0: residual
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)  // u1: y
        };

        _sigGemmBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemmBatchParams));
        _psoGemmQ4KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ4KBatch));
        _psoGemmQ6KBatch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ6KBatch));
        _psoGemmQ8_0Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmQ8_0Batch));
        _psoGemmFp32Batch = _ctx.CreatePipelineState(_sigGemmBatch, _ctx.CompileShader(D3D12Shaders.GemmFp32Batch));

        // 10. Batched RMSNorm Root Signature: (Params b0, weight t0, x u0, dst u1)
        var rmsNormBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 3), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRmsNormBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, rmsNormBatchParams));
        _psoRmsNormBatch = _ctx.CreatePipelineState(_sigRmsNormBatch, _ctx.CompileShader(D3D12Shaders.RmsNormBatch));

        // 11. Batched RoPE Root Signature: (Params b0, q u0, k u1)
        var ropeBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigRopeBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, ropeBatchParams));
        _psoRopeBatch = _ctx.CreatePipelineState(_sigRopeBatch, _ctx.CompileShader(D3D12Shaders.RoPEBatch));

        // 12. Batched KvStore Root Signature: (Params b0, k u0, v u1, k_cache u2, v_cache u3)
        var kvStoreBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 5), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigKvStoreBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, kvStoreBatchParams));
        _psoKvStoreBatch = _ctx.CreatePipelineState(_sigKvStoreBatch, _ctx.CompileShader(D3D12Shaders.KvCacheStoreBatch));

        // 13. Batched Attention Root Signature: (Params b0, q u0, k_cache u1, v_cache u2, attn_out u3)
        var attnBatchParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 7), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(3, 0), ShaderVisibility.All)
        };
        _sigAttentionBatch = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, attnBatchParams));
        _psoAttentionBatch = _ctx.CreatePipelineState(_sigAttentionBatch, _ctx.CompileShader(D3D12Shaders.AttentionBatch));
    }
}
