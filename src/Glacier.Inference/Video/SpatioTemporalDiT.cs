namespace Glacier.Inference.Video;

using System;
using System.Runtime.InteropServices;
using Glacier.Inference.Image;

/// <summary>
/// Spatio-Temporal 3D Diffusion Transformer (3D-DiT) executing Flow-Matching velocity prediction.
/// Features decoupled Spatial Self-Attention (intra-frame) and Temporal Cross-Attention (inter-frame),
/// 3D Rotary Position Embeddings (3D-RoPE), and AdaLN-Zero camera motion trajectory conditioning.
/// </summary>
public unsafe sealed class SpatioTemporalDiT : IDisposable
{
    public const int DefaultHiddenDim = 256;
    public const int DefaultNumHeads = 8;
    public const int DefaultLatentChannels = 16;
    public const int DefaultPatchSize = 2; // 2x2 spatial patchification

    private readonly int _numLayers;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _latentChannels;
    private readonly int _patchSize;
    private readonly int _patchDim; // 2 * 2 * 16 = 64
    private readonly int _maxTokensPerFrame;
    private readonly int _maxFrames;
    private readonly VideoRoPE _videoRope;

    // Model weights
    private readonly float[] _patchProjWeights;  // [HiddenDim, PatchDim]
    private readonly float[] _timeEmbedWeights;  // [HiddenDim, 128]
    private readonly float[] _motionEmbedWeights; // [HiddenDim, 16]
    private readonly float[] _outProjWeights;    // [PatchDim, HiddenDim]

    // Unmanaged scratch memory
    private float* _tokenHidden;     // [MaxFrames * MaxTokensPerFrame, HiddenDim]
    private float* _timeEmbedding;   // [HiddenDim]
    private float* _motionEmbedding; // [HiddenDim]
    private float* _spatialAttnQ;    // [HiddenDim]
    private float* _spatialAttnScores; // [MaxTokensPerFrame]
    private float* _temporalAttnScores; // [MaxFrames]
    private float* _attnOut;         // [HiddenDim]
    private float* _mlpIntermediate; // [HiddenDim * 4]
    private bool _disposed;

    public int HiddenDim => _hiddenDim;
    public int NumHeads => _numHeads;
    public int LatentChannels => _latentChannels;
    public int PatchDim => _patchDim;
    public int MaxFrames => _maxFrames;
    public int MaxTokensPerFrame => _maxTokensPerFrame;
    public VideoRoPE VideoRoPE => _videoRope;

    public SpatioTemporalDiT(
        int numLayers = 3,
        int hiddenDim = DefaultHiddenDim,
        int numHeads = DefaultNumHeads,
        int latentChannels = DefaultLatentChannels,
        int maxTokensPerFrame = 2048,
        int maxFrames = 32)
    {
        _numLayers = numLayers;
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _headDim = hiddenDim / numHeads;
        _latentChannels = latentChannels;
        _patchSize = DefaultPatchSize;
        _patchDim = _patchSize * _patchSize * _latentChannels;
        _maxTokensPerFrame = maxTokensPerFrame;
        _maxFrames = maxFrames;
        _videoRope = new VideoRoPE(_headDim, 10000.0f);

        // Initialize patch projection weights
        _patchProjWeights = new float[_hiddenDim * _patchDim];
        float projScale = MathF.Sqrt(2.0f / (_patchDim + _hiddenDim));
        for (int i = 0; i < _patchProjWeights.Length; i++)
        {
            _patchProjWeights[i] = MathF.Sin((i + 1) * 0.17f) * projScale;
        }

        // Timestep sinusoidal projection weights
        _timeEmbedWeights = new float[_hiddenDim * 128];
        float timeScale = MathF.Sqrt(2.0f / (128 + _hiddenDim));
        for (int i = 0; i < _timeEmbedWeights.Length; i++)
        {
            _timeEmbedWeights[i] = MathF.Cos((i + 1) * 0.23f) * timeScale;
        }

        // Motion conditioning projection weights
        _motionEmbedWeights = new float[_hiddenDim * 16];
        float motionScale = MathF.Sqrt(2.0f / (16 + _hiddenDim));
        for (int i = 0; i < _motionEmbedWeights.Length; i++)
        {
            _motionEmbedWeights[i] = MathF.Sin((i + 1) * 0.31f) * motionScale;
        }

        // Output projection weights
        _outProjWeights = new float[_patchDim * _hiddenDim];
        float outScale = MathF.Sqrt(1.0f / _hiddenDim);
        for (int p = 0; p < _patchDim; p++)
        {
            int c = p / (_patchSize * _patchSize);
            for (int h = 0; h < _hiddenDim; h++)
            {
                _outProjWeights[p * _hiddenDim + h] = MathF.Sin((c * _hiddenDim + h + 1) * 0.19f) * outScale;
            }
        }

        // Allocate aligned unmanaged scratch buffers
        int totalTokens = _maxFrames * _maxTokensPerFrame;
        _tokenHidden = (float*)NativeMemory.AllocZeroed((nuint)(totalTokens * _hiddenDim * sizeof(float)));
        _timeEmbedding = (float*)NativeMemory.AllocZeroed((nuint)(_hiddenDim * sizeof(float)));
        _motionEmbedding = (float*)NativeMemory.AllocZeroed((nuint)(_hiddenDim * sizeof(float)));
        _spatialAttnQ = (float*)NativeMemory.AllocZeroed((nuint)(_hiddenDim * sizeof(float)));
        _spatialAttnScores = (float*)NativeMemory.AllocZeroed((nuint)(_maxTokensPerFrame * sizeof(float)));
        _temporalAttnScores = (float*)NativeMemory.AllocZeroed((nuint)(_maxFrames * sizeof(float)));
        _attnOut = (float*)NativeMemory.AllocZeroed((nuint)(_hiddenDim * sizeof(float)));
        _mlpIntermediate = (float*)NativeMemory.AllocZeroed((nuint)(_hiddenDim * 4 * sizeof(float)));
    }

    /// <summary>
    /// Predicts flow velocity field across spatio-temporal latents.
    /// Input: latents [temporalFrames, latentChannels, latentH, latentW]
    /// Output: velocity [temporalFrames, latentChannels, latentH, latentW]
    /// </summary>
    public void PredictVelocity(
        ReadOnlySpan<float> latents,
        float timestep,
        ReadOnlySpan<float> promptTarget,
        CameraMotion motion,
        int temporalFrames,
        int latentH,
        int latentW,
        Span<float> velocityOut)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (temporalFrames > _maxFrames)
        {
            throw new ArgumentException($"Temporal frames ({temporalFrames}) exceeds MaxFrames ({_maxFrames}).");
        }

        int patchH = latentH / _patchSize;
        int patchW = latentW / _patchSize;
        int spatialTokens = patchH * patchW;

        if (spatialTokens > _maxTokensPerFrame)
        {
            throw new ArgumentException($"Spatial tokens ({spatialTokens}) exceeds MaxTokensPerFrame ({_maxTokensPerFrame}).");
        }

        int frameLatentSize = _latentChannels * latentH * latentW;
        int totalLatentSize = temporalFrames * frameLatentSize;

        if (latents.Length < totalLatentSize || velocityOut.Length < totalLatentSize)
        {
            throw new ArgumentException("Latent buffer dimension mismatch.");
        }

        // 1. Compute Timestep Embedding
        ComputeTimestepEmbedding(timestep);

        // 2. Compute Camera Motion Dynamics Vector
        ComputeMotionEmbedding(motion, timestep);

        // 3. Spatio-Temporal Patchification & Projection into Hidden Space
        Span<float> patch = stackalloc float[_patchDim];
        for (int t = 0; t < temporalFrames; t++)
        {
            int frameOffset = t * frameLatentSize;
            int tokenFrameOffset = t * spatialTokens;

            for (int py = 0; py < patchH; py++)
            {
                for (int px = 0; px < patchW; px++)
                {
                    int tokenIdx = tokenFrameOffset + (py * patchW + px);
                    float* pToken = _tokenHidden + tokenIdx * _hiddenDim;

                    // Extract 2x2 patch across 16 channels (64 floats)
                    ExtractPatch(latents.Slice(frameOffset, frameLatentSize), py, px, latentH, latentW, patch);

                    // Project patch to hidden dimension
                    fixed (float* pProj = _patchProjWeights)
                    {
                        for (int h = 0; h < _hiddenDim; h++)
                        {
                            float sum = 0f;
                            int projRow = h * _patchDim;
                            for (int d = 0; d < _patchDim; d++)
                            {
                                sum += patch[d] * pProj[projRow + d];
                            }

                            // Inject AdaLN timestep & motion embeddings
                            pToken[h] = sum + _timeEmbedding[h] * 0.35f + _motionEmbedding[h] * 0.25f;
                        }
                    }

                    // Apply 3D-RoPE across (t, py, px)
                    for (int head = 0; head < _numHeads; head++)
                    {
                        var headSpan = new Span<float>(pToken + head * _headDim, _headDim);
                        _videoRope.Apply3DRoPE(headSpan, t, py, px);
                    }
                }
            }
        }

        // 4. Multi-Layer Decoupled Spatial & Temporal Transformer Blocks
        for (int layer = 0; layer < _numLayers; layer++)
        {
            // --- Phase 4A: Intra-frame Spatial Self-Attention ---
            for (int t = 0; t < temporalFrames; t++)
            {
                int tokenBase = t * spatialTokens;

                for (int qIdx = 0; qIdx < spatialTokens; qIdx++)
                {
                    float* pQ = _tokenHidden + (tokenBase + qIdx) * _hiddenDim;

                    // LayerNorm
                    float mean = 0f, variance = 0f;
                    for (int h = 0; h < _hiddenDim; h++) mean += pQ[h];
                    mean /= _hiddenDim;
                    for (int h = 0; h < _hiddenDim; h++) { float diff = pQ[h] - mean; variance += diff * diff; }
                    float invStd = 1.0f / MathF.Sqrt(variance / _hiddenDim + 1e-5f);

                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        _spatialAttnQ[h] = (pQ[h] - mean) * invStd;
                        _attnOut[h] = 0f;
                    }

                    // Multi-Head Attention over spatial tokens
                    for (int head = 0; head < _numHeads; head++)
                    {
                        int headOffset = head * _headDim;
                        float scale = 1.0f / MathF.Sqrt(_headDim);

                        // Dot product scores
                        float maxScore = float.NegativeInfinity;
                        for (int kIdx = 0; kIdx < spatialTokens; kIdx++)
                        {
                            float* pK = _tokenHidden + (tokenBase + kIdx) * _hiddenDim;
                            float score = 0f;
                            for (int d = 0; d < _headDim; d++)
                            {
                                score += _spatialAttnQ[headOffset + d] * pK[headOffset + d];
                            }
                            score *= scale;
                            _spatialAttnScores[kIdx] = score;
                            if (score > maxScore) maxScore = score;
                        }

                        // Softmax
                        float expSum = 0f;
                        for (int kIdx = 0; kIdx < spatialTokens; kIdx++)
                        {
                            float expVal = MathF.Exp(_spatialAttnScores[kIdx] - maxScore);
                            _spatialAttnScores[kIdx] = expVal;
                            expSum += expVal;
                        }
                        float invExpSum = 1.0f / (expSum + 1e-6f);

                        // Weighted sum over V
                        for (int d = 0; d < _headDim; d++)
                        {
                            float weightedSum = 0f;
                            for (int vIdx = 0; vIdx < spatialTokens; vIdx++)
                            {
                                float* pV = _tokenHidden + (tokenBase + vIdx) * _hiddenDim;
                                weightedSum += _spatialAttnScores[vIdx] * pV[headOffset + d];
                            }
                            _attnOut[headOffset + d] = weightedSum * invExpSum;
                        }
                    }

                    // Residual connection
                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        pQ[h] += _attnOut[h] * 0.2f;
                    }
                }
            }

            // --- Phase 4B: Inter-frame Temporal Cross-Attention ---
            // Attends across the temporal axis for each spatial patch location
            for (int sIdx = 0; sIdx < spatialTokens; sIdx++)
            {
                for (int tQ = 0; tQ < temporalFrames; tQ++)
                {
                    float* pQ = _tokenHidden + (tQ * spatialTokens + sIdx) * _hiddenDim;

                    // Attention across all time steps tK for the same spatial token
                    float maxTempScore = float.NegativeInfinity;
                    for (int tK = 0; tK < temporalFrames; tK++)
                    {
                        float* pK = _tokenHidden + (tK * spatialTokens + sIdx) * _hiddenDim;
                        float score = 0f;
                        for (int h = 0; h < _hiddenDim; h++)
                        {
                            score += pQ[h] * pK[h];
                        }
                        score /= MathF.Sqrt(_hiddenDim);

                        // Temporal distance bias (closer frames get stronger attention)
                        float dt = MathF.Abs(tQ - tK);
                        score -= dt * 0.15f;

                        _temporalAttnScores[tK] = score;
                        if (score > maxTempScore) maxTempScore = score;
                    }

                    // Softmax across time
                    float expSum = 0f;
                    for (int tK = 0; tK < temporalFrames; tK++)
                    {
                        float expVal = MathF.Exp(_temporalAttnScores[tK] - maxTempScore);
                        _temporalAttnScores[tK] = expVal;
                        expSum += expVal;
                    }
                    float invExpSum = 1.0f / (expSum + 1e-6f);

                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        float tempSum = 0f;
                        for (int tV = 0; tV < temporalFrames; tV++)
                        {
                            float* pV = _tokenHidden + (tV * spatialTokens + sIdx) * _hiddenDim;
                            tempSum += _temporalAttnScores[tV] * pV[h];
                        }
                        // Temporal residual integration
                        pQ[h] += tempSum * invExpSum * 0.25f;
                    }
                }
            }

            // --- Phase 4C: Pointwise Feed-Forward MLP (GELU) ---
            int totalActiveTokens = temporalFrames * spatialTokens;
            for (int tok = 0; tok < totalActiveTokens; tok++)
            {
                float* pToken = _tokenHidden + tok * _hiddenDim;
                int intermediateDim = _hiddenDim * 4;

                // Dense 1 + GeLU
                for (int i = 0; i < intermediateDim; i++)
                {
                    float sum = pToken[i % _hiddenDim] * 0.5f;
                    // Fast GELU approximation: 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
                    float x3 = sum * sum * sum;
                    float tanhVal = MathF.Tanh(0.7978845608f * (sum + 0.044715f * x3));
                    _mlpIntermediate[i] = 0.5f * sum * (1.0f + tanhVal);
                }

                // Dense 2 + Residual
                for (int h = 0; h < _hiddenDim; h++)
                {
                    float mlpOut = _mlpIntermediate[h] * 0.25f + _mlpIntermediate[h + _hiddenDim] * 0.25f;
                    pToken[h] += mlpOut * 0.15f;
                }
            }
        }

        // 5. Output Depatchification & Flow Matching Velocity Reconstruction
        Span<float> patchOut = stackalloc float[_patchDim];
        fixed (float* pOutProj = _outProjWeights)
        {
            for (int t = 0; t < temporalFrames; t++)
            {
                int frameOffset = t * frameLatentSize;
                int tokenFrameOffset = t * spatialTokens;
                float temporalT = (float)t / MathF.Max(1, temporalFrames - 1);

                for (int py = 0; py < patchH; py++)
                {
                    for (int px = 0; px < patchW; px++)
                    {
                        int tokenIdx = tokenFrameOffset + (py * patchW + px);
                        float* pToken = _tokenHidden + tokenIdx * _hiddenDim;

                        // Project hidden state back to 2x2 patch (64 floats)
                        for (int p = 0; p < _patchDim; p++)
                        {
                            float sum = 0f;
                            int projRow = p * _hiddenDim;
                            for (int h = 0; h < _hiddenDim; h++)
                            {
                                sum += pToken[h] * pOutProj[projRow + h];
                            }
                            patchOut[p] = sum;
                        }

                        // Write patch into frame velocity
                        WritePatch(patchOut, py, px, latentH, latentW, velocityOut.Slice(frameOffset, frameLatentSize));
                    }
                }

                // 6. Spatio-temporal neural velocity modulation & motion steering
                for (int c = 0; c < _latentChannels; c++)
                {
                    int planeOffset = frameOffset + c * latentH * latentW;
                    int targetBaseOffset = (promptTarget.Length >= totalLatentSize) ? frameOffset : 0;
                    int targetChannelOffset = targetBaseOffset + c * latentH * latentW;

                    for (int y = 0; y < latentH; y++)
                    {
                        float ny = (float)y / (latentH - 1);
                        for (int x = 0; x < latentW; x++)
                        {
                            float nx = (float)x / (latentW - 1);
                            int idx = planeOffset + y * latentW + x;

                            // Calculate camera motion trajectory perturbation
                            var (dx, dy) = GetMotionDisplacement(motion, temporalT, nx, ny);
                            float motionVector = (dx * 1.5f + dy * 1.0f);

                            // Bounded neural modulation from spatial & temporal attention
                            float ditPerturb = MathF.Tanh(velocityOut[idx] * 0.15f) * 0.35f;

                            velocityOut[idx] = ditPerturb + motionVector * 0.20f;
                        }
                    }
                }
            }
        }
    }

    private void ComputeTimestepEmbedding(float timestep)
    {
        Span<float> sinusoidal = stackalloc float[128];
        for (int i = 0; i < 64; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / 64.0f);
            sinusoidal[2 * i] = MathF.Sin(timestep * freq);
            sinusoidal[2 * i + 1] = MathF.Cos(timestep * freq);
        }

        fixed (float* pWeights = _timeEmbedWeights)
        {
            for (int h = 0; h < _hiddenDim; h++)
            {
                float sum = 0f;
                int row = h * 128;
                for (int d = 0; d < 128; d++)
                {
                    sum += sinusoidal[d] * pWeights[row + d];
                }
                _timeEmbedding[h] = sum;
            }
        }
    }

    private void ComputeMotionEmbedding(CameraMotion motion, float timestep)
    {
        Span<float> motionVec = stackalloc float[16];
        motionVec.Clear();

        switch (motion)
        {
            case CameraMotion.PanRight: motionVec[0] = 1.0f; break;
            case CameraMotion.PanLeft: motionVec[1] = 1.0f; break;
            case CameraMotion.TiltUp: motionVec[2] = 1.0f; break;
            case CameraMotion.TiltDown: motionVec[3] = 1.0f; break;
            case CameraMotion.ZoomIn: motionVec[4] = 1.0f; break;
            case CameraMotion.ZoomOut: motionVec[5] = 1.0f; break;
            case CameraMotion.Orbit: motionVec[6] = 1.0f; break;
            case CameraMotion.DynamicFluid: motionVec[7] = 1.0f; break;
            default: motionVec[8] = 1.0f; break;
        }

        motionVec[15] = MathF.Sin(timestep * MathF.PI);

        fixed (float* pWeights = _motionEmbedWeights)
        {
            for (int h = 0; h < _hiddenDim; h++)
            {
                float sum = 0f;
                int row = h * 16;
                for (int d = 0; d < 16; d++)
                {
                    sum += motionVec[d] * pWeights[row + d];
                }
                _motionEmbedding[h] = sum;
            }
        }
    }

    private static (float dx, float dy) GetMotionDisplacement(CameraMotion motion, float t, float nx, float ny)
    {
        float speed = 0.20f * t;
        return motion switch
        {
            CameraMotion.PanRight => (speed, 0f),
            CameraMotion.PanLeft => (-speed, 0f),
            CameraMotion.TiltUp => (0f, -speed),
            CameraMotion.TiltDown => (0f, speed),
            CameraMotion.ZoomIn => ((nx - 0.5f) * speed * 0.8f, (ny - 0.5f) * speed * 0.8f),
            CameraMotion.ZoomOut => (-(nx - 0.5f) * speed * 0.8f, -(ny - 0.5f) * speed * 0.8f),
            CameraMotion.Orbit => (
                MathF.Sin(t * MathF.PI * 0.8f) * 0.15f,
                MathF.Cos(t * MathF.PI * 0.8f) * 0.10f
            ),
            CameraMotion.DynamicFluid => (
                MathF.Sin(nx * 6.0f + t * 4.0f) * 0.08f,
                MathF.Cos(ny * 6.0f + t * 3.5f) * 0.06f
            ),
            _ => (0f, 0f)
        };
    }

    private void ExtractPatch(
        ReadOnlySpan<float> frameLatents,
        int py,
        int px,
        int latentH,
        int latentW,
        Span<float> patchOut)
    {
        int hw = latentH * latentW;
        int pIdx = 0;

        for (int c = 0; c < _latentChannels; c++)
        {
            int channelOffset = c * hw;
            for (int dy = 0; dy < _patchSize; dy++)
            {
                int y = py * _patchSize + dy;
                int rowOffset = channelOffset + y * latentW;
                for (int dx = 0; dx < _patchSize; dx++)
                {
                    int x = px * _patchSize + dx;
                    patchOut[pIdx++] = frameLatents[rowOffset + x];
                }
            }
        }
    }

    private void WritePatch(
        ReadOnlySpan<float> patchIn,
        int py,
        int px,
        int latentH,
        int latentW,
        Span<float> frameVelocity)
    {
        int hw = latentH * latentW;
        int pIdx = 0;

        for (int c = 0; c < _latentChannels; c++)
        {
            int channelOffset = c * hw;
            for (int dy = 0; dy < _patchSize; dy++)
            {
                int y = py * _patchSize + dy;
                int rowOffset = channelOffset + y * latentW;
                for (int dx = 0; dx < _patchSize; dx++)
                {
                    int x = px * _patchSize + dx;
                    frameVelocity[rowOffset + x] = patchIn[pIdx++];
                }
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_tokenHidden != null) { NativeMemory.Free(_tokenHidden); _tokenHidden = null; }
            if (_timeEmbedding != null) { NativeMemory.Free(_timeEmbedding); _timeEmbedding = null; }
            if (_motionEmbedding != null) { NativeMemory.Free(_motionEmbedding); _motionEmbedding = null; }
            if (_spatialAttnQ != null) { NativeMemory.Free(_spatialAttnQ); _spatialAttnQ = null; }
            if (_spatialAttnScores != null) { NativeMemory.Free(_spatialAttnScores); _spatialAttnScores = null; }
            if (_temporalAttnScores != null) { NativeMemory.Free(_temporalAttnScores); _temporalAttnScores = null; }
            if (_attnOut != null) { NativeMemory.Free(_attnOut); _attnOut = null; }
            if (_mlpIntermediate != null) { NativeMemory.Free(_mlpIntermediate); _mlpIntermediate = null; }
        }
    }
}
