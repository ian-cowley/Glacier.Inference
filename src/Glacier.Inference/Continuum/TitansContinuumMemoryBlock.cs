namespace Glacier.Inference.Continuum;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// High-performance 64-byte aligned unmanaged HOPE / Titans continuum associative memory module.
/// Implements linear O(1) per-token coupled momentum-memory recurrence with AVX-512 and AVX2 SIMD acceleration,
/// spectral stability enforcement, and exact reverse-mode adjoint gradient backpropagation.
/// </summary>
public sealed unsafe class TitansContinuumMemoryBlock : IContinuumMemoryModule
{
    private const nuint CacheLineAlignment = 64;

    private readonly int _dimension;
    private readonly long _matrixBytes;
    private readonly long _paddedMatrixBytes;
    private readonly long _scratchBytes;
    private readonly long _stateSnapshotBytes;
    private readonly long _totalBytes;

    private byte* _buffer;
    private AssociativeMemoryBlockHeader* _header;
    private float* _mPtr;
    private float* _sPtr;
    private float* _adjMPtr;
    private float* _adjSPtr;
    private float* _scratchRPtr;
    private float* _scratchDrPtr;
    private bool _disposed;

    public ContinuumFrequency Frequency
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->Frequency;
        }
    }

    public ContinuumStateFlags StateFlags
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->StateFlags;
        }
        set
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _header->StateFlags = value;
        }
    }

    public int Dimension => _dimension;

    public int ClockStride
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->ClockStride;
        }
    }

    public long SequenceTokenClock
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->SequenceTokenClock;
        }
    }

    public float RetentionRateLambda
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->RetentionRateLambda;
        }
        set
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _header->RetentionRateLambda = Math.Clamp(value, 0.0f, 1.0f);
        }
    }

    public float MomentumDecayTheta
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->MomentumDecayTheta;
        }
        set
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _header->MomentumDecayTheta = Math.Clamp(value, 0.0f, 0.999999f);
        }
    }

    public float LearningRateEta
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->LearningRateEta;
        }
        set
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _header->LearningRateEta = value;
        }
    }

    public float LastSurpriseMetric
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _header->LastSurpriseMetric;
        }
    }

    public long TotalUnmanagedBytes => _totalBytes;

    /// <summary>
    /// Gets the byte size required for <see cref="SaveState"/> and <see cref="RestoreState"/> (Header + M + S).
    /// </summary>
    public long StateSnapshotBytes => _stateSnapshotBytes;

    public TitansContinuumMemoryBlock(
        int dimension,
        ContinuumFrequency frequency = ContinuumFrequency.PerToken,
        int clockStride = 1,
        float retentionRateLambda = 0.995f,
        float momentumDecayTheta = 0.50f,
        float learningRateEta = 0.25f)
    {
        if (dimension <= 0 || dimension > ushort.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Dimension must be between 1 and 65535.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clockStride);
        if (retentionRateLambda < 0.0f || retentionRateLambda > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(retentionRateLambda), retentionRateLambda, "Retention rate lambda must be in [0, 1].");
        }
        if (momentumDecayTheta < 0.0f || momentumDecayTheta >= 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(momentumDecayTheta), momentumDecayTheta, "Momentum decay theta must be in [0, 1).");
        }
        ArgumentOutOfRangeException.ThrowIfNegative(learningRateEta);

        _dimension = dimension;
        _matrixBytes = (long)dimension * dimension * sizeof(float);
        _paddedMatrixBytes = AlignUp(_matrixBytes, (long)CacheLineAlignment);
        _scratchBytes = AlignUp((long)dimension * sizeof(float), (long)CacheLineAlignment);

        long headerBytes = sizeof(AssociativeMemoryBlockHeader); // 64 bytes
        _stateSnapshotBytes = headerBytes + (2 * _paddedMatrixBytes);
        _totalBytes = headerBytes + (4 * _paddedMatrixBytes) + (2 * _scratchBytes);

        _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)_totalBytes, CacheLineAlignment);
        NativeMemory.Clear(_buffer, (nuint)_totalBytes);

        _header = (AssociativeMemoryBlockHeader*)_buffer;
        _mPtr = (float*)(_buffer + headerBytes);
        _sPtr = (float*)(_buffer + headerBytes + _paddedMatrixBytes);
        _adjMPtr = (float*)(_buffer + headerBytes + (2 * _paddedMatrixBytes));
        _adjSPtr = (float*)(_buffer + headerBytes + (3 * _paddedMatrixBytes));
        _scratchRPtr = (float*)(_buffer + headerBytes + (4 * _paddedMatrixBytes));
        _scratchDrPtr = (float*)(_buffer + headerBytes + (4 * _paddedMatrixBytes) + _scratchBytes);

        _header->Frequency = frequency;
        _header->StateFlags = ContinuumStateFlags.Active;
        _header->Dimension = (ushort)dimension;
        _header->ClockStride = clockStride;
        _header->SequenceTokenClock = 0;
        _header->RetentionRateLambda = retentionRateLambda;
        _header->MomentumDecayTheta = momentumDecayTheta;
        _header->LearningRateEta = learningRateEta;
        _header->LastSurpriseMetric = 0.0f;
        _header->MemoryPointer = (ulong)_mPtr;
        _header->MomentumPointer = (ulong)_sPtr;
        _header->AdjointMemoryPointer = (ulong)_adjMPtr;
        _header->AdjointMomentumPointer = (ulong)_adjSPtr;
    }

    /// <summary>
    /// Gets a read-only span view over the current <c>[Dimension x Dimension]</c> associative memory matrix <c>M_t</c>.
    /// </summary>
    public ReadOnlySpan<float> GetMemoryMatrixSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<float>(_mPtr, _dimension * _dimension);
    }

    /// <summary>
    /// Gets a read-only span view over the current <c>[Dimension x Dimension]</c> momentum matrix <c>S_t</c>.
    /// </summary>
    public ReadOnlySpan<float> GetMomentumMatrixSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<float>(_sPtr, _dimension * _dimension);
    }

    /// <summary>
    /// Evaluates <c>output = M_t * query</c> using tiered AVX-512 / AVX2 SIMD matrix-vector multiplication.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<float> query, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (query.Length < _dimension)
        {
            throw new ArgumentException($"Query length ({query.Length}) must be at least Dimension ({_dimension}).", nameof(query));
        }
        if (output.Length < _dimension)
        {
            throw new ArgumentException($"Output length ({output.Length}) must be at least Dimension ({_dimension}).", nameof(output));
        }

        fixed (float* pQ = query)
        fixed (float* pOut = output)
        {
            MatVecMulSimd(_mPtr, pQ, pOut, _dimension);
        }
    }

    /// <summary>
    /// Updates <c>S_t</c> and <c>M_t</c> via the coupled Titans associative recurrence with spectral stability clamping.
    /// Returns the L2 surprise metric <c>||M_{t-1} k_t - v_t||_2</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public float UpdateAssociativeState(
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        float? lambdaOverride = null,
        float? thetaOverride = null,
        float? etaOverride = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (key.Length < _dimension)
        {
            throw new ArgumentException($"Key length ({key.Length}) must be at least Dimension ({_dimension}).", nameof(key));
        }
        if (value.Length < _dimension)
        {
            throw new ArgumentException($"Value length ({value.Length}) must be at least Dimension ({_dimension}).", nameof(value));
        }

        if ((_header->StateFlags & ContinuumStateFlags.Frozen) != 0)
        {
            return 0.0f;
        }

        float lambda = Math.Clamp(lambdaOverride ?? _header->RetentionRateLambda, 0.0f, 1.0f);
        float theta = Math.Clamp(thetaOverride ?? _header->MomentumDecayTheta, 0.0f, 0.999999f);
        float eta = Math.Max(0.0f, etaOverride ?? _header->LearningRateEta);

        fixed (float* pK = key)
        fixed (float* pV = value)
        {
            // Enforce Jury Spectral Stability Theorem: 0 <= eta * ||k||^2 < (1 + lambda) * (1 + theta)
            float keyNormSq = DotProductSimd(pK, pK, _dimension);
            if (keyNormSq > 1e-12f)
            {
                float maxStableEta = 0.999f * (1.0f + lambda) * (1.0f + theta) / keyNormSq;
                if (eta > maxStableEta)
                {
                    eta = maxStableEta;
                }
            }

            // 1. Compute prediction residual r_t = M_{t-1} * k_t - v_t and L2 surprise ||r_t||_2
            MatVecMulSimd(_mPtr, pK, _scratchRPtr, _dimension);
            float surpriseSq = SubtractAndNormSqSimd(_scratchRPtr, pV, _dimension);
            float surprise = MathF.Sqrt(surpriseSq);

            // 2. Coupled Momentum & Memory Update:
            //    S_t = theta * S_{t-1} - eta * r_t * k_t^T
            //    M_t = lambda * M_{t-1} + S_t
            Rank1CoupledUpdateSimd(_mPtr, _sPtr, _scratchRPtr, pK, _dimension, lambda, theta, eta);

            _header->LastSurpriseMetric = surprise;
            _header->SequenceTokenClock++;
            return surprise;
        }
    }

    /// <summary>
    /// Executes one reverse-mode adjoint backpropagation step through the coupled Titans recurrence.
    /// Given the pre-step matrices <paramref name="prevMemoryM"/> (<c>M_{t-1}</c>) and <paramref name="prevMomentumS"/> (<c>S_{t-1}</c>),
    /// post-step memory <paramref name="postMemoryM"/> (<c>M_t</c>), inputs (<paramref name="key"/>, <paramref name="value"/>, <paramref name="query"/>),
    /// and upstream output gradient <paramref name="dOutput"/> (<c>dL/dy_t</c>), propagates internal adjoint states
    /// <c>AdjointM</c> and <c>AdjointS</c> backward to <c>t-1</c> and computes parameter gradients.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void BackwardStep(
        ReadOnlySpan<float> prevMemoryM,
        ReadOnlySpan<float> prevMomentumS,
        ReadOnlySpan<float> postMemoryM,
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> dOutput,
        Span<float> dKey,
        Span<float> dValue,
        Span<float> dQuery,
        out float dLambda,
        out float dTheta,
        out float dEta,
        float? lambdaOverride = null,
        float? thetaOverride = null,
        float? etaOverride = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int dim = _dimension;
        int matLen = dim * dim;

        if (prevMemoryM.Length < matLen) throw new ArgumentException("prevMemoryM too small.", nameof(prevMemoryM));
        if (prevMomentumS.Length < matLen) throw new ArgumentException("prevMomentumS too small.", nameof(prevMomentumS));
        if (postMemoryM.Length < matLen) throw new ArgumentException("postMemoryM too small.", nameof(postMemoryM));
        if (key.Length < dim || value.Length < dim || query.Length < dim || dOutput.Length < dim)
            throw new ArgumentException("Input vector length smaller than Dimension.");
        if (dKey.Length < dim || dValue.Length < dim || dQuery.Length < dim)
            throw new ArgumentException("Gradient output vector length smaller than Dimension.");

        float lambda = Math.Clamp(lambdaOverride ?? _header->RetentionRateLambda, 0.0f, 1.0f);
        float theta = Math.Clamp(thetaOverride ?? _header->MomentumDecayTheta, 0.0f, 0.999999f);
        float eta = Math.Max(0.0f, etaOverride ?? _header->LearningRateEta);

        fixed (float* pPrevM = prevMemoryM)
        fixed (float* pPrevS = prevMomentumS)
        fixed (float* pPostM = postMemoryM)
        fixed (float* pK = key)
        fixed (float* pV = value)
        fixed (float* pQ = query)
        fixed (float* pDy = dOutput)
        fixed (float* pDk = dKey)
        fixed (float* pDv = dValue)
        fixed (float* pDq = dQuery)
        {
            // Recompute r_t = M_{t-1} * k_t - v_t
            MatVecMulSimd(pPrevM, pK, _scratchRPtr, dim);
            for (int i = 0; i < dim; i++)
            {
                _scratchRPtr[i] -= pV[i];
            }

            // 1. dQuery = M_t^T * dOutput, and accumulate dOutput * q_t^T into AdjointM (tilde{M}_t)
            for (int j = 0; j < dim; j++) pDq[j] = 0.0f;
            for (int i = 0; i < dim; i++)
            {
                float dyi = pDy[i];
                float* adjMRow = _adjMPtr + (i * dim);
                float* postMRow = pPostM + (i * dim);
                for (int j = 0; j < dim; j++)
                {
                    adjMRow[j] += dyi * pQ[j];
                    pDq[j] += postMRow[j] * dyi;
                }
            }

            // 2. Compute scalar gate gradients and dr_t = -eta * G_t * k_t where G_t = tilde{M}_t + AdjointS
            double accDLam = 0.0;
            double accDTheta = 0.0;
            double accDEta = 0.0;

            for (int i = 0; i < dim; i++)
            {
                float ri = _scratchRPtr[i];
                float* adjMRow = _adjMPtr + (i * dim);
                float* adjSRow = _adjSPtr + (i * dim);
                float* prevMRow = pPrevM + (i * dim);
                float* prevSRow = pPrevS + (i * dim);

                double rowGk = 0.0;
                for (int j = 0; j < dim; j++)
                {
                    float gij = adjMRow[j] + adjSRow[j];
                    accDLam += (double)adjMRow[j] * prevMRow[j];
                    accDTheta += (double)gij * prevSRow[j];
                    accDEta -= (double)gij * ri * pK[j];
                    rowGk += (double)gij * pK[j];
                }

                float dri = (float)(-eta * rowGk);
                _scratchDrPtr[i] = dri;
                pDv[i] = -dri;
            }

            dLambda = (float)accDLam;
            dTheta = (float)accDTheta;
            dEta = (float)accDEta;

            // 3. dKey_j = -eta * (G_t^T * r_t)_j + (M_{t-1}^T * dr_t)_j
            for (int j = 0; j < dim; j++)
            {
                double sumGtR = 0.0;
                double sumMtDr = 0.0;
                for (int i = 0; i < dim; i++)
                {
                    int idx = (i * dim) + j;
                    float gij = _adjMPtr[idx] + _adjSPtr[idx];
                    sumGtR += (double)gij * _scratchRPtr[i];
                    sumMtDr += (double)pPrevM[idx] * _scratchDrPtr[i];
                }
                pDk[j] = (float)((-eta * sumGtR) + sumMtDr);
            }

            // 4. Propagate AdjointM and AdjointS to step t-1:
            //    AdjointM_{t-1} = lambda * tilde{M}_t + dr_t * k_t^T
            //    AdjointS_{t-1} = theta * G_t
            for (int i = 0; i < dim; i++)
            {
                float dri = _scratchDrPtr[i];
                float* adjMRow = _adjMPtr + (i * dim);
                float* adjSRow = _adjSPtr + (i * dim);
                for (int j = 0; j < dim; j++)
                {
                    float tildeM = adjMRow[j];
                    float gij = tildeM + adjSRow[j];
                    adjMRow[j] = (lambda * tildeM) + (dri * pK[j]);
                    adjSRow[j] = theta * gij;
                }
            }
        }
    }

    /// <summary>
    /// Clears only the reverse-mode adjoint gradient accumulators (<c>AdjointM</c> and <c>AdjointS</c>).
    /// </summary>
    public void ResetAdjointState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        NativeMemory.Clear(_adjMPtr, (nuint)(2 * _paddedMatrixBytes));
    }

    /// <summary>
    /// Synaptic Pruning: Clears volatile associative memory <c>M_t</c>, momentum <c>S_t</c>, and adjoint states to zero.
    /// </summary>
    public void ResetVolatileState()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        long bodyBytes = _totalBytes - sizeof(AssociativeMemoryBlockHeader);
        NativeMemory.Clear(_buffer + sizeof(AssociativeMemoryBlockHeader), (nuint)bodyBytes);
        _header->SequenceTokenClock = 0;
        _header->LastSurpriseMetric = 0.0f;
    }

    public void SaveState(Span<byte> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < _stateSnapshotBytes)
        {
            throw new ArgumentException($"Destination buffer ({destination.Length} bytes) is smaller than StateSnapshotBytes ({_stateSnapshotBytes} bytes).", nameof(destination));
        }

        fixed (byte* dstPtr = destination)
        {
            Buffer.MemoryCopy(_buffer, dstPtr, destination.Length, _stateSnapshotBytes);
        }
    }

    public void RestoreState(ReadOnlySpan<byte> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (source.Length < _stateSnapshotBytes)
        {
            throw new ArgumentException($"Source buffer ({source.Length} bytes) is smaller than StateSnapshotBytes ({_stateSnapshotBytes} bytes).", nameof(source));
        }

        fixed (byte* srcPtr = source)
        {
            var srcHeader = (AssociativeMemoryBlockHeader*)srcPtr;
            if (srcHeader->Dimension != _dimension)
            {
                throw new InvalidOperationException($"Snapshot dimension ({srcHeader->Dimension}) does not match block dimension ({_dimension}).");
            }

            Buffer.MemoryCopy(srcPtr, _buffer, _stateSnapshotBytes, _stateSnapshotBytes);
            // Rebind internal pointers to current unmanaged allocation
            _header->MemoryPointer = (ulong)_mPtr;
            _header->MomentumPointer = (ulong)_sPtr;
            _header->AdjointMemoryPointer = (ulong)_adjMPtr;
            _header->AdjointMomentumPointer = (ulong)_adjSPtr;
        }
    }

    // =========================================================================
    // LOW-LEVEL SIMD KERNELS (AVX-512F -> AVX2/FMA -> Scalar Fallback)
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MatVecMulSimd(float* mat, float* vec, float* dst, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            float* row = mat + (i * dim);
            int j = 0;
            float sum = 0.0f;

            if (Avx512F.IsSupported && dim >= 16)
            {
                Vector512<float> acc512 = Vector512<float>.Zero;
                for (; j <= dim - 16; j += 16)
                {
                    Vector512<float> vm = Avx512F.LoadVector512(row + j);
                    Vector512<float> vx = Avx512F.LoadVector512(vec + j);
                    acc512 = Avx512F.FusedMultiplyAdd(vm, vx, acc512);
                }
                sum += Vector512.Sum(acc512);
            }

            if (Avx2.IsSupported && j <= dim - 8)
            {
                Vector256<float> acc256 = Vector256<float>.Zero;
                for (; j <= dim - 8; j += 8)
                {
                    Vector256<float> vm = Avx.LoadVector256(row + j);
                    Vector256<float> vx = Avx.LoadVector256(vec + j);
                    acc256 = Fma.IsSupported ? Fma.MultiplyAdd(vm, vx, acc256) : Avx.Add(acc256, Avx.Multiply(vm, vx));
                }
                sum += Vector256.Sum(acc256);
            }

            for (; j < dim; j++)
            {
                sum += row[j] * vec[j];
            }

            dst[i] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(float* a, float* b, int dim)
    {
        int j = 0;
        float sum = 0.0f;

        if (Avx512F.IsSupported && dim >= 16)
        {
            Vector512<float> acc512 = Vector512<float>.Zero;
            for (; j <= dim - 16; j += 16)
            {
                Vector512<float> va = Avx512F.LoadVector512(a + j);
                Vector512<float> vb = Avx512F.LoadVector512(b + j);
                acc512 = Avx512F.FusedMultiplyAdd(va, vb, acc512);
            }
            sum += Vector512.Sum(acc512);
        }

        if (Avx2.IsSupported && j <= dim - 8)
        {
            Vector256<float> acc256 = Vector256<float>.Zero;
            for (; j <= dim - 8; j += 8)
            {
                Vector256<float> va = Avx.LoadVector256(a + j);
                Vector256<float> vb = Avx.LoadVector256(b + j);
                acc256 = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, acc256) : Avx.Add(acc256, Avx.Multiply(va, vb));
            }
            sum += Vector256.Sum(acc256);
        }

        for (; j < dim; j++)
        {
            sum += a[j] * b[j];
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float SubtractAndNormSqSimd(float* rInOut, float* v, int dim)
    {
        int j = 0;
        float normSq = 0.0f;

        if (Avx512F.IsSupported && dim >= 16)
        {
            Vector512<float> acc512 = Vector512<float>.Zero;
            for (; j <= dim - 16; j += 16)
            {
                Vector512<float> vr = Avx512F.LoadVector512(rInOut + j);
                Vector512<float> vv = Avx512F.LoadVector512(v + j);
                Vector512<float> diff = Avx512F.Subtract(vr, vv);
                Avx512F.Store(rInOut + j, diff);
                acc512 = Avx512F.FusedMultiplyAdd(diff, diff, acc512);
            }
            normSq += Vector512.Sum(acc512);
        }

        if (Avx2.IsSupported && j <= dim - 8)
        {
            Vector256<float> acc256 = Vector256<float>.Zero;
            for (; j <= dim - 8; j += 8)
            {
                Vector256<float> vr = Avx.LoadVector256(rInOut + j);
                Vector256<float> vv = Avx.LoadVector256(v + j);
                Vector256<float> diff = Avx.Subtract(vr, vv);
                Avx.Store(rInOut + j, diff);
                acc256 = Fma.IsSupported ? Fma.MultiplyAdd(diff, diff, acc256) : Avx.Add(acc256, Avx.Multiply(diff, diff));
            }
            normSq += Vector256.Sum(acc256);
        }

        for (; j < dim; j++)
        {
            float d = rInOut[j] - v[j];
            rInOut[j] = d;
            normSq += d * d;
        }

        return normSq;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Rank1CoupledUpdateSimd(
        float* mPtr,
        float* sPtr,
        float* rPtr,
        float* kPtr,
        int dim,
        float lambda,
        float theta,
        float eta)
    {
        for (int i = 0; i < dim; i++)
        {
            float negEtaRi = -eta * rPtr[i];
            float* rowM = mPtr + (i * dim);
            float* rowS = sPtr + (i * dim);
            int j = 0;

            if (Avx512F.IsSupported && dim >= 16)
            {
                Vector512<float> vTheta512 = Vector512.Create(theta);
                Vector512<float> vLambda512 = Vector512.Create(lambda);
                Vector512<float> vNegEtaRi512 = Vector512.Create(negEtaRi);

                for (; j <= dim - 16; j += 16)
                {
                    Vector512<float> vk = Avx512F.LoadVector512(kPtr + j);
                    Vector512<float> vs = Avx512F.LoadVector512(rowS + j);
                    Vector512<float> vm = Avx512F.LoadVector512(rowM + j);

                    Vector512<float> vsNew = Avx512F.FusedMultiplyAdd(vNegEtaRi512, vk, Avx512F.Multiply(vTheta512, vs));
                    Vector512<float> vmNew = Avx512F.FusedMultiplyAdd(vLambda512, vm, vsNew);

                    Avx512F.Store(rowS + j, vsNew);
                    Avx512F.Store(rowM + j, vmNew);
                }
            }

            if (Avx2.IsSupported && j <= dim - 8)
            {
                Vector256<float> vTheta256 = Vector256.Create(theta);
                Vector256<float> vLambda256 = Vector256.Create(lambda);
                Vector256<float> vNegEtaRi256 = Vector256.Create(negEtaRi);

                for (; j <= dim - 8; j += 8)
                {
                    Vector256<float> vk = Avx.LoadVector256(kPtr + j);
                    Vector256<float> vs = Avx.LoadVector256(rowS + j);
                    Vector256<float> vm = Avx.LoadVector256(rowM + j);

                    Vector256<float> vsNew = Fma.IsSupported
                        ? Fma.MultiplyAdd(vNegEtaRi256, vk, Avx.Multiply(vTheta256, vs))
                        : Avx.Add(Avx.Multiply(vTheta256, vs), Avx.Multiply(vNegEtaRi256, vk));

                    Vector256<float> vmNew = Fma.IsSupported
                        ? Fma.MultiplyAdd(vLambda256, vm, vsNew)
                        : Avx.Add(Avx.Multiply(vLambda256, vm), vsNew);

                    Avx.Store(rowS + j, vsNew);
                    Avx.Store(rowM + j, vmNew);
                }
            }

            for (; j < dim; j++)
            {
                float sNew = (theta * rowS[j]) + (negEtaRi * kPtr[j]);
                rowS[j] = sNew;
                rowM[j] = (lambda * rowM[j]) + sNew;
            }
        }
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) & ~(alignment - 1);

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_buffer != null)
            {
                NativeMemory.AlignedFree(_buffer);
                _buffer = null;
                _header = null;
                _mPtr = null;
                _sPtr = null;
                _adjMPtr = null;
                _adjSPtr = null;
                _scratchRPtr = null;
                _scratchDrPtr = null;
            }
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    ~TitansContinuumMemoryBlock()
    {
        Dispose();
    }
}
