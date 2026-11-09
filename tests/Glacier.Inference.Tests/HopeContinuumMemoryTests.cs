namespace Glacier.Inference.Tests;

using System;
using System.Runtime.InteropServices;
using Glacier.Inference.Continuum;
using Xunit;

public unsafe class HopeContinuumMemoryTests
{
    [Fact]
    public void AssociativeMemoryBlockHeader_Layout_IsStrictly64BytesAndCacheLineAligned()
    {
        Assert.Equal(64, sizeof(AssociativeMemoryBlockHeader));

        using var block = new TitansContinuumMemoryBlock(dimension: 64, ContinuumFrequency.PerToken, clockStride: 1);
        byte[] snapshot = new byte[block.StateSnapshotBytes];
        block.SaveState(snapshot);

        fixed (byte* ptr = snapshot)
        {
            var header = (AssociativeMemoryBlockHeader*)ptr;
            Assert.Equal(ContinuumFrequency.PerToken, header->Frequency);
            Assert.Equal(ContinuumStateFlags.Active, header->StateFlags);
            Assert.Equal(64, header->Dimension);
            Assert.Equal(1, header->ClockStride);
            Assert.Equal(0UL, header->MemoryPointer % 64UL);
            Assert.Equal(0UL, header->MomentumPointer % 64UL);
            Assert.Equal(0UL, header->AdjointMemoryPointer % 64UL);
            Assert.Equal(0UL, header->AdjointMomentumPointer % 64UL);
        }
    }

    [Fact]
    public void TitansContinuumMemoryBlock_ForwardAndUpdate_RecallsOrthogonalAssociationsAccurately()
    {
        const int dim = 32;
        using var block = new TitansContinuumMemoryBlock(
            dimension: dim,
            frequency: ContinuumFrequency.PerToken,
            clockStride: 1,
            retentionRateLambda: 1.0f,
            momentumDecayTheta: 0.0f,
            learningRateEta: 1.0f);

        Random rng = new Random(42);
        float[][] keys = CreateOrthogonalBasis(rng, dim, count: 8);
        float[][] vals = new float[8][];

        for (int i = 0; i < 8; i++)
        {
            vals[i] = CreateRandomUnitVector(rng, dim);
            float surprise = block.UpdateAssociativeState(keys[i], vals[i]);
            Assert.True(surprise > 0.9f);
        }

        Assert.Equal(8L, block.SequenceTokenClock);

        float[] recalled = new float[dim];
        for (int i = 0; i < 8; i++)
        {
            block.Forward(keys[i], recalled);
            float cosSim = CosineSimilarity(recalled, vals[i]);
            Assert.True(cosSim > 0.999f, $"Expected cosine similarity > 0.999 for association {i}, got {cosSim:F6}.");
        }
    }

    [Fact]
    public void TitansContinuumMemoryBlock_SpectralStabilityGuard_ClampsOutOfBoundsEtaAndPreventsDivergence()
    {
        const int dim = 16;
        // Pass an intentionally unstable learning rate eta = 50.0f (well above (1 + lambda)*(1 + theta) / ||k||^2)
        using var block = new TitansContinuumMemoryBlock(
            dimension: dim,
            frequency: ContinuumFrequency.PerToken,
            clockStride: 1,
            retentionRateLambda: 0.99f,
            momentumDecayTheta: 0.50f,
            learningRateEta: 50.0f);

        Random rng = new Random(99);
        float[] k = CreateRandomUnitVector(rng, dim);
        float[] v = CreateRandomUnitVector(rng, dim);
        float[] outVec = new float[dim];

        for (int step = 0; step < 500; step++)
        {
            float surprise = block.UpdateAssociativeState(k, v);
            Assert.False(float.IsNaN(surprise) || float.IsInfinity(surprise), $"Diverged at step {step}");
        }

        block.Forward(k, outVec);
        for (int d = 0; d < dim; d++)
        {
            Assert.False(float.IsNaN(outVec[d]) || float.IsInfinity(outVec[d]));
        }
    }

    [Fact]
    public void TitansContinuumMemoryBlock_BackwardStep_MatchesCentralFiniteDifferencesWithin1e4()
    {
        const int dim = 8;
        const int steps = 3;
        const float lambda = 0.94f;
        const float theta = 0.55f;
        const float eta = 0.25f;

        Random rng = new Random(1337);
        float[][] K = new float[steps][];
        float[][] V = new float[steps][];
        float[][] Q = new float[steps][];
        float[][] Targets = new float[steps][];

        for (int t = 0; t < steps; t++)
        {
            K[t] = CreateRandomUnitVector(rng, dim);
            V[t] = CreateRandomUnitVector(rng, dim);
            Q[t] = CreateRandomUnitVector(rng, dim);
            Targets[t] = CreateRandomUnitVector(rng, dim);
        }

        using var block = new TitansContinuumMemoryBlock(dim, ContinuumFrequency.PerToken, 1, lambda, theta, eta);

        float[][] snapM = new float[steps + 1][];
        float[][] snapS = new float[steps + 1][];
        float[][] outputs = new float[steps][];

        float RunForwardTrajectory(float lamOverride, float theOverride, float etaOverride)
        {
            block.ResetVolatileState();
            snapM[0] = block.GetMemoryMatrixSpan().ToArray();
            snapS[0] = block.GetMomentumMatrixSpan().ToArray();
            float totalLoss = 0.0f;

            for (int t = 0; t < steps; t++)
            {
                block.UpdateAssociativeState(K[t], V[t], lamOverride, theOverride, etaOverride);
                snapM[t + 1] = block.GetMemoryMatrixSpan().ToArray();
                snapS[t + 1] = block.GetMomentumMatrixSpan().ToArray();

                outputs[t] = new float[dim];
                block.Forward(Q[t], outputs[t]);
                for (int i = 0; i < dim; i++)
                {
                    float diff = outputs[t][i] - Targets[t][i];
                    totalLoss += 0.5f * diff * diff;
                }
            }
            return totalLoss;
        }

        RunForwardTrajectory(lambda, theta, eta);
        block.ResetAdjointState();

        float[][] gradK = new float[steps][];
        float[][] gradV = new float[steps][];
        float[][] gradQ = new float[steps][];
        float totalDLam = 0.0f, totalDThe = 0.0f, totalDEta = 0.0f;

        for (int t = steps - 1; t >= 0; t--)
        {
            gradK[t] = new float[dim];
            gradV[t] = new float[dim];
            gradQ[t] = new float[dim];
            float[] dOut = new float[dim];
            for (int i = 0; i < dim; i++) dOut[i] = outputs[t][i] - Targets[t][i];

            block.BackwardStep(
                snapM[t],
                snapS[t],
                snapM[t + 1],
                K[t],
                V[t],
                Q[t],
                dOut,
                gradK[t],
                gradV[t],
                gradQ[t],
                out float dLam,
                out float dThe,
                out float dEt,
                lambda,
                theta,
                eta);

            totalDLam += dLam;
            totalDThe += dThe;
            totalDEta += dEt;
        }

        // Optimal central finite-difference step for FP32 (eps_mach^(1/3) ~ 3e-3)
        const float eps = 2.5e-3f;
        for (int t = 0; t < steps; t++)
        {
            for (int d = 0; d < dim; d++)
            {
                float origK = K[t][d];
                K[t][d] = origK + eps;
                float lpK = RunForwardTrajectory(lambda, theta, eta);
                K[t][d] = origK - eps;
                float lmK = RunForwardTrajectory(lambda, theta, eta);
                K[t][d] = origK;
                float numK = (lpK - lmK) / (2.0f * eps);
                float absErrK = MathF.Abs(gradK[t][d] - numK);
                Assert.True(absErrK < 2e-4f, $"K[{t}][{d}] gradient mismatch: analytical={gradK[t][d]}, numerical={numK}, absErr={absErrK}");

                float origV = V[t][d];
                V[t][d] = origV + eps;
                float lpV = RunForwardTrajectory(lambda, theta, eta);
                V[t][d] = origV - eps;
                float lmV = RunForwardTrajectory(lambda, theta, eta);
                V[t][d] = origV;
                float numV = (lpV - lmV) / (2.0f * eps);
                float absErrV = MathF.Abs(gradV[t][d] - numV);
                Assert.True(absErrV < 2e-4f, $"V[{t}][{d}] gradient mismatch: analytical={gradV[t][d]}, numerical={numV}, absErr={absErrV}");
            }
        }
    }

    [Fact]
    public void TitansContinuumMemoryBlock_NonMultipleOf16Dimension_ExercisesAvx512Avx2AndScalarTailCleanly()
    {
        // Dimension 29 exercises 1x16 (AVX-512) + 1x8 (AVX2) + 5 scalar tail elements
        const int dim = 29;
        using var block = new TitansContinuumMemoryBlock(
            dimension: dim,
            frequency: ContinuumFrequency.ShortWindow,
            clockStride: 64,
            retentionRateLambda: 1.0f,
            momentumDecayTheta: 0.0f,
            learningRateEta: 1.0f);

        Random rng = new Random(77);
        float[] k = CreateRandomUnitVector(rng, dim);
        float[] v = CreateRandomUnitVector(rng, dim);
        float[] output = new float[dim];

        block.UpdateAssociativeState(k, v);
        block.Forward(k, output);

        float sim = CosineSimilarity(output, v);
        Assert.True(sim > 0.9999f, $"Expected cosine similarity > 0.9999 for dim=29, got {sim:F6}");
    }

    [Fact]
    public void TitansContinuumMemoryBlock_FrozenFlag_SuppressesStateUpdates()
    {
        const int dim = 16;
        using var block = new TitansContinuumMemoryBlock(dim);
        block.StateFlags |= ContinuumStateFlags.Frozen;

        float[] k = new float[dim];
        float[] v = new float[dim];
        k[0] = 1.0f;
        v[0] = 1.0f;

        float surprise = block.UpdateAssociativeState(k, v);
        Assert.Equal(0.0f, surprise);
        Assert.Equal(0L, block.SequenceTokenClock);
    }

    [Fact]
    public void HopeContinuumHierarchy_MultiFrequencyRetention_OutperformsSingleFrequencyOverLongStream()
    {
        const int dim = 32;
        const int totalSteps = 10_000;
        Random rng = new Random(2026);

        float[] needleKey = CreateRandomUnitVector(rng, dim);
        float[] needleVal = CreateRandomUnitVector(rng, dim);

        // Distractors orthogonal to needleKey
        float[][] distractors = new float[64][];
        for (int i = 0; i < 64; i++)
        {
            float[] d = CreateRandomUnitVector(rng, dim);
            float proj = DotProduct(d, needleKey);
            for (int j = 0; j < dim; j++) d[j] -= proj * needleKey[j];
            NormalizeInPlace(d);
            distractors[i] = d;
        }

        using var singleTier = new TitansContinuumMemoryBlock(
            dim, ContinuumFrequency.PerToken, 1, retentionRateLambda: 0.995f, momentumDecayTheta: 0.4f, learningRateEta: 0.5f);
        using var hopeHierarchy = HopeContinuumHierarchy.CreateDefault4Tier(dim, surpriseThreshold: 0.65f);

        // Inject needle at step 1 with high salience
        singleTier.UpdateAssociativeState(needleKey, needleVal);
        hopeHierarchy.Step(needleKey, needleVal, explicitSurprise: 1.0f);

        // Stream 10,000 distractor tokens
        for (int t = 1; t < totalSteps; t++)
        {
            ReadOnlySpan<float> dk = distractors[t & 63];
            ReadOnlySpan<float> dv = distractors[(t + 17) & 63];
            singleTier.UpdateAssociativeState(dk, dv);
            hopeHierarchy.Step(dk, dv, explicitSurprise: 0.01f);
        }

        float[] outSingle = new float[dim];
        float[] outHope = new float[dim];
        singleTier.Forward(needleKey, outSingle);
        hopeHierarchy.Forward(needleKey, outHope);

        float simSingle = CosineSimilarity(outSingle, needleVal);
        float simHope = CosineSimilarity(outHope, needleVal);

        Assert.True(simHope > 0.95f, $"Expected HOPE hierarchy cosine similarity > 0.95 after 10k tokens, got {simHope:F4}");
        Assert.True(simHope > simSingle + 0.50f, $"Expected HOPE ({simHope:F4}) to substantially outperform single-frequency decay ({simSingle:F4})");
    }

    [Fact]
    public void HopeContinuumHierarchy_SynapticPruning_WipesVolatileTiersWhilePreservingConsolidatedTiers()
    {
        const int dim = 16;
        using var hierarchy = HopeContinuumHierarchy.CreateDefault4Tier(dim, surpriseThreshold: 0.5f);

        Random rng = new Random(123);
        float[] k = CreateRandomUnitVector(rng, dim);
        float[] v = CreateRandomUnitVector(rng, dim);

        hierarchy.Step(k, v, explicitSurprise: 1.0f);
        Assert.Equal(1L, hierarchy.GetTier(0).SequenceTokenClock);
        Assert.Equal(1L, hierarchy.GetTier(3).SequenceTokenClock);

        // Prune volatile fast tiers 0 and 1 (simulating Sleep Phase 3 Synaptic Pruning)
        hierarchy.PruneVolatileTiers(maxTierIndexToPrune: 1);

        Assert.Equal(0L, hierarchy.GetTier(0).SequenceTokenClock);
        Assert.Equal(0L, hierarchy.GetTier(1).SequenceTokenClock);
        Assert.Equal(1L, hierarchy.GetTier(2).SequenceTokenClock);
        Assert.Equal(1L, hierarchy.GetTier(3).SequenceTokenClock);

        // Verify recall still succeeds from consolidated tiers 2 and 3
        float[] output = new float[dim];
        hierarchy.Forward(k, output);
        Assert.True(CosineSimilarity(output, v) > 0.99f);
    }

    [Fact]
    public void HopeContinuumHierarchy_SaveAndRestoreState_SupportsBitExactSpeculativeRollback()
    {
        const int dim = 32;
        using var hierarchy = HopeContinuumHierarchy.CreateDefault4Tier(dim);
        Random rng = new Random(555);

        float[] k1 = CreateRandomUnitVector(rng, dim);
        float[] v1 = CreateRandomUnitVector(rng, dim);
        hierarchy.Step(k1, v1, explicitSurprise: 0.9f);

        byte[] checkpoint = new byte[hierarchy.StateSnapshotBytes];
        hierarchy.SaveState(checkpoint);

        float[] expectedBeforeDraft = new float[dim];
        hierarchy.Forward(k1, expectedBeforeDraft);

        // Mutate state with 16 rejected speculative draft tokens
        for (int i = 0; i < 16; i++)
        {
            float[] dk = CreateRandomUnitVector(rng, dim);
            float[] dv = CreateRandomUnitVector(rng, dim);
            hierarchy.Step(dk, dv, explicitSurprise: 0.9f);
        }

        Assert.Equal(17L, hierarchy.GlobalTokenClock);

        // Rollback state to checkpoint
        hierarchy.RestoreState(checkpoint);
        Assert.Equal(1L, hierarchy.GlobalTokenClock);

        float[] actualAfterRestore = new float[dim];
        hierarchy.Forward(k1, actualAfterRestore);

        for (int d = 0; d < dim; d++)
        {
            Assert.Equal(expectedBeforeDraft[d], actualAfterRestore[d]);
        }
    }

    [Fact]
    public void HopeContinuumHierarchy_HotPath_AllocatesZeroManagedBytes()
    {
        const int dim = 64;
        using var hierarchy = HopeContinuumHierarchy.CreateDefault4Tier(dim);
        float[] k = new float[dim];
        float[] v = new float[dim];
        float[] output = new float[dim];
        k[0] = 1.0f;
        v[1] = 1.0f;

        // Warmup JIT across all tier strides (stride reaches up to 256)
        for (int w = 0; w < 300; w++)
        {
            hierarchy.Step(k, v);
            hierarchy.Forward(k, output);
        }

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            hierarchy.Step(k, v);
            hierarchy.Forward(k, output);
        }
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(0L, allocAfter - allocBefore);
    }

    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = 0f;
        for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
        return dot;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = 0f, magA = 0f, magB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return (magA > 1e-12f && magB > 1e-12f) ? dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB)) : 0f;
    }

    private static float[] CreateRandomUnitVector(Random rng, int dim)
    {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        NormalizeInPlace(v);
        return v;
    }

    private static void NormalizeInPlace(Span<float> v)
    {
        float mag = 0f;
        for (int i = 0; i < v.Length; i++) mag += v[i] * v[i];
        mag = MathF.Sqrt(mag);
        if (mag > 1e-12f)
        {
            for (int i = 0; i < v.Length; i++) v[i] /= mag;
        }
    }

    private static float[][] CreateOrthogonalBasis(Random rng, int dim, int count)
    {
        float[][] basis = new float[count][];
        for (int i = 0; i < count; i++)
        {
            float[] v = CreateRandomUnitVector(rng, dim);
            for (int pass = 0; pass < 2; pass++)
            {
                for (int j = 0; j < i; j++)
                {
                    float proj = DotProduct(v, basis[j]);
                    for (int d = 0; d < dim; d++) v[d] -= proj * basis[j][d];
                }
                NormalizeInPlace(v);
            }
            basis[i] = v;
        }
        return basis;
    }
}
