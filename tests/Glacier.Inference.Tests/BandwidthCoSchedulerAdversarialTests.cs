namespace Glacier.Inference.Tests;

using System;
using System.Collections.Generic;
using Glacier.Inference.Memory;
using Glacier.Inference.Scheduling;
using Xunit;

public sealed class BandwidthCoSchedulerAdversarialTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public void BoundaryCondition_M_ZeroOrNegative_ReturnsZero(int m)
    {
        var scheduler = new BandwidthCoScheduler();
        int q = scheduler.CalculateOptimalGpuExpertCount(m);
        Assert.Equal(0, q);
    }

    [Fact]
    public void BoundaryCondition_M_One_SelectsOptimalProcessor()
    {
        // Case 1: GPU faster than CPU (Bp=100, Bh=10)
        var gpuFast = new BandwidthCoScheduler(initialPcieGBps: 100.0, initialCpuGBps: 10.0, vramGBps: 500.0);
        Assert.Equal(1, gpuFast.CalculateOptimalGpuExpertCount(1));

        // Case 2: CPU faster than GPU (Bp=10, Bh=100)
        var cpuFast = new BandwidthCoScheduler(initialPcieGBps: 10.0, initialCpuGBps: 100.0, vramGBps: 500.0);
        Assert.Equal(0, cpuFast.CalculateOptimalGpuExpertCount(1));
    }

    [Fact]
    public void BoundaryCondition_M_64_ClampedAndBounded()
    {
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);
        int q = scheduler.CalculateOptimalGpuExpertCount(64);
        Assert.InRange(q, 0, 64);
        Assert.Equal(10, q); // round(64 * 0.1553) = 10
    }

    [Fact]
    public void BandwidthAsymmetry_PcieMuchGreaterThanHost_RoutesAllToGpu()
    {
        // Bp = 10,000 GB/s, Bh = 10 GB/s
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 10000.0, initialCpuGBps: 10.0, vramGBps: 10000.0);
        int m = 16;
        int q = scheduler.CalculateOptimalGpuExpertCount(m);
        Assert.Equal(m, q);
    }

    [Fact]
    public void BandwidthAsymmetry_HostMuchGreaterThanPcie_RoutesAllToCpu()
    {
        // Bh = 10,000 GB/s, Bp = 5 GB/s
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 5.0, initialCpuGBps: 10000.0, vramGBps: 300.0);
        int m = 16;
        int q = scheduler.CalculateOptimalGpuExpertCount(m);
        Assert.Equal(0, q);
    }

    [Fact]
    public void DiscreteOptimality_EmpiricalCounterExample_SelectsMinimumLatency()
    {
        // Physical parameters measured during 500MB hardware benchmark:
        // Bp = 12.51 GB/s, Bh = 35.19 GB/s, Bvram = 272.0 GB/s, m = 6
        double bp = 12.51;
        double bh = 35.19;
        double bv = 272.0;
        int m = 6;
        long blockSize = 500L * 1024 * 1024;
        double sizeGB = blockSize / (1024.0 * 1024.0 * 1024.0);

        var scheduler = new BandwidthCoScheduler(bp, bh, bv);
        int qSelected = scheduler.CalculateOptimalGpuExpertCount(m);

        // Compute exact latencies for all possible integer assignments q in [0, m]
        double alpha = (1.0 / bp) + (1.0 / bv);
        double beta = 1.0 / bh;

        int qOracle = -1;
        double minLatency = double.MaxValue;

        for (int q = 0; q <= m; q++)
        {
            double gpuTime = q * sizeGB * alpha;
            double cpuTime = (m - q) * sizeGB * beta;
            double stepTime = Math.Max(gpuTime, cpuTime);
            if (stepTime < minLatency)
            {
                minLatency = stepTime;
                qOracle = q;
            }
        }

        double selectedGpuTime = qSelected * sizeGB * alpha;
        double selectedCpuTime = (m - qSelected) * sizeGB * beta;
        double selectedStepTime = Math.Max(selectedGpuTime, selectedCpuTime);

        // Remediation verification:
        // The discrete candidate evaluation selects q = 1, matching the exact oracle minimum!
        Assert.Equal(1, qOracle);
        Assert.Equal(qOracle, qSelected);
        Assert.Equal(minLatency, selectedStepTime, precision: 6);
    }

    [Fact]
    public void Telemetry_NaNParsing_IgnoredAndPreventsPoisoning()
    {
        var scheduler = new BandwidthCoScheduler(10.0, 50.0, 200.0);
        // Passing NaN, Infinity, or zero values must be rejected without mutating bandwidth
        scheduler.RecordPcieTransfer(1024 * 1024, double.NaN);
        scheduler.RecordPcieTransfer(1024 * 1024, double.PositiveInfinity);
        scheduler.RecordPcieTransfer(1024 * 1024, 0.0);
        scheduler.RecordPcieTransfer(0, 10.0);

        scheduler.RecordCpuExecution(1024 * 1024, double.NaN);
        scheduler.RecordGpuExecution(1024 * 1024, double.NaN);

        // Bandwidth remains strictly finite and uncorrupted
        Assert.False(double.IsNaN(scheduler.PcieBandwidthGBps));
        Assert.True(double.IsFinite(scheduler.PcieBandwidthGBps));
        Assert.Equal(10.0, scheduler.PcieBandwidthGBps);

        Assert.False(double.IsNaN(scheduler.CpuBandwidthGBps));
        Assert.True(double.IsFinite(scheduler.CpuBandwidthGBps));
        Assert.Equal(50.0, scheduler.CpuBandwidthGBps);

        Assert.False(double.IsNaN(scheduler.VramBandwidthGBps));
        Assert.True(double.IsFinite(scheduler.VramBandwidthGBps));
        Assert.Equal(200.0, scheduler.VramBandwidthGBps);

        var telemetry = scheduler.GetTelemetry();
        Assert.Equal(0, telemetry.PcieSamplesRecorded);
        Assert.Equal(0, telemetry.CpuSamplesRecorded);

        // Subsequent CalculateOptimalGpuExpertCount succeeds cleanly
        int q = scheduler.CalculateOptimalGpuExpertCount(6);
        Assert.True(q >= 0 && q <= 6);
    }

    [Fact]
    public void Schedule_AccountsForResidentHits_BalancesGpuWorkload()
    {
        // If GPU already has 4 resident hits to compute in this layer,
        // it is already busy for 4 * (1/Bvram).
        // BandwidthCoScheduler.Schedule calculates q* factoring in resident load,
        // which avoids overloading the GPU and routes all 6 missing experts to CPU (q*=0).
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 12.5, initialCpuGBps: 65.0, vramGBps: 272.0);
        using var slotManager = new GpuResidentSlotManager(slotCount: 8, slotByteSize: 1024, allocateVram: false);

        // Put 4 experts in VRAM
        for (int i = 0; i < 4; i++)
        {
            slotManager.AllocateOrEvictSlot(0, i, out _, out _, out _, out _);
        }

        // Activated: [0, 1, 2, 3] (resident) + [4, 5, 6, 7, 8, 9] (missing, m=6)
        int[] activated = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var decision = scheduler.Schedule(0, activated, slotManager, 50 * 1024 * 1024);

        Assert.Equal(4, decision.GpuResidentHits.Count);
        Assert.Equal(6, decision.MissingCount);
        // Correctly offsets resident workload: q* = 0 (routing all to CPU to minimize step latency)
        Assert.Equal(0, decision.QStar);
        Assert.Empty(decision.GpuTransfers);
        Assert.Equal(6, decision.CpuExecutions.Count);
    }

    [Fact]
    public void Schedule_ClampsTransfersToAvailableUnpinnedSlots_PreventsThrashing()
    {
        // Fast GPU bandwidths where unconstrained q* = 5
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 100.0, initialCpuGBps: 10.0, vramGBps: 500.0);
        using var slotManager = new GpuResidentSlotManager(slotCount: 4, slotByteSize: 1024, allocateVram: false);

        // Slot 0 pinned
        slotManager.AllocateOrEvictSlot(0, 99, out _, out _, out _, out _);
        slotManager.PinExpert(0, 99);

        // Slot 1 and 2 resident hits for layer 0
        slotManager.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(0, 2, out _, out _, out _, out _);

        // Slot 3 empty.
        // Total available unpinned slots for transfer in layer 0: 4 - 1 pinned - 2 resident = 1 slot.
        int[] activated = new[] { 1, 2, 10, 11, 12, 13, 14 }; // 2 resident, 5 missing
        var decision = scheduler.Schedule(0, activated, slotManager, 50 * 1024 * 1024);

        // Unconstrained q* would be 5, but clamped to 1 available unpinned slot
        Assert.Equal(2, decision.GpuResidentHits.Count);
        Assert.Equal(5, decision.MissingCount);
        Assert.Equal(1, decision.QStar);
        Assert.Single(decision.GpuTransfers);
        Assert.Equal(4, decision.CpuExecutions.Count);
    }

    [Fact]
    public void Schedule_AllSlotsPinned_ClampsTransfersToZero_NoCrash()
    {
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 100.0, initialCpuGBps: 10.0, vramGBps: 500.0);
        using var slotManager = new GpuResidentSlotManager(slotCount: 2, slotByteSize: 1024, allocateVram: false);

        slotManager.AllocateOrEvictSlot(1, 0, out _, out _, out _, out _);
        slotManager.PinExpert(1, 0);
        slotManager.AllocateOrEvictSlot(1, 1, out _, out _, out _, out _);
        slotManager.PinExpert(1, 1);

        int[] activated = new[] { 5, 6, 7 }; // 3 missing
        var decision = scheduler.Schedule(0, activated, slotManager, 50 * 1024 * 1024);

        Assert.Equal(0, decision.QStar);
        Assert.Empty(decision.GpuTransfers);
        Assert.Equal(3, decision.CpuExecutions.Count);
    }

    [Fact]
    public void Schedule_HighTransferLoad_ExecutesTransfers_ResidentHitsNeverEvicted()
    {
        // Adversarial setup: GPU bandwidth is extraordinarily fast,
        // so unconstrained co-scheduling would demand transferring ALL missing experts to GPU.
        var scheduler = new BandwidthCoScheduler(initialPcieGBps: 1000.0, initialCpuGBps: 1.0, vramGBps: 2000.0);
        using var slotManager = new GpuResidentSlotManager(slotCount: 8, slotByteSize: 1024, allocateVram: false);

        int layerIdx = 3;

        // Populate slots:
        // Slots 0, 1, 2: Resident hits for layer 3 (Experts 10, 11, 12)
        slotManager.AllocateOrEvictSlot(layerIdx, 10, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(layerIdx, 11, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(layerIdx, 12, out _, out _, out _, out _);

        // Slots 3, 4: Pinned experts from another layer (must NEVER be evicted)
        slotManager.AllocateOrEvictSlot(layerIdx: 0, expertIdx: 90, out _, out _, out _, out _);
        slotManager.PinExpert(layerIdx: 0, expertIdx: 90);
        slotManager.AllocateOrEvictSlot(layerIdx: 0, expertIdx: 91, out _, out _, out _, out _);
        slotManager.PinExpert(layerIdx: 0, expertIdx: 91);

        // Slots 5, 6, 7: Old unpinned experts from layer 1 (evictable)
        slotManager.AllocateOrEvictSlot(layerIdx: 1, expertIdx: 50, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(layerIdx: 1, expertIdx: 51, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(layerIdx: 1, expertIdx: 52, out _, out _, out _, out _);

        // Available slots for layer 3 transfer: 8 total - 2 pinned - 3 resident = 3 available (slots 5, 6, 7).
        int availableSlots = slotManager.GetAvailableTransferSlots(layerIdx, new[] { 10, 11, 12 });
        Assert.Equal(3, availableSlots);

        // Activate: 3 resident hits (10, 11, 12) + 10 missing experts (100..109)
        int[] activated = new[] { 10, 11, 12, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
        var decision = scheduler.Schedule(layerIdx, activated, slotManager, 50 * 1024 * 1024);

        // Clamping assertion: Even though unconstrained q* would be 10, it MUST be clamped to 3!
        Assert.Equal(3, decision.GpuResidentHits.Count);
        Assert.Equal(10, decision.MissingCount);
        Assert.Equal(3, decision.QStar);
        Assert.Equal(3, decision.GpuTransfers.Count);
        Assert.Equal(7, decision.CpuExecutions.Count);

        // EMPIRICAL HARNESS: Actually execute all transfers through AllocateOrEvictSlot!
        foreach (int transferExp in decision.GpuTransfers)
        {
            bool evicted = slotManager.AllocateOrEvictSlot(
                layerIdx,
                transferExp,
                out int allocatedSlot,
                out int evictedL,
                out int evictedE,
                out _);

            Assert.True(evicted);
            // Verify that NEITHER of the evicted experts was a resident hit for layer 3!
            Assert.False(evictedL == layerIdx && decision.GpuResidentHits.Contains(evictedE),
                $"CRITICAL VIOLATION: Resident hit (Layer {evictedL}, Expert {evictedE}) was evicted during intra-layer transfer!");
            // Verify that pinned experts were never evicted
            Assert.False(evictedL == 0 && (evictedE == 90 || evictedE == 91),
                "CRITICAL VIOLATION: Pinned expert was evicted!");
        }

        // POST-TRANSFER RESIDENCY VERIFICATION:
        // Every single resident hit MUST still be present in VRAM!
        foreach (int hitExp in decision.GpuResidentHits)
        {
            bool stillResident = slotManager.TryGetSlot(layerIdx, hitExp, out int slot, out _);
            Assert.True(stillResident, $"Resident hit expert {hitExp} was lost from VRAM!");
        }

        // All transferred experts MUST now be resident in VRAM
        foreach (int transferExp in decision.GpuTransfers)
        {
            bool transferred = slotManager.TryGetSlot(layerIdx, transferExp, out _, out _);
            Assert.True(transferred, $"Transferred expert {transferExp} is missing from VRAM!");
        }

        // Pinned experts must still be resident
        Assert.True(slotManager.TryGetSlot(0, 90, out _, out _));
        Assert.True(slotManager.TryGetSlot(0, 91, out _, out _));
    }

    [Fact]
    public void DiscreteOptimality_MonteCarloOracle_ExhaustiveVerification()
    {
        var rng = new Random(42);
        for (int i = 0; i < 5000; i++)
        {
            double bp = rng.NextDouble() * 150.0 + 1.0;
            double bh = rng.NextDouble() * 150.0 + 1.0;
            double bv = rng.NextDouble() * 400.0 + 50.0;
            int m = rng.Next(1, 32);
            int r = rng.Next(0, 16);

            var scheduler = new BandwidthCoScheduler(bp, bh, bv);
            int qSelected = scheduler.CalculateOptimalGpuExpertCount(m, r);

            double alpha = (1.0 / bp) + (1.0 / bv);
            double beta = 1.0 / bh;
            double gamma = 1.0 / bv;

            // Exhaustive oracle over all integers q in [0, m]
            double minOracleStep = double.MaxValue;
            for (int q = 0; q <= m; q++)
            {
                double gpuTime = r * gamma + q * alpha;
                double cpuTime = (m - q) * beta;
                double stepTime = Math.Max(gpuTime, cpuTime);
                if (stepTime < minOracleStep)
                {
                    minOracleStep = stepTime;
                }
            }

            double selectedGpuTime = r * gamma + qSelected * alpha;
            double selectedCpuTime = (m - qSelected) * beta;
            double selectedStepTime = Math.Max(selectedGpuTime, selectedCpuTime);

            Assert.True(selectedStepTime <= minOracleStep + 1e-9,
                $"Discrete optimality failed: Selected {selectedStepTime:F6} vs Oracle {minOracleStep:F6} (qSelected={qSelected}, m={m}, r={r})");
        }
    }

    [Fact]
    public void SlotClamping_RandomizedStressHarness_IntraLayerEvictionInvariant()
    {
        var rng = new Random(1337);

        for (int iter = 0; iter < 100; iter++)
        {
            int slotCount = rng.Next(4, 20);
            using var slotManager = new GpuResidentSlotManager(slotCount, 1024, allocateVram: false);
            var scheduler = new BandwidthCoScheduler(
                initialPcieGBps: rng.NextDouble() * 50.0 + 5.0,
                initialCpuGBps: rng.NextDouble() * 50.0 + 5.0,
                vramGBps: rng.NextDouble() * 200.0 + 100.0);

            int layerIdx = rng.Next(0, 5);

            // Populate some resident hits
            int numHits = rng.Next(1, Math.Min(slotCount, 6));
            var hitExperts = new List<int>();
            for (int h = 0; h < numHits; h++)
            {
                int exp = 100 + h;
                slotManager.AllocateOrEvictSlot(layerIdx, exp, out _, out _, out _, out _);
                hitExperts.Add(exp);
            }

            // Pin some unrelated slots
            int numPinned = rng.Next(0, Math.Max(0, slotCount - numHits));
            for (int p = 0; p < numPinned; p++)
            {
                int pExp = 200 + p;
                slotManager.AllocateOrEvictSlot(layerIdx: 9, expertIdx: pExp, out _, out _, out _, out _);
                slotManager.PinExpert(9, pExp);
            }

            // Fill remaining with old evictable slots
            int numOld = slotCount - numHits - numPinned;
            for (int o = 0; o < numOld; o++)
            {
                slotManager.AllocateOrEvictSlot(layerIdx: 8, expertIdx: 300 + o, out _, out _, out _, out _);
            }

            // Generate activated experts (hits + missing)
            int numMissing = rng.Next(1, 15);
            var activated = new List<int>(hitExperts);
            for (int m = 0; m < numMissing; m++)
            {
                activated.Add(400 + m);
            }

            var decision = scheduler.Schedule(layerIdx, activated.ToArray(), slotManager, 50 * 1024 * 1024);

            int availableSlots = slotManager.GetAvailableTransferSlots(layerIdx, hitExperts);
            Assert.True(decision.QStar <= availableSlots,
                $"QStar ({decision.QStar}) exceeded available slots ({availableSlots})!");

            // Execute transfers
            foreach (int transferExp in decision.GpuTransfers)
            {
                slotManager.AllocateOrEvictSlot(layerIdx, transferExp, out _, out int evL, out int evE, out _);
                Assert.False(evL == layerIdx && hitExperts.Contains(evE),
                    $"Resident hit {evE} was evicted during transfer of {transferExp}!");
            }

            // Verify all resident hits remain in VRAM
            foreach (int hit in hitExperts)
            {
                Assert.True(slotManager.TryGetSlot(layerIdx, hit, out _, out _),
                    $"Resident hit {hit} was evicted in iteration {iter}!");
            }
        }
    }

    [Fact]
    public void GetAvailableTransferSlots_SpanAndListOverloads_IdenticalResults()
    {
        using var slotManager = new GpuResidentSlotManager(slotCount: 8, slotByteSize: 1024, allocateVram: false);
        slotManager.AllocateOrEvictSlot(0, 1, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(0, 2, out _, out _, out _, out _);
        slotManager.AllocateOrEvictSlot(1, 3, out _, out _, out _, out _);
        slotManager.PinExpert(1, 3);

        int[] hits = new[] { 1, 2 };
        int listResult = slotManager.GetAvailableTransferSlots(0, (IReadOnlyList<int>)hits);
        int spanResult = slotManager.GetAvailableTransferSlots(0, (ReadOnlySpan<int>)hits.AsSpan());

        Assert.Equal(listResult, spanResult);
        Assert.Equal(5, listResult); // 8 total - 1 pinned - 2 resident hits = 5
    }
}

