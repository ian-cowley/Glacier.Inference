namespace Glacier.Inference.Memory;

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

/// <summary>
/// Thread-safe empirical per-(layer, expert) routing frequency profile with exponential window decay
/// and Native-AOT-compatible binary persistence (<c>.gmoe.profile</c>).
/// </summary>
public sealed class ExpertRoutingProfile
{
    public const uint ProfileMagic = 0x46525047; // "GPRF" Little-Endian
    public const uint CurrentVersion = 1;
    private const int HeaderSizeBytes = 32;

    private readonly int _numLayers;
    private readonly int _numExperts;
    private readonly double _decayFactor;
    private readonly int _adaptWindowSteps;
    private readonly double[] _scores;
    private readonly long[] _totalActivations;
    private readonly Lock _lock = new();
    private long _stepCount;

    public int NumLayers => _numLayers;
    public int NumExperts => _numExperts;
    public double DecayFactor => _decayFactor;
    public int AdaptWindowSteps => _adaptWindowSteps;
    public long StepCount
    {
        get
        {
            lock (_lock)
            {
                return _stepCount;
            }
        }
    }

    public ExpertRoutingProfile(
        int numLayers,
        int numExperts,
        double decayFactor = 0.995,
        int adaptWindowSteps = 64)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numExperts);
        if (!double.IsFinite(decayFactor) || decayFactor <= 0.0 || decayFactor > 1.0)
            throw new ArgumentOutOfRangeException(nameof(decayFactor), "Decay factor must be in (0.0, 1.0].");

        _numLayers = numLayers;
        _numExperts = numExperts;
        _decayFactor = decayFactor;
        _adaptWindowSteps = Math.Max(1, adaptWindowSteps);

        int total = checked(numLayers * numExperts);
        _scores = new double[total];
        _totalActivations = new long[total];
    }

    private ExpertRoutingProfile(
        int numLayers,
        int numExperts,
        double decayFactor,
        int adaptWindowSteps,
        long stepCount,
        double[] scores,
        long[] totalActivations)
    {
        _numLayers = numLayers;
        _numExperts = numExperts;
        _decayFactor = decayFactor;
        _adaptWindowSteps = adaptWindowSteps;
        _stepCount = stepCount;
        _scores = scores;
        _totalActivations = totalActivations;
    }

    /// <summary>
    /// Records router activations for a layer and triggers window decay every <see cref="AdaptWindowSteps"/> calls.
    /// </summary>
    public void RecordActivations(
        int layerIdx,
        ReadOnlySpan<int> expertIndices,
        ReadOnlySpan<float> weights = default)
    {
        if ((uint)layerIdx >= (uint)_numLayers || expertIndices.IsEmpty)
            return;

        int layerOffset = layerIdx * _numExperts;

        lock (_lock)
        {
            for (int k = 0; k < expertIndices.Length; k++)
            {
                int expIdx = expertIndices[k];
                if ((uint)expIdx >= (uint)_numExperts) continue;

                double inc = (!weights.IsEmpty && k < weights.Length && float.IsFinite(weights[k]) && weights[k] > 0f)
                    ? Math.Max(0.05, weights[k])
                    : 1.0;

                int flatIdx = layerOffset + expIdx;
                _scores[flatIdx] += inc;
                _totalActivations[flatIdx]++;
            }

            _stepCount++;
            if (_stepCount % _adaptWindowSteps == 0)
            {
                ApplyDecayLocked(_decayFactor);
            }
        }
    }

    /// <summary>
    /// Applies exponential score decay across all (layer, expert) entries.
    /// </summary>
    public void ApplyDecay(double? customFactor = null)
    {
        double factor = Math.Clamp(customFactor ?? _decayFactor, 0.0, 1.0);
        lock (_lock)
        {
            ApplyDecayLocked(factor);
        }
    }

    private void ApplyDecayLocked(double factor)
    {
        for (int i = 0; i < _scores.Length; i++)
        {
            _scores[i] *= factor;
        }
    }

    public double GetScore(int layerIdx, int expertIdx)
    {
        if ((uint)layerIdx >= (uint)_numLayers || (uint)expertIdx >= (uint)_numExperts)
            return 0.0;

        lock (_lock)
        {
            return _scores[layerIdx * _numExperts + expertIdx];
        }
    }

    public long GetTotalActivations(int layerIdx, int expertIdx)
    {
        if ((uint)layerIdx >= (uint)_numLayers || (uint)expertIdx >= (uint)_numExperts)
            return 0;

        lock (_lock)
        {
            return _totalActivations[layerIdx * _numExperts + expertIdx];
        }
    }

    /// <summary>
    /// Copies up to <paramref name="destination"/>.Length top-scoring expert indices for <paramref name="layerIdx"/>,
    /// ordered by descending decayed score, descending total activations, then ascending expert index.
    /// </summary>
    public int GetTopExpertsForLayer(int layerIdx, Span<int> destination, bool includeZeroScores = false)
    {
        if ((uint)layerIdx >= (uint)_numLayers || destination.IsEmpty)
            return 0;

        int layerOffset = layerIdx * _numExperts;
        int n = _numExperts;

        Span<double> scoresCopy = n <= 1024 ? stackalloc double[n] : new double[n];
        Span<long> actsCopy = n <= 1024 ? stackalloc long[n] : new long[n];
        Span<int> indices = n <= 1024 ? stackalloc int[n] : new int[n];

        lock (_lock)
        {
            _scores.AsSpan(layerOffset, n).CopyTo(scoresCopy);
            _totalActivations.AsSpan(layerOffset, n).CopyTo(actsCopy);
        }

        int candidateCount = 0;
        for (int e = 0; e < n; e++)
        {
            if (includeZeroScores || scoresCopy[e] > 0.0 || actsCopy[e] > 0)
            {
                indices[candidateCount++] = e;
            }
        }

        int toWrite = Math.Min(candidateCount, destination.Length);
        // Partial selection sort avoids managed Comparer<T>/delegate allocations
        for (int i = 0; i < toWrite; i++)
        {
            int bestPos = i;
            int bestExp = indices[i];
            double bestScore = scoresCopy[bestExp];
            long bestAct = actsCopy[bestExp];

            for (int j = i + 1; j < candidateCount; j++)
            {
                int exp = indices[j];
                double sc = scoresCopy[exp];
                long act = actsCopy[exp];

                bool better = sc > bestScore ||
                              (Math.Abs(sc - bestScore) <= 1e-12 && (act > bestAct || (act == bestAct && exp < bestExp)));
                if (better)
                {
                    bestPos = j;
                    bestExp = exp;
                    bestScore = sc;
                    bestAct = act;
                }
            }

            if (bestPos != i)
            {
                indices[bestPos] = indices[i];
                indices[i] = bestExp;
            }
            destination[i] = bestExp;
        }

        return toWrite;
    }

    /// <summary>
    /// Copies up to <paramref name="destination"/>.Length globally hottest (layer, expert, score) tuples across all layers.
    /// </summary>
    public int GetTopLayerExpertPairs(
        Span<(int LayerIndex, int ExpertIndex, double Score)> destination,
        bool includeZeroScores = false)
    {
        if (destination.IsEmpty)
            return 0;

        int total = _numLayers * _numExperts;
        double[]? rentedScores = total > 1024 ? new double[total] : null;
        long[]? rentedActs = total > 1024 ? new long[total] : null;
        int[]? rentedIndices = total > 1024 ? new int[total] : null;

        Span<double> scoresCopy = rentedScores ?? stackalloc double[total];
        Span<long> actsCopy = rentedActs ?? stackalloc long[total];
        Span<int> flatIndices = rentedIndices ?? stackalloc int[total];

        lock (_lock)
        {
            _scores.AsSpan().CopyTo(scoresCopy);
            _totalActivations.AsSpan().CopyTo(actsCopy);
        }

        int candidateCount = 0;
        for (int i = 0; i < total; i++)
        {
            if (includeZeroScores || scoresCopy[i] > 0.0 || actsCopy[i] > 0)
            {
                flatIndices[candidateCount++] = i;
            }
        }

        int toWrite = Math.Min(candidateCount, destination.Length);
        for (int i = 0; i < toWrite; i++)
        {
            int bestPos = i;
            int bestFlat = flatIndices[i];
            double bestScore = scoresCopy[bestFlat];
            long bestAct = actsCopy[bestFlat];

            for (int j = i + 1; j < candidateCount; j++)
            {
                int flat = flatIndices[j];
                double sc = scoresCopy[flat];
                long act = actsCopy[flat];

                bool better = sc > bestScore ||
                              (Math.Abs(sc - bestScore) <= 1e-12 && (act > bestAct || (act == bestAct && flat < bestFlat)));
                if (better)
                {
                    bestPos = j;
                    bestFlat = flat;
                    bestScore = sc;
                    bestAct = act;
                }
            }

            if (bestPos != i)
            {
                flatIndices[bestPos] = flatIndices[i];
                flatIndices[i] = bestFlat;
            }

            int layerIdx = bestFlat / _numExperts;
            int expIdx = bestFlat - (layerIdx * _numExperts);
            destination[i] = (layerIdx, expIdx, bestScore);
        }

        return toWrite;
    }

    /// <summary>
    /// Atomically persists the routing frequency profile to disk without reflection.
    /// </summary>
    public void Save(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);

        string tmpPath = filePath + ".tmp";
        int total = _scores.Length;
        double[] scoresSnapshot = new double[total];
        long[] actsSnapshot = new long[total];
        long stepSnapshot;

        lock (_lock)
        {
            _scores.AsSpan().CopyTo(scoresSnapshot);
            _totalActivations.AsSpan().CopyTo(actsSnapshot);
            stepSnapshot = _stepCount;
        }

        Span<byte> header = stackalloc byte[HeaderSizeBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(0, 4), ProfileMagic);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(4, 4), CurrentVersion);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(header.Slice(8, 4), _numLayers);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(header.Slice(12, 4), _numExperts);
        System.Buffers.Binary.BinaryPrimitives.WriteDoubleLittleEndian(header.Slice(16, 8), _decayFactor);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(header.Slice(24, 8), stepSnapshot);

        using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            fs.Write(header);
            fs.Write(MemoryMarshal.AsBytes(scoresSnapshot.AsSpan()));
            fs.Write(MemoryMarshal.AsBytes(actsSnapshot.AsSpan()));
            fs.Flush();
        }

        File.Move(tmpPath, filePath, overwrite: true);
    }

    public static ExpertRoutingProfile Load(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Expert routing profile file not found.", filePath);

        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (fs.Length < HeaderSizeBytes)
            throw new InvalidDataException("Expert routing profile file is smaller than the 32-byte header.");

        Span<byte> header = stackalloc byte[HeaderSizeBytes];
        fs.ReadExactly(header);

        uint magic = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(0, 4));
        uint version = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(4, 4));
        int numLayers = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(header.Slice(8, 4));
        int numExperts = System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(header.Slice(12, 4));
        double decayFactor = System.Buffers.Binary.BinaryPrimitives.ReadDoubleLittleEndian(header.Slice(16, 8));
        long stepCount = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(header.Slice(24, 8));

        if (magic != ProfileMagic)
            throw new InvalidDataException($"Invalid profile magic 0x{magic:X8}.");
        if (version != CurrentVersion)
            throw new InvalidDataException($"Unsupported profile version {version}.");
        if (numLayers <= 0 || numExperts <= 0)
            throw new InvalidDataException("Profile dimensions must be strictly positive.");

        long totalEntries = (long)numLayers * numExperts;
        long expectedBytes = HeaderSizeBytes + (totalEntries * (sizeof(double) + sizeof(long)));
        if (fs.Length != expectedBytes)
            throw new InvalidDataException($"Profile file length {fs.Length} does not match expected {expectedBytes} bytes.");

        double safeDecay = (!double.IsFinite(decayFactor) || decayFactor <= 0.0 || decayFactor > 1.0) ? 0.995 : decayFactor;
        long safeStep = Math.Max(0L, stepCount);

        double[] scores = new double[totalEntries];
        long[] totalActs = new long[totalEntries];

        fs.ReadExactly(MemoryMarshal.AsBytes(scores.AsSpan()));
        fs.ReadExactly(MemoryMarshal.AsBytes(totalActs.AsSpan()));

        for (int i = 0; i < scores.Length; i++)
        {
            if (!double.IsFinite(scores[i]) || scores[i] < 0.0)
                scores[i] = 0.0;
            if (totalActs[i] < 0)
                totalActs[i] = 0;
        }

        return new ExpertRoutingProfile(numLayers, numExperts, safeDecay, 64, safeStep, scores, totalActs);
    }

    public static bool TryLoad(
        string filePath,
        int expectedLayers,
        int expectedExperts,
        out ExpertRoutingProfile? profile)
    {
        profile = null;
        try
        {
            if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
                return false;

            var loaded = Load(filePath);
            if (loaded.NumLayers != expectedLayers || loaded.NumExperts != expectedExperts)
                return false;

            profile = loaded;
            return true;
        }
        catch
        {
            return false;
        }
    }
}
