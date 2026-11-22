namespace Glacier.Inference.Hardware;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Recognized OpenAI-compatible local NPU inference sidecar runtimes.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<NpuSidecarRuntime>))]
public enum NpuSidecarRuntime
{
    Unknown,
    FastFlowLM,
    LemonadeServer,
    CustomOpenAi
}

/// <summary>
/// Descriptor for a reachable local OpenAI-compatible NPU sidecar endpoint.
/// </summary>
public sealed record NpuSidecarEndpoint(
    string BaseUrl,
    NpuSidecarRuntime Runtime,
    IReadOnlyList<string> Models,
    double LatencyMs)
{
    public string? PrimaryModel => Models.Count > 0 ? Models[0] : null;
}

internal sealed class OpenAiModelsListResponse
{
    [JsonPropertyName("data")]
    public List<OpenAiModelEntry>? Data { get; set; }
}

internal sealed class OpenAiModelEntry
{
    [JsonPropertyName("id")]
    public string? Id { get; set; }
}

[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(OpenAiModelsListResponse))]
internal partial class NpuSidecarJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Zero-dependency, NativeAOT-compatible prober for local OpenAI-compatible NPU sidecar runtimes
/// (AMD FastFlowLM <c>flm serve</c>, AMD Lemonade Server, or a user-configured <c>GLACIER_NPU_ENDPOINT</c>).
/// </summary>
public static class NpuSidecarDiscovery
{
    /// <summary>
    /// Optional environment variable allowing operators to point Glacier at a custom local NPU sidecar URL.
    /// </summary>
    public const string EndpointEnvVar = "GLACIER_NPU_ENDPOINT";

    /// <summary>
    /// Default FastFlowLM (<c>flm serve</c>) OpenAI-compatible base URL.
    /// </summary>
    public const string DefaultFastFlowLmUrl = "http://127.0.0.1:52625";

    /// <summary>
    /// Default Lemonade Server OpenAI-compatible base URLs.
    /// </summary>
    public const string DefaultLemonadeUrl = "http://127.0.0.1:8000";
    public const string DefaultLemonadeAltUrl = "http://127.0.0.1:13305";

    /// <summary>
    /// Returns the ordered list of candidate local NPU sidecar endpoints to probe.
    /// </summary>
    public static IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)> GetDefaultCandidateEndpoints(
        string? explicitEndpoint = null)
    {
        var list = new List<(string BaseUrl, NpuSidecarRuntime Runtime)>(4);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void TryAdd(string? url, NpuSidecarRuntime runtime)
        {
            if (string.IsNullOrWhiteSpace(url))
                return;

            string normalized = NormalizeBaseUrl(url);
            if (seen.Add(normalized))
                list.Add((normalized, runtime));
        }

        TryAdd(explicitEndpoint, InferRuntime(explicitEndpoint));
        TryAdd(Environment.GetEnvironmentVariable(EndpointEnvVar), InferRuntime(Environment.GetEnvironmentVariable(EndpointEnvVar)));
        TryAdd(DefaultFastFlowLmUrl, NpuSidecarRuntime.FastFlowLM);
        TryAdd(DefaultLemonadeUrl, NpuSidecarRuntime.LemonadeServer);
        TryAdd(DefaultLemonadeAltUrl, NpuSidecarRuntime.LemonadeServer);

        return list;
    }

    /// <summary>
    /// Default time-to-live for cached <see cref="ProbeCachedAsync"/> outcomes (both positive and negative).
    /// </summary>
    public static readonly TimeSpan DefaultCacheTtl = TimeSpan.FromSeconds(5);

    private sealed class CacheSlot
    {
        public readonly SemaphoreSlim Gate = new(1, 1);
        public bool HasValue;
        public long UpdatedAtTimestamp;
        public long Generation;
        public NpuSidecarEndpoint? Endpoint;
    }

    private static readonly object s_cacheLock = new();
    private static readonly Dictionary<string, CacheSlot> s_nullClientSlots = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConditionalWeakTable<HttpClient, Dictionary<string, CacheSlot>> s_clientSlots = new();
    private static long _cacheGeneration;

    /// <summary>
    /// Clears all cached <see cref="ProbeCachedAsync"/> results and invalidates any in-flight cache writes.
    /// </summary>
    public static void InvalidateCache()
    {
        Interlocked.Increment(ref _cacheGeneration);
        lock (s_cacheLock)
        {
            s_nullClientSlots.Clear();
            s_clientSlots.Clear();
        }
    }

    /// <summary>
    /// Probes default (or caller-supplied) local OpenAI-compatible NPU sidecar endpoints using a short-lived
    /// TTL cache (both positive and negative outcomes) with stampede coalescing per target/client slot.
    /// Non-positive <paramref name="perEndpointTimeout"/> values (other than <see cref="Timeout.InfiniteTimeSpan"/>)
    /// are clamped to <see cref="TimeSpan.Zero"/>.
    /// </summary>
    public static async Task<NpuSidecarEndpoint?> ProbeCachedAsync(
        TimeSpan? cacheTtl = null,
        HttpClient? httpClient = null,
        IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? candidates = null,
        TimeSpan? perEndpointTimeout = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        TimeSpan effectiveTtl = cacheTtl ?? DefaultCacheTtl;
        if (effectiveTtl <= TimeSpan.Zero)
        {
            return await ProbeAsync(httpClient, candidates, perEndpointTimeout, cancellationToken)
                .ConfigureAwait(false);
        }

        var targets = candidates ?? GetDefaultCandidateEndpoints();
        string targetsKey = BuildTargetsCacheKey(targets);
        CacheSlot slot = GetOrCreateCacheSlot(targetsKey, httpClient);

        await slot.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            long currentGen = Volatile.Read(ref _cacheGeneration);
            if (slot.HasValue &&
                slot.Generation == currentGen &&
                Stopwatch.GetElapsedTime(slot.UpdatedAtTimestamp) < effectiveTtl)
            {
                return slot.Endpoint;
            }

            NpuSidecarEndpoint? probed = await ProbeAsync(httpClient, targets, perEndpointTimeout, cancellationToken)
                .ConfigureAwait(false);

            if (Volatile.Read(ref _cacheGeneration) == currentGen)
            {
                slot.Endpoint = probed;
                slot.UpdatedAtTimestamp = Stopwatch.GetTimestamp();
                slot.Generation = currentGen;
                slot.HasValue = true;
            }

            return probed;
        }
        finally
        {
            slot.Gate.Release();
        }
    }

    /// <summary>
    /// Probes default (or caller-supplied) local OpenAI-compatible NPU sidecar endpoints concurrently
    /// while strictly preserving candidate priority order, returning the highest-priority server
    /// responding cleanly to <c>GET /v1/models</c>.
    /// Non-positive <paramref name="perEndpointTimeout"/> values (other than <see cref="Timeout.InfiniteTimeSpan"/>)
    /// are clamped to <see cref="TimeSpan.Zero"/>.
    /// </summary>
    public static async Task<NpuSidecarEndpoint?> ProbeAsync(
        HttpClient? httpClient = null,
        IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)>? candidates = null,
        TimeSpan? perEndpointTimeout = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var targets = candidates ?? GetDefaultCandidateEndpoints();
        if (targets.Count == 0)
            return null;

        bool ownsClient = httpClient == null;
        HttpClient client = httpClient ?? new HttpClient();
        TimeSpan timeout = NormalizeTimeout(perEndpointTimeout);

        if (targets.Count == 1)
        {
            try
            {
                return await ProbeSingleCoreAsync(
                    client,
                    targets[0].BaseUrl,
                    targets[0].Runtime,
                    timeout,
                    cancellationToken,
                    cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                if (ownsClient)
                    client.Dispose();
            }
        }

        var raceCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var tasks = new Task<NpuSidecarEndpoint?>[targets.Count];
        for (int i = 0; i < targets.Count; i++)
        {
            var (baseUrl, runtime) = targets[i];
            tasks[i] = ProbeSingleCoreAsync(client, baseUrl, runtime, timeout, cancellationToken, raceCts.Token);
        }

        try
        {
            for (int i = 0; i < tasks.Length; i++)
            {
                NpuSidecarEndpoint? discovered = await tasks[i].ConfigureAwait(false);
                if (discovered != null)
                {
                    raceCts.Cancel();
                    return discovered;
                }
            }

            return null;
        }
        finally
        {
            _ = Task.WhenAll(tasks).ContinueWith(
                static (t, state) =>
                {
                    _ = t.Exception;
                    var (ownedClient, cts) = ((HttpClient?, CancellationTokenSource))state!;
                    cts.Dispose();
                    ownedClient?.Dispose();
                },
                (ownsClient ? client : null, raceCts),
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
    }

    /// <summary>
    /// Probes a single OpenAI-compatible base URL via <c>GET /v1/models</c>.
    /// Non-positive <paramref name="timeout"/> values (other than <see cref="Timeout.InfiniteTimeSpan"/>)
    /// are clamped to <see cref="TimeSpan.Zero"/>.
    /// </summary>
    public static Task<NpuSidecarEndpoint?> ProbeSingleAsync(
        HttpClient httpClient,
        string baseUrl,
        NpuSidecarRuntime runtimeHint = NpuSidecarRuntime.Unknown,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        return ProbeSingleCoreAsync(
            httpClient,
            baseUrl,
            runtimeHint,
            NormalizeTimeout(timeout),
            cancellationToken,
            cancellationToken);
    }

    private static TimeSpan NormalizeTimeout(TimeSpan? requestedTimeout)
    {
        TimeSpan timeout = requestedTimeout ?? TimeSpan.FromMilliseconds(350);
        if (timeout == Timeout.InfiniteTimeSpan)
            return timeout;
        if (timeout <= TimeSpan.Zero)
            return TimeSpan.Zero;
        TimeSpan maxSupported = TimeSpan.FromMilliseconds(uint.MaxValue - 2);
        return timeout > maxSupported ? maxSupported : timeout;
    }

    private static async Task<NpuSidecarEndpoint?> ProbeSingleCoreAsync(
        HttpClient httpClient,
        string baseUrl,
        NpuSidecarRuntime runtimeHint,
        TimeSpan timeout,
        CancellationToken callerCancellationToken,
        CancellationToken raceToken)
    {
        if (string.IsNullOrWhiteSpace(baseUrl))
            return null;

        string normalized = NormalizeBaseUrl(baseUrl);
        if (string.IsNullOrWhiteSpace(normalized))
            return null;

        string probeUrl = BuildModelsProbeUrl(normalized);
        if (Uri.TryCreate(probeUrl, UriKind.Absolute, out Uri? absoluteProbeUri))
        {
            if (!string.Equals(absoluteProbeUri.Scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(absoluteProbeUri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }
        }
        else if (httpClient.BaseAddress == null)
        {
            return null;
        }

        NpuSidecarRuntime runtime = runtimeHint == NpuSidecarRuntime.Unknown ? InferRuntime(normalized) : runtimeHint;

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(raceToken);
        cts.CancelAfter(timeout);

        var sw = Stopwatch.StartNew();
        try
        {
            using var resp = await httpClient.GetAsync(probeUrl, cts.Token).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
                return null;

            string body = await resp.Content.ReadAsStringAsync(cts.Token).ConfigureAwait(false);
            sw.Stop();

            var parsed = JsonSerializer.Deserialize(body, NpuSidecarJsonContext.Default.OpenAiModelsListResponse);
            // Require valid JSON response with a data array (prevents false positives from unrelated HTTP servers)
            if (parsed?.Data == null)
                return null;

            var models = new List<string>(parsed.Data.Count);
            foreach (var entry in parsed.Data)
            {
                if (!string.IsNullOrWhiteSpace(entry?.Id))
                    models.Add(entry.Id.Trim());
            }

            return new NpuSidecarEndpoint(normalized, runtime, models.AsReadOnly(), sw.Elapsed.TotalMilliseconds);
        }
        catch (OperationCanceledException) when (!callerCancellationToken.IsCancellationRequested)
        {
            return null;
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (JsonException)
        {
            return null;
        }
        catch (UriFormatException)
        {
            return null;
        }
        catch (NotSupportedException)
        {
            return null;
        }
    }

    private static CacheSlot GetOrCreateCacheSlot(string targetsKey, HttpClient? httpClient)
    {
        lock (s_cacheLock)
        {
            Dictionary<string, CacheSlot> bucket;
            if (httpClient == null)
            {
                bucket = s_nullClientSlots;
            }
            else if (!s_clientSlots.TryGetValue(httpClient, out bucket!))
            {
                bucket = new Dictionary<string, CacheSlot>(StringComparer.OrdinalIgnoreCase);
                s_clientSlots.Add(httpClient, bucket);
            }

            if (!bucket.TryGetValue(targetsKey, out CacheSlot? slot))
            {
                slot = new CacheSlot();
                bucket[targetsKey] = slot;
            }

            return slot;
        }
    }

    private static string BuildTargetsCacheKey(IReadOnlyList<(string BaseUrl, NpuSidecarRuntime Runtime)> targets)
    {
        if (targets.Count == 0)
            return string.Empty;

        var sb = new StringBuilder(targets.Count * 32);
        for (int i = 0; i < targets.Count; i++)
        {
            if (i > 0)
                sb.Append('|');
            string url = string.IsNullOrWhiteSpace(targets[i].BaseUrl)
                ? string.Empty
                : NormalizeBaseUrl(targets[i].BaseUrl);
            sb.Append(url);
            sb.Append('#');
            sb.Append((int)targets[i].Runtime);
        }
        return sb.ToString();
    }

    private static string NormalizeBaseUrl(string url)
    {
        string trimmed = url.Trim();
        int suffixIdx = trimmed.IndexOfAny(['?', '#']);
        string basePart = (suffixIdx >= 0 ? trimmed[..suffixIdx] : trimmed).TrimEnd('/');
        string suffix = suffixIdx >= 0 ? trimmed[suffixIdx..] : string.Empty;

        if (basePart.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
            basePart = basePart[..^3].TrimEnd('/');

        if (string.IsNullOrWhiteSpace(basePart))
            return string.Empty;

        return basePart + suffix;
    }

    private static string BuildModelsProbeUrl(string normalizedBaseUrl)
    {
        int suffixIdx = normalizedBaseUrl.IndexOfAny(['?', '#']);
        string basePart = suffixIdx >= 0 ? normalizedBaseUrl[..suffixIdx] : normalizedBaseUrl;
        string suffix = suffixIdx >= 0 ? normalizedBaseUrl[suffixIdx..] : string.Empty;
        return $"{basePart}/v1/models{suffix}";
    }

    private static NpuSidecarRuntime InferRuntime(string? url)
    {
        if (string.IsNullOrWhiteSpace(url))
            return NpuSidecarRuntime.Unknown;
        if (Uri.TryCreate(url.Trim(), UriKind.Absolute, out Uri? uri))
        {
            return uri.Port switch
            {
                52625 => NpuSidecarRuntime.FastFlowLM,
                8000 or 13305 => NpuSidecarRuntime.LemonadeServer,
                _ => NpuSidecarRuntime.CustomOpenAi
            };
        }
        return NpuSidecarRuntime.CustomOpenAi;
    }
}
