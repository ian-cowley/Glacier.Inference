namespace Glacier.Inference.Image;

using System;

/// <summary>
/// Flow Matching (Rectified Flow) ODE Scheduler.
/// Solves the reverse probability flow trajectory from Gaussian noise (t=1.0) to data (t=0.0)
/// using deterministic Euler stepping. Used by state-of-the-art Diffusion Transformers
/// (Flux.1 [schnell], Stable Diffusion 3.5, and LTX-Video).
/// </summary>
public sealed class FlowMatchingScheduler
{
    private readonly int _numSteps;
    private readonly float _timeShift;
    private readonly float[] _timesteps;

    public int NumSteps => _numSteps;
    public float TimeShift => _timeShift;
    public ReadOnlySpan<float> Timesteps => _timesteps;

    public FlowMatchingScheduler(int numSteps = 4, float timeShift = 1.0f)
    {
        if (numSteps < 1) throw new ArgumentOutOfRangeException(nameof(numSteps), "Steps must be >= 1");

        _numSteps = numSteps;
        _timeShift = timeShift;
        _timesteps = new float[numSteps + 1];

        if (MathF.Abs(timeShift - 1.0f) < 1e-5f)
        {
            // Linear equidistant schedule for models without time-shift (e.g. FLUX Schnell)
            for (int i = 0; i <= numSteps; i++)
            {
                _timesteps[i] = 1.0f - (float)i / numSteps;
            }
        }
        else
        {
            // Standard FlowMatchEulerDiscreteScheduler formulation (Wan 2.1 / SD3 / Flux Dev)
            for (int i = 0; i < numSteps; i++)
            {
                float t = (numSteps == 1) ? 1000.0f : 1000.0f - (float)i * (1000.0f - 1.0f) / (numSteps - 1);
                float sigma = t / 1000.0f;
                _timesteps[i] = ApplyTimeShift(sigma, _timeShift);
            }
            _timesteps[numSteps] = 0.0f;
        }
    }

    /// <summary>
    /// Executes a single first-order Euler integration step along the predicted velocity field:
    /// x_{t + dt} = x_t + dt * v_theta(x_t, t)
    /// </summary>
    public static void Step(
        Span<float> latents,
        ReadOnlySpan<float> velocity,
        float currentT,
        float nextT)
    {
        float dt = nextT - currentT; // Negative since currentT > nextT

        int len = Math.Min(latents.Length, velocity.Length);
        for (int i = 0; i < len; i++)
        {
            latents[i] += dt * velocity[i];
        }
    }

    /// <summary>
    /// Computes the Flux/SD3 resolution-dependent time-shift function:
    /// t_shifted = (shift * t) / (1 + (shift - 1) * t)
    /// </summary>
    public static float ApplyTimeShift(float t, float shift)
    {
        if (MathF.Abs(shift - 1.0f) < 1e-5f) return t;
        return (shift * t) / (1.0f + (shift - 1.0f) * t);
    }
}
