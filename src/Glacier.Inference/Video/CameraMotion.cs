namespace Glacier.Inference.Video;

/// <summary>
/// Camera and spatio-temporal motion dynamics for Generative Video production.
/// Controls camera trajectory and temporal scene evolution across video frames.
/// </summary>
public enum CameraMotion
{
    /// <summary>Static camera with intrinsic scene dynamics.</summary>
    Static = 0,

    /// <summary>Horizontal pan from left to right across the scene.</summary>
    PanRight = 1,

    /// <summary>Horizontal pan from right to left across the scene.</summary>
    PanLeft = 2,

    /// <summary>Vertical tilt upwards toward sky / canopy.</summary>
    TiltUp = 3,

    /// <summary>Vertical tilt downwards toward ground / reflections.</summary>
    TiltDown = 4,

    /// <summary>Forward camera push / dolly zoom in.</summary>
    ZoomIn = 5,

    /// <summary>Backward camera pull / dolly zoom out.</summary>
    ZoomOut = 6,

    /// <summary>Orbital rotational roll around central focal point.</summary>
    Orbit = 7,

    /// <summary>Dynamic organic fluid motion (auroras, water, smoke, atmospheric drift).</summary>
    DynamicFluid = 8
}
