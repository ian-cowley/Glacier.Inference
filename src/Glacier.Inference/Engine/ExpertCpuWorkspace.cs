namespace Glacier.Inference.Engine;

internal sealed class ExpertCpuWorkspace
{
    public readonly float[] Gate;
    public readonly float[] Up;
    public readonly float[] Act;
    public readonly float[] ActSums;
    public readonly float[] DownOut;

    public ExpertCpuWorkspace(int expertFfnDim, int dim)
    {
        Gate = new float[expertFfnDim];
        Up = new float[expertFfnDim];
        Act = new float[expertFfnDim];
        ActSums = new float[(expertFfnDim + 31) / 32];
        DownOut = new float[dim];
    }
}
