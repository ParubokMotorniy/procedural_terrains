using System.Collections.Generic;
using UnityEngine;
public static class CommonDefines
{
    public enum AvailablePipelineSteps
    {
        //generators
        UN,
        RMD,
        FFT,
        SDF,
        //eroders
        HE,
        TE,
        PE
    };

    public static List<UltimatePipelineStep> buildPipelineFromEnum(List<AvailablePipelineSteps> pipelineDescription)
    {
        List<UltimatePipelineStep> eventualPipeline = new();
        foreach (AvailablePipelineSteps step in pipelineDescription)
        {
            switch (step)
            {
                case AvailablePipelineSteps.UN:
                    eventualPipeline.Add(GameObject.FindObjectsByType<UNDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.RMD:
                    eventualPipeline.Add(GameObject.FindObjectsByType<RMDDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.FFT:
                    eventualPipeline.Add(GameObject.FindObjectsByType<FFTDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.SDF:
                    eventualPipeline.Add(GameObject.FindObjectsByType<SDFDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.HE:
                    eventualPipeline.Add(GameObject.FindObjectsByType<CellularHydraulicErosionDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.TE:
                    eventualPipeline.Add(GameObject.FindObjectsByType<ThermalErosionDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
                case AvailablePipelineSteps.PE:
                    eventualPipeline.Add(GameObject.FindObjectsByType<ParticleHydraulicErosionDispatcher>(FindObjectsSortMode.None)[0]);
                    break;
            }
        }

        return eventualPipeline;
    }
}
