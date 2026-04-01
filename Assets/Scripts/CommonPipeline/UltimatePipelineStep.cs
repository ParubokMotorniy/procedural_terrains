using UnityEngine;
using GpuGenerationPipeline;
using CpuGenerationPipeline;
using System.Threading.Tasks;
using System.Collections;
using System;

public abstract class UltimatePipelineStep : TunableObject, CpuPipelineStep, PipelineStep
{
    public abstract Task ExecuteStepCpu(CpuPipelineContext pipelineContext);
    public abstract CpuInputExpectations GetStepExpectationsCpu();
    public abstract void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState);

    public abstract void ExecuteStepGpu(PipelineContext pipelineContext);
    public abstract InputExpectations GetStepExpectationsGpu();
    public abstract void UpdateHeightmapStateGpu(ref HeightmapProperties previousState);

    public abstract void FreeResources();
    public static void FreeAllResourcesInScene()
    {
        foreach (UltimatePipelineStep activeStep in FindObjectsByType<UltimatePipelineStep>(FindObjectsSortMode.None))
        {
            activeStep.FreeResources();
        }
    }
    public abstract void RandomizeParameters(System.Random random);
}
