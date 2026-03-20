using UnityEngine;
using GpuGenerationPipeline;
using CpuGenerationPipeline;
using System.Threading.Tasks;

public abstract class UltimatePipelineStep : MonoBehaviour, CpuPipelineStep, PipelineStep
{
    public abstract Task ExecuteStepCpu(CpuPipelineContext pipelineContext);
    public abstract CpuInputExpectations GetStepExpectationsCpu();
    public abstract void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState);

    public abstract void ExecuteStepGpu(PipelineContext pipelineContext);
    public abstract InputExpectations GetStepExpectationsGpu();
    public abstract void UpdateHeightmapStateGpu(ref HeightmapProperties previousState);

    public abstract void RenderParametersTuningGUI();
    public abstract string GUIStepTitle();
}
