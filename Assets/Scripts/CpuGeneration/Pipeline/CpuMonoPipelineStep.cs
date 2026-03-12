using UnityEngine;

namespace CpuGenerationPipeline
{
    public abstract class CpuMonoPipelineStep : MonoBehaviour, CpuPipelineStep
    {
        public abstract void ExecuteStep(CpuPipelineContext pipelineContext);
        public abstract InputExpectations GetStepExpectations();
        public abstract void UpdateHeightmapState(ref HeightmapProperties previousState);
    }
}