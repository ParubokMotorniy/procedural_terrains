using UnityEngine;

namespace GpuGenerationPipeline
{
    [System.Flags]
    public enum InputExpectations //defines what the step assumes about the input it receives from the previosu step
    {
        None = 0,
        HeightMapNormalized = 1 << 0,  // 1
    }

    [System.Flags]

    public enum HeightmapProperties
    {
        Created = 0,
        Normalized = 1 << 0,
    }

    public interface PipelineStep
    {
        public void ExecuteStepGpu(PipelineContext pipelineContext);
        public InputExpectations GetStepExpectationsGpu();
        public void UpdateHeightmapStateGpu(ref HeightmapProperties previousState);
    }
}