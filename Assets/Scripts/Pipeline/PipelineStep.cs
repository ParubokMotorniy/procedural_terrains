using UnityEngine;

namespace GenerationPipeline
{
    [System.Flags]
    public enum InputExpectations //defines what the step assumes about the input it receives from the previosu step
    {
        None = 0,
        HeightMapNormalized = 1 << 0,  // 1
    }

    public interface PipelineStep
    {
        public void ExecuteStep(PipelineContext pipelineContext);
        public InputExpectations GetStepExpectations();
    }
}