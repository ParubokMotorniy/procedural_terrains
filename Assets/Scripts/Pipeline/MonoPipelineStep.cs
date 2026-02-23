using UnityEngine;

namespace GenerationPipeline
{
    public abstract class MonoPipelineStep : MonoBehaviour, PipelineStep
    {
        public abstract void ExecuteStep(PipelineContext pipelineContext);
        public abstract InputExpectations GetStepExpectations();
        public abstract void UpdateHeightmapState(ref HeightmapProperties previousState);
    }
}