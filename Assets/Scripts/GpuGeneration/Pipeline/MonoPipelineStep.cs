using UnityEngine;

namespace GenerationPipeline
{
    public abstract class MonoPipelineStep : MonoBehaviour, PipelineStep
    {
        public abstract void ExecuteStep(PipelineContext pipelineContext);
        public abstract InputExpectations GetStepExpectations();
        public abstract void UpdateHeightmapState(ref HeightmapProperties previousState);

        protected static HeightmapNormalizer normalizationShader = new HeightmapNormalizer();
        protected void RunInternalNormalization(PipelineContext pipelineContext)
        {
            normalizationShader.ExecuteStep(pipelineContext);
        }
    }
}