using UnityEngine;
using UnityEngine.Assertions;

namespace GenerationPipeline
{
    public abstract class MultiFormatPipelineStep : MonoPipelineStep
    {
        public abstract void StepInitialization(PipelineContext pipelineContext);
        public abstract void StepBody(PipelineContext pipelineContext);
        public abstract void StepConclusion(PipelineContext pipelineContext);

        protected static HeightmapNormalizer normalizationShader = new HeightmapNormalizer();
        protected void RunInternalNormalization(PipelineContext pipelineContext)
        {
            normalizationShader.ExecuteStep(pipelineContext);
        }
        public override void ExecuteStep(PipelineContext pipelineContext)
        {
            StepInitialization(pipelineContext);
            StepBody(pipelineContext);
            StepConclusion(pipelineContext);
        }
    }
}