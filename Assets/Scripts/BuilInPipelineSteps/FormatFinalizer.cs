using UnityEngine;

namespace GenerationPipeline
{
    public class FormatFinalizer : PipelineStep
    {
        public void ExecuteStep(PipelineContext pipelineContext)
        {
            //TODO: can probably be done more effectively
            Graphics.Blit(pipelineContext.intermediateHeightmap, pipelineContext.finalHeightmap);
        }

        public InputExpectations GetStepExpectations()
        {
            return InputExpectations.HeightMapNormalized;
        }
    }
}