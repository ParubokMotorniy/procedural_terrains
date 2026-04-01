using System;
using UnityEngine;

namespace GpuGenerationPipeline
{
    public class FormatFinalizer : PipelineStep
    {
        public void ExecuteStepGpu(PipelineContext pipelineContext)
        {
            //TODO: can probably be done more effectively
            pipelineContext.AppendTextureCopyToCommandBuffer(pipelineContext.intermediateHeightmap, pipelineContext.finalHeightmap);
        }

        public InputExpectations GetStepExpectationsGpu()
        {
            return InputExpectations.HeightMapNormalized;
        }

        public void RandomizeParameters(System.Random random)
        {
        }

        public void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
        {

        }
    }
}