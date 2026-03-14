using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace CpuGenerationPipeline
{
    public class CpuFormatFinalizer : CpuPipelineStep
    {
        public Task ExecuteStep(CpuPipelineContext pipelineContext)
        {
            pipelineContext.intermediateHeightmap.Apply();
            Graphics.Blit(pipelineContext.intermediateHeightmap, pipelineContext.finalHeightmap);
            return Task.CompletedTask;
        }

        public InputExpectations GetStepExpectations()
        {
            return InputExpectations.HeightMapNormalized;
        }

        public void UpdateHeightmapState(ref HeightmapProperties previousState)
        {

        }
    }
}