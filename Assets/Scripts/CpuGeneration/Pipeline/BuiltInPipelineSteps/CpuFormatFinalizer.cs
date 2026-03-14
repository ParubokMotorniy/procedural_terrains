using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace CpuGenerationPipeline
{
    public class CpuFormatFinalizer : CpuPipelineStep
    {
        public Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
        {
            pipelineContext.intermediateHeightmap.Apply();
            Graphics.Blit(pipelineContext.intermediateHeightmap, pipelineContext.finalHeightmap);
            return Task.CompletedTask;
        }

        public CpuInputExpectations GetStepExpectationsCpu()
        {
            return CpuInputExpectations.HeightMapNormalized;
        }

        public void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
        {

        }
    }
}