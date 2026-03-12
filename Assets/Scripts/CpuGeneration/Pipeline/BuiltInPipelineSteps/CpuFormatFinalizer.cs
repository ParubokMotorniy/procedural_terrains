using UnityEngine;

namespace CpuGenerationPipeline
{
    public class CpuFormatFinalizer : CpuPipelineStep
    {
        public InputExpectations GetStepExpectations()
        {
            return InputExpectations.HeightMapNormalized;
        }

        public void UpdateHeightmapState(ref HeightmapProperties previousState)
        {

        }
    }
}