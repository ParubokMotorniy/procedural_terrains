using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using Unity.VisualScripting;

namespace CpuGenerationPipeline
{
    [ExecuteAlways]
    public class CpuHeightmapNormalizer : CpuPipelineStep
    {

        public InputExpectations GetStepExpectations()
        {
            return InputExpectations.None;
        }

        public void UpdateHeightmapState(ref HeightmapProperties previousState)
        {
            previousState |= HeightmapProperties.Normalized;
        }
    }
}