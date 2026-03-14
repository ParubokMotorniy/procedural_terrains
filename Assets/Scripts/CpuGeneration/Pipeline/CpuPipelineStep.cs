using UnityEngine;
using System;

namespace CpuGenerationPipeline
{
    [System.Flags]
    public enum CpuInputExpectations //defines what the step assumes about the input it receives from the previosu step
    {
        None = 0,
        HeightMapNormalized = 1 << 0,  // 1
    }

    [System.Flags]

    public enum CpuHeightmapProperties
    {
        Created = 0,
        Normalized = 1 << 0,
    }

    public interface CpuPipelineStep
    {
        public System.Threading.Tasks.Task ExecuteStepCpu(CpuPipelineContext pipelineContext);
        public CpuInputExpectations GetStepExpectationsCpu();
        public void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState);
    }
}