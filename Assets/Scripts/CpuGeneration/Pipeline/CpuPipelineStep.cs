using UnityEngine;
using System;

namespace CpuGenerationPipeline
{
    [System.Flags]
    public enum InputExpectations //defines what the step assumes about the input it receives from the previosu step
    {
        None = 0,
        HeightMapNormalized = 1 << 0,  // 1
    }

    [System.Flags]

    public enum HeightmapProperties
    {
        Created = 0,
        Normalized = 1 << 0,
    }

    public interface CpuPipelineStep
    {
        public async System.Threading.Tasks.Task ExecuteStep(CpuPipelineContext pipelineContext) { throw new NotImplementedException(); }
        public InputExpectations GetStepExpectations();
        public void UpdateHeightmapState(ref HeightmapProperties previousState);
    }
}