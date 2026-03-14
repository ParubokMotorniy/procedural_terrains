using System;
using UnityEngine;

namespace CpuGenerationPipeline
{
    public abstract class CpuMonoPipelineStep : MonoBehaviour, CpuPipelineStep
    {
        public abstract System.Threading.Tasks.Task ExecuteStepCpu(CpuPipelineContext pipelineContext);
        public abstract InputExpectations GetStepExpectationsCpu();
        public abstract void UpdateHeightmapStateCpu(ref HeightmapProperties previousState);
    }
}