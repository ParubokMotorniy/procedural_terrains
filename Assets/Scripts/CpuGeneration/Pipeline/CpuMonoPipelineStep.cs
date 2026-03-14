using System;
using UnityEngine;

namespace CpuGenerationPipeline
{
    public abstract class CpuMonoPipelineStep : MonoBehaviour, CpuPipelineStep
    {
        public abstract System.Threading.Tasks.Task ExecuteStep(CpuPipelineContext pipelineContext);
        public abstract InputExpectations GetStepExpectations();
        public abstract void UpdateHeightmapState(ref HeightmapProperties previousState);
    }
}