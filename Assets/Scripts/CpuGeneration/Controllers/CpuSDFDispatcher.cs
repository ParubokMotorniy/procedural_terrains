using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;
using System.Threading.Tasks;

public class CpuSDFDispatcher : CpuMonoPipelineStep
{
    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    public override Task ExecuteStep(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    //both are implicitly cleared
    // private ComputeBuffer floodingBuffer1;
    // private ComputeBuffer floodingBuffer2;
    // private ComputeBuffer inputContinentHeightmap;

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }
}
