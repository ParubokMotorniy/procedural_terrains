using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;

public class CpuSDFDispatcher : CpuMonoPipelineStep
{
    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    //both are implicitly cleared
    // private ComputeBuffer floodingBuffer1;
    // private ComputeBuffer floodingBuffer2;
    // private ComputeBuffer inputContinentHeightmap;

    public override void ExecuteStep(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }
}
