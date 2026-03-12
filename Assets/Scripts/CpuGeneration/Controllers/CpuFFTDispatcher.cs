using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;

public class CpuFFTDispatcher : CpuMonoPipelineStep
{

    [Range(4, 8)]
    public int inverseGroupScaleFactor = 4;

    [Range(0.01f, 5.0f)]
    public float fractalDimension = 0.1f;

    [Range(0.01f, 1.0f)]
    public float fracCoefficientsConsidered = 0.01f;

    private const int generationGroupSize = 16;
    private const int inverseGroupSize = 16;

    //implicitly cleared
    // private ComputeBuffer coefficientsBuffer;

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    public override void ExecuteStep(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }
}
