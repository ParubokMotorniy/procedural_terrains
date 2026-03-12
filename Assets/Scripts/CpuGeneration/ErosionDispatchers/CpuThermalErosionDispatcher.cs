using CpuGenerationPipeline;
using UnityEngine.Assertions;
using Unity.Mathematics;
using UnityEngine;
using System;

public class CpuThermalErosionDispatcher : CpuMonoPipelineStep
{
    [Range(0.01f, 1.0f)]
    public float distributionCoefficient = 0.75f;

    [Range(0.001f, 1.0f)]
    public float talusThreshold = 0.15f;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    public override void ExecuteStep(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;
    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
    }
}
