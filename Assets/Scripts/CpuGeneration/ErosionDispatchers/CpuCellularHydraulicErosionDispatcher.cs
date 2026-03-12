using System;
using CpuGenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class CpuCellularHydraulicErosionDispatcher : CpuMonoPipelineStep
{
    [SerializeField]
    
    [Range(5, 100)]
    public int erosionIterationLimit = 25;

    [Range(0.0001f, 1.0f)]
    public float solubilityConstant;

    [Range(0.0001f, 1.0f)]
    public float evaporationConstant;

    [Range(0.0001f, 1.0f)]
    public float capacityConstant;

    [Range(0.0001f, 1.0f)]
    public float rainSolubilityConstant;

    [Range(0.0001f, 10.0f)]
    public float maxWaterDepth;

    [Range(0.0001f, 1.0f)]
    public float depositionConstant;

    [Range(0.001f, 1.0f)]
    public float rainNoiseFrequency = 0.25f;

    public override void ExecuteStep(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    // private ComputeBuffer texelParametersBuffer;
    // private ComputeBuffer waterPipesBuffer;

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
    }
}
