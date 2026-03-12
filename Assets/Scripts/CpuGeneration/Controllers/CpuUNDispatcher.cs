using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using CpuGenerationPipeline;
using System;

public class CpuUNDispatcher : CpuMonoPipelineStep
{

    [Range(0.001f, 2.0f)]
    public float noiseFrequency = 0.001f;

    [Range(0.01f, 1.0f)]
    public float persistence = 0.2f;

    [Range(-10.0f, 10.0f)]
    public float sharpness = 0.0f;

    [Range(0.001f, 10.0f)]
    public float slopeErosion = 0.001f;

    [Range(1, 20)]
    public int numOctaves = 5;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength = 0.01f;

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
