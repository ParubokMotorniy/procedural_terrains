using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;
using UnityEngine.Rendering;

public class CpuRMDDispatcher : CpuMonoPipelineStep
{
    [Range(1, 16)]
    public uint numSubdivisions = 2;

    [Range(0.01f, 1.0f)]
    public float H = 0.85f;

    [SerializeField]
    public bool addExtraNoise;

    [Range(0.01f, 10.0f)]
    public float worleyFrequency;

    [Range(0.01f, 10.0f)]
    public float perlinFrequency;

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
