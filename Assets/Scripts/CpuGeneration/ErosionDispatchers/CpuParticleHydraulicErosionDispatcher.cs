using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CpuGenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class CpuParticleHydraulicErosionDispatcher : CpuMonoPipelineStep
{

    [Range(7, 32)]
    public int numSimultaneousParticles = 10;

    [Range(0.01f, 1.0f)]
    public float inertia = 0.1f;

    [Range(0.01f, 100.0f)]
    public float capacity = 10.0f;

    [Range(0.0001f, 1.0f)]
    public float minSlope = 0.01f;

    [Range(0.01f, 1.0f)]
    public float deposition = 0.7f;

    [Range(0.01f, 1.0f)]
    public float erosion = 0.35f;

    [Range(0.01f, 1.0f)]
    public float gravity = 0.4f;

    [Range(0.01f, 1.0f)]
    public float evaporation = 0.02f;

    [Range(0.01f, 0.45f)]
    public float waterDeathThreshold = 0.05f;

    [Range(1, 3)]
    public int erosionNeighborhood = 1;

    [Range(1, 1000)]
    public int numSimulationSteps = 1;

    [Range(1, 10)]
    public int numSimulationWaves = 1;

    [Range(0.001f, 10.0f)]
    public float rainNoiseFrequency = 1.0f;

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    // [StructLayout(LayoutKind.Sequential)]
    // private struct ErosionParticle
    // {
    //     Vector2 pos;
    //     Vector2 dir;
    //     float vel;
    //     float w;
    //     float s;
    // };

    // private ComputeBuffer particlesBuffer;

    // struct TexelPipes
    // {
    //     Vector3 inPipes1;
    //     Vector3 inPipes2;
    //     Vector3 inPipes3;
    //     Vector3 outPipes1;
    //     Vector3 outPipes2;
    //     Vector3 outPipes3;
    // };

    // private ComputeBuffer pipesBuffer;


    public override InputExpectations GetStepExpectationsCpu()
        => InputExpectations.HeightMapNormalized;
    public override void UpdateHeightmapStateCpu(ref HeightmapProperties previousState)
    { }
}
