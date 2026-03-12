using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using GenerationPipeline;
using System;

public class UNDispatcher: MonoPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

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

    private static readonly int PID_Result = Shader.PropertyToID("Result");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_noiseFrequency = Shader.PropertyToID("noiseFrequency");
    private static readonly int PID_sharpness = Shader.PropertyToID("sharpness");
    private static readonly int PID_slopeErosion = Shader.PropertyToID("slopeErosion");
    private static readonly int PID_perturbationStrength = Shader.PropertyToID("perturbationStrength");
    private static readonly int PID_numOctaves = Shader.PropertyToID("numOctaves");
    private static readonly int PID_persistence = Shader.PropertyToID("persistence");
    private static readonly int PID_randomFloats = Shader.PropertyToID("randomFloats");
    private static readonly int PID_textureDimensions = Shader.PropertyToID("textureDimensions");

    public override void ExecuteStep(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 1);
        int numLinearThreads = optimalGroupSize * numGroups;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");

        pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_Result, pipelineContext.intermediateHeightmap);

        {
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerThread, textureSize / numLinearThreads);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_noiseFrequency, noiseFrequency);
        }

        {
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_sharpness, sharpness);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_slopeErosion, slopeErosion);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_perturbationStrength, perturbationStrength);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_numOctaves, numOctaves);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_persistence, persistence);
            pipelineContext.SetRandomFloats(shaderToDispatch, PID_randomFloats);
            pipelineContext.SetUniformInts(shaderToDispatch, PID_textureDimensions, new int[] { textureSize, textureSize });
        }

        pipelineContext.AppendDispatchToCommandBuffer(
            shaderToDispatch,
            kernelIdx,
            new Vector3(numGroups, numGroups, 1)
        );
    }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }
}
