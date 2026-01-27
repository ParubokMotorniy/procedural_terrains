using UnityEngine;
using UnityEngine.Assertions;
using GenerationPipeline;
using Unity.Mathematics;

public class UNDispatcher : GenerationPipeline.MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(0.001f, 2.0f)]
    public float noiseFrequency = 0.001f;

    [Range(0.01f, 1.0f)]
    public float persistence = 0.2f;

    [Range(-10.0f, 10.0f)]
    public float sharpness = 0.0f;

    [Range(0.01f, 10.0f)]
    public float slopeErosion = 0.01f;

    [Range(1, 20)]
    public int numOctaves = 5;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength = 0.01f;
    private const int groupSize = 32;

    public override void StepInitialization(PipelineContext pipelineContext)
    {
    }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");
        shaderToDispatch.SetTexture(kernelIdx, Shader.PropertyToID("Result"), pipelineContext.intermediateHeightmap);
        shaderToDispatch.SetInt("texelsPerThread", textureSize / numLinearThreads);
        shaderToDispatch.SetFloat("noiseFrequency", noiseFrequency);

        shaderToDispatch.SetFloat("sharpness", sharpness);
        shaderToDispatch.SetFloat("slopeErosion", slopeErosion);
        shaderToDispatch.SetFloat("perturbationStrength", perturbationStrength);
        shaderToDispatch.SetInt("numOctaves", numOctaves);
        shaderToDispatch.SetFloat("persistence", persistence);

        shaderToDispatch.Dispatch(kernelIdx, numGroups, numGroups, 1);
    }

    public override void StepConclusion(PipelineContext pipelineContext)
    {
    }

    public override InputExpectations GetStepExpectations()
    {
        return InputExpectations.None;
    }
}
