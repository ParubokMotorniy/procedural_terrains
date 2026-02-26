using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;
using UnityEngine.Rendering;

public class RMDDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

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

    private readonly int[] groupSizes = new int[] { 32, 16, 8 };

    private static readonly int PID_threadDomainTexelWidth = Shader.PropertyToID("threadDomainTexelWidth");
    private static readonly int PID_threadSubdomainsX = Shader.PropertyToID("threadSubdomainsX");
    private static readonly int PID_threadSubdomainsY = Shader.PropertyToID("threadSubdomainsY");
    private static readonly int PID_worleyFrequency = Shader.PropertyToID("worleyFrequency");
    private static readonly int PID_perlinFrequency = Shader.PropertyToID("perlinFrequency");
    private static readonly int PID_noiseDisplacement = Shader.PropertyToID("noiseDisplacement");
    private static readonly int PID_octaveAmplitude = Shader.PropertyToID("octaveAmplitude");
    private static readonly int PID_texelWidthDivisionFactor = Shader.PropertyToID("texelWidthDivisionFactor");
    private static readonly int PID_texelWidthDivided = Shader.PropertyToID("texelWidthDivided");
    private static readonly int PID_Result = Shader.PropertyToID("Result");
    private static readonly int PID_targetDimensions = Shader.PropertyToID("targetDimensions");

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        int numLinearThreads = textureSize / texelsPerThreadDomain;
        Assert.IsTrue(textureSize % texelsPerThreadDomain == 0, "Can't fit integer number of domains into the texture!");

        int numGroups = 0;
        int groupSize = 0;
        foreach (int candidateGroupSize in groupSizes)
        {
            if (numLinearThreads % candidateGroupSize == 0)
            {
                numGroups = numLinearThreads / candidateGroupSize;
                groupSize = candidateGroupSize;
            }
        }

        Assert.IsTrue(numGroups != 0 && groupSize != 0, "Underoccupied groups requested!");

        var appropriateShaderKeyword = new LocalKeyword(shaderToDispatch, "GROUP_" + groupSize);
        pipelineContext.SetKeyword(shaderToDispatch, ref appropriateShaderKeyword, true);

        int initializationKernelIdx = shaderToDispatch.FindKernel("IntializeTexture");
        int transition12KernelIdx = shaderToDispatch.FindKernel("Transition12");
        int transition21KernelIdx = shaderToDispatch.FindKernel("Transition21");
        int extraNoiseKernelIdx = shaderToDispatch.FindKernel("AddExtraNoise");

        foreach (int kernelIdx in new[] { initializationKernelIdx, transition12KernelIdx, transition21KernelIdx, extraNoiseKernelIdx })
        {
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_Result, pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadDomainTexelWidth, texelsPerThreadDomain);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadSubdomainsX, numLinearThreads);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadSubdomainsY, numLinearThreads);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_worleyFrequency, worleyFrequency);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_perlinFrequency, perlinFrequency);
        pipelineContext.SetUniformInts(shaderToDispatch, PID_targetDimensions, new int[] { textureSize, textureSize });

        float octaveAmplitude = 1.0f;
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
        pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, initializationKernelIdx, new Vector3(numGroups, numGroups, 1));

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            int divisionFactor = (int)math.pow(2, sub);
            int divided = texelsPerThreadDomain / (int)math.pow(2, sub + 1);

            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivisionFactor, divisionFactor);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivided, divided);

            octaveAmplitude *= H;
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
            pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition12KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernelIdx, new Vector3(numGroups, numGroups, 1));
            }

            octaveAmplitude *= H;
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
            pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition21KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernelIdx, new Vector3(numGroups, numGroups, 1));
            }
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }
}
