using System.Text.RegularExpressions;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using System.IO;
using System;
using GenerationPipeline;

public class RMDDispatcher : GenerationPipeline.MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

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

    private const int groupSize = 4;
    public override void StepInitialization(PipelineContext pipelineContext)
    {
    }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;
        int textureSize = numLinearThreads * texelsPerThreadDomain;

        RenderTexture noiseRenderTexture = new RenderTexture(textureSize, textureSize, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        noiseRenderTexture.Create();

        Assert.IsTrue(noiseRenderTexture.IsCreated());

        int initializationKernelIdx = shaderToDispatch.FindKernel("IntializeTexture");
        int transition12KernelIdx = shaderToDispatch.FindKernel("Transition12");
        int transition21KernelIdx = shaderToDispatch.FindKernel("Transition21");
        int extraNoiseKernel = shaderToDispatch.FindKernel("AddExtraNoise");

        shaderToDispatch.SetInt("threadDomainTexelWidth", texelsPerThreadDomain);
        shaderToDispatch.SetInt("threadSubdomainsX", numLinearThreads);
        shaderToDispatch.SetInt("threadSubdomainsY", numLinearThreads);
        shaderToDispatch.SetFloat("worleyFrequency", worleyFrequency);
        shaderToDispatch.SetFloat("perlinFrequency", perlinFrequency);

        shaderToDispatch.SetTexture(initializationKernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition12KernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition21KernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(extraNoiseKernel, "Result", noiseRenderTexture);

        System.Random rng = new System.Random();
        shaderToDispatch.SetFloats("noiseDisplacement", new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() });

        float octaveAmplitude = 1.0f;
        shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, initializationKernelIdx, new Vector3(numGroups, numGroups, 1));

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            shaderToDispatch.SetInt("texelWidthDivisionFactor", (int)math.pow(2, sub));
            shaderToDispatch.SetInt("texelWidthDivided", texelsPerThreadDomain / (int)math.pow(2, sub + 1));

            octaveAmplitude *= H;
            shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
            shaderToDispatch.SetFloats("noiseDisplacement", new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() });
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition12KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                shaderToDispatch.SetFloats("noiseDisplacement", new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() });
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernel, new Vector3(numGroups, numGroups, 1));
            }

            octaveAmplitude *= H;
            shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
            shaderToDispatch.SetFloats("noiseDisplacement", new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() });
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition21KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                shaderToDispatch.SetFloats("noiseDisplacement", new float[] { (float)rng.NextDouble(), (float)rng.NextDouble() });
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernel, new Vector3(numGroups, numGroups, 1));
            }
        }

        //interpolate the custom texture into the target one
        pipelineContext.AppendTextureCopyToCommandBuffer(noiseRenderTexture, pipelineContext.intermediateHeightmap);
    }

    public override void StepConclusion(PipelineContext pipelineContext)
    {
    }

    public override InputExpectations GetStepExpectations()
    {
        return InputExpectations.None;
    }
}
