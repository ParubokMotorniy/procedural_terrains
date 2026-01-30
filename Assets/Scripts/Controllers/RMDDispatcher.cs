using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;

public class RMDDispatcher : MultiFormatPipelineStep
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

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;
        int textureSize = numLinearThreads * texelsPerThreadDomain;

        Assert.IsTrue(textureSize <= pipelineContext.GetHeightmapSize(), "RMD heightmap can lose detail when downsampled into a smaller target texture.");

        var noiseRenderTexture = new RenderTexture(textureSize, textureSize, 0)
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
        int extraNoiseKernelIdx = shaderToDispatch.FindKernel("AddExtraNoise");

        foreach (int kernelIdx in new[] { initializationKernelIdx, transition12KernelIdx, transition21KernelIdx, extraNoiseKernelIdx })
        {
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_Result, noiseRenderTexture);
        }

        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadDomainTexelWidth, texelsPerThreadDomain);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadSubdomainsX, numLinearThreads);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_threadSubdomainsY, numLinearThreads);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_worleyFrequency, worleyFrequency);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_perlinFrequency, perlinFrequency);

        var rng = new System.Random();
        float[] noiseDisp2 = new float[2];

        void SetNoiseDisplacement()
        {
            noiseDisp2[0] = (float)rng.NextDouble();
            noiseDisp2[1] = (float)rng.NextDouble();
            pipelineContext.SetUniformFloats(shaderToDispatch, PID_noiseDisplacement, noiseDisp2);
        }

        float octaveAmplitude = 1.0f;
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
        SetNoiseDisplacement();

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, initializationKernelIdx, new Vector3(numGroups, numGroups, 1));

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            int divisionFactor = (int)math.pow(2, sub);
            int divided = texelsPerThreadDomain / (int)math.pow(2, sub + 1);

            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivisionFactor, divisionFactor);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivided, divided);

            octaveAmplitude *= H;
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
            SetNoiseDisplacement();
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition12KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                SetNoiseDisplacement();
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernelIdx, new Vector3(numGroups, numGroups, 1));
            }

            octaveAmplitude *= H;
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
            SetNoiseDisplacement();
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition21KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                SetNoiseDisplacement();
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernelIdx, new Vector3(numGroups, numGroups, 1));
            }
        }

        // Interpolate/transfer into the pipeline's target heightmap.
        pipelineContext.AppendTextureCopyToCommandBuffer(noiseRenderTexture, pipelineContext.intermediateHeightmap);
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;
}
