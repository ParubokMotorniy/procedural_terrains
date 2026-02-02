using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;

public class SDFDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    private const int groupSize = 32;

    private static readonly int PID_buffer1 = Shader.PropertyToID("buffer1");
    private static readonly int PID_buffer2 = Shader.PropertyToID("buffer2");
    private static readonly int PID_inputTexture = Shader.PropertyToID("inputTexture");
    private static readonly int PID_outputTexture = Shader.PropertyToID("outputTexture");

    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_bufferSideLength = Shader.PropertyToID("bufferSideLength");
    private static readonly int PID_maskBorderWidth = Shader.PropertyToID("maskBorderWidth");
    private static readonly int PID_shoreBaseHeight = Shader.PropertyToID("shoreBaseHeight");
    private static readonly int PID_shoreDistanceThreshold = Shader.PropertyToID("shoreDistanceThreshold");
    private static readonly int PID_simplexFrequency = Shader.PropertyToID("simplexFrequency");

    private static readonly int PID_currentSourceBuffer = Shader.PropertyToID("currentSourceBuffer");
    private static readonly int PID_floodStepSize = Shader.PropertyToID("floodStepSize");

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        // TODO: the texture can actually be reworked to be a computebuffer
        RenderTexture coastlineTexture = new RenderTexture(textureSize, textureSize, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        {coastlineTexture.Create();
        Assert.IsTrue(coastlineTexture.IsCreated());}

        ComputeBuffer buffer1 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer1.IsValid());

        ComputeBuffer buffer2 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer2.IsValid());

        int maskToSeedBufferKernelIdx = shaderToDispatch.FindKernel("MaskToSeedBuffer");
        int floodingStepKernelIdx = shaderToDispatch.FindKernel("FloodingStep");
        int seedBufferToHieghtmapKernelIdx = shaderToDispatch.FindKernel("SeedBufferToHieghtmap");
        int coastlineGeneratorKernel = shaderToDispatch.FindKernel("CoastlineGenerator");
        int sDFPostprocessorKernel = shaderToDispatch.FindKernel("SDFPostprocessor");

        int[] kernels =
        {
            maskToSeedBufferKernelIdx,
            floodingStepKernelIdx,
            seedBufferToHieghtmapKernelIdx,
            coastlineGeneratorKernel,
            sDFPostprocessorKernel
        };

        foreach (int kernelIdx in kernels)
        {
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_buffer1, buffer1);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_buffer2, buffer2);
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_inputTexture, coastlineTexture);
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_outputTexture, pipelineContext.intermediateHeightmap);
        }

        // Uniforms
        pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerThread, textureSize / numLinearThreads);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_bufferSideLength, textureSize);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_maskBorderWidth, (int)(textureSize * 0.2f)); // fix at 20%

        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_shoreDistanceThreshold, textureSize * 0.1f);   // fix at 10%
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_shoreBaseHeight, maxDistanceToSeed * 0.005f); // fix at 0.5%
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_simplexFrequency, baseSimplexFrequency);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            pipelineContext.SetUniformInt(shaderToDispatch, PID_currentSourceBuffer, currentReadBuffer);
        };

        Vector3 dispatchGroups = new Vector3(numGroups, numGroups, 1);

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, coastlineGeneratorKernel, dispatchGroups);
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, maskToSeedBufferKernelIdx, dispatchGroups);

        while (currentFloodStep > 1)
        {
            updateSourceBuffer();
            currentFloodStep /= 2;
            pipelineContext.SetUniformInt(shaderToDispatch, PID_floodStepSize, currentFloodStep);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, floodingStepKernelIdx, dispatchGroups);
        }

        // Extra iteration to improve SDF accuracy
        updateSourceBuffer();
        pipelineContext.SetUniformInt(shaderToDispatch, PID_floodStepSize, 1);
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, floodingStepKernelIdx, dispatchGroups);

        // Distance computation
        updateSourceBuffer();
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, seedBufferToHieghtmapKernelIdx, dispatchGroups);

        // Normalization of the output SDF texture
        RunInternalNormalization(pipelineContext);

        // Heightmap postprocessing
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, sDFPostprocessorKernel, dispatchGroups);

        // TODO: these will become obsolete if we decide to play with endless generation
        // buffer1.Release();
        // buffer2.Release();
        // coastlineTexture.Release();
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;
}
