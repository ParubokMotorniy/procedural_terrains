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

    private const int groupSize = 16;

    public override void StepInitialization(PipelineContext pipelineContext)
    {
    }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        //TODO: the texture can actually be reworked to be a computebuffer
        RenderTexture coastlineTexture = new RenderTexture(textureSize, textureSize, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        {
            coastlineTexture.Create();

            Assert.IsTrue(coastlineTexture.IsCreated());
        }

        ComputeBuffer buffer1 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer1.IsValid());

        ComputeBuffer buffer2 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer2.IsValid());

        int maskToSeedBufferKernelIdx = shaderToDispatch.FindKernel("MaskToSeedBuffer");
        int floodingStepKernelIdx = shaderToDispatch.FindKernel("FloodingStep");
        int seedBufferToHieghtmapKernelIdx = shaderToDispatch.FindKernel("SeedBufferToHieghtmap");
        int coastlineGeneratorKernel = shaderToDispatch.FindKernel("CoastlineGenerator");
        int sDFPostprocessorKernel = shaderToDispatch.FindKernel("SDFPostprocessor");

        foreach (int kernelIdx in new int[] { maskToSeedBufferKernelIdx, floodingStepKernelIdx, seedBufferToHieghtmapKernelIdx, coastlineGeneratorKernel, sDFPostprocessorKernel })
        {
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, Shader.PropertyToID("buffer1"), buffer1);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, Shader.PropertyToID("buffer2"), buffer2);
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, Shader.PropertyToID("inputTexture"), coastlineTexture);
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, Shader.PropertyToID("outputTexture"), pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("texelsPerThread"), textureSize / numLinearThreads);
        pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("bufferSideLength"), textureSize);
        pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("maskBorderWidth"), (int)(textureSize * 0.2f)); //fix at 20%
        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        pipelineContext.SetUniformFloat(shaderToDispatch, Shader.PropertyToID("shoreBaseHeight"), maxDistanceToSeed * 0.005f); //fix at 0.5%
        pipelineContext.SetUniformFloat(shaderToDispatch, Shader.PropertyToID("shoreDistanceThreshold"), textureSize * 0.1f); //fit at 10%
        pipelineContext.SetUniformFloat(shaderToDispatch, Shader.PropertyToID("simplexFrequency"), baseSimplexFrequency);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("currentSourceBuffer"), currentReadBuffer);
        };

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, coastlineGeneratorKernel, new Vector3(numGroups, numGroups, 1));
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, maskToSeedBufferKernelIdx, new Vector3(numGroups, numGroups, 1));
        while (currentFloodStep > 1)
        {
            updateSourceBuffer();
            currentFloodStep /= 2;
            pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("floodStepSize"), currentFloodStep);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, floodingStepKernelIdx, new Vector3(numGroups, numGroups, 1));
        }
        //extra iteration to improve SDF accuracy
        {
            updateSourceBuffer();
            pipelineContext.SetUniformInt(shaderToDispatch, Shader.PropertyToID("floodStepSize"), 1);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, floodingStepKernelIdx, new Vector3(numGroups, numGroups, 1));
        }
        //distance computation
        {
            updateSourceBuffer();
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, seedBufferToHieghtmapKernelIdx, new Vector3(numGroups, numGroups, 1));
        }

        //normalization of the output SDF texture
        RunInternalNormalization(pipelineContext);

        // heightmap postprocessing
        {
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, sDFPostprocessorKernel, new Vector3(numGroups, numGroups, 1));
        }


        //TODO: these will become obsolete if we decide to play with endless generation
        // buffer1.Release();
        // buffer2.Release();
        // coastlineTexture.Release();
    }

    public override void StepConclusion(PipelineContext pipelineContext)
    {

    }

    public override InputExpectations GetStepExpectations()
    {
        return InputExpectations.None;
    }
}
