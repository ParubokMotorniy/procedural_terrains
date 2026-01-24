using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;

public class SDFDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(1, 8)]
    public int groupScaleFactor = 1;

    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    private const int groupSize = 16;

    public override void StepInitialization(PipelineContext pipelineContext)
    {
    }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();

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
            shaderToDispatch.SetBuffer(kernelIdx, "buffer1", buffer1);
            shaderToDispatch.SetBuffer(kernelIdx, "buffer2", buffer2);
            shaderToDispatch.SetTexture(kernelIdx, "inputTexture", coastlineTexture);
            shaderToDispatch.SetTexture(kernelIdx, "outputTexture", pipelineContext.intermediateHeightmap);
        }

        shaderToDispatch.SetInt("texelsPerThread", textureSize / (groupSize * groupScaleFactor));
        shaderToDispatch.SetInt("bufferSideLength", textureSize);
        shaderToDispatch.SetInt("maskBorderWidth", (int)(textureSize * 0.2f)); //fix at 20%
        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        shaderToDispatch.SetFloat("shoreBaseHeight", maxDistanceToSeed * 0.005f); //fix at 0.5%
        shaderToDispatch.SetFloat("shoreDistanceThreshold", textureSize * 0.1f); //fit at 10%
        shaderToDispatch.SetFloat("simplexFrequency", baseSimplexFrequency);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            shaderToDispatch.SetInt("currentSourceBuffer", currentReadBuffer);
        };

        shaderToDispatch.Dispatch(coastlineGeneratorKernel, groupScaleFactor, groupScaleFactor, 1);
        shaderToDispatch.Dispatch(maskToSeedBufferKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        while (currentFloodStep > 1)
        {
            updateSourceBuffer();
            currentFloodStep /= 2;
            shaderToDispatch.SetInt("floodStepSize", currentFloodStep);
            shaderToDispatch.Dispatch(floodingStepKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }
        //extra iteration to improve SDF accuracy
        {
            updateSourceBuffer();
            shaderToDispatch.SetInt("floodStepSize", 1);
            shaderToDispatch.Dispatch(floodingStepKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }
        //distance computation
        {
            updateSourceBuffer();
            shaderToDispatch.Dispatch(seedBufferToHieghtmapKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }

        //normalization of the output SDF texture
        RunInternalNormalization(pipelineContext);

        // heightmap postprocessing
        {
            shaderToDispatch.Dispatch(sDFPostprocessorKernel, groupScaleFactor, groupScaleFactor, 1);
        }   

        RenderTextureDumper.SaveRFloatToExr(coastlineTexture, "test_coastline.exr");

        buffer1.Release();
        buffer2.Release();
        coastlineTexture.Release();

        Debug.Log("Terrain has been regenerated!");
    }

    public override void StepConclusion(PipelineContext pipelineContext)
    {

    }

    public override InputExpectations GetStepExpectations()
    {
        return InputExpectations.None;
    }
}
