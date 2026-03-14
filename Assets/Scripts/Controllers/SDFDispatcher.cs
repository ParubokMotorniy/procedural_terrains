using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using System.Threading.Tasks;

public class SDFDispatcher: UltimatePipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    //both are implicitly cleared
    private ComputeBuffer floodingBuffer1;
    private ComputeBuffer floodingBuffer2;
    private ComputeBuffer inputContinentHeightmap;

    private static readonly int PID_buffer1 = Shader.PropertyToID("floodingBuffer1");
    private static readonly int PID_buffer2 = Shader.PropertyToID("floodingBuffer2");
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
    private static readonly int PID_randomFloats = Shader.PropertyToID("randomFloats");

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numLinearGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 1);
        int numLinearThreads = optimalGroupSize * numLinearGroups;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");

        {
            int neededBufferSize = textureSize * textureSize;
            if (inputContinentHeightmap is null || !inputContinentHeightmap.IsValid() || inputContinentHeightmap.count != neededBufferSize)
            {
                inputContinentHeightmap = new ComputeBuffer(neededBufferSize, sizeof(float));
                Assert.IsTrue(inputContinentHeightmap.IsValid());
            }
        }

        {
            int neededBufferSize = textureSize * textureSize * 2;
            if (floodingBuffer1 is null || !floodingBuffer1.IsValid() || floodingBuffer1.count != neededBufferSize)
            {
                floodingBuffer1 = new ComputeBuffer(neededBufferSize, sizeof(float));
                Assert.IsTrue(floodingBuffer1.IsValid());
            }
            if (floodingBuffer2 is null || !floodingBuffer2.IsValid() || floodingBuffer2.count != neededBufferSize)
            {
                floodingBuffer2 = new ComputeBuffer(neededBufferSize, sizeof(float));
                Assert.IsTrue(floodingBuffer2.IsValid());
            }
        }

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
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_buffer1, floodingBuffer1);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_buffer2, floodingBuffer2);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_inputTexture, inputContinentHeightmap);
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
        pipelineContext.SetRandomFloats(shaderToDispatch, PID_randomFloats);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            pipelineContext.SetUniformInt(shaderToDispatch, PID_currentSourceBuffer, currentReadBuffer);
        };

        Vector3 dispatchGroups = new Vector3(numLinearGroups, numLinearGroups, 1);

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
        pipelineContext.RunInternalNormalization();

        // Heightmap postprocessing
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, sDFPostprocessorKernel, dispatchGroups);
    }

    public override InputExpectations GetStepExpectationsGpu() => InputExpectations.None;

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        throw new NotImplementedException();
    }

    public override CpuInputExpectations GetStepExpectationsCpu()
    {
        throw new NotImplementedException();
    }

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
        throw new NotImplementedException();
    }
}
