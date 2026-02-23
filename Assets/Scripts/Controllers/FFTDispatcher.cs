using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;

public class FFTDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(4, 8)]
    public int inverseGroupScaleFactor = 4;

    [Range(0.01f, 5.0f)]
    public float fractalDimension = 0.1f;

    [Range(0.01f, 1.0f)]
    public float fracCoefficientsConsidered = 0.01f;

    private const int generationGroupSize = 16;
    private const int inverseGroupSize = 16;

    //implicitly cleared
    private ComputeBuffer coefficientsBuffer;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_coefficientsBuffer = Shader.PropertyToID("coefficients");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_coefficientmapDimensions = Shader.PropertyToID("coefficientmapDimensions");

    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_invTexelsPerThread = Shader.PropertyToID("invTexelsPerThread");
    private static readonly int PID_texelsPerInverseGroup = Shader.PropertyToID("texelsPerInverseGroup");
    private static readonly int PID_inverseDispatchGroupOffset = Shader.PropertyToID("inverseDispatchGroupOffset");

    private static readonly int PID_fractalDimension = Shader.PropertyToID("fractalDimension");
    private static readonly int PID_randomSeeds = Shader.PropertyToID("randomSeeds");


    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int actualCoefficientsComputed = (int)math.pow(2, (int)math.floor(math.log2(fracCoefficientsConsidered * textureSize)));

        int coefficientsBufferSizeX = math.min(actualCoefficientsComputed, textureSize / 2);
        int coefficientsBufferSizeY = actualCoefficientsComputed;

        int neededBufferSize = coefficientsBufferSizeX * coefficientsBufferSizeY * 2;
        if (coefficientsBuffer is null || !coefficientsBuffer.IsValid() || coefficientsBuffer.count != neededBufferSize)
        {
            coefficientsBuffer = new ComputeBuffer(neededBufferSize, sizeof(float));
            Assert.IsTrue(coefficientsBuffer.IsValid());
        }

        int coefficientGeneratorKernelIdx = shaderToDispatch.FindKernel("CoefficientGenerator");
        int inverseFFTKernelIdx = shaderToDispatch.FindKernel("InverseFFT");

        int[] kernels =
        {
            coefficientGeneratorKernelIdx,
            inverseFFTKernelIdx,
        };

        foreach (int kernelIdx in kernels)
        {
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_coefficientsBuffer, coefficientsBuffer);
        }

        // uniforms
        int numGenerationGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = generationGroupSize * numGenerationGroups;
        int texelsPerThreadX = coefficientsBufferSizeX / numLinearThreads;
        int texelsPerThreadY = coefficientsBufferSizeY / numLinearThreads;

        int numInverseGroups = (int)math.pow(2, inverseGroupScaleFactor);
        int invTexelsPerThread = actualCoefficientsComputed / inverseGroupSize;
        int numTexelsPerInverseGroup = textureSize / numInverseGroups;

        Assert.IsTrue(coefficientsBufferSizeX % numLinearThreads == 0 && coefficientsBufferSizeX % numLinearThreads == 0 && texelsPerThreadX != 0 && texelsPerThreadY != 0, "Generation texels must be distributed among threads evenly!");
        Assert.IsTrue(actualCoefficientsComputed % inverseGroupSize == 0 && textureSize % numInverseGroups == 0 && invTexelsPerThread != 0, "Inverse texels must be distributed among threads evenly!");

        {
            pipelineContext.SetUniformInts(shaderToDispatch, PID_texelsPerThread, new int[] { texelsPerThreadX, texelsPerThreadY });
            pipelineContext.SetUniformInt(shaderToDispatch, PID_invTexelsPerThread, invTexelsPerThread);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerInverseGroup, numTexelsPerInverseGroup);

            pipelineContext.SetUniformInts(shaderToDispatch, PID_heightmapDimensions, new int[] { textureSize, textureSize });
            pipelineContext.SetUniformInts(shaderToDispatch, PID_coefficientmapDimensions, new int[] { coefficientsBufferSizeX, coefficientsBufferSizeY });

            pipelineContext.SetUniformFloat(shaderToDispatch, PID_fractalDimension, fractalDimension);
            pipelineContext.SetRandomInts(shaderToDispatch, PID_randomSeeds);
        }

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, coefficientGeneratorKernelIdx, new Vector3(numGenerationGroups, numGenerationGroups, 1));

        for (int dX = 0; dX < numTexelsPerInverseGroup; ++dX)
        {
            for (int dY = 0; dY < numTexelsPerInverseGroup; ++dY)
            {
                pipelineContext.SetUniformInts(shaderToDispatch, PID_inverseDispatchGroupOffset, new int[] { dX, dY });
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, inverseFFTKernelIdx, new Vector3(numInverseGroups, numInverseGroups, 1));
            }
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;
}
