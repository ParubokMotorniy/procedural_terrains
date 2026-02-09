using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using GenerationPipeline;
using System.Reflection;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

public class FFTDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0, 3)]
    public int groupScaleFactor = 0;
    [Range(0.01f, 5.0f)]
    public float fractalDimension = 0.1f;

    private const int groupSize = 32;
    private ComputeBuffer coefficientsBuffer;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_coefficientsBuffer = Shader.PropertyToID("coefficients");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_halfHeightmapDimensions = Shader.PropertyToID("halfHeightmapDimensions");

    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_invTexelsPerThread = Shader.PropertyToID("invTexelsPerThread");

    private static readonly int PID_fractalDimension = Shader.PropertyToID("fractalDimension");


    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");

        coefficientsBuffer = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(coefficientsBuffer.IsValid());

        int coefficientGeneratorKernelIdx = shaderToDispatch.FindKernel("CoefficientGenerator");
        int inverseFFTKernelIdx = shaderToDispatch.FindKernel("InverseFFT");
        // int clearerKernelIdx = shaderToDispatch.FindKernel("Clearer");

        int[] kernels =
        {
            coefficientGeneratorKernelIdx,
            inverseFFTKernelIdx,
            // clearerKernelIdx
        };

        foreach (int kernelIdx in kernels)
        {
            pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(shaderToDispatch, kernelIdx, PID_coefficientsBuffer, coefficientsBuffer);
        }

        // Uniforms
        int invTexelsPerThread = textureSize / numLinearThreads;
        Assert.IsTrue(invTexelsPerThread % 2 == 0, "Coefficients texels must be distributed among threads evenly!");
        int texelsPerThread = invTexelsPerThread / 2;

        Debug.Log(textureSize);
        Debug.Log(invTexelsPerThread);
        Debug.Log(texelsPerThread);

        pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerThread, texelsPerThread);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_invTexelsPerThread, invTexelsPerThread);
        pipelineContext.SetUniformInts(shaderToDispatch, PID_heightmapDimensions, new int[] { textureSize, textureSize });
        pipelineContext.SetUniformInts(shaderToDispatch, PID_halfHeightmapDimensions, new int[] { textureSize / 2, textureSize / 2 });
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_fractalDimension, fractalDimension);

        // pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, clearerKernelIdx, new Vector3(numGroups, numGroups, 1));
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, coefficientGeneratorKernelIdx, new Vector3(numGroups, numGroups, 1));
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, inverseFFTKernelIdx, new Vector3(numGroups, numGroups, 1));
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;
}
