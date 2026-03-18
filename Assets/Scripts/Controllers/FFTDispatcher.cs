using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using System.Threading.Tasks;
using Unity.Collections;

public class FFTDispatcher : UltimatePipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

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
    private float2[,] cpuCoefficientsBuffer;

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


    public override void ExecuteStepGpu(PipelineContext pipelineContext)
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
        var (optimalGroupSize, numGenerationGroups) = GenerationUtilities.GetOptimalNumberOfGroups(math.min(coefficientsBufferSizeX, coefficientsBufferSizeY), new int[] { generationGroupSize }, Int32.MaxValue, 1);
        int numLinearThreads = optimalGroupSize * numGenerationGroups;

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

    public override InputExpectations GetStepExpectationsGpu() => InputExpectations.None;

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    float2 GetEffectiveCoefficient(uint2 mathematicalCoordinates, float2[,] coefficients, uint2 heightmapDimensions)
    {
        int sizeX = coefficients.GetLength(0);
        int sizeY = coefficients.GetLength(1);

        if (mathematicalCoordinates.x < sizeX && mathematicalCoordinates.y < sizeY)
        {
            return coefficients[mathematicalCoordinates.x, mathematicalCoordinates.y];
        }

        uint2 targetSymmetricalCoefficient = (heightmapDimensions - new uint2(1, 1)) - mathematicalCoordinates;

        if (mathematicalCoordinates.x == 0)
        {
            targetSymmetricalCoefficient.x = 0;
        }

        if (mathematicalCoordinates.y == 0)
        {
            targetSymmetricalCoefficient.y = 0;
        }

        targetSymmetricalCoefficient.x = math.min(targetSymmetricalCoefficient.x, (uint)(sizeX - 1));
        targetSymmetricalCoefficient.y = math.min(targetSymmetricalCoefficient.y, (uint)(sizeY - 1));

        float2 computedCoefficient = coefficients[targetSymmetricalCoefficient.x, targetSymmetricalCoefficient.y];
        computedCoefficient.y = -computedCoefficient.y;
        return computedCoefficient;
    }

    void CoefficientGenerator(float2[,] coefficients, CpuPipelineContext pipelineContext)
    {
        int sizeX = coefficients.GetLength(0);
        int sizeY = coefficients.GetLength(1);

        float exponent = -(fractalDimension + 1.0f) * 0.5f;

        for (uint tX = 0; tX < sizeX; ++tX)
        {
            for (uint tY = 0; tY < sizeY; ++tY)
            {
                uint2 coefficientCoordinates = new uint2(tX, tY);

                float2 randomUniform = pipelineContext.GetRandomFloats();
                float randomGaussian = CpuComputeUtilities.sampleGaussBoxMuller(randomUniform);

                float randomPhase = CpuComputeUtilities.TWO_PI * randomUniform.x;
                float randomMagnitude = 0.0f;

                if (coefficientCoordinates.x != 0 || coefficientCoordinates.y != 0)
                {
                    float2 c = (float2)coefficientCoordinates;
                    float d2 = math.dot(c, c);
                    d2 = math.max(d2, CpuComputeUtilities.EPS);

                    randomMagnitude = math.pow(d2, exponent) * randomGaussian;
                }

                float2 baseCoefficient = randomMagnitude * new float2(math.cos(randomPhase), math.sin(randomPhase));
                coefficients[tX, tY] = baseCoefficient;
            }
        }
    }

    void InverseFFT(float2[,] coefficients, uint2 heightmapDimensions, NativeArray<float> nativeHeightmapArray)
    {
        int coeffSizeX = coefficients.GetLength(0);
        int coeffSizeY = coefficients.GetLength(1);

        float2 invSize = 1.0f / (float2)heightmapDimensions;

        for (uint hX = 0; hX < heightmapDimensions.x; ++hX)
        {
            for (uint hY = 0; hY < heightmapDimensions.y; ++hY)
            {
                float2 exponentRatioPrecompute = new float2(hX, hY) * invSize;

                float finalHeight = 0.0f;
                float localErrorOut = 0.0f;
                float localError = 0.0f;

                for (uint x = 0; x < coeffSizeX; ++x)
                {
                    float fx = x;
                    for (uint y = 0; y < coeffSizeY; ++y)
                    {
                        float fy = y;

                        uint2 coefficientCoords = new uint2(x, y);
                        float2 coefficient = GetEffectiveCoefficient(coefficientCoords, coefficients, heightmapDimensions);

                        float ePower = CpuComputeUtilities.TWO_PI * (fx * exponentRatioPrecompute.x + fy * exponentRatioPrecompute.y);

                        float2 complexExponent = new float2(math.cos(ePower), math.sin(ePower));

                        float product = coefficient.x * complexExponent.x - coefficient.y * complexExponent.y;

                        finalHeight = CpuComputeUtilities.accurateSum(finalHeight, product, out localErrorOut);
                        localError += localErrorOut;
                    }
                }

                nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(heightmapDimensions, new uint2(hX, hY))] = finalHeight + localError;
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        uint2 heightmapDimensions = new uint2((uint)textureSize, (uint)textureSize);

        int actualCoefficientsComputed = (int)math.pow(2, (int)math.floor(math.log2(fracCoefficientsConsidered * textureSize)));
        int coefficientsBufferSizeX = math.min(actualCoefficientsComputed, textureSize / 2);
        int coefficientsBufferSizeY = actualCoefficientsComputed;

        if (cpuCoefficientsBuffer is null || cpuCoefficientsBuffer.GetLength(0) != coefficientsBufferSizeX || cpuCoefficientsBuffer.GetLength(1) != coefficientsBufferSizeY)
        {
            cpuCoefficientsBuffer = new float2[coefficientsBufferSizeX, coefficientsBufferSizeY];
        }

        CoefficientGenerator(cpuCoefficientsBuffer, pipelineContext);

        InverseFFT(cpuCoefficientsBuffer, heightmapDimensions, pipelineContext.intermediateHeightmap.GetRawTextureData<float>());

        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu() => CpuInputExpectations.None;

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
        previousState &= ~CpuHeightmapProperties.Normalized;

    }
}
