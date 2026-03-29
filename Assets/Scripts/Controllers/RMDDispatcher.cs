using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using UnityEngine.Rendering;
using System.Threading.Tasks;
using Unity.Collections;
using Microsoft.Unity.VisualStudio.Editor;

public class RMDDispatcher : UltimatePipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(1, 16)]
    public uint numSubdivisions = 2;

    [Range(0.01f, 10.0f)]
    public float H = 0.85f;

    [SerializeField]
    public bool addExtraNoise;

    [Range(0.01f, 10.0f)]
    public float worleyFrequency;

    // [Range(0.01f, 10.0f)]
    // public float perlinFrequency;

    private readonly int[] groupSizes = new int[] { 32, 16, 8 };

    private static readonly float RANGE_THRSH_CEIL = 0.6f;
    private static readonly float RANGE_THRSH_FLOOR = 0.35f;
    private static readonly float EFFECTIVE_RANGE = 0.25f;
    private static readonly float EPSILON = 3.0e-2f;
    private static readonly float RIDGE_FALLOF_SPEED = 0.5f;

    private static readonly int PID_threadDomainTexelWidth = Shader.PropertyToID("threadDomainTexelWidth");
    private static readonly int PID_threadSubdomainsX = Shader.PropertyToID("threadSubdomainsX");
    private static readonly int PID_threadSubdomainsY = Shader.PropertyToID("threadSubdomainsY");
    private static readonly int PID_worleyFrequency = Shader.PropertyToID("worleyFrequency");
    // private static readonly int PID_perlinFrequency = Shader.PropertyToID("perlinFrequency");
    private static readonly int PID_noiseDisplacement = Shader.PropertyToID("noiseDisplacement");
    private static readonly int PID_octaveAmplitude = Shader.PropertyToID("octaveAmplitude");
    private static readonly int PID_texelWidthDivisionFactor = Shader.PropertyToID("texelWidthDivisionFactor");
    private static readonly int PID_texelWidthDivisionIteration = Shader.PropertyToID("texelWidthDivisionIteration");
    private static readonly int PID_texelWidthDivided = Shader.PropertyToID("texelWidthDivided");
    private static readonly int PID_Result = Shader.PropertyToID("Result");
    private static readonly int PID_targetDimensions = Shader.PropertyToID("targetDimensions");

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
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
        // pipelineContext.SetUniformFloat(shaderToDispatch, PID_perlinFrequency, perlinFrequency);
        pipelineContext.SetUniformInts(shaderToDispatch, PID_targetDimensions, new int[] { textureSize, textureSize });

        float octaveAmplitude = 1.0f;
        float scalingFactor = (float)math.pow(0.5, 0.5 * H);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
        pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);

        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, initializationKernelIdx, new Vector3(numGroups, numGroups, 1));

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            int divisionFactor = (int)math.pow(2, sub);
            int divided = texelsPerThreadDomain / (int)math.pow(2, sub + 1);

            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivisionFactor, divisionFactor);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivisionIteration, sub + 1);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelWidthDivided, divided);

            octaveAmplitude *= scalingFactor;
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_octaveAmplitude, octaveAmplitude);
            pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, transition12KernelIdx, new Vector3(numGroups, numGroups, 1));

            if (addExtraNoise)
            {
                pipelineContext.SetRandomFloats(shaderToDispatch, PID_noiseDisplacement);
                pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, extraNoiseKernelIdx, new Vector3(numGroups, numGroups, 1));
            }

            octaveAmplitude *= scalingFactor;
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

    public override InputExpectations GetStepExpectationsGpu() => InputExpectations.None;

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    private float SampleGaussianNoise(float2 noiseTextureIdx)
    {
        return CpuComputeUtilities.sampleGaussBoxMuller(new float2(math.min(noiseTextureIdx.x, 0.99f), noiseTextureIdx.y));
    }

    private float SampleNoise(float2 noiseTextureIdx)
    {
        // return noise.snoise(new float2(SampleGaussianNoise(noiseTextureIdx) * perlinFrequency));
        return SampleGaussianNoise(noiseTextureIdx);
    }

    private float SampleWorleyNoise(float2 noiseTextureIdx)
    {
        float2 f1_f2 = noise.cellular(noiseTextureIdx);

        if (math.abs(f1_f2.x - f1_f2.y) >= EPSILON)
        {
            return f1_f2.x;
        }

        if (f1_f2.x >= RANGE_THRSH_CEIL)
        {
            return f1_f2.x;
        }

        return (float)((math.max(f1_f2.x - RANGE_THRSH_FLOOR, 0.0) * RIDGE_FALLOF_SPEED) / EFFECTIVE_RANGE);
    }

    private void InitializeHeightmap(int textureSize, int texelsPerThreadDomain, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, float octaveAmplitude, float2 noiseDisplacement)
    {
        int numThreadDomains = textureSize / texelsPerThreadDomain;
        for (int x = 0; x < numThreadDomains; ++x)
        {
            for (int y = 0; y < numThreadDomains; ++y)
            {
                uint2 topLeftCellIdx = (uint2)(new int2(x, y) * texelsPerThreadDomain);
                float2 worleyCoordinates = (topLeftCellIdx / (float2)intermediateHeightmap.heightmapDimensions) * worleyFrequency + noiseDisplacement * worleyFrequency;
                intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, topLeftCellIdx)] = SampleWorleyNoise(worleyCoordinates) * octaveAmplitude;
            }
        }
    }

    private void AddExtraNoiseRoutine(int textureSize, NativeArray<float> nativeHeightmapArray, float octaveAmplitude, CpuPipelineContext pipelineContext)
    {
        for (int x = 0; x < textureSize; ++x)
        {
            for (int y = 0; y < textureSize; ++y)
            {
                nativeHeightmapArray[x * textureSize + y] += SampleGaussianNoise(pipelineContext.GetRandomFloats()) * octaveAmplitude;
            }
        }
    }

    private void Transition12(int textureSize, int texelsPerThreadDomain, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, int texelWidthDivided, int texelWidthDivisionFactor, float octaveAmplitude, CpuPipelineContext pipelineContext)
    {
        int numThreadDomains = textureSize / texelsPerThreadDomain;
        for (int xW = 0; xW < numThreadDomains; ++xW)
        {
            for (int yW = 0; yW < numThreadDomains; ++yW)
            {
                uint2 topLeftCellIdx = (uint2)(new int2(xW, yW) * texelsPerThreadDomain);

                for (int x = 0; x < texelWidthDivisionFactor; ++x)
                {
                    for (int y = 0; y < texelWidthDivisionFactor; ++y)
                    {
                        int2 subdivisionIdx = (int2)(topLeftCellIdx.xy + new uint2((uint)(texelWidthDivided + 2 * texelWidthDivided * x), (uint)(texelWidthDivided + 2 * texelWidthDivided * y)));

                        double averageNeighbors =
                            (intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(subdivisionIdx + new int2(texelWidthDivided, texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))] +
                             intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(subdivisionIdx + new int2(texelWidthDivided, -texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))] +
                             intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(subdivisionIdx + new int2(-texelWidthDivided, texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))] +
                             intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(subdivisionIdx + new int2(-texelWidthDivided, -texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))]) /
                            4.0;

                        intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)subdivisionIdx)] = (float)(averageNeighbors + SampleNoise(pipelineContext.GetRandomFloats()) * octaveAmplitude);
                    }
                }
            }
        }
    }

    private void Transition21(int textureSize, int texelsPerThreadDomain, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, int texelWidthDivided, int texelWidthDivisionFactor, int texelWidthDivisionIteration, float octaveAmplitude, CpuPipelineContext pipelineContext)
    {
        int numThreadDomains = textureSize / texelsPerThreadDomain;
        for (int xW = 0; xW < numThreadDomains; ++xW)
        {
            for (int yW = 0; yW < numThreadDomains; ++yW)
            {
                uint2 topLeftCellIdx = (uint2)(new int2(xW, yW) * texelsPerThreadDomain);

                int numYPasses = (int)math.pow(2, texelWidthDivisionIteration);
                int numXPasses = texelWidthDivisionFactor;

                for (uint y = 0; y < numYPasses; ++y)
                {
                    int xShiftAmount = (1 - (int)(y % 2)) * texelWidthDivided;
                    for (uint x = 0; x < numXPasses; ++x)
                    {
                        int2 diamondIdx = (int2)(topLeftCellIdx.xy + new uint2((uint)(x * 2 * texelWidthDivided + xShiftAmount), (uint)(y * texelWidthDivided)));

                        float neighborLeft = 0.0f;
                        float neighborRight = 0.0f;
                        float neighborTop = 0.0f;
                        float neighborBottom = 0.0f;

                        int numberValid = 0;

                        if (diamondIdx.x != 0)
                        {
                            neighborLeft = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(diamondIdx + new int2(-texelWidthDivided, 0), (int2)intermediateHeightmap.heightmapDimensions))];
                            numberValid++;
                        }

                        if (diamondIdx.x != intermediateHeightmap.heightmapDimensions.x - 1)
                        {
                            neighborRight = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(diamondIdx + new int2(+texelWidthDivided, 0), (int2)intermediateHeightmap.heightmapDimensions))];
                            numberValid++;
                        }

                        if (diamondIdx.y != 0)
                        {
                            neighborTop = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(diamondIdx + new int2(0, -texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))];
                            numberValid++;
                        }

                        if (diamondIdx.y != intermediateHeightmap.heightmapDimensions.y - 1)
                        {
                            neighborBottom = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(diamondIdx + new int2(0, +texelWidthDivided), (int2)intermediateHeightmap.heightmapDimensions))];
                            numberValid++;
                        }

                        intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)diamondIdx)] = ((neighborLeft +
                                               neighborRight +
                                               neighborTop +
                                               neighborBottom) /
                                              numberValid) +
                                             SampleNoise(pipelineContext.GetRandomFloats()) * octaveAmplitude;
                    }
                }
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();

        Assert.IsTrue(textureSize % texelsPerThreadDomain == 0, "Can't fit integer number of domains into the texture!");

        float octaveAmplitude = 1.0f;
        InitializeHeightmap(textureSize, texelsPerThreadDomain, intermediateHeightmap, 1.0f, pipelineContext.GetRandomFloats());

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            int texelWidthDivisionFactor = (int)math.pow(2, sub);
            int texelWidthDivided = texelsPerThreadDomain / (int)math.pow(2, sub + 1);
            int seed = texelWidthDivided + sub;

            octaveAmplitude *= H;
            Transition12(textureSize, texelsPerThreadDomain, intermediateHeightmap, texelWidthDivided, texelWidthDivisionFactor, octaveAmplitude, pipelineContext);

            if (addExtraNoise)
            {
                AddExtraNoiseRoutine(textureSize, intermediateHeightmap.nativeHeightmapArray, octaveAmplitude, pipelineContext);
            }

            octaveAmplitude *= H;
            Transition21(textureSize, texelsPerThreadDomain, intermediateHeightmap, texelWidthDivided, texelWidthDivisionFactor, sub + 1, octaveAmplitude, pipelineContext);

            if (addExtraNoise)
            {
                AddExtraNoiseRoutine(textureSize, intermediateHeightmap.nativeHeightmapArray, octaveAmplitude, pipelineContext);
            }
        }

        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu()
    {
        return CpuInputExpectations.None;
    }

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
        previousState &= ~CpuHeightmapProperties.Normalized;
    }

    public override void RenderParametersTuningGUI()
    {
        GUILayout.Label($"Num Subdivisions: {numSubdivisions}");
        float ns = GUILayout.HorizontalSlider(numSubdivisions, 1f, 16f);
        numSubdivisions = (uint)Mathf.RoundToInt(ns);

        GUILayout.Label($"H: {H:F3}");
        H = GUILayout.HorizontalSlider(H, 0.01f, 1.0f);

        addExtraNoise = GUILayout.Toggle(addExtraNoise, "Add Extra Noise");

        GUILayout.Label($"Worley Frequency: {worleyFrequency:F3}");
        worleyFrequency = GUILayout.HorizontalSlider(worleyFrequency, 0.01f, 10.0f);

        // GUILayout.Label($"Perlin Frequency: {perlinFrequency:F3}");
        // perlinFrequency = GUILayout.HorizontalSlider(perlinFrequency, 0.01f, 10.0f);
    }

    public override string GUIStepTitle() => "RMD generator";

    public override void FreeResources()
    {
    }
}
