using CpuGenerationPipeline;
using GpuGenerationPipeline;
using UnityEngine.Assertions;
using Unity.Mathematics;
using UnityEngine;
using System;
using System.Threading.Tasks;
using Unity.Collections;

public class ThermalErosionDispatcher : UltimatePipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(0.01f, 1.0f)]
    public float distributionCoefficient = 0.75f;

    [Range(0.001f, 1.0f)]
    public float talusThreshold = 0.15f;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_permuteACore = Shader.PropertyToID("permuteACore");
    private static readonly int PID_permuteAStripH = Shader.PropertyToID("permuteAStripH");
    private static readonly int PID_permuteAStripV = Shader.PropertyToID("permuteAStripV");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_distributionCoefficient = Shader.PropertyToID("distributionCoefficient");
    private static readonly int PID_talusThreshold = Shader.PropertyToID("talusThreshold");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_iterationIdx = Shader.PropertyToID("iterationIdx");
    private static readonly float BIG_FLOAT = 3.402823466e+38f;

    public override InputExpectations GetStepExpectationsGpu()
        => InputExpectations.HeightMapNormalized;

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 4);

        int numLinearThreads = optimalGroupSize * numGroups;
        int texelsPerThread = textureSize / numLinearThreads;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");
        Assert.IsTrue(texelsPerThread >= 4, "A thread must have at least 4 texels to porcess");

        // modular affine permutation
        int texelsPerThreadSquared = (int)math.pow(texelsPerThread - 2, 2);
        int permuteACore = GenerationUtilities.ComputeCoprime(texelsPerThreadSquared, 17);
        int permuteAStripH = GenerationUtilities.ComputeCoprime(2 * texelsPerThread, 29);
        int permuteAStripV = GenerationUtilities.ComputeCoprime(2 * (texelsPerThread - 2), 35);

        int coreKernelIdx = erosionComputeShader.FindKernel("ThermalCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("ThermalBorderEroder");

        foreach (int kernelIdx in new[] { coreKernelIdx, borderKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, texelsPerThread);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteACore, permuteACore);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripH, permuteAStripH);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripV, permuteAStripV);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_distributionCoefficient, distributionCoefficient);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_talusThreshold, talusThreshold);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });

        var dispatchGroups = new Vector3(numGroups, numGroups, 1);
        for (int i = 0; i < erosionIterationLimit; ++i)
        {
            pipelineContext.SetUniformInt(erosionComputeShader, PID_iterationIdx, i + 1);

            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
        }
    }

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
    }

    // diagonal texels are further away from the kernel center, and so we assume they receive less material than direct neighbors
    static readonly float[,] heightDistributionDistanceCoefficients =
    {
        {0.85f, 1.0f, 0.85f},
        {1.0f, 0.0f, 1.0f},
        {0.85f, 1.0f, 0.85f}};

    float computeSingleHeightIncrement(float heightDistance, float distanceSum, float totalHeightRemoved)
    {
        return totalHeightRemoved * (heightDistance / distanceSum);
    }

    // determines how material is distirbuted around a texel due to thermal erosion
    void computeHeightDistribution(int2 processedTexel, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, float[,] heightSpreadValue)
    {

        float texelHeight = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)processedTexel)];

        float sumError = 0.0f;
        float sumErrorOut = 0.0f;
        float exceedingSum = 0.0f;
        float maxHeightDistance = 0.0f;
        int numCriticalNeighbors = 0;
        float minHeightDistance = BIG_FLOAT;

        for (int x = -1; x < 2; ++x)
        {
            for (int y = -1; y < 2; ++y)
            {
                int2 ncoord = CpuComputeUtilities.terrainWrap(processedTexel + new int2(x, y), (int2)intermediateHeightmap.heightmapDimensions);
                float d = texelHeight - intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)ncoord)];

                if (d > talusThreshold)
                {
                    maxHeightDistance = math.max(maxHeightDistance, d);
                    minHeightDistance = math.min(minHeightDistance, d);
                    numCriticalNeighbors++;
                }
                else
                {
                    d = 0.0f;
                }

                exceedingSum = CpuComputeUtilities.accurateSum(exceedingSum, d, out sumErrorOut);
                sumError += sumErrorOut;
                heightSpreadValue[y + 1, x + 1] = d;
            }
        }
        exceedingSum += sumError;

        if (numCriticalNeighbors > 0)
        {
            // the smaller the distance betweem "average" neighbor -> the more prominent the "avalanche"
            float averageNeighborHeight = (texelHeight * (float)numCriticalNeighbors - exceedingSum) * math.rcp(numCriticalNeighbors);
            float avalancheRatio = (float)math.clamp(1.0 - (averageNeighborHeight / texelHeight), 0.0, 1.0);

            uint seed = CpuComputeUtilities.seedFromXYPass((uint)processedTexel.x, (uint)processedTexel.y, numCriticalNeighbors);
            float u1 = CpuComputeUtilities.u01FromUint(CpuComputeUtilities.pcgHash(seed));

            // zero-mean gaussian is shifted to the estimated ratio. u2 is kept at zero to ensure sigma=1
            float dynamicDistributionCoefficient = (float)(math.clamp(CpuComputeUtilities.sampleGaussBoxMuller(new float2(u1, 0.0f)) + avalancheRatio, 0.3, 1.0) * distributionCoefficient);

            float localSoftness = CpuComputeUtilities.computeSoftnessCoefficient(CpuComputeUtilities.computeGradientAtPoint(intermediateHeightmap.nativeHeightmapArray, intermediateHeightmap.heightmapDimensions, processedTexel), texelHeight, 0.1f);
            float totalHeightRemoved = (float)((numCriticalNeighbors > 0) ? localSoftness * dynamicDistributionCoefficient * (maxHeightDistance - talusThreshold) : 0.0);

            // to make sure the central texel does not end up higher than its closest neighbor
            float heightConsistencyThreshold = (float)((minHeightDistance * exceedingSum) / (minHeightDistance + exceedingSum + 1.0e-5));
            totalHeightRemoved = (totalHeightRemoved <= heightConsistencyThreshold) ? totalHeightRemoved : heightConsistencyThreshold;

            for (int y = -1; y < 2; ++y)
            {
                for (int x = -1; x < 2; ++x)
                {
                    float d = heightSpreadValue[y + 1, x + 1];
                    heightSpreadValue[y + 1, x + 1] = heightDistributionDistanceCoefficients[y + 1, x + 1] * computeSingleHeightIncrement(d, exceedingSum, totalHeightRemoved);
                }
            }

            // how much height the central texel loses
            heightSpreadValue[1, 1] = -totalHeightRemoved;
        }
    }

    void applyKernel(int2 processedTexel, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, float[,] sharedHeightSpreadValue)
    {
        computeHeightDistribution(processedTexel, intermediateHeightmap, sharedHeightSpreadValue);
        for (int i = -1; i < 2; ++i)
        {
            for (int j = -1; j < 2; ++j)
            {
                int2 p = CpuComputeUtilities.terrainWrap(processedTexel + new int2(i, j), (int2)intermediateHeightmap.heightmapDimensions);
                intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)p)] += sharedHeightSpreadValue[j + 1, i + 1];
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int heightmapSize = pipelineContext.GetHeightmapSize();
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();
        float[,] sharedHeightSpreadValue = new float[3, 3];

        int numTotalTexels = (int)math.pow(heightmapSize, 2);
        int permuteA = GenerationUtilities.ComputeCoprime(numTotalTexels, 101);
        for (int i = 0; i < erosionIterationLimit; ++i)
        {
            float randomUniform = pipelineContext.GetRandomFloat();
            uint permuteB = (uint)((1 + (int)(randomUniform * numTotalTexels)) % numTotalTexels);

            // iterates over the grid in a permuted order
            for (uint idx = 0; idx < numTotalTexels; ++idx)
            {
                uint permutedIdx = (uint)(((long)permuteA * idx + permuteB) % numTotalTexels);

                int x = (int)(permutedIdx / heightmapSize);
                int y = (int)(permutedIdx % heightmapSize);

                Array.Clear(sharedHeightSpreadValue, 0, sharedHeightSpreadValue.Length);
                applyKernel(new int2(x, y), intermediateHeightmap, sharedHeightSpreadValue);
            }
        }

        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu() => CpuInputExpectations.HeightMapNormalized;

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {

    }
}
