using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using Unity.Collections;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Assertions;

public class CellularHydraulicErosionDispatcher : UltimatePipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;
    [Range(5, 100)]
    public int erosionIterationLimit = 25;

    [Range(0.0001f, 1.0f)]
    public float solubilityConstant;

    [Range(0.0001f, 1.0f)]
    public float evaporationConstant;

    [Range(0.0001f, 1.0f)]
    public float capacityConstant;

    [Range(0.0001f, 1.0f)]
    public float rainSolubilityConstant;

    [Range(0.0001f, 10.0f)]
    public float maxWaterDepth;

    [Range(0.0001f, 1.0f)]
    public float depositionConstant;

    [Range(0.001f, 1.0f)]
    public float rainNoiseFrequency = 0.25f;

    //GPU
    private ComputeBuffer texelParametersBuffer;
    private ComputeBuffer waterPipesBuffer;
    private ComputeBuffer gradientsBuffer;

    //CPU
    [StructLayout(LayoutKind.Sequential)]
    struct TexelParameters
    {
        public float waterLevel;
        public float sedimentLevel;
    };

    private struct TexelPipes
    {
        public float[,] inWaterPipes;
        public float[,] inSedimentPipes;
    };

    static readonly uint2[,] pipeMap = new uint2[3, 3]
    {
        { new uint2(2, 2), new uint2(2, 1), new uint2(2, 0)},
        { new uint2(1, 2), new uint2(1, 1), new uint2(1, 0)},
        { new uint2(0, 2), new uint2(0, 1), new uint2(0, 0)},
    };

    static readonly float HEIGHT_EPS = 1.0e-3f;
    static readonly float MIN_WATER = 0.3f;

    TexelParameters[,] actualTexelParameters;
    TexelPipes[,] texelPipes;
    float2[,] cpuGradientsBuffer;

    //GUI
    private Vector2 scroll;

    //uniform IDs

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_waterLevel = Shader.PropertyToID("texelParameters");
    private static readonly int PID_pipesBuffer = Shader.PropertyToID("pipesBuffer");
    private static readonly int PID_gradientsBuffer = Shader.PropertyToID("gradientsBuffer");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_solubilityConstant = Shader.PropertyToID("solubilityConstant");
    private static readonly int PID_rainSolubilityConstant = Shader.PropertyToID("rainSolubilityConstant");
    private static readonly int PID_depositionConstant = Shader.PropertyToID("depositionConstant");
    private static readonly int PID_maxWaterDepth = Shader.PropertyToID("maxWaterDepth");
    private static readonly int PID_capacityConstant = Shader.PropertyToID("capacityConstant");
    private static readonly int PID_evaporationConstant = Shader.PropertyToID("evaporationConstant");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_randomSeeds = Shader.PropertyToID("randomSeeds");
    private static readonly int PID_iterationIdx = Shader.PropertyToID("iterationIdx");
    private static readonly int PID_rainNoiseFrequency = Shader.PropertyToID("rainNoiseFrequency");

    public override InputExpectations GetStepExpectationsGpu()
        => InputExpectations.HeightMapNormalized;

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 4);
        int numLinearThreads = optimalGroupSize * numGroups;

        int texelsPerThread = textureSize / numLinearThreads;
        var dispatchGroups = new Vector3(numGroups, numGroups, 1);

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");
        Assert.IsTrue(texelsPerThread >= 4, "A thread must have at least 4 texels to porcess");

        {
            int neededBufferSize = textureSize * textureSize;
            if (texelParametersBuffer is null || !texelParametersBuffer.IsValid() || texelParametersBuffer.count != neededBufferSize)
            {
                texelParametersBuffer = new ComputeBuffer(neededBufferSize, Marshal.SizeOf<TexelParameters>());
                Assert.IsTrue(texelParametersBuffer.IsValid());
            }
        }
        {
            int neededBufferSize = textureSize * textureSize;
            if (waterPipesBuffer is null || !waterPipesBuffer.IsValid() || waterPipesBuffer.count != neededBufferSize)
            {
                waterPipesBuffer = new ComputeBuffer(neededBufferSize, sizeof(float) * 18);
                Assert.IsTrue(waterPipesBuffer.IsValid());
            }
        }
        {
            int neededBufferSize = textureSize * textureSize;
            if (gradientsBuffer is null || !gradientsBuffer.IsValid() || gradientsBuffer.count != neededBufferSize)
            {
                gradientsBuffer = new ComputeBuffer(neededBufferSize, sizeof(float) * 2);
                Assert.IsTrue(gradientsBuffer.IsValid());
            }
        }

        int rainDropKernelIdx = erosionComputeShader.FindKernel("RainDropper");
        int waterDistributorKernelIdx = erosionComputeShader.FindKernel("WaterDistributor");
        int sedimentDistributorKernelIdx = erosionComputeShader.FindKernel("SedimentDistributor");
        int waterEvaporatorKernelIdx = erosionComputeShader.FindKernel("WaterEvaporator");
        int resourceInitializerKernelIdx = erosionComputeShader.FindKernel("ResourceInitializer");
        int finalWaterEvaporatorKernelIdx = erosionComputeShader.FindKernel("FinalWaterEvaporator");
        int pipePlumberKernelIdx = erosionComputeShader.FindKernel("PipePlumber");

        foreach (int kernelIdx in new[] { rainDropKernelIdx, waterEvaporatorKernelIdx, resourceInitializerKernelIdx, finalWaterEvaporatorKernelIdx, waterDistributorKernelIdx, sedimentDistributorKernelIdx, pipePlumberKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_waterLevel, texelParametersBuffer);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_pipesBuffer, waterPipesBuffer);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_gradientsBuffer, gradientsBuffer);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, texelsPerThread);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_evaporationConstant, evaporationConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_solubilityConstant, solubilityConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_capacityConstant, capacityConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_rainSolubilityConstant, rainSolubilityConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_depositionConstant, depositionConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_maxWaterDepth, maxWaterDepth);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_rainNoiseFrequency, rainNoiseFrequency);

        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, resourceInitializerKernelIdx, dispatchGroups);
        for (int d = 0; d < erosionIterationLimit; ++d)
        {
            pipelineContext.SetRandomFloats(erosionComputeShader, PID_randomSeeds);
            pipelineContext.SetUniformInt(erosionComputeShader, PID_iterationIdx, d + 1);

            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, pipePlumberKernelIdx, dispatchGroups);
            if (d % 5 == 0)
                pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, rainDropKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, waterDistributorKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, sedimentDistributorKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, waterEvaporatorKernelIdx, dispatchGroups);
        }
        // pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, finalWaterEvaporatorKernelIdx, dispatchGroups);
    }

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {

    }

    void RainDropper(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, TexelParameters[,] actualTexelParameters, float2 randomSeeds, float2[,] cpuGradientsBuffer)
    {
        updateGradients(intermediateHeightmap, cpuGradientsBuffer);
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 processedTexel = new uint2(x, y);
                int lin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, processedTexel);

                float currentWaterLevel = actualTexelParameters[x, y].waterLevel;
                float currentHeight = intermediateHeightmap.nativeHeightmapArray[lin];
                float maxAllowedExtraWater = currentHeight / rainSolubilityConstant;

                float extraRainWater = math.min(maxAllowedExtraWater, CpuComputeUtilities.pinkNoise2D((uint2)(((float2)processedTexel + randomSeeds * (float2)intermediateHeightmap.heightmapDimensions) * rainNoiseFrequency)));
                extraRainWater = (float)(extraRainWater < MIN_WATER ? 0.0 : extraRainWater);
                float localSoftness = CpuComputeUtilities.computeSoftnessCoefficient(cpuGradientsBuffer[processedTexel.x, processedTexel.y], currentHeight, 0.1f);
                float heightLoss = extraRainWater * rainSolubilityConstant * localSoftness;

                actualTexelParameters[x, y].waterLevel = extraRainWater + currentWaterLevel;     // step 1
                actualTexelParameters[x, y].sedimentLevel += math.min(currentHeight, heightLoss);     // step 2
                intermediateHeightmap.nativeHeightmapArray[lin] = (float)math.max(currentHeight - heightLoss, 0.0); // step 2
            }
        }
    }

    // computes how water is distributed from texel
    void computeWaterDistribution(int2 processedTexel, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, TexelParameters[,] actualTexelParameters, float[,] waterSpreadValue)
    {
        int centerLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)processedTexel);
        float waterLevelAtTexel = actualTexelParameters[processedTexel.x, processedTexel.y].waterLevel;
        float texelHeight = intermediateHeightmap.nativeHeightmapArray[centerLin] + waterLevelAtTexel;

        float sumError = 0.0f;
        float sumErrorOut = 0.0f;
        float exceedingSum = 0.0f;
        int numCriticalNeighbors = 0;

        {
            for (int x = -1; x < 2; ++x)
            {
                for (int y = -1; y < 2; ++y)
                {
                    int2 neighborCoord = CpuComputeUtilities.terrainWrap(processedTexel + new int2(x, y), (int2)intermediateHeightmap.heightmapDimensions);
                    int nLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)neighborCoord);

                    float d = texelHeight - (intermediateHeightmap.nativeHeightmapArray[nLin] + actualTexelParameters[neighborCoord.x, neighborCoord.y].waterLevel);
                    if (d > HEIGHT_EPS)
                    {
                        numCriticalNeighbors++;
                        exceedingSum = CpuComputeUtilities.accurateSum(exceedingSum, d, out sumErrorOut);
                        sumError += sumErrorOut;
                    }
                    else
                    {
                        d = 0.0f;
                    }
                    waterSpreadValue[y + 1, x + 1] = d;
                }
            }
        }
        exceedingSum += sumError;

        if (numCriticalNeighbors == 0)
        {
            for (int yy = 0; yy < 3; ++yy)
            {
                for (int xx = 0; xx < 3; ++xx)
                {
                    waterSpreadValue[yy, xx] = 0.0f;
                }
            }
            return;
        }

        // average height of the critical neighbors
        float invCrit = 1.0f / (float)numCriticalNeighbors;
        float averageNeighborHeight = ((float)numCriticalNeighbors * texelHeight - exceedingSum) * invCrit;

        float totalWaterLost = texelHeight - averageNeighborHeight;
        totalWaterLost = math.max(0.0f, totalWaterLost);
        totalWaterLost = math.min(waterLevelAtTexel, totalWaterLost);

        {
            float invSum = (float)(exceedingSum > 0 ? 1.0 / exceedingSum : 0.0);

            for (int x = -1; x < 2; ++x)
            {
                for (int y = -1; y < 2; ++y)
                {
                    float d = waterSpreadValue[y + 1, x + 1];

                    float dWater = totalWaterLost * d * invSum;

                    waterSpreadValue[y + 1, x + 1] = dWater;
                }
            }
        }

        // how much water the central texel loses
        waterSpreadValue[1, 1] = -totalWaterLost;
        return;
    }

    void applyWaterKernel(int2 processedTexel, CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, TexelParameters[,] actualTexelParameters, TexelPipes[,] texelPipes, float[,] sharedWaterSpreadLevel)
    {
        computeWaterDistribution(processedTexel, intermediateHeightmap, actualTexelParameters, sharedWaterSpreadLevel); // step 3

        for (int i = -1; i < 2; ++i)
        {
            for (int j = -1; j < 2; ++j)
            {
                int2 target = CpuComputeUtilities.terrainWrap(processedTexel + new int2(i, j), (int2)intermediateHeightmap.heightmapDimensions);
                uint2 targetInPipe = pipeMap[i + 1, j + 1];
                texelPipes[target.x, target.y].inWaterPipes[targetInPipe.x, targetInPipe.y] = sharedWaterSpreadLevel[j + 1, i + 1];
            }
        }
    }

    void WaterDistributor(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, TexelParameters[,] actualTexelParameters, TexelPipes[,] texelPipes)
    {
        float[,] sharedWaterSpreadLevel = new float[3, 3];
        for (int x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (int y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                Array.Clear(sharedWaterSpreadLevel, 0, 9); //I just wanna do it explicitly
                applyWaterKernel(new int2(x, y), intermediateHeightmap, actualTexelParameters, texelPipes, sharedWaterSpreadLevel);
            }
        }
    }

    float computeLargerVelocity(int2 processedTexel, uint2 heightmapDimensions, TexelPipes[,] pipesBuffer)
    {
        int2 signedHeightmapDimension = (int2)heightmapDimensions;

        int2 neighborTop = CpuComputeUtilities.terrainWrap(processedTexel - new int2(0, 1), signedHeightmapDimension);
        int2 neighborBottom = CpuComputeUtilities.terrainWrap(processedTexel + new int2(0, 1), signedHeightmapDimension);

        int2 neighborLeft = CpuComputeUtilities.terrainWrap(processedTexel - new int2(1, 0), signedHeightmapDimension);
        int2 neighborRight = CpuComputeUtilities.terrainWrap(processedTexel + new int2(1, 0), signedHeightmapDimension);

        float vX = (float)(0.5 * (pipesBuffer[neighborRight.x, neighborRight.y].inWaterPipes[1, 0] + pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[1, 0] - pipesBuffer[neighborLeft.x, neighborLeft.y].inWaterPipes[1, 2] - pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[1, 2]));

        float vY = (float)(0.5 * (pipesBuffer[neighborBottom.x, neighborBottom.y].inWaterPipes[0, 1] + pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[0, 1] - pipesBuffer[neighborTop.x, neighborTop.y].inWaterPipes[2, 1] - pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[2, 1]));

        float sNormSq = vX * vX + vY * vY;

        int2 neighborTopLeft = CpuComputeUtilities.terrainWrap(processedTexel - new int2(1, 1), signedHeightmapDimension);
        int2 neighborBottomRight = CpuComputeUtilities.terrainWrap(processedTexel + new int2(1, 1), signedHeightmapDimension);

        int2 neighborTopRight = CpuComputeUtilities.terrainWrap(processedTexel + new int2(1, -1), signedHeightmapDimension);
        int2 neighborBottomLeft = CpuComputeUtilities.terrainWrap(processedTexel + new int2(-1, 1), signedHeightmapDimension);

        // from top left to bottom right
        float vL = (float)(0.5 * (pipesBuffer[neighborBottomRight.x, neighborBottomRight.y].inWaterPipes[0, 0] + pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[0, 0] - pipesBuffer[neighborTopLeft.x, neighborTopLeft.y].inWaterPipes[2, 2] - pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[2, 2]));

        // from bottom left to top right
        float vR = (float)(0.5 * (pipesBuffer[neighborTopRight.x, neighborTopRight.y].inWaterPipes[2, 0] + pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[2, 0] - pipesBuffer[neighborBottomLeft.x, neighborBottomLeft.y].inWaterPipes[0, 2] - pipesBuffer[processedTexel.x, processedTexel.y].inWaterPipes[0, 2]));

        float dNormSq = vL * vL + vR * vR;

        return math.sqrt(math.max(sNormSq, dNormSq));
    }

    void updateGradients(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, float2[,] cpuGradientsBuffer)
    {
        for (int x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (int y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                int2 processedTexel = new int2(x, y);
                float2 heightGradientAtTexel = CpuComputeUtilities.computeGradientAtPoint(intermediateHeightmap.nativeHeightmapArray, intermediateHeightmap.heightmapDimensions, processedTexel);
                cpuGradientsBuffer[processedTexel.x, processedTexel.y] = heightGradientAtTexel;

            }
        }
    }
    void SedimentDistributor(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, TexelParameters[,] actualTexelParameters, TexelPipes[,] texelPipes, float2[,] cpuGradientsBuffer)
    {
        updateGradients(intermediateHeightmap, cpuGradientsBuffer);

        int2 signedHeightmapDimension = (int2)intermediateHeightmap.heightmapDimensions;
        for (int x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (int y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                int2 processedTexel = new int2(x, y);
                int centerLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)processedTexel);

                float waterLevelAtTexel = actualTexelParameters[processedTexel.x, processedTexel.y].waterLevel;
                float sedimentLevelAtTexel = actualTexelParameters[processedTexel.x, processedTexel.y].sedimentLevel;
                float terrainHeightAtTexel = intermediateHeightmap.nativeHeightmapArray[centerLin];

                float sedimentAdjustment = 0.0f;

                {
                    float2 heightGradientAtTexel = cpuGradientsBuffer[processedTexel.x, processedTexel.y];

                    float velocityNorm = computeLargerVelocity(processedTexel, intermediateHeightmap.heightmapDimensions, texelPipes);

                    float gradientNorm = math.length(heightGradientAtTexel);
                    float sinAlpha = (float)(gradientNorm / math.sqrt(1.0 + gradientNorm * gradientNorm));

                    float depthAdjustment = (float)(waterLevelAtTexel >= maxWaterDepth ? 1.0 : (1.0 - (maxWaterDepth - waterLevelAtTexel) / maxWaterDepth));
                    float localSedimentCapacity = (float)(capacityConstant * math.max(sinAlpha, 0.2) * velocityNorm * depthAdjustment);

                    if (sedimentLevelAtTexel <= localSedimentCapacity)
                    {
                        // sediment amount is increased
                        float localSoftness = CpuComputeUtilities.computeSoftnessCoefficient(heightGradientAtTexel, terrainHeightAtTexel, 0.1f);
                        // we can't dissolve more than height
                        sedimentAdjustment = math.min(localSoftness * (localSedimentCapacity - sedimentLevelAtTexel) * solubilityConstant, terrainHeightAtTexel);
                    }
                    else
                    {
                        // sediment amount is reduced
                        sedimentAdjustment = -(sedimentLevelAtTexel - localSedimentCapacity) * depositionConstant;
                    }
                    // resultHeightmap[processedTexel] = gradientNorm;
                }

                float waterLostAtTexel = math.abs(texelPipes[processedTexel.x, processedTexel.y].inWaterPipes[1, 1]);
                float intermediateSedimentLevel = sedimentLevelAtTexel + sedimentAdjustment;

                for (int i = -1; i < 2; ++i)
                {
                    for (int j = -1; j < 2; ++j)
                    {
                        int2 processedNeighbor = CpuComputeUtilities.terrainWrap(processedTexel + new int2(i, j), signedHeightmapDimension);

                        uint2 targetInPipe = pipeMap[i + 1, j + 1];

                        float waterReceived = texelPipes[processedNeighbor.x, processedNeighbor.y].inWaterPipes[targetInPipe.x, targetInPipe.y];
                        float sedimentReceived = waterLevelAtTexel > CpuComputeUtilities.EPS ? intermediateSedimentLevel * (waterReceived / waterLevelAtTexel) : 0.0f;

                        texelPipes[processedNeighbor.x, processedNeighbor.y].inSedimentPipes[targetInPipe.x, targetInPipe.y] = sedimentReceived;
                    }
                }

                float invWater = waterLevelAtTexel > CpuComputeUtilities.EPS ? 1.0f / waterLevelAtTexel : 0.0f;
                // this change will be taken into account during evaporation stage
                texelPipes[processedTexel.x, processedTexel.y].inSedimentPipes[1, 1] = /*due depth*/ sedimentAdjustment - /*due water flow*/ (intermediateSedimentLevel * waterLostAtTexel * invWater);
                intermediateHeightmap.nativeHeightmapArray[centerLin] = math.max(0.0f, terrainHeightAtTexel - sedimentAdjustment);
            }
        }
    }

    void WaterEvaporator(uint2 heightmapDimensions, TexelParameters[,] actualTexelParameters, TexelPipes[,] texelPipes)
    {
        float evap = math.saturate(evaporationConstant);

        for (uint x = 0; x < heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < heightmapDimensions.y; ++y)
            {
                uint2 processedTexel = new uint2(x, y);

                float extraPipeWater = 0.0f;
                float extraPipeSediment = 0.0f;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        extraPipeWater += texelPipes[processedTexel.x, processedTexel.y].inWaterPipes[i, j];
                        extraPipeSediment += texelPipes[processedTexel.x, processedTexel.y].inSedimentPipes[i, j];
                    }
                }
                float waterLevelAtTexel = (float)math.max(0.0, actualTexelParameters[processedTexel.x, processedTexel.y].waterLevel + extraPipeWater);
                float sedimentLevelAtTexel = (float)math.max(0.0, actualTexelParameters[processedTexel.x, processedTexel.y].sedimentLevel + extraPipeSediment);

                float newWaterLevel = (float)(waterLevelAtTexel * (1.0 - evap));

                actualTexelParameters[processedTexel.x, processedTexel.y].waterLevel = newWaterLevel;
                actualTexelParameters[processedTexel.x, processedTexel.y].sedimentLevel = sedimentLevelAtTexel;
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();

        {
            if (actualTexelParameters is null || actualTexelParameters.GetLength(0) != textureSize || actualTexelParameters.GetLength(1) != textureSize)
            {
                actualTexelParameters = new TexelParameters[textureSize, textureSize];
            }
        }
        {
            if (cpuGradientsBuffer is null || cpuGradientsBuffer.GetLength(0) != textureSize || cpuGradientsBuffer.GetLength(1) != textureSize)
            {
                cpuGradientsBuffer = new float2[textureSize, textureSize];
            }
        }
        {
            if (texelPipes is null || texelPipes.GetLength(0) != textureSize || texelPipes.GetLength(1) != textureSize)
            {
                texelPipes = new TexelPipes[textureSize, textureSize];
                for (int x = 0; x < textureSize; ++x)
                {
                    for (int y = 0; y < textureSize; ++y)
                    {
                        texelPipes[x, y].inWaterPipes = new float[3, 3];
                        texelPipes[x, y].inSedimentPipes = new float[3, 3];
                    }
                }
            }
        }

        Array.Clear(actualTexelParameters, 0, actualTexelParameters.Length);
        for (int d = 0; d < erosionIterationLimit; ++d)
        {
            for (int x = 0; x < textureSize; x++)
                for (int y = 0; y < textureSize; y++)
                {
                    Array.Clear(texelPipes[x, y].inWaterPipes, 0, 9);
                    Array.Clear(texelPipes[x, y].inSedimentPipes, 0, 9);
                }
            if (d % 5 == 0)
                RainDropper(intermediateHeightmap, actualTexelParameters, pipelineContext.GetRandomFloats(), cpuGradientsBuffer);
            WaterDistributor(intermediateHeightmap, actualTexelParameters, texelPipes);
            SedimentDistributor(intermediateHeightmap, actualTexelParameters, texelPipes, cpuGradientsBuffer);
            WaterEvaporator(intermediateHeightmap.heightmapDimensions, actualTexelParameters, texelPipes);
        }

        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu() => CpuInputExpectations.HeightMapNormalized;

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
    }

    public override void RenderParametersTuningGUI()
    {
        scroll = GUILayout.BeginScrollView(scroll, GUILayout.Height(400));

        GUILayout.Label("Iterations");

        GUILayout.Label($"Erosion Iteration Limit: {erosionIterationLimit}");
        erosionIterationLimit = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(erosionIterationLimit, 5f, 100f)
        );

        GUILayout.Space(5);
        GUILayout.Label("Core Constants");

        GUILayout.Label($"Solubility: {solubilityConstant:F4}");
        solubilityConstant = GUILayout.HorizontalSlider(solubilityConstant, 0.0001f, 1.0f);

        GUILayout.Label($"Evaporation: {evaporationConstant:F4}");
        evaporationConstant = GUILayout.HorizontalSlider(evaporationConstant, 0.0001f, 1.0f);

        GUILayout.Label($"Capacity: {capacityConstant:F4}");
        capacityConstant = GUILayout.HorizontalSlider(capacityConstant, 0.0001f, 1.0f);

        GUILayout.Label($"Deposition: {depositionConstant:F4}");
        depositionConstant = GUILayout.HorizontalSlider(depositionConstant, 0.0001f, 1.0f);

        GUILayout.Space(5);
        GUILayout.Label("Rain");

        GUILayout.Label($"Rain Solubility: {rainSolubilityConstant:F4}");
        rainSolubilityConstant = GUILayout.HorizontalSlider(rainSolubilityConstant, 0.0001f, 1.0f);

        GUILayout.Label($"Rain Noise Frequency: {rainNoiseFrequency:F3}");
        rainNoiseFrequency = GUILayout.HorizontalSlider(rainNoiseFrequency, 0.001f, 1.0f);

        GUILayout.Space(5);
        GUILayout.Label("Water");

        GUILayout.Label($"Max Water Depth: {maxWaterDepth:F4}");
        maxWaterDepth = GUILayout.HorizontalSlider(maxWaterDepth, 0.0001f, 10.0f);

        GUILayout.EndScrollView();
    }

    public override string GUIStepTitle() => "CHE eroder";
}
