using Unity.Mathematics;
using UnityEngine;
using System;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using System.Threading.Tasks;
using UnityEngine.Rendering;
public class SDFDispatcher : UltimatePipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0.025f, 4.0f)]
    public float baseSimplexFrequency = 1.0f;

    [Range(0.01f, 1.0f)]
    public float flowPerturbationStrength = 0.01f;

    [Range(0.01f, 1.0f)]
    public float basePerturbationStrength = 0.01f;

    [SerializeField]
    public bool applyTerrainMask = false;

    private static readonly uint2 MAX_COORD = new uint2(UInt32.MaxValue, UInt32.MaxValue);

    //both are implicitly cleared
    private ComputeBuffer floodingBuffer1;
    private ComputeBuffer floodingBuffer2;
    private ComputeBuffer inputContinentHeightmap;

    //both are implicitly cleared
    private uint2[,] cpuFloodingBuffer1;
    private uint2[,] cpuFloodingBuffer2;

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
    private static readonly int PID_flowPerturbationStrength = Shader.PropertyToID("flowPerturbationStrength");
    private static readonly int PID_basePerturbationStrength = Shader.PropertyToID("basePerturbationStrength");

    private static readonly int PID_currentSourceBuffer = Shader.PropertyToID("currentSourceBuffer");
    private static readonly int PID_floodStepSize = Shader.PropertyToID("floodStepSize");
    private static readonly int PID_randomInts = Shader.PropertyToID("randomInts");

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numLinearGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 1);
        int numLinearThreads = optimalGroupSize * numLinearGroups;

        if (RuntimeAssert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly! Try adjusting heightmap or threadgroup size."))
            return;

        {
            int neededBufferSize = textureSize * textureSize;
            if (inputContinentHeightmap is null || !inputContinentHeightmap.IsValid() || inputContinentHeightmap.count != neededBufferSize)
            {
                inputContinentHeightmap = new ComputeBuffer(neededBufferSize, sizeof(float));
                if (RuntimeAssert.IsTrue(inputContinentHeightmap.IsValid(), "Failed to initialize the required resources! Try restarting the app or explicitly freeing the resources!"))
                    return;
            }
        }

        {
            int neededBufferSize = textureSize * textureSize * 2;
            if (floodingBuffer1 is null || !floodingBuffer1.IsValid() || floodingBuffer1.count != neededBufferSize)
            {
                floodingBuffer1 = new ComputeBuffer(neededBufferSize, sizeof(float));
                if (RuntimeAssert.IsTrue(floodingBuffer1.IsValid(), "Failed to initialize the required resources! Try restarting the app or explicitly freeing the resources!"))
                    return;
            }
            if (floodingBuffer2 is null || !floodingBuffer2.IsValid() || floodingBuffer2.count != neededBufferSize)
            {
                floodingBuffer2 = new ComputeBuffer(neededBufferSize, sizeof(float));
                if (RuntimeAssert.IsTrue(floodingBuffer2.IsValid(), "Failed to initialize the required resources! Try restarting the app or explicitly freeing the resources!"))
                    return;
            }
        }

        int maskToSeedBufferKernelIdx = shaderToDispatch.FindKernel("MaskToSeedBuffer");
        int floodingStepKernelIdx = shaderToDispatch.FindKernel("FloodingStep");
        int seedBufferToHieghtmapKernelIdx = shaderToDispatch.FindKernel("SeedBufferToHieghtmap");
        int coastlineGeneratorKernel = shaderToDispatch.FindKernel("CoastlineGenerator");
        int sDFPostprocessorKernel = shaderToDispatch.FindKernel("SDFPostprocessor");

        // Kernels
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

        {
            var useMaskShaderKeyword = new LocalKeyword(shaderToDispatch, "USE_MASK");
            pipelineContext.SetKeyword(shaderToDispatch, ref useMaskShaderKeyword, applyTerrainMask);

            var noMaskShaderKeyword = new LocalKeyword(shaderToDispatch, "NOT_USE_MASK");
            pipelineContext.SetKeyword(shaderToDispatch, ref noMaskShaderKeyword, !applyTerrainMask);

        }

        // Uniforms
        pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerThread, textureSize / numLinearThreads);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_bufferSideLength, textureSize);
        pipelineContext.SetUniformInt(shaderToDispatch, PID_maskBorderWidth, (int)(textureSize * 0.2f)); // fix at 20%

        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_shoreDistanceThreshold, textureSize * 0.1f);   // fix at 10%
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_shoreBaseHeight, maxDistanceToSeed * 0.005f); // fix at 0.5%
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_simplexFrequency, baseSimplexFrequency);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_flowPerturbationStrength, flowPerturbationStrength);
        pipelineContext.SetUniformFloat(shaderToDispatch, PID_basePerturbationStrength, basePerturbationStrength);
        pipelineContext.SetRandomInts(shaderToDispatch, PID_randomInts, pipelineContext.GetHeightmapSize());

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

        // Normalization of the output SDF texture (with height written to it)
        pipelineContext.RunInternalNormalization();

        // Heightmap postprocessing
        pipelineContext.AppendDispatchToCommandBuffer(shaderToDispatch, sDFPostprocessorKernel, dispatchGroups);
    }

    public override InputExpectations GetStepExpectationsGpu() => InputExpectations.None;

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    // tells if a texel is considered part of the mask
    bool texelIsWhiteForMask(float texelColor)
    {
        // return sqrt(texelColor) > 0.7;
        return texelColor > 0.49;
    }

    // tells if texel is considered part of land when 'punching' lakes
    bool texelIsWhiteForLake(float texelColor)
    {
        return texelColor > 0.03;
    }

    // fractal accumulation of simplx noise
    float fbm(float2 uv, float startFreq)
    {
        float freq = startFreq;
        float amp = 1.0f;
        float n = 0.0f;
        for (int i = 0; i < 5; i++)
        {
            n += amp * noise.snoise(uv * freq);
            freq *= 2.0f;
            amp *= 0.5f;
        }

        return n;
    }

    float3 computeClosestPointOnEllipse(float a, float b, float2 pointOfInterest)
    {
        float scaling = (a * b) / math.sqrt(pointOfInterest.x * pointOfInterest.x * b * b + pointOfInterest.y * pointOfInterest.y * a * a);
        float2 actualEllipsePoint = pointOfInterest * scaling;

        return math.float3(actualEllipsePoint, scaling);
    }

    float maskCoefficient(float2 texelPosition, uint2 heightmapDimensions, uint2 maskBorderWidth)
    {
        uint2 halfLength = (uint2)((float2)heightmapDimensions * 0.5f);
        float innerA = halfLength.x - maskBorderWidth.x;
        float innerB = halfLength.y - maskBorderWidth.y;

        float outerA = halfLength.x;
        float outerB = halfLength.y;

        float3 closestInner = computeClosestPointOnEllipse(innerA, innerB, texelPosition);
        float3 closestOuter = computeClosestPointOnEllipse(outerA, outerB, texelPosition);

        float falloffLength = math.length(closestOuter.xy - closestInner.xy);
        float distanceToInner = math.length(texelPosition - closestInner.xy);

        return (float)(1.0 - math.clamp((distanceToInner / falloffLength) * (closestInner.z < 1.0f ? 1.0f : 0.0f), 0.0, 1.0));
    }

    float computeFlowNoise(float2 uv, float startFreq)
    {
        float amp = 0.9f;
        float2 gsum = new float2(0.2f, 0.2f);

        float disp = 0.0f;
        for (int i = 0; i < 5; i++)
        {
            float3 psrdSample = amp * noise.srdnoise(uv * startFreq + gsum * flowPerturbationStrength, 0.5f * startFreq);
            disp += psrdSample.x;
            gsum += psrdSample.yz * amp;

            startFreq *= 2.0f;
            amp *= 0.5f;
        }

        return disp;
    }

    float coastlineComputer(float2 texelCoord, uint2 heightmapDimensions, float2 randomUVOffset)
    {
        float2 floatDimensions = (float2)heightmapDimensions;
        float2 uv = randomUVOffset + (texelCoord / floatDimensions);

        float addFlow = 1.0f - (0.85f + computeFlowNoise(uv, baseSimplexFrequency / 1.3f));
        float subFlow = 1.0f - (0.75f + computeFlowNoise(uv, baseSimplexFrequency / 1.7f));

        float maskValue = applyTerrainMask ? maskCoefficient(texelCoord - floatDimensions * 0.5f, heightmapDimensions, (uint2)(floatDimensions * 0.2f)) : 1.0f;
        float2 q = new float2(fbm(uv + new float2(0.0f, 0.7f), baseSimplexFrequency),
                          fbm(uv + new float2(0.7f, 0.0f), baseSimplexFrequency));
        float n = maskValue * fbm(uv + basePerturbationStrength * q, baseSimplexFrequency) + addFlow - subFlow;

        float smoke = 0.5f * maskValue + n * 0.25f;
        return math.clamp(smoke, 0.0f, 1.0f);
    }

    void CoastlineGenerator(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, CpuPipelineContext pipelineContext)
    {
        var randomUVOffset = pipelineContext.GetRandomInts((int)intermediateHeightmap.heightmapDimensions.x);
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                var texelCoord = new uint2(x, y);
                intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, texelCoord)] = coastlineComputer(texelCoord, intermediateHeightmap.heightmapDimensions, randomUVOffset);
            }
        }
    }

    void MaskToSeedBuffer(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, uint2[,] floodingBuffer1)
    {
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 kernelCenter = new uint2(x, y);
                uint kernelCenterLin = CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, kernelCenter);

                if (texelIsWhiteForMask(intermediateHeightmap.nativeHeightmapArray[(int)kernelCenterLin]))
                {
                    // center is white
                    bool blackFound = false;
                    for (int i = -1; i < 2 && !blackFound; ++i)
                    {
                        for (int j = -1; j < 2 && !blackFound; ++j)
                        {
                            // uint2 targetNeighbor = new uint2(math.clamp((int)kernelCenter.x + i, 0, bufferSideLength - 1),
                            //  math.clamp((int)kernelCenter.y + j, 0, bufferSideLength - 1)); <- old
                            uint2 targetNeighbor = (uint2)math.clamp((int2)kernelCenter + new int2(i, j), 0, intermediateHeightmap.heightmapDimensions - new uint2(1, 1));
                            float neighborColor = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, targetNeighbor)];
                            if (!texelIsWhiteForMask(neighborColor))
                            {
                                floodingBuffer1[kernelCenter.x, kernelCenter.y] = kernelCenter;
                                blackFound = true;
                            }
                        }
                    }

                    if (!blackFound)
                    {
                        floodingBuffer1[kernelCenter.x, kernelCenter.y] = MAX_COORD;
                    }
                }
                else
                {
                    // center is black
                    floodingBuffer1[kernelCenter.x, kernelCenter.y] = MAX_COORD;
                }
            }
        }
    }

    void FloodingStep(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, int floodStepSize, uint2[,] currentReadBuffer, uint2[,] currentWriteBuffer)
    {
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 cellCoords = new uint2(x, y);

                uint minSquareDistance = Int32.MaxValue;
                uint2 closestSeed = MAX_COORD;

                for (int i = -1; i < 2; ++i)
                {
                    int targetX = (int)cellCoords.x + i * floodStepSize;

                    if (targetX < 0 || targetX >= intermediateHeightmap.heightmapDimensions.x)
                    {
                        continue;
                    }

                    for (int j = -1; j < 2; ++j)
                    {
                        int targetY = (int)cellCoords.y + j * floodStepSize;

                        if (targetY < 0 || targetY >= intermediateHeightmap.heightmapDimensions.y)
                        {
                            continue;
                        }

                        uint2 closestSeedAtTarget = currentReadBuffer[(uint)targetX, (uint)targetY];

                        if (closestSeedAtTarget.x == Int32.MaxValue && closestSeedAtTarget.y == Int32.MaxValue)
                        {
                            continue;
                        }

                        uint2 closestSeedDistanceVector = closestSeedAtTarget - cellCoords;
                        uint squareDistance = math.dot(closestSeedDistanceVector, closestSeedDistanceVector);
                        if (squareDistance < minSquareDistance)
                        {
                            minSquareDistance = squareDistance;
                            closestSeed = closestSeedAtTarget;
                        }
                    }
                }

                currentWriteBuffer[cellCoords.x, cellCoords.y] = closestSeed;
            }
        }
    }

    void SeedBufferToHieghtmap(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, uint2[,] currentReadBuffer, float shoreBaseHeight, float shoreDistanceThreshold)
    {
        float maxHeight = Single.MinValue;
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 cellCoords = new uint2(x, y);

                uint2 seedCoordinates = currentReadBuffer[cellCoords.x, cellCoords.y];
                float actualDistance = math.length((int2)seedCoordinates - (int2)cellCoords);

                var targetCellLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, cellCoords);
                float currentHeight = intermediateHeightmap.nativeHeightmapArray[targetCellLin];
                float distanceAtTexel;
                if (texelIsWhiteForMask(currentHeight))
                {
                    distanceAtTexel = actualDistance + shoreBaseHeight;
                }
                else
                {
                    distanceAtTexel = (float)math.lerp(shoreBaseHeight, 0.0, math.min(actualDistance, shoreDistanceThreshold) / shoreDistanceThreshold);
                }

                maxHeight = math.max(maxHeight, distanceAtTexel);
            }
        }

        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 cellCoords = new uint2(x, y);

                uint2 seedCoordinates = currentReadBuffer[cellCoords.x, cellCoords.y];
                float actualDistance = math.length((int2)seedCoordinates - (int2)cellCoords);

                var targetCellLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, cellCoords);
                float currentHeight = intermediateHeightmap.nativeHeightmapArray[targetCellLin];
                float distanceAtTexel;
                if (texelIsWhiteForMask(currentHeight))
                {
                    distanceAtTexel = actualDistance + shoreBaseHeight;
                }
                else
                {
                    distanceAtTexel = (float)math.lerp(shoreBaseHeight, 0.0, math.min(actualDistance, shoreDistanceThreshold) / shoreDistanceThreshold);
                }
                //normalize local distance
                distanceAtTexel /= maxHeight;

                //"lake" formation
                currentHeight *= texelIsWhiteForLake(currentHeight) ? 1.0f : 0.0f;

                // clamp goes here since we expect more 'noisy' features to appear where tectonic processes were the most porminent - at mountains
                intermediateHeightmap.nativeHeightmapArray[targetCellLin] = (float)(distanceAtTexel + currentHeight * math.clamp(distanceAtTexel, 0.45, 1.0));
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();

        {
            if (cpuFloodingBuffer1 is null || cpuFloodingBuffer1.GetLength(0) != textureSize || cpuFloodingBuffer1.GetLength(1) != textureSize)
            {
                cpuFloodingBuffer1 = new uint2[textureSize, textureSize];
            }
            if (cpuFloodingBuffer2 is null || cpuFloodingBuffer2.GetLength(0) != textureSize || cpuFloodingBuffer2.GetLength(1) != textureSize)
            {
                cpuFloodingBuffer2 = new uint2[textureSize, textureSize];
            }
        }

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
        };

        CoastlineGenerator(intermediateHeightmap, pipelineContext);
        MaskToSeedBuffer(intermediateHeightmap, cpuFloodingBuffer1);

        while (currentFloodStep > 1)
        {
            updateSourceBuffer();
            currentFloodStep /= 2;
            FloodingStep(intermediateHeightmap, currentFloodStep, currentReadBuffer == 0 ? cpuFloodingBuffer1 : cpuFloodingBuffer2, currentReadBuffer == 0 ? cpuFloodingBuffer2 : cpuFloodingBuffer1);
        }

        // Extra iteration to improve SDF accuracy
        updateSourceBuffer();
        FloodingStep(intermediateHeightmap, 1, currentReadBuffer == 0 ? cpuFloodingBuffer1 : cpuFloodingBuffer2, currentReadBuffer == 0 ? cpuFloodingBuffer2 : cpuFloodingBuffer1);

        // Distance computation and postprocessing
        updateSourceBuffer();
        SeedBufferToHieghtmap(intermediateHeightmap, currentReadBuffer == 0 ? cpuFloodingBuffer1 : cpuFloodingBuffer2, maxDistanceToSeed * 0.005f, textureSize * 0.1f);

        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu() => CpuInputExpectations.None;

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
        previousState &= ~CpuHeightmapProperties.Normalized;
    }

    public override void RenderParametersTuningGUI()
    {
        GUILayout.Label($"Base simplex frequency: {baseSimplexFrequency:F3}");
        baseSimplexFrequency = GUILayout.HorizontalSlider(baseSimplexFrequency, 0.01f, 4.0f);

        GUILayout.Label($"Simplex noise perturbation strength: {basePerturbationStrength:F3}");
        basePerturbationStrength = GUILayout.HorizontalSlider(basePerturbationStrength, 0.01f, 1.0f);

        GUILayout.Label($"Flow noise perturbation strength: {flowPerturbationStrength:F3}");
        flowPerturbationStrength = GUILayout.HorizontalSlider(flowPerturbationStrength, 0.01f, 1.0f);
    }

    public override string GUIStepTitle() => "SDF generator";

    public override void FreeResources()
    {
        floodingBuffer1?.Release();
        floodingBuffer2?.Release();
        inputContinentHeightmap?.Release();

        cpuFloodingBuffer1 = null;
        cpuFloodingBuffer2 = null;
    }

    public override void RandomizeParameters(System.Random random)
    {
        //baseSimplexFrequency -> ignored since defines scale 
        basePerturbationStrength = (float)math.max(0.01, random.NextDouble());
        flowPerturbationStrength = (float)math.max(0.01, random.NextDouble());
    }
}
