using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using CpuGenerationPipeline;
using GpuGenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class ParticleHydraulicErosionDispatcher : UltimatePipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(7, 32)]
    public int numSimultaneousParticles = 10;

    [Range(1, 3)]
    public int erosionNeighborhood = 1;

    [Range(1, 1000)]
    public int numSimulationSteps = 1;

    [Range(1, 10)]
    public int numSimulationWaves = 1;

    [Range(0.01f, 1.0f)]
    public float inertia = 0.1f;

    [Range(0.01f, 64.0f)]
    public float capacity = 2.0f;

    [Range(0.0001f, 1.0f)]
    public float minSlope = 0.01f;

    [Range(0.01f, 1.0f)]
    public float deposition = 0.7f;

    [Range(0.01f, 1.0f)]
    public float erosion = 0.35f;

    [Range(0.01f, 10.0f)]
    public float gravity = 0.4f;

    [Range(0.01f, 1.0f)]
    public float evaporation = 0.02f;

    [Range(0.01f, 0.45f)]
    public float waterDeathThreshold = 0.05f;

    [Range(1.0f, 10.0f)]
    public float rainNoiseFrequency = 1.0f;

    [StructLayout(LayoutKind.Sequential)]
    private struct ErosionParticle
    {
        public float2 pos;
        public float2 dir;
        public float vel;
        public float w;
        public float s;
    };

    private ComputeBuffer gpuParticlesBuffer;

    struct GpuTexelPipes
    {
        float3 inPipes1;
        float3 inPipes2;
        float3 inPipes3;
        float3 outPipes1;
        float3 outPipes2;
        float3 outPipes3;
    };

    private ComputeBuffer gpuPipesBuffer;

    //CPU
    static readonly uint2[,] pipeMap =
    {
        { new uint2(2, 2), new uint2(2, 2), new uint2(2, 1), new uint2(2, 1), new uint2(2, 1), new uint2(2, 0), new uint2(2, 0)},
        { new uint2(2, 2), new uint2(2, 2), new uint2(2, 1), new uint2(2, 1), new uint2(2, 1), new uint2(2, 0), new uint2(2, 0)},
        { new uint2(1, 2), new uint2(1, 2), new uint2(2, 2), new uint2(2, 1), new uint2(2, 0), new uint2(1, 0), new uint2(1, 0)},
        { new uint2(1, 2), new uint2(1, 2), new uint2(1, 2), new uint2(1, 1), new uint2(1, 0), new uint2(1, 0), new uint2(1, 0)},
        { new uint2(1, 2), new uint2(1, 2), new uint2(0, 2), new uint2(0, 1), new uint2(0, 0), new uint2(1, 0), new uint2(1, 0)},
        { new uint2(0, 2), new uint2(0, 2), new uint2(0, 1), new uint2(0, 1), new uint2(0, 1), new uint2(0, 0), new uint2(0, 0)},
        { new uint2(0, 2), new uint2(0, 2), new uint2(0, 1), new uint2(0, 1), new uint2(0, 1), new uint2(0, 0), new uint2(0, 0)}};
    static readonly int pipeMapCenterCoord = 3;
    static readonly float PLAIN_HEIGHT_THRESHOLD = 1.0e-3f;

    static readonly float[,] depositionMapWeights =
    {
        { 0.02f, 0.15f, 0.02f},
        { 0.15f, 0.3f, 0.15f},
        { 0.02f, 0.15f, 0.02f}};

    struct CpuTexelPipes
    {
        public float[,] inPipes;
        public float[,] outPipes;
    };

    ErosionParticle[] particlesBuffer;
    CpuTexelPipes[,] pipesBuffer;

    //GUI
    private Vector2 scroll;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_particlesBuffer = Shader.PropertyToID("particlesBuffer");
    private static readonly int PID_pipesBuffer = Shader.PropertyToID("pipesBuffer");
    private static readonly int PID_particlesPerThread = Shader.PropertyToID("particlesPerThread");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_inertia = Shader.PropertyToID("inertia");
    private static readonly int PID_capacity = Shader.PropertyToID("capacity");
    private static readonly int PID_minSlope = Shader.PropertyToID("minSlope");
    private static readonly int PID_deposition = Shader.PropertyToID("deposition");
    private static readonly int PID_erosion = Shader.PropertyToID("erosion");
    private static readonly int PID_gravity = Shader.PropertyToID("gravity");
    private static readonly int PID_evaporation = Shader.PropertyToID("evaporation");
    private static readonly int PID_erosionNeighborhood = Shader.PropertyToID("erosionNeighborhood");
    private static readonly int PID_erosionDistanceSumPrecompute = Shader.PropertyToID("erosionDistanceSumPrecompute");
    private static readonly int PID_randomInts = Shader.PropertyToID("randomInts");
    private static readonly int PID_rainNoiseFrequency = Shader.PropertyToID("rainNoiseFrequency");

    private float precomputeDistanceSum()
    {
        float erosionDistanceSumPrecompute = 0.0f;

        float actualNeighborRadius = 1.5f * erosionNeighborhood;
        for (int x = -1 * erosionNeighborhood; x <= erosionNeighborhood; ++x)
        {
            for (int y = -1 * erosionNeighborhood; y <= erosionNeighborhood; ++y)
            {
                erosionDistanceSumPrecompute += actualNeighborRadius - math.sqrt(x * x + y * y);
            }
        }
        return erosionDistanceSumPrecompute;
    }

    public override InputExpectations GetStepExpectationsGpu()
        => InputExpectations.HeightMapNormalized;

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int preferredGroupSize = pipelineContext.preferredGlobalGroupSize;
        int numActualParticles = (int)math.pow(2, numSimultaneousParticles);
        var (optimalIntegrateGroupSize, numIntegrationGroups) = GenerationUtilities.GetOptimalNumberOfGroups(numActualParticles, new int[] { preferredGroupSize }, Int32.MaxValue, 1);
        int particlesPerThread = numActualParticles / (preferredGroupSize * numIntegrationGroups);
        Assert.IsTrue(numActualParticles % (preferredGroupSize * numIntegrationGroups) == 0, "Particles must be distributed among threads evenly!");

        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalResolveGroupSize, numResolveGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { preferredGroupSize }, Int32.MaxValue, 1);
        int numTexelsPerThread = textureSize / (numResolveGroups * preferredGroupSize);
        Assert.IsTrue(textureSize % (numResolveGroups * preferredGroupSize) == 0, "Texels must be distributed among threads evenly!");

        var integrateDispatchGroups = new Vector3(numIntegrationGroups, 1, 1);
        var resolveDispatchGroups = new Vector3(numResolveGroups, numResolveGroups, 1);

        int particlesInitializerKernelIdx = erosionComputeShader.FindKernel("ParticlesInitializer");
        int changeResolverKernelIdx = erosionComputeShader.FindKernel("ChangeResolver");
        int pipesInitializerKernelIdx = erosionComputeShader.FindKernel("PipesInitializer");
        int integratorKernelIdx = erosionComputeShader.FindKernel("Integrator");
        int garbageCollectorKernelIdx = erosionComputeShader.FindKernel("GarbageCollector");

        {
            Debug.LogWarning("Size of a particle struct: " + Marshal.SizeOf<ErosionParticle>());
            if (gpuParticlesBuffer is null || !gpuParticlesBuffer.IsValid() || gpuParticlesBuffer.count != numActualParticles)
            {
                gpuParticlesBuffer = new ComputeBuffer(numActualParticles, Marshal.SizeOf<ErosionParticle>());
                Assert.IsTrue(gpuParticlesBuffer.IsValid());
            }
        }

        {
            Debug.LogWarning("Size of a pipe struct: " + Marshal.SizeOf<GpuTexelPipes>());
            int neededBufferSize = textureSize * textureSize;
            if (gpuPipesBuffer is null || !gpuPipesBuffer.IsValid() || gpuPipesBuffer.count != neededBufferSize)
            {
                gpuPipesBuffer = new ComputeBuffer(neededBufferSize, Marshal.SizeOf<GpuTexelPipes>());
                Assert.IsTrue(gpuPipesBuffer.IsValid());
            }
        }

        foreach (int kernelIdx in new[] { pipesInitializerKernelIdx, particlesInitializerKernelIdx, integratorKernelIdx, garbageCollectorKernelIdx, changeResolverKernelIdx })
        {
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_particlesBuffer, gpuParticlesBuffer);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_pipesBuffer, gpuPipesBuffer);
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        float erosionDistanceSumPrecompute = precomputeDistanceSum();

        pipelineContext.SetUniformInt(erosionComputeShader, PID_particlesPerThread, particlesPerThread);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, numTexelsPerThread);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[] { textureSize, textureSize });
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_inertia, inertia);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_capacity, capacity);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_minSlope, minSlope);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_deposition, deposition);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_erosion, erosion);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_gravity, gravity);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_evaporation, evaporation);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_erosionNeighborhood, erosionNeighborhood);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_erosionDistanceSumPrecompute, erosionDistanceSumPrecompute);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_rainNoiseFrequency, rainNoiseFrequency);

        {
            int garbageCollectorRunPeriod = (int)math.floor(math.log2(2 * waterDeathThreshold) / math.log2(1.0f - evaporation));
            Assert.IsTrue(math.abs(evaporation) >= 1.0e-5 && math.abs(waterDeathThreshold - 0.5) >= 1.0e-5, "Broken GC period");
            Debug.LogWarning("GC period: " + garbageCollectorRunPeriod);
            for (int w = 0; w < numSimulationWaves; ++w)
            {
                pipelineContext.SetRandomInts(erosionComputeShader, PID_randomInts);
                pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, particlesInitializerKernelIdx, integrateDispatchGroups);
                pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, pipesInitializerKernelIdx, resolveDispatchGroups);

                {
                    int gcRunInsertionPeriod = 0;
                    for (int s = 0; s < numSimulationSteps; ++s)
                    {
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, integratorKernelIdx, integrateDispatchGroups);
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, changeResolverKernelIdx, resolveDispatchGroups);
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, pipesInitializerKernelIdx, resolveDispatchGroups);
                        if (s / garbageCollectorRunPeriod != gcRunInsertionPeriod)
                        {
                            pipelineContext.SetRandomInts(erosionComputeShader, PID_randomInts);
                            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, garbageCollectorKernelIdx, integrateDispatchGroups);
                            gcRunInsertionPeriod = s / garbageCollectorRunPeriod;
                        }
                    }
                }
            }
        }
    }
    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    { }

    bool checkParticleIsValid(uint2 heightmapDimensions, ErosionParticle particle)
    {
        float2 particlePos = particle.pos;

        return (particlePos.x >= (float)(erosionNeighborhood + 1) && particlePos.y >= (float)(erosionNeighborhood + 1)) && (particlePos.x < (float)(heightmapDimensions.x - erosionNeighborhood - 1) && particlePos.y < (float)(heightmapDimensions.y - erosionNeighborhood - 1)) && particle.w > waterDeathThreshold;

    }

    float sampleHeightmapBilinear(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, float2 coord)
    {
        int2 signedHeightmapDimensions = (int2)intermediateHeightmap.heightmapDimensions;

        int2 basePart = new int2(math.floor(coord));
        float2 fracPart = math.frac(coord);

        uint2 wrappedBase = (uint2)CpuComputeUtilities.terrainWrap(basePart, signedHeightmapDimensions);

        float v00 = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, wrappedBase)];
        float v10 = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(basePart + new int2(1, 0), signedHeightmapDimensions))];
        float v01 = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(basePart + new int2(0, 1), signedHeightmapDimensions))];
        float v11 = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (uint2)CpuComputeUtilities.terrainWrap(basePart + new int2(1, 1), signedHeightmapDimensions))];

        float v0 = math.lerp(v00, v10, fracPart.x);
        float v1 = math.lerp(v01, v11, fracPart.x);

        return math.lerp(v0, v1, fracPart.y);
    }

    float removeSediment(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, int2 erosionCenter, float targetAmountToRemove, float erosionDistanceSumPrecompute, CpuTexelPipes[,] pipesBuffer)
    {
        float actualSedimentRemoved = 0.0f;
        float actualNeighborRadius = 1.5f * (float)erosionNeighborhood;
        int2 signedHeightmapDimensions = (int2)intermediateHeightmap.heightmapDimensions;

        for (int x = -1 * erosionNeighborhood; x <= erosionNeighborhood; ++x)
        {
            for (int y = -1 * erosionNeighborhood; y <= erosionNeighborhood; ++y)
            {
                float fx = (float)x;
                float fy = (float)y;

                float dist = math.sqrt(fx * fx + fy * fy);
                float weight = (actualNeighborRadius - dist) / math.max(erosionDistanceSumPrecompute, CpuComputeUtilities.EPS);

                uint2 affectedNeighborCoordinate = (uint2)(erosionCenter + new int2(x, y));
                float currentTexelHeight = intermediateHeightmap.nativeHeightmapArray[(int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, affectedNeighborCoordinate)];

                float localSoftness = CpuComputeUtilities.computeSoftnessCoefficient(CpuComputeUtilities.computeGradientAtPoint(intermediateHeightmap.nativeHeightmapArray, intermediateHeightmap.heightmapDimensions, (int2)affectedNeighborCoordinate), currentTexelHeight, 0.1f);
                float localAmountRemoved = (float)math.min(currentTexelHeight, math.max(weight, 0.0) * targetAmountToRemove * localSoftness);

                actualSedimentRemoved += localAmountRemoved;

                uint2 targetOutPipe = pipeMap[x + pipeMapCenterCoord, y + pipeMapCenterCoord];
                pipesBuffer[affectedNeighborCoordinate.x, affectedNeighborCoordinate.y].outPipes[targetOutPipe.x, targetOutPipe.y] += localAmountRemoved;
            }
        }

        return actualSedimentRemoved;
    }

    void depositSediment(uint2 heightmapDimensions, int2 erosionCenter, float amountToDeposit, CpuTexelPipes[,] pipesBuffer)
    {
        for (int x = -1; x <= 1; ++x)
        {
            for (int y = -1; y <= 1; ++y)
            {
                float increment = depositionMapWeights[x + 1, y + 1] * amountToDeposit;
                uint2 affectedNeighborCoordinates = (uint2)(erosionCenter + new int2(x, y));
                uint2 targetInPipe = pipeMap[x + pipeMapCenterCoord, y + pipeMapCenterCoord];
                pipesBuffer[affectedNeighborCoordinates.x, affectedNeighborCoordinates.y].inPipes[targetInPipe.x, targetInPipe.y] += increment;
            }
        }
    }

    void GarbageCollector(uint2 heightmapDimensions, ErosionParticle[] particlesBuffer, CpuPipelineContext pipelineContext)
    {
        for (uint idx = 0; idx < particlesBuffer.Length; ++idx)
        {
            uint processedParticleIdx = idx;
            if (checkParticleIsValid(heightmapDimensions, particlesBuffer[processedParticleIdx]))
                continue;

            // reinitialize a dead particle
            float2 uniformCoefficients = pipelineContext.GetRandomFloats();
            float dropSize = math.abs(CpuComputeUtilities.sampleGaussBoxMuller(math.max(uniformCoefficients, new float2(CpuComputeUtilities.EPS))));

            particlesBuffer[processedParticleIdx].pos = uniformCoefficients * (float2)heightmapDimensions;
            particlesBuffer[processedParticleIdx].dir = float2.zero;
            particlesBuffer[processedParticleIdx].vel = dropSize;
            particlesBuffer[processedParticleIdx].w = dropSize;
            particlesBuffer[processedParticleIdx].s = 0.0f;
        }
    }

    void ParticlesInitializer(uint2 heightmapDimensions, ErosionParticle[] particlesBuffer, CpuPipelineContext pipelineContext)
    {
        uint2 randomNoiseOffset = (uint2)pipelineContext.GetRandomInts();
        for (uint idx = 0; idx < particlesBuffer.Length; ++idx)
        {
            uint processedParticleIdx = idx;

            float2 uniformCoefficients = pipelineContext.GetRandomFloats();

            // here the pink noise accounts for the 'area' a particle covers.
            // While uniform coordinates guarantee uniform coverage of the entire area of the terrain,
            // pink-distributed size allows to achieve the typical 'rain' texture (implicitly)
            float2 particlePosition = uniformCoefficients * (float2)heightmapDimensions;
            float dropSize = CpuComputeUtilities.pinkNoise2D((uint2)(randomNoiseOffset + particlePosition * rainNoiseFrequency));

            particlesBuffer[processedParticleIdx].pos = particlePosition;
            particlesBuffer[processedParticleIdx].dir = float2.zero;
            particlesBuffer[processedParticleIdx].vel = dropSize;
            particlesBuffer[processedParticleIdx].w = dropSize;
            particlesBuffer[processedParticleIdx].s = 0.0f;
        }
    }

    void Integrator(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, ErosionParticle[] particlesBuffer, CpuTexelPipes[,] pipesBuffer, float erosionDistanceSumPrecompute)
    {
        int2 signedHeightmapDimensions = (int2)intermediateHeightmap.heightmapDimensions;
        for (uint idx = 0; idx < particlesBuffer.Length; ++idx)
        {
            uint processedParticleIdx = idx;
            if (!checkParticleIsValid(intermediateHeightmap.heightmapDimensions, particlesBuffer[processedParticleIdx]))
                continue;

            float2 oldParticlePosition = particlesBuffer[processedParticleIdx].pos;
            float oldParticleVelocity = particlesBuffer[processedParticleIdx].vel;
            float oldSedimentValue = particlesBuffer[processedParticleIdx].s;
            float oldWaterValue = particlesBuffer[processedParticleIdx].w;

            uint2 floorOldParticlePosition = new uint2(math.floor(oldParticlePosition));
            float2 interpolationCoefficients = math.frac(oldParticlePosition);

            int sampleSelf = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, floorOldParticlePosition);
            int sampleRight = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions,
                     (floorOldParticlePosition + new uint2(1, 0)));
            int sampleTop = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (floorOldParticlePosition + new uint2(1, 1)));
            int sampleBottom = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, (floorOldParticlePosition + new uint2(0, 1)));

            float2 currentGradient = new float2(
                (float)(intermediateHeightmap.nativeHeightmapArray[sampleRight] - intermediateHeightmap.nativeHeightmapArray[sampleSelf]) * (float)(1.0 - interpolationCoefficients.y)
                 + (float)(intermediateHeightmap.nativeHeightmapArray[sampleTop] - intermediateHeightmap.nativeHeightmapArray[sampleBottom]) * interpolationCoefficients.y,

                (float)(intermediateHeightmap.nativeHeightmapArray[sampleBottom] - intermediateHeightmap.nativeHeightmapArray[sampleSelf]) * (float)(1.0 - interpolationCoefficients.x)
                + (float)(intermediateHeightmap.nativeHeightmapArray[sampleTop] - intermediateHeightmap.nativeHeightmapArray[sampleRight]) * interpolationCoefficients.x);

            float2 dirCandidate = particlesBuffer[processedParticleIdx].dir * inertia - currentGradient * (1.0f - inertia);
            float2 newParticleDirection = (math.dot(dirCandidate, dirCandidate) > CpuComputeUtilities.EPS) ? math.normalize(dirCandidate) : float2.zero;
            float2 newParticlePosition = oldParticlePosition + newParticleDirection;
            float oldHeight = sampleHeightmapBilinear(intermediateHeightmap, oldParticlePosition);
            float newHeight = sampleHeightmapBilinear(intermediateHeightmap, newParticlePosition);
            float heightDelta = newHeight - oldHeight;

            float sedimentValueUpdate = 0.0f;
            if (heightDelta > PLAIN_HEIGHT_THRESHOLD)
            {
                float amountToDeposit = math.min(oldSedimentValue, math.max(math.abs(intermediateHeightmap.nativeHeightmapArray[sampleSelf] - oldHeight), heightDelta));
                depositSediment(intermediateHeightmap.heightmapDimensions, (int2)floorOldParticlePosition, amountToDeposit, pipesBuffer);
                sedimentValueUpdate = -amountToDeposit;
            }
            else if (heightDelta < -PLAIN_HEIGHT_THRESHOLD)
            {
                float newCapacity = math.max(-heightDelta, minSlope) * oldParticleVelocity * oldWaterValue * capacity;

                if ((oldSedimentValue - newCapacity) > CpuComputeUtilities.EPS)
                {
                    float amountToDeposit = (oldSedimentValue - newCapacity) * deposition;
                    depositSediment(intermediateHeightmap.heightmapDimensions, (int2)floorOldParticlePosition, amountToDeposit, pipesBuffer);
                    sedimentValueUpdate = -amountToDeposit;
                }
                else if ((oldSedimentValue - newCapacity) < -CpuComputeUtilities.EPS)
                {
                    float amountToRemove = math.min((newCapacity - oldSedimentValue) * erosion, -heightDelta);
                    sedimentValueUpdate = removeSediment(intermediateHeightmap, (int2)floorOldParticlePosition, amountToRemove, erosionDistanceSumPrecompute, pipesBuffer);
                }
            }

            float v2 = oldParticleVelocity * oldParticleVelocity - heightDelta * gravity;
            float newParticleVelocity = math.sqrt(math.max(v2, 0.0f));
            float newW = math.max(oldWaterValue * (1.0f - evaporation), 0.0f);

            particlesBuffer[processedParticleIdx].pos = newParticlePosition;
            particlesBuffer[processedParticleIdx].dir = newParticleDirection;
            particlesBuffer[processedParticleIdx].vel = newParticleVelocity;
            particlesBuffer[processedParticleIdx].w = newW;
            particlesBuffer[processedParticleIdx].s = math.max(oldSedimentValue + sedimentValueUpdate, 0.0f);
        }
    }

    void ChangeResolver(CpuPipelineContext.CpuIntermediateHeightmap intermediateHeightmap, ErosionParticle[] particlesBuffer, CpuTexelPipes[,] pipesBuffer)
    {
        for (uint x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (uint y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                uint2 targetTexel = new uint2(x, y);
                int targetTexelLin = (int)CpuComputeUtilities.index2dTo1d(intermediateHeightmap.heightmapDimensions, targetTexel);
                float totalSedimentRemoved = 0.0f;
                float totalSedimentAdded = 0.0f;
                for (int pX = 0; pX < 3; ++pX)
                {
                    for (int pY = 0; pY < 3; ++pY)
                    {
                        totalSedimentRemoved += pipesBuffer[targetTexel.x, targetTexel.y].outPipes[pX, pY];
                        totalSedimentAdded += pipesBuffer[targetTexel.x, targetTexel.y].inPipes[pX, pY];
                    }
                }

                intermediateHeightmap.nativeHeightmapArray[targetTexelLin] = math.max(intermediateHeightmap.nativeHeightmapArray[targetTexelLin] + totalSedimentAdded - totalSedimentRemoved, 0.0f);
            }
        }
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numActualParticles = (int)math.pow(2, numSimultaneousParticles);
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();
        Assert.IsTrue(erosionNeighborhood <= pipeMapCenterCoord);

        {
            if (particlesBuffer is null || particlesBuffer.Length != numActualParticles)
            {
                particlesBuffer = new ErosionParticle[numActualParticles];
            }
        }

        {
            if (pipesBuffer is null || pipesBuffer.GetLength(0) != textureSize || pipesBuffer.GetLength(1) != textureSize)
            {
                pipesBuffer = new CpuTexelPipes[textureSize, textureSize];
                for (int x = 0; x < textureSize; ++x)
                {
                    for (int y = 0; y < textureSize; ++y)
                    {
                        pipesBuffer[x, y].inPipes = new float[3, 3];
                        pipesBuffer[x, y].outPipes = new float[3, 3];
                    }
                }
            }
        }

        {
            int garbageCollectorRunPeriod = (int)math.floor(math.log2(2 * waterDeathThreshold) / math.log2(1.0f - evaporation));
            Assert.IsTrue(math.abs(evaporation) >= 1.0e-5 && math.abs(waterDeathThreshold - 0.5) >= 1.0e-5, "Broken GC period");
            Debug.LogWarning("GC period: " + garbageCollectorRunPeriod);
            float erosionDistanceSumPrecompute = precomputeDistanceSum();
            var pipePlumber = new Action(() =>
            {
                for (int x = 0; x < textureSize; x++)
                    for (int y = 0; y < textureSize; y++)
                    {
                        Array.Clear(pipesBuffer[x, y].outPipes, 0, 9);
                        Array.Clear(pipesBuffer[x, y].inPipes, 0, 9);
                    }
            });
            for (int w = 0; w < numSimulationWaves; ++w)
            {
                ParticlesInitializer(intermediateHeightmap.heightmapDimensions, particlesBuffer, pipelineContext);
                pipePlumber();
                {
                    int gcRunInsertionPeriod = 0;
                    for (int s = 0; s < numSimulationSteps; ++s)
                    {
                        Integrator(intermediateHeightmap, particlesBuffer, pipesBuffer, erosionDistanceSumPrecompute);
                        ChangeResolver(intermediateHeightmap, particlesBuffer, pipesBuffer);
                        pipePlumber();
                        if (s / garbageCollectorRunPeriod != gcRunInsertionPeriod)
                        {
                            GarbageCollector(intermediateHeightmap.heightmapDimensions, particlesBuffer, pipelineContext);
                            gcRunInsertionPeriod = s / garbageCollectorRunPeriod;
                        }
                    }
                }
            }
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

        GUILayout.Label("Simulation");

        GUILayout.Label($"Particles: {numSimultaneousParticles}");
        numSimultaneousParticles = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(numSimultaneousParticles, 7f, 32f)
        );

        GUILayout.Label($"Steps: {numSimulationSteps}");
        numSimulationSteps = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(numSimulationSteps, 1f, 1000f)
        );

        GUILayout.Label($"Waves: {numSimulationWaves}");
        numSimulationWaves = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(numSimulationWaves, 1f, 10f)
        );

        GUILayout.Space(5);
        GUILayout.Label("Physics");

        GUILayout.Label($"Inertia: {inertia:F3}");
        inertia = GUILayout.HorizontalSlider(inertia, 0.01f, 1.0f);

        GUILayout.Label($"Gravity: {gravity:F3}");
        gravity = GUILayout.HorizontalSlider(gravity, 0.01f, 10.0f);

        GUILayout.Label($"Evaporation: {evaporation:F3}");
        evaporation = GUILayout.HorizontalSlider(evaporation, 0.01f, 1.0f);

        GUILayout.Space(5);
        GUILayout.Label("Sediment");

        GUILayout.Label($"Capacity: {capacity:F2}");
        capacity = GUILayout.HorizontalSlider(capacity, 0.01f, 64.0f);

        GUILayout.Label($"Min Slope: {minSlope:F4}");
        minSlope = GUILayout.HorizontalSlider(minSlope, 0.0001f, 1.0f);

        GUILayout.Label($"Deposition: {deposition:F3}");
        deposition = GUILayout.HorizontalSlider(deposition, 0.01f, 1.0f);

        GUILayout.Label($"Erosion: {erosion:F3}");
        erosion = GUILayout.HorizontalSlider(erosion, 0.01f, 1.0f);

        GUILayout.Space(5);
        GUILayout.Label("Lifetime");

        GUILayout.Label($"Water Death Threshold: {waterDeathThreshold:F3}");
        waterDeathThreshold = GUILayout.HorizontalSlider(waterDeathThreshold, 0.01f, 0.5f);

        GUILayout.Space(5);
        GUILayout.Label("Neighborhood");

        GUILayout.Label($"Erosion Neighborhood: {erosionNeighborhood}");
        erosionNeighborhood = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(erosionNeighborhood, 1f, 3f)
        );

        GUILayout.Space(5);
        GUILayout.Label("Noise");

        GUILayout.Label($"Rain Noise Frequency: {rainNoiseFrequency:F3}");
        rainNoiseFrequency = GUILayout.HorizontalSlider(rainNoiseFrequency, 1.0f, 10.0f);

        GUILayout.EndScrollView();
    }

    public override string GUIStepTitle() => "PHE eroder";

    public override void FreeResources()
    {
        pipesBuffer = null;
        particlesBuffer = null;

        gpuPipesBuffer?.Release();
        gpuParticlesBuffer?.Release();
    }

    public override void RandomizeParameters(System.Random random)
    {
        //numSimultaneousParticles -> ignored to minimize the number of conflicts
        // erosionNeighborhood
        // numSimulationSteps
        // numSimulationWaves
        // rainNoiseFrequency

        inertia = (float)math.max(random.NextDouble(), 0.01);
        capacity = (float)math.max(random.NextDouble() * 64.0, 0.01);
        minSlope = (float)math.max(random.NextDouble(), 0.01);
        deposition = (float)math.max(random.NextDouble(), 0.01);
        erosion = (float)math.max(random.NextDouble(), 0.01);
        gravity = (float)math.max(random.NextDouble() * 10.0, 0.01);
        evaporation = (float)math.max(random.NextDouble(), 0.01);
        waterDeathThreshold = math.clamp((float)random.NextDouble(), 0.01f, 0.5f);
    }
}
