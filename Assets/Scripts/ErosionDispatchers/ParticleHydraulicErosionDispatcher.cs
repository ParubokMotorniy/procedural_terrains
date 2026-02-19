using System;
using System.Runtime.InteropServices;
using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class ParticleHydraulicErosionDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(1, 3)]
    public int groupScaleFactor = 0;

    [Range(7, 32)]
    public int numSimultaneousParticles = 10;

    [Range(0.01f, 1.0f)]
    public float inertia = 0.1f;

    [Range(0.01f, 100.0f)]
    public float capacity = 10.0f;

    [Range(0.01f, 1.0f)]
    public float minSlope = 0.03f;

    [Range(0.01f, 1.0f)]
    public float deposition = 0.7f;

    [Range(0.01f, 1.0f)]
    public float erosion = 0.35f;

    [Range(0.01f, 1.0f)]
    public float gravity = 0.4f;

    [Range(0.01f, 1.0f)]
    public float evaporation = 0.02f;

    [Range(0.01f, 0.45f)]
    public float waterDeathThreshold = 0.05f;

    [Range(1, 3)]
    public int erosionNeighborhood = 1;

    [Range(1, 1000)]
    public int numSimulationSteps = 1;

    [Range(1, 10)]
    public int numSimulationWaves = 1;

    private const int integrateGroupSize = 64;
    private const int resolveGroupSize = 32;

    [StructLayout(LayoutKind.Sequential)]
    private struct ErosionParticle
    {
        Vector2 pos;
        Vector2 dir;
        float vel;
        float w;
        float s;
    };

    private ComputeBuffer particlesBuffer;

    struct TexelPipes
    {
        Vector3 inPipes1;
        Vector3 inPipes2;
        Vector3 inPipes3;
        Vector3 outPipes1;
        Vector3 outPipes2;
        Vector3 outPipes3;
    };

    private ComputeBuffer pipesBuffer;

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

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numActualParticles = (int)math.pow(2, numSimultaneousParticles);
        int particlesPerThread = numActualParticles / (integrateGroupSize * numGroups);
        int numTexelsPerThread = textureSize / (numGroups * resolveGroupSize);

        var integrateDispatchGroups = new Vector3(numGroups, 1, 1);
        var resolveDispatchGroups = new Vector3(numGroups, numGroups, 1);

        Assert.IsTrue(numActualParticles % (integrateGroupSize * numGroups) == 0, "Particles must be distributed among threads evenly!");
        Assert.IsTrue(textureSize % (numGroups * resolveGroupSize) == 0, "Texels must be distributed among threads evenly!");

        int particlesInitializerKernelIdx = erosionComputeShader.FindKernel("ParticlesInitializer");
        int changeResolverKernelIdx = erosionComputeShader.FindKernel("ChangeResolver");
        int pipesInitializerKernelIdx = erosionComputeShader.FindKernel("PipesInitializer");
        int integratorKernelIdx = erosionComputeShader.FindKernel("Integrator");
        int garbageCollectorKernelIdx = erosionComputeShader.FindKernel("GarbageCollector");

        {
            Debug.LogWarning("Size of a particle struct: " + Marshal.SizeOf<ErosionParticle>());
            particlesBuffer = new ComputeBuffer(numActualParticles, Marshal.SizeOf<ErosionParticle>());
            Assert.IsTrue(particlesBuffer.IsValid());
        }

        {
            Debug.LogWarning("Size of a pipe struct: " + Marshal.SizeOf<TexelPipes>());
            pipesBuffer = new ComputeBuffer(textureSize * textureSize, Marshal.SizeOf<TexelPipes>());
            Assert.IsTrue(pipesBuffer.IsValid());
        }

        foreach (int kernelIdx in new[] { pipesInitializerKernelIdx, particlesInitializerKernelIdx, integratorKernelIdx, garbageCollectorKernelIdx, changeResolverKernelIdx })
        {
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_particlesBuffer, particlesBuffer);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_pipesBuffer, pipesBuffer);
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        float erosionDistanceSumPrecompute = 0.0f;
        {
            float actualNeighborRadius = 1.5f * erosionNeighborhood;
            for (int x = -1 * erosionNeighborhood; x <= erosionNeighborhood; ++x)
            {
                for (int y = -1 * erosionNeighborhood; y <= erosionNeighborhood; ++y)
                {
                    erosionDistanceSumPrecompute += actualNeighborRadius - math.sqrt(x * x + y * y);
                }
            }
        }

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

        {
            int garbageCollectorRunPeriod = (int)math.floor(math.log2(2 * waterDeathThreshold) / math.log2(1.0f - evaporation));
            Assert.IsTrue(math.abs(evaporation) >= 1.0e-5 && math.abs(waterDeathThreshold - 0.5) >= 1.0e-5, "Broken GC period");
            Debug.LogWarning("GC period: " + garbageCollectorRunPeriod);
            for (int w = 0; w < numSimulationWaves; ++w)
            {
                pipelineContext.SetRandomInts(erosionComputeShader, PID_randomInts);
                pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, particlesInitializerKernelIdx, integrateDispatchGroups);

                {
                    int gcRunInsertionPeriod = 0;
                    for (int s = 0; s < numSimulationSteps; ++s)
                    {
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, pipesInitializerKernelIdx, resolveDispatchGroups);
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, integratorKernelIdx, integrateDispatchGroups);
                        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, changeResolverKernelIdx, resolveDispatchGroups);
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

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
