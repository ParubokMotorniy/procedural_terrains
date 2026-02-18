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

    [Range(100, 1000)]
    public int numSimultaneousParticles = 0;

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

    [Range(0, 3)]
    public int erosionNeighborhood = 0;

    [Range(1, 1000)]
    public int numSimulationSteps = 1;

    private const int groupSize = 64;

    //TODO: add extra field for limiting number of steps
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

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_particlesBuffer = Shader.PropertyToID("particlesBuffer");
    private static readonly int PID_particlesPerThread = Shader.PropertyToID("particlesPerThread");
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
        int numLinearThreads = groupSize * numGroups;
        int particlesPerThread = numSimultaneousParticles / numLinearThreads;
        var dispatchGroups = new Vector3(numGroups, 1, 1);

        Assert.IsTrue(numSimultaneousParticles % numLinearThreads == 0, "Particles must be distributed among threads evenly!");

        int particlesInitializerKernelIdx = erosionComputeShader.FindKernel("ParticlesInitializer");
        int integratorKernelIdx = erosionComputeShader.FindKernel("Integrator");

        Debug.LogWarning("Size of a particle struct: " + Marshal.SizeOf<ErosionParticle>());
        particlesBuffer = new ComputeBuffer(numSimultaneousParticles, Marshal.SizeOf<ErosionParticle>());
        Assert.IsTrue(particlesBuffer.IsValid());

        foreach (int kernelIdx in new[] { particlesInitializerKernelIdx, integratorKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_particlesBuffer, particlesBuffer);
        }

        float erosionDistanceSumPrecompute = 0.0f;
        for (int x = -erosionNeighborhood; x <= erosionNeighborhood; ++x)
        {
            for (int y = -erosionNeighborhood; y <= erosionNeighborhood; ++y)
            {
                erosionDistanceSumPrecompute += erosionNeighborhood - math.sqrt(x * x + y * y);
            }
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_particlesPerThread, particlesPerThread);
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
        pipelineContext.SetRandomInts(erosionComputeShader, PID_randomInts);

        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, particlesInitializerKernelIdx, dispatchGroups);
        for (int s = 0; s < numSimulationSteps; ++s)
        {
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, integratorKernelIdx, dispatchGroups);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
