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

    private const int groupSize = 64;

    //TODO: add extra field for limiting number of steps
    [StructLayout(LayoutKind.Sequential)]
    private struct ErosionParticle
    {
        Vector2 pos;
        Vector2 dir;
        Vector2 vel;
        float w;
        float s;
    };

    private ComputeBuffer particlesBuffer;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_particlesBuffer = Shader.PropertyToID("particlesBuffer");
    private static readonly int PID_particlesPerThread = Shader.PropertyToID("particlesPerThread");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_randomSeeds = Shader.PropertyToID("randomSeeds");

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

        particlesBuffer = new ComputeBuffer(numSimultaneousParticles, Marshal.SizeOf<ErosionParticle>());
        Assert.IsTrue(particlesBuffer.IsValid());

        foreach (int kernelIdx in new[] { particlesInitializerKernelIdx, integratorKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_particlesBuffer, particlesBuffer);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_particlesPerThread, particlesPerThread);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[] { textureSize, textureSize });

        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, particlesInitializerKernelIdx, dispatchGroups);
        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, integratorKernelIdx, dispatchGroups);
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
