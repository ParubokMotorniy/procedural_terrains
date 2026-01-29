using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;

public class ThermalErosionDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(0.01f, 1.0f)]
    public float distributionCoefficient = 0.75f;

    [Range(0.01f, 1.0f)]
    public float talusThreshold = 0.15f;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(1, 100)]
    public int erosionIterationLimit = 25; // TODO: border update serialization scheme may remove iterative CPU dispatching

    private const int groupSize = 32;

    // Property IDs (cached)
    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_distributionCoefficient = Shader.PropertyToID("distributionCoefficient");
    private static readonly int PID_talusThreshold = Shader.PropertyToID("talusThreshold");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized; // TODO: skip normalization by adjusting talus threshold to effective height range

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        int coreKernelIdx = erosionComputeShader.FindKernel("ThermalCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("ThermalBorderEroder");

        foreach (int kernelIdx in new[] { coreKernelIdx, borderKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, textureSize / numLinearThreads);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_distributionCoefficient, distributionCoefficient);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_talusThreshold, talusThreshold);

        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });

        var dispatchGroups = new Vector3(numGroups, numGroups, 1);
        for (int i = 0; i < erosionIterationLimit; ++i)
        {
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
