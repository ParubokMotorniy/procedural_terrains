using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class CellularHydraulicErosionDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    [Range(0.01f, 1.0f)]
    public float solubilityConstant;

    [Range(0.01f, 1.0f)]
    public float evaporationConstant;

    private const int groupSize = 32;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_waterLevel = Shader.PropertyToID("waterLevel");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_solubilityConstant = Shader.PropertyToID("solubilityConstant");
    private static readonly int PID_evaporationConstant = Shader.PropertyToID("evaporationConstant");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_noiseDisplacement = Shader.PropertyToID("noiseDisplacement");

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        int rainDropKernelIdx = erosionComputeShader.FindKernel("RainDropper");
        int coreKernelIdx = erosionComputeShader.FindKernel("HydraulicCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("HydraulicBorderEroder");
        int waterEvaporatorKernelIdx = erosionComputeShader.FindKernel("WaterEvaporator");

        ComputeBuffer waterLevelBuffer = new ComputeBuffer(textureSize * textureSize, sizeof(float));
        Assert.IsTrue(waterLevelBuffer.IsValid());

        foreach (int kernelIdx in new[] { rainDropKernelIdx, coreKernelIdx, borderKernelIdx, waterEvaporatorKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_waterLevel, waterLevelBuffer);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, textureSize / numLinearThreads);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_evaporationConstant, evaporationConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_solubilityConstant, solubilityConstant);

        var dispatchGroups = new Vector3(numGroups, numGroups, 1);
        for (int d = 0; d < erosionIterationLimit; ++d)
        {
            pipelineContext.SetRandomFloats(erosionComputeShader, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, rainDropKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, waterEvaporatorKernelIdx, dispatchGroups);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
