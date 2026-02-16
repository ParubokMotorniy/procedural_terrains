using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class CellularHydraulicErosionDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(1, 3)]
    public int groupScaleFactor = 0;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    [Range(0.001f, 1.0f)]
    public float solubilityConstant;

    [Range(0.001f, 1.0f)]
    public float evaporationConstant;

    private const int groupSize = 32;
    private ComputeBuffer waterLevelBuffer;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_waterLevel = Shader.PropertyToID("waterLevel");
    private static readonly int PID_permuteACore = Shader.PropertyToID("permuteACore");
    private static readonly int PID_permuteAStripH = Shader.PropertyToID("permuteAStripH");
    private static readonly int PID_permuteAStripV = Shader.PropertyToID("permuteAStripV");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_solubilityConstant = Shader.PropertyToID("solubilityConstant");
    private static readonly int PID_evaporationConstant = Shader.PropertyToID("evaporationConstant");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_noiseDisplacement = Shader.PropertyToID("noiseDisplacement");
    private static readonly int PID_iterationIdx = Shader.PropertyToID("iterationIdx");

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;
        int texelsPerThread = textureSize / numLinearThreads;
        var dispatchGroups = new Vector3(numGroups, numGroups, 1);

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");
        Assert.IsTrue(texelsPerThread >= 4, "A thread must have at least 4 texels to porcess");

        int texelsPerThreadSquared = (int)math.pow(texelsPerThread - 2, 2);
        int permuteACore = GenerationUtilities.ComputeCoprime(texelsPerThreadSquared, 3);
        int permuteAStripH = GenerationUtilities.ComputeCoprime(2 * texelsPerThread, 7);
        int permuteAStripV = GenerationUtilities.ComputeCoprime(2 * (texelsPerThread - 2), 11);

        Debug.Log(texelsPerThreadSquared + " coprime with " + permuteACore);
        Debug.Log((2 * texelsPerThread) + " coprime with " + permuteAStripH);
        Debug.Log((2 * (texelsPerThread - 2)) + " coprime with " + permuteAStripV);

        int rainDropKernelIdx = erosionComputeShader.FindKernel("RainDropper");
        int coreKernelIdx = erosionComputeShader.FindKernel("HydraulicCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("HydraulicBorderEroder");
        int waterEvaporatorKernelIdx = erosionComputeShader.FindKernel("WaterEvaporator");
        int resourceInitializerKernelIdx = erosionComputeShader.FindKernel("ResourceInitializer");
        int finalWaterEvaporatorKernelIdx = erosionComputeShader.FindKernel("FinalWaterEvaporator");

        waterLevelBuffer = new ComputeBuffer(textureSize * textureSize, sizeof(float));
        Assert.IsTrue(waterLevelBuffer.IsValid());

        foreach (int kernelIdx in new[] { rainDropKernelIdx, coreKernelIdx, borderKernelIdx, waterEvaporatorKernelIdx, resourceInitializerKernelIdx, finalWaterEvaporatorKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
            pipelineContext.BindComputeBuffer(erosionComputeShader, kernelIdx, PID_waterLevel, waterLevelBuffer);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, texelsPerThread);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteACore, permuteACore);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripH, permuteAStripH);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripV, permuteAStripV);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_evaporationConstant, evaporationConstant);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_solubilityConstant, solubilityConstant);

        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, resourceInitializerKernelIdx, dispatchGroups);
        for (int d = 0; d < erosionIterationLimit; ++d)
        {
            pipelineContext.SetRandomFloats(erosionComputeShader, PID_noiseDisplacement);
            pipelineContext.SetUniformInt(erosionComputeShader, PID_iterationIdx, d);

            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, rainDropKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, waterEvaporatorKernelIdx, dispatchGroups);
        }
        pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, finalWaterEvaporatorKernelIdx, dispatchGroups);
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
