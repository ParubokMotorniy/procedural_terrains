using System;
using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

public class CellularHydraulicErosionDispatcher : MultiFormatPipelineStep
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


    private const int groupSize = 32;

    private ComputeBuffer texelParametersBuffer;
    private ComputeBuffer waterPipesBuffer;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_waterLevel = Shader.PropertyToID("texelParameters");
    private static readonly int PID_pipesBuffer = Shader.PropertyToID("pipesBuffer");
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

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized;

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { groupSize }, Int32.MaxValue, 4);
        int numLinearThreads = optimalGroupSize * numGroups;

        int texelsPerThread = textureSize / numLinearThreads;
        var dispatchGroups = new Vector3(numGroups, numGroups, 1);

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");
        Assert.IsTrue(texelsPerThread >= 4, "A thread must have at least 4 texels to porcess");

        {
            int neededBufferSize = textureSize * textureSize;
            if (texelParametersBuffer is null || !texelParametersBuffer.IsValid() || texelParametersBuffer.count != neededBufferSize)
            {
                texelParametersBuffer = new ComputeBuffer(neededBufferSize, 2 * sizeof(float));
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

    public override void StepConclusion(PipelineContext pipelineContext) { }

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
    }
}
