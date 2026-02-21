using GenerationPipeline;
using UnityEngine.Assertions;
using Unity.Mathematics;
using UnityEngine;

public class ThermalErosionDispatcher : MultiFormatPipelineStep
{
    [SerializeField]
    public ComputeShader erosionComputeShader;

    [Range(0.01f, 1.0f)]
    public float distributionCoefficient = 0.75f;

    [Range(0.001f, 1.0f)]
    public float talusThreshold = 0.15f;

    [Range(1, 3)]
    public int groupScaleFactor = 0;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    private const int groupSize = 32;

    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_permuteACore = Shader.PropertyToID("permuteACore");
    private static readonly int PID_permuteAStripH = Shader.PropertyToID("permuteAStripH");
    private static readonly int PID_permuteAStripV = Shader.PropertyToID("permuteAStripV");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_distributionCoefficient = Shader.PropertyToID("distributionCoefficient");
    private static readonly int PID_talusThreshold = Shader.PropertyToID("talusThreshold");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_iterationIdx = Shader.PropertyToID("iterationIdx");

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized; // TODO: skip normalization by adjusting talus threshold to effective height range

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;
        int texelsPerThread = textureSize / numLinearThreads;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");
        Assert.IsTrue(texelsPerThread >= 4, "A thread must have at least 4 texels to porcess");

        int texelsPerThreadSquared = (int)math.pow(texelsPerThread - 2, 2);
        int permuteACore = GenerationUtilities.ComputeCoprime(texelsPerThreadSquared, 17);
        int permuteAStripH = GenerationUtilities.ComputeCoprime(2 * texelsPerThread, 29);
        int permuteAStripV = GenerationUtilities.ComputeCoprime(2 * (texelsPerThread - 2), 35);

        Debug.Log(permuteACore);
        Debug.Log(permuteAStripH);
        Debug.Log(permuteAStripV);

        int coreKernelIdx = erosionComputeShader.FindKernel("ThermalCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("ThermalBorderEroder");

        foreach (int kernelIdx in new[] { coreKernelIdx, borderKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, texelsPerThread);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteACore, permuteACore);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripH, permuteAStripH);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteAStripV, permuteAStripV);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_distributionCoefficient, distributionCoefficient);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_talusThreshold, talusThreshold);
        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });

        var dispatchGroups = new Vector3(numGroups, numGroups, 1);
        for (int i = 0; i < erosionIterationLimit; ++i)
        {
            pipelineContext.SetUniformInt(erosionComputeShader, PID_iterationIdx, i + 1);

            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
