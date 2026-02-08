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

    [Range(0.01f, 1.0f)]
    public float talusThreshold = 0.15f;

    [Range(0, 3)]
    public int groupScaleFactor = 0;

    [Range(1, 100)]
    public int erosionIterationLimit = 25;

    private const int groupSize = 32;

    // Property IDs (cached)
    private static readonly int PID_resultHeightmap = Shader.PropertyToID("resultHeightmap");
    private static readonly int PID_permuteA = Shader.PropertyToID("permuteA");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_distributionCoefficient = Shader.PropertyToID("distributionCoefficient");
    private static readonly int PID_talusThreshold = Shader.PropertyToID("talusThreshold");
    private static readonly int PID_heightmapDimensions = Shader.PropertyToID("heightmapDimensions");
    private static readonly int PID_noiseDisplacement = Shader.PropertyToID("noiseDisplacement");

    public override InputExpectations GetStepExpectations()
        => InputExpectations.HeightMapNormalized; // TODO: skip normalization by adjusting talus threshold to effective height range

    public override void StepInitialization(PipelineContext pipelineContext) { }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");

        int coreKernelIdx = erosionComputeShader.FindKernel("ThermalCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("ThermalBorderEroder");

        int texelsPerThreadSquared = (int)math.pow(textureSize / numLinearThreads, 2);
        int permuteA = 1;
        while (true)
        {
            permuteA += 2; //only odd numbers have a chance
            int gcd = 0;
            for (gcd = permuteA; gcd > 0; --gcd)
            {
                if ((texelsPerThreadSquared % gcd) == 0 && (permuteA % gcd) == 0)
                {
                    //largest so far, no need to seek further
                    break;
                }
            }
            if (gcd == 1)
            {
                break;
            }
        }

        foreach (int kernelIdx in new[] { coreKernelIdx, borderKernelIdx })
        {
            pipelineContext.BindTexture(erosionComputeShader, kernelIdx, PID_resultHeightmap, pipelineContext.intermediateHeightmap);
        }

        pipelineContext.SetUniformInt(erosionComputeShader, PID_texelsPerThread, textureSize / numLinearThreads);
        pipelineContext.SetUniformInt(erosionComputeShader, PID_permuteA, permuteA);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_distributionCoefficient, distributionCoefficient);
        pipelineContext.SetUniformFloat(erosionComputeShader, PID_talusThreshold, talusThreshold);

        pipelineContext.SetUniformInts(erosionComputeShader, PID_heightmapDimensions, new int[2] { textureSize, textureSize });

        var dispatchGroups = new Vector3(numGroups, numGroups, 1);
        for (int i = 0; i < erosionIterationLimit; ++i)
        {
            pipelineContext.SetRandomFloats(erosionComputeShader, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, coreKernelIdx, dispatchGroups);
            pipelineContext.SetRandomFloats(erosionComputeShader, PID_noiseDisplacement);
            pipelineContext.AppendDispatchToCommandBuffer(erosionComputeShader, borderKernelIdx, dispatchGroups);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext) { }
}
