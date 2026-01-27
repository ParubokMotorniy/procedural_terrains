using GenerationPipeline;
using Unity.Mathematics;
using UnityEngine;

public class ThermalErosionDispatcher : GenerationPipeline.MultiFormatPipelineStep
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
    public int erosionIterationLimit = 25; //TODO: a complex border update serialization scheme can allow to get rid of iterative dispatching on CPU

    private readonly int groupSize = 32;

    public override InputExpectations GetStepExpectations()
    {
        return InputExpectations.HeightMapNormalized; //TODO: I can save on normalization if the talus threshold is adjusted to the effective range of heights emerging from the previous stage.
    }

    public override void StepBody(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        int numGroups = (int)math.pow(2, groupScaleFactor);
        int numLinearThreads = groupSize * numGroups;

        int coreKernelIdx = erosionComputeShader.FindKernel("ThermalCoreEroder");
        int borderKernelIdx = erosionComputeShader.FindKernel("ThermalBorderEroder");

        erosionComputeShader.SetTexture(coreKernelIdx, Shader.PropertyToID("resultHeightmap"), pipelineContext.intermediateHeightmap);
        erosionComputeShader.SetTexture(borderKernelIdx, Shader.PropertyToID("resultHeightmap"), pipelineContext.intermediateHeightmap);

        erosionComputeShader.SetInt("texelsPerThread", textureSize / numLinearThreads);
        erosionComputeShader.SetFloat("distributionCoefficient", distributionCoefficient);
        erosionComputeShader.SetFloat("talusThreshold", talusThreshold);
        erosionComputeShader.SetInts("heightmapDimensions", new int[2] { textureSize, textureSize });

        for (int d = 0; d < erosionIterationLimit; ++d)
        {
            erosionComputeShader.Dispatch(coreKernelIdx, numGroups, numGroups, 1);
            erosionComputeShader.Dispatch(borderKernelIdx, numGroups, numGroups, 1);
        }
    }

    public override void StepConclusion(PipelineContext pipelineContext)
    {

    }

    public override void StepInitialization(PipelineContext pipelineContext)
    {

    }
}
