using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using GpuGenerationPipeline;
using CpuGenerationPipeline;
using System;
using System.Threading.Tasks;

public class UNDispatcher : UltimatePipelineStep
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(0.001f, 2.0f)]
    public float noiseFrequency = 0.001f;

    [Range(0.01f, 1.0f)]
    public float persistence = 0.2f;

    [Range(-1.0f, 1.0f)]
    public float sharpness = 0.0f;

    [Range(0.001f, 10.0f)]
    public float slopeErosion = 0.001f;

    [Range(1, 20)]
    public int numOctaves = 5;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength = 0.01f;

    private static readonly int PID_Result = Shader.PropertyToID("Result");
    private static readonly int PID_texelsPerThread = Shader.PropertyToID("texelsPerThread");
    private static readonly int PID_noiseFrequency = Shader.PropertyToID("noiseFrequency");
    private static readonly int PID_sharpness = Shader.PropertyToID("sharpness");
    private static readonly int PID_slopeErosion = Shader.PropertyToID("slopeErosion");
    private static readonly int PID_perturbationStrength = Shader.PropertyToID("perturbationStrength");
    private static readonly int PID_numOctaves = Shader.PropertyToID("numOctaves");
    private static readonly int PID_persistence = Shader.PropertyToID("persistence");
    private static readonly int PID_randomFloats = Shader.PropertyToID("randomFloats");
    private static readonly int PID_textureDimensions = Shader.PropertyToID("textureDimensions");

    public override void ExecuteStepGpu(PipelineContext pipelineContext)
    {
        int textureSize = pipelineContext.GetHeightmapSize();
        var (optimalGroupSize, numGroups) = GenerationUtilities.GetOptimalNumberOfGroups(textureSize, new int[] { pipelineContext.preferredGlobalGroupSize }, Int32.MaxValue, 1);
        int numLinearThreads = optimalGroupSize * numGroups;

        Assert.IsTrue(textureSize % numLinearThreads == 0, "Texels must be distributed among threads evenly!");

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");

        pipelineContext.BindTexture(shaderToDispatch, kernelIdx, PID_Result, pipelineContext.intermediateHeightmap);

        {
            pipelineContext.SetUniformInt(shaderToDispatch, PID_texelsPerThread, textureSize / numLinearThreads);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_noiseFrequency, noiseFrequency);
        }

        {
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_sharpness, sharpness);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_slopeErosion, slopeErosion);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_perturbationStrength, perturbationStrength);
            pipelineContext.SetUniformInt(shaderToDispatch, PID_numOctaves, numOctaves);
            pipelineContext.SetUniformFloat(shaderToDispatch, PID_persistence, persistence);
            pipelineContext.SetRandomFloats(shaderToDispatch, PID_randomFloats);
            pipelineContext.SetUniformInts(shaderToDispatch, PID_textureDimensions, new int[] { textureSize, textureSize });
        }

        pipelineContext.AppendDispatchToCommandBuffer(
            shaderToDispatch,
            kernelIdx,
            new Vector3(numGroups, numGroups, 1)
        );
    }

    public override InputExpectations GetStepExpectationsGpu() => InputExpectations.None;

    public override void UpdateHeightmapStateGpu(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }

    public override Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
    {
        // return Task.Run(() =>
        // {

        // });
        var intermediateHeightmap = pipelineContext.GetCpuIntemediateHeightmap();

        float2 extraDisplacement = (float2)intermediateHeightmap.heightmapDimensions * pipelineContext.GetRandomFloats();

        for (int x = 0; x < intermediateHeightmap.heightmapDimensions.x; ++x)
        {
            for (int y = 0; y < intermediateHeightmap.heightmapDimensions.y; ++y)
            {
                float2 floatIdx = extraDisplacement + new float2(x, y);

                float2 slopeErosionDerivativeSum = (float2)math.clamp(new float2(noise.snoise(floatIdx)), -0.15, 0.15).xx;
                float2 perturbDerivativeSum = float2.zero;
                float noiseResult = 0.0f;

                float octaveAmplitude = 1.0f;
                float octaveFrequency = noiseFrequency;
                float2 samplePosition = float2.zero;

                for (int o = 0; o < numOctaves; ++o)
                {
                    samplePosition = (floatIdx * octaveFrequency) + perturbDerivativeSum;

                    float3 startNoiseValue = noise.psrdnoise(samplePosition, new float2(1e15f), 0.0f);
                    float featureNoise = startNoiseValue.x;
                    float2 featureDerivative = startNoiseValue.yz;

                    float ridgedNoise = 1.0f - math.abs(featureNoise);
                    float billowNoise = ridgedNoise * ridgedNoise;
                    featureNoise = (float)math.lerp(featureNoise, billowNoise, math.max(0.0, sharpness));
                    featureNoise = (float)math.lerp(featureNoise, ridgedNoise, math.abs(math.min(0.0, sharpness)));

                    slopeErosionDerivativeSum += featureDerivative * slopeErosion;
                    perturbDerivativeSum += octaveAmplitude * perturbationStrength * featureDerivative;
                    float derivativeNorm = math.dot(slopeErosionDerivativeSum, slopeErosionDerivativeSum);
                    noiseResult += octaveAmplitude * featureNoise * (1.0f / (1.0f + derivativeNorm));

                    octaveAmplitude *= persistence;
                    octaveFrequency *= 2.0f;
                }

                intermediateHeightmap.nativeHeightmapArray[(int)(x * intermediateHeightmap.heightmapDimensions.y + y)] = noiseResult;
            }
        }
        return Task.CompletedTask;
    }

    public override CpuInputExpectations GetStepExpectationsCpu() => CpuInputExpectations.None;

    public override void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
    {
        previousState &= ~CpuHeightmapProperties.Normalized;
    }

    public override void RenderParametersTuningGUI()
    {
        GUILayout.Label("Noise");

        GUILayout.Label($"Frequency: {noiseFrequency:F4}");
        noiseFrequency = GUILayout.HorizontalSlider(noiseFrequency, 0.001f, 2.0f);

        GUILayout.Label($"Persistence: {persistence:F3}");
        persistence = GUILayout.HorizontalSlider(persistence, 0.01f, 1.0f);

        GUILayout.Label($"Octaves: {numOctaves}");
        numOctaves = Mathf.RoundToInt(
            GUILayout.HorizontalSlider(numOctaves, 1f, 20f)
        );

        GUILayout.Space(5);
        GUILayout.Label("Shape");

        GUILayout.Label($"Sharpness: {sharpness:F3}");
        sharpness = GUILayout.HorizontalSlider(sharpness, -1.0f, 1.0f);

        GUILayout.Space(5);
        GUILayout.Label("Erosion");

        GUILayout.Label($"Slope Erosion: {slopeErosion:F4}");
        slopeErosion = GUILayout.HorizontalSlider(slopeErosion, 0.001f, 10.0f);

        GUILayout.Space(5);
        GUILayout.Label("Perturbation");

        GUILayout.Label($"Strength: {perturbationStrength:F3}");
        perturbationStrength = GUILayout.HorizontalSlider(perturbationStrength, 0.01f, 10.0f);
    }

    public override string GUIStepTitle() => "UN generator";
}
