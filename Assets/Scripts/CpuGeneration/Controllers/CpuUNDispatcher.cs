using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using CpuGenerationPipeline;
using System;
using System.Threading.Tasks;

public class CpuUNDispatcher : CpuMonoPipelineStep
{

    [Range(0.001f, 2.0f)]
    public float noiseFrequency = 0.001f;

    [Range(0.01f, 1.0f)]
    public float persistence = 0.2f;

    [Range(-10.0f, 10.0f)]
    public float sharpness = 0.0f;

    [Range(0.001f, 10.0f)]
    public float slopeErosion = 0.001f;

    [Range(1, 20)]
    public int numOctaves = 5;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength = 0.01f;

    public override Task ExecuteStep(CpuPipelineContext pipelineContext)
    {
        return Task.Run(() =>
        {
            int2 textureDimensions = new int2(pipelineContext.intermediateHeightmap.width, pipelineContext.intermediateHeightmap.height);
            float2 extraDisplacement = (float2)textureDimensions * pipelineContext.GetRandomFloats();
            var nativeHeightmapArray = pipelineContext.intermediateHeightmap.GetRawTextureData<float>();

            for (int x = 0; x < textureDimensions.x; ++x)
            {
                for (int y = 0; y < textureDimensions.y; ++y)
                {
                    int2 actualTextureIdx = new int2(x, y);
                    float2 floatIdx = extraDisplacement + (float2)actualTextureIdx;

                    float2 slopeErosionDerivativeSum = (float2)math.clamp(new float2(noise.snoise(floatIdx)), -0.15, 0.15).xx;
                    float2 perturbDerivativeSum = new(0.0);
                    float noiseResult = 0.0f;

                    float octaveAmplitude = 1.0f;
                    float octaveFrequency = noiseFrequency;
                    float2 samplePosition = new(0.0f);

                    for (int o = 0; o < numOctaves; ++o)
                    {
                        samplePosition = (floatIdx * octaveFrequency) + perturbDerivativeSum;

                        float3 startNoiseValue = noise.psrdnoise(samplePosition, 1.0f, 0.0f);
                        float featureNoise = startNoiseValue.x;
                        float2 featureDerivative = startNoiseValue.yz;

                        float ridgedNoise = 1.0f - math.abs(featureNoise);
                        float billowNoise = ridgedNoise * ridgedNoise;
                        featureNoise = (float)math.lerp(featureNoise, billowNoise, math.max(0.0, sharpness));
                        featureNoise = (float)math.lerp(featureNoise, ridgedNoise, math.abs(math.min(0.0, sharpness)));

                        slopeErosionDerivativeSum += featureDerivative * slopeErosion;
                        perturbDerivativeSum += featureDerivative * perturbationStrength * octaveAmplitude;
                        float derivativeNorm = math.dot(slopeErosionDerivativeSum, slopeErosionDerivativeSum);
                        noiseResult += octaveAmplitude * featureNoise * (1.0f / (1.0f + derivativeNorm));

                        octaveAmplitude *= persistence;
                        octaveFrequency *= 2.0f;
                    }

                    nativeHeightmapArray[actualTextureIdx.x * textureDimensions.y + actualTextureIdx.y] = noiseResult;
                }
            }
        });
    }

    public override InputExpectations GetStepExpectations() => InputExpectations.None;

    public override void UpdateHeightmapState(ref HeightmapProperties previousState)
    {
        previousState &= ~HeightmapProperties.Normalized;
    }
}
