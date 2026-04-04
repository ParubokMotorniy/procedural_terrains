using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using Unity.VisualScripting;
using System.Threading.Tasks;
using System;

namespace CpuGenerationPipeline
{
    [ExecuteAlways]
    public class CpuHeightmapNormalizer : CpuPipelineStep
    {
        public Task ExecuteStepCpu(CpuPipelineContext pipelineContext)
        {
            // return Task.Run(() =>
            // {
            // });

            int2 textureDimensions = new int2(pipelineContext.intermediateHeightmap.width, pipelineContext.intermediateHeightmap.height);
            var nativeHeightmapArray = pipelineContext.intermediateHeightmap.GetRawTextureData<float>();

            float minHeight = Single.MaxValue;
            float maxHeight = Single.MinValue;
            for (int x = 0; x < textureDimensions.x; ++x)
            {
                for (int y = 0; y < textureDimensions.y; ++y)
                {
                    float oldHeight = nativeHeightmapArray[x * textureDimensions.y + y];
                    minHeight = math.min(oldHeight, minHeight);
                    maxHeight = math.max(oldHeight, maxHeight);
                }
            }

            float span = maxHeight - minHeight;

            for (int x = 0; x < textureDimensions.x; ++x)
            {
                for (int y = 0; y < textureDimensions.y; ++y)
                {
                    float oldHeight = nativeHeightmapArray[x * textureDimensions.y + y];
                    nativeHeightmapArray[x * textureDimensions.y + y] = (oldHeight - minHeight) / span;
                }
            }
            return Task.CompletedTask;
        }

        public CpuInputExpectations GetStepExpectationsCpu()
        {
            return CpuInputExpectations.None;
        }

        public void RandomizeParameters(System.Random random)
        {
        }

        public void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
        {
            previousState |= CpuHeightmapProperties.Normalized;
        }
    }
}