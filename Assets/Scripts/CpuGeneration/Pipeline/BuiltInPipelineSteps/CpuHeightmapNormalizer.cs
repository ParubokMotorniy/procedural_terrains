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
            return Task.Run(() =>
            {
                int2 textureDimensions = new int2(pipelineContext.intermediateHeightmap.width, pipelineContext.intermediateHeightmap.height);
                var nativeHeightmapArray = pipelineContext.intermediateHeightmap.GetRawTextureData<float>();

                float minHeight = Single.MaxValue;
                float maxHeight = Single.MinValue;
                for (int x = 0; x < textureDimensions.x; ++x)
                {
                    for (int y = 0; y < textureDimensions.y; ++y)
                    {
                        minHeight = math.min(nativeHeightmapArray[x * textureDimensions.y + y], minHeight);
                        maxHeight = math.max(nativeHeightmapArray[x * textureDimensions.y + y], maxHeight);
                    }
                }

                float span = math.abs(minHeight) + math.abs(maxHeight);

                for (int x = 0; x < textureDimensions.x; ++x)
                {
                    for (int y = 0; y < textureDimensions.y; ++y)
                    {
                        float oldHeight = nativeHeightmapArray[x * textureDimensions.y + y];
                        nativeHeightmapArray[x * textureDimensions.y + y] = (oldHeight + minHeight) / span;
                    }
                }
            });
        }

        public CpuInputExpectations GetStepExpectationsCpu()
        {
            return CpuInputExpectations.None;
        }

        public void UpdateHeightmapStateCpu(ref CpuHeightmapProperties previousState)
        {
            previousState |= CpuHeightmapProperties.Normalized;
        }
    }
}