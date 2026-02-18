using UnityEngine;
using UnityEngine.Assertions;
using Unity.Mathematics;
using Unity.VisualScripting;

namespace GenerationPipeline
{
    [ExecuteAlways]
    public class HeightmapNormalizer : PipelineStep
    {
        private ComputeShader normalizationShader;

        public void ExecuteStep(PipelineContext pipelineContext)
        {
            if (normalizationShader is null)
            {
                normalizationShader = (ComputeShader)Resources.Load("ComputeShaders/TerrainNormalizer");
            }

            int textureSize = pipelineContext.GetHeightmapSize();
            Assert.IsTrue(textureSize >= 8); //normalization groups are at least 8 threads wide 
            int largestGroupSize = (int)math.pow(2, math.ceil(math.log2(math.clamp(textureSize, 8, 32))));
            int normalizationKernel = normalizationShader.FindKernel("Normalizer" + largestGroupSize);
            pipelineContext.BindTexture(normalizationShader, normalizationKernel, Shader.PropertyToID("Result"), pipelineContext.intermediateHeightmap);
            pipelineContext.SetUniformInt(normalizationShader, Shader.PropertyToID("texelsPerThread"), (int)math.ceil((float)textureSize / largestGroupSize));
            pipelineContext.SetUniformFloat(normalizationShader, Shader.PropertyToID("desiredMaxHeight"), 1.0f);

            {
                pipelineContext.AppendDispatchToCommandBuffer(normalizationShader, normalizationKernel, new Vector3(1, 1, 1));
            }
            Debug.LogWarning("Heightmap has been normalized. Max height now: " + 1.0);
        }

        public InputExpectations GetStepExpectations()
        {
            return InputExpectations.None;
        }
    }
}