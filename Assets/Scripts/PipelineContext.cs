using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace GenerationPipeline
{
    // TODO: maybe, implement chained static constructor
    public class PipelineContext
    {
        public RenderTexture intermediateHeightmap
        {
            get;
            protected set;
        }
        public RenderTexture finalHeightmap
        {
            get;
            protected set;
        }

        protected CommandBuffer terrainPipelineCMD;

        public PipelineContext(RenderTexture passHeightmap)
        {
            intermediateHeightmap = passHeightmap;
            terrainPipelineCMD = new CommandBuffer();

            finalHeightmap = new RenderTexture(intermediateHeightmap.width, intermediateHeightmap.height, 0)
            {
                graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16_UNorm,
                useMipMap = false,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            finalHeightmap.Create();
            Assert.IsTrue(finalHeightmap.IsCreated());
        }
        public int GetHeightmapSize() { return intermediateHeightmap.height; }

        public void AppendDispatchToCommandBuffer(ComputeShader shader, int kernelIdx, Vector3 dispatchDimensions)
        {
            terrainPipelineCMD.DispatchCompute(shader, kernelIdx, (int)dispatchDimensions.x, (int)dispatchDimensions.y, (int)dispatchDimensions.z);
        }

        public void AppendTextureCopyToCommandBuffer(Texture src, Texture dest)
        {
            terrainPipelineCMD.Blit(src, dest);
        }

        public virtual async Task ExecuteBuffer() //TODO: this ideally shouldn't be public and only available for invocation from the generator
        {
            Graphics.ExecuteCommandBuffer(terrainPipelineCMD);
        }
    }
}