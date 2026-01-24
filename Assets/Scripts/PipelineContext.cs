using UnityEngine;
using UnityEngine.Assertions;

namespace GenerationPipeline
{
    public class PipelineContext
    {
        public RenderTexture intermediateHeightmap
        {
            get;
            private set;
        }
        public RenderTexture finalHeightmap
        {
            get;
            private set;
        }
        public PipelineContext(RenderTexture passHeightmap)
        {
            intermediateHeightmap = passHeightmap;
            
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
        // TODO: maybe, implement chained static constructor
    }
}