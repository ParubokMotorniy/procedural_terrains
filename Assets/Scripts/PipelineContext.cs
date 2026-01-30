using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using Random = System.Random;

namespace GenerationPipeline
{
    // TODO: maybe, implement chained static constructor
    //TODO: add callbacks for the stages to use
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
        protected Random randomGenerator;
        protected static float[] randomFloatsArray = new float[2];

        public PipelineContext(RenderTexture passHeightmap)
        {
            intermediateHeightmap = passHeightmap;
            terrainPipelineCMD = new CommandBuffer();
            randomGenerator = new System.Random();

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

        public void SetUniformInt(ComputeShader shader, int uniformId, int value)
        {
            terrainPipelineCMD.SetComputeIntParam(shader, uniformId, value);
        }
        public void SetUniformInts(ComputeShader shader, int uniformId, int[] values)
        {
            terrainPipelineCMD.SetComputeIntParams(shader, uniformId, values);
        }

        public void SetUniformFloat(ComputeShader shader, int uniformId, float value)
        {
            terrainPipelineCMD.SetComputeFloatParam(shader, uniformId, value);
        }

        public void SetUniformFloats(ComputeShader shader, int uniformId, float[] values)
        {
            terrainPipelineCMD.SetComputeFloatParams(shader, uniformId, values);
        }
        public void SetRandomFloat(ComputeShader shader, int uniformId)
        {
            terrainPipelineCMD.SetComputeFloatParam(shader, uniformId, (float)randomGenerator.NextDouble());
        }

        public void SetRandomFloats(ComputeShader shader, int uniformId)
        {
            randomFloatsArray[0] = (float)randomGenerator.NextDouble();
            randomFloatsArray[1] = (float)randomGenerator.NextDouble();
            terrainPipelineCMD.SetComputeFloatParams(shader, uniformId, randomFloatsArray);
        }

        public void BindTexture(ComputeShader shader, int kernelIdx, int uniformId, Texture texToBind)
        {
            terrainPipelineCMD.SetComputeTextureParam(shader, kernelIdx, uniformId, texToBind);
        }

        public void BindComputeBuffer(ComputeShader shader, int kernelIdx, int uniformId, ComputeBuffer bufferToBind)
        {
            terrainPipelineCMD.SetComputeBufferParam(shader, kernelIdx, uniformId, bufferToBind);
        }

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