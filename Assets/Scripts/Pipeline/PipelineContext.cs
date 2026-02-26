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

        public int preferredGlobalGroupSize
        {
            get;
            protected set;
        }

        protected CommandBuffer terrainPipelineCMD;
        protected Random randomGenerator;
        protected static float[] randomFloatsArray = new float[2];
        protected static int[] randomIntsArray = new int[2];

        public PipelineContext(RenderTexture passHeightmap, int seed, int preferredGlobalGroupSize)
        {
            intermediateHeightmap = passHeightmap;
            terrainPipelineCMD = new CommandBuffer();
            randomGenerator = new System.Random(seed);
            this.preferredGlobalGroupSize = preferredGlobalGroupSize;

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
        public void SetRandomInts(ComputeShader shader, int uniformId)
        {
            randomIntsArray[0] = randomGenerator.Next();
            randomIntsArray[1] = randomGenerator.Next();
            terrainPipelineCMD.SetComputeIntParams(shader, uniformId, randomIntsArray);
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

        public void SetKeyword(ComputeShader targetShader, ref LocalKeyword keywordToSet, bool value)
        {
            terrainPipelineCMD.SetKeyword(targetShader, keywordToSet, value);
        }

        public void SetGlobalKeyword(ref GlobalKeyword keywordToSet, bool value)
        {
            terrainPipelineCMD.SetKeyword(keywordToSet, value);
        }
    }
}