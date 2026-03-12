using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;
using Random = System.Random;

namespace CpuGenerationPipeline
{
    public class CpuPipelineContext
    {
        public float[,] intermediateHeightmap
        {
            get;
            protected set;
        }
        public RenderTexture finalHeightmap
        {
            get;
            protected set;
        }

        protected Random randomGenerator;

        public CpuPipelineContext(float[,] passHeightmap, RenderTexture finalHeightmap, int seed)
        {
            intermediateHeightmap = passHeightmap;
            this.finalHeightmap = finalHeightmap;
            randomGenerator = new System.Random(seed);
        }

        public int GetHeightmapSize() { return intermediateHeightmap.GetLength(0); }

        public float GetRandomFloat()
        {
            return (float)randomGenerator.NextDouble();
        }

        public float GetRandomInt()
        {
            return randomGenerator.Next();
        }

        public Unity.Mathematics.float2 GetRandomFloats()
        {
            return new Unity.Mathematics.float2((float)randomGenerator.NextDouble(), (float)randomGenerator.NextDouble());
        }
        public Unity.Mathematics.int2 GetRandomInts()
        {
            return new Unity.Mathematics.int2(randomGenerator.Next(), randomGenerator.Next());
        }
    }
}