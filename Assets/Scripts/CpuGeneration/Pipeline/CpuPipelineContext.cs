using System;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using Unity.Mathematics;
using Random = System.Random;

namespace CpuGenerationPipeline
{
    public class CpuPipelineContext
    {
        public struct CpuIntermediateHeightmap
        {
            public NativeArray<float> nativeHeightmapArray;
            public uint2 heightmapDimensions;

            public CpuIntermediateHeightmap(NativeArray<float> nativeArray, uint2 uint2) : this()
            {
                this.nativeHeightmapArray = nativeArray;
                this.heightmapDimensions = uint2;
            }
        }

        public CpuIntermediateHeightmap GetCpuIntemediateHeightmap()
        {
            return new CpuIntermediateHeightmap(intermediateHeightmap.GetRawTextureData<float>(), new uint2((uint)GetHeightmapSize(), (uint)GetHeightmapSize()));
        }

        public Texture2D intermediateHeightmap
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

        public CpuPipelineContext(Texture2D passHeightmap, RenderTexture finalHeightmap, int seed)
        {
            intermediateHeightmap = passHeightmap;
            this.finalHeightmap = finalHeightmap;
            randomGenerator = new System.Random(seed);
        }

        public int GetHeightmapSize() { return intermediateHeightmap.width; }

        public float GetRandomFloat()
        {
            return (float)randomGenerator.NextDouble();
        }

        public float GetRandomInt()
        {
            return randomGenerator.Next();
        }

        public float2 GetRandomFloats()
        {
            return new float2((float)randomGenerator.NextDouble(), (float)randomGenerator.NextDouble());
        }
        public int2 GetRandomInts()
        {
            return new int2(randomGenerator.Next(), randomGenerator.Next());
        }
    }
}