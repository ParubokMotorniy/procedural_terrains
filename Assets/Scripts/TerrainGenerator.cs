using UnityEngine;
using UnityEngine.Assertions;
using System.Collections.Generic;
using System;
using Unity.Mathematics;

namespace GenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class TerrainGenerator : MonoBehaviour
    {
        [Range(5, 12)]
        public int terrainSize = 5;

        [Range(0.001f, 32.0f)]
        public float terrainScale = 1.0f;

        [SerializeField]
        bool enableProfiling = false;

        [SerializeField]
        int generatorSeed = 1;

#if UNITY_EDITOR
        [SerializeField]
        bool dumpTextures = false;

        [SerializeField]
        bool useTestTexture = false;

        public Texture2D testTexture;
#endif

        public List<MonoPipelineStep> pipelineSteps;

        private RenderTexture intermediateHeightmap;

        //built-ins
        private readonly HeightmapNormalizer normalizer = new HeightmapNormalizer();
        private readonly FormatFinalizer finalizer = new FormatFinalizer();


        //TODO: instead of manually selecting the number of threads to dispatch, I may want to compile my compute shaders in multiple variants with unity-keywords system. However, that can be postponed I believe owing to the complexity of writing code with tons of defines
        [ContextMenu("Regenerate terrain")]
        async void RegenerateTerrain()
        {
            int textureSize = (int)math.pow(2, terrainSize);

            { //intermediate heightmap creation
                if (intermediateHeightmap && intermediateHeightmap.IsCreated())
                { intermediateHeightmap.Release(); }

                intermediateHeightmap = new RenderTexture(textureSize, textureSize, 0)
                {
                    graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
                    useMipMap = false,
                    enableRandomWrite = true,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp
                };

                intermediateHeightmap.Create();

                Assert.IsTrue(intermediateHeightmap.IsCreated());
            }

            PipelineContext currentContext = enableProfiling ? new ProfilingPipelineContext(intermediateHeightmap, generatorSeed) : new PipelineContext(intermediateHeightmap, generatorSeed);

            GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", currentContext.finalHeightmap);
            GetComponent<Renderer>().sharedMaterial.SetFloat("_HeightScale", terrainScale);

            List<PipelineStep> augmentedPipeline = new List<PipelineStep>();

            //inserts extra built-in steps if requested by a step
            foreach (PipelineStep inputStep in pipelineSteps)
            {
                if ((inputStep.GetStepExpectations() & InputExpectations.HeightMapNormalized) != 0)
                {
                    augmentedPipeline.Add(normalizer);
                }
                augmentedPipeline.Add(inputStep);
            }
            augmentedPipeline.Add(normalizer);
            augmentedPipeline.Add(finalizer);

#if UNITY_EDITOR
            if (useTestTexture)
            {
                currentContext.AppendTextureCopyToCommandBuffer(testTexture, intermediateHeightmap);
            }
#endif
            //executes the complete pipeline
            foreach (PipelineStep step in augmentedPipeline)
            {
                step.ExecuteStep(currentContext);
            }

            await currentContext.ExecuteBuffer();

#if UNITY_EDITOR
            if (dumpTextures)
            {
                RenderTextureDumper.SaveRFloatToExr(intermediateHeightmap, "heightmap_intermediate.exr");
                RenderTextureDumper.SaveRFloatToExr(currentContext.finalHeightmap, "heightmap_final.exr");
            }
#endif
            Debug.Log("Terrain has been regenerated!");
        }

        void Start()
        {
            RegenerateTerrain();
        }

        void OnValidate()
        {
            GetComponent<Renderer>().sharedMaterial.SetFloat("_HeightScale", terrainScale);
        }

        void OnDrawGizmos()
        {
            MeshFilter meshFilter = GetComponent<MeshFilter>();
            if (meshFilter == null || meshFilter.sharedMesh == null)
                return;

            Mesh mesh = meshFilter.sharedMesh;

            Bounds localBounds = mesh.bounds;

            Vector3 size = new Vector3(
                localBounds.size.x,
                terrainScale,
                localBounds.size.z
            );

            Vector3 center = new Vector3(
                localBounds.center.x,
                terrainScale / 2.0f,
                localBounds.center.z
            );

            Gizmos.color = Color.green;

            Matrix4x4 oldMatrix = Gizmos.matrix;
            Gizmos.matrix = transform.localToWorldMatrix;

            Gizmos.DrawWireCube(center, size);

            Gizmos.matrix = oldMatrix;
        }
    }
}
