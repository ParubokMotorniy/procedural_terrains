using UnityEngine;
using UnityEngine.Assertions;
using System.Collections.Generic;
using System;

namespace GenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class TerrainGenerator : MonoBehaviour
    {
        [Range(1, 4)]
        public int terrainSize = 1;

        [Range(0.001f, 32.0f)]
        public float terrainScale = 1.0f;

        public List<MonoPipelineStep> pipelineSteps;

        private RenderTexture intermediateHeightmap;

        //built-ins
        private readonly HeightmapNormalizer normalizer = new HeightmapNormalizer();
        private readonly FormatFinalizer finalizer = new FormatFinalizer();


        [ContextMenu("Regenerate terrain")]
        void RegenerateTerrain()
        {
            int textureSize = terrainSize * 1024;

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

            PipelineContext currentContext = new PipelineContext(intermediateHeightmap);

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

            //executes the complete pipeline
            foreach (PipelineStep step in augmentedPipeline)
            {
                step.ExecuteStep(currentContext);
            }
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
