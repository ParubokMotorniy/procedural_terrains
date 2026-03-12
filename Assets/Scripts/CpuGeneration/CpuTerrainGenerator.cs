using UnityEngine;
using UnityEngine.Assertions;
using System.Collections.Generic;
using System;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System.IO;
using System.Text;

namespace CpuGenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class CpuTerrainGenerator : MonoBehaviour
    {
        [Range(5, 12)]
        public int terrainSize = 5;

        [Range(0.001f, 32.0f)]
        public float terrainScale = 1.0f;

        [SerializeField]
        public int generatorSeed = 1;

        [Range(5, 100)]
        public int numSamples = 5;

#if UNITY_EDITOR
        [SerializeField]
        bool dumpTextures = false;

        [SerializeField]
        bool useTestTexture = false;

        public Texture2D testTexture;
#endif

        public List<CpuMonoPipelineStep> pipelineSteps;

        private float[,] intermediateHeightmap;

        private RenderTexture finalHeightmap;

        //built-ins
        private readonly CpuHeightmapNormalizer normalizer = new CpuHeightmapNormalizer();
        private readonly CpuFormatFinalizer finalizer = new CpuFormatFinalizer();

        // int synchronizedIterationsLeft = 0;
        // (long cpuSideMs, long cpuSideTicks)[] synchronizedResults;
        // Action postCollectionAction;
        // bool previousReadPending = false;

        private struct ExecutablePipeline
        {
            public List<CpuPipelineStep> pipelineToRun;
            public CpuPipelineContext pipelineContext;

            public ExecutablePipeline(List<CpuPipelineStep> pipeline, CpuPipelineContext context)
            {
                pipelineToRun = pipeline;
                pipelineContext = context;
            }

            public async System.Threading.Tasks.Task RunPipeline()
            {
                foreach (CpuPipelineStep step in pipelineToRun)
                {
                    await step.ExecuteStep(pipelineContext);
                }
            }
        }


        private ExecutablePipeline buildPipeline(int pipelineSeed)
        {
            int textureSize = (int)math.pow(2, terrainSize);

            // intermediate heightmap creation
            if (intermediateHeightmap == null ||
                intermediateHeightmap.GetLength(0) != textureSize ||
                intermediateHeightmap.GetLength(1) != textureSize)
            {
                intermediateHeightmap = new float[textureSize, textureSize];
            }

            // final heightmap creation
            if (finalHeightmap == null ||
                finalHeightmap.width != textureSize ||
                finalHeightmap.height != textureSize)
            {
                if (finalHeightmap != null)
                {
                    finalHeightmap.Release();
                }

                finalHeightmap = new RenderTexture(textureSize, textureSize, 0)
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

            CpuPipelineContext currentContext = new CpuPipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed);

            Renderer renderer = GetComponent<Renderer>();
            renderer.sharedMaterial.SetTexture("_HeightMap", finalHeightmap);
            renderer.sharedMaterial.SetFloat("_HeightScale", terrainScale);

            List<CpuPipelineStep> augmentedPipeline = new List<CpuPipelineStep>();

            HeightmapProperties runningProperties = HeightmapProperties.Created;

            foreach (CpuPipelineStep inputStep in pipelineSteps)
            {
                if ((inputStep.GetStepExpectations() & InputExpectations.HeightMapNormalized) != 0 &&
                    (runningProperties & HeightmapProperties.Normalized) == 0)
                {
                    augmentedPipeline.Add(normalizer);
                    normalizer.UpdateHeightmapState(ref runningProperties);
                }

                augmentedPipeline.Add(inputStep);
                inputStep.UpdateHeightmapState(ref runningProperties);
            }

            augmentedPipeline.Add(normalizer);
            augmentedPipeline.Add(finalizer);

            return new ExecutablePipeline(augmentedPipeline, currentContext);
        }

        [ContextMenu("Regenerate terrain")]
        async void RegenerateTerrain()
        {
            var currentPipeline = buildPipeline(generatorSeed);

            await currentPipeline.RunPipeline();

#if UNITY_EDITOR
            if (dumpTextures)
            {
                RenderTextureDumper.SaveRFloatToExr(finalHeightmap, "heightmap_final.exr");
            }
#endif

            Debug.Log("Terrain has been regenerated!");
        }

        void OnValidate()
        {
            Renderer renderer = GetComponent<Renderer>();
            if (renderer != null && renderer.sharedMaterial != null)
            {
                renderer.sharedMaterial.SetFloat("_HeightScale", terrainScale);
            }
        }

        void OnGUI()
        {
            if (GUI.Button(new Rect(25, 25, 200, 25), "Generate terrain"))
            {
                RegenerateTerrain();
            }
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


        // [ContextMenu("Collect statistics")]
        // async void CollectStatistics()
        // {
        //     //minimize draw calls
        //     int oldVSync = QualitySettings.vSyncCount;
        //     int oldFrameRate = Application.targetFrameRate;
        //     var currentCameras = Camera.allCameras;
        //     {
        //         QualitySettings.vSyncCount = 0;
        //         Application.targetFrameRate = -1;
        //         foreach (var cam in currentCameras)
        //         {
        //             cam.enabled = false;
        //         }
        //         foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
        //         { light.enabled = false; }
        //     }

        //     (long cpuSide, long gpuSide)[] result = new (long cpuSide, long gpuSide)[numSamples];
        //     Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples"));

        //     //runs a number of samples, syncing each time to avoid obtaining corrupted heightmaps
        //     for (int s = 0; s < numSamples; ++s)
        //     {
        //         var newContext = buildPipeline(true, numSamples + generatorSeed + s);
        //         //executes the complete pipeline
        //         foreach (CpuPipelineStep step in augmentedPipeline)
        //         {
        //             step.ExecuteStep(currentContext);
        //         }
        //         result[s] = (newContext.gpuMilliseconds, newContext.gpuFrameTime);
        //         RenderTextureDumper.SaveRFloatToExr(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples/cpu_terrain_" + s + ".exr"));
        //     }
        //     {
        //         string path = Path.Combine(Application.persistentDataPath, "performance_evaluation.txt");
        //         var sb = new StringBuilder();
        //         sb.AppendLine("CpuTime (ms)\tGpuTime (ns)");
        //         foreach (var (cpuSide, gpuSide) in result)
        //         {
        //             sb.Append(cpuSide);
        //             sb.Append('\t');
        //             sb.AppendLine(gpuSide.ToString());
        //         }
        //         File.WriteAllText(path, sb.ToString());
        //     }

        //     {
        //         QualitySettings.vSyncCount = oldVSync;
        //         Application.targetFrameRate = oldFrameRate;
        //         foreach (var cam in currentCameras)
        //         {
        //             cam.enabled = true;
        //         }
        //         foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
        //         { light.enabled = true; }
        //     }

        //     Debug.Log("Collection done");
        // }


        // [ContextMenu("Collect synchronized statistics")]
        // async void CollectSynchronizedStatistics()
        // {
        //     if (synchronizedIterationsLeft > 0)
        //         return;

        //     //minimize draw calls
        //     int oldVSync = QualitySettings.vSyncCount;
        //     int oldFrameRate = Application.targetFrameRate;
        //     var currentCameras = Camera.allCameras;
        //     {
        //         QualitySettings.vSyncCount = 0;
        //         Application.targetFrameRate = -1;
        //         foreach (var cam in currentCameras)
        //         {
        //             cam.enabled = false;
        //         }
        //         foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
        //         { light.enabled = false; }
        //     }

        //     synchronizedResults = new (long cpuSideMs, long cpuSideTicks, long gpuSideNs)[numSamples];
        //     Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples"));

        //     postCollectionAction = () =>
        //     {
        //         QualitySettings.vSyncCount = oldVSync;
        //         Application.targetFrameRate = oldFrameRate;
        //         foreach (var cam in currentCameras)
        //         {
        //             cam.enabled = true;
        //         }
        //         foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
        //         { light.enabled = true; }
        //     };

        //     synchronizedIterationsLeft = numSamples;
        //     previousReadPending = false;
        // }

        // [ExecuteAlways]
        // async void Update()
        // {
        //     if (synchronizedIterationsLeft > 0 && !previousReadPending)
        //     {
        //         synchronizedIterationsLeft -= 1;

        //         int myIteration = synchronizedIterationsLeft;

        //         var newContext = (CpuProfilingPipelineContext)buildPipeline(true, numSamples + generatorSeed + myIteration);
        //         previousReadPending = true;
        //         await newContext.ExecuteBuffer();

        //         synchronizedResults[myIteration] = (newContext.gpuMilliseconds, newContext.gpuTicks);

        //         RenderTextureDumper.SaveRFloatToExr(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples/cpu_terrain_" + myIteration + ".exr"), false);
        //         previousReadPending = false;

        //         if (myIteration == 0)
        //         {
        //             string path = Path.Combine(Application.persistentDataPath, "performance_evaluation_cpu.txt");
        //             var sb = new StringBuilder();
        //             sb.AppendLine("CpuTime (ms)\tCpuTime (ticks)");
        //             foreach (var (cpuSide, cpuTicks) in synchronizedResults)
        //             {
        //                 sb.Append(cpuSide);
        //                 sb.Append('\t');
        //                 sb.AppendLine(cpuTicks.ToString());
        //             }
        //             File.WriteAllText(path, sb.ToString());

        //             postCollectionAction();

        //             Debug.Log("Collection done!");
        //         }

        //     }
        // }
    }
}