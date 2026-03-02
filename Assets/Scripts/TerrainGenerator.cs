using UnityEngine;
using UnityEngine.Assertions;
using System.Collections.Generic;
using System;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine.Rendering;
using UnityEditor;
using System.IO;
using System.Text;

namespace GenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class TerrainGenerator : MonoBehaviour
    {
        [Range(5, 12)]
        public int terrainSize = 5;

        [Range(3, 6)]
        public int preferredGroupSizePower = 5;

        [Range(0.001f, 32.0f)]
        public float terrainScale = 1.0f;

        [SerializeField]
        bool enableProfiling = false;

        [SerializeField]
        bool enableMetricEvaluation = false;

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

        public List<MonoPipelineStep> pipelineSteps;

        private RenderTexture intermediateHeightmap;

        private RenderTexture finalHeightmap;
        //built-ins
        private readonly HeightmapNormalizer normalizer = new HeightmapNormalizer();
        private readonly FormatFinalizer finalizer = new FormatFinalizer();

        //TODO: getting rid of two-step grid iteration in erosion algos improves appearance but introduces non-conservitivity and races
        //TODO: when adding basic combination UI, I may want to devise some resource clearing technique.

        //benchmarking design: 
        //I can keep the same context. Just teach it to read profiler recordings. And stall CPU to read back a new value each frame. That's it. 
        //The question of dumping the resulting texture is still unclear, but it can be put off for now.
        //Buffers will have to be regenrated each frame so as to insert new seeds.
        //CPU blocking can be kept async. The important part is that measurements are split by synchronization
        //generator can read stuff async as well.

        private PipelineContext buildPipeline(bool enablePipelineProfiling, int pipelineSeed)
        {
            int textureSize = (int)math.pow(2, terrainSize);

            { //intermediate heightmap creation

                if (!intermediateHeightmap || intermediateHeightmap.width != textureSize && intermediateHeightmap.height != textureSize)
                {

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
            }

            { //final heightmap creation

                if (!finalHeightmap || finalHeightmap.width != textureSize && finalHeightmap.height != textureSize)
                {

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
            }

            int preferredGroupSize = (int)math.pow(2, preferredGroupSizePower);
            PipelineContext currentContext = enablePipelineProfiling ? new ProfilingPipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed, preferredGroupSize) : new PipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed, preferredGroupSize);

            var groupSizeKeyword = GlobalKeyword.Create("GLOBAL_GROUP_" + preferredGroupSize);
            currentContext.SetGlobalKeyword(ref groupSizeKeyword, true);

            GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", finalHeightmap);
            GetComponent<Renderer>().sharedMaterial.SetFloat("_HeightScale", terrainScale);

            List<PipelineStep> augmentedPipeline = new List<PipelineStep>();

            //inserts extra built-in steps if requested by a step
            HeightmapProperties runningProperties = HeightmapProperties.Created;
            foreach (PipelineStep inputStep in pipelineSteps)
            {
                if ((inputStep.GetStepExpectations() & InputExpectations.HeightMapNormalized) != 0 && (runningProperties & HeightmapProperties.Normalized) == 0)
                {
                    augmentedPipeline.Add(normalizer);
                    normalizer.UpdateHeightmapState(ref runningProperties);
                }
                augmentedPipeline.Add(inputStep);
                inputStep.UpdateHeightmapState(ref runningProperties);
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

            return currentContext;
        }

        [ContextMenu("Regenerate terrain")]
        async void RegenerateTerrain()
        {
            PipelineContext currentContext = buildPipeline(enableProfiling, generatorSeed);
            await currentContext.ExecuteBuffer();

            if (enableMetricEvaluation)
            {
                MetricEvaluator.ComputeMetrics(finalHeightmap);
            }

#if UNITY_EDITOR
            if (dumpTextures)
            {
                RenderTextureDumper.SaveRFloatToExr(intermediateHeightmap, "heightmap_intermediate.exr");
                RenderTextureDumper.SaveRFloatToExr(finalHeightmap, "heightmap_final.exr");
            }
#endif

            Debug.Log("Terrain has been regenerated!");
        }

        [ContextMenu("Collect statistics")]
        async void CollectStatistics()
        {
            //minimize draw calls
            int oldVSync = QualitySettings.vSyncCount;
            int oldFrameRate = Application.targetFrameRate;
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
                foreach (var cam in Camera.allCameras)
                {
                    cam.enabled = false;
                }
            }

            (long cpuSide, long gpuSide)[] result = new (long cpuSide, long gpuSide)[numSamples];
            Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples"));

            //runs a number of samples, syncing each time to avoid obtaining corrupted heightmaps
            for (int s = 0; s < numSamples; ++s)
            {
                var newContext = (ProfilingPipelineContext)buildPipeline(true, numSamples + generatorSeed + s);
                await newContext.ExecuteBuffer();
                result[s] = (newContext.gpuMilliseconds, newContext.gpuFrameTime);
                RenderTextureDumper.SaveRFloatToExr(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples/terrain_" + s + ".exr"), false);
            }

            {
                string path = Path.Combine(Application.persistentDataPath, "performance_evaluation.txt");
                var sb = new StringBuilder();
                sb.AppendLine("CpuTime (ms)\tGpuTime (ns)");
                foreach (var (cpuSide, gpuSide) in result)
                {
                    sb.Append(cpuSide);
                    sb.Append('\t');
                    sb.AppendLine(gpuSide.ToString());
                }
                File.WriteAllText(path, sb.ToString());
            }

            {
                QualitySettings.vSyncCount = oldVSync;
                Application.targetFrameRate = oldFrameRate;
                foreach (var cam in Camera.allCameras)
                {
                    cam.enabled = true;
                }
            }
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
