using UnityEngine;
using UnityEngine.Assertions;
using System.Collections.Generic;
using System;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System.IO;
using System.Text;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using UnityEditor.Experimental.GraphView;
using System.Threading.Tasks;

namespace CpuGenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class CpuTerrainGenerator : TunableGenerator
    {
        [Range(5, 12)]
        public int terrainSize = 5;

        [Range(0.001f, 32.0f)]
        public float terrainScale = 1.0f;

        [SerializeField]
        public int generatorSeed = 1;

        [Range(5, 100)]
        public int numSamples = 5;

        [SerializeField]
        public bool randomizeParameters = false;

#if UNITY_EDITOR
        [SerializeField]
        bool dumpTextures = false;

        [SerializeField]
        bool useTestTexture = false;

        public Texture2D testTexture;
#endif

        public List<UltimatePipelineStep> pipelineSteps;

        private Texture2D intermediateHeightmap;
        private RenderTexture finalHeightmap;

        //built-ins
        private readonly CpuHeightmapNormalizer normalizer = new CpuHeightmapNormalizer();
        private readonly CpuFormatFinalizer finalizer = new CpuFormatFinalizer();

        //TODO: consider using intermediate float array to allow async execution of the pipeline for non-profiling purposes

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
                if (pipelineToRun is null || pipelineContext is null)
                    return;
                foreach (CpuPipelineStep step in pipelineToRun)
                {
                    await step.ExecuteStepCpu(pipelineContext);
                }
            }

            public async System.Threading.Tasks.Task RunPipelineRandomized(int seed)
            {
                if (pipelineToRun is null || pipelineContext is null)
                    return;

                System.Random parameterRandomizer = new System.Random(seed);
                foreach (CpuPipelineStep step in pipelineToRun)
                {
                    step.RandomizeParameters(parameterRandomizer);
                    await step.ExecuteStepCpu(pipelineContext);
                }
            }
        }


        private ExecutablePipeline buildPipeline(int pipelineSeed)
        {
            int textureSize = (int)math.pow(2, terrainSize);

            // intermediate heightmap creation
            if (intermediateHeightmap == null ||
                intermediateHeightmap.width != textureSize ||
                intermediateHeightmap.height != textureSize)
            {
                intermediateHeightmap = new Texture2D(textureSize, textureSize, TextureFormat.RFloat, false, true);
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
                if (RuntimeAssert.IsTrue(finalHeightmap.IsCreated(), "Failed to create the heightmap! Try restarting the app."))
                    return new ExecutablePipeline(null, null);
            }

            CpuPipelineContext currentContext = new CpuPipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed);

            Renderer renderer = GetComponent<Renderer>();
            renderer.sharedMaterial.SetTexture("_HeightMap", finalHeightmap);
            renderer.sharedMaterial.SetFloat("_HeightScale", terrainScale);

            List<CpuPipelineStep> augmentedPipeline = new List<CpuPipelineStep>();

            CpuHeightmapProperties runningProperties = CpuHeightmapProperties.Created;

            foreach (CpuPipelineStep inputStep in pipelineSteps)
            {
                if ((inputStep.GetStepExpectationsCpu() & CpuInputExpectations.HeightMapNormalized) != 0 &&
                    (runningProperties & CpuHeightmapProperties.Normalized) == 0)
                {
                    augmentedPipeline.Add(normalizer);
                    normalizer.UpdateHeightmapStateCpu(ref runningProperties);
                }

                augmentedPipeline.Add(inputStep);
                inputStep.UpdateHeightmapStateCpu(ref runningProperties);
            }

            augmentedPipeline.Add(normalizer);
            augmentedPipeline.Add(finalizer);

            return new ExecutablePipeline(augmentedPipeline, currentContext);
        }

        [ContextMenu("Regenerate terrain")]
        async void RegenerateTerrain()
        {
            var currentPipeline = buildPipeline(generatorSeed);

            if (randomizeParameters)
                await currentPipeline.RunPipelineRandomized(generatorSeed);
            else
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

        public override void RenderParametersTuningGUI()
        {
            GUILayout.Label("Terrain");

            GUILayout.Label($"Terrain Size: {terrainSize}");
            terrainSize = Mathf.RoundToInt(
                GUILayout.HorizontalSlider(terrainSize, 5f, 12f)
            );

            GUILayout.Label($"Terrain Scale: {terrainScale:F3}");
            terrainScale = GUILayout.HorizontalSlider(terrainScale, 0.001f, 32.0f);

            GUILayout.Space(5);
            GUILayout.Label("Sampling");

            GUILayout.Label($"Num Samples: {numSamples}");
            numSamples = Mathf.RoundToInt(
                GUILayout.HorizontalSlider(numSamples, 5f, 100f)
            );

            GUILayout.Space(5);
            GUILayout.Label("Seed");

            GUILayout.BeginHorizontal();
            GUILayout.Label("Generator Seed", GUILayout.Width(120));

            string seedStr = generatorSeed.ToString();
            string newSeedStr = GUILayout.TextField(seedStr, GUILayout.Width(80));

            if (int.TryParse(newSeedStr, out int parsedSeed))
                generatorSeed = parsedSeed;

            GUILayout.EndHorizontal();

            GUILayout.Space(5);
            GUILayout.Label("Generation parameters randomization");
            GUILayout.BeginHorizontal();
            GUILayout.Label("Randomize parameters");
            randomizeParameters = GUILayout.Toggle(randomizeParameters, "");
            GUILayout.EndHorizontal();

        }

        public override string GUIStepTitle() => "CPU pipeline parameters";

        public override void drawExtraCustomUI()
        {
            GUILayout.Label("Procedures");
            if (GUILayout.Button("Generate terrain"))
            {
                if (pipeline.Count == 0)
                    return;
                pipelineSteps = CommonDefines.buildPipelineFromEnum(pipeline);
                RegenerateTerrain();
            }
            if (GUILayout.Button("Collect statistics"))
            {
                if (pipeline.Count == 0)
                    return;
                pipelineSteps = CommonDefines.buildPipelineFromEnum(pipeline);
                CollectStatistics();
            }
            if (GUILayout.Button("Free resources"))
            {
                UltimatePipelineStep.FreeAllResourcesInScene();
            }
        }

        public override string getPipelineName() => "CPU pipeline constructor";

        [ContextMenu("Collect statistics")]
        async void CollectStatistics()
        {
            //minimize draw calls
            int oldVSync = QualitySettings.vSyncCount;
            int oldFrameRate = Application.targetFrameRate;
            var currentCameras = Camera.allCameras;
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
                foreach (var cam in currentCameras)
                {
                    cam.enabled = false;
                }
                foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
                { light.enabled = false; }
            }

            Stopwatch cpuProfilingStopwatch = new Stopwatch();
            System.Random randomSeedGenerator = new(generatorSeed);

            (long cpuMs, long cpuTicks)[] runtimeResults = new (long cpuMs, long cpuTicks)[numSamples];
            Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples_cpu"));

            //runs a number of samples, syncing each time to avoid obtaining corrupted heightmaps
            for (int s = 0; s < numSamples; ++s)
            {
                var currentPipeline = buildPipeline(randomSeedGenerator.Next());
                cpuProfilingStopwatch.Restart();
                if (randomizeParameters)
                    await currentPipeline.RunPipelineRandomized(randomSeedGenerator.Next());
                else
                    await currentPipeline.RunPipeline();
                runtimeResults[s] = (cpuProfilingStopwatch.ElapsedMilliseconds, cpuProfilingStopwatch.ElapsedTicks);
                RenderTextureDumper.SaveRFloatToExr(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples_cpu/cpu_terrain_" + s + ".exr"));
            }
            {
                string path = Path.Combine(Application.persistentDataPath, "cpu_performance_evaluation.txt");
                var sb = new StringBuilder();
                sb.AppendLine("CpuTime (ms)\tCpuTime (ticks)");
                foreach (var (cpuMs, cpuTicks) in runtimeResults)
                {
                    sb.Append(cpuMs);
                    sb.Append('\t');
                    sb.AppendLine(cpuTicks.ToString());
                }
                File.WriteAllText(path, sb.ToString());
            }

            {
                QualitySettings.vSyncCount = oldVSync;
                Application.targetFrameRate = oldFrameRate;
                foreach (var cam in currentCameras)
                {
                    cam.enabled = true;
                }
                foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
                { light.enabled = true; }
            }

            UltimatePipelineStep.FreeAllResourcesInScene();
            Debug.Log("Collection done");
        }
    }
}