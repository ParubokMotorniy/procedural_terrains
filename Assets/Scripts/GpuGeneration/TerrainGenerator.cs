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
using UnityEngine.Experimental.GlobalIllumination;

namespace GpuGenerationPipeline
{
    [RequireComponent(typeof(Renderer))]
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class TerrainGenerator : TunableGenerator
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

        [SerializeField]
        public bool randomizeParameters = false;

#if UNITY_EDITOR
        [SerializeField]
        bool dumpTextures = false;

        [SerializeField]
        bool useTestTexture = false;

        public Texture2D testTexture;
#endif

        [SerializeField]
        public List<UltimatePipelineStep> pipelineSteps;

        private RenderTexture intermediateHeightmap;

        private RenderTexture finalHeightmap;
        //built-ins
        private readonly HeightmapNormalizer normalizer = new HeightmapNormalizer();
        private readonly FormatFinalizer finalizer = new FormatFinalizer();

        int synchronizedIterationsLeft = 0;
        (long cpuSideMs, long cpuSideTicks, long gpuSideNs)[] synchronizedResults;
        Action postCollectionAction;
        System.Random syncRandomSeedGenerator;

        //this flag synchronizes submissions of command buffers, in order to avoid starting overwriting the heightmap while it's being read back.
        bool previousReadPending = false;


        private PipelineContext buildPipeline(bool enablePipelineProfiling, int pipelineSeed, bool ifRandomizeParameters)
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

                    if (RuntimeAssert.IsTrue(intermediateHeightmap.IsCreated(), "Failed to initialize the required resources! Try restarting the app or explicitly freeing the resources!"))
                        return null;
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
                    if (RuntimeAssert.IsTrue(finalHeightmap.IsCreated(), "Failed to initialize the required resources! Try restarting the app or explicitly freeing the resources!"))
                        return null;
                }
            }

            int preferredGroupSize = (int)math.pow(2, preferredGroupSizePower);
            PipelineContext currentContext = enablePipelineProfiling ? new ProfilingPipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed, preferredGroupSize) : new PipelineContext(intermediateHeightmap, finalHeightmap, pipelineSeed, preferredGroupSize);

            for (int s = 3; s < 7; ++s)
            {
                int groupSizeToDisable = (int)math.pow(2, s);
                var groupSizeKeywordToDisable = GlobalKeyword.Create("GLOBAL_GROUP_" + groupSizeToDisable);
                currentContext.SetGlobalKeyword(ref groupSizeKeywordToDisable, false);
            }
            var groupSizeKeyword = GlobalKeyword.Create("GLOBAL_GROUP_" + preferredGroupSize);
            currentContext.SetGlobalKeyword(ref groupSizeKeyword, true);

            GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", finalHeightmap);
            GetComponent<Renderer>().sharedMaterial.SetFloat("_HeightScale", terrainScale);

            List<PipelineStep> augmentedPipeline = new List<PipelineStep>();

            //inserts extra built-in steps if requested by a step
            HeightmapProperties runningProperties = HeightmapProperties.Created;
            foreach (PipelineStep inputStep in pipelineSteps)
            {
                if ((inputStep.GetStepExpectationsGpu() & InputExpectations.HeightMapNormalized) != 0 && (runningProperties & HeightmapProperties.Normalized) == 0)
                {
                    augmentedPipeline.Add(normalizer);
                    normalizer.UpdateHeightmapStateGpu(ref runningProperties);
                }
                augmentedPipeline.Add(inputStep);
                inputStep.UpdateHeightmapStateGpu(ref runningProperties);
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
            if (ifRandomizeParameters)
            {
                System.Random parameterRandomizer = new System.Random(pipelineSeed);
                foreach (PipelineStep step in augmentedPipeline)
                {
                    step.RandomizeParameters(parameterRandomizer);
                    step.ExecuteStepGpu(currentContext);
                }
            }
            else
            {
                foreach (PipelineStep step in augmentedPipeline)
                {
                    step.ExecuteStepGpu(currentContext);
                }
            }

            return currentContext;
        }

        [ContextMenu("Regenerate terrain")]
        async void RegenerateTerrain()
        {
            PipelineContext currentContext = buildPipeline(enableProfiling, generatorSeed, randomizeParameters);
            if (currentContext is null)
                return;
            await currentContext.ExecuteBuffer();

            if (enableMetricEvaluation)
            {
                MetricEvaluator.ComputeMetrics(finalHeightmap);
            }

#if UNITY_EDITOR
            if (dumpTextures)
            {
                RenderTextureDumper.SaveRFloatToJpg(intermediateHeightmap, "heightmap_intermediate.jpg");
                RenderTextureDumper.SaveRFloatToJpg(finalHeightmap, "heightmap_final.jpg");
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
                foreach (GUIManager guiManager in FindObjectsByType<GUIManager>(FindObjectsSortMode.None))
                {
                    guiManager.SetGUIRenderingEnabled(false);
                }
            }

            (long gpuFenceTime, long gpuSide)[] result = new (long gpuFenceTime, long gpuSide)[numSamples];
            Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples_gpu"));
            System.Random randomSeedGenerator = new(generatorSeed);

            //runs a number of samples, syncing each time to avoid obtaining corrupted heightmaps
            for (int s = 0; s < numSamples; ++s)
            {
                var newContext = (ProfilingPipelineContext)buildPipeline(true, randomSeedGenerator.Next(), randomizeParameters);
                if (newContext is null)
                    continue;
                await newContext.ExecuteBuffer();
                result[s] = (newContext.gpuFenceMilliseconds, newContext.gpuFrameTime);
                RenderTextureDumper.SaveRFloatToJpg(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples_gpu/async_gpu_terrain_" + s + ".jpg"));
            }

            //TODO: I might want to make first barrier optional and instead measure time from the moment of dispatch
            {
                string path = Path.Combine(Application.persistentDataPath, "async_gpu_performance_evaluation.txt");
                var sb = new StringBuilder();
                sb.AppendLine("GpuFenceTime (ms)\tGpuFrameTime (ns)");
                foreach (var (gpuFenceTime, gpuSide) in result)
                {
                    sb.Append(gpuFenceTime);
                    sb.Append('\t');
                    sb.AppendLine(gpuSide.ToString());
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
                foreach (GUIManager guiManager in FindObjectsByType<GUIManager>(FindObjectsSortMode.None))
                {
                    guiManager.SetGUIRenderingEnabled(true);
                }
            }

            UltimatePipelineStep.FreeAllResourcesInScene();
            Debug.Log("Collection done");
        }


        [ContextMenu("Collect synchronized statistics")]
        async void CollectSynchronizedStatistics()
        {
            if (synchronizedIterationsLeft > 0)
                return;

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
                foreach (GUIManager guiManager in FindObjectsByType<GUIManager>(FindObjectsSortMode.None))
                {
                    guiManager.SetGUIRenderingEnabled(false);
                }
            }

            synchronizedResults = new (long cpuSideMs, long cpuSideTicks, long gpuSideNs)[numSamples];
            Directory.CreateDirectory(Path.Combine(Application.persistentDataPath, "./samples_gpu"));
            syncRandomSeedGenerator = new System.Random(generatorSeed);

            postCollectionAction = () =>
            {
                QualitySettings.vSyncCount = oldVSync;
                Application.targetFrameRate = oldFrameRate;
                foreach (var cam in currentCameras)
                {
                    cam.enabled = true;
                }
                foreach (var light in FindObjectsByType<Light>(FindObjectsSortMode.None))
                { light.enabled = true; }
                foreach (GUIManager guiManager in FindObjectsByType<GUIManager>(FindObjectsSortMode.None))
                {
                    guiManager.SetGUIRenderingEnabled(true);
                }
            };

            synchronizedIterationsLeft = numSamples;
            previousReadPending = false;
        }

        [ExecuteAlways]
        async void Update()
        {
            if (synchronizedIterationsLeft > 0 && !previousReadPending)
            {
                synchronizedIterationsLeft -= 1;

                int myIteration = synchronizedIterationsLeft;

                var newContext = (ProfilingPipelineContext)buildPipeline(true, syncRandomSeedGenerator.Next(), randomizeParameters);
                if (newContext is null)
                    return;

                previousReadPending = true;
                await newContext.ExecuteBuffer();

                synchronizedResults[myIteration] = (newContext.gpuFenceMilliseconds, newContext.gpuFenceTicks, newContext.gpuFrameTime);

                RenderTextureDumper.SaveRFloatToJpg(finalHeightmap, Path.Combine(Application.persistentDataPath, "./samples_gpu/sync_gpu_terrain_" + myIteration + ".jpg"), false);
                previousReadPending = false;

                if (myIteration == 0)
                {
                    string path = Path.Combine(Application.persistentDataPath, "sync_gpu_performance_evaluation.txt");
                    var sb = new StringBuilder();
                    sb.AppendLine("FenceTime (ms)\tFenceTime (ticks)\tFrameTime (ns)");
                    foreach (var (gpuFenceTime, fenceTicks, frameTime) in synchronizedResults)
                    {
                        sb.Append(gpuFenceTime);
                        sb.Append('\t');
                        sb.Append(fenceTicks);
                        sb.Append('\t');
                        sb.AppendLine(frameTime.ToString());
                    }
                    File.WriteAllText(path, sb.ToString());

                    postCollectionAction();

                    UltimatePipelineStep.FreeAllResourcesInScene();
                    Debug.Log("Collection done!");
                }

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

        public override void RenderParametersTuningGUI()
        {

            GUILayout.Label("Terrain");

            GUILayout.Label($"Terrain Size: {terrainSize}");
            terrainSize = Mathf.RoundToInt(
                GUILayout.HorizontalSlider(terrainSize, 5f, 12f)
            );

            GUILayout.Label($"Group Size Power: {preferredGroupSizePower}");
            preferredGroupSizePower = Mathf.RoundToInt(
                GUILayout.HorizontalSlider(preferredGroupSizePower, 3f, 6f)
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

        public override string GUIStepTitle() => "GPU pipeline parameters";

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
            if (GUILayout.Button("Collect sync statistics"))
            {
                if (pipeline.Count == 0)
                    return;
                pipelineSteps = CommonDefines.buildPipelineFromEnum(pipeline);
                CollectSynchronizedStatistics();
            }
            if (GUILayout.Button("Free resources"))
            {
                UltimatePipelineStep.FreeAllResourcesInScene();
            }
        }

        public override string getPipelineName() => "GPU pipeline constructor";
    }
}
