using System.Text.RegularExpressions;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using System.IO;
using System;


[RequireComponent(typeof(Renderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class MidpointDisplacementController : MonoBehaviour
{
    [ContextMenu("Regenerate terrain")]
    void RegenerateTerrain()
    {
        if (noiseRenderTexture && noiseRenderTexture.IsCreated())
        { noiseRenderTexture.Release(); }

        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        int textureSize = groupSize * texelsPerThreadDomain * groupScaleFactor;

        noiseRenderTexture = new RenderTexture(textureSize, textureSize, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        noiseRenderTexture.Create();

        Assert.IsTrue(noiseRenderTexture.IsCreated());

        GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", noiseRenderTexture);

        int initializationKernelIdx = shaderToDispatch.FindKernel("IntializeTexture");
        int transition12KernelIdx = shaderToDispatch.FindKernel("Transition12");
        int transition21KernelIdx = shaderToDispatch.FindKernel("Transition21");
        int extraNoiseKernel = shaderToDispatch.FindKernel("AddExtraNoise");

        shaderToDispatch.SetInt("threadDomainTexelWidth", texelsPerThreadDomain);
        shaderToDispatch.SetInt("threadSubdomainsX", groupSize * groupScaleFactor);
        shaderToDispatch.SetInt("threadSubdomainsY", groupSize * groupScaleFactor);
        shaderToDispatch.SetFloat("worleyFrequency", worleyFrequency);

        shaderToDispatch.SetTexture(initializationKernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition12KernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition21KernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(extraNoiseKernel, "Result", noiseRenderTexture);

        shaderToDispatch.SetTexture(initializationKernelIdx, "NoiseSource", whiteNoiseTexture);
        shaderToDispatch.SetTexture(transition12KernelIdx, "NoiseSource", whiteNoiseTexture);
        shaderToDispatch.SetTexture(transition21KernelIdx, "NoiseSource", whiteNoiseTexture);
        shaderToDispatch.SetTexture(extraNoiseKernel, "NoiseSource", whiteNoiseTexture);

        System.Random rng = new System.Random();
        shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));

        float octaveAmplitude = 1.0f;
        shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
        shaderToDispatch.Dispatch(initializationKernelIdx, groupScaleFactor, groupScaleFactor, 1);

        for (int sub = 0; sub < numSubdivisions; ++sub)
        {
            shaderToDispatch.SetInt("texelWidthDivisionFactor", (int)math.pow(2, sub));
            shaderToDispatch.SetInt("texelWidthDivided", texelsPerThreadDomain / (int)math.pow(2, sub + 1));

            octaveAmplitude *= H;
            shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
            shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));
            shaderToDispatch.Dispatch(transition12KernelIdx, groupScaleFactor, groupScaleFactor, 1);

            if (addExtraNoise)
            {
                shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));
                shaderToDispatch.Dispatch(extraNoiseKernel, groupScaleFactor, groupScaleFactor, 1);
            }

            octaveAmplitude *= H;
            shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);
            shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));
            shaderToDispatch.Dispatch(transition21KernelIdx, groupScaleFactor, groupScaleFactor, 1);

            if (addExtraNoise)
            {
                shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));
                shaderToDispatch.Dispatch(extraNoiseKernel, groupScaleFactor, groupScaleFactor, 1);
            }
        }

        Assert.IsTrue(textureSize >= 8); //normalization groups are at least 8 threads wide 

        int largestGroupSize = (int)math.pow(2, math.ceil(math.log2(math.clamp(textureSize, 8, 32))));

        int normalizationKernel = normalizationShader.FindKernel("Normalizer" + largestGroupSize);
        normalizationShader.SetTexture(normalizationKernel, "Result", noiseRenderTexture);
        normalizationShader.SetInt("texelsPerThread", (int)math.ceil((float)textureSize / largestGroupSize));
        normalizationShader.Dispatch(normalizationKernel, 1, 1, 1);

        // EditorUtility.SetDirty(this);
        Debug.Log("Terrain has been regenerated!");
    }

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [SerializeField]
    private ComputeShader normalizationShader;

    [Range(1, 8)]
    public int groupScaleFactor = 1;

    [Range(0.001f, 32.0f)]
    public float terrainScale = 1.0f;

    [Range(1, 16)]
    public uint numSubdivisions = 2;

    [Range(0.01f, 1.0f)]
    public float H = 0.85f;

    [SerializeField]
    public Texture2D whiteNoiseTexture;

    [SerializeField]
    public bool addExtraNoise;

    [Range(0.01f, 10.0f)]
    public float worleyFrequency;

    private RenderTexture noiseRenderTexture;
    private const int groupSize = 4;

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
