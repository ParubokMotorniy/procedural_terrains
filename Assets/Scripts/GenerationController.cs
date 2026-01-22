using System.Text.RegularExpressions;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using System.IO;

[RequireComponent(typeof(Renderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class GenerationControllerr : MonoBehaviour
{
    [ContextMenu("Regenerate terrain")]
    void RegenerateTerrain()
    {
        if (noiseRenderTexture && noiseRenderTexture.IsCreated())
        { noiseRenderTexture.Release(); }

        int textureSize = terrainSize * 1024;
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

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");
        shaderToDispatch.SetTexture(kernelIdx, Shader.PropertyToID("Result"), noiseRenderTexture);
        shaderToDispatch.SetInt("texelsPerThread", textureSize / (groupScaleFactor * groupSize));
        shaderToDispatch.SetFloat("noiseFrequency", noiseFrequency);

        shaderToDispatch.SetFloat("sharpness", sharpness);
        shaderToDispatch.SetFloat("slopeErosion", slopeErosion);
        shaderToDispatch.SetFloat("perturbationStrength", perturbationStrength);
        shaderToDispatch.SetInt("numOctaves", numOctaves);

        shaderToDispatch.Dispatch(kernelIdx, groupScaleFactor, groupScaleFactor, groupScaleFactor);

        // EditorUtility.SetDirty(this);

        Assert.IsTrue(textureSize >= 8); //normalization groups are at least 8 threads wide 

        int largestGroupSize = (int)math.pow(2, math.ceil(math.log2(math.clamp(textureSize, 8, 32))));

        int normalizationKernel = normalizationShader.FindKernel("Normalizer" + largestGroupSize);
        normalizationShader.SetTexture(normalizationKernel, "Result", noiseRenderTexture);
        normalizationShader.SetInt("texelsPerThread", (int)math.ceil((float)textureSize / largestGroupSize));
        normalizationShader.Dispatch(normalizationKernel, 1, 1, 1);

        Debug.Log("Terrain has been regenerated!");
    }

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [SerializeField]
    private ComputeShader normalizationShader;

    [Range(1, 8)]
    public int groupScaleFactor = 1;

    [Range(0.001f, 2.0f)]
    public float noiseFrequency = 0.001f;

    [Range(-10.0f, 10.0f)]
    public float sharpness = 0.0f;

    [Range(0.01f, 10.0f)]
    public float slopeErosion = 0.01f;

    [Range(1, 20)]
    public int numOctaves = 5;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength = 0.01f;

    [Range(0.001f, 32.0f)]
    public float terrainScale = 1.0f;

    [Range(1,3)]
    public int terrainSize = 1;

    private RenderTexture noiseRenderTexture;
    private const int groupSize = 32;

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
