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

        //TODO: add texture scaling (texels per division)
        int texelsPerThreadDomain = (int)math.pow(2, numSubdivisions);
        int textureSize = (groupSize * texelsPerThreadDomain * groupScaleFactor) + 1; //1 closes off the last row

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
        GetComponent<Renderer>().sharedMaterial.SetFloat("_HeightScale", terrainScale);

        int initializationKernelIdx = shaderToDispatch.FindKernel("IntializeTexture");
        int transition12KernelIdx = shaderToDispatch.FindKernel("Transition12");
        int transition21KernelIdx = shaderToDispatch.FindKernel("Transition21");

        shaderToDispatch.SetInt("threadCellTexelWidth", texelsPerThreadDomain);
        shaderToDispatch.SetFloat("noiseFrequency", noiseFrequency);
        shaderToDispatch.SetInt("threadSubdomainsX", groupSize * groupScaleFactor);
        shaderToDispatch.SetInt("threadSubdomainsY", groupSize * groupScaleFactor);

        shaderToDispatch.SetTexture(initializationKernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition12KernelIdx, "Result", noiseRenderTexture);
        shaderToDispatch.SetTexture(transition21KernelIdx, "Result", noiseRenderTexture);

        System.Random rng = new System.Random();
        shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));

        shaderToDispatch.Dispatch(initializationKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        float octaveAmplitude = 1.0f;
        for (int sub = 1; sub <= numSubdivisions; ++sub, octaveAmplitude *= 0.5f)
        {
            shaderToDispatch.SetInt("texelWidthDivisionFactor", sub);
            shaderToDispatch.SetInt("texelWidthDivided", texelsPerThreadDomain / (int)math.pow(2, sub));
            shaderToDispatch.SetFloat("octaveAmplitude", octaveAmplitude);

            shaderToDispatch.SetVector("noiseDisplacement", new Vector4((float)rng.NextDouble(), (float)rng.NextDouble(), 0.0f, 0.0f));

            shaderToDispatch.Dispatch(transition12KernelIdx, groupScaleFactor, groupScaleFactor, 1);
            shaderToDispatch.Dispatch(transition21KernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }

        // EditorUtility.SetDirty(this);

        Debug.Log("Terrain has been regenerated!");
    }

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(1, 8)]
    public int groupScaleFactor = 1;

    [Range(0.01f, 2.0f)]
    public float noiseFrequency = 0.01f;

    [Range(0.001f, 32.0f)]
    public float terrainScale = 1.0f;

    [Range(1, 16)]
    public uint numSubdivisions = 2;

    private RenderTexture noiseRenderTexture;
    private const int groupSize = 16;

    void Start()
    {
        RegenerateTerrain();
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
