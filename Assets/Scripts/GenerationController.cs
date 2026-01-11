using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

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

        noiseRenderTexture = new RenderTexture(dispatchDimension, dispatchDimension, 0)
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

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");
        shaderToDispatch.SetTexture(kernelIdx, Shader.PropertyToID("Result"), noiseRenderTexture);
        shaderToDispatch.SetInt("dispatchDimension", dispatchDimension);
        shaderToDispatch.SetFloat("noiseFrequency", noiseFrequency);

        shaderToDispatch.SetFloat("sharpness", sharpness);
        shaderToDispatch.SetFloat("slopeErosion", slopeErosion);
        shaderToDispatch.SetFloat("perturbationStrength", perturbationStrength);
        shaderToDispatch.SetInt("numOctaves", numOctaves);

        shaderToDispatch.Dispatch(kernelIdx, dispatchDimension, dispatchDimension, 1);

        // EditorUtility.SetDirty(this);

        Debug.Log("Terrain has been regenerated!");
    }

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(8, 2048)]
    public int dispatchDimension;

    [Range(0.01f, 10.0f)]
    public float noiseFrequency;

    [Range(-10.0f, 10.0f)]
    public float sharpness;

    [Range(0.01f, 10.0f)]
    public float slopeErosion;

    [Range(1, 20)]
    public int numOctaves;

    [Range(0.01f, 10.0f)]
    public float perturbationStrength;

    [Range(1.0f, 32.0f)]
    public float terrainScale;

    private RenderTexture noiseRenderTexture;

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
