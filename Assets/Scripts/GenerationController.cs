using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

[RequireComponent(typeof(Renderer))]
[RequireComponent(typeof(MeshFilter))]
public class GenerationControllerr : MonoBehaviour
{
    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(1, 64)]
    public int dispatchDimension;

    [Range(1.0f, 32.0f)]
    public float terrainScale;

    private RenderTexture noiseRenderTexture;

    void Start()
    {
        noiseRenderTexture = new RenderTexture(dispatchDimension, dispatchDimension, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R8G8B8A8_UNorm,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        noiseRenderTexture.Create();

        Assert.IsTrue(noiseRenderTexture.IsCreated());

        int kernelIdx = shaderToDispatch.FindKernel("UberNoiseTerrainGenerator");
        shaderToDispatch.SetTexture(kernelIdx, Shader.PropertyToID("Result"), noiseRenderTexture);
        shaderToDispatch.SetInt("dispatchDimension", dispatchDimension);
        shaderToDispatch.Dispatch(kernelIdx, dispatchDimension, dispatchDimension, 1);

        GetComponent<Renderer>().material.SetTexture("_HeightMap", noiseRenderTexture);
        GetComponent<Renderer>().material.SetFloat("_HeightScale", terrainScale);
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
