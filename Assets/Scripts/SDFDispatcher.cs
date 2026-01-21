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
public class SDFDispatcher : MonoBehaviour
{
    [ContextMenu("Regenerate terrain")]
    void RegenerateTerrain()
    {
        int textureSize = inputTestMaskTexture.width;

        RenderTexture noiseRenderTexture = new RenderTexture(textureSize, textureSize, 0)
        {
            graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
            useMipMap = false,
            enableRandomWrite = true,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };

        noiseRenderTexture.Create();

        Assert.IsTrue(noiseRenderTexture.IsCreated());

        ComputeBuffer buffer1 = new ComputeBuffer(textureSize * textureSize, sizeof(float));
        Assert.IsTrue(buffer1.IsValid());

        ComputeBuffer buffer2 = new ComputeBuffer(textureSize * textureSize, sizeof(float));
        Assert.IsTrue(buffer2.IsValid());

        GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", noiseRenderTexture);

        int maskToSeedBufferKernelIdx = shaderToDispatch.FindKernel("MaskToSeedBuffer");
        int floodingStepKernelIdx = shaderToDispatch.FindKernel("FloodingStep");
        int seedBufferToHieghtmapKernelIdx = shaderToDispatch.FindKernel("SeedBufferToHieghtmap");

        foreach (int kernelIdx in new int[] { maskToSeedBufferKernelIdx, floodingStepKernelIdx, seedBufferToHieghtmapKernelIdx })
        {
            shaderToDispatch.SetBuffer(kernelIdx, "buffer1", buffer1);
            shaderToDispatch.SetBuffer(kernelIdx, "buffer2", buffer2);
            shaderToDispatch.SetTexture(kernelIdx, "inputTexture", inputTestMaskTexture);
            shaderToDispatch.SetTexture(kernelIdx, "outputTexture", noiseRenderTexture);
        }

        shaderToDispatch.SetInt("texelsPerThread", textureSize / (groupSize * groupScaleFactor));
        shaderToDispatch.SetInt("bufferSideLength", textureSize);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        shaderToDispatch.Dispatch(maskToSeedBufferKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        while (currentFloodStep > 1)
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            shaderToDispatch.SetInt("currentSourceBuffer",  currentReadBuffer);
            currentFloodStep /= 2;
            shaderToDispatch.SetInt("floodStepSize", currentFloodStep);
            shaderToDispatch.Dispatch(floodingStepKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }
        currentReadBuffer = (currentReadBuffer + 1) % 2;
        shaderToDispatch.SetInt("currentSourceBuffer", 0);
        shaderToDispatch.Dispatch(seedBufferToHieghtmapKernelIdx, groupScaleFactor, groupScaleFactor, 1);

        RenderTextureDumper.SaveRFloatToExr(noiseRenderTexture, "./sdf.exr");

        // Assert.IsTrue(textureSize >= 8); //normalization groups are at least 8 threads wide 

        // int largestGroupSize = (int)math.pow(2, math.ceil(math.log2(math.clamp(textureSize, 8, 32))));

        // int normalizationKernel = normalizationShader.FindKernel("Normalizer" + largestGroupSize);
        // normalizationShader.SetTexture(normalizationKernel, "Result", noiseRenderTexture);
        // normalizationShader.SetInt("texelsPerThread", (int)math.ceil((float)textureSize / largestGroupSize));
        // normalizationShader.Dispatch(normalizationKernel, 1, 1, 1);

        // Debug.Log("Terrain has been regenerated!");
    }

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [SerializeField]
    private ComputeShader normalizationShader;

    [SerializeField]
    private Texture2D inputTestMaskTexture;

    [Range(1, 8)]
    public int groupScaleFactor = 1;

    [Range(0.001f, 32.0f)]
    public float terrainScale = 1.0f;

    private const int groupSize = 16;

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
