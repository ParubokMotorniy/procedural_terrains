using System.Text.RegularExpressions;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using System.IO;
using System;
using Unity.VisualScripting;

[RequireComponent(typeof(Renderer))]
[RequireComponent(typeof(MeshFilter))]
[ExecuteAlways]
public class SDFDispatcher : MonoBehaviour
{
    [ContextMenu("Regenerate terrain")]
    void RegenerateTerrain()
    {
        int textureSize = heightmapSize * 1024;

        {
            if (noiseRenderTexture && noiseRenderTexture.IsCreated())
            { noiseRenderTexture.Release(); }

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
        }

        GetComponent<Renderer>().sharedMaterial.SetTexture("_HeightMap", noiseRenderTexture);

        {
            if (coastlineTexture && coastlineTexture.IsCreated())
            { coastlineTexture.Release(); }

            coastlineTexture = new RenderTexture(textureSize, textureSize, 0)
            {
                graphicsFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
                useMipMap = false,
                enableRandomWrite = true,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };

            coastlineTexture.Create();

            Assert.IsTrue(coastlineTexture.IsCreated());
        }

        ComputeBuffer buffer1 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer1.IsValid());

        ComputeBuffer buffer2 = new ComputeBuffer(textureSize * textureSize * 2, sizeof(float));
        Assert.IsTrue(buffer2.IsValid());

        int maskToSeedBufferKernelIdx = shaderToDispatch.FindKernel("MaskToSeedBuffer");
        int floodingStepKernelIdx = shaderToDispatch.FindKernel("FloodingStep");
        int seedBufferToHieghtmapKernelIdx = shaderToDispatch.FindKernel("SeedBufferToHieghtmap");
        int coastlineGeneratorKernel = shaderToDispatch.FindKernel("CoastlineGenerator");
        int sDFPostprocessorKernel = shaderToDispatch.FindKernel("SDFPostprocessor");

        foreach (int kernelIdx in new int[] { maskToSeedBufferKernelIdx, floodingStepKernelIdx, seedBufferToHieghtmapKernelIdx, coastlineGeneratorKernel, sDFPostprocessorKernel })
        {
            shaderToDispatch.SetBuffer(kernelIdx, "buffer1", buffer1);
            shaderToDispatch.SetBuffer(kernelIdx, "buffer2", buffer2);
            shaderToDispatch.SetTexture(kernelIdx, "inputTexture", coastlineTexture);
            shaderToDispatch.SetTexture(kernelIdx, "outputTexture", noiseRenderTexture);
        }

        shaderToDispatch.SetInt("texelsPerThread", textureSize / (groupSize * groupScaleFactor));
        shaderToDispatch.SetInt("bufferSideLength", textureSize);
        shaderToDispatch.SetInt("maskBorderWidth", (int)(textureSize * 0.2f)); //fix at 20%
        float maxDistanceToSeed = math.sqrt(2 * textureSize * textureSize);
        shaderToDispatch.SetFloat("shoreBaseHeight", maxDistanceToSeed * 0.005f); //fix at 0.5%
        shaderToDispatch.SetFloat("simplexFrequency", baseSimplexFrequency);

        int currentReadBuffer = 1;
        int currentFloodStep = textureSize;

        Action updateSourceBuffer = () =>
        {
            currentReadBuffer = (currentReadBuffer + 1) % 2;
            shaderToDispatch.SetInt("currentSourceBuffer", currentReadBuffer);
        };

        shaderToDispatch.Dispatch(coastlineGeneratorKernel, groupScaleFactor, groupScaleFactor, 1);
        shaderToDispatch.Dispatch(maskToSeedBufferKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        while (currentFloodStep > 1)
        {
            updateSourceBuffer();
            currentFloodStep /= 2;
            shaderToDispatch.SetInt("floodStepSize", currentFloodStep);
            shaderToDispatch.Dispatch(floodingStepKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }
        //extra iteration to improve SDF accuracy
        {
            updateSourceBuffer();
            shaderToDispatch.SetInt("floodStepSize", 1);
            shaderToDispatch.Dispatch(floodingStepKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }
        //distance computation
        {
            updateSourceBuffer();
            shaderToDispatch.Dispatch(seedBufferToHieghtmapKernelIdx, groupScaleFactor, groupScaleFactor, 1);
        }

        Assert.IsTrue(textureSize >= 8); //normalization groups are at least 8 threads wide 
        int largestGroupSize = (int)math.pow(2, math.ceil(math.log2(math.clamp(textureSize, 8, 32))));
        int normalizationKernel = normalizationShader.FindKernel("Normalizer" + largestGroupSize);
        normalizationShader.SetTexture(normalizationKernel, "Result", noiseRenderTexture);
        normalizationShader.SetInt("texelsPerThread", (int)math.ceil((float)textureSize / largestGroupSize));
        normalizationShader.SetFloat("desiredMaxHeight", 1.0f);

        //normalization of the output SDF texture
        {
            normalizationShader.Dispatch(normalizationKernel, 1, 1, 1);
        }

        // heightmap postprocessing
        {
            shaderToDispatch.Dispatch(sDFPostprocessorKernel, groupScaleFactor, groupScaleFactor, 1);
        }

        //normalization of the final heightmap
        {
            normalizationShader.Dispatch(normalizationKernel, 1, 1, 1);
        }

        // RenderTextureDumper.SaveRFloatToExr(coastlineTexture, "./coastline.exr");
        // RenderTextureDumper.SaveRFloatToExr(noiseRenderTexture, "./sdf.exr");

        buffer1.Release();
        buffer2.Release();

        Debug.Log("Terrain has been regenerated!");
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

    [Range(0.025f, 3.0f)]
    public float baseSimplexFrequency = 1.0f;

    [Range(1, 3)]
    public int heightmapSize = 1;

    private const int groupSize = 16;

    private RenderTexture noiseRenderTexture;
    private RenderTexture coastlineTexture;

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
