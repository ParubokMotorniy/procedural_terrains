using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

[RequireComponent(typeof(Renderer))]
public class GenerationControllerr : MonoBehaviour
{

    [SerializeField]
    private ComputeShader shaderToDispatch;

    [Range(1, 64)]
    public int dispatchDimension;

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

        // GetComponent<Renderer>().material.mainTexture = noiseRenderTexture;
        GetComponent<Renderer>().material.SetTexture("_HeightMap", noiseRenderTexture);
        GetComponent<Renderer>().material.SetFloat("_HeightScale", 5.0f);
    }

    void Update()
    {

    }
}
