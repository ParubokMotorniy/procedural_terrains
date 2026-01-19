using UnityEngine;
using UnityEngine.Rendering;
using System.IO;

public static class RenderTextureDumper
{
    public static void SaveRFloatToExr(RenderTexture rt, string filePath)
    {
        if (rt == null) { Debug.LogError("RT is null"); return; }
        if (!rt.IsCreated()) { Debug.LogError("RT not created"); return; }

        AsyncGPUReadback.Request(rt, 0, TextureFormat.RFloat, req =>
        {
            if (req.hasError)
            {
                Debug.LogError("AsyncGPUReadback error.");
                return;
            }

            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RFloat, false, true);
            tex.SetPixelData(req.GetData<float>(), 0);
            tex.Apply(false, false);

            // Encode to EXR in float mode (preserves real values)
            byte[] bytes = tex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat);
            File.WriteAllBytes(filePath, bytes);

            UnityEngine.Object.DestroyImmediate(tex);
            Debug.Log($"Saved EXR: {filePath}");
        });
    }
}
