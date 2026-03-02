using UnityEngine;
using UnityEngine.Rendering;
using System.IO;
using System.Collections.Generic;
using System;

public static class MetricEvaluator
{
    private static List<Action<Texture2D>> metricsList = new List<Action<Texture2D>>() { (tex) =>
            {
                int width = tex.width;
    int height = tex.height;

    Color[] pixels = tex.GetPixels();

    float GetHeight(int x, int y)
    {
        return pixels[x + y * width].r;
    }

    int n = 0;
    double mean = 0.0;
    double m2 = 0.0;

                for (int y = 0; y<height; ++y)
                {
                    for (int x = 0; x<width; ++x)
                    {
                        float center = GetHeight(x, y);

    float maxDelta = 0f;

                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                if (dx == 0 && dy == 0) continue;

                                int nx = x + dx;
    int ny = y + dy;

                                if ((uint) nx >= (uint) width || (uint) ny >= (uint) height)
                                    continue;

    float neighbor = GetHeight(nx, ny);
    float delta = Mathf.Abs(neighbor - center);
                                if (delta > maxDelta) maxDelta = delta;
                            }
                        }

                        n++;
double d = maxDelta - mean;
mean += d / n;
double d2 = maxDelta - mean;
m2 += d * d2;
                    }
                }

                double variance = (n > 1) ? (m2 / n) : 0.0; // population variance; use (n-1) for sample variance
double stdDev = Math.Sqrt(variance);

double erosionScore = stdDev / mean;

Debug.Log("Erosion score: " + erosionScore);
            }};
    // metricsList.Add();

    public static void ComputeMetrics(RenderTexture rt)
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

        foreach (var metricComputer in metricsList)
        {
            metricComputer(tex);
        }
    });
    }
}