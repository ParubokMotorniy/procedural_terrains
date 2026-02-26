using UnityEngine;
using UnityEngine.Rendering;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;

namespace GenerationPipeline
{
    public class ProfilingPipelineContext : PipelineContext
    {
        private GraphicsFence pipelineStartFence;
        private GraphicsFence pipelineEndFence;
        private Stopwatch cpuProfilingStopwatch;
        private Stopwatch gpuProfilingStopwatch;

        private List<Action<Texture2D>> metricsList = new List<Action<Texture2D>>();

        public ProfilingPipelineContext(RenderTexture passHeightmap, int seed, int preferredGroupSize) : base(passHeightmap, seed, preferredGroupSize)
        {
            Debug.Log("System supports fences ? :" + SystemInfo.supportsGraphicsFence);

            pipelineStartFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            cpuProfilingStopwatch = new Stopwatch();
            gpuProfilingStopwatch = new Stopwatch();

            metricsList.Add(tex =>
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

                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
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

                                if ((uint)nx >= (uint)width || (uint)ny >= (uint)height)
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
            });

            cpuProfilingStopwatch.Start();
        }

        public override async Task ExecuteBuffer()
        {

            pipelineEndFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            TimeSpan millisecondsAtStart = new();
            TimeSpan millisecondsAtEnd = new();

            gpuProfilingStopwatch.Start();
            Task cmdStartTask = PipelineProfiler.WaitForFenceAsync(pipelineStartFence, () => { millisecondsAtStart = gpuProfilingStopwatch.Elapsed; });
            Task cmdEndTask = PipelineProfiler.WaitForFenceAsync(pipelineEndFence, () => { millisecondsAtEnd = gpuProfilingStopwatch.Elapsed; });

            Graphics.ExecuteCommandBuffer(terrainPipelineCMD);
            cpuProfilingStopwatch.Stop();

            AsyncGPUReadback.Request(finalHeightmap, 0, TextureFormat.RFloat, req =>
            {
                if (req.hasError)
                {
                    Debug.LogError("AsyncGPUReadback error.");
                    return;
                }

                var tex = new Texture2D(finalHeightmap.width, finalHeightmap.height, TextureFormat.RFloat, false, true);
                tex.SetPixelData(req.GetData<float>(), 0);
                tex.Apply(false, false);

                foreach (var metricComputer in metricsList)
                {
                    metricComputer(tex);
                }
            });
            await Task.WhenAll(cmdStartTask, cmdEndTask);

            long cpuMilliseconds = cpuProfilingStopwatch.ElapsedMilliseconds;
            long gpuMilliseconds = (long)(millisecondsAtEnd.TotalMilliseconds - millisecondsAtStart.TotalMilliseconds);
            long gpuTicks = millisecondsAtEnd.Ticks - millisecondsAtStart.Ticks;

            //TODO: prettify these for belivable benchmarking
            //TODO: compute some roughness metric
            Debug.Log("CPU housekeeping time (ms): " + cpuMilliseconds + ". Ticks: " + cpuProfilingStopwatch.ElapsedTicks);
            Debug.Log("GPU processing time (ms): " + gpuMilliseconds + ". Ticks: " + gpuTicks);
        }
    }
}
