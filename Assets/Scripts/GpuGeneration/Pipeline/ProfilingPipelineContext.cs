using UnityEngine;
using UnityEngine.Rendering;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using Unity.Profiling;

namespace GpuGenerationPipeline
{
    public class ProfilingPipelineContext : PipelineContext
    {
        private GraphicsFence pipelineStartFence;
        private GraphicsFence pipelineEndFence;
        private Stopwatch cpuProfilingStopwatch;
        private Stopwatch gpuProfilingStopwatch;

        private static ProfilerRecorder gpuTimeProfiler = ProfilerRecorder.StartNew(ProfilerCategory.Render, "GPU Frame Time");

        public long cpuMilliseconds { get; private set; }
        public long cpuTicks { get; private set; }

        public long gpuFenceMilliseconds { get; private set; }
        public long gpuFenceTicks { get; private set; }

        public long gpuFrameTime { get; private set; }

        public ProfilingPipelineContext(RenderTexture passHeightmap, RenderTexture finalHeightmap, int seed, int preferredGroupSize) : base(passHeightmap, finalHeightmap, seed, preferredGroupSize)
        {

            pipelineStartFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            cpuProfilingStopwatch = new Stopwatch();
            gpuProfilingStopwatch = new Stopwatch();

            Debug.LogWarning("System supports fences ? : " + SystemInfo.supportsGraphicsFence);
            Debug.LogWarning("Stopwatches are high-res ? : " + Stopwatch.IsHighResolution);

            cpuProfilingStopwatch.Start();
        }

        public override async Task ExecuteBuffer()
        {
            pipelineEndFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            TimeSpan millisecondsAtStart = new();
            TimeSpan millisecondsAtEnd = new();

            gpuProfilingStopwatch.Start();
            Task cmdStartTask = PipelineProfiler.WaitForFenceAsync(pipelineStartFence, () => { millisecondsAtStart = gpuProfilingStopwatch.Elapsed; });
            Task cmdEndTask = PipelineProfiler.WaitForFenceAsync(pipelineEndFence, () => { millisecondsAtEnd = gpuProfilingStopwatch.Elapsed; gpuFrameTime = gpuTimeProfiler.LastValue; });

            Graphics.ExecuteCommandBuffer(terrainPipelineCMD);
            cpuProfilingStopwatch.Stop();

            await Task.WhenAll(cmdStartTask, cmdEndTask);

            cpuMilliseconds = cpuProfilingStopwatch.ElapsedMilliseconds;
            cpuTicks = cpuProfilingStopwatch.ElapsedTicks;

            gpuFenceMilliseconds = (long)(millisecondsAtEnd.TotalMilliseconds - millisecondsAtStart.TotalMilliseconds);
            gpuFenceTicks = millisecondsAtEnd.Ticks - millisecondsAtStart.Ticks;

            Debug.Log("CPU housekeeping time (ms): " + cpuMilliseconds + ". Ticks: " + cpuTicks);
            Debug.Log("GPU processing time (ms): " + gpuFenceMilliseconds + ". Ticks: " + gpuFenceTicks);
            Debug.Log("Total frame time (ns) : " + gpuFrameTime);
        }
    }
}
