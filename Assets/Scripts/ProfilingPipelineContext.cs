using UnityEngine;
using UnityEngine.Rendering;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;

namespace GenerationPipeline
{
    public class ProfilingPipelineContext : PipelineContext
    {
        private GraphicsFence pipelineStartFence;
        private GraphicsFence pipelineEndFence;
        private Stopwatch cpuProfilingStopwatch;
        private Stopwatch gpuProfilingStopwatch;

        public ProfilingPipelineContext(RenderTexture passHeightmap) : base(passHeightmap)
        {
            Debug.Log("System supports fences ? :" +  SystemInfo.supportsGraphicsFence);

            pipelineStartFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            cpuProfilingStopwatch = new Stopwatch();
            gpuProfilingStopwatch = new Stopwatch();

            cpuProfilingStopwatch.Start();
        }

        public override async Task ExecuteBuffer()
        {
            pipelineEndFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.CPUSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            long millisecondsAtStart = 0;
            long millisecondsAtEnd = 0;

            gpuProfilingStopwatch.Start();
            Task cmdStartTask = PipelineProfiler.WaitForFenceAsync(pipelineStartFence, () => { millisecondsAtStart = gpuProfilingStopwatch.ElapsedMilliseconds; });
            Task cmdEndTask = PipelineProfiler.WaitForFenceAsync(pipelineEndFence, () => { millisecondsAtEnd = gpuProfilingStopwatch.ElapsedMilliseconds; });

            Graphics.ExecuteCommandBuffer(terrainPipelineCMD);
            cpuProfilingStopwatch.Stop();

            await Task.WhenAll(cmdStartTask, cmdEndTask);

            long cpuMilliseconds = cpuProfilingStopwatch.ElapsedMilliseconds;
            long gpuMilliseconds = millisecondsAtEnd - millisecondsAtStart;

            //TODO: prettify these for belivable benchmarking
            Debug.Log("CPU housekeeping time (ms): " + cpuMilliseconds);
            Debug.Log("GPU processing time (ms): " + gpuMilliseconds);
        }
    }
}
