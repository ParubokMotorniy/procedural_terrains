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
            pipelineStartFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.AsyncQueueSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);
            cpuProfilingStopwatch = new Stopwatch();

            cpuProfilingStopwatch.Start();
        }

        public override async Task ExecuteBuffer()
        {
            pipelineEndFence = terrainPipelineCMD.CreateGraphicsFence(UnityEngine.Rendering.GraphicsFenceType.AsyncQueueSynchronisation, UnityEngine.Rendering.SynchronisationStageFlags.AllGPUOperations);

            // Graphics.ExecuteCommandBufferAsync(terrainPipelineCMD, ComputeQueueType.Urgent);

            long millisecondsAtStart = 0;
            long millisecondsAtEnd = 0;
            gpuProfilingStopwatch.Start();
            cpuProfilingStopwatch.Stop();

            Task cmdStartTask = PipelineProfiler.WaitForFenceAsync(pipelineStartFence, () => { millisecondsAtStart = gpuProfilingStopwatch.ElapsedMilliseconds; });
            Task cmdEndTask = PipelineProfiler.WaitForFenceAsync(pipelineEndFence, () => { millisecondsAtEnd = gpuProfilingStopwatch.ElapsedMilliseconds; });
            Graphics.ExecuteCommandBuffer(terrainPipelineCMD);

            await Task.WhenAll(cmdStartTask, cmdEndTask);

            long cpuMilliseconds = cpuProfilingStopwatch.ElapsedMilliseconds;
            long gpuMilliseconds = millisecondsAtEnd - millisecondsAtStart;

            Debug.Log("CPU housekeeping time (ms): " + cpuMilliseconds + "\nGPU processing time (ms): " + gpuMilliseconds);
        }
    }
}
