using System;
using System.Threading.Tasks;
using UnityEngine;

public class PipelineProfiler
{
    public static async Task WaitForFenceAsync(UnityEngine.Rendering.GraphicsFence fence, Action fencePassReaction)
    {
        while (!fence.passed)
        {
            await Task.Yield();
        }
        
        fencePassReaction();
    }
}
