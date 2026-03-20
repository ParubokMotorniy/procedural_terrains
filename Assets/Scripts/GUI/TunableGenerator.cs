using System.Collections.Generic;
using UnityEngine;

public abstract class TunableGenerator : TunableObject
{
    public List<CommonDefines.AvailablePipelineSteps> pipeline = new();
    public Vector2 scroll = Vector2.zero;
    public bool pipelineFoldout = false;
    public abstract void drawExtraCustomUI();
    public abstract string getPipelineName();
}
