using System;
using System.Collections.Generic;
using UnityEngine;

public class GUIManager : MonoBehaviour
{
    [SerializeField]
    List<TunableObject> stepsToRender;
    List<bool> foldouts;

    [SerializeField]
    List<TunableGenerator> generatorsToRender;

    private static Rect stepsRect = new Rect(10, 10, 300, 600);
    private static Rect pipelineRect = new Rect(900, 10, 300, 600);

    void Start()
    {
        foldouts = new List<bool>(stepsToRender.Count);
        for (int i = 0; i < stepsToRender.Count; i++)
            foldouts.Add(false);
    }

    void DrawSection(int index, TunableObject section)
    {
        if (GUILayout.Button(
            (foldouts[index] ? "▼ " : "▶ ") + section.GUIStepTitle(),
            GUI.skin.label))
        {
            foldouts[index] = !foldouts[index];
        }

        if (foldouts[index])
        {
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.BeginHorizontal();
            GUILayout.Space(10);
            GUILayout.BeginVertical();

            section.RenderParametersTuningGUI();

            GUILayout.EndVertical();
            GUILayout.EndHorizontal();

            GUILayout.EndVertical();
        }
    }

    void OnGUI()
    {
        {
            GUILayout.BeginArea(stepsRect, GUI.skin.box);
            GUILayout.Label("Tuners");
            for (int i = 0; i < stepsToRender.Count; i++)
            {
                DrawSection(i, stepsToRender[i]);
            }

            GUILayout.EndArea();
        }

        {
            GUILayout.BeginArea(pipelineRect, GUI.skin.box);
            GUILayout.Label("Generators");

            for (int i = 0; i < generatorsToRender.Count; i++)
            {
                DrawPipelineList(generatorsToRender[i]);
            }

            GUILayout.EndArea();
        }
    }

    public static void DrawPipelineList(
        TunableGenerator generator
    )
    {
        if (GUILayout.Button(
                (generator.pipelineFoldout ? "▼ " : "▶ ") + generator.getPipelineName(),
                GUI.skin.label))
        {
            generator.pipelineFoldout = !generator.pipelineFoldout;
        }

        if (!generator.pipelineFoldout)
        {
            return;
        }

        GUILayout.BeginHorizontal();

        if (GUILayout.Button("+", GUILayout.Width(30)))
        {
            generator.pipeline.Add(
                generator.pipeline.Count > 0 ? generator.pipeline[^1] : default
            );
        }

        GUI.enabled = generator.pipeline.Count > 0;

        if (GUILayout.Button("-", GUILayout.Width(30)))
        {
            generator.pipeline.RemoveAt(generator.pipeline.Count - 1);
        }

        if (GUILayout.Button("Clear"))
        {
            generator.pipeline.Clear();
        }

        GUI.enabled = true;

        GUILayout.EndHorizontal();

        GUILayout.Space(5);

        generator.scroll = GUILayout.BeginScrollView(generator.scroll);

        string[] names = Enum.GetNames(typeof(CommonDefines.AvailablePipelineSteps));

        for (int i = 0; i < generator.pipeline.Count; i++)
        {
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.Label($"Step {i}");

            int current = Convert.ToInt32(generator.pipeline[i]);
            int selected = GUILayout.Toolbar(current, names);

            generator.pipeline[i] = (CommonDefines.AvailablePipelineSteps)Enum.ToObject(typeof(CommonDefines.AvailablePipelineSteps), selected);

            GUILayout.EndVertical();
        }

        GUILayout.EndScrollView();

        GUILayout.Space(5);
        generator.drawExtraCustomUI();

        return;
    }
}