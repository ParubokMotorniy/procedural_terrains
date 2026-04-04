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

    private const float PANEL_WIDTH = 350f;
    private const float MARGIN = 10f;
    private Vector2 tunerScroll;
    private Vector2 pipelineScroll;

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
        float height = Screen.height - 2 * MARGIN;

        Rect stepsRect = new Rect(
            MARGIN,
            MARGIN,
            PANEL_WIDTH,
            height
        );

        Rect pipelineRect = new Rect(
            Screen.width - PANEL_WIDTH - MARGIN,
            MARGIN,
            PANEL_WIDTH,
            height
        );

        {
            GUILayout.BeginArea(stepsRect, GUI.skin.box);
            GUILayout.Label("Tuners");
            tunerScroll = GUILayout.BeginScrollView(tunerScroll);
            for (int i = 0; i < stepsToRender.Count; i++)
            {
                DrawSection(i, stepsToRender[i]);
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        {
            GUILayout.BeginArea(pipelineRect, GUI.skin.box);
            GUILayout.Label("Pipelines");
            pipelineScroll = GUILayout.BeginScrollView(pipelineScroll);

            for (int i = 0; i < generatorsToRender.Count; i++)
            {
                DrawPipelineList(generatorsToRender[i]);
            }

            GUILayout.EndScrollView();
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