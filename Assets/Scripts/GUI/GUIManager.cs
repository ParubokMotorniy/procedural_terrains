using System.Collections.Generic;
using UnityEngine;

public class GUIManager : MonoBehaviour
{
    [SerializeField]
    List<UltimatePipelineStep> stepsToRender;
    List<bool> foldouts;

    void Start()
    {
        foldouts = new List<bool>(stepsToRender.Count);
        for (int i = 0; i < stepsToRender.Count; i++)
            foldouts.Add(false);
    }

    void DrawSection(int index, UltimatePipelineStep section)
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
        GUILayout.BeginArea(new Rect(10, 10, 300, 500), GUI.skin.box);

        for (int i = 0; i < stepsToRender.Count; i++)
        {
            DrawSection(i, stepsToRender[i]);
        }

        GUILayout.EndArea();
    }
}